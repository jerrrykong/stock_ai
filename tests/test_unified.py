"""unified 的防泄漏、episode 隔离、真实 SB3 短训练及接口回归。"""
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from gymnasium.utils.env_checker import check_env

from stock_ai.cli.main import cli
from stock_ai.config.loader import ConfigLoader
from stock_ai.config.schema import AppConfig
from stock_ai.scale.errors import ScaleError
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.replay import ReplayError, ReplayService
from stock_ai.scale.ui.server import ScaleUI
from stock_ai.scale.unified import (CONTRACT, DEFAULT_RANGES, FEATURES, EpisodePoolEnv,
                                    UnifiedRunner, UnifiedTradingEnv, build_dataset,
                                    relative_features, read_status, write_control)


def frame(scale=1):
    dates = pd.DatetimeIndex([])
    for year in [2021, 2022, 2023, 2024, 2025]:
        dates = dates.append(pd.bdate_range(f'{year}-01-04', periods=12))
    close = (10 + np.arange(len(dates)) * .03 + np.sin(np.arange(len(dates))) * .01) * scale
    return pd.DataFrame({'date': dates, 'open': close * .999, 'high': close * 1.01,
                         'low': close * .99, 'close': close, 'volume': (1000 + np.arange(len(dates))) * scale})


@pytest.fixture
def config(tmp_path):
    cfg = AppConfig()
    cfg.paths.root = tmp_path
    ConfigLoader.resolve_paths(cfg)
    cfg.data.cache_dir.mkdir(parents=True)
    frame().to_parquet(cfg.data.cache_dir / '000001.parquet')
    frame(100).to_parquet(cfg.data.cache_dir / '600519.parquet')
    cfg.sample.window = 3
    cfg.train.n_steps = 8
    cfg.train.batch_size = 8
    cfg.train.n_epochs = 1
    return cfg


def test_relative_features_causal_and_scale_invariant():
    df = frame()
    _, first = relative_features(df)
    _, scaled = relative_features(frame(100))
    np.testing.assert_allclose(first, scaled, atol=1e-12)
    changed = df.copy()
    changed.loc[30:, ['open', 'high', 'low', 'close', 'volume']] *= 1000
    _, second = relative_features(changed)
    np.testing.assert_array_equal(first[:30], second[:30])


def test_dataset_train_only_fit_and_episode_isolation(config):
    data = build_dataset(config.data.cache_dir, 3)
    expected = []
    for symbol in ['000001', '600519']:
        df, values = relative_features(pd.read_parquet(config.data.cache_dir / f'{symbol}.parquet'))
        expected.append(values[df.date.dt.year <= 2021])
    np.testing.assert_allclose(data.normalizer.mean, np.concatenate(expected).mean(axis=0), atol=1e-12)
    original = data.normalizer.to_dict()
    for p in config.data.cache_dir.glob('*.parquet'):
        df = pd.read_parquet(p)
        df.loc[df.date.dt.year > 2021, ['open', 'high', 'low', 'close', 'volume']] *= 10000
        df.to_parquet(p)
    updated = build_dataset(config.data.cache_dir, 3)
    assert updated.normalizer.to_dict() == original
    assert data.split['ranges'] == DEFAULT_RANGES
    for name, episodes in data.pools.items():
        assert episodes
        for episode in episodes:
            assert len(np.unique(episode.sample.symbols)) == 1
            assert np.isfinite(episode.sample.obs).all()
            assert episode.sample.obs.shape[1:] == (3, len(FEATURES))
            years = episode.sample.dates.astype('U8')
            assert int(years[0][:4]) >= DEFAULT_RANGES[name][0]
            assert int(years[-1][:4]) <= DEFAULT_RANGES[name][1]
    env = EpisodePoolEnv(data.pools['train'], config.env)
    check_env(env, skip_render_check=True)
    ids = {env.reset()[1]['episode_id'] for _ in range(30)}
    assert len(ids) == 2
    env.inner.step(1)
    env.reset()
    assert env.inner.state.shares == 0


def test_lock_and_missing_training_cache_failure(config):
    lock = config.paths.root / 'data/scale/state/unified.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('existing', encoding='utf-8')
    with pytest.raises(ScaleError, match='已有任务'):
        UnifiedRunner(config).run(8, 'locked')
    assert lock.read_text(encoding='utf-8') == 'existing'
    lock.unlink()
    config.data.cache_dir = config.paths.root / 'empty_cache'
    with pytest.raises(ScaleError, match='没有股票缓存'):
        UnifiedRunner(config).run(8, 'failed')
    assert read_status(config.paths.root)['process_state'] == 'failed'
    assert not lock.exists()


def test_unified_reward_next_day_without_changing_old_env(config):
    data = build_dataset(config.data.cache_dir, 3)
    sample = data.pools['train'][0].sample
    env = UnifiedTradingEnv(sample, config.env)
    env.reset()
    _, reward, _, _, _ = env.step(1)
    assert env.state.portfolio_value == pytest.approx(env.state.cash + env.state.shares * sample.prices[1])
    assert np.isfinite(reward)


def test_invalid_ranges_and_no_cache(config, tmp_path):
    with pytest.raises(ScaleError, match='不重叠'):
        build_dataset(config.data.cache_dir, ranges={'train': [2006, 2022], 'val': [2022, 2023], 'test': [2024, 2025]})
    with pytest.raises(ScaleError, match='没有股票缓存'):
        build_dataset(tmp_path / 'missing')
    with pytest.raises(ScaleError, match='run_id'):
        UnifiedRunner(config).run(8, '../bad')


def test_real_sb3_train_resume_replay(config):
    pytest.importorskip('stable_baselines3')
    directory = UnifiedRunner(config).run(16, 'smoke', n_envs=2, checkpoint_steps=8)
    for name in ['model.zip', 'manifest.json', 'normalizer.json', 'split.json', 'metrics.json']:
        assert (directory / name).is_file()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['contract'] == CONTRACT
    assert manifest['observation_shape'] == [3, len(FEATURES)]
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['status'] == 'done'
    assert metrics['evaluation']['val']['used_for_training'] is False
    assert len(metrics['evaluation']['test']['episodes']) == 4
    state = read_status(config.paths.root)
    assert state['progress'] == 1 and state['checkpoint_timesteps'] >= 16
    assert not Path(config.paths.root / 'data/scale/state/unified.lock').exists()
    service = ReplayService(config, ScalePaths(config.paths.root))
    catalog = service.catalog()
    assert catalog['unified_runs'][0]['compatible']
    replay = service.run({'symbol': '600519', 'start': '20240104', 'end': '20240119', 'model_id': 'unified:smoke'})
    assert replay['ok'] and replay['model']['mode'] == 'unified'
    assert replay['daily'] and not replay['temporal_risk']
    assert any('统一模型' in s for s in replay['warnings'])
    norm_before = (directory / 'normalizer.json').read_bytes()
    UnifiedRunner(config).run(32, 'smoke', resume=True)
    assert read_status(config.paths.root)['timesteps'] >= 32
    assert (directory / 'normalizer.json').read_bytes() == norm_before
    changed = pd.read_parquet(config.data.cache_dir / '000001.parquet')
    changed.loc[0, 'volume'] *= 2
    changed.to_parquet(config.data.cache_dir / '000001.parquet')
    with pytest.raises(ScaleError, match='sources'):
        UnifiedRunner(config).run(40, 'smoke', resume=True)
    manifest['contract'] = 'bad'
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert not service.catalog()['models'][0]['compatible']
    with pytest.raises(ReplayError, match='不兼容'):
        service.run({'symbol': '600519', 'start': '20240104', 'end': '20240119', 'model_id': 'unified:smoke'})


def test_pause_resume_stop_checkpoint(config, monkeypatch):
    pytest.importorskip('stable_baselines3')
    from stock_ai.train import sb3_trainer
    original = sb3_trainer.learn_unified
    runner = UnifiedRunner(config)
    phases = []
    def learn(trainer, remaining, tick):
        write_control(config.paths.root, 'pause')
        def resume(_):
            phases.append(read_status(config.paths.root)['process_state'])
            write_control(config.paths.root, 'resume')
        monkeypatch.setattr('stock_ai.scale.unified.time.sleep', resume)
        def controlled(model):
            keep = tick(model)
            if model.num_timesteps >= 3:
                write_control(config.paths.root, 'stop')
                return tick(model)
            return keep
        original(trainer, remaining, controlled)
    monkeypatch.setattr(sb3_trainer, 'learn_unified', learn)
    directory = runner.run(16, 'controlled')
    assert 'paused' in phases
    assert (directory / 'model.zip').exists()
    assert read_status(config.paths.root)['process_state'] == 'stopped'
    assert read_status(config.paths.root)['timesteps'] == 3


def test_ui_status_control_and_config_snapshot(config, monkeypatch):
    paths = ScalePaths(config.paths.root)
    ui = ScaleUI(config, paths, port=0)
    server = ui.make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/api/unified/status') as response:
            data = json.load(response)
        assert data['ok'] and data['process_state'] == 'idle' and data['runs'] == []
        assert ui.unified_control({'action': 'pause'})['ok']
        assert not ui.unified_control({'action': 'start', 'timesteps': 'bad'})['ok']
        assert not ui.unified_control({'action': 'start', 'resume': True})['ok']
        calls = []
        class Proc:
            pid = 123
            def poll(self):
                return None
        def popen(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return Proc()
        monkeypatch.setattr('stock_ai.scale.ui.server.subprocess.Popen', popen)
        assert ui.unified_control({'action': 'start', 'timesteps': 8})['ok']
        assert ui.unified_control({'action': 'start', 'timesteps': 8})['already']
        assert len(calls) == 1 and 'unified-train' in calls[0][0]
        snapshot = json.loads(calls[0][1]['env']['STOCKAI_UNIFIED_CONFIG'])
        assert snapshot['data']['cache_dir'] == str(config.data.cache_dir)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cli_aliases_and_short_train(config, tmp_workspace):
    pytest.importorskip('stable_baselines3')
    config_file = tmp_workspace / 'config.json.yaml'
    import yaml
    config_file.write_text(yaml.safe_dump(config.to_dict()), encoding='utf-8')
    runner = CliRunner()
    for command in [['scale', 'unified-train'], ['scale', 'unified', 'train']]:
        result = runner.invoke(cli, command + ['--help'])
        assert result.exit_code == 0 and '--resume' in result.output
    result = runner.invoke(cli, ['scale', 'unified', 'train', '--config', str(config_file), '--timesteps', '8', '--run-id', 'cli_smoke'])
    assert result.exit_code == 0, result.output
    assert '产物' in result.output
    result = runner.invoke(cli, ['scale', 'unified', 'status', '--config', str(config_file)])
    assert result.exit_code == 0 and 'cli_smoke' in result.output

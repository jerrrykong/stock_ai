"""Mechanism tests only: no policy training or project cache modification."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from stock_ai.config.schema import AppConfig, EnvConfig
from stock_ai.data.corporate_actions import CorporateAction, CorporateActionError, CorporateActionLedger
from stock_ai.env.corporate_action_env import CorporateActionExecutionEnv, EXECUTION_VERSION
from stock_ai.env.strict_env import ExecutionBars, StrictExecutionEnv
from stock_ai.scale.errors import ScaleError
from stock_ai.scale.unified import (EXECUTION_CORPORATE_ACTIONS, EXECUTION_STRICT,
                                    StrictEpisodePoolEnv, UnifiedRunner, build_dataset,
                                    evaluate_strict_episode, make_strict_env)

SOURCE = ("https://example.org/synthetic-mechanism-fixture",)


def coverage(start="20230101", end="20251231"):
    return {"start": start, "end": end, "status": "verified",
            "scope": "all_shareholder_distributions_and_capital_changes",
            "sources": list(SOURCE), "verified_at": "2026-10-16",
            "notes": "Synthetic complete fixture, not market data."}


def event(**changes):
    base = dict(event_id="synthetic-1", kind="distribution", status="implemented",
                record_date="20230103", ex_date="20230104", pay_date="20230109",
                listing_date="20230110", cash_per_share_gross="1", bonus_ratio="0",
                transfer_ratio="1", listing_date_source="sina_red_share_listing",
                sources=SOURCE)
    return CorporateAction(**(base | changes))


def config(**changes):
    return EnvConfig(**(dict(initial_cash=10000., commission_rate=0., stamp_tax_rate=0.,
                            slippage_rate=0., min_commission=0., lot_size=100,
                            reward_type="gross", lambda_turnover=0.,
                            lambda_drawdown=0., lambda_leverage=0.) | changes))


def market(dates=None, closes=None, events=None, symbol="600519"):
    dates = dates or ["20230102", "20230103", "20230104", "20230105", "20230109", "20230110"]
    closes = np.asarray(closes if closes is not None else [10, 10, 4.5, 4.5, 4.5, 4.5], dtype="float64")
    ledger = CorporateActionLedger(symbol, coverage(), tuple(events if events is not None else [event()]))
    return ExecutionBars(np.asarray(dates), closes.copy(), closes * 1.01, closes * .99,
                         closes.copy(), np.ones(len(dates)) * 1e6, symbol, ledger)


def environment(bars=None, cfg=None, **kwargs):
    bars = bars or market()
    return CorporateActionExecutionEnv(np.zeros((bars.n_steps, 3, 9), dtype="float32"),
                                       bars, cfg or config(), **kwargs)


def assert_identity(env):
    s = env.state
    position = s.shares * s.prev_close
    unrealized = position - s.shares * s.cost_basis
    assert s.portfolio_value == pytest.approx(s.cash + position + s.dividend_receivable)
    assert s.portfolio_value == pytest.approx(env.cfg.initial_cash + s.realized_pnl + unrealized + s.dividend_income)
    assert 0 <= s.locked_shares <= s.shares


def test_distribution_recognition_not_mechanical_loss_and_payment_not_new_income():
    env = environment()
    env.reset()
    _, _, _, _, record = env.step(1)
    assert env.state.shares == 1000
    assert record["corporate_actions"][0]["stage"] == "record"
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(0)
    assert env.state.shares == 2000 and env.state.locked_shares == 1000
    assert env.state.available_shares == 1000
    assert env.state.dividend_receivable == 1000 and env.state.cash == 0
    assert env.state.cost_basis == 5 and env.state.dividend_income == 1000
    assert info["limit_reference"] == 4.5 and info["limit_down"] == 4.05
    assert_identity(env)
    env.step(0)
    _, reward, _, _, info = env.step(0)
    assert env.state.cash == 1000 and env.state.dividend_receivable == 0
    assert reward == pytest.approx(0)
    assert any(a["stage"] == "cash_payment" for a in info["corporate_actions"])
    _, reward, done, _, info = env.step(0)
    assert env.state.locked_shares == 0 and env.state.available_shares == 2000
    assert reward == pytest.approx(0) and done
    assert any(a["stage"] == "stock_listing" for a in info["corporate_actions"])
    assert_identity(env)


def test_ex_date_sell_only_available_then_listed_stock_sale():
    env = environment()
    env.reset()
    env.step(1)
    _, reward, _, _, info = env.step(2)
    assert info["trade_ok"] and "待上市" in info["blocked_reason"]
    assert env.state.shares == env.state.locked_shares == 1000
    assert env.state.cash == 4500 and env.state.dividend_receivable == 1000
    assert reward == pytest.approx(0)
    assert env.trades[-1]["shares"] == 1000
    assert_identity(env)
    _, _, _, _, info = env.step(2)
    assert not info["trade_ok"] and "待上市" in info["blocked_reason"]
    env.step(0)
    env.step(2)
    assert env.state.shares == 0 and env.state.cash == 10000
    assert env.state.dividend_income == 1000 and env.state.realized_pnl == -1000
    assert_identity(env)


def test_record_date_holder_sells_before_ex_keeps_cash_and_stock_rights():
    dates = ["20230102", "20230103", "20230104", "20230105", "20230109", "20230110"]
    bars = market(dates=dates, closes=[10, 10, 10, 4.5, 4.5, 4.5],
                  events=[event(ex_date="20230105")])
    env = environment(bars)
    env.reset(); env.step(1); env.step(2)
    assert env.state.shares == 0
    env.step(0)
    assert env.state.shares == 1000 and env.state.locked_shares == 1000
    assert env.state.dividend_receivable == 1000
    assert env.state.cash == 10000
    assert_identity(env)


def test_record_date_sell_has_no_rights_and_ex_date_buyer_has_no_old_rights():
    bars = market(dates=["20230102", "20230103", "20230104", "20230105", "20230109", "20230110"],
                  closes=[10, 10, 10, 4.5, 4.5, 4.5],
                  events=[event(record_date="20230104", ex_date="20230105")])
    env = environment(bars)
    env.reset(); env.step(1); env.step(2); env.step(1)
    assert env.state.dividend_income == 0 and env.state.locked_shares == 0
    assert all(a.get("eligible_shares", 0) == 0 for a in env.corporate_action_log)
    assert_identity(env)


def test_receivable_is_not_spendable_before_payment():
    env = environment()
    env.reset(); env.step(1)
    _, _, _, _, info = env.step(1)
    assert not info["trade_ok"] and "应收" in info["blocked_reason"]
    assert len(env.trades) == 1 and env.state.dividend_receivable == 1000
    env.step(0); env.step(1)
    assert len(env.trades) == 2 and env.trades[-1]["shares"] == 200
    assert env.state.dividend_receivable == 0
    assert_identity(env)


def test_payment_and_listing_on_nonbar_date_catch_up_before_next_open():
    bars = market(events=[event(pay_date="20230107", listing_date="20230108")])
    env = environment(bars)
    env.reset(); env.step(1); env.step(0); env.step(0)
    _, _, _, _, info = env.step(2)
    assert env.state.shares == 0 and env.state.cash == 10000
    logs = info["corporate_actions"]
    assert {a["effective_date"] for a in logs if "effective_date" in a} == {"20230107", "20230108"}
    assert_identity(env)


def test_same_day_ex_cash_and_listing_do_not_double_recognize():
    env = environment(market(events=[event(pay_date="20230104", listing_date="20230104")]))
    env.reset(); env.step(1)
    _, reward, _, _, _ = env.step(0)
    assert env.state.cash == 1000 and env.state.shares == 2000 and env.state.locked_shares == 0
    assert env.state.dividend_receivable == 0 and env.state.dividend_income == 1000
    assert reward == pytest.approx(0)
    env.step(0)
    assert env.state.dividend_income == 1000
    assert_identity(env)


def test_start_after_record_has_zero_rights_but_correct_ex_reference():
    bars = market(events=[event(record_date="20221230", ex_date="20230103")],
                  closes=[10, 4.5, 4.5, 4.5, 4.5, 4.5])
    env = environment(bars)
    env.reset(); _, _, _, _, info = env.step(1)
    assert env.state.shares == 2200 and env.state.locked_shares == 0
    assert env.state.dividend_income == 0
    assert info["limit_reference"] == 4.5


def test_cross_year_continuous_episode_keeps_rights():
    ev = event(record_date="20231229", ex_date="20240102", pay_date="20240104", listing_date="20240104")
    bars = market(dates=["20231228", "20231229", "20240102", "20240103", "20240104"],
                  closes=[10, 10, 4.5, 4.5, 4.5], events=[ev])
    env = environment(bars)
    env.reset(); env.step(1); env.step(0); env.step(0); env.step(0)
    assert env.state.shares == 2000 and env.state.dividend_income == 1000
    assert env.state.portfolio_value == pytest.approx(10000)


@pytest.mark.parametrize("which", ["record", "ex"])
def test_missing_record_or_ex_bar_refused_instead_of_date_guess(which):
    ev = event(record_date="20230107", ex_date="20230109") if which == "record" else event(ex_date="20230107")
    with pytest.raises(CorporateActionError, match="缺少真实日线"):
        environment(market(events=[ev]))


def test_fractional_shares_fail_not_silently_truncated():
    env = environment(market(events=[event(transfer_ratio="0.0001")]))
    env.reset(); env.step(1)
    before = env.state.snapshot()
    with pytest.raises(CorporateActionError, match="碎股"):
        env.step(0)
    assert env.state.snapshot() == before


def test_halted_ex_date_still_settles_entitlements_but_rejects_order():
    bars = market()
    bars.volume[2] = 0
    env = environment(bars)
    env.reset(); env.step(1); _, reward, _, _, info = env.step(2)
    assert not info["trade_ok"] and info["blocked_reason"] == "停牌"
    assert env.state.shares == 2000 and env.state.dividend_receivable == 1000
    assert reward == pytest.approx(0)


def test_same_environment_benchmark_and_excess_are_fee_and_ca_consistent():
    cfg = config(reward_type="excess", commission_rate=.001, stamp_tax_rate=.001,
                 min_commission=5., slippage_rate=.001)
    env = environment(cfg=cfg)
    env.reset()
    rewards = []
    for _ in range(env.n_steps - 1):
        _, reward, _, _, info = env.step(1)
        rewards.append(reward)
        assert info["step_excess"] == pytest.approx(0)
        assert_identity(env)
    assert rewards == pytest.approx([0.] * (env.n_steps - 1))
    np.testing.assert_allclose(env.equity, env.benchmark_equity)
    np.testing.assert_allclose(env.equity, env.buy_and_hold())
    assert env.buy_and_hold()[0] == cfg.initial_cash
    assert env.metrics()["excess_return"] == pytest.approx(0)


def test_v1_keeps_old_loss_even_if_bars_carries_ledger():
    bars = market()
    old = StrictExecutionEnv(np.zeros((6, 3, 9)), bars, config())
    old.reset(); old.step(1); old.step(0)
    assert old.state.shares == 1000 and old.state.portfolio_value == 4500
    new = make_strict_env(np.zeros((6, 3, 9)), bars, config(), execution_version=EXECUTION_VERSION)
    new.reset(); new.step(1); new.step(0)
    assert new.state.portfolio_value == 10000


def test_coverage_missing_and_unsupported_fail_closed():
    bars = market()
    bars.corporate_actions = None
    with pytest.raises(CorporateActionError, match="账本"):
        environment(bars)
    bars = market()
    bars.corporate_actions = replace(bars.corporate_actions, coverage=coverage("20230104"))
    with pytest.raises(CorporateActionError, match="覆盖不足"):
        environment(bars)
    unsupported = event(kind="rights_issue")
    with pytest.raises(CorporateActionError, match="不支持"):
        environment(market(events=[unsupported]))


def test_reset_and_gym_contract_and_pool_version():
    bars = market(events=[], closes=[10] * 6)
    env = environment(bars)
    check_env(env, skip_render_check=True)
    env.reset(); env.step(1); env.reset()
    assert env.state.shares == env.state.dividend_income == 0
    pool = StrictEpisodePoolEnv([("fixture", np.zeros((6, 3, 9)), bars)], config(),
                                execution_version=EXECUTION_VERSION)
    assert isinstance(pool.inner, CorporateActionExecutionEnv)
    check_env(pool, skip_render_check=True)
    pool.close()


def dataset_files(tmp_path):
    feature, raw, actions = [tmp_path / part for part in ("hfq", "raw", "actions")]
    for path in (feature, raw, actions):
        path.mkdir()
    dates = pd.DatetimeIndex([])
    for year in (2021, 2022, 2023):
        dates = dates.append(pd.bdate_range(f"{year}-01-04", periods=4))
    closes = np.linspace(10, 10.5, len(dates))
    df = pd.DataFrame(dict(date=dates, open=closes, high=closes*1.01, low=closes*.99,
                           close=closes, volume=np.ones(len(dates))*1e6))
    df.to_parquet(raw / "600519.parquet", index=False)
    hfq = df.copy()
    hfq.loc[:, ["open", "high", "low", "close"]] *= 10
    hfq.to_parquet(feature / "600519.parquet", index=False)
    ledger = CorporateActionLedger("600519", coverage("20210101", "20231231"))
    (actions / "600519.json").write_text(json.dumps(ledger.to_dict()), encoding="utf-8")
    return feature, raw, actions


def dataset(tmp_path):
    feature, raw, actions = dataset_files(tmp_path)
    return build_dataset(feature, 3, ranges={"train": [2021, 2021], "val": [2022, 2022], "test": [2023, 2023]},
                         execution_dir=raw, execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=actions)


def test_dataset_v2_hashes_raw_and_events_preserves_ledger_slicing(tmp_path):
    data = dataset(tmp_path)
    source = data.sources[0]
    assert len(source["execution_sha256"]) == len(source["corporate_actions_sha256"]) == 64
    for pool in data.strict_pools.values():
        assert pool[0][2].corporate_actions.schema == "corporate-actions-v1"
    class Hold:
        def predict(self, obs):
            return 0, None
    record = evaluate_strict_episode(Hold(), data.strict_pools["test"][0], config(),
                                     execution_version=EXECUTION_CORPORATE_ACTIONS)
    assert record["execution_version"] == EXECUTION_VERSION
    assert record["benchmark_return_cc"] is None


@pytest.mark.parametrize("missing", ["raw", "events", "day"])
def test_dataset_v2_never_silently_drops_missing_inputs(tmp_path, missing):
    feature, raw, actions = dataset_files(tmp_path)
    if missing == "raw":
        (raw / "600519.parquet").unlink()
    elif missing == "events":
        (actions / "600519.json").unlink()
    else:
        df = pd.read_parquet(raw / "600519.parquet").drop(index=1)
        df.to_parquet(raw / "600519.parquet", index=False)
    with pytest.raises((CorporateActionError, ScaleError)):
        build_dataset(feature, 3, ranges={"train": [2021, 2021], "val": [2022, 2022], "test": [2023, 2023]},
                      execution_dir=raw, execution_version=EXECUTION_CORPORATE_ACTIONS,
                      corporate_actions_dir=actions)


def test_resume_execution_override_refused_before_any_writes_or_training(tmp_path, monkeypatch):
    cfg = AppConfig()
    cfg.paths.root = tmp_path
    run = tmp_path / "models" / "unified" / "old"
    run.mkdir(parents=True)
    (run / "model.zip").write_bytes(b"old-policy-not-loaded")
    (run / "manifest.json").write_text(json.dumps({"execution_version": EXECUTION_STRICT}), encoding="utf-8")
    before = {str(p): p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr("stock_ai.scale.unified.build_dataset", lambda *a, **k: pytest.fail("must refuse before data build"))
    with pytest.raises(Exception, match="execution_version"):
        UnifiedRunner(cfg).run(8, "old", resume=True, execution_version=EXECUTION_VERSION)
    assert before == {str(p): p.read_bytes() for p in run.iterdir()}
    assert not (tmp_path / "data").exists()


def test_cli_ca_validate_import_and_snapshot_are_explicit(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from stock_ai.cli.main import cli
    from stock_ai.data.corporate_actions import load_ledger
    source = tmp_path / "reviewed.json"
    source.write_text(json.dumps(CorporateActionLedger("600519", coverage(), (event(),)).to_dict()), encoding="utf-8")
    runner = CliRunner()
    command = ["scale", "corporate-actions"]
    validate = runner.invoke(cli, command + ["validate", str(source), "--start", "20230102", "--end", "20230110"])
    assert validate.exit_code == 0, validate.output
    assert json.loads(validate.output)["read_only"] is True
    output = tmp_path / "imported"
    imported = runner.invoke(cli, command + ["import", str(source), "--start", "20230102", "--end", "20230110", "--output-dir", str(output)])
    assert imported.exit_code == 0, imported.output
    target = output / "600519.json"
    before = target.read_bytes()
    assert load_ledger(target).to_dict() == load_ledger(source).to_dict()
    again = runner.invoke(cli, command + ["import", str(source), "--start", "20230102", "--end", "20230110", "--output-dir", str(output)])
    assert again.exit_code != 0 and target.read_bytes() == before
    calls = []
    def fetch(symbol, directory):
        calls.append((symbol, directory))
        return dict(symbol=symbol, status="unreviewed", count=0, total=0,
                    fetched_at="2026-10-16T00:00:00Z", response_sha256="0" * 64,
                    raw_response_path="mock-response", metadata_path="mock-snapshot")
    monkeypatch.setattr("stock_ai.data.corporate_action_sources.fetch_cninfo_snapshot", fetch)
    result = runner.invoke(cli, command + ["fetch-snapshot", "--symbol", "600519", "--output-dir", str(tmp_path / "sources")])
    assert result.exit_code == 0 and "unreviewed" in result.output, result.output
    assert calls == [("600519", tmp_path / "sources")]


def test_runner_v2_manifest_and_version_wiring_without_training(tmp_path, monkeypatch):
    from stock_ai.train import sb3_trainer
    feature, raw, actions = dataset_files(tmp_path)
    cfg = AppConfig()
    cfg.paths.root = tmp_path
    cfg.data.cache_dir = raw
    cfg.sample.window = 3
    cfg.env = config(reward_type="excess")
    captured = {}
    class FakeEnv:
        def close(self):
            captured["closed"] = True
    class FrozenModel:
        num_timesteps = 8
    class FakeTrainer:
        model = FrozenModel()
        env = FakeEnv()
        def predict(self, obs):
            return 0, None
        def save(self, path):
            path.write_bytes(b"NOT_A_REAL_MODEL_mock_only")
    def make(*args, **kwargs):
        captured.update(kwargs)
        return FakeTrainer()
    monkeypatch.setattr(sb3_trainer, "make_unified_trainer", make)
    monkeypatch.setattr(sb3_trainer, "learn_unified", lambda *a, **k: pytest.fail("No training is permitted"))
    directory = UnifiedRunner(cfg).run(8, "mock_only", ranges={"train": [2021, 2021], "val": [2022, 2022], "test": [2023, 2023]},
                                       feature_dir=feature, corporate_actions_dir=actions,
                                       execution_version=EXECUTION_VERSION)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert captured["execution_version"] == EXECUTION_VERSION and captured["strict"]
    assert captured["closed"]
    assert manifest["execution_version"] == EXECUTION_VERSION
    assert manifest["reward_version"] == "excess-ca-constant-buy-v2"
    assert manifest["corporate_actions_schema"] == "corporate-actions-v1"
    assert manifest["tax_policy"] == "gross_before_personal_dividend_tax"
    assert manifest["benchmark_definition"] == "ca_gross_constant_buy_same_environment"
    assert all(len(source["execution_sha256"]) == len(source["corporate_actions_sha256"]) == 64 for source in manifest["sources"])
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["status"] == "done"
    assert metrics["evaluation"]["val"]["episodes"][0]["execution_version"] == EXECUTION_VERSION
    before = (directory / "manifest.json").read_bytes()
    with (actions / "600519.json").open("a") as handle:
        handle.write(" ")
    monkeypatch.setattr(sb3_trainer, "make_unified_trainer", lambda *a, **k: pytest.fail("Changed sources must reject before trainer load"))
    with pytest.raises(ScaleError, match="sources"):
        UnifiedRunner(cfg).run(8, "mock_only", resume=True, execution_version=EXECUTION_VERSION)
    assert (directory / "manifest.json").read_bytes() == before
    assert (directory / "model.zip").read_bytes() == b"NOT_A_REAL_MODEL_mock_only"


@pytest.mark.parametrize("kind", ["feature", "execution", "corporate_actions"])
def test_dataset_v2_one_read_snapshot_binds_hashes_features_and_episode_bars(tmp_path, monkeypatch, kind):
    """Concurrent path replacement cannot put different bytes in sources and pools."""
    import hashlib
    from io import BytesIO
    from pathlib import Path
    from stock_ai.scale import unified

    feature, raw, actions = dataset_files(tmp_path)
    feature_path, raw_path, ledger_path = feature / "600519.parquet", raw / "600519.parquet", actions / "600519.json"
    original_read = Path.read_bytes
    snapshots = {p.resolve(): original_read(p) for p in (feature_path, raw_path, ledger_path)}
    expected_feature = pd.read_parquet(BytesIO(snapshots[feature_path.resolve()]))
    expected_raw = pd.read_parquet(BytesIO(snapshots[raw_path.resolve()]))
    target = {"feature": feature_path, "execution": raw_path, "corporate_actions": ledger_path}[kind].resolve()
    if kind == "corporate_actions":
        changed = json.loads(snapshots[target])
        # New invalid coverage would reject if the loader reopened the path for episode construction.
        changed["coverage"]["status"] = "unverified"
        replacement = json.dumps(changed).encode("utf-8")
    else:
        changed = pd.read_parquet(BytesIO(snapshots[target]))
        changed.loc[:, ["open", "high", "low", "close"]] *= 2
        changed["volume"] *= 3
        buffer = BytesIO()
        changed.to_parquet(buffer, index=False)
        replacement = buffer.getvalue()
    counts = {p: 0 for p in snapshots}

    def swap_after_read(path):
        content = original_read(path)
        resolved = path.resolve()
        if resolved in counts:
            counts[resolved] += 1
        if resolved == target:
            path.write_bytes(replacement)
        return content

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    monkeypatch.setattr(unified, "sha256_file", lambda *a: pytest.fail("v2 禁止解析后按路径重读指纹"))
    data = build_dataset(feature, 3, ranges={"train": [2021, 2021], "val": [2022, 2022], "test": [2023, 2023]},
                         execution_dir=raw, execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=actions)
    assert all(count == 1 for count in counts.values())
    source = data.sources[0]
    for path, key in ((feature_path, "sha256"), (raw_path, "execution_sha256"), (ledger_path, "corporate_actions_sha256")):
        assert source[key] == hashlib.sha256(snapshots[path.resolve()]).hexdigest()
    assert hashlib.sha256(original_read(target)).hexdigest() != hashlib.sha256(snapshots[target]).hexdigest()
    _, values = unified.relative_features(expected_feature)
    expected_obs = unified.observations(unified.normalize(values, data.normalizer), 3)
    ledger_objects = []
    for index, name in enumerate(("train", "val", "test")):
        _, obs, bars = data.strict_pools[name][0]
        np.testing.assert_array_equal(obs, expected_obs[index * 4:(index + 1) * 4])
        np.testing.assert_array_equal(bars.close, expected_raw.close.to_numpy()[index * 4:(index + 1) * 4])
        assert bars.corporate_actions.sha256 == source["corporate_actions_sha256"]
        assert bars.corporate_actions.coverage["status"] == "verified"
        ledger_objects.append(bars.corporate_actions)
    assert all(ledger is ledger_objects[0] for ledger in ledger_objects)

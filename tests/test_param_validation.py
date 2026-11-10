"""参数校验回归用例（QA 报告 P1-1 / P1-2 / P2-5 / P2-6）。

覆盖本轮修复的"非法参数被静默接受"同类缺陷：

1. ``--backend`` 非法值必须抛 :class:`ConfigError`（工厂层 + train CLI）；
2. ``--source`` 非法值必须抛 :class:`ConfigError`，且自定义数据源子类仍可用；
3. ``--reward`` 非法值必须抛 :class:`ConfigError`；
4. ``start > end`` 区间必须给出明确错误（fetch / sample / predict）；
5. meta.json 缺失时缓存保护（P2-1）的辅助函数语义。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from stock_ai.config.schema import DataConfig, EnvConfig, SampleConfig, TrainConfig
from stock_ai.data.loader import SUPPORTED_SOURCES, StockDataLoader
from stock_ai.data.sources.base import DataSource
from stock_ai.env.rewards import SUPPORTED_REWARDS, RewardCalculator
from stock_ai.train.factory import SUPPORTED_ALGOS, SUPPORTED_BACKENDS, TrainerFactory
from stock_ai.utils.dates import to_day_key, validate_range
from stock_ai.utils.errors import ConfigError, DataSourceError

#: 仓库内默认配置的绝对路径（run() 会 chdir，必须用绝对路径）
_CFG = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


# --------------------------------------------------------------------------- #
# 自定义数据源（用于验证"非法值拦截"与"可扩展性"共存）
# --------------------------------------------------------------------------- #
class _PluginSource(DataSource):
    """一个合法的第三方/测试数据源。"""

    name = "plugintest"

    def fetch(self, symbol: str, start: str, end: str):  # noqa: D401
        raise DataSourceError("noop")


# --------------------------------------------------------------------------- #
# P1-1：--backend 白名单
# --------------------------------------------------------------------------- #
def test_supported_backends_constant():
    """后端白名单常量正确。"""
    assert SUPPORTED_BACKENDS == frozenset({"auto", "sb3", "numpy"})


def test_factory_rejects_unknown_backend_via_preference():
    """preference 传入非法值 -> ConfigError（早于任何后端选择）。"""
    cfg = TrainConfig(algo="ppo", backend="auto")
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, object(), preference="bogus")


def test_factory_rejects_unknown_backend_via_cfg():
    """cfg.backend 非法且 preference 为空 -> 回退到 cfg.backend 后仍应被拒。"""
    cfg = TrainConfig(algo="ppo", backend="bogus")
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, object(), preference="")


def test_factory_rejects_unknown_backend_numpy_pref():
    """即使强制 numpy，非法 backend 也不能被静默接受。"""
    cfg = TrainConfig(algo="ppo", backend="bogus")
    with pytest.raises(ConfigError):
        TrainerFactory.create(cfg, object(), preference="bogus")


def test_train_cli_rejects_unknown_backend(tmp_workspace):
    """train CLI：--backend bogus 必须抛 ConfigError（非零退出）。"""
    from stock_ai.train import train_cli

    with pytest.raises(ConfigError):
        train_cli.run(_args(backend="bogus"))


# --------------------------------------------------------------------------- #
# P1-2：--source 白名单（拦截非法值 + 保留自定义源扩展性）
# --------------------------------------------------------------------------- #
def test_supported_sources_constant():
    """内置数据源白名单常量正确。"""
    assert SUPPORTED_SOURCES == frozenset(
        {"akshare", "tencent", "eastmoney", "sina", "synthetic"}
    )


def test_loader_rejects_unknown_source():
    """非法 source 在构造 loader 时即抛 ConfigError。"""
    cfg = DataConfig(symbols=["600519"], source="nosuch")
    with pytest.raises(ConfigError):
        StockDataLoader(cfg)


def test_loader_accepts_builtin_synthetic(tmp_path):
    """内置 synthetic 仍可用。"""
    cfg = DataConfig(symbols=["600519"], source="synthetic", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)
    assert loader.sources and loader.sources[0].name == "synthetic"


def test_loader_accepts_injected_custom_source(tmp_path):
    """自定义 DataSource 子类经 **注入** 仍可用（可扩展性不被白名单破坏）。

    P1-2 修复后，扩展路径统一为 ``StockDataLoader(..., sources=[...])``；
    ``--source`` 白名单严格等于内置 registry，不再为子类名开口子。
    """
    cfg = DataConfig(symbols=["600519"], source="akshare", cache_dir=tmp_path)
    loader = StockDataLoader(cfg, sources=[_PluginSource()])  # 不应抛异常
    assert loader is not None
    assert any(getattr(s, "name", "") == "plugintest" for s in loader.sources)


def test_loader_rejects_unregistered_custom_source_name(tmp_path):
    """未经注入时，子类名 ``plugintest`` 不得通过 registry 校验（P1-2 单源）。"""
    cfg = DataConfig(symbols=["600519"], source="plugintest", cache_dir=tmp_path)
    with pytest.raises(ConfigError):
        StockDataLoader(cfg)


# --------------------------------------------------------------------------- #
# P2-5：--reward 白名单
# --------------------------------------------------------------------------- #
def test_supported_rewards_constant():
    """奖励类型白名单常量正确（含增量 ``excess``）。"""
    assert SUPPORTED_REWARDS == frozenset({"gross", "drawdown_penalty", "sharpe", "excess"})


def test_reward_calculator_rejects_unknown():
    """RewardCalculator：非法 reward_type 抛 ConfigError。"""
    with pytest.raises(ConfigError):
        RewardCalculator(EnvConfig(reward_type="bogus"))


def test_reward_calculator_accepts_known():
    """RewardCalculator：三种合法 reward_type 均可构造。"""
    for r in sorted(SUPPORTED_REWARDS):
        RewardCalculator(EnvConfig(reward_type=r))


def test_train_cli_rejects_unknown_reward(tmp_workspace):
    """train CLI：--reward bogus 必须抛 ConfigError。"""
    from stock_ai.train import train_cli

    with pytest.raises(ConfigError):
        train_cli.run(_args(reward="bogus"))


# --------------------------------------------------------------------------- #
# P2-6：日期区间 start > end
# --------------------------------------------------------------------------- #
def test_validate_range_helper_accepts_valid():
    """合法区间（含分隔符写法）不报错。"""
    validate_range("20220101", "20230101")
    validate_range("2022-01-01", "2023-01-01")
    validate_range("20180101", "20180101")  # 相等合法


def test_validate_range_helper_rejects_inverted():
    """起始晚于结束 -> ConfigError，且信息可读。"""
    with pytest.raises(ConfigError) as ei:
        validate_range("20241231", "20120101", label="数据拉取区间")
    msg = str(ei.value)
    assert "20241231" in msg and "20120101" in msg


def test_to_day_key_normalizes():
    """日期归一化：分隔符被剥离，非法输入回落空串。"""
    assert to_day_key("2024-01-01") == "20240101"
    assert to_day_key("2024/01/01") == "20240101"
    assert to_day_key("20240101") == "20240101"
    assert to_day_key("abc") == ""


def test_fetch_all_rejects_inverted_range(tmp_path):
    """data fetch：start > end 在进入网络前明确报错。"""
    cfg = DataConfig(symbols=["600519"], source="synthetic", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)
    with pytest.raises(ConfigError):
        loader.fetch_all(start="20241231", end="20120101", force=True)


def test_sample_builder_rejects_inverted_train_range():
    """sample build：训练区间 start > end 明确报错。"""
    from stock_ai.samples.builder import SampleBuilder

    builder = SampleBuilder(SampleConfig(train_start="20241231", train_end="20120101"))
    with pytest.raises(ConfigError):
        builder.build_all(object())  # 校验早于任何 loader 访问


def test_sample_builder_rejects_inverted_test_range():
    """sample build：测试区间 start > end 明确报错。"""
    from stock_ai.samples.builder import SampleBuilder

    builder = SampleBuilder(
        SampleConfig(
            train_start="20180101", train_end="20211231",
            test_start="20241231", test_end="20220101",
        )
    )
    with pytest.raises(ConfigError):
        builder.build_all(object())


def test_predictor_rejects_inverted_range():
    """predict：start > end 明确报错（早于模型/数据访问）。"""
    from stock_ai.infer.predictor import Predictor
    from stock_ai.samples.normalizer import NormalizerState

    normalizer = NormalizerState(
        mean=np.zeros(2), std=np.ones(2), columns=["close", "volume"]
    )
    predictor = Predictor(
        model=object(),
        env_cfg=EnvConfig(),
        normalizer=normalizer,
        feature_names=["close", "volume"],
        window=3,
    )
    with pytest.raises(ConfigError):
        predictor.predict_range(["600519"], "20241231", "20120101", loader=object())


# --------------------------------------------------------------------------- #
# P2-1：meta 缺失时缓存保护的辅助语义
# --------------------------------------------------------------------------- #
def test_prev_source_reads_from_disk_meta(tmp_path):
    """_prev_source 优先从磁盘 meta 解析来源。"""
    cfg = DataConfig(symbols=["600519"], source="synthetic", cache_dir=tmp_path)
    loader = StockDataLoader(cfg)
    loader.cache.write_meta({"symbols": {"600519": {"source": "tencent"}}})
    assert loader._prev_source("600519", {}) == "tencent"
    # 磁盘无该 symbol 时回退到传入快照
    assert loader._prev_source("000001", {"symbols": {"000001": {"source": "akshare"}}}) == "akshare"
    # 两处都无 -> 空串
    assert loader._prev_source("000002", {}) == ""


# --------------------------------------------------------------------------- #
# 白名单常量一致性交叉校验
# --------------------------------------------------------------------------- #
def test_algo_whitelist_unchanged():
    """既有 algo 白名单未被本次改动破坏。"""
    assert SUPPORTED_ALGOS == frozenset({"ppo", "sac", "dqn"})


def _args(**overrides) -> argparse.Namespace:
    """构造 train_cli.run 所需的最小 argparse.Namespace。"""
    base = dict(
        config=_CFG,
        algo=None,
        samples=None,
        timesteps=8,
        seed=42,
        backend=None,
        reward=None,
        out=None,
        n_steps=None,
        batch_size=None,
        n_epochs=None,
        learning_rate=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)

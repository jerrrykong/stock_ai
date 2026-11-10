"""独立 unified 契约：逐股票因果特征、时间切分、冻结统计量及共享 PPO。"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from stock_ai.env.trading_env import TradingEnv
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.samples.schema import SampleSplit
from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.errors import ScaleError
from stock_ai.utils.hashing import sha256_file

CONTRACT = "unified-relative-v1"
FEATURES = ["return_1", "open_relative", "high_relative", "low_relative",
            "ma5_relative", "ma20_relative", "momentum_5", "volatility_20", "volume_relative"]
DEFAULT_RANGES = {"train": [2006, 2021], "val": [2022, 2023], "test": [2024, 2025]}
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")

#: 旧语义：同日收盘模拟成交（保留，向后兼容）。
EXECUTION_LEGACY = "legacy-same-day-close-v1"
#: 新语义：T 日信号 -> T+1 日真实开盘成交 + 板块涨跌停拒单。
EXECUTION_STRICT = "strict-t1-open-v1"
EXECUTION_CORPORATE_ACTIONS = "strict-t1-open-ca-v2"
STRICT_EXECUTION_VERSIONS = (EXECUTION_STRICT, EXECUTION_CORPORATE_ACTIONS)
EXECUTION_VERSIONS = (EXECUTION_LEGACY, *STRICT_EXECUTION_VERSIONS)

#: 奖励语义版本：``excess`` 为「相对买入持有基准的超额收益 + 因果 close-to-close」。
REWARD_VERSION_EXCESS = "excess-causal-close-v1"
#: 其余历史奖励类型统一标记为 legacy（语义可区分）。
REWARD_VERSION_LEGACY = "legacy"

#: 评估退化守卫阈值：零交易 episode 占比超过该值即标记 ``degenerate``。
#: 零交易 episode 的超额恒等于 ``−benchmark_return``（一年不交易，基准涨就记负、
#: 基准跌就记正），是定义性假象而非技能。混入 ``mean_excess_return`` 会造出
#: 「全部超额为正、有交易口径为负」的假象（v2 val 即 +2.14% vs 有交易 −4.22%）。
DEGENERATE_ZERO_TRADE_FRACTION = 0.05


def unified_dir(root):
    return Path(root) / "models" / "unified"


def status_path(root):
    return Path(root) / "data" / "scale" / "state" / "unified.json"


def control_path(root):
    return status_path(root).with_name("unified_control.json")


def read_status(root):
    path = status_path(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"process_state": "idle", "timesteps": 0}


def write_control(root, action):
    if action not in {"pause", "resume", "stop"}:
        raise ScaleError("unified 控制动作必须是 pause/resume/stop")
    atomic_write_json({"action": action}, control_path(root))


def relative_features(frame):
    """只用当前及过去数据；不使用 bfill，不使用绝对价格/成交量。"""
    df = frame.copy()
    required = ["open", "high", "low", "close", "volume"]
    if not {"date", *required}.issubset(df.columns):
        raise ScaleError("缓存缺少 date/OHLCV 列")
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    values = df[required].to_numpy(dtype="float64")
    if df.empty:
        raise ScaleError("缓存为空")
    if (df.date.isna().any() or df.date.duplicated().any() or not np.isfinite(values).all()
            or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any()
            or (df.high < df[["open", "close", "low"]].max(axis=1)).any()
            or (df.low > df[["open", "close", "high"]].min(axis=1)).any()):
        raise ScaleError("缓存日期或 OHLCV 不合法")
    close = df.close
    ret = close.pct_change(fill_method=None).fillna(0)
    volume_mean = df.volume.rolling(20, min_periods=1).mean()
    result = pd.DataFrame({
        "return_1": ret,
        "open_relative": df.open / close - 1,
        "high_relative": df.high / close - 1,
        "low_relative": df.low / close - 1,
        "ma5_relative": close / close.rolling(5, min_periods=1).mean() - 1,
        "ma20_relative": close / close.rolling(20, min_periods=1).mean() - 1,
        "momentum_5": close.pct_change(5, fill_method=None).fillna(0),
        "volatility_20": ret.rolling(20, min_periods=2).std(ddof=0).fillna(0),
        "volume_relative": df.volume / volume_mean.replace(0, np.nan) - 1,
    }).fillna(0)[FEATURES].to_numpy(dtype="float64")
    if not np.isfinite(result).all():
        raise ScaleError("相对特征含非有限值")
    return df, result


def normalize(values, norm):
    if (norm.columns != FEATURES or norm.mean.shape != (len(FEATURES),)
            or norm.std.shape != (len(FEATURES),) or not np.isfinite(norm.mean).all()
            or not np.isfinite(norm.std).all() or (norm.std <= 0).any()):
        raise ScaleError("unified normalizer 与特征契约不兼容")
    return np.clip((values - norm.mean) / norm.std, -10, 10).astype("float32")


def observations(values, window):
    padded = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
    return np.lib.stride_tricks.sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)


@dataclasses.dataclass
class Episode:
    id: str
    sample: SampleSplit
    #: 真实未复权 K 线（严格撮合唯一价格来源）；旧模式为 ``None``。
    bars: object = None


@dataclasses.dataclass
class Dataset:
    pools: dict
    normalizer: NormalizerState
    split: dict
    sources: list
    #: ``strict`` 时使用的 episode 池（``(id, obs, bars)`` 三元组），否则为空。
    strict_pools: dict = dataclasses.field(default_factory=dict)
    execution_version: str = EXECUTION_LEGACY


def _load_execution_bars(cache_dir, symbol, hfq_df, *, raw_frame=None):
    """从不复权缓存读出真实 OHLCV，并与 hfq 交易日对齐。

    后复权价只用于特征；成交必须用真实未复权价。这里按日期做 inner join，
    保证特征序列与成交序列严格一一对应（同一 ``T`` 个交易日）。

    Args:
        cache_dir: 不复权缓存目录（``data/raw``）。
        symbol: 股票代码。
        hfq_df: 后复权 DataFrame（提供对齐用的日期序列）。

    Returns:
        :class:`~stock_ai.env.strict_env.ExecutionBars`，或在数据不可用时返回 ``None``。

    Raises:
        ScaleError: 存在不复权缓存但与 hfq 无共同交易日（口径或区间严重错配）。
    """
    from stock_ai.env.strict_env import ExecutionBars

    path = Path(cache_dir) / f"{symbol}.parquet"
    if raw_frame is None:
        if not path.is_file():
            return None
        raw = pd.read_parquet(path)
    else:
        raw = raw_frame.copy()
    raw["date"] = pd.to_datetime(raw["date"].astype(str), errors="coerce")
    if raw["date"].isna().any():
        raise ScaleError(f"不复权缓存日期非法: {symbol}")
    keys = ["date", "open", "high", "low", "close", "volume"]
    left = pd.DataFrame({"date": pd.to_datetime(hfq_df["date"].astype(str)).to_numpy()})
    aligned = left.merge(raw[keys], on="date", how="inner").sort_values(
        "date", kind="stable"
    ).reset_index(drop=True)
    if len(aligned) == 0:
        raise ScaleError(
            f"{symbol} 的后复权与不复权缓存没有共同交易日；请检查 data/raw_hfq 与 data/raw 的区间是否一致"
        )
    values = aligned[["open", "high", "low", "close", "volume"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ScaleError(f"{symbol} 不复权缓存含非有限或非正价格")
    return ExecutionBars(
        dates=aligned["date"].dt.strftime("%Y%m%d").to_numpy(),
        open=aligned["open"].to_numpy(), high=aligned["high"].to_numpy(),
        low=aligned["low"].to_numpy(), close=aligned["close"].to_numpy(),
        volume=aligned["volume"].to_numpy(), symbol=symbol,
    )


def build_dataset(cache_dir, window=20, ranges=None, progress=None,
                  execution_dir=None, execution_version=None, corporate_actions_dir=None):
    """构建统一数据集。

    Args:
        cache_dir: **特征**缓存目录。严格撮合模式下应为后复权目录 ``data/raw_hfq``。
        window: 滑窗长度。
        ranges: ``train/val/test`` 年份区间。
        progress: 进度回调 ``(i, n, symbol)``。
        execution_dir: 真实未复权缓存目录（``data/raw``）。给定时启用严格撮合
            （T 日信号 -> T+1 日真实开盘成交）。
        execution_version: 显式指定执行语义（``strict-t1-open-v1`` / ``legacy``）。

    Returns:
        :class:`Dataset`；``execution_version == EXECUTION_STRICT`` 时
        ``strict_pools`` 给出 ``(id, obs, ExecutionBars)`` 三元组。
    """
    ranges = ranges or DEFAULT_RANGES
    version = execution_version or (EXECUTION_STRICT if execution_dir else EXECUTION_LEGACY)
    if version not in EXECUTION_VERSIONS:
        raise ScaleError(
            f"未知 execution_version: {version!r}（可选: {sorted(EXECUTION_VERSIONS)}）"
        )
    strict = version in STRICT_EXECUTION_VERSIONS
    ca_version = version == EXECUTION_CORPORATE_ACTIONS
    if strict and not execution_dir:
        raise ScaleError("严格撮合必须提供 execution_dir（不复权缓存目录）")
    if ca_version and not corporate_actions_dir:
        raise ScaleError("公司行为 v2 必须提供 corporate_actions_dir 及逐股票已核验覆盖账本")
    if ca_version and Path(cache_dir).resolve() == Path(execution_dir).resolve():
        raise ScaleError("公司行为 v2 后复权特征与未复权执行缓存不能是同一目录")
    if not 1 <= window <= 512:
        raise ScaleError("window 必须为 1~512")
    if set(ranges) != {"train", "val", "test"}:
        raise ScaleError("时间切分必须包含 train/val/test")
    last = 0
    for name in ("train", "val", "test"):
        start, end = ranges[name]
        if not 1900 <= start <= end <= 2100 or start <= last:
            raise ScaleError("train/val/test 年份区间必须严格递增且不重叠")
        last = end
    files = [p for p in sorted(Path(cache_dir).glob("*.parquet"))
             if re.fullmatch(r"[0-9]{6}", p.stem) and p.resolve().parent == Path(cache_dir).resolve()]
    if not files:
        raise ScaleError(
            f"没有股票缓存；unified 不会联网或生成合成数据（{cache_dir}）"
        )
    prepared, sources = [], []
    ca_bars = {}
    count, mean, m2 = 0, np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    for i, path in enumerate(files):
        if progress:
            progress(i, len(files), path.stem)
        # v2 冻结一次文件快照：特征、日历校验、episode 与 manifest 指纹共享字节。
        if ca_version:
            feature_content = path.read_bytes()
            feature_hash = hashlib.sha256(feature_content).hexdigest()
            feature_frame = pd.read_parquet(BytesIO(feature_content))
        else:
            feature_frame = pd.read_parquet(path)
        df, features = relative_features(feature_frame)
        if strict:
            # 特征与成交必须严格同源同日：跨数据源（后复权 vs 不复权）交易日
            # 可能不一致（停牌记录差异），这里以真实 K 线覆盖到的日期为准过滤，
            # 避免后续出现「特征有该日但无成交价」的错位。
            if ca_version:
                execution_path = Path(execution_dir) / path.name
                if (execution_path.resolve().parent != Path(execution_dir).resolve()
                        or not execution_path.is_file()):
                    raise ScaleError(f"{path.stem} 未复权执行缓存缺失或路径不安全；v2 禁止静默丢弃股票")
                execution_content = execution_path.read_bytes()
                execution_hash = hashlib.sha256(execution_content).hexdigest()
                raw_frame = pd.read_parquet(BytesIO(execution_content))
                real = _load_execution_bars(execution_dir, path.stem, df, raw_frame=raw_frame)
            else:
                real = _load_execution_bars(execution_dir, path.stem, df)
            if real is None:
                if ca_version:
                    raise ScaleError(f"{path.stem} 缺少未复权执行缓存；v2 禁止静默丢弃股票")
                continue
            if ca_version:
                # v2 不沿用 inner-join 静默丢日。只比较实际用于 episode 的日历区间。
                raw_days = pd.to_datetime(raw_frame["date"].astype(str), errors="raise")
                if raw_days.duplicated().any():
                    raise ScaleError(f"{path.stem} 未复权日期重复")
                lower, upper = ranges["train"][0], ranges["test"][1]
                feature_days = set(df.loc[df.date.dt.year.between(lower, upper), "date"])
                execution_days = set(raw_days[raw_days.dt.year.between(lower, upper)])
                if feature_days != execution_days:
                    raise ScaleError(f"{path.stem} v2 特征/执行日期不齐；禁止丢日或回退")
            keep = df.date.dt.strftime("%Y%m%d").isin(set(real.dates.tolist())).to_numpy()
            if not keep.all():
                df = df.loc[keep].reset_index(drop=True)
                features = features[keep]
        years = df.date.dt.year.to_numpy()
        mask = (years >= ranges["train"][0]) & (years <= ranges["train"][1])
        train = features[mask]
        if len(train):
            # 合并各股票 train 统计量，绝不串接股票计算指标/窗口。
            n = len(train)
            local_mean = train.mean(axis=0)
            delta = local_mean - mean
            m2 += ((train - local_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        prepared.append((path.stem, df, features, years))
        source = {"symbol": path.stem, "file": str(path.resolve()),
                  "sha256": feature_hash if ca_version else sha256_file(path),
                  "rows": len(df), "start": df.date.min().strftime("%Y%m%d") if len(df) else None,
                  "end": df.date.max().strftime("%Y%m%d") if len(df) else None}
        if ca_version:
            from stock_ai.data.corporate_actions import load_ledger
            event_path = Path(corporate_actions_dir) / f"{path.stem}.json"
            if event_path.resolve().parent != Path(corporate_actions_dir).resolve():
                raise ScaleError("公司行为账本路径超出受控缓存目录")
            ledger = load_ledger(event_path, symbol=path.stem)
            relevant = df.date[df.date.dt.year.between(ranges["train"][0], ranges["test"][1])]
            if len(relevant):
                ledger.require_coverage(relevant.min().strftime("%Y%m%d"), relevant.max().strftime("%Y%m%d"))
            real.corporate_actions = ledger
            ca_bars[path.stem] = real
            source.update(execution_file=str(execution_path.resolve()),
                          execution_sha256=execution_hash,
                          corporate_actions_file=ledger.source_path,
                          corporate_actions_sha256=ledger.sha256,
                          corporate_actions_schema=ledger.schema,
                          corporate_actions_coverage=dict(ledger.coverage))
        sources.append(source)
    if count < 2:
        raise ScaleError("train 年份内不足两行缓存，无法拟合全局 normalizer")
    std = np.sqrt(m2 / count)
    std[std < 1e-8] = 1.0
    norm = NormalizerState(mean, std, list(FEATURES))
    pools = {name: [] for name in ranges}
    strict_pools = {name: [] for name in ranges}
    split = {"ranges": ranges, "episodes": {name: [] for name in ranges}, "excluded": [],
             "normalizer_fit_split": "train", "normalizer_fit_rows": count,
             "execution_version": version,
             "warmup": "只使用同股票当日及更早历史；允许前一切分作为 warmup，不参与该切分动作"}
    for symbol, df, features, years in prepared:
        obs = observations(normalize(features, norm), window)
        dates = df.date.dt.strftime("%Y%m%d").to_numpy()
        # v2 禁止第二遍按路径重读 raw/账本，否则 episode 可能与 sources 对应不同版本。
        bars = ca_bars[symbol] if ca_version else (_load_execution_bars(execution_dir, symbol, df) if strict else None)
        for name, (start, end) in ranges.items():
            for year in range(start, end + 1):
                indices = np.flatnonzero(years == year)
                if len(indices) < 2:
                    if len(indices):
                        split["excluded"].append({"id": f"{symbol}_{year}", "split": name, "reason": "不足两行"})
                    continue
                first, stop = int(indices[0]), int(indices[-1]) + 1
                sample = SampleSplit(obs[first:stop], dates[first:stop], np.repeat(symbol, len(indices)),
                                     df.close.to_numpy()[first:stop], dates[first], dates[stop - 1])
                uid = f"{symbol}_{year}"
                pools[name].append(Episode(uid, sample))
                if bars is not None:
                    # 真实 K 线与特征同源同日，逐一截取，日期一一对应。
                    episode_bars = _slice_bars(bars, dates[first:stop], symbol)
                    if ca_version:
                        # Validate every episode now, not randomly after PPO has started.
                        from stock_ai.config.schema import EnvConfig
                        probe = make_strict_env(obs[first:stop], episode_bars, EnvConfig(),
                                                execution_version=version, _with_benchmark=False)
                        probe.close()
                    strict_pools[name].append((uid, obs[first:stop], episode_bars))
                split["episodes"][name].append({"id": uid, "symbol": symbol, "year": year,
                                                "rows": len(indices), "start": dates[first], "end": dates[stop - 1]})
    if not pools["train"]:
        raise ScaleError("没有至少两行的 train episode")
    if strict and not strict_pools["train"]:
        raise ScaleError("严格撮合模式下没有可用的 train episode（不复权缓存缺失或无法对齐）")
    split["strict_episodes"] = {k: len(v) for k, v in strict_pools.items()}
    return Dataset(pools, norm, split, sources, strict_pools, version)


def _slice_bars(bars, dates, symbol):
    """按日期子集裁剪 :class:`ExecutionBars`（保持与特征序列同长同序）。"""
    from stock_ai.env.strict_env import ExecutionBars

    index = {str(d): i for i, d in enumerate(bars.dates)}
    rows = [index[str(d)] for d in dates if str(d) in index]
    if len(rows) != len(dates):
        # build_dataset 已在特征阶段按真实成交日过滤，正常不会走到这里；
        # 保留失败是最后一道防线，绝不放行错位序列。
        raise ScaleError(
            f"{symbol} 真实 K 线与特征交易日不对齐"
            f"（K线 {len(bars.dates)} 日 / 特征 {len(dates)} 日，缺 "
            f"{len(dates) - len(rows)} 日）"
        )
    rows = np.asarray(rows, dtype="int64")
    return ExecutionBars(dates=np.asarray(dates), open=bars.open[rows], high=bars.high[rows],
                         low=bars.low[rows], close=bars.close[rows], volume=bars.volume[rows],
                         symbol=symbol, corporate_actions=bars.corporate_actions)


def make_strict_env(obs, bars, cfg, execution_version=EXECUTION_STRICT, **kwargs):
    """Never infer a model's execution version from the presence of a ledger."""
    if execution_version == EXECUTION_CORPORATE_ACTIONS:
        from stock_ai.env.corporate_action_env import CorporateActionExecutionEnv
        return CorporateActionExecutionEnv(obs, bars, cfg, **kwargs)
    if execution_version == EXECUTION_STRICT:
        from stock_ai.env.strict_env import StrictExecutionEnv
        return StrictExecutionEnv(obs, bars, cfg, **kwargs)
    raise ScaleError(f"不支持的严格撮合版本: {execution_version!r}")


class UnifiedTradingEnv(TradingEnv):
    """保留旧成交规则，但奖励按成交后下一日估值；旧 TradingEnv 不变。"""
    def _compute_reward(self, prev_value, value, turnover):
        return 0.0

    def step(self, action):
        previous = self.state.portfolio_value
        obs, _, terminated, truncated, info = super().step(action)
        value = self.state.mark_to_market(float(self.prices[self.idx]))
        self._equity[-1] = value
        reward = self.reward_calc.compute(previous, value, info["turnover"], self.state.position_ratio)
        info.update(portfolio_value=value, cash=self.state.cash, position_ratio=self.state.position_ratio)
        return obs, float(reward), terminated, truncated, info


class EpisodePoolEnv(gym.Env):
    def __init__(self, episodes, cfg, seed=42):
        super().__init__()
        self.episodes, self.cfg = episodes, cfg
        self.inner = UnifiedTradingEnv(episodes[0].sample, cfg, seed=seed)
        self.observation_space, self.action_space = self.inner.observation_space, self.inner.action_space
        self.np_random = np.random.default_rng(seed)
        self.episode_id = episodes[0].id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        episode = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        self.inner.close()
        self.inner = UnifiedTradingEnv(episode.sample, self.cfg)
        self.episode_id = episode.id
        obs, info = self.inner.reset(seed=seed)
        return obs, {**info, "episode_id": self.episode_id}

    def step(self, action):
        obs, reward, term, trunc, info = self.inner.step(action)
        return obs, reward, term, trunc, {**info, "episode_id": self.episode_id}

    def close(self):
        self.inner.close()


class StrictEpisodePoolEnv(gym.Env):
    """严格撮合 episode 池（``(id, obs, bars)`` 三元组），供 SB3 共享 PPO 使用。"""
    def __init__(self, episodes, cfg, seed=42, execution_version=EXECUTION_STRICT):
        super().__init__()
        self.execution_version = execution_version
        self.episodes, self.cfg = list(episodes), cfg
        self.inner = make_strict_env(episodes[0][1], episodes[0][2], cfg,
                                     execution_version=execution_version, seed=seed)
        self.observation_space, self.action_space = self.inner.observation_space, self.inner.action_space
        self.np_random = np.random.default_rng(seed)
        self.episode_id = episodes[0][0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        episode = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        self.inner.close()
        self.inner = make_strict_env(episode[1], episode[2], self.cfg,
                                     execution_version=self.execution_version)
        self.episode_id = episode[0]
        obs, info = self.inner.reset(seed=seed)
        return obs, {**info, "episode_id": self.episode_id}

    def step(self, action):
        obs, reward, term, trunc, info = self.inner.step(action)
        return obs, reward, term, trunc, {**info, "episode_id": self.episode_id}

    def close(self):
        self.inner.close()


def make_pool_env(episodes, cfg, seed=42, strict=False, strict_episodes=None,
                  execution_version=EXECUTION_STRICT):
    """按执行语义构造 episode 池环境（旧语义行为完全不变）。"""
    if strict:
        return StrictEpisodePoolEnv(strict_episodes or episodes, cfg, seed,
                                    execution_version=execution_version)
    return EpisodePoolEnv(episodes, cfg, seed)


def _execution_warning(version: str) -> str:
    """按执行语义给出 manifest ``warning`` 文案。"""
    if version == EXECUTION_CORPORATE_ACTIONS:
        return ("公司行为 v2：仅已核验覆盖账本；登记日持仓决定权益，除权日确认税前分红应收及"
                "送转待上市股，到账/上市单独结算，除权参考价计算涨跌停。个人差别化股息税未建模，"
                "未上市股份按同股 raw 收盘估值，碎股/配股/特殊方案拒绝。excess 基准为同环境恒买。"
                "ST/新股限制仍为近似，非实盘；续训不恢复环境/RNG精确轨迹。")
    if version == EXECUTION_STRICT:
        return (
            "旧 v1 未结算分红送转，不是可靠投资总回报；T 日收盘信号 -> T+1 日真实开盘价成交；涨跌停按板块规则（主板10%/"
            "创业板科创板20%/北交所30%/主板ST 5%近似）一字板拒单；特征用后复权价，"
            "成交用未复权真实价。新股上市首日无涨跌幅限制未建模；ST 名单未联网获取，"
            "按非 ST 处理。非实盘撮合。续训不恢复环境/RNG精确轨迹。"
        )
    return ("同日收盘模拟成交；复权/公司行为风险需核验；非实盘撮合。"
            "续训不恢复环境/RNG精确轨迹。")


def _reward_payload(env_cfg):
    """由 env 配置推导 ``(reward_type, reward_version, reward_params)``。

    - ``reward_type`` 归一化到小写；缺省视为 ``drawdown_penalty``（与
      :class:`~stock_ai.env.rewards.RewardCalculator` 一致）。
    - ``reward_version``：``excess`` 写 :data:`REWARD_VERSION_EXCESS`，
      其余写 :data:`REWARD_VERSION_LEGACY`（语义可区分）。
    - ``reward_params``：含四个 λ 与 ``reward_type``。

    Args:
        env_cfg: :class:`~stock_ai.config.schema.EnvConfig`。

    Returns:
        ``(reward_type, reward_version, reward_params)`` 三元组。
    """
    reward_type = str(getattr(env_cfg, "reward_type", None) or "drawdown_penalty").lower()
    reward_version = (
        REWARD_VERSION_EXCESS if reward_type == "excess" else REWARD_VERSION_LEGACY
    )
    reward_params = {
        "reward_type": reward_type,
        "lambda_drawdown": float(getattr(env_cfg, "lambda_drawdown", 0.0)),
        "lambda_turnover": float(getattr(env_cfg, "lambda_turnover", 0.0)),
        "lambda_leverage": float(getattr(env_cfg, "lambda_leverage", 0.0)),
    }
    return reward_type, reward_version, reward_params


def evaluate_strict_episode(trainer, episode, env_cfg, execution_version=EXECUTION_STRICT):
    """在严格撮合语义下回放单个 episode，返回收益/回撤/奖励/磨损指标。"""
    env = make_strict_env(episode[1], episode[2], env_cfg, execution_version=execution_version)
    obs, _ = env.reset()
    reward_sum = 0.0
    for _ in range(episode[2].n_steps - 1):
        action, _ = trainer.predict(obs)
        obs, reward, terminated, _, _ = env.step(action)
        reward_sum += reward
        if terminated:
            break
    equity = env.equity
    metrics = env.metrics()
    # 纯 close-to-close 累计基准收益（不含费用/滑点），用于 reward↔metric 精确对齐。
    closes = np.asarray(episode[2].close, dtype="float64")
    benchmark_return_cc = (
        float(closes[-1] / closes[0] - 1.0) if closes.size >= 2 and closes[0] > 1e-9 else 0.0
    )
    if execution_version == EXECUTION_CORPORATE_ACTIONS:
        # Do not label an executable, fee-inclusive CA baseline as raw close/close.
        benchmark_return_cc = None
    record = {
        "episode_id": episode[0],
        "total_return": float(equity[-1] / equity[0] - 1) if len(equity) else 0.0,
        "max_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity))) if len(equity) else 0.0,
        "reward": float(reward_sum),
        "trades": len(env.trades),
        # 零交易 episode 的超额恒等于 −benchmark_return：一年不交易，基准涨就记负、
        # 基准跌就记正。这是定义性假象而非技能，汇总必须单列，不能与有交易口径混算。
        "zero_trade": len(env.trades) == 0,
        "execution_version": execution_version,
        "benchmark_return_cc": benchmark_return_cc,
        "benchmark_definition": ("ca_gross_constant_buy_same_environment" if execution_version == EXECUTION_CORPORATE_ACTIONS
                                 else "legacy_raw_close_to_close"),
        "friction": {
            "trades": metrics["trades"],
            "total_notional": metrics["total_notional"],
            "turnover_ratio": metrics["total_turnover_ratio"],
            "cost_drag_bps": metrics["cost_drag_bps"],
            "avg_wear_bps": metrics["avg_wear_bps"],
            "strategy_return": metrics["strategy_return"],
            "benchmark_return": metrics["benchmark_return"],
            "excess_return": metrics["excess_return"],
        },
    }
    env.close()
    return record


class UnifiedRunner:
    def __init__(self, cfg):
        self.cfg, self.root = cfg, Path(cfg.paths.root)
        self.state = {}
        self.started = time.monotonic()

    def update(self, **kwargs):
        self.state.update(kwargs, updated_at=datetime.now().isoformat(), elapsed_sec=time.monotonic() - self.started)
        target = self.state.get("target_timesteps", 0)
        self.state["progress"] = min(1.0, self.state.get("timesteps", 0) / target) if target else 0.0
        atomic_write_json(self.state, status_path(self.root))
        if self.state.get("run_id"):
            atomic_write_json(self.state, unified_dir(self.root) / self.state["run_id"] / "status.json")

    def command(self):
        path = control_path(self.root)
        return json.loads(path.read_text(encoding="utf-8")).get("action", "") if path.exists() else ""

    def boundary(self, save=None):
        action = self.command()
        if action == "pause":
            previous_phase = self.state.get("process_state", "running")
            if save:
                save()
            self.update(process_state="paused")
            while self.command() == "pause":
                time.sleep(0.2)
            action = self.command()
            self.update(process_state=previous_phase)
        if action == "stop":
            return False
        return True

    def run(self, timesteps=None, run_id=None, resume=False, window=None, ranges=None,
            n_envs=1, checkpoint_steps=10000, execution_version=None, feature_dir=None,
            corporate_actions_dir=None):
        total = int(timesteps if timesteps is not None else self.cfg.train.timesteps)
        if total < 1 or not 1 <= n_envs <= 32 or checkpoint_steps < 1:
            raise ScaleError("timesteps/checkpoint_steps 必须为正数；n_envs 必须为 1~32")
        if resume and not run_id:
            raise ScaleError("续训必须指定 --run-id")
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        if not RUN_ID.fullmatch(run_id):
            raise ScaleError("run_id 仅允许字母、数字、下划线与连字符")
        directory = unified_dir(self.root) / run_id
        if directory.resolve().parent != unified_dir(self.root).resolve():
            raise ScaleError("非法 unified run 目录")
        if directory.exists() and not resume:
            raise ScaleError("run 已存在；请换 run_id 或显式 --resume")
        if resume and not (directory / "model.zip").is_file():
            raise ScaleError("续训 run 缺少 model.zip")
        version = execution_version or EXECUTION_LEGACY
        if version not in EXECUTION_VERSIONS:
            raise ScaleError(
                f"未知 execution_version: {version!r}（可选: {sorted(EXECUTION_VERSIONS)}）"
            )
        # Preflight before any state/model writes; an explicit v1->v2 resume is never ignored.
        old = json.loads((directory / "manifest.json").read_text(encoding="utf-8")) if resume else None
        if old:
            frozen_version = old.get("execution_version", EXECUTION_LEGACY)
            if execution_version is not None and execution_version != frozen_version:
                raise ScaleError("续训拒绝：execution_version 与冻结 manifest 不一致；新语义必须新 run")
            version = frozen_version
            if version not in EXECUTION_VERSIONS:
                raise ScaleError("续训拒绝：未知的冻结执行版本")
            window = window or old["window"]
            ranges = ranges or old["ranges"]
        strict = version in STRICT_EXECUTION_VERSIONS
        feature_value = feature_dir or (old or {}).get("feature_dir")
        features_dir = Path(feature_value) if feature_value else Path(self.cfg.data.cache_dir)
        ca_value = corporate_actions_dir or (old or {}).get("corporate_actions_dir")
        ca_dir = Path(ca_value) if ca_value else self.root / "data" / "corporate_actions"
        if version == EXECUTION_CORPORATE_ACTIONS and not ca_dir.is_dir():
            raise ScaleError(f"公司行为 v2 缺少已核验账本目录：{ca_dir}；禁止启动训练或回退 v1")
        lock = status_path(self.root).with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ScaleError("unified 已有任务/锁；请先检查运行进程，不会自动覆盖") from exc
        os.close(fd)
        trainer = None
        metrics = {"status": "preparing", "evaluation": {}}
        try:
            window = window or self.cfg.sample.window
            self.state = {"run_id": run_id, "process_state": "preparing", "timesteps": 0,
                          "target_timesteps": total, "pid": os.getpid(), "error": "",
                          "execution_version": version}
            directory.mkdir(parents=True, exist_ok=True)
            write_control(self.root, "resume")
            self.update()
            def progress(i, n, symbol):
                self.update(cache_processed=i, cache_total=n, current_symbol=symbol)
                if not self.boundary():
                    raise InterruptedError("准备阶段已停止")
            data = build_dataset(features_dir, window, ranges, progress,
                                 execution_dir=self.cfg.data.cache_dir if strict else None,
                                 execution_version=version,
                                 corporate_actions_dir=ca_dir if version == EXECUTION_CORPORATE_ACTIONS else None)
            self.update(cache_processed=len(data.sources), current_symbol="")
            reward_type, reward_version, reward_params = _reward_payload(self.cfg.env)
            if version == EXECUTION_CORPORATE_ACTIONS and reward_type == "excess":
                from stock_ai.env.corporate_action_env import REWARD_VERSION_EXCESS as ca_reward_version
                reward_version = ca_reward_version
            manifest = {"run_id": run_id, "mode": "unified", "contract": CONTRACT, "backend": "sb3",
                        "algo": "ppo", "seed": self.cfg.random_seed, "window": window,
                        "features": FEATURES, "observation_shape": [window, len(FEATURES)],
                        "actions": {"0": "hold", "1": "buy", "2": "sell"},
                        "normalizer_clip": 10, "ranges": data.split["ranges"], "sources": data.sources,
                        "env_config": dataclasses.asdict(self.cfg.env), "train_config": dataclasses.asdict(self.cfg.train),
                        "episode_sampling": "uniform_stock_year", "created_at": datetime.now().isoformat(),
                        "execution_version": version,
                        "reward_type": reward_type,
                        "reward_version": reward_version,
                        "reward_params": reward_params,
                        "feature_dir": str(features_dir.resolve()),
                        "execution_dir": str(Path(self.cfg.data.cache_dir).resolve()) if strict else "",
                        "feature_adjust": "hfq" if strict else "raw(as-cached)",
                        "execution_adjust": "none(真实未复权价)" if strict else "n/a",
                        "warning": _execution_warning(version)}
            if version == EXECUTION_CORPORATE_ACTIONS:
                manifest.update(corporate_actions_dir=str(ca_dir.resolve()),
                                corporate_actions_schema="corporate-actions-v1",
                                tax_policy="gross_before_personal_dividend_tax",
                                benchmark_definition="ca_gross_constant_buy_same_environment")
            if old:
                if version == EXECUTION_CORPORATE_ACTIONS:
                    for key in ("corporate_actions_dir", "corporate_actions_schema", "tax_policy",
                                "benchmark_definition", "reward_version", "feature_dir", "execution_dir"):
                        if old.get(key) != manifest.get(key):
                            raise ScaleError(f"续训拒绝：{key} 与冻结 manifest 不一致")
                for key in ("contract", "window", "ranges", "features", "sources", "env_config", "seed"):
                    if old[key] != manifest[key]:
                        raise ScaleError(f"续训拒绝：{key} 与冻结 manifest 不一致")
                if old.get("execution_version", EXECUTION_LEGACY) != version:
                    raise ScaleError("续训拒绝：execution_version 与冻结 manifest 不一致")
                # 奖励类型纳入一致性校验：旧 manifest 无该字段时按 drawdown_penalty 处理
                # （兼容 legacy / 上一个 strict run 的续训），不一致时明确报错而非静默继承。
                frozen_reward = str(old.get("reward_type", "drawdown_penalty")).lower()
                if frozen_reward != reward_type:
                    raise ScaleError(
                        "续训拒绝：reward_type 与冻结 manifest 不一致"
                        f"（冻结 {frozen_reward!r} / 当前 {reward_type!r}）；"
                        "请在续训时显式传入相同的 --reward 与 --lambda-* 参数"
                    )
                saved_norm = json.loads((directory / "normalizer.json").read_text(encoding="utf-8"))
                if saved_norm != data.normalizer.to_dict():
                    raise ScaleError("续训拒绝：冻结 normalizer 不一致")
            else:
                atomic_write_json(manifest, directory / "manifest.json")
                atomic_write_json(data.normalizer.to_dict(), directory / "normalizer.json")
                atomic_write_json(data.split, directory / "split.json")
            metrics = {"status": "running", "evaluation": {},
                       "episode_counts": {k: len(v) for k, v in data.pools.items()},
                       "execution_version": version,
                       "reward_type": reward_type,
                       "reward_version": reward_version,
                       "reward_params": reward_params}
            atomic_write_json(metrics, directory / "metrics.json")
            from stock_ai.train.sb3_trainer import make_unified_trainer
            trainer = make_unified_trainer(data.pools["train"], self.cfg, n_envs, total,
                                           directory / "model.zip" if resume else None,
                                           strict=strict, strict_episodes=data.strict_pools["train"],
                                           execution_version=version)
            self.update(process_state="running", timesteps=int(trainer.model.num_timesteps),
                        episode_counts=metrics["episode_counts"], observation_shape=manifest["observation_shape"])
            def save():
                temporary = directory / "checkpoint_pending.zip"
                trainer.save(temporary)
                os.replace(temporary, directory / "model.zip")
                self.update(checkpoint_timesteps=int(trainer.model.num_timesteps))
            last_save = int(trainer.model.num_timesteps)
            def tick(model):
                nonlocal last_save
                steps = int(model.num_timesteps)
                self.state["timesteps"] = steps
                if steps - last_save >= checkpoint_steps:
                    save()
                    last_save = steps
                if steps % 32 < n_envs:
                    training_metrics = {k: float(v) for k, v in model.logger.name_to_value.items()
                                        if k.startswith("train/") and np.isscalar(v) and np.isfinite(v)}
                    self.update(training_metrics=training_metrics)
                return self.boundary(save)
            from stock_ai.train.sb3_trainer import learn_unified
            remaining = max(0, total - int(trainer.model.num_timesteps))
            if remaining:
                learn_unified(trainer, remaining, tick)
            save()
            stopped = self.command() == "stop"
            if not stopped:
                self.update(process_state="evaluating")
                for name in ("val", "test"):
                    records = []
                    source_episodes = data.strict_pools[name] if strict else data.pools[name]
                    for position, episode in enumerate(source_episodes):
                        episode_id = episode[0] if strict else episode.id
                        self.update(process_state="evaluating", evaluation_split=name,
                                    evaluation_episode=episode_id, evaluation_done=len(records),
                                    evaluation_total=len(source_episodes))
                        if not self.boundary(save):
                            stopped = True
                            break
                        if strict:
                            records.append(evaluate_strict_episode(trainer, episode, self.cfg.env, execution_version=version))
                            if stopped:
                                break
                            continue
                        env = UnifiedTradingEnv(episode.sample, self.cfg.env)
                        obs, _ = env.reset()
                        reward_sum = 0.0
                        for _ in range(episode.sample.n_steps - 1):
                            if not self.boundary(save):
                                stopped = True
                                break
                            action, _ = trainer.predict(obs)
                            obs, reward, _, _, _ = env.step(action)
                            reward_sum += reward
                        if not stopped:
                            equity = env.equity
                            records.append({"episode_id": episode.id, "total_return": equity[-1] / equity[0] - 1,
                                            "max_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity))),
                                            "reward": reward_sum, "trades": len(env.trades)})
                        env.close()
                        if stopped:
                            break
                    summary = {"episodes": records,
                               "mean_return": float(np.mean([r["total_return"] for r in records])) if records else None,
                               "complete": not stopped, "used_for_training": False}
                    if strict and records:
                        # 交易磨损汇总（总换手 / 成本拖累 / 平均单笔磨损 / 对基准）
                        traded = [r for r in records if not r.get("zero_trade", False)]
                        zero = [r for r in records if r.get("zero_trade", False)]
                        excess_all = [r["friction"]["excess_return"] for r in records
                                      if r["friction"]["excess_return"] is not None]
                        excess_traded = [r["friction"]["excess_return"] for r in traded
                                         if r["friction"]["excess_return"] is not None]
                        excess_zero = [r["friction"]["excess_return"] for r in zero
                                       if r["friction"]["excess_return"] is not None]
                        summary["friction"] = {
                            "episodes": len(records),
                            "total_trades": sum(int(r["friction"]["trades"]) for r in records),
                            "total_notional": sum(float(r["friction"]["total_notional"]) for r in records),
                            "mean_cost_drag_bps": float(np.mean([r["friction"]["cost_drag_bps"] for r in records])),
                            "mean_avg_wear_bps": float(np.mean([r["friction"]["avg_wear_bps"] for r in records])),
                            "mean_strategy_return": float(np.mean([r["friction"]["strategy_return"] for r in records])),
                            "mean_benchmark_return": float(np.mean([r["friction"]["benchmark_return"] for r in records
                                                                      if r["friction"]["benchmark_return"] is not None])),
                            "mean_excess_return": float(np.mean(excess_all)) if excess_all else None,
                            "mean_benchmark_return_cc": float(np.mean(
                                [r["benchmark_return_cc"] for r in records
                                 if r.get("benchmark_return_cc") is not None])
                                ) if any(r.get("benchmark_return_cc") is not None for r in records) else None,
                            # 零交易单列：零交易 episode 的「超额」恒等于 −benchmark_return，
                            # 是定义性假象。诚实口径是 mean_excess_return_traded。
                            "zero_trade_episodes": len(zero),
                            "zero_trade_fraction": len(zero) / len(records),
                            "mean_excess_return_traded": float(np.mean(excess_traded)) if excess_traded else None,
                            "mean_excess_return_zero_trade": float(np.mean(excess_zero)) if excess_zero else None,
                            # 退化守卫：零交易占比超过 5% 即标记，避免「全部超额」被
                            # 躺平 episode 抬起来（v2 曾出现 val +2.14% 全是有交易 −4.22%
                            # 与零交易 +7.05% 混合出来的假象）。
                            "degenerate": bool(len(zero) / len(records) > DEGENERATE_ZERO_TRADE_FRACTION),
                            "degenerate_reason": (
                                f"零交易 episode 占比 {len(zero) / len(records):.1%} 超过 "
                                f"{DEGENERATE_ZERO_TRADE_FRACTION:.0%}；mean_excess_return 混合了"
                                f"定义性假象，请以 mean_excess_return_traded 为准"
                                if len(zero) / len(records) > DEGENERATE_ZERO_TRADE_FRACTION else None
                            ),
                        }
                    metrics["evaluation"][name] = summary
                    atomic_write_json(metrics, directory / "metrics.json")
                    if stopped:
                        break
            metrics.update(status="stopped" if stopped else "done", timesteps=int(trainer.model.num_timesteps))
            atomic_write_json(metrics, directory / "metrics.json")
            self.update(process_state=metrics["status"], timesteps=metrics["timesteps"], metrics=metrics)
            return directory
        except (KeyboardInterrupt, InterruptedError):
            if trainer:
                trainer.save(directory / "model.zip")
            metrics.update(status="stopped", timesteps=int(trainer.model.num_timesteps) if trainer else 0)
            atomic_write_json(metrics, directory / "metrics.json")
            self.update(process_state="stopped", timesteps=metrics["timesteps"])
            return directory
        except Exception as exc:
            metrics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_write_json(metrics, directory / "metrics.json")
            self.update(process_state="failed", error=metrics["error"])
            if isinstance(exc, ScaleError):
                raise
            raise ScaleError(f"unified 训练失败：{exc}") from exc
        finally:
            if trainer:
                trainer.env.close()
            lock.unlink(missing_ok=True)

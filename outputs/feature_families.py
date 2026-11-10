"""逐族增量特征构造器（T01 · 只读 · 零落盘 · 零写回 · 零缓存）。

本模块是 P0 验证阶段「落点 1」的实现：在内存里由
``data/raw_hfq_v2_train``（197 只，含**真实** ``amount``）与 ``data/raw``
（未复权 OHLC）按 ``%Y%m%d`` 8 位日期键现场构造逐族特征。
**不改 src/、不碰 tests/、不写 data/、不新增任何派生 parquet/缓存文件。**

契约要点（来源：outputs/feature_data_plan.md §2.2/§2.3/§3.3/§3.4/§3.5/§10）
  * 基线 9 特征名冻结，以 ``stock_ai.scale.unified.FEATURES`` 为唯一权威，
    并由 ``unified.relative_features`` **逐位产出**，绝不手写复刻公式。
  * 一次 feature_set 只允许 ``("base",)`` 或 ``("base", 单一族)``。
  * 只产出无量纲相对量；绝对价格 / 绝对金额 / 绝对成交量禁止直接进观测。
  * 缩放自洽：价格 ×s、``volume`` ×s、``amount`` ×s²，
    且 ``frame`` 与 ``raw_frame`` 同步缩放 → 所有族输出 ``atol=1e-12`` 不变。
  * 因果：单只股票序列内一次性 rolling，只用 ``[t-W+1, t]``；
    绝不串接股票；无 ``bfill``、无 ``shift(-)`` 进特征；min_periods 兜底；
    窗口内 fillna(0)；末尾 ``np.isfinite`` 全量校验。
  * 金额分子锁定 hfq 目录真实 ``amount``；价格分母锁定 ``data/raw`` 未复权 OHLC。
    **禁用 data/raw 的 amount、禁用 close_hfq 作金额分母。**

明确禁止（不得出现在注册表或任何构造代码中）：
    市场指数独立列 / 市场 beta 残差 / regime 门控 / 联合量 / 横截面选股 /
    调参重训 / 任何未来日历信息 / data/raw 的 amount / close_hfq 金额分母。

预留：``inst_`` / ``cal_`` / ``ca_``（A 股制度 / 日历 / 公司行为，P1）仅前缀
注释、本轮不实现；``account`` 为运行时注入族，仅注册列名、本轮不构造
（TODO(T04)：独立排期，观测来自 env 运行时而非 frame）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs"))

from stock_ai.scale.unified import FEATURES as _UNIFIED_FEATURES  # noqa: E402
from stock_ai.scale.unified import relative_features as _unified_relative_features  # noqa: E402

__all__ = [
    "ROOT", "RAW_HFQ_V2_TRAIN", "RAW", "EPS", "DAY_KEY_FORMAT",
    "BASE_FEATURES", "FEATURE_GROUPS", "RUNTIME_ONLY_GROUPS",
    "CONTRACT_V2", "PreconditionReport", "PreconditionError",
    "load_config", "active_features", "feature_set_id", "contract_id",
    "validate_single_variable", "align_frames_by_day_key",
    "compute_precondition", "assert_amount_precondition",
    "build_features", "build_family_frame", "self_test",
]

# --------------------------------------------------------------------------- #
# 常量与冻结契约
# --------------------------------------------------------------------------- #
RAW_HFQ_V2_TRAIN: Path = ROOT / "data" / "raw_hfq_v2_train"
RAW: Path = ROOT / "data" / "raw"
CONFIG_PATH: Path = ROOT / "outputs" / "feature_experiments.yaml"

#: 数值下限（§3.4 ε=1e-12）。注意：零分母兜底一律用 ``where=`` 判定 + 默认值，
#: 绝不用「分母 + ε」——绝对 ε 在整体缩放下会破坏 atol=1e-12 不变性。
EPS: float = 1e-12
#: 统一 8 位日期键；严禁 20060104 与 2006-01-04 混用（历史出过 merge bug，§10）
DAY_KEY_FORMAT: str = "%Y%m%d"

CONTRACT_V2: str = "unified-relative-v2"

#: 基线 9 特征名冻结，**从 unified.FEATURES 导入为权威，不手写复刻**（§2.1/§10）
BASE_FEATURES: tuple[str, ...] = tuple(_UNIFIED_FEATURES)

#: 逐族注册表：列名 / 列数 / 列序严格按 plan §3.3 逐列实现。
#: 前缀 inst_ / cal_ / ca_ 属 P1，仅在此预留注释，不注册、不实现。
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "base": BASE_FEATURES,
    # P0-1 成交额 / 流动性（6 列）
    "liquidity": (
        "liq_amihud_z20",        # zscore_20(|ret_1| / amount_hfq)
        "liq_amt_rank250",       # pct_rank_250(amount_hfq)
        "liq_amt_logratio",      # log(amount / rolling_median_250(amount))
        "liq_vwap_dev",          # close_raw / (amount/volume) - 1（价格分母必须未复权）
        "liq_vwap_dev_ma5",      # rolling_mean_5(liq_vwap_dev)
        "liq_pv_div20",          # rolling_corr(ret_1, Δlog(amount), 20)
    ),
    # P0-2 隔夜 / 日内收益分解（8 列，全 hfq）
    "overnight": (
        "od_overnight",          # open_t / close_{t-1} - 1
        "od_intraday",           # close_t / open_t - 1
        "od_overnight_ma5",
        "od_intraday_ma5",
        "od_overnight_vol20",
        "od_intraday_vol20",
        "od_abs_diff",           # |overnight| - |intraday|
        "od_overnight_share20",  # Σ|o| / (Σ|o| + Σ|i|) over 20
    ),
    # P0-3 K 线形态（9 列，全 hfq）
    "candlestick": (
        "cdl_upper_shadow",
        "cdl_lower_shadow",
        "cdl_close_location",
        "cdl_body",
        "cdl_bar_range",
        "cdl_upper_shadow_ma5",
        "cdl_lower_shadow_ma5",
        "cdl_close_location_ma5",
        "cdl_oneword",
    ),
    # P0-4 波动率结构（5 列，全 hfq，窗口 20）
    "volatility": (
        "vol_parkinson20",
        "vol_gk20",
        "vol_downside_ratio20",
        "vol_gap_freq20",
        "vol_path_ratio20",
    ),
    # P0-5 账户状态（6 列）：**仅注册列名**，运行时由 env state 注入，
    # 非落盘族、本轮不构造（TODO(T04) 独立排期；plan §3.3 / PRD §2.5）。
    "account": (
        "acct_position_flag",
        "acct_unrealized_pnl",
        "acct_holding_days_norm",
        "acct_cash_ratio",
        "acct_position_ratio",
        "acct_blocked_buy_5",
    ),
}

#: 运行时注入族：不参与落盘特征构造（本轮拒绝构造，明确报错）
RUNTIME_ONLY_GROUPS: tuple[str, ...] = ("account",)


def load_config(path: Path | str | None = None) -> dict:
    """读取冻结配置。每次调用都重新读盘（**零缓存**），返回解析后的字典。

    Args:
        path: 配置文件路径；默认 ``outputs/feature_experiments.yaml``。

    Returns:
        冻结参数字典。
    """
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"冻结配置格式非法: {cfg_path}")
    return cfg


def _normalize_groups(groups: Iterable[str] | str | None) -> tuple[str, ...]:
    """把 groups 规范成元组；``None`` → ``("base",)``，字符串 → 单元素元组。"""
    if groups is None:
        return ("base",)
    if isinstance(groups, str):
        return (groups,)
    return tuple(str(g) for g in groups)


def active_features(groups: Iterable[str] | str | None) -> list[str]:
    """按注册表声明序拼接特征名并去重；``("base",)`` 严格等于 ``BASE_FEATURES``。

    Args:
        groups: 特征族序列，例如 ``("base", "liquidity")``。

    Returns:
        有序、无重复的特征列名列表。
    """
    result: list[str] = []
    seen: set[str] = set()
    for group in _normalize_groups(groups):
        if group not in FEATURE_GROUPS:
            raise KeyError(f"未知特征族 {group!r}；允许: {sorted(FEATURE_GROUPS)}")
        for name in FEATURE_GROUPS[group]:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def validate_single_variable(groups: Iterable[str] | str | None) -> bool:
    """强制单变量原则：只允许 ``("base",)`` 或 ``("base", 单一族)``。

    Args:
        groups: 特征族序列。

    Returns:
        ``True``（校验通过）。

    Raises:
        ValueError: 缺 base / 未知族 / 重复族 / 出现 ≥2 个新族。
    """
    normalized = _normalize_groups(groups)
    if not normalized:
        raise ValueError("单变量校验失败：groups 不能为空，至少需含 'base'")
    unknown = [g for g in normalized if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"单变量校验失败：未知特征族 {unknown}；允许 {sorted(FEATURE_GROUPS)}")
    if "base" not in normalized:
        raise ValueError("单变量校验失败：必须包含 'base'（现状 9 特征）")
    new_groups = [g for g in normalized if g != "base"]
    if len(set(new_groups)) != len(new_groups):
        raise ValueError(f"单变量校验失败：特征族重复 {new_groups}")
    if len(new_groups) >= 2:
        raise ValueError(
            "单变量校验失败：只允许 groups=('base',) 或 ('base', 单一族)，"
            f"检测到 ≥2 个新族 {new_groups}；族间必须完全隔离（plan §6/§7.4）"
        )
    return True


def feature_set_id(groups: Iterable[str] | str | None) -> str:
    """确定性生成细粒度契约标识：base → ``v2-base``，单族 → ``v2-<族>``。"""
    normalized = _normalize_groups(groups)
    validate_single_variable(normalized)
    new_groups = [g for g in normalized if g != "base"]
    return f"v2-{new_groups[0] if new_groups else 'base'}"


def contract_id(groups: Iterable[str] | str | None) -> str:
    """续训/回放比对键：``unified-relative-v2+{feature_set_id}``（任一变化 → 新 run）。"""
    return f"{CONTRACT_V2}+{feature_set_id(groups)}"


# --------------------------------------------------------------------------- #
# 数据前提断言（§3.5 / PRD v2 §2.1⑦）
# --------------------------------------------------------------------------- #
@dataclass
class PreconditionReport:
    """三条数据前提断言的实测数值、逐条结论与使用的 amount 目录。"""

    std_amount_ratio: float                 # ① std(amount/(volume*close_raw))
    vwap_in_range_ratio: float             # ② 判定用（含价格精度容差）
    vwap_in_range_ratio_strict: float      # ② 严格 [low_raw, high_raw] 口径（诊断）
    vwap_ne_close_ratio: float             # ③a 占比(vwap != close_raw)
    vwap_dev_std: float                    # ③b std(close_raw/vwap - 1)
    corr_with_volume_relative: float       # ③c corr(vwap_dev, volume_relative)
    passed: bool
    amount_dir: str
    failed_checks: tuple[str, ...] = ()
    checks: dict = field(default_factory=dict)
    n_rows: int = 0

    def to_dict(self) -> dict:
        """转成可写入 manifest / metrics.json 的字典（本轮 T01 不写盘）。"""
        return {
            "std_amount_ratio": self.std_amount_ratio,
            "vwap_in_range_ratio": self.vwap_in_range_ratio,
            "vwap_in_range_ratio_strict": self.vwap_in_range_ratio_strict,
            "vwap_ne_close_ratio": self.vwap_ne_close_ratio,
            "vwap_dev_std": self.vwap_dev_std,
            "corr_with_volume_relative": self.corr_with_volume_relative,
            "passed": self.passed,
            "amount_dir": self.amount_dir,
            "failed_checks": list(self.failed_checks),
            "checks": self.checks,
            "n_rows": self.n_rows,
        }


class PreconditionError(Exception):
    """数据前提断言失败：判定该 amount 目录不可用、该族**直接止损**，不进回测。"""

    def __init__(self, report: PreconditionReport):
        self.report = report
        detail = (
            f"① std(amount/(volume×close_raw))={report.std_amount_ratio:.8f} "
            f"② vwap∈[low,high] 占比={report.vwap_in_range_ratio:.8f} "
            f"③ vwap≠close 占比={report.vwap_ne_close_ratio:.8f} / "
            f"std(vwap_dev)={report.vwap_dev_std:.8f} / "
            f"corr={report.corr_with_volume_relative:.8f}"
        )
        super().__init__(
            f"数据前提断言未通过（amount 目录: {report.amount_dir or '未声明'}）；"
            f"触发条款: {'、'.join(report.failed_checks) or '未知'}；实测 {detail}。"
            "判定该 amount 目录不可用、该特征族不可构造，直接止损，不进入回测；"
            "绝不降级填充常数（plan §3.5 / PRD §4.4 第 8 条）。"
        )


def _safe_divide(numer: np.ndarray, denom: np.ndarray, *,
                 default: float = 0.0) -> np.ndarray:
    """零分母兜底除法：分母非有限或 ≤0 处返回 ``default``，绝不产生 NaN/Inf。

    使用 ``where=`` 而非「分母 + ε」：绝对 ε 会随整体缩放改变相对权重，
    从而破坏 §3.4 要求的 ``atol=1e-12`` 缩放不变性。
    """
    num = np.asarray(numer, dtype="float64")
    den = np.asarray(denom, dtype="float64")
    out = np.full_like(num, float(default))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ok = np.isfinite(den) & (den > 0.0)
        np.divide(num, den, out=out, where=ok)
    return out


def _finite(arr: np.ndarray | pd.Series, *, default: float = 0.0) -> np.ndarray:
    """把 NaN/±Inf 统一替换为默认值，保证末尾 ``np.isfinite`` 全量通过。"""
    values = np.asarray(arr, dtype="float64")
    return np.nan_to_num(values, nan=float(default), posinf=float(default),
                         neginf=float(default))


def align_frames_by_day_key(frame: pd.DataFrame,
                            raw_frame: pd.DataFrame) -> pd.DataFrame:
    """按 ``%Y%m%d`` 8 位字符串日期键 inner-merge hfq 与未复权帧（fail-closed）。

    统一用 ``pd.to_datetime(...).dt.strftime("%Y%m%d")`` 生成键，
    **严禁** ``20060104`` 与 ``2006-01-04`` 混用（plan §1.6 / §10）。
    ``raw_frame`` 的 OHLC 改名为 ``*_raw``；``amount`` / ``volume`` 只取 ``frame``
    （hfq 真实成交额）；``data/raw`` 的 ``amount`` 在进入本函数前即被丢弃。

    Args:
        frame: hfq 帧，需含 ``date, open, high, low, close, volume, amount``。
        raw_frame: 未复权帧，需含 ``date, open, high, low, close``。

    Returns:
        对齐后按 ``date`` 升序、``reset_index`` 的 DataFrame。

    Raises:
        ValueError: 日期键重复 / 键不唯一 / 对齐后为空 / 缺列。
    """
    hfq = frame.copy()
    raw = raw_frame.copy()
    hfq_need = {"date", "open", "high", "low", "close", "volume", "amount"}
    raw_need = {"date", "open", "high", "low", "close"}
    if not hfq_need.issubset(hfq.columns):
        raise ValueError(f"hfq 帧缺列: {sorted(hfq_need - set(hfq.columns))}")
    if not raw_need.issubset(raw.columns):
        raise ValueError(f"raw 帧缺列: {sorted(raw_need - set(raw.columns))}")

    # 唯一的日期归一入口：datetime / 8 位整数 / ISO 字符串 一律 → %Y%m%d
    hfq["_day_key"] = pd.to_datetime(hfq["date"].astype(str),
                                     errors="raise").dt.strftime(DAY_KEY_FORMAT)
    raw["_day_key"] = pd.to_datetime(raw["date"].astype(str),
                                     errors="raise").dt.strftime(DAY_KEY_FORMAT)
    if hfq["_day_key"].duplicated().any():
        raise ValueError("hfq 帧存在重复 %Y%m%d 日期键，无法 one_to_one 对齐")
    if raw["_day_key"].duplicated().any():
        raise ValueError("raw 帧存在重复 %Y%m%d 日期键，无法 one_to_one 对齐")

    raw_prices = raw[["_day_key", "open", "high", "low", "close"]].rename(
        columns={"open": "open_raw", "high": "high_raw",
                 "low": "low_raw", "close": "close_raw"})
    merged = hfq.merge(raw_prices, on="_day_key", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("按 %Y%m%d 对齐后为空；fail-closed，绝不 bfill / 回退 / 静默丢日")
    merged["date"] = pd.to_datetime(merged["date"].astype(str), errors="raise")
    return merged.sort_values("date").reset_index(drop=True)


def compute_precondition(aligned: pd.DataFrame, *, amount_dir: str = "",
                         cfg: dict | None = None) -> PreconditionReport:
    """对**已按日期键对齐**的帧计算三条断言的实测数值（不抛异常，供 CLI 打印）。

    输入可以是单只股票，也可以是整个目录聚合后的多只帧（含 ``symbol`` 列）：
    ``volume_relative`` 在有 ``symbol`` 列时逐股票 rolling（绝不串接股票）。

    Args:
        aligned: :func:`align_frames_by_day_key` 的输出（或同样列集合的聚合帧）。
        amount_dir: 实际使用的 amount 目录名，写入报告。
        cfg: 冻结配置；``None`` 时读取 yaml。

    Returns:
        :class:`PreconditionReport`（含实测值、逐条 pass/fail 与 ``passed``）。
    """
    config = cfg if cfg is not None else load_config()
    thresholds = config["assertions"]
    std_min = float(thresholds["std_min"])
    in_range_min = float(thresholds["vwap_in_range_min"])
    tol = float(thresholds["vwap_range_tolerance"])
    dev_std_min = float(thresholds["vwap_dev_std_min"])
    ne_close_min = float(thresholds["vwap_ne_close_ratio_min"])
    collinearity_max = float(thresholds["collinearity_max"])

    amount = np.asarray(aligned["amount"], dtype="float64")
    volume = np.asarray(aligned["volume"], dtype="float64")
    close_raw = np.asarray(aligned["close_raw"], dtype="float64")
    low_raw = np.asarray(aligned["low_raw"], dtype="float64")
    high_raw = np.asarray(aligned["high_raw"], dtype="float64")
    if amount.size == 0:
        raise ValueError("对齐帧为空，无法计算数据前提断言")

    # ① std( amount / (volume × close_raw) ) —— 挡合成值（data/raw 下恒为 1 → std=0）
    ratio = _safe_divide(amount, volume * close_raw, default=1.0)
    std_amount_ratio = float(np.std(ratio))

    # vwap = amount_hfq / volume（真实未复权成交均价，元/股）
    vwap = _safe_divide(amount, volume, default=0.0)

    # ② 占比(vwap ∈ [low_raw, high_raw])。相对容差随价格同步缩放，保持口径不变。
    in_range = (vwap >= low_raw * (1.0 - tol)) & (vwap <= high_raw * (1.0 + tol))
    in_range_strict = ((vwap >= low_raw * (1.0 - EPS)) &
                       (vwap <= high_raw * (1.0 + EPS)))
    vwap_in_range_ratio = float(np.mean(in_range))
    vwap_in_range_ratio_strict = float(np.mean(in_range_strict))

    # ③ vwap_dev = close_raw / vwap - 1（分子未复权收盘，分母真实 vwap）
    vwap_dev = _safe_divide(close_raw, vwap, default=1.0) - 1.0
    # 「vwap != close_raw」用相对判据，保证缩放下结论完全一致
    rel_gap = np.abs(_safe_divide(vwap, close_raw, default=1.0) - 1.0)
    vwap_ne_close_ratio = float(np.mean(rel_gap > 1e-9))
    vwap_dev_std = float(np.std(vwap_dev))

    # volume_relative 与 unified.py:103 同口径（逐股票滚动，绝不串接）
    vol_series = pd.Series(volume)
    if "symbol" in aligned.columns:
        vol_rel = aligned.groupby("symbol", sort=False)["volume"].transform(
            lambda s: s / s.rolling(20, min_periods=1).mean().replace(0, np.nan) - 1.0
        )
        vol_rel_arr = _finite(np.asarray(vol_rel, dtype="float64"))
    else:
        vol_mean = vol_series.rolling(20, min_periods=1).mean().replace(0, np.nan)
        vol_rel_arr = _finite((vol_series / vol_mean - 1.0).to_numpy(dtype="float64"))

    if np.std(vol_rel_arr) <= 0.0 or np.std(vwap_dev) <= 0.0:
        # 常数序列 → 视为完全共线 / 不可用，判失败
        corr = 1.0
    else:
        corr = float(np.corrcoef(vwap_dev, vol_rel_arr)[0, 1])
        if not np.isfinite(corr):
            corr = 1.0

    checks: dict = {
        "①": {
            "name": "std(amount/(volume×close_raw)) > std_min",
            "value": std_amount_ratio, "threshold": std_min,
            "pass": bool(std_amount_ratio > std_min),
        },
        "②": {
            "name": "占比(vwap ∈ [low_raw, high_raw]) > vwap_in_range_min",
            "value": vwap_in_range_ratio, "strict_value": vwap_in_range_ratio_strict,
            "threshold": in_range_min, "tolerance": tol,
            "pass": bool(vwap_in_range_ratio > in_range_min),
        },
        "③a": {
            "name": "占比(vwap != close_raw) > vwap_ne_close_ratio_min",
            "value": vwap_ne_close_ratio, "threshold": ne_close_min,
            "pass": bool(vwap_ne_close_ratio > ne_close_min),
        },
        "③b": {
            "name": "std(vwap_dev) > vwap_dev_std_min",
            "value": vwap_dev_std, "threshold": dev_std_min,
            "pass": bool(vwap_dev_std > dev_std_min),
        },
        "③c": {
            "name": "|corr(vwap_dev, volume_relative)| < collinearity_max",
            "value": corr, "threshold": collinearity_max,
            "pass": bool(abs(corr) < collinearity_max),
        },
    }
    # 三条断言：① / ② / ③(③a ∧ ③b ∧ ③c)
    group_a3 = all(checks[k]["pass"] for k in ("③a", "③b", "③c"))
    failed: list[str] = []
    if not checks["①"]["pass"]:
        failed.append("①")
    if not checks["②"]["pass"]:
        failed.append("②")
    if not group_a3:
        failed.append("③")

    return PreconditionReport(
        std_amount_ratio=std_amount_ratio,
        vwap_in_range_ratio=vwap_in_range_ratio,
        vwap_in_range_ratio_strict=vwap_in_range_ratio_strict,
        vwap_ne_close_ratio=vwap_ne_close_ratio,
        vwap_dev_std=vwap_dev_std,
        corr_with_volume_relative=corr,
        passed=not failed,
        amount_dir=str(amount_dir),
        failed_checks=tuple(failed),
        checks=checks,
        n_rows=int(amount.size),
    )


def assert_amount_precondition(frame: pd.DataFrame, raw_frame: pd.DataFrame, *,
                               amount_dir: str | None = None) -> PreconditionReport:
    """三条数据前提断言（§3.5），**任一不满足即抛 :class:`PreconditionError`**。

    这是 P0-1 落地前第一道 gate：失败判定该 amount 目录不可用、该特征族
    不可构造、**直接止损，不进入回测**，绝不降级填充常数。

    Args:
        frame: hfq 帧（真实 ``amount``）。
        raw_frame: 未复权帧（``close_raw`` / ``high_raw`` / ``low_raw``）。
        amount_dir: 目录名；未传时从 ``frame.attrs["amount_dir"]`` 推断。

    Returns:
        全通过时的 :class:`PreconditionReport`（含实测数值）。

    Raises:
        PreconditionError: 任一断言未通过。
    """
    aligned = align_frames_by_day_key(frame, raw_frame)
    directory = amount_dir if amount_dir is not None else str(frame.attrs.get("amount_dir", ""))
    report = compute_precondition(aligned, amount_dir=directory)
    if not report.passed:
        raise PreconditionError(report)
    return report


# --------------------------------------------------------------------------- #
# 通用滚动算子（全部因果：只用 [t-W+1, t]，min_periods 兜底，窗口内 fillna(0)）
# --------------------------------------------------------------------------- #
def _roll_mean(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    return _finite(pd.Series(np.asarray(x, dtype="float64"))
                   .rolling(window, min_periods=min_periods).mean().to_numpy())


def _roll_std(x: np.ndarray, window: int, min_periods: int = 2) -> np.ndarray:
    return _finite(pd.Series(np.asarray(x, dtype="float64"))
                   .rolling(window, min_periods=min_periods).std(ddof=0).to_numpy())


def _roll_sum(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    return _finite(pd.Series(np.asarray(x, dtype="float64"))
                   .rolling(window, min_periods=min_periods).sum().to_numpy())


def _roll_pct_rank(x: np.ndarray, window: int) -> np.ndarray:
    """逐股票滚动分位（0~1）；``rolling.rank(pct=True)`` 只看窗口内历史，天然因果。"""
    return _finite(pd.Series(np.asarray(x, dtype="float64"))
                   .rolling(window, min_periods=1).rank(pct=True).to_numpy(),
                   default=0.5)


def _roll_corr(x: np.ndarray, y: np.ndarray, window: int,
               min_periods: int = 2) -> np.ndarray:
    return _finite(pd.Series(np.asarray(x, dtype="float64"))
                   .rolling(window, min_periods=min_periods)
                   .corr(pd.Series(np.asarray(y, dtype="float64"))).to_numpy())


def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """逐股票因果滚动 z-score（``min_periods=1``、``ddof=0``、中心化求方差）。

    与 pandas ``rolling.std`` 的 ``E[x²]-E[x]²`` 不同，这里先减窗口均值再平方，
    避免 ``illq`` 在整体缩放下降到 1e-15 量级时的数值相消（否则缩放不变性
    ``atol=1e-12`` 会被破坏）。窗口严格为 ``[t-W+1, t]``：用有效性掩码处理
    序列前缀（``min_periods=1`` 语义），**不填充、不 bfill、不看任何 T 之后的值**，
    因此因果性（改第 30 行之后前 30 行逐位不变）与缩放不变性同时成立。
    """
    values = np.asarray(x, dtype="float64")
    n = int(values.size)
    if n == 0:
        return np.zeros(0, dtype="float64")
    offsets = np.arange(-window + 1, 1, dtype="int64")[None, :]
    t_index = np.arange(n, dtype="int64")[:, None]
    valid = (t_index + offsets) >= 0
    gather = np.clip(t_index + offsets, 0, n - 1)
    win = values[gather]                     # (n, window)，每行只看 [t-W+1, t]
    count = valid.sum(axis=1).astype("float64")
    mean = (win * valid).sum(axis=1) / count
    centered = (win - mean[:, None]) * valid
    var = (centered ** 2).sum(axis=1) / count
    std = np.sqrt(np.maximum(var, 0.0))
    return _safe_divide(centered[:, -1], std, default=0.0)


def _build_liquidity(aligned: pd.DataFrame, out: pd.DataFrame,
                     cfg: dict) -> dict[str, np.ndarray]:
    """P0-1 成交额 / 流动性族（6 列，plan §3.3）。

    ``liq_vwap_dev`` 分子 = ``raw_frame.close``（未复权），
    分母 ``vwap = amount/volume`` = hfq 目录**真实 amount**；
    绝不用 ``close_hfq``、绝不用 ``data/raw`` 的 amount。
    """
    amount = np.asarray(aligned["amount"], dtype="float64")
    volume = np.asarray(aligned["volume"], dtype="float64")
    close_raw = np.asarray(aligned["close_raw"], dtype="float64")
    ret_1 = np.asarray(out["return_1"], dtype="float64")   # 与 unified 逐位一致

    # ① Amihud 非流动性的滚动 z-score（绝对值跨股票不可比，必须标准化）
    #    illq = |ret_1| / amount 在整体缩放下 ×1/s²，z-score 数学上严格尺度不变。
    #    用 :func:`_rolling_zscore` 的「窗口内中心化再平方」实现，避免 pandas
    #    rolling.std 的 E[x²]-E[x]² 在 1e-15 量级数值相消；窗口严格因果。
    illq = _safe_divide(np.abs(ret_1), amount, default=0.0)
    amihud_z20 = _rolling_zscore(illq, 20)

    # ② 成交额 250 日滚动分位（0~1）
    amt_rank250 = _roll_pct_rank(amount, 250)

    # ③ 成交额 / 250 日滚动中位数 的无量纲对数比
    amt_median = _finite(pd.Series(amount).rolling(250, min_periods=1)
                         .median().to_numpy(), default=0.0)
    amt_ratio = _safe_divide(amount, amt_median, default=1.0)
    amt_logratio = np.log(np.maximum(amt_ratio, EPS))

    # ④ VWAP 偏离：close_raw / (amount_hfq/volume) - 1
    vwap = _safe_divide(amount, volume, default=0.0)
    vwap_dev = _safe_divide(close_raw, vwap, default=1.0) - 1.0
    # ⑤ 5 日均值
    vwap_dev_ma5 = _roll_mean(vwap_dev, 5)

    # ⑥ 量价背离：corr(ret_1, Δlog(amount), 20)
    prev_amount = np.concatenate(([np.nan], amount[:-1]))
    dlog_amount = np.log(np.maximum(_safe_divide(amount, prev_amount, default=1.0), EPS))
    pv_div20 = _roll_corr(ret_1, dlog_amount, 20)

    return {
        "liq_amihud_z20": amihud_z20,
        "liq_amt_rank250": amt_rank250,
        "liq_amt_logratio": amt_logratio,
        "liq_vwap_dev": vwap_dev,
        "liq_vwap_dev_ma5": vwap_dev_ma5,
        "liq_pv_div20": pv_div20,
    }


def _build_overnight(aligned: pd.DataFrame, out: pd.DataFrame,
                     cfg: dict) -> dict[str, np.ndarray]:
    """P0-2 隔夜 / 日内收益分解（8 列，全 hfq，plan §3.3）。

    一致性自检（**断言，不是特征列**）：
    ``(1+od_overnight)(1+od_intraday) - 1 == return_1``，``|diff| < 1e-9``。
    首行 ``return_1`` 被 ``pct_change().fillna(0)`` 定义为 0，无前收，
    属定义性边界，自检从 t>=1 起（数学恒等式对 t>=1 精确成立）。
    """
    open_ = np.asarray(aligned["open"], dtype="float64")
    close = np.asarray(aligned["close"], dtype="float64")
    prev_close = np.concatenate(([np.nan], close[:-1]))

    # 跨日隔夜：open_t / close_{t-1} - 1；首行无前收 → 0
    od_overnight = _safe_divide(open_, prev_close, default=1.0) - 1.0
    # 日内：close_t / open_t - 1
    od_intraday = _safe_divide(close, open_, default=1.0) - 1.0

    sum_abs_o = _roll_sum(np.abs(od_overnight), 20)
    sum_abs_i = _roll_sum(np.abs(od_intraday), 20)
    total = sum_abs_o + sum_abs_i

    features = {
        "od_overnight": od_overnight,
        "od_intraday": od_intraday,
        "od_overnight_ma5": _roll_mean(od_overnight, 5),
        "od_intraday_ma5": _roll_mean(od_intraday, 5),
        "od_overnight_vol20": _roll_std(od_overnight, 20),
        "od_intraday_vol20": _roll_std(od_intraday, 20),
        "od_abs_diff": np.abs(od_overnight) - np.abs(od_intraday),
        "od_overnight_share20": _safe_divide(sum_abs_o, total, default=0.0),
    }

    # 一致性硬断言：不满足即构造器报错（不产生任何特征列）
    ret_1 = np.asarray(out["return_1"], dtype="float64")
    decomp = (1.0 + od_overnight) * (1.0 + od_intraday) - 1.0
    diff = np.abs(decomp[1:] - ret_1[1:])
    if diff.size and float(np.max(diff)) >= 1e-9:
        raise ValueError(
            f"od 一致性自检失败：|(1+overnight)(1+intraday)-1 - return_1| "
            f"max={float(np.max(diff)):.3e} >= 1e-9"
        )
    return features


def _build_candlestick(aligned: pd.DataFrame, out: pd.DataFrame,
                       cfg: dict) -> dict[str, np.ndarray]:
    """P0-3 K 线形态族（9 列，全 hfq）。``high==low`` → 0.5 且 ``cdl_oneword=1``。"""
    open_ = np.asarray(aligned["open"], dtype="float64")
    high = np.asarray(aligned["high"], dtype="float64")
    low = np.asarray(aligned["low"], dtype="float64")
    close = np.asarray(aligned["close"], dtype="float64")
    body_max = np.maximum(open_, close)
    body_min = np.minimum(open_, close)

    upper_shadow = _safe_divide(high - body_max, close, default=0.0)
    lower_shadow = _safe_divide(body_min - low, close, default=0.0)
    # 一字板 high==low：分母 0 → 默认 0.5（相等性在整体缩放下保持）
    close_location = _safe_divide(close - low, high - low, default=0.5)
    body = _safe_divide(close - open_, close, default=0.0)
    bar_range = _safe_divide(high - low, close, default=0.0)
    oneword = (high == low).astype("float64")

    return {
        "cdl_upper_shadow": upper_shadow,
        "cdl_lower_shadow": lower_shadow,
        "cdl_close_location": close_location,
        "cdl_body": body,
        "cdl_bar_range": bar_range,
        "cdl_upper_shadow_ma5": _roll_mean(upper_shadow, 5),
        "cdl_lower_shadow_ma5": _roll_mean(lower_shadow, 5),
        "cdl_close_location_ma5": _roll_mean(close_location, 5),
        "cdl_oneword": oneword,
    }


def _build_volatility(aligned: pd.DataFrame, out: pd.DataFrame,
                      cfg: dict) -> dict[str, np.ndarray]:
    """P0-4 波动率结构族（5 列，全 hfq，窗口 20，plan §3.3）。

    ``vol_gap_freq20`` 内部重算隔夜跳空 ``od_overnight``（由 hfq OHLC 直接得到，
    是公式要求的输入量），**不引入 overnight 注册族的任何特征列**，
    保证单变量隔离；``c=1.5`` 取自 ``generator_params.gap_threshold_c``。
    ``vol_path_ratio20`` 的分母使用 base 特征 ``volatility_20``。
    """
    open_ = np.asarray(aligned["open"], dtype="float64")
    high = np.asarray(aligned["high"], dtype="float64")
    low = np.asarray(aligned["low"], dtype="float64")
    close = np.asarray(aligned["close"], dtype="float64")
    ret_1 = np.asarray(out["return_1"], dtype="float64")
    vol20 = np.asarray(out["volatility_20"], dtype="float64")
    ln2 = float(np.log(2.0))
    gap_c = float(cfg["generator_params"]["gap_threshold_c"])

    hl = np.log(np.maximum(_safe_divide(high, low, default=1.0), EPS))
    co = np.log(np.maximum(_safe_divide(close, open_, default=1.0), EPS))
    hl2 = hl ** 2

    # Parkinson：sqrt(mean(ln(H/L)^2) / (4 ln2))
    parkinson2 = _finite(pd.Series(hl2 / (4.0 * ln2))
                         .rolling(20, min_periods=1).mean().to_numpy())
    parkinson20 = np.sqrt(np.maximum(parkinson2, 0.0))

    # Garman-Klass：偶发负值 clamp 0
    gk_term = 0.5 * hl2 - (2.0 * ln2 - 1.0) * (co ** 2)
    gk2 = _finite(pd.Series(gk_term).rolling(20, min_periods=1).mean().to_numpy())
    gk20 = np.sqrt(np.maximum(gk2, 0.0))

    # 上下行半方差比
    down = np.minimum(ret_1, 0.0) ** 2
    up = np.maximum(ret_1, 0.0) ** 2
    sigma_down = np.sqrt(_roll_mean(down, 20))
    sigma_up = np.sqrt(_roll_mean(up, 20))
    downside_ratio20 = _safe_divide(sigma_down, sigma_up, default=0.0)

    # 跳空频率：mean(|overnight| > c * vol_parkinson20) over 20
    prev_close = np.concatenate(([np.nan], close[:-1]))
    overnight = _safe_divide(open_, prev_close, default=1.0) - 1.0
    gap_flag = (np.abs(overnight) > gap_c * parkinson20).astype("float64")
    gap_freq20 = _roll_mean(gap_flag, 20)

    # 日内路径波动 / 收盘波动
    path_ratio20 = _safe_divide(parkinson20, vol20, default=0.0)

    return {
        "vol_parkinson20": parkinson20,
        "vol_gk20": gk20,
        "vol_downside_ratio20": downside_ratio20,
        "vol_gap_freq20": gap_freq20,
        "vol_path_ratio20": path_ratio20,
    }


#: 落盘族构造器注册表（account 不在其中：运行时注入，本轮不构造）
_FAMILY_BUILDERS: dict[str, Callable[[pd.DataFrame, pd.DataFrame, dict],
                                     dict[str, np.ndarray]]] = {
    "liquidity": _build_liquidity,
    "overnight": _build_overnight,
    "candlestick": _build_candlestick,
    "volatility": _build_volatility,
}


def build_features(frame: pd.DataFrame, raw_frame: pd.DataFrame,
                   groups: Iterable[str] | str | None, *, symbol: str = "",
                   amount_dir: str = "", check_precondition: bool = True) -> pd.DataFrame:
    """由内存中的 hfq / raw 帧构造单只股票面板（纯函数，只读、零落盘）。

    这是 :func:`build_family_frame` 的可注入内核，供 ``--self-test`` 做
    缩放不变性 / 因果性验证（可关闭断言以隔离「数据前提」与「特征构造」）。
    **内部第一件事即数据前提断言**，失败即抛、绝不降级填充常数。

    Args:
        frame: hfq 帧（真实 amount）。
        raw_frame: 未复权帧。
        groups: 特征族，必须满足 :func:`validate_single_variable`。
        symbol: 股票代码，写入输出列。
        amount_dir: 实际 amount 目录名（写入断言报告）。
        check_precondition: 是否执行三条断言；仅自检场景可置 False。

    Returns:
        列 = ``date, symbol, close(hfq), close_raw, high_raw, low_raw,
        volume, amount(真实), *active_features(groups)`` 的 DataFrame。
    """
    normalized = _normalize_groups(groups)
    validate_single_variable(normalized)

    # ---- 内部第一件事：三条数据前提断言（fail-closed，直接止损）----
    if check_precondition:
        assert_amount_precondition(frame, raw_frame, amount_dir=amount_dir or None)

    runtime = [g for g in normalized if g in RUNTIME_ONLY_GROUPS]
    if runtime:
        raise ValueError(
            f"TODO(T04)：特征族 {runtime} 为运行时账户状态注入族，"
            "观测来自 env state 而非 frame，本轮 T01 仅注册列名、不构造；"
            "必须与 P0-1~P0-4 完全隔离、独立排期（plan §3.3 / §6 T04）"
        )

    aligned = align_frames_by_day_key(frame, raw_frame)
    aligned["date"] = pd.to_datetime(aligned["date"].astype(str), errors="raise")
    aligned = aligned.sort_values("date").reset_index(drop=True)

    # ---- base 9 列：直接调用 unified.relative_features 取得，逐位一致 ----
    _, base_values = _unified_relative_features(aligned)
    if base_values.shape != (len(aligned), len(BASE_FEATURES)):
        raise ValueError(
            f"unified.relative_features 返回形状 {base_values.shape} 不符预期 "
            f"{(len(aligned), len(BASE_FEATURES))}"
        )

    sym = symbol or str(frame.attrs.get("symbol", ""))
    out = pd.DataFrame({
        "date": aligned["date"].to_numpy(),
        "symbol": sym,
        "close": np.asarray(aligned["close"], dtype="float64"),
        "close_raw": np.asarray(aligned["close_raw"], dtype="float64"),
        "high_raw": np.asarray(aligned["high_raw"], dtype="float64"),
        "low_raw": np.asarray(aligned["low_raw"], dtype="float64"),
        "volume": np.asarray(aligned["volume"], dtype="float64"),
        "amount": np.asarray(aligned["amount"], dtype="float64"),
    })
    for index, name in enumerate(BASE_FEATURES):
        out[name] = base_values[:, index]

    cfg = load_config()
    for group in normalized:
        if group == "base":
            continue
        built = _FAMILY_BUILDERS[group](aligned, out, cfg)
        for name in FEATURE_GROUPS[group]:
            if name not in built:
                raise ValueError(f"族 {group} 缺少注册列 {name}")
            out[name] = _finite(np.asarray(built[name], dtype="float64"))

    feature_cols = active_features(normalized)
    matrix = out[feature_cols].to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ValueError("特征面板含 NaN/Inf；末尾 np.isfinite 全量校验失败")

    meta_cols = ["date", "symbol", "close", "close_raw", "high_raw",
                 "low_raw", "volume", "amount"]
    return out[[*meta_cols, *feature_cols]]


def build_family_frame(symbol: str, groups: Iterable[str] | str | None, *,
                       amount_dir: Path | str = RAW_HFQ_V2_TRAIN,
                       raw_dir: Path | str = RAW) -> pd.DataFrame:
    """只读构造单只股票的逐族特征面板（**零落盘 / 零写回 / 零缓存**）。

    同时只读两份受保护 parquet：
      * ``data/raw_hfq_v2_train/<symbol>.parquet`` → 真实 ``amount`` + 后复权 OHLCV
      * ``data/raw/<symbol>.parquet``              → 未复权 OHLC（唯一价格分母）
    按 ``%Y%m%d`` 8 位日期键对齐（严禁与 ISO 格式混用）。
    ``data/raw`` 的 ``amount`` 在进入构造器前即被忽略，并由断言①挡住误用。

    Args:
        symbol: 6 位股票代码（文件名 stem）。
        groups: 特征族，例如 ``("base", "liquidity")``。
        amount_dir: 真实 amount 目录，默认 ``data/raw_hfq_v2_train``。
        raw_dir: 未复权 OHLC 目录，默认 ``data/raw``。

    Returns:
        列 = ``date, symbol, close(hfq), close_raw, high_raw, low_raw,
        volume, amount(真实), *active_features(groups)`` 的 DataFrame。

    Raises:
        FileNotFoundError: 两份 parquet 任一缺失（fail-closed，不回退）。
        PreconditionError: 三条数据前提断言任一失败（直接止损）。
    """
    amount_path = Path(amount_dir)
    raw_path = Path(raw_dir)
    hfq_file = amount_path / f"{symbol}.parquet"
    raw_file = raw_path / f"{symbol}.parquet"
    if not hfq_file.exists() or not raw_file.exists():
        raise FileNotFoundError(
            f"只读输入缺失: hfq={hfq_file} (exists={hfq_file.exists()}), "
            f"raw={raw_file} (exists={raw_file.exists()})；fail-closed，绝不回退"
        )
    frame = pd.read_parquet(hfq_file)
    raw_frame = pd.read_parquet(raw_file)
    frame.attrs["amount_dir"] = str(amount_path)
    frame.attrs["symbol"] = symbol
    return build_features(frame, raw_frame, groups, symbol=symbol,
                          amount_dir=str(amount_path), check_precondition=True)


# --------------------------------------------------------------------------- #
# 自检（python outputs/feature_families.py --self-test）
# --------------------------------------------------------------------------- #
def self_test(symbol: str | None = None) -> bool:
    """模块级自检：契约 / base 逐位一致 / 列名列数 / 缩放 / 因果 / od / 兜底 / 断言。

    Args:
        symbol: 指定样本股票；默认取 ``data/raw_hfq_v2_train`` 首只。

    Returns:
        全部通过返回 ``True``。
    """
    from numpy.testing import assert_allclose, assert_array_equal

    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        flag = "通过" if ok else "失败"
        print(f"  [{flag}] {name}" + (f"  | {detail}" if detail else ""))

    sym = symbol or sorted(RAW_HFQ_V2_TRAIN.glob("*.parquet"))[0].stem
    hfq = pd.read_parquet(RAW_HFQ_V2_TRAIN / f"{sym}.parquet")
    raw = pd.read_parquet(RAW / f"{sym}.parquet")
    print("=" * 78)
    print(f"feature_families 自检 | 样本 {sym} | hfq {len(hfq)} 行 / raw {len(raw)} 行")
    print("=" * 78)

    # ---- ① 契约 ----
    print("\n[1] 契约与注册表")
    record("BASE_FEATURES 冻结 9 名且来自 unified.FEATURES",
           len(BASE_FEATURES) == 9 and list(BASE_FEATURES) == list(_UNIFIED_FEATURES),
           f"len={len(BASE_FEATURES)}")
    record("active_features(('base',)) 严格等于 BASE_FEATURES",
           active_features(("base",)) == list(BASE_FEATURES))
    record("feature_set_id / contract_id 确定性",
           feature_set_id(("base",)) == "v2-base"
           and feature_set_id(("base", "liquidity")) == "v2-liquidity"
           and contract_id(("base", "liquidity"))
           == "unified-relative-v2+v2-liquidity")
    try:
        validate_single_variable(("base", "liquidity", "overnight"))
        sv_ok = False
    except ValueError:
        sv_ok = True
    record("validate_single_variable 拒绝 base + ≥2 新族", sv_ok)
    try:
        build_features(hfq, raw, ("base", "account"), symbol=sym, check_precondition=False)
        acct_ok = False
    except ValueError:
        acct_ok = True
    record("account 运行时族仅注册列名、本轮拒绝构造（TODO(T04)）", acct_ok)

    # ---- ② 各族列名 / 列数 / base 逐位一致 ----
    print("\n[2] 各族列名 / 列数 / base 逐位一致")
    _, unified_values = _unified_relative_features(hfq.copy())
    base_frame = build_family_frame(sym, ("base",))
    got = base_frame[list(BASE_FEATURES)].to_numpy(dtype="float64")
    record("base 9 列逐位等于 unified.relative_features",
           np.array_equal(got, unified_values), f"shape={got.shape}")
    family_sizes = (("liquidity", 6), ("overnight", 8),
                    ("candlestick", 9), ("volatility", 5))
    for group, size in family_sizes:
        groups = ("base", group)
        frame = build_family_frame(sym, groups)
        expected_cols = ["date", "symbol", "close", "close_raw", "high_raw",
                         "low_raw", "volume", "amount", *active_features(groups)]
        cols = list(FEATURE_GROUPS[group])
        ok = (len(cols) == size and list(frame.columns) == expected_cols
              and np.isfinite(frame[cols].to_numpy(dtype="float64")).all())
        record(f"{group}: {size} 列，列名/列序严格一致且全 finite", ok,
               f"columns={len(cols)}")

    # ---- ③ 缩放不变性（价格 ×s、volume ×s、amount ×s²，两帧同步）----
    print("\n[3] 缩放不变性（s=100，atol=1e-12）")
    scale = 100.0
    hfq_s = hfq.copy()
    raw_s = raw.copy()
    for col in ("open", "high", "low", "close"):
        hfq_s[col] = hfq[col] * scale
        raw_s[col] = raw[col] * scale
    hfq_s["volume"] = hfq["volume"] * scale
    hfq_s["amount"] = hfq["amount"] * (scale ** 2)
    raw_s["volume"] = raw["volume"] * scale
    raw_s["amount"] = raw["amount"] * (scale ** 2)
    for group, _size in family_sizes:
        groups = ("base", group)
        plain = build_features(hfq, raw, groups, symbol=sym, check_precondition=True)
        scaled = build_features(hfq_s, raw_s, groups, symbol=sym, check_precondition=True)
        cols = list(FEATURE_GROUPS[group])
        try:
            assert_allclose(plain[cols].to_numpy(dtype="float64"),
                            scaled[cols].to_numpy(dtype="float64"),
                            atol=1e-12, rtol=0.0)
            ok, detail = True, "全部族列 assert_allclose(atol=1e-12) 通过"
        except AssertionError as exc:
            ok, detail = False, str(exc).splitlines()[0][:140]
        record(f"{group} 缩放不变性", ok, detail)

    # ---- ④ 因果性（第 30 行及之后自洽缩放，前 30 行须逐位不变）----
    print("\n[4] 因果性（改第 30 行之后，前 30 行不变）")
    hfq_key = pd.to_datetime(hfq["date"].astype(str)).dt.strftime(DAY_KEY_FORMAT)
    raw_key = pd.to_datetime(raw["date"].astype(str)).dt.strftime(DAY_KEY_FORMAT)
    cut_key = hfq_key.iloc[30]
    hfq_mask = (hfq_key >= cut_key).to_numpy()
    raw_mask = (raw_key >= cut_key).to_numpy()
    perturb = 1.3
    hfq_c = hfq.copy()
    raw_c = raw.copy()
    for col in ("open", "high", "low", "close"):
        hfq_c.loc[hfq_mask, col] = hfq.loc[hfq_mask, col] * perturb
        raw_c.loc[raw_mask, col] = raw.loc[raw_mask, col] * perturb
    hfq_c.loc[hfq_mask, "volume"] = hfq.loc[hfq_mask, "volume"] * perturb
    hfq_c.loc[hfq_mask, "amount"] = hfq.loc[hfq_mask, "amount"] * (perturb ** 2)
    raw_c.loc[raw_mask, "volume"] = raw.loc[raw_mask, "volume"] * perturb
    raw_c.loc[raw_mask, "amount"] = raw.loc[raw_mask, "amount"] * (perturb ** 2)
    for group, _size in family_sizes:
        groups = ("base", group)
        plain = build_features(hfq, raw, groups, symbol=sym, check_precondition=True)
        changed = build_features(hfq_c, raw_c, groups, symbol=sym, check_precondition=True)
        cols = list(FEATURE_GROUPS[group])
        try:
            assert_array_equal(plain[cols].to_numpy(dtype="float64")[:30],
                               changed[cols].to_numpy(dtype="float64")[:30])
            ok, detail = True, "前 30 行 assert_array_equal 通过"
        except AssertionError as exc:
            ok, detail = False, str(exc).splitlines()[0][:140]
        record(f"{group} 因果性", ok, detail)

    # ---- ⑤ od 一致性 ----
    print("\n[5] od 一致性自检")
    od_frame = build_family_frame(sym, ("base", "overnight"))
    overnight = od_frame["od_overnight"].to_numpy(dtype="float64")
    intraday = od_frame["od_intraday"].to_numpy(dtype="float64")
    ret_1 = od_frame["return_1"].to_numpy(dtype="float64")
    decomp = (1.0 + overnight) * (1.0 + intraday) - 1.0
    diff = np.abs(decomp[1:] - ret_1[1:])
    record("(1+od_overnight)(1+od_intraday)-1 == return_1（t>=1）",
           diff.size > 0 and float(np.max(diff)) < 1e-9,
           f"max|diff|={float(np.max(diff)):.3e}")

    # ---- ⑥ 一字板 / 零分母兜底 ----
    print("\n[6] 一字板兜底（high==low）无 NaN/Inf")
    wb_frame = hfq.head(60).copy()
    wb_raw = raw.head(60).copy()
    word_rows = [5, 6, 7]
    word_hfq_price = wb_frame.loc[word_rows, "close"].to_numpy(dtype="float64")
    word_raw_price = wb_raw.loc[word_rows, "close"].to_numpy(dtype="float64")
    for col in ("open", "high", "low", "close"):
        wb_frame.loc[word_rows, col] = word_hfq_price
        wb_raw.loc[word_rows, col] = word_raw_price
    for group, _size in family_sizes:
        frame = build_features(wb_frame, wb_raw, ("base", group), symbol=sym,
                               check_precondition=False)
        arr = frame[list(FEATURE_GROUPS[group])].to_numpy(dtype="float64")
        record(f"{group} 一字板下全 finite", bool(np.isfinite(arr).all()))
    cdl = build_features(wb_frame, wb_raw, ("base", "candlestick"), symbol=sym,
                         check_precondition=False)
    record("一字板 cdl_close_location=0.5 且 cdl_oneword=1",
           bool(np.allclose(cdl.loc[word_rows, "cdl_close_location"], 0.5)
                and np.allclose(cdl.loc[word_rows, "cdl_oneword"], 1.0)))

    # ---- ⑦ 三条断言：正确目录通过 / 合成 amount 失败 ----
    print("\n[7] 三条数据前提断言")
    report = assert_amount_precondition(hfq, raw, amount_dir=str(RAW_HFQ_V2_TRAIN))
    record("data/raw_hfq_v2_train 三条断言全通过", report.passed,
           f"①={report.std_amount_ratio:.8f} ②={report.vwap_in_range_ratio:.8f} "
           f"③corr={report.corr_with_volume_relative:.8f}")

    # 合成 amount 目录：直接拿 data/raw 自身当 frame（amount=close×volume）
    synthetic = pd.read_parquet(RAW / f"{sym}.parquet")
    try:
        assert_amount_precondition(synthetic, raw, amount_dir=str(RAW))
        syn_ok, syn_detail = False, "未抛错（异常）"
    except PreconditionError as exc:
        syn_ok = ("①" in exc.report.failed_checks) and ("直接止损" in str(exc))
        syn_detail = (f"触发={list(exc.report.failed_checks)}, "
                      f"①std={exc.report.std_amount_ratio:.8f}")
    record("合成 amount(data/raw) 断言①失败并抛 PreconditionError『直接止损』",
           syn_ok, syn_detail)
    try:
        build_features(synthetic, raw, ("base", "liquidity"), symbol=sym,
                       amount_dir=str(RAW), check_precondition=True)
        gate_ok, gate_detail = False, "未抛错（异常）"
    except PreconditionError:
        gate_ok, gate_detail = True, "构造入口第一件事即断言、未降级填充常数"
    record("构造入口 fail-closed：合成 amount 不进入特征构造", gate_ok, gate_detail)

    total = len(results)
    passed = sum(1 for _name, ok, _detail in results if ok)
    print("\n" + "=" * 78)
    print(f"自检汇总: {passed}/{total} 通过  →  "
          f"{'全部通过' if passed == total else '存在失败项'}")
    print("=" * 78)
    return passed == total


def main(argv: list[str] | None = None) -> int:
    """CLI：``--self-test`` 运行模块级自检；``--symbol`` 指定样本股票。"""
    parser = argparse.ArgumentParser(description="逐族特征构造器自检（T01，只读）")
    parser.add_argument("--self-test", action="store_true", help="运行全部自检")
    parser.add_argument("--symbol", default=None, help="自检样本股票代码（默认首只）")
    args = parser.parse_args(argv)
    if args.self_test:
        return 0 if self_test(symbol=args.symbol) else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

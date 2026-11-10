"""绩效指标（**全纯函数**，K14 —— 无副作用，便于单测对照手算）。

覆盖 PRD §6 全部字段：``annual_return, total_return, max_drawdown, sharpe,
win_rate, turnover, num_trades, benchmark_return``。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe",
    "win_rate",
    "turnover",
    "compute_all",
]

#: 一年的交易日数
PERIODS_PER_YEAR: int = 252

_EPS: float = 1e-12


def _clean(equity: "np.ndarray") -> "np.ndarray":
    """转为 1D float64 数组并剔除 NaN/Inf。"""
    arr = np.asarray(equity, dtype="float64").reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def total_return(equity: "np.ndarray") -> float:
    """累计收益率 ``(last / first) - 1``。

    Args:
        equity: 权益曲线。

    Returns:
        累计收益率；长度 < 2 或首值为 0 时返回 0。
    """
    e = _clean(equity)
    if e.size < 2 or abs(e[0]) < _EPS:
        return 0.0
    return float(e[-1] / e[0] - 1.0)


def annual_return(
    equity: "np.ndarray",
    dates: "np.ndarray | None" = None,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """年化收益率（几何年化）。

    优先按 ``dates`` 的真实天数计算年数；否则按 ``len(equity)/periods`` 估算。

    Args:
        equity: 权益曲线。
        dates: 对应日期数组（可选）。
        periods: 年交易日数。

    Returns:
        年化收益率；数据不足返回 0。
    """
    e = _clean(equity)
    if e.size < 2 or abs(e[0]) < _EPS:
        return 0.0
    total = e[-1] / e[0]
    if total <= 0:
        return -1.0

    years = 0.0
    if dates is not None and len(dates) == e.size and e.size >= 2:
        try:
            import pandas as pd

            d0 = pd.to_datetime(str(dates[0]))
            d1 = pd.to_datetime(str(dates[-1]))
            days = (d1 - d0).days
            years = days / 365.25 if days > 0 else 0.0
        except Exception:  # pragma: no cover
            years = 0.0
    if years <= 1e-6:
        years = e.size / float(periods)
    if years <= 1e-9:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(equity: "np.ndarray") -> float:
    """最大回撤（正数，如 ``0.25`` 表示 25%）。

    Args:
        equity: 权益曲线。

    Returns:
        最大回撤幅度（>= 0）。
    """
    e = _clean(equity)
    if e.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(e)
    denom = np.where(np.abs(running_max) < _EPS, _EPS, running_max)
    dd = (running_max - e) / denom
    dd = np.where(running_max <= _EPS, 0.0, dd)
    return float(np.max(dd)) if dd.size else 0.0


def sharpe(
    returns: "np.ndarray",
    rf: float = 0.0,
    periods: int = PERIODS_PER_YEAR,
) -> float:
    """年化夏普比率。

    Args:
        returns: 单期收益率序列。
        rf: 无风险利率（年化）。
        periods: 年交易日数。

    Returns:
        年化夏普比率；标准差为 0 时返回 0。
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    excess = r - (float(rf) / float(periods))
    std = float(np.std(excess, ddof=1)) if r.size > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(float(periods)))


def win_rate(trades: list[dict]) -> float:
    """胜率 = 盈利平仓笔数 / 平仓笔数。

    仅统计带 ``realized_pnl`` 的卖出记录。

    Args:
        trades: 成交记录列表。

    Returns:
        胜率 ``[0,1]``；无平仓返回 0。
    """
    closed = [t for t in trades if "realized_pnl" in t]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if float(t["realized_pnl"]) > 0)
    return float(wins) / float(len(closed))


def turnover(trades: list[dict], equity: "np.ndarray") -> float:
    """换手率 = 总成交额 / 平均权益。

    Args:
        trades: 成交记录列表。
        equity: 权益曲线。

    Returns:
        累计换手率（>= 0）；无成交或权益为 0 返回 0。
    """
    e = _clean(equity)
    if not trades or e.size == 0:
        return 0.0
    avg_eq = float(np.mean(np.abs(e)))
    if avg_eq < _EPS:
        return 0.0
    total_notional = sum(abs(float(t.get("notional", 0.0))) for t in trades)
    return float(total_notional / avg_eq)


def compute_all(
    equity: "np.ndarray",
    dates: "np.ndarray",
    trades: list[dict],
    benchmark_equity: "np.ndarray | None" = None,
) -> dict:
    """一次性计算全部指标（PRD §6 字段）。

    Args:
        equity: 策略权益曲线。
        dates: 日期数组。
        trades: 成交记录。
        benchmark_equity: buy&hold 基线权益曲线（可选）。

    Returns:
        指标 dict，含 ``annual_return, total_return, max_drawdown, sharpe,
        win_rate, turnover, num_trades, benchmark_return``。
    """
    e = _clean(equity)
    rets = np.diff(e) / np.where(np.abs(e[:-1]) < _EPS, _EPS, e[:-1]) if e.size >= 2 else np.array([])

    metrics = {
        "annual_return": annual_return(e, dates),
        "total_return": total_return(e),
        "max_drawdown": max_drawdown(e),
        "sharpe": sharpe(rets),
        "win_rate": win_rate(trades),
        "turnover": turnover(trades, e),
        "num_trades": int(len(trades)),
        "benchmark_return": float(total_return(benchmark_equity))
        if benchmark_equity is not None
        else 0.0,
    }
    return metrics

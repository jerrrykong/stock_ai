"""择时回测台架：带强制自检的忠实实现。

本模块是本项目所有「策略是否有效」结论的**唯一度量口径**。它解决的正是
本轮反复踩到的坑：前视泄漏、概率错位、NaN 污染、复利口径污染。

核心语义（严格因果）
--------------------
  信号在第 t 日**收盘**算出（含 close[t] 及以前的信息）
  → 只能在第 t+1 日生效，赚 t → t+1 的收益
  → days[k] = ret[k+1] × pos[k]

三个强制自检（``self_check``）
---------------------------
  1. 作弊信号（明日收益符号）→ 毛超额必须显著为正
     证明 harness 能识别真实优势；若不为正，harness 本身坏了。
  2. 随机信号 → 毛超额必须显著为**负**
     随机择时少赚 beta 又承担复利拖累，故必然为负，不是 ≈0。
  3. 常数信号（全满仓）→ 毛超额必须 ≈ 0（等价于买入持有）
     证明基准与策略的收益序列严格对齐。

任何用本台架得出的结论，都应先跑 ``self_check``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["TimingBacktest", "BacktestResult", "split_by_year"]

#: 往返摩擦：买入 佣金0.025%+滑点0.1% = 0.125%；卖出 佣金0.025%+印花税0.05%+滑点0.1% = 0.175%
DEFAULT_ROUND_TRIP_COST = 0.003


@dataclass
class BacktestResult:
    """一次择时回测的汇总。"""

    net_excess: float          # 扣摩擦后的超额（相对同段买入持有）
    gross_excess: float        # 扣费前的超额
    switches: float            # 平均年换手（单利口径：切换次数 × 252 / 调仓间隔）
    exposure: float            # 平均在市占比
    n_episodes: int            # 参与的 (股票, 年) episode 数
    per_episode: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"净超额 {self.net_excess:+.2%}  毛超额 {self.gross_excess:+.2%}  "
                f"年换手 {self.switches:.1f}  在市 {self.exposure:.1%}  "
                f"({self.n_episodes} episode)")


class TimingBacktest:
    """忠实择时回测台架（严格因果 + 强制自检）。

    Args:
        frame: 需含列 ``date``（datetime）、``symbol``、``close``、``year``。
        cost: 每次换仓的往返摩擦比例。
        rebalance: 最短调仓间隔（交易日）。1 = 每日可换。
        min_len: episode 最少交易日数。
    """

    def __init__(self, frame: pd.DataFrame, cost: float = DEFAULT_ROUND_TRIP_COST,
                 rebalance: int = 1, min_len: int = 40,
                 global_len: int | None = None):
        # 必须保留原 frame 的 ``_row``（全局行号），它是概率数组的对齐依据。
        if "_row" not in frame.columns:
            frame = frame.copy()
            frame["_row"] = np.arange(len(frame))
        self.frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
        # 子集内位置（与 self.frame 行序一致）
        self.frame["_pos"] = np.arange(len(self.frame))
        self.cost = float(cost)
        self.rebalance = int(max(1, rebalance))
        self.min_len = int(min_len)
        # 全局（未切分）数据的行数；用于接受全局概率数组
        self._global_len = int(global_len) if global_len is not None else len(self.frame)

    # ------------------------------------------------------------------ #
    def _episode(self, group: pd.DataFrame, signal: np.ndarray) -> dict | None:
        """单个 (股票, 年) episode 的收益分解。signal 与 group 等长（0/1 目标仓位）。"""
        close = group["close"].to_numpy()
        if len(close) < self.min_len:
            return None
        ret = np.zeros(len(close))
        ret[1:] = close[1:] / close[:-1] - 1.0
        n = len(close)
        pos = np.zeros(n)
        held = False
        last_change = -10 ** 9
        for t in range(n - 1):
            want = signal[t]
            if want != held and (t - last_change) >= self.rebalance:
                held = want
                last_change = t
            pos[t] = held
        # 严格因果：信号 t 吃 ret[t+1]
        days = ret[1:] * pos[:-1]
        # 换手比例：|Δpos| 之和（满仓↔空仓 = 1.0，即 100% 换手）
        turnover = float(np.abs(np.diff(pos[:-1])).sum())
        # 摩擦按换手比例计：往返成本 × 换手比例
        friction = turnover * self.cost
        strat = float(np.prod(1 + days) - 1)
        hold = float(np.prod(1 + ret[1:]) - 1)
        return {"net": strat - friction - hold,
                "gross": strat - hold,
                "turnover": turnover,
                "exposure": float(pos[:-1].mean())}

    # ------------------------------------------------------------------ #
    def run(self, signal_map: Callable[[pd.DataFrame], np.ndarray]) -> BacktestResult:
        """按 ``signal_map(group) -> 目标仓位序列`` 跑回测。

        Args:
            signal_map: 接收单个 episode 的 DataFrame，返回同长的 0/1 目标仓位。
                **必须在 group 的信息闭包内计算**（只用 date 及以前的列）。

        Returns:
            :class:`BacktestResult`。
        """
        net, gross, switches, exposures = [], [], [], []
        for _key, group in self.frame.groupby(["symbol", "year"], sort=False):
            group = group.sort_values("date")
            signal = np.asarray(signal_map(group), dtype="float64")
            if len(signal) != len(group):
                raise ValueError(
                    f"signal_map 长度 {len(signal)} != episode 长度 {len(group)}")
            row = self._episode(group, signal)
            if row is None:
                continue
            net.append(row["net"])
            gross.append(row["gross"])
            switches.append(row["turnover"])
            exposures.append(row["exposure"])
        if not net:
            return BacktestResult(0.0, 0.0, 0.0, 0.0, 0)
        return BacktestResult(
            net_excess=float(np.mean(net)),
            gross_excess=float(np.mean(gross)),
            # 年换手：episode 内的换手比例 × (每年交易日数 / episode 天数)
            switches=float(np.mean(switches) * 252.0 / max(self._mean_episode_len(), 1)),
            exposure=float(np.mean(exposures)),
            n_episodes=len(net),
            per_episode=list(zip(net, gross)),
        )

    def _mean_episode_len(self) -> float:
        lengths = [len(g) for _k, g in self.frame.groupby(["symbol", "year"], sort=False)]
        return float(np.mean(lengths)) if lengths else 1.0

    # ------------------------------------------------------------------ #
    def run_probability(self, prob: np.ndarray, threshold: float = 0.5) -> BacktestResult:
        """用概率数组回测。

        ``prob`` 可以是**全局**数组（长度 = ``global_len``），也可以是本子集
        对应的数组（长度 = len(self.frame)）。
        - 全局数组：用 ``_row``（全局行号）直接索引
        - 子集数组：用 ``_pos``（子集内位置）索引
        两者都不会错位。
        """
        prob = np.asarray(prob, dtype="float64")
        if len(prob) == len(self.frame) and len(self.frame) != self._global_len:
            index_col = "_pos"          # 已经是子集数组
        elif len(prob) == self._global_len:
            index_col = "_row"          # 全局数组
        elif len(prob) == len(self.frame):
            index_col = "_pos"
        else:
            raise ValueError(
                f"概率长度 {len(prob)} 既不等于子集长度 {len(self.frame)}，"
                f"也不等于全局长度 {self._global_len}")

        def signal_map(group: pd.DataFrame) -> np.ndarray:
            return (prob[group[index_col].to_numpy()] > threshold).astype(float)

        return self.run(signal_map)

    # ------------------------------------------------------------------ #
    def self_check(self) -> dict:
        """三项强制自检。返回各检查的结论与是否通过。"""
        checks = {}

        # 1) 常数满仓 → 毛超额必须 ≈ 0（基准与策略严格对齐）
        full = self.run(lambda g: np.ones(len(g)))
        checks["constant_full"] = {
            "gross_excess": full.gross_excess,
            "pass": abs(full.gross_excess) < 1e-9,
            "expect": "毛超额 ≈ 0",
        }

        # 2) 作弊信号（明日收益符号）→ 毛超额必须显著为正
        def cheat(group: pd.DataFrame) -> np.ndarray:
            close = group["close"].to_numpy()
            nxt = np.roll(close, -1) / close - 1.0   # 明日收益（作弊）
            nxt[-1] = 0.0
            return (nxt > 0).astype(float)

        cheated = self.run(cheat)
        checks["cheating_signal"] = {
            "gross_excess": cheated.gross_excess,
            "pass": cheated.gross_excess > 0.05,
            "expect": "显著为正（证明 harness 能识别真实优势）",
        }

        # 3) 随机信号 → 毛超额必须显著为负（少赚 beta + 复利拖累）
        rng = np.random.default_rng(20261020)
        rand = self.run_probability(rng.random(self._global_len))
        checks["random_signal"] = {
            "gross_excess": rand.gross_excess,
            "pass": rand.gross_excess < 0,
            "expect": "显著为负（不是 ≈0：随机择时少赚 beta 且承担复利拖累）",
        }

        checks["all_passed"] = all(v["pass"] for k, v in checks.items()
                                   if isinstance(v, dict))
        return checks

    # ------------------------------------------------------------------ #
    def report_self_check(self) -> None:
        checks = self.self_check()
        print("===== 回测台架自检 =====")
        for name, label in (("constant_full", "常数满仓"),
                            ("cheating_signal", "作弊信号"),
                            ("random_signal", "随机信号")):
            c = checks[name]
            mark = "✓" if c["pass"] else "✗"
            print(f"  {mark} {label:10s} 毛超额 {c['gross_excess']:+.4%}  "
                  f"（期望：{c['expect']}）")
        print(f"  → {'全部通过，台架可信' if checks['all_passed'] else '**未通过，台架结果不可采信**'}")
        print()


def split_by_year(frame: pd.DataFrame, ranges: dict) -> dict:
    """按年份区间切分，返回 {split_name: 子集}。"""
    out = {}
    for name, (start, end) in ranges.items():
        out[name] = frame[(frame["year"] >= start) & (frame["year"] <= end)].copy()
    return out

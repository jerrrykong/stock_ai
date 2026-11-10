"""合成数据源（离线降级）。

当网络不可用 / 无法访问 AKShare 时，用确定性随机游走生成合法的 OHLCV 日线，
保证全链路（fetch -> sample -> train -> backtest -> predict）可离线验证。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stock_ai.data.schema import COLUMNS, validate_frame
from stock_ai.data.sources.base import DataSource
from stock_ai.utils.errors import DataSourceError
from stock_ai.utils.logging import LoggerFactory

__all__ = ["SyntheticSource"]

logger = LoggerFactory.get(__name__)

_TRADING_DAYS_PER_YEAR = 252


class SyntheticSource(DataSource):
    """确定性随机游走合成数据源（离线兜底）。"""

    name = "synthetic"

    def __init__(
        self,
        adjust: str = "none",
        seed: int = 42,
        start_price: float = 100.0,
        annual_drift: float = 0.06,
        annual_vol: float = 0.28,
    ) -> None:
        """初始化。

        Args:
            adjust: 复权方式（合成数据无实际意义，保留接口一致）。
            seed: 随机种子（保证同一 symbol 同一区间结果可复现）。
            start_price: 起始价格中枢。
            annual_drift: 年化漂移。
            annual_vol: 年化波动率。
        """
        self.adjust: str = adjust
        self.seed: int = int(seed)
        self.start_price: float = float(start_price)
        self.annual_drift: float = float(annual_drift)
        self.annual_vol: float = float(annual_vol)

    def fetch(self, symbol: str, start: str, end: str) -> "Any":
        """生成 ``[start, end]`` 区间的合成日线。

        Args:
            symbol: 股票代码（仅用于播种，保证同代码可复现）。
            start: 起始日期 ``YYYYMMDD``。
            end: 结束日期 ``YYYYMMDD``。

        Returns:
            规范化 DataFrame。
        """
        import pandas as pd

        dates = pd.bdate_range(start=_fmt(start), end=_fmt(end))
        if len(dates) == 0:
            raise DataSourceError(f"合成数据区间为空: {start}~{end}")

        seed = (self.seed + _stable_hash(symbol)) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        n = len(dates)

        dt = 1.0 / _TRADING_DAYS_PER_YEAR
        mu = (self.annual_drift - 0.5 * self.annual_vol**2) * dt
        sigma = self.annual_vol * np.sqrt(dt)

        log_ret = rng.normal(mu, sigma, size=n)
        close = self.start_price * np.exp(np.cumsum(log_ret))

        # 由 close 反推 open/high/low
        open_ = close * (1.0 + rng.normal(0.0, 0.004, size=n))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, size=n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, size=n)))
        volume = np.abs(rng.normal(2.0e6, 5.0e5, size=n))

        df = pd.DataFrame(
            {
                "date": dates,
                "open": np.round(open_, 2),
                "high": np.round(high, 2),
                "low": np.round(low, 2),
                "close": np.round(close, 2),
                "volume": volume,
                "amount": np.round(volume * close, 2),
                "adj_factor": 1.0,
            }
        )
        logger.debug("合成数据生成: %s %d 行 (%s~%s)", symbol, len(df), start, end)
        return validate_frame(df[COLUMNS])


def _stable_hash(text: str) -> int:
    """稳定的非随机字符串哈希（跨进程一致，不用 Python 的随机化 hash）。"""
    h = 0
    for ch in str(text):
        h = (h * 31 + ord(ch)) % (2**31 - 1)
    return h


def _fmt(date: str) -> str:
    """``YYYYMMDD`` -> ``YYYY-MM-DD``。"""
    s = str(date).replace("-", "")
    if len(s) != 8:
        return s
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"

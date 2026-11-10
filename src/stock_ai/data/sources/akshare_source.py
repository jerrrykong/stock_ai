"""AKShare 数据源适配器（主源）。

AKShare 为可选依赖：导入失败时 :meth:`fetch` 抛 :class:`DataSourceError`，
由 :class:`StockDataLoader` 的 fallback 链接管。
"""

from __future__ import annotations

from typing import Any

from stock_ai.data.schema import COLUMNS, validate_frame
from stock_ai.data.sources.base import DataSource
from stock_ai.utils.errors import ConfigError, DataSourceError
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.retry import retry_with_backoff

__all__ = ["AkShareSource"]

logger = LoggerFactory.get(__name__)

#: 复权方式 -> AKShare ``adjust`` 取值（``""`` 表示不复权）
_ADJUST_MAP: dict[str, str] = {"qfq": "qfq", "hfq": "hfq", "none": ""}


class AkShareSource(DataSource):
    """AKShare 主数据源（``ak.stock_zh_a_hist``）。"""

    name = "akshare"

    def __init__(
        self,
        adjust: str = "none",
        retry_times: int = 3,
        retry_backoff: float = 1.5,
        timeout: float = 15.0,
    ) -> None:
        """初始化。

        Args:
            adjust: 复权方式 ``qfq`` / ``hfq`` / ``none``（走 :data:`_ADJUST_MAP`，
                ``none`` 映射为空串）。
            retry_times: 重试次数。
            retry_backoff: 退避基数（秒）。
            timeout: 超时（秒，AKShare 内部不直接支持，仅用于语义一致）。

        Raises:
            ConfigError: ``adjust`` 不在支持集合内。
        """
        from stock_ai.config.schema import normalize_adjust

        self.adjust: str = normalize_adjust(adjust)
        if self.adjust not in _ADJUST_MAP:
            raise ConfigError(
                f"AKShare 源不支持的复权方式: {self.adjust!r}"
                f"（可选: {sorted(_ADJUST_MAP)}）",
                context={"adjust": self.adjust, "valid": sorted(_ADJUST_MAP)},
            )
        self.retry_times: int = int(retry_times)
        self.retry_backoff: float = float(retry_backoff)
        self.timeout: float = float(timeout)

    # ------------------------------------------------------------------ #
    def fetch(self, symbol: str, start: str, end: str) -> "Any":
        """拉取日线（带指数退避重试）。"""

        @retry_with_backoff(times=self.retry_times, backoff=self.retry_backoff)
        def _call() -> "Any":
            import akshare as ak

            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=_ADJUST_MAP[self.adjust],
            )
            if raw is None or len(raw) == 0:
                raise DataSourceError(f"AKShare 返回空数据: {symbol} {start}~{end}")
            return raw

        try:
            raw = _call()
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(
                f"AKShare 拉取失败: {symbol} ({exc})", context={"symbol": symbol}
            ) from exc

        return self._normalize(raw)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(raw: "Any") -> "Any":
        """把 AKShare 中文列名映射为内部 schema 并校验。"""
        import pandas as pd

        # AKShare 常见中文列 -> 内部列
        rename = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "_amplitude",
            "涨跌幅": "_pct_chg",
            "涨跌额": "_chg",
            "换手率": "_turnover",
        }
        df = raw.rename(columns=rename)

        # 处理可能的英文列名
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
            "Amount": "amount",
        })

        keep = ["date", "open", "high", "low", "close", "volume", "amount"]
        present = [c for c in keep if c in df.columns]
        df = df[present].copy()
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0

        # AKShare 的 volume 单位是「手」，乘以 100 转股
        if "volume" in df.columns:
            try:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100.0
            except Exception:  # pragma: no cover
                pass

        # 补齐缺失列后按 COLUMNS 排序交给 validate_frame
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col != "adj_factor" else 1.0
        return validate_frame(df[COLUMNS])

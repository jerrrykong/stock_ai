"""东方财富数据源适配器（fallback）。

使用 ``https://push2his.eastmoney.com/api/qt/stock/kline/get`` 免费接口，
返回 JSON：``data.klines = ["2023-01-03,open,close,high,low,volume,amount,...", ...]``。
"""

from __future__ import annotations

from typing import Any

from stock_ai.data.schema import COLUMNS, validate_frame
from stock_ai.data.sources.base import DataSource
from stock_ai.utils.errors import ConfigError, DataSourceError
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.retry import retry_with_backoff

__all__ = ["EastMoneySource"]

logger = LoggerFactory.get(__name__)

_BASE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

#: 复权方式 -> 东方财富 ``fqt`` 取值（0=不复权, 1=前复权, 2=后复权）
_ADJUST_MAP: dict[str, int] = {"qfq": 1, "hfq": 2, "none": 0}


class EastMoneySource(DataSource):
    """东方财富 fallback 数据源。"""

    name = "eastmoney"

    def __init__(
        self,
        adjust: str = "none",
        retry_times: int = 3,
        retry_backoff: float = 1.5,
        timeout: float = 15.0,
    ) -> None:
        """初始化。

        Args:
            adjust: 复权方式 ``qfq`` / ``hfq`` / ``none``（走 :data:`_ADJUST_MAP`）。
            retry_times: 重试次数。
            retry_backoff: 退避基数（秒）。
            timeout: HTTP 超时（秒）。

        Raises:
            ConfigError: ``adjust`` 不在支持集合内。
        """
        from stock_ai.config.schema import normalize_adjust

        self.adjust: str = normalize_adjust(adjust)
        if self.adjust not in _ADJUST_MAP:
            raise ConfigError(
                f"东方财富源不支持的复权方式: {self.adjust!r}"
                f"（可选: {sorted(_ADJUST_MAP)}）",
                context={"adjust": self.adjust, "valid": sorted(_ADJUST_MAP)},
            )
        self.retry_times: int = int(retry_times)
        self.retry_backoff: float = float(retry_backoff)
        self.timeout: float = float(timeout)

    @staticmethod
    def _secid(symbol: str) -> str:
        """转东方财富 secid：``1.600519``（沪） / ``0.000001``（深）。"""
        s = str(symbol).strip()
        if "." in s:
            s = s.split(".")[0]
        market = "1" if s.startswith(("6", "9")) else "0"
        return f"{market}.{s}"

    def fetch(self, symbol: str, start: str, end: str) -> "Any":
        """拉取日线（带指数退避重试）。"""

        @retry_with_backoff(times=self.retry_times, backoff=self.retry_backoff)
        def _call() -> list:
            import requests

            fqt = _ADJUST_MAP[self.adjust]
            params = {
                "secid": self._secid(symbol),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",     # 日线
                "fqt": str(fqt),
                "beg": start,
                "end": end,
                "lmt": "10000",
            }
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                )
            }
            resp = requests.get(
                _BASE_URL, params=params, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or {}
            klines = data.get("klines") or []
            if not klines:
                raise DataSourceError(f"东方财富返回空数据: {symbol}")
            return klines

        try:
            klines = _call()
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(
                f"东方财富拉取失败: {symbol} ({exc})", context={"symbol": symbol}
            ) from exc

        return self._normalize(klines)

    @staticmethod
    def _normalize(klines: list) -> "Any":
        """解析东方财富 klines 字符串列表。"""
        import pandas as pd

        records: list[dict[str, Any]] = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 7:
                continue
            records.append(
                {
                    "date": parts[0],
                    "open": _to_float(parts[1]),
                    "close": _to_float(parts[2]),
                    "high": _to_float(parts[3]),
                    "low": _to_float(parts[4]),
                    "volume": _to_float(parts[5]) * 100.0,  # 手 -> 股
                    "amount": _to_float(parts[6]),
                    "adj_factor": 1.0,
                }
            )
        if not records:
            raise DataSourceError("东方财富解析后无有效数据行")
        df = pd.DataFrame.from_records(records)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col != "adj_factor" else 1.0
        return validate_frame(df[COLUMNS])


def _to_float(value: Any) -> float:
    """容错转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

"""新浪行情数据源适配器（fallback）。

使用 ``https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/
CN_MarketData.getKLineData`` 免费接口，返回**纯 JSON 数组**::

    [{"day":"2020-01-02","open":"1128.000","high":"1145.060",
      "low":"1116.000","close":"1130.000","volume":"14809916"}, ...]

特点（与腾讯源互补，用于规避腾讯接口的窗口速率风控）：

- **一次返回整段历史**（``datalen`` 取足够大即可覆盖个股全生命周期，约 6000+ 根），
  无需分页；
- ``volume`` 单位本就是**股**（腾讯源返回的是手，需 ×100 换算）；
- 价格与腾讯源**同口径、逐日一致**（已实测 600519 @2020-01-02：两边 close 均 1130.000）；
- 该接口仅提供**不复权**数据，故 :meth:`fetch` 在 ``adjust != "none"`` 时抛
  :class:`DataSourceError`（让 fallback 链自动改用其它源，而不是静默给出错口径数据）。
"""

from __future__ import annotations

from typing import Any

from stock_ai.data.schema import COLUMNS, validate_frame
from stock_ai.data.sources.base import DataSource
from stock_ai.utils.errors import DataSourceError
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.retry import retry_with_backoff

__all__ = ["SinaSource"]

logger = LoggerFactory.get(__name__)

_BASE_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
#: 单次请求的 K 线根数上限。A 股历史最长约 8500 交易日（1990 年至今），
#: 取 10000 可一次覆盖任何个股的全部历史，从而免去分页。
_DATALEN: int = 10000


class SinaSource(DataSource):
    """新浪行情 fallback 数据源（不复权、单次全历史）。"""

    name = "sina"

    def __init__(
        self,
        adjust: str = "none",
        retry_times: int = 3,
        retry_backoff: float = 1.5,
        timeout: float = 20.0,
    ) -> None:
        """初始化。

        Args:
            adjust: 复权方式。**新浪该接口仅支持 ``none``（不复权）**；传入其它值时
                不在此处报错（避免 loader 构造 registry 时整体失败），而是在
                :meth:`fetch` 抛 :class:`DataSourceError` 交由 fallback 链处理。
            retry_times: 重试次数。
            retry_backoff: 退避基数（秒）。
            timeout: HTTP 超时（秒）。
        """
        from stock_ai.config.schema import normalize_adjust

        self.adjust: str = normalize_adjust(adjust)
        self.retry_times: int = int(retry_times)
        self.retry_backoff: float = float(retry_backoff)
        self.timeout: float = float(timeout)

    @staticmethod
    def _prefix(symbol: str) -> str:
        """给 6 位代码加市场前缀（sh/sz/bj）。"""
        s = str(symbol).strip()
        if "." in s:
            s = s.split(".")[0]
        if s.startswith(("6", "9")):
            return f"sh{s}"
        if s.startswith(("0", "2", "3")):
            return f"sz{s}"
        if s.startswith(("4", "8")):
            return f"bj{s}"
        return f"sh{s}"

    def fetch(self, symbol: str, start: str, end: str) -> "Any":
        """拉取日线（一次请求取全历史，再按 ``[start, end]`` 裁剪）。

        Args:
            symbol: 股票代码。
            start: 起始 ``YYYYMMDD``。
            end: 结束 ``YYYYMMDD``。

        Returns:
            规范化后的日线 DataFrame。

        Raises:
            DataSourceError: 复权口径非 ``none``；或请求/解析失败、区间内无数据。
        """
        if self.adjust != "none":
            raise DataSourceError(
                f"新浪源仅提供不复权（none）数据，无法满足 adjust={self.adjust!r}",
                context={"symbol": symbol, "adjust": self.adjust},
            )

        code = self._prefix(symbol)
        start_norm = _norm_date(start)
        end_norm = _norm_date(end)

        @retry_with_backoff(times=self.retry_times, backoff=self.retry_backoff)
        def _call() -> list:
            import requests

            params = {"symbol": code, "scale": "240", "ma": "no", "datalen": str(_DATALEN)}
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                ),
                "Referer": "https://finance.sina.com.cn/",
            }
            resp = requests.get(
                _BASE_URL, params=params, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            return _load_rows(resp.text)

        try:
            rows = _call()
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - fallback 需吞掉所有源异常
            raise DataSourceError(
                f"新浪行情拉取失败: {symbol} ({exc})", context={"symbol": symbol}
            ) from exc

        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            day = _norm_date(row.get("day", ""))
            if not day or day < start_norm or day > end_norm:
                continue
            close = _to_float(row.get("close"))
            volume = _to_float(row.get("volume"))
            records.append(
                {
                    "date": day,
                    "open": _to_float(row.get("open")),
                    "high": _to_float(row.get("high")),
                    "low": _to_float(row.get("low")),
                    "close": close,
                    "volume": volume,               # 新浪本就是「股」
                    "amount": close * volume,       # 接口不含成交额，按 close*volume 估算
                    "adj_factor": 1.0,              # 不复权
                }
            )

        if not records:
            raise DataSourceError(
                f"新浪行情在区间 {start}~{end} 无数据: {symbol}",
                context={"symbol": symbol, "start": start, "end": end},
            )

        import pandas as pd

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


def _norm_date(value: Any) -> str:
    """把 ``YYYY-MM-DD`` / ``YYYYMMDD`` 归一化为 ``YYYYMMDD``。"""
    return str(value).strip().replace("-", "").replace("/", "")


def _load_rows(text: str) -> list:
    """解析新浪返回文本为行列表。

    正常返回是 JSON 数组；但该接口偶尔会返回带 BOM / 前导空白的文本，
    或 ``null``（无数据），故做防御性处理。

    Args:
        text: HTTP 响应正文。

    Returns:
        行列表（元素为 dict）。

    Raises:
        DataSourceError: 文本为空或结构非预期。
    """
    import json

    raw = (text or "").strip().lstrip("\ufeff")
    if not raw:
        raise DataSourceError("新浪行情返回空响应")
    if raw in ("null", "[]"):
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"新浪行情返回非法 JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise DataSourceError(f"新浪行情返回非预期结构: {type(payload).__name__}")
    return payload

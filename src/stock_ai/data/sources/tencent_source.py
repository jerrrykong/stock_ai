"""腾讯行情数据源适配器（fallback）。

使用 ``https://web.ifzq.gtimg.cn/appstock/app/fqkline/get`` 免费接口，
不带 ``_var`` 参数时返回**纯 JSON**：
``data.<code>.<qfqday|day> = [[date, open, close, high, low, volume], ...]``。
解析时对可能存在的 JSONP 前缀（``kline_day={...}``）做防御性兼容。腾讯单次最多返回
约 800 根 K 线（实测 >800 会被静默截断、``>=3000`` 直接返回 ``param error``），且长区间
请求返回的是**区间尾部**数据，故本模块按页从区间末端向前分页拉取直至覆盖 ``start``。
"""

from __future__ import annotations

from typing import Any

from stock_ai.data.schema import COLUMNS, validate_frame
from stock_ai.data.sources.base import DataSource
from stock_ai.utils.errors import ConfigError, DataSourceError
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.retry import retry_with_backoff

__all__ = ["TencentSource"]

logger = LoggerFactory.get(__name__)

_BASE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

#: 单次请求允许的最大 K 线根数（实测腾讯对 >~800 会静默截断、>=3000 直接 param error）
_MAX_BARS_PER_REQ: int = 800
#: 单只股票分页拉取的最大页数（防御性上限，避免异常情况下死循环）
_MAX_PAGES: int = 50

#: 复权方式 -> (返回数据字段名, 接口 param 的复权标识)。
#: 腾讯 ``fqkline`` 接口用不同字段承载不同复权口径：``qfqday`` / ``hfqday`` / ``day``。
_ADJUST_MAP: dict[str, tuple[str, str]] = {
    "qfq": ("qfqday", "qfq"),
    "hfq": ("hfqday", "hfq"),
    "none": ("day", ""),
}


class TencentSource(DataSource):
    """腾讯行情 fallback 数据源。"""

    name = "tencent"

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
                f"腾讯源不支持的复权方式: {self.adjust!r}"
                f"（可选: {sorted(_ADJUST_MAP)}）",
                context={"adjust": self.adjust, "valid": sorted(_ADJUST_MAP)},
            )
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
        """拉取日线（自动分页；每页带指数退避重试）。

        腾讯单次最多返回约 :data:`_MAX_BARS_PER_REQ` 根 K 线，且长区间请求返回的是
        区间**尾部**数据。因此这里以 ``end`` 为起点向前分页，每页取到数据后把 ``end``
        前移到该页最早日期的前一日，直至覆盖 ``start`` 或数据耗尽。

        Args:
            symbol: 股票代码。
            start: 起始 ``YYYYMMDD``。
            end: 结束 ``YYYYMMDD``。

        Returns:
            规范化后的日线 DataFrame。

        Raises:
            DataSourceError: 所有分页均无数据或请求异常。
        """
        code = self._prefix(symbol)
        field, api_adjust = _ADJUST_MAP[self.adjust]
        start_norm = _norm_date(start)

        @retry_with_backoff(times=self.retry_times, backoff=self.retry_backoff)
        def _call(page_end: str) -> list:
            import requests

            count = _estimate_bars(start, page_end)
            params = {
                "param": f"{code},day,{_fmt(start)},{_fmt(page_end)},{count},{api_adjust}",
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
            payload = _load_json(resp.text)
            data = payload.get("data")
            if not isinstance(data, dict):
                # 例如 {"code":0,"msg":"param error","data":[]} —— data 为 list
                return []
            node = data.get(code) or {}
            if not isinstance(node, dict):
                return []
            rows = node.get(field)
            if rows is None:
                # 兼容：部分响应缺少目标字段时回退，但**显式告警**（不静默改变复权口径）
                alt = node.get("day") or node.get("qfqday") or node.get("hfqday")
                if alt is not None:
                    logger.warning(
                        "腾讯返回缺少字段 %s，回退使用其它字段（口径可能与请求 %s 不一致）",
                        field, self.adjust,
                    )
                    rows = alt
                else:
                    rows = []
            return list(rows or [])

        all_rows: list = []
        cur_end = _norm_date(end)
        try:
            for _ in range(_MAX_PAGES):
                rows = _call(cur_end)
                if not rows:
                    break
                # 该页为较晚的区间，置于已有数据之前
                all_rows = list(rows) + all_rows
                first_date = _norm_date(rows[0][0])
                if first_date <= start_norm:
                    break
                nxt_end = _prev_day(rows[0][0])
                if nxt_end >= cur_end or nxt_end < start_norm:
                    # 无进展 / 已越过起始边界，停止分页
                    break
                cur_end = nxt_end
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - fallback 需吞掉所有源异常
            raise DataSourceError(
                f"腾讯行情拉取失败: {symbol} ({exc})", context={"symbol": symbol}
            ) from exc

        if not all_rows:
            raise DataSourceError(f"腾讯行情返回空数据: {symbol}")
        return self._normalize(all_rows)

    @staticmethod
    def _normalize(rows: list) -> "Any":
        """把腾讯返回的行列表转成标准 DataFrame。"""
        import pandas as pd

        records: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 6:
                continue
            records.append(
                {
                    "date": row[0],
                    "open": _to_float(row[1]),
                    "close": _to_float(row[2]),
                    "high": _to_float(row[3]),
                    "low": _to_float(row[4]),
                    "volume": _to_float(row[5]) * 100.0,  # 手 -> 股
                    "amount": 0.0,
                    "adj_factor": 1.0,
                }
            )
        if not records:
            raise DataSourceError("腾讯行情解析后无有效数据行")
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


def _fmt(date: str) -> str:
    """``YYYYMMDD`` -> ``YYYY-MM-DD``（腾讯接口要求）。"""
    s = str(date).replace("-", "")
    if len(s) != 8:
        return s
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _estimate_bars(start: str, end: str) -> int:
    """按请求区间日历天数估算单页所需 K 线根数。

    A 股约 244 交易日 / 365 天，取 ``days * 0.75 + 60`` 作冗余估算，
    下限 320，上限 :data:`_MAX_BARS_PER_REQ`（腾讯单次请求的实测安全上限）；
    日期无法解析时回退 320。

    Args:
        start: 起始 ``YYYYMMDD``。
        end: 结束 ``YYYYMMDD``。

    Returns:
        估算的 K 线根数。
    """
    from datetime import datetime

    try:
        s = datetime.strptime(str(start).replace("-", ""), "%Y%m%d")
        e = datetime.strptime(str(end).replace("-", ""), "%Y%m%d")
        days = max(0, (e - s).days)
    except (TypeError, ValueError):
        return 320
    count = int(days * 0.75) + 60
    return max(320, min(_MAX_BARS_PER_REQ, count))


def _norm_date(value: Any) -> str:
    """把 ``YYYY-MM-DD`` / ``YYYYMMDD`` 归一化为 ``YYYYMMDD``。"""
    return str(value).strip().replace("-", "").replace("/", "")


def _prev_day(value: Any) -> str:
    """返回给定日期的前一自然日（``YYYYMMDD``）。

    Args:
        value: ``YYYY-MM-DD`` 或 ``YYYYMMDD``。

    Returns:
        前一日的 ``YYYYMMDD`` 字符串。
    """
    from datetime import datetime, timedelta

    d = datetime.strptime(_norm_date(value), "%Y%m%d") - timedelta(days=1)
    return d.strftime("%Y%m%d")


def _load_json(text: str) -> dict:
    """解析腾讯返回文本，兼容纯 JSON 与 JSONP（``kline_day={...};``）两种形式。

    先剥离第一个 ``{`` 之前的 JSONP 前缀，再用 :meth:`json.JSONDecoder.raw_decode`
    从第一个 JSON 值开始解析，从而容忍结尾的分号/换行等多余字符。

    Args:
        text: HTTP 响应正文。

    Returns:
        解析后的 dict。

    Raises:
        DataSourceError: 文本为空、非法 JSON，或顶层不是对象。
    """
    import json

    raw = (text or "").strip()
    brace = raw.find("{")
    if brace > 0:
        # 剥离 ``xxx=`` JSONP 前缀，只保留 JSON 对象部分
        raw = raw[brace:]
    if not raw:
        raise DataSourceError("腾讯行情返回空响应")
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"腾讯行情返回非法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataSourceError(f"腾讯行情返回非预期结构: {type(payload).__name__}")
    return payload

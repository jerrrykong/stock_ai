"""A 股涨跌停幅度规则（按板块 + 生效日期分段），纯函数、无副作用、便于单测。

规则依据（交易所现行办法）：

===========  ==========  ============================================
板块         幅度        生效条件
===========  ==========  ============================================
主板         10%        沪 600/601/603/605、深 000/001/002/003
创业板       20%        300/301，自 2020-08-24 注册制起全面实施
科创板       20%        688/689，自 2019-07-22 开市即为 20%
北交所       30%        4/8 开头
主板风险警示 5%         ST / *ST（**近似**）
===========  ==========  ============================================

**ST 说明（重要）**：ST/*ST 名单需要额外数据源，本项目**不联网获取**，
因此默认按非 ST 处理（主板 10%）。调用方可传入 ``st=True`` 使用 5% 近似值；
注意主板 ST 为 5%，而创业板/科创板风险警示股票仍为 20%、北交所为 30%
（与主板不同），这一点已在 :func:`price_limit` 内按板块区分。

**新股上市首日**无涨跌幅限制（沪深主板/创业板/科创板均为上市首日不设限，
仅上市后第 2 个交易日起适用）；本模块需要调用方在首日自行跳过
（:func:`price_limit` 无法从代码与日期推断"是否该股的首个交易日"）。
"""

from __future__ import annotations

__all__ = [
    "MAIN_LIMIT",
    "GROWTH_LIMIT",
    "STAR_LIMIT",
    "BSE_LIMIT",
    "ST_LIMIT",
    "CHINEXT_FULL_FROM",
    "STAR_FROM",
    "board_of",
    "price_limit",
    "limit_prices",
    "is_blocked_at_open",
    "PRICE_EPS",
]

#: 主板涨跌幅
MAIN_LIMIT: float = 0.10
#: 创业板涨跌幅（注册制全面实施后）
GROWTH_LIMIT: float = 0.20
#: 科创板涨跌幅
STAR_LIMIT: float = 0.20
#: 北交所涨跌幅
BSE_LIMIT: float = 0.30
#: 主板风险警示股票（ST / *ST）涨跌幅
ST_LIMIT: float = 0.05

#: 创业板全面实施 20% 涨跌幅的日期（注册制首日）
CHINEXT_FULL_FROM: str = "20200824"
#: 科创板开市日期
STAR_FROM: str = "20190722"

#: 价格比较容差（元）
PRICE_EPS: float = 1e-6


def _code(symbol: object) -> str:
    """取出 6 位数字代码（容忍 ``600519.SH`` 之类后缀）。"""
    s = str(symbol or "").strip()
    if "." in s:
        s = s.split(".")[0]
    return s


def _day(date: object) -> str:
    """把 ``datetime`` / ``YYYY-MM-DD`` / ``YYYYMMDD`` 归一化为 ``YYYYMMDD``。"""
    if hasattr(date, "strftime"):
        return date.strftime("%Y%m%d")
    s = str(date or "").strip().replace("-", "").replace("/", "")
    return s


def board_of(symbol: object) -> str:
    """按代码判定板块。

    Args:
        symbol: 股票代码。

    Returns:
        ``"star"``（科创板）/ ``"growth"``（创业板）/ ``"bse"``（北交所）/ ``"main"``（主板）。
    """
    code = _code(symbol)
    if len(code) != 6 or not code.isdigit():
        return "main"
    if code[0] == "6" and code[1:3] in {"88", "89"}:
        return "star"
    if code.startswith("30"):
        return "growth"
    if code[0] in {"4", "8"}:
        return "bse"
    return "main"


def price_limit(symbol: object, date: object, st: bool = False) -> float:
    """返回该股在 ``date`` 当日的涨跌幅上限（正数，如 ``0.10``）。

    Args:
        symbol: 股票代码（决定板块）。
        date: 交易日（决定分段生效日期）。
        st: 是否为风险警示股票（ST/*ST）。仅影响主板，其他板块不受影响。

    Returns:
        涨跌幅（``0.05`` / ``0.10`` / ``0.20`` / ``0.30``）。
    """
    board, day = board_of(symbol), _day(date)
    if board == "bse":
        return BSE_LIMIT
    if board == "star":
        return STAR_LIMIT if day >= STAR_FROM else MAIN_LIMIT
    if board == "growth":
        return GROWTH_LIMIT if day >= CHINEXT_FULL_FROM else MAIN_LIMIT
    return ST_LIMIT if st else MAIN_LIMIT


def limit_prices(
    prev_close: float,
    symbol: object,
    date: object,
    st: bool = False,
) -> tuple[float, float]:
    """由昨收计算涨停价与跌停价（四舍五入到 0.01 元，与交易所一致）。

    Args:
        prev_close: 上一交易日**未复权**收盘价。
        symbol: 股票代码。
        date: 交易日。
        st: 是否为风险警示股票。

    Returns:
        ``(涨停价, 跌停价)``。``prev_close <= 0`` 时返回 ``(0.0, 0.0)``。
    """
    if prev_close is None or float(prev_close) <= 0:
        return 0.0, 0.0
    rate = price_limit(symbol, date, st)
    base = float(prev_close)
    return round(base * (1.0 + rate) + 1e-9, 2), round(base * (1.0 - rate) + 1e-9, 2)


def is_blocked_at_open(
    open_price: float,
    prev_close: float,
    symbol: object,
    date: object,
    side: str,
    st: bool = False,
) -> bool:
    """判断在当日**开盘价**上是否因一字涨跌停而无法成交。

    规则：开盘价已达（或越过）涨停价则买不进；已达（或越过）跌停价则卖不出。
    该判定比"当日是否涨停"更严格：一字板之外，开盘即封板的股票同样无法在
    开盘价成交（这是保守且符合实盘的做法）。

    Args:
        open_price: 当日真实（未复权）开盘价。
        prev_close: 上一交易日未复权收盘价。
        symbol: 股票代码。
        date: 交易日。
        side: ``"buy"`` 或 ``"sell"``。
        st: 是否为风险警示股票。

    Returns:
        该方向无法在开盘成交返回 ``True``。
    """
    up, down = limit_prices(prev_close, symbol, date, st)
    if up <= 0 or down <= 0 or open_price is None:
        return False
    if str(side).lower() == "buy":
        return float(open_price) >= up - PRICE_EPS
    return float(open_price) <= down + PRICE_EPS

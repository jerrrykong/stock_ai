"""数据层 schema：列名、dtype 常量与校验（K3）。

严格列顺序：``["date","open","high","low","close","volume","amount","adj_factor"]``
任何写入前必须过 :func:`validate_frame`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stock_ai.utils.errors import DataSourceError

__all__ = ["COLUMNS", "DTYPES", "OHLCV_COLUMNS", "SymbolInfo", "validate_frame", "empty_frame"]

#: parquet 标准列顺序（K3）
COLUMNS: list[str] = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
]

#: 基础 OHLCV 列（不含 date/amount/adj_factor）
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

#: dtype 映射（K3）
DTYPES: dict[str, str] = {
    "date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "amount": "float64",
    "adj_factor": "float64",
}


@dataclass
class SymbolInfo:
    """单只股票的缓存概要信息。

    ``adjust`` 记录该标的缓存所用的复权口径（``qfq``/``hfq``/``none``）；
    默认空串以兼容早期无此字段的 ``meta.json``（读取时视为"未知口径"）。
    """

    symbol: str
    start: str
    end: str
    rows: int
    source: str
    updated_at: str
    missing_days: int = 0
    adjust: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转为普通 dict。"""
        return {
            "symbol": self.symbol,
            "start": self.start,
            "end": self.end,
            "rows": self.rows,
            "source": self.source,
            "updated_at": self.updated_at,
            "missing_days": self.missing_days,
            "adjust": self.adjust,
        }


def empty_frame() -> "Any":
    """返回一个符合 schema 的空 DataFrame。"""
    import pandas as pd

    df = pd.DataFrame({c: pd.Series(dtype=_pandas_dtype(DTYPES[c])) for c in COLUMNS})
    return df


def _pandas_dtype(dtype: str) -> str:
    """把 numpy dtype 字符串转为 pandas 可接受字符串。"""
    return "datetime64[ns]" if dtype == "datetime64[ns]" else dtype


def validate_frame(df: "Any") -> "Any":
    """校验并规范化日线 DataFrame（K3）。

    执行：列补齐/重排、dtype 转换、date 升序、去重、剔除非法行。

    Args:
        df: 待校验的 DataFrame。

    Returns:
        规范化后的 DataFrame（重新索引）。

    Raises:
        DataSourceError: 缺少关键列或全部行非法。
    """
    import pandas as pd

    if df is None or len(df) == 0:
        raise DataSourceError("数据源返回空数据集")

    out = df.copy()

    # 缺列补齐（amount / adj_factor 允许缺省，用中性值填充）
    for col in COLUMNS:
        if col not in out.columns:
            if col == "amount":
                out[col] = 0.0
            elif col == "adj_factor":
                out[col] = 1.0
            elif col == "date":
                raise DataSourceError(f"数据源返回缺少必要列: {col}")
            elif col in OHLCV_COLUMNS:
                raise DataSourceError(f"数据源返回缺少必要列: {col}")

    # 强制 dtype
    try:
        out["date"] = pd.to_datetime(out["date"])
        for col in ("open", "high", "low", "close", "volume", "amount", "adj_factor"):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    except Exception as exc:
        raise DataSourceError(f"数据列 dtype 转换失败: {exc}") from exc

    # 列顺序固定
    out = out[COLUMNS]

    # 剔除价格非法行
    before = len(out)
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[(out["close"] > 0) & (out["open"] > 0)]
    if len(out) == 0:
        raise DataSourceError(f"全部 {before} 行数据非法（NaN/非正价格）")

    # 升序 + 去重（保留最后一条）
    out = out.sort_values("date", kind="stable")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # volume/amount 用 0 填充（停牌日可能缺失）
    out["volume"] = out["volume"].fillna(0.0)
    out["amount"] = out["amount"].fillna(0.0)
    out["adj_factor"] = out["adj_factor"].fillna(1.0)

    return out

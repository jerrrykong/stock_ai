"""数据层：多源 fallback 拉取、parquet 缓存、增量刷新。"""

from __future__ import annotations

from stock_ai.data.cache import DataCache
from stock_ai.data.loader import StockDataLoader
from stock_ai.data.schema import COLUMNS, DTYPES, SymbolInfo, validate_frame

__all__: list[str] = [
    "COLUMNS",
    "DTYPES",
    "SymbolInfo",
    "validate_frame",
    "DataCache",
    "StockDataLoader",
]

"""数据源抽象基类（端口）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["DataSource"]


class DataSource(ABC):
    """数据源端口。

    子类必须实现 :meth:`fetch`；可选覆盖 :meth:`probe`。
    """

    #: 数据源名称（小写，如 ``"akshare"``）
    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str) -> "Any":
        """拉取指定股票在 ``[start, end]`` 区间的日线。

        Args:
            symbol: 股票代码（6 位，如 ``"600519"``）。
            start: 起始日期 ``YYYYMMDD``。
            end: 结束日期 ``YYYYMMDD``。

        Returns:
            规范化的 ``pandas.DataFrame``（列见 :data:`stock_ai.data.schema.COLUMNS`）。

        Raises:
            DataSourceError: 拉取或解析失败。
        """
        raise NotImplementedError

    def probe(self, symbol: str) -> bool:
        """探测数据源是否可用（默认实现为尝试拉取最近 5 天）。

        Args:
            symbol: 股票代码。

        Returns:
            可用返回 ``True``，否则 ``False``（不抛异常）。
        """
        try:
            df = self.fetch(symbol, "20240101", "20240110")
            return df is not None and len(df) > 0
        except Exception:
            return False

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self.__class__.__name__} name={self.name!r}>"

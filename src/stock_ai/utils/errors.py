"""统一异常体系（K6）。

所有业务异常继承 :class:`StockAIError`；子类语义：
    - :class:`DataSourceError`  数据源调用/解析失败
    - :class:`ConfigError`      配置缺失或非法
    - :class:`ArtifactError`    文件契约缺失/损坏
    - :class:`NotTradableError` 不可成交（仅环境内部使用，不外抛）
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "StockAIError",
    "DataSourceError",
    "NotTradableError",
    "ConfigError",
    "ArtifactError",
]


class StockAIError(Exception):
    """所有 stock_ai 业务异常的基类。

    Args:
        message: 人类可读的错误信息。
        context: 附加上下文（键值对），便于日志与排障。
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class DataSourceError(StockAIError):
    """数据源相关异常（网络失败、返回结构异常、无数据）。"""


class NotTradableError(StockAIError):
    """不可成交异常（T+1 冻结、涨跌停、停牌）。仅环境内部使用。"""


class ConfigError(StockAIError):
    """配置加载/解析/校验异常。"""


class ArtifactError(StockAIError):
    """文件契约（parquet/npz/zip/json）缺失或损坏。"""

"""编排层专用异常（增量）。

``ScaleError`` 继承既有 :class:`~stock_ai.utils.errors.StockAIError` 异常族，
因此 ``cli.main.main()`` 的兜底逻辑（``StockAIError -> [错误] ... -> exit(1)``）
**自动生效**，无需改动主入口。
"""

from __future__ import annotations

from stock_ai.utils.errors import StockAIError

__all__ = ["ScaleError"]


class ScaleError(StockAIError):
    """大规模训练/检测编排层的业务异常。

    覆盖：名单缺失、单元计划缺失、状态不一致、控制信号非法、
    单元级前置条件缺失（数据/模型/归一化）等**编排**语义错误。

    Args:
        message: 人类可读的错误信息。
        context: 附加上下文（键值对），便于日志与排障。
    """

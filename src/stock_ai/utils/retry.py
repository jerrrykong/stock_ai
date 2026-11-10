"""重试工具：指数退避装饰器。

用于所有外部数据源调用（AKShare / 腾讯 / 东方财富）。
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from stock_ai.utils.logging import LoggerFactory

__all__ = ["retry_with_backoff"]

logger = LoggerFactory.get(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    times: int = 3,
    backoff: float = 1.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.0,
) -> Callable[[F], F]:
    """指数退避重试装饰器。

    第 k 次失败后等待 ``backoff ** k`` 秒（k 从 1 开始）；``times`` 为**总尝试次数**。

    Args:
        times: 总尝试次数（``times<=1`` 表示不重试）。
        backoff: 退避基数（秒）。
        exceptions: 需要重试的异常类型元组。
        jitter: 每次等待附加的随机抖动上界（秒）。

    Returns:
        装饰器。
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = max(1, int(times))
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: B902 - 需按需重试
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    delay = float(backoff) ** attempt
                    if jitter > 0:
                        import random

                        delay += random.uniform(0.0, jitter)
                    logger.warning(
                        "调用 %s 第 %d/%d 次失败: %s；%.2fs 后重试",
                        getattr(func, "__name__", repr(func)),
                        attempt,
                        attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator

"""日志工厂（K5）。

统一控制台（UTF-8）+ 文件双通道。格式：
``[%(asctime)s] %(levelname)-7s %(name)s | %(message)s``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["LoggerFactory"]

_LOG_FORMAT: str = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


class LoggerFactory:
    """日志器工厂，保证重复调用不会重复挂载 handler。"""

    @staticmethod
    def setup(
        name: str,
        log_dir: Path | None = None,
        level: str = "INFO",
        log_file: str = "train.log",
    ) -> logging.Logger:
        """创建/获取一个带控制台（+可选文件）双通道的 logger。

        Args:
            name: logger 名称，惯例传 ``__name__``。
            log_dir: 若提供，则在同一目录下写入 ``log_file``（UTF-8）。
            level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR）。
            log_file: 日志文件名。

        Returns:
            配置完成的 :class:`logging.Logger`。
        """
        global _configured

        logger = logging.getLogger(name)
        lvl = _parse_level(level)
        logger.setLevel(lvl)
        logger.propagate = False

        # 控制台 handler（只挂一个）
        if not any(isinstance(h, logging.StreamHandler) and
                   getattr(h, "_stockai_console", False) for h in logger.handlers):
            try:
                stream = sys.stdout
                # Python 3.7+ 支持 reconfigure；重定向场景下可能失败，忽略即可
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # pragma: no cover
                pass
            console = logging.StreamHandler(stream)
            console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            console._stockai_console = True  # type: ignore[attr-defined]
            logger.addHandler(console)

        # 文件 handler
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fpath = log_dir / log_file
            already = any(
                isinstance(h, logging.FileHandler)
                and Path(getattr(h, "baseFilename", "")) == fpath.resolve()
                for h in logger.handlers
            )
            if not already:
                fh = logging.FileHandler(str(fpath), encoding="utf-8", mode="a")
                fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
                logger.addHandler(fh)

        _configured = True
        return logger

    @staticmethod
    def get(name: str) -> logging.Logger:
        """获取模块级 logger（K5 约定：``logging.getLogger(__name__)``）。

        Args:
            name: 通常传 ``__name__``。

        Returns:
            :class:`logging.Logger`，未配置时提供一个基础控制台输出。
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger = LoggerFactory.setup(name)
        return logger


def _parse_level(level: str) -> int:
    """把字符串级别转为 logging 常量，非法值退回 INFO。"""
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)

"""包入口：``python -m stock_ai <cmd>`` -> :func:`stock_ai.cli.main.main`。

在导入 CLI 之前先行兜底设置控制台编码，避免中文日志乱码。
"""

from __future__ import annotations

import os


def _prepare_console() -> None:
    """在 Windows 无头/重定向环境下兜底设置 UTF-8 控制台编码（K13）。"""
    try:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:  # pragma: no cover - 环境极端异常时不应阻断启动
        pass


def main() -> None:
    """CLI 主入口。"""
    _prepare_console()
    from stock_ai.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()

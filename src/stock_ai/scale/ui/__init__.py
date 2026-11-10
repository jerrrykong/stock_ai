"""本机 UI 子包（stdlib ``http.server`` + 单页 HTML/JS）。

**零新增依赖**：仅使用标准库 ``http.server`` / ``socketserver`` / ``urllib.parse`` /
``json`` / ``subprocess`` / ``webbrowser`` 与 ``threading``。

**无自动重启**：UI 仅在用户显式点击「启动」时 ``Popen`` 一次子进程；不注册 Windows
服务 / 计划任务 / 守护进程，UI 退出即彻底结束。
"""

from __future__ import annotations

from stock_ai.scale.ui.server import ScaleRequestHandler, ScaleUI

__all__ = ["ScaleUI", "ScaleRequestHandler"]

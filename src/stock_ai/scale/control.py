"""控制信号文件（``control.json``）读写与消费。

协议（架构 §8.4）::

    { "phase": "train", "cmd": "pause|resume|stop", "requested_at": "YYYYMMDD_HHMMSS" }

- 编排器**仅在单元边界**轮询；``phase`` 不匹配的命令忽略。
- UI 的 ``start`` 不写 ``control.json``（直接起子进程）；``pause/resume/stop`` 才写。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from stock_ai.scale.atomic import atomic_write_json
from stock_ai.scale.paths import ScalePaths
from stock_ai.utils.io import read_json

__all__ = ["ControlSignal", "ControlStore", "VALID_CMDS"]

#: 合法控制指令。
VALID_CMDS: frozenset[str] = frozenset({"pause", "resume", "stop"})


class ControlSignal:
    """一条控制信号。

    Args:
        phase: 目标阶段（``train|detect``）。
        cmd: ``pause|resume|stop``。
        requested_at: 请求时间戳。
    """

    def __init__(self, phase: str, cmd: str, requested_at: str = "") -> None:
        """初始化。"""
        self.phase: str = str(phase)
        self.cmd: str = str(cmd)
        self.requested_at: str = requested_at or datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> dict:
        """转为普通 dict。"""
        return {
            "phase": self.phase,
            "cmd": self.cmd,
            "requested_at": self.requested_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSignal":
        """从 dict 构造。"""
        return cls(
            phase=str(d.get("phase", "")),
            cmd=str(d.get("cmd", "")),
            requested_at=str(d.get("requested_at", "")),
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ControlSignal(phase={self.phase!r}, cmd={self.cmd!r})"


class ControlStore:
    """控制信号文件读写（原子写）。

    Args:
        paths: 路径集合。
    """

    def __init__(self, paths: ScalePaths) -> None:
        """初始化。"""
        self.paths: ScalePaths = paths
        self.path: Path = paths.control_json()

    # ------------------------------------------------------------------ #
    def poll(self, phase: str) -> str:
        """读取控制指令。

        Args:
            phase: 当前阶段。

        Returns:
            ``""``（无指令/文件缺失/损坏/phase 不匹配）或 ``pause|resume|stop``。
        """
        p = self.path
        if not p.exists():
            return ""
        try:
            data = read_json(p)
        except Exception:  # noqa: BLE001 - 损坏信号忽略
            return ""
        if not isinstance(data, dict):
            return ""
        sig = ControlSignal.from_dict(data)
        if sig.phase and sig.phase != phase:
            return ""
        return sig.cmd if sig.cmd in VALID_CMDS else ""

    # ------------------------------------------------------------------ #
    def write(self, phase: str, cmd: str) -> None:
        """原子写入控制指令。

        Raises:
            ValueError: ``cmd`` 非法。
        """
        if cmd not in VALID_CMDS:
            raise ValueError(f"非法控制指令: {cmd!r}（可选 {sorted(VALID_CMDS)}）")
        atomic_write_json(ControlSignal(phase, cmd).to_dict(), self.path)

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """删除控制文件（幂等）。"""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass

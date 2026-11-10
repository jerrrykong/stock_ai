"""原子写盘工具（状态/契约文件专用）。

**为什么单独放在 ``scale/atomic.py``**：既有 ``utils/io.py`` 被 221 项测试覆盖，
本增量遵循「零回归 / 最小改动」原则，**不动** ``utils/io.py``；原子写盘能力在
本模块内独立实现。

写盘策略（架构 §8.8）：``path.parent.mkdir`` → 写 ``<name>.<pid>.tmp`` →
``flush + fsync`` → ``os.replace(tmp, path)``。``os.replace`` 在同一文件系统内
为原子操作，从而保证**中断不产生半截 JSON**、**并发读不会读到写一半的文件**。

Windows 加固（并发读）：目标文件被并发读取（``open()->read->close``）时，
``os.replace`` 会因共享冲突抛 ``PermissionError``/``OSError(winerror=32)``。
此类错误是瞬时的，故 :func:`_fsync_and_replace` 对其做**有界高频小睡重试**
（见 ``_REPLACE_*`` 常量），避免单次瞬时占用使整批 ``scale train``/``scale detect``
中止；无冲突时首次即成功、零额外延迟。
"""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_json", "atomic_write_text"]

#: ``os.replace`` 对「Windows 并发读瞬时占用」的最大重试次数（安全上限）。
_REPLACE_MAX_ATTEMPTS: int = 500
#: 每次重试之间的小睡（秒）：足够小，以便在读者 open/close 的间隙中命中。
_REPLACE_SLEEP_SEC: float = 0.002
#: 重试总时长上限（秒）：避免读方长期占用时无限阻塞；超时后抛出。
_REPLACE_DEADLINE_SEC: float = 1.0


def _tmp_path(path: Path) -> Path:
    """构造同目录下的临时文件路径 ``<name>.<pid>.tmp``。"""
    p = Path(path)
    return p.parent / f"{p.name}.{os.getpid()}.tmp"


def _is_transient_replace_error(exc: BaseException) -> bool:
    """判断 ``os.replace`` 失败是否为「Windows 并发读 / 杀软瞬时占用」，可安全重试。

    ``os.replace`` 在 Windows 上走 ``MoveFileEx(MOVEFILE_REPLACE_EXISTING)``；
    当目标文件此刻被其它进程/线程以**不允许删除共享**的方式打开（例如 Python 内建
    ``open()`` 正在读取 ``state/*.json``）时会失败：

      - ``PermissionError``（``[WinError 5] 拒绝访问``）
      - ``OSError`` 且 ``winerror == 32``（``ERROR_SHARING_VIOLATION``）

    此类错误是**瞬时**的：待读方释放句柄后重试即可成功，属正常并发现象而非数据错误。

    Args:
        exc: ``os.replace`` 抛出的异常。

    Returns:
        属可重试的瞬时错误返回 ``True``，否则 ``False``。
    """
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in (5, 32):
            return True
        # 跨平台兜底：EACCES / EPERM / EBUSY 亦属瞬时占用
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EBUSY):
            return True
    return False


def _fsync_and_replace(tmp: Path, path: Path) -> Path:
    """把已落盘的临时文件原子替换到目标路径（对 Windows 并发读做有界重试）。

    说明：写入阶段已在文件句柄上 ``flush + fsync``；此处仅做 ``os.replace``。
    （Windows 上对只读句柄再 ``fsync`` 会报 ``Bad file descriptor``，故不重复。）

    Windows 上 ``os.replace`` 会因目标文件被并发读取（共享冲突）而抛
    ``PermissionError`` / ``OSError(winerror=32)``。读者对文件的占用是**高频、
    短暂**的（``open()->read->close``），故采用**高频小睡重试**策略：
    以 :data:`_REPLACE_SLEEP_SEC`（2ms）为间隔重试，直到成功或到达
    :data:`_REPLACE_DEADLINE_SEC`（1s）总时长上限。无并发冲突时**首次即成功、
    零延迟**；存在冲突时可在读者 open/close 的间隙中命中，从而保证
    ``scale train``/``scale detect`` 不因单次瞬时占用而中止整批。

    Args:
        tmp: 已落盘的临时文件路径。
        path: 目标路径。

    Returns:
        目标路径。

    Raises:
        OSError: 重试耗尽（超时 / 超过上限）或不可重试的写盘错误。
    """
    tmp = Path(tmp)
    target = Path(path)
    last_exc: OSError | None = None
    deadline = time.monotonic() + _REPLACE_DEADLINE_SEC

    for _attempt in range(_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(str(tmp), str(target))
            return target
        except OSError as exc:
            last_exc = exc
            if _is_transient_replace_error(exc) and time.monotonic() < deadline:
                # 让出 GIL 片刻，等待读方释放句柄后立即重试
                time.sleep(_REPLACE_SLEEP_SEC)
                continue
            break

    # 重试耗尽 / 超时 / 不可重试错误：清理临时文件，避免残留半截数据，然后抛出
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    assert last_exc is not None  # 循环至少执行一次，正常路径已 return
    raise last_exc


def atomic_write_json(obj: Any, path: Path, indent: int = 2) -> Path:
    """原子写出 JSON（UTF-8，``ensure_ascii=False``）。

    Args:
        obj: 可 JSON 序列化对象（``Path`` / ``numpy`` 由兜底转换处理）。
        path: 目标路径。
        indent: 缩进。

    Returns:
        目标路径。

    Raises:
        OSError: 目录不可写或替换失败。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    text = json.dumps(obj, ensure_ascii=False, indent=indent, default=_json_default)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    return _fsync_and_replace(tmp, target)


def atomic_write_text(text: str, path: Path, encoding: str = "utf-8") -> Path:
    """原子写出纯文本。

    Args:
        text: 文本内容。
        path: 目标路径。
        encoding: 编码（默认 UTF-8）。

    Returns:
        目标路径。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    with tmp.open("w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    return _fsync_and_replace(tmp, target)


def _json_default(obj: Any) -> Any:
    """JSON 序列化兜底：``Path`` / numpy / datetime。"""
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:  # pragma: no cover - numpy 必装，仅保险
        pass
    from datetime import datetime

    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

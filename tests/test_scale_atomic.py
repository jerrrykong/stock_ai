"""原子写盘对 Windows 并发读（共享冲突）的有界重试用例（P1 回归）。

背景：``os.replace`` 在 Windows 上会因目标文件被并发读取而抛
``PermissionError`` / ``OSError(winerror=32)``；这正是 QA 复现的
``scale train`` 整批中止根因。修复后需保证：

- 瞬时共享冲突 → 小退避重试后成功（不中止整批）；
- 不可重试错误 → **不重试**、立即抛出且清理临时文件；
- 持续冲突 → 重试耗尽后抛出（有界，不无限阻塞）。
"""

from __future__ import annotations

import json
import os

import pytest

from stock_ai.scale import atomic as atomic_mod
from stock_ai.scale.atomic import (
    _REPLACE_MAX_ATTEMPTS,
    atomic_write_json,
    atomic_write_text,
)


def _set_winerror(exc: OSError, code: int) -> OSError:
    """给 ``OSError`` 注入 ``winerror`` 属性（跨平台模拟 Windows 语义）。"""
    exc.winerror = code  # type: ignore[attr-defined]
    return exc


def test_replace_retries_on_transient_permission_error(tmp_path, monkeypatch):
    """前两次抛 WinError 5，第三次成功 → 重试生效，产物正确、无残留 tmp。"""
    target = tmp_path / "x.json"
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _set_winerror(PermissionError(13, "Access denied"), 5)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    atomic_write_json({"a": 1, "中文": "值"}, target)

    assert calls["n"] == 3
    assert json.loads(target.read_text(encoding="utf-8"))["中文"] == "值"
    assert list(tmp_path.glob("*.tmp")) == []


def test_replace_retries_on_sharing_violation(tmp_path, monkeypatch):
    """WinError 32（ERROR_SHARING_VIOLATION）同样可重试。"""
    target = tmp_path / "s.json"
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _set_winerror(OSError(13, "sharing violation"), 32)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    atomic_write_text("hello", target)

    assert calls["n"] == 2
    assert target.read_text(encoding="utf-8") == "hello"


def test_replace_does_not_retry_non_transient(tmp_path, monkeypatch):
    """非瞬时错误（如 ENOENT）立即抛出、不重试，并清理临时文件。"""
    target = tmp_path / "n.json"
    calls = {"n": 0}

    def boom(src, dst):
        calls["n"] += 1
        raise OSError(2, "No such file or directory")  # errno=ENOENT，不可重试

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json({"a": 1}, target)

    assert calls["n"] == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_replace_retry_exhausted_raises_and_cleans(tmp_path, monkeypatch):
    """持续共享冲突 → 到达时长上限后有界抛出，且不残留临时文件。"""
    target = tmp_path / "e.json"
    calls = {"n": 0}

    def always(src, dst):
        calls["n"] += 1
        raise _set_winerror(PermissionError(13, "Access denied"), 32)

    # 缩短总时长上限，使测试快速到达上限
    monkeypatch.setattr(atomic_mod, "_REPLACE_DEADLINE_SEC", 0.05)
    monkeypatch.setattr(atomic_mod, "_REPLACE_SLEEP_SEC", 0.001)
    monkeypatch.setattr(os, "replace", always)
    with pytest.raises(PermissionError):
        atomic_write_json({"a": 1}, target)

    assert calls["n"] > 1                    # 确实重试了多次
    assert calls["n"] <= _REPLACE_MAX_ATTEMPTS
    assert list(tmp_path.glob("*.tmp")) == []  # 临时文件被清理


def test_is_transient_replace_error_classification():
    """分类函数：瞬时错误 → True，其它 → False。"""
    from stock_ai.scale.atomic import _is_transient_replace_error

    assert _is_transient_replace_error(_set_winerror(PermissionError(), 5)) is True
    assert _is_transient_replace_error(_set_winerror(OSError(), 32)) is True
    assert _is_transient_replace_error(OSError(2, "enoent")) is False
    assert _is_transient_replace_error(ValueError("x")) is False

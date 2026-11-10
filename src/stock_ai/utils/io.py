"""IO 工具：parquet / npz / json / yaml 读写，时间戳与文件名清洗（K2/K12/K13）。

- 所有文件读写显式 ``encoding="utf-8"``。
- 路径一律 ``pathlib.Path``，写文件时转 ``str(p)``。
- 时间戳统一 ``YYYYMMDD_HHMMSS``。
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.utils.errors import ArtifactError

__all__ = [
    "read_parquet",
    "write_parquet",
    "write_npz",
    "read_npz",
    "read_json",
    "write_json",
    "read_yaml",
    "write_yaml",
    "timestamp",
    "sanitize_filename",
    "ensure_utf8_console",
    "zip_write_files",
    "zip_read_files",
]

_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def ensure_utf8_console() -> None:
    """兜底设置控制台 UTF-8 编码（K13）。

    在 Windows 上重定向 stdout/stderr 时也不会因中文而抛 UnicodeEncodeError。
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass


def timestamp() -> str:
    """返回 ``YYYYMMDD_HHMMSS`` 格式时间戳（K2）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str) -> str:
    """清洗 Windows 非法文件名字符（K12）。

    Args:
        name: 原始名称（如股票代码 ``600519.SH``）。

    Returns:
        可安全用于文件系统的名称。
    """
    cleaned = _ILLEGAL_FILENAME_CHARS.sub("_", str(name)).strip(" .")
    return cleaned or "unnamed"


# --------------------------------------------------------------------------- #
# Parquet
# --------------------------------------------------------------------------- #
def read_parquet(path: Path) -> "Any":
    """读取 parquet 文件。

    Args:
        path: 文件路径。

    Returns:
        ``pandas.DataFrame``。

    Raises:
        ArtifactError: 文件缺失或损坏。
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise ArtifactError(
            f"parquet 文件不存在: {p}（请先执行 `stock_ai data fetch`）",
            context={"path": str(p)},
        )
    try:
        return pd.read_parquet(str(p))
    except Exception as exc:
        raise ArtifactError(
            f"parquet 读取失败: {p} ({exc})", context={"path": str(p)}
        ) from exc


def write_parquet(df: "Any", path: Path) -> Path:
    """写出 parquet（自动创建父目录）。

    Args:
        df: ``pandas.DataFrame``。
        path: 目标路径。

    Returns:
        目标路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(p), index=False, engine="pyarrow")
    return p


# --------------------------------------------------------------------------- #
# NPZ
# --------------------------------------------------------------------------- #
def write_npz(path: Path, **arrays: "np.ndarray") -> Path:
    """写出 npz（自动创建父目录）。

    Args:
        path: 目标路径。
        **arrays: 键值对，key 为 npz 内的键名（见 K7）。

    Returns:
        目标路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), **arrays)
    return p


def read_npz(path: Path) -> dict[str, "np.ndarray"]:
    """读取 npz 为普通 dict。

    Args:
        path: 文件路径。

    Returns:
        键值对 dict。

    Raises:
        ArtifactError: 文件缺失或损坏。
    """
    p = Path(path)
    if not p.exists():
        raise ArtifactError(
            f"npz 文件不存在: {p}（请先执行 `stock_ai sample build`）",
            context={"path": str(p)},
        )
    try:
        with np.load(str(p), allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    except Exception as exc:
        raise ArtifactError(f"npz 读取失败: {p} ({exc})") from exc


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict:
    """读取 JSON（UTF-8）。

    Args:
        path: 文件路径。

    Returns:
        解析后的 dict。

    Raises:
        ArtifactError: 文件缺失或损坏。
    """
    import json

    p = Path(path)
    if not p.exists():
        raise ArtifactError(
            f"JSON 文件不存在: {p}（请先执行上游命令生成）", context={"path": str(p)}
        )
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        raise ArtifactError(f"JSON 读取失败: {p} ({exc})") from exc


def write_json(obj: dict, path: Path, indent: int = 2) -> Path:
    """写出 JSON（UTF-8，``ensure_ascii=False``）。

    Args:
        obj: 可序列化对象。
        path: 目标路径。
        indent: 缩进。

    Returns:
        目标路径。
    """
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=_json_default)
    return p


def _json_default(obj: Any) -> Any:
    """JSON 序列化兜底：numpy / Path / datetime。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #
def read_yaml(path: Path) -> dict:
    """读取 YAML（UTF-8）。"""
    import yaml

    p = Path(path)
    if not p.exists():
        raise ArtifactError(f"YAML 文件不存在: {p}", context={"path": str(p)})
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_yaml(obj: dict, path: Path) -> Path:
    """写出 YAML（UTF-8，允许中文）。"""
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, allow_unicode=True, sort_keys=False)
    return p


# --------------------------------------------------------------------------- #
# ZIP（模型文件契约）
# --------------------------------------------------------------------------- #
def zip_write_files(path: Path, files: dict[str, bytes]) -> Path:
    """把内存中的多个文件打包成一个 zip。

    Args:
        path: 目标 zip 路径。
        files: ``{归档内文件名: 字节内容}``。

    Returns:
        目标路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(p), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return p


def zip_read_files(path: Path) -> dict[str, bytes]:
    """读取 zip 内全部文件到内存。

    Args:
        path: zip 路径。

    Returns:
        ``{归档内文件名: 字节内容}``。

    Raises:
        ArtifactError: 文件缺失或损坏。
    """
    p = Path(path)
    if not p.exists():
        raise ArtifactError(
            f"模型文件不存在: {p}（请先执行 `stock_ai train`）", context={"path": str(p)}
        )
    try:
        with zipfile.ZipFile(str(p), "r") as zf:
            return {n: zf.read(n) for n in zf.namelist()}
    except Exception as exc:
        raise ArtifactError(f"模型 zip 读取失败: {p} ({exc})") from exc

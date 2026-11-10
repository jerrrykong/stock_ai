"""数据哈希工具（K16）。

``data_hash`` = 对所有参与构建的 raw parquet 文件字节按文件名排序后依次
``sha256.update``。它是复现链的锚点：sample -> meta -> model sidecar -> backtest。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["sha256_file", "sha256_files", "hash_arrays", "hash_dataframes"]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算单个文件的 sha256。

    Args:
        path: 文件路径。
        chunk_size: 分块大小（字节）。

    Returns:
        十六进制摘要字符串。
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    """按文件名排序后依次 update，得到稳定的组合摘要（K16）。

    Args:
        paths: 参与哈希的文件路径列表。

    Returns:
        组合 sha256 十六进制摘要；空列表返回空串摘要。

    Raises:
        FileNotFoundError: 任一文件不存在。
    """
    h = hashlib.sha256()
    ordered = sorted((Path(p) for p in paths), key=lambda p: p.name)
    for p in ordered:
        if not p.exists():
            raise FileNotFoundError(f"哈希输入文件不存在: {p}")
        h.update(p.name.encode("utf-8"))
        h.update(b"\x00")
        h.update(Path(p).read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


def hash_arrays(arrays: dict[str, "np.ndarray"]) -> str:
    """对一组 numpy 数组计算稳定摘要（按 key 排序）。

    Args:
        arrays: ``{key: ndarray}``。

    Returns:
        sha256 十六进制摘要。
    """
    h = hashlib.sha256()
    for key in sorted(arrays.keys()):
        arr = np.ascontiguousarray(np.asarray(arrays[key]))
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def hash_dataframes(dfs: dict[str, Any]) -> str:
    """对一组 DataFrame 计算稳定摘要（基于 parquet 字节）。

    Args:
        dfs: ``{name: DataFrame}``。

    Returns:
        sha256 十六进制摘要。
    """
    import io

    h = hashlib.sha256()
    for key in sorted(dfs.keys()):
        buf = io.BytesIO()
        df = dfs[key]
        df.to_parquet(buf, index=False, engine="pyarrow")
        h.update(key.encode("utf-8"))
        h.update(buf.getvalue())
    return h.hexdigest()

"""哈希稳定性单测（K16）。"""

from __future__ import annotations

import numpy as np

from stock_ai.utils.hashing import hash_arrays, sha256_file, sha256_files


def test_same_data_same_hash(tmp_path):
    """同数据同 hash；异数据异 hash。"""
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_bytes(b"hello world")
    f2.write_bytes(b"hello world")
    assert sha256_file(f1) == sha256_file(f2)

    f2.write_bytes(b"hello world!")
    assert sha256_file(f1) != sha256_file(f2)


def test_sha256_files_order_independent(tmp_path):
    """多文件组合哈希与传入顺序无关（按文件名排序）。"""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"AAA")
    b.write_bytes(b"BBB")
    assert sha256_files([a, b]) == sha256_files([b, a])


def test_sha256_files_detects_change(tmp_path):
    """内容变化会导致 hash 变化。"""
    a = tmp_path / "a.bin"
    a.write_bytes(b"AAA")
    h1 = sha256_files([a])
    a.write_bytes(b"AAB")
    assert sha256_files([a]) != h1


def test_hash_arrays_stable():
    """numpy 数组哈希稳定。"""
    arr = {"x": np.arange(10, dtype="float64"), "y": np.ones(5)}
    assert hash_arrays(arr) == hash_arrays(dict(reversed(list(arr.items()))))


def test_hash_arrays_detects_change():
    """数组内容变化 -> hash 变化。"""
    a = {"x": np.arange(10)}
    b = {"x": np.arange(10) + 1}
    assert hash_arrays(a) != hash_arrays(b)

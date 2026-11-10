"""新浪数据源适配器测试。

覆盖：字段契约与区间裁剪、volume 单位（股，不 ×100）、成交额估算、
非 ``none`` 复权口径明确拒绝、异常/空响应防御，以及 loader 注册。

新浪接口是腾讯被风控（HTTP 501）时的可靠兜底，其正确性直接影响大规模
拉取的数据口径，故单列回归。
"""

from __future__ import annotations

import json

import pytest

from stock_ai.data.schema import COLUMNS
from stock_ai.data.sources.sina_source import SinaSource, _load_rows
from stock_ai.utils.errors import DataSourceError


class _Resp:
    """最小 HTTP 响应替身。"""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_sina(monkeypatch, text: str) -> None:
    """把 ``requests.get`` 替换为返回固定文本的假实现。"""
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(text))


#: 覆盖区间外的点（2005 / 2026）用于验证裁剪
_FAKE = json.dumps(
    [
        {"day": "2005-12-30", "open": "10.0", "high": "11.0", "low": "9.5",
         "close": "10.5", "volume": "1000"},
        {"day": "2006-01-04", "open": "10.5", "high": "11.5", "low": "10.2",
         "close": "11.0", "volume": "2000"},
        {"day": "2026-10-09", "open": "20.0", "high": "21.0", "low": "19.0",
         "close": "20.5", "volume": "3000"},
    ]
)


def test_fetch_filters_range_and_keeps_share_volume(monkeypatch):
    """区间裁剪正确；volume 保持「股」（新浪口径），并补出 amount/adj_factor。"""
    _patch_sina(monkeypatch, _FAKE)
    src = SinaSource(adjust="none", retry_times=1)

    df = src.fetch("600519", "20060101", "20251231")

    assert list(df["date"].dt.strftime("%Y%m%d")) == ["20060104"]   # 2005/2026 被裁掉
    row = df.iloc[0]
    assert row["close"] == 11.0
    assert row["volume"] == 2000.0            # 不乘 100（与腾讯源的「手→股 ×100」不同）
    assert row["amount"] == 11.0 * 2000.0     # 接口无成交额，按 close*volume 估算
    assert row["adj_factor"] == 1.0
    assert list(df.columns) == COLUMNS


def test_adjust_non_none_is_rejected(monkeypatch):
    """非不复权口径必须明确拒绝（交由 fallback 链改用其它源），不得静默给错口径。"""
    _patch_sina(monkeypatch, _FAKE)
    src = SinaSource(adjust="qfq", retry_times=1)

    with pytest.raises(DataSourceError, match="仅提供不复权"):
        src.fetch("600519", "20060101", "20251231")


def test_empty_response_raises():
    """空响应抛 DataSourceError。"""
    with pytest.raises(DataSourceError):
        _load_rows("")


def test_non_json_body_raises():
    """非 JSON（如风控页 HTML）抛 DataSourceError，不得静默当作空数据。"""
    with pytest.raises(DataSourceError):
        _load_rows("<html><body>501 Not Implemented</body></html>")


def test_null_body_returns_empty():
    """``null`` 视为无数据（返回空列表，由调用方决定是否报错）。"""
    assert _load_rows("null") == []
    assert _load_rows("[]") == []


def test_bom_prefixed_json_parsed():
    """带 BOM 的 JSON 应能正常解析。"""
    rows = _load_rows("\ufeff" + _FAKE)
    assert len(rows) == 3


@pytest.mark.parametrize(
    ("symbol", "expect"),
    [
        ("600519", "sh600519"),
        ("688981", "sh688981"),
        ("000001", "sz000001"),
        ("300750", "sz300750"),
        ("600519.SH", "sh600519"),
    ],
)
def test_prefix(symbol, expect):
    """市场前缀推导（与腾讯源保持一致）。"""
    assert SinaSource._prefix(symbol) == expect


def test_sina_registered_in_loader():
    """新浪源必须已注册进 loader 的单一真源与 fallback 链。"""
    from stock_ai.data.loader import SUPPORTED_SOURCES, _FALLBACK_CHAIN

    assert "sina" in SUPPORTED_SOURCES
    assert "sina" in _FALLBACK_CHAIN

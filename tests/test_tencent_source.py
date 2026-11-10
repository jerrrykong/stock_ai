"""腾讯行情数据源单测：纯 JSON / JSONP 解析、动态 count、空数据抛错（全程离线）。

覆盖回归点（见 tencent_source 缺陷修复）：
1. 不带 ``_var`` 时接口返回纯 JSON，应能正常解析出合法 frame；
2. 带 ``kline_day=`` JSONP 前缀时也能解析；
3. 无 rows / 空数据时抛 :class:`DataSourceError`；
4. K 线根数按区间动态估算（下限 320、上限 10000）。

所有用例均 mock 掉 ``requests.get``，**禁止任何真实网络请求**。
"""

from __future__ import annotations

import json

import pytest
import requests

from stock_ai.data.schema import COLUMNS
from stock_ai.data.sources.tencent_source import (
    TencentSource,
    _MAX_BARS_PER_REQ,
    _estimate_bars,
    _load_json,
)
from stock_ai.utils.errors import DataSourceError


class _FakeResp:
    """最小化的假 HTTP 响应对象。"""

    def __init__(self, text: str, status: int = 200) -> None:
        """初始化。"""
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        """状态码 >=400 时抛 HTTPError。"""
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):  # noqa: ANN201 - 故意保留以验证新实现不再依赖它
        """旧实现依赖 resp.json()；这里让它直接失败以证明已不再调用。"""
        raise AssertionError("新实现不应再调用 resp.json()")


def _rows(n: int = 5) -> list[list[str]]:
    """构造 n 行腾讯格式的日线记录。"""
    return [
        [f"2024-01-{i + 2:02d}", "10.0", "10.5", "10.8", "9.9", "1234.0"]
        for i in range(n)
    ]


def _payload(rows: list[list[str]], code: str = "sh600519") -> str:
    """构造腾讯纯 JSON 响应文本。"""
    return json.dumps({"code": 0, "msg": "", "data": {code: {"qfqday": rows}}})


# ---------------------------------------------------------------- 解析
def test_pure_json_parses(monkeypatch):
    """① 纯 JSON 正常解析并得到合法 frame。"""
    src = TencentSource(adjust="qfq", retry_times=1)
    body = _payload(_rows(5))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(body))

    df = src.fetch("600519", "20240101", "20240110")

    assert list(df.columns) == COLUMNS
    assert len(df) == 5
    # 成交量 手 -> 股（×100）
    assert float(df["volume"].iloc[0]) == 1234.0 * 100.0


def test_jsonp_prefix_parses(monkeypatch):
    """② 带 ``kline_day=`` JSONP 前缀也能解析。"""
    src = TencentSource(adjust="qfq", retry_times=1)
    body = "kline_day=" + _payload(_rows(3)) + ";"
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(body))

    df = src.fetch("600519", "20240101", "20240110")

    assert list(df.columns) == COLUMNS
    assert len(df) == 3


def test_empty_rows_raises(monkeypatch):
    """③ 无 rows / 空数据抛 DataSourceError（不重试时立即抛）。"""
    src = TencentSource(adjust="qfq", retry_times=1)
    body = json.dumps({"code": 0, "data": {"sh600519": {"qfqday": []}}})
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(body))

    with pytest.raises(DataSourceError):
        src.fetch("600519", "20240101", "20240110")


def test_data_list_shape_raises(monkeypatch):
    """④ data 为 list（``param error`` 形态）应抛 DataSourceError 而非 AttributeError。"""
    src = TencentSource(adjust="qfq", retry_times=1)
    body = json.dumps({"code": 0, "msg": "param error", "data": []})
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(body))

    with pytest.raises(DataSourceError):
        src.fetch("600519", "20240101", "20240110")


def test_pagination_backfills_long_range(monkeypatch):
    """⑤ 长区间按页向前回填：拼齐多页、升序，且每页 count <= _MAX_BARS_PER_REQ。"""
    page1 = [[f"2024-12-{d:02d}", "10", "11", "12", "9", "100"] for d in (29, 30, 31)]
    page2 = [[f"2024-12-{d:02d}", "10", "11", "12", "9", "100"] for d in (26, 27, 28)]
    calls = {"n": 0, "counts": []}

    def _fake_get(url, params=None, headers=None, timeout=None):  # noqa: ANN001
        calls["n"] += 1
        calls["counts"].append(int(str(params["param"]).split(",")[4]))
        pages = {1: page1, 2: page2}
        return _FakeResp(_payload(pages.get(calls["n"], [])))

    monkeypatch.setattr(requests, "get", _fake_get)
    src = TencentSource(adjust="qfq", retry_times=1)
    df = src.fetch("600519", "20240101", "20241231")

    assert calls["n"] == 3                     # page1 -> page2 -> 空页停止
    assert all(c <= _MAX_BARS_PER_REQ for c in calls["counts"])
    assert len(df) == 6
    assert df["date"].is_monotonic_increasing


# ---------------------------------------------------------------- count 估算
def test_estimate_bars_lower_bound():
    """短区间取下限 320。"""
    assert _estimate_bars("20240101", "20240110") == 320


def test_estimate_bars_long_range():
    """长区间（2012~2024）估算值 >320 且不超过单页上限。"""
    count = _estimate_bars("20120101", "20241231")
    assert 320 < count <= _MAX_BARS_PER_REQ


def test_estimate_bars_bad_input_fallback():
    """无法解析的日期回退 320。"""
    assert _estimate_bars("bad", "worse") == 320


def test_estimate_bars_request_uses_dynamic_count(monkeypatch):
    """实际请求 param 中的 count 应为动态估算值（非写死 320）。"""
    captured: dict = {}

    class _Capture:
        def __init__(self) -> None:
            self.text = _payload(_rows(2))
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url, params=None, headers=None, timeout=None):  # noqa: ANN001
        captured["params"] = params
        return _Capture()

    monkeypatch.setattr(requests, "get", _fake_get)
    src = TencentSource(adjust="qfq", retry_times=1)
    src.fetch("600519", "20120101", "20241231")

    param = str(captured["params"]["param"])
    count = int(param.split(",")[4])
    assert count == _estimate_bars("20120101", "20241231")


# ---------------------------------------------------------------- _load_json
def test_load_json_rejects_garbage():
    """非法/空文本抛 DataSourceError。"""
    with pytest.raises(DataSourceError):
        _load_json("not json at all")
    with pytest.raises(DataSourceError):
        _load_json("")

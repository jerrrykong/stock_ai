"""Handler 路由（status/units/data/control）+ 无自启断言用例（T05）。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from stock_ai.config.schema import AppConfig
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.ui import server as ui_server
from stock_ai.scale.ui.server import ScaleUI


@pytest.fixture
def ui(tmp_path):
    """在临时端口启动 UI 服务。"""
    cfg = AppConfig()
    cfg.paths.root = tmp_path
    cfg.paths.raw = tmp_path / "data" / "raw"
    cfg.data.cache_dir = tmp_path / "data" / "raw"
    paths = ScalePaths(tmp_path)
    paths.ensure()
    instance = ScaleUI(cfg, paths, logger=None, host="127.0.0.1", port=0)
    srv = instance.make_server()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance, srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _base(srv) -> str:
    port = srv.server_address[1]
    return f"http://127.0.0.1:{port}"


def _get(srv, path: str):
    with urllib.request.urlopen(_base(srv) + path, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


def _post(srv, path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _base(srv) + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


# --------------------------------------------------------------------------- #
def test_index_page_served(ui):
    _, srv = ui
    status, body = _get(srv, "/")
    assert status == 200
    assert "大规模训练控制台" in body
    # 5 个 Tab
    for tab in ("概览", "单元", "检测", "数据", "日志"):
        assert tab in body


def test_api_status_ok(ui):
    _, srv = ui
    status, body = _get(srv, "/api/status?phase=train")
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["phase"] == "train"
    assert "counts" in data and "total" in data


def test_api_units_ok(ui, tmp_path):
    instance, srv = ui
    # 造一个状态文件
    from stock_ai.scale.state import ScaleState, StateStore, UnitState

    store = StateStore(instance.paths)
    st = ScaleState(phase="train", total=2)
    st.units = {
        "600519_2019": UnitState(status="done"),
        "600519_2020": UnitState(status="failed", error="x"),
    }
    store.save(st)

    status, body = _get(srv, "/api/units?phase=train&page_size=50")
    assert status == 200
    data = json.loads(body)
    assert data["ok"] and data["total"] == 2
    # 状态筛选
    _, body2 = _get(srv, "/api/units?phase=train&status=failed")
    assert json.loads(body2)["total"] == 1
    # 关键字搜索
    _, body3 = _get(srv, "/api/units?phase=train&q=2020")
    assert json.loads(body3)["total"] == 1


def test_api_data_ok(ui):
    _, srv = ui
    status, body = _get(srv, "/api/data")
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert "items" in data and "symbols_count" in data


def test_api_logs_ok(ui):
    _, srv = ui
    status, body = _get(srv, "/api/logs?phase=train&tail=10")
    assert status == 200
    assert json.loads(body)["ok"] is True


def test_control_pause_writes_control_json(ui):
    instance, srv = ui
    status, body = _post(srv, "/api/control", {"phase": "train", "action": "pause"})
    assert status == 200
    assert json.loads(body)["ok"] is True
    assert instance.paths.control_json().exists()

    _, body2 = _get(srv, "/api/control/status?phase=train")
    assert json.loads(body2)["cmd"] == "pause"


def test_control_rejects_bad_action(ui):
    _, srv = ui
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(srv, "/api/control", {"phase": "train", "action": "explode"})
    assert exc.value.code == 400


def test_plot_missing_returns_404(ui):
    _, srv = ui
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(srv, "/api/plot?unit_id=600519_2019&kind=curve")
    assert exc.value.code == 404


def test_no_autostart_and_single_spawn(ui, monkeypatch):
    """UI-8：初始无子进程；``start`` 仅启动一次（不自启/不复活）。"""
    instance, _srv = ui
    assert instance.procs == {}
    assert instance.running("train") is False

    calls: list[list[str]] = []

    class _FakeProc:
        pid = 424242

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        return _FakeProc()

    monkeypatch.setattr(ui_server.subprocess, "Popen", _fake_popen)

    r1 = instance.control("train", "start")
    assert r1["ok"] is True and r1["already"] is False
    r2 = instance.control("train", "start")
    assert r2["already"] is True
    assert len(calls) == 1
    assert calls[0][1:4] == ["-m", "stock_ai", "scale"]


def test_ui_binds_localhost_only(ui):
    instance, srv = ui
    assert instance.host == "127.0.0.1"
    assert srv.server_address[0] == "127.0.0.1"

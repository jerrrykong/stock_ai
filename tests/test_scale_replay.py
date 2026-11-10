"""独立研究回放：假模型、临时缓存及随机端口，不触碰现有服务/产物。"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_ai.config.schema import AppConfig
from stock_ai.samples.features import FeatureEngineer
from stock_ai.scale import replay
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.replay import ReplayError, ReplayService
from stock_ai.scale.ui.server import ScaleUI


class FakeModel:
    obs_shape = (3, 3)
    n_actions = 3

    def __init__(self, actions=()):
        self.actions = list(actions)
        self.seen = []

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.seen.append(obs.copy())
        i = len(self.seen) - 1
        return self.actions[i] if i < len(self.actions) else 0, None


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    cfg = AppConfig()
    cfg.paths.root = tmp_path
    cfg.paths.raw = tmp_path / "data" / "raw"
    cfg.paths.raw.mkdir(parents=True)
    paths = ScalePaths(tmp_path)
    paths.models_dir.mkdir(parents=True)
    dates = pd.bdate_range("2023-01-02", periods=80)
    prices = 10.0 + np.arange(80) * 0.05
    df = pd.DataFrame({"date": dates, "open": prices - 0.02,
                       "high": prices + 0.1, "low": prices - 0.1,
                       "close": prices, "volume": 1000 + np.arange(80)})
    df.to_parquet(cfg.paths.raw / "600519.parquet", index=False)
    model = FakeModel()
    monkeypatch.setattr(replay, "_load_model", lambda _: model)
    monkeypatch.setattr(FeatureEngineer, "fit", lambda *a: pytest.fail("回放禁止 refit"))
    service = ReplayService(cfg, paths)
    w = SimpleNamespace(cfg=cfg, paths=paths, df=df, model=model, service=service)
    add_model(w, "600519_2022")
    return w


def add_model(w, uid):
    symbol, year = uid.split("_")
    w.paths.model_zip(uid).write_bytes(b"fake model")
    w.paths.sidecar(uid).write_text(json.dumps({"unit_id": uid, "symbol": symbol,
                                               "year": int(year), "backend": "fake",
                                               "algo": "ppo", "model_path": "../../ignored.zip"}), encoding="utf-8")
    w.paths.norm(uid).write_text(json.dumps({"columns": ["close", "ma20", "volume"],
                                            "mean": [1, 2, 3], "std": [2, 3, 4]}), encoding="utf-8")


def payload(w, lo=30, hi=34, **overrides):
    p = {"symbol": "600519", "start": w.df.date.iloc[lo].strftime("%Y%m%d"),
         "end": w.df.date.iloc[hi].strftime("%Y%m%d"), "model_id": "auto"}
    p.update(overrides)
    return p


@pytest.fixture
def http(workspace):
    ui = ScaleUI(workspace.cfg, workspace.paths, port=0)
    server = ui.make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield workspace, ui, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, path, data=None, raw=None, headers=None):
    if data is not None:
        raw = json.dumps(data).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                                 data=raw, headers=headers or {})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        body = response.read().decode("utf-8")
        return response.status, body


@pytest.mark.parametrize("changes", [
    {"symbol": "../600519"}, {"symbol": "600519/.."}, {"symbol": "６００５１９"},
    {"symbol": 600519}, {"start": "20230230"}, {"start": "2023-2-1"},
    {"start": "20231301"}, {"end": None}, {"start": "20240101", "end": "20230101"},
    {"start": "20000101", "end": "20230101"},
    {"model_id": "../../600519_2022"}, {"model_id": "E:/models/x.zip"},
    {"model_id": "600519_2022.zip"}, {"model_id": "..%2f600519_2022"},
    {"model_id": ["600519_2022"]}, {"model_path": "/secret.zip"},
])
def test_api_validation(http, changes):
    w, _, server = http
    status, body = request(server, "/api/replay", payload(w, **changes))
    assert status == 400
    assert json.loads(body)["error"]


@pytest.mark.parametrize("raw", [b"[1]", b"null", b"1", b"{broken"])
def test_invalid_json_body(http, raw):
    _, _, server = http
    assert request(server, "/api/replay", raw=raw)[0] == 400


def test_body_size_limit(http):
    _, _, server = http
    assert request(server, "/api/replay", raw=b"x" * 16385)[0] == 400


def test_catalog_only_legal_existing_models(http):
    w, _, server = http
    w.paths.model_zip("600519_2021").write_bytes(b"missing sidecar")
    w.paths.model_zip("evil").write_bytes(b"invalid name")
    add_model(w, "600519_2020")
    w.paths.sidecar("600519_2020").write_text('{"year": 2025}', encoding="utf-8")
    status, body = request(server, "/api/replay/catalog")
    data = json.loads(body)
    assert status == 200
    assert data["symbols"] == ["600519"]
    assert [m["id"] for m in data["models"]] == ["600519_2022"]
    assert "model_path" not in body


def test_zero_trades_and_read_only(http, monkeypatch):
    w, ui, server = http
    # 任何写报告/控制/训练路径都不应被调用。
    monkeypatch.setattr(ui, "control", lambda *a: pytest.fail("不允许控制训练"))
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in w.paths.root.rglob("*") if p.is_file()}
    status, body = request(server, "/api/replay", payload(w))
    result = json.loads(body, parse_constant=lambda x: pytest.fail(f"非有限 JSON: {x}"))
    assert status == 200 and result["ok"]
    assert result["trades"] == [] and result["summary"]["trades"] == 0
    assert len(w.model.seen) == 4 and len(result["daily"]) == 5
    assert all(d["equity"] == w.cfg.env.initial_cash for d in result["daily"])
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in w.paths.root.rglob("*") if p.is_file()}
    assert before == after
    assert ui.procs == {} and not w.paths.state_dir.exists() and not w.paths.reports_dir.exists()


def test_execution_alignment_and_final_close_valuation(workspace):
    w = workspace
    w.model.actions = [1, 2, 1, 0]
    result = w.service.run(payload(w))
    daily, trades = result["daily"], result["trades"]
    expected = w.df.date.iloc[30:35].dt.strftime("%Y%m%d").tolist()
    assert [d["date"] for d in daily] == expected
    assert [t["date"] for t in trades] == expected[:3]
    assert [d["action"] for d in daily] == [1, 2, 1, 0, None]
    assert [d["trade_ok"] for d in daily] == [True, True, True, False, False]
    assert all(d["equity"] == pytest.approx(d["cash"] + d["shares"] * d["close"]) for d in daily)
    assert daily[-1]["shares"] > 0
    assert daily[-1]["equity"] > daily[-2]["equity"]
    assert daily[-1]["execution"] is False
    assert "无新执行" in daily[-1]["blocked_reason"]
    assert not any(t["date"] == expected[-1] for t in trades)


def test_blocked_signals_are_not_trades(workspace):
    w = workspace
    w.model.actions = [2, 1, 1, 0]
    result = w.service.run(payload(w))
    assert result["daily"][0]["action"] == 2
    assert not result["daily"][0]["trade_ok"]
    assert result["daily"][0]["blocked_reason"]
    assert not result["daily"][2]["trade_ok"]
    assert result["summary"]["trades"] == 1


def test_single_day_is_valuation_only(workspace):
    w = workspace
    w.model.actions = [1]
    result = w.service.run(payload(w, hi=30))
    assert w.model.seen == []
    assert result["trades"] == []
    assert result["daily"][0]["action"] is None
    assert result["daily"][0]["baseline"] == w.cfg.env.initial_cash


def test_warmup_full_history_and_frozen_transform(workspace, monkeypatch):
    w = workspace
    seen_lengths = []
    original = FeatureEngineer.transform
    def transform(self, df, norm):
        seen_lengths.append(len(df))
        return original(self, df, norm)
    monkeypatch.setattr(FeatureEngineer, "transform", transform)
    result = w.service.run(payload(w))
    assert seen_lengths == [80]
    assert result["warmup_rows"] == 30
    first = w.model.seen[0]
    assert first.shape == (3, 3)
    expected_ma = w.df.close.rolling(20, min_periods=1).mean().iloc[28:31].to_numpy()
    np.testing.assert_allclose(first[:, 1], (expected_ma - 2) / 3, rtol=1e-6)
    np.testing.assert_allclose(first[:, 0], (w.df.close.iloc[28:31] - 1) / 2)


def test_warmup_start_padding_warning(workspace):
    result = workspace.service.run(payload(workspace, lo=0, hi=2))
    assert any("复制填充" in w for w in result["warnings"])
    np.testing.assert_array_equal(workspace.model.seen[0][0], workspace.model.seen[0][2])


def test_strict_past_auto_never_uses_same_year_or_future(workspace):
    w = workspace
    add_model(w, "600519_2023")
    add_model(w, "600519_2025")
    add_model(w, "600519_2021")
    assert w.service.run(payload(w))["model"]["id"] == "600519_2022"
    for uid in ["600519_2022", "600519_2021"]:
        w.paths.model_zip(uid).unlink()
    with pytest.raises(ReplayError, match="不会回退") as exc:
        w.service.run(payload(w))
    assert exc.value.status == 404


@pytest.mark.parametrize("uid", ["600519_2023", "600519_2025", "000001_2025"])
def test_manual_same_year_future_and_cross_symbol_warning(workspace, uid):
    w = workspace
    add_model(w, uid)
    result = w.service.run(payload(w, model_id=uid))
    assert result["temporal_risk"] is True
    assert result["selection"] == "manual_research"
    assert any("前视" in s for s in result["warnings"])
    assert any("研究回放" in s for s in result["warnings"])
    if uid.startswith("000001"):
        assert any("训练代码" in s for s in result["warnings"])


def test_missing_norm_no_refit(http):
    w, _, server = http
    w.paths.norm("600519_2022").unlink()
    status, body = request(server, "/api/replay", payload(w, model_id="600519_2022"))
    assert status == 404 and "禁止重新 fit" in body
    assert w.model.seen == []


@pytest.mark.parametrize("patch", [
    {"mean": [[1, 2, 3]]}, {"std": [1, 2]}, {"std": [1, 0, 3]},
    {"std": [1, -1, 3]}, {"mean": [1, float("nan"), 3]},
    {"columns": []}, {"columns": ["close", "close", "volume"]},
    {"columns": ["close", "unknown", "volume"]},
])
def test_invalid_norm_shape_finite_and_columns(http, patch):
    w, _, server = http
    path = w.paths.norm("600519_2022")
    norm = json.loads(path.read_text())
    norm.update(patch)
    path.write_text(json.dumps(norm), encoding="utf-8")
    status, body = request(server, "/api/replay", payload(w))
    assert status == 500 and json.loads(body)["error"]
    assert w.model.seen == []


@pytest.mark.parametrize("shape,n_actions", [(None, 3), ((4, 4), 3), ((600, 3), 3), ((3,), 3), ((3, 3), 2)])
def test_model_shape_validation(http, shape, n_actions):
    w, _, server = http
    w.model.obs_shape = shape
    w.model.n_actions = n_actions
    assert request(server, "/api/replay", payload(w))[0] == 500
    assert w.model.seen == []


@pytest.mark.parametrize("overrides", [{"symbol": "000001"}, {"model_id": "600519_2019"},
                                       {"start": "20240101", "end": "20240201"}])
def test_missing_cache_model_or_range(http, overrides):
    w, _, server = http
    assert request(server, "/api/replay", payload(w, **overrides))[0] == 404


def test_range_bar_limit(http, monkeypatch):
    w, _, server = http
    monkeypatch.setattr(replay, "MAX_BARS", 3)
    assert request(server, "/api/replay", payload(w))[0] == 400


@pytest.mark.parametrize("field", ["close", "volume"])
def test_nonfinite_cache_rejected(http, field):
    w, _, server = http
    w.df[field] = w.df[field].astype("float64")
    w.df.loc[31, field] = float("inf")
    w.df.to_parquet(w.cfg.paths.raw / "600519.parquet")
    assert request(server, "/api/replay", payload(w))[0] == 500


def test_nonfinite_json_result_is_not_success(http, monkeypatch):
    w, _, server = http
    monkeypatch.setattr(w.service, "_run", lambda *a: {"value": float("inf")})
    # 使用 HTTP 实例的服务。
    _, ui, _ = http
    monkeypatch.setattr(ui.replay, "_run", lambda *a: {"value": float("inf")})
    status, body = request(server, "/api/replay", payload(w))
    assert status == 500
    assert "Infinity" not in body


def test_model_load_failure_and_lock_released(http, monkeypatch):
    w, _, server = http
    def fail(_):
        raise ValueError("模型损坏")
    monkeypatch.setattr(replay, "_load_model", fail)
    assert request(server, "/api/replay", payload(w))[0] == 500
    monkeypatch.setattr(replay, "_load_model", lambda _: w.model)
    assert request(server, "/api/replay", payload(w))[0] == 200


def test_parallel_request_conflict(http, monkeypatch):
    w, _, server = http
    entered, release = threading.Event(), threading.Event()
    def load(_):
        entered.set()
        assert release.wait(5)
        return w.model
    monkeypatch.setattr(replay, "_load_model", load)
    responses = []
    thread = threading.Thread(target=lambda: responses.append(request(server, "/api/replay", payload(w))))
    thread.start()
    try:
        assert entered.wait(5)
        status, body = request(server, "/api/replay", payload(w))
        assert status == 409 and "正在加载或推理" in body
    finally:
        release.set()
        thread.join(5)
    assert responses[0][0] == 200


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/%2e%2e/server.py",
                                   "/static/vendor/../../server.py", "/static/index.html",
                                   "/static/vendor/echarts.min.js/../secret", "/static/vendor/not-listed.js"])
def test_static_whitelist_traversal(http, path):
    _, _, server = http
    assert request(server, path)[0] == 404


def test_static_resources_and_independent_tab(http):
    _, _, server = http
    assert request(server, "/static/replay.js")[0] == 200
    status, js = request(server, "/static/vendor/echarts.min.js")
    assert status == 200 and len(js) > 500000
    _, html = request(server, "/")
    assert 'data-tab="replay"' in html
    assert '/static/vendor/echarts.min.js' in html
    assert 'https://' not in html
    assert 'setInterval(refreshStatus, 1500)' in html


@pytest.mark.parametrize("action", [float("nan"), float("inf"), 1.5, 3, [1, 2]])
def test_invalid_model_action_returns_500(http, action):
    w, _, server = http
    w.model.actions = [action]
    assert request(server, "/api/replay", payload(w))[0] == 500


def test_model_flattened_shape_mismatch(http):
    w, _, server = http
    w.model.obs_dim = 100
    assert request(server, "/api/replay", payload(w))[0] == 500


def test_hyphen_dates_and_actual_cached_range(workspace):
    w = workspace
    result = w.service.run(payload(w, start="2023-02-11", end="2023-02-17"))
    assert result["requested_range"]["start"] == "20230211"
    assert result["actual_range"] == {"start": "20230213", "end": "20230217"}


def test_no_network_during_replay(http, monkeypatch):
    w, _, server = http
    import socket
    import requests
    original_connect = socket.socket.connect
    def connect(sock, address):
        assert address[0] == "127.0.0.1", "禁止访问外部行情源"
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("禁止联网行情"))
    assert request(server, "/api/replay", payload(w))[0] == 200


def test_reject_external_symlink_model(workspace, tmp_path):
    w = workspace
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"must not load")
    add_model(w, "600519_2020")
    path = w.paths.model_zip("600519_2020")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("当前 Windows 用户没有符号链接权限")
    assert not any(m["id"] == "600519_2020" for m in w.service.models())


@pytest.fixture
def strict_workspace(workspace, monkeypatch):
    from stock_ai.scale import unified
    from stock_ai.samples.normalizer import NormalizerState

    w = workspace
    w.feature_dir = w.paths.root / "data" / "raw_hfq"
    w.feature_dir.mkdir()
    # 不能只对价格乘常数：相对特征本身具有尺度不变性，必须让 hfq 与 raw 走势不同。
    w.hfq = w.df.copy()
    factor = 2 + np.arange(len(w.df)) * 0.025
    w.hfq.loc[:, ["open", "high", "low", "close"]] = w.hfq[["open", "high", "low", "close"]].mul(factor, axis=0)
    w.hfq["volume"] = 2000 + np.arange(len(w.df)) ** 2
    w.hfq.to_parquet(w.feature_dir / "600519.parquet", index=False)
    w.run = w.paths.root / "models" / "unified" / "strict_fixture"
    w.run.mkdir(parents=True)
    w.norm = NormalizerState(np.linspace(-0.02, 0.02, len(unified.FEATURES)),
                             np.linspace(0.03, 0.1, len(unified.FEATURES)), list(unified.FEATURES))
    env = asdict(w.cfg.env)
    env["initial_cash"] = 100000.0
    env["reward_type"] = "excess"
    w.manifest = {"run_id": w.run.name, "mode": "unified", "contract": unified.CONTRACT,
                  "backend": "sb3", "algo": "ppo", "window": 3, "features": unified.FEATURES,
                  "observation_shape": [3, len(unified.FEATURES)],
                  "ranges": {"train": [2006, 2021], "val": [2022, 2023], "test": [2024, 2025]},
                  "env_config": env, "execution_version": unified.EXECUTION_STRICT,
                  "feature_adjust": "hfq", "execution_adjust": "none(真实未复权价)",
                  "feature_dir": str(w.feature_dir), "execution_dir": str(w.cfg.paths.raw),
                  "reward_type": "excess", "reward_version": "excess-causal-close-v1",
                  "reward_params": {key: env[key] for key in
                                    ("reward_type", "lambda_drawdown", "lambda_turnover", "lambda_leverage")}}
    (w.run / "manifest.json").write_text(json.dumps(w.manifest), encoding="utf-8")
    (w.run / "normalizer.json").write_text(json.dumps(w.norm.to_dict()), encoding="utf-8")
    (w.run / "model.zip").write_bytes(b"fake strict model")
    w.model.obs_shape = (3, len(unified.FEATURES))
    w.model_id = "unified:" + w.run.name
    monkeypatch.setattr(unified, "build_dataset", lambda *a, **k: pytest.fail("回放禁止重建数据集或拟合 norm"))
    return w


def strict_payload(w, **kwargs):
    return payload(w, model_id=w.model_id, **kwargs)


def patch_manifest(w, **values):
    w.manifest.update(values)
    (w.run / "manifest.json").write_text(json.dumps(w.manifest), encoding="utf-8")


def test_strict_catalog_contract_and_reward_metadata(strict_workspace):
    w = strict_workspace
    patch_manifest(w, created_at="2026-10-15T09:56:07.454212")
    (w.run / "status.json").write_text(json.dumps({"updated_at": "2026-10-15T10:24:39"}), encoding="utf-8")
    item = w.service.catalog()["unified_runs"][0]
    assert item["id"] == w.model_id and item["compatible"]
    for key in ("execution_version", "feature_adjust", "reward_type", "reward_version", "reward_params", "ranges", "window", "created_at"):
        assert item[key] == w.manifest[key]
    assert item["updated_at"] == "2026-10-15T10:24:39"
    assert item["model_available"] and item["normalizer_available"]
    assert w.service.catalog()["models"][-1] == item


def test_strict_signal_execution_dates_snapshots_and_accounting(strict_workspace):
    w = strict_workspace
    w.model.actions = [1, 2, 0, 1]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in w.paths.root.rglob("*") if p.is_file()}
    result = w.service.run(strict_payload(w))
    daily, trades = result["daily"], result["trades"]
    expected_dates = w.df.date.iloc[30:35].dt.strftime("%Y%m%d").tolist()
    assert result["execution_version"] == "strict-t1-open-v1"
    assert result["feature_adjust"] == "hfq" and result["initial_cash"] == 100000
    assert result["source"]["feature_file"] == str((w.feature_dir / "600519.parquet").resolve())
    assert result["source"]["execution_file"] == str((w.cfg.paths.raw / "600519.parquet").resolve())
    assert [row["date"] for row in daily] == expected_dates
    assert [row["action"] for row in daily] == [1, 2, 0, 1, None]
    assert [row["executed_action"] for row in daily] == [None, 1, 2, 0, 1]
    assert [row["execution_signal_date"] for row in daily] == [None, *expected_dates[:-1]]
    assert [row["pending_execution_date"] for row in daily] == [*expected_dates[1:], None]
    assert [row["execution"] for row in daily] == [False, True, True, True, True]
    assert [row["trade_ok"] for row in daily] == [False, True, True, False, True]
    assert [row["shares"] > 0 for row in daily] == [False, True, False, False, True]
    assert daily[0]["cash"] == daily[0]["equity"] == daily[0]["baseline"] == 100000
    assert daily[0]["daily_return"] == daily[0]["drawdown"] == daily[0]["fee"] == 0
    assert daily[-1]["signal_status"] == "no_signal_final_day" and daily[-1]["trade_ok"]
    assert [trade["date"] for trade in trades] == [expected_dates[1], expected_dates[2], expected_dates[4]]
    assert [trade["signal_date"] for trade in trades] == [expected_dates[0], expected_dates[1], expected_dates[3]]
    assert len(w.model.seen) == len(daily) - 1
    realized, previous, peak = 0.0, 100000.0, 100000.0
    for row in daily:
        todays = [trade for trade in trades if trade["date"] == row["date"]]
        realized += sum(trade["realized_pnl"] for trade in todays)
        assert row["equity"] == pytest.approx(row["cash"] + row["position_value"])
        assert row["equity"] == pytest.approx(100000 + realized + row["unrealized_pnl"])
        assert row["position_value"] == row["shares"] * row["close"]
        assert row["position"] == pytest.approx(row["position_value"] / row["equity"])
        assert row["unrealized_pnl"] == pytest.approx((row["close"] - row["cost_basis"]) * row["shares"])
        assert row["daily_return"] == pytest.approx(row["equity"] / previous - 1)
        peak = max(peak, row["equity"])
        assert row["drawdown"] == pytest.approx(1 - row["equity"] / peak)
        assert row["fee"] == pytest.approx(sum(trade["fee"] for trade in todays))
        assert row["slippage_cost"] == pytest.approx(sum(abs(trade["price"] - row["open"]) * trade["shares"] for trade in todays))
        for trade in todays:
            assert trade["open_price"] == row["open"]
            sign = 1 if trade["side"] == "buy" else -1
            assert trade["price"] == pytest.approx(row["open"] * (1 + sign * w.cfg.env.slippage_rate))
        previous = row["equity"]
    summary = result["summary"]
    assert summary["total_fee"] == pytest.approx(sum(trade["fee"] for trade in trades))
    assert summary["slippage_cost"] == pytest.approx(sum(row["slippage_cost"] for row in daily))
    assert summary["cost_drag_bps"] == pytest.approx(summary["total_fee"] / np.mean([row["equity"] for row in daily]) * 10000)
    assert summary["max_drawdown"] == max(row["drawdown"] for row in daily)
    assert summary["average_position"] == pytest.approx(np.mean([row["position"] for row in daily]))
    assert summary["holding_days"] == 2
    assert summary["excess_return"] == pytest.approx(summary["total_return"] - summary["baseline_return"])
    assert "不含滑点" in result["cost_note"]
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in w.paths.root.rglob("*") if p.is_file()}
    assert before == after
    assert not w.paths.state_dir.exists() and not w.paths.reports_dir.exists()


def test_strict_uses_hfq_frozen_norm_and_training_filter_order(strict_workspace):
    from stock_ai.scale.unified import normalize, observations, relative_features

    w = strict_workspace
    # 区间前缺失的真实交易日仍先参与 hfq 指标计算，之后才被窗口过滤。
    raw = w.df.drop(index=29)
    raw.to_parquet(w.cfg.paths.raw / "600519.parquet", index=False)
    norm_before = (w.run / "normalizer.json").read_bytes()
    result = w.service.run(strict_payload(w))
    _, features = relative_features(w.hfq)
    keep = np.arange(len(w.hfq)) != 29
    expected = observations(normalize(features[keep], w.norm), 3)
    np.testing.assert_array_equal(w.model.seen[0], expected[29])
    assert result["warmup_rows"] == 29
    assert result["source"]["filtered_history_rows"] == 1
    assert (w.run / "normalizer.json").read_bytes() == norm_before
    _, incorrect_raw_features = relative_features(w.df)
    assert not np.allclose(w.model.seen[0], observations(normalize(incorrect_raw_features, w.norm), 3)[30])
    for i, row in enumerate(result["daily"]):
        assert row["close"] == w.df.close.iloc[30 + i]
        assert row["open"] == w.df.open.iloc[30 + i]


def test_strict_features_causal_no_future_observation(strict_workspace):
    w = strict_workspace
    first = w.service.run(strict_payload(w))
    first_seen = [obs.copy() for obs in w.model.seen]
    changed = w.hfq.copy()
    changed.loc[32:, ["open", "high", "low", "close", "volume"]] *= 8
    changed.to_parquet(w.feature_dir / "600519.parquet", index=False)
    w.model.seen.clear()
    second = w.service.run(strict_payload(w))
    np.testing.assert_array_equal(first_seen[:2], w.model.seen[:2])
    assert not np.allclose(first_seen[2], w.model.seen[2])
    assert first["daily"][:2] == second["daily"][:2]


@pytest.mark.parametrize("mode", ["hold", "sell_empty", "cash", "halt", "limit_up", "limit_down"])
def test_strict_rejected_orders_and_zero_trades(strict_workspace, mode):
    w = strict_workspace
    if mode == "hold":
        w.model.actions = [0] * 4
    elif mode == "sell_empty":
        w.model.actions = [2] * 4
    elif mode == "cash":
        patch_manifest(w, env_config={**w.manifest["env_config"], "initial_cash": 1.0})
        w.model.actions = [1] * 4
    elif mode == "halt":
        w.df.loc[31:34, "volume"] = 0
        w.model.actions = [1] * 4
    elif mode == "limit_up":
        # 执行日开盘涨停，不能拿 hfq 价格或信号日价格代替真实开盘判断。
        price = round(w.df.close.iloc[30] * 1.1, 2)
        for idx in range(31, 35):
            w.df.loc[idx, ["open", "high", "low", "close"]] = price
            price = round(price * 1.1, 2)
        w.model.actions = [1] * 4
    else:
        price = round(w.df.close.iloc[31] * 0.9, 2)
        w.df.loc[32, ["open", "high", "low", "close"]] = price
        w.model.actions = [1, 2, 0, 0]
    w.df.to_parquet(w.cfg.paths.raw / "600519.parquet", index=False)
    result = w.service.run(strict_payload(w))
    if mode == "limit_down":
        assert len(result["trades"]) == 1
        assert result["daily"][2]["executed_action"] == 2
        assert "跌停" in result["daily"][2]["blocked_reason"]
        assert result["daily"][2]["shares"] > 0
    else:
        assert result["trades"] == []
        assert all(row["equity"] == result["initial_cash"] for row in result["daily"])
        assert all(not row["trade_ok"] for row in result["daily"])
        assert result["summary"]["holding_days"] == result["summary"]["total_fee"] == result["summary"]["slippage_cost"] == 0
        assert result["summary"]["average_position"] == result["summary"]["max_drawdown"] == 0
        if mode != "hold":
            assert all(row["blocked_reason"] for row in result["daily"][1:])


def test_strict_baseline_same_t1_open_rules_no_legacy_lookahead(strict_workspace, monkeypatch):
    from stock_ai.env.strict_env import StrictExecutionEnv

    w = strict_workspace
    w.model.actions = [1] * 4
    monkeypatch.setattr(replay.Backtester, "_buy_and_hold", lambda *a: pytest.fail("strict 禁止 legacy 基准"))
    monkeypatch.setattr(StrictExecutionEnv, "buy_and_hold", lambda *a: pytest.fail("strict 禁止旧前视基准"))
    monkeypatch.setattr(StrictExecutionEnv, "metrics", lambda *a: pytest.fail("metrics 内含旧基准，禁止调用"))
    limit = round(w.df.close.iloc[30] * 1.1, 2)
    w.df.loc[31, ["open", "high", "low", "close"]] = limit
    w.df.loc[32, "volume"] = 0
    w.df.to_parquet(w.cfg.paths.raw / "600519.parquet", index=False)
    result = w.service.run(strict_payload(w))
    daily = result["daily"]
    assert all(row["baseline"] == pytest.approx(row["equity"]) for row in daily)
    assert all(row["baseline"] == result["initial_cash"] for row in daily[:3])
    assert [trade["date"] for trade in result["trades"]] == [daily[3]["date"]]
    assert result["summary"]["excess_return"] == pytest.approx(0)
    assert "初始现金" in result["baseline_note"] and "metrics" in result["baseline_note"]


def test_strict_day_close_snapshot_precedes_next_open_state(strict_workspace):
    w = strict_workspace
    # 若误用 step 后账户回填信号日，首日会提前持仓且现金立即减少。
    w.model.actions = [0, 0, 0, 1]
    result = w.service.run(strict_payload(w))
    daily = result["daily"]
    assert all(row["shares"] == 0 and row["cash"] == result["initial_cash"] for row in daily[:-1])
    assert daily[-2]["action"] == 1 and not daily[-2]["trade_ok"]
    assert daily[-1]["action"] is None and daily[-1]["trade_ok"]
    assert daily[-1]["executed_action"] == 1 and daily[-1]["shares"] > 0
    assert result["trades"][0]["date"] == daily[-1]["date"]
    assert result["trades"][0]["signal_date"] == daily[-2]["date"]


def test_strict_single_day_no_prediction_or_two_bar_environment(strict_workspace, monkeypatch):
    from stock_ai.env import strict_env

    w = strict_workspace
    monkeypatch.setattr(strict_env, "StrictExecutionEnv", lambda *a, **k: pytest.fail("单日不能创建至少两 bar 环境"))
    w.model.actions = [1]
    result = w.service.run(strict_payload(w, hi=30))
    assert w.model.seen == [] and result["trades"] == []
    row = result["daily"][0]
    assert row["action"] is row["executed_action"] is row["execution_signal_date"] is None
    assert row["equity"] == row["cash"] == row["baseline"] == result["initial_cash"]
    assert row["position_value"] == row["unrealized_pnl"] == row["daily_return"] == row["fee"] == row["slippage_cost"] == 0
    assert result["summary"]["executions"] == result["summary"]["trades"] == 0


@pytest.mark.parametrize("symbol,lo,hi", [("600519", 10, 12), ("000001", 22, 25), ("300001", 45, 51)])
def test_strict_any_cached_symbol_and_dates(strict_workspace, symbol, lo, hi):
    w = strict_workspace
    w.df.to_parquet(w.cfg.paths.raw / f"{symbol}.parquet", index=False)
    w.hfq.to_parquet(w.feature_dir / f"{symbol}.parquet", index=False)
    w.model.actions = [1] * (hi - lo)
    result = w.service.run(strict_payload(w, symbol=symbol, lo=lo, hi=hi))
    assert result["symbol"] == symbol
    assert result["actual_range"] == {"start": w.df.date.iloc[lo].strftime("%Y%m%d"),
                                      "end": w.df.date.iloc[hi].strftime("%Y%m%d")}
    assert len(result["daily"]) == hi - lo + 1
    assert result["daily"][0]["shares"] == 0 and result["daily"][1]["shares"] > 0
    assert len(w.model.seen) == hi - lo


def test_strict_continuous_cross_year_no_account_reset(strict_workspace):
    w = strict_workspace
    dates = pd.bdate_range("2022-12-12", periods=len(w.df))
    w.df["date"] = w.hfq["date"] = dates
    w.df.to_parquet(w.cfg.paths.raw / "600519.parquet", index=False)
    w.hfq.to_parquet(w.feature_dir / "600519.parquet", index=False)
    w.model.actions = [1] + [0] * 14
    result = w.service.run(strict_payload(w, lo=10, hi=25))
    daily = result["daily"]
    assert daily[0]["date"].startswith("2022") and daily[-1]["date"].startswith("2023")
    assert all(row["shares"] == daily[1]["shares"] > 0 for row in daily[1:])
    assert all(row["cash"] == daily[1]["cash"] for row in daily[1:])
    assert len(result["trades"]) == 1
    assert any("连续跨年" in warning for warning in result["warnings"])


@pytest.mark.parametrize("cache", ["feature", "execution"])
def test_strict_missing_cache_rejected_without_fallback(strict_workspace, cache):
    w = strict_workspace
    path = (w.feature_dir if cache == "feature" else w.cfg.paths.raw) / "600519.parquet"
    path.unlink()
    with pytest.raises(ReplayError, match="缺失|不会回退") as exc:
        w.service.run(strict_payload(w))
    assert exc.value.status == 404 and w.model.seen == []


@pytest.mark.parametrize("cache", ["feature", "execution"])
@pytest.mark.parametrize("problem", ["date", "duplicate", "ohlcv", "missing_column", "nonfinite"])
def test_strict_illegal_cache_rejected(strict_workspace, cache, problem):
    w = strict_workspace
    df = (w.hfq if cache == "feature" else w.df).copy()
    if problem == "date":
        df["date"] = df["date"].astype(str)
        df.loc[31, "date"] = "not-a-date"
    elif problem == "duplicate":
        df.loc[31, "date"] = df.date.iloc[30]
    elif problem == "ohlcv":
        df.loc[31, "low"] = df.high.iloc[31] + 10
    elif problem == "missing_column":
        df = df.drop(columns="open")
    else:
        df.loc[31, "close"] = float("nan")
    path = (w.feature_dir if cache == "feature" else w.cfg.paths.raw) / "600519.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(ReplayError):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


@pytest.mark.parametrize("cache", ["feature", "execution"])
def test_strict_missing_requested_date_is_error_not_inner_join(strict_workspace, cache):
    w = strict_workspace
    df = (w.hfq if cache == "feature" else w.df).drop(index=31)
    path = (w.feature_dir if cache == "feature" else w.cfg.paths.raw) / "600519.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(ReplayError, match="日期不齐"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


@pytest.mark.parametrize("version", ["strict-t2-open-v9", "", None, 42, ["strict-t1-open-v1"]])
def test_unified_unknown_execution_version_never_downgrades(strict_workspace, version):
    w = strict_workspace
    patch_manifest(w, execution_version=version)
    item = w.service.catalog()["unified_runs"][0]
    assert item["execution_version"] == version and not item["compatible"]
    with pytest.raises(ReplayError, match="未知 execution_version"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


def test_legacy_unified_missing_version_keeps_same_day_close(strict_workspace, monkeypatch):
    from stock_ai.scale.unified import EXECUTION_LEGACY, normalize, observations, relative_features
    from stock_ai.env import strict_env

    w = strict_workspace
    for key in ("execution_version", "feature_adjust", "feature_dir", "execution_dir", "reward_type", "reward_params", "reward_version"):
        w.manifest.pop(key)
    # 模拟存量 legacy normalizer 的契约，只校验执行语义，不重拟合。
    patch_manifest(w)
    monkeypatch.setattr(strict_env, "StrictExecutionEnv", lambda *a, **k: pytest.fail("legacy 不可使用 strict 环境"))
    w.model.actions = [1, 2, 1, 0]
    result = w.service.run(strict_payload(w))
    daily = result["daily"]
    assert result["execution_version"] == EXECUTION_LEGACY
    assert result["model"]["reward_type"] == "excess"
    assert [row["action"] for row in daily] == [1, 2, 1, 0, None]
    assert [row["trade_ok"] for row in daily] == [True, True, True, False, False]
    assert [trade["date"] for trade in result["trades"]] == [row["date"] for row in daily[:3]]
    _, features = relative_features(w.df)
    np.testing.assert_array_equal(w.model.seen[0], observations(normalize(features, w.norm), 3)[30])
    assert daily[0]["shares"] > 0 and not daily[-1]["execution"]


@pytest.mark.parametrize("key,value", [("feature_dir", "https://invalid.example/cache"),
                                        ("execution_dir", "//invalid/cache"),
                                        ("feature_dir", "../../outside"),
                                        ("execution_dir", ""), ("feature_dir", None)])
def test_strict_manifest_paths_are_controlled_local_only(strict_workspace, key, value):
    w = strict_workspace
    patch_manifest(w, **{key: value})
    with pytest.raises(ReplayError, match="安全|受控|缺失"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


@pytest.mark.parametrize("cache", ["feature", "execution"])
def test_strict_external_symbol_symlink_rejected(strict_workspace, tmp_path, cache):
    w = strict_workspace
    outside = tmp_path.parent / f"{tmp_path.name}-outside-{cache}.parquet"
    w.df.to_parquet(outside, index=False)
    path = (w.feature_dir if cache == "feature" else w.cfg.paths.raw) / "600519.parquet"
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("当前 Windows 用户没有符号链接权限")
    with pytest.raises(ReplayError, match="路径不安全"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


def test_strict_manifest_cache_paths_override_unrelated_config_raw(strict_workspace):
    from stock_ai.scale.paths import ScalePaths

    w = strict_workspace
    w.cfg.paths.raw = w.paths.root / "unrelated-empty-cache"
    service = ReplayService(w.cfg, ScalePaths(w.paths.root))
    result = service.run(strict_payload(w))
    assert result["ok"]
    assert result["source"]["execution_dir"] == str((w.paths.root / "data" / "raw").resolve())


def test_strict_all_prediction_windows_match_frozen_training_transforms(strict_workspace):
    from stock_ai.scale import unified

    w = strict_workspace
    _, values = unified.relative_features(w.hfq)
    expected = unified.observations(unified.normalize(values, w.norm), w.manifest["window"])
    w.service.run(strict_payload(w))
    np.testing.assert_array_equal(np.asarray(w.model.seen), expected[30:34])


def test_strict_manifest_sources_cannot_override_controlled_symbol_file(strict_workspace):
    w = strict_workspace
    patch_manifest(w, sources=[{"symbol": "600519", "file": "../../untrusted.parquet"}])
    result = w.service.run(strict_payload(w))
    assert result["ok"]
    assert result["source"]["feature_file"] == str((w.feature_dir / "600519.parquet").resolve())


def test_strict_manifest_relative_dirs_and_no_network(strict_workspace, monkeypatch):
    import socket
    import requests

    w = strict_workspace
    patch_manifest(w, feature_dir="data/raw_hfq", execution_dir="data/raw")
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("回放禁止联网"))
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("回放禁止联网行情"))
    assert w.service.run(strict_payload(w))["ok"]


@pytest.mark.parametrize("field", ["norm", "shape", "feature_adjust", "same_dir"])
def test_strict_contract_validation(strict_workspace, field):
    w = strict_workspace
    if field == "norm":
        norm = w.norm.to_dict()
        norm["columns"] = list(reversed(norm["columns"]))
        (w.run / "normalizer.json").write_text(json.dumps(norm), encoding="utf-8")
    elif field == "shape":
        w.model.obs_shape = (4, 9)
    elif field == "feature_adjust":
        patch_manifest(w, feature_adjust="qfq")
    else:
        patch_manifest(w, feature_dir=str(w.cfg.paths.raw))
    with pytest.raises(ReplayError):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


def test_strict_limits_lock_and_failure_release(strict_workspace, monkeypatch):
    w = strict_workspace
    monkeypatch.setattr(replay, "MAX_BARS", 3)
    with pytest.raises(ReplayError, match="最多") as exc:
        w.service.run(strict_payload(w))
    assert exc.value.status == 400
    monkeypatch.setattr(replay, "MAX_BARS", 2500)
    assert replay._INFERENCE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(ReplayError, match="正在加载或推理") as exc:
            w.service.run(strict_payload(w))
        assert exc.value.status == 409
    finally:
        replay._INFERENCE_LOCK.release()
    w.model.actions = [float("nan")]
    with pytest.raises(ReplayError, match="非法离散动作"):
        w.service.run(strict_payload(w))
    w.model.seen.clear()
    w.model.actions = [0] * 4
    assert w.service.run(strict_payload(w))["ok"]


def add_ca_ledger(w, **coverage_changes):
    """仅供机制测试的合成账本，不表示该股票的真实公司行为。"""
    w.ledger_dir = w.paths.root / "data" / "corporate_actions"
    w.ledger_dir.mkdir(exist_ok=True)
    day = lambda index: w.df.date.iloc[index].strftime("%Y%m%d")
    w.ledger_data = {
        "schema": "corporate-actions-v1", "symbol": "600519",
        "coverage": {"start": day(0), "end": day(79), "status": "verified",
                     "scope": "all_shareholder_distributions_and_capital_changes",
                     "sources": ["https://invalid.example/fixture/coverage"],
                     "verified_at": "2026-10-16", "notes": "机制测试合成数据，不是真实股票公告"},
        "events": [{"event_id": "synthetic-distribution", "kind": "distribution", "status": "implemented",
                    "record_date": day(31), "ex_date": day(32), "pay_date": day(34), "listing_date": day(34),
                    "cash_per_share_gross": "1", "bonus_ratio": "0", "transfer_ratio": "0.5",
                    "reference_method": "standard_cash_bonus", "fractional_policy": "reject",
                    "listing_date_source": "sina_red_share_listing",
                    "sources": ["https://invalid.example/fixture/distribution"]}]}
    w.ledger_data["coverage"].update(coverage_changes)
    w.ledger_path = w.ledger_dir / "600519.json"
    w.ledger_path.write_text(json.dumps(w.ledger_data), encoding="utf-8")
    return w.ledger_path


@pytest.fixture
def ca_workspace(strict_workspace):
    from stock_ai.scale.unified import EXECUTION_CORPORATE_ACTIONS

    w = strict_workspace
    add_ca_ledger(w)
    reference = round((w.df.close.iloc[31] - 1) / 1.5, 2)
    for index in range(32, len(w.df)):
        price = reference + (index - 32) * 0.02
        w.df.loc[index, ["open", "high", "low", "close"]] = price
    w.df.to_parquet(w.cfg.paths.raw / "600519.parquet", index=False)
    # 普通机制夹具按新股票回放，显式保留其他训练股票的来源清单；
    # 需要测试冻结匹配的用例会用 freeze_ca_sources 覆盖为本股票三份实际指纹。
    patch_manifest(w, execution_version=EXECUTION_CORPORATE_ACTIONS,
                   corporate_actions_dir="data/corporate_actions", reward_version="excess-ca-constant-buy-v2",
                   sources=[{"symbol": "000001", "sha256": "1" * 64,
                             "execution_sha256": "2" * 64, "corporate_actions_sha256": "3" * 64}])
    return w


def freeze_ca_sources(w):
    from stock_ai.utils.hashing import sha256_file
    source = {"symbol": "600519", "file": "../../untrusted-feature.parquet",
              "execution_file": "../../untrusted-raw.parquet", "corporate_actions_file": "../../untrusted-ca.json",
              "sha256": sha256_file(w.feature_dir / "600519.parquet"),
              "execution_sha256": sha256_file(w.cfg.paths.raw / "600519.parquet"),
              "corporate_actions_sha256": sha256_file(w.ledger_path)}
    patch_manifest(w, sources=[source])
    return source


def test_ca_catalog_and_default_versioned_execution(ca_workspace):
    w = ca_workspace
    item = w.service.catalog()["unified_runs"][0]
    assert item["compatible"] and item["execution_version"] == "strict-t1-open-ca-v2"
    w.model.actions = [1, 0, 2, 0]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in w.paths.root.rglob("*") if p.is_file()}
    result = w.service.run(strict_payload(w))
    assert result["execution_version"] == result["training_execution_version"] == "strict-t1-open-ca-v2"
    assert result["selection"] == "manual_research" and result["execution_mode"] == "default"
    assert result["reward_version"] == "excess-ca-constant-buy-v2"
    daily = result["daily"]
    assert [row["executed_action"] for row in daily] == [None, 1, 0, 2, 0]
    entitled = daily[1]["shares"]
    assert entitled > 0 and daily[2]["shares"] == entitled * 1.5
    assert daily[2]["dividend_receivable"] == daily[2]["dividend_income"] == entitled
    assert daily[2]["locked_shares"] == entitled * 0.5
    assert daily[2]["available_shares"] == entitled
    assert daily[3]["shares"] == daily[3]["locked_shares"] == entitled * 0.5
    assert daily[3]["available_shares"] == 0 and daily[3]["dividend_receivable"] == entitled
    assert daily[4]["locked_shares"] == daily[4]["dividend_receivable"] == 0
    assert daily[4]["available_shares"] == entitled * 0.5
    for row in daily:
        assert row["equity"] == pytest.approx(row["cash"] + row["shares"] * row["close"] + row["dividend_receivable"])
        assert row["equity"] == row["portfolio_value"]
        assert row["equity"] == pytest.approx(result["initial_cash"] + row["realized_pnl"] + row["unrealized_pnl"] + row["dividend_income"])
        assert row["unrealized_pnl"] == pytest.approx(row["shares"] * (row["close"] - row["cost_basis"]))
        assert row["position"] == pytest.approx(row["position_value"] / row["equity"])
    assert daily[2]["equity"] > daily[2]["cash"] + daily[2]["position_value"]
    assert daily[2]["limit_reference"] == round((daily[1]["close"] - 1) / 1.5, 2)
    assert daily[2]["limit_down"] < daily[2]["open"] < daily[2]["limit_up"]
    assert result["summary"]["dividend_income"] == entitled
    assert result["summary"]["final_dividend_receivable"] == result["summary"]["final_locked_shares"] == 0
    assert result["summary"]["holding_days"] == 4
    assert result["summary"]["final_available_shares"] == entitled * 0.5
    assert [event["stage"] for event in result["corporate_actions"]] == ["record", "ex_entitlement", "cash_payment", "stock_listing"]
    assert [event["stage"] for event in daily[2]["corporate_actions"]] == ["ex_entitlement"]
    assert all(event["sources"] == w.ledger_data["events"][0]["sources"] for event in result["corporate_actions"])
    assert result["corporate_actions_ledger"] == w.ledger_data
    source = result["source"]
    import hashlib
    assert source["corporate_actions_file"] == str(w.ledger_path.resolve())
    assert source["corporate_actions_sha256"] == hashlib.sha256(w.ledger_path.read_bytes()).hexdigest()
    assert source["corporate_actions_coverage"] == w.ledger_data["coverage"]
    assert source["corporate_actions_schema"] == "corporate-actions-v1"
    assert source["tax_policy"] == result["tax_policy"] == "gross_before_personal_dividend_tax"
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in w.paths.root.rglob("*") if p.is_file()}
    assert before == after


def test_ca_end_before_payment_and_listing_preserves_receivable_and_locked(ca_workspace):
    w = ca_workspace
    w.model.actions = [1, 0]
    result = w.service.run(strict_payload(w, hi=32))
    final = result["daily"][-1]
    assert final["dividend_receivable"] > 0 and final["locked_shares"] > 0
    assert result["summary"]["final_dividend_receivable"] == final["dividend_receivable"]
    assert result["summary"]["final_locked_shares"] == final["locked_shares"]
    assert result["summary"]["final_equity"] == final["cash"] + final["position_value"] + final["dividend_receivable"]
    assert [event["stage"] for event in result["corporate_actions"]] == ["record", "ex_entitlement"]


def test_ca_baseline_is_identical_constant_buy_environment(ca_workspace, monkeypatch):
    from stock_ai.env.corporate_action_env import CorporateActionExecutionEnv
    w = ca_workspace
    w.model.actions = [1] * 4
    monkeypatch.setattr(replay.Backtester, "_buy_and_hold", lambda *a: pytest.fail("v2 禁止 legacy 基准"))
    monkeypatch.setattr(CorporateActionExecutionEnv, "buy_and_hold", lambda *a: pytest.fail("直接读取环境独立恒买曲线"))
    result = w.service.run(strict_payload(w))
    assert all(row["equity"] == row["baseline"] for row in result["daily"])
    assert result["summary"]["excess_return"] == pytest.approx(0)
    assert "同一已核验账本" in result["baseline_note"]


@pytest.mark.parametrize("single_day", [False, True])
@pytest.mark.parametrize("problem", ["missing", "unverified", "insufficient", "unknown_kind", "missing_listing", "missing_pay"])
def test_ca_ledger_fail_closed_even_single_day(ca_workspace, single_day, problem):
    w = ca_workspace
    if problem == "missing":
        w.ledger_path.unlink()
    else:
        if problem == "unverified":
            w.ledger_data["coverage"]["status"] = "pending"
        elif problem == "insufficient":
            w.ledger_data["coverage"]["end"] = w.df.date.iloc[29].strftime("%Y%m%d")
        elif problem == "unknown_kind":
            w.ledger_data["events"][0]["kind"] = "rights_issue"
            w.ledger_data["events"][0]["record_date"] = w.df.date.iloc[30].strftime("%Y%m%d")
        else:
            w.ledger_data["events"][0].pop("listing_date" if problem == "missing_listing" else "pay_date")
        w.ledger_path.write_text(json.dumps(w.ledger_data), encoding="utf-8")
    with pytest.raises(ReplayError, match="账本|公司行为"):
        w.service.run(strict_payload(w, hi=30 if single_day else 34))
    assert w.model.seen == []


def test_ca_first_day_record_log_does_not_infer_prior_entitlement(ca_workspace):
    w = ca_workspace
    w.model.actions = [1, 0]
    result = w.service.run(strict_payload(w, lo=31, hi=33))
    first, ex_day = result["daily"][:2]
    assert first["shares"] == first["dividend_income"] == first["dividend_receivable"] == 0
    assert first["corporate_actions"][0]["stage"] == "record"
    assert first["corporate_actions"][0]["eligible_shares"] == 0
    assert ex_day["shares"] > 0 and ex_day["dividend_income"] == ex_day["dividend_receivable"] == ex_day["locked_shares"] == 0


def test_ca_single_day_covered_no_environment_or_prediction(ca_workspace, monkeypatch):
    from stock_ai.scale import unified
    w = ca_workspace
    monkeypatch.setattr(unified, "make_strict_env", lambda *a, **k: pytest.fail("单日只估值，不能创建两 bar 环境"))
    result = w.service.run(strict_payload(w, hi=30))
    row = result["daily"][0]
    assert w.model.seen == [] and result["trades"] == result["corporate_actions"] == []
    assert row["equity"] == row["cash"] == row["portfolio_value"] == row["baseline"] == result["initial_cash"]
    assert row["available_shares"] == row["locked_shares"] == row["dividend_receivable"] == row["dividend_income"] == 0
    assert row["limit_reference"] is row["limit_up"] is row["limit_down"] is None


@pytest.mark.parametrize("value", ["https://invalid.example/cache", "//invalid/cache", "../../outside", "", None])
def test_ca_manifest_ledger_directory_is_controlled(ca_workspace, value):
    w = ca_workspace
    patch_manifest(w, corporate_actions_dir=value)
    with pytest.raises(ReplayError, match="安全|受控|缺失"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


def test_ca_ledger_directory_symlink_cannot_redirect_fixed_research_path(ca_workspace, tmp_path):
    w = ca_workspace
    outside = tmp_path.parent / f"{tmp_path.name}-outside-ledger-dir"
    outside.mkdir()
    (outside / "600519.json").write_bytes(w.ledger_path.read_bytes())
    linked = w.paths.root / "linked-ca"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("当前 Windows 用户没有符号链接权限")
    patch_manifest(w, corporate_actions_dir="linked-ca")
    with pytest.raises(ReplayError, match="受控"):
        w.service.run(strict_payload(w))
    w.ledger_path.unlink()
    w.ledger_dir.rmdir()
    w.ledger_dir.symlink_to(outside, target_is_directory=True)
    patch_manifest(w, execution_version="strict-t1-open-v1")
    with pytest.raises(ReplayError, match="受控"):
        w.service.run(strict_payload(w, execution_mode="corporate_actions_research"))
    assert w.model.seen == []


def test_ca_ledger_external_symbol_symlink_rejected(ca_workspace, tmp_path):
    w = ca_workspace
    outside = tmp_path.parent / f"{tmp_path.name}-outside-ledger.json"
    outside.write_bytes(w.ledger_path.read_bytes())
    w.ledger_path.unlink()
    try:
        w.ledger_path.symlink_to(outside)
    except OSError:
        pytest.skip("当前 Windows 用户没有符号链接权限")
    with pytest.raises(ReplayError, match="路径不安全"):
        w.service.run(strict_payload(w))


def test_ca_research_opt_in_keeps_original_policy_and_manifest(ca_workspace):
    w = ca_workspace
    patch_manifest(w, execution_version="strict-t1-open-v1", reward_version="excess-causal-close-v1",
                   corporate_actions_dir="../../must-not-use")
    item = w.service.catalog()["unified_runs"][0]
    assert "遗漏" in item["warning"]
    w.model.actions = [1] * 4
    original = w.service.run(strict_payload(w))
    assert original["execution_version"] == "strict-t1-open-v1"
    assert "dividend_receivable" not in original["daily"][2]
    assert any("不能视为投资总回报" in warning for warning in original["warnings"])
    w.model.seen.clear()
    before = {str(p): p.read_bytes() for p in w.paths.root.rglob("*") if p.is_file()}
    corrected = w.service.run(strict_payload(w, execution_mode="corporate_actions_research"))
    assert corrected["training_execution_version"] == corrected["model"]["execution_version"] == "strict-t1-open-v1"
    assert corrected["execution_version"] == "strict-t1-open-ca-v2"
    assert corrected["selection"] == corrected["execution_mode"] == "corporate_actions_research"
    assert corrected["training_reward_version"] == "excess-causal-close-v1"
    assert corrected["reward_version"] == "excess-ca-constant-buy-v2"
    assert any("不是已修复并重训" in warning and "错误 v1 奖励" in warning for warning in corrected["warnings"])
    assert corrected["source"]["corporate_actions_file"] == str(w.ledger_path.resolve())
    assert corrected["summary"]["final_equity"] > original["summary"]["final_equity"]
    assert before == {str(p): p.read_bytes() for p in w.paths.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("model_kind", ["auto", "unit", "legacy", "v2"])
def test_ca_research_rejects_non_v1_models(ca_workspace, model_kind):
    w = ca_workspace
    selected = w.model_id
    if model_kind in ("auto", "unit"):
        selected = "auto" if model_kind == "auto" else "600519_2022"
    elif model_kind == "legacy":
        patch_manifest(w, execution_version="legacy-same-day-close-v1")
    with pytest.raises(ReplayError, match="仅支持") as exc:
        w.service.run(payload(w, model_id=selected, execution_mode="corporate_actions_research"))
    assert exc.value.status == 400 and w.model.seen == []


@pytest.mark.parametrize("mode", [None, "", "corporate_actions", 1, {}, []])
def test_ca_execution_mode_validation(workspace, mode):
    with pytest.raises(ReplayError, match="execution_mode") as exc:
        workspace.service.run(payload(workspace, execution_mode=mode))
    assert exc.value.status == 400


def test_ca_receivable_after_full_sale_is_not_position_or_floating_pnl(ca_workspace):
    w = ca_workspace
    event = w.ledger_data["events"][0]
    event.update(bonus_ratio="0", transfer_ratio="0", listing_date=None)
    w.ledger_path.write_text(json.dumps(w.ledger_data), encoding="utf-8")
    reference = w.df.close.iloc[31] - 1
    for index in range(32, 35):
        w.df.loc[index, ["open", "high", "low", "close"]] = reference
    w.df.to_parquet(w.cfg.paths.raw / "600519.parquet", index=False)
    w.model.actions = [1, 0, 2]
    result = w.service.run(strict_payload(w, hi=33))
    last = result["daily"][-1]
    assert last["shares"] == last["available_shares"] == last["locked_shares"] == 0
    assert last["position"] == last["position_value"] == last["unrealized_pnl"] == 0
    assert last["dividend_receivable"] == last["dividend_income"] > 0
    assert last["equity"] == last["cash"] + last["dividend_receivable"]
    assert result["summary"]["holding_days"] == 2
    assert result["summary"]["final_dividend_receivable"] == last["dividend_receivable"]


def test_ca_verified_empty_ledger_is_explicitly_allowed(ca_workspace):
    w = ca_workspace
    w.ledger_data["events"] = []
    w.ledger_path.write_text(json.dumps(w.ledger_data), encoding="utf-8")
    result = w.service.run(strict_payload(w))
    assert result["corporate_actions"] == []
    assert result["corporate_actions_ledger"]["coverage"]["status"] == "verified"
    assert all(row["dividend_receivable"] == row["locked_shares"] == 0 for row in result["daily"])


def test_ca_coverage_cannot_shrink_requested_range_to_cached_dates(ca_workspace):
    w = ca_workspace
    start = w.df.date.iloc[30].strftime("%Y%m%d")
    w.ledger_data["coverage"]["start"] = start
    w.ledger_path.write_text(json.dumps(w.ledger_data), encoding="utf-8")
    earlier = (w.df.date.iloc[30].to_pydatetime() - timedelta(days=1)).strftime("%Y%m%d")
    with pytest.raises(ReplayError, match="覆盖不足"):
        w.service.run(strict_payload(w, start=earlier, end=start))
    assert w.model.seen == []


def test_ca_replay_never_fetches_or_trains(ca_workspace, monkeypatch):
    import socket
    import requests
    w = ca_workspace
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("公司行为回放禁止联网"))
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("公司行为回放禁止补数"))
    assert w.service.run(strict_payload(w))["ok"]
    patch_manifest(w, execution_version="strict-t1-open-v1")
    assert w.service.run(strict_payload(w, execution_mode="corporate_actions_research"))["ok"]


def test_ca_http_endpoint_research_and_missing_coverage(ca_workspace, http):
    w = ca_workspace
    _, _, server = http
    patch_manifest(w, execution_version="strict-t1-open-v1", reward_version="excess-causal-close-v1")
    status, body = request(server, "/api/replay", strict_payload(w, execution_mode="corporate_actions_research"))
    data = json.loads(body)
    assert status == 200 and data["execution_version"] == "strict-t1-open-ca-v2"
    assert data["training_execution_version"] == "strict-t1-open-v1"
    w.ledger_data["coverage"]["status"] = "pending"
    w.ledger_path.write_text(json.dumps(w.ledger_data), encoding="utf-8")
    status, body = request(server, "/api/replay", strict_payload(w, hi=30, execution_mode="corporate_actions_research"))
    assert status == 400 and not json.loads(body)["ok"]
    assert "verified" in json.loads(body)["error"]


def test_ca_research_does_not_bypass_coverage_or_accept_request_paths(ca_workspace):
    w = ca_workspace
    patch_manifest(w, execution_version="strict-t1-open-v1")
    add_ca_ledger(w, status="pending")
    with pytest.raises(ReplayError, match="账本校验"):
        w.service.run(strict_payload(w, hi=30, execution_mode="corporate_actions_research"))
    with pytest.raises(ReplayError, match="不支持的请求字段"):
        w.service.run(strict_payload(w, execution_mode="corporate_actions_research", corporate_actions_dir=str(w.ledger_dir)))
    assert w.model.seen == []


def test_ca_frozen_sources_match_complete_controlled_files(ca_workspace):
    w = ca_workspace
    frozen = freeze_ca_sources(w)
    result = w.service.run(strict_payload(w))
    source = result["source"]
    audit = source["training_source_audit"]
    assert audit["status"] == "matched" and audit["symbol_in_training_sources"]
    assert source["feature_sha256"] == frozen["sha256"]
    assert source["execution_sha256"] == frozen["execution_sha256"]
    assert source["corporate_actions_sha256"] == frozen["corporate_actions_sha256"]
    for name in ("feature", "execution", "corporate_actions"):
        assert audit[name]["status"] == "matched"
        assert audit[name]["current_sha256"] == audit[name]["training_sha256"]
    assert source["feature_file"] == str((w.feature_dir / "600519.parquet").resolve())
    assert source["execution_file"] == str((w.cfg.paths.raw / "600519.parquet").resolve())
    assert source["corporate_actions_file"] == str(w.ledger_path.resolve())
    assert any("完整文件 SHA256 均与" in warning for warning in result["warnings"])


@pytest.mark.parametrize("key", ["sha256", "execution_sha256", "corporate_actions_sha256"])
@pytest.mark.parametrize("value", [None, "", "not-a-hash", "0" * 64])
def test_ca_frozen_hash_missing_or_mismatched_refuses_before_prediction(ca_workspace, key, value):
    w = ca_workspace
    frozen = freeze_ca_sources(w)
    if value is None:
        frozen.pop(key)
    else:
        frozen[key] = value
    patch_manifest(w, sources=[frozen])
    with pytest.raises(ReplayError, match="SHA256|冻结训练来源"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


@pytest.mark.parametrize("kind", ["feature", "execution", "corporate_actions"])
def test_ca_frozen_complete_file_hash_includes_outside_replay_window(ca_workspace, kind):
    w = ca_workspace
    freeze_ca_sources(w)
    if kind == "corporate_actions":
        # 仅空白变化也必须改变冻结完整文件指纹，不能重序列化后比较。
        w.ledger_path.write_text(w.ledger_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        frame = (w.hfq if kind == "feature" else w.df).copy()
        frame.loc[79, "volume"] += 1
        directory = w.feature_dir if kind == "feature" else w.cfg.paths.raw
        frame.to_parquet(directory / "600519.parquet", index=False)
    with pytest.raises(ReplayError, match="完整文件 SHA256"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


def test_ca_new_symbol_discloses_current_hashes_without_frozen_match_claim(ca_workspace):
    from stock_ai.utils.hashing import sha256_file
    w = ca_workspace
    freeze_ca_sources(w)
    symbol = "000001"
    w.df.to_parquet(w.cfg.paths.raw / f"{symbol}.parquet", index=False)
    w.hfq.to_parquet(w.feature_dir / f"{symbol}.parquet", index=False)
    ledger = {**w.ledger_data, "symbol": symbol}
    ledger_path = w.ledger_dir / f"{symbol}.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    result = w.service.run(strict_payload(w, symbol=symbol))
    source = result["source"]
    assert source["training_source_audit"]["status"] == "not_in_training_sources"
    assert not source["training_source_audit"]["symbol_in_training_sources"]
    assert source["feature_sha256"] == sha256_file(w.feature_dir / f"{symbol}.parquet")
    assert source["execution_sha256"] == sha256_file(w.cfg.paths.raw / f"{symbol}.parquet")
    assert source["corporate_actions_sha256"] == sha256_file(ledger_path)
    assert any("不宣称与训练来源一致" in warning for warning in result["warnings"])


def test_ca_research_historical_raw_hash_not_recorded_is_not_backfilled(ca_workspace):
    w = ca_workspace
    frozen = freeze_ca_sources(w)
    frozen.pop("execution_sha256")
    frozen.pop("corporate_actions_sha256")
    patch_manifest(w, execution_version="strict-t1-open-v1", sources=[frozen])
    result = w.service.run(strict_payload(w, execution_mode="corporate_actions_research"))
    audit = result["source"]["training_source_audit"]
    assert audit["status"] == "partial"
    assert audit["feature"]["status"] == "matched"
    assert audit["execution"]["status"] == "not_recorded"
    assert audit["execution"]["training_sha256"] is None
    assert audit["execution"]["current_sha256"] == result["source"]["execution_sha256"]
    assert audit["corporate_actions"]["status"] == "current_reviewed_ledger_not_training_source"
    assert audit["corporate_actions"]["training_sha256"] is None
    assert any("not_recorded" in warning for warning in result["warnings"])


def test_ca_research_recorded_raw_hash_must_still_match(ca_workspace):
    w = ca_workspace
    frozen = freeze_ca_sources(w)
    frozen["execution_sha256"] = "0" * 64
    patch_manifest(w, execution_version="strict-t1-open-v1", sources=[frozen])
    with pytest.raises(ReplayError, match="execution 完整文件 SHA256"):
        w.service.run(strict_payload(w, execution_mode="corporate_actions_research"))
    assert w.model.seen == []


def test_ca_frozen_source_file_does_not_redirect_controlled_cache(ca_workspace):
    w = ca_workspace
    frozen = freeze_ca_sources(w)
    wrong = w.df.copy()
    wrong.loc[79, "volume"] += 100
    outside = w.paths.root / "untrusted-raw.parquet"
    wrong.to_parquet(outside, index=False)
    from stock_ai.utils.hashing import sha256_file
    frozen.update(execution_file=str(outside), execution_sha256=sha256_file(outside))
    patch_manifest(w, sources=[frozen])
    with pytest.raises(ReplayError, match="execution 完整文件 SHA256"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


@pytest.mark.parametrize("problem", ["missing", "empty", "null", "object", "non_dict",
                                      "no_symbol", "numeric_symbol", "short_symbol", "unicode_symbol",
                                      "duplicate_match", "duplicate_other", "other_missing_hash", "other_invalid_hash"])
def test_ca_manifest_sources_must_be_nonempty_valid_unique_list(ca_workspace, monkeypatch, problem):
    w = ca_workspace
    source = freeze_ca_sources(w)
    values = {
        "empty": [], "null": None, "object": {}, "non_dict": [None],
        "no_symbol": [{}], "numeric_symbol": [{"symbol": 600519}],
        "short_symbol": [{"symbol": "60051"}], "unicode_symbol": [{"symbol": "６００５１９"}],
        "duplicate_match": [source, dict(source)],
        "duplicate_other": [source, {"symbol": "000001"}, {"symbol": "000001"}],
        "other_missing_hash": [{"symbol": "000001"}],
        "other_invalid_hash": [{**source, "symbol": "000001", "execution_sha256": "bad"}],
    }
    if problem == "missing":
        w.manifest.pop("sources")
        patch_manifest(w)
    else:
        patch_manifest(w, sources=values[problem])
    monkeypatch.setattr(replay, "_replay_model", lambda *a: pytest.fail("非法 sources 必须在模型加载前拒绝"))
    with pytest.raises(ReplayError, match="sources|重复|冻结训练来源"):
        w.service.run(strict_payload(w))
    assert w.model.seen == []


@pytest.mark.parametrize("problem", ["missing", "empty"])
def test_ca_v1_research_can_disclose_absent_historical_sources(ca_workspace, problem):
    w = ca_workspace
    if problem == "missing":
        w.manifest.pop("sources")
    else:
        w.manifest["sources"] = []
    patch_manifest(w, execution_version="strict-t1-open-v1")
    result = w.service.run(strict_payload(w, execution_mode="corporate_actions_research"))
    audit = result["source"]["training_source_audit"]
    assert audit["status"] == "not_recorded" and not audit["symbol_in_training_sources"]
    assert audit["feature"]["status"] == audit["execution"]["status"] == "not_recorded"
    assert audit["corporate_actions"]["status"] == "current_reviewed_ledger_not_training_source"


@pytest.mark.parametrize("kind", ["feature", "execution"])
def test_ca_source_audit_hashes_exact_parquet_bytes_used_for_inference(ca_workspace, monkeypatch, kind):
    """Replace the path after read: parsed data and its audit must retain one snapshot."""
    import hashlib
    from io import BytesIO
    from pathlib import Path
    from stock_ai.scale import unified

    w = ca_workspace
    frozen = freeze_ca_sources(w)
    directory = w.feature_dir if kind == "feature" else w.cfg.paths.raw
    target = (directory / "600519.parquet").resolve()
    replacement = (w.hfq if kind == "feature" else w.df).copy()
    replacement["volume"] *= 2
    if kind == "feature":
        replacement.loc[:, ["open", "high", "low", "close"]] *= 1.1
    else:
        replacement.loc[:, ["open", "high", "low", "close"]] += .01
    buffer = BytesIO()
    replacement.to_parquet(buffer, index=False)
    changed_bytes = buffer.getvalue()
    original = Path.read_bytes
    reads = []

    def replace_after_snapshot(path):
        data = original(path)
        if path.resolve() == target:
            reads.append(data)
            path.write_bytes(changed_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_snapshot)
    w.model.actions = [1, 0, 2, 0]
    result = w.service.run(strict_payload(w))
    assert len(reads) == 1
    key = "feature_sha256" if kind == "feature" else "execution_sha256"
    frozen_key = "sha256" if kind == "feature" else "execution_sha256"
    assert result["source"][key] == hashlib.sha256(reads[0]).hexdigest() == frozen[frozen_key]
    assert hashlib.sha256(original(target)).hexdigest() != result["source"][key]
    assert result["source"]["training_source_audit"]["status"] == "matched"
    for i, row in enumerate(result["daily"]):
        assert row["close"] == w.df.close.iloc[30 + i]
    _, values = unified.relative_features(w.hfq)
    expected = unified.observations(unified.normalize(values, w.norm), w.manifest["window"])
    np.testing.assert_array_equal(np.asarray(w.model.seen), expected[30:34])

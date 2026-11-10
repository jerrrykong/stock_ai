"""巨潮源快照的纯模拟测试，不调用真实接口、不写真实公司行为目录。"""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from stock_ai.data import corporate_action_sources as sources
from stock_ai.data.corporate_actions import CorporateActionError, load_ledger

_ENDPOINT = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"
_URL = _ENDPOINT + "?scode=300750"


def _row(**overrides):
    row = {
        "F012N": 25.2, "F018D": "2023-04-25", "F006D": "2023-04-18",
        "F011N": 8.0, "F010N": None, "F007V": "10转增8股派25.2元(含税)",
        "F025D": "2023-04-26", "F023D": "2023-04-26", "F001V": "2022年报",
        "F020D": "2023-04-26", "F044V": "年度分红",
    }
    row.update(overrides)
    return row


def _payload(rows=None):
    records = [_row()] if rows is None else rows
    return {"resultcode": 200, "resultmsg": "success", "count": len(records),
            "total": len(records), "records": records}


def _response(payload=None, raw=None, *, status=200, url=_URL):
    if raw is None:
        raw = json.dumps(_payload() if payload is None else payload,
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return SimpleNamespace(
        status_code=status, url=url, content=raw, encoding="utf-8", close=Mock(),
        headers={"Content-Type": "application/json;charset=UTF-8"},
    )


@pytest.fixture
def mocked_request(monkeypatch):
    response = _response()
    post = Mock(return_value=response)
    requests = SimpleNamespace(post=post, RequestException=TimeoutError)
    deps = (requests, SimpleNamespace(__version__="测试版"), object(), object())
    monkeypatch.setattr(sources, "_request_dependencies", lambda: deps)
    key = Mock(return_value="仅请求中存在的动态头")
    monkeypatch.setattr(sources, "_enckey", key)
    return post, response, key


def test_snapshot_preserves_exact_original_bytes_and_metadata(tmp_path, mocked_request):
    post, response, _ = mocked_request
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path / "evidence")
    raw_path = Path(snapshot["raw_response_path"])
    meta_path = Path(snapshot["metadata_path"])
    assert raw_path.read_bytes() == response.content
    assert snapshot["response_sha256"] == hashlib.sha256(response.content).hexdigest()
    assert snapshot["response_bytes"] == len(response.content)
    assert json.loads(meta_path.read_text(encoding="utf-8")) == snapshot
    assert snapshot["method"] == "POST"
    assert snapshot["url"] == _URL
    assert snapshot["params"] == {"scode": "300750"}
    assert snapshot["resultcode"] == 200
    assert snapshot["count"] == snapshot["total"] == 1
    assert snapshot["snapshot_counts_matched"] is True
    assert snapshot["akshare_version"] == "测试版"
    assert datetime.fromisoformat(snapshot["fetched_at"]).utcoffset().total_seconds() == 0
    assert snapshot["status"] == "unreviewed"
    post.assert_called_once()
    assert post.call_args.args == (_ENDPOINT,)
    assert post.call_args.kwargs["params"] == {"scode": "300750"}
    assert post.call_args.kwargs["timeout"] == 20
    assert post.call_args.kwargs["allow_redirects"] is False
    assert post.call_args.kwargs["headers"]["Accept-Enckey"] == "仅请求中存在的动态头"
    assert "仅请求中存在的动态头" not in meta_path.read_text(encoding="utf-8")
    response.close.assert_called_once()


def test_candidates_do_not_invent_zero_listing_or_execution_method(tmp_path, mocked_request):
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    event = snapshot["candidates"][0]
    assert event["status"] == "unreviewed"
    assert event["cash_per_share_gross"] == "2.52"
    assert Decimal(event["transfer_ratio"]) == Decimal("0.8")
    assert event["bonus_ratio"] is None
    assert event["listing_date"] is None
    assert event["reference_method"] is None
    assert event["share_credit_date"] == "20230426"
    assert event["record_date"] == "20230425"
    assert event["ex_date"] == event["pay_date"] == "20230426"
    assert event["announcement_date"] == "20230418"
    assert event["sources"] == [_URL]
    assert event["source_row_index"] == 0
    assert "bonus_ratio" in event["unknown_fields"]
    assert "listing_date" in event["unknown_fields"]
    assert "reference_method" in event["unknown_fields"]
    assert event["invalid_fields"] == []
    assert "coverage" not in snapshot
    assert "events" not in snapshot
    assert "implemented" not in {snapshot["status"], event["status"]}
    with pytest.raises(CorporateActionError):
        load_ledger(snapshot["metadata_path"])


def test_source_status_or_coverage_cannot_certify_snapshot(tmp_path, mocked_request):
    post, _, _ = mocked_request
    row = _row()
    row.update(status="implemented", listing_date="20230426", reference_method="standard_cash_bonus")
    payload = _payload([row])
    payload.update(status="verified", coverage={"status": "verified"})
    post.return_value = _response(payload)
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert snapshot["status"] == "unreviewed"
    assert "coverage" not in snapshot
    assert snapshot["candidates"][0]["status"] == "unreviewed"
    assert snapshot["candidates"][0]["listing_date"] is None
    assert snapshot["candidates"][0]["reference_method"] is None


def test_repeated_snapshots_never_replace_earlier_files(tmp_path, mocked_request):
    first = sources.fetch_cninfo_snapshot("300750", tmp_path)
    first_raw = Path(first["raw_response_path"]).read_bytes()
    first_metadata = Path(first["metadata_path"]).read_bytes()
    second = sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert second["raw_response_path"] != first["raw_response_path"]
    assert second["metadata_path"] != first["metadata_path"]
    assert Path(first["raw_response_path"]).read_bytes() == first_raw
    assert Path(first["metadata_path"]).read_bytes() == first_metadata
    assert len(list(tmp_path.iterdir())) == 4


@pytest.mark.parametrize("symbol", ["30075", "300750/../x", "３００７５０", 300750, None])
def test_invalid_symbol_fails_before_request(tmp_path, mocked_request, symbol):
    post, _, key = mocked_request
    with pytest.raises(CorporateActionError, match="6 位"):
        sources.fetch_cninfo_snapshot(symbol, tmp_path)
    post.assert_not_called()
    key.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", ["data/corporate_actions", "data/corporate_actions/subdir",
                                    "DATA/Corporate_Actions"])
def test_unreviewed_snapshots_cannot_enter_execution_ledger_directory(
    tmp_path, mocked_request, suffix,
):
    post, _, _ = mocked_request
    with pytest.raises(CorporateActionError, match="不得写入"):
        sources.fetch_cninfo_snapshot("300750", tmp_path / suffix)
    post.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", [301, 302, 403, 500])
def test_http_failure_writes_no_success_artifact(tmp_path, mocked_request, status):
    post, _, _ = mocked_request
    response = _response(status=status)
    post.return_value = response
    with pytest.raises(CorporateActionError, match="HTTP 失败"):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert list(tmp_path.iterdir()) == []
    response.close.assert_called_once()


@pytest.mark.parametrize("url", [
    "http://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139?scode=300750",
    "https://example.com/api/sysapi/p_sysapi1139?scode=300750",
    "https://webapi.cninfo.com.cn/api/other?scode=300750",
])
def test_response_downgrade_or_redirect_is_rejected(tmp_path, mocked_request, url):
    post, _, _ = mocked_request
    response = _response(url=url)
    post.return_value = response
    with pytest.raises(CorporateActionError, match="URL 偏离"):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    response.close.assert_called_once()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("resultcode", [401, "200", True, None])
def test_http_200_is_not_business_success(tmp_path, mocked_request, resultcode):
    post, _, _ = mocked_request
    payload = _payload()
    payload["resultcode"] = resultcode
    post.return_value = _response(payload)
    with pytest.raises(CorporateActionError, match="业务失败"):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("count", 0), ("total", 2), ("count", "1"), ("total", "1"),
    ("count", True), ("total", -1), ("total", None),
    ("records", {}), ("records", ["not-object"]),
])
def test_missing_or_truncated_response_counts_fail(tmp_path, mocked_request, field, value):
    post, _, _ = mocked_request
    payload = _payload()
    payload[field] = value
    post.return_value = _response(payload)
    with pytest.raises(CorporateActionError):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("pages", 2), ("pages", "1"), ("pages", True), ("totalPages", 2),
    ("pageCount", 2), ("page", 2), ("pageNum", 0), ("pageNumber", 2),
    ("hasMore", True), ("has_more", 0), ("hasNext", True), ("has_next", None),
])
def test_response_cannot_silently_ignore_pagination(tmp_path, mocked_request, field, value):
    post, _, _ = mocked_request
    payload = _payload()
    payload[field] = value
    post.return_value = _response(payload)
    with pytest.raises(CorporateActionError, match="分页|后续页"):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_explicit_single_page_response_is_allowed_but_unreviewed(tmp_path, mocked_request):
    post, _, _ = mocked_request
    payload = _payload()
    payload.update(pages=1, pageNumber=1, hasMore=False)
    post.return_value = _response(payload)
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert snapshot["status"] == "unreviewed"
    assert "coverage" not in snapshot


def test_empty_snapshot_is_not_a_negative_coverage_proof(tmp_path, mocked_request):
    post, _, _ = mocked_request
    post.return_value = _response(_payload([]))
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert snapshot["count"] == snapshot["total"] == 0
    assert snapshot["candidates"] == []
    assert snapshot["status"] == "unreviewed"
    assert "coverage" not in snapshot


@pytest.mark.parametrize("raw", [b"html error", b"\xff", b"[]", b"null",
                                  b'{"resultcode":200,"resultcode":401}',
                                  b'{"resultcode":200,"records":[],"count":NaN,"total":0}'])
def test_invalid_or_ambiguous_json_is_rejected(tmp_path, mocked_request, raw):
    post, _, _ = mocked_request
    post.return_value = _response(raw=raw)
    with pytest.raises(CorporateActionError):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["F012N", "F010N", "F011N"])
@pytest.mark.parametrize("value", [None, "", "NaN", "Infinity", "-1", True, {}, "未知"])
def test_unknown_or_invalid_leg_stays_unknown(tmp_path, mocked_request, field, value):
    post, _, _ = mocked_request
    post.return_value = _response(_payload([_row(**{field: value})]))
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    target = {"F012N": "cash_per_share_gross", "F010N": "bonus_ratio",
              "F011N": "transfer_ratio"}[field]
    event = snapshot["candidates"][0]
    assert event[target] is None
    assert target in event["unknown_fields"]
    if value is not None:
        assert target in event["invalid_fields"]
    assert event["status"] == "unreviewed"


def test_missing_three_legs_are_not_defaulted_to_zero(tmp_path, mocked_request):
    post, _, _ = mocked_request
    row = _row()
    for name in ("F012N", "F010N", "F011N"):
        del row[name]
    post.return_value = _response(_payload([row]))
    event = sources.fetch_cninfo_snapshot("300750", tmp_path)["candidates"][0]
    for name in ("cash_per_share_gross", "bonus_ratio", "transfer_ratio"):
        assert event[name] is None
        assert name in event["unknown_fields"]


def test_leg_conversion_is_exact_even_beyond_decimal_context_precision(tmp_path, mocked_request):
    post, _, _ = mocked_request
    row = _row(F012N="25.200000000000000000000000000001", F010N=0, F011N="8")
    post.return_value = _response(_payload([row]))
    event = sources.fetch_cninfo_snapshot("300750", tmp_path)["candidates"][0]
    assert event["cash_per_share_gross"] == "2.5200000000000000000000000000001"
    assert Decimal(event["bonus_ratio"]) == Decimal(0)
    assert Decimal(event["transfer_ratio"]) == Decimal("0.8")


@pytest.mark.parametrize("value", ["2023-02-29", "20230426", "2023-04-26 00:00:00", "--", 20230426])
def test_invalid_dates_are_not_silently_repaired(tmp_path, mocked_request, value):
    post, _, _ = mocked_request
    post.return_value = _response(_payload([_row(F025D=value)]))
    event = sources.fetch_cninfo_snapshot("300750", tmp_path)["candidates"][0]
    assert event["share_credit_date"] is None
    assert event["listing_date"] is None
    assert "share_credit_date" in event["invalid_fields"]


def test_same_day_regular_and_special_rows_remain_separate_candidates(tmp_path, mocked_request):
    post, _, _ = mocked_request
    first = _row(F012N="20.11", F011N=None)
    second = copy.deepcopy(first)
    second.update(F012N="30.17", F044V="特别分红")
    post.return_value = _response(_payload([first, second]))
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert len(snapshot["candidates"]) == 2
    assert snapshot["candidates"][0]["candidate_id"] != snapshot["candidates"][1]["candidate_id"]
    assert [row["cash_per_share_gross"] for row in snapshot["candidates"]] == ["2.011", "3.017"]
    assert snapshot["status"] == "unreviewed"


def test_unexpected_text_field_types_remain_unknown_but_raw_response_is_saved(
    tmp_path, mocked_request,
):
    post, response, _ = mocked_request
    response = _response(_payload([_row(F044V=12.3, F007V={"unexpected": 3.2})]))
    post.return_value = response
    snapshot = sources.fetch_cninfo_snapshot("300750", tmp_path)
    event = snapshot["candidates"][0]
    assert event["distribution_type"] is None
    assert event["source_plan"] is None
    assert "distribution_type" in event["invalid_fields"]
    assert "source_plan" in event["invalid_fields"]
    assert Path(snapshot["raw_response_path"]).read_bytes() == response.content


def test_timeout_is_reported_without_creating_output(tmp_path, mocked_request):
    post, _, _ = mocked_request
    post.side_effect = TimeoutError("测试超时")
    with pytest.raises(CorporateActionError, match="请求失败"):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("missing", ["requests", "akshare", "py_mini_racer"])
def test_missing_dependencies_are_explicit_and_never_search_project_venv(monkeypatch, missing):
    original = builtins.__import__
    fakes = {
        "akshare": SimpleNamespace(), "requests": SimpleNamespace(),
        "py_mini_racer": SimpleNamespace(), "akshare.datasets": SimpleNamespace(get_ths_js=Mock()),
    }

    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise ModuleNotFoundError(f"缺少 {name}")
        if name in fakes:
            return fakes[name]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(CorporateActionError, match="managed.*不会借用项目 .venv"):
        sources._request_dependencies()


def test_enckey_uses_existing_js_and_closes_runtime(tmp_path):
    script = tmp_path / "cninfo.js"
    script.write_text("function getResCode1(){return '本地动态头';}", encoding="utf-8")
    runtime = Mock()
    runtime.call.return_value = "本地动态头"
    context = Mock()
    context.__enter__ = Mock(return_value=runtime)
    context.__exit__ = Mock(return_value=False)
    mini_racer = SimpleNamespace(MiniRacer=Mock(return_value=context))
    get_ths_js = Mock(return_value=script)
    assert sources._enckey(mini_racer, get_ths_js) == "本地动态头"
    get_ths_js.assert_called_once_with("cninfo.js")
    runtime.eval.assert_called_once_with(script.read_text(encoding="utf-8"))
    runtime.call.assert_called_once_with("getResCode1")
    context.__exit__.assert_called_once()


def test_missing_local_script_is_dependency_error_not_network_fallback(tmp_path):
    get_ths_js = Mock(return_value=tmp_path / "missing.js")
    with pytest.raises(CorporateActionError, match="无法使用当前 AKShare"):
        sources._enckey(SimpleNamespace(MiniRacer=Mock()), get_ths_js)


def test_file_collision_never_overwrites_existing_response(tmp_path, mocked_request, monkeypatch):
    path_open = Path.open
    original = "原有审计证据".encode("utf-8")

    def collide(path, mode="r", *args, **kwargs):
        if mode == "xb":
            path.write_bytes(original)
        return path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", collide)
    with pytest.raises(CorporateActionError, match="不会覆盖"):
        sources.fetch_cninfo_snapshot("300750", tmp_path)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].read_bytes() == original

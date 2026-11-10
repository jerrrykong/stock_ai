"""公司行为 JSON 数据契约测试；仅写 pytest 临时目录，不联网或补建真实数据。"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from stock_ai.data.corporate_actions import (
    CorporateAction,
    CorporateActionError,
    CorporateActionLedger,
    load_ledger,
)

_SOURCE = "http://static.cninfo.com.cn/finalpage/2023-04-18/1216457838.PDF"


def _event(**overrides):
    event = {
        "event_id": "300750-20230426-distribution",
        "kind": "distribution",
        "status": "implemented",
        "record_date": "20230425",
        "ex_date": "20230426",
        "pay_date": "20230426",
        "listing_date": "20230426",
        "cash_per_share_gross": "2.52",
        "bonus_ratio": "0",
        "transfer_ratio": "0.8",
        "reference_method": "standard_cash_bonus",
        "sources": [_SOURCE],
        "fractional_policy": "reject",
        "listing_date_source": "sina_red_share_listing",
    }
    event.update(overrides)
    return event


def _document(events=None):
    return {
        "schema": "corporate-actions-v1",
        "symbol": "300750",
        "coverage": {
            "start": "20230101",
            "end": "20231231",
            "status": "verified",
            "scope": "all_shareholder_distributions_and_capital_changes",
            "sources": [_SOURCE],
            "verified_at": "2026-10-16T14:23:00+00:00",
            "notes": "仅用于校验的显式审阅声明，不代表真实完整区间证明。",
        },
        "events": [_event()] if events is None else events,
    }


def _write(tmp_path, document):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_preserves_decimals_dates_sources_and_raw_hash(tmp_path):
    path = _write(tmp_path, _document())
    before = path.read_bytes()
    ledger = load_ledger(path, symbol="300750")
    assert ledger.schema == "corporate-actions-v1"
    assert ledger.symbol == "300750"
    assert ledger.source_path == str(path.resolve())
    assert ledger.sha256 == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert isinstance(ledger.events, tuple)
    event = ledger.events[0]
    assert event.record_date == "20230425"
    assert event.ex_date == event.pay_date == event.listing_date == "20230426"
    assert event.cash_per_share_gross == Decimal("2.52")
    assert event.bonus_ratio == Decimal(0)
    assert event.transfer_ratio == Decimal("0.8")
    assert event.sources == (_SOURCE,)
    ledger.require_coverage("20230101", "20231231")


def test_to_dict_round_trips_without_derived_metadata(tmp_path):
    ledger = load_ledger(_write(tmp_path, _document()))
    document = ledger.to_dict()
    assert set(document) == {"schema", "symbol", "coverage", "events"}
    assert document["events"][0]["cash_per_share_gross"] == "2.52"
    second = load_ledger(_write(tmp_path, document))
    assert second.to_dict() == document
    document["coverage"]["sources"].clear()
    document["events"][0]["sources"].clear()
    assert ledger.coverage["sources"] == [_SOURCE]
    assert ledger.events[0].sources == (_SOURCE,)


def test_event_is_frozen_and_copies_source_list():
    source_list = [_SOURCE]
    event = CorporateAction(**_event(sources=source_list))
    source_list.clear()
    assert event.sources == (_SOURCE,)
    with pytest.raises(FrozenInstanceError):
        event.ex_date = "20230427"


def test_missing_file_and_invalid_json_raise_domain_error(tmp_path):
    with pytest.raises(CorporateActionError, match="无法读取"):
        load_ledger(tmp_path / "missing.json")
    path = tmp_path / "invalid.json"
    path.write_bytes(b"not json")
    with pytest.raises(CorporateActionError, match="合法 UTF-8 JSON"):
        load_ledger(path)
    path.write_bytes(b"\xff")
    with pytest.raises(CorporateActionError, match="合法 UTF-8 JSON"):
        load_ledger(path)


@pytest.mark.parametrize("location", ["ledger", "coverage", "event"])
def test_unknown_fields_are_rejected(tmp_path, location):
    document = _document()
    target = {
        "ledger": document,
        "coverage": document["coverage"],
        "event": document["events"][0],
    }[location]
    target["unknown"] = "不能静默忽略"
    with pytest.raises(CorporateActionError, match="未知字段"):
        load_ledger(_write(tmp_path, document))


@pytest.mark.parametrize("metadata", ["source_path", "sha256"])
def test_file_cannot_claim_its_own_path_or_hash(tmp_path, metadata):
    document = _document()
    document[metadata] = "用户提供但不可相信的派生值"
    with pytest.raises(CorporateActionError, match="未知字段"):
        load_ledger(_write(tmp_path, document))


@pytest.mark.parametrize("field", ["schema", "symbol", "coverage", "events"])
def test_missing_top_level_fields_raise(tmp_path, field):
    document = _document()
    del document[field]
    with pytest.raises(CorporateActionError, match="缺少字段"):
        load_ledger(_write(tmp_path, document))


@pytest.mark.parametrize("field", [
    "start", "end", "status", "scope", "sources", "verified_at", "notes",
])
def test_missing_coverage_fields_raise(tmp_path, field):
    document = _document(events=[])
    del document["coverage"][field]
    with pytest.raises(CorporateActionError, match="coverage.*缺少字段"):
        load_ledger(_write(tmp_path, document))


@pytest.mark.parametrize("field,value", [
    ("start", "20230229"),
    ("end", "2023-12-31"),
    ("start", 20230101),
    ("end", "20220101"),
    ("status", "unverified"),
    ("status", ""),
    ("scope", "dividend_only"),
    ("sources", []),
    ("sources", "https://example.com/review"),
    ("sources", ["file:///local/review.pdf"]),
    ("verified_at", " "),
    ("notes", ""),
])
def test_invalid_coverage_never_becomes_verified_empty_data(tmp_path, field, value):
    document = _document(events=[])
    document["coverage"][field] = value
    with pytest.raises(CorporateActionError):
        load_ledger(_write(tmp_path, document))


def test_zero_events_requires_explicit_verified_negative_coverage(tmp_path):
    document = _document(events=[])
    document["coverage"]["notes"] = "测试声明：已审阅该区间所有分配及资本变动，未发现事件。"
    ledger = load_ledger(_write(tmp_path, document))
    assert ledger.events == ()
    ledger.require_coverage("20230426", "20230426")
    del document["coverage"]
    with pytest.raises(CorporateActionError, match="缺少字段"):
        load_ledger(_write(tmp_path, document))


@pytest.mark.parametrize("start,end", [
    ("20221231", "20230426"),
    ("20230426", "20240101"),
    ("20221231", "20240101"),
])
def test_coverage_must_include_both_requested_endpoints(tmp_path, start, end):
    ledger = load_ledger(_write(tmp_path, _document()))
    with pytest.raises(CorporateActionError, match="覆盖不足"):
        ledger.require_coverage(start, end)


@pytest.mark.parametrize("start,end", [
    ("20230427", "20230426"),
    ("20230229", "20230426"),
    ("2023-04-25", "20230426"),
])
def test_invalid_requested_range_raises(tmp_path, start, end):
    ledger = load_ledger(_write(tmp_path, _document()))
    with pytest.raises(CorporateActionError):
        ledger.require_coverage(start, end)


@pytest.mark.parametrize("field,value", [
    ("schema", "corporate-actions-v2"),
    ("symbol", "30075"),
    ("symbol", "３００７５０"),
    ("symbol", 300750),
    ("events", {}),
    ("coverage", None),
])
def test_invalid_ledger_shape_or_identity_raises(tmp_path, field, value):
    document = _document()
    document[field] = value
    with pytest.raises(CorporateActionError):
        load_ledger(_write(tmp_path, document))


def test_requested_symbol_must_match_file(tmp_path):
    with pytest.raises(CorporateActionError, match="代码不匹配"):
        load_ledger(_write(tmp_path, _document()), symbol="600519")


@pytest.mark.parametrize("field,value", [
    ("event_id", ""),
    ("kind", " "),
    ("status", "proposed"),
    ("status", "announced"),
    ("record_date", None),
    ("ex_date", None),
    ("record_date", "20230426"),
    ("record_date", "20230427"),
    ("record_date", "20230229"),
    ("ex_date", "2023-04-26"),
    ("pay_date", None),
    ("pay_date", "20230425"),
    ("listing_date", None),
    ("listing_date", "20230425"),
    ("listing_date_source", None),
    ("listing_date_source", "unknown_source"),
    ("listing_date_source", "share_credit_date"),
    ("reference_method", "hfq_raw_ratio"),
    ("fractional_policy", "floor"),
    ("sources", []),
    ("sources", ["http://"]),
    ("sources", ["ftp://example.com/announcement.pdf"]),
    ("sources", ["https://example.com/a b"]),
])
def test_invalid_distribution_is_rejected_at_load(tmp_path, field, value):
    with pytest.raises(CorporateActionError):
        load_ledger(_write(tmp_path, _document([_event(**{field: value})])))


@pytest.mark.parametrize("source, listing", [
    ("arrival_equals_ex_date", "20230426"),
    ("sina_red_share_listing", "20230426"),
    ("sina_red_share_listing", "20230504"),
])
def test_declared_listing_sources_are_accepted(tmp_path, source, listing):
    event = _event(listing_date=listing, listing_date_source=source)
    ledger = load_ledger(_write(tmp_path, _document([event])))
    assert ledger.events[0].listing_date == listing
    assert ledger.events[0].listing_date_source == source


def test_arrival_equals_ex_date_cannot_carry_a_later_lockup_date(tmp_path):
    """到账日晚于除权日意味着可能锁定期，禁止借「到账即上市」口径混入。"""
    event = _event(listing_date="20230508", listing_date_source="arrival_equals_ex_date")
    with pytest.raises(CorporateActionError, match="必须等于 ex_date"):
        load_ledger(_write(tmp_path, _document([event])))


def test_listing_date_source_is_optional_without_listing_date(tmp_path):
    cash = _event(transfer_ratio="0", listing_date=None, listing_date_source=None)
    ledger = load_ledger(_write(tmp_path, _document([cash])))
    assert ledger.events[0].listing_date is None
    assert ledger.events[0].listing_date_source is None


@pytest.mark.parametrize("field", ["cash_per_share_gross", "bonus_ratio", "transfer_ratio"])
@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity", "-0.1", "未知", None, True])
def test_invalid_amounts_and_ratios_raise(tmp_path, field, value):
    with pytest.raises(CorporateActionError, match="非负有限数值"):
        load_ledger(_write(tmp_path, _document([_event(**{field: value})])))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numeric_constants_raise(tmp_path, constant):
    text = json.dumps(_document()).replace('"2.52"', constant)
    path = tmp_path / "invalid-number.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CorporateActionError, match="非有限数值"):
        load_ledger(path)


def test_json_duplicate_keys_raise_at_any_level(tmp_path):
    text = json.dumps(_document()).replace('"status": "implemented"',
                                            '"status": "implemented", "status": "proposed"')
    path = tmp_path / "duplicate-key.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CorporateActionError, match="重复字段"):
        load_ledger(path)


def test_zero_distribution_is_not_a_negative_coverage_record(tmp_path):
    event = _event(cash_per_share_gross="0", bonus_ratio="0", transfer_ratio="0")
    with pytest.raises(CorporateActionError, match="总分配不得为 0"):
        load_ledger(_write(tmp_path, _document([event])))


def test_numeric_json_values_preserve_decimal_precision(tmp_path):
    text = json.dumps(_document()).replace('"2.52"', "2.52000000000000000001")
    path = tmp_path / "decimal.json"
    path.write_text(text, encoding="utf-8")
    ledger = load_ledger(path)
    assert ledger.events[0].cash_per_share_gross == Decimal("2.52000000000000000001")


def test_cash_only_and_share_only_events_need_only_applicable_dates(tmp_path):
    cash = _event(transfer_ratio="0", listing_date=None)
    cash_ledger = load_ledger(_write(tmp_path, _document([cash])))
    cash_ledger.require_coverage("20230425", "20230427")
    share = _event(cash_per_share_gross="0", pay_date=None)
    share_ledger = load_ledger(_write(tmp_path, _document([share])))
    share_ledger.require_coverage("20230425", "20230427")


def test_duplicate_event_ids_raise_even_when_dates_differ(tmp_path):
    first = _event()
    second = _event(record_date="20230503", ex_date="20230504", pay_date="20230504",
                    listing_date="20230504")
    with pytest.raises(CorporateActionError, match="重复 event_id"):
        load_ledger(_write(tmp_path, _document([first, second])))


def test_same_ex_date_conflict_requires_explicit_review_and_merge(tmp_path):
    first = _event()
    second = _event(event_id="300750-extra-special", cash_per_share_gross="3.017")
    with pytest.raises(CorporateActionError, match="同 ex_date 事件冲突"):
        load_ledger(_write(tmp_path, _document([first, second])))


def test_multiple_distinct_ex_dates_are_preserved(tmp_path):
    first = _event()
    second = _event(event_id="second", record_date="20230503", ex_date="20230504",
                    pay_date="20230505", listing_date="20230508")
    ledger = load_ledger(_write(tmp_path, _document([second, first])))
    assert [event.event_id for event in ledger.events] == ["second", first["event_id"]]
    ledger.require_coverage("20230101", "20231231")


def test_cross_boundary_event_and_pending_payment_are_not_discarded(tmp_path):
    event = _event(record_date="20221229", ex_date="20221230", pay_date="20230105",
                   listing_date="20230106")
    ledger = load_ledger(_write(tmp_path, _document([event])))
    ledger.require_coverage("20230101", "20230106")
    assert ledger.events[0].record_date == "20221229"
    assert ledger.events[0].pay_date == "20230105"
    assert ledger.events[0].listing_date == "20230106"


def _unsupported(**overrides):
    event = {
        "event_id": "unsupported-rights-issue",
        "kind": "rights_issue",
        "status": "implemented",
        "record_date": "20230425",
        "ex_date": "20230426",
        "sources": [_SOURCE],
    }
    event.update(overrides)
    return event


def test_unsupported_event_can_be_archived_but_blocks_related_interval(tmp_path):
    ledger = load_ledger(_write(tmp_path, _document([_unsupported()])))
    assert ledger.events[0].kind == "rights_issue"
    assert ledger.events[0].cash_per_share_gross == Decimal(0)
    with pytest.raises(CorporateActionError, match="不支持的公司行为"):
        ledger.require_coverage("20230425", "20230426")
    ledger.require_coverage("20230101", "20230424")
    ledger.require_coverage("20230427", "20231231")


def test_unsupported_event_lifecycle_blocks_even_when_ex_precedes_start(tmp_path):
    event = _unsupported(record_date="20221229", ex_date="20221230", listing_date="20230106")
    ledger = load_ledger(_write(tmp_path, _document([event])))
    with pytest.raises(CorporateActionError, match="不支持的公司行为"):
        ledger.require_coverage("20230103", "20230105")
    ledger.require_coverage("20230107", "20231231")


@pytest.mark.parametrize("missing", ["record_date", "ex_date", "both"])
def test_unlocatable_unsupported_event_blocks_all_intervals(tmp_path, missing):
    event = _unsupported()
    if missing == "both":
        event["record_date"] = event["ex_date"] = None
    else:
        event[missing] = None
    ledger = load_ledger(_write(tmp_path, _document([event])))
    with pytest.raises(CorporateActionError, match="不支持的公司行为"):
        ledger.require_coverage("20230101", "20230102")


def test_coverage_is_copied_and_revalidated_if_caller_mutates_it():
    document = _document()
    ledger = CorporateActionLedger(
        symbol="300750", coverage=document["coverage"], events=(CorporateAction(**_event()),),
    )
    document["coverage"]["sources"].clear()
    document["coverage"]["status"] = "unverified"
    ledger.require_coverage("20230425", "20230426")
    ledger.coverage["status"] = "unverified"
    with pytest.raises(CorporateActionError, match="verified"):
        ledger.require_coverage("20230425", "20230426")


def test_missing_reference_method_requires_explicit_review_but_constructor_keeps_default(tmp_path):
    event = _event()
    del event["reference_method"]
    assert CorporateAction(**event).reference_method == "standard_cash_bonus"
    with pytest.raises(CorporateActionError, match="缺少字段.*reference_method"):
        load_ledger(_write(tmp_path, _document([event])))


def test_minimal_event_uses_documented_defaults_and_missing_required_fields_fail(tmp_path):
    event = _event()
    del event["fractional_policy"]
    ledger = load_ledger(_write(tmp_path, _document([event])))
    assert ledger.events[0].fractional_policy == "reject"
    assert ledger.events[0].reference_method == "standard_cash_bonus"
    assert ledger.events[0].bonus_ratio == Decimal(0)
    for field in (
        "event_id", "kind", "status", "sources", "cash_per_share_gross", "bonus_ratio",
        "transfer_ratio", "reference_method",
    ):
        invalid = copy.deepcopy(event)
        del invalid[field]
        with pytest.raises(CorporateActionError, match="缺少字段"):
            load_ledger(_write(tmp_path, _document([invalid])))

"""显式审阅的公司行为账本：只读载入、来源追溯和区间准入校验。

``verified`` 是账本提供者的审阅声明，不是本模块对接口完整性的证明。
本模块不联网，不从复权价格比例推断事件。

上市日口径：送转股份的 ``listing_date`` 只能来自两个可辩护来源，且必须由
``listing_date_source`` 显式声明：

* ``sina_red_share_listing``：新浪红股上市日，权威上市日。
* ``arrival_equals_ex_date``：巨潮股份到账日恰好等于除权日——股份在除权日
  到账即上市，不存在锁定期。契约强制校验 ``listing_date == ex_date``，
  因此到账日晚于除权日的疑似锁定期事件无法借此口径混入。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "CorporateAction",
    "CorporateActionError",
    "CorporateActionLedger",
    "load_ledger",
]

_SCHEMA = "corporate-actions-v1"
_SCOPE = "all_shareholder_distributions_and_capital_changes"
_COVERAGE_FIELDS = {"start", "end", "status", "scope", "sources", "verified_at", "notes"}
_EVENT_FIELDS = {
    "event_id", "kind", "status", "record_date", "ex_date", "pay_date", "listing_date",
    "cash_per_share_gross", "bonus_ratio", "transfer_ratio", "reference_method", "sources",
    "fractional_policy", "listing_date_source",
}
_LISTING_SOURCES = {"sina_red_share_listing", "arrival_equals_ex_date"}


class CorporateActionError(ValueError):
    """公司行为账本缺证、格式非法或不适用于请求区间。"""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorporateActionError(f"{label} 必须为非空字符串")
    return value


def _date(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}", value):
        raise CorporateActionError(f"{label} 必须为 YYYYMMDD 日期")
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise CorporateActionError(f"{label} 日期非法: {value!r}") from exc
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise CorporateActionError(f"{label} 必须为非负有限数值")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CorporateActionError(f"{label} 必须为非负有限数值") from exc
    if not result.is_finite() or result < 0:
        raise CorporateActionError(f"{label} 必须为非负有限数值")
    return result


def _sources(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise CorporateActionError(f"{label} 必须为非空 http(s) URL 列表")
    for item in value:
        if not isinstance(item, str) or any(char.isspace() for char in item):
            raise CorporateActionError(f"{label} 含非法来源 URL")
        try:
            parts = urlsplit(item)
            valid = parts.scheme in {"http", "https"} and bool(parts.hostname)
        except ValueError:
            valid = False
        if not valid:
            raise CorporateActionError(f"{label} 含非法来源 URL: {item!r}")
    return tuple(value)


def _keys(value: Any, allowed: set[str], required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise CorporateActionError(f"{label} 必须为对象")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise CorporateActionError(f"{label} 含未知字段: {sorted(unknown)}")
    if missing:
        raise CorporateActionError(f"{label} 缺少字段: {sorted(missing)}")


def _coverage(value: Any) -> dict[str, Any]:
    _keys(value, _COVERAGE_FIELDS, _COVERAGE_FIELDS, "coverage")
    start = _date(value["start"], "coverage.start")
    end = _date(value["end"], "coverage.end")
    if start > end:
        raise CorporateActionError("coverage.start 不得晚于 coverage.end")
    if value["status"] != "verified":
        raise CorporateActionError("coverage.status 必须为显式审阅的 verified")
    if value["scope"] != _SCOPE:
        raise CorporateActionError(f"coverage.scope 必须为 {_SCOPE}")
    _text(value["verified_at"], "coverage.verified_at")
    _text(value["notes"], "coverage.notes")
    return {**value, "sources": list(_sources(value["sources"], "coverage.sources"))}


@dataclass(frozen=True)
class CorporateAction:
    """每一原股的分配事件；未知 kind 仅用于阻断，不能参与结算。"""

    event_id: str
    kind: str
    status: str
    record_date: str | None = None
    ex_date: str | None = None
    pay_date: str | None = None
    listing_date: str | None = None
    cash_per_share_gross: Decimal | str = Decimal(0)
    bonus_ratio: Decimal | str = Decimal(0)
    transfer_ratio: Decimal | str = Decimal(0)
    reference_method: str = "standard_cash_bonus"
    sources: tuple[str, ...] = ()
    fractional_policy: str = "reject"
    listing_date_source: str | None = None

    def __post_init__(self) -> None:
        _text(self.event_id, "event.event_id")
        _text(self.kind, f"{self.event_id}.kind")
        if self.status != "implemented":
            raise CorporateActionError(f"{self.event_id}.status 必须为 implemented")
        for name in ("record_date", "ex_date", "pay_date", "listing_date"):
            _date(getattr(self, name), f"{self.event_id}.{name}", optional=True)
        for name in ("cash_per_share_gross", "bonus_ratio", "transfer_ratio"):
            object.__setattr__(self, name, _decimal(getattr(self, name), f"{self.event_id}.{name}"))
        object.__setattr__(self, "sources", _sources(self.sources, f"{self.event_id}.sources"))
        _text(self.reference_method, f"{self.event_id}.reference_method")
        if self.fractional_policy != "reject":
            raise CorporateActionError(f"{self.event_id}.fractional_policy 只支持 reject")
        if self.record_date and self.ex_date and self.record_date >= self.ex_date:
            raise CorporateActionError(f"{self.event_id} 必须满足 record_date < ex_date")
        for name in ("pay_date", "listing_date"):
            value = getattr(self, name)
            if value and self.ex_date and value < self.ex_date:
                raise CorporateActionError(f"{self.event_id}.{name} 不得早于 ex_date")
        if self.kind == "distribution":
            self._require_distribution()

    def _require_distribution(self) -> None:
        if self.reference_method != "standard_cash_bonus":
            raise CorporateActionError(f"{self.event_id} 不支持此除权参考价方法")
        if self.record_date is None or self.ex_date is None:
            raise CorporateActionError(f"{self.event_id} 缺少 record_date 或 ex_date")
        if not any(value > 0 for value in (
            self.cash_per_share_gross, self.bonus_ratio, self.transfer_ratio,
        )):
            raise CorporateActionError(f"{self.event_id} 总分配不得为 0；无事件不能伪装成分配事件")
        if self.cash_per_share_gross > 0 and self.pay_date is None:
            raise CorporateActionError(f"{self.event_id} 现金分红缺少 pay_date")
        if (self.bonus_ratio > 0 or self.transfer_ratio > 0) and self.listing_date is None:
            raise CorporateActionError(f"{self.event_id} 送转股份缺少 listing_date，到账日不能替代")
        if self.listing_date is not None:
            if self.listing_date_source not in _LISTING_SOURCES:
                raise CorporateActionError(
                    f"{self.event_id}.listing_date_source 必须为 {sorted(_LISTING_SOURCES)} 之一，"
                    f"实际 {self.listing_date_source!r}"
                )
            if (self.listing_date_source == "arrival_equals_ex_date"
                    and self.listing_date != self.ex_date):
                raise CorporateActionError(
                    f"{self.event_id} listing_date_source=arrival_equals_ex_date 时 "
                    f"listing_date 必须等于 ex_date({self.ex_date})，实际 {self.listing_date}；"
                    f"到账日晚于除权日的疑似锁定期事件不得按到账日上市"
                )

    def _overlaps(self, start: str, end: str) -> bool:
        # 缺少资格或除权日期的未知行为无法安全定位，不能默认为与区间无关。
        if self.record_date is None or self.ex_date is None:
            return True
        last_date = max(value for value in (
            self.record_date, self.ex_date, self.pay_date, self.listing_date,
        ) if value is not None)
        return self.record_date <= end and last_date >= start

    def to_dict(self) -> dict[str, Any]:
        """输出可直接 JSON 序列化的事件，保留十进制精度。"""
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "status": self.status,
            "record_date": self.record_date,
            "ex_date": self.ex_date,
            "pay_date": self.pay_date,
            "listing_date": self.listing_date,
            "cash_per_share_gross": str(self.cash_per_share_gross),
            "bonus_ratio": str(self.bonus_ratio),
            "transfer_ratio": str(self.transfer_ratio),
            "reference_method": self.reference_method,
            "sources": list(self.sources),
            "fractional_policy": self.fractional_policy,
            "listing_date_source": self.listing_date_source,
        }


@dataclass(frozen=True)
class CorporateActionLedger:
    """带显式区间审阅声明的只读账本；文件元数据由 load_ledger 派生。"""

    symbol: str
    coverage: dict[str, Any]
    events: tuple[CorporateAction, ...] = field(default_factory=tuple)
    schema: str = _SCHEMA
    source_path: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise CorporateActionError(f"不支持的公司行为 schema: {self.schema!r}")
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[0-9]{6}", self.symbol):
            raise CorporateActionError("symbol 必须为 6 位股票代码")
        object.__setattr__(self, "coverage", _coverage(self.coverage))
        if not isinstance(self.events, (tuple, list)):
            raise CorporateActionError("events 必须为事件序列")
        event_ids: set[str] = set()
        ex_dates: set[str] = set()
        for event in self.events:
            if not isinstance(event, CorporateAction):
                raise CorporateActionError("events 必须由 CorporateAction 组成")
            if event.event_id in event_ids:
                raise CorporateActionError(f"重复 event_id: {event.event_id}")
            if event.ex_date is not None and event.ex_date in ex_dates:
                raise CorporateActionError(
                    f"同 ex_date 事件冲突: {event.ex_date}；请先审阅并合并同日分配"
                )
            event_ids.add(event.event_id)
            if event.ex_date is not None:
                ex_dates.add(event.ex_date)
        object.__setattr__(self, "events", tuple(self.events))

    def require_coverage(self, start: str, end: str) -> None:
        """拒绝未覆盖的区间及生命周期与区间相交的未支持行为。"""
        start = _date(start, "请求 start")
        end = _date(end, "请求 end")
        if start > end:
            raise CorporateActionError("请求 start 不得晚于 end")
        coverage = _coverage(self.coverage)
        if coverage["start"] > start or coverage["end"] < end:
            raise CorporateActionError(
                f"{self.symbol} 公司行为覆盖不足: 请求 {start}~{end}，"
                f"已审阅 {coverage['start']}~{coverage['end']}"
            )
        for event in self.events:
            if event._overlaps(start, end):
                if event.kind != "distribution":
                    raise CorporateActionError(
                        f"{self.symbol} 区间含不支持的公司行为: {event.event_id} ({event.kind})"
                    )
                event._require_distribution()

    def to_dict(self) -> dict[str, Any]:
        """返回可再次载入的账本，不将派生路径或哈希混入原始契约。"""
        return {
            "schema": self.schema,
            "symbol": self.symbol,
            "coverage": _coverage(self.coverage),
            "events": [event.to_dict() for event in self.events],
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorporateActionError(f"JSON 含重复字段: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CorporateActionError(f"JSON 含非有限数值: {value}")


def load_ledger(path: str | Path, symbol: str | None = None) -> CorporateActionLedger:
    """只读本地 JSON；严格校验字段并对原始文件字节计算 SHA-256。"""
    try:
        source_path = Path(path).expanduser().resolve()
        raw = source_path.read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise CorporateActionError(f"无法读取公司行为账本: {path}") from exc
    try:
        data = json.loads(
            raw.decode("utf-8-sig"),
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorporateActionError(f"公司行为账本不是合法 UTF-8 JSON: {source_path}") from exc
    fields = {"schema", "symbol", "coverage", "events"}
    _keys(data, fields, fields, "ledger")
    if symbol is not None and data["symbol"] != symbol:
        raise CorporateActionError(f"账本股票代码不匹配: 期望 {symbol!r}，实际 {data['symbol']!r}")
    if not isinstance(data["events"], list):
        raise CorporateActionError("ledger.events 必须为数组")
    events = []
    for index, event in enumerate(data["events"]):
        required = {"event_id", "kind", "status", "sources"}
        if isinstance(event, dict) and event.get("kind") == "distribution":
            required |= {
                "cash_per_share_gross", "bonus_ratio", "transfer_ratio", "reference_method",
            }
        _keys(event, _EVENT_FIELDS, required, f"events[{index}]")
        events.append(CorporateAction(**event))
    return CorporateActionLedger(
        schema=data["schema"],
        symbol=data["symbol"],
        coverage=data["coverage"],
        events=tuple(events),
        source_path=str(source_path),
        sha256=hashlib.sha256(raw).hexdigest(),
    )

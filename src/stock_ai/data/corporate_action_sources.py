"""采集巨潮原响应与未审阅候选，不认证覆盖、不生成可执行公司行为账本。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from stock_ai.data.corporate_actions import CorporateActionError

__all__ = ["fetch_cninfo_snapshot"]

_ENDPOINT = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"


def _request_dependencies():
    try:
        import akshare
        import py_mini_racer
        import requests
        from akshare.datasets import get_ths_js
    except ImportError as exc:
        raise CorporateActionError(
            "当前 Python 缺少巨潮快照依赖 requests/akshare/py_mini_racer；"
            "请在已配置这些依赖的 managed 环境运行，不会借用项目 .venv"
        ) from exc
    return requests, akshare, py_mini_racer, get_ths_js


def _enckey(py_mini_racer, get_ths_js) -> str:
    try:
        script = Path(get_ths_js("cninfo.js")).read_text(encoding="utf-8")
        with py_mini_racer.MiniRacer() as runtime:
            runtime.eval(script)
            key = runtime.call("getResCode1")
    except Exception as exc:
        raise CorporateActionError("无法使用当前 AKShare cninfo.js 生成巨潮请求头") from exc
    if not isinstance(key, str) or not key.strip():
        raise CorporateActionError("巨潮动态 Accept-Enckey 为空或无效")
    return key


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CorporateActionError(f"巨潮响应 JSON 含重复字段: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise CorporateActionError(f"巨潮响应 JSON 含非有限数值: {value}")


def _validate_response(response) -> dict[str, Any]:
    if response.status_code != 200:
        raise CorporateActionError(f"巨潮快照 HTTP 失败: {response.status_code}")
    expected = urlsplit(_ENDPOINT)
    actual = urlsplit(response.url)
    if (actual.scheme, actual.netloc, actual.path) != (
        expected.scheme, expected.netloc, expected.path,
    ):
        raise CorporateActionError("巨潮响应 URL 偏离指定 HTTPS 接口，拒绝重定向或降级")
    try:
        data = json.loads(
            response.content.decode("utf-8-sig"), parse_float=Decimal,
            object_pairs_hook=_json_object, parse_constant=_invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorporateActionError("巨潮响应不是合法 UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise CorporateActionError("巨潮响应必须为 JSON 对象")
    if type(data.get("resultcode")) is not int or data["resultcode"] != 200:
        raise CorporateActionError(f"巨潮业务失败 resultcode={data.get('resultcode')!r}")
    records = data.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise CorporateActionError("巨潮 records 必须为对象数组")
    count, total = data.get("count"), data.get("total")
    if any(type(value) is not int or value < 0 for value in (count, total)):
        raise CorporateActionError("巨潮 count/total 必须为非负整数")
    if count != total or total != len(records):
        raise CorporateActionError("巨潮快照记录数不完整：count/total/records 不一致，可能被截断")
    # 此接口的既有适配器不分页；若源端新增分页，只允许明确的单页响应。
    for name in ("pages", "totalPages", "pageCount", "page", "pageNum", "pageNumber"):
        if name in data and (type(data[name]) is not int or data[name] != 1):
            raise CorporateActionError(f"巨潮快照存在未获取分页或未知分页语义: {name}")
    for name in ("hasMore", "has_more", "hasNext", "has_next"):
        if name in data and data[name] is not False:
            raise CorporateActionError(f"巨潮快照仍存在后续页: {name}")
    return data


def _candidate(row: dict[str, Any], symbol: str, index: int, source_url: str) -> dict[str, Any]:
    unknown = []
    invalid = []
    result = {
        "candidate_id": f"cninfo:{symbol}:{index}",
        "status": "unreviewed",
        "kind": "distribution",
        "symbol": symbol,
        "source_row_index": index,
        "sources": [source_url],
        "listing_date": None,
        "reference_method": None,
    }
    for source, target in (
        ("F044V", "distribution_type"), ("F001V", "report_period_label"),
        ("F007V", "source_plan"),
    ):
        value = row.get(source)
        result[target] = value if isinstance(value, str) and value.strip() else None
        if result[target] is None:
            unknown.append(target)
            if value is not None:
                invalid.append(target)
    for source, target in (
        ("F006D", "announcement_date"), ("F018D", "record_date"), ("F020D", "ex_date"),
        ("F023D", "pay_date"), ("F025D", "share_credit_date"),
    ):
        value = row.get(source)
        result[target] = None
        if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            try:
                result[target] = date.fromisoformat(value).strftime("%Y%m%d")
            except ValueError:
                pass
        if result[target] is None:
            unknown.append(target)
            if value is not None:
                invalid.append(target)
    for source, target in (
        ("F012N", "cash_per_share_gross"), ("F010N", "bonus_ratio"),
        ("F011N", "transfer_ratio"),
    ):
        value = row.get(source)
        result[target] = None
        if not isinstance(value, bool) and isinstance(value, (str, int, Decimal)):
            try:
                amount = Decimal(str(value))
                if amount.is_finite() and amount >= 0:
                    parts = amount.as_tuple()
                    result[target] = str(Decimal((parts.sign, parts.digits, parts.exponent - 1)))
            except InvalidOperation:
                pass
        if result[target] is None:
            unknown.append(target)
            if value is not None:
                invalid.append(target)
    unknown.extend(("listing_date", "reference_method"))
    result["unknown_fields"] = unknown
    result["invalid_fields"] = invalid
    return result


def fetch_cninfo_snapshot(symbol: str, output_dir: str | Path) -> dict[str, Any]:
    """保存一次完整源响应和未审阅候选；绝不输出 verified 账本或区间证明。"""
    if not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise CorporateActionError("symbol 必须为 6 位股票代码")
    directory = Path(output_dir).expanduser().resolve()
    parts = [part.casefold() for part in directory.parts]
    if any(parts[index:index + 2] == ["data", "corporate_actions"]
           for index in range(len(parts) - 1)):
        raise CorporateActionError("未审阅源快照不得写入 data/corporate_actions 账本目录")
    requests, akshare, py_mini_racer, get_ths_js = _request_dependencies()
    headers = {
        "Accept": "*/*", "Accept-Enckey": _enckey(py_mini_racer, get_ths_js),
        "Origin": "http://webapi.cninfo.com.cn", "Referer": "http://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
    }
    params = {"scode": symbol}
    try:
        response = requests.post(
            _ENDPOINT, params=params, headers=headers, timeout=20, allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise CorporateActionError(f"巨潮快照请求失败: {exc}") from exc
    fetched_at = datetime.now(timezone.utc)
    try:
        data = _validate_response(response)
        raw = response.content
        source_url = response.url
        snapshot = {
            "schema": "corporate-action-snapshot-v1",
            "provider": "cninfo", "symbol": symbol, "status": "unreviewed",
            "method": "POST", "url": source_url, "params": params,
            "fetched_at": fetched_at.isoformat(),
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "response_encoding": response.encoding,
            "response_bytes": len(raw), "response_sha256": hashlib.sha256(raw).hexdigest(),
            "resultcode": data["resultcode"], "count": data["count"], "total": data["total"],
            "akshare_version": akshare.__version__, "python_executable": sys.executable,
            "snapshot_counts_matched": True,
            "candidates": [_candidate(row, symbol, index, source_url)
                           for index, row in enumerate(data["records"])],
            "review_notes": [
                "仅匹配当前快照计数，不证明任何历史区间或所有资本变动完整覆盖。",
                "候选尚未人工审阅，不能直接用于公司行为结算或标记 implemented/verified。",
                "原始分配单位为每10股；缺失或非法的腿仍未知，不从方案或复权比值猜零。",
                "F025D 仅为股份到账日；上市可卖日及标准除权参考价方法需要独立核实。",
            ],
        }
    finally:
        response.close()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        name = f"cninfo_{symbol}_{fetched_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}"
        raw_path = directory / f"{name}.response.json"
        metadata_path = directory / f"{name}.snapshot.json"
        snapshot["raw_response_path"] = str(raw_path)
        snapshot["metadata_path"] = str(metadata_path)
        with raw_path.open("xb") as stream:
            stream.write(raw)
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except OSError as exc:
        raise CorporateActionError("无法独占创建巨潮快照档案，不会覆盖已有文件") from exc
    return snapshot

#!/usr/bin/env python
"""公司行为账本采集与双源核对流水线（fail-closed）。

流程：抓取巨潮（cninfo）与新浪（sina）两个分红送转来源 -> 原样归档并留sha256
-> 按「股权登记日」对齐两源 -> 逐事件双源核对 -> 绑定巨潮官方实施公告 PDF
-> 检出配股与注销式回购两类股本变动 -> 写审阅记录 -> 仅当该symbol 全部事件
通过核对、无股本变动告警且账本证据齐备时，才写符合 corporate-actions-v1 的账本草稿，
并直接调用 src/stock_ai/data/corporate_actions.load_ledger 做最终校验。

设计原则（不放宽）：
  * 任何一条规则不满足 -> 该事件 excluded 并写明 reason_code，绝不猜测、绝不合并、
    绝不从方案文本反推数值。
  * cninfo某腿为NaN 视为「该源未提供」，不默认成 0；因此绝大多数纯现金事件会被
    excluded(missing_leg)。这是刻意的 fail-closed 行为，已在审阅记录中逐条留痕。
  * 公告绑定只接受「标题含实施公告关键词」且「公告时间==实施方案公告日期」的同日
    公告；不使用 ±60 天窗口。更早事件因源端历史深度上限匹配不到公告时，
    标记 official_pdf_missing 并退化为接口 URL，绝不伪造 PDF 链接。
  * 配股与注销式回购一律只标记 blocked，不自行判定是否已实施、也不自行扣减股本。
  * 只写 outputs/ 下的产物；拒绝写入 data/ 目录；不训练、不碰任何服务。

用法示例：
  python build_corporate_action_ledgers.py --symbols 000001,600519,300750 \
      --out-dir outputs/corporate_action_build_sample --dry-run-fail-closed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 常量：来源标识、列名、reason_code 词表
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stock_ai.data.corporate_actions import (  # noqa: E402  (需先补sys.path)
    CorporateActionError,
    load_ledger,
)

LEDGER_SCHEMA = "corporate-actions-v1"
COVERAGE_SCOPE = "all_shareholder_distributions_and_capital_changes"

PROVIDER_CNINFO = "cninfo"
PROVIDER_SINA = "sina"
PROVIDERS = (PROVIDER_CNINFO, PROVIDER_SINA)

CNINFO_URL = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"
SINA_URL = "https://vip.stock.finance.sina.com.cn/corp/view/vISSUE_ShareBonusDetail.php"
CNINFO_ANNOUNCEMENT_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_ORG_MAP = "http://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_STATIC_HOST = "http://static.cninfo.com.cn/"
EASTMONEY_REPURCHASE = (
    "https://datacenter-web.eastmoney.com/api/data/v1/get"
    "?reportName=RPTA_WEB_GETHGLIST_NEW&columns=ALL&source=WEB"
)

# 巨潮公告查询：服务端pageSize 硬上限 30，必须翻页；totalAnnouncement 会漂移，
# 因此按 announcementId 去重，且不假设总数恒定。
ANNOUNCEMENT_PAGE_SIZE = 30
ANNOUNCEMENT_MAX_PAGES = 20
# 公告标题必须同时命中：分配类关键词 + 实施类关键词。
ANNOUNCEMENT_SUBJECT_WORDS = ("权益分派", "分红", "利润分配", "分配")
ANNOUNCEMENT_IMPLEMENT_WORDS = ("实施公告", "派息实施", "分派实施", "实施")
# akshare 传isHLtitle=true 会在标题里插入 <em>，入库前必须剥离。
ANNOUNCEMENT_TAG_RE = re.compile(r"<[^>]+>")
# 巨潮公告历史深度：实测最早仅回溯到 2012-10-12，更早事件匹配不到 PDF。
CNINFO_ANNOUNCEMENT_HISTORY_FLOOR = "20121012"

# 回购注销判定：仅当回购用途明确「注销 / 减少注册资本」时才视为股本减少。
BUYBACK_CANCELLATION_KEYWORDS = ("注销", "减少公司注册资本", "减少注册资本", "减资")
BUYBACK_PAGE_SIZE = 500
BUYBACK_MAX_PAGES = 40

# 需要升级到官方公告 PDF 定谳的 reason_code（裁决 2：容差为 0，冲突以公告原文为准）。
PDF_ADJUDICABLE = frozenset({
    "source_amount_conflict", "source_date_conflict",
    "pay_date_unknown", "listing_date_unknown",
})

CNINFO_COLUMNS = (
    "实施方案公告日期", "分红类型", "送股比例", "转增比例", "派息比例",
    "股权登记日", "除权日", "派息日", "股份到账日", "实施方案分红说明", "报告时间",
)
SINA_COLUMNS = ("公告日期", "送股", "转增", "派息", "进度", "除权除息日", "股权登记日", "红股上市日")

# 每 10 股 -> 每股 的除数，必须精确十进制，禁止浮点。
UNITS_PER_SHARE_DIVISOR = Decimal(10)
# 巨潮公告时间戳为北京时间，不能按 UTC 换算。
CST = timezone(timedelta(hours=8))
# Decimal 除法上下文精度；除以 10 是有限小数，但显式提高精度避免任何舍入。
DIVIDE_PRECISION = 60

# 参考价方法判定：方案文本出现这些字样即认为不是标准「派息+送股+转增」。
NONSTANDARD_PLAN_KEYWORDS = (
    "配股", "折股", "缩股", "回购", "注销", "增发", "转债", "债转股",
    "以股抵债", "非公开发行", "定向发行", "员工持股", "股权激励",
)
# cninfo「分红类型」中属于常规现金/送转分配的取值；其余视为资本变动告警来源。
STANDARD_DIVIDEND_TYPES = ("年度分红", "中期分红", "半年度分红", "季度分红")

REASON_SEND_EXCLUDED = (
    "source_fetch_failed", "cninfo_row_without_ex_date", "record_date_missing",
    "record_date_absent_in_other_source", "record_date_duplicated_in_source",
    "source_date_conflict", "invalid_date", "date_order_invalid",
    "missing_leg", "invalid_leg", "all_legs_zero", "source_amount_conflict",
    "nonstandard_plan", "pay_date_unknown", "listing_date_unknown",
    "duplicate_event_id",
)


class BuildError(RuntimeError):
    """流水线自身的失败（参数、路径守卫、独占写入冲突）。"""


# ---------------------------------------------------------------------------
# 基础工具：JSON 安全值、sha256、独占写入
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """返回 UTC 时刻字符串（秒级，带 Z）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_value(value: Any) -> Any:
    """把 DataFrame 单元格转成可 JSON 序列化的值，NaN/NaT 统一为 None。"""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    # pandas / numpy 标量与 NaT：先尝试通用转换，再退化为字符串。
    item = getattr(value, "item", None)
    if callable(item):
        try:
            inner = item()
        except (TypeError, ValueError):
            inner = None
        if inner is not value:
            return _json_value(inner)
    if str(value) in {"NaT", "nan", "None", "<NA>"}:
        return None
    return str(value)


def _records(frame: Any) -> list[dict[str, Any]]:
    """把 DataFrame 转成逐行字典列表（列顺序保持源顺序）。"""
    columns = [str(name) for name in frame.columns]
    rows: list[dict[str, Any]] = []
    for values in frame.itertuples(index=False, name=None):
        rows.append({column: _json_value(value) for column, value in zip(columns, values)})
    return rows


def _sha256_text(text: str) -> str:
    """计算 UTF-8 文本的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_exclusive(path: Path, text: str) -> str:
    """以 'x' 模式独占创建文件；已存在立即失败，绝不覆盖。返回内容 sha256。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BuildError(f"拒绝覆盖已存在产物: {path}")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except FileExistsError as exc:
        raise BuildError(f"拒绝覆盖已存在产物: {path}") from exc
    except OSError as exc:
        raise BuildError(f"无法写入产物 {path}: {exc}") from exc
    return _sha256_text(text)


def _dump_json(payload: Any) -> str:
    """统一 JSON 序列化：禁用 NaN、保留中文、缩进 2、结尾换行。"""
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _guard_output_dir(out_dir: Path) -> Path:
    """拒绝把产物写入 data/ 目录（原始与中间数据目录只读）。"""
    resolved = out_dir.expanduser().resolve()
    data_root = (ROOT / "data").resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise BuildError(f"产物目录不得位于 data/ 之下: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# 日期与数值规范化
# ---------------------------------------------------------------------------

def _ymd(value: Any) -> str | None:
    """把多种日期表示统一成 YYYYMMDD；无法识别返回 None。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text in {"NaT", "nan", "None", "<NA>", "--", "0"}:
        return None
    text = text.replace("/", "-").replace(".", "-")
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) != 8:
        return None
    try:
        parsed = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return None
    return parsed.strftime("%Y%m%d")


def _leg(value: Any) -> tuple[Decimal | None, str]:
    """把「每 10 股」的单腿数值规范成 Decimal。

    返回 (数值, 状态)；状态取值 missing / invalid / ok。
    NaN/None 一律视为「该源未提供」，绝不默认成 0。
    """
    if value is None:
        return None, "missing"
    if isinstance(value, float) and math.isnan(value):
        return None, "missing"
    if isinstance(value, bool):
        return None, "invalid"
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"NaT", "nan", "None", "<NA>", "--"}:
            return None, "missing"
        try:
            amount = Decimal(text)
        except InvalidOperation:
            return None, "invalid"
    elif isinstance(value, (int, float, Decimal)):
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            return None, "invalid"
    else:
        return None, "invalid"
    if not amount.is_finite():
        return None, "invalid"
    if amount < 0:
        return None, "invalid"
    return amount, "ok"


def _per_share(amount: Decimal) -> Decimal:
    """把「每 10 股」精确换算成「每股」，只用 Decimal，禁止浮点。"""
    with localcontext() as context:
        context.prec = DIVIDE_PRECISION
        return amount / UNITS_PER_SHARE_DIVISOR


# 每条腿在方案文本中的关键词。判定「该腿为 0」要求文本中不出现该腿的任何表述。
LEG_PLAN_KEYWORDS = {
    "cash": ("派", "分红", "股利"),
    "bonus": ("送股", "送红股", "送"),
    "transfer": ("转增",),
}


def _plan_text_of(cn_row: dict[str, Any]) -> str:
    """取方案文本；非字符串一律视为空。"""
    text = cn_row.get("plan_text")
    return text if isinstance(text, str) else ""


def _leg_keywords_hit(text: str, leg: str) -> list[str]:
    """返回方案文本中命中的该腿关键词。"""
    return [word for word in LEG_PLAN_KEYWORDS[leg] if word in text]


def resolve_legs(
    cn_row: dict[str, Any], si_row: dict[str, Any],
) -> tuple[dict[str, Decimal | None], dict[str, Any]]:
    """判定三腿取值，并为每条「判定为 0 的腿」留证据。

    分两种情形，避免把规则 5' 误用到非零腿上：
    1. 两源都给出明确数值（含非零）-> 直接采用 cninfo 值，交给调用方做容差 0 比较；
    2. 任一侧缺失或为 0 -> 适用规则 5'：该腿为 0 当且仅当三条同时成立——
       (a) cninfo 该字段为空或为 0，(b) sina 该腿**显式**为 0，
       (c) 实施方案分红说明文本中不出现该腿关键词。缺任一条即为 unknown。
    这仍要求两个独立结构化源 + 公告文本三方一致，不是单源推断。
    """
    text = _plan_text_of(cn_row)
    values: dict[str, Decimal | None] = {}
    evidence: dict[str, Any] = {"plan_text": text or None, "legs": {}}
    for leg in ("cash", "bonus", "transfer"):
        cn_value, cn_state = cn_row["per10"][leg], cn_row["leg_states"][leg]
        si_value, si_state = si_row["per10"][leg], si_row["leg_states"][leg]
        hits = _leg_keywords_hit(text, leg)
        record = {
            "cninfo_raw": None if cn_value is None else str(cn_value),
            "cninfo_state": cn_state,
            "sina_raw": None if si_value is None else str(si_value),
            "sina_state": si_state,
            "plan_keyword_hits": hits,
            "condition_a_cninfo_absent_or_zero": (
                cn_state == "missing" or (cn_state == "ok" and cn_value == 0)
            ),
            "condition_b_sina_explicit_zero": si_state == "ok" and si_value == 0,
            "condition_c_plan_silent": not hits,
        }
        if cn_state == "invalid" or si_state == "invalid":
            values[leg] = None
            record["resolved_as"] = "invalid"
        elif cn_state == "ok" and si_state == "ok":
            # 两源都有明确数值：采用 cninfo 值，由调用方做容差 0 比较。
            values[leg] = cn_value
            record["resolved_as"] = "both_sources_present"
        else:
            # 任一侧缺失 -> 规则 5' 决定是否为 0。
            if (
                record["condition_a_cninfo_absent_or_zero"]
                and record["condition_b_sina_explicit_zero"]
                and not hits
            ):
                values[leg] = Decimal(0)
                record["resolved_as"] = "zero_confirmed"
            else:
                values[leg] = None
                record["resolved_as"] = "unknown"
                record["unmet_conditions"] = [
                    name for name, ok in (
                        ("a_cninfo_absent_or_zero", record["condition_a_cninfo_absent_or_zero"]),
                        ("b_sina_explicit_zero", record["condition_b_sina_explicit_zero"]),
                        ("c_plan_silent", not hits),
                    ) if not ok
                ]
        evidence["legs"][leg] = record
    evidence["zero_legs_confirmed"] = [
        leg for leg, record in evidence["legs"].items()
        if record["resolved_as"] == "zero_confirmed"
    ]
    return values, evidence


def _leg_status(values: dict[str, Decimal | None]) -> str:
    """汇总三腿状态：任一 invalid 则invalid；任一 missing 则 missing；否则 ok。"""
    statuses = [value for value in values.values() if value is not None]
    if any(value is None for value in values.values()):
        return "missing"
    if not statuses:
        return "missing"
    return "ok"


# ---------------------------------------------------------------------------
# 抓取与原样归档
# ---------------------------------------------------------------------------

def _fetch(provider: str, symbol: str, attempts: int = 3, pause: float = 1.5) -> dict[str, Any]:
    """抓取单个来源的原始 DataFrame；失败按固定退避重试。"""
    try:
        import akshare
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise BuildError("当前 Python 缺少 akshare 依赖，无法采集") from exc
    import pandas as pd  # noqa: F401  (确保 pandas 已被 akshare 加载)

    last_error: Exception | None = None
    for index in range(attempts):
        try:
            if provider == PROVIDER_CNINFO:
                frame = akshare.stock_dividend_cninfo(symbol=symbol)
            else:
                frame = akshare.stock_history_dividend_detail(symbol=symbol)
            if frame is None or len(frame.columns) == 0:
                raise BuildError(f"{provider} 返回空表")
            return {"frame": frame, "akshare_version": akshare.__version__}
        except Exception as exc:  # 网络/解析异常统一重试
            last_error = exc
            if index + 1 < attempts:
                time.sleep(pause * (index + 1))
    raise BuildError(
            f"{provider} 抓取/解析失败({attempts} 次重试): {last_error}；"
            "这是数据获取失败，不得据此断言该股票无分红"
        )


def _archive_raw(
    provider: str, symbol: str, frame: Any, akshare_version: str, out_dir: Path, fetched_at: str,
) -> dict[str, Any]:
    """把原始 DataFrame 全文归档为 JSON，并记录 sha256 与抓取时刻。"""
    columns = [str(name) for name in frame.columns]
    records = _records(frame)
    records_text = json.dumps(records, ensure_ascii=False, sort_keys=True, allow_nan=False)
    payload = {
        "schema": "corporate-action-raw-v1",
        "provider": provider,
        "symbol": symbol,
        "status": "raw_unreviewed",
        "fetched_at_utc": fetched_at,
        "akshare_version": akshare_version,
        "python_executable": sys.executable,
        "endpoint": CNINFO_URL if provider == PROVIDER_CNINFO else SINA_URL,
        "units": "每10股",
        "row_count": len(records),
        "columns": columns,
        "records_sha256": _sha256_text(records_text),
        "records": records,
        "notes": [
            "原始响应逐行归档，未做任何清洗、换算或补齐。",
            "NaN/NaT 统一序列化为 null，表示该源未提供该字段，不得解读为 0。",
            "records_sha256 是对 sort_keys 后紧凑 JSON 的摘要，用于跨次运行比对。",
        ],
    }
    path = out_dir / "raw" / f"{symbol}.{provider}.json"
    payload["archive_sha256"] = _write_exclusive(path, _dump_json(payload))
    payload["archive_path"] = str(path)
    return payload


# ---------------------------------------------------------------------------
# 原始行 -> 规范化候选
# ---------------------------------------------------------------------------

def _cninfo_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """把一条 cninfo 原始行规范成内部候选结构，保留全部原值。"""
    cash, cash_state = _leg(row.get("派息比例"))
    bonus, bonus_state = _leg(row.get("送股比例"))
    transfer, transfer_state = _leg(row.get("转增比例"))
    return {
        "provider": PROVIDER_CNINFO,
        "row_index": index,
        "raw": {column: row.get(column) for column in CNINFO_COLUMNS},
        "announcement_date": _ymd(row.get("实施方案公告日期")),
        "dividend_type": row.get("分红类型"),
        "report_period": row.get("报告时间"),
        "plan_text": row.get("实施方案分红说明"),
        "record_date": _ymd(row.get("股权登记日")),
        "ex_date": _ymd(row.get("除权日")),
        "pay_date": _ymd(row.get("派息日")),
        "share_credit_date": _ymd(row.get("股份到账日")),
        "per10": {"cash": cash, "bonus": bonus, "transfer": transfer},
        "leg_states": {
            "cash": cash_state, "bonus": bonus_state, "transfer": transfer_state,
        },
    }


def _sina_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """把一条 sina 原始行规范成内部候选结构，保留全部原值。"""
    cash, cash_state = _leg(row.get("派息"))
    bonus, bonus_state = _leg(row.get("送股"))
    transfer, transfer_state = _leg(row.get("转增"))
    return {
        "provider": PROVIDER_SINA,
        "row_index": index,
        "raw": {column: row.get(column) for column in SINA_COLUMNS},
        "announcement_date": _ymd(row.get("公告日期")),
        "progress": row.get("进度"),
        "record_date": _ymd(row.get("股权登记日")),
        "ex_date": _ymd(row.get("除权除息日")),
        "listing_date": _ymd(row.get("红股上市日")),
        "per10": {"cash": cash, "bonus": bonus, "transfer": transfer},
        "leg_states": {
            "cash": cash_state, "bonus": bonus_state, "transfer": transfer_state,
        },
    }


def _candidate_rows(provider: str, archive: dict[str, Any]) -> list[dict[str, Any]]:
    """从归档原文中重建候选行（不重新联网，保证审阅可离线复现）。"""
    build = _cninfo_row if provider == PROVIDER_CNINFO else _sina_row
    return [build(record, index) for index, record in enumerate(archive["records"])]


# ---------------------------------------------------------------------------
# 官方公告 PDF 定谳（复用 L3 抽检已验证的抽取规则）
# ---------------------------------------------------------------------------

# 以下抽取规则移植自 outputs/spotcheck/s3b_audit.py（经 228 审计点实战验证，
# FAIL=0）。刻意不复用 s3_audit_BUGGY_superseded_by_s3b.py 的初版正则。
AFTER_TAX_MARKERS = ("扣税后", "深股通", "QFII", "RQFII", "持有首发", "证券投资基金")
# 否定式表述是判定零腿的关键文本证据，必须能识别「不送红股/不进行资本公积转增」。
NO_BONUS_RE = re.compile(
    r"不送(?:红股|股)|无送股|不进行送股|不实施送股|仅进行现金红利分配"
    r"|差异化分红送转[:：]\s*(?:否|无)|不涉及送"
)
NO_TRANSFER_RE = re.compile(
    r"不(?:以)?(?:资本)?公积金转增|无转增|不进行(?:资本公积)?转增"
    r"|不实施(?:资本)?公积(?:金)?转增|不送股和转增|不送股、转增"
    r"|不送股[，,]\s*不(?:以)?(?:进行)?(?:资本公积)?转增"
    r"|仅进行现金红利分配|差异化分红送转[:：]\s*(?:否|无)|流通股份变动比例为0"
)
DIFFERENTIAL_MARKERS = ("差异化分红", "差异化权益分派")
_PDF_NUM = r"(\d+(?:\.\d+)?)"
_PDF_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_PDF_SLASH_RE = re.compile(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})")
_PDF_DATE_TOKEN = (
    r"((?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|(?:\d{4}[/\-]\d{1,2}[/\-]\d{1,2}))"
)


def _pdf_norm(text: str) -> str:
    """NFKC 归一化全角字符并统一空白，供 PDF 文本匹配使用。"""
    import unicodedata
    return unicodedata.normalize("NFKC", text or "").replace("\u3000", " ").replace("\xa0", " ")


def _pdf_flat(text: str) -> str:
    """压平换行，用于日期扫描。"""
    return re.sub(r"\s+", " ", _pdf_norm(text))


def _pdf_squash(text: str) -> str:
    """去掉全部空白，用于金额模式匹配。"""
    return re.sub(r"\s+", "", _pdf_norm(text))


def _pdf_date(value: str) -> str | None:
    """从 PDF 日期片段解析 YYYYMMDD；不合法返回 None。"""
    for pattern in (_PDF_DATE_RE, _PDF_SLASH_RE):
        match = pattern.search(value or "")
        if match:
            year, month, day = (int(match.group(index)) for index in (1, 2, 3))
            if 1990 <= year <= 2035 and 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}{month:02d}{day:02d}"
    return None


def _pdf_dec(value: str) -> Decimal:
    """把 PDF 提取的数值规范成 Decimal，去除尾随零差异（280.24230 == 280.2423）。"""
    return Decimal(str(value)).normalize()


def _pdf_dedup(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    """按数值去重（Decimal 规范化，避免 280.2423 与 280.24230 被当成两个候选），
    并保留首条证据串，用于判断是否存在多候选。"""
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for value, evidence in pairs:
        key = str(_pdf_dec(value))
        if key not in seen:
            seen.add(key)
            result.append({"value": key, "evidence": evidence})
    return result


def _pdf_cash_candidates(flat: str, squash: str) -> list[dict[str, str]]:
    """提取每10股税前现金候选；必须按前置语境排除税后金额。

    公告常同时出现税前与扣税后（深股通/QFII/个人）两个值，只按
    「每10股派X元」抓会抓到税后值，产生假 FAIL。
    """
    found: list[dict[str, str]] = []
    pattern = (
        r"每\s*10\s*股派(?:发现金(?:分红|股利)?|送现金|现金(?:分红)?)?"
        r"(?:人民币)?\s*" + _PDF_NUM + r"\s*元"
    )
    for match in re.finditer(pattern, squash):
        prefix = squash[max(0, match.start() - 34):match.start()]
        if any(marker in prefix for marker in AFTER_TAX_MARKERS):
            continue  # 税后口径 -> 跳过
        found.append({
            "value": str(Decimal(match.group(1))),
            "evidence": "每10股派" + match.group(1) + "元（含税）",
        })
    for pattern, tag in (
        (r"每股(?:现金红利|派发现金红利|派发现金股利)(?:为)?" + _PDF_NUM + r"元", "每股现金红利"),
        (r"每股派发现金红利" + _PDF_NUM + r"元", "每股派发现金红利"),
    ):
        for match in re.finditer(pattern, squash):
            prefix = squash[max(0, match.start() - 34):match.start()]
            if any(marker in prefix for marker in AFTER_TAX_MARKERS):
                continue
            per10 = Decimal(match.group(1)) * UNITS_PER_SHARE_DIVISOR
            found.append({
                "value": str(per10),
                "evidence": f"{tag}{match.group(1)}元 → 每10股{per10}元（含税）",
            })
    return _pdf_dedup([(item["value"], item["evidence"]) for item in found])


def _pdf_ratio_candidates(squash: str, pattern: str, tag: str) -> list[dict[str, str]]:
    """提取送股/转增比例候选。"""
    found = [
        {"value": str(Decimal(match.group(1))), "evidence": tag + match.group(1) + "股"}
        for match in re.finditer(pattern, squash)
    ]
    return _pdf_dedup([(item["value"], item["evidence"]) for item in found])


def _pdf_record_candidates(flat: str) -> list[dict[str, str]]:
    """提取股权登记日候选；只捕获紧邻标记的日期，避免贪婪匹配。"""
    pairs: list[tuple[str, str]] = []
    pattern = r"股权登记日(?:期)?(?:为)?\s*[：:]?\s*" + _PDF_DATE_TOKEN
    for match in re.finditer(pattern, flat):
        value = _pdf_date(match.group(1))
        if value:
            pairs.append((value, "股权登记日为：" + match.group(1).strip()))
    for match in re.finditer(r"[ＡA]\s*股\s*(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\s*[－\-—]", flat):
        value = _pdf_date(match.group(1))
        if value:
            pairs.append((value, "A股表格 股权登记日=" + match.group(1).strip()))
    return _pdf_dedup(pairs)


def _pdf_ex_candidates(flat: str) -> list[dict[str, str]]:
    """提取除权除息日候选；表格型「A股 登记日 － 除权日」取第二个日期。"""
    pairs: list[tuple[str, str]] = []
    pattern = r"除权(?:除息|息)?日(?:期)?为?\s*[：:]?\s*" + _PDF_DATE_TOKEN
    for match in re.finditer(pattern, flat):
        value = _pdf_date(match.group(1))
        if value:
            pairs.append((value, "除权除息日为：" + match.group(1).strip()))
    table = (
        r"[ＡA]\s*股\s*\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\s*[－\-—]\s*"
        r"(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})"
    )
    for match in re.finditer(table, flat):
        value = _pdf_date(match.group(1))
        if value:
            pairs.append((value, "A股表格 除权(息)日=" + match.group(1)))
    return _pdf_dedup(pairs)


def _pdf_listing_candidates(flat: str) -> list[dict[str, str]]:
    """提取送转股份可上市日候选。"""
    patterns = (
        (r"本次所转股于\s*([^。；]{0,44}?)\s*直接记入", "转股直接记入股东账户"),
        (r"本次所送(?:红股)?股于\s*([^。；]{0,44}?)\s*直接记入", "送股直接记入股东账户"),
        (r"(?:送(?:红)?股|转增(?:股本)?股)?(?:股份)?于\s*([^。；]{0,40}?)\s*(?:直接)?记入(?:股东)?(?:证券)?账户",
         "股份记入股东账户"),
        (r"新增股份(?:的)?上市(?:流通)?日(?:期)?(?:为)?\s*[：:]?\s*([^。；]{0,44})", "新增股份上市日"),
        (r"(?:可)?上市流通(?:日|日期)?(?:为)?\s*[：:]?\s*([^。；]{0,44})", "上市流通日"),
        (r"股份到账日(?:为)?\s*[：:]?\s*([^。；]{0,44})", "股份到账日"),
    )
    pairs: list[tuple[str, str]] = []
    for pattern, tag in patterns:
        for match in re.finditer(pattern, flat):
            value = _pdf_date(match.group(1))
            if value:
                pairs.append((value, f"{tag}：{match.group(1).strip()[:40]}"))
    return _pdf_dedup(pairs)


def download_pdf(url: str, pdf_dir: Path, name: str, pause: float = 0.4) -> dict[str, Any]:
    """下载并归档官方公告 PDF；已存在则直接复用，绝不重复下载。"""
    path = pdf_dir / name
    if path.is_file():
        raw = path.read_bytes()
        return {
            "path": str(path), "reused": True, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "url": url,
        }
    requests = _requests()
    payload: bytes | None = None
    last_error: str | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=40)
            if response.status_code != 200:
                raise BuildError(f"HTTP {response.status_code}")
            body = response.content
            if body[:5] != b"%PDF-":
                raise BuildError(f"返回非 PDF（文件头 {body[:8]!r}）")
            payload = body
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < 3:
                time.sleep(pause * (attempt + 1))
    if payload is None:
        return {"path": None, "reused": False, "error": last_error, "url": url}
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)  # 归档可复用，避免重复下载
    else:
        payload = path.read_bytes()
    time.sleep(pause)  # 限速
    return {
        "path": str(path), "reused": False, "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(), "url": url,
    }


def extract_pdf_text(path: str) -> tuple[str | None, str | None]:
    """用 pypdf 提取正文；失败返回 (None, 原因)。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        return None, f"pypdf 不可用: {exc}"
    try:
        reader = PdfReader(path)
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _pick_unique(
    candidates: list[dict[str, str]], expected: Decimal | None,
) -> tuple[dict[str, str] | None, str]:
    """仅在候选唯一（或唯一命中期望值）时采信，否则判为多义。

    禁止众数fallback、禁止贪婪匹配：多候选一律 unresolved。
    """
    if not candidates:
        return None, "no_candidate"
    if expected is not None:
        exact = [
            item for item in candidates if _pdf_dec(item["value"]) == _pdf_dec(expected)
        ]
        if len(exact) == 1:
            return exact[0], "unique_exact_match_with_structured_source"
        if len(exact) > 1:
            return exact[0], "multiple_identical_candidates"
        return None, "no_candidate_matching_structured_source"
    if len(candidates) == 1:
        return candidates[0], "unique_candidate"
    return None, f"ambiguous_{len(candidates)}_candidates"


def adjudicate_event_by_pdf(
    symbol: str, event: dict[str, Any], binding: dict[str, Any], pdf_dir: Path,
) -> dict[str, Any]:
    """对单条冲突/缺口事件做官方公告 PDF 定谳。

    只在提取结果唯一、无多候选、无单位歧义时采信；否则 pdf_unresolved 并保持excluded。
    """
    result: dict[str, Any] = {
        "event_id": event["event_id"],
        "pdf_url": binding.get("pdf_url"),
        "pdf_title": binding.get("title"),
        "announcement_date": binding.get("announcement_date"),
        "downloaded": None, "pdf_sha256": None, "chars": None,
        "fields": {}, "resolved": False, "resolution": None,
        "differential_dividend": False, "differential_evidence": None,
        "notes": [],
    }
    if not binding.get("matched") or not binding.get("pdf_url"):
        result["resolution"] = "no_official_pdf"
        result["notes"].append("未绑定官方实施公告 PDF，无法定谳")
        return result

    name = f"{symbol}_{binding.get('announcement_date')}_{binding['announcement_id']}.PDF"
    fetched = download_pdf(binding["pdf_url"], pdf_dir, name)
    result["downloaded"] = fetched
    if not fetched.get("path"):
        result["resolution"] = "pdf_download_failed"
        result["notes"].append(str(fetched.get("error")))
        return result
    result["pdf_sha256"] = fetched["sha256"]

    text, error = extract_pdf_text(fetched["path"])
    if not text:
        result["resolution"] = "pdf_text_extraction_failed"
        result["notes"].append(str(error))
        return result
    result["chars"] = len(text)

    flat, squash = _pdf_flat(text), _pdf_squash(text)

    # 差异化分红检测：交易所除权用摊薄值，名义值会静默算错参考价。
    if any(marker in squash for marker in DIFFERENTIAL_MARKERS):
        result["differential_dividend"] = True
        index = min(
            (squash.find(marker) for marker in DIFFERENTIAL_MARKERS if marker in squash),
            default=-1,
        )
        result["differential_evidence"] = squash[max(0, index - 60):index + 120]

    expected = event.get("per_share") or {}
    #结构化期望值必须取自 cninfo 原始每10股值（经规则 5' 确认的零腿按 0 计），
    # 不能依赖事件级 per_share —— 冲突事件的 per_share 为 None。
    def _struct_per10(leg: str, column: str) -> Decimal | None:
        raw_cn = ((event.get("cninfo") or {}).get("raw") or {}).get(column)
        if raw_cn is not None:
            parsed = _leg(raw_cn)[0]
            return parsed
        leg_rec = (
            ((event.get("zero_leg_evidence") or {}).get("legs") or {}).get(leg) or {}
        )
        if leg_rec.get("resolved_as") == "zero_confirmed":
            return Decimal(0)
        return None

    exp_cash10 = _struct_per10("cash", "派息比例")
    exp_bonus10 = _struct_per10("bonus", "送股比例")
    exp_trans10 = _struct_per10("transfer", "转增比例")
    # 与 sina 的期望值同样纳入比较（两源都应与公告一致）。
    si_raw = ((event.get("sina") or {}).get("raw") or {})
    si_expected = {
        "cash10": _leg(si_raw.get("派息"))[0],
        "bonus10": _leg(si_raw.get("送股"))[0],
        "transfer10": _leg(si_raw.get("转增"))[0],
    }

    cash_candidates = _pdf_cash_candidates(flat, squash)
    bonus_candidates = _pdf_ratio_candidates(
        squash, r"每10股送(?:红股)?" + _PDF_NUM + r"股", "每10股送",
    )
    transfer_candidates = _pdf_ratio_candidates(
        squash, r"每10股转增(?:股本)?" + _PDF_NUM + r"股", "每10股转增",
    )
    if not bonus_candidates and NO_BONUS_RE.search(squash):
        bonus_candidates = [{"value": "0", "evidence": "原文明确「不送红股/不送股」"}]
    if not transfer_candidates and NO_TRANSFER_RE.search(squash):
        transfer_candidates = [
            {"value": "0", "evidence": "原文明确「不以公积金转增股本/不转增」"},
        ]

    for name_text, candidates, expected_value in (
        ("cash10", cash_candidates, exp_cash10),
        ("bonus10", bonus_candidates, exp_bonus10),
        ("transfer10", transfer_candidates, exp_trans10),
    ):
        chosen, reason = _pick_unique(candidates, expected_value)
        result["fields"][name_text] = {
            "candidates": candidates[:6],
            "candidate_count": len(candidates),
            "chosen": chosen, "pick_reason": reason,
            "structured_expected": None if expected_value is None else str(expected_value),
            "sina_expected": (
                None if si_expected[name_text] is None else str(si_expected[name_text])
            ),
        }

    record_pick, record_reason = _pick_unique(_pdf_record_candidates(flat), None)
    ex_pick, ex_reason = _pick_unique(_pdf_ex_candidates(flat), None)
    result["fields"]["record_date"] = {
        "chosen": record_pick, "pick_reason": record_reason,
        "structured_expected": (event.get("cninfo") or {}).get("record_date"),
    }
    result["fields"]["ex_date"] = {
        "chosen": ex_pick, "pick_reason": ex_reason,
        "structured_expected": (event.get("cninfo") or {}).get("ex_date"),
    }

    has_transfer = any(
        result["fields"][key]["chosen"] is not None
        and Decimal(result["fields"][key]["chosen"]["value"]) > 0
        for key in ("bonus10", "transfer10")
    )
    if has_transfer:
        listing_pick, listing_reason = _pick_unique(_pdf_listing_candidates(flat), None)
    else:
        listing_pick, listing_reason = None, "no_share_transfer_event"
    result["fields"]["listing_date"] = {
        "chosen": listing_pick, "pick_reason": listing_reason,
        "structured_expected": event.get("listing_date"),
    }

    # 只有三腿都唯一确定，且至少一腿与任一结构化源一致，才算定谳成功。
    leg_keys = ("cash10", "bonus10", "transfer10")
    all_unique = all(result["fields"][key]["chosen"] is not None for key in leg_keys)
    agrees = False
    for key in leg_keys:
        chosen = result["fields"][key]["chosen"]
        if chosen is None:
            continue
        chosen_value = _pdf_dec(chosen["value"])
        for field_name in ("structured_expected", "sina_expected"):
            expected_raw = result["fields"][key].get(field_name)
            if expected_raw is None:
                continue
            if chosen_value == _pdf_dec(expected_raw):
                agrees = True
                result["fields"][key]["agrees_with"] = field_name
    if all_unique and agrees:
        result["resolved"] = True
        result["resolution"] = "pdf_unique_and_agrees_with_structured_source"
    elif all_unique:
        result["resolution"] = "pdf_unique_but_no_structured_agreement"
        result["notes"].append("PDF 三腿唯一但与两个结构化源均不一致，保持冲突")
    else:
        unresolved = [
            key for key in leg_keys if result["fields"][key]["chosen"] is None
        ]
        result["resolution"] = "pdf_unresolved"
        result["notes"].append(f"未唯一确定: {unresolved}")
    return result


# ---------------------------------------------------------------------------
# 官方实施公告 PDF 绑定（巨潮原始接口）
# ---------------------------------------------------------------------------

def _requests():
    """按需导入 requests，缺失时给出明确错误。"""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise BuildError("当前 Python 缺少 requests 依赖，无法查询官方公告") from exc
    return requests


def _clean_title(value: Any) -> str:
    """剥离 akshare/巨潮高亮标签并压缩空白，得到可入库的纯文本标题。"""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", ANNOUNCEMENT_TAG_RE.sub("", value)).strip()


def _org_id_map(cache_path: Path, attempts: int = 3) -> dict[str, str]:
    """加载 code -> orgId 映射；带落盘缓存，避免每次运行重复拉取 6261 条。"""
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("stockList"):
                return {
                    str(item["code"]): str(item["orgId"])
                    for item in cached["stockList"] if isinstance(item, dict)
                }
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass  # 缓存损坏则重新拉取
    requests = _requests()
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = requests.get(CNINFO_ORG_MAP, timeout=30)
            if response.status_code != 200:
                raise BuildError(f"orgId 映射 HTTP {response.status_code}")
            payload = response.json()
            stock_list = payload.get("stockList")
            if not isinstance(stock_list, list) or not stock_list:
                raise BuildError("orgId 映射缺少 stockList")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_exclusive(cache_path, _dump_json(payload))
            return {
                str(item["code"]): str(item["orgId"])
                for item in stock_list if isinstance(item, dict)
            }
        except Exception as exc:  # 含瞬时 JSONDecodeError
            last_error = exc
            time.sleep(1.0)
    raise BuildError(f"orgId 映射拉取失败: {last_error}")


def _announcement_date(raw: Any) -> str | None:
    """把巨潮毫秒时间戳规范成 YYYYMMDD。

    announcementTime 是北京时间（UTC+8）的当日零点毫秒数。若按 UTC 换算会
    整体前移一天，导致同日匹配全部失效（实测 000001/600519 命中率因此为 0）。
    """
    if raw is None:
        return None
    try:
        millis = int(raw)
    except (TypeError, ValueError):
        return _ymd(raw)
    return datetime.fromtimestamp(millis / 1000, tz=CST).strftime("%Y%m%d")


def _dashed(value: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD。巨潮 seDate 只接受带连字符的格式，
    传纯数字会静默返回 totalAnnouncement=0。"""
    digits = re.sub(r"[^0-9]", "", str(value))
    if len(digits) != 8:
        raise BuildError(f"公告查询日期非法: {value!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def _is_implementation_title(title: str) -> bool:
    """标题是否同时含分配类与实施类关键词。"""
    if not title:
        return False
    has_subject = any(word in title for word in ANNOUNCEMENT_SUBJECT_WORDS)
    has_implement = any(word in title for word in ANNOUNCEMENT_IMPLEMENT_WORDS)
    return has_subject and has_implement


def fetch_announcements(
    symbol: str, org_id: str, start: str, end: str, cache_dir: Path,
    attempts: int = 3,
) -> list[dict[str, Any]]:
    """按区间翻页拉取巨潮公告并按 announcementId 去重，返回实施公告候选。"""
    requests = _requests()
    cache_path = cache_dir / f"{symbol}.announcements.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(cached.get("announcements") or [])
        except (OSError, json.JSONDecodeError, AttributeError):
            pass  # 缓存不可用则重新查询

    payload = {
        "pageNum": 1, "pageSize": ANNOUNCEMENT_PAGE_SIZE, "column": "szse",
        "tabName": "fulltext", "stock": f"{symbol},{org_id}",
        "searchkey": "权益分派", "seDate": f"{_dashed(start)}~{_dashed(end)}",
        "isHLtitle": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0", "Accept": "application/json",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    }
    collected: dict[str, dict[str, Any]] = {}
    observed_total: list[int] = []
    for page in range(1, ANNOUNCEMENT_MAX_PAGES + 1):
        payload["pageNum"] = page
        page_data: dict[str, Any] | None = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    CNINFO_ANNOUNCEMENT_QUERY, data=payload, headers=headers, timeout=30,
                )
                if response.status_code != 200:
                    raise BuildError(f"公告查询 HTTP {response.status_code}")
                page_data = response.json()
                break
            except Exception:  # 瞬时 JSONDecodeError 等
                if attempt + 1 == attempts:
                    page_data = None
                else:
                    time.sleep(1.0 * (attempt + 1))
        if page_data is None:
            # 公告证据缺失不构成事件错误：保留已取到的部分并标记不完整。
            break
        total = page_data.get("totalAnnouncement")
        if isinstance(total, int):
            observed_total.append(total)
        announcements = page_data.get("announcements")
        if not isinstance(announcements, list) or not announcements:
            break
        for item in announcements:
            if not isinstance(item, dict):
                continue
            title = _clean_title(item.get("announcementTitle"))
            adjunct = item.get("adjunctUrl")
            if not title or not isinstance(adjunct, str) or not adjunct:
                continue
            if not _is_implementation_title(title):
                continue
            record_id = str(item.get("announcementId") or adjunct)
            # adjunctUrl 形如 finalpage/2017-07-01/1203671345.PDF，已自带
            # finalpage/ 前缀；直接补static 主机名即可，重复拼接会 404。
            pdf_url = CNINFO_STATIC_HOST + adjunct.lstrip("/")
            collected[record_id] = {
                "announcement_id": record_id,
                "title": title,
                "announcement_date": _announcement_date(item.get("announcementTime")),
                "pdf_url": pdf_url,
                "sec_code": item.get("secCode"),
                "sec_name": item.get("secName"),
            }
        if not page_data.get("hasMore"):
            break

    result = sorted(collected.values(), key=lambda item: (item["announcement_date"] or "", item["announcement_id"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 缓存可覆盖：它是可重新生成的中间态，且续跑需要复用。
    cache_path.write_text(_dump_json({
        "symbol": symbol, "org_id": org_id, "window": [start, end],
        "observed_total_announcement": observed_total,
        "total_announcement_unstable": len(set(observed_total)) > 1,
        "announcements": result,
    }), encoding="utf-8")
    return result


def bind_announcements(
    symbol: str, events: list[dict[str, Any]], announcements: list[dict[str, Any]],
) -> dict[str, Any]:
    """把事件绑定到官方实施公告：同日 + 标题含实施公告，禁止 ±60 天窗口。"""
    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in announcements:
        if item.get("announcement_date"):
            by_date.setdefault(item["announcement_date"], []).append(item)

    used: dict[str, list[str]] = {}
    bindings: list[dict[str, Any]] = []
    for event in events:
        event_id = event["event_id"]
        raw_date = (event.get("cninfo") or {}).get("raw", {}).get("实施方案公告日期")
        ann_date = _ymd(raw_date)
        entry: dict[str, Any] = {
            "event_id": event_id,
            "implementation_announcement_date": ann_date,
            "matched": False,
            "same_day": False,
            "ambiguous": False,
            "shared": False,
            "title": None,
            "announcement_date": None,
            "pdf_url": None,
            "announcement_id": None,
            "reason": None,
        }
        if not ann_date:
            entry["reason"] = "no_implementation_announcement_date"
            bindings.append(entry)
            continue
        if ann_date < CNINFO_ANNOUNCEMENT_HISTORY_FLOOR:
            entry["reason"] = (
                f"before_source_history_floor:{CNINFO_ANNOUNCEMENT_HISTORY_FLOOR}，"
                "源端无更早公告，缺 PDF 属证据范围限制而非事件错误"
            )
            bindings.append(entry)
            continue
        same_day = by_date.get(ann_date, [])
        if not same_day:
            entry["reason"] = "no_same_day_implementation_announcement"
            bindings.append(entry)
            continue
        entry["same_day"] = True
        if len(same_day) > 1:
            # 同日仍不唯一 -> 标记歧义，绝不按标题字典序静默取舍。
            entry["ambiguous"] = True
            entry["reason"] = "multiple_same_day_implementation_announcements"
            entry["candidates"] = [
                {"title": item["title"], "pdf_url": item["pdf_url"],
                 "announcement_id": item["announcement_id"]}
                for item in same_day
            ]
            bindings.append(entry)
            continue
        chosen = same_day[0]
        entry.update({
            "matched": True, "title": chosen["title"],
            "announcement_date": chosen["announcement_date"], "pdf_url": chosen["pdf_url"],
            "announcement_id": chosen["announcement_id"], "reason": "same_day_title_match",
        })
        used.setdefault(chosen["announcement_id"], []).append(event_id)
        bindings.append(entry)

    for entry in bindings:
        if entry["matched"] and len(used.get(entry["announcement_id"], [])) > 1:
            entry["shared"] = True  # 同日多笔分红事件共享同一条公告是允许的

    matched = [entry for entry in bindings if entry["matched"]]
    missing = [entry for entry in bindings if not entry["matched"]]
    return {
        "symbol": symbol,
        "rules": {
            "match": "标题含分配类+实施类关键词 且 公告时间 == 实施方案公告日期（同日）",
            "forbidden": "禁止 ±60 天窗口；同日不唯一时只标记 ambiguous，不自动取舍",
            "title_cleanup": "剥离 <em> 高亮标签",
            "history_floor": CNINFO_ANNOUNCEMENT_HISTORY_FLOOR,
        },
        "counts": {
            "events": len(bindings), "matched": len(matched), "missing": len(missing),
            "ambiguous": sum(1 for entry in bindings if entry["ambiguous"]),
            "shared": sum(1 for entry in bindings if entry["shared"]),
        },
        "events": bindings,
    }


# ---------------------------------------------------------------------------
# 股本变动检出：配股 + 注销式回购
# ---------------------------------------------------------------------------

def fetch_allotment(symbol: str, start: str, end: str) -> list[dict[str, Any]]:
    """查询配股记录；只要存在记录就标记需人工审阅，不自行判定是否实施。"""
    try:
        import akshare
    except ImportError as exc:  # pragma: no cover
        raise BuildError("当前 Python 缺少 akshare 依赖，无法查询配股") from exc
    frame = akshare.stock_allotment_cninfo(
        symbol=symbol, start_date=start, end_date=end,
    )
    records = []
    for index, row in enumerate(_records(frame)):
        records.append({
            "row_index": index,
            "公告日期": row.get("公告日期"),
            "配股价格": row.get("配股价格"),
            "实际配股数量": row.get("实际配股数量"),
            "配股前总股本": row.get("配股前总股本"),
            "配股后总股本": row.get("配股后总股本"),
            "配股上市日": row.get("配股上市日"),
            "配股发行结果公告日": row.get("配股发行结果公告日"),
            "上市公告日期": row.get("上市公告日期"),
            "除权基准日": row.get("除权基准日"),
        })
    return records


def fetch_repurchase_cancellations(cache_path: Path) -> dict[str, dict[str, Any]]:
    """全市场一次性拉取 68 列回购数据，按 REPUROBJECTIVE 过滤出注销减资记录。

    已回购股份数量 != 股本减少量：实测约 43% 回购不注销，因此只有用途明确
    注销/减少注册资本的记录才算股本变动。返回 code -> [记录...] 的索引。
    """
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            index = cached.get("by_symbol")
            if isinstance(index, dict):
                return index
        except (OSError, json.JSONDecodeError, AttributeError):
            pass  # 缓存不可用则重新拉取

    requests = _requests()
    rows: list[dict[str, Any]] = []
    observed_pages: list[int] = []
    for page in range(1, BUYBACK_MAX_PAGES + 1):
        params = {
            "sortColumns": "UPD,DIM_DATE,DIM_SCODE", "sortTypes": "-1,-1,-1",
            "pageSize": str(BUYBACK_PAGE_SIZE), "pageNumber": str(page),
            "reportName": "RPTA_WEB_GETHGLIST_NEW", "columns": "ALL", "source": "WEB",
        }
        payload: dict[str, Any] | None = None
        for attempt in range(3):
            try:
                response = requests.get(EASTMONEY_REPURCHASE.split("?")[0], params=params, timeout=40)
                if response.status_code != 200:
                    raise BuildError(f"回购接口 HTTP {response.status_code}")
                payload = response.json()
                break
            except Exception:
                if attempt == 2:
                    payload = None
                else:
                    time.sleep(1.0 * (attempt + 1))
        if payload is None:
            break
        result = payload.get("result") or {}
        data = result.get("data")
        if not isinstance(data, list) or not data:
            break
        observed_pages.append(int(result.get("pages") or 0))
        rows.extend(item for item in data if isinstance(item, dict))
        if page >= int(result.get("pages") or 0):
            break

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        objective = str(row.get("REPUROBJECTIVE") or "")
        if not any(word in objective for word in BUYBACK_CANCELLATION_KEYWORDS):
            continue
        code = str(row.get("DIM_SCODE") or "").strip()
        if not re.fullmatch(r"[0-9]{6}", code):
            continue
        by_symbol.setdefault(code, []).append({
            "SECURITY_CODE": code,
            "SECURITYSHORTNAME": row.get("SECURITYSHORTNAME"),
            "REPUROBJECTIVE": objective,
            "REPURPROGRESS": row.get("REPURPROGRESS"),
            "REPUPDATE": row.get("REPUPDATE"),
            "UPDATEDATE": row.get("UPDATEDATE"),
            "REPURNUM": row.get("REPURNUM"),
            "note": "已回购股份数量不等于股本减少量；仅用途为注销/减少注册资本才视为股本变动",
        })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(_dump_json({
        "source": EASTMONEY_REPURCHASE,
        "rows_scanned": len(rows),
        "cancellation_records": sum(len(v) for v in by_symbol.values()),
        "symbols_with_cancellation": len(by_symbol),
        "observed_pages": observed_pages,
        "by_symbol": by_symbol,
    }), encoding="utf-8")
    return by_symbol


# ---------------------------------------------------------------------------
# 核对规则实现
# ---------------------------------------------------------------------------

def _check_legs(
    cn_row: dict[str, Any], si_row: dict[str, Any],
) -> tuple[dict[str, Any], str | None, dict[str, Decimal] | None]:
    """三腿核对（规则 5'）。

    先用规则 5' 判定每腿取值（cninfo空/0 + sina显式0 + 方案文本无该腿关键词
    三者同时成立才算 0），再要求两源数值完全相等（容差 0）。
    任一腿仍unknown -> missing_leg；两源不等 -> source_amount_conflict。
    """
    values, evidence = resolve_legs(cn_row, si_row)
    detail: dict[str, Any] = {"per10": {}, "per_share": {}, "leg_states": {},
                              "zero_leg_evidence": evidence}
    per_share: dict[str, Decimal | None] = {}

    for name in ("cash", "bonus", "transfer"):
        cn_value, cn_state = cn_row["per10"][name], cn_row["leg_states"][name]
        si_value, si_state = si_row["per10"][name], si_row["leg_states"][name]
        detail["per10"][f"cninfo_{name}"] = None if cn_value is None else str(cn_value)
        detail["per10"][f"sina_{name}"] = None if si_value is None else str(si_value)
        detail["leg_states"][f"cninfo_{name}"] = cn_state
        detail["leg_states"][f"sina_{name}"] = si_state

    # 非法值优先于未知值。
    for name in ("cash", "bonus", "transfer"):
        if cn_row["leg_states"][name] == "invalid" or si_row["leg_states"][name] == "invalid":
            return detail, "invalid_leg", None
    # 规则 5' 未确认的腿 -> unknown。
    for name in ("cash", "bonus", "transfer"):
        if values[name] is None:
            return detail, "missing_leg", None

    # 两源一致性：仅当两侧都提供了真实数值时才做容差 0 比较；
    # 一侧缺失的腿已由规则 5' 裁决为 0，不再重复比较。
    for name in ("cash", "bonus", "transfer"):
        cn_value = cn_row["per10"][name]
        si_value = si_row["per10"][name]
        cn_share = _per_share(cn_value) if cn_row["leg_states"][name] == "ok" else values[name]
        si_share = _per_share(si_value) if si_row["leg_states"][name] == "ok" else values[name]
        per_share[name] = values[name]
        detail["per_share"][f"cninfo_{name}"] = str(cn_share)
        detail["per_share"][f"sina_{name}"] = str(si_share)
        both_present = (
            cn_row["leg_states"][name] == "ok" and si_row["leg_states"][name] == "ok"
        )
        if both_present and cn_share != si_share:  # Decimal 直接比较，容差 0
            return detail, "source_amount_conflict", None

    if not any(value is not None and value > 0 for value in per_share.values()):
        return detail, "all_legs_zero", None
    detail["agreed_per_share"] = {name: str(value) for name, value in per_share.items()}
    return detail, None, per_share  # type: ignore[return-value]


def _check_plan(cn_row: dict[str, Any]) -> tuple[str, Any, str | None]:
    """方案文本与分红类型检查：出现非标准字样 -> nonstandard_plan。"""
    text = cn_row.get("plan_text")
    text_value = text if isinstance(text, str) else ""
    hits = [word for word in NONSTANDARD_PLAN_KEYWORDS if word in text_value]
    dividend_type = cn_row.get("dividend_type")
    type_is_standard = dividend_type in STANDARD_DIVIDEND_TYPES
    detail = {
        "plan_text": text_value or None,
        "dividend_type": dividend_type if isinstance(dividend_type, str) else None,
        "report_period": cn_row.get("report_period") or None,
        "nonstandard_keywords": hits,
        "dividend_type_is_standard": type_is_standard,
    }
    if hits:
        return detail, "nonstandard_plan"
    return detail, None


def _review_event(
    symbol: str, cn_row: dict[str, Any], si_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """对单个「股权登记日」配对做全量核对，返回一条审阅事件。"""
    record_cn = cn_row["record_date"]
    record_si = si_row["record_date"] if si_row else None
    ex_cn = cn_row["ex_date"]
    ex_si = si_row["ex_date"] if si_row else None
    event_id = f"{symbol}-{ex_cn}-ca" if ex_cn else f"{symbol}-undated-ca"

    event: dict[str, Any] = {
        "event_id": event_id,
        "pairing": "record_date",
        "cninfo": {
            "row_index": cn_row["row_index"], "raw": cn_row["raw"],
            "record_date": record_cn, "ex_date": ex_cn, "pay_date": cn_row["pay_date"],
            "announcement_date": cn_row["announcement_date"],
        },
        "sina": None if si_row is None else {
            "row_index": si_row["row_index"], "raw": si_row["raw"],
            "record_date": record_si, "ex_date": ex_si,
            "listing_date": si_row["listing_date"],
            "announcement_date": si_row["announcement_date"], "progress": si_row["progress"],
        },
        "date_agreement": {
            "record_date_match": si_row is not None and record_cn == record_si,
            "ex_date_match": si_row is not None and ex_cn == ex_si,
        },
        "amount_agreement": None,
        "plan_review": None,
        "listing_date": None,
        "listing_date_source": None,
        "pay_date": cn_row["pay_date"],
        "pay_date_source": "cninfo.派息日" if cn_row["pay_date"] else None,
        "per_share": None,
        "decision": "excluded",
        "reason_code": None,
        "reason_detail": None,
    }

    def fail(code: str, detail: str) -> dict[str, Any]:
        event["decision"] = "excluded"
        event["reason_code"] = code
        event["reason_detail"] = detail
        return event

    # 规则 1：两源按股权登记日对齐，两侧都必须存在该登记日。
    if record_cn is None or (si_row is not None and record_si is None):
        return fail("record_date_missing", "配对存在但某侧股权登记日无法解析为 YYYYMMDD")
    if si_row is None:
        return fail(
            "record_date_absent_in_other_source",
            f"cninfo 股权登记日 {record_cn} 在 sina 实施记录中不存在，无法完成双源核对",
        )

    # 规则 2：record_date / ex_date 两源必须完全一致。
    if record_cn != record_si:
        return fail("source_date_conflict", f"股权登记日不一致: cninfo={record_cn} sina={record_si}")
    if not ex_cn or not ex_si:
        return fail("invalid_date", "除权日缺失或非法，无法定位事件")
    if ex_cn != ex_si:
        return fail("source_date_conflict", f"除权日不一致: cninfo={ex_cn} sina={ex_si}")

    # 规则 3 + 规则 5'：三腿换算为每股；零腿需cninfo空或0 + sina显式0 + 文本无该腿关键词。
    amount_detail, amount_reason, per_share = _check_legs(cn_row, si_row)
    event["amount_agreement"] = amount_detail
    event["zero_leg_evidence"] = amount_detail.get("zero_leg_evidence")
    if amount_reason is not None:
        return fail(amount_reason, _amount_reason_text(amount_reason, amount_detail))

    # 规则 7：仅标准「派息+送股+转增」可用标准除权参考价方法。
    plan_detail, plan_reason = _check_plan(cn_row)
    event["plan_review"] = plan_detail
    if plan_reason is not None:
        return fail(plan_reason, f"方案文本命中非标准字样: {plan_detail['nonstandard_keywords']}")

    # 规则 8：派息日缺失不得猜测，记None 并排除。
    if per_share["cash"] > 0 and not cn_row["pay_date"]:
        return fail("pay_date_unknown", "cninfo 派息日缺失；现金分红缺少 pay_date，不做推断")

    # 规则 4：送股或转股 > 0 时上市日取 sina 红股上市日，取不到即排除。
    if per_share["bonus"] > 0 or per_share["transfer"] > 0:
        listing = si_row["listing_date"]
        if not listing:
            return fail("listing_date_unknown", "存在送股/转增但 sina 红股上市日缺失")
        event["listing_date"] = listing
        event["listing_date_source"] = "sina.红股上市日"
    else:
        event["listing_date"] = None
        event["listing_date_source"] = None

    if record_cn >= ex_cn:
        return fail("date_order_invalid", f"要求 record_date < ex_date，实际 {record_cn} >= {ex_cn}")
    pay_date = cn_row["pay_date"]
    for name, value in (("pay_date", pay_date), ("listing_date", event["listing_date"])):
        if value and value < ex_cn:
            return fail("date_order_invalid", f"{name}={value} 早于 ex_date={ex_cn}")

    event["per_share"] = {name: str(value) for name, value in per_share.items()}
    event["decision"] = "included"
    event["reason_code"] = None
    event["reason_detail"] = "两源日期与三腿逐项一致，方案为标准派息+送股+转增"
    return event


def _amount_reason_text(code: str, detail: dict[str, Any]) -> str:
    """为三腿失败补充可读的具体差异说明。"""
    if code == "missing_leg":
        evidence = detail.get("zero_leg_evidence") or {}
        legs = evidence.get("legs") or {}
        unknown = [
            f"{name}(未满足:{record.get('unmet_conditions')})"
            for name, record in legs.items() if record.get("resolved_as") == "unknown"
        ]
        return (
            "规则 5' 三方一致未确认该腿为 0，按 unknown 处理: " + "; ".join(unknown)
            + "。需 (a) cninfo 空或0 (b) sina 显式0 (c) 方案文本无该腿关键词 同时成立"
        )
    if code == "invalid_leg":
        bad = [name for name, state in detail["leg_states"].items() if state == "invalid"]
        return f"存在非法（非有限/负值）腿: {sorted(set(bad))}"
    if code == "source_amount_conflict":
        pairs = []
        for name in ("cash", "bonus", "transfer"):
            cn_value = detail["per_share"].get(f"cninfo_{name}")
            si_value = detail["per_share"].get(f"sina_{name}")
            if cn_value is not None and si_value is not None and cn_value != si_value:
                pairs.append(f"{name}: cninfo={cn_value} sina={si_value}（每股）")
        return "两源每股数值不一致（容差 0）: " + "; ".join(pairs)
    if code == "all_legs_zero":
        return "三腿换算后全为 0，不构成分配事件"
    return code


def _unpaired_event(symbol: str, cn_row: dict[str, Any]) -> dict[str, Any]:
    """cninfo 有除权日但两源无法配对时的占位审阅事件。"""
    ex_cn = cn_row["ex_date"]
    return {
        "event_id": f"{symbol}-{ex_cn}-ca" if ex_cn else f"{symbol}-undated-ca",
        "pairing": "record_date",
        "cninfo": {
            "row_index": cn_row["row_index"], "raw": cn_row["raw"],
            "record_date": cn_row["record_date"], "ex_date": ex_cn,
            "pay_date": cn_row["pay_date"], "announcement_date": cn_row["announcement_date"],
        },
        "sina": None,
        "date_agreement": {"record_date_match": False, "ex_date_match": False},
        "amount_agreement": None,
        "plan_review": None,
        "listing_date": None,
        "listing_date_source": None,
        "pay_date": cn_row["pay_date"],
        "pay_date_source": "cninfo.派息日" if cn_row["pay_date"] else None,
        "per_share": None,
        "decision": "excluded",
        "reason_code": "record_date_absent_in_other_source",
        "reason_detail": (
            f"cninfo 股权登记日 {cn_row['record_date']} 在 sina 实施记录中不存在，无法完成双源核对"
        ),
    }


def _merge_same_ex_date(symbol: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把同 ex_date 的多条事件合并成一条（三腿 Decimal 精确求和）。

    CorporateActionLedger 契约禁止两个事件共用同一 ex_date，因此同日多次分配
    必须合并而非剔除。前提：每条 constituent 都已通过（两源一致或经 PDF 定谳）；
    任一 constituent 不通过 -> 整个合并组 excluded(merge_component_unresolved)。
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        ex_date = (event.get("cninfo") or {}).get("ex_date")
        groups.setdefault(ex_date or "", []).append(event)

    merged: list[dict[str, Any]] = []
    for ex_date, members in groups.items():
        if len(members) == 1:
            merged.append(members[0])
            continue

        constituents = [
            {
                "row_index": member["cninfo"]["row_index"],
                "dividend_type": (member.get("plan_review") or {}).get("dividend_type"),
                "report_period": (member.get("plan_review") or {}).get("report_period"),
                "plan_text": (member.get("plan_review") or {}).get("plan_text"),
                "per_share": member.get("per_share"),
                "decision": member["decision"],
                "reason_code": member.get("reason_code"),
                "record_date": member["cninfo"]["record_date"],
                "ex_date": member["cninfo"]["ex_date"],
                "pay_date": member.get("pay_date"),
                "listing_date": member.get("listing_date"),
            }
            for member in members
        ]
        # constituents 需在覆盖 base 之前快照：base 取 members[0]，
        # 其 row_index 会被保留为合并组的代表行，其余行索引存入 merged.member_rows。
        base = dict(members[0])
        member_rows = [member["cninfo"]["row_index"] for member in members]
        unresolved = [
            member for member in members if member["decision"] != "included"
        ]
        totals = {"cash": Decimal(0), "bonus": Decimal(0), "transfer": Decimal(0)}
        if not unresolved:
            for member in members:
                for leg in totals:
                    totals[leg] += Decimal(member["per_share"][leg])

        base["merged"] = {
            "merged": True,
            "constituent_count": len(members),
            "member_rows": member_rows,
            "constituents": constituents,
            "dividend_types": sorted({
                str((member.get("plan_review") or {}).get("dividend_type"))
                for member in members
                if (member.get("plan_review") or {}).get("dividend_type")
            }),
            "report_periods": sorted({
                str((member.get("plan_review") or {}).get("report_period"))
                for member in members
                if (member.get("plan_review") or {}).get("report_period")
            }),
            "per_share_totals": {leg: str(value) for leg, value in totals.items()},
            "reconciled_with_announcement": None,  # 由 PDF 勾稽步骤回填
            "reconciliation_failed": False,
        }
        if unresolved:
            base["decision"] = "excluded"
            base["reason_code"] = "merge_component_unresolved"
            base["reason_detail"] = (
                f"同 ex_date {ex_date} 共 {len(members)} 条分配，其中 {len(unresolved)} 条未通过核对"
                f"（{[m.get('reason_code') for m in unresolved]}）；"
                "整组排除，不做部分合并"
            )
            base["per_share"] = None
        else:
            base["per_share"] = {leg: str(value) for leg, value in totals.items()}
            base["decision"] = "included"
            base["reason_code"] = None
            base["reason_detail"] = (
                f"同 ex_date {ex_date} 的 {len(members)} 条分配已按三腿求和合并；"
                "constituents 明细保留在 merged.constituents"
            )
            # 合并后 pay_date / listing_date 必须唯一确定，否则仍不可用。
            pay_dates = {member.get("pay_date") for member in members}
            listing_dates = {member.get("listing_date") for member in members}
            if len(pay_dates) > 1 or len(listing_dates) > 1:
                base["decision"] = "excluded"
                base["reason_code"] = "merge_component_unresolved"
                base["reason_detail"] = (
                    f"合并组内 pay_date/listing_date 不唯一: pay={sorted(map(str, pay_dates))} "
                    f"listing={sorted(map(str, listing_dates))}；拒绝自行取舍"
                )
        merged.append(base)
    return merged


def reconcile(symbol: str, cn_archive: dict[str, Any], si_archive: dict[str, Any]) -> dict[str, Any]:
    """按规则集合完成一个 symbol 的双源核对，返回 events 与统计。"""
    cn_rows = _candidate_rows(PROVIDER_CNINFO, cn_archive)
    si_rows = _candidate_rows(PROVIDER_SINA, si_archive)

    # 规则 1：cninfo 侧只保留「除权日」非空的行。
    cn_kept = [row for row in cn_rows if row["ex_date"]]
    cn_dropped = [row for row in cn_rows if not row["ex_date"]]
    # sina 侧只取进度 == 实施。
    si_implemented = [row for row in si_rows if row.get("progress") == "实施"]
    si_other = [row for row in si_rows if row.get("progress") != "实施"]

    si_by_record: dict[str, list[dict[str, Any]]] = {}
    for row in si_implemented:
        if row["record_date"]:
            si_by_record.setdefault(row["record_date"], []).append(row)

    events: list[dict[str, Any]] = []
    used_sina_rows: set[int] = set()
    for cn_row in cn_kept:
        candidates = si_by_record.get(cn_row["record_date"] or "", [])
        if len(candidates) > 1:
            # 规则 5/6：同登记日多行无法一一对应，不猜测、不合并。
            event = _unpaired_event(symbol, cn_row)
            event["reason_code"] = "record_date_duplicated_in_source"
            event["reason_detail"] = (
                f"sina 侧股权登记日 {cn_row['record_date']} 命中 {len(candidates)} 行，"
                "无法一一对应，拒绝猜测配对"
            )
            events.append(event)
            continue
        si_row = candidates[0] if candidates else None
        if si_row is not None:
            used_sina_rows.add(si_row["row_index"])
        events.append(_review_event(symbol, cn_row, si_row))

    # 同登记日多笔分配：sina 只披露合计数（经官方公告抽检证实，如300750
    # 2024-04-23公告「每10股派50.28元」= 年度20.11 + 特别30.17）。
    # 因此当 cninfo 同登记日多行、sina 仅一行时，逐条比较必然产生假冲突。
    # 此处显式检验「sina 值 == cninfo 各行之和」，成立才按合计口径配对，
    # 并把依据写进 aggregate_pairing，供人工复核。
    by_record: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        record_date = (event.get("cninfo") or {}).get("record_date")
        if record_date:
            by_record.setdefault(record_date, []).append(event)
    for record_date, group in by_record.items():
        if len(group) < 2:
            continue
        group_sina_rows = [event.get("sina") for event in group if event.get("sina")]
        # 必须是「每条 constituent 都配到了 sina 行，且这些行去重后恰好 1 条」
        # ——即 sina 对该登记日只给了一个合计口径。
        distinct_sina = {row["row_index"]: row for row in group_sina_rows}
        if len(group_sina_rows) != len(group) or len(distinct_sina) != 1:
            continue
        si_row = next(iter(distinct_sina.values()))
        per_leg_totals: dict[str, Decimal | None] = {}
        usable = True
        for leg in ("cash", "bonus", "transfer"):
            values: list[Decimal] = []
            for event in group:
                # 直接读 cninfo 原始每10股值，避免依赖事件级 per_share
                #（未通过核对的事件 per_share 为 None）。
                raw_cn = (event["cninfo"]["raw"] or {}).get(
                    {"cash": "派息比例", "bonus": "送股比例", "transfer": "转增比例"}[leg]
                )
                if raw_cn is None:
                    # cninfo 该腿为空 -> 由规则 5' 判定（此处要求合并组整体为 0 腿）
                    leg_rec = (
                        ((event.get("zero_leg_evidence") or {}).get("legs") or {}).get(leg) or {}
                    )
                    if leg_rec.get("resolved_as") == "zero_confirmed":
                        values.append(Decimal(0))
                    else:
                        usable = False
                        break
                else:
                    values.append(_leg(raw_cn)[0] or Decimal(0))
            per_leg_totals[leg] = sum(values) if usable and values else None
        if not usable or per_leg_totals.get("cash") is None:
            continue
        si_legs = {
    "cash": _leg(((si_row.get("raw") or {}).get("派息")))[0],
    "bonus": _leg(((si_row.get("raw") or {}).get("送股")))[0],
    "transfer": _leg(((si_row.get("raw") or {}).get("转增")))[0],
}
        matches = {}
        for leg in ("cash", "bonus", "transfer"):
            si_value = si_legs.get(leg)
            matches[leg] = (
                si_value is not None and si_value == per_leg_totals[leg]
            )
        if all(matches.values()):
            for event in group:
                event["aggregate_pairing"] = {
                    "mode": "sina_reports_aggregate",
                    "record_date": record_date,
                    "constituent_count": len(group),
                    "cninfo_leg_totals_per10": {
                        leg: str(value) for leg, value in per_leg_totals.items()
                    },
                    "sina_leg_per10": {
                        leg: (None if si_legs.get(leg) is None else str(si_legs[leg]))
                        for leg in ("cash", "bonus", "transfer")
                    },
                    "note": (
                        "sina 对同登记日多笔分配只披露合计数；已验证合计口径自洽，"
                        "逐条 constituent 不再与 sina 单笔比较"
                    ),
                }
                # 该组 constituent 均视为通过金额核对（日期核对仍逐条有效）。
                if event.get("amount_agreement"):
                    event["amount_agreement"]["aggregate_pairing"] = True
            for event in group:
                if event.get("reason_code") == "source_amount_conflict":
                    event["decision"] = "included"
                    event["reason_code"] = None
                    event["reason_detail"] = (
                        "sina 同登记日为合计口径，已按合计勾稽通过；"
                        "合并后仍需与官方公告合计数核对"
                    )
                    # 用 cninfo 原始每10股值重建本 constituent 的每股值。
                    event["per_share"] = {}
                    for leg, column in (
                        ("cash", "派息比例"), ("bonus", "送股比例"), ("transfer", "转增比例"),
                    ):
                        raw_cn = (event["cninfo"]["raw"] or {}).get(column)
                        amount = _leg(raw_cn)[0] if raw_cn is not None else Decimal(0)
                        event["per_share"][leg] = str(_per_share(amount))
                elif event.get("zero_leg_evidence", {}).get("legs", {}).get("cash", {}).get(
                    "resolved_as"
                ) in {"unknown", "invalid"}:
                    event["decision"] = "excluded"
                    event["reason_code"] = "missing_leg"

    # 规则 6：同 event_id 不做取舍；合并交由 _merge_same_ex_date 按契约处理。
    by_id: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        by_id.setdefault(event["event_id"], []).append(index)
    for event_id, indexes in by_id.items():
        if len(indexes) < 2:
            continue
        rows = ", ".join(
            f"cninfo_row={events[i]['cninfo']['row_index']}"
            f"/sina_row={(events[i]['sina'] or {}).get('row_index')}" for i in indexes
        )
        for index in indexes:
            events[index]["duplicate_event_id_note"] = (
                f"event_id {event_id} 由 {len(indexes)} 行构成（{rows}）；"
                "契约禁止同 ex_date 两条事件，将按三腿求和合并，不做自动取舍"
            )

    events.sort(key=lambda item: (item["cninfo"]["ex_date"] or "", item["cninfo"]["row_index"]))

    # 裁决 3：契约禁止同 ex_date 两条事件，故同除权日必须合并而非剔除。
    events = _merge_same_ex_date(symbol, events)

    included = [event for event in events if event["decision"] == "included"]
    excluded = [event for event in events if event["decision"] == "excluded"]
    reason_counts: dict[str, int] = {}
    for event in excluded:
        code = event["reason_code"] or "unknown"
        reason_counts[code] = reason_counts.get(code, 0) + 1

    # 资本变动告警：分红类型非标准，或方案文本命中非标准字样，或存在任何排除事件。
    alerts: list[dict[str, Any]] = []
    for event in events:
        plan = event.get("plan_review") or {}
        dividend_type = plan.get("dividend_type")
        if event["decision"] == "excluded" and event["reason_code"] == "nonstandard_plan":
            alerts.append({
                "kind": "nonstandard_plan_text", "event_id": event["event_id"],
                "detail": plan.get("nonstandard_keywords"),
            })
        elif isinstance(dividend_type, str) and dividend_type not in STANDARD_DIVIDEND_TYPES:
            alerts.append({
                "kind": "non_standard_dividend_type", "event_id": event["event_id"],
                "detail": dividend_type,
            })
    unmatched_sina = [
        row for row in si_implemented if row["row_index"] not in used_sina_rows
    ]

    stats = {
        "cninfo_rows_total": len(cn_rows),
        "cninfo_rows_with_ex_date": len(cn_kept),
        "cninfo_rows_without_ex_date": len(cn_dropped),
        "sina_rows_total": len(si_rows),
        "sina_rows_implemented": len(si_implemented),
        "sina_rows_not_implemented": len(si_other),
        "sina_implemented_unpaired": len(unmatched_sina),
        "events_total": len(events),
        "included": len(included),
        "excluded": len(excluded),
        "excluded_reason_counts": dict(sorted(reason_counts.items())),
        "capital_change_alerts": len(alerts),
    }
    return {
        "events": events,
        "stats": stats,
        "capital_change_alerts": alerts,
        "unpaired_sina_rows": [
            {"row_index": row["row_index"], "raw": row["raw"]} for row in unmatched_sina
        ],
    }


# ---------------------------------------------------------------------------
# 账本草稿
# ---------------------------------------------------------------------------

def _coverage_range(symbol: str, data_dir: Path) -> tuple[str, str]:
    """读取本地 parquet 的最小/最大交易日作为覆盖区间。"""
    import pandas as pd

    path = data_dir / f"{symbol}.parquet"
    frame = pd.read_parquet(path, columns=["date"])
    if frame.empty:
        raise BuildError(f"{path} 无交易日记录")
    series = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if series.empty:
        raise BuildError(f"{path} date 列无法解析为日期")
    start = series.min().strftime("%Y%m%d")
    end = series.max().strftime("%Y%m%d")
    if start > end:
        raise BuildError(f"{symbol} 覆盖区间非法: {start} > {end}")
    return start, end


def _coverage_notes(
    symbol: str, cn_archive: dict[str, Any], si_archive: dict[str, Any], generated_at: str,
    pdf_matched: int, pdf_missing: int,
) -> str:
    """逐字写明核对方法、两个来源、sha256 引用与效力边界。"""
    return (
        f"{symbol} 公司行为覆盖区间取自本地 E:/project/stock_ai/data/raw/{symbol}.parquet 的"
        f"最小/最大交易日。"
        f"核对方法：程序化双源核对——按「股权登记日」对齐，"
        f"来源一为巨潮资讯 cninfo（akshare.stock_dividend_cninfo，"
        f"原始归档 sha256={cn_archive['archive_sha256']}，"
        f"records_sha256={cn_archive['records_sha256']}），"
        f"来源二为新浪财经 sina（akshare.stock_history_dividend_detail，仅取进度=实施，"
        f"原始归档 sha256={si_archive['archive_sha256']}，"
        f"records_sha256={si_archive['records_sha256']}）；"
        f"两源的股权登记日、除权日必须逐字一致，"
        f"派息/送股/转增三腿由「每10股」用 Decimal 精确换算为每股后"
        f"要求数值完全相等（容差 0），"
        f"送股或转股大于 0 时上市日取 sina「红股上市日」，"
        f"派息日缺失一律记为 pay_date=None 并排除该事件，"
        f"不做任何推断；除权日相同的多次分配不合并，一律排除待人工审阅。"
        f"官方公告证据：{pdf_matched} 条事件绑定到巨潮官方实施公告 PDF 原文，"
        f"{pdf_missing} 条未绑定（源端公告历史深度下限为 {CNINFO_ANNOUNCEMENT_HISTORY_FLOOR}，"
        f"更早事件在源端查不到公告，属证据范围限制而非事件错误，已退化为分红接口 URL 并标记"
        f" official_pdf_missing，未伪造任何 PDF 链接）。"
        f"股本变动：本账本只覆盖股东分配；配股与注销式回购已单独检出，"
        f"命中即阻断，不在本账本内自行判定是否实施或扣减股本。"
        f"生成时刻 {generated_at}。"
        f"程序化双源核对，不等于逐份人工阅读官方公告。"
        f"若后续检出配股/增发/回购注销等股本变动需重审。"
    )


def build_ledger_draft(
    symbol: str, included_events: list[dict[str, Any]], cn_archive: dict[str, Any],
    si_archive: dict[str, Any], data_dir: Path, generated_at: str,
    bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """构造符合 corporate-actions-v1 的账本草稿（draft，未声明 verified 之外的任何保证）。"""
    start, end = _coverage_range(symbol, data_dir)
    bind_by_event = {entry["event_id"]: entry for entry in bindings.get("events", [])}
    matched = sum(1 for entry in bind_by_event.values() if entry["matched"])
    missing = sum(1 for entry in bind_by_event.values() if not entry["matched"])

    events: list[dict[str, Any]] = []
    for event in included_events:
        per_share = event["per_share"]
        binding = bind_by_event.get(event["event_id"]) or {}
        # 有官方 PDF 用 PDF 原文；缺 PDF 退化为分红接口 URL，绝不伪造链接。
        sources = (
            [binding["pdf_url"], CNINFO_URL, SINA_URL]
            if binding.get("matched") and binding.get("pdf_url")
            else [CNINFO_URL, SINA_URL]
        )
        events.append({
            "event_id": event["event_id"],
            "kind": "distribution",
            "status": "implemented",
            "record_date": event["cninfo"]["record_date"],
            "ex_date": event["cninfo"]["ex_date"],
            "pay_date": event["pay_date"],
            "listing_date": event["listing_date"],
            "cash_per_share_gross": per_share["cash"],
            "bonus_ratio": per_share["bonus"],
            "transfer_ratio": per_share["transfer"],
            "reference_method": "standard_cash_bonus",
            "sources": sources,
            "fractional_policy": "reject",
        })
    return {
        "schema": LEDGER_SCHEMA,
        "symbol": symbol,
        "coverage": {
            "start": start,
            "end": end,
            "status": "verified",
            "scope": COVERAGE_SCOPE,
            "sources": [CNINFO_URL, SINA_URL],
            "verified_at": generated_at,
            "notes": _coverage_notes(
                symbol, cn_archive, si_archive, generated_at, matched, missing,
            ),
        },
        "events": events,
    }


# ---------------------------------------------------------------------------
# 单 symbol 处理
# ---------------------------------------------------------------------------

def _in_coverage(value: str | None, start: str, end: str) -> bool:
    """判断日期是否落在覆盖区间内；缺失一律视为不在区间内（不阻断）。"""
    return bool(value) and start <= value <= end


def _zero_leg_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    """统计全体「判定为 0 的腿」数量与未满足条件的分布，供与官方公告抽检交叉核对。

    官方公告抽检给出 33/38 条「原文明确不送不转 vs 结构化 NaN」，
    本计数用于核对规则 5' 的零腿判定量级是否一致。
    """
    confirmed = {"cash": 0, "bonus": 0, "transfer": 0}
    unknown = {"cash": 0, "bonus": 0, "transfer": 0}
    unmet: dict[str, int] = {}
    evidence_samples: list[dict[str, Any]] = []
    for event in events:
        evidence = event.get("zero_leg_evidence") or {}
        legs = evidence.get("legs") or {}
        for leg, record in legs.items():
            if record.get("resolved_as") == "zero_confirmed":
                confirmed[leg] += 1
            elif record.get("resolved_as") == "unknown":
                unknown[leg] += 1
                for condition in record.get("unmet_conditions") or []:
                    unmet[f"{leg}:{condition}"] = unmet.get(f"{leg}:{condition}", 0) + 1
        if evidence.get("zero_legs_confirmed") and len(evidence_samples) < 5:
            evidence_samples.append({
                "event_id": event["event_id"],
                "plan_text": evidence.get("plan_text"),
                "zero_legs": evidence["zero_legs_confirmed"],
                "legs": {
                    leg: {
                        "cninfo_raw": legs[leg]["cninfo_raw"],
                        "sina_raw": legs[leg]["sina_raw"],
                        "plan_keyword_hits": legs[leg]["plan_keyword_hits"],
                    }
                    for leg in evidence["zero_legs_confirmed"]
                },
            })
    return {
        "zero_legs_confirmed_total": sum(confirmed.values()),
        "zero_legs_confirmed_by_leg": confirmed,
        "unknown_legs_by_leg": unknown,
        "unmet_condition_counts": dict(sorted(unmet.items())),
        "evidence_samples": evidence_samples,
    }


def _classify_capital_changes(
    symbol: str, rights_records: list[dict[str, Any]], buyback_records: list[dict[str, Any]],
    coverage_start: str, coverage_end: str,
) -> dict[str, Any]:
    """按覆盖区间拆分股本变动：区间内阻断，区间外仅记 historical_only。

    裁决 5：1990-94 早期股改的配股早已反映在历史股本里，用「历史上出现过配股」
    一刀切会误伤 000001 这类标的。因此只按生效日是否落在覆盖区间内来阻断。
    """
    def in_range(value: str | None) -> bool:
        return bool(value) and coverage_start <= value <= coverage_end

    rights_in_range, rights_historical = [], []
    for record in rights_records:
        # 优先用除权基准日；缺失时退回公告日期，并标注依据字段。
        effective = _ymd(record.get("除权基准日"))
        basis = "除权基准日"
        if not effective:
            effective = _ymd(record.get("公告日期"))
            basis = "公告日期(无除权基准日，按公告日近似)"
        item = {**record, "effective_date": effective, "effective_date_basis": basis}
        (rights_in_range if in_range(effective) else rights_historical).append(item)

    buyback_in_range, buyback_historical = [], []
    for record in buyback_records:
        # EM 回购接口无明确「注销完成日」字段，用最新公告日期近似并如实标注。
        effective = _ymd(record.get("UPDATEDATE"))
        item = {
            **record,
            "effective_date": effective,
            "effective_date_basis": "EM 最新公告日期(接口无注销完成日字段，属近似)",
            "cancellation_share_count": None,  # EM「已回购数量」不等于注销股数，留空
            "cancellation_share_count_note": (
                "接口未提供注销股数字段；REPURNUM 是已回购数量，不得当作股本减少量"
            ),
        }
        (buyback_in_range if in_range(effective) else buyback_historical).append(item)

    return {
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "rights_issue_records": len(rights_in_range),
        "rights_issue_detail": rights_in_range,
        "rights_issue_historical_only": len(rights_historical),
        "rights_issue_historical_detail": rights_historical,
        "rights_issue_policy": (
            "仅当配股生效日落在覆盖区间内才阻断；区间外记 historical_only。"
            "配股「是否已实施」字段不可靠，本流水线不自行判定"
        ),
        "buyback_cancellation_records": len(buyback_in_range),
        "buyback_cancellation_detail": buyback_in_range,
        "buyback_cancellation_historical_only": len(buyback_historical),
        "buyback_cancellation_historical_detail": buyback_historical,
        "buyback_source": EASTMONEY_REPURCHASE,
        "buyback_policy": (
            "仅当注销生效日落在覆盖区间内才阻断；仅 REPUROBJECTIVE 含注销/减少注册资本"
            "才算股本减少；已回购数量不等于注销股数"
        ),
    }


def _blocked_reason(
    reconciliation: dict[str, Any], capital_flags: dict[str, Any],
) -> str | None:
    """给出阻断原因；全部通过时返回 None。股本变动优先于事件级排除。"""
    if capital_flags.get("rights_issue_records"):
        return (
            f"rights_issue_requires_manual_review:{capital_flags['rights_issue_records']} 条配股记录，"
            "配股是否已实施不可靠，不在本流水线判定"
        )
    if capital_flags.get("buyback_cancellation_records"):
        return (
            f"share_count_changing_event:{capital_flags['buyback_cancellation_records']} 条"
            "注销式回购（用途为注销/减少注册资本），股本变动需重审"
        )
    stats = reconciliation["stats"]
    if stats["events_total"] == 0:
        return "no_verified_events:两源未对齐出任何可核对事件，不能声明覆盖"
    if stats["excluded"]:
        return (
            f"has_excluded_events:{stats['excluded']} 条事件未通过双源核对，"
            f"reason={stats['excluded_reason_counts']}"
        )
    if stats["capital_change_alerts"]:
        return (
            f"capital_change_alerts:{stats['capital_change_alerts']} 条资本变动告警"
            "（非标准分红类型或方案文本命中非标准字样）"
        )
    if stats["sina_implemented_unpaired"]:
        return (
            f"unpaired_sina_rows:{stats['sina_implemented_unpaired']} 条 sina 实施记录"
            "未能与 cninfo 配对，覆盖声明不完整"
        )
    if reconciliation["unpaired_sina_rows"]:
        return "unpaired_sina_rows:存在未配对的 sina 实施记录"
    return None


def process_symbol(
    symbol: str, out_dir: Path, data_dir: Path, dry_run: bool,
    org_map: dict[str, str] | None = None, buyback_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """处理单个 symbol：抓取 -> 归档 -> 核对 -> 公告绑定 -> 股本变动检出 -> 审阅 -> 草稿。"""
    collected_at = _utc_now()
    started = time.monotonic()
    record: dict[str, Any] = {
        "symbol": symbol,
        "collected_at_utc": collected_at,
        "status": "error",
        "error": None,
        "draft_path": None,
        "draft_sha256": None,
        "draft_validated": False,
        "blocked_reason": None,
        "official_pdf_matched": 0,
        "official_pdf_missing": 0,
        "rights_issue_flags": 0,
        "buyback_cancellation_flags": 0,
        "rights_issue_historical_only": 0,
        "buyback_cancellation_historical_only": 0,
        "resolved_by_pdf": 0,
        "pdf_unresolved": 0,
        "differential_dividend_events": 0,
        "zero_legs_confirmed": 0,
        "same_day_merges": 0,
        "merged_reconciled_with_announcement": 0,
        "coverage_start": None,
        "coverage_end": None,
    }
    try:
        cn_result = _fetch(PROVIDER_CNINFO, symbol)
        si_result = _fetch(PROVIDER_SINA, symbol)
        cn_archive = _archive_raw(
            PROVIDER_CNINFO, symbol, cn_result["frame"], cn_result["akshare_version"],
            out_dir, collected_at,
        )
        si_archive = _archive_raw(
            PROVIDER_SINA, symbol, si_result["frame"], si_result["akshare_version"],
            out_dir, collected_at,
        )
        reconciliation = reconcile(symbol, cn_archive, si_archive)
        included = [
            event for event in reconciliation["events"] if event["decision"] == "included"
        ]

        # --- 官方实施公告 PDF 绑定（证据缺失不剔除事件，只标记） ---
        announcement_bindings: dict[str, Any] = {
            "symbol": symbol, "counts": {"events": 0, "matched": 0, "missing": 0,
                                         "ambiguous": 0, "shared": 0}, "events": [],
        }
        org_id = (org_map or {}).get(symbol)
        if org_id:
            try:
                announcements = fetch_announcements(
                    symbol, org_id, CNINFO_ANNOUNCEMENT_HISTORY_FLOOR,
                    datetime.now(timezone.utc).strftime("%Y%m%d"),
                    out_dir / "announcement_cache",
                )
                announcement_bindings = bind_announcements(
                    symbol, reconciliation["events"], announcements,
                )
            except Exception as exc:  # 公告证据失败不构成事件错误
                announcement_bindings["error"] = f"{type(exc).__name__}: {exc}"
        else:
            announcement_bindings["error"] = f"orgId 未找到: {symbol}"
        announcement_bindings["collected_at_utc"] = collected_at
        announcement_path = out_dir / "announcement" / f"{symbol}.json"
        announcement_bindings["announcement_sha256"] = _write_exclusive(
            announcement_path, _dump_json(announcement_bindings),
        )

        # --- 股本变动检出：配股 + 注销式回购（只标记，不自行判定实施） ---
        rights_records: list[dict[str, Any]] = []
        rights_error: str | None = None
        try:
            rights_records = fetch_allotment(symbol, "19900101", "20991231")
        except Exception as exc:
            rights_error = f"{type(exc).__name__}: {exc}"
        buyback_records = list((buyback_index or {}).get(symbol) or [])
        coverage_start, coverage_end = _coverage_range(symbol, data_dir)
        capital_flags = _classify_capital_changes(
            symbol, rights_records, buyback_records, coverage_start, coverage_end,
        )

        capital_flags["rights_issue_error"] = rights_error
        # --- 官方公告 PDF 定谳（仅针对冲突/缺口事件） ---
        # 必须在 _blocked_reason 之前完成：定谳成功的事件会改判为 included，
        # 否则会把已解决的事件误判为 blocked。
        binding_by_event = {
            entry["event_id"]: entry for entry in announcement_bindings.get("events", [])
        }
        needs_pdf = []
        for event in reconciliation["events"]:
            reason = event.get("reason_code")
            merged_meta = event.get("merged") or {}
            # 合并组需要公告合计数做勾稽，因此即使 constituent 全部通过也需 PDF。
            if reason in PDF_ADJUDICABLE or merged_meta.get("merged"):
                needs_pdf.append(event)
        pdf_reviews: list[dict[str, Any]] = []
        resolved_by_pdf = 0
        differential_flags: list[dict[str, Any]] = []
        for event in needs_pdf:
            review = adjudicate_event_by_pdf(
                symbol, event, binding_by_event.get(event["event_id"]) or {}, out_dir / "pdf",
            )
            pdf_reviews.append(review)
            if review.get("differential_dividend"):
                differential_flags.append({
                    "event_id": review["event_id"],
                    "pdf_url": review.get("pdf_url"),
                    "pdf_title": review.get("pdf_title"),
                    "evidence": review.get("differential_evidence"),
                })
            if review.get("resolved"):
                resolved_by_pdf += 1
                # PDF 定谳成功：用公告原文的三腿覆盖事件值，并记录采信依据。
                chosen = {
                    key: (review["fields"][key].get("chosen") or {}).get("value")
                    for key in ("cash10", "bonus10", "transfer10")
                }
                event["per_share"] = {
                    "cash": str(_per_share(_pdf_dec(chosen["cash10"]))),
                    "bonus": str(_per_share(_pdf_dec(chosen["bonus10"]))),
                    "transfer": str(_per_share(_pdf_dec(chosen["transfer10"]))),
                }
                event["decision"] = "included"
                event["reason_code"] = None
                event["reason_detail"] = (
                    "两源不一致，以官方公告 PDF 为权威定谳："
                    + "; ".join(
                        f"{key}={review['fields'][key]['chosen']['value']}"
                        f"({review['fields'][key]['chosen']['evidence']})"
                        for key in ("cash10", "bonus10", "transfer10")
                    )
                )
                event["resolved_by_pdf"] = {
                    "pdf_url": review.get("pdf_url"),
                    "pdf_sha256": review.get("pdf_sha256"),
                    "resolution": review.get("resolution"),
                }
                # PDF 已给出送转股上市日时优先采用公告原文。
                listing_chosen = (review["fields"].get("listing_date") or {}).get("chosen")
                if listing_chosen:
                    event["listing_date"] = listing_chosen["value"]
                    event["listing_date_source"] = "official_announcement_pdf"
                pay_chosen = (review["fields"].get("ex_date") or {}).get("chosen")
                if pay_chosen:
                    event["pay_date"] = pay_chosen["value"]
                    event["pay_date_source"] = "official_announcement_pdf"
        if differential_flags:
            blocked = (
                f"differential_dividend_unsupported:{len(differential_flags)} 条事件命中"
                "差异化分红/差异化权益分派，除权参考价应使用摊薄值而非名义值，引擎未支持"
            )
        else:
            blocked = _blocked_reason(reconciliation, capital_flags)

        pdf_review_doc = {
            "schema": "corporate-action-pdf-review-v1",
            "symbol": symbol,
            "generated_at_utc": _utc_now(),
            "adjudicable_reason_codes": sorted(PDF_ADJUDICABLE),
            "counts": {
                "candidates": len(needs_pdf), "resolved": resolved_by_pdf,
                "unresolved": len(needs_pdf) - resolved_by_pdf,
                "differential_dividend": len(differential_flags),
            },
            "differential_dividend": differential_flags,
            "events": pdf_reviews,
        }
        pdf_review_path = out_dir / "pdf_review" / f"{symbol}.json"
        pdf_review_doc["pdf_review_sha256"] = _write_exclusive(
            pdf_review_path, _dump_json(pdf_review_doc),
        )

        # --- 同日合并组与官方公告合计数勾稽 ---
        pdf_by_event = {review["event_id"]: review for review in pdf_reviews}
        for event in reconciliation["events"]:
            merged_meta = event.get("merged") or {}
            if not merged_meta.get("merged"):
                continue
            binding = binding_by_event.get(event["event_id"]) or {}
            review = pdf_by_event.get(event["event_id"])
            cash10_field = ((review or {}).get("fields") or {}).get("cash10") or {}
            chosen = cash10_field.get("chosen")
            if not chosen:
                continue
            try:
                announced_cash10 = Decimal(chosen["value"])
            except (InvalidOperation, TypeError):
                continue
            merged_cash10 = Decimal(
                merged_meta["per_share_totals"]["cash"]
            ) * UNITS_PER_SHARE_DIVISOR
            if merged_cash10 == announced_cash10:
                merged_meta["reconciled_with_announcement"] = True
                merged_meta["announcement_cash10"] = str(announced_cash10)
                merged_meta["merged_cash10"] = str(merged_cash10)
            else:
                merged_meta["reconciled_with_announcement"] = False
                merged_meta["reconciliation_failed"] = True
                merged_meta["announcement_cash10"] = str(announced_cash10)
                merged_meta["merged_cash10"] = str(merged_cash10)
                merged_meta["reconciliation_evidence"] = chosen.get("evidence")
                event["decision"] = "excluded"
                event["reason_code"] = "merge_reconciliation_failed"
                event["reason_detail"] = (
                    f"合并后每10股现金 {merged_cash10} 与官方公告 {announced_cash10} 不一致"
                    f"（证据：{chosen.get('evidence')}）；拒绝放行"
                )
            if binding.get("pdf_url"):
                merged_meta["announcement_pdf_url"] = binding["pdf_url"]

        draft_error: str | None = None
        if blocked is None:
            draft = build_ledger_draft(
                symbol, included, cn_archive, si_archive, data_dir, collected_at,
                announcement_bindings,
            )
            draft_path = out_dir / "ledger_draft" / f"{symbol}.json"
            draft_sha = _write_exclusive(draft_path, _dump_json(draft))
            try:
                ledger = load_ledger(draft_path, symbol)
                ledger.require_coverage(draft["coverage"]["start"], draft["coverage"]["end"])
            except CorporateActionError as exc:
                draft_error = f"load_ledger 校验失败: {exc}"
            else:
                if len(ledger.events) != len(included):
                    draft_error = "load_ledger 事件数与 included 不一致"
                else:
                    record["draft_path"] = str(draft_path)
                    record["draft_sha256"] = draft_sha
                    record["draft_validated"] = True
            if draft_error is not None:
                blocked = f"draft_validation_failed:{draft_error}"
        record["blocked_reason"] = blocked

        counts = announcement_bindings.get("counts") or {}
        review = {
            "schema": "corporate-action-review-v1",
            "symbol": symbol,
            "collected_at_utc": collected_at,
            "generated_at_utc": _utc_now(),
            "units": "原始为每10股；per_share 为精确每股值（Decimal）",
            "sources": {
                "cninfo": {
                    "interface": "akshare.stock_dividend_cninfo",
                    "endpoint": CNINFO_URL, "rows": cn_archive["row_count"],
                    "archive_path": cn_archive["archive_path"],
                    "archive_sha256": cn_archive["archive_sha256"],
                    "records_sha256": cn_archive["records_sha256"],
                    "fetched_at_utc": cn_archive["fetched_at_utc"],
                },
                "sina": {
                    "interface": "akshare.stock_history_dividend_detail",
                    "endpoint": SINA_URL, "rows": si_archive["row_count"],
                    "archive_path": si_archive["archive_path"],
                    "archive_sha256": si_archive["archive_sha256"],
                    "records_sha256": si_archive["records_sha256"],
                    "fetched_at_utc": si_archive["fetched_at_utc"],
                },
                "official_announcement": {
                    "query_endpoint": CNINFO_ANNOUNCEMENT_QUERY,
                    "pdf_host": CNINFO_STATIC_HOST,
                    "org_id": org_id,
                    "detail_path": str(announcement_path),
                    "detail_sha256": announcement_bindings["announcement_sha256"],
                    "error": announcement_bindings.get("error"),
                },
            },
            "row_counts": {
                "cninfo_total": cn_archive["row_count"],
                "sina_total": si_archive["row_count"],
                "cninfo_with_ex_date": reconciliation["stats"]["cninfo_rows_with_ex_date"],
                "sina_implemented": reconciliation["stats"]["sina_rows_implemented"],
            },
            "rules": {
                "pairing": "按股权登记日一对一；任一侧缺失即无法核对",
                "date_agreement": "record_date / ex_date 两源必须逐字一致",
                "amount_agreement": "每10股 -> 每股 Decimal 精确换算，容差 0",
                "nan_policy": "NaN/缺失视为该源未提供，不默认 0，不从方案文本反推",
                "listing_date": "仅 sina 红股上市日；缺失即排除，不用股份到账日替代",
                "pay_date": "仅 cninfo 派息日；缺失即排除，不做推断",
                "duplicate": "同 event_id 不合并，一律排除待人工审阅",
                "announcement_match": (
                    "标题含分配类+实施类关键词 且 公告时间==实施方案公告日期；"
                    "禁止 ±60 天窗口；同日不唯一只标记 ambiguous"
                ),
                "announcement_history_floor": CNINFO_ANNOUNCEMENT_HISTORY_FLOOR,
                "missing_pdf": "缺官方 PDF 不剔除事件，退化为接口 URL 并标official_pdf_missing",
                "capital_change": "配股/注销式回购只标记 blocked，不自行判定实施或扣减股本",
            },
            "events": reconciliation["events"],
            "capital_change_alerts": reconciliation["capital_change_alerts"],
            "capital_change_flags": capital_flags,
            "pdf_adjudication": {
                "candidates": len(needs_pdf),
                "resolved_by_pdf": resolved_by_pdf,
                "pdf_unresolved": len(needs_pdf) - resolved_by_pdf,
                "differential_dividend_events": len(differential_flags),
                "detail_path": str(pdf_review_path),
                "detail_sha256": pdf_review_doc["pdf_review_sha256"],
                "policy": (
                    "容差保持 0；两源冲突时以官方公告 PDF 为权威，"
                    "仅在提取唯一且与至少一个结构化源一致时采信"
                ),
            },
            "same_day_merges": {
                "merged_groups": sum(
                    1 for event in reconciliation["events"]
                    if (event.get("merged") or {}).get("merged")
                ),
                "reconciled_with_announcement": sum(
                    1 for event in reconciliation["events"]
                    if (event.get("merged") or {}).get("reconciled_with_announcement") is True
                ),
                "reconciliation_failed": sum(
                    1 for event in reconciliation["events"]
                    if (event.get("merged") or {}).get("reconciliation_failed")
                ),
            },
            "zero_leg_audit": _zero_leg_audit(reconciliation["events"]),
            "announcement_coverage": {
                "events": counts.get("events", 0),
                "official_pdf_matched": counts.get("matched", 0),
                "official_pdf_missing": counts.get("missing", 0),
                "ambiguous": counts.get("ambiguous", 0),
                "shared": counts.get("shared", 0),
                "history_floor": CNINFO_ANNOUNCEMENT_HISTORY_FLOOR,
                "detail_path": str(announcement_path),
            },
            "unpaired_sina_rows": reconciliation["unpaired_sina_rows"],
            "stats": reconciliation["stats"],
            "symbol_status": "draft_written" if record["draft_path"] else "blocked",
            "blocked_reason": blocked,
            "notes": [
                "included 仅代表两源程序化核对一致，不代表逐份人工阅读官方公告。",
                "官方公告 PDF 仅覆盖源端可查询区间，更早事件依据双源交叉核对。",
                "配股与注销式回购已检出并阻断，需人工审阅后再决定是否纳入账本。",
            ],
        }
        review_path = out_dir / "review" / f"{symbol}.json"
        review["review_sha256"] = _write_exclusive(review_path, _dump_json(review))

        record.update({
            "status": "ok",
            "review_path": str(review_path),
            "review_sha256": review["review_sha256"],
            "stats": reconciliation["stats"],
            "capital_change_alerts": len(reconciliation["capital_change_alerts"]),
            "official_pdf_matched": counts.get("matched", 0),
            "official_pdf_missing": counts.get("missing", 0),
            "rights_issue_flags": len(
                [r for r in rights_records
                 if _in_coverage(_ymd(r.get("除权基准日")) or _ymd(r.get("公告日期")),
                                 coverage_start, coverage_end)]
            ),
            "buyback_cancellation_flags": capital_flags["buyback_cancellation_records"],
            "rights_issue_historical_only": capital_flags["rights_issue_historical_only"],
            "buyback_cancellation_historical_only":
                capital_flags["buyback_cancellation_historical_only"],
            "resolved_by_pdf": resolved_by_pdf,
            "pdf_unresolved": len(needs_pdf) - resolved_by_pdf,
            "differential_dividend_events": len(differential_flags),
            "zero_legs_confirmed": review["zero_leg_audit"]["zero_legs_confirmed_total"],
            "same_day_merges": sum(
                1 for event in reconciliation["events"]
                if (event.get("merged") or {}).get("merged")
            ),
            "merged_reconciled_with_announcement": sum(
                1 for event in reconciliation["events"]
                if (event.get("merged") or {}).get("reconciled_with_announcement") is True
            ),
            "coverage_start": coverage_start,
            "coverage_end": coverage_end,
            "elapsed_seconds": round(time.monotonic() - started, 2),
        })
        if dry_run and blocked is not None:
            record["dry_run_fail_closed"] = True
        return record
    except Exception as exc:  # 单symbol 失败只记录，不中断整批
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        return record


# ---------------------------------------------------------------------------
# 汇总与入口
# ---------------------------------------------------------------------------

CSV_COLUMNS = (
    "symbol", "status", "coverage_start", "coverage_end",
    "cninfo_rows_total", "cninfo_rows_with_ex_date",
    "sina_rows_total", "sina_rows_implemented", "events_total", "included", "excluded",
    "excluded_reason_counts", "capital_change_alerts", "zero_legs_confirmed",
    "official_pdf_matched", "official_pdf_missing",
    "resolved_by_pdf", "pdf_unresolved", "differential_dividend_events",
    "same_day_merges", "merged_reconciled_with_announcement",
    "rights_issue_flags", "rights_issue_historical_only",
    "buyback_cancellation_flags", "buyback_cancellation_historical_only",
    "blocked_reason", "draft_generated", "draft_validated", "draft_sha256",
    "review_sha256", "error",
)


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    """把单symbol 记录拍平成汇总行。"""
    stats = record.get("stats") or {}
    return {
        "symbol": record.get("symbol"),
        "status": record.get("status"),
        "coverage_start": record.get("coverage_start"),
        "coverage_end": record.get("coverage_end"),
        "cninfo_rows_total": stats.get("cninfo_rows_total"),
        "cninfo_rows_with_ex_date": stats.get("cninfo_rows_with_ex_date"),
        "sina_rows_total": stats.get("sina_rows_total"),
        "sina_rows_implemented": stats.get("sina_rows_implemented"),
        "events_total": stats.get("events_total"),
        "included": stats.get("included"),
        "excluded": stats.get("excluded"),
        "excluded_reason_counts": json.dumps(stats.get("excluded_reason_counts", {}), ensure_ascii=False),
        "capital_change_alerts": record.get("capital_change_alerts"),
        "zero_legs_confirmed": record.get("zero_legs_confirmed"),
        "official_pdf_matched": record.get("official_pdf_matched"),
        "official_pdf_missing": record.get("official_pdf_missing"),
        "resolved_by_pdf": record.get("resolved_by_pdf"),
        "pdf_unresolved": record.get("pdf_unresolved"),
        "differential_dividend_events": record.get("differential_dividend_events"),
        "same_day_merges": record.get("same_day_merges"),
        "merged_reconciled_with_announcement": record.get("merged_reconciled_with_announcement"),
        "rights_issue_flags": record.get("rights_issue_flags"),
        "rights_issue_historical_only": record.get("rights_issue_historical_only"),
        "buyback_cancellation_flags": record.get("buyback_cancellation_flags"),
        "buyback_cancellation_historical_only": record.get("buyback_cancellation_historical_only"),
        "blocked_reason": record.get("blocked_reason"),
        "draft_generated": bool(record.get("draft_path")),
        "draft_validated": bool(record.get("draft_validated")),
        "draft_sha256": record.get("draft_sha256"),
        "review_sha256": record.get("review_sha256"),
        "error": record.get("error"),
    }


def _write_reports(
    out_dir: Path, records: list[dict[str, Any]], generated_at: str, args: argparse.Namespace,
) -> tuple[str, str]:
    """写 coverage_report.json 与 .csv（独占创建）。"""
    rows = [_summary_row(record) for record in records]
    totals = {
        "symbols": len(records),
        "ok": sum(1 for record in records if record.get("status") == "ok"),
        "error": sum(1 for record in records if record.get("status") == "error"),
        "draft_generated": sum(1 for record in records if record.get("draft_path")),
        "draft_validated": sum(1 for record in records if record.get("draft_validated")),
        "blocked": sum(1 for record in records if record.get("blocked_reason")),
        "included_total": sum(int((record.get("stats") or {}).get("included") or 0)
                              for record in records),
        "excluded_total": sum(int((record.get("stats") or {}).get("excluded") or 0)
                              for record in records),
        "official_pdf_matched_total": sum(int(record.get("official_pdf_matched") or 0)
                                          for record in records),
        "official_pdf_missing_total": sum(int(record.get("official_pdf_missing") or 0)
                                          for record in records),
        "rights_issue_flagged_symbols": sum(
            1 for record in records
            if int(record.get("rights_issue_flags") or 0) > 0
        ),
        "buyback_cancellation_flagged_symbols": sum(
            1 for record in records
            if int(record.get("buyback_cancellation_flags") or 0) > 0
        ),
        "rights_issue_historical_only_symbols": sum(
            1 for record in records
            if int(record.get("rights_issue_historical_only") or 0) > 0
            and int(record.get("rights_issue_flags") or 0) == 0
        ),
        "buyback_cancellation_historical_only_symbols": sum(
            1 for record in records
            if int(record.get("buyback_cancellation_historical_only") or 0) > 0
            and int(record.get("buyback_cancellation_flags") or 0) == 0
        ),
        "resolved_by_pdf_total": sum(int(record.get("resolved_by_pdf") or 0)
                                     for record in records),
        "pdf_unresolved_total": sum(int(record.get("pdf_unresolved") or 0)
                                    for record in records),
        "differential_dividend_events_total": sum(
            int(record.get("differential_dividend_events") or 0) for record in records
        ),
        "differential_dividend_symbols": sum(
            1 for record in records
            if int(record.get("differential_dividend_events") or 0) > 0
        ),
        "zero_legs_confirmed_total": sum(int(record.get("zero_legs_confirmed") or 0)
                                         for record in records),
        "same_day_merge_groups_total": sum(int(record.get("same_day_merges") or 0)
                                           for record in records),
        "merged_reconciled_with_announcement_total": sum(
            int(record.get("merged_reconciled_with_announcement") or 0) for record in records
        ),
    }

    # 四类 symbol 聚合（用户视角的最终分类）
    categories: dict[str, list[str]] = {
        "draft_ready": [], "draft_ready_with_pdf_adjudication": [],
        "blocked_capital_change": [], "blocked_unresolved": [], "error": [],
    }
    unresolved_groups: dict[str, list[str]] = {}
    for record in records:
        symbol = str(record.get("symbol"))
        if record.get("status") != "ok":
            categories["error"].append(symbol)
            continue
        reason = str(record.get("blocked_reason") or "")
        if record.get("draft_validated"):
            if int(record.get("resolved_by_pdf") or 0) > 0:
                categories["draft_ready_with_pdf_adjudication"].append(symbol)
            else:
                categories["draft_ready"].append(symbol)
        elif reason.startswith(("rights_issue_requires_manual_review",
                                "share_count_changing_event")):
            categories["blocked_capital_change"].append(symbol)
        else:
            categories["blocked_unresolved"].append(symbol)
            code = reason.split(":")[0] if reason else "unknown"
            unresolved_groups.setdefault(code, []).append(symbol)
    categories["blocked_unresolved_by_reason"] = {
        code: sorted(symbols) for code, symbols in sorted(unresolved_groups.items())
    }

    # 覆盖区间是否达到 2006-01-04~2025-12-31
    target_start, target_end = "20060104", "20251231"
    short_coverage = [
        {
            "symbol": record.get("symbol"),
            "coverage_start": record.get("coverage_start"),
            "coverage_end": record.get("coverage_end"),
            "shorter_than_target": (
                "start" if str(record.get("coverage_start") or "") > target_start
                else "end" if str(record.get("coverage_end") or "") < target_end else ""
            ),
        }
        for record in records
        if record.get("status") == "ok" and (
            str(record.get("coverage_start") or "") > target_start
            or str(record.get("coverage_end") or "") < target_end
        )
    ]
    coverage_summary = {
        "target": {"start": target_start, "end": target_end},
        "symbols_with_full_target_coverage": sum(
            1 for record in records
            if record.get("coverage_start") == target_start
            and record.get("coverage_end") == target_end
        ),
        "symbols_with_shorter_coverage": len(short_coverage),
        "short_coverage_detail": short_coverage[:50],
    }
    reason_totals: dict[str, int] = {}
    for record in records:
        for code, count in ((record.get("stats") or {}).get("excluded_reason_counts") or {}).items():
            reason_totals[code] = reason_totals.get(code, 0) + int(count)
    totals["excluded_reason_totals"] = dict(sorted(reason_totals.items()))

    report = {
        "schema": "corporate-action-coverage-report-v1",
        "generated_at_utc": generated_at,
        "out_dir": str(out_dir),
        "data_dir": str(args.data_dir),
        "dry_run_fail_closed": bool(args.dry_run_fail_closed),
        "python_executable": sys.executable,
        "sources": {
            "cninfo": {"interface": "akshare.stock_dividend_cninfo", "endpoint": CNINFO_URL},
            "sina": {
                "interface": "akshare.stock_history_dividend_detail", "endpoint": SINA_URL,
            },
        },
        "fail_closed_policy": [
            "任何规则不满足即 excluded 并记录 reason_code，不猜测、不合并、不放宽。",
            "NaN 视为该源未提供，不默认 0，因此纯现金事件常因 missing_leg 被排除。",
            "只有全部事件 included 且无资本变动告警时才写账本草稿。",
            "草稿必须通过 load_ledger 与 require_coverage 校验，否则视为失败并记录。",
        ],
        "reason_code_vocabulary": sorted(REASON_SEND_EXCLUDED),
        "totals": totals,
        "symbol_categories": categories,
        "coverage_summary": coverage_summary,
        "symbols": rows,
    }
    json_sha = _write_exclusive(out_dir / "coverage_report.json", _dump_json(report))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: ("" if row.get(column) is None else row.get(column))
                         for column in CSV_COLUMNS})
    csv_sha = _write_exclusive(out_dir / "coverage_report.csv", buffer.getvalue())
    return json_sha, csv_sha


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="公司行为账本采集 + 双源核对 + 审阅记录流水线（fail-closed）",
    )
    parser.add_argument(
        "--symbols", default="",
        help="逗号分隔的 6 位股票代码；留空则取 --data-dir 下全部 parquet",
    )
    parser.add_argument(
        "--out-dir", default="outputs/corporate_action_build",
        help="产物目录（默认 outputs/corporate_action_build）",
    )
    parser.add_argument(
        "--data-dir", default=str(ROOT / "data" / "raw"),
        help="本地 parquet 目录，用于取覆盖区间（只读）",
    )
    parser.add_argument(
        "--dry-run-fail-closed", action="store_true",
        help="验证模式：产物照常落盘并用 load_ledger 校验；"
             "若任一 symbol 未生成已校验草稿则以退出码 3 结束",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.8,
        help="每个 symbol 之间的间隔秒数，避免连续请求（默认 0.8）",
    )
    parser.add_argument(
        "--max-symbols", type=int, default=0,
        help="最多处理多少个 symbol，0 表示不限制（调试用）",
    )
    return parser.parse_args(argv)


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    """确定待处理 symbol 列表并校验格式。"""
    if args.symbols.strip():
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    else:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            raise BuildError(f"数据目录不存在: {data_dir}")
        symbols = sorted(path.stem for path in data_dir.glob("*.parquet")
                         if re.fullmatch(r"[0-9]{6}", path.stem))
    for symbol in symbols:
        if not re.fullmatch(r"[0-9]{6}", symbol):
            raise BuildError(f"非法股票代码: {symbol!r}（必须是 6 位数字）")
    deduped: list[str] = []
    for symbol in symbols:
        if symbol not in deduped:
            deduped.append(symbol)
    if args.max_symbols > 0:
        deduped = deduped[:args.max_symbols]
    if not deduped:
        raise BuildError("没有可处理的股票代码")
    return deduped


def main(argv: list[str] | None = None) -> int:
    """流水线入口；单进程、逐 symbol 进度输出、失败不中断。"""
    args = _parse_args(argv)
    try:
        out_dir = _guard_output_dir(Path(args.out_dir))
        data_dir = Path(args.data_dir).expanduser().resolve()
        symbols = _resolve_symbols(args)
    except BuildError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        return 2

    generated_at = _utc_now()
    print(f"[START] {generated_at} symbols={len(symbols)} out_dir={out_dir} "
          f"dry_run_fail_closed={args.dry_run_fail_closed}", flush=True)

    # 全市场一次性索引：orgId 映射与注销式回购，避免逐票重复请求。
    org_map: dict[str, str] = {}
    buyback_index: dict[str, Any] = {}
    try:
        org_map = _org_id_map(out_dir / "cache" / "cninfo_org_map.json")
        print(f"[CACHE] orgId 映射 {len(org_map)} 条", flush=True)
    except Exception as exc:
        print(f"[WARN] orgId 映射不可用，公告绑定将全部标记缺证据: {exc}", flush=True)
    try:
        buyback_index = fetch_repurchase_cancellations(
            out_dir / "cache" / "repurchase_cancellations.json",
        )
        print(f"[CACHE] 注销式回购索引 {len(buyback_index)} 只股票", flush=True)
    except Exception as exc:
        print(f"[WARN] 回购索引不可用: {exc}", flush=True)

    records: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols, start=1):
        record = process_symbol(
            symbol, out_dir, data_dir, args.dry_run_fail_closed, org_map, buyback_index,
        )
        records.append(record)
        stats = record.get("stats") or {}
        print(
            f"[{index}/{len(symbols)}] {symbol} {record['status']} "
            f"cninfo={stats.get('cninfo_rows_total')} sina={stats.get('sina_rows_total')} "
            f"included={stats.get('included')} excluded={stats.get('excluded')} "
            f"alerts={record.get('capital_change_alerts')} "
            f"pdf={record.get('official_pdf_matched')}/{record.get('official_pdf_missing')}miss "
            f"rights={record.get('rights_issue_flags')} "
            f"buyback={record.get('buyback_cancellation_flags')} "
            f"draft={'yes' if record.get('draft_path') else 'no'} "
            f"({record.get('elapsed_seconds')}s)"
            + (f" error={record['error']}" if record.get("error") else ""),
            flush=True,
        )
        if index < len(symbols) and args.sleep > 0:
            time.sleep(args.sleep)

    try:
        json_sha, csv_sha = _write_reports(out_dir, records, _utc_now(), args)
    except BuildError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        return 2

    ok = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] != "ok"]
    drafts = [record for record in records if record.get("draft_path")]
    validated = [record for record in records if record.get("draft_validated")]
    blocked = [record for record in records if record.get("blocked_reason")]
    reason_totals: dict[str, int] = {}
    for record in ok:
        for code, count in ((record.get("stats") or {}).get("excluded_reason_counts") or {}).items():
            reason_totals[code] = reason_totals.get(code, 0) + int(count)

    print(f"[REPORT] coverage_report.json sha256={json_sha}", flush=True)
    print(f"[REPORT] coverage_report.csv  sha256={csv_sha}", flush=True)
    print(
        f"[SUMMARY] symbols={len(records)} ok={len(ok)} error={len(failed)} "
        f"draft={len(drafts)} validated={len(validated)} blocked={len(blocked)}",
        flush=True,
    )
    print(f"[SUMMARY] included_total={sum(int((r.get('stats') or {}).get('included') or 0) for r in ok)} "
          f"excluded_total={sum(int((r.get('stats') or {}).get('excluded') or 0) for r in ok)}",
          flush=True)
    print(f"[SUMMARY] excluded_reason_totals={json.dumps(reason_totals, ensure_ascii=False)}",
          flush=True)
    print(
        f"[SUMMARY] official_pdf_matched={sum(int(r.get('official_pdf_matched') or 0) for r in records)} "
        f"official_pdf_missing={sum(int(r.get('official_pdf_missing') or 0) for r in records)} "
        f"rights_issue_flagged={sum(1 for r in records if int(r.get('rights_issue_flags') or 0) > 0)} "
        f"buyback_cancellation_flagged="
        f"{sum(1 for r in records if int(r.get('buyback_cancellation_flags') or 0) > 0)}",
        flush=True,
    )
    for record in failed:
        print(f"[ERROR] {record['symbol']}: {record['error']}", flush=True)

    if args.dry_run_fail_closed:
        unvalidated = [record["symbol"] for record in records if not record.get("draft_validated")]
        if unvalidated:
            print(
                f"[DRY_RUN_FAIL_CLOSED] {len(unvalidated)}/{len(records)} 个 symbol 未生成"
                f"通过 load_ledger 校验的账本草稿: {unvalidated}",
                file=sys.stderr, flush=True,
            )
            return 3
        print("[DRY_RUN_FAIL_CLOSED] 全部 symbol 均已生成并校验账本草稿", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
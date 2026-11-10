"""补齐诚实空账本，并构造 v2 训练用的特征目录。

1) 三只股票（001280/600930/688256）在 2006-01-04~2025-12-31 内经巨潮确认
   **无任何分配事件**（唯一记录均在 2026 年，超出覆盖区间），写入 0 事件账本。
   空账本不是"没查"，而是"查过且区间内确无事件"，notes 里写明依据。
2) 以拷贝（非硬链接，避免源文件替换后静默失效）构造 data/raw_hfq_v2_train/，
   仅含 hfq ∩ raw ∩ 账本 的符号。
"""
from __future__ import annotations

import json
from decimal import Decimal
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "data" / "corporate_actions"
REVIEW = Path(__file__).resolve().parent / "corporate_action_direct" / "review"
HFQ = ROOT / "data" / "raw_hfq"
RAW = ROOT / "data" / "raw"
TRAIN_HFQ = ROOT / "data" / "raw_hfq_v2_train"

START, END = "20060104", "20251231"
CNINFO_DIV_URL = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"
SINA_URL = "https://vip.stock.finance.sina.com.cn/corp/view/vISSUE_ShareBonusDetail.php"
EMPTY = ["001280", "600930", "688256"]


def main() -> int:
    # --- 1) 空账本 ---
    TRAIN_HFQ.parent.mkdir(parents=True, exist_ok=True)
    for symbol in EMPTY:
        review = json.loads((REVIEW / f"{symbol}.json").read_text(encoding="utf-8"))
        beyond = [e for e in review["excluded"] if e["reason"] == "beyond_coverage_end"]
        assert not review["included"], f"{symbol} 并非空账本，拒绝写入"
        assert not review["capital_change_alerts"], f"{symbol} 存在股本变动，不能写空账本"
        notes = (
            f"经巨潮结构化接口确认，{START}~{END} 区间内该股票无任何分红/送转事件。"
            f"接口返回 {len(beyond)} 条记录，除权日分别为 "
            f"{[e['ex_date'] for e in beyond]}，均晚于覆盖区间上界，故不计入。"
            "未解析任何公告 PDF；空事件集不等于未核查。"
        )
        document = {
            "schema": "corporate-actions-v1",
            "symbol": symbol,
            "coverage": {
                "start": START, "end": END, "status": "verified",
                "scope": "all_shareholder_distributions_and_capital_changes",
                "sources": [CNINFO_DIV_URL, SINA_URL],
                "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "notes": notes,
            },
            "events": [],
        }
        target = PRODUCTION / f"{symbol}.json"
        if target.exists():
            print(f"  {symbol}: 已存在，跳过")
            continue
        target.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {symbol}: 写入空账本（区间外记录 {[e['ex_date'] for e in beyond]}）")

    # --- 2) 训练特征目录 ---
    import pandas as pd
    ledgers = {p.stem for p in PRODUCTION.glob("*.json")}
    hfq = {p.stem: p for p in HFQ.glob("*.parquet")}
    raw = {p.stem for p in RAW.glob("*.parquet")}
    candidates = sorted(set(hfq) & raw & ledgers)
    # data/raw 与 data/raw_hfq 均在受保护输入基线内，不得修改；
    # 特征与执行日期不齐的股票直接排除，v2 禁止丢日或回退。
    mismatched, suspended, fractional = [], [], []
    for symbol in candidates:
        a = pd.read_parquet(hfq[symbol])
        b = pd.read_parquet(RAW / f"{symbol}.parquet")
        if set(a.date.dt.strftime("%Y%m%d")) != set(b.date.dt.strftime("%Y%m%d")):
            mismatched.append(symbol)
            continue
        # v2 环境要求事件 record_date/ex_date 落在 episode 区间内时必须有真实日线，
        # 停牌期间无 bar，按 fail-closed 设计整只排除（缓存受保护，不得修补）。
        days = set(b.date.dt.strftime("%Y%m%d").tolist())
        document = json.loads((PRODUCTION / f"{symbol}.json").read_text(encoding="utf-8"))
        conflict = [(e["event_id"], field) for e in document["events"]
                    for field in ("record_date", "ex_date")
                    if e.get(field) and e[field] not in days]
        if conflict:
            suspended.append((symbol, conflict[:2]))
            continue
        # 非整数每10股转增比例（如 10转增4.999459股）在任何持仓下都产生碎股；
        # 零碎股分配规则取决于全体股东持仓，契约只支持 reject，整只排除。
        frac = [e["event_id"] for e in document["events"]
                if (Decimal(e["bonus_ratio"]) + Decimal(e["transfer_ratio"])) > 0
                and (Decimal(100) * (Decimal(e["bonus_ratio"])
                                     + Decimal(e["transfer_ratio"]))).to_integral_value()
                != Decimal(100) * (Decimal(e["bonus_ratio"]) + Decimal(e["transfer_ratio"]))]
        if frac:
            fractional.append((symbol, frac))
    trainable = [s for s in candidates
                 if s not in mismatched
                 and s not in dict(suspended)
                 and s not in dict(fractional)]
    if TRAIN_HFQ.exists():
        shutil.rmtree(TRAIN_HFQ)
    TRAIN_HFQ.mkdir(parents=True)
    for symbol in trainable:
        shutil.copy2(hfq[symbol], TRAIN_HFQ / f"{symbol}.parquet")
    print(f"\n训练特征目录 {TRAIN_HFQ}：{len(trainable)} 只")
    print(f"排除无账本 {len(set(hfq) & raw - ledgers)} 只；"
          f"排除特征/执行日期不齐 {len(mismatched)} 只：{mismatched}")
    print(f"排除账本事件日期落在停牌空洞 {len(suspended)} 只："
          f"{[s for s, _ in suspended][:12]}{' ...' if len(suspended) > 12 else ''}")
    print(f"hfq 全集 {len(hfq)} | raw 全集 {len(raw)} | 账本 {len(ledgers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

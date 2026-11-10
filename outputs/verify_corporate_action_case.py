"""Read-only real-bar/event acceptance, explicitly NOT verified historical coverage.

Uses the actual 2023-037 implementation notice and local 300750 raw bars. An
in-memory mechanism-fixture coverage statement is never written/imported into
production data/corporate_actions. No policy is trained and no server is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

from stock_ai.config.schema import EnvConfig
from stock_ai.data.corporate_actions import CorporateAction, CorporateActionError, CorporateActionLedger, load_ledger
from stock_ai.env.corporate_action_env import CorporateActionExecutionEnv
from stock_ai.env.strict_env import ExecutionBars, StrictExecutionEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NOTICE = "https://static.cninfo.com.cn/finalpage/2023-04-18/1216457838.PDF"
EXPECTED_PDF = "8063ac7204c79d9f856b7a2e21887aa650bc5e2b2c1bbbecb81ab54473d0939b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output_path: Path | None = None):
    pdf = OUT / "catl_2022_distribution_official.pdf"
    assert sha(pdf) == EXPECTED_PDF
    text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
    compact = re.sub(r"\s+", "", text)
    assert "2023-037" in compact and "25.20" in compact and "本年度不送股" in compact
    raw_path = ROOT / "data/raw/300750.parquet"
    hfq_path = ROOT / "data/raw_hfq/300750.parquet"
    raw = pd.read_parquet(raw_path)
    days = pd.to_datetime(raw.date.astype(str))
    market = raw.loc[days.between("2023-04-24", "2023-04-28")].copy()
    market["date"] = pd.to_datetime(market.date.astype(str)).dt.strftime("%Y%m%d")
    assert market.date.tolist() == ["20230424", "20230425", "20230426", "20230427", "20230428"]
    snapshots = sorted((OUT / "corporate_action_sources").glob("cninfo_300750_*.snapshot.json"))
    snap = json.loads(snapshots[-1].read_text(encoding="utf-8"))
    assert sha(snap["raw_response_path"]) == snap["response_sha256"]
    candidate = next(item for item in snap["candidates"] if item["ex_date"] == "20230426")
    assert Decimal(candidate["cash_per_share_gross"]) == Decimal("2.52")
    assert Decimal(candidate["transfer_ratio"]) == Decimal("0.8")
    event = CorporateAction(event_id="300750-20230426-2023-037", kind="distribution", status="implemented",
                            record_date="20230425", ex_date="20230426", pay_date="20230426", listing_date="20230426",
                            cash_per_share_gross="2.52", bonus_ratio="0", transfer_ratio="0.8",
                            reference_method="standard_cash_bonus", sources=(NOTICE,))
    # This is a mechanical test harness, NOT a declaration that all capital actions
    # in this or any wider window have been independently reviewed.
    fixture_coverage = {"start": "20230424", "end": "20230428", "status": "verified",
                        "scope": "all_shareholder_distributions_and_capital_changes",
                        "sources": [NOTICE], "verified_at": datetime.now().astimezone().isoformat(),
                        "notes": "IN-MEMORY TEST HARNESS ONLY. Real event, assumed event-isolated window; not historical coverage proof and not importable as production evidence."}
    ledger = CorporateActionLedger("300750", fixture_coverage, (event,))
    bars = ExecutionBars(market.date.to_numpy(), *[market[key].to_numpy() for key in ("open", "high", "low", "close", "volume")], "300750", ledger)
    obs = np.zeros((len(market), 20, 9), dtype="float32")
    cfg = EnvConfig(initial_cash=1_000_000, commission_rate=.00025, stamp_tax_rate=.0005,
                    slippage_rate=.001, min_commission=5, reward_type="gross")
    records, event_log, trades = [], {}, {}
    for label, cls in (("v1", StrictExecutionEnv), ("v2", CorporateActionExecutionEnv)):
        env = cls(obs, bars, cfg)
        try:
            env.reset()
            for index, day in enumerate(market.date):
                info, reward = {}, None
                if index:
                    # Fixed mechanics: T=4/24 buy -> 4/25 open; then hold. Not policy signals.
                    _, reward, _, _, info = env.step(1 if index == 1 else 0)
                state = env.state
                # The legacy reset leaves portfolio_value at zero until its first
                # mark. Compare daily close snapshots, not that unmarked internal
                # reset field: both versions start with the same cash and equity.
                state.mark_to_market(float(market.close.iloc[index]))
                row = {"version": label, "date": day, "close": float(market.close.iloc[index]),
                       "cash": state.cash, "shares": state.shares,
                       "available_shares": getattr(state, "available_shares", state.shares),
                       "locked_shares": getattr(state, "locked_shares", 0),
                       "dividend_receivable": getattr(state, "dividend_receivable", 0),
                       "dividend_income": getattr(state, "dividend_income", 0),
                       "equity": state.portfolio_value, "reward_gross": reward,
                       "limit_reference": info.get("limit_reference"),
                       "limit_up": info.get("limit_up"), "limit_down": info.get("limit_down")}
                records.append(row)
                if label == "v2":
                    unrealized = state.shares * (state.prev_close - state.cost_basis)
                    assert np.isclose(state.portfolio_value, state.cash + state.shares * state.prev_close + state.dividend_receivable)
                    assert np.isclose(state.portfolio_value, cfg.initial_cash + state.realized_pnl + unrealized + state.dividend_income)
            trades[label] = env.trades
            event_log[label] = env.corporate_action_log if label == "v2" else []
        finally:
            env.close()
    assert all(row["equity"] == cfg.initial_cash for row in records if row["date"] == "20230424")
    before = next(row for row in records if (row["version"], row["date"]) == ("v1", "20230425"))
    old = next(row for row in records if (row["version"], row["date"]) == ("v1", "20230426"))
    new = next(row for row in records if (row["version"], row["date"]) == ("v2", "20230426"))
    assert before["shares"] == 2500 and new["shares"] == 4500 and new["dividend_income"] == 6300
    assert new["locked_shares"] == new["dividend_receivable"] == 0
    assert np.isclose(new["equity"] - old["equity"], 2000 * 224.5 + 6300)
    assert len(trades["v1"]) == len(trades["v2"]) == 1
    assert len(event_log["v2"]) == 4
    assert new["limit_reference"] == 212.99 and new["limit_down"] == 170.39 and new["limit_up"] == 255.59
    blocked = None
    try:
        load_ledger(ROOT / "data/corporate_actions/300750.json")
    except CorporateActionError as exc:
        blocked = str(exc)
    assert blocked, "Acceptance must not fabricate/install production verified coverage."
    result = {"generated_at": datetime.now().astimezone().isoformat(), "schema": "corporate-action-case-acceptance-v1",
              "case_type": "real_market_event_mechanism_acceptance_NOT_complete_historical_backtest",
              "symbol": "300750", "range": {"start": "20230424", "end": "20230428"},
              "coverage_proven": False, "production_ledger_installed": False,
              "coverage_warning": "仅核实单事件，未取得全资本行为区间覆盖证明；使用内存机制夹具，不允许导入为verified历史。生产v2请求仍因缺覆盖阻断。",
              "source": {"notice_url": NOTICE, "notice_file": str(pdf), "notice_sha256": sha(pdf),
                         "raw_file": str(raw_path), "raw_sha256": sha(raw_path),
                         "hfq_file": str(hfq_path), "hfq_sha256": sha(hfq_path),
                         "snapshot_file": str(snapshots[-1]), "response_sha256": snap["response_sha256"], "fetched_at": snap["fetched_at"]},
              "event": event.to_dict(), "config": asdict(cfg),
              "action_description": "固定动作：2023-04-24收盘信号买入，4/25开盘成交，其后持有；非模型表现、非买卖建议。20x9零观测仅作机制驱动，未供policy预测。",
              "checks": {"accounting_identities": True, "same_single_trade_no_ca_turnover": True,
                         "official_event_verified": True, "snapshot_is_unreviewed": snap["status"] == "unreviewed",
                         "production_coverage_fail_closed": True, "production_error": blocked},
              "comparison_ex_date": {"old": old, "corrected": new, "equity_difference": new["equity"] - old["equity"]},
              "daily": records, "trades": trades, "corporate_actions": event_log["v2"]}
    path = output_path or OUT / "corporate_action_case_300750_20230426.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    csv_path = path.with_suffix(".csv")
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(json.dumps({"case_file": str(path), "v1_equity": old["equity"], "v2_equity": new["equity"],
                      "difference": result["comparison_ex_date"]["equity_difference"], "production_blocked": True}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON output path; existing results are never overwritten.")
    run(parser.parse_args().output)

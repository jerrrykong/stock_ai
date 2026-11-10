"""直接确认：巨潮是否能按股票代码直接请求分红/送转/配股等除权数据。

只做只读探测，不写生产目录，不分析公告 PDF。
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import akshare as ak
import pandas as pd

OUT = Path(__file__).resolve().parent / "probe_cninfo_direct"
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["000001", "600519", "300750", "000002", "601398"]
RANGE_START, RANGE_END = "20060104", "20251231"


def probe_dividend(symbol: str) -> dict:
    try:
        df = ak.stock_dividend_cninfo(symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if df is None or df.empty:
        return {"ok": True, "rows": 0, "columns": [], "sample": []}
    path = OUT / f"dividend_{symbol}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "ok": True,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "first": json.loads(df.head(1).to_json(orient="records", force_ascii=False)),
        "last": json.loads(df.tail(1).to_json(orient="records", force_ascii=False)),
        "ex_date_nonnull": int(df["除权日"].notna().sum()) if "除权日" in df.columns else None,
        "file": str(path),
    }


def probe_allotment(symbol: str) -> dict:
    try:
        df = ak.stock_allotment_cninfo(symbol=symbol, start_date=RANGE_START, end_date=RANGE_END)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if df is None or df.empty:
        return {"ok": True, "rows": 0, "columns": []}
    path = OUT / f"allotment_{symbol}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    key_cols = [c for c in df.columns if any(k in str(c) for k in ("除权", "登记", "上市", "缴款", "价格", "股本", "配股数量"))]
    return {
        "ok": True,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "key_columns_present": [str(c) for c in key_cols],
        "first": json.loads(df.head(1).to_json(orient="records", force_ascii=False)),
        "file": str(path),
    }


def probe_sina(symbol: str) -> dict:
    try:
        df = ak.stock_history_dividend_detail(symbol=symbol)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if df is None or df.empty:
        return {"ok": True, "rows": 0, "columns": []}
    path = OUT / f"sina_{symbol}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "ok": True,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "first": json.loads(df.head(1).to_json(orient="records", force_ascii=False)),
        "file": str(path),
    }


def main() -> int:
    print(f"akshare={ak.__version__} pandas={pd.__version__}")
    result: dict = {"akshare": ak.__version__, "symbols": {}}
    for symbol in SYMBOLS:
        print(f"\n=== {symbol} ===")
        entry: dict = {}
        for name, fn in (("dividend_cninfo", probe_dividend),
                         ("allotment_cninfo", probe_allotment),
                         ("sina_detail", probe_sina)):
            try:
                info = fn(symbol)
            except Exception:  # noqa: BLE001
                info = {"ok": False, "traceback": traceback.format_exc(limit=3)}
            entry[name] = info
            if info.get("ok"):
                print(f"  {name}: rows={info.get('rows')} cols={len(info.get('columns', []))}")
            else:
                print(f"  {name}: FAILED {info.get('error') or info.get('traceback')}")
        result["symbols"][symbol] = entry
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""静态筛查：用真实前收价对全部事件计算除权参考价与涨跌停，提前抓出运行期才暴露的失败。

复现 src/stock_ai/env/corporate_action_env.py:_limit_reference 的算术，
先用宁德时代 2023-04-26（已验证 参考价 212.99 / 涨停 255.59 / 跌停 170.39）校准。
只读，不修改任何数据。
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stock_ai.env.limits import price_limit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw_hfq_v2_train"
RAW = ROOT / "data" / "raw"
LEDGERS = ROOT / "data" / "corporate_actions"
CENT = Decimal("0.01")


def reference_prices(prev_close: Decimal, cash: Decimal, ratio: Decimal,
                     rate: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    reference = (prev_close - cash) / (Decimal(1) + ratio)
    if reference <= 0:
        return reference, Decimal(0), Decimal(0)
    reference = reference.quantize(CENT, rounding=ROUND_HALF_UP)
    up = (reference * (1 + rate)).quantize(CENT, rounding=ROUND_HALF_UP)
    down = (reference * (1 - rate)).quantize(CENT, rounding=ROUND_HALF_UP)
    if up - reference < CENT:
        up = reference + CENT
    if reference - down < CENT:
        down = reference - CENT
    return reference, up, down


def main() -> int:
    # --- 校准：宁德时代 2023-04-26 ---
    catl = pd.read_parquet(RAW / "300750.parquet")
    closes = catl.set_index(catl.date.dt.strftime("%Y%m%d")).close
    prev = Decimal(str(closes.loc["20230425"]))
    ref, up, down = reference_prices(prev, Decimal("2.52"), Decimal("0.8"),
                                     Decimal(str(price_limit("300750", "20230426"))))
    print(f"[校准] 300750 20230426: 参考价={ref} 涨停={up} 跌停={down} "
          f"(期望 212.99 / 255.59 / 170.39)")
    assert (str(ref), str(up), str(down)) == ("212.99", "255.59", "170.39"), "复现逻辑与引擎不一致"
    print("[校准] 一致\n")

    # --- 全量筛查 ---
    failures = []
    checked = 0
    for path in sorted(TRAIN.glob("*.parquet")):
        symbol = path.stem
        document = json.loads((LEDGERS / f"{symbol}.json").read_text(encoding="utf-8"))
        raw = pd.read_parquet(RAW / f"{symbol}.parquet")
        closes = raw.set_index(raw.date.dt.strftime("%Y%m%d")).close
        days = list(closes.index)
        for event in document["events"]:
            ex = event["ex_date"]
            if ex not in closes.index:
                continue
            position = days.index(ex)
            if position == 0:
                continue
            prev = Decimal(str(closes.iloc[position - 1]))
            cash = Decimal(event["cash_per_share_gross"])
            ratio = Decimal(event["bonus_ratio"]) + Decimal(event["transfer_ratio"])
            rate = Decimal(str(price_limit(symbol, ex)))
            ref, up, down = reference_prices(prev, cash, ratio, rate)
            checked += 1
            if ref <= 0 or down <= 0:
                failures.append((symbol, event["event_id"], ex, str(prev), str(ref),
                                 str(up), str(down), str(rate)))
    print(f"已检查事件 {checked} 个 | 除权参考价/涨跌停不可建模 {len(failures)} 个")
    for item in failures[:20]:
        print("  ", item)
    if failures:
        symbols = sorted({f[0] for f in failures})
        print(f"\n涉及股票 {len(symbols)} 只: {symbols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

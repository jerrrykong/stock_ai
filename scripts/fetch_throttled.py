"""限速 + 容错的批量补拉工具（应对行情源风控）。

背景
----
腾讯行情接口（``web.ifzq.gtimg.cn``）对**单 IP** 存在窗口速率风控：连续快速拉取约
150 只后返回 ``501``（WAF 风控页），约 30 分钟后自动解封。而项目的 ``scale fetch``
既**无节流**，在真实源全部失败时还会写入 ``synthetic`` 假数据（带 WARNING，但会落盘），
对"以真实数据做大规模训练"是不可接受的。

本脚本用于**可控补拉**：
- 逐只串行，带可配节流（``--throttle``）与批间休息（``--batch`` / ``--rest``）；
- **默认剔除 synthetic**（``--allow-synthetic`` 才启用），真实源全失败即记 FAIL，不写假数据；
- 连续失败达阈值判定为触发风控，自动长冷却（``--cooldown``）后继续；
- 单只失败**不中断**整批；失败的代码写入 ``<symbols-file>.failed`` 便于下轮重试。

用法::

    PYTHONPATH=src python scripts/fetch_throttled.py \
        --symbols-file .missing_symbols.txt \
        --source tencent --throttle 3.5 --batch 30 --rest 90

退出码：0 表示脚本正常跑完（是否有 FAIL 见输出与 ``.failed`` 文件）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from stock_ai.config.loader import ConfigLoader  # noqa: E402
from stock_ai.data.loader import StockDataLoader  # noqa: E402
from stock_ai.scale.symbols import resolve_year_range  # noqa: E402
from stock_ai.utils.logging import LoggerFactory  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="限速批量补拉（应对行情源风控）")
    ap.add_argument("--config", default="configs/default.yaml", help="配置文件")
    ap.add_argument("--symbols-file", required=True, help="待补代码列表文件（每行一个）")
    ap.add_argument("--source", default="tencent", help="首选数据源（默认 tencent）")
    ap.add_argument("--start", default=None, help="起始 YYYYMMDD（默认取 scale 年份区间）")
    ap.add_argument("--end", default=None, help="结束 YYYYMMDD（默认取 scale 年份区间）")
    ap.add_argument("--throttle", type=float, default=3.5, help="每只之间的间隔秒（默认 3.5）")
    ap.add_argument("--batch", type=int, default=30, help="每拉多少只做一次批间休息")
    ap.add_argument("--rest", type=float, default=90.0, help="批间休息秒（默认 90）")
    ap.add_argument("--max-fail-streak", type=int, default=8, help="连续失败多少只判为风控")
    ap.add_argument("--cooldown", type=float, default=600.0, help="判定风控后的长冷却秒")
    ap.add_argument("--allow-synthetic", action="store_true", help="允许 synthetic 兜底（默认禁止）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = ConfigLoader.load(args.config)
    cfg.ensure_dirs()
    cfg.data.adjust = "none"
    cfg.data.source = args.source

    yr = resolve_year_range(cfg.scale.years)
    start = args.start or f"{yr[0]}0101"
    end = args.end or f"{yr[1]}1231"

    logger = LoggerFactory.setup(
        "stock_ai.fetch_throttled", cfg.paths.logs, cfg.log_level, log_file="fetch_throttled.log"
    )
    loader = StockDataLoader(cfg.data, logger)

    if not args.allow_synthetic:
        loader.sources = [s for s in loader.sources if getattr(s, "name", "") != "synthetic"]
    logger.info("数据源链: %s", [getattr(s, "name", "?") for s in loader.sources])

    syms = [ln.strip() for ln in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not syms:
        print("[fetch_throttled] 待补列表为空，无需拉取")
        return 0

    print(f"[fetch_throttled] 待补 {len(syms)} 只，区间 {start}~{end}，"
          f"source={args.source}，throttle={args.throttle}s，batch={args.batch}/{args.rest}s")
    print(f"[fetch_throttled] 数据源链: {[getattr(s, 'name', '?') for s in loader.sources]}")

    meta = loader.cache.read_meta()
    ok: list[str] = []
    failed: list[str] = []
    streak = 0

    for i, sym in enumerate(syms, 1):
        try:
            df = loader.fetch_symbol(sym, start, end)
            loader.cache.write(sym, df)
            info = loader._make_info(sym, df, loader._last_source, adjust=cfg.data.adjust)
            loader.cache.update_symbol_meta(info, meta)
            loader.cache.write_meta(meta)
            ok.append(sym)
            streak = 0
            print(f"[{i}/{len(syms)}] OK   {sym} [{loader._last_source}] {len(df)} rows", flush=True)
        except Exception as exc:  # noqa: BLE001 - 逐只容错，绝不中断整批
            failed.append(sym)
            streak += 1
            print(f"[{i}/{len(syms)}] FAIL {sym}: {str(exc)[:150]}", flush=True)
            logger.warning("补拉失败 %s: %s", sym, exc)
            if streak >= args.max_fail_streak:
                print(f"--- 连续失败 {streak} 只，判定触发风控，冷却 {args.cooldown:.0f}s ---", flush=True)
                time.sleep(args.cooldown)
                streak = 0

        if i % args.batch == 0 and i < len(syms):
            print(f"--- 已完成 {i}/{len(syms)}，批间休息 {args.rest:.0f}s ---", flush=True)
            time.sleep(args.rest)
        else:
            time.sleep(args.throttle)

    Path(args.symbols_file + ".failed").write_text("\n".join(failed), encoding="utf-8")
    print(f"\n[fetch_throttled] 完成：OK={len(ok)} FAIL={len(failed)} -> {args.symbols_file}.failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

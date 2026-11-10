"""数据层独立入口：``python -m stock_ai.data.fetch_cli``。

支持 ``fetch`` / ``refresh`` / ``info`` 三个动作，可脱离主 CLI 单独运行。
"""

from __future__ import annotations

import argparse
import sys

from stock_ai.config.loader import ConfigLoader
from stock_ai.data.loader import StockDataLoader
from stock_ai.utils.errors import StockAIError
from stock_ai.utils.io import ensure_utf8_console, write_json
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(prog="python -m stock_ai.data.fetch_cli", description="A 股日线数据拉取/刷新/概览")
    parser.add_argument("action", choices=["fetch", "refresh", "info"], nargs="?", default="fetch")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--source", default=None, help="akshare|tencent|eastmoney|sina|synthetic")
    parser.add_argument("--adjust", default=None, help="复权方式 qfq|hfq|none（默认 none 不复权；不传则读配置）")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """执行数据命令，返回退出码。"""
    cfg = ConfigLoader.load(args.config)
    overrides: dict[str, object] = {}
    if args.symbols:
        overrides["data.symbols"] = list(args.symbols)
    if args.start:
        overrides["data.start"] = args.start
    if args.end:
        overrides["data.end"] = args.end
    if args.source:
        overrides["data.source"] = args.source
    if getattr(args, "adjust", None):
        overrides["data.adjust"] = args.adjust
    if args.cache_dir:
        overrides["data.cache_dir"] = args.cache_dir
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()

    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup("stock_ai.data", cfg.paths.logs, cfg.log_level)
    loader = StockDataLoader(cfg.data, logger)

    if args.action == "fetch":
        frames = loader.fetch_all(start=args.start, end=args.end, force=args.force)
        print(f"[data fetch] 完成 {len(frames)} 只股票 -> {cfg.data.cache_dir}")
        for sym, df in frames.items():
            print(f"  {sym}: {len(df)} 行")
    elif args.action == "refresh":
        frames = loader.refresh_all(args.symbols)
        print(f"[data refresh] 完成 {len(frames)} 只股票")
        for sym, df in frames.items():
            print(f"  {sym}: {len(df)} 行")
    else:  # info
        infos = loader.info(args.symbols)
        for info in infos:
            print(
                f"  {info.symbol}: {info.rows} 行 {info.start}~{info.end} "
                f"source={info.source} adjust={info.adjust or '-'} gaps={info.missing_days}"
            )
        write_json({"symbols": [i.to_dict() for i in infos]}, cfg.paths.reports / "data_info.json")
    return 0


def main() -> None:
    """CLI 主入口。"""
    ensure_utf8_console()
    args = build_parser().parse_args()
    try:
        sys.exit(run(args))
    except StockAIError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

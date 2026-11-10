"""样本层独立入口：``python -m stock_ai.samples.build_cli``。"""

from __future__ import annotations

import argparse
import sys

from stock_ai.config.loader import ConfigLoader
from stock_ai.data.loader import StockDataLoader
from stock_ai.samples.builder import SampleBuilder
from stock_ai.utils.errors import StockAIError
from stock_ai.utils.io import ensure_utf8_console
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(prog="python -m stock_ai.samples.build_cli", description="构建 walk-forward 训练/验证/测试样本")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    """执行样本构建，返回退出码。"""
    cfg = ConfigLoader.load(args.config)
    overrides: dict[str, object] = {}
    if args.symbols:
        overrides["data.symbols"] = list(args.symbols)
    if args.train_start:
        overrides["sample.train_start"] = args.train_start
    if args.train_end:
        overrides["sample.train_end"] = args.train_end
    if args.test_start:
        overrides["sample.test_start"] = args.test_start
    if args.test_end:
        overrides["sample.test_end"] = args.test_end
    if args.features:
        overrides["sample.features"] = list(args.features)
    if args.window is not None:
        overrides["sample.window"] = int(args.window)
    if args.embargo_days is not None:
        overrides["sample.embargo_days"] = int(args.embargo_days)
    if args.out:
        overrides["paths.samples"] = args.out
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()

    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup("stock_ai.samples", cfg.paths.logs, cfg.log_level)

    loader = StockDataLoader(cfg.data, logger)
    builder = SampleBuilder(cfg.sample, logger)

    syms = list(args.symbols) if args.symbols else (cfg.data.symbols or None)
    bundle = builder.build_all(loader, symbols=syms)
    paths = builder.save(bundle, cfg.paths.samples, seed=cfg.random_seed)

    print(f"[sample build] 输出目录: {cfg.paths.samples}")
    print(f"  train: {bundle.train.n_steps} 步")
    print(f"  val  : {bundle.val.n_steps} 步")
    print(f"  test : {bundle.test.n_steps} 步")
    print(f"  features: {len(bundle.feature_names)} 维, window={bundle.window}")
    print(f"  data_hash: {bundle.data_hash}")
    print(f"  meta: {paths['meta']}")
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

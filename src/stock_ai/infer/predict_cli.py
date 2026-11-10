"""推理层独立入口：``python -m stock_ai.infer.predict_cli``。

契约：读 ``models/*.zip`` + ``data/samples/meta.json``（冻结归一化状态）+ ``data/raw``
-> 写 ``reports/predict_<ts>.csv`` + ``reports/predict_<ts>.json``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_ai.config.loader import ConfigLoader
from stock_ai.data.loader import StockDataLoader
from stock_ai.eval.backtest_cli import _load_model
from stock_ai.infer.predictor import Predictor
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.utils.errors import ArtifactError, StockAIError
from stock_ai.utils.io import ensure_utf8_console, read_json, timestamp
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m stock_ai.infer.predict_cli", description="对未知区间预测动作建议"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, help="模型 zip 路径")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start", required=True, help="起始 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束 YYYYMMDD")
    parser.add_argument("--meta", default=None, help="samples/meta.json 路径（冻结归一化状态）")
    parser.add_argument("--out", default=None, help="输出目录")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    """执行预测，返回退出码。"""
    cfg = ConfigLoader.load(args.config)
    overrides: dict[str, object] = {}
    if args.symbols:
        overrides["data.symbols"] = list(args.symbols)
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()

    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup("stock_ai.infer", cfg.paths.logs, cfg.log_level)

    # ---- 冻结归一化状态（K9：绝不重新 fit）----
    meta_path = Path(args.meta) if args.meta else Path(cfg.paths.samples) / "meta.json"
    if not meta_path.exists():
        raise ArtifactError(
            f"样本 meta 不存在: {meta_path}（请先执行 `stock_ai sample build`）"
        )
    meta = read_json(meta_path)
    normalizer = NormalizerState.from_dict(meta.get("normalizer", {}))
    feature_names = list(meta.get("feature_names", cfg.sample.features))
    window = int(meta.get("window", cfg.sample.window))
    logger.info("复用冻结归一化状态（%d 特征, window=%d, hash=%s）",
                normalizer.n_features, window, str(meta.get("data_hash", ""))[:8])

    # ---- 加载模型 ----
    model = _load_model(Path(args.model))

    # ---- 数据 ----
    loader = StockDataLoader(cfg.data, logger)
    symbols = list(args.symbols) if args.symbols else (cfg.data.symbols or meta.get("symbols") or [])
    if not symbols:
        raise ArtifactError("未指定股票池：请用 --symbols 或配置 symbols")

    predictor = Predictor(
        model=model,
        env_cfg=cfg.env,
        normalizer=normalizer,
        feature_names=feature_names,
        window=window,
        mode="discrete",
    )
    rows = predictor.predict_range(symbols, args.start, args.end, loader)

    # ---- 落盘 ----
    ts = timestamp()
    out_dir = Path(args.out) if args.out else Path(cfg.paths.reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = predictor.to_csv(rows, out_dir / f"predict_{ts}.csv")
    json_path = predictor.to_json(rows, out_dir / f"predict_{ts}.json")

    # 统计动作分布
    counts = {"买": 0, "卖": 0, "持有": 0}
    for r in rows:
        counts[r.action] = counts.get(r.action, 0) + 1

    print(f"[predict] {args.start}~{args.end} 共 {len(rows)} 条")
    print(f"  买: {counts.get('买', 0)}  卖: {counts.get('卖', 0)}  持有: {counts.get('持有', 0)}")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
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

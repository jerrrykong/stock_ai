"""评估层独立入口：``python -m stock_ai.eval.backtest_cli``。

契约：读 ``models/*.zip`` + ``data/samples/test.npz`` + ``meta.json``
-> 写 ``reports/backtest_<ts>.json`` + ``reports/equity_<ts>.png``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_ai.config.loader import ConfigLoader
from stock_ai.eval.backtester import Backtester
from stock_ai.eval.plot import EquityPlotter
from stock_ai.samples.schema import SampleSplit
from stock_ai.train.factory import TrainerFactory
from stock_ai.utils.errors import ArtifactError, StockAIError
from stock_ai.utils.io import ensure_utf8_console, read_json, read_npz, timestamp
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["build_parser", "run", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m stock_ai.eval.backtest_cli", description="模型回测评估"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, help="模型 zip 路径")
    parser.add_argument("--samples", default=None, help="test.npz 路径或目录")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--fee", type=float, default=None)
    parser.add_argument("--slippage", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _load_model(model_path: Path, backend_hint: str = "auto") -> "object":
    """按后端自动加载模型（先试 SB3，再试 NumPy）。"""
    from stock_ai.train.ppo_numpy import NumpyPPOTrainer

    p = Path(model_path)
    if p.suffix.lower() != ".zip":
        p = p.with_suffix(".zip")
    if not p.exists():
        raise ArtifactError(
            f"模型文件不存在: {p}（请先执行 `stock_ai train`）", context={"path": str(p)}
        )

    # 通过 zip 内容判断后端
    import json
    import zipfile

    backend = "numpy"
    try:
        with zipfile.ZipFile(str(p)) as zf:
            names = set(zf.namelist())
            if "backend.json" in names:
                backend = json.loads(zf.read("backend.json").decode("utf-8")).get(
                    "backend", "numpy"
                )
            elif "data" in names and "policy.pth" in names:
                backend = "sb3"
            elif "weights.npz" in names:
                backend = "numpy"
    except Exception:  # noqa: BLE001
        backend = backend_hint if backend_hint in ("sb3", "numpy") else "numpy"

    if backend == "sb3" and TrainerFactory.backend_available("sb3"):
        from stock_ai.train.sb3_trainer import SB3Trainer

        return SB3Trainer.load(p)
    return NumpyPPOTrainer.load(p)


def run(args: argparse.Namespace) -> int:
    """执行回测，返回退出码。"""
    cfg = ConfigLoader.load(args.config)
    overrides: dict[str, object] = {}
    if args.cash is not None:
        overrides["env.initial_cash"] = float(args.cash)
    if args.fee is not None:
        overrides["env.commission_rate"] = float(args.fee)
    if args.slippage is not None:
        overrides["env.slippage_rate"] = float(args.slippage)
    if args.start:
        overrides["sample.test_start"] = args.start
    if args.end:
        overrides["sample.test_end"] = args.end
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()

    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup("stock_ai.eval", cfg.paths.logs, cfg.log_level)

    # 解析 test.npz
    samples_p = Path(args.samples) if args.samples else Path(cfg.paths.samples) / "test.npz"
    if samples_p.is_dir():
        samples_p = samples_p / "test.npz"
    elif samples_p.suffix != ".npz":
        samples_p = samples_p.parent / f"{samples_p.name}.npz"
    if not samples_p.exists():
        raise ArtifactError(
            f"测试样本不存在: {samples_p}（请先执行 `stock_ai sample build`）"
        )
    split = SampleSplit.from_npz_dict(read_npz(samples_p))
    logger.info("载入测试样本: %d 步 x %d 特征", split.n_steps, split.n_features)

    meta_path = samples_p.parent / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}

    # 加载模型
    model = _load_model(Path(args.model))
    backend = getattr(
        model, "backend_name",
        "numpy" if model.__class__.__name__.startswith("Numpy") else "sb3",
    )
    algo = "ppo"
    stem = Path(args.model).stem
    for key in ("sac", "dqn", "ppo"):
        if stem.startswith(key):
            algo = key
            break

    # 回测
    bt = Backtester(model, cfg.env, split, mode="discrete")
    test_range = (
        meta.get("test_range", {}).get("start", ""),
        meta.get("test_range", {}).get("end", ""),
    )
    result = bt.run(
        backend=backend,
        algo=algo,
        model_path=str(args.model),
        data_hash=str(meta.get("data_hash", "")),
        test_range=(test_range[0], test_range[1]),
    )

    # 落盘
    ts = timestamp()
    out_dir = Path(args.out) if args.out else Path(cfg.paths.reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"backtest_{ts}.json"
    result.to_json(json_path)
    png_path = out_dir / f"equity_{ts}.png"
    try:
        EquityPlotter().plot(result, png_path, title=f"backtest {algo}/{backend} {test_range[0]}~{test_range[1]}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("权益曲线绘制失败（不阻塞）: %s", exc)

    m = result.metrics
    print(f"[backtest] backend={backend} algo={algo}")
    print(f"  年化收益: {m['annual_return'] * 100:.2f}%")
    print(f"  累计收益: {m['total_return'] * 100:.2f}%")
    print(f"  最大回撤: {m['max_drawdown'] * 100:.2f}%")
    print(f"  夏普比率: {m['sharpe']:.3f}")
    print(f"  胜率    : {m['win_rate'] * 100:.1f}%")
    print(f"  换手率  : {m['turnover']:.3f}")
    print(f"  成交笔数: {m['num_trades']}")
    print(f"  买入持有: {m['benchmark_return'] * 100:.2f}%")
    print(f"  报告: {json_path}")
    print(f"  曲线: {png_path}")
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

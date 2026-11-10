"""训练层独立入口：``python -m stock_ai.train.train_cli``。

契约：读 ``data/samples/train.npz`` + ``meta.json`` -> 写
``models/<algo>_<ts>.zip`` + sidecar ``.json`` + ``logs/<run>/curve.png``。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stock_ai.config.loader import ConfigLoader
from stock_ai.config.schema import TrainConfig
from stock_ai.samples.schema import SampleSplit
from stock_ai.utils.errors import ArtifactError, ConfigError, StockAIError
from stock_ai.utils.io import ensure_utf8_console, read_json, read_npz, timestamp, write_json
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["build_parser", "run", "main", "samples_path"]

#: 训练时使用的动作模式（当前 CLI 固定为离散：买/卖/持有）
_TRAIN_ACTION_MODE: str = "discrete"


def samples_path(samples: str | None, samples_dir: Path) -> Path:
    """解析训练样本路径（支持直接给 ``.npz`` 或给目录）。"""
    if samples:
        p = Path(samples)
        if p.suffix == ".npz":
            return p
        if p.is_dir():
            return p / "train.npz"
        # 视为文件名前缀（如 "train"）
        return (p.parent if p.parent != Path(".") else samples_dir) / f"{p.name}.npz"
    return Path(samples_dir) / "train.npz"


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(prog="python -m stock_ai.train.train_cli", description="PPO/SAC/DQN 训练")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algo", default=None, help="ppo|sac|dqn")
    parser.add_argument("--samples", default=None, help="train.npz 路径或目录")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", default=None, help="auto|sb3|numpy")
    parser.add_argument("--reward", default=None, help="gross|drawdown_penalty|sharpe")
    parser.add_argument("--out", default=None, help="模型输出目录")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    """执行训练，返回退出码。"""
    cfg = ConfigLoader.load(args.config)
    overrides: dict[str, object] = {}
    if args.algo:
        overrides["train.algo"] = args.algo
    if args.timesteps is not None:
        overrides["train.timesteps"] = int(args.timesteps)
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.backend:
        overrides["train.backend"] = args.backend
    if args.reward:
        overrides["env.reward_type"] = args.reward
    if args.out:
        overrides["train.out_dir"] = args.out
    if args.n_steps is not None:
        overrides["train.n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        overrides["train.batch_size"] = int(args.batch_size)
    if args.n_epochs is not None:
        overrides["train.n_epochs"] = int(args.n_epochs)
    if args.learning_rate is not None:
        overrides["train.learning_rate"] = float(args.learning_rate)
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()

    # ---- 参数前置校验（必须早于后端选择，不能被后端回退吞掉）----
    # PRD 验收：参数非法应有明确提示并非零退出。
    from stock_ai.env.rewards import SUPPORTED_REWARDS
    from stock_ai.train.factory import SUPPORTED_ALGOS, SUPPORTED_BACKENDS

    algo = str(cfg.train.algo or "ppo").lower()
    if algo not in SUPPORTED_ALGOS:
        raise ConfigError(
            f"不支持的算法: {cfg.train.algo!r}（可选: {sorted(SUPPORTED_ALGOS)}）",
            context={"algo": cfg.train.algo, "valid": sorted(SUPPORTED_ALGOS)},
        )
    # SAC 仅支持连续动作空间；当前 CLI 训练环境为离散动作（买/卖/持有）。
    if algo == "sac" and _TRAIN_ACTION_MODE != "continuous":
        raise ConfigError(
            "SAC 需要连续动作空间，请使用 ppo/dqn，"
            "或启用 continuous 动作模式（当前训练动作模式为 discrete）",
            context={"algo": "sac", "action_mode": _TRAIN_ACTION_MODE},
        )

    # 非法 --backend 不得被 auto 回退链静默吞掉（P1-1）。
    backend = str(cfg.train.backend or "auto").lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"不支持的后端: {cfg.train.backend!r}（可选: {sorted(SUPPORTED_BACKENDS)}）",
            context={"backend": cfg.train.backend, "valid": sorted(SUPPORTED_BACKENDS)},
        )

    # 非法 --reward 不得被静默按 drawdown_penalty 处理（P2-5）。
    reward = str(cfg.env.reward_type or "drawdown_penalty").lower()
    if reward not in SUPPORTED_REWARDS:
        raise ConfigError(
            f"不支持的奖励类型: {cfg.env.reward_type!r}"
            f"（可选: {sorted(SUPPORTED_REWARDS)}）",
            context={"reward_type": cfg.env.reward_type, "valid": sorted(SUPPORTED_REWARDS)},
        )

    set_global_seed(cfg.random_seed)
    run_id = timestamp()
    run_dir = Path(cfg.paths.logs) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = LoggerFactory.setup(
        "stock_ai.train", run_dir, cfg.log_level, log_file="train.log"
    )
    logger.info("训练启动 run_id=%s backend=%s algo=%s", run_id, cfg.train.backend, cfg.train.algo)

    train_npz = samples_path(args.samples, cfg.paths.samples)
    if not train_npz.exists():
        hint = f"（--samples {args.samples!r} 解析为 {train_npz}）" if args.samples else ""
        raise ArtifactError(
            f"训练样本不存在: {train_npz}{hint}（请先执行 `stock_ai sample build`）"
        )
    meta_path = train_npz.parent / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    split = SampleSplit.from_npz_dict(read_npz(train_npz))
    logger.info("载入训练样本: %d 步 x %d 特征", split.n_steps, split.n_features)

    # ---- 环境 ----
    from gymnasium.utils.env_checker import check_env

    from stock_ai.env.trading_env import TradingEnv
    from stock_ai.train.factory import TrainerFactory

    env = TradingEnv(split, cfg.env, seed=cfg.random_seed, action_mode=_TRAIN_ACTION_MODE)
    try:
        check_env(env, skip_render_check=True)
        logger.info("check_env 通过 ✓")
    except Exception as exc:  # noqa: BLE001
        logger.warning("check_env 警告: %s", exc)

    # ---- 工厂创建训练器 ----
    trainer = TrainerFactory.create(cfg.train, env, preference=cfg.train.backend)
    backend = getattr(trainer, "backend_name", "numpy")
    algo = str(cfg.train.algo).lower()
    logger.info("后端选定: %s (%s)", backend, getattr(trainer, "algo_name", algo))

    # ---- 训练 ----
    t0 = time.time()
    trainer.learn(int(cfg.train.timesteps), seed=cfg.random_seed)
    elapsed = time.time() - t0

    # ---- 保存模型 ----
    model_name = f"{algo}_{run_id}"
    model_path = Path(cfg.train.out_dir) / f"{model_name}.zip"
    trainer.save(model_path)

    # ---- sidecar json（K11）----
    sidecar = {
        "algo": algo,
        "backend": backend,
        "seed": int(cfg.random_seed),
        "data_hash": str(meta.get("data_hash", "")),
        "train_range": meta.get("train_range", {}),
        "test_range": meta.get("test_range", {}),
        "feature_names": list(meta.get("feature_names", [])),
        "window": int(meta.get("window", cfg.sample.window)),
        "env_cfg": cfg.to_dict()["env"],
        "train_cfg": cfg.to_dict()["train"],
        "created_at": run_id,
    }
    write_json(sidecar, Path(cfg.train.out_dir) / f"{model_name}.json")

    # ---- 训练曲线 ----
    reward_hist = list(getattr(trainer, "reward_history", []))
    loss_hist = list(getattr(trainer, "loss_history", []))
    curve_path = run_dir / "curve.png"
    best = max(reward_hist) if reward_hist else 0.0
    try:
        from stock_ai.eval.plot import EquityPlotter

        EquityPlotter().plot_training_curve(
            {"mean_reward": reward_hist, "loss": loss_hist},
            curve_path,
            title=f"train curve ({algo}/{backend})",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("训练曲线绘制失败（不阻塞）: %s", exc)

    # ---- 训练结果 json ----
    result = {
        "run_id": run_id,
        "backend": backend,
        "algo": algo,
        "timesteps": int(cfg.train.timesteps),
        "best_mean_reward": float(best),
        "elapsed_sec": float(elapsed),
        "model_path": str(model_path),
        "data_hash": str(meta.get("data_hash", "")),
    }
    write_json(result, run_dir / "train_result.json")

    logger.info("训练完成: 模型=%s 用时=%.1fs", model_path, elapsed)
    print(f"[train] 完成 backend={backend} algo={algo}")
    print(f"  模型: {model_path}")
    print(f"  sidecar: {Path(cfg.train.out_dir) / (model_name + '.json')}")
    print(f"  曲线: {curve_path}")
    print(f"  用时: {elapsed:.1f}s, best_mean_reward={best:.4f}")
    print(f"  run_dir: {run_dir}")
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

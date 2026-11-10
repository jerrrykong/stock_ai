"""``train`` 子命令。"""

from __future__ import annotations

import click

from stock_ai.config.loader import ConfigLoader
from stock_ai.train.train_cli import run as train_run
from stock_ai.utils.errors import StockAIError

__all__ = ["train"]


@click.command("train")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--algo", default=None, help="ppo|sac|dqn")
@click.option("--samples", default=None, help="train.npz 路径或目录")
@click.option("--timesteps", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--backend", default=None, help="auto|sb3|numpy")
@click.option("--reward", default=None, help="gross|drawdown_penalty|sharpe")
@click.option("--out", default=None, help="模型输出目录")
@click.option("--n-steps", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--n-epochs", type=int, default=None)
@click.option("--learning-rate", type=float, default=None)
def train(config, algo, samples, timesteps, seed, backend, reward, out,
          n_steps, batch_size, n_epochs, learning_rate) -> None:
    """训练模型并保存 ``models/<algo>_<ts>.zip``。"""
    import argparse

    args = argparse.Namespace(
        config=config, algo=algo, samples=samples, timesteps=timesteps, seed=seed,
        backend=backend, reward=reward, out=out, n_steps=n_steps,
        batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
    )
    code = train_run(args)
    if code != 0:
        raise StockAIError(f"训练失败，退出码 {code}")

"""``backtest`` 子命令。"""

from __future__ import annotations

import click

from stock_ai.eval.backtest_cli import run as backtest_run

__all__ = ["backtest"]


@click.command("backtest")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--model", required=True, help="模型 zip 路径")
@click.option("--samples", default=None, help="test.npz 路径或目录")
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--cash", type=float, default=None, help="初始资金")
@click.option("--fee", type=float, default=None, help="佣金率")
@click.option("--slippage", type=float, default=None, help="滑点率")
@click.option("--out", default=None, help="报告输出目录")
@click.option("--seed", type=int, default=None)
def backtest(config, model, samples, start, end, cash, fee, slippage, out, seed) -> None:
    """加载模型回测，输出指标 + 权益曲线。"""
    import argparse

    args = argparse.Namespace(
        config=config, model=model, samples=samples, start=start, end=end,
        cash=cash, fee=fee, slippage=slippage, out=out, seed=seed,
    )
    backtest_run(args)

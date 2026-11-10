"""``predict`` 子命令。"""

from __future__ import annotations

import click

from stock_ai.infer.predict_cli import run as predict_run

__all__ = ["predict"]


@click.command("predict")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--model", required=True, help="模型 zip 路径")
@click.option("--symbols", multiple=True, help="股票代码，可多次指定")
@click.option("--start", required=True, help="起始 YYYYMMDD")
@click.option("--end", required=True, help="结束 YYYYMMDD")
@click.option("--meta", default=None, help="samples/meta.json 路径")
@click.option("--out", default=None, help="输出目录")
@click.option("--seed", type=int, default=None)
def predict(config, model, symbols, start, end, meta, out, seed) -> None:
    """对未知区间预测每日动作建议（CSV + JSON）。"""
    import argparse

    args = argparse.Namespace(
        config=config, model=model, symbols=list(symbols) if symbols else None,
        start=start, end=end, meta=meta, out=out, seed=seed,
    )
    predict_run(args)

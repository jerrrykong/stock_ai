"""``sample build`` 子命令。"""

from __future__ import annotations

import click

from stock_ai.config.loader import ConfigLoader
from stock_ai.data.loader import StockDataLoader
from stock_ai.samples.builder import SampleBuilder
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["sample"]


@click.group()
def sample() -> None:
    """样本：构建 walk-forward 训练/验证/测试样本。"""


@sample.command("build")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--symbols", multiple=True)
@click.option("--train-start", default=None)
@click.option("--train-end", default=None)
@click.option("--test-start", default=None)
@click.option("--test-end", default=None)
@click.option("--features", multiple=True, help="特征名，可多次指定")
@click.option("--window", type=int, default=None)
@click.option("--embargo-days", type=int, default=None)
@click.option("--out", default=None, help="样本输出目录")
@click.option("--seed", type=int, default=None)
def sample_build(config, symbols, train_start, train_end, test_start, test_end,
                 features, window, embargo_days, out, seed) -> None:
    """构建样本并落盘 ``data/samples/{train,val,test}.npz`` + ``meta.json``。"""
    cfg = ConfigLoader.load(config)
    overrides: dict = {}
    if symbols:
        overrides["data.symbols"] = list(symbols)
    if train_start:
        overrides["sample.train_start"] = train_start
    if train_end:
        overrides["sample.train_end"] = train_end
    if test_start:
        overrides["sample.test_start"] = test_start
    if test_end:
        overrides["sample.test_end"] = test_end
    if features:
        overrides["sample.features"] = list(features)
    if window is not None:
        overrides["sample.window"] = int(window)
    if embargo_days is not None:
        overrides["sample.embargo_days"] = int(embargo_days)
    if out:
        overrides["paths.samples"] = out
    if seed is not None:
        overrides["random_seed"] = seed
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()

    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup("stock_ai.samples", cfg.paths.logs, cfg.log_level)

    loader = StockDataLoader(cfg.data, logger)
    builder = SampleBuilder(cfg.sample, logger)
    bundle = builder.build_all(loader, symbols=list(symbols) if symbols else None)
    paths = builder.save(bundle, cfg.paths.samples, seed=cfg.random_seed)

    click.echo(f"[sample build] 输出目录: {cfg.paths.samples}")
    click.echo(f"  train: {bundle.train.n_steps} 步")
    click.echo(f"  val  : {bundle.val.n_steps} 步")
    click.echo(f"  test : {bundle.test.n_steps} 步")
    click.echo(f"  features: {len(bundle.feature_names)} 维, window={bundle.window}")
    click.echo(f"  data_hash: {bundle.data_hash}")
    click.echo(f"  meta: {paths['meta']}")

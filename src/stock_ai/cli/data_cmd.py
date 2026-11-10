"""``data fetch|refresh|info`` 子命令。"""

from __future__ import annotations

import click

from stock_ai.config.loader import ConfigLoader
from stock_ai.data.loader import StockDataLoader
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["data"]


def _build_loader(config: str, overrides: dict) -> "object":
    """按配置构建 loader（公共逻辑）。"""
    cfg = ConfigLoader.load(config)
    cfg.merge_cli_overrides(overrides)
    cfg.ensure_dirs()
    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup("stock_ai.data", cfg.paths.logs, cfg.log_level)
    return cfg, StockDataLoader(cfg.data, logger)


@click.group()
def data() -> None:
    """数据：拉取 / 增量刷新 / 概览。"""


@data.command("fetch")
@click.option("--config", "config", default="configs/default.yaml", help="配置文件")
@click.option("--symbols", multiple=True, help="股票代码，可多次指定")
@click.option("--start", default=None, help="起始日期 YYYYMMDD")
@click.option("--end", default=None, help="结束日期 YYYYMMDD")
@click.option("--source", default=None, help="akshare|tencent|eastmoney|sina|synthetic")
@click.option("--adjust", default=None, help="复权方式 qfq|hfq|none（默认 none 不复权；不传则读配置）")
@click.option("--cache-dir", default=None, help="缓存目录")
@click.option("--force", is_flag=True, help="忽略缓存强制重拉")
@click.option("--seed", type=int, default=None)
def data_fetch(config, symbols, start, end, source, adjust, cache_dir, force, seed) -> None:
    """全量拉取日线并落盘 ``data/raw/*.parquet``。"""
    overrides: dict = {}
    if symbols:
        overrides["data.symbols"] = list(symbols)
    if start:
        overrides["data.start"] = start
    if end:
        overrides["data.end"] = end
    if source:
        overrides["data.source"] = source
    if adjust:
        overrides["data.adjust"] = adjust
    if cache_dir:
        overrides["data.cache_dir"] = cache_dir
    if seed is not None:
        overrides["random_seed"] = seed

    cfg, loader = _build_loader(config, overrides)
    frames = loader.fetch_all(start=start, end=end, force=force)
    click.echo(f"[data fetch] 完成 {len(frames)} 只股票 -> {cfg.data.cache_dir} (adjust={cfg.data.adjust})")
    for sym, df in frames.items():
        src = loader._symbol_sources.get(sym) or "cached"
        click.echo(f"  {sym}: {len(df)} 行 [{src}]")


@data.command("refresh")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--symbols", multiple=True)
@click.option("--adjust", default=None, help="复权方式 qfq|hfq|none（默认 none 不复权；不传则读配置）")
@click.option("--cache-dir", default=None)
@click.option("--seed", type=int, default=None)
def data_refresh(config, symbols, adjust, cache_dir, seed) -> None:
    """增量刷新缓存（只拉缺失的最新日期）；复权口径变更时自动全量重拉并保留既有区间。"""
    overrides: dict = {}
    if symbols:
        overrides["data.symbols"] = list(symbols)
    if adjust:
        overrides["data.adjust"] = adjust
    if cache_dir:
        overrides["data.cache_dir"] = cache_dir
    if seed is not None:
        overrides["random_seed"] = seed

    cfg, loader = _build_loader(config, overrides)
    frames = loader.refresh_all(list(symbols) if symbols else None)
    click.echo(f"[data refresh] 完成 {len(frames)} 只股票 (adjust={cfg.data.adjust})")
    for sym, df in frames.items():
        click.echo(f"  {sym}: {len(df)} 行")


@data.command("fetch-hfq")
@click.option("--config", "config", default="configs/default.yaml", help="配置文件")
@click.option("--symbols", multiple=True, help="股票代码，可多次指定")
@click.option("--start", default=None, help="起始日期 YYYYMMDD")
@click.option("--end", default=None, help="结束日期 YYYYMMDD")
@click.option("--source", multiple=True, help="hfq 可用源 akshare|eastmoney|tencent（可多次指定；默认 akshare,eastmoney）")
@click.option("--throttle", type=float, default=3.0, help="每只间隔秒（默认 3.0，防 WAF）")
@click.option("--batch", type=int, default=30, help="每拉多少只批间休息")
@click.option("--rest", type=float, default=90.0, help="批间休息秒")
@click.option("--max-fail-streak", type=int, default=8, help="连续失败多少只判为风控")
@click.option("--cooldown", type=float, default=600.0, help="风控冷却秒")
@click.option("--force", is_flag=True, help="忽略已存在缓存重拉")
@click.option("--max-bad-ratio", type=float, default=0.0,
              help="允许剔除的源端异常行占比上限（默认 0=严格失败不剔除）")
@click.option("--strict-consistency", is_flag=True,
              help="复权有效性校验不通过时直接判失败（默认仅标记隔离）")
def data_fetch_hfq(config, symbols, start, end, source, throttle, batch, rest,
                   max_fail_streak, cooldown, force, max_bad_ratio,
                   strict_consistency) -> None:
    """拉取**后复权（hfq）**日线到 ``data/raw_hfq``（独立目录，绝不覆盖 data/raw）。

    真实成交仍使用 ``data/raw`` 的不复权价；hfq 只用于特征/训练信号。
    进度随时可用 ``stock_ai data hfq-status`` 查询。
    """
    import json

    from stock_ai.data.hfq import fetch_hfq_all
    from stock_ai.scale.symbols import load_symbols

    cfg = ConfigLoader.load(config)
    cfg.ensure_dirs()
    if symbols:
        syms = [str(s) for s in symbols]
    elif cfg.data.symbols:
        syms = [str(s) for s in cfg.data.symbols]
    else:
        syms = load_symbols(cfg.scale.symbols_file)
    sources = [str(s).strip().lower() for s in source] or None
    if sources:
        bad = [s for s in sources if s not in ("akshare", "eastmoney", "tencent")]
        if bad:
            raise click.BadParameter(
                f"--source 仅支持 akshare|eastmoney|tencent（sina 接口不支持复权），收到 {bad}"
            )
    if not start or not end:
        from stock_ai.scale.symbols import resolve_year_range

        year_range = resolve_year_range(cfg.scale.years)
        start = start or f"{year_range[0]}0101"
        end = end or f"{year_range[1]}1231"

    click.echo(f"[data fetch-hfq] {len(syms)} 只，区间 {start}~{end}，"
               f"源={sources or 'akshare,eastmoney'}")
    result = fetch_hfq_all(
        cfg.paths.root, syms, start, end, sources=sources, throttle=throttle,
        batch=batch, rest=rest, max_fail_streak=max_fail_streak, cooldown=cooldown,
        force=force, raw_dir=cfg.data.cache_dir, max_bad_ratio=max_bad_ratio,
        strict_consistency=strict_consistency,
    )
    click.echo(f"[data fetch-hfq] OK={len(result['ok'])} FAIL={len(result['failed'])} "
               f"复权异常={len(result.get('inconsistent', []))} -> data/raw_hfq")
    for report in result["reports"]:
        click.echo("  " + json.dumps(report.to_dict(), ensure_ascii=False))
    if result["failed"]:
        click.echo("  失败清单: data/raw_hfq/fetch_failed.txt")


@data.command("hfq-status")
@click.option("--config", "config", default="configs/default.yaml")
def data_hfq_status(config) -> None:
    """查看后复权拉取进度（读取 data/raw_hfq/fetch_progress.json）。"""
    import json

    from stock_ai.data.hfq import read_progress

    cfg = ConfigLoader.load(config)
    click.echo(json.dumps(read_progress(cfg.paths.root), ensure_ascii=False, indent=2))


@data.command("info")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--symbols", multiple=True)
@click.option("--cache-dir", default=None)
def data_info(config, symbols, cache_dir) -> None:
    """查看缓存概览（行数 / 区间 / 复权口径 / 缺口）。"""
    overrides: dict = {}
    if cache_dir:
        overrides["data.cache_dir"] = cache_dir
    _cfg, loader = _build_loader(config, overrides)
    infos = loader.info(list(symbols) if symbols else None)
    if not infos:
        click.echo("[data info] 缓存为空，请先执行 `stock_ai data fetch`")
        return
    click.echo(f"[data info] 共 {len(infos)} 只股票")
    for info in infos:
        click.echo(
            f"  {info.symbol}: {info.rows} 行 {info.start}~{info.end} "
            f"source={info.source} adjust={info.adjust or '-'} "
            f"gaps={info.missing_days} updated={info.updated_at}"
        )

"""``scale`` 子命令组：大规模训练 / 检测 / 状态 / 本机 UI（增量）。

命令族::

    python -m stock_ai scale fetch   # 拉取名单数据（复用 data 层，adjust=none）
    python -m stock_ai scale units   # 生成 units.json + split.json（可复现）
    python -m stock_ai scale train   # 长期训练任务（可中断/续跑，--only-failed）
    python -m stock_ai scale detect  # 检测编排（选模型 + 回测）
    python -m stock_ai scale status  # 打印进度（--phase train|detect）
    python -m stock_ai scale ui      # 启动本机 UI（127.0.0.1:8765）

所有异常经既有 ``cli.main.main()`` 的 ``StockAIError`` 兜底 → 友好提示 + ``exit(1)``。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click

from stock_ai.config.loader import ConfigLoader
from stock_ai.samples.builder import SampleBuilder
from stock_ai.scale.artifacts import UnitArtifacts
from stock_ai.scale.normalizer_store import NormalizerStore
from stock_ai.scale.orchestrator import ScaleOrchestrator
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.sampler import UnitSampler
from stock_ai.scale.symbols import config_hash, load_symbols, resolve_year_range
from stock_ai.scale.unit_detector import UnitDetector
from stock_ai.scale.unit_trainer import UnitTrainer
from stock_ai.scale.units import SplitPlan, UnitScanner
from stock_ai.utils.io import read_json
from stock_ai.utils.logging import LoggerFactory
from stock_ai.utils.seed import set_global_seed

__all__ = ["scale"]


# --------------------------------------------------------------------------- #
# 公共装配
# --------------------------------------------------------------------------- #
def _prepare(config: str, phase: str = "scratch", seed: int | None = None):
    """载入配置、解析路径、准备 logger，返回 ``(cfg, logger, paths, seed)``。"""
    cfg = ConfigLoader.load(config)
    if seed is not None:
        cfg.random_seed = int(seed)
    cfg.ensure_dirs()
    paths = ScalePaths(cfg.paths.root)
    paths.ensure()
    set_global_seed(cfg.random_seed)
    logger = LoggerFactory.setup(
        f"stock_ai.scale.{phase}", cfg.paths.logs, cfg.log_level, log_file=f"{phase}.log"
    )
    return cfg, logger, paths, int(cfg.random_seed)


def _make_loader(cfg, logger):
    """按配置构造数据加载器（复用既有 fallback 链）。"""
    from stock_ai.data.loader import StockDataLoader

    return StockDataLoader(cfg.data, logger)


def _scale_config_payload(cfg, year_range, symbols_source: str) -> dict:
    """组装用于 ``config_hash`` 的确定性载荷（AC3）。"""
    sc = cfg.scale
    return {
        "years": [int(year_range[0]), int(year_range[1])],
        "symbols_source": str(symbols_source),
        "ratio": {k: float(v) for k, v in dict(sc.split_ratio).items()},
        "seed": int(cfg.random_seed),
        "train": {
            "timesteps": int(sc.train.timesteps),
            "algo": str(sc.train.algo),
            "backend": str(sc.train.backend),
        },
        "detect": {
            "model_policy": str(sc.detect.model_policy),
            "mode": str(sc.detect.mode),
        },
    }


# --------------------------------------------------------------------------- #
# CLI 组
# --------------------------------------------------------------------------- #
@click.group("scale")
def scale() -> None:
    """大规模训练 / 检测编排 + 本机 UI（沪深300 × 年度单元）。"""


@scale.command("fetch")
@click.option("--config", "config", default="configs/default.yaml", help="配置文件")
@click.option("--source", default=None, help="akshare|tencent|eastmoney|sina|synthetic")
@click.option("--symbols", multiple=True, help="覆盖名单（可多次指定）")
@click.option("--force", is_flag=True, help="忽略缓存强制重拉")
@click.option("--seed", type=int, default=None)
@click.option(
    "--throttle", type=float, default=0.0,
    help="每只之间的间隔秒（默认 0；长名单建议 2~4，避免行情源窗口风控）",
)
@click.option(
    "--allow-synthetic", is_flag=True,
    help="真实源全失败时允许写 synthetic 兜底（默认禁止：合成数据会污染训练集）",
)
def scale_fetch(config, source, symbols, force, seed, throttle, allow_synthetic) -> None:
    """拉取名单数据（区间 = 最近 20 年，adjust=none，复用 data 层 fallback）。

    默认**禁用 synthetic 兜底**：真实源全部失败即记为失败并继续，绝不写入合成
    假数据（对"真实数据大规模训练"而言假数据是有害的）。逐只容错，单只失败
    不中断整批；失败清单写入 ``<cache_dir>/fetch_failed.txt`` 供下轮补拉。
    """
    cfg, logger, _paths, sd = _prepare(config, "fetch", seed)
    overrides: dict = {}
    if source:
        overrides["data.source"] = source
    if symbols:
        overrides["data.symbols"] = list(symbols)
    cfg.merge_cli_overrides(overrides)
    cfg.data.adjust = "none"

    year_range = resolve_year_range(cfg.scale.years)
    start, end = f"{year_range[0]}0101", f"{year_range[1]}1231"
    syms = list(symbols) if symbols else load_symbols(cfg.scale.symbols_file)

    loader = _make_loader(cfg, logger)
    explicit_synthetic = (cfg.data.source or "").lower() == "synthetic"
    if not allow_synthetic and not explicit_synthetic:
        # 严禁**隐式**写入合成数据：真实源全失败就记失败，稍后重试。
        # （用户显式 --source synthetic 时尊重其明确意图，不过滤。）
        loader.sources = [s for s in loader.sources if getattr(s, "name", "") != "synthetic"]
    click.echo(f"[scale fetch] 数据源链: {[getattr(s, 'name', '?') for s in loader.sources]}")
    if throttle:
        click.echo(f"[scale fetch] 节流: {float(throttle):.1f}s/只")

    frames = loader.fetch_all(
        syms, start, end, force=force, throttle=float(throttle), on_error="skip",
    )
    fails: dict = dict(getattr(loader, "last_failures", {}) or {})
    dist = Counter(loader._symbol_sources.get(s, "cached") for s in frames)

    click.echo(
        f"[scale fetch] 完成 {len(frames)}/{len(syms)} 只 -> {cfg.data.cache_dir} "
        f"(adjust={cfg.data.adjust}, {start}~{end}, seed={sd})"
    )
    click.echo(f"  来源分布: {dict(dist)}")
    if fails:
        failed_path = Path(cfg.data.cache_dir) / "fetch_failed.txt"
        failed_path.write_text("\n".join(sorted(fails)), encoding="utf-8")
        click.echo(f"  ⚠ 失败 {len(fails)} 只（未写入任何假数据）: {sorted(fails)[:10]}")
        click.echo(f"  失败清单: {failed_path}")
    if dist.get("synthetic"):
        click.echo(
            "  ⚠ 警告：本次写入了 synthetic 合成数据（仅显式 --allow-synthetic 时可能），"
            "若用于真实训练请先核实。"
        )


@scale.command("units")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--seed", type=int, default=None)
@click.option("--symbols", multiple=True, help="覆盖名单（可多次指定）")
def scale_units(config, seed, symbols) -> None:
    """生成 ``units.json`` + ``split.json``（同种子可字节复现）。"""
    cfg, logger, paths, sd = _prepare(config, "units", seed)
    builder = SampleBuilder(cfg.sample, logger)
    loader = _make_loader(cfg, logger)

    syms = list(symbols) if symbols else load_symbols(cfg.scale.symbols_file)
    year_range = resolve_year_range(cfg.scale.years)

    scanner = UnitScanner(paths, builder, loader)
    units = scanner.scan_units(syms, year_range)
    scanner.write_units(units, scanner.last_status, scanner.last_steps)

    available = [u for u in units if scanner.last_status.get(u.unit_id) == "available"]
    chash = config_hash(
        _scale_config_payload(cfg, year_range, str(cfg.scale.symbols_file))
    )
    plan = scanner.build_split(
        available, sd, dict(cfg.scale.split_ratio), chash,
        str(cfg.scale.symbols_file), year_range, len(units), len(syms),
    )
    scanner.write_split(plan)

    counts = {"available": 0, "insufficient": 0, "missing": 0}
    for st in scanner.last_status.values():
        counts[st] = counts.get(st, 0) + 1
    click.echo(
        f"[scale units] 代码 {len(syms)} × 年份 {year_range[0]}~{year_range[1]} "
        f"= 单元 {len(units)}"
    )
    click.echo(
        f"  available={counts['available']} insufficient={counts['insufficient']} "
        f"missing={counts['missing']}"
    )
    click.echo(
        f"  train={len(plan.train_units)} detect={len(plan.detect_units)} "
        f"seed={sd} ratio={dict(cfg.scale.split_ratio)}"
    )
    click.echo(f"  单元清单: {paths.units_json}")
    click.echo(f"  划分计划: {paths.split_json}")


def _load_split(paths: ScalePaths) -> SplitPlan:
    """载入 ``split.json``（缺失时给出友好提示）。"""
    from stock_ai.scale.errors import ScaleError

    if not paths.split_json.exists():
        raise ScaleError(
            "缺少 split.json：请先执行 `python -m stock_ai scale units`",
            context={"path": str(paths.split_json)},
        )
    return SplitPlan.from_dict(read_json(paths.split_json))


@scale.command("train")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--seed", type=int, default=None)
@click.option("--only-failed", is_flag=True, help="仅重跑 failed 单元")
@click.option("--timesteps", type=int, default=None, help="覆盖 scale.train.timesteps")
@click.option("--algo", default=None, help="覆盖 scale.train.algo (ppo|sac|dqn)")
@click.option("--backend", default=None, help="覆盖 scale.train.backend (auto|sb3|numpy)")
def scale_train(config, seed, only_failed, timesteps, algo, backend) -> None:
    """长期训练任务：逐单元训练，可中断（Ctrl+C）/续跑（跳过 done）。"""
    cfg, logger, paths, sd = _prepare(config, "train", seed)
    if timesteps is not None:
        cfg.scale.train.timesteps = int(timesteps)
    if algo:
        cfg.scale.train.algo = str(algo)
    if backend:
        cfg.scale.train.backend = str(backend)

    builder = SampleBuilder(cfg.sample, logger)
    sampler = UnitSampler(builder)
    runner = UnitTrainer(
        cfg, paths, sampler, NormalizerStore(paths), UnitArtifacts(paths), sd
    )
    orch = ScaleOrchestrator(
        cfg, "train", runner, paths, logger, sd, only_failed=only_failed
    )
    code = orch.run()
    if code != 0:
        sys.exit(code)


@scale.command("detect")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--seed", type=int, default=None)
@click.option("--model-policy", default=None,
              help="nearest_past|nearest|max_year|none（覆盖配置）")
def scale_detect(config, seed, model_policy) -> None:
    """检测编排：为每个检测单元选模型 + 回测，产出 ``reports/units/<unit>.json``。"""
    cfg, logger, paths, sd = _prepare(config, "detect", seed)
    if model_policy:
        cfg.scale.detect.model_policy = str(model_policy)

    # 提前校验模型策略：非法值直接报错（不静默回退到 nearest_past）
    from stock_ai.scale.errors import ScaleError
    from stock_ai.scale.model_policy import SUPPORTED_POLICIES

    pol = str(cfg.scale.detect.model_policy or "nearest_past").lower()
    if pol not in SUPPORTED_POLICIES:
        raise ScaleError(
            f"未知的检测模型策略: {cfg.scale.detect.model_policy!r}；"
            f"支持的策略为 {sorted(SUPPORTED_POLICIES)}",
            context={"policy": str(cfg.scale.detect.model_policy)},
        )

    plan = _load_split(paths)
    builder = SampleBuilder(cfg.sample, logger)
    sampler = UnitSampler(builder)
    runner = UnitDetector(
        cfg, paths, sampler, NormalizerStore(paths), UnitArtifacts(paths), plan, sd
    )
    orch = ScaleOrchestrator(cfg, "detect", runner, paths, logger, sd)
    code = orch.run()
    if code != 0:
        sys.exit(code)


@scale.command("status")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--phase", type=click.Choice(["train", "detect"]), default="train")
def scale_status(config, phase) -> None:
    """打印进度：总数 / 完成 / 失败 / 剩余 / ETA。"""
    cfg, _logger, paths, _sd = _prepare(config, "status")
    from stock_ai.scale.state import StateStore

    store = StateStore(paths)
    state = store.load(phase)
    counts = state.recompute_counts()
    processed = counts.get("done", 0) + counts.get("failed", 0)
    remaining = max(0, int(state.total) - processed - counts.get("skipped", 0))
    click.echo(f"[scale status] phase={phase} process_state={state.process_state}")
    click.echo(f"  总数 total   : {state.total}")
    click.echo(f"  完成 done    : {counts.get('done', 0)}")
    click.echo(f"  失败 failed  : {counts.get('failed', 0)}")
    click.echo(f"  跳过 skipped : {counts.get('skipped', 0)}")
    click.echo(f"  剩余 remain  : {remaining}")
    click.echo(f"  当前单元     : {state.current_unit or '-'}")
    click.echo(
        f"  平均单单元   : {state.avg_unit_sec:.2f}s  ETA={state.eta_sec:.0f}s"
    )


@scale.command("unified-train")
@click.option("--config", default="configs/default.yaml", help="配置文件")
@click.option("--timesteps", type=click.IntRange(min=1), default=None, help="run 累计目标步数")
@click.option("--run-id", default=None, help="运行 ID；续训时必须指定")
@click.option("--resume", is_flag=True, help="从该 run 的 checkpoint 续训（非精确 RNG 恢复）")
@click.option("--window", type=click.IntRange(1, 512), default=None)
@click.option("--train-years", nargs=2, type=int, default=None, help="默认 2006 2021")
@click.option("--val-years", nargs=2, type=int, default=None, help="默认 2022 2023")
@click.option("--test-years", nargs=2, type=int, default=None, help="默认 2024 2025")
@click.option("--n-envs", type=click.IntRange(1, 32), default=1)
@click.option("--checkpoint-steps", type=click.IntRange(min=1), default=10000)
@click.option("--execution", type=click.Choice(["legacy", "strict", "strict-ca"]), default="legacy",
              help="legacy=同日收盘；strict=旧v1次日开盘(漏公司行为)；strict-ca=v2已核验分红送转")
@click.option("--corporate-actions-dir", default=None,
              help="strict-ca 逐股票已审阅事件账本目录（默认 data/corporate_actions；缺覆盖拒绝）")
@click.option("--feature-dir", default=None,
              help="特征缓存目录（strict 建议指向 data/raw_hfq 后复权）")
@click.option("--reward", default=None,
              help="覆盖 env.reward_type（gross|drawdown_penalty|sharpe|excess）")
@click.option("--lambda-turnover", type=float, default=None, help="覆盖 env.lambda_turnover")
@click.option("--lambda-drawdown", type=float, default=None, help="覆盖 env.lambda_drawdown")
@click.option("--lambda-leverage", type=float, default=None, help="覆盖 env.lambda_leverage")
@click.option("--seed", type=int, default=None)
def unified_train(config, timesteps, run_id, resume, window, train_years, val_years,
                  test_years, n_envs, checkpoint_steps, execution, feature_dir, reward,
                  lambda_turnover, lambda_drawdown, lambda_leverage, seed, corporate_actions_dir):
    """所有缓存股票按时间切分，用一个共享 PPO 训练；旧单元模式不变。"""
    from stock_ai.scale.unified import (
        DEFAULT_RANGES, EXECUTION_LEGACY, EXECUTION_STRICT, EXECUTION_CORPORATE_ACTIONS, UnifiedRunner,
    )
    from stock_ai.env.rewards import SUPPORTED_REWARDS
    from stock_ai.scale.errors import ScaleError

    import json
    import os
    from stock_ai.config.schema import AppConfig

    snapshot = os.environ.get("STOCKAI_UNIFIED_CONFIG")
    cfg = AppConfig.from_dict(json.loads(snapshot)) if snapshot else ConfigLoader.load(config)
    if seed is not None:
        cfg.random_seed = seed

    # ---- 奖励配置覆盖（仅当用户显式给出时才改写；缺省路径字节级不变）----
    # 1) --reward：校验白名单后覆盖 reward_type。
    if reward is not None:
        reward_value = str(reward).strip().lower()
        if reward_value not in SUPPORTED_REWARDS:
            raise ScaleError(
                f"不支持的奖励类型: {reward!r}（可选: {sorted(SUPPORTED_REWARDS)}）",
                context={"reward": str(reward), "valid": sorted(SUPPORTED_REWARDS)},
            )
        cfg.env.reward_type = reward_value
    # 2) excess 预设：**仅当用户显式给了 --reward 且其解析为 excess 时**才应用，
    #    对仍为 None 的 λ 补齐预设（lambda_leverage 也纳入，见下）。
    #    若 YAML 里直接写了 reward_type: excess 而未带 --reward，则尊重 YAML 的 λ。
    if reward is not None and str(cfg.env.reward_type).lower() == "excess":
        if lambda_turnover is None:
            cfg.env.lambda_turnover = 0.002
        if lambda_drawdown is None:
            cfg.env.lambda_drawdown = 0.0
        # lambda_leverage 预设为 0：本环境 long-only 且买入即用尽现金
        # （max_position=1.0），position_ratio∈[0,1]，持仓即≈1，没有任何方式
        # 表达超额杠杆。而 rewards.py 的 excess 分支**每一步**都收
        # λ_leverage·|pos|，于是这一项退化成「持仓时长税」而非杠杆风险惩罚：
        # 满仓年成本 = 0.01 × 约 231 步 ≈ 2.28 奖励点，而年基准波幅仅约 0.41
        # （5.6 倍），使「永不买入」成为全局最优（v1 A/B：λ=0.01 时 30% episode
        # 零交易，λ=0 时 0.2%）。换手惩罚按次收、口径正确，保留 0.002。
        if lambda_leverage is None:
            cfg.env.lambda_leverage = 0.0
    # 3) 用户显式给出的 λ 优先于预设。
    if lambda_turnover is not None:
        cfg.env.lambda_turnover = float(lambda_turnover)
    if lambda_drawdown is not None:
        cfg.env.lambda_drawdown = float(lambda_drawdown)
    if lambda_leverage is not None:
        cfg.env.lambda_leverage = float(lambda_leverage)

    supplied = {"train": train_years, "val": val_years, "test": test_years}
    ranges = ({k: list(v or DEFAULT_RANGES[k]) for k, v in supplied.items()}
              if any(supplied.values()) else None)
    version = {"strict": EXECUTION_STRICT, "strict-ca": EXECUTION_CORPORATE_ACTIONS,
               "legacy": EXECUTION_LEGACY}[execution]
    # A CLI default is not an explicit request to change a frozen resume version.
    if resume:
        from click.core import ParameterSource
        ctx = click.get_current_context()
        if ctx.get_parameter_source("execution") == ParameterSource.DEFAULT:
            version = None
    click.echo(f"[unified] 执行语义={version or '继承冻结 manifest'}"
               + (f"；特征目录={feature_dir}" if feature_dir else "")
               + "；进度见 /api/unified/status 或 scale unified status")
    directory = UnifiedRunner(cfg).run(timesteps, run_id, resume, window, ranges, n_envs,
                                       checkpoint_steps, execution_version=version,
                                       feature_dir=feature_dir,
                                       corporate_actions_dir=corporate_actions_dir)
    click.echo(f"[unified] 产物: {directory}")


@scale.group("unified")
def unified():
    """统一模型训练、状态及控制。"""


unified.add_command(unified_train, "train")


@unified.command("status")
@click.option("--config", default="configs/default.yaml")
def unified_status(config):
    import json
    from stock_ai.scale.unified import read_status

    cfg = ConfigLoader.load(config)
    click.echo(json.dumps(read_status(cfg.paths.root), ensure_ascii=False, indent=2))


@unified.command("control")
@click.argument("action", type=click.Choice(["pause", "resume", "stop"]))
@click.option("--config", default="configs/default.yaml")
def unified_control(action, config):
    from stock_ai.scale.unified import write_control

    cfg = ConfigLoader.load(config)
    write_control(cfg.paths.root, action)
    click.echo(f"[unified] 已请求 {action}")


@scale.group("corporate-actions")
def corporate_actions():
    """已核验公司行为账本的只读检查与显式导入；不自动猜事件。"""


@corporate_actions.command("fetch-snapshot")
@click.option("--symbol", required=True, help="六位股票代码")
@click.option("--config", default="configs/default.yaml")
@click.option("--output-dir", default=None, help="默认 outputs/corporate_action_sources；不写执行账本")
def corporate_actions_fetch_snapshot(symbol, config, output_dir):
    """归档巨潮原响应和未审阅候选，不生成 verified 覆盖或启动训练。"""
    from stock_ai.data.corporate_action_sources import fetch_cninfo_snapshot
    from stock_ai.data.corporate_actions import CorporateActionError
    from stock_ai.scale.errors import ScaleError
    import json
    cfg = ConfigLoader.load(config)
    target = Path(output_dir) if output_dir else Path(cfg.paths.root) / "outputs" / "corporate_action_sources"
    try:
        snapshot = fetch_cninfo_snapshot(symbol, target)
    except CorporateActionError as exc:
        raise ScaleError(str(exc)) from exc
    fields = ("symbol", "status", "count", "total", "fetched_at", "response_sha256",
              "raw_response_path", "metadata_path")
    click.echo(json.dumps({key: snapshot[key] for key in fields}, ensure_ascii=False, indent=2))
    click.echo("仅归档未审阅快照；计数匹配不证明历史完整，不能直接用于 v2 训练或回放。")


@corporate_actions.command("validate")
@click.argument("ledger", type=click.Path(exists=True, dir_okay=False))
@click.option("--start", required=True, help="覆盖区间 YYYYMMDD")
@click.option("--end", required=True, help="覆盖区间 YYYYMMDD")
def corporate_actions_validate(ledger, start, end):
    from stock_ai.data.corporate_actions import load_ledger, CorporateActionError
    from stock_ai.scale.errors import ScaleError
    import json
    try:
        data = load_ledger(ledger)
        data.require_coverage(start, end)
    except CorporateActionError as exc:
        raise ScaleError(str(exc)) from exc
    click.echo(json.dumps({"ok": True, "symbol": data.symbol, "schema": data.schema,
                           "coverage": data.coverage, "events": len(data.events),
                           "sha256": data.sha256, "read_only": True}, ensure_ascii=False, indent=2))


@corporate_actions.command("import")
@click.argument("ledger", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", default="configs/default.yaml")
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--output-dir", default=None, help="默认项目 data/corporate_actions")
def corporate_actions_import(ledger, config, start, end, output_dir):
    """导入显式审阅账本，目标已存在时拒绝覆盖。"""
    from stock_ai.data.corporate_actions import load_ledger, CorporateActionError
    from stock_ai.scale.errors import ScaleError
    import json
    try:
        data = load_ledger(ledger)
        data.require_coverage(start, end)
    except CorporateActionError as exc:
        raise ScaleError(str(exc)) from exc
    cfg = ConfigLoader.load(config)
    target_dir = Path(output_dir) if output_dir else Path(cfg.paths.root) / "data" / "corporate_actions"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{data.symbol}.json"
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(data.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ScaleError(f"公司行为账本已存在，拒绝覆盖：{target}") from exc
    click.echo(f"[corporate-actions] 已导入 {data.symbol} -> {target}；仅声明覆盖区间可用，未训练")


@scale.command("ui")
@click.option("--config", "config", default="configs/default.yaml")
@click.option("--port", type=int, default=8765, help="监听端口（默认 8765）")
@click.option("--no-open", is_flag=True, help="不自动打开浏览器")
def scale_ui(config, port, no_open) -> None:
    """启动本机 UI（仅监听 127.0.0.1，无登录，无自动重启服务）。"""
    cfg, logger, paths, _sd = _prepare(config, "ui")
    from stock_ai.scale.ui.server import ScaleUI

    ui = ScaleUI(cfg, paths, logger=logger, host="127.0.0.1", port=int(port))
    ui.serve(open_browser=not no_open)

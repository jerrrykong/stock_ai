"""可复用逐族增量验证台架（T01 · 只搭台架，不跑回测 / 不训练 / 不写盘）。

本模块把 plan §7.1 的「唯一度量口径」固化为可复用函数，供
T02（复现两个对照基线）与 T03（P0-1~P0-4 逐族单变量实验）直接调用：

  * :func:`build_panel`        逐位复用 ``diagnose_residual_overfit_vs_regime.py:39-58``
  * :func:`fit_logistic`       照抄 ``:94-106``（Adam、clip ±30）
  * :func:`fit_and_extrapolate` 复刻 ``:203-229``（70/30 时间外推）
  * :func:`self_check`         调用 ``TimingBacktest.self_check()``
  * :func:`evaluate`           主判据 + 辅判据 + 8 条止损 → :class:`Verdict`

冻结参数全部来自 ``outputs/feature_experiments.yaml``（§7.3），**只变 groups**。
主判据 = late 评估段 ``gross_excess > 0`` 且 ``> 现状基线 +0.85%``（Q9 已确认）；
摩擦 0.3% / ``rebalance=20``（Q10 已确认）。
IC / 监督学习**只作诊断字段，绝不参与 evaluate 判定**。

Q3/Q4 已确认「不落盘派生目录 + 契约升级 + 新 run_id」：本台架与后续实验编排
**不提供任何 ``--resume`` 参数**，绝不对 v3/v4 旧 run 续训。

TODO(T02)：``reproduce_controls()``、``baselines.json`` / ``self_check.json``。
TODO(T03)：结果写盘（``comparison.csv`` / ``yearly_decomp.csv`` /
``metrics.json`` / ``verdict.json``）与逐族实验编排。
本轮 T01 这些入口仅作清晰占位，**不实现写盘、不创建结果目录、不跑实验**。

模块导入无任何副作用：不读数据、不写文件、不跑回测、不训练。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs"))

from feature_families import (  # noqa: E402
    RAW, RAW_HFQ_V2_TRAIN, build_family_frame, contract_id, feature_set_id,
    load_config, validate_single_variable, active_features,
)
from timing_backtest import BacktestResult, TimingBacktest  # noqa: E402

__all__ = [
    "ROOT", "FEATURE_DIR", "SPLITS", "HORIZON", "BacktestResult", "Verdict",
    "StopRule", "STOP_RULES", "REQUIRED_EVALUATION_METRICS",
    "load_config", "build_panel", "fit_logistic",
    "fit_and_extrapolate", "self_check", "evaluate",
    "validate_single_variable", "active_features", "feature_set_id",
    "contract_id", "reproduce_controls", "write_results", "run_experiments",
]

FEATURE_DIR: Path = RAW_HFQ_V2_TRAIN
SPLITS: dict[str, tuple[int, int]] = {
    "train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025),
}
HORIZON: int = 20


# --------------------------------------------------------------------------- #
# 面板构造（逐位复用 diagnose_residual_overfit_vs_regime.py:39-58）
# --------------------------------------------------------------------------- #
def build_panel(groups: tuple[str, ...] | list[str],
                cfg: dict | None = None) -> pd.DataFrame:
    """构造 197 只训练池面板，列集合 = ``active_features(groups)``。

    与参考实现 :39-58 逐位一致的口径：
      * 从 ``data/raw_hfq_v2_train`` 读全部 ``*.parquet``；
      * 用 **hfq ``close``** 算 ``ret`` / ``fwd = close.shift(-20)/close - 1``；
      * ``year = date.dt.year``、``symbol = path.stem``；
      * 按 ``["symbol", "date"]`` 排序并 ``reset_index``。
    差异（本台架契约所需）：base 列来自 ``unified.relative_features``，
    新族列来自 ``feature_families.build_family_frame``；额外保留 ``_row``
    全局索引供 ``TimingBacktest.run_probability`` 对齐（``timing_backtest.py:153-177``）。

    Args:
        groups: 特征族，例如 ``("base", "liquidity")``；由 ``validate_single_variable`` 校验。
        cfg: 冻结配置；``None`` 时读 yaml。

    Returns:
        含 ``date, symbol, close, year, ret, fwd, _row, *active_features(groups)`` 的面板。
    """
    config = cfg if cfg is not None else load_config()
    validate_single_variable(groups)
    cols = active_features(groups)
    horizon = int(config["frozen"]["horizon"])
    if horizon != HORIZON:
        raise ValueError(
            f"冻结 horizon 不一致：yaml={horizon}，台架常量={HORIZON}；"
            "任一项变动即中止实验（plan §7.3）"
        )

    frames: list[pd.DataFrame] = []
    for path in sorted(FEATURE_DIR.glob("*.parquet")):
        family = build_family_frame(path.stem, groups,
                                    amount_dir=RAW_HFQ_V2_TRAIN, raw_dir=RAW)
        frames.append(family[["date", "close", "symbol", *cols]].copy())
    if not frames:
        raise ValueError(f"{FEATURE_DIR} 下无 parquet；fail-closed，不进入回测")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # 与参考实现完全一致：hfq close、groupby(symbol) 的 pct_change / shift(-HORIZON)
    panel["ret"] = panel.groupby("symbol")["close"].pct_change()
    panel["fwd"] = (panel.groupby("symbol")["close"].shift(-horizon)
                    / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    # 全局行号：run_probability 的全局概率数组按 _row 索引（timing_backtest.py:163-175）
    panel["_row"] = np.arange(len(panel))
    return panel


# --------------------------------------------------------------------------- #
# 逻辑回归（照抄 diagnose_residual_overfit_vs_regime.py:94-106）
# --------------------------------------------------------------------------- #
def fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 250,
                 lr: float = 0.05, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """Adam 批量逻辑回归，与参考实现 :94-106 逐行一致（含 clip ±30）。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    mw = np.zeros(d)
    vw = np.zeros(d)
    mb = 0.0
    vb = 0.0
    for t in range(1, epochs + 1):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * grad_w ** 2
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * grad_b ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return w, b


# --------------------------------------------------------------------------- #
# 70/30 时间外推（复刻 diagnose_residual_overfit_vs_regime.py:203-229）
# --------------------------------------------------------------------------- #
def fit_and_extrapolate(panel: pd.DataFrame, cols: list[str] | tuple[str, ...],
                        cfg: dict | None = None) -> BacktestResult:
    """拟合 early(2006-2016) → 评估 late(2017-2021)，返回 :class:`BacktestResult`。

    逐位复刻参考实现 :203-229：
      ``data = panel[fwd.notna() & |fwd|<1.0]`` → ``_row = arange(len(data))``；
      ``mean/std`` 只用 early、``std<=0→1``、``Z=clip((X-mean)/std,-10,10)``；
      ``prob = sigmoid(Zw+b)``（**全局数组，长度 = len(data)**）；
      ``sub = data[late_mask]``；
      ``TimingBacktest(sub, rebalance=20, global_len=len(data))``；
      ``run_probability(prob)`` —— 全局 ``prob`` 通过子集保留的 ``_row``
      全局行号索引，绝不错位（``timing_backtest.py:153-177``）。

    Args:
        panel: :func:`build_panel` 输出。
        cols: 参与拟合的特征列名（须为 ``active_features(groups)``）。
        cfg: 冻结配置；``None`` 时读 yaml。

    Returns:
        late 评估段的 :class:`BacktestResult`。
    """
    config = cfg if cfg is not None else load_config()
    ext = config["extrapolation"]
    frozen = config["frozen"]

    # 冻结切分自检：cut = int(16*0.7) = 11 → 2006-2016 / 2017-2021
    train_years = list(range(int(ext["train_year_start"]),
                             int(ext["train_year_end"]) + 1))
    cut = int(len(train_years) * float(ext["ratio"]))
    if cut != int(ext["cut"]):
        raise ValueError(f"冻结切分不一致：计算 cut={cut}，yaml cut={ext['cut']}")
    early = list(ext["early_fit_years"])
    late = list(ext["late_eval_years"])
    if early != train_years[:cut] or late != train_years[cut:]:
        raise ValueError("冻结切分不一致：early/late 年份与 cut 推导结果不符")

    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < float(ext["fwd_abs_max"])].reset_index(drop=True)
    # 过滤后重设全局行号：prob 是全局数组（长度 len(data)），按 _row 索引
    data["_row"] = np.arange(len(data))
    y = (data["fwd"].to_numpy(dtype="float64") > 0).astype(float)

    early_mask = data["year"].isin(early).to_numpy()
    late_mask = data["year"].isin(late).to_numpy()
    if not early_mask.any() or not late_mask.any():
        raise ValueError("early / late 评估段为空，无法时间外推")

    X = data[list(cols)].to_numpy(dtype="float64")
    mean = X[early_mask].mean(0)
    std = X[early_mask].std(0)
    std[std <= 0] = 1.0
    clip = float(frozen["clip"])
    Z = np.clip((X - mean) / std, -clip, clip)
    logreg = frozen["logreg"]
    w, b = fit_logistic(Z[early_mask], y[early_mask],
                        epochs=int(logreg["epochs"]),
                        lr=float(logreg["lr"]), l2=float(logreg["l2"]))
    prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))

    sub = data[late_mask].copy()   # 含 _row（全局行号）
    backtest = TimingBacktest(
        sub, cost=float(frozen["cost"]), rebalance=int(frozen["rebalance"]),
        min_len=int(frozen["min_len"]), global_len=len(data),
    )
    return backtest.run_probability(prob, threshold=float(frozen["threshold"]))


# --------------------------------------------------------------------------- #
# 台架三项自检
# --------------------------------------------------------------------------- #
def self_check(panel: pd.DataFrame | None = None,
               cfg: dict | None = None) -> dict:
    """调用 ``TimingBacktest.self_check()``，返回含 ``all_passed`` 的结果。

    三项期望（plan §7.2 / PRD §4.1）：
      ① 常数满仓 ``|gross| < 1e-9``；
      ② 作弊信号 ``> +5%``（证明 harness 能识别真实优势）；
      ③ 随机信号 **显著 < 0**（不是 ≈0：随机择时少赚 beta 又承担复利拖累）。
    ``all_passed`` 为假时该次实验结果**不可采信**。

    Args:
        panel: 可选面板；``None`` 时用 base 面板的 train 段（调用时才读数据）。
        cfg: 冻结配置；``None`` 时读 yaml。

    Returns:
        ``TimingBacktest.self_check()`` 字典（含 ``all_passed``）。
    """
    config = cfg if cfg is not None else load_config()
    if panel is None:
        panel = build_panel(("base",), cfg=config)
    start, end = SPLITS["train"]
    frame = panel[panel["year"].between(start, end)].copy()
    frame = frame.drop(columns=[c for c in ("_row",) if c in frame.columns])
    frame = frame.reset_index(drop=True)
    frame["_row"] = np.arange(len(frame))
    frozen = config["frozen"]
    backtest = TimingBacktest(
        frame, cost=float(frozen["cost"]), rebalance=int(frozen["rebalance"]),
        min_len=int(frozen["min_len"]), global_len=len(frame),
    )
    checks = backtest.self_check()
    if "all_passed" not in checks:
        raise ValueError("TimingBacktest.self_check() 未返回 all_passed")
    return checks


# --------------------------------------------------------------------------- #
# 判定：主判据 + 8 条止损
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """逐族实验判定结果（T03 才会写 verdict.json，T01 只定义结构）。"""

    feature_set_id: str
    result: str                       # "PASS" / "STOP"
    triggered_stops: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转成可写盘的字典（本轮 T01 不写盘）。"""
        return {
            "feature_set_id": self.feature_set_id,
            "result": self.result,
            "triggered_stops": list(self.triggered_stops),
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class StopRule:
    """一条止损规则：``predicate(ctx)`` 为真即命中。"""

    code: str
    label: str
    predicate: Callable[[dict], bool]


def _yearly_concentration(ctx: dict) -> float:
    """正超额中「最大单一年份」的占比；用于 ③ 过拟合判定。"""
    yearly: dict = ctx["yearly_gross"]
    if not yearly:
        return 0.0
    positives = [float(v) for v in yearly.values() if float(v) > 0.0]
    total = float(sum(positives))
    if total <= 0.0:
        return 0.0
    return max(positives) / total


#: 8 条止损（PRD §4.4 / plan §7.2）。逐条判定，命中即 ``result="STOP"``。
#: **IC / 监督学习字段不参与任何 predicate**（只进 diagnostics）。
STOP_RULES: tuple[StopRule, ...] = (
    StopRule(
        "S1", "① 时间外推 gross_excess ≤ 0 或 ≤ 现状基线 +0.85%",
        lambda c: (c["gross"] <= c["gross_min"]
                   or c["gross"] <= c["base_threshold"]),
    ),
    StopRule(
        "S2", "② 毛正但换手 > 1.5×现状且净 < 0 → 拟合噪声",
        lambda c: (c["gross"] > 0.0
                   and c["base_switches"] is not None
                   and c["switches"] > c["turnover_multiple_max"]
                   * max(c["base_switches"], 0.0)
                   and c["net"] < 0.0),
    ),
    StopRule(
        "S3", "③ 逐年分解正超额集中在个别年份 → 过拟合",
        lambda c: _yearly_concentration(c) > c["yearly_top1_share_max"],
    ),
    StopRule(
        "S4", "④ 只在 val/test 单段为正、时间外推为负",
        lambda c: bool(c["val_test_only_positive"]) and c["gross"] <= 0.0,
    ),
    StopRule(
        "S5", "⑤ 择时技能项 ≤ 0 或零交易占比 > 5% → 退化",
        lambda c: (c["timing_skill"] <= c["timing_skill_min"]
                   or c["zero_trade_fraction"] > c["zero_trade_fraction_max"]),
    ),
    StopRule(
        "S6", "⑥ 即使 IC 稳定，只要毛超额不达标即无效",
        lambda c: (bool(c["ic_stable"])
                   and (c["gross"] <= c["gross_min"]
                        or c["gross"] <= c["base_threshold"])),
    ),
    StopRule(
        "S7", "⑦ 2006–2007 段数字不作「有效」依据（若正超额仅来自该段）",
        lambda c: bool(c["early_2006_2007_only"]),
    ),
    StopRule(
        "S8", "⑧ 三条数据前提断言任一不通过 → 直接止损，不进回测",
        lambda c: not bool(c["data_precondition_passed"]),
    ),
)


#: ``evaluate()`` 必需的判据输入；任一缺失即抛 ``ValueError``（fail-closed），
#: 绝不把「调用方没填」静默当作「无问题」。
#: ``friction_drag`` / ``mean_excess_return_traded`` / ``base_result`` 保持可选。
REQUIRED_EVALUATION_METRICS: tuple[str, ...] = (
    "data_precondition_passed",  # 或 data_precondition dict
    "self_check_passed",         # 或 self_check dict
    "timing_skill",              # S5 择时技能项
    "zero_trade_fraction",       # S5 零交易占比
    "yearly_gross",              # S3 逐年分解（允许空 dict，但必须显式提供）
    "val_test_only_positive",    # S4 只在 val/test 单段为正
    "ic_stable",                 # S6 IC 稳定但毛超额不达标
    "early_2006_2007_only",      # S7 2006–2007 段不作依据
)

_METRIC_ALIASES: dict[str, str] = {
    "data_precondition_passed": "data_precondition",
    "self_check_passed": "self_check",
}


def _missing_required_metrics(metrics: dict) -> list[str]:
    """返回缺失的必需判据字段；dict 形态可替代对应的 ``*_passed`` 布尔。"""
    missing: list[str] = []
    for key in REQUIRED_EVALUATION_METRICS:
        if key in metrics:
            continue
        alias = _METRIC_ALIASES.get(key)
        if alias is not None and alias in metrics:
            continue
        missing.append(key)
    return missing


def _extract_passed(metrics: dict, key: str, default: bool = False) -> bool:
    """从 metrics 提取布尔通过标志（fail-closed：缺省/不可解析一律 False）。

    同时兼容 ``TimingBacktest.self_check()`` 的 ``all_passed`` 键（权威）与
    ``PreconditionReport.to_dict()`` 的 ``passed`` 键（兼容旧形态），
    以及显式 ``*_passed`` 布尔。**任一显式 False 即判 False**（冲突时以
    False 为准），避免只认单一键造成的静默放行（QA Round 1 缺陷 1 / 2）。
    """
    flags: list[bool] = []
    if key in metrics:
        flags.append(bool(metrics[key]))
    dict_key = "data_precondition" if "precondition" in key else "self_check"
    value = metrics.get(dict_key)
    if isinstance(value, dict):
        flags.extend(bool(value[candidate])
                     for candidate in ("all_passed", "passed") if candidate in value)
    elif isinstance(value, bool):
        flags.append(value)
    if not flags:
        return bool(default)
    return all(flags)


def evaluate(result: BacktestResult, *,
             base_result: BacktestResult | None = None,
             cfg: dict | None = None,
             metrics: dict | None = None,
             feature_set_id: str = "") -> Verdict:
    """主判据 + 辅判据 + 8 条止损 → :class:`Verdict`。

    主判据（唯一决定有效性）：``gross_excess > 0`` 且 ``> +0.85%``；
    若传入 ``base_result``，门槛取「冻结 +0.85% 与同段实测基线」的较大者。
    辅判据（净超额 / 换手 / 摩擦 / 在市 / 技能项 / 零交易占比 / 数据断言 /
    台架自检 / 逐年分解）**全部写入返回 metrics**。
    IC 与监督学习指标只落入 ``metrics["diagnostics"]``，**不参与任何判定**。

    Args:
        result: late 评估段 :class:`BacktestResult`。
        base_result: 同段现状 9 特征基线结果（用于换手倍数与主判据门槛），可选。
        cfg: 冻结配置；``None`` 时读 yaml。
        metrics: 辅判据输入。**必需字段**见 :data:`REQUIRED_EVALUATION_METRICS`：
            ``data_precondition_passed`` / ``self_check_passed``（或对应的
            ``data_precondition`` / ``self_check`` dict）、``timing_skill`` /
            ``zero_trade_fraction`` / ``yearly_gross``（允许空 dict）/
            ``val_test_only_positive`` / ``ic_stable`` / ``early_2006_2007_only``。
            可选：``friction_drag`` / ``mean_excess_return_traded`` /
            ``ic`` / ``supervised`` / ``diagnostics``。
        feature_set_id: 本族标识（如 ``v2-liquidity``）。

    Returns:
        :class:`Verdict`；命中任一止损即 ``result="STOP"`` 并记录条款编号。
        ``metrics`` 只回显调用方实际提供的内容，另加
        ``data_precondition_passed`` / ``self_check_passed`` 两个显式布尔
        记录真实解析结果，**缺字段时绝不伪造** ``passed: True`` /
        ``all_passed: True``。

    Raises:
        ValueError: 任一必需字段缺失，**fail-closed**（错误信息列出缺失字段）。
            不会静默当作「无问题」放行，也不会输出一个伪造的 PASS 结论。
    """
    config = cfg if cfg is not None else load_config()
    main = config["main_criterion"]
    stop = config["stop_rules"]
    supplied = dict(metrics or {})

    # fail-closed：判据必需字段缺失 → 抛错，绝不静默当作「无问题」
    missing = _missing_required_metrics(supplied)
    if missing:
        raise ValueError(
            f"evaluate() fail-closed：判据必需字段缺失 {missing}；"
            "必须在 metrics 中显式提供（data_precondition / self_check "
            "可用对应的 *_passed 布尔替代）。绝不静默当作『无问题』通过；"
            "T03 编排须补全字段后重跑。"
        )

    gross = float(result.gross_excess)
    net = float(result.net_excess)
    switches = float(result.switches)
    exposure = float(result.exposure)
    n_episodes = int(result.n_episodes)
    base_switches = (float(base_result.switches)
                     if base_result is not None else None)
    base_gross = (float(base_result.gross_excess)
                  if base_result is not None
                  else float(config["controls"]["base"]["value"]))
    base_threshold = max(float(main["base_value"]), base_gross)

    # ctx 直接索引：必需字段已校验，不再给 S3/S4/S6/S7 的输入「无问题」默认值
    ctx: dict = {
        "gross": gross,
        "net": net,
        "switches": switches,
        "base_switches": base_switches,
        "base_threshold": base_threshold,
        "gross_min": float(main["gross_excess_min"]),
        "turnover_multiple_max": float(stop["turnover_multiple_max"]),
        "zero_trade_fraction": float(supplied["zero_trade_fraction"]),
        "zero_trade_fraction_max": float(stop["zero_trade_fraction_max"]),
        "timing_skill": float(supplied["timing_skill"]),
        "timing_skill_min": float(stop["timing_skill_min"]),
        "yearly_gross": dict(supplied["yearly_gross"]),
        "yearly_top1_share_max": float(stop["yearly_top1_share_max"]),
        "val_test_only_positive": bool(supplied["val_test_only_positive"]),
        "ic_stable": bool(supplied["ic_stable"]),
        "early_2006_2007_only": bool(supplied["early_2006_2007_only"]),
        "data_precondition_passed": _extract_passed(
            supplied, "data_precondition_passed"),
        "self_check_passed": _extract_passed(supplied, "self_check_passed"),
    }

    triggered: list[str] = []
    rule_details: list[dict] = []
    for rule in STOP_RULES:
        hit = bool(rule.predicate(ctx))
        if hit:
            triggered.append(rule.code)
        rule_details.append({"code": rule.code, "label": rule.label,
                             "triggered": hit})

    # 摩擦拖累：毛 - 净；「毛超额 < 摩擦拖累」直接判无效（PRD §4.3）
    friction_drag = float(supplied.get("friction_drag", gross - net))
    if bool(stop["friction_drag_exceeds_gross_is_invalid"]) and gross <= friction_drag:
        if "S1" not in triggered:
            triggered.append("S1")
    # 台架自检未通过 → 结果不可采信（等价于止损）
    if not ctx["self_check_passed"] and "S8" not in triggered:
        triggered.append("S8")
    triggered = sorted(set(triggered))

    main_passed = gross > ctx["gross_min"] and gross > base_threshold
    metrics_out: dict = {
        # 主判据
        "main_criterion_passed": bool(main_passed),
        "gross_excess": gross,
        "base_gross_excess": base_gross,
        "base_threshold": base_threshold,
        # 辅判据（缺一不可，全部同时报告）
        "net_excess": net,
        "switches": switches,
        "base_switches": base_switches,
        "friction_drag": friction_drag,
        "exposure": exposure,
        "n_episodes": n_episodes,
        "timing_skill": ctx["timing_skill"],
        "zero_trade_fraction": ctx["zero_trade_fraction"],
        "mean_excess_return_traded": float(
            supplied.get("mean_excess_return_traded", gross)),
        "yearly_gross": ctx["yearly_gross"],
        "yearly_top1_share": _yearly_concentration(ctx),
        # 只回显调用方实际提供的内容；未提供为 None，绝不伪造通过标志
        "data_precondition": supplied.get("data_precondition"),
        "self_check": supplied.get("self_check"),
        # 显式布尔记录两个硬 gate 的真实解析结果
        "data_precondition_passed": ctx["data_precondition_passed"],
        "self_check_passed": ctx["self_check_passed"],
        "stop_rules": rule_details,
        # 仅诊断，绝不参与判定
        "diagnostics": {
            "ic_stable": ctx["ic_stable"],
            "ic": supplied.get("ic"),
            "supervised": supplied.get("supervised"),
            "note": "IC / 监督学习仅作诊断字段，不参与 evaluate 判定（PRD §4.3）",
        },
    }
    return Verdict(
        feature_set_id=feature_set_id,
        result="STOP" if triggered else "PASS",
        triggered_stops=triggered,
        metrics=metrics_out,
    )

# --------------------------------------------------------------------------- #
# T02 / T03 预留接口（本轮 T01 明确不实现：不写盘、不跑回测、不创建结果目录）
# --------------------------------------------------------------------------- #
def reproduce_controls(cfg: dict | None = None) -> dict:
    """TODO(T02)：复现两个对照基线（base +0.85% / 阴性 −5.98%）并做容差比对。

    T02 将在 ``outputs/reproduce_controls.py`` 中实现；阴性残差列
    ``resid_cum_5 / resid_cum_20 / beta_level`` **只在该脚本局部构造**，
    绝不进 ``FEATURE_GROUPS`` / ``feature_set_id`` / 生产契约（plan §7.6）。
    本轮 T01 不跑回测、不写 ``baselines.json``。
    """
    raise NotImplementedError(
        "TODO(T02): reproduce_controls() 由 outputs/reproduce_controls.py 实现；"
        "T01 只搭台架，不跑回测、不写盘。"
    )


def write_results(feature_set_id: str = "", payload: dict | None = None) -> None:
    """TODO(T03)：写 ``comparison.csv`` / ``yearly_decomp.csv`` /
    ``metrics.json`` / ``verdict.json`` 到独立 ``feature_set_id`` 子目录。

    本轮 T01 **不实现写盘、不创建 feature_experiment_results/ 目录**。
    """
    raise NotImplementedError(
        "TODO(T03): 结果写盘由 outputs/run_feature_experiments.py 实现；"
        "T01 零落盘、不创建结果目录。"
    )


def run_experiments(groups_order: tuple[str, ...] | None = None,
                    cfg: dict | None = None) -> dict:
    """TODO(T03)：按 P0-1 → P0-2 → P0-3 → P0-4 顺序逐族单变量编排。

    每族独立 ``feature_set_id``、独立 verdict、先判止损再进下一族；
    P0-1 前先跑 ``verify_amount_precondition.py``。
    **不提供 --resume**（Q3/Q4：新增特征必须新 run，禁止对 v3/v4 续训）。
    本轮 T01 不实现编排、不跑实验。
    """
    raise NotImplementedError(
        "TODO(T03): 逐族实验编排由 outputs/run_feature_experiments.py 实现；"
        "T01 不跑回测、不训练、不产出实验结论，且不提供 --resume。"
    )

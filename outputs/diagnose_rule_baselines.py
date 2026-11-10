"""给 PPO 定及格线：在同一环境、同一成本下，简单规则策略能拿多少超额？

如果 10 行代码的规则能拿 +2% 而 PPO 是 -5%，问题是训练；如果规则也是负的，
说明信号在这个执行约束（满仓、T+1 开盘、70bps 成本）下不可用。

规则集（只用 9 个已有特征，不引入新信息）：
  buyhold        恒定持有（≈ 基准）
  cash           恒定空仓
  ma20_rev       ma20_relative < 阈值时持有，否则空仓（均值回归）
  ma5_rev        ma5_relative 同理
  mom5           momentum_5 > 0 持有
  combo          多特征 z 分加总，低于阈值空仓
  oracle_fwd     用未来 20 日收益做信号的完美择时上界（前视，仅作上界参考）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.config.schema import EnvConfig  # noqa: E402
from stock_ai.scale.unified import (  # noqa: E402
    DEFAULT_RANGES, EXECUTION_CORPORATE_ACTIONS, FEATURES, build_dataset, make_strict_env,
)

RUN = ROOT / "models" / "unified" / "20261020_105633_578f25aa"
#: 阈值用 train 段的特征分位数定，不看 val/test，避免泄漏。
QUANTILES = (0.2, 0.35, 0.5, 0.65, 0.8)


def training_cfg() -> EnvConfig:
    params = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["reward_params"]
    cfg = EnvConfig()
    cfg.reward_type = params["reward_type"]
    cfg.lambda_drawdown = float(params["lambda_drawdown"])
    cfg.lambda_turnover = float(params["lambda_turnover"])
    cfg.lambda_leverage = float(params["lambda_leverage"])
    return cfg


def feature_panel() -> dict:
    """各 symbol 的 9 特征矩阵（未归一化，与 relative_features 一致）。"""
    from stock_ai.scale.unified import relative_features
    import pandas as pd

    panel = {}
    for path in sorted((ROOT / "data" / "raw_hfq_v2_train").glob("*.parquet")):
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        df, values = relative_features(raw)
        panel[path.stem] = values
    return panel


def make_signals(values: np.ndarray, symbol: str, thresholds: dict,
                 forward: np.ndarray | None) -> dict[str, np.ndarray]:
    """对每只股票生成各规则的「目标持仓」序列（1=持有，0=空仓）。"""
    idx = {name: i for i, name in enumerate(FEATURES)}
    ma20 = values[:, idx["ma20_relative"]]
    ma5 = values[:, idx["ma5_relative"]]
    mom5 = values[:, idx["momentum_5"]]
    low = values[:, idx["low_relative"]]
    vol = values[:, idx["volatility_20"]]

    out = {}
    for q in QUANTILES:
        out[f"ma20_rev_q{q}"] = (ma20 < thresholds["ma20_relative"][q]).astype(float)
        out[f"ma5_rev_q{q}"] = (ma5 < thresholds["ma5_relative"][q]).astype(float)
    out["mom5_pos"] = (mom5 > 0).astype(float)
    out["low_pos"] = (low > 0).astype(float)
    # combo：对负向特征取反求和（负 IC 者），标准化后低于阈值空仓
    score = (-ma20 - mom5 - vol)
    for q in (0.35, 0.5, 0.65):
        out[f"combo_q{q}"] = (score > thresholds["_combo"][q]).astype(float)
    if forward is not None:
        # oracle：未来 20 日收益为正则持有（前视上界）
        out["oracle_fwd20"] = (forward > 0).astype(float)
    return out


def rollout(env, target: np.ndarray | None, model=None):
    """按目标持仓序列驱动环境。target=None 时用模型。

    模型分支必须把 step() 返回的新观测回灌给 predict —— 复用 reset() 的初始
    观测会让模型退化成恒定动作策略（曾因此得出过错误结论）。
    """
    obs, info = env.reset()
    done = False
    step = 0
    held = False
    days_in_market = 0
    while not done:
        if target is None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
        else:
            want = target[min(step, len(target) - 1)]
            if want > 0.5 and not held:
                action = 1
            elif want <= 0.5 and held:
                action = 2
            else:
                action = 0
        obs, _, terminated, _, info = env.step(action)
        held = info["shares"] > 0
        days_in_market += 1 if held else 0
        done = bool(terminated)
        step += 1
    env._days_in_market = days_in_market
    env._n_steps = step
    return env


def main() -> int:
    cfg = training_cfg()
    data = build_dataset(ROOT / "data" / "raw_hfq_v2_train", 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    from stable_baselines3 import PPO
    model = PPO.load(str(RUN / "model.zip"))

    print("计算特征面板与 train 段阈值 ...")
    panel = feature_panel()

    # train 段（2006-2021）的分位数阈值
    train_values = []
    for symbol, values in panel.items():
        dates = data.strict_pools["train"]
        # 用 symbol 的原始 parquet 日期判断 train 段
        pass
    # 直接用数据源的日期：重新读 parquet 的 date
    import pandas as pd
    train_rows = []
    for path in sorted((ROOT / "data" / "raw_hfq_v2_train").glob("*.parquet")):
        raw = pd.read_parquet(path, columns=["date"])
        mask = (raw.date.dt.year >= 2006) & (raw.date.dt.year <= 2021)
        train_rows.append(panel[path.stem][mask.to_numpy()])
    train_stack = np.concatenate(train_rows, axis=0)
    thresholds = {}
    for i, name in enumerate(FEATURES):
        thresholds[name] = {q: float(np.quantile(train_stack[:, i], q)) for q in QUANTILES}
    idx = {name: i for i, name in enumerate(FEATURES)}
    combo_score = -train_stack[:, idx["ma20_relative"]] \
        - train_stack[:, idx["momentum_5"]] - train_stack[:, idx["volatility_20"]]
    thresholds["_combo"] = {q: float(np.quantile(combo_score, q)) for q in (0.35, 0.5, 0.65)}

    # 逐 episode 计算规则结果（val / test）
    results: dict[str, dict[str, list]] = {}
    policy_names = None
    for split in ("val", "test"):
        pool = data.strict_pools[split]
        for uid, obs, bars in pool:
            symbol = uid.split("_")[0]
            values = panel.get(symbol)
            if values is None:
                continue
            # episode 在原序列中的位置：用日期对齐
            dates = [str(d) for d in bars.dates]
            all_dates = None
            raw = pd.read_parquet(ROOT / "data" / "raw_hfq_v2_train" / f"{symbol}.parquet",
                                  columns=["date", "close"])
            all_dates = raw.date.dt.strftime("%Y%m%d").to_numpy()
            pos = {d: i for i, d in enumerate(all_dates)}
            start = pos.get(dates[0])
            if start is None:
                continue
            n = len(dates)
            ep_values = values[start:start + n]
            if len(ep_values) < n:
                continue
            # oracle 需要未来 20 日收益（用 hfq 收盘）
            closes = raw.close.to_numpy()
            fwd = np.full(n, np.nan)
            for k in range(n):
                j = start + k + 20
                if j < len(closes):
                    fwd[k] = closes[j] / closes[start + k] - 1.0
            signals = make_signals(ep_values, symbol, thresholds, fwd)
            if policy_names is None:
                policy_names = list(signals.keys())
            for name, target in signals.items():
                env = make_strict_env(obs, bars, cfg,
                                      execution_version=EXECUTION_CORPORATE_ACTIONS)
                rollout(env, target)
                metrics = env.metrics()
                env.close()
                bucket = results.setdefault(name, {"excess": [], "trades": [],
                                                   "turnover": [], "split": [],
                                                   "exposure": []})
                bucket["excess"].append(metrics["excess_return"])
                bucket["trades"].append(metrics["trades"])
                bucket["turnover"].append(metrics["total_turnover_ratio"])
                bucket["split"].append(split)
                bucket["exposure"].append(env._days_in_market / max(env._n_steps, 1))
            # 基准：恒定持有
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            rollout(env, np.ones(n))
            metrics = env.metrics()
            env.close()
            bucket = results.setdefault("buyhold", {"excess": [], "trades": [],
                                                    "turnover": [], "split": [],
                                                    "exposure": []})
            bucket["excess"].append(metrics["excess_return"])
            bucket["trades"].append(metrics["trades"])
            bucket["turnover"].append(metrics["total_turnover_ratio"])
            bucket["split"].append(split)
            bucket["exposure"].append(env._days_in_market / max(env._n_steps, 1))
            # 空仓
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            rollout(env, np.zeros(n))
            metrics = env.metrics()
            env.close()
            bucket = results.setdefault("cash", {"excess": [], "trades": [],
                                                  "turnover": [], "split": [],
                                                  "exposure": []})
            bucket["excess"].append(metrics["excess_return"])
            bucket["trades"].append(metrics["trades"])
            bucket["turnover"].append(metrics["total_turnover_ratio"])
            bucket["split"].append(split)
            bucket["exposure"].append(env._days_in_market / max(env._n_steps, 1))
        print(f"  {split} 完成")

    # 模型
    for split in ("val", "test"):
        for uid, obs, bars in data.strict_pools[split]:
            env = make_strict_env(obs, bars, cfg,
                                  execution_version=EXECUTION_CORPORATE_ACTIONS)
            rollout(env, None, model)
            metrics = env.metrics()
            env.close()
            bucket = results.setdefault("PPO(v3)", {"excess": [], "trades": [],
                                                     "turnover": [], "split": [],
                                                     "exposure": []})
            bucket["excess"].append(metrics["excess_return"])
            bucket["trades"].append(metrics["trades"])
            bucket["turnover"].append(metrics["total_turnover_ratio"])
            bucket["split"].append(split)
            bucket["exposure"].append(env._days_in_market / max(env._n_steps, 1))

    print()
    order = ["buyhold", "cash", "PPO(v3)"] + [p for p in (policy_names or [])
                                              if p not in ("buyhold", "cash")]
    for split in ("val", "test"):
        print(f"===== {split} 段：各策略相对「同环境恒买」的超额 =====")
        print(f"    {'策略':<18s}{'超额均值':>12s}{'中位数':>12s}{'为正占比':>10s}"
              f"{'平均交易':>10s}{'年换手':>10s}{'在市占比':>10s}")
        rows = []
        for name in order:
            bucket = results.get(name)
            if not bucket:
                continue
            values = np.array([v for v, s in zip(bucket["excess"], bucket["split"])
                               if s == split])
            trades = np.mean([v for v, s in zip(bucket["trades"], bucket["split"])
                              if s == split])
            turnover = np.mean([v for v, s in zip(bucket["turnover"], bucket["split"])
                                 if s == split])
            exposure = np.mean([v for v, s in zip(bucket["exposure"], bucket["split"])
                                if s == split])
            if len(values) == 0:
                continue
            rows.append((name, values, trades, turnover, exposure))
        for name, values, trades, turnover, exposure in sorted(
                rows, key=lambda r: -float(np.mean(r[1]))):
            print(f"    {name:<18s}{np.mean(values):>+11.2%}{np.median(values):>+11.2%}"
                  f"{float((values > 0).mean()):>10.1%}{trades:>10.1f}{turnover:>10.2f}"
                  f"{exposure:>10.1%}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

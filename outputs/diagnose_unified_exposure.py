"""Read-only diagnosis of frozen unified policies; never train or fit normalizers.

Reuses the production feature pipeline and StrictExecutionEnv. Writes only
new diagnostic artifacts next to this script, not application/model/cache files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.config.loader import ConfigLoader
from stock_ai.config.schema import EnvConfig
from stock_ai.env.strict_env import ExecutionBars, StrictExecutionEnv
from stock_ai.eval.backtest_cli import _load_model
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.scale import replay as replay_module
from stock_ai.scale.paths import ScalePaths
from stock_ai.scale.replay import ReplayService, _read_cache
from stock_ai.scale.unified import FEATURES, normalize, observations, relative_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RUN_IDS = ["20261015_095549_25eab142", "20261014_185149_8e273f02"]
LATEST = RUN_IDS[0]
EPS = 1e-10


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(path):
    stat = path.stat()
    return {"sha256": digest(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def finite(value):
    return float(value) if np.isfinite(value) else None


def corr(a, b):
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return finite(np.corrcoef(a, b)[0, 1])


def streaks(mask):
    result, length = [], 0
    for flag in mask:
        if flag:
            length += 1
        elif length:
            result.append(length)
            length = 0
    if length:
        result.append(length)
    return result


def distribution(trainer, obs):
    model = trainer.model
    model.policy.set_training_mode(False)
    actions, probabilities = [], []
    for start in range(0, len(obs), 512):
        batch = np.ascontiguousarray(obs[start:start + 512], dtype="float32")
        predicted, _ = trainer.predict(batch, deterministic=True)
        tensor, _ = model.policy.obs_to_tensor(batch)
        dist = model.policy.get_distribution(tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().astype("float64")
        if not np.allclose(probs.sum(axis=1), 1, atol=2e-6):
            raise AssertionError("Invalid probability sum")
        predicted = np.asarray(predicted, dtype="int64").reshape(-1)
        if not np.array_equal(predicted, probs.argmax(axis=1)):
            raise AssertionError("SB3 deterministic prediction does not equal argmax")
        actions.extend(predicted.tolist())
        probabilities.append(probs)
        del tensor, dist
    return np.asarray(actions, dtype="int64"), np.concatenate(probabilities)


def simulate(obs, bars, cfg, actions=None, mode="policy", detailed=False):
    env = StrictExecutionEnv(obs, bars, cfg)
    env.reset(seed=42)
    n = bars.n_steps
    positions, shares, equities = [0.0], [0], [float(cfg.initial_cash)]
    counter, rejection = Counter(), Counter()
    successful = Counter()
    state_actions = {"flat": Counter(), "held": Counter()}
    reward_sum, return_sum, market_sum = 0.0, 0.0, 0.0
    turnover_sum = 0.0
    first_bought = False
    daily = []
    for i in range(n - 1):
        if mode == "always_buy":
            action = 1
        elif mode == "entry_once":
            action = 0 if first_bought else 1
        else:
            action = int(actions[i])
        counter[action] += 1
        state_actions["held" if env.state.shares else "flat"][action] += 1
        _, reward, _, _, info = env.step(action)
        reward_sum += reward
        turnover_sum += float(info["turnover"])
        market_sum += float(info["benchmark_return"])
        value = float(env.state.portfolio_value)
        return_sum += value / equities[-1] - 1
        if info["trade_ok"]:
            successful[action] += 1
            if action == 1:
                first_bought = True
        elif action:
            rejection[f"{action}:{info['blocked_reason']}"] += 1
        position = float(env.state.shares * bars.close[i + 1] / value)
        if abs(value - (env.state.cash + env.state.shares * bars.close[i + 1])) > 1e-6:
            raise AssertionError("Account identity violated")
        if not -EPS <= position <= 1 + EPS:
            raise AssertionError("Position out of bounds")
        positions.append(position)
        shares.append(int(env.state.shares))
        equities.append(value)
        if detailed:
            daily.append({"date": str(bars.dates[i + 1]), "signal_date": str(bars.dates[i]),
                          "action": action, "position": position, "shares": int(env.state.shares),
                          "cash": float(env.state.cash), "equity": value,
                          "reward": float(reward), "benchmark_return_cc": float(info["benchmark_return"]),
                          "trade_ok": bool(info["trade_ok"]), "blocked_reason": info["blocked_reason"]})
    equity = np.asarray(equities)
    position = np.asarray(positions)
    strategy_returns = equity[1:] / equity[:-1] - 1
    market_returns = bars.close[1:] / bars.close[:-1] - 1
    down = market_returns < 0
    held = np.asarray(shares) > 0
    runs = streaks(held)
    fees = sum(t["fee"] for t in env.trades)
    slip = sum(abs(t["price"] - t["open_price"]) * t["shares"] for t in env.trades)
    expected_reward = (return_sum - market_sum
                       - cfg.lambda_turnover * turnover_sum
                       - cfg.lambda_leverage * position[1:].sum())
    if cfg.lambda_drawdown == 0 and abs(reward_sum - expected_reward) > 1e-9:
        raise AssertionError("Reward decomposition mismatch")
    total_realized = sum(t.get("realized_pnl", 0.0) for t in env.trades)
    last_unrealized = env.state.shares * (bars.close[-1] - env.state.cost_basis)
    if abs(equity[-1] - (cfg.initial_cash + total_realized + last_unrealized)) > 1e-5:
        raise AssertionError("Realized/unrealized identity violated")
    result = {
        "bars": n, "signals": n - 1, "holding_days": int(held.sum()),
        "holding_fraction": float(held.mean()), "average_position": float(position.mean()),
        "full_position_days": int((position >= 0.95).sum()),
        "full_position_fraction": float((position >= 0.95).mean()),
        "longest_holding_days": max(runs) if runs else 0,
        "holding_spells": len(runs), "open_at_end": bool(held[-1]),
        "total_return": float(equity[-1] / cfg.initial_cash - 1),
        "max_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity))),
        "daily_market_correlation": corr(strategy_returns, market_returns),
        "market_down_days": int(down.sum()),
        "pre_down_position_sum": float(position[:-1][down].sum()),
        "pre_down_average_position": float(position[:-1][down].mean()) if down.any() else None,
        "actions": {str(k): int(counter[k]) for k in range(3)},
        "actions_by_starting_state": {key: {str(k): int(value[k]) for k in range(3)}
                                      for key, value in state_actions.items()},
        "successful_orders": {str(k): int(successful[k]) for k in range(3)},
        "rejections": dict(rejection), "trades": len(env.trades),
        "total_fee": float(fees), "slippage_cost": float(slip),
        "cost_drag_bps": float(fees / equity.mean() * 10000),
        "turnover_sum": turnover_sum, "reward_sum": reward_sum,
        "sum_simple_returns": return_sum, "sum_market_cc_returns": market_sum,
        "reward_turnover_penalty": cfg.lambda_turnover * turnover_sum,
        "reward_exposure_penalty": cfg.lambda_leverage * float(position[1:].sum()),
        "final_equity": float(equity[-1]),
    }
    if detailed:
        result["daily"] = daily
        result["equity"] = equity.tolist()
        result["positions"] = position.tolist()
        result["trade_details"] = list(env.trades)
    env.close()
    return result


def mean(records, key):
    values = [row[key] for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate(records):
    if not records:
        return {"episodes": 0}
    count = len(records)
    bars = sum(r["bars"] for r in records)
    signals = sum(r["signals"] for r in records)
    down_days = sum(r["market_down_days"] for r in records)
    actions = {str(k): sum(r["actions"][str(k)] for r in records) for k in range(3)}
    successes = {str(k): sum(r["successful_orders"][str(k)] for r in records) for k in range(3)}
    rejections = Counter()
    for row in records:
        rejections.update(row["rejections"])
    result = {
        "episodes": count, "symbols": len({r["symbol"] for r in records}),
        "bars": bars, "signals": signals,
        "mean_position": mean(records, "average_position"),
        "weighted_mean_position": sum(r["average_position"] * r["bars"] for r in records) / bars,
        "holding_fraction": sum(r["holding_days"] for r in records) / bars,
        "full_position_fraction": sum(r["full_position_days"] for r in records) / bars,
        "episodes_position_ge_90pct": sum(r["average_position"] >= 0.9 for r in records),
        "episodes_position_le_10pct": sum(r["average_position"] <= 0.1 for r in records),
        "mean_return": mean(records, "total_return"),
        "mean_baseline_return": mean(records, "baseline_return"),
        "mean_excess_return": mean(records, "excess_return"),
        "median_excess_return": float(np.median([r["excess_return"] for r in records])),
        "excess_win_fraction": sum(r["excess_return"] > 0 for r in records) / count,
        "mean_max_drawdown": mean(records, "max_drawdown"),
        "mean_baseline_max_drawdown": mean(records, "baseline_max_drawdown"),
        "mean_trades": mean(records, "trades"),
        "mean_daily_market_correlation": mean(records, "daily_market_correlation"),
        "pre_down_average_position": sum(r["pre_down_position_sum"] for r in records) / down_days if down_days else None,
        "market_down_days": down_days,
        "mean_cost_drag_bps": mean(records, "cost_drag_bps"),
        "mean_turnover_sum": mean(records, "turnover_sum"),
        "mean_reward_exposure_penalty": mean(records, "reward_exposure_penalty"),
        "mean_reward_turnover_penalty": mean(records, "reward_turnover_penalty"),
        "actions": actions, "action_fractions": {k: v / signals for k, v in actions.items()},
        "successful_orders": successes, "rejections": dict(rejections),
        "mean_policy_entropy": sum(r["mean_policy_entropy"] * r["signals"] for r in records) / signals,
        "mean_probabilities": [sum(r["mean_probabilities"][k] * r["signals"] for r in records) / signals for k in range(3)],
        "confident_90pct_fraction": sum(r["confident_90pct_count"] for r in records) / signals,
        "mean_longest_holding_days": mean(records, "longest_holding_days"),
        "no_sell_signal_episodes": sum(r["actions"]["2"] == 0 for r in records),
    }
    result["actions_by_starting_state"] = {
        state: {str(k): sum(r["actions_by_starting_state"][state][str(k)] for r in records)
                for k in range(3)} for state in ("flat", "held")}
    for key in ("average_position", "holding_fraction", "total_return", "excess_return", "max_drawdown"):
        result[key + "_percentiles"] = dict(zip(("p10", "p25", "p50", "p75", "p90"),
                                                 np.percentile([r[key] for r in records], [10, 25, 50, 75, 90]).tolist()))
    return result


def main():
    started = time.monotonic()
    metas, norms, models, frozen, saved_metrics, split = {}, {}, {}, {}, {}, None
    for run_id in RUN_IDS:
        directory = ROOT / "models" / "unified" / run_id
        metas[run_id] = read_json(directory / "manifest.json")
        meta = metas[run_id]
        if meta["execution_version"] != "strict-t1-open-v1" or meta["feature_adjust"] != "hfq":
            raise AssertionError("Unexpected model contract")
        norms[run_id] = NormalizerState.from_dict(read_json(directory / "normalizer.json"))
        models[run_id] = _load_model(directory / "model.zip")
        saved_metrics[run_id] = read_json(directory / "metrics.json")
        current_split = read_json(directory / "split.json")
        if split is not None and current_split != split:
            raise AssertionError("Runs do not share the same frozen split")
        split = current_split
        for name in ("manifest.json", "normalizer.json", "model.zip", "split.json", "status.json", "metrics.json"):
            path = directory / name
            frozen[str(path)] = fingerprint(path)
    if norms[RUN_IDS[0]].to_dict() != norms[RUN_IDS[1]].to_dict():
        raise AssertionError("Two models use different normalizers")
    cfg = EnvConfig(**metas[LATEST]["env_config"])
    by_symbol = {}
    for name in ("val", "test"):
        for episode in split["episodes"][name]:
            by_symbol.setdefault(episode["symbol"], []).append((name, episode))
    source_by_symbol = {r["symbol"]: r for r in metas[LATEST]["sources"]}
    records, baseline_records, details = [], [], []
    adjustment_candidates = []
    checks = {"model_input_hashes_match_manifest": True, "source_dates_exact": True,
              "accounting_identities": True, "policy_predict_equals_probability_argmax": True,
              "saved_evaluation_reproduced": True, "ui_replay_crosschecks": []}
    model_parameters = {}
    for run_id, trainer in models.items():
        model = trainer.model
        model_parameters[run_id] = {"policy_class": type(model.policy).__name__,
            "feature_extractor": type(model.policy.features_extractor).__name__,
            "observation_shape": list(model.observation_space.shape),
            "num_timesteps": int(model.num_timesteps), "n_envs": int(model.n_envs),
            "gamma": float(model.gamma), "gae_lambda": float(model.gae_lambda),
            "ent_coef": float(model.ent_coef), "target_kl": model.target_kl,
            "n_steps": int(model.n_steps), "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs), "net_arch": model.policy.net_arch}
    selected = {("000001", 2024), ("600519", 2022), ("300750", 2023), ("600519", 2024)}
    train_cc = []
    for si, (symbol, eps) in enumerate(sorted(by_symbol.items()), 1):
        feature_path = Path(metas[LATEST]["feature_dir"]) / f"{symbol}.parquet"
        raw_path = Path(metas[LATEST]["execution_dir"]) / f"{symbol}.parquet"
        for path in (feature_path, raw_path):
            frozen[str(path)] = fingerprint(path)
        if digest(feature_path) != source_by_symbol[symbol]["sha256"]:
            checks["model_input_hashes_match_manifest"] = False
            raise AssertionError(f"Feature source changed since training: {symbol}")
        feature_df = _read_cache(feature_path, feature_path.parent, "hfq")
        raw = _read_cache(raw_path, raw_path.parent, "raw")
        feature_df, values = relative_features(feature_df)
        keep = feature_df.date.isin(raw.date).to_numpy()
        dates = feature_df.date.dt.strftime("%Y%m%d").to_numpy()[keep]
        values = values[keep]
        tensor = observations(normalize(values, norms[LATEST]), metas[LATEST]["window"])
        market = raw.set_index(raw.date.dt.strftime("%Y%m%d")).loc[dates]
        # Screening only: mismatches can indicate corporate actions OR source issues.
        # Do not turn these candidates into invented event records or adjust holdings.
        raw_returns = pd.Series(market.close.to_numpy()).pct_change().to_numpy()
        hfq_returns = pd.Series(feature_df.close.to_numpy()[keep]).pct_change().to_numpy()
        mismatch = np.flatnonzero((dates >= "20220101") & (dates <= "20251231")
                                 & (np.abs(raw_returns - hfq_returns) > 0.05))
        for j in mismatch:
            adjustment_candidates.append({"symbol": symbol, "date": str(dates[j]),
                "year": int(dates[j][:4]), "raw_return": float(raw_returns[j]),
                "hfq_return": float(hfq_returns[j]), "difference": float(raw_returns[j] - hfq_returns[j]),
                "previous_raw_close": float(market.close.iloc[j - 1]),
                "raw_open": float(market.open.iloc[j]), "raw_close": float(market.close.iloc[j]),
                "classification": "candidate_not_individually_verified"})
        predictions = {run: distribution(trainer, tensor[(dates >= "20220101") & (dates <= "20251231")])
                       for run, trainer in models.items()}
        all_eval = np.flatnonzero((dates >= "20220101") & (dates <= "20251231"))
        lookup = {idx: j for j, idx in enumerate(all_eval)}
        for train_ep in split["episodes"]["train"]:
            if train_ep["symbol"] == symbol:
                rows = market[(dates >= train_ep["start"]) & (dates <= train_ep["end"])]
                if len(rows) >= 2:
                    train_cc.append({"episode_id": train_ep["id"],
                                     "close_to_close_return": float(rows.close.iloc[-1] / rows.close.iloc[0] - 1)})
        for split_name, ep in eps:
            ids = np.flatnonzero((dates >= ep["start"]) & (dates <= ep["end"]))
            if len(ids) != ep["rows"]:
                raise AssertionError(f"Frozen split row count changed: {ep['id']}")
            raw_dates = set(raw.loc[(raw.date.dt.year == ep["year"]), "date"].dt.strftime("%Y%m%d"))
            feature_dates = set(feature_df.loc[(feature_df.date.dt.year == ep["year"]), "date"].dt.strftime("%Y%m%d"))
            if raw_dates != feature_dates or set(dates[ids]) != raw_dates:
                raise AssertionError(f"UI would reject unaligned requested dates: {ep['id']}")
            frame = market.iloc[ids]
            bars = ExecutionBars(dates=dates[ids], symbol=symbol,
                **{key: frame[key].to_numpy(dtype="float64") for key in ("open", "high", "low", "close", "volume")})
            obs = tensor[ids]
            base = simulate(obs, bars, cfg, mode="always_buy")
            once = simulate(obs, bars, cfg, mode="entry_once")
            # Fixed causal non-optimized baseline. Current close vs own 20-day SMA.
            rule_actions = np.where(values[ids, FEATURES.index("ma20_relative")] > 0, 1, 2)
            ma20 = simulate(obs, bars, cfg, actions=rule_actions)
            baseline_records.append({"episode_id": ep["id"], "symbol": symbol, "year": ep["year"], "split": split_name,
                "always_buy_return": base["total_return"], "always_buy_drawdown": base["max_drawdown"],
                "entry_once_return": once["total_return"], "entry_once_drawdown": once["max_drawdown"],
                "ma20_return": ma20["total_return"], "ma20_drawdown": ma20["max_drawdown"],
                "ma20_average_position": ma20["average_position"], "ma20_trades": ma20["trades"]})
            for run_id in RUN_IDS:
                pred, probs = predictions[run_id]
                local = np.asarray([lookup[int(idx)] for idx in ids], dtype="int64")
                actions = pred[local]
                probabilities = probs[local[:-1]]
                result = simulate(obs, bars, EnvConfig(**metas[run_id]["env_config"]), actions=actions,
                                  detailed=(run_id == LATEST and (symbol, ep["year"]) in selected))
                if "daily" in result:
                    details.append({"run_id": run_id, "episode_id": ep["id"],
                                    "initial_cash": cfg.initial_cash,
                                    "initial_date": str(bars.dates[0]),
                                    "baseline_return": base["total_return"],
                                    **{k: result.pop(k) for k in ("daily", "positions", "equity", "trade_details")}})
                result.update({"run_id": run_id, "episode_id": ep["id"], "symbol": symbol,
                    "year": ep["year"], "split": split_name, "start": ep["start"], "end": ep["end"],
                    "baseline_return": base["total_return"], "baseline_max_drawdown": base["max_drawdown"],
                    "entry_once_return": once["total_return"], "ma20_return": ma20["total_return"],
                    "excess_return": result["total_return"] - base["total_return"],
                    "mean_policy_entropy": float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)), axis=1))),
                    "mean_probabilities": probabilities.mean(axis=0).tolist(),
                    "confident_90pct_count": int((probabilities.max(axis=1) >= 0.9).sum())})
                saved = next(r for r in saved_metrics[run_id]["evaluation"][split_name]["episodes"] if r["episode_id"] == ep["id"])
                if abs(saved["total_return"] - result["total_return"]) > EPS or saved["trades"] != result["trades"] or abs(saved["max_drawdown"] - result["max_drawdown"]) > EPS:
                    raise AssertionError(f"Saved evaluation not reproduced: {run_id}/{ep['id']}")
                records.append(result)
        if si % 25 == 0 or si == len(by_symbol):
            print(json.dumps({"progress_symbols": si, "total_symbols": len(by_symbol), "records": len(records),
                              "elapsed_sec": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
    # Real production replay path crosschecks, sharing only already-loaded immutable policies.
    app_cfg = ConfigLoader.load(ROOT / "configs" / "default.yaml")
    service = ReplayService(app_cfg, ScalePaths(ROOT))
    loader = replay_module._replay_model
    def cached_loader(path, features):
        return models[Path(path).parent.name], metas[Path(path).parent.name]["window"], len(features)
    replay_module._replay_model = cached_loader
    try:
        for symbol, year in sorted(selected):
            result = service.run({"model_id": "unified:" + LATEST, "symbol": symbol,
                                  "start": f"{year}0101", "end": f"{year}1231"})
            row = next(r for r in records if r["run_id"] == LATEST and r["symbol"] == symbol and r["year"] == year)
            for key in ("total_return", "max_drawdown", "average_position", "trades", "holding_days", "baseline_return"):
                if abs(row[key] - result["summary"][key]) > EPS:
                    raise AssertionError(f"UI replay mismatch: {symbol}/{year}/{key}")
            checks["ui_replay_crosschecks"].append({"symbol": symbol, "year": year, "matched": True})
    finally:
        replay_module._replay_model = loader
    candidate_episodes = {f"{r['symbol']}_{r['year']}" for r in adjustment_candidates}
    sensitivity = {}
    for run_id in RUN_IDS:
        # Diagnostic sensitivity exclusion, NOT a corrected backtest and not an unbiased sample.
        sensitivity[run_id] = aggregate([r for r in records if r["run_id"] == run_id
                                         and r["episode_id"] not in candidate_episodes])
    model_summaries = {}
    for run_id in RUN_IDS:
        subset = [r for r in records if r["run_id"] == run_id]
        model_summaries[run_id] = {"all": aggregate(subset),
            "by_split": {key: aggregate([r for r in subset if r["split"] == key]) for key in ("val", "test")},
            "by_year": {str(year): aggregate([r for r in subset if r["year"] == year]) for year in range(2022, 2026)},
            "by_baseline_direction": {key: aggregate([r for r in subset if (r["baseline_return"] < 0) == is_down])
                                      for key, is_down in (("negative", True), ("nonnegative", False))},
            "by_baseline_regime": {key: aggregate([r for r in subset if predicate(r["baseline_return"])])
                for key, predicate in (("down_below_minus5pct", lambda x: x < -0.05),
                                       ("sideways_minus5_to_plus5pct", lambda x: -0.05 <= x <= 0.05),
                                       ("up_above_plus5pct", lambda x: x > 0.05))}}
    baseline_summaries = {}
    for label, subset in [("all", baseline_records)] + [(str(y), [r for r in baseline_records if r["year"] == y]) for y in range(2022, 2026)]:
        baseline_summaries[label] = {"episodes": len(subset), **{key: mean(subset, key) for key in (
            "always_buy_return", "always_buy_drawdown", "entry_once_return", "entry_once_drawdown", "ma20_return", "ma20_drawdown", "ma20_average_position", "ma20_trades")}}
    protected_files = len(frozen)
    for path, before in frozen.items():
        if fingerprint(Path(path)) != before:
            raise AssertionError(f"Protected input changed during diagnosis: {path}")
    checks["protected_input_files_unchanged"] = protected_files
    result = {
        "generated_at": datetime.now().isoformat(), "elapsed_sec": time.monotonic() - started,
        "scope": {"symbols": len(by_symbol), "years": [2022, 2023, 2024, 2025],
                  "episodes_per_model": len(baseline_records), "val_episodes": len(split["episodes"]["val"]),
                  "test_episodes": len(split["episodes"]["test"]), "yearly_reset": True,
                  "source": "local frozen manifest + cached hfq features / unadjusted OHLCV execution",
                  "held_at_close": True, "inference": "deterministic argmax; probabilities measured without sampling",
                  "averaging": "return, drawdown, mean_position: equal weight stock-year episodes, not a compound portfolio return",
                  "baseline": "independent StrictExecutionEnv daily buy, initial cash at first close, same fees/slippage/rejections",
                  "pre_down_position": "previous close position on intervals with a negative next close-to-close raw price return; not proof of causality",
                  "caveats": ["val/test years were inspected during earlier tuning; not pristine generalization evidence",
                              "ST/new-listing exemptions/dividends/splits/corporate actions not fully modeled",
                              "cross-stock holdout absent; current constituent universe may have survivorship bias",
                              "annual episode resets differ from continuous multiyear UI replay"]},
        "checks": checks, "models": {run: {"parameters": model_parameters[run],
            "reward_params": metas[run]["reward_params"], "model_sha256": frozen[str(ROOT / "models" / "unified" / run / "model.zip")]["sha256"]} for run in RUN_IDS},
        "summaries": model_summaries, "baseline_summaries": baseline_summaries,
        "train_market_context": {"episodes_measured": len(train_cc),
            "mean_stock_year_close_to_close_return": mean(train_cc, "close_to_close_return"),
            "positive_stock_year_fraction": sum(r["close_to_close_return"] > 0 for r in train_cc) / len(train_cc) if train_cc else None,
            "note": "descriptive raw-price drift only, no simulation and not total return incl corporate actions"},
        "corporate_action_audit": {
            "screen": "absolute difference between same-date raw and hfq close-to-close return > 0.05",
            "candidate_days": len(adjustment_candidates),
            "candidate_symbols": len({r["symbol"] for r in adjustment_candidates}),
            "candidate_episodes": len(candidate_episodes),
            "warning": "Candidates are NOT all confirmed corporate events. Source differences must also be audited. Returns and drawdowns use an environment without entitlement accounting, and are not reliable investment performance.",
            "verified_case": {"symbol": "300750", "date": "20230426", "record_date": "20230425",
                "shares_multiplier": 1.8, "gross_cash_dividend_per_old_share": 2.52,
                "official_source": "https://www.catl.com/uploads/1/file/public/202304/20230421100127_yuyuyobjuc.pdf",
                "corroboration": "https://static.cninfo.com.cn/finalpage/2024-06-18/1220392100.PDF",
                "basis": "Official company disclosures; no vendor historical quote values used"},
            "sensitivity_excluding_candidate_episodes": sensitivity,
            "sensitivity_warning": "Not an entitlement-corrected backtest; screening threshold misses small dividends and selection is hindsight-conditioned.",
        },
        "episodes": records, "baselines": baseline_records, "selected_daily": details,
        "adjustment_candidates": adjustment_candidates,
    }
    write_json(OUT / "unified_exposure_diagnosis_20261016.json", result)
    fields = ["run_id", "symbol", "year", "split", "bars", "average_position", "holding_fraction", "full_position_fraction",
              "longest_holding_days", "total_return", "baseline_return", "excess_return", "max_drawdown", "trades",
              "cost_drag_bps", "mean_policy_entropy", "pre_down_average_position", "buy_signals", "hold_signals", "sell_signals", "successful_sells", "mean_p_hold", "mean_p_buy", "mean_p_sell"]
    with (OUT / "unified_exposure_episodes_20261016.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            flat = {k: row[k] for k in fields if k in row}
            flat.update(buy_signals=row["actions"]["1"], hold_signals=row["actions"]["0"], sell_signals=row["actions"]["2"],
                        successful_sells=row["successful_orders"]["2"],
                        mean_p_hold=row["mean_probabilities"][0], mean_p_buy=row["mean_probabilities"][1], mean_p_sell=row["mean_probabilities"][2])
            writer.writerow(flat)
    with (OUT / "unified_adjustment_candidates_20261016.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adjustment_candidates[0]))
        writer.writeheader()
        writer.writerows(adjustment_candidates)
    summary = {k: result[k] for k in ("generated_at", "elapsed_sec", "scope", "checks", "models", "summaries", "baseline_summaries", "train_market_context", "corporate_action_audit")}
    write_json(OUT / "unified_exposure_summary_20261016.json", summary)
    print(json.dumps({"done": True, "elapsed_sec": result["elapsed_sec"], "checks": checks,
                      "year_summary_latest": {year: {k: value[k] for k in ("episodes", "mean_position", "holding_fraction", "mean_return", "mean_baseline_return", "mean_excess_return", "mean_trades", "action_fractions")}
                                              for year, value in model_summaries[LATEST]["by_year"].items()}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

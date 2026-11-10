"""只读缓存的研究回放；不使用 predictor、不改动批量回测的执行语义。"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.env.trading_env import TradingEnv
from stock_ai.eval.backtest_cli import _load_model
from stock_ai.eval.backtester import Backtester
from stock_ai.samples.features import BASE_COLUMNS, FeatureEngineer
from stock_ai.samples.schema import SampleSplit
from stock_ai.scale.normalizer_store import NormalizerStore
from stock_ai.scale.paths import ScalePaths

_SYMBOL = re.compile(r"[0-9]{6}\Z")
_MODEL = re.compile(r"([0-9]{6})_([0-9]{4})\Z")
MAX_BARS = 2500
MAX_RANGE_DAYS = 3660
MAX_HISTORY = 30000
# 同一进程内所有 UI 实例共享保护，避免多个请求同时加载/推理。
_INFERENCE_LOCK = threading.Lock()


class ReplayError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _date(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}|[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ReplayError(f"{name} 必须为 YYYYMMDD 或 YYYY-MM-DD")
    compact = value.replace("-", "")
    try:
        datetime.strptime(compact, "%Y%m%d")
    except ValueError as exc:
        raise ReplayError(f"{name} 不是有效日期") from exc
    return compact


def _file(path: Path, directory: Path) -> bool:
    # 精确父目录约束同时拒绝指向目录外的符号链接。
    return path.resolve().parent == directory.resolve() and path.is_file()


def _read_cache(path: Path, directory: Path, label: str = "缓存", *,
                with_sha256: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, str]:
    if not _file(path, directory):
        raise ReplayError(f"{label}缺失或路径不安全；回放不会联网补数据，也不会回退其他复权口径", 404)
    # 解析和审计必须使用同一次读取的字节，不能解析后再打开路径计算另一份指纹。
    content = path.read_bytes()
    df = pd.read_parquet(BytesIO(content))
    if not 0 < len(df) <= MAX_HISTORY:
        raise ReplayError(f"{label}历史必须为 1~{MAX_HISTORY} 行", 400)
    if not {"date", *BASE_COLUMNS}.issubset(df.columns):
        raise ReplayError(f"{label}缺少 date/OHLCV 列", 500)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="raise").dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ReplayError(f"{label}存在空日期或重复交易日", 500)
    values = df[list(BASE_COLUMNS)].to_numpy(dtype="float64")
    df[list(BASE_COLUMNS)] = values
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
            or (df.high < df[["open", "close", "low"]].max(axis=1)).any()
            or (df.low > df[["open", "close", "high"]].min(axis=1)).any()):
        raise ReplayError(f"{label} OHLCV 非有限或价格/成交量不合法", 500)
    return (df, hashlib.sha256(content).hexdigest()) if with_sha256 else df


def _replay_model(path: Path, features: list[str]):
    model = _load_model(path)
    shape = getattr(model, "obs_shape", None)
    if shape is None:
        shape = getattr(getattr(getattr(model, "model", None), "observation_space", None), "shape", None)
    if shape is None or len(shape) != 2:
        raise ReplayError("模型缺少兼容的二维观测 shape", 500)
    window, count = map(int, shape)
    if (not 1 <= window <= 512 or count != len(features)
            or getattr(model, "obs_dim", window * count) != window * count):
        raise ReplayError("模型观测 shape 与冻结 norm 不兼容", 500)
    inner = getattr(model, "model", None)
    n_actions = getattr(model, "n_actions", getattr(getattr(inner, "action_space", None), "n", None))
    if n_actions != 3:
        raise ReplayError("回放仅支持现有三动作离散模型", 500)
    return model, window, count


def _predict_action(model: Any, obs: np.ndarray) -> int:
    action, _ = model.predict(obs, deterministic=True)
    value = np.asarray(action).reshape(-1)
    if (value.size != 1 or not np.isfinite(value).all()
            or float(value[0]) not in (0, 1, 2)):
        raise ReplayError("模型返回非法离散动作", 500)
    return int(value[0])


def _summarize(daily: list[dict], trades: list[dict], initial_cash: float) -> dict:
    by_date = {row["date"]: row for row in daily}
    previous, peak = initial_cash, initial_cash
    for row in daily:
        row["position_value"] = float(row["shares"] * row["close"])
        row.setdefault("position", row["position_value"] / max(row["equity"], 1e-9))
        row["unrealized_pnl"] = row["position_value"] - row["shares"] * row["cost_basis"]
        row["daily_return"] = row["equity"] / previous - 1 if previous > 0 else 0.0
        peak = max(peak, row["equity"])
        row["drawdown"] = max(0.0, 1 - row["equity"] / peak) if peak > 0 else 0.0
        row["fee"], row["slippage_cost"] = 0.0, 0.0
        previous = row["equity"]
    for trade in trades:
        row = by_date[trade["date"]]
        # 滑点已计入成交价；这里只披露磨损，不能再从权益中扣除。
        reference = float(trade.get("open_price", row["close"]))
        slippage = abs(float(trade["price"]) - reference) * int(trade["shares"])
        trade["slippage_cost"] = slippage
        trade.setdefault("realized_pnl", 0.0)
        row["fee"] += float(trade["fee"])
        row["slippage_cost"] += slippage
    total_return = daily[-1]["equity"] / initial_cash - 1
    baseline_return = daily[-1]["baseline"] / initial_cash - 1
    total_fee = float(sum(row["fee"] for row in daily))
    average_equity = float(np.mean([row["equity"] for row in daily]))
    summary = {"bars": len(daily), "executions": max(0, len(daily) - 1), "trades": len(trades),
               "final_equity": daily[-1]["equity"], "total_return": total_return,
               "baseline_return": baseline_return, "excess_return": total_return - baseline_return,
               "max_drawdown": max(row["drawdown"] for row in daily),
               "average_position": float(np.mean([row["position"] for row in daily])),
               "holding_days": sum(row["shares"] > 0 for row in daily), "total_fee": total_fee,
               "slippage_cost": float(sum(row["slippage_cost"] for row in daily)),
               "cost_drag_bps": total_fee / average_equity * 10000 if average_equity > 0 else 0.0}
    if "dividend_receivable" in daily[-1]:
        summary.update(dividend_income=daily[-1]["dividend_income"],
                       final_dividend_receivable=daily[-1]["dividend_receivable"],
                       final_locked_shares=daily[-1]["locked_shares"],
                       final_available_shares=daily[-1]["available_shares"],
                       realized_pnl=daily[-1]["realized_pnl"],
                       unrealized_pnl=daily[-1]["unrealized_pnl"])
    return summary


_COST_NOTE = "cost_drag_bps = 总手续费 / 逐日平均权益 × 10000；手续费含佣金及卖出印花税，不含滑点。滑点已计入成交价，单列披露而不重复扣减。cost_basis 为含买入费用的每股持仓均价。"


class ReplayService:
    def __init__(self, cfg: Any, paths: ScalePaths):
        self.cfg = cfg
        self.paths = paths
        self.raw_dir = Path(cfg.paths.raw)
        self.normalizers = NormalizerStore(paths)

    def models(self) -> list[dict]:
        items = []
        for path in sorted(self.paths.models_dir.glob("*.zip")):
            match = _MODEL.fullmatch(path.stem)
            if not match or not _file(path, self.paths.models_dir):
                continue
            symbol, year = match.group(1), int(match.group(2))
            if not 1900 <= year <= 2100:
                continue
            uid = path.stem
            # 不信任 sidecar.model_path；模型身份必须与受控文件名一致。
            sidecar = self.paths.sidecar(uid)
            if not _file(sidecar, self.paths.models_dir):
                continue
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if (meta.get("unit_id") != uid or meta.get("symbol") != symbol
                        or meta.get("year") != year):
                    continue
            except (ValueError, OSError, AttributeError):
                continue
            norm_ok = _file(self.paths.norm(uid), self.paths.models_dir)
            items.append({"id": uid, "symbol": symbol, "year": year,
                          "backend": str(meta.get("backend", "unknown")),
                          "algo": str(meta.get("algo", "unknown")),
                          "normalizer_available": norm_ok})
        items.extend(r for r in self.unified_runs() if r["model_available"])
        return items

    def unified_runs(self) -> list[dict]:
        from stock_ai.scale.unified import (CONTRACT, EXECUTION_LEGACY, EXECUTION_STRICT,
                                            EXECUTION_CORPORATE_ACTIONS, EXECUTION_VERSIONS, STRICT_EXECUTION_VERSIONS,
                                            FEATURES, RUN_ID, unified_dir)

        directory = unified_dir(self.paths.root)
        items = []
        for run in sorted(directory.glob("*")):
            if not RUN_ID.fullmatch(run.name) or not run.is_dir() or run.resolve().parent != directory.resolve():
                continue
            try:
                manifest_path = run / "manifest.json"
                if not _file(manifest_path, run):
                    continue
                meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                if meta.get("run_id") != run.name or meta.get("mode") != "unified":
                    continue
                version = meta.get("execution_version", EXECUTION_LEGACY)
                feature_adjust = meta.get("feature_adjust", "raw(as-cached)" if version == EXECUTION_LEGACY else None)
                compatible = (meta.get("contract") == CONTRACT and meta.get("features") == FEATURES
                              and meta.get("observation_shape") == [meta.get("window"), len(FEATURES)]
                              and meta.get("algo") == "ppo" and meta.get("backend") == "sb3"
                              and version in EXECUTION_VERSIONS
                              and (version not in STRICT_EXECUTION_VERSIONS or feature_adjust == "hfq"))
                env_cfg = meta.get("env_config", {})
                reward_type = meta.get("reward_type", env_cfg.get("reward_type"))
                reward_params = meta.get("reward_params", {key: env_cfg[key] for key in
                                         ("reward_type", "lambda_drawdown", "lambda_turnover", "lambda_leverage")
                                         if key in env_cfg})
                warning = "统一相对特征契约；仅手选，训练区间内存在重叠风险。"
                if version not in EXECUTION_VERSIONS:
                    warning = f"未知 execution_version: {version!r}，禁止回放，不会降级。"
                elif not compatible:
                    warning = "unified 特征/执行契约不兼容，禁止回放。"
                elif version == EXECUTION_STRICT:
                    warning += " 旧 v1 遗漏分红送转等公司行为，历史收益不能视为投资总回报；研究修正必须显式勾选，且不会修复旧训练奖励。"
                elif version == EXECUTION_CORPORATE_ACTIONS:
                    warning += " 公司行为账本必须显式 verified 覆盖请求完整区间；训练股票须 hfq/未复权/账本完整文件指纹匹配。税前股息口径，不含个人差异化股息税。"
                st = json.loads((run / "status.json").read_text(encoding="utf-8")) if _file(run / "status.json", run) else {}
                items.append({"id": "unified:" + run.name, "run_id": run.name, "mode": "unified",
                              "symbol": "*", "year": int(meta["ranges"]["train"][1]),
                              "created_at": meta.get("created_at"), "updated_at": st.get("updated_at"),
                              "backend": "sb3", "algo": "ppo", "contract": meta.get("contract"),
                              "execution_version": version, "feature_adjust": feature_adjust,
                              "reward_type": reward_type, "reward_version": meta.get("reward_version"),
                              "reward_params": reward_params, "ranges": meta.get("ranges"), "window": meta.get("window"),
                              "compatible": compatible, "process_state": st.get("process_state", "unknown"),
                              "timesteps": st.get("timesteps", 0), "model_available": _file(run / "model.zip", run),
                              "normalizer_available": _file(run / "normalizer.json", run), "warning": warning,
                              **({"tax_policy": meta.get("tax_policy", "gross_before_personal_dividend_tax"),
                                  "corporate_actions_coverage_required": True}
                                 if version == EXECUTION_CORPORATE_ACTIONS else {})})
            except (ValueError, OSError, KeyError, TypeError, AttributeError):
                continue
        return items

    def catalog(self) -> dict:
        symbols = [p.stem for p in sorted(self.raw_dir.glob("*.parquet"))
                   if _SYMBOL.fullmatch(p.stem) and _file(p, self.raw_dir)]
        return {"ok": True, "symbols": symbols, "models": self.models(), "unified_runs": self.unified_runs(),
                "limits": {"max_bars": MAX_BARS, "max_range_days": MAX_RANGE_DAYS}}

    def run(self, payload: Any) -> dict:
        if not isinstance(payload, dict):
            raise ReplayError("请求体必须是 JSON 对象")
        if set(payload) - {"symbol", "start", "end", "model_id", "execution_mode"}:
            raise ReplayError("存在不支持的请求字段；只接受模型 ID 与执行模式，不接受文件路径")
        execution_mode = payload.get("execution_mode", "default")
        if execution_mode not in ("default", "corporate_actions_research"):
            raise ReplayError("execution_mode 只接受 default 或 corporate_actions_research")
        symbol = payload.get("symbol")
        if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
            raise ReplayError("symbol 必须为六位股票代码")
        start, end = _date(payload.get("start"), "start"), _date(payload.get("end"), "end")
        span = (datetime.strptime(end, "%Y%m%d") - datetime.strptime(start, "%Y%m%d")).days
        if span < 0 or span > MAX_RANGE_DAYS:
            raise ReplayError(f"日期区间必须顺序正确且不超过 {MAX_RANGE_DAYS} 天")
        model_id = payload.get("model_id", "auto")
        from stock_ai.scale.unified import RUN_ID
        unified_id = isinstance(model_id, str) and model_id.startswith("unified:") and RUN_ID.fullmatch(model_id[8:])
        if not isinstance(model_id, str) or (model_id != "auto" and not _MODEL.fullmatch(model_id) and not unified_id):
            raise ReplayError("model_id 必须是列表中的模型 ID 或 auto，禁止任意路径")
        if execution_mode == "corporate_actions_research" and not unified_id:
            raise ReplayError("公司行为研究执行覆写仅支持显式选择的 unified v1 旧模型")
        if not _INFERENCE_LOCK.acquire(blocking=False):
            raise ReplayError("已有回放正在加载或推理，请稍后重试", 409)
        try:
            result = self._run(symbol, start, end, model_id, execution_mode)
            # 强制检查有限值；不要把 NaN/Infinity 或部分回放作为成功结果返回。
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            return result
        except ReplayError:
            raise
        except Exception as exc:
            raise ReplayError(f"回放失败（缓存、norm 或模型损坏/不兼容）：{type(exc).__name__}: {exc}", 500) from exc
        finally:
            _INFERENCE_LOCK.release()

    def _run(self, symbol: str, start: str, end: str, model_id: str,
             execution_mode: str = "default") -> dict:
        raw_path = self.raw_dir / f"{symbol}.parquet"
        models = self.models()
        if model_id == "auto":
            candidates = [m for m in models if m["symbol"] == symbol
                          and m["year"] < int(start[:4]) and m["normalizer_available"]]
            if not candidates:
                raise ReplayError("没有同代码、年份严格早于回放起始年的模型及冻结 norm；不会回退到同年/未来模型", 404)
            selected = max(candidates, key=lambda m: (m["year"], m["id"]))
        else:
            selected = next((m for m in models if m["id"] == model_id), None)
            if selected is None:
                raise ReplayError("模型 ID 不在合法现有模型列表中", 404)
        uid = selected["id"]
        if not selected["normalizer_available"]:
            raise ReplayError("模型的冻结 norm 缺失；禁止重新 fit", 404)
        is_unified = selected.get("mode") == "unified"
        if is_unified:
            from stock_ai.scale.unified import (EXECUTION_LEGACY, EXECUTION_STRICT, EXECUTION_VERSIONS,
                                                STRICT_EXECUTION_VERSIONS, unified_dir)
            from stock_ai.samples.normalizer import NormalizerState
            version = selected["execution_version"]
            if version not in EXECUTION_VERSIONS:
                raise ReplayError(f"未知 execution_version: {version!r}，禁止回放，不会降级")
            if not selected["compatible"]:
                raise ReplayError("unified 特征/执行契约不兼容，禁止使用旧特征回放")
            if execution_mode == "corporate_actions_research" and version != EXECUTION_STRICT:
                raise ReplayError("公司行为研究执行覆写仅支持 strict-t1-open-v1 旧模型；其他版本必须按原契约回放")
            run = unified_dir(self.paths.root) / selected["run_id"]
            if any(not _file(run / name, run) for name in ("manifest.json", "normalizer.json", "model.zip")):
                raise ReplayError("unified 产物缺失或路径不安全", 404)
            meta = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if meta.get("execution_version", EXECUTION_LEGACY) != version:
                raise ReplayError("unified manifest 执行版本发生变化，请刷新模型列表")
            norm = NormalizerState.from_dict(json.loads((run / "normalizer.json").read_text(encoding="utf-8")))
        else:
            norm = self.normalizers.load(uid)
        features = norm.columns
        if (not features or len(set(features)) != len(features)
                or norm.mean.shape != (len(features),) or norm.std.shape != (len(features),)
                or not np.isfinite(norm.mean).all() or not np.isfinite(norm.std).all()
                or (norm.std <= 0).any()):
            raise ReplayError("冻结 norm 的 columns/mean/std shape 或有限值不合法", 500)
        if is_unified and version in STRICT_EXECUTION_VERSIONS:
            return self._run_strict(symbol, start, end, selected, meta, run, norm, execution_mode)
        engineer = None if is_unified else FeatureEngineer(features)
        df = _read_cache(raw_path, self.raw_dir)
        dates = df.date.dt.strftime("%Y%m%d").to_numpy()
        indices = np.flatnonzero((dates >= start) & (dates <= end))
        if not len(indices):
            raise ReplayError("所选区间没有缓存交易日", 404)
        if len(indices) > MAX_BARS:
            raise ReplayError(f"回放最多 {MAX_BARS} 个交易日", 400)
        model, window, count = _replay_model(run / "model.zip" if is_unified else self.paths.model_zip(uid), features)
        # 完整缓存先 transform，后裁区间；指标及滑窗均包含区间前的 warmup。
        if is_unified:
            from stock_ai.scale.unified import relative_features, normalize
            _, values = relative_features(df)
            transformed = normalize(values, norm)
            meta = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if meta["observation_shape"] != [window, count]:
                raise ReplayError("unified 模型 shape 与 manifest 不一致", 500)
        else:
            transformed = engineer.transform(df, norm)
        if not np.isfinite(transformed).all():
            raise ReplayError("冻结 transform 产生非有限值", 500)
        observations = []
        for i in indices:
            segment = transformed[max(0, i - window + 1):i + 1]
            if len(segment) < window:
                segment = np.concatenate([np.repeat(segment[:1], window - len(segment), axis=0), segment])
            observations.append(segment)
        market = df.iloc[indices].reset_index(drop=True)
        sample = SampleSplit(obs=np.asarray(observations), dates=dates[indices],
                             symbols=np.repeat(symbol, len(indices)), prices=market.close.to_numpy(),
                             requested_start=start, requested_end=end)
        warnings = ["同日完整行情 → 同日收盘（含滑点/费用）模拟成交，不等于实盘；不是次日开盘执行。",
                    "兼容既有 TradingEnv：成交限制使用其收盘价/代理成交量规则，图表 OHLCV 来自真实缓存；并非完整实盘撮合。",
                    "末日只按收盘估值，无新信号、无新执行、不强制平仓。",
                    "旧执行版本未结算完整分红送转等公司行为，不能将历史结果视为投资总回报。",
                    "年份筛选不保证全链路无泄漏：缓存复权口径、特征及产物来源仍需独立核验。"]
        if model_id != "auto":
            warnings.insert(0, "研究回放警告：手选现有模型，不能据此宣称样本外或无泄漏。")
        temporal_risk = selected["year"] >= int(start[:4])
        if temporal_risk:
            warnings.insert(0, f"前视/训练重叠风险：模型年份 {selected['year']} 不早于回放起始年 {start[:4]}（同年或未来模型）。")
        if is_unified:
            warnings.insert(0, "统一模型：使用冻结全局 normalizer 与 unified-relative-v1；所有缓存股票共享 PPO 参数。")
        elif selected["symbol"] != symbol:
            warnings.insert(0, "研究回放警告：模型训练代码与回放股票不同。")
        if indices[0] < max(window - 1, 20):
            warnings.append("缓存起点历史不足，指标使用已有历史、滑窗前端复制填充；没有联网补齐。")
        if is_unified:
            from stock_ai.config.schema import EnvConfig
            from stock_ai.scale.unified import UnifiedTradingEnv
            env_cfg = EnvConfig(**meta["env_config"])
            env = UnifiedTradingEnv(sample, env_cfg, seed=self.cfg.random_seed)
        else:
            env_cfg = self.cfg.env
            env = TradingEnv(sample, env_cfg, seed=self.cfg.random_seed)
        obs, _ = env.reset(seed=self.cfg.random_seed)
        daily = []
        try:
            for i in range(sample.n_steps):
                date = str(sample.dates[i])
                execution = i < sample.n_steps - 1
                if execution:
                    # 必须记录 step 前 idx/date；info.date 已经是下一日，不能用于图表。
                    execution_idx = env.idx
                    date = str(env.dates[execution_idx])
                    action = _predict_action(model, obs)
                    obs, _, _, _, info = env.step(action)
                else:
                    env.state.mark_to_market(float(sample.prices[i]))
                    info = {"action": None, "trade_ok": False,
                            "blocked_reason": "末日仅收盘估值，无新执行"}
                row = market.iloc[i]
                daily_equity = float(env.state.cash + env.state.shares * row.close) if is_unified else float(env.state.portfolio_value)
                daily_position = float(env.state.shares * row.close / max(daily_equity, 1e-9)) if is_unified else float(env.state.position_ratio)
                daily.append({"date": date, "idx": i, "execution": execution,
                              **{c: float(row[c]) for c in BASE_COLUMNS},
                              "action": info["action"], "trade_ok": bool(info["trade_ok"]),
                              "blocked_reason": info["blocked_reason"],
                              "cash": float(env.state.cash), "shares": int(env.state.shares),
                              "equity": daily_equity, "cost_basis": float(env.state.cost_basis),
                              "executed_action": info["action"] if execution else None,
                              "execution_signal_date": date if execution else None,
                              "pending_execution_date": None,
                              "signal_status": "same_day_simulated" if execution else "no_signal_final_day",
                              "position": daily_position})
            trades = [dict(t) for t in env.trades]
        finally:
            env.close()
        baseline = (Backtester._buy_and_hold(sample, env_cfg) if sample.n_steps > 1
                    else np.array([env_cfg.initial_cash], dtype="float64"))
        for row, value in zip(daily, baseline):
            row["baseline"] = float(value)
        from stock_ai.scale.unified import EXECUTION_LEGACY
        return {"ok": True, "symbol": symbol, "model": selected,
                "selection": "auto_strict_past" if model_id == "auto" else "manual_research",
                "execution_version": EXECUTION_LEGACY, "feature_adjust": selected.get("feature_adjust", "raw(as-cached)"),
                "source": {"feature_file": str(raw_path.resolve()), "execution_file": str(raw_path.resolve()),
                           "read_only": True, "normalizer": "frozen"},
                "temporal_risk": temporal_risk, "warnings": warnings,
                "requested_range": {"start": start, "end": end},
                "actual_range": {"start": daily[0]["date"], "end": daily[-1]["date"]},
                "warmup_rows": int(indices[0]), "window": window,
                "baseline_note": "首日买入持有，含滑点/买入费用，不含末日清仓费用；单日区间不执行。",
                "cost_note": _COST_NOTE,
                "initial_cash": float(env_cfg.initial_cash), "daily": daily, "trades": trades,
                "summary": _summarize(daily, trades, float(env_cfg.initial_cash))}

    def _manifest_cache_dir(self, meta: dict, key: str) -> Path:
        value = meta.get(key)
        if (not isinstance(value, str) or not value.strip() or "://" in value
                or value.replace("\\", "/").startswith("//") or "\x00" in value):
            raise ReplayError(f"strict manifest 的 {key} 缺失或不是安全本地目录", 500)
        directory = Path(value)
        if not directory.is_absolute():
            directory = self.paths.root / directory
        resolved = directory.resolve()
        configured = set()
        for path in (self.raw_dir, Path(self.cfg.data.cache_dir)):
            configured.add((path if path.is_absolute() else self.paths.root / path).resolve())
        if ((not resolved.is_relative_to(self.paths.root.resolve()) and resolved not in configured)
                or (key == "corporate_actions_dir" and not resolved.is_relative_to(self.paths.root.resolve()))
                or resolved.as_posix().startswith("//")):
            raise ReplayError(f"strict manifest 的 {key} 超出受控本地缓存目录", 500)
        if not resolved.is_dir():
            raise ReplayError(f"strict {key} 本地缓存目录缺失；不会联网或回退", 404)
        return resolved

    def _strict_source_audit(self, meta: dict, symbol: str, feature_sha256: str,
                             execution_sha256: str, ledger: Any, research: bool) -> dict:
        sources = meta.get("sources", [])
        if (not isinstance(sources, list) or (not research and not sources)
                or any(not isinstance(item, dict) for item in sources)):
            raise ReplayError("strict manifest 的 sources 缺失、为空或不是合法来源列表", 500)
        seen = set()
        for item in sources:
            source_symbol = item.get("symbol")
            if not isinstance(source_symbol, str) or not _SYMBOL.fullmatch(source_symbol):
                raise ReplayError("strict manifest 的 sources 含非法六位股票代码", 500)
            if source_symbol in seen:
                raise ReplayError(f"strict manifest 含重复 {source_symbol} 训练来源，无法核对冻结 SHA256", 500)
            seen.add(source_symbol)
        if not research:
            for item in sources:
                for key in ("sha256", "execution_sha256", "corporate_actions_sha256"):
                    if not isinstance(item.get(key), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", item[key]):
                        raise ReplayError(f"{item['symbol']} 冻结训练来源缺失或非法 {key}，禁止声称来源一致", 500)
        frozen = next((item for item in sources if item["symbol"] == symbol), None)
        current = {"feature_sha256": feature_sha256,
                   "execution_sha256": execution_sha256,
                   "corporate_actions_sha256": ledger.sha256}
        audit = {"status": "not_recorded" if research else "not_in_training_sources",
                 "symbol": symbol, "symbol_in_training_sources": frozen is not None}
        for name, frozen_key in (("feature", "sha256"), ("execution", "execution_sha256"),
                                 ("corporate_actions", "corporate_actions_sha256")):
            observed = current[name + "_sha256"]
            expected = frozen.get(frozen_key) if frozen is not None else None
            if research and name == "corporate_actions":
                # v1 未使用公司行为账本训练，当前审阅证据不能补成历史训练证据。
                status, expected = "current_reviewed_ledger_not_training_source", None
            elif frozen is None:
                status = "not_recorded" if research else "not_in_training_sources"
            elif expected is None and research:
                status = "not_recorded"
            else:
                if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                    raise ReplayError(f"{symbol} 冻结训练来源缺失或非法 {frozen_key}，禁止声称来源一致", 500)
                if observed != expected.lower():
                    raise ReplayError(f"{symbol} {name} 完整文件 SHA256 与冻结训练来源不一致，禁止回放", 500)
                status = "matched"
            audit[name] = {"status": status, "current_sha256": observed, "training_sha256": expected}
        if frozen is not None:
            statuses = [audit[name]["status"] for name in ("feature", "execution")]
            if not research:
                audit["status"] = "matched"
            else:
                audit["status"] = ("matched" if all(status == "matched" for status in statuses)
                                   else "not_recorded" if all(status == "not_recorded" for status in statuses)
                                   else "partial")
        return {**current, "training_source_audit": audit}

    def _run_strict(self, symbol: str, start: str, end: str, selected: dict,
                    meta: dict, run: Path, norm: Any, execution_mode: str = "default") -> dict:
        from stock_ai.config.schema import EnvConfig
        from stock_ai.env.strict_env import ExecutionBars
        from stock_ai.scale.unified import (EXECUTION_CORPORATE_ACTIONS, FEATURES, make_strict_env,
                                            normalize, observations, relative_features)

        training_version = selected["execution_version"]
        research = execution_mode == "corporate_actions_research"
        execution_version = EXECUTION_CORPORATE_ACTIONS if research else training_version
        with_actions = execution_version == EXECUTION_CORPORATE_ACTIONS
        ledger = None
        if with_actions:
            from stock_ai.data.corporate_actions import CorporateActionError, load_ledger
            # 研究覆写只使用固定账本目录，不接受请求路径，也不改写模型 manifest。
            ledger_meta = {"corporate_actions_dir": "data/corporate_actions"} if research else meta
            ledger_dir = self._manifest_cache_dir(ledger_meta, "corporate_actions_dir")
            ledger_path = ledger_dir / f"{symbol}.json"
            if not _file(ledger_path, ledger_dir):
                raise ReplayError("公司行为账本缺失或路径不安全；禁止回退 v1 或假定无事件", 404)
            try:
                ledger = load_ledger(ledger_path, symbol=symbol)
                # 单日回放也不能绕过覆盖准入，且不把请求区间悄然收缩为缓存首尾。
                ledger.require_coverage(start, end)
            except CorporateActionError as exc:
                raise ReplayError(f"公司行为账本校验失败：{exc}") from exc
        if norm.columns != FEATURES or meta.get("feature_adjust") != "hfq":
            raise ReplayError("strict 必须使用 hfq 相对特征与冻结 unified normalizer", 500)
        feature_dir = self._manifest_cache_dir(meta, "feature_dir")
        execution_dir = self._manifest_cache_dir(meta, "execution_dir")
        if feature_dir == execution_dir:
            raise ReplayError("strict 后复权特征与未复权执行缓存不能是同一目录", 500)
        feature_path, execution_path = feature_dir / f"{symbol}.parquet", execution_dir / f"{symbol}.parquet"
        feature_df, feature_sha256 = _read_cache(feature_path, feature_dir, "strict 后复权 hfq 特征缓存", with_sha256=True)
        raw, execution_sha256 = _read_cache(execution_path, execution_dir, "strict 未复权真实执行缓存", with_sha256=True)
        source_audit = (self._strict_source_audit(meta, symbol, feature_sha256, execution_sha256, ledger, research)
                        if with_actions else {})
        # 必须与训练顺序一致：先在完整 hfq 历史算指标，再过滤真实交易日，最后冻结归一化及滑窗。
        feature_df, values = relative_features(feature_df)
        feature_dates = feature_df.date.dt.strftime("%Y%m%d").to_numpy()
        raw_dates = raw.date.dt.strftime("%Y%m%d").to_numpy()
        feature_requested = set(feature_dates[(feature_dates >= start) & (feature_dates <= end)])
        raw_requested = set(raw_dates[(raw_dates >= start) & (raw_dates <= end)])
        if feature_requested != raw_requested:
            missing_feature = sorted(raw_requested - feature_requested)
            missing_execution = sorted(feature_requested - raw_requested)
            raise ReplayError("strict 特征与真实执行缓存日期不齐；禁止丢日或回退"
                              f"（缺 hfq: {missing_feature[:5]}；缺未复权: {missing_execution[:5]}）", 500)
        if not raw_requested:
            raise ReplayError("所选区间没有缓存交易日", 404)
        if len(raw_requested) > MAX_BARS:
            raise ReplayError(f"回放最多 {MAX_BARS} 个交易日", 400)
        keep = feature_df.date.isin(raw.date).to_numpy()
        filtered_history = int((~keep & (feature_dates < start)).sum())
        dates = feature_dates[keep]
        transformed = normalize(values[keep], norm)
        if not np.isfinite(transformed).all():
            raise ReplayError("strict 冻结 transform 产生非有限值", 500)
        model, window, count = _replay_model(run / "model.zip", norm.columns)
        if meta.get("observation_shape") != [window, count] or meta.get("window") != window:
            raise ReplayError("unified 模型 shape 与 manifest 不一致", 500)
        indices = np.flatnonzero((dates >= start) & (dates <= end))
        obs_values = observations(transformed, window)[indices]
        actual_dates = dates[indices]
        market = raw.set_index(raw_dates).loc[actual_dates].reset_index(drop=True)
        bars = ExecutionBars(dates=actual_dates, symbol=symbol, corporate_actions=ledger,
                             **{key: market[key].to_numpy(dtype="float64") for key in BASE_COLUMNS})
        env_cfg = EnvConfig(**meta["env_config"])
        initial_cash = float(env_cfg.initial_cash)
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ReplayError("strict manifest 初始现金必须为有限正数", 500)
        temporal_risk = selected["year"] >= int(start[:4])
        warnings = ["统一模型：使用冻结全局 normalizer 与 unified-relative-v1；后复权特征仅用于观测，未复权真实价用于执行及估值。",
                    "研究回放警告：手选现有模型，不能据此宣称样本外或无泄漏。",
                    "T 日收盘生成信号，T+1 交易日真实开盘按模型记录的严格执行版本撮合；不是同日收盘成交，也不是实盘撮合。",
                    "首日空仓按初始现金估值；末日执行前日信号但不产生新信号，不强制平仓。",
                    "回放区间连续跨年，不按年度重置账户；与训练及存量评估的逐股票年度 episode 不同。",
                    "严格基准独立按同环境每日买入信号模拟，首点为初始现金；不复用旧 buy_and_hold，可能与存量 metrics 的初始点、基准及收益口径不同。",
                    "ST 名单未联网获取，按非 ST 处理；新股上市无涨跌幅限制未完整建模，缓存来源与复权口径仍需独立核验。"]
        if with_actions:
            warnings.append("公司行为 v2：仅结算显式 verified 覆盖账本中的受支持分红送转；未知或未覆盖事件拒绝回放。权益含税前股息应收与待上市锁股估值，不代表全部可用资金；个人差异化股息税未计。")
            audit = source_audit["training_source_audit"]
            if research:
                warnings.append("研究执行来源：公司行为账本为当前显式审阅账本，不是 v1 训练来源；缺失的历史文件指纹标记为 not_recorded，不能据当前 SHA256 补证历史一致。")
            elif audit["status"] == "matched":
                warnings.append("本股票 hfq / 未复权 / 公司行为账本完整文件 SHA256 均与 manifest 冻结训练来源匹配；这不单独证明无泄漏。")
            else:
                warnings.append("本股票未列入该模型冻结训练 sources；仅披露当前 hfq / 未复权 / 公司行为完整文件 SHA256，不宣称与训练来源一致。")
        else:
            warnings.append("旧 v1 遗漏分红送转等公司行为：未复权价格除权下跌未补权益，训练奖励及历史收益存在缺陷；本结果不能视为投资总回报。")
        if research:
            warnings.insert(0, "强警告：公司行为修正研究回放仅把旧 v1 policy 放到 v2 环境执行；旧 policy 仍由遗漏公司行为的错误 v1 奖励训练，不是已修复并重训的模型。模型、冻结 normalizer 与 manifest 均未修改。")
        if temporal_risk:
            warnings.insert(0, f"前视/训练重叠风险：训练截至 {selected['year']} 年，不早于回放起始年 {start[:4]}。")
        if indices[0] < max(window - 1, 20):
            warnings.append("缓存起点历史不足，指标使用已有历史、滑窗前端复制填充；没有联网补齐。")
        if filtered_history:
            warnings.append(f"区间外历史按真实执行日过滤了 {filtered_history} 行 hfq；指标先在完整 hfq 历史计算，与训练顺序一致。")
        daily, trades, baseline, action_log = [], [], [initial_cash], []
        if len(indices) == 1:
            row = market.iloc[0]
            snapshot = {"date": str(actual_dates[0]), "idx": 0, "execution": False,
                        **{key: float(row[key]) for key in BASE_COLUMNS},
                        "action": None, "executed_action": None, "execution_signal_date": None,
                        "pending_execution_date": None, "signal_status": "no_signal_final_day",
                        "trade_ok": False, "blocked_reason": "单日区间仅收盘估值，无信号、无执行",
                        "cash": initial_cash, "shares": 0, "cost_basis": 0.0, "equity": initial_cash}
            if with_actions:
                from stock_ai.env.corporate_action_env import CorporateActionState
                state = CorporateActionState(cash=initial_cash, prev_close=float(row.close))
                state.mark_to_market(float(row.close))
                snapshot.update(state.snapshot(), equity=float(state.portfolio_value),
                                corporate_actions=[], limit_reference=None, limit_up=None, limit_down=None)
            daily.append(snapshot)
        else:
            env = make_strict_env(obs_values, bars, env_cfg, execution_version=execution_version,
                                  seed=self.cfg.random_seed)
            benchmark = None
            try:
                if not with_actions:
                    benchmark = make_strict_env(obs_values, bars, env_cfg, execution_version=execution_version,
                                                seed=self.cfg.random_seed)
                    benchmark.reset(seed=self.cfg.random_seed)
                obs, info = env.reset(seed=self.cfg.random_seed)
                info.update(trade_ok=False, blocked_reason="首日空仓，仅生成收盘信号，待下一交易日执行")
                previous_action = None
                for i, date in enumerate(actual_dates):
                    row = market.iloc[i]
                    # 在产生下一日动作前保存当日收盘状态；v2 应收必须由状态估值而非手算现金加市值。
                    env.state.mark_to_market(float(row.close))
                    account = env.state.snapshot()
                    snapshot = {"date": str(date), "idx": i, "execution": i > 0,
                                **{key: float(row[key]) for key in BASE_COLUMNS},
                                "executed_action": previous_action,
                                "execution_signal_date": str(actual_dates[i - 1]) if i else None,
                                "trade_ok": bool(info["trade_ok"]), "blocked_reason": info["blocked_reason"],
                                "cash": account["cash"], "shares": account["shares"],
                                "cost_basis": account["cost_basis"], "equity": account["portfolio_value"]}
                    if with_actions:
                        snapshot.update(account, corporate_actions=list(info.get("corporate_actions", [])),
                                        limit_reference=info.get("limit_reference"),
                                        limit_up=info.get("limit_up"), limit_down=info.get("limit_down"))
                    action = _predict_action(model, obs) if i < len(indices) - 1 else None
                    snapshot.update(action=action,
                                    pending_execution_date=str(actual_dates[i + 1]) if action is not None else None,
                                    signal_status="pending_next_open" if action is not None else "no_signal_final_day")
                    daily.append(snapshot)
                    if action is not None:
                        obs, _, _, _, info = env.step(action)
                        previous_action = action
                        if benchmark is not None:
                            benchmark.step(1)
                            baseline.append(float(benchmark.state.portfolio_value))
                trades = [dict(trade) for trade in env.trades]
                if with_actions:
                    baseline = env.benchmark_equity.tolist()
                    action_log = env.corporate_action_log
            finally:
                env.close()
                if benchmark is not None:
                    benchmark.close()
        for row, value in zip(daily, baseline):
            row["baseline"] = float(value)
        result = {"ok": True, "symbol": symbol, "model": selected,
                  "selection": "corporate_actions_research" if research else "manual_research",
                  "execution_version": execution_version, "feature_adjust": "hfq",
                  "execution_adjust": meta.get("execution_adjust", "none(真实未复权价)"),
                  "reward_type": selected.get("reward_type"), "reward_version": selected.get("reward_version"),
                  "reward_params": selected.get("reward_params"),
                  "source": {"feature_file": str(feature_path.resolve()), "execution_file": str(execution_path.resolve()),
                             "feature_dir": str(feature_dir), "execution_dir": str(execution_dir),
                             "feature_adjust": "hfq", "execution_adjust": "none",
                             "read_only": True, "normalizer": "frozen", "filtered_history_rows": filtered_history},
                  "temporal_risk": temporal_risk, "warnings": warnings,
                  "requested_range": {"start": start, "end": end},
                  "actual_range": {"start": daily[0]["date"], "end": daily[-1]["date"]},
                  "warmup_rows": int(indices[0]), "window": window,
                  "baseline_note": "同版本严格环境独立恒买入保持：首日空仓初始现金，T 日信号在 T+1 开盘执行，遵守涨跌停/停牌/整手/费用/滑点；末日不清仓。存量 metrics 基准可能口径不同。",
                  "cost_note": _COST_NOTE, "initial_cash": initial_cash, "daily": daily, "trades": trades,
                  "summary": _summarize(daily, trades, initial_cash)}
        if with_actions:
            from stock_ai.env.corporate_action_env import REWARD_VERSION_EXCESS
            tax_policy = "gross_before_personal_dividend_tax"
            result.update(corporate_actions=action_log, corporate_actions_ledger=ledger.to_dict(),
                          training_execution_version=training_version, execution_mode=execution_mode,
                          tax_policy=tax_policy, training_reward_version=selected.get("reward_version"))
            if str(env_cfg.reward_type).lower() == "excess":
                result["reward_version"] = REWARD_VERSION_EXCESS
            result["source"].update(source_audit)
            result["source"].update(corporate_actions_dir=str(ledger_dir),
                                    corporate_actions_file=ledger.source_path,
                                    corporate_actions_sha256=ledger.sha256,
                                    corporate_actions_schema=ledger.schema,
                                    corporate_actions_coverage=ledger.coverage, tax_policy=tax_policy)
            result["baseline_note"] += " 公司行为基准与策略使用同一已核验账本，均包含税前股息应收、派息和送转股份上市结算。"
            result["cost_note"] += " v2 总权益=现金+全部持股市值+股息应收；浮动盈亏只计股价相对摊薄成本变化，不含单列股息累计。待上市股份计入估值但不可卖。"
        return result

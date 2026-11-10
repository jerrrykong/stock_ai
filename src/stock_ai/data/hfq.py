"""后复权（hfq）数据层：独立缓存目录 + 节流容错拉取 + 强校验 + 除权跳空修复统计。

为什么需要独立目录
------------------
后复权价**只能用于特征与训练信号**，成交执行必须用真实未复权价。因此
hfq 数据绝不写入既有的 ``data/raw``（不复权缓存），而是落到
``data/raw_hfq``，由 :mod:`stock_ai.scale.unified` 同时读入两套价格。

已实测的源特性（决定了本模块的默认策略）
-----------------------------------------
- ``akshare`` / ``eastmoney``：``hfq`` 返回**全历史**且不会出现负价 —— 可用；
- ``akshare`` / ``eastmoney`` 的 ``qfq`` 只返回近约 10 年（≈2300 行），覆盖不足 —— 不可用；
- ``sina`` 接口**不支持复权**（``SinaSource.fetch`` 在 ``adjust != none`` 时直接抛错）；
- ``tencent`` 支持 hfq 但存在窗口速率风控（连续快速拉约 150 只后返回 501 WAF 页）。

因此 :data:`HFQ_SOURCES` 默认只含 ``akshare`` / ``eastmoney``，``tencent`` 需显式
``--source tencent`` 才启用，且必须配合节流。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stock_ai.data.sources.akshare_source import AkShareSource
from stock_ai.data.sources.eastmoney_source import EastMoneySource
from stock_ai.data.sources.tencent_source import TencentSource
from stock_ai.utils.errors import DataSourceError
from stock_ai.utils.io import write_json
from stock_ai.utils.logging import LoggerFactory

__all__ = [
    "HFQ_DIR_NAME",
    "HFQ_SOURCES",
    "GAP_RETURN_THRESHOLD",
    "HfqReport",
    "hfq_cache_dir",
    "make_hfq_source",
    "fetch_hfq_symbol",
    "validate_hfq_frame",
    "gap_repair_stats",
    "compare_adjust",
    "execution_prices_from_hfq",
    "fetch_hfq_all",
    "read_progress",
]

logger = LoggerFactory.get(__name__)

#: hfq 缓存目录名（相对 ``data/``）；**绝不**复用 ``raw``（不复权）。
HFQ_DIR_NAME: str = "raw_hfq"

#: hfq 可用源（按优先级）。``sina`` 不支持复权；``tencent`` 有 WAF 风险需显式启用。
HFQ_SOURCES: tuple[str, ...] = ("akshare", "eastmoney")

#: 除权跳空阈值：A 股日涨跌幅绝对值超过 11% 即视为"非正常波动"（除权/除息跳空）。
GAP_RETURN_THRESHOLD: float = 0.11

_PROGRESS_NAME = "fetch_progress.json"


def hfq_cache_dir(root: Any) -> Path:
    """返回 hfq 缓存目录 ``<root>/data/raw_hfq`` 并确保存在。

    Args:
        root: 工程根目录（通常为 ``AppConfig.paths.root``）。

    Returns:
        已创建的目录路径。
    """
    path = Path(root) / "data" / HFQ_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress_path(directory: Path) -> Path:
    """进度文件路径（进度可查，不依赖进程存活）。"""
    return Path(directory) / _PROGRESS_NAME


def read_progress(root: Any) -> dict:
    """读取 hfq 拉取进度（不存在返回空进度）。"""
    path = _progress_path(hfq_cache_dir(root))
    if not path.exists():
        return {"total": 0, "done": 0, "failed": 0, "symbols": [], "in_progress": False}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"total": 0, "done": 0, "failed": 0, "symbols": [], "in_progress": False}


def make_hfq_source(name: str, retry_times: int = 3, retry_backoff: float = 1.5,
                    timeout: float = 15.0):
    """按名字构造 hfq 数据源（``adjust`` 固定为 ``hfq``）。

    Args:
        name: ``akshare`` / ``eastmoney`` / ``tencent``。
        retry_times: 重试次数。
        retry_backoff: 退避基数（秒）。
        timeout: 超时（秒）。

    Returns:
        配置为 ``adjust="hfq"`` 的数据源实例。

    Raises:
        DataSourceError: 源名不在 :data:`HFQ_SOURCES` + ``tencent`` 之内。
    """
    factories = {
        "akshare": AkShareSource,
        "eastmoney": EastMoneySource,
        "tencent": TencentSource,
    }
    if name not in factories:
        raise DataSourceError(
            f"hfq 不支持的数据源: {name!r}（可用: {sorted(factories)}；"
            f"sina 接口不支持复权）",
            context={"source": name},
        )
    return factories[name](adjust="hfq", retry_times=retry_times,
                           retry_backoff=retry_backoff, timeout=timeout)


# --------------------------------------------------------------------------- #
# 强校验
# --------------------------------------------------------------------------- #
def validate_hfq_frame(df: "Any", symbol: str = "", min_rows: int = 2,
                       max_bad_ratio: float = 0.0) -> "Any":
    """校验后复权日线：**非正价格直接抛错**，绝不静默剔除。

    与 :func:`stock_ai.data.schema.validate_frame` 的差别在于：后者会剔除
    非正价格行（对前复权早年负价是合理的降级），而后复权数据出现非正价格
    一定是数据错误，必须显式失败而不是悄悄丢历史。

    实测存在极少数「源端脏数据」案例（如 ``000408`` 在 2006-08-08 单行
    ``low=-0.18``，且 hfq/raw 累乘比值非单调），并非复权算法本身有问题。
    对此类**极少数**异常行，调用方可显式传入 ``max_bad_ratio > 0`` 允许
    剔除并在报告中记录；但默认仍为 0.0，即保持严格失败。

    Args:
        df: 待校验的 DataFrame。
        symbol: 股票代码（仅用于报错信息）。
        min_rows: 最小行数。
        max_bad_ratio: 允许剔除的异常行占比上限（0=严格，>0 为显式降级）。

    Returns:
        规范化后的 DataFrame（列顺序固定、日期升序去重）。

    Raises:
        DataSourceError: 缺列、空数据、非正/非有限价格、行数不足。
    """
    import numpy as np
    import pandas as pd

    if df is None or len(df) == 0:
        raise DataSourceError(f"hfq 数据为空: {symbol or '?'}", context={"symbol": symbol})
    required = ("date", "open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataSourceError(
            f"hfq 数据缺少列 {missing}: {symbol or '?'}", context={"symbol": symbol}
        )
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"].astype(str), errors="coerce")
    if out["date"].isna().any():
        raise DataSourceError(f"hfq 数据存在非法日期: {symbol or '?'}",
                              context={"symbol": symbol})
    prices = out[["open", "high", "low", "close"]].to_numpy(dtype="float64")
    if not np.isfinite(prices).all():
        raise DataSourceError(f"hfq 价格含 NaN/Inf: {symbol or '?'}",
                              context={"symbol": symbol})
    if (prices <= 0).any():
        bad_rows = np.argwhere((prices <= 0).any(axis=1))[:, 0]
        ratio = len(bad_rows) / max(len(out), 1)
        row = int(bad_rows[0])
        if float(max_bad_ratio) <= 0.0:
            raise DataSourceError(
                f"hfq 出现非正价格（后复权不应为负）: {symbol or '?'} "
                f"第 {row + 1} 行 {out['date'].iloc[row].date()} "
                f"open={out['open'].iloc[row]} high={out['high'].iloc[row]} "
                f"low={out['low'].iloc[row]} close={out['close'].iloc[row]}",
                context={"symbol": symbol, "row": row + 1},
            )
        if ratio > float(max_bad_ratio):
            raise DataSourceError(
                f"hfq 非正价格占比 {ratio:.4%} 超过允许上限 {float(max_bad_ratio):.4%}: "
                f"{symbol or '?'}（{len(bad_rows)}/{len(out)} 行）",
                context={"symbol": symbol, "bad_rows": len(bad_rows), "total": len(out)},
            )
        # 显式降级：仅剔除这些行，且必须保证剔除后仍有足够样本
        keep = ~((out[["open", "high", "low", "close"]].to_numpy(dtype="float64")) <= 0).any(axis=1)
        out = out[keep].reset_index(drop=True)
        if len(out) < int(min_rows):
            raise DataSourceError(
                f"hfq 剔除异常行后行数不足: {symbol or '?'} 仅 {len(out)} 行",
                context={"symbol": symbol, "rows": len(out)},
            )
    out = out.sort_values("date", kind="stable")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if len(out) < int(min_rows):
        raise DataSourceError(
            f"hfq 行数不足: {symbol or '?'} 仅 {len(out)} 行（要求 >= {min_rows}）",
            context={"symbol": symbol, "rows": len(out)},
        )
    return out


@dataclass
class HfqReport:
    """单只股票的后复权校验结果。"""

    symbol: str
    rows: int
    start: str
    end: str
    first_close: float
    last_close: float
    min_price: float
    max_price: float
    gaps_gt_threshold: int
    max_abs_return: float
    source: str = ""
    adjusted_gaps: "int | None" = None
    raw_gaps: "int | None" = None
    execution_overlap: int = 0
    notes: list = field(default_factory=list)
    #: 复权因子单调性是否通过；``None`` 表示缺少不复权数据可比对。
    consistent: "bool | None" = None

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的 dict。"""
        return {
            "symbol": self.symbol, "rows": self.rows, "start": self.start,
            "end": self.end, "first_close": self.first_close,
            "last_close": self.last_close, "min_price": self.min_price,
            "max_price": self.max_price, "gaps_gt_threshold": self.gaps_gt_threshold,
            "max_abs_return": self.max_abs_return, "source": self.source,
            "raw_gaps": self.raw_gaps, "adjusted_gaps": self.adjusted_gaps,
            "execution_overlap": self.execution_overlap, "notes": list(self.notes),
            "consistent": self.consistent,
        }


def gap_repair_stats(df: "Any", threshold: float = GAP_RETURN_THRESHOLD) -> tuple[int, float]:
    """统计 ``|日收益| > threshold`` 的次数与最大绝对日收益（除权跳空代理指标）。

    Args:
        df: 日线 DataFrame（需含 ``close``）。
        threshold: 阈值（默认 :data:`GAP_RETURN_THRESHOLD` = 0.11）。

    Returns:
        ``(超过阈值的次数, 最大绝对日收益)``。
    """
    import numpy as np

    if df is None or len(df) < 2:
        return 0, 0.0
    close = np.asarray(df["close"], dtype="float64")
    ret = close[1:] / close[:-1] - 1.0
    ret = ret[np.isfinite(ret)]
    if ret.size == 0:
        return 0, 0.0
    return int((np.abs(ret) > float(threshold)).sum()), float(np.abs(ret).max())


def hfq_consistency(hfq_df: "Any", raw_df: "Any",
                    diff_threshold: float = 0.05,
                    max_diff_ratio: float = 0.01) -> dict:
    """校验后复权是否真正修复了除权跳空（比较**收益序列**，而非跨源价格）。

    后复权应满足：复权日收益与不复权日收益的差异**只集中在除权日**，
    因此绝大多数交易日的两者收益几乎相同（实测正常股票差异>5% 的天数
    仅占 0.07%~0.08%）。若数据源的复权因子本身有问题（如 ``000408``
    在 2006 年），差异会扩散到 2%+，且复权后波动反而放大。

    注意：**不能**直接比较 ``hfq/raw`` 的价格比值。hfq 来自 akshare/eastmoney，
    而不复权缓存来自 sina，两者复权基准日不同且各自取整，比值天然带噪声，
    会产生大量假阳性（实测逐日比值有 250/300 被误判）。

    Args:
        hfq_df: 后复权 DataFrame（需含 ``date``/``close``）。
        raw_df: 不复权 DataFrame（需含 ``date``/``close``）。
        diff_threshold: 判定「收益显著不同」的阈值，默认 5%。
        max_diff_ratio: 允许的显著不同天数占比上限，默认 1%。

    Returns:
        ``{"ok": bool, "diff_ratio": float, "diff_days": int, "reason": str}``。
    """
    import numpy as np
    import pandas as pd

    if hfq_df is None or raw_df is None or len(hfq_df) < 2 or len(raw_df) < 2:
        return {"ok": True, "diff_ratio": 0.0, "diff_days": 0, "reason": "数据不足，跳过"}
    a = pd.DataFrame({"date": pd.to_datetime(hfq_df["date"].astype(str)),
                      "adj": np.asarray(hfq_df["close"], dtype="float64")})
    b = pd.DataFrame({"date": pd.to_datetime(raw_df["date"].astype(str)),
                      "raw": np.asarray(raw_df["close"], dtype="float64")})
    m = a.merge(b, on="date").sort_values("date", kind="stable")
    if len(m) < 3:
        return {"ok": True, "diff_ratio": 0.0, "diff_days": 0, "reason": "共同交易日不足"}
    rh = m["adj"].pct_change()
    rr = m["raw"].pct_change()
    valid = rh.notna() & rr.notna()
    if int(valid.sum()) < 3:
        return {"ok": True, "diff_ratio": 0.0, "diff_days": 0, "reason": "可比样本不足"}
    diff_days = int(((rh - rr).abs()[valid] > float(diff_threshold)).sum())
    ratio = diff_days / int(valid.sum())
    ok = ratio <= float(max_diff_ratio)
    return {
        "ok": ok,
        "diff_ratio": float(ratio),
        "diff_days": diff_days,
        "reason": f"收益差异仅 {diff_days}/{int(valid.sum())} 天（{ratio:.2%}），复权正常"
                  if ok else
                  f"收益差异扩散到 {diff_days}/{int(valid.sum())} 天（{ratio:.2%}），复权因子可疑",
    }


def compare_adjust(hfq_df: "Any", raw_df: "Any" = None,
                   threshold: float = GAP_RETURN_THRESHOLD) -> dict:
    """对比复权前后的大幅跳空次数（复权跳空修复统计）。

    后复权除掉了除权除息造成的向下跳空，因此正常情况下
    ``adjusted_gaps <= raw_gaps``。

    Args:
        hfq_df: 后复权 DataFrame。
        raw_df: 不复权 DataFrame（可选；缺失则只报后复权侧）。
        threshold: 跳空阈值。

    Returns:
        ``{"adjusted_gaps": int, "raw_gaps": int | None, "max_abs_return": float,
        "raw_max_abs_return": float | None}``。
    """
    adjusted, adj_max = gap_repair_stats(hfq_df, threshold)
    out = {"adjusted_gaps": adjusted, "raw_gaps": None,
           "max_abs_return": adj_max, "raw_max_abs_return": None}
    if raw_df is not None and len(raw_df):
        raw, raw_max = gap_repair_stats(raw_df, threshold)
        out["raw_gaps"] = raw
        out["raw_max_abs_return"] = raw_max
    return out


def execution_prices_from_hfq(hfq_df: "Any", raw_df: "Any") -> "Any":
    """用不复权缓存给出**真实成交价**，并与 hfq 按日期对齐。

    后复权价只用于特征；撮合必须用真实价。真实价来自 ``raw_df``（不复权），
    仅保留 hfq 中存在的交易日（保证特征与成交时序一一对应）。

    Args:
        hfq_df: 后复权 DataFrame。
        raw_df: 不复权 DataFrame。

    Returns:
        与 hfq 对齐的不复权 OHLCV（行数相同、顺序相同）。

    Raises:
        DataSourceError: 对齐后无可用真实价。
    """
    import pandas as pd

    if raw_df is None or len(raw_df) == 0:
        raise DataSourceError("缺少不复权缓存，无法推导真实成交价")
    keys = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in keys if c not in raw_df.columns]
    if missing:
        raise DataSourceError(f"不复权缓存缺少列 {missing}")
    right = raw_df[keys].copy()
    right["date"] = pd.to_datetime(right["date"].astype(str))
    left = pd.DataFrame({"date": pd.to_datetime(hfq_df["date"].astype(str)).to_numpy()})
    aligned = left.merge(right, on="date", how="inner")
    aligned = aligned.sort_values("date", kind="stable").reset_index(drop=True)
    if len(aligned) == 0:
        raise DataSourceError("hfq 与不复权缓存没有共同交易日")
    return aligned[keys]


# --------------------------------------------------------------------------- #
# 拉取
# --------------------------------------------------------------------------- #
def fetch_hfq_symbol(
    symbol: str,
    start: str,
    end: str,
    sources: "list[str] | None" = None,
    retry_times: int = 3,
    retry_backoff: float = 1.5,
    timeout: float = 15.0,
    max_bad_ratio: float = 0.0,
) -> tuple["Any", str]:
    """拉取单只股票的后复权日线（多源 fallback，逐源强校验）。

    Args:
        symbol: 股票代码。
        start: 起始 ``YYYYMMDD``。
        end: 结束 ``YYYYMMDD``。
        sources: 源名列表（默认 :data:`HFQ_SOURCES`）。
        retry_times: 每源重试次数。
        retry_backoff: 退避基数。
        timeout: 超时。
        max_bad_ratio: 允许剔除的源端异常行占比上限（0=严格失败）。

    Returns:
        ``(已校验的 DataFrame, 实际生效的源名)``。

    Raises:
        DataSourceError: 全部源失败或全部源数据未通过强校验。
    """
    errors: list[str] = []
    for name in list(sources or HFQ_SOURCES):
        source = make_hfq_source(name, retry_times, retry_backoff, timeout)
        try:
            raw = source.fetch(symbol, start, end)
            return validate_hfq_frame(raw, symbol, max_bad_ratio=max_bad_ratio), name
        except Exception as exc:  # noqa: BLE001 - fallback 需吞掉所有源异常
            errors.append(f"{name}: {exc}")
            logger.warning("hfq 源 %s 失败 %s: %s", name, symbol, exc)
    raise DataSourceError(
        f"hfq 拉取失败: {symbol} ({start}~{end})",
        context={"symbol": symbol, "errors": errors},
    )


def fetch_hfq_all(
    root: Any,
    symbols: "list[str]",
    start: str,
    end: str,
    sources: "list[str] | None" = None,
    throttle: float = 3.0,
    batch: int = 30,
    rest: float = 90.0,
    max_fail_streak: int = 8,
    cooldown: float = 600.0,
    force: bool = False,
    raw_dir: "Path | None" = None,
    max_bad_ratio: float = 0.0,
    strict_consistency: bool = False,
) -> dict:
    """限速 + 逐只容错地批量拉取 hfq 数据到 ``data/raw_hfq``。

    - 逐只串行，``throttle`` 控制每只间隔，``batch``/``rest`` 控制批间休息；
    - 单只失败**不中断**整批，失败代码写入 ``fetch_failed.txt``；
    - 连续失败达 ``max_fail_streak`` 判定触发风控，冷却 ``cooldown`` 秒后继续；
    - 每只拉完立即写 ``fetch_progress.json``，进度可查；
    - ``raw_dir`` 存在时附带输出除权跳空修复统计（对比不复权缓存）。

    Args:
        root: 工程根目录。
        symbols: 股票代码列表。
        start: 起始 ``YYYYMMDD``。
        end: 结束 ``YYYYMMDD``。
        sources: 源名列表。
        throttle: 每只间隔秒。
        batch: 批大小。
        rest: 批间休息秒。
        max_fail_streak: 连续失败阈值。
        cooldown: 风控冷却秒。
        force: 忽略已存在的缓存文件。
        raw_dir: 不复权缓存目录（用于跳空修复统计，可为 None）。

    Returns:
        ``{"ok": [...], "failed": [...], "reports": [...]}``。
    """
    import pandas as pd

    directory = hfq_cache_dir(root)
    raw_path = Path(raw_dir) if raw_dir is not None else None
    ok: list[str] = []
    failed: list[str] = []
    #: 复权有效性校验未通过但已保留的代码（源端因子异常，建议训练时排除）
    inconsistent: list[str] = []
    reports: list[HfqReport] = []
    streak = 0
    syms = [str(s) for s in symbols]
    total = len(syms)
    write_json({"total": total, "done": 0, "failed": 0, "symbols": [], "in_progress": True,
                "start": start, "end": end, "sources": list(sources or HFQ_SOURCES)},
               _progress_path(directory))
    for index, symbol in enumerate(syms, 1):
        target = directory / f"{symbol}.parquet"
        try:
            if force or not target.exists():
                df, source_name = fetch_hfq_symbol(symbol, start, end, sources,
                                                  max_bad_ratio=max_bad_ratio)
                df.to_parquet(str(target), index=False, engine="pyarrow")
            else:
                df = pd.read_parquet(str(target))
                source_name = "cached"
            df = validate_hfq_frame(df, symbol, max_bad_ratio=max_bad_ratio)
            adjusted, adj_max = gap_repair_stats(df)
            raw_gaps, overlap, notes = None, 0, []
            consistent = None
            if raw_path is not None and (raw_path / f"{symbol}.parquet").exists():
                raw_df = pd.read_parquet(str(raw_path / f"{symbol}.parquet"))
                raw_gaps = gap_repair_stats(raw_df)[0]
                try:
                    overlap = len(execution_prices_from_hfq(df, raw_df))
                except DataSourceError as exc:
                    notes.append(f"未复权对齐失败: {exc}")
                # 复权有效性：源端因子异常时复权数据比不复权更糟，必须显式隔离
                consistent = hfq_consistency(df, raw_df)
                if not consistent["ok"]:
                    notes.append(f"复权异常（隔离）: {consistent['reason']}")
                    inconsistent.append(symbol)
                    if strict_consistency:
                        raise DataSourceError(
                            f"复权有效性校验失败: {symbol} — {consistent['reason']}",
                            context={"symbol": symbol, **consistent},
                        )
            if source_name == "cached" and force is False and not target.stat().st_size:
                notes.append("空文件")
            reports.append(HfqReport(
                symbol=symbol, rows=len(df),
                start=df["date"].iloc[0].strftime("%Y%m%d"),
                end=df["date"].iloc[-1].strftime("%Y%m%d"),
                first_close=float(df["close"].iloc[0]), last_close=float(df["close"].iloc[-1]),
                min_price=float(df[["open", "high", "low", "close"]].to_numpy().min()),
                max_price=float(df[["open", "high", "low", "close"]].to_numpy().max()),
                gaps_gt_threshold=adjusted, max_abs_return=adj_max, source=source_name,
                raw_gaps=raw_gaps, adjusted_gaps=adjusted, execution_overlap=overlap,
                notes=notes, consistent=None if consistent is None else bool(consistent["ok"]),
            ))
            ok.append(symbol)
            streak = 0
            flag = " [复权异常已隔离]" if consistent is not None and not consistent["ok"] else ""
            print(f"[hfq {index}/{total}] OK   {symbol} [{source_name}] {len(df)} rows{flag}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - 逐只容错，绝不中断整批
            failed.append(symbol)
            streak += 1
            print(f"[hfq {index}/{total}] FAIL {symbol}: {str(exc)[:180]}", flush=True)
            logger.warning("hfq 拉取失败 %s: %s", symbol, exc)
            if streak >= max(int(max_fail_streak), 1):
                print(f"--- 连续失败 {streak} 只，判定触发风控，冷却 {cooldown:.0f}s ---",
                      flush=True)
                time.sleep(float(cooldown))
                streak = 0
        write_json({"total": total, "done": len(ok), "failed": len(failed),
                    "symbols": ok, "failed_symbols": failed,
                    "inconsistent_symbols": inconsistent,
                    "in_progress": index < total,
                    "start": start, "end": end,
                    "sources": list(sources or HFQ_SOURCES)},
                   _progress_path(directory))
        if index % max(int(batch), 1) == 0 and index < total:
            print(f"--- 已完成 {index}/{total}，批间休息 {rest:.0f}s ---", flush=True)
            time.sleep(float(rest))
        elif index < total:
            time.sleep(float(throttle))
    (directory / "fetch_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    if inconsistent:
        (directory / "inconsistent.txt").write_text("\n".join(inconsistent), encoding="utf-8")
        print(f"--- 复权异常（建议训练排除）{len(inconsistent)} 只: {inconsistent} ---",
              flush=True)
    return {"ok": ok, "failed": failed, "inconsistent": inconsistent, "reports": reports}

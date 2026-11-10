"""数据加载器：多源 fallback、增量合并、fetch/refresh/info。"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from stock_ai.config.schema import DataConfig, normalize_adjust
from stock_ai.data.cache import DataCache
from stock_ai.data.schema import SymbolInfo, validate_frame
from stock_ai.data.sources.akshare_source import AkShareSource
from stock_ai.data.sources.base import DataSource
from stock_ai.data.sources.eastmoney_source import EastMoneySource
from stock_ai.data.sources.sina_source import SinaSource
from stock_ai.data.sources.synthetic_source import SyntheticSource
from stock_ai.data.sources.tencent_source import TencentSource
from stock_ai.utils.dates import validate_range
from stock_ai.utils.errors import ConfigError, DataSourceError
from stock_ai.utils.io import timestamp
from stock_ai.utils.logging import LoggerFactory

__all__ = ["StockDataLoader", "SUPPORTED_SOURCES"]

logger = LoggerFactory.get(__name__)

#: 主源之后的自动 fallback 顺序（不含 synthetic，synthetic 由 --source 显式指定或最终兜底）
#: ``sina`` 置于链尾：它对**不复权**数据可一次返回全历史且无窗口速率风控，
#: 是腾讯接口被风控（HTTP 501）时的可靠兜底。
_FALLBACK_CHAIN: tuple[str, ...] = ("akshare", "tencent", "eastmoney", "sina")


def _make_akshare(cfg: DataConfig) -> DataSource:
    """构造 AKShare 源（唯一真源工厂之一）。"""
    return AkShareSource(
        adjust=cfg.adjust, retry_times=cfg.retry_times,
        retry_backoff=cfg.retry_backoff, timeout=cfg.timeout,
    )


def _make_tencent(cfg: DataConfig) -> DataSource:
    """构造腾讯源（唯一真源工厂之一）。"""
    return TencentSource(
        adjust=cfg.adjust, retry_times=cfg.retry_times,
        retry_backoff=cfg.retry_backoff, timeout=cfg.timeout,
    )


def _make_eastmoney(cfg: DataConfig) -> DataSource:
    """构造东方财富源（唯一真源工厂之一）。"""
    return EastMoneySource(
        adjust=cfg.adjust, retry_times=cfg.retry_times,
        retry_backoff=cfg.retry_backoff, timeout=cfg.timeout,
    )


def _make_sina(cfg: DataConfig) -> DataSource:
    """构造新浪源（唯一真源工厂之一）。"""
    return SinaSource(
        adjust=cfg.adjust, retry_times=cfg.retry_times,
        retry_backoff=cfg.retry_backoff, timeout=cfg.timeout,
    )


def _make_synthetic(cfg: DataConfig) -> DataSource:
    """构造合成源（唯一真源工厂之一）。"""
    return SyntheticSource(adjust=cfg.adjust, seed=42)


#: 内置数据源构造器注册表 —— **唯一真源**（P1-2）：``name -> factory(cfg)``。
#: :data:`SUPPORTED_SOURCES` 与运行时 registry 均由此派生，从根本上杜绝
#: "校验白名单"与"构造真源"不一致所导致的"校验通过却被静默忽略"。
_SOURCE_FACTORIES: dict[str, Callable[[DataConfig], DataSource]] = {
    "akshare": _make_akshare,
    "tencent": _make_tencent,
    "eastmoney": _make_eastmoney,
    "sina": _make_sina,
    "synthetic": _make_synthetic,
}

#: 支持的数据源白名单（由 registry 唯一推导，单一真源，P1-2）
SUPPORTED_SOURCES: frozenset[str] = frozenset(_SOURCE_FACTORIES)


class StockDataLoader:
    """多源数据加载器。

    按 :data:`_FALLBACK_CHAIN` 顺序构造数据源；:meth:`fetch_symbol` 逐个尝试，
    全部失败抛 :class:`DataSourceError`。若 ``config.source == "synthetic"``，
    则只使用合成数据源（离线可跑）。
    """

    def __init__(
        self,
        config: DataConfig,
        logger: "Any | None" = None,
        sources: "list[DataSource] | None" = None,
    ) -> None:
        """初始化。

        Args:
            config: 数据层配置。
            logger: 可选外部 logger（默认模块级 logger）。
            sources: 可选**注入**的数据源链（保留测试/插件的可扩展性，P1-2）。
                给定时跳过 built-in registry 校验，直接以其作为源链；为 None 时
                按 ``config.source`` 从 built-in registry 构造。

        Raises:
            ConfigError: ``config.source`` 非法（未注入且不在 registry 内）或
                ``config.adjust`` 非法。
        """
        self.config: DataConfig = config
        # ---- 复权口径归一化 + 校验（A）：保证后续 meta/缓存判定使用 canonical 值 ----
        self.config.adjust = normalize_adjust(self.config.adjust)
        self.logger = logger or LoggerFactory.get(__name__)
        self.cache: DataCache = DataCache(Path(config.cache_dir))
        self.sources: list[DataSource] = (
            list(sources) if sources is not None else self._build_sources()
        )
        self._last_source: str = ""
        self._symbol_sources: dict[str, str] = {}
        #: 最近一次 :meth:`fetch_all` 中逐只失败（``on_error="skip"`` 时填充）
        self.last_failures: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def _build_sources(self) -> list[DataSource]:
        """按 built-in registry 构造数据源链（registry 由唯一真源派生）。

        Raises:
            ConfigError: ``config.source`` 不在 :data:`SUPPORTED_SOURCES` 内（P1-2）。
        """
        cfg = self.config
        # registry 由唯一真源 _SOURCE_FACTORIES 派生 —— 与 SUPPORTED_SOURCES 同源
        registry: dict[str, DataSource] = {
            name: factory(cfg) for name, factory in _SOURCE_FACTORIES.items()
        }

        preferred = (cfg.source or "akshare").lower()
        # 非法 --source 必须在这里明确报错，绝不能被默认 fallback 链静默吞掉。
        if preferred not in registry:
            raise ConfigError(
                f"不支持的数据源: {cfg.source!r}（可选: {sorted(SUPPORTED_SOURCES)}）",
                context={"source": cfg.source, "valid": sorted(SUPPORTED_SOURCES)},
            )
        if preferred == "synthetic":
            return [registry["synthetic"]]

        order: list[str] = [preferred]
        for name in _FALLBACK_CHAIN:
            if name not in order:
                order.append(name)
        # synthetic 作为最终兜底，保证离线时链路不崩
        order.append("synthetic")
        return [registry[n] for n in order]

    # ------------------------------------------------------------------ #
    def fetch_symbol(self, symbol: str, start: str, end: str) -> "Any":
        """按 fallback 链尝试拉取单只股票。

        Args:
            symbol: 股票代码。
            start: 起始 ``YYYYMMDD``。
            end: 结束 ``YYYYMMDD``。

        Returns:
            规范化 DataFrame。

        Raises:
            DataSourceError: 所有数据源均失败。
        """
        errors: list[str] = []
        for src in self.sources:
            try:
                df = src.fetch(symbol, start, end)
                if df is not None and len(df) > 0:
                    self._last_source = src.name
                    self._symbol_sources[symbol] = src.name
                    self.logger.info(
                        "拉取成功 %s [%s] %d 行 (%s~%s)",
                        symbol, src.name, len(df), start, end,
                    )
                    return validate_frame(df)
            except Exception as exc:  # noqa: BLE001 - fallback 需吞掉所有源异常
                errors.append(f"{src.name}: {exc}")
                self.logger.warning("数据源 %s 失败 (%s): %s", src.name, symbol, exc)
                continue
        raise DataSourceError(
            f"所有数据源均失败: {symbol} ({start}~{end})",
            context={"symbol": symbol, "errors": errors},
        )

    # ------------------------------------------------------------------ #
    def _prev_source(self, symbol: str, meta: dict) -> str:
        """解析某股票既有缓存的来源（P2-1）。

        优先以磁盘上的 ``meta.json`` 为准（传入的 ``meta`` 可能是过期快照），
        其次回退到调用方传入的 ``meta``。均取不到时返回空串。

        Args:
            symbol: 股票代码。
            meta: 调用方持有的 meta 快照（可能为 ``{}``）。

        Returns:
            来源字符串（如 ``"tencent"`` / ``"synthetic"``），未知则 ``""``。
        """
        for snapshot in (self.cache.read_meta(), meta or {}):
            entry = (snapshot.get("symbols", {}) or {}).get(symbol, {}) or {}
            src = str(entry.get("source", "") or "")
            if src:
                return src
        return ""

    # ------------------------------------------------------------------ #
    def _cached_adjust(self, symbol: str, meta: dict) -> str:
        """读取既有缓存记录的复权口径（A5）；无记录（旧 meta）返回空串。"""
        entry = (meta.get("symbols", {}) or {}).get(symbol, {}) or {}
        return str(entry.get("adjust", "") or "")

    # ------------------------------------------------------------------ #
    def _should_protect_cache(self, symbol: str, meta: dict) -> bool:
        """判断是否应保护既有缓存不被隐式 synthetic 兜底覆盖。

        仅当满足全部条件时返回 True：
        1. 本次拉取最终来自 synthetic（真实源全部失败的隐式兜底，
           用户显式 ``--source synthetic`` 时不保护，尊重明确意图）；
        2. 缓存中已有该股票的数据；
        3. 已有缓存的来源不是 synthetic（真实来源），**或来源无法确定**
           （meta.json 缺失/损坏，此时保守保护——P2-1）。
        """
        if self._last_source != "synthetic":
            return False
        if (self.config.source or "").lower() == "synthetic":
            return False
        if not self.cache.exists(symbol):
            return False
        prev = self._prev_source(symbol, meta)
        if prev == "synthetic":
            # 既有缓存本身就是 synthetic（离线产物），再次被 synthetic 覆盖无损失。
            return False
        # prev 为真实来源，或为空（来源不可确定）-> 保守保护真实缓存。
        return True

    # ------------------------------------------------------------------ #
    def fetch_all(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        force: bool = False,
        throttle: float = 0.0,
        on_error: str = "raise",
    ) -> dict[str, "Any"]:
        """批量拉取并落盘缓存。

        Args:
            symbols: 股票列表（None 取 config.symbols）。
            start: 起始日期（None 取 config.start）。
            end: 结束日期（None 取 config.end）。
            force: 为 True 时忽略缓存强制重拉。
            throttle: 每只之间的间隔秒（默认 0，即不限速）。行情源（如腾讯）
                存在窗口速率风控，长名单批量拉取时应显式设置 ≥2s。
            on_error: ``"raise"``（默认，任一标的失败即上抛，保持既有行为）或
                ``"skip"``（逐只容错：失败记入 :attr:`last_failures` 并继续后续）。

        Returns:
            ``{symbol: df}``（仅含成功项）。

        Raises:
            ConfigError: ``on_error`` 取值非法。
            DataSourceError: ``on_error="raise"`` 且某标的全部数据源失败。
        """
        if on_error not in ("raise", "skip"):
            raise ConfigError(
                f"on_error 仅支持 'raise' | 'skip'，收到 {on_error!r}",
                context={"on_error": on_error},
            )
        syms = self._resolve_symbols(symbols)
        s = start or self.config.start
        e = end or self.config.end
        # ---- 参数前置校验（P2-6）：非法区间在此明确报错 ----
        validate_range(s, e, label="数据拉取区间", hint="请检查 --start/--end")
        result: dict[str, "Any"] = {}
        meta = self.cache.read_meta()
        self.last_failures = {}

        for idx, sym in enumerate(syms):
            # ---- 可选节流：避免触发行情源风控（如腾讯 501 WAF）----
            if throttle and idx:
                time.sleep(float(throttle))
            if not force and self.cache.exists(sym):
                try:
                    cached = self.cache.read(sym)
                    prev_adjust = self._cached_adjust(sym, meta)
                    if prev_adjust != self.config.adjust:
                        # ---- 复权口径一致性（A5）：绝不命中不同口径的旧缓存 ----
                        self.logger.warning(
                            "缓存复权口径 %r 与请求 %r 不一致，忽略缓存并全量重拉: %s",
                            prev_adjust or "(未知)", self.config.adjust, sym,
                        )
                    elif self._covers(cached, s, e):
                        self.logger.info("缓存命中 %s (%d 行，跳过网络)", sym, len(cached))
                        result[sym] = cached
                        continue
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("缓存读取失败 %s，将重拉: %s", sym, exc)

            try:
                df = self.fetch_symbol(sym, s, e)
            except Exception as exc:  # noqa: BLE001 - on_error="skip" 时逐只容错，不中断整批
                if on_error == "raise":
                    raise
                self.last_failures[sym] = str(exc)
                self.logger.error("拉取失败（已跳过，继续后续）%s: %s", sym, exc)
                continue
            if self._should_protect_cache(sym, meta):
                # 隐式 synthetic 兜底（非用户显式 --source synthetic）不得覆盖真实缓存
                cached = self.cache.read(sym)
                prev = self._prev_source(sym, meta)
                self._symbol_sources[sym] = prev or "cached"
                self.logger.warning(
                    "[保护] 所有真实数据源失败，synthetic 兜底数据未写入 %s "
                    "(已有真实缓存 %s 行)。请稍后重试 data fetch。",
                    sym, len(cached),
                )
                result[sym] = cached
                continue
            self.cache.write(sym, df)
            result[sym] = df
            if (
                self._last_source == "synthetic"
                and (self.config.source or "").lower() != "synthetic"
            ):
                # P2-4：隐式兜底时给出可区分的提示，帮助判断"代码错误"还是"网络不通"。
                self.logger.warning(
                    "[兜底] 真实数据源均失败，%s 使用合成数据离线兜底；"
                    "若为真实标的请核对代码是否正确或网络是否可用。",
                    sym,
                )
            info = self._make_info(
                sym, df, self._last_source, adjust=self.config.adjust,
            )
            self.cache.update_symbol_meta(info, meta)

        self.cache.write_meta(meta)
        return result

    # ------------------------------------------------------------------ #
    def refresh_symbol(self, symbol: str, today: str | None = None) -> "Any":
        """增量刷新：读缓存 -> 取 max(date) -> 拉增量 -> merge -> 落盘。

        Args:
            symbol: 股票代码。
            today: 增量截止日期（默认取 ``config.end``，再退化为今天）。

        Returns:
            合并后的完整 DataFrame。
        """
        end = today or self.config.end or datetime.now().strftime("%Y%m%d")
        meta = self.cache.read_meta()
        if not self.cache.exists(symbol):
            self.logger.info("无缓存，执行全量拉取: %s", symbol)
            df = self.fetch_symbol(symbol, self.config.start, end)
            self.cache.write(symbol, df)
            info = self._make_info(
                symbol, df, self._last_source, adjust=self.config.adjust,
            )
            self.cache.update_symbol_meta(info, meta)
            return df

        # ---- 复权口径一致性（A5）：不同口径**绝不**做增量拼接 ----
        prev_adjust = self._cached_adjust(symbol, meta)
        if prev_adjust != self.config.adjust:
            # 选择"退化为全量重拉"而非硬报错：refresh 是便捷操作，全量重拉安全且不会
            # 阻塞用户；同时避免出现 "qfq 老数据 + hfq 新数据" 的混合口径污染。
            # P3：重拉起点取 ``min(缓存现有起日, config.start)`` —— 口径变更时**尽量
            # 保留用户既有区间**，绝不因 config.start 较晚而**静默收缩**历史数据
            # （例如 hfq 有 2012 起数据，refresh 到 qfq 不应缩到 2018）。
            old = self.cache.read(symbol)
            have_start = old["date"].min().strftime("%Y%m%d")
            have_end = old["date"].max().strftime("%Y%m%d")
            fetch_start = min(have_start, self.config.start or have_start)
            self.logger.warning(
                "[复权口径变更] 缓存复权 %r 与请求 %r 不一致，将全量重拉并重建区间: "
                "%s~%s -> %s~%s（如需固定区间请显式指定 --start/--end）: %s",
                prev_adjust or "(未知)", self.config.adjust,
                have_start, have_end, fetch_start, end, symbol,
            )
            df = self.fetch_symbol(symbol, fetch_start, end)
            if self._should_protect_cache(symbol, meta):
                cached = self.cache.read(symbol)
                self._symbol_sources[symbol] = self._prev_source(symbol, meta) or "cached"
                self.logger.warning(
                    "[保护] synthetic 兜底数据未写入 %s（保留既有缓存）。", symbol,
                )
                return cached
            self.cache.write(symbol, df)
            info = self._make_info(
                symbol, df, self._last_source, adjust=self.config.adjust,
            )
            self.cache.update_symbol_meta(info, self.cache.read_meta())
            return df

        old = self.cache.read(symbol)
        last = old["date"].max()
        inc_start = (last + __import__("pandas").Timedelta(days=1)).strftime("%Y%m%d")

        if inc_start > end:
            self.logger.info("缓存已最新: %s (last=%s)", symbol, last.strftime("%Y%m%d"))
            return old

        new = self.fetch_symbol(symbol, inc_start, end)
        if self._should_protect_cache(symbol, self.cache.read_meta()):
            self.logger.warning(
                "[保护] 增量拉取来自 synthetic 兜底，已跳过写入 %s（保留真实缓存至 %s）。",
                symbol, last.strftime("%Y%m%d"),
            )
            return old
        merged = self.merge_incremental(old, new)
        self.cache.write(symbol, merged)

        gaps = self.cache.find_gaps(merged)
        if gaps:
            self.logger.warning("检测到 %d 个缺口日 (%s): %s", len(gaps), symbol, gaps[:10])

        info = self._make_info(
            symbol, merged, self._last_source, gaps=len(gaps),
            adjust=self.config.adjust,
        )
        self.cache.update_symbol_meta(info, self.cache.read_meta())
        return merged

    def refresh_all(self, symbols: list[str] | None = None) -> dict[str, "Any"]:
        """批量增量刷新。"""
        syms = self._resolve_symbols(symbols)
        return {sym: self.refresh_symbol(sym) for sym in syms}

    # ------------------------------------------------------------------ #
    def load_cached(self, symbol: str) -> "Any":
        """读取缓存（缺失抛 ArtifactError）。"""
        return self.cache.read(symbol)

    def load_all_cached(self, symbols: list[str] | None = None) -> dict[str, "Any"]:
        """批量读取缓存。"""
        syms = self._resolve_symbols(symbols)
        out: dict[str, "Any"] = {}
        for sym in syms:
            if self.cache.exists(sym):
                out[sym] = self.cache.read(sym)
            else:
                self.logger.warning("缓存缺失，跳过: %s", sym)
        return out

    # ------------------------------------------------------------------ #
    def info(self, symbols: list[str] | None = None) -> list[SymbolInfo]:
        """汇总缓存概览。"""
        syms = symbols if symbols else self.cache.list_symbols()
        meta = self.cache.read_meta().get("symbols", {})
        out: list[SymbolInfo] = []
        for sym in syms:
            if not self.cache.exists(sym):
                continue
            try:
                df = self.cache.read(sym)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("读取失败 %s: %s", sym, exc)
                continue
            entry = meta.get(sym, {})
            out.append(
                self._make_info(
                    sym,
                    df,
                    entry.get("source", "unknown"),
                    updated_at=entry.get("updated_at", ""),
                    adjust=entry.get("adjust", ""),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def merge_incremental(old: "Any", new: "Any") -> "Any":
        """合并两段日线：拼接 -> 去重（保留新值）-> 升序 -> 校验。

        Args:
            old: 旧数据。
            new: 增量数据。

        Returns:
            合并后的 DataFrame。
        """
        import pandas as pd

        frames = [f for f in (old, new) if f is not None and len(f) > 0]
        if not frames:
            raise DataSourceError("merge_incremental: 输入均为空")
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("date", kind="stable")
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        return validate_frame(merged.reset_index(drop=True))

    # ------------------------------------------------------------------ #
    def _resolve_symbols(self, symbols: list[str] | None) -> list[str]:
        """解析符号列表：CLI > config > symbols_hs300.yaml。"""
        if symbols:
            return [str(s) for s in symbols]
        if self.config.symbols:
            return [str(s) for s in self.config.symbols]

        # 回退到 configs/symbols_hs300.yaml
        for candidate in ("configs/symbols_hs300.yaml", "symbols_hs300.yaml"):
            p = Path(candidate)
            if p.exists():
                try:
                    import yaml

                    with p.open("r", encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                    syms = data.get("symbols", [])
                    if syms:
                        self.logger.info("从 %s 载入股票池 (%d 只)", p, len(syms))
                        return [str(s) for s in syms]
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("载入股票池失败 %s: %s", p, exc)
        raise DataSourceError("未指定股票池：请在 config 或 --symbols 中提供 symbols")

    @staticmethod
    def _covers(df: "Any", start: str, end: str) -> bool:
        """判断缓存是否覆盖请求区间（允许边界小误差）。"""
        if df is None or len(df) == 0:
            return False
        import pandas as pd

        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        have_s = pd.to_datetime(df["date"]).min()
        have_e = pd.to_datetime(df["date"]).max()
        # 起始需不晚于请求起始；结束允许最多 5 天误差（周末/节假日）
        tol = pd.to_timedelta(5, unit="D")
        return bool(have_s <= s + tol and have_e >= e - tol)

    @staticmethod
    def _make_info(
        symbol: str,
        df: "Any",
        source: str,
        updated_at: str = "",
        gaps: int = 0,
        adjust: str = "",
    ) -> SymbolInfo:
        """由 DataFrame 构造 :class:`SymbolInfo`（含复权口径 ``adjust``）。"""
        dates = df["date"]
        return SymbolInfo(
            symbol=symbol,
            start=dates.min().strftime("%Y%m%d"),
            end=dates.max().strftime("%Y%m%d"),
            rows=int(len(df)),
            source=source or "unknown",
            updated_at=updated_at or timestamp(),
            missing_days=int(gaps),
            adjust=adjust or "",
        )

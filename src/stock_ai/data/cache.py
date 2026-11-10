"""数据缓存：parquet 落盘 + meta.json 管理 + 缺口检测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stock_ai.data.schema import SymbolInfo, validate_frame
from stock_ai.utils.errors import ArtifactError
from stock_ai.utils.io import read_json, read_parquet, sanitize_filename, write_json, write_parquet
from stock_ai.utils.logging import LoggerFactory

__all__ = ["DataCache"]

logger = LoggerFactory.get(__name__)

_META_NAME = "meta.json"


class DataCache:
    """parquet 缓存管理器（``data/raw/<symbol>.parquet`` + ``data/meta.json``）。"""

    def __init__(self, cache_dir: Path) -> None:
        """初始化。

        Args:
            cache_dir: 缓存目录。
        """
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    @property
    def meta_path(self) -> Path:
        """``meta.json`` 路径。"""
        return self.cache_dir / _META_NAME

    def path_of(self, symbol: str) -> Path:
        """某代码对应的 parquet 路径。"""
        return self.cache_dir / f"{sanitize_filename(symbol)}.parquet"

    def exists(self, symbol: str) -> bool:
        """缓存文件是否存在。"""
        return self.path_of(symbol).exists()

    # ------------------------------------------------------------------ #
    def read(self, symbol: str) -> "Any":
        """读取缓存。

        Raises:
            ArtifactError: 缓存缺失或损坏。
        """
        p = self.path_of(symbol)
        if not p.exists():
            raise ArtifactError(
                f"缓存缺失: {p}（请先执行 `stock_ai data fetch --symbols {symbol}`）",
                context={"symbol": symbol},
            )
        df = read_parquet(p)
        return validate_frame(df)

    def write(self, symbol: str, df: "Any") -> Path:
        """写入缓存（写入前过 :func:`validate_frame`）。"""
        clean = validate_frame(df)
        p = write_parquet(clean, self.path_of(symbol))
        logger.debug("缓存写入: %s (%d 行)", p.name, len(clean))
        return p

    # ------------------------------------------------------------------ #
    def read_meta(self) -> dict:
        """读取 ``meta.json``（不存在返回空 dict）。"""
        if not self.meta_path.exists():
            return {}
        try:
            return read_json(self.meta_path)
        except ArtifactError:
            logger.warning("meta.json 损坏，忽略: %s", self.meta_path)
            return {}

    def write_meta(self, meta: dict) -> None:
        """写出 ``meta.json``。"""
        write_json(meta, self.meta_path)

    def update_symbol_meta(self, info: SymbolInfo, all_meta: dict | None = None) -> dict:
        """更新某只股票在 meta 中的条目。

        Args:
            info: 该股票的概要信息。
            all_meta: 现有 meta（None 则读取）。

        Returns:
            更新后的 meta dict。
        """
        meta = all_meta if all_meta is not None else self.read_meta()
        symbols = meta.setdefault("symbols", {})
        symbols[info.symbol] = info.to_dict()
        meta["updated_at"] = info.updated_at
        self.write_meta(meta)
        return meta

    # ------------------------------------------------------------------ #
    def find_gaps(self, df: "Any", freq: str = "B") -> list[str]:
        """检测缺失的交易日（以业务日历 ``freq='B'`` 近似）。

        Args:
            df: 日线 DataFrame。
            freq: 期望频率，默认工作日。

        Returns:
            缺失日期字符串列表（``YYYYMMDD``）。
        """
        import pandas as pd

        if df is None or len(df) < 2:
            return []
        dates = pd.to_datetime(df["date"]).sort_values()
        full = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq=freq)
        present = set(dates.dt.normalize().tolist())
        missing = [d.strftime("%Y%m%d") for d in full if d not in present]
        return missing

    def list_symbols(self) -> list[str]:
        """列出缓存目录中已存在的股票代码。"""
        return sorted(p.stem for p in self.cache_dir.glob("*.parquet"))

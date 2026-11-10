"""样本构建器：拼 obs 矩阵、walk-forward 切分、计算 data_hash、落盘 npz+meta。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stock_ai.config.schema import SampleConfig
from stock_ai.samples.features import FeatureEngineer
from stock_ai.samples.normalizer import NormalizerState
from stock_ai.samples.schema import NPZ_KEYS, SampleBundle, SampleSplit
from stock_ai.samples.split import date_to_int, walk_forward_split
from stock_ai.utils.errors import ArtifactError
from stock_ai.utils.hashing import sha256_files
from stock_ai.utils.io import timestamp, write_json, write_npz
from stock_ai.utils.logging import LoggerFactory

__all__ = ["SampleBuilder"]

logger = LoggerFactory.get(__name__)

_EMPTY_OBS_SHAPE_ERR = "样本为空：请检查日期区间与 window 设置"


class SampleBuilder:
    """把原始日线转换为可训练的样本张量。

    Args:
        config: 样本层配置。
        logger: 可选外部 logger。
    """

    def __init__(self, config: SampleConfig, logger: "Any | None" = None) -> None:
        """初始化。"""
        self.config: SampleConfig = config
        self.logger = logger or LoggerFactory.get(__name__)
        self.engineer: FeatureEngineer = FeatureEngineer(config.features)

    # ------------------------------------------------------------------ #
    def build_all(
        self,
        loader: "Any",
        symbols: list[str] | None = None,
        data_hash: str = "",
    ) -> SampleBundle:
        """从缓存的原始日线构建 train/val/test 三切分。

        Args:
            loader: :class:`stock_ai.data.loader.StockDataLoader` 实例。
            symbols: 股票列表（None 取 config）。
            data_hash: 预先计算的原始数据摘要（None 时由调用方/此处计算）。

        Returns:
            :class:`SampleBundle`。

        Raises:
            ConfigError: 训练/测试区间 ``start > end``（P2-6）。
        """
        from stock_ai.utils.dates import validate_range

        # ---- 参数前置校验（P2-6）：区间非法时给出直观提示 ----
        validate_range(
            self.config.train_start, self.config.train_end,
            label="训练区间", hint="请检查 train_start/train_end",
        )
        validate_range(
            self.config.test_start, self.config.test_end,
            label="测试区间", hint="请检查 test_start/test_end",
        )

        frames = loader.load_all_cached(symbols)
        if not frames:
            raise ArtifactError("无可用缓存数据：请先执行 `stock_ai data fetch`")
        syms = list(frames.keys())

        if not data_hash:
            paths = [loader.cache.path_of(s) for s in syms if loader.cache.exists(s)]
            data_hash = self.compute_data_hash(paths)

        # ---- 1) 仅用【训练窗口】数据 fit 归一化状态（K9）----
        train_df = self._concat_in_window(frames, self.config.train_start, self.config.train_end)
        if len(train_df) < self.config.window + 2:
            raise ArtifactError(
                f"训练窗口数据不足: {len(train_df)} 行 < window+2",
                context={"train_df": len(train_df)},
            )
        state = self.engineer.fit(train_df)
        state.data_hash = data_hash
        self.logger.info(
            "归一化统计量已冻结（仅训练窗口 %d 行，%d 特征）",
            len(train_df), state.n_features,
        )

        # ---- 2) 用冻结状态构建三个切分 ----
        val_range = self._val_range()

        train_split = self.build_split(frames, state, self.config.train_start, self.config.train_end)
        val_split = self.build_split(frames, state, val_range[0], val_range[1])
        test_split = self.build_split(frames, state, self.config.test_start, self.config.test_end)

        if train_split.n_steps == 0:
            raise ArtifactError(f"训练样本为空 ({_EMPTY_OBS_SHAPE_ERR})")
        if test_split.n_steps == 0:
            raise ArtifactError(f"测试样本为空 ({_EMPTY_OBS_SHAPE_ERR})")

        bundle = SampleBundle(
            train=train_split,
            val=val_split,
            test=test_split,
            normalizer=state,
            feature_names=list(self.config.features),
            window=int(self.config.window),
            seed=0,
            data_hash=data_hash,
            symbols=syms,
            embargo_days=int(self.config.embargo_days),
            train_range=(self.config.train_start, self.config.train_end),
            test_range=(self.config.test_start, self.config.test_end),
        )
        return bundle

    # ------------------------------------------------------------------ #
    def build_split(
        self,
        dfs: dict[str, "Any"],
        state: NormalizerState,
        start: str,
        end: str,
    ) -> SampleSplit:
        """在 ``[start, end]`` 窗口上构建单个 :class:`SampleSplit`。

        Args:
            dfs: ``{symbol: raw_df}``。
            state: 冻结归一化状态。
            start: 窗口起 ``YYYYMMDD``。
            end: 窗口止。

        Returns:
            :class:`SampleSplit`（obs/dates/symbols/prices 已按时间全局排序）。
        """
        s, e = date_to_int(start), date_to_int(end)
        obs_list: list[np.ndarray] = []
        dates_list: list[str] = []
        syms_list: list[str] = []
        prices_list: list[float] = []

        for sym, raw in dfs.items():
            if raw is None or len(raw) == 0:
                continue
            df = raw.copy()
            day = np.asarray([date_to_int(x) for x in df["date"].dt.strftime("%Y%m%d")])
            mask = (day >= s) & (day <= e)
            sub = df.loc[mask].reset_index(drop=True)
            if len(sub) < 2:
                continue

            obs = self.engineer.make_obs_matrix(sub, state, self.config.window)
            obs_list.append(obs)
            dates_list.extend(sub["date"].dt.strftime("%Y%m%d").tolist())
            syms_list.extend([sym] * len(sub))
            prices_list.extend(sub["close"].astype("float64").tolist())

        if not obs_list:
            return SampleSplit(
                obs=np.zeros((0, self.config.window, state.n_features), dtype="float32"),
                dates=np.array([], dtype="<U8"),
                symbols=np.array([], dtype=object),
                prices=np.array([], dtype="float64"),
                requested_start=start,
                requested_end=end,
            )

        obs_all = np.concatenate(obs_list, axis=0)
        dates_arr = np.asarray(dates_list, dtype="<U8")
        syms_arr = np.asarray(syms_list, dtype=object)
        prices_arr = np.asarray(prices_list, dtype="float64")

        # 按日期稳定排序（同日内保留 symbol 顺序），严禁打乱语义
        order = np.argsort(dates_arr.astype("int64"), kind="stable")
        obs_all = obs_all[order]
        dates_arr = dates_arr[order]
        syms_arr = syms_arr[order]
        prices_arr = prices_arr[order]

        # 数值健康检查
        if not np.all(np.isfinite(obs_all)):
            n_bad = int(np.sum(~np.isfinite(obs_all)))
            self.logger.warning("obs 含 %d 个非法值，已置零", n_bad)
            obs_all = np.nan_to_num(obs_all, nan=0.0, posinf=0.0, neginf=0.0)

        return SampleSplit(
            obs=obs_all.astype("float32"),
            dates=dates_arr,
            symbols=syms_arr,
            prices=prices_arr,
            requested_start=start,
            requested_end=end,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_data_hash(paths: list[Path]) -> str:
        """对原始 parquet 文件字节计算 sha256（K16）。"""
        if not paths:
            return ""
        return sha256_files(paths)

    # ------------------------------------------------------------------ #
    def save(self, bundle: SampleBundle, out_dir: Path, seed: int = 42) -> dict[str, Path]:
        """落盘 ``train.npz`` / ``val.npz`` / ``test.npz`` / ``meta.json``。

        Args:
            bundle: 待保存的样本包。
            out_dir: 输出目录。
            seed: 写入 meta 的随机种子。

        Returns:
            ``{"train": path, "val": path, "test": path, "meta": path}``。
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bundle.seed = int(seed)

        paths: dict[str, Path] = {}
        for name, split in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
            p = write_npz(out / f"{name}.npz", **split.to_npz_dict())
            paths[name] = p
            self.logger.info("写入 %s: %d 步 x %d 特征", p.name, split.n_steps, split.n_features)

        meta_path = out / "meta.json"
        write_json(bundle.meta_dict(), meta_path)
        paths["meta"] = meta_path
        return paths

    # ------------------------------------------------------------------ #
    def _val_range(self) -> tuple[str, str]:
        """从训练窗口尾部切出验证区间（避免与训练样本完全重叠）。"""
        import pandas as pd

        ts = pd.to_datetime(self.config.train_start, format="%Y%m%d")
        te = pd.to_datetime(self.config.train_end, format="%Y%m%d")
        total_days = (te - ts).days
        ratio = max(0.0, min(0.9, float(self.config.val_ratio)))
        if total_days <= 30 or ratio <= 0.0:
            # 无有效验证区间时返回空区间（build_split 会得到 0 行）
            return self.config.train_end, self.config.train_end
        split_ts = te - pd.to_timedelta(int(total_days * ratio), unit="D")
        return split_ts.strftime("%Y%m%d"), self.config.train_end

    @staticmethod
    def _concat_in_window(dfs: dict[str, "Any"], start: str, end: str) -> "Any":
        """拼接所有股票在 ``[start, end]`` 窗口内的行。"""
        import pandas as pd

        s, e = date_to_int(start), date_to_int(end)
        parts = []
        for raw in dfs.values():
            if raw is None or len(raw) == 0:
                continue
            day = np.asarray([date_to_int(x) for x in raw["date"].dt.strftime("%Y%m%d")])
            mask = (day >= s) & (day <= e)
            sub = raw.loc[mask]
            if len(sub) > 0:
                parts.append(sub)
        if not parts:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return pd.concat(parts, ignore_index=True)

"""配置加载器：YAML 载入 -> 路径解析 -> CLI 覆盖合并 -> ensure_dirs。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from stock_ai.config.schema import AppConfig

__all__ = ["ConfigLoader"]


class ConfigLoader:
    """配置加载与后处理工具类。"""

    @staticmethod
    def load(path: Path | str = "configs/default.yaml") -> AppConfig:
        """载入配置文件。

        Args:
            path: YAML 路径；若不存在则回退到纯代码默认值（不抛异常，方便离线启动）。

        Returns:
            :class:`AppConfig` 实例（已解析路径）。
        """
        p = Path(path)
        if p.exists():
            cfg = AppConfig.from_yaml(p)
        else:
            cfg = AppConfig()
        cfg = ConfigLoader.apply_env_overrides(cfg)
        return ConfigLoader.resolve_paths(cfg)

    @staticmethod
    def resolve_paths(cfg: AppConfig, base: Path | None = None) -> AppConfig:
        """将相对路径解析为相对于 ``base``（默认 ``paths.root``）的绝对路径。

        Args:
            cfg: 待处理配置。
            base: 基准目录，默认取 ``cfg.paths.root``。

        Returns:
            处理后的 ``cfg``（原地修改）。
        """
        root = Path(base) if base is not None else Path(cfg.paths.root)
        root = root.resolve()

        cfg.paths.root = root

        def _abs(p: Path) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (root / pp)

        cfg.paths.raw = _abs(cfg.paths.raw)
        cfg.paths.samples = _abs(cfg.paths.samples)
        cfg.paths.models = _abs(cfg.paths.models)
        cfg.paths.logs = _abs(cfg.paths.logs)
        cfg.paths.reports = _abs(cfg.paths.reports)

        cfg.data.cache_dir = _abs(cfg.data.cache_dir)
        cfg.train.out_dir = _abs(cfg.train.out_dir)
        # 大规模训练编排：符号名单文件同样解析为绝对路径（增量，1 行）
        cfg.scale.symbols_file = _abs(cfg.scale.symbols_file)
        return cfg

    @staticmethod
    def apply_env_overrides(cfg: AppConfig) -> AppConfig:
        """从环境变量覆盖敏感/环境相关配置。

        支持：
            - ``STOCKAI_SEED`` -> ``random_seed``
            - ``STOCKAI_LOG_LEVEL`` -> ``log_level``
            - ``STOCKAI_DATA_SOURCE`` -> ``data.source``
            - ``STOCKAI_BACKEND`` -> ``train.backend``
            - ``HTTPS_PROXY`` / ``HTTP_PROXY`` 由 requests 自动读取，无需处理。

        Args:
            cfg: 待处理配置。

        Returns:
            处理后的 ``cfg``（原地修改）。
        """
        seed = os.environ.get("STOCKAI_SEED")
        if seed:
            try:
                cfg.random_seed = int(seed)
            except ValueError:
                pass

        level = os.environ.get("STOCKAI_LOG_LEVEL")
        if level:
            cfg.log_level = level

        source = os.environ.get("STOCKAI_DATA_SOURCE")
        if source:
            cfg.data.source = source

        backend = os.environ.get("STOCKAI_BACKEND")
        if backend:
            cfg.train.backend = backend

        return cfg

    @staticmethod
    def merge(cfg: AppConfig, overrides: dict[str, Any]) -> AppConfig:
        """便捷方法：合并 CLI 覆盖。"""
        return cfg.merge_cli_overrides(overrides)

"""``doctor`` 环境自检命令（P2，R-18）。

检查：Python 版本、关键依赖、可选依赖（akshare/torch/SB3）、目录可写、数据/模型产物、
以及**训练后端可用性**（架构 §1.3 要求 doctor 显式报告）。
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import click

from stock_ai import __version__
from stock_ai.config.loader import ConfigLoader
from stock_ai.train.factory import TrainerFactory

__all__ = ["doctor"]


def _check_module(name: str, import_name: str | None = None) -> tuple[bool, str]:
    """检查模块是否可导入，返回 ``(可用, 版本或说明)``。"""
    mod = import_name or name
    try:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            return False, "未安装"
        module = importlib.import_module(mod)
        version = getattr(module, "__version__", "?")
        return True, str(version)
    except Exception as exc:  # noqa: BLE001
        return False, f"导入失败: {exc}"


@click.command("doctor")
@click.option("--config", "config", default="configs/default.yaml")
def doctor(config: str) -> None:
    """环境自检：依赖 / 后端 / 目录 / 产物。"""
    click.echo("=" * 60)
    click.echo(f"stock_ai doctor  (v{__version__})")
    click.echo("=" * 60)

    # ---- 1) Python ----
    click.echo("\n[1] Python 环境")
    click.echo(f"  Python  : {sys.version.split()[0]}")
    click.echo(f"  平台    : {platform.platform()}")
    click.echo(f"  解释器  : {sys.executable}")

    # ---- 2) 核心依赖 ----
    click.echo("\n[2] 核心依赖")
    core = [
        ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
        ("scipy", "scipy"), ("click", "click"), ("yaml", "pyyaml"),
        ("matplotlib", "matplotlib"), ("gymnasium", "gymnasium"),
        ("requests", "requests"),
    ]
    for import_name, disp in core:
        ok, ver = _check_module(disp, import_name)
        click.echo(f"  {'✓' if ok else '✗'} {disp:<12} {ver}")

    # ---- 3) 可选依赖 ----
    click.echo("\n[3] 可选依赖")
    optional = [("akshare", "akshare"), ("torch", "torch"),
                ("stable_baselines3", "stable_baselines3")]
    for import_name, disp in optional:
        ok, ver = _check_module(disp, import_name)
        click.echo(f"  {'✓' if ok else '○'} {disp:<20} {ver}")

    # ---- 4) 训练后端可用性 ----
    click.echo("\n[4] 训练后端可用性")
    backends = TrainerFactory.available_backends()
    for name, ok in backends.items():
        click.echo(f"  {'✓' if ok else '✗'} {name}")
    primary = "SB3" if backends.get("sb3") else "NumpyPPO(兜底)"
    click.echo(f"  -> 将使用: {primary}")

    # ---- 5) 目录与产物 ----
    click.echo("\n[5] 目录与产物")
    try:
        cfg = ConfigLoader.load(config)
        cfg.ensure_dirs()
        for label, p in (
            ("raw", cfg.paths.raw), ("samples", cfg.paths.samples),
            ("models", cfg.paths.models), ("logs", cfg.paths.logs),
            ("reports", cfg.paths.reports),
        ):
            writable = _writable(Path(p))
            click.echo(f"  {'✓' if writable else '✗'} {label:<8} {p}")

        raw_files = list(Path(cfg.paths.raw).glob("*.parquet"))
        samp = Path(cfg.paths.samples)
        models = list(Path(cfg.paths.models).glob("*.zip"))
        click.echo(f"  raw parquet: {len(raw_files)} 个")
        click.echo(f"  samples    : train={_exists(samp / 'train.npz')} "
                   f"test={_exists(samp / 'test.npz')} meta={_exists(samp / 'meta.json')}")
        click.echo(f"  models     : {len(models)} 个 .zip")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  ✗ 目录检查失败: {exc}")

    click.echo("\n" + "=" * 60)
    click.echo("doctor 完成。若 SB3 不可用，项目将以纯 NumPy PPO 兜底运行。")
    click.echo("=" * 60)


def _writable(path: Path) -> bool:
    """判断目录是否可写（非破坏性：仅做权限检查，不写入/删除探针文件）。"""
    import os

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return False
    return os.access(str(path), os.W_OK) and path.is_dir()


def _exists(path: Path) -> bool:
    """``yes`` / ``no`` 标记。"""
    return "yes" if path.exists() else "no"

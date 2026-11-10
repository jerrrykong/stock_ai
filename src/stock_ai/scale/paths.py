"""``data/scale`` / ``models/units`` / ``reports/units`` / ``logs/scale`` 路径派生。

所有路径均由既有 :class:`~stock_ai.config.schema.PathConfig.root` 派生，
全程使用 :class:`pathlib.Path`，保证 Windows / POSIX 一致。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ScalePaths"]


class ScalePaths:
    """大规模训练/检测的全套产物路径。

    Args:
        root: 工程根目录（通常为 ``AppConfig.paths.root``，已 resolve）。
    """

    def __init__(self, root: Path) -> None:
        """初始化并派生全部子路径。"""
        self.root: Path = Path(root)

        self.scale_dir: Path = self.root / "data" / "scale"
        self.units_json: Path = self.scale_dir / "units.json"
        self.split_json: Path = self.scale_dir / "split.json"
        self.state_dir: Path = self.scale_dir / "state"

        self.models_dir: Path = self.root / "models" / "units"
        self.reports_dir: Path = self.root / "reports" / "units"
        self.logs_dir: Path = self.root / "logs" / "scale"

    # ------------------------------------------------------------------ #
    def ensure(self) -> None:
        """创建全部目录（幂等）。"""
        for d in (
            self.scale_dir,
            self.state_dir,
            self.models_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def state_of(self, phase: str) -> Path:
        """阶段状态文件路径：``state/{train,detect}.json``。"""
        return self.state_dir / f"{phase}.json"

    def control_json(self) -> Path:
        """控制信号文件路径：``state/control.json``。"""
        return self.state_dir / "control.json"

    # ------------------------------------------------------------------ #
    def model_zip(self, unit_id: str) -> Path:
        """单元模型路径：``models/units/<unit>.zip``。"""
        return self.models_dir / f"{unit_id}.zip"

    def sidecar(self, unit_id: str) -> Path:
        """单元 sidecar：``models/units/<unit>.json``。"""
        return self.models_dir / f"{unit_id}.json"

    def norm(self, unit_id: str) -> Path:
        """单元归一化状态：``models/units/<unit>.norm.json``。"""
        return self.models_dir / f"{unit_id}.norm.json"

    def curve(self, unit_id: str) -> Path:
        """单元训练曲线：``models/units/<unit>.curve.png``。"""
        return self.models_dir / f"{unit_id}.curve.png"

    def report(self, unit_id: str) -> Path:
        """单元检测报告：``reports/units/<unit>.json``。"""
        return self.reports_dir / f"{unit_id}.json"

    def report_png(self, unit_id: str) -> Path:
        """单元回测资金曲线：``reports/units/<unit>.png``。"""
        return self.reports_dir / f"{unit_id}.png"

    # ------------------------------------------------------------------ #
    def log_of(self, phase: str, ts: str) -> Path:
        """结构化日志路径：``logs/scale/<phase>_<ts>.log``。"""
        return self.logs_dir / f"{phase}_{ts}.log"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ScalePaths(root={self.root})"

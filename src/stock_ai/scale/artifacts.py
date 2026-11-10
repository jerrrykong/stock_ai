"""单元产物路径与**双重校验**（状态 + 产物文件）。

「完成」判定（Q11）：状态文件标记 ``done`` **并且** 产物文件存在且非空，
二者缺一不可。这样即使进程在写盘瞬间被杀，重启也能识别出半成品并回滚。

产物命名（架构 §8.1/§8.5）::

    models/units/<unit>.zip | <unit>.json | <unit>.norm.json | <unit>.curve.png
    reports/units/<unit>.json | <unit>.png
"""

from __future__ import annotations

from pathlib import Path

from stock_ai.scale.paths import ScalePaths

__all__ = ["UnitArtifacts"]


def _nonempty(path: Path) -> bool:
    """文件存在且大小 > 0。"""
    try:
        p = Path(path)
        return p.exists() and p.stat().st_size > 0
    except OSError:  # pragma: no cover - 极端 FS 错误
        return False


class UnitArtifacts:
    """单元产物路径与完成校验。

    Args:
        paths: 路径集合。
    """

    def __init__(self, paths: ScalePaths) -> None:
        """初始化。"""
        self.paths: ScalePaths = paths

    # ------------------------------------------------------------------ #
    def model_zip(self, unit_id: str) -> Path:
        """模型 zip 路径。"""
        return self.paths.model_zip(unit_id)

    def sidecar(self, unit_id: str) -> Path:
        """模型 sidecar 路径。"""
        return self.paths.sidecar(unit_id)

    def norm(self, unit_id: str) -> Path:
        """归一化状态路径。"""
        return self.paths.norm(unit_id)

    def curve(self, unit_id: str) -> Path:
        """训练曲线路径。"""
        return self.paths.curve(unit_id)

    def report(self, unit_id: str) -> Path:
        """检测报告路径。"""
        return self.paths.report(unit_id)

    def report_png(self, unit_id: str) -> Path:
        """回测资金曲线路径。"""
        return self.paths.report_png(unit_id)

    # ------------------------------------------------------------------ #
    def is_model_done(self, unit_id: str) -> bool:
        """训练单元是否完成：``zip`` + ``sidecar`` + ``norm`` 均存在且非空。"""
        return (
            _nonempty(self.model_zip(unit_id))
            and _nonempty(self.sidecar(unit_id))
            and _nonempty(self.norm(unit_id))
        )

    def is_report_done(self, unit_id: str) -> bool:
        """检测单元是否完成：``report`` 存在且非空。"""
        return _nonempty(self.report(unit_id))

    def is_done(self, phase: str, unit_id: str) -> bool:
        """按阶段判定单元是否完成（train → model；detect → report）。"""
        if phase == "train":
            return self.is_model_done(unit_id)
        if phase == "detect":
            return self.is_report_done(unit_id)
        # 未知阶段：任一产物存在即认为完成（保守）
        return self.is_model_done(unit_id) or self.is_report_done(unit_id)

    # ------------------------------------------------------------------ #
    def available_train_models(self) -> set[str]:
        """返回已训练完成（模型 zip 非空）的 ``unit_id`` 集合。"""
        out: set[str] = set()
        d = self.paths.models_dir
        if not d.exists():
            return out
        for p in d.glob("*.zip"):
            if _nonempty(p):
                out.add(p.stem)
        return out

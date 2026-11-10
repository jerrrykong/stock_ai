"""绘图：权益曲线 PNG + 训练曲线 PNG（matplotlib Agg，K15）。

Windows 无头环境友好；中文字体找不到时**自动回退英文标签，绝不崩**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # K15：禁止弹窗

import matplotlib.pyplot as plt  # noqa: E402

from stock_ai.utils.logging import LoggerFactory  # noqa: E402

__all__ = ["EquityPlotter", "setup_cjk_font"]

logger = LoggerFactory.get(__name__)

#: 候选中文字体
_CJK_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS",
)

_cjk_available: bool | None = None


def setup_cjk_font() -> bool:
    """尝试启用中文字体；成功返回 ``True``，失败则用英文标签。

    Returns:
        是否成功设置中文字体。
    """
    global _cjk_available
    if _cjk_available is not None:
        return _cjk_available
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in _CJK_CANDIDATES:
            if name in installed:
                plt.rcParams["font.sans-serif"] = [name] + list(_CJK_CANDIDATES)
                plt.rcParams["axes.unicode_minus"] = False
                _cjk_available = True
                logger.debug("启用中文字体: %s", name)
                return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("字体探测失败: %s", exc)
    plt.rcParams["axes.unicode_minus"] = False
    _cjk_available = False
    logger.info("未找到中文字体，图形将使用英文标签（不影响功能）")
    return False


class EquityPlotter:
    """权益曲线 / 训练曲线绘制器。"""

    def __init__(self) -> None:
        """初始化并尝试启用中文字体。"""
        self.has_cjk: bool = setup_cjk_font()

    # ------------------------------------------------------------------ #
    def _label(self, zh: str, en: str) -> str:
        """按字体可用性选择中/英文标签。"""
        return zh if self.has_cjk else en

    # ------------------------------------------------------------------ #
    def plot(
        self,
        result: "Any",
        out_path: Path,
        title: str = "",
    ) -> Path:
        """绘制策略 vs 基线权益曲线。

        Args:
            result: :class:`~stock_ai.eval.schema.BacktestResult`。
            out_path: 输出 PNG 路径。
            title: 图标题（空则用默认）。

        Returns:
            输出路径。
        """
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        equity = list(getattr(result, "equity_curve", []))
        baseline = list(getattr(result, "baseline_equity", []))
        dates = list(getattr(result, "dates", []))

        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=110)
        x = list(range(len(equity)))
        if equity:
            ax.plot(x, equity, label=self._label("策略净值", "Strategy"), color="#cc3333", lw=1.6)
        if baseline:
            ax.plot(
                list(range(len(baseline))), baseline,
                label=self._label("买入持有", "Buy & Hold"),
                color="#3366cc", lw=1.3, linestyle="--",
            )
        ax.set_xlabel(self._label("交易日", "Trading day"))
        ax.set_ylabel(self._label("组合市值（元）", "Portfolio value"))
        ax.set_title(title or self._label("权益曲线", "Equity Curve"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # x 轴刻度标注部分真实日期（有的话）
        if dates and len(dates) == len(equity):
            n = len(dates)
            tick_idx = [int(i) for i in _even_ticks(n, 6)]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(dates[i]) for i in tick_idx], rotation=30, ha="right")

        fig.tight_layout()
        fig.savefig(str(p))
        plt.close(fig)
        logger.info("权益曲线已保存: %s", p)
        return p

    # ------------------------------------------------------------------ #
    def plot_training_curve(
        self,
        series: dict[str, list[float]],
        out_path: Path,
        title: str = "",
    ) -> Path:
        """绘制训练曲线（支持多条子曲线）。

        Args:
            series: ``{曲线名: 数值列表}``。
            out_path: 输出 PNG 路径。
            title: 图标题。

        Returns:
            输出路径。
        """
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        valid = {k: list(v) for k, v in series.items() if v}
        fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
        if not valid:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            for name, values in valid.items():
                ax.plot(range(len(values)), values, label=name, lw=1.5)
            ax.legend(loc="best")
        ax.set_xlabel(self._label("迭代轮次", "Iteration"))
        ax.set_ylabel(self._label("数值", "Value"))
        ax.set_title(title or self._label("训练曲线", "Training Curve"))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(p))
        plt.close(fig)
        logger.info("训练曲线已保存: %s", p)
        return p


def _even_ticks(n: int, count: int) -> list[int]:
    """在 ``[0, n-1]`` 上均匀取 ``count`` 个整数刻度。"""
    if n <= 0:
        return []
    count = max(2, min(count, n))
    step = max(1, (n - 1) // max(1, count - 1))
    ticks = list(range(0, n, step))
    if ticks and ticks[-1] != n - 1:
        ticks.append(n - 1)
    return ticks

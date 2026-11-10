"""大规模训练编排 + 检测编排 + 本机 UI（增量子包）。

本子包在既有 6 条 CLI 流水线（``data/sample/train/backtest/predict/doctor``）之上**叠加**一层
可复现、可中断、可续跑的「沪深300 × 年度单元」批量训练与检测编排能力，并提供仅本机可访问的
轻量 Web UI（stdlib ``http.server``）。

设计要点（详见 ``docs/ARCHITECTURE_SCALE_TRAINING.md``）：

- **零新增第三方依赖**：全部能力由既有依赖与标准库覆盖。
- **零回归**：所有能力均为叠加，既有文件仅做最小追加。
- **可复现**：``split.json`` 不含时钟字段，同种子字节一致（AC3）。
- **断点续训**：状态 + 产物双重校验，``done`` 单元重启后跳过且不覆盖。
- **归一化一致性**：训练时持久化 ``NormalizerState``，检测侧复用、绝不重新 fit（Q2）。
"""

from __future__ import annotations

from stock_ai.scale.errors import ScaleError

__all__ = ["ScaleError", "__version__"]

#: 子包版本（与架构文档 v1.0 对齐）。
__version__: str = "1.0.0"

"""推理结果 dataclass（PRD §6 契约）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["PredictionRow"]

#: 动作中文标签（PRD §6 契约：买/卖/持有）
ACTION_LABELS: dict[int, str] = {0: "持有", 1: "买", 2: "卖"}


@dataclass
class PredictionRow:
    """单日预测记录。

    Attributes:
        date: 日期 ``YYYYMMDD``。
        symbol: 股票代码。
        action: 动作（买/卖/持有）。
        target_position: 目标仓位 ``[0,1]``。
        signal: 信号强度 ``[-1,1]``（买 +，卖 -）。
        price: 当日结算价。
        prob: 动作概率（可选）。
    """

    date: str
    symbol: str
    action: str
    target_position: float
    signal: float
    price: float
    prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """转为 dict（CSV/JSON 友好的扁平结构）。"""
        return {
            "date": self.date,
            "symbol": self.symbol,
            "action": self.action,
            "target_position": round(float(self.target_position), 4),
            "signal": round(float(self.signal), 4),
            "price": round(float(self.price), 4),
        }

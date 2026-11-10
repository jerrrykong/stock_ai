"""评估层：回测、指标、权益曲线、基线对比。"""

from __future__ import annotations

from stock_ai.eval.backtester import Backtester
from stock_ai.eval.metrics import compute_all
from stock_ai.eval.schema import BacktestResult
from stock_ai.eval.simulator import PortfolioSimulator

__all__: list[str] = ["Backtester", "BacktestResult", "PortfolioSimulator", "compute_all"]

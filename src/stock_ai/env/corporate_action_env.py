"""Versioned raw-price execution with audited shareholder distributions.

v1 is intentionally unchanged. v2 is a gross (pre personal dividend tax) ledger:
record-date ownership -> ex-date receivable/locked shares -> payment/listing.
Unlisted shares are marked at the underlying raw close, not treated as liquid cash.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np

from stock_ai.data.corporate_actions import CorporateActionError
from stock_ai.env.account import EnvState
from stock_ai.env.limits import price_limit
from stock_ai.env.rules import trade_cost
from stock_ai.env.strict_env import ExecutionBars, StrictExecutionEnv

EXECUTION_VERSION = "strict-t1-open-ca-v2"
REWARD_VERSION_EXCESS = "excess-ca-constant-buy-v2"
CENT = Decimal("0.01")


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _money(value: Decimal) -> float:
    return float(value.quantize(CENT, rounding=ROUND_HALF_UP))


@dataclass
class CorporateActionState(EnvState):
    """shares includes ex-date stock entitlement; locked shares are not sellable."""

    locked_shares: int = 0
    dividend_receivable: float = 0.0
    dividend_income: float = 0.0
    realized_pnl: float = 0.0

    @property
    def available_shares(self) -> int:
        return self.shares - self.locked_shares

    @property
    def position_ratio(self) -> float:
        return self.shares * self.prev_close / self.portfolio_value if self.portfolio_value > 0 else 0.0

    def mark_to_market(self, price: float) -> float:
        self.portfolio_value = self.cash + self.shares * float(price) + self.dividend_receivable
        return float(self.portfolio_value)

    def snapshot(self) -> dict:
        return {**super().snapshot(), "available_shares": self.available_shares,
                "locked_shares": self.locked_shares,
                "dividend_receivable": float(self.dividend_receivable),
                "dividend_income": float(self.dividend_income),
                "realized_pnl": float(self.realized_pnl)}


class CorporateActionExecutionEnv(StrictExecutionEnv):
    """T close -> next raw open, fail closed without reviewed CA coverage.

    Every episode starts flat. A record date before the episode therefore confers
    zero rights; a record inside it must be represented by an actual bar. Ex dates
    inside the interval must also have a bar (a zero-volume suspension bar is OK).
    Payment/listing on a non-bar date is settled before the next available open.
    """

    def __init__(self, obs, bars: ExecutionBars, env_cfg, reward_cfg=None,
                 seed=42, st=False, *, _with_benchmark=True):
        self.ledger = bars.corporate_actions
        if self.ledger is None:
            raise CorporateActionError("v2 缺少公司行为账本；禁止回退 v1 或把无事件视为无公司行为")
        if self.ledger.symbol != bars.symbol:
            raise CorporateActionError("公司行为账本代码与真实行情不一致")
        dates = [str(day) for day in bars.dates]
        try:
            for day in dates:
                if len(day) != 8 or datetime.strptime(day, "%Y%m%d").strftime("%Y%m%d") != day:
                    raise ValueError(day)
        except ValueError as exc:
            raise CorporateActionError("v2 真实行情日期必须为有效 YYYYMMDD") from exc
        if not dates or dates != sorted(set(dates)):
            raise CorporateActionError("v2 真实行情日期必须唯一且严格递增")
        arrays = [np.asarray(getattr(bars, key)) for key in ("open", "high", "low", "close", "volume")]
        if any(a.shape != (len(dates),) for a in arrays):
            raise CorporateActionError("v2 真实 OHLCV 长度不一致")
        matrix = bars.prices_matrix()
        if (not np.isfinite(matrix).all() or (matrix[:, :4] <= 0).any()
                or (matrix[:, 4] < 0).any()
                or (bars.high < np.maximum.reduce([bars.open, bars.close, bars.low])).any()
                or (bars.low > np.minimum.reduce([bars.open, bars.close, bars.high])).any()):
            raise CorporateActionError("v2 真实 OHLCV 非有限或不合法")
        self.ledger.require_coverage(dates[0], dates[-1])
        self.events = [event for event in self.ledger.events
                       if event.kind == "distribution" and event.record_date <= dates[-1]
                       and event.ex_date >= dates[0]]
        date_set = set(dates)
        for event in self.events:
            for field in ("record_date", "ex_date"):
                day = getattr(event, field)
                if dates[0] <= day <= dates[-1] and day not in date_set:
                    raise CorporateActionError(f"{event.event_id} 的 {field}={day} 缺少真实日线；禁止猜测登记/除权状态")
        # This implementation cannot assign a later distribution to stock which is
        # still awaiting listing. Refuse a potentially eligible overlap up front,
        # rather than discovering it after a training episode has already started.
        for earlier in self.events:
            if (earlier.bonus_ratio + earlier.transfer_ratio <= 0
                    or earlier.record_date <= dates[0]):
                continue  # first-close starts flat, so no earlier entitlement exists
            for later in self.events:
                if (dates[0] <= later.record_date <= dates[-1]
                        and earlier.ex_date <= later.record_date < earlier.listing_date):
                    raise CorporateActionError(
                        "登记日可能存在未上市股份；嵌套分配资格未建模，拒绝该区间")
        self._with_benchmark = bool(_with_benchmark)
        self._benchmark = None
        super().__init__(obs, bars, env_cfg, reward_cfg=reward_cfg, seed=seed, st=st)
        if not np.isfinite(self.obs).all():
            raise CorporateActionError("v2 观测含非有限值")
        self._clear_ledger_state()
        if self._with_benchmark:
            self._benchmark = CorporateActionExecutionEnv(
                obs, bars, env_cfg, seed=seed, st=st, _with_benchmark=False)

    def _clear_ledger_state(self):
        self.state = CorporateActionState(cash=float(self.cfg.initial_cash),
                                          portfolio_value=float(self.cfg.initial_cash))
        self._eligible = {}
        self._pending = {}
        self._settled = set()
        self._action_log = []
        self._daily_actions = []
        self._benchmark_curve = [float(self.cfg.initial_cash)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._clear_ledger_state()
        self.state.prev_close = float(self.bars.close[0])
        first = str(self.bars.dates[0])
        # Starting flat is an explicit boundary, never an inferred prior holding.
        for event in self.events:
            if event.record_date < first:
                self._eligible[event.event_id] = 0
        self._record(first)
        if self._benchmark is not None:
            self._benchmark.reset(seed=seed, options=options)
        return self._observation(), self._info(0.0, False, "reset", 0)

    def _log(self, event, stage, date, **values):
        item = {"event_id": event.event_id, "stage": stage, "date": date,
                "record_date": event.record_date, "ex_date": event.ex_date,
                "pay_date": event.pay_date, "listing_date": event.listing_date,
                "sources": list(event.sources),
                "account_after": {key: getattr(self.state, key) for key in
                                  ("cash", "shares", "locked_shares", "available_shares",
                                   "dividend_receivable", "dividend_income", "cost_basis", "realized_pnl")},
                **values}
        self._action_log.append(item)
        self._daily_actions.append(item)

    def _record(self, date):
        for event in self.events:
            if event.record_date == date and event.event_id not in self._eligible:
                if self.state.locked_shares:
                    raise CorporateActionError("登记日存在未上市股份；嵌套分配资格未建模，拒绝猜测")
                quantity = int(self.state.shares)
                self._eligible[event.event_id] = quantity
                self._log(event, "record", date, eligible_shares=quantity)

    def _open_actions(self, previous_date, date):
        due = [event for event in self.events if previous_date < event.ex_date <= date]
        # Validate all quantities before changing any balance. No silent floor of odd rights.
        plans = []
        for event in due:
            if event.event_id in self._settled:
                raise CorporateActionError("公司行为重复结算")
            if event.event_id not in self._eligible:
                raise CorporateActionError(f"{event.event_id} 缺少登记日持仓快照")
            eligible = self._eligible[event.event_id]
            added = _decimal(eligible) * (_decimal(event.bonus_ratio) + _decimal(event.transfer_ratio))
            if added != added.to_integral_value():
                raise CorporateActionError(f"{event.event_id} 产生碎股 {added}；分配算法未建模，拒绝舍入")
            dividend = _money(_decimal(eligible) * _decimal(event.cash_per_share_gross))
            plans.append((event, eligible, int(added), dividend))
        for event, eligible, added, dividend in plans:
            old_basis = self.state.cost_basis * self.state.shares
            self.state.shares += added
            self.state.locked_shares += added
            self.state.cost_basis = old_basis / self.state.shares if self.state.shares else 0.0
            self.state.dividend_receivable += dividend
            self.state.dividend_income += dividend
            self._pending[event.event_id] = {"event": event, "cash": dividend, "shares": added}
            self._settled.add(event.event_id)
            self._log(event, "ex_entitlement", date, eligible_shares=eligible,
                      dividend_gross=dividend, added_shares=added)
        for pending in self._pending.values():
            event = pending["event"]
            if pending["cash"] and event.pay_date <= date:
                amount = pending["cash"]
                self.state.dividend_receivable -= amount
                self.state.cash += amount
                pending["cash"] = 0.0
                self._log(event, "cash_payment", date, effective_date=event.pay_date, cash_paid=amount)
            if pending["shares"] and event.listing_date <= date:
                quantity = pending["shares"]
                self.state.locked_shares -= quantity
                pending["shares"] = 0
                self._log(event, "stock_listing", date, effective_date=event.listing_date,
                          unlocked_shares=quantity)
        if abs(self.state.dividend_receivable) < 1e-8:
            self.state.dividend_receivable = 0.0

    def _limit_reference(self, previous_close, date):
        events = [event for event in self.events if event.ex_date == date]
        cash = sum((_decimal(event.cash_per_share_gross) for event in events), Decimal(0))
        ratio = sum((_decimal(event.bonus_ratio) + _decimal(event.transfer_ratio)
                     for event in events), Decimal(0))
        reference = (_decimal(previous_close) - cash) / (Decimal(1) + ratio)
        if reference <= 0:
            raise CorporateActionError("除权参考价非正；可能为特殊分配，禁止标准公式猜测")
        # Exchange price tick is one fen; ROUND_HALF_UP, not binary bankers rounding.
        reference = reference.quantize(CENT, rounding=ROUND_HALF_UP)
        if reference <= 0:
            raise CorporateActionError("除权参考价低于最小价格单位；拒绝非正参考价")
        rate = _decimal(price_limit(self.bars.symbol, date, self.st))
        up = (reference * (1 + rate)).quantize(CENT, rounding=ROUND_HALF_UP)
        down = (reference * (1 - rate)).quantize(CENT, rounding=ROUND_HALF_UP)
        if up - reference < CENT:
            up = reference + CENT
        if reference - down < CENT:
            down = reference - CENT
        if down <= 0:
            raise CorporateActionError("价格过低，最小变动单位涨跌停无法安全建模")
        return float(reference), float(up), float(down)

    def _checkpoint(self):
        """Only account-sized data, never copy observations/OHLCV on each step."""
        return {"state": copy(self.state), "idx": self.idx,
                "eligible": self._eligible.copy(),
                "pending": {key: value.copy() for key, value in self._pending.items()},
                "settled": self._settled.copy(), "log_size": len(self._action_log),
                "daily_actions": list(self._daily_actions), "trade_size": len(self._trades),
                "equity_size": len(self._equity), "benchmark_size": len(self._benchmark_curve),
                "reward_state": self.reward_calc.__dict__.copy(),
                "benchmark": self._benchmark._checkpoint() if self._benchmark is not None else None}

    def _rollback(self, checkpoint):
        self.state, self.idx = checkpoint["state"], checkpoint["idx"]
        self._eligible, self._pending = checkpoint["eligible"], checkpoint["pending"]
        self._settled = checkpoint["settled"]
        self._daily_actions = checkpoint["daily_actions"]
        del self._action_log[checkpoint["log_size"]:]
        del self._trades[checkpoint["trade_size"]:]
        del self._equity[checkpoint["equity_size"]:]
        del self._benchmark_curve[checkpoint["benchmark_size"]:]
        self.reward_calc.__dict__.clear()
        self.reward_calc.__dict__.update(checkpoint["reward_state"])
        if self._benchmark is not None:
            self._benchmark._rollback(checkpoint["benchmark"])

    def step(self, action):
        checkpoint = self._checkpoint()
        try:
            return self._step(action)
        except Exception:
            # A benchmark/fractional/record failure must not leave a half-filled order.
            self._rollback(checkpoint)
            raise

    def _step(self, action):
        if self.idx >= self.n_steps - 1:
            raise RuntimeError("episode 已结束，请先 reset")
        raw_action = np.asarray(action).reshape(-1)
        if raw_action.size != 1 or not np.isfinite(raw_action).all() or raw_action[0] not in (0, 1, 2):
            raise ValueError("v2 动作必须为 0/1/2 单个离散值")
        act = int(raw_action[0])
        signal_idx, exec_idx = self.idx, self.idx + 1
        previous_date = str(self.bars.dates[signal_idx])
        date = str(self.bars.dates[exec_idx])
        previous_value = self.state.mark_to_market(float(self.bars.close[signal_idx]))
        reference, up, down = self._limit_reference(float(self.bars.close[signal_idx]), date)
        self._daily_actions = []
        self._open_actions(previous_date, date)
        opening = float(self.bars.open[exec_idx])
        halted = float(self.bars.volume[exec_idx]) <= 0
        trade_ok, reason, turnover = False, "", 0.0
        rate = price_limit(self.bars.symbol, date, self.st)
        if act == 1:
            if halted:
                reason = "停牌"
            elif opening >= up - 1e-6:
                reason = f"开盘涨停({up:.2f})不可买入"
            else:
                price = opening * (1 + float(self.cfg.slippage_rate))
                lot = int(self.cfg.lot_size)
                quantity = int(max(0.0, self.state.cash) // price // lot) * lot
                while quantity > 0:
                    notional = quantity * price
                    fee = trade_cost(notional, "buy", self.cfg.commission_rate,
                                     self.cfg.stamp_tax_rate, self.cfg.min_commission)
                    if notional + fee <= self.state.cash + 1e-8:
                        break
                    quantity -= lot
                if quantity <= 0:
                    reason = "可用现金不足以覆盖 1 手及费用（分红应收不可用于买入）"
                else:
                    basis = self.state.cost_basis * self.state.shares + notional + fee
                    self.state.shares += quantity
                    self.state.cost_basis = basis / self.state.shares
                    self.state.cash -= notional + fee
                    self.state.last_buy_idx = exec_idx
                    trade_ok = True
                    turnover = notional / max(previous_value, 1e-9)
                    self._trades.append({"date": date, "signal_date": previous_date, "side": "buy",
                                         "open_price": opening, "price": price, "shares": quantity,
                                         "notional": notional, "fee": fee, "limit_rate": rate,
                                         "limit_reference": reference})
        elif act == 2:
            quantity = self.state.available_shares
            if self.state.shares <= 0:
                reason = "无持仓"
            elif quantity <= 0:
                reason = "仅有待上市股份，暂无可卖股数"
            elif halted:
                reason = "停牌"
            elif self.state.last_buy_idx >= 0 and exec_idx - self.state.last_buy_idx < 1:
                reason = "T+1 冻结（当日买入不可卖出）"
            elif opening <= down + 1e-6:
                reason = f"开盘跌停({down:.2f})不可卖出"
            else:
                price = opening * (1 - float(self.cfg.slippage_rate))
                notional = quantity * price
                fee = trade_cost(notional, "sell", self.cfg.commission_rate,
                                 self.cfg.stamp_tax_rate, self.cfg.min_commission)
                realized = notional - fee - self.state.cost_basis * quantity
                self.state.cash += notional - fee
                self.state.realized_pnl += realized
                self.state.shares -= quantity
                if not self.state.shares:
                    self.state.cost_basis, self.state.last_buy_idx = 0.0, -1
                reason = "待上市股份保留，仅卖出可卖部分" if self.state.locked_shares else ""
                trade_ok = True
                turnover = notional / max(previous_value, 1e-9)
                self._trades.append({"date": date, "signal_date": previous_date, "side": "sell",
                                     "open_price": opening, "price": price, "shares": quantity,
                                     "notional": notional, "fee": fee, "realized_pnl": realized,
                                     "limit_rate": rate, "limit_reference": reference})
        self.state.prev_close = float(self.bars.close[exec_idx])
        value = self.state.mark_to_market(self.state.prev_close)
        self._record(date)
        benchmark_return = 0.0
        if self._benchmark is not None:
            before = self._benchmark.state.portfolio_value
            self._benchmark.step(1)
            after = self._benchmark.state.portfolio_value
            benchmark_return = after / before - 1 if before > 0 else 0.0
            self._benchmark_curve.append(float(after))
        reward = self.reward_calc.compute(previous_value, value, turnover,
                                          self._position_ratio(value), benchmark_return=benchmark_return)
        self._equity.append(float(value))
        self.idx = exec_idx
        info = self._info(turnover, trade_ok, reason, act)
        info.update(execution_date=date, open_price=opening, limit_reference=reference,
                    limit_up=up, limit_down=down, benchmark_return=benchmark_return,
                    step_excess=(value / previous_value - 1 - benchmark_return))
        return self._observation(), float(reward), self.idx >= self.n_steps - 1, False, info

    def _info(self, turnover, trade_ok, blocked_reason, action):
        return {**super()._info(turnover, trade_ok, blocked_reason, action),
                **self.state.snapshot(), "corporate_actions": deepcopy(self._daily_actions),
                "execution_version": EXECUTION_VERSION,
                "tax_policy": "gross_before_personal_dividend_tax"}

    @property
    def corporate_action_log(self):
        return deepcopy(self._action_log)

    @property
    def benchmark_equity(self):
        return np.asarray(self._benchmark_curve, dtype="float64")

    def buy_and_hold(self):
        # An independent identical environment, never the old look-ahead baseline.
        benchmark = CorporateActionExecutionEnv(self.obs, self.bars, self.cfg,
                                                st=self.st, _with_benchmark=False)
        try:
            benchmark.reset()
            for _ in range(self.n_steps - 1):
                benchmark.step(1)
            return benchmark.equity
        finally:
            benchmark.close()

    def close(self):
        if self._benchmark is not None:
            self._benchmark.close()

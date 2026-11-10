"""Independent transaction/aliasing tests; only synthetic in-memory fixtures."""
from copy import deepcopy

import numpy as np
import pytest

from stock_ai.config.schema import EnvConfig
from stock_ai.data.corporate_actions import CorporateAction, CorporateActionError, CorporateActionLedger
from stock_ai.env.corporate_action_env import CorporateActionExecutionEnv
from stock_ai.env.strict_env import ExecutionBars

SOURCE = ("https://example.org/synthetic-transaction-audit",)
DATES = np.asarray(["20230102", "20230103", "20230104", "20230105", "20230106", "20230109", "20230110"])


def event(**changes):
    values = dict(event_id="synthetic-a", kind="distribution", status="implemented",
                  record_date="20230103", ex_date="20230104", pay_date="20230106",
                  listing_date="20230110", cash_per_share_gross="1", transfer_ratio="1",
                  listing_date_source="sina_red_share_listing", sources=SOURCE)
    return CorporateAction(**(values | changes))


def environment(events=None, prices=None, benchmark=True):
    prices = np.asarray(prices if prices is not None else [10, 10, 4.5, 4.5, 4.5, 4.5, 4.5])
    coverage = dict(start="20230101", end="20231231", status="verified",
                    scope="all_shareholder_distributions_and_capital_changes",
                    sources=list(SOURCE), verified_at="2026-10-16",
                    notes="Synthetic mechanism fixture, NOT real historical coverage.")
    ledger = CorporateActionLedger("600519", coverage, tuple(events if events is not None else [event()]))
    bars = ExecutionBars(DATES, prices, prices * 1.01, prices * .99, prices,
                         np.full(len(prices), 1e6), "600519", ledger)
    cfg = EnvConfig(initial_cash=10500, commission_rate=0, stamp_tax_rate=0,
                    slippage_rate=0, min_commission=0, lot_size=100, reward_type="gross")
    return CorporateActionExecutionEnv(np.zeros((len(prices), 3, 9)), bars, cfg,
                                       _with_benchmark=benchmark)


def snapshot(env):
    return deepcopy(dict(account=vars(env.state), idx=env.idx, eligible=env._eligible,
                         pending=env._pending, settled=env._settled, actions=env._action_log,
                         daily=env._daily_actions, trades=env._trades, equity=env._equity,
                         benchmark_curve=env._benchmark_curve, reward=vars(env.reward_calc),
                         benchmark=snapshot(env._benchmark) if env._benchmark is not None else None))


def assert_clean_failure(env, action, error, match=None):
    before = snapshot(env)
    objects = (env.obs, env.bars, env.bars.open, env.bars.close, env.ledger)
    with pytest.raises(error, match=match):
        env.step(action)
    assert snapshot(env) == before
    assert all(a is b for a, b in zip(objects, (env.obs, env.bars, env.bars.open, env.bars.close, env.ledger)))


@pytest.mark.parametrize("action", [-1, 3, 1.5, np.nan, np.inf, [], [0, 1]])
def test_invalid_actions_before_ex_leave_everything_unchanged(action):
    env = environment()
    env.reset(); env.step(1)
    assert_clean_failure(env, action, ValueError)
    env.step(0)
    assert env.state.dividend_income == 1000


@pytest.mark.parametrize("stage", ["_open_actions", "_record"])
@pytest.mark.parametrize("with_benchmark", [False, True])
def test_late_event_failure_rolls_back_asset_trade_and_ledger(monkeypatch, stage, with_benchmark):
    env = environment(benchmark=with_benchmark)
    env.reset(); env.step(1)
    original = getattr(env, stage)
    def fail_after(*args):
        original(*args)
        raise CorporateActionError("injected late failure")
    with monkeypatch.context() as patch:
        patch.setattr(env, stage, fail_after)
        assert_clean_failure(env, 2, CorporateActionError, "injected")
    env.step(2)
    assert env.state.shares == env.state.locked_shares == 1000
    assert len(env.trades) == 2


@pytest.mark.parametrize("stage,index", [("ex_entitlement", 2), ("cash_payment", 4), ("stock_listing", 6)])
@pytest.mark.parametrize("target", ["main", "benchmark"])
def test_each_event_stage_failure_restores_accounts_and_retries_once(monkeypatch, stage, index, target):
    env = environment()
    env.reset(); env.step(1)
    while env.idx < index - 1:
        env.step(0)
    account = env if target == "main" else env._benchmark
    original = account._log
    def fail_after(event, actual_stage, *args, **kwargs):
        original(event, actual_stage, *args, **kwargs)
        if actual_stage == stage:
            raise CorporateActionError("injected stage failure: " + stage)
    with monkeypatch.context() as patch:
        patch.setattr(account, "_log", fail_after)
        assert_clean_failure(env, 2, CorporateActionError, "injected stage")
    env.step(2)
    assert sum(item["stage"] == stage for item in account.corporate_action_log) == 1
    assert env.state.dividend_income == env._benchmark.state.dividend_income == 1000


@pytest.mark.parametrize("method", ["_info", "_observation"])
@pytest.mark.parametrize("target", ["main", "benchmark"])
def test_return_path_failure_after_reward_and_index_rolls_back(monkeypatch, method, target):
    env = environment()
    env.reset(); env.step(1)
    account = env if target == "main" else env._benchmark
    original = getattr(account, method)
    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected return failure")
    with monkeypatch.context() as patch:
        patch.setattr(account, method, fail_after)
        assert_clean_failure(env, 2, RuntimeError, "return failure")
    env.step(2)
    assert env.idx == env._benchmark.idx == 2
    assert env.state.dividend_income == 1000
    assert len(env.trades) == 2


def test_benchmark_fractional_failure_rolls_back_main_trade():
    env = environment(events=[event(transfer_ratio="0.0001")], prices=[10] * 7)
    env.reset(); env.step(0)  # main has no rights; independent benchmark owns 1,000
    assert env.state.shares == 0 and env._benchmark.state.shares == 1000
    assert_clean_failure(env, 1, CorporateActionError, "碎股")


@pytest.mark.parametrize("target", ["main", "benchmark"])
def test_reward_failure_after_peak_update_rolls_back_both_accounts(monkeypatch, target):
    env = environment()
    env.reset(); env.step(1)
    reward = env.reward_calc if target == "main" else env._benchmark.reward_calc
    original = reward.compute
    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise ArithmeticError("injected reward failure")
    with monkeypatch.context() as patch:
        patch.setattr(reward, "compute", fail_after)
        assert_clean_failure(env, 2, ArithmeticError, "reward")
    env.step(2)
    assert len(env.trades) == 2


def test_internal_record_guard_also_rolls_back_trade_on_injected_unsupported_state():
    env = environment(benchmark=False)
    env.reset()
    env.state.shares = env.state.locked_shares = 100
    env.state.cost_basis = 10
    env.state.cash -= 1000
    env.state.mark_to_market(10)
    assert_clean_failure(env, 1, CorporateActionError, "未上市股份")


def test_external_event_logs_and_info_cannot_mutate_internal_ledger():
    env = environment()
    env.reset(); _, _, _, _, info = env.step(1)
    original = snapshot(env)
    info["corporate_actions"][0]["sources"].clear()
    info["corporate_actions"][0]["eligible_shares"] = -100
    log = env.corporate_action_log
    log[0]["sources"].clear(); log[0]["stage"] = "tampered"
    log.clear()
    assert snapshot(env) == original
    env.step(0)
    assert env.state.dividend_income == 1000


def test_nested_pending_listing_rejected_at_construction():
    later = event(event_id="synthetic-b", record_date="20230105", ex_date="20230106",
                  pay_date="20230109", listing_date="20230110", transfer_ratio="0")
    with pytest.raises(CorporateActionError, match="嵌套"):
        environment(events=[event(), later])


@pytest.mark.parametrize("record,listing", [("20230110", "20230110"), ("20230105", "20230105")])
def test_record_on_listing_day_is_not_nested(record, listing):
    later = event(event_id="synthetic-b", record_date=record,
                  ex_date="20230111", pay_date="20230111", listing_date=None, transfer_ratio="0")
    env = environment(events=[event(listing_date=listing), later])
    env.reset()
    for _ in range(len(DATES) - 1):
        env.step(1)
    assert env.state.locked_shares == 0


def test_first_close_has_no_rights_and_does_not_false_positive_nested():
    first = event(record_date="20230102", ex_date="20230103")
    later = event(event_id="synthetic-b", record_date="20230105", ex_date="20230106",
                  listing_date=None, transfer_ratio="0")
    env = environment(events=[first, later], prices=[10] * 7)
    env.reset()
    for _ in range(len(DATES) - 1):
        env.step(0)
    assert env.state.dividend_income == 0


def test_terminated_step_raises_without_changing_reward_or_benchmark():
    env = environment()
    env.reset()
    for _ in range(len(DATES) - 1):
        env.step(0)
    assert_clean_failure(env, 1, RuntimeError, "结束")


@pytest.mark.parametrize("reference,up,down", [("0.05", .06, .04), ("1.05", 1.16, .95)])
def test_limits_use_half_up_and_minimum_tick(reference, up, down):
    env = environment(events=[], prices=[float(reference)] * 7, benchmark=False)
    actual = env._limit_reference(float(reference), "20230103")
    assert actual == (float(reference), up, down)


def test_zero_reference_after_tick_rounding_is_refused():
    env = environment(events=[event(cash_per_share_gross="9.999", transfer_ratio="0")], prices=[10] * 7)
    env.reset(); env.step(0)
    assert_clean_failure(env, 0, CorporateActionError, "最小价格单位")

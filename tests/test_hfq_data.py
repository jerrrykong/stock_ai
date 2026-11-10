"""需求 3：后复权（hfq）数据层的校验与拉取路径测试。

覆盖：
1. 非正价格**直接失败**而非静默剔除；
2. 日期覆盖 / 行数 / 首末价校验；
3. 除权跳空修复统计（复权后 ``|日收益| > 11%`` 次数对比）；
4. hfq 与不复权缓存按交易日对齐，产出真实成交价；
5. 独立缓存目录 ``data/raw_hfq``，绝不覆盖 ``data/raw``；
6. 逐只容错 + 进度可查 + 失败清单；
7. 源选择约束（sina 不支持复权 / qfq 覆盖不足不可用）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_ai.cli.data_cmd import data_fetch_hfq, data_hfq_status
from stock_ai.data.hfq import (
    GAP_RETURN_THRESHOLD,
    HFQ_DIR_NAME,
    compare_adjust,
    execution_prices_from_hfq,
    fetch_hfq_all,
    gap_repair_stats,
    hfq_cache_dir,
    make_hfq_source,
    read_progress,
    validate_hfq_frame,
)
from stock_ai.utils.errors import DataSourceError


def _frame(rows: int = 20, start: str = "2020-01-02", price: float = 10.0):
    dates = pd.bdate_range(start, periods=rows)
    return pd.DataFrame({
        "date": dates,
        "open": price * 1.0, "high": price * 1.02, "low": price * 0.98,
        "close": price, "volume": 1e6, "amount": price * 1e6, "adj_factor": 1.0,
    })


# --------------------------------------------------------------------------- #
# 1. 非正价格直接失败
# --------------------------------------------------------------------------- #
def test_non_positive_price_raises_instead_of_silent_drop():
    """后复权出现负价必须抛错，不能像 validate_frame 那样静默剔行。"""
    df = _frame()
    df.loc[5, ["open", "high", "low", "close"]] = -1.23
    with pytest.raises(DataSourceError, match="非正价格"):
        validate_hfq_frame(df, "600519")


def test_nan_price_raises():
    """NaN 价格同样直接失败。"""
    df = _frame()
    df.loc[3, "close"] = float("nan")
    with pytest.raises(DataSourceError, match="非正价格|NaN"):
        validate_hfq_frame(df, "600519")


def test_missing_columns_and_empty_raise():
    """缺列 / 空表直接失败。"""
    with pytest.raises(DataSourceError, match="缺少列"):
        validate_hfq_frame(pd.DataFrame({"date": ["2020-01-02"]}), "600519")
    with pytest.raises(DataSourceError, match="为空"):
        validate_hfq_frame(pd.DataFrame(), "600519")


def test_too_few_rows_raises():
    """行数低于下限直接失败。"""
    with pytest.raises(DataSourceError, match="行数不足"):
        validate_hfq_frame(_frame(rows=1), "600519", min_rows=2)


def test_validate_sorts_and_dedups_dates():
    """校验后日期升序且重复日期只保留最后一条。"""
    df = _frame(rows=5)
    dup = pd.concat([df, df.tail(1)], ignore_index=True)
    out = validate_hfq_frame(dup, "600519")
    assert list(out["date"]) == sorted(out["date"])
    assert out["date"].is_unique


# --------------------------------------------------------------------------- #
# 2. 日期覆盖 / 行数 / 首末价
# --------------------------------------------------------------------------- #
def test_date_coverage_rows_and_first_last_price():
    """日期覆盖、行数、首末价均可被校验报告正确读出。"""
    df = _frame(rows=30, start="2015-01-05", price=12.5)
    out = validate_hfq_frame(df, "600519")
    assert len(out) == 30
    assert out["date"].min().strftime("%Y%m%d") == "20150105"
    assert out["date"].max() == out["date"].iloc[-1]
    assert float(out["close"].iloc[0]) == pytest.approx(12.5)
    assert float(out["close"].iloc[-1]) == pytest.approx(12.5)


def test_hfq_report_captures_first_last_and_range(tmp_path, monkeypatch):
    """:class:`HfqReport` 输出首末价与区间，可直接用于人工核验。"""
    import stock_ai.data.hfq as hfq_mod

    directory = hfq_cache_dir(tmp_path)
    hfq_df = _frame(rows=10, start="2019-01-02", price=8.0)
    monkeypatch.setattr(hfq_mod, "fetch_hfq_symbol",
                        lambda symbol, start, end, sources=None, **kw:
                        (validate_hfq_frame(hfq_df, symbol), "akshare"))
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    _frame(rows=10, start="2019-01-02", price=2.0).to_parquet(
        tmp_path / "data" / "raw" / "600519.parquet", index=False
    )
    result = fetch_hfq_all(tmp_path, ["600519"], "20190101", "20191231",
                           throttle=0, batch=100, rest=0, raw_dir=tmp_path / "data" / "raw")
    report = result["reports"][0]
    assert report.symbol == "600519" and report.rows == 10
    assert report.first_close == pytest.approx(8.0) and report.last_close == pytest.approx(8.0)
    assert report.start == "20190102"
    assert report.source == "akshare"
    assert report.execution_overlap == 10


# --------------------------------------------------------------------------- #
# 3. 除权跳空修复统计
# --------------------------------------------------------------------------- #
def test_gap_repair_stats_counts_large_moves():
    """``|日收益| > 11%`` 次数被正确统计。"""
    df = _frame(rows=5, price=10.0)
    df["close"] = [10.0, 12.0, 9.0, 9.5, 9.6]
    count, worst = gap_repair_stats(df)
    assert count == 2 and worst == pytest.approx(0.25, rel=1e-6)


def test_hfq_removes_ex_dividend_gap_vs_raw():
    """不复权存在除权向下跳空时，后复权侧的 ``|日收益|>11%`` 次数更少。"""
    dates = pd.bdate_range("2020-01-02", periods=6)
    # 不复权：第 3 天除权，从 20 跳到 10（-50%）
    raw = pd.DataFrame({"date": dates, "close": [20.0, 20.0, 10.0, 10.1, 10.2, 10.3]})
    # 后复权：同一天被复权因子拉回，无跳空
    hfq = pd.DataFrame({"date": dates, "close": [40.0, 40.0, 40.2, 40.4, 40.5, 40.6]})
    compared = compare_adjust(hfq, raw, threshold=GAP_RETURN_THRESHOLD)
    assert compared["raw_gaps"] == 1
    assert compared["adjusted_gaps"] == 0
    assert compared["raw_max_abs_return"] == pytest.approx(0.5)
    assert compared["max_abs_return"] < 0.01


def test_compare_adjust_without_raw_only_reports_adjusted():
    """没有不复权缓存时只报后复权侧，不编造对照数字。"""
    out = compare_adjust(_frame(), None)
    assert out["raw_gaps"] is None and out["raw_max_abs_return"] is None


# --------------------------------------------------------------------------- #
# 4. hfq -> 真实成交价对齐
# --------------------------------------------------------------------------- #
def test_execution_prices_aligned_to_hfq_dates():
    """真实成交价按 hfq 的交易日 inner join，缺日会被剔除且不引入新日期。"""
    hfq = pd.DataFrame({
        "date": pd.bdate_range("2020-01-02", periods=5),
        "open": [11.0] * 5, "high": [11.0] * 5, "low": [11.0] * 5,
        "close": [11.0] * 5, "volume": [100.0] * 5,
    })
    raw = pd.DataFrame({
        "date": list(pd.bdate_range("2020-01-02", periods=3)) + list(pd.bdate_range("2020-02-03", periods=3)),
        "open": [1.1] * 6, "high": [1.1] * 6, "low": [1.1] * 6,
        "close": [1.1] * 6, "volume": [10.0] * 6,
    })
    aligned = execution_prices_from_hfq(hfq, raw)
    assert len(aligned) == 3
    assert aligned["close"].tolist() == [1.1, 1.1, 1.1]
    assert set(aligned["date"]).issubset(set(hfq["date"]))


def test_execution_prices_without_raw_raises():
    """缺少不复权缓存时明确报错（绝不拿复权价当成交价）。"""
    hfq = _frame(rows=4)
    with pytest.raises(DataSourceError, match="不复权"):
        execution_prices_from_hfq(hfq, None)


# --------------------------------------------------------------------------- #
# 5. 独立缓存目录，不覆盖不复权数据
# --------------------------------------------------------------------------- #
def test_hfq_dir_is_separate_from_raw(tmp_path, monkeypatch):
    """hfq 落在 ``data/raw_hfq``，与 ``data/raw`` 完全隔离。"""
    import stock_ai.data.hfq as hfq_mod

    assert hfq_cache_dir(tmp_path).name == HFQ_DIR_NAME == "raw_hfq"
    directory = hfq_cache_dir(tmp_path)
    assert directory == tmp_path / "data" / "raw_hfq"
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_df = _frame(rows=5, price=3.0)
    raw_df.to_parquet(raw_dir / "600519.parquet", index=False)
    hfq_df = _frame(rows=5, price=45.0)
    monkeypatch.setattr(hfq_mod, "fetch_hfq_symbol",
                        lambda symbol, start, end, sources=None, **kw:
                        (validate_hfq_frame(hfq_df, symbol), "akshare"))
    fetch_hfq_all(tmp_path, ["600519"], "20200101", "20201231", throttle=0,
                  batch=100, rest=0, raw_dir=raw_dir)
    assert (directory / "600519.parquet").exists()
    # 不复权缓存逐格未被改动（价格量纲仍是 3.0）
    after = pd.read_parquet(raw_dir / "600519.parquet")
    pd.testing.assert_frame_equal(after, raw_df)
    assert float(after["close"].iloc[0]) == pytest.approx(3.0)
    # hfq 是独立的一份，价格量纲不同
    assert float(pd.read_parquet(directory / "600519.parquet")["close"].iloc[0]) == pytest.approx(45.0)


# --------------------------------------------------------------------------- #
# 6. 逐只容错 + 进度可查
# --------------------------------------------------------------------------- #
def test_fetch_all_is_per_symbol_fault_tolerant_and_reports_progress(tmp_path, monkeypatch):
    """单只失败不中断整批；进度文件可查；失败清单落盘。"""
    import stock_ai.data.hfq as hfq_mod

    good = _frame(rows=8, price=10.0)
    bad = _frame(rows=8, price=10.0)
    bad.loc[2, "close"] = -5.0

    def fake_fetch(symbol, start, end, sources=None, **kwargs):
        if symbol == "000002":
            return validate_hfq_frame(bad, symbol), "akshare"
        return validate_hfq_frame(good, symbol), "akshare"

    monkeypatch.setattr(hfq_mod, "fetch_hfq_symbol", fake_fetch)
    result = fetch_hfq_all(tmp_path, ["600519", "000002", "000001"], "20200101", "20201231",
                           throttle=0, batch=100, rest=0, max_fail_streak=99,
                           raw_dir=tmp_path / "data" / "raw")
    assert result["ok"] == ["600519", "000001"]
    assert result["failed"] == ["000002"]
    assert (hfq_cache_dir(tmp_path) / "fetch_failed.txt").read_text(
        encoding="utf-8").strip() == "000002"
    progress = read_progress(tmp_path)
    assert progress["total"] == 3 and progress["done"] == 2 and progress["failed"] == 1
    assert progress["in_progress"] is False
    assert "600519" in progress["symbols"]
    assert (hfq_cache_dir(tmp_path) / "fetch_progress.json").is_file()
    # 失败的股票绝不落盘（不做部分写入）
    assert not (hfq_cache_dir(tmp_path) / "000002.parquet").exists()


def test_fetch_all_survives_network_exception(tmp_path, monkeypatch):
    """网络异常也只记该只失败，不中断整批。"""
    import stock_ai.data.hfq as hfq_mod

    def boom(symbol, start, end, sources=None, **kwargs):
        if symbol == "000001":
            raise DataSourceError("模拟网络失败")
        return validate_hfq_frame(_frame(rows=6), symbol), "eastmoney"

    monkeypatch.setattr(hfq_mod, "fetch_hfq_symbol", boom)
    result = fetch_hfq_all(tmp_path, ["000001", "600519"], "20200101", "20201231",
                           throttle=0, batch=100, rest=0, max_fail_streak=99)
    assert result["failed"] == ["000001"] and result["ok"] == ["600519"]


# --------------------------------------------------------------------------- #
# 7. 源选择约束
# --------------------------------------------------------------------------- #
def test_hfq_sources_exclude_sina_and_qfq():
    """默认源只含 akshare/eastmoney（sina 无复权、qfq 覆盖不足）。"""
    from stock_ai.data.hfq import HFQ_SOURCES

    assert HFQ_SOURCES == ("akshare", "eastmoney")
    with pytest.raises(DataSourceError, match="sina"):
        make_hfq_source("sina")
    with pytest.raises(DataSourceError, match="不支持的数据源"):
        make_hfq_source("bogus")


def test_hfq_source_is_configured_as_hfq():
    """构造出的源口径必须是 hfq（而非 none/qfq）。"""
    for name in ("akshare", "eastmoney", "tencent"):
        assert make_hfq_source(name).adjust == "hfq"


def test_cli_rejects_invalid_source(tmp_path):
    """``data fetch-hfq --source sina`` 明确拒绝（sina 不支持复权）。"""
    from click.testing import CliRunner

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(data_fetch_hfq, ["--symbols", "600519", "--source", "sina"])
    assert result.exit_code != 0
    assert "sina" in (result.output or "")


def test_cli_hfq_status_reads_progress(tmp_path):
    """``data hfq-status`` 输出进度 JSON。"""
    from click.testing import CliRunner

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(data_hfq_status, [])
    assert result.exit_code == 0
    assert json.loads(result.output)["total"] == 0


def test_cli_help_exposes_new_commands():
    """两条新命令挂在 ``data`` 组下。"""
    from click.testing import CliRunner

    from stock_ai.cli.data_cmd import data

    names = set(data.commands)
    assert {"fetch-hfq", "hfq-status"} <= names
    assert CliRunner().invoke(data, ["fetch-hfq", "--help"]).exit_code == 0


# ---------------------------------------------------------------- #
# 复权有效性校验（hfq_consistency）
# ---------------------------------------------------------------- #

def _frames(dates, adj, raw):
    """构造最小 hfq/raw DataFrame 对。"""
    import pandas as pd
    return (pd.DataFrame({"date": pd.to_datetime(dates), "close": adj}),
            pd.DataFrame({"date": pd.to_datetime(dates), "close": raw}))


def test_consistency_ok_when_only_exdiv_day_differs():
    """除权日有差异、其余日一致 -> 通过。

    注意：单点跳空会同时污染该日与其次日�� ``pct_change``，因此差异天数
    为 2 而非 1，仍远低于 1% 上限。
    """
    from stock_ai.data.hfq import hfq_consistency

    dates = pd.bdate_range("2020-01-01", periods=400).strftime("%Y-%m-%d").tolist()
    raw = [100.0 + i for i in range(400)]
    adj = list(raw)
    adj[30] = raw[30] * 0.6  # 模拟一次除权跳空，仅当天及次日收益差异大
    a, b = _frames(dates, adj, raw)
    res = hfq_consistency(a, b)
    assert res["ok"] is True
    assert res["diff_days"] <= 3


def test_consistency_detects_broken_factor():
    """复权因子异常导致收益差异扩散 -> 判失败。"""
    from stock_ai.data.hfq import hfq_consistency

    dates = pd.bdate_range("2006-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    raw = [10.0 + 0.01 * i for i in range(200)]
    # 因子在 ±30% 间反复跳动，收益序列被严重污染
    adj = [v * (1.0 + 0.30 * (((i % 5) - 2) / 2.0)) for i, v in enumerate(raw)]
    a, b = _frames(dates, adj, raw)
    res = hfq_consistency(a, b)
    assert res["ok"] is False
    assert res["diff_ratio"] > 0.01


def test_consistency_identical_series_always_ok():
    """复权与不复权完全一致 -> 无差异，通过。"""
    from stock_ai.data.hfq import hfq_consistency

    dates = pd.bdate_range("2021-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    raw = [50.0 + 0.2 * i for i in range(80)]
    a, b = _frames(dates, raw, raw)
    res = hfq_consistency(a, b)
    assert res["ok"] is True
    assert res["diff_days"] == 0


def test_consistency_tolerates_empty_input():
    """空/不足输入不抛错，视为跳过。"""
    from stock_ai.data.hfq import hfq_consistency

    assert hfq_consistency(None, None)["ok"] is True
    a, b = _frames(["2020-01-01"], [1.0], [1.0])
    assert hfq_consistency(a, b)["ok"] is True


def test_validate_allows_limited_bad_rows_when_opted_in():
    """显式允许少量异常行时可降级剔除；超限仍失败。"""
    from stock_ai.data.hfq import validate_hfq_frame
    from stock_ai.utils.errors import DataSourceError

    dates = pd.bdate_range("2020-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    close = [10.0 + 0.01 * i for i in range(200)]
    low = list(close)
    low[100] = -0.18  # 单行脏数据
    df = pd.DataFrame({"date": dates, "open": close, "high": close,
                       "low": low, "close": close, "volume": [1000.0] * 200})

    with pytest.raises(DataSourceError):
        validate_hfq_frame(df, "000408")           # 默认严格
    out = validate_hfq_frame(df, "000408", max_bad_ratio=0.01)
    assert len(out) == 199                          # 降级剔除 1 行
    with pytest.raises(DataSourceError):
        validate_hfq_frame(df, "000408", max_bad_ratio=0.0001)  # 超限仍失败

"""v2 vs v3 受控对比：唯一变量是 lambda_leverage（0.01 vs 0.0）。

其余（197 个数据源、执行语义 strict-t1-open-ca-v2、奖励 excess、
种子 42、窗口 20、切分 2006-2021/2022-2023/2024-2025、PPO 超参）逐项相同。

v2 的 metrics.json 早于评估口径修正，缺新字段；零交易口径从
per-episode 的 trades 计数回算，两边口径一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "v2  λ_leverage=0.01": "20261019_105053_7383897c",
    "v3  λ_leverage=0.0 ": "20261020_105633_578f25aa",
}


def split_stats(summary: dict) -> dict:
    """从 evaluation summary 提取口径一致的统计量。"""
    episodes = summary["episodes"]
    traded = [e for e in episodes if e.get("trades", 0) > 0]
    zero = [e for e in episodes if e.get("trades", 0) == 0]
    ex = lambda rows: (float(np.mean([r["friction"]["excess_return"] for r in rows]))
                        if rows else None)
    friction = summary.get("friction", {})
    zero_fraction = len(zero) / len(episodes) if episodes else 0.0
    # v2 的 metrics.json 早于退化守卫，缺 degenerate 字段；按同一阈值回算，
    # 否则会把「字段缺失」误当成「未退化」。
    degenerate = friction.get("degenerate")
    if degenerate is None:
        degenerate = zero_fraction > 0.05
    return {
        "episodes": len(episodes),
        "zero_trade_episodes": len(zero),
        "zero_trade_fraction": zero_fraction,
        "mean_trades": float(np.mean([e.get("trades", 0) for e in episodes])) if episodes else 0.0,
        "excess_all": ex(episodes),
        "excess_traded": ex(traded),
        "excess_zero": ex(zero),
        "mean_reward": float(np.mean([e["reward"] for e in episodes])) if episodes else None,
        "mean_benchmark": friction.get("mean_benchmark_return"),
        "mean_cost_drag_bps": friction.get("mean_cost_drag_bps"),
        "degenerate": bool(degenerate),
        "degenerate_reason": friction.get("degenerate_reason")
        or (f"零交易 episode 占比 {zero_fraction:.1%} 超过 5%"
            if degenerate else "零交易占比未超 5% 阈值"),
        "degenerate_field_present": friction.get("degenerate") is not None,
    }


def main() -> int:
    loaded = {}
    for label, run_id in RUNS.items():
        path = ROOT / "models" / "unified" / run_id / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        loaded[label] = (metrics, manifest)
        print(f"{label}: status={metrics.get('status')}  "
              f"steps={metrics.get('timesteps')}  "
              f"reward_params={json.dumps(manifest['reward_params'], ensure_ascii=False)}")
    print()

    for split in ("val", "test"):
        print(f"===== {split} =====")
        header = (f"{'口径':22s} {'v2 (λ=0.01)':>14s} {'v3 (λ=0.0)':>14s} {'变化':>14s}")
        print(header)
        print("-" * len(header))
        stats = {label: split_stats(loaded[label][0]["evaluation"][split])
                 for label in RUNS}
        fields = [
            ("episode 数", "episodes", "{:.0f}"),
            ("零交易 episode", "zero_trade_episodes", "{:.0f}"),
            ("零交易占比", "zero_trade_fraction", "{:.2%}"),
            ("平均交易笔数", "mean_trades", "{:.1f}"),
            ("超额·全部", "excess_all", "{:+.2%}"),
            ("超额·仅有交易", "excess_traded", "{:+.2%}"),
            ("超额·零交易", "excess_zero", "{:+.2%}"),
            ("平均奖励", "mean_reward", "{:+.4f}"),
            ("基准收益", "mean_benchmark", "{:+.2%}"),
            ("费用拖累(bps)", "mean_cost_drag_bps", "{:.2f}"),
        ]
        for name, key, fmt in fields:
            a = stats["v2  λ_leverage=0.01"][key]
            b = stats["v3  λ_leverage=0.0 "][key]
            sa = fmt.format(a) if a is not None else "—"
            sb = fmt.format(b) if b is not None else "—"
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
                delta = f"{(b - a) / abs(a):+.1%}"
            else:
                delta = "—"
            print(f"{name:22s} {sa:>14s} {sb:>14s} {delta:>14s}")
        print()
        for label in RUNS:
            s = stats[label]
            flag = "退化" if s["degenerate"] else "正常"
            src = "" if s["degenerate_field_present"] else "（按阈值回算，v2 无该字段）"
            print(f"  {label}: 退化守卫={flag}{src} —— {s['degenerate_reason']}")
        print()

    # 关键结论
    v2v = split_stats(loaded["v2  λ_leverage=0.01"][0]["evaluation"]["test"])
    v3v = split_stats(loaded["v3  λ_leverage=0.0 "][0]["evaluation"]["test"])
    print("===== test 段关键对比 =====")
    print(f"  零交易占比   : {v2v['zero_trade_fraction']:.1%}  →  {v3v['zero_trade_fraction']:.1%}")
    print(f"  有交易超额   : {v2v['excess_traded']:+.2%}  →  {v3v['excess_traded']:+.2%}")
    print(f"  平均交易笔数 : {v2v['mean_trades']:.1f}  →  {v3v['mean_trades']:.1f}")
    print(f"  费用拖累     : {v2v['mean_cost_drag_bps']:.2f} bps  →  {v3v['mean_cost_drag_bps']:.2f} bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

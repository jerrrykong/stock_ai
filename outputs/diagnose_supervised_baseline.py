"""关键判别实验：同一套特征上，监督学习能否跑赢买入持有？

这是区分「问题在特征」和「问题在 RL」的最低成本实验：
  - 监督模型用同一批特征预测未来超额。若它在 test 上也拿不到正边际
    → 特征本身没有可执行的 alpha，再怎么调 RL 都是徒劳；
  - 若监督模型能拿到正边际 → 特征里有信号，RL 的问题在观测/容量/探索。

严格无泄漏：特征只用 t 时刻及以前；标签为 t+1..t+20 的未来收益；
切分与训练期一致；归一化器只在 train 段拟合。
只用 torch 实现（环境已有，无新依赖）。

评价口径：择时策略相对「永远满仓」基准的边际
  edge = E[收益 | 看多] - E[收益 | 全样本]
  annual ≈ edge × (每年调仓次数 12.5)   —— 粗估，仅用于横向比较
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.scale.unified import FEATURES  # noqa: E402

FEATURE_DIR = ROOT / "data" / "raw_hfq_v2_train"
SPLITS = {"train": (2006, 2021), "val": (2022, 2023), "test": (2024, 2025)}
HORIZON = 20


def build_panel():
    from stock_ai.scale.unified import relative_features
    frames = []
    for path in sorted(FEATURE_DIR.glob("*.parquet")):
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        df, values = relative_features(raw)
        out = df[["date", "close"]].copy()
        out["symbol"] = path.stem
        for i, name in enumerate(FEATURES):
            out[name] = values[:, i]
        frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["fwd"] = (panel.groupby("symbol")["close"].shift(-HORIZON)
                    / panel["close"] - 1.0)
    panel["year"] = panel["date"].dt.year
    return panel


def split_mask(panel, split):
    start, end = SPLITS[split]
    return ((panel.year >= start) & (panel.year <= end)).to_numpy()


def fit_logistic(X, y, epochs=200, lr=0.05, l2=1e-3, seed=42):
    """numpy 实现的逻辑回归（带 L2），梯度下降 + Adam。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    mw = np.zeros(d); vw = np.zeros(d)
    mb = 0.0; vb = 0.0
    for t in range(1, epochs + 1):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        mw = 0.9 * mw + 0.1 * grad_w; vw = 0.999 * vw + 0.001 * grad_w ** 2
        mb = 0.9 * mb + 0.1 * grad_b; vb = 0.999 * vb + 0.001 * grad_b ** 2
        w -= lr * (mw / (1 - 0.9 ** t)) / (np.sqrt(vw / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return w, b


def fit_mlp(X, y, hidden=32, epochs=60, lr=3e-3, l2=1e-4, seed=42):
    """torch 小 MLP。"""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = len(xt)
    batch = 4096
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            optimizer.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
    net.eval()
    return net


def predict_proba(net, X):
    import torch
    with torch.no_grad():
        logits = net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def main() -> int:
    print("加载面板 ...")
    panel = build_panel()
    data = panel[panel["fwd"].notna()].copy()
    data = data[data["fwd"].abs() < 1.0]  # 剔除极端值（可能是数据问题）

    X = data[list(FEATURES)].to_numpy(dtype="float64")
    y = (data["fwd"].to_numpy() > 0).astype(np.float64)
    fwd = data["fwd"].to_numpy()

    train = split_mask(data, "train")
    val = split_mask(data, "val")
    test = split_mask(data, "test")

    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0)
    std[std <= 0] = 1.0
    Z = np.clip((X - mean) / std, -10, 10)

    print(f"\n样本：train {train.sum():,} / val {val.sum():,} / test {test.sum():,}")
    print(f"正例率：train {y[train].mean():.1%} / test {y[test].mean():.1%}")

    # ---- 逻辑回归 ----
    t0 = time.time()
    w, b = fit_logistic(Z[train], y[train])
    elapsed = time.time() - t0
    print(f"\n--- 逻辑回归（拟合 {elapsed:.1f}s）---")
    print(f"  权重: " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(FEATURES, w)))
    lr_results = {}
    for split, mask in (("train", train), ("val", val), ("test", test)):
        prob = 1.0 / (1.0 + np.exp(-np.clip(Z[mask] @ w + b, -30, 30)))
        signal = prob > 0.5
        if signal.sum() < 50:
            print(f"  {split}: 样本不足")
            continue
        picked = fwd[mask][signal]
        edge = float(picked.mean() - fwd[mask].mean())
        hit = float((picked > 0).mean())
        lr_results[split] = (hit, edge, edge * 12.5, float(signal.mean()))
        print(f"  {split:5s}: 看多占比 {signal.mean():5.1%}  命中率 {hit:5.1%}  "
              f"边际 {edge:+.3%}  年化≈{edge * 12.5:+.2%}")

    # ---- 小 MLP ----
    t0 = time.time()
    net = fit_mlp(Z[train], y[train])
    elapsed = time.time() - t0
    print(f"\n--- MLP(32)（拟合 {elapsed:.1f}s）---")
    mlp_results = {}
    for split, mask in (("train", train), ("val", val), ("test", test)):
        prob = predict_proba(net, Z[mask])
        signal = prob > 0.5
        if signal.sum() < 50:
            print(f"  {split}: 样本不足")
            continue
        picked = fwd[mask][signal]
        edge = float(picked.mean() - fwd[mask].mean())
        hit = float((picked > 0).mean())
        mlp_results[split] = (hit, edge, edge * 12.5, float(signal.mean()))
        print(f"  {split:5s}: 看多占比 {signal.mean():5.1%}  命中率 {hit:5.1%}  "
              f"边际 {edge:+.3%}  年化≈{edge * 12.5:+.2%}")

    print("\n===== 结论 =====")
    for label, table in (("logreg", lr_results), ("mlp", mlp_results)):
        for split in ("val", "test"):
            if split in table:
                hit, edge, annual, cov = table[split]
                print(f"  {label:6s} {split:5s}: 命中率 {hit:5.1%}  边际 {edge:+.3%}  "
                      f"年化≈{annual:+.2%}  看多 {cov:.1%}")
    print()
    print("  test 边际为正 → 特征含 alpha，RL 的问题在观测/容量/探索")
    print("  test 边际≈0 或为负 → 特征无可执行 alpha，调 RL 无用，须先换特征")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

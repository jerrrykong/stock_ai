"""吞吐探针：n_envs=1 vs 4 的实际步速差异。

DummyVecEnv 是**串行**执行各环境（不是子进程），所以 n_envs=4 意味着每步做 4 次
模型前向，而不是一次批量前向。若模型前向开销占比高，n_envs=4 可能反而更慢。

方案 A 若要选 n_envs，先把这个量出来，别拍脑袋。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.config.schema import AppConfig, EnvConfig  # noqa: E402
from stock_ai.scale.unified import (  # noqa: E402
    DEFAULT_RANGES, EXECUTION_CORPORATE_ACTIONS, build_dataset,
)

RUN = ROOT / "models" / "unified" / "20261020_105633_578f25aa"


def main() -> int:
    params = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["reward_params"]
    cfg = AppConfig()
    cfg.env = EnvConfig()
    cfg.env.reward_type = params["reward_type"]
    cfg.env.lambda_drawdown = float(params["lambda_drawdown"])
    cfg.env.lambda_turnover = float(params["lambda_turnover"])
    cfg.env.lambda_leverage = float(params["lambda_leverage"])

    print("构建数据集 ...")
    t0 = time.time()
    data = build_dataset(ROOT / "data" / "raw_hfq_v2_train", 20, DEFAULT_RANGES, None,
                         execution_dir=ROOT / "data" / "raw",
                         execution_version=EXECUTION_CORPORATE_ACTIONS,
                         corporate_actions_dir=ROOT / "data" / "corporate_actions")
    print(f"  用时 {time.time() - t0:.1f}s，train episode {len(data.pools['train'])}")
    print()

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stock_ai.scale.unified import make_pool_env

    for n_envs in (1, 4):
        env = DummyVecEnv([
            lambda i=i: make_pool_env(
                data.pools["train"], cfg.env, cfg.random_seed + i,
                strict=True, strict_episodes=data.strict_pools["train"],
                execution_version=EXECUTION_CORPORATE_ACTIONS)
            for i in range(n_envs)])
        from stable_baselines3 import PPO
        # 用训练真实超参：n_steps=2048 / n_epochs=10 / batch_size=64
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    learning_rate=3e-4, verbose=0, device="cpu", seed=42)
        # 预热（首个 rollout 会触发 CA env 构造开销）
        model.learn(total_timesteps=2048 * n_envs, reset_num_timesteps=False)
        t0 = time.time()
        steps = 2048 * n_envs * 3
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        elapsed = time.time() - t0
        rate = steps / elapsed
        print(f"  n_envs={n_envs}: {rate:7.0f} 步/秒  "
              f"（{steps} 步 / {elapsed:.1f}s）"
              f"  → 900 万步需 {9_000_000 / rate / 60:.0f} 分钟")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

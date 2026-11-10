# 第 5 篇：训练与 PPO 算法详解（train）—— 「演员」如何被训练

> 本篇解读 `src/stock_ai/train/` 子包，是全套文档里**最偏算法**的一篇。你会看到 PPO（近端策略优化）这个著名强化学习算法，是如何在代码里一步步实现的——而且本项目还**手写了一个不依赖 PyTorch 的纯 NumPy 版本**，非常适合用来「读懂 PPO 到底在算什么」。即使你不懂深度学习，跟着本篇也能看懂主干。

---

## 5.1 训练层在流水线中的位置

```
data/samples/train.npz  +  TradingEnv  ──>  Trainer(PPO)  ──>  models/*.zip
                                              │
                              SB3Trainer (主, 需 torch+SB3)
                              NumpyPPOTrainer (兜底, 仅 numpy)
```

---

## 5.2 训练器统一接口（base.py 的 `Enrollment` 协议）

为了让「回测 / 预测」和具体算法解耦，本项目定义了一套端口（Port，Ports & Adapters 模式）：

```python
class Enrollment(Protocol):
    algo_name: str
    def learn(self, total_timesteps, seed=None) -> None: ...
    def predict(self, obs, deterministic=True) -> tuple: ...
    def save(self, path) -> None: ...
    def load(cls, path, env=None) -> "Enrollment": ...
```

`TrainResult` 是训练产物摘要（run 目录、模型路径、算法、后端、步数、最佳奖励、耗时、曲线图）。

> 任何后端（SB3 或 NumPy）都实现这个协议，所以下游 `backtest`/`predict` 完全不用关心背后是哪个库。

**扩展阅读**：端口与适配器模式 https://en.wikipedia.org/wiki/Adapter_pattern

---

## 5.3 后端工厂（factory.py）—— 自动选主/兜底后端

`TrainerFactory.create(cfg, env, preference)` 的逻辑（已在第 1 篇 §1.6 提过，这里看细节）：

1. **前置校验**：非法 `--algo`（只支持 ppo/sac/dqn）或 `--backend`（只支持 auto/sb3/numpy）直接抛 `ConfigError`，**绝不被 auto 回退静默吞掉**。
2. `preference == "numpy"` → 强制纯 NumPy
3. `preference == "sb3"` → 若 SB3/torch 不可用，抛 `ArtifactError`（明确告诉用户去装依赖或换 backend）
4. `preference == "auto"` → 探测到 SB3 可用就用 SB3；**只有**「缺依赖/动态库/运行期环境」类异常才回退 NumPy；参数/配置类错误（如 `ConfigError`）必须原样上抛。

`backend_available(name)` 用 `importlib.util.find_spec` 探测，不抛异常。

---

## 5.4 SB3Trainer（sb3_trainer.py，简述）

当环境装了 `torch + stable-baselines3` 时，主后端就是它。它本质是 **Stable-Baselines3 的 PPO 封装**：把 `TradingEnv` 包成 SB3 能用的 `VecEnv`，调用 `PPO(env).learn()`，并把模型按 SB3 原生 `model.save()` 存成 zip。算法细节由 SB3 库实现，本项目只做「接线 + 配置 + 保存 sidecar json」。

> 想深入 SB3 版 PPO 的细节，看官方文档：https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

---

## 5.5 纯 NumPy PPO 兜底（ppo_numpy.py + policy.py）—— 逐行读懂 PPO

这是本篇重点。它**不依赖任何深度学习框架**，用 numpy 手写了一个完整的 PPO。读懂它，就读懂了 PPO 的本质。

### 5.5.0 PPO 算法总览（再看一遍）

PPO 训练一轮（一个「更新周期」）大致是：

```
① 收集经验（Rollout）：用当前策略在环境里玩 n_steps 步，记下每步的
   (观测, 动作, 旧log概率, 奖励, 价值, 是否结束)
② 计算优势（GAE）：评估每步动作「比平均好多少」
③ 多轮小更新（Epoch）：用裁剪目标函数反复微调策略，多次
④ 回到 ①，用更新后的策略再收集经验
```

下面逐一对应代码。

---

### 5.5.1 策略网络（policy.py 的 `NumpyActorCritic`）

**结构**（两层 MLP，激活 `tanh`）：
```
观测 (展平) ──> Dense(64) tanh ──> Dense(64) tanh ─> ┬─> 策略头 Wpi,bpi ─> logits(3个动作)
                                                    └─> 价值头 Wv,bv  ─> value(标量)
```

- 参数：`W0,b0,W1,b1`（隐藏层）+ `Wpi,bpi`（策略头）+ `Wv,bv`（价值头），参数量很小。
- 初始化：`_init_layer` 用 `gain/sqrt(in_dim)` 缩放；策略头用 `gain=0.01` 让初始 softmax 接近均匀（初始不偏袒任何动作）。
- 还实现了 `orthogonal_init`（PPO 常用的正交初始化，保证训练稳定）。

#### 前向传播 `_forward_all`
```python
h = tanh(x @ W0 + b0)
h = tanh(h @ W1 + b1)
logits = h @ Wpi + bpi          # 策略：未归一化的动作分数
value  = (h @ Wv + bv).reshape(-1)   # 价值：状态值估计
```

#### 取动作 `act`
```python
probs = softmax(logits)          # 把 logits 变成概率分布
action = 按概率采样 (或 deterministic 时取 argmax)
logp = log(probs[action])        # 这个动作被选中的对数概率
```
> `logp` 很关键：PPO 要比较「旧策略」和「新策略」对同一动作的概率变化，靠的就是它。

#### 批量算 logp/value/熵 `log_prob_value`
供训练用，额外算 `entropy`（策略的随机程度，用作探索性奖励）。

---

### 5.5.2 收集经验：`learn()` 的 Rollout 循环

```python
buffer = RolloutBuffer(n_steps, obs_dim)
obs, _ = env.reset(seed=seed)
while total_steps < total_timesteps:
    buffer.reset()
    for _ in range(n_steps):                      # ① 收集 n_steps 步
        action, logp, value = policy.act(flat, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(flat, action, logp, reward, value, done)
        obs = next_obs; total_steps += 1
        if done: env.reset(...)                   # 一回合结束，重开
```

`RolloutBuffer` 是个定长缓冲区，存满 `n_steps` 步后进入更新。

---

### 5.5.3 广义优势估计 GAE（`compute_gae`）

**背景知识：优势（Advantage）**
我们想知道「这一步动作，比『随便走走』好多少」。优势 = 这个动作带来的收益 − 基线（这里基线用价值函数 `V(s)`）。

GAE 是 PPO 用来**平滑估计优势**的方法，引入 λ 平衡「偏差 vs 方差」：

```python
for i in reversed(range(t)):                       # 从后往前算
    nonterminal = 1 - dones[i]
    delta = rewards[i] + gamma * next_value * nonterminal - values[i]
    last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
    advantages[i] = last_gae
    next_value = values[i]
returns = advantages + values                       # 目标回报 = 优势 + 价值
```

- `gamma`（默认 0.99）：未来奖励的折扣因子，越远的奖励权重越低。
- `gae_lambda`（默认 0.95）：GAE 的平滑系数。
- 最后把 `advantages` **标准化**（减均值除标准差），让更新更稳定。

> GAE 原论文：https://arxiv.org/abs/1506.02438

---

### 5.5.4 多 epoch 更新 + 裁剪目标（核心！）

```python
for _epoch in range(n_epochs):                     # ③ 对这批经验反复更新 n_epochs 次
    shuffle(idxs)
    for mb in batches(idxs, batch_size):
        loss, grads = policy.compute_loss_and_grads(
            obs[mb], actions[mb], old_logp[mb], advantages[mb], returns[mb],
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5)
        optimizer.step(grads)                       # 手写 Adam 更新参数
```

#### 裁剪目标函数（`compute_loss_and_grads` 内部）

这是 PPO 的灵魂。核心几行：
```python
ratio = exp(chosen_logp - old_logp)                # 新策略/旧策略 对同一动作的概率比
clipped = clip(ratio, 1-clip_range, 1+clip_range)  # 把 ratio 限制在 [0.8, 1.2]
policy_loss = -mean(min(ratio*adv, clipped*adv))    # 取两者较小者
```

**为什么 clip？**
- `ratio > 1`：新策略比旧策略更倾向这个动作。如果同时 `advantage > 0`（动作好），我们想增大它——但 clip 防止一下增太猛。
- `ratio < 1`：新策略更不倾向这个动作。如果 `advantage < 0`（动作差），我们想减小它——但同样 clip 防过度。
- 取 `min` 再取负（因为是「要最大化」的目标，代码里用最小化负目标实现），本质是**只在更新有益且不过头时改进**，过头的部分被裁掉。

> 这就是「Proximal（近端）」——每次只让策略小幅变化，远离旧策略的更新被拒绝，训练稳如老狗。

#### 总损失
```python
total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
```
- `value_loss`：价值头要准确预测 `returns`（均方误差）。
- `entropy`：策略的熵（随机度），`ent_coef > 0` 时鼓励探索，避免过早僵化。本项目默认 `ent_coef=0.0`。

#### 反向传播（手写梯度）
`compute_loss_and_grads` 末尾手算了每个参数对损失的梯度（`d_logits → d_Wpi/d_bpi/d_Wv/d_bv → 反传到隐藏层 d_W0/d_b0/d_W1/d_b1`），这是一份**教科书级的前向+反向实现**，没有任何框架黑盒。

---

### 5.5.5 手写 Adam 优化器（`AdamOptimizer`）

```python
m = beta1*m + (1-beta1)*g      # 一阶矩（动量）
v = beta2*v + (1-beta2)*(g*g)   # 二阶矩（梯度平方的滑动平均）
p -= lr_t * m / (sqrt(v) + eps) # 自适应步长更新
```
标准 Adam（`beta1=0.9, beta2=0.999, eps=1e-8`），带 bias correction（`lr_t` 里那串修正项）。详见：https://arxiv.org/abs/1412.6980

---

### 5.5.6 模型保存与加载（zip 契约）

`save(path)` 把模型打包成 `models/*.zip`：
```
ppo_numpy.zip
├── weights.npz       # 策略权重 (param::W0, param::b0, ...)
├── hparams.json      # 超参 + 元信息
└── backend.json      # {"backend": "numpy", "algo": "ppo_numpy"}
```
`load(path)` 反向解包，校验文件完整性，重建网络。回测/预测时靠 `backend.json` 自动识别该用哪个加载器。

---

## 5.6 训练超参数（TrainConfig 速查）

| 参数 | 默认 | 含义 |
|---|---|---|
| `algo` | `ppo` | 算法（ppo/sac/dqn） |
| `timesteps` | 200000 | 总训练步数 |
| `learning_rate` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 平滑系数 |
| `clip_range` | 0.2 | 裁剪范围（PPO 核心） |
| `n_steps` | 2048 | 每次 rollout 收集的步数 |
| `batch_size` | 64 | 每次更新的小批量 |
| `n_epochs` | 10 | 每批经验反复更新的轮数 |
| `ent_coef` | 0.0 | 熵系数（鼓励探索） |
| `backend` | `auto` | auto/sb3/numpy |

---

## 5.7 这一层的工程亮点

1. **端口与适配器**：`Enrollment` 协议让 SB3 和 NumPy 后端可插拔，下游零耦合。
2. **纯 NumPy 兜底**：不依赖深度学习库也能完整跑 PPO，保证环境受限时全链路可用。
3. **PPO 全手写**：GAE、clipped surrogate、反向传播、Adam 全部显式实现，可读性强、可调试。
4. **固定种子 + 模型契约**：save/load 完整还原，配合 `data_hash` 保证可复现。

**扩展阅读**
- PPO 原论文：https://arxiv.org/abs/1707.06347
- OpenAI Spinning Up · PPO：https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Adam 优化器：https://arxiv.org/abs/1412.6980
- （进阶）Schulman 关于 PPO 的博客解读：http://joschu.net/blog/kl-approx.html

---

## 5.8 本篇小结

- 训练层用 **PPO** 训练策略网络（一个两层小 MLP，同时输出动作概率和价值）。
- 训练循环：收集经验 → GAE 算优势 → 多 epoch 裁剪更新 → 重复。
- **裁剪（clip）** 是 PPO 稳定的关键：只允许策略小幅偏离旧策略。
- 本项目**手写纯 NumPy PPO**，是理解算法本质的最佳样本；SB3 作为主后端只是它的封装。

下一篇看「训练好的模型怎么被检验和用起来」：[第 6 篇：评估与预测](./06_评估与预测.md)

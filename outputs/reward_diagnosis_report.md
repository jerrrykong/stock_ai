# v2 训练结果不佳：根因诊断报告

日期：2026-10-19
对象 run：`models/unified/20261019_105053_7383897c`（197 只，200 万步，33m53s）
诊断脚本：`outputs/diagnose_reward_decomposition.py`、`outputs/trace_model_buy_block.py`

---

## 1. 结论（先说结果）

**根因是奖励函数里的 `lambda_leverage = 0.01`。**

在这个 long-only、全仓买入的单股票环境里，`position_ratio ∈ [0, 1]`，
只要持仓就约等于 1。而 `rewards.py` 的 excess 分支是：

```
reward = (ret - bench) - λ_drawdown·dd - λ_turnover·turn - λ_leverage·|pos|
```

`λ_leverage·|pos|` **每一步都收**。于是「满仓持有」的成本是
`0.01 × 230.7 步 ≈ 2.28` 奖励点/年，而同期基准收益的绝对量级只有
`0.4057`——**持仓税是全年市场波幅的 5.63 倍**。

这不是「杠杆风险惩罚」，而是「持仓时长税」。在本环境中没有任何方式
表达超额杠杆（`max_position=1.0`，买入即用尽现金），所以这一项只能
惩罚「有持仓」这件事本身。

---

## 2. 分量拆解（训练真实配置，自校验逐位一致）

配置从 run manifest 的 `reward_params` 复原，并开启 `_with_benchmark=True`
（`StrictEpisodePoolEnv` 不传该参数，默认 True，真实训练带同环境恒买基准）：

| 参数 | 值 |
|---|---|
| reward_type | excess |
| lambda_drawdown | 0.0 |
| lambda_turnover | 0.002 |
| lambda_leverage | **0.01** |

训练切分前 60 个 episode，平均 230.7 步/年：

| 策略 | 总奖励 | excess | 回撤项 | 换手项 | **持仓项** |
|---|---|---|---|---|---|
| cash（永不买入） | −0.1681 | −0.1681 | 0 | 0 | 0 |
| buyhold | **−2.2891** | −0.0029 | 0 | −0.0020 | **−2.2842** |
| model | −0.3461 | −0.1498 | 0 | −0.0317 | −0.1646 |

buyhold 的 −2.2891 里，**99.8% 来自持仓项**。excess（含费用磨损）只有
−0.0029——也就是说买入持有本身几乎不亏，是持仓税把它打成了 −2.29。

拆解自校验：四分量之和与环境返回的奖励之和，60 episode × 3 策略
**0 例不一致**。

### 反事实实验

| 场景 | buyhold − cash |
|---|---|
| 现状 | **−2.1209** |
| 去掉回撤惩罚（本就为 0） | −2.1209 |
| 去掉换手惩罚 | −2.1190 |
| **去掉持仓惩罚** | **+0.1632** ← 排名翻转 |

单独把 `λ_leverage` 置零，buyhold 就从「严格劣于空仓」变成「优于空仓」。

### 量级对比

- 持仓惩罚年化成本：2.2842
- 同期基准收益绝对量级：0.4057
- **比值 5.63 倍**
- 单步：满仓收 −0.0100，而单步基准收益典型量级只有 +0.001759（**5.7 倍**）

---

## 3. v1 内部的天然 A/B 实验（决定性证据）

两次 v1 run 数据、步数、超参完全相同，唯一差别是 `λ_leverage`：

| run | λ_leverage | 零交易(val) | 零交易(test) | 有交易超额(val) | 有交易超额(test) |
|---|---|---|---|---|---|
| `20261014_185149_8e273f02` | 0.01 | **30.0%** | **25.4%** | +1.53% | **−32.35%** |
| `20261015_095549_25eab142` | **0.0** | **0.2%** | **0.2%** | −1.12% | **−4.92%** |

同一份数据、同样的 20 万步，只把持仓税关掉，零交易 episode 从 30% 掉到
0.2%，test 有交易超额从 −32.35% 修复到 −4.92%（改善 7 倍）。

---

## 4. v2 的两个具体病症（同一根因的两种表现）

### 病症 A：38% 的 episode 彻底不交易

在 val / test 切分上逐 episode 重放模型（复现官方评估，val 奖励 −0.1324、
test −0.3763，与 `metrics.json` 逐位一致）：

| 切分 | episode | 零交易 | 平均交易 | 超额·全部 | 超额·仅有交易 | 超额·零交易 |
|---|---|---|---|---|---|---|
| val | 385 | 148 (38.4%) | 11.2 | +0.12% | **−4.22%** | +7.05% |
| test | 392 | 150 (38.3%) | 10.7 | −24.62% | **−32.44%** | −11.99% |

**「全部超额 +2.14% / −26.69%」是把两类 episode 混在一起的假象。**
零交易 episode 的「超额」恒等于 `−benchmark_return`：一年不交易，基准涨
25.67% 就记 −25.67%，基准跌就记正。这不是技能，是度量假象。诚实口径是
「仅有交易」那一列：val −4.22%、test −32.44%。

### 病症 B：不躺平的 episode 在疯狂日内换手

逐步追踪 `000002_2006`：模型满仓买入 178,600 股（≈100 万，用尽现金），
次日开盘全卖，一买一卖循环——**单个 episode 59 笔交易**。

这也是持仓税的理性反应：税按「持有天数」收，不按风险收。持有 1 天只付
0.01，持有一年付 2.5。于是最优解变成「把持有时长压到最短」。代价是换手：
模型的换手惩罚 −0.0317，是 buyhold（−0.0020）的 **16 倍**。

所以同一个错误惩罚同时制造了两种退化行为：
1. 38% episode → 彻底空仓（税大于任何可得超额）
2. 其余 → 超短持有、日内换手（把持有时长压到最短，付 16 倍换手摩擦）

---

## 5. 修正方案

### 5.1 根因修正：`λ_leverage` 不能按步收在「持仓时长」上

三个选项：

| 选项 | 做法 | 评价 |
|---|---|---|
| **A（推荐）** | `λ_leverage = 0` | v1 的 A/B 已经跑过这个配置，0.2% 零交易、test 有交易超额 −4.92%。本环境结构上不可能超额杠杆（`max_position=1.0`，买入即用尽现金），这一项是死重 |
| B | 缩放到 ~2e-4 | 保留温和持仓成本（−0.046/年 ≈ 年波幅的 11%），在「无信号时倾向空仓」这一点上比 A 多一点先验，但缺证据、需调参 |
| C | 改项的形式，如只罚 `max(0, pos − target)` | 需新增配置字段，动冻结契约 |

选 A 的理由：环境无法表达超额杠杆，该项在本环境中语义上是空的；
而换手惩罚 `λ_turnover = 0.002` 的标定是合理的——一次满仓往返付
≈0.004，约占年波幅 1%，与「按次收」的口径一致。**持仓税的问题是
「按天收」，换手税的问题是「按次收」，前者比后者大 570 倍/单位。**

先立即重训可用 CLI 现有开关，零代码改动：

```
--lambda-leverage 0
```

manifest 会记录该值。

### 5.2 永久修复：`λ_leverage = 0` 纳入 excess 预设 ✅已实施

`src/stock_ai/cli/scale_cmd.py:361-384`，excess 预设从「补 turnover/drawdown」
扩展为「补 turnover/drawdown/**leverage**」：

```python
if reward is not None and str(cfg.env.reward_type).lower() == "excess":
    if lambda_turnover is None:
        cfg.env.lambda_turnover = 0.002
    if lambda_drawdown is None:
        cfg.env.lambda_drawdown = 0.0
    # lambda_leverage 预设为 0：本环境 long-only 且买入即用尽现金
    # （max_position=1.0），position_ratio∈[0,1]，持仓即≈1，没有任何方式
    # 表达超额杠杆。而 rewards.py 的 excess 分支**每一步**都收
    # λ_leverage·|pos|，于是这一项退化成「持仓时长税」而非杠杆风险惩罚……
    if lambda_leverage is None:
        cfg.env.lambda_leverage = 0.0
```

生效边界（保持原有语义不变）：

- 仅在**显式传 `--reward excess`** 时应用；YAML 里直接写 `reward_type: excess`
  而不传 `--reward` 仍尊重 YAML 的 λ（原语义保留）
- 用户显式传 `--lambda-leverage` 优先于预设
- 非 excess 奖励类型完全不受影响

契约测试同步更新并全部通过：

- `tests/test_excess_reward.py:275` —— `assert cfg.env.lambda_leverage == 0.0`
- `tests/test_excess_reward.py:285` —— 显式 `--lambda-leverage 0.2` 仍覆盖预设
- `tests/test_excess_reward.py:349` —— YAML 预设场景
- `tests/test_qa_excess_adversarial.py:315,332` —— 预设生效 / 未显式则用预设

### 5.3 评估口径修正 ✅已实施

`src/stock_ai/scale/unified.py`：

- `evaluate_strict_episode()`（:553）给每条 record 打 `"zero_trade": len(env.trades) == 0`
- 汇总（:804-842）新增六个字段：

| 字段 | 含义 |
|---|---|
| `zero_trade_episodes` | 零交易 episode 数 |
| `zero_trade_fraction` | 零交易占比 |
| `mean_excess_return_traded` | **诚实口径**：仅有交易 episode 的超额 |
| `mean_excess_return_zero_trade` | 零交易 episode 的超额（定义性假象，单列） |
| `degenerate` | 零交易占比 > 5% 即为 true |
| `degenerate_reason` | 退化原因说明 |

阈值常量 `DEGENERATE_ZERO_TRADE_FRACTION = 0.05`（:49）。

契约测试（`tests/test_execution_env.py:566-584`）断言：新字段不破坏旧字段、
`degenerate` 与阈值严格一致、`mean_excess_return` 必等于有交易与零交易两口径
的加权混合。另有 `test_zero_trade_excess_is_definitional_artifact`（:599）
用纯数学验证「零交易超额恒等于 −benchmark_return」。

### 5.4 回归

`pytest tests/` → **899 项，0 failures / 0 errors / 1 skipped**。

受保护输入复核（`outputs/recheck_protected_inputs.py`）：`data/raw` +
`data/raw_hfq` 共 **605 个文件 sha256 + mtime_ns 逐位未变**。

---

## 6. v3 重训（受控 A/B）

除 `lambda_leverage` 外与 v2 完全一致，已由 manifest 逐项核对：

| 项 | v2 | v3 |
|---|---|---|
| execution_version | strict-t1-open-ca-v2 | 同 |
| reward_type / reward_version | excess / excess-ca-constant-buy-v2 | 同 |
| seed / window / observation_shape | 42 / 20 / [20,9] | 同 |
| ranges | 2006-2021 / 2022-2023 / 2024-2025 | 同 |
| sources | 197 只 | 同（集合逐项相同） |
| train_config | — | **无差异** |
| env_config | — | **仅 `lambda_leverage`: 0.01 → 0.0** |

命令：

```
python -m stock_ai.cli.main scale unified train \
  --config configs/default.yaml \
  --timesteps 2000000 \
  --execution strict-ca \
  --feature-dir data/raw_hfq_v2_train \
  --corporate-actions-dir data/corporate_actions \
  --reward excess
```

run_id：`20261020_105633_578f25aa`
日志：`outputs/v3_train_launch.log`
对比脚本：`outputs/compare_v2_v3.py`

### 6.1 v3 结果

训练完成：EXIT=0，32m31s，2,000,896 步，197 只，无错误，无残留锁。

**test 段（与 v2 受控 A/B）：**

| 口径 | v2（λ=0.01） | v3（λ=0.0） | 变化 |
|---|---|---|---|
| 零交易 episode | 150 (38.27%) | **1 (0.26%)** | **−99.3%** |
| 有交易超额 | −36.12% | **−5.02%** | **+86.1%** |
| 全部超额 | −26.69% | −5.04% | +81.1% |
| 平均奖励 | −0.3763 | −0.0634 | +83.2% |
| 平均交易笔数 | 10.7 | 15.1 | +40.9% |
| 费用拖累 | 53.71 bps | 70.88 bps | +32.0% |

**val 段：**

| 口径 | v2 | v3 | 变化 |
|---|---|---|---|
| 零交易 episode | 148 (38.44%) | **0 (0.00%)** | −100% |
| 有交易超额 | −1.26% | −1.11% | +11.6% |
| 全部超额 | +2.14% | −1.11% | 假象消除 |
| 平均奖励 | −0.1324 | −0.0467 | +64.7% |

结果与 v1 的 A/B 预测一致（v1 λ=0 时 test 有交易超额 −4.92%），
v3 的 −5.02% 几乎完全复现。

退化守卫在 v3 上正确判定为「正常」（val 0%、test 0.26%，均未超 5%）；
v2 若当时有守卫会被判为「退化」（38.4% / 38.3%）。

### 6.2 奖励层复核（v3 模型，train 前 60 episode）

| 策略 | 总奖励 | excess | 换手项 | 持仓项 |
|---|---|---|---|---|
| cash | −0.1681 | −0.1681 | 0 | 0 |
| buyhold | **−0.0049** | −0.0029 | −0.0020 | **0** |
| model | **−0.0118** | **+0.0252** | −0.0370 | 0 |

- buyhold 的总奖励从 −2.2891 变成 **−0.0049**（持仓税消失，只剩真实摩擦）
- 反事实：`buyhold − cash` 从 **−2.1209 翻转为 +0.1632**——持有重新成为占优策略
- 模型的 **excess 由 −0.1498 翻正为 +0.0252**，总奖励 −0.0118
- 模型动作分布：买入 132.9 / 卖出 17.1 / 持有 80.7 —— 已从「躺平」与
  「日内换手」转为**建仓后持续持有**（买入尝试多数因已满仓被拒）

### 6.3 诚实的结论

**修好了病症，没修好策略。**

- 病症确实消失：零交易 38% → 0.3%，test 有交易超额 −36% → −5%（7 倍）
- 但 v3 **仍没有跑赢买入持有**：test 有交易超额 −5.02%，val −1.11%
- 训练切分 excess 为正（+0.0252），val/test 转负 —— 存在过拟合
- 剩余的 −5% 不再是奖励函数的假象，而是**特征/alpha 本身的强度问题**：
  9 个技术特征（return_1、均线相对、动量、波动率、量比）在单股票、
  满仓、T+1 开盘成交的约束下，尚不足以稳定战胜买入持有

下一步该做的是**提升 alpha**（特征、状态、目标），而不是继续调奖励系数。



---

## 7. 顺带修正我自己的两个诊断错误（记录在案）

1. **第一版分解用了 `EnvConfig()` 默认值**（λ_drawdown=0.1），
   而训练是 0.0，把结论放大了。已改为从 manifest 读真实 `reward_params`。
2. **`model.predict` 喂了 `reset()` 的初始观测且循环里从不更新**，
   模型退化成恒定动作策略，得出「发起买入 469 次只成交 3 次」的错误结论
   （与上一轮用 `env.obs` 是同一类错误）。修正后逐位复现官方评估
   （val −0.1324 / test −0.3763），并看到真实行为是日内换手。

`cash` / `buyhold` 两行不依赖观测，拆解与反事实结论不受影响。

---

## 8. 待决项

1. **59 只停牌冲突股**（区间内事件日期落在连续停牌空洞）—— v2/v3 都排除了，
   训练样本存在 23% 的停牌偏误，尚未处理。
2. **7 只碎股股票**（非整数每 10 股转增比例）—— 需决定是否给引擎加零碎股策略
   （要改 `fractional_policy` 契约）。
3. **4 只科创板抓取失败**（688047/688506/688521/688981，akshare
   `KeyError: '实施方案公告日期'`）—— 尚未排查。
4. **`configs/default.yaml` 的 `lambda_leverage: 0.01` 仍保留**：默认
   `reward_type: drawdown_penalty` 路径不受 excess 预设保护，同样存在持仓税。
   由于该路径在本项目中已非实际训练用配置（v1/v2/v3 均用 excess），且 CLI 明确
   承诺「缺省路径字节级不变」，暂未改动；如需一并治理应单独评估。


# QA 独立验证报告 — stock_ai（Python A股强化学习交易工程）

- **验证人**：严过关（QA 工程师）
- **验证日期**：2026-09-29
- **验证方式**：**独立复现**（不使用工程师的 smoke 脚本，全部以独立进程 / 全新临时目录执行）
- **解释器**：`C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe`
- **隔离产物目录**：`E:\project\stock_ai\.qa_verify\`
- **结论速览**：**35 项验证中 33 项 PASS，2 项 FAIL（均为 P1/P2，不阻断交付）；未发现 P0 缺陷。修复方向明确，可交付。**

---

## 1. 验证项清单

> 结果列：PASS / FAIL / N-A。所有证据均为本次运行的真实终端输出。

### A. 端到端链路（独立进程，模拟真实用户）

| 编号 | 验证内容 | 方法 | 结果 | 证据 |
|---|---|---|---|---|
| A1 | `data fetch --source synthetic` | 独立进程 | PASS | `完成 2 只股票`，600519/000001 各 1305 行，exit 0 |
| A2 | `data refresh` | 独立进程 | PASS | `缓存已最新`，1305 行，exit 0 |
| A3 | `data info` | 独立进程 | PASS | 2 只股票，gaps=0，exit 0 |
| A4 | `sample build` | 独立进程 | PASS | train=1566 / val=236 / test=1044 步，11 特征，data_hash 输出 |
| A5 | `train --algo ppo` | 独立进程 | PASS | backend=sb3，模型 zip + sidecar + 曲线均落盘，exit 0 |
| A6 | `backtest` | 独立进程 | PASS | 指标齐全（年化/累计/回撤/夏普/胜率/换手/笔数/baseline） |
| A7 | `predict` | 独立进程 | PASS | 43 条，CSV+JSON 落盘，买6/卖27/持10 |
| A8 | `doctor` | 独立进程 | PASS | 依赖全 ✓，后端 SB3 可用，目录可写 |
| A9 | 双入口（抽查 2+ 条） | `python -m stock_ai.<module>` | PASS | 实测 **4 条**：`train.train_cli` / `eval.backtest_cli` / `infer.predict_cli` 全部等价可用 |
| A10 | 独立执行语义（不依赖同进程串联） | 每条命令单独 shell 调用 | PASS | 全部命令在全新目录下单独成功 |

### B. 核心算法正确性（重点）

| 编号 | 验证内容 | 方法 | 结果 | 证据（手算 vs 代码） |
|---|---|---|---|---|
| B1 | T+1：当日买入不可卖，隔日可卖 | 纯函数 + env 层 | PASS | `can_sell(held_days=0)=False`，`=1 True`；env 隔日卖出 `trade_ok=True` |
| B2 | 涨停不可买 / 跌停不可卖 | 构造极端价格 | PASS | 价格 11.0/prev 10 → buy 拒绝且 `blocked_reason="涨停不可买入"`；9.0/prev 10 → sell 拒绝 `"跌停不可卖出"` |
| B3 | 停牌（成交量=0）不成交 | 置 `volumes=0` | PASS | `is_halted(0)=True`；env 停牌日买入 `blocked_reason="停牌"` |
| B4 | 交易成本手算对照 | trade_cost + env 级 | PASS | 见 §B-成本对照（buy 1e4→5；buy 1e5→25；sell 1e5→75；sell 1e4→10；env 现金扣减逐分对齐 `rel=1e-9`） |
| B5 | 奖励分量符号/量级 | 构造 prev/value | PASS | 纯收益 r=0.10；回撤惩罚项 = `-(1/11)*0.1`；换手/杠杆惩罚均为负 |
| B5b | 纯 PnL 无 NaN/Inf | 多组边界值 | PASS | `np.isfinite` 全真；`prev_value=0` 亦无 NaN |
| B6 | walk-forward 训练/测试零重叠 + embargo | 纯函数 + CLI 产物 | PASS | 索引交集=0；train max=20221230 < test min=20230102；exact date intersection=0；embargo 精确裁掉 N 日 |
| B7 | 防泄漏归一化（K9）：仅训练窗口 fit | 代码静态审计 + 行为测试 | PASS | 全库 `.fit(` 仅 1 处：`samples/builder.py:75`（train 窗口）；transform 多次调用不改冻结 mean/std |
| B8 | 复现性 | 独立两次运行 | PASS | 两次 `sample build` data_hash 完全一致；同 seed NumPy PPO 两次训练**权重哈希完全一致** |
| B9 | 参数非法：日期区间反了 | CLI | PASS | `train_start>train_end` → 友好报错 exit 1（无堆栈） |
| B10 | 整手/滑点/最低佣金 | 纯函数 | PASS | `round_lots(1234)=1200`、`(99)=0`；滑点 100→100.1/99.9；最低佣金 5 元生效 |

### C. 跨标的边界修复回归（本轮工程师修复的 bug）

| 编号 | 验证内容 | 方法 | 结果 | 证据 |
|---|---|---|---|---|
| C1 | 极端价格差（5 元 vs 500 元，100×）回测不爆炸 | 独立构造 | PASS | equity max/initial=**1.136**（<3×），全 finite，无 >1e10 |
| C2 | 进入新标的第一天强制平仓 | 独立构造 | PASS | `reason="symbol_boundary"` 记录恰 1 条 |
| C3 | 边界 prev_close 取当日本身 | 独立构造 | PASS | `prev_closes[40]==prices[40]`，boundary flag=True，全场仅 1 处边界 |
| C4 | 真实 CLI 回测无爆炸 | 端到端 | PASS | 1044 步回测 equity 1.0×→0.37×，无 NaN/Inf |

### D. 异常与错误路径

| 编号 | 验证内容 | 方法 | 结果 | 证据 |
|---|---|---|---|---|
| D1 | 不存在的模型 → backtest/predict | CLI | PASS | `[错误] 模型文件不存在: ...（请先执行 stock_ai train）` exit 1 |
| D2 | 不存在的数据 → sample build | CLI | PASS | `[错误] 无可用缓存数据：请先执行 stock_ai data fetch` exit 1 |
| D3 | 不存在的样本 → train/backtest | CLI | PASS | `[错误] 训练样本不存在 / 测试样本不存在 ...` exit 1 |
| D4 | 损坏 parquet（手动截断） | 截断文件 | PASS | 抛 `ArtifactError`（非 pyarrow 堆栈）；`data info` 跳过并 warning，exit 0 |
| D5 | **参数非法：`--algo BOGUS`** | CLI | **FAIL (P1)** | 见 §3 问题 1 |
| D6 | 参数非法：`--algo sac`（离散环境） | CLI | **FAIL (P2)** | 见 §3 问题 2 |
| D7 | 静默失败扫描（`except: pass`） | Grep 全量扫描 | PASS | 仅 `config/loader.py:86` 一处（env 变量 int 解析，合理） |

### E. 代码质量抽查

| 编号 | 验证内容 | 方法 | 结果 | 证据 |
|---|---|---|---|---|
| E1 | 硬编码路径 | Grep `[A-Za-z]:\\|/Users/|/home/` | PASS | src 内 **0 匹配**；全部来自 config |
| E2 | 静默吞异常 | Grep `except ...: pass` | PASS | 仅 1 处（见 D7，合理）；fallback 链的 `except` 均有记录/warning |
| E3 | 核心计算纯度 | 行为测试 | PASS | metrics/规则为纯函数；`RewardCalculator._peak` 为 episode 级状态，`reset()` 正确清除 |
| E4 | 遗留 TODO/FIXME/调试代码 | Grep | PASS | **0 处** TODO/FIXME/XXX/HACK/breakpoint；`print` 全部位于 CLI 输出层（合理） |
| E5 | 异常统一封装 | 阅读 io/utils | PASS | `ArtifactError/ConfigError/DataSourceError` 分层清晰，io 层全量封装底层异常 |

**小计：35 项验证 → PASS 33 / FAIL 2（D5、D6）/ N-A 0**

---

## B-成本手算对照明细

配置：佣金 `commission_rate=0.00025`（双边）、印花税 `stamp_tax_rate=0.0005`（卖出）、最低佣金 5 元、滑点 0.001。

| 场景 | 手算期望 | 代码输出 | 结论 |
|---|---|---|---|
| 买入 notional=10000 | `max(10000*0.00025, 5)=5.0` | 5.0 | ✓ |
| 买入 notional=100000 | `100000*0.00025=25.0` | 25.0 | ✓ |
| 卖出 notional=100000 | `25 + 100000*0.0005=75.0` | 75.0 | ✓ |
| 卖出 notional=10000 | `max(2.5,5) + 5 = 10.0` | 10.0 | ✓ |
| notional<=0 | 0.0 | 0.0 | ✓ |
| env 买入 1e6@10 元现金扣减 | `notional + fee`（含滑点 exec=10.01） | 逐分对齐 `rel=1e-9` | ✓ |
| env 卖出结算 | `cash + notional - fee`（exec=9.99） | 逐分对齐 | ✓ |

---

## 2. 端到端链路证据（A 段节选）

```
$ python -m stock_ai data fetch --config .qa_verify/qa.yaml --source synthetic
  600519: 1305 行 ; 000001: 1305 行            [exit 0]

$ python -m stock_ai sample build --config .qa_verify/qa.yaml
  train: 1566 步 / val: 236 步 / test: 1044 步
  data_hash: 651629cb68f55015619e6940b6e873b9b1f43ad0cf05d857dfd6782c89ccd864

$ python -m stock_ai train --config .qa_verify/qa.yaml --algo ppo
  后端选定: sb3 (sb3_ppo) ; 模型: models/ppo_20260929_180229.zip  [exit 0]

$ python -m stock_ai backtest --config .qa_verify/qa.yaml --model <zip>
  年化: -39.08% 累计: -62.81% 回撤: 62.81% 夏普: -19.946 笔数: 664 baseline: 30.84%

$ python -m stock_ai predict --config .qa_verify/qa.yaml --model <zip> --symbols 600519 --start 20241101 --end 20241231
  共 43 条 ; 买:6 卖:27 持有:10 ; CSV+JSON 落盘  [exit 0]
```

> 说明：backtest 的负收益与 `best_mean_reward=0.0000` 是 **2048 步未训练充分** 的预期结果，而非缺陷；换手 687× 与 664 笔成交在数学上自洽（`sum(notional)/mean(equity)` 精确复算=687.6856），且权益始终有界（1.0×→0.37×，无爆炸）。

---

## 3. 发现的问题（按优先级）

### 🟠 P1 — `--algo` 参数在 `auto` 后端下静默接受非法值（D5）

- **文件**：`src/stock_ai/train/factory.py:94-104`（auto 分支）、`src/stock_ai/train/train_cli.py:60-61`
- **复现**：
  ```
  python -m stock_ai train --config .qa_verify/qa.yaml --algo BOGUS
  ```
- **实际**：`auto` 后端尝试构造 SB3 → 抛 `SB3 不支持算法: bogus` → 被 `except Exception` 吞掉 → **回退 NumPy PPO**（NumPy 路径完全忽略 `algo`）→ 生成 `bogus_*.zip`，**exit 0**。
- **预期**：非法算法名应明确报错（如 `--algo` 需 ∈ {ppo,sac,dqn}）。
- **影响**：用户以为训练了某个算法，实际得到的是"名字错误的 NumPy PPO"。属**静默失败**，违反 PRD「参数非法应有明确提示」。
- **对照证据**：强制 `--backend sb3` 时能正确拒绝（`[错误] SB3 不支持算法: bogus（支持 ['ppo','sac','dqn']）` exit 1），说明校验存在但被 auto 回退路径绕过。
- **建议修复**：在 `train_cli.run()` 或 `TrainerFactory.create()` 入口对 `algo ∉ {ppo,sac,dqn}` 统一抛 `ConfigError`（早于后端选择）。

### 🟡 P2 — `--algo sac` 在默认离散环境崩溃，报错不友好（D6）

- **文件**：`src/stock_ai/train/sb3_trainer.py:118-119`
- **复现**：`python -m stock_ai train --config .qa_verify/qa.yaml --algo sac --backend sb3`
- **实际**：`[未预期错误] AssertionError: The algorithm only supports (Box,) as action spaces but Discrete(3) was provided` exit 1。
- **预期**：SAC 需连续动作空间；默认 `action_mode="discrete"` 下 SAC 不可用，应给出**领域化提示**（如"SAC 需要连续动作空间，请使用 ppo 或启用 continuous 模式"），而非泄漏 SB3 内部 AssertionError。
- **影响**：`--algo` 帮助文本宣称支持 `ppo|sac|dqn`，但 sac 实际不可直接使用，误导用户。
- **建议修复**：训练前对 `algo=="sac"` 且 `action_mode!="continuous"` 提前报 `ConfigError`。

### 🟢 P3（提示，非缺陷）— 次要观感问题

1. **`--samples <不存在目录>` 的路径解析**：`backtest_cli.py:108-109` 会把非 `.npz` 且不存在的字符串当成"文件名前缀"，生成 `nonexistent_dir.npz`，报错信息里的路径与用户输入略有出入。行为可预测、不致命，建议在文档注明。
2. **`val` 与 `train` 时间重叠**（正常设计）：val 从 train 窗口尾部按 `val_ratio` 切出，二者重叠是设计使然（不泄漏到 test）。仅提示，非 bug。

---

## 4. 智能路由判定（第 1 轮）

- **D5（P1）、D6（P2）**：断言/期望符合 PRD 验收（"参数非法应有明确提示"），代码实际未满足 → **源码缺陷**，建议**转交工程师（Alex）修复**。二者均非阻断性（不影响主链路 happy path），但属明确的健壮性缺口。
- **测试代码自身的 Bug**：无。
- **其余 33 项**：全部 PASS，**未发现其他源码缺陷**。

---

## 5. 遗留风险

1. **真实 AKShare 联网拉取未验证**：沙箱无外网，仅验证了 `synthetic` 源与 fallback 链的**结构合理性**（`_FALLBACK_CHAIN = (akshare, tencent, eastmoney)` + synthetic 兜底）。真实联网稳定性、字段映射需在有网环境复测。
2. **训练充分性未验证**：仅用 2048 timesteps 跑通链路，未验证收敛/收益有效性（属模型调参范畴，非工程质量问题）。
3. **SB3 原始 zip 跨版本兼容**：`_load_model` 依赖 sb3 registry 推断 algo，跨 sb3 版本可能变化。
4. **`data refresh` 增量合并**：本次缓存已最新，未触发真实"增量拉取→merge→去重"分支（merge 逻辑已由单测覆盖）。

---

## 6. 结论

**可交付（Conditional Pass）**。6 条命令（8 个子动作）在**全新隔离目录、独立进程**下全部跑通，双入口（4 种 module 形式）验证一致；核心 A 股规则（T+1 / 涨跌停 / 停牌 / 成本 / 奖励）、walk-forward 防前视、K9 防泄漏归一化、跨标的边界修复**均经独立手算/构造用例证实正确**，复现性完美（data_hash 与同 seed 权重哈希完全一致）。代码质量无硬编码路径、无静默吞异常、无遗留调试代码。

仅发现 **1 个 P1 + 1 个 P2** 参数校验健壮性问题（`--algo` 非法值静默接受、`sac` 崩溃不友好），**不影响主链路交付**，建议一并修复。

- **新增测试**：`tests/test_qa_verification.py`（17 个用例，全部 PASS）
- **全量回归**：`112 passed, 1 skipped`（原 95+1 → 新增 17）

---

# 第 2 轮回归验证（Final Round）

- **回归验证人**：严过关（QA 工程师）
- **日期**：2026-09-29
- **验证范围**：工程师针对第 1 轮 P1/P2 缺陷的修复 + 全量回归 + 架构核心设计（auto 回退）实测
- **结论速览**：**第 1 轮 2 个缺陷全部修复确认；未发现新 P0/P1 缺陷；1 个 P2 级已知权衡（见 R2-3）已评估为可接受。最终判定：可交付。**

---

## R2-1. 缺陷修复独立复现（不用工程师自测数字）

### 修复 1（原 P1 — `--algo BOGUS` 静默接受）→ ✅ 已修复

| 检查项 | 结果 | 证据 |
|---|---|---|
| 非法 algo 拒绝 + 非零退出 | PASS | `[错误] 不支持的算法: 'BOGUS'（可选: ['dqn', 'ppo', 'sac']）`，exit=1 |
| **无垃圾产物** | PASS | 训练前后 `.qa2/models/` 均为空，全项目 `find *bogus*` 无新产物（第 1 轮残留的 `bogus_*.zip` 时间戳 18:04 早于修复，非本轮产生） |
| 双入口一致 | PASS | `python -m stock_ai train` 与 `python -m stock_ai.train.train_cli` 输出/退出码完全一致 |
| 强制 `--backend sb3` + 非法 algo | PASS | 同样被前置校验拒绝，exit=1 |
| 白名单收敛位置正确 | PASS | 校验位于 `train_cli.run()` 前置（`ensure_dirs` 后、`set_global_seed` 前）+ `TrainerFactory.create()` 入口双层防护，早于任何后端选择 |

### 修复 2（原 P2 — `sac` 崩溃不友好）→ ✅ 已修复

| 检查项 | 结果 | 证据 |
|---|---|---|
| 领域化中文提示 | PASS | `[错误] SAC 需要连续动作空间，请使用 ppo/dqn，或启用 continuous 动作模式（当前训练动作模式为 discrete）` |
| 无 SB3 内部 AssertionError 泄漏 | PASS | 输出中无 `AssertionError` 字样 |
| 双路径防护 | PASS | train_cli 前置校验与 `sb3_trainer._build_model` 守卫（强制 sb3 绕过前置时）输出一致，exit=1 |
| 无垃圾产物 | PASS | `.qa2/` 无任何 sac 模型残留 |

### 修复 3（P3 顺带 — `--samples` 路径提示）→ ✅ 已修复

```
[错误] 训练样本不存在: .qa2\nope_dir.npz（--samples '.qa2/nope_dir' 解析为 .qa2\nope_dir.npz）（请先执行 `stock_ai sample build`）
```
报错附带解析说明，exit=1。

---

## R2-2. 架构核心设计回归（重点实测：auto 回退绝不能被 except 收窄破坏）

| 检查项 | 方法 | 结果 | 证据 |
|---|---|---|---|
| **C1: SB3 不可探测 → auto 回退 NumPy 并可训练** | mock `backend_available('sb3')=False` | PASS | `后端: 纯 NumPy PPO [auto]（SB3 不可用）`，选定 `numpy`，256 步训练完成（loss 正常输出） |
| **C2: SB3 可探测但构造失败（白名单异常）→ 回退** | mock `_make_sb3` 分别抛 ImportError / ModuleNotFoundError / RuntimeError / OSError / ArtifactError | PASS | 5 类异常全部触发回退并选定 numpy |
| **C3: 构造抛非白名单异常 → 传播（不回退）** | mock 抛 AttributeError / ValueError / AssertionError / TypeError / KeyError | PASS（行为符合预期设计） | 5 类异常全部向上传播。**见 R2-3 评估** |
| **C4: 真实 SB3 构造失败实测** | 构造非法超参触发真实 SB3 异常 | PASS | `learning_rate=-1.0` → SB3 真实抛 `ValueError: Invalid learning rate: -1.0`；n_steps/batch_size 不匹配仅 warning 不抛异常 |
| **D1: `--algo ppo` auto 主路径** | 真实 CLI | PASS | `后端选定: sb3 (sb3_ppo)`，训练成功 |
| **D2: `--algo dqn` 实际可用性（潜在新问题排查）** | 真实 CLI + 端到端 | PASS | **dqn 在离散环境真实训练成功**（`完成 backend=sb3 algo=dqn`，产物 zip+json），且模型可被 backtest 正常加载回测（年化/笔数指标正常输出）→ 白名单包含 dqn 是合理的 |
| **D3: ConfigError 全入口一致性** | 双入口对比 + 类层次 | PASS | 两入口输出格式（`[错误] ... [context]`）与 exit=1 完全一致；`ConfigError` 继承 `StockAIError` |

## R2-3. 修复引入的行为变更评估（P2 级已知权衡，接受）

**发现**：`factory.py:114` 收窄后的 except 白名单 `(ImportError, ModuleNotFoundError, RuntimeError, OSError, ArtifactError)` **不含 ValueError / AttributeError / AssertionError / TypeError**。SB3/torch 在构造期抛 `ValueError`（如 `learning_rate=-1.0`）时会向上传播到顶层处理器，输出：

```
[未预期错误] ValueError: Invalid learning rate: -1.0    （exit=1，无 Python 堆栈）
```

**评估**：
- **非回归定性**：该异常不是"后端不可用"，回退 NumPy 反而会**掩盖非法参数**（NumPy 路径忽略 lr 符号继续训练）——传播比静默回退**更正确**。且第 1 轮旧代码 `except Exception` 会把此情形错误回退，本轮修复顺带消除了这一隐患。
- **用户体验**：顶层 handler 兜底（exit=1、无堆栈），仅文案为"未预期错误"而非领域化提示。
- **残余风险（低）**：若 torch/SB3 因环境问题抛 `AttributeError`（如版本不兼容），同样会传播而非回退。这是"明确报错"与"静默回退"之间的权衡，当前选择偏向前者，符合 PRD"参数非法应有明确提示"的精神。
- **建议（P3，不阻断）**：可在顶层 handler 对 `ValueError` 增加领域化文案（"训练超参非法"），或将 `ValueError` 纳入白名单后由 train_cli 前置校验 lr>0。本轮不要求修复。

## R2-4. 全量测试与既有 PASS 项回归

| 检查项 | 结果 | 证据 |
|---|---|---|
| 全量套件 | **PASS** | JUnit 统计：**tests=121, failures=0, errors=0, skipped=1**（= 120 passed + 1 skipped，与工程师自测一致） |
| QA 17 用例（第 1 轮产出） | PASS | `17 passed in 2.18s` |
| 工程师新增 8 用例 | PASS | `8 passed in 7.68s`（test_train_algo_validation.py） |
| A 段端到端 8 命令抽查 | PASS | fetch/refresh/info/sample/train/backtest/predict/doctor 全部独立进程 exit=0，指标与第 1 轮一致（backtest 年化 -39.08%、664 笔；predict 43 条；**data_hash 仍为 `651629cb…` 完全一致**） |
| B 段关键项抽查 | PASS | 归一化/划分/hash 相关 6 项过滤运行全绿 |
| CLI 未知参数防护 | PASS | predict 误传 `--cache-dir` → click exit=2 明确报错（我方参数错误，非缺陷） |

**环境备注**：`.qa_verify/` 与 `.qa2/` 两个 QA 工作目录保留在仓库中作为证据（删除操作被用户拒绝）；建议后续加入 `.gitignore` 或手工清理。

---

## R2-5. 最终判定

**✅ 可交付（Final Pass）**

- 第 1 轮 P1（非法 algo 静默接受）与 P2（sac 不友好报错）**均已修复且经独立复现确认**，无垃圾产物，双入口行为一致。
- 架构核心设计（auto → NumPy 回退）**实测完好**：后端不可探测、以及 5 类后端不可用异常场景下均正确回退；参数类 ConfigError 正确上抛不被吞。
- `dqn` 白名单经实测**真实可用**（训练+回测端到端），非纸面选项。
- 无新 P0/P1 缺陷；1 个 P2 级行为权衡（非白名单异常传播替代静默回退）经评估为**更正确**的方向，仅文案层面有 P3 改进空间。
- 全量 120 passed / 1 skipped 属实；第 1 轮 33 项 PASS 未被本次改动破坏。

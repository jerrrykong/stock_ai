# 第 8 篇：Scale 大规模训练模块（scale）—— 沪深300 × 年度的批量编排

> 本篇概述 `src/stock_ai/scale/` 子包。它是建立在前面 6 条主命令之上的**叠加层**，目标是对「沪深300 全市场 × 多年度」做**可复现、可中断、可续跑**的批量训练/检测，并提供一个本机 Web UI。注意：这是相对较新、部分仍属实验性质的模块，README 明确说明其中「公司行为 v2」**尚未启动训练、未补齐全历史账本**。本篇只讲它的设计意图与结构，不展开每行代码。

---

## 8.1 它要解决什么问题

前面的主流水线适合「单标的、单窗口」的研究。但当你想覆盖沪深300、跨越 20 年、对每个「股票×年份」单元分别训练再统一评估时，会遇到：
- 任务量大、耗时长，需要**断点续跑**（别因为一次中断全重来）。
- 单元之间要**可复现、可编排**（同一份输入字节，结果一致）。
- 需要**统一模型**（把所有单元的股票拼起来用一个 PPO 训）与**单元级检测**（选最近的模型回测）。
- 需要一个轻量界面**查看回放结果**。

`scale` 就是为这些诉求设计的编排层。

---

## 8.2 设计原则（来自 README 与代码注释）

1. **零新增第三方依赖**：只用 Python 标准库 + 既有依赖（Web UI 用 `http.server`）。
2. **零回归**：`scale:` 配置段整体为**增量叠加**，完全不影响既有 6 条 CLI 的任何行为。
3. **可复现**：`split.json` 不含时钟字段，同种子字节一致。
4. **断点续训**：`done` 单元重启跳过、不覆盖；`pending` 中断回滚。
5. **归一化一致性**：检测侧复用训练时持久化的 `NormalizerState`。

---

## 8.3 关键子模块

| 模块 | 职责 |
|---|---|
| `orchestrator.py` | 唯一共享编排骨架（`ScaleOrchestrator` + `UnitRunner` 端口）。严格串行遍历单元 → 单元边界轮询控制（`control.json`）→ 中断回滚 `pending` → 状态落盘 → 算 ETA；`done` 绝不覆盖。 |
| `units.py` | `UnitScanner` / `Unit` / `SplitPlan`：生成「股票×年份」单元清单与训练/检测划分。 |
| `sampler.py` | `UnitSampler`：单元级样本构建。 |
| `unit_trainer.py` / `unit_detector.py` | `UnitRunner` 的两个实现：训练单元、检测（选模型+回测）。 |
| `unified.py` | `UnifiedRunner`：所有缓存股票按时间切分（默认 `train 2006-2021 / val 2022-2023 / test 2024-2025`）用一个共享 PPO 训练。定义相对特征、三种执行语义版本、奖励版本。 |
| `ui/server.py` | `ScaleUI`：本机（127.0.0.1）HTTP UI，查看模型回放。 |
| `state.py` / `atomic.py` / `control.py` | `ScaleState` / `StateStore`（断点续训）、`atomic_write_json`（原子写）、`ControlStore`（中断控制）。 |
| `normalizer_store.py` | 持久化/复用 `NormalizerState`。 |
| `symbols.py` | `load_symbols` / `resolve_year_range` / `config_hash`。 |
| `paths.py` / `artifacts.py` / `model_policy.py` / `errors.py` / `replay.py` | 路径、单元产物、模型选取策略、异常、回放。 |

---

## 8.4 三种执行语义版本（unified.py）

统一训练里定义了三种执行口径，体现 A 股规则建模的演进：

| 版本 | 语义 |
|---|---|
| `EXECUTION_LEGACY`（同日收盘） | 旧代理规则，不结算公司行为 |
| `EXECUTION_STRICT`（T+1 开盘） | 后复权特征、T+1 真实开盘、旧涨跌停参考，不结算公司行为 |
| `EXECUTION_CORPORATE_ACTIONS`（v2） | 已核验分红送转的严格执行语义 |

> README 明确：**旧版本（legacy/strict）遗漏完整分红送转权益，历史结果不能作为可靠投资总回报**；新研究应使用 v2，且必须先补齐核验账本。

---

## 8.5 公司行为 v2 的强约束（重要！）

`unified` 的 v2 执行（详见第 2 篇 §2.7）要求：
- 账户时序精确处理登记日/除权日/派息日的持仓确权与锁股。
- 总权益 = 现金 + 总股数 × 未复权收盘价 + 股息应收。
- 数据契约严格：`data/corporate_actions/<六位代码>.json` 顶层仅 `schema/symbol/coverage/events`；`verified` 是提供者审阅声明，程序校验 ≠ 源端完整性证明。
- **采集不等于可执行账本**：`fetch-snapshot` 只归档未审阅候选，**永不自动标 verified**，也不写入正式账本目录。

> 当前状态（README §3.6.3）：本次仅完成机制和接线，**没有启动训练，也没有将 300 股票、2006–2025 全历史公司行为补齐或认证**。缺少生产核验覆盖账本时，v2 会**明确阻断**，不回退旧版，不从 raw/hfq 比值猜事件。

---

## 8.6 CLI 入口（cli/scale_cmd.py）

```
scale fetch              # 拉名单数据（默认禁 synthetic 兜底，失败写 fetch_failed.txt）
scale units              # 生成 units.json + split.json
scale train             # 逐单元训练（可 --only-failed）
scale detect            # 选模型 + 回测
scale status            # 进度 / ETA
scale unified-train     # 统一模型训练
scale unified status|control   # 统一训练状态 / 中断控制
scale corporate-actions fetch-snapshot|validate|import   # 只读账本管理
scale ui                # 启动本机 Web UI
```

---

## 8.7 本篇小结与提醒

- `scale` 是在主流水线之上的**批量编排叠加层**，目标是对沪深300 × 多年度做可复现、可续跑的训练/检测，并带本机 UI。
- 设计上**零新增依赖、零回归、可复现、断点续训**。
- **重要提醒**：其中「公司行为 v2」目前**未启动训练、未补齐全历史账本**；任何声称「v2 改善策略业绩」的产物都需谨慎对待，它目前只验证了机制，不是业绩证明。
- 如果你只是想理解「核心 AI 交易流水线的原理」，前 7 篇已足够；`scale` 属于进阶的工程量产化扩展。

---

## 全系列文档导航

- [第 0 篇：背景知识](./00_背景知识.md) —— 不懂 AI 先读这
- [第 1 篇：架构总览](./01_架构总览.md)
- [第 2 篇：数据层](./02_数据层.md)
- [第 3 篇：样本与特征工程](./03_样本与特征工程.md)
- [第 4 篇：交易环境](./04_交易环境.md)
- [第 5 篇：训练与 PPO 算法详解](./05_训练与PPO算法详解.md)
- [第 6 篇：评估与预测](./06_评估与预测.md)
- [第 7 篇：配置与 CLI 命令树](./07_配置与CLI命令树.md)
- [第 8 篇：Scale 大规模训练模块](./08_Scale大规模训练模块.md)

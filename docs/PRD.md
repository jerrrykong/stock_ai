# PRD：A 股强化学习交易工程（stock_ai）

> 版本：v1.0 ｜ 作者：许清楚（Product Manager）｜ 日期：2025
> 模式：简单 PRD（无竞品分析）

---

## 0. 项目信息

| 项 | 内容 |
|---|---|
| Language | 中文 |
| Programming Language | Python 3.13（CLI / Click 或 Typer）+ Stable-Baselines3 + Gymnasium |
| 运行环境 | Windows 10/11，PowerShell / Git Bash |
| 工作目录 | `E:\project\stock_ai` |
| 项目名 | `stock_ai` |
| 数据源 | 公开免费 API（AKShare / 腾讯行情 / 东方财富），本地缓存 parquet/sqlite |
| 原始需求 | 一个可独立运行的 A 股 RL 交易工程工具箱：读数据→建样本→训练→回测→预测，全流程拆成独立 CLI 子命令 |

**核心设计原则**（源自前序 RL 研究报告）：
1. RL 学习的是**交易决策**（MDP：状态→动作→奖励），不预测价格。
2. 数据必须**按时间顺序 walk-forward 划分**，严禁随机打乱。
3. 必须建模 A 股特有机制：T+1、涨跌停、停牌、手续费（印花税 0.05% 卖出）、滑点。
4. 优先 PPO（离散 买/卖/持有），SAC（连续仓位）作为进阶。
5. 每个模块**可独立执行**（各自独立 CLI 命令），互不阻塞。

---

## 1. 产品目标

**一句话**：把「数据获取 → 样本构建 → 强化学习训练 → 回测评估 → 未知数据预测」做成一条可复现、可分工、每个环节都能独立运行的 A 股量化交易工程流水线。

| # | 可衡量目标 | 衡量标准 |
|---|---|---|
| G1 | 数据可本地化 | `data fetch` 一条命令可拉取指定股票池日线并落盘缓存；二次刷新仅增量更新（<10s） |
| G2 | 样本可复现 | 给定同一数据快照与随机种子，`sample build` 产出的训练/检测样本完全一致（hash 相同） |
| G3 | 训练可独立 | `train` 在无网络环境下仅依赖本地样本即可完成，产出可复用模型文件 `.zip` |
| G4 | 评估可解释 | `backtest` 输出年化收益、最大回撤、夏普、胜率、换手率等指标 + 权益曲线图 |
| G5 | 预测可落地 | `predict` 对未知时间段输出每日 买/卖/持有 或 目标仓位 建议（CSV/JSON） |

---

## 2. 目标用户与使用场景

| 用户 | 场景 |
|---|---|
| Jerry（主用户，软件工程师 / 量化爱好者） | 单机 Windows 上从零搭 RL 交易实验，反复迭代特征/奖励/环境，验证策略 |
| 量化研究者 / 学生 | 复制工程结构，作为 RL 交易学习/研究脚手架 |
| 未来协作者 | 分模块接手（如只做数据、只做回测），靠 CLI 契约解耦协作 |

**典型使用路径**：
```
data fetch  →  sample build  →  train  →  backtest  →  predict
（可任意插队、重跑任一环节，只要上游产物存在）
```

---

## 3. 用户故事

| 编号 | 角色 | 我想要 | 以便于 | 优先级 |
|---|---|---|---|---|
| US-01 | 用户 | 用一条命令从公开 API 拉 A 股日线并缓存本地 | 断网也能重复实验，不用每次联网 | P0 |
| US-02 | 用户 | 独立刷新/增量更新日线数据 | 保持数据最新而不重拉全量 | P0 |
| US-03 | 用户 | 基于本地数据生成训练/检测样本 | 用统一、无泄漏的样本做训练与检验 | P0 |
| US-04 | 用户 | 选择算法（PPO/SAC）训练并保存模型 | 对比不同算法效果 | P0 |
| US-05 | 用户 | 加载模型做回测评估 | 量化策略收益与风险 | P0 |
| US-06 | 用户 | 用模型对未知数据集预测 | 得到实盘/未来交易建议 | P0 |
| US-07 | 用户 | 每个环节都能独立跑（独立 CLI） | 分步调试、并行开发、失败重试 | P0 |
| US-08 | 用户 | 切换股票池 / 时间范围 / 特征集 | 快速做多组对比实验 | P1 |

---

## 4. 需求池（P0 / P1 / P2）

### P0（Must have）

| 编号 | 需求名称 | 验收标准 |
|---|---|---|
| R-01 | 日线数据获取与缓存 | 给定股票代码列表与日期区间，从公开免费 API 拉取日线（OHLCV+复权），落盘 parquet/sqlite；重复运行命中缓存不重复请求 |
| R-02 | 数据增量刷新 | `refresh` 仅拉取缓存中缺失的最新日期，更新后数据连续无缺口（或明确标记缺口） |
| R-03 | 样本构建（walk-forward） | 按时间顺序划分，训练/验证/检测区间不重叠、不打乱；输出样本文件 + 元数据（区间、种子、特征列表） |
| R-04 | A 股交易环境 | Gymnasium 环境实现 T+1、涨跌停不可成交、停牌、手续费（含卖出印花税 0.05%）、滑点 |
| R-05 | 算法训练 | 支持 PPO（默认，离散 买/卖/持有），可切换 SAC；产出 `.zip` 模型 + 训练日志 + 曲线 |
| R-06 | 回测评估 | 加载模型在检测样本上回测，输出收益/回撤/夏普/胜率/换手等指标 + 权益曲线 |
| R-07 | 未知数据预测 | 加载模型对指定未知区间输出每日动作/仓位建议（CSV + JSON） |
| R-08 | 独立 CLI 子命令 | 6 大功能各对应至少一条独立命令，均可在不跑其他命令前提下直接调用 |
| R-09 | 配置与随机种子 | 全局 yaml/json 配置 + seed，保证可复现 |

### P1（Should have）

| 编号 | 需求名称 | 验收标准 |
|---|---|---|
| R-10 | 多股票池支持 | 从配置文件读入股票列表或指数成分，批量处理 |
| R-11 | 特征工程可配置 | 特征集（均线/动量/波动率/量比等）可增删，配置化 |
| R-12 | 奖励函数可配置 | 支持多种奖励（收益/夏普/回撤惩罚），配置切换 |
| R-13 | 训练过程可视化 | TensorBoard 日志 + 保存训练曲线图 |
| R-14 | 回测对比多模型 | 同一次回测可对比多个模型/基线（如买入持有） |

### P2（Nice to have）

| 编号 | 需求名称 | 验收标准 |
|---|---|---|
| R-15 | 超参搜索 | 简单网格/随机搜索接口 |
| R-16 | 数据质量报告 | 缺失/异常/停牌统计报告 |
| R-17 | 实验记录与追踪 | 每次实验参数、指标落库，便于回溯 |
| R-18 | 打包/环境自检 | `doctor` 命令检查依赖与网络连通性 |

---

## 5. 系统模块划分与 CLI 命令设计（重点章节）

**统一入口**：`python -m stock_ai <command> [options]`（亦提供 `stockai` 脚本别名）

每个模块 = 一个独立 CLI 子命令，输入输出通过「文件」解耦，命令之间无强依赖。

| 命令 | 主要参数 | 输入 | 输出产物 | 说明 |
|---|---|---|---|---|
| `data fetch` | `--symbols`（代码或配置文件）、`--start`、`--end`、`--source`（akshare/tencent/eastmoney）、`--cache-dir` | 公开 API（网络） | `data/raw/<symbol>.parquet`、`data/meta.json` | 拉取日线并建本地缓存（首次全量） |
| `data refresh` | `--symbols`、`--cache-dir` | 已有缓存 + API | 增量更新后的 `data/raw/*.parquet` | 仅拉缺失最新数据，独立可运行 |
| `data info` | `--cache-dir` | 缓存 | 控制台/JSON 数据概览 | 查看缓存股票、区间、行数、缺口 |
| `sample build` | `--symbols`、`--train-start`、`--train-end`、`--test-start`、`--test-end`、`--features`、`--window`、`--seed` | `data/raw/*.parquet` | `data/samples/train.npz`、`data/samples/test.npz`、`data/samples/meta.json` | walk-forward 生成训练/检测样本（无泄漏） |
| `train` | `--algo`（ppo/sac，默认 ppo）、`--samples`、`--timesteps`、`--seed`、`--reward`、`--out` | `data/samples/train.npz` | `models/<algo>_<ts>.zip`、`logs/<run>/`、训练曲线图 | 选择算法训练，产出模型；可独立运行 |
| `backtest` | `--model`、`--samples`（默认 test）、`--start`、`--end`、`--cash`、`--fee`、`--slippage` | 模型 `.zip` + `data/samples/test.npz` | `reports/backtest_<ts>.json`、`reports/equity_<ts>.png` | 加载模型回测评估，输出指标与权益曲线 |
| `predict` | `--model`、`--symbols`、`--start`、`--end`（未知区间）、`--out` | 模型 `.zip` + `data/raw`（或在线补充数据） | `reports/predict_<ts>.csv`、`reports/predict_<ts>.json` | 对未知数据集预测，输出每日动作/仓位建议 |
| `doctor`（P2） | — | 本地环境 | 自检报告 | 依赖/网络/目录自检 |

**模块划分**：

| 模块 | 目录 | 职责 | 对应命令 |
|---|---|---|---|
| Data Layer | `stock_ai/data/` | 拉取、缓存、刷新、数据源适配 | `data fetch/refresh/info` |
| Sample Layer | `stock_ai/samples/` | 特征工程、walk-forward 切分 | `sample build` |
| Env Layer | `stock_ai/env/` | Gymnasium 交易环境（A 股规则） | 被 train/backtest 复用 |
| Training | `stock_ai/train/` | SB3 算法封装、训练循环、日志 | `train` |
| Evaluation | `stock_ai/eval/` | 回测、指标计算、绘图 | `backtest` |
| Inference | `stock_ai/infer/` | 加载模型预测未知数据 | `predict` |
| Config/CLI | `stock_ai/cli/`、`configs/` | 命令注册、参数解析、配置加载 | 全部 |

---

## 6. 核心数据契约

| 类型 | 路径约定 | 格式 | 关键字段/约定 |
|---|---|---|---|
| 原始日线缓存 | `data/raw/<symbol>.parquet` | Parquet | `date, open, high, low, close, volume, amount, adj_factor`；date 升序、UTC+8 交易日 |
| 缓存元数据 | `data/meta.json` | JSON | 每只股票的最早/最新日期、行数、数据源、更新时间 |
| 训练样本 | `data/samples/train.npz` | NumPy npz | `obs, action_space, reward_cfg, 区间信息` |
| 检测样本 | `data/samples/test.npz` | npz | 同上，时间区间严格晚于训练区间 |
| 样本元数据 | `data/samples/meta.json` | JSON | `symbols, features, train_range, test_range, window, seed, data_hash` |
| 模型文件 | `models/<algo>_<timestamp>.zip` | SB3 zip | 含算法名、配置、seed 于 sidecar json |
| 训练日志 | `logs/<run>/` | TensorBoard + PNG | `tensorboard/`、`curve.png` |
| 回测报告 | `reports/backtest_<ts>.json` + `equity_<ts>.png` | JSON + PNG | 指标见下 |
| 预测结果 | `reports/predict_<ts>.csv` / `.json` | CSV/JSON | `date, symbol, action(买/卖/持有), target_position, signal` |

**回测指标字段**：`annual_return, total_return, max_drawdown, sharpe, win_rate, turnover, num_trades, benchmark_return(买入持有)`。

**命名约定**：时间戳统一 `YYYYMMDD_HHMMSS`；配置文件放 `configs/`（如 `configs/default.yaml`），命令行参数覆盖配置。

---

## 7. 非功能需求

| 类别 | 要求 |
|---|---|
| 性能 | 单股票 5 年日线 fetch < 30s（首次）；增量刷新 < 10s；缓存命中零网络请求 |
| 可复现 | 固定 seed（Python/NumPy/torch/SB3）+ 数据快照 hash，同参数必得同结果 |
| 依赖 | Python 3.13；akshare、pandas、numpy、pyarrow、stable-baselines3、gymnasium、torch、matplotlib、click/typer、pyyaml；`requirements.txt` + `pyproject.toml` 锁定 |
| Windows 兼容 | 路径用 `pathlib`，避免硬编码 `/`；控制台 UTF-8；无 Unix-only 依赖；parquet 避免文件名非法字符 |
| 容错 | API 失败自动重试（指数退避）+ 本地缓存兜底；数据缺口显式告警 |
| 可维护 | 模块解耦，命令间仅靠文件契约通信；关键路径有日志与单元测试 |
| 可移植 | 数据/模型/报告目录相对工作目录，便于整体拷贝 |

---

## 8. 待确认问题（含默认取值，不阻塞开发）

| # | 问题 | 默认取值建议 |
|---|---|---|
| Q1 | 初始股票池范围？（个股 / 指数成分 / 自定义列表） | 默认沪深 300 成分前若干只，可由 `configs/default.yaml` 自定义 |
| Q2 | 数据源首选哪个？是否需要多源冗余？ | 默认 AKShare（免费、稳定），腾讯/东财作为备选 fallback |
| Q3 | 动作空间形态？ | PPO 用离散 3 动作（买/卖/持有），SAC 用连续单维仓位 [0,1] |
| Q4 | 初始资金 / 手续费 / 滑点默认值？ | 初始 100 万；佣金 0.025%（双边）、印花税 0.05%（卖出）、滑点 0.1% |
| Q5 | 特征集默认包含哪些？ | 默认 OHLCV + MA(5/10/20) + 动量 + 波动率 + 量比，可配置扩展 |
| Q6 | 单只股票还是组合持仓？ | v1 默认单标的独立训练；组合持仓列为 P1/P2 |

---

*本 PRD 为简单版，聚焦「做什么 + 验收标准 + CLI 契约」，供架构师与技术团队直接落地。*

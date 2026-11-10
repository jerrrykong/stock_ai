# 第 7 篇：配置与 CLI 命令树（config + cli）

> 本篇把「怎么用配置调行为」和「有哪些命令可用」讲清楚。前面各篇都提到了配置对象和命令，这里集中成一张「控制面板说明书」。读完你就能不翻代码，直接用参数和 YAML 调出想要的实验。

---

## 7.1 配置对象总览（config/schema.py）

所有配置集中在 `AppConfig` 这个根对象，下辖 5 个子配置 + 1 个 scale 段。任意层只接收配置对象，不直接解析命令行——这是单一职责。

### AppConfig 顶层字段
| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `data` | DataConfig | — | 数据层配置 |
| `sample` | SampleConfig | — | 样本层配置 |
| `env` | EnvConfig | — | 交易环境配置 |
| `train` | TrainConfig | — | 训练配置 |
| `paths` | PathConfig | — | 运行时目录 |
| `scale` | ScaleConfig | — | 大规模训练（第 8 篇） |
| `random_seed` | int | 42 | 全局随机种子（可复现核心） |
| `log_level` | str | INFO | 日志级别 |

### DataConfig（数据）
| 字段 | 默认 | 说明 |
|---|---|---|
| `symbols` | [] | 股票代码列表 |
| `start` / `end` | 20180101 / 20241231 | 区间 |
| `source` | akshare | 数据源（akshare/tencent/eastmoney/sina/synthetic） |
| `cache_dir` | data/raw | 缓存目录 |
| `adjust` | none | 复权口径（qfq/hfq/none） |
| `retry_times` / `retry_backoff` / `timeout` | 3 / 1.5 / 15.0 | 重试与超时 |

### SampleConfig（样本）
| 字段 | 默认 | 说明 |
|---|---|---|
| `window` | 20 | 历史窗口步数 |
| `features` | 11 列 | 特征列名（见第 3 篇） |
| `train_start` / `train_end` | 20180101 / 20211231 | 训练区间 |
| `test_start` / `test_end` | 20220101 / 20241231 | 测试区间 |
| `embargo_days` | 5 | 训练/测试隔离交易日 |
| `normalize` | True | 是否标准化（统计量仅 train fit） |
| `val_ratio` | 0.15 | 验证集占比（从训练尾部切） |

### EnvConfig（交易环境，见第 4 篇）
| 字段 | 默认 | 说明 |
|---|---|---|
| `initial_cash` | 1,000,000 | 初始资金 |
| `commission_rate` | 0.00025 | 佣金（双边） |
| `stamp_tax_rate` | 0.0005 | 印花税（仅卖出） |
| `slippage_rate` | 0.001 | 滑点 |
| `min_commission` | 5.0 | 最低佣金 |
| `reward_type` | drawdown_penalty | 奖励类型（gross/drawdown_penalty/sharpe/excess） |
| `lambda_drawdown` / `lambda_turnover` / `lambda_leverage` | 0.1 / 0.001 / 0.01 | 奖励惩罚系数 |
| `max_position` | 1.0 | 最大仓位 |
| `lot_size` | 100 | 每手股数 |

### TrainConfig（训练，见第 5 篇）
| 字段 | 默认 | 说明 |
|---|---|---|
| `algo` | ppo | 算法（ppo/sac/dqn） |
| `timesteps` | 200000 | 训练步数 |
| `learning_rate` | 3e-4 | 学习率 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 折扣 / GAE 系数 |
| `clip_range` | 0.2 | 裁剪范围 |
| `n_steps` / `batch_size` / `n_epochs` | 2048 / 64 / 10 | 滚动步数 / 批大小 / 更新轮数 |
| `ent_coef` | 0.0 | 熵系数 |
| `backend` | auto | auto/sb3/numpy |
| `out_dir` | models | 模型输出目录 |

### PathConfig（目录）
`root / raw / samples / models / logs / reports`，均相对 `paths.root` 解析为绝对路径。

---

## 7.2 配置来源优先级

```
CLI 参数  >  YAML 配置  >  代码默认值
```

工作流程（`config/loader.py` 的 `ConfigLoader`）：
1. 若 YAML 存在 → `AppConfig.from_yaml`；否则用代码默认值（不抛异常，便于离线启动）。
2. `apply_env_overrides`：环境变量 `STOCKAI_SEED` / `STOCKAI_LOG_LEVEL` / `STOCKAI_DATA_SOURCE` / `STOCKAI_BACKEND`。
3. `resolve_paths`：相对路径以 `paths.root` 为基准转绝对。
4. `merge_cli_overrides`：把 CLI 的扁平 dotted key（如 `data.symbols`、`train.algo`）覆盖进配置对象。

> `from_dict` **忽略未知键**，所以 YAML 里多写字段不会报错；`merge_cli_overrides` 对值为 `None` 的键直接跳过，方便可选参数。

---

## 7.3 CLI 命令树（cli/）

统一入口 `python -m stock_ai <command>`（或安装后 `stockai`）。`cli/main.py` 是 `click.Group`，延迟导入各子命令避免循环依赖；捕获 `StockAIError` 后 `exit(1)`（K6 规范，错误可读不崩溃）。

| 命令 | 作用 | 映射到层 |
|---|---|---|
| `data fetch` | 拉取日线并缓存 | `data.loader` |
| `data fetch-hfq` | 拉取后复权（训练专用） | `data.hfq` |
| `data hfq-status` | 后复权拉取进度 | `data.hfq` |
| `data refresh` | 增量刷新 | `data.loader` |
| `data info` | 缓存概览 | `data.cache` |
| `sample build` | 构建 walk-forward 样本 | `samples.builder` |
| `train` | 训练模型 | `train.train_cli` |
| `backtest` | 回测评估 | `eval.backtester` |
| `predict` | 未知区间预测 | `infer.predictor` |
| `doctor` | 环境自检 | 全局 |
| `scale ...` | 大规模训练编排 | `scale.*`（第 8 篇） |

### 常用命令示例
```bash
# 数据：新浪源批量（无风控），不复权
python -m stock_ai data fetch --symbols 600519,000001 --start 20120101 --end 20241231 --source sina

# 数据：后复权（独立目录）
python -m stock_ai data fetch-hfq --symbols 600519 --start 20060101 --end 20251231 --throttle 3

# 样本
python -m stock_ai sample build --symbols 600519 --window 20

# 训练（强制纯 NumPy 后端，离线可用）
python -m stock_ai train --algo ppo --timesteps 50000 --backend numpy

# 回测
python -m stock_ai backtest --model models/ppo_xxx.zip

# 预测
python -m stock_ai predict --model models/ppo_xxx.zip --symbols 600519 --start 20250101

# 自检
python -m stock_ai doctor
```

### 独立模块入口（便于分工调试）
每个子模块还有 `-m` 直入口，例如 `python -m stock_ai.data.fetch_cli`、`python -m stock_ai.samples.build_cli`、`python -m stock_ai.train.train_cli`、`python -m stock_ai.eval.backtest_cli`、`python -m stock_ai.infer.predict_cli`。

---

## 7.4 doctor 自检（doctor_cmd.py）

`doctor` 一键检查：Python 版本、核心依赖（numpy/pandas/gymnasium 等）、可选依赖（akshare / torch / SB3）、训练后端可用性、目录可写、产物存在性。装环境出问题先跑它。

---

## 7.5 本篇小结

- 配置集中在 `AppConfig`，6 个子配置覆盖数据/样本/环境/训练/目录/大规模。
- 优先级：**CLI > YAML > 代码默认**；YAML 忽略未知键，安全容错。
- CLI 是 7 个主命令（+ scale），每条都映射到具体层，且模块有独立入口便于分工。
- 环境变量 `STOCKAI_*` 可做轻量覆盖。

下一篇（最后一篇）看大规模训练编排：[第 8 篇：Scale 大规模训练模块](./08_Scale大规模训练模块.md)

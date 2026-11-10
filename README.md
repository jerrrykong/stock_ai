# stock_ai —— A 股强化学习交易工程

> 把「数据获取 → 样本构建 → RL 训练 → 回测评估 → 未知数据预测」做成一条**可复现、可分工、每个环节独立可运行**的 A 股量化交易流水线。

![Python](https://img.shields.io/badge/python-3.11%2B-blue)

![License](https://img.shields.io/badge/license-MIT-green)

---

## 1. 它是什么

一个基于 **Gymnasium + PPO** 的 A 股日线交易实验工具箱。强化学习学的是**交易决策 MDP**（状态→动作→奖励），而不是预测价格：

- **状态**：最近 `window` 步的归一化技术特征窗口
- **动作**：`0=持有 / 1=买入 / 2=卖出`（离散）
- **奖励**：`ret − λ₁·drawdown − λ₂·turnover − λ₃·leverage`

**A 股规则内建**：T+1、涨跌停不可成交、停牌、佣金双边 0.025%、印花税卖出 0.05%、滑点 0.1%、100 股整手。

**无前视偏差**：walk-forward 时间切分 + embargo，归一化统计量**只在训练窗口 fit 并冻结**。

---

## 2. 快速开始

### 2.1 一键搭建环境（Windows PowerShell）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
```

脚本会：升级 pip → 装核心依赖 → 装 torch(CPU) → 装 SB3（可选）→ `pip install -e .` → 运行 `doctor` 自检。

> **重要**：若 torch / SB3 安装失败，**项目依然完整可用** —— 会自动走纯 NumPy PPO 兜底后端。

> ⚠️ **`No module named stock_ai` 的原因**：所有依赖和本包都装在上面的 **venv** 里，你终端里的 `python` 默认是系统 Python，看不到它。**新开终端后必须先激活 venv**（或用完整路径调用，见 2.3）。

### 2.2 手动安装（等效）

```powershell
$PY = "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

# 1) 核心依赖
& $PY -m pip install "numpy>=2.1,<2.6" "pandas>=2.2,<3.0" "scipy>=1.14" `
    "pyarrow>=17,<22" "click>=8.1" "pyyaml>=6.0" "matplotlib>=3.9" `
    "requests>=2.32" "tqdm>=4.66" "gymnasium==1.1.1"

# 2) torch CPU（固定索引）
& $PY -m pip install "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cpu

# 3) SB3（可选）
& $PY -m pip install "stable-baselines3>=2.6,<2.9"

# 4) 可编辑安装
& $PY -m pip install -e .

# 5) 自检
& $PY -m stock_ai doctor
```

### 2.3 激活环境（每次新开终端都要做）

```powershell
# 方式一：激活 venv（推荐，之后直接用 python / stockai）
& "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\Activate.ps1"
python -m stock_ai doctor          # 验证安装成功

# 方式二：不激活，直接用 venv 解释器的绝对路径
& "C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -m stock_ai doctor

# 方式三：用 entry point（激活后可用）
stockai doctor
```

### 2.4 统一入口 run.ps1（推荐日常使用）

```powershell
.\run.ps1                                  # 项目全景摘要: 原始数据/样本/模型/回测报告/预测输出/日志
.\run.ps1 data-fetch -Symbols 600519,600666 -Start 20120101 -End 20260929
.\run.ps1 data-refresh -Symbols 600519     # 增量刷新
.\run.ps1 sample-build
.\run.ps1 train -Algo ppo -Timesteps 200000
.\run.ps1 backtest -Model models\ppo_xxx.zip
.\run.ps1 predict -Model models\ppo_xxx.zip -Start 20250101
.\run.ps1 doctor
.\run.ps1 train -Help                      # 查看某命令完整参数
.\run.ps1 cli -CliArgs "data fetch --symbols 600519 --seed 7"   # 透传任意参数组合
```

> 首次运行若提示禁止脚本: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2.5 端到端跑通

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1
```

**离线可用**：若网络不通，用 `--source synthetic` 生成确定性合成数据，全链路依然可跑通：

```powershell
python -m stock_ai data fetch --symbols 600519 --source synthetic --start 20200101 --end 20241231
```

---

## 3. Linux / 远程部署（Bash 移植）

本项目同时提供 **Windows (PowerShell)** 与 **Linux / macOS (Bash)** 两套**功能对等**的入口：脚本一一对应，产物与文件契约完全一致。本节覆盖 **Linux 本地运行** 与 **远程主机部署**。

### 3.1 脚本对照表

| Windows (PowerShell)       | Linux / macOS (Bash)      | 作用                                                               |
| -------------------------- | ------------------------- | ---------------------------------------------------------------- |
| `run.ps1`                  | `run.sh`                  | 统一入口：`status` / `help` / 六大命令快捷方式 / `cli` 参数透传                   |
| `scripts/setup_env.ps1`    | `scripts/setup_env.sh`    | 一键搭建环境（建 venv → 装依赖 → `pip install -e .` → `doctor`）             |
| `scripts/run_pipeline.ps1` | `scripts/run_pipeline.sh` | 端到端 6 步流水线（fetch → sample → train → backtest → predict → status） |
| —（无）                       | `scripts/deploy.sh`       | 部署：同步本地项目到远程主机（打包 → scp → 远程解包；连接信息全部由参数/环境变量传入） |

> `run.sh` 与 `scripts/*.sh` 均为 **LF 行尾、无 BOM、`#!/usr/bin/env bash` + `set -euo pipefail`**。

### 3.2 命令对照（参数风格差异）

Windows 使用 **PascalCase 单横线**参数；Linux 使用 **GNU 风格 `--kebab-case`**：

| 语义   | Windows (PowerShell)                                                        | Linux (Bash)                                                                     |
| ---- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 拉取数据 | `.\run.ps1 data-fetch -Symbols 600519,000001 -Start 20120101 -End 20241231` | `bash run.sh data-fetch --symbols 600519,000001 --start 20120101 --end 20241231` |
| 训练   | `.\run.ps1 train -Algo ppo -Timesteps 200000 -Backend auto`                 | `bash run.sh train --algo ppo --timesteps 200000 --backend auto`                 |

```bash
# Linux 常用示例
bash run.sh                                  # 项目全景摘要 (status)
bash run.sh data-fetch --symbols 600519,600666 --start 20120101 --end 20241231
bash run.sh data-refresh --symbols 600519    # 增量刷新
bash run.sh sample-build
bash run.sh train --algo ppo --timesteps 200000 --backend auto
bash run.sh backtest --model models/ppo_xxx.zip
bash run.sh predict --model models/ppo_xxx.zip --start 20250101 --end 20250601
bash run.sh doctor
bash run.sh train --help                     # 查看某命令完整参数
bash run.sh cli "data fetch --symbols 600519 --seed 7"   # 透传任意参数组合
```

### 3.3 一键安装 + 跑通（Linux 本地）

```bash
# 1) 搭建环境（venv 建在项目内 ./.venv，无需 sudo）
bash scripts/setup_env.sh

# 2) 端到端 6 步流水线
bash scripts/run_pipeline.sh

# 3) 或按需单条运行
bash run.sh doctor
```

`run.sh` 会按 `./.venv/bin/python` → `python3` → `python` 顺序自动定位解释器；也可用环境变量 **`STOCKAI_PYTHON`** 显式指定。

### 3.4 远程主机一键部署

`deploy.sh` 的连接信息**全部通过命令行参数或环境变量传入**，脚本内不含任何主机 IP、SSH 账号：

```bash
# 本地：同步项目到远程（打包 -> scp -> 远程解包，自动排除 .venv/data/models/...）
bash scripts/deploy.sh --host <REMOTE_HOST> --user <REMOTE_USER> --dir '~/stock_ai'

# 远程：建环境 + 跑通
ssh <REMOTE_USER>@<REMOTE_HOST>
cd ~/stock_ai && bash scripts/setup_env.sh && bash scripts/run_pipeline.sh
```

可用参数（优先级：命令行参数 > 环境变量）：

| 参数      | 环境变量       | 默认值       | 说明                        |
| --------- | -------------- | ------------ | --------------------------- |
| `--host`  | `REMOTE_HOST`  | （无，必填） | 远程主机 IP 或域名           |
| `--user`  | `REMOTE_USER`  | （无，必填） | 远程 SSH 用户                |
| `--dir`   | `REMOTE_DIR`   | `~/stock_ai` | 远程项目目录（~ 在远程展开） |
| `--port`  | `SSH_PORT`     | `22`         | SSH 端口                    |

> 需要 SSH 密钥认证（脚本以 `BatchMode=yes` 运行，不会交互输密码）；主机密钥校验采用 `accept-new`：首次连接自动记录新主机密钥，密钥变化时仍会拒绝。
> `deploy.sh` 就地覆盖远程**代码**，保留远程运行时产物（打包时排除 `./data`、`./models`、`./reports`、`./logs`、`./.venv`、`./.git`）。**不要把真实主机 IP / 账号提交到版本库或文档。**

### 3.5 Linux 环境适配要点

- **venv 建在项目内 `./.venv`**：**无需 sudo**，属用户级安装，不污染系统环境。
- **pip 必须走镜像源**（该环境 `pypi.org` 不通）：  
  `-i ${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}`，可用环境变量 `PIP_INDEX_URL` 覆盖。
- **torch 走 CPU 索引**：`--index-url https://download.pytorch.org/whl/cpu`。
- **内存受限（约 1.9 G、无 swap）时 torch 装不上也不致命**：会自动回退到**纯 NumPy PPO 兜底后端**，全链路依旧可跑通。
- 脚本统一 **LF / 无 BOM / `set -euo pipefail`**；`run.sh` 会导出 `PYTHONIOENCODING=utf-8` / `LANG` / `OMP_NUM_THREADS`，避免远程 `C` locale 导致中文乱码。
- 可用 **`STOCKAI_PYTHON=/path/to/python`** 指定解释器；`TIMESTEPS` 环境变量可覆盖流水线训练步数。

### 3.6 数据源可用性（实测结论）

| 数据源         | 复权支持             | 单次返回                     | 风控                                                                | 备注         |
| ----------- | ---------------- | ------------------------ | ----------------------------------------------------------------- | ---------- |
| `tencent`   | qfq / hfq / none | 约 800 根（**自动分页**）        | ⚠️ **有**：连续快速拉约 150 只后返回 `501`（`waf.tencent.com` 风控页），约 30 分钟自动解封 | 项目原主源      |
| `sina`      | **仅 `none`**     | **全部历史**（单次请求，约 6000+ 根） | 未见                                                                | **批量拉取推荐** |
| `eastmoney` | none             | —                        | 部分网络下连接被断开                                                        | 不可依赖       |
| `akshare`   | —                | —                        | 其主接口依赖上述接口                                                        | 可能未安装      |

> ⚠️ **腾讯接口存在窗口速率风控**：批量拉长名单（如沪深300 × 20 年）时**必须**改用 `sina`  
> 或显式加 `--throttle`，否则会在拉到约 150 只后被封并触发合成的假数据兜底。

```bash
# 推荐：新浪源批量拉取（不复权、无风控、一次取全历史）
bash run.sh scale-fetch --source sina --throttle 0.5

# 用腾讯源拉取真实行情（不复权 none，默认）——注意风控，--throttle 建议 ≥2
bash run.sh data-fetch --symbols 600519 --start 20120101 --end 20241231 --source tencent --force

# 前复权（qfq）：对高分红股早年可能返回负价（见下方已知限制）
bash run.sh data-fetch --symbols 600519 --start 20120101 --end 20241231 --source tencent --adjust qfq --force

# 后复权（hfq）：按上市首日基准向后累积，无负价
bash run.sh data-fetch --symbols 600519 --start 20120101 --end 20241231 --source tencent --adjust hfq --force
```

- **新浪源与腾讯源数据一致**：同一标的同一天**价格逐日一致、`volume` 口径一致（均为「股」）**  
  （已实测 600519 @2020-01-02：`close` 均 1130.000，`volume` 14809916 vs 14809900），  
  故二者可互为 fallback。新浪源在 `--adjust qfq|hfq` 时会**明确报错并自动改用其它源**（不静默给错口径）。
- **`--allow-synthetic` 默认关闭**：真实源全部失败时**绝不写入合成假数据**（避免污染训练集），  
  失败标的记入 `<cache_dir>/fetch_failed.txt` 供下轮补拉；逐只容错，单只失败不中断整批。
- **`--adjust` 默认 `none`（不复权）**：即直接取原始成交价，天然规避下面提到的前复权早年负价问题。  
  可取值为 `qfq`（前复权）/ `hfq`（后复权）/ `none`（不复权，默认）；也可在 `configs/*.yaml` 的  
  `data.adjust` 中设定，CLI `--adjust` 优先级最高。
- **单次请求上限约 800 根**：长区间由代码**自动分页**，从区间末端向前逐页拉取并合并（无重复、严格升序）。
- ⚠️ **已知限制（外部数据特性，非代码缺陷）**：腾讯**前复权（qfq）** 对**高分红股**的**早年**会返回**负价**  
  （如 600519 在 2013 年 `close ≈ -189`）。此类行按 `validate_frame` 契约（要求价格 > 0）被剔除，  
  故**贵州茅台用 `qfq` 时仅保留 2016-07 起的 2060 行**；默认的 `none` 与 `hfq` 均**不受影响**，可拉满 2012 年起。
- ✅ **建议**：默认的 **`none`（不复权）** 即可拿到完整原始历史；如需复权序列，用 **后复权（`--adjust hfq`）**——  
  后复权按上市首日为基准向后累积，**不会出现负价**，同样可取回完整历史（600519 约 3000 行）。三种口径对照：
  | 复权方式       | 600519 起始 | 600519 行数 | 说明           |
  | ---------- | --------- | --------- | ------------ |
  | `none`（默认） | 2012 起    | ≈3000     | 原始价，无复权      |
  | `hfq`      | 2012 起    | ≈3000     | 后复权，无负价，完整历史 |
  | `qfq`      | 2016-07 起 | ≈2060     | 早年负价被剔除      |
- **复权口径由 `data fetch` 决定**：`sample build` / `train` / `backtest` / `predict` **只读缓存**，  
  不接受 `--adjust`；如需切换口径，请重新执行 `data fetch`（或 `data refresh`）并带上 `--adjust`。
- 离线场景可显式改用合成源：`--source synthetic`（合成源忽略 `--adjust`，仅保留参数一致性）。

### 3.6.1 训练专用后复权数据（`data fetch-hfq`）

`data/raw` 存放**不复权**真实价（供撮合），`data/raw_hfq` 是**独立的后复权缓存**（供特征/训练信号）。
两条路径完全隔离：`fetch-hfq` **绝不覆盖 `data/raw`**。

后复权价只用于特征，成交执行必须用真实未复权价 —— 严格撮合训练会**同时读两套价格**，
按交易日 inner join 对齐（`stock_ai.env.strict_env`）。

```bash
# 拉取后复权到 data/raw_hfq（源仅 akshare/eastmoney；sina 接口不支持复权）
python -m stock_ai data fetch-hfq --symbols 600519 --symbols 000001 \
    --start 20060101 --end 20251231 --throttle 3

# 进度随时可查（逐只容错，单只失败不中断整批）
python -m stock_ai data hfq-status
```

- **为什么是 hfq**：实测 `akshare`/`eastmoney` 的 `hfq` 返回**全历史**且**不会出现负价**；  
  `qfq` 只返回近约 10 年（≈2300 行）覆盖不足；`sina` 接口不支持复权；`tencent` 支持但有
  窗口 WAF（连续快速拉约 150 只后 501），需显式 `--source tencent` 并配合 `--throttle`。
- **强校验**：非正价格**直接失败**（不像 `validate_frame` 那样静默剔行）；同时输出
  日期覆盖、行数、首末价、最低价。
- **除权跳空修复统计**：输出 `|日收益| > 11%` 的次数（不复权侧 vs 后复权侧），
  后复权侧通常显著减少，说明除权跳空已被修复。
- 节流参数：`--throttle`（每只间隔秒）、`--batch`/`--rest`（批间休息）、
  `--max-fail-streak`/`--cooldown`（连续失败判风控后冷却）；失败清单落 `data/raw_hfq/fetch_failed.txt`。

### 3.6.2 严格撮合旧版本（`--execution strict`）

`scale unified train --execution strict` 保留旧严格版本以复现历史实验；CLI 默认仍为 `legacy`，不会静默改变旧任务。**这两个旧版本均遗漏完整分红送转权益，历史结果不能作为可靠投资总回报。** 新研究应使用下一节公司行为 v2，并先补齐核验账本。

| 参数 | 冻结执行版本 | 执行语义 |
| --- | --- | --- |
| `legacy` | `legacy-same-day-close-v1` | 同日收盘模拟，旧代理规则，不结算公司行为 |
| `strict` | `strict-t1-open-v1` | 后复权特征、T+1 真实开盘，旧涨跌停参考，不结算公司行为 |
| `strict-ca` | `strict-t1-open-ca-v2` | 后复权特征、T+1 真实开盘、核验公司行为结算及除权参考价；缺证拒绝 |

```bash
python -m stock_ai scale unified train --execution strict \
    --feature-dir data/raw_hfq --timesteps 200000 --run-id my_run
```

- 板块规则：主板 10%、创业板（300/301，2020-08-24 注册制起）/ 科创板（688/689）20%、北交所 30%；
  主板 ST 按 5% 近似（**ST 名单未联网获取**，默认按非 ST 处理，已在 manifest `warning` 注明）。
- 费用按真实未复权成交金额计算：佣金双边 0.025%（最低 5 元）、印花税卖出 0.05%、滑点 0.1%。
- `manifest.json` 记录 `execution_version` / `feature_dir` / `execution_dir` / `feature_adjust`；
  `metrics.json` 的 val/test 额外给出 `friction` 交易磨损汇总
  （总换手、交易次数、成本拖累 bps、平均单笔磨损、与买入持有基准的对比）。

### 3.6.3 公司行为 v2：先核验账本，再训练 / 回放

本次仅完成机制和接线，**没有启动训练，也没有将 300 股票、2006–2025 全历史公司行为补齐或认证**。当前缺少生产核验覆盖账本，v2 会明确阻断，不回退旧版，不从 raw/hfq 比值猜事件。实现和限制见 [架构增量说明](docs/ARCHITECTURE_SCALE_TRAINING.md#公司行为-v2-增量)；验收记录见 [QA 报告增量](docs/QA_REPORT_SCALE_TRAINING.md#公司行为-v2-增量验收)。

- **账户时序**：登记日交易后的收盘持仓确权；除权开盘前确认税前股息应收、送转股及锁股；派息仅应收转现金、上市仅解锁，不重复确认收益。每段首日空仓，首日及之前登记不反推持仓资格。非交易日派息 / 上市在下一可用开盘前补记，并保留有效日期。
- **总权益** = 现金 + 总股数 × 未复权收盘价 + 股息应收；可卖股数 = 总股数 − 待上市锁股。成本含买入费用，送转摊薄成本；现金分红单列，不重复冲减成本。未上市股按同股 raw 收盘估值，不代表可用现金。
- **税与机制边界**：口径为 `gross_before_personal_dividend_tax`，未实现个人 FIFO 差别化股息税。碎股、配股、特殊方案、未知行为、登记 / 除权缺真实日线、未上市股份嵌套登记资格均拒绝。ST 历史身份、新股无涨跌幅限制及真实流动性并未完整建模；费用仍按配置，不是历史逐日税费表。
- **数据契约**：`data/corporate_actions/<六位代码>.json` 顶层仅 `schema / symbol / coverage / events`；schema 为 `corporate-actions-v1`。覆盖必须显式声明范围、来源、审阅时间和说明。`verified` 是提供者审阅声明，程序校验不等于源端完整性证明；审阅确认无事件时才可用空 events。现金 / 送股 / 转增三腿及 `reference_method=standard_cash_bonus` 必须显式；股份到账日不可替代上市日。
- **采集不等于可执行账本**：`fetch-snapshot` 归档原字节、SHA256 和 `unreviewed` 候选，永不自动标 verified。接口 count=total 也不能证明历史完整；须独立核对官方实施公告与其他资本行为，缺失字段保持未知。

```bash
# 仅归档未审阅来源，不生成生产覆盖
python -m stock_ai scale corporate-actions fetch-snapshot --symbol 300750
# 对已完成独立审阅的本地账本做只读检查；以下路径仅为示意
python -m stock_ai scale corporate-actions validate reviewed/300750.json --start 20230101 --end 20231231
# 显式导入，目标存在时拒绝覆盖；不要将测试夹具或未审阅快照导入
python -m stock_ai scale corporate-actions import reviewed/300750.json --start 20230101 --end 20231231
```

**统一训练接线（此处不执行训练）**：新 run 使用 `--execution strict-ca --feature-dir data/raw_hfq --corporate-actions-dir data/corporate_actions`；仍为一股票一年 episode，行情观测 `unified-relative-v1` 和三动作不变。v2 使用相同约束、独立账户的恒买基准；excess 奖励版本为 `excess-ca-constant-buy-v2`。新 UI 的执行口径默认 v2 并显式发送，CLI / 后端缺省仍保留 legacy 兼容。

**冻结与续训**：v2 manifest 冻结 hfq、raw、账本的完整文件 SHA256。v2 训练准备对每个输入仅读取一次快照，解析、日历校验、episode 和指纹共用该份字节；回放也用同一字节解析与审计，避免并发替换造成数据 / 指纹错配（不代表多文件跨源原子快照）。即使只追加区间外行情或账本空白也改变哈希，不能冒充原实验。原生 v2 的 sources 必须是非空、代码合法且无重复、三类指纹格式齐全的来源清单；缺整张清单不能冒充新股票。已记录训练股票在 v2 回放时必须三者匹配；确实未列入清单的新股票只披露当前指纹，不声称冻结一致。续训沿用冻结执行版本，不允许将旧 run 改为 v2；源变动拒绝，且不改旧模型、manifest、normalizer（准备失败可更新该 run 的状态 / metrics）。

**UI 查看**：手动重启现有 UI 服务使其加载新后端后，进入“模型回放”，选模型、股票和日期。原生 v2 展示 22 列逐日表、应收 / 锁股 / 可卖和事件详情，CSV / JSON 保留完整事件、来源、核验覆盖与指纹。旧 strict v1 默认仍按旧语义；仅显式勾选“公司行为修正研究回放”才用 v2 执行，必须同样满足账本覆盖。它只是旧 policy 的研究执行覆写，**不是已修复重训模型**；旧 raw 训练哈希缺失标 `not_recorded`，不补造历史证据。切换筛选而不重跑不会改变已完成结果的导出。

本次保留既有 8766 / 8767 服务，**未自动重启或部署**。真实案例与验收产物在 `outputs/corporate_action_*`，仅验证机制，不是策略业绩改善证明。

### 3.7 常见问题（FAQ）

- **`No module named stock_ai`**：说明当前解释器不对（venv 未建好 / 未指定 / 系统 Python 看不到 venv）。  
  先运行 `bash scripts/setup_env.sh`，或用 `STOCKAI_PYTHON=/path/to/.venv/bin/python bash run.sh ...` 指定。  
  `run.sh` 已内置该检查，会直接给出引导，而非零散的 `ModuleNotFoundError`。
- **非法参数会明确报错**：`--source` / `--adjust` / `--backend` / `--reward` 传入不支持的值时，  
  会打印**可选值白名单**并以**非零退出码**结束，**不再静默接受**或按默认值继续。
- **切换复权口径会告警并重拉**：`meta.json` 会记录每个标的的复权口径（`adjust`）。  
  当 `data fetch` / `data refresh` 的 `--adjust` 与既有缓存不一致时，**不会命中旧口径缓存**，  
  而是**全量重拉**（`data refresh` 亦**不做跨口径增量拼接**）；其中 `data refresh` 的重拉起点取  
  `min(缓存现有起日, config.start)`，即**保留用户既有区间、不主动收缩**（避免静默丢弃历史数据），  
  避免出现 `qfq 老数据 + hfq 新数据` 的口径混用。

---

## 4. 命令手册

统一入口：`python -m stock_ai <command>`（或安装后的 `stockai <command>`）。  
每个模块另有独立入口 `python -m stock_ai.<module> <args>`（便于分工与调试）。

| 命令             | 作用                 | 输入 → 输出                                                                        |
| -------------- | ------------------ | ------------------------------------------------------------------------------ |
| `data fetch`   | 拉取日线并缓存            | API → `data/raw/*.parquet` + `data/meta.json`                                  |
| `data fetch-hfq` | 拉取后复权（训练专用）    | API → `data/raw_hfq/*.parquet`（不复权缓存不受影响）                                     |
| `data hfq-status` | 后复权拉取进度          | `fetch_progress.json` → 控制台                                                     |
| `data refresh` | 增量刷新               | 缓存 → 更新后的 parquet                                                              |
| `data info`    | 缓存概览               | 缓存 → 控制台                                                                       |
| `sample build` | 构建 walk-forward 样本 | parquet → `data/samples/{train,val,test}.npz` + `meta.json`                    |
| `train`        | 训练模型               | `train.npz` → `models/<algo>_<ts>.zip` + sidecar json + `logs/<run>/curve.png` |
| `backtest`     | 回测评估               | 模型 + `test.npz` → `reports/backtest_<ts>.json` + `equity_<ts>.png`             |
| `predict`      | 未知区间预测             | 模型 + `meta.json` + `data/raw` → `reports/predict_<ts>.{csv,json}`              |
| `doctor`       | 环境自检               | — → 自检报告                                                                       |

### 常用参数速查

```bash
# 数据
python -m stock_ai data fetch --symbols 600519 000001 --start 20200101 --end 20241231
python -m stock_ai data fetch --symbols 600519 --adjust hfq            # 后复权（可规避 qfq 早年负价）
python -m stock_ai data refresh --symbols 600519 --adjust hfq         # 口径变更时自动全量重拉
python -m stock_ai data info

# 训练专用后复权（独立目录 data/raw_hfq，真实成交仍用 data/raw）
python -m stock_ai data fetch-hfq --symbols 600519 --symbols 000001 \
    --start 20060101 --end 20251231 --throttle 3
python -m stock_ai data hfq-status

# 旧严格语义复现（遗漏公司行为）；新v2须先核验账本，见3.6.3
python -m stock_ai scale unified train --execution strict \
    --feature-dir data/raw_hfq --timesteps 200000 --run-id my_run

# 样本
python -m stock_ai sample build --symbols 600519 --window 20 \
    --train-start 20200101 --train-end 20221231 \
    --test-start 20230101 --test-end 20241231

# 训练（后端自动选择；--backend numpy 可强制兜底后端）
python -m stock_ai train --algo ppo --timesteps 50000 --seed 42

# 回测
python -m stock_ai backtest --model models/ppo_20250101_120000.zip

# 预测未知区间
python -m stock_ai predict --model models/ppo_20250101_120000.zip \
    --symbols 600519 --start 20250101 --end 20250301
```


### 独立入口示例

```bash
python -m stock_ai.data.fetch_cli fetch --symbols 600519 --start 20230101 --end 20231231
python -m stock_ai.samples.build_cli --symbols 600519 --window 20
python -m stock_ai.train.train_cli --algo ppo --timesteps 5000
python -m stock_ai.eval.backtest_cli --model models/ppo_x.zip
python -m stock_ai.infer.predict_cli --model models/ppo_x.zip --symbols 600519 --start 20250101 --end 20250301
```

---

## 5. 项目结构

```
stock_ai/
├── run.ps1 / run.sh         # 统一入口（Windows PowerShell / Linux Bash）
├── configs/                 # default.yaml / fast_dev.yaml / symbols_hs300.yaml
├── scripts/                 # setup_env.{ps1,sh} / run_pipeline.{ps1,sh} / deploy.sh
├── src/stock_ai/
│   ├── config/              # AppConfig dataclass + ConfigLoader
│   ├── utils/               # errors / logging / seed / io / hashing / retry / dates
│   ├── data/                # 多源 fallback + parquet 缓存 + 增量刷新
│   ├── samples/             # 特征工程 + 防泄漏归一化 + walk-forward
│   ├── env/                 # TradingEnv + A 股规则纯函数 + 奖励
│   ├── train/               # SB3 封装 + 纯 NumPy PPO 兜底 + 后端工厂
│   ├── eval/                # 回测 + 指标 + 绘图
│   ├── infer/               # 未知区间预测（复用冻结归一化状态）
│   └── cli/                 # click 子命令树
└── tests/                   # 单元测试
```

**分层依赖**（下层绝不 import 上层）：

```
cli → train/eval/infer → env/samples → data → config/utils
```

**命令间通信 = 文件契约**：`parquet / npz / zip / json`，绝不通过内存对象传递。

---

## 6. 后端选择（主 / 兜底）

| 优先级 | 后端 | 触发条件 |
|---|---|---|
| L0 | **SB3 PPO**（Primary） | `stable_baselines3` + `torch` 均可导入 |
| L2 | **纯 NumPy PPO**（兜底） | SB3/torch 不可用，或 `--backend numpy` |
| L3 | 规则基线 | `backtest` 的 buy&hold 始终可用 |

两者共享统一接口 `learn / predict / save / load`，以及统一的 `models/*.zip` 文件契约：

- **SB3**：原生 `model.save()` 的 zip
- **NumPy**：`weights.npz` + `hparams.json` + `backend.json` 打包的 zip

`backtest` / `predict` 通过读取 zip 内的 `backend.json` 自动识别后端并正确加载。

---

## 7. 复现说明

1. **固定种子**：全局 `--seed`（默认 42）对齐 Python / NumPy / torch。
2. **数据快照哈希**：`sample build` 计算所有 raw parquet 的 sha256 → 写入 `samples/meta.json`。
3. **复现链**：`meta.data_hash` → 写入模型 sidecar json → 回填 `backtest` 报告。
   同数据 + 同参数 → 同 `data_hash` → 同结果。

验证：

```bash
# 两次构建应得到相同 data_hash
python -m stock_ai sample build --symbols 600519
python -m stock_ai sample build --symbols 600519   # data_hash 完全一致
```

---

## 8. 测试

```bash
python -m pytest -q
```

覆盖：A 股规则纯函数（T+1/涨跌停/停牌/成本）、奖励数值、特征无泄漏、
walk-forward 零重叠 + embargo、`check_env` 合规、指标对照手算、哈希稳定性、
6 条命令 `--help` 与端到端冒烟。

---

## 9. 配置

`configs/default.yaml` 为全量配置；CLI 参数优先级高于 YAML：

```
CLI 参数  >  YAML 配置  >  代码默认值
```

关键节：`data` / `sample` / `env` / `train` / `paths`。详见 `configs/default.yaml`。

---

## 10. 产物目录

```
data/raw/<symbol>.parquet      # 原始日线缓存
data/samples/*.npz + meta.json # 样本与元数据
models/<algo>_<ts>.zip + .json # 模型 + sidecar
logs/<run>/                    # train.log + curve.png
reports/                       # backtest_*.json / equity_*.png / predict_*.{csv,json}
```

---

## 11. 许可

MIT

# QA 验收报告：大规模训练编排 + 检测编排 + 本机 UI（增量）

> 验证人：严过关（QA 工程师）｜ 日期：2026-10-10
> 结论：**R1 有条件通过（1×P1 + 2×P2）→ 工程师已修复 → R2 回归全绿，判定通过（路由 NoOne）**（详见 §7）
> 上游文档：`docs/PRD_SCALE_TRAINING.md`（§7 AC1–AC12）＋ `docs/ARCHITECTURE_SCALE_TRAINING.md`
> 验证对象：`src/stock_ai/scale/**`、`src/stock_ai/cli/scale_cmd.py`、`configs/symbols_hs300_full.yaml`、`tests/test_scale_*.py`

---

## 0. 验证方法与独立性声明

- **独立于工程师自验**：未采信 `tests/test_scale_*.py` 的结论；另写独立脚本（`.qa_edward/qa_*.py`、`.qa_real/`、`.qa_full/`）以**真实子进程**驱动 CLI 复现/证伪。
- **强制突变测试**：主动删除/篡改产物与状态，检验系统是否被骗判为 `done`。
- 解释器：`C:\Users\jerrykong\.workbuddy\binaries\python\envs\default\Scripts\python.exe`（Python 3.13.14，含 sb3 2.8.0 / akshare / torch）。
- 环境：`PYTHONPATH=src`、`PYTHONIOENCODING=utf-8`；系统日期 **2026-10-10**。
- 隔离：所有落盘均在 `.qa_edward` / `.qa_real` / `.qa_full` 独立工作区，未污染仓库默认 `data/`。
- **限制**：本机仓库**无 `.git`**（`git status` → `fatal: not a git repository`），故「改动前字节基线」无法用 git 取得；改用「既有测试全绿 + 结构性只增不改 + 6 条 CLI `--help` 全 exit 0」三重佐证（见 §2 AC11）。

---

## 1. 逐条验收（AC1–AC12）

| # | 场景 | 判定 | 关键证据 |
|---|---|---|---|
| AC1 | 数据 | ✅ PASS | `scale fetch --source tencent`（真实源）拉 600519/000001/600036 各 **726 行**（20220104~20241231），`meta.adjust=none`；`--source synthetic` 拉 **300 只 × 20060101~20251231**。600519 收盘 ¥2051.23→¥1524.00（真实茅台量级）。 |
| AC2 | 单元 | ✅ PASS | 300 代码 × 20 年 = **6000 单元**，`available=6000`（∈[5000,6500]），`train=4800 detect=1200`；`units.json` 记录 `status` 与 `steps` 可区分 available/insufficient/missing。 |
| AC3 | 可复现 | ✅ PASS | `scale units --seed 42` 连跑两次 `split.json` sha256 **完全一致**（`58b4d361…`）；`--seed 7` → `ee680f27…`（种子真生效）；`train∪detect==available`、无重叠、比值 8:2（12:3）。 |
| AC4 | 训练续跑 | ✅ PASS | 真实子进程跑到 ~30% **硬杀**（`TerminateProcess`）：杀后 `done=3 running=1`；重启后 `done=12`，`counts.done == total-failed-skipped`。 |
| AC5 | 无覆盖 | ✅ PASS | 已完成单元 `.zip` 的 mtime / sidecar `created_at` / zip sha256 **三者全部未变**（跳过且不覆盖）。 |
| AC6 | 失败隔离 | ✅ PASS | 注入 1 个必失败单元 → `failed=1` + `error` 记录，其余 11 单元 `done`，进程 exit 0；`--only-failed` 精准重跑该单元 → done。 |
| AC7 | 检测策略 | ✅ PASS | `nearest_past`：`000001_2020/2021→000001_2022`（无 past 回退 max_year）、`600036_2021→600036_2020`（≤检测年最大），三者 `assigned_model` 全部符合策略；无匹配 → `skipped`+reason。 |
| AC8 | UI 控制 | ✅ PASS | `POST /api/control {pause}` 写 `control.json`；编排器**在单元边界**进入 `paused`，其后**不再出现新 `running`**、done 冻结；`resume` 恢复推进；`stop` 收敛为 `stopped` 且进程 exit 0（隔离复现，见 §2-8）。 |
| AC9 | UI 观察 | ✅ PASS | `/`(13.9KB HTML, 5 Tab)、`/api/status`、`/api/units`、`/api/data`、`/api/logs`、`/api/control/status` 均 200 且 JSON/HTML 合法；未知路由 404；非法动作 400。 |
| AC10 | 无自启 | ✅ PASS | `Get-ScheduledTask` 匹配 stock/scale = **0**；`Get-Service` 匹配 = **0**；代码无 `win32service/nssm/schtasks/daemonize`；UI 仅 `Popen` 一次、退出后端口释放且不复活。 |
| AC11 | 零回归 | ✅ PASS | 既有（排除 scale）**221 passed / 1 skipped**；全量 **270 passed / 1 skipped**；6 条 CLI + `scale` `--help` 全部 exit 0、结构完好（只增不改）。 |
| AC12 | Windows | ✅ PASS | 全程 Windows + Git Bash 真跑；中文/UTF-8 无乱码；`os.replace` 原子写、`pathlib`、无 fork 依赖。 |

> AC12 附注：唯一 Windows 原生隐患即下方 **P1**（并发读下 `os.replace` 共享冲突），已单列。

---

## 2. 必验清单（12 项，按风险降序）

### 1) 默认后端路径（**工程师自验未覆盖**）✅ PASS
绕开其 `--backend numpy`，不加 `--backend`（默认 `auto`）跑真实训练：
```
INFO stock_ai.train.factory | 后端: SB3 (ppo) [auto]
INFO stock_ai.train.sb3_trainer | SB3 模型已保存: .../600519_2020.zip
[1/1] 600519_2020 -> done (2.53s)
```
产物齐全：`.zip` / `.json` / `.norm.json` / `.curve.png` **4/4 存在**。默认路径可用且稳定。

### 2) 断点续训幂等性（最高风险）✅ PASS
- 硬杀遗留 `running=000002_2022` → 重启后**无残留 running**，该单元被**重训为 done**（zip 重新生成）。
- 已完成 3 单元 zip `mtime`+`created_at`+`sha256` **逐字节未变**。
- 最终 `counts.done=12 == total-failed-skipped`。
- **突变（核心）**：手动删除 `600519_2021.zip` → 重启后该单元**被判定未完成并重训**（sha `1690…`→`ba99…`，`created_at` 更新）。**未出现"仍判 done 不校验产物"** → 无 P0。

### 3) 检测绝不重新 fit（正确性关键）✅ PASS
- 代码路径：`UnitDetector` → `sampler.builder.build_split(..., state=load(assigned))` → `make_obs_matrix` → `FeatureEngineer.transform`（**无 fit**）。
- **运行时钩子证据**：对 `FeatureEngineer.fit/transform` 装计数器，跑真实检测单元：
  `FeatureEngineer.fit 调用次数 = 0；transform = 1`。
- **突变**：删除 assigned model 的 `.norm.json` → 该检测单元 `status=skipped`、`reason="normalizer_missing"`、**未产出报告**（未静默 refit / 未静默通过）；同批其它单元（norm 仍在）正常 done。

### 4) 可复现性（AC3）✅ PASS
seed42 两次 sha256 相同、seed7 不同；`train∪detect==available`、无交集、8:2（整数误差 0 ≤1）；`split.json` 无 `created_at` 等时钟字段。

### 5) 年份公式（U1）✅ PASS
`resolve_year_range(None)` 直接调用（系统时钟 2026-10-10）→ **`(2006, 2025)`**（20 个完整自然年）；`date(2027,6,1)`→`(2007,2026)`（真动态）；`[2010]` 单元素 → `ScaleError`。
**硬编码检索**：`src/stock_ai/scale/**`、`config/`、`scale_cmd.py` 中出现的 `2006/2025` **仅在 docstring 与报错文案**，逻辑无硬编码年份。

### 6) 零回归 ✅ PASS（含限制说明）
- 既有（排除 6 个 `test_scale_*.py`）：**221 passed / 1 skipped**（进度点计数：72+72+72+5 dots + 1 `s`）。
- 全量：**270 passed / 1 skipped**（72×3+54 dots + 1 `s`）——与工程师报 270/1 一致，新增 49 项只增不减。
- 6 条既有 CLI（`data/sample/train/backtest/predict/doctor`）+ `scale` `--help` 全部 exit 0、选项/描述完好。
- `cli/main.py` 改动为纯追加（新增 `scale` 组）。
- ⚠️ **限制**：无 `.git` 无法做「改动前逐字节」基线；工程师 `test_existing_six_clis_help_contract_unchanged` 仅校验关键子串而非逐字节。**建议**（非阻断）：仓库纳入 VCS 或固化 help 快照，以支撑 AC11 的严格字节级断言。

### 7) 原子写（AC4 相关）⚠️ 原子性 PASS / **并发健壮性 P1 FAIL**
- **原子性成立**：6 写者×300 次 + 高频读压测，`JSONDecodeError = 0`（从不出现半截 JSON）；无残留 `.tmp`；`utils/io.py` **未引入** atomic/os.replace（全仓 `os.replace` 仅在 `scale/atomic.py:37`）。
- **P1 缺陷（详见 §3）**：Windows 下 `os.replace` 无重试，**并发读**可致 `PermissionError`。量化（1 写者，N 读者）：

  | 读者频率 | 写者侧失败 | 读者侧失败 |
  |---|---|---|
  | 紧凑循环（2 读线程） | 2115 / 3000 | 2279 / 3000 |
  | 0.2s（≈UI 快刷） | 0 / 3000 | 4 / 3000 |
  | 2.0s（≈UI 正常轮询） | **1 / 3000** | 0 / 3000 |

  端到端复现：训练中并发读 `state/train.json` → `scale train` **exit 1**，stderr：`[未预期错误] PermissionError: [WinError 5] 拒绝访问: '…train.json.<pid>.tmp' -> '…train.json'`。

### 8) UI（AC8/AC9/AC10）✅ PASS
- **监听**：`netstat` → `TCP 127.0.0.1:8801 LISTENING`（**仅回环**，无 `0.0.0.0`/`[::]`）。
- 端点全部 200/合法；`POST /api/control {pause}` 落 `control.json`；非法动作 400；未知路由 404。
- **控制链路（隔离复现）**：pause→`paused`（无新 running、done 冻结）→resume→推进→stop→`stopped`、进程 exit 0。
- **无自启**：计划任务/服务匹配数均 0；UI 退出端口释放、不复活。
- ⚠️ 关联 P1：在最激进的 UI 轮询下，UI 拉起的 train 子进程曾因 P1 中途死亡（state 冻结在 `running`、`control.json=stop` 未被消费）——即 P1 在**真实 UI 场景**下的表现。

### 9) 真实数据获取路径（AC1）✅ PASS
- 本机 `tencent` 真源：600519/000001/600036，各 726 行、2022-01-04~2024-12-31、`adjust=none`、价格量与真实一致。基于真实数据跑通 `units→train→detect`（9 单元 / train 7 / detect 2）。
- `akshare` 源：本机 hist 端点被代理阻断（`ProxyError`）；`--source akshare --force` 表现**符合 D-3**：逐只记录失败原因并**自动回退 tencent**，整批不中断（2/2 成功）。**未**真拉 300×20。

### 10) 名单真实性（AC1/D-1）✅ PASS
- `symbols_hs300_full.yaml`：**300** 只、**全部合法 6 位数字**、**无重复**、前缀分布合理（600/601/603/605/688/000/001/002/003/300/301/302）。
- **与 akshare 实时沪深300 成分（`index_stock_cons_csindex`，2026-10-09）交叉核对：300/300 命中，覆盖率 100%**，0 只不在成分内；16 只知名成分（茅台/平安/招行/美的/长江电力/平安保险/五粮液/比亚迪/宁德时代/中芯国际/恒瑞/格力/工行/中信证券/京东方/海康）全部在列。
- 结论：名单实为**当前真实全量沪深300 成分股**（优于"近似"要求），无凭空编造。

### 11) 边界与异常 ✅ PASS（除 1 项 P2）
- `Unit.parse`：`600519_2015`✅、`600519.SH_2015`✅（点后缀）、`600519_SH_2015`✅；`2015`/`600519_`/`_2015`/`600519_abc` 均抛 `ValueError`。
- `mode∈{predict,both}` → **明确 `skipped` + `reason=mode_not_supported:*`**（不静默）。
- `workers>1` → **明确报错**（`ScaleError`，exit 1，文案含"workers>1 暂不支持"）。
- `--seed abc` → click 用法错误非崩溃；`years=[2010]`/非数字 → `ScaleError`；`years=[]` 视作 None。
- 失败隔离见 AC6。
- ⚠️ **P2**：非法 `model_policy`（如 `bogus`）**静默退化为 `nearest_past`**（`ModelSelector.select('bogus',…)` 返回 `600519_2018`），未报错。

### 12) `scale status` ✅ PASS
`总数/完成/失败/跳过/剩余/当前单元/ETA` 与状态文件实数**逐项一致**（total=12、done=12、remain=0…）。

---

## 3. 缺陷清单（分级）

| 级别 | 模块/文件 | 现象 | 复现 | 影响 / 建议 |
|---|---|---|---|---|
| **P1** | `src/stock_ai/scale/atomic.py::_fsync_and_replace`（L28-46，`os.replace` L37）；`src/stock_ai/scale/state.py::StateStore.save`（L250-256） | Windows 下 `os.replace` 未做重试；当 UI/CLI 正读取状态文件时替换失败，抛 `PermissionError (WinError 5/32)`，**未被捕获**，经 `orchestrator._loop→run` 上抛，`main()` 记"[未预期错误]"，**整批 `scale train`/`detect` 中止(exit 1)**。 | 训练中并发高频读 `state/train.json`：`scale train` exit 1。量化：UI 正常轮询(2s) 约 **1/3000** 次写失败；`state.json` mtime 停在 `running`、`control.json` 的 `stop` 长期不被消费。 | 与"长期任务稳定可续跑"目标直接冲突（尽管进度因双重校验仍可续，故定 **P1** 非 P0）。**建议**：`os.replace` 对 `OSError(winerror∈{5,32})`/`PermissionError` 做 **3~5 次小退避重试**；或在 `StateStore.save`/`load` 捕获瞬时 `OSError`；或读方以 `FILE_SHARE_DELETE` 打开。修复点仅限上述 2 文件。 |
| **P2** | `src/stock_ai/scale/model_policy.py::ModelSelector.select`（L48） | 非法 `model_policy` **静默**按 `nearest_past` 处理，无告警/报错。 | `select('bogus', '600519', 2019, [...], {...})` → `'600519_2018'`。 | 拼写错误会被静默吞掉，误导检测口径。**建议**：`policy not in SUPPORTED_POLICIES` 时抛 `ScaleError` 或记 WARNING。 |
| **P2** | `src/stock_ai/scale/units.py::UnitScanner.write_units` | `units.json` 仅记 `status`+`steps`，无显式 `reason` 字段。 | 读 `units.json`。 | AC2"insufficient/missing 有原因"仅以 `status` 隐式满足；建议补 `reason`（如 `no_cache`/`too_few_steps`）便于排障。 |

> 未发现 P0（无"删除产物/状态仍被判 done"、无"检测重新 fit"、无"零回归被破坏"）。

---

## 4. 突变测试汇总（反证通过）

| 突变 | 期望 | 结果 |
|---|---|---|
| 删除某 done 单元 `.zip` → 重启 | 判未完成并重训 | ✅ 重训，sha/created_at 变化 |
| 硬杀遗留 `running` → 重启 | 重置 pending 后重跑 | ✅ 无残留 running，最终 done |
| 删 assigned model 的 `.norm.json` → 检测 | `skipped`+`normalizer_missing`，不 refit | ✅ 未产出报告，fit=0 |
| 训练中并发读状态文件 | 不出现半截 JSON | ✅ 0 非法 JSON（但暴露 P1 写失败） |
| 注入必失败单元 | `failed`+继续整批 | ✅ failed=1，其余 done，exit 0 |

---

## 5. 智能路由判定

- **路由 → `Engineer`（工程师 Alex）**：存在 1 项 **P1** 源码缺陷（`atomic.py` / `state.py` 并发 `os.replace` 无重试可致整批中止）与 2 项 **P2**（`model_policy` 非法值静默、`units.json` 缺 reason）。均为**源码问题**，需工程师修复，非测试代码问题。
- 其余 AC1–AC12 与 12 项必验清单**全部 PASS**。
- 建议修复后仅需回归：① 并发读+写压测（写者侧 0 失败）；② `scale train/detect` 正常跑通；③ 全量 pytest 仍 270/1。

---

## 6. 附录：关键命令与产物路径

- 独立脚本：`.qa_edward/qa_crash.py`、`qa_mutation.py`、`qa_no_refit.py`、`qa_boundary.py`、`qa_ui.py`、`qa_ui_short.py`、`qa_control.py`、`qa_atomic.py`、`qa_atomic2.py`、`qa_abort.py`、`qa_hs300.py`。
- 工作区：`.qa_edward/`（合成+控制）、`.qa_real/`（真实 tencent）、`.qa_full/`（300×20 合成）。
- 关键状态：`.qa_edward/data/scale/state/{train,detect,control}.json`、`data/scale/split.json`、`models/units/*`、`reports/units/*`。
- pytest 原始输出：`.qa_base.txt`（全量 270/1）、`.qa_pre.txt`（既有 221/1）。

---

## 7. Round 2 回归（修复验证）

> 工程师改动：`scale/atomic.py`（`os.replace` 有界小睡重试）、`scale/model_policy.py`（未知策略抛 `ScaleError`）、`scale/units.py`（记录 `reason`）、`cli/scale_cmd.py`（detect 前置校验 `--model-policy`）；新增 `tests/test_scale_atomic.py` 等 7 项。以下均为**独立复测**（非采信其自验）。

| 项 | R1 结果 | R2 结果 |
|---|---|---|
| **P1** 并发读致整批中止 | FAIL（写者 2115/3000 失败；`scale train` exit 1） | ✅ **FIXED** —— 我的 `.qa_edward/qa_atomic2.py`：tight / 0.2s / 2s 三档 **writer_errors 全为 `{}`（0/3000）**；我的端到端 `.qa_edward/qa_abort2.py`：真实子进程 `scale train` + 紧读者 → **exit 0、24.8s、reader 187828 次、stderr 无 PermissionError、最终 finished 12/12**（对照 R1：exit 1 中止）。 |
| **P2-1** 非法 `model_policy` 静默 | FAIL（静默=nearest_past） | ✅ **FIXED** —— `scale detect --model-policy bogus` → **exit 1** + `[错误] 未知的检测模型策略: 'bogus'…`；`ModelSelector.select('bogus',…)` 抛 `ScaleError`；`nearest_past`/`none` 仍正常。 |
| **P2-2** `units.json` 缺 `reason` | FAIL（无字段） | ✅ **FIXED** —— 每条记录含 `reason`：`available→''`、`missing→'no_cache'`、`insufficient→'too_few_steps'`（分布 `{('',15),('no_cache',885)}`）。 |
| 全量回归 | 270/1 | ✅ **277 passed / 1 skipped**（72+72+72+61 dots + `s`；+7 新增，零回归） |
| 断点续训幂等（`.qa_edward/qa_crash2.py`） | PASS | ✅ **5/5 PASS**（硬杀 → 重启：done 单元 sha/created_at/mtime 未变、无残留 running、`done==total-failed-skipped`、被杀的 `000002_2022` 重训为 done） |
| 检测不 refit（`.qa_edward/qa_no_refit.py`） | PASS | ✅ `FeatureEngineer.fit=0 / transform=1` |
| AC3 可复现 | PASS | ✅ seed42 两次 split.json sha256 **`58b4d361…` 与 R1 完全一致** |
| 突变套件（`.qa_edward/qa_mutation.py`） | 10/10 | ✅ **10/10 PASS**（删 .zip 重训、`normalizer_missing` 不 refit） |

### 残余观察（**非缺陷**，可接受）
- 病态**紧读者**（远超真实 UI 频率，单进程紧循环）下，**读方**偶发瞬时 `PermissionError`（R2 中 tight 档约 8293 次/3000 写；真实 2s 轮询档为 0）。读方已优雅处理（`StateStore.load` 宽 `except` 返回空态、UI 500 自愈），且 **写方 0 失败** —— 即"任务不会被并发读中止"这一关键不变量成立。根因是 Windows 读句柄不带 `FILE_SHARE_DELETE` 的固有行为，`utils/io.py` 不在本次改动范围。**判定：可接受，无需修复。**

### Round 2 路由判定
- **→ `NoOne`**：P1 + 2×P2 全部修复且经独立复测确认；核心不变量（断点续训幂等、检测不 refit、可复现、零回归、失败隔离）全部保持；残余仅为 Windows 读侧固有瞬时行为且被优雅处理。
- 无 P0 / 无剩余阻断项。验证通过。

## 公司行为 v2 增量验收

2026-10-16 增量记录；上文为历史编排验收，不将其中的训练运行、数据覆盖或总数算作本轮结果。本轮限定公司行为机制和统一模型接线，禁止真实训练，既有 8766 / 8767 服务不重启。

### 结算、版本与来源

- 独立执行版本 `strict-t1-open-ca-v2`；旧 `strict-t1-open-v1` / legacy 保留原语义并标明遗漏公司行为。观测 `unified-relative-v1`、默认 20×9、三动作、hfq 特征 / raw 撮合、T 日信号 / T+1 开盘及逐股票年度 episode 不变。
- 登记日收盘确权，除权确认税前应收和送转锁股，派息转现金、上市解锁；总权益含应收，锁股不可卖。非 bar 到账 / 上市在下一可用开盘前补记并保留有效日。验证现金 / 股份 / 成本 / 已实现及未实现损益恒等式、独立恒买基准及异常完整回滚。
- 显式 `verified` 覆盖仅是提供者审阅声明，程序不证明源端完整。缺账本、覆盖不足、未知机制、碎股、配股、非标准方案、缺登记 / 除权日线和未上市股份嵌套登记资格拒绝。JSON 三类分配和参考方法不得缺省猜零。原响应归档始终 `unreviewed`，不自动安装执行账本。
- 原生 v2 必须存在非空、六位代码合法唯一且三 hash 格式齐全的 sources 清单；缺整张清单不能冒充新股票。训练内股票三份完整文件 hash 必须匹配；真正新股票仅披露当前 hash。旧 v1 研究默认关闭，显式启用时记录训练 / 执行双版本，缺历史 raw hash 标 `not_recorded`，不能补造历史证据。
- v2 训练准备对 hfq/raw/账本各读一次，数据解析、日历校验、来源指纹与年度 episode 共享该份快照；回放 parquet 解析和 hash 也来自同一次字节读取。反证测试在读取后替换磁盘路径，确认来源、观测及执行仍对应原快照；不声称多文件跨源原子一致。

### 本轮主会话组合回归（不真实学习）

- `outputs/verify_corporate_action_regression.py`：10 个测试文件 **551 passed、5 deselected、0 failure / error，exit 0**。5 个排除项均包含真实训练；脚本额外禁用 NumPy / SB3 包装器及 PPO / SAC / DQN 的真实 learn，实际调用记录为空。6 条 warning 为 Gymnasium 既有观测 Box 无穷边界提示，不改变观测契约。
- 原始证据：`outputs/corporate_action_regression_final.xml`、`outputs/corporate_action_regression_final.json`。JUnit 的测试用例累计耗时与 pytest 墙钟耗时不同，不混同。
- 指定 Node 执行 `node --check src/stock_ai/scale/ui/static/replay.js` 和 `tests/test_scale_replay_ui.js` 均 exit 0；覆盖旧版、v2 22 列、应收 / 锁股 / 完整事件、导出绑定完成结果、研究模式响应版本保护、新 run 默认显式 strict-ca、resume 不传 execution，以及 index 所有内联脚本语法。
- 前阶段 259 与 87 项、目标回归和开发者自验是迭代记录，**不与本轮 551 重复累计**。失败 / 中断的审查或浏览器运行不计通过。

### 真实宁德时代单事件机制夹具

- 使用宁德时代（300750）2023-037 实施公告：<https://static.cninfo.com.cn/finalpage/2023-04-18/1216457838.PDF>，2023-04-18 发布；PDF SHA256 为 `8063ac7204c79d9f856b7a2e21887aa650bc5e2b2c1bbbecb81ab54473d0939b`。行情来自本地 raw，区间 2023-04-24～28，文件 hash 见 JSON，不是本轮拉取的最新行情。
- 固定动作仅检验机制：4/24 信号、4/25 开盘买 2500 股后持有。4/26 正确变为 4500 股并计税前股息 6300 元；旧 v1 权益 565006.00125 元，v2 为 1020306.00125 元，相差 455300 元（2000 股 × 当日 raw 收盘 224.50 元 + 6300 元）。除权参考价 212.99 元，涨停 255.59 元、跌停 170.39 元。
- 脚本每日收盘显式估值，断言两版本首日都是 100 万元；这修正了原验收文件中旧环境 reset 未 mark 导致的首日展示 0，不修改旧环境执行语义。最终产物为 `outputs/corporate_action_case_300750_20230426_final.json` / `.csv`，旧产物保留不覆盖。
- 该结果不是模型策略表现，也不是 300 股票、2006–2025 全历史完成证明。`coverage_proven=false`、`production_ledger_installed=false`；生产缺账本仍拒绝，不将内存测试覆盖声明导入生产。

### 隔离真实浏览器验收

- `outputs/verify_corporate_action_ui.py` + `outputs/verify_corporate_action_browser.cjs` 使用现有 Playwright 和独立缓存 Chromium，随机 loopback 端口，模型 / 行情 / 账本全部来自 outputs 内合成夹具；不加载生产模型，不触发真实训练。
- 最终 `outputs/corporate_action_ui_acceptance_final/acceptance.json`：**passed=true、浏览器子进程 exit 0、19 项检查全部 true**。5 日 / 2 笔成交、v2 22 列、四联图轴与缩放、应收 / 锁股 / 事件详情、真实三类下载与 API / CSV / JSON 交叉一致，筛选变化不污染完成结果，均通过。
- 旧 v1 研究默认关；显式启用披露训练 v1 / 执行 v2、历史 raw `not_recorded` 和强警告；取消后恢复 17 列旧语义。页面新 run 发送 strict-ca，resume 不传 execution。页面控制请求先拦截，再独立验证隔离后端返回 400；训练进程数为 0。
- 页面与控制台异常均为空，9 个夹具文件 hash / mtime / 集合不变；`browser_closed=true`、`server_closed=true`。原 8766 / 8767 PID 保持 61144 / 71088，没有重新加载本次后端。
- 首次浏览器脚本阻塞已停止；另一次预检因过期 Chromium 路径失败，证据保留，不计入成功。修正为已存在独立缓存路径后 probe2 与最终运行分别成功，不安装任何新依赖。网络恢复期间误重复提交一次相同 final 命令，因目录已存在在 mkdir 阶段 FileExistsError / exit 1，未创建夹具 / 服务 / 浏览器，也未覆盖证据；最终成功只取实际完成的那一次，不累计失败 / 拒绝运行。最终 screenshot 和下载在上述 final 目录，仅为合成机制，不是真实股票表现。

### 受保护输入复核

`outputs/verify_corporate_action_protected.py` 只读对照验收前基线：**643 / 643 文件内容 SHA256 与 mtime_ns 均不变**，无新增 / 缺失 / 变化 / 读取错误。范围为 `data/raw` 302 个文件、`data/raw_hfq` 303 个、`models/unified` 38 个；范围不是全盘个人文件扫描，也不声称覆盖未列入基线的其他目录。最终证据 `outputs/corporate_action_protected_inputs_final.json`；生产 `data/corporate_actions` 下仍无 JSON 账本，未安装伪造的 verified 覆盖。

### 交付边界

个人差异化股息税、ST 历史身份、新股无涨跌幅限制、真实流动性和历史逐日税费仍不完整；未来引擎 / 舍入规则变化必须版本化。现有模型未重训，研究覆写不等于修正旧训练奖励。代码接线可验收与生产数据全覆盖、授权训练和手动部署是四种不同状态。

# QA 报告 —— stock_ai 远程 Linux 部署 & Bash 脚本移植验证

- **报告人**：QA 工程师 严过关（software-qa-engineer）
- **被验对象**：`run.sh` / `scripts/{setup_env,run_pipeline}.sh` / `src/stock_ai/data/sources/tencent_source.py` / `tests/test_tencent_source.py`
- **远程主机**：`<user>@<host>`（内网验证机，信息已脱敏），项目目录 `<remote-dir>`，解释器 `.venv/bin/python`（Python 3.11.6）
- **验证方式**：**独立逐文件哈希比对 + 真机实跑 + 绕过项目代码直接 curl 源站 + 独立编写的测试用例**
- **结论**：**未发现"部署是假象"类问题；部署真实、脚本可跑、腾讯源修复有效。发现 2 个 P1 参数校验缺陷、若干 P2 健壮性/数据质量问题。**

> 说明：本报告与既有 `docs/QA_REPORT.md` 独立，未覆盖原文件。

---

## 0. 总体判定

| 维度 | 结论 |
|---|---|
| 远程部署真实性 | ✅ 通过（63 个 src `.py` + 4 个 bash 脚本 + 目标测试文件，与本地 **逐字节**一致） |
| Bash 脚本可用性 | ✅ 通过（7 条命令 + status/help/cli 全部独立进程可跑，退出码正确） |
| 腾讯源修复正确性 | ✅ 通过（真机拉满，负价论断经独立 curl 证实，分页无重复/缺口） |
| 参数与异常健壮性 | ⚠️ 部分通过（`--algo` 已校验；`--backend`/`--source` **未校验**，静默接受非法值） |
| 复现性 | ✅ 通过（data_hash 与 numpy 权重均可复现） |
| **智能路由** | **→ 工程师（Alex）**：2 个 P1 源码缺陷（参数校验），详见 §5 |

全量测试：**147 用例，144 通过，3 失败**（3 个失败均为本次 QA 新增用例，用于记录 §5 的真实缺陷）。

---

## 1. A — 远程部署真实性（本轮最高优先级）

### A1. src 逐文件哈希比对
对本地与远程 `src/**/*.py`（63 个）计算 `md5sum` 后排序逐行 `diff`：

```
$ diff /tmp/local_src.txt /tmp/remote_src.txt     # 本地/远程各 63 行
(无输出)                                          # 完全一致
SRC_IDENTICAL_COUNT=63
```

**方法**：本地 `find src -name '*.py' | sort | xargs md5sum` 与远程同命令 → 结果 `diff`。

### A2. 4 个 bash 脚本哈希 + LF/无 BOM

| 文件 | 本地 md5 = 远程 md5 | 行尾 CR 数 | 首 3 字节 |
|---|---|---|---|
| `run.sh` | `953f034685d6fa8d074eda80ddb9c3b2` ✅ | 0 | `23 21 2f`（`#!/` 非 BOM） |
| `scripts/setup_env.sh` | `8647a0492e6e9e2688f16f256d66fe2e` ✅ | 0 | `23 21 2f` |
| `scripts/run_pipeline.sh` | `75cb5d4361bcb6e3b1984e332c456073` ✅ | 0 | `23 21 2f` |

`file` 判定均为 `Bourne-Again shell script, Unicode text, UTF-8 text executable`。**PASS**。

### A3. 测试文件同步

`tests/test_tencent_source.py` 本地/远程 md5 均为 `44d1173b0e89dd0b175246bc71ab88f5`。**PASS**。

### A4. 远程 venv 依赖（实测版本）

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| Python | 3.11.6 | numpy | 2.4.6 |
| pandas | 2.3.3 | gymnasium | 1.1.1 |
| torch | 2.7.1+cpu | stable-baselines3 | 2.8.0 |
| pyarrow | 21.0.0 | matplotlib | 3.11.2 |
| click | 8.5.0 | PyYAML | 6.0.3 |
| requests | 2.34.2 | tqdm | 4.70.1 |
| scipy | 1.17.1 | pytest | 9.1.1 |

`stock_ai` 为可编辑安装（`-e <remote-dir>`，v1.0.0）。**PASS**。

### A5. ps1 原文件未被动过
`run.ps1`/`setup_env.ps1`/`run_pipeline.ps1` 本地与远程 md5 完全一致，且改动时间（Sep 29）早于 bash 工作（Oct 10）。**PASS**。

---

## 2. B — 腾讯源修复独立验证（工程师核心改动）

### B1. 真机实拉 600519（独立执行）
```
$ rm -f data/raw/600519.parquet
$ bash run.sh data-fetch --symbols 600519 --start 20120101 --end 20241231 --source tencent --force
[INFO] 拉取成功 600519 [tencent] 2060 行 (20120101~20241231)
600519: 2060 行 [tencent]          # FETCH_EXIT=0
$ bash run.sh data-info --symbols 600519
  600519: 2060 行 20160705~20241231 source=tencent gaps=0
```
断言 `source=tencent` ✅、非 synthetic ✅、非 320 行 ✅。实际 2060 行 / 20160705~20241231。**PASS**。

### B2. 低分红标的拉满（000001）
```
$ bash run.sh data-fetch --symbols 000001 --start 20120101 --end 20241231 --source tencent --force
[INFO] 拉取成功 000001 [tencent] 3145 行 (20120101~20241231)
```
3145 行（3000 量级）✅，覆盖 20120104~20241231。**PASS**。

### B3. 独立复核"600519 负价"论断（绕过项目代码，直接 curl）
**论断成立。** 直接请求源站：

```
count=320  -> rows=320, first=2023-09-05
count=640  -> rows=640
count=800  -> rows=800                 # 实测有效上限
count=1000 -> rows=640                 # >800 被静默截断到 640（非 800）
count=2000 -> rows=640
count=3000 -> data=[] , msg="param error"   # >=3000 直接报错
```
✅ 证实文档所述"单次约 800 根、>800 静默截断、>=3000 param error"，故 `_MAX_BARS_PER_REQ=800` 取值正确。

再取 2016 年附近 qfq 原始数据，确认**早年确实为负价**：
```
GET .../fqkline/get?param=sh600519,day,2000-01-01,2016-12-31,800,qfq
2013-09-23 O=-187.999 C=-188.975 H=-187.991 L=-189.603
2013-09-24 O=-188.818 C=-192.289 ...
min close = -207.198
first close>0: 2016-07-04 close=2.316
```
`validate_frame`（`src/stock_ai/data/schema.py:131`）规则为 `close>0 & open>0`，故负价行被剔除 → 600519 只能保留 2016-07-05 起的 2060 行。**工程师论断与代码行为完全吻合，非分页漏数据。PASS**。

### B4. 分页正确性（无重复/缺口/升序）
对缓存 parquet 独立分析：

| 标的 | 行数 | 区间 | 重复日期 | 升序 | NaN | close≤0 |
|---|---|---|---|---|---|---|
| 000001 | 3145 | 2012-01-04~2024-12-31 | 0 | ✅ | 无 | 0 |
| 600519 | 2060 | 2016-07-05~2024-12-31 | 0 | ✅ | 无 | 0 |

**跨标的缺口核对**（最严格）：在 2016-07-05~2024-12-31 重叠窗口内，000001 的交易日中 600519 仅缺 5 天 `20160926~20160930`。经 curl 复核，该 5 天 600519 的 qfq `open/close` 为负（如 09-26 C=-2.854、09-27 O=-2.704），**同为负价被剔除，非分页 bug**。其余大间隔（≤17 天）均为春节/国庆休市。**PASS（附数据质量说明，见 P2-3）**。

### B5. fallback 链未被破坏
- `data refresh` 增量：`缓存已最新: 000001 (last=20241231)` ✅
- `--source synthetic`：隔离目录下 `拉取成功 600000 [synthetic] 260 行` ✅

### B6. 缓存保护语义（独立验证）
自行编写（不复用 `test_synthetic_fallback_guard.py`）：真实源预写缓存 → 令全部真实源失败、synthetic 兜底 → 断言 **parquet 字节完全未变、返回旧真实数据**。通过。增量刷新同样保留真实历史。**PASS**。

---

## 3. C — Bash 脚本功能对等性与健壮性

### C1. 6 条 CLI + doctor 独立进程可跑（逐条实测）
| 命令 | 退出码 | 关键输出 |
|---|---|---|
| `data-fetch` | 0 | `600519: 2060 行 [tencent]` |
| `data-refresh` | 0 | `缓存已最新: 000001` |
| `data-info` | 0 | `2060 行 ... source=tencent` |
| `sample-build` | 0 | `train 728 / val 112 / test 484 步, data_hash=6763043e...` |
| `train`(SB3) | 0 | `backend=sb3 algo=ppo, 模型=ppo_....zip` |
| `backtest` | 0 | `年化7.32% 累计15.12% mdd17.90% sharpe0.482 trades25` |
| `predict` | 0 | `242 条, 买130 卖102 持有10` |
| `doctor` | 0 | 全绿，`-> 将使用: SB3` |

均为**独立 python 进程**，非单脚本串联。**PASS**。

### C2. `run.sh status`
6 大板块齐全：`[1]原始日线数据`（显示 `来源=tencent` ✅）/`[2]样本数据`/`[3]训练模型`/`[4]回测报告`/`[5]预测输出`/`[6]最近日志`；中文无乱码；退出码 0。**PASS**。

### C3. help / 子命令 help / cli 透传
- `bash run.sh help` → 完整帮助，RC=0 ✅
- `bash run.sh train --help` → 完整 CLI 选项，RC=0 ✅
- `bash run.sh cli "data info"` → 正确词分割并执行，RC=0 ✅

### C4. 纯 NumPy 后端兜底
`--backend numpy` 训练（0.3s）→ 回测 → 预测全链路 RC=0；status `[3]` 可见 `[ppo/numpy]` 模型。**架构核心兜底未因 SB3 可用而失效。PASS**。

### C5. 脚本层健壮性边界
| 场景 | 结果 | 判定 |
|---|---|---|
| 无参数执行 | 显示 status 摘要，RC=0 | ✅ |
| 未知命令 `bogus-cmd` | `[错误] 未知命令` + help，RC=1 | ✅ |
| 非项目目录 + 绝对路径 | `$ROOT` 推导正确，RC=0 | ✅ |
| `STOCKAI_PYTHON=/nonexistent` | 明确报错，RC=1 | ✅ |
| `.venv` 临时改名 | RC=1（`No module named stock_ai`），**已还原** | ⚠️ 见 P2-2 |
| `run_pipeline.sh` 连跑两次 | 两次均 RC=0，指标一致（幂等） | ✅ |

> 破坏性测试（改名 venv）均已在同一条命令内还原，并复验 `.venv/bin/python --version` 正常。

### C6. 与 ps1 对等性（静态对照）
| 项 | `run.ps1` | `run.sh` | 判定 |
|---|---|---|---|
| 命令集 | status/help/cli/data-fetch/data-refresh/data-info/sample-build/train/backtest/predict/doctor | 同 | ✅ 对等 |
| status 6 板块 | [1]~[6] | 同 | ✅ 对等 |
| 参数 | Symbols/Start/End/Source/Force/Algo/Timesteps/Backend/Model/Out/Config | 同 + `--reward/--samples/--window` 且 predict 支持 `--symbols` | ✅ bash 为超集 |
| 错误处理 | 未知命令→help+exit1 | 同 | ✅ 对等 |

`setup_env.sh`（6 步，含建 venv + 镜像源 + torch/SB3 失败不致命）与 `run_pipeline.sh`（6 步，含源/步数环境变量覆盖）均为 ps1 的**功能超集**，**无能力缺失**。**PASS**。

---

## 4. D — 异常与错误路径

| 编号 | 场景 | 命令 | 结果 | 判定 |
|---|---|---|---|---|
| D1 | 不存在代码 999999 | `data-fetch --symbols 999999 --source tencent` | 三真实源失败→synthetic 260 行，RC=0 | ⚠️ 见 P2-4 |
| D2a | `--algo BOGUS` | `train --algo BOGUS` | `[错误] 不支持的算法`，RC=1 | ✅ |
| D2b | **`--backend bogus`** | `train --backend bogus` | **静默按 auto/SB3 训练，RC=0** | ❌ **P1-1** |
| D2c | `start>end` | `data-fetch --start 20241231 --end 20120101` | 所有源失败 + `合成数据区间为空`，RC=1 | ✅ |
| D2d | **`--source nosuch`** | `data-fetch --source nosuch` | **静默忽略，走默认源链，RC=0** | ❌ **P1-2** |
| D2e | `--reward bogus` | `train --reward bogus` | 静默接受，RC=0 | ⚠️ P2-5 |
| D3 | 模型文件缺失 | `backtest/predict --model models/nope.zip` | `[错误] 模型文件不存在`，RC=1 | ✅ |
| D4 | 损坏 parquet（截断） | `data-info` / `data-fetch` | 警告 + `缓存为空` / `缓存读取失败，将重拉`，不崩溃 | ✅ |
| D5 | 断网/源不可用 | 见 D1、B6 | 回退 synthetic，**不覆盖真实缓存**，日志清晰 | ✅ |

> 注：D2c 的错误提示为"所有数据源均失败"而非"start>end 非法"，能明确报错并非零退出，但提示可更友好（P2-6）。

### D6. 静默失败扫描
`src/` 无 `except: pass`；33 处 `except Exception` 均带 `noqa`/注释，绝大多数伴随 `logger.warning` 后再决策（非静默吞）。腾讯源 `_load_json` 对空/非法 JSON 显式 `raise DataSourceError`；分页循环对空行 `break` 后若 `all_rows` 为空则抛错，**未掩盖真实错误**。**PASS**。

---

## 5. 问题清单（按优先级）

### 🔴 P1-1　`--backend` 非法值被静默接受
- **文件/行号**：`src/stock_ai/train/factory.py:80`（`pref = str(preference or cfg.backend or "auto").lower()` 之后**无白名单校验**）；`src/stock_ai/train/train_cli.py:86-102` 仅校验 `algo`，未校验 `backend`。
- **复现**：`bash run.sh train --config configs/fast_dev.yaml --backend bogus --timesteps 100`
- **预期**：明确报错（如 `不支持的 backend: 'bogus'（可选: auto|sb3|numpy）`）+ 非零退出。
- **实际**：`后端: SB3 (ppo) [auto]` → `backend=sb3`，**RC=0**。非法值被静默降级为 auto。

### 🔴 P1-2　`--source` 非法值被静默接受
- **文件/行号**：`src/stock_ai/data/loader.py:70-82`（`_build_sources` 对未知 `preferred` 直接忽略）；`src/stock_ai/config/schema.py:32`（`source` 无校验）。
- **复现**：`bash run.sh data-fetch --symbols 600519 --source nosuch --cache-dir data/_qa_sym --force`
- **预期**：明确报错 + 非零退出。
- **实际**：静默走默认源链（akshare→tencent…），成功拉到 600519 数据，**RC=0**。

> 两者与上一轮已修复的 `--algo` 静默接受属**同一类缺陷**（参数白名单缺失），`--algo` 已在 `train_cli.py:91` 修好，但 `--backend`/`--source` 遗漏。

### 🟠 P2-1　meta.json 丢失时真实缓存失去保护
- **文件/行号**：`src/stock_ai/data/loader.py:136-137`（`prev = meta.get("symbols",{}).get(symbol,{}).get("source","")`；`return bool(prev) and prev != "synthetic"`）。
- **复现**：真实缓存存在但删除 `data/raw/meta.json` → 真实源全挂 → synthetic 兜底 → **覆盖了真实 parquet**。
- **证据**：本次新增用例 `test_meta_missing_still_protects_real_cache` 失败。
- **影响**：meta.json 损坏/丢失（可发生）时，离线兜底会覆盖真实数据。建议改为"缓存来源未知时按真实处理（保护）"。

### 🟡 P2-2　venv 缺失时的报错不够友好
- `run.sh` 在 `.venv` 缺失但系统存在 `python3` 时回退到系统解释器，报 `No module named stock_ai`（RC=1，非静默，但非"请先运行 setup_env.sh"的引导信息）。建议 `resolve_python` 校验目标解释器可 `import stock_ai` 或优先提示。

### 🟡 P2-3　600519 前复权负价导致数据块缺失（数据质量）
- 600519 qfq 在 2012-01~2016-07-04 及 2016-09-26~30 为负价，被 `validate_frame` 剔除 → 仅 2060 行且有 5 日空洞。属腾讯 qfq 数据特性，非代码逻辑错误，但建议：对这类高累计分红标的，可回退 `day`（不复权）字段或做裁剪/告警，避免静默丢历史。

### 🟡 P2-4　不存在的股票代码与"断网"无法区分
- 999999 与网络故障同样落到 synthetic 兜底（RC=0），用户无法据此判断"代码写错"还是"网络不通"。建议对 `symbols` 增加格式/存在性预校验。

### 🟡 P2-5　`--reward` 非法值静默接受（同类）
- `bash run.sh train --reward bogus` RC=0，无校验。建议与 `algo` 一致加白名单。

### 🟡 P2-6　`start>end` 提示可更友好（建议）
- 目前靠"所有源失败"间接报错，虽非零退出，但根因提示不直观。

---

## 6. 代码质量抽查（E）

| 项 | 结果 |
|---|---|
| `_MAX_PAGES=50` 死循环风险 | ✅ 无。`for _ in range(_MAX_PAGES)` 硬上限；每轮 `nxt_end = _prev_day(...)`，并有 `if nxt_end >= cur_end: break` 兜底，终止可靠 |
| `_prev_day` 跨月/跨年 | ✅ `2024-03-01→20240229`（闰年）、`20240101→20231231` 均正确 |
| 硬编码 Windows 路径 | ✅ 无（`run.sh`/`scripts/*.sh`/`src/**` 均未命中 `C:\`、`jerrykong`、`Scripts/python.exe`） |
| 静默吞异常 | ✅ 无 `except: pass`；广泛捕获处均有日志/上抛 |
| 实验残留 | ✅ `data/raw` 仅 2 个真实 parquet + meta.json；QA 自建的隔离目录已清理。仓库根目录存在的历史 `.tmp_*`/`.pytmp*` 为早期产物（非本次），未纳入部署（`tar` 不含顶层 `data` 之外无影响），建议清理 |

---

## 7. 复现性（F）

- **F1**：同 seed 连续两次 `sample build` → `data_hash = 6763043eaaf5c3e5493b7b608bb805396993c823175022ccc5f403f79b97bbc1`（两次完全一致）✅
- **F2**：同 seed 两次 `train --backend numpy` → 模型 zip 内部文件 `sha256` 完全一致
  `inner-sha256 = 10d23c48e33438a9ca94cd81e6f448133ad088d2ef56a331615565e8d412423d`（两次相同）✅
  （外层 zip 的 md5 因归档时间戳略有差异，权重本身可复现）

---

## 8. 交付物

1. 本报告：`docs/QA_REPORT_LINUX.md`
2. 新增测试：`tests/test_qa_linux_verification.py`（12 用例，9 通过 / 3 失败）
   - 通过：隐式 synthetic 缓存保护、显式 synthetic 覆盖、增量保护、损坏 parquet 重拉、`_load_json`/`_prev_day`/`_estimate_bars`、**分页装配无重复且覆盖 start**
   - 失败（= 缺陷证据）：`test_meta_missing_still_protects_real_cache`(P2-1)、`test_invalid_backend_is_rejected`(P1-1)、`test_invalid_source_is_rejected`(P1-2)

---

## 9. 遗留风险

- P1-1/P1-2 参数校验缺陷会误导用户（显式指定 `--backend sb3` 拼错即静默变 auto，可能产生非预期训练/成本）。
- P2-1 在 meta.json 损坏场景下存在**真实数据被合成数据覆盖**的风险。
- P2-3 高分红标的的前复权历史被静默截断，训练可用区间受限但不报错。

---

## 10. 最终判定

**部署真实性、腾讯源修复、Bash 脚本移植三项核心目标全部达成且经独立验证。** 未发现"部署是假象"类问题。

**存在 2 个 P1 源码缺陷（参数校验）→ 建议路由至工程师（Alex）修复；其余为 P2 健壮性/数据质量改进项。** 远程环境已恢复至**可正常使用**状态（venv 完好、真实缓存完好、无 QA 残留）。

---

### 附：关键复现命令速查
```bash
# P1-1
ssh <user>@<host> "cd <remote-dir> && bash run.sh train --config configs/fast_dev.yaml --backend bogus --timesteps 100"   # 期望 RC!=0，实际 RC=0
# P1-2
ssh <user>@<host> "cd <remote-dir> && bash run.sh data-fetch --symbols 600519 --source nosuch --force"                    # 期望 RC!=0，实际 RC=0
# 回归
ssh <user>@<host> "cd <remote-dir> && .venv/bin/python -m pytest tests/test_qa_linux_verification.py -v"
```

---
---

# 第 2 轮回归验证（工程师修复后复核）

- **复核人**：QA 严过关　**日期**：第 2 轮（最后 1 轮）
- **方法**：不采信自测数字，**独立逐文件哈希 + 照搬原复现命令 + 独立构造用例 + CLI 实测**
- **结论**：**工程师申报的 2×P1 + 4×P2 修复全部属实且有效；部署一致、全量测试 0 失败。新增 1 项 P2 设计残余漏洞（P1-2 白名单取舍所致）。判定：可交付。**

## R1. 复核项清单

| 编号 | 复核内容 | 方法 | 结果 |
|---|---|---|---|
| R-4.1 | 本地↔远程一致性（src+tests+run.sh+scripts） | 89 文件 `md5sum` 归一化 `diff` | ✅ 全部一致 |
| R-4.2 | `utils/dates.py` 已同步/可导入/被引用 | 哈希+`grep` | ✅（loader/builder/predictor 均引用 `validate_range`） |
| R-1 | P1-1 `--backend bogus` | 照搬原命令 | ✅ 明确报错 + RC=1，无 `bogus_*` 垃圾产物 |
| R-1 | P1-2 `--source nosuch` | 照搬原命令 | ✅ 明确报错 + RC=1 |
| R-2 | **P1-2 白名单残余漏洞审计** | 自建用例 + 探针脚本 | ⚠️ **成立（P2）**，详见 R2 |
| R-3 | P2-1 meta 缺失保护 | 独立构造 + CLI 代理断网 | ✅ 保护生效；反向语义（显式 synthetic / 无缓存）未被误伤 |
| R-3 | P2-5 `--reward` | CLI 实测 | ✅ `nosuch`→RC=1；`gross/drawdown_penalty/sharpe` 可用 |
| R-3 | P2-2 venv 缺失提示 | `STOCKAI_PYTHON=/usr/bin/python3` | ✅ 友好中文引导 + RC=1；正常路径 RC=0 |
| R-3 | P2-6 `start>end` | data-fetch / sample-build / predict 三处 | ✅ 均明确中文错误 + RC=1；边界未误伤 |
| R-3 | P2-4 兜底日志/RC 语义 | 代理断网实测 | ✅ `[兜底]`/`[保护]` 可区分；离线仍 RC=0 |
| R-4.3 | 全量测试 | `.venv/bin/python -m pytest -q` | ✅ **173 passed / 1 skipped / 1 xfailed / 0 failed** |
| R-4.4 | 代码质量（7 改动文件） | grep | ✅ 无静默吞异常、无 Windows 路径、`set -euo pipefail` 完好 |
| R-4.5 | 无回归 | 6 命令/status/NumPy 后端/真实拉取/复现性 | ✅ 全部保持 |

## R2. ⚠️ P1-2 白名单设计的残余漏洞（独立判定）

**工程师设计**：校验集合 = 静态白名单 `SUPPORTED_SOURCES` **∪ 所有 `DataSource` 子类的 `name`**（`_known_source_names()`），以避免纯静态白名单打红使用自定义源名（`realsrc`/`ok`/`failing`）的既有用例。

**独立结论：残余漏洞成立。**

证据一（探针脚本，定义了一个 `name="unregistered_custom"` 的 `DataSource` 子类）：
```
known contains 'unregistered_custom': True          # 校验通过
loader.sources names: ['akshare', 'tencent', 'eastmoney', 'synthetic']   # 自定义源被丢弃
custom source actually used: False                  # 静默忽略 → 走默认 fallback 链
```

证据二：新增用例 `test_p1_2_residual_unregistered_source_must_error` 断言"未注册名应报错"——当前 **XFAIL**（`DID NOT RAISE ConfigError`）。

**根因**：`loader.py` 中存在**两个真源**——`_known_source_names()`（用于校验）与 `_build_sources()` 内的 `registry`（用于实际构造）。二者可能不一致：名字可"已知"却"未注册"，于是 `if preferred in registry` 为假 → 被静默跳过 → 返回默认链（功能上等价于 RC=0 的静默接受，与 P1-2 同类）。

**严重级别：P2（非 P1）**。理由：**经发布的 CLI 不可达**——全新进程内只存在 4 个内置子类，`--source <任意未知值>` 会被正确拒绝（已验证 `nosuch`→RC=1）。仅在**同进程内定义了未注册子类**（测试夹具/插件/库嵌入）时才触发；且触发后退化为默认链（仍能取到真实数据），不产生数据正确性风险。

**建议修法（只评估，未改 src）**：
1. **首选**：单一真源——以 `registry` 键为准做校验，`_build_sources` 对未注册名明确 `raise ConfigError`。
2. 为保留测试可用自定义源的便利，给 `StockDataLoader.__init__` 增加可注入参数（如 `sources: list[DataSource] | None = None` 或可覆盖的 `_registry`），让用自定义源的用例**直接注入**而非依赖 `_build_sources` 接受任意名；随后既有 6+ 条用例同步改造（把 `source` 设为已注册值 + 注入 sources）。
3. 最小改动版：保留 `_known_source_names()` 校验，但在其后追加 `if preferred not in registry: raise ConfigError("数据源 X 未在当前运行时注册")`——需同步调整上述自定义名用例。

**边界评估**：
- `--source TENCENT`（大小写不一致）→ ✅ 正确识别（`preferred=...lower()` 后再匹配），CLI 实测 RC=0 拉到 tencent 数据。
- `name=""`（空串子类）→ 被 `if n and ...` 跳过，不污染集合；`cfg.source=""` 视为默认 `akshare`，不报错（合理）。
- `name` 为非字符串 → 循环内 `str(n).lower()` 已兜底（合理）。

## R3. P2 修复逐项实测证据

- **P2-1（关键）**：独立构造"真实缓存 + 删除 meta.json"，断网触发隐式 synthetic 兜底 →
  `[保护] ... synthetic 兜底数据未写入 600519 (已有真实缓存 242 行)`，parquet md5 **前后一致**，RC=0。
  反向：① 显式 `--source synthetic` 正常写入 ✅；② 首次无缓存（离线）→ `[兜底]` 日志 + 正常写入 260 行 ✅（未被过度保护误伤）。
- **P2-5**：`--reward nosuch` → `[错误] 不支持的奖励类型: 'nosuch'（可选: ['drawdown_penalty','gross','sharpe']）` RC=1；`--reward gross` RC=0 ✅。
- **P2-2**：`STOCKAI_PYTHON=/usr/bin/python3 bash run.sh status` → `[错误] 当前解释器未安装 stock_ai（或依赖不完整）...` + 引导 + RC=1；正常 `bash run.sh status` RC=0 ✅。
- **P2-6**：`data-fetch`/`sample-build`/`predict` 三处 `start>end` 均明确中文错误 + RC=1；边界（`start==end` 单日→1 行、正常区间→242 行）RC=0 ✅。
- **P2-4**：离线时日志区分 `[保护]`（覆盖真实缓存的兜底被拒）与 `[兜底]`（无缓存时合成数据写入）；**RC 语义未变（仍 0）**，fallback 契约未破坏 ✅。

## R4. 无回归确认

- 真实缓存保持 600519=2060 行 / 000001=3145 行，`来源=tencent`（status `[1]`）。
- 6 条命令 + doctor 独立可跑；`run.sh status` RC=0、6 板块完整。
- 纯 NumPy 后端 train→backtest→predict 全链路 RC=0。
- 复现性：两次 `sample build` → `data_hash=6763043e...`（与第 1 轮一致）。

## R5. 第 2 轮总结

| 项 | 结果 |
|---|---|
| 2×P1（backend / source 校验） | ✅ 已修好（RC=1 + 明确报错 + 无垃圾产物） |
| P2-1/P2-2/P2-4/P2-5/P2-6 | ✅ 全部修复到位 |
| P1-2 设计残余漏洞 | ⚠️ 成立，判定 **P2**（CLI 不可达），已给建议 |
| 部署一致性 | ✅ 89 文件逐字节一致 |
| 全量测试 | ✅ 173 passed / 1 skipped / 1 xfailed / 0 failed |
| 是否可交付 | ✅ **可以交付**（残余漏洞为可选项，不影响发布） |

> 新增 `xfail` 用例 `test_p1_2_residual_unregistered_source_must_error`：当前 XFAIL = 漏洞存在；若工程师采纳建议修复，该用例将自动 XPASS，可据此移除 `xfail` 标记。

---
---

# 第 3 轮验证（新功能验收 + 测试削弱自查）

- **复核人**：QA 严过关
- **被验对象**：`--adjust` 复权功能（qfq/hfq/none）+ P1-2 残余漏洞修复 + 缓存口径一致性；并**自查工程师对我测试文件的改动**
- **方法**：不采信自测数字；**独立脚本（16 项）+ 真机 CLI + 逐文件哈希 + 突变测试（mutation testing）反证用例有效性**
- **结论**：**两项新功能均正确；hfq 与 none 价格确实不同；CLI 覆盖路径无法绕过校验；P1-2 残留漏洞已彻底修复；我的测试文件未被削弱（反而被加强），但我在自查中发现并修复了自己 Round-1 分页测试的一个真实盲点。判定：可交付。**

## T1. 验证项清单

| 编号 | 验证内容 | 方法 | 结果 |
|---|---|---|---|
| 一.1 | `normalize_adjust` 归一化/白名单 | 独立脚本 | ✅ `None/""/空白`→`none`；大小写归一；非法→`ConfigError` |
| 一.2 | `DataConfig.__post_init__` 构造即校验 | 独立脚本 | ✅ 非法值构造即抛 |
| 一.3 | akshare `none`→`""` 映射 | 离线检查 `_ADJUST_MAP` | ✅ `{"qfq":"qfq","hfq":"hfq","none":""}` |
| 一.4 | 腾讯三口径字段/param 映射 | 离线检查 | ✅ `qfq→(qfqday,qfq)`、`hfq→(hfqday,hfq)`、`none→(day,"")` |
| 一.5 | **CLI 覆盖路径能否绕过校验** | 模拟 `setattr` 覆盖 + 构造 loader | ✅ **不可绕过**（见 T2） |
| 一.6 | **hfq 与 none 真实价格是否不同** | 真机网络拉取 600519/600000 | ✅ **不同（38/38 个共同日全不同）**（见 T3） |
| 一.7 | `--adjust bogus` / `--adjust BOGUS` | CLI 实测 | ✅ 明确报错 + **RC=1**（fetch 与 refresh 均验证） |
| 二.1 | 旧 `meta.json`（无 `adjust` 字段） | 独立构造 + CLI | ✅ `data info` 显示 `adjust=-`；`status` 显示 `复权=-` |
| 二.2 | 旧 meta 触发保守重拉 + 自愈 | 独立脚本 | ✅ 不误命中旧缓存；重拉后 meta 自动写回 `adjust` |
| 二.3 | 口径切换不命中旧缓存 | 独立脚本 + 真机 | ✅ 记录 WARNING + 全量重拉，不拼接混合口径 |
| 三.1 | registry 与白名单单一真源 | 独立脚本 | ✅ `SUPPORTED_SOURCES == frozenset(_SOURCE_FACTORIES)` |
| 三.2 | 未注册的合法子类名必须报错 | 独立脚本 + `pytest.raises` | ✅ `unregistered_custom/plugintest/nosuch` 均抛 `ConfigError` |
| 三.3 | 注入路径保留可扩展性 | 独立脚本 | ✅ `StockDataLoader(..., sources=[...])` 仍可用 |
| 三.4 | `--source` 大小写不敏感 | 独立脚本 | ✅ `TENCENT/Tencent/tEnCeNt` 均正确识别 |
| 三.5 | 旧 `_known_source_names` 已删除 | 独立脚本 | ✅ 无残留引用 |
| 四.1 | **测试文件是否被削弱** | **突变测试反证** | ✅ **未被削弱**（见 T4） |
| 四.2 | **自查分页测试盲点** | 突变测试 | ⚠️ 发现并修复（见 T5） |
| 五.1 | 本地↔远程一致性 | 100 文件 `md5sum` 归一化 `diff` | ✅ 100/100 逐字节一致 |
| 五.2 | 全量测试无回归 | 远程 pytest | ✅ **219 passed / 1 skipped / 0 failed** |

独立脚本：`.qa_r3/qa_r3_verify.py`，**16/16 通过**。

## T2. CLI 覆盖路径无法绕过校验（关键）

`AppConfig.merge_cli_overrides()` 对 `data.adjust` 采用**纯 `setattr`，不做校验**（这是 `--adjust` 覆盖进入配置的实际路径）。因此专门审计"绕过 `__post_init__` 后能否把非法口径送进系统"：

```
cfg = DataConfig(source="synthetic"); cfg.adjust = "BOGUS"   # 绕过 __post_init__
StockDataLoader(cfg)                                          # → 抛 ConfigError ✅
```

**结论：不可绕过。** 防线是**纵深防御**——`loader.__init__` 第 102 行 `self.config.adjust = normalize_adjust(self.config.adjust)`，且各数据源构造器（`AkShareSource`/`TencentSource`）内部再次 `normalize_adjust` + 白名单校验。CLI 实测 `--adjust BOGUS` → RC=1、无垃圾产物。

## T3. hfq 与 none 价格"确实不同"（关键，真机网络）

| 标的 | 区间 | 共同交易日 | 价格不同日数 | 示例（首日收盘） |
|---|---|---|---|---|
| 600519 | 2024-01-02~2024-03-01 | 38 | **38（全部）** | `20240102`: hfq=**10530.129** vs none=**1685.01** |
| 600000 | 同上 | 38 | 38 | qfq 与 hfq 亦逐日不同 |

> 说明：`hfq`（后复权）对 600519 这类历史分红极多的标的，早期价格会被放大到万元级；`none` 为原始价。三者字段不同（`qfqday`/`hfqday`/`day`），映射与 param 拼接均已独立核对。**复权映射真实生效，绝非"换个名字拿到同一份数据"。**

## T4. 测试削弱自查（关键，工程师改动了我的测试文件）

工程师对我 `tests/test_qa_linux_verification.py` 的改动：① 将 `source="realsrc"` 等用例改为**注入**（`sources=[...]`）以适配 P1-2 修复；② 将我留的 `xfail` 用例改为**硬断言**并**参数化**（3 个名字）；③ 新增注入路径用例。**无删除、无弱化迹象**——但"看着像加强"不等于"真的能抓到 bug"。故用**突变测试反证**（在 `/tmp` 的源码副本上植入原始缺陷，看用例是否会红）：

| 突变 | 植入的缺陷 | 期望 | 实测 |
|---|---|---|---|
| A | 还原 P1-2：未知 `--source` 静默回退 akshare | 用例应失败 | ✅ **4 条失败**（`test_invalid_source_is_rejected` + 3×参数化残余用例） |
| B | 关闭缓存保护（`_should_protect_cache` 恒 `False`） | 用例应失败 | ✅ **3 条失败**（隐式兜底覆盖、增量刷新保留、meta 缺失保护） |
| C | 破坏分页（`_MAX_PAGES=1`） | 用例应失败 | ⚠️ 见 T5（我原测试未抓到 → 已修复） |

**结论：A/B 两类核心用例在缺陷复现时全部转红 → 工程师的改动既未削弱、也未"假强化"，断言依然真实守护被防行为。** 反之，改动前的 `source="realsrc"` 直传写法在修复后会因 `ConfigError` 而无法运行；工程师改用注入是**必要且正确**的适配（且注入路径本身也是一条被保留的能力，见 `test_p1_2_unregistered_source_via_injection_works`）。

附带发现：`.pytest_cache/lastfailed` 记录工程师**自己**的 `test_param_validation.py::test_loader_accepts_custom_source_subclass` 曾失败——该用例断言的是**旧的（有漏洞的）行为**（子类名可经 config 通过校验）。工程师已将其正确拆分为 `test_loader_accepts_injected_custom_source` + `test_loader_rejects_unregistered_custom_source_name`。属工程师改自己的测试，合理。

## T5. ⚠️ 自查发现并修复：我 Round-1 分页测试的盲点

突变 C（`_MAX_PAGES=1`，即"只拉一页"）**未能让我原本的分页用例转红**——原用例主序列仅 600 行，小于单页上限 800，一次请求即可拿全，`len(df)>=550` 的门槛过松，**实际上没有验证到"分页装配"**。这是**我自己的测试盲点**（工程师未改动该用例）。

已修复（仅改我自己的测试文件）：
- 主序列 600 → **1200** 交易日（> 单页上限 800，强制多页）；
- 门槛 `>=550` → **`>=1150`**，并保留"无重复/严格升序/覆盖 start"断言。

修复后复测：**正确代码下 21 passed；突变 C 下该用例转红**（`assert '20220718' <= '20210104'`，即单页只拿到区间尾部、未覆盖 start）。该增强已同步至本地与远程。

## T6. 部署一致性 & 全量测试

- **逐文件哈希**：本地↔远程 `src/**` + `tests/**` + `configs/**` + `scripts/**` + `run.sh`/`run.ps1`/`pyproject.toml`/`requirements*.txt`/`README.md` 共 **100 个文件 md5 逐字节一致**（含本轮增强后的测试文件）。工程师口径为 98 个，差异仅因我的比较集合多含 `configs/*.yaml` 与 `requirements-dev.txt`，非不一致。
- **行尾/编码**：`run.sh` LF、无 BOM；`run.ps1` UTF-8+BOM（PowerShell 期望）；测试文件 LF、无 BOM。
- **全量测试**：远程 `.venv/bin/python -m pytest` → **219 passed / 1 skipped / 0 failed**（独立复跑两次一致；`1 skipped` 为环境相关的可选依赖用例）。

## T7. 遗留观察（非阻断，P3）

**`refresh` 跨复权口径变更时会把区间"缩到 `config.start..config.end`"。** 实测：000001 原为 `2012-01-02~2024-12-31`（3392 行，`adjust=none`），执行 `data refresh --adjust qfq` 后变为 `2018-01-02~2024-12-31`（1699 行，`adjust=qfq`）——**2012–2017 数据被静默丢弃**，仅有一条"缓存复权口径不一致，退化为全量重拉"的 WARNING 日志，未显式提示区间收缩。

- **评估**：口径切换时"全量重拉"本身是**正确**的（避免 qfq/hfq 混合拼接）；重拉范围取 `config.start..config.end` 也符合"refresh 刷新到配置区间"的语义。属**可用性提示不足**，非数据正确性缺陷。
- **建议（可选，不阻断交付）**：口径变更且旧缓存起点早于 `config.start` 时，追加一条 WARNING 明示"区间将由 A 收缩为 B"，或重拉范围取 `min(旧起点, config.start)..config.end` 以保留历史。

## T8. 第 3 轮最终判定

| 项 | 结果 |
|---|---|
| `--adjust`（qfq/hfq/none）功能 | ✅ 正确（归一化/映射/大小写/默认值/CLI 校验均通过） |
| hfq 与 none 价格是否真的不同 | ✅ **是**（共同交易日逐日不同；600519 首日 hfq=10530.129 / none=1685.01） |
| CLI 覆盖路径能否绕过校验 | ✅ **否**（`loader.__init__` + 源构造器纵深防御，`--adjust BOGUS`→RC=1） |
| P1-2 残余漏洞 | ✅ **彻底修复**（registry 单一真源，未注册名→`ConfigError`，注入路径保留） |
| 测试是否被削弱 | ✅ **未被削弱**（突变 A/B 反证有效；工程师改动为必要适配 + 加强） |
| 自查分页盲点 | ⚠️ 已发现并修复（增强后含突变 C 反证） |
| 部署一致性 | ✅ 100/100 文件逐字节一致 |
| 全量测试 | ✅ **219 passed / 1 skipped / 0 failed** |
| 遗留项 | P3 一项（refresh 区间收缩提示不足），不阻断 |
| **是否可交付** | ✅ **可以交付** |

**智能路由 → NoOne（本轮全部通过；仅 1 项 P3 观察项，供后续优化）**

---
---

# P3 修复确认（收尾闭环）

- **复核人**：QA 严过关
- **被验改动**：`refresh_symbol()` 口径不一致分支——重拉起点改为 `fetch_start = min(haversine...)`（实际为 `min(have_start, self.config.start or have_start)`），WARNING 显式打印区间 A~B -> C~D
- **方法**：**复现上轮原始场景 + 突变测试反证 + 内容哈希比对 + 逐文件哈希 + 全量测试 + RC 抽查**
- **结论**：**P3 已修复且真实有效；不做跨口径增量拼接（refresh 结果与独立 qfq 全量拉取内容哈希完全一致）；无回归。判定：可交付。**

## P1. 复现上轮场景（远程实跑，真实网络）

```
# 前置：000001 已为 hfq
$ data fetch --symbols 000001 --start 20120101 --end 20241231 --source tencent --adjust hfq --force
  000001: 3145 行 20120104~20241231  adjust=hfq
$ data refresh --adjust qfq
  WARNING [复权口径变更] 缓存复权 'hfq' 与请求 'qfq' 不一致，将全量重拉并重建区间:
          20120104~20241231 -> 20120104~20241231（如需固定区间请显式指定 --start/--end）: 000001
$ data info --symbols 000001
  000001: 3145 行 20120104~20241231  source=tencent  adjust=qfq   ← 区间未收缩、口径已切换
```

**对照上轮（修复前）**：同场景 refresh 后曾收缩为 `2018-01-02~2024-12-31 / 1699 行`。现为 `20120104~20241231 / 3145 行`。**已修复。** WARNING 亦显式给出 `A~B -> C~D`，不再静默。

## P2. 突变测试反证（在 `/tmp` 源码副本上进行，测后还原）

约定：`-k refresh_adjust_mismatch` 命中 3 条用例——新增 `preserves_range`、新增 `expands_when_config_start_earlier`、既有 `no_incremental_mix`。

| 突变 | 改动 | 期望 | 实测（转红者） |
|---|---|---|---|
| **M1（team-lead 指定）** | `min(...)` → `self.config.start` | 新用例转红 | ⚠️ **仅 1/2 转红**：`preserves_range` 红；`expands_when_config_start_earlier` **未红** |
| **M3（补充）** | `min(...)` → `max(...)` | 应能捕获"不扩展" | ✅ **3/3 转红**（含 `expands` 与 `no_incremental_mix`） |

**结论（需澄清 team-lead 的一处预期）**：
- 新用例**不是假断言**——`preserves_range` 在 team-lead 指定的 M1 突变下**确实转红**（断言 `src2.calls == [("20120104","20241231")]`，突变为 `20180101` 即失败）。
- 但 **M1 下 2 条用例并非都转红**：`expands_when_config_start_earlier` 在该用例中 `config.start(20120104)` 本身就是 `min`，故 `self.config.start == min(...)`，M1 与正确实现结果相同 → 该用例**无法区分** M1。
- 该 `expands` 用例**并非无用**：它针对的是"**永不扩展**"类错误（如误用 `max`/`have_start`），在 M3 突变下转红。即：**它守护的是另一类回归，而非 P3 这条 bug 本身**。
- 修复已还原（`diff` 确认与原始字节一致）。

## P3. 无跨口径增量拼接（内容哈希铁证）

refresh 后真实 parquet 与"独立全新 qfq 全量拉取"逐列内容哈希比对：

```
refresh 后  data/raw/000001.parquet : 3145 行 20120104~20241231, close 0.611~22.325
            content_hash = 55f4ebbc8325e963c1c7a814e3bf53c7
fresh 独立  data fetch --adjust qfq --start 20120101 --end 20241231 --force
            content_hash = 55f4ebbc8325e963c1c7a814e3bf53c7
=> MATCH（完全一致）
```

日期升序、无重复；`close` 全程处于 qfq 量纲（0.6~22，而非 hfq 的成百上千）→ **不存在 "hfq 老段 + qfq 新段" 混拼**。代码路径亦证实：口径不一致分支调用一次 `fetch_symbol` 后直接 `cache.write(df)`，**不走 `merge_incremental`**。

## P4. 哈希一致性 & 全量测试

- 本地↔远程 **100 个文件**（`src/**`+`tests/**`+`configs/**`+`scripts/**`+`run.sh`/`run.ps1`/`pyproject.toml`/`requirements*.txt`/`README.md`）`md5` 逐字节一致（0 差异）；工程师口径 98 个，差异仅因我的比较集合多含 `configs/*.yaml` 与 `requirements-dev.txt`，非不一致。
- 远程全量测试：**221 passed / 1 skipped / 0 failed**（与工程师申报一致，独立复跑）。

## P5. 无回归抽查

| 抽查 | 期望 | 实测 |
|---|---|---|
| `--source nosuch` | RC≠0 | ✅ RC=1 |
| `--adjust bogus` | RC≠0 | ✅ RC=1 |
| `--backend bogus` | RC≠0 | ✅ RC=1 |
| `--reward nosuch` | RC≠0 | ✅ RC=1 |
| `--adjust qfq/hfq/none` | RC=0 | ✅ 全部 RC=0 |
| `run.sh status/help/doctor/data-info` | RC=0 | ✅ 全部 RC=0 |
| `run.sh {data-fetch,data-refresh,sample-build,train,backtest,predict} --help` | RC=0 | ✅ 全部 RC=0 |

## P6. 最终判定（P3 修复确认）

| 项 | 结果 |
|---|---|
| refresh 跨口径不再静默收缩区间 | ✅ 已修复（3145 行 2012~2024 保持不变） |
| WARNING 显式化 | ✅ `A~B -> C~D` + 提示 `--start/--end` |
| 突变反证新用例有效性 | ✅ `preserves_range` 在指定突变 M1 下转红；`expands` 在 M3 下转红 |
| 无跨口径混拼 | ✅ refresh 结果与独立 qfq 全量拉取内容哈希一致 |
| 哈希一致 / 全量测试 | ✅ 100 文件一致 / **221 passed, 1 skipped, 0 failed** |
| 无回归 | ✅ 校验类 RC=1，正常类 RC=0 |
| 环境 | ✅ QA 残留已清，`data/raw` 保留 000001(qfq) 与 600519(none)，可正常使用 |
| **是否可交付** | ✅ **可以交付** |

> 澄清项（供 team-lead 知悉）：team-lead 预期"指定突变下 **2 条**新用例都转红"——实测**仅 1 条**（`preserves_range`）转红；`expands` 因用例内 `config.start` 恰为 `min` 而不区分该突变，但在反向突变 `max` 下会转红。二者非"假断言"，仅覆盖的回归面不同。


# 特征增量 PRD：A 股单股择时 PPO 系统的数据可行性分析与补充方案

> 本文是「部分工作流」**第一步**的唯一交付物：只做**数据可行性分析 + 特征增量 PRD**。
> 不写生产代码、不训练模型、不修改 `src/` 下任何文件。所有结论均经本地只读核查（证据见 §6）。
> **v2 修订（2026-11-09）**：修正成交额口径这一处结论性错误，详见 §1.1 的口径修正框、§2.1⑦ 与 §5。

## 0. 项目信息与结论摘要

| 项 | 内容 |
|---|---|
| Language | 简体中文 |
| Programming Language | 本项目为 Python 3 + pandas/numpy + Gymnasium + Stable-Baselines3(PPO) 的研究型后端，**无前端**；PRD 模板默认的 Vite+React+MUI+Tailwind 不适用，此处以实际技术栈为准 |
| Project Name | `stock_ai_feature_increment` |
| 原始需求复述 | 「我们能获取的数据有限，你分析一下，还有哪些特征数据是可以补充用于训练的？给我一个可行的方案。」 |
| 交付范围 | 数据资产盘点 → 候选特征族 → 优先级/不做清单 → 验收门槛 → 风险与待确认 |

### 0.1 一句话结论

**数据不是不够，是被丢掉了。** 现有落盘的 `data/raw_hfq/*.parquet` 中的**真实成交额 `amount`**、后复权 OHLC、265 份已核验公司行为账本、涨跌停规则、以及可零成本构造的 4859 天交易日历，共同支撑 **8 个可立即验证的新特征族**，全部属于「完全本地可派生」，**一行外部数据都不用拉**。真正需要新增数据源的（基本面/资金流/历史成分股）暂不做。换手率虽接口现成，但重拉会破坏受保护输入基线，且其核心信息已被零成本代理覆盖，建议列 P2 有条件而非 P0。

> **⚠️ 关键口径修正（决定 P0-1 成败）**：`data/raw/*.parquet` 的 `amount` 是**合成值 `close × volume`**（`sina_source.py:153`），不含任何独立信息，**禁止**用于 VWAP 偏离 / 量价背离 / Amihud，否则特征恒为常数或与 `volume_relative` 完全共线；`data/raw_hfq/*.parquet` 的 `amount` 才是真实成交额，且它正是训练特征目录（`unified.py:196`）。详见 §1.1、§2.0、§2.1、§5。

### 0.2 产品定义

**Product Goals（3 个正交目标）**
1. **打开信息空间**：在零新增外部数据源前提下，把「已有但未用 / 已有但被丢弃」的信息转成可用特征，补上现有 9 特征全是价格衍生、无成交额/无事件/无制度/无账户状态的空白（对应 `improvement_plan.md` §2 方案 D 与 §3 第 1 步）。
2. **固化一套可信契约**：为每个特征明确公式、依赖列与文件、因果口径、量纲规则与验收判据，杜绝前视泄漏与复权尺度错误。
3. **给出可执行路径与止损**：P0 可立即验证、P1 需少量工程、P2 需拍板；同时给出「什么情况下判定该特征族无效」的硬条件，避免重蹈已证伪方向。

**User Stories**
- As 架构师, I want 每个候选特征都标明依赖文件/列/公式/档位/量纲，so that 我能直接排期且不会把 hfq 后复权价、未复权成交额与 `data/raw` 的合成 amount 混用。
- As 量化研究员, I want 所有新特征在**同一时间外推台架**上与现状 9 特征基线对照，so that 我判断的是真实毛超额而不是 IC 假象或 val 单点。
- As 项目所有者, I want 方案以「几乎不新增外部数据源」为主，并明确哪些方向因已实测失败而禁止，so that 在「数据有限」约束下拿到能落地的路径而不是许愿式清单。
- As 后续维护者, I want 一份带证据索引的「不做」清单，so that 不再重复投入已验证失败的方向。

---

## 1. 数据资产盘点表（四档）

### 1.0 四档与用户三档的对应

| 用户表述 | 本文档位 | 含义 |
|---|---|---|
| 完全本地可派生 | **档 A 已有且未用** | 数据已落盘、无需联网、直接算 |
| 接口现成但需一次性重拉 | **档 B 已有但被丢弃需重拉** | 接口返回过，但落盘时被 schema 丢弃，历史缓存里没有 |
| 必须新数据源（暂不做） | **档 C 需新接口** + **档 D 不可得/禁止** | 项目内无现成适配器或数据不存在；其中「禁止」是已实测证伪 |

### 1.1 档 A：已有且未用（零新增数据，P0/P1 全部来源）

| 资产 | 路径 | 列 / 字段 | 覆盖范围 | 可用性证据（本地实测） |
|---|---|---|---|---|
| 未复权日线（**严格撮合执行价**） | `data/raw/*.parquet` | `date, open, high, low, close, volume, amount, adj_factor` | 300 只，2006-01-04~2025-12-31，单只 21~4851 行 | `open/high/low/close` 为真实未复权价，`adj_factor` 全 1.0；**但 `amount` 是合成值 `close×volume`**：5 只共 2 万余行 `amount/(volume*close)` 的 **std=0、最大偏离 0**，`|amount−close×volume|` 最大为 0，根因 `sina_source.py:153`（接口不含成交额，按 `close*volume` 估算）+ `schema.py:111-118` 缺列中性填充 → **不含独立信息，禁止用于任何金额型特征** |
| 后复权日线（**训练特征目录**） | `data/raw_hfq/*.parquet` | 同上 8 列 | 300 只，同区间，文件名与 raw 完全一致 | `close_hfq/close_raw` 比值 **26.5~159.4（均值 94.3）**；**`amount` 是真实成交额**：与 `close_raw×volume` 有自然偏离（std≈0.0095、p5=0.9848 / p50=0.9999 / p95=1.0144、最大偏离 8.2%），且 `amount/volume` 落在 raw 当日 `[low,high]` 内占比 **≥99.6%**、整数值占比高 → 未被复权缩放。**这是 `build_dataset(cache_dir)` 在严格撮合模式下的特征目录**（`unified.py:196`），也是成交额族唯一可用的 amount 源；严格撮合执行价取 `data/raw` 的未复权 OHLC（`unified.py:149, 252-264`） |
| 训练特征池 | `data/raw_hfq_v2_train/*.parquet` | 同上 8 列 | **197 只**，是 300 只子集 | `v2_train ⊆ hfq = True`，`hfq − v2_train = 103`（见 §2.10） |
| 公司行为账本 | `data/corporate_actions/*.json` | `schema, symbol, coverage, events[]`；事件含 `event_id, kind, status, record_date, ex_date, pay_date, listing_date, cash_per_share_gross, bonus_ratio, transfer_ratio, reference_method, sources, fractional_policy, listing_date_source` | **265 份全 `verified`**，3483 个 `distribution` 事件，ex_date 20060331~20251229；**197 只训练标的 100% 被覆盖** | 实测 3424 事件有现金分红、56 有送股、330 有转增；`000001.json` 有 18 事件、coverage 20060104~20251231；定义与 `require_coverage` 见 `corporate_actions.py:37-42, 253-271` |
| A 股涨跌停规则 | `src/stock_ai/env/limits.py` | `board_of / price_limit / limit_prices / is_blocked_at_open` | 板块 × 生效日期分段，纯函数无副作用 | 主板 10%、创业板 20%（20200824 起）、科创板 20%（20190722 起）、北交所 30%、主板 ST 5% 近似；**ST 名单需额外数据源，当前按非 ST 处理** |
| 日期工具 | `src/stock_ai/utils/dates.py` | `to_day_key / validate_range` | 仅日期键归一 | **无交易日历**；但 300 只 parquet 日期并集 = **4859 天**（20060104~20251231，年化 **242.9**，与真实 A 股 242~245 高度吻合）→ 可零成本构造完整日历 |
| 隐含：停牌空洞 | 由「日历并集 − 个股日期」派生 | — | 抽样 60 只：`volume=0` 与 `amount=0` 天数均为 **0**，`>10 天` 日期缺口 **404** | 停牌在本项目表现为**「缺行」而非「零成交行」**，必须靠日历差集识别 |
| 现有观测契约 | `src/stock_ai/scale/unified.py` | `CONTRACT="unified-relative-v1"`，`FEATURES` 9 个，`observation_shape=[20,9]` | 训练/val/test = 2006-2021 / 2022-2023 / 2024-2025 | `relative_features()`（`unified.py:75-107`）只用 OHLCV 的 9 个相对量；**完全不用 `amount`、不用公司行为、不用日历、不含账户状态** |

> **⚠️ 成交额口径修正（2026-11-09 复核，决定 P0-1 是否为假特征）**
> ① `data/raw` 的 `amount` = `close × volume` **合成值**（`sina_source.py:153`），5 只共 2 万余行 `amount/(volume*close)` 的 **std=0**、最大偏离 0 → **禁止**用于 VWAP 偏离 / 量价背离 / Amihud，否则特征恒为常数或与 `volume_relative` 完全共线。
> ② `data/raw_hfq` 的 `amount` 是**真实成交额**：整数值为主、与 `close×volume` 有 ±1% 左右自然偏离（p5=0.9848 / p50=0.9999 / p95=1.0144）、`amount/volume` 落在 raw 当日 `[low,high]` 内 ≥99.6%，且**它才是训练特征目录**（`unified.py:196`）→ 成交额/流动性族**仍然可行，但必须显式锁定 `data/raw_hfq` 的 amount**。
> ③ **尺度陷阱**：hfq 的 OHLC 是后复权价（累计约 105 倍），而 `amount` 是未复权真实金额 → 涉及金额/价格比值与跨期比较的特征，价格分母必须用 `data/raw` 的未复权 OHLC，或把 hfq 价按当日复权因子换算统一口径；**直接用 `close_hfq` 会引入随复权因子变化的虚假趋势**。企稳做法：优先构造**无量纲、同尺度**的相对量（`close_raw/vwap−1`、`amount/rolling_mean(amount)` 分位、`|ret|/amount` 的滚动 z-score）。
> ④ **数据前提自检断言（落地前强制）**：`std(amount/(volume*close)) > 1e-3` 且 `vwap ∈ [low, high]` 占比 > 0.99；不满足即判定该目录 amount 不可用、该特征族不可构造、**直接止损**，不得拿常数特征去跑回测（详见 §2.1⑦、§4.1）。

> 现有 9 特征：`return_1, open_relative, high_relative, low_relative, ma5_relative, ma20_relative, momentum_5, volatility_20, volume_relative`。注意 `open_relative = open/close − 1` 是**同日**开盘相对同日收盘，**不是隔夜收益**——隔夜/日内分解确实是新增信息。

### 1.2 档 B：已有但被丢弃需重拉（接口现成，历史缓存无）

| 资产 | 路径 / 证据 | 覆盖范围 | 可用性证据 |
|---|---|---|---|
| 换手率 `_turnover` | `src/stock_ai/data/sources/akshare_source.py:107-123` | 接口按标的全历史返回 | `_normalize` 已把 AKShare「换手率→`_turnover`」rename，但 `keep = [date,open,high,low,close,volume,amount]` **把 `_turnover/_amplitude/_pct_chg/_chg` 四列全部丢掉** |
| 振幅 / 涨跌幅 / 涨跌额 | 同上；`eastmoney_source.py:83, 124-138` | 东财 `fields2` 已请求 `f51~f61` 共 11 字段 | 东财 `_normalize` **只解析 `parts[0..6]`**（日期/开/收/高/低/量/额），`parts[7..10]`（振幅/涨跌幅/涨跌额/换手率）未取；`akshare_source.py` 同样把 `_amplitude/_pct_chg/_chg` 丢弃 |
| 根因（schema 层） | `src/stock_ai/data/schema.py:19-28, 90-150` | 全局 | `COLUMNS` 固定 8 列，`validate_frame` 会把其他列补齐/丢弃；`df[COLUMNS]` 强制裁剪 |
| 重拉前提与风险 | `outputs/prepare_v2_training_inputs.py:70-71`；`unified.py:244-300, 698-724` | 全局 | `data/raw` 与 `data/raw_hfq` **在受保护输入基线内，不得修改**；v2 manifest 冻结每个 source 的 `sha256` 与冻结 normalizer，续训校验会拒绝不一致 → 重拉改变字节会使已训练 v3/v4 run 不可复现、续训失败 |

> 判断见 §2.9：**技术上可行，但不进 P0/P1**。

### 1.3 档 C：需新接口（暂不做，P2）

| 资产 | 用途 | 为何暂不做 |
|---|---|---|
| 历史成分股（含退市股） | 修复幸存者偏差（`configs/symbols_hs300_full.yaml` 是抓取日为当前的静态近似 HS300，2006 年在册仅 60 只 → 2025 年 197 只） | `improvement_plan.md` §1.13 / 方案 C：是**组合化**的前置，不是单股择时特征；当前 197 只不足以支撑组合研究 |
| 基本面 / 估值 / 财报真实披露日 | PE/PB/ROE、盈利质量、财报窗口精确口径 | 需新数据源，成本最高；`improvement_plan.md` §2 方案 D 明确列为「未验证，数据源需新增」 |
| 资金流（北向 / 主力 / 大单） | 资金推动型因子 | 同上 |
| 真实历史流通股本 | 换手率的精确分母 | 接口返回的换手率是「按口径追溯计算」，跨 20 年可比性存疑 |
| ST/*ST 时点名单 | 涨跌停 5% 精确判定 | `limits.py:15-18` 明确不联网获取，当前按非 ST 近似 |
| 隐含波动率 / 期权 / 逐笔 Level-2 | 已实现 vs 隐含波动、期限结构 | 需新数据源（且部分为档 D） |

### 1.4 档 D：不可得 / 禁止（含实测证伪）

| 资产 | 判定 | 证据 |
|---|---|---|
| 盘中逐笔、真实委托簿、日内真实 VWAP、分笔明细 | **不可得**：只有日线 OHLC + `data/raw_hfq` 的汇总成交额 | `data/` 全库仅 797 个 parquet / 274 个 json / 3 个 npz / 2 个 txt，无任何分钟或逐笔文件 |
| 市场指数 / 大盘上下文**作为独立特征列** | **禁止**（数据可得但已实测有害） | `improvement_plan.md` §1.9：test 净超额 −2.73% → **−10.24%**，换手 8.5 → **26.1**，摩擦 2.54% → 7.83% |
| 市场 beta 残差特征 | **禁止** | §1.16：train 内部时间外推毛超额 **−5.98%**（现状 +0.85%），是噪声；val +3.00% 是运气 |
| regime 门控 / 联合量 / 横截面选股 | **禁止** | §1.18 / §1.9 / §1.12：正超额来自暴露度假象、test 毛超额 −4.79%、test IC t=−1.1 不显著 |
| 真实「下一个交易日 / 距下一节假日 / 距下一 ex_date」 | **禁止**：属于未来信息 | 只能用 T 及以前的滞后量（见 §2.7 / §2.8） |
| **`data/raw` 的 `amount` 用于金额型特征** | **禁止**：合成值，无独立信息 | `sina_source.py:153`；5 只 2 万余行 std=0（见 §1.1 口径修正框） |

> 全库核查结论：`data` 一级子目录仅 `corporate_actions / raw / raw_hfq / raw_hfq_v2_train / samples / scale`；`data/scale` 仅 `split.json / state/* / units.json`；按 `index / calendar / financial / fund / north / money` 搜文件名**结果为空** → 无指数、无日历、无财务、无资金流数据。

---

## 2. 候选特征族清单

### 2.0 全局规则（所有族必须遵守）

**因果性总口径**（与 `unified.py::relative_features` 一致）：特征必须在**单只股票完整时间序列内**一次性滚动计算，再按年份切 episode；**绝不串接股票计算窗口/分位数**（`unified.py:289` 注释「绝不串接股票计算指标/窗口」）。所有滚动统计只用 `T` 及以前；`normalize()` 的 mean/std 只用 train 段拟合、clip ±10，`warmup` 允许前一切分但不参与该切分动作。

**量纲 / 尺度陷阱总表（关键）**——`close_hfq/close_raw = 26.5~159.4 倍`，而 `amount` 是未复权真实金额，混用会造出约 100 倍的假信号；且 `data/raw` 的 amount 是合成值，用它会得到常数假特征：

| 特征类型 | 必须使用的价格源 | 原因 |
|---|---|---|
| 比率 / 收益型（收益、隔夜日内、影线、跳空、波动率、浮盈） | **hfq** | 复权后价格连续，除权日不会产生假跳空；比值无量纲 |
| 金额 / 绝对量型（成交额分位、Amihud、**VWAP 偏离**、股息率分母） | **分子取 `data/raw_hfq` 的 `amount`；价格分母取 `data/raw` 的未复权 OHLC**（或把 hfq 价按当日复权因子换算回未复权） | `amount_hfq` 才是真实成交额；**绝不能用 `close_hfq`**（~100 倍假偏离 + 随复权因子变化的虚假趋势），**绝不能用 `data/raw` 的合成 `amount`**（`vwap ≡ close` → 常数假特征） |
| 涨跌停价 / 一字板判定 | **raw** | `limits.py::limit_prices` 明确要求 `prev_close` 为**未复权**收盘 |
| 公司行为现金（每股税前分红、除权参考价） | **raw** | `cash_per_share_gross` 是「每原股税前现金（元）」，与未复权价同尺度 |
| 账户状态浮盈 / 持仓成本 | **hfq** | 与特征价格同源，避免除权造成的假亏损 |

**附加陷阱**：① `data/raw` 的 `amount = close×volume` 是合成值，任何「`close/vwap−1`」在它上面**恒等于 0**、与 `volume_relative` 完全共线，是假特征；② `amount_hfq` 跨 20 年、跨股票量级差异巨大（通胀、市值增长、单只 amount 均值约 1e9 元），**绝不能把绝对金额直接喂进观测**——必须转成同股票滚动分位、对数比、`|ret|/amount` 的滚动 z-score 等无量纲相对量，否则会被 `clip(±10)` 截断并主导梯度；③ `high == low`（一字板）时分母为 0 需显式兜底。

---

### 2.1 成交额 / 流动性族 ★ P0 首选

**① 构造公式 / 口径**（全部只用 `T` 及以前；**`amount` 必须取 `data/raw_hfq` 的真实成交额，`data/raw` 的合成 amount 禁用**）
- **Amihud 非流动性**：`ILLIQ_t = |ret_t| / amount_hfq_t`；`ret` 为日收益（比率，复权无关），`amount_hfq` 为真实成交额（元）。**输出用滚动 z-score 或同股票 250 日滚动分位**——`|ret|/amount` 的绝对值跨股票 / 跨期不可比，**不要直接用绝对值**。
- **成交额分位**：`amt_rank_t = pct_rank(amount_hfq_t, window=250)`（0~1）；或 `log(amount_hfq_t / rolling_median_250(amount_hfq))`。
- **VWAP 偏离**：`vwap_t = amount_hfq_t / volume_t`（**真实未复权成交均价，元/股**：`amount` 未被复权缩放、`volume` 为未复权股数）；`vwap_dev_t = close_raw_t / vwap_t − 1`（价格分母用 `data/raw` 未复权收盘，**不能用 `close_hfq`**）。可再加 `vwap_dev_ma5`。
  > ⚠️ 若误用 `data/raw` 的 `amount`，则 `vwap ≡ close_raw`、`vwap_dev ≡ 0`，是**常数假特征**，必须靠 ⑦ 的自检断言挡住。
- **量价背离**：`pv_div_t = corr(ret_k, Δlog(amount_hfq_k), window=20)`；或离散版 `1{ret_5>0} · 1{amt_rank_5 < amt_rank_20}`（价涨量缩）。

**② 依赖**：**`data/raw_hfq/*.parquet` 的 `amount, volume, open, high, low, close`**（真实成交额 + 后复权 OHLC）+ **`data/raw/*.parquet` 的 `open, high, low, close`**（未复权真实价，作 VWAP 偏离的价格分母与 ⑦ 的区间校验）；**`data/raw` 的 `amount` 禁用**。两目录按 `date` inner join（复用 `unified.py::_load_execution_bars` 的对齐语义）。

**③ 因果性**：`amount_hfq_t / volume_t / close_t` 都是 T 日收盘后已确定的当日汇总量；滚动窗口只用 `[t−W, t]`；`ret` 用 `close.pct_change`，无 bfill、无未来值。分位数用 `rolling(250).rank(pct=True)` 且 `min_periods` 合理设置，天然只回看。

**④ 档位**：**档 A 零新增数据**（`data/raw_hfq` 已落盘，且是 `unified.py:196` 的训练特征目录）。

**⑤ 经济学依据 / 对应失败点**：A 股短期最有效的因子族之一，且当前**完全缺失**（`improvement_plan.md` §2 方案 D 明确标注）。Amihud 刻画「单位成交额推动的价格冲击」，是流动性与冲击成本代理；成交额分位刻画活跃度突变；VWAP 偏离刻画日内买卖压力与收盘操纵；量价背离刻画「放量滞涨/缩量上涨」。试图解决：**9 特征全是价格衍生、连真实 `amount` 都没用**，且监督学习在同批特征上 test 年化仅 +0.32%（≈无 alpha）。同时它不引入市场上下文，规避 §1.9 的失败模式。

**⑥ 尺度陷阱**：金额分子必须取 `data/raw_hfq` 的真实 `amount`；价格分母必须取 `data/raw` 的未复权 OHLC（或按当日复权因子把 hfq 价换算回未复权），**严禁** `close_hfq / (amount_hfq/volume)`（~100 倍假偏离 + 随复权因子变化的虚假趋势）；**严禁**用 `data/raw` 的合成 `amount`（常数假特征）。绝对金额必须转无量纲相对量。

**⑦ 数据前提自检断言（落地前强制，不满足即止损）**

新增特征落盘前，必须对**实际使用的 amount 目录**逐日验算并断言：
1. `std( amount / (volume × close_raw) ) > 1e-3` —— 挡住「合成值 `close×volume`」（该值 std=0）；
2. `vwap = amount / volume` 落在 `data/raw` 当日 `[low, high]` 内的行占比 **> 0.99** —— 挡住口径错配；
3. `vwap != close_raw` 的行占比显著 > 0（等价于 ①），且 `vwap_dev` 的标准差 > 0、`vwap_dev` 与 `volume_relative` 的相关系数显著 < 1 —— 挡住常数 / 共线假特征。

> 任一断言不满足 → **判定该目录 amount 不可用、该特征族不可构造，直接止损**，不得拿常数特征去跑回测。
> 实测：`data/raw` 断言 ① = **0（失败）**；`data/raw_hfq` 断言 ① ≈ **0.0095（通过）**、② **≥99.6%（通过）**。此断言的唯一目的就是**挡住误用 `data/raw` 合成 amount**。

---

### 2.2 隔夜 / 日内收益分解 ★ P0

**① 构造**（用 `hfq` 价格）
- `overnight_t = open_t / close_{t−1} − 1`（隔夜跳空收益）
- `intraday_t = close_t / open_t − 1`（日内收益）
- `ret_1 = (1+overnight)(1+intraday) − 1`（校验：应等于现有 `return_1`，用于一致性自检）
- 派生：`overnight_ma5 / intraday_ma5`、`overnight_vol20 / intraday_vol20`、`|overnight|−|intraday|`（哪一段主导）、`overnight_share_20`。

**② 依赖**：`data/raw_hfq/*.parquet` 的 `date, open, close`。

**③ 因果性**：`open_t`、`close_t` 均为 T 日数据，`close_{t−1}` 为过去；滚动统计回看。现有 `open_relative = open_t/close_t − 1` 是**同日**口径，本族提供的是「跨日隔夜」信息，不重复。

**④ 档位**：**档 A 零新增数据**。

**⑤ 依据 / 失败点**：A 股隔夜与日内收益的驱动不同（隔夜承载隔夜信息/集合竞价/涨跌停封单，日内承载交易行为），且「隔夜跳空」与「日内反转」在 A 股有稳定的异象证据；现有 `return_1` 把两者混成一个数，模型无法区分「跳空高开后回落」与「平开高走」。试图解决：**信息被压缩丢失**，属于零成本的信息解耦，不引入任何外部变量。

**⑥ 尺度陷阱**：必须用 **hfq**，否则除权日 `open_raw/close_raw_prev` 会出现巨额假跳空（除权缺口）。比值无量纲，无需额外缩放。

---

### 2.3 K 线形态族 ★ P0

**① 构造**（全部 `hfq`，比值无量纲）
- 上影线 `upper_shadow = (high − max(open,close)) / close`
- 下影线 `lower_shadow = (min(open,close) − low) / close`
- 收盘位置 `close_location = (close − low) / (high − low)`（一字板 `high==low` 时置 0.5 并打标记）
- 实体 `body = (close − open) / close`；振幅 `bar_range = (high − low) / close`
- 跳空 `gap = open_t / close_{t−1} − 1`（与 §2.2 的 `overnight` 同源，作为形态族输入）
- 可选：`upper_shadow_ma5 / lower_shadow_ma5`、`close_location_ma5`、连续阴阳计数。

**② 依赖**：`data/raw_hfq/*.parquet` 的 `open, high, low, close`。

**③ 因果性**：只用 T 日 OHLC 与 T−1 收盘；滚动均值回看。与现有 `high_relative/low_relative`（`high/close−1`、`low/close−1`）不重复：本族刻画的是**影线相对实体/全幅的结构**，而非相对收盘的偏离。

**④ 档位**：**档 A 零新增数据**。

**⑤ 依据 / 失败点**：上下影线与收盘位置刻画日内多空博弈与「长下影支撑 / 长上影抛压」，是 A 股技术分析中信息量最集中的形态量；现有 `volatility_20` 只用收盘，完全丢掉了 `high/low` 的日内路径信息。试图解决：**OHLC 的 high/low 只被当成相对收盘的偏离，未结构化利用**。

**⑥ 尺度陷阱**：用 **hfq** 保证除权日影线/跳空不失真；`high==low` 需兜底；所有量已是比值，天然落在 ±10 clip 内。

---

### 2.4 波动率结构族 ★ P0

**① 构造**（`hfq` 价格，20 日窗口）
- **Parkinson**：`σ_P = sqrt( (1/(4·ln2·N)) · Σ (ln(H_k/L_k))² )`
- **Garman-Klass**：`σ_GK = sqrt( mean( 0.5·(ln(H/L))² − (2ln2−1)·(ln(C/O))² ) )`（负值兜底为 0）
- **上下行半方差**：`σ²_down = mean(min(ret,0)²)`、`σ²_up = mean(max(ret,0)²)`；`downside_ratio = σ_down/(σ_up+ε)`
- **跳空频率**：`gap_freq_20 = mean( |gap_k| > c·σ_P , k=t−19..t )`（c 取 1.5~2）
- 派生：`σ_P/volatility_20`（日内路径波动 vs 收盘波动之比，刻画「日内震荡但收盘不动」）。

**② 依赖**：`data/raw_hfq/*.parquet` 的 `open, high, low, close`。

**③ 因果性**：全部为 `[t−19, t]` 滚动，只用 T 及以前；`ret` 为 `close.pct_change`。

**④ 档位**：**档 A 零新增数据**。

**⑤ 依据 / 失败点**：Parkinson/GK 用高低价估计波动，效率显著高于 close-to-close 标准差（现有 `volatility_20` 是唯一波动特征）；上下行半方差区分「下跌波动」与「上涨波动」，A 股散户主导下下行波动有独立定价含义；跳空频率刻画信息到达强度。试图解决：**波动率被压缩成单一 20 日 std，结构信息全丢**。

**⑥ 尺度陷阱**：用 **hfq**（`ln(H/L)` 在除权日若用 raw 会失真）；`σ` 已是比率，无需缩放；`σ_GK` 偶发负值需 clamp。

---

### 2.5 账户状态族 ★ P0（但需契约升级，单独实验）

**① 构造**（来自环境运行时状态，**非落盘数据**）
- `position_flag ∈ {0,1}`（是否持仓）
- `unrealized_pnl = close_hfq_T / entry_price_hfq − 1`（浮动盈亏，用 hfq 同源）
- `holding_days_norm = holding_days / 250`；`cash_ratio`、`position_ratio`
- `blocked_buy_recent_5`（最近 5 日被「现金不足」阻断买入的次数，环境已能记录）

**② 依赖**：`src/stock_ai/env/trading_env.py` / `strict_env.py` / `corporate_action_env.py` 的 `state`（`info` 里已有 `position_ratio / cash / portfolio_value`，`unified.py:409`）；无需任何外部数据。

**③ 因果性**：T 日决策前，账户状态是 **T−1 日收盘后已确定**的量；不含任何 T+1 信息。

**④ 档位**：**档 A 零新增外部数据**（但属于观测契约扩展，不是「数据资产」）。

**⑤ 依据 / 失败点**：`improvement_plan.md` §1.5 观测盲区——持仓时 **68.8%** 的动作是「买入」，**59,178 次**全部被现金不足阻断；空仓时 30.6% 发卖出（必然无效）。观测 `(20,9)` 全是单股相对价格，模型不知道自己有没有持仓/成本/时长。
> **重要定性：这是纯信息缺失修复，不是 alpha。** 期望效果是消除无效动作、降低换手与摩擦，**不期待毛超额因此转正**。方案 B 已被否决，但 `improvement_plan.md` §2 方案 B 明确「持仓状态那部分（§1.5）是纯信息缺失修复，与 alpha 无关，可以单独做」。

**⑥ 尺度陷阱**：`unrealized_pnl` 用 **hfq** 价（避免除权日假亏损）；`holding_days` 需归一化。
**契约影响**：会改变 `observation_shape`，必然触发 `CONTRACT` 从 `unified-relative-v1` 升级；`unified.py:698-720` 的续训校验会拒绝 `features / observation_shape / sources / normalizer` 不一致 → **必须新 run 重训，旧 run 归档不动，不能续训**（需用户拍板，见 §5-Q3）。

---

### 2.6 A 股制度族 ☆ P1

**① 构造**
- `limit_rate_t = price_limit(symbol, T)` → 0.05 / 0.10 / 0.20 / 0.30（复用 `limits.py`）
- `at_limit_up / at_limit_down / one_word_board`：用 `limit_prices(prev_close_raw, symbol, T)` 与 **raw** 的 `open/high/low/close` 比较；一字板 `high_raw == low_raw == limit_price`
- `blocked_buy / blocked_sell`：`is_blocked_at_open(open_raw, prev_close_raw, symbol, T, side)` 直接复用
- `board_onehot`：`board_of(symbol)` → main / growth / star / bse
- `days_since_last_bar`（停牌）：个股日期 vs 交易日历并集的差集；`resume_flag = 1{days_since_last_bar>1}`（复牌首日）
- `listing_age = T − 该股首个交易日`（次新股标记）

**② 依赖**：`limits.py`（纯函数）+ `data/raw/*.parquet`（未复权 OHLC）+ 交易日历并集（§2.7）。

**③ 因果性**：`price_limit(symbol, T)` 由**代码前缀 + 已生效的历史规则日期**决定（`STAR_FROM=20190722`、`CHINEXT_FULL_FROM=20200824` 是固定的历史事实，不是未来信息）；`prev_close_raw`、`open_raw` 均为 T 及以前；停牌差集用**过去**的缺席天数。

**④ 档位**：**档 A 零新增数据**。

**⑤ 依据 / 失败点**：A 股涨跌停制度是最强的本土制度约束——一字板导致「信号发出但无法成交」，涨停后次日动量/反转、ST 5% 限制、板块 20%/30% 差异，都是海外模型没有的信息；且直接对应 `unified.py` 的 `EXECUTION_STRICT / EXECUTION_CORPORATE_ACTIONS` 撮合语义，属**自洽的信息补全**。试图解决：观测里完全没有「这笔交易能不能成交」的制度上下文，模型持续发出被阻断的无效动作（§1.5）。

**⑥ 尺度陷阱**：涨跌停价必须用 **raw**（`limits.py:129` 明确要求未复权昨收）；`board_onehot` 与 `limit_rate` 需做 one-hot / 归一化。
**已知缺陷**：ST 名单不可得（档 C），只能按非 ST 近似；新股上市首日无涨跌幅限制未建模（`limits.py:20-22`）→ 需在文档标注为近似，不可当精确标签。

**为何排 P1 而非 P0**：需要逐日 join `raw` 与 `hfq`、构造日历并集、复用撮合模块的判定，工程量与出错风险高于 §2.1~2.4；且这些是低频/稀疏信号，对「年换手 17 倍」的边际改善可能有限。

---

### 2.7 日历族 ☆ P1

**① 构造**（交易日历 = 300 只 parquet 日期并集，实测 **4859 天 / 20060104~20251231 / 年化 242.9**）
- 月内位置：`trading_day_of_month`（月内第几个交易日）、`days_from_month_start`、`is_first_3_td / is_last_3_td_of_month`（**月末判定用「月内交易日数 ≥ 月内已过天数」的历史口径，或用 `month_end_lag = 月末公历日 − T` 的确定性日历**，不用未来交易日）
- 季末：`is_quarter_end_month`（3/6/9/12 月）+ `days_from_quarter_start`
- 节假日：`long_break_just_ended = 1{T − prev_calendar_trade_day > 4}`（用**过去**缺口反推长假，覆盖春节/国庆），`days_since_long_break`。**严禁**「距下一个长假还有 N 天」（未来信息）
- 财报披露窗口：公历月份标记（4 月=年报+一季报、8 月=半年报、10 月=三季报的法定截止窗口），确定性日历、T 可知；**标注为近似**，真实披露日属档 C
- `weekday_onehot`（周一~周五）

**② 依赖**：300 只 parquet 的 `date` 列并集（零成本）+ 公历（Python 标准库）。

**③ 因果性**：所有标记只依赖 `T` 的公历日期和**T 及以前**的交易日历；`prev_calendar_trade_day` 是过去日。公历月份/星期是确定性知识，不是价格信息，不构成前视。

**④ 档位**：**档 A 零新增数据**（`dates.py` 无日历，但并集即可构造，无需外部交易日历接口）。

**⑤ 依据 / 失败点**：A 股存在显著的「月初/月末效应」「季末资金面/机构调仓」「长假前缩量、节后开门红」「财报窗口波动放大」等日历异象；这类信息在现有 9 个价格相对量里完全不存在，且是**正交于价格的新信息源**。试图解决：特征族单一、无时间与制度维度。

**⑥ 尺度陷阱**：全部是 0/1 或归一化计数，无量纲问题。
**为何排 P1**：需先落一份冻结的交易日历文件并纳入 manifest（新增一个「派生数据资产」），且要严格审阅「月末/节假日」口径是否会偷看未来交易日，审阅成本高于 P0。

---

### 2.8 公司行为族 ☆ P1（**前视风险最高，口径必须锁死**）

**① 构造**（严格 `ex_date ≤ T` 才算已发生）
- **滚动股息率**：`div_yield_250(T) = Σ cash_per_share_gross(k) / close_raw_T`，其中 `k ∈ { events : ex_date_k ∈ (T−250, T] }`。分母用 **raw 未复权价**。
- **除权事件滞后窗**：`days_since_last_ex(T) = T − max{ex_date ≤ T}`；`ex_within_20d = 1{days_since_last_ex ≤ 20}`；`ex_within_60d`
- **送转比例**：`last_bonus_transfer = bonus_ratio + transfer_ratio`（最近已发生事件）；`transfer_cum_250 = Σ(bonus_ratio+transfer_ratio)` over `ex_date ∈ (T−250, T]`
- `n_distributions_250`、`has_cash_div_250`

**② 依赖**：`data/corporate_actions/*.json` 的 `ex_date, cash_per_share_gross, bonus_ratio, transfer_ratio` + `data/raw/*.parquet` 的 `close`（股息率分母）+ 交易日历（把 `ex_date` 映射到交易日序号）。

**③ 因果性（本方案最需要强调的一条）**
> `ex_date` 是**未来日历信息**：在 T 日观测时，「下一个除权除息日还有几天」属于偷看未来。账本里虽有 `record_date`（股权登记日，早于 `ex_date`）、`pay_date`、`listing_date`，但**没有「公告日/预案日」字段**，因此无法安全地构造「公告已知但除权未发生」的窗口。

- **强制口径**：事件一律按 `ex_date ≤ T` 才算「已发生」，所有特征都是**滞后量**（`days_since_*`、过去 250 日累计），绝不出现 `days_to_next_ex` 或 `T < ex_date ≤ T+20` 的前瞻窗口。
- 若未来要做「公告效应」，必须先补公告日（档 C）并重新审阅，本轮不做。
- 覆盖保障：265 份账本全 `verified`，**197 只训练标的 100% 覆盖**（`corporate_actions.py::require_coverage` 会在缺覆盖时抛错），故在训练池上可直接使用、无缺口。
- `ex_date` 落在停牌空洞的股票已被 `prepare_v2_training_inputs.py` 整只排除（见 §2.10），因此训练池内事件日期与日线严格对齐。

**④ 档位**：**档 A 零新增数据**（账本已落盘且已人工审阅）。

**⑤ 依据 / 失败点**：A 股现金分红/送转存在明确的除权除息事件效应与「填权/贴权」预期；高送转在 A 股是长期异象；滚动股息率是价值/低波动的代理，且与纯动量特征正交。试图解决：9 特征全是价格衍生、**完全无事件/基本面代理信息**，且不引入市场指数（规避 §1.9）。

**⑥ 尺度陷阱**：`cash_per_share_gross` 是「每**原股**税前现金（元）」，与**未复权** `close` 同尺度；若误用 `close_hfq`（约为 raw 的 26.5~159.4 倍），股息率会被低估约 100 倍 → **必须用 raw**。`bonus_ratio/transfer_ratio` 是每原股比例，无量纲，无复权问题。`div_yield` 需按 train 段拟合 normalizer。

**为何排 P1**：需把 JSON 事件按 `ex_date` 逐日 join 到 197 只 × 4859 天的特征表，且口径审阅（前视、税前/税后、送转股份上市日）成本最高；信号频率低（每年 1~2 次），对短期择时的边际可能有限。

---

### 2.9 判断一：换手率「一次性重拉」

**事实**：AKShare `stock_zh_a_hist` 与东财 klines 接口**现成返回**换手率/振幅/涨跌幅/涨跌额，被 `akshare_source.py:121` 的 `keep` 与 `eastmoney_source.py:128-138` 的 `parts[0..6]` 丢弃，再经 `schema.py:132` 的 `df[COLUMNS]` 裁剪。历史缓存中确实没有，需一次性重拉 300 只全历史才能补齐。

**但判定：不进 P0/P1，列 P2 有条件。四条硬理由**

1. **破坏受保护基线**：`prepare_v2_training_inputs.py:70-71` 明确「`data/raw` 与 `data/raw_hfq` 均在受保护输入基线内，不得修改」。
2. **破坏冻结契约**：`unified.py:244-300, 698-724` 冻结每个 source 的 `sha256` 与 normalizer；重拉改变文件字节 → 已训练 v3/v4 run 不可复现，续训会被 `续训拒绝：...不一致` 拦下。
3. **口径不可比**：`configs/symbols_hs300_full.yaml` 是抓取日为当前的**幸存者名单**；按当前日重拉可能与旧缓存行数/日期不一致 → 触发 v2「特征/执行日期不齐」fail-closed 整只排除。且换手率 = 成交量/流通股本，接口返回的是按当前口径**追溯计算**的值，跨 20 年（含大量送转）可比性存疑；真实历史流通股本属档 C。
4. **边际价值低**：换手率想提供的核心信息是「**相对流动性冲击**」，而这已被**零成本代理完整覆盖**——现有 `volume_relative`（已有）+ §2.1 的成交额滚动分位 / Amihud / 量价背离（基于 `data/raw_hfq` 的真实 amount），全部是**尺度无关的相对量**，不需要真实流通股本。

**若用户坚持的最小可行路径（不写实现，仅路径判断）**：新建**独立目录** `data/raw_turnover/`，**不覆盖**受保护基线；只拉 197 只训练标的；生成独立的 manifest 版本号与 normalizer；旧 run 归档不动、不续训；先在一只样本上核对换手率与 `volume_relative` 的相关性，若高度冗余则直接放弃。**启动前必须用户拍板（§5-Q1）**。

---

### 2.10 判断二：103 只未入训练池样本

**精确分解（实测 + `prepare_v2_training_inputs.py:64-112`）**

```
300 只 (data/raw_hfq)
 ├─ 197 只  → data/raw_hfq_v2_train（实际训练池）
 └─ 103 只未入池
     ├─  35 只：无公司行为账本（候选 = hfq ∩ raw ∩ 账本；300 − 265 = 35）
     └─  68 只：有账本但被 fail-closed 排除，三类原因
          ① 特征/执行(hfq vs raw)日期不齐
          ② 账本事件 record_date/ex_date 落在停牌空洞（无日线）
          ③ 非整数送转比例产生碎股（契约只支持 reject）
```
佐证：账本 265 = 197（训练）+ 68（非训练），实测 `tr ⊆ ca = True`、`tr − ca = 0`；`data/raw_hfq/inconsistent.txt` 记录了 `000408` 这一例不一致。

**四条判断**

1. **性质不同**：这 103 只是「**可补充的训练样本**」，不是「**可补充的特征数据**」。本次任务问的是特征，两者不能混为一谈。
2. **不能零成本并入**：35 只无账本 → v2 `require_coverage`（`corporate_actions.py:253-271`）直接抛错，v2 禁止静默丢弃股票；68 只是数据质量 fail-closed，强行并入会破坏严格对齐契约与 manifest 快照。
3. **扩池不解决幸存者偏差**：股票池是「抓取日为当前日」的静态近似 HS300，2006 年在册仅 60 只 → 2025 年 197 只，全是「活到今天的赢家」。扩到 300 只只是**增加样本量**，仍是同一批幸存者；`improvement_plan.md` §1.13 明确「任何组合层面的回测都必须用时点成分股（含退市股）才有意义」，那是档 C 的数据工程。
4. **P0 阶段必须冻结 197 只**：特征族验收要求**单变量对照**（只变特征集、不变样本），同时改样本 + 改特征无法归因；这是方法论要求，不是技术限制。

**建议**：**P2 / 暂不做**。待 P0 特征族验证出正毛超额后，单独开一个「扩池」实验（需先补 35 只账本 + 处理 68 只 fail-closed，属数据工程）；并且优先级应放在「补含退市股的历史成分股（档 C）」而非简单加样本。

---

## 3. 优先级与「不做」清单

### 3.1 P0 — 零新增数据、可立即验证（先做，且严格单变量逐个验）

| 编号 | 特征族 | 档 | 排序理由 |
|---|---|---|---|
| P0-1 | **成交额/流动性族**（Amihud、成交额分位、VWAP 偏离、量价背离） | A | `improvement_plan.md` §2 方案 D 明确列为「A 股短期最有效因子族之一，且**完全缺失**」，§3 第 1 步排第一位；**`data/raw_hfq` 的真实 `amount` 已落盘且经 §2.1⑦ 口径核验通过**，边际最高、成本最低。⚠️ **`data/raw` 的合成 amount 禁用**，落地前第一件事是跑 §2.1⑦ 数据前提自检断言 |
| P0-2 | **隔夜/日内收益分解** | A | §2 方案 D / §3 第 1 步第 ② 项；现有 `return_1` 混两者，纯信息解耦，零风险 |
| P0-3 | **K 线形态族** | A | 同一 OHLCV 重新参数化，`high/low` 结构信息此前完全未用；成本最低 |
| P0-4 | **波动率结构族** | A | §3 第 1 步第 ③ 项；现有只有单个 `volatility_20`，Path 信息全丢 |
| P0-5 | **账户状态族**（单独实验） | A（运行时） | §1.5 观测盲区，是纯信息缺失修复；**单独做**是因为它会触发契约升级与重训，不能与前四项混在同一对照里 |

**P0 共同理由**：全部「已有且未用」，**一行外部数据都不用拉**，可用现成台架立即做时间外推检验；且不引入任何市场上下文或 beta 残差，规避 §1.9 / §1.16 的失败模式。

### 3.2 P1 — 零新增数据，但需额外工程 / 口径审阅

| 编号 | 特征族 | 档 | 为何不是 P0 |
|---|---|---|---|
| P1-1 | A 股制度族（涨跌停/一字板/停牌/板块生效日） | A | 需逐日 join raw 与 hfq、构造日历、复用撮合判定；低频稀疏信号 |
| P1-2 | 日历族（月初月末/季末/长假滞后/财报法定窗口） | A | 需落一份冻结交易日历并纳入 manifest；「月末/节假日」口径需严格审阅防前视 |
| P1-3 | 公司行为族（滚动股息率/除权滞后窗/送转比例） | A | 需事件按 `ex_date` 逐日 join；**前视口径审阅成本最高**；信号频率低 |

### 3.3 P2 — 需重拉 / 需新数据源 / 需用户拍板

- **P2-1** 换手率（含振幅/涨跌幅/涨跌额）一次性重拉 —— 条件严格，见 §2.9。
- **P2-2** 扩训练池 103 只 —— 见 §2.10，需先补账本/处理 fail-closed。
- **P2-3** 历史成分股（含退市股）—— 档 C，解决幸存者偏差，是组合化的前置（`improvement_plan.md` 方案 C）。
- **P2-4** 基本面/估值/资金流/真实流通股本/ST 名单/隐含波动率 —— 档 C，成本最高但天花板最高；**仅在 P0/P1 全部失败后才启动**。

### 3.4 明确的「不做」清单（附实测证据）

| # | 禁止方向 | 证据（`outputs/improvement_plan.md`） |
|---|---|---|
| 1 | **市场指数 / 大盘上下文作为独立特征列** | §1.9：test 净超额 **−2.73% → −10.24%**，换手 **8.5 → 26.1**，摩擦 2.54% → 7.83%；毛超额本来就 ≈0，市场特征只给了拟合噪声的自由度 |
| 2 | **市场 beta 残差特征** | §1.16：train 内部时间外推毛超额 **−5.98%**（现状 **+0.85%**），val +3.00% 纯属运气（只落在 2022/2023 两年）→ 残差本身就是噪声 |
| 3 | **regime 门控规则** | §1.18：test 正超额 **100% 来自暴露度项**（+0.47%），技能项 **−0.37%**；是「牛市里低暴露」的假象 |
| 4 | **联合量（regime × 个股位置）** | §1.9 / §1.18：`validate_joint_signal.py` test 毛超额 **−4.79%** |
| 5 | **横截面选股** | §1.12：test 每次超额 K=5/10/20/40 **全负**；test 日均 IC t=**−1.1 不显著**；train 正超额在 val/test 崩塌 |
| 6 | **加大训练量 / 调参重训** | §2 方案 A：v4 900 万步，技能项 **−0.59% → −2.04%**，交易笔数 15→23，成本拖累 71→115 bps，是「学噪声」 |
| 7 | **单靠降换手 / 降摩擦** | §1.11 / §1.14：年换手 17 倍需毛边际 ≥ **5.1%** 才打平；调仓间隔 1→20 后 test 毛超额 −0.19% → **−7.36%**。**降换手只能省摩擦，不能创造边际** |
| 8 | **把「IC 稳定」或「val 单点」当判据** | §1.10 / §1.16 / §4 误区 2：IC 稳定的是「个股相对市场残差」的秩相关，不是绝对收益；监督学习 test 年化仅 **+0.32%** |
| 9 | **用「全部超额」（含零交易 episode）评估** | §4 误区 1 + `unified.py:49` `DEGENERATE_ZERO_TRADE_FRACTION=0.05`：零交易超额恒等于 `−benchmark_return`，是定义性假象 |
| 10 | **用 `days_to_next_ex` / 「距下一节假日」/ 任何未来日历信息** | 前视泄漏，见 §2.7 / §2.8 |
| 11 | **用 `data/raw` 的 `amount` 构造任何金额型特征** | 合成值 `close×volume`（`sina_source.py:153`，std=0）→ `vwap ≡ close`、`vwap_dev ≡ 0`，是常数假特征；见 §1.1、§2.1⑦ |

> 补充：早期 **2006–2007 年的数字不作依据**（§1.17：在册仅 60 只、2007 极端牛市，现状组 2006 −34.45% / 2007 −87.24%）。

---

## 4. 验收指标与门槛

### 4.1 沿用台架（不得另起炉灶）

| 台架 | 作用 |
|---|---|
| `outputs/timing_backtest.py` | `TimingBacktest`，**所有「策略是否有效」结论的唯一度量口径**；严格因果（T 收盘信号 → T+1 生效）；返回 `net_excess / gross_excess / switches / exposure / n_episodes` |
| `outputs/timing_backtest.py::self_check` | **三项强制自检**（每次实验前必须 `all_passed`）：① 常数满仓毛超额 ≈0（\|gross\|<1e-9）② 作弊信号毛超额 > +5% ③ 随机信号毛超额 < 0（不是 ≈0） |
| **§2.1⑦ 数据前提自检断言（新增，落地前强制）** | `std( amount / (volume × close_raw) ) > 1e-3` 且 `vwap ∈ [low, high]` 占比 > 0.99；`vwap_dev` 标准差 > 0 且与 `volume_relative` 不共线。不满足即判定该目录 amount 不可用、该特征族不可构造、**直接止损**。此断言专门用于**挡住误用 `data/raw` 合成 amount** |
| `outputs/verify_unified_backtest.py` | 统一契约下的净/毛超额 + 换手 + 在市对照 |
| `outputs/validate_residual_lowturnover.py` | train 内部时间外推检验（含上述自检） |
| `outputs/diagnose_residual_overfit_vs_regime.py:203-229` | 时间外推参考实现：拟合 train 前 70%（2006~2016）→ 评估后 30%（2017~2021），`rebalance=20`，`TimingBacktest.run_probability` |
| `outputs/diagnose_supervised_baseline.py` | 监督学习基线（logreg / MLP），用于「特征有没有线性/非线性信号」的辅助诊断，**不作验收判据** |

### 4.2 主判据（唯一决定「有效 / 无效」）

> **训练段内部时间外推的毛超额 `gross_excess` 必须为正，且高于现状 9 特征基线。**

- 切分：拟合 train 前 70% 年份 → 评估 train 后 30% 年份（参考 `diagnose_residual_overfit_vs_regime.py:205-209`）；`rebalance=20`，摩擦沿用 0.3% 往返。
- **基线数值（必须同时复现作为对照）**：现状 9 特征 **+0.85%**；已失败的残差特征 **−5.98%**（`improvement_plan.md` §1.16）。
- 门槛：新增族评估段 `gross_excess > 0` **且** `> 现状基线 +0.85%`；建议要求**多个切点 / 多个窗口长度下都为正**，避免单点运气（§1.16 的教训就是 val 单点骗人）。
- **实验设计**：单变量、只变特征集；冻结 197 只训练池、时间切分、随机种子、摩擦参数、normalizer 拟合口径；每次必须同时报告「现状 9 特征」在同一评估段的数值。
- **前置条件**：金额型特征（P0-1）必须先通过 §2.1⑦ 数据前提自检断言，否则不得进入本判据评估。

### 4.3 辅判据（缺一不可，必须同时报告）

| 指标 | 要求 | 依据 |
|---|---|---|
| 净超额 `net_excess` | 与毛超额同时报告；毛正净负须说明是否被摩擦吃光 | §4 误区 3 |
| 年换手 `switches` | 不得显著高于现状基线（§1.9 的 8.5 → 26.1 是恶化信号）；若毛超额为正但换手 > 1.5× 现状，先怀疑拟合噪声 | §1.9 |
| 摩擦拖累 | 明确报告；**17 倍年换手需毛边际 ≥ 5.1% 才打平**；「毛超额 < 摩擦拖累」直接判无效 | §1.11 |
| 在市占比 `exposure` | 报告，用于识别「低暴露在牛市躺赢」的假象 | §1.18 |
| **择时技能项** | 需 > 0 且高于恒定满仓（`diagnose_gated_rule_skill.py` 口径） | §1.4 / §1.18 |
| **退化守卫** | 零交易 episode 占比 ≤ **5%**（`DEGENERATE_ZERO_TRADE_FRACTION=0.05`）；只看 `mean_excess_return_traded` | §4 误区 1 |
| **数据前提自检** | §2.1⑦ 三条断言全通过；未通过则本次实验结果**不可采信**（常数 / 共线假特征） | 本 PRD §2.1⑦ |
| 台架自检 | `self_check()` 三项全通过，否则该次实验结果**不可采信** | `timing_backtest.py:180-217` |
| IC / 监督学习 | **只作诊断，绝不作验收判据** | §1.10 / §4 误区 2 |

### 4.4 止损条件（满足任一条即判定该特征族无效，停止投入）

1. 时间外推评估段 `gross_excess ≤ 0` 或 `≤ 现状基线 +0.85%` → 无效。
2. 毛超额为正但换手 > 1.5× 现状且净超额 < 0 → 判定为**拟合噪声**（§1.9 模式），无效。
3. 逐年分解显示正超额集中在个别年份（如仅 2022/2023）→ **过拟合**（§1.16 H_overfit 模式），无效。
4. 只在 val / test 单段为正、时间外推为负 → 无效（§1.16 明确 val 单点会骗人）。
5. 择时技能项 ≤ 0，或零交易占比 > 5% → 退化，无效。
6. **即使 IC 稳定（时序或横截面），只要毛超额不达标即无效**（§1.10「IC 稳定 ≠ 有可执行 alpha」）。
7. 2006–2007 段的数字不作为「有效」依据（§1.17）。
8. **§2.1⑦ 数据前提自检任一断言不通过**（例如误用 `data/raw` 的合成 `amount` 导致 `vwap_dev ≡ 0`、或 `vwap` 落在 `[low,high]` 占比不达标）→ 判定该 amount 目录不可用、该特征族**不可构造**，**直接止损，不进入回测**。

### 4.5 特别提醒：不要重蹈两个覆辙

- **「IC 稳定 ≠ 可交易」**：§1.10 中稳定的 IC 是「个股相对市场残差」的秩相关，而单股择时赚的是**绝对收益**；从它构造的交易信号不可靠（§1.16 已让残差路线被否）。IC 只能用来诊断，不能用来验收。
- **「val 单点骗人」**：残差特征在 val 毛超额 +3.00%，看似有效，但 train 内部时间外推 −5.98%，且逐年分解显示正超额只落在 2022/2023 两年 → **主判据必须是 train 内部时间外推**，val/test 只作参考。
- 另：不要用「加了 X 特征后模型表现变了」推断「X 有用」——§1.9 加了市场特征后超额从 −2.73% 变到 −10.24%，**变化很大但方向是负的**。
- **新增第三个覆辙（数据口径）**：不要把「落盘有 `amount`」等同于「有真实成交额」。`data/raw` 的 `amount` 是 `close×volume` 合成值，用它构造的 VWAP 偏离恒为 0 —— 必须先做 §2.1⑦ 断言再谈回测。

---

## 5. 风险与待确认问题（需用户拍板）

| # | 待拍板问题 | 影响 | 建议 |
|---|---|---|---|
| **Q1** | 是否允许**一次性重拉 300 只全历史换手率**（含振幅/涨跌幅/涨跌额）？ | 破坏受保护输入基线；v3/v4 run 的 `sha256` / normalizer 契约失效、不可续训；幸存者名单口径可能不齐触发 fail-closed | **否**。先用 §2.1 零成本代理；若坚持，走「独立目录 + 独立 manifest + 只拉 197 只」的最小路径 |
| **Q2** | 是否允许**扩展训练池至 300 只**（并入 103 只）？ | 需补 35 只账本 + 处理 68 只 fail-closed；且 P0 阶段改样本会污染单变量对照 | **P0 阶段否（冻结 197 只）**；后续单独开「扩池」实验 |
| **Q3** | 是否接受**特征契约版本升级**（`unified-relative-v1` → v2）导致 `observation_shape` 变化、必须**新 run 重训、旧 run 不能续训**？ | `unified.py:698-720` 会拒绝不一致的 manifest/normalizer；这是所有新增特征（尤其账户状态族）的必然结果 | **接受**，但新 run 独立、旧归档不动；不覆盖 `data/raw_hfq_v2_train/` |
| **Q4** | 是否接受**新增特征用独立特征目录 + 新 manifest**，不改动现有 `data/raw*` / `data/raw_hfq_v2_train/`？ | 保护基线、保证旧 run 可复现；代价是多一份派生数据 | **是** |
| **Q5** | 是否需要**补齐含退市股的历史成分股**（档 C 新数据源）以解决幸存者偏差？ | 是组合化的前置（`improvement_plan.md` 方案 C），但不是单股择时特征 | **暂不做**，等 P0 出信号后再评估 |
| **Q6** | 公司行为口径：是否接受**一律按 `ex_date ≤ T` 才算已发生**的保守口径（放弃「公告已知但除权未发生」窗口，因账本无公告日字段）？ | 决定 §2.8 全部特征是否有前视；若不接受则需先补公告日（档 C） | **是**（这是唯一无前视口径） |
| **Q7** | 是否接受用**公历月份构造财报披露窗口**（法定截止日近似，非真实披露日）？ | 日历族的一个子项；真实披露日需档 C | **是**，但必须在文档/代码中标注为近似 |
| **Q8** | ST 名单不可得，是否接受**按非 ST 近似**（`limits.py` 现状）？ | 涨跌停 5% 判定存在已知偏差 | **是**，标注为已知缺陷，不当精确标签 |
| **Q9** | 是否同意**主判据 = 时间外推毛超额 > 现状基线 +0.85%** 及 §4.4 的全部止损条件（含新增第 8 条数据前提自检）？ | 决定后续所有实验的验收口径 | **是**（这是 `improvement_plan.md` §3 已确立的方法论铁律） |
| **Q10** | 摩擦参数是否沿用 **0.3% 往返、`rebalance=20` 对照**？ | 影响毛/净超额与换手门槛的可比性 | **是**，保持与历史结论可比 |

**其他已知风险（不需拍板但需登记）**

- **⚠️ 成交额口径（最高优先级，决定 P0-1 真假）**：`data/raw/*.parquet` 的 `amount` 是 `sina_source.py:153` 生成的**合成值 `close×volume`**（5 只共 2 万余行 `amount/(volume*close)` 的 std=0、最大偏离 0、`|amount−close×volume|`=0），**不含独立信息，所有金额型特征一律禁用**；**`data/raw_hfq/*.parquet` 的 `amount` 才是真实成交额**（`std(amount/(volume*close_raw))≈0.0095`、p5/p95=0.9848/1.0144、`vwap` 落在 raw `[low,high]` 内 ≥99.6%），且它是 `unified.py:196` 的训练特征目录 → **金额型特征的分子必须统一以 `data/raw_hfq` 为准，价格分母用 `data/raw` 的未复权 OHLC（或按当日复权因子换算），绝不能用 `close_hfq`**。落地前强制跑 §2.1⑦ 断言。**v1 原稿在此处给出的「以 raw 目录为金额源」建议为错误结论，v2 已彻底反转，请勿沿用。**
- 股票池幸存者偏差：2006 年在册仅 60 只 → 2025 年 197 只，**任何组合层面的结论都必须先补时点成分股**（§1.13）。
- 停牌在本项目是「缺行」而非「零成交行」，停牌标记必须由日历差集派生；单只最少仅 21 个交易日（次新股/长期停牌），滚动窗口需 `min_periods` 兜底。
- 新股上市首日无涨跌幅限制、个人差别化股息税未建模（`limits.py:20-22`、`unified.py:480-491`），属既有近似，本方案沿用不扩大。
- `data/raw_hfq` 的 `amount` 与 `data/raw` 的 `close×volume` 相关系数 0.99974、相对差均值 0.65%（最大 8.2%），这不是抓取批次差异，而是**真实成交额 VWAP 相对收盘价的自然偏离**（p5/p95=0.9848/1.0144），正是 VWAP 偏离特征可用的依据。

---

## 6. 附录：核查证据索引

本文所有数字均来自本地只读核查（未联网、未写 `src/`、未训练）：

| 结论 | 核查方式 |
|---|---|
| 300/300/197 只、8 列、区间、`amount>0` 1.0、`adj_factor` 全 1.0 | `.venv/Scripts/python.exe` 读 `data/raw`、`data/raw_hfq`、`data/raw_hfq_v2_train` 首个 parquet + 文件名集合 |
| **`data/raw` 的 `amount` 是合成值 `close×volume`**：5 只（000001/000002/600519/000063/300750）共 2 万余行 `amount/(volume*close)` 的 std=0、最大偏离 0；`|amount−close×volume|` 最大为 0；根因 `sina_source.py:153` + `schema.py:111-118` 缺列填充 | `.venv/Scripts/python.exe` 逐日验算 + `src/stock_ai/data/sources/sina_source.py:143-155` |
| **`data/raw_hfq` 的 `amount` 是真实成交额**：`amount/(volume*close_raw)` std≈0.0095、p5=0.9848 / p50=0.9999 / p95=1.0144、最大偏离 8.2%；`vwap=amount/volume` 落在 raw 当日 `[low,high]` 内占比 ≥99.6%（000001=0.9996、300750=0.9962） | 同上，两目录按 `date` merge 计算 |
| hfq/raw 收盘比 26.5~159.4（均值 94.3）→ 金额型特征严禁用 `close_hfq` | 读 `000001.parquet` 两目录 merge |
| `data/raw_hfq` 是训练特征目录、`data/raw` 是严格撮合执行目录；`build_dataset` 对受保护输入全程 `read_bytes()` 只读，manifest 只写入新 run 目录 | `src/stock_ai/scale/unified.py:149, 196, 232-300, 640-689, 698-724` |
| 停牌 = 缺行、抽样 60 只 `volume=0`/`amount=0` 均为 0、`>10 天` 缺口 404 | 抽样 60 只统计 |
| 交易日历并集 4859 天 / 年化 242.9 / 单只 21~4851 行 | 300 只 `date` 列并集 |
| 公司行为 265 份全 verified、3483 事件、3424 现金 / 56 送 / 330 转、ex_date 20060331~20251229、197 训练标的 100% 覆盖 | 遍历 `data/corporate_actions/*.json` + 集合运算 |
| 103 = 35（无账本）+ 68（fail-closed） | 集合运算 + `outputs/prepare_v2_training_inputs.py:64-112` |
| 9 特征、`CONTRACT`、观测 `(20,9)`、续训校验 | `src/stock_ai/scale/unified.py:26-29, 75-107, 698-724` |
| 换手率/振幅/涨跌幅/涨跌额被丢弃 | `src/stock_ai/data/sources/akshare_source.py:107-138`、`eastmoney_source.py:83, 124-138`、`src/stock_ai/data/schema.py:19-28, 132` |
| 公司行为字段与 `require_coverage` | `src/stock_ai/data/corporate_actions.py:37-42, 253-271` |
| 涨跌停分段与 `is_blocked_at_open` | `src/stock_ai/env/limits.py:43-56, 99-174` |
| 无交易日历 / 无指数/财务/资金流数据 | `src/stock_ai/utils/dates.py` 全文；`data/` 全目录文件类型统计 + 关键词文件名搜索（结果为空） |
| 台架自检三项、时间外推 70/30、`gross_excess` 口径 | `outputs/timing_backtest.py:33-51, 180-239`；`outputs/diagnose_residual_overfit_vs_regime.py:203-229` |
| 全部证伪结论 | `outputs/improvement_plan.md` §1.9 / §1.10 / §1.11 / §1.12 / §1.13 / §1.14 / §1.16 / §1.17 / §1.18 / §2 / §3 / §4 |

---

**下一步交接建议（给架构师）**：按 §3.1 的 P0-1 → P0-2 → P0-3 → P0-4 顺序，每个族**单独**在 `TimingBacktest` 上做 §4.2 的时间外推对照；P0-5（账户状态族）因触发契约升级，独立排期并先走 Q3/Q4 拍板。P1 与 P2 在 P0 出结论前不启动。**P0-1 落地前第一件事是跑 §2.1⑦ 的数据前提自检断言，并把金额分子锁定在 `data/raw_hfq` 的 `amount`；`data/raw` 的合成 amount 一律禁用。**

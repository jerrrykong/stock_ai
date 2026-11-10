"""Build an offline self-contained report from the read-only diagnostic artifact."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
source = json.loads((OUT / "unified_exposure_diagnosis_20261016.json").read_text(encoding="utf-8"))
payload = {key: source[key] for key in ("generated_at", "scope", "checks", "models", "summaries", "baseline_summaries", "corporate_action_audit", "episodes")}
payload["episodes"] = [{key: row[key] for key in ("run_id", "episode_id", "symbol", "year", "split", "bars", "average_position", "holding_days", "holding_fraction", "longest_holding_days", "total_return", "baseline_return", "excess_return", "max_drawdown", "trades", "actions", "mean_probabilities", "mean_policy_entropy")} for row in payload["episodes"]]
payload["examples"] = [row for row in payload["episodes"] if row["run_id"] == "20261015_095549_25eab142" and (row["symbol"], row["year"]) in {("000001", 2024), ("600519", 2022), ("600519", 2024), ("300750", 2023)}]
vendor = (ROOT / "src" / "stock_ai" / "scale" / "ui" / "static" / "vendor" / "echarts.min.js").read_text(encoding="utf-8")
if re.search(r"</script", vendor, flags=re.I):
    raise ValueError("Unexpected closing script tag in chart library")

html = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>综合 PPO 高持仓诊断｜2026-10-16</title>
<style>
:root{color-scheme:light;--ink:#192335;--muted:#606d80;--line:#dce3ee;--paper:#f4f6fa;--accent:#365fc5;--up:#ba3444;--down:#17815b;--warn:#9b5b16}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1240px;margin:0 auto;padding:38px 28px 60px}h1{font-size:34px;line-height:1.32;margin:10px 0 16px;letter-spacing:-.7px}h2{font-size:23px;margin:0 0 13px}h3{font-size:17px;margin:0 0 8px}p{margin:8px 0}.eyebrow{font-size:12px;letter-spacing:1.6px;color:var(--accent);font-weight:700}.sub,.note,.source{color:var(--muted)}.sub{max-width:1020px}.note,.source{font-size:12px}.source{margin-top:10px}.tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.tag{border:1px solid var(--line);border-radius:99px;padding:4px 12px;background:#fff;font-size:12px}.callout{padding:20px 24px;margin:24px 0;border:1px solid #edd6b9;border-left:5px solid #ba7425;background:#fff9ef;border-radius:10px}.callout strong{color:#81511b}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{padding:18px;background:#fff;border:1px solid var(--line);border-radius:12px}.card .label{color:var(--muted);font-size:12px}.number{font-size:33px;font-weight:700;line-height:1.4;letter-spacing:-.6px}.card .detail{font-size:12px;color:var(--muted)}section{padding:25px;background:#fff;border:1px solid var(--line);border-radius:14px;margin:20px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.two section{margin:0}.chart{height:340px;width:100%}.chart.short{height:300px}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}.toolbar .spacer{flex:1}button,select,input{font:inherit;font-size:13px;border:1px solid #cbd5e4;border-radius:7px;padding:7px 11px;background:#fff;color:var(--ink)}button{cursor:pointer}button:hover{background:#eef3ff}button.active{border-color:var(--accent);background:#edf2ff;color:#244fac}button:disabled{opacity:.45;cursor:default}input{min-width:160px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #e7ecf4;white-space:nowrap}th{color:#536076;background:#f7f9fc;font-size:12px}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#f8faff}table.compact th,table.compact td{padding:7px 9px;font-size:12px}.positive{color:var(--up)}.negative{color:var(--down)}.neutral{color:var(--ink)}.badge{display:inline-block;font-size:11px;padding:2px 7px;border-radius:4px;background:#edf2ff;color:#33579d;margin-bottom:6px}.badge.critical{background:#fff0ed;color:#a54528}.evidence{display:grid;grid-template-columns:30px 1fr;gap:12px;padding:16px 0;border-top:1px solid var(--line)}.evidence:first-child{border:0}.index{font-size:18px;font-weight:700;color:var(--accent)}code{font:12px/1.6 Consolas,monospace;background:#f0f3f8;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}pre{padding:15px;background:#f6f8fc;border:1px solid var(--line);border-radius:8px;font:13px/1.8 Consolas,"Microsoft YaHei",monospace;white-space:pre-wrap;overflow-wrap:anywhere}.steps{counter-reset:step;list-style:none;padding:0}.steps li{position:relative;padding:0 0 20px 40px}.steps li:before{counter-increment:step;content:counter(step);position:absolute;left:0;top:2px;border-radius:50%;width:25px;height:25px;text-align:center;line-height:25px;background:#edf2ff;color:var(--accent);font-size:12px;font-weight:700}.steps p{font-size:14px}.foot{font-size:12px;color:var(--muted);padding-top:18px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.warninline{background:#fff8ed;border-radius:7px;padding:10px 13px;color:#8a5a22;font-size:13px}.formula{font:14px/1.9 Consolas,"Microsoft YaHei",monospace}.hidden{display:none!important}.status{font-size:12px;color:var(--muted)}footer{text-align:center;color:var(--muted);font-size:12px;margin-top:28px}.pillgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:18px 0}.pillgrid>div{padding:16px;background:#f7f9fd;border-radius:9px}.pillgrid strong{display:block;font-size:20px}.blocktext{font-size:14px}details{margin:12px 0}summary{cursor:pointer;color:#3c5c95}ul{padding-left:20px}#renderStatus{display:none;color:#a33;padding:12px;border:1px solid #d99}@media(max-width:850px){main{padding:24px 14px}h1{font-size:27px}.cards{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}.pillgrid{grid-template-columns:1fr}section{padding:18px}.chart{height:320px}}@media print{body{background:#fff}main{padding:0;max-width:none}section,.card{break-inside:avoid}.toolbar{display:none}a{color:var(--ink)}}
</style>
</head>
<body>
<main>
<header>
<div class="eyebrow">MODEL DIAGNOSTICS / UNIFIED PPO</div>
<h1>高仓位已确认；<br>收益真实性尚未过关。</h1>
<p class="sub">最新模型不是被卖单“卡住”，而是很少决定退出。全量验证显示持仓偏置明显，同时发现公司行为未记账这一基础缺陷：现有奖励与收益可能把除权当作亏损。下一步应先修环境和状态设计，而不是直接追加训练。</p>
<div class="tags"><span class="tag">只读诊断 · 未重训</span><span class="tag">300 只股票 · 2022–2025</span><span class="tag">每模型 1,188 个年度区间</span><span class="tag">本地缓存 / 冻结模型</span><span class="tag">诊断日期 2026-10-16</span></div>
</header>
<div class="callout"><strong>先明确判断边界：</strong>高持仓并不自动意味着策略差，正收益也不证明有择时能力。本次可以可靠确认的是“在当前模拟环境中，策略长期近满仓、与市场高度同步”；不能据此宣称真实投资回报或正式样本外泛化。公司行为修复前，所有收益和回撤都只视作现环境复现值。</div>
<div class="cards">
<div class="card"><div class="label">最新模型 · 平均收盘仓位</div><div class="number">91.21%</div><div class="detail">先逐日均值，再对股票×年度等权</div></div>
<div class="card"><div class="label">最新模型 · 持仓日占比</div><div class="number">91.70%</div><div class="detail">全部 285,283 根日线合并统计</div></div>
<div class="card"><div class="label">模拟下跌分组 · 平均仓位</div><div class="number">93.29%</div><div class="detail">恒买基准年收益＜−5%，共 484 段</div></div>
<div class="card"><div class="label">日收益与标的相关性</div><div class="number">0.944</div><div class="detail">各年度有效 Pearson 相关系数均值</div></div>
</div>
<p class="source">统一来源：冻结运行 <code>20261015_095549_25eab142</code>，本地 <code>data/raw_hfq</code> 特征与 <code>data/raw</code> 真实价，2022–2025 全部验证 / 测试年度段。2022/23/24/25 分别 293/297/298/300 段；账户每年重置，非连续四年组合。</p>

<section>
<h2>01 / 高持仓不是几个样例的巧合</h2>
<p>四个年度平均仓位均超过 90%。766 / 1,188 段的平均仓位 ≥90%；248 段全年没有卖出信号。剔除存在大幅 raw/hfq 收益差的 159 段后，其余 1,029 段平均仓位仍为 <strong>91.16%</strong>，说明该行为并不只是少数除权异常造成的。</p>
<div class="toolbar"><button class="active" data-toggle="exposure" data-mode="chart">看图</button><button data-toggle="exposure" data-mode="table">精确表格</button><span class="spacer"></span><span class="note">上一轮 λ=0.01；最新 λ=0，均约 200 万步</span></div>
<div id="exposureChart" class="chart"></div>
<div id="exposureTable" class="scroll hidden"><table><thead><tr><th>年度</th><th>区间数</th><th>最新平均仓位</th><th>最新持仓日占比</th><th>上一轮平均仓位</th><th>最新卖出信号比例</th></tr></thead><tbody></tbody></table></div>
<div class="two" style="margin-top:18px"><div><h3>涨跌区间也没有明显仓位分化</h3><div id="regimeChart" class="chart short"></div></div><div><h3>信号看起来很多，真正退出很少</h3><div id="actionChart" class="chart short"></div></div></div>
<p class="source">仓位 = 当日收盘持仓市值 / 当日收盘权益；持仓日 = 收盘股数＞0。上/下/震荡分组按同约束恒买基准年度收益 >5% / &lt;−5% / 其余，属事后诊断分组，不是实时市场状态标签。该收益分组同样受公司行为漏记影响，不能当作真实牛熊行情分类。“剔除候选段”只是敏感性检查，不是修正回测，也不保证剔除了小额分红或所有数据问题。</p>
</section>

<section>
<h2>02 / 不是卖不出去，而是不太选择卖</h2>
<div class="pillgrid"><div><span class="note">有持仓时发出的卖出信号</span><strong>8,397 次</strong><span class="note">只占有持仓信号日的 3.22%</span></div><div><span class="note">实际卖出成交</span><strong>8,394 次</strong><span class="note">仅 3 次被当前跌停规则拒绝</span></div><div><span class="note">无现金仍重复买入的拒单</span><strong>134,635 次</strong><span class="note">51% 买入信号 ≠ 51% 有效交易</span></div></div>
<p>全部信号中：买入 <strong>50.97%</strong>、保持不动 <strong>44.30%</strong>、卖出 <strong>4.73%</strong>。其中另有 5,036 次卖出发生于空仓，无法成交。买入和保持不动在已经接近满仓时都会延续敞口。</p>
<p>动作概率本身也很尖锐：83.08% 的决策日，某一个动作的概率 ≥90%；平均策略熵为 0.128 nats（三动作最大值为 1.099）。这证明当前输出较确定，<strong>但不能单凭熵认定训练失败的唯一原因</strong>，也不能靠改成随机采样就宣称策略修复。</p>
<p class="source">来源：同上最新冻结模型的 284,095 次确定性预测、完整 categorical 概率和真实执行回执。这里的 3 次拒单仅描述当前缺少公司行为处理的撮合规则，不等同于真实市场受限次数。</p>
</section>

<section>
<h2>03 / 更优先的缺陷：除权后少算了股数和现金</h2>
<span class="badge critical">已确认 · 重训前阻断项</span>
<p>使用未复权价格进行真实价格撮合是正确方向，<strong>但必须同步处理公司行为权益</strong>。当前环境只在买卖分支改变股数 / 现金，未把转增股份和应收 / 实收股息入账，涨跌停参考价也仍直接使用除权前昨收。</p>
<h3>宁德时代（300750） / 2023-04-26</h3>
<div class="scroll"><table><thead><tr><th>项目</th><th>2023-04-25</th><th>2023-04-26</th></tr></thead><tbody>
<tr><td>本地 raw 收盘价</td><td>¥385.90</td><td>¥224.50</td></tr>
<tr><td>本地 hfq 收盘价（仅特征）</td><td>387.16</td><td>407.88</td></tr>
<tr><td>当前环境持股数</td><td>2,400 股</td><td>仍 2,400 股</td></tr>
<tr><td>当前环境现金</td><td>¥13,802.77</td><td>仍 ¥13,802.77</td></tr>
<tr><td>当前环境账户权益</td><td>¥939,962.77</td><td>¥552,602.77</td></tr>
<tr><td>官方公司行为</td><td>股权登记日</td><td>每 10 股转 8 股、派 ¥25.20（含税）</td></tr>
</tbody></table></div>
<p>按官方方案，该笔原有 2,400 股应对应 <strong>4,320 股</strong>及 <strong>¥6,048 税前现金分红权益</strong>；实际可用现金、税款与到账时点需要事件账本处理。当前模拟漏记这些权益，于是把 raw 的 −41.82% 价格跳变记成大额损失，而 hfq 同日变化为 +5.35%。</p>
<p class="warninline">这是单事件机制复核，不是已完成修正的回测。不能直接用 hfq 当真实成交价补洞，也不能仅从 hfq/raw 比值“猜”分红送转方案。</p>
<p class="source">行情证据：本地 <code>data/raw/300750.parquet</code> / <code>data/raw_hfq/300750.parquet</code>，2023-04-25~26；账户证据：本次完整逐日输出。公司一手证据：<a href="https://static.cninfo.com.cn/finalpage/2024-06-18/1220392100.PDF" target="_blank" rel="noopener">宁德时代公告 2024-034（2024-06-18），第 10 页</a>，已下载并核实原文；该页复述 2023 年权益分派日期及方案。原实施公告：<a href="https://www.catl.com/uploads/1/file/public/202304/20230421100127_yuyuyobjuc.pdf" target="_blank" rel="noopener">2022 年年度权益分派实施公告</a>。</p>
<p>全缓存筛查另发现 <strong>168 个交易日、95 只股票、159 个年度段</strong>的 raw/hfq 日收益差超过 5 个百分点。这些只是<strong>候选异常</strong>，可能涉及公司行为或跨源口径问题，未逐项核实，不能把 168 天都当作已确认除权事件。</p>
</section>

<section>
<h2>04 / 收益对照支持“未证明有效择时”，不是简单的零收益</h2>
<p class="warninline">以下为现环境可复现值，非可靠的含公司行为总回报。收益是股票×年度等权均值，不是账户累计四年收益；“严格恒买基准”也缺同一公司行为处理，因此偏差不会保证自动抵消。</p>
<div class="toolbar"><button class="active" data-toggle="return" data-mode="chart">看图</button><button data-toggle="return" data-mode="table">精确表格</button></div>
<div id="returnChart" class="chart"></div>
<div id="returnTable" class="scroll hidden"><table><thead><tr><th>年度</th><th>最新策略</th><th>严格恒买</th><th>超额（百分点）</th><th>最新最大回撤均值</th><th>基准最大回撤均值</th></tr></thead><tbody></tbody></table></div>
<div class="scroll" style="margin-top:16px"><table><thead><tr><th>模型 / 区间</th><th>平均仓位</th><th>策略收益均值</th><th>严格恒买均值</th><th>超额（百分点）</th></tr></thead><tbody id="compareRows"></tbody></table></div>
<p>最新模型验证期均值 −7.38% 对基准 −6.33%，测试期 +22.15% 对 +27.11%。测试期的正收益有明显市场暴露成分，不能当作主动择时成功。另一方面，2022 年仍跑赢基准约 1.55 个百分点，一些个股年度也有改善，所以它<strong>并非所有区间都完全没有作用</strong>。</p>
<p>上一轮 λ=0.01 的实际平均仓位只有 <strong>5.74%</strong>，最新 λ=0 后升至 <strong>91.21%</strong>。这是本次逐日实测，不是用收益比值反推。两轮从“长期空仓”走向“长期重仓”，说明当前设定对惩罚很敏感，不能认为去掉仓位惩罚就完成了修复。</p>
<details><summary>查看其他固定基准及解释边界</summary><div class="scroll"><table><thead><tr><th>年度</th><th>严格恒买</th><th>只首次买入</th><th>固定 20 日均线规则</th><th>空仓现金</th></tr></thead><tbody id="baselineRows"></tbody></table></div><p class="note">全部复用同一严格环境。首次买入规则会重试直到首次成交，然后不加仓；20 日均线规则为当日 hfq 收盘高于自身 20 日均线则次日买入，否则卖出，未调参；现金利率设 0。均线未稳定胜出，不能作为已经验证的替代策略。</p></details>
<p class="source">来源：最新和上一轮冻结模型，本次各 1,188 段同约束评估。新基准首点为初始现金、每日买入信号、T+1 开盘执行，和旧 metrics 基准有口径差异，因此新超额 −1.04 / −4.96pp 不与旧 −1.21 / −4.93pp 混用。</p>
</section>

<section>
<h2>05 / 确认事实与待验证归因</h2>
<div class="evidence"><div class="index">1</div><div><span class="badge critical">确认 · 环境真实性缺陷</span><h3>公司行为没有进入账户、奖励与除权日参考价</h3><p class="blocktext">这是收益可信度问题，不是仅在页面补一句说明就能解决。同一缺陷也存在于训练环境，训练奖励可能受公司行为污染；本次已核实的宁德时代事件位于验证期，不能单凭它断言训练期污染程度。修正后仍需重新验证，不能只修回放就把旧模型当作已验证模型。</p><p class="source"><code>env/strict_env.py:49–71, 228–234, 247–321, 375–393</code>；<code>env/limits.py:120–174</code>；<code>data/schema.py:19–28</code>。本地源代码路径均以 <code>src/stock_ai/</code> 为前缀。</p></div></div>
<div class="evidence"><div class="index">2</div><div><span class="badge">确认 · 不完整状态</span><h3>策略不知自己是否持仓，动作效果却依赖持仓</h3><p class="blocktext">20×9 观测全部是行情特征，缺当前仓位、现金、可卖数量等；PPO 使用非循环 MLP，对同一行情输入，空仓与满仓得到相同动作分布。它仍可能学到行情到敞口的映射，但难以对账户状态做闭环控制。</p><p class="source"><code>scale/unified.py:24–26, 263–281</code>；<code>env/strict_env.py:401–403</code>；<code>train/sb3_trainer.py:171–182, 206–222</code>。</p></div></div>
<div class="evidence"><div class="index">3</div><div><span class="badge">确认 · 奖励理论边界</span><h3>仅减去市场收益，不会自动创造择时优势</h3><div class="formula">reward = 账户日收益 − 市场日收益 − 0.002 × 实际换手</div><p class="blocktext">在市场路径外生、episode 长度与终止不依赖动作、其他 λ 相同的条件下，市场收益项对任何动作序列都是同一项。故同路径折扣奖励之差不变，理论最优动作排序也不变。它可以改变奖励尺度、critic 和有限样本优化轨迹，<strong>并不是“完全无用”</strong>；但不能宣称换成 excess 就从数学上解决了持仓退化。</p><p class="source"><code>env/rewards.py:107–128</code>；<code>env/strict_env.py:318–324, 375–393</code>。当前最新 λdrawdown=0、λleverage=0；gross 类型忽略惩罚，纯基准消融应使用相同 λ 的 drawdown_penalty 对 excess，而非直接拿 gross 冒充。</p></div></div>
<div class="evidence"><div class="index">4</div><div><span class="badge">确认配置 · 归因仍需消融</span><h3>极端惩罚尺度、探索和模型选择缺少防退化验收</h3><p class="blocktext">旧 λleverage=0.01 是每个满仓日减 0.01 奖励（1% / 100 bps 奖励量纲），不是只惩罚“超过 100% 的杠杆”；近 240 日相加约 −2.4 奖励单位，不是投资收益 −240%。最新直接取消后又缺其他风险约束。两轮 ent_coef=0、target_kl=None，模型是最终 checkpoint，非验证期最优选择；这些是风险线索，尚不能单独认定为主因。</p><p class="source"><code>env/rewards.py:125–126</code>；<code>train/sb3_trainer.py:163–182</code>；<code>scale/unified.py:616–642, 676–703</code>；两轮 manifest 及模型实际参数。</p></div></div>
<details><summary>其余工程问题，不应与主因混淆</summary><ul class="blocktext"><li><code>EnvConfig.max_position</code> 未接入 strict 买入预算。现模型设为 1.0，因此这不是本次接近满仓的直接解释；仅改配置数值也不会限制仓位。</li><li>训练池是有放回随机年度采样，200 万步不等于每段都训练了相同轮次。记录了池大小，未保存逐 episode 访问覆盖率；仅能估算约 2.73 次数据集总步数当量，不能当作实际覆盖率。</li><li>验证与测试已经被用于观察和调参判断；时间留出仍是同一批股票，不能证明对全新股票的泛化。当前成分股回溯也存在幸存者偏差风险。</li><li>现有环境与页面测试通过只证明实现符合当前契约，不证明契约符合真实账户经济意义。</li></ul></details>
</section>

<section>
<h2>06 / 建议修正顺序：先可信，再可控，最后才训练</h2>
<ol class="steps">
<li><h3>先修数据与账户：保留 raw 撮合，补公司行为账本</h3><p>接入可追溯的送转 / 拆并股、现金分红、登记 / 除权 / 到账、除权参考价；同步股数、权益 / 应收、成本与可卖数量，并统一基准和奖励。无事件支持时明确阻断或标记无效，不静默当普通交易日。</p></li>
<li><h3>补账户观测，收敛动作语义</h3><p>加入真实市值仓位、现金、可卖状态等；优先用可重复的目标敞口动作（例如空仓 / 持有目标仓位）代替依赖隐含状态的重复买卖。分档仓位按实际需求再决定，不用硬性低仓位制造“学会择时”的假象。</p></li>
<li><h3>明确风险收益目标，先做量纲校准和机制单测</h3><p>保留实际费用与滑点；换手正则与风险项分别披露，避免同一费用无意重复计入。下跌、横盘、上涨、无信号合成路径都要能解释，不能奖励强制频繁交易，或用持仓天数惩罚硬逼出交易。</p></li>
<li><h3>加入退化门禁与受控小实验，再决定全量训练</h3><p>保持一年 episode 不变，保存抽样覆盖率、动作 / 概率、仓位 / 拒单、收益 / 回撤、简单基准；从小规模、多随机种子、单变量消融开始。模型选择只用开发验证集，新股票留出与走步时段最后评估；不再反复用既有 test 选超参数。</p></li>
</ol>
<p class="warninline">兼容性要求：账户状态、目标动作或公司行为处理会改变模型含义。应发布新的观测 / 动作 / 执行 / 奖励契约，旧模型保留旧版研究回放并明确警告；禁止给旧 20×9 模型偷偷补维、改动作编号、重写旧 manifest 或继续冒用同一版本。</p>
<p><strong>当前状态：</strong>本次仅完成诊断和证据输出，未修改业务代码、未重启已有服务、未启动新训练。上述是待实施方案，不是已经修复的功能。</p>
</section>

<section>
<h2>07 / 逐股票×年度查阅</h2>
<p class="note">可切换两轮模型、年份并按六位代码查询；收益列仍是存在公司行为缺陷的模拟值。表内每行起始空仓，不同于跨年连续回放。</p>
<div class="toolbar"><label>模型 <select id="modelFilter"><option value="20261015_095549_25eab142">最新 · λ=0</option><option value="20261014_185149_8e273f02">上一轮 · λ=0.01</option></select></label><label>年份 <select id="yearFilter"><option value="all">全部</option><option>2022</option><option>2023</option><option>2024</option><option>2025</option></select></label><label>代码 <input id="symbolFilter" placeholder="例如 000001" maxlength="6" inputmode="numeric"></label><label>排序 <select id="sortFilter"><option value="symbol">代码 / 年份</option><option value="position_desc">仓位从高到低</option><option value="excess_asc">模拟超额从低到高</option></select></label></div>
<div id="rowCount" class="status"></div>
<div class="scroll"><table class="compact"><thead><tr><th>代码 / 年度</th><th>平均仓位</th><th>持仓日 / 总日</th><th>最长持仓</th><th>卖出信号</th><th>成交笔数</th><th>模拟收益</th><th>模拟超额 pp</th></tr></thead><tbody id="episodeRows"></tbody></table></div>
<div class="toolbar"><button id="previous">上一页</button><span id="pageLabel" class="status"></span><button id="next">下一页</button></div>
</section>

<section>
<h2>08 / 可复核性与交付清单</h2>
<ul class="blocktext"><li>完整复现两轮模型存量评估的策略收益、回撤和成交数，共 2,376 条记录；另独立模拟三种固定基准。</li><li>4 组生产回放交叉核验：000001/2024、300750/2023、600519/2022、600519/2024，仓位、持仓日、收益、成交与基准均一致。</li><li>模型 categorical 概率 argmax 与实际确定性预测一致；账务恒等式成立。<strong>但恒等式成立不能证明漏记的公司行为权益已正确计入。</strong></li><li>612 个模型 / normalizer / manifest / 状态 / metrics / 切分及行情输入文件的 hash 与修改时间保持不变，hfq 源文件 hash 与训练 manifest 一致。真实执行 raw 的训练时逐文件 hash 未写入旧 manifest，只能证明本次诊断前后未改变，不能补证训练时版本。</li><li>ST、新股无涨跌幅限制、公司行为、跨源数据口径及幸存者偏差仍有限制；没有新股票留出，既有 test 也不是干净的最终确认集。</li></ul>
<div class="scroll"><table><thead><tr><th>文件</th><th>用途</th></tr></thead><tbody><tr><td><code>unified_exposure_episodes_20261016.csv</code></td><td style="text-align:left">两轮模型全部逐股票年度指标，2,376 行</td></tr><tr><td><code>unified_adjustment_candidates_20261016.csv</code></td><td style="text-align:left">168 个待核实的大幅 raw/hfq 收益差候选</td></tr><tr><td><code>unified_exposure_diagnosis_20261016.json</code></td><td style="text-align:left">完整指标、信号 / 拒单 / 概率统计及 4 组逐日证据</td></tr><tr><td><code>diagnose_unified_exposure.py</code></td><td style="text-align:left">只读可复运行脚本，复用生产特征与执行环境</td></tr></tbody></table></div>
<details><summary>设计参考与证据层级</summary><p class="note">工程判定以本地源码、冻结产物、完整推理及官方公告为准。<a href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html" target="_blank" rel="noopener">SB3 PPO 官方文档</a>用于核对 ent_coef、target_kl 和非循环 MLP。外部教程和研究仅作机制参考，不采用其绩效数字或未经本项目验证的参数建议；本项目的收益、仓位与归因以实际执行和源码为准。外部检索曾返回与本地不一致的可比历史报价，本报告未混用；公司行为方案仅取官方公告。</p></details>
<p class="foot"><strong>免责声明</strong>：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。市场有风险，投资需谨慎。任何投资决策应结合个人风险承受能力、资金状况和投资目标独立判断，必要时咨询持牌专业机构。过往表现不预示未来收益。</p>
</section>
<div id="renderStatus"></div>
<footer>离线可查看 · 图表库已内嵌 · 数据截止 2025-12-31 · 诊断 2026-10-16 · 全部数值来自本次本地执行，非实时行情</footer>
</main>
<script>/*VENDOR*/</script>
<script id="reportData" type="application/json">/*DATA*/</script>
<script>
"use strict";
(function(){
const D=JSON.parse(document.getElementById("reportData").textContent);
const latest="20261015_095549_25eab142", previous="20261014_185149_8e273f02";
const A=D.summaries[latest], B=D.summaries[previous], years=["2022","2023","2024","2025"];
const palette={blue:"#365fc5",orange:"#bd7f33",ink:"#24324b",red:"#bc3849",green:"#21865e",gray:"#78869c"};
const charts={};
const p=x=>(x*100).toFixed(2)+"%";
const pp=x=>(x*100).toFixed(2);
const colored=x=>'<span class="'+(x>0?"positive":x<0?"negative":"neutral")+'">'+p(x)+"</span>";
const tableBody=id=>document.querySelector("#"+id+" tbody");
function initChart(id,option){const chart=echarts.init(document.getElementById(id));chart.setOption(option);charts[id]=chart;return chart;}
const common={backgroundColor:"#fff",color:[palette.blue,palette.orange,palette.gray],textStyle:{color:palette.ink,fontFamily:"Microsoft YaHei, Segoe UI, sans-serif"},tooltip:{trigger:"axis",valueFormatter:v=>Number(v).toFixed(2)+"%"},legend:{bottom:0,textStyle:{color:palette.ink}},grid:{left:54,right:18,top:26,bottom:55},xAxis:{type:"category",data:years,axisLabel:{color:palette.ink},axisLine:{lineStyle:{color:"#dce3ee"}}},yAxis:{type:"value",axisLabel:{formatter:"{value}%",color:"#606d80"},splitLine:{lineStyle:{color:"#edf0f6"}}}};
initChart("exposureChart",{...common,yAxis:{...common.yAxis,min:0,max:100},series:[{name:"最新模型 λ=0",type:"bar",data:years.map(y=>A.by_year[y].mean_position*100),itemStyle:{borderRadius:[3,3,0,0]}},{name:"上一轮 λ=0.01",type:"bar",data:years.map(y=>B.by_year[y].mean_position*100),itemStyle:{borderRadius:[3,3,0,0]}}]});
tableBody("exposureTable").innerHTML=years.map(y=>{const a=A.by_year[y],b=B.by_year[y];return '<tr><td>'+y+'</td><td>'+a.episodes+'</td><td>'+p(a.mean_position)+'</td><td>'+p(a.holding_fraction)+'</td><td>'+p(b.mean_position)+'</td><td>'+p(a.action_fractions["2"])+'</td></tr>';}).join("");
const regimes=["down_below_minus5pct","sideways_minus5_to_plus5pct","up_above_plus5pct"];
initChart("regimeChart",{...common,xAxis:{...common.xAxis,data:["下跌＜−5%","震荡 ±5%","上涨＞5%"]},yAxis:{...common.yAxis,min:0,max:100},series:[{name:"最新平均仓位",type:"bar",barMaxWidth:60,data:regimes.map(k=>A.by_baseline_regime[k].mean_position*100)}]});
initChart("actionChart",{...common,xAxis:{...common.xAxis,data:["最新 λ=0","上一轮 λ=0.01"]},yAxis:{...common.yAxis,min:0,max:100},series:[{name:"买入",type:"bar",stack:"actions",data:[A.all.action_fractions["1"]*100,B.all.action_fractions["1"]*100],itemStyle:{color:palette.red}},{name:"保持",type:"bar",stack:"actions",data:[A.all.action_fractions["0"]*100,B.all.action_fractions["0"]*100],itemStyle:{color:palette.gray}},{name:"卖出",type:"bar",stack:"actions",data:[A.all.action_fractions["2"]*100,B.all.action_fractions["2"]*100],itemStyle:{color:palette.green}}]});
function signedBars(values){return values.map(v=>({value:v,itemStyle:{color:v>=0?palette.red:palette.green}}));}
initChart("returnChart",{...common,color:[palette.red,palette.red,palette.gray],series:[{name:"最新策略（现环境）",type:"bar",data:signedBars(years.map(y=>A.by_year[y].mean_return*100))},{name:"严格恒买（现环境）",type:"bar",itemStyle:{opacity:0.55},data:signedBars(years.map(y=>A.by_year[y].mean_baseline_return*100))},{name:"上一轮策略（现环境）",type:"line",symbolSize:7,data:years.map(y=>B.by_year[y].mean_return*100),lineStyle:{width:2}}]});
tableBody("returnTable").innerHTML=years.map(y=>{const a=A.by_year[y];return '<tr><td>'+y+'</td><td>'+colored(a.mean_return)+'</td><td>'+colored(a.mean_baseline_return)+'</td><td class="'+(a.mean_excess_return>=0?"positive":"negative")+'">'+pp(a.mean_excess_return)+'</td><td>'+p(a.mean_max_drawdown)+'</td><td>'+p(a.mean_baseline_max_drawdown)+'</td></tr>';}).join("");
document.getElementById("compareRows").innerHTML=[["最新 / 验证期 2022–23",A.by_split.val],["最新 / 测试期 2024–25",A.by_split.test],["上一轮 / 验证期 2022–23",B.by_split.val],["上一轮 / 测试期 2024–25",B.by_split.test]].map(([label,a])=>'<tr><td>'+label+'</td><td>'+p(a.mean_position)+'</td><td>'+colored(a.mean_return)+'</td><td>'+colored(a.mean_baseline_return)+'</td><td class="'+(a.mean_excess_return>=0?"positive":"negative")+'">'+pp(a.mean_excess_return)+'</td></tr>').join("");
document.getElementById("baselineRows").innerHTML=years.map(y=>{const a=D.baseline_summaries[y];return '<tr><td>'+y+'</td><td>'+colored(a.always_buy_return)+'</td><td>'+colored(a.entry_once_return)+'</td><td>'+colored(a.ma20_return)+'</td><td>0.00%</td></tr>';}).join("");
for(const button of document.querySelectorAll("[data-toggle]"))button.addEventListener("click",()=>{const key=button.dataset.toggle,mode=button.dataset.mode;const chartId=key==="exposure"?"exposureChart":"returnChart";document.getElementById(chartId).classList.toggle("hidden",mode!=="chart");document.getElementById(key+"Table").classList.toggle("hidden",mode!=="table");for(const sibling of document.querySelectorAll('[data-toggle="'+key+'"]'))sibling.classList.toggle("active",sibling===button);if(mode==="chart")charts[chartId].resize();});
let page=0, filtered=[];const pageSize=40;
function updateRows(reset=true){if(reset)page=0;const run=document.getElementById("modelFilter").value,year=document.getElementById("yearFilter").value,symbol=document.getElementById("symbolFilter").value.trim();filtered=D.episodes.filter(r=>r.run_id===run&&(year==="all"||String(r.year)===year)&&r.symbol.includes(symbol));const sort=document.getElementById("sortFilter").value;filtered.sort((a,b)=>sort==="position_desc"?b.average_position-a.average_position:sort==="excess_asc"?a.excess_return-b.excess_return:a.symbol.localeCompare(b.symbol)||a.year-b.year);const pageCount=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pageCount-1);document.getElementById("episodeRows").innerHTML=filtered.slice(page*pageSize,(page+1)*pageSize).map(r=>'<tr><td><code>'+r.symbol+'</code> / '+r.year+'</td><td>'+p(r.average_position)+'</td><td>'+r.holding_days+' / '+r.bars+'</td><td>'+r.longest_holding_days+' 日</td><td>'+r.actions["2"]+'</td><td>'+r.trades+'</td><td>'+colored(r.total_return)+'</td><td class="'+(r.excess_return>=0?"positive":"negative")+'">'+pp(r.excess_return)+'</td></tr>').join("");document.getElementById("rowCount").textContent="筛选结果 "+filtered.length+" 段；每页 "+pageSize+" 行。";document.getElementById("pageLabel").textContent=(page+1)+" / "+pageCount;document.getElementById("previous").disabled=page===0;document.getElementById("next").disabled=page>=pageCount-1;}
for(const id of ["modelFilter","yearFilter","sortFilter"])document.getElementById(id).addEventListener("change",()=>updateRows());document.getElementById("symbolFilter").addEventListener("input",()=>updateRows());document.getElementById("previous").addEventListener("click",()=>{page--;updateRows(false);});document.getElementById("next").addEventListener("click",()=>{page++;updateRows(false);});updateRows();
window.addEventListener("resize",()=>Object.values(charts).forEach(chart=>chart.resize()));
window.diagnosisQA={charts,D,get filteredCount(){return filtered.length;},get chartCount(){return Object.keys(charts).length;}};
})();
</script>
</body>
</html>
'''
html = html.replace("/*VENDOR*/", vendor).replace("/*DATA*/", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
path = OUT / "unified_exposure_diagnosis_20261016.html"
path.write_text(html, encoding="utf-8")
print(json.dumps({"report": str(path), "bytes": path.stat().st_size, "episode_rows": len(payload["episodes"])}, ensure_ascii=False))

"""Build an offline acceptance page from actual local evidence, never market estimates."""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TARGET = OUT / "corporate_action_acceptance_20261016.html"


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    regression = read("corporate_action_regression_final.json")
    protected = read("corporate_action_protected_inputs_final.json")
    case = read("corporate_action_case_300750_20230426_final.json")
    browser = read("corporate_action_ui_acceptance_final/acceptance.json")
    assert regression["exit_code"] == 0 and regression["failures"] == regression["errors"] == 0
    assert regression["learning_attempts"] == []
    assert protected["passed"] and browser["passed"]
    assert browser["browser_closed"] and browser["server_closed"] and browser["fixture_files_unchanged"]
    assert browser["page_errors"] == [] and browser["console_errors"] == []
    assert browser["training_process_count"] == 0 and all(browser["checks"].values())
    vendor = (ROOT / "src/stock_ai/scale/ui/static/vendor/echarts.min.js").read_text(encoding="utf-8")
    if "</script" in vendor.lower():
        raise ValueError("Chart library contains a closing script tag")
    evidence = {"regression": regression, "protected": {k: v for k, v in protected.items() if k != "files"},
                "case": case, "browser": browser}
    payload = json.dumps(evidence, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    old = case["comparison_ex_date"]["old"]
    new = case["comparison_ex_date"]["corrected"]
    case_rows = "".join(
        f"<tr><td>{html.escape(row['version'])}</td><td>{row['date']}</td><td>{row['close']:,.2f}</td>"
        f"<td>{row['shares']:,}</td><td>{row['cash']:,.2f}</td><td>{row['dividend_receivable']:,.2f}</td>"
        f"<td>{row['dividend_income']:,.2f}</td><td>{row['equity']:,.2f}</td></tr>" for row in case["daily"])
    document = r'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>公司行为 v2 · 修正与验收</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--ink:#18243a;--muted:#5c6b80;--line:#dce3ed;--red:#c83c42;--blue:#285cc8;--green:#16815c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:40px 24px 64px}h1{font-size:34px;line-height:1.3;margin:8px 0 12px}h2{font-size:23px;margin:0 0 14px}h3{font-size:16px;margin:8px 0}p{margin:8px 0}.eyebrow{letter-spacing:2px;font-size:12px;font-weight:700;color:var(--blue)}.subtitle,.note{color:var(--muted)}.note{font-size:13px}.pill{border:1px solid #b8d8cb;background:#edf9f2;color:#176744;display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}.warning{padding:16px 20px;border-left:4px solid #b17920;background:#fff7e7;color:#6d4a16;border-radius:6px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card,section{background:var(--panel);border:1px solid var(--line);border-radius:12px}.card{padding:20px}.number{font-size:30px;line-height:1.35;font-weight:700;margin:8px 0}.label{font-size:13px;color:var(--muted)}section{padding:26px;margin:18px 0}.status{display:grid;grid-template-columns:1fr 2fr;gap:10px 24px}.status>div{padding:8px 0;border-bottom:1px solid #edf0f5}.ok{color:var(--green);font-weight:600}.pending{color:#8c601d;font-weight:600}.chart{height:360px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:14px 0}button{background:#fff;border:1px solid #c9d3e2;color:var(--ink);padding:7px 12px;border-radius:6px;cursor:pointer}button.active{background:#edf3ff;border-color:#7b9edf;color:#2453ae}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:13px}th,td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:right}th{background:#f5f7fc;color:#445672}th:first-child,td:first-child{text-align:left}.chart{height:360px;width:100%;min-width:0}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:14px 0}button{background:#fff;border:1px solid #c9d3e2;color:var(--ink);padding:7px 12px;border-radius:6px;cursor:pointer}button.active{background:#edf3ff;border-color:#7b9edf;color:#2453ae}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.step{padding:17px;background:#f5f8fe;border:1px solid #dfe7f5;border-radius:8px;min-width:0}.step b{display:block;color:var(--blue);margin-bottom:5px}.two{display:grid;grid-template-columns:1fr 1fr;gap:26px}ul{padding-left:20px;margin:8px 0}li{margin-bottom:7px}code{font:12px/1.5 Consolas,monospace;background:#eef2f8;border-radius:3px;padding:2px 5px;overflow-wrap:anywhere}a{color:#225ab8}details{margin:14px 0}summary{cursor:pointer;color:var(--blue)}pre{white-space:pre-wrap;word-break:break-all;overflow-wrap:anywhere;max-height:420px;overflow:auto;background:#f6f8fc;padding:16px;border:1px solid var(--line);font:12px/1.6 Consolas,monospace}.footer{margin:24px 0;font-size:12px;color:var(--muted)}[hidden]{display:none!important}section>*{min-width:0}@media(max-width:800px){.cards,.flow,.two{grid-template-columns:1fr}.status{grid-template-columns:1fr}}@media(max-width:600px){.cards,.flow{grid-template-columns:1fr}main{padding:24px 14px}h1{font-size:26px}.chart{height:300px}}@media print{body{background:#fff}.card,section{break-inside:avoid}button{display:none}}
</style></head><body><main>
<header><div class="eyebrow">STOCK AI / EXECUTION CORRECTNESS</div><h1>公司行为 v2：先把权益算对</h1>
<p class="subtitle">登记、除权、派息、股份上市分开记账；旧模型不静默换口径。</p>
<span class="pill">代码机制及接线验收通过 · 未训练 · 未部署</span>
<p class="note">验收日期 2026-10-16 · 从本地测试与公告证据生成 · 可离线打开</p></header>
<div class="warning"><strong>完成的是机制，不是全历史修复。</strong> 300 股票、2006–2025 生产公司行为覆盖尚未建立。缺核验账本仍会拒绝 v2 请求，不会回退旧版或猜事件。长期高持仓 / 策略效果没有因此被证明改善。</div>
<div class="cards"><div class="card"><div class="label">后端组合回归</div><div class="number">__TESTS__</div><div class="note">通过；5 个真实训练用例排除</div></div><div class="card"><div class="label">真实学习调用</div><div class="number">0</div><div class="note">包装器及具体算法双重禁用</div></div><div class="card"><div class="label">受保护文件</div><div class="number">__FILES__</div><div class="note">SHA256 + 修改时间全部不变</div></div><div class="card"><div class="label">页面与导出</div><div class="number">22 列</div><div class="note">4 图联动；CSV / JSON 一致</div></div></div>
<section><h2>交付状态：四件事明确分开</h2><div class="status">
<div class="ok">代码机制与统一接线：通过</div><div>新执行版本 <code>strict-t1-open-ca-v2</code>，独立恒买基准、应收 / 锁股、事件明细、来源冻结、CLI / UI / 导出。</div>
<div class="pending">生产数据覆盖：尚未完成</div><div><code>verified</code> 是提供者的审阅声明，不是程序证明。归档接口原响应始终标 <code>unreviewed</code>，须核对官方实施公告与其他资本行为。</div>
<div class="pending">修正版本模型：尚未训练</div><div>现有 policy、manifest、normalizer 均未迁移。旧 v1 放入 v2 的显式研究执行，不等于修复并重训旧模型。</div>
<div class="pending">运行中服务：尚未部署</div><div>原有 8766 / 8767 保持运行且没有重启；新后端需用户自行安排加载。本次隔离浏览器和临时服务已关闭。</div>
</div></section>
<section><h2>同一个真实事件，旧账户漏了什么？</h2>
<p>宁德时代（300750），2023-04-26 除权。固定动作：4/24 收盘买入信号 → 4/25 开盘买入 2500 股后持有。<strong>仅机制夹具，不是模型表现。</strong></p>
<div class="note">来源：公司 2023-037 实施公告（2023-04-18）与本地 raw，区间 2023-04-24～28；公告及文件 SHA256 均在随附 JSON 中。</div>
<div class="toolbar"><button class="active" id="show-chart" type="button">权益曲线</button><button id="show-table" type="button">精确逐日数据</button><span class="note">现金 + raw 市值 + 税前股息应收</span></div>
<div id="case-chart" class="chart" role="img" aria-label="旧版与公司行为修正版账户权益对比"></div>
<div id="case-table" class="scroll" hidden><table><thead><tr><th>版本</th><th>日期</th><th>raw 收盘 / 元</th><th>总股数</th><th>现金 / 元</th><th>股息应收 / 元</th><th>累计股息 / 元</th><th>权益 / 元</th></tr></thead><tbody>__CASE_ROWS__</tbody></table></div>
<div class="scroll"><table><thead><tr><th>2023-04-26</th><th>旧 v1</th><th>修正 v2</th></tr></thead><tbody>
<tr><td>总持股</td><td>2500 股</td><td>4500 股</td></tr><tr><td>税前股息累计</td><td>0 元</td><td>6300 元</td></tr>
<tr><td>账户权益</td><td>__OLD_EQUITY__ 元</td><td>__NEW_EQUITY__ 元</td></tr>
<tr><td>正确除权参考价</td><td>仍误用前 raw 收盘</td><td>212.99 元</td></tr><tr><td>涨停 / 跌停</td><td>463.08 / 308.72 元</td><td>255.59 / 170.39 元</td></tr>
</tbody></table></div>
<p><strong>差额 455300 元 = 漏计 2000 股 × 224.50 元 + 6300 元税前股息。</strong>这不是策略新增盈利，而是旧账户漏计权益。本例分红和转增同日到账 / 上市，所以当日收盘没有剩余应收与锁股。</p>
<p class="note">两版本首日均按收盘显式估值为 100 万元。最终文件使用 <code>_final</code> 后缀修正旧验收首日展示 0；原产物保留。单事件核实不证明完整覆盖，测试声明未导入生产。</p>
<p><a href="https://static.cninfo.com.cn/finalpage/2023-04-18/1216457838.PDF" target="_blank" rel="noopener">查看官方实施公告</a> · <a href="corporate_action_case_300750_20230426_final.json">真实案例完整 JSON</a> · <a href="corporate_action_case_300750_20230426_final.csv">逐日 CSV</a></p>
</section>
<section><h2>四个时间点，不再混成一次价格调整</h2><div class="flow">
<div class="step"><b>① 登记日收盘</b>以当日交易后的持仓确权；首日及之前资格不倒推。</div>
<div class="step"><b>② 除权日开盘前</b>确认税前股息应收、送转股份与锁股；修正涨跌停参考价。</div>
<div class="step"><b>③ 派息日</b>仅把应收转为现金；不重复增加收入，也不再次冲减成本。</div>
<div class="step"><b>④ 股份上市日</b>解除待上市锁股；上市前参与估值，但不能卖出。</div>
</div><p><code>equity = cash + shares * raw_close + dividend_receivable</code></p><p><code>initial_cash + realized_pnl + unrealized_pnl + dividend_income = equity</code></p>
<p class="note">非 bar 派息 / 上市在下一可用开盘前补记并保留有效日期。碎股、配股、特殊方案、未知行为、缺登记 / 除权 bar 及未上市股嵌套登记资格直接拒绝。</p></section>
<section><h2>验收覆盖与不变量</h2><div class="two"><div><h3>机制与数据完整性</h3><ul>
<li>现金、股数、成本、股息应收、可卖 / 锁股及收益恒等式。</li><li>独立同约束恒买基准，异常恢复主账户、基准、事件、交易与内置奖励状态。</li><li>覆盖不足、重复事件、未知字段、未知 / 特殊机制、路径越界都拒绝。</li><li>v2 来源清单非空、六位代码唯一、三 hash 完整；新股票不伪装训练来源匹配。</li><li>每文件一次字节快照贯穿解析、指纹和 episode；读取后替换路径的反证测试通过。</li></ul></div><div><h3>真实浏览器（合成夹具 / 假 policy）</h3><ul>
<li>22 列逐日表、4 图联动、应收 / 锁股及完整事件详情。</li><li>逐日 CSV、成交 CSV、完整 JSON；改控件不重跑不污染已完成结果。</li><li>研究开关默认关，旧模型训练 / 执行双版本，缺 raw hash 标 <code>not_recorded</code>。</li><li>回切旧 v1 恢复 17 列；新建发送 strict-ca，resume 不覆写。</li><li>训练请求被拦截且后端拒绝；无页面异常，夹具文件未变化。</li></ul></div></div>
<p class="note">后端最终 10 文件 __TESTS__ 项通过，5 个真实训练测试排除；6 条 warning 为既有观测 Box 无穷边界提示。前阶段 / 目标测试不重复累计。首次浏览器阻塞运行已停止，不计入成功证据。</p>
<details><summary>查看机器可读验收摘要</summary><pre id="evidence"></pre></details>
<p><a href="corporate_action_regression_final.json">组合回归摘要</a> · <a href="corporate_action_regression_final.xml">JUnit 原始证据</a> · <a href="corporate_action_protected_inputs_final.json">643 文件保护核对</a> · <a href="corporate_action_ui_acceptance_final/acceptance.json">浏览器验收证据</a></p>
</section>
<section><h2>怎么查看、接下来做什么</h2><ol>
<li>现在可先看本页真实案例与随附原始证据；截图中的其他股票数据仅是合成页面机制测试，不是真实收益。</li>
<li>用户手动加载新后端后，在“模型回放”选择模型 / 股票 / 日期。旧 v1 默认仍按旧语义；只有显式勾选“公司行为修正研究”才以 v2 执行，且必须有覆盖该请求区间的审阅账本。</li>
<li>先建立并独立核验所需股票与年份的完整公司行为覆盖，再另外授权训练 / 对照评估。修正会计基础后，仍须重新检验高持仓、账户可观测性、动作机制和奖励量纲。</li>
</ol><div class="warning">仍未覆盖个人差异化股息税、完整 ST 历史身份、新股无涨跌幅限制、真实流动性与逐日历史税费。保留默认 20×9 行情观测和原三动作，不在本次静默改变模型契约。文件级快照一致性不代表多数据源原子快照或无泄漏证明。</div>
<p class="note">代码位置：<code>data/corporate_actions.py</code>、<code>data/corporate_action_sources.py</code>、<code>env/corporate_action_env.py</code>、<code>scale/unified.py</code>、<code>scale/replay.py</code>、<code>train/sb3_trainer.py</code>、<code>cli/scale_cmd.py</code>、<code>scale/ui/</code>；说明在 README 3.6.3、架构与 QA 增量章节。</p>
</section><footer class="footer">生成时间：__GENERATED__ · 本页仅为工程正确性验收，无个股买卖或仓位建议。原始行情、旧模型及既有服务未被本次验收改写 / 重启。</footer>
</main><script>__VENDOR__</script><script>
const D=__PAYLOAD__;
const c=echarts.init(document.getElementById("case-chart"));
const dates=D.case.daily.filter(r=>r.version==="v1").map(r=>r.date.slice(4,6)+"/"+r.date.slice(6));
c.setOption({color:["#7a889b","#285cc8"],tooltip:{trigger:"axis",valueFormatter:v=>"¥"+Number(v).toLocaleString("zh-CN",{minimumFractionDigits:2,maximumFractionDigits:2})},legend:{data:["旧 v1：遗漏公司行为","v2：补齐本事件权益"],top:8},grid:{top:55,left:74,right:24,bottom:38},xAxis:{type:"category",data:dates,boundaryGap:false,axisLabel:{color:"#5c6b80"}},yAxis:{type:"value",name:"账户权益 / 元",axisLabel:{color:"#5c6b80",formatter:v=>(v/10000)+"万"},splitLine:{lineStyle:{color:"#e8edf5"}}},series:[{name:"旧 v1：遗漏公司行为",type:"line",symbolSize:7,lineStyle:{type:"dashed"},data:D.case.daily.filter(r=>r.version==="v1").map(r=>r.equity)},{name:"v2：补齐本事件权益",type:"line",symbolSize:7,data:D.case.daily.filter(r=>r.version==="v2").map(r=>r.equity)}]});
function mode(table){document.getElementById("case-chart").hidden=table;document.getElementById("case-table").hidden=!table;document.getElementById("show-chart").classList.toggle("active",!table);document.getElementById("show-table").classList.toggle("active",table);if(!table)c.resize();}
document.getElementById("show-chart").addEventListener("click",()=>mode(false));
document.getElementById("show-table").addEventListener("click",()=>mode(true));
const {case:caseData,...summary}=D;document.getElementById("evidence").textContent=JSON.stringify(summary,null,2);
window.addEventListener("resize",()=>c.resize());window.caAcceptance={evidence:D,chart:c};
</script></body></html>'''
    replacements = {"__TESTS__": str(regression["tests"]), "__FILES__": str(protected["checked_count"]),
                    "__CASE_ROWS__": case_rows, "__OLD_EQUITY__": f"{old['equity']:,.5f}",
                    "__NEW_EQUITY__": f"{new['equity']:,.5f}", "__GENERATED__": datetime.now().astimezone().isoformat(),
                    "__VENDOR__": vendor, "__PAYLOAD__": payload}
    for key, value in replacements.items():
        document = document.replace(key, value)
    with TARGET.open("x", encoding="utf-8") as stream:
        stream.write(document)
    print(TARGET)


if __name__ == "__main__":
    main()

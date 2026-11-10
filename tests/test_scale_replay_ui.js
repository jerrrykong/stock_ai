/* 无浏览器依赖的 UI 行为测试：执行真实 replay.js，模拟 DOM/fetch/ECharts/下载。 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/stock_ai/scale/ui/static/replay.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../src/stock_ai/scale/ui/static/index.html'), 'utf8');

class Element {
  constructor() {
    this.children = []; this.value = ''; this.textContent = ''; this.hidden = false;
    this.dataset = {}; this.listeners = {}; this.disabled = false; this.checked = false;
    this.classList = {toggle: () => {}};
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) {
    this.children = children.flatMap(c => c.fragment ? c.children : [c]);
    if (this.children[0] && 'value' in this.children[0]) this.value = this.children[0].value;
  }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  querySelectorAll() { return this.children; }
  remove() {}
  click() { if (this.onClick) this.onClick(this); }
}
function harness(catalog, result, search = '') {
  const elements = new Map(), events = {}, dispatches = [], downloads = [], blobs = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const state = {catalog, result, failure: false, calls: [], option: null};
  const chart = {on: (name, fn) => {events[name] = fn;}, setOption: o => {state.option = o;}, resize: () => {}, dispatchAction: a => dispatches.push(a)};
  const context = {
    document: {
      getElementById: get,
      createElement: tag => {
        const element = new Element();
        if (tag === 'a') element.onClick = anchor => downloads.push({filename: anchor.download, blob: blobs.get(anchor.href)});
        return element;
      },
      createDocumentFragment: () => Object.assign(new Element(), {fragment: true}), body: new Element()
    },
    Option: function (label, value) {this.textContent = label; this.value = value;},
    fetch: async (url, options) => {
      state.calls.push({url, payload: options && JSON.parse(options.body)});
      return {ok: !state.failure, json: async () => state.failure ? {ok: false, error: '没有严格过去模型'} : url.includes('catalog') ? state.catalog : state.result};
    },
    echarts: {init: () => chart}, requestAnimationFrame: fn => fn(),
    window: {echarts: {}, addEventListener: () => {}}, console, location: {search}, URLSearchParams, Blob,
    URL: {createObjectURL: blob => {const url = 'blob:test-' + blobs.size; blobs.set(url, blob); return url;}, revokeObjectURL: () => {}},
    setTimeout: fn => fn()
  };
  vm.createContext(context); vm.runInContext(source, context);
  const submit = () => get('replay-form').listeners.submit({preventDefault: () => {}});
  return {get, state, events, dispatches, downloads, context, submit};
}
function legacyData() {
  const catalog = {ok: true, symbols: ['600519'], models: [{id: '600519_2025', symbol: '600519', year: 2025, backend: 'fake', normalizer_available: true}]};
  const daily = [0, 1, 2].map(i => ({date: `2023010${i + 2}`, open: 10, high: 11, low: 9, close: 10 + i,
    volume: 100, action: i === 2 ? null : 1, execution: i !== 2, trade_ok: i === 0,
    blocked_reason: i === 1 ? '现金不足' : i === 2 ? '末日仅估值，无新执行' : '', cash: 100, shares: 100,
    position: 0.9, equity: 1100 + i * 100, baseline: 1100 + i * 100}));
  const result = {ok: true, symbol: '600519', initial_cash: 1000, model: catalog.models[0], temporal_risk: true, selection: 'manual_research',
    warnings: ['未来模型风险', '同日模拟成交，不等于实盘'], actual_range: {start: daily[0].date, end: daily[2].date},
    warmup_rows: 30, baseline_note: '基线说明', summary: {bars: 3, executions: 2, trades: 1, final_equity: 1300, total_return: 0.3, baseline_return: 0.3},
    daily, trades: [{date: daily[0].date, side: 'buy', price: 10.01, shares: 100, notional: 1001, fee: 5}]};
  return {catalog, result};
}
(async () => {
  const legacy = legacyData(), h = harness(legacy.catalog, legacy.result);
  const {get, state, events, dispatches} = h;
  await h.context.window.scaleReplay.open();
  assert.equal(get('replay-symbol').value, '600519');
  assert.equal(get('replay-model').value, 'auto');
  get('replay-start').value = '2023-01-02'; get('replay-end').value = '2023-01-04';
  get('replay-model').value = '600519_2025'; get('replay-model').listeners.change();
  assert.match(get('replay-selection-warning').textContent, /前视风险/);
  assert.match(get('replay-selection-warning').className, /replay-risk/);
  await h.submit();
  assert.equal(state.calls.at(-1).payload.model_id, '600519_2025');
  assert.equal(get('replay-result').hidden, false); assert.equal(get('replay-run').disabled, false);
  assert.equal(get('replay-daily').children.length, 3); assert.equal(get('replay-trades').children.length, 1);
  assert.equal(state.option.grid.length, 4); assert.equal(state.option.xAxis.length, 4);
  assert.equal(state.option.dataZoom[0].xAxisIndex.length, 4);
  assert.equal(state.option.series[1].data.length, 2); assert.equal(state.option.series[3].data.length, 1);
  assert.equal(state.option.series[8].data.length, 3);
  assert.equal(state.option.series[8].name, '买入持有基线');
  get('replay-next').onclick(); assert.match(get('replay-detail').textContent, /现金不足/);
  get('replay-daily').children[2].onclick(); assert.match(get('replay-detail').textContent, /不产生新信号/);
  assert.equal(get('replay-next').disabled, true); assert.ok(dispatches.some(a => a.type === 'dataZoom'));
  events.updateAxisPointer({axesInfo: [{value: 0}]}); assert.match(get('replay-detail').textContent, /20230102/);
  state.failure = true; await h.submit();
  assert.equal(get('replay-result').hidden, true); assert.match(get('replay-status').textContent, /没有严格过去模型/);
  assert.equal(get('replay-run').disabled, false);

  const newest = {id: 'unified:20261015_095549_25eab142', run_id: '20261015_095549_25eab142', mode: 'unified', year: 2021,
    backend: 'sb3', compatible: true, normalizer_available: true, execution_version: 'strict-t1-open-v1', feature_adjust: 'hfq',
    ranges: {train: [2006, 2021], val: [2022, 2023], test: [2024, 2025]}, reward_params: {lambda_turnover: .002, lambda_leverage: 0}};
  const older = {...newest, id: 'unified:20261014_185149_8e273f02', run_id: '20261014_185149_8e273f02', reward_params: {lambda_leverage: .01}};
  const customSmoke = {...newest, id: 'unified:strict_real', run_id: 'strict_real', created_at: '2026-10-13T17:46:24'};
  const catalog = {ok: true, symbols: ['000001', '600519'], models: [...legacy.catalog.models, older, newest, customSmoke]};
  const daily = legacy.result.daily.map((day, i) => ({...day, action: i === 2 ? null : 1,
    trade_ok: i === 1, executed_action: i === 0 ? null : 1, execution_signal_date: i === 0 ? null : legacy.result.daily[i - 1].date,
    blocked_reason: i === 0 ? '首日空仓，仅产生信号' : i === 2 ? '现金不足' : '', position_value: 1000,
    cost_basis: 10.01, unrealized_pnl: 100, fee: i === 1 ? 5 : 0, slippage_cost: i === 1 ? 1 : 0,
    drawdown: 0, daily_return: i === 1 ? .1 : 0}));
  const trades = [{signal_date: daily[0].date, date: daily[1].date, side: 'buy', open_price: 10, price: 10.01, shares: 100, notional: 1001, fee: 5}];
  const result = {...legacy.result, model: newest, execution_version: newest.execution_version, temporal_risk: false, daily, trades,
    summary: {...legacy.result.summary, max_drawdown: .12, average_position: .6, holding_days: 2, total_fee: 5, slippage_cost: 1, cost_drag_bps: 40}};
  const s = harness(catalog, result), g = s.get;
  await s.context.window.scaleReplay.open();
  assert.equal(g('replay-model').value, newest.id, '首次进入应优先选最新综合模型');
  assert.equal(g('replay-start').value, '2024-01-01'); assert.equal(g('replay-end').value, '2025-12-31');
  assert.match(g('replay-model-note').textContent, /仓位惩罚 0/);
  assert.match(g('replay-execution-note').textContent, /T\+1/);
  g('replay-range-val').onclick(); assert.equal(g('replay-start').value, '2022-01-01');
  g('replay-range-test').onclick(); assert.equal(g('replay-start').value, '2024-01-01');
  await s.context.window.scaleReplay.selectModel(older.id); assert.equal(g('replay-model').value, older.id);
  await g('replay-latest').onclick(); assert.equal(g('replay-model').value, newest.id);
  await s.submit();
  assert.equal(g('replay-result').hidden, false);
  assert.equal(g('replay-daily').children[0].children.length, 17);
  assert.equal(g('replay-trades').children[0].children.length, 10);
  assert.equal(g('replay-trades').children[0].children[0].textContent, daily[0].date);
  assert.equal(g('replay-trades').children[0].children[1].textContent, daily[1].date);
  assert.equal(s.state.option.series[3].data[0][0], daily[1].date, '成交标记必须在执行日');
  assert.equal(s.state.option.series[8].name, '严格恒买基准', '严格基准应与旧买入持有基线分开标注');
  g('replay-trades').children[0].onclick();
  assert.match(g('replay-detail').textContent, /来源信号日 20230102/);
  assert.match(g('replay-detail').textContent, /浮动盈亏/);
  g('replay-model').value = older.id; g('replay-symbol').value = '000001';
  g('replay-export-daily').onclick(); g('replay-export-trades').onclick(); g('replay-export-json').onclick();
  assert.equal(s.downloads.length, 3);
  assert.ok(s.downloads.every(d => d.filename.includes('20261015_095549_25eab142_600519')), '导出应绑定已完成结果，不读取后来改动的控件');
  const csv = await s.downloads[0].blob.text();
  assert.match(csv, /仓位比例_0至1/); assert.match(csv, /20230102/); assert.match(csv, /600519/);
  const saved = JSON.parse(await s.downloads[2].blob.text()); assert.equal(saved.model.id, newest.id);
  result.daily[0].blocked_reason = '=MALICIOUS()'; g('replay-export-daily').onclick();
  assert.match(await s.downloads.at(-1).blob.text(), /'=MALICIOUS\(\)/, '导出文本应防止电子表格公式注入');
  s.state.failure = true; await s.submit();
  const count = s.downloads.length; g('replay-export-json').onclick(); assert.equal(s.downloads.length, count, '失败后不得导出上次结果');

  const legacyUnified = {...newest, execution_version: 'legacy-same-day-close-v1', feature_adjust: 'raw(as-cached)'};
  const legacyUnifiedTest = harness({...catalog, models: [legacyUnified]}, {...result, model: legacyUnified, execution_version: legacyUnified.execution_version});
  await legacyUnifiedTest.context.window.scaleReplay.open();
  await legacyUnifiedTest.submit();
  assert.equal(legacyUnifiedTest.get('replay-result').hidden, false, '旧综合模型完整执行版本不能被误判为未知版本');
  assert.match(legacyUnifiedTest.get('replay-execution-note').textContent, /同日收盘/);

  const stale = {...newest}; delete stale.execution_version;
  const oldServer = harness({...catalog, models: [stale]}, result);
  await oldServer.context.window.scaleReplay.open(); await oldServer.submit();
  assert.equal(oldServer.state.calls.filter(c => c.url === '/api/replay').length, 0, '旧后端必须阻止综合模型错误撮合');
  assert.match(oldServer.get('replay-status').textContent, /手动重启/);

  const deep = harness(catalog, result, '?model_id=' + encodeURIComponent(older.id) + '&symbol=600519&start=2024-06-01&end=2024-09-30');
  await deep.context.window.scaleReplay.open();
  assert.equal(deep.get('replay-model').value, older.id); assert.equal(deep.get('replay-symbol').value, '600519');
  assert.equal(deep.get('replay-start').value, '2024-06-01'); assert.equal(deep.get('replay-end').value, '2024-09-30');
  const caModel = {...newest, id: 'unified:ca_fixture', run_id: 'ca_fixture', execution_version: 'strict-t1-open-ca-v2'};
  const caSource = {corporate_actions_file: 'fixture/data/corporate_actions/600519.json', corporate_actions_sha256: 'a'.repeat(64),
    feature_sha256: 'b'.repeat(64), execution_sha256: 'c'.repeat(64),
    training_source_audit: {status: 'matched', feature: {status: 'matched'}, execution: {status: 'matched'}, corporate_actions: {status: 'matched'}},
    corporate_actions_coverage: {start: '20230101', end: '20231231', status: 'verified'}};
  const caEvent = {event_id: 'synthetic-event', stage: 'ex_entitlement', date: daily[1].date,
    record_date: daily[0].date, ex_date: daily[1].date, pay_date: daily[2].date, listing_date: daily[2].date,
    sources: ['https://invalid.example/fixture/distribution']};
  const caDaily = daily.map((day, i) => ({...day, available_shares: 100, locked_shares: i === 1 ? 50 : 0,
    dividend_receivable: i === 1 ? 100 : 0, dividend_income: i ? 100 : 0, realized_pnl: 0,
    limit_reference: i ? 6 : null, limit_up: i ? 6.6 : null, limit_down: i ? 5.4 : null,
    corporate_actions: i === 1 ? [caEvent] : []}));
  const caResult = {...result, model: caModel, execution_version: caModel.execution_version,
    training_execution_version: caModel.execution_version, execution_mode: 'default', selection: 'manual_research',
    source: caSource, tax_policy: 'gross_before_personal_dividend_tax', daily: caDaily,
    corporate_actions: [caEvent], corporate_actions_ledger: {events: [caEvent]},
    summary: {...result.summary, dividend_income: 100, final_dividend_receivable: 0, final_available_shares: 100, final_locked_shares: 0}};
  const ca = harness({...catalog, models: [caModel, newest]}, caResult);
  await ca.context.window.scaleReplay.open();
  await ca.context.window.scaleReplay.selectModel(caModel.id);
  assert.equal(ca.get('replay-ca-research').disabled, true, '原生 v2 不允许研究覆写');
  await ca.submit();
  assert.equal(ca.get('replay-result').hidden, false, '原生 v2 默认按模型契约回放');
  assert.equal(ca.state.calls.at(-1).payload.execution_mode, undefined, '默认回放不发送研究执行覆写');
  assert.equal(ca.state.option.series[8].name, '严格恒买基准');
  assert.equal(ca.get('replay-daily').children[0].children.length, 22);
  assert.equal(ca.get('replay-ca-head-receivable').hidden, false);
  assert.match(ca.get('replay-ca-note').textContent, /SHA-256.*a{64}/);
  assert.match(ca.get('replay-ca-note').textContent, /冻结来源核对：matched/);
  assert.match(ca.get('replay-ca-note').textContent, /当前未复权 SHA-256：c{64}/);
  ca.get('replay-next').onclick();
  assert.match(ca.get('replay-detail').textContent, /股息应收 ¥100/);
  assert.match(ca.get('replay-detail').textContent, /待上市锁股 50/);
  assert.match(ca.get('replay-detail').textContent, /除权确权/);
  assert.match(ca.get('replay-detail').textContent, /invalid\.example\/fixture\/distribution/);
  ca.get('replay-export-daily').onclick(); ca.get('replay-export-json').onclick();
  const caCsv = await ca.downloads[0].blob.text();
  assert.match(caCsv, /可卖股数/); assert.match(caCsv, /待上市锁股数/); assert.match(caCsv, /股息应收_CNY/);
  assert.match(caCsv, /公司行为完整事件_JSON/); assert.match(caCsv, /synthetic-event/);
  assert.match(caCsv, /invalid\.example\/fixture\/distribution/); assert.match(caCsv, /a{64}/);
  assert.match(caCsv, /冻结训练来源核对_JSON/); assert.match(caCsv, /b{64}/); assert.match(caCsv, /c{64}/);
  const caSaved = JSON.parse(await ca.downloads[1].blob.text());
  assert.equal(caSaved.corporate_actions[0].sources[0], caEvent.sources[0]);
  assert.equal(caSaved.source.corporate_actions_sha256, caSource.corporate_actions_sha256);

  const researchResult = {...caResult, model: newest, training_execution_version: newest.execution_version,
    execution_mode: 'corporate_actions_research', selection: 'corporate_actions_research',
    warnings: ['强警告：旧 policy 仍由错误 v1 奖励训练，不是已修复重训模型']};
  const researchLegacy = {...legacyUnified, id: 'unified:legacy_fixture', run_id: 'legacy_fixture'};
  const research = harness({...catalog, models: [newest, caModel, researchLegacy]}, researchResult);
  await research.context.window.scaleReplay.open();
  await research.context.window.scaleReplay.selectModel(newest.id);
  assert.equal(research.get('replay-ca-research').disabled, false);
  assert.equal(research.get('replay-ca-research').checked, false, '研究修正必须默认不勾选');
  research.get('replay-ca-research').checked = true;
  research.get('replay-ca-research').listeners.change();
  assert.match(research.get('replay-execution-note').textContent, /不是已修复重训模型/);
  await research.submit();
  assert.equal(research.state.calls.at(-1).payload.execution_mode, 'corporate_actions_research');
  assert.equal(research.get('replay-result').hidden, false, '显式研究模式应接受训练 v1 / 执行 v2');
  assert.match(research.get('replay-warnings').textContent, /错误 v1 奖励/);
  assert.match(research.get('replay-warnings').className, /replay-risk/, '旧奖励研究覆写必须醒目警告');
  research.state.result = {...researchResult, training_execution_version: caModel.execution_version};
  await research.submit();
  assert.equal(research.get('replay-result').hidden, true, '研究模式不能把旧模型伪标成已按 v2 训练');
  assert.match(research.get('replay-status').textContent, /缺少旧模型研究执行覆写标识/);
  research.state.result = researchResult;
  await research.context.window.scaleReplay.selectModel(caModel.id);
  assert.equal(research.get('replay-ca-research').checked, false, '切换非 v1 应清除研究覆写');
  assert.equal(research.get('replay-ca-research').disabled, true);
  await research.context.window.scaleReplay.selectModel(newest.id);
  assert.equal(research.get('replay-ca-research').checked, false, '切回 v1 不可隐式恢复勾选');
  await research.submit();
  assert.equal(research.state.calls.at(-1).payload.execution_mode, undefined);
  assert.equal(research.get('replay-result').hidden, true, '未勾选时拒绝后端擅自执行 v2');
  research.get('replay-ca-research').checked = true;
  research.state.result = {...researchResult, selection: 'manual_research'};
  await research.submit();
  assert.equal(research.get('replay-result').hidden, true, '研究响应缺标识必须拒绝');
  assert.match(research.get('replay-status').textContent, /缺少旧模型研究执行覆写标识/);
  research.state.result = {...researchResult, execution_version: newest.execution_version};
  await research.submit();
  assert.equal(research.get('replay-result').hidden, true, '研究响应不得降级为 v1');
  await research.context.window.scaleReplay.selectModel(researchLegacy.id);
  assert.equal(research.get('replay-ca-research').checked, false);
  assert.equal(research.get('replay-ca-research').disabled, true, 'legacy 不允许研究覆写');
  await research.context.window.scaleReplay.selectModel(caModel.id);
  research.state.result = caResult;
  await research.submit();
  assert.equal(research.get('replay-ca-note').hidden, false);
  research.state.result = {...result, model: newest};
  await research.context.window.scaleReplay.selectModel(newest.id);
  await research.submit();
  assert.equal(research.get('replay-result').hidden, false);
  assert.equal(research.get('replay-ca-note').hidden, true, '回到 v1 必须隐藏上次公司行为审计块');
  assert.equal(research.get('replay-ca-head-receivable').hidden, true);
  assert.equal(research.get('replay-daily').children[0].children.length, 17);

  const executionSelect = /<select id="unified-execution">([\s\S]*?)<\/select>/.exec(html);
  assert.ok(executionSelect, '训练页必须有新 run 执行口径选择器');
  const selectedExecution = /<option value="([^"]+)" selected>/.exec(executionSelect[1]);
  assert.equal(selectedExecution[1], 'strict-ca', '新训练必须默认公司行为 v2');
  assert.match(executionSelect[1], /value="strict"[^>]*>[^<]*遗漏公司行为/);
  assert.match(executionSelect[1], /value="legacy"[^>]*>[^<]*遗漏公司行为/);
  assert.match(html, /账本未核验、覆盖不足或存在不支持事件会拒绝/);
  const inlineScripts = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/g)];
  inlineScripts.forEach((match, index) => new vm.Script(match[1], {filename: `index-inline-${index}`}));
  const controlScript = /async function unifiedCtl\([\s\S]*?\n}\n(?=refreshUnified\(\);)/.exec(html);
  assert.ok(controlScript, '测试应执行页面真实控制函数，不拷贝实现');
  const trainElements = new Map([
    ['unified-steps', {value: '100'}], ['unified-run', {value: 'fixture-existing-run'}],
    ['unified-execution', {value: selectedExecution[1]}], ['unified-msg', {}]
  ]);
  const trainCalls = [];
  let confirmed = true, refreshes = 0;
  const trainingContext = {
    $: id => trainElements.get(id), confirm: () => confirmed,
    fetch: async (url, options) => {
      trainCalls.push({url, payload: JSON.parse(options.body)});
      return {json: async () => ({ok: true, msg: '仅测试桩，不启动训练'})};
    },
    refreshUnified: () => {refreshes++;}
  };
  vm.createContext(trainingContext);
  vm.runInContext(controlScript[0], trainingContext);
  assert.equal(trainCalls.length, 0, '载入函数和选择口径不能自动请求训练');
  await trainingContext.unifiedCtl('start', false);
  assert.equal(trainCalls.at(-1).url, '/api/unified/control');
  assert.equal(trainCalls.at(-1).payload.execution, 'strict-ca');
  assert.equal(trainCalls.at(-1).payload.timesteps, 100);
  trainElements.get('unified-execution').value = 'strict';
  await trainingContext.unifiedCtl('start', false);
  assert.equal(trainCalls.at(-1).payload.execution, 'strict', '旧 v1 必须显式选择');
  trainElements.get('unified-execution').value = 'legacy';
  await trainingContext.unifiedCtl('start', true);
  assert.equal(trainCalls.at(-1).payload.resume, true);
  assert.equal(trainCalls.at(-1).payload.run_id, 'fixture-existing-run');
  assert.equal(trainCalls.at(-1).payload.execution, undefined, '续训必须继承冻结版本，不发送执行覆写');
  await trainingContext.unifiedCtl('resume');
  assert.equal(trainCalls.at(-1).payload.execution, undefined);
  confirmed = false;
  const beforeCancel = trainCalls.length;
  await trainingContext.unifiedCtl('start', false);
  assert.equal(trainCalls.length, beforeCancel, '取消确认不能发送训练控制请求');
  assert.equal(refreshes, beforeCancel);

  console.log('回放 UI 测试通过：旧版回归、公司行为审计/冻结指纹导出、显式研究保护、训练新建默认v2与续训继承');
})().catch(error => {console.error(error); process.exitCode = 1;});

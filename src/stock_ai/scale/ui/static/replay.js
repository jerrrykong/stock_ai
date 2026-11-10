/* 回放独立状态：不接入训练状态定时刷新。 */
(() => {
  'use strict';
  const el = id => document.getElementById('replay-' + id);
  const number = value => value === null || value === undefined || !Number.isFinite(Number(value)) ? '-' : Number(value).toLocaleString('zh-CN', {maximumFractionDigits: 2});
  const percent = value => value === null || value === undefined || !Number.isFinite(Number(value)) ? '-' : (Number(value) * 100).toFixed(2) + '%';
  const actionName = value => value === null || value === undefined ? '无' : ['持有 / 不操作', '买入', '卖出'][value] || '未知';
  const selectedModel = () => catalog && catalog.models.find(m => m.id === el('model').value);
  const V1 = 'strict-t1-open-v1', CA_V2 = 'strict-t1-open-ca-v2';
  const strictVersions = new Set([V1, CA_V2]);
  const strict = () => result && strictVersions.has(result.execution_version);
  const corporateActions = () => result && result.execution_version === CA_V2;
  const researchEligible = model => model && model.mode === 'unified' && model.execution_version === V1 && model.compatible && model.normalizer_available;
  const eventStages = {record: '登记', ex_entitlement: '除权确权', cash_payment: '派息', stock_listing: '股份上市'};
  const eventSummary = events => (events || []).map(event => `${event.event_id} / ${eventStages[event.stage] || event.stage}`).join('；');
  const executedAction = day => day.executed_action === undefined ? day.action : day.executed_action;
  const controls = ['run', 'catalog-refresh', 'latest', 'symbol', 'start', 'end', 'model', 'range-val', 'range-test', 'ca-research'];
  let catalog = null, catalogLoading = null, result = null, chart = null, selectedDay = 0, busy = false;

  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || '请求失败');
    return data;
  }
  function modelTime(model) {
    if (model.created_at) {
      const time = Date.parse(model.created_at);
      if (Number.isFinite(time)) return time;
    }
    const stamp = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(model.run_id || '');
    return stamp ? Date.parse(`${stamp[1]}-${stamp[2]}-${stamp[3]}T${stamp[4]}:${stamp[5]}:${stamp[6]}`) : 0;
  }
  const newestFirst = (a, b) => modelTime(b) - modelTime(a) || b.run_id.localeCompare(a.run_id);
  function latestUnified() {
    return catalog && catalog.models.filter(m => m.mode === 'unified' && m.compatible && m.normalizer_available)
      .sort(newestFirst)[0];
  }
  function applyRange(split, onlyIfEmpty = false) {
    const model = selectedModel();
    const range = model && model.ranges && model.ranges[split];
    if (!range || (onlyIfEmpty && (el('start').value || el('end').value))) return;
    el('start').value = `${range[0]}-01-01`;
    el('end').value = `${range[1]}-12-31`;
    selectionWarning();
  }
  function selectionWarning() {
    const id = el('model').value;
    const model = selectedModel();
    const unified = model && model.mode === 'unified';
    const version = model && model.execution_version;
    const eligible = researchEligible(model);
    const research = el('ca-research');
    if (!eligible) research.checked = false;
    research.disabled = busy || !eligible;
    const year = Number(el('start').value.slice(0, 4));
    const risk = model && year && model.year >= year;
    const box = el('selection-warning');
    box.hidden = id === 'auto';
    box.className = 'replay-warning' + (risk ? ' replay-risk' : '');
    box.textContent = '研究回放：手选模型不能证明样本外或无泄漏。' +
      (risk ? ` 模型训练结束年份 ${model.year} 不早于起始年 ${year}，存在训练重叠 / 前视风险。` : '') +
      (unified ? ` 综合模型 ${model.run_id}：${model.compatible ? '相对特征契约兼容' : '契约不兼容，禁止回放'}。${model.warning || ''}` : '') +
      (model && !unified && model.symbol !== el('symbol').value ? ' 模型训练股票与回放股票不同。' : '') +
      (model && !model.normalizer_available ? ' 冻结 normalizer 缺失，运行会报错，不会重新拟合。' : '');
    let note = '旧单元 / legacy：同日完整行情生成信号，同日收盘含滑点和费用模拟成交；末日仅估值。';
    if (unified && strictVersions.has(version)) {
      note = '严格回放：T 日收盘信号 → T+1 交易日真实开盘撮合。特征用后复权，成交和持仓估值用未复权价；信号与成交分开显示。首日空仓，末日仍处理前日信号，但不生成新信号、不强制清仓。';
      if (version === CA_V2 || research.checked) {
        note += ' 公司行为 v2：必须具有显式核验覆盖账本；展示税前股息应收、可卖与待上市锁股、派息及上市事件。未覆盖或不支持的行为拒绝回放。';
      } else {
        note += ' 旧 v1 遗漏公司行为，训练奖励及历史收益有缺陷，不能视为投资总回报。';
      }
      if (research.checked) note += ' 强警告：仅研究执行覆写；旧 policy 仍由错误 v1 奖励训练，不是已修复重训模型，不修改模型或 manifest。';
    } else if (unified && !version) {
      note = '当前服务仍为旧版回放后端，缺少执行版本信息。为防止综合模型被错误地按旧撮合回放，本页已禁止运行；请手动重启 UI 服务后刷新页面。';
    } else if (unified && version !== 'legacy-same-day-close-v1') {
      note = '不支持的执行版本：' + version + '。请使用匹配的回放服务，不会降级为同日成交。';
    }
    el('execution-note').textContent = note + ' 每次回放以模型配置初始资金、空仓起步；跨年区间连续持仓，不等于逐年独立评估的平均值。仅使用本地缓存。';
    const params = model && model.reward_params;
    el('model-note').textContent = model ? `已选：${model.id} · ${unified ? '跨股票共享模型' : '单股票年度模型'} · ${version || (unified ? '旧服务待重启' : 'legacy')} · 特征复权 ${model.feature_adjust || '-'}${params ? ` · 换手惩罚 ${params.lambda_turnover} / 仓位惩罚 ${params.lambda_leverage}` : ''}` : '自动模式只选择同代码的历史单元模型，不会自动选综合模型。';
    ['val', 'test'].forEach(split => {
      const range = model && model.ranges && model.ranges[split];
      el('range-' + split).textContent = (split === 'val' ? '验证期' : '测试期') + (range ? ` ${range.join('–')}` : '（需选综合模型）');
      el('range-' + split).disabled = busy || !range;
    });
  }
  function modelLabel(model) {
    if (model.mode !== 'unified') return `${model.id} / ${model.backend}${model.normalizer_available ? '' : ' / 缺少 norm'}`;
    const params = model.reward_params;
    return `综合 ${model.run_id} / ${strictVersions.has(model.execution_version) ? 'T+1 开盘' + (model.execution_version === CA_V2 ? ' · 公司行为 v2' : ' · 旧 v1') : model.execution_version || '旧服务'}${params ? ` / λ仓位=${params.lambda_leverage}` : ''}${model.compatible ? '' : ' / 不兼容'}${model.normalizer_available ? '' : ' / 缺少 norm'}`;
  }
  async function loadCatalog() {
    if (catalogLoading) return catalogLoading;
    if (busy) return catalog;
    el('catalog-refresh').disabled = true;
    catalogLoading = (async () => {
      try {
        const data = await request('/api/replay/catalog');
        const previousSymbol = el('symbol').value, previousModel = el('model').value;
        const firstLoad = !catalog;
        catalog = data;
        el('symbol').replaceChildren(...data.symbols.map(s => new Option(s, s)));
        if (data.symbols.includes(previousSymbol)) el('symbol').value = previousSymbol;
        const unified = data.models.filter(m => m.mode === 'unified').sort(newestFirst);
        const units = data.models.filter(m => m.mode !== 'unified');
        const group = (label, models) => {
          const node = document.createElement('optgroup'); node.label = label;
          models.forEach(model => node.appendChild(new Option(modelLabel(model), model.id)));
          return node;
        };
        el('model').replaceChildren(new Option('自动：同代码且年份严格早于起始年（旧单元模型）', 'auto'),
          group('综合模型 / unified（最新优先）', unified), group('旧单股票年度模型', units));
        if (!firstLoad && (previousModel === 'auto' || data.models.some(m => m.id === previousModel))) el('model').value = previousModel;
        else if (latestUnified()) el('model').value = latestUnified().id;
        el('status').className = data.symbols.length ? '' : 'err';
        el('status').textContent = `已加载 ${data.symbols.length} 个缓存股票、${unified.length} 个综合模型。选择股票和区间后点击“运行回放”。不会联网、触发训练或更改模型。`;
        applyRange('test', true);
        selectionWarning();
        return data;
      } catch (error) {
        el('status').className = 'err'; el('status').textContent = error.message;
        return null;
      } finally { el('catalog-refresh').disabled = false; catalogLoading = null; }
    })();
    return catalogLoading;
  }
  async function selectModel(id) {
    if (busy) return;
    if (!await loadCatalog()) return;
    const model = id ? catalog.models.find(m => m.id === id) : latestUnified();
    if (!model) {
      el('status').className = 'err';
      el('status').textContent = id ? `模型 ${id} 无可用 checkpoint，无法回放。请选择已保存模型。` : '没有可用的综合模型及冻结 normalizer。';
      return;
    }
    el('model').value = model.id;
    applyRange('test', true);
    selectionWarning();
    el('status').className = '';
    el('status').textContent = `已选 ${model.id}。请选股票及日期后运行；切换模型不会重新训练，也不会自动执行回放。`;
  }
  function fillTable(target, records, values) {
    const fragment = document.createDocumentFragment();
    records.forEach((record, index) => {
      const row = document.createElement('tr'); row.dataset.index = index;
      values(record).forEach(value => {
        const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell);
      });
      row.onclick = () => selectDay(result.daily.findIndex(day => day.date === record.date), true);
      fragment.appendChild(row);
    });
    if (!records.length) {
      const row = document.createElement('tr'), cell = document.createElement('td');
      cell.colSpan = target === el('daily') ? (corporateActions() ? 22 : 17) : 10;
      cell.textContent = '无实际成交：信号不等于成交。'; row.appendChild(cell); fragment.appendChild(row);
    }
    target.replaceChildren(fragment);
  }
  function tradeSlip(trade) {
    if (trade.slippage_cost != null) return trade.slippage_cost;
    return trade.open_price == null ? null : Math.abs(trade.price - trade.open_price) * trade.shares;
  }
  function selectDay(index, focus = false) {
    if (!result) return;
    selectedDay = Math.max(0, Math.min(result.daily.length - 1, Number(index)));
    const day = result.daily[selectedDay];
    el('day').value = selectedDay;
    el('prev').disabled = selectedDay === 0; el('next').disabled = selectedDay === result.daily.length - 1;
    const lines = [
      `${day.date} · 收盘信号：${actionName(day.action)}${day.action == null ? '（区间结束，不产生新信号）' : ''}`,
      `当日${strict() ? '开盘' : '收盘'}执行：${actionName(executedAction(day))} · 来源信号日 ${day.execution_signal_date || '-'} · ${day.trade_ok ? '已模拟成交' : day.blocked_reason || '无成交'}`,
      `未复权行情：开 ${number(day.open)} / 高 ${number(day.high)} / 低 ${number(day.low)} / 收 ${number(day.close)} / 量 ${number(day.volume)}`,
      `现金 ¥${number(day.cash)} / 持股 ${number(day.shares)} 股 / 仓位 ${percent(day.position)} / 持仓市值 ¥${number(day.position_value)}`,
      `成本 / 股 ¥${number(day.cost_basis)} / 浮动盈亏 ¥${number(day.unrealized_pnl)} / 权益 ¥${number(day.equity)} / 基线 ¥${number(day.baseline)}`,
      `日收益 ${percent(day.daily_return)} / 回撤 ${percent(day.drawdown)} / 当日手续费 ¥${number(day.fee)} / 滑点成本 ¥${number(day.slippage_cost)}`,
      ...(corporateActions() ? [
        `公司行为：可卖 ${number(day.available_shares)} 股 / 待上市锁股 ${number(day.locked_shares)} 股 / 股息应收 ¥${number(day.dividend_receivable)} / 税前股息累计 ¥${number(day.dividend_income)}`,
        `除权涨跌停参考 ¥${number(day.limit_reference)} / 涨停 ¥${number(day.limit_up)} / 跌停 ¥${number(day.limit_down)}`,
        `当日事件：${eventSummary(day.corporate_actions) || '无'}；总权益含应收，浮盈不含单列股息。`,
        ...(day.corporate_actions || []).map(event => `${event.event_id} · 登记 ${event.record_date} / 除权 ${event.ex_date} / 派息 ${event.pay_date || '-'} / 上市 ${event.listing_date || '-'} · 来源 ${(event.sources || []).join(' ')}`)
      ] : []),
      ...result.trades.filter(t => t.date === day.date).map(t => `${t.signal_date || t.date} 信号 → ${t.date} ${t.side === 'buy' ? '买入' : '卖出'} ${number(t.shares)} 股，成交价 ¥${number(t.price)}，费用 ¥${number(t.fee)}，已实现盈亏 ¥${number(t.realized_pnl)}`)
    ];
    el('detail').textContent = lines.join('\n');
    el('daily').querySelectorAll('tr').forEach((row, i) => row.classList.toggle('selected', i === selectedDay));
    if (focus && chart) {
      const total = result.daily.length;
      chart.dispatchAction({type: 'dataZoom', start: Math.max(0, (selectedDay - 25) / total * 100), end: Math.min(100, (selectedDay + 26) / total * 100)});
      chart.dispatchAction({type: 'showTip', seriesIndex: 0, dataIndex: selectedDay});
    }
  }
  function renderChart() {
    if (!window.echarts) throw new Error('本地 ECharts 静态资源未加载，请检查服务静态资源。');
    if (!chart) {
      chart = echarts.init(el('chart'), 'dark');
      chart.on('click', event => {
        if (!result) return;
        const date = Array.isArray(event.value) && typeof event.value[0] === 'string' ? event.value[0] : event.name;
        const index = result.daily.findIndex(d => d.date === date);
        if (index >= 0) selectDay(index);
      });
      chart.on('updateAxisPointer', event => {
        if (!result) return;
        const info = event.axesInfo && event.axesInfo[0];
        if (info) {
          const index = typeof info.value === 'number' ? info.value : result.daily.findIndex(d => d.date === info.value);
          if (index >= 0) selectDay(index);
        }
      });
    }
    const rows = result.daily, dates = rows.map(d => d.date);
    const baselineLabel = strict() ? '严格恒买基准' : '买入持有基线';
    const scatter = (name, action, filled) => ({
      name, type: 'scatter', xAxisIndex: 0, yAxisIndex: 0,
      symbol: filled ? 'circle' : 'triangle', symbolRotate: action === 2 && !filled ? 180 : 0,
      symbolSize: filled ? 10 : 15, symbolOffset: [0, action === 1 ? 16 : -16],
      itemStyle: {color: filled ? (action === 1 ? '#ff6868' : '#38d99b') : 'transparent', borderColor: action === 1 ? '#ff6868' : '#38d99b', borderWidth: 2},
      data: filled ? result.trades.filter(t => t.side === (action === 1 ? 'buy' : 'sell')).map(t => [t.date, t.price]) : rows.filter(d => d.action === action).map(d => [d.date, d.close]),
      z: filled ? 7 : 6
    });
    chart.setOption({
      backgroundColor: '#1a2029', animation: false,
      legend: {top: 0, data: ['K线', '买信号', '卖信号', '买成交', '卖成交', '权益', baselineLabel]},
      tooltip: {trigger: 'axis', axisPointer: {type: 'cross'}},
      axisPointer: {link: [{xAxisIndex: 'all'}]},
      grid: [{left: 85, right: 45, top: 70, height: 280}, {left: 85, right: 45, top: 395, height: 80}, {left: 85, right: 45, top: 515, height: 70}, {left: 85, right: 45, top: 625, height: 110}],
      xAxis: [0, 1, 2, 3].map(i => ({type: 'category', gridIndex: i, data: dates, boundaryGap: true, axisLabel: {show: i === 3}, axisPointer: {show: true}})),
      yAxis: [{name: '价格 ¥', scale: true, gridIndex: 0}, {name: '成交量', scale: true, gridIndex: 1}, {name: '仓位 %', min: 0, max: 100, gridIndex: 2}, {name: '权益 / 基线 ¥', scale: true, gridIndex: 3}],
      dataZoom: [{type: 'inside', xAxisIndex: [0, 1, 2, 3], start: 0, end: 100}, {type: 'slider', xAxisIndex: [0, 1, 2, 3], bottom: 10, start: 0, end: 100}],
      series: [
        {name: 'K线', type: 'candlestick', data: rows.map(d => [d.open, d.close, d.low, d.high]), itemStyle: {color: '#ef5350', color0: '#22c997', borderColor: '#ef5350', borderColor0: '#22c997'}},
        scatter('买信号', 1, false), scatter('卖信号', 2, false), scatter('买成交', 1, true), scatter('卖成交', 2, true),
        {name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: rows.map(d => ({value: d.volume, itemStyle: {color: d.close >= d.open ? '#ef5350' : '#22c997'}}))},
        {name: '仓位 %', type: 'line', step: 'end', showSymbol: false, xAxisIndex: 2, yAxisIndex: 2, data: rows.map(d => d.position * 100), lineStyle: {color: '#ac91ff'}},
        {name: '权益', type: 'line', showSymbol: false, xAxisIndex: 3, yAxisIndex: 3, data: rows.map(d => d.equity), lineStyle: {color: '#65b6ff'}},
        {name: baselineLabel, type: 'line', showSymbol: false, xAxisIndex: 3, yAxisIndex: 3, data: rows.map(d => d.baseline), lineStyle: {color: '#f5bd59', type: 'dashed'}}
      ]
    }, true);
    chart.resize();
  }
  function renderSummary() {
    const s = result.summary;
    const pairs = [
      ['模型', result.model.id], ['股票 / 实际区间', `${result.symbol} · ${result.actual_range.start}–${result.actual_range.end}`],
      ['执行版本', result.execution_version || 'legacy'], ['交易日 / 成交', `${s.bars} 日 / ${s.trades} 笔`],
      ['初始资金', '¥' + number(result.initial_cash)], ['期末权益', '¥' + number(s.final_equity)],
      ['策略收益', percent(s.total_return)], [strict() ? '严格恒买基准' : '买入持有', percent(s.baseline_return)],
      ['超额', number((s.excess_return == null ? s.total_return - s.baseline_return : s.excess_return) * 100) + ' 个百分点'],
      ['最大回撤', percent(s.max_drawdown)], ['平均仓位', percent(s.average_position)],
      ['持仓天数', `${s.holding_days == null ? '-' : s.holding_days} / ${s.bars}`],
      ['手续费合计', '¥' + number(s.total_fee)], ['滑点成本估计', '¥' + number(s.slippage_cost)],
      ['费用拖累', number(s.cost_drag_bps) + ' bps（仅手续费）'], ['区间前历史', `${result.warmup_rows} 行`]
    ];
    if (corporateActions()) {
      pairs.push(['训练执行版本', result.training_execution_version],
        ['回放类型', result.selection === 'corporate_actions_research' ? '旧模型研究执行覆写（未重训）' : '模型原生 v2'],
        ['税前股息累计', '¥' + number(s.dividend_income)], ['期末股息应收', '¥' + number(s.final_dividend_receivable)],
        ['期末可卖 / 锁股', `${number(s.final_available_shares)} / ${number(s.final_locked_shares)} 股`]);
    }
    const source = result.source || {}, coverage = source.corporate_actions_coverage;
    const audit = source.training_source_audit;
    const auditText = audit ? `\n冻结来源核对：${audit.status}；后复权 ${audit.feature.status} / 未复权 ${audit.execution.status} / 账本 ${audit.corporate_actions.status}\n当前后复权 SHA-256：${source.feature_sha256}\n当前未复权 SHA-256：${source.execution_sha256}` : '';
    el('ca-note').hidden = !corporateActions();
    el('ca-note').textContent = corporateActions() ? `公司行为账本：${source.corporate_actions_file || '-'}\nSHA-256：${source.corporate_actions_sha256 || '-'}\n核验覆盖：${coverage ? coverage.start + '–' + coverage.end + ' / ' + coverage.status : '-'}${auditText}\n税务口径：税前，不含个人差异化股息税。完整 JSON 保留全部事件与来源链接；逐日 CSV 保留事件 JSON、来源及账本指纹。` : '';
    const fragment = document.createDocumentFragment();
    pairs.forEach(([label, value]) => {
      const span = document.createElement('span'), bold = document.createElement('b');
      span.textContent = label + ' '; bold.textContent = value; span.appendChild(bold); fragment.appendChild(span);
    });
    el('summary').replaceChildren(fragment);
  }
  function setBusy(value) {
    busy = value; controls.forEach(id => { el(id).disabled = value; });
    if (!value) selectionWarning();
  }
  async function run(event) {
    event.preventDefault();
    if (busy) return;
    const model = selectedModel();
    if (model && model.mode === 'unified' && !strictVersions.has(model.execution_version) && model.execution_version !== 'legacy-same-day-close-v1') {
      el('status').className = 'err';
      el('status').textContent = '综合模型执行版本不可用：请手动重启 UI 服务加载新后端，再刷新本页；禁止使用旧撮合代替严格回放。';
      return;
    }
    const research = Boolean(researchEligible(model) && el('ca-research').checked);
    setBusy(true); result = null;
    el('result').hidden = true; el('warnings').hidden = true; el('ca-note').hidden = true; el('summary').replaceChildren();
    el('status').className = ''; el('status').textContent = '正在读取缓存和冻结参数，按所选模型执行语义推理…';
    try {
      const payload = {symbol: el('symbol').value, start: el('start').value, end: el('end').value, model_id: el('model').value};
      if (research) payload.execution_mode = 'corporate_actions_research';
      result = await request('/api/replay', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      if (!Array.isArray(result.daily) || !result.daily.length) throw new Error('回放没有返回逐日记录。');
      const expectedVersion = research ? CA_V2 : model && model.execution_version;
      if (model && model.mode === 'unified' && result.execution_version !== expectedVersion) throw new Error('响应执行版本与所选模型 / 研究模式不一致，已拒绝展示。');
      if (research && (result.training_execution_version !== V1 || result.selection !== 'corporate_actions_research')) throw new Error('响应缺少旧模型研究执行覆写标识，已拒绝展示。');
      if (!research && result.selection === 'corporate_actions_research') throw new Error('未显式勾选研究执行覆写，已拒绝展示。');
      el('result').hidden = false; el('warnings').hidden = false;
      el('warnings').className = 'replay-warning' + (result.temporal_risk || research ? ' replay-risk' : '');
      el('warnings').textContent = result.warnings.join('\n');
      renderSummary();
      el('baseline-note').textContent = result.baseline_note;
      el('day').max = result.daily.length - 1;
      ['available', 'locked', 'receivable', 'dividend', 'events'].forEach(id => { el('ca-head-' + id).hidden = !corporateActions(); });
      fillTable(el('daily'), result.daily, d => [d.date, actionName(d.action), d.execution_signal_date || '-', actionName(executedAction(d)), d.trade_ok ? '已成交' : d.blocked_reason || '不操作', number(d.cash), number(d.shares), percent(d.position), number(d.position_value), number(d.cost_basis), number(d.unrealized_pnl), number(d.equity), percent(d.daily_return), percent(d.drawdown), number(d.fee), number(d.slippage_cost), number(d.baseline), ...(corporateActions() ? [number(d.available_shares), number(d.locked_shares), number(d.dividend_receivable), number(d.dividend_income), eventSummary(d.corporate_actions) || '无'] : [])]);
      fillTable(el('trades'), result.trades, t => [t.signal_date || t.date, t.date, t.side === 'buy' ? '买入' : '卖出', number(t.open_price), number(t.price), number(t.shares), number(t.notional), number(t.fee), number(tradeSlip(t)), number(t.realized_pnl)]);
      renderChart(); selectDay(0);
      el('status').textContent = '回放完成。点击图表、逐日表或成交表联动查看详情；可导出本次逐日持仓、成交或完整数据。';
    } catch (error) {
      result = null; el('result').hidden = true; el('ca-note').hidden = true; el('status').className = 'err'; el('status').textContent = error.message;
    } finally { setBusy(false); }
  }
  function csvCell(value) {
    if (value === null || value === undefined) return '""';
    let text = String(value);
    if (typeof value === 'string' && /^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replace(/"/g, '""') + '"';
  }
  function download(kind) {
    if (!result || busy) return;
    let content, extension, type;
    if (kind === 'json') {
      content = JSON.stringify(result, null, 2); extension = 'json'; type = 'application/json;charset=utf-8';
    } else {
      const common = [['symbol', '股票'], ['model_id', '模型'], ['execution_version', '执行版本'],
        ['training_execution_version', '训练执行版本'], ['selection', '回放选择模式'], ['execution_mode', '请求执行模式'],
        ['reward_version', '执行奖励版本'], ['training_reward_version', '训练奖励版本'],
        ['corporate_actions_file', '公司行为账本路径'], ['corporate_actions_schema', '公司行为账本结构版本'], ['corporate_actions_sha256', '公司行为账本SHA256'],
        ['corporate_actions_coverage', '公司行为核验覆盖_JSON'], ['tax_policy', '股息税口径'],
        ['feature_sha256', '当前后复权SHA256'], ['execution_sha256', '当前未复权SHA256'],
        ['training_source_audit', '冻结训练来源核对_JSON']];
      const columns = kind === 'daily' ? [
        ['date', '日期'], ['action_name', '收盘信号'], ['execution_signal_date', '执行来源信号日'], ['executed_action_name', '当日执行动作'],
        ['pending_execution_date', '待执行日'], ['signal_status', '信号状态'], ['execution', '是否执行日'],
        ['trade_ok', '是否成交'], ['blocked_reason', '未成交原因'], ['open', '开盘价'], ['high', '最高价'], ['low', '最低价'], ['close', '收盘价'], ['volume', '成交量'],
        ['cash', '现金_CNY'], ['shares', '持股数'], ['position', '仓位比例_0至1'], ['position_value', '持仓市值_CNY'], ['cost_basis', '每股成本_CNY'], ['unrealized_pnl', '浮动盈亏_CNY'],
        ['equity', '权益_CNY'], ['baseline', '基线权益_CNY'], ['daily_return', '日收益率_0至1'], ['drawdown', '回撤_0至1'], ['fee', '当日手续费_CNY'], ['slippage_cost', '当日滑点成本_CNY']
      ] : [['signal_date', '信号日'], ['date', '成交日'], ['side', '方向'], ['open_price', '参考开盘价'], ['price', '成交价'], ['shares', '成交股数'], ['notional', '成交金额_CNY'], ['fee', '手续费_CNY'], ['slippage_cost', '滑点成本_CNY'], ['realized_pnl', '已实现盈亏_CNY'], ['limit_reference', '涨跌停参考价'], ['limit_rate', '涨跌幅限制比例']];
      if (kind === 'daily') columns.push(
        ['available_shares', '可卖股数'], ['locked_shares', '待上市锁股数'], ['dividend_receivable', '股息应收_CNY'],
        ['dividend_income', '税前股息累计_CNY'], ['realized_pnl', '已实现盈亏累计_CNY'],
        ['limit_reference', '涨跌停参考价'], ['limit_up', '涨停价'], ['limit_down', '跌停价'],
        ['corporate_actions_summary', '公司行为摘要'], ['corporate_actions', '公司行为完整事件_JSON']);
      const fields = common.concat(columns), source = result.source || {};
      const rows = (kind === 'daily' ? result.daily : result.trades).map(record => ({
        ...record, symbol: result.symbol, model_id: result.model.id, execution_version: result.execution_version || 'legacy',
        training_execution_version: result.training_execution_version || result.execution_version,
        selection: result.selection, execution_mode: result.execution_mode,
        reward_version: result.reward_version, training_reward_version: result.training_reward_version,
        corporate_actions_file: source.corporate_actions_file, corporate_actions_schema: source.corporate_actions_schema,
        corporate_actions_sha256: source.corporate_actions_sha256,
        feature_sha256: source.feature_sha256, execution_sha256: source.execution_sha256,
        training_source_audit: source.training_source_audit ? JSON.stringify(source.training_source_audit) : '',
        corporate_actions_coverage: source.corporate_actions_coverage ? JSON.stringify(source.corporate_actions_coverage) : '',
        tax_policy: result.tax_policy, corporate_actions_summary: eventSummary(record.corporate_actions),
        corporate_actions: record.corporate_actions ? JSON.stringify(record.corporate_actions) : '',
        action_name: actionName(record.action), executed_action_name: actionName(executedAction(record)),
        ...(kind === 'trades' ? {signal_date: record.signal_date || record.date, slippage_cost: tradeSlip(record)} : {})
      }));
      content = '\uFEFF' + [fields.map(([, title]) => csvCell(title)).join(','), ...rows.map(row => fields.map(([key]) => csvCell(row[key])).join(','))].join('\r\n');
      extension = 'csv'; type = 'text/csv;charset=utf-8';
    }
    const blob = new Blob([content], {type}), url = URL.createObjectURL(blob);
    const anchor = document.createElement('a'); anchor.href = url;
    anchor.download = `${result.model.id.replace(':', '_')}_${result.symbol}_${result.actual_range.start}_${result.actual_range.end}_${kind}.${extension}`;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  el('form').addEventListener('submit', run);
  el('catalog-refresh').onclick = loadCatalog;
  el('latest').onclick = () => selectModel(null);
  ['val', 'test'].forEach(split => { el('range-' + split).onclick = () => applyRange(split); });
  ['symbol', 'model', 'start', 'end', 'ca-research'].forEach(id => el(id).addEventListener('change', selectionWarning));
  el('day').oninput = event => selectDay(event.target.value, true);
  el('prev').onclick = () => selectDay(selectedDay - 1, true);
  el('next').onclick = () => selectDay(selectedDay + 1, true);
  ['daily', 'trades', 'json'].forEach(kind => { el('export-' + kind).onclick = () => download(kind); });
  window.addEventListener('resize', () => { if (chart) chart.resize(); });
  let initialSelectionApplied = false;
  window.scaleReplay = {
    open: async () => {
      if (!catalog) await loadCatalog();
      if (!initialSelectionApplied && catalog) {
        initialSelectionApplied = true;
        const params = new URLSearchParams(location.search);
        const id = params.get('model_id');
        if (id) await selectModel(id);
        if (catalog.symbols.includes(params.get('symbol'))) el('symbol').value = params.get('symbol');
        ['start', 'end'].forEach(key => { if (/^\d{4}-\d{2}-\d{2}$/.test(params.get(key) || '')) el(key).value = params.get(key); });
        selectionWarning();
      }
      if (chart) requestAnimationFrame(() => chart.resize());
    },
    selectModel
  };
})();

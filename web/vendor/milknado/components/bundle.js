/* @ds-bundle: {"format":4,"namespace":"Milknado","components":[{"name":"Button"},{"name":"StatusGlyph"},{"name":"StatusBadge"},{"name":"ContextMeter"},{"name":"AgentRow"},{"name":"AgentRoster"},{"name":"Console"},{"name":"GraphNode"},{"name":"StatusStrip"},{"name":"GroupNode"},{"name":"MikadoGraph"},{"name":"GraphToolbar"},{"name":"Minimap"},{"name":"OutlineTree"},{"name":"AncestorPath"}]} */
/* Milknado — React 18 components. Classic script: reads window.React, assigns window.Milknado. Styles: bundle.css; tokens: tokens.css. */
(function () {
  var React = window.React;
  var h = React.createElement;
  function cx() { var out = []; for (var i = 0; i < arguments.length; i++) if (arguments[i]) out.push(arguments[i]); return out.join(' '); }

  var GLYPH_FOR = { pending: 'pending', ready: 'pending', running: 'running', 'at-risk': 'at-risk', failed: 'at-risk', blocked: 'blocked', done: 'done', listening: 'running', idle: 'pending' };
  var LABEL_FOR = { pending: 'Pending', ready: 'Ready', running: 'Running', 'at-risk': 'At risk', failed: 'Failed', blocked: 'Blocked', done: 'Done', listening: 'Listening', idle: 'Idle' };

  /* StatusGlyph — the 10px mark for a state, in currentColor. */
  function StatusGlyph(p) {
    return h('span', { className: cx('mk-glyph', 'mk-glyph-' + (GLYPH_FOR[p.state] || 'pending'), p.className), 'aria-hidden': 'true' });
  }

  /* Button — outlined, never filled. */
  function Button(p) {
    var variant = p.variant || 'secondary';
    return h('button', { type: p.type || 'button', className: cx('mk', 'mk-btn', 'mk-btn-' + variant, p.icon && 'mk-btn-icon', p.className), disabled: p.disabled, onClick: p.onClick, 'aria-label': p.ariaLabel, title: p.title },
      p.glyph ? h('span', { 'aria-hidden': 'true' }, p.glyph) : null, p.children, p.count != null ? h('span', { className: 'mk-btn-count' }, p.count) : null);
  }

  /* StatusBadge — a status word with its glyph on the -soft ground. */
  function StatusBadge(p) {
    var st = p.state || 'pending';
    var tone = st === 'ready' ? 'pending' : st === 'failed' ? 'at-risk' : st === 'listening' ? 'running' : st === 'idle' ? 'pending' : st;
    return h('span', { className: cx('mk', 'mk-badge', 'mk-badge-' + tone, p.className) }, h(StatusGlyph, { state: st }), p.children || LABEL_FOR[st] || st);
  }

  /* ContextMeter — context use as a 4px bar. */
  function ContextMeter(p) {
    var pct = Math.max(0, Math.min(100, Number(p.percent) || 0));
    var tone = pct > 80 ? 'is-full' : pct >= 50 ? 'is-warn' : '';
    return h('div', { className: cx('mk', 'mk-meter', tone, p.className), role: 'meter', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': p.label || 'context' },
      p.label === false ? null : h('div', { className: 'mk-meter-head' }, h('span', null, p.label || 'context'), h('span', { className: 'mk-meter-value' }, pct + '%')),
      h('div', { className: 'mk-meter-track' }, h('i', { className: 'mk-meter-fill', style: { width: pct + '%' } })));
  }

  /* AgentRow — one agent in the roster. */
  function AgentRow(p) {
    var st = p.status || 'idle';
    return h('button', { type: 'button', className: cx('mk', 'mk-agent', 'mk-agent-' + st, p.className), 'aria-selected': !!p.selected, onClick: p.onClick },
      h('span', { className: 'mk-dot', 'aria-hidden': 'true' }),
      h('span', { className: 'mk-agent-name' }, h('b', null, p.name), p.sub ? h('span', { className: 'mk-agent-sub' }, p.sub) : null),
      p.figure != null ? h('span', { className: 'mk-agent-figure' }, p.figure) : null);
  }

  /* AgentRoster — the AGENTS section of the rail. */
  function AgentRoster(p) {
    var agents = p.agents || [];
    return h('div', { className: cx('mk', 'mk-roster', p.className), role: 'listbox', 'aria-label': p.title || 'Agents' },
      h('div', { className: 'mk-roster-head' }, h('span', { className: 'mk-kicker' }, p.title || 'Agents'), p.count != null ? h('span', { className: 'mk-roster-count' }, p.count) : null),
      agents.map(function (a) { return h(AgentRow, { key: a.id || a.name, name: a.name, status: a.status, sub: a.sub, figure: a.figure, selected: p.selected === (a.id || a.name), onClick: function () { if (p.onSelect) p.onSelect(a.id || a.name); } }); }));
  }

  /* Console — the session log with its guidance input. */
  function Console(p) {
    var tabs = p.tabs || ['Session', 'Changes', 'Details'];
    var active = p.activeTab || tabs[0];
    var lines = p.lines || [];
    var _d = React.useState(''); var draft = p.value != null ? p.value : _d[0]; var setDraft = _d[1];
    var ref = React.useRef(null);
    React.useEffect(function () { if (ref.current && p.autoScroll !== false) ref.current.scrollTop = ref.current.scrollHeight; }, [lines.length]);
    function send() { var t = String(draft).trim(); if (!t) return; if (p.onSend) p.onSend(t); if (p.value == null) setDraft(''); }
    return h('div', { className: cx('mk', 'mk-console', p.className), style: p.style },
      h('div', { className: 'mk-console-tabs', role: 'tablist' }, tabs.map(function (t) { return h('button', { key: t, type: 'button', role: 'tab', 'aria-selected': t === active, className: cx('mk-console-tab', 'mk-kicker', t === active && 'is-live'), onClick: function () { if (p.onTab) p.onTab(t); } }, t); })),
      h('div', { className: 'mk-console-lines', ref: ref, role: 'log' },
        lines.length ? lines.map(function (l, i) { return h('div', { key: i, className: cx('mk-line', l.tone && l.tone !== 'plain' && 'mk-line-' + l.tone) }, h('span', { className: 'mk-line-time' }, l.time), h('span', null, l.text)); })
          : h('div', { className: 'mk-console-empty' }, h('div', null, p.emptyTitle || 'No output yet'), h('div', null, p.emptyHint || 'Lines appear when the agent starts its first iteration.'))),
      p.input === false ? null : h('div', { className: 'mk-console-input' },
        h('input', { className: 'mk-input', value: draft, placeholder: p.placeholder || 'Guidance — reaches the next iteration', 'aria-label': 'Guidance', onChange: function (e) { if (p.onChange) p.onChange(e.target.value); if (p.value == null) setDraft(e.target.value); }, onKeyDown: function (e) { if (e.key === 'Enter') { e.preventDefault(); send(); } } }),
        h(Button, { variant: 'primary', onClick: send, className: 'mk-btn-sm' }, p.sendLabel || 'Send')));
  }

  /* GraphNode — one node: status, title, one metadata line. */
  function GraphNode(p) {
    var st = p.state || 'pending';
    var kind = p.kind || 'task';
    var statusWord = p.statusText || (st === 'ready' ? 'ready' : st);
    return h('button', { type: 'button', className: cx('mk', 'mk-node', 'mk-node-' + st, kind === 'goal' && 'mk-node-goal', kind === 'subgoal' && 'mk-node-subgoal', p.compact && 'mk-node-compact', p.className), 'aria-selected': !!p.selected, onClick: p.onClick, style: p.style, title: p.title },
      h('span', { className: 'mk-node-status' }, h('span', { className: 'mk-kicker', style: { display: 'inline-flex', alignItems: 'center', gap: 6 } }, h(StatusGlyph, { state: st }), statusWord), p.flavor ? h('span', { className: 'mk-node-flavor' }, p.flavor) : null),
      h('span', { className: 'mk-node-title' }, p.title),
      (p.meta || p.agent || p.artifact) ? h('span', { className: 'mk-node-meta' }, p.agent ? h('span', { className: 'mk-node-agent' }, p.agent) : null, p.meta ? h('span', null, p.meta) : null, p.artifact ? h('span', { className: 'mk-node-artifact', title: 'artifact awaiting review' }, '▣ ' + (p.artifact === true ? '' : p.artifact)) : null) : null,
      st === 'running' && p.progress != null ? h('span', { className: 'mk-node-bar' }, h('i', { style: { width: Math.max(0, Math.min(100, p.progress)) + '%' } })) : null);
  }

  /* Edge geometry. 'elbow': vertical-horizontal-vertical with rounded corners (default). 'curve': one cubic S-curve. 'square': sharp elbows. */
  function edgePath(x1, y1, x2, y2, mid, style) {
    if (style === 'square') return 'M' + x1 + ' ' + y1 + ' V' + mid + ' H' + x2 + ' V' + y2;
    if (style === 'curve') return 'M' + x1 + ' ' + y1 + ' C' + x1 + ' ' + mid + ' ' + x2 + ' ' + mid + ' ' + x2 + ' ' + y2;
    var dx = x2 - x1, r = Math.min(8, Math.abs(dx) / 2, Math.abs(mid - y1), Math.abs(y2 - mid)); if (Math.abs(dx) < 1) return 'M' + x1 + ' ' + y1 + ' V' + y2;
    var sx = dx > 0 ? 1 : -1;
    return 'M' + x1 + ' ' + y1 + ' V' + (mid - r) + ' Q' + x1 + ' ' + mid + ' ' + (x1 + sx * r) + ' ' + mid + ' H' + (x2 - sx * r) + ' Q' + x2 + ' ' + mid + ' ' + x2 + ' ' + (mid + r) + ' V' + y2;
  }

  /* Tree helpers shared by the graph, the outline, the minimap and the path. */
  function indexNodes(nodes) {
    var byId = {}, children = {};
    nodes.forEach(function (n) { byId[n.id] = n; });
    nodes.forEach(function (n) { if (n.parent != null && byId[n.parent]) (children[n.parent] = children[n.parent] || []).push(n); });
    return { byId: byId, children: children };
  }
  function ancestorsOf(id, byId) { var out = []; var c = byId[id]; while (c) { out.unshift(c); c = c.parent != null ? byId[c.parent] : null; } return out; }
  function descendantsOf(id, children) { var out = []; (function walk(i) { (children[i] || []).forEach(function (c) { out.push(c); walk(c.id); }); })(id); return out; }
  function countStates(list) { var c = { total: list.length, done: 0, running: 0, blocked: 0, ready: 0, failed: 0, pending: 0 }; list.forEach(function (n) { var k = n.state || 'pending'; if (k === 'at-risk') k = 'failed'; if (c[k] != null) c[k]++; }); return c; }
  function subtreeDone(id, byId, children) { var d = descendantsOf(id, children); return d.length > 0 && d.every(function (n) { return n.state === 'done'; }) && byId[id].state === 'done'; }

  /* StatusStrip — counts as a stacked bar: done, running, ready, blocked, failed, pending. */
  function StatusStrip(p) {
    var c = p.counts || countStates(p.nodes || []); var total = c.total || 1;
    var segs = [['done', c.done], ['running', c.running], ['ready', c.ready], ['blocked', c.blocked], ['failed', c.failed], ['pending', c.pending]];
    return h('span', { className: cx('mk', 'mk-strip', p.className), role: 'img', 'aria-label': (c.done + ' of ' + c.total + ' done') + (c.blocked ? ', ' + c.blocked + ' blocked' : '') + (c.running ? ', ' + c.running + ' running' : '') },
      h('span', { className: 'mk-strip-bar' }, segs.map(function (s) { return s[1] ? h('i', { key: s[0], className: 'mk-strip-' + s[0], style: { flex: s[1] } }) : null; })),
      p.label === false ? null : h('span', { className: 'mk-strip-label' }, c.done + ' / ' + c.total + (p.short ? '' : (c.blocked ? ' · ' + c.blocked + ' blocked' : '') + (c.running ? ' · ' + c.running + ' running' : ''))));
  }

  /* GroupNode — a collapsed subtree: the sub-goal title, a status strip and the count. */
  function GroupNode(p) {
    var c = p.counts || countStates(p.descendants || []);
    var allDone = c.total > 0 && c.done === c.total;
    return h('button', { type: 'button', className: cx('mk', 'mk-group', allDone && 'mk-group-done', c.blocked && 'mk-group-blocked', c.running && 'mk-group-running', p.className), 'aria-selected': !!p.selected, 'aria-expanded': false, onClick: p.onClick, style: p.style, title: p.title },
      h('span', { className: 'mk-node-status' }, h('span', { className: 'mk-kicker', style: { display: 'inline-flex', alignItems: 'center', gap: 6 } }, h(StatusGlyph, { state: allDone ? 'done' : c.blocked ? 'blocked' : c.running ? 'running' : 'pending' }), allDone ? 'done' : c.total + ' nodes'), h('span', { className: 'mk-group-expand', 'aria-hidden': 'true' }, '⌄')),
      h('span', { className: 'mk-node-title' }, p.title),
      h(StatusStrip, { counts: c, label: !allDone, short: true }));
  }

  /* Level of detail from zoom: 'card' >= 0.8, 'pill' >= 0.5, 'dot' below. */
  function lodFor(zoom) { zoom = zoom == null ? 1 : zoom; return zoom >= 0.8 ? 'card' : zoom >= 0.5 ? 'pill' : 'dot'; }

  /* MikadoGraph — the goal tree on a sunken canvas, with collapse, focus, filter and level of detail. */
  function MikadoGraph(p) {
    var all = p.nodes || [];
    var W = p.width || 680, H = typeof p.height === 'number' ? p.height : 300, levelGap = p.levelGap || 100, top = 16;
    var ix = indexNodes(all), byId = ix.byId, children = ix.children;
    var collapsed = {}; (p.collapsed || []).forEach(function (id) { collapsed[id] = true; });
    if (p.collapseDone !== false) all.forEach(function (n) { if ((children[n.id] || []).length && subtreeDone(n.id, byId, children) && n.parent != null) collapsed[n.id] = true; });
    var focusSet = null;
    if (p.focus != null && byId[p.focus]) { focusSet = {}; ancestorsOf(p.focus, byId).forEach(function (n) { focusSet[n.id] = true; }); descendantsOf(p.focus, children).forEach(function (n) { focusSet[n.id] = true; }); }
    var hidden = {};
    all.forEach(function (n) { if (collapsed[n.id]) descendantsOf(n.id, children).forEach(function (d) { hidden[d.id] = true; }); });
    if (p.hideDone) all.forEach(function (n) { if (n.state === 'done' && !collapsed[n.id] && (children[n.id] || []).every(function (c) { return c.state === 'done'; })) { hidden[n.id] = true; } });
    if (p.filter) { var keep = {}; all.forEach(function (n) { if (n.state === p.filter) ancestorsOf(n.id, byId).forEach(function (a) { keep[a.id] = true; }); }); all.forEach(function (n) { if (!keep[n.id]) hidden[n.id] = true; }); }
    if (focusSet) all.forEach(function (n) { if (!focusSet[n.id]) hidden[n.id] = hidden[n.id] || 'faded'; });
    var nodes = all.filter(function (n) { return hidden[n.id] !== true; });
    var lod = p.lod || lodFor(p.zoom);
    function depth(n) { var d = 0, c = n; while (c && c.parent != null && byId[c.parent]) { d++; c = byId[c.parent]; } return d; }
    var levels = {}; nodes.forEach(function (n) { var d = depth(n); (levels[d] = levels[d] || []).push(n); });
    var pos = {}, gap = p.gap || 16, pad = 16;
    var widest = 0; Object.keys(levels).forEach(function (d) { widest = Math.max(widest, levels[d].length); });
    var pillW = Math.max(96, Math.min(120, Math.floor((W - pad * 2 - gap * (widest - 1)) / Math.max(1, widest))));
    function widthAt(l, n) { return l === 'dot' ? 12 : l === 'pill' ? pillW : n.kind === 'goal' ? 200 : 148; }
    function heightAt(l, n) { return l === 'dot' ? 12 : l === 'pill' ? 24 : collapsed[n.id] ? 62 : n.kind === 'goal' ? 56 : 44; }
    function rowsNeeded(l, row) { var total = row.reduce(function (s, n) { return s + widthAt(l, n); }, 0) + gap * (row.length - 1); return Math.max(1, Math.ceil(total / (W - pad * 2))); }
    /* Semantic zoom: a level that needs more than two staggered rows at this detail steps the whole graph down (cards → pills → dots). */
    if (p.autoLod !== false) { var order = ['card', 'pill', 'dot']; var i = order.indexOf(lod); while (i < 2 && Object.keys(levels).some(function (d) { return rowsNeeded(order[i], levels[d]) > 2; })) i++; lod = order[i]; }
    function widthOf(n) { return widthAt(lod, n); }
    function heightOf(n) { return heightAt(lod, n); }
    var lg = lod === 'dot' ? 20 : lod === 'pill' ? 32 : Math.max(32, levelGap - 56);
    var y = top;
    Object.keys(levels).sort(function (a, b) { return a - b; }).forEach(function (d) {
      var row = levels[d]; var k = rowsNeeded(lod, row); var subs = []; for (var r = 0; r < k; r++) subs.push([]);
      row.forEach(function (n, i) { subs[i % k].push(n); });
      var rowH = Math.max.apply(null, row.map(heightOf));
      subs.forEach(function (sub, r) {
        var total = sub.reduce(function (s, n) { return s + widthOf(n); }, 0) + gap * (sub.length - 1);
        var x = total <= W - pad * 2 ? (W - total) / 2 : pad; var step = total <= W - pad * 2 ? 0 : (W - pad * 2 - total) / Math.max(1, sub.length - 1);
        sub.forEach(function (n) { pos[n.id] = { x: x + widthOf(n) / 2, y: y + r * (rowH + 6) }; x += widthOf(n) + gap + step; });
      });
      y += k * (rowH + 6) - 6 + lg;
    });
    var contentH = y - lg + pad + 24; if (p.height == null || p.height === 'auto') H = Math.max(120, contentH); else H = Math.max(H, p.grow === false ? 0 : contentH);
    var sel = p.selected != null ? byId[p.selected] : null;
    var hot = {}; if (sel) ancestorsOf(sel.id, byId).forEach(function (n) { hot[n.id] = true; });
    var edges = [];
    nodes.forEach(function (n) {
      var links = []; if (n.parent != null && byId[n.parent] && pos[n.parent]) links.push({ from: byId[n.parent], extra: false });
      (n.extra || []).forEach(function (id) { if (byId[id] && pos[id]) links.push({ from: byId[id], extra: true }); });
      links.forEach(function (l) {
        var a = pos[l.from.id], b = pos[n.id]; var ab = a.y + heightOf(l.from); var mid = (ab + b.y) / 2;
        edges.push({ key: l.from.id + '-' + n.id + (l.extra ? 'x' : ''), d: edgePath(a.x, ab, b.x, b.y, mid, p.edgeStyle), hot: hot[n.id] && hot[l.from.id] && !l.extra, critical: n.state === 'blocked', extra: l.extra, faded: hidden[n.id] === 'faded' || hidden[l.from.id] === 'faded' });
      });
    });
    var edgeBudget = p.edgeBudget || 200; if (edges.length > edgeBudget) edges = edges.filter(function (e) { return e.hot || e.critical; });
    React.useEffect(function () { if (p.onLayout) p.onLayout({ positions: pos, visible: nodes.map(function (n) { return n.id; }), collapsed: Object.keys(collapsed), lod: lod }); });
    return h('div', { className: cx('mk', 'mk-graph', 'mk-graph-lod-' + lod, p.className), style: { width: W, height: H }, onClick: function () { if (p.onSelect) p.onSelect(null); } },
      h('svg', { viewBox: '0 0 ' + W + ' ' + H, 'aria-hidden': 'true' }, edges.map(function (e) { return h('path', { key: e.key, d: e.d, className: cx('mk-edge', e.hot && 'mk-edge-hot', e.critical && 'mk-edge-critical', e.extra && 'mk-edge-extra', e.faded && 'mk-edge-faded') }); })),
      nodes.map(function (n) {
        var q = pos[n.id]; var faded = hidden[n.id] === 'faded';
        var click = function (e) { e.stopPropagation(); if (p.onSelect) p.onSelect(n.id); };
        var body;
        if (lod === 'dot') body = h('span', { className: cx('mk-dotnode', 'mk-dotnode-' + (n.state || 'pending')), title: n.title, onClick: click });
        else if (collapsed[n.id] && lod === 'pill') { var gc = countStates(descendantsOf(n.id, children)); body = h('button', { type: 'button', className: cx('mk-pill', 'mk-pill-group', gc.blocked ? 'mk-pill-blocked' : gc.running ? 'mk-pill-running' : gc.done === gc.total ? 'mk-pill-done' : 'mk-pill-pending'), style: { width: pillW }, 'aria-selected': p.selected === n.id, 'aria-expanded': false, title: n.title + ' \u00B7 ' + gc.done + ' / ' + gc.total, onClick: function (e) { e.stopPropagation(); if (p.onExpand) p.onExpand(n.id); else if (p.onSelect) p.onSelect(n.id); } }, h(StatusGlyph, { state: gc.blocked ? 'blocked' : gc.running ? 'running' : gc.done === gc.total ? 'done' : 'pending' }), h('span', { className: 'mk-pill-title' }, n.title), h('span', { className: 'mk-pill-count' }, gc.done + '/' + gc.total)); }
        else if (collapsed[n.id]) body = h(GroupNode, { title: n.title, descendants: descendantsOf(n.id, children), selected: p.selected === n.id, onClick: function (e) { e.stopPropagation(); if (p.onExpand) p.onExpand(n.id); else if (p.onSelect) p.onSelect(n.id); } });
        else if (lod === 'pill') body = h('button', { type: 'button', className: cx('mk-pill', 'mk-pill-' + (n.state || 'pending'), n.kind === 'goal' && 'mk-pill-goal'), style: { width: pillW }, 'aria-selected': p.selected === n.id, title: n.title, onClick: click }, h(StatusGlyph, { state: n.state || 'pending' }), h('span', { className: 'mk-pill-title' }, n.title));
        else body = h(GraphNode, { compact: p.compact !== false, kind: n.kind, state: n.state, title: n.title, flavor: p.showFlavor ? n.flavor : null, statusText: n.statusText, meta: p.compact === false ? n.meta : null, agent: p.compact === false ? n.agent : null, artifact: p.compact === false ? n.artifact : null, progress: n.progress, selected: p.selected === n.id, onClick: click });
        return h('div', { key: n.id, className: cx('mk-graph-node', faded && 'is-faded'), style: { left: q.x, top: q.y } }, body);
      }),
      p.hint !== false ? h('div', { className: 'mk-graph-hint' }, p.hint || 'drag between nodes to add a dependency · dashed = extra prerequisite') : null);
  }

  /* GraphToolbar — search, filters, focus, collapse, node style and zoom, top right of the canvas. */
  function GraphToolbar(p) {
    var _q = React.useState(''); var q = p.query != null ? p.query : _q[0];
    var matches = q ? (p.nodes || []).filter(function (n) { return String(n.title).toLowerCase().indexOf(q.toLowerCase()) >= 0; }).slice(0, 8) : [];
    var filters = [['ready', 'Ready'], ['running', 'Running'], ['blocked', 'Blocked']];
    return h('div', { className: cx('mk', 'mk-toolbar', p.className), role: 'toolbar', 'aria-label': 'Graph' },
      h('div', { className: 'mk-toolbar-search' },
        h('input', { className: 'mk-input', value: q, placeholder: 'Jump to node…', 'aria-label': 'Jump to node', onChange: function (e) { _q[1](e.target.value); if (p.onQuery) p.onQuery(e.target.value); }, onKeyDown: function (e) { if (e.key === 'Enter' && matches[0] && p.onJump) { p.onJump(matches[0].id); _q[1](''); } if (e.key === 'Escape') _q[1](''); } }),
        matches.length ? h('ul', { className: 'mk-toolbar-results', role: 'listbox' }, matches.map(function (n) { return h('li', { key: n.id, role: 'option', className: 'mk-toolbar-result', onMouseDown: function () { if (p.onJump) p.onJump(n.id); _q[1](''); } }, h(StatusGlyph, { state: n.state || 'pending' }), h('span', { className: 'mk-toolbar-result-title' }, n.title), h('span', { className: 'mk-toolbar-result-path' }, (p.pathFor ? p.pathFor(n.id) : ''))); })) : null),
      h('div', { className: 'mk-seg', role: 'group', 'aria-label': 'Filter' }, filters.map(function (f) { return h('button', { key: f[0], type: 'button', className: cx('mk-seg-opt', p.filter === f[0] && 'is-on'), 'aria-pressed': p.filter === f[0], onClick: function () { if (p.onFilter) p.onFilter(p.filter === f[0] ? null : f[0]); } }, f[1]); })),
      h('button', { type: 'button', className: cx('mk-btn', 'mk-btn-secondary', 'mk-btn-sm', p.hideDone && 'is-on'), 'aria-pressed': !!p.hideDone, onClick: function () { if (p.onHideDone) p.onHideDone(!p.hideDone); } }, 'Hide done'),
      h('button', { type: 'button', className: cx('mk-btn', 'mk-btn-secondary', 'mk-btn-sm', p.focus && 'is-on'), 'aria-pressed': !!p.focus, disabled: !p.canFocus && !p.focus, title: 'Show the selected node with its ancestors and descendants', onClick: function () { if (p.onFocus) p.onFocus(!p.focus); } }, 'Focus'),
      h('button', { type: 'button', className: 'mk-btn mk-btn-secondary mk-btn-sm', onClick: p.onCollapseAll, title: 'Collapse all groups' }, '⌃ all'),
      h('button', { type: 'button', className: 'mk-btn mk-btn-secondary mk-btn-sm', onClick: p.onExpandAll, title: 'Expand all groups' }, '⌄ all'),
      h('div', { className: 'mk-seg', role: 'group', 'aria-label': 'Node style' }, ['card', 'pill', 'dot'].map(function (s) { return h('button', { key: s, type: 'button', className: cx('mk-seg-opt', (p.lod || 'card') === s && 'is-on'), 'aria-pressed': (p.lod || 'card') === s, onClick: function () { if (p.onLod) p.onLod(s); } }, s + 's'); })),
      h('div', { className: 'mk-toolbar-zoom' }, h('button', { type: 'button', className: 'mk-btn mk-btn-secondary mk-btn-icon mk-btn-ctl', 'aria-label': 'Zoom in', onClick: p.onZoomIn }, '+'), h('button', { type: 'button', className: 'mk-btn mk-btn-secondary mk-btn-icon mk-btn-ctl', 'aria-label': 'Zoom out', onClick: p.onZoomOut }, '−'), h('button', { type: 'button', className: 'mk-btn mk-btn-secondary mk-btn-sm mk-btn-ctl', onClick: p.onFit }, 'fit')));
  }

  /* Minimap — the whole graph as a grid of state-coloured blocks, with the viewport. Groups, not nodes, past 200. */
  function Minimap(p) {
    var nodes = p.nodes || []; var W = p.width || 160, Hh = p.height || 96;
    var ix = indexNodes(nodes); var byId = ix.byId, children = ix.children;
    var items = nodes;
    if (nodes.length > (p.groupAbove || 200)) { items = nodes.filter(function (n) { return n.kind === 'goal' || n.kind === 'subgoal'; }).map(function (g) { var d = descendantsOf(g.id, children); var c = countStates(d); return { id: g.id, state: c.blocked ? 'blocked' : c.running ? 'running' : c.done === c.total && c.total ? 'done' : 'pending', weight: Math.max(1, d.length), parent: g.parent, kind: g.kind }; }); }
    function depth(n) { var d = 0, c = n; while (c && c.parent != null && byId[c.parent]) { d++; c = byId[c.parent]; } return d; }
    var levels = {}; items.forEach(function (n) { var d = depth(n); (levels[d] = levels[d] || []).push(n); });
    var rows = Object.keys(levels).length || 1; var rh = Hh / rows;
    var blocks = [];
    Object.keys(levels).forEach(function (d) { var row = levels[d]; var tw = row.reduce(function (s, n) { return s + (n.weight || 1); }, 0); var x = 0; row.forEach(function (n) { var w = W * (n.weight || 1) / tw; blocks.push({ id: n.id, x: x + 1, y: d * rh + 1, w: Math.max(2, w - 2), h: Math.max(2, rh - 2), state: n.state || 'pending' }); x += w; }); });
    var vp = p.viewport;
    return h('div', { className: cx('mk', 'mk-minimap', p.className), style: { width: W, height: Hh }, role: 'img', 'aria-label': 'Overview of the graph' },
      h('svg', { viewBox: '0 0 ' + W + ' ' + Hh, width: W, height: Hh }, blocks.map(function (b) { return h('rect', { key: b.id, x: b.x, y: b.y, width: b.w, height: b.h, rx: 1, className: cx('mk-minimap-block', 'is-' + b.state, p.selected === b.id && 'is-selected'), onClick: function () { if (p.onJump) p.onJump(b.id); } }); }),
        vp ? h('rect', { className: 'mk-minimap-viewport', x: vp.x * W, y: vp.y * Hh, width: vp.w * W, height: vp.h * Hh }) : null));
  }

  /* OutlineTree — the graph as an indented list: the same data, the same states, collapsible. */
  function OutlineTree(p) {
    var nodes = p.nodes || []; var ix = indexNodes(nodes); var byId = ix.byId, children = ix.children;
    var roots = nodes.filter(function (n) { return n.parent == null || !byId[n.parent]; });
    var _c = React.useState({}); var closed = p.collapsed ? {} : _c[0]; (p.collapsed || []).forEach(function (id) { closed[id] = true; });
    function toggle(id) { if (p.onToggle) p.onToggle(id); else { var n = Object.assign({}, _c[0]); n[id] = !n[id]; _c[1](n); } }
    function row(n, depth) {
      var kids = children[n.id] || []; var isClosed = !!closed[n.id]; var c = kids.length ? countStates(descendantsOf(n.id, children)) : null;
      var match = p.query ? String(n.title).toLowerCase().indexOf(p.query.toLowerCase()) >= 0 : true;
      var showKids = kids.length && !isClosed;
      var out = [];
      if (match || !p.query) out.push(h('div', { key: n.id, className: cx('mk-outline-row', 'mk-outline-' + (n.state || 'pending'), p.selected === n.id && 'is-selected'), style: { paddingLeft: 8 + depth * 16 }, role: 'treeitem', 'aria-level': depth + 1, 'aria-expanded': kids.length ? !isClosed : undefined, 'aria-selected': p.selected === n.id, tabIndex: p.selected === n.id ? 0 : -1, onClick: function () { if (p.onSelect) p.onSelect(n.id); }, onKeyDown: function (e) { if (e.key === 'ArrowRight' && kids.length && isClosed) toggle(n.id); if (e.key === 'ArrowLeft' && kids.length && !isClosed) toggle(n.id); if (e.key === 'Enter' && p.onOpen) p.onOpen(n.id); } },
        kids.length ? h('button', { type: 'button', className: 'mk-outline-twisty', 'aria-label': isClosed ? 'Expand' : 'Collapse', onClick: function (e) { e.stopPropagation(); toggle(n.id); } }, isClosed ? '›' : '⌄') : h('span', { className: 'mk-outline-twisty' }),
        h(StatusGlyph, { state: n.state || 'pending' }),
        h('span', { className: cx('mk-outline-title', n.kind === 'goal' && 'is-goal', n.kind === 'subgoal' && 'is-subgoal') }, n.title),
        n.agent ? h('span', { className: 'mk-outline-agent' }, n.agent) : null,
        c ? h('span', { className: 'mk-outline-count' }, isClosed ? c.done + ' / ' + c.total : '') : null));
      if (showKids || p.query) kids.forEach(function (k) { out = out.concat(row(k, depth + 1)); });
      return out;
    }
    return h('div', { className: cx('mk', 'mk-outline', p.className), role: 'tree', 'aria-label': p.label || 'Nodes' }, roots.map(function (r) { return row(r, 0); }));
  }

  /* AncestorPath — the path from the goal to a node, as a breadcrumb. */
  function AncestorPath(p) {
    var ix = indexNodes(p.nodes || []); var path = p.id != null ? ancestorsOf(p.id, ix.byId) : [];
    var maxItems = p.max || 4; var items = path;
    if (path.length > maxItems) items = [path[0], { id: '…', title: '…', gap: true }].concat(path.slice(path.length - (maxItems - 1)));
    return h('nav', { className: cx('mk', 'mk-path', p.className), 'aria-label': 'Path to the goal' }, items.map(function (n, i) {
      return h(React.Fragment, { key: n.id }, i ? h('span', { className: 'mk-path-sep', 'aria-hidden': 'true' }, '›') : null,
        n.gap ? h('span', { className: 'mk-path-gap' }, '…') : h('button', { type: 'button', className: cx('mk-path-item', i === items.length - 1 && 'is-current'), 'aria-current': i === items.length - 1 ? 'page' : undefined, onClick: function () { if (p.onSelect) p.onSelect(n.id); } }, n.title));
    }));
  }

  window.Milknado = { Button: Button, StatusGlyph: StatusGlyph, StatusBadge: StatusBadge, ContextMeter: ContextMeter, AgentRow: AgentRow, AgentRoster: AgentRoster, Console: Console, GraphNode: GraphNode, StatusStrip: StatusStrip, GroupNode: GroupNode, MikadoGraph: MikadoGraph, GraphToolbar: GraphToolbar, Minimap: Minimap, OutlineTree: OutlineTree, AncestorPath: AncestorPath, lodFor: lodFor, tree: { index: indexNodes, ancestors: ancestorsOf, descendants: descendantsOf, counts: countStates } };
})();

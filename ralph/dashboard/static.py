"""Static HTML+JS+CSS for the dashboard, served as a single inline page.

Kept as a Python string so there's no build step, no packaged-data wiring, and
no new runtime dependency. Vanilla JS polls `/api/instances` on an interval.
"""

from __future__ import annotations


def render_index(refresh_seconds: float = 1.5) -> str:
    # Clamp to sane bounds — a user could still inject anything via query param.
    if refresh_seconds < 0.25:
        refresh_seconds = 0.25
    if refresh_seconds > 60:
        refresh_seconds = 60
    refresh_ms = int(refresh_seconds * 1000)
    return _TEMPLATE.replace("__REFRESH_MS__", str(refresh_ms))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ralph Dashboard</title>
<style>
  :root {
    --bg: #0e1013;
    --panel: #161a20;
    --panel-2: #1c2129;
    --fg: #e6e6e6;
    --muted: #8892a0;
    --accent: #6aa8ff;
    --good: #3ecf8e;
    --warn: #f5a524;
    --bad: #ef4b4b;
    --border: #252a33;
  }
  html, body { background: var(--bg); color: var(--fg); margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "SF Mono", Menlo, monospace; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex;
    align-items: baseline; justify-content: space-between; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: 0.02em; }
  header .meta { color: var(--muted); font-size: 12px; }
  main { padding: 16px 20px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
  .card.stopped { opacity: 0.55; }
  .card.stale { border-color: var(--warn); }
  .card h2 { font-size: 14px; margin: 0; font-weight: 600; display: flex;
    justify-content: space-between; align-items: baseline; gap: 8px; }
  .card h2 .title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge { font-size: 10px; padding: 2px 8px; border-radius: 999px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; }
  .badge.running { background: rgba(62, 207, 142, 0.15); color: var(--good); }
  .badge.stopped { background: rgba(239, 75, 75, 0.15); color: var(--bad); }
  .badge.stale   { background: rgba(245, 165, 36, 0.15); color: var(--warn); }
  .row { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
  .row strong { color: var(--fg); font-weight: 500; }
  .stories-hero { display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px; }
  .stories-hero .count { font-size: 28px; font-weight: 600; color: var(--fg);
    letter-spacing: -0.01em; line-height: 1; }
  .stories-hero .count .total { color: var(--muted); font-weight: 400; }
  .stories-hero .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.08em; }
  .stories-hero .tool-phase { font-size: 11px; color: var(--muted); }
  .story { background: var(--panel-2); padding: 8px 10px; border-radius: 6px;
    font-size: 12px; line-height: 1.4; }
  .story .id { color: var(--accent); font-weight: 600; }
  .story .title { color: var(--fg); }
  .bar { height: 6px; background: var(--panel-2); border-radius: 3px; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: var(--accent);
    transition: width 0.4s ease; }
  .bar > span.good { background: var(--good); }
  .bar > span.warn { background: var(--warn); }
  .bar > span.bad { background: var(--bad); }
  .bar-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center;
    font-size: 11px; color: var(--muted); }
  .bucket { margin-top: 6px; }
  .bucket .bar-row .ident { color: var(--fg); font-weight: 500; }
  .bucket .bar-row .ident.active::after { content: ' ◀'; color: var(--accent); }
  .bucket .forecast { font-size: 10px; color: var(--muted); margin-top: 2px;
    text-align: right; }
  .bucket-source { color: var(--muted); font-size: 10px; margin-left: 4px;
    text-transform: uppercase; letter-spacing: 0.04em; }
  pre.tail { background: #070809; border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 10px; font-size: 11px; line-height: 1.4; height: 160px; overflow: auto;
    white-space: pre-wrap; word-break: break-all; margin: 0; }
  .tiny { font-size: 11px; color: var(--muted); }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: var(--good); margin-right: 6px; vertical-align: middle; }
  .dot.stopped { background: var(--bad); }
  .dot.stale { background: var(--warn); }
</style>
</head>
<body>
<header>
  <h1><span class="dot" id="connection-dot"></span>Ralph Dashboard</h1>
  <div class="meta" id="meta">polling\u2026</div>
</header>
<main>
  <div id="empty" class="empty" hidden>No active ralph instances. Start one with <code>ralph run</code>.</div>
  <div id="grid" class="grid"></div>
</main>
<script>
const REFRESH_MS = __REFRESH_MS__;
const TAIL_BYTES = 8192;
const grid = document.getElementById('grid');
const emptyEl = document.getElementById('empty');
const metaEl = document.getElementById('meta');
const dot = document.getElementById('connection-dot');

function fmtDuration(sec) {
  if (sec == null || !isFinite(sec)) return '';
  sec = Math.max(0, Math.floor(sec));
  if (sec < 60) return sec + 's';
  const m = Math.floor(sec / 60);
  if (m < 60) return m + 'm ' + (sec % 60) + 's';
  const h = Math.floor(m / 60);
  return h + 'h ' + (m % 60) + 'm';
}

function ageSeconds(isoString) {
  if (!isoString) return null;
  const t = Date.parse(isoString);
  if (isNaN(t)) return null;
  return (Date.now() - t) / 1000;
}

function progressClass(pct) {
  if (pct >= 90) return 'bad';
  if (pct >= 70) return 'warn';
  return 'good';
}

function createCard(pid) {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.pid = pid;
  card.innerHTML = `
    <h2>
      <span class="title" data-f="title"></span>
      <span class="badge" data-f="badge"></span>
    </h2>
    <div class="stories-hero">
      <div>
        <div class="label">stories complete</div>
        <div class="count"><span data-f="passing"></span><span class="total"> / <span data-f="total"></span></span></div>
      </div>
      <div class="tool-phase" data-f="tool-phase"></div>
    </div>
    <div>
      <div class="bar-row">
        <span data-f="pct-label"></span>
        <span data-f="iter-label"></span>
      </div>
      <div class="bar"><span data-f="pct-bar"></span></div>
    </div>
    <div data-f="story-wrap"></div>
    <div data-f="usage-wrap" hidden>
      <div class="bar-row">
        <span>5h window usage</span>
        <span data-f="usage-label"></span>
      </div>
      <div class="bar"><span data-f="usage-bar"></span></div>
    </div>
    <div data-f="buckets-wrap" hidden></div>
    <pre class="tail" data-f="tail">loading\u2026</pre>
    <div class="row tiny">
      <span data-f="ident"></span>
      <span data-f="timing"></span>
    </div>
  `;
  return card;
}

function updateCard(card, inst) {
  const f = (name) => card.querySelector(`[data-f="${name}"]`);
  card.className = 'card ' + inst.status;

  const title = inst.prd_project || inst.cwd || ('PID ' + inst.pid);
  f('title').textContent = title;
  const badge = f('badge');
  badge.className = 'badge ' + inst.status;
  badge.textContent = inst.status;

  const progress = inst.prd_progress || {passing: 0, total: 0};
  const pct = progress.total ? (progress.passing / progress.total) * 100 : 0;
  f('passing').textContent = progress.passing;
  f('total').textContent = progress.total;

  const phase = inst.phase || (inst.two_phase ? '\u2014' : 'single');
  f('tool-phase').textContent = `${inst.tool} \u00b7 ${phase}`;

  f('pct-label').textContent = `${pct.toFixed(0)}% of stories passing`;
  f('iter-label').textContent = `iter ${inst.iteration || 0}/${inst.max_iterations || '?'}`;
  const pctBar = f('pct-bar');
  pctBar.className = progressClass(pct);
  pctBar.style.width = pct + '%';

  const storyWrap = f('story-wrap');
  const story = inst.current_story;
  if (story) {
    storyWrap.innerHTML = '<div class="story"><span class="id">' +
      escapeHtml(story.id || '') + '</span> \u00b7 ' +
      '<span class="title">' + escapeHtml(story.title || '') + '</span></div>';
  } else {
    storyWrap.innerHTML = '<div class="story tiny">No active story</div>';
  }

  const usage = inst.usage || {};
  const buckets = Array.isArray(usage.buckets) ? usage.buckets : [];
  const usageWrap = f('usage-wrap');
  const bucketsWrap = f('buckets-wrap');

  if (buckets.length >= 2) {
    // Multi-bucket view: one bar per (provider, account). This is the new
    // mode triggered by --ccs-pool or when running through CCS with multiple
    // discovered accounts. The single-bar view is hidden in this case to
    // avoid double-counting.
    usageWrap.hidden = true;
    bucketsWrap.hidden = false;
    bucketsWrap.innerHTML = buckets.map(renderBucket).join('');
  } else if (buckets.length === 1) {
    // Exactly one bucket — render as the classic single bar so CCS users
    // with just one configured profile see the same UI as host-claude.
    bucketsWrap.hidden = true;
    usageWrap.hidden = false;
    const b = buckets[0];
    const pct = typeof b.percentage === 'number' ? b.percentage : 0;
    f('usage-label').textContent = `${b.identity || 'anthropic'} · ${pct.toFixed(1)}%`;
    const bar = f('usage-bar');
    bar.className = progressClass(pct);
    bar.style.width = Math.min(100, pct) + '%';
  } else {
    // Legacy / no-buckets: fall back to the flat usage.percentage field.
    bucketsWrap.hidden = true;
    const usagePct = typeof usage.percentage === 'number' ? usage.percentage : null;
    if (usagePct != null) {
      usageWrap.hidden = false;
      f('usage-label').textContent = `${usagePct.toFixed(1)}%`;
      const bar = f('usage-bar');
      bar.className = progressClass(usagePct);
      bar.style.width = Math.min(100, usagePct) + '%';
    } else {
      usageWrap.hidden = true;
    }
  }

  const startedAge = fmtDuration(ageSeconds(inst.started_at));
  const hbAge = inst.heartbeat_age_seconds;
  const hbLabel = hbAge != null ? fmtDuration(hbAge) + ' ago' : '\u2014';
  f('ident').textContent = `PID ${inst.pid} \u00b7 ${inst.host || ''}`;
  f('timing').textContent = `started ${startedAge} ago \u00b7 hb ${hbLabel}`;
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fmtTokens(n) {
  if (typeof n !== 'number' || !isFinite(n)) return '?';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(0) + 'k';
  return String(n);
}

function renderBucket(b) {
  const pct = typeof b.percentage === 'number' ? b.percentage : 0;
  const cls = progressClass(pct);
  const active = b.is_active ? ' active' : '';
  const ident = escapeHtml(b.identity || b.account || 'anthropic');
  const source = b.source && b.source !== 'jsonl' && b.source !== 'empty'
    ? `<span class="bucket-source">${escapeHtml(b.source)}</span>` : '';

  // Limit provenance — tells the user whether the denominator is real or a guess.
  let limitTag = '';
  if (b.limit_source === 'detected' || b.limit_source === 'cached') {
    const hits = typeof b.limit_hit_count === 'number' ? b.limit_hit_count : 0;
    limitTag = `detected from ${hits} hit${hits === 1 ? '' : 's'}`;
  } else if (b.limit_source === 'override') {
    limitTag = 'override';
  } else {
    limitTag = 'default';
  }

  let forecast = '';
  if (typeof b.forecast_iters_to_90 === 'number' && b.forecast_iters_to_90 > 0) {
    forecast = `forecast: 90% in ~${b.forecast_iters_to_90} iter${b.forecast_iters_to_90 === 1 ? '' : 's'}`;
  } else if (pct >= 90 && typeof b.forecast_iters_to_100 === 'number' && b.forecast_iters_to_100 > 0) {
    forecast = `forecast: saturate in ~${b.forecast_iters_to_100} iter${b.forecast_iters_to_100 === 1 ? '' : 's'}`;
  }

  // Show a range when the single max-observed window is higher than the
  // conservative P90 ceiling we pace against — tells the user "you've burst
  // higher than we're treating as your cap". For accounts where P90 ≈ max
  // (consistent usage) the display stays compact.
  const isEmpirical = b.limit_source === 'detected' || b.limit_source === 'cached';
  const maxObs = typeof b.limit_max_observed === 'number' ? b.limit_max_observed : 0;
  const limitStr = (isEmpirical && maxObs > b.limit)
    ? `${fmtTokens(b.limit)} (P90) — max ${fmtTokens(maxObs)}`
    : fmtTokens(b.limit);

  const limitLine = `limit ${limitStr} · ${escapeHtml(limitTag)}`;
  const subtext = forecast ? `${limitLine} · ${forecast}` : limitLine;

  return `<div class="bucket">
    <div class="bar-row">
      <span class="ident${active}">${ident}${source}</span>
      <span>${pct.toFixed(1)}%</span>
    </div>
    <div class="bar"><span class="${cls}" style="width:${Math.min(100, pct)}%"></span></div>
    <div class="forecast">${subtext}</div>
  </div>`;
}

async function fetchTail(pid, el) {
  try {
    const res = await fetch(`/api/instances/${pid}/tail?bytes=${TAIL_BYTES}`);
    if (!res.ok) return;
    const text = await res.text();
    const newContent = text || '(no output yet)';
    // Only touch the DOM if content actually changed — avoids scroll jumps.
    if (el.textContent !== newContent) {
      const wasPinned = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
      el.textContent = newContent;
      if (wasPinned) el.scrollTop = el.scrollHeight;
    }
  } catch {
    // Leave the previous tail in place on transient failures.
  }
}

function sync(instances) {
  const seen = new Set();
  for (const inst of instances) {
    const pidKey = String(inst.pid);
    seen.add(pidKey);
    let card = grid.querySelector(`.card[data-pid="${pidKey}"]`);
    if (!card) {
      card = createCard(pidKey);
      grid.appendChild(card);
    }
    updateCard(card, inst);
    const tailEl = card.querySelector('[data-f="tail"]');
    if (tailEl) fetchTail(inst.pid, tailEl);
  }
  for (const card of [...grid.querySelectorAll('.card')]) {
    if (!seen.has(card.dataset.pid)) card.remove();
  }
  emptyEl.hidden = instances.length !== 0;
}

async function tick() {
  try {
    const res = await fetch('/api/instances');
    if (!res.ok) throw new Error('http ' + res.status);
    const body = await res.json();
    sync(body.instances || []);
    metaEl.textContent = `${(body.instances || []).length} instance(s) \u00b7 updated ${new Date().toLocaleTimeString()}`;
    dot.className = 'dot';
  } catch (err) {
    metaEl.textContent = 'disconnected \u2014 retrying\u2026';
    dot.className = 'dot stopped';
  }
}

tick();
setInterval(tick, REFRESH_MS);
</script>
</body>
</html>
"""

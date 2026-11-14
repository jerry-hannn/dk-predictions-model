// DKeX — Minimalist UI-First Client for DraftKings Predictions

const state = {
  mode: 'all',
  sport: 'all',
  market: 'all',
  viewMode: localStorage.getItem('dkex_view_mode') === 'table' ? 'table' : 'grid',
  ledgerFilter: 'ALL',
  searchQuery: '',
  picks: [],
  trades: [],
  lastCalcEval: null,
  backtestLoaded: false,
  guideLoaded: false,
  gridVisibleForPolling: true,
  authEnabled: false,
  authenticated: true,
};

let bannerTimer = null;
let autoRefreshTimer = null;
let calcDebounceTimer = null;

// --- Theme Management ---
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const sunIcon = document.getElementById('theme-icon-sun');
const moonIcon = document.getElementById('theme-icon-moon');

function syncThemeIcons() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const isDark = current === 'dark';
  sunIcon.hidden = !isDark;
  moonIcon.hidden = isDark;
}

themeToggleBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try {
    localStorage.setItem('dkex_theme', next);
  } catch (_) {}
  syncThemeIcons();
});
syncThemeIcons();

// --- Dialog Light-Dismiss Helper ---
function bindDialogDismiss(dialogEl, closeBtnIds = []) {
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    dialogEl.addEventListener('click', (event) => {
      if (event.target !== dialogEl) return;
      const rect = dialogEl.getBoundingClientRect();
      const inside =
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width;
      if (!inside) dialogEl.close();
    });
  }
  closeBtnIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', () => dialogEl.close());
  });
}

const authDialog = document.getElementById('auth-dialog');
const settingsDialog = document.getElementById('settings-dialog');
const newTradeDialog = document.getElementById('new-trade-dialog');
const settleDialog = document.getElementById('settle-dialog');

authDialog.addEventListener('cancel', (e) => {
  if (!state.authenticated) e.preventDefault();
});

bindDialogDismiss(settingsDialog, ['close-settings-btn', 'cancel-settings-btn']);
bindDialogDismiss(newTradeDialog, ['close-new-trade-btn', 'cancel-new-trade-btn']);
bindDialogDismiss(settleDialog, ['close-dialog-btn', 'cancel-dialog-btn']);

// --- Authenticated Fetch Wrapper ---
async function apiFetch(url, options = {}) {
  const resp = await fetch(url, options);
  if (resp.status === 401 && !url.startsWith('/api/auth/')) {
    state.authenticated = false;
    promptPasscodeUnlock();
    throw new Error('Authentication required');
  }
  return resp;
}

function promptPasscodeUnlock() {
  document.getElementById('auth-error-msg').hidden = true;
  document.getElementById('auth-passcode-input').value = '';
  if (!authDialog.open) {
    authDialog.showModal();
  }
}

document.getElementById('auth-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const passcode = document.getElementById('auth-passcode-input').value;
  const errEl = document.getElementById('auth-error-msg');
  const btn = document.getElementById('auth-submit-btn');
  btn.disabled = true;
  errEl.hidden = true;

  try {
    const resp = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ passcode }),
    });
    if (!resp.ok) {
      errEl.hidden = false;
      return;
    }
    state.authenticated = true;
    authDialog.close();
    await bootWorkspace();
  } catch (_) {
    errEl.textContent = 'Connection error. Try again.';
    errEl.hidden = false;
  } finally {
    btn.disabled = false;
  }
});

document.getElementById('logout-btn').addEventListener('click', async () => {
  await fetch('/api/auth/logout', { method: 'POST' });
  state.authenticated = false;
  promptPasscodeUnlock();
});

// --- Tabs Navigation ---
const tablist = document.querySelector('.nav-tabs');
const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
const panels = tabs.map((t) => document.getElementById(t.getAttribute('aria-controls')));

function selectTab(selected) {
  tabs.forEach((tab, idx) => {
    const isSelected = tab === selected;
    tab.setAttribute('aria-selected', String(isSelected));
    tab.tabIndex = isSelected ? 0 : -1;
    panels[idx].hidden = !isSelected;
  });

  syncAutoRefreshTimer();

  if (selected.id === 'tab-ledger') loadLedger();
  if (selected.id === 'tab-backtest' && !state.backtestLoaded) loadBacktest();
  if (selected.id === 'tab-guide' && !state.guideLoaded) loadGuide();
  if (selected.id === 'tab-calculator') runCustomCalculator();
}

tabs.forEach((tab, idx) => {
  tab.tabIndex = idx === 0 ? 0 : -1;
  tab.addEventListener('click', () => selectTab(tab));
});

tablist.addEventListener('keydown', (e) => {
  const cur = tabs.indexOf(document.activeElement);
  if (cur === -1) return;
  let next = -1;
  if (e.key === 'ArrowRight') next = (cur + 1) % tabs.length;
  if (e.key === 'ArrowLeft') next = (cur - 1 + tabs.length) % tabs.length;
  if (e.key === 'Home') next = 0;
  if (e.key === 'End') next = tabs.length - 1;
  if (next !== -1) {
    e.preventDefault();
    tabs[next].focus();
    selectTab(tabs[next]);
  }
});

// --- Toast / Status Banner ---
function showBanner(msg, isError = false) {
  const wrap = document.getElementById('status-banner');
  const txt = document.getElementById('status-banner-text');
  txt.textContent = msg;
  wrap.classList.toggle('error', isError);
  wrap.hidden = false;
  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => {
    wrap.hidden = true;
  }, 5000);
}

document.getElementById('dismiss-banner-btn').addEventListener('click', () => {
  document.getElementById('status-banner').hidden = true;
});

// --- Formatting & State Helpers ---
function getBankroll() {
  const raw = parseFloat(document.getElementById('global-bankroll').value);
  return Number.isFinite(raw) && raw > 0 ? raw : 1000.0;
}

function getFee() {
  const raw = parseFloat(document.getElementById('global-fee').value);
  return Number.isFinite(raw) && raw >= 0 ? raw : 0.015;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str ?? '';
  return div.innerHTML;
}

function formatMoney(val) {
  const n = Number(val) || 0;
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatCompactDollars(val) {
  const n = Number(val) || 0;
  return Number.isInteger(n)
    ? `$${n.toLocaleString('en-US')}`
    : `$${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function updateHeaderChip() {
  const bk = getBankroll();
  const feeCents = (getFee() * 100).toFixed(1).replace(/\.0$/, '');
  document.getElementById('header-bankroll-chip').textContent = `${formatCompactDollars(bk)} · ${feeCents}¢`;
}

function tierPillClass(tier) {
  if (tier === 'A+' || tier === 'A') return 'tier-pill tier-high';
  if (tier === 'B') return 'tier-pill tier-mid';
  return 'tier-pill tier-pass';
}

function formatAgeText(ageSec) {
  if (ageSec == null) return '';
  const s = Math.max(0, Math.round(Number(ageSec)));
  if (s < 60) return ` · ${s}s`;
  return ` · ${Math.round(s / 60)}m`;
}

function updateOddsApiPill(connected, cachedCount = 0, quota = null) {
  const pill = document.getElementById('odds-api-pill');
  const txt = document.getElementById('odds-api-text');
  if (connected) {
    const rem = quota && quota.requests_remaining != null ? ` · ${quota.requests_remaining} cr` : '';
    txt.textContent = cachedCount > 0 ? `Sharp (${cachedCount})${rem}` : `Sharp API${rem}`;
    pill.classList.add('connected');
  } else {
    txt.textContent = 'ESPN / DK Feed';
    pill.classList.remove('connected');
  }
}

function updateKPIs(ledgerSummary, picksList) {
  if (ledgerSummary) {
    document.getElementById('kpi-bankroll').textContent = `$${formatMoney(ledgerSummary.current_bankroll)}`;

    const pnlEl = document.getElementById('kpi-pnl');
    const pnl = Number(ledgerSummary.realized_pnl) || 0;
    const pnlSign = pnl > 0 ? '+' : pnl < 0 ? '-' : '';
    pnlEl.textContent = `${pnlSign}$${formatMoney(Math.abs(pnl))}`;
    pnlEl.className = `summary-value ${pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : ''}`;

    const roi = Number(ledgerSummary.roi_pct) || 0;
    const roiSign = roi > 0 ? '+' : '';
    document.getElementById('kpi-roi').textContent = `(${roiSign}${(roi * 100).toFixed(1)}%)`;

    document.getElementById('kpi-exposure').textContent = `$${formatMoney(ledgerSummary.open_exposure)}`;

    document.getElementById('kpi-hitrate').textContent =
      ledgerSummary.hit_rate !== null ? `${(ledgerSummary.hit_rate * 100).toFixed(1)}%` : '—';
    document.getElementById('kpi-exp-hitrate').textContent =
      ledgerSummary.expected_win_rate !== null
        ? `exp ${(ledgerSummary.expected_win_rate * 100).toFixed(0)}%`
        : '';

    const openBadge = document.getElementById('open-trades-badge');
    const openCnt = Number(ledgerSummary.open_trades_count) || 0;
    openBadge.textContent = String(openCnt);
    openBadge.hidden = openCnt === 0;

    const subMeta = document.getElementById('ledger-sub-meta');
    if (ledgerSummary.avg_clv_cents !== null && ledgerSummary.avg_clv_cents !== undefined) {
      const clvSign = ledgerSummary.avg_clv_cents >= 0 ? '+' : '';
      subMeta.textContent = `Avg Closing Line Value: ${clvSign}${ledgerSummary.avg_clv_cents.toFixed(1)}¢ · ${ledgerSummary.settled_trades_count} settled, ${openCnt} open.`;
    }
  }

  if (picksList) {
    const locks = picksList.filter((p) => p.confidence_tier === 'A+' || p.confidence_tier === 'A').length;
    const qualifying = picksList.filter((p) => p.confidence_tier !== 'PASS').length;
    const liveCnt = picksList.filter((p) => p.game_state === 'LIVE' && p.confidence_tier !== 'PASS').length;
    document.getElementById('kpi-locks').textContent = String(locks);
    document.getElementById('kpi-total-picks').textContent =
      liveCnt > 0 ? `${qualifying} active · ${liveCnt} live` : `${qualifying} qualifying`;

    const livePill = document.getElementById('live-pill-count');
    const allLive = picksList.filter((p) => p.game_state === 'LIVE').length;
    livePill.textContent = String(allLive);
    livePill.hidden = allLive === 0;
  }
}

// --- Auto-Refresh Live Polling ---
const picksGridEl = document.getElementById('picks-grid');
picksGridEl.addEventListener('contentvisibilityautostatechange', (event) => {
  state.gridVisibleForPolling = !event.skipped;
  syncAutoRefreshTimer();
});

document.addEventListener('visibilitychange', syncAutoRefreshTimer);

function syncAutoRefreshTimer() {
  if (autoRefreshTimer) {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
  const enabled = document.getElementById('auto-refresh-live').checked;
  const picksTabVisible = !document.getElementById('panel-picks').hidden;
  if (!enabled || document.hidden || !picksTabVisible || !state.gridVisibleForPolling) {
    return;
  }
  autoRefreshTimer = setInterval(async () => {
    if (!document.hidden && !document.getElementById('panel-picks').hidden) {
      await refreshLiveLines(true);
    }
  }, 60000);
}

// --- Server Config & Settings Dialog ---
async function loadServerConfig() {
  try {
    const res = await apiFetch('/api/config');
    if (!res.ok) return;
    const cfg = await res.json();

    document.getElementById('global-bankroll').value = Number(cfg.bankroll).toFixed(2);

    const feeSelect = document.getElementById('global-fee');
    const matchingOpt = Array.from(feeSelect.options).find(
      (o) => Math.abs(parseFloat(o.value) - Number(cfg.default_fee_dollars)) < 1e-4
    );
    if (matchingOpt) feeSelect.value = matchingOpt.value;

    document.getElementById('setting-kelly').value = Number(cfg.kelly_fraction || 0.25).toFixed(2);
    document.getElementById('setting-max-bet').value = (Number(cfg.max_single_bet_pct || 0.04) * 100).toFixed(1);

    const minProbPct = Math.round(Number(cfg.min_win_prob) * 100);
    document.getElementById('filter-min-prob').value = String(minProbPct);
    document.getElementById('min-prob-val').textContent = `${minProbPct}%`;

    const minEdgePct = Number((Number(cfg.min_net_edge) * 100).toFixed(1));
    document.getElementById('filter-min-edge').value = String(minEdgePct);
    document.getElementById('min-edge-val').textContent = `+${minEdgePct.toFixed(1)}%`;

    const keyHint = document.getElementById('setting-odds-hint');
    if (cfg.odds_api_connected) {
      keyHint.textContent = `Connected (${cfg.odds_api_key_masked || 'active'}). Enter a new key to replace, or type CLEAR to remove.`;
    } else {
      keyHint.textContent = 'Zero-key mode uses public ESPN & DraftKings feeds automatically.';
    }

    document.getElementById('logout-btn').hidden = !cfg.auth_enabled;

    updateHeaderChip();
    updateOddsApiPill(
      Boolean(cfg.odds_api_connected),
      Number(cfg.sharp_events_cached || 0),
      cfg.odds_quota || null
    );
  } catch (err) {
    if (err.message !== 'Authentication required') {
      console.error('Failed loading /api/config:', err);
    }
  }
}

document.getElementById('open-settings-btn').addEventListener('click', () => {
  document.getElementById('setting-odds-key').value = '';
  settingsDialog.showModal();
});

document.getElementById('settings-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const keyRaw = document.getElementById('setting-odds-key').value.trim();
  const payload = {
    bankroll: getBankroll(),
    default_fee_dollars: getFee(),
    kelly_fraction: Math.min(1.0, Math.max(0.05, parseFloat(document.getElementById('setting-kelly').value) || 0.25)),
    max_single_bet_pct: Math.min(0.25, Math.max(0.01, (parseFloat(document.getElementById('setting-max-bet').value) || 4) / 100)),
    min_win_prob: parseFloat(document.getElementById('filter-min-prob').value) / 100,
    min_net_edge: parseFloat(document.getElementById('filter-min-edge').value) / 100,
  };
  if (keyRaw.toUpperCase() === 'CLEAR') {
    payload.the_odds_api_key = '';
  } else if (keyRaw !== '') {
    payload.the_odds_api_key = keyRaw;
  }

  try {
    const resp = await apiFetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    settingsDialog.close();
    await loadServerConfig();
    await loadPicks();
    if (!document.getElementById('panel-calculator').hidden) runCustomCalculator();
    if (!document.getElementById('panel-ledger').hidden) loadLedger();
    showBanner('Workspace settings saved.');
  } catch (err) {
    showBanner(`Failed saving settings: ${err}`, true);
  }
});

// --- Markets Toolbar & Filtering ---
document.querySelectorAll('[data-seg]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.seg;
    const val = btn.dataset.val;
    state[group] = val;
    document.querySelectorAll(`[data-seg="${group}"]`).forEach((b) => {
      b.classList.toggle('active', b === btn);
    });
    loadPicks();
  });
});

document.getElementById('toggle-filters-btn').addEventListener('click', (e) => {
  const form = document.getElementById('filters-form');
  const nextHidden = !form.hidden;
  form.hidden = nextHidden;
  e.currentTarget.setAttribute('aria-expanded', String(!nextHidden));
});

function applyViewModeUI() {
  const isTable = state.viewMode === 'table';
  document.getElementById('view-grid-btn').classList.toggle('active', !isTable);
  document.getElementById('view-table-btn').classList.toggle('active', isTable);
  document.getElementById('picks-grid').hidden = isTable;
  document.getElementById('picks-table-wrap').hidden = !isTable;
}

document.getElementById('view-grid-btn').addEventListener('click', () => {
  state.viewMode = 'grid';
  try { localStorage.setItem('dkex_view_mode', 'grid'); } catch (_) {}
  applyViewModeUI();
  renderPicks();
});

document.getElementById('view-table-btn').addEventListener('click', () => {
  state.viewMode = 'table';
  try { localStorage.setItem('dkex_view_mode', 'table'); } catch (_) {}
  applyViewModeUI();
  renderPicks();
});
applyViewModeUI();

document.getElementById('filter-search').addEventListener('input', (e) => {
  state.searchQuery = e.target.value.trim().toLowerCase();
  renderPicks();
});

document.getElementById('filters-form').addEventListener('submit', (e) => e.preventDefault());

document.getElementById('filter-min-prob').addEventListener('input', (e) => {
  document.getElementById('min-prob-val').textContent = `${e.target.value}%`;
});
document.getElementById('filter-min-edge').addEventListener('input', (e) => {
  document.getElementById('min-edge-val').textContent = `+${parseFloat(e.target.value).toFixed(1)}%`;
});

['filter-date', 'filter-min-prob', 'filter-min-edge', 'filter-show-all', 'filter-live-props'].forEach((id) => {
  document.getElementById(id).addEventListener('change', loadPicks);
});
document.getElementById('auto-refresh-live').addEventListener('change', syncAutoRefreshTimer);

// --- Load & Render Markets ---
async function loadPicks() {
  const minProb = parseFloat(document.getElementById('filter-min-prob').value) / 100;
  const minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  const showAll = document.getElementById('filter-show-all').checked;
  const liveProps = document.getElementById('filter-live-props').checked;
  const dateVal = document.getElementById('filter-date').value;

  const params = new URLSearchParams({
    mode: state.mode,
    sport: state.sport,
    market: state.market,
    min_prob: String(minProb),
    min_edge: String(minEdge),
    fee: String(getFee()),
    bankroll: String(getBankroll()),
    include_passes: String(showAll),
    live_props: String(liveProps),
  });
  if (dateVal) params.set('date', dateVal);

  const grid = document.getElementById('picks-grid');
  if (!state.picks.length) {
    grid.innerHTML = '<div class="empty-state"><p>Pricing pre-game and live contracts...</p></div>';
  }

  try {
    const res = await apiFetch(`/api/picks?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.picks = data.picks || [];
    if (data.odds_quota) {
      updateOddsApiPill(Boolean(data.odds_api_connected), undefined, data.odds_quota);
    }
    updateKPIs(data.ledger_summary, state.picks);
    renderPicks();
  } catch (err) {
    if (err.message !== 'Authentication required') {
      grid.innerHTML = `<div class="empty-state"><h3>Unable to load markets</h3><p>${escapeHtml(String(err))}</p></div>`;
    }
  }
}

function filteredPicksWithIndices() {
  const q = state.searchQuery;
  return state.picks
    .map((p, idx) => ({ p, idx }))
    .filter(({ p }) => {
      if (!q) return true;
      const hay = `${p.selection_label} ${p.event_name} ${p.short_name || ''} ${p.market_type} ${p.sport}`.toLowerCase();
      return hay.includes(q);
    });
}

function buildPickCardInnerHtml(p, idx) {
  const isLive = p.game_state === 'LIVE';
  const edgeCls = p.net_edge >= 0 ? 'positive' : 'negative';
  const edgeSign = p.net_edge >= 0 ? '+' : '';

  const liveBadge = isLive
    ? `<span class="live-indicator"><span class="live-dot"></span>LIVE ${escapeHtml(p.status_detail || '')}${p.score_display ? ` · ${escapeHtml(p.score_display)}` : ''}</span>`
    : '';

  const drivers = p.drivers || [];
  const primaryDriver = drivers[0] ? `<p class="driver-primary">${escapeHtml(drivers[0])}</p>` : '';
  const extraDrivers = drivers.slice(1);
  const hasExtra = extraDrivers.length > 0 || Boolean(p.injury_notes) || Boolean(p.line_movement);

  const detailsHtml = hasExtra
    ? `<details class="card-details">
        <summary>Details &amp; exit (${(p.fair_midgame_exit_price * 100).toFixed(0)}¢ target)</summary>
        <div class="details-body">
          ${p.line_movement ? `<p class="mono">${escapeHtml(p.line_movement)}</p>` : ''}
          ${extraDrivers.map((d) => `<p>• ${escapeHtml(d)}</p>`).join('')}
          ${p.injury_notes ? `<p class="warning-text">Injuries: ${escapeHtml(p.injury_notes)}</p>` : ''}
          <p>Question: ${escapeHtml(p.contract_question || p.selection_label)}</p>
        </div>
      </details>`
    : '';

  const qty = Math.max(1, p.recommended_contracts || 1);
  const btnLabel =
    p.confidence_tier === 'PASS'
      ? `Log (${qty}x)`
      : `Log ${escapeHtml(p.recommended_side)} · ${p.recommended_contracts}x ($${p.recommended_outlay.toFixed(2)})`;
  const btnCls = p.confidence_tier === 'PASS' ? 'btn btn-secondary btn-sm log-trade-btn' : 'btn btn-primary btn-sm log-trade-btn';

  return `
    <div class="pick-header">
      <div class="pick-meta">
        <span class="sport-label">${escapeHtml(p.sport)}</span>
        <span>·</span>
        <span>${escapeHtml(p.market_type)}</span>
        ${liveBadge}
      </div>
      <span class="${tierPillClass(p.confidence_tier)}">${escapeHtml(p.confidence_tier)} ${p.confidence_score}</span>
    </div>

    <div>
      <h3 class="pick-title">${escapeHtml(p.selection_label)}</h3>
      <p class="pick-subtitle">${escapeHtml(p.short_name || p.event_name)} · ${escapeHtml(p.line_source || 'ESPN/DK')}${formatAgeText(p.odds_age_seconds)}</p>
    </div>

    <div class="metrics-strip">
      <div class="stat-col">
        <span>Side / Ask</span>
        <strong>${escapeHtml(p.recommended_side)} ${(p.contract_price * 100).toFixed(0)}¢</strong>
      </div>
      <div class="stat-col">
        <span>Win Prob</span>
        <strong>${(p.model_prob * 100).toFixed(1)}%</strong>
      </div>
      <div class="stat-col">
        <span>Net Edge</span>
        <strong class="${edgeCls}">${edgeSign}${(p.net_edge * 100).toFixed(1)}%</strong>
      </div>
      <div class="stat-col">
        <span>Max Buy</span>
        <strong>${(p.fair_entry_max_price * 100).toFixed(0)}¢</strong>
      </div>
    </div>

    ${primaryDriver}
    ${detailsHtml}

    <div class="pick-actions">
      <div class="inline-pricer" title="Test live DraftKings contract ask price">
        <select id="dk-side-${idx}" class="dk-side-select" data-idx="${idx}" aria-label="Contract side">
          <option value="YES" ${p.recommended_side === 'YES' ? 'selected' : ''}>YES</option>
          <option value="NO" ${p.recommended_side === 'NO' ? 'selected' : ''}>NO</option>
        </select>
        <div class="dk-cents-wrap">
          <input
            id="dk-px-${idx}"
            class="dk-cents-input"
            type="number"
            min="1"
            max="99"
            step="0.5"
            value="${Number((p.contract_price * 100).toFixed(1))}"
            data-idx="${idx}"
            aria-label="Contract price in cents"
          >
          <span class="cents-suffix">¢</span>
        </div>
      </div>
      <button type="button" class="${btnCls}" data-idx="${idx}">${btnLabel}</button>
    </div>
  `;
}

async function logPickByIndex(idx) {
  const pick = state.picks[idx];
  if (!pick) return;
  const qty = Math.max(1, pick.recommended_contracts || 1);
  try {
    const resp = await apiFetch('/api/ledger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sport: pick.sport,
        event_name: pick.short_name || pick.event_name,
        market_type: pick.market_type,
        selection_label: pick.selection_label,
        side: pick.recommended_side,
        entry_price: pick.contract_price,
        fee_per_contract: pick.fee_per_contract,
        contracts: qty,
        model_prob: pick.model_prob,
        fair_exit_price: pick.fair_midgame_exit_price,
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showBanner(
      `Logged ${qty}x ${pick.recommended_side} "${pick.selection_label}" @ ${(pick.contract_price * 100).toFixed(1)}¢ ($${(qty * pick.all_in_cost).toFixed(2)})`
    );
    const led = await (await apiFetch(`/api/ledger?bankroll=${getBankroll()}`)).json();
    updateKPIs(led.summary, state.picks);
  } catch (err) {
    showBanner(`Failed logging position: ${err}`, true);
  }
}

async function reevaluateSingleCard(idx) {
  const pick = state.picks[idx];
  if (!pick) return;
  const sideEl = document.getElementById(`dk-side-${idx}`);
  const pxEl = document.getElementById(`dk-px-${idx}`);
  const side = sideEl ? sideEl.value : pick.recommended_side;
  const cents = Math.min(99, Math.max(1, parseFloat(pxEl ? pxEl.value : '50') || 50));
  const minProb = parseFloat(document.getElementById('filter-min-prob').value) / 100;
  const minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;

  try {
    const resp = await apiFetch('/api/evaluate-contract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        yes_true_prob: pick.yes_true_prob,
        side,
        contract_price: cents / 100,
        fee_per_contract: getFee(),
        bankroll: getBankroll(),
        min_win_prob: minProb,
        min_net_edge: minEdge,
        is_live: pick.game_state === 'LIVE',
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const updated = await resp.json();
    pick.recommended_side = updated.side;
    Object.assign(pick, updated);

    const cardEl = document.querySelector(`.pick-card[data-idx="${idx}"]`);
    if (cardEl) {
      cardEl.innerHTML = buildPickCardInnerHtml(pick, idx);
      bindCardEvents(cardEl);
    }
    updateKPIs(null, state.picks);
  } catch (err) {
    showBanner(`Could not re-evaluate contract: ${err}`, true);
  }
}

function bindCardEvents(container) {
  container.querySelectorAll('.dk-side-select').forEach((sel) => {
    sel.addEventListener('change', (e) => {
      reevaluateSingleCard(parseInt(e.currentTarget.dataset.idx, 10));
    });
  });

  container.querySelectorAll('.dk-cents-input').forEach((inp) => {
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        reevaluateSingleCard(parseInt(e.currentTarget.dataset.idx, 10));
      }
    });
    inp.addEventListener('change', (e) => {
      reevaluateSingleCard(parseInt(e.currentTarget.dataset.idx, 10));
    });
  });

  container.querySelectorAll('.log-trade-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      logPickByIndex(parseInt(e.currentTarget.dataset.idx, 10));
    });
  });
}

function renderPicks() {
  const rows = filteredPicksWithIndices();
  const grid = document.getElementById('picks-grid');
  const tbody = document.getElementById('picks-table-tbody');

  if (!rows.length) {
    const hint = state.searchQuery
      ? `No contracts matching "${state.searchQuery}". Clear the search box to see all active lines.`
      : state.mode === 'live'
        ? 'No live in-play games currently clear your edge filter. Switch timing to "All" or click Refresh.'
        : 'Open "Filters" to lower Min Win Prob, enable "Show PASS contracts", or click Refresh to pull latest lines.';

    const emptyHtml = `<div class="empty-state"><h3>No matching contracts</h3><p>${escapeHtml(hint)}</p></div>`;
    grid.innerHTML = emptyHtml;
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:2rem;">${escapeHtml(hint)}</td></tr>`;
    return;
  }

  if (state.viewMode === 'grid') {
    grid.innerHTML = rows
      .map(({ p, idx }) => `<article class="pick-card" data-idx="${idx}">${buildPickCardInnerHtml(p, idx)}</article>`)
      .join('');
    bindCardEvents(grid);
  } else {
    tbody.innerHTML = rows
      .map(({ p, idx }) => {
        const isLive = p.game_state === 'LIVE';
        const edgeCls = p.net_edge >= 0 ? 'positive' : 'negative';
        const edgeSign = p.net_edge >= 0 ? '+' : '';
        const stateLabel = isLive
          ? `<span class="live-indicator"><span class="live-dot"></span>${escapeHtml(p.score_display || p.status_detail || 'LIVE')}</span>`
          : escapeHtml(p.short_name || p.event_name);
        const qty = Math.max(1, p.recommended_contracts || 1);
        return `
          <tr>
            <td><span class="${tierPillClass(p.confidence_tier)}">${escapeHtml(p.confidence_tier)} ${p.confidence_score}</span></td>
            <td><span class="sport-label">${escapeHtml(p.sport)}</span> · ${stateLabel}</td>
            <td><strong>${escapeHtml(p.selection_label)}</strong></td>
            <td class="mono">${escapeHtml(p.recommended_side)}</td>
            <td class="num-col">${(p.contract_price * 100).toFixed(0)}¢</td>
            <td class="num-col">${(p.fair_entry_max_price * 100).toFixed(0)}¢</td>
            <td class="num-col">${(p.model_prob * 100).toFixed(1)}%</td>
            <td class="num-col ${edgeCls}">${edgeSign}${(p.net_edge * 100).toFixed(1)}%</td>
            <td class="num-col">${p.recommended_contracts}x ($${p.recommended_outlay.toFixed(2)})</td>
            <td><button type="button" class="btn btn-secondary btn-sm log-trade-btn" data-idx="${idx}">Log ${qty}x</button></td>
          </tr>
        `;
      })
      .join('');
    bindCardEvents(tbody);
  }
}

// --- Standalone Contract Analyzer ---
['calc-side-yes', 'calc-side-no'].forEach((id) => {
  document.getElementById(id).addEventListener('click', (e) => {
    const side = e.currentTarget.dataset.side;
    document.getElementById('calc-side').value = side;
    document.getElementById('calc-side-yes').classList.toggle('active', side === 'YES');
    document.getElementById('calc-side-no').classList.toggle('active', side === 'NO');
    runCustomCalculator();
  });
});

async function runCustomCalculator(e) {
  if (e) e.preventDefault();
  const side = document.getElementById('calc-side').value;
  const yesProb = Math.min(99.5, Math.max(0.5, parseFloat(document.getElementById('calc-prob').value) || 65)) / 100;
  const price = Math.min(99, Math.max(1, parseFloat(document.getElementById('calc-price').value) || 55)) / 100;
  const isLive = document.getElementById('calc-is-live').checked;

  try {
    const resp = await apiFetch('/api/evaluate-contract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        yes_true_prob: yesProb,
        side,
        contract_price: price,
        fee_per_contract: getFee(),
        bankroll: getBankroll(),
        min_win_prob: parseFloat(document.getElementById('filter-min-prob').value) / 100,
        min_net_edge: parseFloat(document.getElementById('filter-min-edge').value) / 100,
        is_live: isLive,
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const ev = await resp.json();
    state.lastCalcEval = ev;

    const edgeCls = ev.net_edge >= 0 ? 'positive' : 'negative';
    const verdictEl = document.getElementById('calc-verdict-title');
    verdictEl.textContent =
      ev.confidence_tier === 'PASS'
        ? 'PASS — Insufficient Edge at Ask'
        : `BUY ${ev.side} · Conviction ${ev.confidence_tier}`;
    verdictEl.className = `verdict-title ${ev.confidence_tier === 'PASS' ? '' : 'positive'}`;

    const pillEl = document.getElementById('calc-tier-pill');
    pillEl.textContent = `${ev.confidence_tier} · ${ev.confidence_score}/100`;
    pillEl.className = tierPillClass(ev.confidence_tier);

    document.getElementById('calc-output-body').innerHTML = `
      <div class="kv-row"><span>Side True Win Probability</span><strong>${(ev.model_prob * 100).toFixed(1)}%</strong></div>
      <div class="kv-row"><span>All-In Cost (Ask + Fee)</span><strong>${(ev.all_in_cost * 100).toFixed(2)}¢ (${ev.american_equivalent > 0 ? '+' : ''}${ev.american_equivalent})</strong></div>
      <div class="kv-row"><span>Break-Even Win Rate</span><strong>${(ev.breakeven_prob * 100).toFixed(2)}%</strong></div>
      <div class="kv-row"><span>Post-Fee Net Edge</span><strong class="${edgeCls}">${ev.net_edge >= 0 ? '+' : ''}${(ev.net_edge * 100).toFixed(2)}%</strong></div>
      <div class="kv-row"><span>Expected ROI</span><strong class="${edgeCls}">${ev.roi_pct >= 0 ? '+' : ''}${(ev.roi_pct * 100).toFixed(2)}%</strong></div>
      <div class="kv-row"><span>Max Limit Entry Price</span><strong>${(ev.fair_entry_max_price * 100).toFixed(1)}¢</strong></div>
      <div class="kv-row"><span>Fair Mid-Game Exit Target</span><strong>${(ev.fair_midgame_exit_price * 100).toFixed(1)}¢</strong></div>
      <div class="kv-row"><span>${isLive ? 'In-Play ' : ''}Quarter-Kelly Position</span><strong>${ev.recommended_contracts} contracts ($${ev.recommended_outlay.toFixed(2)})</strong></div>
    `;
  } catch (err) {
    if (err.message !== 'Authentication required') {
      showBanner(`Analyzer error: ${err}`, true);
    }
  }
}

document.getElementById('custom-calc-form').addEventListener('submit', runCustomCalculator);
['calc-prob', 'calc-price', 'calc-is-live'].forEach((id) => {
  document.getElementById(id).addEventListener('input', () => {
    if (calcDebounceTimer) clearTimeout(calcDebounceTimer);
    calcDebounceTimer = setTimeout(() => runCustomCalculator(), 180);
  });
});

document.getElementById('calc-log-btn').addEventListener('click', () => {
  const ev = state.lastCalcEval;
  if (ev) {
    document.getElementById('nt-side').value = ev.side;
    document.getElementById('nt-contracts').value = String(Math.max(1, ev.recommended_contracts || 1));
    document.getElementById('nt-price').value = String(Number((ev.contract_price * 100).toFixed(1)));
    document.getElementById('nt-prob').value = String(Number((ev.model_prob * 100).toFixed(1)));
  }
  newTradeDialog.showModal();
});

// --- Trade Ledger & Manual Trade Logging ---
document.querySelectorAll('[data-ledger-filter]').forEach((btn) => {
  btn.addEventListener('click', () => {
    state.ledgerFilter = btn.dataset.ledgerFilter;
    document.querySelectorAll('[data-ledger-filter]').forEach((b) => {
      b.classList.toggle('active', b === btn);
    });
    renderLedgerRows();
  });
});

document.getElementById('new-trade-btn').addEventListener('click', () => {
  document.getElementById('nt-event').value = '';
  document.getElementById('nt-selection').value = '';
  newTradeDialog.showModal();
});

document.getElementById('new-trade-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const entryPrice = Math.min(99, Math.max(1, parseFloat(document.getElementById('nt-price').value) || 55)) / 100;
  const modelProb = Math.min(99, Math.max(1, parseFloat(document.getElementById('nt-prob').value) || 60)) / 100;
  const fee = getFee();
  const fairExit = Math.min(0.99, Math.max(0.01, modelProb + fee));

  try {
    const resp = await apiFetch('/api/ledger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sport: document.getElementById('nt-sport').value,
        market_type: document.getElementById('nt-market').value,
        event_name: document.getElementById('nt-event').value.trim(),
        selection_label: document.getElementById('nt-selection').value.trim(),
        side: document.getElementById('nt-side').value,
        contracts: Math.max(1, parseInt(document.getElementById('nt-contracts').value, 10) || 1),
        entry_price: entryPrice,
        fee_per_contract: fee,
        model_prob: modelProb,
        fair_exit_price: fairExit,
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    newTradeDialog.close();
    showBanner('Position logged to ledger.');
    await loadLedger();
  } catch (err) {
    showBanner(`Could not log position: ${err}`, true);
  }
});

function renderLedgerRows() {
  const tbody = document.getElementById('ledger-tbody');
  const filtered = state.trades.filter((t) => {
    if (state.ledgerFilter === 'OPEN') return t.status === 'OPEN';
    if (state.ledgerFilter === 'SETTLED') return t.status !== 'OPEN';
    return true;
  });

  if (!filtered.length) {
    const msg = state.trades.length
      ? `No ${state.ledgerFilter.toLowerCase()} positions.`
      : 'No positions logged yet. Log directly from any Market card or click "+ Log Position".';
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:2rem;">${escapeHtml(msg)}</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((t) => {
      const pnlCls = t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : '';
      const pnlStr = t.status === 'OPEN' ? '—' : `${t.pnl >= 0 ? '+' : '-'}$${Math.abs(t.pnl).toFixed(2)}`;
      const probTxt =
        t.status === 'OPEN' && t.live_model_prob != null
          ? `${(t.model_prob * 100).toFixed(0)}% → <strong>${(t.live_model_prob * 100).toFixed(1)}%</strong>`
          : `${(t.model_prob * 100).toFixed(1)}%`;
      const guidanceTxt =
        t.status === 'OPEN' && t.live_guidance
          ? escapeHtml(t.live_guidance)
          : `Exit ${(t.fair_exit_price * 100).toFixed(0)}¢`;

      return `
        <tr>
          <td>
            <strong>${escapeHtml(t.selection_label)}</strong>
            <div style="font-size:0.72rem;color:var(--text-muted);">${escapeHtml(t.event_name)} · ${escapeHtml(t.market_type)}</div>
          </td>
          <td class="mono">${escapeHtml(t.sport)}</td>
          <td class="mono"><strong>${escapeHtml(t.side)}</strong></td>
          <td class="num-col">${t.contracts} @ ${(t.entry_price * 100).toFixed(1)}¢</td>
          <td class="num-col">$${t.total_cost.toFixed(2)}</td>
          <td class="num-col">${probTxt}</td>
          <td style="font-size:0.77rem;color:var(--text-secondary);">${guidanceTxt}</td>
          <td><span class="status-badge ${escapeHtml(t.status)}">${escapeHtml(t.status)}</span></td>
          <td class="num-col ${pnlCls}"><strong>${pnlStr}</strong></td>
          <td>
            <div class="row-actions">
              <button type="button" class="btn btn-secondary btn-sm settle-open-btn" data-id="${t.id}">Settle</button>
              <button type="button" class="btn btn-danger-ghost btn-sm delete-trade-btn" data-id="${t.id}" aria-label="Remove position">Remove</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join('');

  tbody.querySelectorAll('.settle-open-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const id = parseInt(e.currentTarget.dataset.id, 10);
      const trade = state.trades.find((x) => x.id === id);
      if (!trade) return;
      document.getElementById('settle-trade-id').value = String(trade.id);
      document.getElementById('settle-trade-summary').textContent =
        `${trade.contracts}x ${trade.side} — ${trade.selection_label} (All-in cost $${trade.total_cost.toFixed(2)})`;
      document.getElementById('settle-status').value = 'WON';
      document.getElementById('sold-fields').hidden = true;
      document.getElementById('settle-closing-price').value = '';
      settleDialog.showModal();
    });
  });

  tbody.querySelectorAll('.delete-trade-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      const id = parseInt(e.currentTarget.dataset.id, 10);
      const delResp = await apiFetch(`/api/ledger/${id}`, { method: 'DELETE' });
      if (delResp.ok) {
        showBanner(`Removed position #${id}.`);
        loadLedger();
      }
    });
  });
}

async function loadLedger() {
  try {
    const resp = await apiFetch(`/api/ledger?bankroll=${getBankroll()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.trades = data.trades || [];
    updateKPIs(data.summary, state.picks);
    renderLedgerRows();
  } catch (err) {
    if (err.message !== 'Authentication required') {
      showBanner(`Failed loading ledger: ${err}`, true);
    }
  }
}

document.getElementById('settle-status').addEventListener('change', (e) => {
  document.getElementById('sold-fields').hidden = e.target.value !== 'SOLD';
});

document.getElementById('settle-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const id = parseInt(document.getElementById('settle-trade-id').value, 10);
  const status = document.getElementById('settle-status').value;
  const sellPxRaw = parseFloat(document.getElementById('settle-sell-price').value || '75') / 100;
  const sellFeeRaw = parseFloat(document.getElementById('settle-sell-fee').value || '1.5') / 100;
  const clvRaw = document.getElementById('settle-closing-price').value;

  const payload = {
    status,
    exit_or_settle_price: status === 'SOLD' ? sellPxRaw : null,
    exit_fee_per_contract: status === 'SOLD' ? sellFeeRaw : 0.0,
    closing_price: clvRaw ? parseFloat(clvRaw) / 100 : null,
  };

  const resp = await apiFetch(`/api/ledger/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (resp.ok) {
    settleDialog.close();
    showBanner(`Position #${id} settled as ${status}.`);
    loadLedger();
  } else {
    showBanner('Could not save settlement — verify sale price is between 1¢ and 99¢.', true);
  }
});

// --- Calibration & Engine Operations ---
function renderBacktestData(data) {
  const container = document.getElementById('backtest-container');
  container.innerHTML = Object.entries(data)
    .map(([lg, info]) => {
      const sim = info.simulation;
      const tiersHtml = ['A+', 'A', 'B']
        .map((tierKey) => {
          const t = sim.by_tier ? sim.by_tier[tierKey] : null;
          if (!t) return '';
          const wr = t.win_rate != null ? `${(t.win_rate * 100).toFixed(1)}%` : '—';
          const roiCls = t.roi_pct >= 0 ? 'positive' : 'negative';
          return `
            <div class="kv-row">
              <span>Tier ${tierKey} (${t.bets} trades)</span>
              <strong class="${roiCls}">${wr} win · ${t.roi_pct >= 0 ? '+' : ''}${(t.roi_pct * 100).toFixed(1)}% ROI</strong>
            </div>
          `;
        })
        .join('');

      const binsHtml = (info.calibration_bins || [])
        .map((b) => {
          const predPct = (b.predicted_prob * 100).toFixed(1);
          const actPct = (b.actual_win_rate * 100).toFixed(1);
          return `
            <div class="cal-bin-row">
              <span class="mono">${escapeHtml(b.bin_label)}</span>
              <div class="cal-bar-track" title="Predicted ${predPct}% vs Actual ${actPct}% (${b.count} games)">
                <div class="cal-bar-actual" style="width:${Math.min(100, Math.max(2, b.actual_win_rate * 100))}%"></div>
              </div>
              <span class="mono" style="text-align:right;color:var(--text-secondary);">${predPct}% → <strong>${actPct}%</strong></span>
            </div>
          `;
        })
        .join('');

      return `
        <div class="surface-panel">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <h3 class="panel-heading">${escapeHtml(lg.toUpperCase())} Walk-Forward</h3>
            <span class="mono" style="font-size:0.75rem;color:var(--text-muted);">${info.oos_games} OOS games</span>
          </div>

          <div class="cal-metrics">
            <div class="stat-col"><span>Brier Score</span><strong>${info.brier_score.toFixed(4)}</strong></div>
            <div class="stat-col"><span>Log-Loss</span><strong>${info.log_loss.toFixed(4)}</strong></div>
            <div class="stat-col"><span>Qualifying Trades</span><strong>${sim.total_bets}</strong></div>
            <div class="stat-col"><span>Actual Hit Rate</span><strong class="positive">${(sim.hit_rate * 100).toFixed(1)}%</strong></div>
            <div class="stat-col"><span>Post-Fee ROI</span><strong class="${sim.roi_pct >= 0 ? 'positive' : 'negative'}">${sim.roi_pct >= 0 ? '+' : ''}${(sim.roi_pct * 100).toFixed(2)}%</strong></div>
            <div class="stat-col"><span>Max Drawdown</span><strong>${(sim.max_drawdown_pct * 100).toFixed(1)}%</strong></div>
          </div>

          <h4 class="sub-heading">Conviction Tier Breakdown</h4>
          <div class="kv-list">${tiersHtml}</div>

          <h4 class="sub-heading">Calibration Reliability (Pred → Actual)</h4>
          <div>${binsHtml}</div>
        </div>
      `;
    })
    .join('');
}

async function loadBacktest() {
  const container = document.getElementById('backtest-container');
  container.innerHTML = '<div class="empty-state"><p>Loading walk-forward calibration metrics...</p></div>';
  try {
    const resp = await apiFetch('/api/backtest-metrics');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.backtestLoaded = true;
    renderBacktestData(data);
  } catch (err) {
    if (err.message !== 'Authentication required') {
      container.innerHTML = `<div class="empty-state"><p>Error loading backtest: ${escapeHtml(String(err))}</p></div>`;
    }
  }
}

document.getElementById('run-backtest-btn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = 'Running Backtest...';
  try {
    const resp = await apiFetch(`/api/backtest?sport=all&fee=${getFee()}`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.backtestLoaded = true;
    renderBacktestData(data);
    showBanner('Chronological walk-forward backtest completed.');
  } catch (err) {
    showBanner(`Backtest failed: ${err}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Re-run Backtest';
  }
});

document.getElementById('train-models-btn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = 'Training...';
  try {
    const resp = await apiFetch('/api/train?sport=all', { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    await loadPicks();
    showBanner('NFL, NBA, and Player Props ensemble models retrained and calibrated.');
  } catch (err) {
    showBanner(`Model training failed: ${err}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Retrain Models';
  }
});

document.getElementById('sync-btn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  btn.textContent = 'Syncing & Training...';
  try {
    const resp = await apiFetch('/api/sync?sport=all', { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    await loadServerConfig();
    await loadPicks();
    showBanner('Historical schedules, player logs, live feeds, and models synced.');
  } catch (err) {
    showBanner(`Sync error: ${err}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sync Data & Retrain';
  }
});

// --- Guide Markdown Renderer ---
async function loadGuide() {
  try {
    const resp = await apiFetch('/api/guide');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.guideLoaded = true;
    const md = data.markdown || '';
    const blocks = md.split(/```([\s\S]*?)```/g);
    let html = '';
    for (let i = 0; i < blocks.length; i++) {
      if (i % 2 === 1) {
        const code = blocks[i].replace(/^[a-zA-Z0-9_-]+\n/, '');
        html += `<pre><code>${escapeHtml(code)}</code></pre>`;
      } else {
        const lines = blocks[i].split('\n');
        let inList = false;
        for (const ln of lines) {
          const safe = escapeHtml(ln)
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
          const trimmed = ln.trim();
          if (trimmed.startsWith('- ') || /^\d+\.\s/.test(trimmed)) {
            if (!inList) {
              html += '<ul>';
              inList = true;
            }
            const itemText = safe.trim().replace(/^(-|\d+\.)\s+/, '');
            html += `<li>${itemText}</li>`;
          } else {
            if (inList) {
              html += '</ul>';
              inList = false;
            }
            if (ln.startsWith('# ')) html += `<h1>${safe.slice(2)}</h1>`;
            else if (ln.startsWith('## ')) html += `<h2>${safe.slice(3)}</h2>`;
            else if (ln.startsWith('### ')) html += `<h3>${safe.slice(4)}</h3>`;
            else if (trimmed !== '' && trimmed !== '---') html += `<p>${safe}</p>`;
          }
        }
        if (inList) html += '</ul>';
      }
    }
    document.getElementById('guide-content').innerHTML = html;
  } catch (err) {
    if (err.message !== 'Authentication required') {
      document.getElementById('guide-content').innerHTML = `<p>Failed loading guide: ${escapeHtml(String(err))}</p>`;
    }
  }
}

// --- Fast Live Feed Refresh ---
async function refreshLiveLines(silent = false) {
  const btn = document.getElementById('refresh-live-btn');
  const icon = document.getElementById('refresh-icon');
  const label = document.getElementById('refresh-btn-label');
  const dateVal = document.getElementById('filter-date').value;
  const qs = new URLSearchParams({ sport: state.sport });
  if (dateVal) qs.set('date', dateVal);

  btn.disabled = true;
  icon.classList.add('spin');
  label.textContent = 'Refreshing';

  try {
    const resp = await apiFetch(`/api/refresh-live?${qs.toString()}`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    await loadServerConfig();
    await loadPicks();
    if (!document.getElementById('panel-ledger').hidden) await loadLedger();
    if (!silent) {
      showBanner('Live scoreboards, box scores, injuries, and lines refreshed.');
    }
  } catch (err) {
    if (!silent && err.message !== 'Authentication required') {
      showBanner(`Live refresh error: ${err}`, true);
    }
  } finally {
    btn.disabled = false;
    icon.classList.remove('spin');
    label.textContent = 'Refresh';
  }
}

document.getElementById('refresh-live-btn').addEventListener('click', () => refreshLiveLines(false));

// --- Boot Lifecycle ---
async function bootWorkspace() {
  await loadServerConfig();
  await loadPicks();
  runCustomCalculator();
}

async function initApp() {
  try {
    const authRes = await fetch('/api/auth/status');
    if (authRes.ok) {
      const auth = await authRes.json();
      state.authEnabled = Boolean(auth.auth_enabled);
      state.authenticated = Boolean(auth.authenticated);
      document.getElementById('logout-btn').hidden = !state.authEnabled;
      if (state.authEnabled && !state.authenticated) {
        promptPasscodeUnlock();
        return;
      }
    }
  } catch (_) {}
  await bootWorkspace();
}

initApp();

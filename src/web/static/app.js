// DKeX & Traditional DK/FD — Minimalist UI-First Client

const state = {
  traditionalMode: false,
  mode: 'all',
  sport: 'all',
  market: 'all',
  book: 'all',
  strategyTab: 'all',
  sbCalcMode: 'ev',
  viewMode: localStorage.getItem('dkex_view_mode') === 'table' ? 'table' : 'grid',
  ledgerFilter: 'ALL',
  ledgerVenueFilter: 'ALL',
  searchQuery: '',
  picks: [],
  sbSlate: null,
  totalCandidates: 0,
  gamesCount: 0,
  trades: [],
  lastLedgerSummary: null,
  lastCalcEval: null,
  backtestLoaded: false,
  backtestModeLoaded: null,
  guideLoaded: false,
  gridVisibleForPolling: true,
  authEnabled: false,
  authenticated: true,
  serverConfig: {},
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

// Build a readable Error from a non-OK response (FastAPI `detail` string / 422 list / plain text).
async function httpError(resp) {
  let detail = '';
  try {
    const ct = resp.headers.get('content-type') || '';
    if (ct.includes('application/json')) {
      const body = await resp.json();
      const d = body && body.detail;
      if (typeof d === 'string') {
        detail = d;
      } else if (Array.isArray(d)) {
        detail = d
          .map((x) => {
            if (!x || !x.msg) return String(x);
            const field = Array.isArray(x.loc) ? x.loc[x.loc.length - 1] : '';
            return field ? `${field}: ${x.msg}` : x.msg;
          })
          .join('; ');
      }
    } else {
      detail = (await resp.text()).trim().slice(0, 200);
    }
  } catch (_) {}
  return new Error(detail ? `HTTP ${resp.status}: ${detail}` : `HTTP ${resp.status}`);
}

// Fetch JSON or throw a descriptive error.
async function apiJson(url, options = {}) {
  const resp = await apiFetch(url, options);
  if (!resp.ok) throw await httpError(resp);
  return resp.json();
}

function isSilentError(err) {
  return Boolean(err) && (err.name === 'AbortError' || err.message === 'Authentication required');
}

function errText(err) {
  return err && err.message ? err.message : String(err);
}

function reportError(prefix, err) {
  if (isSilentError(err)) return;
  console.error(prefix, err);
  showBanner(`${prefix}: ${errText(err)}`, true);
}

// Disable a button while an async action runs to prevent double-submits.
async function withBusy(btn, fn) {
  if (btn && btn.disabled) return undefined;
  if (btn) {
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
  }
  try {
    return await fn();
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.removeAttribute('aria-busy');
    }
  }
}

// Monotonic per-key request sequencing so stale responses never overwrite newer ones.
const requestSeq = Object.create(null);
function nextSeq(key) {
  requestSeq[key] = (requestSeq[key] || 0) + 1;
  return requestSeq[key];
}
function isLatestSeq(key, seq) {
  return requestSeq[key] === seq;
}

function setSegActive(buttons, activeBtn) {
  buttons.forEach((b) => {
    const on = b === activeBtn;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
}

function promptPasscodeUnlock() {
  const errEl = document.getElementById('auth-error-msg');
  errEl.hidden = true;
  errEl.textContent = 'Invalid passcode.';
  document.getElementById('auth-passcode-input').value = '';
  if (!authDialog.open) {
    authDialog.showModal();
  }
}

authDialog.addEventListener('cancel', (e) => {
  if (state.authEnabled && !state.authenticated) {
    e.preventDefault();
  }
});

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
      errEl.textContent = resp.status === 401 ? 'Invalid passcode.' : errText(await httpError(resp));
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

document.getElementById('logout-btn').addEventListener('click', async (e) => {
  await withBusy(e.currentTarget, async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
    } catch (err) {
      console.error('Logout request failed:', err);
    }
    state.authenticated = false;
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    promptPasscodeUnlock();
  });
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
  if (selected.id === 'tab-backtest' && (!state.backtestLoaded || state.backtestModeLoaded !== state.traditionalMode)) {
    loadBacktest();
  }
  if (selected.id === 'tab-guide' && !state.guideLoaded) loadGuide();
  if (selected.id === 'tab-calculator') {
    if (state.traditionalMode) runSportsbookAnalyzer();
    else runCustomCalculator();
  }
  if (selected.id === 'tab-strategies' && state.traditionalMode) {
    renderStrategies();
  }
}

tabs.forEach((tab, idx) => {
  tab.tabIndex = idx === 0 ? 0 : -1;
  tab.addEventListener('click', () => selectTab(tab));
});

tablist.addEventListener('keydown', (e) => {
  const visibleTabs = tabs.filter((t) => !t.hidden);
  const cur = visibleTabs.indexOf(document.activeElement);
  if (cur === -1) return;
  let next = -1;
  if (e.key === 'ArrowRight') next = (cur + 1) % visibleTabs.length;
  if (e.key === 'ArrowLeft') next = (cur - 1 + visibleTabs.length) % visibleTabs.length;
  if (e.key === 'Home') next = 0;
  if (e.key === 'End') next = visibleTabs.length - 1;
  if (next !== -1) {
    e.preventDefault();
    visibleTabs[next].focus();
    selectTab(visibleTabs[next]);
  }
});

// --- Toast / Status Banner ---
function showBanner(msg, isError = false) {
  const wrap = document.getElementById('status-banner');
  const txt = document.getElementById('status-banner-text');
  txt.textContent = msg;
  wrap.classList.toggle('error', isError);
  wrap.setAttribute('role', isError ? 'alert' : 'status');
  wrap.hidden = false;
  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => {
    wrap.hidden = true;
  }, isError ? 9000 : 5000);
}

document.getElementById('dismiss-banner-btn').addEventListener('click', () => {
  if (bannerTimer) clearTimeout(bannerTimer);
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

const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function escapeHtml(str) {
  return String(str ?? '').replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch]);
}

// Null/NaN-safe number formatting (avoids "NaN" / TypeError on missing API fields).
function toFixedSafe(val, digits = 2, fallback = '—') {
  if (val === null || val === undefined || val === '') return fallback;
  const n = Number(val);
  return Number.isFinite(n) ? n.toFixed(digits) : fallback;
}

function pctSafe(val, digits = 1, fallback = '—') {
  if (val === null || val === undefined || val === '') return fallback;
  return toFixedSafe(Number(val) * 100, digits, fallback);
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

function formatAmerican(odds) {
  if (odds === null || odds === undefined || Number.isNaN(Number(odds))) return '—';
  const n = Math.round(Number(odds));
  return n > 0 ? `+${n}` : `${n}`;
}

function venueBadgeHtml(venue) {
  const v = String(venue || 'DKEX').toUpperCase();
  if (v === 'DRAFTKINGS' || v === 'DK') return '<span class="venue-pill venue-dk">DK</span>';
  if (v === 'FANDUEL' || v === 'FD') return '<span class="venue-pill venue-fd">FD</span>';
  return '<span class="venue-pill venue-dkex">DKeX</span>';
}

function venueDisplayName(venue) {
  const v = String(venue || 'DKEX').toUpperCase();
  if (v === 'DRAFTKINGS' || v === 'DK') return 'DraftKings';
  if (v === 'FANDUEL' || v === 'FD') return 'FanDuel';
  return 'DKeX';
}

function updateHeaderChip() {
  const bk = getBankroll();
  if (state.traditionalMode) {
    document.getElementById('header-bankroll-chip').textContent = `${formatCompactDollars(bk)} · DK/FD`;
  } else {
    const feeCents = (getFee() * 100).toFixed(1).replace(/\.0$/, '');
    document.getElementById('header-bankroll-chip').textContent = `${formatCompactDollars(bk)} · ${feeCents}¢`;
  }
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
    txt.textContent = cachedCount > 0 ? `DK+FD Sharp (${cachedCount})${rem}` : `DK+FD API${rem}`;
    pill.classList.add('connected');
  } else {
    txt.textContent = state.traditionalMode ? 'DK Feed (No FD Key)' : 'ESPN / DK Feed';
    pill.classList.remove('connected');
  }
}

// --- Exclusive Mode Switching (DKeX vs Traditional DK/FD) ---
function syncSettingsDialogModeFields() {
  const isTrad = document.getElementById('setting-traditional-mode').checked;
  document.getElementById('setting-dkex-fee-wrap').hidden = isTrad;
  document.getElementById('setting-arb-round-wrap').hidden = !isTrad;
  document.getElementById('setting-sb-extras').hidden = !isTrad;
}

function applyWorkspaceMode() {
  const isTrad = state.traditionalMode;

  document.getElementById('brand-mark').textContent = isTrad ? 'DK · FD' : 'DKeX';
  document.getElementById('brand-sub').textContent = isTrad
    ? 'DraftKings & FanDuel Sportsbook Engine'
    : 'DraftKings Predictions Model';
  document.title = isTrad
    ? 'DK · FD — Traditional Sportsbooks & Arbitrage Engine'
    : 'DKeX — DraftKings Predictions Model';

  const stratTab = document.getElementById('tab-strategies');
  stratTab.hidden = !isTrad;
  if (!isTrad && stratTab.getAttribute('aria-selected') === 'true') {
    selectTab(document.getElementById('tab-picks'));
  }

  document.getElementById('sb-book-filter').hidden = !isTrad;
  document.getElementById('kpi-book-split').hidden = !isTrad;
  document.getElementById('kpi-locks-label').textContent = isTrad ? 'Arbs & +EV Plays' : 'Conviction Plays';
  document.getElementById('filter-edge-label').textContent = isTrad ? 'Min EV %' : 'Min Net Edge';
  document.getElementById('filter-show-all-label').textContent = isTrad ? 'Show PASS lines' : 'Show PASS contracts';

  // Analyzer exclusive swap
  document.getElementById('analyzer-dkex').hidden = isTrad;
  document.getElementById('analyzer-sportsbook').hidden = !isTrad;

  // Ledger multi-venue controls
  document.getElementById('ledger-venue-filters').hidden = !isTrad;
  document.getElementById('ledger-venue-strip').hidden = !isTrad;
  document.getElementById('ledger-table-caption').textContent = isTrad
    ? 'Unified Sportsbook & Position Ledger'
    : 'Position Ledger & Cash-Out Monitor';

  // Calibration headers
  document.getElementById('backtest-heading').textContent = isTrad
    ? 'Traditional DK/FD Strategy Verification & Walk-Forward Backtest'
    : 'Walk-Forward Calibration & Model Engine';
  document.getElementById('backtest-sub').textContent = isTrad
    ? 'Zero-lookahead out-of-sample profitability audit across +EV Line Shopping, Arbitrage, Middles, Bonus Bets, Profit Boosts, and Steam Lag.'
    : 'Strict chronological out-of-sample evaluation with zero lookahead bias and itemized contract fee accounting.';
  document.getElementById('run-backtest-btn').textContent = isTrad
    ? 'Re-run Sportsbook Backtest'
    : 'Re-run Backtest';

  // Table headers for Markets
  const thead = document.getElementById('picks-table-thead');
  if (isTrad) {
    thead.innerHTML = `
      <tr>
        <th scope="col">Tier</th>
        <th scope="col">Matchup / State</th>
        <th scope="col">Selection</th>
        <th scope="col">Best Book</th>
        <th scope="col" class="num-col">DK / FD Odds</th>
        <th scope="col" class="num-col">Fair</th>
        <th scope="col" class="num-col">Win %</th>
        <th scope="col" class="num-col">EV %</th>
        <th scope="col" class="num-col">Kelly $</th>
        <th scope="col">Action</th>
      </tr>`;
  } else {
    thead.innerHTML = `
      <tr>
        <th scope="col">Tier</th>
        <th scope="col">Matchup / State</th>
        <th scope="col">Contract</th>
        <th scope="col">Side</th>
        <th scope="col" class="num-col">Ask ¢</th>
        <th scope="col" class="num-col">Max Buy</th>
        <th scope="col" class="num-col">Win %</th>
        <th scope="col" class="num-col">Net Edge</th>
        <th scope="col" class="num-col">Kelly Size</th>
        <th scope="col">Action</th>
      </tr>`;
  }

  // New trade modal adaptation
  document.getElementById('nt-venue-wrap').hidden = !isTrad;
  document.getElementById('nt-strategy-wrap').hidden = !isTrad;
  document.getElementById('nt-dkex-fields').hidden = isTrad;
  document.getElementById('nt-sb-fields').hidden = !isTrad;
  document.getElementById('new-trade-dialog-title').textContent = isTrad
    ? 'Log Sportsbook Bet'
    : 'Log DKeX Position';

  updateHeaderChip();
}

function updateKPIs(ledgerSummary, picksList) {
  if (ledgerSummary) {
    state.lastLedgerSummary = ledgerSummary;
    document.getElementById('kpi-bankroll').textContent = `$${formatMoney(ledgerSummary.current_bankroll)}`;

    const splitEl = document.getElementById('kpi-book-split');
    if (state.traditionalMode && ledgerSummary.by_venue) {
      const dkB = ledgerSummary.by_venue.DRAFTKINGS?.current_bankroll ?? 0;
      const fdB = ledgerSummary.by_venue.FANDUEL?.current_bankroll ?? 0;
      splitEl.textContent = `DK ${formatCompactDollars(dkB)} · FD ${formatCompactDollars(fdB)}`;
      splitEl.hidden = false;
    } else {
      splitEl.hidden = true;
    }

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
      ledgerSummary.hit_rate != null ? `${pctSafe(ledgerSummary.hit_rate, 1)}%` : '—';
    document.getElementById('kpi-exp-hitrate').textContent =
      ledgerSummary.expected_win_rate != null
        ? `exp ${pctSafe(ledgerSummary.expected_win_rate, 0)}%`
        : '';

    const openBadge = document.getElementById('open-trades-badge');
    const openCnt = Number(ledgerSummary.open_trades_count) || 0;
    openBadge.textContent = String(openCnt);
    openBadge.hidden = openCnt === 0;

    const subMeta = document.getElementById('ledger-sub-meta');
    if (ledgerSummary.avg_clv_cents !== null && ledgerSummary.avg_clv_cents !== undefined) {
      const clvSign = ledgerSummary.avg_clv_cents >= 0 ? '+' : '';
      const unit = state.traditionalMode ? '%' : '¢';
      subMeta.textContent = `Avg Closing Line Value: ${clvSign}${ledgerSummary.avg_clv_cents.toFixed(1)}${unit} · ${ledgerSummary.settled_trades_count} settled, ${openCnt} open.`;
    }
  }

  if (picksList) {
    const qualifying = picksList.filter((p) => p.confidence_tier !== 'PASS').length;
    const liveCnt = picksList.filter((p) => p.game_state === 'LIVE' && p.confidence_tier !== 'PASS').length;
    if (state.traditionalMode && state.sbSlate) {
      const sc = state.sbSlate.strategy_counts || {};
      const arbAndSteam = (sc.arbitrage || 0) + (sc.middles_ev || 0) + (sc.steam || 0);
      document.getElementById('kpi-locks').textContent = String(qualifying + (sc.arbitrage || 0));
      document.getElementById('kpi-total-picks').textContent =
        `${sc.arbitrage || 0} arb · ${qualifying} +EV · ${sc.middles_ev || 0} mid`;

      const stratBadge = document.getElementById('strategies-badge');
      const totalStrat = arbAndSteam + (sc.bonus_bet_70plus || 0);
      stratBadge.textContent = String(totalStrat);
      stratBadge.hidden = totalStrat === 0;
    } else {
      const locks = picksList.filter((p) => p.confidence_tier === 'A+' || p.confidence_tier === 'A').length;
      document.getElementById('kpi-locks').textContent = String(locks);
      document.getElementById('kpi-total-picks').textContent =
        liveCnt > 0 ? `${qualifying} active · ${liveCnt} live` : `${qualifying} qualifying`;
    }

    const livePill = document.getElementById('live-pill-count');
    const allLive = picksList.filter((p) => p.game_state === 'LIVE').length;
    livePill.textContent = String(allLive);
    livePill.hidden = allLive === 0;
  }
}

// --- Auto-Refresh Live Polling ---
const AUTO_REFRESH_MS = 60000;
let lastLiveRefreshAt = 0;
const picksGridEl = document.getElementById('picks-grid');
picksGridEl.addEventListener('contentvisibilityautostatechange', (event) => {
  state.gridVisibleForPolling = !event.skipped;
  syncAutoRefreshTimer();
});

document.addEventListener('visibilitychange', () => {
  const wasRunning = Boolean(autoRefreshTimer);
  syncAutoRefreshTimer();
  // Catch up immediately when returning to a tab whose data went stale while hidden.
  if (!wasRunning && autoRefreshTimer && Date.now() - lastLiveRefreshAt >= AUTO_REFRESH_MS) {
    refreshLiveLines(true);
  }
});

function shouldAutoRefresh() {
  if (!document.getElementById('auto-refresh-live').checked) return false;
  if (document.hidden || !state.authenticated) return false;
  const picksPanelVisible = !document.getElementById('panel-picks').hidden;
  const stratPanelVisible = !document.getElementById('panel-strategies').hidden;
  if (!picksPanelVisible && !stratPanelVisible) return false;
  // Only gate on card-grid on-screen visibility when the card grid is actually the active view.
  if (picksPanelVisible && state.viewMode === 'grid' && !state.gridVisibleForPolling) return false;
  return true;
}

function syncAutoRefreshTimer() {
  const run = shouldAutoRefresh();
  if (run && autoRefreshTimer) return; // keep existing cadence instead of resetting it
  if (!run) {
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
    return;
  }
  autoRefreshTimer = setInterval(() => {
    if (!document.hidden) refreshLiveLines(true);
  }, AUTO_REFRESH_MS);
}

// --- Server Config & Settings Dialog ---
let filtersSyncedForMode = null;

async function loadServerConfig() {
  try {
    const res = await apiFetch('/api/config');
    if (!res.ok) throw await httpError(res);
    const cfg = await res.json();
    state.serverConfig = cfg;
    const prevMode = state.traditionalMode;
    state.traditionalMode = Boolean(cfg.traditional_sportsbooks_enabled);
    const incDkex = Boolean(cfg.include_dkex_cross_arb ?? true);

    // Don't clobber edits the user is making in an open Settings dialog (e.g. during auto-refresh).
    if (!settingsDialog.open) {
      document.getElementById('setting-traditional-mode').checked = state.traditionalMode;
      document.getElementById('global-bankroll').value = toFixedSafe(cfg.bankroll, 2, '1000.00');

      const feeSelect = document.getElementById('global-fee');
      const matchingOpt = Array.from(feeSelect.options).find(
        (o) => Math.abs(parseFloat(o.value) - Number(cfg.default_fee_dollars)) < 1e-4
      );
      if (matchingOpt) feeSelect.value = matchingOpt.value;

      const roundSelect = document.getElementById('setting-arb-rounding');
      const roundOpt = Array.from(roundSelect.options).find(
        (o) => Math.abs(parseFloat(o.value) - Number(cfg.arb_stake_rounding ?? 1.0)) < 1e-4
      );
      if (roundOpt) roundSelect.value = roundOpt.value;

      document.getElementById('setting-dk-bankroll').value =
        cfg.dk_bankroll != null ? Number(cfg.dk_bankroll).toFixed(2) : '';
      document.getElementById('setting-fd-bankroll').value =
        cfg.fd_bankroll != null ? Number(cfg.fd_bankroll).toFixed(2) : '';
      document.getElementById('setting-min-arb-roi').value = (Number(cfg.min_arb_roi ?? 0.005) * 100).toFixed(1);
      document.getElementById('setting-include-dkex-arb').checked = incDkex;

      document.getElementById('setting-kelly').value = Number(cfg.kelly_fraction || 0.25).toFixed(2);
      document.getElementById('setting-max-bet').value = (Number(cfg.max_single_bet_pct || 0.04) * 100).toFixed(1);
    }

    // Only (re)seed the slate filters from server defaults on first load or after a mode switch,
    // so periodic live refreshes don't reset the user's slider / toggle choices.
    if (filtersSyncedForMode !== state.traditionalMode) {
      filtersSyncedForMode = state.traditionalMode;
      document.getElementById('strat-include-dkex').checked = incDkex;

      const minProbPct = state.traditionalMode
        ? 40
        : Math.round(Number(cfg.min_win_prob ?? 0.6) * 100);
      document.getElementById('filter-min-prob').value = String(minProbPct);
      document.getElementById('min-prob-val').textContent = `${minProbPct}%`;

      const minEdgePct = state.traditionalMode
        ? Number((Number(cfg.min_sb_ev_edge ?? 0.02) * 100).toFixed(1))
        : Number((Number(cfg.min_net_edge ?? 0.025) * 100).toFixed(1));
      document.getElementById('filter-min-edge').value = String(minEdgePct);
      document.getElementById('min-edge-val').textContent = `+${minEdgePct.toFixed(1)}%`;
    }

    const keyHint = document.getElementById('setting-odds-hint');
    if (cfg.odds_api_connected) {
      keyHint.textContent = `Connected (${cfg.odds_api_key_masked || 'active'}). Streaming live DraftKings, FanDuel & sharp consensus odds.`;
    } else {
      keyHint.textContent = 'Zero-key mode uses public ESPN DraftKings feeds. Add a free key to enable live FanDuel odds & automated DK ↔ FD arbitrage.';
    }

    document.getElementById('logout-btn').hidden = !cfg.auth_enabled;

    syncSettingsDialogModeFields();
    applyWorkspaceMode();
    updateOddsApiPill(
      Boolean(cfg.odds_api_connected),
      Number(cfg.sharp_events_cached || 0),
      cfg.odds_quota || null
    );

    if (prevMode !== state.traditionalMode) {
      state.backtestLoaded = false;
      state.picks = [];
      state.sbSlate = null;
    }
  } catch (err) {
    if (!isSilentError(err)) {
      console.error('Failed loading /api/config:', err);
    }
  }
}

document.getElementById('setting-traditional-mode').addEventListener('change', syncSettingsDialogModeFields);

let settingsSaved = false;
function openSettingsDialog() {
  if (settingsDialog.open) return;
  settingsSaved = false;
  document.getElementById('setting-odds-key').value = '';
  syncSettingsDialogModeFields();
  settingsDialog.showModal();
}

// Cancel / Escape / light-dismiss: revert unsaved edits, since the bankroll & fee inputs in this
// dialog are read live by getBankroll()/getFee() for every sizing request.
settingsDialog.addEventListener('close', () => {
  if (!settingsSaved && state.authenticated) loadServerConfig();
});

['open-settings-btn', 'notice-open-settings-btn', 'strat-open-settings-btn'].forEach((id) => {
  const btn = document.getElementById(id);
  if (btn) btn.addEventListener('click', openSettingsDialog);
});

document.getElementById('settings-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const keyRaw = document.getElementById('setting-odds-key').value.trim();
  const isTrad = document.getElementById('setting-traditional-mode').checked;
  const dkBkRaw = document.getElementById('setting-dk-bankroll').value.trim();
  const fdBkRaw = document.getElementById('setting-fd-bankroll').value.trim();

  const payload = {
    traditional_sportsbooks_enabled: isTrad,
    bankroll: getBankroll(),
    default_fee_dollars: getFee(),
    kelly_fraction: Math.min(1.0, Math.max(0.05, parseFloat(document.getElementById('setting-kelly').value) || 0.25)),
    max_single_bet_pct: Math.min(0.25, Math.max(0.01, (parseFloat(document.getElementById('setting-max-bet').value) || 4) / 100)),
    arb_stake_rounding: parseFloat(document.getElementById('setting-arb-rounding').value) || 0.0,
    min_arb_roi: Math.min(0.2, Math.max(0, (parseFloat(document.getElementById('setting-min-arb-roi').value) || 0.5) / 100)),
    include_dkex_cross_arb: document.getElementById('setting-include-dkex-arb').checked,
    dk_bankroll: dkBkRaw === '' ? 0 : Math.max(0, parseFloat(dkBkRaw) || 0),
    fd_bankroll: fdBkRaw === '' ? 0 : Math.max(0, parseFloat(fdBkRaw) || 0),
  };
  if (isTrad) {
    payload.min_sb_ev_edge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  } else {
    payload.min_win_prob = Math.max(0.50, parseFloat(document.getElementById('filter-min-prob').value) / 100);
    payload.min_net_edge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  }
  if (keyRaw.toUpperCase() === 'CLEAR') {
    payload.the_odds_api_key = '';
  } else if (keyRaw !== '') {
    payload.the_odds_api_key = keyRaw;
  }

  await withBusy(document.getElementById('save-settings-btn'), async () => {
    try {
      await apiJson('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      settingsSaved = true;
      settingsDialog.close();
      await loadServerConfig();
      await loadPicks();
      if (!document.getElementById('panel-calculator').hidden) {
        if (state.traditionalMode) runSportsbookAnalyzer();
        else runCustomCalculator();
      }
      if (!document.getElementById('panel-ledger').hidden) loadLedger();
      if (!document.getElementById('panel-backtest').hidden) loadBacktest();
      showBanner(
        state.traditionalMode
          ? 'Switched to Traditional DraftKings & FanDuel Sportsbooks mode.'
          : 'Workspace settings saved (DKeX mode).'
      );
    } catch (err) {
      reportError('Failed saving settings', err);
    }
  });
});

// --- Markets Toolbar & Filtering ---
document.querySelectorAll('[data-seg]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.seg;
    const val = btn.dataset.val;
    if (state[group] === val && btn.classList.contains('active')) return;
    state[group] = val;
    setSegActive(document.querySelectorAll(`[data-seg="${group}"]`), btn);
    loadPicks();
  });
});

document.querySelectorAll('[data-strat-tab]').forEach((btn) => {
  btn.addEventListener('click', () => {
    state.strategyTab = btn.dataset.stratTab;
    setSegActive(document.querySelectorAll('[data-strat-tab]'), btn);
    renderStrategies();
  });
});

['strat-arb-stake', 'strat-bonus-size', 'strat-boost-pct', 'strat-include-dkex'].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', () => { if (state.traditionalMode) loadPicks(); });
});

document.getElementById('toggle-filters-btn').addEventListener('click', (e) => {
  const form = document.getElementById('filters-form');
  const nextHidden = !form.hidden;
  form.hidden = nextHidden;
  e.currentTarget.setAttribute('aria-expanded', String(!nextHidden));
});

function applyViewModeUI() {
  const isTable = state.viewMode === 'table';
  const gridBtn = document.getElementById('view-grid-btn');
  const tableBtn = document.getElementById('view-table-btn');
  setSegActive([gridBtn, tableBtn], isTable ? tableBtn : gridBtn);
  document.getElementById('picks-grid').hidden = isTable;
  document.getElementById('picks-table-wrap').hidden = !isTable;
}

function setViewMode(mode) {
  if (state.viewMode === mode) return;
  state.viewMode = mode;
  try { localStorage.setItem('dkex_view_mode', mode); } catch (_) {}
  applyViewModeUI();
  renderPicks();
  syncAutoRefreshTimer();
}

document.getElementById('view-grid-btn').addEventListener('click', () => setViewMode('grid'));
document.getElementById('view-table-btn').addEventListener('click', () => setViewMode('table'));
applyViewModeUI();

let searchDebounceTimer = null;
document.getElementById('filter-search').addEventListener('input', (e) => {
  const next = e.target.value.trim().toLowerCase();
  if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    if (next === state.searchQuery) return;
    state.searchQuery = next;
    renderPicks();
  }, 120);
});

document.getElementById('filters-form').addEventListener('submit', (e) => e.preventDefault());

document.getElementById('filter-min-prob').addEventListener('input', (e) => {
  document.getElementById('min-prob-val').textContent = `${e.target.value}%`;
});
document.getElementById('filter-min-edge').addEventListener('input', (e) => {
  document.getElementById('min-edge-val').textContent = `+${parseFloat(e.target.value).toFixed(1)}%`;
});

['filter-date', 'filter-min-prob', 'filter-min-edge', 'filter-show-all', 'filter-live-props'].forEach((id) => {
  document.getElementById(id).addEventListener('change', () => loadPicks());
});
document.getElementById('auto-refresh-live').addEventListener('change', syncAutoRefreshTimer);

document.getElementById('quick-toggle-pass-btn').addEventListener('click', () => {
  const cb = document.getElementById('filter-show-all');
  cb.checked = !cb.checked;
  loadPicks();
});

function formatEventStart(startTimeIso, statusDetail) {
  if (startTimeIso) {
    const dt = new Date(startTimeIso);
    if (!Number.isNaN(dt.getTime())) {
      const now = new Date();
      const startDay = new Date(dt.getFullYear(), dt.getMonth(), dt.getDate());
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const diffDays = Math.round((startDay - today) / 86400000);
      const timeStr = dt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
      if (diffDays === 0) return `Today ${timeStr}`;
      if (diffDays === 1) return `Tomorrow ${timeStr}`;
      const dateStr = dt.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
      return `${dateStr} · ${timeStr}`;
    }
  }
  return statusDetail || '';
}

function updateSlateMetaBar() {
  const bar = document.getElementById('slate-meta-bar');
  const txt = document.getElementById('slate-meta-text');
  const btn = document.getElementById('quick-toggle-pass-btn');
  const showAll = document.getElementById('filter-show-all').checked;

  if (!state.totalCandidates && !state.picks.length) {
    bar.hidden = true;
    return;
  }

  bar.hidden = false;
  const gamesWord = state.gamesCount === 1 ? 'game' : 'games';
  let propNote = '';
  if (state.traditionalMode && state.sbSlate && state.sbSlate.prop_diagnostics && state.market !== 'game') {
    const pd = state.sbSlate.prop_diagnostics;
    if (pd.last_error) {
      propNote = ` · Prop sync: ${pd.last_error}`;
    } else if (pd.events_queried > 0) {
      propNote = ` · Odds API props: ${pd.odds_api_events_with_fd}/${pd.events_queried} games w/ FD`;
    }
  }
  if (showAll) {
    const actionCount = state.picks.filter((p) => p.confidence_tier !== 'PASS').length;
    txt.textContent = `Showing all ${state.picks.length} priced lines (${actionCount} conviction) across ${state.gamesCount} upcoming ${gamesWord}${propNote}`;
    btn.textContent = 'Hide PASS lines';
    btn.hidden = false;
  } else {
    const pickWord = state.picks.length === 1 ? 'play' : 'plays';
    txt.textContent = `Showing ${state.picks.length} conviction ${pickWord} of ${state.totalCandidates} priced lines across ${state.gamesCount} upcoming ${gamesWord}${propNote}`;
    btn.textContent = `Show all ${state.totalCandidates} lines`;
    btn.hidden = state.totalCandidates <= state.picks.length;
  }
}

// --- Load & Render Markets ---
let picksAbortCtrl = null;

function setPicksBusy(busy) {
  ['picks-grid', 'picks-table-wrap', 'strategies-container'].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (busy) el.setAttribute('aria-busy', 'true');
    else el.removeAttribute('aria-busy');
  });
}

async function loadPicks() {
  const minProb = parseFloat(document.getElementById('filter-min-prob').value) / 100;
  const minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  const showAll = document.getElementById('filter-show-all').checked;
  const liveProps = document.getElementById('filter-live-props').checked;
  const dateVal = document.getElementById('filter-date').value;

  // Cancel any older in-flight slate request so a slow stale response can't overwrite newer filters.
  if (picksAbortCtrl) picksAbortCtrl.abort();
  const ctrl = new AbortController();
  picksAbortCtrl = ctrl;

  const grid = document.getElementById('picks-grid');
  const tbody = document.getElementById('picks-table-tbody');
  if (!state.picks.length) {
    const loadingMsg = state.traditionalMode
      ? 'Scanning DraftKings &amp; FanDuel lines, arbs, middles, promos, and steam lag...'
      : 'Pricing pre-game and live contracts...';
    grid.innerHTML = `<div class="empty-state"><p>${loadingMsg}</p></div>`;
    tbody.innerHTML = `<tr><td colspan="10" class="table-empty">${loadingMsg}</td></tr>`;
  }
  setPicksBusy(true);

  try {
    if (state.traditionalMode) {
      const arbStake = parseFloat(document.getElementById('strat-arb-stake')?.value || '100') || 100;
      const bonusSize = parseFloat(document.getElementById('strat-bonus-size')?.value || '50') || 50;
      const boostPct = parseFloat(document.getElementById('strat-boost-pct')?.value || '0.25') || 0.25;
      const includeDkex = Boolean(document.getElementById('strat-include-dkex')?.checked);
      const params = new URLSearchParams({
        mode: state.mode,
        sport: state.sport,
        market: state.market,
        book: state.book,
        min_prob: String(Math.max(0.20, minProb)),
        min_ev: String(minEdge),
        bankroll: String(getBankroll()),
        arb_stake: String(Math.max(1, arbStake)),
        bonus_bet_size: String(Math.max(1, bonusSize)),
        boost_pct: String(boostPct),
        include_passes: String(showAll),
        include_dkex_cross_arb: String(includeDkex),
        live_props: String(liveProps),
      });
      if (dateVal) params.set('date', dateVal);

      const data = await apiJson(`/api/sportsbook/slate?${params.toString()}`, { signal: ctrl.signal });
      if (ctrl !== picksAbortCtrl) return;
      state.sbSlate = data;
      state.picks = data.picks || [];
      state.totalCandidates = Number(data.total_candidates ?? state.picks.length);
      state.gamesCount = Number(data.games_count ?? 0);

      const showFdNotice = !data.has_fanduel_odds && !data.odds_api_connected;
      document.getElementById('sb-fanduel-notice').hidden = !showFdNotice;
      document.getElementById('strat-fanduel-notice').hidden = !showFdNotice;

      if (data.odds_quota) {
        updateOddsApiPill(Boolean(data.odds_api_connected), Number(state.serverConfig.sharp_events_cached || 0), data.odds_quota);
      }
      updateKPIs(data.ledger_summary, state.picks);
      updateSlateMetaBar();
      renderPicks();
      renderStrategies();
      return;
    }

    document.getElementById('sb-fanduel-notice').hidden = true;
    const params = new URLSearchParams({
      mode: state.mode,
      sport: state.sport,
      market: state.market,
      min_prob: String(Math.max(0.50, minProb)),
      min_edge: String(minEdge),
      fee: String(getFee()),
      bankroll: String(getBankroll()),
      include_passes: String(showAll),
      live_props: String(liveProps),
    });
    if (dateVal) params.set('date', dateVal);

    const data = await apiJson(`/api/picks?${params.toString()}`, { signal: ctrl.signal });
    if (ctrl !== picksAbortCtrl) return;
    state.sbSlate = null;
    state.picks = data.picks || [];
    state.totalCandidates = Number(data.total_candidates ?? state.picks.length);
    state.gamesCount = Number(data.games_count ?? 0);
    if (data.odds_quota) {
      updateOddsApiPill(Boolean(data.odds_api_connected), Number(state.serverConfig.sharp_events_cached || 0), data.odds_quota);
    }
    updateKPIs(data.ledger_summary, state.picks);
    updateSlateMetaBar();
    renderPicks();
  } catch (err) {
    if (ctrl !== picksAbortCtrl || isSilentError(err)) return;
    console.error('Failed loading markets:', err);
    const msg = escapeHtml(errText(err));
    grid.innerHTML = `<div class="empty-state" role="alert"><h3>Unable to load markets</h3><p>${msg}</p></div>`;
    tbody.innerHTML = `<tr><td colspan="10" class="table-empty negative">Unable to load markets: ${msg}</td></tr>`;
  } finally {
    if (ctrl === picksAbortCtrl) {
      picksAbortCtrl = null;
      setPicksBusy(false);
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

function formatSelectionLabelForSide(pick, side) {
  const sideU = String(side || 'YES').toUpperCase() === 'NO' ? 'NO' : 'YES';
  const parts = String(pick.short_name || '').split(' @ ').map((s) => s.trim());
  const away = parts[0] || '';
  const home = parts[1] || '';
  const mktType = String(pick.market_type || '');

  if (mktType.includes('Prop:') && pick.player_name && pick.line_value != null) {
    const propLabel = mktType.split('Prop:')[1].trim();
    const teamMatch = String(pick.selection_label || '').match(/\(([A-Z]{2,4})\)/);
    const teamTag = teamMatch ? ` (${teamMatch[1]})` : '';
    const ln = Number(pick.line_value).toFixed(1);
    return sideU === 'YES'
      ? `${pick.player_name}${teamTag} Over ${ln} ${propLabel}`
      : `${pick.player_name}${teamTag} Under ${ln} ${propLabel} (Over: NO)`;
  }

  if (mktType.endsWith('Spread') && away && home && pick.line_value != null) {
    const sp = Number(pick.line_value);
    const homeFmt = `${sp >= 0 ? '+' : ''}${sp.toFixed(1)}`;
    const awayFmt = `${-sp >= 0 ? '+' : ''}${(-sp).toFixed(1)}`;
    return sideU === 'YES'
      ? `${home} ${homeFmt} Cover`
      : `${away} ${awayFmt} Cover (${home} ${homeFmt}: NO)`;
  }

  if (mktType.endsWith('Total') && pick.line_value != null) {
    const tot = Number(pick.line_value).toFixed(1);
    return sideU === 'YES'
      ? `Over ${tot} Total Points`
      : `Under ${tot} Total Points (Over: NO)`;
  }

  if (mktType.endsWith('Moneyline') && away && home) {
    return sideU === 'YES' ? `${home} to Win` : `${away} to Win (${home}: NO)`;
  }

  return pick.selection_label;
}

function buildPickCardInnerHtml(p, idx) {
  const isLive = p.game_state === 'LIVE';
  const edgeCls = p.net_edge >= 0 ? 'positive' : 'negative';
  const edgeSign = p.net_edge >= 0 ? '+' : '';

  const liveBadge = isLive
    ? `<span class="live-indicator"><span class="live-dot"></span>LIVE ${escapeHtml(p.status_detail || '')}${p.score_display ? ` · ${escapeHtml(p.score_display)}` : ''}</span>`
    : '';
  const startLabel = !isLive ? formatEventStart(p.start_time, p.status_detail) : '';

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

  const canEditLine = Boolean(p.override_key) && p.line_value !== null && p.line_value !== undefined;
  const defLineVal = p.default_line_value != null ? Number(p.default_line_value) : Number(p.line_value);
  const lineTooltip = p.is_custom_line
    ? `Custom DK line (${Number(p.line_value)}) — synced default: ${defLineVal}`
    : `Synced line (${defLineVal}) — edit if DraftKings offers a different line`;

  const lineEditorHtml = canEditLine
    ? `<div class="dk-line-wrap" title="${escapeHtml(lineTooltip)}">
        <div class="dk-line-field">
          <span class="line-prefix">Ln</span>
          <input
            id="dk-ln-${idx}"
            class="dk-line-input ${p.is_custom_line ? 'is-custom' : ''}"
            type="number"
            step="0.5"
            value="${Number.isFinite(Number(p.line_value)) ? Number(p.line_value) : ''}"
            data-idx="${idx}"
            aria-label="Offered DraftKings line for ${escapeHtml(p.selection_label)}"
          >
        </div>
        ${
          p.is_custom_line
            ? `<button type="button" class="line-reset-btn" data-reset-idx="${idx}" title="Reset to synced line (${escapeHtml(String(defLineVal))})" aria-label="Reset custom line">↺</button>`
            : ''
        }
      </div>`
    : '';

  const lineSrcText =
    p.is_custom_line && p.default_line_value != null
      ? `${p.line_source || 'Custom Line'} (sync ${Number(p.default_line_value)})`
      : p.line_source || 'ESPN/DK';

  const qty = Math.max(1, p.recommended_contracts || 1);
  const btnLabel =
    p.confidence_tier === 'PASS'
      ? `Log (${qty}x)`
      : `Log ${escapeHtml(p.recommended_side)} · ${p.recommended_contracts ?? qty}x ($${toFixedSafe(p.recommended_outlay, 2)})`;
  const btnCls = p.confidence_tier === 'PASS' ? 'btn btn-secondary btn-sm log-trade-btn' : 'btn btn-primary btn-sm log-trade-btn';

  return `
    <div class="pick-header">
      <div class="pick-meta">
        <span class="sport-label">${escapeHtml(p.sport)}</span>
        <span aria-hidden="true">·</span>
        <span>${escapeHtml(p.market_type)}</span>
        ${liveBadge}
      </div>
      <span class="${tierPillClass(p.confidence_tier)}">${escapeHtml(p.confidence_tier)} ${escapeHtml(p.confidence_score ?? '')}</span>
    </div>

    <div>
      <h3 class="pick-title">${escapeHtml(p.selection_label)}</h3>
      <p class="pick-subtitle">${escapeHtml(p.short_name || p.event_name)}${startLabel ? ` · ${escapeHtml(startLabel)}` : ''} · ${escapeHtml(lineSrcText)}${formatAgeText(p.odds_age_seconds)}</p>
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
      <div class="inline-pricer" title="Configure exact DraftKings line and contract ask price">
        ${lineEditorHtml}
        <select id="dk-side-${idx}" class="dk-side-select" data-idx="${idx}" aria-label="Contract side for ${escapeHtml(p.selection_label)}">
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
            value="${Number.isFinite(Number(p.contract_price)) ? Number((p.contract_price * 100).toFixed(1)) : ''}"
            data-idx="${idx}"
            aria-label="Contract price in cents for ${escapeHtml(p.selection_label)}"
          >
          <span class="cents-suffix" aria-hidden="true">¢</span>
        </div>
      </div>
      <button type="button" class="${btnCls}" data-idx="${idx}">${btnLabel}</button>
    </div>
  `;
}

// After logging a position, refresh KPIs (and the ledger table if visible) without letting a
// secondary failure masquerade as a failed log.
async function refreshLedgerAfterLog() {
  if (!document.getElementById('panel-ledger').hidden) {
    await loadLedger();
    return;
  }
  try {
    const led = await apiJson(`/api/ledger?bankroll=${getBankroll()}`);
    updateKPIs(led.summary, state.picks);
  } catch (err) {
    if (!isSilentError(err)) console.error('Ledger summary refresh failed:', err);
  }
}

async function postLedgerTrade(body) {
  return apiJson('/api/ledger', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function logPickByIndex(idx) {
  const pick = state.picks[idx];
  if (!pick) return;
  const qty = Math.max(1, pick.recommended_contracts || 1);
  try {
    await postLedgerTrade({
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
    });
    showBanner(
      `Logged ${qty}x ${pick.recommended_side} "${pick.selection_label}" @ ${pctSafe(pick.contract_price, 1)}¢ ($${toFixedSafe(qty * Number(pick.all_in_cost), 2)})`
    );
    await refreshLedgerAfterLog();
  } catch (err) {
    reportError('Failed logging position', err);
  }
}

// Re-render a single card in place, keeping keyboard focus on the equivalent control.
function rerenderPickCard(idx) {
  const cardEl = document.querySelector(`.pick-card[data-idx="${idx}"]`);
  if (!cardEl || !state.picks[idx]) return false;
  const active = document.activeElement;
  const focusCls =
    active && cardEl.contains(active)
      ? ['dk-line-input', 'dk-side-select', 'dk-cents-input', 'log-trade-btn', 'line-reset-btn'].find((c) =>
          active.classList.contains(c)
        )
      : null;
  cardEl.innerHTML = buildPickCardInnerHtml(state.picks[idx], idx);
  if (focusCls) {
    const target = cardEl.querySelector(`.${focusCls}`) || cardEl.querySelector('.dk-line-input, .dk-cents-input');
    if (target) target.focus();
  }
  return true;
}

async function applyCardLineOverride(idx, { reset = false, includePriceSide = false } = {}) {
  const pick = state.picks[idx];
  if (!pick || !pick.override_key) return;

  let rawLine = null;
  if (!reset) {
    const lnEl = document.getElementById(`dk-ln-${idx}`);
    rawLine = lnEl ? parseFloat(lnEl.value) : Number(pick.line_value);
    if (!Number.isFinite(rawLine)) {
      showBanner('Enter a valid numeric line (for example 20.5 or -3.5).', true);
      return;
    }
    if (!includePriceSide && Math.abs(rawLine - Number(pick.line_value)) < 1e-6) {
      return;
    }
  }

  const minProb = parseFloat(document.getElementById('filter-min-prob').value) / 100;
  const minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  const liveProps = document.getElementById('filter-live-props').checked;
  const dateVal = document.getElementById('filter-date').value;

  const payload = {
    override_key: pick.override_key,
    sport: pick.sport,
    event_id: String(pick.event_id),
    pick_id: pick.pick_id || '',
    date: dateVal || null,
    line_value: reset ? null : rawLine,
    reset,
    fee_per_contract: getFee(),
    bankroll: getBankroll(),
    min_win_prob: minProb,
    min_net_edge: minEdge,
    live_props: liveProps,
  };

  if (includePriceSide && !reset) {
    const sideEl = document.getElementById(`dk-side-${idx}`);
    const pxEl = document.getElementById(`dk-px-${idx}`);
    const side = sideEl ? sideEl.value : pick.recommended_side;
    const cents = Math.min(99, Math.max(1, parseFloat(pxEl ? pxEl.value : '50') || 50));
    payload.side = side;
    payload.contract_price = cents / 100;
  }

  const seqKey = `card:${idx}`;
  const seq = nextSeq(seqKey);
  try {
    const data = await apiJson('/api/line-override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!data || !data.pick) throw new Error('Invalid response payload');
    // Drop stale responses: a newer edit on this card, or the slate was reloaded meanwhile.
    if (!isLatestSeq(seqKey, seq) || state.picks[idx] !== pick) return;

    state.picks[idx] = data.pick;
    if (state.viewMode !== 'grid' || !rerenderPickCard(idx)) {
      renderPicks();
    }
    updateKPIs(null, state.picks);
    updateSlateMetaBar();
    if (reset) {
      showBanner(`Reset "${data.pick.selection_label}" to synced line (${data.pick.line_value}).`);
    } else if (!includePriceSide) {
      showBanner(`Updated custom line: ${data.pick.selection_label} · Win ${pctSafe(data.pick.model_prob, 1)}%`);
    }
  } catch (err) {
    reportError('Could not update custom line', err);
  }
}

async function reevaluateSingleCard(idx) {
  const pick = state.picks[idx];
  if (!pick) return;

  if (pick.is_custom_line && pick.override_key) {
    await applyCardLineOverride(idx, { reset: false, includePriceSide: true });
    return;
  }

  const sideEl = document.getElementById(`dk-side-${idx}`);
  const pxEl = document.getElementById(`dk-px-${idx}`);
  const side = sideEl ? sideEl.value : pick.recommended_side;
  const cents = Math.min(99, Math.max(1, parseFloat(pxEl ? pxEl.value : '50') || 50));
  const minProb = parseFloat(document.getElementById('filter-min-prob').value) / 100;
  const minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;

  const seqKey = `card:${idx}`;
  const seq = nextSeq(seqKey);
  try {
    const updated = await apiJson('/api/evaluate-contract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        yes_true_prob: pick.yes_true_prob,
        side,
        contract_price: cents / 100,
        fee_per_contract: getFee(),
        bankroll: getBankroll(),
        min_win_prob: Math.max(0.5, minProb),
        min_net_edge: minEdge,
        is_live: pick.game_state === 'LIVE',
      }),
    });
    if (!isLatestSeq(seqKey, seq) || state.picks[idx] !== pick) return;
    pick.recommended_side = updated.side;
    pick.selection_label = formatSelectionLabelForSide(pick, updated.side);
    Object.assign(pick, updated);

    if (state.viewMode === 'grid') rerenderPickCard(idx);
    updateKPIs(null, state.picks);
  } catch (err) {
    reportError('Could not re-evaluate contract', err);
  }
}

// Submit an editable card input once per distinct value (Enter + the follow-up `change` event
// would otherwise fire two identical requests).
function submitCardInput(inp) {
  const idx = parseInt(inp.dataset.idx, 10);
  if (!Number.isInteger(idx)) return;
  const key = `${inp.className}|${inp.value}`;
  if (inp.dataset.lastSubmitted === key) return;
  inp.dataset.lastSubmitted = key;
  const pending = inp.classList.contains('dk-line-input') ? applyCardLineOverride(idx) : reevaluateSingleCard(idx);
  Promise.resolve(pending).finally(() => {
    if (inp.dataset.lastSubmitted === key) delete inp.dataset.lastSubmitted;
  });
}

// Delegated handlers for pick cards / table rows: bound once on the stable containers instead of
// re-attaching listeners to every control on each render.
function bindPicksDelegation(container) {
  container.addEventListener('click', (e) => {
    const logBtn = e.target.closest('.log-trade-btn');
    if (logBtn && container.contains(logBtn)) {
      const idx = parseInt(logBtn.dataset.idx, 10);
      withBusy(logBtn, () => logPickByIndex(idx));
      return;
    }
    const sbBtn = e.target.closest('.log-sb-pick-btn');
    if (sbBtn && container.contains(sbBtn)) {
      const idx = parseInt(sbBtn.dataset.sbIdx, 10);
      withBusy(sbBtn, () => logSportsbookPickByIndex(idx));
      return;
    }
    const resetBtn = e.target.closest('.line-reset-btn');
    if (resetBtn && container.contains(resetBtn)) {
      const idx = parseInt(resetBtn.dataset.resetIdx, 10);
      withBusy(resetBtn, () => applyCardLineOverride(idx, { reset: true }));
    }
  });

  container.addEventListener('change', (e) => {
    const el = e.target;
    if (el.matches('.dk-line-input, .dk-cents-input')) {
      submitCardInput(el);
    } else if (el.matches('.dk-side-select')) {
      reevaluateSingleCard(parseInt(el.dataset.idx, 10));
    }
  });

  container.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const el = e.target;
    if (el.matches && el.matches('.dk-line-input, .dk-cents-input')) {
      e.preventDefault();
      submitCardInput(el);
    }
  });
}
bindPicksDelegation(document.getElementById('picks-grid'));
bindPicksDelegation(document.getElementById('picks-table-tbody'));

function buildSportsbookCardInnerHtml(p, idx) {
  const isLive = p.game_state === 'LIVE';
  const evCls = p.ev_pct >= 0 ? 'positive' : 'negative';
  const evSign = p.ev_pct >= 0 ? '+' : '';
  const liveBadge = isLive
    ? `<span class="live-indicator"><span class="live-dot"></span>LIVE ${escapeHtml(p.status_detail || '')}${p.score_display ? ` · ${escapeHtml(p.score_display)}` : ''}</span>`
    : '';
  const startLabel = !isLive ? formatEventStart(p.start_time, p.status_detail) : '';

  const dkBest = p.book === 'DRAFTKINGS';
  const fdBest = p.book === 'FANDUEL';
  const dkLnTxt = p.dk_line != null ? ` (${p.dk_line > 0 && p.market_type.includes('Spread') ? '+' : ''}${p.dk_line})` : '';
  const fdLnTxt = p.fd_line != null ? ` (${p.fd_line > 0 && p.market_type.includes('Spread') ? '+' : ''}${p.fd_line})` : '';
  const shopPill =
    p.line_shopping_gain_pct != null && p.line_shopping_gain_pct >= 0.004
      ? `<span class="strat-pill strat-arb" title="Line Shopping Gain: betting this side on ${dkBest ? 'DraftKings' : 'FanDuel'} yields +${(p.line_shopping_gain_pct * 100).toFixed(1)}% more EV than the other book">SHOP +${(p.line_shopping_gain_pct * 100).toFixed(1)}%</span>`
      : '';

  const drivers = p.drivers || [];
  const primaryDriver = drivers[0] ? `<p class="driver-primary">${escapeHtml(drivers[0])}</p>` : '';
  const extraDrivers = drivers.slice(1);
  const detailsHtml =
    extraDrivers.length > 0 || p.injury_notes
      ? `<details class="card-details">
          <summary>Model &amp; line details (Max acceptable ${formatAmerican(p.max_acceptable_odds)})</summary>
          <div class="details-body">
            ${extraDrivers.map((d) => `<p>• ${escapeHtml(d)}</p>`).join('')}
            ${p.injury_notes ? `<p class="warning-text">Injuries: ${escapeHtml(p.injury_notes)}</p>` : ''}
          </div>
        </details>`
      : '';

  const stakeVal = Math.max(1, Number(p.recommended_stake || 0) || Math.round(getBankroll() * 0.01));
  const bestShort = dkBest ? 'DK' : 'FD';
  const btnCls =
    p.confidence_tier === 'PASS'
      ? 'btn btn-secondary btn-sm log-sb-pick-btn'
      : 'btn btn-primary btn-sm log-sb-pick-btn';
  const btnLbl =
    p.confidence_tier === 'PASS'
      ? `Log on ${bestShort} ($${stakeVal.toFixed(0)})`
      : `Log on ${bestShort} · $${toFixedSafe(p.recommended_stake, 0)} (${formatAmerican(p.american_odds)})`;

  return `
    <div class="pick-header">
      <div class="pick-meta">
        <span class="sport-label">${escapeHtml(p.sport)}</span>
        <span aria-hidden="true">·</span>
        <span>${escapeHtml(p.market_type)}</span>
        ${venueBadgeHtml(p.book)}
        ${shopPill}
        ${liveBadge}
      </div>
      <span class="${tierPillClass(p.confidence_tier)}">${escapeHtml(p.confidence_tier)} ${escapeHtml(p.confidence_score ?? '')}</span>
    </div>

    <div>
      <h3 class="pick-title">${escapeHtml(p.selection_label)}</h3>
      <p class="pick-subtitle">${escapeHtml(p.short_name || p.event_name)}${startLabel ? ` · ${escapeHtml(startLabel)}` : ''} · ${escapeHtml(p.line_source || 'DK/FD')}${formatAgeText(p.odds_age_seconds)}</p>
    </div>

    <div class="book-odds-strip">
      <div class="book-odds-box ${dkBest ? 'is-best' : ''}">
        <span>DraftKings${escapeHtml(dkLnTxt)}</span>
        <strong class="mono">${p.dk_odds != null ? `${formatAmerican(p.dk_odds)}${dkBest ? ' ★' : ''}` : '—'}</strong>
      </div>
      <div class="book-odds-box ${fdBest ? 'is-best' : ''}">
        <span>FanDuel${escapeHtml(fdLnTxt)}</span>
        <strong class="mono">${p.fd_odds != null ? `${formatAmerican(p.fd_odds)}${fdBest ? ' ★' : ''}` : '—'}</strong>
      </div>
    </div>

    <div class="metrics-strip">
      <div class="stat-col">
        <span>Best Odds</span>
        <strong>${formatAmerican(p.american_odds)}</strong>
      </div>
      <div class="stat-col">
        <span>Fair Odds</span>
        <strong>${formatAmerican(p.fair_american_odds)}</strong>
      </div>
      <div class="stat-col">
        <span>Expected EV</span>
        <strong class="${evCls}">${evSign}${(p.ev_pct * 100).toFixed(1)}%</strong>
      </div>
      <div class="stat-col">
        <span>Win Prob</span>
        <strong>${(p.model_prob * 100).toFixed(1)}%</strong>
      </div>
    </div>

    ${primaryDriver}
    ${detailsHtml}

    <div class="pick-actions">
      <span class="mono" style="font-size:0.75rem;color:var(--text-secondary);">
        ${p.recommended_stake > 0 ? `Kelly $${p.recommended_stake.toFixed(2)} → +$${p.potential_profit.toFixed(2)} win` : `Edge ${evSign}${(p.prob_edge * 100).toFixed(1)}%`}
      </span>
      <button type="button" class="${btnCls}" data-sb-idx="${idx}">${btnLbl}</button>
    </div>
  `;
}

async function logSportsbookPickByIndex(idx) {
  const p = state.picks[idx];
  if (!p) return;
  const stake = Number(p.recommended_stake) > 0 ? Number(p.recommended_stake) : Math.max(5, Math.round(getBankroll() * 0.01));
  try {
    await postLedgerTrade({
      sport: p.sport,
      event_name: p.short_name || p.event_name,
      market_type: p.market_type,
      selection_label: p.selection_label,
      side: p.side,
      venue: p.book,
      bet_format: 'SPORTSBOOK',
      odds_american: p.american_odds,
      stake,
      model_prob: p.model_prob,
      strategy_tag: 'EV',
    });
    showBanner(`Logged $${stake.toFixed(2)} on ${venueDisplayName(p.book)} — ${p.selection_label} (${formatAmerican(p.american_odds)})`);
    await refreshLedgerAfterLog();
  } catch (err) {
    reportError('Failed logging sportsbook bet', err);
  }
}

function renderPicks() {
  const rows = filteredPicksWithIndices();
  const grid = document.getElementById('picks-grid');
  const tbody = document.getElementById('picks-table-tbody');

  if (!rows.length) {
    const hint = state.searchQuery
      ? `No lines matching "${state.searchQuery}". Clear the search box to see all active lines.`
      : state.mode === 'live'
        ? 'No live in-play games currently clear your edge filter. Switch timing to "All" or click Refresh.'
        : 'Open "Filters" to lower thresholds, enable "Show PASS lines", or click Refresh to pull latest lines.';

    const emptyHtml = `<div class="empty-state"><h3>No matching lines</h3><p>${escapeHtml(hint)}</p></div>`;
    grid.innerHTML = emptyHtml;
    tbody.innerHTML = `<tr><td colspan="10" class="table-empty">${escapeHtml(hint)}</td></tr>`;
    return;
  }

  // Click/change/keydown handlers are delegated on #picks-grid / #picks-table-tbody (see bindPicksDelegation).
  if (state.traditionalMode) {
    if (state.viewMode === 'grid') {
      grid.innerHTML = rows
        .map(({ p, idx }) => `<article class="pick-card" data-idx="${idx}">${buildSportsbookCardInnerHtml(p, idx)}</article>`)
        .join('');
    } else {
      tbody.innerHTML = rows
        .map(({ p, idx }) => {
          const isLive = p.game_state === 'LIVE';
          const evCls = p.ev_pct >= 0 ? 'positive' : 'negative';
          const evSign = p.ev_pct >= 0 ? '+' : '';
          const startTxt = !isLive ? formatEventStart(p.start_time, p.status_detail) : '';
          const stateLabel = isLive
            ? `<span class="live-indicator"><span class="live-dot"></span>${escapeHtml(p.score_display || p.status_detail || 'LIVE')}</span>`
            : `${escapeHtml(p.short_name || p.event_name)}${startTxt ? ` <span class="muted">· ${escapeHtml(startTxt)}</span>` : ''}`;
          const stakeDisp = Number(p.recommended_stake) > 0 ? `$${toFixedSafe(p.recommended_stake, 0)}` : '—';
          return `
            <tr>
              <td><span class="${tierPillClass(p.confidence_tier)}">${escapeHtml(p.confidence_tier)} ${escapeHtml(p.confidence_score ?? '')}</span></td>
              <td><span class="sport-label">${escapeHtml(p.sport)}</span> · ${stateLabel}</td>
              <td><strong>${escapeHtml(p.selection_label)}</strong></td>
              <td>${venueBadgeHtml(p.book)}</td>
              <td class="num-col mono">${formatAmerican(p.dk_odds)} / ${formatAmerican(p.fd_odds)}</td>
              <td class="num-col mono">${formatAmerican(p.fair_american_odds)}</td>
              <td class="num-col">${pctSafe(p.model_prob, 1)}%</td>
              <td class="num-col ${evCls}">${evSign}${pctSafe(p.ev_pct, 1)}%</td>
              <td class="num-col">${stakeDisp}</td>
              <td><button type="button" class="btn btn-secondary btn-sm log-sb-pick-btn" data-sb-idx="${idx}" aria-label="Log ${escapeHtml(p.selection_label)} on ${escapeHtml(venueDisplayName(p.book))}">Log</button></td>
            </tr>`;
        })
        .join('');
    }
    return;
  }

  if (state.viewMode === 'grid') {
    grid.innerHTML = rows
      .map(({ p, idx }) => `<article class="pick-card" data-idx="${idx}">${buildPickCardInnerHtml(p, idx)}</article>`)
      .join('');
  } else {
    tbody.innerHTML = rows
      .map(({ p, idx }) => {
        const isLive = p.game_state === 'LIVE';
        const edgeCls = p.net_edge >= 0 ? 'positive' : 'negative';
        const edgeSign = p.net_edge >= 0 ? '+' : '';
        const startTxt = !isLive ? formatEventStart(p.start_time, p.status_detail) : '';
        const stateLabel = isLive
          ? `<span class="live-indicator"><span class="live-dot"></span>${escapeHtml(p.score_display || p.status_detail || 'LIVE')}</span>`
          : `${escapeHtml(p.short_name || p.event_name)}${startTxt ? ` <span class="muted">· ${escapeHtml(startTxt)}</span>` : ''}`;
        const syncedLine = escapeHtml(String(p.default_line_value ?? p.line_value));
        const customResetBtn = p.is_custom_line
          ? ` <button type="button" class="line-reset-btn" data-reset-idx="${idx}" title="Reset to synced line (${syncedLine})" aria-label="Reset ${escapeHtml(p.selection_label)} to synced line ${syncedLine}">↺</button>`
          : '';
        const qty = Math.max(1, p.recommended_contracts || 1);
        return `
          <tr>
            <td><span class="${tierPillClass(p.confidence_tier)}">${escapeHtml(p.confidence_tier)} ${escapeHtml(p.confidence_score ?? '')}</span></td>
            <td><span class="sport-label">${escapeHtml(p.sport)}</span> · ${stateLabel}</td>
            <td><strong>${escapeHtml(p.selection_label)}</strong>${customResetBtn}</td>
            <td class="mono">${escapeHtml(p.recommended_side)}</td>
            <td class="num-col">${pctSafe(p.contract_price, 0)}¢</td>
            <td class="num-col">${pctSafe(p.fair_entry_max_price, 0)}¢</td>
            <td class="num-col">${pctSafe(p.model_prob, 1)}%</td>
            <td class="num-col ${edgeCls}">${edgeSign}${pctSafe(p.net_edge, 1)}%</td>
            <td class="num-col">${p.recommended_contracts ?? 0}x ($${toFixedSafe(p.recommended_outlay, 2)})</td>
            <td><button type="button" class="btn btn-secondary btn-sm log-trade-btn" data-idx="${idx}" aria-label="Log ${qty}x ${escapeHtml(p.recommended_side)} ${escapeHtml(p.selection_label)}">Log ${qty}x</button></td>
          </tr>
        `;
      })
      .join('');
  }
}

// --- Traditional DK/FD Strategies Workspace ---
async function logStrategyPair(legA, legB, successMsg) {
  try {
    await apiJson('/api/ledger/pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ leg_a: legA, leg_b: legB }),
    });
    showBanner(successMsg);
    await refreshLedgerAfterLog();
  } catch (err) {
    reportError('Could not log paired strategy', err);
  }
}

function renderStrategies() {
  const container = document.getElementById('strategies-container');
  const slate = state.sbSlate;
  if (!slate) {
    container.innerHTML = '<div class="empty-state"><p>Loading DK &amp; FD strategies...</p></div>';
    return;
  }

  const arbs = slate.arbitrage || [];
  const mids = slate.middles || [];
  const promos = slate.promos || { low_hold: [], bonus_bet: [], profit_boost: [] };
  const steams = slate.steam || [];

  const setCount = (id, n) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = String(n);
    el.hidden = n === 0;
  };
  setCount('strat-cnt-arb', arbs.length);
  setCount('strat-cnt-mid', mids.length);
  setCount('strat-cnt-promo', (promos.bonus_bet?.length || 0) + (promos.low_hold?.length || 0) + (promos.profit_boost?.length || 0));
  setCount('strat-cnt-steam', steams.length);

  const tab = state.strategyTab;
  let html = '';

  // 1. Arbitrage Section
  if (tab === 'all' || tab === 'arbitrage') {
    const cards = arbs
      .map((a, idx) => {
        const legA = a.leg_a;
        const legB = a.leg_b;
        const roiPct = ((a.is_guaranteed_after_rounding ? a.min_roi_pct : a.exact_roi_pct) * 100).toFixed(2);
        const typeBadge =
          a.arb_type === 'DKEX_CROSS'
            ? '<span class="strat-pill strat-middle">DKeX CROSS-ARB</span>'
            : '<span class="strat-pill strat-arb">DK ↔ FD ARB</span>';
        const roundNote = a.is_guaranteed_after_rounding
          ? `<span class="positive mono">Guaranteed +$${a.min_guaranteed_profit.toFixed(2)}${a.max_profit > a.min_guaranteed_profit ? ` to +$${a.max_profit.toFixed(2)}` : ''}</span>`
          : `<span class="warning-text mono">Exact +$${a.exact_guaranteed_profit.toFixed(2)} (rounding tight)</span>`;
        const legAPx = legA.is_dkex ? `${(legA.dkex_ask_price * 100).toFixed(0)}¢ + fee (${legA.dkex_contracts}x)` : `${formatAmerican(legA.american_odds)} · $${legA.rounded_stake.toFixed(2)}`;
        const legBPx = legB.is_dkex ? `${(legB.dkex_ask_price * 100).toFixed(0)}¢ + fee (${legB.dkex_contracts}x)` : `${formatAmerican(legB.american_odds)} · $${legB.rounded_stake.toFixed(2)}`;
        return `
          <article class="strategy-card">
            <div class="pick-header">
              <div class="pick-meta">
                ${typeBadge}
                <span class="sport-label">${escapeHtml(a.sport)}</span> · <span>${escapeHtml(a.market_type)}</span>
              </div>
              <span class="tier-pill tier-high">+${roiPct}% ROI</span>
            </div>
            <div style="font-size:0.78rem;color:var(--text-secondary);">${escapeHtml(a.short_name || a.event_name)}${a.is_cross_number_freeroll ? ' · <strong class="positive">Includes Free-Roll Middle!</strong>' : ''}</div>
            <div class="strategy-legs">
              <div class="strategy-leg">
                <div class="strategy-leg-top">${venueBadgeHtml(legA.book)}<strong class="mono">${legAPx}</strong></div>
                <div>${escapeHtml(legA.label)}</div>
                <span class="muted mono">Payout $${legA.payout_if_wins_rounded.toFixed(2)}</span>
              </div>
              <div class="strategy-leg">
                <div class="strategy-leg-top">${venueBadgeHtml(legB.book)}<strong class="mono">${legBPx}</strong></div>
                <div>${escapeHtml(legB.label)}</div>
                <span class="muted mono">Payout $${legB.payout_if_wins_rounded.toFixed(2)}</span>
              </div>
            </div>
            <div class="strategy-foot">
              <div>${roundNote} <span class="muted">on $${a.rounded_total_stake.toFixed(2)}</span></div>
              <button type="button" class="btn btn-primary btn-sm log-arb-pair-btn" data-arb-idx="${idx}">Log Arb Pair</button>
            </div>
          </article>`;
      })
      .join('');
    html += `
      <div>
        <div class="strategy-section-head">
          <h3>Cross-Book Arbitrage (${arbs.length})</h3>
          <span class="field-hint">Guaranteed profit across opposing legs with anti-limiting stake rounding</span>
        </div>
        ${arbs.length ? `<div class="strategy-grid">${cards}</div>` : '<div class="empty-state"><p>No live arbitrage pairs above your Min Arb ROI threshold right now. Use the Analyzer tab to test manual arb splits.</p></div>'}
      </div>`;
  }

  // 2. Middles Section
  if (tab === 'all' || tab === 'middles') {
    const cards = mids
      .map((m, idx) => {
        const keysHtml = (m.key_numbers_crossed || []).length
          ? `<span class="strat-pill strat-arb">KEY #${escapeHtml(m.key_numbers_crossed.join(', #'))}</span>`
          : '';
        const evCls = m.ev_roi_pct >= 0 ? 'positive' : 'negative';
        return `
          <article class="strategy-card">
            <div class="pick-header">
              <div class="pick-meta">
                <span class="strat-pill strat-middle">MIDDLE</span>
                ${keysHtml}
                <span class="sport-label">${escapeHtml(m.sport)}</span> · <span>${escapeHtml(m.market_type)}</span>
              </div>
              <span class="${m.is_positive_ev ? 'tier-pill tier-high' : 'tier-pill tier-pass'}">${m.ev_roi_pct >= 0 ? '+' : ''}${(m.ev_roi_pct * 100).toFixed(1)}% EV</span>
            </div>
            <div style="font-size:0.8rem;"><strong>${escapeHtml(m.middle_window_label)}</strong> <span class="muted">(${escapeHtml(m.short_name || m.event_name)})</span></div>
            <div class="strategy-legs">
              <div class="strategy-leg">
                <div class="strategy-leg-top">${venueBadgeHtml(m.book_a)}<strong class="mono">${formatAmerican(m.odds_a)}</strong></div>
                <div>${escapeHtml(m.label_a)}</div>
                <span class="muted mono">Stake $${m.stake_a.toFixed(2)}</span>
              </div>
              <div class="strategy-leg">
                <div class="strategy-leg-top">${venueBadgeHtml(m.book_b)}<strong class="mono">${formatAmerican(m.odds_b)}</strong></div>
                <div>${escapeHtml(m.label_b)}</div>
                <span class="muted mono">Stake $${m.stake_b.toFixed(2)}</span>
              </div>
            </div>
            <div class="strategy-foot">
              <div class="mono">
                Hit <strong>${(m.p_both_win * 100).toFixed(1)}%</strong> (BE ${(m.breakeven_middle_prob * 100).toFixed(1)}%) ·
                <span class="${evCls}">Win +$${m.double_win_profit.toFixed(0)} / Miss -$${m.miss_cost.toFixed(2)}</span>
              </div>
              <button type="button" class="btn btn-secondary btn-sm log-mid-pair-btn" data-mid-idx="${idx}">Log Middle Pair</button>
            </div>
          </article>`;
      })
      .join('');
    html += `
      <div>
        <div class="strategy-section-head">
          <h3>Middles &amp; Polish Middles (${mids.length})</h3>
          <span class="field-hint">Discrepant DK vs. FD spreads, totals, and player props scored with exact distributions</span>
        </div>
        ${mids.length ? `<div class="strategy-grid">${cards}</div>` : '<div class="empty-state"><p>No discrepant DK vs. FD spread, total, or prop numbers detected on the current slate.</p></div>'}
      </div>`;
  }

  // 3. Promos, Bonus Bets & Low-Hold Section
  if (tab === 'all' || tab === 'promos') {
    const bonList = promos.bonus_bet || [];
    const boostList = promos.profit_boost || [];
    const lhList = promos.low_hold || [];

    const bonCards = bonList.slice(0, 8).map((b, idx) => `
      <article class="strategy-card">
        <div class="pick-header">
          <div class="pick-meta">
            <span class="strat-pill strat-promo">BONUS BET</span>
            <span class="sport-label">${escapeHtml(b.sport)}</span> · <span>${escapeHtml(b.market_type)}</span>
          </div>
          <span class="${b.retention_pct >= 0.70 ? 'tier-pill tier-high' : 'tier-pill tier-mid'}">${(b.retention_pct * 100).toFixed(1)}% Retention</span>
        </div>
        <div style="font-size:0.78rem;color:var(--text-secondary);">${escapeHtml(b.short_name || b.event_name)} · Hold ${(b.two_way_hold * 100).toFixed(2)}%</div>
        <div class="strategy-legs">
          <div class="strategy-leg">
            <div class="strategy-leg-top">${venueBadgeHtml(b.bonus_book)}<strong class="mono">${formatAmerican(b.bonus_odds)}</strong></div>
            <div>Bonus: ${escapeHtml(b.bonus_label)}</div>
            <span class="muted mono">$${b.bonus_amount.toFixed(0)} Bonus Bet</span>
          </div>
          <div class="strategy-leg">
            <div class="strategy-leg-top">${venueBadgeHtml(b.hedge_book)}<strong class="mono">${formatAmerican(b.hedge_odds)}</strong></div>
            <div>Hedge: ${escapeHtml(b.hedge_label)}</div>
            <span class="muted mono">Cash Hedge $${b.hedge_stake.toFixed(2)}</span>
          </div>
        </div>
        <div class="strategy-foot">
          <span class="positive mono">Guaranteed Cash: +$${b.guaranteed_cash.toFixed(2)}</span>
          <button type="button" class="btn btn-secondary btn-sm log-bon-pair-btn" data-bon-idx="${idx}">Log Conversion</button>
        </div>
      </article>`).join('');

    const boostCards = boostList.slice(0, 6).map((bp, idx) => `
      <article class="strategy-card">
        <div class="pick-header">
          <div class="pick-meta">
            <span class="strat-pill strat-promo">+${Math.round(bp.profit_boost_pct * 100)}% BOOST</span>
            ${venueBadgeHtml(bp.book)}
            <span class="sport-label">${escapeHtml(bp.sport)}</span>
          </div>
          <span class="tier-pill tier-high">+${(bp.ev_pct * 100).toFixed(1)}% EV</span>
        </div>
        <div><strong>${escapeHtml(bp.selection_label)}</strong> <span class="muted">(${escapeHtml(bp.short_name || bp.event_name)})</span></div>
        <div class="strategy-foot">
          <span class="mono">${formatAmerican(bp.american_odds)} → <strong>${formatAmerican(bp.effective_american_odds)}</strong> · Win ${(bp.model_prob * 100).toFixed(1)}%</span>
          <button type="button" class="btn btn-secondary btn-sm log-boost-btn" data-boost-idx="${idx}">Log Boost</button>
        </div>
      </article>`).join('');

    const lhRows = lhList.slice(0, 6).map((lh) => `
      <div class="kv-row">
        <span>${escapeHtml(lh.sport)} · ${escapeHtml(lh.short_name)} — ${venueBadgeHtml(lh.leg_a.book)} ${escapeHtml(lh.leg_a.label)} (${formatAmerican(lh.leg_a.odds)}) vs ${venueBadgeHtml(lh.leg_b.book)} ${escapeHtml(lh.leg_b.label)} (${formatAmerican(lh.leg_b.odds)})</span>
        <strong class="positive mono">${(lh.hold_pct * 100).toFixed(2)}% hold</strong>
      </div>`).join('');

    html += `
      <div>
        <div class="strategy-section-head">
          <h3>Bonus Bet Conversions, Profit Boosts &amp; Low-Hold Rollover</h3>
          <span class="field-hint">${bonList.length} bonus pairs · ${boostList.length} boost targets · ${lhList.length} low-hold pairs</span>
        </div>
        ${bonCards ? `<div class="strategy-grid" style="margin-bottom:0.85rem;">${bonCards}</div>` : ''}
        ${boostCards ? `<div class="strategy-grid" style="margin-bottom:0.85rem;">${boostCards}</div>` : ''}
        ${lhRows ? `<div class="surface-panel"><h4 class="sub-heading" style="margin-top:0;">Lowest-Hold Rollover Pairs (&le; 2.5% Hold)</h4><div class="kv-list">${lhRows}</div></div>` : ''}
        ${!bonCards && !boostCards && !lhRows ? '<div class="empty-state"><p>No promo/low-hold opportunities on the current slate. Use the Analyzer tab to compute any manual Bonus Bet or Profit Boost.</p></div>' : ''}
      </div>`;
  }

  // 4. Steam / Off-Market Lag Section
  if (tab === 'all' || tab === 'steam') {
    const cards = steams
      .map((s, idx) => `
        <article class="strategy-card">
          <div class="pick-header">
            <div class="pick-meta">
              <span class="strat-pill strat-steam">STEAM LAG</span>
              ${venueBadgeHtml(s.stale_book)}
              <span class="sport-label">${escapeHtml(s.sport)}</span> · <span>${escapeHtml(s.market_type)}</span>
            </div>
            <span class="tier-pill tier-high">+${(s.ev_vs_sharp_pct * 100).toFixed(1)}% vs Sharp</span>
          </div>
          <div><strong>${escapeHtml(s.recommended_label)}</strong> <span class="muted">(${escapeHtml(s.short_name || s.event_name)})</span></div>
          <p class="driver-primary">${escapeHtml(s.summary)}</p>
          <div class="strategy-foot">
            <span class="mono">Sharp ${(s.sharp_prob * 100).toFixed(1)}% vs Book ${(s.stale_prob * 100).toFixed(1)}% (+${(s.prob_lag * 100).toFixed(1)}% lag)</span>
            <button type="button" class="btn btn-primary btn-sm log-steam-btn" data-steam-idx="${idx}">Log Lag Play</button>
          </div>
        </article>`)
      .join('');
    html += `
      <div>
        <div class="strategy-section-head">
          <h3>Steam / Off-Market Lag Alerts (${steams.length})</h3>
          <span class="field-hint">DK or FD prices lagging behind sharp-only consensus</span>
        </div>
        ${steams.length ? `<div class="strategy-grid">${cards}</div>` : '<div class="empty-state"><p>No DK or FD lines are materially lagging sharp consensus right now.</p></div>'}
      </div>`;
  }

  container.innerHTML = html;
}

// Delegated 1-click Strategy Loggers (bound once on #strategies-container)
document.getElementById('strategies-container').addEventListener('click', (e) => {
  const slate = state.sbSlate;
  if (!slate) return;

  const arbBtn = e.target.closest('.log-arb-pair-btn');
  if (arbBtn) {
    const a = (slate.arbitrage || [])[parseInt(arbBtn.dataset.arbIdx, 10)];
    if (!a) return;
    const toLeg = (lg) =>
      lg.is_dkex
        ? {
            sport: a.sport,
            event_name: a.short_name || a.event_name,
            market_type: a.market_type,
            selection_label: lg.label,
            side: lg.side === 'HOME' || lg.side === 'OVER' ? 'YES' : 'NO',
            entry_price: lg.dkex_ask_price,
            fee_per_contract: lg.dkex_fee,
            contracts: lg.dkex_contracts || 1,
            model_prob: lg.implied_prob,
            fair_exit_price: lg.implied_prob,
            venue: 'DKEX',
            bet_format: 'CONTRACT',
            strategy_tag: 'ARB',
          }
        : {
            sport: a.sport,
            event_name: a.short_name || a.event_name,
            market_type: a.market_type,
            selection_label: lg.label,
            side: lg.side,
            venue: lg.book,
            bet_format: 'SPORTSBOOK',
            odds_american: lg.american_odds,
            stake: lg.rounded_stake,
            model_prob: lg.implied_prob,
            strategy_tag: 'ARB',
          };
    withBusy(arbBtn, () =>
      logStrategyPair(
        toLeg(a.leg_a),
        toLeg(a.leg_b),
        `Logged Arbitrage Pair on ${a.short_name} (+${pctSafe(a.min_roi_pct, 2)}% ROI)`
      )
    );
    return;
  }

  const midBtn = e.target.closest('.log-mid-pair-btn');
  if (midBtn) {
    const m = (slate.middles || [])[parseInt(midBtn.dataset.midIdx, 10)];
    if (!m) return;
    withBusy(midBtn, () =>
      logStrategyPair(
        {
          sport: m.sport,
          event_name: m.short_name || m.event_name,
          market_type: m.market_type,
          selection_label: m.label_a,
          side: m.side_a,
          venue: m.book_a,
          bet_format: 'SPORTSBOOK',
          odds_american: m.odds_a,
          stake: m.stake_a,
          model_prob: 0.5,
          strategy_tag: 'MIDDLE',
        },
        {
          sport: m.sport,
          event_name: m.short_name || m.event_name,
          market_type: m.market_type,
          selection_label: m.label_b,
          side: m.side_b,
          venue: m.book_b,
          bet_format: 'SPORTSBOOK',
          odds_american: m.odds_b,
          stake: m.stake_b,
          model_prob: 0.5,
          strategy_tag: 'MIDDLE',
        },
        `Logged Middle Pair on ${m.short_name}`
      )
    );
    return;
  }

  const bonBtn = e.target.closest('.log-bon-pair-btn');
  if (bonBtn) {
    const b = (slate.promos?.bonus_bet || [])[parseInt(bonBtn.dataset.bonIdx, 10)];
    if (!b) return;
    withBusy(bonBtn, () =>
      logStrategyPair(
        {
          sport: b.sport,
          event_name: b.short_name || b.event_name,
          market_type: b.market_type,
          selection_label: b.bonus_label,
          side: b.bonus_side,
          venue: b.bonus_book,
          bet_format: 'SPORTSBOOK',
          odds_american: b.bonus_odds,
          stake: b.bonus_amount,
          is_bonus_bet: true,
          model_prob: 0.25,
          strategy_tag: 'BONUS_BET',
        },
        {
          sport: b.sport,
          event_name: b.short_name || b.event_name,
          market_type: b.market_type,
          selection_label: b.hedge_label,
          side: b.hedge_side,
          venue: b.hedge_book,
          bet_format: 'SPORTSBOOK',
          odds_american: b.hedge_odds,
          stake: b.hedge_stake,
          is_bonus_bet: false,
          model_prob: 0.75,
          strategy_tag: 'BONUS_BET',
        },
        `Logged Bonus Bet Conversion (+$${toFixedSafe(b.guaranteed_cash, 2)} guaranteed cash)`
      )
    );
    return;
  }

  const boostBtn = e.target.closest('.log-boost-btn');
  if (boostBtn) {
    const bp = (slate.promos?.profit_boost || [])[parseInt(boostBtn.dataset.boostIdx, 10)];
    if (!bp) return;
    const stake = Math.max(5, Number(bp.recommended_stake || 25));
    withBusy(boostBtn, async () => {
      try {
        await postLedgerTrade({
          sport: bp.sport,
          event_name: bp.short_name || bp.event_name,
          market_type: bp.market_type,
          selection_label: `${bp.selection_label} (+${Math.round(bp.profit_boost_pct * 100)}% Boost)`,
          side: bp.side,
          venue: bp.book,
          bet_format: 'SPORTSBOOK',
          odds_american: bp.effective_american_odds,
          stake,
          model_prob: bp.model_prob,
          strategy_tag: 'BOOST',
        });
        showBanner(`Logged Boosted Play on ${venueDisplayName(bp.book)} (${formatAmerican(bp.effective_american_odds)})`);
        await refreshLedgerAfterLog();
      } catch (err) {
        reportError('Failed logging boosted play', err);
      }
    });
    return;
  }

  const steamBtn = e.target.closest('.log-steam-btn');
  if (steamBtn) {
    const s = (slate.steam || [])[parseInt(steamBtn.dataset.steamIdx, 10)];
    if (!s) return;
    const stake = Math.max(5, Math.round(getBankroll() * 0.02));
    withBusy(steamBtn, async () => {
      try {
        await postLedgerTrade({
          sport: s.sport,
          event_name: s.short_name || s.event_name,
          market_type: s.market_type,
          selection_label: s.recommended_label,
          side: s.recommended_side,
          venue: s.stale_book,
          bet_format: 'SPORTSBOOK',
          odds_american: s.stale_odds,
          stake,
          model_prob: s.sharp_prob,
          strategy_tag: 'STEAM',
        });
        showBanner(`Logged Steam Lag play on ${venueDisplayName(s.stale_book)} (${formatAmerican(s.stale_odds)})`);
        await refreshLedgerAfterLog();
      } catch (err) {
        reportError('Failed logging steam play', err);
      }
    });
  }
});

// --- Standalone Contract Analyzer (DKeX) ---
const calcSideBtns = [document.getElementById('calc-side-yes'), document.getElementById('calc-side-no')];
calcSideBtns.forEach((btn) => {
  btn.addEventListener('click', (e) => {
    const side = e.currentTarget.dataset.side;
    document.getElementById('calc-side').value = side;
    setSegActive(calcSideBtns, e.currentTarget);
    runCustomCalculator();
  });
});

async function runCustomCalculator(e) {
  if (e) e.preventDefault();
  const seq = nextSeq('calc-dkex');
  const side = document.getElementById('calc-side').value;
  const yesProb = Math.min(99.5, Math.max(0.5, parseFloat(document.getElementById('calc-prob').value) || 65)) / 100;
  const price = Math.min(99, Math.max(1, parseFloat(document.getElementById('calc-price').value) || 55)) / 100;
  const isLive = document.getElementById('calc-is-live').checked;

  try {
    const ev = await apiJson('/api/evaluate-contract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        yes_true_prob: yesProb,
        side,
        contract_price: price,
        fee_per_contract: getFee(),
        bankroll: getBankroll(),
        min_win_prob: Math.max(0.50, parseFloat(document.getElementById('filter-min-prob').value) / 100),
        min_net_edge: parseFloat(document.getElementById('filter-min-edge').value) / 100,
        is_live: isLive,
      }),
    });
    if (!isLatestSeq('calc-dkex', seq)) return;
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
      <div class="kv-row"><span>Side True Win Probability</span><strong>${pctSafe(ev.model_prob, 1)}%</strong></div>
      <div class="kv-row"><span>All-In Cost (Ask + Fee)</span><strong>${pctSafe(ev.all_in_cost, 2)}¢ (${ev.american_equivalent > 0 ? '+' : ''}${ev.american_equivalent})</strong></div>
      <div class="kv-row"><span>Break-Even Win Rate</span><strong>${pctSafe(ev.breakeven_prob, 2)}%</strong></div>
      <div class="kv-row"><span>Post-Fee Net Edge</span><strong class="${edgeCls}">${ev.net_edge >= 0 ? '+' : ''}${pctSafe(ev.net_edge, 2)}%</strong></div>
      <div class="kv-row"><span>Expected ROI</span><strong class="${edgeCls}">${ev.roi_pct >= 0 ? '+' : ''}${pctSafe(ev.roi_pct, 2)}%</strong></div>
      <div class="kv-row"><span>Max Limit Entry Price</span><strong>${pctSafe(ev.fair_entry_max_price, 1)}¢</strong></div>
      <div class="kv-row"><span>Fair Mid-Game Exit Target</span><strong>${pctSafe(ev.fair_midgame_exit_price, 1)}¢</strong></div>
      <div class="kv-row"><span>${isLive ? 'In-Play ' : ''}Quarter-Kelly Position</span><strong>${ev.recommended_contracts ?? 0} contracts ($${toFixedSafe(ev.recommended_outlay, 2)})</strong></div>
    `;
  } catch (err) {
    if (isLatestSeq('calc-dkex', seq)) reportError('Analyzer error', err);
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
  document.getElementById('nt-venue').value = 'DKEX';
  document.getElementById('nt-dkex-fields').hidden = false;
  document.getElementById('nt-sb-fields').hidden = true;
  if (ev) {
    document.getElementById('nt-side').value = ev.side;
    document.getElementById('nt-contracts').value = String(Math.max(1, ev.recommended_contracts || 1));
    document.getElementById('nt-price').value = String(Number((ev.contract_price * 100).toFixed(1)));
    document.getElementById('nt-prob').value = String(Number((ev.model_prob * 100).toFixed(1)));
  }
  newTradeDialog.showModal();
});

// --- Traditional DK/FD 3-Mode Analyzer ---
const sbCalcBtns = Array.from(document.querySelectorAll('[data-sb-calc]'));
sbCalcBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.sbCalcMode = btn.dataset.sbCalc;
    setSegActive(sbCalcBtns, btn);
    document.getElementById('sb-calc-panel-ev').hidden = state.sbCalcMode !== 'ev';
    document.getElementById('sb-calc-panel-arb').hidden = state.sbCalcMode !== 'arb';
    document.getElementById('sb-calc-panel-bonus').hidden = state.sbCalcMode !== 'bonus';
    runSportsbookAnalyzer();
  });
});

function syncArbBookLabels() {
  const bA = document.getElementById('sb-arb-book-a').value;
  const bB = document.getElementById('sb-arb-book-b').value;
  document.getElementById('sb-arb-odds-a-lbl').textContent =
    bA === 'DKEX' ? 'Leg 1 DKeX Ask (¢, 1–99)' : 'Leg 1 Odds (American)';
  document.getElementById('sb-arb-odds-b-lbl').textContent =
    bB === 'DKEX' ? 'Leg 2 DKeX Ask (¢, 1–99)' : 'Leg 2 Odds (American)';
}
['sb-arb-book-a', 'sb-arb-book-b'].forEach((id) => {
  document.getElementById(id).addEventListener('change', () => {
    syncArbBookLabels();
    runSportsbookArbCalc();
  });
});

async function runSportsbookEvCalc(e) {
  if (e) e.preventDefault();
  const dkRaw = document.getElementById('sb-ev-dk').value.trim();
  const fdRaw = document.getElementById('sb-ev-fd').value.trim();
  if (dkRaw === '' && fdRaw === '') {
    showBanner('Enter DraftKings odds, FanDuel odds, or both.', true);
    return;
  }
  const seq = nextSeq('calc-sb-ev');
  const probRaw = document.getElementById('sb-ev-prob').value.trim();
  const shA = document.getElementById('sb-ev-sharp-a').value.trim();
  const shB = document.getElementById('sb-ev-sharp-b').value.trim();
  const boost = parseFloat(document.getElementById('sb-ev-boost').value) || 0;

  const payload = {
    dk_odds: dkRaw !== '' ? parseFloat(dkRaw) : null,
    fd_odds: fdRaw !== '' ? parseFloat(fdRaw) : null,
    true_prob: shA && shB ? null : (probRaw !== '' ? parseFloat(probRaw) / 100 : 0.55),
    sharp_odds_side: shA ? parseFloat(shA) : null,
    sharp_odds_opp: shB ? parseFloat(shB) : null,
    bankroll: getBankroll(),
    profit_boost_pct: boost,
  };
  try {
    const data = await apiJson('/api/sportsbook/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!isLatestSeq('calc-sb-ev', seq)) return;
    const b = data.best;
    const evCls = b.ev_pct >= 0 ? 'positive' : 'negative';
    document.getElementById('sb-ev-verdict').textContent =
      b.confidence_tier === 'PASS'
        ? `PASS — Best on ${venueDisplayName(b.book)} (${formatAmerican(b.effective_american_odds)})`
        : `BET ${venueDisplayName(b.book)} ${formatAmerican(b.effective_american_odds)}`;
    document.getElementById('sb-ev-verdict').className = `verdict-title ${b.confidence_tier === 'PASS' ? '' : 'positive'}`;
    const pill = document.getElementById('sb-ev-tier');
    pill.textContent = `${b.confidence_tier} · ${b.confidence_score}/100`;
    pill.className = tierPillClass(b.confidence_tier);

    const dkRow = data.books.DRAFTKINGS
      ? `<div class="kv-row"><span>DraftKings (${formatAmerican(data.books.DRAFTKINGS.effective_american_odds)})</span><strong class="${data.books.DRAFTKINGS.ev_pct >= 0 ? 'positive' : 'negative'}">${data.books.DRAFTKINGS.ev_pct >= 0 ? '+' : ''}${pctSafe(data.books.DRAFTKINGS.ev_pct, 2)}% EV</strong></div>`
      : '';
    const fdRow = data.books.FANDUEL
      ? `<div class="kv-row"><span>FanDuel (${formatAmerican(data.books.FANDUEL.effective_american_odds)})</span><strong class="${data.books.FANDUEL.ev_pct >= 0 ? 'positive' : 'negative'}">${data.books.FANDUEL.ev_pct >= 0 ? '+' : ''}${pctSafe(data.books.FANDUEL.ev_pct, 2)}% EV</strong></div>`
      : '';
    document.getElementById('sb-ev-output').innerHTML = `
      <div class="kv-row"><span>True Win Prob / Fair Odds</span><strong>${pctSafe(data.true_prob, 1)}% (${formatAmerican(data.fair_american_odds)})</strong></div>
      ${dkRow}
      ${fdRow}
      ${data.line_shopping_gain_pct != null ? `<div class="kv-row"><span>Line Shopping Gain</span><strong class="positive">+${pctSafe(data.line_shopping_gain_pct, 2)}% EV</strong></div>` : ''}
      <div class="kv-row"><span>Best Expected ROI</span><strong class="${evCls}">${b.ev_pct >= 0 ? '+' : ''}${pctSafe(b.ev_pct, 2)}%</strong></div>
      <div class="kv-row"><span>Recommended Kelly Stake</span><strong>$${toFixedSafe(b.recommended_stake, 2)} (Win +$${toFixedSafe(b.potential_profit, 2)})</strong></div>
    `;
  } catch (err) {
    if (isLatestSeq('calc-sb-ev', seq)) reportError('Evaluator error', err);
  }
}

async function runSportsbookArbCalc(e) {
  if (e) e.preventDefault();
  const bA = document.getElementById('sb-arb-book-a').value;
  const bB = document.getElementById('sb-arb-book-b').value;
  const valA = parseFloat(document.getElementById('sb-arb-odds-a').value);
  const valB = parseFloat(document.getElementById('sb-arb-odds-b').value);
  if (!Number.isFinite(valA) || !Number.isFinite(valB)) {
    showBanner('Enter valid odds or contract prices for both legs.', true);
    return;
  }
  const seq = nextSeq('calc-sb-arb');
  const total = parseFloat(document.getElementById('sb-arb-total').value) || 100;
  const fixedA = document.getElementById('sb-arb-fixed-a').value.trim();
  const step = parseFloat(document.getElementById('sb-arb-round').value) || 0;

  const payload = {
    calc_mode: 'arb',
    book_a: bA,
    odds_a: bA === 'DKEX' ? null : valA,
    dkex_ask_a: bA === 'DKEX' ? (valA > 1 ? valA / 100 : valA) : null,
    book_b: bB,
    odds_b: bB === 'DKEX' ? null : valB,
    dkex_ask_b: bB === 'DKEX' ? (valB > 1 ? valB / 100 : valB) : null,
    total_stake: total,
    fixed_stake_a: fixedA ? parseFloat(fixedA) : null,
    rounding_step: step,
  };
  try {
    const r = await apiJson('/api/sportsbook/arb-calc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!isLatestSeq('calc-sb-arb', seq)) return;
    const isArb = Boolean(r.is_arb);
    const vEl = document.getElementById('sb-arb-verdict');
    vEl.textContent = isArb
      ? `ARBITRAGE · +${pctSafe(r.exact_roi_pct, 2)}% Guaranteed ROI`
      : `HEDGE · ${pctSafe(r.hold_pct, 2)}% Two-Way Hold`;
    vEl.className = `verdict-title ${isArb ? 'positive' : ''}`;
    const pill = document.getElementById('sb-arb-pill');
    pill.textContent = isArb ? (r.is_guaranteed_after_rounding ? 'GUARANTEED' : 'ROUNDING TIGHT') : 'NO ARB';
    pill.className = isArb ? 'tier-pill tier-high' : 'tier-pill tier-pass';

    const profitCls = r.min_guaranteed_profit > 0 ? 'positive' : 'negative';
    document.getElementById('sb-arb-output').innerHTML = `
      <div class="kv-row"><span>${venueDisplayName(r.leg_a.book)} Leg 1 (${formatAmerican(r.leg_a.american_odds)})</span><strong>Stake $${toFixedSafe(r.leg_a.rounded_stake, 2)} (Exact $${toFixedSafe(r.leg_a.exact_stake, 2)})</strong></div>
      <div class="kv-row"><span>${venueDisplayName(r.leg_b.book)} Leg 2 (${formatAmerican(r.leg_b.american_odds)})</span><strong>Stake $${toFixedSafe(r.leg_b.rounded_stake, 2)} (Exact $${toFixedSafe(r.leg_b.exact_stake, 2)})</strong></div>
      <div class="kv-row"><span>Total Rounded Outlay</span><strong>$${toFixedSafe(r.rounded_total_stake, 2)}</strong></div>
      <div class="kv-row"><span>Worst-Case Profit (Rounded)</span><strong class="${profitCls}">${r.min_guaranteed_profit >= 0 ? '+' : '-'}$${toFixedSafe(Math.abs(r.min_guaranteed_profit), 2)} (${pctSafe(r.min_roi_pct, 2)}%)</strong></div>
      <div class="kv-row"><span>Best-Case Profit (Rounded)</span><strong class="${r.max_profit >= 0 ? 'positive' : 'negative'}">${r.max_profit >= 0 ? '+' : '-'}$${toFixedSafe(Math.abs(r.max_profit), 2)}</strong></div>
      <div class="kv-row"><span>Implied Cost Sum</span><strong>${pctSafe(r.implied_sum, 2)}%</strong></div>
    `;
  } catch (err) {
    if (isLatestSeq('calc-sb-arb', seq)) reportError('Arb calc error', err);
  }
}

async function runSportsbookBonusCalc(e) {
  if (e) e.preventDefault();
  const seq = nextSeq('calc-sb-bonus');
  const bookA = document.getElementById('sb-bon-book').value;
  const bookB = bookA === 'DRAFTKINGS' ? 'FANDUEL' : 'DRAFTKINGS';
  const payload = {
    calc_mode: 'bonus_bet',
    book_a: bookA,
    odds_a: parseFloat(document.getElementById('sb-bon-odds').value) || 380,
    book_b: bookB,
    odds_b: parseFloat(document.getElementById('sb-bon-hedge-odds').value) || -440,
    bonus_amount: parseFloat(document.getElementById('sb-bon-amt').value) || 50,
  };
  try {
    const r = await apiJson('/api/sportsbook/arb-calc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!isLatestSeq('calc-sb-bonus', seq)) return;
    const retPct = pctSafe(r.retention_pct, 1);
    document.getElementById('sb-bon-verdict').textContent = `+$${toFixedSafe(r.guaranteed_cash, 2)} Guaranteed Cash (${retPct}%)`;
    document.getElementById('sb-bon-verdict').className = 'verdict-title positive';
    const pill = document.getElementById('sb-bon-pill');
    pill.textContent = r.retention_pct >= 0.70 ? 'OPTIMAL 70%+' : 'MODERATE';
    pill.className = r.retention_pct >= 0.70 ? 'tier-pill tier-high' : 'tier-pill tier-mid';

    document.getElementById('sb-bon-output').innerHTML = `
      <div class="kv-row"><span>Place Bonus Bet (${venueDisplayName(r.bonus_book)})</span><strong>$${toFixedSafe(r.bonus_amount, 2)} @ ${formatAmerican(r.bonus_odds)}</strong></div>
      <div class="kv-row"><span>Place Cash Hedge (${venueDisplayName(r.hedge_book)})</span><strong>$${toFixedSafe(r.hedge_stake, 2)} @ ${formatAmerican(r.hedge_odds)}</strong></div>
      <div class="kv-row"><span>Guaranteed Net Cash (Either Outcome)</span><strong class="positive">+$${toFixedSafe(r.guaranteed_cash, 2)}</strong></div>
      <div class="kv-row"><span>Conversion Retention %</span><strong class="positive">${retPct}%</strong></div>
      <div class="kv-row"><span>Underlying Market Hold</span><strong>${pctSafe(r.two_way_hold, 2)}%</strong></div>
    `;
  } catch (err) {
    if (isLatestSeq('calc-sb-bonus', seq)) reportError('Bonus calc error', err);
  }
}

function runSportsbookAnalyzer() {
  if (state.sbCalcMode === 'ev') runSportsbookEvCalc();
  else if (state.sbCalcMode === 'arb') runSportsbookArbCalc();
  else runSportsbookBonusCalc();
}

document.getElementById('sb-ev-form').addEventListener('submit', runSportsbookEvCalc);
document.getElementById('sb-arb-form').addEventListener('submit', runSportsbookArbCalc);
document.getElementById('sb-bonus-form').addEventListener('submit', runSportsbookBonusCalc);

// --- Unified Multi-Venue Trade Ledger ---
const ledgerFilterBtns = Array.from(document.querySelectorAll('[data-ledger-filter]'));
ledgerFilterBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.ledgerFilter = btn.dataset.ledgerFilter;
    setSegActive(ledgerFilterBtns, btn);
    renderLedgerRows();
  });
});

const venueFilterBtns = Array.from(document.querySelectorAll('[data-venue-filter]'));
venueFilterBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    state.ledgerVenueFilter = btn.dataset.venueFilter;
    setSegActive(venueFilterBtns, btn);
    renderLedgerRows();
  });
});

document.getElementById('nt-venue').addEventListener('change', (e) => {
  const isDkex = e.target.value === 'DKEX';
  document.getElementById('nt-dkex-fields').hidden = !isDkex;
  document.getElementById('nt-sb-fields').hidden = isDkex;
});

document.getElementById('new-trade-btn').addEventListener('click', () => {
  document.getElementById('nt-event').value = '';
  document.getElementById('nt-selection').value = '';
  if (state.traditionalMode) {
    document.getElementById('nt-venue').value = 'DRAFTKINGS';
    document.getElementById('nt-dkex-fields').hidden = true;
    document.getElementById('nt-sb-fields').hidden = false;
  } else {
    document.getElementById('nt-venue').value = 'DKEX';
    document.getElementById('nt-dkex-fields').hidden = false;
    document.getElementById('nt-sb-fields').hidden = true;
  }
  newTradeDialog.showModal();
});

document.getElementById('new-trade-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const submitBtn = document.getElementById('save-new-trade-btn');
  const useSb = state.traditionalMode && document.getElementById('nt-venue').value !== 'DKEX';
  let body;
  if (useSb) {
    const odds = parseInt(document.getElementById('nt-sb-odds').value, 10) || -110;
    const stake = parseFloat(document.getElementById('nt-sb-stake').value) || 25;
    const prob = Math.min(99, Math.max(1, parseFloat(document.getElementById('nt-sb-prob').value) || 55)) / 100;
    body = {
      sport: document.getElementById('nt-sport').value,
      market_type: document.getElementById('nt-market').value,
      event_name: document.getElementById('nt-event').value.trim(),
      selection_label: document.getElementById('nt-selection').value.trim(),
      side: document.getElementById('nt-sb-side').value,
      venue: document.getElementById('nt-venue').value,
      bet_format: 'SPORTSBOOK',
      odds_american: odds,
      stake,
      is_bonus_bet: document.getElementById('nt-sb-bonus').checked,
      strategy_tag: document.getElementById('nt-strategy').value,
      model_prob: prob,
    };
  } else {
    const entryPrice = Math.min(99, Math.max(1, parseFloat(document.getElementById('nt-price').value) || 55)) / 100;
    const modelProb = Math.min(99, Math.max(1, parseFloat(document.getElementById('nt-prob').value) || 60)) / 100;
    const fee = getFee();
    body = {
      sport: document.getElementById('nt-sport').value,
      market_type: document.getElementById('nt-market').value,
      event_name: document.getElementById('nt-event').value.trim(),
      selection_label: document.getElementById('nt-selection').value.trim(),
      side: document.getElementById('nt-side').value,
      contracts: Math.max(1, parseInt(document.getElementById('nt-contracts').value, 10) || 1),
      entry_price: entryPrice,
      fee_per_contract: fee,
      model_prob: modelProb,
      fair_exit_price: Math.min(0.99, Math.max(0.01, modelProb + fee)),
      venue: 'DKEX',
      bet_format: 'CONTRACT',
    };
  }

  await withBusy(submitBtn, async () => {
    try {
      await apiJson('/api/ledger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      newTradeDialog.close();
      showBanner('Position logged to ledger.');
      await loadLedger();
    } catch (err) {
      reportError('Could not log position', err);
    }
  });
});

function renderVenueBankrollStrip(summary) {
  const strip = document.getElementById('ledger-venue-strip');
  if (!state.traditionalMode || !summary || !summary.by_venue) {
    strip.hidden = true;
    return;
  }
  const venues = ['DRAFTKINGS', 'FANDUEL'];
  if ((summary.by_venue.DKEX?.open_trades_count || 0) + (summary.by_venue.DKEX?.settled_trades_count || 0) > 0) {
    venues.push('DKEX');
  }
  strip.hidden = false;
  strip.innerHTML = venues
    .map((v) => {
      const info = summary.by_venue[v];
      if (!info) return '';
      const pnlCls = info.realized_pnl > 0 ? 'positive' : info.realized_pnl < 0 ? 'negative' : '';
      const pnlSign = info.realized_pnl > 0 ? '+' : info.realized_pnl < 0 ? '-' : '';
      return `
        <div class="venue-bankroll-card">
          <div class="venue-bankroll-top">
            ${venueBadgeHtml(v)}
            <strong class="mono">$${formatMoney(info.current_bankroll)}</strong>
          </div>
          <div class="venue-bankroll-meta mono">
            <span class="${pnlCls}">P&amp;L ${pnlSign}$${formatMoney(Math.abs(info.realized_pnl))}</span>
            <span>Open $${formatMoney(info.open_exposure)}</span>
            <span>${info.settled_trades_count ?? 0} settled</span>
          </div>
        </div>`;
    })
    .join('');
}

const KNOWN_STATUS_CLASSES = new Set(['OPEN', 'WON', 'LOST', 'SOLD', 'PUSH', 'VOID', 'CASHOUT']);

function renderLedgerRows() {
  const tbody = document.getElementById('ledger-tbody');
  const filtered = state.trades.filter((t) => {
    if (state.ledgerFilter === 'OPEN' && t.status !== 'OPEN') return false;
    if (state.ledgerFilter === 'SETTLED' && t.status === 'OPEN') return false;
    if (state.traditionalMode && state.ledgerVenueFilter !== 'ALL') {
      if (String(t.venue || 'DKEX').toUpperCase() !== state.ledgerVenueFilter) return false;
    }
    return true;
  });

  if (!filtered.length) {
    const msg = state.trades.length
      ? 'No matching positions for this filter.'
      : 'No positions logged yet. Log directly from any Market/Strategy card or click "+ Log Position".';
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:2rem;">${escapeHtml(msg)}</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((t) => {
      const isSb = String(t.bet_format || 'CONTRACT').toUpperCase() === 'SPORTSBOOK';
      const pnlVal = Number(t.pnl || 0);
      const pnlCls = pnlVal > 0 ? 'positive' : pnlVal < 0 ? 'negative' : '';
      const pnlStr = t.status === 'OPEN' ? '—' : `${pnlVal >= 0 ? '+' : '-'}$${toFixedSafe(Math.abs(pnlVal), 2)}`;
      const probTxt =
        t.status === 'OPEN' && t.live_model_prob != null
          ? `${pctSafe(t.model_prob, 0)}% → <strong>${pctSafe(t.live_model_prob, 1)}%</strong>`
          : `${pctSafe(t.model_prob, 1)}%`;
      const guidanceTxt =
        t.status === 'OPEN' && t.live_guidance
          ? escapeHtml(t.live_guidance)
          : isSb
            ? `Payout $${toFixedSafe(t.potential_payout ?? 0, 2)}`
            : `Exit ${pctSafe(t.fair_exit_price, 0)}¢`;

      const sizePriceTxt = isSb
        ? `$${toFixedSafe(t.stake ?? t.total_cost, 2)} @ ${formatAmerican(t.odds_american)}${t.is_bonus_bet ? ' (Bonus)' : ''}`
        : `${t.contracts ?? 0} @ ${pctSafe(t.entry_price, 1)}¢`;
      const stratBadge =
        t.strategy_tag && t.strategy_tag !== 'EV'
          ? ` <span class="strat-pill">${escapeHtml(t.strategy_tag)}</span>`
          : '';
      const vBadge = state.traditionalMode || (t.venue && t.venue !== 'DKEX') ? `${venueBadgeHtml(t.venue)} ` : '';
      const statusUpper = String(t.status || '').toUpperCase();
      const statusCls = KNOWN_STATUS_CLASSES.has(statusUpper) ? statusUpper : '';

      return `
        <tr>
          <td>
            <div>${vBadge}<strong>${escapeHtml(t.selection_label)}</strong>${stratBadge}</div>
            <div style="font-size:0.72rem;color:var(--text-muted);">${escapeHtml(t.event_name)} · ${escapeHtml(t.market_type)}</div>
          </td>
          <td class="mono">${escapeHtml(t.sport)}</td>
          <td class="mono"><strong>${escapeHtml(t.side)}</strong></td>
          <td class="num-col mono">${sizePriceTxt}</td>
          <td class="num-col">$${toFixedSafe(t.total_cost, 2)}</td>
          <td class="num-col">${probTxt}</td>
          <td style="font-size:0.77rem;color:var(--text-secondary);">${guidanceTxt}</td>
          <td><span class="status-badge ${statusCls}">${escapeHtml(t.status)}</span></td>
          <td class="num-col ${pnlCls}"><strong>${pnlStr}</strong></td>
          <td>
            <div class="row-actions">
              <button type="button" class="btn btn-secondary btn-sm settle-open-btn" data-id="${t.id}">Settle</button>
              <button type="button" class="btn btn-danger-ghost btn-sm delete-trade-btn" data-id="${t.id}" aria-label="Remove position ${escapeHtml(t.selection_label)}">Remove</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join('');
}

// Delegated ledger row actions (bound once on #ledger-tbody)
document.getElementById('ledger-tbody').addEventListener('click', (e) => {
  const settleBtn = e.target.closest('.settle-open-btn');
  if (settleBtn) {
    const id = parseInt(settleBtn.dataset.id, 10);
    const trade = state.trades.find((x) => x.id === id);
    if (!trade) return;
    const isSb = String(trade.bet_format || 'CONTRACT').toUpperCase() === 'SPORTSBOOK';
    document.getElementById('settle-trade-id').value = String(trade.id);
    document.getElementById('settle-trade-format').value = isSb ? 'SPORTSBOOK' : 'CONTRACT';
    document.getElementById('settle-trade-summary').textContent = isSb
      ? `${venueDisplayName(trade.venue)} — ${trade.selection_label} ($${toFixedSafe(trade.stake ?? trade.total_cost, 2)} @ ${formatAmerican(trade.odds_american)})`
      : `${trade.contracts}x ${trade.side} — ${trade.selection_label} (All-in cost $${toFixedSafe(trade.total_cost, 2)})`;

    document.getElementById('opt-settle-sold').hidden = isSb;
    document.getElementById('opt-settle-push').hidden = !isSb;
    document.getElementById('opt-settle-void').hidden = !isSb;
    document.getElementById('opt-settle-cashout').hidden = !isSb;
    document.getElementById('settle-clv-cents-wrap').hidden = isSb;
    document.getElementById('settle-clv-odds-wrap').hidden = !isSb;

    document.getElementById('settle-status').value = 'WON';
    document.getElementById('sold-fields').hidden = true;
    document.getElementById('cashout-fields').hidden = true;
    document.getElementById('settle-closing-price').value = '';
    document.getElementById('settle-closing-odds').value = '';
    settleDialog.showModal();
    return;
  }

  const delBtn = e.target.closest('.delete-trade-btn');
  if (delBtn) {
    const id = parseInt(delBtn.dataset.id, 10);
    const trade = state.trades.find((x) => x.id === id);
    const label = trade ? `${trade.selection_label} (#${id})` : `#${id}`;
    if (!window.confirm(`Remove position ${label} from your ledger? This cannot be undone.`)) return;
    withBusy(delBtn, async () => {
      try {
        await apiJson(`/api/ledger/${id}`, { method: 'DELETE' });
        showBanner(`Removed position #${id}.`);
        await loadLedger();
      } catch (err) {
        reportError(`Could not remove position #${id}`, err);
      }
    });
  }
});

async function loadLedger() {
  const seq = nextSeq('ledger');
  try {
    const data = await apiJson(`/api/ledger?bankroll=${getBankroll()}`);
    if (!isLatestSeq('ledger', seq)) return;
    state.trades = data.trades || [];
    const noticeEl = document.getElementById('ledger-guidance-notice');
    if (noticeEl) {
      if (data.guidance_error) {
        noticeEl.textContent = `${data.guidance_error} — showing saved positions without live hold/exit signals.`;
        noticeEl.hidden = false;
      } else {
        noticeEl.textContent = '';
        noticeEl.hidden = true;
      }
    }
    updateKPIs(data.summary, state.picks);
    renderVenueBankrollStrip(data.summary);
    renderLedgerRows();
  } catch (err) {
    if (isLatestSeq('ledger', seq)) reportError('Failed loading ledger', err);
  }
}

document.getElementById('settle-status').addEventListener('change', (e) => {
  document.getElementById('sold-fields').hidden = e.target.value !== 'SOLD';
  document.getElementById('cashout-fields').hidden = e.target.value !== 'CASHOUT';
});

document.getElementById('settle-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const submitBtn = document.getElementById('save-settle-btn');
  const id = parseInt(document.getElementById('settle-trade-id').value, 10);
  const fmt = document.getElementById('settle-trade-format').value;
  const status = document.getElementById('settle-status').value;

  const payload = { status };
  if (fmt === 'SPORTSBOOK') {
    if (status === 'CASHOUT') {
      payload.cashout_amount = parseFloat(document.getElementById('settle-cashout-amt').value || '0');
    }
    const clvOdds = document.getElementById('settle-closing-odds').value.trim();
    if (clvOdds) payload.closing_odds_american = parseInt(clvOdds, 10);
  } else {
    const sellPxRaw = parseFloat(document.getElementById('settle-sell-price').value || '75') / 100;
    const sellFeeRaw = parseFloat(document.getElementById('settle-sell-fee').value || '1.5') / 100;
    const clvRaw = document.getElementById('settle-closing-price').value;
    payload.exit_or_settle_price = status === 'SOLD' ? sellPxRaw : null;
    payload.exit_fee_per_contract = status === 'SOLD' ? sellFeeRaw : 0.0;
    payload.closing_price = clvRaw ? parseFloat(clvRaw) / 100 : null;
  }

  await withBusy(submitBtn, async () => {
    try {
      await apiJson(`/api/ledger/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      settleDialog.close();
      showBanner(`Position #${id} settled as ${status}.`);
      await loadLedger();
    } catch (err) {
      reportError('Could not save settlement', err);
    }
  });
});

// --- Calibration & Strategy Backtest Verification ---
function renderSportsbookBacktest(report) {
  const container = document.getElementById('backtest-container');
  const comb = report.combined || {};
  const verdicts = comb.verdicts || {};
  const ev = comb.ev_line_shopping || {};
  const arb = comb.arbitrage || {};
  const mid = comb.middles || {};
  const promo = comb.promos_and_boosts || {};
  const steam = comb.steam_lag || {};

  const vBadge = (status) => {
    if (status === 'VERIFIED_PROFITABLE') return '<span class="strat-pill verdict-verified">VERIFIED PROFITABLE</span>';
    if (status === 'POSITIVE_FILTER_LIFT' || status === 'POSITIVE_LIFT' || status === 'POSITIVE_CLV') {
      return `<span class="strat-pill verdict-lift">${escapeHtml(status.replace(/_/g, ' '))}</span>`;
    }
    return `<span class="strat-pill">${escapeHtml((status || 'MIXED').replace(/_/g, ' '))}</span>`;
  };

  const pct = (v, signed = true) => `${signed && v >= 0 ? '+' : ''}${(Number(v || 0) * 100).toFixed(2)}%`;

  const stressRows = Object.values(report.stress_regimes || {})
    .map((sr) => `
      <tr>
        <td><strong>${escapeHtml(sr.label)}</strong></td>
        <td class="num-col ${sr.combined_roi_pct >= 0 ? 'positive' : 'negative'}"><strong>${pct(sr.combined_roi_pct)}</strong> ($${formatMoney(sr.combined_net_pnl)})</td>
        <td class="num-col positive">${pct(sr.arb_roi_pct)} (${sr.arb_count})</td>
        <td class="num-col positive">${(sr.bonus_retention_pct * 100).toFixed(1)}%</td>
        <td class="num-col ${sr.boost_25_roi_pct >= 0 ? 'positive' : 'negative'}">${pct(sr.boost_25_roi_pct)}</td>
        <td class="num-col ${sr.steam_roi_pct >= 0 ? 'positive' : 'negative'}">${pct(sr.steam_roi_pct)}</td>
        <td class="num-col ${sr.middles_roi_pct >= 0 ? 'positive' : 'negative'}">${pct(sr.middles_roi_pct)} (${sr.middles_count})</td>
      </tr>`)
    .join('');

  const leagueCards = Object.entries(report.by_league || {})
    .map(([lg, info]) => {
      const binsHtml = (info.calibration_bins || [])
        .map((b) => `
          <div class="cal-bin-row">
            <span class="mono">${escapeHtml(b.bin_label)}</span>
            <div class="cal-bar-track"><div class="cal-bar-actual" style="width:${Math.min(100, Math.max(2, b.actual_win_rate * 100))}%"></div></div>
            <span class="mono" style="text-align:right;color:var(--text-secondary);">${(b.predicted_prob * 100).toFixed(1)}% → <strong>${(b.actual_win_rate * 100).toFixed(1)}%</strong></span>
          </div>`)
        .join('');
      const lgEv = info.ev_line_shopping || {};
      return `
        <div class="surface-panel">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <h3 class="panel-heading">${escapeHtml(lg.toUpperCase())} Out-of-Sample Calibration</h3>
            <span class="mono" style="font-size:0.75rem;color:var(--text-muted);">${info.oos_games} OOS games</span>
          </div>
          <div class="cal-metrics">
            <div class="stat-col"><span>Brier Score</span><strong>${info.brier_score.toFixed(4)}</strong></div>
            <div class="stat-col"><span>Log-Loss</span><strong>${info.log_loss.toFixed(4)}</strong></div>
            <div class="stat-col"><span>+EV CLV</span><strong class="${lgEv.avg_clv_pct >= 0 ? 'positive' : 'negative'}">${pct(lgEv.avg_clv_pct)}</strong></div>
            <div class="stat-col"><span>Combined P&amp;L</span><strong class="${info.combined_net_pnl >= 0 ? 'positive' : 'negative'}">${info.combined_net_pnl >= 0 ? '+' : '-'}$${formatMoney(Math.abs(info.combined_net_pnl))}</strong></div>
          </div>
          <h4 class="sub-heading">Probability Reliability (Pred → Actual)</h4>
          <div>${binsHtml}</div>
        </div>`;
    })
    .join('');

  container.innerHTML = `
    <div class="surface-panel bt-full-span">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
        <div>
          <h3 class="panel-heading">Traditional DK/FD Strategy Profitability Scorecard</h3>
          <span class="field-hint">${comb.oos_games || 0} strict chronological OOS games · Zero lookahead verified: ${report.chronological_verified ? 'YES' : 'NO'}</span>
        </div>
        <div class="mono" style="font-size:1rem;">
          Combined OOS P&amp;L: <strong class="${comb.combined_net_pnl >= 0 ? 'positive' : 'negative'}">${comb.combined_net_pnl >= 0 ? '+' : '-'}$${formatMoney(Math.abs(comb.combined_net_pnl || 0))} (${pct(comb.combined_roi_pct)})</strong>
        </div>
      </div>

      <div class="scorecard-grid">
        <div class="scorecard-item">
          <div class="pick-header"><strong>Cross-Book Arbitrage</strong>${vBadge(verdicts.arbitrage)}</div>
          <div class="kv-list">
            <div class="kv-row"><span>Executed (Rounded-Safe)</span><strong>${arb.arbs_executed_rounded} / ${arb.arbs_found}</strong></div>
            <div class="kv-row"><span>Realized ROI / P&amp;L</span><strong class="positive">${pct(arb.realized_roi_pct)} (+$${formatMoney(arb.rounded_net_pnl)})</strong></div>
            <div class="kv-row"><span>Min Trade P&amp;L / Max DD</span><strong class="positive">+$${Number(arb.min_trade_pnl || 0).toFixed(2)} / ${(arb.max_drawdown_pct * 100).toFixed(1)}%</strong></div>
          </div>
        </div>

        <div class="scorecard-item">
          <div class="pick-header"><strong>Bonus Bet Converter</strong>${vBadge(verdicts.bonus_bet_conversion)}</div>
          <div class="kv-list">
            <div class="kv-row"><span>Conversions Verified</span><strong>${promo.bonus_conversions_count} ($${formatMoney(promo.bonus_face_total)})</strong></div>
            <div class="kv-row"><span>Realized Retention</span><strong class="positive">${(promo.bonus_realized_retention_pct * 100).toFixed(2)}%</strong></div>
            <div class="kv-row"><span>Max Outcome Drift</span><strong class="positive">$${Number(promo.bonus_max_outcome_drift || 0).toFixed(3)}</strong></div>
          </div>
        </div>

        <div class="scorecard-item">
          <div class="pick-header"><strong>Profit Boosts (+25% / +50%)</strong>${vBadge(verdicts.profit_boost)}</div>
          <div class="kv-list">
            <div class="kv-row"><span>+25% Boost ROI</span><strong class="positive">${pct(promo.boost_25_roi_pct)} (+$${formatMoney(promo.boost_25_net_pnl)})</strong></div>
            <div class="kv-row"><span>+50% Boost ROI</span><strong class="positive">${pct(promo.boost_50_roi_pct)} (+$${formatMoney(promo.boost_50_net_pnl)})</strong></div>
            <div class="kv-row"><span>Lift vs Unboosted</span><strong class="positive">${pct(promo.boost_25_roi_lift_pct)}</strong></div>
          </div>
        </div>

        <div class="scorecard-item">
          <div class="pick-header"><strong>Steam / Off-Market Lag</strong>${vBadge(verdicts.steam_lag)}</div>
          <div class="kv-list">
            <div class="kv-row"><span>Qualifying Lag Plays</span><strong>${steam.total_bets} (${(steam.hit_rate * 100).toFixed(1)}% win)</strong></div>
            <div class="kv-row"><span>Realized ROI / P&amp;L</span><strong class="${steam.roi_pct >= 0 ? 'positive' : 'negative'}">${pct(steam.roi_pct)} ($${formatMoney(steam.net_pnl)})</strong></div>
            <div class="kv-row"><span>Avg CLV / Sharpe</span><strong class="positive">${pct(steam.avg_clv_pct)} / ${steam.sharpe_ratio ?? '—'}</strong></div>
          </div>
        </div>

        <div class="scorecard-item">
          <div class="pick-header"><strong>+EV Line Shopping</strong>${vBadge(verdicts.ev_line_shopping)}</div>
          <div class="kv-list">
            <div class="kv-row"><span>Moneyline +EV ROI</span><strong class="${(ev.by_market?.Moneyline?.roi_pct ?? 0) >= 0 ? 'positive' : 'negative'}">${pct(ev.by_market?.Moneyline?.roi_pct ?? 0)} (${ev.by_market?.Moneyline?.bets ?? 0} bets)</strong></div>
            <div class="kv-row"><span>Avg CLV (All Markets)</span><strong class="${ev.avg_clv_pct >= 0 ? 'positive' : 'negative'}">${pct(ev.avg_clv_pct)}</strong></div>
            <div class="kv-row"><span>Low-Hold Savings</span><strong class="positive">${pct(promo.low_hold_savings_vs_vig_pct)} (${promo.low_hold_pairs_found} pairs)</strong></div>
          </div>
        </div>

        <div class="scorecard-item">
          <div class="pick-header"><strong>Middles &amp; Key Numbers</strong>${vBadge(verdicts.middles)}</div>
          <div class="kv-list">
            <div class="kv-row"><span>Model-Filtered Middles</span><strong>${mid.total_middles} (${mid.key_number_middles} NFL key #)</strong></div>
            <div class="kv-row"><span>Filtered vs Blind ROI</span><strong class="${mid.blind_vs_filtered_roi_lift_pct >= 0 ? 'positive' : 'negative'}">${pct(mid.roi_pct)} vs ${pct(mid.blind_roi_pct)}</strong></div>
            <div class="kv-row"><span>Model Filter Lift</span><strong class="${mid.blind_vs_filtered_roi_lift_pct >= 0 ? 'positive' : 'negative'}">${pct(mid.blind_vs_filtered_roi_lift_pct)}</strong></div>
          </div>
        </div>
      </div>

      <h4 class="sub-heading">3-Regime Stress Sensitivity Matrix</h4>
      <div class="table-container" style="margin-top:0.4rem;">
        <table class="data-table">
          <thead>
            <tr>
              <th>Market Regime</th>
              <th class="num-col">Combined ROI / P&amp;L</th>
              <th class="num-col">Arbitrage</th>
              <th class="num-col">Bonus Ret.</th>
              <th class="num-col">+25% Boost</th>
              <th class="num-col">Steam Lag</th>
              <th class="num-col">Middles</th>
            </tr>
          </thead>
          <tbody>${stressRows}</tbody>
        </table>
      </div>
    </div>
    ${leagueCards}
  `;
}

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
  const seq = nextSeq('backtest');
  const reqMode = state.traditionalMode;
  container.innerHTML = '<div class="empty-state"><p>Loading walk-forward verification metrics...</p></div>';
  try {
    const endpoint = reqMode ? '/api/sportsbook/backtest-metrics' : '/api/backtest-metrics';
    const data = await apiJson(endpoint);
    if (!isLatestSeq('backtest', seq) || state.traditionalMode !== reqMode) return;
    state.backtestLoaded = true;
    state.backtestModeLoaded = reqMode;
    if (reqMode) renderSportsbookBacktest(data);
    else renderBacktestData(data);
  } catch (err) {
    if (isLatestSeq('backtest', seq) && !isSilentError(err)) {
      container.innerHTML = `<div class="empty-state"><p>Error loading backtest: ${escapeHtml(errText(err))}</p></div>`;
    }
  }
}

document.getElementById('run-backtest-btn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  if (btn.disabled) return;
  const reqMode = state.traditionalMode;
  const seq = nextSeq('backtest');
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.textContent = 'Running Backtest...';
  try {
    const url = reqMode
      ? `/api/sportsbook/backtest?sport=all&arb_rounding=${parseFloat(document.getElementById('setting-arb-rounding').value) || 1}`
      : `/api/backtest?sport=all&fee=${getFee()}`;
    const data = await apiJson(url, { method: 'POST' });
    if (isLatestSeq('backtest', seq) && state.traditionalMode === reqMode) {
      state.backtestLoaded = true;
      state.backtestModeLoaded = reqMode;
      if (reqMode) renderSportsbookBacktest(data);
      else renderBacktestData(data);
    }
    showBanner(
      reqMode
        ? 'Traditional DK/FD walk-forward strategy verification completed.'
        : 'Chronological walk-forward backtest completed.'
    );
  } catch (err) {
    reportError('Backtest failed', err);
  } finally {
    btn.disabled = false;
    btn.removeAttribute('aria-busy');
    btn.textContent = state.traditionalMode ? 'Re-run Sportsbook Backtest' : 'Re-run Backtest';
  }
});

document.getElementById('train-models-btn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  if (btn.disabled) return;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.textContent = 'Training...';
  try {
    await apiJson('/api/train?sport=all', { method: 'POST' });
    await loadPicks();
    showBanner('NFL, NBA, and Player Props ensemble models retrained and calibrated.');
  } catch (err) {
    reportError('Model training failed', err);
  } finally {
    btn.disabled = false;
    btn.removeAttribute('aria-busy');
    btn.textContent = 'Retrain Models';
  }
});

document.getElementById('sync-btn').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  if (btn.disabled) return;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.textContent = 'Syncing & Training...';
  try {
    await apiJson('/api/sync?sport=all', { method: 'POST' });
    await loadServerConfig();
    await loadPicks();
    showBanner('Historical schedules, player logs, live feeds, and models synced.');
  } catch (err) {
    reportError('Sync error', err);
  } finally {
    btn.disabled = false;
    btn.removeAttribute('aria-busy');
    btn.textContent = 'Sync Data & Retrain';
  }
});

// --- Guide Markdown Renderer ---
async function loadGuide() {
  try {
    const data = await apiJson('/api/guide');
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
    if (!isSilentError(err)) {
      document.getElementById('guide-content').innerHTML = `<p>Failed loading guide: ${escapeHtml(errText(err))}</p>`;
    }
  }
}

// --- Fast Live Feed Refresh ---
async function refreshLiveLines(silent = false) {
  const btn = document.getElementById('refresh-live-btn');
  if (btn.disabled) return;
  const icon = document.getElementById('refresh-icon');
  const label = document.getElementById('refresh-btn-label');
  const dateVal = document.getElementById('filter-date').value;
  const qs = new URLSearchParams({ sport: state.sport });
  if (dateVal) qs.set('date', dateVal);

  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  icon.classList.add('spin');
  label.textContent = 'Refreshing';

  try {
    await apiJson(`/api/refresh-live?${qs.toString()}`, { method: 'POST' });
    lastLiveRefreshAt = Date.now();
    await loadServerConfig();
    await loadPicks();
    if (!document.getElementById('panel-ledger').hidden) await loadLedger();
    if (!silent) {
      showBanner('Live scoreboards, box scores, injuries, and lines refreshed.');
    }
  } catch (err) {
    if (!silent) reportError('Live refresh error', err);
  } finally {
    btn.disabled = false;
    btn.removeAttribute('aria-busy');
    icon.classList.remove('spin');
    label.textContent = 'Refresh';
  }
}

document.getElementById('refresh-live-btn').addEventListener('click', () => refreshLiveLines(false));

// --- Boot Lifecycle ---
async function bootWorkspace() {
  await loadServerConfig();
  await loadPicks();
  if (state.traditionalMode) runSportsbookAnalyzer();
  else runCustomCalculator();
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

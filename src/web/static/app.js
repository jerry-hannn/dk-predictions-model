// DraftKings Predictions (DKeX) Pre-Game & Live In-Play Dashboard Client

const state = {
  picks: [],
  trades: [],
  backtestLoaded: false,
  guideLoaded: false,
  gridVisibleForPolling: true,
};

let bannerTimer = null;
let autoRefreshTimer = null;

// --- Tabs Setup ---
const tablist = document.querySelector('.tablist');
const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
const panels = tabs.map((t) => document.getElementById(t.id.replace('tab-', 'panel-')));

function selectTab(selected) {
  tabs.forEach((tab, idx) => {
    const isSelected = tab === selected;
    tab.setAttribute('aria-selected', String(isSelected));
    tab.toggleAttribute('focusgroupstart', isSelected);
    tab.tabIndex = isSelected ? 0 : -1;
    panels[idx].hidden = !isSelected;
  });

  selected.scrollIntoView({ block: 'nearest', inline: 'nearest', container: 'nearest' });
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

// Roving arrow-key fallback for browsers without native `focusgroup`
if (!Object.hasOwn(HTMLElement.prototype, 'focusGroup')) {
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
}

// --- Modal Dialog Light-Dismiss Fallback (`light-dismiss-a-dialog` guide) ---
const settleDialog = document.getElementById('settle-dialog');
if (!('closedBy' in HTMLDialogElement.prototype)) {
  settleDialog.addEventListener('click', (event) => {
    if (event.target !== settleDialog) return;
    const rect = settleDialog.getBoundingClientRect();
    const isDialogContent =
      rect.top <= event.clientY &&
      event.clientY <= rect.top + rect.height &&
      rect.left <= event.clientX &&
      event.clientX <= rect.left + rect.width;
    if (isDialogContent) return;
    settleDialog.close();
  });
}
document.getElementById('close-dialog-btn').addEventListener('click', () => settleDialog.close());
document.getElementById('cancel-dialog-btn').addEventListener('click', () => settleDialog.close());

document.getElementById('settle-status').addEventListener('change', (e) => {
  document.getElementById('sold-fields').hidden = e.target.value !== 'SOLD';
});

// --- Pause Live Polling When Off-Screen (`efficient-background-processing` guide) ---
const picksGridEl = document.getElementById('picks-grid');
picksGridEl.addEventListener('contentvisibilityautostatechange', (event) => {
  state.gridVisibleForPolling = !event.skipped;
  syncAutoRefreshTimer();
});

if (!('contentVisibility' in document.documentElement.style) && 'IntersectionObserver' in window) {
  const gridObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        state.gridVisibleForPolling = entry.isIntersecting;
      });
      syncAutoRefreshTimer();
    },
    { rootMargin: '200px' }
  );
  gridObserver.observe(picksGridEl);
}

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

// --- Helpers ---
function getBankroll() {
  const raw = parseFloat(document.getElementById('global-bankroll').value);
  return Number.isFinite(raw) && raw > 0 ? raw : 25.0;
}

function getFee() {
  const raw = parseFloat(document.getElementById('global-fee').value);
  return Number.isFinite(raw) && raw >= 0 ? raw : 0.015;
}

function showBanner(msg, isError = false) {
  const el = document.getElementById('status-banner');
  el.textContent = msg;
  el.classList.toggle('error', isError);
  el.hidden = false;
  if (bannerTimer) clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => {
    el.hidden = true;
  }, 5500);
}

function updateOddsApiPill(connected, cachedCount = 0, quota = null) {
  const pill = document.getElementById('odds-api-pill');
  if (connected) {
    const rem = quota && quota.requests_remaining != null ? ` • ${quota.requests_remaining} cr` : '';
    pill.textContent = cachedCount > 0
      ? `Sharp Odds API: Connected (${cachedCount} Live${rem})`
      : `Sharp Odds API: Connected${rem}`;
    pill.classList.add('connected');
  } else {
    pill.textContent = 'ESPN / DraftKings Public Feed';
    pill.classList.remove('connected');
  }
}

function tierBadgeClass(tier) {
  if (tier === 'A+') return 'tier-Aplus-badge';
  if (tier === 'A') return 'tier-A-badge';
  if (tier === 'B') return 'tier-B-badge';
  return 'tier-PASS-badge';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str ?? '';
  return div.innerHTML;
}

function formatAgeBadge(ageSec) {
  if (ageSec == null) return '';
  const s = Math.max(0, Math.round(Number(ageSec)));
  if (s < 60) return ` • ${s}s ago`;
  return ` • ${Math.round(s / 60)}m ago`;
}

// --- KPI Rendering ---
function updateKPIs(ledgerSummary, picksList) {
  if (ledgerSummary) {
    document.getElementById('kpi-bankroll').textContent = `$${ledgerSummary.current_bankroll.toFixed(2)}`;
    const pnlSign = ledgerSummary.realized_pnl >= 0 ? '+' : '';
    document.getElementById('kpi-pnl').textContent = `Realized P&L: ${pnlSign}$${ledgerSummary.realized_pnl.toFixed(2)}`;
    document.getElementById('kpi-hitrate').textContent =
      ledgerSummary.hit_rate !== null ? `${(ledgerSummary.hit_rate * 100).toFixed(1)}%` : '—';
    document.getElementById('kpi-exp-hitrate').textContent =
      ledgerSummary.expected_win_rate !== null
        ? `Expected: ${(ledgerSummary.expected_win_rate * 100).toFixed(1)}%`
        : 'Log trades to track';
    const roiSign = ledgerSummary.roi_pct >= 0 ? '+' : '';
    document.getElementById('kpi-roi').textContent = `${roiSign}${(ledgerSummary.roi_pct * 100).toFixed(1)}%`;
    document.getElementById('kpi-exposure').textContent = `Open Exposure: $${ledgerSummary.open_exposure.toFixed(2)}`;
  }
  if (picksList) {
    const locks = picksList.filter((p) => p.confidence_tier === 'A+' || p.confidence_tier === 'A').length;
    const qualifying = picksList.filter((p) => p.confidence_tier !== 'PASS').length;
    const liveCnt = picksList.filter((p) => p.game_state === 'LIVE' && p.confidence_tier !== 'PASS').length;
    document.getElementById('kpi-locks').textContent = String(locks);
    document.getElementById('kpi-total-picks').textContent =
      liveCnt > 0
        ? `${qualifying} qualifying (${liveCnt} LIVE in-play)`
        : `${qualifying} qualifying high-confidence plays`;
  }
}

// --- Load Config (.env Bankroll, Fee, Sharp Odds API Status & Quota) ---
async function loadServerConfig() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) return;
    const cfg = await res.json();
    document.getElementById('global-bankroll').value = Number(cfg.bankroll).toFixed(2);

    const feeSelect = document.getElementById('global-fee');
    const matchingOpt = Array.from(feeSelect.options).find(
      (o) => Math.abs(parseFloat(o.value) - Number(cfg.default_fee_dollars)) < 1e-4
    );
    if (matchingOpt) feeSelect.value = matchingOpt.value;

    const minProbPct = Math.round(Number(cfg.min_win_prob) * 100);
    document.getElementById('filter-min-prob').value = String(minProbPct);
    document.getElementById('min-prob-val').textContent = `${minProbPct}%`;

    const minEdgePct = Number((Number(cfg.min_net_edge) * 100).toFixed(1));
    document.getElementById('filter-min-edge').value = String(minEdgePct);
    document.getElementById('min-edge-val').textContent = `+${minEdgePct.toFixed(1)}%`;

    updateOddsApiPill(
      Boolean(cfg.odds_api_connected),
      Number(cfg.sharp_events_cached || 0),
      cfg.odds_quota || null
    );
  } catch (err) {
    console.error('Failed loading /api/config:', err);
  }
}

async function persistServerConfig() {
  try {
    await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bankroll: getBankroll(),
        default_fee_dollars: getFee(),
      }),
    });
  } catch (err) {
    console.error('Failed saving /api/config:', err);
  }
}

// --- Load & Render Picks ---
async function loadPicks() {
  const mode = document.getElementById('filter-mode').value;
  const sport = document.getElementById('filter-sport').value;
  const market = document.getElementById('filter-market').value;
  const minProb = parseFloat(document.getElementById('filter-min-prob').value) / 100;
  const minEdge = parseFloat(document.getElementById('filter-min-edge').value) / 100;
  const showAll = document.getElementById('filter-show-all').checked;

  const params = new URLSearchParams({
    mode,
    sport,
    market,
    min_prob: String(minProb),
    min_edge: String(minEdge),
    fee: String(getFee()),
    bankroll: String(getBankroll()),
    include_passes: String(showAll),
  });

  const grid = document.getElementById('picks-grid');
  grid.innerHTML = '<p class="muted">Evaluating pre-game &amp; live in-play distributions...</p>';

  try {
    const res = await fetch(`/api/picks?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.picks = data.picks || [];
    if (data.odds_quota) {
      updateOddsApiPill(Boolean(data.odds_api_connected), undefined, data.odds_quota);
    }
    updateKPIs(data.ledger_summary, state.picks);
    renderPicks();
  } catch (err) {
    grid.innerHTML = `<div class="card"><h3>Error Loading Picks</h3><p class="muted">${escapeHtml(String(err))}</p></div>`;
  }
}

function buildPickCardInnerHtml(p, idx) {
  const isLive = p.game_state === 'LIVE';
  const edgeClass = p.net_edge >= 0 ? 'positive' : 'negative';
  const edgeSign = p.net_edge >= 0 ? '+' : '';
  const driversHtml = (p.drivers || [])
    .map((d) => `<li>${escapeHtml(d)}</li>`)
    .join('');
  const injHtml = p.injury_notes
    ? `<p class="injury-note">Injuries: ${escapeHtml(p.injury_notes)}</p>`
    : '';
  const liveTagHtml = isLive
    ? `<span class="live-tag">LIVE • ${escapeHtml(p.status_detail || '')}${p.score_display ? ` • ${escapeHtml(p.score_display)}` : ''}</span>`
    : '';
  const lineMoveHtml = p.line_movement
    ? `<span class="line-move-pill">${escapeHtml(p.line_movement)}</span>`
    : '';
  const srcAgeHtml = `<span>Source: ${escapeHtml(p.line_source || 'ESPN/DK')}${formatAgeBadge(p.odds_age_seconds)}</span>`;

  const qtyForLog = Math.max(1, p.recommended_contracts || 1);
  const outlayForLog = (qtyForLog * p.all_in_cost).toFixed(2);
  const actionLabel =
    p.confidence_tier === 'PASS'
      ? `Log Anyway (${qtyForLog}x)`
      : `Log ${escapeHtml(p.recommended_side)} (${p.recommended_contracts}x = $${p.recommended_outlay.toFixed(2)})`;
  const btnClass = p.confidence_tier === 'PASS' ? 'btn btn-secondary btn-sm log-trade-btn' : 'btn btn-primary btn-sm log-trade-btn';

  return `
    <div class="pick-top">
      <div class="tag-row">
        <span class="league-tag">${escapeHtml(p.sport)}</span>
        ${liveTagHtml}
        <span class="market-tag">${escapeHtml(p.market_type)}</span>
      </div>
      <span class="tier-badge ${tierBadgeClass(p.confidence_tier)}">
        ${escapeHtml(p.confidence_tier)} • ${p.confidence_score}
      </span>
    </div>

    <div>
      <h3 class="pick-title">${escapeHtml(p.selection_label)}</h3>
      <p class="pick-matchup">${escapeHtml(p.event_name)} • Question: <em>${escapeHtml(p.contract_question || p.selection_label)}</em></p>
    </div>

    <div class="line-meta-bar">
      ${srcAgeHtml}
      ${lineMoveHtml}
    </div>

    <div class="pick-metrics">
      <div class="metric-cell">
        <span>${escapeHtml(p.recommended_side)} Win %</span>
        <strong>${(p.model_prob * 100).toFixed(1)}%</strong>
      </div>
      <div class="metric-cell">
        <span>Net Edge (After Fee)</span>
        <strong class="${edgeClass}">${edgeSign}${(p.net_edge * 100).toFixed(1)}%</strong>
      </div>
      <div class="metric-cell">
        <span>Max Limit Buy</span>
        <strong>${(p.fair_entry_max_price * 100).toFixed(0)}¢</strong>
      </div>
    </div>

    <div class="price-checker-row">
      <div class="price-checker-left">
        <label for="dk-side-${idx}">Side &amp; ¢:</label>
        <select id="dk-side-${idx}" class="dk-side-select" data-idx="${idx}">
          <option value="YES" ${p.recommended_side === 'YES' ? 'selected' : ''}>YES</option>
          <option value="NO" ${p.recommended_side === 'NO' ? 'selected' : ''}>NO</option>
        </select>
        <input
          id="dk-px-${idx}"
          class="dk-cents-input"
          type="number"
          min="1"
          max="99"
          step="0.5"
          value="${Number((p.contract_price * 100).toFixed(1))}"
          data-idx="${idx}"
        >
        <button type="button" class="btn btn-secondary btn-sm check-px-btn" data-idx="${idx}">Check ¢</button>
      </div>
      <span class="muted">Exit: <strong>${(p.fair_midgame_exit_price * 100).toFixed(0)}¢</strong></span>
    </div>

    <ul class="drivers-list">${driversHtml}</ul>
    ${injHtml}

    <div class="card-footer">
      <span>${isLive ? 'In-Play ' : ''}Quarter-Kelly: <strong>${p.recommended_contracts}</strong> sh ($${p.recommended_outlay.toFixed(2)} / $${outlayForLog} min)</span>
      <button type="button" class="${btnClass}" data-idx="${idx}">
        ${actionLabel}
      </button>
    </div>
  `;
}

function cardCssClass(p) {
  const classes = ['pick-card'];
  if (p.confidence_tier === 'A+') classes.push('tier-Aplus');
  if (p.game_state === 'LIVE') classes.push('is-live-card');
  return classes.join(' ');
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
    const resp = await fetch('/api/evaluate-contract', {
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
      cardEl.className = cardCssClass(pick);
      cardEl.innerHTML = buildPickCardInnerHtml(pick, idx);
      bindCardEvents(cardEl);
    }
    updateKPIs(null, state.picks);
  } catch (err) {
    showBanner(`Could not re-evaluate contract: ${err}`, true);
  }
}

function bindCardEvents(container) {
  container.querySelectorAll('.check-px-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      reevaluateSingleCard(parseInt(e.currentTarget.dataset.idx, 10));
    });
  });

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
    btn.addEventListener('click', async (e) => {
      const i = parseInt(e.currentTarget.dataset.idx, 10);
      const pick = state.picks[i];
      const qty = Math.max(1, pick.recommended_contracts || 1);
      try {
        const resp = await fetch('/api/ledger', {
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
          `Logged ${qty}x ${pick.recommended_side} on "${pick.selection_label}" @ ${(pick.contract_price * 100).toFixed(1)}¢ (Total $${(qty * pick.all_in_cost).toFixed(2)})`
        );
        const led = await (await fetch(`/api/ledger?bankroll=${getBankroll()}`)).json();
        updateKPIs(led.summary, state.picks);
      } catch (err) {
        showBanner(`Failed logging trade: ${err}`, true);
      }
    });
  });
}

function renderPicks() {
  const grid = document.getElementById('picks-grid');
  if (!state.picks.length) {
    const mode = document.getElementById('filter-mode').value;
    const hint = mode === 'live'
      ? 'No NFL/NBA games are currently in-play with qualifying edge, or switch Slate Mode to "All Active".'
      : 'Try lowering Min Win Prob to 57%, clicking "Refresh Live Lines", or checking "Show PASS markets" to enter live DraftKings contract prices.';
    grid.innerHTML = `
      <div class="card">
        <h3>No Contracts Match Current Filters</h3>
        <p class="muted">${escapeHtml(hint)}</p>
      </div>`;
    return;
  }

  grid.innerHTML = state.picks
    .map((p, idx) => `<article class="${cardCssClass(p)}" data-idx="${idx}">${buildPickCardInnerHtml(p, idx)}</article>`)
    .join('');

  bindCardEvents(grid);
}

// --- Standalone Contract Analyzer ---
async function runCustomCalculator(e) {
  if (e) e.preventDefault();
  const side = document.getElementById('calc-side').value;
  const yesProb = Math.min(99.5, Math.max(0.5, parseFloat(document.getElementById('calc-prob').value) || 65)) / 100;
  const price = Math.min(99, Math.max(1, parseFloat(document.getElementById('calc-price').value) || 55)) / 100;
  const isLive = document.getElementById('calc-is-live').checked;

  try {
    const resp = await fetch('/api/evaluate-contract', {
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
    const edgeCls = ev.net_edge >= 0 ? 'positive' : 'negative';
    document.getElementById('calc-output-body').innerHTML = `
      <div class="metric-cell"><span>Decision Tier</span><strong class="${edgeCls}">${ev.confidence_tier} (${ev.confidence_score}/100)</strong></div>
      <div class="metric-cell"><span>Chosen Side Win %</span><strong>${(ev.model_prob * 100).toFixed(1)}%</strong></div>
      <div class="metric-cell"><span>All-In Cost (w/ Fee)</span><strong>${(ev.all_in_cost * 100).toFixed(2)}¢ (${ev.american_equivalent > 0 ? '+' : ''}${ev.american_equivalent})</strong></div>
      <div class="metric-cell"><span>Break-Even Win %</span><strong>${(ev.breakeven_prob * 100).toFixed(2)}%</strong></div>
      <div class="metric-cell"><span>Fee-Adjusted Net Edge</span><strong class="${edgeCls}">${ev.net_edge >= 0 ? '+' : ''}${(ev.net_edge * 100).toFixed(2)}%</strong></div>
      <div class="metric-cell"><span>Expected ROI</span><strong class="${edgeCls}">${ev.roi_pct >= 0 ? '+' : ''}${(ev.roi_pct * 100).toFixed(2)}%</strong></div>
      <div class="metric-cell"><span>Max Limit Entry Price</span><strong>${(ev.fair_entry_max_price * 100).toFixed(1)}¢</strong></div>
      <div class="metric-cell"><span>${isLive ? 'In-Play ' : ''}Quarter-Kelly (${`$${getBankroll().toFixed(2)}`})</span><strong>${ev.recommended_contracts} sh ($${ev.recommended_outlay.toFixed(2)})</strong></div>
    `;
  } catch (err) {
    showBanner(`Analyzer error: ${err}`, true);
  }
}

// --- Trade Ledger & Live Position Guidance ---
async function loadLedger() {
  try {
    const resp = await fetch(`/api/ledger?bankroll=${getBankroll()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.trades = data.trades || [];
    updateKPIs(data.summary, state.picks);

    const tbody = document.getElementById('ledger-tbody');
    if (!state.trades.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="muted">No DKeX positions logged yet. Click "Log YES/NO" on any Pick card to track pre-game &amp; live positions.</td></tr>';
      return;
    }

    tbody.innerHTML = state.trades
      .map((t) => {
        const pnlCls = t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : '';
        const pnlStr = t.status === 'OPEN' ? '—' : `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}`;
        const probTxt =
          t.status === 'OPEN' && t.live_model_prob != null
            ? `${(t.model_prob * 100).toFixed(0)}% → <strong>${(t.live_model_prob * 100).toFixed(1)}%</strong>`
            : `${(t.model_prob * 100).toFixed(1)}%`;
        const guidanceTxt =
          t.status === 'OPEN' && t.live_guidance
            ? escapeHtml(t.live_guidance)
            : `Exit Target ${(t.fair_exit_price * 100).toFixed(0)}¢`;
        return `
          <tr>
            <th scope="row">${escapeHtml(t.selection_label)}</th>
            <td>${escapeHtml(t.sport)}</td>
            <td><strong>${escapeHtml(t.side)}</strong></td>
            <td>${t.contracts} @ ${(t.entry_price * 100).toFixed(1)}¢</td>
            <td>$${t.total_cost.toFixed(2)}</td>
            <td>${probTxt}</td>
            <td>${guidanceTxt}</td>
            <td>${escapeHtml(t.status)}</td>
            <td class="${pnlCls}"><strong>${pnlStr}</strong></td>
            <td>
              <div class="ledger-actions-cell">
                <button type="button" class="btn btn-secondary btn-sm settle-open-btn" data-id="${t.id}">Settle</button>
                <button type="button" class="btn btn-danger btn-sm delete-trade-btn" data-id="${t.id}" aria-label="Delete trade">Delete</button>
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
          `${trade.contracts}x ${trade.side} — ${trade.selection_label} (Cost $${trade.total_cost.toFixed(2)})`;
        document.getElementById('settle-status').value = 'WON';
        document.getElementById('sold-fields').hidden = true;
        document.getElementById('settle-closing-price').value = '';
        settleDialog.showModal();
      });
    });

    tbody.querySelectorAll('.delete-trade-btn').forEach((btn) => {
      btn.addEventListener('click', async (e) => {
        const id = parseInt(e.currentTarget.dataset.id, 10);
        const delResp = await fetch(`/api/ledger/${id}`, { method: 'DELETE' });
        if (delResp.ok) {
          showBanner(`Deleted trade #${id}.`);
          loadLedger();
        }
      });
    });
  } catch (err) {
    showBanner(`Failed loading ledger: ${err}`, true);
  }
}

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

  const resp = await fetch(`/api/ledger/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (resp.ok) {
    settleDialog.close();
    showBanner(`Trade #${id} settled as ${status}.`);
    loadLedger();
  } else {
    showBanner('Could not save settlement—verify sale price is between 1¢ and 99¢.', true);
  }
});

// --- Backtest Metrics ---
async function loadBacktest() {
  const container = document.getElementById('backtest-container');
  container.innerHTML = '<p class="muted">Loading out-of-sample walk-forward evaluation...</p>';
  try {
    const resp = await fetch('/api/backtest-metrics');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.backtestLoaded = true;

    container.innerHTML = Object.entries(data)
      .map(([lg, info]) => {
        const sim = info.simulation;
        const binsHtml = (info.calibration_bins || [])
          .map(
            (b) =>
              `<li><strong>${escapeHtml(b.bin_label)}</strong> (${b.count} games): Predicted ${(b.predicted_prob * 100).toFixed(1)}% vs Actual ${(b.actual_win_rate * 100).toFixed(1)}%</li>`
          )
          .join('');
        return `
          <div class="card">
            <h3>${escapeHtml(lg.toUpperCase())} Walk-Forward Results (${info.oos_games} OOS Games)</h3>
            <div class="calc-metrics-grid">
              <div class="metric-cell"><span>Brier Score</span><strong>${info.brier_score.toFixed(4)}</strong></div>
              <div class="metric-cell"><span>Log-Loss</span><strong>${info.log_loss.toFixed(4)}</strong></div>
              <div class="metric-cell"><span>Qualifying Trades</span><strong>${sim.total_bets}</strong></div>
              <div class="metric-cell"><span>Actual Hit Rate</span><strong class="positive">${(sim.hit_rate * 100).toFixed(1)}%</strong></div>
              <div class="metric-cell"><span>Fee-Adjusted Net ROI</span><strong class="${sim.roi_pct >= 0 ? 'positive' : 'negative'}">${sim.roi_pct >= 0 ? '+' : ''}${(sim.roi_pct * 100).toFixed(2)}%</strong></div>
              <div class="metric-cell"><span>Max Drawdown</span><strong>${(sim.max_drawdown_pct * 100).toFixed(1)}%</strong></div>
            </div>
            <h4>Probability Calibration Reliability</h4>
            <ul class="drivers-list">${binsHtml}</ul>
          </div>
        `;
      })
      .join('');
  } catch (err) {
    container.innerHTML = `<div class="card"><p class="muted">Error loading backtest metrics: ${escapeHtml(String(err))}</p></div>`;
  }
}

// --- Guide Loader ---
async function loadGuide() {
  try {
    const resp = await fetch('/api/guide');
    const data = await resp.json();
    state.guideLoaded = true;
    const md = data.markdown || '';
    const blocks = md.split(/```([\s\S]*?)```/g);
    let html = '';
    for (let i = 0; i < blocks.length; i++) {
      if (i % 2 === 1) {
        const code = blocks[i].replace(/^[a-z]+\n/, '');
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
    document.getElementById('guide-content').innerHTML = `<p class="muted">Failed loading guide: ${escapeHtml(String(err))}</p>`;
  }
}

// --- Fast Refresh Live Lines vs Full Sync ---
async function refreshLiveLines(silent = false) {
  const btn = document.getElementById('refresh-live-btn');
  const sport = document.getElementById('filter-sport').value;
  btn.disabled = true;
  btn.textContent = 'Refreshing...';
  try {
    const resp = await fetch(`/api/refresh-live?sport=${encodeURIComponent(sport)}`, { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    await loadServerConfig();
    await loadPicks();
    if (!silent) {
      showBanner('Live scoreboards, box scores, injuries, & DraftKings/Sharp lines updated!');
    }
  } catch (err) {
    if (!silent) showBanner(`Live refresh error: ${err}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Refresh Live Lines';
  }
}

// --- Filter, Config & Sync Listeners ---
document.getElementById('filters-form').addEventListener('submit', (e) => e.preventDefault());

document.getElementById('filter-min-prob').addEventListener('input', (e) => {
  document.getElementById('min-prob-val').textContent = `${e.target.value}%`;
});
document.getElementById('filter-min-edge').addEventListener('input', (e) => {
  document.getElementById('min-edge-val').textContent = `+${parseFloat(e.target.value).toFixed(1)}%`;
});

['filter-mode', 'filter-sport', 'filter-market', 'filter-min-prob', 'filter-min-edge', 'filter-show-all'].forEach((id) => {
  document.getElementById(id).addEventListener('change', loadPicks);
});

document.getElementById('auto-refresh-live').addEventListener('change', syncAutoRefreshTimer);

['global-bankroll', 'global-fee'].forEach((id) => {
  document.getElementById(id).addEventListener('change', async () => {
    await persistServerConfig();
    await loadPicks();
    if (!document.getElementById('panel-calculator').hidden) runCustomCalculator();
    if (!document.getElementById('panel-ledger').hidden) loadLedger();
  });
});

document.getElementById('custom-calc-form').addEventListener('submit', runCustomCalculator);
['calc-side', 'calc-prob', 'calc-price', 'calc-is-live'].forEach((id) => {
  document.getElementById(id).addEventListener('change', () => runCustomCalculator());
});

document.getElementById('refresh-live-btn').addEventListener('click', () => refreshLiveLines(false));

document.getElementById('sync-btn').addEventListener('click', async () => {
  const btn = document.getElementById('sync-btn');
  btn.disabled = true;
  btn.textContent = 'Syncing & Training...';
  try {
    const resp = await fetch('/api/sync', { method: 'POST' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    await loadServerConfig();
    await loadPicks();
    showBanner('Full historical schedules, player logs, live lines, and ensemble models retrained!');
  } catch (err) {
    showBanner(`Sync error: ${err}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Full Sync & Retrain';
  }
});

// --- Boot ---
async function initApp() {
  await loadServerConfig();
  await loadPicks();
  runCustomCalculator();
}

initApp();

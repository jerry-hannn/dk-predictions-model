# DraftKings Predictions (DKeX) — Execution & Hosting Guide

Welcome to the **DKeX NFL & NBA Prediction Workspace**, designed specifically for **DraftKings Predictions** across **pre-game lines** and **live in-play trading**.

Unlike fixed-odds sportsbooks that embed a ~4.5%–8% house vig, DraftKings Predictions is a peer-to-peer binary exchange where **YES / NO** event contracts trade between **`1¢` (`$0.01`) and `99¢` (`$0.99`)** and settle at **`$1.00`** if correct (or `$0.00` if incorrect).

---

## 1. UI-First Daily Workflow

Everything runs directly inside the web interface (`dkmodel` locally or your hosted URL remotely):

1. **Configure Portfolio (`$1,000 · 1.5¢` button in top bar)**:
   - Set your active **Bankroll ($)**, **Exchange Fee Tier** (`1.5¢` standard, `1.0¢` base, `2.0¢` high, or `0.0¢` promo), **Kelly Multiplier** (`0.25` default), and optional **The Odds API Key**.
   - Settings persist automatically in SQLite (`DATA_DIR`), surviving container restarts when hosted remotely.
2. **Scan Markets (`Markets` tab)**:
   - Filter instantaneously by **All / Live / Pre-Game**, **NFL / NBA**, **Games / Props**, or search any team/player.
   - Switch between **Cards** and dense **Table** views.
   - Click **Refresh** (~1–2s) to pull live ESPN scoreboards, active player box scores, injury reports, and sharp sportsbook lines, or enable **Auto-refresh (60s)** inside `Filters` (automatically pauses when the browser tab is backgrounded).
3. **Verify Exact Ask `¢` Before Buying**:
   - Every market card includes an inline **`YES/NO` + `¢`** input box. Type the exact live offer shown on your DraftKings app to re-compute post-fee edge and Quarter-Kelly contract quantity on the spot, or open the **Analyzer** tab for standalone calculations.
4. **Track Open Positions & Cash-Outs (`Ledger` tab)**:
   - Click **Log** on any market card (or **+ Log Position** in the Ledger) to record your entry price and itemized contract fee.
   - During live games, open positions automatically display live win probability shifts and **`HOLD TO $1.00`** vs **`CONSIDER CASH OUT`** guidance.
5. **Model Maintenance (`Calibration` tab)**:
   - Run **Sync Data & Retrain**, **Retrain Models**, or **Re-run Backtest** directly from the Calibration view whenever you want to refresh multi-season historical schedules and verify out-of-sample reliability.

---

## 2. How the Pre-Game & Live Models Price Contracts

### Dual-Gate Conviction Filtering
Every qualifying play must clear **both** post-fee thresholds (`Break-Even % = Ask Price + Per-Contract Fee`):
- **`A+` (Score 85–100)**: Calibrated Win Prob >= 65% and Post-Fee Net Edge >= +4.0% with fresh lines and sharp-market agreement.
- **`A` (Score 72–84)**: Win Prob >= 62% and Post-Fee Net Edge >= +3.0%.
- **`B` (Score 60–71)**: Win Prob >= 60% and Post-Fee Net Edge >= +2.5%.
- **`PASS` (`< 60`)**: Thin or negative post-fee edge, insufficient win rate, or blocked by live endgame latency guardrails.

### Live In-Play Conditioning (`LIVE`)
When a game tips off or kicks off, the engine transitions automatically from pre-game priors to state-space conditioning:
- **Brownian Margin Drift & Shrinking Clock Variance**: Combines live score differential with remaining-clock fundamental drift, NFL possession / red-zone expected points, and late-game NBA foul-variance adjustments.
- **Empirical-Bayes Total Mean Reversion**: Regresses hot first-half shooting bursts 85%–88% toward pre-game efficiency priors, surfacing high-edge **`NO` (Under)** contracts when retail order books overreact.
- **Live Box-Score Player Props**: Conditions on accumulated stats and live minutes while applying **NBA foul-trouble haircuts** and **Q3/Q4 blowout starter-benching haircuts** (`22+` pts in Q3, `16+` pts in Q4).

---

## 3. Execution Rules on DraftKings Predictions

### Rule 1: Trade Dead Balls (Timeouts, Quarter Breaks, Halftime)
Television streams lag arena play by 8–25 seconds. Execute live positions during **Halftime**, **End of Quarter**, **Official Timeouts**, or **NFL Commercial Breaks** when clocks are stopped and order books stabilize.

### Rule 2: Respect the `<90s` Endgame Latency Lock
Inside the final 90 seconds of one-score games (`<= 6` pts NBA, `<= 8` pts NFL), a single whistle swings win probability by 35%+. The model automatically forces **`PASS` (`Endgame latency lock`)**—never chase final-minute coin flips.

### Rule 3: Use Limit Orders (`1¢–2¢` Inside Wide Spreads)
Live bid-ask spreads widen to `3¢–5¢` mid-game. Never sweep asks above **`Max Buy`**. Post **Limit Orders** at or `1¢–2¢` below `Max Buy` during timeouts and let retail takers fill your price.

### Rule 4: Hold to `$1.00` by Default
Selling early incurs a second per-contract exchange fee (`~1.5¢`) plus bid-ask spread. Default to **`HOLD TO $1.00`** ($0 settlement fee), exiting early only when live bids exceed your fee-adjusted break-even exit or when Q3 blowout benching threatens a live player prop Over.

---

## 4. Remote Hosting & Security

To access your workspace from your phone or laptop anywhere:
- **Passcode Lock (`APP_PASSCODE`)**: Set `APP_PASSCODE` in your hosting environment. Public health probes (`/healthz`) remain accessible for load balancers while all portfolio, market, and model APIs require unlocking via a signed `HttpOnly` session cookie or `Authorization: Bearer <passcode>` header.
- **Persistent Storage (`DATA_DIR`)**: Mount a persistent volume at `/data` and set `DATA_DIR=/data`. Your SQLite ledger, API caches, UI settings, and trained `.joblib` artifacts stay intact across container deployments.

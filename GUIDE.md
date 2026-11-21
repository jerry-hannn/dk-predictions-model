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
   - Click **Refresh** (~1–2s) to pull live ESPN scoreboards, active player box scores, injury reports, DraftKings game & player prop lines (zero API key required via ESPN/DraftKings feeds, plus optional multi-book consensus via The Odds API), or enable **Auto-refresh (60s)** inside `Filters`.
3. **Configure Exact DK Line (`Ln`) & Ask (`¢`) Before Buying**:
   - Every Spread, Total, and Player Prop card includes an inline **`Ln`** input alongside **`YES/NO` + `¢`**. If DraftKings Predictions offers an alternate or shifted number (e.g., `20.5` PRA instead of `19.5`), type `20.5` right on the card—true win probability, market prior, `Max Buy`, `Net Edge`, and Quarter-Kelly sizing recalculate immediately and persist in SQLite until you click **`↺`** to reset to the synced line.
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

---

## 5. Traditional Sportsbooks Mode (DraftKings & FanDuel)

In addition to DraftKings Predictions (DKeX) binary contracts, the workspace includes a full **Traditional Sportsbooks Mode** for **DraftKings Sportsbook (DK)** and **FanDuel Sportsbook (FD)** behind an exclusive global toggle.

### Enabling & Configuring Traditional Mode
1. Open **Workspace Settings** (top-right portfolio pill).
2. Flip **Traditional Sportsbooks Mode (DK & FD)** ON (`TRADITIONAL_SPORTSBOOKS_ENABLED=true`).
   - When **OFF** (default), the UI runs exclusively in DraftKings Predictions (DKeX) binary-contract mode.
   - When **ON**, the workspace switches exclusively to Traditional DK/FD views: Markets (American odds & DK vs. FD line shopping), the **Strategies** tab, the 3-mode Sportsbook **Analyzer**, the **Calibration** Strategy Verification Scorecard, and the **Unified Multi-Venue Ledger**.
3. Configure optional **DK Bankroll ($)** and **FD Bankroll ($)** splits (used for per-book exposure tracking), **Anti-Limiting Arb Rounding** (`$0` exact, `$1`, or `$5`), **Min Arb ROI %**, **Min +EV Edge %**, and **Include DKeX cross-arbs** (default ON).
4. Add a free **The Odds API** key to unlock simultaneous DraftKings, FanDuel, and sharp-book (`LowVig` / `BetOnline` / `Bovada`) game & player prop quotes using zero additional API credits.

### The 5 Traditional DK/FD Strategies (`Strategies` Tab)
1. **Cross-Book Arbitrage (`DK ↔ FD` + Optional `DKeX ↔ DK/FD`)**:
   - Flags opposing pairs where implied probability sum `< 1.00` (e.g., `DK +115` vs. `FD -105`, or fee-adjusted DKeX YES/NO contracts vs. DK/FD American odds).
   - Displays both **exact** stakes and **anti-limiting rounded** stakes (`$1` or `$5` increments). Any slim arbitrage whose minimum guaranteed profit drops to `<= $0` after rounding is explicitly flagged so you never accidentally lock in a rounding loss.
   - Toggle **Include DKeX cross-arbs** right inside the Arbitrage header to filter cross-venue binary-contract arbs on or off.
2. **+EV Line Shopping (`Markets` Tab)**:
   - Evaluates both DraftKings and FanDuel American odds against our calibrated win probabilities and sharp-only consensus, routing each play to the best book with **Fractional Kelly ($)** sizing and a **sharp-alignment CLV guard** that suppresses bets priced worse than sharp consensus.
3. **Middles & Polish Middles**:
   - Detects discrepant DK vs. FD spreads, totals, and player props (e.g., `DK Home -2.5` + `FD Away +3.5`, or integer **Polish middles** like `-2.5 / +3`) and computes exact double-win & push-win probabilities using historical NFL key-number (`3, 7, 6, 10, 4, 14`) distributions and NBA margin models.
   - Only surfaces middles whose predicted hit rate clears the vig break-even probability (filtering out negative-EV "blind middles").
4. **Low-Hold, Bonus Bet Converter & Profit Boost Optimizer**:
   - **Bonus Bet / Free-Bet Converter**: Ranks high-retention (`70%+`) DK vs. FD pairs where placing the profit-only Bonus Bet on a `+300` to `+500` underdog and hedging cash on the opposite favorite locks in risk-free cash regardless of outcome.
   - **Profit Boost Optimizer (`+25%` / `+30%` / `+50%`)**: Applies the exact sportsbook boost formula `b' = b × (1 + boost)` to every game & prop and ranks the highest-EV boosted bets.
   - **Low-Hold Rollover Pairs**: Ranks near-zero-hold (`<= 2.5%`) pairs for clearing deposit match or VIP play-through requirements cheaply.
5. **Steam / Off-Market Lag Scanner**:
   - Compares DraftKings and FanDuel lines against **sharp-only consensus** (strictly excluding DK & FD from the anchor) to catch stale retail prices and off-market spreads/totals before books adjust.

### 3-Mode Sportsbook Analyzer & Unified Multi-Venue Ledger
- **Analyzer (`Analyzer` Tab)**: Switch between **+EV & Profit Boost** (custom model prob or sharp-book devig), **Arb & Hedge** (supports `DK`, `FD`, and `DKeX` legs + fixed-leg live hedge locking), and **Bonus Bet Converter**.
- **Unified Multi-Venue Ledger (`Ledger` Tab)**: Stores `DK`, `FD`, and `DKeX` positions in a single SQLite ledger with venue filters, strategy tags (`EV`, `ARB`, `MIDDLE`, `BONUS_BET`, `BOOST`, `STEAM`), 1-click atomic paired-leg logging (`Log Both Legs`), Bonus Bet (`$0` cash risk, profit-only payout) accounting, `PUSH` / `VOID` / `CASHOUT` settlement, and per-book bankroll cards.

### Reading the Calibration Strategy Verification Scorecard
- When Traditional Sportsbooks Mode is ON, the **Calibration** tab renders our zero-lookahead walk-forward **Strategy Verification Scorecard** across historical NFL and NBA seasons plus a **3-Regime Stress Sensitivity Matrix** (`Standard 4.5% Vig`, `Tight 4.8% Vig Stress`, and `Active Steam / High Dispersion`). Click **Re-run Sportsbook Backtest** at any time to re-verify profitability, CLV, rounding invariants, and max drawdown directly from the UI.


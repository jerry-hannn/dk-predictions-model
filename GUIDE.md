# DraftKings Predictions (DKeX) NFL & NBA Model — Complete Pre-Game & Live Betting Guide

Welcome to your **NFL & NBA High-Confidence Event Contract Engine**, built specifically for **DraftKings Predictions (DKeX)** across both **real-time pre-game lines** and **live in-play (mid-game) trading**.

Unlike traditional sportsbooks where you bet fixed odds against the house (paying a hidden ~4.5%–8% "vig"), DraftKings Predictions is a **CFTC-regulated peer-to-peer prediction exchange** where binary **Yes / No** event contracts trade between **$0.01 (`1¢`) and $0.99 (`99¢`)** and settle at **$1.00** if correct (or `$0.00` if incorrect).

This guide covers:
1. Running the model via **CLI** (primary fast live workflow) or **Local Web Dashboard**
2. How the model prices **pre-game** and **live in-play** contracts
3. Step-by-step **live betting execution rules** on DraftKings Predictions (commercial-break Limit Orders, latency locks, blowout benching, and mid-game cash-outs)
4. Bankroll sizing, free-tier API credit conservation, and long-term discipline

---

## 1. Quickstart: CLI & Optional Web Dashboard

All commands run inside the local virtual environment `.venv` in `/Users/jerryhan/Documents/CS/model`.

### Fast CLI Live & Pre-Game Commands (Recommended for Live Action)
```bash
# 1. Scan LIVE in-play games + fresh pre-game lines right now (refreshes ESPN live boxscores + DraftKings/Sharp odds)
.venv/bin/dkmodel live --sport all

# 2. Continuous live terminal monitor during NFL Sunday or NBA slates (refreshes every 60s; Ctrl+C to stop)
.venv/bin/dkmodel live --sport nba --watch 60

# 3. Pull fresh lines filtered to ONLY live in-play games (or --mode pregame / --mode all)
.venv/bin/dkmodel picks --sport all --mode live --fresh

# 4. Pull live sportsbook player prop lines too (uses extra Odds API credits when >40 credits remain)
.venv/bin/dkmodel picks --sport nba --mode all --fresh --live-props

# 5. Instantly check an exact live ¢ contract price on your DraftKings Predictions screen
# Pass --live for in-play conservative 0.80x Quarter-Kelly sizing
.venv/bin/dkmodel check --prob 0.71 --price 62 --side YES --fee 0.015 --live

# 6. Full historical schedule/player-log sync + model retraining (run 1-2x per week)
.venv/bin/dkmodel sync --sport all
.venv/bin/dkmodel train --sport all

# 7. Run zero-leakage walk-forward historical backtests
.venv/bin/dkmodel backtest --sport all
```

### Optional Local Web Dashboard (Start Only When Wanted)
The web server stays **off** by default so nothing runs in the background unless you launch it:
```bash
.venv/bin/dkmodel dashboard
```
Open **`http://127.0.0.1:8000`** when running, and press `Ctrl+C` to stop it cleanly when done.
- Use **Refresh Live Lines** (fast, ~1–2s, zero retraining) during games, or toggle **Auto-Refresh (60s)** (which automatically pauses whenever the tab or grid is off-screen to save API credits and CPU).
- Switch **Slate Mode** between `All Active (Live + Pre-Game)`, `LIVE In-Play Only`, and `Pre-Game Only`.

---

## 2. How the Pre-Game & Live In-Play Models Work

Many bettors fall into two traps on prediction markets:
- **Trap 1 (Overpriced Heavy Favorites)**: Blindly buying `80¢` favorites that only win `76%` of the time—one `$0.815` loss wipes out more than four `$0.185` wins.
- **Trap 2 (Naive Live Extrapolation)**: Buying NBA `Over` contracts at halftime because two teams shot 62% from three in the 1st half, ignoring second-half shooting regression and pace slowdown.

### Dual-Gate Filtering (`Calibrated Win % >= 60%` + `Fee-Adjusted Net Edge >= +2.5%`)
Every pre-game and live recommendation must clear **both** gates after DraftKings' itemized fee (`Break-Even % = Contract Price + Fee`):
- **`A+` (Score 85–100 — Core Lock)**: Win Prob $\ge 65\%$ AND Net Edge $\ge +4.0\%$ after fees, fresh lines, and strong model + sharp-book agreement.
- **`A` (Score 72–84 — Strong Play)**: Win Prob $\ge 62\%$ AND Net Edge $\ge +3.0\%$ after fees.
- **`B` (Score 60–71 — Solid Value)**: Win Prob $\ge 60\%$ AND Net Edge $\ge +2.5\%$ after fees.
- **`PASS` (`< 60`)**: Insufficient win probability, negative/thin post-fee edge, or locked by live endgame latency guardrails.

### Live In-Play Conditioning Engine (`state == "IN"`)
When an NFL or NBA game kicks off or tips off, the engine automatically switches from pre-game priors to **live state-space conditioning**:
1. **Conditional Brownian Drift + Remaining Clock Variance**:
   - Projects final margin from `Current Score Diff + (Remaining Clock Fraction × Pre-Game Fundamental Drift)` with remaining volatility shrinking by $\sigma_{\text{rem}} = \sigma_0 \sqrt{r}$ (plus +1.8 late close-game foul variance in NBA).
   - **NFL Situational EP**: Adds live possession value (`+1.85 pts`) and Red-Zone scoring threat (`+3.60 pts`) directly from ESPN's live situation feed, blended in log-odds space with live play-by-play win probability and de-vigged live sportsbook odds.
2. **Empirical Bayes Live Total Mean-Reversion**:
   - Instead of doubling a hot 1st-half pace, remaining points regress **85%–88%** toward pre-game efficiency expectations. When retail traders bid live NBA `YES (Over)` contracts way too high after a 135-point first half, the model systematically surfaces high-probability **`BUY NO` (Under)** contracts!
3. **Live Box-Score Player Props + Foul/Blowout Haircuts**:
   - Pulls live player box scores (`MIN`, `PTS`, `REB`, `AST`, `3PT`, `PF`, passing/rushing/receiving) and projects `Current Accumulated Stat + Remaining Rate × Expected Remaining Minutes`.
   - Automatically applies **NBA foul-trouble haircuts** (e.g., 3+ fouls in 1H or 4+ in Q3) and **Q3/Q4 blowout starter-benching haircuts** (`22+` point leads in Q3, `16+` in Q4) so you never buy an Over on a star about to sit the 4th quarter.
   - Skips any player flagged `DNP` or `Ejected`, and locks all player props inside the final `2.5` minutes of regulation.
4. **Sharp-vs-DraftKings Live Lag Detection**:
   - Anchors evaluated spread/total/prop lines and contract ask prices to **DraftKings' actual live number**, while comparing against sharpness-weighted consensus (`LowVig`, `BetOnline`, `Bovada`, `FanDuel`, `BetMGM`). When sharp books move 30–90 seconds ahead of retail DKeX liquidity, the card highlights the lag!

---

## 3. Live In-Play Execution Rules on DraftKings Predictions

### Rule 1: Trade Dead Balls (Timeouts, Quarter Breaks, Halftime, NFL Commercials)
In-play order books move fast during live possessions, and television broadcasts lag real arena action by **8–25 seconds**:
- **Best Windows**: Execute live trades during **Halftime**, **End of Quarter**, **Official Timeouts**, or **NFL Change-of-Possession Commercial Breaks**. Lines are stable, clock is stopped, and you won't getadversely filled right as a three-pointer or interception happens.
- **Always Verify Live `¢` Ask**: Before tapping buy, run `.venv/bin/dkmodel check --prob <MODEL_PROB> --price <LIVE_ASK_CENTS> --side <YES|NO> --live` (or enter `¢` in the card's `Check ¢` box) to verify the exact live ask still clears **`Max Limit Buy`**.

### Rule 2: Respect the `<90s` Endgame Latency Lock
When a game is within **1 score** (`<= 6 pts` NBA, `<= 8 pts` NFL) with **under 90 seconds remaining**, a single whistle or rebound swings win probability by 35%+.
- The engine automatically forces **`PASS` (`Endgame latency lock`)** inside `<90s` of close games. Do not chase final-minute coin flips against low-latency market makers.

### Rule 3: Use Limit Orders (`1¢–2¢` Inside Wide Live Spreads)
Live DKeX bid-ask spreads widen from `~1.5¢` pre-game to `3¢–5¢` mid-game.
- Never market-sweep a wide ask above **`Max Limit Buy`**. During halftime or timeouts, post a **Limit Order** at or `1¢–2¢` below `Max Limit Buy` and let retail bettors fill you.

### Rule 4: Managing Open Positions (`HOLD TO $1.00` vs. Mid-Game Cash-Out)
Selling early on DraftKings Predictions costs a **second transaction fee (`~1.5¢`)** plus the live bid-ask spread:
- **Default (`HOLD TO $1.00`)**: Holding to settlement pays `$1.00` with **$0.00 exit fee**. `dkmodel live` monitors all `OPEN` Trade Ledger positions and displays your exact fee-adjusted break-even cash-out bid.
- **When to Cash Out (`CONSIDER CASH OUT`)**: If you hold a live Player Prop `Over (YES)` at `84¢+` in Q3 and the game turns into a `24+` point blowout (starters likely benched for Q4), selling into a strong `84¢–88¢` bid locks profit and eliminates Q4 benching risk.

---

## 4. Bankroll Guardrails & Free-Tier API Credit Conservation

1. **Conservative Sizing (`0.25x` Pre-Game / `0.20x` Live Quarter-Kelly)**:
   - Pre-game picks use uncertainty-shrunk Quarter-Kelly (`4%` max single-contract cap; `6%` same-game correlation cap, with 1-contract micro-bankroll support on `$25` accounts when edge clears thresholds).
   - Live in-play picks apply an extra **`0.80x` live Kelly multiplier** and wider uncertainty shrinkage to account for mid-game quote latency.
2. **Smart Free-Tier Odds API Protection (500 Credits / Month)**:
   - **Zero-Credit Baseline**: ESPN's live scoreboard, live boxscore (`/summary`), injury feeds, and embedded DraftKings opening/live lines use **0 API credits**.
   - **50% Credit Reduction**: Multi-book sharp polling uses `regions=us` (**3 credits** per refresh instead of 6 for `us,eu`), cached for `90s` during live games and `10m` pre-game.
   - **Automatic Low-Quota Guard**: `dkmodel live` displays your exact remaining credits (`x-requests-remaining`). If remaining credits drop to **`<= 15`**, automatic background Odds API calls pause seamlessly and fall back to ESPN/DraftKings live feeds so you never burn out your monthly key unexpectedly.

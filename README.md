# DraftKings Predictions (DKeX) NFL & NBA Model

Calibrated pre-game and live in-play sports prediction engine for **DraftKings Predictions** (`$0.01–$0.99` binary event contracts with `$1.00` settlement payout and per-contract exchange fee accounting) across **NFL** and **NBA** markets.

## Features

- **NFL & NBA Game Models**: Strictly chronological walk-forward Elo, Four-Factors, pace, rest/B2B fatigue, weather, and injury-adjusted Ridge/Platt-calibrated estimators (`src/models/`).
- **Player Props Engine**: Empirical-Bayes parametric rate projections (Gamma, Negative Binomial, Poisson, Student-$t$) with live box-score conditioning, foul-trouble, and blowout-benching adjustments.
- **DraftKings Predictions Contract Math**: Shin de-vigging, log-odds Bayesian market blending, `$0.015`/contract fee-adjusted break-even & Max Buy `¢` limits, Quarter-Kelly position sizing, and live latency/stale-odds locks (`src/engine/`).
- **CLI & Trade Ledger**: Fast terminal workflows (`dkmodel picks`, `dkmodel live`, `dkmodel check`, `dkmodel log-bet`, `dkmodel settle`) backed by local SQLite caching.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env to set BANKROLL and optional THE_ODDS_API_KEY

# Train models & run out-of-sample walk-forward backtests
dkmodel train --sport all
dkmodel backtest --sport all

# Scan upcoming & live slates
dkmodel picks --sport all --fresh
dkmodel live --sport nba

# Verify an exact DraftKings contract ask price before buying
dkmodel check --prob 0.58 --price 0.51 --fee 0.015
```

See **[GUIDE.md](./GUIDE.md)** for the full betting workflow, bankroll sizing rules, and live trading playbook.

## Testing

```bash
pytest -v
```

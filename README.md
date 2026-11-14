# DraftKings Predictions (DKeX) — NFL & NBA Web Platform

Minimalist, UI-first quantitative prediction and valuation web application for **DraftKings Predictions** (`$0.01–$0.99` binary event contracts settling at `$1.00` with itemized exchange fee accounting) across **NFL** and **NBA** pre-game and live in-play markets.

## Highlights

- **Minimalist Web Workspace**: Clean monochrome dark & light UI covering live/pre-game **Markets** (card & dense table views), **Contract Analyzer**, **Position Ledger & Live Cash-Out Monitor**, **Walk-Forward Calibration**, and **Settings**.
- **NFL & NBA Game Models**: Chronological walk-forward Elo, Four-Factors, pace, rest/B2B fatigue, weather, and injury-adjusted Ridge + Platt-calibrated estimators (`src/models/`).
- **Live In-Play & Player Props Engine**: Empirical-Bayes player rate projections and live state-space conditioning with foul-trouble haircuts, Q3/Q4 blowout-benching adjustments, and `<90s` endgame latency locks.
- **DraftKings Contract Math**: Shin de-vigging, log-odds Bayesian market blending, `$0.015`/contract fee-adjusted break-even & Max Buy `¢` limits, and uncertainty-shrunk Quarter-Kelly position sizing (`src/engine/`).
- **Remote-Hostable Out of the Box**: Single web-server entrypoint (`dkmodel`), `/healthz` probe, persistent `DATA_DIR` SQLite config + ledger storage, optional `APP_PASSCODE` remote lock, and turnkey configs for **Docker Compose**, **Fly.io**, **Render**, and **Railway**.

## Quickstart (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Launch the web application (defaults to http://0.0.0.0:8000)
dkmodel
```

Open **`http://localhost:8000`** in your browser. Bankroll, fee tier, Kelly sizing, Odds API key, live feed refresh, model retraining, and walk-forward backtests are all managed directly inside the UI.

## Remote Hosting

Set `APP_PASSCODE` so remote visitors must unlock the workspace before viewing or modifying positions and settings.

### Docker Compose (VPS / Homelab / Tailscale)

```bash
APP_PASSCODE="your-strong-passcode" docker compose up -d --build
```

### Cloud PaaS (Fly.io / Render / Railway)

- **Fly.io**: `fly launch --copy-config` -> `fly volumes create dkex_data --size 1` -> `fly secrets set APP_PASSCODE="your-strong-passcode"` -> `fly deploy`
- **Render**: Connect repo as a **Blueprint** (`render.yaml` provisions a 1 GB `/data` disk and auto-generates `APP_PASSCODE`).
- **Railway**: Deploy from GitHub (`railway.toml`), attach a volume mounted at `/data`, and set `DATA_DIR=/data` + `APP_PASSCODE`.

See **[GUIDE.md](./GUIDE.md)** for the complete execution playbook, contract pricing math, and remote hosting reference.

## Testing

```bash
pytest -v
```

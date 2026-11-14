"""FastAPI web server for the UI-first DraftKings Predictions (DKeX) NFL & NBA platform."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.backtest.walk_forward import run_walk_forward_backtest
from src.config import PROJECT_ROOT, get_settings, update_env_settings
from src.engine.recommender import ModelHub
from src.storage.db import Database

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE_NAME = "dkex_session"

app = FastAPI(
    title="DraftKings Predictions NFL & NBA Live Model API",
    version="2.0.0",
)


def _derive_session_token(passcode: str) -> str:
    return hmac.new(
        passcode.encode("utf-8"),
        b"dkex-ui-session-v1",
        hashlib.sha256,
    ).hexdigest()


def _is_request_authenticated(request: Request, passcode: str) -> bool:
    if not passcode:
        return True

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        candidate = auth_header[7:].strip()
        if hmac.compare_digest(candidate, passcode):
            return True

    passcode_header = request.headers.get("x-app-passcode", "").strip()
    if passcode_header and hmac.compare_digest(passcode_header, passcode):
        return True

    cookie_val = request.cookies.get(SESSION_COOKIE_NAME, "")
    if cookie_val and hmac.compare_digest(cookie_val, _derive_session_token(passcode)):
        return True

    return False


@app.middleware("http")
async def security_and_cache_middleware(request: Request, call_next):
    path = request.url.path

    if path.startswith("/api/") and not path.startswith("/api/auth/"):
        passcode = get_settings().app_passcode
        if passcode and not _is_request_authenticated(request, passcode):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
            )

    response: Response = await call_next(request)
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _get_hub() -> ModelHub:
    settings = get_settings()
    return ModelHub(settings=settings, db=Database(settings.db_path))


class AuthLoginRequest(BaseModel):
    passcode: str = Field(..., min_length=1)


class ConfigUpdateRequest(BaseModel):
    bankroll: Optional[float] = Field(None, gt=0.0)
    default_fee_dollars: Optional[float] = Field(None, ge=0.0, le=0.10)
    kelly_fraction: Optional[float] = Field(None, ge=0.05, le=1.0)
    max_single_bet_pct: Optional[float] = Field(None, ge=0.01, le=0.25)
    max_same_game_pct: Optional[float] = Field(None, ge=0.01, le=0.35)
    min_win_prob: Optional[float] = Field(None, ge=0.50, le=0.95)
    min_net_edge: Optional[float] = Field(None, ge=-0.10, le=0.30)
    the_odds_api_key: Optional[str] = None


class ContractEvalRequest(BaseModel):
    yes_true_prob: float = Field(..., gt=0.0, lt=1.0)
    side: str = Field("YES", pattern="^(YES|NO|yes|no)$")
    contract_price: float = Field(..., ge=0.01, le=0.99)
    fee_per_contract: float = Field(0.015, ge=0.0, le=0.10)
    bankroll: float = Field(25.0, gt=0.0)
    min_win_prob: float = Field(0.60, ge=0.50, le=0.95)
    min_net_edge: float = Field(0.025, ge=-0.10, le=0.30)
    is_live: bool = False


class CreateTradeRequest(BaseModel):
    sport: str
    event_name: str
    market_type: str
    selection_label: str
    side: str = Field(..., pattern="^(YES|NO|yes|no)$")
    entry_price: float = Field(..., ge=0.01, le=0.99)
    fee_per_contract: float = Field(0.015, ge=0.0, le=0.10)
    contracts: int = Field(..., ge=1)
    model_prob: float = Field(..., gt=0.0, lt=1.0)
    fair_exit_price: float = Field(..., ge=0.01, le=0.99)
    notes: str = ""


class UpdateTradeRequest(BaseModel):
    status: str = Field(..., pattern="^(OPEN|WON|LOST|SOLD|open|won|lost|sold)$")
    exit_or_settle_price: Optional[float] = Field(None, ge=0.0, le=1.0)
    exit_fee_per_contract: float = Field(0.0, ge=0.0, le=0.10)
    closing_price: Optional[float] = Field(None, ge=0.01, le=0.99)
    notes: Optional[str] = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/auth/status")
def auth_status(request: Request) -> dict[str, bool]:
    passcode = get_settings().app_passcode
    return {
        "auth_enabled": bool(passcode),
        "authenticated": _is_request_authenticated(request, passcode),
    }


@app.post("/api/auth/login")
def auth_login(req: AuthLoginRequest, response: Response) -> dict[str, bool]:
    passcode = get_settings().app_passcode
    if passcode and not hmac.compare_digest(req.passcode.strip(), passcode):
        raise HTTPException(status_code=401, detail="Invalid passcode.")
    if passcode:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=_derive_session_token(passcode),
            httponly=True,
            samesite="lax",
            max_age=30 * 24 * 3600,
        )
    return {"authenticated": True}


@app.post("/api/auth/logout")
def auth_logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(key=SESSION_COOKIE_NAME, samesite="lax")
    return {"authenticated": False}


def _mask_api_key(raw: str) -> str:
    if not raw:
        return ""
    if len(raw) <= 4:
        return "••••"
    return f"••••{raw[-4:]}"


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    hub = _get_hub()
    s = hub.settings
    nfl_cached, _ = hub.db.get_cache_with_age("odds_api:NFL:game_odds", allow_stale=True)
    nba_cached, _ = hub.db.get_cache_with_age("odds_api:NBA:game_odds", allow_stale=True)
    return {
        "bankroll": s.bankroll,
        "default_fee_dollars": s.default_fee_dollars,
        "kelly_fraction": s.kelly_fraction,
        "max_single_bet_pct": s.max_single_bet_pct,
        "max_same_game_pct": s.max_same_game_pct,
        "min_win_prob": s.min_win_prob,
        "min_net_edge": s.min_net_edge,
        "odds_api_connected": bool(s.the_odds_api_key),
        "odds_api_key_masked": _mask_api_key(s.the_odds_api_key),
        "sharp_events_cached": len(nfl_cached or []) + len(nba_cached or []),
        "odds_quota": hub.odds.get_quota_status(),
        "auth_enabled": bool(s.app_passcode),
    }


@app.post("/api/config")
def save_config(req: ConfigUpdateRequest) -> dict[str, Any]:
    updates: dict[str, str] = {}
    if req.bankroll is not None:
        updates["BANKROLL"] = f"{req.bankroll:.2f}"
    if req.default_fee_dollars is not None:
        updates["DEFAULT_FEE_DOLLARS"] = f"{req.default_fee_dollars:.4f}"
    if req.kelly_fraction is not None:
        updates["KELLY_FRACTION"] = f"{req.kelly_fraction:.3f}"
    if req.max_single_bet_pct is not None:
        updates["MAX_SINGLE_BET_PCT"] = f"{req.max_single_bet_pct:.3f}"
    if req.max_same_game_pct is not None:
        updates["MAX_SAME_GAME_PCT"] = f"{req.max_same_game_pct:.3f}"
    if req.min_win_prob is not None:
        updates["MIN_WIN_PROB"] = f"{req.min_win_prob:.3f}"
    if req.min_net_edge is not None:
        updates["MIN_NET_EDGE"] = f"{req.min_net_edge:.3f}"
    if req.the_odds_api_key is not None:
        updates["THE_ODDS_API_KEY"] = req.the_odds_api_key.strip()
    if updates:
        update_env_settings(updates)
    return get_config()


@app.get("/api/picks")
def get_picks(
    sport: str = Query("all"),
    market: str = Query("all"),
    mode: str = Query("all"),
    date: Optional[str] = Query(None),
    min_prob: Optional[float] = Query(None, ge=0.50, le=0.95),
    min_edge: Optional[float] = Query(None, ge=-0.20, le=0.30),
    fee: Optional[float] = Query(None, ge=0.0, le=0.10),
    bankroll: Optional[float] = Query(None, gt=0.0),
    include_passes: bool = Query(False),
    fresh: bool = Query(False),
    live_props: bool = Query(False),
) -> dict[str, Any]:
    hub = _get_hub()
    eff_bankroll = bankroll if bankroll is not None else hub.settings.bankroll
    picks = hub.generate_picks(
        sport=sport,
        market_filter=market,
        mode=mode,
        target_date=date,
        min_win_prob=min_prob,
        min_net_edge=min_edge,
        fee_per_contract=fee,
        bankroll=eff_bankroll,
        include_passes=include_passes,
        fresh=fresh,
        live_props=live_props,
    )
    ledger = hub.db.ledger_summary(starting_bankroll=eff_bankroll)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(picks),
        "live_count": sum(1 for p in picks if p.get("game_state") == "LIVE"),
        "high_confidence_count": sum(1 for p in picks if p["confidence_tier"] in ("A+", "A")),
        "odds_api_connected": bool(hub.settings.the_odds_api_key),
        "odds_quota": hub.odds.get_quota_status(),
        "picks": picks,
        "ledger_summary": ledger,
    }


@app.post("/api/evaluate-contract")
def evaluate_contract_endpoint(req: ContractEvalRequest) -> dict[str, Any]:
    hub = _get_hub()
    return hub.reevaluate_custom_price(
        yes_true_prob=req.yes_true_prob,
        side=req.side,
        contract_price=req.contract_price,
        fee_per_contract=req.fee_per_contract,
        bankroll=req.bankroll,
        min_win_prob=req.min_win_prob,
        min_net_edge=req.min_net_edge,
        is_live=req.is_live,
    )


@app.post("/api/refresh-live")
def refresh_live_endpoint(
    sport: str = Query("all"),
    date: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Fast live refresh of ESPN scoreboards, live box scores, injuries, and Odds API lines."""
    hub = _get_hub()
    return {"refreshed": hub.refresh_live_feeds(sport=sport, target_date=date)}


@app.post("/api/sync")
def sync_endpoint(
    sport: str = Query("all"),
    date: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Full historical schedule/log sync + model retraining."""
    hub = _get_hub()
    sync_info = hub.sync_data(sport=sport, target_date=date, force=True)
    train_info = hub.train_all(sport=sport)
    return {"synced": sync_info, "trained": train_info}


@app.post("/api/train")
def train_endpoint(sport: str = Query("all")) -> dict[str, Any]:
    """Train and calibrate game & player prop models using local schedule/log data."""
    hub = _get_hub()
    return {"trained": hub.train_all(sport=sport)}


@app.get("/api/backtest-metrics")
def backtest_metrics_endpoint() -> dict[str, Any]:
    settings = get_settings()
    metrics_file = settings.models_dir / "backtest_metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    bundled_metrics = PROJECT_ROOT / "models" / "backtest_metrics.json"
    if bundled_metrics.exists():
        return json.loads(bundled_metrics.read_text(encoding="utf-8"))
    return run_walk_forward_backtest(sport="all", save_artifact=True)


@app.post("/api/backtest")
def run_backtest_endpoint(
    sport: str = Query("all"),
    fee: float = Query(0.015, ge=0.0, le=0.10),
) -> dict[str, Any]:
    """Run fresh chronological walk-forward validation right from the UI."""
    save_artifact = sport.lower() == "all"
    return run_walk_forward_backtest(
        sport=sport,
        fee_per_contract=fee,
        save_artifact=save_artifact,
    )


@app.get("/api/ledger")
def list_ledger(bankroll: Optional[float] = Query(None, gt=0.0)) -> dict[str, Any]:
    hub = _get_hub()
    bk = bankroll if bankroll is not None else hub.settings.bankroll
    trades = hub.db.list_trades()
    open_guidance = (
        {op["id"]: op for op in hub.evaluate_open_positions(fresh=False)}
        if any(t["status"] == "OPEN" for t in trades)
        else {}
    )
    enriched = [
        {
            **t,
            "live_guidance": open_guidance[t["id"]]["live_signal"],
            "live_model_prob": open_guidance[t["id"]]["live_model_prob"],
            "score_display": open_guidance[t["id"]].get("score_display", ""),
            "game_state": open_guidance[t["id"]].get("game_state", ""),
        }
        if t["id"] in open_guidance
        else t
        for t in trades
    ]
    return {
        "summary": hub.db.ledger_summary(starting_bankroll=bk),
        "trades": enriched,
    }


@app.post("/api/ledger")
def create_ledger_trade(req: CreateTradeRequest) -> dict[str, Any]:
    settings = get_settings()
    db = Database(settings.db_path)
    created = db.log_trade(
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        sport=req.sport,
        event_name=req.event_name,
        market_type=req.market_type,
        selection_label=req.selection_label,
        side=req.side,
        entry_price=req.entry_price,
        fee_per_contract=req.fee_per_contract,
        contracts=req.contracts,
        model_prob=req.model_prob,
        fair_exit_price=req.fair_exit_price,
        notes=req.notes,
    )
    return created


@app.patch("/api/ledger/{trade_id}")
def update_ledger_trade(trade_id: int, req: UpdateTradeRequest) -> dict[str, Any]:
    settings = get_settings()
    db = Database(settings.db_path)
    try:
        updated = db.update_trade(
            trade_id,
            status=req.status,
            exit_or_settle_price=req.exit_or_settle_price,
            exit_fee_per_contract=req.exit_fee_per_contract,
            closing_price=req.closing_price,
            notes=req.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Trade not found.")
    return updated


@app.delete("/api/ledger/{trade_id}")
def delete_ledger_trade(trade_id: int) -> dict[str, Any]:
    settings = get_settings()
    db = Database(settings.db_path)
    ok = db.delete_trade(trade_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Trade not found.")
    return {"deleted": trade_id}


@app.get("/api/guide")
def get_guide_markdown() -> dict[str, str]:
    guide_path = PROJECT_ROOT / "GUIDE.md"
    if guide_path.exists():
        return {"markdown": guide_path.read_text(encoding="utf-8")}
    return {"markdown": "Guide not generated yet."}

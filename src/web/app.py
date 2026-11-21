"""FastAPI web server for the UI-first DraftKings Predictions (DKeX) NFL & NBA platform."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from src.backtest.sportsbook_backtest import run_sportsbook_backtest
from src.backtest.walk_forward import run_walk_forward_backtest
from src.config import PROJECT_ROOT, get_settings, update_env_settings
from src.engine.recommender import ModelHub
from src.storage.db import Database

def _resolve_static_dir() -> Path:
    pkg_static = Path(__file__).resolve().parent / "static"
    if pkg_static.is_dir():
        return pkg_static
    root_static = PROJECT_ROOT / "src" / "web" / "static"
    if root_static.is_dir():
        return root_static
    return pkg_static


STATIC_DIR = _resolve_static_dir()
SESSION_COOKIE_NAME = "dkex_session"
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DraftKings Predictions & Sportsbook Live Model API",
    version="2.1.0",
)


def _derive_session_token(passcode: str) -> str:
    return hmac.new(
        passcode.encode("utf-8"),
        b"dkex-ui-session-v1",
        hashlib.sha256,
    ).hexdigest()


def _safe_equals(candidate: str, expected: str) -> bool:
    """Constant-time comparison that also handles non-ASCII input (str compare_digest raises)."""
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def _is_request_authenticated(request: Request, passcode: str) -> bool:
    if not passcode:
        return True

    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        candidate = auth_header[7:].strip()
        if _safe_equals(candidate, passcode):
            return True

    passcode_header = request.headers.get("x-app-passcode", "").strip()
    if passcode_header and _safe_equals(passcode_header, passcode):
        return True

    cookie_val = request.cookies.get(SESSION_COOKIE_NAME, "")
    if cookie_val and _safe_equals(cookie_val, _derive_session_token(passcode)):
        return True

    return False


@app.middleware("http")
async def security_and_cache_middleware(request: Request, call_next):
    path = request.url.path

    if path.startswith("/api/") and not path.startswith("/api/auth/"):
        passcode = await run_in_threadpool(lambda: get_settings().app_passcode)
        if passcode and not _is_request_authenticated(request, passcode):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
                headers={"Cache-Control": "no-store"},
            )

    response: Response = await call_next(request)
    if path == "/" or path.startswith("/static/"):
        # Always revalidate (ETag / Last-Modified -> cheap 304s) so deploys show up immediately
        response.headers["Cache-Control"] = "no-cache"
    elif path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# Slate / ledger JSON payloads can be several hundred KB; compress for remote clients
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _get_hub() -> ModelHub:
    settings = get_settings()
    return ModelHub(settings=settings, db=Database(settings.db_path))


def _validate_american_odds(value: Optional[float]) -> Optional[float]:
    """American odds must be <= -100 or >= +100 (e.g. -110, +150); values in (-100, 100) are invalid."""
    if value is None:
        return value
    if not (abs(float(value)) >= 100.0 and abs(float(value)) <= 100000.0):
        raise ValueError("American odds must be <= -100 or >= +100 (e.g. -110 or +150).")
    return value


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
    traditional_sportsbooks_enabled: Optional[bool] = None
    dk_bankroll: Optional[float] = Field(None, ge=0.0)
    fd_bankroll: Optional[float] = Field(None, ge=0.0)
    arb_stake_rounding: Optional[float] = Field(None, ge=0.0, le=25.0)
    min_arb_roi: Optional[float] = Field(None, ge=0.0, le=0.20)
    min_sb_ev_edge: Optional[float] = Field(None, ge=0.0, le=0.25)
    include_dkex_cross_arb: Optional[bool] = None
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


class SportsbookEvalRequest(BaseModel):
    true_prob: Optional[float] = Field(None, gt=0.01, lt=0.99)
    sharp_odds_side: Optional[float] = None
    sharp_odds_opp: Optional[float] = None
    dk_odds: Optional[float] = None
    fd_odds: Optional[float] = None
    bankroll: Optional[float] = Field(None, gt=0.0)
    profit_boost_pct: float = Field(0.0, ge=0.0, le=2.0)
    is_live: bool = False

    _check_odds = field_validator(
        "sharp_odds_side", "sharp_odds_opp", "dk_odds", "fd_odds"
    )(_validate_american_odds)


class SportsbookArbCalcRequest(BaseModel):
    calc_mode: str = Field("arb", pattern="^(arb|bonus_bet)$")
    book_a: str = "DRAFTKINGS"
    odds_a: Optional[float] = None
    dkex_ask_a: Optional[float] = Field(None, ge=0.01, le=0.99)
    dkex_fee_a: Optional[float] = Field(None, ge=0.0, le=0.10)
    book_b: str = "FANDUEL"
    odds_b: Optional[float] = None
    dkex_ask_b: Optional[float] = Field(None, ge=0.01, le=0.99)
    dkex_fee_b: Optional[float] = Field(None, ge=0.0, le=0.10)
    total_stake: float = Field(100.0, gt=0.0)
    fixed_stake_a: Optional[float] = Field(None, gt=0.0)
    rounding_step: Optional[float] = Field(None, ge=0.0, le=25.0)
    bonus_amount: float = Field(50.0, gt=0.0)

    _check_odds = field_validator("odds_a", "odds_b")(_validate_american_odds)


class LineOverrideRequest(BaseModel):
    override_key: str = Field(..., min_length=3)
    sport: str = Field(..., min_length=2)
    event_id: str = Field(..., min_length=1)
    pick_id: str = ""
    date: Optional[str] = None
    line_value: Optional[float] = Field(None, ge=-100.0, le=600.0)
    contract_price: Optional[float] = Field(None, ge=0.01, le=0.99)
    side: Optional[str] = Field(None, pattern="^(YES|NO|yes|no)$")
    reset: bool = False
    fee_per_contract: Optional[float] = Field(None, ge=0.0, le=0.10)
    bankroll: Optional[float] = Field(None, gt=0.0)
    min_win_prob: Optional[float] = Field(None, ge=0.50, le=0.95)
    min_net_edge: Optional[float] = Field(None, ge=-0.20, le=0.30)
    live_props: bool = True


class CreateTradeRequest(BaseModel):
    sport: str
    event_name: str
    market_type: str
    selection_label: str
    side: str = Field(..., min_length=1)
    entry_price: float = Field(0.50, ge=0.01, le=0.99)
    fee_per_contract: float = Field(0.015, ge=0.0, le=0.10)
    contracts: int = Field(1, ge=1)
    model_prob: float = Field(0.50, gt=0.0, lt=1.0)
    fair_exit_price: float = Field(0.50, ge=0.01, le=0.99)
    notes: str = ""
    venue: str = "DKEX"
    bet_format: Optional[str] = None
    odds_american: Optional[int] = None
    stake: Optional[float] = Field(None, gt=0.0)
    is_bonus_bet: bool = False
    strategy_tag: str = "EV"
    group_id: Optional[str] = None

    _check_odds = field_validator("odds_american")(_validate_american_odds)


class CreateTradePairRequest(BaseModel):
    leg_a: CreateTradeRequest
    leg_b: CreateTradeRequest
    group_id: Optional[str] = None


class UpdateTradeRequest(BaseModel):
    status: str = Field(
        ...,
        pattern="^(OPEN|WON|LOST|SOLD|PUSH|VOID|CASHOUT|open|won|lost|sold|push|void|cashout)$",
    )
    exit_or_settle_price: Optional[float] = Field(None, ge=0.0)
    exit_fee_per_contract: float = Field(0.0, ge=0.0, le=0.10)
    closing_price: Optional[float] = Field(None, ge=0.01, le=0.99)
    closing_odds_american: Optional[int] = None
    cashout_amount: Optional[float] = Field(None, ge=0.0)
    notes: Optional[str] = None

    _check_odds = field_validator("closing_odds_american")(_validate_american_odds)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": "2.1.0"}


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


_LOGIN_FAILURES: dict[str, list[float]] = {}
_LOGIN_FAILURES_LOCK = threading.Lock()
_LOGIN_WINDOW_SECONDS = 300.0
_LOGIN_MAX_FAILURES = 10
# Global ceiling across all clients: forwarded client IPs can be spoofed, so per-IP limits alone
# don't stop a distributed/rotating brute-force attempt.
_LOGIN_GLOBAL_KEY = "__all_clients__"
_LOGIN_MAX_GLOBAL_FAILURES = 100


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _recent_failures(key: str, now: float) -> list[float]:
    attempts = [t for t in _LOGIN_FAILURES.get(key, []) if now - t < _LOGIN_WINDOW_SECONDS]
    if attempts:
        _LOGIN_FAILURES[key] = attempts
    else:
        _LOGIN_FAILURES.pop(key, None)
    return attempts


def _record_login_failure(key: str, now: float) -> None:
    if len(_LOGIN_FAILURES) > 2048:
        for k in list(_LOGIN_FAILURES):
            _recent_failures(k, now)
    _LOGIN_FAILURES.setdefault(key, []).append(now)
    _LOGIN_FAILURES.setdefault(_LOGIN_GLOBAL_KEY, []).append(now)


def _is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto == "https" or request.url.scheme == "https"


@app.post("/api/auth/login")
def auth_login(req: AuthLoginRequest, request: Request, response: Response) -> dict[str, bool]:
    passcode = get_settings().app_passcode
    if passcode:
        key = _client_key(request)
        now = time.time()
        with _LOGIN_FAILURES_LOCK:
            if (
                len(_recent_failures(key, now)) >= _LOGIN_MAX_FAILURES
                or len(_recent_failures(_LOGIN_GLOBAL_KEY, now)) >= _LOGIN_MAX_GLOBAL_FAILURES
            ):
                raise HTTPException(
                    status_code=429,
                    detail="Too many failed attempts. Please wait a few minutes and try again.",
                )
        if not _safe_equals(req.passcode.strip(), passcode):
            with _LOGIN_FAILURES_LOCK:
                _record_login_failure(key, now)
            raise HTTPException(status_code=401, detail="Invalid passcode.")
        with _LOGIN_FAILURES_LOCK:
            _LOGIN_FAILURES.pop(key, None)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=_derive_session_token(passcode),
            httponly=True,
            samesite="lax",
            secure=_is_https(request),
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
        "traditional_sportsbooks_enabled": s.traditional_sportsbooks_enabled,
        "dk_bankroll": s.dk_bankroll,
        "fd_bankroll": s.fd_bankroll,
        "arb_stake_rounding": s.arb_stake_rounding,
        "min_arb_roi": s.min_arb_roi,
        "min_sb_ev_edge": s.min_sb_ev_edge,
        "include_dkex_cross_arb": s.include_dkex_cross_arb,
        "odds_api_connected": bool(s.the_odds_api_key),
        "odds_api_key_masked": _mask_api_key(s.the_odds_api_key),
        "sharp_events_cached": len(nfl_cached or []) + len(nba_cached or []),
        "odds_quota": hub.odds.get_quota_status(),
        "line_overrides_count": len(hub.db.get_line_overrides()),
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
    if req.traditional_sportsbooks_enabled is not None:
        updates["TRADITIONAL_SPORTSBOOKS_ENABLED"] = (
            "true" if req.traditional_sportsbooks_enabled else "false"
        )
    if req.dk_bankroll is not None:
        updates["DK_BANKROLL"] = "" if req.dk_bankroll <= 0 else f"{req.dk_bankroll:.2f}"
    if req.fd_bankroll is not None:
        updates["FD_BANKROLL"] = "" if req.fd_bankroll <= 0 else f"{req.fd_bankroll:.2f}"
    if req.arb_stake_rounding is not None:
        updates["ARB_STAKE_ROUNDING"] = f"{req.arb_stake_rounding:.2f}"
    if req.min_arb_roi is not None:
        updates["MIN_ARB_ROI"] = f"{req.min_arb_roi:.4f}"
    if req.min_sb_ev_edge is not None:
        updates["MIN_SB_EV_EDGE"] = f"{req.min_sb_ev_edge:.4f}"
    if req.include_dkex_cross_arb is not None:
        updates["INCLUDE_DKEX_CROSS_ARB"] = "true" if req.include_dkex_cross_arb else "false"
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
    live_props: bool = Query(True),
) -> dict[str, Any]:
    hub = _get_hub()
    eff_bankroll = bankroll if bankroll is not None else hub.settings.bankroll
    all_candidates = hub.generate_picks(
        sport=sport,
        market_filter=market,
        mode=mode,
        target_date=date,
        min_win_prob=min_prob,
        min_net_edge=min_edge,
        fee_per_contract=fee,
        bankroll=eff_bankroll,
        include_passes=True,
        fresh=fresh,
        live_props=live_props,
    )
    picks = (
        all_candidates
        if include_passes
        else [c for c in all_candidates if c["confidence_tier"] != "PASS"]
    )
    ledger = hub.db.ledger_summary(
        starting_bankroll=eff_bankroll,
        dk_bankroll=hub.settings.dk_bankroll,
        fd_bankroll=hub.settings.fd_bankroll,
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(picks),
        "total_candidates": len(all_candidates),
        "games_count": len({c["event_id"] for c in all_candidates}),
        "live_count": sum(1 for p in picks if p.get("game_state") == "LIVE"),
        "high_confidence_count": sum(1 for p in picks if p["confidence_tier"] in ("A+", "A")),
        "odds_api_connected": bool(hub.settings.the_odds_api_key),
        "odds_quota": hub.odds.get_quota_status(),
        "picks": picks,
        "ledger_summary": ledger,
    }


@app.get("/api/sportsbook/slate")
def get_sportsbook_slate(
    sport: str = Query("all"),
    market: str = Query("all"),
    mode: str = Query("all"),
    book: str = Query("all"),
    date: Optional[str] = Query(None),
    min_ev: Optional[float] = Query(None, ge=-0.20, le=0.30),
    min_prob: Optional[float] = Query(None, ge=0.20, le=0.95),
    bankroll: Optional[float] = Query(None, gt=0.0),
    arb_stake: float = Query(100.0, gt=0.0),
    arb_rounding: Optional[float] = Query(None, ge=0.0, le=25.0),
    min_arb_roi: Optional[float] = Query(None, ge=0.0, le=0.20),
    bonus_bet_size: float = Query(50.0, gt=0.0),
    boost_pct: float = Query(0.25, ge=0.0, le=2.0),
    include_passes: bool = Query(False),
    include_dkex_cross_arb: Optional[bool] = Query(None),
    fresh: bool = Query(False),
    live_props: bool = Query(True),
) -> dict[str, Any]:
    """Traditional DraftKings & FanDuel slate: +EV line shopping, arbitrage, middles, promos, & steam lag."""
    hub = _get_hub()
    eff_bankroll = bankroll if bankroll is not None else hub.settings.bankroll
    slate = hub.generate_sportsbook_slate(
        sport=sport,
        market_filter=market,
        mode=mode,
        book_filter=book,
        target_date=date,
        min_ev=min_ev,
        min_prob=min_prob,
        bankroll=eff_bankroll,
        arb_stake=arb_stake,
        arb_rounding=arb_rounding,
        min_arb_roi=min_arb_roi,
        bonus_bet_size=bonus_bet_size,
        boost_pct=boost_pct,
        include_passes=include_passes,
        include_dkex_cross_arb=include_dkex_cross_arb,
        fresh=fresh,
        live_props=live_props,
    )
    slate["ledger_summary"] = hub.db.ledger_summary(
        starting_bankroll=eff_bankroll,
        dk_bankroll=hub.settings.dk_bankroll,
        fd_bankroll=hub.settings.fd_bankroll,
    )
    return slate


@app.post("/api/sportsbook/evaluate")
def evaluate_sportsbook_endpoint(req: SportsbookEvalRequest) -> dict[str, Any]:
    hub = _get_hub()
    try:
        return hub.reevaluate_sportsbook_bet(
            true_prob=req.true_prob,
            sharp_odds_side=req.sharp_odds_side,
            sharp_odds_opp=req.sharp_odds_opp,
            dk_odds=req.dk_odds,
            fd_odds=req.fd_odds,
            bankroll=req.bankroll,
            profit_boost_pct=req.profit_boost_pct,
            is_live=req.is_live,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sportsbook/arb-calc")
def sportsbook_arb_calc_endpoint(req: SportsbookArbCalcRequest) -> dict[str, Any]:
    hub = _get_hub()
    try:
        if req.calc_mode == "bonus_bet":
            return hub.calculate_bonus_bet_conversion(
                bonus_book=req.book_a,
                bonus_odds=float(req.odds_a if req.odds_a is not None else 350),
                hedge_book=req.book_b,
                hedge_odds=float(req.odds_b if req.odds_b is not None else -400),
                bonus_amount=req.bonus_amount,
            )
        return hub.calculate_arb_or_hedge(
            book_a=req.book_a,
            odds_a=req.odds_a,
            dkex_ask_a=req.dkex_ask_a,
            dkex_fee_a=req.dkex_fee_a,
            book_b=req.book_b,
            odds_b=req.odds_b,
            dkex_ask_b=req.dkex_ask_b,
            dkex_fee_b=req.dkex_fee_b,
            total_stake=req.total_stake,
            fixed_stake_a=req.fixed_stake_a,
            rounding_step=req.rounding_step,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sportsbook/backtest-metrics")
def sportsbook_backtest_metrics_endpoint() -> dict[str, Any]:
    settings = get_settings()
    metrics_file = settings.models_dir / "sportsbook_backtest_metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    bundled = PROJECT_ROOT / "models" / "sportsbook_backtest_metrics.json"
    if bundled.exists():
        return json.loads(bundled.read_text(encoding="utf-8"))
    return run_sportsbook_backtest(
        sport="all",
        arb_rounding=settings.arb_stake_rounding,
        include_dkex_cross=settings.include_dkex_cross_arb,
        save_artifact=True,
    )


@app.post("/api/sportsbook/backtest")
def run_sportsbook_backtest_endpoint(
    sport: str = Query("all"),
    regime: str = Query("standard"),
    arb_rounding: float = Query(1.0, ge=0.0, le=25.0),
    include_dkex_cross: bool = Query(True),
) -> dict[str, Any]:
    """Run fresh zero-lookahead walk-forward Traditional DK/FD strategy verification backtest."""
    save_artifact = sport.lower() == "all" and regime == "standard"
    return run_sportsbook_backtest(
        sport=sport,
        regime=regime,
        arb_rounding=arb_rounding,
        include_dkex_cross=include_dkex_cross,
        save_artifact=save_artifact,
    )


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


@app.post("/api/line-override")
def line_override_endpoint(req: LineOverrideRequest) -> dict[str, Any]:
    """Persist or reset a custom line override and recompute the market card."""
    hub = _get_hub()
    try:
        updated_pick = hub.apply_line_override(
            override_key=req.override_key,
            sport=req.sport,
            event_id=req.event_id,
            pick_id=req.pick_id,
            target_date=req.date,
            line_value=req.line_value,
            contract_price=req.contract_price,
            side=req.side,
            reset=req.reset,
            fee_per_contract=req.fee_per_contract,
            bankroll=req.bankroll,
            min_win_prob=req.min_win_prob,
            min_net_edge=req.min_net_edge,
            live_props=req.live_props,
        )
        return {"pick": updated_pick}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/line-overrides")
def clear_line_overrides_endpoint() -> dict[str, int]:
    """Clear all saved custom line overrides."""
    settings = get_settings()
    db = Database(settings.db_path)
    return {"cleared": db.clear_line_overrides()}


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
def list_ledger(
    bankroll: Optional[float] = Query(None, gt=0.0),
    venue: Optional[str] = Query(None),
) -> dict[str, Any]:
    hub = _get_hub()
    bk = bankroll if bankroll is not None else hub.settings.bankroll
    trades = hub.db.list_trades(venue=venue)
    open_guidance: dict[Any, dict[str, Any]] = {}
    guidance_error: Optional[str] = None
    if any(t["status"] == "OPEN" for t in trades):
        try:
            open_guidance = {op["id"]: op for op in hub.evaluate_open_positions(fresh=False)}
        except Exception as exc:  # noqa: BLE001 - never block access to the ledger itself
            logger.exception("Live position guidance failed")
            guidance_error = f"Live guidance unavailable: {exc.__class__.__name__}"
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
        "summary": hub.db.ledger_summary(
            starting_bankroll=bk,
            dk_bankroll=hub.settings.dk_bankroll,
            fd_bankroll=hub.settings.fd_bankroll,
        ),
        "trades": enriched,
        "guidance_error": guidance_error,
    }


def _trade_req_to_kwargs(req: CreateTradeRequest, created_at: str) -> dict[str, Any]:
    return {
        "created_at": created_at,
        "sport": req.sport,
        "event_name": req.event_name,
        "market_type": req.market_type,
        "selection_label": req.selection_label,
        "side": req.side,
        "entry_price": req.entry_price,
        "fee_per_contract": req.fee_per_contract,
        "contracts": req.contracts,
        "model_prob": req.model_prob,
        "fair_exit_price": req.fair_exit_price,
        "notes": req.notes,
        "venue": req.venue,
        "bet_format": req.bet_format,
        "odds_american": req.odds_american,
        "stake": req.stake,
        "is_bonus_bet": req.is_bonus_bet,
        "strategy_tag": req.strategy_tag,
        "group_id": req.group_id,
    }


@app.post("/api/ledger")
def create_ledger_trade(req: CreateTradeRequest) -> dict[str, Any]:
    settings = get_settings()
    db = Database(settings.db_path)
    try:
        created = db.log_trade(
            **_trade_req_to_kwargs(req, datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return created


@app.post("/api/ledger/pair")
def create_ledger_trade_pair(req: CreateTradePairRequest) -> dict[str, Any]:
    """Atomically log both legs of an Arbitrage, Middle, or Bonus Bet Hedge with shared group_id."""
    settings = get_settings()
    db = Database(settings.db_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        legs = db.log_trade_pair(
            _trade_req_to_kwargs(req.leg_a, ts),
            _trade_req_to_kwargs(req.leg_b, ts),
            group_id=req.group_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"group_id": legs[0]["group_id"], "trades": legs}


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
            closing_odds_american=req.closing_odds_american,
            cashout_amount=req.cashout_amount,
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

"""Regression tests for security, concurrency, parsing, and API hardening fixes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.config import update_env_settings
from src.data.espn_client import ESPNClient
from src.data.odds_client import OddsClient
from src.engine.recommender import ModelHub
from src.storage.db import Database
from src.web import app as web_app


def test_auth_non_ascii_passcode_and_rate_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("APP_PASSCODE", "secret-pass")
    with web_app._LOGIN_FAILURES_LOCK:
        web_app._LOGIN_FAILURES.clear()

    client = TestClient(web_app.app)

    # Non-ASCII passcode attempt must return 401, not 500 from hmac.compare_digest(str, str)
    r_unicode = client.post("/api/auth/login", json={"passcode": "héllo🔒"})
    assert r_unicode.status_code == 401

    # Exceeding failure threshold triggers 429 rate limit
    for _ in range(web_app._LOGIN_MAX_FAILURES - 1):
        r = client.post("/api/auth/login", json={"passcode": "wrong"})
        assert r.status_code == 401
    r_throttled = client.post("/api/auth/login", json={"passcode": "secret-pass"})
    assert r_throttled.status_code == 429

    # Cleanup global throttle state so later tests aren't affected
    with web_app._LOGIN_FAILURES_LOCK:
        web_app._LOGIN_FAILURES.clear()


def test_american_odds_validation_rejects_impossible_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "odds.sqlite3"))
    monkeypatch.delenv("APP_PASSCODE", raising=False)
    client = TestClient(web_app.app)

    for bad_odds in (0, 50, -99):
        r = client.post(
            "/api/sportsbook/evaluate",
            json={"true_prob": 0.55, "dk_odds": bad_odds},
        )
        assert r.status_code == 422

    r_ok = client.post(
        "/api/sportsbook/evaluate",
        json={"true_prob": 0.55, "fd_odds": -105},
    )
    assert r_ok.status_code == 200
    assert "FANDUEL" in r_ok.json()["books"]


def test_ledger_degrades_gracefully_when_live_guidance_fails(tmp_path: Path, monkeypatch) -> None:
    db_file = tmp_path / "ledger.sqlite3"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.delenv("APP_PASSCODE", raising=False)

    client = TestClient(web_app.app)
    r_create = client.post(
        "/api/ledger",
        json={
            "sport": "NBA",
            "event_name": "NYK @ BOS",
            "market_type": "Moneyline",
            "selection_label": "BOS to Win",
            "side": "YES",
            "entry_price": 0.58,
            "contracts": 5,
            "model_prob": 0.65,
            "fair_exit_price": 0.665,
        },
    )
    assert r_create.status_code == 200

    def _boom(self, fresh: bool = False):
        raise RuntimeError("upstream outage")

    monkeypatch.setattr(ModelHub, "evaluate_open_positions", _boom)
    r_list = client.get("/api/ledger")
    assert r_list.status_code == 200
    payload = r_list.json()
    assert len(payload["trades"]) == 1
    assert payload["guidance_error"] is not None
    assert "RuntimeError" in payload["guidance_error"]


def test_log_trade_pair_is_atomic_on_invalid_second_leg(tmp_path: Path) -> None:
    db = Database(tmp_path / "atomic.sqlite3")
    leg_ok = {
        "created_at": "2026-11-21 12:00",
        "sport": "NBA",
        "event_name": "NYK @ BOS",
        "market_type": "Moneyline",
        "selection_label": "BOS ML",
        "side": "HOME",
        "venue": "DRAFTKINGS",
        "bet_format": "SPORTSBOOK",
        "odds_american": -110,
        "stake": 50.0,
        "model_prob": 0.55,
    }
    leg_bad = {
        **leg_ok,
        "selection_label": "NYK ML",
        "side": "AWAY",
        "venue": "FANDUEL",
        "odds_american": 50,  # invalid American odds in (-100, 100)
    }
    try:
        db.log_trade_pair(leg_ok, leg_bad)
        assert False, "Expected ValueError on invalid second leg"
    except ValueError:
        pass
    assert db.list_trades() == []


def test_open_position_prop_matching_does_not_cross_players(tmp_path: Path, monkeypatch) -> None:
    db = Database(tmp_path / "props_match.sqlite3")
    db.log_trade(
        created_at="2026-11-21 12:00",
        sport="NBA",
        event_name="NYK @ BOS",
        market_type="Prop: Points",
        selection_label="Jaylen Brown (BOS) Over 24.5 Points",
        side="YES",
        entry_price=0.52,
        contracts=5,
        model_prob=0.58,
        fair_exit_price=0.60,
    )
    hub = ModelHub(db=db)
    # Candidate list only contains a DIFFERENT player's Points prop in the same game
    monkeypatch.setattr(
        hub,
        "generate_picks",
        lambda **kwargs: [
            {
                "sport": "NBA",
                "short_name": "NYK @ BOS",
                "event_name": "NYK @ BOS",
                "market_type": "Prop: Points",
                "selection_label": "Jayson Tatum (BOS) Over 27.5 Points",
                "player_name": "Jayson Tatum",
                "yes_true_prob": 0.91,
                "game_state": "LIVE",
                "drivers": [],
            }
        ],
    )
    evaluated = hub.evaluate_open_positions(fresh=False)
    assert len(evaluated) == 1
    assert evaluated[0]["game_state"] == "UNKNOWN"
    assert abs(evaluated[0]["live_model_prob"] - 0.58) < 1e-6


def test_espn_nba_spread_sign_for_short_home_abbr() -> None:
    # Golden State ("GS") is the home favorite by 5.5 -> home_spread must be -5.5, not +5.5
    odds_list = [
        {
            "provider": {"name": "DraftKings"},
            "details": "GS -5.5",
            "overUnder": 228.5,
        }
    ]
    mkt = ESPNClient._extract_embedded_odds(odds_list, home_abbr="GSW")
    assert mkt["home_spread"] == -5.5


def test_odds_client_reads_meta_quota_key(tmp_path: Path) -> None:
    db = Database(tmp_path / "quota.sqlite3")
    db.set_cache(
        "odds_api:meta:quota",
        {"requests_remaining": 10, "requests_used": 490},
        ttl_seconds=3600,
    )
    client = OddsClient(api_key="test-key", db=db)
    status = client.get_quota_status()
    assert status["requests_remaining"] == 10
    assert status["low_quota_guard"] is True


def test_update_env_settings_handles_backslashes_and_newlines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("THE_ODDS_API_KEY=old\n", encoding="utf-8")
    update_env_settings({"THE_ODDS_API_KEY": r"abc\1def" + "\nINJECTED=1"}, env_path=env_file)
    text = env_file.read_text(encoding="utf-8")
    assert r"THE_ODDS_API_KEY=abc\1defINJECTED=1" in text
    assert "\nINJECTED=1" not in text

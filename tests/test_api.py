"""Integration tests for FastAPI endpoints and SQLite trade ledger settlement math."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.web.app import app


def test_api_endpoints_and_ledger_lifecycle(tmp_path: Path, monkeypatch) -> None:
    db_file = tmp_path / "test_dk.sqlite3"
    models_dir = tmp_path / "models"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("MODELS_DIR", str(models_dir))

    # Ensure cached CSVs exist in data/ without needing live network during tests
    NFLLoader().load_schedules()
    NFLLoader().load_player_logs()
    NBALoader().load_games()
    NBALoader().load_player_logs()

    client = TestClient(app)

    # 1. Index HTML loads
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "DraftKings Predictions Model" in r_idx.text

    # 1b. Config endpoint returns bankroll & updates cleanly
    monkeypatch.setenv("BANKROLL", "25.00")
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    assert abs(r_cfg.json()["bankroll"] - 25.0) < 1e-4

    r_cfg_save = client.post("/api/config", json={"bankroll": 35.50})
    assert r_cfg_save.status_code == 200
    assert abs(r_cfg_save.json()["bankroll"] - 35.50) < 1e-4

    # 2. Evaluate contract endpoint
    r_eval = client.post(
        "/api/evaluate-contract",
        json={
            "yes_true_prob": 0.70,
            "side": "YES",
            "contract_price": 0.60,
            "fee_per_contract": 0.015,
            "bankroll": 1000.0,
            "min_win_prob": 0.60,
            "min_net_edge": 0.025,
        },
    )
    assert r_eval.status_code == 200
    ev = r_eval.json()
    assert ev["confidence_tier"] == "A+"
    assert ev["recommended_contracts"] > 0

    # 3. Picks endpoint
    r_picks = client.get("/api/picks?sport=all&include_passes=true")
    assert r_picks.status_code == 200
    picks_payload = r_picks.json()
    assert picks_payload["count"] > 0

    # 4. Ledger lifecycle: Create -> Settle WON -> Create -> Settle SOLD early
    r_create = client.post(
        "/api/ledger",
        json={
            "sport": "NBA",
            "event_name": "NYK @ BOS",
            "market_type": "Moneyline",
            "selection_label": "BOS to Win",
            "side": "YES",
            "entry_price": 0.60,
            "fee_per_contract": 0.015,
            "contracts": 20,
            "model_prob": 0.69,
            "fair_exit_price": 0.695,
        },
    )
    assert r_create.status_code == 200
    trade1 = r_create.json()
    assert abs(trade1["total_cost"] - 12.30) < 1e-4  # 20 * $0.615 = $12.30

    # Settle WON ($1.00 * 20 = $20.00 -> +$7.70 net profit)
    r_won = client.patch(
        f"/api/ledger/{trade1['id']}",
        json={"status": "WON", "closing_price": 0.65},
    )
    assert r_won.status_code == 200
    assert abs(r_won.json()["pnl"] - 7.70) < 1e-2

    # Create second trade and sell early mid-game at 78c with 1.5c exit fee
    r_create2 = client.post(
        "/api/ledger",
        json={
            "sport": "NFL",
            "event_name": "BUF @ KC",
            "market_type": "Prop: Passing Yards",
            "selection_label": "Patrick Mahomes Over 262.5 Passing Yards",
            "side": "YES",
            "entry_price": 0.52,
            "fee_per_contract": 0.015,
            "contracts": 10,
            "model_prob": 0.64,
            "fair_exit_price": 0.648,
        },
    )
    trade2 = r_create2.json()  # total_cost = 10 * 0.535 = $5.35
    r_sold = client.patch(
        f"/api/ledger/{trade2['id']}",
        json={
            "status": "SOLD",
            "exit_or_settle_price": 0.78,
            "exit_fee_per_contract": 0.015,
        },
    )
    assert r_sold.status_code == 200
    # Proceeds = 10 * (0.78 - 0.015) = $7.65; PnL = 7.65 - 5.35 = +$2.30
    assert abs(r_sold.json()["pnl"] - 2.30) < 1e-2

    r_led = client.get("/api/ledger?bankroll=1000")
    summary = r_led.json()["summary"]
    assert abs(summary["realized_pnl"] - 10.00) < 1e-2
    assert summary["hit_rate"] == 1.0
    assert summary["avg_clv_cents"] == 5.0  # 65c closing - 60c entry = +5.0c CLV

    # 5. Guide endpoint
    r_guide = client.get("/api/guide")
    assert r_guide.status_code == 200
    assert "Limit Orders" in r_guide.json()["markdown"]

    # 6. Healthcheck and UI-first model maintenance endpoints
    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_train = client.post("/api/train?sport=nba")
    assert r_train.status_code == 200
    assert "nba_game" in r_train.json()["trained"]


def test_remote_passcode_auth_and_data_dir_persistence(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "remote_volume"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_PASSCODE", "secret-dkex-2026")

    from src.cli import build_parser
    from src.config import Settings
    from src.storage.db import Database

    settings = Settings.from_env()
    assert settings.db_path == data_dir / "dk_model.sqlite3"
    assert settings.models_dir == data_dir / "models"

    # Verify CLI parser defaults to remote-ready host/port without subcommands
    parsed = build_parser().parse_args(["--host", "0.0.0.0", "--port", "8080"])
    assert parsed.host == "0.0.0.0"
    assert parsed.port == 8080

    # Verify SQLite app_settings roundtrip
    db = Database(settings.db_path)
    db.set_app_settings({"KELLY_FRACTION": "0.30"})
    assert db.get_app_settings()["KELLY_FRACTION"] == "0.30"

    client = TestClient(app)

    # Public routes remain accessible for load-balancer probes and HTML shell
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200

    st_before = client.get("/api/auth/status").json()
    assert st_before == {"auth_enabled": True, "authenticated": False}

    # Protected API routes reject unauthenticated access
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/ledger").status_code == 401

    # Bearer token works for headless/remote API calls
    r_bearer = client.get("/api/config", headers={"Authorization": "Bearer secret-dkex-2026"})
    assert r_bearer.status_code == 200

    # Wrong passcode rejected
    r_bad = client.post("/api/auth/login", json={"passcode": "wrong-pass"})
    assert r_bad.status_code == 401

    # Correct passcode sets session cookie and unlocks API
    r_ok = client.post("/api/auth/login", json={"passcode": "secret-dkex-2026"})
    assert r_ok.status_code == 200
    assert client.get("/api/auth/status").json()["authenticated"] is True

    # Masked API key & extended config save
    r_save = client.post(
        "/api/config",
        json={"kelly_fraction": 0.20, "the_odds_api_key": "abcdef123456"},
    )
    assert r_save.status_code == 200
    cfg_payload = r_save.json()
    assert cfg_payload["odds_api_connected"] is True
    assert cfg_payload["odds_api_key_masked"] == "••••3456"
    assert "abcdef" not in cfg_payload["odds_api_key_masked"]

    # Logout re-locks the session
    client.post("/api/auth/logout")
    assert client.get("/api/config").status_code == 401

    # Static assets always serve cleanly even when locked
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


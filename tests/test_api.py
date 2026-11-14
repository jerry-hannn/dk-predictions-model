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

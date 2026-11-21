"""Integration tests for Traditional DK/FD toggle, multi-book extraction, strategy APIs, and unified ledger migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from src.data.espn_client import ESPNClient
from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.data.odds_client import OddsClient
from src.storage.db import Database
from src.web.app import app


def _sample_multi_book_odds_event() -> dict:
    """Multi-book The Odds API v4 fixture with DraftKings, FanDuel, and sharp books (LowVig + BetOnline)."""
    return {
        "id": "odds-evt-nba-1",
        "sport_key": "basketball_nba",
        "commence_time": "2026-11-22T00:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "New York Knicks",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": 115},
                            {"name": "New York Knicks", "price": -135},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -110, "point": -2.5},
                            {"name": "New York Knicks", "price": -110, "point": 2.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 224.5},
                            {"name": "Under", "price": -110, "point": 224.5},
                        ],
                    },
                ],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -105},
                            {"name": "New York Knicks", "price": -105},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -110, "point": -3.5},
                            {"name": "New York Knicks", "price": -110, "point": 3.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 226.5},
                            {"name": "Under", "price": -110, "point": 226.5},
                        ],
                    },
                ],
            },
            {
                "key": "lowvig",
                "title": "LowVig.ag",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -145},
                            {"name": "New York Knicks", "price": 130},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -105, "point": -3.0},
                            {"name": "New York Knicks", "price": -105, "point": 3.0},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -105, "point": 225.5},
                            {"name": "Under", "price": -105, "point": 225.5},
                        ],
                    },
                ],
            },
            {
                "key": "betonlineag",
                "title": "BetOnline.ag",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -140},
                            {"name": "New York Knicks", "price": 125},
                        ],
                    },
                ],
            },
        ],
    }


def test_odds_client_extracts_dk_fd_and_sharp_only_consensus() -> None:
    evt = _sample_multi_book_odds_event()
    cons = OddsClient.extract_consensus([evt], "Boston Celtics", "New York Knicks")
    assert cons is not None

    assert cons["has_draftkings"] is True
    assert cons["has_fanduel"] is True
    assert cons["sharp_books_count"] == 2

    # DraftKings exact quotes preserved
    assert cons["dk_home_ml"] == 115
    assert cons["dk_away_ml"] == -135
    assert cons["dk_home_spread"] == -2.5
    assert cons["dk_total_line"] == 224.5

    # FanDuel exact quotes preserved
    assert cons["fd_home_ml"] == -105
    assert cons["fd_away_ml"] == -105
    assert cons["fd_home_spread"] == -3.5
    assert cons["fd_total_line"] == 226.5

    # Sharp-only consensus excludes DK & FD (~57.5% Boston win prob from LowVig/BetOnline)
    assert cons["sharp_only_home_win_prob"] is not None
    assert cons["sharp_only_home_win_prob"] > 0.55
    assert cons["sharp_only_home_spread"] == -3.0
    assert cons["sharp_only_total_line"] == 225.5

    # Multi-book player prop extraction preserves DK and FD side-by-side
    prop_payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "point": 26.5, "price": -110},
                            {"name": "Under", "description": "Jayson Tatum", "point": 26.5, "price": -110},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "point": 27.5, "price": 102},
                            {"name": "Under", "description": "Jayson Tatum", "point": 27.5, "price": -125},
                        ],
                    }
                ],
            },
        ]
    }
    props = OddsClient.extract_player_prop_lines("NBA", prop_payload)
    key = ("jayson tatum", "points")
    assert key in props
    assert props[key]["dk_line"] == 26.5
    assert props[key]["fd_line"] == 27.5
    assert props[key]["dk_over_odds"] == -110
    assert props[key]["fd_under_odds"] == -125
    assert "draftkings" in props[key]["books"]
    assert "fanduel" in props[key]["books"]


def test_legacy_dkex_ledger_migration_and_multi_venue_accounting(tmp_path: Path) -> None:
    """Existing legacy DKeX SQLite table migrates cleanly without losing rows and supports DK/FD trades."""
    db_file = tmp_path / "legacy_dk.sqlite3"
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                sport TEXT NOT NULL,
                event_name TEXT NOT NULL,
                market_type TEXT NOT NULL,
                selection_label TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('YES', 'NO')),
                entry_price REAL NOT NULL,
                fee_per_contract REAL NOT NULL DEFAULT 0.015,
                contracts INTEGER NOT NULL,
                total_cost REAL NOT NULL,
                model_prob REAL NOT NULL,
                fair_exit_price REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'WON', 'LOST', 'SOLD')),
                exit_or_settle_price REAL,
                closing_price REAL,
                pnl REAL,
                notes TEXT DEFAULT ''
            );
            """
        )
        conn.execute(
            """
            INSERT INTO trades (
                created_at, sport, event_name, market_type, selection_label, side,
                entry_price, fee_per_contract, contracts, total_cost, model_prob,
                fair_exit_price, status, exit_or_settle_price, closing_price, pnl, notes
            ) VALUES (
                '2026-11-20 18:00', 'NBA', 'NYK @ BOS', 'Moneyline', 'BOS to Win', 'YES',
                0.60, 0.015, 10, 6.15, 0.68, 0.685, 'WON', 1.0, 0.64, 3.85, 'Legacy DKeX trade'
            );
            """
        )

    db = Database(db_file)
    trades = db.list_trades()
    assert len(trades) == 1
    assert trades[0]["venue"] == "DKEX"
    assert trades[0]["bet_format"] == "CONTRACT"
    assert abs(trades[0]["pnl"] - 3.85) < 1e-4

    # Log atomic DK <-> FD arbitrage pair
    pair = db.log_trade_pair(
        {
            "created_at": "2026-11-21 19:00",
            "sport": "NBA",
            "event_name": "NYK @ BOS",
            "market_type": "Moneyline",
            "selection_label": "BOS ML",
            "side": "HOME",
            "venue": "DRAFTKINGS",
            "bet_format": "SPORTSBOOK",
            "odds_american": 115,
            "stake": 48.0,
            "model_prob": 0.50,
            "strategy_tag": "ARB",
        },
        {
            "created_at": "2026-11-21 19:00",
            "sport": "NBA",
            "event_name": "NYK @ BOS",
            "market_type": "Moneyline",
            "selection_label": "NYK ML",
            "side": "AWAY",
            "venue": "FANDUEL",
            "bet_format": "SPORTSBOOK",
            "odds_american": -105,
            "stake": 52.0,
            "model_prob": 0.50,
            "strategy_tag": "ARB",
        },
    )
    assert len(pair) == 2
    assert pair[0]["group_id"] == pair[1]["group_id"]
    assert pair[0]["venue"] == "DRAFTKINGS"
    assert pair[1]["venue"] == "FANDUEL"

    # Settle DK leg WON (+$55.20) & FD leg LOST (-$52.00) -> net guaranteed arb profit +$3.20
    dk_won = db.update_trade(pair[0]["id"], status="WON", closing_odds_american=105)
    fd_lost = db.update_trade(pair[1]["id"], status="LOST", closing_odds_american=-110)
    assert dk_won is not None and abs(dk_won["pnl"] - 55.20) < 1e-2
    assert fd_lost is not None and abs(fd_lost["pnl"] - (-52.00)) < 1e-2
    assert dk_won["closing_odds_american"] == 105

    # Log Bonus Bet ($0 cash risk, profit-only payout): $50 at +380 -> WON yields +$190, LOST yields $0
    bb = db.log_trade(
        created_at="2026-11-21 19:05",
        sport="NFL",
        event_name="BUF @ KC",
        market_type="Moneyline",
        selection_label="Underdog ML (Bonus Bet)",
        side="AWAY",
        venue="DRAFTKINGS",
        bet_format="SPORTSBOOK",
        odds_american=380,
        stake=50.0,
        is_bonus_bet=True,
        strategy_tag="BONUS_BET",
    )
    assert bb["total_cost"] == 0.0
    assert abs(bb["potential_payout"] - 190.0) < 1e-2
    bb_lost = db.update_trade(bb["id"], status="LOST")
    assert bb_lost is not None and bb_lost["pnl"] == 0.0
    bb_won = db.update_trade(bb["id"], status="WON")
    assert bb_won is not None and abs(bb_won["pnl"] - 190.0) < 1e-2

    # Test PUSH, VOID, CASHOUT statuses
    t_push = db.log_trade(
        created_at="2026-11-21 19:10",
        sport="NFL",
        event_name="BUF @ KC",
        market_type="Spread",
        selection_label="KC -3",
        side="HOME",
        venue="FANDUEL",
        bet_format="SPORTSBOOK",
        odds_american=-110,
        stake=50.0,
        strategy_tag="EV",
    )
    pushed = db.update_trade(t_push["id"], status="PUSH")
    assert pushed is not None and pushed["pnl"] == 0.0
    cashed = db.update_trade(t_push["id"], status="CASHOUT", cashout_amount=65.0)
    assert cashed is not None and abs(cashed["pnl"] - 15.0) < 1e-2

    summary = db.ledger_summary(starting_bankroll=1000.0, dk_bankroll=500.0, fd_bankroll=500.0)
    assert "by_venue" in summary
    assert summary["by_venue"]["DRAFTKINGS"]["starting_bankroll"] == 500.0
    assert summary["by_venue"]["FANDUEL"]["starting_bankroll"] == 500.0
    assert abs(summary["by_venue"]["DKEX"]["realized_pnl"] - 3.85) < 1e-2


def test_sportsbook_api_toggle_slate_analyzers_and_pair_ledger(tmp_path: Path, monkeypatch) -> None:
    """End-to-end API verification for config toggle, /api/sportsbook/* routes, and /api/ledger/pair."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sb_test.sqlite3"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    for key in (
        "TRADITIONAL_SPORTSBOOKS_ENABLED",
        "DK_BANKROLL",
        "FD_BANKROLL",
        "ARB_STAKE_ROUNDING",
        "MIN_ARB_ROI",
        "MIN_SB_EV_EDGE",
        "INCLUDE_DKEX_CROSS_ARB",
    ):
        monkeypatch.delenv(key, raising=False)

    NFLLoader().load_schedules()
    NBALoader().load_games()

    client = TestClient(app)

    # 1. Default config has traditional_sportsbooks_enabled == False
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert cfg["traditional_sportsbooks_enabled"] is False
    assert cfg["arb_stake_rounding"] == 1.0
    assert cfg["include_dkex_cross_arb"] is True

    # 2. Toggle Traditional Sportsbooks ON & persist per-book bankrolls
    r_save = client.post(
        "/api/config",
        json={
            "traditional_sportsbooks_enabled": True,
            "dk_bankroll": 600.0,
            "fd_bankroll": 400.0,
            "arb_stake_rounding": 5.0,
            "min_arb_roi": 0.005,
            "include_dkex_cross_arb": True,
        },
    )
    assert r_save.status_code == 200
    saved = r_save.json()
    assert saved["traditional_sportsbooks_enabled"] is True
    assert saved["dk_bankroll"] == 600.0
    assert saved["fd_bankroll"] == 400.0
    assert saved["arb_stake_rounding"] == 5.0

    # 3. Sportsbook Evaluator endpoint (+EV & Profit Boost)
    r_eval = client.post(
        "/api/sportsbook/evaluate",
        json={
            "true_prob": 0.58,
            "dk_odds": -105,
            "fd_odds": -115,
            "bankroll": 1000.0,
            "profit_boost_pct": 0.25,
        },
    )
    assert r_eval.status_code == 200
    ev = r_eval.json()
    assert ev["best"]["book"] == "DRAFTKINGS"
    assert ev["best"]["ev_pct"] > 0.20
    assert ev["line_shopping_gain_pct"] > 0.0

    # 4. Sportsbook Arb & Hedge Calculator (DK <-> FD, DKeX <-> FD, and Bonus Bet Converter)
    r_arb = client.post(
        "/api/sportsbook/arb-calc",
        json={
            "calc_mode": "arb",
            "book_a": "DRAFTKINGS",
            "odds_a": 115,
            "book_b": "FANDUEL",
            "odds_b": -105,
            "total_stake": 100.0,
            "rounding_step": 1.0,
        },
    )
    assert r_arb.status_code == 200
    arb_res = r_arb.json()
    assert arb_res["is_arb"] is True
    assert arb_res["min_guaranteed_profit"] > 0.0

    r_cross = client.post(
        "/api/sportsbook/arb-calc",
        json={
            "calc_mode": "arb",
            "book_a": "DKEX",
            "dkex_ask_a": 0.42,
            "dkex_fee_a": 0.015,
            "book_b": "FANDUEL",
            "odds_b": -110,
            "total_stake": 100.0,
            "rounding_step": 1.0,
        },
    )
    assert r_cross.status_code == 200
    assert r_cross.json()["is_arb"] is True
    assert r_cross.json()["arb_type"] == "DKEX_CROSS"

    r_bb = client.post(
        "/api/sportsbook/arb-calc",
        json={
            "calc_mode": "bonus_bet",
            "book_a": "DRAFTKINGS",
            "odds_a": 380,
            "book_b": "FANDUEL",
            "odds_b": -440,
            "bonus_amount": 50.0,
        },
    )
    assert r_bb.status_code == 200
    assert r_bb.json()["retention_pct"] >= 0.70

    # 5. Sportsbook Slate endpoint returns structured 5-strategy response
    r_slate = client.get("/api/sportsbook/slate?sport=nba&include_passes=true&live_props=false")
    assert r_slate.status_code == 200
    slate = r_slate.json()
    for key in ("picks", "arbitrage", "middles", "promos", "steam", "strategy_counts", "ledger_summary"):
        assert key in slate
    assert "bonus_bet" in slate["promos"]
    assert "profit_boost" in slate["promos"]
    assert "low_hold" in slate["promos"]

    # 6. Sportsbook Backtest Metrics endpoint loads verified metrics
    r_bt = client.get("/api/sportsbook/backtest-metrics")
    assert r_bt.status_code == 200
    bt = r_bt.json()
    assert bt["chronological_verified"] is True
    assert bt["combined"]["verdicts"]["arbitrage"] == "VERIFIED_PROFITABLE"
    assert "stress_regimes" in bt

    # 7. Atomic Pair Ledger endpoint + venue filter
    r_pair = client.post(
        "/api/ledger/pair",
        json={
            "leg_a": {
                "sport": "NBA",
                "event_name": "NYK @ BOS",
                "market_type": "Moneyline",
                "selection_label": "BOS ML",
                "side": "HOME",
                "venue": "DRAFTKINGS",
                "bet_format": "SPORTSBOOK",
                "odds_american": 115,
                "stake": 48.0,
                "model_prob": 0.50,
                "strategy_tag": "ARB",
            },
            "leg_b": {
                "sport": "NBA",
                "event_name": "NYK @ BOS",
                "market_type": "Moneyline",
                "selection_label": "NYK ML",
                "side": "AWAY",
                "venue": "FANDUEL",
                "bet_format": "SPORTSBOOK",
                "odds_american": -105,
                "stake": 52.0,
                "model_prob": 0.50,
                "strategy_tag": "ARB",
            },
        },
    )
    assert r_pair.status_code == 200
    pair_payload = r_pair.json()
    assert len(pair_payload["trades"]) == 2
    assert pair_payload["group_id"]

    r_dk_only = client.get("/api/ledger?venue=DRAFTKINGS")
    assert r_dk_only.status_code == 200
    dk_trades = r_dk_only.json()["trades"]
    assert len(dk_trades) == 1
    assert dk_trades[0]["venue"] == "DRAFTKINGS"
    assert r_dk_only.json()["summary"]["by_venue"]["DRAFTKINGS"]["starting_bankroll"] == 600.0


def test_sportsbook_slate_surfaces_all_five_strategies_from_multibook_odds(
    tmp_path: Path, monkeypatch
) -> None:
    """Inject a live multi-book DK+FD+Sharp NBA event and verify all 5 strategies surface cleanly."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sb_multibook.sqlite3"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")
    monkeypatch.setenv("TRADITIONAL_SPORTSBOOKS_ENABLED", "true")

    fake_game = {
        "event_id": "401999999",
        "sport": "NBA",
        "date": "2026-11-22",
        "commence_time": "2026-11-22T00:00:00Z",
        "event_name": "New York Knicks at Boston Celtics",
        "short_name": "NYK @ BOS",
        "start_time": "2026-11-22T00:00:00Z",
        "home_abbr": "BOS",
        "away_abbr": "NYK",
        "home_name": "Boston Celtics",
        "away_name": "New York Knicks",
        "state": "PRE",
        "home_score": 0,
        "away_score": 0,
        "period": 0,
        "clock_display": "",
        "status_detail": "7:00 PM ET",
        "remaining_minutes": 48.0,
        "market": {},
        "situation": {},
    }
    from src.engine.recommender import ModelHub

    monkeypatch.setattr(
        ModelHub,
        "_get_upcoming_events",
        lambda self, sport, target_date=None, mode="all", force_refresh=False: [fake_game]
        if sport.upper() == "NBA"
        else [],
    )
    monkeypatch.setattr(
        ESPNClient,
        "fetch_league_injuries",
        lambda self, sport, force_refresh=False: {},
    )
    monkeypatch.setattr(
        OddsClient,
        "fetch_game_odds",
        lambda self, sport, force_refresh=False, live_mode=False: [_sample_multi_book_odds_event()]
        if sport.upper() == "NBA"
        else [],
    )

    client = TestClient(app)
    r = client.get("/api/sportsbook/slate?sport=nba&include_passes=true&live_props=false&arb_rounding=1")
    assert r.status_code == 200
    slate = r.json()

    assert slate["has_fanduel_odds"] is True
    assert len(slate["picks"]) >= 3  # Moneyline + Spread + Total
    assert len(slate["arbitrage"]) >= 1  # DK BOS +115 vs FD NYK -105
    assert len(slate["middles"]) >= 2    # Spread (-2.5/+3.5) + Total (224.5/226.5)
    assert len(slate["steam"]) >= 1      # DK BOS +115 lagging LowVig/BetOnline -145/-140
    assert len(slate["promos"]["profit_boost"]) >= 1
    assert any(a["arb_type"] == "DK_FD" and a["is_guaranteed_after_rounding"] for a in slate["arbitrage"])


def test_prop_book_merging_multiline_fanduel_and_no_fake_dk_fabrication(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify multi-line FD prop pairing, ESPN DK + OddsAPI FD merging, and zero fake DK -110 cards."""
    # 1. Multi-line FD payload (alt line + main line) must pair each point cleanly and select main line
    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "point": 24.5, "price": -180},
                            {"name": "Under", "description": "Jayson Tatum", "point": 24.5, "price": 140},
                            {"name": "Over", "description": "Jayson Tatum", "point": 26.5, "price": -108},
                            {"name": "Under", "description": "Jayson Tatum", "point": 26.5, "price": -112},
                        ],
                    }
                ],
            },
            {
                "key": "betmgm",
                "markets": [
                    {
                        "key": "player_threes",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "point": 3.5, "price": -115},
                            {"name": "Under", "description": "Jayson Tatum", "point": 3.5, "price": -115},
                        ],
                    }
                ],
            },
        ]
    }
    parsed = OddsClient.extract_player_prop_lines("NBA", payload)
    pts = parsed[("jayson tatum", "points")]
    assert pts["source"] == "FanDuel"
    assert pts["fd_line"] == 26.5
    assert pts["fd_over_odds"] == -108
    assert pts["fd_under_odds"] == -112
    assert "fanduel" in pts["books"]

    # 2. End-to-end slate: ESPN supplies DK points, Odds API supplies FD points + BetMGM-only threes.
    # Result: points card shows BOTH DK and FD; BetMGM-only & unpriced Model Ref Lines are NOT fabricated as DK -110!
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sb_props_merge.sqlite3"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")
    monkeypatch.setenv("TRADITIONAL_SPORTSBOOKS_ENABLED", "true")

    from datetime import datetime, timedelta, timezone
    from src.engine.recommender import ModelHub
    from src.models.player_props import PlayerPropProjection, PlayerPropsEngine

    soon_iso = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    fake_game = {
        "event_id": "401999998",
        "sport": "NBA",
        "date": "2026-11-22",
        "commence_time": soon_iso,
        "event_name": "New York Knicks at Boston Celtics",
        "short_name": "NYK @ BOS",
        "start_time": soon_iso,
        "home_abbr": "BOS",
        "away_abbr": "NYK",
        "home_name": "Boston Celtics",
        "away_name": "New York Knicks",
        "state": "PRE",
        "home_score": 0,
        "away_score": 0,
        "period": 0,
        "clock_display": "",
        "status_detail": "7:00 PM ET",
        "remaining_minutes": 48.0,
        "market": {},
        "situation": {},
    }

    espn_dk_prop = {
        "401999998": {
            ("jayson tatum", "points"): {
                "line": 25.5,
                "over_prob": 0.50,
                "dk_over_prob": 0.50,
                "source": "DraftKings",
                "dk_line": 25.5,
                "dk_over_odds": -110,
                "dk_under_odds": -110,
                "books": {
                    "draftkings": {
                        "line": 25.5,
                        "over_odds": -110,
                        "under_odds": -110,
                        "over_prob": 0.50,
                    }
                },
            }
        }
    }

    monkeypatch.setattr(
        ModelHub,
        "_get_upcoming_events",
        lambda self, sport, target_date=None, mode="all", force_refresh=False: [fake_game]
        if sport.upper() == "NBA"
        else [],
    )
    monkeypatch.setattr(ESPNClient, "fetch_league_injuries", lambda self, sport, force_refresh=False: {})
    monkeypatch.setattr(ESPNClient, "fetch_slate_player_props", lambda self, sport, events, force_refresh=False: espn_dk_prop if sport.upper() == "NBA" else {})
    monkeypatch.setattr(OddsClient, "fetch_game_odds", lambda self, sport, force_refresh=False, live_mode=False: [_sample_multi_book_odds_event()] if sport.upper() == "NBA" else [])
    monkeypatch.setattr(OddsClient, "fetch_event_player_props", lambda self, sport, eid, force_refresh=False, live_mode=False: parsed if sport.upper() == "NBA" else {})

    def _fake_nba_props(self, **kwargs):
        live_mkts = kwargs.get("live_prop_markets") or {}
        tatum_pts = live_mkts.get(("jayson tatum", "points"))
        tatum_3s = live_mkts.get(("jayson tatum", "fg3m"))
        return [
            PlayerPropProjection(
                sport="NBA",
                player_name="Jayson Tatum",
                team="BOS",
                opponent="NYK",
                prop_type="points",
                prop_label="Points",
                line_value=25.5,
                projected_mean=28.0,
                projected_std=6.5,
                over_prob=0.65,
                under_prob=0.35,
                drivers=["Proj 28.0 vs 25.5"],
                injury_clarity=1.0,
                market_over_prob=0.50,
                line_source=tatum_pts["source"] if tatum_pts else "Model Ref Line",
                book_lines=tatum_pts.get("books") if tatum_pts else None,
            ),
            # BetMGM-only prop -> must NOT become fake DK -110
            PlayerPropProjection(
                sport="NBA",
                player_name="Jayson Tatum",
                team="BOS",
                opponent="NYK",
                prop_type="fg3m",
                prop_label="3-Pointers Made",
                line_value=3.5,
                projected_mean=2.0,
                projected_std=1.5,
                over_prob=0.25,
                under_prob=0.75,
                drivers=["Proj 2.0 vs 3.5"],
                injury_clarity=1.0,
                market_over_prob=0.50,
                line_source=tatum_3s["source"] if tatum_3s else "Model Ref Line",
                book_lines=tatum_3s.get("books") if tatum_3s else None,
            ),
            # Unpriced Model Ref Line prop -> must NOT become fake DK -110
            PlayerPropProjection(
                sport="NBA",
                player_name="Jaylen Brown",
                team="BOS",
                opponent="NYK",
                prop_type="fg3m",
                prop_label="3-Pointers Made",
                line_value=2.5,
                projected_mean=1.4,
                projected_std=1.2,
                over_prob=0.25,
                under_prob=0.75,
                drivers=["Proj 1.4 vs 2.5 (Model Ref Line)"],
                injury_clarity=1.0,
                market_over_prob=0.50,
                line_source="Model Ref Line",
                book_lines=None,
            ),
        ]

    monkeypatch.setattr(PlayerPropsEngine, "project_nba_game_props", _fake_nba_props)

    client = TestClient(app)
    r = client.get("/api/sportsbook/slate?sport=nba&market=props&include_passes=true&live_props=true")
    assert r.status_code == 200
    slate = r.json()

    # Only the genuine DK+FD Jayson Tatum Points prop should appear (zero fake Model Ref / BetMGM-only DK cards)
    assert len(slate["picks"]) == 1
    p = slate["picks"][0]
    assert p["player_name"] == "Jayson Tatum"
    assert p["line_source"] == "DK + FD"
    assert p["dk_line"] == 25.5
    assert p["fd_line"] == 26.5
    assert p["dk_odds"] is not None
    assert p["fd_odds"] is not None
    # And because DK=25.5 vs FD=26.5, a Player Prop Middle is automatically detected!
    assert any("Points" in m["market_type"] for m in slate["middles"])


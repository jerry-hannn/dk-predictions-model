"""Unit and integration tests for real-time pre-game and live in-play betting pipelines."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.data.espn_client import ESPNClient, compute_game_clock_progress
from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.data.odds_client import OddsClient
from src.engine.recommender import ModelHub
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import PlayerPropsEngine
from src.storage.db import Database
from src.web.app import app


def test_compute_game_clock_progress_states_and_overtime() -> None:
    assert compute_game_clock_progress("NBA", "PRE", 0, 0.0) == (0.0, 48.0)
    assert compute_game_clock_progress("NFL", "POST", 4, 0.0) == (1.0, 0.0)
    assert compute_game_clock_progress("NBA", "IN", 2, 0.0, "Halftime") == (0.5, 24.0)
    assert compute_game_clock_progress("NFL", "IN", 2, 0.0, "Halftime") == (0.5, 30.0)

    el_nba, rem_nba = compute_game_clock_progress("NBA", "IN", 4, 360.0, "6:00 - 4th")
    assert abs(el_nba - 0.875) < 1e-4
    assert abs(rem_nba - 6.0) < 1e-4

    el_nfl, rem_nfl = compute_game_clock_progress("NFL", "IN", 3, 450.0, "7:30 - 3rd")
    assert abs(el_nfl - 0.625) < 1e-4
    assert abs(rem_nfl - 22.5) < 1e-4

    el_ot, rem_ot = compute_game_clock_progress("NBA", "IN", 5, 180.0, "3:00 - OT")
    assert el_ot >= 0.90
    assert abs(rem_ot - 3.0) < 1e-4


def test_espn_2026_nested_draftkings_odds_and_live_boxscore_parser() -> None:
    odds_list = [
        {
            "provider": {"id": "100", "name": "Draft Kings", "displayName": "DraftKings"},
            "moneyline": {
                "home": {"open": {"odds": "-135"}, "close": {"odds": "-165"}},
                "away": {"open": {"odds": "+114"}, "close": {"odds": "+140"}},
            },
            "pointSpread": {
                "home": {
                    "open": {"line": "-2.5", "odds": "-110"},
                    "close": {"line": "-3.5", "odds": "-108"},
                },
                "away": {
                    "open": {"line": "+2.5", "odds": "-110"},
                    "close": {"line": "+3.5", "odds": "-112"},
                },
            },
            "total": {
                "over": {
                    "open": {"line": "o224.5", "odds": "-110"},
                    "close": {"line": "o228.5", "odds": "-115"},
                },
                "under": {
                    "open": {"line": "u224.5", "odds": "-110"},
                    "close": {"line": "u228.5", "odds": "-105"},
                },
            },
        }
    ]
    parsed = ESPNClient._extract_embedded_odds(odds_list, "BOS")
    assert parsed["line_source"] == "DraftKings"
    assert parsed["dk_home_ml"] == -165
    assert parsed["dk_home_spread"] == -3.5
    assert parsed["spread_move"] == -1.0  # moved from -2.5 to -3.5
    assert parsed["dk_total_line"] == 228.5
    assert parsed["total_move"] == 4.0  # moved from 224.5 to 228.5
    assert 0.58 < parsed["home_win_prob"] < 0.64
    assert 0.49 < parsed["home_cover_prob"] < 0.52
    assert 0.50 < parsed["over_prob"] < 0.54

    # Live NBA summary payload
    nba_raw = {
        "winprobability": [{"homeWinPercentage": 0.74}],
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "BOS"},
                    "statistics": [
                        {
                            "labels": ["MIN", "FG", "3PT", "FT", "REB", "AST", "PF", "PTS"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Jayson Tatum"},
                                    "starter": True,
                                    "active": True,
                                    "stats": ["18:30", "7-12", "3-6", "5-5", "6", "4", "2", "22"],
                                },
                                {
                                    "athlete": {"displayName": "Bench Ejected"},
                                    "starter": False,
                                    "ejected": True,
                                    "stats": ["4:00", "0-1", "0-0", "0-0", "0", "0", "1", "0"],
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    }
    summary = ESPNClient.parse_summary_payload("NBA", nba_raw)
    assert summary["live_win_prob"] == 0.74
    jt = summary["players"]["jayson tatum"]
    assert jt["minutes"] == 18.5
    assert jt["points"] == 22.0
    assert jt["fg3m"] == 3.0
    assert jt["pra"] == 32.0
    assert summary["players"]["bench ejected"]["active"] is False


def test_odds_client_consensus_dk_extraction_props_and_quota_guard(tmp_path: Path) -> None:
    events = [
        {
            "id": "ev_live_1",
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-11-12T02:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -150},
                                {"name": "New York Knicks", "price": 126},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Boston Celtics", "point": -3.5, "price": -105},
                                {"name": "New York Knicks", "point": 3.5, "price": -115},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 225.5, "price": -112},
                                {"name": "Under", "point": 225.5, "price": -108},
                            ],
                        },
                    ],
                },
                {
                    "key": "lowvig",
                    "last_update": "2026-11-12T02:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -190},
                                {"name": "New York Knicks", "price": 168},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Boston Celtics", "point": -4.5, "price": -108},
                                {"name": "New York Knicks", "point": 4.5, "price": -104},
                            ],
                        },
                    ],
                },
            ],
        }
    ]
    cons = OddsClient.extract_consensus(events, "Boston Celtics", "New York Knicks")
    assert cons is not None
    assert cons["dk_home_ml"] == -150
    assert cons["dk_home_spread"] == -3.5
    assert cons["home_spread"] == -3.5  # Anchored to DraftKings tradable spread
    assert cons["spread_dispersion"] == 1.0
    # Sharp LowVig (-190) pulls sharp_home_win_prob higher than DK's (-150) alone
    assert cons["sharp_home_win_prob"] > cons["dk_home_win_prob"]

    # Player prop extraction
    prop_payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"description": "Jayson Tatum", "name": "Over", "point": 28.5, "price": -114},
                            {"description": "Jayson Tatum", "name": "Under", "point": 28.5, "price": -106},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"description": "Jayson Tatum", "name": "Over", "point": 27.5, "price": -115},
                            {"description": "Jayson Tatum", "name": "Under", "point": 27.5, "price": -105},
                        ],
                    }
                ],
            },
        ]
    }
    props = OddsClient.extract_player_prop_lines("NBA", prop_payload)
    jt_pts = props[("jayson tatum", "points")]
    assert jt_pts["line"] == 27.5
    assert jt_pts["source"] == "DraftKings"
    assert 0.50 < jt_pts["dk_over_prob"] < 0.54

    # Low-quota guard returns cached stale odds without hitting network
    db = Database(tmp_path / "quota.sqlite3")
    db.set_cache("odds_api:NBA:game_odds", events, ttl_seconds=-10)  # expired cache
    db.set_cache(
        "odds_api:meta:quota",
        {"requests_remaining": 8, "requests_used": 492, "updated_at": 1.0},
        ttl_seconds=3600,
    )
    client = OddsClient(api_key="dummy_key_not_called", db=db)
    fallback_events = client.fetch_game_odds("NBA", force_refresh=False, live_mode=True)
    assert len(fallback_events) == 1
    assert fallback_events[0]["id"] == "ev_live_1"


def test_nfl_and_nba_live_in_play_conditioning_and_mean_reversion() -> None:
    nfl_df = NFLLoader().load_schedules()
    nba_df = NBALoader().load_games()

    nfl_model = NFLGameModel()
    nfl_model.fit(nfl_df)
    nba_model = NBAGameModel()
    nba_model.fit(nba_df)

    # NFL live score, clock, & red-zone possession monotonicity
    pre_nfl = nfl_model.predict_matchup(
        home_team="KC", away_team="BUF", spread_line=-2.5, total_line=48.5
    )
    live_lead = nfl_model.predict_matchup(
        home_team="KC",
        away_team="BUF",
        spread_line=-2.5,
        total_line=48.5,
        live_state={
            "state": "IN",
            "home_score": 28,
            "away_score": 14,
            "elapsed_fraction": 0.78,
            "remaining_minutes": 13.2,
            "possession_team": "KC",
            "is_red_zone": True,
            "status_detail": "13:12 - 4th",
        },
    )
    live_trail = nfl_model.predict_matchup(
        home_team="KC",
        away_team="BUF",
        spread_line=-2.5,
        total_line=48.5,
        live_state={
            "state": "IN",
            "home_score": 14,
            "away_score": 28,
            "elapsed_fraction": 0.78,
            "remaining_minutes": 13.2,
            "possession_team": "BUF",
            "is_red_zone": False,
            "status_detail": "13:12 - 4th",
        },
    )
    assert live_lead.is_live is True
    assert live_lead.home_win_prob > pre_nfl.home_win_prob > live_trail.home_win_prob
    assert live_lead.home_win_prob > 0.85
    assert live_trail.home_win_prob < 0.18

    # NBA live hot 1st-half total mean-reversion (136 pts in 1H shouldn't naively project 272!)
    pre_nba = nba_model.predict_matchup(home_team="BOS", away_team="NYK", total_line=224.5)
    live_hot_half = nba_model.predict_matchup(
        home_team="BOS",
        away_team="NYK",
        total_line=238.5,
        live_state={
            "state": "IN",
            "home_score": 70,
            "away_score": 66,
            "elapsed_fraction": 0.50,
            "remaining_minutes": 24.0,
            "status_detail": "Halftime",
        },
    )
    assert live_hot_half.is_live is True
    assert pre_nba.expected_total < live_hot_half.expected_total < 262.0


def test_live_player_props_conditioning_foul_trouble_blowout_and_lock() -> None:
    nfl_logs = NFLLoader().load_player_logs()
    nba_logs = NBALoader().load_player_logs()
    props_engine = PlayerPropsEngine()
    props_engine.fit(nfl_logs, nba_logs)

    pre_props = props_engine.project_nba_game_props(home_team="BOS", away_team="NYK")
    p_pre = next(p for p in pre_props if p.team == "BOS" and p.prop_type == "points")
    top_player = p_pre.player_name

    # Halftime hot start on exact same line -> higher Over probability
    live_box_hot = {
        top_player.lower(): {
            "player_name": top_player,
            "team": "BOS",
            "starter": True,
            "active": True,
            "did_not_play": False,
            "ejected": False,
            "minutes": 18.0,
            "points": round(p_pre.line_value * 0.72, 1),
            "rebounds": 4.0,
            "assists": 3.0,
            "fg3m": 2.0,
            "pra": round(p_pre.line_value * 0.72 + 7.0, 1),
            "fouls": 1.0,
        }
    }
    live_props_hot = props_engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        live_state={
            "state": "IN",
            "period": 2,
            "elapsed_fraction": 0.50,
            "remaining_minutes": 24.0,
            "home_score": 58,
            "away_score": 55,
            "status_detail": "Halftime",
        },
        live_boxscore=live_box_hot,
        live_prop_markets={
            (top_player.lower(), "points"): {
                "line": p_pre.line_value,
                "over_prob": 0.52,
                "dk_over_prob": 0.52,
                "source": "DraftKings",
            }
        },
    )
    p_hot = next(p for p in live_props_hot if p.player_name == top_player and p.prop_type == "points")
    assert p_hot.is_live is True
    assert p_hot.line_source == "DraftKings"
    assert p_hot.over_prob > p_pre.over_prob

    # Q4 30-point blowout haircuts remaining starter projection vs close Q4 game
    live_box_q4 = {
        top_player.lower(): {
            **live_box_hot[top_player.lower()],
            "minutes": 29.0,
            "points": 20.0,
        }
    }
    close_q4 = props_engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        live_state={
            "state": "IN",
            "period": 4,
            "elapsed_fraction": 0.79,
            "remaining_minutes": 10.0,
            "home_score": 92,
            "away_score": 90,
        },
        live_boxscore=live_box_q4,
    )
    blowout_q4 = props_engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        live_state={
            "state": "IN",
            "period": 4,
            "elapsed_fraction": 0.79,
            "remaining_minutes": 10.0,
            "home_score": 115,
            "away_score": 85,
        },
        live_boxscore=live_box_q4,
    )
    p_close = next(p for p in close_q4 if p.player_name == top_player and p.prop_type == "points")
    p_blow = next(p for p in blowout_q4 if p.player_name == top_player and p.prop_type == "points")
    assert p_blow.projected_mean < p_close.projected_mean

    # Final <2.5m lock returns empty prop list
    locked = props_engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        live_state={
            "state": "IN",
            "period": 4,
            "elapsed_fraction": 0.96,
            "remaining_minutes": 1.8,
            "home_score": 104,
            "away_score": 101,
        },
        live_boxscore=live_box_q4,
    )
    assert locked == []


def test_recommender_latency_lock_open_position_monitor_and_live_api(
    tmp_path: Path, monkeypatch
) -> None:
    db_file = tmp_path / "live_hub.sqlite3"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("BANKROLL", "25.00")

    hub = ModelHub(db=Database(db_file))
    hub.ensure_loaded()

    # Inject a synthetic live close endgame (<90s left) and a normal Q3 live game into ESPNClient
    fake_live_events = [
        {
            "event_id": "live_lock_999",
            "sport": "NBA",
            "event_name": "New York Knicks @ Boston Celtics",
            "short_name": "NYK @ BOS",
            "start_time": "2026-11-12T00:30Z",
            "state": "IN",
            "status": "STATUS_IN_PROGRESS",
            "status_detail": "0:55 - 4th",
            "period": 4,
            "clock_seconds": 55.0,
            "display_clock": "0:55",
            "elapsed_fraction": 0.981,
            "remaining_minutes": 0.92,
            "home_score": 106,
            "away_score": 104,
            "live_win_prob": 0.78,
            "possession_team": "BOS",
            "is_red_zone": False,
            "down_distance": "",
            "home_abbr": "BOS",
            "away_abbr": "NYK",
            "home_name": "Boston Celtics",
            "away_name": "New York Knicks",
            "indoor": True,
            "temp_f": 70.0,
            "wind_mph": 0.0,
            "market": {
                "line_source": "DraftKings",
                "home_win_prob": 0.62,
                "dk_home_ml": -160,
                "home_spread": -2.5,
                "home_cover_prob": 0.50,
                "total_line": 214.5,
                "over_prob": 0.50,
            },
        },
        {
            "event_id": "live_q3_888",
            "sport": "NBA",
            "event_name": "Los Angeles Lakers @ Golden State Warriors",
            "short_name": "LAL @ GSW",
            "start_time": "2026-11-12T03:00Z",
            "state": "IN",
            "status": "STATUS_IN_PROGRESS",
            "status_detail": "6:00 - 3rd",
            "period": 3,
            "clock_seconds": 360.0,
            "display_clock": "6:00",
            "elapsed_fraction": 0.625,
            "remaining_minutes": 18.0,
            "home_score": 82,
            "away_score": 68,
            "live_win_prob": 0.86,
            "possession_team": None,
            "is_red_zone": False,
            "down_distance": "",
            "home_abbr": "GSW",
            "away_abbr": "LAL",
            "home_name": "Golden State Warriors",
            "away_name": "Los Angeles Lakers",
            "indoor": True,
            "temp_f": 70.0,
            "wind_mph": 0.0,
            "market": {
                "line_source": "DraftKings",
                "home_win_prob": 0.74,
                "dk_home_ml": -285,
                "home_spread": -11.5,
                "home_cover_prob": 0.51,
                "total_line": 229.5,
                "over_prob": 0.50,
                "odds_age_seconds": 300.0,  # >240s triggers stale warning
            },
        },
    ]

    monkeypatch.setattr(
        ESPNClient,
        "fetch_scoreboard",
        lambda self, sport, target_date=None, force_refresh=False: (
            fake_live_events if sport.upper() == "NBA" else []
        ),
    )
    monkeypatch.setattr(
        ESPNClient,
        "fetch_live_boxscore",
        lambda self, sport, event_id, force_refresh=False: {
            "players": {},
            "live_win_prob": None,
            "market": {},
        },
    )
    monkeypatch.setattr(
        ESPNClient,
        "fetch_league_injuries",
        lambda self, sport, force_refresh=False: {},
    )
    monkeypatch.setattr(
        OddsClient,
        "fetch_game_odds",
        lambda self, sport, force_refresh=False, live_mode=False: [],
    )

    picks = hub.generate_picks(
        sport="NBA",
        market_filter="game",
        bankroll=25.0,
        include_passes=True,
        mode="live",
    )
    assert len(picks) > 0
    assert all(p["game_state"] == "LIVE" for p in picks)

    # Close <90s game must be latency-locked to PASS
    locked_picks = [p for p in picks if p["event_id"] == "live_lock_999"]
    assert len(locked_picks) > 0
    assert all(p["confidence_tier"] == "PASS" for p in locked_picks)
    assert any("latency lock" in d.lower() for p in locked_picks for d in p["drivers"])

    # Stale >240s Q3 game must carry staleness warning
    stale_picks = [p for p in picks if p["event_id"] == "live_q3_888"]
    assert any("m old" in d.lower() for p in stale_picks for d in p["drivers"])

    # Open position monitor test (`HOLD TO $1.00` vs `EDGE REVERSED`)
    hub.db.log_trade(
        created_at="2026-11-11 18:00",
        sport="NBA",
        event_name="LAL @ GSW",
        market_type="Moneyline",
        selection_label="GSW to Win",
        side="NO",
        entry_price=0.42,
        fee_per_contract=0.015,
        contracts=2,
        model_prob=0.58,
        fair_exit_price=0.59,
    )
    monitored = hub.evaluate_open_positions(fresh=False)
    gsw_no_trade = next(
        t for t in monitored if t["selection_label"] == "GSW to Win" and t["side"] == "NO"
    )
    assert gsw_no_trade["live_model_prob"] is not None
    assert gsw_no_trade["live_model_prob"] < 0.25
    assert "EDGE REVERSED" in gsw_no_trade["live_signal"]

    # FastAPI endpoints (`mode=live`, `/api/ledger` live guidance, and `/api/refresh-live`)
    client = TestClient(app)
    r_live = client.get("/api/picks?sport=NBA&mode=live&include_passes=true")
    assert r_live.status_code == 200
    assert r_live.json()["live_count"] > 0

    r_led = client.get("/api/ledger?bankroll=25")
    assert r_led.status_code == 200
    assert "EDGE REVERSED" in r_led.json()["trades"][0]["live_guidance"]

    r_ref = client.post("/api/refresh-live?sport=NBA")
    assert r_ref.status_code == 200
    assert "refreshed" in r_ref.json()


def test_espn_draftkings_prop_sync_normalization_and_custom_lines() -> None:
    from src.engine.dk_math import normalize_player_name

    # 1. Unicode diacritic & suffix normalization
    assert normalize_player_name("Nikola Jokić") == "nikola jokic"
    assert normalize_player_name("Jimmy Butler III") == "jimmy butler"
    assert normalize_player_name("P.J. Washington Jr.") == "p j washington"

    # 2. ESPN Core API DraftKings prop parser (NBA with juice + multiple alt lines)
    nba_items = [
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/3157465?lang=en"},
            "type": {"id": "90", "name": "Total Points, Rebounds, and Assists"},
            "odds": {
                "american": "-121",
                "total": {"value": 20.5, "alternative": "Over"},
            },
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/3157465?lang=en"},
            "type": {"id": "90", "name": "Total Points, Rebounds, and Assists"},
            "odds": {
                "american": "-109",
                "total": {"value": 20.5, "alternative": "Under"},
            },
        },
        # Alt line at 24.5 heavily skewed (+230 / -310) -> main 20.5 line must win
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/3157465?lang=en"},
            "type": {"id": "90", "name": "Total Points, Rebounds, and Assists"},
            "odds": {
                "american": "+230",
                "total": {"value": 24.5, "alternative": "Over"},
            },
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/3157465?lang=en"},
            "type": {"id": "90", "name": "Total Points, Rebounds, and Assists"},
            "odds": {
                "american": "-310",
                "total": {"value": 24.5, "alternative": "Under"},
            },
        },
    ]
    parsed_nba = ESPNClient.extract_espn_prop_lines(
        "NBA", nba_items, {"3157465": "Duncan Robinson"}
    )
    dr_pra = parsed_nba[("duncan robinson", "pra")]
    assert dr_pra["line"] == 20.5
    assert dr_pra["source"] == "DraftKings"
    assert 0.505 < dr_pra["over_prob"] < 0.535

    # 3. NFL ESPN Core API prop parser (when American juice is omitted by feed)
    nfl_items = [
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3139477"},
            "type": {"id": "8", "name": "Total Passing Yards"},
            "current": {"target": {"value": 271.5}},
        }
    ]
    parsed_nfl = ESPNClient.extract_espn_prop_lines(
        "NFL", nfl_items, {"3139477": "Patrick Mahomes"}
    )
    pm_pass = parsed_nfl[("patrick mahomes", "passing_yards")]
    assert pm_pass["line"] == 271.5
    assert pm_pass["over_prob"] is None

    # 4. PlayerPropsEngine synced default_line_value vs custom_lines override
    nfl_logs = NFLLoader().load_player_logs()
    nba_logs = NBALoader().load_player_logs()
    props_engine = PlayerPropsEngine()
    props_engine.fit(nfl_logs, nba_logs)

    synced_props = props_engine.project_nba_game_props(
        home_team="MIA",
        away_team="DET",
        live_prop_markets={("duncan robinson", "pra"): dr_pra},
    )
    p_sync = next(
        p for p in synced_props if normalize_player_name(p.player_name) == "duncan robinson" and p.prop_type == "pra"
    )
    assert p_sync.line_value == 20.5
    assert p_sync.default_line_value == 20.5
    assert p_sync.is_custom_line is False
    assert p_sync.line_source == "DraftKings"

    # Override to a higher 22.5 line -> Over probability drops monotonically & preserves default_line_value=20.5
    custom_props = props_engine.project_nba_game_props(
        home_team="MIA",
        away_team="DET",
        live_prop_markets={("duncan robinson", "pra"): dr_pra},
        custom_lines={("duncan robinson", "pra"): 22.5},
    )
    p_cust = next(
        p for p in custom_props if normalize_player_name(p.player_name) == "duncan robinson" and p.prop_type == "pra"
    )
    assert p_cust.line_value == 22.5
    assert p_cust.default_line_value == 20.5
    assert p_cust.is_custom_line is True
    assert p_cust.line_source == "Custom Line"
    assert p_cust.over_prob < p_sync.over_prob
    assert p_cust.market_over_prob is not None and p_cust.market_over_prob < p_sync.market_over_prob

    # NFL prop market without American juice still prices cleanly without crashing
    nfl_synced = props_engine.project_nfl_game_props(
        home_team="KC",
        away_team="BUF",
        live_prop_markets={("patrick mahomes", "passing_yards"): pm_pass},
    )
    pm_proj = next(
        p for p in nfl_synced if normalize_player_name(p.player_name) == "patrick mahomes" and p.prop_type == "passing_yards"
    )
    assert pm_proj.line_value == 271.5
    assert pm_proj.line_source == "DraftKings"

    # 5. Verify fetch_event_player_props constructs /leagues/{lg}/ Core API path & caches cleanly
    import tempfile
    import src.data.espn_client as espn_mod

    called_urls: list[str] = []

    class _FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"pageCount": 1, "items": nba_items}

    orig_http_get = espn_mod.http_get
    try:
        def _spy_get(url: str, **kwargs):
            called_urls.append(url)
            if "propBets" in url:
                return _FakeResp()
            raise RuntimeError("unexpected url")

        espn_mod.http_get = _spy_get  # type: ignore[assignment]
        with tempfile.TemporaryDirectory() as td:
             tmp_db = Database(Path(td) / "props_url.sqlite3")
             tmp_db.set_cache("espn:roster:NBA:8", {"3157465": "Duncan Robinson"}, ttl_seconds=3600)
             c_espn = ESPNClient(db=tmp_db)
             fetched = c_espn.fetch_event_player_props("NBA", "401909982", home_id="14", away_id="8")
             assert fetched[("duncan robinson", "pra")]["line"] == 20.5
             assert any(
                 "/sports/basketball/leagues/nba/events/401909982/competitions/401909982/odds/100/propBets" in u
                 for u in called_urls
             )
    finally:
        espn_mod.http_get = orig_http_get  # type: ignore[assignment]



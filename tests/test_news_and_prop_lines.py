"""Regression tests for ESPN milestone/combo prop filtering, news/IR injury parsing, and backup role shifts."""

from __future__ import annotations

import pandas as pd

from src.data.espn_client import (
    ESPNClient,
    classify_injury_status,
    is_promoted_starter_comment,
)
from src.engine.recommender import ModelHub
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import PlayerPropsEngine
from tests.test_human_readable_drivers import _assert_plain_sentences


def test_espn_prop_parser_rejects_milestones_and_plus_combos() -> None:
    """Milestone ladders (e.g. Mahomes 150+) and 'Plus' combo markets must never become O/U lines."""
    athletes = {
        "3139477": "Patrick Mahomes",
        "3918298": "Josh Allen",
        "4429501": "James Cook",
    }

    # Case 1: Mahomes ONLY has Passing Yards Milestones (150+, 160+, ...) and Passing Plus Rushing (253.5)
    # Josh Allen has a real id=8 Total Passing Yards (217.5) plus milestones.
    items = [
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3139477"},
            "type": {"id": "194", "name": "Passing Yards Milestones"},
            "current": {"target": {"value": 150.0, "displayValue": "150+"}},
            "odds": None,
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3139477"},
            "type": {"id": "194", "name": "Passing Yards Milestones"},
            "current": {"target": {"value": 236.0, "displayValue": "236+"}},
            "odds": None,
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3139477"},
            "type": {"id": "18", "name": "Total Passing Plus Rushing Yards (incl. overtime)"},
            "current": {"target": {"value": 253.5, "displayValue": "253.5"}},
            "odds": None,
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/4429501"},
            "type": {"id": "196", "name": "Rushing Yards Milestones"},
            "current": {"target": {"value": 10.0, "displayValue": "10+"}},
            "odds": None,
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/4429501"},
            "type": {"id": "20", "name": "Total Rushing Plus Receiving Yards (incl. overtime)"},
            "current": {"target": {"value": 100.5, "displayValue": "100.5"}},
            "odds": None,
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3918298"},
            "type": {"id": "194", "name": "Passing Yards Milestones"},
            "current": {"target": {"value": 150.0, "displayValue": "150+"}},
            "odds": None,
        },
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3918298"},
            "type": {"id": "8", "name": "Total Passing Yards (incl. overtime)"},
            "current": {"target": {"value": 217.5, "displayValue": "217.5"}},
            "odds": None,
        },
        # Even if a standard type ever carries a "+" milestone displayValue, it must be skipped
        {
            "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/3139477"},
            "type": {"id": "999", "name": "Passing Yards"},
            "current": {"target": {"value": 175.0, "displayValue": "175+"}},
            "odds": None,
        },
    ]

    parsed = ESPNClient.extract_espn_prop_lines("NFL", items, athletes)
    assert ("patrick mahomes", "passing_yards") not in parsed
    assert ("patrick mahomes", "rushing_yards") not in parsed
    assert ("james cook", "rushing_yards") not in parsed
    assert ("james cook", "receiving_yards") not in parsed
    assert parsed[("josh allen", "passing_yards")]["line"] == 217.5


def test_injury_status_and_comment_news_classification() -> None:
    """Verify IR, Suspension, Day-To-Day, and ESPN shortComment news signals are classified accurately."""
    assert classify_injury_status({"status": "Injured Reserve", "comment": ""}) == "OUT"
    assert classify_injury_status({"status": "Suspension", "comment": ""}) == "OUT"
    assert classify_injury_status({"status": "Doubtful", "comment": ""}) == "DOUBTFUL"
    assert classify_injury_status({"status": "Day-To-Day", "comment": "Sore ankle."}) == "QUESTIONABLE"
    assert (
        classify_injury_status(
            {
                "status": "Day-To-Day",
                "comment": "Kalkbrenner (hip) has been ruled out for Wednesday's matchup with Chicago.",
            }
        )
        == "OUT"
    )
    assert (
        classify_injury_status(
            {
                "status": "Questionable",
                "comment": "Cleared to play tonight without restrictions.",
            }
        )
        == "QUESTIONABLE"
    )
    assert (
        is_promoted_starter_comment(
            {
                "status": "Active",
                "comment": "Head coach said Johnson is preparing to start Sunday.",
            }
        )
        is True
    )
    assert (
        is_promoted_starter_comment(
            {
                "status": "Out",
                "comment": "Will not start due to hamstring injury.",
            }
        )
        is False
    )

    summary = ModelHub._format_injury_summary(
        {
            "KC": [
                {"player_name": "Starter RB", "status": "Injured Reserve", "comment": ""},
                {
                    "player_name": "Late Scratch",
                    "status": "Day-To-Day",
                    "comment": "Late Scratch has been ruled out for tonight.",
                },
            ],
            "BUF": [],
        },
        home_abbr="KC",
        away_abbr="BUF",
    )
    assert "Starter RB (Injured Reserve)" in summary
    assert "Late Scratch (Out)" in summary


def test_nfl_backup_rb_and_qb_adjust_for_starter_out_and_role_shifts() -> None:
    """When an NFL starter is Out/IR, backups must receive workload expansion and role-shift protection."""
    rows = []
    for wk in range(1, 9):
        # Starter QB1 + Backup QB2 (with relief cameos) + Starter RB1 + Backup RB2
        rows.append(
            {
                "player_id": "qb1", "player_name": "Starter QB", "position": "QB",
                "team": "KC", "opponent": "BUF", "season": 2026, "week": wk,
                "attempts": 34, "completions": 23, "passing_yards": 265.0, "passing_tds": 2,
                "carries": 3, "rushing_yards": 15.0, "targets": 0, "receptions": 0, "receiving_yards": 0.0,
                "game_order": 202600 + wk,
            }
        )
        if wk <= 3:
            rows.append(
                {
                    "player_id": "qb2", "player_name": "Backup QB", "position": "QB",
                    "team": "KC", "opponent": "BUF", "season": 2026, "week": wk,
                    "attempts": 4, "completions": 2, "passing_yards": 24.0, "passing_tds": 0,
                    "carries": 1, "rushing_yards": 2.0, "targets": 0, "receptions": 0, "receiving_yards": 0.0,
                    "game_order": 202600 + wk,
                }
            )
        rows.append(
            {
                "player_id": "rb1", "player_name": "Starter RB", "position": "RB",
                "team": "KC", "opponent": "BUF", "season": 2026, "week": wk,
                "attempts": 0, "completions": 0, "passing_yards": 0.0, "passing_tds": 0,
                "carries": 18, "rushing_yards": 86.0, "targets": 4, "receptions": 3, "receiving_yards": 25.0,
                "game_order": 202600 + wk,
            }
        )
        rows.append(
            {
                "player_id": "rb2", "player_name": "Backup RB", "position": "RB",
                "team": "KC", "opponent": "BUF", "season": 2026, "week": wk,
                "attempts": 0, "completions": 0, "passing_yards": 0.0, "passing_tds": 0,
                "carries": 5, "rushing_yards": 20.0, "targets": 1, "receptions": 1, "receiving_yards": 7.0,
                "game_order": 202600 + wk,
            }
        )
        # Opponent rows so opponent allowances populate
        rows.append(
            {
                "player_id": "opp_qb", "player_name": "Opp QB", "position": "QB",
                "team": "BUF", "opponent": "KC", "season": 2026, "week": wk,
                "attempts": 32, "completions": 21, "passing_yards": 245.0, "passing_tds": 2,
                "carries": 4, "rushing_yards": 22.0, "targets": 0, "receptions": 0, "receiving_yards": 0.0,
                "game_order": 202600 + wk,
            }
        )

    nfl_df = pd.DataFrame(rows)
    engine = PlayerPropsEngine()
    engine.fit(nfl_logs=nfl_df, nba_logs=pd.DataFrame())

    # 1. Starter RB on Injured Reserve + Starter QB Out, DraftKings posts starter lines for Backup RB & Backup QB
    props = engine.project_nfl_game_props(
        home_team="KC",
        away_team="BUF",
        injuries_by_team={
            "KC": [
                {"player_name": "Starter RB", "position": "RB", "status": "Injured Reserve", "comment": ""},
                {"player_name": "Starter QB", "position": "QB", "status": "Out", "comment": ""},
            ]
        },
        live_prop_markets={
            ("backup rb", "rushing_yards"): {"line": 64.5, "over_prob": 0.50, "dk_over_prob": 0.50, "source": "DraftKings"},
            ("backup qb", "passing_yards"): {"line": 205.5, "over_prob": 0.50, "dk_over_prob": 0.50, "source": "DraftKings"},
        },
    )

    # Injured players must be excluded (including Injured Reserve!)
    assert not any(p.player_name in ("Starter RB", "Starter QB") for p in props)

    rb2_proj = next(p for p in props if p.player_name == "Backup RB" and p.prop_type == "rushing_yards")
    assert rb2_proj.projected_mean > 50.0  # Was ~20; must NOT stay at 20 and hammer Under 64.5!
    assert rb2_proj.under_prob < 0.72
    _assert_plain_sentences(rb2_proj.drivers, "NFL backup RB promoted")
    assert any("Starter RB" in d or "expanded" in d.lower() for d in rb2_proj.drivers)

    qb2_proj = next(p for p in props if p.player_name == "Backup QB" and p.prop_type == "passing_yards")
    assert qb2_proj.projected_mean > 155.0  # Was 24 from relief cameos; must NOT hammer Under 205.5!
    assert qb2_proj.injury_clarity <= 0.60
    assert qb2_proj.under_prob < 0.72
    _assert_plain_sentences(qb2_proj.drivers, "NFL backup QB relief-only")


def test_nba_backup_minutes_bump_and_breaking_news_role_shift_guard() -> None:
    """NBA backups get minutes bumps when same-position starters sit, and market role-shift guard catches late news."""
    rows = []
    for d in range(1, 13):
        gday = f"2026-11-{d:02d}"
        rows.append(
            {
                "player_id": "g1", "player_name": "Star Guard", "team": "BOS", "opponent": "NYK",
                "gameday": gday, "minutes": 35.0, "points": 27.0, "rebounds": 5.0, "assists": 7.0,
                "fg3m": 3.0, "fga": 20.0, "fta": 6.0, "pra": 39.0, "position": "G",
            }
        )
        rows.append(
            {
                "player_id": "g2", "player_name": "Backup Guard", "team": "BOS", "opponent": "NYK",
                "gameday": gday, "minutes": 16.0, "points": 8.0, "rebounds": 2.0, "assists": 2.5,
                "fg3m": 1.0, "fga": 6.0, "fta": 2.0, "pra": 12.5, "position": "G",
            }
        )
        rows.append(
            {
                "player_id": "nyk1", "player_name": "Opp Guard", "team": "NYK", "opponent": "BOS",
                "gameday": gday, "minutes": 34.0, "points": 24.0, "rebounds": 4.0, "assists": 6.0,
                "fg3m": 2.5, "fga": 18.0, "fta": 5.0, "pra": 34.0, "position": "G",
            }
        )

    nba_df = pd.DataFrame(rows)
    engine = PlayerPropsEngine()
    engine.fit(nfl_logs=pd.DataFrame(), nba_logs=nba_df)

    # 1. Baseline healthy team (no starter out)
    base_props = engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        live_prop_markets={
            ("backup guard", "points"): {"line": 8.5, "over_prob": 0.50, "dk_over_prob": 0.50, "source": "DraftKings"},
        },
    )
    bg_base = next(p for p in base_props if p.player_name == "Backup Guard" and p.prop_type == "points")

    # 2. Star Guard is Day-To-Day with "has been ruled out" in comment; DK posts starter line 16.5 for Backup Guard
    out_props = engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        injuries_by_team={
            "BOS": [
                {
                    "player_name": "Star Guard",
                    "status": "Day-To-Day",
                    "comment": "Star Guard (knee) has been ruled out for tonight's game.",
                }
            ]
        },
        live_prop_markets={
            ("backup guard", "points"): {"line": 16.5, "over_prob": 0.50, "dk_over_prob": 0.50, "source": "DraftKings"},
        },
    )
    assert not any(p.player_name == "Star Guard" for p in out_props)
    bg_out = next(p for p in out_props if p.player_name == "Backup Guard" and p.prop_type == "points")
    assert bg_out.projected_mean > bg_base.projected_mean + 5.0
    assert bg_out.under_prob < 0.70
    _assert_plain_sentences(bg_out.drivers, "NBA backup guard promoted")

    # 3. Late-breaking scratch NOT yet in ESPN injuries, but DK posts 16.5 for an 8 PPG backup -> role-shift guard fires
    late_props = engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        injuries_by_team={},
        live_prop_markets={
            ("backup guard", "points"): {"line": 16.5, "over_prob": 0.50, "dk_over_prob": 0.50, "source": "DraftKings"},
        },
    )
    bg_late = next(p for p in late_props if p.player_name == "Backup Guard" and p.prop_type == "points")
    assert bg_late.projected_mean > 13.0
    assert bg_late.injury_clarity <= 0.60
    assert bg_late.under_prob < 0.70
    assert any("expanded role" in d.lower() for d in bg_late.drivers)
    _assert_plain_sentences(bg_late.drivers, "NBA late-scratch role-shift guard")


def test_nfl_and_nba_game_models_handle_ir_and_comment_ruled_out() -> None:
    """Verify NFLGameModel recognizes QB on IR and NBAGameModel recognizes Day-To-Day ruled-out comment."""
    from src.data.nba_loader import NBALoader
    from src.data.nfl_loader import NFLLoader

    nfl_m = NFLGameModel()
    nfl_m.fit(NFLLoader.generate_synthetic_nfl_schedules(seed=19))
    nfl_clean = nfl_m.predict_matchup(home_team="KC", away_team="BUF")
    nfl_ir = nfl_m.predict_matchup(
        home_team="KC",
        away_team="BUF",
        injuries_by_team={"KC": [{"player_name": "Patrick Mahomes", "position": "QB", "status": "Injured Reserve"}]},
    )
    assert nfl_ir.expected_margin < nfl_clean.expected_margin - 1.5
    assert any("injured reserve" in d.lower() for d in nfl_ir.drivers)
    _assert_plain_sentences(nfl_ir.drivers, "NFL IR QB game prediction")

    nba_m = NBAGameModel()
    nba_m.fit(
        NBALoader.generate_synthetic_nba_games(seed=23),
        player_logs=NBALoader.generate_synthetic_nba_player_logs(seed=23),
    )
    nba_clean = nba_m.predict_matchup(home_team="BOS", away_team="NYK")
    nba_ruled_out = nba_m.predict_matchup(
        home_team="BOS",
        away_team="NYK",
        injuries_by_team={
            "BOS": [
                {
                    "player_name": "Jayson Tatum",
                    "status": "Day-To-Day",
                    "comment": "Tatum (ankle) has been ruled out for tonight.",
                }
            ]
        },
    )
    assert nba_ruled_out.expected_margin < nba_clean.expected_margin - 1.5
    assert any("Jayson Tatum" in d for d in nba_ruled_out.drivers)
    _assert_plain_sentences(nba_ruled_out.drivers, "NBA Day-To-Day ruled-out comment")


def test_cross_book_prop_merge_drops_wildly_divergent_estimated_espn_line(tmp_path, monkeypatch) -> None:
    """An unpriced estimated ESPN DK line that wildly disagrees with a real FD line must not create phantom arbs."""
    from datetime import datetime, timedelta, timezone
    from src.data.odds_client import OddsClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "sb_diverge.sqlite3"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")
    monkeypatch.setenv("TRADITIONAL_SPORTSBOOKS_ENABLED", "true")

    soon_iso = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    fake_ev = {
        "event_id": "401873082",
        "sport": "NFL",
        "event_name": "Kansas City Chiefs @ Buffalo Bills",
        "short_name": "KC @ BUF",
        "start_time": soon_iso,
        "state": "PRE",
        "status": "STATUS_SCHEDULED",
        "status_detail": "8:20 PM ET",
        "period": 0,
        "clock_seconds": 0.0,
        "display_clock": "0:00",
        "elapsed_fraction": 0.0,
        "remaining_minutes": 60.0,
        "home_score": 0,
        "away_score": 0,
        "home_abbr": "BUF",
        "away_abbr": "KC",
        "home_name": "Buffalo Bills",
        "away_name": "Kansas City Chiefs",
        "indoor": False,
        "temp_f": 45.0,
        "wind_mph": 6.0,
        "market": {"home_spread": -2.5, "total_line": 48.5, "home_win_prob": 0.56},
    }

    hub = ModelHub()
    hub.train_all("all")

    monkeypatch.setattr(hub, "_get_upcoming_events", lambda sp, **kw: [fake_ev] if sp == "NFL" else [])
    monkeypatch.setattr(hub.espn, "fetch_league_injuries", lambda sp, force_refresh=False: {})
    monkeypatch.setattr(
        hub.odds,
        "fetch_game_odds",
        lambda sp, force_refresh=False, live_mode=False: [
            {
                "id": "odds_kc_buf",
                "home_team": "Buffalo Bills",
                "away_team": "Kansas City Chiefs",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Buffalo Bills", "price": -130},
                                    {"name": "Kansas City Chiefs", "price": 110},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        if sp == "NFL"
        else [],
    )
    # ESPN provides a stale/estimated DK 150.0 line; Odds API provides real FanDuel 235.5 line (no DK)
    monkeypatch.setattr(
        hub.espn,
        "fetch_slate_player_props",
        lambda sp, events, force_refresh=False: {
            "401873082": {
                ("patrick mahomes", "passing_yards"): {
                    "line": 150.0,
                    "over_prob": None,
                    "dk_over_prob": None,
                    "source": "DraftKings",
                    "dk_line": 150.0,
                    "books": {
                        "draftkings": {
                            "line": 150.0,
                            "over_odds": -115,
                            "under_odds": -115,
                            "over_prob": 0.50,
                            "odds_estimated": True,
                        }
                    },
                }
            }
        }
        if sp == "NFL"
        else {},
    )
    monkeypatch.setattr(
        OddsClient,
        "fetch_event_player_props",
        lambda self, sp, oid, force_refresh=False, live_mode=False: {
            ("patrick mahomes", "passing_yards"): {
                "line": 235.5,
                "over_prob": 0.50,
                "dk_over_prob": 0.50,
                "source": "FanDuel",
                "fd_line": 235.5,
                "fd_over_odds": -110,
                "fd_under_odds": -110,
                "books": {
                    "fanduel": {
                        "line": 235.5,
                        "over_odds": -110,
                        "under_odds": -110,
                        "over_prob": 0.50,
                    }
                },
            }
        },
    )

    slate = hub.generate_sportsbook_slate(sport="nfl", market_filter="props", include_passes=True)
    pm_picks = [p for p in slate["picks"] if p.get("player_name") == "Patrick Mahomes" and "Passing Yards" in p.get("market_type", "")]
    assert len(pm_picks) == 1
    assert pm_picks[0]["line_value"] == 235.5
    assert pm_picks[0]["dk_line"] is None
    assert pm_picks[0]["fd_line"] == 235.5
    assert not any("Patrick Mahomes" in m.get("middle_window_label", "") for m in slate["middles"])


def test_is_csv_stale_helper(tmp_path) -> None:
    import os
    import time
    from src.data.nba_loader import is_csv_stale

    f = tmp_path / "sample.csv"
    assert is_csv_stale(f) is True
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    assert is_csv_stale(f, max_age_hours=18.0) is False
    old_ts = time.time() - (24.0 * 3600.0)
    os.utime(f, (old_ts, old_ts))
    assert is_csv_stale(f, max_age_hours=18.0) is True


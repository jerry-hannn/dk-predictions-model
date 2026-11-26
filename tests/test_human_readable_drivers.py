"""Verify all user-facing 'why' drivers and reasons are plain-English full sentences."""

from __future__ import annotations

import re

from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.engine.parlay_math import (
    ParlayLeg,
    estimate_leg_pair_correlation,
    evaluate_sportsbook_parlay,
)
from src.engine.recommender import ModelHub
from src.engine.sportsbook_math import evaluate_steam_lag
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import PlayerPropsEngine

BANNED_JARGON = [
    r"\bEWMA\b",
    r"\bOpp-Adj\b",
    r"\bopp-adj\b",
    r"\bposs/48m\b",
    r"\bunit factor\b",
    r"\bpace factor\b",
    r"\bProj\b",
    r"\bPF\)",
    r"\bpt diff\b",
    r"\brem\b",
    r"\bm left\b",
    r"\bm old\b",
    r"\bRZ\b",
    r"\bModel proj\b",
]


def _assert_plain_sentences(items: list[str], context: str) -> None:
    assert items, f"Expected non-empty reasons for {context}"
    for text in items:
        assert text == text.strip(), f"Leading/trailing whitespace in {context}: {text!r}"
        assert text[0].isupper(), f"Expected capitalized sentence in {context}: {text!r}"
        assert text.endswith("."), f"Expected full sentence ending with period in {context}: {text!r}"
        for pattern in BANNED_JARGON:
            assert not re.search(pattern, text), (
                f"Found jargon ({pattern}) in {context}: {text!r}"
            )


def test_game_model_drivers_are_plain_sentences() -> None:
    nfl_df = NFLLoader().load_schedules()
    nfl_model = NFLGameModel()
    nfl_model.fit(nfl_df)

    nfl_pre = nfl_model.predict_matchup(
        home_team="KC",
        away_team="BUF",
        home_rest=10,
        away_rest=6,
        is_indoor=False,
        wind_mph=20.0,
        injuries_by_team={"BUF": [{"player_name": "Josh Allen", "position": "QB", "status": "Out"}]},
    )
    _assert_plain_sentences(nfl_pre.drivers, "NFL pregame")

    nfl_live = nfl_model.predict_matchup(
        home_team="KC",
        away_team="BUF",
        is_indoor=True,
        live_state={
            "state": "IN",
            "home_score": 17,
            "away_score": 14,
            "remaining_minutes": 18.0,
            "status_detail": "Q3 3:00",
            "possession_team": "KC",
            "is_red_zone": True,
        },
    )
    _assert_plain_sentences(nfl_live.drivers, "NFL live")

    nba_df = NBALoader().load_games()
    nba_model = NBAGameModel()
    nba_model.fit(nba_df)

    nba_pre = nba_model.predict_matchup(
        home_team="DEN",
        away_team="LAL",
        injuries_by_team={"LAL": [{"player_name": "LeBron James", "status": "Out"}]},
    )
    _assert_plain_sentences(nba_pre.drivers, "NBA pregame")

    nba_live = nba_model.predict_matchup(
        home_team="BOS",
        away_team="LAL",
        live_state={
            "state": "IN",
            "home_score": 82,
            "away_score": 76,
            "remaining_minutes": 14.0,
            "status_detail": "Q3 2:00",
        },
    )
    _assert_plain_sentences(nba_live.drivers, "NBA live")


def test_player_prop_drivers_are_plain_sentences_and_preserve_blowout_signal() -> None:
    nfl_logs = NFLLoader().load_player_logs()
    nba_logs = NBALoader().load_player_logs()
    props_engine = PlayerPropsEngine()
    props_engine.fit(nfl_logs, nba_logs)

    nba_pre = props_engine.project_nba_game_props(home_team="BOS", away_team="NYK")
    assert nba_pre
    for pp in nba_pre[:10]:
        _assert_plain_sentences(pp.drivers, f"NBA pregame prop {pp.player_name} {pp.prop_type}")

    p_pre = next(p for p in nba_pre if p.team == "BOS" and p.prop_type == "points")
    top_player = p_pre.player_name
    nba_live = props_engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        live_state={
            "state": "IN",
            "period": 4,
            "remaining_minutes": 7.0,
            "home_score": 112,
            "away_score": 84,
        },
        live_boxscore={
            top_player.lower(): {
                "player_name": top_player,
                "team": "BOS",
                "starter": True,
                "active": True,
                "did_not_play": False,
                "ejected": False,
                "minutes": 26.0,
                "points": 18.0,
                "rebounds": 5.0,
                "assists": 4.0,
                "fg3m": 2.0,
                "pra": 27.0,
                "fouls": 5.0,
            }
        },
    )
    p_live = next(p for p in nba_live if p.player_name == top_player and p.prop_type == "points")
    _assert_plain_sentences(p_live.drivers, "NBA live blowout prop")
    assert any("blowout" in d.lower() for d in p_live.drivers)

    nfl_teams = nfl_logs["team"].dropna().unique().tolist()
    home_nfl, away_nfl = nfl_teams[0], nfl_teams[1]
    nfl_pre = props_engine.project_nfl_game_props(
        home_team=home_nfl, away_team=away_nfl, wind_mph=20.0, is_indoor=False
    )
    assert nfl_pre
    for pp in nfl_pre[:10]:
        _assert_plain_sentences(pp.drivers, f"NFL pregame prop {pp.player_name} {pp.prop_type}")


def test_recommender_steam_and_parlay_reasons_are_plain_sentences() -> None:
    hub = ModelHub()

    ev = {
        "sport": "NBA",
        "event_id": "test_1",
        "event_name": "LAL @ BOS",
        "short_name": "LAL @ BOS",
        "home_abbr": "BOS",
        "away_abbr": "LAL",
        "start_time": "2026-11-26T00:00:00Z",
        "state": "IN",
        "status_detail": "Q4 1:10",
        "home_score": 102,
        "away_score": 100,
        "remaining_minutes": 1.1,
    }
    mkt = {
        "odds_age_seconds": 360.0,
        "spread_move": -1.5,
        "open_home_spread": -3.5,
        "home_spread": -5.0,
        "total_move": 2.0,
        "open_total_line": 222.5,
        "total_line": 224.5,
        "line_source": "DraftKings",
    }
    _, _, _, _, safety = hub._live_safety_context(ev, mkt, 1.0)
    _assert_plain_sentences(safety, "live safety drivers")

    live_fields = hub._event_live_fields(ev, mkt)
    _assert_plain_sentences([live_fields["line_movement"]], "line movement")

    steam_line = evaluate_steam_lag(
        book="DRAFTKINGS",
        side_label="BOS -4.5",
        side="HOME",
        book_odds=-110,
        book_devig_prob=0.50,
        sharp_prob=0.56,
        book_line=-4.5,
        sharp_line=-6.0,
    )
    assert steam_line is not None
    _assert_plain_sentences([steam_line.summary], "steam line lag")

    steam_price = evaluate_steam_lag(
        book="FANDUEL",
        side_label="BOS ML",
        side="HOME",
        book_odds=-110,
        book_devig_prob=0.50,
        sharp_prob=0.57,
    )
    assert steam_price is not None
    _assert_plain_sentences([steam_price.summary], "steam price lag")

    leg_a = ParlayLeg(
        leg_id="a",
        sport="NBA",
        event_id="g1",
        event_name="LAL @ BOS",
        short_name="LAL @ BOS",
        market_type="Moneyline",
        selection_label="BOS to Win",
        side="HOME",
        model_prob=0.65,
        american_odds=-150,
    )
    leg_b = ParlayLeg(
        leg_id="b",
        sport="NBA",
        event_id="g1",
        event_name="LAL @ BOS",
        short_name="LAL @ BOS",
        market_type="Spread",
        selection_label="BOS -3.5",
        side="HOME",
        model_prob=0.56,
        american_odds=-110,
    )
    _, reason = estimate_leg_pair_correlation(leg_a, leg_b)
    _assert_plain_sentences([reason], "parlay correlation reason")

    res = evaluate_sportsbook_parlay(
        [leg_a, leg_b], custom_parlay_odds=145, correlation_overrides={"0:1": 0.25}
    )
    _assert_plain_sentences([res["correlation_pairs"][0]["reason"]], "parlay custom override reason")

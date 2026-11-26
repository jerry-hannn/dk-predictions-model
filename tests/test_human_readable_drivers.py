"""Verify all user-facing 'why' drivers and reasons are plain-English, specific, and logically consistent."""

from __future__ import annotations

import re

from src.data.espn_client import ESPNClient
from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.engine.parlay_math import (
    ParlayLeg,
    estimate_leg_pair_correlation,
    evaluate_sportsbook_parlay,
)
from src.engine.recommender import ModelHub, _assemble_prop_drivers
from src.engine.sportsbook_math import evaluate_steam_lag
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import (
    PlayerPropProjection,
    PlayerPropsEngine,
    build_prop_primary_reason,
    prop_median_for_type,
)

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


def test_game_model_drivers_and_outdoor_nfl_weather() -> None:
    nfl_df = NFLLoader().load_schedules()
    nfl_model = NFLGameModel()
    nfl_model.fit(nfl_df)

    nfl_clear = nfl_model.predict_matchup(
        home_team="BUF",
        away_team="KC",
        is_indoor=False,
        wind_mph=5.0,
        temp_f=62.0,
        weather_condition="Clear",
    )
    nfl_snow = nfl_model.predict_matchup(
        home_team="BUF",
        away_team="KC",
        is_indoor=False,
        wind_mph=19.0,
        temp_f=24.0,
        weather_condition="Snow",
        injuries_by_team={"KC": [{"player_name": "Patrick Mahomes", "position": "QB", "status": "Out"}]},
    )
    _assert_plain_sentences(nfl_snow.drivers, "NFL outdoor snow")
    assert nfl_snow.expected_total < nfl_clear.expected_total - 2.0
    assert any("snow" in d.lower() for d in nfl_snow.drivers)

    nfl_rain = nfl_model.predict_matchup(
        home_team="SEA",
        away_team="SF",
        is_indoor=False,
        wind_mph=14.0,
        temp_f=44.0,
        weather_condition="Rain",
    )
    _assert_plain_sentences(nfl_rain.drivers, "NFL outdoor rain")
    assert any("rain" in d.lower() for d in nfl_rain.drivers)

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


def test_player_props_positional_defense_weather_and_skew_consistency() -> None:
    nfl_logs = NFLLoader().load_player_logs()
    nba_logs = NBALoader().load_player_logs()
    props_engine = PlayerPropsEngine()
    props_engine.fit(nfl_logs, nba_logs)

    # Positional opponent stats populated for both NFL (QB/RB/WR/TE) and NBA (G/F/C)
    assert ("PHI", "TE") in props_engine.nfl_opp_pos_stats
    assert ("CIN", "TE") in props_engine.nfl_opp_pos_stats
    assert (
        props_engine.nfl_opp_pos_stats[("CIN", "TE")]["receiving_yards"]["factor"]
        > props_engine.nfl_opp_pos_stats[("PHI", "TE")]["receiving_yards"]["factor"]
    )
    assert any(pos == "G" for _, pos in props_engine.nba_opp_pos_stats)
    assert any(pos == "F" for _, pos in props_engine.nba_opp_pos_stats)
    assert any(pos == "C" for _, pos in props_engine.nba_opp_pos_stats)

    # NBA pregame props: positional defense + named sidelined star + blowout spread risk
    base_bos = [
        p
        for p in props_engine.project_nba_game_props(home_team="BOS", away_team="NYK")
        if p.team == "BOS"
    ]
    assert base_bos
    sidelined_name = base_bos[0].player_name
    nba_pre = props_engine.project_nba_game_props(
        home_team="BOS",
        away_team="NYK",
        expected_spread=-15.0,
        injuries_by_team={"BOS": [{"player_name": sidelined_name, "status": "Out"}]},
    )
    assert nba_pre
    for pp in nba_pre[:10]:
        _assert_plain_sentences(pp.drivers, f"NBA pregame prop {pp.player_name} {pp.prop_type}")
        assert pp.position in ("G", "F", "C")
    bos_drivers = [d for pp in nba_pre if pp.team == "BOS" for d in pp.drivers]
    assert any(sidelined_name in d for d in bos_drivers)
    assert any("lopsided" in d.lower() for d in bos_drivers)
    all_nba_drivers = [d for pp in nba_pre for d in pp.drivers]
    assert any(w in d.lower() for d in all_nba_drivers for w in ("guards", "forwards", "centers"))

    # NBA live blowout preserves "blowout" keyword for open-position monitoring
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

    # NFL outdoor snow vs clear weather affects passing down & rushing up, with specific reasons
    nfl_clear = props_engine.project_nfl_game_props(
        home_team="KC",
        away_team="CIN",
        is_indoor=False,
        wind_mph=5.0,
        temp_f=65.0,
        weather_condition="Clear",
    )
    nfl_snow = props_engine.project_nfl_game_props(
        home_team="KC",
        away_team="CIN",
        is_indoor=False,
        wind_mph=20.0,
        temp_f=25.0,
        weather_condition="Snow",
    )
    assert nfl_snow
    for pp in nfl_snow[:12]:
        _assert_plain_sentences(pp.drivers, f"NFL snow prop {pp.player_name} {pp.prop_type}")

    qb_clear = next((p for p in nfl_clear if p.prop_type == "passing_yards"), None)
    qb_snow = next(
        (p for p in nfl_snow if qb_clear and p.player_name == qb_clear.player_name and p.prop_type == "passing_yards"),
        None,
    )
    assert qb_clear is not None and qb_snow is not None
    assert qb_snow.projected_mean < qb_clear.projected_mean
    assert any("snow" in d.lower() for d in qb_snow.drivers)

    rb_clear = next((p for p in nfl_clear if p.prop_type == "rushing_yards" and p.position == "RB"), None)
    rb_snow = next(
        (p for p in nfl_snow if rb_clear and p.player_name == rb_clear.player_name and p.prop_type == "rushing_yards"),
        None,
    )
    assert rb_clear is not None and rb_snow is not None
    assert rb_snow.projected_mean > rb_clear.projected_mean
    assert any("rushing volume" in d.lower() for d in rb_snow.drivers)

    # Positional phrasing in NFL drivers (quarterbacks / running backs / wide receivers / tight ends)
    all_nfl_drivers = [d for pp in nfl_clear for d in pp.drivers]
    assert any(
        w in d.lower()
        for d in all_nfl_drivers
        for w in ("quarterbacks", "running backs", "wide receivers", "tight ends")
    )

    # Skew-driven Under explanation never contradicts an Under pick when mean > line > median
    med = prop_median_for_type("receiving_yards", mean=75.3, std=48.0)
    assert med < 66.5
    skew_reason = build_prop_primary_reason(
        player_name="Matthew Golden",
        prop_type="receiving_yards",
        prop_label="Receiving Yards",
        side="UNDER",
        line_val=66.5,
        proj_mean=75.3,
        proj_median=med,
        side_prob=0.53,
        line_source="DraftKings",
    )
    _assert_plain_sentences([skew_reason], "skew-driven under reason")
    assert "typical outcome" in skew_reason.lower()
    assert "below the 66.5 line" in skew_reason.lower()


def test_cross_book_prop_driver_uses_winning_line_and_espn_weather_parsing() -> None:
    mock_pp = PlayerPropProjection(
        sport="NFL",
        player_name="Patrick Mahomes",
        team="KC",
        opponent="CIN",
        prop_type="passing_yards",
        prop_label="Passing Yards",
        line_value=252.5,
        projected_mean=268.0,
        projected_std=45.0,
        over_prob=0.60,
        under_prob=0.40,
        drivers=["Old line 252.5."],
        injury_clarity=1.0,
        projected_median=261.0,
        position="QB",
        context_drivers=["CIN allows 12% more passing yards per game to quarterbacks than the league average (255.0 vs. 227.9)."],
    )
    assembled = _assemble_prop_drivers(
        pp=mock_pp,
        side="OVER",
        line_val=249.5,
        side_prob=0.63,
        line_source="DK + FD",
        safety_drivers=[],
        shop_driver="FanDuel offers a better line (249.5 at -110) than DraftKings (252.5 at -110), adding 2.4% in expected value.",
    )
    _assert_plain_sentences(assembled, "sportsbook cross-book prop drivers")
    assert any("249.5" in d and "18.5 above" in d for d in assembled)
    assert not any("252.5 line" in d for d in assembled)

    # ESPN weather condition + NFL dome fallback parsing
    raw_ev = {
        "id": "401999999",
        "date": "2026-11-29T18:00:00Z",
        "weather": {"displayValue": "Heavy Rain", "temperature": 38, "windSpeed": 17},
        "competitions": [
            {
                "status": {"type": {"state": "pre", "name": "STATUS_SCHEDULED", "shortDetail": "Sun 1:00 PM"}},
                "venue": {"indoor": False},
                "competitors": [
                    {"homeAway": "home", "team": {"id": "2", "abbreviation": "BUF", "displayName": "Buffalo Bills"}},
                    {"homeAway": "away", "team": {"id": "12", "abbreviation": "KC", "displayName": "Kansas City Chiefs"}},
                ],
                "odds": [],
            }
        ],
    }
    parsed = ESPNClient()._parse_event("NFL", raw_ev)
    assert parsed is not None
    assert parsed["indoor"] is False
    assert parsed["weather_condition"] == "Heavy Rain"
    assert parsed["wind_mph"] == 17.0
    assert parsed["temp_f"] == 38.0
    assert parsed["precip_prob"] >= 70.0


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

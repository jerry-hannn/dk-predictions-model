"""Zero-leakage chronological feature tests & model probability calibration bounds."""

import numpy as np

from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import PlayerPropsEngine


def test_nfl_chronological_features_have_zero_future_leakage() -> None:
    sched = NFLLoader.generate_synthetic_nfl_schedules(seed=7)
    model = NFLGameModel()
    feats_orig, _ = model.build_chronological_features(sched)

    # Corrupt all scores in the second half of the dataset and verify first-half features are identical
    sched_mutated = sched.copy()
    mid = len(sched_mutated) // 2
    sched_mutated.loc[mid:, "home_score"] = 99.0
    sched_mutated.loc[mid:, "away_score"] = 0.0

    feats_mut, _ = model.build_chronological_features(sched_mutated)
    for col in NFLGameModel.FEATURE_COLS:
        np.testing.assert_allclose(
            feats_orig.loc[: mid - 1, col].to_numpy(),
            feats_mut.loc[: mid - 1, col].to_numpy(),
            atol=1e-9,
        )

    # Also verify row `mid` itself does NOT see row `mid`'s own corrupted 99-0 score!
    for col in NFLGameModel.FEATURE_COLS:
        assert abs(float(feats_orig.loc[mid, col]) - float(feats_mut.loc[mid, col])) < 1e-9


def test_nba_chronological_features_have_zero_future_leakage() -> None:
    games = NBALoader.generate_synthetic_nba_games(seed=11)
    model = NBAGameModel()
    feats_orig, _ = model.build_chronological_features(games)

    games_mut = games.copy()
    mid = len(games_mut) // 2
    games_mut.loc[mid:, "home_score"] = 160
    games_mut.loc[mid:, "away_score"] = 70

    feats_mut, _ = model.build_chronological_features(games_mut)
    for col in NBAGameModel.FEATURE_COLS:
        np.testing.assert_allclose(
            feats_orig.loc[:mid, col].to_numpy(),
            feats_mut.loc[:mid, col].to_numpy(),
            atol=1e-9,
        )


def test_models_and_props_produce_bounded_calibrated_probabilities() -> None:
    nfl_sched = NFLLoader.generate_synthetic_nfl_schedules(seed=42)
    nfl_logs = NFLLoader.generate_synthetic_nfl_player_logs(seed=42)
    nba_games = NBALoader.generate_synthetic_nba_games(seed=101)
    nba_logs = NBALoader.generate_synthetic_nba_player_logs(seed=101)

    nfl_m = NFLGameModel()
    nfl_metrics = nfl_m.fit(nfl_sched)
    assert nfl_metrics["brier"] < 0.26

    p_nfl = nfl_m.predict_matchup(home_team="KC", away_team="LV", spread_line=-7.5, total_line=44.5)
    assert 0.01 < p_nfl.home_win_prob < 0.99
    assert 0.01 < p_nfl.home_cover_prob < 0.99
    assert 0.01 < p_nfl.over_prob < 0.99
    assert p_nfl.home_win_prob > 0.60  # Strong KC team favored at home vs LV

    nba_m = NBAGameModel()
    nba_metrics = nba_m.fit(nba_games, player_logs=nba_logs)
    assert nba_metrics["brier"] < 0.26

    p_nba = nba_m.predict_matchup(
        home_team="BOS",
        away_team="WAS",
        spread_line=-9.5,
        total_line=228.5,
        injuries_by_team={"WAS": [{"player_name": "Starter", "status": "Out"}]},
    )
    assert 0.60 < p_nba.home_win_prob < 0.99

    props = PlayerPropsEngine()
    props.fit(nfl_logs=nfl_logs, nba_logs=nba_logs)
    nba_props = props.project_nba_game_props(
        home_team="BOS",
        away_team="LAL",
        expected_pace=101.5,
        injuries_by_team={"BOS": [{"player_name": "Jaylen Brown", "status": "Out"}]},
    )
    assert len(nba_props) > 0
    for pp in nba_props:
        assert 0.0 < pp.over_prob < 1.0
        assert abs((pp.over_prob + pp.under_prob) - 1.0) < 1e-4

    nfl_props = props.project_nfl_game_props(
        home_team="KC", away_team="BUF", expected_spread=-2.5, expected_total=48.5
    )
    assert len(nfl_props) > 0
    for pp in nfl_props:
        assert 0.0 < pp.over_prob < 1.0


def test_nba_loader_drops_in_progress_midgame_rows() -> None:
    import pandas as pd

    raw_team = pd.DataFrame(
        [
            {
                "GAME_ID": "0022600223",
                "SEASON_STR": "2026-27",
                "GAME_DATE": "2026-11-11",
                "TEAM_ABBREVIATION": "NOP",
                "MATCHUP": "NOP vs. BKN",
                "WL": "L",
                "PTS": 104,
                "FGM": 38,
                "FGA": 86,
                "FG3M": 11,
                "FTA": 22,
                "OREB": 10,
                "TOV": 14,
            },
            {
                "GAME_ID": "0022600223",
                "SEASON_STR": "2026-27",
                "GAME_DATE": "2026-11-11",
                "TEAM_ABBREVIATION": "BKN",
                "MATCHUP": "BKN @ NOP",
                "WL": "W",
                "PTS": 140,
                "FGM": 52,
                "FGA": 91,
                "FG3M": 19,
                "FTA": 19,
                "OREB": 11,
                "TOV": 10,
            },
            # Active Q2 in-progress game (WL is None/NaN, halftime score 32-28)
            {
                "GAME_ID": "0022600226",
                "SEASON_STR": "2026-27",
                "GAME_DATE": "2026-11-11",
                "TEAM_ABBREVIATION": "POR",
                "MATCHUP": "POR vs. OKC",
                "WL": None,
                "PTS": 28,
                "FGM": 10,
                "FGA": 34,
                "FG3M": 3,
                "FTA": 6,
                "OREB": 4,
                "TOV": 6,
            },
            {
                "GAME_ID": "0022600226",
                "SEASON_STR": "2026-27",
                "GAME_DATE": "2026-11-11",
                "TEAM_ABBREVIATION": "OKC",
                "MATCHUP": "OKC @ POR",
                "WL": None,
                "PTS": 32,
                "FGM": 12,
                "FGA": 36,
                "FG3M": 4,
                "FTA": 5,
                "OREB": 5,
                "TOV": 4,
            },
        ]
    )
    paired = NBALoader._pair_team_logs(raw_team)
    assert len(paired) == 1
    assert paired.iloc[0]["game_id"] == "0022600223"

    raw_players = pd.DataFrame(
        [
            {
                "PLAYER_ID": "1",
                "PLAYER_NAME": "Completed Player",
                "TEAM_ABBREVIATION": "BKN",
                "MATCHUP": "BKN @ NOP",
                "GAME_DATE": "2026-11-11",
                "WL": "W",
                "MIN": 34,
                "PTS": 28,
                "REB": 6,
                "AST": 5,
                "FG3M": 4,
                "FGA": 18,
                "FTA": 6,
            },
            {
                "PLAYER_ID": "2",
                "PLAYER_NAME": "InProgress Player",
                "TEAM_ABBREVIATION": "OKC",
                "MATCHUP": "OKC @ POR",
                "GAME_DATE": "2026-11-11",
                "WL": None,
                "MIN": 14,
                "PTS": 11,
                "REB": 2,
                "AST": 2,
                "FG3M": 1,
                "FGA": 8,
                "FTA": 2,
            },
        ]
    )
    norm_p = NBALoader._normalize_player_logs(raw_players)
    assert len(norm_p) == 1
    assert norm_p.iloc[0]["player_name"] == "Completed Player"


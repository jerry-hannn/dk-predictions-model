"""NFL Game Moneyline, Key-Number Spread Cover, and Total Over/Under Model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import atomic_joblib_dump
from src.config import get_settings
from src.engine.dk_math import clamp_prob, inv_logit, logit


@dataclass
class NFLGamePrediction:
    home_team: str
    away_team: str
    expected_margin: float  # Home - Away points
    expected_total: float
    home_win_prob: float
    home_cover_prob: float
    over_prob: float
    spread_line: float
    total_line: float
    drivers: list[str]
    injury_clarity: float
    is_live: bool = False


class NFLGameModel:
    """
    Chronological Elo + Opponent-Adjusted Scoring/Efficiency NFL Engine with
    Empirical Key-Number Margin Kernel (3, 7, 6, 10, 4, 14) and Platt calibration.
    """

    FEATURE_COLS = [
        "elo_diff",
        "adj_margin_diff",
        "off_vs_def_home",
        "off_vs_def_away",
        "rest_diff",
        "is_indoor",
        "high_wind",
        "cold_weather",
        "div_game",
    ]

    TOTAL_FEATURE_COLS = [
        "combined_offense",
        "combined_defense_allowed",
        "pace_scoring_env",
        "is_indoor",
        "wind_speed",
        "cold_weather",
        "div_game",
    ]

    def __init__(self) -> None:
        self.margin_reg: Pipeline | None = None
        self.total_reg: Pipeline | None = None
        self.win_clf: CalibratedClassifierCV | None = None
        self.residual_std: float = 13.1
        self.total_std: float = 9.8
        self.league_pts: float = 22.2
        self.empirical_residuals: np.ndarray = np.array([], dtype=float)
        self.latest_team_state: dict[str, dict[str, float]] = {}
        self.is_fitted: bool = False

    def build_chronological_features(
        self, schedules: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """
        Walk forward strictly chronologically (row by row sorted by gameday, game_id).
        Computes pre-game features BEFORE updating team ratings with that game's score.
        """
        df = schedules.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        elo: dict[str, float] = {}
        off_ewma: dict[str, float] = {}
        def_ewma: dict[str, float] = {}
        adj_margin_ewma: dict[str, float] = {}
        lg_pts: float = 22.2
        current_gday: str | None = None
        day_pts_sum: float = 0.0
        day_pts_cnt: int = 0
        last_season_seen: int | None = None

        feat_rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            gday = str(row["gameday"])
            if current_gday is not None and gday != current_gday and day_pts_cnt > 0:
                lg_pts = 0.96 * lg_pts + 0.04 * (day_pts_sum / day_pts_cnt)
                day_pts_sum = 0.0
                day_pts_cnt = 0
            current_gday = gday

            season = int(row["season"])
            # Between seasons, regress Elo & efficiency toward rolling league mean (standard NFL shrinkage)
            if last_season_seen is not None and season != last_season_seen:
                for t in list(elo.keys()):
                    elo[t] = 1500.0 + 0.67 * (elo[t] - 1500.0)
                    off_ewma[t] = lg_pts + 0.60 * (off_ewma[t] - lg_pts)
                    def_ewma[t] = lg_pts + 0.60 * (def_ewma[t] - lg_pts)
                    adj_margin_ewma[t] = 0.60 * adj_margin_ewma[t]
            last_season_seen = season

            h = str(row["home_team"]).upper()
            a = str(row["away_team"]).upper()
            for t in (h, a):
                elo.setdefault(t, 1500.0)
                off_ewma.setdefault(t, lg_pts)
                def_ewma.setdefault(t, lg_pts)
                adj_margin_ewma.setdefault(t, 0.0)

            h_rest = float(row.get("home_rest", 7.0) or 7.0)
            a_rest = float(row.get("away_rest", 7.0) or 7.0)
            is_indoor = int(row.get("is_indoor", 0) or 0)
            wind = 0.0 if is_indoor else float(row.get("wind", 5.0) or 5.0)
            temp = 70.0 if is_indoor else float(row.get("temp", 65.0) or 65.0)
            div_game = int(row.get("div_game", 0) or 0)

            # Pre-game strictly causal features
            feat = {
                "game_id": row["game_id"],
                "season": season,
                "week": int(row.get("week", 1)),
                "gameday": str(row["gameday"]),
                "home_team": h,
                "away_team": a,
                "elo_diff": (elo[h] - elo[a]) / 25.0,  # in approximate point units
                "adj_margin_diff": adj_margin_ewma[h] - adj_margin_ewma[a],
                "off_vs_def_home": off_ewma[h] + def_ewma[a] - lg_pts,
                "off_vs_def_away": off_ewma[a] + def_ewma[h] - lg_pts,
                "combined_offense": off_ewma[h] + off_ewma[a],
                "combined_defense_allowed": def_ewma[h] + def_ewma[a],
                "pace_scoring_env": 0.5 * (off_ewma[h] + off_ewma[a] + def_ewma[h] + def_ewma[a]),
                "rest_diff": np.clip(h_rest - a_rest, -7.0, 8.0),
                "is_indoor": is_indoor,
                "wind_speed": wind,
                "high_wind": int(wind >= 15.0),
                "cold_weather": int(temp <= 34.0 and not is_indoor),
                "div_game": div_game,
                "home_spread": row.get("home_spread"),
                "total_line": row.get("total_line"),
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
            }
            feat_rows.append(feat)

            # Post-game update ONLY if valid completed scores exist
            if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                hs = float(row["home_score"])
                as_ = float(row["away_score"])
                margin = hs - as_

                # Elo update with Margin-of-Victory multiplier (FiveThirtyEight-style)
                elo_diff_hfa = (elo[h] + 48.0) - elo[a]
                exp_h = 1.0 / (1.0 + 10.0 ** (-elo_diff_hfa / 400.0))
                outcome_h = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
                mov_mult = np.log(abs(margin) + 1.0) * (
                    2.2 / ((elo_diff_hfa if margin >= 0 else -elo_diff_hfa) * 0.001 + 2.2)
                )
                delta = 20.0 * mov_mult * (outcome_h - exp_h)
                elo[h] += delta
                elo[a] -= delta

                # Damped (0.5) opponent-adjusted scoring EWMA (contraction-mapped, prevents common-mode drift)
                alpha = 0.22
                opp_adj_h_pts = hs - 0.5 * (def_ewma[a] - lg_pts)
                opp_adj_a_pts = as_ - 0.5 * (def_ewma[h] - lg_pts)
                opp_adj_h_def = as_ - 0.5 * (off_ewma[a] - lg_pts)
                opp_adj_a_def = hs - 0.5 * (off_ewma[h] - lg_pts)

                prev_h_margin = adj_margin_ewma[h]
                prev_a_margin = adj_margin_ewma[a]
                adj_margin_ewma[h] = (1 - alpha) * prev_h_margin + alpha * (
                    (hs - as_ - 1.8) + 0.5 * prev_a_margin
                )
                adj_margin_ewma[a] = (1 - alpha) * prev_a_margin + alpha * (
                    (as_ - hs + 1.8) + 0.5 * prev_h_margin
                )
                off_ewma[h] = (1 - alpha) * off_ewma[h] + alpha * opp_adj_h_pts
                off_ewma[a] = (1 - alpha) * off_ewma[a] + alpha * opp_adj_a_pts
                def_ewma[h] = (1 - alpha) * def_ewma[h] + alpha * opp_adj_h_def
                def_ewma[a] = (1 - alpha) * def_ewma[a] + alpha * opp_adj_a_def
                day_pts_sum += 0.5 * (hs + as_)
                day_pts_cnt += 1

        if day_pts_cnt > 0:
            lg_pts = 0.96 * lg_pts + 0.04 * (day_pts_sum / day_pts_cnt)
        self.league_pts = lg_pts
        state = {
            t: {
                "elo": elo[t],
                "off_ewma": off_ewma[t],
                "def_ewma": def_ewma[t],
                "adj_margin": adj_margin_ewma[t],
            }
            for t in elo
        }
        return pd.DataFrame(feat_rows), state

    def fit(self, schedules: pd.DataFrame) -> dict[str, float]:
        feat_df, state = self.build_chronological_features(schedules)
        self.latest_team_state = state

        completed = feat_df.dropna(subset=["home_score", "away_score"]).copy()
        completed = completed[completed["home_score"] != completed["away_score"]].copy()
        if len(completed) < 40:
            raise ValueError("Need at least 40 completed NFL games to fit NFLGameModel.")

        X = completed[self.FEATURE_COLS].to_numpy(dtype=float)
        y_margin = (completed["home_score"] - completed["away_score"]).to_numpy(dtype=float)
        y_win = (y_margin > 0).astype(int)

        X_tot = completed[self.TOTAL_FEATURE_COLS].to_numpy(dtype=float)
        y_tot = (completed["home_score"] + completed["away_score"]).to_numpy(dtype=float)

        self.margin_reg = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=18.0))]
        )
        self.margin_reg.fit(X, y_margin)

        preds_margin = self.margin_reg.predict(X)
        residuals = y_margin - preds_margin
        self.residual_std = float(max(10.5, min(15.5, np.std(residuals, ddof=1))))
        # Store empirical residuals for key-number (3, 7, 6, 10) aware spread cover kernel
        self.empirical_residuals = np.round(residuals).astype(float)

        self.total_reg = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=22.0))]
        )
        self.total_reg.fit(X_tot, y_tot)
        tot_res = y_tot - self.total_reg.predict(X_tot)
        self.total_std = float(max(8.5, min(13.5, np.std(tot_res, ddof=1))))

        base_clf = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.4, max_iter=500))]
        )
        n_splits = min(4, max(2, len(completed) // 80))
        self.win_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=n_splits),
        )
        self.win_clf.fit(X, y_win)
        self.is_fitted = True

        in_sample_probs = self.win_clf.predict_proba(X)[:, 1]
        brier = float(np.mean((in_sample_probs - y_win) ** 2))
        return {
            "n_games": float(len(completed)),
            "margin_std": round(self.residual_std, 3),
            "total_std": round(self.total_std, 3),
            "brier": round(brier, 4),
        }

    def _spread_cover_probability(
        self, expected_margin: float, home_spread: float, scale_override: float | None = None
    ) -> float:
        """
        Compute P(Home Margin + home_spread > 0) conditional on non-push (or half-win on push
        for binary DKeX contracts which typically use half-point hooks e.g. -3.5, -6.5).
        Combines smooth Gaussian CDF with empirical NFL key-number residuals.
        """
        threshold = -home_spread  # Home covers if Margin > threshold
        sigma = scale_override if scale_override is not None else self.residual_std
        gauss_p = float(1.0 - norm.cdf(threshold, loc=expected_margin, scale=sigma))
        if scale_override is None and len(self.empirical_residuals) >= 80:
            sim_margins = np.round(expected_margin + self.empirical_residuals)
            wins = np.sum(sim_margins > threshold)
            pushes = np.sum(np.abs(sim_margins - threshold) < 1e-6)
            denom = len(sim_margins) - pushes
            emp_p = float(wins / denom) if denom > 0 else gauss_p
            return clamp_prob(0.55 * emp_p + 0.45 * gauss_p)
        return clamp_prob(gauss_p)

    def predict_matchup(
        self,
        *,
        home_team: str,
        away_team: str,
        spread_line: float | None = None,
        total_line: float | None = None,
        home_rest: float = 7.0,
        away_rest: float = 7.0,
        is_indoor: bool = False,
        wind_mph: float = 5.0,
        temp_f: float = 65.0,
        div_game: bool = False,
        injuries_by_team: dict[str, list[dict[str, Any]]] | None = None,
        live_state: dict[str, Any] | None = None,
    ) -> NFLGamePrediction:
        if not self.is_fitted or self.margin_reg is None or self.total_reg is None or self.win_clf is None:
            raise RuntimeError("NFLGameModel must be fitted before calling predict_matchup.")

        h = home_team.upper()
        a = away_team.upper()
        lg_pts = float(getattr(self, "league_pts", 22.2))
        hs = self.latest_team_state.get(
            h, {"elo": 1500.0, "off_ewma": lg_pts, "def_ewma": lg_pts, "adj_margin": 0.0}
        )
        as_ = self.latest_team_state.get(
            a, {"elo": 1500.0, "off_ewma": lg_pts, "def_ewma": lg_pts, "adj_margin": 0.0}
        )

        # Evaluate injury adjustments (especially QB Out / Doubtful / Questionable)
        inj_adj_margin = 0.0
        injury_clarity = 1.0
        drivers: list[str] = []

        if injuries_by_team:
            for team_code, sign in ((h, -1.0), (a, 1.0)):
                for item in injuries_by_team.get(team_code, []):
                    pos = str(item.get("position", "")).upper()
                    status = str(item.get("status", "")).lower()
                    pname = item.get("player_name", "")
                    if pos == "QB":
                        if "out" in status or "doubtful" in status:
                            inj_adj_margin += sign * 4.5
                            status_text = "doubtful to play" if "doubtful" in status else "out"
                            drivers.append(
                                f"{team_code} quarterback {pname} is {status_text}, which weakens their offensive outlook."
                            )
                        elif "questionable" in status:
                            injury_clarity = min(injury_clarity, 0.65)
                    elif pos in ("WR", "LT", "OT", "CB", "DE", "EDGE") and "out" in status:
                        inj_adj_margin += sign * 0.45

        eff_wind = 0.0 if is_indoor else wind_mph
        eff_temp = 70.0 if is_indoor else temp_f
        rest_diff = float(np.clip(home_rest - away_rest, -7.0, 8.0))

        x_vec = np.array(
            [
                [
                    (hs["elo"] - as_["elo"]) / 25.0,
                    hs["adj_margin"] - as_["adj_margin"],
                    hs["off_ewma"] + as_["def_ewma"] - lg_pts,
                    as_["off_ewma"] + hs["def_ewma"] - lg_pts,
                    rest_diff,
                    int(is_indoor),
                    int(eff_wind >= 15.0),
                    int(eff_temp <= 34.0 and not is_indoor),
                    int(div_game),
                ]
            ],
            dtype=float,
        )
        x_tot = np.array(
            [
                [
                    hs["off_ewma"] + as_["off_ewma"],
                    hs["def_ewma"] + as_["def_ewma"],
                    0.5 * (hs["off_ewma"] + as_["off_ewma"] + hs["def_ewma"] + as_["def_ewma"]),
                    int(is_indoor),
                    eff_wind,
                    int(eff_temp <= 34.0 and not is_indoor),
                    int(div_game),
                ]
            ],
            dtype=float,
        )

        pre_exp_margin = float(self.margin_reg.predict(x_vec)[0]) + inj_adj_margin
        pre_exp_total = float(self.total_reg.predict(x_tot)[0]) - abs(inj_adj_margin) * 0.35

        clf_prob = float(self.win_clf.predict_proba(x_vec)[0, 1])
        margin_win_prob = float(1.0 - norm.cdf(0.0, loc=pre_exp_margin, scale=self.residual_std))
        pre_home_win_prob = clamp_prob(0.55 * clf_prob + 0.45 * margin_win_prob)

        is_live = bool(live_state and str(live_state.get("state", "")).upper() == "IN")

        if is_live and live_state is not None:
            h_sc = float(live_state.get("home_score", 0) or 0)
            a_sc = float(live_state.get("away_score", 0) or 0)
            rem_mins = float(live_state.get("remaining_minutes", 60.0) or 0.0)
            rem_frac = max(0.005, min(1.0, rem_mins / 60.0))
            elapsed_frac = 1.0 - rem_frac

            poss_team = str(live_state.get("possession_team") or "").upper()
            is_rz = bool(live_state.get("is_red_zone", False))
            poss_pts = 3.6 if is_rz else 1.85
            poss_adj = poss_pts if poss_team == h else (-poss_pts if poss_team == a else 0.0)

            cur_diff = h_sc - a_sc
            cur_tot = h_sc + a_sc
            exp_margin = cur_diff + rem_frac * pre_exp_margin + poss_adj

            # Empirical Bayes live total mean-reversion (88% reverts to pre-game efficiency)
            scoring_surplus = cur_tot - (elapsed_frac * pre_exp_total)
            exp_total = cur_tot + rem_frac * pre_exp_total + 0.12 * rem_frac * scoring_surplus

            sigma_margin_rem = max(2.4, self.residual_std * float(np.sqrt(max(0.03, rem_frac))))
            sigma_tot_rem = max(2.8, self.total_std * float(np.sqrt(max(0.03, rem_frac))))

            brownian_wp = clamp_prob(
                float(1.0 - norm.cdf(0.0, loc=exp_margin, scale=sigma_margin_rem))
            )
            live_pbp_wp = live_state.get("live_win_prob")
            if live_pbp_wp is not None:
                home_win_prob = clamp_prob(
                    inv_logit(0.45 * logit(float(live_pbp_wp)) + 0.55 * logit(brownian_wp))
                )
            else:
                home_win_prob = brownian_wp

            eff_spread = (
                float(spread_line)
                if spread_line is not None
                else round(-exp_margin * 2.0) / 2.0
            )
            eff_total = (
                float(total_line)
                if total_line is not None
                else round(exp_total * 2.0) / 2.0
            )
            home_cover_prob = self._spread_cover_probability(
                exp_margin, eff_spread, scale_override=sigma_margin_rem
            )
            over_prob = clamp_prob(
                float(1.0 - norm.cdf(eff_total, loc=exp_total, scale=sigma_tot_rem))
            )

            detail = str(live_state.get("status_detail") or f"{rem_mins:.0f} minutes left")
            poss_note = (
                f" with {poss_team} in the red zone"
                if (poss_team and is_rz)
                else (f" with {poss_team} on offense" if poss_team else "")
            )
            proj_h = 0.5 * (exp_total + exp_margin)
            proj_a = 0.5 * (exp_total - exp_margin)
            drivers.insert(
                0,
                f"Live ({detail}): {a} {int(a_sc)}, {h} {int(h_sc)}{poss_note}, with a projected final of {h} {proj_h:.1f} to {a} {proj_a:.1f}.",
            )
        else:
            exp_margin = pre_exp_margin
            exp_total = pre_exp_total
            home_win_prob = pre_home_win_prob
            eff_spread = (
                float(spread_line)
                if spread_line is not None
                else round(-exp_margin * 2.0) / 2.0
            )
            eff_total = (
                float(total_line)
                if total_line is not None
                else round(exp_total * 2.0) / 2.0
            )
            home_cover_prob = self._spread_cover_probability(exp_margin, eff_spread)
            over_prob = clamp_prob(
                float(1.0 - norm.cdf(eff_total, loc=exp_total, scale=self.total_std))
            )

        # Build concise human-readable model drivers
        elo_pts = (hs["elo"] - as_["elo"]) / 25.0
        if abs(elo_pts) >= 1.5:
            leader = h if elo_pts > 0 else a
            drivers.append(
                f"{leader} holds a {abs(elo_pts):.1f}-point overall team strength advantage."
            )
        adj_diff = hs["adj_margin"] - as_["adj_margin"]
        if abs(adj_diff) >= 1.8:
            leader = h if adj_diff > 0 else a
            drivers.append(
                f"{leader} has outperformed opponents by {abs(adj_diff):.1f} more points per game recently."
            )
        if abs(rest_diff) >= 2:
            rested = h if rest_diff > 0 else a
            rest_days = int(abs(rest_diff))
            day_word = "day" if rest_days == 1 else "days"
            drivers.append(
                f"{rested} enters this matchup with {rest_days} extra {day_word} of rest."
            )
        if eff_wind >= 15.0:
            drivers.append(
                f"Strong {eff_wind:.0f} mph winds should make passing tougher and keep scoring lower."
            )
        if is_indoor:
            drivers.append(
                "Playing indoors removes weather factors and tends to help offenses score."
            )
        if not drivers:
            drivers.append(
                f"The model projects a final score of {h} {0.5*(exp_total+exp_margin):.1f} to {a} {0.5*(exp_total-exp_margin):.1f}."
            )

        return NFLGamePrediction(
            home_team=h,
            away_team=a,
            expected_margin=round(exp_margin, 2),
            expected_total=round(exp_total, 2),
            home_win_prob=round(home_win_prob, 4),
            home_cover_prob=round(home_cover_prob, 4),
            over_prob=round(over_prob, 4),
            spread_line=eff_spread,
            total_line=eff_total,
            drivers=drivers[:4],
            injury_clarity=injury_clarity,
            is_live=is_live,
        )

    def save(self, path: Path | None = None) -> Path:
        target = path or (get_settings().models_dir / "nfl_game_model.joblib")
        return atomic_joblib_dump(self, target)

    @classmethod
    def load(cls, path: Path | None = None) -> "NFLGameModel":
        target = path or (get_settings().models_dir / "nfl_game_model.joblib")
        return joblib.load(target)

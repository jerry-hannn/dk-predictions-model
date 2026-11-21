"""NBA Game Moneyline, Spread Cover, and Total Over/Under Model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models import atomic_joblib_dump
from src.config import get_settings
from src.engine.dk_math import clamp_prob, inv_logit, logit


@dataclass
class NBAGamePrediction:
    home_team: str
    away_team: str
    expected_margin: float
    expected_total: float
    expected_pace: float
    home_win_prob: float
    home_cover_prob: float
    over_prob: float
    spread_line: float
    total_line: float
    drivers: list[str]
    injury_clarity: float
    is_live: bool = False


class NBAGameModel:
    """
    Chronological NBA Four-Factors, Opponent-Adjusted Net Rating, Pace, and
    Schedule-Fatigue (B2B / 3-in-4 / Altitude) Engine with Injury Impact Adjustment.
    """

    FEATURE_COLS = [
        "net_rtg_diff",
        "efg_diff",
        "tov_diff",
        "oreb_diff",
        "ftr_diff",
        "home_b2b",
        "away_b2b",
        "rest_diff",
        "altitude_home",
    ]

    TOTAL_FEATURE_COLS = [
        "matchup_pace",
        "combined_off_rtg",
        "combined_def_rtg",
        "combined_efg",
        "fatigue_sum",
    ]

    def __init__(self) -> None:
        self.margin_reg: Pipeline | None = None
        self.total_reg: Pipeline | None = None
        self.win_clf: CalibratedClassifierCV | None = None
        self.margin_scale: float = 12.9
        self.total_scale: float = 15.2
        self.league_ortg: float = 114.0
        self.latest_team_state: dict[str, dict[str, Any]] = {}
        self.star_scoring_share: dict[tuple[str, str], float] = {}
        self.is_fitted: bool = False

    def register_player_shares(self, player_logs: pd.DataFrame) -> None:
        """Compute player point/minute impact shares for live NBA injury NetRtg adjustments."""
        if len(player_logs) == 0:
            return
        grp = (
            player_logs.groupby(["team", "player_name"], as_index=False)
            .agg(avg_pts=("points", "mean"), avg_min=("minutes", "mean"), games=("points", "count"))
        )
        for _, r in grp.iterrows():
            if float(r["avg_min"]) >= 20.0 and int(r["games"]) >= 3:
                # Approximate NetRtg swing when a high-minute rotation/star player sits:
                # 30 PPG / 35 MPG star -> ~3.8 pts impact; 18 PPG starter -> ~1.8 pts impact
                impact = max(0.5, (float(r["avg_pts"]) - 10.0) * 0.17)
                self.star_scoring_share[(str(r["team"]).upper(), str(r["player_name"]).lower())] = round(
                    min(4.8, impact), 2
                )

    def build_chronological_features(
        self, games: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        df = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
        state: dict[str, dict[str, Any]] = {}
        feat_rows: list[dict[str, Any]] = []
        lg_ortg: float = 114.0
        current_gday: str | None = None
        day_ortg_sum: float = 0.0
        day_ortg_cnt: int = 0
        last_season_seen: str | None = None

        for _, row in df.iterrows():
            gdate_str = str(row["gameday"])
            if current_gday is not None and gdate_str != current_gday and day_ortg_cnt > 0:
                lg_ortg = 0.96 * lg_ortg + 0.04 * (day_ortg_sum / day_ortg_cnt)
                day_ortg_sum = 0.0
                day_ortg_cnt = 0
            current_gday = gdate_str

            season_str = str(row.get("season", "") or "")
            if last_season_seen is not None and season_str and season_str != last_season_seen:
                # Offseason roster/coaching regression toward league average
                for st in state.values():
                    st["off_rtg"] = lg_ortg + 0.60 * (st["off_rtg"] - lg_ortg)
                    st["def_rtg"] = lg_ortg + 0.60 * (st["def_rtg"] - lg_ortg)
                    st["net_rtg"] = st["off_rtg"] - st["def_rtg"]
                    st["pace"] = 99.2 + 0.60 * (st["pace"] - 99.2)
                    st["efg"] = 0.542 + 0.60 * (st["efg"] - 0.542)
                    st["tov_pct"] = 0.134 + 0.60 * (st["tov_pct"] - 0.134)
                    st["oreb"] = 10.6 + 0.60 * (st["oreb"] - 10.6)
                    st["ftr"] = 0.245 + 0.60 * (st["ftr"] - 0.245)
            if season_str:
                last_season_seen = season_str

            h = str(row["home_team"]).upper()
            a = str(row["away_team"]).upper()
            cur_dt = datetime.strptime(gdate_str[:10], "%Y-%m-%d")

            for t in (h, a):
                if t not in state:
                    state[t] = {
                        "off_rtg": lg_ortg,
                        "def_rtg": lg_ortg,
                        "net_rtg": 0.0,
                        "pace": 99.2,
                        "efg": 0.542,
                        "tov_pct": 0.134,
                        "oreb": 10.6,
                        "ftr": 0.245,
                        "recent_dates": [],
                    }

            hs = state[h]
            as_ = state[a]

            h_rest = (
                min(5, (cur_dt - hs["recent_dates"][-1]).days)
                if hs["recent_dates"]
                else 3
            )
            a_rest = (
                min(5, (cur_dt - as_["recent_dates"][-1]).days)
                if as_["recent_dates"]
                else 3
            )
            h_b2b = int(h_rest == 1)
            a_b2b = int(a_rest == 1)
            h_3in4 = int(sum(1 for d in hs["recent_dates"][-3:] if (cur_dt - d).days <= 3) >= 2)
            a_3in4 = int(sum(1 for d in as_["recent_dates"][-3:] if (cur_dt - d).days <= 3) >= 2)

            matchup_pace = 0.5 * (hs["pace"] + as_["pace"])

            feat_rows.append(
                {
                    "game_id": str(row["game_id"]),
                    "gameday": gdate_str,
                    "home_team": h,
                    "away_team": a,
                    "net_rtg_diff": hs["net_rtg"] - as_["net_rtg"],
                    "efg_diff": (hs["efg"] - as_["efg"]) * 100.0,
                    "tov_diff": (as_["tov_pct"] - hs["tov_pct"]) * 100.0,
                    "oreb_diff": hs["oreb"] - as_["oreb"],
                    "ftr_diff": (hs["ftr"] - as_["ftr"]) * 100.0,
                    "home_b2b": h_b2b + 0.5 * h_3in4,
                    "away_b2b": a_b2b + 0.5 * a_3in4,
                    "rest_diff": h_rest - a_rest,
                    "altitude_home": int(h in ("DEN", "UTA")),
                    "matchup_pace": matchup_pace,
                    "combined_off_rtg": hs["off_rtg"] + as_["off_rtg"],
                    "combined_def_rtg": hs["def_rtg"] + as_["def_rtg"],
                    "combined_efg": (hs["efg"] + as_["efg"]) * 100.0,
                    "fatigue_sum": h_b2b + a_b2b + 0.5 * (h_3in4 + a_3in4),
                    "home_score": row.get("home_score"),
                    "away_score": row.get("away_score"),
                }
            )

            # Update state AFTER building features (strict zero-leakage; skip incomplete/tied mid-game rows)
            if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
                h_pts = float(row["home_score"])
                a_pts = float(row["away_score"])
                if h_pts != a_pts and h_pts >= 65.0 and a_pts >= 65.0:
                    pace = float(row.get("pace", 99.2) or 99.2)
                    h_ortg_obs = (h_pts / max(80.0, pace)) * 100.0
                    a_ortg_obs = (a_pts / max(80.0, pace)) * 100.0

                    # Damped (0.5) opponent-adjusted efficiency update (contraction-mapped, zero common-mode drift)
                    adj_h_off = h_ortg_obs - 0.5 * (as_["def_rtg"] - lg_ortg)
                    adj_a_off = a_ortg_obs - 0.5 * (hs["def_rtg"] - lg_ortg)
                    adj_h_def = a_ortg_obs - 0.5 * (as_["off_rtg"] - lg_ortg)
                    adj_a_def = h_ortg_obs - 0.5 * (hs["off_rtg"] - lg_ortg)

                    alpha = 0.16
                    hs["off_rtg"] = (1 - alpha) * hs["off_rtg"] + alpha * adj_h_off
                    hs["def_rtg"] = (1 - alpha) * hs["def_rtg"] + alpha * adj_h_def
                    hs["net_rtg"] = hs["off_rtg"] - hs["def_rtg"]
                    hs["pace"] = (1 - alpha) * hs["pace"] + alpha * pace
                    hs["efg"] = (1 - alpha) * hs["efg"] + alpha * float(row.get("home_efg", 0.542))
                    hs["tov_pct"] = (1 - alpha) * hs["tov_pct"] + alpha * float(row.get("home_tov_pct", 0.134))
                    hs["oreb"] = (1 - alpha) * hs["oreb"] + alpha * float(row.get("home_oreb", 10.6))
                    hs["ftr"] = (1 - alpha) * hs["ftr"] + alpha * float(row.get("home_ftr", 0.245))
                    hs["recent_dates"] = (hs["recent_dates"] + [cur_dt])[-6:]

                    as_["off_rtg"] = (1 - alpha) * as_["off_rtg"] + alpha * adj_a_off
                    as_["def_rtg"] = (1 - alpha) * as_["def_rtg"] + alpha * adj_a_def
                    as_["net_rtg"] = as_["off_rtg"] - as_["def_rtg"]
                    as_["pace"] = (1 - alpha) * as_["pace"] + alpha * pace
                    as_["efg"] = (1 - alpha) * as_["efg"] + alpha * float(row.get("away_efg", 0.542))
                    as_["tov_pct"] = (1 - alpha) * as_["tov_pct"] + alpha * float(row.get("away_tov_pct", 0.134))
                    as_["oreb"] = (1 - alpha) * as_["oreb"] + alpha * float(row.get("away_oreb", 10.6))
                    as_["ftr"] = (1 - alpha) * as_["ftr"] + alpha * float(row.get("away_ftr", 0.245))
                    as_["recent_dates"] = (as_["recent_dates"] + [cur_dt])[-6:]
                    day_ortg_sum += 0.5 * (h_ortg_obs + a_ortg_obs)
                    day_ortg_cnt += 1

        if day_ortg_cnt > 0:
            lg_ortg = 0.96 * lg_ortg + 0.04 * (day_ortg_sum / day_ortg_cnt)
        self.league_ortg = lg_ortg
        return pd.DataFrame(feat_rows), state

    def fit(
        self, games: pd.DataFrame, player_logs: pd.DataFrame | None = None
    ) -> dict[str, float]:
        if player_logs is not None:
            self.register_player_shares(player_logs)

        feat_df, state = self.build_chronological_features(games)
        self.latest_team_state = state

        completed = feat_df.dropna(subset=["home_score", "away_score"]).copy()
        completed = completed[completed["home_score"] != completed["away_score"]].copy()
        if len(completed) < 50:
            raise ValueError("Need at least 50 completed NBA games to fit NBAGameModel.")

        X = completed[self.FEATURE_COLS].to_numpy(dtype=float)
        y_margin = (completed["home_score"] - completed["away_score"]).to_numpy(dtype=float)
        y_win = (y_margin > 0).astype(int)

        X_tot = completed[self.TOTAL_FEATURE_COLS].to_numpy(dtype=float)
        y_tot = (completed["home_score"] + completed["away_score"]).to_numpy(dtype=float)

        self.margin_reg = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=24.0))]
        )
        self.margin_reg.fit(X, y_margin)
        res = y_margin - self.margin_reg.predict(X)
        self.margin_scale = float(max(11.0, min(15.5, np.std(res, ddof=1))))

        self.total_reg = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=28.0))]
        )
        self.total_reg.fit(X_tot, y_tot)
        tot_res = y_tot - self.total_reg.predict(X_tot)
        self.total_scale = float(max(12.5, min(18.5, np.std(tot_res, ddof=1))))

        base_clf = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.45, max_iter=500))]
        )
        n_splits = min(4, max(2, len(completed) // 100))
        self.win_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=n_splits),
        )
        self.win_clf.fit(X, y_win)
        self.is_fitted = True

        probs = self.win_clf.predict_proba(X)[:, 1]
        brier = float(np.mean((probs - y_win) ** 2))
        return {
            "n_games": float(len(completed)),
            "margin_std": round(self.margin_scale, 3),
            "total_std": round(self.total_scale, 3),
            "brier": round(brier, 4),
        }

    def predict_matchup(
        self,
        *,
        home_team: str,
        away_team: str,
        game_date: str | None = None,
        spread_line: float | None = None,
        total_line: float | None = None,
        injuries_by_team: dict[str, list[dict[str, Any]]] | None = None,
        live_state: dict[str, Any] | None = None,
    ) -> NBAGamePrediction:
        if not self.is_fitted or self.margin_reg is None or self.total_reg is None or self.win_clf is None:
            raise RuntimeError("NBAGameModel must be fitted before calling predict_matchup.")

        h = home_team.upper()
        a = away_team.upper()
        default_st = {
            "off_rtg": 114.0,
            "def_rtg": 114.0,
            "net_rtg": 0.0,
            "pace": 99.2,
            "efg": 0.542,
            "tov_pct": 0.134,
            "oreb": 10.6,
            "ftr": 0.245,
            "recent_dates": [],
        }
        hs = self.latest_team_state.get(h, default_st)
        as_ = self.latest_team_state.get(a, default_st)

        cur_dt = (
            datetime.strptime(game_date[:10], "%Y-%m-%d")
            if game_date
            else datetime.now()
        )
        h_rest = (
            min(5, max(1, (cur_dt - hs["recent_dates"][-1]).days))
            if hs["recent_dates"]
            else 2
        )
        a_rest = (
            min(5, max(1, (cur_dt - as_["recent_dates"][-1]).days))
            if as_["recent_dates"]
            else 2
        )
        h_b2b = int(h_rest == 1)
        a_b2b = int(a_rest == 1)
        matchup_pace = 0.5 * (float(hs["pace"]) + float(as_["pace"]))

        drivers: list[str] = []
        inj_margin_adj = 0.0
        injury_clarity = 1.0

        if injuries_by_team:
            for team_code, sign in ((h, -1.0), (a, 1.0)):
                team_pen = 0.0
                for item in injuries_by_team.get(team_code, []):
                    pname = str(item.get("player_name", ""))
                    status = str(item.get("status", "")).lower()
                    impact = self.star_scoring_share.get(
                        (team_code, pname.lower()), 0.85
                    )
                    if "out" in status or "doubtful" in status:
                        if impact >= 1.5 and team_pen < 6.5:
                            drivers.append(f"{team_code} missing {pname} ({item.get('status')})")
                        team_pen = min(7.0, team_pen + impact)
                    elif "questionable" in status and impact >= 2.2:
                        injury_clarity = min(injury_clarity, 0.72)
                inj_margin_adj += sign * team_pen

        x_vec = np.array(
            [
                [
                    hs["net_rtg"] - as_["net_rtg"],
                    (hs["efg"] - as_["efg"]) * 100.0,
                    (as_["tov_pct"] - hs["tov_pct"]) * 100.0,
                    hs["oreb"] - as_["oreb"],
                    (hs["ftr"] - as_["ftr"]) * 100.0,
                    float(h_b2b),
                    float(a_b2b),
                    float(h_rest - a_rest),
                    int(h in ("DEN", "UTA")),
                ]
            ],
            dtype=float,
        )
        x_tot = np.array(
            [
                [
                    matchup_pace,
                    hs["off_rtg"] + as_["off_rtg"],
                    hs["def_rtg"] + as_["def_rtg"],
                    (hs["efg"] + as_["efg"]) * 100.0,
                    float(h_b2b + a_b2b),
                ]
            ],
            dtype=float,
        )

        pre_exp_margin = float(self.margin_reg.predict(x_vec)[0]) + inj_margin_adj
        pre_exp_total = float(self.total_reg.predict(x_tot)[0]) - abs(inj_margin_adj) * 0.30

        clf_prob = float(self.win_clf.predict_proba(x_vec)[0, 1])
        dist_win = float(
            1.0 - student_t.cdf(0.0, df=12, loc=pre_exp_margin, scale=self.margin_scale * 0.92)
        )
        pre_home_win_prob = clamp_prob(0.50 * clf_prob + 0.50 * dist_win)

        is_live = bool(live_state and str(live_state.get("state", "")).upper() == "IN")

        if is_live and live_state is not None:
            h_sc = float(live_state.get("home_score", 0) or 0)
            a_sc = float(live_state.get("away_score", 0) or 0)
            rem_mins = float(live_state.get("remaining_minutes", 48.0) or 0.0)
            rem_frac = max(0.005, min(1.0, rem_mins / 48.0))
            elapsed_frac = 1.0 - rem_frac

            cur_diff = h_sc - a_sc
            cur_tot = h_sc + a_sc
            exp_margin = cur_diff + rem_frac * pre_exp_margin

            # Empirical Bayes live total mean-reversion (85% of shooting heat reverts to pre-game Four-Factors)
            scoring_surplus = cur_tot - (elapsed_frac * pre_exp_total)
            foul_endgame = 1.8 if (rem_mins <= 2.0 and abs(cur_diff) <= 7.0) else 0.0
            exp_total = cur_tot + rem_frac * pre_exp_total + 0.15 * rem_frac * scoring_surplus + foul_endgame

            # Update live tempo slightly for downstream player prop projections
            matchup_pace = float(
                np.clip(
                    matchup_pace * (1.0 + 0.15 * (scoring_surplus / max(60.0, pre_exp_total))),
                    92.0,
                    108.0,
                )
            )

            sigma_margin_rem = max(
                2.3, (self.margin_scale * 0.92) * float(np.sqrt(max(0.025, rem_frac)))
            )
            sigma_tot_rem = max(
                2.8, (self.total_scale * 0.93) * float(np.sqrt(max(0.025, rem_frac)))
            )

            brownian_wp = clamp_prob(
                float(1.0 - student_t.cdf(0.0, df=12, loc=exp_margin, scale=sigma_margin_rem))
            )
            live_pbp_wp = live_state.get("live_win_prob")
            if live_pbp_wp is not None:
                home_win_prob = clamp_prob(
                    inv_logit(0.40 * logit(float(live_pbp_wp)) + 0.60 * logit(brownian_wp))
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
            home_cover_prob = clamp_prob(
                float(
                    1.0
                    - student_t.cdf(-eff_spread, df=12, loc=exp_margin, scale=sigma_margin_rem)
                )
            )
            over_prob = clamp_prob(
                float(1.0 - student_t.cdf(eff_total, df=14, loc=exp_total, scale=sigma_tot_rem))
            )

            detail = str(live_state.get("status_detail") or f"{rem_mins:.1f}m left")
            drivers.insert(
                0,
                f"LIVE {detail}: {a} {int(a_sc)} - {h} {int(h_sc)} | Proj final {0.5*(exp_total+exp_margin):.1f}-{0.5*(exp_total-exp_margin):.1f}",
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
            home_cover_prob = clamp_prob(
                float(
                    1.0
                    - student_t.cdf(
                        -eff_spread, df=12, loc=exp_margin, scale=self.margin_scale * 0.92
                    )
                )
            )
            over_prob = clamp_prob(
                float(
                    1.0
                    - student_t.cdf(
                        eff_total, df=14, loc=exp_total, scale=self.total_scale * 0.93
                    )
                )
            )

        net_diff = hs["net_rtg"] - as_["net_rtg"]
        if abs(net_diff) >= 2.0:
            leader = h if net_diff > 0 else a
            drivers.append(f"{leader} +{abs(net_diff):.1f} Opp-Adj Net Rating advantage")
        if a_b2b and not h_b2b:
            drivers.append(f"{a} on fatigued back-to-back road spot")
        elif h_b2b and not a_b2b:
            drivers.append(f"{h} playing on back-to-back")
        if h in ("DEN", "UTA"):
            drivers.append(f"{h} high-altitude home court edge")
        drivers.append(f"Projected pace: {matchup_pace:.1f} poss/48m")

        return NBAGamePrediction(
            home_team=h,
            away_team=a,
            expected_margin=round(exp_margin, 2),
            expected_total=round(exp_total, 2),
            expected_pace=round(matchup_pace, 1),
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
        target = path or (get_settings().models_dir / "nba_game_model.joblib")
        return atomic_joblib_dump(self, target)

    @classmethod
    def load(cls, path: Path | None = None) -> "NBAGameModel":
        target = path or (get_settings().models_dir / "nba_game_model.joblib")
        return joblib.load(target)

"""NFL & NBA Player Props Engine with Pre-Game & Live In-Play Boxscore Conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom, poisson, t as student_t

from src.engine.dk_math import clamp_prob


@dataclass
class PlayerPropProjection:
    sport: str
    player_name: str
    team: str
    opponent: str
    prop_type: str
    prop_label: str
    line_value: float
    projected_mean: float
    projected_std: float
    over_prob: float
    under_prob: float
    drivers: list[str]
    injury_clarity: float
    current_stat: float = 0.0
    is_live: bool = False
    market_over_prob: float | None = None
    dk_over_prob: float | None = None
    line_source: str = "Model Ref Line"


def _ewma_mean_std(values: np.ndarray, span: int = 6) -> tuple[float, float]:
    """Compute recency-weighted EWMA mean and robust sample std."""
    if len(values) == 0:
        return 0.0, 1.0
    weights = np.exp(-np.arange(len(values))[::-1] / max(1.0, float(span)))
    weights /= weights.sum()
    mean = float(np.sum(weights * values))
    if len(values) >= 3:
        var = float(np.sum(weights * ((values - mean) ** 2)))
        std = max(0.8, np.sqrt(var * (len(values) / (len(values) - 1.0))))
    else:
        std = max(1.0, 0.28 * max(1.0, mean))
    return mean, float(std)


def _count_over_prob(mean: float, std: float, line: float) -> float:
    """
    Survival probability P(X > line) for discrete count stats (Rebounds, Assists, 3PM,
    Receptions, Passing TDs) using Negative Binomial (when overdispersed) or Poisson.
    """
    if line < 0.0:
        return clamp_prob(0.995)
    mu = max(0.05, mean)
    k_floor = int(np.floor(line))  # e.g. line 5.5 -> P(X >= 6) = 1 - CDF(5)
    var = max(mu * 1.02, std * std)
    if var <= mu * 1.06:
        return clamp_prob(float(1.0 - poisson.cdf(k_floor, mu=mu)))
    p = mu / var
    n = (mu * mu) / (var - mu)
    return clamp_prob(float(1.0 - nbinom.cdf(k_floor, n=n, p=p)))


def _yardage_over_prob(mean: float, std: float, line: float) -> float:
    """
    Survival probability P(X > line) for right-skewed NFL yardage props using Gamma distribution.
    """
    if line <= 0.0:
        return clamp_prob(0.995)
    mu = max(1.0, mean)
    sigma = max(1.5, std)
    shape = (mu / sigma) ** 2
    scale = (sigma * sigma) / mu
    return clamp_prob(float(1.0 - gamma.cdf(line, a=shape, scale=scale)))


def _scoring_over_prob(mean: float, std: float, line: float) -> float:
    """Survival probability P(X > line) for NBA Points & PRA using heavy-tailed Student-t."""
    if line < -0.25:
        return clamp_prob(0.995)
    sigma = max(0.85, std)
    return clamp_prob(float(1.0 - student_t.cdf(line, df=10, loc=mean, scale=sigma * 0.92)))


class PlayerPropsEngine:
    """Generates calibrated pre-game and live in-play player prop distributions."""

    NBA_PROPS = [
        ("points", "Points", _scoring_over_prob),
        ("rebounds", "Rebounds", _count_over_prob),
        ("assists", "Assists", _count_over_prob),
        ("fg3m", "3-Pointers Made", _count_over_prob),
        ("pra", "Pts + Reb + Ast", _scoring_over_prob),
    ]

    NFL_PROPS = [
        ("passing_yards", "Passing Yards", {"QB"}, _yardage_over_prob),
        ("passing_tds", "Passing TDs", {"QB"}, _count_over_prob),
        ("rushing_yards", "Rushing Yards", {"RB", "QB"}, _yardage_over_prob),
        ("receiving_yards", "Receiving Yards", {"WR", "TE", "RB"}, _yardage_over_prob),
        ("receptions", "Receptions", {"WR", "TE", "RB"}, _count_over_prob),
    ]

    def __init__(self) -> None:
        self.nba_logs: pd.DataFrame = pd.DataFrame()
        self.nfl_logs: pd.DataFrame = pd.DataFrame()
        self.nba_opp_def_factor: dict[str, dict[str, float]] = {}
        self.nfl_opp_def_factor: dict[str, dict[str, float]] = {}

    def fit(self, nfl_logs: pd.DataFrame, nba_logs: pd.DataFrame) -> None:
        self.nfl_logs = nfl_logs.sort_values("game_order").reset_index(drop=True)
        self.nba_logs = nba_logs.sort_values("gameday").reset_index(drop=True)
        self._compute_opponent_allowances()

    def _compute_opponent_allowances(self) -> None:
        if len(self.nba_logs) > 0:
            team_game = (
                self.nba_logs.groupby(["opponent", "gameday"], as_index=False)
                .agg(
                    points=("points", "sum"),
                    rebounds=("rebounds", "sum"),
                    assists=("assists", "sum"),
                    fg3m=("fg3m", "sum"),
                )
            )
            lg_means = {
                c: max(1.0, float(team_game[c].mean()))
                for c in ("points", "rebounds", "assists", "fg3m")
            }
            by_opp = team_game.groupby("opponent", as_index=False).mean(numeric_only=True)
            for _, r in by_opp.iterrows():
                opp = str(r["opponent"]).upper()
                pts_f = float(np.clip(r["points"] / lg_means["points"], 0.90, 1.10))
                reb_f = float(np.clip(r["rebounds"] / lg_means["rebounds"], 0.91, 1.09))
                ast_f = float(np.clip(r["assists"] / lg_means["assists"], 0.90, 1.10))
                fg3_f = float(np.clip(r["fg3m"] / lg_means["fg3m"], 0.88, 1.12))
                self.nba_opp_def_factor[opp] = {
                    "points": pts_f,
                    "rebounds": reb_f,
                    "assists": ast_f,
                    "fg3m": fg3_f,
                    "pra": (0.60 * pts_f + 0.22 * reb_f + 0.18 * ast_f),
                }

        if len(self.nfl_logs) > 0:
            team_wk = (
                self.nfl_logs.groupby(["opponent", "season", "week"], as_index=False)
                .agg(
                    passing_yards=("passing_yards", "sum"),
                    passing_tds=("passing_tds", "sum"),
                    rushing_yards=("rushing_yards", "sum"),
                    receiving_yards=("receiving_yards", "sum"),
                    receptions=("receptions", "sum"),
                )
            )
            lg_means = {
                c: max(0.5, float(team_wk[c].mean()))
                for c in (
                    "passing_yards",
                    "passing_tds",
                    "rushing_yards",
                    "receiving_yards",
                    "receptions",
                )
            }
            by_opp = team_wk.groupby("opponent", as_index=False).mean(numeric_only=True)
            for _, r in by_opp.iterrows():
                opp = str(r["opponent"]).upper()
                self.nfl_opp_def_factor[opp] = {
                    c: float(np.clip(r[c] / lg_means[c], 0.92, 1.08))
                    for c in lg_means
                }

    def project_nba_game_props(
        self,
        *,
        home_team: str,
        away_team: str,
        expected_pace: float = 99.2,
        expected_spread: float = 0.0,
        cutoff_date: str | None = None,
        injuries_by_team: dict[str, list[dict[str, Any]]] | None = None,
        custom_lines: dict[tuple[str, str], float] | None = None,
        live_prop_markets: dict[tuple[str, str], dict[str, Any]] | None = None,
        live_state: dict[str, Any] | None = None,
        live_boxscore: dict[str, dict[str, Any]] | None = None,
    ) -> list[PlayerPropProjection]:
        if len(self.nba_logs) == 0:
            return []

        logs = self.nba_logs
        if cutoff_date:
            logs = logs[logs["gameday"] < cutoff_date]
        if len(logs) == 0:
            return []

        is_live = bool(live_state and str(live_state.get("state", "")).upper() == "IN")
        rem_mins = float(live_state.get("remaining_minutes", 48.0)) if is_live and live_state else 48.0
        # Skip live player props inside the final 2.5 minutes or if live boxscore is unavailable
        if is_live and (rem_mins < 2.5 or not live_boxscore):
            return []
        rem_frac = max(0.02, min(1.0, rem_mins / 48.0)) if is_live else 1.0
        live_period = int(live_state.get("period", 1) or 1) if is_live and live_state else 0
        live_abs_diff = (
            abs(float(live_state.get("home_score", 0) or 0) - float(live_state.get("away_score", 0) or 0))
            if is_live and live_state
            else 0.0
        )

        latest_team_map = (
            logs.sort_values("gameday")
            .groupby("player_name", as_index=False)
            .last()
            .set_index("player_name")["team"]
            .to_dict()
        )
        max_day = str(logs["gameday"].max())
        recent_cutoff = (
            pd.to_datetime(max_day) - pd.Timedelta(days=28)
        ).strftime("%Y-%m-%d")

        pace_factor = float(np.clip(expected_pace / 99.2, 0.93, 1.08))
        blowout_haircut = 0.92 if (not is_live and abs(expected_spread) >= 13.5) else 1.0

        out: list[PlayerPropProjection] = []
        for team, opp in ((home_team.upper(), away_team.upper()), (away_team.upper(), home_team.upper())):
            t_logs = logs[
                (logs["team"] == team)
                & (logs["player_name"].map(latest_team_map) == team)
            ]
            if len(t_logs) == 0:
                continue
            recently_active = set(
                t_logs[t_logs["gameday"] >= recent_cutoff]["player_name"].unique()
            )

            out_names: set[str] = set()
            q_names: set[str] = set()
            if injuries_by_team:
                for item in injuries_by_team.get(team, []):
                    st = str(item.get("status", "")).lower()
                    nm = str(item.get("player_name", "")).lower()
                    if "out" in st or "doubtful" in st:
                        out_names.add(nm)
                    elif "questionable" in st:
                        q_names.add(nm)

            player_summaries = (
                t_logs.groupby("player_name", as_index=False)
                .agg(avg_min=("minutes", "mean"), avg_pts=("points", "mean"), n=("points", "count"))
                .sort_values("avg_pts", ascending=False)
            )
            vacated_pts = sum(
                float(r["avg_pts"])
                for _, r in player_summaries.iterrows()
                if str(r["player_name"]).lower() in out_names and float(r["avg_min"]) >= 22.0
            )
            usage_boost = float(np.clip(1.0 + (vacated_pts * 0.0045), 1.0, 1.12))

            top_players = [
                str(r["player_name"])
                for _, r in player_summaries.iterrows()
                if str(r["player_name"]).lower() not in out_names
                and (not recently_active or str(r["player_name"]) in recently_active)
                and float(r["avg_min"]) >= 24.0
                and int(r["n"]) >= 4
            ][:3]

            opp_factors = self.nba_opp_def_factor.get(opp, {})

            for pname in top_players:
                p_live = (live_boxscore or {}).get(pname.lower()) if is_live else None
                if is_live:
                    if p_live is None or p_live.get("ejected") or p_live.get("did_not_play"):
                        continue
                    if live_period >= 2 and float(p_live.get("minutes", 0.0) or 0.0) < 1.0:
                        continue

                p_all = t_logs[t_logs["player_name"] == pname]
                p_all_active = p_all[p_all["minutes"] >= 12.0]
                if len(p_all_active) < 4:
                    continue
                p_hist = p_all_active.tail(15)
                min_vals = p_hist["minutes"].to_numpy(dtype=float)
                ewma_min, _ = _ewma_mean_std(min_vals, span=5)
                season_min = float(np.mean(p_all_active["minutes"]))
                full_proj_min = (0.60 * ewma_min + 0.40 * season_min) * blowout_haircut
                inj_clarity = 0.68 if pname.lower() in q_names else 1.0

                # Live blowout benching & foul-trouble adjustments
                live_min_mult = 1.0
                live_notes: list[str] = []
                if is_live:
                    if live_period >= 4 and live_abs_diff >= 18.0:
                        live_min_mult *= 0.38
                        live_notes.append(f"Q4 blowout ({int(live_abs_diff)} pt diff) starter bench risk")
                    elif live_period == 3 and live_abs_diff >= 20.0:
                        live_min_mult *= 0.68
                        live_notes.append(f"Blowout ({int(live_abs_diff)} pt diff) minutes haircut")
                    elif live_period >= 4 and live_abs_diff >= 14.0 and rem_mins <= 6.0:
                        live_min_mult *= 0.55
                        live_notes.append("Late lead/deficit rotation risk")

                    if p_live is not None:
                        fouls = float(p_live.get("fouls", 0.0) or 0.0)
                        if (
                            (live_period == 1 and fouls >= 2)
                            or (live_period == 2 and fouls >= 3)
                            or (live_period == 3 and fouls >= 4)
                            or (live_period >= 4 and fouls >= 5)
                        ):
                            live_min_mult *= 0.78
                            inj_clarity = min(inj_clarity, 0.72)
                            live_notes.append(f"Foul trouble ({int(fouls)} PF)")

                for col, label, tail_fn in self.NBA_PROPS:
                    vals = p_hist[col].to_numpy(dtype=float)
                    season_vals = p_all_active[col].to_numpy(dtype=float)
                    ewma_val, std_val = _ewma_mean_std(vals, span=7)
                    season_avg = float(np.mean(season_vals))
                    if col == "fg3m" and season_avg < 1.5:
                        continue
                    if col == "assists" and season_avg < 3.5:
                        continue
                    if col == "rebounds" and season_avg < 4.5:
                        continue

                    season_rate = season_avg / max(15.0, season_min)
                    recent_rate = ewma_val / max(15.0, ewma_min)
                    blended_rate = 0.55 * season_rate + 0.45 * recent_rate

                    opp_mult = float(np.clip(opp_factors.get(col, 1.0), 0.94, 1.06))
                    pace_adj = float(np.clip(pace_factor, 0.96, 1.04))
                    boost_mult = (
                        usage_boost
                        if col in ("points", "assists", "pra")
                        else (1.0 + (usage_boost - 1.0) * 0.5)
                    )

                    base_std = max(std_val, 0.22 * season_avg)
                    mkt_prop = (
                        live_prop_markets.get((pname.lower(), col))
                        if live_prop_markets
                        else None
                    )
                    mkt_anchor = 0.55 * season_avg + 0.45 * ewma_val

                    if is_live:
                        cur_stat = float(p_live.get(col, 0.0) or 0.0) if p_live else 0.0
                        rem_proj_min = max(0.25, full_proj_min * rem_frac * live_min_mult)
                        rem_mean = blended_rate * rem_proj_min * pace_adj * opp_mult * boost_mult
                        proj_mean = cur_stat + rem_mean
                        eff_std = max(
                            0.65,
                            base_std * float(np.sqrt(max(0.04, rem_proj_min / max(12.0, full_proj_min)))),
                        )

                        if custom_lines and (pname, col) in custom_lines:
                            line_val = float(custom_lines[(pname, col)])
                            line_src = "Custom Live Line"
                        elif mkt_prop is not None:
                            line_val = float(mkt_prop["line"])
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                        else:
                            # Realistic live line tracks current accumulated stat + remaining market expectation
                            line_val = float(max(0.5, np.floor(cur_stat + rem_frac * mkt_anchor) + 0.5))
                            line_src = "Live Est Line"

                        # Skip low-count props late in games when no live sportsbook prop feed is attached
                        rem_mkt_mean = rem_frac * mkt_anchor
                        if (
                            mkt_prop is None
                            and not (custom_lines and (pname, col) in custom_lines)
                            and rem_mkt_mean < 0.85
                        ):
                            continue

                        # If player already cleared an old pregame line and no fresh live line exists, skip stale auto-win
                        if cur_stat > line_val and not (custom_lines and (pname, col) in custom_lines):
                            continue

                        need_rem = line_val - cur_stat
                        raw_over_p = tail_fn(rem_mean, eff_std, need_rem)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(rem_mkt_mean, eff_std, need_rem) - 0.50)
                        )
                        eff_mkt_over = mkt_prop["over_prob"] if mkt_prop else round(est_mkt_over, 4)
                        eff_dk_over = (
                            mkt_prop.get("dk_over_prob") if mkt_prop else round(est_mkt_over, 4)
                        )
                        drivers = [
                            f"LIVE: Has {cur_stat:.0f} now + {rem_mean:.1f} rem ({rem_proj_min:.1f}m left) = {proj_mean:.1f} vs {line_val:.1f}"
                        ] + live_notes[:2]
                    else:
                        cur_stat = 0.0
                        raw_proj_mean = blended_rate * full_proj_min * pace_adj * opp_mult * boost_mult
                        eff_std = base_std
                        if custom_lines and (pname, col) in custom_lines:
                            line_val = float(custom_lines[(pname, col)])
                            line_src = "Custom Line"
                            proj_mean = raw_proj_mean
                        elif mkt_prop is not None:
                            line_val = float(mkt_prop["line"])
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                            proj_mean = raw_proj_mean
                        else:
                            line_val = float(max(0.5, np.floor(mkt_anchor) + 0.5))
                            line_src = "Model Ref Line"
                            # Shrink toward market anchor when comparing against a synthetic ref line
                            proj_mean = mkt_anchor + 0.65 * (raw_proj_mean - mkt_anchor)

                        raw_over_p = tail_fn(proj_mean, eff_std, line_val)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(mkt_anchor, eff_std, line_val) - 0.50)
                        )
                        eff_mkt_over = (
                            mkt_prop["over_prob"] if mkt_prop else round(est_mkt_over, 4)
                        )
                        eff_dk_over = (
                            mkt_prop.get("dk_over_prob") if mkt_prop else round(est_mkt_over, 4)
                        )
                        src_note = f"{line_src} — verify DK line/¢" if line_src == "Model Ref Line" else line_src
                        drivers = [
                            f"Proj {proj_mean:.1f} vs {line_val:.1f} ({src_note}, {full_proj_min:.1f}m)",
                        ]
                        if usage_boost > 1.02:
                            drivers.append(f"+{(usage_boost - 1.0)*100:.1f}% usage boost from injured teammates")
                        if abs(opp_mult - 1.0) >= 0.03:
                            dir_txt = "soft" if opp_mult > 1.0 else "stingy"
                            drivers.append(f"{opp} {dir_txt} positional defense ({opp_mult:.2f}x)")
                        if abs(pace_factor - 1.0) >= 0.02:
                            drivers.append(f"Matchup pace factor {pace_factor:.2f}x")

                    out.append(
                        PlayerPropProjection(
                            sport="NBA",
                            player_name=pname,
                            team=team,
                            opponent=opp,
                            prop_type=col,
                            prop_label=label,
                            line_value=line_val,
                            projected_mean=round(proj_mean, 2),
                            projected_std=round(eff_std, 2),
                            over_prob=round(over_p, 4),
                            under_prob=round(1.0 - over_p, 4),
                            drivers=drivers[:3],
                            injury_clarity=inj_clarity,
                            current_stat=cur_stat,
                            is_live=is_live,
                            market_over_prob=eff_mkt_over,
                            dk_over_prob=eff_dk_over,
                            line_source=line_src,
                        )
                    )
        return out

    def project_nfl_game_props(
        self,
        *,
        home_team: str,
        away_team: str,
        expected_spread: float = 0.0,
        expected_total: float = 45.5,
        is_indoor: bool = False,
        wind_mph: float = 5.0,
        max_game_order: int | None = None,
        injuries_by_team: dict[str, list[dict[str, Any]]] | None = None,
        custom_lines: dict[tuple[str, str], float] | None = None,
        live_prop_markets: dict[tuple[str, str], dict[str, Any]] | None = None,
        live_state: dict[str, Any] | None = None,
        live_boxscore: dict[str, dict[str, Any]] | None = None,
    ) -> list[PlayerPropProjection]:
        if len(self.nfl_logs) == 0:
            return []

        logs = self.nfl_logs
        if max_game_order is not None:
            logs = logs[logs["game_order"] < max_game_order]
        if len(logs) == 0:
            return []

        is_live = bool(live_state and str(live_state.get("state", "")).upper() == "IN")
        rem_mins = float(live_state.get("remaining_minutes", 60.0)) if is_live and live_state else 60.0
        if is_live and (rem_mins < 3.0 or not live_boxscore):
            return []
        rem_frac = max(0.02, min(1.0, rem_mins / 60.0)) if is_live else 1.0
        live_period = int(live_state.get("period", 1) or 1) if is_live and live_state else 0

        latest_nfl_team = (
            logs.sort_values("game_order")
            .groupby("player_name", as_index=False)
            .last()
            .set_index("player_name")["team"]
            .to_dict()
        )
        max_order = int(logs["game_order"].max())

        env_pass_mult = 1.03 if is_indoor else (0.91 if wind_mph >= 16.0 else 1.0)
        tot_mult = float(np.clip(expected_total / 45.5, 0.90, 1.10))

        out: list[PlayerPropProjection] = []
        for team, opp, is_home in (
            (home_team.upper(), away_team.upper(), True),
            (away_team.upper(), home_team.upper(), False),
        ):
            t_logs = logs[
                (logs["team"] == team)
                & (logs["player_name"].map(latest_nfl_team) == team)
            ]
            if len(t_logs) == 0:
                continue

            out_names: set[str] = set()
            q_names: set[str] = set()
            if injuries_by_team:
                for item in injuries_by_team.get(team, []):
                    st = str(item.get("status", "")).lower()
                    nm = str(item.get("player_name", "")).lower()
                    if "out" in st or "doubtful" in st:
                        out_names.add(nm)
                    elif "questionable" in st:
                        q_names.add(nm)

            # Game-script pass/rush volume tilt (uses live scoreboard deficit when in-play!)
            if is_live and live_state is not None:
                h_sc = float(live_state.get("home_score", 0) or 0)
                a_sc = float(live_state.get("away_score", 0) or 0)
                team_deficit = (a_sc - h_sc) if is_home else (h_sc - a_sc)  # >0 means trailing
                trailing_tilt = float(np.clip(team_deficit / 10.0, -1.2, 1.2))
                pass_script = 1.0 + 0.12 * trailing_tilt
                rush_script = 1.0 - 0.14 * trailing_tilt
            else:
                team_spread = expected_spread if is_home else -expected_spread
                trailing_tilt = float(np.clip(team_spread / 10.0, -0.8, 0.8))
                pass_script = 1.0 + 0.04 * trailing_tilt
                rush_script = 1.0 - 0.05 * trailing_tilt

            opp_factors = self.nfl_opp_def_factor.get(opp, {})

            recent_t_logs = t_logs[t_logs["game_order"] >= max_order - 6]
            if len(recent_t_logs) == 0:
                recent_t_logs = t_logs

            active_qb1: str | None = None
            qb_logs = recent_t_logs[
                (recent_t_logs["position"].str.upper() == "QB")
                & (~recent_t_logs["player_name"].str.lower().isin(out_names))
            ]
            if len(qb_logs) > 0:
                active_qb1 = str(
                    qb_logs.sort_values(["game_order", "attempts"]).iloc[-1]["player_name"]
                )

            grouped = (
                recent_t_logs.groupby(["player_name", "position"], as_index=False)
                .agg(
                    n=("game_order", "count"),
                    py=("passing_yards", "mean"),
                    ry=("rushing_yards", "mean"),
                    recy=("receiving_yards", "mean"),
                )
            )
            grouped = grouped[grouped["n"] >= 2]

            for _, prow in grouped.iterrows():
                pname = str(prow["player_name"])
                pos = str(prow["position"]).upper()
                if pname.lower() in out_names:
                    continue
                if pos == "QB" and (pname != active_qb1 or float(prow["py"]) < 145.0):
                    continue
                if pos == "RB" and float(prow["ry"]) < 38.0 and float(prow["recy"]) < 22.0:
                    continue
                if pos in ("WR", "TE") and float(prow["recy"]) < 38.0:
                    continue

                p_live = (live_boxscore or {}).get(pname.lower()) if is_live else None
                if is_live and p_live is None and live_period >= 2:
                    continue
                p_all = t_logs[t_logs["player_name"] == pname]
                p_hist = p_all.tail(12)
                inj_clarity = 0.68 if pname.lower() in q_names else 1.0

                for col, label, allowed_pos, tail_fn in self.NFL_PROPS:
                    if pos not in allowed_pos:
                        continue
                    vals = p_hist[col].to_numpy(dtype=float)
                    ewma_val, std_val = _ewma_mean_std(vals, span=5)
                    if col == "rushing_yards" and ewma_val < 28.0:
                        continue
                    if col == "receiving_yards" and ewma_val < 34.0:
                        continue
                    if col == "receptions" and ewma_val < 3.2:
                        continue
                    if col == "passing_tds" and ewma_val < 1.05:
                        continue

                    opp_mult = float(opp_factors.get(col, 1.0))
                    if "passing" in col or "receiving" in col or col == "receptions":
                        script_env = pass_script * env_pass_mult * (1.0 if is_live else tot_mult)
                    else:
                        script_env = rush_script * (1.0 if is_live else tot_mult)

                    season_avg = float(np.mean(p_all[col].to_numpy(dtype=float)))
                    shrunk_base = 0.55 * season_avg + 0.45 * ewma_val
                    base_std = max(std_val, 0.25 * season_avg)
                    mkt_anchor = 0.60 * season_avg + 0.40 * ewma_val
                    mkt_prop = (
                        live_prop_markets.get((pname.lower(), col))
                        if live_prop_markets
                        else None
                    )

                    if is_live:
                        cur_stat = float(p_live.get(col, 0.0) or 0.0) if p_live else 0.0
                        rem_mean = shrunk_base * rem_frac * float(np.clip(opp_mult, 0.93, 1.07)) * script_env
                        proj_mean = cur_stat + rem_mean
                        eff_std = max(0.6, base_std * float(np.sqrt(max(0.04, rem_frac))))

                        if custom_lines and (pname, col) in custom_lines:
                            line_val = float(custom_lines[(pname, col)])
                            line_src = "Custom Live Line"
                        elif mkt_prop is not None:
                            line_val = float(mkt_prop["line"])
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                        else:
                            line_val = float(max(0.5, np.floor(cur_stat + rem_frac * mkt_anchor) + 0.5))
                            line_src = "Live Est Line"

                        rem_mkt_mean = rem_frac * mkt_anchor
                        if (
                            mkt_prop is None
                            and not (custom_lines and (pname, col) in custom_lines)
                            and rem_mkt_mean < 0.85
                        ):
                            continue

                        if cur_stat > line_val and not (custom_lines and (pname, col) in custom_lines):
                            continue

                        need_rem = line_val - cur_stat
                        raw_over_p = tail_fn(rem_mean, eff_std, need_rem)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(rem_mkt_mean, eff_std, need_rem) - 0.50)
                        )
                        eff_mkt_over = mkt_prop["over_prob"] if mkt_prop else round(est_mkt_over, 4)
                        eff_dk_over = (
                            mkt_prop.get("dk_over_prob") if mkt_prop else round(est_mkt_over, 4)
                        )
                        drivers = [
                            f"LIVE: Has {cur_stat:.0f} now + {rem_mean:.1f} rem ({rem_mins:.0f}m left) = {proj_mean:.1f} vs {line_val:.1f}"
                        ]
                        if abs(trailing_tilt) >= 0.6:
                            script_txt = "trailing pass-heavy script" if trailing_tilt > 0 else "leading clock-kill script"
                            drivers.append(f"Live {script_txt}")
                    else:
                        cur_stat = 0.0
                        raw_proj_mean = shrunk_base * float(np.clip(opp_mult, 0.93, 1.07)) * script_env
                        eff_std = base_std
                        if custom_lines and (pname, col) in custom_lines:
                            line_val = float(custom_lines[(pname, col)])
                            line_src = "Custom Line"
                            proj_mean = raw_proj_mean
                        elif mkt_prop is not None:
                            line_val = float(mkt_prop["line"])
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                            proj_mean = raw_proj_mean
                        else:
                            line_val = float(max(0.5, np.floor(mkt_anchor) + 0.5))
                            line_src = "Model Ref Line"
                            proj_mean = mkt_anchor + 0.65 * (raw_proj_mean - mkt_anchor)

                        raw_over_p = tail_fn(proj_mean, eff_std, line_val)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(mkt_anchor, eff_std, line_val) - 0.50)
                        )
                        eff_mkt_over = (
                            mkt_prop["over_prob"] if mkt_prop else round(est_mkt_over, 4)
                        )
                        eff_dk_over = (
                            mkt_prop.get("dk_over_prob") if mkt_prop else round(est_mkt_over, 4)
                        )
                        src_note = f"{line_src} — verify DK line/¢" if line_src == "Model Ref Line" else line_src
                        drivers = [
                            f"Proj {proj_mean:.1f} vs {line_val:.1f} ({src_note}, EWMA {ewma_val:.1f})"
                        ]
                        if abs(opp_mult - 1.0) >= 0.03:
                            drivers.append(f"vs {opp} unit factor {opp_mult:.2f}x")
                        if not is_indoor and wind_mph >= 16.0 and "passing" in col:
                            drivers.append(f"Wind ({wind_mph:.0f} mph) dampens air yardage")
                        elif is_indoor and ("passing" in col or "receiving" in col):
                            drivers.append("Fast indoor/dome track boost")

                    out.append(
                        PlayerPropProjection(
                            sport="NFL",
                            player_name=pname,
                            team=team,
                            opponent=opp,
                            prop_type=col,
                            prop_label=label,
                            line_value=line_val,
                            projected_mean=round(proj_mean, 2),
                            projected_std=round(eff_std, 2),
                            over_prob=round(over_p, 4),
                            under_prob=round(1.0 - over_p, 4),
                            drivers=drivers[:3],
                            injury_clarity=inj_clarity,
                            current_stat=cur_stat,
                            is_live=is_live,
                            market_over_prob=eff_mkt_over,
                            dk_over_prob=eff_dk_over,
                            line_source=line_src,
                        )
                    )
        return out

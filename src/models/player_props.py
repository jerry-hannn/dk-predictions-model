"""NFL & NBA Player Props Engine with Pre-Game & Live In-Play Boxscore Conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom, poisson, t as student_t

from src.data.espn_client import classify_injury_status, is_promoted_starter_comment
from src.engine.dk_math import clamp_prob, normalize_player_name


_ROLE_SHIFT_MIN_GAP: dict[str, float] = {
    "passing_yards": 35.0,
    "passing_tds": 0.55,
    "rushing_yards": 16.0,
    "receiving_yards": 16.0,
    "receptions": 1.5,
    "points": 4.5,
    "pra": 6.5,
    "rebounds": 2.0,
    "assists": 2.0,
    "fg3m": 1.0,
}


def _apply_market_role_shift_guard(
    *,
    pname: str,
    col: str,
    proj_mean: float,
    eff_std: float,
    mkt_line: float,
    inj_clarity: float,
    has_role_news: bool = False,
    relief_only_qb: bool = False,
) -> tuple[float, float, float, str | None]:
    """
    Detect when a verified live sportsbook line diverges sharply from a player's historical-log
    baseline (e.g. a backup promoted into a starter role after breaking injury/lineup news, or
    a starter returning on a minutes limit). Dampens stale-role projections toward the market,
    widens variance, and lowers `injury_clarity` so the model does not fire blind high-edge
    Unders on backups stepping into starter snaps/minutes.
    """
    if mkt_line <= 0.0:
        return proj_mean, eff_std, inj_clarity, None

    min_gap = _ROLE_SHIFT_MIN_GAP.get(col, 4.0)
    up_gap = mkt_line - proj_mean
    down_gap = proj_mean - mkt_line
    up_ratio = mkt_line / max(0.5, proj_mean)
    down_ratio = proj_mean / max(0.5, mkt_line)

    # 1. Relief-only backup QB starting: historical logs are garbage-time cameos, not full starts
    if relief_only_qb and col in ("passing_yards", "passing_tds") and up_gap > 0.0:
        adj_mean = 0.15 * proj_mean + 0.85 * mkt_line
        adj_std = max(eff_std, 0.30 * mkt_line)
        adj_clarity = min(inj_clarity, 0.55)
        note = (
            f"Limited full-start history for {pname} means the {mkt_line:.1f} line reflects an expanded starting role."
        )
        return adj_mean, adj_std, adj_clarity, note

    # 2. Upward role shift (backup promoted / starter out / late lineup news)
    up_ratio_thresh = 1.22 if has_role_news else 1.38
    up_gap_thresh = (0.65 * min_gap) if has_role_news else min_gap
    if up_gap >= up_gap_thresh and up_ratio >= up_ratio_thresh:
        # Severity scales with how far market sits above historical baseline (capped at 0.85)
        sev = float(np.clip(0.52 + 0.35 * (up_ratio - up_ratio_thresh), 0.55, 0.85))
        if has_role_news:
            sev = min(0.88, sev + 0.10)
        adj_mean = (1.0 - sev) * proj_mean + sev * mkt_line
        adj_std = max(eff_std, 0.28 * mkt_line)
        adj_clarity = min(inj_clarity, 0.58)
        note = (
            f"The posted {mkt_line:.1f} line sits well above {pname}'s prior-role average ({proj_mean:.1f}), signaling an expanded role or lineup change."
        )
        return adj_mean, adj_std, adj_clarity, note

    # 3. Extreme downward role shift (minutes restriction / demotion / return from injury)
    if down_gap >= (1.15 * min_gap) and down_ratio >= 1.48:
        sev = float(np.clip(0.50 + 0.30 * (down_ratio - 1.48), 0.52, 0.80))
        adj_mean = (1.0 - sev) * proj_mean + sev * mkt_line
        adj_std = max(eff_std, 0.28 * mkt_line)
        adj_clarity = min(inj_clarity, 0.62)
        note = (
            f"The posted {mkt_line:.1f} line sits well below {pname}'s prior average ({proj_mean:.1f}), signaling a reduced role or minutes limit."
        )
        return adj_mean, adj_std, adj_clarity, note

    return proj_mean, eff_std, inj_clarity, None


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
    default_line_value: float | None = None
    is_custom_line: bool = False
    book_lines: dict[str, dict[str, Any]] | None = None
    projected_median: float | None = None
    position: str = ""
    context_drivers: list[str] | None = None
    live_time_note: str = ""


_PLAIN_STAT_NAMES: dict[str, str] = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "fg3m": "three-pointers",
    "pra": "points, rebounds, and assists",
    "passing_yards": "passing yards",
    "passing_tds": "passing touchdowns",
    "rushing_yards": "rushing yards",
    "receiving_yards": "receiving yards",
    "receptions": "receptions",
}

_POSITION_PLURALS: dict[str, str] = {
    "QB": "quarterbacks",
    "RB": "running backs",
    "WR": "wide receivers",
    "TE": "tight ends",
    "G": "guards",
    "F": "forwards",
    "C": "centers",
}


def _plain_stat_name(col: str, fallback: str) -> str:
    return _PLAIN_STAT_NAMES.get(col, fallback.lower())


def _plain_pos_plural(pos: str) -> str:
    return _POSITION_PLURALS.get(str(pos or "").upper(), "")


def _lookup_player_prop(
    mapping: dict[tuple[str, str], Any] | None,
    pname: str,
    col: str,
) -> Any | None:
    if not mapping:
        return None
    for key in (
        (pname, col),
        (pname.lower(), col),
        (normalize_player_name(pname), col),
    ):
        if key in mapping:
            return mapping[key]
    return None


def _has_any_player_prop(
    mapping: dict[tuple[str, str], Any] | None,
    pname: str,
) -> bool:
    if not mapping:
        return False
    candidates = {pname, pname.lower(), normalize_player_name(pname)}
    return any(k[0] in candidates for k in mapping)


def _lookup_live_player(
    live_boxscore: dict[str, dict[str, Any]] | None,
    pname: str,
) -> dict[str, Any] | None:
    if not live_boxscore:
        return None
    return live_boxscore.get(pname.lower()) or live_boxscore.get(normalize_player_name(pname))


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


def prop_over_prob_for_line(
    sport: str,
    prop_type: str,
    mean: float,
    std: float,
    line: float,
    *,
    shrink_factor: float = 0.78,
) -> float:
    """
    Evaluate calibrated Over probability P(X > line) using the exact stat-appropriate
    parametric distribution (Student-t for scoring, NegBin/Poisson for counts, Gamma for yards).
    """
    pt = str(prop_type).lower()
    if pt in ("points", "pra"):
        raw = _scoring_over_prob(mean, std, line)
    elif pt in ("passing_yards", "rushing_yards", "receiving_yards"):
        raw = _yardage_over_prob(mean, std, line)
    else:
        raw = _count_over_prob(mean, std, line)
    return clamp_prob(0.50 + shrink_factor * (raw - 0.50))


def prop_median_for_type(prop_type: str, mean: float, std: float) -> float:
    """
    Compute the 50th-percentile (typical/median) outcome for the prop's exact distribution.
    For right-skewed yardage (Gamma) and count (NegBin/Poisson) stats, the median sits below
    the arithmetic mean when game-to-game variance is large.
    """
    pt = str(prop_type).lower()
    if pt in ("points", "pra"):
        return round(float(mean), 2)
    if pt in ("passing_yards", "rushing_yards", "receiving_yards"):
        mu = max(1.0, float(mean))
        sigma = max(1.5, float(std))
        shape = (mu / sigma) ** 2
        scale = (sigma * sigma) / mu
        return round(float(gamma.ppf(0.5, a=shape, scale=scale)), 2)
    mu = max(0.05, float(mean))
    var = max(mu * 1.02, float(std) ** 2)
    if var <= mu * 1.06:
        return round(float(poisson.ppf(0.5, mu=mu)), 2)
    p = mu / var
    n = (mu * mu) / (var - mu)
    return round(float(nbinom.ppf(0.5, n=n, p=p)), 2)


def build_prop_primary_reason(
    *,
    player_name: str,
    prop_type: str,
    prop_label: str,
    side: str,
    line_val: float,
    proj_mean: float,
    proj_median: float | None,
    side_prob: float,
    line_source: str = "DraftKings",
    is_live: bool = False,
    current_stat: float = 0.0,
    live_time_note: str = "",
) -> str:
    """
    Build a concise, side- and line-aware plain-English primary reason that never contradicts
    the recommended Over/Under side (accounting for right-skewed medians and price-driven picks).
    """
    stat_name = _plain_stat_name(prop_type, prop_label)
    side_u = "OVER" if str(side).upper() in ("OVER", "YES") else "UNDER"

    if is_live:
        time_part = f" with {live_time_note}" if live_time_note else ""
        return (
            f"{player_name} has {current_stat:.0f} {stat_name} so far and projects to finish "
            f"around {proj_mean:.1f}{time_part}, against the {line_val:.1f} line."
        )

    is_est = line_source == "Model Ref Line"
    est_adj = "estimated " if is_est else ""
    verify_note = "; verify the posted DraftKings line" if is_est else ""

    if side_u == "OVER":
        if proj_mean >= line_val + 0.25:
            gap = proj_mean - line_val
            return (
                f"The model projects {player_name} for {proj_mean:.1f} {stat_name}, "
                f"{gap:.1f} above the {est_adj}{line_val:.1f} line{verify_note}."
            )
        return (
            f"The model projects {player_name} near {proj_mean:.1f} {stat_name} with a "
            f"{side_prob * 100:.0f}% chance to clear the {est_adj}{line_val:.1f} line, "
            f"making the current price attractive{verify_note}."
        )

    # UNDER side
    if proj_mean <= line_val - 0.25:
        gap = line_val - proj_mean
        return (
            f"The model projects {player_name} for {proj_mean:.1f} {stat_name}, "
            f"{gap:.1f} below the {est_adj}{line_val:.1f} line{verify_note}."
        )
    if proj_median is not None and proj_median <= line_val + 0.05 and proj_mean > line_val:
        return (
            f"Boom-or-bust game logs put {player_name}'s typical outcome around {proj_median:.1f} "
            f"{stat_name}, below the {est_adj}{line_val:.1f} line even though spike games lift "
            f"the average to {proj_mean:.1f}{verify_note}."
        )
    return (
        f"The model gives {player_name} a {side_prob * 100:.0f}% chance to stay under the "
        f"{est_adj}{line_val:.1f} {stat_name} line (projecting {proj_mean:.1f}), making the "
        f"current price attractive{verify_note}."
    )


def _format_positional_defense_driver(
    *,
    opp: str,
    pos: str,
    stat_name: str,
    pos_info: dict[str, float] | None,
    fallback_mult: float,
) -> str | None:
    """Format a specific plain-English opponent positional defense reason when meaningful."""
    pos_plural = _plain_pos_plural(pos)
    if pos_info and pos_plural:
        pct_diff = float(pos_info.get("pct_diff", 0.0))
        opp_avg = float(pos_info.get("opp_avg", 0.0))
        lg_avg = float(pos_info.get("lg_avg", 0.0))
        factor = float(pos_info.get("factor", 1.0))
        if abs(pct_diff) >= 5.0 or abs(factor - 1.0) >= 0.025:
            if pct_diff > 0:
                return (
                    f"{opp} allows {abs(pct_diff):.0f}% more {stat_name} per game to {pos_plural} "
                    f"than the league average ({opp_avg:.1f} vs. {lg_avg:.1f})."
                )
            return (
                f"{opp} holds {pos_plural} to {abs(pct_diff):.0f}% fewer {stat_name} per game "
                f"than the league average ({opp_avg:.1f} vs. {lg_avg:.1f})."
            )
        return None

    if abs(fallback_mult - 1.0) >= 0.03:
        pct_diff = abs(fallback_mult - 1.0) * 100.0
        target_txt = pos_plural or "opponents"
        if fallback_mult > 1.0:
            return f"{opp} allows {pct_diff:.0f}% more {stat_name} per game to {target_txt} than average."
        return f"{opp} holds {target_txt} to {pct_diff:.0f}% fewer {stat_name} per game than average."
    return None


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
        self.nba_opp_pos_stats: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        self.nfl_opp_pos_stats: dict[tuple[str, str], dict[str, dict[str, float]]] = {}

    def fit(self, nfl_logs: pd.DataFrame, nba_logs: pd.DataFrame) -> None:
        from src.data.nba_loader import NBALoader

        self.nfl_logs = (
            nfl_logs.sort_values("game_order").reset_index(drop=True)
            if len(nfl_logs) > 0 and "game_order" in nfl_logs.columns
            else pd.DataFrame()
        )
        nba_sorted = (
            nba_logs.sort_values("gameday").reset_index(drop=True)
            if len(nba_logs) > 0 and "gameday" in nba_logs.columns
            else pd.DataFrame()
        )
        if len(nba_sorted) > 0 and "position" not in nba_sorted.columns:
            ldr = NBALoader()
            nba_sorted = ldr.attach_player_positions(nba_sorted, ldr.load_player_positions())
        self.nba_logs = nba_sorted
        self._team_map_cache = {}
        self._compute_opponent_allowances()

    def __getstate__(self) -> dict[str, Any]:
        # Derived lookup caches are rebuilt lazily; keep them out of the pickled artifact
        state = self.__dict__.copy()
        state.pop("_team_map_cache", None)
        return state

    def _cached_latest_team_map(
        self, league: str, logs: pd.DataFrame, order_col: str, *, cacheable: bool
    ) -> dict[str, str]:
        """
        Map each player to the team on their most recent log row. This full-table sort + groupby
        is identical for every game on a slate, so memoize it when no as-of cutoff is applied.
        """
        cache: dict[str, dict[str, str]] | None = getattr(self, "_team_map_cache", None)
        if cacheable and cache is not None and league in cache:
            return cache[league]
        team_map = (
            logs.sort_values(order_col)
            .groupby("player_name", as_index=False)
            .last()
            .set_index("player_name")["team"]
            .to_dict()
        )
        if cacheable:
            if cache is None:
                cache = {}
                self._team_map_cache = cache
            cache[league] = team_map
        return team_map

    def _compute_opponent_allowances(self) -> None:
        self.nba_opp_def_factor = {}
        self.nfl_opp_def_factor = {}
        self.nba_opp_pos_stats = {}
        self.nfl_opp_pos_stats = {}

        if len(self.nba_logs) > 0:
            if "position" not in self.nba_logs.columns:
                from src.data.nba_loader import NBALoader

                ldr = NBALoader()
                self.nba_logs = ldr.attach_player_positions(
                    self.nba_logs, ldr.load_player_positions()
                )

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

            # Position-specific NBA opponent allowances (G / F / C) reindexed over all team games
            all_nba_games = team_game[["opponent", "gameday"]].drop_duplicates()
            nba_stat_cols = ("points", "rebounds", "assists", "fg3m", "pra")
            work_nba = self.nba_logs.copy()
            if "pra" not in work_nba.columns:
                work_nba["pra"] = work_nba["points"] + work_nba["rebounds"] + work_nba["assists"]
            for pos in ("G", "F", "C"):
                sub = work_nba[work_nba["position"].astype(str).str.upper() == pos]
                if len(sub) == 0:
                    continue
                pos_game = (
                    sub.groupby(["opponent", "gameday"], as_index=False)[list(nba_stat_cols)]
                    .sum()
                )
                pos_full = all_nba_games.merge(pos_game, on=["opponent", "gameday"], how="left").fillna(0.0)
                pos_lg = {c: max(0.5, float(pos_full[c].mean())) for c in nba_stat_cols}
                pos_by_opp = pos_full.groupby("opponent", as_index=False).agg(
                    n_games=("gameday", "count"),
                    **{c: (c, "mean") for c in nba_stat_cols},
                )
                for _, r in pos_by_opp.iterrows():
                    opp = str(r["opponent"]).upper()
                    n_g = float(r["n_games"])
                    shrink_w = n_g / (n_g + 16.0)
                    stats_map: dict[str, dict[str, float]] = {}
                    for c in nba_stat_cols:
                        opp_avg = float(r[c])
                        lg_avg = pos_lg[c]
                        raw_ratio = opp_avg / lg_avg
                        shrunk = float(np.clip(1.0 + shrink_w * (raw_ratio - 1.0), 0.90, 1.10))
                        stats_map[c] = {
                            "factor": round(shrunk, 4),
                            "opp_avg": round(opp_avg, 2),
                            "lg_avg": round(lg_avg, 2),
                            "pct_diff": round((raw_ratio - 1.0) * 100.0, 1),
                            "n_games": n_g,
                        }
                    self.nba_opp_pos_stats[(opp, pos)] = stats_map

        if len(self.nfl_logs) > 0:
            nfl_stat_cols = (
                "passing_yards",
                "passing_tds",
                "rushing_yards",
                "receiving_yards",
                "receptions",
            )
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
            lg_means = {c: max(0.5, float(team_wk[c].mean())) for c in nfl_stat_cols}
            by_opp = team_wk.groupby("opponent", as_index=False).mean(numeric_only=True)
            for _, r in by_opp.iterrows():
                opp = str(r["opponent"]).upper()
                self.nfl_opp_def_factor[opp] = {
                    c: float(np.clip(r[c] / lg_means[c], 0.92, 1.08))
                    for c in lg_means
                }

            # Position-specific NFL opponent allowances (QB / RB / WR / TE)
            if "position" in self.nfl_logs.columns:
                all_nfl_games = team_wk[["opponent", "season", "week"]].drop_duplicates()
                pos_col = self.nfl_logs["position"].astype(str).str.upper()
                for pos in ("QB", "RB", "WR", "TE"):
                    sub = self.nfl_logs[pos_col == pos]
                    if len(sub) == 0:
                        continue
                    pos_wk = (
                        sub.groupby(["opponent", "season", "week"], as_index=False)[list(nfl_stat_cols)]
                        .sum()
                    )
                    pos_full = all_nfl_games.merge(
                        pos_wk, on=["opponent", "season", "week"], how="left"
                    ).fillna(0.0)
                    pos_lg = {c: max(0.15, float(pos_full[c].mean())) for c in nfl_stat_cols}
                    pos_by_opp = pos_full.groupby("opponent", as_index=False).agg(
                        n_games=("week", "count"),
                        **{c: (c, "mean") for c in nfl_stat_cols},
                    )
                    for _, r in pos_by_opp.iterrows():
                        opp = str(r["opponent"]).upper()
                        n_g = float(r["n_games"])
                        shrink_w = n_g / (n_g + 12.0)
                        stats_map = {}
                        for c in nfl_stat_cols:
                            opp_avg = float(r[c])
                            lg_avg = pos_lg[c]
                            raw_ratio = opp_avg / lg_avg
                            shrunk = float(np.clip(1.0 + shrink_w * (raw_ratio - 1.0), 0.89, 1.11))
                            stats_map[c] = {
                                "factor": round(shrunk, 4),
                                "opp_avg": round(opp_avg, 2),
                                "lg_avg": round(lg_avg, 2),
                                "pct_diff": round((raw_ratio - 1.0) * 100.0, 1),
                                "n_games": n_g,
                            }
                        self.nfl_opp_pos_stats[(opp, pos)] = stats_map

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

        latest_team_map = self._cached_latest_team_map(
            "nba", logs, "gameday", cacheable=not cutoff_date
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
            promoted_names: set[str] = set()
            if injuries_by_team:
                for item in injuries_by_team.get(team, []):
                    cls_st = classify_injury_status(item)
                    nm = str(item.get("player_name", "")).lower()
                    nm_norm = normalize_player_name(nm)
                    if cls_st in ("OUT", "DOUBTFUL"):
                        out_names.add(nm)
                        if nm_norm:
                            out_names.add(nm_norm)
                    elif cls_st == "QUESTIONABLE":
                        q_names.add(nm)
                        if nm_norm:
                            q_names.add(nm_norm)
                    if is_promoted_starter_comment(item):
                        promoted_names.add(nm)
                        if nm_norm:
                            promoted_names.add(nm_norm)

            player_summaries = (
                t_logs.groupby("player_name", as_index=False)
                .agg(
                    avg_min=("minutes", "mean"),
                    avg_pts=("points", "mean"),
                    n=("points", "count"),
                    pos=("position", "last"),
                )
                .sort_values("avg_pts", ascending=False)
            )
            nba_logs_by_name = {
                str(name): grp for name, grp in t_logs.groupby("player_name", sort=False)
            }
            sidelined_info: list[dict[str, Any]] = []
            for _, r in player_summaries.iterrows():
                s_name = str(r["player_name"])
                if (
                    s_name.lower() in out_names
                    or normalize_player_name(s_name) in out_names
                ) and float(r["avg_min"]) >= 20.0:
                    s_grp = nba_logs_by_name.get(s_name)
                    s_days = (
                        set(s_grp[s_grp["minutes"] >= 12.0]["gameday"].astype(str))
                        if s_grp is not None
                        else set()
                    )
                    sidelined_info.append(
                        {
                            "name": s_name,
                            "avg_min": float(r["avg_min"]),
                            "avg_pts": float(r["avg_pts"]),
                            "pos": str(r.get("pos", "")).upper(),
                            "played_days": s_days,
                        }
                    )
            sidelined_stars = [s["name"] for s in sidelined_info[:2]]

            if live_prop_markets or custom_lines:
                cand_rows = [
                    str(r["player_name"])
                    for _, r in player_summaries.iterrows()
                    if str(r["player_name"]).lower() not in out_names
                    and normalize_player_name(str(r["player_name"])) not in out_names
                    and (not recently_active or str(r["player_name"]) in recently_active)
                    and int(r["n"]) >= 3
                    and (
                        float(r["avg_min"]) >= 24.0
                        or (
                            float(r["avg_min"]) >= 12.0
                            and (
                                _has_any_player_prop(live_prop_markets, str(r["player_name"]))
                                or _has_any_player_prop(custom_lines, str(r["player_name"]))
                                or str(r["player_name"]).lower() in promoted_names
                                or normalize_player_name(str(r["player_name"])) in promoted_names
                            )
                        )
                    )
                ]
                custom_p = [p for p in cand_rows if _has_any_player_prop(custom_lines, p)]
                other_p = [p for p in cand_rows if p not in custom_p]
                top_players = (custom_p + other_p)[:8]
            else:
                top_players = [
                    str(r["player_name"])
                    for _, r in player_summaries.iterrows()
                    if str(r["player_name"]).lower() not in out_names
                    and normalize_player_name(str(r["player_name"])) not in out_names
                    and (not recently_active or str(r["player_name"]) in recently_active)
                    and float(r["avg_min"]) >= 24.0
                    and int(r["n"]) >= 4
                ][:3]

            if not getattr(self, "nba_opp_pos_stats", None) and len(self.nba_logs) > 0:
                self._compute_opponent_allowances()
            opp_factors = self.nba_opp_def_factor.get(opp, {})
            nba_pos_stats = getattr(self, "nba_opp_pos_stats", {}) or {}

            for pname in top_players:
                p_live = _lookup_live_player(live_boxscore, pname) if is_live else None
                if is_live:
                    if p_live is None or p_live.get("ejected") or p_live.get("did_not_play"):
                        continue
                    if live_period >= 2 and float(p_live.get("minutes", 0.0) or 0.0) < 1.0:
                        continue

                p_all = nba_logs_by_name.get(pname, t_logs.iloc[0:0])
                p_all_active = p_all[p_all["minutes"] >= 10.0]
                if len(p_all_active) < 3:
                    continue
                pos = (
                    str(p_all["position"].iloc[-1]).upper()
                    if "position" in p_all.columns and len(p_all) > 0
                    else ""
                )
                opp_pos_map = nba_pos_stats.get((opp, pos), {}) if pos else {}

                p_hist = p_all_active.tail(15)
                p_recent_days = set(p_all_active.tail(8)["gameday"].astype(str))
                min_vals = p_hist["minutes"].to_numpy(dtype=float)
                ewma_min, _ = _ewma_mean_std(min_vals, span=5)
                season_min = float(np.mean(p_all_active["minutes"]))
                base_proj_min = 0.60 * ewma_min + 0.40 * season_min
                is_promoted = (
                    pname.lower() in promoted_names
                    or normalize_player_name(pname) in promoted_names
                )

                # Overlap-aware minutes & usage redistribution when rotation teammates are out
                vacated_same_pos_min = 0.0
                vacated_other_pos_min = 0.0
                vacated_pts = 0.0
                sidelined_same_pos: list[str] = []
                for s_info in sidelined_info:
                    s_days = s_info["played_days"]
                    if p_recent_days and s_days:
                        overlap = len(p_recent_days & s_days) / len(p_recent_days)
                        # If teammate played in player's recent logs, historical stats don't yet reflect their absence
                        w_ov = max(0.30, overlap) if overlap > 0.0 else 0.15
                    else:
                        w_ov = 0.50
                    vacated_pts += s_info["avg_pts"] * w_ov
                    if pos and s_info["pos"] == pos:
                        vacated_same_pos_min += s_info["avg_min"] * w_ov
                        sidelined_same_pos.append(s_info["name"])
                    else:
                        vacated_other_pos_min += s_info["avg_min"] * w_ov

                headroom = max(0.0, 36.5 - base_proj_min)
                raw_min_add = 0.40 * vacated_same_pos_min + 0.12 * vacated_other_pos_min
                if is_promoted and raw_min_add < 6.0:
                    raw_min_add = max(raw_min_add, 8.0)
                min_cap = 14.0 if (is_promoted or vacated_same_pos_min >= 20.0) else 8.0
                min_boost = (
                    min(headroom, min_cap, raw_min_add * min(1.0, headroom / 10.0))
                    if headroom > 0.5 and raw_min_add > 0.5
                    else 0.0
                )
                usage_boost = float(np.clip(1.0 + (vacated_pts * 0.0045), 1.0, 1.14))
                full_proj_min = (base_proj_min + min_boost) * blowout_haircut
                base_inj_clarity = (
                    0.68
                    if (pname.lower() in q_names or normalize_player_name(pname) in q_names)
                    else 1.0
                )

                # Live blowout benching & foul-trouble adjustments
                live_min_mult = 1.0
                live_notes: list[str] = []
                if is_live:
                    if live_period >= 4 and live_abs_diff >= 18.0:
                        live_min_mult *= 0.38
                        live_notes.append(
                            f"With a {int(live_abs_diff)}-point fourth-quarter blowout, starters may sit early."
                        )
                    elif live_period == 3 and live_abs_diff >= 20.0:
                        live_min_mult *= 0.68
                        live_notes.append(
                            f"A {int(live_abs_diff)}-point blowout could cut into this player's remaining minutes."
                        )
                    elif live_period >= 4 and live_abs_diff >= 14.0 and rem_mins <= 6.0:
                        live_min_mult *= 0.55
                        live_notes.append(
                            "A large late-game margin could limit playing time down the stretch."
                        )

                    if p_live is not None:
                        fouls = float(p_live.get("fouls", 0.0) or 0.0)
                        if (
                            (live_period == 1 and fouls >= 2)
                            or (live_period == 2 and fouls >= 3)
                            or (live_period == 3 and fouls >= 4)
                            or (live_period >= 4 and fouls >= 5)
                        ):
                            live_min_mult *= 0.78
                            base_inj_clarity = min(base_inj_clarity, 0.72)
                            live_notes.append(
                                f"Foul trouble ({int(fouls)} personal fouls) could reduce remaining playing time."
                            )

                for col, label, tail_fn in self.NBA_PROPS:
                    vals = p_hist[col].to_numpy(dtype=float)
                    season_vals = p_all_active[col].to_numpy(dtype=float)
                    ewma_val, std_val = _ewma_mean_std(vals, span=7)
                    season_avg = float(np.mean(season_vals))

                    mkt_prop = _lookup_player_prop(live_prop_markets, pname, col)
                    custom_val = _lookup_player_prop(custom_lines, pname, col)
                    has_explicit_line = mkt_prop is not None or custom_val is not None

                    if col == "fg3m" and season_avg < (0.8 if has_explicit_line else 1.5):
                        continue
                    if col == "assists" and season_avg < (1.2 if has_explicit_line else 3.5):
                        continue
                    if col == "rebounds" and season_avg < (1.5 if has_explicit_line else 4.5):
                        continue

                    season_rate = season_avg / max(12.0, season_min)
                    recent_rate = ewma_val / max(12.0, ewma_min)
                    blended_rate = 0.55 * season_rate + 0.45 * recent_rate

                    pos_info = opp_pos_map.get(col)
                    raw_opp_f = (
                        float(pos_info["factor"])
                        if pos_info is not None
                        else float(opp_factors.get(col, 1.0))
                    )
                    opp_mult = float(np.clip(raw_opp_f, 0.92, 1.08))
                    pace_adj = float(np.clip(pace_factor, 0.96, 1.04))
                    boost_mult = (
                        usage_boost
                        if col in ("points", "assists", "pra")
                        else (1.0 + (usage_boost - 1.0) * 0.5)
                    )

                    base_std = max(std_val, 0.22 * season_avg)
                    mkt_anchor = 0.55 * season_avg + 0.45 * ewma_val
                    stat_name = _plain_stat_name(col, label)
                    live_time_note = ""
                    inj_clarity = base_inj_clarity

                    if is_live:
                        cur_stat = float(p_live.get(col, 0.0) or 0.0) if p_live else 0.0
                        rem_proj_min = max(0.25, full_proj_min * rem_frac * live_min_mult)
                        rem_mean = blended_rate * rem_proj_min * pace_adj * opp_mult * boost_mult
                        proj_mean = cur_stat + rem_mean
                        eff_std = max(
                            0.65,
                            base_std * float(np.sqrt(max(0.04, rem_proj_min / max(12.0, full_proj_min)))),
                        )

                        sync_default_line = (
                            float(mkt_prop["line"])
                            if mkt_prop is not None
                            else float(max(0.5, np.floor(cur_stat + rem_frac * mkt_anchor) + 0.5))
                        )
                        if custom_val is not None:
                            line_val = float(custom_val)
                            is_custom = True
                            line_src = "Custom Live Line"
                        elif mkt_prop is not None:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                        else:
                            # Realistic live line tracks current accumulated stat + remaining market expectation
                            line_val = sync_default_line
                            is_custom = False
                            line_src = "Live Est Line"

                        # Skip low-count props late in games when no live sportsbook prop feed is attached
                        rem_mkt_mean = rem_frac * mkt_anchor
                        if mkt_prop is None and custom_val is None and rem_mkt_mean < 0.85:
                            continue

                        # If player already cleared an old pregame line and no fresh live line exists, skip stale auto-win
                        if cur_stat > line_val and custom_val is None:
                            continue

                        need_rem = line_val - cur_stat
                        raw_over_p = tail_fn(rem_mean, eff_std, need_rem)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        proj_median = round(cur_stat + prop_median_for_type(col, rem_mean, eff_std), 2)
                        rem_mkt_center = (
                            max(0.1, float(mkt_prop["line"]) - cur_stat)
                            if mkt_prop is not None
                            else rem_mkt_mean
                        )
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(rem_mkt_center, eff_std, need_rem) - 0.50)
                        )
                        same_as_mkt = (
                            mkt_prop is not None and abs(line_val - float(mkt_prop["line"])) < 0.01
                        )
                        eff_mkt_over = (
                            mkt_prop["over_prob"]
                            if same_as_mkt and mkt_prop.get("over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        eff_dk_over = (
                            mkt_prop["dk_over_prob"]
                            if same_as_mkt and mkt_prop.get("dk_over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        live_time_note = f"{rem_proj_min:.1f} expected minutes left"
                        default_side = "OVER" if over_p >= 0.50 else "UNDER"
                        primary_driver = build_prop_primary_reason(
                            player_name=pname,
                            prop_type=col,
                            prop_label=label,
                            side=default_side,
                            line_val=line_val,
                            proj_mean=proj_mean,
                            proj_median=proj_median,
                            side_prob=over_p if default_side == "OVER" else (1.0 - over_p),
                            line_source=line_src,
                            is_live=True,
                            current_stat=cur_stat,
                            live_time_note=live_time_note,
                        )
                        context_drivers = list(live_notes[:2])
                        drivers = [primary_driver] + context_drivers
                    else:
                        cur_stat = 0.0
                        raw_proj_mean = blended_rate * full_proj_min * pace_adj * opp_mult * boost_mult
                        eff_std = max(base_std, base_std * (full_proj_min / max(12.0, base_proj_min)))
                        sync_default_line = (
                            float(mkt_prop["line"])
                            if mkt_prop is not None
                            else float(max(0.5, np.floor(mkt_anchor) + 0.5))
                        )
                        proj_mean = (
                            raw_proj_mean
                            if mkt_prop is not None
                            else mkt_anchor + 0.65 * (raw_proj_mean - mkt_anchor)
                        )
                        role_shift_note: str | None = None
                        if mkt_prop is not None:
                            proj_mean, eff_std, inj_clarity, role_shift_note = _apply_market_role_shift_guard(
                                pname=pname,
                                col=col,
                                proj_mean=proj_mean,
                                eff_std=eff_std,
                                mkt_line=float(mkt_prop["line"]),
                                inj_clarity=inj_clarity,
                                has_role_news=(is_promoted or bool(sidelined_same_pos) or min_boost >= 2.5),
                            )
                        if custom_val is not None:
                            line_val = float(custom_val)
                            is_custom = True
                            line_src = "Custom Line"
                        elif mkt_prop is not None:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                        else:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = "Model Ref Line"

                        raw_over_p = tail_fn(proj_mean, eff_std, line_val)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        proj_median = prop_median_for_type(col, proj_mean, eff_std)
                        mkt_center = (
                            float(mkt_prop["line"]) if mkt_prop is not None else mkt_anchor
                        )
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(mkt_center, eff_std, line_val) - 0.50)
                        )
                        same_as_mkt = (
                            mkt_prop is not None and abs(line_val - float(mkt_prop["line"])) < 0.01
                        )
                        eff_mkt_over = (
                            mkt_prop["over_prob"]
                            if same_as_mkt and mkt_prop.get("over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        eff_dk_over = (
                            mkt_prop["dk_over_prob"]
                            if same_as_mkt and mkt_prop.get("dk_over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        default_side = "OVER" if over_p >= 0.50 else "UNDER"
                        primary_driver = build_prop_primary_reason(
                            player_name=pname,
                            prop_type=col,
                            prop_label=label,
                            side=default_side,
                            line_val=line_val,
                            proj_mean=proj_mean,
                            proj_median=proj_median,
                            side_prob=over_p if default_side == "OVER" else (1.0 - over_p),
                            line_source=line_src,
                        )
                        context_drivers = []
                        if role_shift_note:
                            context_drivers.append(role_shift_note)
                        pos_def_txt = _format_positional_defense_driver(
                            opp=opp,
                            pos=pos,
                            stat_name=stat_name,
                            pos_info=pos_info,
                            fallback_mult=opp_mult,
                        )
                        if pos_def_txt:
                            context_drivers.append(pos_def_txt)
                        if min_boost >= 2.0 and (sidelined_same_pos or sidelined_stars):
                            names_txt = " and ".join((sidelined_same_pos or sidelined_stars)[:2])
                            context_drivers.append(
                                f"With {names_txt} sidelined, {pname} is projected for about {min_boost:.0f} extra minutes and a larger offensive role."
                            )
                        elif usage_boost > 1.02:
                            if sidelined_stars:
                                names_txt = " and ".join(sidelined_stars)
                                context_drivers.append(
                                    f"With {names_txt} sidelined, {pname} is projected for a {(usage_boost - 1.0)*100:.1f}% boost in offensive usage."
                                )
                            else:
                                context_drivers.append(
                                    f"Injured teammates are expected to increase {pname}'s offensive usage by {(usage_boost - 1.0)*100:.1f}%."
                                )
                        if blowout_haircut < 1.0:
                            context_drivers.append(
                                f"A lopsided {abs(expected_spread):.1f}-point spread raises the risk of reduced fourth-quarter minutes for starters."
                            )
                        if abs(pace_factor - 1.0) >= 0.02:
                            if pace_factor > 1.0:
                                context_drivers.append(
                                    f"A faster projected tempo ({expected_pace:.1f} possessions per game) creates extra opportunities for {stat_name}."
                                )
                            else:
                                context_drivers.append(
                                    f"A slower projected tempo ({expected_pace:.1f} possessions per game) could limit opportunities for {stat_name}."
                                )
                        if abs(ewma_min - season_min) >= 2.5:
                            trend_dir = "climbed" if ewma_min > season_min else "dipped"
                            cmp_dir = "up from" if ewma_min > season_min else "down from"
                            context_drivers.append(
                                f"{pname}'s playing time has {trend_dir} to {ewma_min:.0f} minutes per game recently ({cmp_dir} {season_min:.0f} on the season), averaging {ewma_val:.1f} {stat_name}."
                            )
                        else:
                            context_drivers.append(
                                f"{pname} is projected for about {full_proj_min:.0f} minutes and has averaged {ewma_val:.1f} {stat_name} over recent games."
                            )
                        drivers = [primary_driver] + context_drivers[:3]

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
                            drivers=drivers[:4],
                            injury_clarity=inj_clarity,
                            current_stat=cur_stat,
                            is_live=is_live,
                            market_over_prob=eff_mkt_over,
                            dk_over_prob=eff_dk_over,
                            line_source=line_src,
                            default_line_value=sync_default_line,
                            is_custom_line=is_custom,
                            book_lines=mkt_prop.get("books") if mkt_prop else None,
                            projected_median=proj_median,
                            position=pos,
                            context_drivers=context_drivers[:3],
                            live_time_note=live_time_note,
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
        temp_f: float = 65.0,
        weather_condition: str = "",
        precip_prob: float = 0.0,
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

        latest_nfl_team = self._cached_latest_team_map(
            "nfl", logs, "game_order", cacheable=max_game_order is None
        )
        max_order = int(logs["game_order"].max())

        # Outdoor NFL weather effects on passing/receiving vs. rushing
        eff_wind = 0.0 if is_indoor else max(0.0, float(wind_mph))
        eff_temp = 70.0 if is_indoor else float(temp_f)
        cond_low = "" if is_indoor else str(weather_condition or "").strip().lower()
        has_snow = (not is_indoor) and any(k in cond_low for k in ("snow", "flurr", "blizzard", "sleet"))
        has_rain = (not is_indoor) and (not has_snow) and any(
            k in cond_low for k in ("rain", "shower", "storm", "drizzle")
        )
        likely_precip = (not is_indoor) and (not has_snow) and (not has_rain) and float(precip_prob or 0.0) >= 60.0

        if is_indoor:
            env_pass_mult = 1.03
            env_rush_mult = 1.00
        else:
            env_pass_mult = 1.0
            env_rush_mult = 1.0
            if eff_wind >= 18.0:
                env_pass_mult *= 0.90
                env_rush_mult *= 1.04
            elif eff_wind >= 14.0:
                env_pass_mult *= 0.94
                env_rush_mult *= 1.02
            elif eff_wind >= 11.0:
                env_pass_mult *= 0.97
                env_rush_mult *= 1.01
            if has_snow:
                env_pass_mult *= 0.92
                env_rush_mult *= 1.03
            elif has_rain:
                env_pass_mult *= 0.95
                env_rush_mult *= 1.02
            elif likely_precip:
                env_pass_mult *= 0.97
                env_rush_mult *= 1.01
            if eff_temp <= 32.0:
                env_pass_mult *= 0.96
                env_rush_mult *= 1.02
            env_pass_mult = float(np.clip(env_pass_mult, 0.84, 1.04))
            env_rush_mult = float(np.clip(env_rush_mult, 0.97, 1.08))

        tot_mult = float(np.clip(expected_total / 45.5, 0.90, 1.10))

        if not getattr(self, "nfl_opp_pos_stats", None) and len(self.nfl_logs) > 0:
            self._compute_opponent_allowances()
        nfl_pos_stats = getattr(self, "nfl_opp_pos_stats", {}) or {}

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
            promoted_names: set[str] = set()
            if injuries_by_team:
                for item in injuries_by_team.get(team, []):
                    cls_st = classify_injury_status(item)
                    nm = str(item.get("player_name", "")).lower()
                    nm_norm = normalize_player_name(nm)
                    if cls_st in ("OUT", "DOUBTFUL"):
                        out_names.add(nm)
                        if nm_norm:
                            out_names.add(nm_norm)
                    elif cls_st == "QUESTIONABLE":
                        q_names.add(nm)
                        if nm_norm:
                            q_names.add(nm_norm)
                    if is_promoted_starter_comment(item):
                        promoted_names.add(nm)
                        if nm_norm:
                            promoted_names.add(nm_norm)

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
                & (~recent_t_logs["player_name"].map(normalize_player_name).isin(out_names))
            ]
            if len(qb_logs) > 0:
                active_qb1 = str(
                    qb_logs.sort_values(["game_order", "attempts"]).iloc[-1]["player_name"]
                )

            grouped = (
                t_logs.groupby(["player_name", "position"], as_index=False)
                .agg(
                    n=("game_order", "count"),
                    recent_n=("game_order", lambda s: int((s >= max_order - 6).sum())),
                    py=("passing_yards", "mean"),
                    ry=("rushing_yards", "mean"),
                    recy=("receiving_yards", "mean"),
                    rec=("receptions", "mean"),
                )
            )
            # Split once (row order preserved within each group) instead of a boolean scan per player
            player_logs_by_name = {
                str(name): grp for name, grp in t_logs.groupby("player_name", sort=False)
            }

            # Track sidelined contributors by position for vacated-workload redistribution
            sidelined_by_pos: dict[str, list[dict[str, Any]]] = {"QB": [], "RB": [], "WR": [], "TE": []}
            active_rb_count = 0
            active_pass_catcher_count = 0
            for _, grow in grouped.iterrows():
                gname = str(grow["player_name"])
                gpos = str(grow["position"]).upper()
                is_g_out = gname.lower() in out_names or normalize_player_name(gname) in out_names
                if is_g_out:
                    if gpos in sidelined_by_pos and (
                        (gpos == "QB" and float(grow["py"]) >= 140.0)
                        or (gpos == "RB" and (float(grow["ry"]) >= 28.0 or float(grow["recy"]) >= 18.0))
                        or (gpos in ("WR", "TE") and float(grow["recy"]) >= 25.0)
                    ):
                        s_grp = player_logs_by_name.get(gname)
                        s_orders = (
                            set(s_grp["game_order"].astype(int))
                            if s_grp is not None and len(s_grp) > 0
                            else set()
                        )
                        sidelined_by_pos[gpos].append(
                            {
                                "name": gname,
                                "pos": gpos,
                                "py": float(grow["py"]),
                                "ry": float(grow["ry"]),
                                "recy": float(grow["recy"]),
                                "rec": float(grow["rec"]),
                                "played_orders": s_orders,
                            }
                        )
                else:
                    if gpos == "RB" and (
                        int(grow["recent_n"]) >= 2
                        or _has_any_player_prop(live_prop_markets, gname)
                    ):
                        active_rb_count += 1
                    if gpos in ("WR", "TE") and (
                        float(grow["recy"]) >= 20.0
                        or _has_any_player_prop(live_prop_markets, gname)
                    ):
                        active_pass_catcher_count += 1

            for _, prow in grouped.iterrows():
                pname = str(prow["player_name"])
                pos = str(prow["position"]).upper()
                if pname.lower() in out_names or normalize_player_name(pname) in out_names:
                    continue
                is_promoted = (
                    pname.lower() in promoted_names
                    or normalize_player_name(pname) in promoted_names
                )
                has_player_mkt = (
                    _has_any_player_prop(live_prop_markets, pname)
                    or _has_any_player_prop(custom_lines, pname)
                )
                if not has_player_mkt and not is_promoted and int(prow["recent_n"]) < 2:
                    continue
                if (has_player_mkt or is_promoted) and int(prow["n"]) < 1:
                    continue
                if (
                    pos == "QB"
                    and not has_player_mkt
                    and not is_promoted
                    and (pname != active_qb1 or float(prow["py"]) < 145.0)
                ):
                    continue
                if (
                    not has_player_mkt
                    and not is_promoted
                    and pos == "RB"
                    and not sidelined_by_pos["RB"]
                    and float(prow["ry"]) < 38.0
                    and float(prow["recy"]) < 22.0
                ):
                    continue
                if (
                    not has_player_mkt
                    and not is_promoted
                    and pos in ("WR", "TE")
                    and not (sidelined_by_pos["WR"] or sidelined_by_pos["TE"])
                    and float(prow["recy"]) < 38.0
                ):
                    continue

                p_live = _lookup_live_player(live_boxscore, pname) if is_live else None
                if is_live and p_live is None and live_period >= 2:
                    continue
                p_all = player_logs_by_name.get(pname, t_logs.iloc[0:0])
                p_hist = p_all.tail(12)
                p_recent_orders = set(p_all.tail(6)["game_order"].astype(int))
                opp_pos_map = nfl_pos_stats.get((opp, pos), {})
                base_inj_clarity = (
                    0.68
                    if (pname.lower() in q_names or normalize_player_name(pname) in q_names)
                    else 1.0
                )

                # For QBs, isolate full-game starts (>=18 pass attempts) so 2-attempt relief/kneel-down
                # cameos don't drag a promoted backup QB's baseline down to garbage-time numbers.
                qb_full_starts = (
                    p_all[p_all["attempts"] >= 18.0]
                    if (pos == "QB" and "attempts" in p_all.columns)
                    else p_all.iloc[0:0]
                )
                relief_only_qb = (
                    pos == "QB"
                    and len(qb_full_starts) < 2
                    and (has_player_mkt or is_promoted or bool(sidelined_by_pos["QB"]))
                )

                # Compute overlap-weighted vacated workload from sidelined teammates
                def _overlap_weight(s_orders: set[int]) -> float:
                    if not p_recent_orders or not s_orders:
                        return 0.55
                    ov = len(p_recent_orders & s_orders) / len(p_recent_orders)
                    # If starter played in the backup's recent logs, backup stats do NOT yet reflect starter's absence
                    return max(0.35, ov) if ov > 0.0 else 0.15

                sidelined_same_pos_names: list[str] = []
                vac_rb_ry = 0.0
                vac_rec_yds = 0.0
                vac_recs = 0.0
                if pos == "RB":
                    for s_rb in sidelined_by_pos["RB"]:
                        w_ov = _overlap_weight(s_rb["played_orders"])
                        vac_rb_ry += s_rb["ry"] * w_ov
                        vac_rec_yds += s_rb["recy"] * w_ov * 0.85
                        vac_recs += s_rb["rec"] * w_ov * 0.85
                        sidelined_same_pos_names.append(s_rb["name"])
                if pos in ("WR", "TE", "RB"):
                    for s_pos_k in ("WR", "TE", "RB"):
                        pos_w = 1.0 if s_pos_k == pos else (0.55 if pos in ("WR", "TE") and s_pos_k in ("WR", "TE") else 0.30)
                        for s_pc in sidelined_by_pos[s_pos_k]:
                            if pos == "RB" and s_pos_k == "RB":
                                continue
                            w_ov = _overlap_weight(s_pc["played_orders"])
                            vac_rec_yds += s_pc["recy"] * w_ov * pos_w
                            vac_recs += s_pc["rec"] * w_ov * pos_w
                            if s_pos_k == pos or (pos in ("WR", "TE") and s_pos_k in ("WR", "TE")):
                                if s_pc["name"] not in sidelined_same_pos_names:
                                    sidelined_same_pos_names.append(s_pc["name"])
                if pos == "QB":
                    for s_qb in sidelined_by_pos["QB"]:
                        sidelined_same_pos_names.append(s_qb["name"])

                # Recent volume metrics for concrete, human-friendly baseline explanations
                vol_df = qb_full_starts.tail(12) if (pos == "QB" and len(qb_full_starts) >= 2) else p_hist
                recent_attempts = (
                    _ewma_mean_std(vol_df["attempts"].to_numpy(dtype=float), span=5)[0]
                    if "attempts" in vol_df.columns and len(vol_df) > 0
                    else 0.0
                )
                recent_carries = (
                    _ewma_mean_std(p_hist["carries"].to_numpy(dtype=float), span=5)[0]
                    if "carries" in p_hist.columns and len(p_hist) > 0
                    else 0.0
                )
                recent_targets = (
                    _ewma_mean_std(p_hist["targets"].to_numpy(dtype=float), span=5)[0]
                    if "targets" in p_hist.columns and len(p_hist) > 0
                    else 0.0
                )

                for col, label, allowed_pos, tail_fn in self.NFL_PROPS:
                    if pos not in allowed_pos:
                        continue
                    use_qb_starts = (
                        pos == "QB"
                        and col in ("passing_yards", "passing_tds")
                        and len(qb_full_starts) >= 2
                    )
                    stat_all = qb_full_starts if use_qb_starts else p_all
                    stat_hist = stat_all.tail(12)
                    vals = stat_hist[col].to_numpy(dtype=float)
                    ewma_val, std_val = _ewma_mean_std(vals, span=5)

                    mkt_prop = _lookup_player_prop(live_prop_markets, pname, col)
                    custom_val = _lookup_player_prop(custom_lines, pname, col)
                    has_explicit_line = mkt_prop is not None or custom_val is not None

                    season_avg = float(np.mean(stat_all[col].to_numpy(dtype=float)))
                    shrunk_base = 0.55 * season_avg + 0.45 * ewma_val

                    # Positional teammate-out workload redistribution
                    teammate_boost_note: str | None = None
                    base_before_red = shrunk_base
                    if pos == "RB" and col == "rushing_yards" and vac_rb_ry >= 12.0:
                        rb_share = 0.52 / max(1, min(2, active_rb_count))
                        add_ry = min(55.0, vac_rb_ry * rb_share)
                        shrunk_base = min(118.0, shrunk_base + add_ry)
                        if sidelined_same_pos_names and (shrunk_base - base_before_red) >= 8.0:
                            names_txt = " and ".join(sidelined_same_pos_names[:2])
                            teammate_boost_note = (
                                f"With {names_txt} sidelined, {pname} is projected for an expanded rushing workload."
                            )
                    elif col == "receiving_yards" and vac_rec_yds >= 15.0:
                        pc_div = max(1.5, min(3.5, float(active_pass_catcher_count or 2)))
                        add_recy = min(28.0, (vac_rec_yds * 0.58) / pc_div)
                        shrunk_base = min(115.0, shrunk_base + add_recy)
                        if sidelined_same_pos_names and (shrunk_base - base_before_red) >= 5.0:
                            names_txt = " and ".join(sidelined_same_pos_names[:2])
                            teammate_boost_note = (
                                f"With {names_txt} sidelined, {pname} is projected for an expanded target share."
                            )
                    elif col == "receptions" and vac_recs >= 1.2:
                        pc_div = max(1.5, min(3.5, float(active_pass_catcher_count or 2)))
                        add_rec = min(2.4, (vac_recs * 0.58) / pc_div)
                        shrunk_base = min(9.5, shrunk_base + add_rec)
                        if sidelined_same_pos_names and (shrunk_base - base_before_red) >= 0.45:
                            names_txt = " and ".join(sidelined_same_pos_names[:2])
                            teammate_boost_note = (
                                f"With {names_txt} sidelined, {pname} is projected for an expanded target share."
                            )

                    eff_vol = max(ewma_val, shrunk_base)
                    allow_backup_mkt = has_explicit_line and (
                        is_promoted or bool(sidelined_same_pos_names) or relief_only_qb
                    )
                    if not allow_backup_mkt:
                        if col == "rushing_yards" and eff_vol < (12.0 if has_explicit_line else 28.0):
                            continue
                        if col == "receiving_yards" and eff_vol < (15.0 if has_explicit_line else 34.0):
                            continue
                        if col == "receptions" and eff_vol < (1.5 if has_explicit_line else 3.2):
                            continue
                        if col == "passing_tds" and eff_vol < (0.6 if has_explicit_line else 1.05):
                            continue

                    pos_info = opp_pos_map.get(col)
                    raw_opp_f = (
                        float(pos_info["factor"])
                        if pos_info is not None
                        else float(opp_factors.get(col, 1.0))
                    )
                    opp_mult = float(np.clip(raw_opp_f, 0.90, 1.10))
                    is_air_prop = "passing" in col or "receiving" in col or col == "receptions"
                    if is_air_prop:
                        script_env = pass_script * env_pass_mult * (1.0 if is_live else tot_mult)
                    else:
                        script_env = rush_script * env_rush_mult * (1.0 if is_live else tot_mult)

                    base_std = max(std_val, 0.25 * max(season_avg, shrunk_base))
                    mkt_anchor = 0.60 * season_avg + 0.40 * ewma_val
                    if shrunk_base > base_before_red:
                        mkt_anchor = max(mkt_anchor, shrunk_base)
                    stat_name = _plain_stat_name(col, label)
                    live_time_note = ""
                    inj_clarity = base_inj_clarity

                    if is_live:
                        cur_stat = float(p_live.get(col, 0.0) or 0.0) if p_live else 0.0
                        rem_mean = shrunk_base * rem_frac * opp_mult * script_env
                        proj_mean = cur_stat + rem_mean
                        eff_std = max(0.6, base_std * float(np.sqrt(max(0.04, rem_frac))))

                        sync_default_line = (
                            float(mkt_prop["line"])
                            if mkt_prop is not None
                            else float(max(0.5, np.floor(cur_stat + rem_frac * mkt_anchor) + 0.5))
                        )
                        if custom_val is not None:
                            line_val = float(custom_val)
                            is_custom = True
                            line_src = "Custom Live Line"
                        elif mkt_prop is not None:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                        else:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = "Live Est Line"

                        rem_mkt_mean = rem_frac * mkt_anchor
                        if mkt_prop is None and custom_val is None and rem_mkt_mean < 0.85:
                            continue

                        if cur_stat > line_val and custom_val is None:
                            continue

                        need_rem = line_val - cur_stat
                        raw_over_p = tail_fn(rem_mean, eff_std, need_rem)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        proj_median = round(cur_stat + prop_median_for_type(col, rem_mean, eff_std), 2)
                        rem_mkt_center = (
                            max(0.1, float(mkt_prop["line"]) - cur_stat)
                            if mkt_prop is not None
                            else rem_mkt_mean
                        )
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(rem_mkt_center, eff_std, need_rem) - 0.50)
                        )
                        same_as_mkt = (
                            mkt_prop is not None and abs(line_val - float(mkt_prop["line"])) < 0.01
                        )
                        eff_mkt_over = (
                            mkt_prop["over_prob"]
                            if same_as_mkt and mkt_prop.get("over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        eff_dk_over = (
                            mkt_prop["dk_over_prob"]
                            if same_as_mkt and mkt_prop.get("dk_over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        mins_left = int(round(rem_mins))
                        live_time_note = (
                            "less than a minute left"
                            if rem_mins < 1.0
                            else f"{mins_left} {'minute' if mins_left == 1 else 'minutes'} left"
                        )
                        default_side = "OVER" if over_p >= 0.50 else "UNDER"
                        primary_driver = build_prop_primary_reason(
                            player_name=pname,
                            prop_type=col,
                            prop_label=label,
                            side=default_side,
                            line_val=line_val,
                            proj_mean=proj_mean,
                            proj_median=proj_median,
                            side_prob=over_p if default_side == "OVER" else (1.0 - over_p),
                            line_source=line_src,
                            is_live=True,
                            current_stat=cur_stat,
                            live_time_note=live_time_note,
                        )
                        context_drivers: list[str] = []
                        if abs(trailing_tilt) >= 0.6:
                            if trailing_tilt > 0:
                                context_drivers.append(
                                    "Trailing on the scoreboard should push the offense into a heavier passing game."
                                )
                            else:
                                context_drivers.append(
                                    "Protecting a lead should shift the offense toward running the ball to burn clock."
                                )
                        drivers = [primary_driver] + context_drivers
                    else:
                        cur_stat = 0.0
                        raw_proj_mean = shrunk_base * opp_mult * script_env
                        eff_std = base_std
                        sync_default_line = (
                            float(mkt_prop["line"])
                            if mkt_prop is not None
                            else float(max(0.5, np.floor(mkt_anchor) + 0.5))
                        )
                        proj_mean = (
                            raw_proj_mean
                            if mkt_prop is not None
                            else mkt_anchor + 0.65 * (raw_proj_mean - mkt_anchor)
                        )
                        role_shift_note: str | None = None
                        if mkt_prop is not None:
                            proj_mean, eff_std, inj_clarity, role_shift_note = _apply_market_role_shift_guard(
                                pname=pname,
                                col=col,
                                proj_mean=proj_mean,
                                eff_std=eff_std,
                                mkt_line=float(mkt_prop["line"]),
                                inj_clarity=inj_clarity,
                                has_role_news=(
                                    is_promoted
                                    or bool(sidelined_same_pos_names)
                                    or teammate_boost_note is not None
                                ),
                                relief_only_qb=relief_only_qb,
                            )
                        if custom_val is not None:
                            line_val = float(custom_val)
                            is_custom = True
                            line_src = "Custom Line"
                        elif mkt_prop is not None:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = str(mkt_prop.get("source", "DraftKings"))
                        else:
                            line_val = sync_default_line
                            is_custom = False
                            line_src = "Model Ref Line"

                        raw_over_p = tail_fn(proj_mean, eff_std, line_val)
                        over_p = clamp_prob(0.50 + 0.78 * (raw_over_p - 0.50))
                        proj_median = prop_median_for_type(col, proj_mean, eff_std)
                        mkt_center = (
                            float(mkt_prop["line"]) if mkt_prop is not None else mkt_anchor
                        )
                        est_mkt_over = clamp_prob(
                            0.50 + 0.78 * (tail_fn(mkt_center, eff_std, line_val) - 0.50)
                        )
                        same_as_mkt = (
                            mkt_prop is not None and abs(line_val - float(mkt_prop["line"])) < 0.01
                        )
                        eff_mkt_over = (
                            mkt_prop["over_prob"]
                            if same_as_mkt and mkt_prop.get("over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        eff_dk_over = (
                            mkt_prop["dk_over_prob"]
                            if same_as_mkt and mkt_prop.get("dk_over_prob") is not None
                            else round(est_mkt_over, 4)
                        )
                        default_side = "OVER" if over_p >= 0.50 else "UNDER"
                        primary_driver = build_prop_primary_reason(
                            player_name=pname,
                            prop_type=col,
                            prop_label=label,
                            side=default_side,
                            line_val=line_val,
                            proj_mean=proj_mean,
                            proj_median=proj_median,
                            side_prob=over_p if default_side == "OVER" else (1.0 - over_p),
                            line_source=line_src,
                        )
                        context_drivers = []
                        if role_shift_note:
                            context_drivers.append(role_shift_note)
                        if teammate_boost_note:
                            context_drivers.append(teammate_boost_note)

                        # 1. Position-specific opponent defense
                        pos_def_txt = _format_positional_defense_driver(
                            opp=opp,
                            pos=pos,
                            stat_name=stat_name,
                            pos_info=pos_info,
                            fallback_mult=opp_mult,
                        )
                        if pos_def_txt:
                            context_drivers.append(pos_def_txt)

                        # 2. Outdoor weather or indoor dome environment
                        if not is_indoor:
                            if is_air_prop:
                                if has_snow:
                                    wind_part = f" and {eff_wind:.0f} mph winds" if eff_wind >= 10.0 else ""
                                    context_drivers.append(
                                        f"Outdoor snow ({eff_temp:.0f}°F{wind_part}) should make throwing and catching tougher."
                                    )
                                elif has_rain:
                                    wind_part = f" and {eff_wind:.0f} mph winds" if eff_wind >= 10.0 else ""
                                    context_drivers.append(
                                        f"Outdoor rain ({eff_temp:.0f}°F{wind_part}) tends to dampen passing volume and accuracy."
                                    )
                                elif likely_precip:
                                    context_drivers.append(
                                        f"A {float(precip_prob):.0f}% chance of wet outdoor weather ({eff_temp:.0f}°F) could limit downfield passing."
                                    )
                                elif eff_wind >= 14.0:
                                    cold_part = f" and {eff_temp:.0f}°F temperatures" if eff_temp <= 38.0 else ""
                                    context_drivers.append(
                                        f"Strong {eff_wind:.0f} mph outdoor winds{cold_part} should limit downfield passing."
                                    )
                                elif eff_temp <= 32.0:
                                    context_drivers.append(
                                        f"Freezing {eff_temp:.0f}°F outdoor temperatures typically reduce passing efficiency."
                                    )
                            else:
                                # Rushing prop in adverse outdoor weather
                                if has_snow or has_rain or eff_wind >= 15.0 or eff_temp <= 32.0:
                                    cond_bits = []
                                    if has_snow:
                                        cond_bits.append("snow")
                                    elif has_rain:
                                        cond_bits.append("rain")
                                    if eff_wind >= 14.0:
                                        cond_bits.append(f"{eff_wind:.0f} mph winds")
                                    if eff_temp <= 32.0:
                                        cond_bits.append(f"{eff_temp:.0f}°F cold")
                                    if len(cond_bits) >= 3:
                                        cond_desc = ", ".join(cond_bits[:-1]) + f", and {cond_bits[-1]}"
                                    else:
                                        cond_desc = " and ".join(cond_bits)
                                    context_drivers.append(
                                        f"Tough outdoor weather ({cond_desc}) typically shifts offenses toward heavier rushing volume."
                                    )
                        elif is_air_prop:
                            context_drivers.append(
                                "Playing indoors on fast turf provides a clean environment for the passing game."
                            )

                        # 3. Game total / pregame script context
                        if expected_total >= 48.5:
                            context_drivers.append(
                                f"A high {expected_total:.1f}-point projected game total points to extra offensive volume."
                            )
                        elif expected_total <= 41.0:
                            context_drivers.append(
                                f"A low {expected_total:.1f}-point projected game total could cap overall offensive volume."
                            )
                        elif abs(trailing_tilt) >= 0.35:
                            if is_air_prop and trailing_tilt > 0:
                                context_drivers.append(
                                    f"Projected as an underdog, {team} may need to lean more heavily on the passing game."
                                )
                            elif is_air_prop and trailing_tilt < 0:
                                context_drivers.append(
                                    f"As a solid favorite, {team} could lean more on the run if they build a lead."
                                )
                            elif (not is_air_prop) and trailing_tilt < 0:
                                context_drivers.append(
                                    f"As a solid favorite, {team} is positioned to lean on the run game to protect a lead."
                                )
                            elif (not is_air_prop) and trailing_tilt > 0:
                                context_drivers.append(
                                    f"Projected as an underdog, {team} could shift away from the run if they fall behind."
                                )

                        # 4. Concrete recent workload & form baseline
                        if "passing" in col and recent_attempts >= 15.0:
                            context_drivers.append(
                                f"{pname} has averaged {recent_attempts:.1f} pass attempts and {ewma_val:.1f} {stat_name} per game recently."
                            )
                        elif col == "rushing_yards" and recent_carries >= 5.0:
                            context_drivers.append(
                                f"{pname} has averaged {recent_carries:.1f} carries and {ewma_val:.1f} rushing yards per game recently."
                            )
                        elif ("receiving" in col or col == "receptions") and recent_targets >= 2.5:
                            context_drivers.append(
                                f"{pname} has drawn {recent_targets:.1f} targets per game recently, averaging {ewma_val:.1f} {stat_name}."
                            )
                        else:
                            context_drivers.append(
                                f"{pname} has averaged {ewma_val:.1f} {stat_name} over recent games ({season_avg:.1f} across the full sample)."
                            )

                        drivers = [primary_driver] + context_drivers[:3]

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
                            drivers=drivers[:4],
                            injury_clarity=inj_clarity,
                            current_stat=cur_stat,
                            is_live=is_live,
                            market_over_prob=eff_mkt_over,
                            dk_over_prob=eff_dk_over,
                            line_source=line_src,
                            default_line_value=sync_default_line,
                            is_custom_line=is_custom,
                            book_lines=mkt_prop.get("books") if mkt_prop else None,
                            projected_median=proj_median,
                            position=pos,
                            context_drivers=context_drivers[:3],
                            live_time_note=live_time_note,
                        )
                    )
        return out

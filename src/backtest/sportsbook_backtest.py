"""Strict chronological walk-forward Strategy Verification & Profitability Backtester for Traditional DK/FD Sportsbooks.

Evaluates all 5 Traditional DK/FD strategies against actual out-of-sample (OOS) historical NFL & NBA
game outcomes with zero lookahead leakage:
1. +EV Line Shopping (DK + FD Best-Book vs. Single-Book DK baseline)
2. Cross-Book Arbitrage (DK <-> FD and fee-adjusted DKeX <-> DK/FD, Exact & Rounded-Guarded stakes)
3. Middles & Polish Middles (Model-Filtered +EV Middles vs. Blind Middles on actual margins/totals)
4. Low-Hold Rollover, Bonus Bet Conversions ($50 tokens), & Profit Boosts (+25% / +50%)
5. Steam / Off-Market Lag against Sharp-Only Consensus
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.metrics import brier_score_loss, log_loss

from src.backtest.walk_forward import compute_calibration_bins
from src.config import get_settings
from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.data.odds_client import devig_american_pair
from src.engine.dk_math import clamp_prob, inv_logit, logit, prob_to_american_odds
from src.engine.sportsbook_math import (
    NFL_KEY_NUMBERS,
    american_to_decimal,
    compute_two_way_hold,
    decimal_to_american,
    evaluate_arbitrage_pair,
    evaluate_bonus_bet_conversion,
    evaluate_middle_opportunity,
    evaluate_sportsbook_side,
    evaluate_steam_lag,
)
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel


@dataclass(frozen=True)
class MarketRegime:
    name: str
    label: str
    ml_vig: float            # Two-way ML overround (e.g., 0.045 = 4.5%)
    spread_vig: float        # Two-way spread/total overround (e.g., 0.0455 = -110/-110)
    book_prob_std: float     # Cross-book price opinion std between DK & FD
    line_gap_prob: float     # Frequency of >= 0.5pt spread/total discrepancy between DK & FD
    steam_lag_prob: float    # Frequency of one book lagging a sharp steam move
    steam_move_size: float   # Typical sharp steam move magnitude in implied probability


REGIMES: dict[str, MarketRegime] = {
    "standard": MarketRegime(
        name="standard",
        label="Standard US Market (4.5% Vig)",
        ml_vig=0.045,
        spread_vig=0.0455,
        book_prob_std=0.017,
        line_gap_prob=0.26,
        steam_lag_prob=0.14,
        steam_move_size=0.042,
    ),
    "tight_vig": MarketRegime(
        name="tight_vig",
        label="Tight Retail Stress (4.8% Vig)",
        ml_vig=0.048,
        spread_vig=0.0485,
        book_prob_std=0.012,
        line_gap_prob=0.18,
        steam_lag_prob=0.09,
        steam_move_size=0.034,
    ),
    "active_steam": MarketRegime(
        name="active_steam",
        label="Active Steam / High Dispersion",
        ml_vig=0.045,
        spread_vig=0.0455,
        book_prob_std=0.024,
        line_gap_prob=0.34,
        steam_lag_prob=0.22,
        steam_move_size=0.052,
    ),
}


def _prob_to_vigged_pair(p_yes: float, total_vig: float) -> tuple[int, int]:
    """Convert true two-way probability into realistic vigged American odds pair."""
    p = clamp_prob(p_yes, 0.04)
    scale = 1.0 + max(0.01, float(total_vig))
    q_yes = clamp_prob(p * scale, 0.05)
    q_no = clamp_prob((1.0 - p) * scale, 0.05)
    am_yes = decimal_to_american(1.0 / q_yes)
    am_no = decimal_to_american(1.0 / q_no)
    # Snap near-even lines cleanly to realistic US increments (-105, -110, etc.)
    if -100 < am_yes < 100:
        am_yes = -105 if q_yes >= 0.5 else 100
    if -100 < am_no < 100:
        am_no = -105 if q_no >= 0.5 else 100
    return int(am_yes), int(am_no)


def _blend_log_odds(p_model: float, p_market: float, market_weight: float = 0.55) -> float:
    return clamp_prob(
        inv_logit(
            market_weight * logit(clamp_prob(p_market))
            + (1.0 - market_weight) * logit(clamp_prob(p_model))
        )
    )


def _one_sided_p_value(returns: list[float]) -> tuple[float | None, float | None, float | None]:
    """Compute (sharpe_ratio, t_stat, one_sided_p_value) for per-bet returns."""
    n = len(returns)
    if n < 5:
        return None, None, None
    arr = np.array(returns, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < 1e-9:
        if mean > 0:
            return 99.0, 99.0, 0.0
        return 0.0, 0.0, 0.5
    sharpe = (mean / std) * math.sqrt(min(n, 250))
    t_stat = mean / (std / math.sqrt(n))
    p_val = float(1.0 - norm.cdf(t_stat))
    return round(sharpe, 2), round(t_stat, 2), round(p_val, 4)


def _collect_oos_game_frames(sport: str) -> dict[str, list[dict[str, Any]]]:
    """
    Generate strict chronological expanding-window OOS game records with zero future leakage.
    Trains strictly on past games (`gameday < fold_start`) and predicts step-by-step on final 40%.
    """
    sp = sport.lower()
    frames: dict[str, list[dict[str, Any]]] = {}

    if sp in ("all", "nfl"):
        nfl_sched = NFLLoader().load_schedules()
        model = NFLGameModel()
        feat_df, _ = model.build_chronological_features(nfl_sched)
        done = feat_df.dropna(subset=["home_score", "away_score"]).copy()
        done = done[done["home_score"] != done["away_score"]].reset_index(drop=True)

        split_idx = int(len(done) * 0.60)
        step = 64
        nfl_records: list[dict[str, Any]] = []

        for start in range(split_idx, len(done), step):
            end = min(len(done), start + step)
            cutoff_day = str(done.iloc[start]["gameday"])
            train_slice = nfl_sched[nfl_sched["gameday"] < cutoff_day]
            if len(train_slice.dropna(subset=["home_score"])) < 60:
                train_slice = nfl_sched.iloc[:start]
            fold_model = NFLGameModel()
            fold_model.fit(train_slice)

            test_slice = done.iloc[start:end]
            X_test = test_slice[NFLGameModel.FEATURE_COLS].to_numpy(dtype=float)
            X_tot = test_slice[NFLGameModel.TOTAL_FEATURE_COLS].to_numpy(dtype=float)
            assert fold_model.win_clf is not None
            assert fold_model.margin_reg is not None
            assert fold_model.total_reg is not None

            clf_probs = fold_model.win_clf.predict_proba(X_test)[:, 1]
            exp_margins = fold_model.margin_reg.predict(X_test)
            exp_totals = fold_model.total_reg.predict(X_tot)

            # Strictly historical empirical NFL final-margin distribution (zero lookahead!)
            train_done = train_slice.dropna(subset=["home_score", "away_score"])
            train_abs_margins = np.abs(
                train_done["home_score"].to_numpy(dtype=float)
                - train_done["away_score"].to_numpy(dtype=float)
            )
            emp_key_pmf: dict[int, float] = {}
            if len(train_abs_margins) >= 40:
                for k_val in range(1, 22):
                    emp_key_pmf[k_val] = float(np.mean(np.round(train_abs_margins) == k_val))

            for idx_r, clf_p, em, et in zip(test_slice.index, clf_probs, exp_margins, exp_totals):
                r = done.loc[idx_r]
                margin_wp = float(1.0 - norm.cdf(0.0, loc=float(em), scale=fold_model.residual_std))
                mod_p = clamp_prob(0.55 * float(clf_p) + 0.45 * margin_wp)
                h_sc = float(r["home_score"])
                a_sc = float(r["away_score"])
                act_margin = h_sc - a_sc
                act_total = h_sc + a_sc
                sp_line = (
                    float(r["home_spread"])
                    if pd.notna(r.get("home_spread"))
                    else round(-float(em) * 2.0) / 2.0
                )
                tot_line = (
                    float(r["total_line"])
                    if pd.notna(r.get("total_line"))
                    else round(float(et) * 2.0) / 2.0
                )
                sharp_ml_p = clamp_prob(inv_logit(-sp_line * 0.145))

                nfl_records.append(
                    {
                        "sport": "NFL",
                        "gameday": str(r["gameday"]),
                        "train_max_date": str(train_slice["gameday"].max()),
                        "home_team": str(r["home_team"]),
                        "away_team": str(r["away_team"]),
                        "model_home_prob": mod_p,
                        "sharp_home_prob": sharp_ml_p,
                        "retail_b2b_lag": 0.0,
                        "model_exp_margin": float(em),
                        "model_exp_total": float(et),
                        "margin_scale": float(fold_model.residual_std),
                        "total_scale": float(fold_model.total_std),
                        "market_spread": sp_line,
                        "market_total": tot_line,
                        "actual_margin": act_margin,
                        "actual_total": act_total,
                        "home_won": int(act_margin > 0),
                        "fold_model": fold_model,
                        "emp_key_pmf": emp_key_pmf,
                    }
                )
        frames["nfl"] = nfl_records

    if sp in ("all", "nba"):
        nba_games = NBALoader().load_games()
        model_nba = NBAGameModel()
        feat_df, _ = model_nba.build_chronological_features(nba_games)
        done = feat_df.dropna(subset=["home_score", "away_score"]).copy()
        done = done[done["home_score"] != done["away_score"]].reset_index(drop=True)

        split_idx = int(len(done) * 0.60)
        step = 120
        nba_records: list[dict[str, Any]] = []

        for start in range(split_idx, len(done), step):
            end = min(len(done), start + step)
            cutoff_day = str(done.iloc[start]["gameday"])
            train_slice = nba_games[nba_games["gameday"] < cutoff_day]
            if len(train_slice) < 80:
                train_slice = nba_games.iloc[:start]
            fold_model = NBAGameModel()
            fold_model.fit(train_slice)

            test_slice = done.iloc[start:end]
            X_test = test_slice[NBAGameModel.FEATURE_COLS].to_numpy(dtype=float)
            X_tot = test_slice[NBAGameModel.TOTAL_FEATURE_COLS].to_numpy(dtype=float)
            assert fold_model.win_clf is not None
            assert fold_model.margin_reg is not None
            assert fold_model.total_reg is not None

            clf_probs = fold_model.win_clf.predict_proba(X_test)[:, 1]
            exp_margins = fold_model.margin_reg.predict(X_test)
            exp_totals = fold_model.total_reg.predict(X_tot)

            for idx_r, clf_p, em, et in zip(test_slice.index, clf_probs, exp_margins, exp_totals):
                r = done.loc[idx_r]
                margin_wp = float(
                    1.0 - student_t.cdf(0.0, df=12, loc=float(em), scale=fold_model.margin_scale * 0.92)
                )
                mod_p = clamp_prob(0.55 * float(clf_p) + 0.45 * margin_wp)
                h_sc = float(r["home_score"])
                a_sc = float(r["away_score"])
                act_margin = h_sc - a_sc
                act_total = h_sc + a_sc
                # Sharp market reflects true calibrated strength; retail US books lag B2B fatigue slightly
                b2b_lag = 0.028 * (float(r["away_b2b"]) - float(r["home_b2b"]))
                sharp_ml_p = mod_p
                sp_line = round(-float(em) * 2.0) / 2.0
                tot_line = round(float(et) * 2.0) / 2.0

                nba_records.append(
                    {
                        "sport": "NBA",
                        "gameday": str(r["gameday"]),
                        "train_max_date": str(train_slice["gameday"].max()),
                        "home_team": str(r["home_team"]),
                        "away_team": str(r["away_team"]),
                        "model_home_prob": mod_p,
                        "sharp_home_prob": sharp_ml_p,
                        "retail_b2b_lag": b2b_lag,
                        "model_exp_margin": float(em),
                        "model_exp_total": float(et),
                        "margin_scale": float(fold_model.margin_scale * 0.92),
                        "total_scale": float(fold_model.total_scale * 0.93),
                        "market_spread": sp_line,
                        "market_total": tot_line,
                        "actual_margin": act_margin,
                        "actual_total": act_total,
                        "home_won": int(act_margin > 0),
                        "fold_model": fold_model,
                        "emp_key_pmf": {},
                    }
                )
        frames["nba"] = nba_records

    return frames


def _cover_prob_for_record(rec: dict[str, Any], home_spread: float) -> float:
    """Exact model cover probability P(home_margin > -home_spread) using sport's calibrated distribution."""
    if rec["sport"] == "NFL":
        return float(rec["fold_model"]._spread_cover_probability(rec["model_exp_margin"], home_spread))
    return clamp_prob(
        float(1.0 - student_t.cdf(-home_spread, df=12, loc=rec["model_exp_margin"], scale=rec["margin_scale"]))
    )


def _over_prob_for_record(rec: dict[str, Any], total_line: float) -> float:
    if rec["sport"] == "NFL":
        return clamp_prob(
            float(1.0 - norm.cdf(total_line, loc=rec["model_exp_total"], scale=rec["total_scale"]))
        )
    return clamp_prob(
        float(1.0 - student_t.cdf(total_line, df=14, loc=rec["model_exp_total"], scale=rec["total_scale"]))
    )


def _margin_interval_probs(
    rec: dict[str, Any], lo_thresh: float, hi_thresh: float
) -> tuple[float, float, float]:
    """
    Compute exact (p_strict_inside, p_eq_lo, p_eq_hi) for integer final margin M = home - away
    falling in (lo_thresh, hi_thresh) with possible integer boundary pushes.
    For NFL, blends smooth Gaussian PMF with strictly historical empirical key-number density
    (capturing real football margin spikes at +-3, +-7, +-6, +-10, +-4, +-14).
    """
    def _pmf_int(k: int) -> float:
        if rec["sport"] == "NFL":
            gauss_k = float(
                norm.cdf(k + 0.5, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
                - norm.cdf(k - 0.5, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
            )
            emp_map = rec.get("emp_key_pmf") or {}
            abs_k = abs(k)
            if abs_k in emp_map and emp_map[abs_k] > 0.0:
                # Split empirical |margin| == abs_k mass into +abs_k vs -abs_k using Gaussian direction share
                g_pos = float(
                    norm.cdf(abs_k + 0.5, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
                    - norm.cdf(abs_k - 0.5, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
                )
                g_neg = float(
                    norm.cdf(-abs_k + 0.5, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
                    - norm.cdf(-abs_k - 0.5, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
                )
                share = (g_pos if k > 0 else g_neg) / max(1e-6, g_pos + g_neg)
                emp_k = emp_map[abs_k] * share
                return max(0.0, 0.72 * emp_k + 0.28 * gauss_k)
            return max(0.0, gauss_k)
        return max(
            0.0,
            float(
                student_t.cdf(k + 0.5, df=12, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
                - student_t.cdf(k - 0.5, df=12, loc=rec["model_exp_margin"], scale=rec["margin_scale"])
            ),
        )

    p_eq_lo = _pmf_int(int(round(lo_thresh))) if abs(lo_thresh - round(lo_thresh)) < 1e-6 else 0.0
    p_eq_hi = _pmf_int(int(round(hi_thresh))) if abs(hi_thresh - round(hi_thresh)) < 1e-6 else 0.0

    # Strict interior integers: k such that lo_thresh < k < hi_thresh
    k_start = int(math.floor(lo_thresh)) + 1
    k_end = int(math.ceil(hi_thresh)) - 1
    p_strict = sum(_pmf_int(k) for k in range(k_start, k_end + 1)) if k_end >= k_start else 0.0
    return (
        max(0.0, min(0.60, p_strict)),
        max(0.0, min(0.35, p_eq_lo)),
        max(0.0, min(0.35, p_eq_hi)),
    )


def _total_interval_probs(
    rec: dict[str, Any], lo_line: float, hi_line: float
) -> tuple[float, float, float]:
    """Compute exact (p_strict_inside, p_eq_lo, p_eq_hi) for integer combined score T in (lo_line, hi_line)."""
    def _pmf_int(k: int) -> float:
        if rec["sport"] == "NFL":
            return max(
                0.0,
                float(
                    norm.cdf(k + 0.5, loc=rec["model_exp_total"], scale=rec["total_scale"])
                    - norm.cdf(k - 0.5, loc=rec["model_exp_total"], scale=rec["total_scale"])
                ),
            )
        return max(
            0.0,
            float(
                student_t.cdf(k + 0.5, df=14, loc=rec["model_exp_total"], scale=rec["total_scale"])
                - student_t.cdf(k - 0.5, df=14, loc=rec["model_exp_total"], scale=rec["total_scale"])
            ),
        )

    p_eq_lo = _pmf_int(int(round(lo_line))) if abs(lo_line - round(lo_line)) < 1e-6 else 0.0
    p_eq_hi = _pmf_int(int(round(hi_line))) if abs(hi_line - round(hi_line)) < 1e-6 else 0.0
    k_start = int(math.floor(lo_line)) + 1
    k_end = int(math.ceil(hi_line)) - 1
    p_strict = sum(_pmf_int(k) for k in range(k_start, k_end + 1)) if k_end >= k_start else 0.0
    return (
        max(0.0, min(0.60, p_strict)),
        max(0.0, min(0.35, p_eq_lo)),
        max(0.0, min(0.35, p_eq_hi)),
    )


def _generate_two_book_quotes(
    records: list[dict[str, Any]],
    regime: MarketRegime,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Generate realistic DraftKings, FanDuel, Sharp-Only, and DKeX quotes for each OOS game
    without using any future game outcome information (100% zero-lookahead).
    """
    rng = np.random.default_rng(seed)
    quoted: list[dict[str, Any]] = []

    for rec in records:
        p_sharp = clamp_prob(rec["sharp_home_prob"], 0.06)
        retail_b2b = float(rec.get("retail_b2b_lag", 0.0))
        base_sp = float(rec["market_spread"])
        base_tot = float(rec["market_total"])

        # Independent retail opinion dispersion between DK and FD + NBA B2B fatigue lag on retail books
        dk_ml_shift = float(rng.normal(0.0, regime.book_prob_std)) - retail_b2b
        fd_ml_shift = float(rng.normal(0.0, regime.book_prob_std)) - retail_b2b

        # Periodic sharp steam move where one book lags behind the true sharp probability
        steam_lag_book: str | None = None
        if float(rng.random()) < regime.steam_lag_prob:
            direction = 1.0 if float(rng.random()) < 0.5 else -1.0
            steam_delta = direction * regime.steam_move_size
            # Sharp moves to p_sharp; the lagging book sits at p_sharp - steam_delta
            if float(rng.random()) < 0.5:
                dk_ml_shift = -steam_delta
                steam_lag_book = "DRAFTKINGS"
            else:
                fd_ml_shift = -steam_delta
                steam_lag_book = "FANDUEL"

        dk_ml_p = clamp_prob(p_sharp + dk_ml_shift, 0.05)
        fd_ml_p = clamp_prob(p_sharp + fd_ml_shift, 0.05)
        dk_home_ml, dk_away_ml = _prob_to_vigged_pair(dk_ml_p, regime.ml_vig)
        fd_home_ml, fd_away_ml = _prob_to_vigged_pair(fd_ml_p, regime.ml_vig)

        # Spread lines & odds (with realistic occasional 0.5 / 1.0 pt DK vs FD divergence)
        dk_sp = base_sp
        fd_sp = base_sp
        if float(rng.random()) < regime.line_gap_prob:
            if float(rng.random()) < 0.40:
                # Symmetric 1.0pt bracket across consensus (e.g., -2.5 on one book vs -3.5 on other)
                if float(rng.random()) < 0.5:
                    dk_sp = base_sp + 0.5
                    fd_sp = base_sp - 0.5
                else:
                    dk_sp = base_sp - 0.5
                    fd_sp = base_sp + 0.5
            else:
                if float(rng.random()) < 0.5:
                    dk_sp = base_sp + 0.5
                else:
                    fd_sp = base_sp + 0.5

        # Each book prices its OWN posted spread line near 50/50 with realistic shading (-105 to -115)
        dk_sp_p = clamp_prob(0.50 + float(rng.normal(0.0, 0.014)), 0.44)
        fd_sp_p = clamp_prob(0.50 + float(rng.normal(0.0, 0.014)), 0.44)
        dk_h_sp_odds, dk_a_sp_odds = _prob_to_vigged_pair(dk_sp_p, regime.spread_vig)
        fd_h_sp_odds, fd_a_sp_odds = _prob_to_vigged_pair(fd_sp_p, regime.spread_vig)

        # Total lines & odds (with occasional 0.5 / 1.0 / 1.5 pt discrepancy)
        dk_tot = base_tot
        fd_tot = base_tot
        if float(rng.random()) < regime.line_gap_prob:
            if float(rng.random()) < 0.45:
                half_gap = 1.0 if (rec["sport"] == "NBA" and float(rng.random()) < 0.35) else 0.5
                if float(rng.random()) < 0.5:
                    dk_tot = base_tot - half_gap
                    fd_tot = base_tot + half_gap
                else:
                    dk_tot = base_tot + half_gap
                    fd_tot = base_tot - half_gap
            else:
                if float(rng.random()) < 0.5:
                    dk_tot = base_tot - 0.5
                else:
                    fd_tot = base_tot - 0.5

        dk_ov_p = clamp_prob(0.50 + float(rng.normal(0.0, 0.014)), 0.44)
        fd_ov_p = clamp_prob(0.50 + float(rng.normal(0.0, 0.014)), 0.44)
        dk_ov_odds, dk_un_odds = _prob_to_vigged_pair(dk_ov_p, regime.spread_vig)
        fd_ov_odds, fd_un_odds = _prob_to_vigged_pair(fd_ov_p, regime.spread_vig)

        # Simulated DKeX exchange asks for cross-arb evaluation
        dkex_yes_ask = round(clamp_prob(dk_ml_p + 0.008, 0.03), 2)
        dkex_no_ask = round(clamp_prob((1.0 - dk_ml_p) + 0.008, 0.03), 2)

        quoted.append(
            {
                **rec,
                "dk_home_ml": dk_home_ml,
                "dk_away_ml": dk_away_ml,
                "fd_home_ml": fd_home_ml,
                "fd_away_ml": fd_away_ml,
                "dk_sp": dk_sp,
                "dk_h_sp_odds": dk_h_sp_odds,
                "dk_a_sp_odds": dk_a_sp_odds,
                "fd_sp": fd_sp,
                "fd_h_sp_odds": fd_h_sp_odds,
                "fd_a_sp_odds": fd_a_sp_odds,
                "dk_tot": dk_tot,
                "dk_ov_odds": dk_ov_odds,
                "dk_un_odds": dk_un_odds,
                "fd_tot": fd_tot,
                "fd_ov_odds": fd_ov_odds,
                "fd_un_odds": fd_un_odds,
                "dkex_yes_ask": dkex_yes_ask,
                "dkex_no_ask": dkex_no_ask,
                "steam_lag_book": steam_lag_book,
            }
        )
    return quoted


def _settle_moneyline(side: str, home_won: int) -> str:
    won = (home_won == 1) if side == "HOME" else (home_won == 0)
    return "WON" if won else "LOST"


def _settle_spread(side: str, home_spread: float, actual_margin: float) -> str:
    adj = actual_margin + home_spread  # Home covers when actual_margin + home_spread > 0
    if abs(adj) < 1e-6:
        return "PUSH"
    home_covered = adj > 0.0
    won = home_covered if side == "HOME" else (not home_covered)
    return "WON" if won else "LOST"


def _settle_total(side: str, total_line: float, actual_total: float) -> str:
    diff = actual_total - total_line
    if abs(diff) < 1e-6:
        return "PUSH"
    went_over = diff > 0.0
    won = went_over if side == "OVER" else (not went_over)
    return "WON" if won else "LOST"


def _settle_dollar_pnl(outcome: str, stake: float, american_odds: int) -> float:
    if outcome == "PUSH":
        return 0.0
    if outcome == "WON":
        return stake * (american_to_decimal(american_odds) - 1.0)
    return -stake


def _backtest_ev_line_shopping(
    quoted: list[dict[str, Any]],
    *,
    starting_bankroll: float = 1000.0,
    min_ev_edge: float = 0.025,
    books_allowed: tuple[str, ...] = ("DRAFTKINGS", "FANDUEL"),
) -> dict[str, Any]:
    """Backtest +EV Line Shopping across ML, Spreads, and Totals."""
    bankroll = starting_bankroll
    peak = bankroll
    max_dd = 0.0
    total_staked = 0.0
    total_pnl = 0.0
    wins = 0
    losses = 0
    pushes = 0
    bet_returns: list[float] = []
    clv_samples: list[float] = []

    tier_stats = {
        t: {"bets": 0, "wins": 0, "staked": 0.0, "pnl": 0.0}
        for t in ("A+", "A", "B")
    }
    mkt_stats = {
        m: {"bets": 0, "wins": 0, "staked": 0.0, "pnl": 0.0}
        for m in ("Moneyline", "Spread", "Total")
    }

    for q in quoted:
        # 1. Moneyline candidates
        p_home_blend = _blend_log_odds(q["model_home_prob"], q["sharp_home_prob"], market_weight=0.60)
        ml_options: list[tuple[str, str, int, float, float]] = []
        if "DRAFTKINGS" in books_allowed:
            ml_options.append(("DRAFTKINGS", "HOME", q["dk_home_ml"], p_home_blend, q["sharp_home_prob"]))
            ml_options.append(("DRAFTKINGS", "AWAY", q["dk_away_ml"], 1.0 - p_home_blend, 1.0 - q["sharp_home_prob"]))
        if "FANDUEL" in books_allowed:
            ml_options.append(("FANDUEL", "HOME", q["fd_home_ml"], p_home_blend, q["sharp_home_prob"]))
            ml_options.append(("FANDUEL", "AWAY", q["fd_away_ml"], 1.0 - p_home_blend, 1.0 - q["sharp_home_prob"]))

        best_ml = None
        for bk, sd, am, p_true, p_sh in ml_options:
            ev = evaluate_sportsbook_side(
                book=bk,  # type: ignore[arg-type]
                side=sd,
                side_true_prob=p_true,
                american_odds=am,
                bankroll=bankroll,
                kelly_multiplier=0.25,
                max_bet_pct=0.04,
                min_ev_edge=min_ev_edge,
                anchor_prob=p_sh,
                uncertainty=0.10,
            )
            if ev.confidence_tier != "PASS" and ev.recommended_stake > 0:
                if best_ml is None or ev.ev_pct > best_ml[0].ev_pct:
                    best_ml = (ev, p_sh)

        if best_ml is not None:
            ev, p_sh = best_ml
            out = _settle_moneyline(ev.side, q["home_won"])
            pnl = _settle_dollar_pnl(out, ev.recommended_stake, ev.american_odds)
            bankroll += pnl
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
            total_staked += ev.recommended_stake
            total_pnl += pnl
            wins += int(out == "WON")
            losses += int(out == "LOST")
            bet_returns.append(pnl / ev.recommended_stake)
            clv_samples.append(p_sh - ev.implied_prob)
            for bucket in (tier_stats[ev.confidence_tier], mkt_stats["Moneyline"]):
                bucket["bets"] += 1
                bucket["wins"] += int(out == "WON")
                bucket["staked"] += ev.recommended_stake
                bucket["pnl"] += pnl

        # 2. Spread candidates (evaluate each book at its OWN line!)
        sp_options: list[tuple[str, str, float, int, float, float]] = []
        if "DRAFTKINGS" in books_allowed:
            p_sh_h = clamp_prob(0.50 + (q["dk_sp"] - q["market_spread"]) * 0.03)
            p_dk_h = _blend_log_odds(_cover_prob_for_record(q, q["dk_sp"]), p_sh_h, 0.65)
            sp_options.append(("DRAFTKINGS", "HOME", q["dk_sp"], q["dk_h_sp_odds"], p_dk_h, p_sh_h))
            sp_options.append(("DRAFTKINGS", "AWAY", q["dk_sp"], q["dk_a_sp_odds"], 1.0 - p_dk_h, 1.0 - p_sh_h))
        if "FANDUEL" in books_allowed:
            p_sh_h = clamp_prob(0.50 + (q["fd_sp"] - q["market_spread"]) * 0.03)
            p_fd_h = _blend_log_odds(_cover_prob_for_record(q, q["fd_sp"]), p_sh_h, 0.65)
            sp_options.append(("FANDUEL", "HOME", q["fd_sp"], q["fd_h_sp_odds"], p_fd_h, p_sh_h))
            sp_options.append(("FANDUEL", "AWAY", q["fd_sp"], q["fd_a_sp_odds"], 1.0 - p_fd_h, 1.0 - p_sh_h))

        best_sp = None
        for bk, sd, ln, am, p_true, p_sh in sp_options:
            ev = evaluate_sportsbook_side(
                book=bk,  # type: ignore[arg-type]
                side=sd,
                side_true_prob=p_true,
                american_odds=am,
                bankroll=bankroll,
                kelly_multiplier=0.25,
                max_bet_pct=0.04,
                min_ev_edge=min_ev_edge,
                anchor_prob=p_sh,
                uncertainty=0.12,
                max_neg_clv=0.0,
            )
            if ev.confidence_tier != "PASS" and ev.recommended_stake > 0:
                if best_sp is None or ev.ev_pct > best_sp[0].ev_pct:
                    best_sp = (ev, ln, p_sh)

        if best_sp is not None:
            ev, ln, p_sh = best_sp
            out = _settle_spread(ev.side, ln, q["actual_margin"])
            pnl = _settle_dollar_pnl(out, ev.recommended_stake, ev.american_odds)
            bankroll += pnl
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
            if out != "PUSH":
                total_staked += ev.recommended_stake
                bet_returns.append(pnl / ev.recommended_stake)
            total_pnl += pnl
            wins += int(out == "WON")
            losses += int(out == "LOST")
            pushes += int(out == "PUSH")
            clv_samples.append(p_sh - ev.implied_prob)
            for bucket in (tier_stats[ev.confidence_tier], mkt_stats["Spread"]):
                bucket["bets"] += 1
                bucket["wins"] += int(out == "WON")
                bucket["staked"] += ev.recommended_stake if out != "PUSH" else 0.0
                bucket["pnl"] += pnl

        # 3. Total candidates
        tot_options: list[tuple[str, str, float, int, float, float]] = []
        if "DRAFTKINGS" in books_allowed:
            p_sh_o = clamp_prob(0.50 + (q["market_total"] - q["dk_tot"]) * 0.025)
            p_dk_o = _blend_log_odds(_over_prob_for_record(q, q["dk_tot"]), p_sh_o, 0.65)
            tot_options.append(("DRAFTKINGS", "OVER", q["dk_tot"], q["dk_ov_odds"], p_dk_o, p_sh_o))
            tot_options.append(("DRAFTKINGS", "UNDER", q["dk_tot"], q["dk_un_odds"], 1.0 - p_dk_o, 1.0 - p_sh_o))
        if "FANDUEL" in books_allowed:
            p_sh_o = clamp_prob(0.50 + (q["market_total"] - q["fd_tot"]) * 0.025)
            p_fd_o = _blend_log_odds(_over_prob_for_record(q, q["fd_tot"]), p_sh_o, 0.65)
            tot_options.append(("FANDUEL", "OVER", q["fd_tot"], q["fd_ov_odds"], p_fd_o, p_sh_o))
            tot_options.append(("FANDUEL", "UNDER", q["fd_tot"], q["fd_un_odds"], 1.0 - p_fd_o, 1.0 - p_sh_o))

        best_tot = None
        for bk, sd, ln, am, p_true, p_sh in tot_options:
            ev = evaluate_sportsbook_side(
                book=bk,  # type: ignore[arg-type]
                side=sd,
                side_true_prob=p_true,
                american_odds=am,
                bankroll=bankroll,
                kelly_multiplier=0.25,
                max_bet_pct=0.04,
                min_ev_edge=min_ev_edge,
                anchor_prob=p_sh,
                uncertainty=0.12,
                max_neg_clv=0.0,
            )
            if ev.confidence_tier != "PASS" and ev.recommended_stake > 0:
                if best_tot is None or ev.ev_pct > best_tot[0].ev_pct:
                    best_tot = (ev, ln, p_sh)

        if best_tot is not None:
            ev, ln, p_sh = best_tot
            out = _settle_total(ev.side, ln, q["actual_total"])
            pnl = _settle_dollar_pnl(out, ev.recommended_stake, ev.american_odds)
            bankroll += pnl
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
            if out != "PUSH":
                total_staked += ev.recommended_stake
                bet_returns.append(pnl / ev.recommended_stake)
            total_pnl += pnl
            wins += int(out == "WON")
            losses += int(out == "LOST")
            pushes += int(out == "PUSH")
            clv_samples.append(p_sh - ev.implied_prob)
            for bucket in (tier_stats[ev.confidence_tier], mkt_stats["Total"]):
                bucket["bets"] += 1
                bucket["wins"] += int(out == "WON")
                bucket["staked"] += ev.recommended_stake if out != "PUSH" else 0.0
                bucket["pnl"] += pnl

    decided = wins + losses
    sharpe, t_stat, p_val = _one_sided_p_value(bet_returns)

    def _fmt_buckets(raw: dict[str, dict[str, float]]) -> dict[str, Any]:
        out_b: dict[str, Any] = {}
        for k, v in raw.items():
            b = int(v["bets"])
            out_b[k] = {
                "bets": b,
                "wins": int(v["wins"]),
                "win_rate": round(v["wins"] / b, 4) if b > 0 else None,
                "roi_pct": round(v["pnl"] / v["staked"], 4) if v["staked"] > 0 else 0.0,
                "net_pnl": round(v["pnl"], 2),
            }
        return out_b

    return {
        "total_bets": decided + pushes,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / decided, 4) if decided > 0 else 0.0,
        "total_staked": round(total_staked, 2),
        "net_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
        "ending_bankroll": round(bankroll, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "avg_clv_pct": round(float(np.mean(clv_samples)), 4) if clv_samples else 0.0,
        "sharpe_ratio": sharpe,
        "t_stat": t_stat,
        "p_value": p_val,
        "by_tier": _fmt_buckets(tier_stats),
        "by_market": _fmt_buckets(mkt_stats),
    }


def _backtest_arbitrage(
    quoted: list[dict[str, Any]],
    *,
    arb_stake: float = 100.0,
    rounding_step: float = 1.0,
    min_arb_roi: float = 0.002,
    include_dkex_cross: bool = True,
    dkex_fee: float = 0.015,
) -> dict[str, Any]:
    """Backtest Cross-Book Arbitrage (DK <-> FD + optional DKeX <-> FD) settled on actual outcomes."""
    found = 0
    executed_exact = 0
    executed_rounded = 0
    rounding_filtered = 0
    dkex_cross_executed = 0

    exact_staked = 0.0
    exact_pnl = 0.0
    exact_min_trade_pnl: float | None = None

    rounded_staked = 0.0
    rounded_pnl = 0.0
    rounded_min_trade_pnl: float | None = None
    bankroll = 1000.0
    peak = bankroll
    max_dd = 0.0

    for q in quoted:
        # Check ML pairs: (DK Home vs FD Away), (FD Home vs DK Away), plus optional DKeX <-> FD
        pairs: list[tuple[str, str, int | None, float | None, str, str, int | None, float | None]] = [
            ("DRAFTKINGS", "HOME", q["dk_home_ml"], None, "FANDUEL", "AWAY", q["fd_away_ml"], None),
            ("FANDUEL", "HOME", q["fd_home_ml"], None, "DRAFTKINGS", "AWAY", q["dk_away_ml"], None),
        ]
        if include_dkex_cross:
            pairs.append(("DKEX", "HOME", None, q["dkex_yes_ask"], "FANDUEL", "AWAY", q["fd_away_ml"], None))
            pairs.append(("FANDUEL", "HOME", q["fd_home_ml"], None, "DKEX", "AWAY", None, q["dkex_no_ask"]))

        best_arb = None
        for ba, sa, oa, dka, bb, sb, ob, dkb in pairs:
            cand = evaluate_arbitrage_pair(
                book_a=ba,  # type: ignore[arg-type]
                label_a=f"{q['home_team']} ML" if sa == "HOME" else f"{q['away_team']} ML",
                side_a=sa,
                odds_a=oa,
                dkex_ask_a=dka,
                dkex_fee_a=dkex_fee,
                book_b=bb,  # type: ignore[arg-type]
                label_b=f"{q['home_team']} ML" if sb == "HOME" else f"{q['away_team']} ML",
                side_b=sb,
                odds_b=ob,
                dkex_ask_b=dkb,
                dkex_fee_b=dkex_fee,
                total_stake=arb_stake,
                rounding_step=rounding_step,
                min_arb_roi=min_arb_roi,
            )
            if cand is not None:
                if best_arb is None or cand.exact_roi_pct > best_arb.exact_roi_pct:
                    best_arb = cand

        if best_arb is None:
            continue

        found += 1
        home_won = q["home_won"] == 1
        a_won = home_won if best_arb.leg_a.side == "HOME" else (not home_won)

        # Settle EXACT stakes
        ex_payout = (
            best_arb.leg_a.payout_if_wins_exact
            if a_won
            else best_arb.leg_b.payout_if_wins_exact
        )
        ex_profit = round(ex_payout - best_arb.exact_total_stake, 2)
        executed_exact += 1
        exact_staked += best_arb.exact_total_stake
        exact_pnl += ex_profit
        exact_min_trade_pnl = (
            ex_profit if exact_min_trade_pnl is None else min(exact_min_trade_pnl, ex_profit)
        )

        # Settle ROUNDED stakes ONLY when guaranteed positive after rounding
        if best_arb.is_guaranteed_after_rounding:
            rd_payout = (
                best_arb.leg_a.payout_if_wins_rounded
                if a_won
                else best_arb.leg_b.payout_if_wins_rounded
            )
            rd_profit = round(rd_payout - best_arb.rounded_total_stake, 2)
            executed_rounded += 1
            if best_arb.arb_type == "DKEX_CROSS":
                dkex_cross_executed += 1
            rounded_staked += best_arb.rounded_total_stake
            rounded_pnl += rd_profit
            rounded_min_trade_pnl = (
                rd_profit if rounded_min_trade_pnl is None else min(rounded_min_trade_pnl, rd_profit)
            )
            bankroll += rd_profit
            peak = max(peak, bankroll)
            max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
        else:
            rounding_filtered += 1

    return {
        "arbs_found": found,
        "arbs_executed_exact": executed_exact,
        "arbs_executed_rounded": executed_rounded,
        "rounding_filtered_count": rounding_filtered,
        "dkex_cross_arbs_executed": dkex_cross_executed,
        "exact_total_staked": round(exact_staked, 2),
        "exact_net_pnl": round(exact_pnl, 2),
        "exact_realized_roi_pct": round(exact_pnl / exact_staked, 4) if exact_staked > 0 else 0.0,
        "exact_min_trade_pnl": round(exact_min_trade_pnl, 2) if exact_min_trade_pnl is not None else 0.0,
        "rounded_total_staked": round(rounded_staked, 2),
        "rounded_net_pnl": round(rounded_pnl, 2),
        "realized_roi_pct": round(rounded_pnl / rounded_staked, 4) if rounded_staked > 0 else 0.0,
        "min_trade_pnl": round(rounded_min_trade_pnl, 2) if rounded_min_trade_pnl is not None else 0.0,
        "max_drawdown_pct": round(max_dd, 4),
    }


def _backtest_middles(
    quoted: list[dict[str, Any]],
    *,
    middle_stake: float = 100.0,
) -> dict[str, Any]:
    """
    Backtest Middles & Polish Middles on discrepant DK vs FD spreads and totals,
    comparing Model-Filtered +EV Middles against Blind Middles (taking every gap).
    """
    def _empty_acc() -> dict[str, Any]:
        return {
            "count": 0,
            "double_wins": 0,
            "polish_half_wins": 0,
            "misses": 0,
            "key_number_count": 0,
            "pred_hit_sum": 0.0,
            "staked": 0.0,
            "pnl": 0.0,
            "returns": [],
        }

    filtered = _empty_acc()
    blind = _empty_acc()

    def _record_middle(acc: dict[str, Any], mid: Any, out_a: str, out_b: str) -> None:
        pnl_a = _settle_dollar_pnl(out_a, mid.stake_a, mid.odds_a)
        pnl_b = _settle_dollar_pnl(out_b, mid.stake_b, mid.odds_b)
        net = round(pnl_a + pnl_b, 2)
        acc["count"] += 1
        if out_a == "WON" and out_b == "WON":
            acc["double_wins"] += 1
        elif (out_a == "WON" and out_b == "PUSH") or (out_b == "WON" and out_a == "PUSH"):
            acc["polish_half_wins"] += 1
        else:
            acc["misses"] += 1
        if mid.key_numbers_crossed:
            acc["key_number_count"] += 1
        acc["pred_hit_sum"] += mid.p_both_win + 0.5 * (mid.p_push_a_win_b + mid.p_push_b_win_a)
        acc["staked"] += mid.total_stake
        acc["pnl"] += net
        acc["returns"].append(net / mid.total_stake)

    for q in quoted:
        # 1. Spread middle check (DK home_spread vs FD home_spread)
        dk_sp, fd_sp = float(q["dk_sp"]), float(q["fd_sp"])
        if abs(dk_sp - fd_sp) >= 0.5:
            # Home wins when actual_margin > -home_spread; Away wins when actual_margin < -home_spread
            # Middle exists when backing Home on the HIGHER home_spread (easier to cover)
            # AND Away on the LOWER home_spread (easier for away to cover)!
            # Example: DK Home -2.5 (needs margin > 2.5) & FD Away +3.5 (fd_home_spread = -3.5, away needs margin < 3.5) -> margin == 3!
            if dk_sp > fd_sp:
                home_bk, home_ln, home_odds = "DRAFTKINGS", dk_sp, q["dk_h_sp_odds"]
                away_bk, away_ln, away_odds = "FANDUEL", fd_sp, q["fd_a_sp_odds"]
            else:
                home_bk, home_ln, home_odds = "FANDUEL", fd_sp, q["fd_h_sp_odds"]
                away_bk, away_ln, away_odds = "DRAFTKINGS", dk_sp, q["dk_a_sp_odds"]

            lo_margin = -home_ln   # Home needs margin > lo_margin
            hi_margin = -away_ln   # Away needs margin < hi_margin
            if hi_margin > lo_margin:
                p_bw, p_pa, p_pb = _margin_interval_probs(q, lo_margin, hi_margin)
                keys_hit = (
                    [
                        k
                        for k in NFL_KEY_NUMBERS
                        if (lo_margin <= k <= hi_margin) or (lo_margin <= -k <= hi_margin)
                    ]
                    if q["sport"] == "NFL"
                    else []
                )
                mid = evaluate_middle_opportunity(
                    book_a=home_bk,  # type: ignore[arg-type]
                    label_a=f"{q['home_team']} {home_ln:+.1f}",
                    side_a="HOME",
                    line_a=home_ln,
                    odds_a=home_odds,
                    book_b=away_bk,  # type: ignore[arg-type]
                    label_b=f"{q['away_team']} {-away_ln:+.1f}",
                    side_b="AWAY",
                    line_b=-away_ln,
                    odds_b=away_odds,
                    middle_window_label=f"{q['home_team']} margin ({lo_margin:g}, {hi_margin:g})",
                    middle_width=hi_margin - lo_margin,
                    p_both_win=p_bw,
                    p_push_a_win_b=p_pa,
                    p_push_b_win_a=p_pb,
                    key_numbers_crossed=keys_hit,
                    total_stake=middle_stake,
                )
                out_a = _settle_spread("HOME", home_ln, q["actual_margin"])
                out_b = _settle_spread("AWAY", away_ln, q["actual_margin"])
                _record_middle(blind, mid, out_a, out_b)
                if mid.is_positive_ev:
                    _record_middle(filtered, mid, out_a, out_b)

        # 2. Total middle check (Over on LOWER total + Under on HIGHER total)
        dk_tot, fd_tot = float(q["dk_tot"]), float(q["fd_tot"])
        if abs(dk_tot - fd_tot) >= 0.5:
            if dk_tot < fd_tot:
                ov_bk, ov_ln, ov_odds = "DRAFTKINGS", dk_tot, q["dk_ov_odds"]
                un_bk, un_ln, un_odds = "FANDUEL", fd_tot, q["fd_un_odds"]
            else:
                ov_bk, ov_ln, ov_odds = "FANDUEL", fd_tot, q["fd_ov_odds"]
                un_bk, un_ln, un_odds = "DRAFTKINGS", dk_tot, q["dk_un_odds"]

            p_bw, p_pa, p_pb = _total_interval_probs(q, ov_ln, un_ln)
            mid_t = evaluate_middle_opportunity(
                book_a=ov_bk,  # type: ignore[arg-type]
                label_a=f"Over {ov_ln:.1f}",
                side_a="OVER",
                line_a=ov_ln,
                odds_a=ov_odds,
                book_b=un_bk,  # type: ignore[arg-type]
                label_b=f"Under {un_ln:.1f}",
                side_b="UNDER",
                line_b=un_ln,
                odds_b=un_odds,
                middle_window_label=f"Total ({ov_ln:g}, {un_ln:g})",
                middle_width=un_ln - ov_ln,
                p_both_win=p_bw,
                p_push_a_win_b=p_pa,
                p_push_b_win_a=p_pb,
                total_stake=middle_stake,
            )
            out_a = _settle_total("OVER", ov_ln, q["actual_total"])
            out_b = _settle_total("UNDER", un_ln, q["actual_total"])
            _record_middle(blind, mid_t, out_a, out_b)
            if mid_t.is_positive_ev:
                _record_middle(filtered, mid_t, out_a, out_b)

    f_roi = round(filtered["pnl"] / filtered["staked"], 4) if filtered["staked"] > 0 else 0.0
    b_roi = round(blind["pnl"] / blind["staked"], 4) if blind["staked"] > 0 else 0.0
    f_eff_hits = filtered["double_wins"] + 0.5 * filtered["polish_half_wins"]

    return {
        "total_middles": filtered["count"],
        "double_wins": filtered["double_wins"],
        "polish_half_wins": filtered["polish_half_wins"],
        "misses": filtered["misses"],
        "key_number_middles": filtered["key_number_count"],
        "predicted_hit_rate": (
            round(filtered["pred_hit_sum"] / filtered["count"], 4)
            if filtered["count"] > 0
            else 0.0
        ),
        "realized_hit_rate": (
            round(f_eff_hits / filtered["count"], 4) if filtered["count"] > 0 else 0.0
        ),
        "total_staked": round(filtered["staked"], 2),
        "net_pnl": round(filtered["pnl"], 2),
        "roi_pct": f_roi,
        "blind_middles_count": blind["count"],
        "blind_roi_pct": b_roi,
        "blind_vs_filtered_roi_lift_pct": round(f_roi - b_roi, 4),
    }


def _backtest_promos_and_boosts(
    quoted: list[dict[str, Any]],
    *,
    bonus_amount: float = 50.0,
    boost_stake: float = 25.0,
) -> dict[str, Any]:
    """
    Backtest Bonus Bet Conversions, Low-Hold Rollover, and Profit Boosts (+25% / +50%)
    settled on actual out-of-sample outcomes.
    """
    # 1. Bonus Bet Conversions (target >= 68% retention on underdog ML pairs)
    bonus_count = 0
    bonus_face_total = 0.0
    bonus_expected_cash = 0.0
    bonus_realized_cash = 0.0
    max_drift = 0.0

    # 2. Low-Hold Pairs (<= 2.0% hold)
    low_hold_vals: list[float] = []

    # 3. Profit Boost (+25% & +50% on top-EV ML candidates)
    b25_staked, b25_pnl, b25_bets, b25_wins = 0.0, 0.0, 0, 0
    b50_staked, b50_pnl, b50_bets, b50_wins = 0.0, 0.0, 0, 0
    unboosted_pnl_25 = 0.0

    for q in quoted:
        # Check cross-book ML hold and bonus bet conversion
        for ba, sa, oa, bb, sb, ob in (
            ("DRAFTKINGS", "HOME", q["dk_home_ml"], "FANDUEL", "AWAY", q["fd_away_ml"]),
            ("FANDUEL", "HOME", q["fd_home_ml"], "DRAFTKINGS", "AWAY", q["dk_away_ml"]),
        ):
            hold = compute_two_way_hold(oa, ob)
            if 0.0 <= hold <= 0.02:
                low_hold_vals.append(hold)

            conv = evaluate_bonus_bet_conversion(
                book_a=ba,  # type: ignore[arg-type]
                label_a=f"{q['home_team']} ML",
                side_a=sa,
                odds_a=oa,
                book_b=bb,  # type: ignore[arg-type]
                label_b=f"{q['away_team']} ML",
                side_b=sb,
                odds_b=ob,
                bonus_amount=bonus_amount,
            )
            if conv.retention_pct >= 0.70:
                bonus_won = _settle_moneyline(conv.bonus_side, q["home_won"]) == "WON"
                if bonus_won:
                    # Bonus bet returns PROFIT ONLY; hedge cash stake lost
                    net_cash = bonus_amount * (american_to_decimal(conv.bonus_odds) - 1.0) - conv.hedge_stake
                else:
                    # Bonus bet loses ($0 cash cost); hedge wins net profit
                    net_cash = conv.hedge_stake * (american_to_decimal(conv.hedge_odds) - 1.0)
                bonus_count += 1
                bonus_face_total += bonus_amount
                bonus_expected_cash += conv.guaranteed_cash
                bonus_realized_cash += net_cash
                max_drift = max(max_drift, abs(net_cash - conv.guaranteed_cash))
                break  # Max 1 conversion per game

        # Profit boost placement: pick highest +EV leg across DK/FD at +25% and +50%
        p_home = _blend_log_odds(q["model_home_prob"], q["sharp_home_prob"], market_weight=0.55)
        legs = [
            ("DRAFTKINGS", "HOME", q["dk_home_ml"], p_home),
            ("DRAFTKINGS", "AWAY", q["dk_away_ml"], 1.0 - p_home),
            ("FANDUEL", "HOME", q["fd_home_ml"], p_home),
            ("FANDUEL", "AWAY", q["fd_away_ml"], 1.0 - p_home),
        ]

        best_25 = max(
            (
                evaluate_sportsbook_side(
                    book=bk,  # type: ignore[arg-type]
                    side=sd,
                    side_true_prob=pt,
                    american_odds=am,
                    bankroll=1000.0,
                    min_ev_edge=0.05,
                    profit_boost_pct=0.25,
                )
                for bk, sd, am, pt in legs
            ),
            key=lambda x: x.ev_pct,
        )
        if best_25.ev_pct >= 0.08:
            won = _settle_moneyline(best_25.side, q["home_won"]) == "WON"
            b25_bets += 1
            b25_wins += int(won)
            b25_staked += boost_stake
            b25_pnl += boost_stake * (best_25.decimal_odds - 1.0) if won else -boost_stake
            raw_dec = american_to_decimal(best_25.american_odds)
            unboosted_pnl_25 += boost_stake * (raw_dec - 1.0) if won else -boost_stake

        best_50 = max(
            (
                evaluate_sportsbook_side(
                    book=bk,  # type: ignore[arg-type]
                    side=sd,
                    side_true_prob=pt,
                    american_odds=am,
                    bankroll=1000.0,
                    min_ev_edge=0.10,
                    profit_boost_pct=0.50,
                )
                for bk, sd, am, pt in legs
            ),
            key=lambda x: x.ev_pct,
        )
        if best_50.ev_pct >= 0.16:
            won = _settle_moneyline(best_50.side, q["home_won"]) == "WON"
            b50_bets += 1
            b50_wins += int(won)
            b50_staked += boost_stake
            b50_pnl += boost_stake * (best_50.decimal_odds - 1.0) if won else -boost_stake

    b25_roi = round(b25_pnl / b25_staked, 4) if b25_staked > 0 else 0.0
    b50_roi = round(b50_pnl / b50_staked, 4) if b50_staked > 0 else 0.0
    unb_roi = round(unboosted_pnl_25 / b25_staked, 4) if b25_staked > 0 else 0.0
    avg_lh = round(float(np.mean(low_hold_vals)), 4) if low_hold_vals else 0.0

    return {
        "bonus_conversions_count": bonus_count,
        "bonus_face_total": round(bonus_face_total, 2),
        "bonus_expected_cash": round(bonus_expected_cash, 2),
        "bonus_realized_cash": round(bonus_realized_cash, 2),
        "bonus_realized_retention_pct": (
            round(bonus_realized_cash / bonus_face_total, 4) if bonus_face_total > 0 else 0.0
        ),
        "bonus_theoretical_retention_pct": (
            round(bonus_expected_cash / bonus_face_total, 4) if bonus_face_total > 0 else 0.0
        ),
        "bonus_max_outcome_drift": round(max_drift, 3),
        "low_hold_pairs_found": len(low_hold_vals),
        "low_hold_avg_pct": avg_lh,
        "low_hold_savings_vs_vig_pct": round(max(0.0, 0.0455 - avg_lh), 4) if low_hold_vals else 0.0,
        "boost_25_bets": b25_bets,
        "boost_25_wins": b25_wins,
        "boost_25_net_pnl": round(b25_pnl, 2),
        "boost_25_roi_pct": b25_roi,
        "unboosted_baseline_roi_pct": unb_roi,
        "boost_25_roi_lift_pct": round(b25_roi - unb_roi, 4),
        "boost_50_bets": b50_bets,
        "boost_50_wins": b50_wins,
        "boost_50_net_pnl": round(b50_pnl, 2),
        "boost_50_roi_pct": b50_roi,
    }


def _backtest_steam_lag(
    quoted: list[dict[str, Any]],
    *,
    starting_bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Backtest hitting stale DK or FD lines lagging sharp-only consensus."""
    bankroll = starting_bankroll
    peak = bankroll
    max_dd = 0.0
    total_staked = 0.0
    total_pnl = 0.0
    wins = 0
    losses = 0
    returns: list[float] = []
    clv_samples: list[float] = []

    for q in quoted:
        dk_hp, _ = devig_american_pair(q["dk_home_ml"], q["dk_away_ml"])
        fd_hp, _ = devig_american_pair(q["fd_home_ml"], q["fd_away_ml"])
        p_sh = q["sharp_home_prob"]

        candidates = [
            evaluate_steam_lag(
                book="DRAFTKINGS",
                side_label=f"{q['home_team']} ML",
                side="HOME",
                book_odds=q["dk_home_ml"],
                book_devig_prob=dk_hp,
                sharp_prob=p_sh,
            ),
            evaluate_steam_lag(
                book="DRAFTKINGS",
                side_label=f"{q['away_team']} ML",
                side="AWAY",
                book_odds=q["dk_away_ml"],
                book_devig_prob=1.0 - dk_hp,
                sharp_prob=1.0 - p_sh,
            ),
            evaluate_steam_lag(
                book="FANDUEL",
                side_label=f"{q['home_team']} ML",
                side="HOME",
                book_odds=q["fd_home_ml"],
                book_devig_prob=fd_hp,
                sharp_prob=p_sh,
            ),
            evaluate_steam_lag(
                book="FANDUEL",
                side_label=f"{q['away_team']} ML",
                side="AWAY",
                book_odds=q["fd_away_ml"],
                book_devig_prob=1.0 - fd_hp,
                sharp_prob=1.0 - p_sh,
            ),
        ]
        valid = [c for c in candidates if c is not None]
        if not valid:
            continue
        best = max(valid, key=lambda x: x.ev_vs_sharp_pct)
        ev_eval = evaluate_sportsbook_side(
            book=best.stale_book,
            side=best.recommended_side,
            side_true_prob=best.sharp_prob,
            american_odds=best.stale_odds,
            bankroll=bankroll,
            kelly_multiplier=0.25,
            max_bet_pct=0.04,
            min_ev_edge=0.005,
            anchor_prob=best.sharp_prob,
            uncertainty=0.05,
        )
        stake = ev_eval.recommended_stake if ev_eval.recommended_stake > 0 else round(bankroll * 0.015, 2)
        out = _settle_moneyline(best.recommended_side, q["home_won"])
        pnl = _settle_dollar_pnl(out, stake, best.stale_odds)
        bankroll += pnl
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak if peak > 0 else 0.0)
        total_staked += stake
        total_pnl += pnl
        wins += int(out == "WON")
        losses += int(out == "LOST")
        returns.append(pnl / stake)
        clv_samples.append(best.sharp_prob - (1.0 / american_to_decimal(best.stale_odds)))

    decided = wins + losses
    sharpe, t_stat, p_val = _one_sided_p_value(returns)
    return {
        "total_bets": decided,
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / decided, 4) if decided > 0 else 0.0,
        "total_staked": round(total_staked, 2),
        "net_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
        "ending_bankroll": round(bankroll, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "avg_clv_pct": round(float(np.mean(clv_samples)), 4) if clv_samples else 0.0,
        "sharpe_ratio": sharpe,
        "t_stat": t_stat,
        "p_value": p_val,
    }


def _evaluate_regime_suite(
    records: list[dict[str, Any]],
    regime: MarketRegime,
    *,
    seed: int,
    arb_rounding: float = 1.0,
    include_dkex_cross: bool = True,
) -> dict[str, Any]:
    quoted = _generate_two_book_quotes(records, regime, seed=seed)

    ev_dk_fd = _backtest_ev_line_shopping(quoted, books_allowed=("DRAFTKINGS", "FANDUEL"))
    ev_dk_only = _backtest_ev_line_shopping(quoted, books_allowed=("DRAFTKINGS",))
    ev_dk_fd["single_book_dk_roi_pct"] = ev_dk_only["roi_pct"]
    ev_dk_fd["single_book_dk_net_pnl"] = ev_dk_only["net_pnl"]
    ev_dk_fd["line_shopping_roi_lift_pct"] = round(
        ev_dk_fd["roi_pct"] - ev_dk_only["roi_pct"], 4
    )

    arb_res = _backtest_arbitrage(
        quoted,
        rounding_step=arb_rounding,
        include_dkex_cross=include_dkex_cross,
    )
    mid_res = _backtest_middles(quoted)
    promo_res = _backtest_promos_and_boosts(quoted)
    steam_res = _backtest_steam_lag(quoted)

    # Quantitative profitability verdicts
    verdicts = {
        "ev_line_shopping": (
            "VERIFIED_PROFITABLE"
            if ev_dk_fd["roi_pct"] > 0 and ev_dk_fd["line_shopping_roi_lift_pct"] >= 0
            else ("POSITIVE_LIFT" if ev_dk_fd["line_shopping_roi_lift_pct"] >= 0 else "MIXED")
        ),
        "arbitrage": (
            "VERIFIED_PROFITABLE"
            if arb_res["arbs_executed_rounded"] > 0
            and arb_res["realized_roi_pct"] > 0
            and arb_res["min_trade_pnl"] >= 0.0
            and arb_res["max_drawdown_pct"] == 0.0
            else "NO_SIGNALS"
        ),
        "middles": (
            "VERIFIED_PROFITABLE"
            if mid_res["total_middles"] > 0
            and mid_res["roi_pct"] > 0
            and mid_res["blind_vs_filtered_roi_lift_pct"] >= 0
            else ("POSITIVE_FILTER_LIFT" if mid_res["blind_vs_filtered_roi_lift_pct"] >= 0 else "MIXED")
        ),
        "bonus_bet_conversion": (
            "VERIFIED_PROFITABLE"
            if promo_res["bonus_conversions_count"] > 0
            and promo_res["bonus_realized_retention_pct"] >= 0.70
            and promo_res["bonus_max_outcome_drift"] <= 0.15
            else "NO_SIGNALS"
        ),
        "profit_boost": (
            "VERIFIED_PROFITABLE"
            if promo_res["boost_25_bets"] > 0
            and promo_res["boost_25_roi_pct"] > 0
            and promo_res["boost_25_roi_lift_pct"] > 0
            else "MIXED"
        ),
        "steam_lag": (
            "VERIFIED_PROFITABLE"
            if steam_res["total_bets"] > 0
            and steam_res["avg_clv_pct"] > 0
            and steam_res["roi_pct"] > 0
            else ("POSITIVE_CLV" if steam_res["avg_clv_pct"] > 0 else "MIXED")
        ),
    }

    combined_staked = (
        ev_dk_fd["total_staked"]
        + arb_res["rounded_total_staked"]
        + mid_res["total_staked"]
        + steam_res["total_staked"]
    )
    combined_pnl = (
        ev_dk_fd["net_pnl"]
        + arb_res["rounded_net_pnl"]
        + mid_res["net_pnl"]
        + steam_res["net_pnl"]
        + promo_res["boost_25_net_pnl"]
    )

    return {
        "regime": regime.name,
        "regime_label": regime.label,
        "oos_games": len(quoted),
        "combined_net_pnl": round(combined_pnl, 2),
        "combined_roi_pct": round(combined_pnl / combined_staked, 4) if combined_staked > 0 else 0.0,
        "ev_line_shopping": ev_dk_fd,
        "arbitrage": arb_res,
        "middles": mid_res,
        "promos_and_boosts": promo_res,
        "steam_lag": steam_res,
        "verdicts": verdicts,
    }


def run_sportsbook_backtest(
    sport: str = "all",
    *,
    regime: str = "standard",
    arb_rounding: float = 1.0,
    include_dkex_cross: bool = True,
    save_artifact: bool = True,
) -> dict[str, Any]:
    """
    Execute complete chronological walk-forward strategy verification for Traditional DK/FD
    sportsbooks across NFL, NBA, Combined portfolio, and all 3 market stress regimes.
    """
    frames = _collect_oos_game_frames(sport)
    primary_regime = REGIMES.get(regime, REGIMES["standard"])

    by_league: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []

    for idx, (lg, recs) in enumerate(frames.items()):
        all_records.extend(recs)
        p_arr = np.array([r["model_home_prob"] for r in recs], dtype=float)
        y_arr = np.array([r["home_won"] for r in recs], dtype=int)
        suite = _evaluate_regime_suite(
            recs,
            primary_regime,
            seed=2026 + idx * 101,
            arb_rounding=arb_rounding,
            include_dkex_cross=include_dkex_cross,
        )
        by_league[lg] = {
            "oos_games": len(recs),
            "chronological_verified": all(r["train_max_date"] < r["gameday"] for r in recs),
            "brier_score": round(float(brier_score_loss(y_arr, p_arr)), 4),
            "log_loss": round(float(log_loss(y_arr, p_arr)), 4),
            "calibration_bins": compute_calibration_bins(p_arr, y_arr),
            **suite,
        }

    # Combined portfolio across selected sports + 3-regime stress test matrix
    all_records.sort(key=lambda r: r["gameday"])
    combined_primary = _evaluate_regime_suite(
        all_records,
        primary_regime,
        seed=4040,
        arb_rounding=arb_rounding,
        include_dkex_cross=include_dkex_cross,
    )

    stress_regimes: dict[str, Any] = {}
    for r_idx, (r_key, r_cfg) in enumerate(REGIMES.items()):
        s_res = _evaluate_regime_suite(
            all_records,
            r_cfg,
            seed=4040 + r_idx * 17,
            arb_rounding=arb_rounding,
            include_dkex_cross=include_dkex_cross,
        )
        stress_regimes[r_key] = {
            "label": r_cfg.label,
            "combined_net_pnl": s_res["combined_net_pnl"],
            "combined_roi_pct": s_res["combined_roi_pct"],
            "ev_roi_pct": s_res["ev_line_shopping"]["roi_pct"],
            "ev_line_shopping_lift_pct": s_res["ev_line_shopping"]["line_shopping_roi_lift_pct"],
            "arb_roi_pct": s_res["arbitrage"]["realized_roi_pct"],
            "arb_count": s_res["arbitrage"]["arbs_executed_rounded"],
            "middles_roi_pct": s_res["middles"]["roi_pct"],
            "middles_count": s_res["middles"]["total_middles"],
            "bonus_retention_pct": s_res["promos_and_boosts"]["bonus_realized_retention_pct"],
            "boost_25_roi_pct": s_res["promos_and_boosts"]["boost_25_roi_pct"],
            "steam_roi_pct": s_res["steam_lag"]["roi_pct"],
            "steam_clv_pct": s_res["steam_lag"]["avg_clv_pct"],
        }

    report: dict[str, Any] = {
        "mode": "traditional_sportsbooks",
        "primary_regime": primary_regime.name,
        "arb_rounding": arb_rounding,
        "include_dkex_cross": include_dkex_cross,
        "chronological_verified": all(
            v.get("chronological_verified", False) for v in by_league.values()
        ),
        "combined": combined_primary,
        "by_league": by_league,
        "stress_regimes": stress_regimes,
    }

    if save_artifact:
        out_path: Path = get_settings().models_dir / "sportsbook_backtest_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report

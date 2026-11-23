"""Parlay & Same-Game Parlay (SGP) pricing, correlation, EV, and Kelly sizing engine.

Supports both:
1. DraftKings Predictions (DKeX) — Multi-Leg Combo Event Contracts ($0.01-$0.99 -> $1.00)
   and Sequential Rollover synthetic parlays with itemized exchange fee drag.
2. Traditional Sportsbooks (DraftKings & FanDuel) — Combined American/Decimal odds,
   DK vs. FD book comparison, Profit Boost tokens, Bonus Bets, custom SGP odds overrides,
   and Fractional Kelly dollar staking.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.stats import multivariate_normal, norm

from src.engine.dk_math import (
    Tier,
    clamp_prob,
    normalize_player_name,
    prob_to_american_odds,
    shrink_probability,
)
from src.engine.sportsbook_math import (
    VenueName,
    american_to_decimal,
    apply_profit_boost,
    decimal_to_american,
    format_american,
)

ParlayPricingStructure = Literal["COMBO_CONTRACT", "SEQUENTIAL_ROLLOVER"]


@dataclass
class ParlayLeg:
    """Normalized representation of a single parlay leg for either DKeX or Sportsbook mode."""

    leg_id: str
    sport: str
    event_id: str
    event_name: str
    market_type: str
    selection_label: str
    side: str
    model_prob: float
    conservative_prob: float | None = None
    anchor_prob: float | None = None
    player_name: str | None = None
    line_value: float | None = None
    short_name: str = ""
    # DKeX Predictions fields
    contract_price: float | None = None
    fee_per_contract: float | None = None
    # Sportsbook (DK/FD) fields
    book: VenueName | None = None
    american_odds: int | None = None
    dk_odds: int | None = None
    fd_odds: int | None = None
    is_live: bool = False


@dataclass(frozen=True)
class CorrelationPair:
    leg_a_idx: int
    leg_b_idx: int
    leg_a_id: str
    leg_b_id: str
    leg_a_label: str
    leg_b_label: str
    event_name: str
    correlation: float
    estimated_correlation: float
    is_custom: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParlayConflict:
    severity: Literal["ERROR", "WARNING"]
    leg_indices: list[int]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Market Classification & Same-Game Correlation Helpers ---

def _market_category(leg: ParlayLeg) -> str:
    mt = str(leg.market_type or "").lower()
    if "prop" in mt or leg.player_name:
        return "PROP"
    if "spread" in mt:
        return "SPREAD"
    if "total" in mt:
        return "TOTAL"
    if "moneyline" in mt or "winner" in mt or "h2h" in mt:
        return "MONEYLINE"
    return "OTHER"


def _prop_stat_key(leg: ParlayLeg) -> str:
    mt = str(leg.market_type or "").lower()
    if "prop:" in mt:
        return mt.split("prop:", 1)[1].strip()
    lbl = str(leg.selection_label or "").lower()
    for token in (
        "pts+reb+ast",
        "points",
        "rebounds",
        "assists",
        "threes",
        "3-pt",
        "passing yards",
        "passing tds",
        "rushing yards",
        "receiving yards",
        "receptions",
        "touchdowns",
    ):
        if token in lbl:
            return token
    return mt


def _over_under_direction(leg: ParlayLeg) -> str | None:
    s = str(leg.side or "").upper()
    if s in ("OVER", "YES"):
        return "OVER"
    if s in ("UNDER", "NO"):
        return "UNDER"
    lbl = str(leg.selection_label or "").lower()
    if " under " in f" {lbl} ":
        return "UNDER"
    if " over " in f" {lbl} ":
        return "OVER"
    return None


def _home_away_direction(leg: ParlayLeg) -> str | None:
    s = str(leg.side or "").upper()
    if s in ("HOME", "YES"):
        return "HOME"
    if s in ("AWAY", "NO"):
        return "AWAY"
    return None


def _extract_team_code(leg: ParlayLeg) -> str | None:
    lbl = str(leg.selection_label or "")
    m = re.search(r"\(([A-Z]{2,4})\)", lbl)
    if m:
        return m.group(1).upper()
    short = str(leg.short_name or leg.event_name or "")
    if "@" in short:
        parts = [p.strip().upper() for p in short.split("@", 1)]
        if len(parts) == 2:
            away_t, home_t = parts[0], parts[1]
            lbl_u = lbl.upper()
            if home_t and home_t in lbl_u.split():
                return home_t
            if away_t and away_t in lbl_u.split():
                return away_t
            ha = _home_away_direction(leg)
            if _market_category(leg) in ("MONEYLINE", "SPREAD"):
                return home_t if ha == "HOME" else (away_t if ha == "AWAY" else None)
    return None


def _same_event(a: ParlayLeg, b: ParlayLeg) -> bool:
    if a.event_id and b.event_id:
        return str(a.event_id) == str(b.event_id)
    ea = str(a.short_name or a.event_name or "").strip().lower()
    eb = str(b.short_name or b.event_name or "").strip().lower()
    return bool(ea and eb and ea == eb)


def estimate_leg_pair_correlation(leg_a: ParlayLeg, leg_b: ParlayLeg) -> tuple[float, str]:
    """
    Estimate empirical Gaussian-copula correlation rho in [-0.85, +0.85] for two legs.
    Returns (rho, human_readable_reason).
    """
    if not _same_event(leg_a, leg_b):
        return 0.0, "Independent games"

    cat_a = _market_category(leg_a)
    cat_b = _market_category(leg_b)

    # Order pair deterministically for simpler symmetric rules
    pair = [(cat_a, leg_a), (cat_b, leg_b)]
    cats = {cat_a, cat_b}

    # 1. Moneyline + Spread (same game)
    if cats == {"MONEYLINE", "SPREAD"}:
        dir_a = _home_away_direction(leg_a)
        dir_b = _home_away_direction(leg_b)
        if dir_a and dir_b:
            if dir_a == dir_b:
                return 0.78, "Same-team Moneyline & Spread strongly co-move"
            return -0.65, "Opposing-team Moneyline & Spread negatively correlated"
        return 0.0, "Same-game Moneyline & Spread"

    # 2. Game Total + Player Prop
    if cats == {"TOTAL", "PROP"}:
        tot_leg = leg_a if cat_a == "TOTAL" else leg_b
        prop_leg = leg_b if cat_a == "TOTAL" else leg_a
        tot_dir = _over_under_direction(tot_leg)
        prop_dir = _over_under_direction(prop_leg)
        stat = _prop_stat_key(prop_leg)
        # Rebounds/sacks correlate more weakly with total than scoring/yards
        base_rho = 0.14 if "rebound" in stat else 0.28
        if tot_dir and prop_dir:
            if tot_dir == prop_dir:
                return base_rho, "Higher game scoring/pace lifts player production"
            return -base_rho, "Opposing game total vs player production direction"
        return 0.12, "Same-game Total & Player Prop"

    # 3. Moneyline/Spread + Player Prop
    if ("MONEYLINE" in cats or "SPREAD" in cats) and "PROP" in cats:
        game_leg = leg_a if cat_a in ("MONEYLINE", "SPREAD") else leg_b
        prop_leg = leg_b if cat_a in ("MONEYLINE", "SPREAD") else leg_a
        team_g = _extract_team_code(game_leg)
        team_p = _extract_team_code(prop_leg)
        prop_dir = _over_under_direction(prop_leg) or "OVER"
        sign = 1.0 if prop_dir == "OVER" else -1.0
        if team_g and team_p:
            if team_g == team_p:
                rho = 0.22 * sign
                return round(rho, 2), f"Same-team win/cover correlates with {team_p} player output"
            rho = -0.12 * sign
            return round(rho, 2), f"Opponent player output inversely linked to {team_g} cover"
        return round(0.08 * sign, 2), "Mild same-game script correlation"

    # 4. Player Prop + Player Prop (same game)
    if cat_a == "PROP" and cat_b == "PROP":
        pa = normalize_player_name(leg_a.player_name or "")
        pb = normalize_player_name(leg_b.player_name or "")
        dir_a = _over_under_direction(leg_a) or "OVER"
        dir_b = _over_under_direction(leg_b) or "OVER"
        same_dir = 1.0 if dir_a == dir_b else -1.0
        stat_a = _prop_stat_key(leg_a)
        stat_b = _prop_stat_key(leg_b)

        if pa and pb and pa == pb:
            # Same player, different stats
            if "pts+reb+ast" in (stat_a, stat_b) or "pra" in (stat_a, stat_b):
                return round(0.62 * same_dir, 2), "Same-player composite stat overlaps directly"
            if {"passing", "td"} <= set(f"{stat_a} {stat_b}".split()) or (
                "passing" in stat_a and "passing" in stat_b
            ):
                return round(0.55 * same_dir, 2), "Same-QB passing yards & passing TDs strongly linked"
            return round(0.18 * same_dir, 2), "Same-player minutes & usage create positive co-movement"

        team_a = _extract_team_code(leg_a)
        team_b = _extract_team_code(leg_b)
        if team_a and team_b and team_a == team_b:
            if ("passing" in stat_a and ("receiv" in stat_b or "recept" in stat_b)) or (
                "passing" in stat_b and ("receiv" in stat_a or "recept" in stat_a)
            ):
                return round(0.45 * same_dir, 2), "QB passing & teammate receiving stack"
            if ("assist" in stat_a and "point" in stat_b) or (
                "assist" in stat_b and "point" in stat_a
            ):
                return round(0.24 * same_dir, 2), "Teammate assists & scoring stack"
            if "point" in stat_a and "point" in stat_b:
                return round(-0.08 * same_dir, 2), "Teammate usage competition on scoring"
            return round(0.08 * same_dir, 2), "Same-team offensive environment"

        # Opposing players in same game (pace / shootout / overtime)
        return round(0.12 * same_dir, 2), "Same-game pace & overtime shootout link"

    # 5. Spread + Total (same game)
    if cats == {"SPREAD", "TOTAL"}:
        return 0.05, "Same-game Spread & Total weakly correlated"

    # 6. Same market type (usually conflict if opposing, or alt lines if same direction)
    if cat_a == cat_b and cat_a in ("MONEYLINE", "SPREAD", "TOTAL"):
        return 0.60, f"Same-game {cat_a.title()} markets heavily linked"

    return 0.0, "Same-game mild correlation"


def detect_parlay_conflicts(
    legs: list[ParlayLeg],
    corr_pairs: list[CorrelationPair] | None = None,
) -> list[ParlayConflict]:
    """Detect duplicate legs, mutually exclusive outcomes, and structural parlay warnings."""
    conflicts: list[ParlayConflict] = []
    n = len(legs)

    for i in range(n):
        for j in range(i + 1, n):
            la, lb = legs[i], legs[j]

            # 1. Exact duplicate leg
            same_id = bool(la.leg_id and lb.leg_id and la.leg_id == lb.leg_id)
            same_sel = (
                _same_event(la, lb)
                and str(la.market_type).lower() == str(lb.market_type).lower()
                and str(la.selection_label).strip().lower()
                == str(lb.selection_label).strip().lower()
                and str(la.side).upper() == str(lb.side).upper()
            )
            if same_id or same_sel:
                conflicts.append(
                    ParlayConflict(
                        severity="ERROR",
                        leg_indices=[i, j],
                        message=f"Duplicate leg detected: {la.selection_label}.",
                    )
                )
                continue

            if not _same_event(la, lb):
                continue

            cat_a = _market_category(la)
            cat_b = _market_category(lb)

            # 2. Same game Moneyline opposite sides
            if cat_a == "MONEYLINE" and cat_b == "MONEYLINE":
                conflicts.append(
                    ParlayConflict(
                        severity="ERROR",
                        leg_indices=[i, j],
                        message=f"Cannot combine two Moneyline sides in the same game ({la.short_name or la.event_name}).",
                    )
                )
            # 3. Same game Spread opposite sides
            elif cat_a == "SPREAD" and cat_b == "SPREAD":
                da, db = _home_away_direction(la), _home_away_direction(lb)
                if da and db and da != db:
                    conflicts.append(
                        ParlayConflict(
                            severity="ERROR",
                            leg_indices=[i, j],
                            message=f"Opposing Spread sides in {la.short_name or la.event_name} cannot be parlayed.",
                        )
                    )
            # 4. Same game Total opposite sides
            elif cat_a == "TOTAL" and cat_b == "TOTAL":
                da, db = _over_under_direction(la), _over_under_direction(lb)
                if da and db and da != db:
                    conflicts.append(
                        ParlayConflict(
                            severity="ERROR",
                            leg_indices=[i, j],
                            message=f"Opposing Total Over/Under sides in {la.short_name or la.event_name} cannot be parlayed.",
                        )
                    )
            # 5. Same player + same prop stat
            elif cat_a == "PROP" and cat_b == "PROP":
                pa = normalize_player_name(la.player_name or "")
                pb = normalize_player_name(lb.player_name or "")
                if pa and pb and pa == pb and _prop_stat_key(la) == _prop_stat_key(lb):
                    conflicts.append(
                        ParlayConflict(
                            severity="ERROR",
                            leg_indices=[i, j],
                            message=f"Conflicting {la.player_name} {_prop_stat_key(la)} legs cannot be parlayed.",
                        )
                    )

    if corr_pairs:
        for cp in corr_pairs:
            if cp.correlation <= -0.25:
                conflicts.append(
                    ParlayConflict(
                        severity="WARNING",
                        leg_indices=[cp.leg_a_idx, cp.leg_b_idx],
                        message=(
                            f"Negative SGP correlation (ρ={cp.correlation:+.2f}) between "
                            f"{cp.leg_a_label} & {cp.leg_b_label} reduces joint hit probability."
                        ),
                    )
                )

    if n >= 6:
        conflicts.append(
            ParlayConflict(
                severity="WARNING",
                leg_indices=list(range(n)),
                message=f"{n}-leg parlays carry extreme variance and compounded vig/fee drag; keep stake strictly capped.",
            )
        )

    return conflicts


# --- Gaussian Copula Joint Probability ---

def _nearest_psd_corr(mat: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix with unit diagonal onto the nearest valid PSD correlation matrix."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 1e-4)
    psd = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.maximum(np.diag(psd), 1e-8))
    corr = psd / np.outer(d, d)
    corr = np.clip(corr, -0.995, 0.995)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_joint_probability(
    probs: list[float],
    corr_matrix: np.ndarray | None = None,
) -> float:
    """
    Compute exact/copula joint win probability P(all legs win).
    - Factorizes into independent game components (exact product across independent games).
    - Size-2 correlated components use exact bivariate normal CDF.
    - Size>=3 correlated components use deterministic antithetic Gaussian-copula Monte Carlo.
    """
    n = len(probs)
    if n == 0:
        return 0.0
    p_clamped = [clamp_prob(p, 1e-4) for p in probs]
    if n == 1:
        return float(p_clamped[0])

    if corr_matrix is None:
        return float(max(1e-5, min(0.9999, math.prod(p_clamped))))

    # Connected components of correlated legs
    visited = [False] * n
    components: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp: list[int] = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not visited[v] and abs(float(corr_matrix[u, v])) > 1e-6:
                    visited[v] = True
                    stack.append(v)
        components.append(sorted(comp))

    total_prob = 1.0
    for comp in components:
        k = len(comp)
        if k == 1:
            total_prob *= p_clamped[comp[0]]
        elif k == 2:
            i, j = comp[0], comp[1]
            rho = max(-0.995, min(0.995, float(corr_matrix[i, j])))
            z = [float(norm.ppf(p_clamped[i])), float(norm.ppf(p_clamped[j]))]
            cov = [[1.0, rho], [rho, 1.0]]
            comp_p = float(multivariate_normal.cdf(z, mean=[0.0, 0.0], cov=cov))
            # Respect exact Fréchet bounds
            lo = max(0.0, p_clamped[i] + p_clamped[j] - 1.0)
            hi = min(p_clamped[i], p_clamped[j])
            total_prob *= max(lo, min(hi, comp_p))
        else:
            sub_corr = _nearest_psd_corr(corr_matrix[np.ix_(comp, comp)])
            thresholds = np.array([norm.ppf(p_clamped[idx]) for idx in comp], dtype=float)
            L = np.linalg.cholesky(sub_corr)
            # Deterministic antithetic normal draws (25k pairs = 50k draws)
            rng = np.random.default_rng(1337)
            half = rng.standard_normal((25000, k))
            z_std = np.vstack([half, -half])
            z_corr = z_std @ L.T
            hits = np.all(z_corr <= thresholds[None, :], axis=1)
            comp_p = float(np.mean(hits))
            hi = min(p_clamped[idx] for idx in comp)
            total_prob *= max(1e-5, min(hi, comp_p))

    return float(max(1e-5, min(0.9999, total_prob)))


def build_correlation_matrix(
    legs: list[ParlayLeg],
    correlation_overrides: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[CorrelationPair]]:
    """Build NxN correlation matrix and detailed list of same-game CorrelationPairs."""
    n = len(legs)
    mat = np.eye(n, dtype=float)
    pairs: list[CorrelationPair] = []
    overrides = correlation_overrides or {}

    for i in range(n):
        for j in range(i + 1, n):
            est_rho, reason = estimate_leg_pair_correlation(legs[i], legs[j])
            same_g = _same_event(legs[i], legs[j])

            # Allow override keyed by "i:j", "j:i", "idA|idB", or "idB|idA"
            override_val: float | None = None
            for key in (
                f"{i}:{j}",
                f"{j}:{i}",
                f"{legs[i].leg_id}|{legs[j].leg_id}",
                f"{legs[j].leg_id}|{legs[i].leg_id}",
            ):
                if key in overrides and overrides[key] is not None:
                    override_val = float(overrides[key])
                    break

            if override_val is not None:
                rho = max(-0.85, min(0.85, override_val))
                is_custom = True
            else:
                rho = est_rho
                is_custom = False

            if same_g or abs(rho) > 1e-6:
                mat[i, j] = rho
                mat[j, i] = rho
                pairs.append(
                    CorrelationPair(
                        leg_a_idx=i,
                        leg_b_idx=j,
                        leg_a_id=legs[i].leg_id,
                        leg_b_id=legs[j].leg_id,
                        leg_a_label=legs[i].selection_label,
                        leg_b_label=legs[j].selection_label,
                        event_name=legs[i].short_name or legs[i].event_name,
                        correlation=round(rho, 3),
                        estimated_correlation=round(est_rho, 3),
                        is_custom=is_custom,
                        reason=reason if not is_custom else f"Custom override (model est. {est_rho:+.2f})",
                    )
                )

    return mat, pairs


def _compute_parlay_confidence(
    *,
    num_legs: int,
    joint_prob: float,
    roi_pct: float,
    net_edge: float,
    positive_leg_ratio: float,
    has_error_conflict: bool,
    min_roi: float,
    min_net_edge: float,
    min_joint_prob: float,
) -> tuple[int, Tier]:
    """Score a multi-leg parlay 0-100 and assign A+/A/B/PASS tier."""
    if has_error_conflict or num_legs < 2:
        return 0, "PASS"

    # 1. Expected ROI strength (0 - 52 pts)
    roi_pts = max(0.0, min(52.0, (roi_pct / 0.16) * 52.0))
    # 2. Joint probability / variance stability (0 - 28 pts)
    prob_pts = max(0.0, min(28.0, (math.sqrt(max(0.0, joint_prob)) / math.sqrt(0.55)) * 28.0))
    # 3. Leg quality / lack of negative-EV parlay tax legs (0 - 20 pts)
    leg_pts = max(0.0, min(20.0, positive_leg_ratio * 20.0))

    score = max(0, min(100, int(round(roi_pts + prob_pts + leg_pts))))

    if roi_pct < min_roi or net_edge < min_net_edge or joint_prob < min_joint_prob:
        return min(score, 59), "PASS"

    if score >= 76 and roi_pct >= 0.10:
        return max(score, 85), "A+"
    if score >= 65 and roi_pct >= 0.05:
        return max(score, 72), "A"
    return max(score, 60), "B"


def _parlay_kelly_scale(num_legs: int, has_live: bool) -> float:
    """Apply variance-aware Kelly dampener for multi-leg parlays."""
    if num_legs <= 1:
        base = 1.0
    elif num_legs == 2:
        base = 0.72
    elif num_legs == 3:
        base = 0.58
    else:
        base = max(0.35, 0.75 / math.sqrt(num_legs))
    if has_live:
        base *= 0.80
    return base


# --- 1. Predictions Mode (DKeX) Parlay Evaluator ---

def evaluate_predictions_parlay(
    legs: list[ParlayLeg],
    *,
    pricing_structure: ParlayPricingStructure = "COMBO_CONTRACT",
    fee_per_contract: float = 0.015,
    custom_combo_ask: float | None = None,
    contracts_override: int | None = None,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.04,
    min_net_edge: float = 0.015,
    min_roi: float = 0.03,
    correlation_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a DraftKings Predictions (DKeX) multi-leg parlay / combo contract.

    pricing_structure:
      - "COMBO_CONTRACT": Single binary combo event contract ($0.01-$0.99 -> $1.00) with
        a single exchange fee (`fee_per_contract`). Ask defaults to product of leg asks
        unless `custom_combo_ask` is provided.
      - "SEQUENTIAL_ROLLOVER": Synthetic rollover where winnings from leg k buy leg k+1,
        paying per-leg exchange fees on every leg (cost = prod(ask_i + fee_i)).
    """
    if not legs:
        raise ValueError("At least one leg is required to evaluate a parlay.")

    n = len(legs)
    fee = round(max(0.0, float(fee_per_contract)), 4)
    has_live = any(lg.is_live for lg in legs)

    corr_mat, corr_pairs = build_correlation_matrix(legs, correlation_overrides)
    conflicts = detect_parlay_conflicts(legs, corr_pairs)
    has_error = any(c.severity == "ERROR" for c in conflicts)

    leg_probs = [clamp_prob(lg.model_prob, 0.005) for lg in legs]
    leg_cons_probs = [
        clamp_prob(
            lg.conservative_prob
            if lg.conservative_prob is not None
            else shrink_probability(
                lg.model_prob,
                anchor_prob=lg.anchor_prob,
                uncertainty=0.14 if lg.is_live else 0.10,
            ),
            0.005,
        )
        for lg in legs
    ]

    indep_prob = float(max(1e-5, min(0.9999, math.prod(leg_probs))))
    joint_prob = compute_joint_probability(leg_probs, corr_mat)
    joint_cons_prob = compute_joint_probability(leg_cons_probs, corr_mat)
    corr_impact = joint_prob - indep_prob

    # Per-leg pricing & diagnostics
    leg_asks: list[float] = []
    leg_fees: list[float] = []
    leg_breakdown: list[dict[str, Any]] = []
    pos_legs = 0

    for idx, lg in enumerate(legs):
        ask = round(
            max(0.01, min(0.99, float(lg.contract_price if lg.contract_price is not None else 0.50))),
            4,
        )
        l_fee = round(
            max(0.0, float(lg.fee_per_contract if lg.fee_per_contract is not None else fee)),
            4,
        )
        leg_asks.append(ask)
        leg_fees.append(l_fee)
        l_cost = round(ask + l_fee, 4)
        l_edge = round(leg_probs[idx] - l_cost, 4)
        l_roi = round(l_edge / l_cost, 4) if l_cost > 0 else 0.0
        is_tax = l_edge < 0.0
        if not is_tax:
            pos_legs += 1
        leg_breakdown.append(
            {
                "index": idx,
                "leg_id": lg.leg_id,
                "sport": lg.sport,
                "event_id": lg.event_id,
                "event_name": lg.short_name or lg.event_name,
                "market_type": lg.market_type,
                "selection_label": lg.selection_label,
                "side": lg.side,
                "contract_price": ask,
                "fee_per_contract": l_fee,
                "all_in_cost": l_cost,
                "model_prob": round(leg_probs[idx], 4),
                "net_edge": l_edge,
                "roi_pct": l_roi,
                "is_parlay_tax": is_tax,
                "is_live": lg.is_live,
            }
        )

    implied_combo_ask = round(max(0.01, min(0.99, math.prod(leg_asks))), 4)
    if pricing_structure == "SEQUENTIAL_ROLLOVER":
        all_in_per_leg = [min(0.995, a + f) for a, f in zip(leg_asks, leg_fees)]
        combo_ask = implied_combo_ask
        all_in_cost = round(max(0.01, min(0.995, math.prod(all_in_per_leg))), 4)
        fee_drag = round(max(0.0, all_in_cost - combo_ask), 4)
        effective_fee = fee_drag
    else:
        combo_ask = (
            round(max(0.01, min(0.99, float(custom_combo_ask))), 4)
            if custom_combo_ask is not None
            else implied_combo_ask
        )
        effective_fee = fee
        all_in_cost = round(min(0.999, combo_ask + effective_fee), 4)
        fee_drag = effective_fee

    breakeven_prob = round(min(1.0, all_in_cost), 4)
    net_edge = round(joint_prob - all_in_cost, 4)
    ev_per_contract = round(joint_prob * 1.0 - all_in_cost, 4)
    roi_pct = round(ev_per_contract / all_in_cost, 4) if all_in_cost > 0 else 0.0

    fair_combo_max = round(max(0.01, min(0.98, joint_prob - effective_fee - min_net_edge)), 3)
    fair_am = prob_to_american_odds(joint_prob)
    am_equiv = prob_to_american_odds(all_in_cost)
    eff_decimal = round(1.0 / max(0.01, all_in_cost), 4)

    conf_score, conf_tier = _compute_parlay_confidence(
        num_legs=n,
        joint_prob=joint_prob,
        roi_pct=roi_pct,
        net_edge=net_edge,
        positive_leg_ratio=pos_legs / max(1, n),
        has_error_conflict=has_error,
        min_roi=min_roi,
        min_net_edge=min_net_edge,
        min_joint_prob=0.04,
    )

    # Fractional Kelly sizing on $1 binary combo contract
    parlay_cap = min(max_bet_pct, 0.03)
    k_scale = _parlay_kelly_scale(n, has_live)
    if has_error or n < 2 or all_in_cost >= 0.995 or joint_cons_prob <= all_in_cost or conf_tier == "PASS":
        kelly_applied = 0.0
        rec_contracts = 0
        rec_outlay = 0.0
    else:
        full_kelly = (joint_cons_prob - all_in_cost) / (1.0 - all_in_cost)
        scaled = max(0.0, full_kelly * kelly_multiplier * k_scale)
        kelly_applied = round(min(parlay_cap, scaled), 5)
        target_dollars = bankroll * kelly_applied
        rec_contracts = int(math.floor(target_dollars / all_in_cost))
        if rec_contracts == 0 and bankroll >= all_in_cost and all_in_cost <= bankroll * 0.08:
            rec_contracts = 1
        rec_outlay = round(rec_contracts * all_in_cost, 2)

    active_contracts = (
        max(1, int(contracts_override))
        if contracts_override is not None and contracts_override > 0
        else (rec_contracts if rec_contracts > 0 else 1)
    )
    active_outlay = round(active_contracts * all_in_cost, 2)
    potential_payout = round(active_contracts * 1.0, 2)
    potential_profit = round(potential_payout - active_outlay, 2)
    ev_dollars = round(active_outlay * roi_pct, 2)

    unique_events = {str(lg.event_id or lg.short_name or lg.event_name) for lg in legs}
    is_sgp = len(unique_events) < n

    return {
        "mode": "predictions",
        "pricing_structure": pricing_structure,
        "num_legs": n,
        "is_sgp": is_sgp,
        "has_conflicts": has_error,
        "conflicts": [c.to_dict() for c in conflicts],
        "correlation_pairs": [cp.to_dict() for cp in corr_pairs],
        "legs": leg_breakdown,
        "implied_combo_ask": implied_combo_ask,
        "combo_ask_price": combo_ask,
        "is_custom_combo_ask": custom_combo_ask is not None and pricing_structure == "COMBO_CONTRACT",
        "fee_per_contract": effective_fee,
        "fee_drag": fee_drag,
        "all_in_cost": all_in_cost,
        "breakeven_prob": breakeven_prob,
        "effective_decimal_odds": eff_decimal,
        "american_equivalent": am_equiv,
        "fair_american_odds": fair_am,
        "independent_model_prob": round(indep_prob, 4),
        "joint_model_prob": round(joint_prob, 4),
        "joint_conservative_prob": round(joint_cons_prob, 4),
        "correlation_impact": round(corr_impact, 4),
        "net_edge": net_edge,
        "ev_per_contract": ev_per_contract,
        "roi_pct": roi_pct,
        "fair_combo_max_price": fair_combo_max,
        "kelly_fraction_applied": kelly_applied,
        "recommended_contracts": rec_contracts,
        "recommended_outlay": rec_outlay,
        "active_contracts": active_contracts,
        "active_outlay": active_outlay,
        "potential_payout": potential_payout,
        "potential_profit": potential_profit,
        "ev_dollars": ev_dollars,
        "parlay_tax_legs_count": n - pos_legs,
        "confidence_score": conf_score,
        "confidence_tier": conf_tier,
    }


# --- 2. Traditional Sportsbooks (DK / FD) Parlay Evaluator ---

def _resolve_leg_odds_for_book(leg: ParlayLeg, target_book: str) -> tuple[int | None, VenueName]:
    """Return (american_odds, resolved_book) for a given target book preference."""
    tb = str(target_book or "BEST").upper()
    if tb == "DRAFTKINGS":
        if leg.dk_odds is not None:
            return int(leg.dk_odds), "DRAFTKINGS"
        if leg.book == "DRAFTKINGS" and leg.american_odds is not None:
            return int(leg.american_odds), "DRAFTKINGS"
        return (int(leg.american_odds) if leg.american_odds is not None else None), "DRAFTKINGS"
    if tb == "FANDUEL":
        if leg.fd_odds is not None:
            return int(leg.fd_odds), "FANDUEL"
        if leg.book == "FANDUEL" and leg.american_odds is not None:
            return int(leg.american_odds), "FANDUEL"
        return (int(leg.american_odds) if leg.american_odds is not None else None), "FANDUEL"

    # BEST mode: pick whichever of DK or FD offers better decimal odds
    candidates: list[tuple[float, int, VenueName]] = []
    if leg.dk_odds is not None:
        candidates.append((american_to_decimal(leg.dk_odds), int(leg.dk_odds), "DRAFTKINGS"))
    if leg.fd_odds is not None:
        candidates.append((american_to_decimal(leg.fd_odds), int(leg.fd_odds), "FANDUEL"))
    if candidates:
        best = max(candidates, key=lambda x: x[0])
        return best[1], best[2]
    if leg.american_odds is not None:
        bk: VenueName = leg.book if leg.book in ("DRAFTKINGS", "FANDUEL") else "DRAFTKINGS"
        return int(leg.american_odds), bk
    return None, "DRAFTKINGS"


def evaluate_sportsbook_parlay(
    legs: list[ParlayLeg],
    *,
    book: str = "BEST",
    profit_boost_pct: float = 0.0,
    is_bonus_bet: bool = False,
    custom_parlay_odds: float | None = None,
    stake_override: float | None = None,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.04,
    min_ev_edge: float = 0.02,
    correlation_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a Traditional Sportsbook (DraftKings / FanDuel) Parlay or SGP.
    Automatically compares full-ticket DK vs FD combined odds whenever both books
    are available on all legs.
    """
    if not legs:
        raise ValueError("At least one leg is required to evaluate a parlay.")

    n = len(legs)
    boost = max(0.0, float(profit_boost_pct))
    has_live = any(lg.is_live for lg in legs)

    corr_mat, corr_pairs = build_correlation_matrix(legs, correlation_overrides)
    conflicts = detect_parlay_conflicts(legs, corr_pairs)
    has_error = any(c.severity == "ERROR" for c in conflicts)

    leg_probs = [clamp_prob(lg.model_prob, 0.005) for lg in legs]
    leg_cons_probs = [
        clamp_prob(
            lg.conservative_prob
            if lg.conservative_prob is not None
            else shrink_probability(
                lg.model_prob,
                anchor_prob=lg.anchor_prob,
                uncertainty=0.13 if lg.is_live else 0.10,
            ),
            0.005,
        )
        for lg in legs
    ]

    indep_prob = float(max(1e-5, min(0.9999, math.prod(leg_probs))))
    joint_prob = compute_joint_probability(leg_probs, corr_mat)
    joint_cons_prob = compute_joint_probability(leg_cons_probs, corr_mat)
    corr_impact = joint_prob - indep_prob

    # Compute full-ticket DK and FD comparison whenever possible
    book_comparison: dict[str, dict[str, Any]] = {}
    for bk_name in ("DRAFTKINGS", "FANDUEL"):
        bk_odds_list: list[int] = []
        complete = True
        for lg in legs:
            od = lg.dk_odds if bk_name == "DRAFTKINGS" else lg.fd_odds
            if od is None and lg.book == bk_name and lg.american_odds is not None:
                od = lg.american_odds
            if od is None:
                complete = False
                break
            bk_odds_list.append(int(od))
        if complete and bk_odds_list:
            raw_dec = math.prod(american_to_decimal(o) for o in bk_odds_list)
            raw_am = decimal_to_american(raw_dec)
            eff_dec = 1.0 + (raw_dec - 1.0) * (1.0 + boost) if boost > 0.0 else raw_dec
            eff_am = decimal_to_american(eff_dec) if boost > 0.0 else raw_am
            if is_bonus_bet:
                # Bonus bet expected cash retention per $1 bonus face value = p * (eff_dec - 1)
                ev_v = round(joint_prob * (eff_dec - 1.0), 4)
            else:
                ev_v = round(joint_prob * eff_dec - 1.0, 4)
            book_comparison[bk_name] = {
                "book": bk_name,
                "raw_decimal_odds": round(raw_dec, 4),
                "raw_american_odds": raw_am,
                "effective_decimal_odds": round(eff_dec, 4),
                "effective_american_odds": eff_am,
                "implied_prob": round(1.0 / eff_dec, 4),
                "ev_pct": ev_v,
            }

    # Resolve target venue & combined odds
    req_book = str(book or "BEST").upper()
    if req_book in ("DRAFTKINGS", "FANDUEL"):
        selected_book: VenueName = req_book  # type: ignore[assignment]
    elif book_comparison:
        best_bk_entry = max(
            book_comparison.values(), key=lambda x: x["effective_decimal_odds"]
        )
        selected_book = best_bk_entry["book"]
    else:
        # Mixed legs without complete single-book quotes: default to most common leg book
        dk_cnt = sum(1 for lg in legs if lg.book == "DRAFTKINGS")
        fd_cnt = sum(1 for lg in legs if lg.book == "FANDUEL")
        selected_book = "FANDUEL" if fd_cnt > dk_cnt else "DRAFTKINGS"

    leg_breakdown: list[dict[str, Any]] = []
    leg_decimals: list[float] = []
    pos_legs = 0
    for idx, lg in enumerate(legs):
        od_val, resolved_bk = _resolve_leg_odds_for_book(
            lg, selected_book if req_book in ("DRAFTKINGS", "FANDUEL") or selected_book in book_comparison else "BEST"
        )
        if od_val is None:
            od_val = -110
        l_dec = american_to_decimal(od_val)
        leg_decimals.append(l_dec)
        l_imp = round(1.0 / l_dec, 4)
        l_ev = round(leg_probs[idx] * l_dec - 1.0, 4)
        is_tax = l_ev < 0.0
        if not is_tax:
            pos_legs += 1
        leg_breakdown.append(
            {
                "index": idx,
                "leg_id": lg.leg_id,
                "sport": lg.sport,
                "event_id": lg.event_id,
                "event_name": lg.short_name or lg.event_name,
                "market_type": lg.market_type,
                "selection_label": lg.selection_label,
                "side": lg.side,
                "book": resolved_bk,
                "american_odds": od_val,
                "dk_odds": lg.dk_odds,
                "fd_odds": lg.fd_odds,
                "decimal_odds": round(l_dec, 4),
                "implied_prob": l_imp,
                "model_prob": round(leg_probs[idx], 4),
                "ev_pct": l_ev,
                "is_parlay_tax": is_tax,
                "is_live": lg.is_live,
            }
        )

    unboosted_dec = (
        american_to_decimal(custom_parlay_odds)
        if custom_parlay_odds is not None
        else math.prod(leg_decimals)
    )
    unboosted_am = (
        int(round(float(custom_parlay_odds)))
        if custom_parlay_odds is not None
        else decimal_to_american(unboosted_dec)
    )

    if boost > 0.0:
        eff_dec = apply_profit_boost(unboosted_am, boost) if custom_parlay_odds is not None else (
            1.0 + (unboosted_dec - 1.0) * (1.0 + boost)
        )
        eff_am = decimal_to_american(eff_dec)
    else:
        eff_dec = unboosted_dec
        eff_am = unboosted_am

    b = max(1e-6, eff_dec - 1.0)
    implied_prob = round(1.0 / eff_dec, 4)
    fair_am = prob_to_american_odds(joint_prob)
    prob_edge = round(joint_prob - implied_prob, 4)

    if is_bonus_bet:
        # Expected cash yield per $1 of bonus bet face value (profit-only payout)
        ev_pct = round(joint_prob * b, 4)
        base_ev_pct = round(joint_prob * max(1e-6, unboosted_dec - 1.0), 4)
    else:
        ev_pct = round(joint_prob * eff_dec - 1.0, 4)
        base_ev_pct = round(joint_prob * unboosted_dec - 1.0, 4)

    boost_ev_lift = round(ev_pct - base_ev_pct, 4) if boost > 0.0 else 0.0

    # Parlay line-shopping gain if both DK and FD have complete quotes
    line_shopping_gain_pct: float | None = None
    if "DRAFTKINGS" in book_comparison and "FANDUEL" in book_comparison:
        dk_ev = book_comparison["DRAFTKINGS"]["ev_pct"]
        fd_ev = book_comparison["FANDUEL"]["ev_pct"]
        line_shopping_gain_pct = round(abs(dk_ev - fd_ev), 4)

    conf_score, conf_tier = _compute_parlay_confidence(
        num_legs=n,
        joint_prob=joint_prob,
        roi_pct=ev_pct,
        net_edge=prob_edge if not is_bonus_bet else ev_pct,
        positive_leg_ratio=pos_legs / max(1, n),
        has_error_conflict=has_error,
        min_roi=min_ev_edge if not is_bonus_bet else 0.55,
        min_net_edge=0.003 if not is_bonus_bet else 0.55,
        min_joint_prob=0.02,
    )

    # Variance-aware Fractional Kelly sizing
    parlay_cap = min(max_bet_pct, 0.03)
    k_scale = _parlay_kelly_scale(n, has_live)
    if has_error or n < 2 or conf_tier == "PASS" or is_bonus_bet or (joint_cons_prob * eff_dec - 1.0) <= 0.0:
        kelly_applied = 0.0
        rec_stake = 0.0
    else:
        full_kelly = (joint_cons_prob * b - (1.0 - joint_cons_prob)) / b
        scaled = max(0.0, full_kelly * kelly_multiplier * k_scale)
        kelly_applied = round(min(parlay_cap, scaled), 5)
        raw_stake = bankroll * kelly_applied
        if bankroll >= 200.0:
            rec_stake = round(max(1.0, round(raw_stake)), 2) if raw_stake >= 0.75 else 0.0
        else:
            rec_stake = round(max(0.50, round(raw_stake * 2.0) / 2.0), 2) if raw_stake >= 0.35 else 0.0
        rec_stake = min(rec_stake, round(bankroll * parlay_cap, 2))

    default_preview_stake = rec_stake if rec_stake > 0 else round(max(5.0, min(25.0, bankroll * 0.01)), 2)
    active_stake = (
        round(float(stake_override), 2)
        if stake_override is not None and stake_override > 0
        else default_preview_stake
    )
    potential_profit = round(active_stake * b, 2)
    potential_payout = round(potential_profit if is_bonus_bet else active_stake * eff_dec, 2)
    ev_dollars = round(active_stake * ev_pct, 2)

    unique_events = {str(lg.event_id or lg.short_name or lg.event_name) for lg in legs}
    is_sgp = len(unique_events) < n

    return {
        "mode": "sportsbook",
        "book": selected_book,
        "requested_book": req_book,
        "num_legs": n,
        "is_sgp": is_sgp,
        "is_bonus_bet": is_bonus_bet,
        "profit_boost_pct": round(boost, 4),
        "has_conflicts": has_error,
        "conflicts": [c.to_dict() for c in conflicts],
        "correlation_pairs": [cp.to_dict() for cp in corr_pairs],
        "legs": leg_breakdown,
        "book_comparison": book_comparison,
        "line_shopping_gain_pct": line_shopping_gain_pct,
        "is_custom_odds": custom_parlay_odds is not None,
        "raw_decimal_odds": round(unboosted_dec, 4),
        "raw_american_odds": unboosted_am,
        "effective_decimal_odds": round(eff_dec, 4),
        "effective_american_odds": eff_am,
        "implied_prob": implied_prob,
        "fair_american_odds": fair_am,
        "independent_model_prob": round(indep_prob, 4),
        "joint_model_prob": round(joint_prob, 4),
        "joint_conservative_prob": round(joint_cons_prob, 4),
        "correlation_impact": round(corr_impact, 4),
        "prob_edge": prob_edge,
        "base_ev_pct": base_ev_pct,
        "ev_pct": ev_pct,
        "boost_ev_lift": boost_ev_lift,
        "kelly_fraction_applied": kelly_applied,
        "recommended_stake": rec_stake,
        "active_stake": active_stake,
        "potential_profit": potential_profit,
        "potential_payout": potential_payout,
        "ev_dollars": ev_dollars,
        "parlay_tax_legs_count": n - pos_legs,
        "confidence_score": conf_score,
        "confidence_tier": conf_tier,
    }


# --- 3. Auto-Build Optimal +EV Parlay Generator (Both Modes) ---

def suggest_optimal_parlays(
    candidate_legs: list[ParlayLeg],
    *,
    mode: Literal["predictions", "sportsbook"],
    num_legs: int = 2,
    allow_same_game: bool = False,
    max_results: int = 6,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.04,
    # Predictions options
    pricing_structure: ParlayPricingStructure = "COMBO_CONTRACT",
    fee_per_contract: float = 0.015,
    # Sportsbook options
    book: str = "BEST",
    profit_boost_pct: float = 0.0,
    is_bonus_bet: bool = False,
) -> list[dict[str, Any]]:
    """
    Search top candidate legs from the active slate and return the highest-value
    valid N-leg parlays for either Predictions (DKeX) or Sportsbooks (DK/FD).
    """
    k = max(2, min(4, int(num_legs)))
    if len(candidate_legs) < k:
        return []

    # Score individual legs and prune pool to top 16 for fast exact combinatorial search
    def _leg_rank(lg: ParlayLeg) -> float:
        p = clamp_prob(lg.model_prob, 0.01)
        if mode == "predictions":
            c = float(lg.contract_price if lg.contract_price is not None else 0.50)
            f = float(lg.fee_per_contract if lg.fee_per_contract is not None else fee_per_contract)
            edge = p - (c + f)
            return edge * 2.0 + p * 0.25
        od, _ = _resolve_leg_odds_for_book(lg, book)
        dec = american_to_decimal(od if od is not None else -110)
        ev = p * dec - 1.0
        return ev * 2.0 + p * 0.15

    # Deduplicate by leg_id and keep top legs (preserving game diversity)
    seen_ids: set[str] = set()
    unique_legs: list[ParlayLeg] = []
    for lg in sorted(candidate_legs, key=_leg_rank, reverse=True):
        if lg.leg_id in seen_ids:
            continue
        seen_ids.add(lg.leg_id)
        unique_legs.append(lg)
        if len(unique_legs) >= 16:
            break

    if len(unique_legs) < k:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for combo in itertools.combinations(unique_legs, k):
        combo_list = list(combo)
        if not allow_same_game:
            ev_ids = {str(lg.event_id or lg.short_name or lg.event_name) for lg in combo_list}
            if len(ev_ids) < k:
                continue

        if mode == "predictions":
            ev_res = evaluate_predictions_parlay(
                combo_list,
                pricing_structure=pricing_structure,
                fee_per_contract=fee_per_contract,
                bankroll=bankroll,
                kelly_multiplier=kelly_multiplier,
                max_bet_pct=max_bet_pct,
            )
            if ev_res["has_conflicts"]:
                continue
            roi = float(ev_res["roi_pct"])
            jp = float(ev_res["joint_model_prob"])
            # Require positive EV (or allow slight fallback only if nothing is +EV)
            rank_score = roi * 1.0 + float(ev_res["net_edge"]) * 1.5 + jp * 0.15
            if ev_res["confidence_tier"] != "PASS":
                rank_score += 0.25
            scored.append((rank_score, ev_res))
        else:
            ev_res = evaluate_sportsbook_parlay(
                combo_list,
                book=book,
                profit_boost_pct=profit_boost_pct,
                is_bonus_bet=is_bonus_bet,
                bankroll=bankroll,
                kelly_multiplier=kelly_multiplier,
                max_bet_pct=max_bet_pct,
            )
            if ev_res["has_conflicts"]:
                continue
            ev_v = float(ev_res["ev_pct"])
            jp = float(ev_res["joint_model_prob"])
            rank_score = ev_v * 1.2 + jp * 0.12
            if ev_res["confidence_tier"] != "PASS":
                rank_score += 0.25
            scored.append((rank_score, ev_res))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer +EV combinations first; if none are +EV on a tight slate, return top closest-to-fair
    positive_only = [
        item for item in scored
        if (
            (mode == "predictions" and item[1]["net_edge"] > 0)
            or (mode == "sportsbook" and item[1]["ev_pct"] > 0)
        )
    ]
    pool = positive_only if positive_only else scored

    # Encourage diversity across suggestions (avoid repeating exact same anchor pair in every card)
    selected: list[dict[str, Any]] = []
    leg_usage: dict[str, int] = {}
    max_per_leg = max(2, math.ceil(max_results / 2))

    for _, ev_res in pool:
        if len(selected) >= max_results:
            break
        ids = [str(lg["leg_id"]) for lg in ev_res["legs"]]
        if any(leg_usage.get(lid, 0) >= max_per_leg for lid in ids) and len(pool) > max_results:
            continue
        selected.append(ev_res)
        for lid in ids:
            leg_usage[lid] = leg_usage.get(lid, 0) + 1

    # Top up if diversity filter skipped too many
    if len(selected) < max_results:
        chosen_sigs = {tuple(lg["leg_id"] for lg in s["legs"]) for s in selected}
        for _, ev_res in pool:
            if len(selected) >= max_results:
                break
            sig = tuple(lg["leg_id"] for lg in ev_res["legs"])
            if sig not in chosen_sigs:
                selected.append(ev_res)
                chosen_sigs.add(sig)

    return selected


def format_parlay_summary_label(legs: list[dict[str, Any]], max_len: int = 120) -> str:
    """Create a concise human-readable parlay label for ledger rows."""
    parts = [str(lg.get("selection_label") or "").strip() for lg in legs if lg.get("selection_label")]
    joined = " + ".join(parts)
    if len(joined) <= max_len:
        return joined
    return joined[: max_len - 1].rstrip() + "…"


def format_parlay_notes(evaluation: dict[str, Any]) -> str:
    """Format detailed itemized leg notes for ledger persistence."""
    mode = evaluation.get("mode", "predictions")
    lines: list[str] = []
    for idx, lg in enumerate(evaluation.get("legs", []), start=1):
        if mode == "sportsbook":
            od_str = format_american(lg.get("american_odds", -110))
            lines.append(
                f"Leg {idx}: {lg['selection_label']} ({lg['event_name']}) @ {od_str} [Win {lg['model_prob']*100:.1f}%]"
            )
        else:
            px_c = float(lg.get("contract_price", 0.5)) * 100.0
            lines.append(
                f"Leg {idx}: {lg['side']} {lg['selection_label']} ({lg['event_name']}) @ {px_c:.0f}¢ [Win {lg['model_prob']*100:.1f}%]"
            )
    if evaluation.get("is_sgp"):
        lines.append(
            f"SGP Joint Win: {evaluation['joint_model_prob']*100:.1f}% "
            f"(Indep {evaluation['independent_model_prob']*100:.1f}%)"
        )
    return " | ".join(lines)

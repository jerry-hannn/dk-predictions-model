"""Traditional sportsbook (DraftKings & FanDuel) American-odds math and multi-strategy engine.

Covers:
1. +EV Line Shopping & Fractional Kelly Dollar Sizing
2. Cross-Book Arbitrage (DK <-> FD and optional fee-adjusted DKeX <-> DK/FD) with anti-limiting rounding
3. Middles & Polish Middles (discrepant spreads, totals, and player props)
4. Low-Hold Rollover, Bonus Bet / Free-Bet Conversion, and Profit Boost EV Optimization
5. Steam / Off-Market Lag Detection against Sharp-Only Consensus
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.engine.dk_math import (
    Tier,
    american_to_implied_prob,
    clamp_prob,
    prob_to_american_odds,
    shrink_probability,
)

VenueName = Literal["DRAFTKINGS", "FANDUEL", "DKEX"]

NFL_KEY_NUMBERS = (3, 7, 6, 10, 4, 14, 1, 2)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds (-110, +150) to decimal odds (1.9091, 2.5000)."""
    am = float(american_odds)
    if am == 0.0:
        raise ValueError("American odds cannot be 0.")
    if am < 0.0:
        return 1.0 + (100.0 / (-am))
    return 1.0 + (am / 100.0)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds (> 1.0) to nearest integer American odds."""
    dec = max(1.005, float(decimal_odds))
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return -int(round(100.0 / (dec - 1.0)))


def american_to_profit_per_dollar(american_odds: float) -> float:
    """Return net profit b per $1 staked (decimal_odds - 1.0)."""
    return american_to_decimal(american_odds) - 1.0


def format_american(odds: float | int) -> str:
    val = int(round(float(odds)))
    return f"+{val}" if val > 0 else f"{val}"


def compute_two_way_hold(odds_a: float, odds_b: float) -> float:
    """Return sportsbook overround/hold as a decimal (e.g., -110/-110 -> +0.0476)."""
    qa = american_to_implied_prob(odds_a)
    qb = american_to_implied_prob(odds_b)
    return round(qa + qb - 1.0, 5)


def apply_profit_boost(american_odds: float, boost_pct: float) -> float:
    """
    Apply a traditional DK/FD Profit Boost token (e.g., boost_pct=0.25 for +25%).
    Sportsbook profit boosts multiply the NET PROFIT portion (b' = b * (1 + boost_pct)),
    leaving the returned stake unchanged. Returns boosted decimal odds.
    """
    boost = max(0.0, float(boost_pct))
    b = american_to_profit_per_dollar(american_odds)
    return 1.0 + b * (1.0 + boost)


def dkex_ask_to_effective_decimal(
    ask_price: float, fee_per_contract: float = 0.015
) -> float:
    """
    Convert a DraftKings Predictions (DKeX) binary contract ask + per-contract fee
    into effective decimal odds (`$1.00 / (ask + fee)`).
    """
    cost = max(0.01, min(0.999, float(ask_price) + max(0.0, float(fee_per_contract))))
    return 1.0 / cost


def _round_stake_to_step(amount: float, step: float) -> float:
    if step <= 0.0:
        return round(max(0.0, amount), 2)
    rounded = round(amount / step) * step
    if rounded <= 0.0 and amount > 0.0:
        rounded = step
    return round(float(rounded), 2)


def compute_sportsbook_confidence(
    *,
    win_prob: float,
    ev_pct: float,
    prob_edge: float,
    agreement_score: float = 0.80,
    injury_clarity: float = 1.0,
    min_ev_edge: float = 0.02,
    min_win_prob: float = 0.40,
    min_prob_edge: float = 0.020,
) -> tuple[int, Tier]:
    """
    Compute 0-100 confidence score and tier (A+, A, B, PASS) for traditional American-odds bets.
    Requires both positive expected ROI (`min_ev_edge`) and meaningful probability edge
    (`min_prob_edge`) so minor calibration error on deep longshots never triggers false +EV plays.
    """
    # 1. Expected ROI strength (0 - 48 pts): 0% -> 0, 2% -> 16, 5% -> 34, 9%+ -> 48
    ev_pts = max(0.0, min(48.0, (ev_pct / 0.085) * 48.0))
    # 2. Win probability stability (0 - 28 pts): higher hit rate lowers bankroll variance
    prob_pts = max(0.0, min(28.0, ((win_prob - 0.30) / 0.50) * 28.0))
    # 3. Model-market agreement & line quality (0 - 14 pts)
    agree_pts = max(0.0, min(14.0, agreement_score * 14.0))
    # 4. Injury report clarity (0 - 10 pts)
    inj_pts = max(0.0, min(10.0, injury_clarity * 10.0))

    score = max(0, min(100, int(round(ev_pts + prob_pts + agree_pts + inj_pts))))

    if ev_pct < min_ev_edge or win_prob < min_win_prob or prob_edge < min_prob_edge:
        return min(score, 59), "PASS"

    if ev_pct >= 0.055 and score >= 74:
        return max(score, 85), "A+"
    if ev_pct >= 0.032 and score >= 65:
        return max(score, 72), "A"
    return max(score, 60), "B"


@dataclass(frozen=True)
class SportsbookEvaluation:
    book: VenueName
    side: str
    american_odds: int
    effective_american_odds: int
    decimal_odds: float
    implied_prob: float
    model_prob: float
    conservative_prob: float
    fair_american_odds: int
    max_acceptable_odds: int
    prob_edge: float
    ev_pct: float
    ev_dollars: float
    profit_boost_pct: float
    kelly_fraction_applied: float
    recommended_stake: float
    potential_profit: float
    confidence_score: int
    confidence_tier: Tier

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_sportsbook_side(
    *,
    book: VenueName,
    side: str,
    side_true_prob: float,
    american_odds: float,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.04,
    min_ev_edge: float = 0.02,
    min_win_prob: float = 0.40,
    min_prob_edge: float = 0.020,
    anchor_prob: float | None = None,
    uncertainty: float = 0.12,
    injury_clarity: float = 1.0,
    profit_boost_pct: float = 0.0,
    max_neg_clv: float = 0.010,
) -> SportsbookEvaluation:
    """Evaluate a traditional American-odds sportsbook wager with fractional Kelly staking."""
    am_raw = int(round(float(american_odds)))
    if am_raw == 0:
        am_raw = 100
    boost = max(0.0, float(profit_boost_pct))
    dec = apply_profit_boost(am_raw, boost) if boost > 0.0 else american_to_decimal(am_raw)
    eff_am = decimal_to_american(dec) if boost > 0.0 else am_raw
    b = max(1e-6, dec - 1.0)
    imp_p = round(1.0 / dec, 4)

    p_true = round(clamp_prob(side_true_prob), 4)
    p_cons = round(
        shrink_probability(p_true, anchor_prob=anchor_prob, uncertainty=uncertainty), 4
    )

    ev_pct = round(p_true * dec - 1.0, 4)
    prob_edge = round(p_true - imp_p, 4)
    fair_am = prob_to_american_odds(p_true)

    # Worst decimal odds that still clear min_ev_edge: p_true * d_min - 1 = min_ev_edge
    d_min = max(1.02, (1.0 + max(0.0, min_ev_edge)) / max(0.02, p_true))
    max_acc_am = decimal_to_american(d_min)

    if anchor_prob is not None:
        gap = abs(p_true - anchor_prob)
        agreement = max(0.25, 1.0 - max(0.0, gap - 0.05) * 3.2)
    else:
        agreement = 0.75

    # Relax prob_edge slightly when a large profit boost creates huge ROI on an otherwise tight line
    eff_min_prob_edge = min(min_prob_edge, 0.010) if boost >= 0.20 else min_prob_edge

    conf_score, conf_tier = compute_sportsbook_confidence(
        win_prob=p_true,
        ev_pct=ev_pct,
        prob_edge=prob_edge,
        agreement_score=agreement,
        injury_clarity=injury_clarity,
        min_ev_edge=min_ev_edge,
        min_win_prob=min_win_prob,
        min_prob_edge=eff_min_prob_edge,
    )

    # Sharp-alignment CLV guard: avoid firing unboosted bets at prices materially worse than sharp consensus
    if (
        anchor_prob is not None
        and boost <= 0.0
        and (anchor_prob - imp_p) < -max_neg_clv
    ):
        conf_score = min(conf_score, 58)
        conf_tier = "PASS"

    if conf_tier == "PASS" or (p_cons * dec - 1.0) <= 0.0:
        kelly_applied = 0.0
        stake = 0.0
    else:
        full_kelly = (p_cons * b - (1.0 - p_cons)) / b
        scaled_kelly = max(0.0, full_kelly * kelly_multiplier)
        kelly_applied = round(min(max_bet_pct, scaled_kelly), 5)
        raw_stake = bankroll * kelly_applied
        # Clean dollar rounding: nearest $1 for bankrolls >= $200, nearest $0.50 for smaller bankrolls
        if bankroll >= 200.0:
            stake = round(max(1.0, round(raw_stake)), 2) if raw_stake >= 0.75 else 0.0
        else:
            stake = round(max(0.50, round(raw_stake * 2.0) / 2.0), 2) if raw_stake >= 0.35 else 0.0
        stake = min(stake, round(bankroll * max_bet_pct, 2))

    ev_dollars = round(stake * ev_pct, 2)
    pot_profit = round(stake * b, 2)

    return SportsbookEvaluation(
        book=book,
        side=side,
        american_odds=am_raw,
        effective_american_odds=eff_am,
        decimal_odds=round(dec, 4),
        implied_prob=imp_p,
        model_prob=p_true,
        conservative_prob=p_cons,
        fair_american_odds=fair_am,
        max_acceptable_odds=max_acc_am,
        prob_edge=prob_edge,
        ev_pct=ev_pct,
        ev_dollars=ev_dollars,
        profit_boost_pct=round(boost, 4),
        kelly_fraction_applied=kelly_applied,
        recommended_stake=stake,
        potential_profit=pot_profit,
        confidence_score=conf_score,
        confidence_tier=conf_tier,
    )


@dataclass(frozen=True)
class ArbitrageLeg:
    book: VenueName
    label: str
    side: str
    american_odds: int
    decimal_odds: float
    implied_prob: float
    exact_stake: float
    rounded_stake: float
    payout_if_wins_exact: float
    payout_if_wins_rounded: float
    is_dkex: bool = False
    dkex_ask_price: float | None = None
    dkex_fee: float = 0.0
    dkex_contracts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArbitrageOpportunity:
    arb_type: Literal["DK_FD", "DKEX_CROSS"]
    implied_sum: float
    exact_roi_pct: float
    exact_total_stake: float
    exact_guaranteed_profit: float
    rounding_step: float
    rounded_total_stake: float
    min_guaranteed_profit: float
    max_profit: float
    min_roi_pct: float
    is_guaranteed_after_rounding: bool
    is_cross_number_freeroll: bool
    leg_a: ArbitrageLeg
    leg_b: ArbitrageLeg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def evaluate_arbitrage_pair(
    *,
    book_a: VenueName,
    label_a: str,
    side_a: str,
    odds_a: float | None = None,
    dkex_ask_a: float | None = None,
    dkex_fee_a: float = 0.015,
    book_b: VenueName,
    label_b: str,
    side_b: str,
    odds_b: float | None = None,
    dkex_ask_b: float | None = None,
    dkex_fee_b: float = 0.015,
    total_stake: float = 100.0,
    rounding_step: float = 1.0,
    min_arb_roi: float = 0.001,
    is_cross_number_freeroll: bool = False,
) -> ArbitrageOpportunity | None:
    """
    Evaluate a two-way opposing outcome pair for risk-free arbitrage.
    Supports DK <-> FD (American odds on both legs) and DKeX <-> DK/FD
    (fee-adjusted binary contract on one leg + American odds on the other).
    """
    is_dkex_a = book_a == "DKEX" or dkex_ask_a is not None
    is_dkex_b = book_b == "DKEX" or dkex_ask_b is not None

    if is_dkex_a:
        if dkex_ask_a is None:
            return None
        dec_a = dkex_ask_to_effective_decimal(dkex_ask_a, dkex_fee_a)
        am_a = decimal_to_american(dec_a)
    else:
        if odds_a is None:
            return None
        dec_a = american_to_decimal(odds_a)
        am_a = int(round(float(odds_a)))

    if is_dkex_b:
        if dkex_ask_b is None:
            return None
        dec_b = dkex_ask_to_effective_decimal(dkex_ask_b, dkex_fee_b)
        am_b = decimal_to_american(dec_b)
    else:
        if odds_b is None:
            return None
        dec_b = american_to_decimal(odds_b)
        am_b = int(round(float(odds_b)))

    imp_a = 1.0 / dec_a
    imp_b = 1.0 / dec_b
    implied_sum = imp_a + imp_b
    if implied_sum >= 1.0:
        return None

    exact_roi = (1.0 / implied_sum) - 1.0
    if exact_roi < min_arb_roi:
        return None

    stake_total = max(1.0, float(total_stake))
    exact_stake_a = stake_total * (imp_a / implied_sum)
    exact_stake_b = stake_total * (imp_b / implied_sum)
    exact_payout = stake_total / implied_sum
    exact_profit = exact_payout - stake_total

    # Anti-limiting stake rounding (and whole integer contracts for DKeX)
    contracts_a: int | None = None
    contracts_b: int | None = None

    if is_dkex_a and dkex_ask_a is not None:
        unit_cost_a = float(dkex_ask_a) + max(0.0, float(dkex_fee_a))
        contracts_a = max(1, int(round(exact_stake_a / unit_cost_a)))
        rounded_a = round(contracts_a * unit_cost_a, 2)
        payout_a_rounded = round(contracts_a * 1.0, 2)
    else:
        rounded_a = _round_stake_to_step(exact_stake_a, rounding_step)
        payout_a_rounded = round(rounded_a * dec_a, 2)

    if is_dkex_b and dkex_ask_b is not None:
        unit_cost_b = float(dkex_ask_b) + max(0.0, float(dkex_fee_b))
        contracts_b = max(1, int(round(exact_stake_b / unit_cost_b)))
        rounded_b = round(contracts_b * unit_cost_b, 2)
        payout_b_rounded = round(contracts_b * 1.0, 2)
    else:
        rounded_b = _round_stake_to_step(exact_stake_b, rounding_step)
        payout_b_rounded = round(rounded_b * dec_b, 2)

    rounded_total = round(rounded_a + rounded_b, 2)
    profit_if_a = round(payout_a_rounded - rounded_total, 2)
    profit_if_b = round(payout_b_rounded - rounded_total, 2)
    min_profit = min(profit_if_a, profit_if_b)
    max_profit = max(profit_if_a, profit_if_b)
    min_roi = round(min_profit / rounded_total, 5) if rounded_total > 0 else 0.0

    arb_type: Literal["DK_FD", "DKEX_CROSS"] = (
        "DKEX_CROSS" if (is_dkex_a or is_dkex_b) else "DK_FD"
    )

    return ArbitrageOpportunity(
        arb_type=arb_type,
        implied_sum=round(implied_sum, 5),
        exact_roi_pct=round(exact_roi, 5),
        exact_total_stake=round(stake_total, 2),
        exact_guaranteed_profit=round(exact_profit, 2),
        rounding_step=round(max(0.0, float(rounding_step)), 2),
        rounded_total_stake=rounded_total,
        min_guaranteed_profit=min_profit,
        max_profit=max_profit,
        min_roi_pct=min_roi,
        is_guaranteed_after_rounding=(min_profit > 0.0),
        is_cross_number_freeroll=is_cross_number_freeroll,
        leg_a=ArbitrageLeg(
            book=book_a,
            label=label_a,
            side=side_a,
            american_odds=am_a,
            decimal_odds=round(dec_a, 4),
            implied_prob=round(imp_a, 4),
            exact_stake=round(exact_stake_a, 2),
            rounded_stake=rounded_a,
            payout_if_wins_exact=round(exact_payout, 2),
            payout_if_wins_rounded=payout_a_rounded,
            is_dkex=is_dkex_a,
            dkex_ask_price=round(float(dkex_ask_a), 3) if dkex_ask_a is not None else None,
            dkex_fee=round(float(dkex_fee_a), 4) if is_dkex_a else 0.0,
            dkex_contracts=contracts_a,
        ),
        leg_b=ArbitrageLeg(
            book=book_b,
            label=label_b,
            side=side_b,
            american_odds=am_b,
            decimal_odds=round(dec_b, 4),
            implied_prob=round(imp_b, 4),
            exact_stake=round(exact_stake_b, 2),
            rounded_stake=rounded_b,
            payout_if_wins_exact=round(exact_payout, 2),
            payout_if_wins_rounded=payout_b_rounded,
            is_dkex=is_dkex_b,
            dkex_ask_price=round(float(dkex_ask_b), 3) if dkex_ask_b is not None else None,
            dkex_fee=round(float(dkex_fee_b), 4) if is_dkex_b else 0.0,
            dkex_contracts=contracts_b,
        ),
    )


@dataclass(frozen=True)
class MiddleOpportunity:
    book_a: VenueName
    label_a: str
    side_a: str
    line_a: float
    odds_a: int
    stake_a: float
    book_b: VenueName
    label_b: str
    side_b: str
    line_b: float
    odds_b: int
    stake_b: float
    total_stake: float
    middle_window_label: str
    middle_width: float
    key_numbers_crossed: list[int] = field(default_factory=list)
    p_both_win: float = 0.0
    p_push_a_win_b: float = 0.0
    p_push_b_win_a: float = 0.0
    p_miss: float = 1.0
    breakeven_middle_prob: float = 0.0
    miss_cost: float = 0.0
    double_win_profit: float = 0.0
    polish_win_profit: float = 0.0
    expected_value: float = 0.0
    ev_roi_pct: float = 0.0
    is_positive_ev: bool = False
    is_free_roll_arb: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_middle_opportunity(
    *,
    book_a: VenueName,
    label_a: str,
    side_a: str,
    line_a: float,
    odds_a: float,
    book_b: VenueName,
    label_b: str,
    side_b: str,
    line_b: float,
    odds_b: float,
    middle_window_label: str,
    middle_width: float,
    p_both_win: float,
    p_push_a_win_b: float = 0.0,
    p_push_b_win_a: float = 0.0,
    key_numbers_crossed: list[int] | None = None,
    total_stake: float = 100.0,
) -> MiddleOpportunity:
    """
    Evaluate a cross-book Middle or Polish Middle (e.g., DK Home -2.5 & FD Away +3.5,
    or DK Over 225.5 & FD Under 227.5).

    Stakes are sized for equal single-win return (minimizing miss variance):
      stake_a * dec_a == stake_b * dec_b
    """
    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    imp_a = 1.0 / dec_a
    imp_b = 1.0 / dec_b
    inv_sum = imp_a + imp_b

    tot = max(1.0, float(total_stake))
    stake_a = round(tot * (imp_a / inv_sum), 2)
    stake_b = round(tot - stake_a, 2)

    payout_a = stake_a * dec_a
    payout_b = stake_b * dec_b

    # Outcome payoffs (net profit):
    # 1. Both win (lands strictly inside middle window)
    profit_both = (payout_a + payout_b) - tot
    # 2. Leg A pushes (stake_a refunded) & Leg B wins
    profit_push_a = (stake_a + payout_b) - tot
    # 3. Leg B pushes (stake_b refunded) & Leg A wins
    profit_push_b = (payout_a + stake_b) - tot
    # 4. Miss (one leg wins, other loses)
    profit_miss_a = payout_a - tot
    profit_miss_b = payout_b - tot
    avg_miss_pnl = 0.5 * (profit_miss_a + profit_miss_b)
    miss_cost = max(0.0, -avg_miss_pnl)

    p_bw = max(0.0, min(1.0, float(p_both_win)))
    p_pa = max(0.0, min(1.0 - p_bw, float(p_push_a_win_b)))
    p_pb = max(0.0, min(1.0 - p_bw - p_pa, float(p_push_b_win_a)))
    p_miss = max(0.0, 1.0 - p_bw - p_pa - p_pb)

    ev = (
        p_bw * profit_both
        + p_pa * profit_push_a
        + p_pb * profit_push_b
        + p_miss * avg_miss_pnl
    )
    ev_roi = ev / tot if tot > 0 else 0.0

    # Pure break-even middle probability (ignoring polish push bonus)
    denom = profit_both - avg_miss_pnl
    be_prob = max(0.0, (-avg_miss_pnl / denom)) if denom > 1e-6 else 1.0

    polish_profit = max(profit_push_a, profit_push_b) if (p_pa > 0 or p_pb > 0) else 0.0

    return MiddleOpportunity(
        book_a=book_a,
        label_a=label_a,
        side_a=side_a,
        line_a=round(float(line_a), 2),
        odds_a=int(round(float(odds_a))),
        stake_a=stake_a,
        book_b=book_b,
        label_b=label_b,
        side_b=side_b,
        line_b=round(float(line_b), 2),
        odds_b=int(round(float(odds_b))),
        stake_b=stake_b,
        total_stake=round(tot, 2),
        middle_window_label=middle_window_label,
        middle_width=round(float(middle_width), 2),
        key_numbers_crossed=list(key_numbers_crossed or []),
        p_both_win=round(p_bw, 4),
        p_push_a_win_b=round(p_pa, 4),
        p_push_b_win_a=round(p_pb, 4),
        p_miss=round(p_miss, 4),
        breakeven_middle_prob=round(be_prob, 4),
        miss_cost=round(miss_cost, 2),
        double_win_profit=round(profit_both, 2),
        polish_win_profit=round(polish_profit, 2),
        expected_value=round(ev, 2),
        ev_roi_pct=round(ev_roi, 4),
        is_positive_ev=(ev > 0.0),
        is_free_roll_arb=(avg_miss_pnl >= 0.0),
    )


@dataclass(frozen=True)
class BonusBetConversion:
    bonus_book: VenueName
    bonus_label: str
    bonus_side: str
    bonus_odds: int
    hedge_book: VenueName
    hedge_label: str
    hedge_side: str
    hedge_odds: int
    bonus_amount: float
    hedge_stake: float
    guaranteed_cash: float
    retention_pct: float
    two_way_hold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_bonus_bet_conversion(
    *,
    book_a: VenueName,
    label_a: str,
    side_a: str,
    odds_a: float,
    book_b: VenueName,
    label_b: str,
    side_b: str,
    odds_b: float,
    bonus_amount: float = 50.0,
) -> BonusBetConversion:
    """
    Compute optimal Bonus Bet / Free Bet conversion across DK & FD.
    Bonus bets return PROFIT ONLY (`B * (d_bonus - 1)`), so placing the bonus bet on
    the longer-odds underdog and hedging cash on the favorite maximizes retention:
      hedge_stake = B * (d_bonus - 1) / d_hedge
      guaranteed_cash = B * (d_bonus - 1) * (1 - 1 / d_hedge)
    Automatically selects whichever orientation yields higher cash retention.
    """
    amt = max(1.0, float(bonus_amount))
    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    hold = compute_two_way_hold(odds_a, odds_b)

    # Orientation 1: Bonus on A, Cash hedge on B
    ret_a = (dec_a - 1.0) * (1.0 - (1.0 / dec_b))
    # Orientation 2: Bonus on B, Cash hedge on A
    ret_b = (dec_b - 1.0) * (1.0 - (1.0 / dec_a))

    if ret_a >= ret_b:
        hedge_stake = amt * (dec_a - 1.0) / dec_b
        cash_out = amt * ret_a
        return BonusBetConversion(
            bonus_book=book_a,
            bonus_label=label_a,
            bonus_side=side_a,
            bonus_odds=int(round(float(odds_a))),
            hedge_book=book_b,
            hedge_label=label_b,
            hedge_side=side_b,
            hedge_odds=int(round(float(odds_b))),
            bonus_amount=round(amt, 2),
            hedge_stake=round(hedge_stake, 2),
            guaranteed_cash=round(cash_out, 2),
            retention_pct=round(ret_a, 4),
            two_way_hold=hold,
        )

    hedge_stake = amt * (dec_b - 1.0) / dec_a
    cash_out = amt * ret_b
    return BonusBetConversion(
        bonus_book=book_b,
        bonus_label=label_b,
        bonus_side=side_b,
        bonus_odds=int(round(float(odds_b))),
        hedge_book=book_a,
        hedge_label=label_a,
        hedge_side=side_a,
        hedge_odds=int(round(float(odds_a))),
        bonus_amount=round(amt, 2),
        hedge_stake=round(hedge_stake, 2),
        guaranteed_cash=round(cash_out, 2),
        retention_pct=round(ret_b, 4),
        two_way_hold=hold,
    )


@dataclass(frozen=True)
class SteamLagAlert:
    stale_book: VenueName
    recommended_side: str
    recommended_label: str
    stale_odds: int
    stale_line: float | None
    sharp_prob: float
    stale_prob: float
    prob_lag: float
    line_lag: float | None
    ev_vs_sharp_pct: float
    sharp_source: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_steam_lag(
    *,
    book: VenueName,
    side_label: str,
    side: str,
    book_odds: float,
    book_devig_prob: float | None,
    sharp_prob: float,
    book_line: float | None = None,
    sharp_line: float | None = None,
    sharp_source: str = "Sharp Consensus",
    min_prob_lag: float = 0.025,
    min_line_lag: float = 0.5,
    min_ev_vs_sharp: float = 0.015,
    min_sharp_prob: float = 0.35,
) -> SteamLagAlert | None:
    """
    Flag when DraftKings or FanDuel is lagging behind sharp-only consensus on either
    price (implied probability gap >= min_prob_lag) or line (>= min_line_lag), while
    requiring genuine post-vig +EV against the sharp anchor.
    """
    dec = american_to_decimal(book_odds)
    raw_imp = 1.0 / dec
    p_sharp = clamp_prob(sharp_prob)
    ev_vs_sharp = p_sharp * dec - 1.0

    if p_sharp < min_sharp_prob:
        return None

    ref_book_prob = clamp_prob(book_devig_prob) if book_devig_prob is not None else raw_imp
    prob_lag = p_sharp - ref_book_prob
    line_lag = (
        round(abs(float(book_line) - float(sharp_line)), 2)
        if (book_line is not None and sharp_line is not None)
        else None
    )

    triggered_price = prob_lag >= min_prob_lag and ev_vs_sharp >= min_ev_vs_sharp
    triggered_line = (
        line_lag is not None
        and line_lag >= min_line_lag
        and ev_vs_sharp >= min_ev_vs_sharp
    )

    if not (triggered_price or triggered_line):
        return None

    sharp_fair_am = prob_to_american_odds(p_sharp)
    book_disp = "DraftKings" if book == "DRAFTKINGS" else "FanDuel"
    if triggered_line and book_line is not None and sharp_line is not None:
        summary = (
            f"{book_disp} stale line {book_line:g} ({format_american(book_odds)}) vs "
            f"{sharp_source} {sharp_line:g} (+{ev_vs_sharp*100:.1f}% vs sharp)"
        )
    else:
        summary = (
            f"{book_disp} lagging {sharp_source} by +{prob_lag*100:.1f}% "
            f"({format_american(book_odds)} vs fair {format_american(sharp_fair_am)})"
        )

    return SteamLagAlert(
        stale_book=book,
        recommended_side=side,
        recommended_label=side_label,
        stale_odds=int(round(float(book_odds))),
        stale_line=book_line,
        sharp_prob=round(p_sharp, 4),
        stale_prob=round(ref_book_prob, 4),
        prob_lag=round(prob_lag, 4),
        line_lag=line_lag,
        ev_vs_sharp_pct=round(ev_vs_sharp, 4),
        sharp_source=sharp_source,
        summary=summary,
    )

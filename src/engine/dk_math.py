"""DraftKings Predictions (DKeX) event-contract pricing, fee, and Kelly math."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

Side = Literal["YES", "NO"]
Tier = Literal["A+", "A", "B", "PASS"]


def clamp_prob(p: float, eps: float = 1e-4) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def logit(p: float) -> float:
    p_c = clamp_prob(p)
    return math.log(p_c / (1.0 - p_c))


def inv_logit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def american_to_implied_prob(american_odds: float) -> float:
    """Convert standard American odds (-150, +130) to raw implied probability."""
    if american_odds == 0:
        raise ValueError("American odds cannot be 0.")
    if american_odds < 0:
        return (-american_odds) / ((-american_odds) + 100.0)
    return 100.0 / (american_odds + 100.0)


def prob_to_american_odds(prob: float) -> int:
    """Convert probability (0, 1) or DKeX contract price ($0.01-$0.99) to American odds."""
    p = clamp_prob(prob, 0.005)
    if abs(p - 0.5) < 1e-6:
        return 100
    if p > 0.5:
        return -int(round((p / (1.0 - p)) * 100.0))
    return int(round(((1.0 - p) / p) * 100.0))


def shrink_probability(
    model_prob: float,
    anchor_prob: float | None = None,
    uncertainty: float = 0.12,
) -> float:
    """Shrink probability slightly toward anchor (or 0.50) for conservative Kelly sizing."""
    target = anchor_prob if anchor_prob is not None else 0.50
    u = max(0.0, min(0.50, uncertainty))
    return clamp_prob((1.0 - u) * model_prob + u * target)


def compute_confidence_score(
    *,
    win_prob: float,
    net_edge: float,
    agreement_score: float = 0.80,
    injury_clarity: float = 1.0,
    min_win_prob: float = 0.60,
    min_net_edge: float = 0.025,
) -> tuple[int, Tier]:
    """Compute a 0-100 composite confidence score and tier (A+, A, B, PASS)."""
    # Score components:
    # 1. Win probability strength (0 - 45 pts): 50% -> 10, 60% -> 26, 70% -> 38, 80% -> 45
    prob_pts = max(0.0, min(45.0, (win_prob - 0.45) / 0.35 * 45.0))
    # 2. Fee-adjusted net edge (0 - 35 pts): 0% -> 0, 2.5% -> 16, 5% -> 28, 8%+ -> 35
    edge_pts = max(0.0, min(35.0, (net_edge / 0.075) * 35.0))
    # 3. Model-market agreement & sample stability (0 - 12 pts)
    agree_pts = max(0.0, min(12.0, agreement_score * 12.0))
    # 4. Injury report clarity (0 - 8 pts)
    inj_pts = max(0.0, min(8.0, injury_clarity * 8.0))

    raw_score = int(round(prob_pts + edge_pts + agree_pts + inj_pts))
    score = max(0, min(100, raw_score))

    if win_prob < min_win_prob or net_edge < min_net_edge:
        return min(score, 59), "PASS"

    if score >= 78 and win_prob >= 0.65 and net_edge >= 0.040:
        return max(score, 85), "A+"
    if score >= 70 and win_prob >= 0.62 and net_edge >= 0.030:
        return max(score, 72), "A"
    return max(score, 60), "B"


@dataclass(frozen=True)
class ContractEvaluation:
    side: Side
    contract_price: float
    fee_per_contract: float
    all_in_cost: float
    breakeven_prob: float
    model_prob: float
    conservative_prob: float
    net_edge: float
    ev_per_contract: float
    roi_pct: float
    fair_entry_max_price: float
    fair_midgame_exit_price: float
    kelly_fraction_applied: float
    recommended_contracts: int
    recommended_outlay: float
    american_equivalent: int
    confidence_score: int
    confidence_tier: Tier

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_single_side(
    *,
    side: Side,
    side_true_prob: float,
    contract_price: float,
    fee_per_contract: float = 0.015,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.04,
    min_win_prob: float = 0.60,
    min_net_edge: float = 0.025,
    anchor_prob: float | None = None,
    uncertainty: float = 0.12,
    injury_clarity: float = 1.0,
) -> ContractEvaluation:
    """Evaluate a specific YES or NO DraftKings Predictions binary event contract ($1 payout)."""
    price = round(max(0.01, min(0.99, float(contract_price))), 4)
    fee = round(max(0.0, float(fee_per_contract)), 4)
    all_in_cost = round(price + fee, 4)
    p_true = round(clamp_prob(side_true_prob), 4)

    # Conservative probability shrunk toward anchor/50% for position sizing
    p_cons = round(
        shrink_probability(p_true, anchor_prob=anchor_prob, uncertainty=uncertainty), 4
    )

    # On a $1.00 binary contract, break-even win probability equals total cost basis
    breakeven_prob = round(min(1.0, all_in_cost), 4)
    net_edge = round(p_true - all_in_cost, 4)
    ev_per_contract = round(p_true * 1.0 - all_in_cost, 4)
    roi_pct = round(ev_per_contract / all_in_cost, 4) if all_in_cost > 0 else 0.0

    # Highest ask price where trade still clears min_net_edge + fee
    fair_max_entry = round(max(0.01, min(0.98, p_true - fee - min_net_edge)), 3)
    # Mid-game / pre-settlement target exit price where selling nets full fair EV after sell fee
    fair_exit = round(max(0.01, min(0.99, min(0.99, p_true + fee * 0.5))), 3)

    # Agreement score between model and market anchor
    if anchor_prob is not None:
        gap = abs(p_true - anchor_prob)
        # Sweet spot: moderate edge (2-9%), penalize extreme >18% divergence as potential stale news
        agreement = max(0.25, 1.0 - max(0.0, gap - 0.05) * 3.5)
    else:
        agreement = 0.72

    conf_score, conf_tier = compute_confidence_score(
        win_prob=p_true,
        net_edge=net_edge,
        agreement_score=agreement,
        injury_clarity=injury_clarity,
        min_win_prob=min_win_prob,
        min_net_edge=min_net_edge,
    )

    # Fractional Kelly position sizing using conservative probability:
    # Full Kelly on $1 binary contract costing C: f* = (p_cons - C) / (1 - C)
    if all_in_cost >= 0.995 or p_cons <= all_in_cost or conf_tier == "PASS":
        kelly_applied = 0.0
        contracts = 0
        outlay = 0.0
    else:
        full_kelly = (p_cons - all_in_cost) / (1.0 - all_in_cost)
        scaled_kelly = max(0.0, full_kelly * kelly_multiplier)
        kelly_applied = round(min(max_bet_pct, scaled_kelly), 5)
        target_dollars = bankroll * kelly_applied
        contracts = int(math.floor(target_dollars / all_in_cost))
        # For smaller bankrolls (e.g. $20-$75), ensure a qualifying A+/A/B play sizes at least
        # 1 whole contract ($0.50-$0.75) as long as 1 contract is <= 10% of bankroll.
        if contracts == 0 and bankroll >= all_in_cost and all_in_cost <= bankroll * 0.10:
            contracts = 1
        outlay = round(contracts * all_in_cost, 2)

    return ContractEvaluation(
        side=side,
        contract_price=price,
        fee_per_contract=fee,
        all_in_cost=all_in_cost,
        breakeven_prob=breakeven_prob,
        model_prob=p_true,
        conservative_prob=p_cons,
        net_edge=net_edge,
        ev_per_contract=ev_per_contract,
        roi_pct=roi_pct,
        fair_entry_max_price=fair_max_entry,
        fair_midgame_exit_price=fair_exit,
        kelly_fraction_applied=kelly_applied,
        recommended_contracts=contracts,
        recommended_outlay=outlay,
        american_equivalent=prob_to_american_odds(price),
        confidence_score=conf_score,
        confidence_tier=conf_tier,
    )


def evaluate_two_way_contract(
    *,
    yes_true_prob: float,
    yes_ask_price: float,
    no_ask_price: float | None = None,
    exchange_spread: float = 0.01,
    fee_per_contract: float = 0.015,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bet_pct: float = 0.04,
    min_win_prob: float = 0.60,
    min_net_edge: float = 0.025,
    yes_anchor_prob: float | None = None,
    uncertainty: float = 0.12,
    injury_clarity: float = 1.0,
) -> tuple[ContractEvaluation, ContractEvaluation, ContractEvaluation]:
    """
    Evaluate both YES and NO sides of a DraftKings Predictions event contract and return
    (best_evaluation, yes_evaluation, no_evaluation).
    """
    p_yes = clamp_prob(yes_true_prob)
    p_no = 1.0 - p_yes

    # On DKeX, if only the reference midpoint/ask of YES is known, NO ask typically sits at
    # (1 - yes_mid) + half-spread; if explicit no_ask_price is supplied, use it directly.
    inferred_no_ask = (
        no_ask_price
        if no_ask_price is not None
        else min(0.99, max(0.01, round(1.0 - yes_ask_price + exchange_spread, 3)))
    )

    yes_eval = evaluate_single_side(
        side="YES",
        side_true_prob=p_yes,
        contract_price=yes_ask_price,
        fee_per_contract=fee_per_contract,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        max_bet_pct=max_bet_pct,
        min_win_prob=min_win_prob,
        min_net_edge=min_net_edge,
        anchor_prob=yes_anchor_prob,
        uncertainty=uncertainty,
        injury_clarity=injury_clarity,
    )

    no_anchor = (1.0 - yes_anchor_prob) if yes_anchor_prob is not None else None
    no_eval = evaluate_single_side(
        side="NO",
        side_true_prob=p_no,
        contract_price=inferred_no_ask,
        fee_per_contract=fee_per_contract,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        max_bet_pct=max_bet_pct,
        min_win_prob=min_win_prob,
        min_net_edge=min_net_edge,
        anchor_prob=no_anchor,
        uncertainty=uncertainty,
        injury_clarity=injury_clarity,
    )

    # Prefer a non-PASS side first; tie-break by confidence_score then net_edge
    if yes_eval.confidence_tier != "PASS" and no_eval.confidence_tier == "PASS":
        best = yes_eval
    elif no_eval.confidence_tier != "PASS" and yes_eval.confidence_tier == "PASS":
        best = no_eval
    elif (yes_eval.confidence_score, yes_eval.net_edge, yes_eval.model_prob) >= (
        no_eval.confidence_score,
        no_eval.net_edge,
        no_eval.model_prob,
    ):
        best = yes_eval
    else:
        best = no_eval

    return best, yes_eval, no_eval

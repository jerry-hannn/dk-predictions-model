"""Unit tests for DraftKings Predictions (DKeX) contract math, fees, and Kelly sizing."""

from src.engine.dk_math import (
    american_to_implied_prob,
    evaluate_single_side,
    evaluate_two_way_contract,
    prob_to_american_odds,
)


def test_american_odds_conversions() -> None:
    assert abs(american_to_implied_prob(-150) - 0.60) < 1e-6
    assert abs(american_to_implied_prob(150) - 0.40) < 1e-6
    assert prob_to_american_odds(0.60) == -150
    assert prob_to_american_odds(0.40) == 150


def test_fee_adjusted_breakeven_and_ev() -> None:
    # 60c contract + 1.5c fee = 61.5c cost basis -> 61.5% break-even win rate
    ev = evaluate_single_side(
        side="YES",
        side_true_prob=0.685,
        contract_price=0.60,
        fee_per_contract=0.015,
        bankroll=1000.0,
    )
    assert abs(ev.all_in_cost - 0.615) < 1e-6
    assert abs(ev.breakeven_prob - 0.615) < 1e-6
    assert abs(ev.net_edge - 0.070) < 1e-4
    assert ev.confidence_tier == "A+"
    assert ev.recommended_contracts > 0
    assert ev.recommended_outlay <= 40.01  # Respects 4% max single-bet cap on $1,000


def test_fee_wipes_out_marginal_favorite() -> None:
    # 62c contract with 63% true prob looks +EV before fees, but loses after 1.5c fee (cost 63.5c)
    ev_with_fee = evaluate_single_side(
        side="YES",
        side_true_prob=0.63,
        contract_price=0.62,
        fee_per_contract=0.015,
        bankroll=1000.0,
    )
    assert ev_with_fee.net_edge < 0
    assert ev_with_fee.confidence_tier == "PASS"
    assert ev_with_fee.recommended_contracts == 0

    # With 0c Promo Dollars and 59c price, it passes cleanly
    ev_promo = evaluate_single_side(
        side="YES",
        side_true_prob=0.63,
        contract_price=0.59,
        fee_per_contract=0.0,
        bankroll=1000.0,
    )
    assert ev_promo.net_edge >= 0.039
    assert ev_promo.confidence_tier in ("A", "B")
    assert ev_promo.recommended_contracts > 0


def test_two_way_contract_picks_high_prob_no_side() -> None:
    # Over (YES) only has 31% chance, trading at 42c -> NO has 69% chance at 59c!
    best, yes_ev, no_ev = evaluate_two_way_contract(
        yes_true_prob=0.31,
        yes_ask_price=0.42,
        no_ask_price=0.59,
        fee_per_contract=0.015,
        bankroll=1000.0,
    )
    assert yes_ev.confidence_tier == "PASS"
    assert no_ev.confidence_tier == "A+"
    assert best.side == "NO"
    assert abs(best.model_prob - 0.69) < 1e-4


def test_micro_bankroll_25_dollars_sizes_at_least_one_contract() -> None:
    ev = evaluate_single_side(
        side="YES",
        side_true_prob=0.64,
        contract_price=0.58,
        fee_per_contract=0.015,
        bankroll=25.0,
    )
    assert ev.confidence_tier in ("A", "B")
    assert ev.recommended_contracts >= 1
    assert 0.59 <= ev.recommended_outlay <= 2.50

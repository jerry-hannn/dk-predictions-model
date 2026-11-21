"""Unit tests for Traditional DraftKings & FanDuel sportsbook math and 5-strategy engine."""

from __future__ import annotations

import pytest

from src.engine.sportsbook_math import (
    american_to_decimal,
    american_to_profit_per_dollar,
    apply_profit_boost,
    compute_two_way_hold,
    decimal_to_american,
    dkex_ask_to_effective_decimal,
    evaluate_arbitrage_pair,
    evaluate_bonus_bet_conversion,
    evaluate_middle_opportunity,
    evaluate_sportsbook_side,
    evaluate_steam_lag,
)


def test_american_decimal_hold_and_profit_boost_conversions() -> None:
    assert abs(american_to_decimal(-110) - (1.0 + 100.0 / 110.0)) < 1e-6
    assert abs(american_to_decimal(150) - 2.50) < 1e-6
    assert abs(american_to_profit_per_dollar(-200) - 0.50) < 1e-6
    assert decimal_to_american(2.50) == 150
    assert decimal_to_american(1.50) == -200

    # Standard -110/-110 hold is ~4.76%
    hold = compute_two_way_hold(-110, -110)
    assert abs(hold - 0.04762) < 1e-4

    # Profit boost multiplies NET PROFIT portion only:
    # +200 (b=2.0) with +25% boost -> b'=2.50 -> decimal 3.50 (+250)
    assert abs(apply_profit_boost(200, 0.25) - 3.50) < 1e-6
    assert decimal_to_american(apply_profit_boost(200, 0.25)) == 250
    # -200 (b=0.50) with +50% boost -> b'=0.75 -> decimal 1.75 (-133)
    assert abs(apply_profit_boost(-200, 0.50) - 1.75) < 1e-6
    assert decimal_to_american(apply_profit_boost(-200, 0.50)) == -133

    # DKeX ask + fee effective decimal odds: ask=0.45, fee=0.015 -> 1/0.465
    assert abs(dkex_ask_to_effective_decimal(0.45, 0.015) - (1.0 / 0.465)) < 1e-6


def test_ev_line_shopping_kelly_and_sharp_clv_guard() -> None:
    # Strong +EV play aligned with sharp consensus -> qualifies & sizes positive Kelly $
    res = evaluate_sportsbook_side(
        book="FANDUEL",
        side="HOME",
        side_true_prob=0.60,
        american_odds=-110,
        bankroll=1000.0,
        kelly_multiplier=0.25,
        anchor_prob=0.56,
    )
    assert res.confidence_tier in ("A+", "A", "B")
    assert res.ev_pct > 0.10
    assert res.recommended_stake > 0.0
    assert res.potential_profit > 0.0

    # Sharp-alignment CLV guard: if sharp consensus implies 48% while book charges -125 (55.6%),
    # even a high unboosted model prob should be blocked as PASS (avoid beating negative CLV).
    blocked = evaluate_sportsbook_side(
        book="DRAFTKINGS",
        side="HOME",
        side_true_prob=0.62,
        american_odds=-125,
        bankroll=1000.0,
        anchor_prob=0.48,
        profit_boost_pct=0.0,
    )
    assert blocked.confidence_tier == "PASS"
    assert blocked.recommended_stake == 0.0

    # With a +50% Profit Boost token, even a fair -110 line becomes strongly +EV
    boosted = evaluate_sportsbook_side(
        book="DRAFTKINGS",
        side="OVER",
        side_true_prob=0.50,
        american_odds=-110,
        bankroll=1000.0,
        anchor_prob=0.50,
        profit_boost_pct=0.50,
    )
    assert boosted.effective_american_odds > 100
    assert boosted.ev_pct > 0.15
    assert boosted.confidence_tier in ("A+", "A")
    assert boosted.recommended_stake > 0.0


def test_arbitrage_dk_fd_rounding_safety_and_dkex_cross_arb() -> None:
    # Classic DK +115 vs FD -105 -> implied sum = 0.4651 + 0.5122 = 0.9773 -> ~2.32% ROI
    arb = evaluate_arbitrage_pair(
        book_a="DRAFTKINGS",
        label_a="KC ML",
        side_a="HOME",
        odds_a=115,
        book_b="FANDUEL",
        label_b="BUF ML",
        side_b="AWAY",
        odds_b=-105,
        total_stake=100.0,
        rounding_step=1.0,
        min_arb_roi=0.005,
    )
    assert arb is not None
    assert arb.arb_type == "DK_FD"
    assert arb.implied_sum < 1.0
    assert abs(arb.exact_roi_pct - 0.0232) < 0.002
    assert arb.is_guaranteed_after_rounding is True
    assert arb.min_guaranteed_profit > 0.0
    assert arb.leg_a.rounded_stake % 1.0 == 0.0
    assert arb.leg_b.rounded_stake % 1.0 == 0.0

    # Slim arb (~0.45%) gets wiped out under aggressive $5 rounding -> must flag is_guaranteed_after_rounding=False
    slim = evaluate_arbitrage_pair(
        book_a="DRAFTKINGS",
        label_a="Over 220.5",
        side_a="OVER",
        odds_a=102,
        book_b="FANDUEL",
        label_b="Under 220.5",
        side_b="UNDER",
        odds_b=100,
        total_stake=100.0,
        rounding_step=5.0,
        min_arb_roi=0.001,
    )
    assert slim is not None
    assert slim.exact_guaranteed_profit > 0.0
    # Verify worst-case rounded profit calculation matches legs exactly
    rounded_tot = slim.leg_a.rounded_stake + slim.leg_b.rounded_stake
    min_check = min(
        slim.leg_a.payout_if_wins_rounded - rounded_tot,
        slim.leg_b.payout_if_wins_rounded - rounded_tot,
    )
    assert abs(slim.min_guaranteed_profit - round(min_check, 2)) < 1e-6

    # Tiny arb (0.25%) at $50 total stake with $5 rounding guarantees negative worst-case after rounding
    tiny_unsafe = evaluate_arbitrage_pair(
        book_a="DRAFTKINGS",
        label_a="Side A",
        side_a="HOME",
        odds_a=105,
        book_b="FANDUEL",
        label_b="Side B",
        side_b="AWAY",
        odds_b=-104,
        total_stake=50.0,
        rounding_step=5.0,
        min_arb_roi=0.001,
    )
    assert tiny_unsafe is not None
    assert tiny_unsafe.exact_guaranteed_profit > 0.0
    assert tiny_unsafe.is_guaranteed_after_rounding is False
    assert tiny_unsafe.min_guaranteed_profit <= 0.0

    # DKeX <-> FanDuel cross-venue arb (DKeX YES ask=0.42 + 0.015 fee = 0.435 vs FD -110 -> 0.5238; sum = 0.9588)
    cross = evaluate_arbitrage_pair(
        book_a="DKEX",
        label_a="BOS YES @ 42¢",
        side_a="YES",
        dkex_ask_a=0.42,
        dkex_fee_a=0.015,
        book_b="FANDUEL",
        label_b="NYK ML",
        side_b="AWAY",
        odds_b=-110,
        total_stake=100.0,
        rounding_step=1.0,
        min_arb_roi=0.005,
    )
    assert cross is not None
    assert cross.arb_type == "DKEX_CROSS"
    assert cross.leg_a.is_dkex is True
    assert cross.leg_a.dkex_contracts is not None and cross.leg_a.dkex_contracts > 0
    assert cross.is_guaranteed_after_rounding is True
    assert cross.min_guaranteed_profit > 0.0

    # Non-arb pair (-110 vs -110) returns None
    assert (
        evaluate_arbitrage_pair(
            book_a="DRAFTKINGS",
            label_a="A",
            side_a="HOME",
            odds_a=-110,
            book_b="FANDUEL",
            label_b="B",
            side_b="AWAY",
            odds_b=-110,
        )
        is None
    )


def test_middles_and_polish_middles_key_numbers() -> None:
    # NFL Key Number 3 Middle: DK Home -2.5 (-110) & FD Away +3.5 (-110)
    # Break-even middle probability at -110/-110 is ~4.76%. NFL lands on 3 ~14.5% of games -> strong +EV!
    mid_nfl = evaluate_middle_opportunity(
        book_a="DRAFTKINGS",
        label_a="KC -2.5",
        side_a="HOME",
        line_a=-2.5,
        odds_a=-110,
        book_b="FANDUEL",
        label_b="BUF +3.5",
        side_b="AWAY",
        line_b=3.5,
        odds_b=-110,
        middle_window_label="Home wins by exactly 3",
        middle_width=1.0,
        p_both_win=0.145,
        key_numbers_crossed=[3],
        total_stake=100.0,
    )
    assert mid_nfl.is_positive_ev is True
    assert mid_nfl.key_numbers_crossed == [3]
    assert mid_nfl.breakeven_middle_prob < 0.06
    assert mid_nfl.double_win_profit > 80.0
    assert mid_nfl.miss_cost < 6.0
    assert mid_nfl.ev_roi_pct > 0.08

    # Polish Middle: DK Home -2.5 (-110) & FD Away +3 (-110) -> Margin==3 wins Leg A & pushes Leg B!
    polish = evaluate_middle_opportunity(
        book_a="DRAFTKINGS",
        label_a="KC -2.5",
        side_a="HOME",
        line_a=-2.5,
        odds_a=-110,
        book_b="FANDUEL",
        label_b="BUF +3",
        side_b="AWAY",
        line_b=3.0,
        odds_b=-110,
        middle_window_label="Polish middle on 3",
        middle_width=0.5,
        p_both_win=0.0,
        p_push_b_win_a=0.145,
        key_numbers_crossed=[3],
        total_stake=100.0,
    )
    assert polish.is_positive_ev is True
    assert polish.polish_win_profit > 40.0
    assert polish.expected_value > 0.0

    # Negative-EV blind middle (2% hit rate at -110/-110) -> is_positive_ev == False
    bad_mid = evaluate_middle_opportunity(
        book_a="DRAFTKINGS",
        label_a="Over 225.5",
        side_a="OVER",
        line_a=225.5,
        odds_a=-110,
        book_b="FANDUEL",
        label_b="Under 226.5",
        side_b="UNDER",
        line_b=226.5,
        odds_b=-110,
        middle_window_label="Total == 226",
        middle_width=1.0,
        p_both_win=0.02,
        total_stake=100.0,
    )
    assert bad_mid.is_positive_ev is False
    assert bad_mid.expected_value < 0.0


def test_bonus_bet_conversion_high_retention_and_invariant_payout() -> None:
    # +380 underdog on DK vs -440 favorite on FD with $50 Bonus Bet -> >70% guaranteed retention
    conv = evaluate_bonus_bet_conversion(
        book_a="DRAFTKINGS",
        label_a="Underdog ML",
        side_a="AWAY",
        odds_a=380,
        book_b="FANDUEL",
        label_b="Favorite ML",
        side_b="HOME",
        odds_b=-440,
        bonus_amount=50.0,
    )
    assert conv.bonus_book == "DRAFTKINGS"
    assert conv.hedge_book == "FANDUEL"
    assert conv.retention_pct >= 0.70
    assert conv.guaranteed_cash >= 35.0

    # Verify both win outcomes yield essentially identical net cash profit (within $0.02 rounding)
    dec_bonus = american_to_decimal(conv.bonus_odds)
    dec_hedge = american_to_decimal(conv.hedge_odds)
    net_if_bonus_wins = conv.bonus_amount * (dec_bonus - 1.0) - conv.hedge_stake
    net_if_hedge_wins = conv.hedge_stake * (dec_hedge - 1.0)
    assert abs(net_if_bonus_wins - conv.guaranteed_cash) <= 0.03
    assert abs(net_if_hedge_wins - conv.guaranteed_cash) <= 0.03

    # Passing reversed order (favorite as A, underdog as B) must auto-select underdog for Bonus Bet
    conv_rev = evaluate_bonus_bet_conversion(
        book_a="FANDUEL",
        label_a="Favorite ML",
        side_a="HOME",
        odds_a=-440,
        book_b="DRAFTKINGS",
        label_b="Underdog ML",
        side_b="AWAY",
        odds_b=380,
        bonus_amount=50.0,
    )
    assert conv_rev.bonus_book == "DRAFTKINGS"
    assert abs(conv_rev.retention_pct - conv.retention_pct) < 1e-6


def test_steam_lag_scanner_detects_stale_book_and_ignores_fair_lines() -> None:
    # Sharp consensus moved to 60% (-150 fair), while FanDuel still sits at -115 (~51.5% devigged)
    alert = evaluate_steam_lag(
        book="FANDUEL",
        side_label="BOS ML",
        side="HOME",
        book_odds=-115,
        book_devig_prob=0.515,
        sharp_prob=0.60,
        sharp_source="Sharp Consensus (2 books)",
    )
    assert alert is not None
    assert alert.stale_book == "FANDUEL"
    assert alert.prob_lag >= 0.08
    assert alert.ev_vs_sharp_pct > 0.10

    # Fair aligned line should not trigger alert
    none_alert = evaluate_steam_lag(
        book="DRAFTKINGS",
        side_label="BOS ML",
        side="HOME",
        book_odds=-110,
        book_devig_prob=0.50,
        sharp_prob=0.51,
    )
    assert none_alert is None

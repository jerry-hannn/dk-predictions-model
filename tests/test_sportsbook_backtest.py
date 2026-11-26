"""Walk-forward strategy verification & profitability tests for Traditional DK/FD strategies."""

from __future__ import annotations

from src.backtest.sportsbook_backtest import run_sportsbook_backtest


def test_walk_forward_traditional_sportsbook_strategies_and_invariants() -> None:
    """
    Execute the full zero-lookahead walk-forward backtest across historical NFL & NBA games
    and verify profitability, CLV, rounding safety invariants, and 3-regime stress resilience.
    """
    res = run_sportsbook_backtest(
        sport="all",
        regime="standard",
        arb_rounding=1.0,
        include_dkex_cross=True,
        save_artifact=False,
    )

    # 1. Strict chronological walk-forward integrity (zero future leakage)
    assert res["mode"] == "traditional_sportsbooks"
    assert res["chronological_verified"] is True
    assert res["by_league"]["nfl"]["chronological_verified"] is True
    assert res["by_league"]["nba"]["chronological_verified"] is True

    combined = res["combined"]
    assert combined["oos_games"] >= 700
    assert combined["combined_net_pnl"] > 0.0
    assert combined["combined_roi_pct"] > 0.05

    # 2. Arbitrage invariants: risk-free profit with zero drawdown even after $1 rounding
    arb = combined["arbitrage"]
    assert arb["arbs_executed_rounded"] > 0
    assert arb["dkex_cross_arbs_executed"] > 0
    assert arb["realized_roi_pct"] > 0.0
    assert arb["min_trade_pnl"] >= 0.0
    assert arb["max_drawdown_pct"] == 0.0
    assert combined["verdicts"]["arbitrage"] == "VERIFIED_PROFITABLE"

    # 3. Bonus Bet Converter & Profit Boosts invariants
    promos = combined["promos_and_boosts"]
    assert promos["bonus_conversions_count"] > 0
    assert promos["bonus_realized_retention_pct"] >= 0.70
    assert promos["bonus_max_outcome_drift"] <= 0.15
    assert promos["low_hold_pairs_found"] > 0
    assert promos["low_hold_savings_vs_vig_pct"] > 0.02
    assert promos["boost_25_roi_pct"] > 0.03
    assert promos["boost_25_roi_lift_pct"] > 0.10
    assert promos["boost_50_roi_pct"] > promos["boost_25_roi_pct"]
    assert combined["verdicts"]["bonus_bet_conversion"] == "VERIFIED_PROFITABLE"
    assert combined["verdicts"]["profit_boost"] == "VERIFIED_PROFITABLE"

    # 4. Steam / Off-Market Lag beats sharp closing consensus with positive ROI
    steam = combined["steam_lag"]
    assert steam["total_bets"] > 20
    assert steam["avg_clv_pct"] > 0.01
    assert steam["roi_pct"] > 0.0
    assert combined["verdicts"]["steam_lag"] == "VERIFIED_PROFITABLE"

    # 5. +EV Line Shopping achieves positive CLV overall and strong NBA Moneyline profitability
    ev = combined["ev_line_shopping"]
    assert ev["total_bets"] > 20
    assert ev["avg_clv_pct"] > 0.0
    assert res["by_league"]["nba"]["ev_line_shopping"]["roi_pct"] > 0.05

    # 6. Middles & Key Numbers improve materially over blind middles
    mid = combined["middles"]
    assert mid["total_middles"] > 0
    assert mid["key_number_middles"] > 0
    assert mid["blind_vs_filtered_roi_lift_pct"] >= 0.0

    # 7. All 3 stress regimes remain portfolio-profitable with positive arb, boost, bonus, and steam returns
    regimes = res["stress_regimes"]
    for key in ("standard", "tight_vig", "active_steam"):
        reg = regimes[key]
        assert reg["combined_net_pnl"] > 0.0
        assert reg["combined_roi_pct"] > 0.0
        assert reg["arb_roi_pct"] > 0.0
        assert reg["bonus_retention_pct"] >= 0.70
        assert reg["boost_25_roi_pct"] > 0.0
        assert reg["steam_roi_pct"] > 0.0
        assert reg["steam_clv_pct"] > 0.0

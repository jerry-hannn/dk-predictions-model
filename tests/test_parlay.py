"""Unit and API integration tests for the Parlay & SGP Builder (Predictions & Sportsbooks)."""

from __future__ import annotations

import math
from pathlib import Path

from fastapi.testclient import TestClient

from src.engine.parlay_math import (
    ParlayLeg,
    compute_joint_probability,
    evaluate_predictions_parlay,
    evaluate_sportsbook_parlay,
    suggest_optimal_parlays,
)
from src.web.app import app


def _pred_leg(
    leg_id: str,
    event_id: str,
    event_name: str,
    market_type: str,
    selection: str,
    *,
    side: str = "YES",
    prob: float = 0.68,
    ask: float = 0.58,
    sport: str = "NBA",
    player_name: str | None = None,
) -> ParlayLeg:
    return ParlayLeg(
        leg_id=leg_id,
        sport=sport,
        event_id=event_id,
        event_name=event_name,
        short_name=event_name,
        market_type=market_type,
        selection_label=selection,
        side=side,
        model_prob=prob,
        contract_price=ask,
        fee_per_contract=0.015,
        player_name=player_name,
    )


def _sb_leg(
    leg_id: str,
    event_id: str,
    event_name: str,
    market_type: str,
    selection: str,
    *,
    side: str = "HOME",
    prob: float = 0.58,
    dk_odds: int = -110,
    fd_odds: int | None = -105,
    book: str = "FANDUEL",
    sport: str = "NBA",
    player_name: str | None = None,
) -> ParlayLeg:
    return ParlayLeg(
        leg_id=leg_id,
        sport=sport,
        event_id=event_id,
        event_name=event_name,
        short_name=event_name,
        market_type=market_type,
        selection_label=selection,
        side=side,
        model_prob=prob,
        book=book,  # type: ignore[arg-type]
        american_odds=fd_odds if (book == "FANDUEL" and fd_odds is not None) else dk_odds,
        dk_odds=dk_odds,
        fd_odds=fd_odds,
        player_name=player_name,
    )


# --- 1. Predictions (DKeX) Parlay Math ---

def test_predictions_combo_vs_sequential_rollover() -> None:
    legs = [
        _pred_leg("l1", "g1", "NYK @ BOS", "Moneyline", "BOS to Win", prob=0.70, ask=0.58),
        _pred_leg("l2", "g2", "LAL @ DEN", "Moneyline", "DEN to Win", prob=0.68, ask=0.56),
    ]

    combo = evaluate_predictions_parlay(
        legs,
        pricing_structure="COMBO_CONTRACT",
        fee_per_contract=0.015,
        bankroll=1000.0,
    )
    rollover = evaluate_predictions_parlay(
        legs,
        pricing_structure="SEQUENTIAL_ROLLOVER",
        fee_per_contract=0.015,
        bankroll=1000.0,
    )

    assert combo["mode"] == "predictions"
    assert combo["is_sgp"] is False
    assert combo["has_conflicts"] is False
    # Independent games: joint prob == exact product
    assert abs(combo["joint_model_prob"] - 0.70 * 0.68) < 1e-4
    assert abs(combo["correlation_impact"]) < 1e-6
    # Implied combo ask = 0.58 * 0.56 = 0.3248, plus single 1.5c fee = 0.3398
    assert abs(combo["combo_ask_price"] - 0.3248) < 1e-4
    assert abs(combo["all_in_cost"] - 0.3398) < 1e-4
    assert combo["net_edge"] > 0.10
    assert combo["confidence_tier"] in ("A+", "A")
    assert combo["recommended_contracts"] > 0

    # Sequential rollover compounds per-leg fees: (0.58+0.015)*(0.56+0.015) = 0.342125
    expected_roll_cost = (0.58 + 0.015) * (0.56 + 0.015)
    assert abs(rollover["all_in_cost"] - expected_roll_cost) < 1e-3
    assert rollover["fee_drag"] > combo["fee_drag"]
    assert rollover["net_edge"] < combo["net_edge"]


def test_predictions_custom_combo_ask_and_parlay_tax_flag() -> None:
    legs = [
        _pred_leg("l1", "g1", "NYK @ BOS", "Moneyline", "BOS to Win", prob=0.72, ask=0.60),
        # Negative-edge leg (prob 0.48 < ask 0.55 + 0.015)
        _pred_leg("l2", "g2", "MIA @ PHI", "Spread", "PHI -3.5 Cover", prob=0.48, ask=0.55),
    ]
    res = evaluate_predictions_parlay(legs, custom_combo_ask=0.40, fee_per_contract=0.015)
    assert res["is_custom_combo_ask"] is True
    assert abs(res["combo_ask_price"] - 0.40) < 1e-4
    assert res["parlay_tax_legs_count"] == 1
    assert res["legs"][1]["is_parlay_tax"] is True
    assert res["legs"][0]["is_parlay_tax"] is False
    # 0.72 * 0.48 = 0.3456 < 0.415 all-in -> negative edge -> PASS
    assert res["net_edge"] < 0
    assert res["confidence_tier"] == "PASS"
    assert res["recommended_contracts"] == 0


# --- 2. Traditional Sportsbooks (DK / FD) Parlay Math ---

def test_sportsbook_parlay_dk_vs_fd_comparison_and_boost() -> None:
    legs = [
        _sb_leg("s1", "g1", "NYK @ BOS", "Moneyline", "BOS ML", prob=0.58, dk_odds=-112, fd_odds=-102),
        _sb_leg("s2", "g2", "LAL @ DEN", "Spread", "DEN -3.5", prob=0.57, dk_odds=-110, fd_odds=-104),
    ]

    base = evaluate_sportsbook_parlay(legs, book="BEST", bankroll=1000.0)
    boosted = evaluate_sportsbook_parlay(legs, book="BEST", profit_boost_pct=0.25, bankroll=1000.0)

    assert base["mode"] == "sportsbook"
    assert base["book"] == "FANDUEL"
    assert "DRAFTKINGS" in base["book_comparison"]
    assert "FANDUEL" in base["book_comparison"]
    assert base["book_comparison"]["FANDUEL"]["ev_pct"] > base["book_comparison"]["DRAFTKINGS"]["ev_pct"]
    assert base["line_shopping_gain_pct"] is not None and base["line_shopping_gain_pct"] > 0.05
    assert base["ev_pct"] > 0.15
    assert base["recommended_stake"] > 0

    # +25% Profit Boost multiplies net profit portion by 1.25x
    expected_boosted_dec = 1.0 + (base["raw_decimal_odds"] - 1.0) * 1.25
    assert abs(boosted["effective_decimal_odds"] - expected_boosted_dec) < 1e-3
    assert boosted["boost_ev_lift"] > 0.15
    assert boosted["ev_pct"] > base["ev_pct"]


def test_sportsbook_bonus_bet_and_custom_sgp_odds() -> None:
    legs = [
        _sb_leg("s1", "g1", "KC @ BUF", "Moneyline", "BUF ML", prob=0.56, dk_odds=-110, fd_odds=-110, sport="NFL"),
        _sb_leg("s2", "g2", "SF @ DAL", "Moneyline", "SF ML", prob=0.56, dk_odds=-110, fd_odds=-110, sport="NFL"),
    ]
    bonus = evaluate_sportsbook_parlay(legs, is_bonus_bet=True, stake_override=50.0)
    # Profit-only payout: potential_payout == potential_profit
    assert bonus["is_bonus_bet"] is True
    assert abs(bonus["potential_payout"] - bonus["potential_profit"]) < 1e-6
    assert bonus["ev_pct"] > 0.70  # High retention on ~+264 underdog parlay

    custom = evaluate_sportsbook_parlay(legs, custom_parlay_odds=320, bankroll=1000.0)
    assert custom["is_custom_odds"] is True
    assert custom["raw_american_odds"] == 320
    assert custom["effective_american_odds"] == 320


# --- 3. Same-Game Correlation (Gaussian Copula) & Conflicts ---

def test_sgp_positive_correlation_and_override() -> None:
    # Same game: BOS ML + BOS -3.5 Cover -> strongly positively correlated
    legs = [
        _sb_leg("g1_ml", "evt_bos", "NYK @ BOS", "Moneyline", "BOS Moneyline", side="HOME", prob=0.65, dk_odds=-160, fd_odds=-155),
        _sb_leg("g1_sp", "evt_bos", "NYK @ BOS", "Spread", "BOS -3.5", side="HOME", prob=0.55, dk_odds=-110, fd_odds=-110),
    ]
    res = evaluate_sportsbook_parlay(legs, custom_parlay_odds=145)
    assert res["is_sgp"] is True
    assert res["has_conflicts"] is False
    assert len(res["correlation_pairs"]) == 1
    assert res["correlation_pairs"][0]["correlation"] > 0.5
    # Positive correlation lifts joint prob strictly above independent product (0.65 * 0.55 = 0.3575)
    assert res["joint_model_prob"] > res["independent_model_prob"] + 0.08
    assert res["joint_model_prob"] <= min(0.65, 0.55)

    # Override correlation to 0.0 -> matches independent product exactly
    res_indep = evaluate_sportsbook_parlay(
        legs, custom_parlay_odds=145, correlation_overrides={"0:1": 0.0}
    )
    assert abs(res_indep["joint_model_prob"] - 0.65 * 0.55) < 1e-4

    # 3-leg same-game cluster (ML + Total Over + Player Points Over) is deterministic and bounded
    legs_3 = [
        _pred_leg("p1", "evt_bos", "NYK @ BOS", "Moneyline", "BOS to Win", side="YES", prob=0.66, ask=0.58),
        _pred_leg("p2", "evt_bos", "NYK @ BOS", "Total", "Over 224.5 Total Points", side="YES", prob=0.56, ask=0.50),
        _pred_leg("p3", "evt_bos", "NYK @ BOS", "Prop: Points", "Jayson Tatum (BOS) Over 27.5 Points", side="YES", prob=0.58, ask=0.50, player_name="Jayson Tatum"),
    ]
    r3_a = evaluate_predictions_parlay(legs_3)
    r3_b = evaluate_predictions_parlay(legs_3)
    assert r3_a["is_sgp"] is True
    assert r3_a["joint_model_prob"] == r3_b["joint_model_prob"]
    assert r3_a["joint_model_prob"] > r3_a["independent_model_prob"]


def test_parlay_conflict_detection_blocks_invalid_slips() -> None:
    # Opposing Moneyline sides on same game
    opp_ml = [
        _pred_leg("m1", "evt_1", "NYK @ BOS", "Moneyline", "BOS to Win", side="YES", prob=0.62, ask=0.55),
        _pred_leg("m2", "evt_1", "NYK @ BOS", "Moneyline", "NYK to Win", side="NO", prob=0.38, ask=0.45),
    ]
    res_ml = evaluate_predictions_parlay(opp_ml)
    assert res_ml["has_conflicts"] is True
    assert res_ml["confidence_tier"] == "PASS"
    assert res_ml["recommended_contracts"] == 0

    # Same player opposite prop sides
    opp_prop = [
        _sb_leg("pr1", "evt_1", "NYK @ BOS", "Prop: Points", "Jayson Tatum Over 27.5 Points", side="OVER", prob=0.55, player_name="Jayson Tatum"),
        _sb_leg("pr2", "evt_1", "NYK @ BOS", "Prop: Points", "Jayson Tatum Under 27.5 Points", side="UNDER", prob=0.45, player_name="Jayson Tatum"),
    ]
    res_prop = evaluate_sportsbook_parlay(opp_prop)
    assert res_prop["has_conflicts"] is True
    assert res_prop["recommended_stake"] == 0.0

    # Zero-prob/empty safety on compute_joint_probability
    assert compute_joint_probability([]) == 0.0
    assert abs(compute_joint_probability([0.65]) - 0.65) < 1e-6


# --- 4. Auto-Build +EV Parlay Suggestions ---

def test_suggest_optimal_parlays_both_modes() -> None:
    pred_pool = [
        _pred_leg("a", "g1", "NYK @ BOS", "Moneyline", "BOS to Win", prob=0.70, ask=0.58),
        _pred_leg("b", "g2", "LAL @ DEN", "Moneyline", "DEN to Win", prob=0.69, ask=0.57),
        _pred_leg("c", "g3", "MIA @ PHI", "Total", "Over 218.5", prob=0.66, ask=0.56),
        _pred_leg("d", "g1", "NYK @ BOS", "Spread", "BOS -4.5 Cover", prob=0.63, ask=0.54),
    ]
    sugs_no_sgp = suggest_optimal_parlays(
        pred_pool, mode="predictions", num_legs=2, allow_same_game=False, max_results=5
    )
    assert len(sugs_no_sgp) >= 2
    assert all(not s["is_sgp"] for s in sugs_no_sgp)
    assert all(s["net_edge"] > 0 for s in sugs_no_sgp)

    sugs_sgp = suggest_optimal_parlays(
        pred_pool, mode="predictions", num_legs=2, allow_same_game=True, max_results=6
    )
    assert any(s["is_sgp"] for s in sugs_sgp)

    sb_pool = [
        _sb_leg("x1", "g1", "KC @ BUF", "Moneyline", "BUF ML", prob=0.60, dk_odds=-110, fd_odds=-102),
        _sb_leg("x2", "g2", "SF @ DAL", "Spread", "SF -3.5", prob=0.58, dk_odds=-105, fd_odds=-110),
        _sb_leg("x3", "g3", "BAL @ CIN", "Total", "Over 49.5", prob=0.57, dk_odds=-108, fd_odds=-104),
    ]
    sb_sugs = suggest_optimal_parlays(
        sb_pool, mode="sportsbook", num_legs=3, allow_same_game=False, profit_boost_pct=0.25
    )
    assert len(sb_sugs) == 1
    assert sb_sugs[0]["num_legs"] == 3
    assert sb_sugs[0]["ev_pct"] > 0.30


# --- 5. FastAPI Endpoints & Ledger Persistence ---

def test_parlay_api_endpoints_and_ledger_lifecycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "parlay_test.sqlite3"))
    monkeypatch.setenv("BANKROLL", "1000.0")
    monkeypatch.setenv("APP_PASSCODE", "")

    client = TestClient(app)

    # A. Predictions mode evaluate endpoint
    pred_payload = {
        "mode": "predictions",
        "pricing_structure": "COMBO_CONTRACT",
        "fee_per_contract": 0.015,
        "bankroll": 1000.0,
        "legs": [
            {
                "leg_id": "p1",
                "sport": "NBA",
                "event_id": "evt1",
                "event_name": "NYK @ BOS",
                "market_type": "Moneyline",
                "selection_label": "BOS to Win",
                "side": "YES",
                "model_prob": 0.70,
                "contract_price": 0.58,
            },
            {
                "leg_id": "p2",
                "sport": "NBA",
                "event_id": "evt2",
                "event_name": "LAL @ DEN",
                "market_type": "Moneyline",
                "selection_label": "DEN to Win",
                "side": "YES",
                "model_prob": 0.68,
                "contract_price": 0.55,
            },
        ],
    }
    r_pred = client.post("/api/parlay/evaluate", json=pred_payload)
    assert r_pred.status_code == 200
    d_pred = r_pred.json()
    assert d_pred["mode"] == "predictions"
    assert d_pred["num_legs"] == 2
    assert d_pred["net_edge"] > 0.10
    assert "BOS to Win + DEN to Win" in d_pred["summary_label"]

    # B. Sportsbooks mode evaluate endpoint
    sb_payload = {
        "mode": "sportsbook",
        "book": "BEST",
        "profit_boost_pct": 0.25,
        "bankroll": 1000.0,
        "legs": [
            {
                "leg_id": "s1",
                "sport": "NFL",
                "event_id": "nfl1",
                "event_name": "KC @ BUF",
                "market_type": "Moneyline",
                "selection_label": "BUF Moneyline",
                "side": "HOME",
                "model_prob": 0.59,
                "dk_odds": -110,
                "fd_odds": -102,
            },
            {
                "leg_id": "s2",
                "sport": "NFL",
                "event_id": "nfl2",
                "event_name": "SF @ DAL",
                "market_type": "Spread",
                "selection_label": "SF -3.5",
                "side": "AWAY",
                "model_prob": 0.58,
                "dk_odds": -108,
                "fd_odds": -104,
            },
        ],
    }
    r_sb = client.post("/api/parlay/evaluate", json=sb_payload)
    assert r_sb.status_code == 200
    d_sb = r_sb.json()
    assert d_sb["mode"] == "sportsbook"
    assert d_sb["book"] == "FANDUEL"
    assert d_sb["ev_pct"] > 0.40

    # C. Reject invalid American odds cleanly (422)
    bad_odds = {**sb_payload, "custom_parlay_odds": 50}
    assert client.post("/api/parlay/evaluate", json=bad_odds).status_code == 422

    # D. Auto-suggest endpoint using provided legs_pool for both modes
    r_sug = client.post(
        "/api/parlay/suggest",
        json={
            "mode": "sportsbook",
            "num_legs": 2,
            "allow_same_game": False,
            "legs_pool": sb_payload["legs"],
        },
    )
    assert r_sug.status_code == 200
    assert r_sug.json()["count"] == 1

    # E. Log both Predictions and Sportsbook parlays to Ledger and settle them
    log_pred = client.post(
        "/api/ledger",
        json={
            "sport": "NBA",
            "event_name": "2 Games (2-Leg)",
            "market_type": "Combo Parlay (2-Leg)",
            "selection_label": d_pred["summary_label"],
            "side": "YES",
            "entry_price": d_pred["combo_ask_price"],
            "fee_per_contract": d_pred["fee_per_contract"],
            "contracts": 10,
            "model_prob": d_pred["joint_model_prob"],
            "fair_exit_price": d_pred["joint_model_prob"],
            "venue": "DKEX",
            "bet_format": "CONTRACT",
            "strategy_tag": "PARLAY",
            "notes": d_pred["ledger_notes"],
        },
    )
    assert log_pred.status_code == 200
    pred_trade = log_pred.json()
    assert pred_trade["strategy_tag"] == "PARLAY"

    settle_pred = client.patch(f"/api/ledger/{pred_trade['id']}", json={"status": "WON"})
    assert settle_pred.status_code == 200
    assert settle_pred.json()["pnl"] > 6.0  # 10 contracts settling at $10 minus ~$3.34 cost

    log_sb = client.post(
        "/api/ledger",
        json={
            "sport": "NFL",
            "event_name": "2 Games (2-Leg)",
            "market_type": "Parlay (2-Leg)",
            "selection_label": d_sb["summary_label"],
            "side": "PARLAY",
            "venue": d_sb["book"],
            "bet_format": "SPORTSBOOK",
            "odds_american": d_sb["effective_american_odds"],
            "stake": 25.0,
            "model_prob": d_sb["joint_model_prob"],
            "strategy_tag": "PARLAY",
            "notes": d_sb["ledger_notes"],
        },
    )
    assert log_sb.status_code == 200
    sb_trade = log_sb.json()
    assert sb_trade["strategy_tag"] == "PARLAY"
    assert sb_trade["venue"] == "FANDUEL"

    settle_sb = client.patch(f"/api/ledger/{sb_trade['id']}", json={"status": "WON"})
    assert settle_sb.status_code == 200
    assert settle_sb.json()["pnl"] > 70.0

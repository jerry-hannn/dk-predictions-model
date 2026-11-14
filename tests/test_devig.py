"""Unit tests for Shin and Power de-vigging methods."""

from src.data.odds_client import (
    devig_american_pair,
    devig_multiplicative,
    devig_power,
    devig_shin,
)


def test_equal_vig_symmetric() -> None:
    p1, p2 = devig_american_pair(-110, -110, method="shin")
    assert abs(p1 - 0.50) < 1e-4
    assert abs(p2 - 0.50) < 1e-4
    assert abs((p1 + p2) - 1.0) < 1e-6


def test_favorite_longshot_bias_penalizes_longshot_more_than_multiplicative() -> None:
    # Heavy favorite -350 vs underdog +275
    fav_mult, dog_mult = devig_american_pair(-350, 275, method="multiplicative")
    fav_shin, dog_shin = devig_american_pair(-350, 275, method="shin")
    fav_pow, dog_pow = devig_american_pair(-350, 275, method="power")

    assert abs((fav_shin + dog_shin) - 1.0) < 1e-5
    assert abs((fav_pow + dog_pow) - 1.0) < 1e-5
    # Both Shin and Power recognize bookmakers load disproportionate vig onto the longshot,
    # so true favorite prob > multiplicative favorite prob, and true longshot prob < multiplicative
    assert fav_shin > fav_mult
    assert dog_shin < dog_mult
    assert fav_pow > fav_mult
    assert dog_pow < dog_mult


def test_sub_unity_sum_fallback() -> None:
    a, b = devig_shin(0.48, 0.48)
    m1, m2 = devig_multiplicative(0.48, 0.48)
    assert abs(a - m1) < 1e-6
    assert abs(b - m2) < 1e-6
    p1, p2 = devig_power(0.48, 0.48)
    assert abs(p1 - 0.50) < 1e-6
    assert abs(p2 - 0.50) < 1e-6

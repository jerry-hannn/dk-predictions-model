"""Market odds normalization, Shin/Power de-vigging, and live/quota-safe The Odds API client."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import requests

from src.config import get_settings
from src.engine.dk_math import american_to_implied_prob, clamp_prob, normalize_player_name
from src.storage.db import Database


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    raise_for_status: bool = True,
) -> Any:
    """
    Execute an HTTP GET via httpx, gracefully falling back to requests when container
    NO_PROXY contains unbracketed IPv6 literals (e.g. `::1`) that trigger httpx.InvalidURL.
    """
    if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("ALLOW_PYTEST_NETWORK") != "1":
        raise RuntimeError("Live network disabled during pytest")
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.get(url, params=params)
    except httpx.InvalidURL:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    if raise_for_status:
        resp.raise_for_status()
    return resp

# Sharpness weights when aggregating multi-book consensus
BOOK_SHARPNESS_WEIGHTS: dict[str, float] = {
    "pinnacle": 3.2,
    "circa": 2.6,
    "bookmaker": 2.3,
    "lowvig": 2.2,
    "betfair_ex_eu": 2.1,
    "matchbook": 2.0,
    "betonlineag": 1.85,
    "fanduel": 1.45,
    "draftkings": 1.35,
    "bovada": 1.1,
    "betrivers": 1.05,
    "betmgm": 1.0,
    "caesars": 1.0,
    "espnbet": 1.0,
}

PROP_MARKET_MAP_NBA = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra",
}

PROP_MARKET_MAP_NFL = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}


def devig_multiplicative(raw_prob_a: float, raw_prob_b: float) -> tuple[float, float]:
    total = raw_prob_a + raw_prob_b
    if total <= 0:
        return 0.5, 0.5
    return raw_prob_a / total, raw_prob_b / total


def devig_power(raw_prob_a: float, raw_prob_b: float) -> tuple[float, float]:
    """
    Power method de-vigging: finds exponent k >= 1 such that q_a^k + q_b^k = 1.
    Properly accounts for favorite-longshot bias (longshots carry more vig than favorites).
    """
    qa = clamp_prob(raw_prob_a, 1e-6)
    qb = clamp_prob(raw_prob_b, 1e-6)
    if abs((qa + qb) - 1.0) < 1e-6:
        return qa, qb
    if qa + qb < 1.0:
        return devig_multiplicative(qa, qb)

    lo, hi = 1.0, 10.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        val = (qa**mid) + (qb**mid)
        if val > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    pa = qa**k
    pb = 1.0 - pa
    return clamp_prob(pa), clamp_prob(pb)


def devig_shin(raw_prob_a: float, raw_prob_b: float) -> tuple[float, float]:
    """
    Exact two-outcome Shin (1993) de-vigging formula solving for insider trading proportion z.
    """
    qa = clamp_prob(raw_prob_a, 1e-6)
    qb = clamp_prob(raw_prob_b, 1e-6)
    s = qa + qb
    if s <= 1.0 + 1e-6:
        return devig_multiplicative(qa, qb)

    lo, hi = 0.0, min(0.99, s - 1.0 + 1e-6)
    for _ in range(60):
        z = 0.5 * (lo + hi)
        lhs = math.sqrt(z * z + 4.0 * (1.0 - z) * (qa * qa) / s) + math.sqrt(
            z * z + 4.0 * (1.0 - z) * (qb * qb) / s
        )
        if lhs > 2.0:
            lo = z
        else:
            hi = z
    z = 0.5 * (lo + hi)
    pa = (math.sqrt(z * z + 4.0 * (1.0 - z) * (qa * qa) / s) - z) / (2.0 * (1.0 - z))
    pb = 1.0 - pa
    return clamp_prob(pa), clamp_prob(pb)


def devig_american_pair(
    odds_a: float, odds_b: float, method: str = "shin"
) -> tuple[float, float]:
    """Convert a two-way American odds pair (e.g., -115 / -105 or -165 / +140) to true probs."""
    qa = american_to_implied_prob(odds_a)
    qb = american_to_implied_prob(odds_b)
    if method == "power":
        return devig_power(qa, qb)
    if method == "multiplicative":
        return devig_multiplicative(qa, qb)
    return devig_shin(qa, qb)


def _iso_age_seconds(iso_str: str | None) -> float | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, round((datetime.now(timezone.utc) - dt).total_seconds(), 1))
    except ValueError:
        return None


def _normalize_team_tokens(name: str) -> str:
    s = str(name or "").lower().strip()
    replacements = {
        "la clippers": "los angeles clippers",
        "ny knicks": "new york knicks",
        "ny giants": "new york giants",
        "ny jets": "new york jets",
    }
    return replacements.get(s, s)


def _teams_match(a: str, b: str) -> bool:
    na = _normalize_team_tokens(a)
    nb = _normalize_team_tokens(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    # Mascot match (last word, e.g. "clippers", "timberwolves", "commanders")
    ta = na.split()[-1]
    tb = nb.split()[-1]
    return len(ta) >= 4 and ta == tb


class OddsClient:
    """Multi-book sharp consensus & DraftKings live line client via The Odds API v4."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEYS = {
        "NFL": "americanfootball_nfl",
        "NBA": "basketball_nba",
    }
    QUOTA_CACHE_KEY = "odds_api:quota"

    def __init__(self, api_key: str | None = None, db: Database | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.the_odds_api_key
        self.db = db or Database()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def get_quota_status(self) -> dict[str, Any]:
        q = self.db.get_cache(self.QUOTA_CACHE_KEY) or {}
        rem = q.get("requests_remaining")
        used = q.get("requests_used")
        return {
            "requests_remaining": rem,
            "requests_used": used,
            "low_quota_guard": rem is not None and int(rem) <= 15,
            "updated_at": q.get("updated_at"),
        }

    def _record_quota_headers(self, headers: Any) -> None:
        rem_raw = headers.get("x-requests-remaining")
        used_raw = headers.get("x-requests-used")
        if rem_raw is None and used_raw is None:
            return
        try:
            rem = int(float(rem_raw)) if rem_raw is not None else None
            used = int(float(used_raw)) if used_raw is not None else None
            self.db.set_cache(
                self.QUOTA_CACHE_KEY,
                {
                    "requests_remaining": rem,
                    "requests_used": used,
                    "updated_at": round(time.time(), 1),
                },
                ttl_seconds=86400 * 14,
            )
        except ValueError:
            pass

    def fetch_game_odds(
        self,
        sport: str,
        force_refresh: bool = False,
        live_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Fetch multi-book h2h, spreads, and totals if THE_ODDS_API_KEY is configured.
        Uses `regions=us` (3 credits vs 6 for `us,eu`) and dynamic TTL (90s in-play vs 600s pre-game)
        to preserve free-tier API credits while keeping live lines fresh.
        """
        if not self.enabled:
            return []

        sport_u = sport.upper()
        sport_key = self.SPORT_KEYS.get(sport_u)
        if not sport_key:
            return []

        cache_key = f"odds_api:{sport_u}:game_odds"
        max_fresh_age = 90.0 if live_mode else 600.0

        if not force_refresh:
            cached, age = self.db.get_cache_with_age(cache_key)
            if cached is not None and (age is None or age <= max_fresh_age):
                return cached

        quota = self.get_quota_status()
        if quota["low_quota_guard"] and not force_refresh:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale or []

        url = f"{self.BASE_URL}/sports/{sport_key}/odds"
        # `us` includes DraftKings, FanDuel, BetMGM, Caesars, BetRivers, LowVig, BetOnline, Bovada (3 credits)
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }
        try:
            resp = http_get(url, params=params, timeout=10.0, raise_for_status=False)
            self._record_quota_headers(resp.headers)
            resp.raise_for_status()
            payload = resp.json()
            ttl = 90 if live_mode else 600
            self.db.set_cache(cache_key, payload, ttl_seconds=ttl)
            return payload
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale or []

    def fetch_event_player_props(
        self,
        sport: str,
        odds_event_id: str,
        force_refresh: bool = False,
        live_mode: bool = False,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Opt-in fetch of live DraftKings / US sportsbook player prop lines & implied contract prices
        for a specific event ID. Guarded against low free-tier API quota.
        """
        if not self.enabled or not odds_event_id:
            return {}

        sport_u = sport.upper()
        sport_key = self.SPORT_KEYS.get(sport_u)
        if not sport_key:
            return {}

        cache_key = f"odds_api:{sport_u}:props:{odds_event_id}"
        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached is not None:
                out_cached: dict[tuple[str, str], dict[str, Any]] = {}
                for k, v in cached.items():
                    if "::" not in k:
                        continue
                    p_low, col = k.split("::", 1)
                    out_cached[(p_low, col)] = v
                    norm_p = normalize_player_name(p_low)
                    if norm_p:
                        out_cached.setdefault((norm_p, col), v)
                return out_cached

        quota = self.get_quota_status()
        rem = quota.get("requests_remaining")
        if rem is not None and int(rem) <= 40 and not force_refresh:
            return {}

        prop_map = PROP_MARKET_MAP_NBA if sport_u == "NBA" else PROP_MARKET_MAP_NFL
        url = f"{self.BASE_URL}/sports/{sport_key}/events/{odds_event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(prop_map.keys()),
            "bookmakers": "draftkings,fanduel,betmgm",
            "oddsFormat": "american",
        }
        try:
            resp = http_get(url, params=params, timeout=10.0, raise_for_status=False)
            self._record_quota_headers(resp.headers)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return {}

        parsed = self.extract_player_prop_lines(sport_u, payload)
        serializable = {f"{p_low}::{col}": val for (p_low, col), val in parsed.items()}
        self.db.set_cache(cache_key, serializable, ttl_seconds=120 if live_mode else 600)
        return parsed

    @staticmethod
    def extract_player_prop_lines(
        sport: str, event_payload: dict[str, Any]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Extract DraftKings, FanDuel, and consensus player prop lines, American odds, & de-vigged probabilities."""
        prop_map = (
            PROP_MARKET_MAP_NBA if sport.upper() == "NBA" else PROP_MARKET_MAP_NFL
        )
        # Sort bookmakers so DraftKings is processed last (overwriting or anchoring line)
        books = list(event_payload.get("bookmakers") or [])
        books.sort(key=lambda b: 1 if str(b.get("key", "")).lower() == "draftkings" else 0)

        accum: dict[tuple[str, str], list[tuple[str, float, float, int, int]]] = {}
        for book in books:
            bkey = str(book.get("key", "")).lower()
            for mkt in book.get("markets") or []:
                col = prop_map.get(str(mkt.get("key", "")))
                if not col:
                    continue
                by_player: dict[str, dict[str, Any]] = {}
                for out in mkt.get("outcomes") or []:
                    pname = str(out.get("description") or "").strip().lower()
                    side_nm = str(out.get("name") or "").strip().lower()
                    if not pname or side_nm not in ("over", "under") or "point" not in out:
                        continue
                    by_player.setdefault(pname, {})[side_nm] = out

                for pname, pair in by_player.items():
                    if "over" in pair and "under" in pair:
                        ln = float(pair["over"]["point"])
                        o_px = float(pair["over"]["price"])
                        u_px = float(pair["under"]["price"])
                        po, _ = devig_american_pair(o_px, u_px)
                        accum.setdefault((pname, col), []).append(
                            (bkey, ln, po, int(round(o_px)), int(round(u_px)))
                        )

        out_props: dict[tuple[str, str], dict[str, Any]] = {}
        for key, entries in accum.items():
            dk_entry = next((e for e in entries if e[0] == "draftkings"), None)
            fd_entry = next((e for e in entries if e[0] == "fanduel"), None)
            anchor_entry = dk_entry or entries[-1]
            target_line = anchor_entry[1]
            same_line_probs = [p for _, ln, p, _, _ in entries if abs(ln - target_line) < 0.01]
            mkt_over_p = sum(same_line_probs) / len(same_line_probs)

            book_lines: dict[str, dict[str, Any]] = {}
            for bkey, ln, po, o_px, u_px in entries:
                book_lines[bkey] = {
                    "line": ln,
                    "over_odds": o_px,
                    "under_odds": u_px,
                    "over_prob": round(po, 4),
                }

            val_obj: dict[str, Any] = {
                "line": target_line,
                "over_prob": round(mkt_over_p, 4),
                "dk_over_prob": round(anchor_entry[2], 4),
                "source": (
                    "DraftKings"
                    if anchor_entry[0] == "draftkings"
                    else anchor_entry[0].title()
                ),
                "books": book_lines,
            }
            if dk_entry is not None:
                val_obj["dk_line"] = dk_entry[1]
                val_obj["dk_over_odds"] = dk_entry[3]
                val_obj["dk_under_odds"] = dk_entry[4]
            if fd_entry is not None:
                val_obj["fd_line"] = fd_entry[1]
                val_obj["fd_over_odds"] = fd_entry[3]
                val_obj["fd_under_odds"] = fd_entry[4]
                val_obj["fd_over_prob"] = round(fd_entry[2], 4)

            out_props[key] = val_obj
            norm_p = normalize_player_name(key[0])
            if norm_p:
                out_props.setdefault((norm_p, key[1]), val_obj)
        return out_props

    @staticmethod
    def find_matching_event(
        odds_events: list[dict[str, Any]], home_team: str, away_team: str
    ) -> dict[str, Any] | None:
        for ev in odds_events:
            ev_h = str(ev.get("home_team", ""))
            ev_a = str(ev.get("away_team", ""))
            if _teams_match(home_team, ev_h) and _teams_match(away_team, ev_a):
                return ev
        return None

    @classmethod
    def extract_consensus(
        cls, odds_events: list[dict[str, Any]], home_team: str, away_team: str
    ) -> dict[str, Any] | None:
        """
        Compute sharpness-weighted de-vigged h2h, spread, and total consensus AND extract
        exact DraftKings & FanDuel lines/prices, sharp-only consensus, book dispersion, and freshness.
        """
        matched = cls.find_matching_event(odds_events, home_team, away_team)
        if matched is None:
            return None

        h2h_num, h2h_den = 0.0, 0.0
        sharp_h2h_num, sharp_h2h_den = 0.0, 0.0
        spread_lines: list[tuple[str, float, float, float]] = []  # (bkey, weight, home_line, home_prob)
        sharp_spread_lines: list[tuple[str, float, float, float]] = []
        total_lines: list[tuple[str, float, float, float]] = []   # (bkey, weight, total_pts, over_prob)
        sharp_total_lines: list[tuple[str, float, float, float]] = []

        dk_data: dict[str, Any] = {}
        fd_data: dict[str, Any] = {}
        freshest_age: float | None = None
        books_seen = 0
        sharp_books_seen = 0

        for book in matched.get("bookmakers", []):
            bkey = str(book.get("key", "")).lower()
            w = BOOK_SHARPNESS_WEIGHTS.get(bkey, 1.0)
            books_seen += 1
            is_non_dk_fd = bkey not in ("draftkings", "fanduel")
            if is_non_dk_fd:
                sharp_books_seen += 1

            b_age = _iso_age_seconds(book.get("last_update"))
            if b_age is not None:
                freshest_age = b_age if freshest_age is None else min(freshest_age, b_age)

            for mkt in book.get("markets", []):
                mkey = mkt.get("key")
                outcomes = mkt.get("outcomes", [])
                if len(outcomes) != 2:
                    continue
                m_age = _iso_age_seconds(mkt.get("last_update"))
                if m_age is not None:
                    freshest_age = m_age if freshest_age is None else min(freshest_age, m_age)

                if mkey == "h2h":
                    home_out = next(
                        (o for o in outcomes if _teams_match(str(o.get("name", "")), matched["home_team"])),
                        None,
                    )
                    away_out = next((o for o in outcomes if o is not home_out), None)
                    if home_out and away_out:
                        h_px = float(home_out["price"])
                        a_px = float(away_out["price"])
                        ph, _ = devig_american_pair(h_px, a_px)
                        h2h_num += w * ph
                        h2h_den += w
                        if is_non_dk_fd:
                            sharp_h2h_num += w * ph
                            sharp_h2h_den += w
                        if bkey == "draftkings":
                            dk_data["dk_home_ml"] = int(round(h_px))
                            dk_data["dk_away_ml"] = int(round(a_px))
                            dk_data["dk_home_win_prob"] = round(ph, 4)
                            dk_data["dk_age_seconds"] = m_age or b_age
                        elif bkey == "fanduel":
                            fd_data["fd_home_ml"] = int(round(h_px))
                            fd_data["fd_away_ml"] = int(round(a_px))
                            fd_data["fd_home_win_prob"] = round(ph, 4)
                            fd_data["fd_age_seconds"] = m_age or b_age

                elif mkey == "spreads":
                    home_out = next(
                        (o for o in outcomes if _teams_match(str(o.get("name", "")), matched["home_team"])),
                        None,
                    )
                    away_out = next((o for o in outcomes if o is not home_out), None)
                    if home_out and away_out and "point" in home_out:
                        h_pt = float(home_out["point"])
                        h_px = float(home_out["price"])
                        a_px = float(away_out["price"])
                        ph, _ = devig_american_pair(h_px, a_px)
                        spread_lines.append((bkey, w, h_pt, ph))
                        if is_non_dk_fd:
                            sharp_spread_lines.append((bkey, w, h_pt, ph))
                        if bkey == "draftkings":
                            dk_data["dk_home_spread"] = h_pt
                            dk_data["dk_home_spread_odds"] = int(round(h_px))
                            dk_data["dk_away_spread_odds"] = int(round(a_px))
                            dk_data["dk_home_cover_prob"] = round(ph, 4)
                            dk_data["dk_age_seconds"] = m_age or b_age
                        elif bkey == "fanduel":
                            fd_data["fd_home_spread"] = h_pt
                            fd_data["fd_home_spread_odds"] = int(round(h_px))
                            fd_data["fd_away_spread_odds"] = int(round(a_px))
                            fd_data["fd_home_cover_prob"] = round(ph, 4)
                            fd_data["fd_age_seconds"] = m_age or b_age

                elif mkey == "totals":
                    over_out = next(
                        (o for o in outcomes if str(o.get("name", "")).lower() == "over"),
                        None,
                    )
                    under_out = next(
                        (o for o in outcomes if str(o.get("name", "")).lower() == "under"),
                        None,
                    )
                    if over_out and under_out and "point" in over_out:
                        t_pt = float(over_out["point"])
                        o_px = float(over_out["price"])
                        u_px = float(under_out["price"])
                        po, _ = devig_american_pair(o_px, u_px)
                        total_lines.append((bkey, w, t_pt, po))
                        if is_non_dk_fd:
                            sharp_total_lines.append((bkey, w, t_pt, po))
                        if bkey == "draftkings":
                            dk_data["dk_total_line"] = t_pt
                            dk_data["dk_over_odds"] = int(round(o_px))
                            dk_data["dk_under_odds"] = int(round(u_px))
                            dk_data["dk_over_prob"] = round(po, 4)
                            dk_data["dk_age_seconds"] = m_age or b_age
                        elif bkey == "fanduel":
                            fd_data["fd_total_line"] = t_pt
                            fd_data["fd_over_odds"] = int(round(o_px))
                            fd_data["fd_under_odds"] = int(round(u_px))
                            fd_data["fd_over_prob"] = round(po, 4)
                            fd_data["fd_age_seconds"] = m_age or b_age

        if dk_data and fd_data:
            src_label = f"DK + FD + {books_seen}-Book Sharp"
        elif dk_data:
            src_label = f"DraftKings + {books_seen}-Book Sharp"
        elif fd_data:
            src_label = f"FanDuel + {books_seen}-Book Sharp"
        else:
            src_label = f"{books_seen}-Book Sharp Consensus"

        result: dict[str, Any] = {
            "odds_event_id": str(matched.get("id", "")),
            "books_count": books_seen,
            "sharp_books_count": sharp_books_seen,
            "has_draftkings": bool(dk_data),
            "has_fanduel": bool(fd_data),
            "odds_age_seconds": dk_data.get("dk_age_seconds", fd_data.get("fd_age_seconds", freshest_age)),
            "line_source": src_label,
            "books_detail": {
                "draftkings": dict(dk_data),
                "fanduel": dict(fd_data),
            },
        }
        result.update(dk_data)
        result.update(fd_data)

        if h2h_den > 0:
            sharp_ml_p = round(h2h_num / h2h_den, 4)
            result["home_win_prob"] = sharp_ml_p
            result["sharp_home_win_prob"] = sharp_ml_p
        if sharp_h2h_den > 0:
            result["sharp_only_home_win_prob"] = round(sharp_h2h_num / sharp_h2h_den, 4)

        if spread_lines:
            modal_line = max(
                {ln for _, _, ln, _ in spread_lines},
                key=lambda x: sum(w for _, w, ln, _ in spread_lines if abs(ln - x) < 0.01),
            )
            result["sharp_home_spread"] = modal_line
            all_pts = [ln for _, _, ln, _ in spread_lines]
            result["spread_dispersion"] = round(max(all_pts) - min(all_pts), 2)

            if sharp_spread_lines:
                sharp_modal_sp = max(
                    {ln for _, _, ln, _ in sharp_spread_lines},
                    key=lambda x: sum(w for _, w, ln, _ in sharp_spread_lines if abs(ln - x) < 0.01),
                )
                result["sharp_only_home_spread"] = sharp_modal_sp
                same_sharp_sp = [
                    (w, p) for _, w, ln, p in sharp_spread_lines if abs(ln - sharp_modal_sp) < 0.01
                ]
                if same_sharp_sp:
                    result["sharp_only_home_cover_prob"] = round(
                        sum(w * p for w, p in same_sharp_sp) / sum(w for w, _ in same_sharp_sp), 4
                    )

            # Evaluate the actual DraftKings spread line when available so DKeX picks match what's tradable on DK!
            target_sp = dk_data.get("dk_home_spread", modal_line)
            result["home_spread"] = target_sp
            same_sp = [(w, p) for _, w, ln, p in spread_lines if abs(ln - target_sp) < 0.01]
            if same_sp:
                result["home_cover_prob"] = round(
                    sum(w * p for w, p in same_sp) / sum(w for w, _ in same_sp), 4
                )
            else:
                result["home_cover_prob"] = dk_data.get("dk_home_cover_prob", 0.50)

        if total_lines:
            modal_tot = max(
                {ln for _, _, ln, _ in total_lines},
                key=lambda x: sum(w for _, w, ln, _ in total_lines if abs(ln - x) < 0.01),
            )
            result["sharp_total_line"] = modal_tot
            all_tots = [ln for _, _, ln, _ in total_lines]
            result["total_dispersion"] = round(max(all_tots) - min(all_tots), 2)

            if sharp_total_lines:
                sharp_modal_tot = max(
                    {ln for _, _, ln, _ in sharp_total_lines},
                    key=lambda x: sum(w for _, w, ln, _ in sharp_total_lines if abs(ln - x) < 0.01),
                )
                result["sharp_only_total_line"] = sharp_modal_tot
                same_sharp_tot = [
                    (w, p) for _, w, ln, p in sharp_total_lines if abs(ln - sharp_modal_tot) < 0.01
                ]
                if same_sharp_tot:
                    result["sharp_only_over_prob"] = round(
                        sum(w * p for w, p in same_sharp_tot) / sum(w for w, _ in same_sharp_tot), 4
                    )

            target_tot = dk_data.get("dk_total_line", modal_tot)
            result["total_line"] = target_tot
            same_tot = [(w, p) for _, w, ln, p in total_lines if abs(ln - target_tot) < 0.01]
            if same_tot:
                result["over_prob"] = round(
                    sum(w * p for w, p in same_tot) / sum(w for w, _ in same_tot), 4
                )
            else:
                result["over_prob"] = dk_data.get("dk_over_prob", 0.50)

        return result if (h2h_den > 0 or spread_lines or total_lines) else None

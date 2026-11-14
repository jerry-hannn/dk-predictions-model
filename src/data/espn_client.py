"""Public ESPN JSON API client for NFL & NBA scoreboards, live game state, box scores, DraftKings odds, and injuries."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import httpx

from src.data.odds_client import devig_american_pair
from src.storage.db import Database

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

SPORT_PATHS = {
    "NFL": ("football", "nfl"),
    "NBA": ("basketball", "nba"),
}

TEAM_NAME_TO_ABBR = {
    # NFL (32)
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LA", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
    # NBA (30)
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
    "charlotte hornets": "CHA", "chicago bulls": "CHI", "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL", "denver nuggets": "DEN", "detroit pistons": "DET",
    "golden state warriors": "GSW", "houston rockets": "HOU", "indiana pacers": "IND",
    "la clippers": "LAC", "los angeles clippers": "LAC", "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM", "miami heat": "MIA", "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN", "new orleans pelicans": "NOP", "new york knicks": "NYK",
    "oklahoma city thunder": "OKC", "orlando magic": "ORL", "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX", "portland trail blazers": "POR", "sacramento kings": "SAC",
    "san antonio spurs": "SAS", "toronto raptors": "TOR", "utah jazz": "UTA",
    "washington wizards": "WAS",
}


def _parse_american_str(val: Any) -> float | None:
    """Parse American odds values like '-118', '+114', 'EVEN', or numeric -110."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return 100.0 if float(val) == 0.0 else float(val)
    s = str(val).strip().upper()
    if not s or s in ("N/A", "-", "OFF"):
        return None
    if s in ("EVEN", "EV", "PK"):
        return 100.0
    try:
        parsed = float(s)
        return 100.0 if parsed == 0.0 else parsed
    except ValueError:
        return None


def _parse_line_number(val: Any) -> float | None:
    """Parse line strings such as '-2.5', '+3', 'o46.5', 'u213.5', 'PK', 'EVEN'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().upper()
    if not s:
        return None
    if s in ("EVEN", "PK", "PICK"):
        return 0.0
    s = s.lstrip("OU")
    try:
        return float(s)
    except ValueError:
        return None


def compute_game_clock_progress(
    sport: str,
    state: str,
    period: int,
    clock_seconds: float,
    status_detail: str = "",
) -> tuple[float, float]:
    """
    Return (elapsed_fraction, remaining_minutes) for regulation/overtime NFL & NBA games.
    """
    sport_u = sport.upper()
    reg_total = 48.0 if sport_u == "NBA" else 60.0
    period_len = 12.0 if sport_u == "NBA" else 15.0
    ot_len = 5.0 if sport_u == "NBA" else 10.0

    state_u = state.upper()
    if state_u == "PRE":
        return 0.0, reg_total
    if state_u == "POST":
        return 1.0, 0.0

    detail_low = status_detail.lower()
    if "halftime" in detail_low or "half" in detail_low and period == 2 and clock_seconds <= 0.5:
        return 0.50, reg_total * 0.50

    p = max(1, int(period or 1))
    clk_min = max(0.0, float(clock_seconds or 0.0) / 60.0)

    if p <= 4:
        rem_in_p = min(period_len, clk_min)
        if "end of" in detail_low and clk_min <= 0.05:
            rem_in_p = 0.0
        periods_after = max(0, 4 - p)
        rem_mins = periods_after * period_len + rem_in_p
        elapsed = min(0.995, max(0.0, 1.0 - (rem_mins / reg_total)))
        return round(elapsed, 4), round(rem_mins, 2)

    # Overtime: treat remaining variance relative to full regulation scale
    rem_mins = min(ot_len, max(0.25, clk_min))
    elapsed = min(0.995, max(0.90, 1.0 - (rem_mins / reg_total)))
    return round(elapsed, 4), round(rem_mins, 2)


def _parse_minutes_str(val: str) -> float:
    """Parse ESPN NBA boxscore minute strings like '28', '28:14', or '--'."""
    s = str(val).strip()
    if not s or s in ("--", "DNP", "0"):
        return 0.0
    if ":" in s:
        parts = s.split(":", 1)
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_made_attempted(val: str) -> tuple[float, float]:
    """Parse '5-10' into (5.0, 10.0)."""
    s = str(val).strip()
    sep = "-" if "-" in s else ("/" if "/" in s else None)
    if sep:
        a, b = s.split(sep, 1)
        try:
            return float(a), float(b)
        except ValueError:
            return 0.0, 0.0
    try:
        v = float(s)
        return v, v
    except ValueError:
        return 0.0, 0.0


class ESPNClient:
    SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports"
    CORE_BASE = "https://sports.core.api.espn.com/v2/sports"

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    def fetch_scoreboard(
        self,
        sport: str,
        target_date: str | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Fetch upcoming & live in-progress games for NFL or NBA with live scores, clocks,
        play-by-play win probabilities, and embedded DraftKings odds.
        """
        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]

        dates_to_check: list[str | None]
        if sport_u == "NFL" and target_date is None:
            dates_to_check = [None]
        else:
            dt = (
                datetime.strptime(target_date, "%Y-%m-%d")
                if target_date
                else datetime.now()
            )
            d0 = dt.strftime("%Y%m%d")
            d1 = (dt + timedelta(days=1)).strftime("%Y%m%d")
            dates_to_check = [d0, d1] if target_date is None else [d0]

        normalized_events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for d_param in dates_to_check:
            cache_key = f"espn:scoreboard:{sport_u}:{d_param or 'current_week'}"
            payload = None if force_refresh else self.db.get_cache(cache_key)
            if payload is None:
                url = f"{self.SITE_BASE}/{sp}/{lg}/scoreboard"
                params = {"dates": d_param} if d_param else {}
                try:
                    with httpx.Client(timeout=10.0, headers=HEADERS) as client:
                        resp = client.get(url, params=params)
                        resp.raise_for_status()
                        payload = resp.json()
                    has_live = any(
                        str(
                            ev.get("competitions", [{}])[0]
                            .get("status", {})
                            .get("type", {})
                            .get("state", "")
                        ).lower()
                        == "in"
                        for ev in payload.get("events", [])
                        if ev.get("competitions")
                    )
                    # Keep short 45s TTL while games are actively in progress; 300s otherwise
                    ttl = 45 if has_live else 300
                    self.db.set_cache(cache_key, payload, ttl_seconds=ttl)
                except Exception:
                    payload, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)

            if not payload:
                continue

            for ev in payload.get("events", []):
                ev_id = str(ev.get("id", ""))
                if not ev_id or ev_id in seen_ids:
                    continue
                parsed = self._parse_event(sport_u, ev)
                if parsed is not None:
                    seen_ids.add(ev_id)
                    normalized_events.append(parsed)

            active_today = [
                e for e in normalized_events if e["state"] in ("PRE", "IN")
            ]
            if sport_u == "NBA" and len(active_today) >= 3:
                break

        return normalized_events

    def fetch_live_boxscore(
        self, sport: str, event_id: str, force_refresh: bool = False
    ) -> dict[str, Any]:
        """
        Fetch live player boxscore stats, live win probability, and event pickcenter odds
        from ESPN's zero-key `/summary?event={event_id}` endpoint.
        """
        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]
        cache_key = f"espn:summary:{sport_u}:{event_id}"
        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached is not None:
                return cached

        url = f"{self.SITE_BASE}/{sp}/{lg}/summary"
        try:
            with httpx.Client(timeout=8.5, headers=HEADERS) as client:
                resp = client.get(url, params={"event": event_id})
                resp.raise_for_status()
                raw = resp.json()
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale or {"players": {}, "live_win_prob": None, "market": {}}

        parsed = self.parse_summary_payload(sport_u, raw)
        self.db.set_cache(cache_key, parsed, ttl_seconds=45)
        return parsed

    @classmethod
    def parse_summary_payload(cls, sport: str, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse ESPN game summary JSON into normalized per-player live stats and odds."""
        sport_u = sport.upper()
        players_map: dict[str, dict[str, Any]] = {}

        box_players = raw.get("boxscore", {}).get("players", []) or []
        for team_entry in box_players:
            raw_team_abbr = str(team_entry.get("team", {}).get("abbreviation", ""))
            team_abbr = cls._normalize_abbr(sport_u, raw_team_abbr)

            if sport_u == "NBA":
                for stat_group in team_entry.get("statistics", []) or []:
                    labels = [str(x).upper() for x in (stat_group.get("labels") or [])]
                    idx_map = {lbl: i for i, lbl in enumerate(labels)}
                    for ath_row in stat_group.get("athletes", []) or []:
                        ath = ath_row.get("athlete", {}) or {}
                        pname = str(ath.get("displayName", "")).strip()
                        if not pname:
                            continue
                        stats = ath_row.get("stats") or []
                        dnp = bool(ath_row.get("didNotPlay", False))
                        ejected = bool(ath_row.get("ejected", False))

                        def _get_stat(lbl: str, _s: list[Any] = stats, _m: dict[str, int] = idx_map) -> str:
                            idx = _m.get(lbl)
                            return str(_s[idx]) if idx is not None and idx < len(_s) else "0"

                        mins = 0.0 if dnp else _parse_minutes_str(_get_stat("MIN"))
                        pts = 0.0 if dnp else (_parse_line_number(_get_stat("PTS")) or 0.0)
                        reb = 0.0 if dnp else (_parse_line_number(_get_stat("REB")) or 0.0)
                        ast = 0.0 if dnp else (_parse_line_number(_get_stat("AST")) or 0.0)
                        fg3m, _ = (0.0, 0.0) if dnp else _parse_made_attempted(_get_stat("3PT"))
                        fouls = 0.0 if dnp else (_parse_line_number(_get_stat("PF")) or 0.0)

                        players_map[pname.lower()] = {
                            "player_name": pname,
                            "team": team_abbr,
                            "starter": bool(ath_row.get("starter", False)),
                            "active": bool(ath_row.get("active", True)) and not dnp and not ejected,
                            "did_not_play": dnp,
                            "ejected": ejected,
                            "minutes": round(mins, 2),
                            "points": pts,
                            "rebounds": reb,
                            "assists": ast,
                            "fg3m": fg3m,
                            "pra": pts + reb + ast,
                            "fouls": fouls,
                        }
            else:
                # NFL categories: passing, rushing, receiving
                for stat_group in team_entry.get("statistics", []) or []:
                    cat_name = str(stat_group.get("name", "")).lower()
                    if cat_name not in ("passing", "rushing", "receiving"):
                        continue
                    labels = [str(x).upper() for x in (stat_group.get("labels") or [])]
                    idx_map = {lbl: i for i, lbl in enumerate(labels)}
                    for ath_row in stat_group.get("athletes", []) or []:
                        ath = ath_row.get("athlete", {}) or {}
                        pname = str(ath.get("displayName", "")).strip()
                        if not pname:
                            continue
                        stats = ath_row.get("stats") or []
                        entry = players_map.setdefault(
                            pname.lower(),
                            {
                                "player_name": pname,
                                "team": team_abbr,
                                "active": True,
                                "did_not_play": False,
                                "ejected": False,
                                "passing_yards": 0.0,
                                "passing_tds": 0.0,
                                "passing_attempts": 0.0,
                                "rushing_yards": 0.0,
                                "rushing_attempts": 0.0,
                                "receiving_yards": 0.0,
                                "receptions": 0.0,
                            },
                        )

                        def _nfl_val(lbl: str, _s: list[Any] = stats, _m: dict[str, int] = idx_map) -> str:
                            idx = _m.get(lbl)
                            return str(_s[idx]) if idx is not None and idx < len(_s) else "0"

                        if cat_name == "passing":
                            _, att = _parse_made_attempted(_nfl_val("C/ATT"))
                            entry["passing_attempts"] = att
                            entry["passing_yards"] = _parse_line_number(_nfl_val("YDS")) or 0.0
                            entry["passing_tds"] = _parse_line_number(_nfl_val("TD")) or 0.0
                        elif cat_name == "rushing":
                            entry["rushing_attempts"] = _parse_line_number(_nfl_val("CAR")) or 0.0
                            entry["rushing_yards"] = _parse_line_number(_nfl_val("YDS")) or 0.0
                        elif cat_name == "receiving":
                            entry["receptions"] = _parse_line_number(_nfl_val("REC")) or 0.0
                            entry["receiving_yards"] = _parse_line_number(_nfl_val("YDS")) or 0.0

        live_wp: float | None = None
        wp_list = raw.get("winprobability") or []
        if isinstance(wp_list, list) and wp_list:
            last_wp = wp_list[-1]
            if isinstance(last_wp, dict) and last_wp.get("homeWinPercentage") is not None:
                try:
                    live_wp = float(last_wp["homeWinPercentage"])
                except (TypeError, ValueError):
                    live_wp = None

        # Extract header home abbreviation if pickcenter/odds present
        home_abbr = ""
        header_comps = raw.get("header", {}).get("competitions", []) or []
        if header_comps:
            for c in header_comps[0].get("competitors", []) or []:
                if c.get("homeAway") == "home":
                    home_abbr = cls._normalize_abbr(
                        sport_u, str(c.get("team", {}).get("abbreviation", ""))
                    )
                    break

        odds_blocks = raw.get("odds") or raw.get("pickcenter") or []
        market = (
            cls._extract_embedded_odds(odds_blocks, home_abbr)
            if odds_blocks and home_abbr
            else {}
        )
        return {
            "players": players_map,
            "live_win_prob": live_wp,
            "market": market,
        }

    def fetch_league_injuries(
        self, sport: str, force_refresh: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch league-wide injury report grouped by team abbreviation."""
        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]
        cache_key = f"espn:injuries:{sport_u}"
        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached is not None:
                return cached

        url = f"{self.SITE_BASE}/{sp}/{lg}/injuries"
        injuries_by_team: dict[str, list[dict[str, Any]]] = {}
        try:
            with httpx.Client(timeout=10.0, headers=HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()

            for team_block in data.get("injuries", []):
                team_info = team_block.get("team", {})
                disp = str(
                    team_block.get("displayName")
                    or team_info.get("displayName")
                    or ""
                ).strip().lower()
                raw_abbr = str(
                    team_info.get("abbreviation")
                    or team_block.get("abbreviation")
                    or TEAM_NAME_TO_ABBR.get(disp, "")
                )
                abbr = self._normalize_abbr(sport_u, raw_abbr) if raw_abbr else ""
                items = []
                for inj in team_block.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    pos = athlete.get("position", {})
                    status = str(inj.get("status", "")).strip()
                    items.append(
                        {
                            "player_name": athlete.get("displayName", "Unknown"),
                            "position": pos.get("abbreviation", ""),
                            "status": status,
                            "comment": inj.get("shortComment", ""),
                        }
                    )
                if abbr and items:
                    injuries_by_team[abbr] = items

            self.db.set_cache(cache_key, injuries_by_team, ttl_seconds=600)
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale or {}
        return injuries_by_team

    def _parse_event(self, sport: str, ev: dict[str, Any]) -> dict[str, Any] | None:
        comps = ev.get("competitions", [])
        if not comps:
            return None
        comp = comps[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            return None

        home_team = home.get("team", {})
        away_team = away.get("team", {})
        home_abbr = self._normalize_abbr(sport, str(home_team.get("abbreviation", "")))
        away_abbr = self._normalize_abbr(sport, str(away_team.get("abbreviation", "")))

        status_obj = comp.get("status", {}) or {}
        status_type = status_obj.get("type", {}) or {}
        state = str(status_type.get("state", "pre")).upper()
        status_name = str(status_type.get("name", "STATUS_SCHEDULED"))
        status_detail = str(
            status_type.get("shortDetail")
            or status_type.get("detail")
            or status_name
        )
        period = int(status_obj.get("period", 0) or 0)
        clock_seconds = float(status_obj.get("clock", 0.0) or 0.0)
        display_clock = str(status_obj.get("displayClock", "0:00") or "0:00")

        try:
            home_score = int(float(home.get("score", 0) or 0))
        except (TypeError, ValueError):
            home_score = 0
        try:
            away_score = int(float(away.get("score", 0) or 0))
        except (TypeError, ValueError):
            away_score = 0

        elapsed_frac, rem_mins = compute_game_clock_progress(
            sport=sport,
            state=state,
            period=period,
            clock_seconds=clock_seconds,
            status_detail=status_detail,
        )

        # Live situation (possession, red zone, play-by-play win probability)
        situation = comp.get("situation") or {}
        last_play_prob = (
            situation.get("lastPlay", {}).get("probability", {})
            if isinstance(situation, dict)
            else {}
        )
        live_win_prob: float | None = None
        if isinstance(last_play_prob, dict) and last_play_prob.get("homeWinPercentage") is not None:
            try:
                live_win_prob = float(last_play_prob["homeWinPercentage"])
            except (TypeError, ValueError):
                live_win_prob = None

        poss_id = str(situation.get("possession", "")) if isinstance(situation, dict) else ""
        possession_team: str | None = None
        if poss_id:
            if poss_id == str(home.get("id") or home_team.get("id")):
                possession_team = home_abbr
            elif poss_id == str(away.get("id") or away_team.get("id")):
                possession_team = away_abbr

        is_red_zone = bool(situation.get("isRedZone", False)) if isinstance(situation, dict) else False
        down_distance = str(situation.get("downDistanceText", "") or "") if isinstance(situation, dict) else ""

        venue = comp.get("venue", {}) or {}
        indoor = bool(venue.get("indoor", False))
        weather = ev.get("weather", {}) or {}
        wind_mph = 0.0
        if not indoor and isinstance(weather, dict):
            w_raw = weather.get("windSpeed") or weather.get("gust")
            if w_raw is not None:
                try:
                    wind_mph = float(w_raw)
                except (TypeError, ValueError):
                    wind_mph = 0.0
            elif weather.get("displayValue"):
                m = re.search(r"(\d+(?:\.\d+)?)\s*mph", str(weather["displayValue"]), re.I)
                if m:
                    wind_mph = float(m.group(1))
        temp_f = float(weather.get("temperature", 68.0) or 68.0) if isinstance(weather, dict) else 68.0

        odds_list = comp.get("odds", []) or []
        market_info = self._extract_embedded_odds(odds_list, home_abbr)

        return {
            "event_id": str(ev.get("id")),
            "sport": sport,
            "event_name": f"{away_team.get('displayName', away_abbr)} @ {home_team.get('displayName', home_abbr)}",
            "short_name": f"{away_abbr} @ {home_abbr}",
            "start_time": str(ev.get("date", "")),
            "state": state,
            "status": status_name,
            "status_detail": status_detail,
            "period": period,
            "clock_seconds": clock_seconds,
            "display_clock": display_clock,
            "elapsed_fraction": elapsed_frac,
            "remaining_minutes": rem_mins,
            "home_score": home_score,
            "away_score": away_score,
            "live_win_prob": live_win_prob,
            "possession_team": possession_team,
            "is_red_zone": is_red_zone,
            "down_distance": down_distance,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_name": home_team.get("displayName", home_abbr),
            "away_name": away_team.get("displayName", away_abbr),
            "indoor": indoor,
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "market": market_info,
        }

    @classmethod
    def _extract_embedded_odds(
        cls, odds_list: list[dict[str, Any]], home_abbr: str
    ) -> dict[str, Any]:
        """
        Extract embedded DraftKings / ESPN sportsbook lines & American odds, supporting both
        the 2026 nested (`moneyline`, `pointSpread`, `total` with `open` & `close`) schema and
        legacy top-level fields.
        """
        if not odds_list:
            return {}
        # Prefer DraftKings provider if multiple books appear
        o = next(
            (
                item
                for item in odds_list
                if isinstance(item, dict)
                and "draft" in str((item.get("provider") or {}).get("name", "")).lower()
            ),
            odds_list[0],
        )
        if not isinstance(o, dict):
            return {}

        provider_name = str((o.get("provider") or {}).get("displayName") or (o.get("provider") or {}).get("name") or "ESPN/DK")
        out: dict[str, Any] = {"line_source": provider_name}

        # 1. Moneyline (2026 nested `moneyline.home.close.odds` + legacy `homeTeamOdds.moneyLine`)
        ml_block = o.get("moneyline") or {}
        h_ml = _parse_american_str(
            ((ml_block.get("home") or {}).get("close") or {}).get("odds")
        )
        a_ml = _parse_american_str(
            ((ml_block.get("away") or {}).get("close") or {}).get("odds")
        )
        if h_ml is None or a_ml is None:
            h_ml = _parse_american_str((o.get("homeTeamOdds") or {}).get("moneyLine"))
            a_ml = _parse_american_str((o.get("awayTeamOdds") or {}).get("moneyLine"))

        if h_ml is not None and a_ml is not None:
            ph, _ = devig_american_pair(h_ml, a_ml)
            out["home_win_prob"] = round(ph, 4)
            out["home_ml"] = int(round(h_ml))
            out["away_ml"] = int(round(a_ml))
            out["dk_home_ml"] = int(round(h_ml))
            out["dk_away_ml"] = int(round(a_ml))

        h_ml_open = _parse_american_str(
            ((ml_block.get("home") or {}).get("open") or {}).get("odds")
        )
        a_ml_open = _parse_american_str(
            ((ml_block.get("away") or {}).get("open") or {}).get("odds")
        )
        if h_ml_open is not None and a_ml_open is not None:
            ph_open, _ = devig_american_pair(h_ml_open, a_ml_open)
            out["open_home_win_prob"] = round(ph_open, 4)

        # 2. Point Spread (2026 nested `pointSpread.home.close` + legacy `details`)
        ps_block = o.get("pointSpread") or {}
        h_sp_close = (ps_block.get("home") or {}).get("close") or {}
        a_sp_close = (ps_block.get("away") or {}).get("close") or {}
        h_sp_line = _parse_line_number(h_sp_close.get("line"))
        h_sp_odds = _parse_american_str(h_sp_close.get("odds"))
        a_sp_odds = _parse_american_str(a_sp_close.get("odds"))

        if h_sp_line is not None:
            out["home_spread"] = h_sp_line
            out["dk_home_spread"] = h_sp_line
            if h_sp_odds is not None and a_sp_odds is not None:
                pcov, _ = devig_american_pair(h_sp_odds, a_sp_odds)
                out["home_cover_prob"] = round(pcov, 4)
                out["dk_home_spread_odds"] = int(round(h_sp_odds))
                out["dk_away_spread_odds"] = int(round(a_sp_odds))
            else:
                out["home_cover_prob"] = 0.50
        else:
            details = str(o.get("details", "")).strip()
            if details and details.upper() not in ("EVEN", "PK"):
                parts = details.split()
                if len(parts) == 2:
                    fav_abbr = cls._normalize_abbr("NFL", parts[0])
                    val = _parse_line_number(parts[1])
                    if val is not None:
                        out["home_spread"] = -abs(val) if fav_abbr in (home_abbr, parts[0].upper()) and parts[0].upper() == home_abbr else (
                            -abs(val) if fav_abbr == home_abbr else abs(val)
                        )
                        out["home_cover_prob"] = 0.50
            elif details.upper() in ("EVEN", "PK"):
                out["home_spread"] = 0.0
                out["home_cover_prob"] = 0.50

        h_sp_open = _parse_line_number(((ps_block.get("home") or {}).get("open") or {}).get("line"))
        if h_sp_open is not None and "home_spread" in out:
            out["open_home_spread"] = h_sp_open
            out["spread_move"] = round(float(out["home_spread"]) - h_sp_open, 2)

        # 3. Total Over/Under (2026 nested `total.over.close` + legacy `overUnder`)
        tot_block = o.get("total") or {}
        ov_close = (tot_block.get("over") or {}).get("close") or {}
        un_close = (tot_block.get("under") or {}).get("close") or {}
        tot_line = _parse_line_number(ov_close.get("line"))
        ov_odds = _parse_american_str(ov_close.get("odds"))
        un_odds = _parse_american_str(un_close.get("odds"))

        if tot_line is None and o.get("overUnder") is not None:
            tot_line = _parse_line_number(o.get("overUnder"))
            ov_odds = _parse_american_str(o.get("overOdds")) or -110.0
            un_odds = _parse_american_str(o.get("underOdds")) or -110.0

        if tot_line is not None:
            out["total_line"] = tot_line
            out["dk_total_line"] = tot_line
            if ov_odds is not None and un_odds is not None:
                po, _ = devig_american_pair(ov_odds, un_odds)
                out["over_prob"] = round(po, 4)
                out["dk_over_odds"] = int(round(ov_odds))
                out["dk_under_odds"] = int(round(un_odds))
            else:
                out["over_prob"] = 0.50

        tot_open = _parse_line_number(((tot_block.get("over") or {}).get("open") or {}).get("line"))
        if tot_open is not None and "total_line" in out:
            out["open_total_line"] = tot_open
            out["total_move"] = round(float(out["total_line"]) - tot_open, 2)

        return out

    @staticmethod
    def _normalize_abbr(sport: str, abbr: str) -> str:
        ab = abbr.upper().strip()
        if sport.upper() == "NFL":
            mapping = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}
            return mapping.get(ab, ab)
        if sport.upper() == "NBA":
            mapping = {
                "GS": "GSW",
                "NO": "NOP",
                "NY": "NYK",
                "SA": "SAS",
                "UTAH": "UTA",
                "WSH": "WAS",
            }
            return mapping.get(ab, ab)
        return ab

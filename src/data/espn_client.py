"""Public ESPN JSON API client for NFL & NBA scoreboards, live game state, box scores, DraftKings odds, and injuries."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.data.odds_client import devig_american_pair, http_get
from src.engine.dk_math import normalize_player_name
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

ESPN_PROP_TYPE_MAP_NBA = {
    "1": "points",
    "2": "assists",
    "3": "rebounds",
    "4": "fg3m",
    "90": "pra",
}

ESPN_PROP_TYPE_MAP_NFL = {
    "8": "passing_yards",
    "10": "passing_tds",
    "12": "rushing_yards",
    "13": "receiving_yards",
    "14": "receptions",
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

# NFL home stadiums: (latitude, longitude, is_indoor_or_retractable_dome)
NFL_STADIUMS: dict[str, tuple[float, float, bool]] = {
    "ARI": (33.5276, -112.2626, True),
    "ATL": (33.7554, -84.4008, True),
    "BAL": (39.2780, -76.6227, False),
    "BUF": (42.7738, -78.7870, False),
    "CAR": (35.2258, -80.8528, False),
    "CHI": (41.8623, -87.6167, False),
    "CIN": (39.0954, -84.5160, False),
    "CLE": (41.5061, -81.6995, False),
    "DAL": (32.7473, -97.0945, True),
    "DEN": (39.7439, -105.0201, False),
    "DET": (42.3400, -83.0456, True),
    "GB": (44.5013, -88.0622, False),
    "HOU": (29.6847, -95.4107, True),
    "IND": (39.7601, -86.1639, True),
    "JAX": (30.3239, -81.6373, False),
    "KC": (39.0489, -94.4839, False),
    "LA": (33.9535, -118.3392, True),
    "LAR": (33.9535, -118.3392, True),
    "LAC": (33.9535, -118.3392, True),
    "LV": (36.0909, -115.1833, True),
    "MIA": (25.9580, -80.2389, False),
    "MIN": (44.9736, -93.2575, True),
    "NE": (42.0909, -71.2643, False),
    "NO": (29.9511, -90.0812, True),
    "NYG": (40.8128, -74.0742, False),
    "NYJ": (40.8128, -74.0742, False),
    "PHI": (39.9008, -75.1675, False),
    "PIT": (40.4468, -80.0158, False),
    "SEA": (47.5952, -122.3316, False),
    "SF": (37.4033, -121.9694, False),
    "TB": (27.9759, -82.5033, False),
    "TEN": (36.1665, -86.7713, False),
    "WAS": (38.9076, -76.8645, False),
}


def _wmo_code_to_condition(code: int | None) -> str:
    """Convert an Open-Meteo WMO weather code into a concise plain-English condition label."""
    if code is None:
        return ""
    if code in (0, 1):
        return "Clear"
    if code in (2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 61, 80):
        return "Light Rain"
    if code in (63, 81):
        return "Rain"
    if code in (65, 82):
        return "Heavy Rain"
    if code in (56, 57, 66, 67):
        return "Freezing Rain"
    if code in (71, 77, 85):
        return "Light Snow"
    if code in (73, 75, 86):
        return "Snow"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return ""


def _parse_american_str(val: Any) -> float | None:
    """Parse American odds values like '-118', '+114', 'EVEN', or numeric -110."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        parsed = float(val)
    else:
        s = str(val).strip().upper()
        if not s or s in ("N/A", "-", "OFF"):
            return None
        if s in ("EVEN", "EV", "PK"):
            return 100.0
        try:
            parsed = float(s)
        except ValueError:
            return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    if parsed == 0.0:
        return 100.0
    # |odds| < 100 is not a valid American price and would yield nonsense implied probabilities
    return parsed if abs(parsed) >= 100.0 else None


def _safe_float(val: Any, default: float) -> float:
    """float() that tolerates None / '' / malformed strings from loosely-typed ESPN JSON."""
    if val is None or val == "":
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) or math.isinf(f) else f


def _deserialize_props(cached: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Rebuild `(player_key, stat_col)` prop dict (incl. normalized-name aliases) from cache JSON."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for k, v in (cached or {}).items():
        if "::" not in k:
            continue
        p_low, col = k.split("::", 1)
        out[(p_low, col)] = v
        norm_p = normalize_player_name(p_low)
        if norm_p:
            out.setdefault((norm_p, col), v)
    return out


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


_INJURY_OUT_STATUS_TOKENS = (
    "out",
    "injured reserve",
    "suspen",
    "pup",
    "nfi",
    "inactive",
    "season-ending",
)
_INJURY_CLEAR_RE = re.compile(
    r"\b(?:cleared to play|will play|expected to play|available to play|"
    r"off the injury report|not on the injury report|no injury designation|"
    r"active for (?:today|tonight|sunday|monday|tuesday|wednesday|thursday|friday|saturday))\b",
    re.I,
)
_INJURY_COMMENT_OUT_RE = re.compile(
    r"\b(?:has been ruled out|was ruled out|is ruled out|ruled out|"
    r"will not play|won't play|will sit out|sitting out|will miss|expected to miss|"
    r"sidelined|placed on injured reserve|placed on ir|season-ending|"
    r"out for (?:the )?(?:season|year|game|today|tonight|sunday|monday|tuesday|wednesday|thursday|friday|saturday)|"
    r"did not travel|will not travel|not expected to play|unlikely to play|will not return)\b",
    re.I,
)
_INJURY_COMMENT_QUESTIONABLE_RE = re.compile(
    r"\b(?:questionable|game-time decision|gtd|did not practice|won't practice|will not practice)\b",
    re.I,
)
_PROMOTED_STARTER_COMMENT_RE = re.compile(
    r"\b(?:preparing to start|expected to start|slated to start|set to start|"
    r"will start|named (?:the )?starter|starting (?:qb|quarterback|rb|running back|"
    r"wr|wide receiver|te|tight end|center|point guard|guard|forward|in place of)|"
    r"in line to start|step(?:ping)? into the starting|draw(?:ing)? the start|"
    r"getting the start|make (?:his|the) start)\b",
    re.I,
)


def classify_injury_status(item: dict[str, Any]) -> str:
    """
    Classify an ESPN injury record into 'OUT', 'DOUBTFUL', 'QUESTIONABLE', or 'ACTIVE'
    using BOTH the structured status field (including IR, Suspension, Day-To-Day) AND
    ESPN's `shortComment` news text (catching late scratches before status flips).
    """
    st = str(item.get("status", "") or "").strip().lower()
    comment = str(item.get("comment", "") or "").strip()
    c_low = comment.lower()

    if any(tok in st for tok in _INJURY_OUT_STATUS_TOKENS) or re.search(r"\bir\b", st):
        return "OUT"
    if "doubtful" in st:
        return "DOUBTFUL"

    base = (
        "QUESTIONABLE"
        if any(tok in st for tok in ("questionable", "day-to-day", "day to day", "gtd", "game-time"))
        else "ACTIVE"
    )

    if comment and not _INJURY_CLEAR_RE.search(comment):
        if _INJURY_COMMENT_OUT_RE.search(comment) and "not ruled out" not in c_low:
            return "OUT"
        if base == "ACTIVE":
            if "doubtful" in c_low:
                return "DOUBTFUL"
            if _INJURY_COMMENT_QUESTIONABLE_RE.search(comment):
                return "QUESTIONABLE"

    return base


def is_promoted_starter_comment(item: dict[str, Any]) -> bool:
    """Return True when an active player's ESPN news note explicitly indicates they are starting."""
    if classify_injury_status(item) in ("OUT", "DOUBTFUL"):
        return False
    comment = str(item.get("comment", "") or "").strip()
    if not comment:
        return False
    return bool(_PROMOTED_STARTER_COMMENT_RE.search(comment))


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
        play-by-play win probabilities, and embedded DraftKings odds across upcoming slates.
        """
        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]

        dates_to_check: list[str | None]
        if sport_u == "NFL" and target_date is None:
            dates_to_check = [None]
        elif target_date:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            dates_to_check = [dt.strftime("%Y%m%d")]
        else:
            # Anchor to US Eastern sports day (-4h so late West Coast games stay on tonight's slate)
            now_sports = datetime.now(ZoneInfo("America/New_York")) - timedelta(hours=4)
            dates_to_check = [
                (now_sports + timedelta(days=offset)).strftime("%Y%m%d")
                for offset in (0, 1, 2)
            ]

        normalized_events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def _load_date(d_param: str | None) -> dict[str, Any] | None:
            cache_key = f"espn:scoreboard:{sport_u}:{d_param or 'current_week'}"
            payload = None if force_refresh else self.db.get_cache(cache_key)
            if payload is None:
                url = f"{self.SITE_BASE}/{sp}/{lg}/scoreboard"
                params = {"dates": d_param} if d_param else {}
                try:
                    resp = http_get(url, params=params, headers=HEADERS, timeout=10.0)
                    payload = resp.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Unexpected ESPN scoreboard payload")
                    has_live = any(
                        str(
                            ((ev.get("competitions") or [{}])[0].get("status") or {})
                            .get("type", {})
                            .get("state", "")
                        ).lower()
                        == "in"
                        for ev in payload.get("events") or []
                        if isinstance(ev, dict) and ev.get("competitions")
                    )
                    # Keep short 45s TTL while games are actively in progress; 300s otherwise
                    ttl = 45 if has_live else 300
                    self.db.set_cache(cache_key, payload, ttl_seconds=ttl)
                except Exception:
                    payload, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return payload if isinstance(payload, dict) else None

        # Multi-day NBA slates: fetch the dates concurrently (results kept in date order)
        if len(dates_to_check) > 1:
            with ThreadPoolExecutor(max_workers=len(dates_to_check)) as pool:
                payloads = list(pool.map(_load_date, dates_to_check))
        else:
            payloads = [_load_date(d) for d in dates_to_check]

        for payload in payloads:
            if not payload:
                continue

            for ev in payload.get("events") or []:
                if not isinstance(ev, dict):
                    continue
                ev_id = str(ev.get("id", ""))
                if not ev_id or ev_id in seen_ids:
                    continue
                try:
                    parsed = self._parse_event(sport_u, ev)
                except (TypeError, ValueError, AttributeError, KeyError, IndexError):
                    # One malformed event must not drop the entire slate
                    parsed = None
                if parsed is not None:
                    seen_ids.add(ev_id)
                    normalized_events.append(parsed)

        # Enrich outdoor NFL games missing wind/condition data using free kickoff weather forecasts
        if sport_u == "NFL":
            need_wx = [
                ev
                for ev in normalized_events
                if not ev.get("indoor", False)
                and ev.get("state") in ("PRE", "IN")
                and (not ev.get("weather_condition") or float(ev.get("wind_mph") or 0.0) <= 0.0)
            ]
            if need_wx:
                def _enrich_wx(ev_item: dict[str, Any]) -> None:
                    fc = self.fetch_nfl_weather_forecast(
                        str(ev_item.get("home_abbr", "")),
                        str(ev_item.get("start_time", "")),
                        force_refresh=force_refresh,
                    )
                    if not fc:
                        return
                    if not ev_item.get("has_espn_temp") and fc.get("temp_f") is not None:
                        ev_item["temp_f"] = fc["temp_f"]
                    if float(ev_item.get("wind_mph") or 0.0) <= 0.0 and fc.get("wind_mph") is not None:
                        ev_item["wind_mph"] = fc["wind_mph"]
                    if not ev_item.get("weather_condition") and fc.get("weather_condition"):
                        ev_item["weather_condition"] = fc["weather_condition"]
                    if float(ev_item.get("precip_prob") or 0.0) <= 0.0 and fc.get("precip_prob") is not None:
                        ev_item["precip_prob"] = fc["precip_prob"]

                with ThreadPoolExecutor(max_workers=min(6, len(need_wx))) as pool:
                    list(pool.map(_enrich_wx, need_wx))

        return normalized_events

    def fetch_nfl_weather_forecast(
        self, home_abbr: str, start_time: str, force_refresh: bool = False
    ) -> dict[str, Any] | None:
        """
        Fetch kickoff-hour temperature (°F), wind speed (mph), precipitation probability,
        and weather condition for an outdoor NFL stadium via Open-Meteo (zero API key).
        Cached for 3 hours; returns None on dome stadiums, distant dates, or network failure.
        """
        stadium = NFL_STADIUMS.get(str(home_abbr or "").upper())
        if not stadium or stadium[2]:
            return None
        lat, lon, _ = stadium

        if not start_time:
            return None
        try:
            ko_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
            if ko_dt.tzinfo is None:
                ko_dt = ko_dt.replace(tzinfo=timezone.utc)
            ko_utc = ko_dt.astimezone(timezone.utc)
        except ValueError:
            return None

        now_utc = datetime.now(timezone.utc)
        hrs_diff = (ko_utc - now_utc).total_seconds() / 3600.0
        if hrs_diff < -4.0 or hrs_diff > 168.0:
            return None

        target_hour = ko_utc.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
        cache_key = f"wx:nfl:{home_abbr.upper()}:{target_hour}"
        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if isinstance(cached, dict):
                return cached

        try:
            resp = http_get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "temperature_2m,wind_speed_10m,precipitation_probability,weather_code",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "timezone": "UTC",
                    "forecast_days": 8,
                },
                headers=HEADERS,
                timeout=3.5,
            )
            data = resp.json()
            hourly = data.get("hourly") if isinstance(data, dict) else None
            if not isinstance(hourly, dict):
                return None
            times = hourly.get("time") or []
            if target_hour not in times:
                return None
            idx = times.index(target_hour)
            temps = hourly.get("temperature_2m") or []
            winds = hourly.get("wind_speed_10m") or []
            pprobs = hourly.get("precipitation_probability") or []
            codes = hourly.get("weather_code") or []

            temp_f = _safe_float(temps[idx], 65.0) if idx < len(temps) else None
            wind_mph = _safe_float(winds[idx], 0.0) if idx < len(winds) else None
            precip_prob = _safe_float(pprobs[idx], 0.0) if idx < len(pprobs) else 0.0
            code = int(codes[idx]) if idx < len(codes) and codes[idx] is not None else None

            out = {
                "temp_f": round(float(temp_f), 1) if temp_f is not None else None,
                "wind_mph": round(max(0.0, float(wind_mph)), 1) if wind_mph is not None else None,
                "precip_prob": round(max(0.0, min(100.0, float(precip_prob))), 1),
                "weather_condition": _wmo_code_to_condition(code),
            }
            self.db.set_cache(cache_key, out, ttl_seconds=10800)
            return out
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale if isinstance(stale, dict) else None

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
            resp = http_get(url, params={"event": event_id}, headers=HEADERS, timeout=8.5)
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

                        entry_obj = {
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
                        players_map[pname.lower()] = entry_obj
                        norm_nm = normalize_player_name(pname)
                        if norm_nm:
                            players_map.setdefault(norm_nm, entry_obj)
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
                        norm_nm = normalize_player_name(pname)
                        if norm_nm:
                            players_map.setdefault(norm_nm, entry)

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
            resp = http_get(url, headers=HEADERS, timeout=10.0)
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

            self.db.set_cache(cache_key, injuries_by_team, ttl_seconds=300)
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale or {}
        return injuries_by_team

    def fetch_team_roster(
        self, sport: str, team_id: str, force_refresh: bool = False
    ) -> dict[str, str]:
        """
        Fetch ESPN team roster mapping `{athlete_id: displayName}` for NBA (flat list) or NFL (grouped).
        Cached for 12 hours.
        """
        tid = str(team_id or "").strip()
        if not tid or not tid.isdigit():
            return {}
        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]
        cache_key = f"espn:roster:{sport_u}:{tid}"
        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached is not None:
                return cached

        url = f"{self.SITE_BASE}/{sp}/{lg}/teams/{tid}/roster"
        roster: dict[str, str] = {}
        try:
            resp = http_get(url, headers=HEADERS, timeout=8.5)
            data = resp.json()
            for entry in data.get("athletes") or []:
                if not isinstance(entry, dict):
                    continue
                sub_items = entry.get("items")
                ath_list = sub_items if isinstance(sub_items, list) else [entry]
                for ath in ath_list:
                    if not isinstance(ath, dict):
                        continue
                    aid = str(ath.get("id", "")).strip()
                    dname = str(ath.get("displayName") or ath.get("fullName") or "").strip()
                    if aid and dname:
                        roster[aid] = dname
            if roster:
                self.db.set_cache(cache_key, roster, ttl_seconds=43200)
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            return stale or {}
        return roster

    def fetch_athlete_name(self, sport: str, athlete_id: str) -> str:
        """Single-athlete lookup fallback for recently traded players not yet on cached team roster."""
        aid = str(athlete_id or "").strip()
        if not aid or not aid.isdigit():
            return ""
        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]
        cache_key = f"espn:athlete:{sport_u}:{aid}"
        cached = self.db.get_cache(cache_key)
        if isinstance(cached, str):
            # "" is a short-lived negative-cache entry for unknown athlete IDs
            return cached

        url = f"{self.CORE_BASE}/{sp}/leagues/{lg}/athletes/{aid}"
        try:
            resp = http_get(url, headers=HEADERS, timeout=6.0, raise_for_status=False)
            if resp.status_code == 404:
                self.db.set_cache(cache_key, "", ttl_seconds=6 * 3600)
                return ""
            resp.raise_for_status()
            data = resp.json()
            name = str(data.get("displayName") or data.get("fullName") or "").strip()
            if name:
                self.db.set_cache(cache_key, name, ttl_seconds=86400 * 7)
            return name
        except Exception:
            return ""

    @staticmethod
    def _match_espn_prop_col(sport_u: str, type_obj: dict[str, Any]) -> str | None:
        prop_map = (
            ESPN_PROP_TYPE_MAP_NBA if sport_u == "NBA" else ESPN_PROP_TYPE_MAP_NFL
        )
        type_id = str(type_obj.get("id", "")).strip()
        col = prop_map.get(type_id)
        if col:
            return col
        tname = str(type_obj.get("name", "")).strip().lower()
        if not tname:
            return None
        # Exclude alternate, milestone ladders, quarter/half, longest, and specialty/prop-novelty markets
        if any(
            bad in tname
            for bad in (
                "alternate",
                "alt ",
                "milestone",
                "quarter",
                "1st ",
                "2nd ",
                "3rd ",
                "4th ",
                "half",
                "longest",
                "first ",
                "last ",
                "anytime",
                "exact",
                "team ",
                "to record",
                "to score",
                "to throw",
                "to rush",
                "to catch",
                "to have",
                "double",
                "triple",
            )
        ):
            return None
        if sport_u == "NBA":
            if tname in ("total points, rebounds, and assists", "points + rebounds + assists"):
                return "pra"
            if tname == "total points":
                return "points"
            if tname == "total rebounds":
                return "rebounds"
            if tname == "total assists":
                return "assists"
            if ("3-point" in tname or "three point" in tname) and "plus" not in tname and "+" not in tname:
                return "fg3m"
            return None
        # NFL flexible matching (handles "Total Passing Yards", "Passing Yards (Incl. Overtime)", etc.)
        # Reject all combo markets ("+", " plus ", " and ", "rush + rec", etc.)
        if "+" in tname or " plus " in tname or " and " in tname or "rush + rec" in tname or "combo" in tname:
            return None
        if "passing yards" in tname or "pass yards" in tname:
            return "passing_yards"
        if "passing touchdown" in tname or "pass touchdown" in tname or "passing td" in tname:
            return "passing_tds"
        if "rushing yards" in tname or "rush yards" in tname:
            return "rushing_yards"
        if "receiving yards" in tname or "reception yards" in tname:
            return "receiving_yards"
        if "receptions" in tname and "yards" not in tname and "touchdown" not in tname:
            return "receptions"
        return None

    @staticmethod
    def extract_espn_prop_lines(
        sport: str,
        items: list[dict[str, Any]],
        athlete_names: dict[str, str],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Parse ESPN Core API DraftKings (`odds/100/propBets`) items into `(player_key, stat_col)` dicts.
        Each `(athlete_id, col, line)` group emits Over first and Under second in feed order.
        Milestone ladders (e.g. `150+` displayValue or `Milestones` market types) are strictly excluded.
        """
        sport_u = sport.upper()
        prop_map = ESPN_PROP_TYPE_MAP_NBA if sport_u == "NBA" else ESPN_PROP_TYPE_MAP_NFL
        # Map `(aid, col, line_val)` -> (is_exact_type_id, list of american prices)
        grouped: dict[tuple[str, str, float], list[float | None]] = {}
        exact_type_flag: dict[tuple[str, str, float], bool] = {}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            type_obj = item.get("type") or {}
            col = ESPNClient._match_espn_prop_col(sport_u, type_obj)
            if not col:
                continue
            ath_obj = item.get("athlete") or {}
            aid = str(ath_obj.get("id", "")).strip()
            if not aid:
                ref = str(ath_obj.get("$ref", ""))
                m = re.search(r"/athletes/(\d+)", ref)
                if m:
                    aid = m.group(1)
            if not aid:
                continue

            cur_target_obj = (item.get("current") or {}).get("target") or {}
            cur_disp = str(cur_target_obj.get("displayValue") or "").strip()
            # One-way milestone targets ("150+", "10+", etc.) are never two-way Over/Under lines
            if cur_disp.endswith("+"):
                continue

            cur_target = cur_target_obj.get("value")
            odds_obj = item.get("odds") or {}
            tot_field = odds_obj.get("total")
            tot_val = tot_field.get("value") if isinstance(tot_field, dict) else tot_field
            if isinstance(tot_field, dict) and str(tot_field.get("displayValue") or "").strip().endswith("+"):
                continue
            line_val = _parse_line_number(cur_target if cur_target is not None else tot_val)
            if line_val is None or line_val <= 0.0:
                continue

            am_field = odds_obj.get("american")
            am_raw = am_field.get("value") if isinstance(am_field, dict) else am_field
            am_px = _parse_american_str(am_raw)
            gkey = (aid, col, round(line_val, 2))
            grouped.setdefault(gkey, []).append(am_px)
            is_exact = str(type_obj.get("id", "")).strip() in prop_map
            exact_type_flag[gkey] = exact_type_flag.get(gkey, False) or is_exact

        # Rank key: (prefer real two-way priced lines, prefer exact mapped O/U type_id, lower imbalance)
        best_by_player_col: dict[tuple[str, str], tuple[tuple[int, int, float], dict[str, Any]]] = {}
        for (aid, col, line_val), prices in grouped.items():
            pname = athlete_names.get(aid, "").strip()
            if not pname:
                continue
            over_prob: float | None = None
            dk_over_prob: float | None = None
            dk_over_odds: int | None = None
            dk_under_odds: int | None = None
            imbalance = 0.25
            has_two_way_odds = False
            if len(prices) >= 2 and prices[0] is not None and prices[1] is not None:
                po, _ = devig_american_pair(float(prices[0]), float(prices[1]))
                over_prob = round(po, 4)
                dk_over_prob = round(po, 4)
                dk_over_odds = int(round(float(prices[0])))
                dk_under_odds = int(round(float(prices[1])))
                imbalance = abs(po - 0.50)
                has_two_way_odds = True

            candidate: dict[str, Any] = {
                "line": line_val,
                "over_prob": over_prob,
                "dk_over_prob": dk_over_prob,
                "source": "DraftKings",
            }
            if dk_over_odds is not None and dk_under_odds is not None:
                candidate["dk_line"] = line_val
                candidate["dk_over_odds"] = dk_over_odds
                candidate["dk_under_odds"] = dk_under_odds
                candidate["books"] = {
                    "draftkings": {
                        "line": line_val,
                        "over_odds": dk_over_odds,
                        "under_odds": dk_under_odds,
                        "over_prob": dk_over_prob,
                    }
                }
            else:
                # Real DraftKings line from ESPN without explicit American juice (common on NFL propBets):
                # attach realistic -115/-115 prop reference odds; Odds API overrides with exact live juice when present.
                candidate["dk_line"] = line_val
                candidate["books"] = {
                    "draftkings": {
                        "line": line_val,
                        "over_odds": -115,
                        "under_odds": -115,
                        "over_prob": 0.50,
                        "odds_estimated": True,
                    }
                }
            rank = (
                0 if has_two_way_odds else 1,
                0 if exact_type_flag.get((aid, col, line_val), False) else 1,
                imbalance,
            )
            existing = best_by_player_col.get((aid, col))
            if existing is None or rank < existing[0]:
                best_by_player_col[(aid, col)] = (rank, candidate)

        out_props: dict[tuple[str, str], dict[str, Any]] = {}
        for (aid, col), (_, val_obj) in best_by_player_col.items():
            pname = athlete_names[aid].strip()
            p_low = pname.lower()
            out_props[(p_low, col)] = val_obj
            norm_p = normalize_player_name(pname)
            if norm_p:
                out_props.setdefault((norm_p, col), val_obj)
        return out_props

    def fetch_event_player_props(
        self,
        sport: str,
        event_id: str,
        home_id: str = "",
        away_id: str = "",
        force_refresh: bool = False,
        live_mode: bool = False,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Zero-key DraftKings player prop line & odds sync via ESPN Core API (`odds/100/propBets`).
        """
        eid = str(event_id or "").strip()
        if not eid or not eid.isdigit():
            return {}

        sport_u = sport.upper()
        sp, lg = SPORT_PATHS[sport_u]
        cache_key = f"espn:v4:props:{sport_u}:{eid}"
        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if isinstance(cached, dict):
                return _deserialize_props(cached)

        url = f"{self.CORE_BASE}/{sp}/leagues/{lg}/events/{eid}/competitions/{eid}/odds/100/propBets"
        all_items: list[dict[str, Any]] = []
        partial = False
        try:
            resp = http_get(
                url,
                params={"lang": "en", "region": "us", "limit": 1000, "page": 1},
                headers=HEADERS,
                timeout=8.5,
                raise_for_status=False,
            )
            if resp.status_code == 404:
                self.db.set_cache(cache_key, {}, ttl_seconds=90)
                return {}
            resp.raise_for_status()
            first_page = resp.json()
            all_items.extend(first_page.get("items") or [])
            page_count = int(first_page.get("pageCount", 1) or 1)
            max_pages = 10 if sport_u == "NFL" else 5
            for p_idx in range(2, min(max_pages + 1, page_count + 1)):
                try:
                    p_resp = http_get(
                        url,
                        params={"lang": "en", "region": "us", "limit": 1000, "page": p_idx},
                        headers=HEADERS,
                        timeout=8.5,
                    )
                    all_items.extend(p_resp.json().get("items") or [])
                except Exception:
                    # Keep the page(s) already fetched instead of discarding everything
                    partial = True
                    break
        except Exception:
            stale, _ = self.db.get_cache_with_age(cache_key, allow_stale=True)
            if isinstance(stale, dict) and stale:
                # Same shape as the fresh-cache path (incl. normalized-name aliases)
                return _deserialize_props(stale)
            return {}

        if not all_items:
            self.db.set_cache(cache_key, {}, ttl_seconds=90)
            return {}

        athlete_names: dict[str, str] = {}
        if home_id:
            athlete_names.update(self.fetch_team_roster(sport_u, home_id))
        if away_id:
            athlete_names.update(self.fetch_team_roster(sport_u, away_id))

        missing_aids: list[str] = []
        for item in all_items:
            if not isinstance(item, dict):
                continue
            if not self._match_espn_prop_col(sport_u, item.get("type") or {}):
                continue
            ath_obj = item.get("athlete") or {}
            aid = str(ath_obj.get("id", "")).strip()
            if not aid:
                m = re.search(r"/athletes/(\d+)", str(ath_obj.get("$ref", "")))
                aid = m.group(1) if m else ""
            if aid and aid not in athlete_names and aid not in missing_aids:
                missing_aids.append(aid)

        lookup_aids = missing_aids[:12]
        if lookup_aids:
            # Independent single-athlete lookups: run concurrently instead of N serial round-trips
            with ThreadPoolExecutor(max_workers=min(4, len(lookup_aids))) as pool:
                names = list(pool.map(lambda a: self.fetch_athlete_name(sport_u, a), lookup_aids))
            for aid, nm in zip(lookup_aids, names):
                if nm:
                    athlete_names[aid] = nm

        parsed = self.extract_espn_prop_lines(sport_u, all_items, athlete_names)
        serializable = {f"{p_low}::{col}": val for (p_low, col), val in parsed.items()}
        ttl = 90 if (live_mode or partial) else 300
        self.db.set_cache(cache_key, serializable, ttl_seconds=ttl)
        return parsed

    def fetch_slate_player_props(
        self,
        sport: str,
        events: list[dict[str, Any]],
        force_refresh: bool = False,
    ) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
        """Concurrently sync zero-key DraftKings player props across all active slate events."""
        sport_u = sport.upper()
        valid_events = [
            ev for ev in events if str(ev.get("event_id", "")).strip().isdigit()
        ]
        if not valid_events:
            return {}

        team_ids = sorted(
            {
                tid
                for ev in valid_events
                for tid in (str(ev.get("home_id", "")), str(ev.get("away_id", "")))
                if tid.isdigit()
            }
        )
        if team_ids:
            with ThreadPoolExecutor(max_workers=min(8, len(team_ids))) as pool:
                list(pool.map(lambda tid: self.fetch_team_roster(sport_u, tid), team_ids))

        out: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

        def _fetch_one(ev: dict[str, Any]) -> tuple[str, dict[tuple[str, str], dict[str, Any]]]:
            eid = str(ev["event_id"])
            props = self.fetch_event_player_props(
                sport_u,
                eid,
                home_id=str(ev.get("home_id", "")),
                away_id=str(ev.get("away_id", "")),
                force_refresh=force_refresh,
                live_mode=(ev.get("state") == "IN"),
            )
            return eid, props

        with ThreadPoolExecutor(max_workers=min(8, len(valid_events))) as pool:
            for eid, props in pool.map(_fetch_one, valid_events):
                if props:
                    out[eid] = props
        return out

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
        period = int(_safe_float(status_obj.get("period"), 0.0))
        clock_seconds = _safe_float(status_obj.get("clock"), 0.0)
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
        if "indoor" in venue:
            indoor = bool(venue.get("indoor"))
        elif sport == "NFL" and home_abbr in NFL_STADIUMS:
            indoor = bool(NFL_STADIUMS[home_abbr][2])
        else:
            indoor = sport == "NBA"
        weather = ev.get("weather") or comp.get("weather") or {}
        if not isinstance(weather, dict):
            weather = {}
        wind_mph = 0.0
        weather_condition = str(weather.get("displayValue") or "").strip() if not indoor else ""
        if not indoor:
            w_raw = weather.get("windSpeed") or weather.get("gust")
            if w_raw is not None:
                try:
                    wind_mph = float(w_raw)
                except (TypeError, ValueError):
                    wind_mph = 0.0
            elif weather_condition:
                m = re.search(r"(\d+(?:\.\d+)?)\s*mph", weather_condition, re.I)
                if m:
                    wind_mph = float(m.group(1))
        has_espn_temp = weather.get("temperature") is not None
        # NB: 0°F is a legitimate kickoff temperature; only missing/malformed values default to 68°F
        temp_f = _safe_float(weather.get("temperature"), 68.0)
        cond_low = weather_condition.lower()
        precip_prob = _safe_float(weather.get("precipitationProbability"), 0.0)
        if not indoor and precip_prob <= 0.0 and any(
            k in cond_low for k in ("rain", "snow", "sleet", "storm", "shower", "flurr")
        ):
            precip_prob = 75.0

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
            "home_id": str(home_team.get("id", "")),
            "away_id": str(away_team.get("id", "")),
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_name": home_team.get("displayName", home_abbr),
            "away_name": away_team.get("displayName", away_abbr),
            "indoor": indoor,
            "temp_f": temp_f,
            "has_espn_temp": has_espn_temp,
            "wind_mph": wind_mph,
            "weather_condition": weather_condition,
            "precip_prob": precip_prob,
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
                    raw_fav = parts[0].upper()
                    # Normalize with BOTH league maps: the NFL-only map left NBA short codes
                    # (GS, NY, SA, NO, UTAH) unmatched, flipping the home spread sign.
                    fav_is_home = home_abbr in (
                        raw_fav,
                        cls._normalize_abbr("NFL", raw_fav),
                        cls._normalize_abbr("NBA", raw_fav),
                    )
                    val = _parse_line_number(parts[1])
                    if val is not None:
                        out["home_spread"] = -abs(val) if fav_is_home else abs(val)
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

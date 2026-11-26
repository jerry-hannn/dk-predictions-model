"""NBA historical & current-season team and player game-log loader."""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.odds_client import http_get, normalize_player_name

DATA_DIR = PROJECT_ROOT / "data"

# --- mtime-keyed CSV cache ----------------------------------------------------------------
# The recommender re-loads these multi-MB CSVs on every slate request; parse once per file
# version and hand out copies (callers are free to mutate their frame).
_CSV_CACHE: dict[str, tuple[int, int, pd.DataFrame]] = {}
_CSV_CACHE_LOCK = threading.Lock()


def is_csv_stale(path: Path, max_age_hours: float = 18.0) -> bool:
    """Return True if a local historical CSV is missing or older than `max_age_hours`."""
    try:
        age_sec = time.time() - os.stat(path).st_mtime
    except OSError:
        return True
    return age_sec > max(0.0, float(max_age_hours)) * 3600.0


def read_csv_cached(path: Path) -> pd.DataFrame:
    """`pd.read_csv` memoized on (path, mtime_ns, size); returns a defensive copy."""
    key = str(Path(path).absolute())
    st = os.stat(path)
    with _CSV_CACHE_LOCK:
        hit = _CSV_CACHE.get(key)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2].copy()
    df = pd.read_csv(path)
    with _CSV_CACHE_LOCK:
        _CSV_CACHE[key] = (st.st_mtime_ns, st.st_size, df)
    return df.copy()


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write via temp file + os.replace so concurrent readers never see a half-written CSV."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    with _CSV_CACHE_LOCK:
        _CSV_CACHE.pop(str(path.absolute()), None)


def _col(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    """Column accessor whose fallback is a Series aligned to `df.index` (never a bare scalar)."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index, dtype=object if isinstance(default, str) else None)


def _parse_minutes_series(raw: pd.Series) -> pd.Series:
    """Parse NBA minutes that may be numeric (34) or clock strings ('34:12') into float minutes."""
    numeric = pd.to_numeric(raw, errors="coerce")
    as_str = raw.astype(str)
    clock_mask = numeric.isna() & as_str.str.contains(":", regex=False)
    if clock_mask.any():
        parts = as_str[clock_mask].str.split(":", n=1, expand=True)
        mins = pd.to_numeric(parts[0], errors="coerce") + pd.to_numeric(parts[1], errors="coerce") / 60.0
        numeric = numeric.copy()
        numeric[clock_mask] = mins
    return numeric.fillna(0.0)


NBA_TEAMS = [
    "BOS", "OKC", "CLE", "DEN", "NYK", "MIN", "DAL", "PHX",
    "MIL", "IND", "MEM", "HOU", "LAL", "GSW", "SAC", "MIA",
    "ORL", "LAC", "PHI", "NOP", "ATL", "SAS", "CHI", "TOR",
    "BKN", "DET", "CHA", "POR", "UTA", "WAS",
]


class NBALoader:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.games_csv = self.cache_dir / "nba_games.csv"
        self.player_logs_csv = self.cache_dir / "nba_player_logs.csv"
        self.player_positions_json = self.cache_dir / "nba_player_positions.json"

    @staticmethod
    def _normalize_nba_position(raw_pos: str) -> str:
        p = str(raw_pos or "").strip().upper()
        if not p:
            return ""
        head = p.split("-", 1)[0].split("/", 1)[0].strip()
        if head in ("PG", "SG", "G"):
            return "G"
        if head in ("SF", "PF", "F"):
            return "F"
        if head == "C":
            return "C"
        return ""

    def load_player_positions(self, force_refresh: bool = False) -> dict[str, str]:
        """
        Load cached `{normalized_player_name: 'G'|'F'|'C'}` map from disk, refreshing from
        ESPN's public NBA team rosters when requested. Never fails when offline.
        """
        if self.player_positions_json.exists() and not force_refresh:
            try:
                data = json.loads(self.player_positions_json.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return {str(k): str(v) for k, v in data.items() if v in ("G", "F", "C")}
            except Exception:
                pass

        if not force_refresh and not self.player_positions_json.exists():
            # Avoid unexpected network calls during offline unit tests unless refresh requested
            return {}

        fetched: dict[str, str] = {}
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = httpx.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=40",
                headers=headers,
                timeout=6.0,
            )
            r.raise_for_status()
            teams = r.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            tids = [str(t.get("team", {}).get("id", "")) for t in teams if t.get("team", {}).get("id")]

            def _fetch_one(tid: str) -> dict[str, str]:
                out: dict[str, str] = {}
                try:
                    resp = httpx.get(
                        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster",
                        headers=headers,
                        timeout=6.0,
                    )
                    if resp.status_code != 200:
                        return out
                    for ath in resp.json().get("athletes") or []:
                        if not isinstance(ath, dict):
                            continue
                        name = str(ath.get("displayName") or ath.get("fullName") or "").strip()
                        pos_raw = (ath.get("position") or {}).get("abbreviation", "")
                        pos = self._normalize_nba_position(pos_raw)
                        if name and pos:
                            out[name.lower()] = pos
                            norm = normalize_player_name(name)
                            if norm:
                                out[norm] = pos
                except Exception:
                    return out
                return out

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                for part in pool.map(_fetch_one, tids):
                    fetched.update(part)

            if fetched:
                tmp = self.player_positions_json.with_suffix(".tmp")
                tmp.write_text(json.dumps(fetched, sort_keys=True), encoding="utf-8")
                os.replace(tmp, self.player_positions_json)
                return fetched
        except Exception:
            pass

        if self.player_positions_json.exists():
            try:
                data = json.loads(self.player_positions_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items() if v in ("G", "F", "C")}
            except Exception:
                pass
        return {}

    @staticmethod
    def attach_player_positions(df: pd.DataFrame, pos_map: dict[str, str] | None = None) -> pd.DataFrame:
        """
        Attach a clean NBA position bucket (`G`, `F`, `C`) to each player row using cached
        roster positions with a deterministic per-36 stat-profile fallback for unmapped players.
        """
        if len(df) == 0:
            return df
        if "position" in df.columns and df["position"].isin(["G", "F", "C"]).all():
            return df

        work = df.copy()
        pmap = pos_map or {}

        # Per-36 fallback profile computed per player across active games
        active = work[work["minutes"] >= 8.0] if "minutes" in work.columns else work
        if len(active) == 0:
            active = work
        prof = active.groupby("player_name", as_index=False).agg(
            mins=("minutes", "sum"),
            reb=("rebounds", "sum"),
            ast=("assists", "sum"),
            fg3m=("fg3m", "sum"),
        )
        inferred: dict[str, str] = {}
        for _, r in prof.iterrows():
            nm = str(r["player_name"])
            known = pmap.get(nm.lower()) or pmap.get(normalize_player_name(nm))
            if known in ("G", "F", "C"):
                inferred[nm] = known
                continue
            m = max(1.0, float(r["mins"]))
            reb36 = float(r["reb"]) * 36.0 / m
            ast36 = float(r["ast"]) * 36.0 / m
            fg3_36 = float(r["fg3m"]) * 36.0 / m
            if reb36 >= 10.2 or (reb36 >= 8.6 and fg3_36 < 1.2 and ast36 < 4.5):
                inferred[nm] = "C"
            elif ast36 >= 4.2 or (fg3_36 >= 2.1 and reb36 < 6.4):
                inferred[nm] = "G"
            else:
                inferred[nm] = "F"

        work["position"] = work["player_name"].map(inferred).fillna("F")
        return work

    @staticmethod
    def _espn_nba_abbr(abbr: str) -> str:
        ab = str(abbr or "").upper().strip()
        return {
            "GS": "GSW",
            "NO": "NOP",
            "NY": "NYK",
            "SA": "SAS",
            "UTAH": "UTA",
            "WSH": "WAS",
        }.get(ab, ab)

    @staticmethod
    def _parse_made_att(raw_val: Any) -> tuple[float, float]:
        s = str(raw_val or "").strip()
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                return float(a), float(b)
            except ValueError:
                return 0.0, 0.0
        return 0.0, 0.0

    @staticmethod
    def _parse_espn_min(raw_val: Any) -> float:
        s = str(raw_val or "").strip()
        if not s or s in ("--", "DNP", "0"):
            return 0.0
        if ":" in s:
            a, b = s.split(":", 1)
            try:
                return float(a) + float(b) / 60.0
            except ValueError:
                return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    def _backfill_from_espn(
        self,
        existing_games: pd.DataFrame | None = None,
        existing_logs: pd.DataFrame | None = None,
        max_days: int = 21,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Incrementally backfill completed NBA games & player boxscores from ESPN's public
        scoreboard + summary APIs when stats.nba.com blocks/times out cloud IPs.
        """
        if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("ALLOW_PYTEST_NETWORK") != "1":
            return existing_games, existing_logs

        cur_games = existing_games.copy() if existing_games is not None and len(existing_games) > 0 else None
        cur_logs = existing_logs.copy() if existing_logs is not None and len(existing_logs) > 0 else None

        max_dates: list[str] = []
        if cur_games is not None and "gameday" in cur_games.columns:
            max_dates.append(str(cur_games["gameday"].max())[:10])
        if cur_logs is not None and "gameday" in cur_logs.columns:
            max_dates.append(str(cur_logs["gameday"].max())[:10])
        if not max_dates:
            return cur_games, cur_logs

        start_dt = datetime.strptime(min(max_dates), "%Y-%m-%d") + timedelta(days=1)
        end_dt = datetime.now()
        if start_dt.date() > end_dt.date():
            return cur_games, cur_logs

        # Bound backfill window to keep refreshes fast
        earliest = end_dt - timedelta(days=max_days)
        if start_dt < earliest:
            start_dt = earliest

        date_list: list[tuple[str, str]] = []
        d = start_dt
        while d.date() <= end_dt.date():
            date_list.append((d.strftime("%Y%m%d"), d.strftime("%Y-%m-%d")))
            d += timedelta(days=1)
        if not date_list:
            return cur_games, cur_logs

        headers = {"User-Agent": "Mozilla/5.0"}

        def _fetch_sb_day(item: tuple[str, str]) -> list[tuple[str, str, str]]:
            ymd, iso_day = item
            out_evs: list[tuple[str, str, str]] = []
            try:
                r = http_get(
                    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                    params={"dates": ymd},
                    headers=headers,
                    timeout=8.0,
                    raise_for_status=False,
                )
                if r.status_code != 200:
                    return out_evs
                for ev in r.json().get("events") or []:
                    comps = ev.get("competitions") or []
                    if not comps:
                        continue
                    comp = comps[0]
                    st_state = str(((comp.get("status") or {}).get("type") or {}).get("state", "")).lower()
                    if st_state != "post":
                        continue
                    eid = str(ev.get("id", "")).strip()
                    season_yr = int((ev.get("season") or {}).get("year") or (int(iso_day[:4]) + (1 if int(iso_day[5:7]) >= 8 else 0)))
                    season_str = f"{season_yr - 1}-{str(season_yr)[-2:]}"
                    if eid:
                        out_evs.append((eid, iso_day, season_str))
            except Exception:
                return out_evs
            return out_evs

        completed_evs: list[tuple[str, str, str]] = []
        seen_eids: set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(date_list))) as pool:
            for day_evs in pool.map(_fetch_sb_day, date_list):
                for row_ev in day_evs:
                    if row_ev[0] not in seen_eids:
                        seen_eids.add(row_ev[0])
                        completed_evs.append(row_ev)

        if not completed_evs:
            return cur_games, cur_logs

        def _fetch_summary(ev_tuple: tuple[str, str, str]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
            eid, iso_day, season_str = ev_tuple
            try:
                r = http_get(
                    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                    params={"event": eid},
                    headers=headers,
                    timeout=8.0,
                    raise_for_status=False,
                )
                if r.status_code != 200:
                    return None, []
                raw = r.json()
            except Exception:
                return None, []

            hdr_comps = (raw.get("header") or {}).get("competitions") or []
            if not hdr_comps:
                return None, []
            competitors = hdr_comps[0].get("competitors") or []
            h_comp = next((c for c in competitors if c.get("homeAway") == "home"), None)
            a_comp = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not h_comp or not a_comp:
                return None, []
            h_abbr = self._espn_nba_abbr((h_comp.get("team") or {}).get("abbreviation", ""))
            a_abbr = self._espn_nba_abbr((a_comp.get("team") or {}).get("abbreviation", ""))
            try:
                h_pts = float(h_comp.get("score"))
                a_pts = float(a_comp.get("score"))
            except (TypeError, ValueError):
                return None, []
            if h_pts == a_pts or h_pts < 65 or a_pts < 65:
                return None, []

            team_stats: dict[str, dict[str, float]] = {}
            for tb in (raw.get("boxscore") or {}).get("teams") or []:
                tabbr = self._espn_nba_abbr((tb.get("team") or {}).get("abbreviation", ""))
                smap = {str(st.get("name", "")): st.get("displayValue") for st in (tb.get("statistics") or [])}
                fgm, fga = self._parse_made_att(smap.get("fieldGoalsMade-fieldGoalsAttempted"))
                fg3m, _ = self._parse_made_att(smap.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                _, fta = self._parse_made_att(smap.get("freeThrowsMade-freeThrowsAttempted"))
                try:
                    oreb = float(smap.get("offensiveRebounds") or 10.5)
                except (TypeError, ValueError):
                    oreb = 10.5
                try:
                    tov = float(smap.get("totalTurnovers") or smap.get("turnovers") or 13.5)
                except (TypeError, ValueError):
                    tov = 13.5
                fga_eff = max(50.0, fga if fga > 0 else 88.0)
                poss = max(80.0, fga_eff + 0.44 * fta - oreb + tov)
                team_stats[tabbr] = {
                    "poss": poss,
                    "efg": round((fgm + 0.5 * fg3m) / fga_eff, 4),
                    "tov_pct": round(tov / poss, 4),
                    "oreb": oreb,
                    "ftr": round(fta / fga_eff, 4),
                }

            hs = team_stats.get(h_abbr, {"poss": 99.0, "efg": 0.54, "tov_pct": 0.134, "oreb": 10.5, "ftr": 0.245})
            as_ = team_stats.get(a_abbr, {"poss": 99.0, "efg": 0.54, "tov_pct": 0.134, "oreb": 10.5, "ftr": 0.245})
            game_row = {
                "game_id": f"ESPN_{eid}",
                "season": season_str,
                "gameday": iso_day,
                "home_team": h_abbr,
                "away_team": a_abbr,
                "home_score": h_pts,
                "away_score": a_pts,
                "pace": round(0.5 * (hs["poss"] + as_["poss"]), 2),
                "home_efg": hs["efg"],
                "away_efg": as_["efg"],
                "home_tov_pct": hs["tov_pct"],
                "away_tov_pct": as_["tov_pct"],
                "home_oreb": hs["oreb"],
                "away_oreb": as_["oreb"],
                "home_ftr": hs["ftr"],
                "away_ftr": as_["ftr"],
            }

            p_rows: list[dict[str, Any]] = []
            for team_entry in (raw.get("boxscore") or {}).get("players") or []:
                tabbr = self._espn_nba_abbr((team_entry.get("team") or {}).get("abbreviation", ""))
                opp_abbr = a_abbr if tabbr == h_abbr else h_abbr
                for sg in team_entry.get("statistics") or []:
                    labels = [str(x).upper() for x in (sg.get("labels") or [])]
                    idx = {lbl: i for i, lbl in enumerate(labels)}
                    for ath_row in sg.get("athletes") or []:
                        if ath_row.get("didNotPlay"):
                            continue
                        ath = ath_row.get("athlete") or {}
                        pname = str(ath.get("displayName", "")).strip()
                        pid = str(ath.get("id", "") or pname).strip()
                        stats = ath_row.get("stats") or []
                        if not pname or not stats:
                            continue

                        def _st(lbl: str, _s: list[Any] = stats, _m: dict[str, int] = idx) -> str:
                            i = _m.get(lbl)
                            return str(_s[i]) if i is not None and i < len(_s) else "0"

                        mins = self._parse_espn_min(_st("MIN"))
                        if mins <= 0.0:
                            continue
                        try:
                            pts = float(_st("PTS"))
                        except ValueError:
                            pts = 0.0
                        try:
                            reb = float(_st("REB"))
                        except ValueError:
                            reb = 0.0
                        try:
                            ast = float(_st("AST"))
                        except ValueError:
                            ast = 0.0
                        fg3m, _ = self._parse_made_att(_st("3PT"))
                        _, fga_p = self._parse_made_att(_st("FG"))
                        _, fta_p = self._parse_made_att(_st("FT"))
                        p_rows.append(
                            {
                                "player_id": pid,
                                "player_name": pname,
                                "team": tabbr,
                                "opponent": opp_abbr,
                                "gameday": iso_day,
                                "minutes": round(mins, 1),
                                "points": pts,
                                "rebounds": reb,
                                "assists": ast,
                                "fg3m": fg3m,
                                "fga": fga_p,
                                "fta": fta_p,
                                "pra": pts + reb + ast,
                            }
                        )
            return game_row, p_rows

        new_games: list[dict[str, Any]] = []
        new_players: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            for g_row, p_rows in pool.map(_fetch_summary, completed_evs):
                if g_row is not None:
                    new_games.append(g_row)
                if p_rows:
                    new_players.extend(p_rows)

        if cur_games is not None and new_games:
            add_g = pd.DataFrame(new_games)
            comb_g = pd.concat([cur_games, add_g], ignore_index=True)
            comb_g = comb_g.drop_duplicates(subset=["gameday", "home_team", "away_team"], keep="last")
            cur_games = comb_g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
            write_csv_atomic(cur_games, self.games_csv)

        if cur_logs is not None and new_players:
            add_p = pd.DataFrame(new_players)
            # Keep base columns consistent with existing CSV schema (position is attached on read)
            base_cols = [c for c in cur_logs.columns if c in add_p.columns]
            comb_p = pd.concat([cur_logs[base_cols], add_p[base_cols]], ignore_index=True)
            comb_p = comb_p.drop_duplicates(subset=["player_name", "team", "gameday"], keep="last")
            cur_logs = comb_p.sort_values(["gameday", "team"]).reset_index(drop=True)
            write_csv_atomic(cur_logs, self.player_logs_csv)

        return cur_games, cur_logs

    def load_games(
        self, seasons: list[str] | None = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Load NBA team game results with Four Factors & pace proxies."""
        if self.games_csv.exists() and not force_refresh:
            return read_csv_cached(self.games_csv)

        season_strs = seasons or ["2025-26", "2026-27"]
        try:
            from nba_api.stats.endpoints import leaguegamelog

            frames = []
            for s in season_strs:
                try:
                    lg = leaguegamelog.LeagueGameLog(
                        season=s,
                        player_or_team_abbreviation="T",
                        timeout=6,
                    )
                    part = lg.get_data_frames()[0]
                    if len(part) > 0:
                        part["SEASON_STR"] = s
                        frames.append(part)
                except Exception:
                    continue
            if frames:
                raw = pd.concat(frames, ignore_index=True)
                paired = self._pair_team_logs(raw)
                if len(paired) >= 80:
                    write_csv_atomic(paired, self.games_csv)
                    return paired
        except Exception:
            pass

        # Cloud IPs often get rate-limited/timed out by stats.nba.com: backfill recent games from ESPN
        if self.games_csv.exists():
            existing_g = read_csv_cached(self.games_csv)
            existing_p = read_csv_cached(self.player_logs_csv) if self.player_logs_csv.exists() else None
            updated_g, _ = self._backfill_from_espn(existing_games=existing_g, existing_logs=existing_p)
            return updated_g if updated_g is not None else existing_g
        df = self.generate_synthetic_nba_games()
        write_csv_atomic(df, self.games_csv)
        return df

    def load_player_logs(
        self, seasons: list[str] | None = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Load NBA player box-score logs (minutes, PTS, REB, AST, FG3M, usage proxy, position)."""
        pos_map = self.load_player_positions(force_refresh=force_refresh)
        if self.player_logs_csv.exists() and not force_refresh:
            return self.attach_player_positions(read_csv_cached(self.player_logs_csv), pos_map)

        season_strs = seasons or ["2025-26", "2026-27"]
        try:
            from nba_api.stats.endpoints import leaguegamelog

            frames = []
            for s in season_strs:
                try:
                    lg = leaguegamelog.LeagueGameLog(
                        season=s,
                        player_or_team_abbreviation="P",
                        timeout=6,
                    )
                    part = lg.get_data_frames()[0]
                    if len(part) > 0:
                        frames.append(part)
                except Exception:
                    continue
            if frames:
                raw = pd.concat(frames, ignore_index=True)
                df = self._normalize_player_logs(raw)
                if len(df) >= 200:
                    write_csv_atomic(df, self.player_logs_csv)
                    return self.attach_player_positions(df, pos_map)
        except Exception:
            pass

        if self.player_logs_csv.exists():
            existing_p = read_csv_cached(self.player_logs_csv)
            existing_g = read_csv_cached(self.games_csv) if self.games_csv.exists() else None
            _, updated_p = self._backfill_from_espn(existing_games=existing_g, existing_logs=existing_p)
            return self.attach_player_positions(updated_p if updated_p is not None else existing_p, pos_map)
        df = self.generate_synthetic_nba_player_logs()
        write_csv_atomic(df, self.player_logs_csv)
        return self.attach_player_positions(df, pos_map)

    @staticmethod
    def _pair_team_logs(raw: pd.DataFrame) -> pd.DataFrame:
        work = raw.copy()
        if "WL" in work.columns:
            work = work[
                work["WL"].notna()
                & work["WL"].astype(str).str.strip().str.upper().isin(["W", "L"])
            ].copy()
        if len(work) == 0:
            return pd.DataFrame()
        work["IS_HOME"] = work["MATCHUP"].astype(str).str.contains("vs\\.")
        # Possessions estimate: FGA + 0.44 * FTA - OREB + TOV
        work["POSS"] = (
            pd.to_numeric(work["FGA"], errors="coerce").fillna(88.0)
            + 0.44 * pd.to_numeric(work["FTA"], errors="coerce").fillna(22.0)
            - pd.to_numeric(work["OREB"], errors="coerce").fillna(10.5)
            + pd.to_numeric(work["TOV"], errors="coerce").fillna(13.5)
        ).clip(lower=80.0)
        work["EFG"] = (
            pd.to_numeric(work["FGM"], errors="coerce").fillna(41.0)
            + 0.5 * pd.to_numeric(work["FG3M"], errors="coerce").fillna(13.0)
        ) / pd.to_numeric(work["FGA"], errors="coerce").clip(lower=50.0)
        work["TOV_PCT"] = pd.to_numeric(work["TOV"], errors="coerce") / work["POSS"]
        work["FTR"] = pd.to_numeric(work["FTA"], errors="coerce") / pd.to_numeric(
            work["FGA"], errors="coerce"
        ).clip(lower=50.0)

        homes = work[work["IS_HOME"]].copy()
        aways = work[~work["IS_HOME"]].copy()
        merged = homes.merge(aways, on="GAME_ID", suffixes=("_H", "_A"))

        out = pd.DataFrame(
            {
                "game_id": merged["GAME_ID"].astype(str),
                "season": merged["SEASON_STR_H"].astype(str),
                "gameday": pd.to_datetime(merged["GAME_DATE_H"]).dt.strftime("%Y-%m-%d"),
                "home_team": merged["TEAM_ABBREVIATION_H"].astype(str).str.upper(),
                "away_team": merged["TEAM_ABBREVIATION_A"].astype(str).str.upper(),
                "home_score": pd.to_numeric(merged["PTS_H"], errors="coerce"),
                "away_score": pd.to_numeric(merged["PTS_A"], errors="coerce"),
                "pace": 0.5 * (merged["POSS_H"] + merged["POSS_A"]),
                "home_efg": merged["EFG_H"].round(4),
                "away_efg": merged["EFG_A"].round(4),
                "home_tov_pct": merged["TOV_PCT_H"].round(4),
                "away_tov_pct": merged["TOV_PCT_A"].round(4),
                "home_oreb": pd.to_numeric(merged["OREB_H"], errors="coerce"),
                "away_oreb": pd.to_numeric(merged["OREB_A"], errors="coerce"),
                "home_ftr": merged["FTR_H"].round(4),
                "away_ftr": merged["FTR_A"].round(4),
            }
        )
        out = out.dropna(subset=["home_score", "away_score"])
        out = out[
            (out["home_score"] != out["away_score"])
            & (out["home_score"] >= 65)
            & (out["away_score"] >= 65)
        ]
        return out.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    @staticmethod
    def _normalize_player_logs(raw: pd.DataFrame) -> pd.DataFrame:
        work = raw.copy()
        if "WL" in work.columns:
            work = work[
                work["WL"].notna()
                & work["WL"].astype(str).str.strip().str.upper().isin(["W", "L"])
            ].copy()
        if len(work) == 0:
            return pd.DataFrame()
        # Fallback columns are index-aligned: `work` was filtered above, so a default
        # RangeIndex Series would silently misalign and inject NaNs.
        matchup = _col(work, "MATCHUP", "BOS vs. NYK").astype(str)
        opp = matchup.str.extract(r"(?:vs\.|@)\s*([A-Z]{2,4})")[0].fillna("UNK")
        names = _col(work, "PLAYER_NAME", "Player")
        df = pd.DataFrame(
            {
                "player_id": (work["PLAYER_ID"] if "PLAYER_ID" in work.columns else names).astype(str),
                "player_name": names.astype(str),
                "team": _col(work, "TEAM_ABBREVIATION", "UNK").astype(str).str.upper(),
                "opponent": opp.str.upper(),
                "gameday": pd.to_datetime(work["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
                "minutes": _parse_minutes_series(_col(work, "MIN", 0)),
                "points": pd.to_numeric(_col(work, "PTS", 0), errors="coerce").fillna(0.0),
                "rebounds": pd.to_numeric(_col(work, "REB", 0), errors="coerce").fillna(0.0),
                "assists": pd.to_numeric(_col(work, "AST", 0), errors="coerce").fillna(0.0),
                "fg3m": pd.to_numeric(_col(work, "FG3M", 0), errors="coerce").fillna(0.0),
                "fga": pd.to_numeric(_col(work, "FGA", 0), errors="coerce").fillna(0.0),
                "fta": pd.to_numeric(_col(work, "FTA", 0), errors="coerce").fillna(0.0),
            }
        )
        df["pra"] = df["points"] + df["rebounds"] + df["assists"]
        return df.sort_values(["gameday", "team"]).reset_index(drop=True)

    @staticmethod
    def generate_synthetic_nba_games(seed: int = 101) -> pd.DataFrame:
        """Realistic multi-month NBA historical game dataset with rest/B2B & pace dynamics."""
        rng = np.random.default_rng(seed)
        net_rtg = {
            t: float(v)
            for t, v in zip(NBA_TEAMS, np.linspace(8.2, -8.5, len(NBA_TEAMS)))
        }
        pace_pref = {
            t: float(v)
            for t, v in zip(NBA_TEAMS, rng.normal(99.2, 2.4, len(NBA_TEAMS)))
        }
        start_dt = datetime(2025, 11, 1)
        rows = []
        last_played: dict[str, datetime] = {}
        for day_idx in range(160):
            cur_dt = start_dt + timedelta(days=day_idx)
            gdate = cur_dt.strftime("%Y-%m-%d")
            # 6 to 10 games per night
            n_games = int(rng.integers(5, 10))
            pool = list(NBA_TEAMS)
            rng.shuffle(pool)
            for g in range(n_games):
                home, away = pool[2 * g], pool[2 * g + 1]
                h_b2b = 1 if (home in last_played and (cur_dt - last_played[home]).days == 1) else 0
                a_b2b = 1 if (away in last_played and (cur_dt - last_played[away]).days == 1) else 0
                last_played[home] = cur_dt
                last_played[away] = cur_dt

                pace = float(
                    np.clip(
                        0.5 * (pace_pref[home] + pace_pref[away]) + rng.normal(0, 2.2),
                        91.0,
                        108.0,
                    )
                )
                alt_boost = 1.6 if home in ("DEN", "UTA") else 0.0
                exp_margin = (
                    (net_rtg[home] - net_rtg[away]) * (pace / 100.0)
                    + 2.3
                    + alt_boost
                    - 2.2 * h_b2b
                    + 2.2 * a_b2b
                )
                exp_total = 225.5 + (pace - 99.2) * 2.15 - 1.5 * (h_b2b + a_b2b)

                margin = int(round(rng.normal(exp_margin, 12.8)))
                if margin == 0:
                    margin = 2 if rng.random() < 0.5 else -2
                tot = int(round(rng.normal(exp_total, 15.0)))
                h_pts = max(88, int(round((tot + margin) / 2.0)))
                a_pts = max(85, h_pts - margin)

                rows.append(
                    {
                        "game_id": f"NBA_{gdate}_{away}_{home}",
                        "season": "2025-26" if day_idx < 125 else "2026-27",
                        "gameday": gdate,
                        "home_team": home,
                        "away_team": away,
                        "home_score": h_pts,
                        "away_score": a_pts,
                        "pace": round(pace, 1),
                        "home_efg": round(0.542 + (h_pts - 113) * 0.0018, 4),
                        "away_efg": round(0.538 + (a_pts - 112) * 0.0018, 4),
                        "home_tov_pct": round(float(np.clip(rng.normal(0.132, 0.018), 0.08, 0.20)), 4),
                        "away_tov_pct": round(float(np.clip(rng.normal(0.135, 0.018), 0.08, 0.20)), 4),
                        "home_oreb": int(np.clip(rng.normal(10.8, 2.6), 4, 20)),
                        "away_oreb": int(np.clip(rng.normal(10.4, 2.6), 4, 20)),
                        "home_ftr": round(float(np.clip(rng.normal(0.248, 0.045), 0.12, 0.42)), 4),
                        "away_ftr": round(float(np.clip(rng.normal(0.242, 0.045), 0.12, 0.42)), 4),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def generate_synthetic_nba_player_logs(seed: int = 101) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        players = [
            ("Jayson Tatum", "BOS", 36.2, 27.8, 8.6, 5.4, 3.3),
            ("Jaylen Brown", "BOS", 34.5, 24.2, 5.8, 4.1, 2.2),
            ("Shai Gilgeous-Alexander", "OKC", 34.8, 31.1, 5.5, 6.4, 1.9),
            ("Jalen Williams", "OKC", 33.2, 21.4, 5.4, 5.1, 1.8),
            ("Nikola Jokic", "DEN", 36.4, 29.2, 12.8, 10.1, 1.6),
            ("Jamal Murray", "DEN", 34.0, 21.2, 4.1, 6.3, 2.4),
            ("Luka Doncic", "LAL", 36.8, 29.8, 8.7, 8.5, 3.5),
            ("LeBron James", "LAL", 34.2, 23.6, 7.6, 8.1, 2.1),
            ("Anthony Davis", "DAL", 35.1, 25.4, 11.8, 3.4, 0.7),
            ("Kyrie Irving", "DAL", 34.6, 24.5, 4.8, 5.2, 2.9),
            ("Donovan Mitchell", "CLE", 34.0, 26.5, 4.6, 5.1, 3.4),
            ("Evan Mobley", "CLE", 32.8, 18.8, 9.4, 3.1, 0.9),
            ("Jalen Brunson", "NYK", 35.4, 27.4, 3.4, 7.3, 2.6),
            ("Karl-Anthony Towns", "NYK", 34.5, 24.8, 12.6, 3.2, 2.2),
            ("Anthony Edwards", "MIN", 36.1, 27.9, 5.7, 4.8, 3.7),
            ("Giannis Antetokounmpo", "MIL", 35.2, 30.8, 11.9, 6.1, 0.5),
            ("Stephen Curry", "GSW", 32.8, 25.8, 4.5, 6.2, 4.4),
            ("Kevin Durant", "HOU", 35.5, 26.8, 6.4, 4.3, 2.4),
            ("Devin Booker", "PHX", 36.0, 26.9, 4.3, 6.8, 2.6),
            ("Victor Wembanyama", "SAS", 33.6, 25.6, 11.2, 3.9, 2.8),
        ]
        start_dt = datetime(2026, 10, 22)
        rows = []
        for g_idx in range(18):
            gdate = (start_dt + timedelta(days=g_idx * 2)).strftime("%Y-%m-%d")
            for name, team, mins, pts, reb, ast, fg3 in players:
                m = float(np.clip(rng.normal(mins, 3.1), 22.0, 42.0))
                scale = m / mins
                p = int(max(4, round(rng.normal(pts * scale, 5.6))))
                r = int(max(0, rng.poisson(max(0.5, reb * scale))))
                a = int(max(0, rng.poisson(max(0.5, ast * scale))))
                t3 = int(max(0, rng.poisson(max(0.2, fg3 * scale))))
                opp = str(rng.choice([t for t in NBA_TEAMS if t != team]))
                rows.append(
                    {
                        "player_id": name.lower().replace(" ", "_"),
                        "player_name": name,
                        "team": team,
                        "opponent": opp,
                        "gameday": gdate,
                        "minutes": round(m, 1),
                        "points": p,
                        "rebounds": r,
                        "assists": a,
                        "fg3m": t3,
                        "fga": round(p * 0.74, 0),
                        "fta": round(p * 0.24, 0),
                        "pra": p + r + a,
                    }
                )
        return pd.DataFrame(rows)

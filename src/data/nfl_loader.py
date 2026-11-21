"""NFL historical & current-season schedule, team efficiency, and player log loader."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.nba_loader import _col, read_csv_cached, write_csv_atomic

DATA_DIR = PROJECT_ROOT / "data"

NFL_TEAMS = [
    "KC", "BUF", "BAL", "DET", "PHI", "SF", "GB", "HOU",
    "CIN", "MIN", "LAC", "WAS", "SEA", "TB", "PIT", "DEN",
    "LAR", "LA", "DAL", "ATL", "ARI", "MIA", "IND", "JAX",
    "CHI", "NO", "NYJ", "CLE", "LV", "TEN", "CAR", "NYG", "NE",
]


class NFLLoader:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.schedules_csv = self.cache_dir / "nfl_schedules.csv"
        self.player_logs_csv = self.cache_dir / "nfl_player_logs.csv"

    def load_schedules(
        self, seasons: list[int] | None = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Load NFL game schedules & results (with Vegas lines, rest, roof, wind, temp)."""
        if self.schedules_csv.exists() and not force_refresh:
            return read_csv_cached(self.schedules_csv)

        years = seasons or [2024, 2025, 2026]
        try:
            import nflreadpy as nfl

            valid_years = []
            for y in years:
                try:
                    pl_df = nfl.load_schedules([y])
                    part = pd.DataFrame(pl_df.to_dict(as_series=False))
                    if len(part) > 0:
                        valid_years.append(part)
                except Exception:
                    continue
            if valid_years:
                raw = pd.concat(valid_years, ignore_index=True)
                df = self._normalize_schedules(raw)
                write_csv_atomic(df, self.schedules_csv)
                return df
        except Exception:
            pass

        # A failed forced refresh must not clobber previously fetched real data with synthetic rows
        if self.schedules_csv.exists():
            return read_csv_cached(self.schedules_csv)
        df = self.generate_synthetic_nfl_schedules()
        write_csv_atomic(df, self.schedules_csv)
        return df

    def load_player_logs(
        self, seasons: list[int] | None = None, force_refresh: bool = False
    ) -> pd.DataFrame:
        """Load weekly NFL player stats (passing, rushing, receiving) for prop modeling."""
        if self.player_logs_csv.exists() and not force_refresh:
            return read_csv_cached(self.player_logs_csv)

        years = seasons or [2025, 2026]
        try:
            import nflreadpy as nfl

            frames = []
            for y in years:
                try:
                    pl_df = nfl.load_player_stats([y])
                    part = pd.DataFrame(pl_df.to_dict(as_series=False))
                    if len(part) > 0:
                        frames.append(part)
                except Exception:
                    continue
            if frames:
                raw = pd.concat(frames, ignore_index=True)
                df = self._normalize_player_logs(raw)
                write_csv_atomic(df, self.player_logs_csv)
                return df
        except Exception:
            pass

        if self.player_logs_csv.exists():
            return read_csv_cached(self.player_logs_csv)
        df = self.generate_synthetic_nfl_player_logs()
        write_csv_atomic(df, self.player_logs_csv)
        return df

    @staticmethod
    def _normalize_schedules(raw: pd.DataFrame) -> pd.DataFrame:
        # `_col` fallbacks are index-aligned Series (bare scalar defaults like "KC" have no
        # .astype and previously raised AttributeError when nflreadpy dropped a column).
        df = pd.DataFrame(index=raw.index)
        df["game_id"] = (
            raw["game_id"] if "game_id" in raw.columns else pd.Series(range(len(raw)), index=raw.index)
        ).astype(str)
        df["season"] = pd.to_numeric(_col(raw, "season", 2025), errors="coerce").fillna(2025).astype(int)
        df["week"] = pd.to_numeric(_col(raw, "week", 1), errors="coerce").fillna(1).astype(int)
        df["gameday"] = _col(raw, "gameday", "2025-09-07").astype(str)
        df["home_team"] = _col(raw, "home_team", "KC").astype(str).str.upper()
        df["away_team"] = _col(raw, "away_team", "BUF").astype(str).str.upper()
        df["home_score"] = pd.to_numeric(_col(raw, "home_score", np.nan), errors="coerce")
        df["away_score"] = pd.to_numeric(_col(raw, "away_score", np.nan), errors="coerce")
        # In nflverse/nflreadpy, spread_line is positive when home team is favored!
        # Convert to standard sportsbook convention: negative = home favored (e.g., -3.5)
        raw_spread = pd.to_numeric(_col(raw, "spread_line", np.nan), errors="coerce")
        df["home_spread"] = -raw_spread
        df["total_line"] = pd.to_numeric(_col(raw, "total_line", np.nan), errors="coerce")
        df["home_rest"] = pd.to_numeric(_col(raw, "home_rest", 7), errors="coerce").fillna(7.0)
        df["away_rest"] = pd.to_numeric(_col(raw, "away_rest", 7), errors="coerce").fillna(7.0)
        roof_col = _col(raw, "roof", "outdoors").fillna("outdoors").astype(str)
        df["is_indoor"] = roof_col.str.lower().isin(["dome", "closed"]).astype(int)
        df["temp"] = pd.to_numeric(_col(raw, "temp", 65.0), errors="coerce").fillna(65.0)
        df["wind"] = pd.to_numeric(_col(raw, "wind", 5.0), errors="coerce").fillna(5.0)
        df["div_game"] = pd.to_numeric(_col(raw, "div_game", 0), errors="coerce").fillna(0).astype(int)
        return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    @staticmethod
    def _normalize_player_logs(raw: pd.DataFrame) -> pd.DataFrame:
        team_col = (
            raw["recent_team"]
            if "recent_team" in raw.columns
            else _col(raw, "team", "KC")
        )
        name_col = (
            raw["player_display_name"]
            if "player_display_name" in raw.columns
            else _col(raw, "player_name", "Player")
        )
        opp_col = _col(raw, "opponent_team", "BUF")

        def _num(name: str, default: float = 0.0) -> pd.Series:
            return pd.to_numeric(_col(raw, name, default), errors="coerce").fillna(default)

        df = pd.DataFrame(
            {
                "player_id": (raw["player_id"] if "player_id" in raw.columns else name_col).astype(str),
                "player_name": name_col.astype(str),
                "position": _col(raw, "position", "WR").fillna("WR").astype(str),
                "team": team_col.fillna("UNK").astype(str).str.upper(),
                "opponent": opp_col.fillna("UNK").astype(str).str.upper(),
                "season": _num("season", 2026).astype(int),
                "week": _num("week", 1).astype(int),
                "attempts": _num("attempts"),
                "completions": _num("completions"),
                "passing_yards": _num("passing_yards"),
                "passing_tds": _num("passing_tds"),
                "carries": _num("carries"),
                "rushing_yards": _num("rushing_yards"),
                "targets": _num("targets"),
                "receptions": _num("receptions"),
                "receiving_yards": _num("receiving_yards"),
            }
        )
        df["game_order"] = df["season"] * 100 + df["week"]
        return df.sort_values(["game_order", "team"]).reset_index(drop=True)

    @staticmethod
    def generate_synthetic_nfl_schedules(seed: int = 42) -> pd.DataFrame:
        """Realistic multi-season NFL schedule + results dataset with key-number margins."""
        rng = np.random.default_rng(seed)
        teams = [
            "KC", "BUF", "BAL", "DET", "PHI", "SF", "GB", "HOU",
            "CIN", "MIN", "LAC", "WAS", "SEA", "TB", "PIT", "DEN",
            "LA", "DAL", "ATL", "ARI", "MIA", "IND", "JAX", "CHI",
            "NO", "NYJ", "CLE", "LV", "TEN", "CAR", "NYG", "NE",
        ]
        true_strength = {
            t: float(val)
            for t, val in zip(teams, np.linspace(6.5, -6.5, len(teams)))
        }
        dome_teams = {"DET", "MIN", "NO", "ATL", "DAL", "ARI", "LA", "LAC", "LV", "IND", "HOU"}
        key_offsets = np.array([-14, -10, -7, -6, -4, -3, 3, 4, 6, 7, 10, 14])

        rows: list[dict[str, Any]] = []
        base_dates = {2024: datetime(2024, 9, 8), 2025: datetime(2025, 9, 7), 2026: datetime(2026, 9, 10)}
        for season in [2024, 2025, 2026]:
            max_completed_week = 18 if season < 2026 else 9
            total_weeks = 18 if season < 2026 else 11
            for week in range(1, total_weeks + 1):
                shuffled = list(teams)
                rng.shuffle(shuffled)
                gdate = (base_dates[season] + timedelta(days=(week - 1) * 7)).strftime("%Y-%m-%d")
                for i in range(0, len(shuffled), 2):
                    home, away = shuffled[i], shuffled[i + 1]
                    is_indoor = int(home in dome_teams)
                    wind = 0.0 if is_indoor else float(max(2.0, rng.normal(9.5, 5.5)))
                    temp = 72.0 if is_indoor else float(rng.normal(62.0 - week * 1.4, 10.0))
                    h_rest = int(rng.choice([7, 7, 7, 10, 6, 14]))
                    a_rest = int(rng.choice([7, 7, 7, 10, 6, 14]))

                    exp_margin = (
                        (true_strength[home] - true_strength[away])
                        + 1.9
                        + 0.25 * (h_rest - a_rest)
                    )
                    home_spread = round(-exp_margin * 2.0) / 2.0
                    exp_total = 45.5 + (0.8 if is_indoor else 0.0) - max(0.0, wind - 12.0) * 0.35
                    total_line = round(exp_total * 2.0) / 2.0

                    if week <= max_completed_week:
                        # Mix continuous margin with NFL key-number snapping
                        raw_margin = rng.normal(exp_margin, 13.0)
                        if rng.random() < 0.35:
                            nearest = key_offsets[np.argmin(np.abs(key_offsets - raw_margin))]
                            margin = int(nearest)
                        else:
                            margin = int(round(raw_margin))
                            if margin == 0:
                                margin = 3 if rng.random() < 0.5 else -3
                        tot_pts = max(17, int(round(rng.normal(exp_total, 9.5))))
                        h_score = max(3, int(round((tot_pts + margin) / 2.0)))
                        a_score = max(0, h_score - margin)
                    else:
                        h_score = np.nan
                        a_score = np.nan

                    rows.append(
                        {
                            "game_id": f"{season}_{week:02d}_{away}_{home}",
                            "season": season,
                            "week": week,
                            "gameday": gdate,
                            "home_team": home,
                            "away_team": away,
                            "home_score": h_score,
                            "away_score": a_score,
                            "home_spread": home_spread,
                            "total_line": total_line,
                            "home_rest": h_rest,
                            "away_rest": a_rest,
                            "is_indoor": is_indoor,
                            "temp": round(temp, 1),
                            "wind": round(wind, 1),
                            "div_game": int(i % 6 == 0),
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def generate_synthetic_nfl_player_logs(seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        stars = [
            ("Patrick Mahomes", "QB", "KC", 268.0, 2.0, 22.0, 0.0, 0.0),
            ("Josh Allen", "QB", "BUF", 252.0, 1.9, 38.0, 0.0, 0.0),
            ("Lamar Jackson", "QB", "BAL", 238.0, 1.8, 56.0, 0.0, 0.0),
            ("Jared Goff", "QB", "DET", 262.0, 1.9, 6.0, 0.0, 0.0),
            ("Jalen Hurts", "QB", "PHI", 228.0, 1.5, 44.0, 0.0, 0.0),
            ("Joe Burrow", "QB", "CIN", 278.0, 2.2, 12.0, 0.0, 0.0),
            ("Jayden Daniels", "QB", "WAS", 242.0, 1.6, 46.0, 0.0, 0.0),
            ("C.J. Stroud", "QB", "HOU", 256.0, 1.7, 14.0, 0.0, 0.0),
            ("Saquon Barkley", "RB", "PHI", 0.0, 0.0, 96.0, 3.2, 24.0),
            ("Derrick Henry", "RB", "BAL", 0.0, 0.0, 98.0, 1.6, 12.0),
            ("Jahmyr Gibbs", "RB", "DET", 0.0, 0.0, 78.0, 4.1, 34.0),
            ("Bijan Robinson", "RB", "ATL", 0.0, 0.0, 82.0, 4.3, 35.0),
            ("Ja'Marr Chase", "WR", "CIN", 0.0, 0.0, 4.0, 6.9, 94.0),
            ("Justin Jefferson", "WR", "MIN", 0.0, 0.0, 2.0, 6.6, 92.0),
            ("Amon-Ra St. Brown", "WR", "DET", 0.0, 0.0, 3.0, 7.2, 84.0),
            ("Nico Collins", "WR", "HOU", 0.0, 0.0, 0.0, 6.2, 88.0),
            ("A.J. Brown", "WR", "PHI", 0.0, 0.0, 0.0, 5.8, 86.0),
            ("Travis Kelce", "TE", "KC", 0.0, 0.0, 0.0, 6.1, 61.0),
        ]
        opps = ["KC", "BUF", "BAL", "DET", "PHI", "SF", "GB", "HOU", "CIN", "MIN", "WAS", "DEN"]
        rows = []
        for season in [2025, 2026]:
            max_wk = 17 if season == 2025 else 9
            for wk in range(1, max_wk + 1):
                for name, pos, team, py, ptd, ry, rec, recy in stars:
                    opp = str(rng.choice([o for o in opps if o != team]))
                    p_yds = float(max(0.0, rng.gamma(11.0, py / 11.0))) if py > 0 else 0.0
                    p_tds = int(rng.poisson(ptd)) if ptd > 0 else 0
                    r_yds = float(max(0.0, rng.gamma(4.5, ry / 4.5))) if ry > 0 else 0.0
                    recs = int(rng.poisson(rec)) if rec > 0 else 0
                    rc_yds = float(max(0.0, rng.gamma(4.2, recy / 4.2))) if recy > 0 else 0.0
                    rows.append(
                        {
                            "player_id": name.lower().replace(" ", "_"),
                            "player_name": name,
                            "position": pos,
                            "team": team,
                            "opponent": opp,
                            "season": season,
                            "week": wk,
                            "attempts": round(p_yds / 7.4, 0) if py > 0 else 0.0,
                            "completions": round(p_yds / 11.2, 0) if py > 0 else 0.0,
                            "passing_yards": round(p_yds, 0),
                            "passing_tds": p_tds,
                            "carries": round(r_yds / 4.7, 0) if ry > 15 else 0.0,
                            "rushing_yards": round(r_yds, 0),
                            "targets": round(recs * 1.42, 0) if rec > 0 else 0.0,
                            "receptions": recs,
                            "receiving_yards": round(rc_yds, 0),
                            "game_order": season * 100 + wk,
                        }
                    )
        return pd.DataFrame(rows)

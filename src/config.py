"""Configuration loader for DraftKings Predictions (DKeX) NFL & NBA Model."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    cleaned = str(raw).strip()
    if not cleaned:
        return default
    try:
        return int(cleaned)
    except ValueError:
        return default


def _read_sqlite_settings(db_path: Path) -> dict[str, str]:
    if not db_path.exists():
        return {}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        return {str(k): str(v) for k, v in rows}
    except sqlite3.Error:
        return {}


@dataclass
class Settings:
    bankroll: float = 1000.0
    default_fee_dollars: float = 0.015
    kelly_fraction: float = 0.25
    max_single_bet_pct: float = 0.04
    max_same_game_pct: float = 0.06
    min_win_prob: float = 0.60
    min_net_edge: float = 0.025
    the_odds_api_key: str = ""
    app_passcode: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    db_path: Path = PROJECT_ROOT / "data" / "dk_model.sqlite3"
    models_dir: Path = PROJECT_ROOT / "models"

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        target_env = env_path or ENV_FILE
        file_vals = dotenv_values(target_env) if target_env.exists() else {}
        in_pytest = "PYTEST_CURRENT_TEST" in os.environ

        def _base_lookup(key: str, default: str) -> str:
            if in_pytest and key in os.environ:
                return os.environ[key]
            if key in os.environ and os.environ[key] != "":
                return os.environ[key]
            if key in file_vals and file_vals[key] is not None:
                return str(file_vals[key])
            return os.getenv(key, default)

        data_dir_raw = _base_lookup("DATA_DIR", "").strip()
        base_data_dir = (
            (Path(data_dir_raw) if Path(data_dir_raw).is_absolute() else PROJECT_ROOT / data_dir_raw)
            if data_dir_raw
            else None
        )

        default_db = str(base_data_dir / "dk_model.sqlite3") if base_data_dir else "data/dk_model.sqlite3"
        default_models = str(base_data_dir / "models") if base_data_dir else "models"

        db_raw = _base_lookup("DB_PATH", default_db)
        models_raw = _base_lookup("MODELS_DIR", default_models)
        db_path = Path(db_raw) if Path(db_raw).is_absolute() else PROJECT_ROOT / db_raw
        models_dir = (
            Path(models_raw)
            if Path(models_raw).is_absolute()
            else PROJECT_ROOT / models_raw
        )

        db_vals = _read_sqlite_settings(db_path)

        def _lookup(key: str, default: str) -> str:
            if in_pytest and key in os.environ:
                return os.environ[key]
            if key in db_vals and db_vals[key] is not None:
                return db_vals[key]
            if key in file_vals and file_vals[key] is not None:
                return str(file_vals[key])
            return os.getenv(key, default)

        return cls(
            bankroll=_parse_float(_lookup("BANKROLL", "1000.0"), 1000.0),
            default_fee_dollars=_parse_float(_lookup("DEFAULT_FEE_DOLLARS", "0.015"), 0.015),
            kelly_fraction=_parse_float(_lookup("KELLY_FRACTION", "0.25"), 0.25),
            max_single_bet_pct=_parse_float(_lookup("MAX_SINGLE_BET_PCT", "0.04"), 0.04),
            max_same_game_pct=_parse_float(_lookup("MAX_SAME_GAME_PCT", "0.06"), 0.06),
            min_win_prob=_parse_float(_lookup("MIN_WIN_PROB", "0.60"), 0.60),
            min_net_edge=_parse_float(_lookup("MIN_NET_EDGE", "0.025"), 0.025),
            the_odds_api_key=_lookup("THE_ODDS_API_KEY", "").strip(),
            app_passcode=_base_lookup("APP_PASSCODE", "").strip(),
            host=_base_lookup("HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=_parse_int(_base_lookup("PORT", "8000"), 8000),
            db_path=db_path,
            models_dir=models_dir,
        )


def get_settings() -> Settings:
    return Settings.from_env()


def update_env_settings(updates: dict[str, str], env_path: Path | None = None) -> None:
    """Persist updated user settings into SQLite (for remote containers) and .env (when writable)."""
    if not updates:
        return

    if "PYTEST_CURRENT_TEST" in os.environ and env_path is None:
        for k, v in updates.items():
            os.environ[k] = v
        try:
            current_db = Settings.from_env().db_path
            if current_db.exists():
                import time as _time
                now = _time.time()
                with sqlite3.connect(str(current_db)) as conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL)"
                    )
                    for k, v in updates.items():
                        conn.execute(
                            "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                            (str(k), str(v), now),
                        )
        except sqlite3.Error:
            pass
        return

    # Persist into SQLite so remote volumes retain settings across container restarts
    try:
        current_db = Settings.from_env(env_path=env_path).db_path
        current_db.parent.mkdir(parents=True, exist_ok=True)
        import time as _time
        now = _time.time()
        with sqlite3.connect(str(current_db)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL)"
            )
            for k, v in updates.items():
                conn.execute(
                    "INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                    (str(k), str(v), now),
                )
    except OSError:
        pass
    except sqlite3.Error:
        pass

    target = env_path or ENV_FILE
    try:
        if target.exists():
            content = target.read_text(encoding="utf-8")
        elif (PROJECT_ROOT / ".env.example").exists():
            content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        else:
            content = ""

        for key, val in updates.items():
            pattern = re.compile(rf"^{re.escape(key)}=.*$", flags=re.MULTILINE)
            line = f"{key}={val}"
            if pattern.search(content):
                content = pattern.sub(line, content)
            else:
                content = content.rstrip() + f"\n{line}\n"

        target.write_text(content, encoding="utf-8")
    except OSError:
        # Read-only container filesystems rely on SQLite app_settings above
        pass


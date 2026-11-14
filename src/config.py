"""Configuration loader for DraftKings Predictions (DKeX) NFL & NBA Model."""

from __future__ import annotations

import os
import re
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
    db_path: Path = PROJECT_ROOT / "data" / "dk_model.sqlite3"
    models_dir: Path = PROJECT_ROOT / "models"

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        target_env = env_path or ENV_FILE
        file_vals = dotenv_values(target_env) if target_env.exists() else {}
        in_pytest = "PYTEST_CURRENT_TEST" in os.environ

        def _lookup(key: str, default: str) -> str:
            if in_pytest and key in os.environ:
                return os.environ[key]
            if key in file_vals and file_vals[key] is not None:
                return str(file_vals[key])
            return os.getenv(key, default)

        db_raw = _lookup("DB_PATH", "data/dk_model.sqlite3")
        models_raw = _lookup("MODELS_DIR", "models")
        db_path = Path(db_raw) if Path(db_raw).is_absolute() else PROJECT_ROOT / db_raw
        models_dir = (
            Path(models_raw)
            if Path(models_raw).is_absolute()
            else PROJECT_ROOT / models_raw
        )

        return cls(
            bankroll=_parse_float(_lookup("BANKROLL", "1000.0"), 1000.0),
            default_fee_dollars=_parse_float(_lookup("DEFAULT_FEE_DOLLARS", "0.015"), 0.015),
            kelly_fraction=_parse_float(_lookup("KELLY_FRACTION", "0.25"), 0.25),
            max_single_bet_pct=_parse_float(_lookup("MAX_SINGLE_BET_PCT", "0.04"), 0.04),
            max_same_game_pct=_parse_float(_lookup("MAX_SAME_GAME_PCT", "0.06"), 0.06),
            min_win_prob=_parse_float(_lookup("MIN_WIN_PROB", "0.60"), 0.60),
            min_net_edge=_parse_float(_lookup("MIN_NET_EDGE", "0.025"), 0.025),
            the_odds_api_key=_lookup("THE_ODDS_API_KEY", "").strip(),
            db_path=db_path,
            models_dir=models_dir,
        )


def get_settings() -> Settings:
    return Settings.from_env()


def update_env_settings(updates: dict[str, str], env_path: Path | None = None) -> None:
    """Persist updated user settings (such as BANKROLL or DEFAULT_FEE_DOLLARS) into .env."""
    if "PYTEST_CURRENT_TEST" in os.environ and env_path is None:
        for k, v in updates.items():
            os.environ[k] = v
        return

    target = env_path or ENV_FILE
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

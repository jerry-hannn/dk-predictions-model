"""SQLite persistence layer for API caching, daily prediction snapshots, and DKeX trade ledger."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.config import get_settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    expires_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions_snapshot (
    pick_id TEXT PRIMARY KEY,
    slate_date TEXT NOT NULL,
    sport TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_name TEXT NOT NULL,
    start_time TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection_label TEXT NOT NULL,
    player_name TEXT,
    line_value REAL,
    recommended_side TEXT NOT NULL,
    contract_price REAL NOT NULL,
    fee_per_contract REAL NOT NULL,
    model_prob REAL NOT NULL,
    market_prob REAL,
    conservative_prob REAL NOT NULL,
    net_edge REAL NOT NULL,
    ev_per_contract REAL NOT NULL,
    roi_pct REAL NOT NULL,
    fair_entry_max_price REAL NOT NULL,
    fair_midgame_exit_price REAL NOT NULL,
    recommended_contracts INTEGER NOT NULL,
    recommended_outlay REAL NOT NULL,
    confidence_score INTEGER NOT NULL,
    confidence_tier TEXT NOT NULL,
    drivers_json TEXT NOT NULL,
    injury_notes TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sport TEXT NOT NULL,
    event_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection_label TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('YES', 'NO')),
    entry_price REAL NOT NULL,
    fee_per_contract REAL NOT NULL,
    contracts INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    model_prob REAL NOT NULL,
    fair_exit_price REAL NOT NULL,
    closing_price REAL,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'WON', 'LOST', 'SOLD')),
    exit_or_settle_price REAL,
    exit_fee_per_contract REAL NOT NULL DEFAULT 0.0,
    pnl REAL NOT NULL DEFAULT 0.0,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS line_overrides (
    override_key TEXT PRIMARY KEY,
    line_value REAL NOT NULL,
    contract_price REAL,
    side TEXT,
    updated_at REAL NOT NULL
);
"""


class Database:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_settings().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # --- Persistent UI settings helpers ---
    def get_app_settings(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        return {str(r["key"]): str(r["value"]) for r in rows}

    def set_app_settings(self, updates: dict[str, str]) -> None:
        if not updates:
            return
        now = time.time()
        with self._connect() as conn:
            for k, v in updates.items():
                conn.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (str(k), str(v), now),
                )

    # --- Cache helpers ---
    def get_cache(self, key: str) -> Any | None:
        payload, _ = self.get_cache_with_age(key)
        return payload

    def get_cache_with_age(self, key: str, allow_stale: bool = False) -> tuple[Any | None, float | None]:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, expires_at, updated_at FROM api_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None, None
        age = max(0.0, now - float(row["updated_at"]))
        if not allow_stale and float(row["expires_at"]) < now:
            return None, None
        return json.loads(row["payload_json"]), round(age, 1)

    def set_cache(self, key: str, payload: Any, ttl_seconds: int = 900) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_cache (cache_key, payload_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(payload), now + ttl_seconds, now),
            )

    # --- Custom Line Override helpers ---
    def get_line_overrides(self, max_age_seconds: float = 86400.0) -> dict[str, dict[str, Any]]:
        cutoff = time.time() - max_age_seconds
        with self._connect() as conn:
            conn.execute("DELETE FROM line_overrides WHERE updated_at < ?", (cutoff,))
            rows = conn.execute(
                "SELECT override_key, line_value, contract_price, side, updated_at FROM line_overrides"
            ).fetchall()
        return {
            str(r["override_key"]): {
                "override_key": str(r["override_key"]),
                "line_value": float(r["line_value"]),
                "contract_price": float(r["contract_price"]) if r["contract_price"] is not None else None,
                "side": str(r["side"]) if r["side"] is not None else None,
                "updated_at": float(r["updated_at"]),
            }
            for r in rows
        }

    def set_line_override(
        self,
        override_key: str,
        line_value: float,
        contract_price: float | None = None,
        side: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        norm_side = side.upper() if side and side.upper() in ("YES", "NO") else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO line_overrides (override_key, line_value, contract_price, side, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(override_key) DO UPDATE SET
                    line_value = excluded.line_value,
                    contract_price = excluded.contract_price,
                    side = excluded.side,
                    updated_at = excluded.updated_at
                """,
                (str(override_key), float(line_value), contract_price, norm_side, now),
            )
        return {
            "override_key": str(override_key),
            "line_value": float(line_value),
            "contract_price": contract_price,
            "side": norm_side,
            "updated_at": now,
        }

    def delete_line_override(self, override_key: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM line_overrides WHERE override_key = ?",
                (str(override_key),),
            )
            return cur.rowcount > 0

    def clear_line_overrides(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM line_overrides")
            return int(cur.rowcount or 0)

    # --- Predictions Snapshot ---
    def save_predictions(self, picks: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._connect() as conn:
            for p in picks:
                conn.execute(
                    """
                    INSERT INTO predictions_snapshot (
                        pick_id, slate_date, sport, event_id, event_name, start_time,
                        market_type, selection_label, player_name, line_value,
                        recommended_side, contract_price, fee_per_contract,
                        model_prob, market_prob, conservative_prob, net_edge,
                        ev_per_contract, roi_pct, fair_entry_max_price,
                        fair_midgame_exit_price, recommended_contracts,
                        recommended_outlay, confidence_score, confidence_tier,
                        drivers_json, injury_notes, created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(pick_id) DO UPDATE SET
                        selection_label = excluded.selection_label,
                        line_value = excluded.line_value,
                        recommended_side = excluded.recommended_side,
                        contract_price = excluded.contract_price,
                        fee_per_contract = excluded.fee_per_contract,
                        model_prob = excluded.model_prob,
                        market_prob = excluded.market_prob,
                        conservative_prob = excluded.conservative_prob,
                        net_edge = excluded.net_edge,
                        ev_per_contract = excluded.ev_per_contract,
                        roi_pct = excluded.roi_pct,
                        fair_entry_max_price = excluded.fair_entry_max_price,
                        fair_midgame_exit_price = excluded.fair_midgame_exit_price,
                        recommended_contracts = excluded.recommended_contracts,
                        recommended_outlay = excluded.recommended_outlay,
                        confidence_score = excluded.confidence_score,
                        confidence_tier = excluded.confidence_tier,
                        drivers_json = excluded.drivers_json,
                        injury_notes = excluded.injury_notes,
                        created_at = excluded.created_at
                    """,
                    (
                        p["pick_id"],
                        p["slate_date"],
                        p["sport"],
                        p["event_id"],
                        p["event_name"],
                        p["start_time"],
                        p["market_type"],
                        p["selection_label"],
                        p.get("player_name"),
                        p.get("line_value"),
                        p["recommended_side"],
                        p["contract_price"],
                        p["fee_per_contract"],
                        p["model_prob"],
                        p.get("market_prob"),
                        p["conservative_prob"],
                        p["net_edge"],
                        p["ev_per_contract"],
                        p["roi_pct"],
                        p["fair_entry_max_price"],
                        p["fair_midgame_exit_price"],
                        p["recommended_contracts"],
                        p["recommended_outlay"],
                        p["confidence_score"],
                        p["confidence_tier"],
                        json.dumps(p.get("drivers", [])),
                        p.get("injury_notes", ""),
                        now,
                    ),
                )

    def list_predictions(self, slate_date: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM predictions_snapshot"
        params: tuple[Any, ...] = ()
        if slate_date:
            query += " WHERE slate_date = ?"
            params = (slate_date,)
        query += " ORDER BY confidence_score DESC, net_edge DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["drivers"] = json.loads(item.pop("drivers_json", "[]"))
            out.append(item)
        return out

    # --- Trade Ledger ---
    def log_trade(
        self,
        *,
        created_at: str,
        sport: str,
        event_name: str,
        market_type: str,
        selection_label: str,
        side: str,
        entry_price: float,
        fee_per_contract: float,
        contracts: int,
        model_prob: float,
        fair_exit_price: float,
        notes: str = "",
    ) -> dict[str, Any]:
        total_cost = round(contracts * (entry_price + fee_per_contract), 4)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades (
                    created_at, sport, event_name, market_type, selection_label,
                    side, entry_price, fee_per_contract, contracts, total_cost,
                    model_prob, fair_exit_price, status, pnl, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0, ?)
                """,
                (
                    created_at,
                    sport.upper(),
                    event_name,
                    market_type,
                    selection_label,
                    side.upper(),
                    entry_price,
                    fee_per_contract,
                    contracts,
                    total_cost,
                    model_prob,
                    fair_exit_price,
                    notes,
                ),
            )
            trade_id = cur.lastrowid
            row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(row)

    def update_trade(
        self,
        trade_id: int,
        *,
        status: str,
        exit_or_settle_price: float | None = None,
        exit_fee_per_contract: float = 0.0,
        closing_price: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        status_u = status.upper()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
            if row is None:
                return None
            contracts = int(row["contracts"])
            total_cost = float(row["total_cost"])

            if status_u == "WON":
                settle_px = 1.0
                exit_fee = 0.0  # Expiration settlement at $1.00 incurs no sell trade fee
                gross_payout = contracts * 1.0
                pnl = round(gross_payout - total_cost, 2)
            elif status_u == "LOST":
                settle_px = 0.0
                exit_fee = 0.0
                pnl = round(-total_cost, 2)
            elif status_u == "SOLD":
                if exit_or_settle_price is None:
                    raise ValueError("exit_or_settle_price required when status is SOLD")
                settle_px = float(exit_or_settle_price)
                exit_fee = float(exit_fee_per_contract)
                net_proceeds = contracts * max(0.0, settle_px - exit_fee)
                pnl = round(net_proceeds - total_cost, 2)
            else:
                settle_px = None
                exit_fee = 0.0
                pnl = 0.0

            new_closing = closing_price if closing_price is not None else row["closing_price"]
            new_notes = notes if notes is not None else row["notes"]

            conn.execute(
                """
                UPDATE trades
                SET status = ?,
                    exit_or_settle_price = ?,
                    exit_fee_per_contract = ?,
                    closing_price = ?,
                    pnl = ?,
                    notes = ?
                WHERE id = ?
                """,
                (status_u, settle_px, exit_fee, new_closing, pnl, new_notes, trade_id),
            )
            updated = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(updated)

    def delete_trade(self, trade_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
            return cur.rowcount > 0

    def list_trades(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trades ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def ledger_summary(self, starting_bankroll: float) -> dict[str, Any]:
        trades = self.list_trades()
        settled = [t for t in trades if t["status"] in ("WON", "LOST", "SOLD")]
        open_trades = [t for t in trades if t["status"] == "OPEN"]
        realized_pnl = round(sum(float(t["pnl"]) for t in settled), 2)
        open_exposure = round(sum(float(t["total_cost"]) for t in open_trades), 2)
        total_settled_outlay = sum(float(t["total_cost"]) for t in settled)
        wins = sum(1 for t in settled if float(t["pnl"]) > 0)
        hit_rate = round(wins / len(settled), 4) if settled else None
        expected_win_rate = (
            round(sum(float(t["model_prob"]) for t in settled) / len(settled), 4)
            if settled
            else None
        )
        roi_pct = (
            round(realized_pnl / total_settled_outlay, 4)
            if total_settled_outlay > 0
            else 0.0
        )
        clv_samples = [
            float(t["closing_price"]) - float(t["entry_price"])
            for t in trades
            if t.get("closing_price") is not None
        ]
        avg_clv_cents = (
            round((sum(clv_samples) / len(clv_samples)) * 100, 2)
            if clv_samples
            else None
        )
        return {
            "starting_bankroll": round(starting_bankroll, 2),
            "current_bankroll": round(starting_bankroll + realized_pnl, 2),
            "realized_pnl": realized_pnl,
            "open_exposure": open_exposure,
            "open_trades_count": len(open_trades),
            "settled_trades_count": len(settled),
            "hit_rate": hit_rate,
            "expected_win_rate": expected_win_rate,
            "roi_pct": roi_pct,
            "avg_clv_cents": avg_clv_cents,
        }

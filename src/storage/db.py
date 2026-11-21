"""SQLite persistence layer for API caching, daily prediction snapshots, and multi-venue trade ledger."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.engine.dk_math import american_to_implied_prob
from src.engine.sportsbook_math import american_to_decimal


TRADES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sport TEXT NOT NULL,
    event_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection_label TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    fee_per_contract REAL NOT NULL,
    contracts INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    model_prob REAL NOT NULL,
    fair_exit_price REAL NOT NULL,
    closing_price REAL,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'WON', 'LOST', 'SOLD', 'PUSH', 'VOID', 'CASHOUT')),
    exit_or_settle_price REAL,
    exit_fee_per_contract REAL NOT NULL DEFAULT 0.0,
    pnl REAL NOT NULL DEFAULT 0.0,
    notes TEXT DEFAULT '',
    venue TEXT NOT NULL DEFAULT 'DKEX',
    bet_format TEXT NOT NULL DEFAULT 'CONTRACT',
    odds_american INTEGER,
    stake REAL,
    potential_payout REAL,
    is_bonus_bet INTEGER NOT NULL DEFAULT 0,
    strategy_tag TEXT NOT NULL DEFAULT 'EV',
    group_id TEXT,
    closing_odds_american INTEGER
);
"""


SCHEMA_SQL = f"""
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

{TRADES_TABLE_SQL}

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
            self._migrate_trades_table(conn)

    @staticmethod
    def _migrate_trades_table(conn: sqlite3.Connection) -> None:
        """
        Idempotent migration upgrading legacy DKeX-only `trades` tables (with strict
        CHECK constraints) to the unified multi-venue (DKeX / DraftKings / FanDuel) schema
        while preserving 100% of existing rows and autoincrement IDs.
        """
        cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        row_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'"
        ).fetchone()
        table_sql = str(row_sql["sql"] if row_sql else "")

        needs_rebuild = (
            "venue" not in cols
            or "closing_odds_american" not in cols
            or "'PUSH'" not in table_sql
            or "CHECK(side IN" in table_sql
        )
        if not needs_rebuild:
            return

        conn.execute("DROP TABLE IF EXISTS trades_migrated")
        conn.execute(
            TRADES_TABLE_SQL.replace(
                "CREATE TABLE IF NOT EXISTS trades", "CREATE TABLE trades_migrated"
            )
        )

        # Copy all columns that exist in the legacy table; newly added columns receive defaults
        target_cols = [
            "id",
            "created_at",
            "sport",
            "event_name",
            "market_type",
            "selection_label",
            "side",
            "entry_price",
            "fee_per_contract",
            "contracts",
            "total_cost",
            "model_prob",
            "fair_exit_price",
            "closing_price",
            "status",
            "exit_or_settle_price",
            "exit_fee_per_contract",
            "pnl",
            "notes",
            "venue",
            "bet_format",
            "odds_american",
            "stake",
            "potential_payout",
            "is_bonus_bet",
            "strategy_tag",
            "group_id",
            "closing_odds_american",
        ]
        shared = [c for c in target_cols if c in cols]
        col_list = ", ".join(shared)
        conn.execute(
            f"INSERT INTO trades_migrated ({col_list}) SELECT {col_list} FROM trades"
        )
        conn.execute("DROP TABLE trades")
        conn.execute("ALTER TABLE trades_migrated RENAME TO trades")

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

    # --- Unified Multi-Venue Trade Ledger ---
    @staticmethod
    def _normalize_venue(raw: str | None) -> str:
        s = str(raw or "DKEX").strip().upper()
        if s in ("DK", "DRAFTKINGS", "DRAFTKINGS SPORTSBOOK"):
            return "DRAFTKINGS"
        if s in ("FD", "FANDUEL", "FANDUEL SPORTSBOOK"):
            return "FANDUEL"
        return "DKEX"

    def log_trade(
        self,
        *,
        created_at: str,
        sport: str,
        event_name: str,
        market_type: str,
        selection_label: str,
        side: str,
        entry_price: float = 0.50,
        fee_per_contract: float = 0.015,
        contracts: int = 1,
        model_prob: float = 0.50,
        fair_exit_price: float = 0.50,
        notes: str = "",
        venue: str = "DKEX",
        bet_format: str | None = None,
        odds_american: int | None = None,
        stake: float | None = None,
        is_bonus_bet: bool = False,
        strategy_tag: str = "EV",
        group_id: str | None = None,
    ) -> dict[str, Any]:
        norm_venue = self._normalize_venue(venue)
        if bet_format is None:
            norm_format = "SPORTSBOOK" if norm_venue in ("DRAFTKINGS", "FANDUEL") or (
                odds_american is not None and stake is not None
            ) else "CONTRACT"
        else:
            norm_format = str(bet_format).strip().upper()

        norm_strategy = str(strategy_tag or "EV").strip().upper()

        if norm_format == "SPORTSBOOK":
            if odds_american is None or stake is None or float(stake) <= 0:
                raise ValueError("odds_american and positive stake are required for SPORTSBOOK trades.")
            am_int = int(round(float(odds_american)))
            dec = american_to_decimal(am_int)
            stake_val = round(float(stake), 2)
            total_cost = 0.0 if is_bonus_bet else stake_val
            pot_payout = (
                round(stake_val * (dec - 1.0), 2)
                if is_bonus_bet
                else round(stake_val * dec, 2)
            )
            eff_entry_px = round(1.0 / dec, 4)
            eff_fee = 0.0
            eff_contracts = 1
            eff_fair_exit = round(max(0.01, min(0.99, float(model_prob))), 4)
        else:
            am_int = odds_american
            stake_val = None
            eff_entry_px = float(entry_price)
            eff_fee = float(fee_per_contract)
            eff_contracts = int(contracts)
            total_cost = round(eff_contracts * (eff_entry_px + eff_fee), 4)
            pot_payout = round(eff_contracts * 1.0, 2)
            eff_fair_exit = float(fair_exit_price)

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades (
                    created_at, sport, event_name, market_type, selection_label,
                    side, entry_price, fee_per_contract, contracts, total_cost,
                    model_prob, fair_exit_price, status, pnl, notes,
                    venue, bet_format, odds_american, stake, potential_payout,
                    is_bonus_bet, strategy_tag, group_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0.0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    sport.upper(),
                    event_name,
                    market_type,
                    selection_label,
                    side.upper(),
                    eff_entry_px,
                    eff_fee,
                    eff_contracts,
                    total_cost,
                    float(model_prob),
                    eff_fair_exit,
                    notes,
                    norm_venue,
                    norm_format,
                    am_int,
                    stake_val,
                    pot_payout,
                    1 if is_bonus_bet else 0,
                    norm_strategy,
                    group_id,
                ),
            )
            trade_id = cur.lastrowid
            row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(row)

    def log_trade_pair(
        self,
        leg_a: dict[str, Any],
        leg_b: dict[str, Any],
        *,
        group_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically log a paired two-leg strategy (Arbitrage, Middle, Bonus Bet Hedge)."""
        gid = group_id or f"pair_{int(time.time() * 1000)}_{secrets.token_hex(3)}"
        created_a = self.log_trade(**{**leg_a, "group_id": gid})
        created_b = self.log_trade(**{**leg_b, "group_id": gid})
        return [created_a, created_b]

    def update_trade(
        self,
        trade_id: int,
        *,
        status: str,
        exit_or_settle_price: float | None = None,
        exit_fee_per_contract: float = 0.0,
        closing_price: float | None = None,
        closing_odds_american: int | None = None,
        cashout_amount: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        status_u = status.upper()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
            if row is None:
                return None

            bet_fmt = str(row["bet_format"] or "CONTRACT").upper()
            contracts = int(row["contracts"])
            total_cost = float(row["total_cost"])
            is_bonus = bool(row["is_bonus_bet"])

            if bet_fmt == "SPORTSBOOK":
                stake_val = float(row["stake"] if row["stake"] is not None else total_cost)
                am_odds = int(row["odds_american"] if row["odds_american"] is not None else -110)
                dec = american_to_decimal(am_odds)
                cost_basis = 0.0 if is_bonus else stake_val
                exit_fee = 0.0

                if status_u == "WON":
                    profit = stake_val * (dec - 1.0)
                    settle_px = round(stake_val * (dec - 1.0) if is_bonus else stake_val * dec, 2)
                    pnl = round(profit, 2)
                elif status_u == "LOST":
                    settle_px = 0.0
                    pnl = round(-cost_basis, 2)
                elif status_u in ("PUSH", "VOID"):
                    settle_px = round(cost_basis, 2)
                    pnl = 0.0
                elif status_u in ("CASHOUT", "SOLD"):
                    co_val = cashout_amount if cashout_amount is not None else exit_or_settle_price
                    if co_val is None:
                        raise ValueError("cashout_amount required when status is CASHOUT")
                    settle_px = round(float(co_val), 2)
                    pnl = round(settle_px - cost_basis, 2)
                    status_u = "CASHOUT"
                else:
                    settle_px = None
                    pnl = 0.0
            else:
                if status_u == "WON":
                    settle_px = 1.0
                    exit_fee = 0.0  # Expiration settlement at $1.00 incurs no sell trade fee
                    gross_payout = contracts * 1.0
                    pnl = round(gross_payout - total_cost, 2)
                elif status_u == "LOST":
                    settle_px = 0.0
                    exit_fee = 0.0
                    pnl = round(-total_cost, 2)
                elif status_u in ("PUSH", "VOID"):
                    settle_px = round(float(row["entry_price"]), 4)
                    exit_fee = 0.0
                    pnl = 0.0
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
            new_closing_am = (
                int(closing_odds_american)
                if closing_odds_american is not None
                else row["closing_odds_american"]
            )
            if new_closing_am is not None and new_closing is None:
                new_closing = round(american_to_implied_prob(new_closing_am), 4)
            new_notes = notes if notes is not None else row["notes"]

            conn.execute(
                """
                UPDATE trades
                SET status = ?,
                    exit_or_settle_price = ?,
                    exit_fee_per_contract = ?,
                    closing_price = ?,
                    closing_odds_american = ?,
                    pnl = ?,
                    notes = ?
                WHERE id = ?
                """,
                (
                    status_u,
                    settle_px,
                    exit_fee,
                    new_closing,
                    new_closing_am,
                    pnl,
                    new_notes,
                    trade_id,
                ),
            )
            updated = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(updated)

    def delete_trade(self, trade_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
            return cur.rowcount > 0

    def list_trades(self, venue: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if venue and venue.upper() != "ALL":
                norm_v = self._normalize_venue(venue)
                rows = conn.execute(
                    "SELECT * FROM trades WHERE venue = ? ORDER BY id DESC",
                    (norm_v,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM trades ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _summarize_trade_subset(
        trades: list[dict[str, Any]], starting_bankroll: float
    ) -> dict[str, Any]:
        settled_statuses = ("WON", "LOST", "SOLD", "PUSH", "VOID", "CASHOUT")
        settled = [t for t in trades if t["status"] in settled_statuses]
        decided = [t for t in settled if t["status"] not in ("PUSH", "VOID")]
        open_trades = [t for t in trades if t["status"] == "OPEN"]
        realized_pnl = round(sum(float(t["pnl"]) for t in settled), 2)
        open_exposure = round(sum(float(t["total_cost"]) for t in open_trades), 2)
        total_settled_outlay = sum(
            float(t["total_cost"])
            for t in settled
            if t["status"] not in ("PUSH", "VOID")
        )
        wins = sum(1 for t in decided if float(t["pnl"]) > 0)
        hit_rate = round(wins / len(decided), 4) if decided else None
        expected_win_rate = (
            round(sum(float(t["model_prob"]) for t in decided) / len(decided), 4)
            if decided
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

    def ledger_summary(
        self,
        starting_bankroll: float,
        *,
        dk_bankroll: float | None = None,
        fd_bankroll: float | None = None,
    ) -> dict[str, Any]:
        trades = self.list_trades()
        overall = self._summarize_trade_subset(trades, starting_bankroll)

        dk_start = dk_bankroll if dk_bankroll is not None else round(starting_bankroll * 0.5, 2)
        fd_start = fd_bankroll if fd_bankroll is not None else round(starting_bankroll * 0.5, 2)

        by_venue = {
            "DRAFTKINGS": self._summarize_trade_subset(
                [t for t in trades if str(t.get("venue") or "DKEX").upper() == "DRAFTKINGS"],
                dk_start,
            ),
            "FANDUEL": self._summarize_trade_subset(
                [t for t in trades if str(t.get("venue") or "DKEX").upper() == "FANDUEL"],
                fd_start,
            ),
            "DKEX": self._summarize_trade_subset(
                [t for t in trades if str(t.get("venue") or "DKEX").upper() == "DKEX"],
                starting_bankroll,
            ),
        }
        overall["by_venue"] = by_venue
        return overall

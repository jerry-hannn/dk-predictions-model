"""Ensemble blending, live in-play conditioning, dual-gate DKeX filtering, and slate orchestrator."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from src.config import PROJECT_ROOT, Settings, get_settings
from src.data.espn_client import ESPNClient
from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.data.odds_client import OddsClient
from src.engine.dk_math import (
    ContractEvaluation,
    clamp_prob,
    evaluate_single_side,
    evaluate_two_way_contract,
    inv_logit,
    logit,
)
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import PlayerPropsEngine
from src.storage.db import Database


def blend_probabilities(
    model_prob: float,
    market_prob: float | None,
    market_weight: float = 0.42,
) -> float:
    """Log-opinion-pool blend of calibrated fundamental model and de-vigged market prior."""
    pm = clamp_prob(model_prob)
    if market_prob is None:
        return pm
    pk = clamp_prob(market_prob)
    w = max(0.0, min(1.0, market_weight))
    return clamp_prob(inv_logit(w * logit(pk) + (1.0 - w) * logit(pm)))


def _infer_dkex_ask_pair(
    *,
    dk_yes_prob: float | None,
    sharp_yes_prob: float | None,
    blended_yes_prob: float,
    is_live: bool = False,
) -> tuple[float, float]:
    """
    Infer realistic DraftKings Predictions (DKeX) YES and NO reference ask prices ($0.02-$0.98)
    from DraftKings' own live implied probability (or sharp consensus if DK book unavailable),
    adding realistic DKeX half-spread (+0.8c pre-game, +1.5c live in-play).
    """
    half_spread = 0.015 if is_live else 0.008
    if dk_yes_prob is not None:
        mid_yes = clamp_prob(dk_yes_prob, 0.02)
    elif sharp_yes_prob is not None:
        mid_yes = clamp_prob(sharp_yes_prob, 0.02)
    elif is_live:
        # In-play without a sportsbook quote: exchange book tracks live win probability closely
        mid_yes = clamp_prob(blended_yes_prob, 0.01)
    else:
        # Pre-game fallback when no sportsbook price exists on that specific line
        mid_yes = clamp_prob(0.50 + 0.55 * (blended_yes_prob - 0.50), 0.04)

    yes_ask = round(min(0.98, max(0.02, mid_yes + half_spread)), 2)
    no_ask = round(min(0.98, max(0.02, (1.0 - mid_yes) + half_spread)), 2)
    return yes_ask, no_ask


def _force_pass_evaluation(ev: ContractEvaluation) -> ContractEvaluation:
    """Zero out contract recommendation when a live safety lock triggers."""
    return ContractEvaluation(
        side=ev.side,
        contract_price=ev.contract_price,
        fee_per_contract=ev.fee_per_contract,
        all_in_cost=ev.all_in_cost,
        breakeven_prob=ev.breakeven_prob,
        model_prob=ev.model_prob,
        conservative_prob=ev.conservative_prob,
        net_edge=ev.net_edge,
        ev_per_contract=ev.ev_per_contract,
        roi_pct=ev.roi_pct,
        fair_entry_max_price=ev.fair_entry_max_price,
        fair_midgame_exit_price=ev.fair_midgame_exit_price,
        kelly_fraction_applied=0.0,
        recommended_contracts=0,
        recommended_outlay=0.0,
        american_equivalent=ev.american_equivalent,
        confidence_score=min(ev.confidence_score, 55),
        confidence_tier="PASS",
    )


class ModelHub:
    """Manages training, persistence, live data syncing, and pre-game/in-play DKeX picks."""

    def __init__(
        self,
        settings: Settings | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db or Database(self.settings.db_path)
        self.nfl_loader = NFLLoader()
        self.nba_loader = NBALoader()
        self.espn = ESPNClient(db=self.db)
        self.odds = OddsClient(api_key=self.settings.the_odds_api_key, db=self.db)

        self.nfl_model: NFLGameModel | None = None
        self.nba_model: NBAGameModel | None = None
        self.props_engine: PlayerPropsEngine | None = None

    @property
    def props_path(self) -> Path:
        return self.settings.models_dir / "player_props_engine.joblib"

    def refresh_live_feeds(
        self, sport: str = "all", target_date: str | None = None
    ) -> dict[str, Any]:
        """
        Lightweight live refresh: updates ONLY live ESPN scoreboards, injuries, live box scores,
        and live Odds API lines (skipping slow multi-season historical CSV downloads).
        """
        sp = sport.lower()
        out: dict[str, Any] = {}
        for lg in (["NFL", "NBA"] if sp == "all" else [sp.upper()]):
            sb = self.espn.fetch_scoreboard(lg, target_date=target_date, force_refresh=True)
            live_evs = [e for e in sb if e.get("state") == "IN"]
            for lev in live_evs:
                self.espn.fetch_live_boxscore(lg, lev["event_id"], force_refresh=True)
            inj = self.espn.fetch_league_injuries(lg, force_refresh=True)
            odds = self.odds.fetch_game_odds(
                lg, force_refresh=True, live_mode=bool(live_evs)
            )
            out[lg.lower()] = {
                "scoreboard_events": len(sb),
                "live_in_play_events": len(live_evs),
                "injured_teams_tracked": len(inj),
                "sharp_odds_events": len(odds),
            }
        out["odds_quota"] = self.odds.get_quota_status()
        return out

    def sync_data(
        self, sport: str = "all", target_date: str | None = None, force: bool = True
    ) -> dict[str, Any]:
        sp = sport.lower()
        summary: dict[str, Any] = {}
        if sp in ("all", "nfl"):
            nfl_sched = self.nfl_loader.load_schedules(force_refresh=force)
            nfl_logs = self.nfl_loader.load_player_logs(force_refresh=force)
            nfl_sb = self.espn.fetch_scoreboard("NFL", target_date=target_date, force_refresh=force)
            nfl_live = [e for e in nfl_sb if e.get("state") == "IN"]
            for lev in nfl_live:
                self.espn.fetch_live_boxscore("NFL", lev["event_id"], force_refresh=force)
            nfl_inj = self.espn.fetch_league_injuries("NFL", force_refresh=force)
            nfl_odds = self.odds.fetch_game_odds("NFL", force_refresh=force, live_mode=bool(nfl_live))
            summary["nfl"] = {
                "historical_games": len(nfl_sched),
                "player_log_rows": len(nfl_logs),
                "live_scoreboard_events": len(nfl_sb),
                "in_play_events": len(nfl_live),
                "injured_teams_tracked": len(nfl_inj),
                "sharp_odds_events": len(nfl_odds),
            }
        if sp in ("all", "nba"):
            nba_games = self.nba_loader.load_games(force_refresh=force)
            nba_logs = self.nba_loader.load_player_logs(force_refresh=force)
            nba_sb = self.espn.fetch_scoreboard("NBA", target_date=target_date, force_refresh=force)
            nba_live = [e for e in nba_sb if e.get("state") == "IN"]
            for lev in nba_live:
                self.espn.fetch_live_boxscore("NBA", lev["event_id"], force_refresh=force)
            nba_inj = self.espn.fetch_league_injuries("NBA", force_refresh=force)
            nba_odds = self.odds.fetch_game_odds("NBA", force_refresh=force, live_mode=bool(nba_live))
            summary["nba"] = {
                "historical_games": len(nba_games),
                "player_log_rows": len(nba_logs),
                "live_scoreboard_events": len(nba_sb),
                "in_play_events": len(nba_live),
                "injured_teams_tracked": len(nba_inj),
                "sharp_odds_events": len(nba_odds),
            }
        summary["odds_quota"] = self.odds.get_quota_status()
        return summary

    def train_all(self, sport: str = "all") -> dict[str, Any]:
        sp = sport.lower()
        metrics: dict[str, Any] = {}
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)

        nfl_logs = self.nfl_loader.load_player_logs()
        nba_logs = self.nba_loader.load_player_logs()

        if sp in ("all", "nfl"):
            nfl_sched = self.nfl_loader.load_schedules()
            self.nfl_model = NFLGameModel()
            metrics["nfl_game"] = self.nfl_model.fit(nfl_sched)
            self.nfl_model.save(self.settings.models_dir / "nfl_game_model.joblib")

        if sp in ("all", "nba"):
            nba_games = self.nba_loader.load_games()
            self.nba_model = NBAGameModel()
            metrics["nba_game"] = self.nba_model.fit(nba_games, player_logs=nba_logs)
            self.nba_model.save(self.settings.models_dir / "nba_game_model.joblib")

        self.props_engine = PlayerPropsEngine()
        self.props_engine.fit(nfl_logs=nfl_logs, nba_logs=nba_logs)
        joblib.dump(self.props_engine, self.props_path)
        metrics["player_props"] = {
            "nfl_log_rows": len(nfl_logs),
            "nba_log_rows": len(nba_logs),
        }
        return metrics

    def ensure_loaded(self) -> None:
        primary_dir = self.settings.models_dir
        fallback_dir = PROJECT_ROOT / "models"
        allow_fallback = "PYTEST_CURRENT_TEST" not in os.environ

        def _find_artifact(filename: str) -> Path | None:
            cand = primary_dir / filename
            if cand.exists():
                return cand
            if allow_fallback:
                fb = fallback_dir / filename
                if fb.exists():
                    return fb
            return None

        nfl_path = _find_artifact("nfl_game_model.joblib")
        nba_path = _find_artifact("nba_game_model.joblib")
        props_path = _find_artifact("player_props_engine.joblib")
        if not (nfl_path and nba_path and props_path):
            self.train_all("all")
            return
        if self.nfl_model is None:
            self.nfl_model = NFLGameModel.load(nfl_path)
        if self.nba_model is None:
            self.nba_model = NBAGameModel.load(nba_path)
        if self.props_engine is None:
            self.props_engine = joblib.load(props_path)

    def _get_upcoming_events(
        self,
        sport: str,
        target_date: str | None = None,
        mode: str = "all",
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get active events filtered by mode:
        - 'all': both PRE and IN (live in-play)
        - 'live': strictly IN (live in-play right now)
        - 'pregame': strictly PRE (unstarted upcoming games)
        """
        sport_u = sport.upper()
        mode_l = mode.lower()
        events = self.espn.fetch_scoreboard(
            sport_u, target_date=target_date, force_refresh=force_refresh
        )

        if mode_l == "live":
            return [e for e in events if e.get("state") == "IN"]
        if mode_l == "pregame":
            pre = [e for e in events if e.get("state") == "PRE"]
            if pre or events:
                return pre

        active = [e for e in events if e.get("state") in ("PRE", "IN")]
        if active:
            return active
        if events and mode_l == "all":
            return events[:6]

        # Offline fallback slate (only when completely offline in 'all'/'pregame' mode)
        if sport_u == "NFL":
            sched = self.nfl_loader.load_schedules()
            unplayed = sched[sched["home_score"].isna()].head(10)
            if len(unplayed) == 0:
                unplayed = sched.tail(8)
            out = []
            for _, r in unplayed.iterrows():
                h, a = str(r["home_team"]).upper(), str(r["away_team"]).upper()
                sp_val = float(r["home_spread"]) if not (r["home_spread"] != r["home_spread"]) else -2.5
                tot_val = float(r["total_line"]) if not (r["total_line"] != r["total_line"]) else 45.5
                mkt_p = clamp_prob(inv_logit(-sp_val * 0.145))
                out.append(
                    {
                        "event_id": str(r["game_id"]),
                        "sport": "NFL",
                        "event_name": f"{a} @ {h}",
                        "short_name": f"{a} @ {h}",
                        "start_time": f"{r['gameday']}T18:00:00Z",
                        "state": "PRE",
                        "status": "STATUS_SCHEDULED",
                        "status_detail": str(r["gameday"]),
                        "period": 0,
                        "clock_seconds": 0.0,
                        "display_clock": "0:00",
                        "elapsed_fraction": 0.0,
                        "remaining_minutes": 60.0,
                        "home_score": 0,
                        "away_score": 0,
                        "home_abbr": h,
                        "away_abbr": a,
                        "home_name": h,
                        "away_name": a,
                        "indoor": bool(r.get("is_indoor", 0)),
                        "temp_f": float(r.get("temp", 64.0)),
                        "wind_mph": float(r.get("wind", 6.0)),
                        "market": {
                            "home_spread": sp_val,
                            "total_line": tot_val,
                            "home_win_prob": round(mkt_p, 4),
                            "home_cover_prob": 0.50,
                            "over_prob": 0.50,
                            "line_source": "Schedule Archive",
                        },
                    }
                )
            return out

        cur_date = target_date or datetime.now().strftime("%Y-%m-%d")
        matchups = [
            ("BOS", "NYK", -5.5, 226.5, 0.66),
            ("OKC", "DAL", -6.0, 231.5, 0.68),
            ("DEN", "MIN", -3.5, 223.5, 0.60),
            ("CLE", "MIL", -4.5, 229.5, 0.63),
            ("LAL", "PHX", -2.5, 232.5, 0.57),
            ("GSW", "SAC", -3.0, 234.5, 0.58),
        ]
        return [
            {
                "event_id": f"NBA_{cur_date}_{a}_{h}",
                "sport": "NBA",
                "event_name": f"{a} @ {h}",
                "short_name": f"{a} @ {h}",
                "start_time": f"{cur_date}T23:30:00Z",
                "state": "PRE",
                "status": "STATUS_SCHEDULED",
                "status_detail": cur_date,
                "period": 0,
                "clock_seconds": 0.0,
                "display_clock": "0:00",
                "elapsed_fraction": 0.0,
                "remaining_minutes": 48.0,
                "home_score": 0,
                "away_score": 0,
                "home_abbr": h,
                "away_abbr": a,
                "home_name": h,
                "away_name": a,
                "indoor": True,
                "temp_f": 70.0,
                "wind_mph": 0.0,
                "market": {
                    "home_spread": sp,
                    "total_line": tot,
                    "home_win_prob": mkt_p,
                    "home_cover_prob": 0.50,
                    "over_prob": 0.50,
                    "line_source": "Offline Fallback",
                },
            }
            for h, a, sp, tot, mkt_p in matchups
        ]

    def generate_picks(
        self,
        *,
        sport: str = "all",
        market_filter: str = "all",
        mode: str = "all",
        target_date: str | None = None,
        min_win_prob: float | None = None,
        min_net_edge: float | None = None,
        fee_per_contract: float | None = None,
        bankroll: float | None = None,
        include_passes: bool = False,
        fresh: bool = False,
        live_props: bool = False,
    ) -> list[dict[str, Any]]:
        self.ensure_loaded()
        assert self.nfl_model is not None
        assert self.nba_model is not None
        assert self.props_engine is not None

        eff_min_prob = min_win_prob if min_win_prob is not None else self.settings.min_win_prob
        # Balanced 50/50 markets (Spreads, Totals, Main-Line Props) are anchored near 50¢;
        # a sharp calibrated play sits at 55.5%-59% true win prob (+3% to +6% net edge), whereas
        # outright Moneylines span 10¢-90¢ and enforce the full 60% favorite floor by default.
        eff_balanced_min_prob = (
            min_win_prob if min_win_prob is not None else min(eff_min_prob, 0.555)
        )
        eff_min_edge = min_net_edge if min_net_edge is not None else self.settings.min_net_edge
        eff_fee = fee_per_contract if fee_per_contract is not None else self.settings.default_fee_dollars
        eff_bankroll = bankroll if bankroll is not None else self.settings.bankroll
        slate_date = target_date or datetime.now().strftime("%Y-%m-%d")

        sports = ["NFL", "NBA"] if sport.lower() == "all" else [sport.upper()]
        candidates: list[dict[str, Any]] = []

        for sp in sports:
            events = self._get_upcoming_events(
                sp, target_date=target_date, mode=mode, force_refresh=fresh
            )
            if not events:
                continue
            has_live = any(ev.get("state") == "IN" for ev in events)
            injuries = self.espn.fetch_league_injuries(sp, force_refresh=fresh)
            sharp_odds = self.odds.fetch_game_odds(
                sp, force_refresh=fresh, live_mode=has_live
            )

            for ev in events:
                h = ev["home_abbr"]
                a = ev["away_abbr"]
                is_live = ev.get("state") == "IN"
                mkt = dict(ev.get("market", {}) or {})

                live_box_players: dict[str, dict[str, Any]] | None = None
                if is_live:
                    summary_data = self.espn.fetch_live_boxscore(
                        sp, ev["event_id"], force_refresh=fresh
                    )
                    live_box_players = summary_data.get("players") or {}
                    if ev.get("live_win_prob") is None and summary_data.get("live_win_prob") is not None:
                        ev["live_win_prob"] = summary_data["live_win_prob"]
                    # ESPN summary `pickcenter` holds pre-game open/close lines — copy only opening anchors in-play
                    summary_mkt = summary_data.get("market") or {}
                    for open_k in ("open_home_spread", "open_total_line", "open_home_win_prob"):
                        if open_k in summary_mkt and open_k not in mkt:
                            mkt[open_k] = summary_mkt[open_k]

                consensus = (
                    self.odds.extract_consensus(sharp_odds, ev["home_name"], ev["away_name"])
                    if sharp_odds
                    else None
                )
                if consensus:
                    cons_age = consensus.get("odds_age_seconds")
                    # Never overwrite an active IN-PLAY game with a >5-minute-old stale sportsbook snapshot
                    if is_live and cons_age is not None and float(cons_age) > 300.0:
                        mkt.setdefault("odds_age_seconds", cons_age)
                        if consensus.get("odds_event_id"):
                            mkt["odds_event_id"] = consensus["odds_event_id"]
                    else:
                        # Preserve ESPN opening-line movement fields when merging live sharp consensus
                        for keep_key in ("open_home_spread", "open_total_line", "open_home_win_prob"):
                            if keep_key in mkt and keep_key not in consensus:
                                consensus[keep_key] = mkt[keep_key]
                        if "open_home_spread" in consensus and "home_spread" in consensus:
                            consensus["spread_move"] = round(
                                float(consensus["home_spread"]) - float(consensus["open_home_spread"]), 2
                            )
                        if "open_total_line" in consensus and "total_line" in consensus:
                            consensus["total_move"] = round(
                                float(consensus["total_line"]) - float(consensus["open_total_line"]), 2
                            )
                        mkt.update(consensus)

                if is_live and "home_win_prob" not in mkt and "home_spread" not in mkt:
                    mkt["no_live_book_feed"] = True
                    mkt["line_source"] = "ESPN Live PBP (Verify DK ¢)"
                    if ev.get("live_win_prob") is not None:
                        mkt["home_win_prob"] = round(float(ev["live_win_prob"]), 4)

                live_prop_mkts: dict[tuple[str, str], dict[str, Any]] | None = None
                if (
                    live_props
                    and market_filter.lower() in ("all", "props")
                    and mkt.get("odds_event_id")
                ):
                    live_prop_mkts = self.odds.fetch_event_player_props(
                        sp,
                        str(mkt["odds_event_id"]),
                        force_refresh=fresh,
                        live_mode=is_live,
                    )

                inj_notes = self._format_injury_summary(injuries, h, a)

                if sp == "NFL":
                    gpred = self.nfl_model.predict_matchup(
                        home_team=h,
                        away_team=a,
                        spread_line=mkt.get("home_spread"),
                        total_line=mkt.get("total_line"),
                        is_indoor=bool(ev.get("indoor", False)),
                        wind_mph=float(ev.get("wind_mph", 5.0)),
                        temp_f=float(ev.get("temp_f", 65.0)),
                        injuries_by_team=injuries,
                        live_state=ev if is_live else None,
                    )
                    prop_preds = (
                        self.props_engine.project_nfl_game_props(
                            home_team=h,
                            away_team=a,
                            expected_spread=gpred.spread_line,
                            expected_total=gpred.expected_total,
                            is_indoor=bool(ev.get("indoor", False)),
                            wind_mph=float(ev.get("wind_mph", 5.0)),
                            injuries_by_team=injuries,
                            live_prop_markets=live_prop_mkts,
                            live_state=ev if is_live else None,
                            live_boxscore=live_box_players,
                        )
                        if market_filter.lower() in ("all", "props")
                        else []
                    )
                else:
                    gpred = self.nba_model.predict_matchup(
                        home_team=h,
                        away_team=a,
                        game_date=slate_date,
                        spread_line=mkt.get("home_spread"),
                        total_line=mkt.get("total_line"),
                        injuries_by_team=injuries,
                        live_state=ev if is_live else None,
                    )
                    prop_preds = (
                        self.props_engine.project_nba_game_props(
                            home_team=h,
                            away_team=a,
                            expected_pace=gpred.expected_pace,
                            expected_spread=gpred.spread_line,
                            injuries_by_team=injuries,
                            live_prop_markets=live_prop_mkts,
                            live_state=ev if is_live else None,
                            live_boxscore=live_box_players,
                        )
                        if market_filter.lower() in ("all", "props")
                        else []
                    )

                if market_filter.lower() in ("all", "game", "moneyline", "winner"):
                    candidates.append(
                        self._build_game_winner_pick(
                            slate_date=slate_date,
                            ev=ev,
                            gpred=gpred,
                            mkt=mkt,
                            inj_notes=inj_notes,
                            eff_fee=eff_fee,
                            eff_bankroll=eff_bankroll,
                            eff_min_prob=eff_min_prob,
                            eff_min_edge=eff_min_edge,
                        )
                    )

                if market_filter.lower() in ("all", "game", "spread"):
                    candidates.append(
                        self._build_spread_pick(
                            slate_date=slate_date,
                            ev=ev,
                            gpred=gpred,
                            mkt=mkt,
                            inj_notes=inj_notes,
                            eff_fee=eff_fee,
                            eff_bankroll=eff_bankroll,
                            eff_min_prob=eff_balanced_min_prob,
                            eff_min_edge=eff_min_edge,
                        )
                    )

                if market_filter.lower() in ("all", "game", "total"):
                    candidates.append(
                        self._build_total_pick(
                            slate_date=slate_date,
                            ev=ev,
                            gpred=gpred,
                            mkt=mkt,
                            inj_notes=inj_notes,
                            eff_fee=eff_fee,
                            eff_bankroll=eff_bankroll,
                            eff_min_prob=eff_balanced_min_prob,
                            eff_min_edge=eff_min_edge,
                        )
                    )

                for pp in prop_preds:
                    candidates.append(
                        self._build_prop_pick(
                            slate_date=slate_date,
                            ev=ev,
                            pp=pp,
                            mkt=mkt,
                            inj_notes=inj_notes,
                            eff_fee=eff_fee,
                            eff_bankroll=eff_bankroll,
                            eff_min_prob=eff_balanced_min_prob,
                            eff_min_edge=eff_min_edge,
                        )
                    )

        candidates.sort(
            key=lambda x: (
                x["confidence_tier"] != "PASS",
                x["confidence_score"],
                x["net_edge"],
                x["model_prob"],
            ),
            reverse=True,
        )
        self._apply_same_game_caps(candidates, eff_bankroll)

        actionable = [c for c in candidates if c["confidence_tier"] != "PASS"]
        if actionable:
            self.db.save_predictions(actionable)

        return candidates if include_passes else actionable

    def _live_safety_context(
        self, ev: dict[str, Any], mkt: dict[str, Any], base_clarity: float
    ) -> tuple[bool, bool, float, float, list[str]]:
        """
        Compute (is_live, endgame_locked, kelly_mult, adj_clarity, extra_drivers) for live risk management.
        """
        is_live = ev.get("state") == "IN"
        kelly_mult = self.settings.kelly_fraction * (0.80 if is_live else 1.0)
        clarity = base_clarity
        extra_drivers: list[str] = []

        if not is_live:
            return False, False, kelly_mult, clarity, extra_drivers

        rem_mins = float(ev.get("remaining_minutes", 48.0) or 0.0)
        close_thresh = 8.0 if ev.get("sport") == "NFL" else 6.0
        cur_diff = abs(int(ev.get("home_score", 0) or 0) - int(ev.get("away_score", 0) or 0))
        endgame_locked = rem_mins < 1.5 and cur_diff <= close_thresh
        if endgame_locked:
            extra_drivers.append(
                "Endgame latency lock (<90s in 1-score game — high order-book whipsaw risk)"
            )

        odds_age = mkt.get("odds_age_seconds")
        if mkt.get("no_live_book_feed"):
            endgame_locked = True
            clarity = min(clarity, 0.55)
            extra_drivers.append(
                "No live sportsbook quote (<5m) — refresh live feed or check exact ¢ on DK"
            )
        elif odds_age is not None and float(odds_age) > 240.0:
            clarity = min(clarity, 0.58)
            if float(odds_age) > 480.0:
                endgame_locked = True
            extra_drivers.append(
                f"Live odds {int(float(odds_age)//60)}m old — click Sync/Refresh or verify ¢ on DK"
            )
        return True, endgame_locked, kelly_mult, clarity, extra_drivers

    @staticmethod
    def _event_live_fields(ev: dict[str, Any], mkt: dict[str, Any]) -> dict[str, Any]:
        is_live = ev.get("state") == "IN"
        score_disp = (
            f"{ev['away_abbr']} {ev.get('away_score', 0)} - {ev['home_abbr']} {ev.get('home_score', 0)}"
            if is_live
            else ""
        )
        moves: list[str] = []
        if mkt.get("spread_move") and abs(float(mkt["spread_move"])) >= 0.5:
            moves.append(
                f"Spread {mkt['open_home_spread']:+.1f} → {mkt['home_spread']:+.1f}"
            )
        if mkt.get("total_move") and abs(float(mkt["total_move"])) >= 1.0:
            moves.append(
                f"Total {mkt['open_total_line']:.1f} → {mkt['total_line']:.1f}"
            )
        return {
            "game_state": "LIVE" if is_live else str(ev.get("state", "PRE")),
            "status_detail": str(ev.get("status_detail", "")),
            "score_display": score_disp,
            "remaining_minutes": float(ev.get("remaining_minutes", 48.0)),
            "line_source": str(mkt.get("line_source", "ESPN/DK")),
            "odds_age_seconds": mkt.get("odds_age_seconds"),
            "line_movement": " | ".join(moves),
        }

    def _build_game_winner_pick(
        self,
        *,
        slate_date: str,
        ev: dict[str, Any],
        gpred: Any,
        mkt: dict[str, Any],
        inj_notes: str,
        eff_fee: float,
        eff_bankroll: float,
        eff_min_prob: float,
        eff_min_edge: float,
    ) -> dict[str, Any]:
        h, a = gpred.home_team, gpred.away_team
        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, gpred.injury_clarity
        )
        sharp_home_p = mkt.get("sharp_home_win_prob", mkt.get("home_win_prob"))
        dk_home_p = mkt.get("dk_home_win_prob")
        has_real_book = str(mkt.get("line_source", "")) != "Offline Fallback" and sharp_home_p is not None

        if has_real_book:
            mkt_w = (0.68 if int(mkt.get("books_count", 0) or 0) >= 3 else 0.58) if is_live else 0.60
        else:
            mkt_w = 0.35
        blended_home_p = blend_probabilities(gpred.home_win_prob, sharp_home_p, market_weight=mkt_w)

        ref_home_ask, ref_away_ask = _infer_dkex_ask_pair(
            dk_yes_prob=dk_home_p,
            sharp_yes_prob=sharp_home_p,
            blended_yes_prob=blended_home_p,
            is_live=is_live,
        )

        best, _, _ = evaluate_two_way_contract(
            yes_true_prob=blended_home_p,
            yes_ask_price=ref_home_ask,
            no_ask_price=ref_away_ask,
            fee_per_contract=eff_fee,
            bankroll=eff_bankroll,
            kelly_multiplier=kelly_m,
            max_bet_pct=self.settings.max_single_bet_pct,
            min_win_prob=eff_min_prob,
            min_net_edge=eff_min_edge,
            yes_anchor_prob=sharp_home_p,
            uncertainty=0.13 if is_live else 0.10,
            injury_clarity=clarity,
        )
        if locked or (is_live and dk_home_p is None and sharp_home_p is None):
            best = _force_pass_evaluation(best)
            if is_live and dk_home_p is None and sharp_home_p is None and not locked:
                safety_drivers = safety_drivers + [
                    "Moneyline off-board at live sportsbooks (blowout/suspended — check exact ¢ on DK)"
                ]

        label = f"{h} to Win (Outright)" if best.side == "YES" else f"{a} to Win (or {h} Win: NO)"
        drivers = list(safety_drivers) + list(gpred.drivers)
        if dk_home_p is not None and sharp_home_p is not None and abs(sharp_home_p - dk_home_p) >= 0.03:
            drivers.insert(
                1 if is_live else 0,
                f"DK price lag: Sharp consensus {sharp_home_p*100:.1f}% vs DK {dk_home_p*100:.1f}%",
            )

        out = {
            "pick_id": f"{slate_date}:{ev['sport']}:{ev['event_id']}:winner",
            "slate_date": slate_date,
            "sport": ev["sport"],
            "event_id": ev["event_id"],
            "event_name": ev["event_name"],
            "short_name": ev["short_name"],
            "start_time": ev["start_time"],
            "market_type": "Live Moneyline" if is_live else "Moneyline",
            "selection_label": label,
            "contract_question": f"Will {h} defeat {a}?",
            "player_name": None,
            "line_value": None,
            "yes_true_prob": round(blended_home_p, 4),
            "recommended_side": best.side,
            "contract_price": best.contract_price,
            "fee_per_contract": best.fee_per_contract,
            "all_in_cost": best.all_in_cost,
            "breakeven_prob": best.breakeven_prob,
            "model_prob": best.model_prob,
            "raw_fundamental_prob": round(
                gpred.home_win_prob if best.side == "YES" else (1.0 - gpred.home_win_prob), 4
            ),
            "market_prob": (
                round(sharp_home_p if best.side == "YES" else (1.0 - sharp_home_p), 4)
                if sharp_home_p is not None
                else None
            ),
            "conservative_prob": best.conservative_prob,
            "net_edge": best.net_edge,
            "ev_per_contract": best.ev_per_contract,
            "roi_pct": best.roi_pct,
            "fair_entry_max_price": best.fair_entry_max_price,
            "fair_midgame_exit_price": best.fair_midgame_exit_price,
            "recommended_contracts": best.recommended_contracts,
            "recommended_outlay": best.recommended_outlay,
            "american_equivalent": best.american_equivalent,
            "confidence_score": best.confidence_score,
            "confidence_tier": best.confidence_tier,
            "drivers": drivers[:4],
            "injury_notes": inj_notes,
        }
        out.update(self._event_live_fields(ev, mkt))
        return out

    def _build_spread_pick(
        self,
        *,
        slate_date: str,
        ev: dict[str, Any],
        gpred: Any,
        mkt: dict[str, Any],
        inj_notes: str,
        eff_fee: float,
        eff_bankroll: float,
        eff_min_prob: float,
        eff_min_edge: float,
    ) -> dict[str, Any]:
        h, a = gpred.home_team, gpred.away_team
        sp_line = gpred.spread_line
        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, gpred.injury_clarity
        )
        mkt_cov = mkt.get("home_cover_prob")
        dk_cov = mkt.get("dk_home_cover_prob")
        has_real_book = str(mkt.get("line_source", "")) != "Offline Fallback" and mkt_cov is not None

        mkt_w = (0.68 if is_live else 0.62) if has_real_book else 0.30
        blended_cov = blend_probabilities(gpred.home_cover_prob, mkt_cov, market_weight=mkt_w)

        yes_ask, no_ask = _infer_dkex_ask_pair(
            dk_yes_prob=dk_cov,
            sharp_yes_prob=mkt_cov,
            blended_yes_prob=blended_cov,
            is_live=is_live,
        )

        best, _, _ = evaluate_two_way_contract(
            yes_true_prob=blended_cov,
            yes_ask_price=yes_ask,
            no_ask_price=no_ask,
            fee_per_contract=eff_fee,
            bankroll=eff_bankroll,
            kelly_multiplier=kelly_m,
            max_bet_pct=self.settings.max_single_bet_pct,
            min_win_prob=eff_min_prob,
            min_net_edge=eff_min_edge,
            yes_anchor_prob=mkt_cov if mkt_cov is not None else 0.50,
            uncertainty=0.15 if is_live else 0.12,
            injury_clarity=clarity,
        )
        if locked or (is_live and mkt_cov is None and dk_cov is None):
            best = _force_pass_evaluation(best)
            if is_live and mkt_cov is None and dk_cov is None and not locked:
                safety_drivers = safety_drivers + [
                    "Spread off-board at live sportsbooks — check exact ¢ on DK"
                ]

        home_fmt = f"{sp_line:+.1f}"
        away_fmt = f"{-sp_line:+.1f}"
        label = (
            f"{h} {home_fmt} Cover"
            if best.side == "YES"
            else f"{a} {away_fmt} Cover ({h} {home_fmt}: NO)"
        )

        drivers = list(safety_drivers)
        sharp_sp = mkt.get("sharp_home_spread")
        if sharp_sp is not None and abs(float(sharp_sp) - sp_line) >= 0.5:
            drivers.append(f"Off-market DK spread {sp_line:+.1f} vs Sharp consensus {float(sharp_sp):+.1f}")
        drivers.append(f"Model proj line {h} {-gpred.expected_margin:+.1f} vs {home_fmt}")
        drivers.extend(gpred.drivers[:2])

        out = {
            "pick_id": f"{slate_date}:{ev['sport']}:{ev['event_id']}:spread",
            "slate_date": slate_date,
            "sport": ev["sport"],
            "event_id": ev["event_id"],
            "event_name": ev["event_name"],
            "short_name": ev["short_name"],
            "start_time": ev["start_time"],
            "market_type": "Live Spread" if is_live else "Spread",
            "selection_label": label,
            "contract_question": f"Will {h} cover {home_fmt}?",
            "player_name": None,
            "line_value": sp_line,
            "yes_true_prob": round(blended_cov, 4),
            "recommended_side": best.side,
            "contract_price": best.contract_price,
            "fee_per_contract": best.fee_per_contract,
            "all_in_cost": best.all_in_cost,
            "breakeven_prob": best.breakeven_prob,
            "model_prob": best.model_prob,
            "raw_fundamental_prob": round(
                gpred.home_cover_prob if best.side == "YES" else (1.0 - gpred.home_cover_prob), 4
            ),
            "market_prob": (
                round(mkt_cov if best.side == "YES" else (1.0 - mkt_cov), 4)
                if mkt_cov is not None
                else 0.50
            ),
            "conservative_prob": best.conservative_prob,
            "net_edge": best.net_edge,
            "ev_per_contract": best.ev_per_contract,
            "roi_pct": best.roi_pct,
            "fair_entry_max_price": best.fair_entry_max_price,
            "fair_midgame_exit_price": best.fair_midgame_exit_price,
            "recommended_contracts": best.recommended_contracts,
            "recommended_outlay": best.recommended_outlay,
            "american_equivalent": best.american_equivalent,
            "confidence_score": best.confidence_score,
            "confidence_tier": best.confidence_tier,
            "drivers": drivers[:4],
            "injury_notes": inj_notes,
        }
        out.update(self._event_live_fields(ev, mkt))
        return out

    def _build_total_pick(
        self,
        *,
        slate_date: str,
        ev: dict[str, Any],
        gpred: Any,
        mkt: dict[str, Any],
        inj_notes: str,
        eff_fee: float,
        eff_bankroll: float,
        eff_min_prob: float,
        eff_min_edge: float,
    ) -> dict[str, Any]:
        tot_line = gpred.total_line
        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, gpred.injury_clarity
        )
        mkt_over = mkt.get("over_prob")
        dk_over = mkt.get("dk_over_prob")
        has_real_book = str(mkt.get("line_source", "")) != "Offline Fallback" and mkt_over is not None

        mkt_w = (0.68 if is_live else 0.62) if has_real_book else 0.30
        blended_over = blend_probabilities(gpred.over_prob, mkt_over, market_weight=mkt_w)

        yes_ask, no_ask = _infer_dkex_ask_pair(
            dk_yes_prob=dk_over,
            sharp_yes_prob=mkt_over,
            blended_yes_prob=blended_over,
            is_live=is_live,
        )

        best, _, _ = evaluate_two_way_contract(
            yes_true_prob=blended_over,
            yes_ask_price=yes_ask,
            no_ask_price=no_ask,
            fee_per_contract=eff_fee,
            bankroll=eff_bankroll,
            kelly_multiplier=kelly_m,
            max_bet_pct=self.settings.max_single_bet_pct,
            min_win_prob=eff_min_prob,
            min_net_edge=eff_min_edge,
            yes_anchor_prob=mkt_over if mkt_over is not None else 0.50,
            uncertainty=0.15 if is_live else 0.12,
            injury_clarity=clarity,
        )
        if locked or (is_live and mkt_over is None and dk_over is None):
            best = _force_pass_evaluation(best)
            if is_live and mkt_over is None and dk_over is None and not locked:
                safety_drivers = safety_drivers + [
                    "Total off-board at live sportsbooks — check exact ¢ on DK"
                ]

        label = (
            f"Over {tot_line:.1f} Total Points"
            if best.side == "YES"
            else f"Under {tot_line:.1f} Total Points (Over: NO)"
        )

        drivers = list(safety_drivers)
        sharp_tot = mkt.get("sharp_total_line")
        if sharp_tot is not None and abs(float(sharp_tot) - tot_line) >= 1.0:
            drivers.append(f"Off-market DK total {tot_line:.1f} vs Sharp {float(sharp_tot):.1f}")
        drivers.append(f"Model projected total {gpred.expected_total:.1f} vs {tot_line:.1f} line")
        drivers.extend(gpred.drivers[:2])

        out = {
            "pick_id": f"{slate_date}:{ev['sport']}:{ev['event_id']}:total",
            "slate_date": slate_date,
            "sport": ev["sport"],
            "event_id": ev["event_id"],
            "event_name": ev["event_name"],
            "short_name": ev["short_name"],
            "start_time": ev["start_time"],
            "market_type": "Live Total" if is_live else "Total",
            "selection_label": label,
            "contract_question": f"Will combined score go Over {tot_line:.1f}?",
            "player_name": None,
            "line_value": tot_line,
            "yes_true_prob": round(blended_over, 4),
            "recommended_side": best.side,
            "contract_price": best.contract_price,
            "fee_per_contract": best.fee_per_contract,
            "all_in_cost": best.all_in_cost,
            "breakeven_prob": best.breakeven_prob,
            "model_prob": best.model_prob,
            "raw_fundamental_prob": round(
                gpred.over_prob if best.side == "YES" else (1.0 - gpred.over_prob), 4
            ),
            "market_prob": (
                round(mkt_over if best.side == "YES" else (1.0 - mkt_over), 4)
                if mkt_over is not None
                else 0.50
            ),
            "conservative_prob": best.conservative_prob,
            "net_edge": best.net_edge,
            "ev_per_contract": best.ev_per_contract,
            "roi_pct": best.roi_pct,
            "fair_entry_max_price": best.fair_entry_max_price,
            "fair_midgame_exit_price": best.fair_midgame_exit_price,
            "recommended_contracts": best.recommended_contracts,
            "recommended_outlay": best.recommended_outlay,
            "american_equivalent": best.american_equivalent,
            "confidence_score": best.confidence_score,
            "confidence_tier": best.confidence_tier,
            "drivers": drivers[:4],
            "injury_notes": inj_notes,
        }
        out.update(self._event_live_fields(ev, mkt))
        return out

    def _build_prop_pick(
        self,
        *,
        slate_date: str,
        ev: dict[str, Any],
        pp: Any,
        mkt: dict[str, Any],
        inj_notes: str,
        eff_fee: float,
        eff_bankroll: float,
        eff_min_prob: float,
        eff_min_edge: float,
    ) -> dict[str, Any]:
        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, pp.injury_clarity
        )
        prop_src = str(getattr(pp, "line_source", "Model Ref Line"))
        has_real_prop_book = prop_src not in ("Model Ref Line", "Live Est Line")
        prop_mkt_w = 0.58 if has_real_prop_book else 0.40
        blended_over = (
            blend_probabilities(pp.over_prob, pp.market_over_prob, market_weight=prop_mkt_w)
            if pp.market_over_prob is not None
            else pp.over_prob
        )

        if pp.dk_over_prob is not None or pp.market_over_prob is not None:
            yes_ask, no_ask = _infer_dkex_ask_pair(
                dk_yes_prob=pp.dk_over_prob,
                sharp_yes_prob=pp.market_over_prob,
                blended_yes_prob=blended_over,
                is_live=is_live,
            )
        else:
            yes_ask = 0.53 if is_live else 0.52
            no_ask = 0.53 if is_live else 0.52

        prop_uncert = (
            (0.19 if is_live else 0.15)
            if has_real_prop_book
            else (0.21 if is_live else 0.17)
        )

        best, _, _ = evaluate_two_way_contract(
            yes_true_prob=blended_over,
            yes_ask_price=yes_ask,
            no_ask_price=no_ask,
            fee_per_contract=eff_fee,
            bankroll=eff_bankroll,
            kelly_multiplier=kelly_m,
            max_bet_pct=min(0.03, self.settings.max_single_bet_pct),
            min_win_prob=eff_min_prob,
            min_net_edge=eff_min_edge,
            yes_anchor_prob=pp.market_over_prob if pp.market_over_prob is not None else 0.50,
            uncertainty=prop_uncert,
            injury_clarity=clarity,
        )
        if locked:
            best = _force_pass_evaluation(best)

        label = (
            f"{pp.player_name} ({pp.team}) Over {pp.line_value:.1f} {pp.prop_label}"
            if best.side == "YES"
            else f"{pp.player_name} ({pp.team}) Under {pp.line_value:.1f} {pp.prop_label} (Over: NO)"
        )
        slug = pp.player_name.lower().replace(" ", "_")

        live_fields = self._event_live_fields(ev, mkt)
        live_fields["line_source"] = getattr(pp, "line_source", live_fields["line_source"])

        out = {
            "pick_id": f"{slate_date}:{ev['sport']}:{ev['event_id']}:prop:{slug}:{pp.prop_type}",
            "slate_date": slate_date,
            "sport": ev["sport"],
            "event_id": ev["event_id"],
            "event_name": ev["event_name"],
            "short_name": ev["short_name"],
            "start_time": ev["start_time"],
            "market_type": f"{'Live ' if is_live else ''}Prop: {pp.prop_label}",
            "selection_label": label,
            "contract_question": f"Will {pp.player_name} record Over {pp.line_value:.1f} {pp.prop_label}?",
            "player_name": pp.player_name,
            "line_value": pp.line_value,
            "current_stat": getattr(pp, "current_stat", 0.0),
            "yes_true_prob": round(blended_over, 4),
            "recommended_side": best.side,
            "contract_price": best.contract_price,
            "fee_per_contract": best.fee_per_contract,
            "all_in_cost": best.all_in_cost,
            "breakeven_prob": best.breakeven_prob,
            "model_prob": best.model_prob,
            "raw_fundamental_prob": pp.over_prob if best.side == "YES" else pp.under_prob,
            "market_prob": (
                round(pp.market_over_prob if best.side == "YES" else (1.0 - pp.market_over_prob), 4)
                if pp.market_over_prob is not None
                else 0.50
            ),
            "conservative_prob": best.conservative_prob,
            "net_edge": best.net_edge,
            "ev_per_contract": best.ev_per_contract,
            "roi_pct": best.roi_pct,
            "fair_entry_max_price": best.fair_entry_max_price,
            "fair_midgame_exit_price": best.fair_midgame_exit_price,
            "recommended_contracts": best.recommended_contracts,
            "recommended_outlay": best.recommended_outlay,
            "american_equivalent": best.american_equivalent,
            "confidence_score": best.confidence_score,
            "confidence_tier": best.confidence_tier,
            "drivers": (list(safety_drivers) + list(pp.drivers))[:4],
            "injury_notes": inj_notes,
        }
        out.update(live_fields)
        return out

    def evaluate_open_positions(self, fresh: bool = False) -> list[dict[str, Any]]:
        """
        Mark OPEN Trade Ledger positions to live in-play / current model probabilities and emit
        actionable HOLD TO $1.00 vs TAKE PROFIT vs HEDGE guidance.
        """
        open_trades = [t for t in self.db.list_trades() if t.get("status") == "OPEN"]
        if not open_trades:
            return []

        all_candidates = self.generate_picks(
            sport="all",
            market_filter="all",
            mode="all",
            include_passes=True,
            fresh=fresh,
        )
        results: list[dict[str, Any]] = []
        exit_fee = self.settings.default_fee_dollars

        for trade in open_trades:
            sel_low = str(trade["selection_label"]).lower()
            ev_low = str(trade["event_name"]).lower()
            matched = next(
                (
                    c
                    for c in all_candidates
                    if c["sport"] == trade["sport"]
                    and (
                        c["selection_label"].lower() == sel_low
                        or (
                            c["short_name"].lower() in ev_low
                            and c["market_type"].replace("Live ", "").lower()
                            == str(trade["market_type"]).replace("Live ", "").lower()
                        )
                    )
                ),
                None,
            )
            if matched is None:
                results.append(
                    {
                        **trade,
                        "live_model_prob": trade["model_prob"],
                        "game_state": "UNKNOWN",
                        "score_display": "",
                        "breakeven_cashout_bid": round(min(0.99, trade["model_prob"] + exit_fee), 3),
                        "live_signal": "HOLD TO $1.00",
                    }
                )
                continue

            yes_p = float(matched["yes_true_prob"])
            live_side_p = yes_p if trade["side"] == "YES" else (1.0 - yes_p)
            be_bid = round(min(0.99, live_side_p + exit_fee), 3)
            drivers_txt = " ".join(matched.get("drivers", [])).lower()

            if "blowout" in drivers_txt and live_side_p >= 0.72 and trade["side"] == "YES":
                signal = f"CONSIDER CASH OUT IF BID >= {int(round((live_side_p - 0.03)*100))}¢ (Blowout Bench Risk)"
            elif live_side_p <= 0.35:
                signal = f"EDGE REVERSED ({live_side_p*100:.0f}% win) — Cut Only If Bid > {int(round(be_bid*100))}¢"
            else:
                signal = f"HOLD TO $1.00 (Sell Only If Live Bid >= {int(round(be_bid*100))}¢)"

            results.append(
                {
                    **trade,
                    "live_model_prob": round(live_side_p, 4),
                    "game_state": matched.get("game_state", "PRE"),
                    "status_detail": matched.get("status_detail", ""),
                    "score_display": matched.get("score_display", ""),
                    "breakeven_cashout_bid": be_bid,
                    "live_signal": signal,
                }
            )
        return results

    def _apply_same_game_caps(self, picks: list[dict[str, Any]], bankroll: float) -> None:
        max_game_dollars = max(
            bankroll * self.settings.max_same_game_pct,
            min(bankroll * 0.16, 3.00),
        )
        used_by_event: dict[str, float] = {}

        for p in picks:
            if p["confidence_tier"] == "PASS" or p["recommended_contracts"] <= 0:
                continue
            ev_id = str(p["event_id"])
            already = used_by_event.get(ev_id, 0.0)
            remaining = max(0.0, max_game_dollars - already)
            if p["recommended_outlay"] > remaining:
                cost_per = float(p["all_in_cost"])
                allowed_contracts = int(remaining // cost_per) if cost_per > 0 else 0
                p["recommended_contracts"] = allowed_contracts
                p["recommended_outlay"] = round(allowed_contracts * cost_per, 2)
                if allowed_contracts == 0:
                    p["confidence_tier"] = "PASS"
                    p["drivers"] = ["Capped by same-game 6% correlation limit (top plays prioritized)"] + p["drivers"][:3]
            used_by_event[ev_id] = already + float(p["recommended_outlay"])

    @staticmethod
    def _format_injury_summary(
        injuries: dict[str, list[dict[str, Any]]], home_abbr: str, away_abbr: str
    ) -> str:
        notes = []
        for t in (away_abbr, home_abbr):
            important = [
                f"{i['player_name']} ({i['status']})"
                for i in injuries.get(t, [])
                if any(k in str(i.get("status", "")).lower() for k in ("out", "doubtful", "questionable"))
            ][:3]
            if important:
                notes.append(f"{t}: {', '.join(important)}")
        return " | ".join(notes)

    def reevaluate_custom_price(
        self,
        *,
        yes_true_prob: float,
        side: str,
        contract_price: float,
        fee_per_contract: float,
        bankroll: float,
        min_win_prob: float | None = None,
        min_net_edge: float | None = None,
        is_live: bool = False,
    ) -> dict[str, Any]:
        side_u = "YES" if side.upper() == "YES" else "NO"
        p_side = yes_true_prob if side_u == "YES" else (1.0 - yes_true_prob)
        res = evaluate_single_side(
            side=side_u,
            side_true_prob=p_side,
            contract_price=contract_price,
            fee_per_contract=fee_per_contract,
            bankroll=bankroll,
            kelly_multiplier=self.settings.kelly_fraction * (0.80 if is_live else 1.0),
            max_bet_pct=self.settings.max_single_bet_pct,
            min_win_prob=min_win_prob if min_win_prob is not None else self.settings.min_win_prob,
            min_net_edge=min_net_edge if min_net_edge is not None else self.settings.min_net_edge,
            uncertainty=0.15 if is_live else 0.12,
        )
        return res.to_dict()

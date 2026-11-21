"""Ensemble blending, live in-play conditioning, dual-gate DKeX filtering, and slate orchestrator."""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from scipy.stats import norm, t as student_t

from src.config import PROJECT_ROOT, Settings, get_settings
from src.data.espn_client import ESPNClient
from src.data.nba_loader import NBALoader
from src.data.nfl_loader import NFLLoader
from src.data.odds_client import OddsClient, devig_american_pair
from src.engine.dk_math import (
    ContractEvaluation,
    clamp_prob,
    evaluate_single_side,
    evaluate_two_way_contract,
    inv_logit,
    logit,
    normalize_player_name,
    prob_to_american_odds,
)
from src.engine.sportsbook_math import (
    NFL_KEY_NUMBERS,
    american_to_decimal,
    compute_two_way_hold,
    decimal_to_american,
    evaluate_arbitrage_pair,
    evaluate_bonus_bet_conversion,
    evaluate_middle_opportunity,
    evaluate_sportsbook_side,
    evaluate_steam_lag,
    format_american,
)
from src.models.nba_game import NBAGameModel
from src.models.nfl_game import NFLGameModel
from src.models.player_props import PlayerPropsEngine, prop_over_prob_for_line
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
        DraftKings player props, and live Odds API lines (skipping slow historical CSV downloads).
        """
        sp = sport.lower()
        out: dict[str, Any] = {}
        for lg in (["NFL", "NBA"] if sp == "all" else [sp.upper()]):
            sb = self.espn.fetch_scoreboard(lg, target_date=target_date, force_refresh=True)
            live_evs = [e for e in sb if e.get("state") == "IN"]
            for lev in live_evs:
                self.espn.fetch_live_boxscore(lg, lev["event_id"], force_refresh=True)
            inj = self.espn.fetch_league_injuries(lg, force_refresh=True)
            espn_props = self.espn.fetch_slate_player_props(lg, sb, force_refresh=True)
            odds = self.odds.fetch_game_odds(
                lg, force_refresh=True, live_mode=bool(live_evs)
            )
            out[lg.lower()] = {
                "scoreboard_events": len(sb),
                "live_in_play_events": len(live_evs),
                "injured_teams_tracked": len(inj),
                "espn_prop_events": len(espn_props),
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
            nfl_props = self.espn.fetch_slate_player_props("NFL", nfl_sb, force_refresh=force)
            nfl_odds = self.odds.fetch_game_odds("NFL", force_refresh=force, live_mode=bool(nfl_live))
            summary["nfl"] = {
                "historical_games": len(nfl_sched),
                "player_log_rows": len(nfl_logs),
                "live_scoreboard_events": len(nfl_sb),
                "in_play_events": len(nfl_live),
                "injured_teams_tracked": len(nfl_inj),
                "espn_prop_events": len(nfl_props),
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
            nba_props = self.espn.fetch_slate_player_props("NBA", nba_sb, force_refresh=force)
            nba_odds = self.odds.fetch_game_odds("NBA", force_refresh=force, live_mode=bool(nba_live))
            summary["nba"] = {
                "historical_games": len(nba_games),
                "player_log_rows": len(nba_logs),
                "live_scoreboard_events": len(nba_sb),
                "in_play_events": len(nba_live),
                "injured_teams_tracked": len(nba_inj),
                "espn_prop_events": len(nba_props),
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
        if active or events:
            return active

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
        live_props: bool = True,
        event_id_filter: str | None = None,
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

        overrides = self.db.get_line_overrides()
        sports = ["NFL", "NBA"] if sport.lower() == "all" else [sport.upper()]
        candidates: list[dict[str, Any]] = []

        for sp in sports:
            events = self._get_upcoming_events(
                sp, target_date=target_date, mode=mode, force_refresh=fresh
            )
            if event_id_filter is not None:
                events = [
                    ev for ev in events if str(ev.get("event_id")) == str(event_id_filter)
                ]
            if not events:
                continue
            has_live = any(ev.get("state") == "IN" for ev in events)
            injuries = self.espn.fetch_league_injuries(sp, force_refresh=fresh)
            sharp_odds = self.odds.fetch_game_odds(
                sp, force_refresh=fresh, live_mode=has_live
            )
            espn_slate_props = (
                self.espn.fetch_slate_player_props(sp, events, force_refresh=fresh)
                if live_props and market_filter.lower() in ("all", "props")
                else {}
            )

            for ev in events:
                h = ev["home_abbr"]
                a = ev["away_abbr"]
                ev_id = str(ev["event_id"])
                is_live = ev.get("state") == "IN"
                mkt = dict(ev.get("market", {}) or {})

                live_box_players: dict[str, dict[str, Any]] | None = None
                if is_live:
                    summary_data = self.espn.fetch_live_boxscore(
                        sp, ev_id, force_refresh=fresh
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

                merged_prop_mkts: dict[tuple[str, str], dict[str, Any]] = dict(
                    espn_slate_props.get(ev_id, {})
                )
                if (
                    live_props
                    and market_filter.lower() in ("all", "props")
                    and mkt.get("odds_event_id")
                ):
                    odds_props = self.odds.fetch_event_player_props(
                        sp,
                        str(mkt["odds_event_id"]),
                        force_refresh=fresh,
                        live_mode=is_live,
                    )
                    if odds_props:
                        merged_prop_mkts.update(odds_props)
                live_prop_mkts = merged_prop_mkts if merged_prop_mkts else None

                # Extract persistent user line overrides for this event
                sp_ov_key = f"{sp}:{ev_id}:spread"
                tot_ov_key = f"{sp}:{ev_id}:total"
                sp_override = overrides.get(sp_ov_key)
                tot_override = overrides.get(tot_ov_key)
                prop_prefix = f"{sp}:{ev_id}:prop:"
                custom_prop_lines: dict[tuple[str, str], float] = {}
                for ok, ov_obj in overrides.items():
                    if ok.startswith(prop_prefix):
                        rest = ok[len(prop_prefix):]
                        if ":" in rest:
                            p_norm, p_col = rest.rsplit(":", 1)
                            custom_prop_lines[(p_norm, p_col)] = float(ov_obj["line_value"])

                default_home_spread = mkt.get("home_spread")
                default_total_line = mkt.get("total_line")
                eff_spread_line = (
                    float(sp_override["line_value"]) if sp_override else default_home_spread
                )
                eff_total_line = (
                    float(tot_override["line_value"]) if tot_override else default_total_line
                )

                inj_notes = self._format_injury_summary(injuries, h, a)

                if sp == "NFL":
                    gpred = self.nfl_model.predict_matchup(
                        home_team=h,
                        away_team=a,
                        spread_line=eff_spread_line,
                        total_line=eff_total_line,
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
                            custom_lines=custom_prop_lines or None,
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
                        spread_line=eff_spread_line,
                        total_line=eff_total_line,
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
                            custom_lines=custom_prop_lines or None,
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
                            default_spread_line=default_home_spread,
                            override=sp_override,
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
                            default_total_line=default_total_line,
                            override=tot_override,
                        )
                    )

                for pp in prop_preds:
                    pp_ov_key = f"{sp}:{ev_id}:prop:{normalize_player_name(pp.player_name)}:{pp.prop_type}"
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
                            override=overrides.get(pp_ov_key),
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

    def _evaluate_contract_with_override(
        self,
        *,
        yes_true_prob: float,
        yes_ask: float,
        no_ask: float,
        eff_fee: float,
        eff_bankroll: float,
        kelly_m: float,
        max_bet_pct: float,
        eff_min_prob: float,
        eff_min_edge: float,
        yes_anchor_prob: float | None,
        uncertainty: float,
        clarity: float,
        override: dict[str, Any] | None = None,
    ) -> ContractEvaluation:
        if override and override.get("contract_price") is not None:
            ov_px = float(override["contract_price"])
            ov_side_raw = str(override.get("side") or "").upper()
            if ov_side_raw in ("YES", "NO"):
                side_lit: Any = "YES" if ov_side_raw == "YES" else "NO"
                p_side = yes_true_prob if side_lit == "YES" else (1.0 - yes_true_prob)
                anch_side = (
                    (yes_anchor_prob if side_lit == "YES" else (1.0 - yes_anchor_prob))
                    if yes_anchor_prob is not None
                    else None
                )
                return evaluate_single_side(
                    side=side_lit,
                    side_true_prob=p_side,
                    contract_price=ov_px,
                    fee_per_contract=eff_fee,
                    bankroll=eff_bankroll,
                    kelly_multiplier=kelly_m,
                    max_bet_pct=max_bet_pct,
                    min_win_prob=eff_min_prob,
                    min_net_edge=eff_min_edge,
                    anchor_prob=anch_side,
                    uncertainty=uncertainty,
                    injury_clarity=clarity,
                )
        best, _, _ = evaluate_two_way_contract(
            yes_true_prob=yes_true_prob,
            yes_ask_price=yes_ask,
            no_ask_price=no_ask,
            fee_per_contract=eff_fee,
            bankroll=eff_bankroll,
            kelly_multiplier=kelly_m,
            max_bet_pct=max_bet_pct,
            min_win_prob=eff_min_prob,
            min_net_edge=eff_min_edge,
            yes_anchor_prob=yes_anchor_prob,
            uncertainty=uncertainty,
            injury_clarity=clarity,
        )
        return best

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
            "override_key": None,
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
            "default_line_value": None,
            "is_custom_line": False,
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
        default_spread_line: float | None = None,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        h, a = gpred.home_team, gpred.away_team
        sp_line = float(gpred.spread_line)
        default_sp = (
            float(default_spread_line)
            if default_spread_line is not None
            else round(-gpred.expected_margin * 2.0) / 2.0
        )
        is_custom = override is not None
        override_key = f"{ev['sport']}:{ev['event_id']}:spread"

        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, gpred.injury_clarity
        )
        mkt_cov = mkt.get("home_cover_prob")
        dk_cov = mkt.get("dk_home_cover_prob")
        has_real_book = str(mkt.get("line_source", "")) != "Offline Fallback" and mkt_cov is not None

        if is_custom and mkt_cov is not None and abs(sp_line - default_sp) >= 0.01:
            reg_mins = 60.0 if ev["sport"] == "NFL" else 48.0
            rem_f = (
                max(0.04, float(ev.get("remaining_minutes", reg_mins)) / reg_mins)
                if is_live
                else 1.0
            )
            sp_scale = (13.0 if ev["sport"] == "NFL" else 12.0) * math.sqrt(rem_f)
            mkt_exp_margin = -default_sp + sp_scale * float(
                norm.ppf(clamp_prob(float(mkt_cov), 0.02))
            )
            mkt_cov = round(
                clamp_prob(float(1.0 - norm.cdf(-sp_line, loc=mkt_exp_margin, scale=sp_scale))),
                4,
            )
            dk_cov = mkt_cov

        mkt_w = (0.68 if is_live else 0.62) if has_real_book else 0.30
        blended_cov = blend_probabilities(gpred.home_cover_prob, mkt_cov, market_weight=mkt_w)

        yes_ask, no_ask = _infer_dkex_ask_pair(
            dk_yes_prob=dk_cov,
            sharp_yes_prob=mkt_cov,
            blended_yes_prob=blended_cov,
            is_live=is_live,
        )

        best = self._evaluate_contract_with_override(
            yes_true_prob=blended_cov,
            yes_ask=yes_ask,
            no_ask=no_ask,
            eff_fee=eff_fee,
            eff_bankroll=eff_bankroll,
            kelly_m=kelly_m,
            max_bet_pct=self.settings.max_single_bet_pct,
            eff_min_prob=eff_min_prob,
            eff_min_edge=eff_min_edge,
            yes_anchor_prob=mkt_cov if mkt_cov is not None else 0.50,
            uncertainty=0.15 if is_live else 0.12,
            clarity=clarity,
            override=override,
        )
        if locked or (is_live and mkt_cov is None and dk_cov is None and not is_custom):
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

        live_fields = self._event_live_fields(ev, mkt)
        if is_custom:
            live_fields["line_source"] = "Custom Live Line" if is_live else "Custom Line"

        out = {
            "pick_id": f"{slate_date}:{ev['sport']}:{ev['event_id']}:spread",
            "override_key": override_key,
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
            "default_line_value": default_sp,
            "is_custom_line": is_custom,
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
        out.update(live_fields)
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
        default_total_line: float | None = None,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tot_line = float(gpred.total_line)
        default_tot = (
            float(default_total_line)
            if default_total_line is not None
            else round(gpred.expected_total * 2.0) / 2.0
        )
        is_custom = override is not None
        override_key = f"{ev['sport']}:{ev['event_id']}:total"

        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, gpred.injury_clarity
        )
        mkt_over = mkt.get("over_prob")
        dk_over = mkt.get("dk_over_prob")
        has_real_book = str(mkt.get("line_source", "")) != "Offline Fallback" and mkt_over is not None

        if is_custom and mkt_over is not None and abs(tot_line - default_tot) >= 0.01:
            reg_mins = 60.0 if ev["sport"] == "NFL" else 48.0
            rem_f = (
                max(0.04, float(ev.get("remaining_minutes", reg_mins)) / reg_mins)
                if is_live
                else 1.0
            )
            tot_scale = (11.5 if ev["sport"] == "NFL" else 14.5) * math.sqrt(rem_f)
            mkt_exp_total = default_tot + tot_scale * float(
                norm.ppf(clamp_prob(float(mkt_over), 0.02))
            )
            mkt_over = round(
                clamp_prob(float(1.0 - norm.cdf(tot_line, loc=mkt_exp_total, scale=tot_scale))),
                4,
            )
            dk_over = mkt_over

        mkt_w = (0.68 if is_live else 0.62) if has_real_book else 0.30
        blended_over = blend_probabilities(gpred.over_prob, mkt_over, market_weight=mkt_w)

        yes_ask, no_ask = _infer_dkex_ask_pair(
            dk_yes_prob=dk_over,
            sharp_yes_prob=mkt_over,
            blended_yes_prob=blended_over,
            is_live=is_live,
        )

        best = self._evaluate_contract_with_override(
            yes_true_prob=blended_over,
            yes_ask=yes_ask,
            no_ask=no_ask,
            eff_fee=eff_fee,
            eff_bankroll=eff_bankroll,
            kelly_m=kelly_m,
            max_bet_pct=self.settings.max_single_bet_pct,
            eff_min_prob=eff_min_prob,
            eff_min_edge=eff_min_edge,
            yes_anchor_prob=mkt_over if mkt_over is not None else 0.50,
            uncertainty=0.15 if is_live else 0.12,
            clarity=clarity,
            override=override,
        )
        if locked or (is_live and mkt_over is None and dk_over is None and not is_custom):
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

        live_fields = self._event_live_fields(ev, mkt)
        if is_custom:
            live_fields["line_source"] = "Custom Live Line" if is_live else "Custom Line"

        out = {
            "pick_id": f"{slate_date}:{ev['sport']}:{ev['event_id']}:total",
            "override_key": override_key,
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
            "default_line_value": default_tot,
            "is_custom_line": is_custom,
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
        out.update(live_fields)
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
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        is_live, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
            ev, mkt, pp.injury_clarity
        )
        prop_src = str(getattr(pp, "line_source", "Model Ref Line"))
        is_custom = bool(getattr(pp, "is_custom_line", False) or override is not None)
        default_ln = getattr(pp, "default_line_value", None)
        if default_ln is None:
            default_ln = float(pp.line_value)
        override_key = (
            f"{ev['sport']}:{ev['event_id']}:prop:"
            f"{normalize_player_name(pp.player_name)}:{pp.prop_type}"
        )

        has_real_prop_book = prop_src not in (
            "Model Ref Line",
            "Live Est Line",
            "Custom Line",
            "Custom Live Line",
        )
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

        best = self._evaluate_contract_with_override(
            yes_true_prob=blended_over,
            yes_ask=yes_ask,
            no_ask=no_ask,
            eff_fee=eff_fee,
            eff_bankroll=eff_bankroll,
            kelly_m=kelly_m,
            max_bet_pct=min(0.03, self.settings.max_single_bet_pct),
            eff_min_prob=eff_min_prob,
            eff_min_edge=eff_min_edge,
            yes_anchor_prob=pp.market_over_prob if pp.market_over_prob is not None else 0.50,
            uncertainty=prop_uncert,
            clarity=clarity,
            override=override,
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
            "override_key": override_key,
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
            "default_line_value": float(default_ln),
            "is_custom_line": is_custom,
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
            bet_fmt = str(trade.get("bet_format") or "CONTRACT").upper()
            side_u = str(trade["side"]).upper()

            if bet_fmt == "SPORTSBOOK":
                is_yes_like = side_u in ("YES", "HOME", "OVER")
                live_side_p = yes_p if is_yes_like else (1.0 - yes_p)
                am_entry = int(trade.get("odds_american") or -110)
                dec_entry = american_to_decimal(am_entry)
                # Opposing odds required to lock risk-free hedge: 1/dec_entry + 1/dec_opp < 1
                inv_rem = max(0.02, 1.0 - (1.0 / dec_entry))
                be_hedge_am = decimal_to_american(1.0 / inv_rem)
                if live_side_p >= 0.72:
                    signal = f"IN PROFIT ({live_side_p*100:.0f}% win) — Hedge opp at {format_american(be_hedge_am)}+ to lock profit"
                elif live_side_p <= 0.35:
                    signal = f"EDGE REVERSED ({live_side_p*100:.0f}% win) — Hold unless cashout > EV"
                else:
                    signal = f"HOLD ({live_side_p*100:.0f}% live win · Hedge opp @ {format_american(be_hedge_am)}+)"
                be_bid = round(live_side_p, 3)
            else:
                live_side_p = yes_p if side_u == "YES" else (1.0 - yes_p)
                be_bid = round(min(0.99, live_side_p + exit_fee), 3)
                drivers_txt = " ".join(matched.get("drivers", [])).lower()

                if "blowout" in drivers_txt and live_side_p >= 0.72 and side_u == "YES":
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

    def apply_line_override(
        self,
        *,
        override_key: str,
        sport: str,
        event_id: str,
        pick_id: str = "",
        target_date: str | None = None,
        line_value: float | None = None,
        contract_price: float | None = None,
        side: str | None = None,
        reset: bool = False,
        fee_per_contract: float | None = None,
        bankroll: float | None = None,
        min_win_prob: float | None = None,
        min_net_edge: float | None = None,
        live_props: bool = True,
    ) -> dict[str, Any]:
        """
        Persist or clear a custom line override in SQLite and immediately recompute
        the updated market card using cached slate feeds.
        """
        if reset:
            self.db.delete_line_override(override_key)
        else:
            if line_value is None:
                raise ValueError("line_value is required unless reset=True.")
            self.db.set_line_override(
                override_key=override_key,
                line_value=float(line_value),
                contract_price=contract_price,
                side=side,
            )

        candidates = self.generate_picks(
            sport=sport,
            market_filter="all",
            mode="all",
            target_date=target_date,
            min_win_prob=min_win_prob,
            min_net_edge=min_net_edge,
            fee_per_contract=fee_per_contract,
            bankroll=bankroll,
            include_passes=True,
            fresh=False,
            live_props=live_props,
            event_id_filter=str(event_id),
        )
        matched = next(
            (
                c
                for c in candidates
                if c.get("override_key") == override_key
                or (pick_id and c.get("pick_id") == pick_id)
            ),
            None,
        )
        if matched is None:
            raise KeyError(f"Market not found for override_key={override_key}")
        return matched

    # =========================================================================
    # Traditional Sportsbooks (DraftKings & FanDuel) Slate & Strategy Engine
    # =========================================================================

    def _sb_spread_cover_prob(self, sport: str, gpred: Any, home_spread: float, is_live: bool, rem_mins: float) -> float:
        if sport == "NFL" and self.nfl_model is not None:
            scale_ov = (
                max(2.4, self.nfl_model.residual_std * math.sqrt(max(0.03, rem_mins / 60.0)))
                if is_live
                else None
            )
            return float(
                self.nfl_model._spread_cover_probability(
                    gpred.expected_margin, home_spread, scale_override=scale_ov
                )
            )
        scale = (self.nba_model.margin_scale * 0.92) if self.nba_model is not None else 11.8
        if is_live:
            scale = max(2.3, scale * math.sqrt(max(0.025, rem_mins / 48.0)))
        return clamp_prob(float(1.0 - student_t.cdf(-home_spread, df=12, loc=gpred.expected_margin, scale=scale)))

    def _sb_total_over_prob(self, sport: str, gpred: Any, total_line: float, is_live: bool, rem_mins: float) -> float:
        if sport == "NFL":
            base_s = self.nfl_model.total_std if self.nfl_model is not None else 9.8
            scale = max(2.8, base_s * math.sqrt(max(0.03, rem_mins / 60.0))) if is_live else base_s
            return clamp_prob(float(1.0 - norm.cdf(total_line, loc=gpred.expected_total, scale=scale)))
        base_s = (self.nba_model.total_scale * 0.93) if self.nba_model is not None else 14.1
        scale = max(2.8, base_s * math.sqrt(max(0.025, rem_mins / 48.0))) if is_live else base_s
        return clamp_prob(float(1.0 - student_t.cdf(total_line, df=14, loc=gpred.expected_total, scale=scale)))

    def _sb_margin_interval_probs(
        self, sport: str, gpred: Any, lo_margin: float, hi_margin: float, is_live: bool, rem_mins: float
    ) -> tuple[float, float, float]:
        # Empirical NFL key-number frequency anchors (3: ~14.5%, 7: ~9.2%, 6: ~5.8%, 10: ~5.5%, 4: ~4.9%, 14: ~4.3%)
        nfl_key_base = {3: 0.145, 7: 0.092, 6: 0.058, 10: 0.055, 4: 0.049, 14: 0.043, 1: 0.042, 2: 0.036}

        def _pmf(k: int) -> float:
            if sport == "NFL":
                base_s = self.nfl_model.residual_std if self.nfl_model is not None else 13.1
                scale = max(2.4, base_s * math.sqrt(max(0.03, rem_mins / 60.0))) if is_live else base_s
                g_k = float(
                    norm.cdf(k + 0.5, loc=gpred.expected_margin, scale=scale)
                    - norm.cdf(k - 0.5, loc=gpred.expected_margin, scale=scale)
                )
                abs_k = abs(k)
                if not is_live and abs_k in nfl_key_base:
                    g_pos = float(
                        norm.cdf(abs_k + 0.5, loc=gpred.expected_margin, scale=scale)
                        - norm.cdf(abs_k - 0.5, loc=gpred.expected_margin, scale=scale)
                    )
                    g_neg = float(
                        norm.cdf(-abs_k + 0.5, loc=gpred.expected_margin, scale=scale)
                        - norm.cdf(-abs_k - 0.5, loc=gpred.expected_margin, scale=scale)
                    )
                    share = (g_pos if k > 0 else g_neg) / max(1e-6, g_pos + g_neg)
                    return max(0.0, 0.70 * (nfl_key_base[abs_k] * share) + 0.30 * g_k)
                return max(0.0, g_k)
            base_s = (self.nba_model.margin_scale * 0.92) if self.nba_model is not None else 11.8
            scale = max(2.3, base_s * math.sqrt(max(0.025, rem_mins / 48.0))) if is_live else base_s
            return max(
                0.0,
                float(
                    student_t.cdf(k + 0.5, df=12, loc=gpred.expected_margin, scale=scale)
                    - student_t.cdf(k - 0.5, df=12, loc=gpred.expected_margin, scale=scale)
                ),
            )

        p_lo = _pmf(int(round(lo_margin))) if abs(lo_margin - round(lo_margin)) < 1e-6 else 0.0
        p_hi = _pmf(int(round(hi_margin))) if abs(hi_margin - round(hi_margin)) < 1e-6 else 0.0
        k_s = int(math.floor(lo_margin)) + 1
        k_e = int(math.ceil(hi_margin)) - 1
        p_in = sum(_pmf(k) for k in range(k_s, k_e + 1)) if k_e >= k_s else 0.0
        return max(0.0, min(0.60, p_in)), max(0.0, min(0.35, p_lo)), max(0.0, min(0.35, p_hi))

    def _sb_total_interval_probs(
        self, sport: str, gpred: Any, lo_line: float, hi_line: float, is_live: bool, rem_mins: float
    ) -> tuple[float, float, float]:
        def _pmf(k: int) -> float:
            if sport == "NFL":
                base_s = self.nfl_model.total_std if self.nfl_model is not None else 9.8
                scale = max(2.8, base_s * math.sqrt(max(0.03, rem_mins / 60.0))) if is_live else base_s
                return max(
                    0.0,
                    float(
                        norm.cdf(k + 0.5, loc=gpred.expected_total, scale=scale)
                        - norm.cdf(k - 0.5, loc=gpred.expected_total, scale=scale)
                    ),
                )
            base_s = (self.nba_model.total_scale * 0.93) if self.nba_model is not None else 14.1
            scale = max(2.8, base_s * math.sqrt(max(0.025, rem_mins / 48.0))) if is_live else base_s
            return max(
                0.0,
                float(
                    student_t.cdf(k + 0.5, df=14, loc=gpred.expected_total, scale=scale)
                    - student_t.cdf(k - 0.5, df=14, loc=gpred.expected_total, scale=scale)
                ),
            )

        p_lo = _pmf(int(round(lo_line))) if abs(lo_line - round(lo_line)) < 1e-6 else 0.0
        p_hi = _pmf(int(round(hi_line))) if abs(hi_line - round(hi_line)) < 1e-6 else 0.0
        k_s = int(math.floor(lo_line)) + 1
        k_e = int(math.ceil(hi_line)) - 1
        p_in = sum(_pmf(k) for k in range(k_s, k_e + 1)) if k_e >= k_s else 0.0
        return max(0.0, min(0.60, p_in)), max(0.0, min(0.35, p_lo)), max(0.0, min(0.35, p_hi))

    @staticmethod
    def _sb_prop_interval_probs(pp: Any, lo_line: float, hi_line: float) -> tuple[float, float, float]:
        def _pmf(k: int) -> float:
            p_gt_prev = prop_over_prob_for_line(pp.sport, pp.prop_type, pp.projected_mean, pp.projected_std, k - 0.5)
            p_gt_curr = prop_over_prob_for_line(pp.sport, pp.prop_type, pp.projected_mean, pp.projected_std, k + 0.5)
            return max(0.0, p_gt_prev - p_gt_curr)

        p_lo = _pmf(int(round(lo_line))) if abs(lo_line - round(lo_line)) < 1e-6 else 0.0
        p_hi = _pmf(int(round(hi_line))) if abs(hi_line - round(hi_line)) < 1e-6 else 0.0
        k_s = int(math.floor(lo_line)) + 1
        k_e = int(math.ceil(hi_line)) - 1
        p_in = sum(_pmf(k) for k in range(k_s, k_e + 1)) if k_e >= k_s else 0.0
        return max(0.0, min(0.60, p_in)), max(0.0, min(0.35, p_lo)), max(0.0, min(0.35, p_hi))

    def generate_sportsbook_slate(
        self,
        *,
        sport: str = "all",
        market_filter: str = "all",
        mode: str = "all",
        book_filter: str = "all",
        target_date: str | None = None,
        min_ev: float | None = None,
        min_prob: float | None = None,
        bankroll: float | None = None,
        arb_stake: float = 100.0,
        arb_rounding: float | None = None,
        min_arb_roi: float | None = None,
        bonus_bet_size: float = 50.0,
        boost_pct: float = 0.25,
        include_passes: bool = False,
        include_dkex_cross_arb: bool | None = None,
        fresh: bool = False,
        live_props: bool = True,
    ) -> dict[str, Any]:
        """
        Generate the complete Traditional DraftKings & FanDuel sportsbook slate covering:
        1. +EV Line Shopping (`picks`)
        2. Cross-Book Arbitrage (`arbitrage` — DK<->FD and optional DKeX<->DK/FD)
        3. Middles & Polish Middles (`middles`)
        4. Low-Hold, Bonus Bet Conversion, & Profit Boost targets (`promos`)
        5. Steam / Off-Market Lag Alerts (`steam`)
        """
        self.ensure_loaded()
        assert self.nfl_model is not None
        assert self.nba_model is not None
        assert self.props_engine is not None

        eff_min_ev = min_ev if min_ev is not None else self.settings.min_sb_ev_edge
        eff_min_prob = min_prob if min_prob is not None else 0.40
        eff_bankroll = bankroll if bankroll is not None else self.settings.bankroll
        eff_rounding = arb_rounding if arb_rounding is not None else self.settings.arb_stake_rounding
        eff_min_arb = min_arb_roi if min_arb_roi is not None else self.settings.min_arb_roi
        eff_dkex_cross = (
            include_dkex_cross_arb
            if include_dkex_cross_arb is not None
            else self.settings.include_dkex_cross_arb
        )
        eff_fee = self.settings.default_fee_dollars
        slate_date = target_date or datetime.now().strftime("%Y-%m-%d")

        sports = ["NFL", "NBA"] if sport.lower() == "all" else [sport.upper()]
        book_f = book_filter.lower()

        picks_all: list[dict[str, Any]] = []
        arbitrage_list: list[dict[str, Any]] = []
        middles_list: list[dict[str, Any]] = []
        low_hold_list: list[dict[str, Any]] = []
        bonus_bet_list: list[dict[str, Any]] = []
        profit_boost_list: list[dict[str, Any]] = []
        steam_list: list[dict[str, Any]] = []

        games_seen: set[str] = set()
        has_any_fanduel = False
        prop_diag: dict[str, Any] = {
            "events_queried": 0,
            "odds_api_events_with_dk": 0,
            "odds_api_events_with_fd": 0,
            "last_http_status": None,
            "last_error": None,
        }

        for sp in sports:
            events = self._get_upcoming_events(
                sp, target_date=target_date, mode=mode, force_refresh=fresh
            )
            if not events:
                continue
            has_live = any(ev.get("state") == "IN" for ev in events)
            injuries = self.espn.fetch_league_injuries(sp, force_refresh=fresh)
            sharp_odds = self.odds.fetch_game_odds(sp, force_refresh=fresh, live_mode=has_live)
            espn_slate_props = (
                self.espn.fetch_slate_player_props(sp, events, force_refresh=fresh)
                if live_props and market_filter.lower() in ("all", "props")
                else {}
            )

            for ev in events:
                h = ev["home_abbr"]
                a = ev["away_abbr"]
                ev_id = str(ev["event_id"])
                games_seen.add(ev_id)
                is_live = ev.get("state") == "IN"
                rem_mins = float(ev.get("remaining_minutes", 60.0 if sp == "NFL" else 48.0))
                mkt = dict(ev.get("market", {}) or {})

                live_box_players: dict[str, dict[str, Any]] | None = None
                if is_live:
                    summary_data = self.espn.fetch_live_boxscore(sp, ev_id, force_refresh=fresh)
                    live_box_players = summary_data.get("players") or {}
                    if ev.get("live_win_prob") is None and summary_data.get("live_win_prob") is not None:
                        ev["live_win_prob"] = summary_data["live_win_prob"]

                consensus = (
                    self.odds.extract_consensus(sharp_odds, ev["home_name"], ev["away_name"])
                    if sharp_odds
                    else None
                )
                if consensus:
                    cons_age = consensus.get("odds_age_seconds")
                    if not (is_live and cons_age is not None and float(cons_age) > 300.0):
                        for keep_key in ("open_home_spread", "open_total_line", "open_home_win_prob"):
                            if keep_key in mkt and keep_key not in consensus:
                                consensus[keep_key] = mkt[keep_key]
                        mkt.update(consensus)

                if mkt.get("has_fanduel"):
                    has_any_fanduel = True

                merged_prop_mkts: dict[tuple[str, str], dict[str, Any]] = {}
                for (p_raw, col_k), pv in (espn_slate_props.get(ev_id, {}) or {}).items():
                    norm_k = (normalize_player_name(p_raw) or p_raw.lower(), col_k)
                    pv_copy = dict(pv)
                    pv_copy["books"] = dict(pv.get("books") or {})
                    merged_prop_mkts[norm_k] = pv_copy

                # Only fetch paid per-event Odds API props for live games or games starting within 48h/72h
                within_prop_window = is_live
                if not within_prop_window:
                    st_iso = str(ev.get("start_time") or ev.get("commence_time") or "")
                    if st_iso:
                        try:
                            from datetime import timezone as _tz

                            st_dt = datetime.fromisoformat(st_iso.replace("Z", "+00:00"))
                            if st_dt.tzinfo is None:
                                st_dt = st_dt.replace(tzinfo=_tz.utc)
                            hrs_away = (st_dt - datetime.now(_tz.utc)).total_seconds() / 3600.0
                            max_hrs = 72.0 if sp == "NFL" else 48.0
                            within_prop_window = -6.0 <= hrs_away <= max_hrs
                        except ValueError:
                            within_prop_window = True
                    else:
                        within_prop_window = True

                if (
                    live_props
                    and within_prop_window
                    and market_filter.lower() in ("all", "props")
                    and mkt.get("odds_event_id")
                ):
                    prop_diag["events_queried"] += 1
                    odds_props = self.odds.fetch_event_player_props(
                        sp,
                        str(mkt["odds_event_id"]),
                        force_refresh=fresh,
                        live_mode=is_live,
                    )
                    last_d = getattr(self.odds, "_last_prop_diag", None)
                    if isinstance(last_d, dict):
                        if last_d.get("error"):
                            prop_diag["last_error"] = last_d["error"]
                        if last_d.get("http_status"):
                            prop_diag["last_http_status"] = last_d["http_status"]
                    if odds_props:
                        ev_had_fd = False
                        ev_had_dk = False
                        for (p_raw, col_k), ov in odds_props.items():
                            norm_k = (normalize_player_name(p_raw) or p_raw.lower(), col_k)
                            ov_books = dict(ov.get("books") or {})
                            if "fanduel" in ov_books:
                                ev_had_fd = True
                            if "draftkings" in ov_books:
                                ev_had_dk = True
                            prev = merged_prop_mkts.get(norm_k)
                            if prev is None:
                                ov_copy = dict(ov)
                                ov_copy["books"] = ov_books
                                merged_prop_mkts[norm_k] = ov_copy
                            else:
                                # Preserve both ESPN DraftKings AND Odds API DraftKings/FanDuel books
                                comb_books = dict(prev.get("books") or {})
                                comb_books.update(ov_books)
                                comb = {**prev, **ov, "books": comb_books}
                                if "draftkings" in comb_books and "fanduel" in comb_books:
                                    comb["source"] = "DK + FD"
                                elif "draftkings" in comb_books:
                                    comb["source"] = "DraftKings"
                                elif "fanduel" in comb_books:
                                    comb["source"] = "FanDuel"
                                merged_prop_mkts[norm_k] = comb
                        if ev_had_dk:
                            prop_diag["odds_api_events_with_dk"] += 1
                        if ev_had_fd:
                            prop_diag["odds_api_events_with_fd"] += 1
                live_prop_mkts = merged_prop_mkts if merged_prop_mkts else None

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

                _, locked, kelly_m, clarity, safety_drivers = self._live_safety_context(
                    ev, mkt, gpred.injury_clarity
                )
                live_meta = self._event_live_fields(ev, mkt)
                base_meta = {
                    "slate_date": slate_date,
                    "sport": sp,
                    "event_id": ev_id,
                    "event_name": ev["event_name"],
                    "short_name": ev["short_name"],
                    "start_time": ev["start_time"],
                    "injury_notes": inj_notes,
                    **live_meta,
                }

                # Helper to register Arb + Low-Hold + Bonus Bet for any opposing DK vs FD pair
                def _scan_cross_book_pair(
                    *,
                    mkt_label: str,
                    dk_side_a_label: str,
                    dk_side_a_code: str,
                    dk_odds_a: int | None,
                    dk_side_b_label: str,
                    dk_side_b_code: str,
                    dk_odds_b: int | None,
                    fd_side_a_label: str,
                    fd_side_a_code: str,
                    fd_odds_a: int | None,
                    fd_side_b_label: str,
                    fd_side_b_code: str,
                    fd_odds_b: int | None,
                    allow_dk_a_vs_fd_b: bool = True,
                    allow_fd_a_vs_dk_b: bool = True,
                    is_freeroll_a: bool = False,
                    is_freeroll_b: bool = False,
                    dkex_yes_prob: float | None = None,
                    sharp_yes_prob: float | None = None,
                    blended_yes_prob: float | None = None,
                ) -> None:
                    # 1. DK Side A vs FD Side B
                    if allow_dk_a_vs_fd_b and dk_odds_a is not None and fd_odds_b is not None:
                        arb = evaluate_arbitrage_pair(
                            book_a="DRAFTKINGS",
                            label_a=dk_side_a_label,
                            side_a=dk_side_a_code,
                            odds_a=dk_odds_a,
                            book_b="FANDUEL",
                            label_b=fd_side_b_label,
                            side_b=fd_side_b_code,
                            odds_b=fd_odds_b,
                            total_stake=arb_stake,
                            rounding_step=eff_rounding,
                            min_arb_roi=eff_min_arb,
                            is_cross_number_freeroll=is_freeroll_a,
                        )
                        if arb is not None:
                            arbitrage_list.append({**base_meta, "market_type": mkt_label, **arb.to_dict()})

                        hold = compute_two_way_hold(dk_odds_a, fd_odds_b)
                        if 0.0 <= hold <= 0.025:
                            low_hold_list.append(
                                {
                                    **base_meta,
                                    "market_type": mkt_label,
                                    "hold_pct": hold,
                                    "leg_a": {"book": "DRAFTKINGS", "label": dk_side_a_label, "side": dk_side_a_code, "odds": dk_odds_a},
                                    "leg_b": {"book": "FANDUEL", "label": fd_side_b_label, "side": fd_side_b_code, "odds": fd_odds_b},
                                }
                            )
                        conv = evaluate_bonus_bet_conversion(
                            book_a="DRAFTKINGS",
                            label_a=dk_side_a_label,
                            side_a=dk_side_a_code,
                            odds_a=dk_odds_a,
                            book_b="FANDUEL",
                            label_b=fd_side_b_label,
                            side_b=fd_side_b_code,
                            odds_b=fd_odds_b,
                            bonus_amount=bonus_bet_size,
                        )
                        if conv.retention_pct >= 0.62:
                            bonus_bet_list.append({**base_meta, "market_type": mkt_label, **conv.to_dict()})

                    # 2. FD Side A vs DK Side B
                    if allow_fd_a_vs_dk_b and fd_odds_a is not None and dk_odds_b is not None:
                        arb2 = evaluate_arbitrage_pair(
                            book_a="FANDUEL",
                            label_a=fd_side_a_label,
                            side_a=fd_side_a_code,
                            odds_a=fd_odds_a,
                            book_b="DRAFTKINGS",
                            label_b=dk_side_b_label,
                            side_b=dk_side_b_code,
                            odds_b=dk_odds_b,
                            total_stake=arb_stake,
                            rounding_step=eff_rounding,
                            min_arb_roi=eff_min_arb,
                            is_cross_number_freeroll=is_freeroll_b,
                        )
                        if arb2 is not None:
                            arbitrage_list.append({**base_meta, "market_type": mkt_label, **arb2.to_dict()})

                        hold2 = compute_two_way_hold(fd_odds_a, dk_odds_b)
                        if 0.0 <= hold2 <= 0.025:
                            low_hold_list.append(
                                {
                                    **base_meta,
                                    "market_type": mkt_label,
                                    "hold_pct": hold2,
                                    "leg_a": {"book": "FANDUEL", "label": fd_side_a_label, "side": fd_side_a_code, "odds": fd_odds_a},
                                    "leg_b": {"book": "DRAFTKINGS", "label": dk_side_b_label, "side": dk_side_b_code, "odds": dk_odds_b},
                                }
                            )
                        conv2 = evaluate_bonus_bet_conversion(
                            book_a="FANDUEL",
                            label_a=fd_side_a_label,
                            side_a=fd_side_a_code,
                            odds_a=fd_odds_a,
                            book_b="DRAFTKINGS",
                            label_b=dk_side_b_label,
                            side_b=dk_side_b_code,
                            odds_b=dk_odds_b,
                            bonus_amount=bonus_bet_size,
                        )
                        if conv2.retention_pct >= 0.62:
                            bonus_bet_list.append({**base_meta, "market_type": mkt_label, **conv2.to_dict()})

                    # 3. Optional DKeX <-> DK / FD fee-adjusted cross-arbs
                    if eff_dkex_cross and blended_yes_prob is not None and not is_live:
                        dkex_yes, dkex_no = _infer_dkex_ask_pair(
                            dk_yes_prob=dkex_yes_prob,
                            sharp_yes_prob=sharp_yes_prob,
                            blended_yes_prob=blended_yes_prob,
                            is_live=is_live,
                        )
                        cross_checks = [
                            ("DKEX", dk_side_a_label, dk_side_a_code, None, dkex_yes, "FANDUEL", fd_side_b_label, fd_side_b_code, fd_odds_b, None),
                            ("FANDUEL", fd_side_a_label, fd_side_a_code, fd_odds_a, None, "DKEX", dk_side_b_label, dk_side_b_code, None, dkex_no),
                            ("DKEX", dk_side_a_label, dk_side_a_code, None, dkex_yes, "DRAFTKINGS", dk_side_b_label, dk_side_b_code, dk_odds_b, None),
                            ("DRAFTKINGS", dk_side_a_label, dk_side_a_code, dk_odds_a, None, "DKEX", dk_side_b_label, dk_side_b_code, None, dkex_no),
                        ]
                        for ba, la, sa, oa, dka, bb, lb, sb, ob, dkb in cross_checks:
                            if (oa is None and dka is None) or (ob is None and dkb is None):
                                continue
                            carb = evaluate_arbitrage_pair(
                                book_a=ba,  # type: ignore[arg-type]
                                label_a=la,
                                side_a=sa,
                                odds_a=oa,
                                dkex_ask_a=dka,
                                dkex_fee_a=eff_fee,
                                book_b=bb,  # type: ignore[arg-type]
                                label_b=lb,
                                side_b=sb,
                                odds_b=ob,
                                dkex_ask_b=dkb,
                                dkex_fee_b=eff_fee,
                                total_stake=arb_stake,
                                rounding_step=eff_rounding,
                                min_arb_roi=eff_min_arb,
                            )
                            if carb is not None:
                                arbitrage_list.append({**base_meta, "market_type": mkt_label, **carb.to_dict()})

                # -------------------------------------------------------------
                # 1. MONEYLINE (H2H)
                # -------------------------------------------------------------
                if market_filter.lower() in ("all", "game", "moneyline", "winner"):
                    sharp_hp = mkt.get("sharp_only_home_win_prob", mkt.get("sharp_home_win_prob", mkt.get("home_win_prob")))
                    has_real_ml = str(mkt.get("line_source", "")) != "Offline Fallback" and sharp_hp is not None
                    mkt_w = (0.68 if int(mkt.get("books_count", 0) or 0) >= 3 else 0.58) if is_live else (0.60 if has_real_ml else 0.35)
                    blended_hp = blend_probabilities(gpred.home_win_prob, sharp_hp, market_weight=mkt_w)

                    dk_h_ml = mkt.get("dk_home_ml")
                    dk_a_ml = mkt.get("dk_away_ml")
                    fd_h_ml = mkt.get("fd_home_ml")
                    fd_a_ml = mkt.get("fd_away_ml")

                    # Fallback reference DK odds from implied prob when offline fallback is active
                    if dk_h_ml is None and dk_a_ml is None and sharp_hp is not None and str(mkt.get("line_source", "")) in ("Offline Fallback", "Schedule Archive"):
                        dk_h_ml = prob_to_american_odds(clamp_prob(float(sharp_hp) * 1.022, 0.05))
                        dk_a_ml = prob_to_american_odds(clamp_prob((1.0 - float(sharp_hp)) * 1.022, 0.05))

                    ml_evals: list[dict[str, Any]] = []
                    for bk, h_od, a_od in (
                        ("DRAFTKINGS", dk_h_ml, dk_a_ml),
                        ("FANDUEL", fd_h_ml, fd_a_ml),
                    ):
                        if h_od is None or a_od is None:
                            continue
                        if book_f != "all" and bk.lower() != book_f:
                            continue
                        for sd, od, p_side, sh_side, lbl in (
                            ("HOME", int(h_od), blended_hp, sharp_hp, f"{h} Moneyline"),
                            ("AWAY", int(a_od), 1.0 - blended_hp, (1.0 - sharp_hp) if sharp_hp is not None else None, f"{a} Moneyline"),
                        ):
                            ev_res = evaluate_sportsbook_side(
                                book=bk,  # type: ignore[arg-type]
                                side=sd,
                                side_true_prob=p_side,
                                american_odds=od,
                                bankroll=eff_bankroll,
                                kelly_multiplier=kelly_m,
                                max_bet_pct=self.settings.max_single_bet_pct,
                                min_ev_edge=eff_min_ev,
                                min_win_prob=eff_min_prob,
                                anchor_prob=sh_side,
                                uncertainty=0.13 if is_live else 0.10,
                                injury_clarity=clarity,
                            )
                            ml_evals.append({"eval": ev_res, "label": lbl, "sharp_side": sh_side})

                            # Profit Boost check
                            if boost_pct > 0 and not locked:
                                b_ev = evaluate_sportsbook_side(
                                    book=bk,  # type: ignore[arg-type]
                                    side=sd,
                                    side_true_prob=p_side,
                                    american_odds=od,
                                    bankroll=eff_bankroll,
                                    kelly_multiplier=kelly_m,
                                    max_bet_pct=self.settings.max_single_bet_pct,
                                    min_ev_edge=0.04,
                                    min_win_prob=0.25,
                                    anchor_prob=sh_side,
                                    profit_boost_pct=boost_pct,
                                )
                                if b_ev.ev_pct >= 0.06:
                                    profit_boost_list.append(
                                        {
                                            **base_meta,
                                            "market_type": "Live Moneyline" if is_live else "Moneyline",
                                            "selection_label": lbl,
                                            "base_ev_pct": ev_res.ev_pct,
                                            **b_ev.to_dict(),
                                        }
                                    )

                    if ml_evals:
                        best_item = max(
                            ml_evals,
                            key=lambda x: (
                                x["eval"].confidence_tier != "PASS",
                                x["eval"].ev_pct,
                                x["eval"].confidence_score,
                            ),
                        )
                        best_ev = best_item["eval"]
                        if locked:
                            best_ev = evaluate_sportsbook_side(
                                book=best_ev.book,
                                side=best_ev.side,
                                side_true_prob=best_ev.model_prob,
                                american_odds=best_ev.american_odds,
                                bankroll=0.0,
                                min_ev_edge=99.0,
                            )

                        # Compute line shopping gain vs opposing book on the SAME side
                        other_same = next(
                            (
                                x["eval"]
                                for x in ml_evals
                                if x["eval"].side == best_ev.side and x["eval"].book != best_ev.book
                            ),
                            None,
                        )
                        shop_gain = (
                            round(best_ev.ev_pct - other_same.ev_pct, 4)
                            if other_same is not None
                            else None
                        )
                        drivers = list(safety_drivers) + list(gpred.drivers)
                        if shop_gain is not None and shop_gain >= 0.005:
                            other_nm = "FD" if other_same.book == "FANDUEL" else "DK"
                            best_nm = "DK" if best_ev.book == "DRAFTKINGS" else "FD"
                            drivers.insert(
                                0,
                                f"Line shop: {best_nm} {format_american(best_ev.american_odds)} beats {other_nm} {format_american(other_same.american_odds)} (+{shop_gain*100:.1f}% EV)",
                            )

                        picks_all.append(
                            {
                                **base_meta,
                                "pick_id": f"{slate_date}:{sp}:{ev_id}:sb:ml",
                                "market_type": "Live Moneyline" if is_live else "Moneyline",
                                "selection_label": best_item["label"],
                                "line_value": None,
                                "dk_line": None,
                                "fd_line": None,
                                "dk_odds": (dk_h_ml if best_ev.side == "HOME" else dk_a_ml),
                                "fd_odds": (fd_h_ml if best_ev.side == "HOME" else fd_a_ml),
                                "dk_pair": [dk_h_ml, dk_a_ml] if dk_h_ml is not None else None,
                                "fd_pair": [fd_h_ml, fd_a_ml] if fd_h_ml is not None else None,
                                "line_shopping_gain_pct": shop_gain,
                                "yes_true_prob": round(blended_hp, 4),
                                "drivers": drivers[:4],
                                **best_ev.to_dict(),
                            }
                        )

                    # Cross-book Arb, Low-Hold, Bonus Bet
                    _scan_cross_book_pair(
                        mkt_label="Live Moneyline" if is_live else "Moneyline",
                        dk_side_a_label=f"{h} ML",
                        dk_side_a_code="HOME",
                        dk_odds_a=dk_h_ml,
                        dk_side_b_label=f"{a} ML",
                        dk_side_b_code="AWAY",
                        dk_odds_b=dk_a_ml,
                        fd_side_a_label=f"{h} ML",
                        fd_side_a_code="HOME",
                        fd_odds_a=fd_h_ml,
                        fd_side_b_label=f"{a} ML",
                        fd_side_b_code="AWAY",
                        fd_odds_b=fd_a_ml,
                        dkex_yes_prob=mkt.get("dk_home_win_prob"),
                        sharp_yes_prob=sharp_hp,
                        blended_yes_prob=blended_hp,
                    )

                    # Steam / Lag detection against sharp-only consensus
                    sharp_only_hp = mkt.get("sharp_only_home_win_prob")
                    if sharp_only_hp is not None:
                        sharp_src = f"{int(mkt.get('sharp_books_count', 1))}-Book Sharp"
                        for bk, h_od, a_od, bk_hp in (
                            ("DRAFTKINGS", dk_h_ml, dk_a_ml, mkt.get("dk_home_win_prob")),
                            ("FANDUEL", fd_h_ml, fd_a_ml, mkt.get("fd_home_win_prob")),
                        ):
                            if h_od is None or a_od is None:
                                continue
                            for sd, od, b_p, s_p, lbl in (
                                ("HOME", h_od, bk_hp, float(sharp_only_hp), f"{h} ML"),
                                ("AWAY", a_od, (1.0 - bk_hp) if bk_hp is not None else None, 1.0 - float(sharp_only_hp), f"{a} ML"),
                            ):
                                alert = evaluate_steam_lag(
                                    book=bk,  # type: ignore[arg-type]
                                    side_label=lbl,
                                    side=sd,
                                    book_odds=od,
                                    book_devig_prob=b_p,
                                    sharp_prob=s_p,
                                    sharp_source=sharp_src,
                                )
                                if alert is not None:
                                    steam_list.append({**base_meta, "market_type": "Moneyline", **alert.to_dict()})

                # -------------------------------------------------------------
                # 2. SPREAD
                # -------------------------------------------------------------
                if market_filter.lower() in ("all", "game", "spread"):
                    dk_sp = mkt.get("dk_home_spread")
                    dk_h_sp_od = mkt.get("dk_home_spread_odds")
                    dk_a_sp_od = mkt.get("dk_away_spread_odds")
                    fd_sp = mkt.get("fd_home_spread")
                    fd_h_sp_od = mkt.get("fd_home_spread_odds")
                    fd_a_sp_od = mkt.get("fd_away_spread_odds")

                    if dk_sp is None and mkt.get("home_spread") is not None and str(mkt.get("line_source", "")) in ("Offline Fallback", "Schedule Archive"):
                        dk_sp = float(mkt["home_spread"])
                        dk_h_sp_od = -110
                        dk_a_sp_od = -110

                    sharp_sp = mkt.get("sharp_only_home_spread", mkt.get("sharp_home_spread", mkt.get("home_spread")))
                    sp_evals: list[dict[str, Any]] = []
                    for bk, sp_ln, h_od, a_od, bk_cov in (
                        ("DRAFTKINGS", dk_sp, dk_h_sp_od, dk_a_sp_od, mkt.get("dk_home_cover_prob")),
                        ("FANDUEL", fd_sp, fd_h_sp_od, fd_a_sp_od, mkt.get("fd_home_cover_prob")),
                    ):
                        if sp_ln is None or h_od is None or a_od is None:
                            continue
                        if book_f != "all" and bk.lower() != book_f:
                            continue
                        mod_cov = self._sb_spread_cover_prob(sp, gpred, float(sp_ln), is_live, rem_mins)
                        sh_anchor = (
                            clamp_prob(0.50 + (float(sp_ln) - float(sharp_sp)) * (0.032 if sp == "NFL" else 0.028))
                            if sharp_sp is not None
                            else (bk_cov if bk_cov is not None else 0.50)
                        )
                        blend_cov = blend_probabilities(mod_cov, sh_anchor, market_weight=0.62 if sharp_sp is not None else 0.35)
                        for sd, od, p_s, sh_s, lbl, eff_ln in (
                            ("HOME", int(h_od), blend_cov, sh_anchor, f"{h} {float(sp_ln):+.1f}", float(sp_ln)),
                            ("AWAY", int(a_od), 1.0 - blend_cov, 1.0 - sh_anchor, f"{a} {-float(sp_ln):+.1f}", -float(sp_ln)),
                        ):
                            ev_res = evaluate_sportsbook_side(
                                book=bk,  # type: ignore[arg-type]
                                side=sd,
                                side_true_prob=p_s,
                                american_odds=od,
                                bankroll=eff_bankroll,
                                kelly_multiplier=kelly_m,
                                max_bet_pct=self.settings.max_single_bet_pct,
                                min_ev_edge=eff_min_ev,
                                min_win_prob=eff_min_prob,
                                anchor_prob=sh_s,
                                uncertainty=0.14 if is_live else 0.12,
                                injury_clarity=clarity,
                                max_neg_clv=0.0 if sharp_sp is not None else 0.015,
                            )
                            sp_evals.append({"eval": ev_res, "label": lbl, "line": eff_ln, "home_sp": float(sp_ln), "blend_home": blend_cov})

                    if sp_evals:
                        best_item = max(
                            sp_evals,
                            key=lambda x: (
                                x["eval"].confidence_tier != "PASS",
                                x["eval"].ev_pct,
                                x["eval"].confidence_score,
                            ),
                        )
                        best_ev = best_item["eval"]
                        if locked:
                            best_ev = evaluate_sportsbook_side(
                                book=best_ev.book,
                                side=best_ev.side,
                                side_true_prob=best_ev.model_prob,
                                american_odds=best_ev.american_odds,
                                bankroll=0.0,
                                min_ev_edge=99.0,
                            )
                        other_same = next(
                            (
                                x["eval"]
                                for x in sp_evals
                                if x["eval"].side == best_ev.side and x["eval"].book != best_ev.book
                            ),
                            None,
                        )
                        shop_gain = (
                            round(best_ev.ev_pct - other_same.ev_pct, 4)
                            if other_same is not None
                            else None
                        )
                        drivers = list(safety_drivers) + [
                            f"Model proj line {h} {-gpred.expected_margin:+.1f} vs {best_item['home_sp']:+.1f}"
                        ] + list(gpred.drivers[:2])
                        picks_all.append(
                            {
                                **base_meta,
                                "pick_id": f"{slate_date}:{sp}:{ev_id}:sb:spread",
                                "market_type": "Live Spread" if is_live else "Spread",
                                "selection_label": best_item["label"],
                                "line_value": best_item["line"],
                                "dk_line": (float(dk_sp) if best_ev.side == "HOME" else -float(dk_sp)) if dk_sp is not None else None,
                                "fd_line": (float(fd_sp) if best_ev.side == "HOME" else -float(fd_sp)) if fd_sp is not None else None,
                                "dk_odds": (dk_h_sp_od if best_ev.side == "HOME" else dk_a_sp_od),
                                "fd_odds": (fd_h_sp_od if best_ev.side == "HOME" else fd_a_sp_od),
                                "line_shopping_gain_pct": shop_gain,
                                "yes_true_prob": round(best_item["blend_home"], 4),
                                "drivers": drivers[:4],
                                **best_ev.to_dict(),
                            }
                        )

                    # Spread Arb + Low-Hold + Bonus Bet (only when opposing lines don't leave an unhedged hole!)
                    if dk_sp is not None and fd_sp is not None:
                        _scan_cross_book_pair(
                            mkt_label="Live Spread" if is_live else "Spread",
                            dk_side_a_label=f"{h} {float(dk_sp):+.1f}",
                            dk_side_a_code="HOME",
                            dk_odds_a=dk_h_sp_od,
                            dk_side_b_label=f"{a} {-float(dk_sp):+.1f}",
                            dk_side_b_code="AWAY",
                            dk_odds_b=dk_a_sp_od,
                            fd_side_a_label=f"{h} {float(fd_sp):+.1f}",
                            fd_side_a_code="HOME",
                            fd_odds_a=fd_h_sp_od,
                            fd_side_b_label=f"{a} {-float(fd_sp):+.1f}",
                            fd_side_b_code="AWAY",
                            fd_odds_b=fd_a_sp_od,
                            allow_dk_a_vs_fd_b=(float(dk_sp) >= float(fd_sp)),
                            allow_fd_a_vs_dk_b=(float(fd_sp) >= float(dk_sp)),
                            is_freeroll_a=(float(dk_sp) > float(fd_sp)),
                            is_freeroll_b=(float(fd_sp) > float(dk_sp)),
                            dkex_yes_prob=mkt.get("dk_home_cover_prob"),
                            sharp_yes_prob=mkt.get("home_cover_prob"),
                            blended_yes_prob=gpred.home_cover_prob,
                        )

                        # Spread Middle Detector
                        if abs(float(dk_sp) - float(fd_sp)) >= 0.5 and dk_h_sp_od and dk_a_sp_od and fd_h_sp_od and fd_a_sp_od:
                            if float(dk_sp) > float(fd_sp):
                                hb, hln, hod = "DRAFTKINGS", float(dk_sp), int(dk_h_sp_od)
                                ab, aln, aod = "FANDUEL", float(fd_sp), int(fd_a_sp_od)
                            else:
                                hb, hln, hod = "FANDUEL", float(fd_sp), int(fd_h_sp_od)
                                ab, aln, aod = "DRAFTKINGS", float(dk_sp), int(dk_a_sp_od)
                            lo_m, hi_m = -hln, -aln
                            if hi_m > lo_m:
                                p_bw, p_pa, p_pb = self._sb_margin_interval_probs(sp, gpred, lo_m, hi_m, is_live, rem_mins)
                                keys = (
                                    [k for k in NFL_KEY_NUMBERS if (lo_m <= k <= hi_m) or (lo_m <= -k <= hi_m)]
                                    if sp == "NFL"
                                    else []
                                )
                                mid_obj = evaluate_middle_opportunity(
                                    book_a=hb,  # type: ignore[arg-type]
                                    label_a=f"{h} {hln:+.1f}",
                                    side_a="HOME",
                                    line_a=hln,
                                    odds_a=hod,
                                    book_b=ab,  # type: ignore[arg-type]
                                    label_b=f"{a} {-aln:+.1f}",
                                    side_b="AWAY",
                                    line_b=-aln,
                                    odds_b=aod,
                                    middle_window_label=f"{h} margin between {lo_m:g} and {hi_m:g}",
                                    middle_width=hi_m - lo_m,
                                    p_both_win=p_bw,
                                    p_push_a_win_b=p_pa,
                                    p_push_b_win_a=p_pb,
                                    key_numbers_crossed=keys,
                                    total_stake=arb_stake,
                                )
                                middles_list.append({**base_meta, "market_type": "Spread", **mid_obj.to_dict()})

                    # Spread Steam Lag
                    sharp_only_sp = mkt.get("sharp_only_home_spread")
                    if sharp_only_sp is not None:
                        sharp_src = f"{int(mkt.get('sharp_books_count', 1))}-Book Sharp"
                        for bk, sp_ln, h_od, a_od, b_cov in (
                            ("DRAFTKINGS", dk_sp, dk_h_sp_od, dk_a_sp_od, mkt.get("dk_home_cover_prob")),
                            ("FANDUEL", fd_sp, fd_h_sp_od, fd_a_sp_od, mkt.get("fd_home_cover_prob")),
                        ):
                            if sp_ln is None or h_od is None or a_od is None:
                                continue
                            sh_p_h = clamp_prob(0.50 + (float(sp_ln) - float(sharp_only_sp)) * (0.035 if sp == "NFL" else 0.030))
                            for sd, od, bp, sp_p, lbl, b_l, s_l in (
                                ("HOME", h_od, b_cov, sh_p_h, f"{h} {float(sp_ln):+.1f}", float(sp_ln), float(sharp_only_sp)),
                                ("AWAY", a_od, (1.0 - b_cov) if b_cov is not None else None, 1.0 - sh_p_h, f"{a} {-float(sp_ln):+.1f}", -float(sp_ln), -float(sharp_only_sp)),
                            ):
                                alert = evaluate_steam_lag(
                                    book=bk,  # type: ignore[arg-type]
                                    side_label=lbl,
                                    side=sd,
                                    book_odds=od,
                                    book_devig_prob=bp,
                                    sharp_prob=sp_p,
                                    book_line=b_l,
                                    sharp_line=s_l,
                                    sharp_source=sharp_src,
                                )
                                if alert is not None:
                                    steam_list.append({**base_meta, "market_type": "Spread", **alert.to_dict()})

                # -------------------------------------------------------------
                # 3. TOTAL (OVER / UNDER)
                # -------------------------------------------------------------
                if market_filter.lower() in ("all", "game", "total"):
                    dk_tot = mkt.get("dk_total_line")
                    dk_ov_od = mkt.get("dk_over_odds")
                    dk_un_od = mkt.get("dk_under_odds")
                    fd_tot = mkt.get("fd_total_line")
                    fd_ov_od = mkt.get("fd_over_odds")
                    fd_un_od = mkt.get("fd_under_odds")

                    if dk_tot is None and mkt.get("total_line") is not None and str(mkt.get("line_source", "")) in ("Offline Fallback", "Schedule Archive"):
                        dk_tot = float(mkt["total_line"])
                        dk_ov_od = -110
                        dk_un_od = -110

                    sharp_tot = mkt.get("sharp_only_total_line", mkt.get("sharp_total_line", mkt.get("total_line")))
                    tot_evals: list[dict[str, Any]] = []
                    for bk, t_ln, o_od, u_od, bk_op in (
                        ("DRAFTKINGS", dk_tot, dk_ov_od, dk_un_od, mkt.get("dk_over_prob")),
                        ("FANDUEL", fd_tot, fd_ov_od, fd_un_od, mkt.get("fd_over_prob")),
                    ):
                        if t_ln is None or o_od is None or u_od is None:
                            continue
                        if book_f != "all" and bk.lower() != book_f:
                            continue
                        mod_ov = self._sb_total_over_prob(sp, gpred, float(t_ln), is_live, rem_mins)
                        sh_ov = (
                            clamp_prob(0.50 + (float(sharp_tot) - float(t_ln)) * (0.026 if sp == "NFL" else 0.022))
                            if sharp_tot is not None
                            else (bk_op if bk_op is not None else 0.50)
                        )
                        blend_ov = blend_probabilities(mod_ov, sh_ov, market_weight=0.62 if sharp_tot is not None else 0.35)
                        for sd, od, p_s, sh_s, lbl in (
                            ("OVER", int(o_od), blend_ov, sh_ov, f"Over {float(t_ln):.1f}"),
                            ("UNDER", int(u_od), 1.0 - blend_ov, 1.0 - sh_ov, f"Under {float(t_ln):.1f}"),
                        ):
                            ev_res = evaluate_sportsbook_side(
                                book=bk,  # type: ignore[arg-type]
                                side=sd,
                                side_true_prob=p_s,
                                american_odds=od,
                                bankroll=eff_bankroll,
                                kelly_multiplier=kelly_m,
                                max_bet_pct=self.settings.max_single_bet_pct,
                                min_ev_edge=eff_min_ev,
                                min_win_prob=eff_min_prob,
                                anchor_prob=sh_s,
                                uncertainty=0.14 if is_live else 0.12,
                                injury_clarity=clarity,
                                max_neg_clv=0.0 if sharp_tot is not None else 0.015,
                            )
                            tot_evals.append({"eval": ev_res, "label": lbl, "line": float(t_ln), "blend_ov": blend_ov})

                    if tot_evals:
                        best_item = max(
                            tot_evals,
                            key=lambda x: (
                                x["eval"].confidence_tier != "PASS",
                                x["eval"].ev_pct,
                                x["eval"].confidence_score,
                            ),
                        )
                        best_ev = best_item["eval"]
                        if locked:
                            best_ev = evaluate_sportsbook_side(
                                book=best_ev.book,
                                side=best_ev.side,
                                side_true_prob=best_ev.model_prob,
                                american_odds=best_ev.american_odds,
                                bankroll=0.0,
                                min_ev_edge=99.0,
                            )
                        other_same = next(
                            (
                                x["eval"]
                                for x in tot_evals
                                if x["eval"].side == best_ev.side and x["eval"].book != best_ev.book
                            ),
                            None,
                        )
                        shop_gain = (
                            round(best_ev.ev_pct - other_same.ev_pct, 4)
                            if other_same is not None
                            else None
                        )
                        drivers = list(safety_drivers) + [
                            f"Model proj total {gpred.expected_total:.1f} vs {best_item['line']:.1f}"
                        ] + list(gpred.drivers[:2])
                        picks_all.append(
                            {
                                **base_meta,
                                "pick_id": f"{slate_date}:{sp}:{ev_id}:sb:total",
                                "market_type": "Live Total" if is_live else "Total",
                                "selection_label": best_item["label"],
                                "line_value": best_item["line"],
                                "dk_line": float(dk_tot) if dk_tot is not None else None,
                                "fd_line": float(fd_tot) if fd_tot is not None else None,
                                "dk_odds": (dk_ov_od if best_ev.side == "OVER" else dk_un_od),
                                "fd_odds": (fd_ov_od if best_ev.side == "OVER" else fd_un_od),
                                "line_shopping_gain_pct": shop_gain,
                                "yes_true_prob": round(best_item["blend_ov"], 4),
                                "drivers": drivers[:4],
                                **best_ev.to_dict(),
                            }
                        )

                    if dk_tot is not None and fd_tot is not None:
                        _scan_cross_book_pair(
                            mkt_label="Live Total" if is_live else "Total",
                            dk_side_a_label=f"Over {float(dk_tot):.1f}",
                            dk_side_a_code="OVER",
                            dk_odds_a=dk_ov_od,
                            dk_side_b_label=f"Under {float(dk_tot):.1f}",
                            dk_side_b_code="UNDER",
                            dk_odds_b=dk_un_od,
                            fd_side_a_label=f"Over {float(fd_tot):.1f}",
                            fd_side_a_code="OVER",
                            fd_odds_a=fd_ov_od,
                            fd_side_b_label=f"Under {float(fd_tot):.1f}",
                            fd_side_b_code="UNDER",
                            fd_odds_b=fd_un_od,
                            allow_dk_a_vs_fd_b=(float(dk_tot) <= float(fd_tot)),
                            allow_fd_a_vs_dk_b=(float(fd_tot) <= float(dk_tot)),
                            is_freeroll_a=(float(dk_tot) < float(fd_tot)),
                            is_freeroll_b=(float(fd_tot) < float(dk_tot)),
                            dkex_yes_prob=mkt.get("dk_over_prob"),
                            sharp_yes_prob=mkt.get("over_prob"),
                            blended_yes_prob=gpred.over_prob,
                        )

                        # Total Middle Detector (Over on LOWER line + Under on HIGHER line)
                        if abs(float(dk_tot) - float(fd_tot)) >= 0.5 and dk_ov_od and dk_un_od and fd_ov_od and fd_un_od:
                            if float(dk_tot) < float(fd_tot):
                                ob, oln, ood = "DRAFTKINGS", float(dk_tot), int(dk_ov_od)
                                ub, uln, uod = "FANDUEL", float(fd_tot), int(fd_un_od)
                            else:
                                ob, oln, ood = "FANDUEL", float(fd_tot), int(fd_ov_od)
                                ub, uln, uod = "DRAFTKINGS", float(dk_tot), int(dk_un_od)
                            p_bw, p_pa, p_pb = self._sb_total_interval_probs(sp, gpred, oln, uln, is_live, rem_mins)
                            mid_t = evaluate_middle_opportunity(
                                book_a=ob,  # type: ignore[arg-type]
                                label_a=f"Over {oln:.1f}",
                                side_a="OVER",
                                line_a=oln,
                                odds_a=ood,
                                book_b=ub,  # type: ignore[arg-type]
                                label_b=f"Under {uln:.1f}",
                                side_b="UNDER",
                                line_b=uln,
                                odds_b=uod,
                                middle_window_label=f"Total between {oln:g} and {uln:g}",
                                middle_width=uln - oln,
                                p_both_win=p_bw,
                                p_push_a_win_b=p_pa,
                                p_push_b_win_a=p_pb,
                                total_stake=arb_stake,
                            )
                            middles_list.append({**base_meta, "market_type": "Total", **mid_t.to_dict()})

                # -------------------------------------------------------------
                # 4. PLAYER PROPS (DK vs FD Line Shop, Prop Arbs, Prop Middles)
                # -------------------------------------------------------------
                for pp in prop_preds:
                    bmap = getattr(pp, "book_lines", None) or {}
                    dk_p = bmap.get("draftkings")
                    fd_p = bmap.get("fanduel")
                    if fd_p is not None:
                        has_any_fanduel = True

                    # Never fabricate fake -110 DraftKings odds on unpriced Model Ref Lines or non-DK/FD books
                    if dk_p is None and fd_p is None:
                        continue

                    pp_line_src = (
                        "DK + FD"
                        if (dk_p is not None and fd_p is not None)
                        else ("DraftKings" if dk_p is not None else "FanDuel")
                    )

                    # Cross-book consensus over probability when both DK & FD post the same line
                    same_ln_probs = [
                        float(e["over_prob"])
                        for e in (dk_p, fd_p)
                        if e and e.get("over_prob") is not None
                    ]
                    consensus_same_line = (
                        (sum(same_ln_probs) / len(same_ln_probs))
                        if (
                            dk_p is not None
                            and fd_p is not None
                            and abs(float(dk_p["line"]) - float(fd_p["line"])) < 0.01
                            and same_ln_probs
                        )
                        else None
                    )

                    pp_evals: list[dict[str, Any]] = []
                    for bk, entry in (("DRAFTKINGS", dk_p), ("FANDUEL", fd_p)):
                        if not entry:
                            continue
                        if book_f != "all" and bk.lower() != book_f:
                            continue
                        ln_v = float(entry["line"])
                        o_od = int(entry["over_odds"])
                        u_od = int(entry["under_odds"])
                        mod_ov = (
                            float(pp.over_prob)
                            if abs(ln_v - float(pp.line_value)) < 0.01
                            else prop_over_prob_for_line(sp, pp.prop_type, pp.projected_mean, pp.projected_std, ln_v)
                        )
                        mkt_ov = consensus_same_line if consensus_same_line is not None else entry.get("over_prob")
                        blend_ov = (
                            blend_probabilities(mod_ov, float(mkt_ov), market_weight=0.52)
                            if mkt_ov is not None
                            else mod_ov
                        )
                        for sd, od, p_s, sh_s, lbl in (
                            ("OVER", o_od, blend_ov, mkt_ov, f"{pp.player_name} ({pp.team}) Over {ln_v:.1f} {pp.prop_label}"),
                            ("UNDER", u_od, 1.0 - blend_ov, (1.0 - float(mkt_ov)) if mkt_ov is not None else None, f"{pp.player_name} ({pp.team}) Under {ln_v:.1f} {pp.prop_label}"),
                        ):
                            ev_res = evaluate_sportsbook_side(
                                book=bk,  # type: ignore[arg-type]
                                side=sd,
                                side_true_prob=p_s,
                                american_odds=od,
                                bankroll=eff_bankroll,
                                kelly_multiplier=kelly_m,
                                max_bet_pct=min(0.03, self.settings.max_single_bet_pct),
                                min_ev_edge=eff_min_ev,
                                min_win_prob=eff_min_prob,
                                anchor_prob=sh_s,
                                uncertainty=0.16,
                                injury_clarity=pp.injury_clarity,
                                max_neg_clv=0.055,
                            )
                            pp_evals.append({"eval": ev_res, "label": lbl, "line": ln_v, "blend_ov": blend_ov})

                    if pp_evals:
                        best_item = max(
                            pp_evals,
                            key=lambda x: (
                                x["eval"].confidence_tier != "PASS",
                                x["eval"].ev_pct,
                                x["eval"].confidence_score,
                            ),
                        )
                        best_ev = best_item["eval"]
                        if locked:
                            best_ev = evaluate_sportsbook_side(
                                book=best_ev.book,
                                side=best_ev.side,
                                side_true_prob=best_ev.model_prob,
                                american_odds=best_ev.american_odds,
                                bankroll=0.0,
                                min_ev_edge=99.0,
                            )
                        other_same = next(
                            (
                                x["eval"]
                                for x in pp_evals
                                if x["eval"].side == best_ev.side and x["eval"].book != best_ev.book
                            ),
                            None,
                        )
                        shop_gain = (
                            round(best_ev.ev_pct - other_same.ev_pct, 4)
                            if other_same is not None
                            else None
                        )
                        slug = pp.player_name.lower().replace(" ", "_")
                        picks_all.append(
                            {
                                **base_meta,
                                "pick_id": f"{slate_date}:{sp}:{ev_id}:sb:prop:{slug}:{pp.prop_type}",
                                "market_type": f"{'Live ' if is_live else ''}Prop: {pp.prop_label}",
                                "selection_label": best_item["label"],
                                "player_name": pp.player_name,
                                "line_value": best_item["line"],
                                "dk_line": float(dk_p["line"]) if dk_p else None,
                                "fd_line": float(fd_p["line"]) if fd_p else None,
                                "dk_odds": (dk_p["over_odds"] if best_ev.side == "OVER" else dk_p["under_odds"]) if dk_p else None,
                                "fd_odds": (fd_p["over_odds"] if best_ev.side == "OVER" else fd_p["under_odds"]) if fd_p else None,
                                "line_shopping_gain_pct": shop_gain,
                                "yes_true_prob": round(best_item["blend_ov"], 4),
                                "line_source": pp_line_src,
                                "drivers": (list(safety_drivers) + list(pp.drivers))[:4],
                                **best_ev.to_dict(),
                            }
                        )

                    # Prop Arbitrage, Low-Hold, Bonus Bet & Prop Middles when both DK & FD post the prop
                    if dk_p and fd_p:
                        dk_ln, fd_ln = float(dk_p["line"]), float(fd_p["line"])
                        _scan_cross_book_pair(
                            mkt_label=f"Prop: {pp.prop_label}",
                            dk_side_a_label=f"{pp.player_name} Over {dk_ln:.1f} {pp.prop_label}",
                            dk_side_a_code="OVER",
                            dk_odds_a=int(dk_p["over_odds"]),
                            dk_side_b_label=f"{pp.player_name} Under {dk_ln:.1f} {pp.prop_label}",
                            dk_side_b_code="UNDER",
                            dk_odds_b=int(dk_p["under_odds"]),
                            fd_side_a_label=f"{pp.player_name} Over {fd_ln:.1f} {pp.prop_label}",
                            fd_side_a_code="OVER",
                            fd_odds_a=int(fd_p["over_odds"]),
                            fd_side_b_label=f"{pp.player_name} Under {fd_ln:.1f} {pp.prop_label}",
                            fd_side_b_code="UNDER",
                            fd_odds_b=int(fd_p["under_odds"]),
                            allow_dk_a_vs_fd_b=(dk_ln <= fd_ln),
                            allow_fd_a_vs_dk_b=(fd_ln <= dk_ln),
                            is_freeroll_a=(dk_ln < fd_ln),
                            is_freeroll_b=(fd_ln < dk_ln),
                        )
                        if abs(dk_ln - fd_ln) >= 0.5:
                            if dk_ln < fd_ln:
                                ob, oln, ood = "DRAFTKINGS", dk_ln, int(dk_p["over_odds"])
                                ub, uln, uod = "FANDUEL", fd_ln, int(fd_p["under_odds"])
                            else:
                                ob, oln, ood = "FANDUEL", fd_ln, int(fd_p["over_odds"])
                                ub, uln, uod = "DRAFTKINGS", dk_ln, int(dk_p["under_odds"])
                            p_bw, p_pa, p_pb = self._sb_prop_interval_probs(pp, oln, uln)
                            mid_p = evaluate_middle_opportunity(
                                book_a=ob,  # type: ignore[arg-type]
                                label_a=f"{pp.player_name} Over {oln:.1f} {pp.prop_label}",
                                side_a="OVER",
                                line_a=oln,
                                odds_a=ood,
                                book_b=ub,  # type: ignore[arg-type]
                                label_b=f"{pp.player_name} Under {uln:.1f} {pp.prop_label}",
                                side_b="UNDER",
                                line_b=uln,
                                odds_b=uod,
                                middle_window_label=f"{pp.player_name} {pp.prop_label} ({oln:g}, {uln:g})",
                                middle_width=uln - oln,
                                p_both_win=p_bw,
                                p_push_a_win_b=p_pa,
                                p_push_b_win_a=p_pb,
                                total_stake=arb_stake,
                            )
                            middles_list.append({**base_meta, "market_type": f"Prop: {pp.prop_label}", **mid_p.to_dict()})

        # Sort each strategy feed cleanly
        picks_all.sort(
            key=lambda x: (
                x["confidence_tier"] != "PASS",
                x["ev_pct"],
                x["confidence_score"],
            ),
            reverse=True,
        )
        # Apply same-game dollar cap on qualifying +EV sportsbook picks
        max_game_dollars = max(eff_bankroll * self.settings.max_same_game_pct, 5.0)
        used_game: dict[str, float] = {}
        for p in picks_all:
            if p["confidence_tier"] == "PASS" or float(p["recommended_stake"]) <= 0:
                continue
            eid = str(p["event_id"])
            rem = max(0.0, max_game_dollars - used_game.get(eid, 0.0))
            if float(p["recommended_stake"]) > rem:
                if rem < 1.0:
                    p["recommended_stake"] = 0.0
                    p["confidence_tier"] = "PASS"
                else:
                    p["recommended_stake"] = round(rem, 2)
                    p["ev_dollars"] = round(rem * float(p["ev_pct"]), 2)
            used_game[eid] = used_game.get(eid, 0.0) + float(p["recommended_stake"])

        arbitrage_list.sort(
            key=lambda x: (x["is_guaranteed_after_rounding"], x["min_roi_pct"], x["exact_roi_pct"]),
            reverse=True,
        )
        middles_list.sort(
            key=lambda x: (x["is_positive_ev"], x["ev_roi_pct"], len(x["key_numbers_crossed"])),
            reverse=True,
        )
        low_hold_list.sort(key=lambda x: x["hold_pct"])
        bonus_bet_list.sort(key=lambda x: x["retention_pct"], reverse=True)
        profit_boost_list.sort(key=lambda x: x["ev_pct"], reverse=True)
        steam_list.sort(key=lambda x: (x["ev_vs_sharp_pct"], x["prob_lag"]), reverse=True)

        visible_picks = (
            picks_all
            if include_passes
            else [p for p in picks_all if p["confidence_tier"] != "PASS"]
        )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "games_count": len(games_seen),
            "total_candidates": len(picks_all),
            "count": len(visible_picks),
            "live_count": sum(1 for p in visible_picks if p.get("game_state") == "LIVE"),
            "high_confidence_count": sum(1 for p in visible_picks if p["confidence_tier"] in ("A+", "A")),
            "odds_api_connected": bool(self.settings.the_odds_api_key),
            "has_fanduel_odds": has_any_fanduel,
            "odds_quota": self.odds.get_quota_status(),
            "prop_diagnostics": prop_diag,
            "picks": visible_picks,
            "arbitrage": arbitrage_list[:40],
            "middles": middles_list[:30],
            "promos": {
                "low_hold": low_hold_list[:25],
                "bonus_bet": bonus_bet_list[:25],
                "profit_boost": profit_boost_list[:20],
            },
            "steam": steam_list[:30],
            "strategy_counts": {
                "arbitrage": len(arbitrage_list),
                "middles_ev": sum(1 for m in middles_list if m["is_positive_ev"]),
                "middles_total": len(middles_list),
                "low_hold": len(low_hold_list),
                "bonus_bet_70plus": sum(1 for b in bonus_bet_list if b["retention_pct"] >= 0.70),
                "profit_boost": len(profit_boost_list),
                "steam": len(steam_list),
            },
        }

    def reevaluate_sportsbook_bet(
        self,
        *,
        true_prob: float | None = None,
        sharp_odds_side: float | None = None,
        sharp_odds_opp: float | None = None,
        dk_odds: float | None = None,
        fd_odds: float | None = None,
        bankroll: float | None = None,
        profit_boost_pct: float = 0.0,
        is_live: bool = False,
    ) -> dict[str, Any]:
        """Interactive Traditional Sportsbooks +EV, Line-Shop, & Profit Boost Analyzer."""
        if true_prob is None:
            if sharp_odds_side is None or sharp_odds_opp is None:
                raise ValueError("Provide either true_prob or both sharp_odds_side and sharp_odds_opp.")
            p_side, _ = devig_american_pair(float(sharp_odds_side), float(sharp_odds_opp))
            anchor = p_side
        else:
            p_side = clamp_prob(float(true_prob))
            anchor = (
                devig_american_pair(float(sharp_odds_side), float(sharp_odds_opp))[0]
                if (sharp_odds_side is not None and sharp_odds_opp is not None)
                else None
            )

        bk_val = bankroll if bankroll is not None else self.settings.bankroll
        kelly_m = self.settings.kelly_fraction * (0.80 if is_live else 1.0)

        results: dict[str, Any] = {
            "true_prob": round(p_side, 4),
            "fair_american_odds": prob_to_american_odds(p_side),
            "profit_boost_pct": round(max(0.0, float(profit_boost_pct)), 4),
            "books": {},
        }
        evals: list[dict[str, Any]] = []
        for book_key, odds_val in (("DRAFTKINGS", dk_odds), ("FANDUEL", fd_odds)):
            if odds_val is None:
                continue
            ev = evaluate_sportsbook_side(
                book=book_key,  # type: ignore[arg-type]
                side="SELECTED",
                side_true_prob=p_side,
                american_odds=float(odds_val),
                bankroll=bk_val,
                kelly_multiplier=kelly_m,
                max_bet_pct=self.settings.max_single_bet_pct,
                min_ev_edge=self.settings.min_sb_ev_edge,
                anchor_prob=anchor,
                uncertainty=0.14 if is_live else 0.10,
                profit_boost_pct=profit_boost_pct,
            ).to_dict()
            results["books"][book_key] = ev
            evals.append(ev)

        if not evals:
            raise ValueError("Provide at least one of dk_odds or fd_odds.")

        evals.sort(key=lambda x: x["ev_pct"], reverse=True)
        results["best"] = evals[0]
        if len(evals) == 2:
            results["line_shopping_gain_pct"] = round(evals[0]["ev_pct"] - evals[1]["ev_pct"], 4)
        else:
            results["line_shopping_gain_pct"] = None
        return results

    def calculate_arb_or_hedge(
        self,
        *,
        book_a: str = "DRAFTKINGS",
        odds_a: float | None = None,
        dkex_ask_a: float | None = None,
        dkex_fee_a: float | None = None,
        book_b: str = "FANDUEL",
        odds_b: float | None = None,
        dkex_ask_b: float | None = None,
        dkex_fee_b: float | None = None,
        total_stake: float = 100.0,
        fixed_stake_a: float | None = None,
        rounding_step: float | None = None,
    ) -> dict[str, Any]:
        """Interactive 2-Way Arbitrage & Hedge Calculator (supports DK<->FD and DKeX<->DK/FD)."""
        ba = "DKEX" if book_a.upper() == "DKEX" or dkex_ask_a is not None else (
            "FANDUEL" if "FAN" in book_a.upper() or book_a.upper() == "FD" else "DRAFTKINGS"
        )
        bb = "DKEX" if book_b.upper() == "DKEX" or dkex_ask_b is not None else (
            "FANDUEL" if "FAN" in book_b.upper() or book_b.upper() == "FD" else "DRAFTKINGS"
        )
        fee_a = dkex_fee_a if dkex_fee_a is not None else self.settings.default_fee_dollars
        fee_b = dkex_fee_b if dkex_fee_b is not None else self.settings.default_fee_dollars
        step = rounding_step if rounding_step is not None else self.settings.arb_stake_rounding

        dec_a = (
            (1.0 / max(0.01, min(0.999, float(dkex_ask_a) + fee_a)))
            if ba == "DKEX" and dkex_ask_a is not None
            else american_to_decimal(float(odds_a if odds_a is not None else -110))
        )
        dec_b = (
            (1.0 / max(0.01, min(0.999, float(dkex_ask_b) + fee_b)))
            if bb == "DKEX" and dkex_ask_b is not None
            else american_to_decimal(float(odds_b if odds_b is not None else -110))
        )

        # If user provides a fixed Leg A stake (e.g., hedging an existing open bet), derive total stake for equal payout
        eff_total = float(total_stake)
        if fixed_stake_a is not None and float(fixed_stake_a) > 0:
            hedge_b = float(fixed_stake_a) * dec_a / dec_b
            eff_total = float(fixed_stake_a) + hedge_b

        imp_sum = (1.0 / dec_a) + (1.0 / dec_b)
        arb = evaluate_arbitrage_pair(
            book_a=ba,  # type: ignore[arg-type]
            label_a="Leg A",
            side_a="SIDE_A",
            odds_a=odds_a if ba != "DKEX" else None,
            dkex_ask_a=dkex_ask_a if ba == "DKEX" else None,
            dkex_fee_a=fee_a,
            book_b=bb,  # type: ignore[arg-type]
            label_b="Leg B",
            side_b="SIDE_B",
            odds_b=odds_b if bb != "DKEX" else None,
            dkex_ask_b=dkex_ask_b if bb == "DKEX" else None,
            dkex_fee_b=fee_b,
            total_stake=eff_total,
            rounding_step=step,
            min_arb_roi=-1.0,  # Always evaluate even if hold > 0 for hedge math!
        )
        if arb is not None:
            return {"is_arb": True, "hold_pct": round(imp_sum - 1.0, 5), **arb.to_dict()}

        # Non-arb hedge math (imp_sum >= 1.0)
        stake_a = round(eff_total * ((1.0 / dec_a) / imp_sum), 2)
        stake_b = round(eff_total - stake_a, 2)
        payout = round(stake_a * dec_a, 2)
        net = round(payout - eff_total, 2)
        return {
            "is_arb": False,
            "hold_pct": round(imp_sum - 1.0, 5),
            "implied_sum": round(imp_sum, 5),
            "exact_roi_pct": round(net / eff_total, 5) if eff_total > 0 else 0.0,
            "exact_total_stake": round(eff_total, 2),
            "exact_guaranteed_profit": net,
            "rounded_total_stake": round(stake_a + stake_b, 2),
            "min_guaranteed_profit": net,
            "max_profit": net,
            "min_roi_pct": round(net / eff_total, 5) if eff_total > 0 else 0.0,
            "is_guaranteed_after_rounding": False,
            "leg_a": {
                "book": ba,
                "american_odds": decimal_to_american(dec_a),
                "decimal_odds": round(dec_a, 4),
                "exact_stake": stake_a,
                "rounded_stake": stake_a,
                "payout_if_wins_exact": payout,
                "payout_if_wins_rounded": payout,
            },
            "leg_b": {
                "book": bb,
                "american_odds": decimal_to_american(dec_b),
                "decimal_odds": round(dec_b, 4),
                "exact_stake": stake_b,
                "rounded_stake": stake_b,
                "payout_if_wins_exact": round(stake_b * dec_b, 2),
                "payout_if_wins_rounded": round(stake_b * dec_b, 2),
            },
        }

    @staticmethod
    def calculate_bonus_bet_conversion(
        *,
        bonus_book: str = "DRAFTKINGS",
        bonus_odds: float = 350.0,
        hedge_book: str = "FANDUEL",
        hedge_odds: float = -400.0,
        bonus_amount: float = 50.0,
    ) -> dict[str, Any]:
        bb = "FANDUEL" if "FAN" in bonus_book.upper() or bonus_book.upper() == "FD" else "DRAFTKINGS"
        hb = "FANDUEL" if "FAN" in hedge_book.upper() or hedge_book.upper() == "FD" else "DRAFTKINGS"
        conv = evaluate_bonus_bet_conversion(
            book_a=bb,  # type: ignore[arg-type]
            label_a="Bonus Bet Leg",
            side_a="BONUS",
            odds_a=bonus_odds,
            book_b=hb,  # type: ignore[arg-type]
            label_b="Cash Hedge Leg",
            side_b="HEDGE",
            odds_b=hedge_odds,
            bonus_amount=bonus_amount,
        )
        return conv.to_dict()


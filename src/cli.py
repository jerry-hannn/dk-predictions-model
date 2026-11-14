"""Command-line interface (`dkmodel`) for Pre-Game & Live In-Play DraftKings Predictions."""

from __future__ import annotations

import time
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.backtest.walk_forward import run_walk_forward_backtest
from src.config import get_settings
from src.engine.dk_math import evaluate_single_side
from src.engine.recommender import ModelHub

app = typer.Typer(
    name="dkmodel",
    help="NFL & NBA Pre-Game + Live In-Play Event Contract Engine for DraftKings Predictions (DKeX).",
    add_completion=False,
)
console = Console()


def _format_quota_line(quota: dict) -> str:
    rem = quota.get("requests_remaining")
    used = quota.get("requests_used")
    if rem is None:
        return "Odds API Quota: Not queried yet (or zero-key ESPN mode)"
    guard = " [bold yellow](LOW QUOTA GUARD ACTIVE)[/bold yellow]" if quota.get("low_quota_guard") else ""
    return f"Odds API Credits Remaining: [bold cyan]{rem}[/bold cyan] (Used: {used}){guard}"


def _render_picks_table(picks: list[dict], fee: float, title_prefix: str = "DraftKings Predictions") -> None:
    table = Table(
        title=f"{title_prefix} (Fee=${fee:.3f}/contract)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tier", justify="center", style="bold")
    table.add_column("Sport")
    table.add_column("State / Clock")
    table.add_column("Matchup / Score")
    table.add_column("Contract / Selection")
    table.add_column("Side", justify="center")
    table.add_column("Ref ¢", justify="right")
    table.add_column("Max Buy ¢", justify="right")
    table.add_column("Win Prob", justify="right")
    table.add_column("Net Edge", justify="right")
    table.add_column("Qty ($Outlay)", justify="right")
    table.add_column("Key Driver")

    tier_colors = {"A+": "bright_green", "A": "green", "B": "yellow", "PASS": "dim"}
    for p in picks[:30]:
        col = tier_colors.get(p["confidence_tier"], "white")
        is_live = p.get("game_state") == "LIVE"
        state_txt = (
            f"[bold red]LIVE[/bold red] {p.get('status_detail', '')}"
            if is_live
            else f"[cyan]PRE[/cyan] {p.get('status_detail', '')[:12]}"
        )
        matchup_txt = p["score_display"] if (is_live and p.get("score_display")) else p["short_name"]
        driver = p["drivers"][0] if p.get("drivers") else ""
        table.add_row(
            f"[{col}]{p['confidence_tier']} ({p['confidence_score']})[/{col}]",
            p["sport"],
            state_txt,
            matchup_txt,
            p["selection_label"],
            f"[bold]{p['recommended_side']}[/bold]",
            f"{p['contract_price']*100:.0f}¢",
            f"{p['fair_entry_max_price']*100:.0f}¢",
            f"{p['model_prob']*100:.1f}%",
            f"{p['net_edge']*100:+.1f}%",
            f"{p['recommended_contracts']} (${p['recommended_outlay']:.2f})",
            driver,
        )
    console.print(table)


@app.command("sync")
def sync_cmd(
    sport: str = typer.Option("all", "--sport", "-s", help="nfl, nba, or all"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="YYYY-MM-DD slate date"),
) -> None:
    """Sync historical schedules/logs PLUS live scoreboards, box scores, injuries, and sharp odds."""
    hub = ModelHub()
    with console.status("[bold cyan]Syncing NFL & NBA historical + live data feeds...[/bold cyan]"):
        summary = hub.sync_data(sport=sport, target_date=date, force=True)

    quota = summary.pop("odds_quota", {})
    table = Table(title="Data Sync Summary", show_header=True, header_style="bold cyan")
    table.add_column("League", style="bold")
    table.add_column("Historical Games", justify="right")
    table.add_column("Player Log Rows", justify="right")
    table.add_column("Scoreboard Events", justify="right")
    table.add_column("In-Play Now", justify="right")
    table.add_column("Injured Teams", justify="right")
    table.add_column("Sharp Odds Events", justify="right")

    for lg, stats in summary.items():
        table.add_row(
            lg.upper(),
            str(stats["historical_games"]),
            str(stats["player_log_rows"]),
            str(stats["live_scoreboard_events"]),
            str(stats.get("in_play_events", 0)),
            str(stats["injured_teams_tracked"]),
            str(stats.get("sharp_odds_events", 0)),
        )
    console.print(table)
    console.print(_format_quota_line(quota))


@app.command("train")
def train_cmd(
    sport: str = typer.Option("all", "--sport", "-s", help="nfl, nba, or all"),
) -> None:
    """Train & calibrate NFL game, NBA game, and Player Prop models."""
    hub = ModelHub()
    with console.status("[bold green]Training & calibrating ensemble models...[/bold green]"):
        metrics = hub.train_all(sport=sport)

    table = Table(title="Trained Model Artifacts & Diagnostics", header_style="bold green")
    table.add_column("Component", style="bold")
    table.add_column("Details")
    for comp, info in metrics.items():
        details = ", ".join(f"{k}={v}" for k, v in info.items())
        table.add_row(comp, details)
    console.print(table)


@app.command("backtest")
def backtest_cmd(
    sport: str = typer.Option("all", "--sport", "-s", help="nfl, nba, or all"),
    fee: float = typer.Option(0.015, "--fee", help="Per-contract DKeX fee in dollars"),
) -> None:
    """Run chronological walk-forward backtests with zero lookahead bias."""
    with console.status("[bold magenta]Running walk-forward out-of-sample backtests...[/bold magenta]"):
        res = run_walk_forward_backtest(sport=sport, fee_per_contract=fee, save_artifact=True)

    for lg, data in res.items():
        sim = data["simulation"]
        summary_txt = (
            f"[bold]{lg.upper()} Out-of-Sample Games:[/bold] {data['oos_games']}  |  "
            f"[bold]Brier:[/bold] {data['brier_score']:.4f}  |  "
            f"[bold]LogLoss:[/bold] {data['log_loss']:.4f}\n"
            f"[bold]Simulated DKeX Trades (Prob >= 60% & Fee-Adj Edge >= 2.5%):[/bold] {sim['total_bets']}  |  "
            f"[bold]Hit Rate:[/bold] {sim['hit_rate']*100:.1f}%  |  "
            f"[bold]Net ROI (after ${fee:.3f} fee):[/bold] {sim['roi_pct']*100:+.2f}%  |  "
            f"[bold]Max Drawdown:[/bold] {sim['max_drawdown_pct']*100:.1f}%"
        )
        console.print(Panel(summary_txt, title=f"{lg.upper()} Walk-Forward Validation", border_style="magenta"))

        tier_tbl = Table(show_header=True, header_style="bold")
        tier_tbl.add_column("Confidence Tier")
        tier_tbl.add_column("Trades", justify="right")
        tier_tbl.add_column("Wins", justify="right")
        tier_tbl.add_column("Win Rate", justify="right")
        tier_tbl.add_column("Net ROI", justify="right")
        tier_tbl.add_column("Net P&L ($1k Start)", justify="right")
        for tier_name in ("A+", "A", "B"):
            t = sim["by_tier"][tier_name]
            wr = f"{t['win_rate']*100:.1f}%" if t["win_rate"] is not None else "—"
            tier_tbl.add_row(
                tier_name,
                str(t["bets"]),
                str(t["wins"]),
                wr,
                f"{t['roi_pct']*100:+.1f}%",
                f"${t['net_pnl']:+.2f}",
            )
        console.print(tier_tbl)


@app.command("picks")
def picks_cmd(
    sport: str = typer.Option("all", "--sport", "-s", help="nfl, nba, or all"),
    market: str = typer.Option("all", "--market", "-m", help="all, game, or props"),
    mode: str = typer.Option("all", "--mode", help="all (pregame + live), live, or pregame"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="YYYY-MM-DD"),
    min_prob: Optional[float] = typer.Option(
        None,
        "--min-prob",
        help="Override min win probability (defaults: 60% Moneyline, 55.5% Spread/Total/Props)",
    ),
    min_edge: Optional[float] = typer.Option(
        None, "--min-edge", help="Override minimum net edge after DKeX fees (default 2.5%)"
    ),
    fee: float = typer.Option(0.015, "--fee", help="Per-contract fee ($0.015 default, 0.0 for Promo)"),
    bankroll: Optional[float] = typer.Option(None, "--bankroll", help="Override bankroll ($)"),
    fresh: bool = typer.Option(False, "--fresh", "-f", help="Pull fresh live scoreboards, box scores, & odds now"),
    live_props: bool = typer.Option(False, "--live-props", help="Fetch live sportsbook player prop lines (uses API credits)"),
    show_all: bool = typer.Option(False, "--all-markets", help="Include PASS markets"),
) -> None:
    """Generate ranked high-confidence DraftKings Predictions contracts (pre-game and/or live in-play)."""
    hub = ModelHub()
    eff_bk = bankroll if bankroll is not None else hub.settings.bankroll
    picks = hub.generate_picks(
        sport=sport,
        market_filter=market,
        mode=mode,
        target_date=date,
        min_win_prob=min_prob,
        min_net_edge=min_edge,
        fee_per_contract=fee,
        bankroll=eff_bk,
        include_passes=show_all,
        fresh=fresh,
        live_props=live_props,
    )

    if not picks:
        mp_disp = min_prob if min_prob is not None else hub.settings.min_win_prob
        me_disp = min_edge if min_edge is not None else hub.settings.min_net_edge
        console.print(
            f"[yellow]No contracts matched mode='{mode}' clearing threshold (ML >= {mp_disp*100:.0f}%, 50/50 >= {min(mp_disp, 0.555)*100:.1f}%, Edge >= {me_disp*100:+.1f}%). "
            "Try --fresh, --mode all, or --all-markets to inspect live lines.[/yellow]"
        )
        return

    _render_picks_table(
        picks,
        fee=fee,
        title_prefix=f"DKeX High-Confidence Plays • Mode={mode.upper()} • Bankroll=${eff_bk:,.2f}",
    )
    console.print(_format_quota_line(hub.odds.get_quota_status()))


@app.command("live")
def live_cmd(
    sport: str = typer.Option("all", "--sport", "-s", help="nfl, nba, or all"),
    market: str = typer.Option("all", "--market", "-m", help="all, game, or props"),
    fee: float = typer.Option(0.015, "--fee", help="Per-contract DKeX fee in dollars"),
    live_props: bool = typer.Option(False, "--live-props", help="Also query live sportsbook prop odds"),
    watch: int = typer.Option(0, "--watch", "-w", help="Optional auto-refresh interval in seconds (e.g. 60)"),
    show_all: bool = typer.Option(False, "--all-markets", help="Include PASS markets"),
) -> None:
    """Fast live in-play scanner + open-position cash-out monitor (skips slow historical CSV sync)."""
    hub = ModelHub()

    def _run_once() -> None:
        with console.status("[bold red]Refreshing live scoreboards, box scores, & DraftKings/Sharp lines...[/bold red]"):
            feed_info = hub.refresh_live_feeds(sport=sport)
            picks = hub.generate_picks(
                sport=sport,
                market_filter=market,
                mode="live",
                fee_per_contract=fee,
                include_passes=show_all,
                fresh=False,  # just refreshed above
                live_props=live_props,
            )
            open_pos = hub.evaluate_open_positions(fresh=False)

        quota = feed_info.get("odds_quota", {})
        in_play_total = sum(
            v.get("live_in_play_events", 0)
            for k, v in feed_info.items()
            if isinstance(v, dict) and "live_in_play_events" in v
        )
        console.print(
            Panel(
                f"[bold]Active In-Play Games:[/bold] {in_play_total}  |  {_format_quota_line(quota)}",
                title="Live In-Play Feed Status",
                border_style="red" if in_play_total > 0 else "cyan",
            )
        )

        if picks:
            _render_picks_table(picks, fee=fee, title_prefix="LIVE IN-PLAY High-Confidence Plays")
        else:
            console.print(
                "[yellow]No live in-play games currently have qualifying high-confidence edges "
                "(or no NFL/NBA games are in progress right this minute).[/yellow]"
            )

        if open_pos:
            pos_tbl = Table(title="Open Ledger Positions — Live Hold vs Cash-Out Monitor", header_style="bold yellow")
            pos_tbl.add_column("ID", justify="right")
            pos_tbl.add_column("State / Score")
            pos_tbl.add_column("Position")
            pos_tbl.add_column("Entry ¢", justify="right")
            pos_tbl.add_column("Live Win %", justify="right")
            pos_tbl.add_column("Action Guidance")
            for op in open_pos:
                st = (
                    f"[bold red]LIVE[/bold red] {op.get('score_display', '')}"
                    if op.get("game_state") == "LIVE"
                    else str(op.get("game_state", "PRE"))
                )
                pos_tbl.add_row(
                    str(op["id"]),
                    st,
                    f"{op['contracts']}x {op['side']} {op['selection_label']}",
                    f"{op['entry_price']*100:.1f}¢",
                    f"{op['live_model_prob']*100:.1f}%",
                    op["live_signal"],
                )
            console.print(pos_tbl)

    if watch <= 0:
        _run_once()
        return

    interval = max(30, watch)
    try:
        while True:
            console.clear()
            _run_once()
            console.print(f"[dim]Watching live lines every {interval}s — Press Ctrl+C to stop.[/dim]")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[cyan]Stopped live watcher.[/cyan]")


@app.command("check")
def check_cmd(
    prob: float = typer.Option(..., "--prob", "-p", help="True/model win probability (e.g. 0.68)"),
    price: float = typer.Option(..., "--price", "-c", help="DK contract price ($0.61 or 61 for 61¢)"),
    side: str = typer.Option("YES", "--side", help="YES or NO"),
    fee: float = typer.Option(0.015, "--fee", help="Per-contract fee in dollars (e.g. 0.015)"),
    bankroll: Optional[float] = typer.Option(None, "--bankroll", "-b", help="Bankroll in dollars"),
    live: bool = typer.Option(False, "--live", help="Apply live in-play Kelly & latency haircut"),
) -> None:
    """Evaluate a live DraftKings Predictions contract ask price and calculate Quarter-Kelly size."""
    settings = get_settings()
    px = price / 100.0 if price > 1.0 else price
    pr = prob / 100.0 if prob > 1.0 else prob
    bk = bankroll if bankroll is not None else settings.bankroll

    ev = evaluate_single_side(
        side="YES" if side.upper() == "YES" else "NO",
        side_true_prob=pr,
        contract_price=px,
        fee_per_contract=fee,
        bankroll=bk,
        kelly_multiplier=settings.kelly_fraction * (0.80 if live else 1.0),
        max_bet_pct=settings.max_single_bet_pct,
        min_win_prob=settings.min_win_prob,
        min_net_edge=settings.min_net_edge,
        uncertainty=0.15 if live else 0.12,
    )

    verdict_color = "bright_green" if ev.confidence_tier in ("A+", "A") else ("yellow" if ev.confidence_tier == "B" else "red")
    verdict = f"BUY {ev.side} ({ev.confidence_tier})" if ev.confidence_tier != "PASS" else "PASS — DO NOT BUY AT THIS PRICE"
    mode_tag = " [LIVE IN-PLAY SIZING]" if live else ""

    body = (
        f"[bold {verdict_color}]{verdict}{mode_tag}[/bold {verdict_color}]  (Score: {ev.confidence_score}/100)\n\n"
        f"• Contract Price: [bold]{ev.contract_price*100:.1f}¢[/bold] ({ev.american_equivalent:+d} American) + "
        f"Fee [bold]{ev.fee_per_contract*100:.2f}¢[/bold] = All-In Cost [bold]${ev.all_in_cost:.3f}[/bold]\n"
        f"• Break-Even Win Rate: [bold]{ev.breakeven_prob*100:.2f}%[/bold]  |  Model Win Rate: [bold]{ev.model_prob*100:.2f}%[/bold]\n"
        f"• Net Edge (After Fee): [bold]{ev.net_edge*100:+.2f}%[/bold]  |  Expected ROI: [bold]{ev.roi_pct*100:+.2f}%[/bold] (${ev.ev_per_contract:+.3f}/contract)\n"
        f"• Max Limit Price to Pay: [bold]{ev.fair_entry_max_price*100:.1f}¢[/bold]  |  Fair Mid-Game Exit Target: [bold]{ev.fair_midgame_exit_price*100:.1f}¢[/bold]\n"
        f"• {'In-Play ' if live else ''}Quarter-Kelly Sizing (${bk:,.2f} Bankroll): [bold]{ev.recommended_contracts} contracts[/bold] (${ev.recommended_outlay:.2f} total outlay)"
    )
    console.print(Panel(body, title="DraftKings Predictions Contract Analyzer", border_style=verdict_color))


@app.command("dashboard")
def dashboard_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Launch the interactive Local Web Dashboard."""
    console.print(
        f"[bold cyan]Starting DraftKings Predictions Dashboard at http://{host}:{port}[/bold cyan]"
    )
    uvicorn.run("src.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()

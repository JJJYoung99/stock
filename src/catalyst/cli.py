"""`catalyst` command-line interface.

Subcommands:
    catalyst score    --market kr --ticker 005930 --date 2026-04-30
    catalyst backtest --market sample-kr
    catalyst events   --market sample-kr
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from catalyst.backtest import BacktestConfig, Backtester
from catalyst.config import load_rules
from catalyst.events.csv_store import CsvEventStore
from catalyst.markets import get_adapter
from catalyst.report import (
    render_backtest_metrics,
    render_score_report,
    report_to_json,
)
from catalyst.scoring import Scorer
from catalyst.types import Market

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "sample"


def _parse_market(s: str) -> tuple[str, Market]:
    """Accept 'kr' / 'us' / 'sample-kr' / 'sample-us'. Returns (adapter_kind, market_enum)."""
    s = s.lower()
    if s.startswith("sample-"):
        return "sample", Market(s.split("-", 1)[1].upper())
    return s, Market(s.upper())


def _build_pipeline(market_arg: str, store_path: Path | None):
    kind, mkt = _parse_market(market_arg)
    if kind == "sample":
        adapter = get_adapter("sample", market=mkt)
    else:
        adapter = get_adapter(kind)
    sp = store_path or (_DEFAULT_DATA / f"{mkt.value.lower()}_events.csv")
    store = CsvEventStore(sp)
    cfg = load_rules()
    scorer = Scorer(adapter, store, cfg)
    return adapter, store, scorer, mkt


def _cmd_score(args: argparse.Namespace) -> int:
    _, _, scorer, mkt = _build_pipeline(args.market, Path(args.events) if args.events else None)
    as_of = date.fromisoformat(args.date)
    report = scorer.score(args.ticker, mkt, as_of)
    if args.json:
        print(report_to_json(report))
    else:
        print(render_score_report(report))
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    adapter, store, scorer, _ = _build_pipeline(args.market, Path(args.events) if args.events else None)
    bt = Backtester(
        adapter, store, scorer,
        config=BacktestConfig(
            entry_threshold=args.entry_threshold,
            min_prob_profit=args.min_prob_profit,
            min_analogs=args.min_analogs,
            cost_bps=args.cost_bps,
            holding_horizon=args.horizon,
        ),
    )
    result = bt.run()
    metrics = result.metrics()
    print(render_backtest_metrics(metrics))
    if args.trades_csv:
        result.trades_dataframe().to_csv(args.trades_csv, index=False)
        print(f"\nwrote trades -> {args.trades_csv}")
    if args.equity_csv:
        result.equity_curve.to_csv(args.equity_csv, header=["equity"])
        print(f"wrote equity -> {args.equity_csv}")
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    _, store, _, mkt = _build_pipeline(args.market, Path(args.events) if args.events else None)
    evs = store.query(market=mkt)
    print(f"# {len(evs)} stored events for {mkt.value}")
    for ev in sorted(evs, key=lambda e: e.occurred_at):
        print(f"{ev.occurred_at}  {ev.ticker:<8} {ev.event_type.value:<22} mag={ev.magnitude:+.4f}  src={ev.source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="catalyst", description="Event-driven research framework.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common_market_args = [
        (("--market",), {"required": True, "help": "kr / us / sample-kr / sample-us"}),
        (("--events",), {"default": None, "help": "path to events CSV (defaults to data/sample/<market>_events.csv)"}),
    ]

    sp_score = sub.add_parser("score", help="Score a (ticker, date).")
    for a, kw in common_market_args:
        sp_score.add_argument(*a, **kw)
    sp_score.add_argument("--ticker", required=True)
    sp_score.add_argument("--date", required=True, help="YYYY-MM-DD")
    sp_score.add_argument("--json", action="store_true")
    sp_score.set_defaults(func=_cmd_score)

    sp_bt = sub.add_parser("backtest", help="Walk-forward backtest over the event store.")
    for a, kw in common_market_args:
        sp_bt.add_argument(*a, **kw)
    sp_bt.add_argument("--entry-threshold", type=float, default=0.5)
    sp_bt.add_argument("--min-prob-profit", type=float, default=0.50)
    sp_bt.add_argument("--min-analogs", type=int, default=5)
    sp_bt.add_argument("--cost-bps", type=float, default=20.0)
    sp_bt.add_argument("--horizon", type=int, default=None, help="override per-event horizon_days")
    sp_bt.add_argument("--trades-csv", default=None)
    sp_bt.add_argument("--equity-csv", default=None)
    sp_bt.set_defaults(func=_cmd_backtest)

    sp_ev = sub.add_parser("events", help="List stored events.")
    for a, kw in common_market_args:
        sp_ev.add_argument(*a, **kw)
    sp_ev.set_defaults(func=_cmd_events)

    ns = p.parse_args(argv)
    return int(ns.func(ns) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

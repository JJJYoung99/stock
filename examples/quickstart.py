"""End-to-end demo using the bundled sample data.

Run from the repo root:

    python scripts/gen_sample_data.py    # one-time
    python examples/quickstart.py
"""
from __future__ import annotations

from datetime import date

from catalyst.backtest import BacktestConfig, Backtester
from catalyst.events.csv_store import CsvEventStore
from catalyst.events.detectors import DartContractDetector
from catalyst.markets import get_adapter
from catalyst.report import render_backtest_metrics, render_score_report
from catalyst.scoring import Scorer
from catalyst.types import Market

DATA = "data/sample"


def demo_event_detection():
    print("--- 1. detector: DART single-contract disclosure ----------------")
    payload = {
        "report_nm": "단일판매·공급계약체결",
        "stock_code": "005930",
        "rcept_dt": "2026-04-15",
        "contract_amount": 5.0e11,   # 500 bn KRW
        "market_cap": 5.0e13,        # 50 trn KRW
    }
    events = DartContractDetector().detect(payload)
    for e in events:
        print(f"  -> {e.event_type.value} {e.ticker} mag={e.magnitude:+.4f}")


def demo_score():
    print("\n--- 2. score one event against historical analogs ---------------")
    adapter = get_adapter("sample", market=Market.KR)
    store = CsvEventStore(f"{DATA}/kr_events.csv")
    scorer = Scorer(adapter, store)
    # Pick a real event from the bundled sample CSV.
    target_date = date(2025, 9, 22)
    report = scorer.score(ticker="005930", market=Market.KR, as_of=target_date)
    print(render_score_report(report))


def demo_backtest():
    print("\n--- 3. walk-forward backtest -----------------------------------")
    adapter = get_adapter("sample", market=Market.KR)
    store = CsvEventStore(f"{DATA}/kr_events.csv")
    bt = Backtester(
        adapter, store, Scorer(adapter, store),
        BacktestConfig(entry_threshold=0.3, min_prob_profit=0.45, min_analogs=3),
    )
    result = bt.run()
    print(render_backtest_metrics(result.metrics()))


if __name__ == "__main__":
    demo_event_detection()
    demo_score()
    demo_backtest()

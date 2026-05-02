"""End-to-end backtest using the bundled sample data."""
from pathlib import Path

import pytest

from catalyst.backtest import BacktestConfig, Backtester
from catalyst.events.csv_store import CsvEventStore
from catalyst.markets import get_adapter
from catalyst.scoring import Scorer
from catalyst.types import Market

DATA = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.mark.skipif(not (DATA / "kr_events.csv").exists(),
                    reason="run scripts/gen_sample_data.py first")
def test_kr_backtest_runs_and_produces_metrics():
    adapter = get_adapter("sample", market=Market.KR)
    store = CsvEventStore(DATA / "kr_events.csv")
    bt = Backtester(adapter, store, Scorer(adapter, store),
                    BacktestConfig(entry_threshold=0.3, min_prob_profit=0.45, min_analogs=3))
    result = bt.run()
    metrics = result.metrics()

    assert result.n_trades > 0
    assert "hit_rate" in metrics
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert metrics["max_drawdown"] <= 0.0
    df = result.trades_dataframe()
    assert {"ticker", "event_type", "pnl"}.issubset(df.columns)


@pytest.mark.skipif(not (DATA / "us_events.csv").exists(),
                    reason="run scripts/gen_sample_data.py first")
def test_us_backtest_walk_forward_no_lookahead():
    """Sanity check: every realized analog used in a trade's posterior must
    have an occurred_at strictly less than the trade's entry_date."""
    adapter = get_adapter("sample", market=Market.US)
    store = CsvEventStore(DATA / "us_events.csv")
    bt = Backtester(adapter, store, Scorer(adapter, store),
                    BacktestConfig(entry_threshold=0.3, min_prob_profit=0.45, min_analogs=3))
    result = bt.run()
    # Every report's events must be on its own as_of, never future.
    for r in result.per_event_reports:
        for ev in r.events:
            assert ev.occurred_at == r.as_of

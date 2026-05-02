from datetime import date
from pathlib import Path

from catalyst.events.csv_store import CsvEventStore
from catalyst.types import Event, EventType, Market


def test_append_and_query_roundtrip(tmp_path: Path):
    store = CsvEventStore(tmp_path / "events.csv")
    e1 = Event(
        ticker="005930", market=Market.KR, event_type=EventType.CONTRACT_WIN,
        occurred_at=date(2026, 1, 5), magnitude=0.10, source="test",
    )
    e2 = Event(
        ticker="000660", market=Market.KR, event_type=EventType.CAPITAL_RAISE,
        occurred_at=date(2026, 2, 1), magnitude=0.05, source="test",
    )
    store.append([e1, e2])

    reopened = CsvEventStore(tmp_path / "events.csv")
    all_evs = reopened.all()
    assert len(all_evs) == 2
    # Leading-zero ticker preserved.
    assert any(ev.ticker == "005930" for ev in all_evs)


def test_query_filters(tmp_path: Path):
    store = CsvEventStore(tmp_path / "events.csv")
    store.append([
        Event(ticker="A", market=Market.US, event_type=EventType.EARNINGS_BEAT,
              occurred_at=date(2026, 1, 5), magnitude=0.1, source="t"),
        Event(ticker="B", market=Market.US, event_type=EventType.EARNINGS_BEAT,
              occurred_at=date(2026, 2, 5), magnitude=0.2, source="t"),
        Event(ticker="A", market=Market.US, event_type=EventType.EARNINGS_MISS,
              occurred_at=date(2026, 3, 5), magnitude=0.1, source="t"),
    ])
    beats = store.query(market=Market.US, event_type=EventType.EARNINGS_BEAT)
    assert len(beats) == 2
    before_feb = store.query(before=date(2026, 2, 1))
    assert len(before_feb) == 1
    a_only = store.query(ticker="A")
    assert {e.ticker for e in a_only} == {"A"}

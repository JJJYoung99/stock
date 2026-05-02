# Real-event CSVs

Unlike `data/sample/`, files in this directory describe **real, publicly
reported corporate catalysts**. Magnitudes are computed from the source
articles where the underlying numbers were disclosed; placeholders are
flagged in the `confidence` column and the `raw_json.note` field.

## Files

| File | Window | Notes |
|---|---|---|
| `us_events_2026-05-01.csv` | 2026-04-29 .. 2026-04-30 | Big-tech + KLAC + TWLO. See `reports/CURRENT_2026-05-02.md` for the source articles. |

## Caveats

These CSVs intentionally **do not** contain a long history. They exist
to demonstrate how real catalysts feed into the framework. Posterior
P(profit) and VaR will be prior-dominated until you ingest 2-3 years of
historical events into the same store (use OpenDartReader for KR and
finnhub/SEC for US — see the project README's "Going live with real data"
section).

## Updating

To add a new event:

```python
from datetime import date
from catalyst.events.csv_store import CsvEventStore
from catalyst.types import Event, EventType, Market

store = CsvEventStore("data/real/us_events_2026-05-01.csv")
store.append([
    Event(
        ticker="NVDA",
        market=Market.US,
        event_type=EventType.EARNINGS_BEAT,
        occurred_at=date(2026, 5, 28),
        magnitude=0.10,           # surprise %
        source="finnhub",
        confidence=1.0,
        raw={"eps_actual": 1.10, "eps_consensus": 1.00},
    ),
])
```

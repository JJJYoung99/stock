# catalyst — event-driven research framework

When a stock pops on news — a contract win, an earnings beat, an analyst
target hike — the question is never "is this good news?" but "given this
*kind* of news at this *magnitude*, what does history say happens next?"

`catalyst` turns that question into a numeric pipeline:

```
            news / disclosure / earnings call
                          │
                          ▼
                  ┌─────────────────┐
                  │  EventDetector  │   parses raw payload → typed Event(s)
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │   EventStore    │   append-only history of past events
                  └────────┬────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Scorer:                             │
        │    rule_score   = weight × magnitude │
        │    prob_profit  = Beta posterior     │   from analog returns
        │    expected_r   = mean of analogs    │   measured via MarketAdapter
        │    VaR / CVaR   = empirical 5% tail  │
        └──────────────────────────────────────┘
                           ▼
                ScoreReport / BacktestResult
```

The framework is a **research and backtesting tool**, not an execution
engine. It does not place orders. It tells you, for any (ticker, date),
the historical edge and risk associated with the catalyst that just fired.

## What's inside

| Module | Role |
| --- | --- |
| `catalyst.types` | `Event`, `EventType`, `Score`, `ScoreReport` dataclasses |
| `catalyst.config` | YAML rule-config loader (`configs/event_rules.yaml`) |
| `catalyst.markets` | Market adapters: `sample` (offline CSV), `kr`, `us` |
| `catalyst.events` | Detectors (DART, SEC, earnings, analyst, news headlines) + `CsvEventStore` |
| `catalyst.scoring` | Rule scoring + Bayesian posterior + empirical VaR/CVaR |
| `catalyst.backtest` | Walk-forward backtester, no look-ahead |
| `catalyst.report` | Pretty-printers and JSON serialization |
| `catalyst.cli` | `catalyst score / backtest / events` |

## Install

```bash
pip install -e .                  # core (offline-only, sample adapter)
pip install -e .[kr]              # + FinanceDataReader, pykrx, OpenDartReader
pip install -e .[us]              # + yfinance, finnhub-python, sec-edgar-downloader
pip install -e .[nlp]             # + transformers, torch (for finBERT / KR-FinBert)
pip install -e .[dev]             # + pytest
```

## Quickstart (offline, no network)

```bash
python scripts/gen_sample_data.py
python examples/quickstart.py
```

Or via the CLI:

```bash
# list synthetic events
catalyst events --market sample-kr | head

# score a (ticker, date)
catalyst score --market sample-kr --ticker 005930 --date 2025-09-22

# walk-forward backtest with custom thresholds
catalyst backtest --market sample-kr \
    --entry-threshold 0.3 --min-prob-profit 0.45 --min-analogs 3 \
    --trades-csv /tmp/trades.csv --equity-csv /tmp/equity.csv
```

## Going live with real data

The Korean and US adapters wrap third-party libraries — they do **not**
ship event detectors that hit a network. Plug your own ingestion pipeline
through the existing detectors:

```python
from catalyst.markets import get_adapter
from catalyst.events.csv_store import CsvEventStore
from catalyst.events.detectors import DartContractDetector
from catalyst.scoring import Scorer

# 1. Pull DART filings yourself (OpenDartReader handles auth + paging).
import OpenDartReader
dart = OpenDartReader(api_key="...")
filings = dart.list("005930", start="2026-01-01")

# 2. Convert filings to typed Events.
detector = DartContractDetector()
events = []
for _, row in filings.iterrows():
    events.extend(detector.detect({
        "report_nm": row["report_nm"],
        "stock_code": row["stock_code"],
        "rcept_dt":   row["rcept_dt"],
        "contract_amount": ...,   # parse from filing detail
        "market_cap":      ...,
    }))

# 3. Persist + score.
store = CsvEventStore("data/kr_events.csv")
store.append(events)

adapter = get_adapter("kr")
scorer = Scorer(adapter, store)
report = scorer.score("005930", market=adapter.market, as_of=date.today())
```

## Why these algorithm choices

- **Rule-based first, ML-augmented later.** Sell-side and DART filings are
  structured enough that a tiny weight table beats a half-trained classifier.
  The YAML config doubles as an audit trail for *why* a stock scored what
  it did — easy to challenge, easy to recalibrate.
- **Bayesian posterior, not raw frequency.** With 3-5 historical analogs,
  raw P(win) is dominated by noise. Beta(2,2) prior pulls toward 0.5 and
  the report flags "low analog count" so you don't act on a posterior built
  on three trades.
- **Empirical VaR/CVaR over the same analogs.** The risk number is on the
  *same distribution* as the probability number, so they can't disagree
  about regime. No parametric Gaussian assumption.
- **Walk-forward with strict `before=as_of` queries.** The EventStore
  filter is the single chokepoint that prevents look-ahead bias; if you
  add a new analog selection method, route it through the same query.

## What this framework does *not* try to do

- Position sizing / portfolio construction. Plug `skfolio` or `riskfolio-lib`.
- Order routing / live trading. Use `backtrader` or a brokerage SDK.
- News scraping. Use `KoreaNewsCrawler` (KR) / `finnhub-python` (US).
- NLP-grade event extraction. The headline detector is a keyword fallback;
  for production use a fine-tuned `KR-FinBert-SC` or `ProsusAI/finBERT`.

## References that informed the design

- **Event-study mechanics** — `aniketnmishra/PEAD`, `gen-li/Replicate_PEAD`.
- **Bayesian return-distribution analysis** — chapter 10 of
  `stefan-jansen/machine-learning-for-trading`.
- **KR data plumbing** — `FinanceData/FinanceDataReader`, `sharebook-kr/pykrx`,
  `FinanceData/OpenDartReader`.
- **US data plumbing** — `Finnhub-Stock-API/finnhub-python`,
  `ranaroussi/yfinance`, `jadchaar/sec-edgar-downloader`.
- **Korean financial NLP** — SNU NLP `KR-FinBert-SC`, `lumyjuwon/KoreaNewsCrawler`.
- **Risk metrics** — `ibaris/VaR`, `skfolio`.

## Tests

```bash
python -m pytest -q
```

29 tests, no network dependency.

## License

MIT.

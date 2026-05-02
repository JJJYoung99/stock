"""Generate synthetic OHLCV + event CSVs for offline testing.

Produces, under data/sample/:
    kr_prices.csv, kr_benchmark.csv, kr_events.csv
    us_prices.csv, us_benchmark.csv, us_events.csv

The generator deliberately injects post-event drift so the framework's
Bayesian posterior + backtest can be sanity-checked end-to-end. A fixed
seed makes the output reproducible.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

START = date(2022, 1, 3)
END = date(2025, 12, 31)
SEED = 7

KR_TICKERS = ["005930", "000660", "035420", "207940", "068270",
              "035720", "051910", "006400", "012450", "263750"]
US_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META",
              "TSLA", "AVGO", "CRWD", "PLTR", "ANET"]

# (event_type, drift_mean, drift_sd, horizon_days, magnitude_distribution)
EVENT_TEMPLATES = [
    ("CONTRACT_WIN",      +0.018, 0.05, 20, ("uniform", 0.02, 0.30)),
    ("EARNINGS_BEAT",     +0.022, 0.06, 30, ("uniform", 0.05, 0.40)),
    ("EARNINGS_MISS",     -0.020, 0.06, 30, ("uniform", 0.05, 0.40)),
    ("GUIDANCE_RAISE",    +0.025, 0.05, 40, ("uniform", 0.03, 0.20)),
    ("ANALYST_TARGET_UP", +0.012, 0.04, 15, ("uniform", 0.05, 0.25)),
    ("ANALYST_RATING_UP", +0.008, 0.04, 15, ("uniform", 1.0, 1.0)),
    ("BUYBACK",           +0.010, 0.04, 30, ("uniform", 0.005, 0.05)),
    ("INSIDER_BUY",       +0.005, 0.03, 30, ("uniform", 0.001, 0.02)),
    ("CAPITAL_RAISE",     -0.030, 0.06, 20, ("uniform", 0.02, 0.20)),
    ("CONVERTIBLE_ISSUE", -0.010, 0.04, 20, ("uniform", 0.02, 0.10)),
    ("REGULATORY_PROBE",  -0.040, 0.08, 20, ("uniform", 1.0, 1.0)),
]


def _trading_days(start: date, end: date) -> list[pd.Timestamp]:
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    return list(days)


def _gen_price_path(rng: np.random.Generator, n: int, start_px: float,
                    daily_drift: float = 0.0002, daily_sd: float = 0.018) -> np.ndarray:
    rets = rng.normal(daily_drift, daily_sd, size=n)
    px = np.empty(n, dtype=float)
    px[0] = start_px
    for i in range(1, n):
        px[i] = px[i-1] * (1 + rets[i])
    return px


def _inject_event_drifts(rng: np.random.Generator, prices: np.ndarray,
                         dates: list[pd.Timestamp], events_for_ticker: list[dict]) -> None:
    """Mutate prices in-place by adding event-specific drift over the
    horizon_days following each event."""
    date_to_idx = {d: i for i, d in enumerate(dates)}
    for ev in events_for_ticker:
        idx = date_to_idx.get(pd.Timestamp(ev["occurred_at"]))
        if idx is None:
            continue
        horizon = ev["_horizon_days"]
        drift = ev["_drift_mean"]
        sd = ev["_drift_sd"]
        magnitude_mult = min(ev["magnitude"] / 0.10, 3.0) if ev["magnitude"] < 1.0 else 1.0
        # Scale daily drift over the post-event window.
        daily_extra = drift / max(horizon, 1)
        end = min(idx + horizon, len(prices) - 1)
        for j in range(idx + 1, end + 1):
            shock = rng.normal(daily_extra * magnitude_mult, sd / np.sqrt(horizon))
            prices[j] *= (1 + shock)


def _draw_magnitude(rng: np.random.Generator, dist: tuple) -> float:
    kind, lo, hi = dist
    if kind == "uniform":
        return float(rng.uniform(lo, hi))
    raise ValueError(f"unknown distribution: {kind}")


def _build_events(rng: np.random.Generator, market: str, tickers: list[str],
                  dates: list[pd.Timestamp]) -> list[dict]:
    events: list[dict] = []
    for ticker in tickers:
        # Each ticker gets ~30 events over the 4-year window, sampled with
        # replacement from the templates.
        n = rng.integers(25, 40)
        chosen_dates = rng.choice(dates[10:-50], size=int(n), replace=False)
        for d in sorted(chosen_dates):
            tpl = EVENT_TEMPLATES[int(rng.integers(0, len(EVENT_TEMPLATES)))]
            et, drift_mu, drift_sd, horizon, mag_dist = tpl
            mag = _draw_magnitude(rng, mag_dist)
            events.append({
                "market": market,
                "ticker": ticker,
                "event_type": et,
                "occurred_at": pd.Timestamp(d).date(),
                "magnitude": round(mag, 6),
                "source": "synthetic",
                "confidence": 1.0,
                "raw_json": json.dumps({"synthetic": True, "template": et}, ensure_ascii=False),
                # Internal-only fields used by the price injector.
                "_drift_mean": drift_mu,
                "_drift_sd": drift_sd,
                "_horizon_days": horizon,
            })
    return events


def _ohlcv_from_close(rng: np.random.Generator, close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    volume = (rng.lognormal(13, 0.5, size=n)).astype(int)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _gen_market(market: str, tickers: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = _trading_days(START, END)
    n = len(dates)

    raw_events = _build_events(rng, market, tickers, dates)

    rows = []
    for ticker in tickers:
        start_px = float(rng.uniform(20_000, 200_000)) if market == "KR" else float(rng.uniform(50, 500))
        close = _gen_price_path(rng, n, start_px)
        ev_for_t = [e for e in raw_events if e["ticker"] == ticker]
        _inject_event_drifts(rng, close, dates, ev_for_t)
        ohlcv = _ohlcv_from_close(rng, close)
        ohlcv.insert(0, "date", dates)
        ohlcv.insert(0, "ticker", ticker)
        ohlcv.insert(0, "market", market)
        rows.append(ohlcv)
    prices_df = pd.concat(rows, ignore_index=True)

    # Benchmark — a simpler aggregate path.
    bench_close = _gen_price_path(rng, n, 1000.0, daily_drift=0.0003, daily_sd=0.010)
    bench = _ohlcv_from_close(rng, bench_close)
    bench.insert(0, "date", dates)
    bench.insert(0, "market", market)

    # Strip private fields from the events frame.
    events_df = pd.DataFrame([{
        "market": e["market"], "ticker": e["ticker"], "event_type": e["event_type"],
        "occurred_at": e["occurred_at"], "magnitude": e["magnitude"],
        "source": e["source"], "confidence": e["confidence"], "raw_json": e["raw_json"],
    } for e in raw_events])
    return prices_df, bench, events_df


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    for market, tickers, seed in [("KR", KR_TICKERS, SEED), ("US", US_TICKERS, SEED + 1)]:
        prices, bench, events = _gen_market(market, tickers, seed)
        prices.to_csv(OUT / f"{market.lower()}_prices.csv", index=False)
        bench.to_csv(OUT / f"{market.lower()}_benchmark.csv", index=False)
        events.sort_values(["occurred_at", "ticker"]).to_csv(OUT / f"{market.lower()}_events.csv", index=False)
        print(f"{market}: prices={len(prices)} bench={len(bench)} events={len(events)} -> {OUT}")


if __name__ == "__main__":
    main()

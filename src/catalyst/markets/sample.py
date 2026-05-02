"""Offline CSV adapter used for tests and the bundled examples.

Reads `data/sample/{market}_prices.csv` with columns:
    market,ticker,date,open,high,low,close,volume
and `data/sample/{market}_benchmark.csv` with columns:
    market,date,open,high,low,close,volume

Both files ship with the repo so the framework runs end-to-end without
any network dependency.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from catalyst.markets.base import MarketAdapter
from catalyst.types import Market


class SampleAdapter(MarketAdapter):
    def __init__(self, market: Market | str = Market.KR, data_dir: str | Path | None = None):
        self.market = Market(market) if not isinstance(market, Market) else market
        self.data_dir = Path(data_dir) if data_dir else (Path(__file__).resolve().parents[3] / "data" / "sample")
        self._prices_cache: dict[str, pd.DataFrame] = {}
        self._bench_cache: pd.DataFrame | None = None

    def _load_prices(self) -> pd.DataFrame:
        if self._prices_cache:
            return self._prices_cache["__all__"]
        path = self.data_dir / f"{self.market.value.lower()}_prices.csv"
        if not path.exists():
            raise FileNotFoundError(f"sample price file missing: {path}")
        df = pd.read_csv(path, parse_dates=["date"], dtype={"ticker": str, "market": str})
        df = df[df["market"].str.upper() == self.market.value]
        self._prices_cache["__all__"] = df
        return df

    def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = self._load_prices()
        df = df[df["ticker"] == ticker]
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        out = df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        return out

    def benchmark(self, start: date, end: date) -> pd.DataFrame:
        if self._bench_cache is None:
            path = self.data_dir / f"{self.market.value.lower()}_benchmark.csv"
            if not path.exists():
                raise FileNotFoundError(f"sample benchmark file missing: {path}")
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[df["market"].str.upper() == self.market.value]
            self._bench_cache = df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        bench = self._bench_cache
        return bench.loc[(bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))]

    def universe(self) -> list[str]:
        df = self._load_prices()
        return sorted(df["ticker"].unique().tolist())

"""US market adapter — Yahoo Finance OHLCV, S&P 500 (SPY) as benchmark.

Optional deps: `pip install catalyst[us]`.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from catalyst.markets.base import MarketAdapter
from catalyst.types import Market

_BENCHMARK_TICKER = "SPY"


class USAdapter(MarketAdapter):
    market = Market.US

    def __init__(self):
        try:
            import yfinance as yf  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "USAdapter requires yfinance. Install with `pip install catalyst[us]`."
            ) from e
        import yfinance as yf
        self._yf = yf

    def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = self._yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return df
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def benchmark(self, start: date, end: date) -> pd.DataFrame:
        return self.prices(_BENCHMARK_TICKER, start, end)

    def universe(self) -> list[str]:
        # yfinance does not expose a free universe listing. Users should
        # provide their own watchlist (e.g. S&P 500 from Wikipedia).
        raise NotImplementedError(
            "USAdapter.universe() requires an external watchlist. "
            "Pass a list of tickers explicitly to backtest/score."
        )

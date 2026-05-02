"""Korean market adapter — KRX OHLCV via FinanceDataReader, KOSPI as benchmark.

Optional deps: `pip install catalyst[kr]`.

The Korean event sources (DART corporate disclosures) live in
`catalyst.events.kr_dart`, not here. This adapter only handles prices.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from catalyst.markets.base import MarketAdapter
from catalyst.types import Market

_BENCHMARK_TICKER = "KS11"  # KOSPI


class KoreaAdapter(MarketAdapter):
    market = Market.KR

    def __init__(self):
        try:
            import FinanceDataReader as fdr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "KoreaAdapter requires FinanceDataReader. Install with `pip install catalyst[kr]`."
            ) from e
        import FinanceDataReader as fdr
        self._fdr = fdr

    def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = self._fdr.DataReader(ticker, start, end)
        # FDR returns columns Open, High, Low, Close, Volume, Change.
        df = df.rename(columns=str.lower)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def benchmark(self, start: date, end: date) -> pd.DataFrame:
        return self.prices(_BENCHMARK_TICKER, start, end)

    def universe(self) -> list[str]:
        # KOSPI + KOSDAQ listed tickers. FDR returns a DataFrame with column 'Code'.
        kospi = self._fdr.StockListing("KOSPI")["Code"].tolist()
        kosdaq = self._fdr.StockListing("KOSDAQ")["Code"].tolist()
        return sorted(set(kospi + kosdaq))

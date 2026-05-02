"""Market adapter protocol.

A MarketAdapter is responsible only for raw OHLCV + (optionally) the
benchmark series for excess-return computation. Event ingestion is the
job of `catalyst.events`, not the market adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from catalyst.types import Market


class MarketAdapter(ABC):
    market: Market

    @abstractmethod
    def prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return a DataFrame indexed by DatetimeIndex with columns
        ['open','high','low','close','volume']. Must be inclusive on both ends."""

    @abstractmethod
    def benchmark(self, start: date, end: date) -> pd.DataFrame:
        """Index OHLCV (KOSPI for KR, S&P 500 for US). Same shape as `prices`."""

    @abstractmethod
    def universe(self) -> list[str]:
        """List of tradable tickers in this market for the current snapshot."""

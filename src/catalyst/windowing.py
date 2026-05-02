"""Event-window math: forward returns, drawdowns, analog matching.

Operates on pandas DataFrames indexed by trading-day date. The framework
deliberately works in trading days, not calendar days, so a "20-day horizon"
means 20 closes after the event close.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowReturns:
    horizon_days: int
    cum_return: float
    max_return: float
    min_return: float
    max_drawdown: float
    n_bars: int


def forward_window(prices: pd.DataFrame, event_date: date, horizon_days: int) -> pd.DataFrame:
    """Return the slice of `prices` covering [event_date, event_date + horizon_days].

    The event-date close is used as the entry price (post-publication). Anything
    before `event_date` is excluded.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by DatetimeIndex")
    start = pd.Timestamp(event_date)
    after = prices.loc[prices.index >= start]
    if after.empty:
        return after
    # Take entry bar + next horizon_days trading bars.
    return after.iloc[: horizon_days + 1]


def window_returns(prices: pd.DataFrame, event_date: date, horizon_days: int) -> WindowReturns | None:
    win = forward_window(prices, event_date, horizon_days)
    if len(win) < 2:
        return None
    closes = win["close"].to_numpy(dtype=float)
    entry = closes[0]
    if entry <= 0:
        return None
    rets = closes / entry - 1.0
    running_max = np.maximum.accumulate(closes)
    dd = closes / running_max - 1.0
    return WindowReturns(
        horizon_days=len(win) - 1,
        cum_return=float(rets[-1]),
        max_return=float(rets.max()),
        min_return=float(rets.min()),
        max_drawdown=float(dd.min()),
        n_bars=len(win),
    )


def excess_return(asset_window: WindowReturns, benchmark_window: WindowReturns | None) -> float:
    """Asset cum return minus benchmark cum return, when benchmark is available."""
    if benchmark_window is None:
        return asset_window.cum_return
    return asset_window.cum_return - benchmark_window.cum_return

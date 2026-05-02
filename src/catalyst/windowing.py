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


def abnormal_window_returns(
    asset_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None,
    event_date: date,
    horizon_days: int,
) -> WindowReturns | None:
    """Market-adjusted version of `window_returns`.

    Implements a simplified MacKinlay (1997) event study without estimating
    a full alpha/beta. We use the difference between asset and benchmark
    cumulative returns over the same window:

        AR_t = r_asset,t - r_benchmark,t
        CAR  = sum AR_t over [event_date, event_date + horizon_days]

    The full market-model variant (alpha + beta from a [-250, -30]
    estimation window) is left as an upgrade path; for the analog-event
    posterior the simpler "excess return" variant captures most of the
    variance reduction at a fraction of the data requirements.
    """
    asset_wr = window_returns(asset_prices, event_date, horizon_days)
    if asset_wr is None:
        return None
    if benchmark_prices is None or benchmark_prices.empty:
        return asset_wr
    bench_wr = window_returns(benchmark_prices, event_date, horizon_days)
    if bench_wr is None:
        return asset_wr
    # Subtract the benchmark from each summary stat. CAR-style.
    return WindowReturns(
        horizon_days=asset_wr.horizon_days,
        cum_return=asset_wr.cum_return - bench_wr.cum_return,
        max_return=asset_wr.max_return - bench_wr.max_return,
        min_return=asset_wr.min_return - bench_wr.min_return,
        max_drawdown=asset_wr.max_drawdown - bench_wr.max_drawdown,
        n_bars=asset_wr.n_bars,
    )


def pre_event_runup(prices: pd.DataFrame, event_date: date, lookback_days: int = 20) -> float:
    """Asset return over [-lookback_days, -1] trading days before the event.

    Used by the rule-score dampener (Lee-Swaminathan 2000): catalysts on
    stocks that have already run up tend to deliver smaller post-event
    drift. Returns 0.0 when not enough history is available.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by DatetimeIndex")
    cutoff = pd.Timestamp(event_date)
    pre = prices.loc[prices.index < cutoff]
    if len(pre) < 2:
        return 0.0
    window = pre.iloc[-lookback_days:] if len(pre) >= lookback_days else pre
    closes = window["close"].to_numpy(dtype=float)
    if closes[0] <= 0:
        return 0.0
    return float(closes[-1] / closes[0] - 1.0)

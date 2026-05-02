from datetime import date

import numpy as np
import pandas as pd

from catalyst.windowing import forward_window, window_returns


def _frame(closes: list[float], start: date = date(2024, 1, 2)) -> pd.DataFrame:
    idx = pd.bdate_range(pd.Timestamp(start), periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1] * len(closes),
    }, index=idx)


def test_forward_window_truncates_to_horizon():
    df = _frame([100, 101, 102, 103, 104, 105])
    win = forward_window(df, date(2024, 1, 2), horizon_days=3)
    assert len(win) == 4  # entry bar + 3


def test_window_returns_basic():
    df = _frame([100, 110, 121, 121])
    wr = window_returns(df, date(2024, 1, 2), horizon_days=2)
    assert wr is not None
    assert np.isclose(wr.cum_return, 0.21, atol=1e-6)
    assert wr.max_return >= wr.cum_return
    assert wr.max_drawdown <= 0


def test_window_returns_drawdown():
    df = _frame([100, 110, 90, 95])
    wr = window_returns(df, date(2024, 1, 2), horizon_days=3)
    assert wr is not None
    # peak 110 -> 90 = -18.18% drawdown
    assert np.isclose(wr.max_drawdown, (90 - 110) / 110, atol=1e-6)


def test_window_returns_returns_none_when_too_short():
    df = _frame([100])
    assert window_returns(df, date(2024, 1, 2), horizon_days=5) is None

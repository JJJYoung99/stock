from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from catalyst.config import load_rules
from catalyst.events.csv_store import CsvEventStore
from catalyst.markets.base import MarketAdapter
from catalyst.scoring import (
    AnalogSample,
    Scorer,
    aggregate_rule_scores,
    beta_posterior_prob_profit,
    empirical_risk,
    rule_score,
)
from catalyst.types import Event, EventType, Market


def _ev(et: EventType, mag: float, when: date = date(2026, 1, 1), ticker: str = "T") -> Event:
    return Event(ticker=ticker, market=Market.KR, event_type=et,
                 occurred_at=when, magnitude=mag, source="test")


def test_rule_score_positive_event():
    cfg = load_rules()
    s = rule_score(_ev(EventType.CONTRACT_WIN, 0.10), cfg)
    # weight 1.5 * min(0.10/0.10, 3.0) = 1.5
    assert abs(s - 1.5) < 1e-6


def test_rule_score_below_min_magnitude_zero():
    cfg = load_rules()
    rule = cfg.events[EventType.CONTRACT_WIN]
    s = rule_score(_ev(EventType.CONTRACT_WIN, rule.min_magnitude / 2), cfg)
    assert s == 0.0


def test_rule_score_negative_event_is_negative():
    cfg = load_rules()
    s = rule_score(_ev(EventType.CAPITAL_RAISE, 0.10), cfg)
    assert s < 0


def test_rule_score_capped():
    cfg = load_rules()
    rule = cfg.events[EventType.CONTRACT_WIN]
    huge = _ev(EventType.CONTRACT_WIN, mag=10.0)  # well past cap
    s = rule_score(huge, cfg)
    assert abs(s - rule.weight * rule.magnitude_cap) < 1e-6


def test_aggregate_rule_scores_decay():
    # Two equal positive scores: result < 2 * single because of decay.
    out = aggregate_rule_scores([1.0, 1.0], decay=0.5, cap=10.0)
    assert abs(out - (1.0 + 0.5)) < 1e-6


def test_aggregate_rule_scores_cap():
    out = aggregate_rule_scores([10.0, 10.0, 10.0], decay=1.0, cap=5.0)
    assert out == 5.0


def test_beta_posterior_pulls_to_prior_when_few_samples():
    samples = [AnalogSample(event=_ev(EventType.CONTRACT_WIN, 0.1), cum_return=0.05, max_drawdown=-0.02)]
    p, n = beta_posterior_prob_profit(samples, alpha_prior=2.0, beta_prior=2.0)
    # 1 win, 0 loss, prior(2,2) -> (2+1)/(2+2+1) = 0.6
    assert abs(p - 0.6) < 1e-6
    assert n == 1


def test_beta_posterior_with_many_wins_dominates_prior():
    samples = [AnalogSample(event=_ev(EventType.CONTRACT_WIN, 0.1), cum_return=0.05, max_drawdown=-0.02)
               for _ in range(50)]
    p, _ = beta_posterior_prob_profit(samples, 2.0, 2.0)
    assert p > 0.9


def test_empirical_risk_basic():
    rets = [0.05, -0.02, 0.10, -0.15, 0.08, -0.04]
    samples = [AnalogSample(event=_ev(EventType.CONTRACT_WIN, 0.1), cum_return=r, max_drawdown=-abs(r))
               for r in rets]
    mean_r, var, cvar, max_dd = empirical_risk(samples)
    assert abs(mean_r - np.mean(rets)) < 1e-6
    assert var <= 0  # 5%-VaR on a mostly-mixed distribution lands negative
    assert cvar <= var  # CVaR is at-or-below VaR by definition


# ---------- end-to-end Scorer with a fake adapter ----------------------------

class _StubAdapter(MarketAdapter):
    market = Market.KR

    def __init__(self, returns_per_event: dict[str, float]):
        # Map event date -> realized cum return at horizon.
        self._returns = returns_per_event

    def prices(self, ticker, start, end):
        # 30 trading days from `start`. close[0]=100; close[20] = 100*(1+r); linear in between.
        idx = pd.bdate_range(pd.Timestamp(start), periods=30)
        target_r = self._returns.get(pd.Timestamp(start).date().isoformat(), 0.0)
        closes = np.linspace(100, 100 * (1 + target_r), num=21).tolist() + [100 * (1 + target_r)] * 9
        return pd.DataFrame({
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1] * 30,
        }, index=idx)

    def benchmark(self, start, end):
        # Flat benchmark so abnormal_return == asset_return for this stub.
        idx = pd.bdate_range(pd.Timestamp(start), periods=30)
        return pd.DataFrame({
            "open": [100.0] * 30, "high": [100.0] * 30, "low": [100.0] * 30,
            "close": [100.0] * 30, "volume": [1] * 30,
        }, index=idx)

    def universe(self):
        return ["T"]


def test_scorer_uses_analog_returns(tmp_path: Path):
    store = CsvEventStore(tmp_path / "events.csv")
    # 6 historical CONTRACT_WIN analogs: 5 winners, 1 loser. magnitude band centered at 0.10.
    history = []
    for i, r in enumerate([0.05, 0.04, 0.06, 0.07, 0.05, -0.03]):
        history.append(_ev(EventType.CONTRACT_WIN, 0.10, when=date(2025, 1, i + 2), ticker=f"H{i}"))
    store.append(history)

    returns = {ev.occurred_at.isoformat(): r for ev, r in zip(
        history, [0.05, 0.04, 0.06, 0.07, 0.05, -0.03]
    )}
    adapter = _StubAdapter(returns)
    scorer = Scorer(adapter, store)

    target = _ev(EventType.CONTRACT_WIN, 0.10, when=date(2026, 5, 1), ticker="TARGET")
    sc = scorer.score_event(target)

    assert sc.n_analogs >= 5
    assert sc.prob_profit > 0.5
    assert sc.expected_return > 0
    assert sc.rule_score > 0

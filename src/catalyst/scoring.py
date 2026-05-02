"""Scoring engine: rule_score + Bayesian posterior + risk metrics.

Pipeline (per (ticker, as_of)):

    1. Collect concurrent events on the ticker.
    2. For each event, compute a rule_score from `configs/event_rules.yaml`:
           rule_score(e) = weight(type) * clip(magnitude / scale, 0, cap) * confidence
    3. Find historical analog events (same type, similar magnitude, before
       as_of) in the EventStore. For each analog, compute the realized
       horizon-day return using the market adapter.
    4. Posterior P(profit) := Beta(alpha + wins, beta + losses).
       Expected return  := mean of analog returns.
       Risk metrics     := empirical 5%-VaR and CVaR on the analog return
                           distribution; max drawdown is the worst observed
                           intra-window drawdown across analogs.
    5. Aggregate concurrent events with rank-decay so 1 strong + 5 weak
       events are not equivalent to 6 strong events.

The scorer is deliberately not random / not stochastic. Two callers with the
same EventStore + same prices get identical numbers; that is required for
the backtester to be reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from catalyst.config import RuleConfig, load_rules
from catalyst.events.base import EventStore
from catalyst.markets.base import MarketAdapter
from catalyst.types import Event, EventType, Market, Score, ScoreReport
from catalyst.windowing import abnormal_window_returns, window_returns


# ---------- rule-based score --------------------------------------------------

def _runup_dampener(runup: float, cfg: RuleConfig) -> float:
    """Lee-Swaminathan (2000) / Hong-Stein (1999) post-event drift correction:
    the more a stock has already run up before the catalyst, the smaller
    the residualized post-event drift. We dampen rule_score by:

        max(floor, 1 - max(0, runup) * coef)

    Negative runups (price already fell) leave the score untouched —
    that is, we do not boost mean-reversion plays here.
    """
    if cfg.runup.coef <= 0:
        return 1.0
    return max(cfg.runup.floor, 1.0 - max(0.0, runup) * cfg.runup.coef)


def rule_score(event: Event, cfg: RuleConfig) -> float:
    """Signed weight * scaled magnitude * confidence * runup_dampener.

    The (scale, cap) used to clip magnitude depends on
    `event.extras["magnitude_source"]`:

        * "sue"               -> overrides table for SUE  (scale 1.0, cap 4.0)
        * "revision_z"        -> overrides table for revision-z
        * "baseline_adjusted" -> overrides table or per-event default
        * default / fallback  -> per-event scale + cap from configs/event_rules.yaml
    """
    rule = cfg.for_event(event.event_type)
    if rule is None:
        return 0.0
    if event.magnitude < rule.min_magnitude:
        return 0.0

    source = event.magnitude_source()
    override = cfg.magnitude_override(event.event_type, source)
    scale = override.scale if override else (rule.magnitude_scale or 1.0)
    cap = override.cap if override else rule.magnitude_cap
    if scale <= 0:
        scale = 1.0
    multiplier = min(event.magnitude / scale, cap)

    base = rule.weight * multiplier * event.confidence
    runup = float(event.extras.get("pre_event_runup", 0.0))
    return float(base * _runup_dampener(runup, cfg))


def aggregate_rule_scores(scores: Iterable[float], decay: float, cap: float) -> float:
    """Sum-with-rank-decay (sorted by absolute magnitude, descending)."""
    s = sorted(scores, key=lambda x: -abs(x))
    total = 0.0
    for k, v in enumerate(s):
        total += v * (decay ** k)
    return float(np.clip(total, -cap, cap))


# ---------- analog selection + realized returns ------------------------------

@dataclass(frozen=True)
class AnalogSample:
    event: Event
    cum_return: float
    max_drawdown: float


def select_analogs(
    target: Event,
    store: EventStore,
    cfg: RuleConfig,
    *,
    as_of: date,
) -> list[Event]:
    """Historical events of the same type with magnitude within the configured
    band, restricted to those that occurred strictly before `as_of` (no peek).
    """
    band = cfg.posterior.magnitude_band
    lo = max(target.magnitude * (1 - band), 0.0)
    hi = target.magnitude * (1 + band)
    candidates = store.query(
        market=target.market,
        event_type=target.event_type,
        before=as_of,
    )
    return [
        c for c in candidates
        if lo <= c.magnitude <= hi
        and not (c.ticker == target.ticker and c.occurred_at == target.occurred_at)
    ]


def realized_returns(
    events: list[Event],
    adapter: MarketAdapter,
    horizon_days: int,
    *,
    use_abnormal: bool = True,
) -> list[AnalogSample]:
    """Compute (cum_return, max_drawdown) for each analog using the adapter.

    When `use_abnormal=True` (default), returns are benchmark-adjusted in
    the spirit of MacKinlay (1997) — the cumulative excess return over
    the same window. Falls back to raw returns when no benchmark is
    available, which keeps the function usable with minimal adapters.
    """
    out: list[AnalogSample] = []
    for ev in events:
        # Pull a generous window so we have enough trading days even with holidays.
        start = ev.occurred_at
        end = pd.Timestamp(start) + pd.Timedelta(days=int(horizon_days * 1.8) + 7)
        try:
            prices = adapter.prices(ev.ticker, start, end.date())
        except Exception:
            continue
        if prices is None or prices.empty:
            continue
        prices.index = pd.to_datetime(prices.index)

        bench_prices = None
        if use_abnormal:
            try:
                bench_prices = adapter.benchmark(start, end.date())
                if bench_prices is not None and not bench_prices.empty:
                    bench_prices.index = pd.to_datetime(bench_prices.index)
            except (NotImplementedError, Exception):
                bench_prices = None

        wr = (
            abnormal_window_returns(prices, bench_prices, ev.occurred_at, horizon_days)
            if use_abnormal else window_returns(prices, ev.occurred_at, horizon_days)
        )
        if wr is None:
            continue
        out.append(AnalogSample(event=ev, cum_return=wr.cum_return, max_drawdown=wr.max_drawdown))
    return out


# ---------- posterior + risk -------------------------------------------------

def beta_posterior_prob_profit(
    samples: list[AnalogSample],
    alpha_prior: float,
    beta_prior: float,
) -> tuple[float, int]:
    """Posterior mean of Beta(alpha + wins, beta + losses).

    A 'win' is cum_return > 0 strictly. Returns (prob, n_used).
    """
    n = len(samples)
    wins = sum(1 for s in samples if s.cum_return > 0)
    losses = n - wins
    a = alpha_prior + wins
    b = beta_prior + losses
    return a / (a + b), n


def empirical_risk(samples: list[AnalogSample]) -> tuple[float, float, float, float]:
    """Return (mean_return, var_95, cvar_95, worst_max_drawdown)."""
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    rets = np.array([s.cum_return for s in samples], dtype=float)
    dds = np.array([s.max_drawdown for s in samples], dtype=float)
    var_95 = float(np.quantile(rets, 0.05))
    tail = rets[rets <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95
    return float(rets.mean()), var_95, cvar_95, float(dds.min())


# ---------- the scorer -------------------------------------------------------

class Scorer:
    """High-level entry point. Caches the historical event store + adapter."""

    def __init__(
        self,
        adapter: MarketAdapter,
        store: EventStore,
        cfg: RuleConfig | None = None,
    ):
        self.adapter = adapter
        self.store = store
        self.cfg = cfg or load_rules()

    def score_event(self, event: Event, *, as_of: date | None = None) -> Score:
        as_of = as_of or event.occurred_at
        rule = self.cfg.for_event(event.event_type)
        horizon = rule.horizon_days if rule else self.cfg.default_horizon_days

        rs = rule_score(event, self.cfg)

        analogs = select_analogs(event, self.store, self.cfg, as_of=as_of)
        samples = realized_returns(
            analogs, self.adapter, horizon, use_abnormal=self.cfg.use_abnormal_returns,
        )

        post = self.cfg.posterior
        if len(samples) >= post.min_analogs:
            prob, n_used = beta_posterior_prob_profit(samples, post.alpha_prior, post.beta_prior)
            mean_r, var95, cvar95, max_dd = empirical_risk(samples)
        else:
            # Fall back to prior mean. Risk metrics are unknown -> mark with NaN-ish 0.
            prob = post.alpha_prior / (post.alpha_prior + post.beta_prior)
            n_used = len(samples)
            mean_r, var95, cvar95, max_dd = 0.0, 0.0, 0.0, 0.0

        return Score(
            rule_score=rs,
            prob_profit=float(prob),
            expected_return=float(mean_r),
            var_95=float(var95),
            cvar_95=float(cvar95),
            max_drawdown=float(max_dd),
            n_analogs=int(n_used),
            horizon_days=int(horizon),
        )

    def score(self, ticker: str, market: Market, as_of: date) -> ScoreReport:
        """Score a (ticker, date) using all events that fired on that day."""
        # Find events on or just before `as_of` (same trading day) for this ticker.
        events_today = [
            e for e in self.store.query(market=market, ticker=ticker)
            if e.occurred_at == as_of
        ]
        return self._build_report(ticker, market, as_of, events_today)

    def score_pending(self, events: list[Event], *, as_of: date) -> ScoreReport:
        """Score a list of just-detected events that are not yet in the store
        (e.g. live news). Useful from CLI / dashboards."""
        if not events:
            raise ValueError("score_pending requires at least one event")
        ticker = events[0].ticker
        market = events[0].market
        if any(e.ticker != ticker or e.market != market for e in events):
            raise ValueError("all events must share (ticker, market)")
        return self._build_report(ticker, market, as_of, events)

    def _build_report(
        self,
        ticker: str,
        market: Market,
        as_of: date,
        events: list[Event],
    ) -> ScoreReport:
        per_event: list[Score] = [self.score_event(ev, as_of=as_of) for ev in events]
        notes: list[str] = []

        if not events:
            return ScoreReport(
                ticker=ticker, market=market, as_of=as_of,
                events=tuple(), per_event=tuple(),
                aggregated_rule_score=0.0,
                aggregated_prob_profit=self.cfg.posterior.alpha_prior /
                    (self.cfg.posterior.alpha_prior + self.cfg.posterior.beta_prior),
                aggregated_expected_return=0.0,
                aggregated_var_95=0.0,
                aggregated_cvar_95=0.0,
                horizon_days=self.cfg.default_horizon_days,
                notes=("no events on this date",),
            )

        # Aggregate rule scores with decay.
        agg_rs = aggregate_rule_scores(
            (s.rule_score for s in per_event),
            decay=self.cfg.aggregation.decay,
            cap=self.cfg.aggregation.cap,
        )

        # Pick the dominant event's horizon for the aggregate report.
        dominant = max(zip(events, per_event), key=lambda pair: abs(pair[1].rule_score))
        horizon = dominant[1].horizon_days

        # Aggregate posterior / return / risk: weight each event's metric by
        # |rule_score|. Falls back to simple mean when all rule_scores are 0.
        weights = np.array([abs(s.rule_score) for s in per_event], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        prob = float(np.dot(weights, [s.prob_profit for s in per_event]))
        er = float(np.dot(weights, [s.expected_return for s in per_event]))
        v95 = float(np.dot(weights, [s.var_95 for s in per_event]))
        c95 = float(np.dot(weights, [s.cvar_95 for s in per_event]))

        if any(s.n_analogs < self.cfg.posterior.min_analogs for s in per_event):
            notes.append(
                f"low analog count (min_required={self.cfg.posterior.min_analogs}); "
                "posterior is prior-dominated."
            )
        if agg_rs <= 0:
            notes.append("aggregated rule_score is non-positive — net catalyst is bearish or null.")

        return ScoreReport(
            ticker=ticker, market=market, as_of=as_of,
            events=tuple(events), per_event=tuple(per_event),
            aggregated_rule_score=float(agg_rs),
            aggregated_prob_profit=prob,
            aggregated_expected_return=er,
            aggregated_var_95=v95,
            aggregated_cvar_95=c95,
            horizon_days=int(horizon),
            notes=tuple(notes),
        )

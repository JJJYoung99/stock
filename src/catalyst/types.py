"""Core dataclasses shared across the framework.

Every other module imports from here, so it must stay dependency-free
(no pandas, no numpy at module top level).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Market(str, Enum):
    KR = "KR"
    US = "US"


class EventType(str, Enum):
    CONTRACT_WIN = "CONTRACT_WIN"
    EARNINGS_BEAT = "EARNINGS_BEAT"
    EARNINGS_MISS = "EARNINGS_MISS"
    GUIDANCE_RAISE = "GUIDANCE_RAISE"
    ANALYST_TARGET_UP = "ANALYST_TARGET_UP"
    ANALYST_RATING_UP = "ANALYST_RATING_UP"
    BUYBACK = "BUYBACK"
    INSIDER_BUY = "INSIDER_BUY"
    CAPITAL_RAISE = "CAPITAL_RAISE"
    CONVERTIBLE_ISSUE = "CONVERTIBLE_ISSUE"
    REGULATORY_PROBE = "REGULATORY_PROBE"


@dataclass(frozen=True)
class Event:
    ticker: str
    market: Market
    event_type: EventType
    occurred_at: date
    magnitude: float
    source: str
    raw: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    # Optional numeric/categorical fields used by the SUE/revision-z scorers
    # and the pre-event-runup dampener. See docs in scoring.py.
    # Recognized keys (all optional):
    #   "magnitude_source" : str  -- one of {"sue","revision_z","fallback_pct","car_only","ordinal"}
    #   "consensus_stdev"  : float -- σ of analyst EPS estimates (Bernard-Thomas 1989)
    #   "n_estimates"      : int   -- number of analysts in consensus
    #   "consensus_dispersion": float -- σ of analyst price targets (Stickel 1991)
    #   "prior_target"     : float -- prior consensus target (for revision-z)
    #   "pre_event_runup"  : float -- asset return over [-20, -1] trading days
    #   "season_baseline"  : float -- cross-sectional median magnitude in current season
    extras: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str, str]:
        return (self.market.value, self.ticker, self.occurred_at.isoformat(), self.event_type.value)

    def magnitude_source(self) -> str:
        return str(self.extras.get("magnitude_source", "fallback_pct"))


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    market: Market
    bar_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class Score:
    """Per-event scoring output."""

    rule_score: float
    prob_profit: float
    expected_return: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    n_analogs: int
    horizon_days: int

    @property
    def edge(self) -> float:
        """Heuristic combined edge: rule_score * (prob_profit - 0.5) * 2."""
        return self.rule_score * (self.prob_profit - 0.5) * 2.0


@dataclass(frozen=True)
class ScoreReport:
    """Aggregated score for one (ticker, date) combining N concurrent events."""

    ticker: str
    market: Market
    as_of: date
    events: tuple[Event, ...]
    per_event: tuple[Score, ...]
    aggregated_rule_score: float
    aggregated_prob_profit: float
    aggregated_expected_return: float
    aggregated_var_95: float
    aggregated_cvar_95: float
    horizon_days: int
    notes: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict suitable for pandas / CSV / JSON."""
        return {
            "market": self.market.value,
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat(),
            "n_events": len(self.events),
            "event_types": ",".join(e.event_type.value for e in self.events),
            "rule_score": round(self.aggregated_rule_score, 4),
            "prob_profit": round(self.aggregated_prob_profit, 4),
            "expected_return": round(self.aggregated_expected_return, 4),
            "var_95": round(self.aggregated_var_95, 4),
            "cvar_95": round(self.aggregated_cvar_95, 4),
            "horizon_days": self.horizon_days,
            "notes": " | ".join(self.notes),
        }

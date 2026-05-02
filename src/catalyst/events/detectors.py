"""Concrete event detectors.

Each detector takes a raw payload (a dict) and returns 0..N typed `Event`s.

The DART / SEC / news detectors below are *pattern-matching* heuristics — they
do not call any network APIs, so the framework is testable offline. Production
users should pipe their own DART OpenAPI / SEC EDGAR / Finnhub responses through
these detectors, or replace them with full NLP models from `catalyst[nlp]`.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from catalyst.events.base import EventDetector
from catalyst.types import Event, EventType, Market


def _as_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        # accept YYYY-MM-DD, YYYY.MM.DD, YYYYMMDD
        s = v.replace(".", "-")
        if len(s) == 8 and s.isdigit():
            return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        return date.fromisoformat(s[:10])
    raise TypeError(f"cannot coerce {v!r} to date")


# ---------- KR / DART ---------------------------------------------------------

# DART report-name patterns. Korean disclosure titles are stable; the
# 단일판매·공급계약체결 form is the canonical "contract win" trigger.
_DART_CONTRACT_RE = re.compile(r"단일판매[·\-]?공급계약(체결|체결정정)?")
_DART_CAPRAISE_RE = re.compile(r"유상증자|주주배정|제3자배정")
_DART_BUYBACK_RE = re.compile(r"자기주식취득(결정|신탁계약)?")
_DART_PROBE_RE = re.compile(r"세무조사|불공정거래|조사개시")
_DART_CB_RE = re.compile(r"(전환사채|신주인수권부사채|교환사채)\s*권면(가|총액)")


class DartContractDetector(EventDetector):
    """Korean contract-win disclosures. Magnitude := contract_amount / market_cap."""
    source = "DART"

    def detect(self, payload: dict[str, Any]) -> list[Event]:
        title = str(payload.get("report_nm") or payload.get("title") or "")
        if not _DART_CONTRACT_RE.search(title):
            return []
        ticker = str(payload.get("stock_code") or payload.get("ticker") or "").zfill(6)
        amount = float(payload.get("contract_amount", 0.0))
        mcap = float(payload.get("market_cap", 0.0))
        if mcap <= 0:
            return []
        mag = amount / mcap
        return [Event(
            ticker=ticker,
            market=Market.KR,
            event_type=EventType.CONTRACT_WIN,
            occurred_at=_as_date(payload.get("rcept_dt") or payload.get("date")),
            magnitude=round(mag, 6),
            source=self.source,
            raw={"title": title, "contract_amount": amount, "market_cap": mcap},
        )]


class DartCapitalRaiseDetector(EventDetector):
    """Dilutive capital-raise disclosures (paid-in capital increase / 유상증자)."""
    source = "DART"

    def detect(self, payload: dict[str, Any]) -> list[Event]:
        title = str(payload.get("report_nm") or payload.get("title") or "")
        if _DART_CAPRAISE_RE.search(title):
            offering = float(payload.get("offering_size", 0.0))
            mcap = float(payload.get("market_cap", 0.0))
            if mcap <= 0:
                return []
            return [Event(
                ticker=str(payload.get("stock_code") or "").zfill(6),
                market=Market.KR,
                event_type=EventType.CAPITAL_RAISE,
                occurred_at=_as_date(payload.get("rcept_dt") or payload.get("date")),
                magnitude=round(offering / mcap, 6),
                source=self.source,
                raw={"title": title, "offering_size": offering, "market_cap": mcap},
            )]
        if _DART_CB_RE.search(title):
            offering = float(payload.get("offering_size", 0.0))
            mcap = float(payload.get("market_cap", 0.0))
            if mcap <= 0:
                return []
            return [Event(
                ticker=str(payload.get("stock_code") or "").zfill(6),
                market=Market.KR,
                event_type=EventType.CONVERTIBLE_ISSUE,
                occurred_at=_as_date(payload.get("rcept_dt") or payload.get("date")),
                magnitude=round(offering / mcap, 6),
                source=self.source,
                raw={"title": title},
            )]
        if _DART_BUYBACK_RE.search(title):
            amount = float(payload.get("buyback_amount", 0.0))
            mcap = float(payload.get("market_cap", 0.0))
            if mcap <= 0:
                return []
            return [Event(
                ticker=str(payload.get("stock_code") or "").zfill(6),
                market=Market.KR,
                event_type=EventType.BUYBACK,
                occurred_at=_as_date(payload.get("rcept_dt") or payload.get("date")),
                magnitude=round(amount / mcap, 6),
                source=self.source,
                raw={"title": title},
            )]
        if _DART_PROBE_RE.search(title):
            return [Event(
                ticker=str(payload.get("stock_code") or "").zfill(6),
                market=Market.KR,
                event_type=EventType.REGULATORY_PROBE,
                occurred_at=_as_date(payload.get("rcept_dt") or payload.get("date")),
                magnitude=1.0,
                source=self.source,
                raw={"title": title},
            )]
        return []


# ---------- Earnings (market-agnostic) ---------------------------------------

class EarningsSurpriseDetector(EventDetector):
    """Beats / misses from a normalized earnings payload.

    Magnitude is computed in the SUE-preferred order:

    1. **SUE** (Bernard-Thomas 1989; Foster-Olsen-Shevlin 1984):
       `SUE = (actual - consensus) / consensus_stdev`
       Used when `consensus_stdev > 0` is provided. This is the canonical
       PEAD normalization — it standardizes for analyst dispersion so
       a 3% beat with tight consensus dominates a 10% beat with chaos.
    2. **Baseline-adjusted surprise%**:
       `(surprise% - season_baseline) / |consensus|`
       Subtracts the cross-sectional median surprise so beats during a
       hot earnings season (Q1 2026 had +20.7% mean surprise) don't get
       full credit just for clearing zero.
    3. **Plain surprise%** (legacy fallback):
       `(actual - consensus) / |consensus|`

    Expected payload keys (only `actual_eps`, `consensus_eps` required):
        ticker, market, announce_date | fiscal_period_end,
        actual_eps, consensus_eps,
        consensus_stdev?, n_estimates?, season_baseline?
    """
    source = "earnings"

    def detect(self, payload: dict[str, Any]) -> list[Event]:
        try:
            actual = float(payload["actual_eps"])
            consensus = float(payload["consensus_eps"])
        except (KeyError, TypeError, ValueError):
            return []
        if consensus == 0:
            return []
        raw_surprise = (actual - consensus) / abs(consensus)

        consensus_stdev = payload.get("consensus_stdev")
        season_baseline = payload.get("season_baseline")
        n_estimates = payload.get("n_estimates")

        magnitude_source = "fallback_pct"
        if consensus_stdev is not None and float(consensus_stdev) > 0:
            sue = (actual - consensus) / float(consensus_stdev)
            magnitude_value = abs(sue)
            magnitude_source = "sue"
            sign = 1.0 if sue > 0 else -1.0
        elif season_baseline is not None:
            adjusted = abs(raw_surprise) - abs(float(season_baseline))
            magnitude_value = max(adjusted, 0.0)
            magnitude_source = "baseline_adjusted"
            sign = 1.0 if raw_surprise > 0 else -1.0
        else:
            magnitude_value = abs(raw_surprise)
            sign = 1.0 if raw_surprise > 0 else -1.0

        et = EventType.EARNINGS_BEAT if sign > 0 else EventType.EARNINGS_MISS

        extras: dict[str, Any] = {"magnitude_source": magnitude_source}
        if consensus_stdev is not None:
            extras["consensus_stdev"] = float(consensus_stdev)
        if n_estimates is not None:
            extras["n_estimates"] = int(n_estimates)
        if season_baseline is not None:
            extras["season_baseline"] = float(season_baseline)
        if "pre_event_runup" in payload:
            extras["pre_event_runup"] = float(payload["pre_event_runup"])

        return [Event(
            ticker=str(payload["ticker"]),
            market=Market(str(payload.get("market", "US"))),
            event_type=et,
            occurred_at=_as_date(payload.get("announce_date") or payload.get("fiscal_period_end")),
            magnitude=round(float(magnitude_value), 6),
            source=self.source,
            raw={"actual_eps": actual, "consensus_eps": consensus, "surprise_pct": raw_surprise},
            extras=extras,
        )]


# ---------- Analyst ----------------------------------------------------------

class AnalystTargetDetector(EventDetector):
    """Sell-side target / rating change.

    Magnitude for ANALYST_TARGET_UP is computed in the dispersion-preferred
    order (Stickel 1991; Womack 1996; Jegadeesh-Kim 2010):

    1. **revision_z** (preferred):
       `(new_target - old_target) / consensus_dispersion`
       — z-score of the revision against the cross-section of active
       analyst targets. Removes "noise" upgrades during high-dispersion
       periods.
    2. **% revision**:
       `(new_target - old_target) / old_target`
       Fallback when dispersion is unavailable.

    Expected keys: ticker, market, date, broker?,
        old_target, new_target, consensus_dispersion?, old_rating?, new_rating?
    """
    source = "analyst"

    _RATING_ORDER = {"sell": 0, "underweight": 1, "hold": 2, "neutral": 2,
                     "buy": 3, "overweight": 3, "strong buy": 4}

    def detect(self, payload: dict[str, Any]) -> list[Event]:
        events: list[Event] = []
        ticker = str(payload["ticker"])
        market = Market(str(payload.get("market", "US")))
        when = _as_date(payload.get("date"))

        old_t = payload.get("old_target")
        new_t = payload.get("new_target")
        consensus_dispersion = payload.get("consensus_dispersion")
        if old_t is not None and new_t is not None:
            old_v, new_v = float(old_t), float(new_t)
            if old_v > 0 and new_v > old_v:
                pct_revision = (new_v - old_v) / old_v
                if consensus_dispersion is not None and float(consensus_dispersion) > 0:
                    magnitude_value = (new_v - old_v) / float(consensus_dispersion)
                    magnitude_source = "revision_z"
                else:
                    magnitude_value = pct_revision
                    magnitude_source = "fallback_pct"
                extras: dict[str, Any] = {
                    "magnitude_source": magnitude_source,
                    "pct_revision": float(pct_revision),
                }
                if consensus_dispersion is not None:
                    extras["consensus_dispersion"] = float(consensus_dispersion)
                events.append(Event(
                    ticker=ticker, market=market,
                    event_type=EventType.ANALYST_TARGET_UP,
                    occurred_at=when,
                    magnitude=round(float(magnitude_value), 6),
                    source=self.source,
                    raw={"broker": payload.get("broker"), "old_target": old_v, "new_target": new_v},
                    extras=extras,
                ))

        old_r = str(payload.get("old_rating") or "").lower().strip()
        new_r = str(payload.get("new_rating") or "").lower().strip()
        if old_r and new_r:
            o = self._RATING_ORDER.get(old_r, 2)
            n = self._RATING_ORDER.get(new_r, 2)
            if n > o:
                events.append(Event(
                    ticker=ticker, market=market,
                    event_type=EventType.ANALYST_RATING_UP,
                    occurred_at=when,
                    magnitude=float(n - o),
                    source=self.source,
                    raw={"broker": payload.get("broker"), "old_rating": old_r, "new_rating": new_r},
                ))
        return events


# ---------- Generic news headline (rule-based fallback) ----------------------

# Keyword maps. Crude but useful when you don't have a finBERT model handy.
_KO_KEYWORDS: dict[EventType, list[str]] = {
    EventType.CONTRACT_WIN: ["수주", "공급계약", "납품계약", "MOU 체결"],
    EventType.GUIDANCE_RAISE: ["가이던스 상향", "실적 전망 상향", "연간 전망치 상향"],
    EventType.ANALYST_TARGET_UP: ["목표주가 상향", "목표가 상향"],
    EventType.BUYBACK: ["자사주 매입", "자기주식 취득"],
    EventType.CAPITAL_RAISE: ["유상증자", "신주발행"],
    EventType.REGULATORY_PROBE: ["세무조사", "불공정거래 조사", "검찰 압수수색"],
}
_EN_KEYWORDS: dict[EventType, list[str]] = {
    EventType.CONTRACT_WIN: ["awarded contract", "wins contract", "secures order", "purchase order"],
    EventType.GUIDANCE_RAISE: ["raises guidance", "lifts outlook", "boosts forecast"],
    EventType.ANALYST_TARGET_UP: ["raises price target", "lifts price target", "increases price target"],
    EventType.BUYBACK: ["buyback", "share repurchase"],
    EventType.CAPITAL_RAISE: ["secondary offering", "common stock offering", "equity offering"],
    EventType.REGULATORY_PROBE: ["sec investigation", "doj probe", "subpoena"],
}


class NewsHeadlineDetector(EventDetector):
    """Keyword classifier over headlines. Confidence := 1 / matched_keywords."""
    source = "news"

    def __init__(self, market: Market | str = Market.KR):
        self.market = Market(market) if not isinstance(market, Market) else market
        self._kw = _KO_KEYWORDS if self.market == Market.KR else _EN_KEYWORDS

    def detect(self, payload: dict[str, Any]) -> list[Event]:
        headline = str(payload.get("headline") or payload.get("title") or "")
        if not headline:
            return []
        h = headline.lower() if self.market == Market.US else headline
        out: list[Event] = []
        for et, keywords in self._kw.items():
            if any(kw.lower() in h if self.market == Market.US else kw in h for kw in keywords):
                out.append(Event(
                    ticker=str(payload["ticker"]),
                    market=self.market,
                    event_type=et,
                    occurred_at=_as_date(payload.get("date")),
                    magnitude=float(payload.get("magnitude", 1.0)),
                    source=self.source,
                    raw={"headline": headline},
                    confidence=0.6,  # rule-based headline match: low confidence
                ))
        return out

"""Tests for the SUE / revision-z / runup-dampener / abnormal-returns upgrades."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catalyst.config import load_rules
from catalyst.events.csv_store import CsvEventStore
from catalyst.events.detectors import (
    AnalystTargetDetector,
    EarningsSurpriseDetector,
)
from catalyst.scoring import rule_score
from catalyst.types import Event, EventType, Market
from catalyst.windowing import (
    abnormal_window_returns,
    pre_event_runup,
    window_returns,
)


# ---------- SUE in earnings detector -----------------------------------------

def test_earnings_sue_used_when_stdev_provided():
    det = EarningsSurpriseDetector()
    out = det.detect({
        "ticker": "AAPL", "market": "US", "announce_date": "2026-04-30",
        "actual_eps": 1.20, "consensus_eps": 1.00, "consensus_stdev": 0.05,
        "n_estimates": 18,
    })
    assert len(out) == 1
    ev = out[0]
    # SUE = (1.20 - 1.00) / 0.05 = 4.0  (well above |SUE| > 1 PEAD threshold)
    assert ev.magnitude_source() == "sue"
    assert abs(ev.magnitude - 4.0) < 1e-6
    assert ev.extras.get("consensus_stdev") == 0.05
    assert ev.extras.get("n_estimates") == 18


def test_earnings_baseline_adjusted_when_stdev_missing():
    det = EarningsSurpriseDetector()
    out = det.detect({
        "ticker": "AAPL", "market": "US", "announce_date": "2026-04-30",
        "actual_eps": 1.05, "consensus_eps": 1.00, "season_baseline": 0.20,
    })
    ev = out[0]
    # raw surprise 5%, baseline 20% -> adjusted = max(0.05 - 0.20, 0) = 0
    assert ev.magnitude_source() == "baseline_adjusted"
    assert ev.magnitude == 0.0


def test_earnings_fallback_pct_when_no_stdev_no_baseline():
    det = EarningsSurpriseDetector()
    out = det.detect({
        "ticker": "AAPL", "market": "US", "announce_date": "2026-04-30",
        "actual_eps": 1.20, "consensus_eps": 1.00,
    })
    ev = out[0]
    assert ev.magnitude_source() == "fallback_pct"
    assert abs(ev.magnitude - 0.20) < 1e-6


# ---------- revision-z in analyst detector -----------------------------------

def test_analyst_revision_z_when_dispersion_provided():
    det = AnalystTargetDetector()
    out = det.detect({
        "ticker": "NVDA", "market": "US", "date": "2026-04-30",
        "old_target": 100, "new_target": 120, "consensus_dispersion": 5.0,
    })
    target_ev = next(e for e in out if e.event_type == EventType.ANALYST_TARGET_UP)
    # revision_z = (120 - 100) / 5 = 4.0
    assert target_ev.magnitude_source() == "revision_z"
    assert abs(target_ev.magnitude - 4.0) < 1e-6


def test_analyst_fallback_pct_when_dispersion_missing():
    det = AnalystTargetDetector()
    out = det.detect({
        "ticker": "NVDA", "market": "US", "date": "2026-04-30",
        "old_target": 100, "new_target": 120,
    })
    target_ev = next(e for e in out if e.event_type == EventType.ANALYST_TARGET_UP)
    assert target_ev.magnitude_source() == "fallback_pct"
    assert abs(target_ev.magnitude - 0.20) < 1e-6


# ---------- runup dampener in rule_score -------------------------------------

def _ev(et: EventType, mag: float, source: str = "fallback_pct", runup: float = 0.0) -> Event:
    extras: dict = {"magnitude_source": source}
    if runup:
        extras["pre_event_runup"] = runup
    return Event(
        ticker="T", market=Market.KR, event_type=et,
        occurred_at=date(2026, 5, 1), magnitude=mag, source="t", extras=extras,
    )


def test_runup_dampener_reduces_score():
    cfg = load_rules()
    base = rule_score(_ev(EventType.CONTRACT_WIN, 0.10), cfg)
    after = rule_score(_ev(EventType.CONTRACT_WIN, 0.10, runup=0.30), cfg)
    assert after < base
    # Specifically: dampener = max(0.2, 1 - 0.30 * 1.5) = max(0.2, 0.55) = 0.55
    assert abs(after / base - 0.55) < 1e-6


def test_runup_dampener_floor():
    cfg = load_rules()
    base = rule_score(_ev(EventType.CONTRACT_WIN, 0.10), cfg)
    extreme = rule_score(_ev(EventType.CONTRACT_WIN, 0.10, runup=2.0), cfg)
    assert abs(extreme / base - cfg.runup.floor) < 1e-6


def test_runup_negative_does_not_boost():
    cfg = load_rules()
    base = rule_score(_ev(EventType.CONTRACT_WIN, 0.10), cfg)
    after = rule_score(_ev(EventType.CONTRACT_WIN, 0.10, runup=-0.30), cfg)
    assert abs(after - base) < 1e-9


# ---------- magnitude_source overrides ---------------------------------------

def test_sue_uses_override_scale():
    cfg = load_rules()
    sue_event = _ev(EventType.EARNINGS_BEAT, 4.0, source="sue")
    fallback_event = _ev(EventType.EARNINGS_BEAT, 0.20, source="fallback_pct")
    sue_score = rule_score(sue_event, cfg)
    fallback_score = rule_score(fallback_event, cfg)
    # SUE override scale=1.0, cap=4.0, weight=1.2  -> 1.2 * min(4/1, 4) = 4.8
    assert abs(sue_score - 4.8) < 1e-6
    # fallback magnitude_scale=0.20, cap=2.5  -> 1.2 * min(0.20/0.20, 2.5) = 1.2
    assert abs(fallback_score - 1.2) < 1e-6


# ---------- abnormal-return windowing ---------------------------------------

def _frame(closes: list[float], start: date = date(2024, 1, 2)) -> pd.DataFrame:
    idx = pd.bdate_range(pd.Timestamp(start), periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1] * len(closes),
    }, index=idx)


def test_abnormal_window_returns_subtracts_benchmark():
    asset = _frame([100, 110, 121])  # +10%, +21% cumulative
    bench = _frame([100, 105, 110.25])  # +5%, +10.25% cumulative
    wr = abnormal_window_returns(asset, bench, date(2024, 1, 2), horizon_days=2)
    assert wr is not None
    # CAR = 0.21 - 0.1025 = 0.1075
    assert abs(wr.cum_return - (0.21 - 0.1025)) < 1e-6


def test_abnormal_falls_back_to_raw_when_benchmark_empty():
    asset = _frame([100, 110, 121])
    bench = pd.DataFrame()
    wr = abnormal_window_returns(asset, bench, date(2024, 1, 2), horizon_days=2)
    raw = window_returns(asset, date(2024, 1, 2), horizon_days=2)
    assert wr.cum_return == raw.cum_return


# ---------- pre-event runup helper ------------------------------------------

def test_pre_event_runup_basic():
    # 22 trading days; runup is over the last 20 of the pre-event window.
    closes = list(np.linspace(100, 130, 25))  # +30% over 24 bars
    df = _frame(closes, start=date(2024, 1, 2))
    event_date = df.index[24].date()
    runup = pre_event_runup(df, event_date, lookback_days=20)
    # Index 4..23 is a 20-bar window. close[4]=105.0; close[23]=128.75 -> ~22.6%.
    assert 0.20 <= runup <= 0.25


# ---------- backward-compat: CSV without extras column still loads ----------

def test_csv_store_loads_legacy_csv_without_extras_column(tmp_path: Path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_text(
        "market,ticker,event_type,occurred_at,magnitude,source,confidence,raw_json\n"
        "US,AAPL,EARNINGS_BEAT,2026-04-30,0.05,test,1.0,\n",
        encoding="utf-8",
    )
    store = CsvEventStore(legacy)
    evs = store.all()
    assert len(evs) == 1
    assert evs[0].extras == {}
    assert evs[0].magnitude_source() == "fallback_pct"


# ---------- detector outputs round-trip through CsvEventStore --------------

def test_detector_extras_persist_through_csv(tmp_path: Path):
    det = EarningsSurpriseDetector()
    ev = det.detect({
        "ticker": "AAPL", "market": "US", "announce_date": "2026-04-30",
        "actual_eps": 1.20, "consensus_eps": 1.00, "consensus_stdev": 0.05,
    })[0]
    store = CsvEventStore(tmp_path / "events.csv")
    store.append([ev])

    reopened = CsvEventStore(tmp_path / "events.csv")
    out = reopened.all()
    assert len(out) == 1
    assert out[0].magnitude_source() == "sue"
    assert out[0].extras.get("consensus_stdev") == pytest.approx(0.05)

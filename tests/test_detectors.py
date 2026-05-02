from catalyst.events.detectors import (
    AnalystTargetDetector,
    DartCapitalRaiseDetector,
    DartContractDetector,
    EarningsSurpriseDetector,
    NewsHeadlineDetector,
)
from catalyst.types import EventType, Market


def test_dart_contract_detector_emits_event():
    det = DartContractDetector()
    payload = {
        "report_nm": "단일판매·공급계약체결",
        "stock_code": "005930",
        "rcept_dt": "2026-04-01",
        "contract_amount": 1.5e11,
        "market_cap": 5.0e12,
    }
    out = det.detect(payload)
    assert len(out) == 1
    ev = out[0]
    assert ev.event_type == EventType.CONTRACT_WIN
    assert ev.market == Market.KR
    assert ev.ticker == "005930"
    assert abs(ev.magnitude - 0.03) < 1e-6


def test_dart_contract_ignores_unrelated_filings():
    det = DartContractDetector()
    assert det.detect({"report_nm": "분기보고서", "stock_code": "005930"}) == []


def test_dart_capital_raise_detector_classifies_correctly():
    det = DartCapitalRaiseDetector()
    out = det.detect({
        "report_nm": "유상증자결정",
        "stock_code": "012450",
        "rcept_dt": "2026-04-15",
        "offering_size": 5.0e10,
        "market_cap": 1.0e12,
    })
    assert len(out) == 1
    assert out[0].event_type == EventType.CAPITAL_RAISE


def test_earnings_surprise_detector_beat_and_miss():
    det = EarningsSurpriseDetector()
    beat = det.detect({
        "ticker": "AAPL", "market": "US", "announce_date": "2026-04-30",
        "actual_eps": 1.20, "consensus_eps": 1.00,
    })[0]
    assert beat.event_type == EventType.EARNINGS_BEAT
    assert abs(beat.magnitude - 0.20) < 1e-6

    miss = det.detect({
        "ticker": "AAPL", "market": "US", "announce_date": "2026-04-30",
        "actual_eps": 0.80, "consensus_eps": 1.00,
    })[0]
    assert miss.event_type == EventType.EARNINGS_MISS


def test_analyst_target_detector_emits_target_and_rating():
    det = AnalystTargetDetector()
    out = det.detect({
        "ticker": "NVDA", "market": "US", "date": "2026-04-30",
        "broker": "TestCo", "old_target": 100, "new_target": 130,
        "old_rating": "hold", "new_rating": "buy",
    })
    types = {e.event_type for e in out}
    assert EventType.ANALYST_TARGET_UP in types
    assert EventType.ANALYST_RATING_UP in types


def test_analyst_target_no_event_on_downgrade():
    det = AnalystTargetDetector()
    out = det.detect({
        "ticker": "NVDA", "market": "US", "date": "2026-04-30",
        "old_target": 130, "new_target": 100,
    })
    assert out == []


def test_news_headline_detector_korean():
    det = NewsHeadlineDetector(market=Market.KR)
    out = det.detect({"ticker": "005930", "date": "2026-04-30",
                      "headline": "삼성전자, 1조원 규모 수주 공시"})
    assert len(out) >= 1
    assert any(e.event_type == EventType.CONTRACT_WIN for e in out)


def test_news_headline_detector_english():
    det = NewsHeadlineDetector(market=Market.US)
    out = det.detect({"ticker": "AAPL", "date": "2026-04-30",
                      "headline": "Apple awarded contract by Pentagon"})
    assert any(e.event_type == EventType.CONTRACT_WIN for e in out)

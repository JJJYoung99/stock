from catalyst.config import load_rules
from catalyst.types import EventType


def test_load_default_rules():
    cfg = load_rules()
    assert EventType.CONTRACT_WIN in cfg.events
    rule = cfg.events[EventType.CONTRACT_WIN]
    assert rule.weight > 0
    assert rule.horizon_days > 0
    assert rule.magnitude_scale > 0


def test_negative_event_has_negative_weight():
    cfg = load_rules()
    assert cfg.events[EventType.CAPITAL_RAISE].weight < 0
    assert cfg.events[EventType.REGULATORY_PROBE].weight < 0


def test_posterior_priors_sane():
    cfg = load_rules()
    p = cfg.posterior
    assert p.alpha_prior > 0 and p.beta_prior > 0
    assert 0 < p.magnitude_band < 1

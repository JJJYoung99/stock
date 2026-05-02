"""YAML rule-config loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from catalyst.types import EventType


@dataclass(frozen=True)
class EventRule:
    weight: float
    horizon_days: int
    min_magnitude: float
    magnitude_scale: float
    magnitude_cap: float
    description: str = ""


@dataclass(frozen=True)
class AggregationRule:
    decay: float = 0.6
    cap: float = 5.0


@dataclass(frozen=True)
class PosteriorRule:
    alpha_prior: float = 2.0
    beta_prior: float = 2.0
    min_analogs: int = 5
    magnitude_band: float = 0.5


@dataclass(frozen=True)
class RunupRule:
    coef: float = 1.5
    floor: float = 0.2


@dataclass(frozen=True)
class MagnitudeOverride:
    scale: float
    cap: float


@dataclass(frozen=True)
class RuleConfig:
    events: dict[EventType, EventRule]
    aggregation: AggregationRule
    posterior: PosteriorRule
    runup: RunupRule = RunupRule()
    use_abnormal_returns: bool = True
    magnitude_source_overrides: dict[EventType, dict[str, MagnitudeOverride]] = \
        field(default_factory=dict)
    default_horizon_days: int = 20

    def for_event(self, et: EventType) -> EventRule | None:
        return self.events.get(et)

    def magnitude_override(self, et: EventType, source: str) -> MagnitudeOverride | None:
        return self.magnitude_source_overrides.get(et, {}).get(source)


_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "event_rules.yaml"


def load_rules(path: str | Path | None = None) -> RuleConfig:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"rule config not found: {p}")
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    events: dict[EventType, EventRule] = {}
    for name, cfg in (raw.get("events") or {}).items():
        try:
            et = EventType(name)
        except ValueError:
            # unknown event type in config - skip rather than crash
            continue
        events[et] = EventRule(
            weight=float(cfg["weight"]),
            horizon_days=int(cfg.get("horizon_days", raw.get("default_horizon_days", 20))),
            min_magnitude=float(cfg.get("min_magnitude", 0.0)),
            magnitude_scale=float(cfg.get("magnitude_scale", 1.0)),
            magnitude_cap=float(cfg.get("magnitude_cap", 1.0)),
            description=str(cfg.get("description", "")),
        )

    agg_raw = raw.get("aggregation") or {}
    aggregation = AggregationRule(
        decay=float(agg_raw.get("decay", 0.6)),
        cap=float(agg_raw.get("cap", 5.0)),
    )

    post_raw = raw.get("posterior") or {}
    posterior = PosteriorRule(
        alpha_prior=float(post_raw.get("alpha_prior", 2.0)),
        beta_prior=float(post_raw.get("beta_prior", 2.0)),
        min_analogs=int(post_raw.get("min_analogs", 5)),
        magnitude_band=float(post_raw.get("magnitude_band", 0.5)),
    )

    runup_raw = raw.get("pre_event_runup") or {}
    runup = RunupRule(
        coef=float(runup_raw.get("coef", 1.5)),
        floor=float(runup_raw.get("floor", 0.2)),
    )

    overrides_raw = raw.get("magnitude_source_overrides") or {}
    overrides: dict[EventType, dict[str, MagnitudeOverride]] = {}
    for et_name, src_map in overrides_raw.items():
        try:
            et = EventType(et_name)
        except ValueError:
            continue
        per_source: dict[str, MagnitudeOverride] = {}
        for src, vals in (src_map or {}).items():
            per_source[str(src)] = MagnitudeOverride(
                scale=float(vals.get("scale", 1.0)),
                cap=float(vals.get("cap", 1.0)),
            )
        overrides[et] = per_source

    return RuleConfig(
        events=events,
        aggregation=aggregation,
        posterior=posterior,
        runup=runup,
        use_abnormal_returns=bool(raw.get("use_abnormal_returns", True)),
        magnitude_source_overrides=overrides,
        default_horizon_days=int(raw.get("default_horizon_days", 20)),
    )

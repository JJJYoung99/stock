"""Event ingestion contracts.

Two roles:

* `EventDetector` extracts `Event` records from a raw payload (a DART
  disclosure dict, an SEC 8-K, a news headline, ...). Detectors are pure
  functions: payload in, list of events out.

* `EventStore` is the historical archive of detected events. The scoring
  engine queries it for "analog" events when computing the empirical
  posterior. Stores must be append-only and time-ordered to support
  walk-forward backtesting (no look-ahead).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Iterable

from catalyst.types import Event, EventType, Market


class EventDetector(ABC):
    """Stateless extractor: raw payload -> 0..N events."""

    source: str

    @abstractmethod
    def detect(self, payload: dict[str, Any]) -> list[Event]:
        ...


class EventStore(ABC):
    """Persistent archive of detected events. Read-mostly during backtest."""

    @abstractmethod
    def append(self, events: Iterable[Event]) -> None:
        ...

    @abstractmethod
    def query(
        self,
        *,
        market: Market | None = None,
        event_type: EventType | None = None,
        ticker: str | None = None,
        before: date | None = None,
        since: date | None = None,
    ) -> list[Event]:
        """Return events matching all provided filters. `before` is exclusive
        (strict <), `since` is inclusive (>=). The combination of `before`
        and the calling backtester guarantees no look-ahead."""

    @abstractmethod
    def all(self) -> list[Event]:
        ...

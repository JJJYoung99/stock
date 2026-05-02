"""CSV-backed EventStore for offline use, tests, and the bundled examples.

Schema:
    market,ticker,event_type,occurred_at,magnitude,source,confidence,raw_json
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from catalyst.events.base import EventStore
from catalyst.types import Event, EventType, Market


class CsvEventStore(EventStore):
    columns = [
        "market", "ticker", "event_type", "occurred_at",
        "magnitude", "source", "confidence", "raw_json",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._df: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            df = pd.read_csv(
                self.path,
                parse_dates=["occurred_at"],
                dtype={"ticker": str, "market": str, "event_type": str, "source": str},
            )
            for col in self.columns:
                if col not in df.columns:
                    df[col] = None
            self._df = df[self.columns]
        else:
            self._df = pd.DataFrame(columns=self.columns)

    def _row_to_event(self, row: pd.Series) -> Event:
        raw = {}
        if isinstance(row["raw_json"], str) and row["raw_json"]:
            try:
                raw = json.loads(row["raw_json"])
            except json.JSONDecodeError:
                raw = {"_unparsed": row["raw_json"]}
        confidence = row.get("confidence")
        return Event(
            ticker=str(row["ticker"]),
            market=Market(str(row["market"])),
            event_type=EventType(str(row["event_type"])),
            occurred_at=pd.Timestamp(row["occurred_at"]).date(),
            magnitude=float(row["magnitude"]),
            source=str(row["source"]),
            raw=raw,
            confidence=float(confidence) if pd.notna(confidence) else 1.0,
        )

    def append(self, events: Iterable[Event]) -> None:
        rows = []
        for ev in events:
            rows.append({
                "market": ev.market.value,
                "ticker": ev.ticker,
                "event_type": ev.event_type.value,
                "occurred_at": pd.Timestamp(ev.occurred_at),
                "magnitude": ev.magnitude,
                "source": ev.source,
                "confidence": ev.confidence,
                "raw_json": json.dumps(ev.raw, ensure_ascii=False) if ev.raw else "",
            })
        if not rows:
            return
        new = pd.DataFrame(rows, columns=self.columns)
        self._df = pd.concat([self._df, new], ignore_index=True) if self._df is not None else new
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.path, index=False)

    def query(
        self,
        *,
        market: Market | None = None,
        event_type: EventType | None = None,
        ticker: str | None = None,
        before: date | None = None,
        since: date | None = None,
    ) -> list[Event]:
        df = self._df
        if df is None or df.empty:
            return []
        mask = pd.Series(True, index=df.index)
        if market:
            mask &= df["market"] == market.value
        if event_type:
            mask &= df["event_type"] == event_type.value
        if ticker:
            mask &= df["ticker"] == ticker
        if before:
            mask &= df["occurred_at"] < pd.Timestamp(before)
        if since:
            mask &= df["occurred_at"] >= pd.Timestamp(since)
        return [self._row_to_event(row) for _, row in df[mask].iterrows()]

    def all(self) -> list[Event]:
        if self._df is None or self._df.empty:
            return []
        return [self._row_to_event(row) for _, row in self._df.iterrows()]

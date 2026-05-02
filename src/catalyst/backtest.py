"""Walk-forward event-study backtester.

Procedure:

    1. Sort all stored events by occurred_at ascending.
    2. For each event, compute a Score using ONLY analogs that occurred
       strictly earlier (the EventStore is filtered by `before=as_of`).
    3. Decide entry: enter when aggregated rule_score >= `entry_threshold`
       AND posterior P(profit) >= `min_prob_profit` AND n_analogs >=
       `min_analogs`.
    4. Hold for `horizon_days` trading days, exit at close.
    5. Track per-trade return, equity curve, and the standard performance
       metrics (CAGR, Sharpe, hit rate, max DD, expectancy).

Position sizing is intentionally trivial (equal-weight per trade). The
framework exposes the per-event scores so a user can plug a portfolio
optimizer (skfolio, riskfolio-lib) on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from catalyst.events.base import EventStore
from catalyst.markets.base import MarketAdapter
from catalyst.scoring import Scorer
from catalyst.types import Event, ScoreReport
from catalyst.windowing import window_returns


@dataclass(frozen=True)
class BacktestConfig:
    entry_threshold: float = 0.5      # min aggregated rule_score to enter
    min_prob_profit: float = 0.50     # min posterior probability of profit
    min_analogs: int = 5              # require enough history before trading
    cost_bps: float = 20.0            # round-trip transaction cost (bps)
    holding_horizon: int | None = None  # override horizon_days if set


@dataclass
class Trade:
    event: Event
    entry_date: date
    exit_date: date
    horizon_days: int
    pnl: float
    rule_score: float
    prob_profit: float
    expected_return: float
    n_analogs: int


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    per_event_reports: list[ScoreReport] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def metrics(self) -> dict[str, float]:
        if not self.trades:
            return {
                "n_trades": 0, "hit_rate": 0.0, "avg_pnl": 0.0,
                "expectancy": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "cagr": 0.0, "total_return": 0.0,
            }
        pnls = np.array([t.pnl for t in self.trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        hit_rate = float(len(wins)) / float(len(pnls))
        avg_win = float(wins.mean()) if wins.size else 0.0
        avg_loss = float(losses.mean()) if losses.size else 0.0
        expectancy = hit_rate * avg_win + (1 - hit_rate) * avg_loss

        equity = self.equity_curve.values.astype(float)
        if equity.size > 1:
            running_max = np.maximum.accumulate(equity)
            dd = equity / running_max - 1.0
            max_dd = float(dd.min())
            total_return = float(equity[-1] / equity[0] - 1.0)
            # Annualization assumes ~252 trading days, indexed by exit_date.
            n_days = max((self.equity_curve.index[-1] - self.equity_curve.index[0]).days, 1)
            years = n_days / 365.25
            cagr = float((equity[-1] / equity[0]) ** (1 / years) - 1.0) if years > 0 else 0.0
            daily = pd.Series(equity).pct_change().dropna()
            sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
        else:
            max_dd = total_return = cagr = sharpe = 0.0

        return {
            "n_trades": float(len(pnls)),
            "hit_rate": hit_rate,
            "avg_pnl": float(pnls.mean()),
            "expectancy": float(expectancy),
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "cagr": cagr,
            "total_return": total_return,
        }

    def trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "ticker", "event_type", "entry_date", "exit_date",
                "horizon_days", "pnl", "rule_score", "prob_profit",
                "expected_return", "n_analogs",
            ])
        rows = []
        for t in self.trades:
            rows.append({
                "ticker": t.event.ticker,
                "event_type": t.event.event_type.value,
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "horizon_days": t.horizon_days,
                "pnl": round(t.pnl, 6),
                "rule_score": round(t.rule_score, 4),
                "prob_profit": round(t.prob_profit, 4),
                "expected_return": round(t.expected_return, 4),
                "n_analogs": t.n_analogs,
            })
        return pd.DataFrame(rows)


class Backtester:
    def __init__(
        self,
        adapter: MarketAdapter,
        store: EventStore,
        scorer: Scorer | None = None,
        config: BacktestConfig | None = None,
    ):
        self.adapter = adapter
        self.store = store
        self.scorer = scorer or Scorer(adapter, store)
        self.config = config or BacktestConfig()

    def run(self, events: Iterable[Event] | None = None) -> BacktestResult:
        evs = sorted(events or self.store.all(), key=lambda e: e.occurred_at)
        if not evs:
            return BacktestResult(trades=[], equity_curve=pd.Series(dtype=float))

        # Group concurrent events on (ticker, date) so they are scored together.
        grouped: dict[tuple[str, date], list[Event]] = {}
        for ev in evs:
            grouped.setdefault((ev.ticker, ev.occurred_at), []).append(ev)

        trades: list[Trade] = []
        reports: list[ScoreReport] = []
        cost = self.config.cost_bps / 10_000.0

        # Sort groups by date so the walk-forward order is correct.
        for (ticker, when), batch in sorted(grouped.items(), key=lambda kv: kv[0][1]):
            market = batch[0].market
            report = self.scorer.score_pending(batch, as_of=when)
            reports.append(report)

            if not self._should_enter(report):
                continue

            horizon = self.config.holding_horizon or report.horizon_days
            try:
                buf_end = (pd.Timestamp(when) + pd.Timedelta(days=horizon * 2 + 14)).date()
                prices = self.adapter.prices(ticker, when, buf_end)
            except Exception:
                continue
            if prices is None or prices.empty:
                continue
            prices.index = pd.to_datetime(prices.index)
            wr = window_returns(prices, when, horizon)
            if wr is None:
                continue

            pnl = wr.cum_return - cost
            exit_idx = min(horizon, wr.n_bars - 1)
            exit_date_ts = prices.index[exit_idx]
            exit_dt = exit_date_ts.date() if hasattr(exit_date_ts, "date") else exit_date_ts

            for ev, sc in zip(batch, report.per_event):
                trades.append(Trade(
                    event=ev,
                    entry_date=when,
                    exit_date=exit_dt,
                    horizon_days=horizon,
                    pnl=float(pnl) / max(len(batch), 1),  # split equal across concurrent events
                    rule_score=sc.rule_score,
                    prob_profit=sc.prob_profit,
                    expected_return=sc.expected_return,
                    n_analogs=sc.n_analogs,
                ))

        equity = self._equity_curve(trades)
        return BacktestResult(trades=trades, equity_curve=equity, per_event_reports=reports)

    def _should_enter(self, report: ScoreReport) -> bool:
        c = self.config
        # Skip entries where the posterior is prior-dominated.
        if all(s.n_analogs < c.min_analogs for s in report.per_event):
            return False
        if report.aggregated_rule_score < c.entry_threshold:
            return False
        if report.aggregated_prob_profit < c.min_prob_profit:
            return False
        return True

    @staticmethod
    def _equity_curve(trades: list[Trade]) -> pd.Series:
        if not trades:
            return pd.Series(dtype=float)
        # Aggregate by exit_date so concurrent exits compound on the same day.
        df = pd.DataFrame([{"date": t.exit_date, "pnl": t.pnl} for t in trades])
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date")["pnl"].sum().sort_index()
        equity = (1.0 + daily).cumprod()
        return equity

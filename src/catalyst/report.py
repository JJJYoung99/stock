"""Pretty-printers for ScoreReport and BacktestResult."""
from __future__ import annotations

import json
from typing import Any

from catalyst.types import ScoreReport


def _pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def render_score_report(report: ScoreReport) -> str:
    """Human-readable, single-screen summary."""
    lines: list[str] = []
    lines.append(f"=== {report.market.value} {report.ticker}  as_of={report.as_of}  horizon={report.horizon_days}d ===")
    if not report.events:
        lines.append("(no events on this date)")
        return "\n".join(lines)

    lines.append("Events:")
    for ev, sc in zip(report.events, report.per_event):
        lines.append(
            f"  - {ev.event_type.value:<20} mag={ev.magnitude:+.4f} "
            f"src={ev.source:<10} rule={sc.rule_score:+.3f} "
            f"P(profit)={sc.prob_profit*100:5.1f}% E[r]={_pct(sc.expected_return)} "
            f"n_analogs={sc.n_analogs}"
        )
    lines.append("")
    lines.append("Aggregated:")
    lines.append(f"  rule_score        : {report.aggregated_rule_score:+.3f}")
    lines.append(f"  P(profit)         : {report.aggregated_prob_profit*100:5.1f}%")
    lines.append(f"  E[return,{report.horizon_days}d] : {_pct(report.aggregated_expected_return)}")
    lines.append(f"  VaR  5%           : {_pct(report.aggregated_var_95)}")
    lines.append(f"  CVaR 5%           : {_pct(report.aggregated_cvar_95)}")

    edge = report.aggregated_rule_score * (report.aggregated_prob_profit - 0.5) * 2.0
    verdict = "BULLISH" if edge > 0.2 else "BEARISH" if edge < -0.2 else "NEUTRAL"
    lines.append(f"  edge              : {edge:+.3f}  -> {verdict}")
    if report.notes:
        lines.append("Notes:")
        for n in report.notes:
            lines.append(f"  * {n}")
    return "\n".join(lines)


def report_to_json(report: ScoreReport) -> str:
    payload: dict[str, Any] = report.to_row()
    payload["per_event"] = [
        {
            "event_type": ev.event_type.value,
            "magnitude": ev.magnitude,
            "source": ev.source,
            "confidence": ev.confidence,
            "rule_score": sc.rule_score,
            "prob_profit": sc.prob_profit,
            "expected_return": sc.expected_return,
            "var_95": sc.var_95,
            "cvar_95": sc.cvar_95,
            "max_drawdown": sc.max_drawdown,
            "n_analogs": sc.n_analogs,
            "horizon_days": sc.horizon_days,
        }
        for ev, sc in zip(report.events, report.per_event)
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_backtest_metrics(metrics: dict[str, float]) -> str:
    return (
        f"trades={int(metrics['n_trades'])}  hit_rate={metrics['hit_rate']*100:5.2f}%  "
        f"avg_pnl={_pct(metrics['avg_pnl'])}  expectancy={_pct(metrics['expectancy'])}\n"
        f"total_return={_pct(metrics['total_return'])}  CAGR={_pct(metrics['cagr'])}  "
        f"Sharpe={metrics['sharpe']:.2f}  maxDD={_pct(metrics['max_drawdown'])}"
    )

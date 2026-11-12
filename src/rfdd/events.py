"""Alarm logic and event-level scoring.

A detector gives one score per minute. An operator does not see scores, they see alarms, so everything is
scored on alarms: was each fault EVENT caught, how long did it take, and how many alarms fired on days when
nothing was wrong. Row-level accuracy is never used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    """One labelled fault period."""

    day: str
    fault: str
    severity: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


@dataclass(frozen=True)
class AlarmLogic:
    """k-of-n persistence: alarm when the score beat the threshold in at least k of the last n minutes.

    Minutes where the score is NaN (masked states such as defrost) never vote for an alarm.
    """

    k: int = 20
    n: int = 30
    merge_gap: int = 30  # alarm episodes closer than this many minutes count as one

    def alarms(self, score: pd.Series, threshold: float) -> pd.Series:
        over = (score > threshold).astype(float)  # NaN > x is False
        votes = over.rolling(self.n, min_periods=1).sum()
        return (votes >= self.k).rename("alarm")


def episodes(alarm: pd.Series, merge_gap: int = 30) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous alarm runs, merged when separated by fewer than `merge_gap` minutes."""
    a = alarm.to_numpy(dtype=bool)
    if not a.any():
        return []
    idx = alarm.index
    edges = np.diff(np.concatenate([[0], a.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in zip(starts, ends, strict=True):
        if out and (idx[s] - out[-1][1]) <= pd.Timedelta(minutes=merge_gap):
            out[-1] = (out[-1][0], idx[e])
        else:
            out.append((idx[s], idx[e]))
    return out


def score_event(alarm: pd.Series, event: Event) -> dict[str, object]:
    """Detection and delay for one event. Delay is minutes from fault start to the first alarm minute."""
    window = alarm.loc[event.start : event.end]
    hit = window[window]
    detected = len(hit) > 0
    return {
        "day": event.day,
        "fault": event.fault,
        "severity": event.severity,
        "minutes": event.minutes,
        "detected": detected,
        "delay_min": (hit.index[0] - event.start).total_seconds() / 60.0 if detected else np.nan,
        "alarm_fraction": float(window.mean()) if len(window) else 0.0,
    }


def false_alarms(alarm: pd.Series, merge_gap: int = 30) -> dict[str, float]:
    """False-alarm statistics for a fault-free period."""
    days = max(len(alarm) / 1440.0, 1e-9)
    return {
        "episodes": float(len(episodes(alarm, merge_gap))),
        "per_day": len(episodes(alarm, merge_gap)) / days,
        "alarm_minutes_per_day": float(alarm.sum()) / days,
        "days": days,
    }

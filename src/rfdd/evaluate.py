"""Leave-one-normal-day-out evaluation, scored per fault event.

For each fault-free day in turn: train every detector on the other fault-free days, then score the held-out
normal day (false alarms) and all fault days (detections). Thresholds are set from training days only.
A fault day is therefore scored once per fold, each time by a detector trained on a different set of days,
which shows how much the result depends on which normal days happened to be available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import models
from .events import AlarmLogic, Event, episodes
from .faults import GROUP_OF
from .prepare import Day

METHODS = ["rules", "raw", "hybrid", "temp_alarm"]


@dataclass
class FoldScores:
    """Normalised scores of one detector in one fold: day -> frame of per-group scores."""

    method: str
    held_out: str
    scores: dict[str, pd.DataFrame]


def run_folds(
    days: dict[str, Day], logic: AlarmLogic, methods: list[str] | None = None, held_out: list[str] | None = None
) -> list[FoldScores]:
    normal = [k for k, d in days.items() if d.fault == "normal"]
    faulty = [k for k, d in days.items() if d.fault != "normal"]
    out = []
    for held in held_out or normal:
        train = [days[k] for k in normal if k != held]
        for m in methods or METHODS:
            det = models.make(m).fit(train, logic)
            out.append(FoldScores(m, held, {k: det.score(days[k]) for k in [held, *faulty]}))
    return out


def alarm_series(scores: pd.DataFrame, logic: AlarmLogic, threshold: float) -> pd.Series:
    return models.persist(scores.max(axis=1), logic) > threshold


def diagnose(scores: pd.DataFrame, logic: AlarmLogic, threshold: float) -> pd.Series:
    """Named fault per minute: the fault group furthest past its own threshold, else "unknown"."""
    named = scores.drop(columns=[models.UNKNOWN], errors="ignore")
    level = pd.DataFrame({c: models.persist(named[c], logic) for c in named.columns})
    top = level.fillna(-np.inf).idxmax(axis=1)
    return top.where(level.max(axis=1) > threshold, models.UNKNOWN)


def event_diagnosis(
    scores: pd.DataFrame, alarm: pd.Series, event: Event, logic: AlarmLogic, threshold: float
) -> str | None:
    """What the detector called the fault: the most frequent named diagnosis over the alarmed minutes of the
    event, or "unknown" if no named fault ever passed its threshold. None for detectors that cannot name one."""
    hit = alarm.loc[event.start : event.end]
    hit = hit[hit]
    if hit.empty or scores.shape[1] == 1:
        return None
    names = diagnose(scores, logic, threshold).loc[hit.index]
    named = names[names != models.UNKNOWN]
    return str(named.mode().iloc[0]) if len(named) else models.UNKNOWN


def event_table(
    folds: list[FoldScores], days: dict[str, Day], events: list[Event], logic: AlarmLogic, margin: float
) -> pd.DataFrame:
    """One row per (method, fold, event)."""
    rows = []
    for fs in folds:
        for ev in events:
            sc = fs.scores[ev.day]
            alarm = alarm_series(sc, logic, margin)
            window = alarm.loc[ev.start : ev.end]
            hit = window[window]
            diag = event_diagnosis(sc, alarm, ev, logic, margin)
            rows.append(
                {
                    "method": fs.method,
                    "fold": fs.held_out,
                    "day": ev.day,
                    "fault": ev.fault,
                    "detected": bool(len(hit)),
                    "delay_min": (hit.index[0] - ev.start).total_seconds() / 60.0 if len(hit) else np.nan,
                    "alarm_fraction": float(window.mean()),
                    "diagnosis": diag,
                    "diagnosis_ok": (diag == GROUP_OF[ev.fault]) if diag is not None else np.nan,
                    # Caught for the right reason: no wrong fault named. Detectors that cannot name a fault
                    # (black boxes) and "unknown" verdicts get the benefit of the doubt.
                    "right_reason": bool(len(hit)) and diag in (None, models.UNKNOWN, GROUP_OF[ev.fault]),
                    "peak": float(models.persist(sc.max(axis=1), logic).loc[ev.start : ev.end].max()),
                }
            )
    return pd.DataFrame(rows)


def false_alarm_table(folds: list[FoldScores], days: dict[str, Day], logic: AlarmLogic, margin: float) -> pd.DataFrame:
    """One row per (method, held-out normal day). Excluded (anomalous) minutes are counted separately."""
    rows = []
    for fs in folds:
        day = days[fs.held_out]
        alarm = alarm_series(fs.scores[fs.held_out], logic, margin)
        clean = alarm & day.usable
        rows.append(
            {
                "method": fs.method,
                "day": fs.held_out,
                "days": float(day.usable.mean()),
                "episodes": len(episodes(clean, logic.merge_gap)),
                "alarm_minutes": int(clean.sum()),
                "excluded_minutes": int((~day.usable).sum()),
                "excluded_alarm_minutes": int((alarm & ~day.usable).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame, fa: pd.DataFrame) -> pd.DataFrame:
    """Headline table: one row per method."""
    rows = []
    for m in ev["method"].unique():
        e, f = ev[ev["method"] == m], fa[fa["method"] == m]
        per_event = e.groupby("day")["detected"].mean()
        diag = e.loc[e["detected"], "diagnosis_ok"].dropna()
        rows.append(
            {
                "method": m,
                "events_caught_all_folds": int((per_event == 1.0).sum()),
                "events_caught_right_reason": int((e.groupby("day")["right_reason"].mean() == 1.0).sum()),
                "events_caught_any_fold": int((per_event > 0.0).sum()),
                "events": int(per_event.size),
                "detection_rate": float(e["detected"].mean()),
                "median_delay_min": float(e["delay_min"].median()),
                "false_alarms_per_day": float(f["episodes"].sum() / f["days"].sum()),
                "false_alarm_minutes_per_day": float(f["alarm_minutes"].sum() / f["days"].sum()),
                "normal_days_with_alarm": int((f["episodes"] > 0).sum()),
                "normal_days": len(f),
                "diagnosis_accuracy": float(diag.mean()) if len(diag) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sweep(
    folds: list[FoldScores], days: dict[str, Day], events: list[Event], logic: AlarmLogic, margins: list[float]
) -> pd.DataFrame:
    """Detection rate against false alarms per day across threshold margins."""
    rows = []
    for margin in margins:
        s = summarise(event_table(folds, days, events, logic, margin), false_alarm_table(folds, days, logic, margin))
        s.insert(1, "margin", margin)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)

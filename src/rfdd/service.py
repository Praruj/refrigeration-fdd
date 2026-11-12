"""Scoring a live window of data with the saved models. Shared by the API and the demo.

Input is the tidy signal set (SI units, names from `signals.py`) at 1-minute resolution. At least 30 minutes
are needed before any feature is valid, and a few hours give the defrost-state logic some history.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import features, models, sensors
from .evaluate import diagnose
from .faults import BY_KEY, GROUP_OF
from .prepare import Day

REQUIRED = [
    "t_amb", "t_gc_air_out1", "t_gc_air_out2", "t_gc_out", "t_room", "t_mt_dis", "t_lt_dis",
    "t_ltc_sup", "t_ltc_ret", "t_ltc_eev", "t_ltc_suc", "t_mtc_sup", "t_mtc_ret", "t_mtc_eev", "t_mtc_suc",
    "p_gc", "p_mt_dis", "p_mt_suc", "p_mtc_suc", "p_ltc_suc", "w_mt1", "w_mt2", "w_mt3", "w_lt1", "w_lt2",
    "lt_prod", "mt_prod", "mt_prod_spread",
]  # fmt: skip
# Redundant pressure taps: optional, but when sent they let the sensor check vote out a bad transducer.
OPTIONAL = ["p_mtc_suc2", "p_ltc_suc2", "p_lt_suc", "p_flash", "p_liq"]


@dataclass
class Bundle:
    hybrid: models.HybridDetector
    rules: models.RuleDetector
    logic: Any
    margin: float


def load_bundle(path: Path) -> Bundle:
    with path.open("rb") as f:
        b = pickle.load(f)
    return Bundle(b["hybrid"], b["rules"], b["logic"], b["margin"])


def make_day(tidy: pd.DataFrame, check: bool = True) -> Day:
    tidy = tidy.copy()
    for col in REQUIRED:
        if col not in tidy:
            tidy[col] = np.nan
    bad = sensors.check(tidy)
    if not check:
        bad = bad & False
    clean = sensors.apply(tidy, bad)
    table = features.build(clean)
    return Day("live", "unknown", tidy, clean, bad, table, pd.Series(True, index=tidy.index))


def score(bundle: Bundle, tidy: pd.DataFrame) -> dict[str, Any]:
    """Alarm state, diagnosis and the evidence behind it, at the last minute of the window."""
    day = make_day(tidy)
    scores = bundle.hybrid.score(day)
    level = models.persist(scores.max(axis=1), bundle.logic)
    alarm = bool(level.iloc[-1] > bundle.margin) if pd.notna(level.iloc[-1]) else False
    named = diagnose(scores, bundle.logic, bundle.margin).iloc[-1]
    z = bundle.hybrid.z(day).iloc[-1]
    expected = bundle.hybrid.expected(day).iloc[-1]
    last = day.table.iloc[-1]
    evidence = [
        {
            "feature": f,
            "meaning": features.BY_NAME[f].meaning,
            "value": _num(last[f]),
            "expected": _num(expected[f]),
            "unit": features.BY_NAME[f].unit,
            "z": _num(z[f]),
        }
        for f in z.abs().sort_values(ascending=False).dropna().index[:5]
    ]
    rules_level = models.persist(bundle.rules.score(day).max(axis=1), bundle.logic).iloc[-1]
    return {
        "time": str(day.table.index[-1]),
        "alarm": alarm,
        "alarm_level": _num(level.iloc[-1]),
        "threshold": bundle.margin,
        "diagnosis": named if alarm else None,
        "diagnosis_name": _group_name(named) if alarm else None,
        "group_scores": {k: _num(v) for k, v in scores.iloc[-1].items()},
        "rules_alarm": bool(pd.notna(rules_level) and rules_level > bundle.margin),
        "states": {"lt_case": str(last["lt_state"]), "mt_case": str(last["mt_state"])},
        "untrusted_sensors": [c for c, v in day.bad.iloc[-1].items() if v],
        "evidence": evidence,
    }


def _group_name(group: str) -> str:
    names = [f.name for k, f in BY_KEY.items() if GROUP_OF[k] == group]
    return " / ".join(names) if names else "abnormal, not in the fault table"


def _num(v: object) -> float | None:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, 4)

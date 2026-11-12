"""Sensor-fault checks, run before any process-fault logic.

A drifting probe looks exactly like a sick machine to a model. Three plain checks catch most of it:

1. range     - the reading is outside what the sensor can physically see;
2. flatline  - the reading has not moved at all for 30 minutes;
3. agreement - two or three sensors that measure nearly the same thing disagree, or a reading breaks a
               physical law (suction gas colder than its own saturation temperature, case air warmer than
               the room, refrigerant leaving the gas cooler colder than the air cooling it).

A flagged sensor is removed (set to NaN) and every feature that needs it goes silent instead of alarming.
The checks use fixed engineering tolerances, so nothing is fitted and nothing can leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import co2

RANGE = {"t_": (-60.0, 150.0), "p_": (300.0, 14000.0), "w_": (-0.5, 30.0), "m_": (-1.0, 30.0)}
FLAT_MINUTES = 30

# Groups of pressure taps on the same line. With three taps the odd one out is identified by voting.
PRESSURE_GROUPS = [
    ("p_ltc_suc", "p_ltc_suc2", "p_lt_suc"),
    ("p_mtc_suc", "p_mtc_suc2", "p_mt_suc"),
    ("p_gc", "p_mt_dis"),
    ("p_flash", "p_liq"),
]
PRESSURE_TOL = (0.03, 30.0)  # relative, absolute kPa
AIR_PAIR = ("t_gc_air_out1", "t_gc_air_out2")
AIR_TOL = 3.0  # K
LAW_TOL = 2.0  # K of slack on the physical-law checks
LAW_MINUTES = 10  # this many violating minutes within 30 latch the sensor out

# Which sensors each physics feature needs.
NEEDS = {
    "lt_sh": ["t_ltc_suc", "p_ltc_suc"],
    "lt_td": ["t_ltc_ret", "p_ltc_suc"],
    "lt_air": ["t_ltc_ret"],
    "lt_prod": [],
    "lt_feed": ["t_ltc_eev", "p_ltc_suc"],
    "w_lt": ["w_lt1", "w_lt2"],
    "lt_dis_sh": ["t_lt_dis", "p_mt_suc"],
    "mt_sh": ["t_mtc_suc", "p_mtc_suc"],
    "mt_td": ["t_mtc_ret", "p_mtc_suc"],
    "mt_air_dt": ["t_mtc_ret", "t_mtc_sup"],
    "mt_prod": [],
    "mt_prod_spread": [],
    "mt_feed": ["t_mtc_eev", "p_mtc_suc"],
    "gc_app": ["t_gc_out", "t_amb"],
    "gc_air_dt": ["t_gc_air_out1", "t_gc_air_out2", "t_amb"],
    "p_gc": ["p_gc"],
    "pr_mt": ["p_mt_dis", "p_mt_suc"],
    "w_mt": ["w_mt1", "w_mt2", "w_mt3"],
    "mt_dis_sh": ["t_mt_dis", "t_amb"],
}


def _disagree(a: pd.Series, b: pd.Series) -> pd.Series:
    rel, absolute = PRESSURE_TOL
    return (a - b).abs() > (rel * a.abs().clip(lower=1.0) + absolute)


def check(tidy: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame, True where a sensor reading should not be trusted. Causal."""
    sensors = [c for c in tidy.columns if c[:2] in RANGE]
    bad = pd.DataFrame(False, index=tidy.index, columns=sensors)

    for c in sensors:
        lo, hi = RANGE[c[:2]]
        bad[c] |= tidy[c].isna() | (tidy[c] < lo) | (tidy[c] > hi)
        if c[:2] in ("t_", "p_"):
            bad[c] |= tidy[c].rolling(FLAT_MINUTES).std() < 1e-6

    for full in PRESSURE_GROUPS:
        group = tuple(g for g in full if g in tidy.columns)  # a live feed may not send every redundant tap
        if len(group) < 2:
            continue
        if len(group) == 3:
            a, b, c = (tidy[g] for g in group)
            ab, ac, bc = _disagree(a, b), _disagree(a, c), _disagree(b, c)
            bad[group[0]] |= ab & ac
            bad[group[1]] |= ab & bc
            bad[group[2]] |= ac & bc
        else:
            d = _disagree(tidy[group[0]], tidy[group[1]])
            bad[group[0]] |= d
            bad[group[1]] |= d

    d = (tidy[AIR_PAIR[0]] - tidy[AIR_PAIR[1]]).abs() > AIR_TOL
    bad[AIR_PAIR[0]] |= d
    bad[AIR_PAIR[1]] |= d

    # Physical laws. A violation cannot be a process state, so one of the sensors involved is wrong. A biased
    # probe only breaks the law part of the time (when the process happens to sit near the limit), so
    # violations on LAW_MINUTES of the last 30 minutes latch the sensor out until the window ends, the way a
    # site system keeps a sensor fault active until a technician clears it.
    law = pd.DataFrame(False, index=tidy.index, columns=sensors)
    for t_suc, p_suc in (("t_ltc_suc", "p_ltc_suc"), ("t_mtc_suc", "p_mtc_suc")):
        law[t_suc] = tidy[t_suc] < pd.Series(co2.t_sat(tidy[p_suc]), index=tidy.index) - LAW_TOL
    for t_ret in ("t_ltc_ret", "t_mtc_ret"):
        law[t_ret] = tidy[t_ret] > tidy["t_room"] + LAW_TOL
    law["t_gc_out"] = tidy["t_gc_out"] < tidy["t_amb"] - LAW_TOL
    sustained = law.astype(float).rolling(FLAT_MINUTES, min_periods=1).sum() >= LAW_MINUTES
    latched = sustained.astype(int).cummax().astype(bool)

    # Hold every flag for 30 minutes so a sensor that flickers in and out stays out.
    held = (bad | law).astype(float).rolling(FLAT_MINUTES, min_periods=1).max().astype(bool)
    return held | latched


def apply(tidy: pd.DataFrame, bad: pd.DataFrame) -> pd.DataFrame:
    """Tidy frame with untrusted readings removed.

    Where a pressure tap was voted out and a healthy tap on the same line exists, the healthy reading stands
    in for it, so the features that need that pressure keep working.
    """
    out = tidy.copy()
    out[bad.columns] = out[bad.columns].where(~bad)
    for group in PRESSURE_GROUPS:
        primary, spares = group[0], [g for g in group[1:] if g in out.columns]
        if primary not in out.columns:
            continue
        for spare in spares:
            out[primary] = out[primary].fillna(out[spare])
    return out


def summary(bad: pd.DataFrame) -> dict[str, float]:
    """Fraction of the period each sensor was flagged (only sensors flagged at all)."""
    frac = bad.mean()
    return {k: float(round(v, 3)) for k, v in frac[frac > 0].items()}


def inject(tidy: pd.DataFrame, sensor: str, kind: str, size: float, start: int) -> pd.DataFrame:
    """Copy of the frame with a synthetic sensor fault from row `start` on. Used only for evaluation."""
    out = tidy.copy()
    col = out.columns.get_loc(sensor)
    if kind == "bias":
        out.iloc[start:, col] = out.iloc[start:, col] + size
    elif kind == "stuck":
        out.iloc[start:, col] = out.iloc[start, col]
    elif kind == "drift":
        n = len(out) - start
        out.iloc[start:, col] = out.iloc[start:, col] + np.linspace(0.0, size, n)
    else:
        raise ValueError(kind)
    return out

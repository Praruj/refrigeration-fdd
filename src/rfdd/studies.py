"""Side studies that back up the main benchmark: physics check, sensor faults, and the day-fingerprint trap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from . import features, models, prepare, sensors
from .evaluate import alarm_series
from .events import AlarmLogic, Event
from .faults import BY_KEY, FAULTS
from .prepare import Day


def physics_check(det: models.HybridDetector, days: dict[str, Day], events: list[Event]) -> pd.DataFrame:
    """Did each symptom the fault table predicts show up, and in the predicted direction?

    For every (fault, symptom): the median residual during the event, in units of the model's day-to-day
    noise (z). "confirmed" needs |z| >= 1 in the predicted direction, "opposite" |z| >= 1 the other way,
    and anything smaller is "not visible".
    """
    rows = []
    for ev in events:
        z = det.z(days[ev.day]).loc[ev.start : ev.end]
        table = days[ev.day].table.loc[ev.start : ev.end]
        expected = det.expected(days[ev.day]).loc[ev.start : ev.end]
        for feat, sign in BY_KEY[ev.fault].signature.items():
            med = float(z[feat].median()) if z[feat].notna().any() else np.nan
            verdict = "no data" if np.isnan(med) else "not visible"
            if not np.isnan(med) and abs(med) >= 1.0:
                verdict = "confirmed" if np.sign(med) == sign else "opposite"
            rows.append(
                {
                    "fault": ev.fault,
                    "feature": feat,
                    "predicted": "up" if sign > 0 else "down",
                    "observed_z": med,
                    "observed_change": float((table[feat] - expected[feat]).median()),
                    "unit": features.BY_NAME[feat].unit,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


INJECTIONS = [
    # sensor, kind, size, which check should catch it
    ("p_ltc_suc", "bias", -200.0, "3-way pressure vote"),
    ("p_mtc_suc", "bias", 300.0, "3-way pressure vote"),
    ("p_gc", "bias", 600.0, "pressure pair"),
    ("t_ltc_suc", "bias", -25.0, "law: suction below saturation"),
    ("t_mtc_suc", "bias", -15.0, "law: suction below saturation"),
    ("t_mtc_ret", "bias", 15.0, "law: case air above room air"),
    ("t_gc_out", "bias", -8.0, "law: refrigerant below cooling air"),
    ("t_ltc_ret", "stuck", 0.0, "flatline"),
    ("t_gc_air_out1", "bias", 6.0, "air-temperature pair"),
    ("t_ltc_suc", "bias", 6.0, "none (no redundancy)"),
    ("t_mtc_ret", "bias", 4.0, "none (no redundancy)"),
    ("t_amb", "bias", 5.0, "none (no redundancy)"),
]


def sensor_study(
    days: dict[str, Day], minutes: dict[str, pd.DataFrame], logic: AlarmLogic, margin: float, held_out: list[str]
) -> pd.DataFrame:
    """Inject synthetic sensor faults into held-out normal days.

    For each injection: does the sensor check flag the sensor, and does the hybrid detector raise a process
    alarm, with the check switched on and off? A good outcome is "flagged, no process alarm".
    """
    from . import signals  # local import keeps the module graph simple

    normal = [k for k, d in days.items() if d.fault == "normal"]
    rows = []
    start = 360
    for held in held_out:
        det = models.HybridDetector().fit([days[k] for k in normal if k != held], logic)
        tidy_clean = signals.tidy(minutes[held])
        for sensor, kind, size, expect in INJECTIONS:
            tidy_bad = sensors.inject(tidy_clean, sensor, kind, size, start)
            bad = sensors.check(tidy_bad)
            flagged = float(bad[sensor].iloc[start + 60 :].mean())
            out: dict[str, Any] = {
                "day": held,
                "sensor": sensor,
                "kind": kind,
                "size": size,
                "expected_check": expect,
                "flagged_fraction": flagged,
                "flagged": flagged > 0.5,
            }
            for label, use_check in (("with_check", True), ("without_check", False)):
                tidy = sensors.apply(tidy_bad, bad) if use_check else tidy_bad
                day = Day(held, "normal", days[held].raw, tidy, bad, features.build(tidy), days[held].usable)
                alarm = alarm_series(det.score(day), logic, margin) & day.usable
                baseline = alarm_series(det.score(days[held]), logic, margin) & days[held].usable
                extra = alarm.iloc[start:] & ~baseline.iloc[start:]
                out[f"process_alarm_{label}"] = bool(extra.any())
            rows.append(out)
    return pd.DataFrame(rows)


def fingerprint_trap(days: dict[str, Day], seed: int = 0) -> pd.DataFrame:
    """Why row-level accuracy is not used anywhere in this project.

    A classifier on raw sensors, scored on a random split of rows, separates each fault day from its paired
    normal day almost perfectly. It separates two NORMAL days from each other just as well. So the score
    measures "which day is this", not "is there a fault".
    """
    pairs = [("door", "base_A"), ("cond", "base_E"), ("fan", "base_D"), ("evapblock", "base_A")]
    pairs += [("base_A", "base_D"), ("base_A", "base_E"), ("base_D", "base_E")]
    rows = []
    for a, b in pairs:
        if a not in days or b not in days:
            continue
        xa, xb = days[a].raw, days[b].raw
        cols = [c for c in xa.columns if xa[c].notna().all() and xb[c].notna().all()]
        x = pd.concat([xa[cols], xb[cols]]).to_numpy()
        y = np.r_[np.ones(len(xa)), np.zeros(len(xb))]
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
        clf = HistGradientBoostingClassifier(max_iter=60, random_state=seed).fit(x_tr, y_tr)
        rows.append(
            {
                "day_1": a,
                "day_2": b,
                "kind": "fault vs normal" if days[a].fault != "normal" else "normal vs normal",
                "row_accuracy": float(clf.score(x_te, y_te)),
            }
        )
    return pd.DataFrame(rows)


def state_mask_value(cfg: dict[str, Any], logic: AlarmLogic, margins: list[float], held_out: list[str]) -> pd.DataFrame:
    """Rules and hybrid with and without defrost/recovery masking, across alarm thresholds."""
    from .evaluate import run_folds, sweep

    rows = []
    for label, mask in (("with state masking", True), ("without state masking", False)):
        days = prepare.load(cfg, mask_states=mask)
        folds = run_folds(days, logic, ["rules", "hybrid"], held_out)
        s = sweep(folds, days, prepare.events(cfg, days), logic, margins)
        s.insert(0, "variant", label)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def fault_table_frame() -> pd.DataFrame:
    rows = []
    for f in FAULTS:
        ups = [k for k, v in f.signature.items() if v > 0]
        downs = [k for k, v in f.signature.items() if v < 0]
        rows.append({"fault": f.key, "name": f.name, "group": f.group, "up": ", ".join(ups), "down": ", ".join(downs)})
    return pd.DataFrame(rows)

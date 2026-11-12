"""A small synthetic stand-in for the ORNL data, used by CI and the unit tests (no 1 GB download).

It writes the same raw tags and units as the real logger (degF, psig, kW), with a daily weather swing,
compressor cycling, the real defrost schedule (LT once a day, MT every 4 hours) and three injectable faults.
The levels are set to roughly match the real rack. It is a test fixture, not a model of the plant: none of the
reported results come from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import co2, signals

MINUTES = 1440


def _psig(kpa: np.ndarray) -> np.ndarray:
    return kpa / signals.PSI_TO_KPA - signals.ATM_PSI


def _f(c: np.ndarray) -> np.ndarray:
    return c * 1.8 + 32.0


def _window(n: int, start: str, end: str) -> np.ndarray:
    h0, m0 = map(int, start.split(":"))
    h1, m1 = map(int, end.split(":"))
    w = np.zeros(n, dtype=bool)
    w[h0 * 60 + m0 : h1 * 60 + m1] = True
    return w


def day(
    date: str, t_mean: float, seed: int, fault: str = "normal", start: str = "00:00", end: str = "23:59"
) -> pd.DataFrame:
    """One synthetic test day at 1-minute resolution, with raw tag names."""
    rng = np.random.default_rng(seed)
    n = MINUTES
    h = np.arange(n) / 60.0
    noise = lambda s: rng.normal(0.0, s, n)  # noqa: E731
    on = _window(n, start, end) if fault != "normal" else np.zeros(n, dtype=bool)

    t_amb = t_mean + 6.0 * np.sin(2 * np.pi * (h - 9.0) / 24.0) + noise(0.2)
    t_room = 19.0 + 0.15 * (t_amb - 15.0) + noise(0.1)

    # Defrost schedules: LT 12:15-12:45 (electric, then 90 min pull-down), MT every 4 h from 01:30 for 30 min.
    lt_def = _window(n, "12:15", "12:45")
    mt_def = np.zeros(n, dtype=bool)
    for k in range(6):
        mt_def[90 + 240 * k : 120 + 240 * k] = True
    lt_rec = np.clip(1.0 - (np.arange(n) - 765) / 90.0, 0.0, 1.0) * (np.arange(n) >= 765)

    # LT case
    lt_cycle = (np.sin(2 * np.pi * h * 3.0) > 0.25).astype(float)  # valve feeding ~40 % of the time
    lt_te = -31.0 + 1.0 * lt_cycle + noise(0.2)
    lt_sh = 23.5 + noise(0.8)
    lt_ret = -19.6 + noise(0.15)
    lt_sup = -20.5 + noise(0.15)
    lt_prod = -19.0 + noise(0.05)
    w_lt = 0.8 * lt_cycle + noise(0.02)
    lt_dis = -9.0 + 58.0 + noise(0.8)
    if fault == "lt_door_open":
        lt_cycle = np.where(on, (np.sin(2 * np.pi * h * 3.0) > -0.1).astype(float), lt_cycle)
        w_lt = 0.8 * lt_cycle + noise(0.02)
        lt_dis = lt_dis + 7.0 * on
        lt_ret, lt_prod = lt_ret + 2.0 * on, lt_prod + 1.8 * on
    lt_ret = np.where(lt_def, 10.0 + noise(0.3), lt_ret + 8.0 * lt_rec)
    lt_sup = np.where(lt_def, 15.0 + noise(0.3), lt_sup + 8.0 * lt_rec)
    lt_eev = np.where(lt_cycle > 0, lt_te + 1.0, lt_te + 12.0)
    lt_eev = np.where(lt_def, 5.0 + noise(0.3), lt_eev)

    # MT case
    mt_cycle = (np.sin(2 * np.pi * h * 4.0) > -0.5).astype(float)
    mt_te = -8.9 + noise(0.05)
    mt_sh = 12.0 + noise(0.8)
    mt_ret, mt_sup = 7.0 + noise(0.2), -0.3 + noise(0.2)
    mt_prod = 2.4 + noise(0.05)
    mt_ret = np.where(mt_def, 13.0 + noise(0.3), mt_ret)
    mt_sup = np.where(mt_def, 10.0 + noise(0.3), mt_sup)
    mt_eev = np.where(mt_def, 9.0 + noise(0.3), np.where(mt_cycle > 0, mt_te + 1.0, mt_te + 8.0))

    # Plant: high side follows the weather, transcritical above ~25 degC outdoors.
    hot = np.maximum(t_amb - 5.0, 0.0)
    p_gc = 4400.0 + 110.0 * hot + noise(30.0)
    gc_app = 3.0 + noise(0.3)
    gc_air = 3.0 + noise(0.3)
    w_mt = 3.1 + 0.18 * hot + noise(0.1)
    mt_dis_c = t_amb + 28.0 + noise(0.8)
    if fault == "cond_blockage":
        p_gc, gc_app, gc_air = p_gc + 700.0 * on, gc_app + 5.0 * on, gc_air + 3.5 * on
        w_mt, mt_dis_c = w_mt + 2.5 * on, mt_dis_c + 9.0 * on
    p_mt_suc = co2.p_sat(-9.5) + noise(5.0)

    raw = {
        "W_MT-COMP1": w_mt,
        "W_MT-COMP2": np.abs(noise(0.02)),
        "W_MT-COMP3": np.abs(noise(0.01)),
        "W_LT-COMP1": w_lt,
        "W_LT-COMP2": np.abs(noise(0.005)),
        "M-MTcooler": 1.3 * mt_cycle + noise(0.1),
        "P-GC-IN": _psig(p_gc),
        "P-MT_Dis-OilSepIn": _psig(p_gc + 15.0 + noise(5.0)),
        "P-FlashTank": _psig(3200.0 + noise(10.0)),
        "P-MTcase-LIQ": _psig(3190.0 + noise(10.0)),
        "P-MT_SUC": _psig(p_mt_suc),
        "P-MTcase-SUC": _psig(co2.p_sat(mt_te)),
        "P-MTcase-SUC-inside": _psig(co2.p_sat(mt_te) + 4.0 + noise(2.0)),
        "P-LT-SUC": _psig(co2.p_sat(lt_te) - 3.0 + noise(2.0)),
        "P-LTcase-SUC": _psig(co2.p_sat(lt_te)),
        "P-LTcase-SUC-inside": _psig(co2.p_sat(lt_te) + 4.0 + noise(2.0)),
        "T-GC-Fan2-In": _f(t_amb),
        "T-GC-Fan1-Out": _f(t_amb + gc_air + noise(0.2)),
        "T-GC-Fan2-Out": _f(t_amb + gc_air + noise(0.2)),
        "T-GC-In": _f(t_amb + 45.0 + noise(1.0)),
        "T-GC-DIS": _f(t_amb + gc_app),
        "T-316": _f(t_room),
        "T-MT-Suc": _f(-5.0 + noise(0.5)),
        "T-MT-Dis": _f(mt_dis_c),
        "T-LT-Suc": _f(4.0 + noise(0.5)),
        "T-LT-Dis": _f(lt_dis),
        "T-LTcase-Sup": _f(lt_sup),
        "T-LTcase-Ret": _f(lt_ret),
        "T-LTcase-EEVout": _f(lt_eev),
        "T-LTcase-Suc": _f(lt_te + lt_sh),
        "T-MTCase-Sup": _f(mt_sup),
        "T-MTCase-Ret": _f(mt_ret),
        "T-MTCase-EEVOut": _f(mt_eev),
        "T-MTCase-Suc": _f(mt_te + mt_sh),
    }
    for i, tag in enumerate(signals.LT_PROBES):
        raw[tag] = _f(lt_prod + 0.3 * np.sin(i) + noise(0.05) + 6.0 * lt_rec)
    for i, tag in enumerate(signals.MT_PROBES):
        raw[tag] = _f(mt_prod + 1.6 * np.sin(i) + noise(0.05))
    index = pd.date_range(date, periods=n, freq="1min", name="time")
    return pd.DataFrame(raw, index=index)


SMOKE_DAYS = {
    # key: (date, mean outdoor degC, seed, fault, start, end)
    "base_A": ("2019-11-05", 9.0, 1, "normal", "00:00", "23:59"),
    "base_B": ("2019-07-13", 25.0, 2, "normal", "00:00", "23:59"),
    "base_C": ("2019-08-06", 22.0, 3, "normal", "00:00", "23:59"),
    "base_D": ("2019-11-09", 4.0, 4, "normal", "00:00", "23:59"),
    "door": ("2019-10-29", 16.0, 5, "lt_door_open", "09:40", "14:30"),
    "cond": ("2019-10-21", 16.0, 6, "cond_blockage", "13:05", "15:20"),
}


def smoke_days() -> dict[str, tuple[pd.DataFrame, str]]:
    return {k: (day(d, t, s, f, a, b), f) for k, (d, t, s, f, a, b) in SMOKE_DAYS.items()}

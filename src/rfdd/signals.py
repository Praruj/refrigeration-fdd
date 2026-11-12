"""Map the raw logger channels (degF, psig, kW) to a small set of named SI signals.

Only this module knows the raw tag names. Everything downstream works on the tidy frame:
temperatures in degC, pressures in kPa absolute, power in kW, mass flow in kg/min.
"""

from __future__ import annotations

import pandas as pd

PSI_TO_KPA = 6.894757
ATM_PSI = 14.696

# tidy name -> raw tag
TEMPERATURES = {
    "t_amb": "T-GC-Fan2-In",  # air entering the gas cooler = outdoor air
    "t_gc_air_out1": "T-GC-Fan1-Out",
    "t_gc_air_out2": "T-GC-Fan2-Out",
    "t_gc_in": "T-GC-In",  # refrigerant entering the gas cooler
    "t_gc_out": "T-GC-DIS",  # refrigerant leaving the gas cooler
    "t_room": "T-316",  # air in the test room, next to the cases
    "t_mt_suc": "T-MT-Suc",
    "t_mt_dis": "T-MT-Dis",
    "t_lt_suc": "T-LT-Suc",
    "t_lt_dis": "T-LT-Dis",
    "t_ltc_sup": "T-LTcase-Sup",  # LT case air off the coil
    "t_ltc_ret": "T-LTcase-Ret",  # LT case air onto the coil
    "t_ltc_eev": "T-LTcase-EEVout",
    "t_ltc_suc": "T-LTcase-Suc",  # refrigerant leaving the LT evaporator
    "t_mtc_sup": "T-MTCase-Sup",
    "t_mtc_ret": "T-MTCase-Ret",
    "t_mtc_eev": "T-MTCase-EEVOut",
    "t_mtc_suc": "T-MTCase-Suc",
}
PRESSURES = {
    "p_gc": "P-GC-IN",
    "p_mt_dis": "P-MT_Dis-OilSepIn",
    "p_flash": "P-FlashTank",
    "p_liq": "P-MTcase-LIQ",
    "p_mt_suc": "P-MT_SUC",
    "p_mtc_suc": "P-MTcase-SUC",
    "p_mtc_suc2": "P-MTcase-SUC-inside",
    "p_lt_suc": "P-LT-SUC",
    "p_ltc_suc": "P-LTcase-SUC",
    "p_ltc_suc2": "P-LTcase-SUC-inside",
}
POWERS = {
    "w_mt1": "W_MT-COMP1",
    "w_mt2": "W_MT-COMP2",
    "w_mt3": "W_MT-COMP3",
    "w_lt1": "W_LT-COMP1",
    "w_lt2": "W_LT-COMP2",
}
FLOWS = {"m_mt": "M-MTcooler"}

# Thermocouples inside the two display cases (product simulators and shelf air).
LT_PROBES = [f"T-{a}{b:02d}" for a in (1, 2) for b in range(1, 17)] + [f"T-3{b:02d}" for b in range(1, 9)]
MT_PROBES = [f"T-{a}{b:02d}" for a in (4, 5) for b in range(1, 17)]

# Channels that are dead or not part of the refrigeration system; never given to any model.
IGNORED = [
    "M-LTcooler", "W-CONDENSOR", "T-BP-EEVin", "T-spare-13B", "T-spare-16C", "T-GC-Out", "T-GC-Fan1-In",
    "T-Spare-2D", "T-spare-3D", "T-Spare-4D",
]  # fmt: skip


def f_to_c(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return (x - 32.0) / 1.8


def psig_to_kpa(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return (x + ATM_PSI) * PSI_TO_KPA


def tidy(minute: pd.DataFrame) -> pd.DataFrame:
    """Named SI signals from a 1-minute raw frame."""
    out = pd.DataFrame(index=minute.index)
    for name, tag in TEMPERATURES.items():
        out[name] = f_to_c(minute[tag])
    for name, tag in PRESSURES.items():
        out[name] = psig_to_kpa(minute[tag])
    for name, tag in {**POWERS, **FLOWS}.items():
        out[name] = minute[tag]
    lt, mt = f_to_c(minute[LT_PROBES]), f_to_c(minute[MT_PROBES])
    out["lt_prod"] = lt.mean(axis=1)
    out["mt_prod"] = mt.mean(axis=1)
    out["mt_prod_spread"] = mt.std(axis=1)
    return out


def raw_sensor_matrix(minute: pd.DataFrame) -> pd.DataFrame:
    """Every live raw channel, untouched. This is what the black-box model sees."""
    return minute.drop(columns=[c for c in IGNORED if c in minute.columns])

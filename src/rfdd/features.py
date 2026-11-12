"""Physics features from the vapour-compression cycle.

Each feature is a quantity a refrigeration engineer would read off a pressure-enthalpy chart or a service
gauge set. Raw pressures and temperatures move with the weather; these differences and ratios mostly do not,
which is the whole point of computing them.

All features are causal: a value at minute t uses data up to t only (a unit test checks this).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import co2
from .states import LT_CASE, MT_CASE, STEADY, tag_states


@dataclass(frozen=True)
class Feature:
    name: str
    group: str  # "lt" / "mt": only valid while that case is steady; "plant": always valid
    unit: str
    meaning: str
    drivers: tuple[str, ...] = ()  # operating conditions this feature legitimately depends on
    weather: bool = False  # True if the raw value moves a lot with outdoor temperature


FEATURES: tuple[Feature, ...] = (
    Feature(
        "lt_sh",
        "lt",
        "K",
        "LT evaporator superheat: suction temperature minus evaporating temperature",
        ("t_room", "lt_since"),
    ),
    Feature(
        "lt_td",
        "lt",
        "K",
        "LT coil temperature difference: return air minus evaporating temperature",
        ("t_room", "lt_since"),
    ),
    Feature("lt_air", "lt", "degC", "LT case return-air temperature", ("t_room", "lt_since")),
    Feature("lt_prod", "lt", "degC", "LT product temperature, mean of 40 probes", ("t_room", "lt_since")),
    Feature(
        "lt_feed", "lt", "-", "fraction of time the LT expansion valve is feeding the coil", ("t_room", "lt_since")
    ),
    Feature("w_lt", "lt", "kW", "LT compressor power", ("t_room", "t_amb", "lt_since")),
    Feature("lt_dis_sh", "lt", "K", "LT compressor discharge superheat", ("t_room", "t_amb", "lt_since")),
    Feature("mt_sh", "mt", "K", "MT evaporator superheat", ("t_room", "mt_since")),
    Feature(
        "mt_td",
        "mt",
        "K",
        "MT coil temperature difference: return air minus evaporating temperature",
        ("t_room", "mt_since"),
    ),
    Feature("mt_air_dt", "mt", "K", "air temperature drop across the MT coil", ("t_room", "mt_since")),
    Feature("mt_prod", "mt", "degC", "MT product temperature, mean of 32 probes", ("t_room", "mt_since")),
    Feature("mt_prod_spread", "mt", "K", "spread of MT product temperatures across shelves", ("t_room", "mt_since")),
    Feature(
        "mt_feed", "mt", "-", "fraction of time the MT expansion valve is feeding the coil", ("t_room", "mt_since")
    ),
    Feature(
        "gc_app",
        "plant",
        "K",
        "gas-cooler approach: refrigerant out minus outdoor air",
        ("t_amb", "t_room", "lt_since", "mt_since"),
    ),
    Feature(
        "gc_air_dt",
        "plant",
        "K",
        "air temperature rise across the gas cooler",
        ("t_amb", "t_room", "lt_since", "mt_since"),
    ),
    Feature(
        "p_gc", "plant", "kPa", "gas-cooler (high-side) pressure", ("t_amb", "t_room", "lt_since", "mt_since"), True
    ),
    Feature("pr_mt", "plant", "-", "MT compressor pressure ratio", ("t_amb", "t_room", "lt_since", "mt_since"), True),
    Feature("w_mt", "plant", "kW", "MT compressor power", ("t_amb", "t_room", "lt_since", "mt_since"), True),
    Feature(
        "mt_dis_sh",
        "plant",
        "K",
        "MT compressor discharge temperature above outdoor air",
        ("t_amb", "t_room", "lt_since", "mt_since"),
        True,
    ),
)
FEATURE_NAMES = [f.name for f in FEATURES]
BY_NAME = {f.name: f for f in FEATURES}

# Operating conditions the plant does not control. The hybrid model predicts each feature from its own subset.
DRIVERS = ["t_amb", "t_room", "lt_since", "mt_since"]

# Smallest residual scale the hybrid model may assume, from the instrument accuracies in the data descriptor
# (thermocouples about 1 K, pressure 0.25 % of full scale, power 5 % of reading), reduced where a feature
# averages many probes. A learned noise level below what the sensors can resolve is not believable.
ACCURACY_FLOOR = {
    "lt_sh": 0.5, "lt_td": 0.5, "lt_air": 0.5, "lt_prod": 0.3, "lt_feed": 0.05, "w_lt": 0.03, "lt_dis_sh": 0.5,
    "mt_sh": 0.5, "mt_td": 0.5, "mt_air_dt": 0.5, "mt_prod": 0.3, "mt_prod_spread": 0.1, "mt_feed": 0.05,
    "gc_app": 0.5, "gc_air_dt": 0.5, "p_gc": 50.0, "pr_mt": 0.03, "w_mt": 0.15, "mt_dis_sh": 0.5,
}  # fmt: skip

WINDOW = 30  # minutes of trailing average: longer than one compressor cycle
MIN_VALID = 15


def instantaneous(tidy: pd.DataFrame) -> pd.DataFrame:
    """Per-minute physics quantities, before any state masking or averaging."""
    f = pd.DataFrame(index=tidy.index)
    lt_te = co2.t_sat(tidy["p_ltc_suc"])
    mt_te = co2.t_sat(tidy["p_mtc_suc"])
    f["lt_sh"] = tidy["t_ltc_suc"] - lt_te
    f["lt_td"] = tidy["t_ltc_ret"] - lt_te
    f["lt_air"] = tidy["t_ltc_ret"]
    f["lt_prod"] = tidy["lt_prod"]
    lt_gap = tidy["t_ltc_eev"] - lt_te  # valve outlet close to saturation = liquid is flowing
    f["lt_feed"] = (lt_gap < 6.0).astype(float).where(lt_gap.notna())
    f["w_lt"] = tidy["w_lt1"] + tidy["w_lt2"]
    f["lt_dis_sh"] = tidy["t_lt_dis"] - co2.t_sat(tidy["p_mt_suc"])
    f["mt_sh"] = tidy["t_mtc_suc"] - mt_te
    f["mt_td"] = tidy["t_mtc_ret"] - mt_te
    f["mt_air_dt"] = tidy["t_mtc_ret"] - tidy["t_mtc_sup"]
    f["mt_prod"] = tidy["mt_prod"]
    f["mt_prod_spread"] = tidy["mt_prod_spread"]
    mt_gap = tidy["t_mtc_eev"] - mt_te
    f["mt_feed"] = (mt_gap < 4.0).astype(float).where(mt_gap.notna())
    f["gc_app"] = tidy["t_gc_out"] - tidy["t_amb"]
    f["gc_air_dt"] = tidy[["t_gc_air_out1", "t_gc_air_out2"]].mean(axis=1) - tidy["t_amb"]
    f["p_gc"] = tidy["p_gc"]
    f["pr_mt"] = tidy["p_mt_dis"] / tidy["p_mt_suc"]
    f["w_mt"] = tidy["w_mt1"] + tidy["w_mt2"] + tidy["w_mt3"]
    f["mt_dis_sh"] = tidy["t_mt_dis"] - tidy["t_amb"]
    return f[FEATURE_NAMES]


def build(tidy: pd.DataFrame, mask_states: bool = True) -> pd.DataFrame:
    """Feature table for one day: trailing means of the physics features, drivers and case states.

    With `mask_states` the LT and MT features only average steady minutes and are NaN during defrost and
    recovery. Without it they average everything (used to measure what the masking is worth).
    """
    inst = instantaneous(tidy)
    lt = tag_states(tidy, LT_CASE)
    mt = tag_states(tidy, MT_CASE)
    steady = {"lt": lt["state"] == STEADY, "mt": mt["state"] == STEADY, "plant": pd.Series(True, index=tidy.index)}
    out = pd.DataFrame(index=tidy.index)
    for feat in FEATURES:
        ok = steady[feat.group] if mask_states else steady["plant"]
        avg = inst[feat.name].where(ok).rolling(WINDOW, min_periods=MIN_VALID).mean()
        out[feat.name] = avg.where(ok)
    out["t_amb"] = tidy["t_amb"].rolling(WINDOW, min_periods=1).mean()
    out["t_room"] = tidy["t_room"].rolling(WINDOW, min_periods=1).mean()
    # Minutes since the last defrost, capped: before the first defrost of a file the true value is unknown.
    out["lt_since"] = lt["since_defrost"].fillna(600.0).clip(upper=600.0)
    out["mt_since"] = mt["since_defrost"].fillna(240.0).clip(upper=240.0)
    out["lt_state"] = lt["state"]
    out["mt_state"] = mt["state"]
    out["lt_overrun"] = lt["overrun"]
    out["mt_overrun"] = mt["overrun"]
    return out


def valid_mask(table: pd.DataFrame) -> pd.DataFrame:
    """True where a feature may be judged."""
    return pd.DataFrame({f.name: table[f.name].notna() for f in FEATURES}, index=table.index)


def nan_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=FEATURE_NAMES)

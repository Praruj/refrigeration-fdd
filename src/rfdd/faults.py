"""The fault table: each fault, the physics behind it, and the feature changes it should cause.

The signs come from textbook vapour-compression reasoning, not from fitting this data set. The pipeline
later checks every predicted sign against what the data shows and reports the disagreements.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fault:
    key: str
    name: str
    group: str  # faults in one group share a signature and cannot be told apart from these sensors
    why: str
    signature: dict[str, int]  # feature -> expected direction (+1 up, -1 down)


FAULTS: tuple[Fault, ...] = (
    Fault(
        "lt_door_open",
        "LT case door left open",
        "lt_door_open",
        "Warm, humid room air enters the case. Load rises, so the valve feeds and the compressor runs for "
        "more of the time and the case air warms. Warmer suction gas gives a hotter discharge.",
        {"lt_air": +1, "lt_prod": +1, "lt_feed": +1, "w_lt": +1, "lt_dis_sh": +1},
    ),
    Fault(
        "lt_evap_ice",
        "Ice build-up on the LT evaporator",
        "lt_evap_ice",
        "Frost insulates the fins and chokes the airflow. The coil has to run colder than the air by a "
        "wider margin, the thermostat is never satisfied so the valve feeds nearly all the time, and the "
        "poorly loaded coil returns colder suction gas (lower superheat).",
        {"lt_td": +1, "lt_feed": +1, "w_lt": +1, "lt_sh": -1, "lt_air": +1},
    ),
    Fault(
        "lt_eev_stuck",
        "LT expansion valve stuck closed",
        "lt_eev_stuck",
        "The coil is starved of liquid. What little refrigerant passes is fully evaporated early, so "
        "superheat climbs, the case stops cooling, and the compressor has nothing to pump.",
        {"lt_sh": +1, "lt_td": +1, "lt_air": +1, "lt_prod": +1, "w_lt": -1, "lt_feed": -1},
    ),
    Fault(
        "mt_evap_fan",
        "MT evaporator fan failure",
        "mt_airflow",
        "Less air crosses the coil. The air that does is cooled further (bigger temperature drop), the coil "
        "runs colder relative to the return air, superheat falls, and shelves far from the working fans warm "
        "up, so product temperatures spread out.",
        {"mt_air_dt": +1, "mt_td": +1, "mt_sh": -1, "mt_prod": +1, "mt_prod_spread": +1, "mt_feed": +1},
    ),
    Fault(
        "mt_evap_blockage",
        "MT evaporator air path blocked",
        "mt_airflow",
        "Same mechanism as a failed fan: the coil sees less air.",
        {"mt_air_dt": +1, "mt_td": +1, "mt_sh": -1, "mt_prod": +1, "mt_prod_spread": +1, "mt_feed": +1},
    ),
    Fault(
        "cond_blockage",
        "Gas cooler (condenser) air path blocked",
        "cond_blockage",
        "Less air removes the same heat, so the air warms more across the coil and the refrigerant leaves "
        "further above outdoor temperature. Head pressure and pressure ratio rise, and the MT compressors "
        "draw more power and discharge hotter.",
        {"gc_app": +1, "gc_air_dt": +1, "p_gc": +1, "pr_mt": +1, "w_mt": +1, "mt_dis_sh": +1},
    ),
)

BY_KEY = {f.key: f for f in FAULTS}
GROUPS = list(dict.fromkeys(f.group for f in FAULTS))
GROUP_OF = {f.key: f.group for f in FAULTS}
# One signature per group (members of a group share it by construction).
GROUP_SIGNATURE = {f.group: f.signature for f in FAULTS}

# Fixed engineering margins for the rule-based detector: how far a feature must move from its normal value
# before a technician would call it abnormal. Units follow features.FEATURES.
RULE_MARGIN = {
    "lt_sh": 4.0, "lt_td": 3.0, "lt_air": 1.5, "lt_prod": 1.5, "lt_feed": 0.25, "w_lt": 0.10, "lt_dis_sh": 8.0,
    "mt_sh": 4.0, "mt_td": 3.0, "mt_air_dt": 2.0, "mt_prod": 1.5, "mt_prod_spread": 0.6, "mt_feed": 0.25,
    "gc_app": 3.0, "gc_air_dt": 2.0, "p_gc": 800.0, "pr_mt": 0.35, "w_mt": 2.0, "mt_dis_sh": 8.0,
}  # fmt: skip

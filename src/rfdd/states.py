"""Operating-state tagging: steady, defrost, or recovery after defrost.

Most refrigeration false alarms come from judging a normal transient with a steady-state rule. Each display
case therefore carries a state, and fault logic only looks at steady minutes.

The data set has no defrost relay signal, so defrost is inferred from the case air leaving the coil: with the
valve shut (and the heater on, for the LT case) that air warms well above anything seen while cooling.

A watchdog bounds the mask: a "defrost" longer than the maximum duration is no longer treated as a defrost.
Without it, a valve stuck shut would look like one endless defrost and hide behind the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEADY, DEFROST, RECOVERY = "steady", "defrost", "recovery"


@dataclass(frozen=True)
class StateConfig:
    signal: str  # air temperature leaving the coil, degC
    warm_above: float  # above this the coil is not cooling
    max_defrost_min: int  # watchdog: longest believable defrost
    recovery_min: int  # pull-down time masked after a defrost ends


LT_CASE = StateConfig("t_ltc_sup", warm_above=-12.0, max_defrost_min=120, recovery_min=150)
MT_CASE = StateConfig("t_mtc_sup", warm_above=4.0, max_defrost_min=75, recovery_min=45)


def tag_states(tidy: pd.DataFrame, cfg: StateConfig) -> pd.DataFrame:
    """State per minute, plus minutes since the last defrost ended. Causal: uses past and present only."""
    warm = (tidy[cfg.signal] > cfg.warm_above).to_numpy()
    n = len(warm)
    state = np.full(n, STEADY, dtype=object)
    since = np.full(n, np.nan)
    run = 0  # consecutive warm minutes
    last_end: int | None = None
    overrun = False
    for i in range(n):
        if warm[i]:
            run += 1
            if run > cfg.max_defrost_min:
                overrun = True  # watchdog tripped: this is not a defrost any more
            state[i] = STEADY if overrun else DEFROST
        else:
            if run > 0 and not overrun:
                last_end = i
            run, overrun = 0, False
            if last_end is not None and i - last_end < cfg.recovery_min:
                state[i] = RECOVERY
        if last_end is not None and not warm[i]:
            since[i] = i - last_end
    return pd.DataFrame({"state": state, "since_defrost": since, "overrun": _overrun_flag(warm, cfg)}, index=tidy.index)


def _overrun_flag(warm: np.ndarray, cfg: StateConfig) -> np.ndarray:
    out = np.zeros(len(warm), dtype=bool)
    run = 0
    for i, w in enumerate(warm):
        run = run + 1 if w else 0
        out[i] = run > cfg.max_defrost_min
    return out

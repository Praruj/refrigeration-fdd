"""CO2 (R-744) saturation line, as a small lookup table.

The table was generated once with CoolProp 8.0 (Span-Wagner equation of state) at 2 K steps and is
interpolated linearly in (T, ln P), which is accurate to better than 0.02 K over the range. Keeping the
table in the package means the pipeline needs no thermodynamics library at run time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

T_CRIT_C = 30.98
P_CRIT_KPA = 7377.3

_T_C = np.arange(-50.0, 30.01, 2.0)
_P_KPA = np.array(
    [682.3, 739.5, 800.2, 864.4, 932.5, 1004.5, 1080.5, 1160.7, 1245.2, 1334.2, 1427.8, 1526.1, 1629.3, 1737.5,
     1850.9, 1969.6, 2093.8, 2223.7, 2359.3, 2501.0, 2648.7, 2802.7, 2963.2, 3130.3, 3304.2, 3485.1, 3673.3,
     3868.8, 4072.0, 4283.1, 4502.2, 4729.7, 4965.8, 5210.8, 5465.1, 5729.1, 6003.1, 6287.7, 6583.7, 6891.8,
     7213.7]
)  # fmt: skip
_LNP = np.log(_P_KPA)


def t_sat(p_kpa: ArrayLike) -> NDArray[np.float64]:
    """Saturation temperature in degC for an absolute pressure in kPa.

    Above the table (supercritical side) the value is clipped to 30 degC, so "condensing temperature" stays
    finite when the gas cooler runs transcritical. Callers should use the gas-cooler approach there instead.
    """
    p = np.asarray(p_kpa, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lnp = np.log(np.where(p > 0, p, np.nan))
    return np.interp(lnp, _LNP, _T_C)


def p_sat(t_c: ArrayLike) -> NDArray[np.float64]:
    """Saturation pressure in kPa (absolute) for a temperature in degC."""
    return np.exp(np.interp(np.asarray(t_c, dtype=float), _T_C, _LNP))

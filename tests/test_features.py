import numpy as np
import pandas as pd

from rfdd import features, signals, synth
from rfdd.states import DEFROST, LT_CASE, MT_CASE, RECOVERY, STEADY, tag_states


def _tidy() -> pd.DataFrame:
    return signals.tidy(synth.day("2020-01-01", 15.0, seed=0))


def test_features_are_causal() -> None:
    """Truncating the day must not change any earlier feature value."""
    tidy = _tidy()
    full = features.build(tidy)
    cut = features.build(tidy.iloc[:900])
    cols = [*features.FEATURE_NAMES, "lt_since", "mt_since"]
    pd.testing.assert_frame_equal(full.iloc[:900][cols], cut[cols])


def test_physics_values_are_plausible() -> None:
    f = features.build(_tidy())
    assert 20 < f["lt_sh"].median() < 27  # synthetic superheat is 23.5 K
    assert 10 < f["mt_sh"].median() < 14
    assert 2 < f["gc_app"].median() < 4


def test_defrost_states_and_masking() -> None:
    tidy = _tidy()
    lt = tag_states(tidy, LT_CASE)
    assert (lt["state"] == DEFROST).sum() >= 25  # one 30-minute LT defrost
    assert lt.loc[lt.index[800], "state"] == RECOVERY
    assert lt.loc[lt.index[300], "state"] == STEADY
    mt = tag_states(tidy, MT_CASE)
    assert (mt["state"] == DEFROST).sum() >= 6 * 25  # six MT defrosts
    table = features.build(tidy)
    assert table.loc[mt["state"] != STEADY, "mt_td"].isna().all()
    assert table.loc[mt["state"] != STEADY, "gc_app"].notna().any()  # plant features are never masked


def test_watchdog_stops_endless_defrost_hiding_a_fault() -> None:
    tidy = _tidy()
    tidy.loc[tidy.index[600] :, "t_ltc_sup"] = 5.0  # coil never cools again, like a stuck valve
    states = tag_states(tidy, LT_CASE)
    assert states["state"].iloc[-1] == STEADY
    assert states["overrun"].iloc[-1]
    assert np.isfinite(features.build(tidy)["lt_air"].iloc[-1])

import numpy as np
import pandas as pd

from rfdd import evaluate, models, prepare
from rfdd.faults import FAULTS, GROUP_SIGNATURE, GROUPS, RULE_MARGIN
from rfdd.features import BY_NAME, FEATURE_NAMES


def test_fault_table_is_consistent() -> None:
    for f in FAULTS:
        assert f.signature
        assert set(f.signature) <= set(FEATURE_NAMES)
        assert set(f.signature.values()) <= {-1, 1}
    assert set(RULE_MARGIN) == set(FEATURE_NAMES)
    assert all(BY_NAME[n].drivers for n in FEATURE_NAMES)


def test_persist_matches_alarm_logic(logic) -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 500), index=pd.date_range("2020", periods=500, freq="1min"))
    s.iloc[100:110] = np.nan
    for thr in (0.0, 0.5, 1.5):
        assert (models.persist(s, logic) > thr).equals(logic.alarms(s, thr))


def test_symptom_veto() -> None:
    signed = pd.DataFrame({f: [0.0] for f in FEATURE_NAMES})
    signed[["lt_air", "lt_prod", "lt_sh"]] = 5.0  # warm case + high superheat
    signed["w_lt"] = -3.0  # compressor idle: contradicts an open door
    s = models.symptom_count_scores(signed, {g: list(sig) for g, sig in GROUP_SIGNATURE.items()})
    assert s["lt_door_open"].iloc[0] <= 0.0
    assert s["lt_eev_stuck"].iloc[0] >= 5.0


def test_hybrid_detects_injected_faults_without_false_alarms(smoke_cfg, smoke_days, logic) -> None:
    # Hold out the mild-weather day C: its weather lies inside the range covered by the other three days.
    normal = [d for d in smoke_days.values() if d.fault == "normal"]
    held = smoke_days["base_C"]
    det = models.HybridDetector().fit([d for d in normal if d.key != "base_C"], logic)
    assert set(det.thresholds) == {*GROUPS, models.UNKNOWN}
    assert not evaluate.alarm_series(det.score(held), logic, 1.25).any()
    for ev in prepare.events(smoke_cfg, smoke_days):
        sc = det.score(smoke_days[ev.day])
        alarm = evaluate.alarm_series(sc, logic, 1.25)
        assert alarm.loc[ev.start : ev.end].any(), ev.fault
        assert evaluate.event_diagnosis(sc, alarm, ev, logic, 1.25) in (ev.fault, models.UNKNOWN)


def test_range_guard_abstains_outside_training_weather(smoke_days, logic) -> None:
    normal = [d for d in smoke_days.values() if d.fault == "normal"]
    det = models.HybridDetector().fit(normal, logic)
    day = normal[0]
    hot = prepare.Day(day.key, "normal", day.raw, day.tidy, day.bad, day.table.copy(), day.usable)
    hot.table["t_amb"] = det.envelope["t_amb"][1] + 10.0
    z = det.z(hot)
    assert z["p_gc"].isna().all()  # weather-driven feature: no prediction outside the envelope
    assert z["lt_sh"].notna().any()  # case feature does not depend on outdoor air

import numpy as np
import pandas as pd

from rfdd.events import AlarmLogic, Event, episodes, false_alarms, score_event

IDX = pd.date_range("2020-01-01", periods=1440, freq="1min")


def _score(on: list[tuple[int, int]]) -> pd.Series:
    s = pd.Series(0.0, index=IDX)
    for a, b in on:
        s.iloc[a:b] = 5.0
    return s


def test_short_spike_does_not_alarm_but_sustained_does() -> None:
    logic = AlarmLogic(k=20, n=30)
    assert not logic.alarms(_score([(100, 110)]), 1.0).any()
    alarm = logic.alarms(_score([(100, 200)]), 1.0)
    assert alarm.any()
    assert alarm.idxmax() == IDX[119]  # the 20th minute over the threshold


def test_masked_minutes_never_vote() -> None:
    s = _score([(100, 200)])
    s.iloc[100:200] = np.nan
    assert not AlarmLogic().alarms(s, 1.0).any()


def test_event_delay_and_false_alarm_count() -> None:
    logic = AlarmLogic(k=20, n=30)
    alarm = logic.alarms(_score([(300, 500), (900, 1000)]), 1.0)
    ev = Event("d", "f", "s", IDX[300], IDX[499])
    r = score_event(alarm, ev)
    assert r["detected"] and r["delay_min"] == 19
    assert len(episodes(alarm)) == 2
    assert false_alarms(alarm)["per_day"] == 2
    missed = score_event(alarm, Event("d", "f", "s", IDX[600], IDX[700]))
    assert not missed["detected"] and np.isnan(missed["delay_min"])


def test_close_episodes_merge() -> None:
    alarm = pd.Series(False, index=IDX)
    alarm.iloc[100:120] = True
    alarm.iloc[130:150] = True
    alarm.iloc[400:410] = True
    assert len(episodes(alarm, merge_gap=30)) == 2

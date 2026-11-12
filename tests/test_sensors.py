import pandas as pd

from rfdd import sensors, signals, synth


def _tidy() -> pd.DataFrame:
    return signals.tidy(synth.day("2020-01-01", 15.0, seed=3))


def test_clean_day_has_no_flags() -> None:
    assert not sensors.check(_tidy()).any().any()


def test_injected_faults_are_caught_by_the_right_check() -> None:
    tidy = _tidy()
    cases = [
        ("p_ltc_suc", "bias", -200.0),  # three-way vote
        ("p_gc", "bias", 600.0),  # pair
        ("t_ltc_suc", "bias", -25.0),  # suction gas colder than saturation
        ("t_gc_out", "bias", -8.0),  # refrigerant colder than the air cooling it
        ("t_ltc_ret", "stuck", 0.0),  # flatline
    ]
    for sensor, kind, size in cases:
        bad = sensors.check(sensors.inject(tidy, sensor, kind, size, start=300))
        assert bad[sensor].iloc[400:].mean() > 0.9, sensor
        assert not bad[sensor].iloc[:250].any(), sensor


def test_vote_blames_only_the_odd_one_out() -> None:
    bad = sensors.check(sensors.inject(_tidy(), "p_mtc_suc", "bias", 300.0, start=300))
    assert bad["p_mtc_suc"].iloc[400:].all()
    assert not bad["p_mt_suc"].any()
    assert not bad["p_mtc_suc2"].any()


def test_apply_removes_flagged_readings_and_uses_the_spare_tap() -> None:
    tidy = sensors.inject(_tidy(), "p_ltc_suc", "bias", -200.0, start=300)
    clean = sensors.apply(tidy, sensors.check(tidy))
    healthy = _tidy()
    # the voted-out primary is replaced by the healthy tap on the same line (a few kPa apart)
    assert (clean["p_ltc_suc"].iloc[400:] - healthy["p_ltc_suc"].iloc[400:]).abs().max() < 15.0
    tidy = sensors.inject(_tidy(), "t_gc_out", "bias", -8.0, start=300)
    clean = sensors.apply(tidy, sensors.check(tidy))
    assert clean["t_gc_out"].iloc[400:].isna().all()


def test_law_violation_latches() -> None:
    tidy = _tidy()
    tidy.loc[tidy.index[300] : tidy.index[330], "t_gc_out"] -= 8.0  # 30 minutes impossible, then plausible again
    bad = sensors.check(tidy)
    assert bad["t_gc_out"].iloc[331:].all()

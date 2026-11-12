import numpy as np

from rfdd import co2


def test_saturation_matches_reference_points() -> None:
    # Span-Wagner reference values (kPa abs -> degC)
    assert abs(co2.t_sat(3485.1) - 0.0) < 0.02
    assert abs(co2.t_sat(1427.8) - (-30.0)) < 0.02
    assert abs(co2.t_sat(5729.1) - 20.0) < 0.02


def test_round_trip_and_monotonic() -> None:
    t = np.linspace(-45, 28, 200)
    assert np.allclose(co2.t_sat(co2.p_sat(t)), t, atol=1e-6)
    assert np.all(np.diff(co2.t_sat(np.linspace(700, 7000, 500))) > 0)


def test_supercritical_and_bad_pressure_are_safe() -> None:
    assert co2.t_sat(9000.0) == 30.0
    assert np.isnan(co2.t_sat(-5.0))

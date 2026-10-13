import copy

import numpy as np
import pytest

from dsse import scenarios, wls, wls_np


@pytest.fixture(scope="module")
def setup():
    fd, samples = scenarios.generate(num_samples=8, seed=11)
    return fd, samples


def _objective(fd, rd, v):
    h, _ = wls_np.measurement_model(fd.ybus, v, rd)
    return float(np.sum((rd.z - h) ** 2 / rd.var))


def test_matches_gridcal(setup):
    """Same readings, same weighted least squares: where GridCal's Levenberg-Marquardt reaches the
    minimum, Gauss-Newton lands on it too; where it stops short, Gauss-Newton's objective is lower."""
    fd, samples = setup
    est = wls.Estimator()
    matched = 0
    for s in samples:
        v_gc = est(s)
        e = wls_np.estimate(fd, s)
        assert e.observable and e.converged
        j_gc = _objective(fd, e.readings, v_gc)
        assert e.objective <= j_gc * (1 + 1e-9)
        if j_gc <= e.objective * (1 + 1e-6):
            assert np.abs(np.abs(e.v) - np.abs(v_gc)).max() < 1e-6
            assert np.abs(np.angle(e.v) - np.angle(v_gc)).max() < 1e-6
            matched += 1
    assert matched >= len(samples) - 1


def test_exact_readings_give_the_true_state(setup):
    fd, samples = setup
    s = copy.deepcopy(samples[0])
    s.obs[:, 0], s.obs[:, 2], s.obs[:, 3] = np.abs(s.v), s.s.real, s.s.imag
    s.mask[:, [0, 2, 3]] = 1
    s.var[:, [0, 2, 3]] = 1e-6
    e = wls_np.estimate(fd, s)
    assert np.abs(e.v - s.v).max() < 1e-8


def test_no_meter_means_no_estimate(setup):
    """With only pseudo-measurements nothing reads a voltage, so the voltage level is not determined:
    no estimate. One meter is enough, since every other bus has a pseudo-measurement."""
    fd, samples = setup
    prior = scenarios.pseudo_prior(fd, samples[0].s)
    rng = np.random.default_rng(0)
    none = scenarios.measure(samples[0].v, samples[0].s, set(), set(), prior, rng)
    e = wls_np.estimate(fd, none)
    assert not e.observable and e.v is None
    one = scenarios.measure(samples[0].v, samples[0].s, {12}, set(), prior, rng)
    assert wls_np.estimate(fd, one).observable


def test_pmu_angle_is_an_option(setup):
    fd, samples = setup
    s = samples[1]
    pmus = [i for i in np.flatnonzero(s.mask[:, 1]) if i != 0]
    off, on = wls_np.estimate(fd, s), wls_np.estimate(fd, s, angles=True)
    assert len(on.readings) == len(off.readings) + int(s.mask[:, 1].sum())
    assert pmus and on.converged
    assert np.any(np.abs(on.v - off.v) > 0)


def test_householder_matches_numpy():
    a = np.random.default_rng(0).normal(size=(98, 65))
    assert np.allclose(wls_np.householder_diag(a), np.abs(np.diag(np.linalg.qr(a, mode="r"))), atol=1e-10)


def test_observability_of_real_meters(setup):
    fd, _ = setup
    everything = set(range(fd.n))
    assert wls_np.observable(fd, everything)
    # 21 meters read 63 numbers for 65 unknowns
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert not wls_np.observable(fd, set(rng.choice(fd.n, 21, replace=False).tolist()))
    # two neighbours without meters: the buses beyond them can turn by any angle, unseen
    assert not wls_np.observable(fd, everything - {7, 8})
    assert wls_np.rank_margin(fd, everything - {7, 8}) < 1e-10
    assert wls_np.rank_margin(fd, everything - {7, 9}) > 1e-3


def test_browser_observability_test_matches_the_reduced_form(setup):
    """QR of the full flat-start Jacobian (what the browser runs) against the exact reduced system of
    dsse.observability, on random placements of 22 to 33 meters: same verdict, with a wide gap."""
    from dsse import observability

    fd, _ = setup
    rng = np.random.default_rng(4)
    singular, regular = [], []
    for _ in range(300):
        unmetered = np.sort(rng.choice(fd.n, rng.integers(0, 12), replace=False))
        exact = observability.margins(fd, unmetered[None, :].astype(np.int16))[0] > observability.REDUCED_TOL
        m = wls_np.rank_margin(fd, set(range(fd.n)) - set(unmetered.tolist()))
        (regular if exact else singular).append(m)
    assert singular and regular
    assert max(singular) < 1e-10 < wls_np.OBS_TOL < 1e-6 < min(regular)


def test_gross_error_is_found_and_removed(setup):
    fd, samples = setup
    s = copy.deepcopy(samples[2])
    b = int(np.flatnonzero(s.sensor == scenarios.SENSOR_TYPES.index("smart"))[0])
    clean = wls_np.estimate(fd, s).v
    s.obs[b, 0] += 0.05  # about 7 standard deviations
    r = wls_np.estimate_clean(fd, s)
    assert r["chi_square_before"]["bad"]
    assert [(x["bus"], x["kind"]) for x in r["removed"]] == [(b, wls_np.VM)]
    assert not r["chi_square_after"]["bad"]
    assert np.abs(np.abs(r["estimate"].v) - np.abs(clean)).max() < 0.005

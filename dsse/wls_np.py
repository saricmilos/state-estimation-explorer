"""WLS state estimation in plain numpy, for the website to run in the browser.

The browser can't run GridCal, so this is the estimator the TypeScript port copies, step for step:
dense matrices, a Cholesky solve, nothing a few loops can't redo. It solves the problem GridCal's
estimator solves (dsse.wls), on the same readings:

- the state is the angle of buses 1-32 (bus 0 is the reference, angle 0) and |V| at all 33 buses;
- the readings are |V|, P and Q at buses (P and Q in MW, positive into the grid, divided by S_base
  to per-unit as GridCal does), each weighted by 1 / its variance;
- optionally (off by default, since GridCal has no such reading) a PMU's angle.

GridCal minimises the weighted squared error by Levenberg-Marquardt; this uses Gauss-Newton from a
flat start (every |V| 1, every angle 0), which converges to the same minimum in a few steps. A gain
matrix that can't be factorised means the readings don't determine the state: no estimate, and the
caller is told so instead of getting a number (GridCal's damping returns one regardless).

Also here, with the same matrices: observability from real meters alone (`observable`,
`fewest_meters`) and bad data (`chi_square`, `normalised_residuals`, `estimate_clean`).
"""

from dataclasses import dataclass, field

import numpy as np

from . import grid as G
from .scenarios import Sample

VM, VA, P, Q = 0, 1, 2, 3  # reading kinds, the columns of Sample.obs
TOL = 1e-9  # largest step (p.u. or rad) at which Gauss-Newton stops: GridCal's tolerance
MAX_ITER = 50
# A Cholesky pivot below this, on the gain matrix scaled to a unit diagonal, is a column that depends
# on the others. With pseudo-measurements the gain's smallest pivot is 3e-9 or more on every test
# sample with a meter, and -2e-14 with none (no reading fixes the voltage level): this sits between.
PIVOT = 1e-12
CHI2_CONFIDENCE = 0.99  # the chi-square test flags J(x) above this quantile
RN_LIMIT = 3.0  # a reading whose normalised residual is above this is removed
CRITICAL = 1e-6  # Omega_ii below this share of the reading's variance: a critical reading, untestable
# Observability from real meters: rank_margin is below 1e-10 for placements that are exactly singular
# and above 1e-6 for the others (tests/test_wls_np.py checks it against the exact reduced form of
# dsse/observability.py on random placements), so the line goes between.
OBS_TOL = 1e-8


@dataclass
class Readings:
    """The readings in a fixed order (bus by bus; |V|, angle, P, Q within a bus), in per-unit."""
    bus: np.ndarray
    kind: np.ndarray
    z: np.ndarray
    var: np.ndarray

    def __len__(self):
        return len(self.z)

    def without(self, i: int) -> "Readings":
        keep = np.arange(len(self)) != i
        return Readings(self.bus[keep], self.kind[keep], self.z[keep], self.var[keep])


def readings(smp: Sample, sbase: float, angles: bool = False) -> Readings:
    """Every reading of a sample (pseudo-measurements included). A PMU's angle only if `angles`."""
    bus, kind, z, var = [], [], [], []
    for i in range(len(smp.obs)):
        for k in (VM, VA, P, Q):
            if not smp.mask[i, k] or (k == VA and not angles):
                continue
            scale = sbase if k in (P, Q) else 1.0
            bus.append(i)
            kind.append(k)
            z.append(smp.obs[i, k] / scale)
            var.append(smp.var[i, k] / scale**2)
    return Readings(np.array(bus, dtype=int), np.array(kind, dtype=int), np.array(z), np.array(var))


def measurement_model(ybus: np.ndarray, v: np.ndarray, rd: Readings, slack: int = 0):
    """h(x), what each reading should be at voltages v, and H = dh/dx. x is the angles of every bus
    but the slack, then |V| of every bus."""
    n = len(v)
    vm = np.abs(v)
    i_bus = ybus @ v
    s = v * np.conj(i_bus)  # injections, p.u., positive into the grid
    vn = v / vm
    # MATPOWER's derivatives of the injections (Zimmerman, Tech. Note 2), dense
    ds_dvm = v[:, None] * np.conj(ybus * vn[None, :]) + np.diag(np.conj(i_bus) * vn)
    ds_dva = 1j * v[:, None] * np.conj(np.diag(i_bus) - ybus * v[None, :])
    ang = [b for b in range(n) if b != slack]
    m = len(rd)
    h = np.zeros(m)
    H = np.zeros((m, len(ang) + n))
    for r, (b, k) in enumerate(zip(rd.bus, rd.kind)):
        if k == VM:
            h[r] = vm[b]
            H[r, len(ang) + b] = 1.0
        elif k == VA:
            h[r] = np.angle(v[b])
            if b != slack:
                H[r, ang.index(b)] = 1.0
        else:
            part = np.real if k == P else np.imag
            h[r] = part(s[b])
            H[r, : len(ang)] = part(ds_dva[b, ang])
            H[r, len(ang) :] = part(ds_dvm[b, :])
    return h, H


def cholesky_solve(a: np.ndarray, b: np.ndarray, pivot: float = PIVOT):
    """Solve a x = b for a symmetric positive-definite a, scaled to a unit diagonal first so one
    pivot threshold serves every unit. Returns None when a pivot falls below `pivot`: a column that
    depends on the others, so the readings don't determine that part of the state. b may be a
    vector or a matrix of right-hand sides."""
    d = np.sqrt(np.diag(a))
    if np.any(d == 0):
        return None
    s = a / np.outer(d, d)
    n = len(s)
    L = np.zeros_like(s)
    for j in range(n):
        piv = s[j, j] - L[j, :j] @ L[j, :j]
        if piv < pivot:
            return None
        L[j, j] = np.sqrt(piv)
        L[j + 1 :, j] = (s[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    y = (b.T / d).T
    for j in range(n):  # forward: L y = b
        y[j] = (y[j] - L[j, :j] @ y[:j]) / L[j, j]
    for j in reversed(range(n)):  # backward: L^T x = y
        y[j] = (y[j] - L[j + 1 :, j] @ y[j + 1 :]) / L[j, j]
    return (y.T / d).T


@dataclass
class Estimate:
    v: np.ndarray | None  # complex bus voltages, p.u.; None when the readings don't determine them
    observable: bool
    converged: bool
    iterations: int
    readings: Readings
    residual: np.ndarray = field(default=None)  # z - h(x) at the estimate, p.u.
    H: np.ndarray = field(default=None, repr=False)
    gain: np.ndarray = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """J(x) = sum of squared residuals over variances: the chi-square statistic."""
        return float(np.sum(self.residual**2 / self.readings.var))

    @property
    def dof(self) -> int:
        """Readings minus states: what the chi-square test is taken over."""
        return len(self.readings) - self.H.shape[1]


def estimate_readings(fd: G.Feeder, rd: Readings) -> Estimate:
    """Gauss-Newton from a flat start on the given readings."""
    n = fd.n
    ybus = fd.ybus
    ang = np.array([b for b in range(n) if b != fd.slack])
    va, vm = np.zeros(n), np.ones(n)
    w = 1.0 / rd.var
    for it in range(1, MAX_ITER + 1):
        v = vm * np.exp(1j * va)
        h, H = measurement_model(ybus, v, rd, fd.slack)
        gain = H.T @ (w[:, None] * H)
        dx = cholesky_solve(gain, H.T @ (w * (rd.z - h)))
        if dx is None:
            return Estimate(None, False, False, it, rd)
        va[ang] += dx[: len(ang)]
        vm += dx[len(ang) :]
        if np.max(np.abs(dx)) < TOL:
            break
    v = vm * np.exp(1j * va)
    h, H = measurement_model(ybus, v, rd, fd.slack)
    gain = H.T @ (w[:, None] * H)
    return Estimate(v, True, np.max(np.abs(dx)) < TOL, it, rd, rd.z - h, H, gain)


def estimate(fd: G.Feeder, smp: Sample, angles: bool = False) -> Estimate:
    return estimate_readings(fd, readings(smp, fd.sbase, angles))


def objective(fd: G.Feeder, rd: Readings, v: np.ndarray) -> float:
    """J at any voltages v (another estimator's answer, say), for the readings rd."""
    h, _ = measurement_model(fd.ybus, v, rd, fd.slack)
    return float(np.sum((rd.z - h) ** 2 / rd.var))


# ---------------------------------------------------------------------------------------------
# Bad data


def chi_square_limit(dof: int, confidence: float = CHI2_CONFIDENCE) -> float:
    from scipy.stats import chi2

    return float(chi2.ppf(confidence, dof))


def chi_square(est: Estimate) -> dict:
    """If every reading's error is as its variance says, J(x) at the estimate follows a chi-square
    distribution with (readings - states) degrees of freedom; above its 99% point, something in the
    readings is wrong."""
    limit = chi_square_limit(est.dof)
    return {"J": est.objective, "dof": est.dof, "limit": limit, "bad": est.objective > limit}


def residual_variance(est: Estimate) -> np.ndarray:
    """Omega_ii, the diagonal of Omega = R - H G^-1 H^T, the covariance of the residuals. Omega_ii / R_ii
    is the share of a reading's own error that shows in its residual: near 0 when the other readings
    can't contradict it."""
    gi_ht = cholesky_solve(est.gain, est.H.T.copy())
    return est.readings.var - np.einsum("ij,ji->i", est.H, gi_ht)


def normalised_residuals(est: Estimate) -> np.ndarray:
    """|r_i| / sqrt(Omega_ii). A reading whose error is as its variance says gives about N(0, 1); a
    gross error of e standard deviations gives about e * sqrt(Omega_ii / R_ii). A critical reading
    (one without which the state is not determined) always has a zero residual and can't be
    tested: NaN."""
    omega = residual_variance(est)
    rn = np.full(len(omega), np.nan)
    ok = omega > CRITICAL * est.readings.var
    rn[ok] = np.abs(est.residual[ok]) / np.sqrt(omega[ok])
    return rn


def estimate_clean(fd: G.Feeder, smp: Sample, angles: bool = False, limit: float = RN_LIMIT) -> dict:
    """The largest normalised residual test: estimate, and while the largest |r_N| is above `limit`,
    remove that reading and estimate again. Returns the final estimate, the readings removed in
    order (bus, kind, value), and the chi-square test before and after."""
    rd = readings(smp, fd.sbase, angles)
    est = estimate_readings(fd, rd)
    first = est
    removed = []
    while est.observable:
        rn = normalised_residuals(est)
        if np.all(np.isnan(rn)) or np.nanmax(rn) <= limit:
            break
        i = int(np.nanargmax(rn))
        scale = fd.sbase if rd.kind[i] in (P, Q) else 1.0
        removed.append({"bus": int(rd.bus[i]), "kind": int(rd.kind[i]), "value": float(rd.z[i] * scale), "rn": float(rn[i])})
        rd = rd.without(i)
        est = estimate_readings(fd, rd)
    return {
        "estimate": est, "removed": removed,
        "chi_square_before": chi_square(first) if first.observable else None,
        "chi_square_after": chi_square(est) if est.observable else None,
    }


# ---------------------------------------------------------------------------------------------
# Observability from real meters alone


def meter_readings(buses, pmus=()) -> Readings:
    """What meters at `buses` read, with no pseudo-measurements: |V|, P and Q at each, and the angle
    at a PMU if `pmus` names it (bus 0's angle is the reference and adds nothing)."""
    bus, kind = [], []
    for i in sorted(buses):
        for k in (VM, VA, P, Q):
            if k == VA and (i not in pmus or i == 0):
                continue
            bus.append(i)
            kind.append(k)
    m = len(bus)
    return Readings(np.array(bus, dtype=int), np.array(kind, dtype=int), np.zeros(m), np.ones(m))


def householder_diag(a: np.ndarray) -> np.ndarray:
    """|R_jj| of the QR factorisation of a (rows >= columns), by Householder reflections: how much
    of column j is independent of the columns before it."""
    A = a.astype(float).copy()
    m, n = A.shape
    out = np.zeros(n)
    for j in range(n):
        x = A[j:, j]
        norm = np.sqrt(x @ x)
        if norm == 0:
            continue
        alpha = -norm if x[0] >= 0 else norm
        v = x.copy()
        v[0] -= alpha
        A[j:, j:] -= np.outer(v, (2.0 / (v @ v)) * (v @ A[j:, j:]))
        out[j] = abs(alpha)
    return out


def rank_margin(fd: G.Feeder, buses, pmus=()) -> float:
    """The smallest |R_jj| of H, the meters' Jacobian at a flat start with each column scaled to unit
    length: 0 when the gain matrix H^T H is singular. Taken from H by QR, because forming H^T H
    squares H's condition number and puts placements that are exactly singular as high as 2e-7."""
    rd = meter_readings(buses, pmus)
    n_states = 2 * fd.n - 1
    if len(rd) < n_states:
        return 0.0
    _, H = measurement_model(fd.ybus, np.ones(fd.n, dtype=complex), rd, fd.slack)
    norm = np.sqrt(np.sum(H**2, axis=0))
    if np.any(norm == 0):
        return 0.0
    return float(np.min(householder_diag(H / norm)))


def observable(fd: G.Feeder, buses, pmus=()) -> bool:
    """Whether meters at `buses` alone, with no pseudo-measurements, determine the state: the rank of
    the gain matrix, tested on H by QR."""
    return rank_margin(fd, buses, pmus) > OBS_TOL

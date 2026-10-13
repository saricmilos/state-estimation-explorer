"""Where real meters must be for the feeder to be observable without pseudo-measurements.

    python -m dsse.phase2 observability     # writes data/v2/observability.json

wls_np.observable is the test the browser runs on one placement (QR of the flat-start Jacobian). To
find the fewest meters, every placement worth trying is tested here, in an equivalent smaller form.

At a flat start (every |V| 1, every angle 0, no current flowing), a metered bus's P and Q readings fix
the current drawn at that bus and nothing else. What is left unknown is |V0| and the current J_k drawn
at each unmetered bus, and only the |V| readings and the substation's injection see them:
    |V_i| = |V0| - Re( sum over unmetered k of Z(i,k) J_k ) + known terms
    substation injection = - (sum over unmetered k of J_k) + known terms
where Z(i,k) is the impedance of the path the substation shares with buses i and k. The placement is
observable when this has full column rank: 1 + 2 x (unmetered buses other than the substation).

Two rules cut the search:
- Counting: a meter reads 3 numbers (|V|, P, Q) and the state has 65, so at least 22 meters.
- No two neighbouring buses may both be unmetered: turn every bus beyond the line between them by one
  small angle and let the two unknown injections absorb the change in that line's flow. No reading
  changes, exactly and not only to first order. So the unmetered buses are an independent set of
  the tree, and every independent set of a given size is enumerated.
"""

import numpy as np

from . import grid as G
from . import scenarios, wls_np


def shared_impedance(fd: G.Feeder) -> np.ndarray:
    """Z[i, k]: the impedance of the path from the substation that buses i and k share."""
    d = fd.downstream
    return d.T @ (fd.z[:, None] * d)


def neighbours(fd: G.Feeder) -> list[set]:
    adj = [set() for _ in range(fd.n)]
    for a, b in zip(fd.f, fd.t):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def independent_sets(fd: G.Feeder, size: int) -> np.ndarray:
    """Every set of `size` buses no two of which are neighbours, one per row, sorted."""
    adj = neighbours(fd)
    out = []

    def rec(start, chosen, banned):
        if len(chosen) == size:
            out.append(chosen)
            return
        for v in range(start, fd.n):
            if v not in banned:
                rec(v + 1, chosen + [v], banned | adj[v] | {v})

    rec(0, [], set())
    return np.array(out, dtype=np.int16).reshape(-1, size)


def reduced(fd: G.Feeder, Z: np.ndarray, unmetered: np.ndarray) -> np.ndarray:
    """The reduced system for a batch of placements that all leave the substation metered or all leave
    it unmetered: one row per placement in `unmetered` (bus numbers, sorted)."""
    b, s = unmetered.shape
    mask = np.ones((b, fd.n), dtype=bool)
    mask[np.arange(b)[:, None], unmetered] = False
    metered = np.nonzero(mask)[1].reshape(b, fd.n - s)
    slack_metered = bool(mask[0, fd.slack])
    U = unmetered if slack_metered else unmetered[:, 1:]  # the substation's current is not a state
    zc = Z[metered[:, :, None], U[:, None, :]]  # [b, metered, unmetered]
    rows = metered.shape[1] + (2 if slack_metered else 0)
    A = np.zeros((b, rows, 1 + 2 * U.shape[1]))
    A[:, : metered.shape[1], 0] = 1.0
    A[:, : metered.shape[1], 1::2] = -zc.real
    A[:, : metered.shape[1], 2::2] = zc.imag
    if slack_metered:
        A[:, -2, 1::2] = 1.0
        A[:, -1, 2::2] = 1.0
    return A


def margins(fd: G.Feeder, unmetered: np.ndarray, batch: int = 20000) -> np.ndarray:
    """Smallest over largest singular value of each placement's reduced system: 0 when unobservable."""
    Z = shared_impedance(fd)
    out = np.zeros(len(unmetered))
    for slack_unmetered in (True, False):
        idx = np.flatnonzero(np.any(unmetered == fd.slack, axis=1) == slack_unmetered)
        for c in range(0, len(idx), batch):
            part = idx[c : c + batch]
            A = reduced(fd, Z, unmetered[part])
            if A.shape[1] <= A.shape[2] - 1:  # fewer rows than unknowns
                continue
            sv = np.linalg.svd(A, compute_uv=False)
            out[part] = sv[:, -1] / sv[:, 0]
    return out


# A reduced margin below this is exact singularity in rounding (every singular placement of 22 meters
# lies under 1e-12, see `bands`, and the next one up is above 1e-10); above it, the placement is
# observable. The margin is taken at a flat start, the worst case: with no current flowing, a current's
# in-phase and quadrature parts can be told apart only through lines of different R/X, and IEEE 33's
# lines come in a few conductor types. At a real operating point the readings couple more, so a small
# margin says little about accuracy: `vm_spread` measures that instead.
REDUCED_TOL = 1e-10
BANDS = [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]


def bands(m: np.ndarray) -> list[dict]:
    h, _ = np.histogram(np.maximum(m, 0), bins=BANDS)
    return [{"from": lo, "to": hi, "placements": int(c)} for lo, hi, c in zip(BANDS[:-1], BANDS[1:], h)]


def vm_spread(fd: G.Feeder, buses, v: np.ndarray, s: np.ndarray, kind: str = "smart") -> np.ndarray:
    """The standard deviation of each bus's estimated |V| (p.u.) when meters of one kind at `buses`, and
    nothing else, are read at the operating point (v, s): the square root of the diagonal of G^-1,
    taken through the SVD of W^1/2 H, since G itself is too close to singular to invert."""
    nz = scenarios.NOISE[kind]
    rd = wls_np.meter_readings(buses)
    var = np.empty(len(rd))
    for r, (b, k) in enumerate(zip(rd.bus, rd.kind)):
        if k == wls_np.VM:
            var[r] = nz["vm"] ** 2 + scenarios.BIAS**2 / 3
        else:
            x = s[b].real if k == wls_np.P else s[b].imag
            var[r] = ((nz["pq"] * max(abs(x), scenarios.PQ_FLOOR)) ** 2 + (scenarios.BIAS * abs(x)) ** 2 / 3) / fd.sbase**2
    _, H = wls_np.measurement_model(fd.ybus, v, rd, fd.slack)
    _, sv, vt = np.linalg.svd(H / np.sqrt(var)[:, None], full_matrices=False)
    return np.sqrt(np.sum((vt.T / sv) ** 2, axis=1))[fd.n - 1 :]


def pattern(fd: G.Feeder, unmetered: np.ndarray) -> dict:
    """What every placement in `unmetered` (one per row) has in common: buses always metered, lines
    with exactly one end unmetered every time, and how many gaps the rest of the buses share."""
    gap = np.zeros((len(unmetered), fd.n), dtype=bool)
    gap[np.arange(len(unmetered))[:, None], unmetered] = True
    always = [b for b in range(fd.n) if not gap[:, b].any()]
    pairs = [(int(a), int(b)) for a, b in zip(fd.f, fd.t) if np.all(gap[:, a] ^ gap[:, b])]
    paired = {b for p in pairs for b in p}
    rest = [b for b in range(fd.n) if b not in always and b not in paired]
    in_rest = gap[:, rest].sum(1)
    return {"always_metered": always, "one_of_each_pair_unmetered": pairs, "rest": rest,
            "gaps_in_rest": sorted(set(int(x) for x in in_rest))}


def fewest(fd: G.Feeder) -> dict:
    """The fewest meters for which some placement is observable, where they have to be, and how
    accurately they would estimate |V|."""
    out = {"states": 2 * fd.n - 1, "readings_per_meter": 3, "lower_bound_by_counting": int(np.ceil((2 * fd.n - 1) / 3)),
           "reduced_tol": REDUCED_TOL, "by_meters": {}}
    for meters in range(out["lower_bound_by_counting"], fd.n + 1):
        sets = independent_sets(fd, fd.n - meters)
        m = margins(fd, sets)
        ok = m > REDUCED_TOL
        row = {"placements_without_neighbouring_gaps": int(len(sets)), "observable": int(ok.sum()), "bands": bands(m)}
        out["by_meters"][str(meters)] = row
        if ok.any():
            good = sets[ok]
            row["pattern"] = pattern(fd, good)
            row["unmetered_how_often"] = np.bincount(good.ravel(), minlength=fd.n).tolist()
            # the browser's test (QR of the full flat-start Jacobian) on observable and unobservable placements
            rng = np.random.default_rng(0)
            check = np.concatenate([np.flatnonzero(ok)[rng.permutation(ok.sum())[:200]],
                                    np.flatnonzero(~ok)[rng.permutation((~ok).sum())[:200]]])
            agree = [wls_np.observable(fd, set(range(fd.n)) - set(sets[i].tolist())) == bool(ok[i]) for i in check]
            row["browser_test_agrees"] = {"checked": len(check), "agree": int(sum(agree))}
            # how well these few meters would do: smart meters only, at the base case
            g = G.load_gridcal()
            v, s = G.gridcal_power_flow(g, fd.p_load, fd.q_load)
            spread = np.array([vm_spread(fd, set(range(fd.n)) - set(u.tolist()), v, s) for u in good])
            best = int(np.argmin(spread.max(1)))
            every = vm_spread(fd, set(range(fd.n)), v, s)
            row["vm_sd_smart_meters_base_case"] = {
                "best_placement": good[best].tolist(), "best_largest": float(spread[best].max()),
                "best_median": float(np.median(spread[best])),
                "typical_largest": float(np.median(spread.max(1))),  # the median placement's worst bus
                "all_33_largest": float(every.max()), "all_33_median": float(np.median(every)),
            }
            out["fewest"] = meters
            break
    return out

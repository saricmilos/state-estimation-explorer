"""v2's data: operating points of the feeder and what its meters would read.

Kept from v1: the scenario generator (every load times 1 + 0.5 N(0,1)), 2,000 samples, 10 metered
buses per sample chosen at random (2 of them PMUs, each other meter a SCADA unit with probability
0.15, else a smart meter), each meter's noise level, its small bias, and the node-feature layout the
GNN reads (17 numbers per bus).

Fixed (each one a bug in v1, see reports/phase1-v1.md):
- The true injections are the power flow's, in GridCal's sign convention (a load is negative), and
  the substation's is the 3.9 MW it actually supplies. v1 used +load and 0 MW for the substation.
- An unmetered bus gets a pseudo-measurement made only from what an operator knows in advance: the
  bus's base-case (historical) load, with the uncertainty of the scenario generator (50%). v1
  centred it on the sample's true load, gave it a voltage reading centred on the true voltage, and
  set its variance from the true values, (0.3 x true)^2, which handed the model the answer.
- Meter noise on P and Q is relative, as v1's notebook describes it ("smart meter ~5%"); v1's code
  applied 0.05 MW to loads of 0.05-0.4 MW.
- The variance each estimator is told includes the meter's bias, so WLS weighs a reading by the
  error it really has.
- The target is (|V|, angle), which is what v1's loss assumes; v1 stored (Re V, Im V).
- The graph's edges are the 32 energised lines; v1 also passed messages over 5 open tie switches.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from . import grid as G
from . import paths

SENSOR_TYPES = ["none", "scada", "smart", "pmu", "pseudo"]  # v1's one-hot order

# v1's noise levels (std): |V| in p.u. and angle in rad, absolute; P and Q now as a share of the
# reading. v1's SCADA used its fallback (0.01, None, 0.1, 0.1).
NOISE = {
    "pmu": {"vm": 0.001, "va": 0.0005, "pq": 0.02},
    "smart": {"vm": 0.005, "va": None, "pq": 0.05},
    "scada": {"vm": 0.01, "va": None, "pq": 0.10},
}
BIAS = 0.01  # v1's bias_scale: uniform +-0.01 p.u. on |V|, +-0.01 rad on the angle, +-1% on P and Q
PSEUDO_BIAS = 0.05  # v1's pseudo_bias_scale: +-5% on a pseudo-measurement
LOAD_SPREAD = 0.5  # v1's rand_ratio: each load is base x (1 + 0.5 N(0,1))
PQ_FLOOR = 0.001  # MW: noise on a reading near zero is at least this


@dataclass
class Sample:
    v: np.ndarray  # true complex voltages, p.u.
    s: np.ndarray  # true injections, MW + j MVAr (GridCal's convention: load negative)
    obs: np.ndarray  # [N, 4] |V|, angle, P, Q as read (0 where nothing is read)
    mask: np.ndarray  # [N, 4] 1 where read
    var: np.ndarray  # [N, 4] the variance an estimator is told
    sensor: np.ndarray  # [N] index into SENSOR_TYPES


def pseudo_prior(fd: G.Feeder, s_base: np.ndarray):
    """What an operator knows about each bus's injection before any reading: its base-case value
    and how far the scenario generator can take it. The substation supplies the sum of all loads,
    so its spread is that of the sum."""
    mean = s_base.copy()
    sd_p = LOAD_SPREAD * np.abs(-fd.p_load)
    sd_q = LOAD_SPREAD * np.abs(-fd.q_load)
    others = np.arange(fd.n) != fd.slack
    sd_p[fd.slack] = LOAD_SPREAD * np.sqrt(np.sum(fd.p_load[others] ** 2))
    sd_q[fd.slack] = LOAD_SPREAD * np.sqrt(np.sum(fd.q_load[others] ** 2))
    return mean, np.maximum(sd_p, PQ_FLOOR), np.maximum(sd_q, PQ_FLOOR)


def measure(v, s, metered, pmus, prior, rng, scada_share: float = 0.15) -> Sample:
    """Readings for one operating point. `metered` and `pmus` are sets of bus indices."""
    n = len(v)
    vm, va = np.abs(v), np.angle(v)
    obs, mask, var = np.zeros((n, 4)), np.zeros((n, 4)), np.zeros((n, 4))
    sensor = np.full(n, SENSOR_TYPES.index("pseudo"))
    mean, sd_p, sd_q = prior
    for i in range(n):
        if i in metered:
            kind = "pmu" if i in pmus else ("scada" if rng.random() < scada_share else "smart")
            nz = NOISE[kind]
            sensor[i] = SENSOR_TYPES.index(kind)
            obs[i, 0] = vm[i] + rng.normal(0, nz["vm"]) + rng.uniform(-BIAS, BIAS)
            mask[i, 0], var[i, 0] = 1, nz["vm"] ** 2 + BIAS**2 / 3
            if nz["va"] is not None:
                obs[i, 1] = va[i] + rng.normal(0, nz["va"]) + rng.uniform(-BIAS, BIAS)
                mask[i, 1], var[i, 1] = 1, nz["va"] ** 2 + BIAS**2 / 3
            for col, x in ((2, s[i].real), (3, s[i].imag)):
                sd = nz["pq"] * max(abs(x), PQ_FLOOR)
                b = BIAS * abs(x)
                obs[i, col] = x + rng.normal(0, sd) + rng.uniform(-b, b)
                mask[i, col], var[i, col] = 1, sd**2 + b**2 / 3
        else:
            for col, m, sd in ((2, mean[i].real, sd_p[i]), (3, mean[i].imag, sd_q[i])):
                b = PSEUDO_BIAS * abs(m)
                obs[i, col] = m + rng.uniform(-b, b)
                mask[i, col], var[i, col] = 1, sd**2 + b**2 / 3
    return Sample(v=v, s=s, obs=obs, mask=mask, var=var, sensor=sensor)


def place_meters(n: int, rng, n_metered: int = 10, n_pmu: int = 2):
    """v1's placement: n_metered buses at random, n_pmu of those are PMUs."""
    metered = set(rng.choice(n, size=n_metered, replace=False).tolist())
    pmus = set(rng.choice(sorted(metered), size=min(n_pmu, n_metered), replace=False).tolist())
    return metered, pmus


def to_data(fd: G.Feeder, smp: Sample) -> Data:
    """The PyG graph the GNN reads, in v1's layout: x = readings(4) | mask(4) | variance(4) | type(5)."""
    onehot = np.eye(len(SENSOR_TYPES))[smp.sensor]
    x = np.concatenate([smp.obs, smp.mask, smp.var, onehot], axis=1)
    ei = np.concatenate([np.stack([fd.f, fd.t]), np.stack([fd.t, fd.f])], axis=1)
    ea = np.concatenate([np.stack([fd.r, fd.x], 1)] * 2)
    y = np.stack([np.abs(smp.v), np.angle(smp.v)], axis=1)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(ei, dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        meas=torch.tensor(smp.obs, dtype=torch.float32),
        meas_mask=torch.tensor(smp.mask, dtype=torch.float32),
        meas_var=torch.tensor(smp.var, dtype=torch.float32),
        sensor_type=torch.tensor(smp.sensor, dtype=torch.long),
        is_pseudo=torch.tensor(smp.sensor == SENSOR_TYPES.index("pseudo"), dtype=torch.float32),
    )


def generate(num_samples: int = 2000, seed: int = 42, n_metered=10, n_pmu=2):
    """Operating points and readings, with the true states from GridCal's Newton-Raphson.
    `n_metered` may be a callable rng -> int and `n_pmu` a callable (rng, meters) -> int (for a model
    that sees every number of meters); an int `n_pmu` is capped at the number of meters."""
    g = G.load_gridcal()
    fd = G.feeder(g)
    _, s_base = G.gridcal_power_flow(g, fd.p_load, fd.q_load)
    prior = pseudo_prior(fd, s_base)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(num_samples):
        factor = 1.0 + LOAD_SPREAD * rng.normal(size=fd.n)
        v, s = G.gridcal_power_flow(g, fd.p_load * factor, fd.q_load * factor)
        k = n_metered(rng) if callable(n_metered) else n_metered
        pmu = n_pmu(rng, k) if callable(n_pmu) else min(n_pmu, k)
        metered, pmus = place_meters(fd.n, rng, k, pmu)
        samples.append(measure(v, s, metered, pmus, prior, rng))
    return fd, samples


def any_meters(rng) -> int:
    """For the website's GNN: 0 to 33 meters, each count equally likely."""
    return int(rng.integers(0, 34))


def any_pmus(rng, meters: int) -> int:
    """0 to 2 PMUs among the meters, each count equally likely, never more than the meters."""
    return int(rng.integers(0, min(2, meters) + 1))


def dataset(num_samples: int = 2000, seed: int = 42, name: str = "v2", n_metered=10, n_pmu=2):
    """The samples and their graphs, made once and cached in data/."""
    cache = paths.DATA / f"{name}_33bw_{num_samples}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    fd, samples = generate(num_samples, seed, n_metered, n_pmu)
    out = (fd, samples, [to_data(fd, s) for s in samples])
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out

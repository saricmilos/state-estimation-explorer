"""The feeder: the IEEE 33-bus system of Baran and Wu, as GridCal reads case33bw.m.

GridCal already converts the file's ohms to per-unit on 100 MVA (line 1-2: 0.0922 ohm becomes
0.0575 p.u.). v1 converted them a second time (notebook cell 22), which left every line at 1/1.6028
of its impedance; v2 uses the lines as loaded, and the base case comes out as published: lowest
voltage 0.9131 p.u. at bus 18 (index 17) and 202.7 kW of losses.

Five tie switches in the file are open (status 0); v2 leaves them out of the network the models see,
so it is the radial feeder that is actually energised.
"""

import contextlib
import io
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import paths, v1


@dataclass(frozen=True)
class Feeder:
    n: int
    sbase: float  # MVA
    f: np.ndarray  # from bus of each energised line, the end nearer the substation
    t: np.ndarray  # to bus
    r: np.ndarray  # p.u.
    x: np.ndarray  # p.u.
    p_load: np.ndarray  # base-case load per bus, MW
    q_load: np.ndarray  # MVAr
    slack: int = 0

    @property
    def z(self) -> np.ndarray:
        return self.r + 1j * self.x

    @cached_property
    def parent_line(self) -> np.ndarray:
        """For each bus, the line that feeds it (-1 for the substation)."""
        out = -np.ones(self.n, dtype=int)
        out[self.t] = np.arange(len(self.t))
        return out

    @cached_property
    def order(self) -> np.ndarray:
        """Buses from the substation outwards, each after the bus that feeds it."""
        children = {i: [] for i in range(self.n)}
        for k, (a, b) in enumerate(zip(self.f, self.t)):
            children[a].append(b)
        out, stack = [], [self.slack]
        while stack:
            b = stack.pop(0)
            out.append(b)
            stack.extend(children[b])
        return np.array(out)

    @cached_property
    def downstream(self) -> np.ndarray:
        """D[l, k] = 1 when bus k is fed through line l: the current in line l is the sum of the
        currents drawn at those buses (Kirchhoff's current law on a radial feeder)."""
        d = np.zeros((len(self.t), self.n))
        for b in self.order[1:]:
            k = b
            while k != self.slack:
                line = self.parent_line[k]
                d[line, b] = 1.0
                k = self.f[line]
        return d

    @cached_property
    def ybus(self) -> np.ndarray:
        y = np.zeros((self.n, self.n), dtype=complex)
        ys = 1.0 / self.z
        for a, b, v in zip(self.f, self.t, ys):
            y[a, a] += v
            y[b, b] += v
            y[a, b] -= v
            y[b, a] -= v
        return y


def load_gridcal():
    """case33bw.m through v1's loader (which only renames the buses 0..32), without v1's second
    conversion of the lines."""
    return v1.module("io").load_network_renamed("case33bw.m", sbase=100, folder=str(paths.NETWORKS))


def feeder(grid=None) -> Feeder:
    grid = grid or load_gridcal()
    idx = {b.idtag: i for i, b in enumerate(grid.buses)}
    lines = [ln for ln in grid.lines if ln.active]
    f = np.array([idx[ln.bus_from.idtag] for ln in lines])
    t = np.array([idx[ln.bus_to.idtag] for ln in lines])
    p = np.zeros(len(grid.buses))
    q = np.zeros(len(grid.buses))
    for ld in grid.loads:
        p[idx[ld.bus.idtag]] += ld.P
        q[idx[ld.bus.idtag]] += ld.Q
    fd = Feeder(
        n=len(grid.buses), sbase=float(grid.Sbase), f=f, t=t,
        r=np.array([ln.R for ln in lines]), x=np.array([ln.X for ln in lines]),
        p_load=p, q_load=q,
    )
    # A tree fed from bus 0 with every line written from the substation side, or the sweep is wrong
    reached = set(fd.order.tolist())
    if len(reached) != fd.n or len(lines) != fd.n - 1:
        raise ValueError("the energised lines are not a radial tree fed from bus 0")
    return fd


def sweep(fd: Feeder, p_load, q_load, v0: complex = 1.0, tol: float = 1e-12, max_iter: int = 100):
    """Backward/forward sweep power flow for the radial feeder. Loads in MW/MVAr (constant power),
    one row per scenario. Returns complex bus voltages in p.u., shape [scenarios, buses]."""
    s = (np.atleast_2d(p_load) + 1j * np.atleast_2d(q_load)) / fd.sbase
    v = np.full(s.shape, v0, dtype=complex)
    d = fd.downstream
    for _ in range(max_iter):
        i_load = np.conj(s / v)  # current drawn at each bus
        i_line = i_load @ d.T  # backward: each line carries what is drawn beyond it
        v_new = np.empty_like(v)
        v_new[:, fd.slack] = v0
        for b in fd.order[1:]:  # forward: voltage drop along each line
            line = fd.parent_line[b]
            v_new[:, b] = v_new[:, fd.f[line]] - fd.z[line] * i_line[:, line]
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        v = v_new
    raise RuntimeError("sweep did not converge")


def gridcal_power_flow(grid, p_load, q_load):
    """GridCal's Newton-Raphson power flow (the thesis's solver) with the loads set to p_load,
    q_load (MW, MVAr, one value per bus). Returns complex voltages (p.u.) and complex bus
    injections Sbus (MW + j MVAr, positive into the grid), as GridCal reports them."""
    idx = {b.idtag: i for i, b in enumerate(grid.buses)}
    for ld in grid.loads:
        ld.P = float(p_load[idx[ld.bus.idtag]])
        ld.Q = float(q_load[idx[ld.bus.idtag]])
    with contextlib.redirect_stdout(io.StringIO()):
        res = v1.module("powerflow").run_power_flow(grid)
    if not res.results.converged:
        raise RuntimeError("power flow did not converge")
    return np.array(res.results.voltage), np.array(res.results.Sbus)

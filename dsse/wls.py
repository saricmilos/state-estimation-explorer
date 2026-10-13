"""WLS state estimation by GridCal, the estimator the thesis used (notebook step 4), on v2's readings.

GridCal's estimator takes |V|, P and Q at buses (P and Q in MW, positive into the grid); it has no
voltage-angle measurement, so a PMU's angle is not used by it, as in v1. It solves with
Levenberg-Marquardt, which always returns an answer, so whether the readings determine the state
(observability) has to be checked separately.
"""

import GridCalEngine as gce
import numpy as np

from . import grid as G
from .scenarios import Sample


class Estimator:
    """One GridCal grid whose measurements are replaced for each sample (copying the grid for every
    sample, as v1's generator does, takes most of a second)."""

    def __init__(self, grid=None):
        self.grid = grid or G.load_gridcal()

    def _clear(self):
        g = self.grid
        g.vm_measurements, g.pi_measurements, g.qi_measurements = [], [], []

    def __call__(self, smp: Sample) -> np.ndarray:
        """Complex bus voltages estimated from the sample's readings."""
        g = self.grid
        self._clear()
        sd = np.sqrt(smp.var)
        for i, bus in enumerate(g.buses):
            if smp.mask[i, 0]:
                g.add_vm_measurement(gce.Devices.VmMeasurement(float(smp.obs[i, 0]), float(sd[i, 0]), bus))
            if smp.mask[i, 2]:
                g.add_pi_measurement(gce.Devices.PiMeasurement(float(smp.obs[i, 2]), float(sd[i, 2]), bus))
            if smp.mask[i, 3]:
                g.add_qi_measurement(gce.Devices.QiMeasurement(float(smp.obs[i, 3]), float(sd[i, 3]), bus))
        se = gce.StateEstimation(circuit=g)
        se.run()
        self._clear()
        return np.array(se.results.voltage)


def lm_steps(est: Estimator, smp: Sample) -> str:
    """Which of GridCal's Levenberg-Marquardt steps on one sample were taken ('+') and which refused ('-').

    GridCal 5.3.53 (solve_se_lm) judges each new step by how much the previous one lowered the
    objective: the objective it compares is the one at the current state, before the new step. So a
    step that raises the objective is kept, and the step after it is refused; the state then stops
    changing, every later ratio is exactly 0, every later step is refused, and the damping doubles
    until the step is below the tolerance, where GridCal reports convergence. Recorded by watching
    its Jacobian, which it forms once before the loop and again after each step it takes."""
    import sys

    mod = sys.modules["GridCalEngine.Simulations.StateEstimation.state_estimation"]
    events = []
    jacobian, solve = mod.Jacobian_SE, mod.spsolve

    def jacobian_spy(*a):
        events.append("J")
        return jacobian(*a)

    def solve_spy(*a):
        events.append("s")
        return solve(*a)

    mod.Jacobian_SE, mod.spsolve = jacobian_spy, solve_spy
    try:
        est(smp)
    finally:
        mod.Jacobian_SE, mod.spsolve = jacobian, solve
    steps = "".join(events[1:]).split("s")[1:]
    return "".join("+" if s == "J" else "-" for s in steps)

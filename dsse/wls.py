"""WLS state estimation by GridCal, the estimator the thesis used (notebook step 4), on v2's readings.

GridCal's estimator takes |V|, P and Q at buses (P and Q in MW, positive into the grid); it has no
voltage-angle measurement, so a PMU's angle is not used by it, as in v1. It solves with
Levenberg-Marquardt, which always returns an answer, so whether the readings determine the state
(observability) has to be checked separately.
"""

import copy

import GridCalEngine as gce
import numpy as np

from .scenarios import Sample


def estimate(template, smp: Sample) -> np.ndarray:
    """Complex bus voltages estimated from the sample's readings. `template` is a GridCal grid
    without measurements; it is copied, never changed."""
    g = copy.deepcopy(template)
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
    return np.array(se.results.voltage)

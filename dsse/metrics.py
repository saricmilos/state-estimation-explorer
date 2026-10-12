"""Scoring, two ways.

The thesis way (Tables 8 and 9): take the first test batch of 32 samples, average the predictions over
the 32 and the true states over the 32, and compare the two averaged profiles bus by bus. Errors of
opposite sign in different samples cancel, so what remains is mostly a model's constant offset: a
model that always answers the average voltage profile scores close to zero.

The v2 way: every bus of every test sample, errors taken before any averaging.
"""

import numpy as np


def angle_diff(pred, true):
    """Difference wrapped to (-pi, pi], as the notebook's angle_diff."""
    return np.angle(np.exp(1j * (np.asarray(pred) - np.asarray(true))))


def per_sample(vm_pred, va_pred, vm_true, va_true) -> dict:
    """Arrays of shape [samples, buses]; Vm in p.u., angles in radians."""
    e = np.asarray(vm_pred, dtype=float) - np.asarray(vm_true, dtype=float)
    ea = angle_diff(va_pred, va_true)
    return {
        "vm_mae": float(np.mean(np.abs(e))),
        "vm_rmse": float(np.sqrt(np.mean(e**2))),
        "vm_p95": float(np.percentile(np.abs(e), 95)),
        "vm_max": float(np.max(np.abs(e))),
        "vm_bias": float(np.mean(e)),
        "va_mae": float(np.mean(np.abs(ea))),
        "va_rmse": float(np.sqrt(np.mean(ea**2))),
        "samples": int(e.shape[0]),
    }


def thesis_way(vm_pred, va_pred, vm_true, va_true, batch: int = 32) -> dict:
    """The notebook's evaluation, applied to the first `batch` test samples."""
    vp, ap = np.mean(vm_pred[:batch], axis=0), np.mean(va_pred[:batch], axis=0)
    vt, at = np.mean(vm_true[:batch], axis=0), np.mean(va_true[:batch], axis=0)
    e, ea = vp - vt, angle_diff(ap, at)
    return {
        "vm_mae": float(np.mean(np.abs(e))),
        "vm_rmse": float(np.sqrt(np.mean(e**2))),
        "va_mae": float(np.mean(np.abs(ea))),
        "va_rmse": float(np.sqrt(np.mean(ea**2))),
    }

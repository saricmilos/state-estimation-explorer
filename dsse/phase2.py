"""Phase 2: v2, the thesis with its bugs fixed, on the same kind of test.

    python -m dsse.phase2 gnn --seed 0     # train the thesis's GNN on v2's data
    python -m dsse.phase2 pinn --seed 0    # the PINN, with the physics term fixed
    python -m dsse.phase2 wls              # GridCal's WLS on the same 400 test samples

Same split as v1: the first 1,600 samples train, the last 400 are the test set, nothing is tuned.
Each run writes data/v2/<what>/ (weights, predictions, result.json); dsse.evaluate reports them.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import grid as G
from . import metrics, paths, scenarios, train, wls

RUNS = paths.DATA / "v2"
N_TRAIN = 1600


def split():
    fd, samples, data = scenarios.dataset()
    return fd, samples, data[:N_TRAIN], data[N_TRAIN:], samples[N_TRAIN:]


def truth(test_samples):
    v = np.stack([s.v for s in test_samples])
    return np.abs(v), np.angle(v)


def run_model(kind: str, seed: int, alpha: float = train.ALPHA, tag: str = "") -> dict:
    fd, _, tr, te, te_s = split()
    out_dir = RUNS / f"{kind}{tag}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    t = time.time()
    model, norm = train.fit(tr, kind, seed, fd, alpha=alpha, log=lines.append)
    secs = time.time() - t
    (out_dir / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
    torch.save({"model": model.state_dict(), "norm": norm.state()}, out_dir / "model.pt")
    vm_p, va_p, res = train.predict(model, norm, te, fd)
    vm_t, va_t = truth(te_s)
    np.savez_compressed(out_dir / "pred.npz", vm=vm_p, va=va_p)
    result = {
        "model": kind, "seed": seed, "alpha": alpha if kind == "pinn" else None,
        "train_seconds": round(secs, 1),
        "per_sample": metrics.per_sample(vm_p, va_p, vm_t, va_t),
        "thesis_way": metrics.thesis_way(vm_p, va_p, vm_t, va_t),
        "per_bus_vm_mae": np.mean(np.abs(vm_p - vm_t), axis=0).tolist(),
        "physics_residual_mean": float(res.mean()),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_wls() -> dict:
    fd, _, _, _, te_s = split()
    template = G.load_gridcal()
    t = time.time()
    v = np.stack([wls.estimate(template, s) for s in te_s])
    secs = time.time() - t
    vm_t, va_t = truth(te_s)
    out_dir = RUNS / "wls"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pred.npz", vm=np.abs(v), va=np.angle(v))
    result = {
        "model": "wls", "seconds": round(secs, 1),
        "per_sample": metrics.per_sample(np.abs(v), np.angle(v), vm_t, va_t),
        "thesis_way": metrics.thesis_way(np.abs(v), np.angle(v), vm_t, va_t),
        "per_bus_vm_mae": np.mean(np.abs(np.abs(v) - vm_t), axis=0).tolist(),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gnn", "pinn", "wls"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    res = run_wls() if a.what == "wls" else run_model(a.what, a.seed)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_bus")}, indent=2))

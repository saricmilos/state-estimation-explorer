"""Phase 2: v2, the thesis with its bugs fixed, on the same kind of test.

    python -m dsse.phase2 gnn --seed 0                    # the thesis's GNN and training, v2's data
    python -m dsse.phase2 pinn --seed 0                   # the PINN, physics term fixed
    python -m dsse.phase2 gnn --recipe scaled --seed 0    # training fixed (see dsse.train)
    python -m dsse.phase2 select-alpha --alpha A          # one candidate for the PINN's weight, on validation
    python -m dsse.phase2 pinn --recipe scaled --alpha A --seed 0
    python -m dsse.phase2 wls                             # GridCal's WLS on the same 400 test samples
    python -m dsse.phase2 configs                         # every trained model and WLS, 3 to 20 meters

Same split as v1: the first 1,600 samples train, the last 400 are the test set. The one setting
chosen here, the PINN's alpha under the fixed training, is chosen on samples 1,400-1,599 with the
model trained on the first 1,400; the test set plays no part in it.
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
N_FIT = 1400  # for choosing alpha: fit on the first 1,400, validate on the next 200
ALPHAS = [0.1, 1.0, 10.0]


def split():
    fd, samples, data = scenarios.dataset()
    return fd, samples, data[:N_TRAIN], data[N_TRAIN:], samples[N_TRAIN:]


def truth(test_samples):
    v = np.stack([s.v for s in test_samples])
    return np.abs(v), np.angle(v)


def run_dir(kind: str, recipe: str, seed: int):
    return RUNS / (f"{kind}_seed{seed}" if recipe == "thesis" else f"{kind}-{recipe}_seed{seed}")


def run_model(kind: str, seed: int, recipe: str = "thesis", alpha: float = train.ALPHA) -> dict:
    fd, samples, tr, te, te_s = split()
    out_dir = run_dir(kind, recipe, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    t = time.time()
    model, norm = train.fit(tr, kind, seed, fd, recipe=recipe, alpha=alpha, log=lines.append)
    secs = time.time() - t
    (out_dir / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
    torch.save({"model": model.state_dict(), "norm": norm.state(), "recipe": recipe}, out_dir / "model.pt")
    vm_p, va_p, res = train.predict(model, norm, te, fd, recipe)
    vm_t, va_t = truth(te_s)
    np.savez_compressed(out_dir / "pred.npz", vm=vm_p, va=va_p)
    # the model's error on its own training samples: a converged model has no constant offset there
    vm_tr, va_tr, _ = train.predict(model, norm, tr[:400], fd, recipe)
    vm_trt, va_trt = truth(samples[:400])
    result = {
        "model": kind, "recipe": recipe, "seed": seed, "alpha": alpha if kind == "pinn" else None,
        "on_training_samples": metrics.per_sample(vm_tr, va_tr, vm_trt, va_trt),
        "train_seconds": round(secs, 1),
        "per_sample": metrics.per_sample(vm_p, va_p, vm_t, va_t),
        "thesis_way": metrics.thesis_way(vm_p, va_p, vm_t, va_t),
        "per_bus_vm_mae": np.mean(np.abs(vm_p - vm_t), axis=0).tolist(),
        "physics_residual_mean": float(res.mean()),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def select_alpha(alpha: float, seed: int = 0) -> dict:
    """One candidate for the PINN's alpha under the fixed training (alpha 0 is the GNN), scored by
    |V| MAE on validation samples 1,400-1,599 with the model fitted on the first 1,400."""
    fd, samples, tr, *_ = split()
    fit, val, val_s = tr[:N_FIT], tr[N_FIT:], samples[N_FIT:N_TRAIN]
    vm_t, va_t = truth(val_s)
    kind = "gnn" if alpha == 0.0 else "pinn"
    model, norm = train.fit(fit, kind, seed, fd, recipe="scaled", alpha=alpha, log=lambda _: None)
    vm_p, va_p, res = train.predict(model, norm, val, fd, "scaled")
    out = metrics.per_sample(vm_p, va_p, vm_t, va_t) | {"alpha": alpha, "physics_residual_mean": float(res.mean())}
    folder = RUNS / "select_alpha"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"alpha_{alpha:g}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def chosen_alpha() -> float:
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in (RUNS / "select_alpha").glob("alpha_*.json")]
    rows = [r for r in rows if r["alpha"] in ALPHAS]
    if len(rows) != len(ALPHAS):
        raise RuntimeError("not every alpha has been validated yet")
    return min(rows, key=lambda r: r["vm_mae"])["alpha"]


def run_wls() -> dict:
    fd, _, _, _, te_s = split()
    est = wls.Estimator()
    t = time.time()
    v = np.stack([est(s) for s in te_s])
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


METER_COUNTS = [3, 5, 10, 15, 20]


def load_model(run_dir):
    gnn_cls, pinn_cls = train.thesis_classes()
    saved = torch.load(run_dir / "model.pt", weights_only=False)
    kind = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["model"]
    recipe = saved.get("recipe", "thesis")
    model = (gnn_cls if kind == "gnn" else pinn_cls)(17, train.HIDDEN, train.OUTPUTS[recipe], 2)  # v1's 17 features, (R, X)
    model.load_state_dict(saved["model"])
    return kind, recipe, model, train.Normalizer.from_state(saved["norm"])


def run_configs(num_samples: int = 200) -> dict:
    """The thesis concludes that the learned estimator is robust to 'variations in ... measurement
    configurations', but its models only ever saw 10 meters. Here the models trained on 10 meters
    (and WLS) estimate test samples with 3 to 20 meters: the same 2 PMUs, more or fewer other meters.
    Each meter count is its own set of samples, from its own seed."""
    fd, samples, *_ = split()
    v_tr = np.stack([s.v for s in samples[:N_TRAIN]])
    avg_vm, avg_va = np.abs(v_tr).mean(0), np.angle(v_tr).mean(0)
    models = [load_model(d) + (d.name,) for d in sorted(RUNS.glob("*_seed*")) if (d / "model.pt").exists()]
    est = wls.Estimator()
    out = {}
    for k in METER_COUNTS:
        cache = paths.DATA / f"v2_configs_k{k}_{num_samples}.pt"
        if cache.exists():
            sm = torch.load(cache, weights_only=False)
        else:
            _, sm = scenarios.generate(num_samples, seed=500 + k, n_metered=k)
            torch.save(sm, cache)
        data = [scenarios.to_data(fd, s) for s in sm]
        vm_t, va_t = truth(sm)
        row = {"average_profile": metrics.per_sample(np.broadcast_to(avg_vm, vm_t.shape),
                                                     np.broadcast_to(avg_va, va_t.shape), vm_t, va_t)}
        v = np.stack([est(s) for s in sm])
        row["wls"] = metrics.per_sample(np.abs(v), np.angle(v), vm_t, va_t)
        for kind, recipe, model, norm, name in models:
            vm_p, va_p, _ = train.predict(model, norm, data, fd, recipe)
            row[name] = metrics.per_sample(vm_p, va_p, vm_t, va_t)
        out[str(k)] = row
        print(k, {n: round(r["vm_mae"], 4) for n, r in row.items()}, flush=True)
    (RUNS / "configs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gnn", "pinn", "wls", "configs", "select-alpha"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recipe", choices=["thesis", "scaled"], default="thesis")
    p.add_argument("--alpha", type=float, default=train.ALPHA)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.what == "configs":
        run_configs()
    elif a.what == "select-alpha":
        print(select_alpha(a.alpha))
    else:
        res = run_wls() if a.what == "wls" else run_model(a.what, a.seed, a.recipe, a.alpha)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_bus")}, indent=2))

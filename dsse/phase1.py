"""Phase 1: re-run v1 from its own code and check what it reports.

    python -m dsse.phase1 gnn --seed 0     # train v1's GNN once (seed for torch)
    python -m dsse.phase1 pinn --seed 0    # train v1's PINN once
    python -m dsse.phase1 wls              # v1's WLS on its own measurement draws (Table 7)

Each run writes data/v1/<model>_seed<n>/result.json; dsse.evaluate turns them into the report.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import metrics, paths, v1

RUNS = paths.DATA / "v1"


def _true_state(y):
    """v1 stores the target as (Re V, Im V) and its loss reads it as (|V|, angle). For scoring, the
    true magnitude and angle are computed from what it stores."""
    v = y[..., 0] + 1j * y[..., 1]
    return np.abs(v), np.angle(v)


def predict(ns, dataset, scramble=None, rng=None):
    """Run a trained v1 model over `dataset` with the notebook's normalisation.

    scramble: None, or a list of feature columns whose values at pseudo-measured buses are shuffled
    across samples (same bus, another sample): the leak test. A model that reads the truth from
    those columns gets worse; one that doesn't is unaffected."""
    from torch_geometric.data import Batch

    model = ns["model"]
    model.eval()
    xs = torch.stack([d.x for d in dataset])  # [S, N, F]
    if scramble:
        xs = xs.clone()
        pseudo = torch.stack([d.is_pseudo for d in dataset]).bool()  # [S, N]
        for b in range(xs.shape[1]):
            rows = torch.nonzero(pseudo[:, b]).flatten()
            perm = rows[torch.as_tensor(rng.permutation(len(rows)))]
            for col in scramble:
                xs[rows, b, col] = xs[perm, b, col]
    out = []
    with torch.no_grad():
        for i in range(0, len(dataset), 100):
            chunk = []
            for j, d in enumerate(dataset[i : i + 100]):
                d = d.clone()
                d.x = xs[i + j]
                chunk.append(d)
            batch = ns["normalize_batch"](Batch.from_data_list(chunk))
            out.append(model(batch).reshape(len(chunk), -1, 3).numpy())
    out = np.concatenate(out)
    return out[..., 0], np.arctan2(out[..., 1], out[..., 2])


def score(ns, kind: str) -> dict:
    test, train = ns["test_dataset"], ns["train_dataset"]
    vm_t, va_t = _true_state(np.stack([d.y.numpy() for d in test]))
    vm_p, va_p = predict(ns, test)
    vm_tr, va_tr = _true_state(np.stack([d.y.numpy() for d in train]))
    avg_vm, avg_va = vm_tr.mean(0), va_tr.mean(0)
    rng = np.random.default_rng(0)
    leak = {}
    for name, cols in {"variances": [8, 9, 10, 11], "pseudo_values": [0, 1, 2, 3]}.items():
        vm_s, va_s = predict(ns, test, scramble=cols, rng=rng)
        leak[name] = metrics.per_sample(vm_s, va_s, vm_t, va_t)
    const = lambda a: np.broadcast_to(a, vm_t.shape)  # noqa: E731
    result = {
        "thesis_way_notebook": (v1.thesis_metrics(ns) if kind == "gnn" else v1.pinn_thesis_metrics(ns)),
        "thesis_way": metrics.thesis_way(vm_p, va_p, vm_t, va_t),
        "per_sample": metrics.per_sample(vm_p, va_p, vm_t, va_t),
        "leak_test": leak,
        "average_profile": {
            "thesis_way": metrics.thesis_way(const(avg_vm), const(avg_va), vm_t, va_t),
            "per_sample": metrics.per_sample(const(avg_vm), const(avg_va), vm_t, va_t),
        },
    }
    return result


def train(kind: str, seed: int) -> dict:
    grid = v1.load_grid()
    data = v1.dataset(grid)
    ns = v1.prepare(data, seed)
    run_dir = RUNS / f"{kind}_seed{seed}"
    t = time.time()
    if kind == "gnn":
        v1.train_gnn(ns, run_dir)
    else:
        v1.train_pinn(ns, run_dir, grid)
    result = {"model": kind, "seed": seed, "train_seconds": round(time.time() - t, 1)} | score(ns, kind)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def wls(draws: int = 30) -> dict:
    """Table 7 again: GridCal's WLS on v1's measurements of the base case, `draws` times, then on the
    same draws with the injection bugs undone."""
    out = []
    for s in range(draws):
        row = {"seed": 1000 + s}
        for key, fix in (("as_is", False), ("injections_fixed", True)):
            r = v1.wls_one_sample(seed=1000 + s, fix_injections=fix)
            e = r["v_est"] - r["v_true"]
            row[key] = {"vm_mae": float(np.mean(np.abs(e))), "vm_max": float(np.max(np.abs(e)))}
        row["metered"] = [int(i) for i in np.flatnonzero(~r["meas"]["is_pseudo"])]
        out.append(row)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "wls_table7.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {k: float(np.median([o[k]["vm_mae"] for o in out])) for k in ("as_is", "injections_fixed")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gnn", "pinn", "wls"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    res = wls() if a.what == "wls" else train(a.what, a.seed)
    print(json.dumps({k: v for k, v in res.items() if k != "leak_test"}, indent=2)[:3000])

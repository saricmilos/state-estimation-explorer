"""Phase 2: v2, the thesis with its bugs fixed, on the same kind of test.

    python -m dsse.phase2 gnn --seed 0                    # the thesis's GNN and training, v2's data
    python -m dsse.phase2 pinn --seed 0                   # the PINN, physics term fixed
    python -m dsse.phase2 gnn --recipe scaled --seed 0    # training fixed (see dsse.train)
    python -m dsse.phase2 select-alpha --alpha A          # one candidate for the PINN's weight, on validation
    python -m dsse.phase2 pinn --recipe scaled --alpha A --seed 0
    python -m dsse.phase2 wls                             # GridCal's WLS on the same 400 test samples
    python -m dsse.phase2 wls-np                          # the browser's WLS (numpy) next to GridCal's; bad data
    python -m dsse.phase2 observability                   # the fewest real meters that determine the state
    python -m dsse.phase2 anymeter --seed 0               # the website's GNN, trained on 0 to 33 meters
    python -m dsse.phase2 configs                         # every trained model and WLS, 0 to 33 meters

Same split as v1: the first 1,600 samples train, the last 400 are the test set. The one setting
chosen here, the PINN's alpha under the fixed training, is chosen on samples 1,400-1,599 with the
model trained on the first 1,400; the test set plays no part in it.
Each run writes data/v2/<what>/ (weights, predictions, result.json); dsse.evaluate reports them.
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from . import grid as G
from . import metrics, observability, paths, scenarios, train, wls, wls_np

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


# The website's GNN, "gnn-anymeter": every model above saw exactly 10 meters, and a visitor may place
# from none to all 33. Same class and fixed training as gnn-scaled, on its own data: 4,000 samples,
# each with 0 to 33 meters (0 to 2 of them PMUs), the last 20% its test set. Each seed is fitted on
# the first 3,000 samples; the browser gets the seed with the lowest |V| error on the next 200 (the
# last 200 of the training samples). Neither test set plays a part in that choice.
ANY_SAMPLES, ANY_SEED = 4000, 2026
ANY_TRAIN = ANY_SAMPLES * 4 // 5
ANY_FIT = ANY_TRAIN - 200
ANY_SEEDS = [0, 1, 2]


def anymeter_split():
    fd, samples, data = scenarios.dataset(ANY_SAMPLES, ANY_SEED, "v2-anymeter", scenarios.any_meters, scenarios.any_pmus)
    return fd, samples, data


def run_anymeter(seed: int) -> dict:
    fd, samples, data = anymeter_split()
    out_dir = RUNS / f"gnn-anymeter_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    t = time.time()
    model, norm = train.fit(data[:ANY_FIT], "gnn", seed, fd, recipe="scaled", log=lines.append)
    secs = time.time() - t
    (out_dir / "train_log.txt").write_text("\n".join(lines), encoding="utf-8")
    torch.save({"model": model.state_dict(), "norm": norm.state(), "recipe": "scaled"}, out_dir / "model.pt")

    def score(d, s):
        vm_p, va_p, _ = train.predict(model, norm, d, fd, "scaled")
        return metrics.per_sample(vm_p, va_p, *truth(s))

    # v2's 400 test samples (10 meters), the set every other model is scored on
    _, _, _, te, te_s = split()
    vm_p, va_p, res = train.predict(model, norm, te, fd, "scaled")
    vm_t, va_t = truth(te_s)
    np.savez_compressed(out_dir / "pred.npz", vm=vm_p, va=va_p)
    result = {
        "model": "gnn", "recipe": "scaled", "seed": seed, "alpha": None, "meters": "0 to 33",
        "validation": score(data[ANY_FIT:ANY_TRAIN], samples[ANY_FIT:ANY_TRAIN]),
        "own_test_set": score(data[ANY_TRAIN:], samples[ANY_TRAIN:]),
        "on_training_samples": score(data[:400], samples[:400]),
        "train_seconds": round(secs, 1),
        "per_sample": metrics.per_sample(vm_p, va_p, vm_t, va_t),
        "thesis_way": metrics.thesis_way(vm_p, va_p, vm_t, va_t),
        "per_bus_vm_mae": np.mean(np.abs(vm_p - vm_t), axis=0).tolist(),
        "physics_residual_mean": float(res.mean()),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def chosen_anymeter() -> int:
    """The seed the browser gets: the lowest validation |V| error."""
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in RUNS.glob("gnn-anymeter_seed*/result.json")]
    rows = [r for r in rows if r["seed"] in ANY_SEEDS]
    if len(rows) != len(ANY_SEEDS):
        raise RuntimeError("not every seed of gnn-anymeter has been trained yet")
    return min(rows, key=lambda r: r["validation"]["vm_mae"])["seed"]


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


def run_wls_np() -> dict:
    """The browser's WLS (dsse.wls_np) on the 400 test samples, next to GridCal's estimates of the
    same samples (data/v2/wls, from `phase2 wls`), without and with the PMUs' angles; then its
    bad-data tests: how often they fire on the clean samples, and whether they find one gross error
    (10 standard deviations on one real meter's |V|, P or Q, chosen at random) put into each."""
    fd, _, _, _, te_s = split()
    gc = np.load(RUNS / "wls" / "pred.npz")
    vm_t, va_t = truth(te_s)
    t = time.time()
    ests = [wls_np.estimate(fd, s) for s in te_s]
    secs = time.time() - t
    vm, va = np.stack([np.abs(e.v) for e in ests]), np.stack([np.angle(e.v) for e in ests])
    j_np = np.array([e.objective for e in ests])
    j_gc = np.array([wls_np.objective(fd, e.readings, gc["vm"][i] * np.exp(1j * gc["va"][i])) for i, e in enumerate(ests)])
    stalled = np.flatnonzero(j_gc > j_np * (1 + 1e-6))
    reached = np.setdiff1d(np.arange(len(te_s)), stalled)
    dvm, dva = np.abs(vm - gc["vm"]), np.abs(va - gc["va"])

    def gaps(rows):
        return {"vm_max": float(dvm[rows].max()), "vm_mean": float(dvm[rows].mean()),
                "va_max": float(dva[rows].max()), "va_mean": float(dva[rows].mean()), "samples": int(len(rows))}

    est = wls.Estimator()
    steps = [wls.lm_steps(est, te_s[i]) for i in stalled]
    ang = [wls_np.estimate(fd, s, angles=True) for s in te_s]
    vm_a, va_a = np.stack([np.abs(e.v) for e in ang]), np.stack([np.angle(e.v) for e in ang])

    clean = [wls_np.estimate_clean(fd, s) for s in te_s]
    pseudo = scenarios.SENSOR_TYPES.index("pseudo")
    # how much of a real meter's own error shows in its residual, by kind of reading
    share = {k: [] for k in (wls_np.VM, wls_np.P, wls_np.Q)}
    for s, e in zip(te_s, ests):
        ratio = wls_np.residual_variance(e) / e.readings.var
        for r, (b, k) in enumerate(zip(e.readings.bus, e.readings.kind)):
            if s.sensor[b] != pseudo and k in share:
                share[k].append(float(np.sqrt(max(ratio[r], 0.0))))
    rng = np.random.default_rng(0)
    gross = []
    for s, e in zip(te_s, ests):
        bad = copy.deepcopy(s)
        bus = int(rng.choice(np.flatnonzero(s.sensor != pseudo)))
        kind = int(rng.choice([wls_np.VM, wls_np.P, wls_np.Q]))
        bad.obs[bus, kind] += rng.choice([-1, 1]) * 10 * np.sqrt(s.var[bus, kind])
        r = wls_np.estimate_clean(fd, bad)
        kept = wls_np.estimate(fd, bad)
        gross.append({
            "bus": bus, "kind": kind, "flagged": bool(r["chi_square_before"]["bad"]),
            "found": bool(r["removed"]) and (r["removed"][0]["bus"], r["removed"][0]["kind"]) == (bus, kind),
            "removed": len(r["removed"]),
            "vm_mae_clean": float(np.mean(np.abs(np.abs(e.v) - np.abs(s.v)))),
            "vm_mae_with_error": float(np.mean(np.abs(np.abs(kept.v) - np.abs(s.v)))),
            "vm_mae_after_removal": float(np.mean(np.abs(np.abs(r["estimate"].v) - np.abs(s.v)))),
        })
    result = {
        "model": "wls-np", "seconds_per_estimate": secs / len(te_s),
        "iterations": {"min": min(e.iterations for e in ests), "max": max(e.iterations for e in ests)},
        "all_converged": all(e.converged for e in ests),
        "gap_to_gridcal": {"all": gaps(np.arange(len(te_s))), "where_gridcal_reached_the_minimum": gaps(reached)},
        "gridcal_stopped_short": {
            "samples": [int(i) for i in stalled],
            "objective_ratio": [float(j_gc[i] / j_np[i]) for i in stalled],
            "lm_steps": steps,
        },
        "per_sample": metrics.per_sample(vm, va, vm_t, va_t),
        "per_sample_gridcal": metrics.per_sample(gc["vm"], gc["va"], vm_t, va_t),
        "per_sample_with_pmu_angles": metrics.per_sample(vm_a, va_a, vm_t, va_t),
        "per_bus_vm_mae": np.mean(np.abs(vm - vm_t), axis=0).tolist(),
        "thesis_way": metrics.thesis_way(vm, va, vm_t, va_t),
        "bad_data": {
            "clean_samples": len(te_s),
            "chi_square_flags": sum(bool(c["chi_square_before"]["bad"]) for c in clean),
            "rn_removes_any": sum(bool(c["removed"]) for c in clean),
            "rn_limit": wls_np.RN_LIMIT, "chi_square_confidence": wls_np.CHI2_CONFIDENCE,
            "gross_error_sd": 10,
            "gross": gross,
            "error_share_in_residual": {
                name: {"median": float(np.median(share[k])), "p10": float(np.percentile(share[k], 10)),
                       "p90": float(np.percentile(share[k], 90)), "readings": len(share[k])}
                for name, k in (("vm", wls_np.VM), ("p", wls_np.P), ("q", wls_np.Q))
            },
        },
    }
    out_dir = RUNS / "wls-np"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pred.npz", vm=vm, va=va)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


METER_COUNTS = [0, 1, 3, 5, 10, 15, 20, 33]


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
    configurations', but its models only ever saw 10 meters. Here every trained model and WLS
    (GridCal's, and the browser's in numpy) estimate test samples with 0 to 33 meters: 2 PMUs among
    them (fewer when there are fewer meters), more or fewer other meters. Each meter count is its own
    set of samples, from its own seed. With no meter the numpy WLS has no estimate (nothing reads a
    voltage) and records how many samples it declined; GridCal's damped solver answers regardless."""
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
        gc = paths.DATA / f"v2_configs_k{k}_{num_samples}_gridcal.npy"  # the slow part, kept between runs
        if gc.exists():
            v = np.load(gc)
        else:
            v = np.stack([est(s) for s in sm])
            np.save(gc, v)
        row["wls"] = metrics.per_sample(np.abs(v), np.angle(v), vm_t, va_t)
        ests = [wls_np.estimate(fd, s) for s in sm]
        seen = [i for i, e in enumerate(ests) if e.observable]
        row["wls-np"] = {"unobservable": len(sm) - len(seen), "samples": len(sm)}
        if seen:
            v = np.stack([ests[i].v for i in seen])
            err = np.abs(v) - vm_t[seen]
            offset = err.mean(1, keepdims=True)  # one number per sample: the level of the whole estimate
            row["wls-np"] |= metrics.per_sample(np.abs(v), np.angle(v), vm_t[seen], va_t[seen]) | {
                "unobservable": len(sm) - len(seen), "vm_mae_offset_removed": float(np.mean(np.abs(err - offset)))}
            if k == 1:  # does the one |V| reading's error set that level?
                reading = [sm[i].obs[np.flatnonzero(sm[i].mask[:, 0])[0], 0] - vm_t[i, np.flatnonzero(sm[i].mask[:, 0])[0]] for i in seen]
                row["wls-np"]["offset_reading_correlation"] = float(np.corrcoef(offset[:, 0], reading)[0, 1])
        for kind, recipe, model, norm, name in models:
            vm_p, va_p, _ = train.predict(model, norm, data, fd, recipe)
            row[name] = metrics.per_sample(vm_p, va_p, vm_t, va_t)
        out[str(k)] = row
        print(k, {n: round(r["vm_mae"], 4) for n, r in row.items() if "vm_mae" in r}, flush=True)
    (RUNS / "configs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["gnn", "pinn", "wls", "wls-np", "observability", "configs", "select-alpha", "anymeter"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recipe", choices=["thesis", "scaled"], default="thesis")
    p.add_argument("--alpha", type=float, default=train.ALPHA)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.what == "configs":
        run_configs()
    elif a.what == "observability":
        res = observability.fewest(G.feeder())
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "observability.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "by_meters"}, indent=2))
    elif a.what == "select-alpha":
        print(select_alpha(a.alpha))
    elif a.what in ("anymeter", "wls-np"):
        res = run_anymeter(a.seed) if a.what == "anymeter" else run_wls_np()
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_bus")}, indent=2))
    else:
        res = run_wls() if a.what == "wls" else run_model(a.what, a.seed, a.recipe, a.alpha)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_bus")}, indent=2))

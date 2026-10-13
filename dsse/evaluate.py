"""Write the reports from the runs in data/: python -m dsse.evaluate

reports/phase1-v1.md   v1 re-run from its own code, next to what the thesis published
reports/phase2-v2.md   v2 (the thesis with its bugs fixed): WLS, GNN and PINN on the same test set
reports/results.json   every number in both, for anything that shows them (the website)
"""

import contextlib
import sys
import io
import json

import numpy as np

from . import grid as G
from . import paths, published, v1

V1, V2 = paths.DATA / "v1", paths.DATA / "v2"


def _runs(folder, prefix):
    out = []
    for d in sorted(folder.glob(f"{prefix}_seed*")):
        f = d / "result.json"
        if f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def _get(run, path):
    for k in path.split("."):
        run = run[k]
    return run


def base_cases() -> dict:
    out = {}
    for name, g in (("textbook (v2)", G.load_gridcal()), ("v1, lines divided twice", v1.load_grid())):
        fd = G.feeder(g)
        v, s = G.gridcal_power_flow(g, fd.p_load, fd.q_load)
        out[name] = {"vmin": float(np.abs(v).min()), "vmin_bus": int(np.abs(v).argmin()),
                     "losses_kw": float(s.real.sum() * 1000), "r_line_1_2": float(g.lines[0].R)}
    return out


def collect() -> dict:
    pub = published.thesis_numbers()
    with contextlib.redirect_stdout(io.StringIO()):
        bases = base_cases()
    res = {"published": pub, "base_case": bases, "v1": {}, "v2": {}}

    for kind in ("gnn", "pinn"):
        runs = _runs(V1, kind)
        if not runs:
            continue
        nb = "thesis_way_notebook"
        res["v1"][kind] = {
            "seeds": [r["seed"] for r in runs],
            "thesis_way": {k: _stats([r[nb][k] for r in runs]) for k in ("mae_mag", "rmse_mag", "mae_ang", "circ_rmse")},
            "per_sample": {k: _stats([r["per_sample"][k] for r in runs]) for k in ("vm_mae", "vm_rmse", "vm_max", "va_mae")},
            "average_profile": runs[0]["average_profile"],
            "leak_test": {name: _stats([r["leak_test"][name]["vm_mae"] for r in runs]) for name in runs[0]["leak_test"]},
            "per_bus_bias_seed0": runs[0][nb]["per_bus_mag_err"],
            "runs": [{"seed": r["seed"], "thesis_way_vm_mae": r[nb]["mae_mag"], "thesis_way_va_mae": r[nb]["mae_ang"],
                      "vm_mae": r["per_sample"]["vm_mae"], "vm_bias": r["per_sample"]["vm_bias"],
                      "vm_mae_variances_shuffled": r["leak_test"]["variances"]["vm_mae"],
                      "vm_mae_pseudo_values_shuffled": r["leak_test"]["pseudo_values"]["vm_mae"]} for r in runs],
        }
    wls_file = V1 / "wls_table7.json"
    if wls_file.exists():
        draws = json.loads(wls_file.read_text(encoding="utf-8"))
        res["v1"]["wls"] = {k: _stats([d[k]["vm_mae"] for d in draws]) for k in ("as_is", "injections_fixed")}

    backfill_training_error()
    for name in V2_MODELS:
        runs = _runs(V2, name)
        if runs:
            res["v2"][name] = {
                "seeds": [r["seed"] for r in runs],
                "alpha": runs[0].get("alpha"),
                "per_sample": {k: _stats([r["per_sample"][k] for r in runs]) for k in runs[0]["per_sample"] if k != "samples"},
                "on_training_samples": {k: _stats([r["on_training_samples"][k] for r in runs]) for k in ("vm_mae", "vm_bias")},
                "thesis_way": {k: _stats([r["thesis_way"][k] for r in runs]) for k in runs[0]["thesis_way"]},
                "per_bus_vm_mae": np.mean([r["per_bus_vm_mae"] for r in runs], axis=0).tolist(),
                "physics_residual_mean": _stats([r["physics_residual_mean"] for r in runs]),
                "train_seconds": _stats([r["train_seconds"] for r in runs]),
            }
    wls_v2 = V2 / "wls" / "result.json"
    if wls_v2.exists():
        res["v2"]["wls"] = json.loads(wls_v2.read_text(encoding="utf-8")) | {"physics_residual_mean": wls_residual()}
    res["v2"]["average_profile"] = average_profile_v2()
    sel = sorted(V2.glob("select_alpha/alpha_*.json"))
    if sel:
        res["v2"]["select_alpha"] = [json.loads(f.read_text(encoding="utf-8")) for f in sel]
    cfg = V2 / "configs.json"
    if cfg.exists():
        res["v2"]["configs"] = json.loads(cfg.read_text(encoding="utf-8"))
    wnp = V2 / "wls-np" / "result.json"
    if wnp.exists():
        res["v2"]["wls-np"] = wls_np_summary(json.loads(wnp.read_text(encoding="utf-8")))
    obs = V2 / "observability.json"
    if obs.exists():
        res["v2"]["observability"] = json.loads(obs.read_text(encoding="utf-8"))
    runs = _runs(V2, "gnn-anymeter")
    if runs:
        from . import phase2

        m = {
            "seeds": [r["seed"] for r in runs],
            "meters": runs[0]["meters"],
            "runs": [{"seed": r["seed"], "validation_vm_mae": r["validation"]["vm_mae"], "test_vm_mae": r["per_sample"]["vm_mae"],
                      "own_test_set_vm_mae": r["own_test_set"]["vm_mae"]} for r in runs],
            "per_sample": {k: _stats([r["per_sample"][k] for r in runs]) for k in runs[0]["per_sample"] if k != "samples"},
            "own_test_set": {k: _stats([r["own_test_set"][k] for r in runs]) for k in ("vm_mae", "va_mae")},
            "on_training_samples": {k: _stats([r["on_training_samples"][k] for r in runs]) for k in ("vm_mae", "vm_bias")},
            "thesis_way": {k: _stats([r["thesis_way"][k] for r in runs]) for k in runs[0]["thesis_way"]},
            "per_bus_vm_mae": np.mean([r["per_bus_vm_mae"] for r in runs], axis=0).tolist(),
            "physics_residual_mean": _stats([r["physics_residual_mean"] for r in runs]),
            "train_seconds": _stats([r["train_seconds"] for r in runs]),
            "samples": {"total": phase2.ANY_SAMPLES, "fit": phase2.ANY_FIT, "validation": phase2.ANY_TRAIN - phase2.ANY_FIT,
                        "test": phase2.ANY_SAMPLES - phase2.ANY_TRAIN},
        }
        if len(runs) == len(phase2.ANY_SEEDS):
            chosen = phase2.chosen_anymeter()
            run = next(r for r in runs if r["seed"] == chosen)
            m["chosen_seed"] = chosen
            m["chosen"] = {"per_sample": run["per_sample"], "own_test_set": run["own_test_set"],
                           "per_bus_vm_mae": run["per_bus_vm_mae"], "thesis_way": run["thesis_way"]}
        res["v2"]["gnn-anymeter"] = m
    return res


def wls_np_summary(r: dict) -> dict:
    """The browser's WLS run, with the gross-error trials counted rather than listed."""
    bd = dict(r["bad_data"])
    trials = bd.pop("gross")
    by_kind = {}
    for name, kind in (("vm", 0), ("p", 2), ("q", 3)):
        t = [g for g in trials if g["kind"] == kind]
        by_kind[name] = {"trials": len(t), "flagged": sum(g["flagged"] for g in t), "found": sum(g["found"] for g in t),
                         "vm_mae_clean": float(np.mean([g["vm_mae_clean"] for g in t])),
                         "vm_mae_with_error": float(np.mean([g["vm_mae_with_error"] for g in t])),
                         "vm_mae_after_removal": float(np.mean([g["vm_mae_after_removal"] for g in t]))}
    return r | {"bad_data": bd | {"gross_by_kind": by_kind}}


# v2's models: data fixed with v1's training ("gnn", "pinn"), then training fixed too ("-scaled")
V2_MODELS = ("gnn", "pinn", "gnn-scaled", "pinn-scaled")


def wls_residual() -> float:
    """The same physics residual as the models', for WLS's saved estimates."""
    import torch
    from torch_geometric.data import Batch

    from . import phase2, train

    fd, _, _, te, _ = phase2.split()
    p = np.load(V2 / "wls" / "pred.npz")
    vm, va = (torch.tensor(p[k].reshape(-1), dtype=torch.float32) for k in ("vm", "va"))
    return float(train.physics_residual(vm, va, Batch.from_data_list(list(te)), fd).mean())


def backfill_training_error():
    """Runs made before result.json recorded the error on the model's own training samples get it
    now, from their saved weights."""
    from . import metrics, phase2, train

    todo = [d for d in sorted(V2.glob("*_seed*"))
            if (d / "result.json").exists() and "on_training_samples" not in json.loads((d / "result.json").read_text(encoding="utf-8"))]
    if not todo:
        return
    fd, samples, tr, *_ = phase2.split()
    vm_t, va_t = phase2.truth(samples[:400])
    for d in todo:
        kind, recipe, model, norm = phase2.load_model(d)
        vm_p, va_p, _ = train.predict(model, norm, tr[:400], fd, recipe)
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        r["recipe"] = recipe
        r["on_training_samples"] = metrics.per_sample(vm_p, va_p, vm_t, va_t)
        (d / "result.json").write_text(json.dumps(r, indent=2), encoding="utf-8")


def average_profile_v2() -> dict:
    """The baseline any estimator must beat: always answer the training set's mean profile."""
    from . import metrics, phase2

    fd, samples, *_ = phase2.split()
    tr, te = samples[: phase2.N_TRAIN], samples[phase2.N_TRAIN :]
    v_tr = np.stack([s.v for s in tr])
    vm_t, va_t = phase2.truth(te)
    vm_p = np.broadcast_to(np.abs(v_tr).mean(0), vm_t.shape)
    va_p = np.broadcast_to(np.angle(v_tr).mean(0), va_t.shape)
    return {
        "per_sample": metrics.per_sample(vm_p, va_p, vm_t, va_t),
        "thesis_way": metrics.thesis_way(vm_p, va_p, vm_t, va_t),
        "per_bus_vm_mae": np.mean(np.abs(vm_p - vm_t), axis=0).tolist(),
    }


# ---------------------------------------------------------------------------------------------


def f4(x):
    return f"{x:.4f}"


def rng(s):
    return f"{s['min']:.4f}–{s['max']:.4f}" if s["n"] > 1 else f4(s["mean"])


def phase1_md(r: dict) -> str:
    p, b = r["published"], r["base_case"]
    tb, v1b = b["textbook (v2)"], b["v1, lines divided twice"]
    L = ["# Phase 1: v1 re-run from its own code", ""]
    L += ["Generated by `python -m dsse.evaluate`; every number comes from `data/` (runs of `dsse.phase1`)",
          "and from the thesis's own notebook and PDF (`dsse.published`). Nothing in the thesis repository",
          "was changed: its modules were imported from where they are, and the model, loss, training and",
          "evaluation ran from the notebook's own cells (`dsse.v1`).", ""]
    L += ["## Does it reproduce?", ""]
    L += ["The thesis's metric (Tables 8 and 9): the first 32 test samples, predictions and truths each",
          "averaged over the 32, then compared bus by bus. v1's random draws were unseeded, so its exact",
          "dataset and weights can't be made again; the data was regenerated by its code with seeds and the",
          "models retrained. A number counts as reproduced when the published value lies within the range",
          "of the retrained runs.", ""]
    L += ["| Model | Metric | Published | Re-run (range over seeds) | Reproduced |", "|---|---|---|---|---|"]
    names = {"mae_mag": ("vm_mae", "|V| MAE (p.u.)"), "rmse_mag": ("vm_rmse", "|V| RMSE (p.u.)"),
             "mae_ang": ("va_mae", "angle MAE (rad)")}
    for kind, label in (("gnn", "GNN"), ("pinn", "PINN")):
        if kind not in r["v1"]:
            L.append(f"| {label} | – | – | not run yet | – |")
            continue
        for key, (pkey, mlabel) in names.items():
            s = r["v1"][kind]["thesis_way"][key]
            pv = p[kind][pkey]
            ok = "yes" if s["n"] > 1 and s["min"] <= pv <= s["max"] else ("one run" if s["n"] == 1 else "no")
            L.append(f"| {label} ({s['n']} run{'s' if s['n'] > 1 else ''}) | {mlabel} | {f4(pv)} | {rng(s)} | {ok} |")
    if "wls" in r["v1"]:
        s = r["v1"]["wls"]["as_is"]
        ok = "yes" if s["min"] <= p["wls"]["vm_mae"] <= s["max"] else "no"
        L.append(f"| WLS, one sample ({s['n']} draws) | |V| MAE (p.u.) | {f4(p['wls']['vm_mae'])} | {rng(s)} | {ok} |")
    L += [""]
    q = r["v1"].get("pinn")
    if q and q["thesis_way"]["mae_mag"]["n"] > 1 and q["thesis_way"]["mae_mag"]["min"] > p["pinn"]["vm_mae"]:
        b = [x["vm_bias"] for x in q["runs"]]
        L += [f"The PINN doesn't reproduce: all {len(b)} re-runs land above Table 9, each with a constant",
              f"offset of {min(b):+.4f} to {max(b):+.4f} p.u. from its physics term (below). Table 9 can't be",
              "re-made from the thesis's code, so only the thesis's conclusion about the PINN stands: that its",
              "practical performance was limited.", ""]

    L += ["## What the re-run shows", ""]
    if "gnn" in r["v1"]:
        g = r["v1"]["gnn"]
        ap = g["average_profile"]
        L += ["Three things at once, run by run: the thesis's metric, the error on every bus of every test",
              "sample, the constant offset (mean error), and the error once the leaked features are shuffled",
              "between test samples (same bus, another sample: a model that doesn't read the truth from them",
              "is unaffected). The first row never looks at a measurement: it always answers the training",
              "set's average voltage profile.", "",
              "| On v1's test set | Thesis's metric, |V| MAE | Every bus of every sample, |V| MAE | Offset | Variances shuffled |",
              "|---|---|---|---|---|",
              f"| Average profile, no model | {f4(ap['thesis_way']['vm_mae'])} | {f4(ap['per_sample']['vm_mae'])} | "
              f"{ap['per_sample']['vm_bias']:+.4f} | – |"]
        for kind, label in (("gnn", "GNN"), ("pinn", "PINN")):
            for run in r["v1"].get(kind, {}).get("runs", []):
                L.append(f"| v1 {label}, seed {run['seed']} | {f4(run['thesis_way_vm_mae'])} | {f4(run['vm_mae'])} | "
                         f"{run['vm_bias']:+.4f} | {f4(run['vm_mae_variances_shuffled'])} |")
        gr = g["runs"]
        tw = [x["thesis_way_vm_mae"] for x in gr]
        const_tw, const_ps = ap["thesis_way"]["vm_mae"], ap["per_sample"]["vm_mae"]
        winners = [x for x in gr if x["vm_mae"] < const_ps]
        L += ["",
              "**The thesis's metric can't tell a model from a constant.** Averaging 32 scenarios before",
              "comparing cancels every error that differs from sample to sample, so what is left is a model's",
              "constant offset. "
              + ("The constant scores better than every GNN run on it, and " if const_tw < min(tw) else "")
              + f"the same model scores between {f4(min(tw))} and {f4(max(tw))} ({max(tw) / min(tw):.0f} times as much)",
              "depending on where training stopped. Errors have to be taken per sample.", "",
              "**The inputs contain the answer.** At an unmetered bus v1 gives the model a pseudo-measured",
              "voltage centred on the true voltage, and variances computed from the true values:",
              "Var(V) = (0.3 × true |V|)², Var(P) = (0.3 × true P)². So 23 of the 33 buses carry their true",
              "|V| exactly, in a feature (`gridcal_utils/pmu_smartmeter_pseudo_measurment_generation.py`, lines",
              "133, 200–221). "
              + (f"{len(winners)} of {len(gr)} GNN runs beat the constant per sample (seeds "
                 + ", ".join(str(x["seed"]) for x in winners) + "); with their variances shuffled, "
                 + ("all of them fall behind it." if all(x["vm_mae_variances_shuffled"] > const_ps for x in winners)
                    else "they get worse.")
                 if winners else "No GNN run beats the constant per sample."), ""]
    if "wls" in r["v1"]:
        w = r["v1"]["wls"]
        L += ["**WLS was given wrong injections.** v1 recorded each load as +P where GridCal counts an",
              "injection into the grid as positive, and recorded the substation's injection as 0 MW, the",
              "scheduled output of the slack generator, instead of the 3.8 MW the power flow gives; as a",
              "pseudo-measurement that 0 MW gets a standard deviation of 0.0003 MW, the most trusted reading",
              "in the set. The same draws, with only those two things undone:", "",
              f"| GridCal WLS on v1's measurements of the base case ({w['as_is']['n']} draws) | |V| MAE, mean | range |",
              "|---|---|---|",
              f"| As in the thesis | {f4(w['as_is']['mean'])} | {rng(w['as_is'])} |",
              f"| Signs fixed, substation's 0 MW dropped | {f4(w['injections_fixed']['mean'])} | {rng(w['injections_fixed'])} |", ""]
    L += ["**The feeder isn't the IEEE 33-bus system.** GridCal converts case33bw.m to per-unit when it reads",
          f"it (line 1–2: {tb['r_line_1_2']:.4f} p.u.); notebook cell 22 divides by Z_base again "
          f"({v1b['r_line_1_2']:.4f} p.u.), so every line has 1/1.6 of its impedance:", "",
          "| Base case | Lowest |V| (p.u.) | Losses (kW) |", "|---|---|---|",
          f"| Published IEEE 33-bus (and v2) | {tb['vmin']:.4f} at bus {tb['vmin_bus']} | {tb['losses_kw']:.1f} |",
          f"| v1 | {v1b['vmin']:.4f} at bus {v1b['vmin_bus']} | {v1b['losses_kw']:.1f} |", ""]
    L += ["**The PINN's physics term isn't Ohm's law.** In `pinn_loss` (and eq. 100 of the thesis) the line",
          "current is taken to be the injection current of the line's first bus, conj(S_i / V_i); R and X are",
          "the standardised edge features (mean 0, std 1) rather than the lines' impedances; P and Q are in",
          "MW while V is in p.u.; and the loop runs in Python over every edge of every batch, which is what",
          "made the PINN slow to train. A loss built this way can't be zero at the true state, so it pulls",
          "the model away from it.", ""]
    L += ["**Smaller things.** The target is stored as (Re V, Im V) and read as (|V|, angle), which at",
          "angles under 0.01 rad changes little; the GNN also passes messages over the five open tie",
          "switches; the 'smart meter ~5%' of the notebook is 0.05 MW in the code, on loads of 0.05–0.4 MW;",
          "the thesis text puts the PMU on bus 0 and smart meters on buses 4 and 7, while the code places",
          "10 meters at random in every sample.", ""]
    return "\n".join(L)


LABELS = {
    "average_profile": "Average profile, no measurements",
    "wls": "WLS (GridCal)",
    "wls-np": "WLS (Gauss-Newton in numpy, the browser's)",
    "gnn-anymeter": "GNN trained on 0 to 33 meters (the website's)",
    "gnn": "GNN, thesis's training",
    "pinn": "PINN (Ohm's law fixed), thesis's training",
    "gnn-scaled": "GNN, training fixed",
    "pinn-scaled": "PINN (Ohm's law fixed), training fixed",
}


def _estimator_rows(v2: dict) -> list[tuple[str, dict, dict, int]]:
    """(name, per-sample metrics as plain numbers, thesis-way metrics, runs) for each estimator."""
    out = []
    for name in ("average_profile", "wls", "wls-np"):
        if name in v2:
            out.append((name, v2[name]["per_sample"], v2[name]["thesis_way"], 1))
    for name in V2_MODELS + ("gnn-anymeter",):
        if name in v2:
            m = v2[name]
            out.append((name, {k: s["mean"] for k, s in m["per_sample"].items()},
                        {k: s["mean"] for k, s in m["thesis_way"].items()}, m["per_sample"]["vm_mae"]["n"]))
    return out


def phase2_md(r: dict) -> str:
    v2 = r["v2"]
    L = ["# Phase 2: v2, the thesis with its bugs fixed", ""]
    L += ["Generated by `python -m dsse.evaluate` from the runs of `dsse.phase2`.", "",
          "Kept from v1: the IEEE 33-bus feeder, the scenario generator (each load times 1 + 0.5 N(0,1)),",
          "2,000 samples split 1,600 / 400, 10 metered buses per sample placed at random (2 PMUs), the",
          "meters' noise and bias, the GNN and PINN classes (run from the notebook's cells), the 17 input",
          "features per bus, 100 epochs, Adam at 1e-3, batch 32. Fixed in the data and the physics term:",
          "everything in phase 1 (`dsse/scenarios.py`, `dsse/grid.py`, `dsse/train.py`). The test set is",
          "the last 400 samples and plays no part in any choice.", ""]

    L += ["## Results on the 400 test samples", "",
          "Every bus of every sample; models are the mean over their training runs (seeds). The last",
          "column is the thesis's own metric, for comparison with Tables 7 to 9.", "",
          "| Estimator | Runs | |V| MAE (p.u.) | |V| RMSE | 95th percentile | Largest | Angle MAE (rad) | Thesis's metric |",
          "|---|---|---|---|---|---|---|---|"]
    rows = _estimator_rows(v2)
    for name, ps, tw, n in rows:
        L.append(f"| {LABELS[name]} | {n} | {f4(ps['vm_mae'])} | {f4(ps['vm_rmse'])} | {f4(ps['vm_p95'])} | "
                 f"{f4(ps['vm_max'])} | {f4(ps['va_mae'])} | {f4(tw['vm_mae'])} |")
    L += [""]
    for name in V2_MODELS:
        if name in v2:
            m = v2[name]
            extra = f"; α = {m['alpha']:g}" if m.get("alpha") is not None else ""
            L.append(f"- {LABELS[name]}: |V| MAE over runs {rng(m['per_sample']['vm_mae'])}; mean physics residual "
                     f"of its predictions {m['physics_residual_mean']['mean']:.2e} p.u. per line{extra}.")
    if "wls" in v2:
        L.append(f"- WLS (GridCal): mean physics residual of its estimates {v2['wls']['physics_residual_mean']:.2e} p.u. "
                 f"per line; {v2['wls']['seconds'] / v2['wls']['per_sample']['samples']:.2f} s per estimate.")
    if "wls-np" in v2:
        w = v2["wls-np"]
        L.append(f"- {LABELS['wls-np']}: the same readings and objective as GridCal's; GridCal stops short of its "
                 f"minimum on {len(w['gridcal_stopped_short']['samples'])} of the {w['gap_to_gridcal']['all']['samples']} "
                 f"samples (see 'WLS for the browser'); {w['seconds_per_estimate']:.2f} s per estimate.")
    L += ["", "The physics residual is |V_i − V_j − Z_ij I_ij| on each line, with I_ij from the readings and",
          "pseudo-measurements: it is zero only at the true state with exact readings, so it measures how",
          "far an estimate is from obeying Ohm's law given what the meters say."]
    if rows:
        best = min(rows, key=lambda x: x[1]["vm_mae"])
        L += ["", f"Lowest |V| error: **{LABELS[best[0]]}**.", ""]

    trained = [n for n in V2_MODELS if n in v2]
    if trained:
        L += ["## Why the thesis's training falls short", "",
              "A trained model should have no constant offset on the samples it was trained on: the output",
              "layer's bias alone could remove it. Measured on 400 of each model's own training samples:", "",
              "| Model | |V| MAE on its training samples | Mean error (offset) | |V| MAE on the test set |",
              "|---|---|---|---|"]
        for name in trained:
            m = v2[name]
            L.append(f"| {LABELS[name]} | {f4(m['on_training_samples']['vm_mae']['mean'])} | "
                     f"{m['on_training_samples']['vm_bias']['mean']:+.4f} (runs {m['on_training_samples']['vm_bias']['min']:+.4f} to "
                     f"{m['on_training_samples']['vm_bias']['max']:+.4f}) | {f4(m['per_sample']['vm_mae']['mean'])} |")
        L += ["",
              "With the thesis's training the model outputs |V| near 0.95 and sin, cos of angles within",
              "±0.01 rad in raw units, at a learning rate that stays at 1e-3. Adam's steps are then as large",
              "as the accuracy needed, and training stops wherever the last step left it: errors on training",
              "and test samples are alike (no overfitting), and most of them are a constant offset that",
              "differs from run to run (the thesis's own model had +0.008, Table 8). The fixed training",
              "standardises the outputs, as the thesis already did for the inputs, predicts the angle itself,",
              "and lets the learning rate decay to zero over the same 100 epochs (`dsse/train.py`).", ""]

    if "select_alpha" in v2:
        L += ["## The PINN's weight under the fixed training", "",
              "Rescaling the outputs changes the balance between the data term and the physics term, so α",
              "was chosen again: each candidate fitted on samples 0–1,399 and scored on 1,400–1,599.", "",
              "| α | Validation |V| MAE | Physics residual (p.u.) |", "|---|---|---|"]
        sel = sorted(v2["select_alpha"], key=lambda s: s["alpha"])
        for s in sel:
            label = "0 (the GNN)" if s["alpha"] == 0 else f"{s['alpha']:g}"
            L.append(f"| {label} | {f4(s['vm_mae'])} | {s['physics_residual_mean']:.2e} |")
        pos = [s for s in sel if s["alpha"] > 0]
        err = [s["vm_mae"] for s in pos]
        res_ = [s["physics_residual_mean"] for s in sel]
        trade = all(a <= b for a, b in zip(err, err[1:])) and all(a >= b for a, b in zip(res_, res_[1:]))
        zero = next((s for s in sel if s["alpha"] == 0), None)
        L += [""]
        if zero and pos:
            d = pos[0]["vm_mae"] - zero["vm_mae"]
            L += [f"At α = {pos[0]['alpha']:g}, the weight chosen, the physics term changes the validation error by "
                  f"{d:+.5f} p.u., {abs(d) / zero['vm_mae']:.0%} of it: the PINN is the GNN.", ""]
        if trade:
            L += ["Beyond that, the more weight on Ohm's law, the more closely the estimate obeys it and the worse it gets.",
                  "The residual is computed from the readings, and at 23 of the 33 buses those are forecasts",
                  "that can be 50% off: holding the estimate to them pulls it towards the power flow of the",
                  "forecast. WLS also obeys Ohm's law against the same readings, but weighs each reading by how",
                  "far it can be trusted (1/σ²); the thesis's physics term weighs a forecast like a smart meter.",
                  "A physics term weighted by the readings' variances is the WLS objective itself, which makes",
                  "it the natural next step: the hybrid of chapter 7.", ""]

    if "configs" in v2:
        L += configs_md(v2)
    if "wls-np" in v2:
        L += wls_np_md(v2)
    if "observability" in v2:
        L += observability_md(v2["observability"])
    if "gnn-anymeter" in v2:
        L += anymeter_md(v2)
    return "\n".join(L)


def config_table(v2: dict) -> tuple[list[str], dict]:
    """|V| MAE by number of meters for each estimator (mean over a model's seeds; NaN where an
    estimator gave no estimate). The website's GNN also gets a row for the seed the browser runs."""
    cfg = v2["configs"]
    ks = list(cfg)
    first = cfg[ks[0]]
    names = [n for n in ("average_profile", "wls", "wls-np") if n in first] + [
        n for n in V2_MODELS + ("gnn-anymeter",) if any(k.startswith(n + "_seed") for k in first)]
    keys = {n: [n] if n in first else [k for k in first if k.startswith(n + "_seed")] for n in names}
    chosen = v2.get("gnn-anymeter", {}).get("chosen_seed")
    if chosen is not None:
        names.append("gnn-anymeter-chosen")
        keys["gnn-anymeter-chosen"] = [f"gnn-anymeter_seed{chosen}"]
    table = {}
    for name in names:
        table[name] = [float(np.mean([cfg[k][key]["vm_mae"] for key in keys[name]])) if all("vm_mae" in cfg[k][key] for key in keys[name])
                       else float("nan") for k in ks]
    return ks, table


def _f4(x):
    return "–" if np.isnan(x) else f4(x)


def configs_md(v2: dict) -> list[str]:
    ks, table = config_table(v2)
    labels = LABELS | {"gnn-anymeter-chosen": f"GNN trained on 0 to 33 meters, seed {v2.get('gnn-anymeter', {}).get('chosen_seed')} (in the browser)"}
    L = ["## Fewer or more meters", "",
         "The thesis concludes (chapter 7) that the learned estimator keeps 'satisfactory performance",
         "... with respect to variations in ... measurement configurations', but its models only ever",
         f"saw 10 meters. Here every model, and WLS, estimate new samples with {ks[0]} to {ks[-1]} meters (2 of",
         "them PMUs, fewer when there are fewer meters), 200 samples per count. Every model but the",
         "website's GNN was trained on 10 meters only.", "",
         "| |V| MAE (p.u.) by meters | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
    for name, vals in table.items():
        L.append(f"| {labels[name]} | " + " | ".join(_f4(v) for v in vals) + " |")
    L += [""]
    cand = [n for n in table if n != "gnn-anymeter-chosen"]
    best = [min((n for n in cand if not np.isnan(table[n][i])), key=lambda n: table[n][i]) for i in range(len(ks))]
    wls_names = {"wls", "wls-np"}
    if all(b in wls_names for b in best):
        L.append("WLS has the lowest error at every number of meters.")
    else:
        L.append("Lowest error by number of meters: " + ", ".join(f"{k}: {labels[b]}" for k, b in zip(ks, best)) + ".")
    if "wls-np" in table and np.isnan(table["wls-np"][0]):
        L.append(f"With {ks[0]} meters nothing reads a voltage and the browser's WLS gives no estimate (the gain "
                 "matrix is singular: the voltage level is not determined); GridCal's damped solver answers anyway, "
                 "near its flat start of 1 p.u.")
    ref = table.get("wls-np", table["wls"])
    for name in ("gnn-scaled", "pinn-scaled", "gnn-anymeter"):
        if name in table:
            v = table[name]
            steady = all(a >= b for a, b in zip(v, v[1:]))
            ratios = [a / w for a, w in zip(v, ref) if not np.isnan(w)]
            trained = "trained on 10 meters only" if name != "gnn-anymeter" else "trained on 0 to 33 meters, mean of its seeds"
            L.append(f"{labels[name]}, {trained}: "
                     + ("its error falls steadily as meters are added, from " if steady else "its error goes from ")
                     + f"{f4(v[0])} at {ks[0]} to {f4(v[-1])} at {ks[-1]}; "
                     f"{min(ratios):.2f} to {max(ratios):.2f} times the browser's WLS where that has an estimate.")
    L += ["",
          "With pseudo-measurements at every unmetered bus, one real meter is enough for WLS: the forecasts",
          "fix every injection and the meter fixes the voltage level. Without them, it takes many more",
          "(below).", ""]
    return L


def wls_np_md(v2: dict) -> list[str]:
    w = v2["wls-np"]
    gap, short = w["gap_to_gridcal"], w["gridcal_stopped_short"]
    n_all, n_ok = gap["all"]["samples"], gap["where_gridcal_reached_the_minimum"]["samples"]
    ok = gap["where_gridcal_reached_the_minimum"]
    L = ["## WLS for the browser", "",
         "The page runs WLS in the browser, where GridCal can't go, so `dsse/wls_np.py` solves the same",
         "problem in plain numpy, written to be ported line for line: the angles of buses 1-32 and |V| of",
         "all 33 as the state, |V|, P and Q at buses as the readings, weighted by 1/variance, Gauss-Newton",
         f"from a flat start with a dense Cholesky solve. It converges in {w['iterations']['min']} to "
         f"{w['iterations']['max']} steps, {w['seconds_per_estimate'] * 1000:.0f} ms per estimate here.", "",
         f"On the {n_all} test samples, next to GridCal's estimates of the same readings:", "",
         "| Samples | Largest |V| gap (p.u.) | Mean |V| gap | Largest angle gap (rad) | Mean angle gap |",
         "|---|---|---|---|---|",
         f"| The {n_ok} where GridCal reaches the minimum | {ok['vm_max']:.1e} | {ok['vm_mean']:.1e} | {ok['va_max']:.1e} | {ok['va_mean']:.1e} |",
         f"| All {n_all} | {gap['all']['vm_max']:.1e} | {gap['all']['vm_mean']:.1e} | {gap['all']['va_max']:.1e} | {gap['all']['va_mean']:.1e} |",
         ""]
    if short["samples"]:
        ratios = short["objective_ratio"]
        steps = short["lm_steps"]
        pattern = all(s.rstrip("-").count("-") == 0 and s.endswith("-") for s in steps)
        L += [f"On the other {len(ratios)} GridCal stops short: its weighted squared error there is "
              f"{min(ratios):.0f} to {max(ratios):.0f} times the minimum Gauss-Newton reaches. "
              + ("In every one of them GridCal's Levenberg-Marquardt takes steps until one is refused, then "
                 "refuses every step after it" if pattern else "Its Levenberg-Marquardt refuses steps there")
              + f" ({min(len(s) for s in steps)}-{max(len(s) for s in steps)} iterations, e.g. "
              f"`{steps[0]}`, + taken, - refused): GridCal 5.3.53 judges each step by the objective at the "
              "state before it, so after one refusal the state stops changing, every later ratio is 0, and the "
              "damping grows until the step is under the tolerance, which it reports as converged "
              "(`dsse.wls.lm_steps`).", "",
              f"So WLS is a little better than the v2 table above says: at the minimum its |V| MAE on the test "
              f"set is {f4(w['per_sample']['vm_mae'])}, not {f4(w['per_sample_gridcal']['vm_mae'])}.", ""]
    gain = 1 - w["per_sample_with_pmu_angles"]["vm_mae"] / w["per_sample"]["vm_mae"]
    L += [f"The PMUs' angles, which GridCal can't take, are an option (off by default, to match GridCal); with "
          f"them the |V| MAE is {w['per_sample_with_pmu_angles']['vm_mae']:.5f} against {w['per_sample']['vm_mae']:.5f}"
          + (", barely lower: an angle read to ±0.01 rad (its bias) says little on a feeder whose angles stay "
             "within ±0.01 rad." if gain < 0.05 else f", {gain:.0%} lower."), ""]
    L += bad_data_md(w["bad_data"])
    return L


def bad_data_md(b: dict) -> list[str]:
    g, share = b["gross_by_kind"], b["error_share_in_residual"]
    n = b["clean_samples"]
    names = {"vm": "|V|", "p": "P", "q": "Q"}
    L = ["## Bad data", "",
         f"Two tests from the same matrices. The chi-square test: at the estimate J(x) should follow a "
         f"chi-square distribution with (readings − 65) degrees of freedom, so above its "
         f"{b['chi_square_confidence']:.0%} point something is wrong. The largest normalised residual test: "
         f"r_N = |r_i| / sqrt(Ω_ii), with Ω = R − H G⁻¹ Hᵀ; while the largest is above {b['rn_limit']:g}, remove "
         f"that reading and estimate again.", "",
         f"On the {n} clean test samples the chi-square test fires on {b['chi_square_flags']} and the residual "
         f"test removes a reading from {b['rn_removes_any']}. Then one gross error, {b['gross_error_sd']} standard "
         "deviations on one real meter's |V|, P or Q (chosen at random), in each sample:", "",
         "| Reading with the error | Trials | Chi-square fires | Residual test removes it | |V| MAE clean | with the error | after the test |",
         "|---|---|---|---|---|---|---|"]
    for k in ("vm", "p", "q"):
        x = g[k]
        L.append(f"| {names[k]} | {x['trials']} | {x['flagged']} | {x['found']} | {f4(x['vm_mae_clean'])} | "
                 f"{f4(x['vm_mae_with_error'])} | {f4(x['vm_mae_after_removal'])} |")
    L += ["",
          f"A meter's |V| is checked by everything around it: a median {share['vm']['median']:.0%} of its error shows "
          f"in its own residual. Its P and Q are not: {share['p']['median']:.1%} and {share['q']['median']:.1%} "
          f"(medians), because the only readings that could contradict them are the neighbours' forecasts, which "
          f"are 50% uncertain. An error of e standard deviations gives r_N of about e times that share, so a P "
          f"error would need some {3 / share['p']['median']:.0f} standard deviations to cross {b['rn_limit']:g}. "
          "Without more real meters around it, a wrong P or Q is absorbed, not caught"
          + (f"; at 10 standard deviations it also barely moves the |V| estimate ({f4(g['p']['vm_mae_clean'])} to "
             f"{f4(g['p']['vm_mae_with_error'])} for P), since the readings that absorb it are the uncertain ones."
             if g["p"]["vm_mae_with_error"] < 1.1 * g["p"]["vm_mae_clean"] else "."), ""]
    return L


def observability_md(o: dict) -> list[str]:
    lb = o["lower_bound_by_counting"]
    L = ["## Observability from real meters alone", "",
         "Pseudo-measurements make every placement observable. Without them, the readings themselves must",
         f"determine all {o['states']} unknowns (32 angles and 33 magnitudes), and a meter reads {o['readings_per_meter']} "
         f"numbers, so no fewer than {lb} meters can do it. The test is the rank of the gain matrix at a flat "
         "start (`wls_np.observable`, by QR of the Jacobian). One more rule is exact: two neighbouring buses can't "
         "both be unmetered, because every bus beyond the line between them could turn by the same small angle "
         "with the two unknown injections absorbing the change in that line's flow, and no reading would move.", "",
         "Every placement that obeys that rule was tested (`dsse/observability.py`, in an equivalent smaller "
         "system: a metered bus's P and Q fix its own current, so only |V0| and the unmetered buses' currents "
         "are left, seen through the |V| readings):", "",
         "| Meters | Placements with no two neighbouring gaps | Observable |", "|---|---|---|"]
    for k, row in o["by_meters"].items():
        L.append(f"| {k} | {row['placements_without_neighbouring_gaps']:,} | {row['observable']:,} |")
    L += [""]
    if "fewest" in o:
        k = str(o["fewest"])
        row = o["by_meters"][k]
        pat, sd = row["pattern"], row["vm_sd_smart_meters_base_case"]
        joined = lambda xs: ", ".join(map(str, xs))  # noqa: E731
        L += [f"**The fewest is {k} meters**" + (", as few as counting allows" if int(k) == lb else "")
              + f", and only {row['observable']:,} placements of {k} work, each leaving {o['states'] // 2 + 1 - int(k)} "
              f"buses without a meter, spread along the feeder. Buses {joined(pat['always_metered'])} always have one; "
              "of each pair " + ", ".join(f"{a}-{b}" for a, b in pat["one_of_each_pair_unmetered"])
              + f" exactly one has none; and the other {' or '.join(map(str, pat['gaps_in_rest']))} gaps sit among buses "
              f"{joined(pat['rest'])}. The browser's test agrees with this one on "
              f"{row['browser_test_agrees']['agree']} of {row['browser_test_agrees']['checked']} placements checked.", "",
              f"Observable is not accurate. With smart meters and the base-case loads, the estimate's |V| at the "
              f"worst bus has a standard deviation of {sd['best_largest']:.3f} p.u. at the best of these placements "
              f"and {sd['typical_largest']:.2f} at a typical one; with a smart meter at all 33 buses, "
              f"{sd['all_33_largest']:.4f}. At a "
              "flat start these placements are barely full rank (the smallest singular value of the reduced system is "
              f"under {max(b['to'] for b in row['bands'] if b['placements'] and b['from'] >= o['reduced_tol']):g} of the "
              "largest): with no current flowing, a current's in-phase and quadrature parts can only be told apart "
              "through lines of different R/X, and IEEE 33's lines come in a few conductor types.", ""]
    return L


def anymeter_md(v2: dict) -> list[str]:
    m = v2["gnn-anymeter"]
    s = m["samples"]
    L = ["## A GNN for any number of meters (made for the website)", "",
         "The page lets a visitor place from none to all 33 meters, and every model above saw exactly 10. So",
         "the website's GNN is the same class with the fixed training (gnn-scaled), on its own data: "
         f"{s['total']:,} samples, each with 0 to 33 meters drawn uniformly and 0 to 2 PMUs among them, the last "
         f"{s['test']} its test set. Each of {len(m['seeds'])} seeds was fitted on the first {s['fit']:,} samples, and the "
         f"browser gets the one with the lowest |V| error on the next {s['validation']} (the last of the training "
         "samples); neither test set played a part in that choice.", "",
         "| Seed | Validation |V| MAE | v2's test set (10 meters), |V| MAE | Its own test set (0 to 33 meters), |V| MAE |",
         "|---|---|---|---|"]
    chosen = m.get("chosen_seed")
    for r in m["runs"]:
        mark = " (chosen)" if r["seed"] == chosen else ""
        L.append(f"| {r['seed']}{mark} | {f4(r['validation_vm_mae'])} | {f4(r['test_vm_mae'])} | {f4(r['own_test_set_vm_mae'])} |")
    L += [""]
    if chosen is not None and "gnn-scaled" in v2:
        mine = m["chosen"]["per_sample"]["vm_mae"]
        ref = v2["gnn-scaled"]["per_sample"]["vm_mae"]["mean"]
        wls_ = v2["wls-np"]["per_sample"]["vm_mae"] if "wls-np" in v2 else v2["wls"]["per_sample"]["vm_mae"]
        rel = mine / ref - 1
        L += [f"On v2's test set, where every sample has 10 meters, the chosen seed's |V| MAE is {f4(mine)}: "
              f"{abs(rel):.0%} {'above' if rel > 0 else 'below'} the GNN trained on 10 meters only ({f4(ref)}), and "
              f"{mine / wls_:.2f} times the browser's WLS ({f4(wls_)})."]
        if "configs" in v2:
            ks, table = config_table(v2)
            if "gnn-anymeter-chosen" in table and "wls-np" in table:
                g, w = table["gnn-anymeter-chosen"], table["wls-np"]
                beats = [k for k, a, b in zip(ks, g, w) if not np.isnan(b) and a < b]
                L.append("By number of meters (table above), " + (
                    "it is behind WLS at every count where WLS has an estimate"
                    if not beats else "it is ahead of WLS at " + ", ".join(beats) + " meters")
                    + (f"; with {ks[0]} meters, where WLS has none, its error is {f4(g[0])} against the average "
                       f"profile's {f4(table['average_profile'][0])}." if np.isnan(w[0]) else "."))
        L += [""]
    return L


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the reports use σ, ×, ², which Windows' default can't print
    r = collect()
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS / "results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (paths.REPORTS / "phase1-v1.md").write_text(phase1_md(r), encoding="utf-8")
    (paths.REPORTS / "phase2-v2.md").write_text(phase2_md(r), encoding="utf-8")
    print((paths.REPORTS / "phase1-v1.md").read_text(encoding="utf-8"))
    print((paths.REPORTS / "phase2-v2.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

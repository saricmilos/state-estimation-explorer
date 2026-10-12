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
    return res


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
    "gnn": "GNN, thesis's training",
    "pinn": "PINN (Ohm's law fixed), thesis's training",
    "gnn-scaled": "GNN, training fixed",
    "pinn-scaled": "PINN (Ohm's law fixed), training fixed",
}


def _estimator_rows(v2: dict) -> list[tuple[str, dict, dict, int]]:
    """(name, per-sample metrics as plain numbers, thesis-way metrics, runs) for each estimator."""
    out = []
    for name in ("average_profile", "wls"):
        if name in v2:
            out.append((name, v2[name]["per_sample"], v2[name]["thesis_way"], 1))
    for name in V2_MODELS:
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
        cfg = v2["configs"]
        ks = list(cfg)
        L += ["## Fewer or more meters", "",
              "The thesis concludes (chapter 7) that the learned estimator keeps 'satisfactory performance",
              "... with respect to variations in ... measurement configurations', but its models only ever",
              "saw 10 meters. Here every model trained on 10 meters, and WLS, estimate new samples with 3 to",
              "20 meters (the same 2 PMUs, more or fewer other meters), 200 samples per count:", "",
              "| |V| MAE (p.u.) by meters | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
        names = ["average_profile", "wls"] + [n for n in V2_MODELS if any(k.startswith(n + "_seed") for k in cfg[ks[0]])]
        table = {}
        for name in names:
            vals = []
            for k in ks:
                runs = [v["vm_mae"] for key, v in cfg[k].items() if key == name or key.startswith(name + "_seed")]
                vals.append(float(np.mean(runs)) if runs else float("nan"))
            table[name] = vals
            L.append(f"| {LABELS[name]} | " + " | ".join(f4(v) for v in vals) + " |")
        L += [""]
        best = [min(names, key=lambda n: table[n][i]) for i in range(len(ks))]
        if all(b == "wls" for b in best):
            L.append("WLS has the lowest error at every number of meters.")
        else:
            L.append("Lowest error by number of meters: " + ", ".join(f"{k}: {LABELS[b]}" for k, b in zip(ks, best)) + ".")
        for name in ("gnn-scaled", "pinn-scaled"):
            if name in table:
                v = table[name]
                steady = all(a >= b for a, b in zip(v, v[1:]))
                ratios = [a / w for a, w in zip(v, table["wls"])]
                L.append(f"{LABELS[name]}, trained on 10 meters only: "
                         + ("its error falls steadily as meters are added, from " if steady else "its error goes from ")
                         + f"{f4(v[0])} at {ks[0]} to {f4(v[-1])} at {ks[-1]}; "
                         f"{min(ratios):.2f} to {max(ratios):.2f} times WLS's.")
        L += ["",
              "WLS never fails for lack of meters here: every unmetered bus has a pseudo-measurement, so the",
              "state is always determined. Without pseudo-measurements it would need enough real meters to be",
              "observable; that comparison is not part of the thesis and is not made here.", ""]
    return "\n".join(L)


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

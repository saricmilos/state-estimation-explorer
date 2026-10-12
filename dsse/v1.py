"""v1: the thesis as handed in, run from its own code and never changed.

Its modules (gridcal_utils/) are imported from the thesis repository. The model, loss, training and
evaluation code exists only in the notebook, so those cells are taken from the notebook file and
executed as they are, in the notebook's order. Nothing here edits v1's code; the two things done
around it are said where they happen: the random draws are given seeds, and stdout is quietened.
"""

import contextlib
import importlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from . import paths


@contextlib.contextmanager
def _on_path():
    sys.path.insert(0, str(paths.THESIS))
    try:
        yield
    finally:
        sys.path.remove(str(paths.THESIS))


def module(name: str):
    """Import gridcal_utils.<name> from the thesis repository, as the notebook imports it."""
    with _on_path():
        return importlib.import_module(f"gridcal_utils.{name}")


@contextlib.contextmanager
def quiet():
    """v1 prints a line for every power flow and every generated sample; keep the output readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------------------------
# The notebook's cells


def _code_cells() -> list[str]:
    nb = json.loads(paths.NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell(prefix: str) -> str:
    """The source of the one code cell that starts with `prefix`."""
    found = [s for s in _code_cells() if s.startswith(prefix)]
    if len(found) != 1:
        raise LookupError(f"{len(found)} notebook cells start with {prefix!r}")
    return found[0]


def run(prefix: str, ns: dict) -> None:
    exec(compile(cell(prefix), f"<v1 notebook cell: {prefix[:40]!r}>", "exec"), ns)


# The cells v1 trains and scores with, by their first characters, in the notebook's order.
SPLIT = "train_dataset = data_list1[:int(len(data_list1)*0.8)]"
STATS_DEF = "def compute_stats(dataset):"
STATS = "x_mean, x_std, e_mean, e_std = compute_stats(train_dataset)"
NORMALIZE = "# Helper to normalize a batch"
GNN_CLASS = "class GNNStateEstimator2(torch.nn.Module):"
DIMS = "in_ch = data_list1[0].x.shape[1]"
GNN_INIT = "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel = GNNStateEsti"
GNN_TRAIN = "num_epochs = 100\nfor epoch in range(1, num_epochs+1):"
GNN_EVAL = "model.eval()\nwith torch.no_grad():\n    batch = next(iter(test_loader))"
GNN_ANGLE = "pred_mag = pred_avg[:,0]"
ANGLE_DIFF = "def angle_diff(pred, true):\n    # returns difference in (-pi, pi]"
GNN_METRICS = "ang_err = angle_diff(pred_angle, true_state[:,1])"
PINN_CLASS = "# --------------------\n# 1) Physics-Informed GNN model"
PINN_LOSS = "# 2) Physics-informed loss"
PINN_TRAIN = "# --------------------\n# 3)Training loop"
PINN_EVAL = "# --------------------\n# 1) Evaluate model and recover voltages"
PINN_METRICS = "# --------------------\n# 2) Compute errors"
# Step 4 of the notebook: one sample of measurements and GridCal's WLS on it (Table 7)
SE_PF = "v_mag, v_ang, v_complex = run_power_flow_get_voltage(grid)"
SE_MEAS = "meas_res = make_measurements_with_pseudo("
SE_LISTS = "vm_list = [(meas_res['vm_obs'][i]"
SE_ADD = "add_measurements(grid, gce, vm_list, pi_list, qi_list)"
SE_RUN = "initial_se_results = run_state_estimation(grid)"


def namespace() -> dict:
    """The notebook's globals after its first cell: the same imports, without the autoreload magics."""
    import matplotlib

    matplotlib.use("Agg")
    src = "\n".join(line for line in cell("%load_ext autoreload").splitlines() if not line.startswith("%"))
    ns: dict = {"__name__": "v1_notebook"}
    with _on_path():
        exec(compile(src, "<v1 notebook cell: imports>", "exec"), ns)
    return ns


# ---------------------------------------------------------------------------------------------
# The grid and the data, as the notebook makes them


def load_grid():
    """case33bw as the notebook loads it (cell 6), including its conversion of the lines (cell 22)."""
    grid = module("io").load_network_renamed("case33bw.m", sbase=100, folder=str(paths.NETWORKS))
    with quiet():
        module("powerflow").convert_lines_ohm_to_pu(grid, vbase_kv=12.66)
    return grid


@contextlib.contextmanager
def seeded_measurements(seed: int, owner=None):
    """Give each call of make_measurements_with_pseudo its own seed.

    v1 calls it with seed=None, so every sample's meter placement and noise come from fresh OS
    entropy and its dataset can't be made again. Same distribution, but each draw now gets the next
    seed of one SeedSequence, so the data can be. `owner` is the namespace whose binding is replaced
    (the module by default; the notebook's globals when the notebook's own cells call it).
    """
    gen = module("pmu_smartmeter_pseudo_measurment_generation")
    target = gen.__dict__ if owner is None else owner
    original = target["make_measurements_with_pseudo"]
    counter = iter(np.random.SeedSequence(seed).generate_state(1_000_000))

    def make_measurements_with_pseudo(*args, **kwargs):
        kwargs["seed"] = int(next(counter))
        return original(*args, **kwargs)

    target["make_measurements_with_pseudo"] = make_measurements_with_pseudo
    try:
        yield
    finally:
        target["make_measurements_with_pseudo"] = original


def dataset(grid, num_samples: int = 2000, seed: int = 42):
    """The notebook's 2,000 samples (cell 69), with its settings, made once and cached."""
    gen = module("pmu_smartmeter_pseudo_measurment_generation")
    cache = paths.DATA / f"v1_train_33bw_{num_samples}_seed{seed}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with seeded_measurements(seed + 1), quiet():
        data_list, _ = gen.get_gridcal_data_cached(
            grid=grid,
            num_samples=num_samples,
            cache_path=str(cache),
            seed=seed,
            rand_ratio=0.5,
            batch_size=32,
            frac_measured=0.3,
            frac_pmu=0.05,
            pseudo_noise_scale=0.3,
            bias_scale=(0.0, 0.01),
            pseudo_bias_scale=(0.0, 0.05),
            gross_error_prob=0.0,
            add_self_loop=False,
            undirected=True,
            shuffle=True,
        )
    return data_list


# ---------------------------------------------------------------------------------------------
# Training and scoring, by the notebook's cells


@contextlib.contextmanager
def _cwd(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def prepare(data_list, seed: int) -> dict:
    """The notebook up to its model: split, loaders, normalisation. `seed` seeds torch for the run."""
    ns = namespace()
    ns["data_list1"] = data_list
    torch.manual_seed(seed)
    for prefix in (SPLIT, STATS_DEF):
        run(prefix, ns)
    with quiet():
        run(STATS, ns)
    for prefix in (NORMALIZE, GNN_CLASS, DIMS, ANGLE_DIFF):
        run(prefix, ns)
    return ns


@contextlib.contextmanager
def _log_to(path: Path):
    """The notebook's training prints (every 10th epoch) go to a file, as they are printed."""
    with open(path, "w", encoding="utf-8", buffering=1) as f, contextlib.redirect_stdout(f):
        yield


def train_gnn(ns: dict, run_dir: Path) -> None:
    """Cells: model and optimiser, then the 100-epoch training loop, as written."""
    with _cwd(run_dir):
        run(GNN_INIT, ns)
        with _log_to(run_dir / "train_log.txt"):
            run(GNN_TRAIN, ns)
        torch.save(ns["model"].state_dict(), run_dir / "model.pt")


def thesis_metrics(ns: dict) -> dict:
    """Table 8's numbers by the notebook's own evaluation cells: the first test batch of 32 samples,
    predictions and truths each averaged over the batch, then compared bus by bus."""
    for prefix in (GNN_EVAL, GNN_ANGLE):
        run(prefix, ns)
    with quiet():
        run(GNN_METRICS, ns)
    return {k: float(ns[k]) for k in ("mae_mag", "rmse_mag", "mae_ang", "circ_rmse")} | {
        "per_bus_mag_err": [float(v) for v in ns["mag_err"]],
    }


def train_pinn(ns: dict, run_dir: Path, grid) -> None:
    """The PINN's cells: class, physics loss, and the training loop, as written."""
    ns["grid"] = grid
    run(PINN_CLASS, ns)
    run(PINN_LOSS, ns)
    with _cwd(run_dir):
        with _log_to(run_dir / "train_log.txt"):
            run(PINN_TRAIN, ns)
        torch.save(ns["model"].state_dict(), run_dir / "model.pt")


def pinn_thesis_metrics(ns: dict) -> dict:
    """Table 9's numbers, by the PINN's evaluation cells (same averaging as Table 8)."""
    run(PINN_EVAL, ns)
    with quiet():
        run(PINN_METRICS, ns)
    return {k: float(ns[k]) for k in ("mae_mag", "rmse_mag", "mae_ang", "circ_rmse")} | {
        "per_bus_mag_err": [float(v) for v in ns["mag_err"]],
    }


def wls_one_sample(seed: int, fix_injections: bool = False) -> dict:
    """Step 4 of the notebook (Table 7): one set of measurements on the base case, GridCal's WLS on it,
    run by the notebook's cells on a freshly loaded grid. Returns true and estimated |V| per bus.

    fix_injections: the same draw, with the two injection bugs undone before GridCal sees it: the
    sign of every P and Q flipped (v1 recorded a load as +P, GridCal counts injections into the grid
    as positive) and the substation's P and Q dropped (v1 recorded its injection as 0 MW, the
    scheduled output of the slack generator, where the power flow gives 3.8 MW)."""
    ns = namespace()
    ns["grid"] = load_grid()
    with seeded_measurements(seed, owner=ns), quiet():
        for prefix in (SE_PF, SE_MEAS, SE_LISTS):
            run(prefix, ns)
        if fix_injections:
            for key in ("pi_list", "qi_list"):
                ns[key] = [(-value, std, i) for value, std, i in ns[key] if i != 0]
        for prefix in (SE_ADD, SE_RUN):
            run(prefix, ns)
    v_true = np.asarray(ns["v_mag"])
    v_est = np.abs(np.asarray(ns["initial_se_results"].results.voltage))
    return {"v_true": v_true, "v_est": v_est, "meas": ns["meas_res"]}

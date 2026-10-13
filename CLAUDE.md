# state-estimation-explorer

v2 of Miloš's master's thesis (UPC Barcelona, 2026): state estimation on the IEEE 33-bus feeder
with WLS, a GNN and a PINN. v1 is the thesis as handed in.

| Where | What it is | Rule |
|---|---|---|
| `C:\dev\portfolio\state-estimation-GNN-PINN` | v1: the thesis, its code, notebook and PDF, written by hand in 2025-26. | Read-only, always. Never edit, move, reformat or commit there; its history has no AI commits and keeps none. `dsse/__init__.py` turns off bytecode so importing it leaves no `__pycache__`; `tests/test_v2.py` checks its `git status` stays empty. `THESIS_DIR` overrides the path. |
| this repo | v2, built from October 2026: `dsse/`, `tests/`, `reports/`. | All new work goes here. |

## How v1 is run

`dsse/v1.py` imports v1's `gridcal_utils` modules from the thesis repo and runs the model, loss,
training and evaluation from the notebook's own cells (the copy in `notebooks/`, the one with the
training loops switched on), found by their first characters. The only things done around them:
each random draw of measurements gets a seed (v1 drew from OS entropy, so its data can't be made
again), and stdout is quietened. A bug found in v1 is written down in `reports/phase1-v1.md` and
fixed in v2, never in v1.

## v2

Python 3.11 venv with the thesis's own pins: `pip install -e ".[test]"`, then `pytest`.

- `python -m dsse.phase1 gnn|pinn --seed N`, `python -m dsse.phase1 wls`: v1 re-run.
- `python -m dsse.phase2 gnn|pinn --seed N`, `python -m dsse.phase2 wls`: v2 on the same split.
  `--recipe scaled` is the fixed training (`dsse/train.py` says why); the PINN's alpha under it is
  chosen by `select-alpha --alpha A` runs on samples 1,400-1,599 (`phase2.chosen_alpha()`).
  `configs` scores every trained model and both WLS with 0 to 33 meters; run it after the models.
- For the website (saricmilos.com runs the grid game in the browser, no server):
  - `dsse/wls_np.py` is the WLS the site ports to TypeScript (Gauss-Newton, dense, no GridCal), with
    observability from real meters (QR rank test) and bad data (chi-square, largest normalised
    residual). `phase2 wls-np` scores it next to GridCal's; `phase2 observability` finds the fewest
    real meters (`dsse/observability.py`, every placement with no two neighbouring gaps).
  - `phase2 anymeter --seed 0|1|2`: the site's GNN, trained on 0 to 33 meters (its own 4,000
    samples); the seed with the lowest validation error is the one exported (`phase2.chosen_anymeter()`).
  - `python -m dsse.export`: writes `feeder.json`, `gnn.json`, `results.json` and a README into
    `<site>/src/content/state-estimation/`, and the TypeScript port's parity cases into
    `<site>/scripts/state-estimation/fixtures.json` (`SE_SITE` names the site, else
    `C:\dev\sites\saricmilos.com`). It touches nothing else in the site and commits nothing there.
- `python -m dsse.evaluate`: writes `reports/` from `data/`. Change the code, never the reports;
  any sentence in a report that depends on a number is computed there, not typed.

Runs take about 30 minutes each on the CPU (v1's PINN about 4 hours: its physics loss loops over
edges in Python). `data/` holds the generated datasets and weights and is not committed.

The fixes, each with its reason, are listed at the top of `dsse/scenarios.py` (data and
measurements), in `dsse/grid.py` (the feeder) and in `dsse/train.py` (the PINN's physics term).
Kept from v1 on purpose: the scenario generator, the meter mix, noise and bias levels, the model
class and every training setting. Nothing is tuned on the test set.

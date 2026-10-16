# State estimation on a distribution feeder, v2

My master's thesis, *Advanced State Estimation in Distribution Networks Using Graph and
Physics-Informed Neural Networks* (UPC Barcelona, 2026), estimated the voltage at every bus of the
IEEE 33-bus feeder from a few meters, three ways: weighted least squares (WLS), a graph neural
network (GNN) and a physics-informed one (PINN).

- **v1** is the thesis as I handed it in:
  [state-estimation-GNN-PINN](https://github.com/saricmilos/state-estimation-GNN-PINN). It is never
  changed.
- **v2** is this repository, built in October 2026. It re-runs v1 from its own code, measures what
  v1 reported, then fixes what was wrong and measures again.

## What was wrong with v1

[reports/phase1-v1.md](reports/phase1-v1.md) has the numbers. In short:

1. **The model was handed the answer.** At unmetered buses the inputs were built from the true
   state: a voltage reading centred on the true voltage, and variances computed from the true
   values.
2. **The metric couldn't tell a model from a constant.** Predictions and truths were averaged over
   32 test scenarios before being compared, which cancels every error except a constant offset.
3. **WLS was given wrong injections.** Loads had the wrong sign, and the substation was recorded as
   supplying 0 MW, as the most trusted reading in the set.
4. **The feeder's lines were converted to per-unit twice**, so it was not the IEEE 33-bus system.
5. **The PINN's physics term wasn't Ohm's law.** It used the wrong current, normalised impedances
   and mixed units, so it could not be zero at the true state.

## What v2 finds

[reports/phase2-v2.md](reports/phase2-v2.md): WLS, the thesis's GNN and a PINN with a correct
Ohm's-law term, retrained and scored on every bus of 400 test samples, then on 3 to 20 meters.

1. **On honest data, the thesis's training isn't enough.** Every model ends with a constant
   offset, even on its own training samples, and the GNN does worse than always answering the
   average voltage profile.
2. **With the training fixed** (standardised outputs, a decaying learning rate; same model and
   data), the GNN is steady from run to run and clearly better than that baseline.
3. **WLS is still the most accurate, from 3 meters up.** On this data the load errors are
   independent and Gaussian, which is exactly what WLS assumes, so little is left for a learned
   model to win. The fixed GNN degrades as gracefully as WLS, but stays behind it. With a single
   meter WLS falls behind even the average voltage profile, because that one reading sets the
   level of the whole estimate; there the GNN is about the average profile.
4. **A correct physics term doesn't help as the thesis framed it.** At the weight chosen on
   validation the PINN is the GNN; more weight makes it obey Ohm's law better and estimate worse,
   because the term trusts a 50% load forecast as much as a meter. Weighting it by the readings'
   variances turns it into the WLS objective: the hybrid the thesis proposed as future work.

## For the website

My site runs the feeder in the browser: a visitor places meters, and the power flow, WLS and the
GNN are computed there, with no server. [reports/phase2-v2.md](reports/phase2-v2.md) has the
pieces that needed:

- **WLS without GridCal.** `dsse/wls_np.py` solves the same weighted least squares by
  Gauss-Newton with plain matrices, for a TypeScript port. It lands on GridCal's answer wherever
  GridCal reaches the minimum, and lower where GridCal's solver stalls.
- **Observability.** Without load forecasts the meters alone must determine the state. Counting
  says at least 22 meters; testing every placement shows 22 can do it in only a few thousand
  arrangements, and even those estimate |V| too loosely to use.
- **Bad data.** A wrong voltage reading is caught; a wrong P or Q reading is not, because only
  forecasts could contradict it.
- **A GNN for any number of meters,** trained the fixed way on 0 to 33 meters, since the thesis's
  models only ever saw 10.

`python -m dsse.export` writes what the page loads into the site's repository.

## Run it

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[test]"
.venv\Scripts\pytest
.venv\Scripts\python -m dsse.phase1 gnn --seed 0
.venv\Scripts\python -m dsse.phase2 gnn --seed 0
.venv\Scripts\python -m dsse.evaluate
```

For the website: `dsse.phase2 wls-np`, `observability`, `anymeter --seed 0` (to 2) and `configs`,
then `dsse.evaluate` and `dsse.export`.

v1 is read from `C:\dev\portfolio\state-estimation-GNN-PINN`, or from `THESIS_DIR` if set.

# State estimation on a distribution feeder, v2

My master's thesis, *Advanced State Estimation in Distribution Networks Using Graph and
Physics-Informed Neural Networks* (UPC Barcelona, 2026), estimated the voltage at every bus of the
IEEE 33-bus feeder from a few meters, three ways: weighted least squares (WLS), a graph neural
network (GNN) and a physics-informed one (PINN).

- **v1** is the thesis as I handed it in, code written by hand:
  [state-estimation-GNN-PINN](https://github.com/saricmilos/state-estimation-GNN-PINN). It is never
  changed.
- **v2** is this repository, built with Claude Code in October 2026. I set the questions and checked
  every number. It re-runs v1 from its own code, measures what v1 reported, then fixes what was
  wrong and measures again.

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
Ohm's-law term, retrained with the thesis's settings and scored on every bus of 400 test samples.

## Run it

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[test]"
.venv\Scripts\pytest
.venv\Scripts\python -m dsse.phase1 gnn --seed 0
.venv\Scripts\python -m dsse.phase2 gnn --seed 0
.venv\Scripts\python -m dsse.evaluate
```

v1 is read from `C:\dev\portfolio\state-estimation-GNN-PINN`, or from `THESIS_DIR` if set.

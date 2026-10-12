"""What the thesis published, read from the thesis itself, so reports compare against its numbers.

Tables 8 and 9 come from the notebook at the thesis repository's root, the copy whose saved outputs
are the ones printed in the thesis (its per-bus tables match Tables 8 and 9 digit for digit).
Table 7 was a different run from the one saved in either notebook, so its column of |V| errors is
written out below from the PDF (Milos_Saric_TFM.pdf, printed page 88).
"""

import json
import re

from . import paths

ROOT_NOTEBOOK = paths.THESIS / "state_stimation_GNN_PINN_milos_saric.ipynb"

# Table 7, column "Delta Vm" (SE minus PF, p.u., rounded to 3 decimals as printed), buses 0..32
TABLE7_DVM = [
    -0.001, 0.001, 0.008, 0.009, 0.010, 0.013, 0.012, 0.009, 0.004, -0.003, -0.004, -0.006, -0.004,
    -0.004, -0.004, -0.004, -0.005, -0.005, 0.002, 0.007, 0.008, 0.008, 0.012, 0.020, 0.024, 0.014,
    0.015, 0.022, 0.027, 0.029, 0.033, 0.034, 0.034,
]

_NUM = r"([0-9.eE+-]+)"


def _outputs() -> list[str]:
    nb = json.loads(ROOT_NOTEBOOK.read_text(encoding="utf-8"))
    texts = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                texts.append("".join(o["text"]))
    return texts


def thesis_numbers() -> dict:
    """Vm MAE/RMSE (p.u.) and angle MAE/RMSE (rad) as the thesis reports them."""
    gnn = pinn = None
    for t in _outputs():
        m = re.search(rf"RMSE mag: {_NUM} MAE mag: {_NUM}\s+Circ-RMSE angle: {_NUM} MAE angle: {_NUM}", t)
        if m:
            gnn = m
        m = re.search(rf"RMSE mag: {_NUM}, MAE mag: {_NUM}\s+Circular RMSE angle: {_NUM}, MAE angle: {_NUM}", t)
        if m:
            pinn = m
    if not (gnn and pinn):
        raise LookupError("the thesis notebook's printed metrics were not found")

    def row(m):
        return {"vm_rmse": float(m[1]), "vm_mae": float(m[2]), "va_rmse": float(m[3]), "va_mae": float(m[4])}

    dvm = TABLE7_DVM
    return {
        "gnn": row(gnn),
        "pinn": row(pinn),
        "wls": {"vm_mae": sum(abs(x) for x in dvm) / len(dvm), "vm_max": max(abs(x) for x in dvm)},
    }

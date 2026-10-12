import subprocess

import numpy as np
import pytest
import torch

from dsse import grid as G
from dsse import paths, published, scenarios, train, v1


def test_thesis_repository_is_untouched():
    """Nothing here may write into v1: no edits, no __pycache__, no cache files."""
    out = subprocess.run(
        ["git", "-C", str(paths.THESIS), "status", "--porcelain", "--ignored"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out == ""


def test_v1_cells_are_found():
    for prefix in (v1.GNN_CLASS, v1.GNN_TRAIN, v1.PINN_LOSS, v1.PINN_TRAIN, v1.GNN_METRICS, v1.SE_RUN):
        assert v1.cell(prefix)


def test_published_numbers():
    n = published.thesis_numbers()
    assert n["gnn"]["vm_mae"] == pytest.approx(0.00878853)
    assert n["pinn"]["vm_mae"] == pytest.approx(0.007678)
    assert n["wls"]["vm_mae"] == pytest.approx(0.395 / 33)


def test_v1_divides_the_lines_twice():
    """GridCal's case33bw is per-unit already; v1's cell 22 divides by Z_base again."""
    loaded = G.load_gridcal()
    assert loaded.lines[0].R == pytest.approx(0.0922 / (12.66**2 / 100), rel=1e-6)
    assert v1.load_grid().lines[0].R == pytest.approx(loaded.lines[0].R / (12.66**2 / 100), rel=1e-6)


def test_base_case_is_the_published_one():
    g = G.load_gridcal()
    fd = G.feeder(g)
    v, s = G.gridcal_power_flow(g, fd.p_load, fd.q_load)
    assert np.abs(v).min() == pytest.approx(0.9131, abs=1e-4)
    assert np.argmin(np.abs(v)) == 17
    assert (s.real.sum()) * 1000 == pytest.approx(202.7, abs=0.1)  # losses, kW
    assert len(fd.t) == 32


def test_sweep_matches_newton_raphson():
    g = G.load_gridcal()
    fd = G.feeder(g)
    rng = np.random.default_rng(3)
    for _ in range(5):
        factor = 1 + 0.5 * rng.normal(size=fd.n)
        v, _ = G.gridcal_power_flow(g, fd.p_load * factor, fd.q_load * factor)
        assert np.abs(G.sweep(fd, fd.p_load * factor, fd.q_load * factor)[0] - v).max() < 1e-7


@pytest.fixture(scope="module")
def small():
    fd, samples = scenarios.generate(num_samples=40, seed=7)
    return fd, samples, [scenarios.to_data(fd, s) for s in samples]


def test_pseudo_measurements_carry_nothing_of_the_truth(small):
    """A pseudo bus's reading and variance are the same whatever the sample's true state, up to v1's
    +-5% pseudo bias, and it has no voltage reading."""
    fd, samples, _ = small
    for b in range(fd.n):
        rows = [s for s in samples if s.sensor[b] == scenarios.SENSOR_TYPES.index("pseudo")]
        if len(rows) < 2:
            continue
        assert np.ptp([s.var[b] for s in rows], axis=0).max() == 0
        assert all(s.mask[b, 0] == 0 and s.mask[b, 1] == 0 for s in rows)
        p = np.array([s.obs[b, 2] for s in rows])
        mean = rows[0].obs[b, 2]
        assert np.all(np.abs(p - np.median(p)) <= 2 * scenarios.PSEUDO_BIAS * abs(mean) + 1e-12)


def test_injections_are_the_power_flows(small):
    fd, samples, _ = small
    for s in samples:
        assert s.s[0].real > 1  # the substation supplies the feeder
        assert np.mean(s.s[1:].real < 0) > 0.8  # loads draw power (a few are perturbed past zero)
        assert abs(s.s.real.sum()) < 0.5  # what goes in comes out, less the losses


def test_physics_term_is_zero_at_the_true_state(small):
    from torch_geometric.data import Batch

    fd, samples, data = small
    b = Batch.from_data_list(data[:8])
    b.meas = b.meas.clone()
    s = np.concatenate([smp.s for smp in samples[:8]])
    b.meas[:, 2] = torch.tensor(s.real, dtype=torch.float32)
    b.meas[:, 3] = torch.tensor(s.imag, dtype=torch.float32)
    assert train.physics_residual(b.y[:, 0], b.y[:, 1], b, fd).max().item() < 1e-6


def test_scaled_outputs_round_trip(small):
    """The fixed recipe's outputs are |V| and angle standardised by the training set; a model that
    outputs the standardised truth must give back the truth."""
    _, _, data = small
    norm = train.Normalizer(data)
    y = torch.cat([d.y for d in data[:4]])
    z = (y - norm.y_mean) / norm.y_std
    vm, va = train.to_state(z, "scaled", norm)
    assert torch.allclose(vm, y[:, 0], atol=1e-6) and torch.allclose(va, y[:, 1], atol=1e-6)


def test_graph_has_only_energised_lines(small):
    fd, _, data = small
    assert data[0].edge_index.shape[1] == 2 * 32
    assert data[0].x.shape == (33, 17)

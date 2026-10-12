"""Training v2's GNN and PINN, two ways.

Both use v1's model classes, taken from the notebook's cells (GNNStateEstimator2 and PINNGNN: the
same two NNConv layers, LayerNorm, dropout 0.2 and residual), v1's input features and standardisation,
batch 32, Adam at 1e-3 with weight decay 1e-5, 100 epochs and gradients clipped at 5.

recipe "thesis": v1's training as it was. The model outputs |V|, sin(angle), cos(angle) in raw
  units and the learning rate stays at 1e-3. On this feeder |V| sits near 0.95 and the angles within
  +-0.01 rad, so cos(angle) is 0.9999 whatever the angle and the useful signal is a few thousandths;
  Adam's steps at 1e-3 are as large as that, and a trained model ends with a constant offset of
  0.003-0.01 p.u. even on its own training samples (phase 2 report).
recipe "scaled": the fix. The model outputs |V| and the angle standardised by the training set's
  mean and spread (as v1 already did for the inputs), and the learning rate decays to zero over the
  100 epochs (cosine schedule). Same model class, data and number of epochs.

The PINN adds the physics term (physics_residual) to the loss, weighted by alpha.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import v1
from .grid import Feeder

HIDDEN, EPOCHS, LR, WEIGHT_DECAY, BATCH, CLIP = 64, 100, 1e-3, 1e-5, 32, 5.0
ALPHA = 0.1  # v1's weight on the physics term
OUTPUTS = {"thesis": 3, "scaled": 2}


def thesis_classes():
    """v1's GNN and PINN classes, built from the notebook's own cells."""
    ns = v1.namespace()
    v1.run(v1.GNN_CLASS, ns)
    v1.run(v1.PINN_CLASS, ns)
    return ns["GNNStateEstimator2"], ns["PINNGNN"]


class Normalizer:
    """v1's standardisation of the inputs (mean and std of the training set), except that a feature
    which never varies in training is left unscaled: v1 divided it by 1e-8, so any other value at
    inference would be multiplied by 1e8. Also holds the targets' mean and spread, for "scaled"."""

    KEYS = ("x_mean", "x_std", "e_mean", "e_std", "y_mean", "y_std")

    def __init__(self, train):
        xs = torch.cat([d.x for d in train])
        ex = torch.cat([d.edge_attr for d in train])
        ys = torch.cat([d.y for d in train])
        self.x_mean, self.x_std = xs.mean(0), xs.std(0)
        self.e_mean, self.e_std = ex.mean(0), ex.std(0)
        self.y_mean, self.y_std = ys.mean(0), ys.std(0)  # |V| and angle, over every bus
        self.x_std[self.x_std < 1e-6] = 1.0
        self.e_std[self.e_std < 1e-6] = 1.0

    def __call__(self, batch):
        batch = batch.clone()
        batch.x = (batch.x - self.x_mean) / self.x_std
        batch.edge_attr = (batch.edge_attr - self.e_mean) / self.e_std
        return batch

    def state(self) -> dict:
        return {k: getattr(self, k) for k in self.KEYS}

    @classmethod
    def from_state(cls, state: dict) -> "Normalizer":
        norm = cls.__new__(cls)
        for k, v in state.items():
            setattr(norm, k, v)
        return norm


def to_state(out, recipe: str, norm: Normalizer):
    """The model's output as |V| (p.u.) and angle (rad) per bus."""
    if recipe == "thesis":
        return out[:, 0], torch.atan2(out[:, 1], out[:, 2])
    return norm.y_mean[0] + norm.y_std[0] * out[:, 0], norm.y_mean[1] + norm.y_std[1] * out[:, 1]


def data_loss(out, y, recipe: str, norm: Normalizer):
    if recipe == "thesis":  # v1's loss
        loss_mag = F.mse_loss(out[:, 0], y[:, 0])
        loss_sin = F.mse_loss(out[:, 1], torch.sin(y[:, 1]))
        loss_cos = F.mse_loss(out[:, 2], torch.cos(y[:, 1]))
        return loss_mag + 0.5 * (loss_sin + loss_cos)
    z = (y - norm.y_mean) / norm.y_std  # same weights as v1: the angle counts half
    return F.mse_loss(out[:, 0], z[:, 0]) + 0.5 * F.mse_loss(out[:, 1], z[:, 1])


def physics_residual(vm, va, batch, fd: Feeder):
    """Ohm's law on every energised line, |V_i - V_j - Z_ij I_ij| in p.u., as in the thesis (eq. 101).

    v1 took I_ij to be the injection current of bus i (eq. 100), used the standardised R and X, and
    mixed MW with p.u. Here the line current is what the line really carries on a radial feeder: the
    sum of the currents drawn at every bus beyond it (Kirchhoff's current law), each conj(S_k / V_k)
    with S_k the bus's measured or pseudo-measured injection in p.u. and V_k the predicted voltage;
    R and X are the lines' own, in p.u. With the true voltages and true injections it is zero."""
    b = batch.num_graphs
    v = (vm * torch.exp(1j * va)).reshape(b, fd.n)
    meas = batch.meas.reshape(b, fd.n, 4)
    s = torch.complex(meas[..., 2], meas[..., 3]) / fd.sbase
    i_draw = -torch.conj(s / v)  # current drawn at each bus (s is positive into the grid)
    d = torch.as_tensor(fd.downstream, dtype=torch.complex64)
    i_line = i_draw @ d.T
    z = torch.as_tensor(fd.z, dtype=torch.complex64)
    f, t = torch.as_tensor(fd.f), torch.as_tensor(fd.t)
    return torch.abs(v[:, f] - v[:, t] - z * i_line)


def physics_loss(out, batch, fd, recipe, norm):
    """The mean residual; under "scaled", in units of |V|'s spread, like the data loss."""
    vm, va = to_state(out, recipe, norm)
    r = physics_residual(vm, va, batch, fd).mean()
    return r if recipe == "thesis" else r / norm.y_std[0]


def fit(train, kind: str, seed: int, fd: Feeder, recipe: str = "thesis", alpha: float = ALPHA, log=print):
    gnn_cls, pinn_cls = thesis_classes()
    torch.manual_seed(seed)
    norm = Normalizer(train)
    in_ch, edge_dim = train[0].x.shape[1], train[0].edge_attr.shape[1]
    model = (gnn_cls if kind == "gnn" else pinn_cls)(in_ch, HIDDEN, OUTPUTS[recipe], edge_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if recipe == "scaled" else None
    loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            out = model(norm(batch))
            loss = data_loss(out, batch.y, recipe, norm)
            if kind == "pinn":
                loss = loss + alpha * physics_loss(out, batch, fd, recipe, norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.item() * batch.num_graphs
        if sched:
            sched.step()
        if epoch % 10 == 0 or epoch == 1:
            log(f"epoch {epoch:03d} loss {total / len(train):.6f} ({time.time() - t0:.0f}s)")
    return model, norm


def predict(model, norm, data, fd: Feeder, recipe: str = "thesis"):
    """|V| and angle for every bus of every sample, plus the physics residual of each line."""
    from torch_geometric.data import Batch

    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(list(data))
        vm, va = to_state(model(norm(batch)), recipe, norm)
        res = physics_residual(vm, va, batch, fd).numpy()
    return vm.reshape(len(data), fd.n).numpy(), va.reshape(len(data), fd.n).numpy(), res

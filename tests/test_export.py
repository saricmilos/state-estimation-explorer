import json

import numpy as np
import pytest
import torch

from dsse import export, scenarios, train


@pytest.fixture(scope="module")
def small():
    fd, samples = scenarios.generate(num_samples=12, seed=5, n_metered=scenarios.any_meters, n_pmu=scenarios.any_pmus)
    return fd, samples, [scenarios.to_data(fd, s) for s in samples]


def test_forward_pass_matches_torch(small):
    """gnn.json's forward pass, done by hand in float64 from the exported float32 blobs, is the model's."""
    fd, samples, data = small
    gnn_cls, _ = train.thesis_classes()
    torch.manual_seed(0)
    model = gnn_cls(17, train.HIDDEN, 2, 2)
    for p in model.parameters():  # not just the initial values: LayerNorm's are all ones and zeros
        p.data += 0.1 * torch.randn_like(p)
    norm = train.Normalizer(data)
    g = json.loads(json.dumps(export.gnn_json(model, norm, fd, {})))
    assert len(json.dumps(g)) < 1_000_000
    for s in samples:
        vm_t, va_t = export.gnn_predict(model, norm, fd, s)
        vm_n, va_n = export.forward_np(g, export.features(fd, s))
        assert np.abs(vm_t - vm_n).max() < 1e-5 and np.abs(va_t - va_n).max() < 1e-5


class Replay:
    """Hands back recorded draws in order, checking each is asked for the same way."""

    def __init__(self, draws):
        self.draws = list(draws)

    def _next(self, fn, **args):
        d = self.draws.pop(0)
        assert d["fn"] == fn and all(d[k] == v for k, v in args.items())
        return d["value"]

    def random(self):
        return self._next("random")

    def normal(self, loc, scale):
        return self._next("normal", loc=loc, scale=scale)

    def uniform(self, low, high):
        return self._next("uniform", low=low, high=high)


def test_recorded_draws_give_back_the_readings(small):
    fd, samples, _ = small
    s = samples[3]
    prior = scenarios.pseudo_prior(fd, samples[0].s)
    metered, pmus = {1, 5, 9, 17, 30}, {5, 30}
    rec = export.RecordingRng(np.random.default_rng(1))
    first = scenarios.measure(s.v, s.s, metered, pmus, prior, rec)
    replay = Replay(rec.draws)
    again = scenarios.measure(s.v, s.s, metered, pmus, prior, replay)
    assert not replay.draws
    assert np.array_equal(first.obs, again.obs) and np.array_equal(first.var, again.var)


def test_fixture_cases_cover_what_the_page_needs():
    names = [c[0] for c in export.CASES]
    assert len(set(names)) == len(names) == 24
    assert {c[1] for c in export.CASES} == {0, 3, 10, 33}
    assert {c[3] for c in export.CASES} == {None, 0.5, 1.5}
    assert sum(c[4] for c in export.CASES) == 1

from __future__ import annotations

from pathlib import Path

from pegasus.compute.random import torch_generator


class _FakeGenerator:
    def __init__(self, *, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class _FakeTorch:
    def Generator(self, device=None):
        return _FakeGenerator(device=device)


def test_slice28zb_torch_generator_uses_central_seed_boundary() -> None:
    generator = torch_generator(_FakeTorch(), seed=20260614, device="cpu")
    assert generator.device == "cpu"
    assert generator.seed == 20260614


def test_slice28zb_hsic_no_longer_seeds_local_generators_directly() -> None:
    source = Path("src/pegasus/pirs/hsic.py").read_text(encoding="utf-8")
    assert "from pegasus.compute.random import torch_generator" in source
    assert "torch_generator(torch, seed=seed, device=values.device)" in source
    assert "generator.manual_seed(" not in source
    assert "torch.Generator(" not in source

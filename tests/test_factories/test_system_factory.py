# tests/test_factories/test_system_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_random_system
from src.gamecore.utils.utils import is_stabilizable
from src.gamecore import LinearSystem
from tests.conftest import SEED

@pytest.mark.parametrize("stabilizability", ["individual", "joint"])
@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
def test_make_random_system_valid(stabilizability: str, time_domain: str) -> None:
    """Test generation of random stabilizable systems for both modes."""
    n = 2
    ms = [2, 2]
    system = make_random_system(
        n=n, ms=ms, stabilizability=stabilizability, time_domain=time_domain, sparsity=0.1, seed=SEED
    )

    assert isinstance(system, LinearSystem)
    assert system.A.shape == (n, n)
    assert len(system.Bs) == len(ms)
    for B, m in zip(system.Bs, ms):
        assert B.shape == (n, m)

    if stabilizability == "joint":
        B_total = np.hstack(system.Bs)
        assert is_stabilizable(system.A, B_total)
    else:
        assert all(is_stabilizable(system.A, B) for B in system.Bs)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_target_spectral_radius_is_dimension_independent(n: int) -> None:
    """The realized spectral radius should hit the target exactly, regardless of n."""
    target = 1.5
    system = make_random_system(n=n, ms=[1], target_spectral_radius=target, seed=SEED)
    rho = np.max(np.abs(np.linalg.eigvals(system.A)))
    assert np.isclose(rho, target, atol=1e-8)


def test_target_input_norm() -> None:
    target = 2.0
    system = make_random_system(n=3, ms=[2, 1], target_input_norm=target, seed=SEED)
    for B in system.Bs:
        assert np.isclose(np.linalg.norm(B, ord=2), target, atol=1e-8)

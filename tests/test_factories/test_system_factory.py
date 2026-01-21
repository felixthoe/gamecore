# tests/test_factories/test_system_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_random_system
from src.gamecore.utils.utils import is_stabilizable
from src.gamecore import LinearSystem
from tests.conftest import SEED

@pytest.mark.parametrize("stabilizability", ["individual", "joint"])
@pytest.mark.parametrize("game_type", ["differential", "dynamic"])
def test_make_random_system_valid(stabilizability: str, game_type: str) -> None:
    """Test generation of random stabilizable systems for both modes."""
    n = 2
    ms = [2, 2]
    system = make_random_system(
        n=n, ms=ms, stabilizability=stabilizability, game_type=game_type, sparsity=0.1, seed=SEED
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

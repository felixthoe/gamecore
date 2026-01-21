# tests/test_factories/test_cost_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_random_costs
from src.gamecore import QuadraticCost, LinearSystem
from tests.conftest import SEED


################################
# Shared fixtures
################################

@pytest.fixture
def system() -> LinearSystem:
    """Provides a simple stabilizable differential linear system."""
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B1 = np.array([[0.0], [1.0]])
    B2 = np.array([[0.0], [1.0]])
    return LinearSystem(A=A, Bs=[B1, B2])


################################
# Tests
################################

@pytest.mark.parametrize("q_def", ["pd", "psd"])
@pytest.mark.parametrize("r_jj", ["zero", "psd", "free"])
@pytest.mark.parametrize("r_jk", ["zero", "free"])
def test_make_random_costs(
    system: LinearSystem,
    q_def: str,
    r_jj: str,
    r_jk: str
) -> None:
    # exclude invalid combinations
    if r_jk == "free" and r_jj == "zero":
        pytest.skip("Invalid combination of r_jj and r_jk")
    costs = make_random_costs(
        system=system,
        q_def=q_def,
        r_jj=r_jj,
        r_jk=r_jk,
        seed=SEED,
        sparsity=0.1,
    )
    assert isinstance(costs, list)
    assert len(costs) == system.N
    for cost in costs:
        assert isinstance(cost, QuadraticCost)

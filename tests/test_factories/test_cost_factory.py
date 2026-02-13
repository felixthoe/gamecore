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

@pytest.mark.parametrize("q_i_def", ["pd", "psd"])
@pytest.mark.parametrize("r_i_jj", ["zero", "psd", "free"])
@pytest.mark.parametrize("r_i_jk", ["zero", "free"])
@pytest.mark.parametrize("enforce_psd_r_i", [True, False])
def test_make_random_costs(
    system: LinearSystem,
    q_i_def: str,
    r_i_jj: str,
    r_i_jk: str,
    enforce_psd_r_i: bool
) -> None:
    # exclude invalid combinations
    if r_i_jk == "free" and r_i_jj == "zero" and enforce_psd_r_i:
        pytest.skip("Invalid combination of r_i_jj, r_i_jk, and enforce_psd_r_i")
    costs = make_random_costs(
        system=system,
        q_i_def=q_i_def,
        r_i_jj=r_i_jj,
        r_i_jk=r_i_jk,
        enforce_psd_r_i=enforce_psd_r_i,
        seed=SEED,
        sparsity=0.1,
    )
    assert isinstance(costs, list)
    assert len(costs) == system.N
    for cost in costs:
        assert isinstance(cost, QuadraticCost)

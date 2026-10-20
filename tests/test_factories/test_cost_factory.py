# tests/test_factories/test_cost_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_random_costs, make_random_system
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

@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
@pytest.mark.parametrize("q_i", ["pd", "psd"])
@pytest.mark.parametrize("r_ijj", ["zero", "psd", "free"])
@pytest.mark.parametrize("r_ijk", ["zero", "free"])
@pytest.mark.parametrize("enforce_psd_r_i", [True, False])
def test_make_random_costs(
    system: LinearSystem,
    time_domain: str,
    q_i: str,
    r_ijj: str,
    r_ijk: str,
    enforce_psd_r_i: bool
) -> None:
    # exclude invalid combinations
    if r_ijk == "free" and r_ijj == "zero" and enforce_psd_r_i:
        pytest.skip("Invalid combination of r_ijj, r_ijk, and enforce_psd_r_i")
    costs = make_random_costs(
        system=system,
        time_domain=time_domain,
        q_i=q_i,
        r_ijj=r_ijj,
        r_ijk=r_ijk,
        enforce_psd_r_i=enforce_psd_r_i,
        seed=SEED,
        sparsity=0.1,
    )
    assert isinstance(costs, list)
    assert len(costs) == system.N
    for cost in costs:
        assert isinstance(cost, QuadraticCost)


def test_psd_costs_see_the_zero_modes_of_a_sparse_system() -> None:
    """Detectability of (A, Q_i) includes the zero modes of a singular A."""
    for seed in range(300):
        sparse_system = make_random_system(n=5, ms=[1], sparsity=0.8, seed=seed)
        if np.linalg.svd(sparse_system.A, compute_uv=False)[-1] < 1e-10:
            costs = make_random_costs(sparse_system, q_i="psd", sparsity=0.4, seed=seed)
            assert all(np.linalg.svd(np.vstack([sparse_system.A, cost.Q]), compute_uv=False)[-1] > 1e-10 for cost in costs)

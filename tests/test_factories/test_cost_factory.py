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
@pytest.mark.parametrize("cross_terms", [False, True])
def test_make_random_costs(system: LinearSystem, time_domain: str, q_i: str, cross_terms: bool) -> None:
    costs = make_random_costs(
        system=system,
        time_domain=time_domain,
        q_i=q_i,
        cross_terms=cross_terms,
        seed=SEED,
        sparsity=0.1,
    )
    assert isinstance(costs, list)
    assert len(costs) == system.N
    for cost in costs:
        assert isinstance(cost, QuadraticCost)


@pytest.mark.parametrize("diag", [True, False])
def test_cross_terms_keep_the_diagonal_blocks(diag: bool) -> None:
    """R_i with cross terms is PSD and has the diagonal blocks of the same seed without them."""
    system = make_random_system(n=3, ms=[2, 1, 2], seed=SEED)
    plain = make_random_costs(system, diag=diag, seed=SEED)
    crossed = make_random_costs(system, diag=diag, cross_terms=True, seed=SEED)
    offsets = np.cumsum([0] + system.ms)
    for cost_plain, cost_crossed in zip(plain, crossed):
        R = np.block([[cost_crossed.R[(j, k)] for k in range(system.N)] for j in range(system.N)])
        assert np.min(np.linalg.eigvalsh(R)) >= -1e-10 * np.linalg.norm(R, 2)
        for j in range(system.N):
            assert np.allclose(cost_crossed.R[(j, j)], cost_plain.R[(j, j)])
            assert np.allclose(R[offsets[j]:offsets[j + 1], offsets[j]:offsets[j + 1]], cost_plain.R[(j, j)])
        assert any(np.any(cost_crossed.R[(j, k)]) for j in range(system.N) for k in range(system.N) if j != k)


def test_psd_costs_see_the_zero_modes_of_a_sparse_system() -> None:
    """Detectability of (A, Q_i) includes the zero modes of a singular A."""
    for seed in range(300):
        sparse_system = make_random_system(n=5, ms=[1], sparsity=0.8, seed=seed)
        if np.linalg.svd(sparse_system.A, compute_uv=False)[-1] < 1e-10:
            costs = make_random_costs(sparse_system, q_i="psd", sparsity=0.4, seed=seed)
            assert all(np.linalg.svd(np.vstack([sparse_system.A, cost.Q]), compute_uv=False)[-1] > 1e-10 for cost in costs)

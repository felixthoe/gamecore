# tests/test_factories/test_strategy_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_lqr_strategy, make_random_strategies, make_random_costs, make_random_system
from src.gamecore import FactorySamplingError, LinearSystem, LinearStrategy
from tests.conftest import SEED


################################
# Shared fixtures
################################

@pytest.fixture
def differential_system() -> LinearSystem:
    """Provides a simple stabilizable differential linear system."""
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B1 = np.array([[0.0], [1.0]])
    B2 = np.array([[0.0], [1.0]])
    return LinearSystem(A=A, Bs=[B1, B2])

@pytest.fixture
def dynamic_system() -> LinearSystem:
    """Provides a simple stabilizable dynamic linear system."""
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B1 = np.array([[0.0], [1.0]])
    B2 = np.array([[0.0], [1.0]])
    return LinearSystem(A=A, Bs=[B1, B2])


################################
# Tests
################################

@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
def test_make_lqr_strategy(differential_system: LinearSystem, dynamic_system: LinearSystem, time_domain: str) -> None:
    """Test computation of optimal LQR gain for a single player."""
    if time_domain == "continuous":
        system = differential_system
    else:
        system = dynamic_system
    cost = make_random_costs(system=system, seed=SEED)[0]
    strategy = make_lqr_strategy(system=system, cost=cost, player_idx=0, time_domain=time_domain)
    assert isinstance(strategy, LinearStrategy)
    assert strategy.K.shape == (system.ms[0], system.n)

@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
def test_make_random_strategies(differential_system: LinearSystem, dynamic_system: LinearSystem, time_domain: str) -> None:
    """Test random stabilizing strategies."""
    if time_domain == "continuous":
        system = differential_system
    else:
        system = dynamic_system
    strategies = make_random_strategies(system=system, time_domain=time_domain, seed=SEED)
    assert isinstance(strategies, list)
    assert len(strategies) == system.N
    assert all(isinstance(strategy, LinearStrategy) for strategy in strategies)


def _is_closed_loop_stable(system: LinearSystem, strategies: list[LinearStrategy], time_domain: str) -> bool:
    A_cl = system.A_cl(strategies)
    eigs = np.linalg.eigvals(A_cl)
    if time_domain == "continuous":
        return bool(np.all(np.real(eigs) < 0))
    return bool(np.all(np.abs(eigs) < 1.0))


@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
@pytest.mark.parametrize("strategy_init", ["joint_lqr", "random_pole_placement", "random_bisection"])
def test_make_random_strategies_modes_are_stabilizing(
    differential_system: LinearSystem, dynamic_system: LinearSystem, time_domain: str, strategy_init: str
) -> None:
    """Every strategy_init mode must produce a certified-stabilizing joint gain, not just the right shapes."""
    system = differential_system if time_domain == "continuous" else dynamic_system
    strategies = make_random_strategies(system=system, time_domain=time_domain, strategy_init=strategy_init, seed=SEED)
    assert _is_closed_loop_stable(system, strategies, time_domain)


@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
def test_random_pole_placement_keeps_uncontrollable_mode(time_domain: str) -> None:
    """The stable, uncontrollable third state must keep its eigenvalue; placing it anyway used to
    return arbitrarily large gains."""
    a_u = -0.7 if time_domain == "continuous" else 0.3
    A = np.array([[0.0, 1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 0.0, a_u]])
    B1 = np.array([[0.0], [1.0], [0.0]])
    B2 = np.array([[1.0], [1.0], [0.0]])
    system = LinearSystem(A=A, Bs=[B1, B2])
    strategies = make_random_strategies(system=system, time_domain=time_domain, strategy_init="random_pole_placement", seed=SEED)
    K = np.vstack([s.K for s in strategies])
    eigs = np.linalg.eigvals(A - np.hstack([B1, B2]) @ K)
    assert np.min(np.abs(eigs - a_u)) < 1e-8
    assert np.linalg.norm(K) <= 1e3
    assert _is_closed_loop_stable(system, strategies, time_domain)


def test_random_bisection_is_not_near_optimal(differential_system: LinearSystem) -> None:
    """random_bisection should differ substantially from the near-optimal joint_lqr gain, not be a
    lightly-perturbed copy of it."""
    good = make_random_strategies(system=differential_system, strategy_init="joint_lqr", seed=SEED)
    bad = make_random_strategies(system=differential_system, strategy_init="random_bisection", seed=SEED)
    good_norm = np.concatenate([s.K.flatten() for s in good])
    bad_norm = np.concatenate([s.K.flatten() for s in bad])
    assert not np.allclose(good_norm, bad_norm, rtol=0.5)


@pytest.mark.parametrize("strategy_init", ["joint_lqr", "random_pole_placement", "random_bisection"])
def test_uncontrollable_mode_within_the_margin_is_rejected(strategy_init: str) -> None:
    """No gain moves the uncontrollable third state, whose stability margin 0.02 is below the required 0.05."""
    A = np.array([[0.0, 1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 0.0, -0.01]])
    system = LinearSystem(A=A, Bs=[np.array([[0.0], [1.0], [0.0]]), np.array([[1.0], [1.0], [0.0]])])
    with pytest.raises(FactorySamplingError):
        make_random_strategies(system=system, strategy_init=strategy_init, seed=SEED)


def test_joint_lqr_gain_norm_is_bounded() -> None:
    """The weakly controllable unstable mode needs a joint LQR gain of about 2e4."""
    system = LinearSystem(A=np.diag([1.0, -1.0]), Bs=[np.array([[1e-4], [1.0]])])
    with pytest.raises(FactorySamplingError):
        make_random_strategies(system=system, strategy_init="joint_lqr", seed=SEED)


def test_joint_lqr_meets_the_margin_on_sparse_systems() -> None:
    """Sparse systems have slow modes that a joint LQR without prescribed margin leaves near the stability boundary."""
    for seed in range(50):
        system = make_random_system(n=5, ms=[1, 1], sparsity=0.8, seed=seed)
        try:
            strategies = make_random_strategies(system=system, strategy_init="joint_lqr", seed=seed)
        except FactorySamplingError:
            continue
        assert np.min(-2 * np.real(np.linalg.eigvals(system.A_cl(strategies)))) > 0.05

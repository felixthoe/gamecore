# tests/test_factories/test_strategy_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_lqr_strategy, make_random_strategies, make_random_costs
from src.gamecore import LinearSystem, LinearStrategy
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

@pytest.mark.parametrize("game_type", ["differential", "dynamic"])
def test_make_lqr_strategy(differential_system: LinearSystem, dynamic_system: LinearSystem, game_type: str) -> None:
    """Test computation of optimal LQR gain for a single player."""
    if game_type == "differential":
        system = differential_system
    else:
        system = dynamic_system
    cost = make_random_costs(system=system, seed=SEED)[0]
    strategy = make_lqr_strategy(system=system, cost=cost, player_idx=0, game_type=game_type)
    assert isinstance(strategy, LinearStrategy)
    assert strategy.K.shape == (system.ms[0], system.n)

@pytest.mark.parametrize("game_type", ["differential", "dynamic"])
def test_make_random_strategies(differential_system: LinearSystem, dynamic_system: LinearSystem, game_type: str) -> None:
    """Test random stabilizing strategies."""
    if game_type == "differential":
        system = differential_system
    else:
        system = dynamic_system
    strategies = make_random_strategies(system=system, game_type=game_type, seed=SEED)
    assert isinstance(strategies, list)
    assert len(strategies) == system.N
    assert all(isinstance(strategy, LinearStrategy) for strategy in strategies)
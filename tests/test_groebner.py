# tests/test_groebner.py

import numpy as np
import pytest

from src.gamecore import (
    LQGame, LinearStrategy, LinearSystem, QuadraticCost, LQPlayer,
    groebner_feedback_nash_equilibria
)


################################
# Shared fixtures
################################

@pytest.fixture
def A_stable():
    # Simple stable continuous-time A
    return np.array([[-1.0]], dtype=np.float64)

@pytest.fixture
def Bs_two_players():
    # Player 0: m0=1, Player 1: m1=1
    return [
        np.array([[1.0]], dtype=np.float64),
        np.array([[1.0]], dtype=np.float64),
    ]

@pytest.fixture
def system(A_stable: np.ndarray, Bs_two_players: list[np.ndarray]):
    return LinearSystem(A=A_stable, Bs=Bs_two_players)

@pytest.fixture
def strategies():
    # Gains integrate to a stable A_cl for continuous-time
    K0 = np.array([[0.5]], dtype=np.float64)  # (1x1)
    K1 = np.array([[0.5]], dtype=np.float64)  # (1x1)
    return [LinearStrategy(K0), LinearStrategy(K1)]

@pytest.fixture
def players(strategies: list[LinearStrategy]):
    Q = np.array([[2.0]], dtype=np.float64)
    R00 = np.array([[1.0]], dtype=np.float64)
    R11 = np.array([[1.0]], dtype=np.float64)
    R01 = np.array([[0.5]], dtype=np.float64)
    R10 = np.array([[0.5]], dtype=np.float64)
    p0 = LQPlayer(strategy=strategies[0], cost=QuadraticCost(Q=Q, R={(0,0): R00, (1,1): R11, (0,1): R01, (1,0): R10}), player_idx=0)
    p1 = LQPlayer(strategy=strategies[1], cost=QuadraticCost(Q=Q, R={(0,0): R00, (1,1): R11, (0,1): R01, (1,0): R10}), player_idx=1)
    return [p0, p1]

@pytest.fixture
def lqgame(system: LinearSystem, players: list[LQPlayer]):
    return LQGame(system=system, players=players, type="differential", Sigma0=np.eye(system.n))


def test_lqgame_feedback_nash_equilibria(lqgame: LQGame) -> None:
    """Test computation of feedback Nash strategies via Groebner approach.
    Is only run for very small systems due to high computational complexity."""

    solutions = groebner_feedback_nash_equilibria(lqgame)

    assert isinstance(solutions, list)
    assert len(solutions) == 1 # Expecting one unique Nash equilibrium for this simple case
    for strategy_set in solutions:
        assert isinstance(strategy_set, list)
        assert len(strategy_set) == lqgame.N
        for i, strategy in enumerate(strategy_set):
            assert isinstance(strategy.K, np.ndarray)
            assert strategy.K.shape == (lqgame.ms[i], lqgame.n)

        A_cl = lqgame.system.A.copy()
        for j, B_j in enumerate(lqgame.system.Bs):
            A_cl -= B_j @ strategy_set[j].K
        eigs = np.linalg.eigvals(A_cl)
        assert np.all(np.real(eigs) < 0), "Closed-loop system is not Hurwitz with Nash strategies"
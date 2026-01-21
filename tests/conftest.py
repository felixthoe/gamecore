# tests/conftest.py

import pytest
import numpy as np
import random

from src.gamecore import (
    LQGame,
    LinearSystem,
    LinearStrategy,
    QuadraticCost,
    LQPlayer,
    DataLogger
)

SEED = 42

####### Global Fixtures ########

@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(SEED)
    np.random.seed(SEED)

@pytest.fixture
def temp_logger(tmp_path):
    """Creates a temporary experiment logger."""
    logger = DataLogger(base_dir=tmp_path, folder_name="fixture_logger")
    yield logger

################################
# Fixtures: tiny stable LQ games
################################

@pytest.fixture
def tiny_lqgame_ct():
    """
    Small continuous-time LQ game:
    - n = 2 states
    - N = 2 players
    - m0 = m1 = 1
    - comfortably stable closed-loop
    """
    A = np.array([[-3.0,  0.0],
                  [ 0.0, -3.7]])
    B0 = np.array([[1.0],
                   [0.0]])
    B1 = np.array([[0.0],
                   [1.0]])

    system = LinearSystem(A=A, Bs=[B0, B1])

    K0 = np.array([[0.2, 0.0]])
    K1 = np.array([[0.0, 0.3]])

    Q = np.array([[2.0, 0.0],
                  [0.0, 1.0]])

    R00 = np.array([[1.0]])
    R11 = np.array([[1.0]])
    R01 = np.array([[0.01]])
    R10 = R01.T

    cost0 = QuadraticCost(Q, {
        (0, 0): R00,
        (1, 1): R11,
        (0, 1): R01,
        (1, 0): R10,
    })
    cost1 = QuadraticCost(Q, {
        (0, 0): R00,
        (1, 1): R11,
        (0, 1): R01,
        (1, 0): R10,
    })

    p0 = LQPlayer(
        strategy=LinearStrategy(K0),
        cost=cost0,
        player_idx=0,
        learning_rate=1.0,
    )
    p1 = LQPlayer(
        strategy=LinearStrategy(K1),
        cost=cost1,
        player_idx=1,
        learning_rate=1.0,
    )

    game = LQGame(
        system=system,
        players=[p0, p1],
        type="differential",
        Sigma0=np.eye(2),
    )
    return game

@pytest.fixture
def tiny_lqgame(tiny_lqgame_ct):
    """Alias for tiny continuous-time LQ game."""
    return tiny_lqgame_ct


@pytest.fixture
def tiny_lqgame_dt():
    """
    Small dynamic-time LQ game.
    Only shape and finiteness are tested here.
    """
    A = 0.1 * np.eye(2)
    B0 = np.array([[1.0],
                   [0.0]])
    B1 = np.array([[0.0],
                   [1.0]])

    system = LinearSystem(A=A, Bs=[B0, B1])

    K0 = np.array([[0.1, 0.0]])
    K1 = np.array([[0.0, 0.1]])

    Q = np.eye(2)
    R00 = np.array([[1.0]])
    R11 = np.array([[1.0]])
    R01 = np.array([[0.5]])
    R10 = R01.T

    cost0 = QuadraticCost(Q, {
        (0, 0): R00,
        (1, 1): R11,
        (0, 1): R01,
        (1, 0): R10,
    })
    cost1 = QuadraticCost(Q, {
        (0, 0): R00,
        (1, 1): R11,
        (0, 1): R01,
        (1, 0): R10,
    })

    p0 = LQPlayer(
        strategy=LinearStrategy(K0),
        cost=cost0,
        player_idx=0,
        learning_rate=1.0,
    )
    p1 = LQPlayer(
        strategy=LinearStrategy(K1),
        cost=cost1,
        player_idx=1,
        learning_rate=1.0,
    )

    game = LQGame(
        system=system,
        players=[p0, p1],
        type="dynamic",
        Sigma0=np.eye(2),
    )
    return game
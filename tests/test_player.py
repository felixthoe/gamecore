# tests/test_player.py

import numpy as np
import pytest

from src.gamecore import (
    BasePlayer, LQPlayer, LinearStrategy, QuadraticCost, SystemTrajectory, 
    LinearSystem, DataLogger
)


################################
# Fixtures
################################

@pytest.fixture
def A_2x2():
    return np.array([[0.1, 0.0],
                     [0.0, 0.2]], dtype=np.float64)

@pytest.fixture
def Bs_two_players_2x2():
    return [
        np.array([[1.0],
                  [0.0]], dtype=np.float64),  # player 0, m0=1
        np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=np.float64),  # player 1, m1=2
    ]

@pytest.fixture
def system(A_2x2, Bs_two_players_2x2):
    return LinearSystem(A=A_2x2, Bs=Bs_two_players_2x2)

@pytest.fixture
def strategies():
    K0 = np.array([[0.5, 0.0]], dtype=np.float64)          # (1x2)
    K1 = np.array([[0.1, -0.2],
                   [0.3,  0.4]], dtype=np.float64)         # (2x2)
    return [LinearStrategy(K0), LinearStrategy(K1)]

@pytest.fixture
def quadratic_cost():
    # Q for n=2, R[0] for m0=1, R[1] for m1=2
    Q = np.array([[2.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    R = {
        (0,0): np.array([[1.0]], dtype=np.float64),
        (1,1): np.array([[1.0, 0.0],
                     [0.0, 2.0]], dtype=np.float64),
    }
    return QuadraticCost(Q=Q, R=R)

@pytest.fixture
def linear_strategy():
    return LinearStrategy(np.array([[1.0, 0.0]], dtype=np.float64))


################################
# BasePlayer abstract behavior
################################

def test_baseplayer_is_abstract():
    with pytest.raises(TypeError):
        BasePlayer(strategy=None, cost=None, player_idx=0)


################################
# LQPlayer construction and attributes
################################

def test_lqplayer_constructs_with_linear_strategy(linear_strategy, quadratic_cost):
    p = LQPlayer(strategy=linear_strategy, cost=quadratic_cost, player_idx=1, learning_rate=0.5)
    assert p.player_idx == 1
    assert np.isclose(p.learning_rate, 0.5)
    assert p.m == linear_strategy.m

def test_lqplayer_invalid_types_raise(quadratic_cost):
    # invalid strategy type
    with pytest.raises(TypeError, match="requires a LinearStrategy"):
        LQPlayer(strategy=object(), cost=quadratic_cost, player_idx=0)
    # invalid cost type
    class DummyCost:
        pass
    with pytest.raises(TypeError, match="requires a QuadraticCost"):
        LQPlayer(strategy=LinearStrategy(np.array([[1.0, 0.0]])), cost=DummyCost(), player_idx=0)


################################
# M matrix delegation
################################

def test_M_delegates_to_cost(linear_strategy, quadratic_cost):
    p = LQPlayer(strategy=linear_strategy, cost=quadratic_cost, player_idx=0)
    # Two players example
    K0 = np.array([[1.0, 0.0]], dtype=np.float64)
    K1 = np.array([[0.5, -0.5],
                   [2.0,  1.0]], dtype=np.float64)
    strats = [LinearStrategy(K0), LinearStrategy(K1)]
    M_player = p.M(strats)
    expected = quadratic_cost.M(strats)
    assert np.allclose(M_player, expected)


################################
# strategy_cost using Lyapunov trace
################################

def test_strategy_cost_differential(system, strategies, quadratic_cost):
    p = LQPlayer(strategy=strategies[0], cost=quadratic_cost, player_idx=0)
    # Use identity Sigma0 by default
    val = p.strategy_cost(strategies=strategies, system=system, game_type="differential")
    assert np.isfinite(val)

def test_strategy_cost_dynamic(system, strategies, quadratic_cost):
    p = LQPlayer(strategy=strategies[0], cost=quadratic_cost, player_idx=0)
    val = p.strategy_cost(strategies=strategies, system=system, game_type="dynamic")
    assert np.isfinite(val)

def test_strategy_cost_sigma0_validation(system, strategies, quadratic_cost):
    p = LQPlayer(strategy=strategies[0], cost=quadratic_cost, player_idx=0)
    with pytest.raises(ValueError, match=r"Sigma0 must be a square matrix of shape \(2, 2\)"):
        _ = p.strategy_cost(strategies=strategies, system=system, Sigma0=np.eye(3))


################################
# lyapunov_matrix behavior and error cases
################################

def test_lyapunov_requires_system_or_Acl(strategies, quadratic_cost):
    p = LQPlayer(strategy=strategies[0], cost=quadratic_cost, player_idx=0)
    with pytest.raises(ValueError, match="Either system or A_cl must be provided"):
        _ = p.lyapunov_matrix(strategies=strategies)

def test_lyapunov_strategies_type_validation(system, quadratic_cost):
    p = LQPlayer(strategy=LinearStrategy(np.array([[1.0, 0.0]])), cost=quadratic_cost, player_idx=0)
    with pytest.raises(TypeError, match="All strategies must be instances of LinearStrategy"):
        _ = p.lyapunov_matrix(strategies=[object()], system=system)

def test_lyapunov_with_system(system, strategies, quadratic_cost):
    p = LQPlayer(strategy=strategies[0], cost=quadratic_cost, player_idx=0)
    P = p.lyapunov_matrix(strategies=strategies, system=system, game_type="differential")
    assert P.shape == (system.n, system.n)
    # Discrete time branch
    Pd = p.lyapunov_matrix(strategies=strategies, system=system, game_type="dynamic")
    assert Pd.shape == (system.n, system.n)

def test_lyapunov_with_provided_Acl(system, strategies, quadratic_cost):
    p = LQPlayer(strategy=strategies[0], cost=quadratic_cost, player_idx=0)
    Acl = system.A_cl(strategies)
    P = p.lyapunov_matrix(strategies=strategies, A_cl=Acl, game_type="differential")
    assert P.shape == (system.n, system.n)


################################
# system_trajectory_cost delegates to QuadraticCost
################################

def test_system_trajectory_cost_delegation(quadratic_cost):
    # Simple trajectory: T=2, n=2, u0 in R^1, u1 in R^2
    t = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    u0 = np.ones((2, 1), dtype=np.float64)
    u1 = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    traj = SystemTrajectory(t=t, x=x, us=[u0, u1], costs=None)

    p = LQPlayer(strategy=LinearStrategy(np.array([[1.0, 0.0]])), cost=quadratic_cost, player_idx=0)
    val = p.system_trajectory_cost(traj, game_type="differential")
    assert np.isfinite(val)


################################
# copy semantics
################################

def test_copy_is_deep(linear_strategy, quadratic_cost):
    p = LQPlayer(strategy=linear_strategy, cost=quadratic_cost, player_idx=2, learning_rate=0.3)
    cp = p.copy()
    assert cp is not p
    assert cp.player_idx == p.player_idx
    assert np.isclose(cp.learning_rate, p.learning_rate)
    # Strategy deep-copied
    cp.strategy.K[0, 0] += 10.0 if hasattr(cp.strategy, "K") else 0.0
    if hasattr(p.strategy, "K"):
        assert not np.allclose(cp.strategy.K, p.strategy.K)


################################
# Logging / Loading Round Trip
################################

def test_lq_player_log_and_load(temp_logger: DataLogger, linear_strategy: LinearStrategy, quadratic_cost: QuadraticCost):
    logger = temp_logger
    p = LQPlayer(strategy=linear_strategy, cost=quadratic_cost, player_idx=3, learning_rate=0.7)
    prefix = "player_"

    # Log
    p.log(logger, prefix=prefix)

    # Verify metadata keys saved by BasePlayer.log
    meta_type = logger.load_metadata_entry(f"{prefix}type")
    assert meta_type == "LQPlayer"
    assert int(logger.load_metadata_entry(f"{prefix}player_idx")) == 3
    assert float(logger.load_metadata_entry(f"{prefix}learning_rate")) == 0.7

    # Load via LQPlayer.load (classmethod) to ensure full round trip
    restored = LQPlayer.load(logger, prefix=prefix)
    assert isinstance(restored, LQPlayer)
    assert restored.player_idx == p.player_idx
    assert np.isclose(restored.learning_rate, p.learning_rate)
    # Strategy and cost contents
    assert np.allclose(restored.strategy.K, p.strategy.K)
    assert np.allclose(restored.cost.Q, p.cost.Q)
    for (j,k) in p.cost.R:
        assert np.allclose(restored.cost.R[(j,k)], p.cost.R[(j,k)])
# tests/test_cost.py

import numpy as np
import pytest

from src.gamecore import BaseCost, QuadraticCost, LinearStrategy, SystemTrajectory, DataLogger

################################
# Additional Fixtures
################################

@pytest.fixture
def system_trajectory():
    """
    Create a small, deterministic trajectory:
    - T=4 time steps: t = [0, 1, 2, 3]
    - State n=2: x_k = [k, 2k]
    - Two players: u0 in R^1, u1 in R^2
      u0_k = [1], u1_k = [k, -k]
    """
    T = 4
    t = np.arange(T, dtype=np.float64)
    x = np.stack([np.array([k, 2.0 * k], dtype=np.float64) for k in range(T)], axis=0)  # (T,2)
    u0 = np.ones((T, 1), dtype=np.float64)
    u1 = np.stack([np.array([k, -k], dtype=np.float64) for k in range(T)], axis=0)  # (T,2)
    us = [u0, u1]
    return SystemTrajectory(t=t, x=x, us=us, costs=None)


@pytest.fixture
def Q_good():
    # n=2 as in simple_trajectory
    return np.array([[2.0, 0.0],
                     [0.0, 1.0]], dtype=np.float64)


@pytest.fixture
def R_good():
    return {
        (0,0): np.array([[3.0]], dtype=np.float64),
        (1,1): np.array([[1.0, 0.0],
                     [0.0, 4.0]], dtype=np.float64),
        (0,1): np.array([[0.5, 0.5]], dtype=np.float64),
        (1,0): np.array([[0.5],
                         [0.5]], dtype=np.float64),
    }


################################
# BaseCost abstract behavior
################################

def test_basecost_is_abstract():
    with pytest.raises(TypeError):
        BaseCost()  # abstract methods prevent instantiation


################################
# Construction and validation
################################

def test_quadraticcost_construct_and_validate(Q_good: np.ndarray, R_good: dict[int, np.ndarray]):
    cost = QuadraticCost(Q=Q_good, R=R_good)
    assert isinstance(cost.Q, np.ndarray)
    assert all(isinstance(Rj, np.ndarray) for Rj in cost.R.values())
    assert cost.Q.shape == (2, 2)
    assert cost.R[(0,0)].shape == (1, 1) and cost.R[(1,1)].shape == (2, 2)
    assert cost.R[(0,1)].shape == (1, 2) and cost.R[(1,0)].shape == (2, 1)

def test_quadraticcost_invalid_Q_not_square():
    Q = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)  # 2x3
    with pytest.raises(ValueError, match="Q must be square"):
        QuadraticCost(Q=Q, R={})

def test_quadraticcost_invalid_R_not_square():
    Q = np.eye(2, dtype=np.float64)
    R = {(0,0): np.array([[1.0, 2.0]])}  # 1x2
    with pytest.raises(ValueError, match=r"must have the same shape"):
        QuadraticCost(Q=Q, R=R)


################################
# evaluate_system_trajectory
################################

def test_evaluate_differential_matches_expected(system_trajectory: SystemTrajectory, Q_good: np.ndarray, R_good: dict[int, np.ndarray]):
    cost = QuadraticCost(Q=Q_good, R=R_good)
    traj = system_trajectory
    # Manual expectation:
    # dt = 1.0 (t = [0,1,2,3])
    # state_cost_k = x_k^T Q x_k; with x_k=[k,2k], Q=diag(2,1)
    # -> = [k,2k]^T * [[2,0],[0,1]] * [k,2k] = 2k^2 + 4k^2 = 6k^2
    # control_cost_00 = u0_k^T R0 u0_k = [1]*3*[1] = 3
    # control_cost_11 = u1_k^T R1 u1_k, R1=diag(1,4), u1_k=[k,-k]
    # -> = k^2*1 + (-k)^2*4 = 5k^2
    # control cost_01 = u0_k^T R01 u1_k = [1]*[0.5,0.5]*[k,-k] = 0
    # control cost_10 = u1_k^T R10 u0_k = [k,-k]^T * [0.5;0.5] * [1] = 0
    # total control cost_k = 3 + 5k^2 + 0 + 0 = 5k^2 + 3
    # total cost_k = state_cost_k + control_cost_k = 6k^2 + (5k^2 + 3) = 11k^2 + 3
    # sum_k( state + control ) = sum_k(11k^2 + 3)
    # k=0..3 -> sum 11*(0+1+4+9) + 3*4 = 11*14 + 12 = 166
    # differential -> * dt (dt=1) = 166
    val = cost.evaluate_system_trajectory(traj, game_type="differential")
    assert np.isclose(val, 166.0)

def test_evaluate_dynamic_matches_expected(system_trajectory: SystemTrajectory, Q_good: np.ndarray, R_good: dict[int, np.ndarray]):
    cost = QuadraticCost(Q=Q_good, R=R_good)
    # dynamic: same sum without dt multiplication (but dt=1 here, so still 166)
    assert np.isclose(cost.evaluate_system_trajectory(system_trajectory, game_type="dynamic"), 166.0)

def test_evaluate_invalid_game_type_raises(system_trajectory: SystemTrajectory, Q_good: np.ndarray, R_good: dict[int, np.ndarray]):
    cost = QuadraticCost(Q=Q_good, R=R_good)
    with pytest.raises(ValueError, match="Game type must be either 'differential' or 'dynamic'"):
        _ = cost.evaluate_system_trajectory(system_trajectory, game_type="invalid")

def test_evaluate_wrong_Q_dimension_raises(system_trajectory: SystemTrajectory):
    traj = system_trajectory
    # Q wrong size (3x3 instead of 2x2)
    Q = np.eye(3, dtype=np.float64)
    cost = QuadraticCost(Q=Q, R={})
    with pytest.raises(ValueError, match="Q must match state dimension"):
        _ = cost.evaluate_system_trajectory(traj, game_type="dynamic")

def test_evaluate_wrong_R_dimension_raises(system_trajectory: SystemTrajectory, Q_good: np.ndarray):
    traj = system_trajectory
    # R[1] should be (2x2), provide (1x1) to trigger error
    R_bad = {(1,1): np.array([[1.0]], dtype=np.float64)}
    cost = QuadraticCost(Q=Q_good, R=R_bad)
    with pytest.raises(ValueError, match=r"must match input dimensions"):
        _ = cost.evaluate_system_trajectory(traj, game_type="dynamic")


################################
# M(strategies)
################################

def test_M_matrix_computation(Q_good: np.ndarray):
    # Two players: m0=1, m1=2; n=2
    K0 = np.array([[1.0, 0.0]], dtype=np.float64)          # (1x2)
    K1 = np.array([[0.5, -0.5],
                   [2.0,  1.0]], dtype=np.float64)         # (2x2)
    strategies = [LinearStrategy(K0), LinearStrategy(K1)]

    R = {
        (0,0): np.array([[3.0]], dtype=np.float64),            # (1x1)
        (1,1): np.array([[1.0, 0.0],
                         [0.0, 4.0]], dtype=np.float64),       # (2x2)
        (0,1): np.array([[0.5, 0.5]], dtype=np.float64),    # (1x2)
        (1,0): np.array([[0.5],
                         [0.5]], dtype=np.float64),         # (2x1)
    }
    cost = QuadraticCost(Q=Q_good, R=R)
    M = cost.M(strategies)

    # Expectation: M = Q + K0^T R00 K0 + K1^T R11 K1 + K0^T R01 K1 + K1^T R10 K0
    expected = Q_good.copy()
    expected += K0.T @ R[(0,0)] @ K0
    expected += K1.T @ R[(1,1)] @ K1
    expected += K0.T @ R[(0,1)] @ K1
    expected += K1.T @ R[(1,0)] @ K0
    assert np.allclose(M, expected)


################################
# Copy semantics
################################

def test_copy_is_deep(Q_good: np.ndarray, R_good: dict[int, np.ndarray]):
    cost = QuadraticCost(Q=Q_good, R=R_good)
    cp = cost.copy()
    assert cp is not cost
    assert np.allclose(cp.Q, cost.Q)
    for (j,k) in cost.R:
        assert np.allclose(cp.R[(j,k)], cost.R[(j,k)])

    # Mutating the copy must not affect the original
    cp.Q[0, 0] += 10.0
    cp.R[(0,0)][0, 0] += 5.0
    assert not np.allclose(cp.Q, cost.Q)
    assert not np.allclose(cp.R[(0,0)], cost.R[(0,0)])


################################
# Logging / Loading Round Trip
################################

def test_log_and_load_round_trip(temp_logger: DataLogger, Q_good: np.ndarray, R_good: dict[int, np.ndarray]):
    logger = temp_logger
    cost = QuadraticCost(Q=Q_good, R=R_good)
    prefix = "cost_"

    # Log
    cost.log(logger, prefix=prefix)

    # Verify metadata and arrays
    assert logger.load_metadata_entry(f"{prefix}type") == "QuadraticCost"
    Q_loaded = logger.load_array(f"{prefix}Q")
    assert np.allclose(Q_loaded, Q_good)
    R_keys = logger.load_metadata_entry(f"{prefix}R_keys")
    for loaded_key, orig_key in zip(R_keys, R_good.keys()):
        assert (loaded_key[0], loaded_key[1]) == (orig_key[0], orig_key[1])
    for (j,k) in R_keys:
        Rij_loaded = logger.load_array(f"{prefix}R_{j}{k}")
        assert np.allclose(Rij_loaded, R_good[(j,k)])

    # Load
    restored = QuadraticCost.load(logger, prefix=prefix)
    assert isinstance(restored, QuadraticCost)
    assert np.allclose(restored.Q, Q_good)
    for (j,k) in R_good:
        assert np.allclose(restored.R[(j,k)], R_good[(j,k)])
# tests/test_system.py

import numpy as np
import pytest

from src.gamecore import BaseSystem, LinearSystem, DataLogger, LinearStrategy


#############################
# Additional Fixtures
#############################

@pytest.fixture
def A_2x2():
    # Simple 2x2 system matrix
    return np.array([[1.0, 2.0],
                     [0.0, 1.0]], dtype=np.float64)

@pytest.fixture
def Bs_single_input_2x2():
    # One player, one input column
    # B shape: (n=2, m=1)
    return [np.array([[1.0],
                      [0.0]], dtype=np.float64)]

@pytest.fixture
def Bs_two_players_mixed_inputs_2x2():
    # Two players with mixed input dimensions:
    # B0: (2x1), B1: (2x2)
    return [
        np.array([[1.0],
                  [0.0]], dtype=np.float64),
        np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype=np.float64),
    ]

@pytest.fixture
def x_2():
    return np.array([0.5, -1.5], dtype=np.float64)

@pytest.fixture
def us_single():
    # For Bs_single_input_2x2: one control of shape (1,)
    return [np.array([2.0], dtype=np.float64)]

@pytest.fixture
def us_two_players():
    # For Bs_two_players_mixed_inputs_2x2: u0 in R^1, u1 in R^2
    return [
        np.array([2.0], dtype=np.float64),      # matches B0 (2x1)
        np.array([1.0, -1.0], dtype=np.float64) # matches B1 (2x2)
    ]


#############################
# BaseSystem abstract behavior
#############################

def test_basesystem_is_abstract():
    # Ensure BaseSystem cannot be instantiated directly
    with pytest.raises(TypeError):
        BaseSystem()  # abstract methods/properties prevent instantiation


#############################
# Construction and properties
#############################

def test_linearsystem_basic_attributes(A_2x2: np.ndarray, Bs_single_input_2x2: list[np.ndarray]):
    sys = LinearSystem(A=A_2x2, Bs=Bs_single_input_2x2)
    assert sys.n == 2
    assert sys.N == 1
    assert sys.ms == [1]
    # Ensure arrays are float64
    assert sys.A.dtype == np.float64
    assert all(B.dtype == np.float64 for B in sys.Bs)

def test_linearsystem_invalid_A_non_square():
    A = np.array([[1.0, 2.0, 3.0],
                  [0.0, 1.0, 4.0]], dtype=np.float64)  # 2x3
    Bs = [np.array([[1.0], [0.0]], dtype=np.float64)]
    with pytest.raises(ValueError, match="A must be square"):
        LinearSystem(A=A, Bs=Bs)

def test_linearsystem_invalid_B_row_mismatch(A_2x2: np.ndarray):
    # B has wrong number of rows
    Bs = [np.array([[1.0, 2.0, 3.0]], dtype=np.float64)]  # shape (1,3) but n=2
    with pytest.raises(ValueError, match=r"B_0 must have 2 rows\."):
        LinearSystem(A=A_2x2, Bs=Bs)


#############################
# Dynamics f(x, u)
#############################

def test_f_single_player(A_2x2: np.ndarray, Bs_single_input_2x2: list[np.ndarray], x_2: np.ndarray, us_single: list[np.ndarray]):
    sys = LinearSystem(A=A_2x2, Bs=Bs_single_input_2x2)
    f = sys.f(x_2, us_single)
    # Expected: A@x + B@u
    expected = A_2x2 @ x_2 + Bs_single_input_2x2[0] @ us_single[0]
    assert np.allclose(f, expected)

def test_f_two_players(A_2x2: np.ndarray, Bs_two_players_mixed_inputs_2x2: list[np.ndarray], x_2: np.ndarray, us_two_players: list[np.ndarray]):
    sys = LinearSystem(A=A_2x2, Bs=Bs_two_players_mixed_inputs_2x2)
    f = sys.f(x_2, us_two_players)
    expected = A_2x2 @ x_2 + sum(B @ u for B, u in zip(Bs_two_players_mixed_inputs_2x2, us_two_players))
    assert np.allclose(f, expected)

def test_f_input_mismatch(A_2x2: np.ndarray, Bs_two_players_mixed_inputs_2x2: list[np.ndarray], x_2: np.ndarray):
    """Shape of control input must match shape of B_i."""
    system = LinearSystem(A=A_2x2, Bs=Bs_two_players_mixed_inputs_2x2)
    us = [np.array([1.0], dtype=np.float64), np.array([999.0], dtype=np.float64)]
    with pytest.raises(ValueError):
        system.f(x_2, us)


#############################
# Closed-loop matrix A_cl
#############################

def test_A_cl_single_player(A_2x2: np.ndarray, Bs_single_input_2x2: list[np.ndarray]):
    sys = LinearSystem(A=A_2x2, Bs=Bs_single_input_2x2)
    # Strategy gain K must match input dimension (m) and state n: B (n x m) @ K (m x n) => (n x n)
    K = np.array([[2.0, -1.0]], dtype=np.float64)  # shape (1,2)
    strat = LinearStrategy(K=K)
    Acl = sys.A_cl([strat])
    expected = A_2x2 - Bs_single_input_2x2[0] @ K
    assert np.allclose(Acl, expected)

def test_A_cl_two_players(A_2x2: np.ndarray, Bs_two_players_mixed_inputs_2x2: list[np.ndarray]):
    sys = LinearSystem(A=A_2x2, Bs=Bs_two_players_mixed_inputs_2x2)
    K0 = np.array([[1.0, 0.0]], dtype=np.float64)        # (1x2)
    K1 = np.array([[0.5, -0.5],
                   [2.0,  1.0]], dtype=np.float64)       # (2x2)
    s0 = LinearStrategy(K=K0)
    s1 = LinearStrategy(K=K1)
    Acl = sys.A_cl([s0, s1])
    expected = A_2x2 - (Bs_two_players_mixed_inputs_2x2[0] @ K0 + Bs_two_players_mixed_inputs_2x2[1] @ K1)
    assert np.allclose(Acl, expected)


#############################
# Copy semantics
#############################

def test_copy_is_deep(A_2x2: np.ndarray, Bs_two_players_mixed_inputs_2x2: list[np.ndarray]):
    sys = LinearSystem(A=A_2x2, Bs=Bs_two_players_mixed_inputs_2x2)
    cp = sys.copy()
    # Identity check
    assert cp is not sys
    assert cp.n == sys.n and cp.N == sys.N and cp.ms == sys.ms
    # Values equal
    assert np.allclose(cp.A, sys.A)
    assert all(np.allclose(Bc, Bs) for Bc, Bs in zip(cp.Bs, sys.Bs))
    # Mutating copy should not change original
    cp.A[0, 0] += 10.0
    cp.Bs[0][0, 0] += 5.0
    assert not np.allclose(cp.A, sys.A)
    assert not np.allclose(cp.Bs[0], sys.Bs[0])



#############################
# Logging and loading (round trip)
#############################

def test_linear_system_log_and_load(
    temp_logger: DataLogger,
    A_2x2: np.ndarray, 
    Bs_two_players_mixed_inputs_2x2: list[np.ndarray]
):
    """Test logging and loading of LinearSystem."""
    sys = LinearSystem(A=A_2x2, Bs=Bs_two_players_mixed_inputs_2x2)
    # Log with a prefix for namespacing
    prefix = "sys_"
    sys.log(temp_logger, prefix=prefix)
    # Sanity: metadata keys present
    assert temp_logger.load_metadata_entry(f"{prefix}type") == "LinearSystem"
    assert int(temp_logger.load_metadata_entry(f"{prefix}N")) == sys.N
    assert temp_logger.load_metadata_entry(f"{prefix}n") == sys.n
    assert temp_logger.load_metadata_entry(f"{prefix}ms") == sys.ms

    # Arrays persisted
    A_logged = temp_logger.load_array(f"{prefix}A")
    assert np.allclose(A_logged, sys.A)
    for i in range(sys.N):
        B_logged = temp_logger.load_array(f"{prefix}Bs_{i}")
        assert np.allclose(B_logged, sys.Bs[i])

    # Now load a new system instance from logger
    loaded = LinearSystem.load(temp_logger, prefix=prefix)
    np.testing.assert_array_equal(loaded.A, sys.A)
    assert loaded.N == sys.N
    assert loaded.n == sys.n
    assert loaded.ms == sys.ms
    for B_loaded, B_orig in zip(loaded.Bs, sys.Bs):
        np.testing.assert_array_equal(B_loaded, B_orig)
# tests/test_solver.py

import numpy as np
import pytest

from src.gamecore import LinearSystem, QuadraticCost, LinearStrategy, LQPlayer, LQGame
from src.gamecore.solver import (
    feedback_nash_equilibrium,
    _policy_iteration,
    _care_value_iteration,
    _cdre_finite_horizon_simulation,
)


################################
# Fixtures: small LQ games
################################

@pytest.fixture
def lqgame_ct():
    """
    Small continuous-time LQ game:
    - n=2, N=2, m0=m1=1
    - A stable baseline with modest feedback
    """
    A = np.array([[-1.0, 0.0],
                  [ 0.0, -0.8]], dtype=np.float64)
    B0 = np.array([[1.0],
                   [0.0]], dtype=np.float64)
    B1 = np.array([[0.0],
                   [1.0]], dtype=np.float64)
    system = LinearSystem(A=A, Bs=[B0, B1])

    # Initial strategies
    K0 = np.array([[0.1, 0.0]], dtype=np.float64)
    K1 = np.array([[0.0, 0.0]], dtype=np.float64)

    # Costs: Q PSD and R_ii PD, symmetric
    Q = np.array([[2.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    R0 = np.array([[1.0]], dtype=np.float64)
    R1 = np.array([[1.0]], dtype=np.float64)

    p0 = LQPlayer(strategy=LinearStrategy(K0), cost=QuadraticCost(Q, {(0,0): R0, (1,1): R1}), player_idx=0)
    p1 = LQPlayer(strategy=LinearStrategy(K1), cost=QuadraticCost(Q, {(0,0): R0, (1,1): R1}), player_idx=1)

    return LQGame(system=system, players=[p0, p1], type="differential")

@pytest.fixture
def lqgame_dt():
    """
    Small discrete-time LQ game:
    - A with spectral radius < 1
    """
    A = 0.5 * np.eye(2, dtype=np.float64)
    B0 = np.array([[1.0],
                   [0.0]], dtype=np.float64)
    B1 = np.array([[0.0],
                   [1.0]], dtype=np.float64)
    system = LinearSystem(A=A, Bs=[B0, B1])

    K0 = np.array([[0.05, 0.0]], dtype=np.float64)
    K1 = np.array([[0.0, 0.05]], dtype=np.float64)

    Q = np.eye(2, dtype=np.float64)
    R0 = np.array([[1.0]], dtype=np.float64)
    R1 = np.array([[1.0]], dtype=np.float64)

    p0 = LQPlayer(strategy=LinearStrategy(K0), cost=QuadraticCost(Q, {(0,0): R0, (1,1): R1}), player_idx=0)
    p1 = LQPlayer(strategy=LinearStrategy(K1), cost=QuadraticCost(Q, {(0,0): R0, (1,1): R1}), player_idx=1)

    return LQGame(system=system, players=[p0, p1], type="dynamic", Sigma0=np.eye(2))


################################
# feedback_nash_equilibrium entry
################################

def test_feedback_nash_equilibrium_happy_path_ct(lqgame_ct: LQGame):
    # Should converge with cascade policy iteration on small CT game
    strats = feedback_nash_equilibrium(lqgame_ct)
    assert isinstance(strats, list) and len(strats) == lqgame_ct.N
    for s in strats:
        assert isinstance(s, LinearStrategy)
        assert s.K.shape == (1, lqgame_ct.n)

def test_feedback_nash_equilibrium_initial_strategies_validation(lqgame_ct: LQGame):
    # Wrong type
    with pytest.raises(TypeError, match="must be instances of LinearStrategy"):
        feedback_nash_equilibrium(lqgame_ct, initial_strategies=[object(), object()])
    # Wrong length
    with pytest.raises(ValueError, match="must match number of players"):
        feedback_nash_equilibrium(lqgame_ct, initial_strategies=[LinearStrategy(np.array([[0.1, 0.0]]))])

def test_feedback_nash_equilibrium_stability_requirement(lqgame_ct: LQGame):
    # Unstable initial strategies
    K0_unstable = np.array([[-5.0, 0.0]])
    K1_unstable = np.array([[0.0, -5.0]])
    unstable_strats = [LinearStrategy(K0_unstable), LinearStrategy(K1_unstable)]
    with pytest.raises(ValueError, match="Initial strategies are not stable"):
        feedback_nash_equilibrium(lqgame_ct, initial_strategies=unstable_strats)


################################
# _cascade_policy_iteration
################################

def test_policy_iteration_ct_converges(lqgame_ct: LQGame):
    strats = _policy_iteration(lqgame_ct, initial_strategies=[p.strategy for p in lqgame_ct.players])
    assert len(strats) == lqgame_ct.N
    for s in strats:
        assert isinstance(s, LinearStrategy)

def test_policy_iteration_dt_converges(lqgame_dt: LQGame):
    strats = _policy_iteration(lqgame_dt, initial_strategies=[p.strategy for p in lqgame_dt.players])
    assert len(strats) == lqgame_dt.N
    for s in strats:
        assert isinstance(s, LinearStrategy)

def test_policy_iteration_lyapunov_pd_precondition(lqgame_ct: LQGame, monkeypatch):
    # Make player's lyapunov_matrix return a matrix with non-positive eigenvalues to trigger the precondition error
    def fake_lyap(*args, **kwargs):
        return np.array([[0.0, 0.0], [0.0, -1.0]])
    monkeypatch.setattr(lqgame_ct.players[0], "lyapunov_matrix", lambda strategies, A_cl, game_type: fake_lyap(), raising=True)
    with pytest.raises(ValueError, match="First Lyapunov matrix not positive definite"):
        _policy_iteration(lqgame_ct, initial_strategies=[p.strategy for p in lqgame_ct.players])

def test_policy_iteration_runtime_error_on_nonconvergence(lqgame_ct: LQGame):
    # Simplest is to set max_iteration=0 -> triggers runtime error via else branch
    with pytest.raises(RuntimeError, match="did not converge"):
        _policy_iteration(lqgame_ct, initial_strategies=[p.strategy for p in lqgame_ct.players], max_iteration=0)


################################
# _care_value_iteration
################################

def test_care_value_iteration_ct_converges(lqgame_ct: LQGame):
    strats = _care_value_iteration(lqgame_ct, initial_strategies=[p.strategy for p in lqgame_ct.players])
    assert len(strats) == lqgame_ct.N
    for s in strats:
        assert isinstance(s, LinearStrategy)


################################
# _cdre_finite_horizon_simulation
################################

def test_cdre_finite_horizon_simulation_runs(lqgame_ct: LQGame):
    strats = _cdre_finite_horizon_simulation(lqgame_ct)
    assert isinstance(strats, list) and len(strats) == lqgame_ct.N
    for s in strats:
        assert isinstance(s, LinearStrategy)


################################
# feedback_nash_equilibrium fallback chain
################################

def test_feedback_nash_equilibrium_fallback_chain_ct(monkeypatch, lqgame_ct: LQGame):
    # Force cascade policy iteration to fail, then care to fail, then cdre to fail to reach final RuntimeError
    monkeypatch.setattr("src.gamecore.solver._policy_iteration", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail cascade")), raising=True)
    monkeypatch.setattr("src.gamecore.solver._care_value_iteration", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail care")), raising=True)
    monkeypatch.setattr("src.gamecore.solver._cdre_finite_horizon_simulation", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail cdre")), raising=True)
    with pytest.raises(RuntimeError, match="All available methods to compute feedback Nash strategies have failed"):
        feedback_nash_equilibrium(lqgame_ct)
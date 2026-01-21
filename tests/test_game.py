# tests/test_game.py

import numpy as np
import pytest

from src.gamecore import (
    LQGame, SystemTrajectory, LinearStrategy, DataLogger,
    LinearSystem, QuadraticCost, LQPlayer, BaseGame
)


################################
# Shared fixtures
################################

@pytest.fixture
def A_stable():
    # Simple stable continuous-time A (negative diagonal for continuous)
    return np.array([[-1.0, 0.0],
                     [0.0, -0.5]], dtype=np.float64)

@pytest.fixture
def Bs_two_players():
    # Player 0: m0=1, Player 1: m1=1 (simplify)
    return [
        np.array([[1.0],
                  [0.0]], dtype=np.float64),
        np.array([[0.0],
                  [1.0]], dtype=np.float64),
    ]

@pytest.fixture
def system(A_stable: np.ndarray, Bs_two_players: list[np.ndarray]):
    return LinearSystem(A=A_stable, Bs=Bs_two_players)

@pytest.fixture
def strategies():
    # Gains integrate to a stable A_cl for continuous-time
    K0 = np.array([[0.5, 0.0]], dtype=np.float64)  # (1x2)
    K1 = np.array([[0.0, 0.5]], dtype=np.float64)  # (1x2)
    return [LinearStrategy(K0), LinearStrategy(K1)]

@pytest.fixture
def players(strategies: list[LinearStrategy]):
    Q = np.array([[2.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    R0 = np.array([[1.0]], dtype=np.float64)
    R1 = np.array([[1.0]], dtype=np.float64)
    p0 = LQPlayer(strategy=strategies[0], cost=QuadraticCost(Q=Q, R={(0,0): R0, (1,1): R1}), player_idx=0)
    p1 = LQPlayer(strategy=strategies[1], cost=QuadraticCost(Q=Q, R={(0,0): R0, (1,1): R1}), player_idx=1)
    return [p0, p1]

@pytest.fixture
def game(system: LinearSystem, players: list[LQPlayer]):
    return BaseGame(system=system, players=players, type="differential")

@pytest.fixture
def lqgame(system: LinearSystem, players: list[LQPlayer]):
    return LQGame(system=system, players=players, type="differential", Sigma0=np.eye(system.n))


################################
# BaseGame construction and core properties
################################

def test_basegame_valid_construction(game: BaseGame):
    assert game.N == 2
    assert game.n == 2
    assert game.ms == [1, 1]
    assert len(game.players) == 2

def test_basegame_invalid_system_type(players: list[LQPlayer]):
    with pytest.raises(TypeError, match="System must be an instance of BaseSystem"):
        BaseGame(system=object(), players=players)

def test_basegame_mismatch_N(players: list[LQPlayer]):
    # Create system with different N than players count
    A = np.eye(2, dtype=np.float64)
    Bs = [np.array([[1.0], [0.0]])]  # N=1
    sys = LinearSystem(A=A, Bs=Bs)
    with pytest.raises(ValueError, match="Mismatch between system.N and number of players"):
        BaseGame(system=sys, players=players)

def test_basegame_mismatch_ms(system: LinearSystem):
    # Player with wrong m dimension
    wrong_strategy = LinearStrategy(np.array([[1.0, 0.0],
                                              [0.0, 1.0]]))  # m=2 (should be 1)
    Q = np.eye(system.n, dtype=np.float64)
    R0 = np.array([[1.0]], dtype=np.float64)
    R1 = np.array([[1.0]], dtype=np.float64)
    p0 = LQPlayer(strategy=wrong_strategy, cost=QuadraticCost(Q=Q, R={(0,0): R0, (1,1): R1}), player_idx=0)
    p1 = LQPlayer(strategy=LinearStrategy(np.array([[1.0, 0.0]])), cost=QuadraticCost(Q=Q, R={(0,0): R0, (1,1): R1}), player_idx=1)
    with pytest.raises(ValueError, match="Mismatch between system.ms and players' control dimensions"):
        BaseGame(system=system, players=[p0, p1])

def test_basegame_player_index_consistency(system: LinearSystem, strategies: list[LinearStrategy]):
    Q = np.eye(2, dtype=np.float64)
    R = {(0,0): np.array([[1.0]]), (1,1): np.array([[1.0]])}
    p0 = LQPlayer(strategy=strategies[0], cost=QuadraticCost(Q=Q, R=R), player_idx=1)  # wrong index
    p1 = LQPlayer(strategy=strategies[1], cost=QuadraticCost(Q=Q, R=R), player_idx=1)
    with pytest.raises(ValueError, match="Player index .* should coincide"):
        BaseGame(system=system, players=[p0, p1])

def test_basegame_type_validation(system: LinearSystem, players: list[LQPlayer]):
    with pytest.raises(ValueError, match="Game type must be either 'differential' or 'dynamic'"):
        BaseGame(system=system, players=players, type="invalid")

def test_basegame_accessors(game: BaseGame):
    # strategies and copies
    strats = game.strategies
    strats_cp = game.strategies_copy
    assert len(strats) == game.N and len(strats_cp) == game.N
    assert all(hasattr(s, "copy") for s in strats_cp)


################################
# simulate_system (differential/continuous-time system and dynamic/discrete-time system)
################################

def test_simulate_differential_basic(game: BaseGame):
    x0 = np.array([1.0, -1.0], dtype=np.float64)
    T = 1.0
    traj = game.simulate_system(x0=x0, T=T)
    assert isinstance(traj, SystemTrajectory)
    assert traj.x.shape[1] == game.n
    assert len(traj.us) == game.N
    for u in traj.us:
        assert u.shape[0] == traj.x.shape[0]

def test_simulate_dynamic_basic(system: LinearSystem, players: list[LQPlayer]):
    game = BaseGame(system=system, players=players, type="dynamic")
    x0 = np.array([1.0, -1.0], dtype=np.float64)
    steps = 5
    traj = game.simulate_system(x0=x0, T=steps)
    assert isinstance(traj, SystemTrajectory)
    assert traj.x.shape == (steps + 1, system.n)
    assert len(traj.us) == game.N
    for u in traj.us:
        assert u.shape == (steps + 1, u.shape[1])  # (steps+1, m_i)

def test_simulate_dynamic_T_validation(system: LinearSystem, players: list[LQPlayer]):
    game = BaseGame(system=system, players=players, type="dynamic")
    with pytest.raises(ValueError, match="T must be a positive integer"):
        _ = game.simulate_system(x0=np.ones(system.n), T=0)
    with pytest.raises(ValueError, match="T must be a positive integer"):
        _ = game.simulate_system(x0=np.ones(system.n), T=1.5)


################################
# adopt_strategies and reset
################################

def test_adopt_strategies(game: BaseGame):
    new_strategies = [LinearStrategy(np.array([[0.1, 0.0]])),
                      LinearStrategy(np.array([[0.0, 0.1]]))]
    game.adopt_strategies(new_strategies)
    assert all(np.allclose(p.strategy.K, s.K) for p, s in zip(game.players, new_strategies))

def test_adopt_strategies_validation(game: BaseGame):
    with pytest.raises(ValueError, match="Number of strategies must match number of players"):
        game.adopt_strategies([LinearStrategy(np.array([[1.0, 0.0]]))])
    with pytest.raises(TypeError, match="Strategy 0 must be an instance of BaseStrategy"):
        game.adopt_strategies([object(), LinearStrategy(np.array([[1.0, 0.0]]))])


################################
# LQGame-specific behavior
################################

def test_lqgame_valid_construction(lqgame: LQGame):
    assert isinstance(lqgame, LQGame)
    assert lqgame.N == 2
    assert lqgame.n == 2

def test_lqgame_Q_R_validations(system: LinearSystem, strategies: list[LQPlayer]):
    # Invalid Q (wrong shape)
    Q_bad = np.eye(3)
    Q = np.eye(2)
    with pytest.raises(ValueError, match="Q_0 shape mismatch"):
        LQGame(system=system, players=[LQPlayer(strategy=strategies[0], cost=QuadraticCost(Q_bad, {(0,0): np.array([[1.0]]), (1,1): np.array([[1.0]])}), player_idx=0),
                                      LQPlayer(strategy=strategies[1], cost=QuadraticCost(np.eye(2), {(0,0): np.array([[1.0]]), (1,1): np.array([[1.0]])}), player_idx=1)])
    # R_ii must be positive definite
    R_pd_fail = np.array([[0.0]])  # not PD
    with pytest.raises(ValueError, match=r"R_0,00 must be positive definite"):
        LQGame(system=system, players=[LQPlayer(strategy=strategies[0], cost=QuadraticCost(Q, {(0,0): R_pd_fail, (1,1): np.array([[1.0]])}), player_idx=0),
                                      LQPlayer(strategy=strategies[1], cost=QuadraticCost(Q, {(0,0): np.array([[1.0]]), (1,1): np.array([[1.0]])}), player_idx=1)])
    # Non-symmetric R
    # m1=2 now
    strategies[1] = LinearStrategy(np.array([[0.0, 0.0], [0.0, 0.0]]))  # m1=2
    system.Bs[1] = np.array([[1.0, 0.0], [0.0, 0.0]])  # m1=2
    R_bad = np.array([[0.0, 1.0], [0.0, 1.0]])  # not symmetric
    with pytest.raises(ValueError, match="must be the transpose"):
        LQGame(system=system, players=[LQPlayer(strategy=strategies[0], cost=QuadraticCost(Q, {(0,0): np.array([[1.0]]), (1,1): R_bad}), player_idx=0),
                                      LQPlayer(strategy=strategies[1], cost=QuadraticCost(Q, {(0,0): np.array([[1.0]]), (1,1): np.array([[1.0, 0.0], [0.0, 1.0]])}), player_idx=1)])
    
def test_lqgame_A_cl(strategies: list[LinearStrategy], lqgame: LQGame):
    Acl = lqgame.A_cl()  # uses current strategies
    expected = lqgame.system.A_cl(strategies)
    assert np.allclose(Acl, expected)
    # Validation for wrong strategy type/length
    with pytest.raises(ValueError, match="Number of strategies must match number of players"):
        _ = lqgame.A_cl([strategies[0]])
    with pytest.raises(TypeError, match="Strategy 0 must be an instance of LinearStrategy"):
        _ = lqgame.A_cl([object(), strategies[1]])

def test_lqgame_Ms_and_lyapunov_matrices(lqgame: LQGame):
    Ms = lqgame.Ms()
    assert len(Ms) == lqgame.N
    Ps = lqgame.lyapunov_matrices()
    assert len(Ps) == lqgame.N
    for P in Ps:
        assert P.shape == (lqgame.n, lqgame.n)

def test_lqgame_state_covariance(lqgame: LQGame):
    X = lqgame.state_covariance()
    assert X.shape == (lqgame.n, lqgame.n)

def test_lqgame_strategies_costs(lqgame: LQGame):
    costs = lqgame.strategies_costs()
    assert len(costs) == lqgame.N
    assert all(np.isfinite(c) for c in costs)

def test_lqgame_str_contains_sections(lqgame: LQGame):
    s = str(lqgame)
    assert "== LQGame ==" in s
    assert "Number of players:" in s
    assert "Game type:" in s
    assert "LinearSystem" in s
    assert "LQPlayer" in s


################################
# Logging / Loading Round Trip
################################

def test_log_and_load_round_trip(temp_logger: DataLogger, lqgame: LQGame):
    logger = temp_logger
    prefix = "game_"

    # Log using LQGame.log which calls BaseGame.log and then stores Sigma0
    lqgame.log(logger, prefix=prefix)

    # BaseGame.load should be able to read back types and components
    # It expects certain metadata keys for types; BaseGame.log writes only 'type', 'N', 'game_type'
    # The system and player log() must write their type metadata keys as "{prefix}*_type"
    # Validate presence of Sigma0
    Sigma0_loaded = logger.load_array(f"{prefix}Sigma0")
    assert np.allclose(Sigma0_loaded, lqgame.Sigma0)

    # Now load via BaseGame.load (it will dispatch to LQGame based on metadata)
    restored = BaseGame.load(logger, prefix=prefix)
    assert isinstance(restored, LQGame)
    assert restored.N == lqgame.N
    assert restored.type == lqgame.type
    # Compare system matrices
    assert np.allclose(restored.system.A, lqgame.system.A)
    for i in range(restored.N):
        assert np.allclose(restored.system.Bs[i], lqgame.system.Bs[i])
    # Compare player strategies and costs
    for rp, lp in zip(restored.players, lqgame.players):
        assert np.allclose(rp.strategy.K, lp.strategy.K)
        assert np.allclose(rp.cost.Q, lp.cost.Q)
        for j in lp.cost.R:
            assert np.allclose(rp.cost.R[j], lp.cost.R[j])
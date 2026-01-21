# src/gamecore/factories/strategy_factory.py

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from ..system.linear_system import LinearSystem
from ..cost.quadratic_cost import QuadraticCost
from ..strategy.linear_strategy import LinearStrategy
from .cost_factory import make_random_costs
from ..utils.utils import is_stabilizable

def make_random_strategies(
    system: LinearSystem,
    game_type: str = "differential",
    eps: float = 10.0,
    max_iter: int = 10000,
    seed: int | None = None,
) -> list[LinearStrategy]:
    """
    Generates randomized stabilizing strategies by perturbing LQR gains or zero gains.

    Parameters
    ----------
    system : LinearSystem
        The system on which the game is defined.

    game_type : str
        Type of the game, either "differential" or "dynamic".

    eps : float
        Initial magnitude of perturbation.

    max_iter : int
        Maximum number of attempts to find a stabilizing random strategy.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[LinearStrategy]
        List of stabilizing randomized linear strategies for each player.
    """
    rng = np.random.default_rng(seed)
    A = system.A
    Bs = system.Bs
    n = system.n
    ms = system.ms

    B_total = np.hstack(Bs)
    if not is_stabilizable(A, B_total, game_type=game_type):
        raise RuntimeError(f"Strategy Factory: System is not (jointly) stabilizable.")
    dummy_costs = make_random_costs(system=system, seed=seed)
    dummy_Ks = []
    for i, cost in enumerate(dummy_costs):
        if is_stabilizable(A, Bs[i], game_type=game_type):
            try: # may fail
                dummy_Ks.append(make_lqr_strategy(system=system, cost=cost, player_idx=i, game_type=game_type).K)
            except:
                dummy_Ks.append(np.zeros((ms[i], n)))
        else:
            dummy_Ks.append(np.zeros((ms[i], n)))
    strategies = _generate_stabilizing_strategies(system=system, dummy_Ks=dummy_Ks, game_type=game_type, eps=eps, max_iter=max_iter, rng=rng)
    if not strategies:
        # Some systems react poorly to LQR initialized gains, so we try again with purely zero initialized gains
        dummy_Ks = [np.zeros((m_i, n)) for m_i in ms]
        strategies = _generate_stabilizing_strategies(system=system, dummy_Ks=dummy_Ks, game_type=game_type, eps=eps, max_iter=max_iter, rng=rng)
        if not strategies:
            raise RuntimeError(f"Strategy Factory: Failed to find (jointly) stabilizing gains after {max_iter} attempts.")
    return strategies

def _generate_stabilizing_strategies(system: LinearSystem, dummy_Ks: list[np.ndarray], game_type: str, eps: int, max_iter: int, rng: np.random.Generator):
    eps_vals = np.linspace(0, eps, max_iter)[::-1]
    for k in range(max_iter):
        Ks_perturbed = [K + eps_vals[k] * rng.normal(size=K.shape) for K in dummy_Ks]
        F = system.A.copy()
        for B_i, K_i in zip(system.Bs, Ks_perturbed):
            F -= B_i @ K_i
        if game_type == "differential":
            if np.all(np.real(np.linalg.eigvals(F)) < 0):
                return [LinearStrategy(K_i) for K_i in Ks_perturbed]
        elif game_type == "dynamic":
            if np.all(np.abs(np.linalg.eigvals(F)) < 1.0):
                return [LinearStrategy(K_i) for K_i in Ks_perturbed]

def make_lqr_strategy(
    system: LinearSystem, 
    cost: QuadraticCost, 
    player_idx: int = 0,
    game_type: str = "differential"
) -> LinearStrategy:
    """
    Computes the optimal LQR feedback gain K for a given linear system and quadratic cost.
    This solves the continuous-time algebraic Riccati equation (CARE).

    The cost is assumed to be:
        J = ∫ (xᵀ Q x + uᵀ R u) dt

    Parameters
    ----------
    system : LinearSystem
        The linear system with dynamics dx/dt = A x + B u.
        System should be stabilizable for the specified player.
    cost : QuadraticCost
        The quadratic cost specification for the single player.
    player_idx : int
        The game index of the player, whos strategy is to be calculated.
    game_type : str
        Type of the game, either "differential" or "dynamic".

    Returns
    -------
    LinearStrategy
        The optimal linear feedback strategy with gain matrix K of shape (m, n),
        such that u = -K x minimizes the cost.
    """

    A = system.A
    B = system.Bs[player_idx]
    Q = cost.Q
    R = cost.R[(player_idx, player_idx)]

    if not is_stabilizable(A, B, game_type=game_type):
        raise RuntimeError(f"Strategy Factory: System is not stabilizable by player {player_idx} alone.")
    if Q.shape != (system.n, system.n):
        raise ValueError(f"Strategy Factory: Q must be a square matrix of shape ({system.n}, {system.n}).")
    if R.shape != (system.ms[player_idx], system.ms[player_idx]):
        raise ValueError(f"Strategy Factory: R must be a square matrix of shape ({system.ms[player_idx]}, {system.ms[player_idx]}).")

    # Solve the continuous-time algebraic Riccati equation and compute the optimal gain K
    if game_type == "differential":
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
    elif game_type == "dynamic":
        # Discrete-time algebraic Riccati equation
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    else:
        raise ValueError(f"Unknown game type: {game_type}")
    return LinearStrategy(K=K)
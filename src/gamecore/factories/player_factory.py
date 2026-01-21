# src/gamecore/factories/player_factory.py

from ..player.lq_player import LQPlayer
from ..system.linear_system import LinearSystem
from .cost_factory import make_random_costs
from .strategy_factory import make_random_strategies

def make_random_lq_players(
    system: LinearSystem,
    game_type: str = "differential",
    learning_rate: float | list[float] = 1.0,
    cost_q_def: str = "pd",
    cost_r_jj: str = "free",
    cost_r_jk: str = "zero",
    cost_sparsity: float = 0.0,
    cost_amplitude: float = 10.0,
    cost_diag: bool = True,
    strategy_eps: float = 10.0,
    strategy_max_iter: int = 10000,
    seed: int | None = None,
) -> list[LQPlayer]:
    """
    Creates a list of LQPlayers with random (individually detectable) costs and random stabilizing strategies.

    Parameters
    ----------
    system : LinearSystem
        The system on which the game is defined.
    game_type : str
        Type of the game, either "differential" or "dynamic".
    learning_rate : float | list[float]
        Learning rate for strategy updates. A single value will be broadcasted to all players,
        a list will be distributed per player. The list has to match the number of players in the system.
    cost_q_def : str
        Definiteness of the Q matrices. Either "pd" (positive definite) or "psd" (positive semi-definite).
    cost_r_jj : str
        Constraints on the R_i,jj matrices for j ≠ i. Either "zero" for zero matrices,
        "psd" for positive semidefinite, or "free" for arbitrary matrices. 
    cost_r_jk : str
        Constraints on the R_i,jk matrices for j ≠ k. Either "zero" for zero matrices,
        or "free" for arbitrary matrices. 
        "free" requires cost_r_jj in {"free", "psd"} to be valid.
    cost_sparsity : float
        Fraction of zero entries to introduce in the cost matrices.
    cost_amplitude : float
        Amplitude for the random entries in the cost matrices.
    cost_diag : bool
        Structure of the Q and R matrices. If True, use diagonal matrices,
        if False, use full matrices.
    strategy_eps : float
        Magnitude of initial perturbations in random strategies.
    strategy_max_iter : int
        Maximum attempts to find a stabilizing random strategy.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[LQPlayer]
        A list of randomized players with consistent costs and strategies.
    """
    # Determine learning rates per player
    if isinstance(learning_rate, (float, int)):
        learning_rate = [learning_rate for _ in range(system.N)]
    elif isinstance(learning_rate, list):
        if len(learning_rate) != system.N:
            raise ValueError(f"Number of learning rates ({len(learning_rate)}) has to match the number of players in the system ({system.N}) if given as a list.")
    else:
        raise ValueError(f"Invalid type of argument `learning_rate`. Has to be float|int or list[float|int]. Got: {type(learning_rate)}.")
    
    # Generate random stabilizing strategies
    strategies = make_random_strategies(
        system=system,
        game_type=game_type,
        eps=strategy_eps,
        max_iter=strategy_max_iter,
        seed=seed
    )

    # Generate random costs
    costs = make_random_costs(
        system=system,
        q_def=cost_q_def,
        r_jj=cost_r_jj,
        r_jk=cost_r_jk,
        sparsity=cost_sparsity,
        amplitude=cost_amplitude,
        diag=cost_diag,
        seed=seed
    )

    return [LQPlayer(strategy=strat, cost=ci, player_idx=i, learning_rate=learning_rate[i]) for i, (strat, ci) in enumerate(zip(strategies, costs))]
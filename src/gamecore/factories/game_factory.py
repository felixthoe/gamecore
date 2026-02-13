# src/gamecore/factories/game_factory.py

from ..game.lq_game import LQGame
from .system_factory import make_random_system
from .player_factory import make_random_lq_players

def make_random_lq_game(
    n: int = 2,
    ms: list[int] = [1, 1],
    game_type: str = "differential",
    learning_rate: float | list[float] = 1.0,
    system_stabilizability: str = "individual",
    system_sparsity: float = 0.0,
    system_amplitude: float = 1.0,
    system_max_iter: int = 10000,
    cost_q_i_def: str = "pd",
    cost_r_ijj: str = "free",
    cost_r_ijk: str = "zero",
    cost_enforce_psd_r_i: bool = True,
    cost_sparsity: float = 0.0,
    cost_amplitude: float = 10.0,
    cost_diag: bool = True,
    strategy_eps: float = 5.0,
    strategy_max_iter: int = 10000,
    seed: int | None = None,
) -> LQGame:
    """
    Generates a fully random but stabilizable LQ game with randomized system dynamics, cost structures,
    and feedback strategies. The game is constructed such that the optimization problem of each
    player (with other strategies fixed) is well-defined.

    Parameters
    ----------
    n : int
        State dimension.
    ms : list[int]
        Control dimensions per player.
    game_type : str
        Type of the game, either "differential" or "dynamic".
    learning_rate : float | list[float]
        Learning rate for strategy updates. A single value will be broadcasted to all players,
        a list will be distributed per player. The list has to match the number of players in the system.
    system_stabilizability : str
        Stabilizability assumption for the system. Either "individual" (each (A, B_i) is stabilizable)
        or "joint" (the overall (A, [B_1 ... B_N]) is stabilizable).
    system_sparsity: float
        Fraction of zero entries to introduce in the system matrices.
    system_amplitude: float
        Amplitude for the random entries in the system matrices.
    system_max_iter : int
        Maximum attempts to find a stabilizable system.
    cost_q_i_def : str
        Definiteness of the Q matrices. Either "pd" (positive definite) or "psd" (positive semi-definite).
    cost_r_ijj : str
        Constraints on the R_i,jj matrices for j ≠ i. Either "zero" for zero matrices,
        "psd" for positive semidefinite, or "free" for arbitrary matrices. 
    cost_r_ijk : str
        Constraints on the R_i,jk matrices for j ≠ k. Either "zero" for zero matrices,
        or "free" for arbitrary matrices.
    cost_enforce_psd_r_i : bool
        If True, adjust the generated R_i matrices to ensure they are positive semidefinite.
    cost_sparsity: float
        Fraction of zero entries to introduce in the cost matrices.
    cost_amplitude: float
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
    LQGame
        A fully initialized LQ game with random parameters.
    """
    if game_type not in ["differential", "dynamic"]:
        raise ValueError(f"Game Factory: Invalid game type '{game_type}'. Has to be either 'differential' or 'dynamic'.")
    
    # Generate random stabilizable system
    system = make_random_system(
        n=n, 
        ms=ms, 
        stabilizability=system_stabilizability,
        game_type=game_type,
        sparsity=system_sparsity,
        amplitude=system_amplitude,
        max_iter=system_max_iter,
        seed=seed,
    )

    # Generate random players
    players = make_random_lq_players(
        system=system,
        game_type=game_type,
        learning_rate=learning_rate,
        cost_q_i_def=cost_q_i_def,
        cost_r_ijj=cost_r_ijj,
        cost_r_ijk=cost_r_ijk,
        cost_enforce_psd_r_i=cost_enforce_psd_r_i,
        cost_sparsity=cost_sparsity,
        cost_amplitude=cost_amplitude,
        cost_diag=cost_diag,
        strategy_eps=strategy_eps,
        strategy_max_iter=strategy_max_iter,
        seed=seed,
    )
    
    return LQGame(system=system, players=players, type=game_type)

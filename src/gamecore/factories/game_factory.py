# src/gamecore/factories/game_factory.py

from typing import Literal

from ..game.lq_game import LQGame
from ..time_domain import TimeDomain, resolve_time_domain
from .system_factory import make_random_system
from .player_factory import make_random_lq_players

def make_random_lq_game(
    n: int = 2,
    ms: list[int] = [1, 1],
    time_domain: str | TimeDomain = "continuous",
    learning_rate: float | list[float] = 1.0,
    system_stabilizability: str = "joint",
    system_sparsity: float = 0.0,
    system_amplitude_A: float | None = None,
    system_amplitude_B: float = 1.0,
    system_max_iter: int = 10000,
    system_target_spectral_radius: float | None = None,
    system_target_input_norm: float | None = None,
    cost_q_i: str = "pd",
    cost_r_ijj: str = "free",
    cost_r_ijk: str = "zero",
    cost_enforce_psd_r_i: bool = True,
    cost_sparsity: float = 0.0,
    cost_amplitude: float = 10.0,
    cost_diag: bool = True,
    strategy_init: Literal["joint_lqr", "random_pole_placement", "random_bisection"] = "random_bisection",
    strategy_amplitude: float = 1.0,
    strategy_pole_scale: float = 1.0,
    strategy_bisection_scale: float = 1.0,
    strategy_margin: float = 0.05,
    seed: int | None = None,
) -> LQGame:
    """
    Generates a fully random but stabilizable LQ game with randomized system dynamics, cost structures,
    and stable feedback strategies. The game is constructed such that the optimization problem of each
    player (with other strategies fixed) is well-defined.

    Parameters
    ----------
    n : int
        State dimension.
    ms : list[int]
        Control dimensions per player.
    time_domain : str | TimeDomain
        Whether the game evolves in continuous or discrete time.
    learning_rate : float | list[float]
        Learning rate for strategy updates. A single value will be broadcasted to all players,
        a list will be distributed per player. The list has to match the number of players in the system.
    system_stabilizability : str
        Stabilizability assumption for the system. Either "individual" (each (A, B_i) is stabilizable)
        or "joint" (the overall (A, [B_1 ... B_N]) is stabilizable).
    system_sparsity: float
        Fraction of zero entries to introduce in the system matrices.
    system_amplitude_A: float | None
        Amplitude for the random entries in the system matrix A. If None, scales by sqrt(n).
        Will be without effect if system_target_spectral_radius is given.
    system_amplitude_B: float | None
        Amplitude for the random entries in the input matrices B_i. Default is 1.
        Will be without effect if system_input_norm is given.
    system_max_iter : int
        Maximum attempts to find a stabilizable system.
    system_target_spectral_radius : float, optional
        If given, rescale the system's A so its spectral radius equals this value; see `make_random_system`.
    system_target_input_norm : float, optional
        If given, rescale each B_i so its operator norm equals this value; see `make_random_system`.
    cost_q_i : str
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
        Fraction of zero entries to introduce in the cost matrices. Not applied to those defined positive definite.
    cost_amplitude: float
        Amplitude for the random entries in the cost matrices.
    cost_diag : bool
        Structure of the Q and R matrices. If True, use diagonal matrices,
        if False, use full matrices.
    strategy_init : {"joint_lqr", "random_pole_placement", "random_bisection"}
        How to generate the initial gains; see `make_random_strategies`.
    strategy_amplitude : float
        Dummy cost scale, used by strategy_init="joint_lqr".
    strategy_pole_scale : float
        Target-spectrum scale, used by strategy_init="random_pole_placement".
    strategy_bisection_scale : float
        Search-direction scale, used by strategy_init="random_bisection".
    strategy_margin : float
        Minimum required stability margin of the resulting closed loop; see `make_random_strategies`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    LQGame
        A fully initialized LQ game with random parameters.
    """
    time_domain = resolve_time_domain(time_domain)

    # Generate random stabilizable system
    system = make_random_system(
        n=n,
        ms=ms,
        stabilizability=system_stabilizability,
        time_domain=time_domain,
        sparsity=system_sparsity,
        amplitude_A=system_amplitude_A,
        amplitude_B=system_amplitude_B,
        max_iter=system_max_iter,
        target_spectral_radius=system_target_spectral_radius,
        target_input_norm=system_target_input_norm,
        seed=seed,
    )

    # Generate random players
    players = make_random_lq_players(
        system=system,
        time_domain=time_domain,
        learning_rate=learning_rate,
        cost_q_i=cost_q_i,
        cost_r_ijj=cost_r_ijj,
        cost_r_ijk=cost_r_ijk,
        cost_enforce_psd_r_i=cost_enforce_psd_r_i,
        cost_sparsity=cost_sparsity,
        cost_amplitude=cost_amplitude,
        cost_diag=cost_diag,
        strategy_init=strategy_init,
        strategy_amplitude=strategy_amplitude,
        strategy_pole_scale=strategy_pole_scale,
        strategy_bisection_scale=strategy_bisection_scale,
        strategy_margin=strategy_margin,
        seed=seed,
    )

    # Strategies are already certified stabilizing by make_random_lq_players
    return LQGame(system=system, players=players, time_domain=time_domain, check_stability=False)

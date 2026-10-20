# src/gamecore/factories/player_factory.py

from typing import Literal

from ..player.lq_player import LQPlayer
from ..system.linear_system import LinearSystem
from ..time_domain import TimeDomain
from .cost_factory import make_random_costs
from .strategy_factory import make_random_strategies

def make_random_lq_players(
    system: LinearSystem,
    time_domain: str | TimeDomain = "continuous",
    learning_rate: float | list[float] = 1.0,
    cost_q_i: str = "pd",
    cost_cross_terms: bool = False,
    cost_sparsity: float = 0.0,
    cost_amplitude: float = 10.0,
    cost_diag: bool = True,
    strategy_init: Literal["joint_lqr", "random_pole_placement", "random_bisection"] = "random_bisection",
    strategy_amplitude: float = 1.0,
    strategy_pole_scale: float = 1.0,
    strategy_bisection_scale: float = 1.0,
    strategy_margin: float = 0.05,
    seed: int | None = None,
) -> list[LQPlayer]:
    """
    Creates a list of LQPlayers with random (individually detectable) costs and random stabilizing strategies.

    Parameters
    ----------
    system : LinearSystem
        The system on which the game is defined.
    time_domain : str | TimeDomain
        Whether the game evolves in continuous or discrete time.
    learning_rate : float | list[float]
        Learning rate for strategy updates. A single value will be broadcasted to all players,
        a list will be distributed per player. The list has to match the number of players in the system.
    cost_q_i : str
        Definiteness of the Q matrices. Either "pd" (positive definite) or "psd" (positive semi-definite).
    cost_cross_terms : bool
        Whether the control cost matrices R_i have cross terms R_i,jk (j != k); see `make_random_costs`.
    cost_sparsity : float
        Fraction of zero entries to introduce in the cost matrices. Not applied to those defined positive definite.
    cost_amplitude : float
        Amplitude for the random entries in the cost matrices.
    cost_diag : bool
        Structure of the Q and R matrices. If True, use diagonal matrices,
        if False, use full matrices.
    strategy_init : {"joint_lqr", "random_pole_placement", "random_bisection"}
        How to generate the initial gains; see `make_random_strategies`.
    strategy_amplitude : float
        Dummy cost scale, used by strategy_init="joint_lqr".
    strategy_pole_scale : float
        Target-spectrum scale, used by strategy_init="random_pole_placement" if time_domain is continuous.
    strategy_bisection_scale : float
        Search-direction scale, used by strategy_init="random_bisection".
    strategy_margin : float
        Minimum required stability margin of the resulting closed loop; see `make_random_strategies`.
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
        time_domain=time_domain,
        strategy_init=strategy_init,
        amplitude=strategy_amplitude,
        pole_scale=strategy_pole_scale,
        bisection_scale=strategy_bisection_scale,
        margin=strategy_margin,
        seed=seed,
    )

    # Generate random costs
    costs = make_random_costs(
        system=system,
        time_domain=time_domain,
        q_i=cost_q_i,
        cross_terms=cost_cross_terms,
        sparsity=cost_sparsity,
        amplitude=cost_amplitude,
        diag=cost_diag,
        seed=seed,
    )

    return [LQPlayer(strategy=strat, cost=ci, player_idx=i, learning_rate=learning_rate[i]) for i, (strat, ci) in enumerate(zip(strategies, costs))]

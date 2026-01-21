# src/gamecore/factories/system_factory.py

import numpy as np

from ..system.linear_system import LinearSystem
from ..utils.utils import is_stabilizable, sparsify

def make_random_system(
    n: int,
    ms: list[int],
    stabilizability: str = "individual",
    game_type: str = "differential",
    sparsity: float = 0.0,
    amplitude: float = 1.0,
    max_iter: int = 10000,
    seed: int | None = None,
) -> LinearSystem:
    """
    Generates a random stabilizable linear system.

    Parameters
    ----------
    n : int
        State dimension.

    ms : list[int]
        Control dimensions per player.

    stabilizability : str
        Stabilizability assumption. Either "individual" (each (A, B_i) is stabilizable)
        or "joint" (the overall (A, [B_1 ... B_N]) is stabilizable).

    game_type : str
        Type of the game, either "differential" or "dynamic".

    sparsity : float
        Fraction of zero elements to introduce in A and B matrices.

    amplitude : float
        Amplitude for the random entries in A and B matrices.

    max_iter : int
        Maximum number of attempts per player or jointly.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    LinearSystem
        Stabilizable system.

    Raises
    ------
    RuntimeError
        If no stabilizable system is found.
    """
    rng = np.random.default_rng(seed)

    for _ in range(max_iter):
        A = sparsify(amplitude*rng.standard_normal((n, n)), sparsity, rng)
        Bs = [sparsify(amplitude*rng.standard_normal((n, m)), sparsity, rng) for m in ms]

        if stabilizability == "joint":
            B_total = np.hstack(Bs)
            if is_stabilizable(A, B_total, game_type=game_type):
                return LinearSystem(A=A, Bs=Bs)
        elif stabilizability == "individual":
            if all(is_stabilizable(A, B, game_type=game_type) for B in Bs):
                return LinearSystem(A=A, Bs=Bs)
        else:
            raise ValueError(f"System Factory: Unknown mode for stabilizability: '{stabilizability}'. Use 'joint' or 'individual'.")

    raise RuntimeError(f"System Factory: Failed to generate a stabilizable system for '{stabilizability}' stabilizability after {max_iter} trials.")
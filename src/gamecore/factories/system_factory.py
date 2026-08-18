# src/gamecore/factories/system_factory.py

import numpy as np

from ..system.linear_system import LinearSystem
from ..time_domain import TimeDomain, resolve_time_domain
from ..utils.utils import controls_eigenvalue, sparsify, unstable_eigenvalues


def _apply_target_scaling(A: np.ndarray, Bs: list[np.ndarray], target_spectral_radius: float | None, target_input_norm: float | None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rescale A to hit `target_spectral_radius` and each B_i to hit `target_input_norm`, if given."""
    if target_spectral_radius is not None:
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        if rho > 1e-12:
            A = A * (target_spectral_radius / rho)
    if target_input_norm is not None:
        Bs = [B * (target_input_norm / max(np.linalg.norm(B, ord=2), 1e-12)) for B in Bs]
    return A, Bs


def make_random_system(
    n: int,
    ms: list[int],
    stabilizability: str = "joint",
    time_domain: str | TimeDomain = "continuous",
    sparsity: float = 0.0,
    amplitude_A: float | None = None,
    amplitude_B: float = 1.0,
    max_iter: int = 10000,
    target_spectral_radius: float | None = None,
    target_input_norm: float | None = None,
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

    time_domain : str | TimeDomain
        Whether the game evolves in continuous or discrete time.

    sparsity : float
        Fraction of zero elements to introduce in A and B matrices.

    amplitude_A : float | None
        Amplitude for the random entries in the A matrix. If None,
        scales by 1/sqrt(n). Will be without effect if system_target_spectral_radius is given.

    amplitude_B : float        
        Amplitude for the random entries in B matrices. Default 1.
        Will be without effect if system_input_norm is given.

    max_iter : int
        Maximum number of attempts per player or jointly.

    target_spectral_radius : float, optional
        If given, rescale each candidate A so its spectral radius equals this value exactly,
        before the stabilizability check. Keeps A's spectral scale independent of n and
        sparsity, rather than growing with them under plain i.i.d. sampling. Default is None.

    target_input_norm : float, optional
        If given, rescale each candidate B_i so its operator norm equals this value exactly.
        Default is None (no rescaling).

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
    time_domain = resolve_time_domain(time_domain)
    rng = np.random.default_rng(seed)
    if amplitude_A is None:
        amplitude_A = 1.0 / np.sqrt(n)

    for _ in range(max_iter):
        A = sparsify(amplitude_A*rng.standard_normal((n, n)), sparsity, rng)
        Bs = [sparsify(amplitude_B*rng.standard_normal((n, m)), sparsity, rng) for m in ms]
        A, Bs = _apply_target_scaling(A, Bs, target_spectral_radius, target_input_norm)

        if stabilizability == "joint":
            B_total = np.hstack(Bs)
            if all(controls_eigenvalue(A, lam, B_total) for lam in unstable_eigenvalues(A, time_domain)):
                return LinearSystem(A=A, Bs=Bs)
        elif stabilizability == "individual":
            eigs = unstable_eigenvalues(A, time_domain)
            if all(all(controls_eigenvalue(A, lam, B) for lam in eigs) for B in Bs):
                return LinearSystem(A=A, Bs=Bs)
        else:
            raise ValueError(f"System Factory: Unknown mode for stabilizability: '{stabilizability}'. Use 'joint' or 'individual'.")

    raise RuntimeError(f"System Factory: Failed to generate a stabilizable system for '{stabilizability}' stabilizability after {max_iter} trials.")

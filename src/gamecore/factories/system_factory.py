# src/gamecore/factories/system_factory.py

import numpy as np

from ..system.linear_system import LinearSystem
from ..time_domain import TimeDomain, resolve_time_domain
from ..utils.utils import FactorySamplingError, controls_eigenvalue, sparsify, unstable_eigenvalues


def _apply_target_scaling(A: np.ndarray, Bs: list[np.ndarray], target_spectral_radius: float | None, target_input_norm: float | None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rescale A to hit `target_spectral_radius` and each B_i to hit `target_input_norm`, if given."""
    if target_spectral_radius is not None:
        rho = np.max(np.abs(np.linalg.eigvals(A)))
        if rho > 1e-12:
            A = A * (target_spectral_radius / rho)
    if target_input_norm is not None:
        Bs = [B * (target_input_norm / max(np.linalg.norm(B, ord=2), 1e-12)) for B in Bs]
    return A, Bs


def _sparse_full_rank_input(n: int, m: int, sparsity: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Random sparse (n, m) input matrix whose nonzero pattern contains a random transversal of
    length min(n, m), so it has full rank min(n, m) almost surely despite `sparsity`. Rejection
    sampling for full rank is infeasible for large m at high sparsity.
    """
    mask = rng.random((n, m)) > sparsity
    k = min(n, m)
    mask[rng.permutation(n)[:k], rng.permutation(m)[:k]] = True
    return amplitude * rng.standard_normal((n, m)) * mask


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
    Generates a random stabilizable linear system in which every B_i has full rank
    min(n, m_i), i.e. no player has redundant input channels (e.g. zero columns from `sparsity`).

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
        Fraction of zero elements to introduce in A and B matrices (for B_i up to the entries
        that keep it at full rank).

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
    FactorySamplingError
        If no stabilizable system with full-rank B_i is found within `max_iter` attempts.
    """
    time_domain = resolve_time_domain(time_domain)
    rng = np.random.default_rng(seed)
    if amplitude_A is None:
        amplitude_A = 1.0 / np.sqrt(n)

    for _ in range(max_iter):
        A = sparsify(amplitude_A*rng.standard_normal((n, n)), sparsity, rng)
        Bs = [_sparse_full_rank_input(n, m, sparsity, amplitude_B, rng) for m in ms]
        A, Bs = _apply_target_scaling(A, Bs, target_spectral_radius, target_input_norm)
        if any(np.linalg.matrix_rank(B) < min(B.shape) for B in Bs):
            continue

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

    raise FactorySamplingError(f"System Factory: Failed to generate a stabilizable system with full-rank B_i for '{stabilizability}' stabilizability after {max_iter} trials.")

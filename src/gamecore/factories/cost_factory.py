# src/gamecore/factories/cost_factory.py

import numpy as np

from ..cost.quadratic_cost import QuadraticCost
from ..system.linear_system import LinearSystem
from ..time_domain import TimeDomain, resolve_time_domain
from ..utils.utils import (
    random_PSD_matrix,
    random_PD_matrix,
    is_detectable,
    FactorySamplingError,
)


def _psd_power(M: np.ndarray, p: float) -> np.ndarray:
    """Power M^p of a symmetric positive semidefinite matrix via its eigendecomposition."""
    w, V = np.linalg.eigh(M)
    return (V * np.clip(w, 0.0, None) ** p) @ V.T


def _cross_terms(blocks: list[np.ndarray], rng: np.random.Generator) -> dict[tuple[int, int], np.ndarray]:
    """
    Off-diagonal blocks (j, k), j != k, of R = B C B with B = blockdiag(blocks[j]^(1/2)) and C the
    block correlation matrix of a random Wishart matrix W = G G^T, C_jk = W_jj^(-1/2) W_jk
    W_kk^(-1/2). R is positive semidefinite and has the diagonal blocks `blocks`.

    Parameters
    ----------
    blocks : list[np.ndarray]
        Positive semidefinite diagonal blocks R_jj.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        The blocks R_jk for j != k.
    """
    offsets = np.cumsum([0] + [block.shape[0] for block in blocks])
    G = rng.standard_normal((offsets[-1], offsets[-1]))
    W = G @ G.T
    part = lambda j, k: W[offsets[j]:offsets[j + 1], offsets[k]:offsets[k + 1]]
    roots = [_psd_power(block, 0.5) for block in blocks]
    inverse_roots = [_psd_power(part(j, j), -0.5) for j in range(len(blocks))]
    cross = {}
    for j in range(len(blocks)):
        for k in range(j + 1, len(blocks)):
            cross[(j, k)] = roots[j] @ inverse_roots[j] @ part(j, k) @ inverse_roots[k] @ roots[k]
            cross[(k, j)] = cross[(j, k)].T
    return cross


def make_random_costs(
    system: LinearSystem,
    time_domain: str | TimeDomain = "continuous",
    q_i: str = "pd",
    cross_terms: bool = False,
    sparsity: float = 0.0,
    amplitude: float = 10.0,
    diag: bool = True,
    max_iter: int = 1000,
    seed: int | None = None,
) -> list[QuadraticCost]:
    """
    Generates random quadratic costs for all players in an LQ game. For player i, R_i,ii is
    positive definite and R_i,jj (j != i) positive semidefinite. With cross terms, R_i = B C B with
    B = blockdiag(R_i,jj^(1/2)) and C the block correlation matrix of a random Wishart matrix, so R_i
    is positive semidefinite and has the same diagonal blocks as without them.

    Parameters
    ----------
    system : LinearSystem
        The system on which the game is defined.
    time_domain : str | TimeDomain
        Whether the game evolves in continuous or discrete time. Used to check detectability
        when q_i="psd".
    q_i : str
        Definiteness of the Q_i matrices. Either "pd" for positive definite,
        or "psd" for positive semidefinite.
        "pd" sufficiently ensures that (A, Q_i^{1/2}) is detectable for each player i.
    cross_terms : bool
        Whether R_i has the cross terms R_i,jk (j != k) described above; otherwise they are zero.
    sparsity : float
        Fraction of zero entries to introduce in the cost matrices. Not applied to those defined positive definite.
    amplitude : float
        Amplitude for the random entries in the cost matrices.
    diag : bool
        Structure of the Q and R matrices. If True, use diagonal matrices,
        if False, use full matrices.
    max_iter : int
        Maximum attempts per player to find a Q_i satisfying detectability when q_i="psd".
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[QuadraticCost]
        List of costs per player.
    """
    time_domain = resolve_time_domain(time_domain)
    rng = np.random.default_rng(seed)
    A = system.A
    n = system.n
    ms = system.ms
    N = system.N

    # 1. Q_i
    if q_i == "pd":
        Qs = [random_PD_matrix(n=n, diag=diag, sparsity=0, amplitude=amplitude, rng=rng) for _ in range(N)]
    elif q_i == "psd":
        Qs = []
        for i in range(N):
            for _ in range(max_iter):
                Q_i = random_PSD_matrix(n=n, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng)
                # Ensure (A, Q_i^{1/2}) is detectable
                if is_detectable(A, Q_i, time_domain=time_domain):
                    Qs.append(Q_i)
                    break
            else:
                raise FactorySamplingError(f"Cost Factory: Failed to find a detectable Q_{i} after {max_iter} attempts.")
    else:
        raise ValueError(f"Cost Factory: Unknown value for q_i '{q_i}'. Use 'pd' or 'psd'.")

    # 2. R_i,ii (PD) and R_i,jj for j != i (PSD)
    Rs = [{(i, i): random_PD_matrix(n=m_i, diag=diag, sparsity=0, amplitude=amplitude, rng=rng)} for i, m_i in enumerate(ms)]
    for i in range(N):
        for j, m_j in enumerate(ms):
            if j != i:
                Rs[i][(j, j)] = random_PSD_matrix(n=m_j, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng)

    # 3. R_i,jk for j != k
    if cross_terms:
        for i in range(N):
            Rs[i].update(_cross_terms([Rs[i][(j, j)] for j in range(N)], rng))

    return [QuadraticCost(Q=Qs[i], R=Rs[i]) for i in range(N)]
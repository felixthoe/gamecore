# src/gamecore/factories/cost_factory.py

import numpy as np

from ..cost.quadratic_cost import QuadraticCost
from ..system.linear_system import LinearSystem
from ..utils.utils import (
    random_PSD_matrix, 
    random_PD_matrix, 
    random_symmetric_matrix, 
    random_matrix,
    is_detectable,
)

def make_random_costs(
    system: LinearSystem,
    q_i_def: str = "pd",
    r_ijj: str = "free",             # Diagonal terms u[j].T R_i,jj u[j]
    r_ijk: str = "zero",             # Cross terms j≠k u[j].T R_i,jk u[k]
    enforce_psd_r_i: bool = True,    # Enforce positive semidefiniteness of overall block matrix R_i
    sparsity: float = 0.0,
    amplitude: float = 10.0,
    diag: bool = True,
    seed: int | None = None,
) -> list[QuadraticCost]:
    """
    Generates random quadratic costs for all players in an LQ game.
    Overall control block matrices R_i are ensured to be positive semidefinite.

    Parameters
    ----------
    system : LinearSystem
        The system on which the game is defined.        
    q_i_def : str
        Definiteness of the Q_i matrices. Either "pd" for positive definite,
        or "psd" for positive semidefinite.
        "pd" sufficiently ensures that (A, Q_i^{1/2}) is detectable for each player i.
    r_ijj : str
        Constraints on the R_i,jj matrices for j ≠ i. Either "zero" for zero matrices,
        "psd" for positive semidefinite, or "free" for arbitrary matrices. 
        Positive semidefiniteness of the overall block matrix R_i is enforced if enforce_psd_r_i is True.
    r_ijk : str
        Constraints on the R_i,jk matrices for j ≠ k. Either "zero" for zero matrices,
        or "free" for arbitrary matrices. 
        "free" requires r_ijj in {"free", "psd"} to be valid if enforce_psd_r_i is True.
        Positive semidefiniteness of the overall block matrix R_i is enforced if enforce_psd_r_i is True.
    enforce_psd_r_i : bool
        If True, adjust the generated R_i matrices to ensure they are positive semidefinite.
    sparsity : float
        Fraction of zero entries to introduce in the cost matrices.
    amplitude : float
        Amplitude for the random entries in the cost matrices.
    diag : bool
        Structure of the Q and R matrices. If True, use diagonal matrices,
        if False, use full matrices.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[QuadraticCost]
        List of costs per player.
    """  
    rng = np.random.default_rng(seed)
    A = system.A
    n = system.n
    ms = system.ms
    N = system.N

    # 1. Q_i
    if q_i_def == "pd":
        Qs = [random_PD_matrix(n=n, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng) for _ in range(N)]
    elif q_i_def == "psd":
        Qs = []
        for i in range(N):
            while True:
                Q_i = random_PSD_matrix(n=n, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng)
                # Ensure (A, Q_i^{1/2}) is detectable
                if is_detectable(A, Q_i):
                    Qs.append(Q_i)
                    break
    else:
        raise ValueError(f"Cost Factory: Unknown value for q_i_def '{q_i_def}'. Use 'pd' or 'psd'.")

    # 2. R_i,ii (always PD)
    Rs_iii = [random_PD_matrix(n=m_i, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng) for m_i in ms]
    Rs = [{(i, i): R_iii} for i, R_iii in enumerate(Rs_iii)]

    # 3. R_i,jj
    if r_ijj == "zero":
        # R_i,jj = 0 for j ≠ i
        pass
    elif (r_ijj == "psd"):
        # R_i,jj is PSD for j ≠ i
        for i in range(N):
            for j, m_j in enumerate(ms):
                if j != i:
                    Rs[i][(j,j)] = random_PSD_matrix(n=m_j, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng)
    elif r_ijj == "free":
        # No definiteness constraint on R_i,jj for j ≠ i
        for i in range(N):
            for j, m_j in enumerate(ms):
                if j != i:
                    Rs[i][(j,j)] = random_symmetric_matrix(n=m_j, diag=diag, sparsity=sparsity, amplitude=amplitude, rng=rng)
    else:
        raise ValueError(f"Cost Factory: Unknown value for r_ijj '{r_ijj}'. Use 'zero', 'psd', or 'free'.")
    
    # 4. R_i,jk
    if r_ijk == "zero":
        # R_i,jk = 0 for j ≠ k
        pass
    elif r_ijk == "free":
        if r_ijj == "zero" and enforce_psd_r_i:
            raise ValueError("Cost Factory: r_ijk='free' with r_ijj='zero' and enforce_psd_r_i=True is not valid. The overall block matrix R_i will in general not be positive semidefinite and cannot be adjusted to be so without modifying the zero R_i,jj blocks. Consider setting r_ijj to 'psd' or 'free', or setting enforce_psd_r_i to False.")
        # No constraint on R_i,jk for j ≠ k
        for i in range(N):
            for j in range(N):
                for k in range(j+1, N):
                    R_ijk = random_matrix(m=ms[j], n=ms[k], sparsity=sparsity, amplitude=amplitude, rng=rng)
                    Rs[i][(j,k)] = R_ijk
                    Rs[i][(k,j)] = R_ijk.T  # Ensure symmetry of overall matrix R_i
    else:
        raise ValueError(f"Cost Factory: Unknown value for r_ijk '{r_ijk}'. Use 'zero' or 'free'.")
    
    # 5. Check if overall R_i is positive semidefinite
    for i in range(N):
        # Assemble full R_i matrix
        R_i = np.zeros((sum(ms), sum(ms)))
        index_j = 0
        for j in range(N):
            index_k = 0
            for k in range(N):
                key = (j, k)
                if key in Rs[i]:
                    R_block = Rs[i][key]
                else:
                    R_block = np.zeros((ms[j], ms[k]))
                R_i[index_j:index_j+ms[j], index_k:index_k+ms[k]] = R_block
                index_k += ms[k]
            index_j += ms[j]
        # Check definiteness
        eigvals = np.linalg.eigvalsh(R_i)
        min_eig = np.min(eigvals)
        if min_eig < 0:
            if not enforce_psd_r_i:
                print(f"Warning: R_{i} is not positive semidefinite (min eigenvalue = {min_eig:.4e}), but enforce_psd_r_i is False, so no adjustment is made.")
                continue
            else:
                if r_ijj == "zero":
                    raise ValueError(
                        "Cost Factory: Cannot enforce positive semidefiniteness of R_i with r_ijj='zero' when cross terms are present. We should conceptionally not reach this point as an Exception should have been raised to catch this earlier."
                    )
                # Shift to make positive semidefinite
                shift = (-min_eig + 1e-9)
                for j in range(N):
                    key = (j, j)
                    Rs[i][key] += shift * np.eye(ms[j]) # existence of key of R_i guaranteed by construction

    
    # 6. Construct QuadraticCost objects
    return [QuadraticCost(Q=Qs[i], R=Rs[i]) for i in range(N)]
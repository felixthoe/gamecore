# src/gamecore/utils/utils.py

import numpy as np
from numpy.linalg import matrix_rank
from scipy.stats import ortho_group


def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
    """Check controllability of (A, B) via rank condition."""
    n = A.shape[0]
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack((ctrb, np.linalg.matrix_power(A, i) @ B))
    return matrix_rank(ctrb) == n

def is_observable(A: np.ndarray, C: np.ndarray) -> bool:
    """Checks observability of (A, C)."""
    return is_controllable(A.T, C.T)

def is_stabilizable(A: np.ndarray, B: np.ndarray, game_type: str = "differential") -> bool:
    """Checks stabilizability of (A, B)."""
    eigvals, eigvecs = np.linalg.eig(A.T)
    for i, λ in enumerate(eigvals):
        if game_type == "differential":
            unstable = np.real(λ) > 0
        elif game_type == "dynamic":
            unstable = np.abs(λ) >= 1.0
        else:
            raise ValueError(f"Unknown game type: {game_type}")
        if unstable:
            w = eigvecs[:, i]
            if not controls_mode(w, B):
                return False
    return True

def is_detectable(A: np.ndarray, C: np.ndarray, game_type: str = "differential") -> bool:
    """Checks detectability of (A, C)."""
    return is_stabilizable(A.T, C.T, game_type=game_type)

def controls_mode(w: np.ndarray, B: np.ndarray) -> bool:
    """Checks if the given mode w (left Eigenvector) can be controlled with B."""
    return np.any(np.abs(w.T @ B) > 1e-10)

def observes_mode(v: np.ndarray, C: np.ndarray) -> bool:
    """Checks if the given mode v (right Eigenvector) can be observed with C."""
    return np.any(np.abs(C @ v) > 1e-10)

def is_pos_def(M: np.ndarray) -> bool:
    return np.all(np.linalg.eigvalsh(M) > 0)

def is_pos_semidef(M: np.ndarray) -> bool:
    return np.all(np.linalg.eigvalsh(M) >= 0)

def sparsify(M: np.ndarray, sparsity: float, rng: np.random.Generator,) -> np.ndarray:
    # return a matrix with entries sparsified according to the given sparsity level
    assert 0 <= sparsity < 1.0
    while True:
        mask = rng.random(M.shape) > sparsity
        if np.any(mask):
            return M * mask

def symmetric_sparsify(M: np.ndarray, sparsity: float, rng: np.random.Generator,) -> np.ndarray:
    # return a symmetric matrix with entries sparsified according to the given sparsity level
    mask_upper = rng.random(M.shape) > sparsity
    mask_upper = np.triu(mask_upper)
    mask_symmetric = mask_upper | mask_upper.T
    return M * mask_symmetric

def random_PSD_matrix(n: int, diag: bool, sparsity: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """    
    Generates a random symmetric, positive semi-definite matrix of size n x n.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    diag : bool
        If True, generate a diagonal matrix, otherwise a full matrix.
    sparsity : float
        Fraction of zero entries to introduce in the matrix.
    amplitude : float
        Amplitude for the random entries in the matrix.
    rng : np.random.Generator
        Random number generator for reproducibility.
    Returns
    -------
    np.ndarray
        Random positive semi-definite matrix of size n x n.
    """
    eigvals = rng.uniform(low=0, high=1, size=(n,))
    M = amplitude * sparsify(np.diag(eigvals), sparsity, rng)
    if diag:
        return M
    else:
        V = ortho_group.rvs(dim=n, random_state=rng)
        return V @ M @ V.T

def random_PD_matrix(n: int, diag: bool, sparsity: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a random symmetric, positive definite matrix of size n x n.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    diag : bool
        If True, generate a diagonal matrix, otherwise a full matrix.
    sparsity : float
        Fraction of zero entries to introduce in the matrix.
    amplitude : float
        Amplitude for the random entries in the matrix.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Random positive definite matrix of size n x n.
    """
    eigvals = rng.uniform(low=0, high=1, size=(n,))
    M = amplitude * sparsify(np.diag(eigvals), sparsity, rng) + 1e-3 * np.eye(n)
    if diag:
        return M 
    else:
        V = ortho_group.rvs(dim=n, random_state=rng)
        return V @ M @ V.T 

def random_symmetric_matrix(n: int, diag: bool, sparsity: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a random (possibly indefinite) symmetric matrix of size n x n.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    diag : bool
        If True, generate a diagonal matrix, otherwise a full matrix.
    sparsity : float
        Fraction of zero entries to introduce in the matrix.
    amplitude : float
        Amplitude for the random entries in the matrix.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Random symmetric matrix of size n x n.
    """
    eigvals = rng.uniform(low=-1, high=1, size=(n,))
    M = amplitude * sparsify(np.diag(eigvals), sparsity, rng)
    if diag:
        return M
    else:
        V = ortho_group.rvs(dim=n, random_state=rng)
        return V @ M @ V.T
    
def random_matrix(m: int, n: int, sparsity: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a random asymmetric matrix of size m x n.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    sparsity : float
        Fraction of zero entries to introduce in the matrix.
    amplitude : float
        Amplitude for the random entries in the matrix.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Random asymmetric matrix of size m x n.
    """
    M = amplitude * rng.uniform(low=-1, high=1, size=(m, n))
    return sparsify(M, sparsity, rng)
    
def transposition_matrix(m: int, n: int) -> np.ndarray:
    """
    Construct the transposition matrix T_{m,n} such that
    vec(A.T) = T_{m,n} @ vec(A) for any A in R^{m x n}, using column-major ordering.

    Parameters
    ----------
    m : int
        Rows of A.
    n : int
        Columns of A.

    Returns
    -------
    T : np.ndarray
        Transposition matrix of shape (n*m, m*n).
    """
    T = np.zeros((n * m, m * n))
    for i in range(m):
        for j in range(n):
            # In vec(A) with column-major ordering, A[i, j] → index i + j*m
            row_index = j + i * n           # position of A.T[j, i] in vec(A.T)
            col_index = i + j * m           # position of A[i, j] in vec(A)
            T[row_index, col_index] = 1
    return T
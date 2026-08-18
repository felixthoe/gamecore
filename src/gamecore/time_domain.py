# src/gamecore/time_domain.py

from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.linalg import (
    schur,
    solve,
    solve_continuous_are,
    solve_continuous_lyapunov,
    solve_discrete_are,
    solve_discrete_lyapunov,
)
from scipy.linalg.lapack import get_lapack_funcs


class TimeDomain(ABC):
    """
    Whether a game's underlying system evolves in continuous or discrete time, and the
    corresponding stability/Lyapunov/Riccati conventions. Held as `self.time_domain` on
    `BaseGame`.
    """

    label: str

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        ...

    @property
    def is_discrete(self) -> bool:
        return not self.is_continuous

    @abstractmethod
    def default_horizon(self) -> float | int:
        """Default (system) simulation horizon: a time span for continuous, a step count for discrete."""

    @abstractmethod
    def stability_margin(self, eigs: np.ndarray) -> np.ndarray:
        """
        Elementwise signed distance from the stability boundary: positive inside the stable
        region (larger means further from the boundary), non-positive on or outside it.
        """

    def is_stable_eig(self, eigs: np.ndarray) -> np.ndarray:
        """Elementwise predicate: whether each eigenvalue lies in the open stable region."""
        return self.stability_margin(eigs) > 0

    def is_stable(self, eigs: np.ndarray, margin: float = 0.0) -> bool:
        """Whether every eigenvalue in `eigs` lies at least `margin` inside the stable region."""
        return bool(np.all(self.stability_margin(eigs) > margin))

    @abstractmethod
    def integrate(self, running_values: np.ndarray, dt: np.ndarray) -> float:
        """Aggregate a per-step running cost/quantity into a scalar total."""

    @abstractmethod
    def solve_lyapunov(self, A_cl: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Solve the cost-to-go Lyapunov equation A_cl^T P + P A_cl + M = 0 (continuous) or its
        discrete analogue, with natural (positive) forcing term M. Callers needing the adjoint
        (covariance-propagation) equation A_cl X + X A_cl^T + M = 0 instead must pass `A_cl.T`.
        """

    @abstractmethod
    def solve_lyapunov_batch(self, A_cl: np.ndarray, Ms: list[np.ndarray]) -> list[np.ndarray]:
        """
        Solve `solve_lyapunov(A_cl, M)` for every M in `Ms`, sharing the same A_cl. Reuses the
        matrix decomposition that dominates a single solve (Schur for continuous, the Kronecker
        system's factorization for discrete) across all of them, instead of repeating it once per
        M as a naive per-M loop would. Worthwhile whenever multiple players' cost-to-go matrices
        are needed for the same closed loop, e.g. `LQGame.lyapunov_matrices`.
        """

    @abstractmethod
    def solve_riccati(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve the algebraic Riccati equation for the LQR problem (A, B, Q, R)."""

    @abstractmethod
    def feedback_gain_equation(self, A: np.ndarray, B: np.ndarray, R: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lhs, rhs) such that the optimal feedback gain K solves lhs @ K = rhs."""

    def gain_from_riccati(self, A: np.ndarray, B: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Compute the feedback gain K from a solved Riccati matrix P via `feedback_gain_equation`."""
        lhs, rhs = self.feedback_gain_equation(A, B, R, P)
        return np.linalg.solve(lhs, rhs)

    def __repr__(self) -> str:
        return f"TimeDomain({self.label!r})"


class ContinuousTimeDomain(TimeDomain):
    label = "continuous"

    @property
    def is_continuous(self) -> bool:
        return True

    def default_horizon(self) -> float:
        return 10.0

    def stability_margin(self, eigs: np.ndarray) -> np.ndarray:
        return -2*np.real(eigs)

    def integrate(self, running_values: np.ndarray, dt: np.ndarray) -> float:
        return float(np.sum(running_values * dt))

    def solve_lyapunov(self, A_cl: np.ndarray, M: np.ndarray) -> np.ndarray:
        return solve_continuous_lyapunov(A_cl.T, -M)

    def solve_lyapunov_batch(self, A_cl: np.ndarray, Ms: list[np.ndarray]) -> list[np.ndarray]:
        r, u = schur(A_cl.T, output="real")
        trsyl = get_lapack_funcs("trsyl", (r, Ms[0]))
        tranb = "C" if np.iscomplexobj(A_cl) or any(np.iscomplexobj(M) for M in Ms) else "T"
        results = []
        for M in Ms:
            f = u.conj().T @ (-M) @ u
            y, scale, info = trsyl(r, r, f, tranb=tranb)
            if info < 0:
                raise ValueError(f'?TRSYL exited with illegal value in argument number {-info}.')
            if info == 1:
                warnings.warn('Input "A_cl" has an eigenvalue pair whose sum is very close to or '
                               'exactly zero. The solution is obtained via perturbing the coefficients.',
                               RuntimeWarning, stacklevel=2)
            results.append(u @ (y * scale) @ u.conj().T)
        return results

    def solve_riccati(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        return solve_continuous_are(A, B, Q, R)

    def feedback_gain_equation(self, A: np.ndarray, B: np.ndarray, R: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return R, B.T @ P


class DiscreteTimeDomain(TimeDomain):
    label = "discrete"

    @property
    def is_continuous(self) -> bool:
        return False

    def default_horizon(self) -> int:
        return 100

    def stability_margin(self, eigs: np.ndarray) -> np.ndarray:
        return 1.0 - np.abs(eigs)**2

    def integrate(self, running_values: np.ndarray, dt: np.ndarray) -> float:
        return float(np.sum(running_values))

    def solve_lyapunov(self, A_cl: np.ndarray, M: np.ndarray) -> np.ndarray:
        return solve_discrete_lyapunov(A_cl.T, M)

    def solve_lyapunov_batch(self, A_cl: np.ndarray, Ms: list[np.ndarray]) -> list[np.ndarray]:
        # unconditionally uses scipy's "direct" (Kronecker) method rather than switching to
        # "bilinear" past n=10 the way scipy's own solve_discrete_lyapunov does -- state
        # dimensions here stay well under that, and "direct" is exactly what scipy would pick
        # anyway; revisit if n grows, batching the same way over bilinear's shared Schur-of-B
        n = A_cl.shape[0]
        lhs = np.eye(n * n) - np.kron(A_cl.T, A_cl.T.conj())
        rhs = np.column_stack([M.flatten() for M in Ms])
        x = solve(lhs, rhs)
        return [np.reshape(x[:, i], (n, n)) for i in range(len(Ms))]

    def solve_riccati(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        return solve_discrete_are(A, B, Q, R)

    def feedback_gain_equation(self, A: np.ndarray, B: np.ndarray, R: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return R + B.T @ P @ B, B.T @ P @ A


CONTINUOUS = ContinuousTimeDomain()
DISCRETE = DiscreteTimeDomain()
_BY_LABEL = {CONTINUOUS.label: CONTINUOUS, DISCRETE.label: DISCRETE}


def resolve_time_domain(value: "str | TimeDomain") -> TimeDomain:
    """
    Resolve a string or `TimeDomain` instance to a `TimeDomain`.

    Parameters
    ----------
    value : str | TimeDomain
        Either `"continuous"`/`"discrete"`, or an already-resolved `TimeDomain`.

    Returns
    -------
    TimeDomain
        The resolved time domain.
    """
    if isinstance(value, TimeDomain):
        return value
    try:
        return _BY_LABEL[value]
    except (KeyError, TypeError):
        raise ValueError(f"time_domain must be 'continuous', 'discrete', or a TimeDomain, got {value!r}") from None

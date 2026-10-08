# src/gamecore/factories/strategy_factory.py

from typing import Literal
import warnings
import numpy as np
from scipy.signal import place_poles

from ..cost.quadratic_cost import QuadraticCost
from ..strategy.linear_strategy import LinearStrategy
from ..system.linear_system import LinearSystem
from ..time_domain import TimeDomain, resolve_time_domain
from ..utils.utils import FactorySamplingError, is_stabilizable


def make_lqr_strategy(
    system: LinearSystem,
    cost: QuadraticCost,
    player_idx: int = 0,
    time_domain: str | TimeDomain = "continuous",
) -> LinearStrategy:
    """
    Computes the optimal LQR feedback gain K for a given linear system and quadratic cost,
    treating the other players' inputs as absent.

    The cost is assumed to be:
        J = ∫ (xᵀ Q x + uᵀ R u) dt

    Parameters
    ----------
    system : LinearSystem
        The linear system with dynamics dx/dt = A x + B u.
        System should be stabilizable for the specified player.
    cost : QuadraticCost
        The quadratic cost specification for the single player.
    player_idx : int
        The game index of the player, whos strategy is to be calculated.
    time_domain : str | TimeDomain
        Whether the game evolves in continuous or discrete time.

    Returns
    -------
    LinearStrategy
        The optimal linear feedback strategy with gain matrix K of shape (m, n),
        such that u = -K x minimizes the cost.
    """
    time_domain = resolve_time_domain(time_domain)
    A = system.A
    B = system.Bs[player_idx]
    Q = cost.Q
    R = cost.R[(player_idx, player_idx)]

    if not is_stabilizable(A, B, time_domain=time_domain):
        raise RuntimeError(f"Strategy Factory: System is not stabilizable by player {player_idx} alone.")
    if Q.shape != (system.n, system.n):
        raise ValueError(f"Strategy Factory: Q must be a square matrix of shape ({system.n}, {system.n}).")
    if R.shape != (system.ms[player_idx], system.ms[player_idx]):
        raise ValueError(f"Strategy Factory: R must be a square matrix of shape ({system.ms[player_idx]}, {system.ms[player_idx]}).")

    P = time_domain.solve_riccati(A, B, Q, R)
    K = time_domain.gain_from_riccati(A, B, R, P)
    return LinearStrategy(K=K)


def _joint_lqr_gains(system: LinearSystem, time_domain: TimeDomain, amplitude: float) -> list[np.ndarray]:
    """
    Stacks all players into one super-player and solves a single Riccati equation for the joint
    system with a fixed (I, I) dummy cost, splitting the resulting gain by each player's ms[i].
    Deterministic, O(1), and guaranteed stabilizing whenever (A, B_total) is stabilizable.
    """
    B_total = np.hstack(system.Bs)
    Q_dummy = amplitude * np.eye(system.n)
    R_dummy = np.eye(B_total.shape[1])
    P = time_domain.solve_riccati(system.A, B_total, Q_dummy, R_dummy)
    K_total = time_domain.gain_from_riccati(system.A, B_total, R_dummy, P)
    offsets = np.cumsum([0] + system.ms)
    return [K_total[offsets[i]:offsets[i + 1]] for i in range(system.N)]


def _sample_stable_poles(n: int, time_domain: TimeDomain, rng: np.random.Generator, pole_scale: float, margin: float) -> np.ndarray:
    """Samples n poles (complex ones in conjugate pairs) at least `margin` inside the stable region."""
    poles = []
    while len(poles) < n:
        remaining = n - len(poles)
        want_complex = remaining >= 2 and rng.random() < 0.6
        if time_domain.is_continuous:
            re = -margin - rng.exponential(scale=pole_scale)
            if want_complex:
                im = rng.normal(scale=pole_scale)
                poles += [complex(re, im), complex(re, -im)]
            else:
                poles.append(complex(re, 0.0))
        else:
            radius = rng.uniform(0.02, 1.0 - margin)
            if want_complex:
                theta = rng.uniform(0.0, np.pi)
                poles += [radius * np.exp(1j * theta), radius * np.exp(-1j * theta)]
            else:
                poles.append(complex(radius * rng.choice([-1.0, 1.0]), 0.0))
    return np.array(poles[:n])


def _controllable_subspace(A: np.ndarray, B: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """
    Orthonormal basis V (n, n_c) of the controllable subspace of (A, B), built as an orthonormalized
    Krylov sequence span{B, AB, A^2 B, ...}. Uncontrollable modes (e.g. from sparse A, B) cannot be
    moved by any gain, so pole placement must leave them out.
    """
    n = A.shape[0]
    tol = rtol * max(1.0, np.linalg.norm(A, 2), np.linalg.norm(B, 2))
    V = np.zeros((n, 0))
    W = B
    for _ in range(n):
        W = W - V @ (V.T @ W)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        new = U[:, s > tol]
        if new.shape[1] == 0:
            break
        V = np.hstack([V, new])
        W = A @ new
    return V


def _random_pole_placement_gains(system: LinearSystem, time_domain: TimeDomain, rng: np.random.Generator, pole_scale: float, margin: float, max_gain_norm: float, max_attempts: int = 20) -> list[np.ndarray]:
    """
    Realizes a random target spectrum at least `margin` inside the stable region for the
    controllable part (A_c, B_c) = (V^T A V, V^T B) via centralized pole placement; uncontrollable
    eigenvalues stay where they are. The margin is re-verified post-hoc on the controllable block,
    since `place_poles`' internal robustness optimization can realize gains somewhat off-target.
    Targets are resampled until the gain norm is at most `max_gain_norm`.
    """
    B_total = np.hstack(system.Bs)
    V = _controllable_subspace(system.A, B_total)
    offsets = np.cumsum([0] + system.ms)
    if V.shape[1] == 0:
        return [np.zeros((m_i, system.n)) for m_i in system.ms]
    A_c = V.T @ system.A @ V
    B_c = V.T @ B_total
    # Players sharing input directions (or more inputs than controllable states) make B_c column-
    # rank deficient, which `place_poles` rejects; place with a full-column-rank basis B_r of
    # range(B_c) instead and map back via B_c W_k = B_r.
    U, s, Wt = np.linalg.svd(B_c, full_matrices=False)
    k = int(np.sum(s > 1e-10 * s[0]))
    B_r = U[:, :k] * s[:k]
    for _ in range(max_attempts):
        poles = _sample_stable_poles(V.shape[1], time_domain, rng, pole_scale, margin)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                K_c = Wt[:k].T @ place_poles(A_c, B_r, poles, method="YT").gain_matrix
        except ValueError:
            continue
        if not time_domain.is_stable(np.linalg.eigvals(A_c - B_c @ K_c), margin=margin):
            continue
        K_total = K_c @ V.T
        if np.linalg.norm(K_total) <= max_gain_norm:
            return [K_total[offsets[i]:offsets[i + 1]] for i in range(system.N)]
    raise FactorySamplingError(f"Strategy Factory: random pole placement found no gain with margin >= {margin} and norm <= {max_gain_norm} after {max_attempts} attempts.")


def _random_bisection_gains(system: LinearSystem, time_domain: TimeDomain, rng: np.random.Generator, scale: float, margin: float, max_gain_norm: float, max_directions: int = 1000, bracket_points: int = 25, bisection_steps: int = 10) -> list[np.ndarray]:
    """
    Finds certified-stabilizing gains that are not near-optimal: draws one random direction per
    player, scans a log-spaced grid from K=0 up to gain norm `max_gain_norm` to bracket a
    transition of "stable by at least `margin`", then bisects a bounded number of steps onto that
    transition and returns its stable side. Directions without a transition inside the scanned 
    range are rejected.
    """
    ms = system.ms

    def stable_at(Ks: list[np.ndarray]) -> bool:
        F = system.A.copy()
        for B_i, K_i in zip(system.Bs, Ks):
            F -= B_i @ K_i
        return time_domain.is_stable(np.linalg.eigvals(F), margin=margin)

    for _ in range(max_directions):
        direction = [rng.normal(size=(m_i, system.n)) for m_i in ms]
        direction = [D / np.linalg.norm(D) for D in direction]
        scales = scale * np.logspace(-3, 3, bracket_points)
        scales = scales[scales * np.sqrt(len(ms)) <= max_gain_norm]
        prev_s, prev_stable = 0.0, stable_at([0 * D for D in direction])
        lo = hi = None
        for s in scales:
            st = stable_at([s * D for D in direction])
            if st != prev_stable:
                lo, hi = (prev_s, s) if st else (s, prev_s)
                break
            prev_s, prev_stable = s, st
        if hi is None:
            continue
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if stable_at([mid * D for D in direction]) else (mid, hi)
        return [hi * D for D in direction]
    raise FactorySamplingError(f"Strategy Factory: random bisection found no direction with a margin-{margin} transition below gain norm {max_gain_norm} after {max_directions} tries.")


def make_random_strategies(
    system: LinearSystem,
    time_domain: str | TimeDomain = "continuous",
    strategy_init: Literal["joint_lqr", "random_pole_placement", "random_bisection"] = "random_pole_placement",
    amplitude: float = 1.0,
    pole_scale: float = 1.0,
    bisection_scale: float = 1.0,
    margin: float = 0.05,
    max_gain_norm: float = 1e3,
    seed: int | None = None,
) -> list[LinearStrategy]:
    """
    Generates initial strategies (gain matrices) for all players, all guaranteed jointly
    stabilizing. Each method either succeeds or raises `FactorySamplingError`.

    Parameters
    ----------
    system : LinearSystem
        The system on which the game is defined.
    time_domain : str | TimeDomain
        Whether the game evolves in continuous or discrete time.
    strategy_init : {"joint_lqr", "random_pole_placement", "random_bisection"}
        How to generate the initial gains:
          - "joint_lqr": deterministic, near-optimal. Stacks all players into one super-player
            and solves a single Riccati equation with a fixed dummy cost.
          - "random_pole_placement": realizes a random target spectrum for the controllable part
            of the closed loop via centralized pole placement.
          - "random_bisection": one random direction per player, scaled up to just inside the
            stable region. Certified-stabilizing but deliberately not near-optimal.
    amplitude : float
        Dummy cost scale, used by "joint_lqr".
    pole_scale : float
        Target-spectrum scale, used by "random_pole_placement" if time_domain is continuous.
    bisection_scale : float
        Search-direction scale, used by "random_bisection".
    margin : float
        Minimum required `TimeDomain.stability_margin` of the placed (resp. bisected) closed loop,
        used by "random_pole_placement" and "random_bisection" (ignored by "joint_lqr", whose
        Riccati solution is generically well inside the stable region already).
    max_gain_norm : float
        Upper bound on the Frobenius norm of the stacked gain, used by "random_pole_placement"
        and "random_bisection"; larger samples are rejected and resampled.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[LinearStrategy]
        List of stabilizing strategies for each player.
    """
    time_domain = resolve_time_domain(time_domain)
    rng = np.random.default_rng(seed)
    B_total = np.hstack(system.Bs)
    if not is_stabilizable(system.A, B_total, time_domain=time_domain):
        raise RuntimeError("Strategy Factory: System is not (jointly) stabilizable.")

    if strategy_init == "joint_lqr":
        try:
            Ks = _joint_lqr_gains(system, time_domain, amplitude)
        except np.linalg.LinAlgError as e:
            raise FactorySamplingError(f"Strategy Factory: joint_lqr Riccati solve failed on a near-critical system: {e}") from e
    elif strategy_init == "random_pole_placement":
        Ks = _random_pole_placement_gains(system, time_domain, rng, pole_scale, margin, max_gain_norm)
    elif strategy_init == "random_bisection":
        Ks = _random_bisection_gains(system, time_domain, rng, bisection_scale, margin, max_gain_norm)
    else:
        raise ValueError(f"Strategy Factory: unknown strategy_init '{strategy_init}'.")

    return [LinearStrategy(K) for K in Ks]

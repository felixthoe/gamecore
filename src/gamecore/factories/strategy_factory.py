# src/gamecore/factories/strategy_factory.py

from typing import Literal
import warnings
import numpy as np
from scipy.signal import place_poles

from ..cost.quadratic_cost import QuadraticCost
from ..strategy.linear_strategy import LinearStrategy
from ..system.linear_system import LinearSystem
from ..time_domain import TimeDomain, resolve_time_domain
from ..utils.utils import is_stabilizable


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


def _random_pole_placement_gains(system: LinearSystem, time_domain: TimeDomain, rng: np.random.Generator, pole_scale: float, margin: float, max_attempts: int = 5) -> list[np.ndarray] | None:
    """
    Realizes a random target spectrum at least `margin` inside the stable region via centralized
    pole placement, re-verifying the same margin post-hoc since `place_poles`' internal robustness
    optimization can realize gains somewhat off-target without producing a wrong pole assignment.
    Returns None if no attempt succeeds, so the caller can fall back to `_random_bisection_gains`.
    """
    B_total = np.hstack(system.Bs)
    for _ in range(max_attempts):
        poles = _sample_stable_poles(system.n, time_domain, rng, pole_scale, margin)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = place_poles(system.A, B_total, poles, method="YT")
        except ValueError:
            continue
        K_total = result.gain_matrix
        F = system.A - B_total @ K_total
        if time_domain.is_stable(np.linalg.eigvals(F), margin=margin):
            offsets = np.cumsum([0] + system.ms)
            return [K_total[offsets[i]:offsets[i + 1]] for i in range(system.N)]
    return None


def _random_bisection_gains(system: LinearSystem, time_domain: TimeDomain, rng: np.random.Generator, scale: float, margin: float, max_directions: int = 50, bracket_points: int = 25, bisection_steps: int = 10) -> list[np.ndarray]:
    """
    Finds certified-stabilizing gains that are not near-optimal: draws one random direction per
    player, scans a log-spaced grid from K=0 to bracket a transition into "stable by at least
    `margin`", then bisects a bounded number of steps onto that transition. Bisecting on a margin
    floor rather than on the bare stable/unstable boolean matters: the raw boolean crossing can sit
    anywhere in eigenvalue-space, so a fixed number of scale-space bisection steps gives no control
    over how close the accepted point ends up to the true boundary -- it can land within machine
    epsilon of marginal stability purely by chance, which is exactly the initial condition an
    eta_min^-2 vanilla-gradient blowup needs to make the ODE solver fail at t=0. If a direction
    never reaches the margin across the tested range, the largest tested (still-margin-stable,
    non-trivial) gain is itself a valid answer.
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
        prev_s, prev_stable = 0.0, stable_at([0 * D for D in direction])
        lo = hi = None
        for s in scales:
            st = stable_at([s * D for D in direction])
            if st != prev_stable:
                lo, hi = (prev_s, s) if st else (s, prev_s)
                break
            prev_s, prev_stable = s, st
        if hi is None:
            if prev_stable:
                return [prev_s * D for D in direction]
            continue
        for _ in range(bisection_steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if stable_at([mid * D for D in direction]) else (mid, hi)
        return [hi * D for D in direction]
    raise RuntimeError(f"Strategy Factory: no stabilizing direction found after {max_directions} tries.")


def _random_gains_with_fallback(
    system: LinearSystem,
    time_domain: TimeDomain,
    rng: np.random.Generator,
    strategy_init: str,
    pole_scale: float,
    bisection_scale: float,
    amplitude: float,
    margin: float,
) -> list[np.ndarray]:
    """
    Tries `strategy_init`'s own method first, falls back through the other random method, and
    finally to `joint_lqr` -- mathematically guaranteed to succeed for any stabilizable system,
    though its Riccati solver can still occasionally fail numerically on a near-critical system
    (smallest Hautus-rank singular value close to zero), which is let through as a `RuntimeError`
    since there is no further fallback to offer. Warns when it has to reach `joint_lqr`, since
    that silently trades a deliberately-bad starting point for a near-optimal one.
    """
    if strategy_init == "random_bisection":
        try:
            return _random_bisection_gains(system, time_domain, rng, bisection_scale, margin)
        except RuntimeError:
            pass
        Ks = _random_pole_placement_gains(system, time_domain, rng, pole_scale, margin)
    else:
        Ks = _random_pole_placement_gains(system, time_domain, rng, pole_scale, margin)
        if Ks is None:
            try:
                Ks = _random_bisection_gains(system, time_domain, rng, bisection_scale, margin)
            except RuntimeError:
                Ks = None

    if Ks is not None:
        return Ks

    warnings.warn(
        f"Strategy Factory: {strategy_init} and its fallback both found no gain with margin >= "
        f"{margin}; falling back to joint_lqr (near-optimal, not a deliberately-bad starting "
        "point) for this game.", RuntimeWarning, stacklevel=3,
    )
    try:
        return _joint_lqr_gains(system, time_domain, amplitude)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Strategy Factory: no method (including the joint_lqr fallback) could produce a "
            f"stabilizing gain with margin >= {margin} for this system: {e}"
        ) from e


def make_random_strategies(
    system: LinearSystem,
    time_domain: str | TimeDomain = "continuous",
    strategy_init: Literal["joint_lqr", "random_pole_placement", "random_bisection"] = "random_bisection",
    amplitude: float = 1.0,
    pole_scale: float = 1.0,
    bisection_scale: float = 1.0,
    margin: float = 0.05,
    seed: int | None = None,
) -> list[LinearStrategy]:
    """
    Generates initial strategies (gain matrices) for all players, all guaranteed jointly
    stabilizing.

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
          - "random_pole_placement": realizes a random target closed-loop spectrum via
            centralized pole placement; falls back to "random_bisection" if placement fails.
          - "random_bisection": one random direction per player, scaled up to just inside the
            stable region. Certified-stabilizing but deliberately not near-optimal.
    amplitude : float
        Dummy cost scale, used by "joint_lqr".
    pole_scale : float
        Target-spectrum scale, used by "random_pole_placement" if time_domain is continuous.
    bisection_scale : float
        Search-direction scale, used by "random_bisection".
    margin : float
        Minimum required `TimeDomain.stability_margin` of the resulting closed loop, used by
        "random_pole_placement" and "random_bisection" (ignored by "joint_lqr", whose Riccati
        solution is generically well inside the stable region already). Guards against gains
        that are technically stabilizing but sit close enough to the boundary that the adaptation
        dynamics' eta_min^-2 blowup makes them numerically unusable as a starting point.
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
        Ks = _joint_lqr_gains(system, time_domain, amplitude)
    elif strategy_init in ("random_pole_placement", "random_bisection"):
        Ks = _random_gains_with_fallback(system, time_domain, rng, strategy_init, pole_scale, bisection_scale, amplitude, margin)
    else:
        raise ValueError(f"Strategy Factory: unknown strategy_init '{strategy_init}'.")

    return [LinearStrategy(K) for K in Ks]

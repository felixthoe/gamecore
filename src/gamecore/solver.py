# src/gamecore/solver.py

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are, solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.integrate import solve_ivp

from .strategy.linear_strategy import LinearStrategy
from .game.lq_game import LQGame

def feedback_nash_equilibrium(game: LQGame, initial_strategies: list[LinearStrategy] | None = None, max_iteration: int = 1000, T: float = 1e7, rtol: float = 1e-10, atol: float = 1e-12) -> list[LinearStrategy]:
    """
    Computes the feedback Nash strategies for all players using different algorithms.
    If the cascaded value iteration fails, it falls back to a cascaded policy iteration.
    If the cascaded policy iteration fails too, it falls back to CDRE-based finite horizon simulation.

    Parameters
    ----------
    game : LQGame
        The LQGame instance containing the system dynamics and player objects.
    initial_strategies : list[LinearStrategy], optional
        Initial strategies for the iterative solution scheme.
        If None, uses the current strategies of the game.
    max_iteration : int, optional
        Maximum number of iterations for convergence of the iterative solution scheme. Default is 1000.
    T : float, optional
        The finite horizon for the backward integration in case of CDRE fallback. Default is 1e7.
    rtol : float, optional
        Relative tolerance for convergence check in cascaded policy iteration. Default is 1e-10.
    atol : float, optional
        Absolute tolerance for convergence check in cascaded policy iteration. Default is 1e-12.
    Returns
    -------
    list[LinearStrategy]
        List of LinearStrategy instances for each player.
    """
    # Preliminary checks and initializations
    if initial_strategies is None:
        initial_strategies = game.strategies_copy
    else:
        if not all(isinstance(strategy, LinearStrategy) for strategy in initial_strategies):
            raise TypeError("All initial strategies must be instances of LinearStrategy")
        if not len(initial_strategies) == game.N:
            raise ValueError("Number of initial strategies must match number of players N")
    if not game.is_closed_loop_stable(strategies=initial_strategies):
        raise ValueError("Nash Solver: Initial strategies are not stable")
    
    try:
        return _policy_iteration(
            game=game,
            initial_strategies=initial_strategies, 
            max_iteration=max_iteration, 
            rtol=rtol,
            atol=atol,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Nash Solver: Policy iteration failed due to: {e}")
        print("Falling back to Value iteration...")
        try:
            return _care_value_iteration(
                game=game,
                initial_strategies=initial_strategies, 
                max_iteration=max_iteration, 
                rtol=rtol,
                atol=atol,
            )
        except (RuntimeError, ValueError) as f:
            print(f"Nash Solver: Value iteration failed due to: {f}")
            print("Falling back to CDRE-based finite horizon simulation...")
            try:
                return _cdre_finite_horizon_simulation(
                    game=game,
                    T=T,
                    atol=atol
                )
            except (RuntimeError, ValueError) as g:
                print(f"Nash Solver: CDRE finite horizon simulation failed due to: {g}")
                raise RuntimeError("Nash Solver: All available methods to compute feedback Nash strategies have failed.")

    
def _policy_iteration(game: LQGame, initial_strategies: list[LinearStrategy], cascaded: bool = False, max_iteration: int | None = 1000, rtol: float = 1e-10, atol: float = 1e-12) -> list[LinearStrategy]:
    """
    Computes the feedback Nash strategies for all players oriented at Algorithm 1 of
    Chen et al (2025) "Multiplayer Cascaded Policy Iteration for Nash Differential Games"
    which is a cascaded policy iteration algorithm.
    Extended for dynamic / discrete-time games and for cross control penalties.
    
    Parameters
    ----------
    game : LQGame
        The LQGame instance containing the system dynamics and player objects.
    initial_strategies : list[LinearStrategy], optional
        Initial strategies for the iterative solution scheme.
        If None, uses the current strategies of the game.
        Either way, they are required to be stabilizing.
    cascaded : bool, optional
        If True, uses cascaded policy iteration. If False, uses simultaneous policy iteration.
    max_iteration : int, optional
        Maximum number of iterations for convergence. Default is 1000.
    rtol : float, optional
        Relative tolerance for convergence check. Default is 1e-10.
    atol : float, optional
        Absolute tolerance for convergence check. Default is 1e-12.

    Returns
    -------
    list[LinearStrategy]
        List of LinearStrategy instances for each player.
    """    
    A = game.system.A
    Bs = game.system.Bs
    Qs = [player.cost.Q for player in game.players]
    Rs = [player.cost.R for player in game.players]

    # Preliminary Condition: P_1[-1] must be positive definite
    P_0 = game.lyapunov_matrices(strategies=initial_strategies)[0]
    if not np.all(np.linalg.eigvals(P_0) > 0):
        raise ValueError("Nash Solver: First Lyapunov matrix not positive definite")
    
    # Initialization
    if game.type == "differential":
        Ks = [np.linalg.solve(Rs[0][(0,0)], Bs[0].T @ P_0)] + [np.zeros_like(B.T) for B in Bs[1:]]
    else: # dynamic
        # Formula normally uses A_i = A - sum_{j!=i} B_j K_j, but K_j=0 for initialization for j!=i
        Ks = [np.linalg.solve(Rs[0][(0,0)] + Bs[0].T @ P_0 @ Bs[0], Bs[0].T @ P_0 @ A)] + [np.zeros_like(B.T) for B in Bs[1:]]
    Ks_old = [K.copy() for K in Ks]

    # Iteration until convergence
    for _ in range(max_iteration):

        if cascaded: # cascaded policy iteration
            for i in range(game.N):
                # We dont have to use K(k+1) and K(k) separately, as we always update in place, implicitly considering K(k+1) for j<i and K(k) for j>=i
                A_i = A - sum(Bs[j] @ Ks[j] for j in range(game.N))
                M_i = Qs[i] + sum(Ks[j].T @ R_ijk @ Ks[k] for (j,k), R_ijk in Rs[i].items())  # also called Q_i in paper
                # Policy evaluation
                if game.type == "differential":
                    P_i = solve_continuous_lyapunov(A_i.T, -M_i)
                else:  # dynamic
                    P_i = solve_discrete_lyapunov(A_i.T, M_i)
                # Policy improvement
                if game.type == "differential":
                    Ks[i] = np.linalg.solve(Rs[i][(i,i)], Bs[i].T @ P_i - sum(Rs[i][(i,k)] @ Ks[k] for (j,k) in Rs[i] if j == i and k != i))
                else:  # dynamic
                    Ks[i] = np.linalg.solve(Rs[i][(i,i)] + Bs[i].T @ P_i @ Bs[i], Bs[i].T @ P_i @ (A - sum(Bs[j] @ Ks[j] for j in range(i) if j != i)) - sum(Rs[i][(i,k)] @ Ks[k] for (j,k) in Rs[i] if j == i and k != i))

        else: # simultaneous policy iteration
            F = A - sum(Bs[j] @ Ks_old[j] for j in range(game.N))
            for i in range(game.N):
                M_i = Qs[i] + sum(Ks_old[j].T @ R_ijk @ Ks_old[k] for (j,k), R_ijk in Rs[i].items())
                # Policy evaluation
                if game.type == "differential":
                    P_i = solve_continuous_lyapunov(F.T, -M_i)
                else:  # dynamic
                    P_i = solve_discrete_lyapunov(F.T, M_i)
                # Policy improvement
                if game.type == "differential":
                    Ks[i] = np.linalg.solve(Rs[i][(i,i)], Bs[i].T @ P_i - sum(Rs[i][(i,k)] @ Ks_old[k] for (j,k) in Rs[i] if j == i and k != i))
                else:  # dynamic
                    Ks[i] = np.linalg.solve(Rs[i][(i,i)] + Bs[i].T @ P_i @ Bs[i], Bs[i].T @ P_i @ F - sum(Rs[i][(i,k)] @ Ks_old[k] for (j,k) in Rs[i] if j == i and k != i))
        
        # Check Convergence
        if all(np.allclose(Ks[i], Ks_old[i], rtol=rtol, atol=atol) for i in range(game.N)):
            break
        Ks_old = [K.copy() for K in Ks]
    else:
        raise RuntimeError(f"Feedback Nash strategies did not converge within {max_iteration} iterations")

    return [LinearStrategy(K=Ks[i]) for i in range(game.N)]


def _care_value_iteration(game: LQGame, initial_strategies: list[LinearStrategy], max_iteration: int | None = 1000, rtol: float = 1e-10, atol: float = 1e-12) -> list[LinearStrategy]:
    """
    Computes the feedback Nash strategies for all players oriented at Algorithm 6 of
    Engwerda (2007) "Algorithms for computing Nash equilibria in deterministic LQ games"
    which is basically a cascaded value iteration algorithm.
    Extended for cross control penalties and for dynamic / discrete-time games.
    
    Parameters
    ----------
    game : LQGame
        The LQGame instance containing the system dynamics and player objects.
    initial_strategies : list[LinearStrategy], optional
        Initial strategies for the iterative solution scheme.
        If None, uses the current strategies of the game.
        Either way, they are required to be stabilizing.
    max_iteration : int, optional
        Maximum number of iterations for convergence. Default is 1000.
    rtol : float, optional
        Relative tolerance for convergence check. Default is 1e-10.
    atol : float, optional
        Absolute tolerance for convergence check. Default is 1e-12.

    Returns
    -------
    list[LinearStrategy]
        List of LinearStrategy instances for each player.
    """
    A = game.system.A
    Bs = game.system.Bs    
    Rs = [game.players[i].cost.R for i in range(game.N)]
    
    # Compute initial Lyapunov matrices
    Ps = game.lyapunov_matrices(strategies=initial_strategies)

    # Precompute shared big R matrix
    blocks = []
    for i in range(game.N):
        row = []
        for j in range(game.N):
            row.append(game.players[i].cost.R.get((i, j), np.zeros((game.ms[i], game.ms[j]))))
        blocks.append(row)
    R_big = np.block(blocks)
    offsets = np.cumsum([0] + [game.ms[i] for i in range(game.N)])

    def Ks_from_Ps(Ps: list[np.ndarray]) -> list[np.ndarray]:
        if game.type == "differential":
            rhs = np.vstack([Bs[i].T @ Ps[i] for i in range(game.N)])
            K_vstack = np.linalg.solve(R_big, rhs)
        else:  # dynamic
            rhs = np.vstack([Bs[i].T @ Ps[i] @ A for i in range(game.N)])
            K_vstack = np.linalg.solve(R_big + np.block([[Bs[i].T @ Ps[i] @ Bs[j] for j in range(game.N)] for i in range(game.N)]), rhs)
        return [K_vstack[offsets[i]:offsets[i+1]] for i in range(game.N)]

    # Iteration until convergence
    for _ in range(max_iteration):
        Ps_old = [P.copy() for P in Ps]
        for i, player_i in enumerate(game.players):
            Ks = Ks_from_Ps(Ps) # get current Ks from Ps
            A_i = A.copy()
            for j in range(game.N):
                if j != i:
                    A_i -= Bs[j] @ Ks[j]
            A_i += sum(Bs[i] @ np.linalg.solve(Rs[i][(i,i)], Rs[i][(i,k)] @ Ks[k]) for (j,k) in Rs[i] if j == i and k != i)
            Q_i = player_i.cost.Q.copy()
            Q_i += sum(Ks[j].T @ Rs[i][(j,k)] @ Ks[k] for (j,k) in Rs[i] if j != i and k != i)
            Q_i -= sum(sum(Ks[j].T @ Rs[i][(j,i)] @ np.linalg.solve(Rs[i][(i,i)], Rs[i][(i,m)] @ Ks[m]) for (l,m) in Rs[i] if l == i and m != i) for (j,k) in Rs[i] if j != i and k == i)
            if game.type == "differential":
                Ps[i] = solve_continuous_are(a=A_i, b=Bs[i], r=Rs[i][(i,i)], q=Q_i)
            else:  # dynamic
                Ps[i] = solve_discrete_are(a=A_i, b=Bs[i], r=Rs[i][(i,i)], q=Q_i)
        # Check convergence
        if all(np.allclose(Ps[i], Ps_old[i], rtol=rtol, atol=atol) for i in range(game.N)):
            break
    else:
        raise RuntimeError(f"Feedback Nash strategies did not converge within {max_iteration} iterations")

    # Compute the latest strategies
    K_list = Ks_from_Ps(Ps)
    return [LinearStrategy(K=K_list[i]) for i in range(game.N)]   


def _cdre_finite_horizon_simulation(game: LQGame, T: float = 1e7, atol: float = 1e-12) -> list[LinearStrategy]:
    """
    Integrates the coupled differential Riccati equations (CDREs) backward
    over a finite horizon to approximate the stationary Nash feedback gains.

    Parameters
    ----------
    game : LQGame
        The LQGame instance containing the system dynamics and player objects.
    T : float, optional
        The finite horizon for the backward integration. Default is 1e9.
    atol : float, optional
        Absolute tolerance for convergence check in CDRE integration. Default is 1e-12.
    """

    A = game.system.A
    Bs = game.system.Bs
    Qs = [player.cost.Q for player in game.players]
    Rs = [game.players[i].cost.R for i in range(game.N)]
    N = game.N
    n = game.n

    # Precompute shared big R matrix
    blocks = []
    for i in range(game.N):
        row = []
        for j in range(game.N):
            row.append(game.players[i].cost.R.get((i, j), np.zeros((game.ms[i], game.ms[j]))))
        blocks.append(row)
    R_big = np.block(blocks)
    offsets = np.cumsum([0] + [game.ms[i] for i in range(game.N)])

    def Ks_from_Ps(Ps: list[np.ndarray]) -> list[np.ndarray]:
        if game.type == "differential":
            rhs = np.vstack([Bs[i].T @ Ps[i] for i in range(game.N)])
            K_vstack = np.linalg.solve(R_big, rhs)
        else:  # dynamic
            rhs = np.vstack([Bs[i].T @ Ps[i] @ A for i in range(game.N)])
            matrix = R_big + np.block([[Bs[i].T @ Ps[i] @ Bs[j] for j in range(game.N)] for i in range(game.N)])
            K_vstack = np.linalg.solve(matrix, rhs)
        return [K_vstack[offsets[i]:offsets[i+1]] for i in range(game.N)]

    # Initial condition: P_i(T) = 0
    Ps_T = [np.zeros((n, n)) for _ in range(N)]
    Ps_flat0 = np.concatenate([P.flatten() for P in Ps_T])

    def cdre_rhs(t, Ps_flat):
        Ps = [Ps_flat[i*n*n:(i+1)*n*n].reshape(n, n) for i in range(N)]
        Ks = Ks_from_Ps(Ps)
        dPs_dt = []
        for i in range(N):
            A_i = A.copy()
            for j in range(game.N):
                if j != i:
                    A_i -= Bs[j] @ Ks[j]
            A_i += sum(Bs[i] @ np.linalg.solve(Rs[i][(i,i)], Rs[i][(i,k)] @ Ks[k]) for (j,k) in Rs[i] if j == i and k != i)
            Q_i = Qs[i].copy()
            Q_i += sum(Ks[j].T @ Rs[i][(j,k)] @ Ks[k] for (j,k) in Rs[i] if j != i and k != i)
            Q_i -= sum(sum(Ks[j].T @ Rs[i][(j,i)] @ np.linalg.solve(Rs[i][(i,i)], Rs[i][(i,m)] @ Ks[m]) for (l,m) in Rs[i] if l == i and m != i) for (j,k) in Rs[i] if j != i and k == i)
            if game.type == "differential":
                P_i_dot = -(A_i.T @ Ps[i] + Ps[i] @ A_i - Ps[i] @ Bs[i] @ np.linalg.solve(Rs[i][(i,i)], Bs[i].T @ Ps[i]) + Q_i)
            else:  # dynamic
                P_i_dot = -(Ps[i] - A_i.T @ Ps[i] @ A_i + A_i.T @ Ps[i] @ Bs[i] @ np.linalg.solve(Rs[i][(i,i)] + Bs[i].T @ Ps[i] @ Bs[i], Bs[i].T @ Ps[i] @ A_i) + Q_i)
            dPs_dt.append(P_i_dot.flatten())
        return np.concatenate(dPs_dt)
    
    def stopping_event(t, Ps_flat):
        dPs_dt_flat = cdre_rhs(t, Ps_flat)
        return np.linalg.norm(dPs_dt_flat) - atol
    stopping_event.terminal = True
    stopping_event.direction = -1  # We want to stop when the norm is decreasing

    sol = solve_ivp(
        fun=cdre_rhs, 
        t_span=(T, 0.0), 
        y0=Ps_flat0,
        method="RK45", 
        rtol=1e-13,
        atol=1e-13,
        max_step=T/100,
        events=stopping_event
    )
    if sol.status == -1:
        raise RuntimeError("Nash Solver: CDRE integration failed due to: " + sol.message)
    elif sol.status == 0:
        raise RuntimeError(f"Nash Solver: CDRE integration did not converge within a horizon of {T}.")

    # Extract P_i(t=0) and compute feedback gains
    Ps_0 = [sol.y[i*n*n:(i+1)*n*n, -1].reshape(n, n) for i in range(N)]
    return [LinearStrategy(K=K) for K in Ks_from_Ps(Ps_0)]


def feedback_stackelberg_equilibrium(game: LQGame, leader_index: int = 0, initial_leader_strat: LinearStrategy | None = None, max_iteration: int = 1000, rtol: float = 1e-10, atol: float = 1e-12) -> list[LinearStrategy]:
    """
    Computes the feedback Stackelberg strategies of one equilibrium oriented at
    Nortmann (2025) "Iterative Stackelberg equilibrium finding for linear quadratic differential games"
    which is a policy iteration algorithm.

    Parameters
    ----------
    game : LQGame
        The LQGame instance containing the system dynamics and player objects.
    leader_index : int, optional
        Index of the leader player. Default is 0.
    initial_leader_strat : LinearStrategy | None, optional
        Initial strategy for the leader player in the iterative solution scheme.
        If None, uses the current strategy of the leader player in the game.
    max_iteration : int, optional
        Maximum number of iterations for convergence of the iterative solution scheme. Default is 1000.
    rtol : float, optional
        Relative tolerance for convergence check in cascaded policy iteration. Default is 1e-10.
    atol : float, optional
        Absolute tolerance for convergence check in cascaded policy iteration. Default is 1e-12.
    Returns
    -------
    list[LinearStrategy]
        List of LinearStrategy instances for each player.
    """
    # Preliminary checks and initializations
    if game.N != 2:
        raise NotImplementedError("Stackelberg Solver: Only two-player games are currently supported.")
    if initial_leader_strat is None:
        initial_leader_strat = game.strategies_copy[leader_index]
    else:
        if not isinstance(initial_leader_strat, LinearStrategy):
            raise TypeError("Initial leader strategy must be an instance of LinearStrategy")
        
    A = game.system.A
    Bs = game.system.Bs
    Qs = [game.players[i].cost.Q for i in range(game.N)]
    Rs = [game.players[i].cost.R for i in range(game.N)]

    # Follower update: best response
    follower_index = 1 - leader_index
    def follower_best_response(leader_strategy: LinearStrategy) -> tuple[np.array, LinearStrategy]:
        A_follower = A - Bs[leader_index] @ leader_strategy.K + Bs[follower_index] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[follower_index][(follower_index,leader_index)] @ leader_strategy.K)
        Q_follower = Qs[follower_index] + leader_strategy.K.T @ (Rs[follower_index][(leader_index,leader_index)] - Rs[follower_index][(leader_index,follower_index)] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[follower_index][(follower_index,leader_index)])) @ leader_strategy.K
        R_follower = Rs[follower_index][(follower_index,follower_index)]
        if game.type == "differential":
            P_follower = solve_continuous_are(a=A_follower, b=Bs[follower_index], r=R_follower, q=Q_follower)
            K_follower = np.linalg.solve(R_follower, Bs[follower_index].T @ P_follower - Rs[follower_index][(follower_index,leader_index)] @ leader_strategy.K)
        else:  # dynamic
            P_follower = solve_discrete_are(a=A_follower, b=Bs[follower_index], r=R_follower, q=Q_follower)
            K_follower = np.linalg.solve(R_follower + Bs[follower_index].T @ P_follower @ Bs[follower_index], Bs[follower_index].T @ P_follower @ (A - Bs[leader_index] @ leader_strategy.K) - Rs[follower_index][(follower_index,leader_index)] @ leader_strategy.K)
        return P_follower, LinearStrategy(K=K_follower)
    
    R_dash_leader = Rs[leader_index][(leader_index,leader_index)] - Rs[follower_index][(leader_index, follower_index)] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[leader_index][(follower_index, leader_index)]) + (Rs[follower_index][(leader_index, follower_index)] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[leader_index][(follower_index, follower_index)]) - Rs[leader_index][(leader_index, follower_index)]) @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[follower_index][(follower_index, leader_index)])
    B_dash_leader = Bs[leader_index] - Bs[follower_index] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[follower_index][(follower_index, leader_index)])
    cross_dash_leader = (Rs[follower_index][(leader_index, follower_index)] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Rs[leader_index][(follower_index, follower_index)]) - Rs[leader_index][(leader_index, follower_index)]) @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Bs[follower_index].T)
   
    def leader_lyapunov_update(leader_strategy: LinearStrategy, follower_strategy: LinearStrategy, P_follower: np.array) -> LinearStrategy:
        A_cl = A - Bs[leader_index] @ leader_strategy.K - Bs[follower_index] @ follower_strategy.K
        strategies_list = [None, None]
        strategies_list[leader_index] = leader_strategy
        strategies_list[follower_index] = follower_strategy
        M_leader = game.players[leader_index].M(strategies=strategies_list)
        if game.type == "differential":
            P_leader = solve_continuous_lyapunov(A_cl.T, -M_leader)
            K_leader = np.linalg.solve(R_dash_leader, B_dash_leader.T @ P_leader + cross_dash_leader @ P_follower)
        else:  # dynamic
            P_leader = solve_discrete_lyapunov(A_cl.T, M_leader)
            K_leader = np.linalg.solve(R_dash_leader + B_dash_leader.T @ P_leader @ B_dash_leader, B_dash_leader.T @ P_leader @ (A - Bs[follower_index] @ follower_strategy.K) + cross_dash_leader @ P_follower)
        return LinearStrategy(K=K_leader)

    def leader_riccati_update(leader_strategy: LinearStrategy, follower_strategy: LinearStrategy, P_follower: np.array) -> LinearStrategy:
        A_dash_leader = A - Bs[follower_index] @ np.linalg.solve(Rs[follower_index][(follower_index,follower_index)], Bs[follower_index].T @ P_follower)
        pass # Implementation of Riccati update if needed in future

    leader_strategy_old = initial_leader_strat
    for _ in range(max_iteration):
        # Follower update
        P_follower, follower_strategy = follower_best_response(leader_strategy_old)
        # Leader update
        leader_strategy = leader_lyapunov_update(leader_strategy_old, follower_strategy, P_follower)
        # Check Convergence
        if np.allclose(leader_strategy.K, leader_strategy_old.K, rtol=rtol, atol=atol):
            break
        leader_strategy_old = leader_strategy.copy()
    else:
        raise RuntimeError(f"Feedback Stackelberg strategies did not converge within {max_iteration} iterations")
    
    return [leader_strategy, follower_strategy]
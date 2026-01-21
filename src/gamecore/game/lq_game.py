# src/gamecore/game/lq_game.py

import numpy as np
from scipy.linalg import eigvals, solve_continuous_lyapunov, solve_discrete_lyapunov

from .base_game import BaseGame
from ..player.lq_player import LQPlayer
from ..strategy.linear_strategy import LinearStrategy
from ..system.linear_system import LinearSystem
from ..utils.logger import DataLogger

class LQGame(BaseGame):
    """
    Represents a Linear Quadratic dynamic game with fixed linear feedback strategies.
    """

    def __init__(
        self,          
        system: LinearSystem,
        players: list[LQPlayer],
        type: str = "differential",
        Sigma0: np.ndarray | None = None,
    ):
        """
        Initializes the LQGame with system, players, game type, and initial state covariance.
        
        Parameters
        ----------
        system : LinearSystem
            The linear system shared by all players.
        players : list[LQPlayer]
            List of LQ players in the game.
        type : str
            Type of the game, either "differential" or "dynamic".
        Sigma0 : np.ndarray, optional
            Initial state covariance matrix. If None, defaults to identity matrix.
        """
        # Consistency checks
        if not all(isinstance(player, LQPlayer) for player in players):
            raise TypeError("All players must be instances of LQPlayer")
        if not isinstance(system, LinearSystem):
            raise TypeError("System must be an instance of LinearSystem")
        for i, player in enumerate(players):
            cost = player.cost
            # Check Q_i
            if cost.Q.shape != (system.n, system.n):
                raise ValueError(f"Q_{i} shape mismatch: expected ({system.n}, {system.n}), got {cost.Q.shape}")
            if not np.all(np.linalg.eigvalsh(cost.Q) >= -1e-8):
                raise ValueError(f"Q_{i} must be positive semidefinite")
            if not np.allclose(cost.Q, cost.Q.T):
                raise ValueError(f"Q_{i} must be symmetric")
            # Check R_{i,jk}
            for (j, k), R_ijk in cost.R.items():
                if not (0 <= j < system.N) or not (0 <= k < system.N):
                    raise ValueError(f"Invalid index j={j} or k={k} in R_{i},{j}{k}")
                m_j = system.ms[j]
                m_k = system.ms[k]
                if R_ijk.shape != (m_j, m_k):
                    raise ValueError(f"R_{i},{j}{k} shape mismatch: expected ({m_j}, {m_k}), got {R_ijk.shape}")
                if j == i and k == i and not np.all(np.linalg.eigvalsh(R_ijk) > 0):
                    raise ValueError(f"R_{i},{i}{i} must be positive definite")
        super().__init__(system=system, players=players, type=type)
        self.Sigma0 = Sigma0 if Sigma0 is not None else np.eye(system.n)
        # Ensure static type checkers work properly
        self.system: LinearSystem
        self.players: list[LQPlayer]

        # Closed-loop system stability check
        if not self.is_closed_loop_stable():
            print("Warning: System is not closed loop stable given the current strategies.")
    
    def is_closed_loop_stable(self, strategies: list[LinearStrategy] | None = None) -> bool:
        """
        Check if the closed-loop system is stable.

        Parameters
        ----------
        strategies : list[LinearStrategy], optional
            List of strategies for all players. If None, uses the current strategies.
        
        Returns
        -------
        bool
            True if the closed-loop system is stable, False otherwise.
        """
        if strategies is None:
            strategies = self.strategies
        A_cl = self.system.A_cl(strategies)
        eigs = eigvals(A_cl)
        if self.type == "differential":
            return np.all(np.real(eigs) < 0)
        else: # dynamic
            return np.all(np.abs(eigs) < 1)
    
    def A_cl(self, strategies: list[LinearStrategy] | None = None) -> np.ndarray:
        """
        Computes the closed-loop system matrix A_cl based on the given or current strategies.
        
        Parameters
        ----------
        strategies : list[LinearStrategy], optional
            List of strategies for all players. If None, uses the current strategies.
        
        Returns
        -------
        np.ndarray
            The closed-loop system matrix A_cl.
        """
        if strategies is None:
            strategies = self.strategies
        else:
            if len(strategies) != self.N:
                raise ValueError("Number of strategies must match number of players N")
            for i, strategy in enumerate(strategies):
                if not isinstance(strategy, LinearStrategy):
                    raise TypeError(f"Strategy {i} must be an instance of LinearStrategy")
        return self.system.A_cl(strategies)
    
    def Ms(self) -> list[np.ndarray]:
        """
        Computes the combined cost matrices M_i = Q_i + ∑_{j,k} K_jᵀ R_{i,jk} K_k for all players.
        Wrapper for the cost's M method.
        
        Returns
        -------
        list[np.ndarray]
            List of combined cost matrices M_i for each player.
        """
        return [player.cost.M(strategies=self.strategies) for player in self.players]

    def lyapunov_matrices(self, strategies: list[LinearStrategy] | None = None) -> list[np.ndarray]:
        """
        Computes the Lyapunov matrices for all players under the given or current strategies.
        Wrapper with central computation of the closed-loop system matrix A_cl.
        
        Parameters
        ----------
        strategies : list[LinearStrategy], optional
            List of strategies for all players. If None, uses the current strategies.

        Returns
        -------
        list[np.ndarray]
            List of Lyapunov matrices P_i for each player.
        """
        if strategies is None:
            strategies = self.strategies
        A_cl = self.system.A_cl(strategies)

        return [player.lyapunov_matrix(strategies=strategies, A_cl=A_cl, game_type=self.type) for player in self.players]
    
    def state_covariance(self, strategies: list[LinearStrategy] | None = None) -> np.ndarray:
        """
        Computes the state covariance matrix X under the given or current strategies.
        
        Parameters
        ----------
        strategies : list[LinearStrategy], optional
            List of strategies for all players. If None, uses the current strategies.

        Returns
        -------
        np.ndarray
            State covariance matrix X.
        """
        if strategies is None:
            strategies = self.strategies
        A_cl = self.system.A_cl(strategies)
        if self.type == "differential":
            X = solve_continuous_lyapunov(A_cl, -self.Sigma0)
        else:  # dynamic
            X = solve_discrete_lyapunov(A_cl, self.Sigma0)

        return X

    def strategies_costs(self, strategies: list[LinearStrategy] | None = None) -> list[float]:
        """
        Computes total cost for all players under the current Linear Strategies.
        Wrapper with central computation of the closed-loop system matrix A_cl.
        
        Parameters
        ----------
        strategies : list[LinearStrategy], optional
            List of strategies for all players. If None, uses the current strategies.

        Returns
        -------
        list[float]
            List of total costs for each player.
        """
        strategies = strategies if strategies is not None else self.strategies

        return [player.strategy_cost(strategies=strategies, system=self.system, game_type=self.type, Sigma0=self.Sigma0) for player in self.players]
    
    def __str__(self):
        string = "== LQGame ==\n"
        string += f"Number of players: {self.N}\n"
        string += f"Number of states: {self.n}\n"
        string += f"Number of inputs: {self.ms}\n"
        string += f"Game type: {self.type}\n\n"
        string += self.system.__str__() + "\n"
        for player in self.players:
            string += player.__str__() + "\n"
        return string
    
    def log(self, logger: DataLogger, prefix: str = ""):
        """
        Log the LQGame configuration, including system and player parameters.
        In subclasses, use super().log() along with additional logging for specific parameters.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional prefix for keys.
        """
        super().log(logger, prefix)
        logger.log_array(f"{prefix}Sigma0", self.Sigma0)
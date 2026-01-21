# src/gamecore/player/lq_player.py

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov

from .base_player import BasePlayer
from ..strategy.linear_strategy import LinearStrategy
from ..cost.quadratic_cost import QuadraticCost
from ..system.linear_system import LinearSystem
from ..system_trajectory import SystemTrajectory
from ..utils.logger import DataLogger

@dataclass
class LQPlayer(BasePlayer):
    """
    A player with linear strategy and quadratic cost in an LQ dynamic game.
    
    Attributes
    ----------
    strategy : LinearStrategy
        Linear state-feedback strategy.
    cost : QuadraticCost
        Quadratic cost function.
    player_idx : int
        Index of the player in the game.
    learning_rate : float
        Learning rate for strategy adaptation.
    """

    def __init__(
        self,
        strategy: LinearStrategy,
        cost: QuadraticCost,
        player_idx: int,
        learning_rate: float = 1.0,
    ):
        if not isinstance(strategy, (LinearStrategy)):
            raise TypeError("LQPlayer requires a LinearStrategy.")
        if not isinstance(cost, QuadraticCost):
            raise TypeError("LQPlayer requires a QuadraticCost.")
        super().__init__(strategy, cost, player_idx, learning_rate)
        # Ensure static type checking works properly
        self.strategy : LinearStrategy
        self.cost : QuadraticCost
    
    def strategy_cost(
        self, 
        strategies: list[LinearStrategy], 
        system: LinearSystem, 
        game_type: str = "differential",
        Sigma0: np.ndarray | None = None
    ) -> float:
        """
        Evaluate the player's cost given a set of strategies and system.

        Parameters
        ----------
        strategies : list of LinearStrategy
            List of strategies for all players in the game.
        system : LinearSystem
            The linear system shared by all players.
        game_type : str
            Type of the game, either "differential" or "dynamic".
        Sigma0 : np.ndarray, optional
            Initial state covariance matrix. If None, identity matrix is used.

        Returns
        -------
        float
            Total cost of the player under the given strategies.
        """
        n = system.n
        if Sigma0 is None:
            Sigma0 = np.eye(n)
        else:
            if Sigma0.shape != (n, n):
                raise ValueError(f"Sigma0 must be a square matrix of shape ({n}, {n}).")
        
        P_i = self.lyapunov_matrix(strategies, system=system, game_type=game_type)

        return np.trace(P_i @ Sigma0)

    def system_trajectory_cost(self, trajectory: SystemTrajectory, game_type: str = "differential") -> float:
        """
        Evaluate the quadratic cost from the player's perspective.
        
        Parameters
        ----------
        trajectory : SystemTrajectory
            Full system trajectory including all player inputs.
        game_type : str
            Type of the game, either "differential" or "dynamic".

        Returns
        -------
        float
            Total cost incurred by the player.
        """
        return self.cost.evaluate_system_trajectory(trajectory, game_type=game_type)
    
    def M(self, strategies: list[LinearStrategy]) -> np.ndarray:
        """
        Compute the combined cost matrix M_i = Q_i + ∑_{j,k} K_jᵀ R_{i,jk} K_k.
        Wrapper for the cost function's M method.

        Parameters
        ----------
        strategies : list[LinearStrategy]
            List of strategies for all players in the game.

        Returns
        -------
        np.ndarray
            Combined cost matrix M_i.
        """
        return self.cost.M(strategies)
    
    def lyapunov_matrix(
        self, 
        strategies: list[LinearStrategy], 
        system: LinearSystem | None = None, 
        A_cl: np.ndarray | None = None,
        game_type: str = "differential"
    ) -> np.ndarray:
        """
        Computes the Lyapunov matrix P_i for the player under the given Linear Strategies.
        Either the system or the closed-loop system matrix A_cl must be provided.

        Parameters
        ----------
        strategies : list[LinearStrategy]
            The strategies of all players in the game.
        system : LinearSystem
            The linear system shared by all players.
            If None, A_cl must be provided.
        A_cl : np.ndarray, optional
            Closed-loop system matrix. Equal for all players, so a central computation is preferrable. 
            If None, it will be computed from the system and strategies.
        game_type : str
            type of the game, either "differential" or "dynamic".

        Returns
        -------
        np.ndarray
            Lyapunov matrix P_i.
        """
        if system is None and A_cl is None:
            raise ValueError("Either system or A_cl must be provided.")
        if not all(isinstance(strategy, LinearStrategy) for strategy in strategies):
            raise TypeError("All strategies must be instances of LinearStrategy.")

        if A_cl is None:
            A_cl = system.A_cl(strategies)

        M_i = self.M(strategies)

        if game_type == "differential":
            return solve_continuous_lyapunov(A_cl.T, -M_i)
        else: # dynamic
            return solve_discrete_lyapunov(A_cl.T, M_i)

    def __str__(self):
        """
        Return string representation of the LQPlayer.
        """
        string = f"== LQPlayer {self.player_idx} ==\n"
        string += f"Learning Rate: {self.learning_rate}\n"
        string += f"Cost Function:\n" + self.cost.__str__(self.player_idx)
        string += f"Strategy:\n" + self.strategy.__str__(self.player_idx) + "\n"
        return string
    
    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "LQPlayer":
        """
        Load an LQPlayer from a DataLogger.
        """
        strategy = LinearStrategy.load(logger, prefix=f"{prefix}strategy_")
        cost = QuadraticCost.load(logger, prefix=f"{prefix}cost_")
        player_idx = int(logger.load_metadata_entry(f"{prefix}player_idx"))
        learning_rate = float(logger.load_metadata_entry(f"{prefix}learning_rate"))
        return cls(strategy=strategy, cost=cost, player_idx=player_idx, learning_rate=learning_rate)
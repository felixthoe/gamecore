# src/gamecore/player/base_player.py

from abc import ABC, abstractmethod

from ..strategy.base_strategy import BaseStrategy
from ..strategy.linear_strategy import LinearStrategy
from ..cost.base_cost import BaseCost
from ..cost.quadratic_cost import QuadraticCost
from ..system_trajectory import SystemTrajectory
from ..utils.logger import DataLogger
from ..system.linear_system import LinearSystem

class BasePlayer(ABC):
    """
    Abstract base class representing a player in a differential or dynamic game.

    Attributes
    ----------
    strategy : BaseStrategy
        The player's current control strategy.
    cost : BaseCost
        The player's cost function.
    player_idx : int
        Index of the player in the game.
    learning_rate : float
        Learning rate for strategy adaptation.
    """

    def __init__(self, strategy: BaseStrategy, cost: BaseCost, player_idx: int, learning_rate: float = 1.0):
        self.strategy = strategy
        self.cost = cost
        self.player_idx = player_idx
        self.learning_rate = float(learning_rate)

    @abstractmethod
    def strategy_cost(self, strategies: list[BaseStrategy], system: LinearSystem, game_type: str = "differential") -> float:
        """
        Evaluate the player's cost given a set of strategies and system.

        Parameters
        ----------
        strategies : list of BaseStrategy
            List of strategies for all players in the game.
        system : LinearSystem
            The linear system shared by all players.
        game_type : str
            Type of the game, either "differential" or "dynamic".

        Returns
        -------
        float
            Total cost incurred by the player.
        """
        pass

    @abstractmethod
    def system_trajectory_cost(self, trajectory: SystemTrajectory, game_type: str = "differential", *args, **kwargs) -> float:
        """
        Evaluate the player's cost given a system trajectory.

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
        pass
    
    @property
    def m(self) -> int:
        """Return the number of control inputs for this player."""
        return self.strategy.m

    def copy(self) -> "BasePlayer":
        """
        Create a deep copy of the player instance.
        
        Returns
        -------
        BasePlayer
            A new instance with the same strategy, cost, and player index.
        """
        return self.__class__(
            strategy=self.strategy.copy(),
            cost=self.cost,
            player_idx=self.player_idx,
            learning_rate=self.learning_rate
        )

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        """
        Log player configuration, including type and subcomponents.
        In subclasses, use super().log() along with additional logging for specific parameters.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional prefix for keys.
        """
        logger.log_metadata({
            f"{prefix}type": self.__class__.__name__, 
            f"{prefix}player_idx": self.player_idx,
            f"{prefix}learning_rate": self.learning_rate
        })
        self.strategy.log(logger, prefix=f"{prefix}strategy_")
        self.cost.log(logger, prefix=f"{prefix}cost_")

    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "BasePlayer":
        """
        Load a player from a logger based on saved metadata.
        """
        strategy_class_dict = {
            "LinearStrategy": LinearStrategy,
            # Add other strategy types here if needed
        }
        cost_class_dict = {
            "QuadraticCost": QuadraticCost,
            # Add other cost types here if needed
        }

        player_idx = logger.load_metadata_entry(f"{prefix}player_idx")
        learning_rate = logger.load_metadata_entry(f"{prefix}learning_rate")
        
        strategy_class_str = logger.load_metadata_entry(f"{prefix}strategy_type")
        strategy = strategy_class_dict[strategy_class_str].load(logger, prefix=f"{prefix}strategy_")

        cost_class_str = logger.load_metadata_entry(f"{prefix}cost_type")
        cost = cost_class_dict[cost_class_str].load(logger, prefix=f"{prefix}cost_")

        return cls(strategy, cost, player_idx, learning_rate=learning_rate)
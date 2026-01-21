# src/gamecore/system/base_system.py

import numpy as np
from abc import ABC, abstractmethod

from ..utils.logger import DataLogger
from ..strategy.linear_strategy import LinearStrategy

class BaseSystem(ABC):
    """
    Abstract base class representing the system dynamics in a differential or dynamic game.
    """

    @property
    @abstractmethod
    def N(self) -> int:
        """
        Number of players.
        """
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        """
        Dimension of the state space.
        """
        pass

    @property
    @abstractmethod
    def ms(self) -> list[int]:
        """
        List of input dimensions for each player.
        """
        pass

    @abstractmethod
    def copy(self) -> "BaseSystem":
        """
        Returns a deep copy of the system.

        Returns:
            BaseSystem: A deep copy of the current system.
        """
        pass

    @abstractmethod
    def f(self, x: np.ndarray, us: list[np.ndarray]) -> np.ndarray:
        """
        Computes the time derivative (continuous-time) or next iterate (discrete-time) of the state given the current state and list of control inputs.

        Args:
            x (np.ndarray): Current state vector of shape (n,).
            us (list[np.ndarray]): List of control inputs, one per player.

        Returns:
            np.ndarray: Time derivative dx/dt of the state (continuous-time) or next state (discrete-time).
        """
        pass

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        """
        Logs system parameters using the DataLogger.
        In subclasses, use super().log() along with additional logging for specific parameters.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional prefix for keys.
        """
        logger.log_metadata({
            f"{prefix}type": self.__class__.__name__,
            f"{prefix}N": self.N,
            f"{prefix}n": self.n,
            f"{prefix}ms": self.ms,
        })

    @classmethod
    @abstractmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "BaseSystem":
        """
        Loads system parameters from the DataLogger.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional prefix for keys.

        Returns:
            BaseSystem: Loaded system instance.
        """
        pass
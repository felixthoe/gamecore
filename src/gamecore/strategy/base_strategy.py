# src/gamecore/strategy/base_strategy.py

from abc import ABC, abstractmethod
import numpy as np

from ..utils.logger import DataLogger

class BaseStrategy(ABC):
    """
    Abstract base class for control strategies used by players.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the control law u(x).

        Parameters
        ----------
        x : np.ndarray
            Current state vector, shape (n,).

        Returns
        -------
        u : np.ndarray
            Control action, shape (m,).
        """
        pass

    @property
    @abstractmethod
    def m(self) -> int:
        """
        The dimension of the player's control.
        """
        pass

    @abstractmethod
    def params(self, order: str = "F") -> np.ndarray:
        """
        Get the parameters of the strategy as a flattened array.

        Parameters
        ----------
        order : str, optional
            The order used for flattening arrays ('C' for row-major, 'F' for column-major),
            by default "F".
        Returns
        -------
        np.ndarray
            Flattened array of strategy parameters.
        """
        pass

    @abstractmethod
    def from_params(self, params: np.ndarray, order: str = "F") -> "BaseStrategy":
        """
        Create a new instance of the strategy from a flattened array.

        Parameters
        ----------
        params : np.ndarray
            Flattened array of strategy parameters.
        order : str, optional
            The order used for reshaping arrays ('C' for row-major, 'F' for column-major),
            by default "F".

        Returns
        -------
        BaseStrategy
            Strategy instance with updated parameters.
        """
        pass

    @abstractmethod
    def adopt_params(self, params: np.ndarray, order: str = "F") -> None:
        """
        Set the parameters of the strategy based on the given flattened array.

        Parameters
        ----------
        params : np.ndarray
            Flattened array of strategy parameters.
        order : str, optional
            The order used for reshaping arrays ('C' for row-major, 'F' for column-major),
            by default "F".
        """
        pass

    @abstractmethod
    def copy(self) -> "BaseStrategy":
        """
        Create a deep copy of the strategy instance.

        Returns
        -------
        BaseStrategy
            Deep copy of this strategy.
        """
        pass

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        """
        Save the internal state of the strategy.
        In subclasses, use super().log() along with additional logging for specific parameters.

        Parameters
        ----------
        logger : DataLogger
            Logger instance for saving arrays and metadata.
        prefix : str
            Optional prefix for logger keys.
        """
        logger.log_metadata({
            f"{prefix}type": self.__class__.__name__,
            f"{prefix}m": self.m,
        })

    @classmethod
    @abstractmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "BaseStrategy":
        """
        Load strategy from saved arrays and metadata.

        Parameters
        ----------
        logger : DataLogger
            Logger instance for loading arrays and metadata.
        prefix : str
            Optional prefix for logger keys.

        Returns
        -------
        BaseStrategy
            Restored strategy instance.
        """
        pass
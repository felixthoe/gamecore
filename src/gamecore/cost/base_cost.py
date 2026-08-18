# src/gamecore/cost/base_cost.py

from abc import ABC, abstractmethod

from ..system_trajectory import SystemTrajectory
from ..time_domain import TimeDomain
from ..utils.logger import DataLogger


class BaseCost(ABC):
    """
    Abstract base class for player-specific cost parameter and evaluation.
    """

    def __call__(self, trajectory: SystemTrajectory, time_domain: str | TimeDomain = "continuous") -> float:
        return self.evaluate_system_trajectory(trajectory, time_domain=time_domain)

    @abstractmethod
    def evaluate_system_trajectory(self, trajectory: SystemTrajectory, time_domain: str | TimeDomain = "continuous") -> float:
        """
        Compute the total cost given a system trajectory.

        Parameters
        ----------
        trajectory : SystemTrajectory
            Simulated system trajectory.
        time_domain : str | TimeDomain
            Whether the game evolves in continuous or discrete time.

        Returns
        -------
        float
            Total cost incurred.
        """
        pass

    @abstractmethod
    def copy(self) -> "BaseCost":
        """
        Create a deep copy of the cost function instance.

        Returns
        -------
        BaseCost
            A new instance of the cost function.
        """
        pass

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        """
        Log parameters relevant for the cost function.
        In subclasses, use super().log() along with additional logging for specific parameters.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional prefix for keys.
        """
        logger.log_metadata({
            f"{prefix}type": self.__class__.__name__,
        })

    @classmethod
    @abstractmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "BaseCost":
        """
        Load parameters relevant for the cost function.

        Returns
        -------
        BaseCost
            Reconstructed cost function object.
        """
        pass
# src/gamecore/system_trajectory.py

from dataclasses import dataclass
import numpy as np

from .utils.logger import DataLogger

@dataclass
class SystemTrajectory:
    """
    Represents the simulation result of a continuous or discrete dynamic system.

    Attributes
    ----------
    t : np.ndarray
        Time/Iteration index vector of shape (T,).
    x : np.ndarray
        State trajectory of shape (T, n).
    us : list[np.ndarray]
        List of control trajectories, one per player. Each of shape (T, m_i).
    costs : list[float], optional
        Final costs for each player, if available.
    """

    t: np.ndarray
    x: np.ndarray
    us: list[np.ndarray]
    costs: list[float] | None = None

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        """
        Log the trajectory using the provided logger.

        Parameters
        ----------
        logger : DataLogger
            Logger instance for saving arrays and metadata.
        prefix : str
            Optional prefix for logger keys.
        """
        logger.log_array(f"{prefix}t", self.t)
        logger.log_array(f"{prefix}x", self.x)
        for i, u in enumerate(self.us):
            logger.log_array(f"{prefix}u_{i}", u)
        if self.costs is not None:
            logger.log_array(f"{prefix}costs", np.array(self.costs))
        logger.log_metadata({f"{prefix}N": len(self.us)})

    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "SystemTrajectory":
        """
        Load a previously saved trajectory using the logger.

        Parameters
        ----------
        logger : DataLogger
            Logger instance for loading arrays and metadata.
        prefix : str
            Optional prefix for logger keys.

        Returns
        -------
        SystemTrajectory
            Restored trajectory object.
        """
        t = logger.load_array(f"{prefix}t")
        x = logger.load_array(f"{prefix}x")
        N = int(logger.load_metadata_entry(f"{prefix}N"))
        us = [logger.load_array(f"{prefix}u_{i}") for i in range(N)]

        try:
            costs = logger.load_array(f"{prefix}costs").tolist()
        except FileNotFoundError:
            costs = None

        return cls(t=t, x=x, us=us, costs=costs)

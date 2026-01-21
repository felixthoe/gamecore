# src/gamecore/system/linear_system.py

from dataclasses import dataclass
import numpy as np

from .base_system import BaseSystem
from ..strategy.linear_strategy import LinearStrategy
from ..utils.logger import DataLogger

@dataclass
class LinearSystem(BaseSystem):
    """
    Linear system of the form:
        f = dx/dt = A x + sum_i B_i u_i
    or its discrete-time equivalent:
        f = x_(k+1) = A x_k + sum_i B_i u_i_k

    Args:
        A (np.ndarray): State matrix of shape (n, n).
        Bs (list[np.ndarray]): List of input matrices B_i of shape (n, m_i).
    """

    A: np.ndarray
    Bs: list[np.ndarray]

    def __post_init__(self):
        """
        Post-initialization checks and conversions.
        """
        # Ensure float64 for all matrices
        self.A = np.asarray(self.A, dtype=np.float64)
        self.Bs = [np.asarray(B, dtype=np.float64) for B in self.Bs]

        self._n = self.A.shape[0]
        self._N = len(self.Bs)

        # Sanity checks
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got {self.A.shape}")
        for i, B in enumerate(self.Bs):
            if B.shape[0] != self._n:
                raise ValueError(f"B_{i} must have {self._n} rows.")

    @property
    def N(self) -> int:
        return self._N

    @property
    def n(self) -> int:
        return self._n

    @property
    def ms(self) -> list[int]:
        return [B.shape[1] for B in self.Bs]
    
    def copy(self) -> "LinearSystem":
        """
        Returns a deep copy of the LinearSystem.

        Returns:
            LinearSystem: A deep copy of the current system.
        """
        A_copy = np.copy(self.A)
        Bs_copy = [np.copy(B) for B in self.Bs]
        return LinearSystem(A=A_copy, Bs=Bs_copy)

    def f(self, x: np.ndarray, us: list[np.ndarray]) -> np.ndarray:
        """
        Computes f = dx/dt = A x + sum_i B_i u_i
        or its discrete-time equivalent f = x_(k+1) = A x_k + sum_i B_i u_i_k.

        Args:
            x (np.ndarray): Current state.
            us (list[np.ndarray]): List of player inputs.

        Returns:
            np.ndarray: State derivative (continuous-time) or next state (discrete-time).
        """
        f = self.A @ x
        for B, u in zip(self.Bs, us):
            f += B @ u
        return f
    
    def A_cl(self, strategies: list[LinearStrategy]):
        """
        Computes the closed-loop state matrix A_cl = A - sum_i B_i K_i.

        Args:
            strategies (list[LinearStrategy]): List of player strategies.

        Returns:
            np.ndarray: Closed-loop state matrix.
        """
        A_cl = self.A.copy()
        for B, strat in zip(self.Bs, strategies):
            A_cl -= B @ strat.K
        return A_cl
    
    def __str__(self):
        string = f"== LinearSystem ==\n"
        if self.n == 1:
            string += f"A = {self.A.item()}\n"
            for i, B in enumerate(self.Bs):
                if self.ms[i] == 1:
                    string += f"B_{i} = {B.item()}\n"
                else:
                    string += f"B_{i} =\n{B}\n"
        else:
            string += f"A =\n{self.A}\n"
            for i, B in enumerate(self.Bs):
                string += f"B_{i} =\n{B}\n"

        return string

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        """
        Logs A, Bs, and metadata.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional key prefix.
        """
        super().log(logger, prefix)

        logger.log_array(f"{prefix}A", self.A)
        for i, B in enumerate(self.Bs):
            logger.log_array(f"{prefix}Bs_{i}", B)

    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "LinearSystem":
        """
        Loads a LinearSystem from the DataLogger.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional key prefix.

        Returns:
            LinearSystem: Loaded system instance.
        """
        A = logger.load_array(f"{prefix}A")
        N = int(logger.load_metadata_entry(f"{prefix}N"))
        Bs = [logger.load_array(f"{prefix}Bs_{i}") for i in range(N)]
        return cls(A=A, Bs=Bs)
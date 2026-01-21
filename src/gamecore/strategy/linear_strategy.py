# src/gamecore/strategy/linear_strategy.py

import numpy as np

from ..utils.logger import DataLogger
from .base_strategy import BaseStrategy

class LinearStrategy(BaseStrategy):
    """
    Linear state feedback strategy of the form u(t) = -Kx(t)
    or its discrete-time equivalent u(k) = -Kx(k).
    """

    def __init__(self, K: np.ndarray):
        K = np.asarray(K, dtype=np.float64)
        if K.ndim != 2:
            raise ValueError(f"K must be a 2D array, got shape {K.shape}")
        self.K = K.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x is None:
            raise ValueError("LinearStrategy requires state x, got None.")
        return -self.K @ x
    
    @property
    def m(self) -> int:
        return self.K.shape[0]
    
    def params(self, order: str = "F") -> np.ndarray:
        return self.K.flatten(order=order)
    
    def from_params(self, params: np.ndarray, order: str = "F") -> "LinearStrategy":
        if params.size != self.K.size:
            raise ValueError(f"Parameter size mismatch: expected {self.K.size}, got {params.size}.")
        K = params.reshape(self.K.shape, order=order)
        return LinearStrategy(K)
    
    def adopt_params(self, params: np.ndarray, order: str = "F") -> None:
        if params.size != self.K.size:
            raise ValueError(f"Parameter size mismatch: expected {self.K.size}, got {params.size}.")
        self.K = params.reshape(self.K.shape, order=order)

    def copy(self) -> "LinearStrategy":
        return LinearStrategy(self.K.copy())

    def log(self, logger: DataLogger, prefix: str = "") -> None:
        super().log(logger, prefix)
        logger.log_array(f"{prefix}K", self.K)

    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "LinearStrategy":
        K = logger.load_array(f"{prefix}K")
        return cls(K)
    
    def __str__(self, i: int = None):
        string = ""
        if isinstance(i, int):
            if self.K.size == 1:
                string += f"- K_{i} = {self.K.item()}"
            else:
                string += f"- K_{i} =\n{self.K}\n"
        else:
            string += f"== LinearStrategy ==\n"
            if self.K.size == 1:
                string += f"K = {self.K.item()}"
            else:
                string += f"K =\n{self.K}\n"
        return string
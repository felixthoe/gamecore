# src/gamecore/cost/quadratic_cost.py

from dataclasses import dataclass
import numpy as np

from .base_cost import BaseCost
from ..system_trajectory import SystemTrajectory
from ..utils.logger import DataLogger
from ..strategy.linear_strategy import LinearStrategy

@dataclass
class QuadraticCost(BaseCost):
    """
    Quadratic running cost of the form:

        J_i = ∫ xᵀ Q_i x + ∑_{j,k} u_jᵀ R_{i,jk} u_k dt

    (or its discrete-time equivalent).
    We assume Q_i and R_{i,jj} to be symmetric and furthermore R_{i,jk} = R_{i,kj}^T for j ≠ k.

    Args:
        Q (np.ndarray): 
            State weight matrix (n x n).
        R (dict[(int, int), np.ndarray]):
            Control weight matrices R_{i,jk} ∈ ℝ^{m_j × m_k}.
    """

    Q: np.ndarray
    R: dict[tuple[int, int], np.ndarray]

    def __post_init__(self):
        """
        Post-initialization checks and conversions.
        """
        self.Q = np.asarray(self.Q, dtype=np.float64)
        self.R = {
            (j, k): np.asarray(R_jk, dtype=np.float64)
            for (j, k), R_jk in self.R.items()
        }

        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be square")

        for (j, k), R_jk in self.R.items():
            if j <= k:
                if R_jk.ndim != 2:
                    raise ValueError(f"R[{(j, k)}] must be a 2D matrix")
                if self.R.get((k, j), None) is None:
                    raise ValueError(f"R[{(k, j)}] must be provided for symmetry if R[{(j, k)}] is given")
                else:
                    R_kj = self.R[(k, j)]
                    if R_jk.shape != R_kj.T.shape:
                        raise ValueError(f"R[{(j, k)}] and R[{(k, j)}]^T must have the same shape for symmetry")
                    if not np.allclose(R_jk, self.R[(k, j)].T):
                        raise ValueError(f"R[{(j, k)}] must be the transpose of R[{(k, j)}] for symmetry")

    def evaluate_system_trajectory(self, trajectory: SystemTrajectory, game_type: str = "differential") -> float:
        """
        Evaluate the finite-horizon cost (finite due to the simulated trajectories being finite).

        Parameters
        ----------
        trajectory : SystemTrajectory
            Simulated trajectory.
        game_type : str
            Type of the game, either "differential" or "dynamic".

        Returns
        -------
        float
            Total cost.
        """
        x = trajectory.x
        us = trajectory.us
        t = trajectory.t

        if self.Q.shape != (x.shape[1], x.shape[1]):
            raise ValueError("Q must match state dimension")

        for (j, k), R_jk in self.R.items():
            if R_jk.shape != (us[j].shape[1], us[k].shape[1]):
                raise ValueError(f"R[{(j, k)}] must match input dimensions of u_{j} and u_{k}")
            
        if game_type not in ["differential", "dynamic"]:
            raise ValueError(f"Game type must be either 'differential' or 'dynamic', got '{game_type}'")

        dt = np.diff(t)
        dt = np.append(dt, dt[-1])  # Assume last interval is same as second last
        steps = len(t)

        cost = 0.0
        for step in range(steps):
            state_cost = x[step, :].T @ self.Q @ x[step, :]
            control_cost = sum(
                us[j][step, :].T @ R_jk @ us[k][step, :]
                for (j, k), R_jk in self.R.items()
            )
            if game_type == "differential":
                cost += (state_cost + control_cost) * dt[step]
            else:  # dynamic
                cost += state_cost + control_cost

        return cost
    
    def M(self, strategies: list[LinearStrategy]) -> np.ndarray:
        """
        Compute the combined cost matrix M_i = Q_i + ∑_{j,k} K_jᵀ R_{i,jk} K_k.

        Parameters
        ----------
        strategies : list[LinearStrategy]
            List of LinearStrategy objects for each player.

        Returns
        -------
        np.ndarray
            Combined cost matrix M_i.
        """
        Ks = [p.K for p in strategies]
        M_i = self.Q.copy()
        for (j, k), R_jk in self.R.items():
            M_i += Ks[j].T @ R_jk @ Ks[k]
        return M_i

    def copy(self) -> "QuadraticCost":
        """
        Create a deep copy of the QuadraticCost instance.

        Returns
        -------
        QuadraticCost
            A new instance with the same Q and R matrices.
        """
        return self.__class__(Q=self.Q.copy(), R={(j, k): R_jk.copy() for (j, k), R_jk in self.R.items()})

    def __str__(self, i: int = None):
        string = ""
        if isinstance(i, int):
            if self.Q.shape[0] == 1:
                string += f"- Q_{i} = {self.Q.item()}\n"
            else:
                string += f"- Q_{i} =\n{self.Q}\n"
            for (j, k), R_jk in self.R.items():
                if R_jk.shape[0] == 1:
                    string += f"- R_{i},{j}{k} = {R_jk.item()}\n"
                else:
                    string += f"- R_{i},{j}{k} =\n{R_jk}\n"
        else:
            string += f"== QuadraticCost ==\n"
            if self.Q.shape[0] == 1:
                string += f"Q = {self.Q.item()}\n"
            else:
                string += f"Q =\n{self.Q}\n"
            for (j, k), R_jk in self.R.items():
                if R_jk.shape[0] == 1:
                    string += f"R_{j}{k} = {R_jk.item()}\n"
                else:
                    string += f"R_{j}{k} =\n{R_jk}\n"
        return string
    
    def log(self, logger: DataLogger, prefix: str = "") -> None:
        super().log(logger, prefix)
        logger.log_array(f"{prefix}Q", self.Q)
        logger.log_metadata({f"{prefix}R_keys": list(self.R.keys())})
        for (j, k), R_jk in self.R.items():
            logger.log_array(f"{prefix}R_{j}{k}", R_jk)

    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "QuadraticCost":
        Q = logger.load_array(f"{prefix}Q")
        R_keys = logger.load_metadata_entry(f"{prefix}R_keys")
        R = {(j, k): logger.load_array(f"{prefix}R_{j}{k}") for (j, k) in R_keys}
        return cls(Q=Q, R=R)
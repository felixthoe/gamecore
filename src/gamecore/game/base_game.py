# src/gamecore/game/base_game.py

from abc import ABC
import numpy as np
from scipy.integrate import solve_ivp

from ..system.base_system import BaseSystem
from ..player.base_player import BasePlayer
from ..strategy.base_strategy import BaseStrategy
from ..system_trajectory import SystemTrajectory
from ..utils.logger import DataLogger

class BaseGame(ABC):
    """
    Base class for a general differential or dynamic game.
    
    Attributes
    ----------
    system : BaseSystem
        The dynamic system shared by all players.
    players : list[BasePlayer]
        List of players in the game.
    type : str
        Type of the game, either "differential" or "dynamic".
    """

    def __init__(  
        self,          
        system: BaseSystem,
        players: list[BasePlayer],
        type: str = "differential",
    ):
        if not isinstance(system, BaseSystem):
            raise TypeError("System must be an instance of BaseSystem")
        if system.N != len(players):
            raise ValueError("Mismatch between system.N and number of players")
        if system.ms != [player.strategy.m for player in players]:
            raise ValueError("Mismatch between system.ms and players' control dimensions")
        for i, player in enumerate(players):
            if not isinstance(player, BasePlayer):
                raise TypeError(f"Player {player} must be an instance of BasePlayer")
            if player.player_idx != i:
                raise ValueError(f"Player index {player.player_idx} does not match expected index {i}. The player_idx property of the player should coincide with its index in the players list.")
        if type not in ["differential", "dynamic"]:
            raise ValueError(f"Game type must be either 'differential' or 'dynamic', got '{type}'")
        
        self.system = system
        self.players = players
        self.type = type

    @property
    def N(self):
        return self.system.N

    @property
    def n(self):
        return self.system.n

    @property
    def ms(self):
        return self.system.ms
    
    @property
    def strategies(self):
        """
        Returns the list of strategies for all players.
        """
        return [player.strategy for player in self.players]
    
    @property
    def strategies_copy(self):
        """
        Returns a deep copy of the list of strategies for all players.
        """
        return [player.strategy.copy() for player in self.players]
    
    def strategies_costs(self, strategies: list[BaseStrategy] | None = None) -> list[float]:
        """
        Compute total cost for all players under the given or current strategies.
        
        Parameters
        ----------
        strategies : list[BaseStrategy], optional
            List of strategies for all players. If None, uses the current strategies.

        Returns
        -------
        list[float]
            List of total costs for each player.
        """
        strategies = strategies if strategies is not None else self.strategies
        return [player.strategy_cost(strategies=strategies, system=self.system, game_type=self.type) for player in self.players]
    
    def copy(self) -> "BaseGame":
        """
        Create a deep copy of the game instance.
        """
        system_copy = self.system.copy()
        players_copy = [player.copy() for player in self.players]
        return self.__class__(system=system_copy, players=players_copy, type=self.type)
    
    def simulate_system(self, x0: np.ndarray | None = None, T: float | int | None = None) -> SystemTrajectory:
        """
        Simulates either differential or dynamic system dynamics over the time horizon T with initial state x0.
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial state of the system. If None, defaults to a vector of ones.
        T : float | int | None, optional
            Time horizon for the continuous simulation
            or number of steps for the discrete simulation.
        Returns
        -------
        SystemTrajectory
            The trajectory of the system, including time/steps, state, and controls.
        """             
        # Check validity of time horizon   
        if T is None:
            T = 10.0 if self.type == "differential" else 100

        # Set default initial state
        if x0 is None:
            x0 = np.ones(self.n)
            
        if self.type == "differential":
            return self._simulate_continuous_system(x0=x0, T=T)
        else:  # dynamic
            if not isinstance(T, int):
                if not T.is_integer():
                    raise ValueError("For dynamic games, T must be a positive integer representing the number of steps.")
                else:
                    T = int(T)
            if T <= 0:
                raise ValueError("For dynamic games, T must be a positive integer representing the number of steps.")
            return self._simulate_discrete_system(x0=x0, steps=T)
    
    def _simulate_continuous_system(self, x0: np.ndarray, T: float) -> SystemTrajectory:
        """
        Simulates the continuous system dynamics over the time horizon T with initial state x0.
        Called by simulate_system when game type is "differential".
        """
        def ode(t, x):
            us = [strat(x) for strat in self.strategies]
            return self.system.f(x, us)

        sol = solve_ivp(
            ode, 
            t_span=(0, T), 
            y0=x0, 
            method="LSODA", # LSODA is robust for stiff and non-stiff problems
            rtol=1e-13,
            atol=1e-13,
            max_step=T/1000,
        )

        # Extract the system trajectory
        x_traj = sol.y.T  # shape (num_steps, n)
        u_trajectories = [
            np.array([player.strategy(x) for x in x_traj])
            for player in self.players
        ] # shape (N, num_steps, m_i) for each player i

        traj = SystemTrajectory(t=sol.t, x=x_traj, us=u_trajectories)
        traj.costs = [player.cost(traj, game_type="differential") for player in self.players]
        return traj
    
    def _simulate_discrete_system(self, x0: np.ndarray, steps: int) -> SystemTrajectory:
        """
        Simulates the discrete system dynamics for a given amount of steps with initial state x0.
        Called by simulate_system when game type is "dynamic".
        """                
        x_values = np.zeros((steps + 1, self.n))
        x_values[0] = x0
        u_trajectories = [np.zeros((steps + 1, player.strategy.m)) for player in self.players]

        for k in range(steps+1):
            x = x_values[k]
            us = [player.strategy(x) for player in self.players]
            for i, u in enumerate(us):
                u_trajectories[i][k] = u
            if k < steps:
                x_values[k + 1] = self.system.f(x, us)

        traj = SystemTrajectory(t=np.arange(steps + 1), x=x_values, us=u_trajectories)
        traj.costs = [player.cost(traj, game_type="dynamic") for player in self.players]
        return traj
    
    def adopt_strategies(self, strategies: list[BaseStrategy]) -> None:
        """
        Adopt new strategies for all players.
        
        Parameters
        ----------
        strategies : list[BaseStrategy]
            List of new strategies to be adopted by the players.
        """
        if len(strategies) != self.N:
            raise ValueError("Number of strategies must match number of players N")
        for i, player in enumerate(self.players):
            if not isinstance(strategies[i], BaseStrategy):
                raise TypeError(f"Strategy {i} must be an instance of BaseStrategy")
            player.strategy = strategies[i].copy()

    def log(self, logger: DataLogger, prefix: str = ""):
        """
        Log the game configuration, including system and player parameters.
        In subclasses, use super().log() along with additional logging for specific parameters.

        Args:
            logger (DataLogger): Logger instance.
            prefix (str): Optional prefix for keys.
        """
        logger.log_metadata({
            f"{prefix}type": self.__class__.__name__,
            f"{prefix}N": self.N,
            f"{prefix}game_type": self.type,
        })
        self.system.log(logger, f"{prefix}system_")
        for i, player in enumerate(self.players):
            player.log(logger, f"{prefix}player{i}_")

    @classmethod
    def load(cls, logger: DataLogger, prefix: str = "") -> "BaseGame":
        # avoid circular import by imports here
        from ..system.linear_system import LinearSystem
        from ..player.lq_player import LQPlayer
        from ..game.lq_game import LQGame
        
        system_class_dict = {
            "LinearSystem": LinearSystem,
            # Add other system types here if needed
        }       
        player_class_dict = {
            "LQPlayer": LQPlayer,
            # Add other player types here if needed
        }
        game_class_dict = {
            "BaseGame": BaseGame,
            "LQGame": LQGame,
            # Add other game types here if needed
        }

        game_class_str = logger.load_metadata_entry(f"{prefix}type")
        game_class = game_class_dict.get(game_class_str)
        if game_class is None:
            raise ValueError(f"Unknown game type: {game_class_str}")

        N = logger.load_metadata_entry(f"{prefix}N")
        game_type = logger.load_metadata_entry(f"{prefix}game_type")

        system_class_str = logger.load_metadata_entry(f"{prefix}system_type")
        system = system_class_dict[system_class_str].load(logger, prefix=f"{prefix}system_")
        Sigma0 = logger.load_array(f"{prefix}Sigma0")

        players = []
        for i in range(N):
            player_class_str = logger.load_metadata_entry(f"{prefix}player{i}_type")
            players.append(player_class_dict[player_class_str].load(logger, prefix=f"{prefix}player{i}_"))

        if game_class is LQGame:
            return game_class(system, players, type=game_type, Sigma0=Sigma0)
        else:
            return game_class(system, players, type=game_type)
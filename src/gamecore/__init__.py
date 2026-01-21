# src/gamecore/__init__.py

"""Gamecore: A library for dynamic game theory and control."""

from .system.base_system import BaseSystem
from .cost.base_cost import BaseCost
from .strategy.base_strategy import BaseStrategy
from .player.base_player import BasePlayer
from .game.base_game import BaseGame

from .system.linear_system import LinearSystem
from .system_trajectory import SystemTrajectory
from .cost.quadratic_cost import QuadraticCost
from .strategy.linear_strategy import LinearStrategy
from .player.lq_player import LQPlayer
from .game.lq_game import LQGame

from .factories import (
    make_random_system,
    make_random_costs,
    make_random_strategies,
    make_random_lq_players,
    make_lqr_strategy,
    make_random_lq_game,
)

from.solver import feedback_nash_equilibrium, feedback_stackelberg_equilibrium
from .groebner import groebner_feedback_nash_equilibria

from .utils.logger import DataLogger
from .utils.sweep_runner import SweepRunner
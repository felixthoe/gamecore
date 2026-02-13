# tests/test_factories/test_player_factory.py

import pytest
import numpy as np

from src.gamecore.factories import make_random_lq_players
from src.gamecore import LQPlayer, LinearSystem
from tests.conftest import SEED


################################
# Shared fixtures
################################

@pytest.fixture
def differential_system() -> LinearSystem:
    """Provides a simple stabilizable differential linear system."""
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B1 = np.array([[0.0], [1.0]])
    B2 = np.array([[0.0], [1.0]])
    return LinearSystem(A=A, Bs=[B1, B2])

@pytest.fixture
def dynamic_system() -> LinearSystem:
    """Provides a simple stabilizable dynamic linear system."""
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B1 = np.array([[0.0], [1.0]])
    B2 = np.array([[0.0], [1.0]])
    return LinearSystem(A=A, Bs=[B1, B2])


################################
# Tests
################################

@pytest.mark.parametrize("game_type", ["differential", "dynamic"])
@pytest.mark.parametrize("q_i_def", ["pd", "psd"])
@pytest.mark.parametrize("r_ijj", ["zero", "psd", "free"])
@pytest.mark.parametrize("r_ijk", ["zero", "free"])
@pytest.mark.parametrize("enforce_psd_r_i", [True, False])
def test_make_random_lq_players(
    differential_system: LinearSystem, 
    dynamic_system: LinearSystem,
    game_type: str,
    q_i_def: str,
    r_ijj: str,
    r_ijk: str,
    enforce_psd_r_i: bool,
) -> None:
    # exclude invalid combinations
    if r_ijk == "free" and r_ijj == "zero" and enforce_psd_r_i:
        pytest.skip("Invalid combination of r_ijj, r_ijk, and enforce_psd_r_i")
    if game_type == "differential":
        system = differential_system
    else:
        system = dynamic_system
    players = make_random_lq_players(
        game_type=game_type,
        system=system,
        cost_q_i_def=q_i_def,
        cost_r_ijj=r_ijj,
        cost_r_ijk=r_ijk,
        cost_enforce_psd_r_i=enforce_psd_r_i,
        seed=SEED,
    )
    assert isinstance(players, list)
    assert len(players) == system.N
    for player in players:
        assert isinstance(player, LQPlayer)

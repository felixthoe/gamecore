# tests/test_factories/test_game_factory.py

import pytest
from src.gamecore.factories import make_random_lq_game
from src.gamecore import LQGame
from tests.conftest import SEED


@pytest.mark.parametrize("game_type", ["differential", "dynamic"])
@pytest.mark.parametrize("stabilizability", ["individual", "joint"])
@pytest.mark.parametrize("q_i_def", ["pd", "psd"])
@pytest.mark.parametrize("r_ijj", ["zero", "psd", "free"])
@pytest.mark.parametrize("r_ijk", ["zero", "free"])
@pytest.mark.parametrize("enforce_psd_r_i", [True, False])
def test_make_random_lq_game_valid_configs(
    game_type: str, q_i_def: str, stabilizability: str, r_ijj: str, r_ijk: str, enforce_psd_r_i: bool) -> None:
    # exclude invalid combinations
    if r_ijk == "free" and r_ijj == "zero" and enforce_psd_r_i:
        pytest.skip("Invalid combination of r_ijj, r_ijk, and enforce_psd_r_i")
    game = make_random_lq_game(
        n=3,
        ms=[1, 1],
        game_type=game_type,
        system_stabilizability=stabilizability,
        cost_q_i_def=q_i_def,
        cost_r_ijj=r_ijj,
        cost_r_ijk=r_ijk,
        cost_enforce_psd_r_i=enforce_psd_r_i,
        seed=SEED
    )
    assert isinstance(game, LQGame)
    assert game.system.n == 3
    assert game.system.N == 2
    assert all(player.strategy.K.shape[1] == 3 for player in game.players)

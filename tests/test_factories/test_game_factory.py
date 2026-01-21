# tests/test_factories/test_game_factory.py

import pytest
from src.gamecore.factories import make_random_lq_game
from src.gamecore import LQGame
from tests.conftest import SEED


@pytest.mark.parametrize("game_type", ["differential", "dynamic"])
@pytest.mark.parametrize("stabilizability", ["individual", "joint"])
@pytest.mark.parametrize("q_def", ["pd", "psd"])
@pytest.mark.parametrize("r_jj", ["zero", "psd", "free"])
@pytest.mark.parametrize("r_jk", ["zero", "free"])
def test_make_random_lq_game_valid_configs(
    game_type: str, q_def: str, stabilizability: str, r_jj: str, r_jk: str) -> None:
    # exclude invalid combinations
    if r_jk == "free" and r_jj == "zero":
        pytest.skip("Invalid combination of r_jj and r_jk")
    game = make_random_lq_game(
        n=3,
        ms=[1, 1],
        game_type=game_type,
        system_stabilizability=stabilizability,
        cost_q_def=q_def,
        cost_r_jj=r_jj,
        cost_r_jk=r_jk,
        seed=SEED
    )
    assert isinstance(game, LQGame)
    assert game.system.n == 3
    assert game.system.N == 2
    assert all(player.strategy.K.shape[1] == 3 for player in game.players)

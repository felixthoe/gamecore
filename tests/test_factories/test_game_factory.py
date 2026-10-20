# tests/test_factories/test_game_factory.py

import pytest
from src.gamecore.factories import make_random_lq_game
from src.gamecore import LQGame
from tests.conftest import SEED


@pytest.mark.parametrize("time_domain", ["continuous", "discrete"])
@pytest.mark.parametrize("stabilizability", ["individual", "joint"])
@pytest.mark.parametrize("q_i", ["pd", "psd"])
@pytest.mark.parametrize("cross_terms", [False, True])
def test_make_random_lq_game_valid_configs(time_domain: str, q_i: str, stabilizability: str, cross_terms: bool) -> None:
    game = make_random_lq_game(
        n=3,
        ms=[1, 1],
        time_domain=time_domain,
        system_stabilizability=stabilizability,
        cost_q_i=q_i,
        cost_cross_terms=cross_terms,
        seed=SEED
    )
    assert isinstance(game, LQGame)
    assert game.system.n == 3
    assert game.system.N == 2
    assert all(player.strategy.K.shape[1] == 3 for player in game.players)

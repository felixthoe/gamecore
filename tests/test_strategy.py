# tests/test_strategy.py

import numpy as np
import pytest

from src.gamecore import (
    BaseStrategy,
    LinearStrategy,
    DataLogger
)


################################
# BaseStrategy
################################

def test_basestrategy_is_abstract():
    with pytest.raises(TypeError):
        BaseStrategy()  # abstract methods prevent instantiation


################################
# LinearStrategy
################################

def test_linearstrategy_init_and_call():
    K = np.array([[1.0, -2.0]], dtype=np.float64)  # shape (m=1, n=2)
    strat = LinearStrategy(K)
    assert strat.m == 1
    x = np.array([0.5, -1.0], dtype=np.float64)
    u = strat(x=x)
    expected = -(K @ x)
    assert np.allclose(u, expected)

def test_linearstrategy_call_requires_x():
    K = np.array([[1.0]], dtype=np.float64)
    strat = LinearStrategy(K)
    with pytest.raises(ValueError, match="requires state x"):
        _ = strat(x=None)

def test_linearstrategy_invalid_K_ndim():
    with pytest.raises(ValueError, match="K must be a 2D array"):
        LinearStrategy(np.array([1.0, 2.0]))  # 1D

def test_linearstrategy_params_round_trip():
    K = np.array([[1.0, 0.0], [2.0, -1.0]], dtype=np.float64)
    strat = LinearStrategy(K)
    p = strat.params()
    assert p.shape[0] == K.size
    restored = strat.from_params(p)
    assert isinstance(restored, LinearStrategy)
    assert np.allclose(restored.K, K)

def test_linearstrategy_from_params_wrong_size():
    K = np.array([[1.0, 0.0]], dtype=np.float64)
    strat = LinearStrategy(K)
    wrong = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # size != K.size
    with pytest.raises(ValueError, match="Parameter size mismatch"):
        _ = strat.from_params(wrong)

def test_linearstrategy_copy_is_deep():
    K = np.array([[1.0, 2.0]], dtype=np.float64)
    strat = LinearStrategy(K)
    cp = strat.copy()
    assert cp is not strat
    assert np.allclose(cp.K, strat.K)
    cp.K[0, 0] += 10.0
    assert not np.allclose(cp.K, strat.K)

def test_linear_strategy_log_and_load(temp_logger: DataLogger):
    K = np.array([[0.5, -1.0]])
    strat = LinearStrategy(K)
    prefix = "lin_strategy_"
    strat.log(temp_logger, prefix=prefix)
    # metadata
    assert temp_logger.load_metadata_entry(f"{prefix}type") == "LinearStrategy"
    assert int(temp_logger.load_metadata_entry(f"{prefix}m")) == strat.m
    # array
    K_loaded = temp_logger.load_array(f"{prefix}K")
    assert np.allclose(K_loaded, K)
    loaded = LinearStrategy.load(temp_logger, prefix=prefix)
    assert isinstance(loaded, LinearStrategy)
    np.testing.assert_allclose(loaded.K, K)
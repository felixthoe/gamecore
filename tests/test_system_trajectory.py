# tests/test_trajectory.py

import numpy as np
import pytest

from src.gamecore import SystemTrajectory, DataLogger


#############################
# Additional Fixtures
#############################

@pytest.fixture
def trajectory_data():
    T = 5
    n = 3
    ms = [1, 2]
    t = np.arange(T, dtype=np.float64)
    x = np.stack([np.linspace(0, 1, n, dtype=np.float64) + k for k in range(T)], axis=0)  # (T, n)
    u0 = np.ones((T, ms[0]), dtype=np.float64)  # (T, 1)
    u1 = np.zeros((T, ms[1]), dtype=np.float64)  # (T, 2)
    us = [u0, u1]
    return t, x, us

@pytest.fixture
def costs_example():
    return [1.5, 2.5]  # one per player


#############################
# Construction and attributes
#############################

def test_construction_and_attributes(trajectory_data: tuple[np.ndarray, np.ndarray, list[np.ndarray]], costs_example: list[float]):
    t, x, us = trajectory_data
    traj = SystemTrajectory(t=t, x=x, us=us, costs=costs_example)
    assert isinstance(traj.t, np.ndarray)
    assert isinstance(traj.x, np.ndarray)
    assert isinstance(traj.us, list)
    assert traj.costs == costs_example
    # Shape checks
    T = t.shape[0]
    n = x.shape[1]
    assert x.shape == (T, n)
    for i, u in enumerate(us):
        assert u.shape[0] == T, f"u_{i} wrong time dimension"
        assert u.ndim == 2, f"u_{i} should be 2D with shape (T, m_i)"


#############################
# Logging and loading (round trip)
#############################

def test_log_and_load_trajectory(trajectory_data: tuple[np.ndarray, np.ndarray, list[np.ndarray]], costs_example: list[float], temp_logger: DataLogger):
    """Test whether a trajectory can be logged and reloaded correctly."""
    t, x, us = trajectory_data
    traj_original = SystemTrajectory(t=t, x=x, us=us, costs=costs_example)
    prefix = "traj_"
    traj_original.log(temp_logger, prefix=prefix)

    # Verify metadata and arrays
    assert int(temp_logger.load_metadata_entry(f"{prefix}N")) == len(us)
    t_loaded = temp_logger.load_array(f"{prefix}t")
    x_loaded = temp_logger.load_array(f"{prefix}x")
    assert np.allclose(t_loaded, t)
    assert np.allclose(x_loaded, x)
    for i, u in enumerate(us):
        u_loaded = temp_logger.load_array(f"{prefix}u_{i}")
        assert np.allclose(u_loaded, u)

    costs_loaded = temp_logger.load_array(f"{prefix}costs")
    assert np.allclose(costs_loaded, np.array(costs_example))

    # Load
    traj_loaded = SystemTrajectory.load(temp_logger, prefix=prefix)
    assert isinstance(traj_loaded, SystemTrajectory)
    assert np.allclose(traj_original.t, traj_loaded.t)
    assert np.allclose(traj_original.x, traj_loaded.x)
    assert len(traj_original.us) == len(traj_loaded.us)
    for u_orig, u_load in zip(traj_original.us, traj_loaded.us):
        assert np.allclose(u_orig, u_load)
    assert traj_original.costs == pytest.approx(traj_loaded.costs)

def test_load_without_costs(trajectory_data: tuple[np.ndarray, np.ndarray, list[np.ndarray]], temp_logger: DataLogger):
    """Test loading trajectory when no costs were logged."""
    t, x, us = trajectory_data
    traj_original = SystemTrajectory(t=t, x=x, us=us, costs=None)
    prefix = "traj_"
    traj_original.log(temp_logger, prefix=prefix)

    # cost key should not exist; load should set costs=None
    with pytest.raises(FileNotFoundError):
        _ = temp_logger.load_array(f"{prefix}costs")
    traj_loaded = SystemTrajectory.load(temp_logger, prefix=prefix)
    assert np.allclose(traj_loaded.t, t)
    assert np.allclose(traj_loaded.x, x)
    for i, u in enumerate(us):
        assert np.allclose(traj_loaded.us[i], u)
    assert traj_loaded.costs is None
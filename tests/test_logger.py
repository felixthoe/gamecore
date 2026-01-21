# tests/test_logger.py

import numpy as np
import pytest

from src.gamecore import DataLogger

def test_log_and_load_metadata(temp_logger: DataLogger) -> None:
    """Test logging and loading metadata dictionary."""
    meta = {"foo": 123, "bar": "baz"}
    temp_logger.log_metadata(meta)
    loaded = temp_logger.load_metadata()
    assert loaded["foo"] == 123
    assert loaded["bar"] == "baz"
    # Test single entry loading
    assert temp_logger.load_metadata_entry("foo") == 123
    assert temp_logger.load_metadata_entry("not_found", default="x") == "x"

def test_log_and_load_array(temp_logger: DataLogger) -> None:
    """Test logging and loading a numpy array."""
    arr = np.arange(10)
    temp_logger.log_array("myarr", arr)
    loaded = temp_logger.load_array("myarr")
    np.testing.assert_array_equal(arr, loaded)
    assert temp_logger.has_array("myarr")
    assert not temp_logger.has_array("does_not_exist")

def test_log_and_load_dict_of_arrays(temp_logger: DataLogger) -> None:
    """Test logging and loading a dict of arrays."""
    data = {"a": np.ones(3), "b": np.zeros(3)}
    temp_logger.log_dict_of_arrays("foo", data)
    loaded = temp_logger.load_dict_of_arrays("foo")
    assert set(loaded.keys()) == {"a", "b"}
    np.testing.assert_array_equal(loaded["a"], data["a"])
    np.testing.assert_array_equal(loaded["b"], data["b"])

def test_log_and_load_scalar(temp_logger: DataLogger) -> None:
    """Test logging and loading a scalar value."""
    temp_logger.log_scalar("myscalar", 3.14)
    val = temp_logger.load_scalar("myscalar")
    assert val == pytest.approx(3.14)
    assert temp_logger.has_scalar("myscalar")
    assert not temp_logger.has_scalar("does_not_exist")
    with pytest.raises(FileNotFoundError):
        temp_logger.load_scalar("does_not_exist")

def test_has_metadata_entry(temp_logger: DataLogger) -> None:
    """Test checking for metadata entry existence."""
    temp_logger.log_metadata({"foo": 1})
    assert temp_logger.has_metadata_entry("foo")
    assert not temp_logger.has_metadata_entry("bar")

def test_summarize_prints(temp_logger: DataLogger, capsys) -> None:
    """Test that summarize prints the experiment directory."""
    temp_logger.summarize()
    captured = capsys.readouterr()
    assert "Experiment saved to:" in captured.out
    assert str(temp_logger.dir)

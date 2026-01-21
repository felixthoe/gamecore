# tests/test_sweep_runner.py

import os
import pytest
import json

from src.gamecore.utils.sweep_runner import SweepRunner


#############################
# Additional Fixtures
#############################

# top level as it needs to be pickleable for parallel execution
def dummy_run_trial_fn(seed: int, sweep_params: dict, **kwargs) -> str:
    if sweep_params["param_a"] == "fail":
        raise RuntimeError("Deliberate failure")
    return "success"


@pytest.fixture
def sweep_runner(tmp_path):
    # Sweep over 2 values (1 valid, 1 that triggers an exception)
    sweep_space = {"param_a": ["ok", "fail"], "param_b": [1, 2]}
    base_dir = tmp_path / "sweep_data"

    runner = SweepRunner(
        experiment_name="test_exp",
        base_dir=str(base_dir),
        sweep_space=sweep_space,
        run_trial_fn=dummy_run_trial_fn,
        n_trials=2,
        parallel=False,
        is_valid_sweep_fn=None,
    )
    return runner


#############################
# Tests
#############################

@pytest.mark.parametrize("parallel", [True, False])
def test_sweep_execution_and_logging(sweep_runner: SweepRunner, parallel: bool):
    sweep_runner.parallel = parallel
    sweep_runner.run(dummy_kwarg="test")

    exp_dir = sweep_runner.experiment_logger.dir
    assert os.path.exists(exp_dir)

    result_files = list(os.path.join(dp, f) for dp, dn, filenames in os.walk(exp_dir)
                        for f in filenames if f == "result.json")
    assert len(result_files) == sweep_runner.total_sweeps

    # Check contents of one result.json
    with open(result_files[0], "r") as f:
        result = json.load(f)
    assert "stats_rel" in result
    assert "stats_abs" in result
    assert "trial_outcomes" in result
    assert "duration_stats" in result
    assert isinstance(result["trial_outcomes"], list)
    assert len(result["trial_outcomes"]) == sweep_runner.n_trials


def test_result_aggregation(sweep_runner: SweepRunner):
    sweep_runner.run()

    agg_path = os.path.join(sweep_runner.experiment_logger.dir, "aggregated_stats.json")
    outcome_path = os.path.join(sweep_runner.base_dir, "test_exp", "outcome_to_sweeps.json")
    all_path = os.path.join(sweep_runner.base_dir, "test_exp", "all_sweeps.json")

    for path in [agg_path, outcome_path, all_path]:
        assert os.path.exists(path)

    with open(agg_path, "r") as f:
        aggregated = json.load(f)
    assert isinstance(aggregated, dict)
    assert "stats_abs" in aggregated
    assert "stats_rel" in aggregated
    assert "timestamp" in aggregated
    assert any(count > 0 for count in aggregated["stats_abs"].values())


def test_exception_handling(sweep_runner: SweepRunner):
    sweep_runner.run()

    result_paths = [
        os.path.join(dp, "result.json")
        for dp, _, filenames in os.walk(sweep_runner.experiment_logger.dir)
        for f in filenames if f == "result.json"
    ]

    has_exception = False
    for path in result_paths:
        with open(path, "r") as f:
            res = json.load(f)
            for entry in res.get("trial_outcomes", []):
                if entry.get("outcome") == "exception":
                    has_exception = True
                    break
        if has_exception:
            break

    assert has_exception, "Expected at least one trial with an 'exception' outcome"

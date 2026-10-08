# tests/test_sweep_runner.py

import os
import json

import pytest

from src.gamecore.sweep_runner.runner import SweepRunner
from src.gamecore.sweep_runner.seed_assignment import SeedRegistry
from src.gamecore.sweep_runner.sweep_space import SweepSpace


#############################
# Additional Fixtures
#############################

# top level as it needs to be pickleable for parallel execution
def dummy_run_trial_fn(seed: int, sweep_params: dict, **kwargs) -> str:
    if sweep_params["param_a"] == "fail":
        raise RuntimeError("Deliberate failure")
    return "success"


def primary_seed_fails_trial_fn(seed: int, sweep_params: dict, **kwargs) -> str:
    # primary seeds lie below 2**31, retry seeds above
    if seed < 2**31:
        raise RuntimeError("Deliberate failure on the primary seed")
    return f"seed={seed}"


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
        retry_on_exception=(RuntimeError,),
    )
    return runner


def _result_files(exp_dir):
    return [
        os.path.join(dp, f)
        for dp, _, filenames in os.walk(exp_dir)
        for f in filenames if f == "result.json"
    ]


#############################
# Tests
#############################

@pytest.mark.parametrize("parallel", [True, False])
def test_sweep_execution_and_logging(sweep_runner: SweepRunner, parallel: bool):
    sweep_runner.parallel = parallel
    sweep_runner.run(dummy_kwarg="test")

    exp_dir = sweep_runner.experiment_logger.dir
    assert os.path.exists(exp_dir)

    result_files = _result_files(exp_dir)
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

    has_exception = False
    for path in _result_files(sweep_runner.experiment_logger.dir):
        with open(path, "r") as f:
            res = json.load(f)
        for entry in res.get("trial_outcomes", []):
            if entry.get("outcome") == "exception":
                has_exception = True
                break
        if has_exception:
            break

    assert has_exception, "Expected at least one trial with an 'exception' outcome"


def test_exception_detail_and_failures_report(sweep_runner: SweepRunner):
    """Every failed attempt (not just the last) is recorded and rolled up."""
    sweep_runner.run()
    exp_dir = sweep_runner.experiment_logger.dir
    max_attempts = sweep_runner.max_retries_on_exception

    n_fail_sweeps = 0
    total_exception_entries = 0
    for path in _result_files(exp_dir):
        with open(path, "r") as f:
            res = json.load(f)
        with open(os.path.join(os.path.dirname(path), "params.json"), "r") as f:
            params = json.load(f)

        if params["param_a"] == "fail":
            n_fail_sweeps += 1
            assert len(res["exceptions"]) == sweep_runner.n_trials * max_attempts
            for trial_idx in range(sweep_runner.n_trials):
                trial_exceptions = [e for e in res["exceptions"] if e["trial"] == trial_idx]
                assert len(trial_exceptions) == max_attempts
                for entry in trial_exceptions:
                    assert entry["exception"]
                    assert entry["traceback"]
            total_exception_entries += len(res["exceptions"])
        else:
            assert res["exceptions"] == []

    assert n_fail_sweeps == 2

    with open(os.path.join(exp_dir, "failures.json"), "r") as f:
        failures = json.load(f)
    assert len(failures) == total_exception_entries

    with open(os.path.join(exp_dir, "aggregated_stats.json"), "r") as f:
        aggregated = json.load(f)
    assert aggregated["total_exceptions"] == total_exception_entries
    assert aggregated["sweeps_with_exceptions"] == n_fail_sweeps

    run_log_path = os.path.join(exp_dir, "run.log")
    assert os.path.exists(run_log_path)
    with open(run_log_path, "r") as f:
        assert "sweep=" in f.read()


def test_sweep_space_extension(tmp_path, monkeypatch):
    base_dir = tmp_path / "sweep_data"

    def make_runner(sweep_space):
        return SweepRunner(
            experiment_name="extend_exp",
            base_dir=str(base_dir),
            sweep_space=sweep_space,
            run_trial_fn=dummy_run_trial_fn,
            n_trials=2,
            parallel=False,
        )

    first = make_runner({"param_a": ["ok"], "param_b": [1, 2]})
    first.run()
    exp_dir = first.experiment_logger.dir

    original_results = {}
    for path in _result_files(exp_dir):
        sweep_name = os.path.basename(os.path.dirname(path))
        with open(path, "r") as f:
            original_results[sweep_name] = json.load(f)
    assert len(original_results) == 2

    answers = iter(["c", "e"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    second = make_runner({"param_a": ["ok"], "param_b": [1, 2, 3]})
    second.run()

    assert len(_result_files(exp_dir)) == 3

    # Original sweeps must be untouched (skipped, not re-run).
    for sweep_name, old_result in original_results.items():
        with open(os.path.join(exp_dir, "sweeps", sweep_name, "result.json"), "r") as f:
            kept_result = json.load(f)
        assert kept_result == old_result

    with open(os.path.join(exp_dir, "aggregated_stats.json"), "r") as f:
        aggregated = json.load(f)
    assert aggregated["total_valid_sweeps"] == 3
    assert aggregated["stats_abs"]["success"] == 3 * second.n_trials


def test_sweep_space_change_rejected_when_not_superset(tmp_path, monkeypatch):
    base_dir = tmp_path / "sweep_data"

    def make_runner(sweep_space):
        return SweepRunner(
            experiment_name="reject_exp",
            base_dir=str(base_dir),
            sweep_space=sweep_space,
            run_trial_fn=dummy_run_trial_fn,
            n_trials=2,
            parallel=False,
        )

    first = make_runner({"param_a": ["ok"], "param_b": [1, 2]})
    first.run()
    exp_dir = first.experiment_logger.dir

    agg_path = os.path.join(exp_dir, "aggregated_stats.json")
    with open(agg_path, "r") as f:
        original_aggregated = json.load(f)

    answers = iter(["c", "n"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    # Removes a value -> not a superset -> falls back to warn + overwrite/abort.
    second = make_runner({"param_a": ["ok"], "param_b": [1]})
    second.run()

    with open(agg_path, "r") as f:
        aggregated_after_abort = json.load(f)
    assert aggregated_after_abort == original_aggregated


#############################
# SeedRegistry / SweepSpace unit tests
#############################

def test_seed_registry_persists_and_reloads(tmp_path):
    path = os.path.join(tmp_path, "seed_groups.json")
    registry = SeedRegistry(path=path, base_seed=0, n_trials=5)

    seed_a = registry.get_or_assign((("param", "a"),))
    seed_b = registry.get_or_assign((("param", "b"),))
    registry.save()

    reloaded = SeedRegistry(path=path, base_seed=0, n_trials=5)
    assert reloaded.get_or_assign((("param", "a"),)) == seed_a
    assert reloaded.get_or_assign((("param", "b"),)) == seed_b

    # A newly seen group is appended without disturbing existing assignments.
    seed_c = reloaded.get_or_assign((("param", "c"),))
    assert seed_c not in {seed_a, seed_b}


def test_seed_key_stable_across_sweep_space_key_order():
    space_a = SweepSpace({"x": [1, 2], "y": ["p", "q"]})
    space_b = SweepSpace({"y": ["p", "q"], "x": [1, 2]})

    seed_vary_by_a = sorted(space_a.to_dict().keys())
    seed_vary_by_b = sorted(space_b.to_dict().keys())

    params = {"x": 1, "y": "p"}
    key_a = space_a.seed_key(params, seed_vary_by_a)
    key_b = space_b.seed_key(params, seed_vary_by_b)

    assert SeedRegistry._key_repr(key_a) == SeedRegistry._key_repr(key_b)


def test_retry_seed_does_not_collide_with_primary_blocks(tmp_path):
    n_trials = 5
    registry = SeedRegistry(path=os.path.join(tmp_path, "seed_groups.json"), base_seed=0, n_trials=n_trials)

    primary_seeds = set()
    for i in range(20):
        seed = registry.get_or_assign((("group", i),))
        primary_seeds.update(range(seed, seed + n_trials))

    retry_seeds = {
        registry.retry_seed((("group", i),), trial_idx, attempt)
        for i in range(20)
        for trial_idx in range(n_trials)
        for attempt in range(3)
    }

    assert primary_seeds.isdisjoint(retry_seeds)


def test_non_listed_exception_is_not_retried(tmp_path):
    runner = SweepRunner(
        experiment_name="test_exp",
        base_dir=str(tmp_path / "sweep_data"),
        sweep_space={"param_a": ["fail"], "param_b": [1]},
        run_trial_fn=dummy_run_trial_fn,
        n_trials=3,
        parallel=False,
        retry_on_exception=(ValueError,),
    )
    runner.run()

    (path,) = _result_files(runner.experiment_logger.dir)
    with open(path, "r") as f:
        res = json.load(f)
    assert len(res["exceptions"]) == runner.n_trials
    assert all(e["attempt"] == 0 and not e["retryable"] for e in res["exceptions"])
    assert all(entry["outcome"] == "exception" for entry in res["trial_outcomes"])


def test_synced_sweeps_retry_with_the_same_seed(tmp_path):
    runner = SweepRunner(
        experiment_name="test_exp",
        base_dir=str(tmp_path / "sweep_data"),
        sweep_space={"param_a": ["x"], "param_b": [1, 2]},
        run_trial_fn=primary_seed_fails_trial_fn,
        n_trials=2,
        seed_sync_by=["param_b"],
        parallel=False,
        retry_on_exception=(RuntimeError,),
    )
    runner.run()

    outcomes = []
    for path in _result_files(runner.experiment_logger.dir):
        with open(path, "r") as f:
            outcomes.append([entry["outcome"] for entry in json.load(f)["trial_outcomes"]])
    assert len(outcomes) == 2
    assert outcomes[0] == outcomes[1]
    assert all(o.startswith("seed=") for o in outcomes[0])

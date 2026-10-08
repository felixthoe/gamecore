# src/gamecore/sweep_runner/runner.py

import logging
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import shutil
import time
import traceback
from abc import ABC
from datetime import datetime

import numpy as np
from tqdm import tqdm

from ..utils.logger import DataLogger
from .seed_assignment import SeedRegistry
from .sweep_space import SweepSpace

LOGGER_NAME = "gamecore.sweep_runner"


def always_true(*_) -> bool:
    """
    Default `is_valid_sweep_fn`, top level so it is pickleable for parallel
    execution.
    """
    return True


def _init_worker_logging(log_path: str) -> None:
    """
    `multiprocessing.Pool` initializer: attaches a file handler to the
    sweep-runner logger in each freshly spawned worker process.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


class SweepRunner(ABC):
    """
    Orchestrates parameter sweeps over a grid of experiments.

    Handles resumable, parallel or sequential execution, deterministic and
    comparable seeding, retrying failed trials, and logging and aggregating
    results. See `sweep_space.SweepSpace` for sweep-grid logic and
    `seed_assignment.SeedRegistry` for seed bookkeeping.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (used to create the experiment directory).
    sweep_space : dict
        Dictionary with parameter names as keys and lists of values.
    run_trial_fn : callable
        A function that runs a single trial. Must accept parameters:
        (seed: int, sweep_params: dict, **kwargs) -> str
        and return a string indicating the outcome of the trial.
    n_trials : int, default=100
        Number of trials to run for each parameter configuration.
    is_valid_sweep_fn : callable, default=None
        Function that checks whether a parameter configuration is valid.
        Must accept a single parameter `sweep_params: dict` and return a bool.
    base_dir : str, default="data"
        Base directory in which the experiment directory is created.
    base_seed : int, default=0
        Seed offset for reproducibility.
    seed_sync_by : list, default=None
        Keys of the sweep space whose values should not affect the seed, so
        that sweeps differing only in these keys are comparable.
    parallel : bool, default=True
        Whether to run sweeps in parallel.
    max_workers : int, default=None
        Maximum number of worker processes to use for parallel execution.
        If None, uses all available CPU cores minus one.
    retry_on_exception : tuple[type[BaseException], ...], default=()
        Exception types after which a trial is retried with a new seed (e.g. rejection-sampling
        failures such as `FactorySamplingError`). Any other exception is recorded as outcome
        "exception" and not retried, so it cannot silently bias the sampled trials. Retry seeds
        depend only on the seed group, so sweeps synced by `seed_sync_by` retry with the same seed.
    max_retries_on_exception : int, default=10
        Maximum number of attempts for a trial if retryable exceptions occur.
    """

    INTERIM_AGGREGATION_INTERVAL = 30.0  # seconds between rewrites of the interim result files

    def __init__(
        self,
        experiment_name: str,
        sweep_space: dict,
        run_trial_fn: callable,
        n_trials: int = 100,
        is_valid_sweep_fn: callable = None,
        base_dir: str = "data",
        base_seed: int = 0,
        seed_sync_by: list = None,
        parallel: bool = True,
        max_workers: int = None,
        retry_on_exception: tuple[type[BaseException], ...] = (),
        max_retries_on_exception: int = 10,
    ):
        self.experiment_name = experiment_name
        self.sweep_space = SweepSpace(sweep_space)
        self.run_trial_fn = run_trial_fn
        self.n_trials = n_trials
        self.is_valid_sweep_fn = is_valid_sweep_fn or always_true
        self.base_dir = base_dir
        self.base_seed = base_seed
        if seed_sync_by is None:
            self.seed_sync_by = []
        elif isinstance(seed_sync_by, list):
            self.seed_sync_by = seed_sync_by
        else:
            self.seed_sync_by = [seed_sync_by]
        self.parallel = parallel
        self.all_stats = {}
        self.all_failures = {}
        self.total_sweeps = 0
        self.trial_kwargs = {}
        self.max_workers = max_workers or (multiprocessing.cpu_count() - 1)
        self.retry_on_exception = retry_on_exception
        self.max_retries_on_exception = max_retries_on_exception

    def _logger(self) -> logging.Logger:
        """
        Process-local sweep-runner logger. Fetched by name rather than stored
        as an attribute, so that a `SweepRunner` instance (and its attached
        file handler) never has to be pickled for dispatch to worker
        processes.
        """
        return logging.getLogger(LOGGER_NAME)

    def run(self, **trial_kwargs):
        """
        Run the full parameter sweep.

        Parameters
        ----------
        trial_kwargs : dict
            Additional keyword arguments forwarded to `run_trial_fn`.
        """
        mode = self._prepare_experiment_directory()
        if mode == "abort":
            return

        self.experiment_logger = DataLogger(base_dir=self.base_dir, folder_name=self.experiment_name)
        self._setup_file_logging()

        if mode == "continue":
            if self._check_sweep_space_consistency() == "abort":
                return
        if self._check_seed_sync_by_consistency() == "abort":
            return

        self.seed_registry = SeedRegistry(
            path=os.path.join(self.experiment_logger.dir, "seed_groups.json"),
            base_seed=self.base_seed,
            n_trials=self.n_trials,
        )
        sweep_args = self._create_sweep_args()
        self.seed_registry.save()
        self.trial_kwargs = trial_kwargs
        self.all_stats = {}
        self.all_failures = {}

        if self.parallel:
            print(f"\nRunning in parallel on {self.max_workers} cores...\n")
            # Spawned workers read these when importing numpy. Multithreaded BLAS on the small
            # matrices of a trial gives no speedup but oversubscribes the cores across workers.
            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            with multiprocessing.Pool(
                processes=self.max_workers,
                initializer=_init_worker_logging,
                initargs=(self._log_path,),
            ) as pool:
                self._collect_results(pool.imap_unordered(self._run_single_sweep, sweep_args), total=len(sweep_args))
        else:
            print("\nRunning sequentially ...\n")
            self._collect_results(map(self._run_single_sweep, sweep_args), total=len(sweep_args))

    def _collect_results(self, sweep_loggers, total: int) -> None:
        """
        Aggregate completed sweeps as they arrive. Each aggregation rewrites the interim files for
        all sweeps so far, so it runs at most every `INTERIM_AGGREGATION_INTERVAL` seconds, and
        once more as the final aggregation.

        Parameters
        ----------
        sweep_loggers : iterable of DataLogger | None
            Loggers of the completed sweeps, in completion order.
        total : int
            Number of sweeps, for the progress bar.
        """
        pending, last = [], time.perf_counter()
        for sweep_logger in tqdm(sweep_loggers, total=total):
            if sweep_logger is not None:
                pending.append(sweep_logger)
            if pending and time.perf_counter() - last >= self.INTERIM_AGGREGATION_INTERVAL:
                self._aggregate_results(pending, final=False)
                pending, last = [], time.perf_counter()
        self._aggregate_results(pending, final=True)

    def _prepare_experiment_directory(self) -> str:
        """
        Prepare the experiment directory and return the chosen mode.
        (One of "continue", "overwrite", "abort", or "new".)
        """
        experiment_folder = os.path.join(self.base_dir, self.experiment_name)

        if os.path.exists(experiment_folder):
            print(f"📁 Experiment directory already exists: {experiment_folder}\n")
            while True:
                answer = input("Choose action: [c]ontinue, [o]verwrite, [a]bort: ").strip().lower()
                print("")
                if answer in {"c", "continue"}:
                    print("✅ Continuing experiment. Existing data will be kept.")
                    return "continue"
                elif answer in {"o", "overwrite"}:
                    print("⚠️  Overwriting existing experiment directory.")
                    shutil.rmtree(experiment_folder)
                    os.makedirs(experiment_folder, exist_ok=True)
                    return "overwrite"
                elif answer in {"a", "abort"}:
                    print("❌ Aborting.")
                    return "abort"
                else:
                    print("❓ Invalid input. Please choose 'c', 'o', or 'a'.")
        else:
            print(f"📁 Creating new experiment directory: {experiment_folder}")
            os.makedirs(experiment_folder, exist_ok=True)
            return "new"

    def _setup_file_logging(self) -> None:
        """
        Attach a file handler (writing to `run.log` in the experiment
        directory, appended across resumes) to the sweep-runner logger in the
        main process.
        """
        self._log_path = os.path.join(self.experiment_logger.dir, "run.log")
        logger = self._logger()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        already_attached = any(
            isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == os.path.abspath(self._log_path)
            for h in logger.handlers
        )
        if not already_attached:
            handler = logging.FileHandler(self._log_path, mode="a")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)

    def _check_sweep_space_consistency(self) -> str:
        """
        Compare the current sweep space against the one recorded from a prior
        run (if any) and reconcile any difference.

        Returns
        -------
        str
            One of "new" (nothing recorded yet), "unchanged", "extend",
            "overwrite", or "abort".
        """
        if os.path.exists(os.path.join(self.experiment_logger.dir, "aggregated_stats_interim.json")):
            existing = self.experiment_logger.load_dict(name="aggregated_stats_interim")["sweep_space"]
        elif os.path.exists(os.path.join(self.experiment_logger.dir, "aggregated_stats.json")):
            existing = self.experiment_logger.load_dict(name="aggregated_stats")["sweep_space"]
        else:
            return "new"

        existing_space = SweepSpace(existing)
        if existing_space == self.sweep_space:
            return "unchanged"

        is_superset, added = self.sweep_space.is_superset_of(existing_space)
        if is_superset:
            print("\n📈 The sweep space has grown since the last run:")
            for key, values in added.items():
                print(f"  {key}: + {values}")
            while True:
                answer = input("Choose action: [e]xtend, [o]verwrite, [a]bort: ").strip().lower()
                if answer in {"e", "extend"}:
                    print("✅ Extending experiment with the enlarged sweep space.")
                    self._warn_if_previous_sweeps_now_invalid()
                    return "extend"
                elif answer in {"o", "overwrite"}:
                    print("⚠️  Overwriting existing experiment directory.")
                    shutil.rmtree(self.experiment_logger.dir)
                    os.makedirs(self.experiment_logger.dir, exist_ok=True)
                    return "overwrite"
                elif answer in {"a", "abort"}:
                    print("❌ Aborting.")
                    return "abort"
                else:
                    print("❓ Invalid input. Please choose 'e', 'o', or 'a'.")

        print("\n⚠️  Warning: The sweep space has changed since the last run (not a simple extension).")
        print("Existing sweep space:", existing_space.to_dict())
        print("New sweep space:", self.sweep_space.to_dict())
        while True:
            answer = input("Do you want to overwrite the existing experiment with the new sweep space? [y/n]: ").strip().lower()
            if answer in {"y", "yes"}:
                print("✅ Overwriting with new sweep space.")
                shutil.rmtree(self.experiment_logger.dir)
                os.makedirs(self.experiment_logger.dir, exist_ok=True)
                return "overwrite"
            elif answer in {"n", "no"}:
                print("❌ Aborting due to inconsistent sweep space.")
                return "abort"
            else:
                print("❓ Invalid input. Please enter 'y' or 'n'.")

    def _warn_if_previous_sweeps_now_invalid(self) -> None:
        """
        After extending the sweep space, check whether any previously
        completed sweep is no longer considered valid by `is_valid_sweep_fn`.
        Cannot be auto-reconciled, so this only warns.
        """
        sweeps_dir = os.path.join(self.experiment_logger.dir, "sweeps")
        if not os.path.exists(sweeps_dir):
            return

        invalid = []
        for name in os.listdir(sweeps_dir):
            params_path = os.path.join(sweeps_dir, name, "params.json")
            if not os.path.exists(params_path):
                continue
            sweep_logger = DataLogger(base_dir="", folder_name=os.path.join(sweeps_dir, name))
            params = sweep_logger.load_dict(name="params")
            if not self.is_valid_sweep_fn(params):
                invalid.append(name)

        if invalid:
            preview = invalid[:5] + (["..."] if len(invalid) > 5 else [])
            print(f"\n⚠️  {len(invalid)} previously completed sweep(s) are no longer valid under the new "
                  f"is_valid_sweep_fn: {preview}")
            self._logger().warning("Previously completed sweeps now invalid under new is_valid_sweep_fn: %s", invalid)

    def _check_seed_sync_by_consistency(self) -> str:
        """
        Check that every key in `seed_sync_by` exists in the sweep space. If
        not, warn the user and ask for confirmation to proceed.
        """
        for key in self.seed_sync_by:
            if key not in self.sweep_space.to_dict():
                print(f"\n⚠️  Warning: The key '{key}' in seed_sync_by is not found in the sweep space.")
                print("Available keys:", list(self.sweep_space.to_dict().keys()))
                while True:
                    answer = input("Do you want to ignore this key and proceed without its seed-synchronisation? [y/n]: ").strip().lower()
                    if answer in {"y", "yes"}:
                        print("✅ Proceeding without the key.")
                        return "continue"
                    elif answer in {"n", "no"}:
                        print("❌ Aborting due to inconsistent seed_sync_by.")
                        return "abort"
                    else:
                        print("❓ Invalid input. Please enter 'y' or 'n'.")
        return "ok"

    def _create_sweep_args(self) -> list[tuple]:
        """
        Build the list of (sweep_idx, sweep_params, sweep_seed, seed_key)
        tuples for every valid parameter combination, assigning each a
        deterministic seed via `self.seed_registry`.
        """
        print(f"\nAll parameter combinations: {self.sweep_space.size} (potentially contains invalid combinations)")
        valid_combinations = self.sweep_space.valid_combinations(self.is_valid_sweep_fn)
        self.total_sweeps = len(valid_combinations)
        print(f"Valid parameter combinations: {self.total_sweeps}")

        # Sorted so the seed key is independent of the sweep space dict's own
        # key insertion order, keeping seed-group assignment stable across
        # differently-constructed but semantically identical sweep spaces.
        seed_vary_by = sorted(k for k in self.sweep_space.to_dict().keys() if k not in self.seed_sync_by)

        sweep_args = []
        for sweep_idx, sweep_params in enumerate(valid_combinations):
            seed_key = self.sweep_space.seed_key(sweep_params, seed_vary_by)
            sweep_seed = self.seed_registry.get_or_assign(seed_key)
            sweep_args.append((sweep_idx, sweep_params, sweep_seed, seed_key))

        return sweep_args

    def _run_single_sweep(self, sweep_args: tuple):
        """
        Run a single sweep with the given parameters. May run in a worker
        process.

        Parameters
        ----------
        sweep_args : tuple
            Tuple of (sweep_idx, sweep_params, sweep_seed, seed_key).

        Returns
        -------
        DataLogger
            Logger for the sweep results.
        """
        sweep_idx, sweep_params, sweep_seed, seed_key = sweep_args
        sweep_logger, sweep_name = self._prepare_sweep_logger(sweep_idx, sweep_params)
        if sweep_name is None:
            return sweep_logger

        sweep_hash = os.path.basename(sweep_logger.dir).removeprefix("sweep_")
        sweep_stats, trial_outcomes, trial_durations, exceptions = self._run_all_trials(
            sweep_idx, sweep_hash, sweep_params, sweep_seed, seed_key
        )

        self._save_sweep_result(sweep_logger, sweep_idx, sweep_stats, trial_outcomes, trial_durations, exceptions)

        if not self.parallel:
            self._print_sweep_summary(sweep_name, sweep_stats, trial_durations)
            print("Current progress:")

        return sweep_logger

    def _prepare_sweep_logger(self, sweep_idx: int, sweep_params: dict) -> tuple:
        """
        Check whether the sweep already exists and return its logger.
        If it exists and is complete, returns (logger, None) to signal skip.
        If it exists but is incomplete, removes it and restarts from trial 0.
        """
        sweep_hash = self.sweep_space.content_hash(sweep_params)
        sweep_name = f"sweep_{sweep_hash}"
        sweep_path = os.path.join(self.experiment_logger.dir, "sweeps", sweep_name)

        if os.path.exists(sweep_path):
            if os.path.exists(os.path.join(sweep_path, "result.json")):
                print(f"✅  Skipping completed sweep: {sweep_name}")
                sweep_logger = DataLogger(base_dir="", folder_name=sweep_path)
                return sweep_logger, None
            else:
                print(f"🧹 Incomplete sweep found: {sweep_name} — removing and restarting.")
                shutil.rmtree(sweep_path)

        if not self.parallel:
            print(f"\n\n=== Running sweep {sweep_idx + 1}/{self.total_sweeps}: {sweep_name} ===")
            print(f"Parameters: {sweep_params}")

        sweep_logger = DataLogger(base_dir="", folder_name=sweep_path)
        sweep_logger.log_dict(name="params", data=sweep_params)

        return sweep_logger, sweep_name

    def _run_all_trials(self, sweep_idx: int, sweep_hash: str, sweep_params: dict, sweep_seed: int, seed_key: tuple) -> tuple:
        """
        Run all trials for a given sweep and collect statistics. A trial is
        retried with a new seed only after an exception of a type listed in
        `retry_on_exception`.

        Returns
        -------
        tuple
            (sweep_stats, trial_outcomes, trial_durations, exceptions).
            `trial_outcomes` holds one entry per trial (its final outcome);
            `exceptions` holds one entry per *failed attempt*, including ones
            that were later retried successfully.
        """
        sweep_stats = {}
        trial_outcomes = []
        trial_durations = []
        exceptions = []

        for trial_idx in range(self.n_trials):
            seed = sweep_seed + trial_idx
            attempts = self.max_retries_on_exception

            outcome, duration, exception_info = None, None, None
            for attempt in range(attempts):
                outcome, duration, exception_info = self._run_single_trial(seed, sweep_params)
                if outcome != "exception":
                    break
                if not exception_info["retryable"]:
                    exceptions.append({
                        "trial": trial_idx,
                        "attempt": attempt,
                        "seed": seed,
                        "timestamp": datetime.now().isoformat(),
                        **exception_info,
                    })
                    self._logger().error(
                        "sweep=%s trial=%d seed=%d: non-retryable %s",
                        sweep_hash, trial_idx + 1, seed, exception_info["exception"],
                    )
                    break

                exceptions.append({
                    "trial": trial_idx,
                    "attempt": attempt,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    **exception_info,
                })
                if attempt + 1 < attempts:
                    if not self.parallel:
                        print(f"⚠️  Exception in sweep {sweep_idx+1}, trial {trial_idx+1} "
                              f"(attempt {attempt+1}/{attempts}). Retrying with new seed.")
                    self._logger().warning(
                        "sweep=%s trial=%d attempt=%d/%d seed=%d: %s — retrying with new seed",
                        sweep_hash, trial_idx + 1, attempt + 1, attempts, seed, exception_info["exception"],
                    )
                    seed = self.seed_registry.retry_seed(seed_key, trial_idx, attempt)
                else:
                    if not self.parallel:
                        print(f"❌ Max retries reached for sweep {sweep_idx+1}, trial {trial_idx+1}. "
                              f"Logging exception and moving on.")
                    self._logger().error(
                        "sweep=%s trial=%d: exhausted %d attempt(s), giving up. Last error: %s",
                        sweep_hash, trial_idx + 1, attempts, exception_info["exception"],
                    )

            if duration is not None:
                trial_durations.append(duration)

            sweep_stats[outcome] = sweep_stats.get(outcome, 0) + 1

            entry = {"trial": trial_idx, "seed": seed, "outcome": outcome, "duration": duration}
            if exception_info:
                entry.update(exception_info)
            trial_outcomes.append(entry)

            if not self.parallel:
                print(f"Sweep {sweep_idx+1:03}/{self.total_sweeps:03} - Trial {trial_idx + 1:03}/{self.n_trials:03} - ", end="")
                print("Status: " + ", ".join(f"{k}={v}" for k, v in sweep_stats.items()), end="\r")

        return sweep_stats, trial_outcomes, trial_durations, exceptions

    def _run_single_trial(self, seed: int, sweep_params: dict) -> tuple:
        """
        Run a single trial.

        Returns
        -------
        tuple
            (outcome, duration, exception_info).
        """
        try:
            start_time = time.perf_counter()
            outcome = self.run_trial_fn(seed, sweep_params, **self.trial_kwargs)
            duration = (time.perf_counter() - start_time) / 1e-3  # milliseconds
            return outcome, duration, None
        except Exception as e:
            return "exception", None, {
                "exception": str(e),
                "traceback": traceback.format_exc(),
                "retryable": isinstance(e, self.retry_on_exception),
            }

    def _save_sweep_result(
        self,
        sweep_logger: DataLogger,
        sweep_idx: int,
        sweep_stats: dict,
        trial_outcomes: list[dict],
        trial_durations: list[float],
        exceptions: list[dict],
    ) -> None:
        """
        Save the results of a sweep: statistics, timestamp, durations, trial
        outcomes, and every failed attempt.
        """
        duration_stats = {
            "mean_duration": float(np.mean(trial_durations)) if trial_durations else 0.0,
            "std_duration": float(np.std(trial_durations)) if trial_durations else 0.0,
            "unit": "milliseconds",
        }

        sweep_logger.log_dict(name="result", data={
            "sweep_idx": sweep_idx,
            "stats_rel": {k: f"{v/self.n_trials*100}%" for k, v in sweep_stats.items()},
            "stats_abs": sweep_stats,
            "timestamp": datetime.now().isoformat(),
            "duration_stats": duration_stats,
            "trial_outcomes": trial_outcomes,
            "exceptions": exceptions,
        })

    def _print_sweep_summary(self, sweep_name: str, sweep_stats: dict, trial_durations: list[float]) -> None:
        """Print a summary of the sweep results to the console."""
        print(f"\n=== Results for {sweep_name} ===")
        for k, v in sweep_stats.items():
            print(f"{k}: {v}")
        if trial_durations:
            mean = np.mean(trial_durations)
            std = np.std(trial_durations)
            print(f"Mean trial time: {mean:.3f}ms ± {std:.3f}ms\n")

    def _aggregate_results(self, new_sweep_loggers: list[DataLogger], final: bool = False) -> None:
        """
        Fold newly completed sweeps into the running aggregate and save it.
        `self.all_stats`/`self.all_failures` accumulate in memory across
        calls, so each sweep's `result.json` is read exactly once.

        Parameters
        ----------
        new_sweep_loggers : list[DataLogger]
            Loggers of sweeps completed since the previous call.
        final : bool, default=False
            Whether this is the final aggregation, printing a summary and
            replacing the interim files with the final ones.
        """
        for sweep_logger in new_sweep_loggers:
            sweep_name = os.path.basename(sweep_logger.dir)
            result = sweep_logger.load_dict(name="result")
            self.all_stats[sweep_name] = result["stats_abs"]
            if result.get("exceptions"):
                self.all_failures[sweep_name] = result["exceptions"]

        all_possible_outcomes = set()
        for stats in self.all_stats.values():
            all_possible_outcomes.update(stats.keys())
        aggregated_stats_abs = {
            k: sum(stats.get(k, 0) for stats in self.all_stats.values())
            for k in all_possible_outcomes
        }
        total_trials = sum(aggregated_stats_abs.values())
        aggregated_stats_rel = {
            k: f"{v/total_trials*100:.2f}%" for k, v in aggregated_stats_abs.items()
        } if total_trials else {}

        outcome_to_sweeps = {k: [] for k in aggregated_stats_abs}
        for sweep_name, stats in self.all_stats.items():
            for outcome, count in stats.items():
                if count > 0:
                    outcome_to_sweeps[outcome].append(sweep_name)

        total_exceptions = sum(len(entries) for entries in self.all_failures.values())
        failures_flat = [
            {"sweep": sweep_name, **entry}
            for sweep_name, entries in self.all_failures.items()
            for entry in entries
        ]

        if final:
            print("\n=== All sweeps completed! ===")
            print("\nAggregated absolute statistics:")
            for k, v in aggregated_stats_abs.items():
                print(f"- {k}: {v}")
            print("\nAggregated relative statistics:")
            for k, v in aggregated_stats_rel.items():
                print(f"- {k}: {v}")
            if total_exceptions:
                print(f"\n⚠️  {total_exceptions} exception event(s) across {len(self.all_failures)} sweep(s) "
                      f"— see failures.json / run.log for details.")

            for name in ("aggregated_stats_interim", "outcome_to_sweeps_interim", "all_sweeps_interim", "failures_interim"):
                path = os.path.join(self.experiment_logger.dir, f"{name}.json")
                if os.path.exists(path):
                    os.remove(path)

        data = {}
        if not final:
            data["completed_sweeps"] = len(self.all_stats)
        data.update({
            "total_valid_sweeps": self.total_sweeps,
            "total_trials": self.total_sweeps * self.n_trials,
            "stats_rel": aggregated_stats_rel,
            "stats_abs": aggregated_stats_abs,
            "total_exceptions": total_exceptions,
            "sweeps_with_exceptions": len(self.all_failures),
            "timestamp": datetime.now().isoformat(),
            "seed_sync_by": self.seed_sync_by,
            "sweep_space": self.sweep_space.to_dict(),
            "trial_kwargs": self.trial_kwargs,
        })

        suffix = "" if final else "_interim"
        self.experiment_logger.log_dict(name=f"aggregated_stats{suffix}", data=data)
        self.experiment_logger.log_dict(name=f"outcome_to_sweeps{suffix}", data=outcome_to_sweeps)
        self.experiment_logger.log_dict(name=f"all_sweeps{suffix}", data=self.all_stats)
        self.experiment_logger.log_dict(name=f"failures{suffix}", data=failures_flat)

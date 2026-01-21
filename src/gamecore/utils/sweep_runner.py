# src/gamecore/utils/sweep_runner.py

import itertools
import os
import shutil
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import traceback
from datetime import datetime
from abc import ABC
from tqdm import tqdm
import time
import numpy as np

from .logger import DataLogger

def always_true(*_):
    """
    Top level function as default for is_valid_sweep_fn of SweepRunner, 
    as it needs to be pickleable for parallel execution.
    """
    return True

class SweepRunner(ABC):
    """
    A class for orchestrating parameter sweeps over a grid of experiments.

    This runner handles both parallel and sequential execution, logging,
    and result aggregation.

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
        Must accept a single parameter `sweep_params: dict` and return a boolean.
    base_seed : int, default=0
        Seed offset for reproducibility.
    seed_sync_by : list, default=None
        List of keys of the sweep space to specify sweep parameters that should be 
        synchronized in their seeds, in order to ensure comparability.
    parallel : bool, default=True
        Whether to run sweeps in parallel.
    max_workers : int, default=None
        Maximum number of worker processes to use for parallel execution.
        If None, uses all available CPU cores.
    retry_on_exception : bool, default=True
        Whether to retry a trial with a new seed if an exception occurs.
    max_retries_on_exception : int, default=10
        Maximum number of retries for a trial if exceptions occur.
    """

    def __init__(
        self,
        experiment_name: str,
        sweep_space: dict,
        run_trial_fn: callable,
        n_trials: int = 100,
        is_valid_sweep_fn: callable = None,
        base_dir: str = "data",
        base_seed: int = 0,
        seed_sync_by: list = [],
        parallel: bool = True,
        max_workers: int = None,
        retry_on_exception: bool = True,
        max_retries_on_exception: int = 10,
    ):
        self.experiment_name = experiment_name
        self.sweep_space = self._normalize_sweep_space(sweep_space)
        self.run_trial_fn = run_trial_fn
        self.n_trials = n_trials
        self.is_valid_sweep_fn = is_valid_sweep_fn or always_true
        self.base_dir = base_dir
        self.base_seed = base_seed
        self.seed_sync_by = seed_sync_by
        self.parallel = parallel
        self.all_stats = {}
        self.total_sweeps = 0
        self.trial_kwargs = {}
        self.max_workers = max_workers or (multiprocessing.cpu_count()-1)
        self.retry_on_exception = retry_on_exception
        self.max_retries_on_exception = max_retries_on_exception

    def _normalize_sweep_space(self, sweep_space: dict) -> dict:
        """
        Normalize sweep space by converting all tuples and sets to lists recursively.
        This ensures consistency regardless of list/tuple/set differences, 
        which are saved equally in JSON.
        """
        def tuple_to_list(obj):
            if isinstance(obj, (tuple, list, set)):
                return [tuple_to_list(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: tuple_to_list(v) for k, v in obj.items()}
            else:
                return obj

        return tuple_to_list(sweep_space)
        
    def run(self, **trial_kwargs):
        """
        Run the full parameter sweep.

        Parameters
        ----------
        trial_kwargs : dict
            Additional keyword arguments for the trial function 'run_trial_fn'.
        """
        mode = self._prepare_experiment_directory()
        if mode == "abort":
            return

        self.experiment_logger = DataLogger(base_dir=self.base_dir, folder_name=self.experiment_name)
        if mode == "continue":
            if self._check_sweep_space_consistency() == "abort":
                return
        if self._check_seed_sync_by_consistency() == "abort":
            return
        sweep_args = self._create_sweep_args()
        self.trial_kwargs = trial_kwargs
        sweep_loggers = []

        if self.parallel:
            print(f"\nRunning in parallel on {self.max_workers} cores...\n")
            with multiprocessing.Pool(processes=self.max_workers) as pool:
                for sweep_logger in tqdm(pool.imap_unordered(self._run_single_sweep, sweep_args), total=len(sweep_args)):
                    if sweep_logger is not None:
                        sweep_loggers.append(sweep_logger)
                        self._aggregate_results(sweep_loggers, final=False)
        else:
            print("\nRunning sequentially ...\n")
            for args in tqdm(sweep_args):
                sweep_logger = self._run_single_sweep(args)
                if sweep_logger is not None:
                    sweep_loggers.append(sweep_logger)
                    self._aggregate_results(sweep_loggers, final=False)

        self._aggregate_results(sweep_loggers, final=True)
    
    def _prepare_experiment_directory(self) -> str:
        """
        Prepares the experiment directory and returns chosen mode.
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
    
    def _check_sweep_space_consistency(self) -> str:
        """
        Check if the sweep space has changed since the last run.
        If it has, warn the user and ask for confirmation to proceed.
        """
        if os.path.exists(os.path.join(self.experiment_logger.dir, "aggregated_stats_interim.json")):
            existing_sweep_space = self.experiment_logger.load_dict(name="aggregated_stats_interim")["sweep_space"]
        elif os.path.exists(os.path.join(self.experiment_logger.dir, "aggregated_stats.json")):
            existing_sweep_space = self.experiment_logger.load_dict(name="aggregated_stats")["sweep_space"]
        else:
            return "new"
        if existing_sweep_space != self.sweep_space:
            print("\n⚠️  Warning: The sweep space has changed since the last run.")
            print("Existing sweep space:", existing_sweep_space)
            print("New sweep space:", self.sweep_space)
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
    
    def _check_seed_sync_by_consistency(self) -> str:
        """
        Check if the seed_sync_by parameters are consistent with the keys of the 
        sweep space. If not, warn the user and ask for confirmation to proceed.
        """
        if not isinstance(self.seed_sync_by, list):
            self.seed_sync_by = [self.seed_sync_by]
        for key in self.seed_sync_by:
            if key not in self.sweep_space:
                print(f"\n⚠️  Warning: The key '{key}' in seed_sync_by is not found in the sweep space.")
                print("Available keys:", list(self.sweep_space.keys()))
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

    def _create_sweep_args(self):
        """
        Create a list of arguments for each sweep based on the parameter space.
        Excludes invalid parameter combinations specified by `is_valid_sweep_fn`.
        """
        keys, values = zip(*self.sweep_space.items())
        all_param_combinations = list(itertools.product(*values))
        print(f"\nAll parameter combinations: {len(all_param_combinations)} (potentially contain invalid combinations)")

        valid_combinations = [dict(zip(keys, combo))
                              for combo in all_param_combinations
                              if self.is_valid_sweep_fn(dict(zip(keys, combo)))]
        self.total_sweeps = len(valid_combinations)
        print(f"Valid parameter combinations: {self.total_sweeps}")

        # Determine which parameters affect the seed
        seed_vary_by = [k for k in self.sweep_space.keys() if k not in self.seed_sync_by]

        # Assign deterministic seeds to each unique group over seed_vary_by
        unique_seed_groups = {}
        sweep_args = []
        next_seed_id = 0

        for sweep_idx, sweep_params in enumerate(valid_combinations):
            seed_key = tuple((k, self._make_hashable(sweep_params[k])) for k in seed_vary_by)

            if seed_key not in unique_seed_groups:
                sweep_seed = self.base_seed + next_seed_id * self.n_trials
                unique_seed_groups[seed_key] = sweep_seed
                next_seed_id += 1
            else:
                sweep_seed = unique_seed_groups[seed_key]

            sweep_args.append((sweep_idx, sweep_params, sweep_seed))

        return sweep_args
    
    def _make_hashable(self, value):
        """Recursively convert lists/dicts into hashable types (tuples/frozensets)."""
        if isinstance(value, list):
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, dict):
            return frozenset((k, self._make_hashable(v)) for k, v in value.items())
        return value

    def _run_single_sweep(self, sweep_args: tuple) -> str:
        """
        Run a single sweep with the given parameters.
        Might be run in parallel.

        Parameters
        ----------
        sweep_args : tuple
            Tuple of (sweep_idx, sweep_params, sweep_seed)

        Returns
        -------
        DataLogger
            Logger for the sweep results.
        """
        sweep_idx, sweep_params, sweep_seed = sweep_args
        sweep_logger, sweep_name = self._prepare_sweep_logger(sweep_idx, sweep_params)
        if sweep_name is None:
            return sweep_logger

        sweep_stats, trial_outcomes, trial_durations = self._run_all_trials(
            sweep_idx, sweep_params, sweep_seed
        )

        self._save_sweep_result(
            sweep_logger, sweep_stats, trial_outcomes, trial_durations
        )

        if not self.parallel:
            self._print_sweep_summary(sweep_name, sweep_stats, trial_durations)
            print(f"Current progress:")

        return sweep_logger
    
    def _prepare_sweep_logger(self, sweep_idx: int, sweep_params: dict) -> tuple:
        """
        Checks if the sweep already exists and return the logger for it.
        If it exists and is complete, it returns the logger and None.
        If it exists but is incomplete, it removes it and returns the logger and sweep name.
        """
        sweep_name = f"sweep_{sweep_idx+1:03d}"
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
    
    def _run_all_trials(self, sweep_idx: int, sweep_params: dict, sweep_seed: int) -> tuple:
        """
        Run all trials for a given sweep and collect statistics.
        Returns a tuple of (sweep_stats, trial_outcomes, trial_durations).
        """
        sweep_stats = {}
        trial_outcomes = []
        trial_durations = []

        for trial_idx in range(self.n_trials):
            seed = sweep_seed + trial_idx
            if self.retry_on_exception:
                for attempt in range(self.max_retries_on_exception):
                    outcome, duration, exception_info = self._run_single_trial(
                        seed, sweep_params
                    )
                    if outcome != "exception":
                        break
                    else:
                        print(f"⚠️  Exception in sweep {sweep_idx+1}, trial {trial_idx+1} (attempt {attempt+1}/{self.max_retries_on_exception}). Retrying with new seed.")
                        seed += self.n_trials  # Increment seed to avoid repeated exceptions
                else:
                    print(f"❌ Max retries reached for sweep {sweep_idx+1}, trial {trial_idx+1}. Logging exception and moving on.")
            else:
                outcome, duration, exception_info = self._run_single_trial(
                    seed, sweep_params
                )
            if duration is not None:
                trial_durations.append(duration)

            if outcome not in sweep_stats:
                sweep_stats[outcome] = 0
            sweep_stats[outcome] += 1

            entry = {"trial": trial_idx, "seed": seed, "outcome": outcome, "duration": duration}
            if exception_info:
                entry.update(exception_info)
            trial_outcomes.append(entry)

            if not self.parallel:
                print(f"Sweep {sweep_idx+1:03}/{self.total_sweeps:03} - Trial {trial_idx + 1:03}/{self.n_trials:03} - ", end="")
                print(f"Status: " + ", ".join(f"{k}={v}" for k, v in sweep_stats.items()), end="\r")

        return sweep_stats, trial_outcomes, trial_durations

    def _run_single_trial(self, seed: int, sweep_params: dict) -> tuple:
        """
        Run a single trial and log its outcome.
        Returns a tuple of (outcome, duration, exception_info).
        """ 
        try:
            start_time = time.perf_counter()
            outcome = self.run_trial_fn(seed, sweep_params, **self.trial_kwargs)
            duration = (time.perf_counter() - start_time)/1e-3  # Convert to milliseconds
            return outcome, duration, None
        except Exception as e:
            return "exception", None, {
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        
    def _save_sweep_result(self, sweep_logger: DataLogger, sweep_stats: dict, trial_outcomes: list[dict], trial_durations: list[float]):
        """
        Save the results of a sweep.
        Includes statistics, timestamp, durations and trial outcomes.
        """
        duration_stats = {
            "mean_duration": float(np.mean(trial_durations)) if trial_durations else 0.0,
            "std_duration": float(np.std(trial_durations)) if trial_durations else 0.0,
            "unit": "milliseconds",
        }

        sweep_logger.log_dict(name="result", data={
                "stats_rel": {k: f"{v/self.n_trials*100}%" for k, v in sweep_stats.items()},
                "stats_abs": sweep_stats,
                "timestamp": datetime.now().isoformat(),
                "duration_stats": duration_stats,
                "trial_outcomes": trial_outcomes,
            })

    def _print_sweep_summary(self, sweep_name: str, sweep_stats: dict, trial_durations: list[float]):
        """
        Print a summary of the sweep results to the console.
        """
        print(f"\n=== Results for {sweep_name} ===")
        for k, v in sweep_stats.items():
            print(f"{k}: {v}")
        if trial_durations:
            mean = np.mean(trial_durations)
            std = np.std(trial_durations)
            print(f"Mean trial time: {mean:.3f}ms ± {std:.3f}ms\n")

    def _aggregate_results(self, sweep_loggers: list[DataLogger], final: bool = True):
        """
        Aggregate results from all completed sweeps and save them to a summary file.
        Saves interim results if 'final' is False, which will be deleted after the final sweep.
        """
        for sweep_logger in sweep_loggers:
            sweep_name = os.path.basename(sweep_logger.dir)
            self.all_stats[sweep_name] = sweep_logger.load_dict(name="result")["stats_abs"]
        
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
        }

        outcome_to_sweeps = {k: [] for k in aggregated_stats_abs}
        for sweep_name, stats in self.all_stats.items():
            for outcome, count in stats.items():
                if count > 0:
                    outcome_to_sweeps[outcome].append(sweep_name)

        if final:
            print("\n=== All sweeps completed! ===")
            print("\nAggregated absolute statistics:")
            for k, v in aggregated_stats_abs.items():
                print(f"- {k}: {v}")
            print("\nAggregated relative statistics:")
            for k, v in aggregated_stats_rel.items():
                print(f"- {k}: {v}")

        # Remove interim files if this is the final run
        if final:
            if os.path.exists(os.path.join(self.experiment_logger.dir, "aggregated_stats_interim.json")):
                os.remove(os.path.join(self.experiment_logger.dir, "aggregated_stats_interim.json"))
            if os.path.exists(os.path.join(self.experiment_logger.dir, "outcome_to_sweeps_interim.json")):
                os.remove(os.path.join(self.experiment_logger.dir, "outcome_to_sweeps_interim.json"))
            if os.path.exists(os.path.join(self.experiment_logger.dir, "all_sweeps_interim.json")):
                os.remove(os.path.join(self.experiment_logger.dir, "all_sweeps_interim.json"))
    
        data = {}
        if not final:
            data["completed_sweeps"] = len(self.all_stats)
        data.update({
            "total_valid_sweeps": self.total_sweeps,
            "total_trials": self.total_sweeps * self.n_trials,
            "stats_rel": aggregated_stats_rel,
            "stats_abs": aggregated_stats_abs,
            "timestamp": datetime.now().isoformat(),
            "seed_sync_by": self.seed_sync_by,
            "sweep_space": self.sweep_space,
            "trial_kwargs": self.trial_kwargs,
        })

        self.experiment_logger.log_dict(name="aggregated_stats" if final else "aggregated_stats_interim", data=data)
        self.experiment_logger.log_dict(name="outcome_to_sweeps" if final else "outcome_to_sweeps_interim", data=outcome_to_sweeps)
        self.experiment_logger.log_dict(name="all_sweeps" if final else "all_sweeps_interim", data=self.all_stats)
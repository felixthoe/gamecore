# src/gamecore/sweep_runner/seed_assignment.py

import hashlib
import json
import os


class SeedRegistry:
    """
    Persisted, append-only assignment of deterministic seeds to seed groups.

    Loads an existing seed-group file from `path` if present, so seed
    assignment survives resumes and sweep-space extensions: a seed group that
    has already been assigned a seed keeps it forever, regardless of the
    enumeration order of any later run.

    Parameters
    ----------
    path : str
        Path to the seed-group JSON file (inside the experiment directory).
    base_seed : int
        Seed offset for reproducibility.
    n_trials : int
        Number of trials per sweep; size of each seed block.
    """

    def __init__(self, path: str, base_seed: int, n_trials: int):
        self.path = path
        self.base_seed = base_seed
        self.n_trials = n_trials
        self._groups: dict[str, int] = {}
        self._next_id = 0
        if os.path.exists(path):
            with open(path, "r") as f:
                saved = json.load(f)
            self._groups = saved["groups"]
            self._next_id = saved["next_id"]

    @staticmethod
    def _key_repr(seed_key: tuple) -> str:
        """
        Canonical string representation of a seed key, for JSON persistence.
        `seed_key` must already be JSON-safe (see `SweepSpace.seed_key`);
        `sort_keys` canonicalizes any nested dict-valued parameter so the
        representation is stable across Python processes.
        """
        return json.dumps(seed_key, sort_keys=True)

    def get_or_assign(self, seed_key: tuple) -> int:
        """
        Return the seed block start for `seed_key`, assigning a new one the
        first time this key is seen. Does not persist by itself; call `save`
        once all groups for a run have been assigned.

        Parameters
        ----------
        seed_key : tuple
            Hashable seed-group key, as produced by `SweepSpace.seed_key`.

        Returns
        -------
        int
            Start of this group's `n_trials`-sized seed block.
        """
        key_repr = self._key_repr(seed_key)
        if key_repr not in self._groups:
            self._groups[key_repr] = self.base_seed + self._next_id * self.n_trials
            self._next_id += 1
        return self._groups[key_repr]

    def retry_seed(self, sweep_hash: str, trial_idx: int, attempt: int) -> int:
        """
        Deterministic seed for a retry attempt, derived from a hash of
        `(sweep_hash, trial_idx, attempt)` and offset into a range far above
        any primary seed block, so it cannot collide with another sweep's
        seeds. Needs no shared mutable counter, so it stays safe to call
        independently from parallel worker processes.

        Parameters
        ----------
        sweep_hash : str
            Content hash of the sweep this trial belongs to.
        trial_idx : int
            Index of the trial within its sweep.
        attempt : int
            Retry attempt number (0-based).

        Returns
        -------
        int
            Seed to use for this retry attempt.
        """
        digest = hashlib.blake2b(f"{sweep_hash}:{trial_idx}:{attempt}".encode(), digest_size=8).digest()
        offset = int.from_bytes(digest, "big") % (2**31)
        return self.base_seed + 2**31 + offset

    def save(self) -> None:
        """Persist the current seed-group assignment to disk."""
        with open(self.path, "w") as f:
            json.dump({"groups": self._groups, "next_id": self._next_id}, f, indent=2)

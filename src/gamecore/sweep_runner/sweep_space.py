# src/gamecore/sweep_runner/sweep_space.py

import hashlib
import itertools
import json
import math


class SweepSpace:
    """
    A parameter sweep grid: normalization, enumeration of valid combinations,
    seed-grouping keys, and content-addressed identity of combinations.

    Parameters
    ----------
    sweep_space : dict
        Dictionary with parameter names as keys and lists of values.
    """

    def __init__(self, sweep_space: dict):
        self.space = self._normalize(sweep_space)

    @staticmethod
    def _normalize(sweep_space: dict) -> dict:
        """
        Recursively convert tuples/sets to lists, so that a freshly constructed
        sweep space compares equal to one that has been through a JSON
        save/load round trip.
        """
        def convert(obj):
            if isinstance(obj, (tuple, list, set)):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(sweep_space)

    @property
    def size(self) -> int:
        """
        Total number of parameter combinations before filtering by
        `is_valid_sweep_fn`.
        """
        return math.prod(len(v) for v in self.space.values())

    def valid_combinations(self, is_valid_sweep_fn: callable) -> list[dict]:
        """
        Enumerate all parameter combinations that pass `is_valid_sweep_fn`.

        Parameters
        ----------
        is_valid_sweep_fn : callable
            Function `sweep_params: dict -> bool`.

        Returns
        -------
        list[dict]
            Valid parameter combinations, in `itertools.product` order.
        """
        keys, values = zip(*self.space.items())
        combos = (dict(zip(keys, combo)) for combo in itertools.product(*values))
        return [params for params in combos if is_valid_sweep_fn(params)]

    def seed_key(self, params: dict, seed_vary_by: list[str]) -> tuple:
        """
        Build a JSON-safe key identifying the seed group of `params`,
        restricted to the subset of keys in `seed_vary_by`. Kept JSON-safe
        (rather than converted to a hashable tuple/frozenset structure) so it
        can be serialized to a stable, canonical string by `SeedRegistry`
        regardless of Python's per-process hash randomization.

        Parameters
        ----------
        params : dict
            A single parameter combination.
        seed_vary_by : list[str]
            Sweep-space keys allowed to vary the seed.

        Returns
        -------
        tuple
            Seed-group key, as a tuple of (key, value) pairs.
        """
        return tuple((k, params[k]) for k in seed_vary_by)

    @staticmethod
    def content_hash(params: dict, length: int = 10) -> str:
        """
        Stable short hash identifying a parameter combination, used as a sweep
        folder name so identity survives sweep-space changes.

        Parameters
        ----------
        params : dict
            A single parameter combination.
        length : int, default=10
            Number of hex characters to keep.

        Returns
        -------
        str
            Hex digest truncated to `length` characters.
        """
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()[:length]

    def is_superset_of(self, other: "SweepSpace") -> tuple[bool, dict]:
        """
        Check whether `self` is a valid extension of `other`: same parameter
        keys, and each key's value list a superset of `other`'s.

        Parameters
        ----------
        other : SweepSpace
            The previous sweep space to compare against.

        Returns
        -------
        tuple[bool, dict]
            Whether `self` is a valid superset, and a dict of newly added
            values per key (only meaningful when the first element is True).
        """
        if set(self.space.keys()) != set(other.space.keys()):
            return False, {}

        added = {}
        for key, values in self.space.items():
            old_values = other.space[key]
            if any(v not in values for v in old_values):
                return False, {}
            new_values = [v for v in values if v not in old_values]
            if new_values:
                added[key] = new_values

        return True, added

    def to_dict(self) -> dict:
        """Return the normalized sweep space as a plain dict."""
        return self.space

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SweepSpace):
            return self.space == other.space
        return NotImplemented

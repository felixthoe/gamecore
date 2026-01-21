# src/gamecore/utils/logger.py

import json
import os
import shutil
from datetime import datetime
import numpy as np

class DataLogger:
    """
    Class for organized logging and loading of data.
    """

    def __init__(
            self, 
            base_dir: str = "data", 
            folder_name: str = None, 
            overwrite: bool = False
        ):
        """
        Initialize a new DataLogger for organized logging and loading of data.

        If the experiment directory already exists and `overwrite` is False, 
        existing metadata is loaded from `metadata.json`. Otherwise, a new directory 
        is created (optionally overwriting the old one), and metadata is initialized empty.

        Parameters
        ----------
        base_dir : str, optional
            Base directory where folders are stored (default is "data").
        folder_name : str, optional
            Name of the directory to store results. If None, a timestamp-based name is generated.
        overwrite : bool, optional
            Whether to overwrite an existing experiment directory if it exists (default is False).

        Attributes
        ----------
        folder_name : str
            Name of the experiment (either user-provided or timestamp-based).
        dir : str
            Full path to the experiment directory.
        meta : dict
            Dictionary containing metadata entries. Initialized from disk if present and not overwritten.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = folder_name or f"experiment_{timestamp}"
        self.dir = os.path.join(base_dir, self.folder_name)
        if os.path.exists(self.dir):
            if overwrite:
                shutil.rmtree(self.dir)
                os.makedirs(self.dir)
                self.meta = {}
            else:
                try:
                    self.meta = self.load_metadata()
                except FileNotFoundError:
                    self.meta = {}
        else:
            os.makedirs(self.dir)
            self.meta = {}

    ### Logging and Loading ###

    def log_metadata(self, meta_dict: dict):
        """
        Log metadata entries to the experiment directory.
        If metadata already exists, it is updated with the new entries.
        """
        self.meta.update(meta_dict)
        with open(os.path.join(self.dir, "metadata.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def load_metadata(self) -> dict:
        path = os.path.join(self.dir, "metadata.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found in {self.dir}")

    def load_metadata_entry(self, key: str, default=None):
        meta = self.load_metadata()
        entry = meta.get(key, default)
        if entry is not None:
            return entry
        else:
            raise KeyError(f"Metadata entry '{key}' not found in {self.dir}")
        
    def log_dict(self, name: str, data: dict):
        path = os.path.join(self.dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_dict(self, name: str) -> dict:
        path = os.path.join(self.dir, f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dict file '{name}.json' not found in {self.dir}")
        with open(path, "r") as f:
            return json.load(f)

    def log_array(self, name: str, array: np.ndarray):
        path = os.path.join(self.dir, f"{name}.npy")
        np.save(path, array)
        
    def load_array(self, name: str) -> np.ndarray:
        path = os.path.join(self.dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Array file '{name}.npy' not found in {self.dir}")
        return np.load(path)

    def log_dict_of_arrays(self, name: str, data: dict[str, np.ndarray]):
        for key, arr in data.items():
            self.log_array(f"{name}_{key}", arr)

    def load_dict_of_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        files = [f for f in os.listdir(self.dir) if f.startswith(prefix) and f.endswith(".npy")]
        result = {}
        for fname in files:
            path = os.path.join(self.dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Array file '{fname}' not found in {self.dir}")
            key = fname.removeprefix(prefix).removesuffix(".npy").lstrip("_")
            result[key] = np.load(path)
        return result
    
    def log_scalar(self, name: str, value: float):
        path = os.path.join(self.dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump({"value": value}, f)

    def load_scalar(self, name: str) -> float:
        path = os.path.join(self.dir, f"{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scalar file '{name}.json' not found in {self.dir}")
        with open(path, "r") as f:
            return json.load(f)["value"]
        
    ### Checks ##
    
    def has_array(self, name: str) -> bool:
        path = os.path.join(self.dir, f"{name}.npy")
        return os.path.exists(path)
    
    def has_scalar(self, name: str) -> bool:
        path = os.path.join(self.dir, f"{name}.json")
        return os.path.exists(path)

    def has_metadata_entry(self, key: str) -> bool:
        meta = self.load_metadata()
        return key in meta

    ### Summarize ###

    def summarize(self):
        print(f"Experiment saved to: {self.dir}")

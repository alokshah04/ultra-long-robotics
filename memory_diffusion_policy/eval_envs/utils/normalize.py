"""
from OpenPi Codebase
"""

from copy import deepcopy
import logging
import json
import pathlib
from typing import Callable

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    min: numpydantic.NDArray
    max: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile
    median: numpydantic.NDArray | None = None  # median
         

class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        median = self._compute_quantiles([0.5])[0]
        return NormStats(mean=self._mean, std=stddev, min=self._min, max=self._max, q01=q01, q99=q99, median=median)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats], filename: str = "norm_stats.json") -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str, filename: str = "norm_stats.json") -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / filename
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())



def apply_tree(
    tree: dict[str, np.ndarray], selector: dict[str, NormStats], fn: Callable[[np.ndarray, NormStats], np.ndarray]
) -> dict[str, np.ndarray]:

    def transform(k: str, v: np.ndarray) -> np.ndarray:
        if k in selector:
            return fn(v, selector[k]).astype(np.float32)
        return v

    return {k: transform(k, v) for k, v in tree.items()}

class Normalizer:
    
    def __init__(self, norm_stats: dict[str, NormStats] | None, 
                 norm_type: str = "quantile", eps: float = 1e-6, 
                 chunk_norm_power: float | None = None, chunk_norm_zero_start: bool | None = None):
        self.norm_stats = norm_stats
        if norm_type in ["quantile", "q"]:
            self.norm_fn = self._normalize_quantile
        elif norm_type in ["minmax", "mm"]:
            self.norm_fn = self._normalize_minmax
        elif norm_type in ["normal", "standard", "std"]:
            self.norm_fn = self._normalize
        elif norm_type in ["root_power", "rp"]:
            assert chunk_norm_power is not None
            assert chunk_norm_zero_start is not None
            self.norm_fn = self._normalize_root_power
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.eps = eps
        self.chunk_norm_power = chunk_norm_power
        self.chunk_norm_zero_start = chunk_norm_zero_start

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.norm_stats is None:
            logging.warning("Norm stats are not provided... do nothing for normalization")
            return data
        return apply_tree(deepcopy(data), self.norm_stats, self.norm_fn)
    

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + self.eps)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + self.eps) * 2.0 - 1.0
    
    def _normalize_minmax(self, x, stats: NormStats):
        return (x - stats.min) / (stats.max - stats.min + self.eps) * 2.0 - 1.0
    
    def _normalize_root_power(self, x, stats: NormStats):
        powered_stats_max = np.power(stats.max, self.chunk_norm_power)
        if self.chunk_norm_zero_start:
            powered_stats_min = np.zeros_like(powered_stats_max)
        else:
            powered_stats_min = np.power(stats.min, self.chunk_norm_power)
        return (np.power(x, self.chunk_norm_power) - powered_stats_min) / (powered_stats_max - powered_stats_min + self.eps) * 2.0 - 1.0

class Unnormalizer:
    def __init__(self, norm_stats: dict[str, NormStats] | None, 
                 norm_type: str = "quantile", eps: float = 1e-6, 
                 chunk_norm_power: float | None = None, chunk_norm_zero_start: bool | None = None):
        self.norm_stats = norm_stats
        if norm_type in ["quantile", "q"]:
            self.norm_fn = self._unnormalize_quantile
        elif norm_type in ["minmax", "mm"]:
            self.norm_fn = self._unnormalize_minmax
        elif norm_type in ["normal", "standard", "std"]:
            self.norm_fn = self._unnormalize
        elif norm_type in ["root_power", "rp"]:
            assert chunk_norm_power is not None
            assert chunk_norm_zero_start is not None
            self.norm_fn = self._unnormalize_root_power
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.eps = eps
        self.chunk_norm_power = chunk_norm_power
        self.chunk_norm_zero_start = chunk_norm_zero_start

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.norm_stats is None:
            logging.warning("Norm stats are not provided... do nothing for unnormalization")
            return data
        return apply_tree(deepcopy(data), self.norm_stats, self.norm_fn)

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + self.eps) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + self.eps) + stats.q01
    
    def _unnormalize_minmax(self, x, stats: NormStats):
        return (x + 1.0) / 2.0 * (stats.max - stats.min + self.eps) + stats.min
    
    
    def _unnormalize_root_power(self, x, stats: NormStats):
        powered_stats_max = np.power(stats.max, self.chunk_norm_power)
        if self.chunk_norm_zero_start:
            powered_stats_min = np.zeros_like(powered_stats_max)
        else:
            powered_stats_min = np.power(stats.min, self.chunk_norm_power)
        return np.power((x + 1.0) / 2.0 * (powered_stats_max - powered_stats_min + self.eps) + powered_stats_min, 1.0 / self.chunk_norm_power)


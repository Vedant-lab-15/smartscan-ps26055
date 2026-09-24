"""Random baseline scheduler.

Selects K bands uniformly at random each step, without replacement.
Ignores all observations — pure random exploration.

Used as a lower-bound baseline. Any scheduler claiming to exploit
the channel structure should strictly dominate random selection.
"""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Uniform random band selector.

    Args:
        n_bands: Total number of frequency bands.
        k_scan:  Number of bands to scan simultaneously.
        seed:    Optional random seed for reproducibility.
    """

    def __init__(self, n_bands: int, k_scan: int, seed: int | None = None) -> None:
        if k_scan >= n_bands:
            raise ValueError(f"k_scan ({k_scan}) must be < n_bands ({n_bands})")
        self.n_bands = n_bands
        self.k_scan = k_scan
        self._rng = np.random.default_rng(seed)

    def select_arms(self, belief=None) -> set[int]:
        """Return K bands chosen uniformly at random without replacement.

        Args:
            belief: Ignored (random policy does not use belief state).

        Returns:
            Set of K distinct band indices.
        """
        return set(self._rng.choice(self.n_bands, size=self.k_scan, replace=False).tolist())

    def reset(self, seed: int | None = None) -> None:
        """Reset the RNG (optionally with a new seed)."""
        self._rng = np.random.default_rng(seed)

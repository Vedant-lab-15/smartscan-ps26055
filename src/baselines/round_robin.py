"""Round-Robin baseline scheduler.

Cycles through all N bands in fixed order, scanning K consecutive bands
each step. Ignores all observations — purely time-division multiplexing.

Clarkson (2003) showed that round-robin achieves the theoretical maximum
intercept probability for stationary i.i.d. emitters with unknown duty cycles.
Our scheduler must beat this baseline to claim improvement.

Reference:
    Clarkson, I. V. L. (2003). "The Arithmetic of Receiver Scheduling for
    Electronic Support." Proc. IEEE Aerospace Conference.
"""
from __future__ import annotations


class RoundRobinPolicy:
    """Stateless round-robin band selector.

    Args:
        n_bands: Total number of frequency bands.
        k_scan:  Number of bands to scan simultaneously.
    """

    def __init__(self, n_bands: int, k_scan: int) -> None:
        if k_scan >= n_bands:
            raise ValueError(f"k_scan ({k_scan}) must be < n_bands ({n_bands})")
        self.n_bands = n_bands
        self.k_scan = k_scan
        self._t: int = 0

    def select_arms(self, belief=None) -> set[int]:
        """Return the next K bands in round-robin order.

        Args:
            belief: Ignored (round-robin does not use belief state).

        Returns:
            Set of K band indices to scan this step.
        """
        start = (self._t * self.k_scan) % self.n_bands
        arms = {(start + j) % self.n_bands for j in range(self.k_scan)}
        self._t += 1
        return arms

    def reset(self) -> None:
        """Reset the round-robin cursor to step 0."""
        self._t = 0

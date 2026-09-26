"""Baseline schedulers for comparison against WIQL-UCB.

All baselines share the same interface:
    policy(t, n_bands, k, **kwargs) -> set[int]   (stateless)

or as a callable class with .select_arms(belief) -> set[int].

Available baselines:
    round_robin   — cycles through bands in fixed order, ignores observations
    random        — selects K bands uniformly at random each step
    clarkson      — analytical φ-incommensurable periodic scheduling
                    (Clarkson 2003/2005); requires known emitter parameters
"""
from src.baselines.round_robin import RoundRobinPolicy
from src.baselines.random_policy import RandomPolicy
from src.baselines.clarkson import ClarksonPolicy

__all__ = [
    "RoundRobinPolicy",
    "RandomPolicy",
    "ClarksonPolicy",
]

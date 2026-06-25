"""
round_robin.py — Baseline: assign jobs to GPUs in rotation, ignoring power.
"""
from policies.base import Policy
from typing import Optional


class RoundRobin(Policy):
    name = "round_robin"
    description = "Cycles through GPUs in order. Ignores power. Baseline only."

    def __init__(self):
        self._idx = 0

    def place(self, job, cluster) -> Optional[str]:
        eligible = [g for g in cluster.all_gpus() if g.can_fit(job)]
        if not eligible:
            return None
        gpu = eligible[self._idx % len(eligible)]
        self._idx += 1
        return gpu.id

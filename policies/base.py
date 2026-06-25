"""
policies/base.py — placement policy interface and reference implementations.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cluster.cluster import Cluster, Job, GPU


class Policy(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def place(self, job, cluster) -> Optional[str]:
        """Return GPU id or None."""
        ...

    def _eligible(self, job, cluster) -> list:
        return [g for g in cluster.all_gpus() if g.can_fit(job)]


class RoundRobin(Policy):
    name = "round_robin"
    description = "Cycles through GPUs ignoring power state. Triggers throttle under load."

    def __init__(self):
        self._i = 0

    def place(self, job, cluster) -> Optional[str]:
        el = self._eligible(job, cluster)
        if not el:
            return None
        g = el[self._i % len(el)]
        self._i += 1
        return g.id


class GreedyPowerFirst(Policy):
    name = "greedy_power"
    description = "Prefers rack with most power headroom. Avoids throttle cascades."

    def place(self, job, cluster) -> Optional[str]:
        el = self._eligible(job, cluster)
        if not el:
            return None
        def headroom(g):
            r = cluster.get_rack_for_gpu(g.id)
            return r.headroom_w() if r else 0.0
        return max(el, key=headroom).id


class GreedyLatencyFirst(Policy):
    name = "greedy_latency"
    description = "Prefers fastest GPU. Minimises latency but risks thermal cascade."

    def place(self, job, cluster) -> Optional[str]:
        el = self._eligible(job, cluster)
        if not el:
            return None
        return max(el, key=lambda g: g.tok_per_sec(job)).id


class PowerCapAware(Policy):
    name = "power_cap_aware"
    description = "Hard power cap enforcement. Best for sovereign/edge clusters."

    def place(self, job, cluster) -> Optional[str]:
        el = self._eligible(job, cluster)
        if not el:
            return None

        from cluster.cluster import _nearest_batch
        safe = []
        for g in el:
            r = cluster.get_rack_for_gpu(g.id)
            if not r:
                continue
            mp = g.profile["models"].get(job.model, {})
            bk = _nearest_batch(job.batch_size)
            job_pw = mp.get(bk, {}).get("power_w", g.profile["tdp_w"])
            projected = r.total_power_w() + job_pw - g.profile["idle_w"]
            if projected <= r.power_budget_w:
                safe.append((g, r.headroom_w()))

        if safe:
            return max(safe, key=lambda x: x[1])[0].id
        return min(el, key=lambda g: g.current_power_w()).id


POLICIES = {
    "round_robin":     RoundRobin,
    "greedy_power":    GreedyPowerFirst,
    "greedy_latency":  GreedyLatencyFirst,
    "power_cap_aware": PowerCapAware,
}

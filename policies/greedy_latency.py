"""
greedy_latency.py — Place jobs on the fastest available GPU, packing onto
the same rack first. Ignores power cost. This is the policy that causes
the thermal cascade demo under burst load.
"""
from policies.base import Policy
from typing import Optional


class GreedyLatency(Policy):
    name = "greedy_latency"
    description = "Always picks fastest GPU, packing same rack first. Triggers thermal cascade under burst load."

    def place(self, job, cluster) -> Optional[str]:
        eligible = [g for g in cluster.all_gpus() if g.can_fit(job)]
        if not eligible:
            return None

        def score(gpu):
            tok_s = gpu.tok_per_sec(job)
            rack = cluster.get_rack_for_gpu(gpu.id)
            # aggressively pack onto racks that already have active GPUs
            # this concentrates power and triggers thermal cascade
            rack_active = sum(1 for g in rack.gpus if g.current_jobs) if rack else 0
            packing_bonus = rack_active * 1000
            return tok_s + packing_bonus

        return max(eligible, key=score).id

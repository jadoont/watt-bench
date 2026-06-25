"""
greedy_power.py — Place jobs on the GPU that maximizes tokens/watt while
avoiding racks approaching their thermal cooling limit.
"""
from policies.base import Policy
from typing import Optional


class GreedyPower(Policy):
    name = "greedy_power"
    description = "Maximizes tokens/watt. Avoids racks near cooling limit. Sacrifices some latency."

    def place(self, job, cluster) -> Optional[str]:
        eligible = [g for g in cluster.all_gpus() if g.can_fit(job)]
        if not eligible:
            return None

        def score(gpu):
            rack = cluster.get_rack_for_gpu(gpu.id)
            if not rack:
                return 0.0
            tok_s = gpu.tok_per_sec(job)
            pw = max(gpu.current_power_w(), 1)
            tok_per_watt = tok_s / pw
            # cooling headroom ratio: 1.0 = empty rack, 0.0 = at thermal limit
            cooling_headroom = rack.cooling_limit_w - rack.total_power_w()
            cooling_ratio = max(0.0, cooling_headroom / max(rack.cooling_limit_w, 1))
            return tok_per_watt * (0.5 + cooling_ratio)

        return max(eligible, key=score).id

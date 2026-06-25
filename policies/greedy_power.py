"""
greedy_power.py — Place jobs on the GPU that maximizes tokens/watt while
hard-rejecting any rack where adding this job would exceed the cooling limit.

Two rules enforce thermal safety:
  1. No job stacking: only place on idle GPUs. Stacking causes hidden power
     jumps when the first job completes and a heavier job becomes current_jobs[0].
  2. Projection check: verify that swapping idle draw for active draw on a
     chosen GPU keeps the rack under cooling_limit_w.
"""
from policies.base import Policy
from cluster.cluster import _nearest_batch
from typing import Optional


class GreedyPower(Policy):
    name = "greedy_power"
    description = "Maximizes tokens/watt. Hard-rejects racks that would exceed cooling limit."

    def place(self, job, cluster) -> Optional[str]:
        # Idle-only: exclude GPUs already running a job to keep power model accurate
        eligible = [g for g in cluster.all_gpus() if not g.current_jobs and g.can_fit(job)]
        if not eligible:
            return None

        def score(gpu):
            rack = cluster.get_rack_for_gpu(gpu.id)
            if not rack:
                return -1.0
            mp = gpu.profile["models"].get(job.model, {})
            bk = _nearest_batch(job.batch_size)
            job_pw = mp.get(bk, {}).get("power_w", gpu.profile["tdp_w"])
            tok_s  = mp.get(bk, {}).get("tok_s", 1)
            # GPU is idle, so its contribution to rack.total_power_w() is idle_w.
            # Projected rack power after this placement:
            projected = rack.total_power_w() - gpu.profile["idle_w"] + job_pw
            if projected > rack.cooling_limit_w:
                return -1.0  # hard reject — would trigger thermal throttle
            tok_per_watt = tok_s / max(job_pw, 1)
            headroom_ratio = (rack.cooling_limit_w - projected) / rack.cooling_limit_w
            return tok_per_watt + headroom_ratio * 0.1

        scored = [(g, score(g)) for g in eligible]
        best_g, best_s = max(scored, key=lambda x: x[1])
        if best_s < 0:
            return None  # all racks at thermal limit — queue the job
        return best_g.id

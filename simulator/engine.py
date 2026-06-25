"""
simulator/engine.py — discrete event simulator for watt-bench.

Events: JOB_ARRIVE, JOB_FINISH, THERMAL (every 100ms simulated)
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Optional
from cluster.cluster import Cluster, Job
from policies.base import Policy
from simulator.metrics import MetricsCollector

THERMAL_INTERVAL = 0.1   # seconds


@dataclass(order=True)
class Event:
    time: float
    kind: str = field(compare=False)   # "arrive" | "finish" | "thermal"
    job: Optional[Job] = field(default=None, compare=False)
    gpu_id: Optional[str] = field(default=None, compare=False)


class Simulator:
    def __init__(self, cluster: Cluster, policy: Policy, verbose: bool = False):
        self.cluster = cluster
        self.policy = policy
        self.verbose = verbose

    def run(self, jobs: list[Job]) -> MetricsCollector:
        metrics = MetricsCollector()
        heap: list[Event] = []
        pending: list[Job] = []

        # Reset cluster state
        for g in self.cluster.all_gpus():
            g.current_jobs = []
            g.is_throttled = False
        for r in self.cluster.racks:
            r.throttle_events = 0

        def push(e): heapq.heappush(heap, e)
        def pop():   return heapq.heappop(heap)

        for j in jobs:
            push(Event(j.arrival_time, "arrive", job=j))

        start = jobs[0].arrival_time if jobs else 0.0
        push(Event(start, "thermal"))

        now = 0.0
        while heap:
            ev = pop()
            now = ev.time

            if ev.kind == "arrive":
                gpu_id = self.policy.place(ev.job, self.cluster)
                if gpu_id is None:
                    pending.append(ev.job)
                    metrics.queued += 1
                else:
                    _assign(ev.job, gpu_id, now, self.cluster, metrics, heap, push, self.verbose)

            elif ev.kind == "finish":
                g = self.cluster.get_gpu(ev.gpu_id)
                if g and ev.job in g.current_jobs:
                    g.current_jobs.remove(ev.job)
                metrics.finished.append(ev.job)
                if self.verbose:
                    sl = "MISS" if ev.job.sla_violated else "ok"
                    print(f"  [{now:.2f}s] FINISH {ev.job.id} {ev.job.latency_ms:.0f}ms SLA:{sl}")
                # drain queue
                still_pending = []
                for j in pending:
                    gid = self.policy.place(j, self.cluster)
                    if gid:
                        _assign(j, gid, now, self.cluster, metrics, heap, push, self.verbose)
                    else:
                        still_pending.append(j)
                pending[:] = still_pending

            elif ev.kind == "thermal":
                # Only continue thermal checks while jobs are still running
                any_active = any(g.current_jobs for g in self.cluster.all_gpus())
                any_pending = len(pending) > 0
                any_future_arrivals = any(e.kind == "arrive" for e in heap)
                if not (any_active or any_pending or any_future_arrivals):
                    continue  # no more work — stop spawning thermal events
                for rack in self.cluster.racks:
                    fired = rack.apply_thermal_check()
                    if fired:
                        metrics.throttle_events += 1
                        if self.verbose:
                            print(f"  [{now:.2f}s] ⚠ THROTTLE rack:{rack.id} {rack.total_power_w():.0f}W")
                        for g in rack.gpus:
                            for j in g.current_jobs:
                                if j.start_time is not None and j.end_time is not None and j.end_time > now:
                                    elapsed = now - j.start_time
                                    total_dur = j.end_time - j.start_time
                                    frac_done = min(1.0, elapsed / max(total_dur, 0.001))
                                    remaining_toks = j.output_tokens * (1 - frac_done)
                                    new_toks = g.tok_per_sec(j)
                                    if new_toks > 0:
                                        j.end_time = now + remaining_toks / new_toks
                                        j.throttled = True
                push(Event(now + THERMAL_INTERVAL, "thermal"))

        metrics.finalize()
        return metrics


def _assign(job, gpu_id, now, cluster, metrics, heap, push, verbose):
    g = cluster.get_gpu(gpu_id)
    if not g:
        return
    job.assigned_gpu = gpu_id
    job.start_time = now
    g.current_jobs.append(job)
    toks = g.tok_per_sec(job)
    if toks <= 0:
        toks = 1.0
    duration = job.output_tokens / toks
    job.end_time = now + duration
    push(Event(now + duration, "finish", job=job, gpu_id=gpu_id))
    rack = cluster.get_rack_for_gpu(gpu_id)
    if verbose:
        rpw = rack.total_power_w() if rack else 0
        print(f"  [{now:.2f}s] ASSIGN {job.id} → {gpu_id} | {toks:.0f}tok/s | rack:{rack.id if rack else '?'} {rpw:.0f}W")

"""
simulator/metrics.py — result collection and summary for watt-bench.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from cluster.cluster import Job

MODEL_POWER = {
    "llama3_8b_fp8":   200.0,
    "llama3_70b_fp8":  450.0,
    "llama3_405b_fp8": 600.0,
}


@dataclass
class MetricsCollector:
    finished: list = field(default_factory=list)
    queued: int = 0
    throttle_events: int = 0
    _total_tokens: int = 0
    _total_watt_s: float = 0.0

    def finalize(self):
        for j in self.finished:
            if j.start_time is not None and j.end_time is not None:
                dur = j.end_time - j.start_time
                pw = MODEL_POWER.get(j.model, 350.0)
                if j.throttled:
                    pw *= 0.65
                self._total_watt_s += pw * dur
                self._total_tokens += j.output_tokens

    @property
    def tokens_per_watt(self):
        return self._total_tokens / self._total_watt_s if self._total_watt_s > 0 else 0.0

    @property
    def latencies(self):
        return [j.latency_ms for j in self.finished if j.latency_ms is not None]

    @property
    def p99_ms(self):
        lats = sorted(self.latencies)
        if not lats: return 0.0
        return lats[min(int(len(lats) * 0.99), len(lats)-1)]

    @property
    def p95_ms(self):
        lats = sorted(self.latencies)
        if not lats: return 0.0
        return lats[min(int(len(lats) * 0.95), len(lats)-1)]

    @property
    def p50_ms(self):
        lats = self.latencies
        return statistics.median(lats) if lats else 0.0

    @property
    def sla_miss_pct(self):
        if not self.finished: return 0.0
        return sum(1 for j in self.finished if j.sla_violated) / len(self.finished) * 100

    def summary(self):
        return {
            "tokens_per_watt": round(self.tokens_per_watt, 4),
            "p50_ms":          round(self.p50_ms, 1),
            "p95_ms":          round(self.p95_ms, 1),
            "p99_ms":          round(self.p99_ms, 1),
            "throttle_events": self.throttle_events,
            "sla_miss_pct":    round(self.sla_miss_pct, 2),
            "completed":       len(self.finished),
            "queued":          self.queued,
            "throttled_jobs":  sum(1 for j in self.finished if j.throttled),
            "total_tokens":    self._total_tokens,
        }

    def print_summary(self, policy: str, cluster: str):
        s = self.summary()
        print(f"\n{'─'*52}")
        print(f"  policy:  {policy}  |  cluster: {cluster}")
        print(f"{'─'*52}")
        print(f"  tokens/watt      {s['tokens_per_watt']:.4f}")
        print(f"  P50 latency      {s['p50_ms']:.0f} ms")
        print(f"  P95 latency      {s['p95_ms']:.0f} ms")
        print(f"  P99 latency      {s['p99_ms']:.0f} ms")
        print(f"  throttle events  {s['throttle_events']}")
        print(f"  SLA miss         {s['sla_miss_pct']:.1f}%")
        print(f"  completed        {s['completed']}  |  queued: {s['queued']}")
        print(f"  throttled jobs   {s['throttled_jobs']}")
        print(f"{'─'*52}")

    def leaderboard_row(self, policy: str, cluster: str, trace: str) -> str:
        s = self.summary()
        return (f"| {policy} | {cluster} | {trace} | "
                f"{s['tokens_per_watt']:.4f} | {s['p99_ms']:.0f} | "
                f"{s['throttle_events']} | {s['sla_miss_pct']:.1f}% |")


@dataclass
class BenchResult:
    policy: str
    cluster: str
    trace: str
    metrics: MetricsCollector

    @property
    def tokens_per_watt_hour(self):
        return self.metrics.tokens_per_watt * 3600

    @property
    def p99_latency_ms(self):
        return self.metrics.p99_ms

    @property
    def throttle_events(self):
        return self.metrics.throttle_events

    @property
    def sla_miss_pct(self):
        return self.metrics.sla_miss_pct

    @property
    def jobs_completed(self):
        return len(self.metrics.finished)

    @property
    def jobs_dropped(self):
        return self.metrics.queued

    def summary(self):
        s = self.metrics.summary()
        s["tokens_per_watt_hour"] = round(self.tokens_per_watt_hour, 1)
        s["policy"] = self.policy
        s["cluster"] = self.cluster
        s["trace"] = self.trace
        return s

    def leaderboard_row(self):
        return (
            f"| {self.policy:<20} | {self.cluster:<12} | {self.trace:<12} | "
            f"{self.tokens_per_watt_hour:>14,.0f} | {self.p99_latency_ms:>16,.0f} | "
            f"{self.throttle_events:>15} | {self.sla_miss_pct:>9.1f}% |"
        )

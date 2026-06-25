"""
synthetic.py — Generate realistic inference job traces for watt-bench.

Modeled after BurstGPT and Azure LLM Inference Trace 2023 distributions.
References:
  - BurstGPT: https://github.com/HPMLL/BurstGPT
  - Azure LLM Inference Dataset 2023: https://github.com/Azure/AzurePublicDataset
"""
import random
import math
from typing import List
from cluster.cluster import Job


# Model mix roughly matching Azure 2023 trace distribution
MODEL_MIX = [
    ("llama3_70b_fp8",   0.35),  # flagship, high-value requests
    ("llama3_8b_fp8",    0.45),  # workhorse, most volume
    ("mistral_7b_fp16",  0.20),  # legacy/compatibility
]

# SLA tiers (ms) — loosely based on Azure/Cohere enterprise tiers
SLA_TIERS = [
    (500,   0.10),   # interactive: strict
    (2000,  0.50),   # standard: most requests
    (10000, 0.40),   # batch: relaxed
]


def _weighted_choice(choices):
    r = random.random()
    cumulative = 0
    for item, weight in choices:
        cumulative += weight
        if r < cumulative:
            return item
    return choices[-1][0]


def _prompt_length() -> int:
    """Log-normal distribution matching Azure trace prompt lengths."""
    # Azure 2023: median ~200 tokens, long tail to 4k
    return max(32, int(random.lognormvariate(math.log(200), 1.1)))


def _output_length() -> int:
    """Log-normal distribution matching Azure trace output lengths."""
    # Azure 2023: median ~100 tokens output
    return max(16, int(random.lognormvariate(math.log(100), 0.9)))


def _interarrival_s(rps: float) -> float:
    """Poisson arrivals — exponential interarrival times."""
    return random.expovariate(rps)


def generate(
    n_jobs: int = 500,
    rps: float = 2.0,           # requests per second (average)
    seed: int = 42,
    burst_factor: float = 3.0,  # peak/mean ratio for bursty load
    burst_duration_s: float = 30.0,
) -> List[Job]:
    """
    Generate a synthetic job trace.

    Args:
        n_jobs: total number of inference jobs
        rps: average requests per second
        seed: random seed for reproducibility
        burst_factor: how much RPS spikes during bursts (simulates traffic spikes)
        burst_duration_s: how long each burst lasts in seconds
    """
    random.seed(seed)
    jobs = []
    t = 0.0
    burst_start = random.uniform(30, 90)   # burst starts 30-90s in
    burst_end = burst_start + burst_duration_s

    for i in range(n_jobs):
        # bursty arrivals
        current_rps = rps * burst_factor if burst_start <= t <= burst_end else rps
        t += _interarrival_s(current_rps)

        model = _weighted_choice(MODEL_MIX)
        sla_ms = _weighted_choice(SLA_TIERS)

        jobs.append(Job(
            id=f"job-{i:04d}",
            model=model,
            prompt_tokens=_prompt_length(),
            output_tokens=_output_length(),
            sla_latency_ms=sla_ms,
            arrival_time=round(t, 4),
            batch_size=1,
        ))

    return jobs


def load_profile() -> List[Job]:
    """Standard 500-job load profile used for leaderboard comparisons."""
    return generate(n_jobs=500, rps=2.0, seed=42)


def stress_profile() -> List[Job]:
    """High-load 1000-job profile with aggressive bursts."""
    return generate(n_jobs=1000, rps=5.0, seed=99, burst_factor=5.0)


def sovereign_profile() -> List[Job]:
    """Low-volume, latency-sensitive profile for edge/air-gapped clusters."""
    return generate(n_jobs=200, rps=0.5, seed=7, burst_factor=2.0)

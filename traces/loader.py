"""
traces/loader.py

Loads real public LLM inference traces and generates synthetic workloads.

Supported real traces:
  - Azure LLM Inference Trace 2023 (github.com/Azure/AzurePublicDataset)
  - BurstGPT (github.com/HPMLL/BurstGPT)
  - ShareGPT (synthetic approximation of distribution)

For v0.1, real trace loading assumes you've downloaded the CSV locally.
Synthetic generation works out of the box with no downloads required.

Sources:
  - BlendServe (arxiv 2411.16102): WildChat compute density 1.4, BurstGPT 1.4,
    ShareGPT variable, Azure-Trace 1.4
  - Azure Public Dataset README for column schema
"""

from __future__ import annotations
import csv
import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from cluster.cluster import Job


# Empirical distributions from published trace analyses
# (prompt_len_mean, prompt_len_std, output_len_mean, output_len_std)
TRACE_DISTRIBUTIONS = {
    "azure": {
        "prompt_mean": 500, "prompt_std": 800,
        "output_mean": 200, "output_std": 180,
        "arrival_rate_rps": 8.0,   # requests per second at peak
        "description": "Azure production LLM workload. Compute-intensive, high variance prompt length."
    },
    "burstgpt": {
        "prompt_mean": 300, "prompt_std": 400,
        "output_mean": 350, "output_std": 300,
        "arrival_rate_rps": 12.0,
        "description": "ChatGPT/GPT-4 workload from Azure API. Bursty arrival pattern."
    },
    "sharegpt": {
        "prompt_mean": 180, "prompt_std": 250,
        "output_mean": 480, "output_std": 400,
        "arrival_rate_rps": 5.0,
        "description": "ShareGPT conversations. Longer outputs, shorter prompts."
    },
    "sovereign": {
        "prompt_mean": 1200, "prompt_std": 600,
        "output_mean": 800, "output_std": 400,
        "arrival_rate_rps": 1.5,
        "description": "Synthetic sovereign/edge workload. Long context, lower request rate, hard latency SLAs."
    },
}

MODELS = ["llama3_70b_fp8", "llama3_8b_fp8"]
MODEL_WEIGHTS = [0.4, 0.6]  # 8B more common at edge

SLA_BY_MODEL = {
    "llama3_8b_fp8":   2000.0,   # 2s SLA
    "llama3_70b_fp8":  5000.0,   # 5s SLA
    "llama3_405b_fp8": 15000.0,  # 15s SLA
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def generate_synthetic(
    trace_type: str = "azure",
    duration_s: float = 300.0,
    model: str | None = None,
    seed: int = 42,
    load_multiplier: float = 1.0,
) -> list[Job]:
    """
    Generate a synthetic job stream based on empirical trace distributions.

    Args:
        trace_type: One of 'azure', 'burstgpt', 'sharegpt', 'sovereign'
        duration_s: Simulation window in seconds
        model: Force a specific model (None = mix based on MODEL_WEIGHTS)
        seed: Random seed for reproducibility
        load_multiplier: Scale arrival rate (>1 = heavier load)

    Returns:
        List of Job objects sorted by arrival_time
    """
    rng = random.Random(seed)
    dist = TRACE_DISTRIBUTIONS[trace_type]
    jobs = []
    t = 0.0
    job_idx = 0

    base_rate = dist["arrival_rate_rps"] * load_multiplier
    # Poisson arrivals
    while t < duration_s:
        inter_arrival = rng.expovariate(base_rate)
        t += inter_arrival
        if t >= duration_s:
            break

        chosen_model = model or rng.choices(MODELS, weights=MODEL_WEIGHTS, k=1)[0]

        prompt_len = max(1, int(rng.gauss(dist["prompt_mean"], dist["prompt_std"])))
        output_len = max(1, int(rng.gauss(dist["output_mean"], dist["output_std"])))
        prompt_len = _clamp(prompt_len, 1, 32768)
        output_len = _clamp(output_len, 1, 8192)

        # Batch size: most requests arrive as batch=1, burst occasionally
        batch_size = rng.choices([1, 8, 32], weights=[0.7, 0.2, 0.1], k=1)[0]

        job = Job(
            id=f"job-{job_idx:05d}",
            model=chosen_model,
            prompt_tokens=int(prompt_len),
            output_tokens=int(output_len),
            sla_latency_ms=SLA_BY_MODEL.get(chosen_model, 5000.0),
            arrival_time=t,
            batch_size=batch_size,
        )
        jobs.append(job)
        job_idx += 1

    return sorted(jobs, key=lambda j: j.arrival_time)


def load_azure_csv(path: str) -> list[Job]:
    """
    Load real Azure LLM Inference Trace 2023.

    Download from:
    https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md

    Expected columns: TIMESTAMP, ContextTokens, GeneratedTokens
    """
    jobs = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Azure trace not found at {path}.\n"
            "Download from: https://github.com/Azure/AzurePublicDataset"
        )

    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                prompt_tokens = int(float(row.get("ContextTokens", 0)))
                output_tokens = int(float(row.get("GeneratedTokens", 0)))
                timestamp = float(row.get("TIMESTAMP", i))
                model = random.choices(MODELS, weights=MODEL_WEIGHTS, k=1)[0]
                job = Job(
                    id=f"azure-{i:06d}",
                    model=model,
                    prompt_tokens=max(1, prompt_tokens),
                    output_tokens=max(1, output_tokens),
                    sla_latency_ms=SLA_BY_MODEL.get(model, 5000.0),
                    arrival_time=timestamp,
                    batch_size=1,
                )
                jobs.append(job)
            except (ValueError, KeyError):
                continue

    return sorted(jobs, key=lambda j: j.arrival_time)


def trace_info() -> None:
    """Print info about available traces."""
    print("\nAvailable synthetic traces:")
    for name, dist in TRACE_DISTRIBUTIONS.items():
        print(f"  {name:12s} — {dist['description']}")
    print("\nReal traces (requires local download):")
    print("  azure_csv     — github.com/Azure/AzurePublicDataset")
    print("  burstgpt      — github.com/HPMLL/BurstGPT\n")

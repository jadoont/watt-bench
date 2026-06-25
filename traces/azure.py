"""
traces/azure.py — Real-trace loader for Azure LLM Inference Dataset 2023.

Downloads AzureLLMInferenceTrace_conv.csv on first use and caches it locally.
Falls back to synthetic load_profile if the download fails (offline mode).

Dataset:
  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md
Columns:
  TIMESTAMP       — request arrival datetime
  ContextTokens   — prompt length (tokens)
  GeneratedTokens — output length (tokens)

The dataset has no model field. We infer the model from token characteristics:
  short queries (prompt < 300 and output < 100) → llama3_8b_fp8
  everything else                                → llama3_70b_fp8
This approximates a real mixed-model serving fleet where simpler requests
are routed to a smaller, faster model.
"""
from __future__ import annotations

import csv
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List

from cluster.cluster import Job
from traces.synthetic import load_profile

_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset"
    "/master/data/AzureLLMInferenceTrace_conv.csv"
)
_CACHE = Path(__file__).parent / ".azure_cache.csv"

# Take first N rows. The full trace is ~19K requests over 58 minutes;
# 1000 rows ≈ 3 min of real data at the same arrival rate.
_N_JOBS = 1000


def _parse_ts(s: str) -> float:
    """'2023-11-16 18:15:46.6805900' → seconds as float."""
    base, *rest = s.split(".")
    dt = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
    frac = float("0." + rest[0]) if rest else 0.0
    return dt.timestamp() + frac


def _model_for(prompt_tokens: int, output_tokens: int) -> str:
    if prompt_tokens < 300 and output_tokens < 100:
        return "llama3_8b_fp8"
    return "llama3_70b_fp8"


def _sla_for(output_tokens: int) -> float:
    if output_tokens <= 100:
        return 2000.0    # interactive
    if output_tokens <= 500:
        return 5000.0    # standard
    return 10_000.0      # batch / long generation


def _download() -> bool:
    import ssl

    print("→ Downloading Azure LLM Inference Trace 2023 (~1 MB) ...", flush=True)

    def _fetch(ctx=None) -> None:
        handler = urllib.request.HTTPSHandler(context=ctx) if ctx else urllib.request.HTTPSHandler()
        opener  = urllib.request.build_opener(handler)
        with opener.open(_URL) as resp, open(_CACHE, "wb") as out:
            out.write(resp.read())

    try:
        _fetch()
    except Exception as exc:
        # urllib wraps ssl.SSLCertVerificationError in URLError — check the message.
        # This is common on macOS where Python ships without system certs.
        if "certificate" in str(exc).lower() or "ssl" in str(exc).lower():
            try:
                # Retry without verification for this public dataset.
                # Fix permanently: /Applications/Python 3.x/Install Certificates.command
                _fetch(ssl._create_unverified_context())
                print("  note: SSL cert verification skipped (macOS cert store missing)", flush=True)
            except Exception as exc2:
                print(f"  download failed: {exc2}", flush=True)
                return False
        else:
            print(f"  download failed: {exc}", flush=True)
            return False

    print(f"  cached → {_CACHE}", flush=True)
    return True


def _parse(path: Path) -> List[Job]:
    rows: list[tuple[float, int, int]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts   = _parse_ts(row["TIMESTAMP"])
                ctx  = max(1, int(float(row["ContextTokens"])))
                gen  = max(1, int(float(row["GeneratedTokens"])))
                rows.append((ts, ctx, gen))
            except (KeyError, ValueError):
                continue
            if len(rows) >= _N_JOBS:
                break

    if not rows:
        return []

    t0 = rows[0][0]
    jobs: List[Job] = []
    for i, (ts, ctx, gen) in enumerate(rows):
        model = _model_for(ctx, gen)
        jobs.append(Job(
            id=f"azure-{i:05d}",
            model=model,
            prompt_tokens=ctx,
            output_tokens=gen,
            sla_latency_ms=_sla_for(gen),
            arrival_time=round(ts - t0, 4),
            batch_size=1,
        ))
    return jobs


def azure_profile() -> List[Job]:
    """
    Load first 1000 requests from Azure LLM Inference Dataset 2023.

    Downloads and caches the CSV on first call. Falls back to synthetic
    load_profile if the network is unavailable.
    """
    if not _CACHE.exists():
        if not _download():
            print("  → falling back to synthetic load_profile (offline mode)", flush=True)
            return load_profile()

    try:
        jobs = _parse(_CACHE)
        if not jobs:
            raise ValueError("empty or unparseable trace file")
        model_counts = {}
        for j in jobs:
            model_counts[j.model] = model_counts.get(j.model, 0) + 1
        mix = ", ".join(f"{m}: {n}" for m, n in sorted(model_counts.items()))
        print(f"  loaded {len(jobs)} Azure trace jobs  [{mix}]", flush=True)
        return jobs
    except Exception as exc:
        print(f"  parse error ({exc}) — falling back to synthetic load_profile", flush=True)
        _CACHE.unlink(missing_ok=True)   # remove corrupt cache so next run re-downloads
        return load_profile()

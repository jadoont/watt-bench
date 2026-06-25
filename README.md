# watt-bench

**The first open benchmark for power-constrained LLM inference scheduling.**  
Tokens per watt-hour as a first-class metric.

---

## Why this exists

AI data centers are no longer GPU-constrained. As of mid-2026, [30–50% of planned US data center capacity is delayed or canceled](https://tech-insider.org/us-ai-data-center-delays-cancellations-7gw-capacity-crisis-2026/) — not because of chip shortages, but because of power infrastructure: transformers, switchgear, cooling. GPU utilization at enterprises sits around 5%.

The field has reframed the key metric: **tokens per watt**, not tokens per second. But no public benchmark optimizes for it.

vLLM, SGLang, and TensorRT-LLM are excellent inference engines. [TAPAS (Microsoft Azure Research)](https://arxiv.org/abs/2501.02600) and [ExeGPT (ASPLOS '24)](https://arxiv.org/abs/2404.07947) solve thermal and constraint-aware scheduling at hyperscaler scale. What doesn't exist:

- A **runnable, open-source benchmark** you can execute locally against public traces
- **Power as a hard constraint**, not a soft penalty
- A **policy-pluggable harness** where you can submit your own scheduler and see where it ranks
- A **sovereign/edge preset** modeling air-gapped or forward-deployed inference.


---

## Quickstart

```bash
git clone https://github.com/jadoont/watt-bench
cd watt-bench
python bench.py                              # greedy_power on medium cluster
python bench.py --policy greedy_latency --cluster medium --trace stress  # trigger the cascade
python bench.py --all                        # full comparison matrix
```

No GPU required. The simulator runs against hardware lookup tables derived from published benchmarks.

---

## The thermal cascade demo

This is the result that motivated the benchmark. Under burst load, a latency-greedy policy packs jobs onto the fastest rack until it exceeds the cooling limit. Every job on that rack throttles simultaneously — a cascade of SLA violations from a single placement decision.

```
Policy           Cluster  Trace   Tok/Wh    P99 (ms)  Throttle  SLA Miss
greedy_latency   medium   stress  17,686    1,145      6         0.7%
greedy_power     medium   stress  17,686    1,145      0         0.7%
round_robin      sovereign stress  14,548   2,730      0         2.5% (233 dropped)
```

Same throughput, zero throttle events. The power-aware policy gets there without touching the cooling limit.

---

## Architecture

```
watt-bench/
├── cluster/
│   ├── cluster.py              # GPU, Rack, Cluster classes
│   └── presets/
│       ├── small.json          # 4x H100, single rack
│       ├── medium.json         # 32x mixed H100/A100, tight power budgets
│       └── sovereign.json      # 8x A100, 2kW/rack hard cap (air-gapped/edge)
├── traces/
│   └── synthetic.py            # BurstGPT/Azure-distribution job generator
├── policies/
│   ├── base.py                 # Policy interface — implement place(job, cluster)
│   ├── round_robin.py          # baseline
│   ├── greedy_power.py         # maximize tokens/watt, avoid hot racks
│   └── greedy_latency.py       # fastest GPU first — causes cascade under load
├── simulator/
│   ├── engine.py               # discrete event simulation
│   └── metrics.py              # tokens/watt-hr, P99, throttle events, SLA miss
├── hardware_profiles.json      # GPU perf lookup tables (community-contributed)
├── bench.py                    # CLI entry point
└── results/leaderboard.md      # submit your policy via PR
```

### Key simplifications (documented)

| What's modeled accurately | What's simplified |
|--------------------------|-------------------|
| GPU power states (idle / active / throttled) | KV cache fragmentation (modeled as capacity) |
| Prefill vs decode power split (averaged) | Network topology (intra-rack = full BW) |
| Rack-level power budget as hard cap | Thermal dynamics (budget model, not CFD) |
| Thermal throttle cascade | Actual model execution (lookup tables) |

---

## Cluster presets

**`small`** — 4x H100, 10kW rack. Single-rack scenario, no cross-rack decisions.

**`medium`** — 32x mixed H100/A100 across 4 racks with tight cooling limits. The interesting heterogeneous case: policy must decide whether to pack onto fast H100 racks (throttle risk) or spread across slower A100 racks (power headroom).

**`sovereign`** — 8x A100, 2kW/rack hard cap. Models forward-deployed or air-gapped inference clusters. Every placement decision is a tradeoff. Inspired by sovereign AI infrastructure requirements.

---

## Hardware profiles

`hardware_profiles.json` contains tok/s and power figures per GPU × model × batch size, sourced from published benchmarks:

- H100 SXM5: [Spheron blog Apr 2026](https://www.spheron.network/blog/token-factory-gpu-cloud-tokens-per-watt-guide/), vLLM benchmarks
- A100 SXM4: MLPerf Inference v5.1 submissions
- A10G: Anyscale inference benchmarks 2025

**PRs adding measured hardware profiles are the highest-value contribution.**

---

## Real traces

**Azure LLM Inference Dataset 2023** (`--trace azure`)

watt-bench can run against Microsoft's published production trace. On first use, `bench.py` downloads and caches the CSV automatically:

```bash
python bench.py --trace azure --cluster medium
python bench.py --policy greedy_power --trace azure
```

The dataset has no model field; watt-bench maps requests to `llama3_8b_fp8` (short prompt + output) or `llama3_70b_fp8` (longer context) to approximate a real mixed-model fleet. Falls back to synthetic `load_profile` if the network is unavailable.

**Citation:**

> Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., & Bianchini, R. (2024).  
> *Characterizing Power Management Opportunities for LLMs in the Cloud.*  
> ACM ASPLOS 2024.  
> Dataset: [github.com/Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)

---

## Writing your own policy

```python
# policies/my_policy.py
from policies.base import Policy

class MyPolicy(Policy):
    name = "my_policy"
    description = "Your description here"

    def place(self, job, cluster):
        eligible = [g for g in cluster.all_gpus()
                    if g.can_fit(job)]
        if not eligible:
            return None
        # your placement logic here — return the GPU id string
        return eligible[0].id
```

```bash
# Register in bench.py POLICIES dict, then:
python bench.py --policy my_policy --cluster medium --trace stress
python bench.py --policy my_policy --submit   # append to leaderboard.md
```

---

## Leaderboard

See [`results/leaderboard.md`](results/leaderboard.md). Submit a PR adding your policy row.

---

## Known Limitations

**Sovereign cluster: policy-invariant results under stress.**
The `sovereign` cluster preset models an air-gapped/edge deployment with a hard 2kW/rack cap and only 8 A100s. Under the `stress` trace, the cluster is *capacity-constrained* (not enough GPUs to keep up with 25 RPS burst load), so all three policies produce identical throughput, latency, and throttle metrics — there are simply no placement decisions that change the outcome. This is realistic behavior for forward-deployed inference at the edge. The interesting policy differentiation happens on the `medium` cluster where power headroom, not GPU count, is the binding constraint.

**Power model uses per-model constants, not per-GPU actuals.**
`MetricsCollector` computes tokens/watt-hr using fixed per-model power constants (e.g., 450W for llama3_70b) rather than the per-GPU, per-batch values in `hardware_profiles.json`. This means tok/wh reflects model-level efficiency, not rack-level dispatch decisions. Future work: propagate actual GPU power from the simulation into the metrics layer.

**Single-job-per-GPU power accounting.**
`GPU.current_power_w()` reads only `current_jobs[0]`, so a GPU running multiple small models simultaneously (e.g., three llama3_8b jobs fitting in 80GB VRAM) counts power as if running one. This underestimates rack power in multi-tenant scenarios. The thermal cascade is still correctly demonstrated via rack-level power aggregation.

---

## Prior work

watt-bench is an open-source benchmarking harness, not a production scheduler. For production systems, see:

- **TAPAS** (Microsoft Azure, ASPLOS '25) — thermal/power-aware scheduling at hyperscaler scale
- **ExeGPT** (ASPLOS '24) — constraint-aware resource scheduling for LLM inference
- **vLLM** — continuous batching, PagedAttention, production inference engine
- **SGLang** — RadixAttention, high-throughput serving


---

## Citation

If you use watt-bench in research:

```bibtex
@software{wattbench2026,
  author = {Jadoon, Tayyaba},
  title  = {watt-bench: Power-Constrained LLM Inference Scheduling Benchmark},
  year   = {2026},
  url    = {https://github.com/jadoont/watt-bench}
}
```

---


# watt-bench Leaderboard

> Tokens per watt-hour as a first-class metric for LLM inference scheduling.  
> Submit your policy via PR — add a row to this table.

**Traces:** `load` = 500 jobs, 2 RPS avg | `stress` = 1000 jobs, 5 RPS avg with 5x burst | `sovereign` = 200 jobs, 0.5 RPS

| Policy               | Cluster      | Trace        | Tokens/Watt-hr | P99 Latency (ms) | Throttle Events | SLA Miss % |
|----------------------|--------------|--------------|----------------|------------------|-----------------|------------|
| greedy_power         | medium       | load         |         17,748 |            1,505 |               0 |       0.4% |
| greedy_latency       | medium       | load         |         17,748 |            1,505 |               0 |       0.4% |
| round_robin          | medium       | load         |         16,317 |            2,604 |               0 |       1.8% |
| greedy_power         | medium       | stress       |         17,686 |            1,145 |               0 |       0.7% |
| greedy_latency       | medium       | stress       |         17,686 |            1,145 |               6 |       0.7% |
| round_robin          | sovereign    | stress       |         14,548 |            2,730 |               0 |       2.5% |

**Key finding:** `greedy_latency` and `greedy_power` match on throughput under normal load — but under burst, `greedy_latency` triggers 6 thermal throttle events. Power-aware placement eliminates them at zero throughput cost.

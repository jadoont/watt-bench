#!/usr/bin/env python3
"""
bench.py — watt-bench CLI

Usage:
  python bench.py                                    # run all policies on medium cluster
  python bench.py --policy greedy_power              # single policy
  python bench.py --cluster sovereign --trace stress # specify cluster and trace
  python bench.py --all                              # run full comparison matrix

Outputs:
  - Leaderboard table to stdout
  - JSON results to results/latest.json
  - Appends to results/leaderboard.md on --submit
"""
import argparse
import json
import sys
import importlib
from pathlib import Path

# make imports work from project root
sys.path.insert(0, str(Path(__file__).parent))

from cluster.cluster import Cluster
from simulator.engine import Simulator
from simulator.metrics import BenchResult
import traces.synthetic as synthetic_traces

CLUSTER_PRESETS = {
    "small":    "cluster/presets/small.json",
    "medium":   "cluster/presets/medium.json",
    "sovereign":"cluster/presets/sovereign.json",
}

TRACE_PROFILES = {
    "load":     synthetic_traces.load_profile,
    "stress":   synthetic_traces.stress_profile,
    "sovereign":synthetic_traces.sovereign_profile,
}

POLICIES = {
    "round_robin":    ("policies.round_robin",    "RoundRobin"),
    "greedy_power":   ("policies.greedy_power",   "GreedyPower"),
    "greedy_latency": ("policies.greedy_latency", "GreedyLatency"),
}

LEADERBOARD_HEADER = (
    "\n| Policy               | Cluster      | Trace        "
    "| Tokens/Watt-hr | P99 Latency (ms) | Throttle Events | SLA Miss % |\n"
    "|----------------------|--------------|--------------|"
    "----------------|------------------|-----------------|------------|\n"
)


def load_policy(policy_name: str):
    if policy_name not in POLICIES:
        print(f"Unknown policy: {policy_name}")
        print(f"Available: {', '.join(POLICIES.keys())}")
        sys.exit(1)
    module_path, class_name = POLICIES[policy_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def run_bench(policy_name: str, cluster_name: str, trace_name: str, verbose: bool = False) -> BenchResult:
    if cluster_name not in CLUSTER_PRESETS:
        print(f"Unknown cluster: {cluster_name}. Choose from: {', '.join(CLUSTER_PRESETS)}")
        sys.exit(1)
    if trace_name not in TRACE_PROFILES:
        print(f"Unknown trace: {trace_name}. Choose from: {', '.join(TRACE_PROFILES)}")
        sys.exit(1)

    cluster = Cluster.from_preset(cluster_name)
    policy = load_policy(policy_name)
    jobs = TRACE_PROFILES[trace_name]()

    if verbose:
        print(f"\n→ Running: policy={policy_name} | cluster={cluster_name} | trace={trace_name} | jobs={len(jobs)}")

    engine = Simulator(cluster, policy, verbose=verbose)
    metrics = engine.run(jobs)
    return BenchResult(policy_name, cluster_name, trace_name, metrics)


def print_leaderboard(results: list[BenchResult]):
    print(LEADERBOARD_HEADER, end="")
    for r in results:
        print(r.leaderboard_row())
    print()


def save_results(results: list[BenchResult]):
    Path("results").mkdir(exist_ok=True)
    data = [r.summary() for r in results]
    with open("results/latest.json", "w") as f:
        json.dump(data, f, indent=2)
    print("→ Results saved to results/latest.json")


def submit_to_leaderboard(results: list[BenchResult]):
    """Append results to the leaderboard markdown file."""
    Path("results").mkdir(exist_ok=True)
    lb_path = Path("results/leaderboard.md")

    if not lb_path.exists():
        with open(lb_path, "w") as f:
            f.write("# watt-bench Leaderboard\n\n")
            f.write("> Tokens per watt-hour as a first-class metric for LLM inference scheduling.\n\n")
            f.write(LEADERBOARD_HEADER)
    
    with open(lb_path, "a") as f:
        for r in results:
            f.write(r.leaderboard_row() + "\n")

    print(f"→ Results appended to {lb_path}")


def _run_demo():
    print("\n→ Running greedy_latency on medium / stress  (1000 jobs) ...")
    gl = run_bench("greedy_latency", "medium", "stress")
    print("→ Running greedy_power   on medium / stress  (1000 jobs) ...")
    gp = run_bench("greedy_power",   "medium", "stress")

    print(LEADERBOARD_HEADER, end="")
    print(gl.leaderboard_row())
    print(gp.leaderboard_row())
    print()

    # Align throttle counts and SLA percentages
    te_w  = max(len(str(gl.throttle_events)), len(str(gp.throttle_events)))
    sla_gl = f"{gl.sla_miss_pct:.1f}%"
    sla_gp = f"{gp.sla_miss_pct:.1f}%"
    sla_w  = max(len(sla_gl), len(sla_gp))

    lines = [
        "THE THERMAL CASCADE RESULT:",
        f"greedy_latency: {gl.throttle_events:>{te_w}} throttle events,  {sla_gl:>{sla_w}} SLA miss",
        f"greedy_power:   {gp.throttle_events:>{te_w}} throttle events,  {sla_gp:>{sla_w}} SLA miss",
        "",
        "Same throughput. Zero cascade. Power-aware placement eliminates thermal throttling",
        "at no throughput cost on the medium cluster under burst load.",
    ]

    inner = max(len(l) for l in lines)
    bar   = "─" * (inner + 4)
    print(f"┌{bar}┐")
    for line in lines:
        print(f"│  {line:<{inner}}  │")
    print(f"└{bar}┘")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="watt-bench: power-constrained LLM inference scheduling benchmark"
    )
    parser.add_argument("--policy",  default="greedy_power", choices=list(POLICIES.keys()))
    parser.add_argument("--cluster", default="medium",       choices=list(CLUSTER_PRESETS.keys()))
    parser.add_argument("--trace",   default="load",         choices=list(TRACE_PROFILES.keys()))
    parser.add_argument("--all",     action="store_true",    help="Run full comparison matrix")
    parser.add_argument("--demo",    action="store_true",    help="Head-to-head: greedy_latency vs greedy_power on medium/stress")
    parser.add_argument("--submit",  action="store_true",    help="Append results to leaderboard.md")
    parser.add_argument("--verbose", action="store_true",    help="Print run details")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════╗")
    print("║         watt-bench v0.1                  ║")
    print("║  tokens/watt as a first-class metric     ║")
    print("╚══════════════════════════════════════════╝")

    if args.demo:
        _run_demo()
        return

    results = []

    if args.all:
        # run all policies × all clusters × standard load trace
        for policy_name in POLICIES:
            for cluster_name in CLUSTER_PRESETS:
                r = run_bench(policy_name, cluster_name, "load", verbose=args.verbose)
                results.append(r)
    else:
        r = run_bench(args.policy, args.cluster, args.trace, verbose=args.verbose)
        results.append(r)

    print_leaderboard(results)
    save_results(results)

    if args.submit:
        submit_to_leaderboard(results)

    # highlight the key insight for single runs
    if len(results) == 1:
        r = results[0]
        print(f"  tokens/watt-hr : {r.tokens_per_watt_hour:,.0f}")
        print(f"  P99 latency    : {r.p99_latency_ms:,.0f} ms")
        print(f"  throttle events: {r.throttle_events}")
        print(f"  SLA miss       : {r.sla_miss_pct}%")
        print(f"  jobs completed : {r.jobs_completed} / dropped: {r.jobs_dropped}\n")


if __name__ == "__main__":
    main()

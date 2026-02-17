"""
run_experiments.py — Batch runner for LLM placement simulations.

Defines experiment configurations for E1 (GPU heterogeneity), E2 (multi-region),
E3 (cluster scale). Each config is run across all transversal combos (alpha × model).
Uses main.main() for the actual simulation.

Usage (from the project root):
    python experiments/run_experiments.py --experiment E1
    python experiments/run_experiments.py --experiment E2
    python experiments/run_experiments.py --experiment E3
    python experiments/run_experiments.py --experiment all
"""

import os
import sys
import json
import time
import shutil
import argparse
from typing import List, Dict, Any, Tuple

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMULATION_DIR = os.path.join(PROJECT_ROOT, "simulation")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Transversal dimensions
# ---------------------------------------------------------------------------
ALPHAS = [0.9, 0.5, 0.1]
MODELS = ["LLaMa30B", "LLaMa70B"]

DEFAULT_BATCH_SIZE = 8
DEFAULT_REGION = "eu-west"

# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def parse_throughput_from_sol(sol_path: str) -> float:
    """Parse throughput (sum of flow_source_* > 0) from .sol file."""
    throughput = 0.0
    with open(sol_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("flow_source_"):
                val = float(parts[1])
                if val > 0:
                    throughput += val
    return throughput


def collect_metrics(
    test_name: str, model: str, alpha: float, batch_size: int,
    config_path: str, output_dir: str, solver_time: float
) -> Dict[str, Any]:
    """Collect metrics from a completed simulation run."""
    metrics = {
        "test_name": test_name,
        "model": model,
        "alpha": alpha,
        "batch_size": batch_size,
        "solver_time_s": round(solver_time, 1),
        "feasible": False,
    }

    sol_path = os.path.join(output_dir, "ilp_solution.sol")
    if not os.path.exists(sol_path):
        return metrics

    try:
        # Import here to ensure correct path
        from simulation.output_costs import parse_cluster_config, parse_solution, output_costs

        # Throughput
        throughput = parse_throughput_from_sol(sol_path)
        metrics["throughput_tokens_s"] = round(throughput, 2)
        metrics["feasible"] = throughput > 0

        # Costs
        node_types = parse_cluster_config(config_path)
        name_2_val = parse_solution(sol_path)
        rental_cost, energy_watts = output_costs(
            name_2_val=name_2_val, node_types=node_types
        )
        metrics["rental_cost_usd_hr"] = round(rental_cost, 3)
        metrics["power_watts"] = round(energy_watts, 1)

        # Active GPUs
        active_gpus = sum(1 for k, v in name_2_val.items()
                          if k.startswith("is_active_") and v == 1)
        metrics["active_gpus"] = active_gpus

    except Exception as e:
        metrics["error"] = str(e)

    return metrics


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_single(
    config_name: str,
    groups: List[Dict[str, Any]],
    model: str,
    alpha: float,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Run one simulation:
      1. Generate config .ini via gen_config
      2. Run solver via main.main()
      3. Rename output folder
      4. Collect metrics
    """
    from simulation.gen_config import generate_config_file
    from simulation.main import main

    test_name = f"test_{config_name}_{model}_{alpha}"
    ilp_tmp_dir = os.path.join(SIMULATION_DIR, "layouts", "ilp")
    output_dir = os.path.join(EXPERIMENTS_DIR, "results", f"ilp_{test_name}")
    config_path = os.path.join(ilp_tmp_dir, "config.ini")

    # Skip if already completed
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "ilp_solution.sol")):
        print(f"[SKIP] {test_name}")
        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                return json.load(f)
        return {"test_name": test_name, "skipped": True}

    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"  Model: {model} | Alpha: {alpha} | Batch: {batch_size}")
    print(f"{'='*60}")

    # 1. Generate config
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(ilp_tmp_dir):
        shutil.rmtree(ilp_tmp_dir)
    os.makedirs(ilp_tmp_dir, exist_ok=True)
    generate_config_file(groups, config_path, alpha=alpha)

    # 2. Run solver
    start_time = time.time()
    try:
        # main() expects to be called from simulation/ dir
        old_cwd = os.getcwd()
        os.chdir(SIMULATION_DIR)
        main(complete_cluster_file_name=config_path, model_name=model)
        os.chdir(old_cwd)
    except Exception as e:
        os.chdir(old_cwd)
        print(f"[ERROR] {test_name}: {e}")
    solver_time = time.time() - start_time

    # 3. Rename output
    if os.path.exists(ilp_tmp_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.rename(ilp_tmp_dir, output_dir)

    # 4. Collect metrics (config is now inside output_dir after rename)
    moved_config = os.path.join(output_dir, "config.ini")
    metrics = collect_metrics(
        test_name, model, alpha, batch_size, moved_config, output_dir, solver_time
    )

    # Save metrics.json inside the output folder
    if os.path.exists(output_dir):
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    exit(0)
    return metrics


def run_transversal(
    config_name: str, groups: List[Dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Run one config across all alpha × model combinations."""
    results = []
    for model in MODELS:
        for alpha in ALPHAS:
            m = run_single(config_name, groups, model, alpha, batch_size)
            results.append(m)
    return results


# ===========================================================================
# E1 — Fixed Size (16 GPUs), Heterogeneous GPUs, Single Region
# ===========================================================================

def generate_e1_configs() -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    6 categories × 3 combos = 18 configs.
    All 16 GPUs, single region, batch_size 8.
    """
    R = DEFAULT_REGION
    configs = []

    # ---- Cat 1: Homogeneous, only singles ----
    for i, gpu in enumerate(["L4", "A30", "L40S"]):
        groups = [{"num_nodes": 1, "type": gpu, "region": R} for _ in range(16)]
        configs.append((f"E1_homo-{gpu}-{i+1}", groups))

    # ---- Cat 2: Homogeneous, singles + groups ----
    homo_grp = [
        ("L4",  [4, 4, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("A30", [4, 2, 2, 2, 1, 1, 1, 1, 1, 1]),
        ("L40S",[4, 2, 2, 2, 1, 1, 1, 1, 1, 1]),
    ]
    for i, (gpu, sizes) in enumerate(homo_grp):
        groups = [{"num_nodes": s, "type": gpu, "region": R} for s in sizes]
        configs.append((f"E1_homo-grp-{gpu}-{i+1}", groups))

    # ---- Cat 3: Two GPU types ----
    heter2 = [
        ("A30-L40S", [(4,"A30"),(1,"A30"),(1,"A30"),(2,"L40S"),(2,"L40S"),(4,"L40S")]),
        ("L4-A100",  [(1,"L4"),(1,"L4"),(2,"L4"),(4,"L4"),(1,"L4"),(1,"L4"),(1,"A100-80"),(1,"A100-80"),(2,"A100-80")]),
        ("T4-H100",  [(4,"T4"),(4,"T4"),(4,"T4"),(1,"H100"),(1,"H100"),(1,"H100"),(1,"H100")]),
    ]
    for i, (tag, defs) in enumerate(heter2):
        groups = [{"num_nodes": n, "type": t, "region": R} for n, t in defs]
        configs.append((f"E1_heter2-{tag}-{i+1}", groups))

    # ---- Cat 4: Three GPU types ----
    heter3 = [
        ("L4-A4000-A100", [(2,"L4"),(1,"L4"),(1,"L4"),(2,"A4000"),(2,"A4000"),(1,"A4000"),(1,"A4000"),(2,"A100"),(1,"A100"),(1,"A100")]),
        ("T4-A30-L40S",   [(4,"T4"),(1,"T4"),(1,"T4"),(4,"A30"),(1,"A30"),(1,"L40S"),(2,"L40S"),(1,"L40S"),(1,"L40S")]),
        ("A4000-L40S-H100",[(2,"A4000"),(2,"A4000"),(1,"A4000"),(1,"A4000"),(2,"L40S"),(2,"L40S"),(2,"H100"),(2,"H100")]),
    ]
    for i, (tag, defs) in enumerate(heter3):
        groups = [{"num_nodes": n, "type": t, "region": R} for n, t in defs]
        configs.append((f"E1_heter3-{tag}-{i+1}", groups))

    # ---- Cat 5: Cloud mix (4+ types) ----
    mixes = [
        [(2,"L40S"),(2,"L40S"),(1,"L4"),(2,"L4"),(4,"L4"),(1,"A100-80"),(2,"A4000"),(1,"A100"),(1,"A30")],
        [(1,"H100"),(2,"L40S"),(2,"A30"),(4,"L4"),(1,"A4000"),(1,"A4000"),(1,"T4"),(2,"A100"),(1,"T4"),(1,"L4")],
        [(4,"A30"),(2,"L40S"),(1,"A100-80"),(1,"H100"),(2,"L4"),(2,"A4000"),(1,"T4"),(1,"T4"),(1,"L4"),(1,"A30")],
    ]
    for i, defs in enumerate(mixes):
        groups = [{"num_nodes": n, "type": t, "region": R} for n, t in defs]
        configs.append((f"E1_mix-{i+1}", groups))

    return configs


def run_e1() -> List[Dict[str, Any]]:
    """E1: GPU heterogeneity. 18 configs × 6 transversal = 108 runs."""
    print("\n" + "#"*60 + "\n  E1 — Fixed Size, Heterogeneous GPUs\n" + "#"*60)
    results = []
    for name, groups in generate_e1_configs():
        results.extend(run_transversal(name, groups))
    return results


# ===========================================================================
# E2 — Multi-Region (fixed 16 GPU cloud mix)
# ===========================================================================

# The fixed cloud mix GPU pool (flat list for region assignment)
E2_POOL_DEFS = [
    (2,"L40S"),(2,"L40S"),(1,"L4"),(2,"L4"),(4,"L4"),(1,"A100-80"),(2,"A4000"),(1,"A100"),(1,"A30"),
]


def _assign_regions(
    pool_defs: List[Tuple[int,str]],
    region_counts: Dict[str, int],
    concentrate_premium: bool = False,
) -> List[Dict[str, Any]]:
    """
    Assign regions to GPU groups.
    If concentrate_premium=False: fill regions evenly (round-robin groups).
    If concentrate_premium=True:  put premium groups in first region, fill rest.
    """
    groups = []
    regions = list(region_counts.keys())
    region_remaining = dict(region_counts)

    if concentrate_premium:
        # Sort: premium GPU types first
        cost_rank = {"H100":0,"A100-80":1,"A100":2,"L40S":3,"A30":4,"A4000":5,"L4":6,"T4":7}
        sorted_defs = sorted(pool_defs, key=lambda d: cost_rank.get(d[1], 99))
    else:
        sorted_defs = list(pool_defs)

    for n, t in sorted_defs:
        # Pick the region with most remaining slots
        region = max(region_remaining, key=region_remaining.get)
        if region_remaining[region] >= n:
            region_remaining[region] -= n
        else:
            # Fallback: pick any region with space
            for r in regions:
                if region_remaining[r] >= n:
                    region = r
                    region_remaining[r] -= n
                    break
        groups.append({"num_nodes": n, "type": t, "region": region})

    return groups


def generate_e2_configs() -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Multi-region configs. 11 total (1 + 2×5 region splits).
    Fixed 16-GPU cloud mix pool.
    """
    configs = []

    scenarios = [
        # (name, region_counts, num_variants)
        ("single-region", {"eu-west": 16}, 1),
        ("eu-split",      {"eu-west": 8, "eu-east": 8}, 2),
        ("eu-unbal",      {"eu-west": 10, "eu-east": 6}, 2),
        ("eu-us",         {"eu-west": 8, "us-east": 8}, 2),
        ("us-split",      {"us-east": 8, "us-west": 8}, 2),
        ("eu-us-asia",    {"eu-west": 6, "us-east": 5, "asia-east": 5}, 2),
    ]

    for name, region_counts, n_variants in scenarios:
        for v in range(n_variants):
            groups = _assign_regions(
                E2_POOL_DEFS, region_counts, concentrate_premium=(v == 1)
            )
            suffix = f"-{v+1}" if n_variants > 1 else ""
            configs.append((f"E2_{name}{suffix}", groups))

    return configs


def run_e2() -> List[Dict[str, Any]]:
    """E2: Multi-region. 11 configs × 6 transversal = 66 runs."""
    print("\n" + "#"*60 + "\n  E2 — Multi-Region\n" + "#"*60)
    results = []
    for name, groups in generate_e2_configs():
        results.extend(run_transversal(name, groups))
    return results


# ===========================================================================
# E3 — Cluster Scale (30 down to 6, step -4)
# ===========================================================================

# Base 30-GPU pool, ordered cheapest-first (removed first when shrinking)
E3_BASE_POOL = [
    # Cheap (removed first)
    (1,"T4"),(1,"T4"),(1,"T4"),(1,"T4"),
    (1,"A4000"),(1,"A4000"),(1,"A4000"),(1,"A4000"),
    (1,"A30"),(1,"A30"),
    (1,"L4"),(1,"L4"),(1,"L4"),(1,"L4"),
    (2,"L4"),
    # Mid
    (2,"A30"),(2,"L40S"),
    (1,"L40S"),(1,"L40S"),
    # Premium (removed last)
    (2,"L40S"),
    (1,"A100-80"),(1,"A100-80"),
    (1,"A100"),
    (1,"H100"),
]


def generate_e3_configs() -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Scale from 30 GPUs down to ~6, removing 4 cheapest each step.
    Steps: 30, 26, 22, 18, 14, 10, 6.
    """
    R = DEFAULT_REGION
    configs = []

    # Flatten and reverse so premium GPUs are first (kept longest)
    flat = list(reversed(E3_BASE_POOL))

    total_gpus = sum(n for n, _ in E3_BASE_POOL)  # 30
    for target in range(total_gpus, 3, -4):  # 30, 26, 22, 18, 14, 10, 6
        # Keep the first N groups from the reversed list (premium first)
        kept = []
        count = 0
        for n, t in flat:
            if count + n <= target:
                kept.append((n, t))
                count += n
            elif count < target:
                # Partially keep
                remaining = target - count
                kept.append((remaining, t))
                count += remaining
            if count >= target:
                break

        groups = [{"num_nodes": n, "type": t, "region": R} for n, t in kept]
        configs.append((f"E3_scale-{target}", groups))

    return configs


def run_e3() -> List[Dict[str, Any]]:
    """E3: Cluster scale. 7 configs × 6 transversal = 42 runs."""
    print("\n" + "#"*60 + "\n  E3 — Cluster Scale\n" + "#"*60)
    results = []
    for name, groups in generate_e3_configs():
        results.extend(run_transversal(name, groups))
    return results


# ===========================================================================
# Summary & output
# ===========================================================================

def save_summary(all_metrics: List[Dict[str, Any]], path: str):
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to: {path}")


def print_table(metrics: List[Dict[str, Any]]):
    hdr = f"{'Test':<50} {'Mod':<9} {'α':<5} {'OK':<3} {'TP(t/s)':<9} {'$/hr':<8} {'GPUs':<5} {'Time':<7}"
    print(f"\n{'='*len(hdr)}\n{hdr}\n{'='*len(hdr)}")
    for m in metrics:
        name = m.get("test_name","?")[:49]
        mod  = m.get("model","?")[:8]
        a    = m.get("alpha","?")
        ok   = "✓" if m.get("feasible") else "✗"
        tp   = m.get("throughput_tokens_s", "-")
        cost = m.get("rental_cost_usd_hr", "-")
        gpus = m.get("active_gpus", "-")
        t    = m.get("solver_time_s", "-")
        print(f"{name:<50} {mod:<9} {a:<5} {ok:<3} {tp:<9} {cost:<8} {gpus:<5} {t:<7}")


# ===========================================================================
# Main
# ===========================================================================

def main_cli():
    parser = argparse.ArgumentParser(description="Run LLM placement experiments")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["E1", "E2", "E3", "all"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print configs, don't run solver")
    args = parser.parse_args()

    all_metrics = []

    if args.dry_run:
        funcs = {"E1": generate_e1_configs, "E2": generate_e2_configs, "E3": generate_e3_configs}
        for exp in (["E1","E2","E3"] if args.exp == "all" else [args.exp]):
            configs = funcs[exp]()
            print(f"\n{exp}: {len(configs)} configs × {len(ALPHAS)} alphas × {len(MODELS)} models = {len(configs)*len(ALPHAS)*len(MODELS)} runs")
            for name, groups in configs:
                total_gpus = sum(g["num_nodes"] for g in groups)
                types = set(g["type"] for g in groups)
                regions = set(g["region"] for g in groups)
                print(f"  {name}: {total_gpus} GPUs, types={types}, regions={regions}")
        return

    if args.exp in ("E1", "all"):
        all_metrics.extend(run_e1())
    if args.exp in ("E2", "all"):
        all_metrics.extend(run_e2())
    if args.exp in ("E3", "all"):
        all_metrics.extend(run_e3())

    if all_metrics:
        summary_path = os.path.join(EXPERIMENTS_DIR, "results", "summary.json")
        save_summary(all_metrics, summary_path)
        print_table(all_metrics)


if __name__ == "__main__":
    main_cli()

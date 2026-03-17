# 2026.03.03: Baseline Layer Assignment Strategies

import os
import sys
import random
import json
from typing import List, Dict, Tuple, Any
from configparser import ConfigParser

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.formulas import calc_throughput, check_memory_constraint, calc_rental_cost
from src.specs import MODEL_SPECS, GPU_SPECS, LINK_SPECS, ModelSpec, GPUSpec
from src.utils import GB, MilliSec
from src.ilp_layout_aux import ILPLayout, NodeType, GRB
from simulation.visualize_solution import output_solution

class BaselineLayout:
    def __init__(self, cluster_file: str, model_name: str, batch_size: int, method: str, config_name: str, output_root: str):
        self.cluster_file = cluster_file
        self.model_name = model_name
        self.model_spec = MODEL_SPECS[model_name]
        self.batch_size = batch_size
        self.method = method
        self.config_name = config_name
        self.output_root = output_root
        self.offset = 0 # matching Helix offset
        
        # Load cluster info
        self.parser = ConfigParser()
        self.parser.read(cluster_file)
        
        self.compute_nodes = eval(self.parser["ComputeNodes"]["names"])
        
        # Aggregate nodes into groups
        self.groups = {}
        for node_id in self.compute_nodes:
            entity = f"ComputeNode-{node_id}"
            gpu_type = self.parser[entity]["type"]
            group_id = int(self.parser[entity]["group"])
            
            if group_id not in self.groups:
                self.groups[group_id] = {
                    "node_ids": [],
                    "type": gpu_type,
                    "spec": GPU_SPECS[gpu_type],
                    "count": 0
                }
            self.groups[group_id]["node_ids"].append(str(node_id))
            self.groups[group_id]["count"] += 1

        # Compute group-level metrics
        layer_size_bytes: int = self.model_spec.memory_bytes_per_layer
        for gid, info in self.groups.items():
            # max_num_layers considering group size and user's VRAM factor (0.5)
            one_gpu_cap = int((info["spec"].memory_gb * GB / 2) / layer_size_bytes)
            info["max_num_layers"] = one_gpu_cap * info["count"]
            
            # Group performance (correctly using group size for throughput)
            info["tp"] = calc_throughput(self.model_spec, info["spec"], info["count"], self.batch_size, link_type="PCIe", assigned_layers=info["max_num_layers"])

        # Layers per stage is the capacity of the weakest group
        self.layers_per_stage = min(info["max_num_layers"] for info in self.groups.values())
            
        # Create output directory: results_folder/[strategy]/[config_name]
        if self.method == "rr":
            strategy_folder = "roundrobin"
        elif self.method == "homo":
            strategy_folder = "binpacking"
        else:
            strategy_folder = self.method

        self.output_dir = os.path.join(self.output_root, strategy_folder, self.config_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_solution_ini(self, assignment: Dict[str, Tuple[int, int]], filename: str):
        """
        Save assignment to .ini file in the format:
        [Solution]
        compute_node_X=[layer_indices]
        [Parallelism]
        group_X_tp=1
        """
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w") as f:
            f.write("[Settings]\n")
            f.write(f"offset={self.offset}\n\n")
            
            f.write("[Solution]\n")
            for node_id in self.compute_nodes:
                start, end = assignment.get(str(node_id), (0, 0))
                layers = list(range(start, end))
                sim_node_name = f"compute_node_{int(node_id) + self.offset}"
                f.write(f"{sim_node_name}={layers}\n")
            
            f.write("\n[Parallelism]\n")
            # In baselines, all groups that have nodes assigned use TP
            active_groups = set()
            for gid, info in self.groups.items():
                # Correctly check if any node in the group has assigned layers
                if any(assignment.get(nid, (0, 0))[1] > assignment.get(nid, (0, 0))[0] for nid in info["node_ids"]):
                    active_groups.add(gid)
            
            for gid, info in self.groups.items():
                # TP is 1 only if group is active AND has multiple GPUs
                tp_val = 1 if (gid in active_groups and info["count"] > 1) else 0
                f.write(f"group_{gid}_tp={tp_val}\n")

        print(f"Solution saved to {out_path}")

    def evaluate_ilp(self, assignment: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """Compute resulting throughput and rental cost using ILP logic."""
        layout = ILPLayout()
        # Enable memory to match experimental setup
        layout.from_ini(self.cluster_file, self.model_name, enable_memory=True, batch_size=self.batch_size, tp_only=False)
        layout.build_model(seed=42, model_name=f"eval_{self.method}")
        
        model = layout.ilp_model
        
        # Pin assigned nodes
        for node_idx, (target_start, target_end) in assignment.items():
            target_num = target_end - target_start
            
            var_start = layout.var_node_start[f"start_{node_idx}"]
            model.addConstr(var_start == target_start, name=f"FIX_start_{node_idx}")
            
            if target_num in layout.var_node_hold_layer[node_idx]:
                var_hold = layout.var_node_hold_layer[node_idx][target_num]
                model.addConstr(var_hold == 1, name=f"FIX_hold_{node_idx}")
            else:
                var_is_active = layout.var_node_active[f"{node_idx}"]
                if target_num == 0:
                    model.addConstr(var_is_active == 0, name=f"FIX_inactive_{node_idx}")
                else:
                    print(f"ERROR: Node {node_idx} cannot host {target_num} layers (Max: {layout.ilp_nodes[node_idx].max_num_layers})")
                    return {"throughput": 0, "rental_cost": 0}

        # Explicitly inactivate compute nodes NOT in the assignment (e.g. excluded by swarm fallback).
        # Convention from ilp_layout_aux.py: inactive nodes get start = n_layers and is_active = 0.
        # Without this, the solver explores routing through excluded nodes, wasting time.
        n_layers = layout.model_spec.n_layers
        for node_idx, node in layout.ilp_nodes.items():
            if node.node_type != NodeType.COMPUTE:
                continue
            if node_idx not in assignment:
                var_start = layout.var_node_start[f"start_{node_idx}"]
                var_is_active = layout.var_node_active[node_idx]
                model.addConstr(var_start == n_layers, name=f"FIX_inactive_start_{node_idx}")
                model.addConstr(var_is_active == 0, name=f"FIX_inactive_{node_idx}")

        # Fix TP to ON for all groups in baselines (requested to consider groups)
        for group_tp_var_name, var_tp in layout.var_group_tp_active.items():
            # Extract group ID from 'group_tp_active_X'
            gid = int(group_tp_var_name.split("_")[-1])
            # Check if group is active AND has more than one GPU
            is_active_tp = any(assignment.get(nid, (0, 0))[1] > assignment.get(nid, (0, 0))[0] for nid in self.groups[gid]["node_ids"])
            is_multi_gpu = self.groups[gid]["count"] > 1
            model.addConstr(var_tp == (1 if (is_active_tp and is_multi_gpu) else 0), name=f"FIX_tp_state_{gid}")

        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            # Throughput is the total flow from source
            # ILP objective in ilp_layout_aux is norm_throughput * alpha - ...
            # We want raw throughput. 
            # In ilp_layout_aux.py: throughput = sum(flow_source_X)
            throughput = 0
            for other_idx in layout.ilp_source.connected_node_indices:
                throughput += layout.var_flow[f"flow_source_{other_idx}"].X
            
            # Rental cost
            rental_cost = sum(layout.var_node_active[node_idx].X * layout.ilp_nodes[node_idx].machine_type.rent_cost 
                              for node_idx in layout.ilp_nodes if layout.ilp_nodes[node_idx].node_type == NodeType.COMPUTE)
            
            # Save Gurobi solution file for visualization
            sol_name = os.path.join(self.output_dir, f"{self.method}_sol.sol")
            model.write(sol_name)
            
            return {
                "throughput": throughput,
                "rental_cost": rental_cost
            }
        else:
            print(f"FEASIBILITY ERROR: Baseline {self.method} is infeasible according to ILP constraints.")
            if model.Status == GRB.INFEASIBLE:
                 model.computeIIS()
                 for c in model.getConstrs():
                     if c.IISConstr: print(f"  - Violated: {c.ConstrName}")
            return {"throughput": 0, "rental_cost": 0}

    def roundrobin_layout(self) -> Dict[str, Tuple[int, int]]:
        """
        Round-Robin logic:
        1. Distributes model layers progressively across the available nodes.
        2. Prioritizes nodes with the most remaining layer capacity at each step.
        3. Scatters layers more evenly without the strict constraints of Swarm.
        """
        # Track remaining capacity for each group
        remaining_capacity = {}
        for gid, info in self.groups.items():
            # max_num_layers is the shared capacity for all GPUs in this TP group
            remaining_capacity[gid] = info["max_num_layers"]
                
        # Initially, all groups have 0 layers assigned
        group_assignments = {gid: 0 for gid in remaining_capacity}
        
        layers_to_assign = self.model_spec.n_layers
        
        # Check if total capacity is enough across all available groups
        total_capacity = sum(remaining_capacity.values())
        if layers_to_assign > total_capacity:
            print(f"  [Round-Robin] Infeasible: Model has {layers_to_assign} layers but cluster only holds {total_capacity}. Model too large.")
            return {}

        current_layer = 0
        while current_layer < layers_to_assign:
            # Find the group with the maximum remaining capacity
            best_group = max(
                remaining_capacity.keys(),
                key=lambda gid: (remaining_capacity[gid], int(gid))
            )
            
            if remaining_capacity[best_group] == 0:
                 break
                 
            remaining_capacity[best_group] -= 1
            group_assignments[best_group] += 1
            current_layer += 1
            
        if current_layer < layers_to_assign:
            print(f"  [Round-Robin] Infeasible: Could only assign {current_layer}/{layers_to_assign} layers.")
            return {}

        # Convert group assignments into contiguous chunk assignments
        assignment = {}
        current_start = 0
        
        # Sort groups by the size of their assigned chunk (descending) 
        sorted_groups = sorted(
            [g for g, count in group_assignments.items() if count > 0],
            key=lambda g: (group_assignments[g], int(g)), 
            reverse=True
        )
        
        # Assign layers to individual node_ids inside the chosen groups
        for gid in sorted_groups:
            count = group_assignments[gid]
            for node_id in self.groups[gid]["node_ids"]:
                assignment[node_id] = (current_start, current_start + count)
            current_start += count
            
        return assignment

    def binpacking_layout(self) -> Dict[str, Tuple[int, int]]:
        """
        Bin-Packing logic (ONE replica only):
        1. Rank groups by normalized_performance = tp * max_layers (best first).
        2. Greedily fill the best group with as many layers as its capacity allows.
        3. Move to next group when current is full.
        4. Stop as soon as all N layers are placed.
        """
        group_metrics = {}
        for gid, info in self.groups.items():
            metric = info["tp"] * info["max_num_layers"]
            group_metrics[gid] = metric if metric > 0 else -1
            
        sorted_groups = sorted([gid for gid in group_metrics if group_metrics[gid] > 0], 
                             key=lambda gid: group_metrics[gid], reverse=True)

        n_layers = self.model_spec.n_layers
        total_capacity = sum(self.groups[gid]["max_num_layers"] for gid in sorted_groups)
        if n_layers > total_capacity:
            print(f"  [Bin-Packing] Infeasible: Model has {n_layers} layers but cluster only holds {total_capacity}.")
            return {}

        assignment = {}
        current_progress = 0
        for gid in sorted_groups:
            if current_progress >= n_layers:
                break  # All layers placed — done
            max_cap = self.groups[gid]["max_num_layers"]
            start = current_progress
            end = min(current_progress + max_cap, n_layers)
            for node_id in self.groups[gid]["node_ids"]:
                assignment[node_id] = (start, end)
            current_progress = end
                
        if current_progress < n_layers:
            print(f"  [Bin-Packing] Infeasible: Could only assign {current_progress}/{n_layers} layers.")
            return {}

        return assignment

def baseline(config_root: str, output_root: str, config_name: str, model_name: str, strategy: str):
    
    batch_size = 8
    cluster_file = os.path.join(config_root, config_name, "config.ini")
    
    bl = BaselineLayout(cluster_file, model_name, batch_size, strategy, config_name, output_root)
    
    if strategy == "rr":
        print("\n--- Generating Round-Robin Layout ---")
        assign = bl.roundrobin_layout()
    elif strategy == "homo":
        print("\n--- Generating binpacking Layout ---")
        assign = bl.binpacking_layout()
    else:
        print(f"Unknown strategy: {strategy}")
        sys.exit(1)

    # If assignment is empty (e.g. Round-Robin infeasible), skip ILP evaluation
    if not assign:
        print(f"  [Baseline] Assignment is empty — skipping ILP evaluation (throughput = 0).")
        results = {"throughput": 0, "rental_cost": 0}
        res_path = os.path.join(bl.output_dir, "metrics_baseline.json")
        results_json = {
            "config": os.path.join(config_root, config_name, "config.ini"),
            "strategy": strategy,
            "model": model_name,
            "batch_size": 8,
            "throughput_tokens_s": 0,
            "rental_cost_usd_hr": 0,
            "infeasible": True
        }
        with open(res_path, "w") as f:
            json.dump(results_json, f, indent=4)
        print(f"Results saved to {res_path}")
        return

    # Save .ini solution
    bl.save_solution_ini(assign, f"{strategy}_sol.ini")
    
    # Evaluate using ILP logic
    print(f"Evaluating {strategy} layout using MILP constraints...")
    results = bl.evaluate_ilp(assign)
    
    print(f"\nResults for {strategy.upper()}:")
    print(f"  Throughput: {results['throughput']:.4f} token/s")
    print(f"  Rental Cost: ${results['rental_cost']:.2f}/h")
    
    # Save results to a file
    res_path = os.path.join(bl.output_dir, "metrics_baseline.json")
    results_json = {
        "config": cluster_file,
        "strategy": strategy,
        "model": model_name,
        "batch_size": batch_size,
        "throughput_tokens_s": results["throughput"],
        "rental_cost_usd_hr": results["rental_cost"]
    }
    with open(res_path, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"Results saved to {res_path}")

    # Generate solution graph
    sol_path = os.path.join(bl.output_dir, f"{strategy}_sol.sol")
    if os.path.exists(sol_path):
        print(f"Generating solution graph for {strategy}...")
        output_solution(cluster_file, sol_path, bl.output_dir)

if __name__ == "__main__":

    # Results folder for cluster configs
    cluster_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "results"))
    
    # Output folder for baselines
    baseline_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "baseline"))
    
    configs = [
        "ilp_test_E1_heter2-L4-A100-2_LLaMa30B_0.9",
        "ilp_test_E1_heter2-L4-A100-2_LLaMa70B_0.9",
        "ilp_test_E1_mix-1_LLaMa30B_0.9",
        "ilp_test_E1_mix-1_LLaMa70B_0.9",
        "ilp_test_E2_eu-us-asia-1_LLaMa30B_0.9",
        "ilp_test_E2_eu-us-asia-1_LLaMa70B_0.9",
        "ilp_test_E3_scale-30_LLaMa30B_0.9",
        "ilp_test_E3_scale-30_LLaMa70B_0.9"
    ]

    for config in configs:
        config_name = config
        model_name = config.split("_")[-2]
        
        # Run Round-Robin
        baseline(cluster_results_dir, baseline_results_dir, config_name, model_name, "rr")
        
        # Run binpacking
        baseline(cluster_results_dir, baseline_results_dir, config_name, model_name, "homo")


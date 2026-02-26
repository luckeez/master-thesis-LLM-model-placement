from typing import Dict, List
from enum import Enum
from configparser import ConfigParser

from src.specs import GPUSpec, GPU_SPECS

def parse_cluster_config(ini_file) -> Dict[str, str]:
    node_types = {}
    config = ConfigParser()
    config.read(ini_file)

    compute_nodes = eval(config["ComputeNodes"]["names"])
    for compute_node in compute_nodes:
        entity_name = f"ComputeNode-{compute_node}"
        node_type = config[entity_name]["type"]
        node_types[str(compute_node)] = node_type

    return node_types

def parse_solution(save_sol_path) -> Dict[str, int]:
    name_2_val: Dict[str, int] = {}
    with open(save_sol_path, "r") as file:
        for line in file:
            if line.startswith("is_active"):
                name, val = line.split(" ")
                name_2_val[name] = eval(val)
    
    return name_2_val

def output_costs(name_2_val: Dict[str, int], node_types: Dict[str, str], output_path = None) -> tuple[int, float, float]:
    rental_cost: int = 0
    energy_cost: float = 0.0
    for key, value in name_2_val.items():
        if key.startswith("is_active") and value:
            gpu_id: str = key.split("_")[2]
            gpu_type = node_types[gpu_id]
            gpu: GPUSpec = GPU_SPECS[gpu_type]
            rental_cost += gpu.rent_cost
            energy_cost += gpu.tdp_watts
    
    if output_path:
        print(rental_cost, energy_cost)
        
    return (rental_cost, energy_cost)
    


if __name__ == "__main__":
    name = "aux12_mem"
    node_types = parse_cluster_config(f"./config/{name}.ini")
    name_2_val = parse_solution(f"./layouts/ilp_{name}/ilp_solution.sol")
    output_costs(node_types=node_types, name_2_val=name_2_val, output_path = f"./layouts/ilp_{name}/memory_usage.png")
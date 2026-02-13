import matplotlib.pyplot as plt
from typing import Dict, List
from enum import Enum
from configparser import ConfigParser
import os

from src.utils import GB
from src.ilp_layout_aux import ILPNode
from src.specs import GPU_SPECS, GPUSpec, ModelSpec, MODEL_SPECS

from output_costs import output_costs, parse_cluster_config


class NodeType(Enum):
    COMPUTE = 1
    SOURCE = 2
    SINK = 3
    NIC_IN = 4
    NIC_OUT = 5


def visualize_memory_usage(save_sol_path: str, cluster_config_file_name: str, output_path: str = "memory_usage.png", include_cost = False, old_helix: bool = False) -> None:
    model_name = "LLaMa30B"
    model: ModelSpec = MODEL_SPECS[model_name]
    nodes: List[ILPNode] = []
    gpu_groups: Dict[int, List[str]] = {}


    config: ConfigParser = ConfigParser()
    config.read(cluster_config_file_name)

    if not old_helix:
        compute_nodes: List[str] = eval(config["ComputeNodes"]["names"])
        for compute_node in compute_nodes:
            entity_name: str = f"ComputeNode-{compute_node}"
            node_idx: str = str(compute_node)
            machine_type: GPUSpec = GPU_SPECS[config[entity_name]["type"]]
            group: int = int(config[entity_name]["group"])
            nodes.append(ILPNode(node_index=node_idx, node_type=NodeType.COMPUTE, machine_type=machine_type, group=group))
            gpu_groups.setdefault(group, []).append(node_idx)
    else:
        num_compute_nodes: int = int(config["NodeNames"]["total_compute_nodes"])
        for node_idx in range(num_compute_nodes):
            entity_name: str = f"ComputeNode-{node_idx}"
            machine_type: GPUSpec = GPU_SPECS[config[entity_name]["type"]]
            nodes.append(ILPNode(node_index=str(node_idx), node_type=NodeType.COMPUTE, machine_type=machine_type, group=-1))
    
    name_2_val: Dict[str, int | float] = {}
    with open(save_sol_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            name, val = line.split(" ")
            name_2_val[name] = eval(val)

    # Compute memory usage per layer
    memory_usage_per_layer_bytes: int = model.memory_bytes_per_layer
    
    node_labels: List[str] = []
    memory_usages: List[float] = []
    # Compute memory usage per node
    for node_idx, compute_node in enumerate(nodes):

        compute_node: ILPNode

        layer_count: int = 0
        for key in name_2_val.keys():
            if key.startswith(f"hold_{node_idx}_"):
                layer_count = int(key.split("_")[-1])
                if name_2_val[key] == 1:
                    compute_node.num_assigned_layers = layer_count
                    break
            compute_node.num_assigned_layers = 0
        # assert compute_node.num_assigned_layers > 0, "Bad layer count!"
    
        memory_bytes: int = compute_node.machine_type.memory_gb * GB
        shards: int = 1
        if not old_helix:
            if name_2_val[f"group_tp_active_{compute_node.group}"] == 1:
                shards = len(gpu_groups[compute_node.group])
        
        if compute_node.num_assigned_layers == 0:
            usage = 0.0
        else:
            used_memory_bytes: int = compute_node.num_assigned_layers * memory_usage_per_layer_bytes // shards
            usage: float = used_memory_bytes / memory_bytes * 100.0

        node_labels.append(f"{node_idx}-'{compute_node.machine_type.name}'\n{compute_node.num_assigned_layers}")
        memory_usages.append(usage)
    
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors 

    norm = mcolors.Normalize(vmin=20, vmax=50)
    colors = cm.RdYlGn(norm(memory_usages))

    plt.figure(figsize=(15, 6))
    plt.bar(node_labels, memory_usages, color=colors)
    plt.ylim(0, 100)
    plt.ylabel("Memory Usage (%)")
    title = cluster_config_file_name.split("/")[-1].split(".ini")[0]
    plt.title(f"GPU Memory Usage per Node ---- {title}")

    plt.text(0, -0.02, "Idx-Type\nLayers", ha='left', va='top', transform=plt.gca().transAxes)
    memory_usages_filtered = [m for m in memory_usages if m > 0.0]
    average_usage = sum(memory_usages_filtered) / len(memory_usages_filtered)
    plt.axhline(y=average_usage, color='r', linestyle='--', label='Average Usage')
    plt.text(-0.2, average_usage + 1, f"Avg: {average_usage:.2f}%", color='r', ha='right')

    for i, usage in enumerate(memory_usages):
        plt.text(i, usage + 1, f"{usage:.2f}%", ha='center')

    if include_cost:
        node_types = parse_cluster_config(cluster_config_file_name)
        purchase_cost, rental_cost, energy_cost = output_costs(name_2_val=name_2_val, node_types=node_types)
        cost_text = f"Total Purchase Cost (USD): {purchase_cost} USD\nTotal Rental Cost (USD/h): {round(rental_cost, 3)} USD/h\nEnergy Cost (kWh): {energy_cost} kWh" 
        plt.text(0, 90, cost_text, color="black", ha="left")
    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()
    plt.close()

def find_ini(folder_name: str) -> str:
    files = [f for f in os.listdir(folder_name) if f.endswith('new.ini')]
    if not files:
        return None
    
    ts_file = os.path.join(folder_name, sorted(files)[-1])
    try:
        with open(ts_file, 'r') as f:
            for line in f:
                if line.startswith("config_file"):
                    ini_file = line.split("=", 1)[1].strip()
                    ini_file = ini_file.split("/")[-1]
                    ini_file = ini_file.replace(".ini", "")
                    return ini_file
    except Exception as e:
        return None

if __name__ == "__main__":

    # visualize_memory_usage(
    #         save_sol_path=f"./layouts/ilp_backup/ilp_solution.sol",
    #         cluster_config_file_name=f"./config/single24.ini",
    #         output_path = f"./layouts/ilp_backup/memory_usage.png",
    #         old_helix=True
    #     )
    current_dir = os.getcwd()
    layouts_dir = os.path.join(current_dir, "layouts")
    aux_folders = [d for d in os.listdir(layouts_dir) if os.path.isdir(os.path.join(layouts_dir, d)) and (d.startswith("ilp_aux"))]

    for f in aux_folders:
        print(f"Visualizing memory usage for layout: {f}")
        old_helix = False
        if f.startswith("single"):
            old_helix = True
        config = find_ini(folder_name=f"./layouts/{f}")
        if config is None:
            print(f"Could not find config for layout: {f}, skipping...")
            continue
        if os.path.exists(f"./layouts/{f}/memory_usage.png"):
            print(f"Memory usage plot already exists for layout: {f}, skipping...")
            continue
        visualize_memory_usage(
            save_sol_path=f"./layouts/{f}/ilp_solution.sol",
            cluster_config_file_name=f"./config/{config}.ini",
            output_path = f"./layouts/{f}/memory_usage.png",
            include_cost=True,
            old_helix=old_helix
        )


    # name = "aux12-1group_mem"
    # visualize_memory_usage(
    #         save_sol_path=f"./layouts/ilp_{name}/ilp_solution.sol",
    #         cluster_config_file_name=f"./config/{name}.ini",
    #         output_path = f"./layouts/ilp_{name}/memory_usage.png",
    #         include_cost = True,
    #         old_helix=False
    #     )

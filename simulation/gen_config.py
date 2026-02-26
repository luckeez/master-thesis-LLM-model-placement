import sys
from typing import List, Dict, Union, Any
from itertools import permutations

# Define network parameters
# --------------------------------

# 1. External Network
# Used for links: Source -> NIC_IN, NIC_OUT -> Sink
EXT_BW = "1 * gbps"
EXT_LAT = "0.1 * MilliSec"

# 2. Internal Bus
# Used for links: NIC_IN -> GPU, GPU -> NIC_OUT
PCIE_BW = "64 * gbps" 
PCIE_LAT = "0.01 * MilliSec"

# 3. GPU Interconnect
# Used for links: GPU <-> GPU (only in multi-node groups)
NVLINK_BW = "600 * gbps"
NVLINK_LAT = "0.005 * MilliSec"

# 4. Memory Bandwidth given GPU type
MEMORY = {"T4": 320, "L4": 300, "A100": 1935}

REGION_LATENCY = {
    ("eu-west", "eu-west"): "0.5 * MilliSec",
    ("eu-west", "eu-east"): "5 * MilliSec",
    ("eu-west", "us-east"): "40 * MilliSec",
    ("eu-west", "us-west"): "70 * MilliSec",
    ("eu-west", "asia-east"): "100 * MilliSec",
    ("eu-west", "asia-south"): "85 * MilliSec",

    ("eu-east", "eu-west"): "5 * MilliSec",
    ("eu-east", "eu-east"): "0.5 * MilliSec",
    ("eu-east", "us-east"): "45 * MilliSec",
    ("eu-east", "us-west"): "75 * MilliSec",
    ("eu-east", "asia-east"): "90 * MilliSec",
    ("eu-east", "asia-south"): "75 * MilliSec",

    ("us-east", "eu-west"): "40 * MilliSec",
    ("us-east", "eu-east"): "45 * MilliSec",
    ("us-east", "us-east"): "0.5 * MilliSec",
    ("us-east", "us-west"): "30 * MilliSec",
    ("us-east", "asia-east"): "90 * MilliSec",
    ("us-east", "asia-south"): "100 * MilliSec",

    ("us-west", "eu-west"): "70 * MilliSec",
    ("us-west", "eu-east"): "75 * MilliSec",
    ("us-west", "us-east"): "30 * MilliSec",
    ("us-west", "us-west"): "0.5 * MilliSec",
    ("us-west", "asia-east"): "60 * MilliSec",
    ("us-west", "asia-south"): "80 * MilliSec",

    ("asia-east", "eu-west"): "100 * MilliSec",
    ("asia-east", "eu-east"): "90 * MilliSec",
    ("asia-east", "us-east"): "90 * MilliSec",
    ("asia-east", "us-west"): "60 * MilliSec",
    ("asia-east", "asia-east"): "0.5 * MilliSec",
    ("asia-east", "asia-south"): "25 * MilliSec",

    ("asia-south", "eu-west"): "85 * MilliSec",
    ("asia-south", "eu-east"): "75 * MilliSec",
    ("asia-south", "us-east"): "100 * MilliSec",
    ("asia-south", "us-west"): "80 * MilliSec",
    ("asia-south", "asia-east"): "25 * MilliSec",
    ("asia-south", "asia-south"): "0.5 * MilliSec",
}
# --------------------------------


node_definitions: Dict[str, Dict[str, Any]] = {}
link_definitions: Dict[str, Dict[str, str]] = {}

def get_section_name(node_id: int | str) -> str | None:
    """ Converts an ID (e.g., 0 or "nic_in_0") into its .ini section name. """
    if isinstance(node_id, int):
        return f"ComputeNode-{node_id}"
    if isinstance(node_id, str) and node_id.startswith("nic_"):
        return f"AuxiliaryNode-{node_id}"
    return None

def add_link(a: int | str, b: int | str, bw: str, lat: str):
    """
    Add a link to config.
    Also update the connected_nodes lists for both nodes.
    """
    # 1. Add the link to the dictionary
    link_name = f"Link-{a}-{b}"
    if link_name not in link_definitions:
        link_definitions[link_name] = {"bandwidth": bw, "latency": lat}

    # 2. Update connected_nodes for 'a' (if not source)
    a_section = get_section_name(a)
    if a_section:
        node_definitions[a_section]["connected_nodes"].append(b)

    # 3. Update connected_nodes for 'b' (if not sink)
    b_section = get_section_name(b)
    if b_section:
        node_definitions[b_section]["connected_nodes"].append(a)

def generate_config_file(groups: List[Dict[str, Any]], output_filepath: str, alpha: float = 0.5):
    """
    Generate a .ini configuration file for the cluster topology.

    :param groups: List of dictionaries, e.g.:
                   [{"num_nodes": 1, "type": "A100"},
                    {"num_nodes": 4, "type": "T4"}]
    :param output_filepath: Path to save the file (e.g. "cluster_config.ini")
    """
    # Clear global data structures
    node_definitions.clear()
    link_definitions.clear()
    
    compute_node_ids: List[int] = []
    aux_node_ids: List[str] = []
    all_groups_info = []
    current_node_index = 0

    # --- 1. Define nodes
    group_id = 0
    for group_config in groups:
        num_nodes = group_config["num_nodes"]
        gpu_type = group_config["type"]
        region = group_config["region"]
        
        nic_in_id = f"nic_in_{group_id}"
        nic_out_id = f"nic_out_{group_id}"
        aux_node_ids.extend([nic_in_id, nic_out_id])
        
        gpu_ids_in_group = []
        for _ in range(num_nodes):
            node_id = current_node_index
            compute_node_ids.append(node_id)
            gpu_ids_in_group.append(node_id)
            current_node_index += 1

        # Save group info
        all_groups_info.append({
            "id": group_id,
            "gpus": gpu_ids_in_group,
            "nic_in": nic_in_id,
            "nic_out": nic_out_id,
            "type": gpu_type,
            "region": region
        })

        # Initialize node sections
        node_definitions[f"AuxiliaryNode-{nic_in_id}"] = {
            "type": gpu_type, "group": group_id, "node_type": "NIC_IN", "connected_nodes": []
        }
        node_definitions[f"AuxiliaryNode-{nic_out_id}"] = {
            "type": gpu_type, "group": group_id, "node_type": "NIC_OUT", "connected_nodes": []
        }
        for node_id in gpu_ids_in_group:
            node_definitions[f"ComputeNode-{node_id}"] = {
                "type": gpu_type, "group": group_id, "node_type": "COMPUTE", "connected_nodes": []
            }
        group_id += num_nodes

    # --- 2. Define links ---
    
    for group_info in all_groups_info:
        nic_in = group_info["nic_in"]
        nic_out = group_info["nic_out"]
        gpus = group_info["gpus"]

        # A. Link Source/Sink
        add_link("source", nic_in, EXT_BW, EXT_LAT)
        add_link(nic_out, "sink", EXT_BW, EXT_LAT)

        # B. External link (Inter-Group): NIC_OUT -> all other NIC_IN
        for other_group in all_groups_info:
            if other_group["nic_in"] != nic_in:
                src_region = group_info["region"]
                dst_region = other_group["region"]
                lat = REGION_LATENCY[(src_region, dst_region)]
                add_link(nic_out, other_group["nic_in"], EXT_BW, lat)

        # C. Internal link (NIC-GPU)
        for gpu_id in gpus:
            add_link(nic_in, gpu_id, PCIE_BW, PCIE_LAT)
            add_link(gpu_id, nic_out, PCIE_BW, PCIE_LAT)

        # D. Internal link (GPU-GPU - NVLink)
        if len(gpus) > 1:
            for gpu_a, gpu_b in permutations(gpus, 2):
                add_link(gpu_a, gpu_b, PCIE_BW, PCIE_LAT)

    # --- 3. Write the .ini File ---
    try:
        with open(output_filepath, "w") as f:
            f.write(f"# Topology: {len(groups)} groups.\n\n")

            f.write(f"[Settings]\n")
            f.write(f"optimization_alpha = {alpha}\n")

            # Write node names
            f.write("[ComputeNodes]\n")
            f.write(f"names = {compute_node_ids}\n\n")
            
            f.write("[AuxiliaryNodes]\n")
            f.write(f"names = {aux_node_ids}\n\n")
            
            f.write("[Groups]\n")
            f.write(f"num_groups = {len(groups)}\n\n")

            f.write("[SourceNode]\n")
            f.write(f"connected_nodes = {[g['nic_in'] for g in all_groups_info]}\n\n")

            f.write("[SinkNode]\n")
            f.write(f"connected_nodes = {[g['nic_out'] for g in all_groups_info]}\n\n")

            # Write node sections
            for section_name, values in node_definitions.items():
                f.write(f"[{section_name}]\n")

                values["connected_nodes"] = sorted(list(set(values["connected_nodes"])), key=str)
                for key, val in values.items():
                    f.write(f"{key}={val}\n")
                f.write("\n")

            # Write link sections
            for section_name, values in link_definitions.items():
                f.write(f"[{section_name}]\n")
                for key, val in values.items():
                    f.write(f"{key}={val}\n")
                f.write("\n")

        print(f"Success! Configuration file saved to: {output_filepath}")

    except IOError as e:
        print(f"Error: Unable to write file. {e}", file=sys.stderr)

if __name__ == "__main__":

    name = "test15-1dc-intra-inter.ini"
    cluster_configuration = [
    	{"num_nodes": 4, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "A100", "region": "rack1"},
    	{"num_nodes": 1, "type": "L4", "region": "rack1"}, 
    	{"num_nodes": 1, "type": "L4", "region": "rack1"},

    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "T4", "region": "rack2"},
    	{"num_nodes": 1, "type": "T4", "region": "rack2"},

    	{"num_nodes": 1, "type": "T4", "region": "rack3"},
    	{"num_nodes": 1, "type": "A100", "region": "rack3"},
    	{"num_nodes": 1, "type": "T4", "region": "rack3"},
    	{"num_nodes": 1, "type": "L4", "region": "rack3"}
    ]

    name = "test18-a30.ini"
    cluster_configuration = [
    	{"num_nodes": 4, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "A100", "region": "rack1"},
    	{"num_nodes": 1, "type": "L4", "region": "rack1"}, 
    	{"num_nodes": 1, "type": "L4", "region": "rack1"},

    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "T4", "region": "rack2"},
    	{"num_nodes": 1, "type": "T4", "region": "rack2"},

    	{"num_nodes": 1, "type": "T4", "region": "rack3"},
    	{"num_nodes": 4, "type": "A30", "region": "rack3"},
    	{"num_nodes": 1, "type": "T4", "region": "rack3"},
    	{"num_nodes": 1, "type": "L4", "region": "rack3"}
    ]

    name = "test15-unique.ini"
    cluster_configuration = [
    	{"num_nodes": 4, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "T4", "region": "rack1"}, 
    	{"num_nodes": 1, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "T4", "region": "rack1"},

    	{"num_nodes": 1, "type": "L4", "region": "rack1"},
    	{"num_nodes": 1, "type": "L4", "region": "rack1"},
    	{"num_nodes": 1, "type": "L4", "region": "rack1"},
    	{"num_nodes": 1, "type": "L4", "region": "rack1"},
    	{"num_nodes": 1, "type": "L4", "region": "rack1"},

    	{"num_nodes": 1, "type": "A100", "region": "rack1"},
    	{"num_nodes": 1, "type": "A100", "region": "rack1"},

    ]

    name = "test15-split.ini"
    cluster_configuration = [
    	{"num_nodes": 4, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "T4", "region": "rack1"}, 
    	{"num_nodes": 1, "type": "T4", "region": "rack1"},
    	{"num_nodes": 1, "type": "T4", "region": "rack1"},

    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    	{"num_nodes": 1, "type": "L4", "region": "rack2"},
        
    	{"num_nodes": 1, "type": "A100", "region": "rack2"},
    	{"num_nodes": 1, "type": "A100", "region": "rack2"},

    ]

    name = "test-cloud.ini"
    cluster_configuration = [
        {"num_nodes": 2, "type": "L40S", "region": "eu-west"},
        {"num_nodes": 2, "type": "L40S", "region": "eu-west"}, # 1.9
        {"num_nodes": 1, "type": "L4", "region": "eu-west"}, # 0.9
        {"num_nodes": 2, "type": "L4", "region": "eu-west"}, # 1.72
        {"num_nodes": 4, "type": "L4", "region": "eu-west"}, # 3.37
        {"num_nodes": 1, "type": "A100-80", "region": "eu-west"}, # 1.82
        {"num_nodes": 2, "type": "A4000", "region": "eu-west"}, # 0.30
    ]

    name = "test-cloud-short-a30.ini"
    cluster_configuration = [
        {"num_nodes": 2, "type": "L40S", "region": "eu-west"},
        {"num_nodes": 2, "type": "L40S", "region": "eu-west"}, # 1.72
        {"num_nodes": 4, "type": "A30", "region": "eu-west"}, # 0.9
        # {"num_nodes": 2, "type": "L4", "region": "eu-west"}, # 1.72
        # {"num_nodes": 4, "type": "L4", "region": "eu-west"}, # 3.37
        # {"num_nodes": 1, "type": "A100-80", "region": "eu-west"}, # 1.82
        {"num_nodes": 2, "type": "A4000", "region": "eu-west"}, # 0.30
    ]

    name = "test-cloud-short-a100.ini"
    cluster_configuration = [
        {"num_nodes": 2, "type": "L40S", "region": "eu-west"},
        {"num_nodes": 2, "type": "L40S", "region": "eu-west"}, # 1.9
        {"num_nodes": 1, "type": "A100", "region": "eu-west"}, # 0.9
        # {"num_nodes": 2, "type": "L4", "region": "eu-west"}, # 1.72
        # {"num_nodes": 4, "type": "L4", "region": "eu-west"}, # 3.37
        # {"num_nodes": 1, "type": "A100-80", "region": "eu-west"}, # 1.82
        {"num_nodes": 2, "type": "A4000", "region": "eu-west"}, # 0.30
    ]


    name = "test-config.ini"
    cluster_configuration = [
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
        {"num_nodes": 1, "type": "L4", "region": "eu-west"},
    ]

    # name = "test12-1dc-intra-inter-l4.ini"
    # cluster_configuration = [
    # 	{"num_nodes": 1, "type": "L4", "region": "rack1"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack1"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack1"}, 
    # 	{"num_nodes": 1, "type": "L4", "region": "rack1"},

    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},

    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"}
    # ]

    # name = "test12-1dc-intra-inter-l4-1g.ini"
    # cluster_configuration = [
    # 	{"num_nodes": 4, "type": "L4", "region": "rack1"},

    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack2"},

    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"},
    # 	{"num_nodes": 1, "type": "L4", "region": "rack3"}
    # ]

    

    # Scenario: 4 single nodes + 1 group of 4 T4
    # name = "aux8-1group.ini"
    # cluster_configuration = [
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 4, "type": "T4"},
    # ]

    # name = "aux8.ini"
    # cluster_configuration = [
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    # ]
    
    # Scenario: 9 groups, one with 4 T4
    # name = "aux12-1group.ini"
    # cluster_configuration = [
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 1, "type": "A100"},
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 4, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "A100"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "L4"}
    # ]

    # Scenario: same as aux9 but all single nodes
    # name = "aux12.ini"
    # cluster_configuration = [
    # 	{"num_nodes": 1, "type": "L4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "A100", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "L4", "region": "eu-west"}, 
    # 	{"num_nodes": 1, "type": "L4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "T4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "T4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "T4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "T4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "T4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "A100", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "T4", "region": "eu-west"},
    # 	{"num_nodes": 1, "type": "L4", "region": "eu-west"}
    # ]

    # name = "aux12-2groups.ini"
    # cluster_configuration = [
    #     {"num_nodes": 4, "type": "T4"},
    #     {"num_nodes": 4, "type": "L4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "T4"},
    # ]


    # Scenario: 16 single nodes
    # name = "aux16.ini"
    # cluster_configuration = [
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"}
    # ]

    # name = "aux16-1group.ini"
    # cluster_configuration = [
    #     {"num_nodes": 4, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"}
    # ]

    # name = "aux16-2groups.ini"
    # cluster_configuration = [
        # {"num_nodes": 4, "type": "T4"},
        # {"num_nodes": 4, "type": "L4"},
        # {"num_nodes": 1, "type": "A100"},
        # {"num_nodes": 1, "type": "T4"},
        # {"num_nodes": 1, "type": "T4"},
        # {"num_nodes": 1, "type": "A100"},
        # {"num_nodes": 1, "type": "T4"},
        # {"num_nodes": 1, "type": "L4"},
        # {"num_nodes": 1, "type": "L4"},
        # {"num_nodes": 1, "type": "T4"}
    # ]

    # name = "aux12-l4-1g.ini"
    # cluster_configuration = [
    #     {"num_nodes": 4, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"}
    # ]

    # name = "aux12-l4-2g.ini"
    # cluster_configuration = [
    #     {"num_nodes": 4, "type": "L4"},
    #     {"num_nodes": 4, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"}
    # ]

    # name = "aux12-l4.ini"
    # cluster_configuration = [
    #     {"num_nodes": 1, "type": "L4", "region": "rack1"},
    #     {"num_nodes": 1, "type": "L4", "region": "rack1"},
    #     {"num_nodes": 1, "type": "L4", "region": "rack1"},
    #     {"num_nodes": 1, "type": "L4", "region": "rack1"},

    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},

    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "L4"}
    # ]


    # Scenario: 24 single nodes like original Helix paper
    # name = "aux24helix.ini"
    # cluster_configuration = [
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "A100"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    #     {"num_nodes": 1, "type": "T4"},
    #     {"num_nodes": 1, "type": "L4"},
    # ]

    # name = "aux15-a30.ini"
    # cluster_configuration = [
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 4, "type": "A30"},
    # 	{"num_nodes": 1, "type": "L4"}, 
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "A100"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "L4"}
    # ]

    # name = "aux15-2g-a30.ini"
    # cluster_configuration = [
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 4, "type": "A30"},
    # 	{"num_nodes": 1, "type": "L4"}, 
    # 	{"num_nodes": 1, "type": "L4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 4, "type": "A30"},
    # 	{"num_nodes": 1, "type": "T4"},
    # 	{"num_nodes": 1, "type": "L4"}
    # ]

    generate_config_file(cluster_configuration, f"config/{name}")

import re
import sys
import os
from configparser import ConfigParser
from typing import Dict, Union

from graphviz import Digraph

def parse_cluster_config(ini_file):
    """
    Read cluster configuration from INI file and extract:
    1. Mapping node_id -> node_type
    2. Mapping node_id -> group_id
    """
    node_types = {}
    node_groups = {}
    config = ConfigParser()
    config.read(ini_file)

    compute_nodes = eval(config["ComputeNodes"]["names"])
    for compute_node in compute_nodes:
        entity_name = f"ComputeNode-{compute_node}"
        
        node_type = config[entity_name]["type"]
        node_types[int(compute_node)] = node_type

        group_id = int(config[entity_name]["group"])
        node_groups[int(compute_node)] = group_id

    return node_types, node_groups

def parse_ilp_solution(filename, static_node_groups):
    """
    Parse ILP solution file.
    """
    data = {
        "starts": {},      # node_id -> start_layer
        "holds": {},       # node_id -> num_layers
        "tp_active": {},   # group_id -> bool
        "node_group": {},  # node_id -> group_id
        "group_nodes": {}, # group_id -> list of node_ids
        "edges": []        # list of (from_type, from_id, to_type, to_id, value)
    }

    for n_id, g_id in static_node_groups.items():
        data["node_group"][n_id] = g_id
        if g_id not in data["group_nodes"]:
            data["group_nodes"][g_id] = []
        data["group_nodes"][g_id].append(n_id)

    name_2_val: Dict[str, Union[int, float]] = {}
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2: continue
            name, val = parts[0], parts[1]
            name_2_val[name] = eval(val)

    EPS = 1e-4

    for name, val in name_2_val.items():
        # 1. Start variables
        if name.startswith("start_"):
            n_id = int(name.split("_")[1])
            data["starts"][n_id] = int(val)
            
        elif name.startswith("hold_"):
            parts = name.split("_")
            n_id = int(parts[1])
            if val > 0.5:
                data["holds"][n_id] = int(parts[2])
                
        elif name.startswith("group_tp_active_"):
            g_id = int(name.split("_")[-1])
            data["tp_active"][g_id] = int(val)

        # 2. Edges
        elif val > EPS:
            if name.startswith("flow_source_nic_in_"):
                g_id = int(name.split("_")[-1])
                data["edges"].append(('source', 'source', 'group', g_id, float(val)))
                
            elif name.startswith("flow_nic_out_") and name.endswith("_sink"):
                g_id = int(name.split("_")[3])
                data["edges"].append(('group', g_id, 'sink', 'sink', float(val)))
                
            elif name.startswith("flow_nic_out_") and "_nic_in_" in name:
                parts = name.split("_")
                g_u = int(parts[3])
                g_v = int(parts[6])
                data["edges"].append(('group', g_u, 'group', g_v, float(val)))
                
            elif name.startswith("flow_") and not name.startswith("flow_nic"):
                parts = name.split("_")
                if len(parts) == 3:
                    u, v = int(parts[1]), int(parts[2])
                    data["edges"].append(('node', u, 'node', v, float(val)))

    return data

def draw_graph(data, output_filename, node_types, view=False):
    dot = Digraph(comment='Helix Pipeline', format='png')
    dot.attr(rankdir='LR')
    dot.attr('node', fontname='Helvetica', fontsize='12')

    # 1. Source/Sink nodes
    dot.node('Source', 'Source', shape='diamond', style='filled', fillcolor='lightblue')
    dot.node('Sink', 'Sink', shape='diamond', style='filled', fillcolor='lightblue')

    inactive_nodes = []
    # 2. Draw groups and compute nodes
    for g_id, nodes in data["group_nodes"].items():
        is_tp = data["tp_active"].get(g_id, 0) == 1
        
        nodes.sort()
        
        node_info = []
        for n_id in nodes:
            s = data["starts"].get(n_id, "?")
            h = data["holds"].get(n_id, 0)
            
            t = node_types.get(n_id, "Unknown")

            if h > 0:
                layer_str = f"Layers: {s} - {s + h - 1}"
                node_info.append((n_id, layer_str, t))
            else:
                inactive_nodes.append((n_id, t))
            
        # TP ON
        if is_tp and data["holds"].get(nodes[0], 0) > 0:
            n_ids_str = ",".join(str(n) for n, _, _ in node_info)
            layers_str = node_info[0][1] if node_info else "?"

            g_type = node_info[0][2] if node_info else "?"
            
            label = f"Group {g_id}\n(TP Active)\nNodes: [{n_ids_str}]\n{layers_str}\nType: {g_type}"
            dot.node(f"G{g_id}", label, shape='box', style='filled', fillcolor='lightgrey', width="2", height="1.2", fixedsize="true")
        
        else: # TP OFF
            with dot.subgraph(name=f'cluster_G{g_id}') as c:
                c.attr(label="", style='invis')
                
                for n_id, layer_str, t in node_info:
                    label = f"Node {n_id}\n{layer_str}\nType: {t}"
                    c.node(f"N{n_id}", label, shape='circle', width="1.5", fixedsize="true")

    if inactive_nodes:
        with dot.subgraph(name='cluster_inactive') as c:
            c.attr(label="Inactive Nodes", style='invis')

            prev_inactive = None

            inactive_nodes.sort(key=lambda x: x[0])
            
            for n_id, t in inactive_nodes:
                node_name = f"Inactive_{n_id}"
                label = f"Node {n_id}\nInactive\nType: {t}"
                c.node(node_name, label, shape='circle', color='grey', width="1.5", fixedsize="true")

                if prev_inactive:
                    c.edge(prev_inactive, node_name, style='invis')
                prev_inactive = node_name


    # 3. draw edges
    def get_anchor(type_, id_, is_source=False):
        if type_ == 'source': return 'Source'
        if type_ == 'sink': return 'Sink'
        if type_ == 'group':
            if data["tp_active"].get(id_, 0) == 1:
                return f"G{id_}"
            else:
                nodes = data["group_nodes"].get(id_, [])
                if not nodes: return None
                
                active_nodes = [n for n in nodes if data["holds"].get(n, 0) > 0]
                if not active_nodes: active_nodes = nodes 

                if is_source: 
                    anchor_node = max(active_nodes, key=lambda n: data["starts"].get(n,0) + data["holds"].get(n,0))
                else: 
                    anchor_node = min(active_nodes, key=lambda n: data["starts"].get(n,0))
                
                return f"N{anchor_node}"
        if type_ == 'node':
            g_id = data["node_group"].get(id_, -1)
            if data["tp_active"].get(g_id, 0) == 1:
                return f"G{g_id}"
            else:
                return f"N{id_}"

    throughput = 0.0
    for type_u, id_u, type_v, id_v, tp in data["edges"]:
        u = get_anchor(type_u, id_u, is_source=True)
        v = get_anchor(type_v, id_v, is_source=False)
        
        if u and v and u != v:
            dot.edge(u, v, label=str(round(tp, 1)))
        if type_u == "source":
            throughput += tp

    folder_name = os.path.basename(os.path.dirname(output_filename))
    dot.attr(label=f"{folder_name} - Throughput (token/s): {round(throughput,1)}", labelloc='t', fontsize='18')

    # Output
    output_path = dot.render(output_filename, cleanup=True, view=view)


def output_solution(ini_file: str, sol_path: str, folder_path: str, view=False) -> None:
    node_types, node_groups = parse_cluster_config(ini_file)
    parsed_data = parse_ilp_solution(sol_path, node_groups)
    
    output_name = os.path.join(folder_path, "solution_graph")
    draw_graph(parsed_data, output_name, node_types, view=view)

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
    current_dir = os.getcwd()
    layouts_dir = os.path.join(current_dir, "layouts")
    configs_dir = os.path.join(current_dir, "config")
    
    aux_folders = [d for d in os.listdir(layouts_dir) if os.path.isdir(os.path.join(layouts_dir, d)) and (d.startswith("ilp_test"))]

    if not aux_folders:
        sys.exit(1)

    for folder in aux_folders:
        folder_path = os.path.join(layouts_dir, folder)
        sol_path = os.path.join(folder_path, "ilp_solution.sol")
        
        suffix = find_ini(folder_path)
        if suffix is None:
            suffix = folder.split('ilp_')[-1]
        ini_file = os.path.join(configs_dir, f"{suffix}.ini")
        
        if os.path.exists(sol_path) and os.path.exists(ini_file):
            output_solution(ini_file, sol_path, folder_path)
            
        else:
            if not os.path.exists(sol_path):
                print(f"Solution file not found: {sol_path}")
            if not os.path.exists(ini_file):
                print(f"Config file not found: {ini_file}")
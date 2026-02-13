from simulator.event_simulator.cluster_simulator import ClusterSimulator, ModelName, SchedulingMethod
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import KVParameters, SchedulingMode

import configparser
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_final_cluster():
    # ---------------------------------------- Initialization ---------------------------------------- #
    # load the model placement
    # cluster_file_path is "simulator_cluster_file_name" in layout_args
    machine_num_dict = {"A100": 4, "T4": 12, "L4": 8}
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="config/single20.ini",
        machine_profile_name="config/machine_profile.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./sim_files/maxflow_offline/",
        layout_method=LayoutMethod.LoadExisting,
        machine_num_dict=machine_num_dict
    )
    layout_args = {
        "solution_file_name": "./layouts/ilp/ilp_sol.ini",
        "simulator_cluster_file_name": "./layouts/ilp/simulator_cluster.ini",
    }
    cluster_file_path = layout_synthesizer.synthesize(args=layout_args)

    # initialize the simulator and set scheduler as MaxFlow scheduler
    simulator = ClusterSimulator(model_name=ModelName.LLaMa70B, machine_num_dict=machine_num_dict)
    simulator.from_ini_file(config_file_name=cluster_file_path)
    scheduler_args = {
        "kv_param": KVParameters(expected_kv_hwm=0.85, expected_output_length_ratio=1),
        "scheduling_mode": SchedulingMode.Offline,  # offline
    }
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()

    # load model placement and update scheduler
    finish_model_loading_time = layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # print some status information
    print(f"Max compute throughput = {layout_synthesizer.layout_synthesizer.get_flow_upper_bound()}")
    print(f"Max flow = {simulator.scheduler.core.flow_graph.flow_value}")
    simulator.visualize_cluster(title="model_placement", save_path="./")


def visualize_initial_cluster(config_file_path: str, show_edges: bool = True):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Build graph
    graph = nx.Graph()

    graph.add_node("Source", type="Source", group=-1)
    graph.add_node("Sink", type="Sink", group=-1)

    groups_dict = {"Source": -1, "Sink": -1}
    color_map = ["black", "black"]  # Source and Sink colors

    for section in config.sections():
        if not section.startswith("AuxiliaryNode"):
            continue
        node_id = section.split("-")[1]
        node_type="Auxiliary"
        group = int(config[section]["group"])
        graph.add_node(node_id, type=node_type, group=group)
        color_map.append("gray")

        connected_nodes = eval(config[section]["connected_nodes"])
        for c in connected_nodes:
            if c == "source":
                graph.add_edge("Source", node_id, color="black", width=1)
            elif c == "sink":
                graph.add_edge(node_id, "Sink", color="black", width=1)
            else:
                graph.add_edge(node_id, str(c), color="black", width=1)
    
    for section in config.sections():
        if not section.startswith("ComputeNode"):
            continue
        node_id = section.split("-")[1]
        node_type = config[section]["type"]
        group = int(config[section]["group"])
        graph.add_node(node_id, type=node_type, group=group)

        color_map.append("red" if group != -1 else "blue")

        # connected_nodes = eval(config[section]["connected_nodes"])
        # for c in connected_nodes:
        #     if c == "source":
        #         graph.add_edge("Source", node_id)
        #     elif c == "sink":
        #         graph.add_edge(node_id, "Sink")
        #     else:
        #         graph.add_edge(node_id, str(c))

    for section in config.sections():
        if not section.startswith("InternalLink"):
            continue
        node1 = config[section]["from_node"].split("-")[1]
        node2 = config[section]["to_node"].split("-")[1]

        if graph.nodes()[node1]["group"] != -1:
            graph.add_edge(node1, node2, color="red", width=2)
        else:
            graph.add_edge(node1, node2, color="blue", width=1)

    # # Edge colors
    # for u, v in graph.edges():
    #     gu, gv = graph.nodes[u]["group"], graph.nodes[v]["group"]
    #     graph.edges[u, v]["color"] = "red" if gu == gv and gu != -1 else "black"
    #     graph.edges[u, v]["width"] = 2 if gu == gv and gu != -1 else 1

    # # Node colors
    # groups = {}
    # for n, data in graph.nodes(data=True):
    #     # Group assignment
    #     g = data["group"]
    #     groups.setdefault(g, []).append(n)

    #     # Node color mapping
    #     if data["type"] in ("Source", "Sink"):
    #         color_map.append("black")
    #     elif g!= -1:
    #         color_map.append("red")
    #     else:
    #         color_map.append("gray")
    
    # init_pos = {}
    # for i, (g, nodes) in enumerate(groups.items()):
    #     if g != -1:
    #         base = np.array([i, 0])
    #         for j, n in enumerate(nodes):
    #             r = j//2
    #             c = j%2
    #             init_pos[n] = base + np.array([r*0.1, -c*0.1])
    
    # if len(init_pos) == 0:
    #     pos = nx.spring_layout(graph, seed=0)
    # else:
    #     pos = nx.spring_layout(graph, pos=init_pos, fixed=init_pos.keys(), seed=0)

    pos = nx.spring_layout(graph, seed=0)
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=500, edgecolors="black", node_shape="o")
    
    if show_edges:
        edge_colors = [graph.edges[e]["color"] for e in graph.edges()]
        edge_widths = [graph.edges[e]["width"] for e in graph.edges()]
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=edge_widths, edge_color=edge_colors)
    
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif", font_color="white")

    plt.title("Initial Cluster Topology")
    plt.show()


def main():
    visualize_initial_cluster(config_file_path="./config/single12group.ini", show_edges=True)
    # visualize_initial_cluster(config_file_path="./single24.ini", show_edges=False)

if __name__ == "__main__":
    main()
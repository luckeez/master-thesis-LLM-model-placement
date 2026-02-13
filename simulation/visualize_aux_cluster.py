import configparser
import networkx as nx
import matplotlib.pyplot as plt

def visualize_cluster_links(ini_file_path):
    # 1. Initialize Configuration and Graph
    config = configparser.ConfigParser()
    config.read(ini_file_path)
    
    G = nx.DiGraph() # Directed Graph

    # Lists for node coloring
    compute_nodes = []
    nic_in_nodes = []
    nic_out_nodes = []
    special_nodes = [] # Source/Sink

    # 2. Parse Nodes (to assign types and colors)
    for section in config.sections():
        # Compute Nodes
        if section.startswith("ComputeNode-"):
            # Format: ComputeNode-0 -> node name is "0"
            node_name = section.split("-")[1]
            G.add_node(node_name)
            compute_nodes.append(node_name)
        
        # Auxiliary Nodes (NICs)
        elif section.startswith("AuxiliaryNode-"):
            # Format: AuxiliaryNode-nic_in_0 -> node name is "nic_in_0"
            node_name = section.split("-", 1)[1]
            node_type = config[section].get('node_type', 'UNKNOWN')
            G.add_node(node_name)
            
            if 'NIC_IN' in node_type:
                nic_in_nodes.append(node_name)
            elif 'NIC_OUT' in node_type:
                nic_out_nodes.append(node_name)
        
        # Source and Sink
        elif section == 'SourceNode':
            G.add_node('source')
            special_nodes.append('source')
        elif section == 'SinkNode':
            G.add_node('sink')
            special_nodes.append('sink')

    # 3. Parse Edges EXCLUSIVELY from [Link-...] sections
    # Format expected: [Link-SourceNode-TargetNode]
    for section in config.sections():
        if section.startswith("Link-"):
            parts = section.split('-')
            # Example: Link-nic_out_0-nic_in_1 splits into ['Link', 'nic_out_0', 'nic_in_1']
            if len(parts) >= 3:
                source = parts[1]
                target = parts[2]
                
                # Add directed edge
                G.add_edge(source, target)

    # 4. Visualization
    plt.figure(figsize=(14, 10))
    
    # Layout algorithm: Kamada-Kawai is usually good for clusters
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Draw Nodes by Category
    nx.draw_networkx_nodes(G, pos, nodelist=compute_nodes, node_color='#a6cee3', node_size=600, label='Compute')
    nx.draw_networkx_nodes(G, pos, nodelist=nic_in_nodes, node_color='#b2df8a', node_size=500, label='NIC IN')
    nx.draw_networkx_nodes(G, pos, nodelist=nic_out_nodes, node_color='#1f78b4', node_size=500, label='NIC OUT')
    nx.draw_networkx_nodes(G, pos, nodelist=special_nodes, node_color='#e31a1c', node_size=800, node_shape='s', label='Source/Sink')

    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Draw Edges (Arrows)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=12, alpha=0.5)

    plt.title("Cluster Topology (Unidirectional Links from .ini)", fontsize=14)
    plt.legend(loc="upper right")
    plt.axis('off')
    
    # Save or Show
    plt.tight_layout()
    plt.savefig('cluster_topology.png')
    plt.show()

# Run the function
if __name__ == "__main__":
    visualize_cluster_links('config/aux4.ini')
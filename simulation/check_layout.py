import sys
import os
from gurobipy import GRB

MEMORY = True

from src.ilp_layout_aux import ILPLayout, NodeType
from simulation.visualize_solution import output_solution


CLUSTER_CONFIG = "config/aux12-mem.ini"        
MODEL_NAME = "LLaMa30B"                 

TEST_ASSIGNMENT = {
    # Format: "nodes": {node_id: (start_layer, num_layers)}, "groups": {group_id: is_tp_active}
    "nodes": {
        "0": (0, 10),  
        "1": (10, 11),
        "2": (10, 10),
        "3": (21, 7),
        "4": (20, 7),
        "5": (27, 7),
        "6": (60, 0),
        "7": (60, 0),
        "8": (28, 6),
        "9": (34, 16),
        "10": (50, 5),
        "11": (55, 5),
        # "12": (53, 7),
        # "13": (21, 11),
        # "14": (49, 11),
        # "15": (53, 7)

        # "0": (0, 7),
        # "1": (7, 18),
        # "2": (60, 0),
        # "3": (60, 0),
        # "4": (28, 7),
        # "5": (25, 3),
        # "6": (28, 7),
        # "7": (35, 7),
        # "8": (35, 7),
        # "9": (42, 18),
        # "10": (0, 7),
        # "11": (60, 0)

        # "0": (27, 6),
        # "1": (0, 15),
        # "2": (15, 6),
        # "3": (39, 6),
        # "4": (21, 6),
        # "5": (27, 6),
        # "6": (33, 6),
        # "7": (60, 0),
        # "8": (33, 6),
        # "9": (45, 15),
        # "10": (21, 6),
        # "11": (15, 6)

    },

    "groups": {
        "0": False,
        "1": False,
        "2": False,
        "3": False,
        "4": False,
        # "5": False,
        # "6": False,
        # "7": False,
        "8": False,
        "9": False,
        "10": False,
        "11": False,
        # "12": False,
        # "13": False,
        # "14": False,
        # "15": False
    }
}


def check_feasibility(layout: ILPLayout, assignment: dict):
    model = layout.ilp_model
    
    print("\n" + "="*50)
    print("   CHECK FEASIBILITY & THROUGHPUT")
    print("="*50)
    
    print("--> Applicazione vincoli sui Nodi...")
    for node_idx, (target_start, target_num) in assignment["nodes"].items():
        if str(node_idx) not in layout.ilp_nodes:
            print(f"ERRORE: Nodo {node_idx} non trovato nel cluster!")
            return
        if layout.ilp_nodes[node_idx].node_type != NodeType.COMPUTE:
            continue

        var_start = layout.var_node_start[f"start_{node_idx}"]
        
        # A. Fix start layer
        model.addConstr(var_start == target_start, name=f"FIX_start_{node_idx}")
        
        # B. Fux num layers
        
        if target_num in layout.var_node_hold_layer[node_idx]:
            var_hold = layout.var_node_hold_layer[node_idx][target_num]
            model.addConstr(var_hold == 1, name=f"FIX_hold_{node_idx}_{target_num}")
        else:
            var_is_active = layout.var_node_active[f"{node_idx}"]
            if target_num == 0:
                model.addConstr(var_is_active == 0, name=f"FIX_inactive_{node_idx}")
                continue
            else:
                print(f"ERROR CONFIG:Node {node_idx} cannot host {target_num} layer.")
                print(f"Max supported: {layout.ilp_nodes[node_idx].max_num_layers}")
                return

    for group_id, is_tp in assignment["groups"].items():
        var_name = f"group_tp_active_{group_id}"
        if var_name in layout.var_group_tp_active:
            var_tp = layout.var_group_tp_active[var_name]
            val = 1 if is_tp else 0
            model.addConstr(var_tp == val, name=f"FIX_tp_{group_id}")
        else:
            print(f"WARNING: Group {group_id} not found.")

    model.optimize()

    # Results
    if model.Status == GRB.OPTIMAL:
        print("\n" + "*"*30)
        print("✅  CONFIGURAZIONE VALIDA")
        print("*"*30)
        
        throughput = model.ObjVal
        print(f"\nThroughput: {throughput:.4f} token/s")
        
        for n_id, (s, num) in assignment["nodes"].items():
            g_id = layout.ilp_nodes[n_id].group
            tp_status = "TP ON" if assignment["groups"].get(str(g_id), False) else "TP OFF"
            print(f"  - Node {n_id} (Group {g_id}, {tp_status}): Layers {s} -> {s+num-1} ({num} layers)")

        TEMP_SOL_PATH = "temp_solution.sol"
        model.write(TEMP_SOL_PATH)

        try:
            output_solution(CLUSTER_CONFIG, TEMP_SOL_PATH, "./", view=True)
        except Exception as e:
            print(f"Visualization error: {e}")

        if os.path.exists(TEMP_SOL_PATH):
            pass       
            # os.remove(TEMP_SOL_PATH)

    elif model.Status == GRB.INFEASIBLE:
        print("\n" + "!"*30)
        print("INFEASIBLE")
        print("!"*30)
        
        model.computeIIS()
        print("\nConstraints violated:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  - {c.ConstrName}")

    else:
        pass

if __name__ == "__main__":
    model_name = "LLaMa30B"
    
    layout = ILPLayout()
    layout.from_ini(CLUSTER_CONFIG, MODEL_NAME, MEMORY, batch_size=8)
    
    print("Costruzione modello MILP...")
    layout.build_model(seed=42, model_name="checker")

    check_feasibility(layout, TEST_ASSIGNMENT)

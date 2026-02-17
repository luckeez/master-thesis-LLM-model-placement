# 2026.02.12: Luca Pedercini

import math
import time

import gurobipy as gp

from typing import List, Dict, Tuple, Set
from configparser import ConfigParser
from gurobipy import GRB
from enum import Enum
import matplotlib.pyplot as plt

from src.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec, ATOL, is_close

from src.formulas import calc_throughput
from src.specs import ModelSpec, GPUSpec, LinkSpec, MODEL_SPECS, GPU_SPECS, LINK_SPECS, BYTE_PER_PARAM
from simulation.visualize_solution import output_solution


class NodeType(Enum):
    COMPUTE = 1
    SOURCE = 2
    SINK = 3
    NIC_IN = 4
    NIC_OUT = 5

class ILPLink:
    def __init__(self, from_index: str, to_index: str, throughput: float,
                 bandwidth: float, latency: float) -> None:
        """
        Represent a link in ILP. Contains all information needed.
        Note: The link is bidirectional!

        :param from_index: index of the start node
        :param to_index: index of the destination node
        :param throughput: token throughput over this link
        :param bandwidth: bandwidth of this link
        :param latency: latency of this link
        :return: None
        """
        self.from_index: str = from_index
        self.to_index: str = to_index
        self.throughput: float = throughput
        self.bandwidth: float = bandwidth
        self.latency: float = latency

        # from ilp solution
        # edge_switch: 0 = edge not enabled, 1 = edge enabled
        # forward edge (from -> to)
        self.forward_flow: float = -1.0
        self.forward_edge_switch: int = -1
        self.forward_edge_cond1: int = -1
        self.forward_edge_cond2: int = -1
        # backward edge (to -> from)
        self.backward_flow: float = -1.0
        self.backward_edge_switch: int = -1
        self.backward_edge_cond1: int = -1
        self.backward_edge_cond2: int = -1


class ILPNode:
    def __init__(self, node_index: int = -1, machine_type: GPUSpec = None, mem_bandwidth: float = -1, max_num_layers: int = -1,
                 connected_node_indices: List[int] = [], layer_count_2_throughput: Dict[int, float] = {}, node_type: NodeType = NodeType.COMPUTE, group: int=-1, bottleneck_nic_throughput: float = -1) -> None:
        """
        Represent a compute node in ILP. Contains all information needed.

        :param node_index: index of this node
        :param machine_type: machine type
        :param max_num_layers: max number of layers this machine can hold
        :param connected_node_indices: which nodes are connected to this node
        :param layer_count_2_throughput: throughput when there are k layers on node (bounded by nic speed)
        :return: None
        """
        self.node_index: int = node_index
        self.machine_type: GPUSpec = machine_type
        self.mem_bandwidth: float = mem_bandwidth
        self.max_num_layers: int = max_num_layers
        self.connected_node_indices: List[int] = connected_node_indices
        self.layer_count_2_throughput: Dict[int, float] = layer_count_2_throughput
        self.bottlneck_nic_throughput: float = bottleneck_nic_throughput

        # NEW attributes TP
        self.group: int = group
        self.node_type: NodeType = node_type

        # from ilp solution
        # model on node is [start_idx, end_idx)
        self.start_layer_idx: int = -1
        self.end_layer_idx: int = -1
        self.num_assigned_layers: int = -1


class ILPLayout:
    # Usage:
    # 1. call "from_ini" to load a complete cluster topology and machine profile
    # 2. call "build_model" to build the ILP model
    # 3. call "search_layout" to let Gurobi find the optimal solution
    # 4. visualize solution TODO edit 

    def __init__(self) -> None:
        """
        MILP-based initial layout Synthesizer.

        :return: None
        """
        # loaded problem information
        self.model_spec: ModelSpec | None = None
        # self.model_manager: ModelManager = model_manager
        self.cluster_file_name: str = ""
        self.enable_memory: bool = False
        self.batch_size: int = -1
        self.tp_only: bool = False

        # cluster information
        self.ilp_source: ILPNode | None = None
        self.ilp_sink: ILPNode | None = None
        self.ilp_nodes: Dict[int, ILPNode] = {}
        self.ilp_links: Dict[Tuple[str, str], ILPLink] = {}
        self.cluster_loaded: bool = False
        self.solution_loaded: bool = False

        # ilp model
        self.model_initialized: bool = False
        self.ilp_model: gp.Model | None = None

        # variables
        self.var_node_start: Dict[str, gp.Var] = {}
        self.var_node_hold_layer: Dict[int, Dict[str, gp.Var]] = {}
        self.var_flow: Dict[str, gp.Var] = {}
        self.var_edge_switch: Dict[str, gp.Var] = {}
        self.var_group_starts: Dict[str, gp.Var] = {}
        self.var_group_ends: Dict[str, gp.Var] = {}
        # tmp variables that are only used when allow partial inference
        self.tmp_var_compute_edge_cond1: Dict[str, gp.Var] = {}
        self.tmp_var_compute_edge_cond2: Dict[str, gp.Var] = {}

        # constraints
        self.constr_hold: Dict[str, gp.Constr] = {}
        self.constr_end: Dict[str, gp.Constr] = {}
        self.constr_node_flow: Dict[str, gp.Constr] = {}
        self.constr_node_throughput: Dict[str, gp.Constr] = {}
        self.constr_intergroup_throughput: Dict[str, gp.Constr] = {}
        self.constr_edge_enabled: Dict[str, gp.Constr] = {}
        self.constr_edge_disabled: Dict[str, gp.Constr] = {}
        self.constr_edge_flow: Dict[str, gp.Constr] = {}
        # tmp constraints that are only used when allow partial inference
        self.tmp_constr_cond1_enabled: Dict[str, gp.Constr] = {}
        self.tmp_constr_cond1_disabled: Dict[str, gp.Constr] = {}
        self.tmp_constr_cond2_enabled: Dict[str, gp.Constr] = {}
        self.tmp_constr_cond2_disabled: Dict[str, gp.Constr] = {}

        # NEW attributes TP
        self.gpu_groups: Dict[int, List[int]] = {}
        self.tp_throughput_profiles: Dict[int, Dict[int, float]] = {} # [group_id, Dict[layer_count, thoughput]]

        # NEW variables TP
        self.var_group_tp_active: Dict[str, gp.Var] = {}

        # MEM variables 
        self.var_node_active: Dict[str, gp.Var] = {}
        self.total_vram: int = -1
        self.max_throughput: float = -1
        self.opt_alpha: float = 1.0

        # NEW constraints TP
        self.constr_force_group_off: Dict[str, gp.Constr] = {}
        self.constr_nic_throughput: Dict[str, gp.Constr] = {}
        self.constr_group_synch_start: Dict[str, gp.Constr] = {}
        self.constr_group_synch_hold: Dict[str, gp.Constr] = {}
        self.constr_tp_off_limit: Dict[str, gp.Constr] = {}
        self.constr_tp_on_limit: Dict[str, gp.Constr] = {}
        self.constr_group_throughput: Dict[str, gp.Constr] = {}

        # optimization status
        # stopping criteria
        self.max_run_time: float = -1
        self.early_stop_threshold: float = -1
        self.early_stop_time: float = -1
        # run time
        self.opt_start_time: float = -1
        self.opt_best_obj: float = -1
        self.opt_best_obj_found_time: float = -1
        self.opt_upper_bound: float = -1

        # parameters for simulator cluster file and initial layout generation
        self.node_idx_offset = 2  # since 0 and 1 are reserved for source and sink

        # NEW global max num layers
        self.global_max_num_layers: int = -1  # arbitrary number

    def from_ini(self, cluster_file_name: str, model_name: str, enable_memory: bool, batch_size: int, tp_only: bool) -> None:
        """
        Initialize the ILP using a given cluster topology and machine profiles.

        :param cluster_file_name: name of the file that stores cluster topology
        :param machine_profile_name: name of the file that stores machine profiling results
        :return: None
        """
        # clear the dicts
        self.ilp_nodes.clear()
        self.ilp_links.clear()

        # Enable Memory
        self.enable_memory = enable_memory
        self.batch_size = batch_size
        self.tp_only = tp_only

        # # load machine statistics
        # machine_profile_parser = ConfigParser()
        # machine_profile_parser.read(machine_profile_name)
        # for machine_name in machine_profile_parser.sections():
        #     self.machine_profiles[machine_name] = MachineProfile(machine_name=machine_name,
        #                                                          config=machine_profile_parser)

        # load cluster topology
        cluster_file_parser = ConfigParser()
        cluster_file_parser.read(cluster_file_name)
        self.cluster_file_name = cluster_file_name

        # model
        self.model_spec = MODEL_SPECS[model_name]
        self.bigM = self.model_spec.n_layers + 1  

        # source and sink
        self.ilp_source = ILPNode(connected_node_indices=eval(cluster_file_parser["SourceNode"]["connected_nodes"]),
                                  node_type=NodeType.SOURCE)
        self.ilp_sink = ILPNode(connected_node_indices=eval(cluster_file_parser["SinkNode"]["connected_nodes"]),
                                 node_type=NodeType.SINK)

        # Auxiliary nodes
        auxliary_nodes: List[str] = eval(cluster_file_parser["AuxiliaryNodes"]["names"])
        for aux_node in auxliary_nodes:
            entity_name: str = f"AuxiliaryNode-{aux_node}"
            node_idx: str = aux_node
            machine_type: GPUSpec = GPU_SPECS[cluster_file_parser[entity_name]["type"]]
            node_type_str: str = cluster_file_parser[entity_name]["node_type"]
            if node_type_str == "NIC_IN":
                node_type: NodeType = NodeType.NIC_IN
            else:
                node_type = NodeType.NIC_OUT

            group: int = int(cluster_file_parser[entity_name]["group"])
            connected_nodes: List[int | str] = eval(cluster_file_parser[entity_name]["connected_nodes"])

            self.ilp_nodes[node_idx] = ILPNode(node_index=node_idx,
                                              machine_type=machine_type,
                                              max_num_layers=-1,
                                              connected_node_indices=connected_nodes,
                                              layer_count_2_throughput={},
                                              node_type=node_type,
                                              group=group)
        
        # Compute nodes
        compute_nodes: List[str] = eval(cluster_file_parser["ComputeNodes"]["names"])
        for compute_node in compute_nodes:
            entity_name: str = f"ComputeNode-{compute_node}"
            node_idx: str = str(compute_node)
            machine_type: GPUSpec = GPU_SPECS[cluster_file_parser[entity_name]["type"]]
            group: int = int(cluster_file_parser[entity_name]["group"])
            node_type_str: NodeType = NodeType.COMPUTE
            connected_nodes: List[int | str] = eval(cluster_file_parser[entity_name]["connected_nodes"])
            mem_bandwidth: float = machine_type.memory_bandwidth_gbps
            
            # compute max number of layers that can be stored on this node
            # Note: max # layers = (VRAM size / 2) / layer size
            # HACK use 0.92 factor to have values equal to helix
            # layer_size_bytes: int = max(self.model_manager.get_model_params())
            layer_size_bytes: int = self.model_spec.memory_bytes_per_layer
            max_num_layers: int = int((machine_type.memory_gb * GB/ 2) / layer_size_bytes)
            
            # Save bottleneck nic throughput
            # bottleneck_nic_speed: float = min(machine_type.inbound_nic_speed, machine_type.outbound_nic_speed)
            # bottleneck_nic_throughput: float = bottleneck_nic_speed / self.model_card.activation_size
            
            # NEW Use throughput formula
            layer_count_2_throughput = {}
            for layer_count in range(1, max_num_layers + 1):
                # layer_count_2_throughput[layer_count] = compute_inference_throughput(
                #     mem_bandwidth=mem_bandwidth,
                #     k=layer_count
                # )
                # HACK define here model, gpu and link specs. 
                # TODO integrate specs in the simulator
                layer_count_2_throughput[layer_count] = calc_throughput(
                    model=self.model_spec, gpu=machine_type, n_gpu=1, batch_size=self.batch_size, assigned_layers=layer_count
                )

            # add compute node to group
            self.gpu_groups.setdefault(group, []).append(node_idx)

            # add node
            self.ilp_nodes[node_idx] = ILPNode(node_index=node_idx,
                                                machine_type=machine_type,
                                                mem_bandwidth=mem_bandwidth,
                                                max_num_layers=max_num_layers,
                                                connected_node_indices=connected_nodes,
                                                layer_count_2_throughput=layer_count_2_throughput,
                                                node_type=NodeType.COMPUTE,
                                                group=group,
                                                # bottleneck_nic_throughput=bottleneck_nic_throughput
            )


        # NEW compute tp_throughput_profiles
        for group in self.gpu_groups.keys():

            # store inference throughput profile
            # HACK big semplification
            size: int = len(self.gpu_groups[group])

            # if single gpu in the group, use its own profile: group throughput = node throughput
            if size == 1:
                node_idx = self.gpu_groups[group][0]
                self.tp_throughput_profiles[group] = self.ilp_nodes[node_idx].layer_count_2_throughput

            # else if multiple gpus in the group, build a new profile: group throughput = sum of node throughput
            else:
                type: str = self.ilp_nodes[self.gpu_groups[group][0]].machine_type.name
                mem_bandwidth: float = self.ilp_nodes[self.gpu_groups[group][0]].mem_bandwidth
                max_group_num_layers: int = self.ilp_nodes[self.gpu_groups[group][0]].max_num_layers * size

                group_layer_count_2_throughput: Dict[int, float] = {}
                for layer_count in range(1, max_group_num_layers + 1):
                    # group_inference_throughput: float = compute_group_inference_throughput(
                    #     mem_bandwidth=mem_bandwidth,
                    #     k=layer_count,
                    #     group_size=size
                    # )
                    # TODO link definition in config file for intra-group links
                    gpu: GPUSpec = GPU_SPECS[type]
                    link_type: str = "PCIe"
                    group_inference_throughput: float = calc_throughput(
                        model=self.model_spec, gpu=gpu, n_gpu=size, batch_size=self.batch_size, link_type=link_type, assigned_layers=layer_count
                    )
                    group_layer_count_2_throughput[layer_count] = group_inference_throughput
                self.tp_throughput_profiles[group] = group_layer_count_2_throughput

        # NEW compute global max num layers
        self.global_max_num_layers = self.compute_global_max_num_layers()

        # links
        # Note: links here are bidirectional
        for entity_name in cluster_file_parser.sections():
            if "Link-" in entity_name:
                # end points
                from_idx: str = entity_name.split("-")[1]
                to_idx: str = entity_name.split("-")[2]

                # bandwidth and latency, throughput = capacity
                bandwidth: float = eval(cluster_file_parser[entity_name]["bandwidth"])
                latency: float = eval(cluster_file_parser[entity_name]["latency"])
                if from_idx == "source" or to_idx == "sink":
                    throughput: float = bandwidth / self.model_spec.token_size
                else:
                    throughput: float = bandwidth / self.model_spec.activation_size
                self.ilp_links[(from_idx, to_idx)] = ILPLink(from_index=from_idx,
                                                             to_index=to_idx,
                                                             throughput=throughput,
                                                             bandwidth=bandwidth,
                                                             latency=latency)
                
        if self.enable_memory:
            self.opt_alpha = float(cluster_file_parser["Settings"]["optimization_alpha"])

        # mark cluster as loaded
        self.cluster_loaded = True


    def step1_initialize_ilp(self, seed: int, model_name: str) -> None:
        """
        Initialize the ILP program.

        :param seed: random seed
        :param model_name: name of the model
        :return: None
        """
        self.ilp_model = gp.Model(model_name)
        self.ilp_model.Params.Seed = seed

        # variables
        self.var_node_start.clear()
        self.var_node_hold_layer.clear()
        self.var_flow.clear()
        self.var_edge_switch.clear()
        self.tmp_var_compute_edge_cond1.clear()
        self.tmp_var_compute_edge_cond2.clear()

        # constraints
        self.constr_hold.clear()
        self.constr_end.clear()
        self.constr_node_flow.clear()
        self.constr_node_throughput.clear()
        self.constr_intergroup_throughput.clear()
        self.constr_edge_enabled.clear()
        self.constr_edge_disabled.clear()
        self.constr_edge_flow.clear()
        self.tmp_constr_cond1_enabled.clear()
        self.tmp_constr_cond1_disabled.clear()
        self.tmp_constr_cond2_enabled.clear()
        self.tmp_constr_cond2_disabled.clear()

        # new variables TP
        self.var_group_tp_active.clear()

        # new constraints TP
        self.constr_force_group_off.clear()
        self.constr_nic_throughput.clear()
        self.constr_group_synch_start.clear()
        self.constr_group_synch_hold.clear()
        self.constr_tp_off_limit.clear()
        self.constr_tp_on_limit.clear()
        self.constr_group_throughput.clear()

    def step2_add_variables(self) -> Tuple[int, int, int]:
        """
        Add decision variables into this ILP program.

        :param allow_partial_inference: whether we allow partial inference or not
        :param remove_redundant: remove redundant constraints in the model
        :param start_from_heuristic: whether we start from a heuristic solution
        :param heuristic_sol_path: path to the heuristic solution
        :return: num_int, num_real, num_binary
        """
        num_int, num_real, num_binary = 0, 0, 0


        # Step 2.1: add starting layer index for each node as variable
        # var_name: start_i (s_i)
        # var_type: int
        # var_range: {0, 1, ..., # layers  - 1}
        # number of variables: n
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            if not compute_node.node_type == NodeType.COMPUTE:
                continue
            start_var_name = f"start_{compute_node_idx}"
            start_var = self.ilp_model.addVar(vtype=GRB.INTEGER,lb=0,name=start_var_name)

            self.var_node_start[start_var_name] = start_var
            num_int += 1


        # Step 2.2: add whether each node holds k layers as variable
        # var_name: hold_i_k (b_ik)
        # var_type: bool
        # var_range: {0, 1}
        # number of variables: kn
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            if not compute_node.node_type == NodeType.COMPUTE:
                continue
            self.var_node_hold_layer[compute_node_idx] = {}
            # NEW range(1, global_max_num_layers) defined as 20 (arbitrary number) HACK
            for layer_count in range(1, self.global_max_num_layers+1):
            # for layer_count in range(1, compute_node.max_num_layers + 1):
                hold_var_name = f"hold_{compute_node_idx}_{layer_count}"
                hold_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                 name=hold_var_name)

                # self.var_node_hold_layer[compute_node_idx][hold_var_name] = hold_var
                self.var_node_hold_layer[compute_node_idx][layer_count] = hold_var
                num_binary += 1

        # Step 2.3: add flow over each edge as variable
        # var_name: flow_i_j (f_ij)
        # var_type: continuous
        # var_range: [0, +inf)
        # number of variables: 2e
        for link_name_tuple, link in self.ilp_links.items():
            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                flow_var_name = f"flow_{link_name_tuple[0]}_{link_name_tuple[1]}"
                flow_var = self.ilp_model.addVar(vtype=GRB.CONTINUOUS,
                                                 lb=0,
                                                 ub=GRB.INFINITY,
                                                 name=flow_var_name)
                self.var_flow[flow_var_name] = flow_var
                num_real += 1

        # Step 2.4 add whether each edge is enabled (edge switch) as variable
        # var_name: switch_i_j (d_ij)
        # var_type: bool
        # var_range: {0, 1}
        # number of variables: 2e
        for link_name_tuple, link in self.ilp_links.items():

            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                forward_switch_name = f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"
                forward_switch_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                           name=forward_switch_name)
                self.var_edge_switch[forward_switch_name] = forward_switch_var
                num_binary += 1


        # NEW Step 2.6 add group throughput active variables
        # var_name: group_tp_active_g
        # var_type: bool
        # var_range: {0, 1}
        # number of variables: num_groups
        for group_id in self.gpu_groups.keys():
            group_tp_active_name = f"group_tp_active_{group_id}"
            group_tp_active_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                        name=group_tp_active_name)
            self.var_group_tp_active[group_tp_active_name] = group_tp_active_var
            num_binary += 1

        if self.enable_memory:
            for node_idx, node in self.ilp_nodes.items():
                if node.node_type != NodeType.COMPUTE:
                    continue
                node_active_name = f"is_active_{node_idx}"
                node_active_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                        name=node_active_name)
                self.var_node_active[node_idx] = node_active_var
                num_binary += 1

                if node_idx in self.var_node_hold_layer:
                    layers_held = gp.quicksum(self.var_node_hold_layer[node_idx].values())

                    self.ilp_model.addConstr(layers_held <= node_active_var * self.bigM, name=f"mem_active_constr_{node_idx}")
                    self.ilp_model.addConstr(layers_held >= node_active_var, name=f"mem_active_constr2_{node_idx}")

                    start_var = self.var_node_start[f"start_{node_idx}"]
                    self.ilp_model.addGenConstrIndicator(
                        node_active_var, False,  # When is_active = 0
                        start_var == self.model_spec.n_layers,  # Invalid position
                        name=f"inactive_start_invalid_{node_idx}"
                    )

        return num_int, num_real, num_binary

    def step3_model_placement_constraint(self) -> int:
        """
        Add model placement constraints.

        :return: number of constraints added
        """
        num_constraint = 0

        for compute_node_idx, compute_node in self.ilp_nodes.items():
            # Step 3.1: add constraint: only one model placement is valid
            # constraint_name: hold_constraint_i
            # constraint: \sum_k hold_i_k = 1 (\sum_k b_ik = 1)
            # number of constraints: n
            if not compute_node.node_type == NodeType.COMPUTE:
                continue

            sum_of_hold_var: gp.LinExpr = gp.quicksum(list(self.var_node_hold_layer[compute_node_idx].values()))
            hold_constraint_name = f"hold_constraint_{compute_node_idx}"

            if self.enable_memory:
                hold_constraint: gp.Constr = self.ilp_model.addConstr(sum_of_hold_var <= 1,
                                                                  name=hold_constraint_name)
            else:
                hold_constraint: gp.Constr = self.ilp_model.addConstr(sum_of_hold_var == 1,
                                                                  name=hold_constraint_name)
            self.constr_hold[hold_constraint_name] = hold_constraint
            num_constraint += 1

            # Step 3.2: add constraint: end layer idx on each node should <= # layers
            # constraint_name: end_constraint_i
            # constraint: start_i + \sum_k k * hold_i_k <= m (s_{i} + \sum_k k * b_{ik} <= m)
            # number of constraints: n
            end_layer_idx_expr: gp.LinExpr = self.get_end_layer_index(compute_node_idx=compute_node_idx)
            end_constraint_name = f"end_constraint_{compute_node_idx}"

            end_constraint: gp.Constr = self.ilp_model.addConstr(end_layer_idx_expr <= self.model_spec.n_layers,
                                                                 name=end_constraint_name)
            self.constr_end[end_constraint_name] = end_constraint
            num_constraint += 1

        return num_constraint

    def step4_flow_in_out_constraint(self) -> int:
        """
        Add constraints for flow in = flow out.

        :return: number of constraints added
        """
        num_constraint = 0

        # Step 4.1: add constraint: flow in = flow out for each compute node
        # constraint_name: node_flow_constraint_i
        # constraint: \sum_{u} flow_u_i = \sum_{j} flow_i_j (\sum_u f_{ui} = \sum_j f_{ij})
        # number of constraints: n
        for node_idx, node in self.ilp_nodes.items():
            if node_idx == "source" or node_idx == "sink":
                continue

            # compute flow
            flow_in_list, flow_out_list = [], []

            for other_idx in node.connected_node_indices:

                # check if "in flow" exists
                in_flow_name = f"flow_{other_idx}_{node_idx}"
                if in_flow_name in self.var_flow:
                    flow_in_list.append(self.var_flow[in_flow_name])

                # check if "out flow" exists
                out_flow_name = f"flow_{node_idx}_{other_idx}"
                if out_flow_name in self.var_flow:
                    flow_out_list.append(self.var_flow[out_flow_name])

            # add constraint
            node_flow_constraint_name = f"node_flow_constraint_{node_idx}"
            node_flow_constraint = self.ilp_model.addConstr(gp.quicksum(flow_in_list) == gp.quicksum(flow_out_list),
                                                            name=node_flow_constraint_name)
            self.constr_node_flow[node_flow_constraint_name] = node_flow_constraint
            num_constraint += 1

        return num_constraint

    def step5_node_throughput_constraint(self) -> int:
        """
        Add constraints for flow over compute nodes.

        :return: number of constraints added
        """
        num_constraint = 0

        # Step 5.1: add constraint: flow over each compute node is smaller than its inference throughput
        # constraint_name: node_throughput_constraint_i
        # constraint: \sum_{u} flow_u_i <= \sum_k hold_i_k * throughput at k layers
        #             (\sum_u f_{ui} <= \sum_k b_{ik} * throughput at k layers)
        # number of constraints: n
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            if not compute_node.node_type == NodeType.COMPUTE:
                continue
            # get flow through the node
            flow_in_list: List[gp.Var] = []
            for other_idx in compute_node.connected_node_indices:
                if not other_idx == "sink" and f"flow_{other_idx}_{compute_node_idx}" in self.var_flow:
                    flow_in_list.append(self.var_flow[f"flow_{other_idx}_{compute_node_idx}"])

            # get one hot throughput
            one_hot_throughput_list: List[gp.LinExpr] = []
            
            for layer_count in range(1, compute_node.max_num_layers + 1):
                throughput_at_k: float = compute_node.layer_count_2_throughput[layer_count]
                hold_k: gp.Var = self.var_node_hold_layer[compute_node_idx][layer_count]
                one_hot_throughput_list.append(throughput_at_k * hold_k)

            # NEW: Indicator logic: if TP is OFF (control_var == 0), throughput limited by individual node throughput
            group_id: int = compute_node.group
            control_var: gp.Var = self.var_group_tp_active[f"group_tp_active_{group_id}"]

            # add constraint
            node_tp_constr_name = f"node_throughput_constraint_{compute_node_idx}"
            node_tp_constr = self.ilp_model.addGenConstrIndicator(
                control_var, False,
                gp.quicksum(flow_in_list) <= gp.quicksum(one_hot_throughput_list),
                name=node_tp_constr_name
            )

            self.constr_node_throughput[node_tp_constr_name] = node_tp_constr
            num_constraint += 1



        # ------------------- LATENCY --------------------
        for link_key, link in self.ilp_links.items():
            from_idx, to_idx = link_key
            from_node = self.ilp_nodes.get(from_idx)
            if from_idx == "source":
                from_node = self.ilp_source
            to_node = self.ilp_nodes.get(to_idx)
            if not (from_node and to_node):
                continue
            if (from_node.node_type != NodeType.NIC_OUT and from_node.node_type != NodeType.SOURCE) or to_node.node_type != NodeType.NIC_IN:
                continue
            L_net = link.latency
            if L_net <= 0:
                continue

            switch_var = self.var_edge_switch.get(f"switch_{from_idx}_{to_idx}")
            group_id = to_node.group
            gpu_idx = sorted(self.gpu_groups[group_id])[0] 
            gpu_node = self.ilp_nodes[gpu_idx]

            # get flow through the node
            flow_in_list: List[gp.Var] = []
            for other_idx in gpu_node.connected_node_indices:
                if not other_idx == "sink" and f"flow_{other_idx}_{gpu_idx}" in self.var_flow:
                    flow_in_list.append(self.var_flow[f"flow_{other_idx}_{gpu_idx}"])

            # get one hot throughput
            adjusted_tp: List[gp.LinExpr] = []
            
            for layer_count in range(1, gpu_node.max_num_layers + 1):
                throughput_at_k: float = gpu_node.layer_count_2_throughput[layer_count]
                hold_k: gp.Var = self.var_node_hold_layer[gpu_idx][layer_count]
                throughput_at_k_with_latency: float = self.batch_size / (self.batch_size / throughput_at_k + L_net)
                adjusted_tp.append(throughput_at_k_with_latency * hold_k)
                
            # add constraint
            intergroup_tp_constr_name = f"constr_intergroup_throughput_{gpu_idx}_{from_idx}_{to_idx}"
            intergroup_tp_constr = self.ilp_model.addGenConstrIndicator(
                switch_var, True,
                gp.quicksum(flow_in_list) <= gp.quicksum(adjusted_tp),
                name=intergroup_tp_constr_name
            )

            self.constr_intergroup_throughput[intergroup_tp_constr_name] = intergroup_tp_constr
            num_constraint += 1


        return num_constraint

    def step6_edge_switch_constraint(self) -> int:
        """
        Add constraint for edge switch variables.

        :param allow_partial_inference: whether partial inference is allowed
        :param remove_redundant: remove redundant constraints in the model
        :return: number of constraints added
        """
        num_constraints = 0

        # Step 6.1: build a list of edges that we need to process
        link_name_list: List[Tuple[str, str]] = []
        for link_name_tuple in self.ilp_links.keys():
            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                link_name_list.append((link_name_tuple[0], link_name_tuple[1]))
            # if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
            #     link_name_list.append((link_name_tuple[1], link_name_tuple[0]))

        # Step 6.2: add constraint for edge switch
        for link_name_tuple in link_name_list:
            assert not (link_name_tuple[0] == "source" and link_name_tuple[1] == "sink"), \
                "Found direct link between source and sink!"
            
            if "nic_out" in link_name_tuple[0] or link_name_tuple[0] == "source":

                # EXTERNAL link from source to nic_in_Y
                # Link is ON iff there is at least one node i in To_group such that
                # start_i == 0
                if link_name_tuple[0] == "source":
                    if not self.ilp_nodes[link_name_tuple[1]].node_type == NodeType.NIC_IN:
                        continue
                    # Case 0: link from source to i
                    # switch = cond(At least one i with start_i == 0 in group of nic_in_Y)

                    # ------------ Prop 1: Linearize b = 1 iff a = 0 ------------ #
                    # int a \in [0, m - 1]
                    # bool b = 0 or 1
                    # express b = 1 iff a = 0
                    # if a = 0 then b = 1:	b >= 1 - a
                    # if a > 0 then b = 0:	a <= m(1-b)
                    # ----------------------------------------------------------- #
                    to_group: int = self.ilp_nodes[link_name_tuple[1]].group
                    to_group_nodes: List[str] = self.gpu_groups[to_group]                    

                    # get the variables
                    edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                    
                    is_start_zero_vars: List[gp.Var] = []
                    for to_node_idx in to_group_nodes:
                        node_i_start_var: gp.Var = self.var_node_start[f"start_{to_node_idx}"]

                        # 1. create tmp var for start_i == 0
                        is_start_zero_name = f"is_start_zero_{to_node_idx}_for_{link_name_tuple[0]}_{link_name_tuple[1]}"
                        is_start_zero_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=is_start_zero_name)
                        is_start_zero_vars.append(is_start_zero_var)

                        # 2. Add "Big M" constraint

                        self.ilp_model.addConstr(
                            node_i_start_var <= self.model_spec.n_layers * (1 - is_start_zero_var),
                            name=f"constr_is_start_zero_{to_node_idx}"
                        )
                        num_constraints += 1

                    # 3. Final constraint on switch
                    # switch <= sum(is_start_zero_vars)
                    # if switch == 1 => at least one is_start_zero_var == 1 => at least one start_i == 0
                    self.ilp_model.addConstr(
                        edge_switch_var <= gp.quicksum(is_start_zero_vars),
                        name=f"constr_edge_switch_start_zero_{to_group}_OR"
                    )
                    num_constraints += 1


                # EXTERNAL link from nic_out_X to sink
                # Link is ON iff there is at least one node i in From_group such that
                # end_i == m                
                elif link_name_tuple[1] == "sink":
                    if not self.ilp_nodes[link_name_tuple[0]].node_type == NodeType.NIC_OUT:
                        continue
                    # Case 0: link from i to sink
                    # switch = cond(At least one i with end_i == m in group of nic_out_X)
                    from_group: int = self.ilp_nodes[link_name_tuple[0]].group
                    from_group_nodes: List[str] = self.gpu_groups[from_group]                    

                    # get the variables
                    edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                    
                    is_end_complete_vars: List[gp.Var] = []
                    for from_node_idx in from_group_nodes:
                        # compute node i's end position
                        node_i_end_expr: gp.LinExpr = self.get_end_layer_index(compute_node_idx=from_node_idx)

                        # 1. create tmp var for end_i == m
                        is_end_complete_name = f"is_end_complete_{from_node_idx}"
                        is_end_complete_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=is_end_complete_name)
                        is_end_complete_vars.append(is_end_complete_var)

                        # 2. Add constraint: IF is_end_complete == 1 THEN end_i == m
                        # end_expr >= m * is_end_complete_var
                        self.ilp_model.addConstr(
                            node_i_end_expr >= self.model_spec.n_layers * is_end_complete_var,
                            name=f"constr_is_end_complete_{from_node_idx}"
                        )
                        num_constraints += 1

                        # self.ilp_model.addConstr(
                        #     node_i_start_var <= (self.model_spec.n_layers - 1) + self.model_spec.n_layers * (1 - is_end_complete_var),
                        #     name=f"constr_is_end_complete2_{from_node_idx}"
                        # )
                        # num_constraints += 1


                        is_active_var = self.var_node_active[from_node_idx]
                        self.ilp_model.addGenConstrIndicator(
                            is_active_var, False,  # When is_active = 0
                            is_end_complete_var == 0,  # end cannot be complete
                            name=f"inactive_end_not_complete_{from_node_idx}"
                        )


                    # 3. Final constraint on switch
                    # switch <= sum(is_end_max_vars)
                    # if switch == 1 => at least one is_end_max_var == 1 => at least one end_i == m
                    self.ilp_model.addConstr(
                        edge_switch_var <= gp.quicksum(is_end_complete_vars),
                        name=f"constr_edge_switch_end_max_{from_group}_OR"
                    )
                    num_constraints += 1

        # Codice vecchio con coppie di nodi, 
        # -------------------------------------------------------------------------------------------------
                # # EXTERNAL link from nic_out_X to nic_in_Y
                # # Link is ON iff there is at least one couple of nodes (i in From_group, j in To_group) such that
                # # end_i == start_j
                # elif (self.ilp_nodes[link_name_tuple[0]].node_type == NodeType.NIC_OUT and
                #       self.ilp_nodes[link_name_tuple[1]].node_type == NodeType.NIC_IN):
                #     from_group: int = self.ilp_nodes[link_name_tuple[0]].group
                #     to_group: int = self.ilp_nodes[link_name_tuple[1]].group

                #     # get edge switch variable
                #     edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]

                #     # All-pairs Logic
                #     valid_pairs_vars: List[gp.Var] = []
                #     _m = self.model_spec.n_layers

                #     for gpu_i in self.gpu_groups[from_group]:
                #         # compute node i's end position
                #         end_expr_i: gp.LinExpr = self.get_end_layer_index(compute_node_idx=gpu_i)

                #         for gpu_j in self.gpu_groups[to_group]:

                #             # 1. Get layer expression for this couple
                #             start_var_j: gp.Var = self.var_node_start[f"start_{gpu_j}"]

                #             # 2. create tmp var pair_ok_ij
                #             # pair_ok_ij == 1 iff end_i == start_j
                #             pair_ok_var_name = f"pair_ok_{gpu_i}_{gpu_j}_for_{link_name_tuple[0]}_{link_name_tuple[1]}"
                #             pair_ok_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=pair_ok_var_name)
                #             valid_pairs_vars.append(pair_ok_var)

                #             # 3. Add "Big M" constraints
                #             # IF pair_ok_ij == 1 THEN end_i - start_j == 0
                #             # (pair_ok_ij == 0 IF end_u != start_v)

                #             # (1) m * pair_ok_ij <= m + (start_j - end_i)
                #             self.ilp_model.addConstr(
                #                 _m * pair_ok_var <= _m + (start_var_j - end_expr_i),
                #                 name=f"constr_pair_ok1_{gpu_i}_{gpu_j}"
                #             )
                #             num_constraints += 1

                #             # (2) m * pair_ok_ij <= m - (start_j - end_i)
                #             self.ilp_model.addConstr(
                #                 _m * pair_ok_var <= _m - (start_var_j - end_expr_i),
                #                 name=f"constr_pair_ok2_{gpu_i}_{gpu_j}"
                #             )
                #             num_constraints += 1
                        
                #     # 4. Final constraint on switch
                #     # switch <= sum(pair_ok_ij for all i,j)
                #     # if switch == 1 => at least one pair_ok_ij == 1 => at least one end_i == start_j
                #     # if no valid pairs, switch must be 0
                #     self.ilp_model.addConstr(
                #         edge_switch_var <= gp.quicksum(valid_pairs_vars),
                #         name=f"constr_edge_switch_end_start_{from_group}_{to_group}_OR"
                #     )
                #     num_constraints += 1
            # -------------------------------------------------------------------------------------------------

            # codice nuovo con max end e min start, no coppie di nodi
            # -------------------------------------------------------------------------------------------------
                elif (self.ilp_nodes[link_name_tuple[0]].node_type == NodeType.NIC_OUT and 
                      self.ilp_nodes[link_name_tuple[1]].node_type == NodeType.NIC_IN):
                    from_group: int = self.ilp_nodes[link_name_tuple[0]].group
                    to_group: int = self.ilp_nodes[link_name_tuple[1]].group

                    # get edge switch variable
                    edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                    _m = self.model_spec.n_layers


                    # 1. Compute group_start for to_group
                    # group_start = min(start_j for j in to_group)
                    if f"group_start_{to_group}" in self.var_group_starts:
                        group_start_var = self.var_group_starts[f"group_start_{to_group}"]
                    else:
                        to_group_starts = []
                        for gpu_j in self.gpu_groups[to_group]:
                            to_group_starts.append(self.var_node_start[f"start_{gpu_j}"])

                        group_start_var_name = f"group_start_{to_group}"
                        group_start_var = self.ilp_model.addVar(vtype=GRB.INTEGER, name=group_start_var_name)
                        self.var_group_starts[group_start_var_name] = group_start_var

                        self.ilp_model.addGenConstrMin(group_start_var, to_group_starts, name=f"min_start_{to_group}")
                        num_constraints += 1


                    # 2. Compute group_end for from_group
                    # group_end = max(end_i for i in from_group)
                    if f"group_end_{from_group}" in self.var_group_ends:
                        group_end_var = self.var_group_ends[f"group_end_{from_group}"]
                    else:
                        from_group_ends = []
                        for gpu_i in self.gpu_groups[from_group]:
                            is_active_var = self.var_node_active[gpu_i]
                            end_expr_i = self.get_end_layer_index(compute_node_idx=gpu_i)
                            val_end_var = self.ilp_model.addVar(vtype=GRB.INTEGER, name=f"val_end_{gpu_i}")

                            # Indicator constraints to check if gpu is active and set val_end_var accordingly
                            self.ilp_model.addGenConstrIndicator(
                                is_active_var, True,
                                val_end_var == end_expr_i,
                                name=f"active_end_val_{gpu_i}"
                            )
                            self.ilp_model.addGenConstrIndicator(
                                is_active_var, False,
                                val_end_var == 0,
                                name=f"inactive_end_zero_{gpu_i}"
                            )
                            num_constraints += 2

                            from_group_ends.append(val_end_var)
                    
                        group_end_var_name = f"group_end_{from_group}"
                        group_end_var = self.ilp_model.addVar(vtype=GRB.INTEGER, name=group_end_var_name)
                        self.var_group_ends[group_end_var_name] = group_end_var
                        self.ilp_model.addGenConstrMax(group_end_var, from_group_ends, name=f"max_end_{from_group}")
                        num_constraints += 1


                    # 3. Add constraints for edge switch
                    # switch == 1 => group_end_from == group_start_to

                    # (1) m * switch <= m + (start_to - end_from)
                    self.ilp_model.addConstr(
                         _m * edge_switch_var <= _m + (group_start_var - group_end_var),
                         name=f"constr_link_match_1_{from_group}_{to_group}"
                    )
                    
                    # (2) m * switch <= m - (start_to - end_from)
                    self.ilp_model.addConstr(
                         _m * edge_switch_var <= _m - (group_start_var - group_end_var),
                         name=f"constr_link_match_2_{from_group}_{to_group}"
                    )
                    num_constraints += 2





            # elif (self.ilp_nodes[link_name_tuple[0]].node_type == NodeType.NIC_IN and self.ilp_nodes[link_name_tuple[1]].node_type == NodeType.COMPUTE) or \
            #      (self.ilp_nodes[link_name_tuple[0]].node_type == NodeType.COMPUTE and self.ilp_nodes[link_name_tuple[1]].node_type == NodeType.NIC_OUT):
            #     # INTERNAL DISTRIBUTION link from nic_in_X to compute node or from compute node to nic_out_X
            #     link_type = "INTERNAL_DISTRIBUTION"

            #     edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]

            #     # Force switch to 1
            #     # This link is always ON because it's internal to the machine. No need to add complex conditions.
            #     distrib_force_constr_name = f"force_constr_distrib_{link_name_tuple[0]}_{link_name_tuple[1]}"
            #     distrib_force_constr = self.ilp_model.addConstr(
            #         edge_switch_var == 1,
            #         name=distrib_force_constr_name
            #     )
            #     num_constraints += 1


            else: # INTERNAL COMPUTE link from compute node to compute node
                link_type = "INTERNAL_COMPUTE"

                if not (self.ilp_nodes[link_name_tuple[0]].node_type == NodeType.COMPUTE and
                        self.ilp_nodes[link_name_tuple[1]].node_type == NodeType.COMPUTE):
                    continue
                
                edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                group_i: int = self.ilp_nodes[link_name_tuple[0]].group
                group_j: int = self.ilp_nodes[link_name_tuple[1]].group

                if group_i == group_j:
                    # Case 3.0: link between compute node i and j in the SAME group
                    # 1. TP Permission
                    tp_permission_var: gp.Var = self.var_group_tp_active[f"group_tp_active_{group_i}"]

                    # 2. PP Permission
                    # True if end_i == start_j
                    pp_permission_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=f"pp_permission_{link_name_tuple[0]}_{link_name_tuple[1]}")

                    # Get start/end
                    gpu_i = link_name_tuple[0]
                    gpu_j = link_name_tuple[1]
                    start_j = self.var_node_start[f"start_{gpu_j}"]
                    end_i = self.get_end_layer_index(gpu_i)
                    m = self.model_spec.n_layers

                    # Big-M constraints for pp_permission_var
                    # (1) m * pp_permission_var <= m + (start_j - end_i)
                    self.ilp_model.addConstr(
                        m * pp_permission_var <= m + (start_j - end_i),
                        name=f"constr_pp_permissionA_{gpu_i}_{gpu_j}"
                    )
                    num_constraints += 1

                    # (2) m * pp_permission_var <= m - (start_j - end_i)
                    self.ilp_model.addConstr(
                        m * pp_permission_var <= m - (start_j - end_i),
                        name=f"constr_pp_permissionB_{gpu_i}_{gpu_j}"
                    )
                    num_constraints += 1

                    # 3. Final constraint on switch
                    # switch <= tp_permission_var + pp_permission_var
                    internal_link_constr_name = f"constr_internal_compute_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    internal_link_constr = self.ilp_model.addConstr(
                        edge_switch_var <= tp_permission_var + pp_permission_var,
                        name=internal_link_constr_name
                    )
                    num_constraints += 1
                else:
                    # Case 3.1 and 3.2: link between compute node i and j in DIFFERENT groups (impossible)
                    self.ilp_model.addConstr(edge_switch_var == 0,
                                             name=f"force_off_intergroup_compute_{link_name_tuple[0]}_{link_name_tuple[1]}")
                    num_constraints += 1

        if self.enable_memory:
            for link_name_tuple in link_name_list:
                u = link_name_tuple[0]
                v = link_name_tuple[1]
                edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{u}_{v}"]
                # 1. Check if "u" is a compute node (has is_active)
                # Constraint: switch_u_v <= is_active_u
                if u in self.var_node_active:
                    self.ilp_model.addConstr(
                        edge_switch_var <= self.var_node_active[u],
                        name=f"link_activation_src_{u}_{v}"
                    )

                # 2. Check if "v" is a compute node (has is_active)
                # Constraint: switch_u_v <= is_active_v
                if v in self.var_node_active:
                    self.ilp_model.addConstr(
                        edge_switch_var <= self.var_node_active[v],
                        name=f"link_activation_dst_{u}_{v}"
                    )

        return num_constraints

    def step7_edge_flow_constraint(self) -> int:
        """
        Add constraint for flow over each edge.

        :return: number of constraints added
        """
        num_constraints = 0

        # constraint_name: edge_flow_constr_i_j
        # constraint: flow_i_j <= link_throughput * switch_i_j
        # number of constraints: e
        for link_name_tuple, link in self.ilp_links.items():
            link_throughput = link.throughput

            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                forward_flow_var: gp.Var = self.var_flow[f"flow_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                forward_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                forward_edge_flow_constraint_name = f"edge_flow_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                forward_edge_flow_constraint = self.ilp_model.addConstr(
                    forward_flow_var <= link_throughput * forward_switch_var,
                    name=forward_edge_flow_constraint_name
                )
                self.constr_edge_flow[forward_edge_flow_constraint_name] = forward_edge_flow_constraint
                num_constraints += 1

        return num_constraints

    def step8_group_constraints(self) -> int:
        """
        Add group related constraints.

        :return: number of constraints added
        """
        num_constraints = 0

        # NEW Step 8.0 force group control var to 0 if only 1 gpu in the group
        for group_id, gpu_ids in self.gpu_groups.items():
            if len(gpu_ids) == 1:
                control_var: gp.Var = self.var_group_tp_active[f"group_tp_active_{group_id}"]
                force_off_constr_name = f"force_group_off_constr_{group_id}"
                force_off_constr = self.ilp_model.addConstr(
                    control_var == 0,
                    name=force_off_constr_name
                )
                self.constr_force_group_off[force_off_constr_name] = force_off_constr
                num_constraints += 1
        
        # NEW Step 8.2 add group synchronization constraint
        # 8.2.1 syncronize START layer
        # 8.2.2 syncronize NUMBER of layers
        # 8.2.3 synchronize LIMIT of layers

        for group_id, gpu_ids in self.gpu_groups.items():
            # take the switch variable of the group
            control_var: gp.Var = self.var_group_tp_active[f"group_tp_active_{group_id}"]

            if len(gpu_ids) > 1:
                # reference gpu
                ref_gpu: str = gpu_ids[0]

                # Step 8.2.1: synch START layer
                # constraint_name: group_synch_start_{group_id}_{other_gpu} (other_gpu is every gpu in the group except of the first one)
                # constraint: IF group control_var == 1 THEN s_ref = s_other for every gpu in the group
                for other_gpu in gpu_ids[1:]:

                    group_synch_start_constr_name = f"group_synch_start_constr_{group_id}_{other_gpu}"
                    group_synch_start_constr = self.ilp_model.addGenConstrIndicator(
                        control_var, True,
                        self.var_node_start[f"start_{ref_gpu}"] == self.var_node_start[f"start_{other_gpu}"],
                        name=group_synch_start_constr_name
                    )
                    self.constr_group_synch_start[group_synch_start_constr_name] = group_synch_start_constr
                    num_constraints += 1

                
                # Step 8.2.2: synch NUMBER of layers
                # If TP is ON, synchronize the number of layers assigned to each GPU in the group
                # constraint_name: group_synch_hold_constr_{group_id}_{other_gpu}
                # constraint: IF group control_var == 1 THEN hold_ref_k = hold_other_k for every gpu in the group
                for k in range(1, self.global_max_num_layers + 1):
                    if k in self.var_node_hold_layer[ref_gpu]:
                        var_ref_k: gp.Var = self.var_node_hold_layer[ref_gpu][k]

                        for other_gpu in gpu_ids[1:]:
                            if k in self.var_node_hold_layer[other_gpu]:
                                var_other_k: gp.Var = self.var_node_hold_layer[other_gpu][k]

                                group_synch_hold_constr_name = f"group_synch_hold_constr_{group_id}_{other_gpu}_{k}"
                                group_synch_hold_constr = self.ilp_model.addGenConstrIndicator(
                                    control_var, True,
                                    var_ref_k == var_other_k,
                                    name=group_synch_hold_constr_name
                                )
                                self.constr_group_synch_hold[group_synch_hold_constr_name] = group_synch_hold_constr
                                num_constraints += 1


        # NEW Step 8.2.3: synch LIMIT of layers
        # Constraint A and B

                # Constraint A: If TP is ON, limit = group limit
                # if TP active, GPU cannot activate layers beyond group limit. Apply this only if group has more than 1 gpu
                # constraint_name: tp_on_limit_constr_{gpu_id}_{k}
                # constraint: IF group control_var == 1 THEN hold_gpu_k = 0 for k > group limit

                # get TP limit of the group 
                # HACK for semplification use ref_gpu.max_num_layers. Better use min of the group gpus? (if heterogeneous)
                group_tp_limit_layers: int = self.ilp_nodes[ref_gpu].max_num_layers * len(gpu_ids)
                
                for gpu_id in gpu_ids:
                    for k, var in self.var_node_hold_layer[gpu_id].items():
                        if k > group_tp_limit_layers:
                            tp_on_limit_constr_name = f"tp_on_limit_constr_{gpu_id}_{k}"
                            tp_on_limit_constr = self.ilp_model.addGenConstrIndicator(
                                control_var, True,
                                var == 0,
                                name=tp_on_limit_constr_name
                            )
                            self.constr_tp_on_limit[tp_on_limit_constr_name] = tp_on_limit_constr
                            num_constraints += 1

                


            # Constraint B: If TP is OFF, limit = individual node limit
            # set to 0 all variables that exceed individual node limit. Apply always, even if group has only 1 gpu
            # if TP non active, GPU cannot activate layers beyond its individual limit
            # constraint_name: tp_off_limit_constr_{gpu_id}_{k}
            # constraint: IF group control_var == 0 THEN hold_gpu_k = 0 for k > individual limit
            for gpu_id in gpu_ids:
                individual_limit: int = self.ilp_nodes[gpu_id].max_num_layers

                for k, var in self.var_node_hold_layer[gpu_id].items():                    
                    if (self.tp_only and len(gpu_ids)>1) or k > individual_limit:
                        tp_off_limit_constr_name = f"tp_off_limit_constr_{gpu_id}_{k}"
                        tp_off_limit_constr = self.ilp_model.addGenConstrIndicator(
                            control_var, False,
                            var == 0,
                            name=tp_off_limit_constr_name
                        )
                        self.constr_tp_off_limit[tp_off_limit_constr_name] = tp_off_limit_constr
                        num_constraints += 1


        # NEW Step 8.3 add group throughput constraint IN FLOW
        # TODO unire con quello sopra?
        for group_id, gpu_ids in self.gpu_groups.items():

            # take the switch variable of the group
            control_var: gp.Var = self.var_group_tp_active[f"group_tp_active_{group_id}"]

            # 1. Get NIC_IN node of the group
            nic_in_node_idx: str = f"nic_in_{group_id}"

            # 2. Sum flow of all links (NIC_IN -> GPU_i) for all GPUs in the group
            total_group_flow_in: gp.LinExpr = gp.quicksum(
                self.var_flow[f"flow_{nic_in_node_idx}_{gpu_id}"]
                for gpu_id in gpu_ids
                if f"flow_{nic_in_node_idx}_{gpu_id}" in self.var_flow
            )

            # 3. Compute GROUP THROUGHPUT
            group_throughput_list = []
            ref_gpu = gpu_ids[0]

            for layer_count, throughput_at_k in self.tp_throughput_profiles[group_id].items():
                if layer_count in self.var_node_hold_layer[ref_gpu]:
                    hold_k: gp.Var = self.var_node_hold_layer[ref_gpu][layer_count]
                    group_throughput_list.append(throughput_at_k * hold_k)

            throughput_group = gp.quicksum(group_throughput_list)

            # 4. Add constraint: IF control_var == 1 THEN total_group_flow_in <= throughput_group
            group_tp_constr_name = f"group_throughput_constr_{group_id}"
            group_tp_constr = self.ilp_model.addGenConstrIndicator(
                control_var, True,
                total_group_flow_in <= throughput_group,
                name=group_tp_constr_name
            )
            self.constr_group_throughput[group_tp_constr_name] = group_tp_constr
            num_constraints += 1

        
        # Step 8.6 Group Throughput with Latency
        # NEW: Inter-group latency-adjusted group throughput (TP ON)
        # Same logic as step5 latency, but for the group throughput constraint
        # IF TP ON AND inter-group link active THEN group flow <= T_eff_group(k, L_net)
        for link_key, link in self.ilp_links.items():
            from_idx, to_idx = link_key
            from_node = self.ilp_nodes.get(from_idx)
            if from_idx == "source":
                from_node = self.ilp_source
            to_node = self.ilp_nodes.get(to_idx)
            if not (from_node and to_node):
                continue
            if (from_node.node_type != NodeType.NIC_OUT and from_node.node_type != NodeType.SOURCE) \
                or to_node.node_type != NodeType.NIC_IN:
                continue
            L_net = link.latency
            if L_net <= 0:
                continue

            switch_var = self.var_edge_switch.get(f"switch_{from_idx}_{to_idx}")
            if switch_var is None:
                continue

            group_id = to_node.group
            gpu_ids = self.gpu_groups[group_id]
            if len(gpu_ids) < 2:
                continue  # single-GPU groups can't have TP ON

            control_var = self.var_group_tp_active[f"group_tp_active_{group_id}"]
            nic_in_node_idx = f"nic_in_{group_id}"
            ref_gpu = gpu_ids[0]

            # 1. Total group flow in (same as original)
            total_group_flow_in = gp.quicksum(
                self.var_flow[f"flow_{nic_in_node_idx}_{gpu_id}"]
                for gpu_id in gpu_ids
                if f"flow_{nic_in_node_idx}_{gpu_id}" in self.var_flow
            )

            # 2. Latency-adjusted group throughput
            adjusted_group_tp = []
            for layer_count, throughput_at_k in self.tp_throughput_profiles[group_id].items():
                if layer_count in self.var_node_hold_layer[ref_gpu]:
                    hold_k = self.var_node_hold_layer[ref_gpu][layer_count]
                    T_eff = self.batch_size / (self.batch_size / throughput_at_k + L_net)
                    adjusted_group_tp.append(T_eff * hold_k)

            # 3. AND variable: TP ON AND inter-group link active
            and_var_name = f"tp_on_and_switch_{from_idx}_{to_idx}_{group_id}"
            and_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=and_var_name)
            self.ilp_model.addConstr(and_var <= control_var)
            self.ilp_model.addConstr(and_var <= switch_var)
            self.ilp_model.addConstr(and_var >= control_var + switch_var - 1)

            # 4. IF (TP ON AND switch ON) THEN group flow <= adjusted throughput
            self.ilp_model.addGenConstrIndicator(
                and_var, True,
                total_group_flow_in <= gp.quicksum(adjusted_group_tp),
                name=f"intergroup_group_tp_{group_id}_via_{from_idx}_{to_idx}"
            )
            num_constraints += 1




        if not self.tp_only:
            
            _BigM = 2*self.model_spec.n_layers
            # Step 8.5 pipeline Ordering
            for group_id, gpu_ids in self.gpu_groups.items():
                if len(gpu_ids) < 2:
                    continue

                # take the switch variable of the group
                control_var: gp.Var = self.var_group_tp_active[f"group_tp_active_{group_id}"]

                # enforce ordering
                # for each pair of consecutive gpus in the group
                sorted_gpu_ids = sorted(gpu_ids)  # assuming gpu_ids can be sorted to reflect order

                nic_in = f"nic_in_{group_id}"
                nic_out = f"nic_out_{group_id}"

                # A: TP OFF -> only first GPU gets from NIC_IN
                # If TP OFF (control_var = 0), switch_{nic_in}_{gpu_i} = 0 for every gpu except the FIRST ONE.
                for gpu_id in sorted_gpu_ids[1:]:
                    switch_name = f"switch_{nic_in}_{gpu_id}"
                    if switch_name in self.var_edge_switch:
                        self.ilp_model.addConstr(
                            self.var_edge_switch[switch_name] <= control_var,
                            name=f"pipeline_ordering_start_{group_id}_{gpu_id}"
                        )
                        num_constraints += 1
                
                # # B: TP OFF -> Only last GPU sends to NIC_OUT
                # # If TP OFF (control_var = 0), switch_{gpu_i}_{nic_out} = 0 for every gpu except the LAST ONE.
                # for gpu_id in sorted_gpu_ids[:-1]:
                #     switch_name = f"switch_{gpu_id}_{nic_out}"
                #     if switch_name in self.var_edge_switch:
                #         self.ilp_model.addConstr(
                #             self.var_edge_switch[switch_name] <= control_var,
                #             name=f"pipeline_ordering_end_{group_id}_{gpu_id}"
                #         )
                #         num_constraints += 1


                # B: TP OFF -> enforce ordering between GPUs
                # If TP OFF (control_var = 0), enforce end_i == start_{i+1} OR start_{i+1} == M for every consecutive pair of gpus in the group
                for i in range(len(sorted_gpu_ids) - 1):
                    curr_gpu = sorted_gpu_ids[i]
                    next_gpu = sorted_gpu_ids[i + 1]

                    if self.enable_memory:
                        active_next = self.var_node_active[next_gpu]
                    else:
                        active_next = 1  # always active
                    end_curr = self.get_end_layer_index(curr_gpu)
                    start_next = self.var_node_start[f"start_{next_gpu}"]

                    # B.1: If TP = OFF AND next_gpu is ACTIVE -> end_curr == start_next
                    # Big-M Formula: |start_next - end_curr| <= M * control_var + M * (1 - active_next)
                    self.ilp_model.addConstr(
                        start_next - end_curr <= _BigM * control_var + _BigM * (1 - active_next),
                        name=f"pipeline_ordering_constr1_{curr_gpu}_{next_gpu}"
                    )
                    self.ilp_model.addConstr(
                        end_curr - start_next <= _BigM * control_var + _BigM * (1 - active_next),
                        name=f"pipeline_ordering_constr2_{curr_gpu}_{next_gpu}"
                    )
                    num_constraints += 2

                    # B.2: If TP = OFF AND next_gpu is INACTIVE -> start_next == M (so that it won't receive any layers)
                    # Big-M Formula: start_next >= M - M * active_next - M * control_var
                    self.ilp_model.addConstr(
                        start_next >= self.model_spec.n_layers - _BigM * active_next - _BigM * control_var,
                        name=f"pipeline_ordering_constr3_{curr_gpu}_{next_gpu}"
                    )
                    num_constraints += 1

                    # C: TP OFF -> only last GPU in the pipeline can send to NIC_OUT
                    # Formula: switch_{gpu_i}_{nic_out} <= 1 - active_next + control_var
                    switch_out_name = f"switch_{curr_gpu}_{nic_out}"
                    if switch_name in self.var_edge_switch:
                        switch_out_var = self.var_edge_switch[switch_out_name]

                        self.ilp_model.addConstr(
                            switch_out_var <= 1 - active_next + control_var,
                            name=f"pipeline_ordering_constr4_{curr_gpu}_{next_gpu}"
                        )
                        num_constraints += 1   

            # # C: TP OFF -> enforce ordering between GPUs
            # # If TP OFF (control_var = 0), enforce end_i == start_{i+1} for every consecutive pair of gpus in the group
            # for i in range(len(sorted_gpu_ids) - 1):
            #     curr_gpu = sorted_gpu_ids[i]
            #     next_gpu = sorted_gpu_ids[i + 1]

            #     end_curr = self.get_end_layer_index(curr_gpu)
            #     start_next = self.var_node_start[f"start_{next_gpu}"]

            #     ordering_constr_name = f"pipeline_ordering_constr_{curr_gpu}_{next_gpu}"
            #     ordering_constr = self.ilp_model.addGenConstrIndicator(
            #         control_var, False,
            #         start_next == end_curr,
            #         name=ordering_constr_name
            #     )
            #     num_constraints += 1

        # # DEBUG
        # # Force all TP to OFF
        # for compute_node_idx, compute_node in self.ilp_nodes.items():
        #     if not compute_node.node_type == NodeType.COMPUTE:
        #         continue

        #     tp_active_var = self.var_group_tp_active[f"group_tp_active_{compute_node.group}"]
        #     constr_name = f"force_off_tp_{compute_node_idx}"
        #     constr = self.ilp_model.addConstr(
        #         tp_active_var == 0,
        #         name=constr_name
        #     )



        return num_constraints      

    def build_model(self, seed: int, model_name: str) -> Tuple[int, int, int, int]:
        """
        Build the ILP model.
        Note: 1. Here we build the ILP model exactly as is based on the cluster we just loaded. Optimizations
                 like prune edge / layer fusion / limit on the lower bound of layers on node can be done by
                 changing the cluster description file.

        :param seed: random seed
        :param model_name: name of the ILP model
        :param enable_partial_inference: whether partial inference is enabled or not
        :param remove_redundant: remove redundant constraints in the model
        :param start_from_heuristic: whether to start from a heuristic solution
        :param heuristic_sol_path: path to the heuristic solution
        :return: (int variables, real variables, binary variables, num_constraints)
        """
        # prepare the ILP program
        assert self.cluster_loaded, "Must load a cluster before building the ilp model!"
        num_int, num_real, num_binary, num_constraint = 0, 0, 0, 0

        # Step 1: initial the ILP program
        self.step1_initialize_ilp(seed=seed, model_name=model_name)

        # Step 2: add variables
        cur_num_int, cur_num_real, cur_num_binary = self.step2_add_variables()
        num_int += cur_num_int
        num_real += cur_num_real
        num_binary += cur_num_binary

        # Step 3: add constraint for model placement
        cur_num_constraint = self.step3_model_placement_constraint()
        num_constraint += cur_num_constraint

        # Step 4: add constraint for flow in = flow out
        cur_num_constraint = self.step4_flow_in_out_constraint()
        num_constraint += cur_num_constraint

        # Step 5: add constraint for node throughput
        cur_num_constraint = self.step5_node_throughput_constraint()
        num_constraint += cur_num_constraint

        # Step 6: add constraint for edge switch
        cur_num_constraint = self.step6_edge_switch_constraint()
        num_constraint += cur_num_constraint

        # Step 7: add constraint for flow over edge
        cur_num_constraint = self.step7_edge_flow_constraint()
        num_constraint += cur_num_constraint

        # NEW Step 8: add constraint for groups
        cur_num_constraint = self.step8_group_constraints()
        num_constraint += cur_num_constraint

        self.max_throughput = self.get_flow_upper_bound()
        self.total_vram = sum(node.machine_type.memory_gb * GB for node in self.ilp_nodes.values()
                              if node.node_type == NodeType.COMPUTE)

        # Step 8: set optimization target
        source_flow_out_list: List[gp.Var] = []
        for other_idx in self.ilp_source.connected_node_indices:
            flow_var: gp.Var = self.var_flow[f"flow_source_{other_idx}"]
            source_flow_out_list.append(flow_var)
        if not self.enable_memory:
            self.ilp_model.setObjective(gp.quicksum(source_flow_out_list), GRB.MAXIMIZE)
        else:
            # MEM
            source_flow_out = gp.quicksum(source_flow_out_list)
            self.ilp_model.addConstr(source_flow_out >= 1.0, name="force_min_throughput")
            norm_throughput = source_flow_out / self.max_throughput

            # total active vram
            # total_active_vram = gp.quicksum([self.var_node_active[node_idx] * node.machine_type.memory_gb * GB
            #                                 for node_idx, node in self.ilp_nodes.items() if node.node_type == NodeType.COMPUTE])
            # norm_memory_cost = total_active_vram / self.total_vram

            rental_cost = gp.quicksum([self.var_node_active[node_idx] * node.machine_type.rent_cost
                                            for node_idx, node in self.ilp_nodes.items() if node.node_type == NodeType.COMPUTE])
            norm_rental_cost = rental_cost / sum(node.machine_type.rent_cost for node in self.ilp_nodes.values() if node.node_type == NodeType.COMPUTE)  # normalize rental cost by max

            # Parameter to control trade-off between throughput and rental cost
            # Alpha = 1.0 -> only maximize throughput
            # Alpha = 0.0 -> only minimize rental cost
            # Intermediate values -> trade-off

            final_objective = self.opt_alpha * norm_throughput - (1 - self.opt_alpha) * norm_rental_cost
            self.ilp_model.setObjective(final_objective, GRB.MAXIMIZE)

        # return the size of the ILP problem
        self.model_initialized = True
        return num_int, num_real, num_binary, num_constraint

    def search_layout(self, max_run_time: float, early_stop_threshold: float, early_stop_time: float,
                      save_sol_path: str, save_model_path: str | None = None) -> None:
        """
        Search a layout that maximizes the max flow based on the ILP program.

        :param max_run_time: max running time allowed
        :param early_stop_threshold: a value between 0 and 1 (usually 0.98), if the solution is at least this close
                                     to upper bound, then we may early stop the optimization (see below)
        :param early_stop_time: if the solution reaches early_stop_threshold and no improvement is made in this
                                amount of time, we will early stop the optimization
        :param save_sol_path: save solution into this path
        :param save_model_path: save model into this path
        :return: None
        """
        # check input
        assert self.model_initialized, "Model should be initialized before searching for a layout!"
        assert save_sol_path.endswith(".sol"), "Solution file must end with .sol!"
        assert save_model_path is None or save_model_path.endswith(".lp"), "Model file should end with .lp!"

        # initialize optimization settings
        assert 0 <= early_stop_threshold < 1, "Early stop threshold must be in [0, 1)!"
        self.max_run_time = max_run_time
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_time = early_stop_time
        self.opt_start_time = time.time()
        self.opt_best_obj = -1
        self.opt_best_obj_found_time = self.opt_start_time
        if not self.enable_memory:
            self.opt_upper_bound = self.get_flow_upper_bound()
        else:
            # compute upper bound
            min_rental_cost = min(node.machine_type.rent_cost 
                                for node in self.ilp_nodes.values() 
                                if node.node_type == NodeType.COMPUTE)
            max_rental_cost = sum(node.machine_type.rent_cost 
                                for node in self.ilp_nodes.values() 
                                if node.node_type == NodeType.COMPUTE)

            # min_model_memory = self.model_spec.n_params_billion * BYTE_PER_PARAM * GB * 2  # assume best memory cost is 2x model size
            self.opt_upper_bound = self.opt_alpha * 1.0 - (1 - self.opt_alpha) * (min_rental_cost / max_rental_cost)  # best possible rental cost


        print(f"UPPER BOUND: {self.opt_upper_bound}")

        # define early stopping callback function
        def early_stopping_callback(model, where):
            if where == GRB.Callback.MIP:
                best_objective = model.cbGet(GRB.Callback.MIP_OBJBST)

                # update best objective found
                current_time = time.time()
                if best_objective > self.opt_best_obj:
                    self.opt_best_obj = best_objective
                    self.opt_best_obj_found_time = current_time

                # criteria 1: if max time is reached, then we terminate the search
                if current_time - self.opt_start_time >= self.max_run_time:
                    print(f"[ILP Layout - Info] Early stop because max search time ({self.max_run_time}) is reached")
                    print(f"[ILP Layout - Info] Found: {best_objective}, Upper Bound: {self.opt_upper_bound}.")
                    model.terminate()
                    return

                # criteria 2: if early stop criteria is satisfied
                if best_objective >= self.early_stop_threshold * self.opt_upper_bound:
                    # if no improvement for a long time
                    if current_time - self.opt_best_obj_found_time > self.early_stop_time:
                        print(f"[ILP Layout - Info] Early stop because the best solution found is at least "
                              f"{round(self.early_stop_threshold * 100, 1)}% optimal and no improvement is "
                              f"made in {self.early_stop_time} seconds!")
                        print(f"[ILP Layout - Info] Found: {best_objective}, Upper Bound: {self.opt_upper_bound}.")
                        model.terminate()

                # criteria 3: force early stop if really very optimal
                if best_objective >= 0.995 * self.opt_upper_bound:
                    print(f"[ILP Layout - Info] Early stop because the best solution found is at least 99.5% optimal!")
                    print(f"[ILP Layout - Info] Found: {best_objective}, Upper Bound: {self.opt_upper_bound}.")
                    model.terminate()

        import shutil
        import os
        ini_file = self.cluster_file_name.split("/")[-1]
        ini_path = save_sol_path.replace("ilp_solution.sol", ini_file)
        if not os.path.exists(os.path.dirname(ini_path)):
            shutil.copy(self.cluster_file_name, ini_path)
        print(f"[ILP Layout - Info] Copied cluster file to {ini_path}")
        # solve
        print("# ----------------------------------------- Gurobi ----------------------------------------- #")
        self.ilp_model.optimize(early_stopping_callback)
        print("# ------------------------------------------------------------------------------------------ #")

        best_solution_time = self.opt_best_obj_found_time - self.opt_start_time
        total_time = time.time() - self.opt_start_time

        print(f"[ILP Layout - Result] Best solution found in {best_solution_time:.2f} seconds. "
              f"Total optimization time: {total_time:.2f} seconds.")
        
        # save solution and model
        self.ilp_model.write(save_sol_path)
        if save_model_path is not None:
            self.ilp_model.write(save_model_path)

        self.clean_solution(save_sol_path, best_solution_time, total_time)

        # output solution
        output_path = "./layouts/ilp"
        output_solution(self.cluster_file_name, save_sol_path, output_path)

        # plot memory usage
        self.visualize_memory_usage(save_sol_path)

    def visualize_memory_usage(self, save_sol_path: str) -> None:
        assert self.cluster_loaded, "Cluster must be loaded before we can visualize memory usage!"
        
        name_2_val: Dict[str, int | float] = {}
        with open(save_sol_path, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                name, val = line.split(" ")
                name_2_val[name] = eval(val)

        # Compute memory usage per layer
        memory_usage_per_layer_bytes: int = self.model_spec.memory_bytes_per_layer
        
        node_labels: List[str] = []
        memory_usages: List[float] = []
        # Compute memory usage per node
        for node_idx, compute_node in self.ilp_nodes.items():
            if compute_node.node_type != NodeType.COMPUTE:
                continue

            compute_node: ILPNode
    
            layer_count: int = 0
            for key in name_2_val.keys():
                if key.startswith(f"hold_{node_idx}_"):
                    layer_count = int(key.split("_")[-1])
                    if name_2_val[key] == 1:
                        compute_node.num_assigned_layers = layer_count
                        break
        
            memory_bytes: int = compute_node.machine_type.memory_gb * GB
            shards: int = 1
            if name_2_val[f"group_tp_active_{compute_node.group}"] == 1:
                shards = len(self.gpu_groups[compute_node.group])

            if compute_node.num_assigned_layers == 0:
                used_memory_bytes = 0.0
            else:
                used_memory_bytes: int = compute_node.num_assigned_layers * memory_usage_per_layer_bytes // shards
                usage: float = used_memory_bytes / memory_bytes * 100.0

            node_labels.append(f"Node {node_idx}, {compute_node.machine_type.name}\nLevels: {compute_node.num_assigned_layers}")
            memory_usages.append(usage)
        
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors 

        norm = mcolors.Normalize(vmin=20, vmax=50)
        colors = cm.RdYlGn(norm(memory_usages))

        plt.figure(figsize=(15, 6))
        plt.bar(node_labels, memory_usages, color=colors)
        plt.ylim(0, 100)
        plt.ylabel("Memory Usage (%)")
        plt.title("GPU Memory Usage per Node")

        for i, usage in enumerate(memory_usages):
            plt.text(i, usage + 1, f"{usage:.2f}%", ha='center')

        output_file = save_sol_path.replace("ilp_solution.sol", "memory_usage.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def get_end_layer_index(self, compute_node_idx: str) -> gp.LinExpr:
        """
        Get the end layer index of compute node i.

        :param compute_node_idx: index of the compute node
        :return: a LinExpr that represent the end layer index of compute node i
        """
        start_var: gp.Var = self.var_node_start[f"start_{compute_node_idx}"]
        k_hold_var: List[gp.LinExpr] = []
        # for layer_count in range(1, self.ilp_nodes[compute_node_idx].max_num_layers + 1):
        for layer_count in range(1, self.global_max_num_layers+1):
            hold_var: gp.Var = self.var_node_hold_layer[compute_node_idx][layer_count]
            k_hold_var.append(layer_count * hold_var)

        return start_var + gp.quicksum(k_hold_var)

    def compute_global_max_num_layers(self) -> int:
        """
        Compute the global max number of layers that can be placed on any node.

        :return: None
        """
        max_layers: int = 0
        # individual node max num layers
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            if compute_node.node_type != NodeType.COMPUTE:
                continue
            if compute_node.max_num_layers > max_layers:
                max_layers = compute_node.max_num_layers
        
        # group max num layers
        for group_id, gpu_ids in self.gpu_groups.items():
            if len(gpu_ids) > 1:
                group_tp_limit_layers: int = self.ilp_nodes[gpu_ids[0]].max_num_layers * len(gpu_ids)
                if group_tp_limit_layers > max_layers:
                    max_layers = group_tp_limit_layers

        return max_layers          

    def get_flow_upper_bound(self) -> float:
        """
        Get the upper bound of max flow over this cluster, which is defined as the max flow when all network
        transmissions are instant.

        :return: flow upper bound
        """
        assert self.cluster_loaded, "Cluster must be loaded before we can compute flow upper bound!"
        total_compute_throughput: float = 0

        for group_id, gpu_ids in self.gpu_groups.items():
            # Find MAX throughput that can be generated by this group, considering both TP ON and TP OFF profiles
            
            # A: TP ON profile (group working as single entity)
            group_max_throughput: float = 0
            if group_id in self.tp_throughput_profiles:
                for k, tp_at_k in self.tp_throughput_profiles[group_id].items():
                    group_max_throughput = max(group_max_throughput, tp_at_k * k)

            # B: TP OFF profile (individual nodes working alone, sum of individual capacities)
            individual_sum_throughput: float = 0
            for gpu_id in gpu_ids:
                compute_node = self.ilp_nodes[gpu_id]
                cur_node_max = -1
                for k, tp_at_k in compute_node.layer_count_2_throughput.items():
                    cur_node_max = max(cur_node_max, tp_at_k * k)
                individual_sum_throughput += cur_node_max
            
            # Group contributes with max of the two profiles
            group_max_throughput = max(group_max_throughput, individual_sum_throughput)
            total_compute_throughput += group_max_throughput
        
        return total_compute_throughput / self.model_spec.n_layers

    def clean_solution(self, save_sol_path, best_solution_time, total_time) -> None:

        with open(save_sol_path, "r") as f:
            lines = f.readlines()

        new_line = f"# Best solution found in {best_solution_time:.2f} seconds. Total optimization time: {total_time:.2f} seconds.\n"
        lines.insert(2, new_line)

        new_lines = []
        for line in lines:
            if line.startswith("# "):
                new_lines.append(line)
                continue

            line = line.strip()
            key, value = line.split(" ", 1)[0], line.split(" ", 1)[1]
            
            if key.startswith("start_"):
                value = round(float(value))
                new_line = f"{key} {value}\n"
            elif key.startswith("hold_"):
                value = round(float(value))
                new_line = f"{key} {value}\n"
            elif key.startswith("group_tp_active_"):
                value = round(float(value))
                new_line = f"{key} {value}\n"
            else:
                new_line = line + "\n"
            
            new_lines.append(new_line)

        with open(save_sol_path, "w") as f:
            f.writelines(new_lines)
    
    # def check_link_validity(self, from_idx: str, to_idx: str, allow_partial_inference: bool) -> bool:
    #     """
    #     Check whether the link between the two nodes is valid.

    #     :param from_idx: index of the input node
    #     :param to_idx: index of the output node
    #     :param allow_partial_inference: whether partial inference is allowed
    #     :return: whether the link is valid
    #     """
    #     assert not from_idx == "sink" and not to_idx == "source", "Invalid end points!"
    #     assert not (from_idx == "source" and to_idx == "sink"), "Found edge between source and sink!"
    #     if from_idx == "source":
    #         return self.ilp_nodes[to_idx].start_layer_idx == 0
    #     elif to_idx == "sink":
    #         return self.ilp_nodes[from_idx].end_layer_idx == self.model_spec.n_layers
    #     else:
    #         s_j = self.ilp_nodes[to_idx].start_layer_idx
    #         e_i = self.ilp_nodes[from_idx].end_layer_idx
    #         e_j = self.ilp_nodes[to_idx].end_layer_idx
    #         if allow_partial_inference:
    #             return s_j <= e_i < e_j
    #         else:
    #             return e_i == s_j

    # def load_and_verify_solution(self, save_sol_path: str, allow_partial_inference: bool) -> None:
    #     """
    #     Load a solution from the file and verify it.

    #     :param save_sol_path: the file that saves the solution
    #     :param allow_partial_inference: whether partial inference is allowed
    #     :return: None
    #     """
    #     assert self.cluster_loaded, "Cluster must be loaded before we can load and verify solution!"

    #     # load the variables into a dict
    #     name_2_val: Dict[str, int | float] = {}
    #     with open(save_sol_path, "r") as file:
    #         for line in file:
    #             if line.startswith("#"):
    #                 continue
    #             name, val = line.split(" ")
    #             name_2_val[name] = eval(val)

    #     # load into ilp nodes
    #     for node_idx, compute_node in self.ilp_nodes.items():
    #         if compute_node.node_type != NodeType.COMPUTE:
    #             continue

    #         compute_node: ILPNode

    #         # start layer index
    #         compute_node.start_layer_idx = round(name_2_val[f"start_{node_idx}"])
    #         assert 0 <= compute_node.start_layer_idx, "Bad start layer index!"
    #         assert is_close(compute_node.start_layer_idx, name_2_val[f"start_{node_idx}"]), \
    #             "Start layer index should be an int!"

    #         # check that only one configuration is selected
    #         hold_sum, k_hold_sum = 0, 0
    #         # for layer_count in range(1, compute_node.max_num_layers + 1):
    #         # NEW layer count to global max num layers to account for group tp
    #         for layer_count in range(1, self.global_max_num_layers + 1):
    #             hold_var_val = round(name_2_val[f"hold_{node_idx}_{layer_count}"])
    #             assert is_close(hold_var_val, name_2_val[f"hold_{node_idx}_{layer_count}"]), \
    #                 "Hold var should be an int!"
    #             assert hold_var_val == 0 or hold_var_val == 1, "Hold var must be binary!"
    #             hold_sum += hold_var_val
    #             k_hold_sum += layer_count * hold_var_val
    #         assert hold_sum == 1, f"Only one configuration can be selected (now {hold_sum})!"

    #         # end layer index
    #         compute_node.end_layer_idx = compute_node.start_layer_idx + k_hold_sum
    #         assert compute_node.end_layer_idx <= self.model_spec.n_layers, "Bad end layer index!"

    #     # load into ilp links
    #     for link_name_tuple, link in self.ilp_links.items():
    #         link: ILPLink

    #         # forward edge
    #         if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
    #             from_idx = link_name_tuple[0]
    #             to_idx = link_name_tuple[1]

    #             # load variables
    #             link.forward_flow = name_2_val[f"flow_{from_idx}_{to_idx}"]
    #             assert link.forward_flow >= 0 - ATOL, "Flow should be larger or equal to 0!"
    #             link.forward_edge_switch = round(name_2_val[f"switch_{from_idx}_{to_idx}"])
    #             assert is_close(link.forward_edge_switch, name_2_val[f"switch_{from_idx}_{to_idx}"]), \
    #                 "Switch variable should be an int!"
    #             assert link.forward_edge_switch == 0 or link.forward_edge_switch == 1, "Switch is binary!"
    #             if not from_idx == "source" and not to_idx == "sink" and allow_partial_inference:
    #                 link.forward_edge_cond1 = round(name_2_val[f"edge_cond1_{from_idx}_{to_idx}"])
    #                 link.forward_edge_cond2 = round(name_2_val[f"edge_cond2_{from_idx}_{to_idx}"])
    #                 assert is_close(link.forward_edge_cond1, name_2_val[f"edge_cond1_{from_idx}_{to_idx}"]), \
    #                     "Condition 1 should be an int!"
    #                 assert is_close(link.forward_edge_cond2, name_2_val[f"edge_cond2_{from_idx}_{to_idx}"]), \
    #                     "Condition 2 should be an int!"
    #                 assert link.forward_edge_cond1 == 0 or link.forward_edge_cond1 == 1, "Condition 1 is binary!"
    #                 assert link.forward_edge_cond2 == 0 or link.forward_edge_cond2 == 1, "Condition 2 is binary!"
    #                 if link.forward_edge_switch:
    #                     assert link.forward_edge_cond1 == 1, "Condition 1 should be 1 for switch to be True!"
    #                     assert link.forward_edge_cond2 == 1, "Condition 2 should be 1 for switch to be True!"

    #             # check that the switch can be enabled
    #             if link.forward_edge_switch == 1:
    #                 assert self.check_link_validity(from_idx=from_idx, to_idx=to_idx,
    #                                                 allow_partial_inference=allow_partial_inference), \
    #                     "Found edge that could not be set to enabled!"

    #             # check flow consistency
    #             assert link.forward_flow <= link.throughput * link.forward_edge_switch + ATOL, \
    #                 f"Bad flow over link from {from_idx} to {to_idx}!"

    #         # backward edge
    #         if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
    #             from_idx = link_name_tuple[1]
    #             to_idx = link_name_tuple[0]

    #             # load variables
    #             link.backward_flow = name_2_val[f"flow_{from_idx}_{to_idx}"]
    #             assert link.backward_flow >= 0 - ATOL, "Flow should be larger or equal to 0!"
    #             link.backward_edge_switch = round(name_2_val[f"switch_{from_idx}_{to_idx}"])
    #             assert is_close(link.backward_edge_switch, name_2_val[f"switch_{from_idx}_{to_idx}"]), \
    #                 "Switch variable should be an int!"
    #             assert link.backward_edge_switch == 0 or link.backward_edge_switch == 1, "Switch is binary!"
    #             if not from_idx == "source" and not to_idx == "sink" and allow_partial_inference:
    #                 link.backward_edge_cond1 = round(name_2_val[f"edge_cond1_{from_idx}_{to_idx}"])
    #                 link.backward_edge_cond2 = round(name_2_val[f"edge_cond2_{from_idx}_{to_idx}"])
    #                 assert is_close(link.backward_edge_cond1, name_2_val[f"edge_cond1_{from_idx}_{to_idx}"]), \
    #                     "Condition 1 should be an int!"
    #                 assert is_close(link.backward_edge_cond2, name_2_val[f"edge_cond2_{from_idx}_{to_idx}"]), \
    #                     "Condition 2 should be an int!"
    #                 assert link.backward_edge_cond1 == 0 or link.backward_edge_cond1 == 1, "Condition 1 is binary!"
    #                 assert link.backward_edge_cond2 == 0 or link.backward_edge_cond2 == 1, "Condition 2 is binary!"
    #                 if link.backward_edge_switch:
    #                     assert link.backward_edge_cond1 == 1, "Condition 1 should be 1 for switch to be True!"
    #                     assert link.backward_edge_cond2 == 1, "Condition 2 should be 1 for switch to be True!"

    #             # check that the switch can be enabled
    #             if link.backward_edge_switch == 1:
    #                 assert self.check_link_validity(from_idx=from_idx, to_idx=to_idx,
    #                                                 allow_partial_inference=allow_partial_inference), \
    #                     "Found edge that could not be set to enabled!"

    #             # check flow consistency
    #             assert link.backward_flow <= link.throughput * link.backward_edge_switch + ATOL, \
    #                 f"Bad flow over link from {from_idx} to {to_idx}!"

    #         # forward and backward edge can not have flow at the same time
    #         if "source" not in link_name_tuple and "sink" not in link_name_tuple:
    #             assert (link.forward_flow == 0) or (link.backward_flow == 0), "Only one direction can have flow!"

    #     # check that flow in = flow out and flow < inference throughput for each node
    #     for compute_node_idx, compute_node in self.ilp_nodes.items():
    #         # flow in = flow out
    #         flow_in, flow_out = 0, 0
    #         for other_idx in compute_node.connected_node_indices:
    #             if not other_idx == "sink":
    #                 flow_in += name_2_val[f"flow_{other_idx}_{compute_node_idx}"]
    #             if not other_idx == "source":
    #                 flow_out += name_2_val[f"flow_{compute_node_idx}_{other_idx}"]
    #         assert math.isclose(flow_in, flow_out, abs_tol=ATOL), f"Flow in = {flow_in} != {flow_out} = Flow out!"

    #         # flow < inference throughput
    #         num_layers_on_node = compute_node.end_layer_idx - compute_node.start_layer_idx
    #         assert flow_in <= compute_node.layer_count_2_throughput[num_layers_on_node] + ATOL, \
    #             "Flow in should be smaller than inference throughput!"

    #     self.solution_loaded = True

    # def generate_simulator_cluster(self, cluster_file_path: str, allow_partial_inference: bool) -> None:
    #     """
    #     Generate the cluster file and statistics file that will be used by the simulator.

    #     :param cluster_file_path: path to save the cluster file
    #     :param allow_partial_inference: whether partial inference is allowed
    #     :return: None
    #     """
    #     assert self.solution_loaded, "Solution must be loaded before generating simulator cluster!"

    #     # generate cluster file
    #     with open(cluster_file_path, "w") as file:
    #         # header notes
    #         file.write("# Simulator cluster file generated by ILP layout synthesizer.\n")
    #         file.write("\n")

    #         # write coordinator
    #         file.write(f"[Coordinator]\n")
    #         inbound_nic_speed: float = self.ilp_sink.machine_type.inbound_nic_speed / mbps
    #         outbound_nic_speed: float = self.ilp_source.machine_type.outbound_nic_speed / mbps
    #         file.write(f"inbound_nic_speed={inbound_nic_speed} * mbps\n")
    #         file.write(f"outbound_nic_speed={outbound_nic_speed} * mbps\n")
    #         file.write("\n")

    #         # write machine types
    #         file.write(f"[MachineTypes]\n")
    #         machine_types = list(self.machine_profiles.keys())
    #         machine_types.remove("SourceNode")
    #         machine_types.remove("SinkNode")
    #         file.write(f"types={machine_types}\n")
    #         file.write("\n")

    #         # write node names
    #         file.write("[ComputeNodes]\n")
    #         node_names = [f"compute_node_{self.node_idx_offset + i}" for i in range(len(self.ilp_nodes))]
    #         file.write(f"names={node_names}\n")
    #         file.write("\n")

    #         # write the nodes
    #         for node_idx, ilp_node in self.ilp_nodes.items():
    #             file.write(f"[compute_node_{self.node_idx_offset + node_idx}]\n")
    #             vram_size: float = ilp_node.machine_type.vram_size / MB
    #             file.write(f"vram_size={vram_size} * MB\n")
    #             inbound_nic_speed: float = ilp_node.machine_type.inbound_nic_speed / mbps
    #             file.write(f"inbound_nic_speed={inbound_nic_speed} * mbps\n")
    #             outbound_nic_speed: float = ilp_node.machine_type.outbound_nic_speed / mbps
    #             file.write(f"outbound_nic_speed={outbound_nic_speed} * mbps\n")
    #             disk_speed: float = ilp_node.machine_type.disk_speed / mbps
    #             file.write(f"disk_speed={disk_speed} * mbps\n")
    #             file.write(f"machine_type=\"{ilp_node.machine_type.type_name}\"\n")
    #             kv_cache_capacity: int = self.model_manager.get_kv_cache_capacity(
    #                 machine_type=ilp_node.machine_type.type_name,
    #                 num_on_node_layers=ilp_node.end_layer_idx - ilp_node.start_layer_idx
    #             )
    #             file.write(f"kv_cache_capacity={kv_cache_capacity}\n")
    #             activation_backup_capacity: int = self.model_manager.get_activation_backup_capacity(
    #                 machine_type=ilp_node.machine_type.type_name,
    #                 num_on_node_layers=ilp_node.end_layer_idx - ilp_node.start_layer_idx
    #             )
    #             file.write(f"activation_backup_capacity={activation_backup_capacity}\n")
    #             file.write("\n")

    #         # write the links
    #         # note that we write all links as long as the link is valid (by checking models at two endpoints)
    #         # find all valid links
    #         valid_links: Dict[Tuple[str, str], ILPLink] = {}
    #         for link_name_tuple, ilp_link in self.ilp_links.items():
    #             if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
    #                 forward_link_valid = self.check_link_validity(from_idx=link_name_tuple[0],
    #                                                               to_idx=link_name_tuple[1],
    #                                                               allow_partial_inference=allow_partial_inference)
    #                 if forward_link_valid:
    #                     valid_links[link_name_tuple] = ilp_link
    #             if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
    #                 backward_link_valid = self.check_link_validity(from_idx=link_name_tuple[1],
    #                                                                to_idx=link_name_tuple[0],
    #                                                                allow_partial_inference=allow_partial_inference)
    #                 if backward_link_valid:
    #                     valid_links[(link_name_tuple[1], link_name_tuple[0])] = ilp_link

    #         # remove links associated with nodes with no outbound link
    #         # if a node other than sink does not have any outbound link, then remove all inbound links
    #         # associated with node, as this node can not be used in inference
    #         nodes_with_outbound: Set[int | str] = set()
    #         for link_name_tuple in valid_links.keys():
    #             nodes_with_outbound.add(link_name_tuple[0])
    #         valid_links_with_outbound: Dict[Tuple[int | str, int | str], ILPLink] = {}
    #         for link_name_tuple, ilp_link in valid_links.items():
    #             if link_name_tuple[1] == "sink" or link_name_tuple[1] in nodes_with_outbound:
    #                 valid_links_with_outbound[link_name_tuple] = ilp_link
    #         valid_links = valid_links_with_outbound

    #         # remove links associated with nodes with no inbound link
    #         # if a node other than source does not have any inbound link, then remove all outbound links
    #         # associated with node, as this node can not be used in inference
    #         nodes_with_inbound: Set[int | str] = set()
    #         for link_name_tuple in valid_links.keys():
    #             nodes_with_inbound.add(link_name_tuple[1])
    #         valid_links_with_inbound: Dict[Tuple[str, str], ILPLink] = {}
    #         for link_name_tuple, ilp_link in valid_links.items():
    #             if link_name_tuple[0] == "source" or link_name_tuple[0] in nodes_with_inbound:
    #                 valid_links_with_inbound[link_name_tuple] = ilp_link
    #         valid_links = valid_links_with_inbound

    #         # write the valid link names
    #         file.write("[Links]\n")
    #         valid_link_names = []
    #         for valid_link_name_tuple in valid_links.keys():
    #             from_name = valid_link_name_tuple[0] if valid_link_name_tuple[0] == "source" else \
    #                 f"compute_node_{self.node_idx_offset + valid_link_name_tuple[0]}"
    #             to_name = valid_link_name_tuple[1] if valid_link_name_tuple[1] == "sink" else \
    #                 f"compute_node_{self.node_idx_offset + valid_link_name_tuple[1]}"
    #             valid_link_names.append(f"link_{from_name}_{to_name}")
    #         file.write(f"names={valid_link_names}\n")
    #         file.write("\n")

    #         # write the links
    #         for valid_link_name_tuple, valid_link in valid_links.items():
    #             from_name = valid_link_name_tuple[0] if valid_link_name_tuple[0] == "source" else \
    #                 f"compute_node_{self.node_idx_offset + valid_link_name_tuple[0]}"
    #             to_name = valid_link_name_tuple[1] if valid_link_name_tuple[1] == "sink" else \
    #                 f"compute_node_{self.node_idx_offset + valid_link_name_tuple[1]}"
    #             file.write(f"[link_{from_name}_{to_name}]\n")
    #             file.write(f"in={from_name}\n")
    #             file.write(f"out={to_name}\n")
    #             file.write(f"latency={valid_link.latency * 1000} * MilliSec\n")
    #             file.write(f"bandwidth={valid_link.bandwidth / mbps} * mbps\n")
    #             file.write("\n")

    # def save_layout_solution(self, save_path: str) -> None:
    #     """
    #     Save the layout solution found.
    #     Format:
    #     [Solution]
    #     name_in_cluster_file=[a list of layer ids]

    #     :param save_path: save path of solution file
    #     :return: None
    #     """
    #     assert self.solution_loaded, "Must find a solution before saving!"
    #     with open(save_path, "w") as file:
    #         file.write("[Settings]\n")
    #         file.write(f"offset={self.node_idx_offset}\n")
    #         file.write("\n")
    #         file.write("[Solution]\n")
    #         for ilp_node_idx, ilp_node in self.ilp_nodes.items():
    #             if ilp_node.node_type != NodeType.COMPUTE:
    #                 continue

    #             file.write(f"compute_node_{self.node_idx_offset + int(ilp_node_idx)}=")
    #             file.write(f"{list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))}\n")

    # def set_initial_layout(self, simulator: ClusterSimulator) -> float:
    #     """
    #     Load the initial model layout found by the ILP solver into the simulator.

    #     :param simulator: the cluster simulator to load model into
    #     :return: expected loading time in simulation
    #     """
    #     assert self.solution_loaded, "Must load a solution before setting initial layout for simulator!"
    #     assert simulator.current_time == 0, "Initial layout can only be set at the beginning!"

    #     max_load_time: float = 0
    #     for ilp_node_idx, ilp_node in self.ilp_nodes.items():
    #         # get the corresponding compute node in the simulator
    #         compute_node_name = f"compute_node_{self.node_idx_offset + int(ilp_node_idx)}"
    #         compute_node = simulator.name_2_compute_node[compute_node_name]

    #         # get the model layers to load and corresponding loading time
    #         new_layers = list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))
    #         new_layers_size = sum(self.model_manager.get_model_params()[ilp_node.start_layer_idx:
    #                                                                     ilp_node.end_layer_idx])
    #         loading_time = new_layers_size / compute_node.disk_speed
    #         max_load_time = max(max_load_time, loading_time)

    #         # issue load command
    #         simulator.issue_command_load_model(load_time=simulator.current_time,
    #                                            node_uid=compute_node.node_uid,
    #                                            new_layers=new_layers,
    #                                            request_uids_to_wait=[])

    #     # advance simulator
    #     max_load_time = math.ceil(max_load_time) + 1
    #     simulator.simulate(until=max_load_time)
    #     return max_load_time

    # def get_flow_parameters(self) -> FlowParameters:
    #     """
    #     Get flow parameters based on the loaded cluster file.

    #     :return: FlowParameters
    #     """
    #     assert self.solution_loaded, "Solution must be loaded before FlowParameters can be returned!"
    #     return FlowParameters(token_size=self.model_card.token_size,
    #                           token_activation_size=self.model_card.activation_size)

    # def get_query_manager_parameters(self) -> QueryManagerParameters:
    #     """
    #     Get query manager parameters based on the loaded cluster file.

    #     :return: QueryManagerParameters
    #     """
    #     assert self.solution_loaded, "Solution must be loaded before QueryManagerParameters can be returned!"
    #     return QueryManagerParameters(token_size=self.model_card.token_size,
    #                                   token_activation_size=self.model_card.activation_size,
    #                                   total_num_layers=self.model_spec.n_layers)

    # def get_ilp_max_flow(self) -> float:
    #     """
    #     Get the max flow found by the ILP solver.

    #     :return: max flow.
    #     """
    #     assert self.solution_loaded, "Solution must be loaded before ILP max flow can be returned!"
    #     sum_of_flow = 0
    #     for other_idx in self.ilp_source.connected_node_indices:
    #         sum_of_flow += self.ilp_links[("source", other_idx)].forward_flow
    #     return sum_of_flow



    # def detect_ilp_partial_inference(self) -> bool:
    #     """
    #     Detect whether the solution found by ILP solver uses partial inference in the max flow.

    #     :return: whether partial inference is used
    #     """
    #     assert self.solution_loaded, "Solution must be loaded before we can detect partial inference!"
    #     for link_name_tuple, link in self.ilp_links.items():
    #         if link.forward_flow > 0:
    #             valid_with_no_partial = self.check_link_validity(from_idx=link_name_tuple[0],
    #                                                              to_idx=link_name_tuple[1],
    #                                                              allow_partial_inference=False)
    #             if not valid_with_no_partial:
    #                 return True

    #         if link.backward_flow > 0:
    #             valid_with_no_partial = self.check_link_validity(from_idx=link_name_tuple[1],
    #                                                              to_idx=link_name_tuple[0],
    #                                                              allow_partial_inference=False)
    #             if not valid_with_no_partial:
    #                 return True
    #     return False

# 2026.02.12: Luca Pedercini

import os

from enum import Enum
from datetime import datetime
from typing import List, Dict, Tuple, Any

from src.ilp_layout_aux import ILPLayout

class LayoutSynthesizer:
    def __init__(self, complete_cluster_file_name: str, model_name: str,
                 workspace_path: str) -> None:
        """
        Synthesize initial model layout for the cluster.


        """
        # paths
        self.complete_cluster_file_name: str = complete_cluster_file_name
        self.workspace_path: str = workspace_path

        # model name and statistics
        self.model_name: str = model_name
        self.layout_synthesizer = ILPLayout()

        # make sure workspace path exists and is empty
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path)

    def synthesize(self, args: Dict[str, Any]) -> str:
        """
        Synthesize the initial layout. Below is contents of args for each layout method.

        seed: [Optional] int, seed for the ILP solver (0 if not provided)

        # ILP related
        "max_run_time": float, max ILP search time (only useful when use_existing_sol = False)
        "early_stop_time": float, early stop time (only useful when use_existing_sol = False)
        "early_stop_threshold": float, a value between 0 and 1 (only useful when use_existing_sol = False)

        :param args: a dict of arguments, see above for more info
        :return: simulator_cluster_file_path
        """

        self.layout_synthesizer: ILPLayout

        # remove files in workspace path ("ilp" folder)
        for file in os.listdir(self.workspace_path):
            os.remove(os.path.join(self.workspace_path, file))

        # save args as a file
        trail_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trail_type: str = "new"
        with open(os.path.join(self.workspace_path, f"{trail_name + trail_type}.ini"), "w") as f:
            for k, v in args.items():
                f.write(f"{k} = {v}\n")
            f.write(f"config_file = {self.complete_cluster_file_name}\n")

        # get random seed
        seed: int = args["seed"] if "seed" in args else 0
        model_name: str = args["model_name"]
        enable_memory: bool = args["enable_memory"]
        batch_size: int = args["batch_size"]
        tp_only: bool = args["tp_only"]
        # 1. Load the cluster into layout synthesizer
        self.layout_synthesizer.from_ini(
            cluster_file_name=self.complete_cluster_file_name,
            model_name=model_name,
            enable_memory=enable_memory,
            batch_size=batch_size,
            tp_only=tp_only
        )

        # 2. Build Model
        max_run_time: float = args["max_run_time"]
        early_stop_time: float = args["early_stop_time"]
        early_stop_threshold: float = args["early_stop_threshold"]
        self.layout_synthesizer.build_model(
            seed=seed,
            model_name=trail_name,
        )

        # 3. Search layout
        self.layout_synthesizer.search_layout(
            max_run_time=max_run_time,
            early_stop_time=early_stop_time,
            early_stop_threshold=early_stop_threshold,
            save_sol_path=os.path.join(self.workspace_path, "ilp_solution.sol"),
            save_model_path=os.path.join(self.workspace_path, "ilp_model.lp")
        )

        # 4. Verify Solution

        # self.layout_synthesizer.load_and_verify_solution(
        #     save_sol_path=os.path.join(self.workspace_path, "ilp_solution.sol"),
        #     allow_partial_inference=allow_partial_inference
        # )


    # def set_layout(self, simulator: ClusterSimulator) -> float:
    #     """
    #     Set the initial layout for the simulator.

    #     :param simulator: ClusterSimulator
    #     :return: float, time used to set the layout
    #     """
    #     if self.layout_method == LayoutMethod.ILP:
    #         self.layout_synthesizer: ILPLayout
    #         return self.layout_synthesizer.set_initial_layout(simulator=simulator)

    #     elif self.layout_method == LayoutMethod.Homogeneous:
    #         self.layout_synthesizer: HomogeneousLayout
    #         return self.layout_synthesizer.set_initial_layout(simulator=simulator)

    #     elif self.layout_method == LayoutMethod.Swarm:
    #         self.layout_synthesizer: SwarmLayout
    #         return self.layout_synthesizer.set_initial_layout(simulator=simulator)

    #     elif self.layout_method == LayoutMethod.Petals:
    #         self.layout_synthesizer: PetalsLayout
    #         return self.layout_synthesizer.set_initial_layout(simulator=simulator)

    #     elif self.layout_method == LayoutMethod.LoadExisting:
    #         self.layout_synthesizer: LoadExistingLayout
    #         return self.layout_synthesizer.set_initial_layout(simulator=simulator)

    #     else:
    #         assert False, f"Found unknown layout method: {self.layout_method}!"

    # def get_flow_parameters(self) -> FlowParameters:
    #     """
    #     Get flow parameters based on the loaded cluster file.

    #     :return: FlowParameters
    #     """
    #     return self.layout_synthesizer.get_flow_parameters()

    # def get_query_manager_parameters(self) -> QueryManagerParameters:
    #     """
    #     Get query manager parameters based on the loaded cluster file.

    #     :return: QueryManagerParameters
    #     """
    #     return self.layout_synthesizer.get_query_manager_parameters()

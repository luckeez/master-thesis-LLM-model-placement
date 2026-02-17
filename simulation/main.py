# 2026.02.12: Luca Pedercini

import sys
from src.layout_synthesizer import LayoutSynthesizer
from src.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def ilp_layout(model_name, complete_cluster_file_name):
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=complete_cluster_file_name,
        model_name=model_name,
        workspace_path="./layouts/ilp",
    )

    # setting arguments for ILP layout synthesis
    # see simulator.initial_layout.layout_synthesizer.synthesize for more details about the arguments
    ilp_args = {
        # ILP
        "max_run_time": 1800, # half an hour
        "early_stop_time": 100,
        "early_stop_threshold": 0.95,
        "enable_memory": True,
        "model_name": model_name,
        "batch_size": 8,
        "tp_only": True
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main(complete_cluster_file_name=None, model_name="LLaMa30B"):
    """
    Find a model placement for the cluster. The model placement specifies which
    layers each machine holds.
    """
    layout_method = "ilp"
     # model name, should be one of the keys in MODEL_SPECS in src/specs.py
    
    if not complete_cluster_file_name:
        complete_cluster_file_name = "./config/test-config.ini"

    if layout_method == "ilp":
        # ILP layout synthesis
        # Note: We set the max running time to 10 hours. However, you can stop the process at any time (ctrl + c ONCE)
        # and the best solution found so far will be saved. In this example, we early stop at around 10 minutes.
        # Depending on the random seed, the running time and model placement found may vary.
        ilp_layout(model_name=model_name,
                   complete_cluster_file_name=complete_cluster_file_name)
        print(f"ILP layout synthesis is done! (Results in ./layouts/ilp)")

    else:
        raise ValueError(f"Unknown layout method: {layout_method}")


if __name__ == '__main__':
    main()

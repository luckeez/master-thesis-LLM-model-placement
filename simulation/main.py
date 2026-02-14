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
        "max_run_time": 36000,
        "early_stop_time": 100,
        "early_stop_threshold": 0.995,
        "enable_memory": True,
        "model_name": "LLaMa30B",
        "batch_size": 8
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    """
    Find a model placement for the cluster. The model placement specifies which
    layers each machine holds.
    """
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <layout_method> (ilp)"
    layout_method = sys.argv[1]

    model_name = "LLaMa30B"  # model name, should be one of the keys in MODEL_SPECS in src/specs.py
    
    complete_cluster_file_name = "./config/aux12-1g-mem.ini"

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

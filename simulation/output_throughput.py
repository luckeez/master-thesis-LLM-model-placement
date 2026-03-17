import matplotlib.pyplot as plt
from typing import List
import os

from src.formulas import calc_throughput
import src.formulas as formulas
from src.specs import ModelSpec, MODEL_SPECS, GPUSpec, GPU_SPECS, LinkSpec, LINK_SPECS, BYTE_PER_PARAM, N_CONCURRENT_REQUESTS, AVERAGE_CONTEXT_WINDOW

def simulate_inference_scaling(model: ModelSpec, gpu: GPUSpec, max_gpus: int, assigned_layers = None, batch_size: int = 1, link_type: str = None):
    print(f"Simulating for GPU: {gpu.name}")

    results_n_gpus = []
    results_throughput = []

    for n in range(1, max_gpus + 1):

        # Check if fits in memory
        if not formulas.check_memory_constraint(model=model, gpu=gpu, n_gpu=n, batch_size=batch_size, assigned_layers=assigned_layers):
            results_n_gpus.append(n)
            results_throughput.append(0)
            continue

        if link_type is None:
            if gpu.name == "T4" or "L4" or "A30":
                link_type = "PCIe"
            else:
                link_type = "NVLink"
        throughput = calc_throughput(model=model, gpu=gpu, n_gpu=n, batch_size=batch_size, link_type=link_type, assigned_layers=assigned_layers)
        
        results_n_gpus.append(n)
        results_throughput.append(throughput)

    return results_n_gpus, results_throughput


def simulate_inference_scaling_with_cost(model: ModelSpec, gpu: GPUSpec, max_gpus: int, assigned_layers = None, batch_size: int = 1, link_type: str = None):
    print(f"Simulating for GPU: {gpu.name}")

    results = {
        "n_gpus": [],
        "throughput": [],
        "cost_rent_hourly": [], # USD/ora
        "energy_kwh_hourly": [] # kWh consumati in un'ora
    }

    for n in range(1, max_gpus + 1):

        # Check if fits in memory. Use 0.85 factor to leave some buffer
        if not formulas.check_memory_constraint(model=model, gpu=gpu, n_gpu=n, batch_size=batch_size, assigned_layers=assigned_layers):
            results["n_gpus"].append(n)
            results["throughput"].append(None)
            results["cost_rent_hourly"].append(None)
            results["energy_kwh_hourly"].append(None)
            continue

        if link_type is None:
            if gpu.name == "T4" or "L4":
                link_type = "PCIe"
            else:
                link_type = "NVLink"
        throughput = calc_throughput(model=model, gpu=gpu, n_gpu=n, batch_size=batch_size, link_type=link_type, assigned_layers=assigned_layers)
        if gpu.name == "A30" or gpu.name == "A100-80":
            print(f"Th: {throughput}, gpu: {gpu.name}, n_gpu: {n}")

        rent_cost = formulas.calc_rental_cost(gpu=gpu, n_gpu=n, hours=1)
        energy = formulas.calc_power_cost(gpu=gpu, n_gpu=n, hours=1, cost_per_kwh=0.15)



        results["n_gpus"].append(n)
        results["throughput"].append(throughput)
        results["cost_rent_hourly"].append(rent_cost)
        results["energy_kwh_hourly"].append(energy)

    return results

# ------------ PLOTS ------------
def plot_throughput(list_results: dict[dict], model: ModelSpec, assigned_layers = None, show=False):
    plt.figure(figsize=(10, 6))
    for gpu_name, results in list_results.items():
        plt.plot(results["n_gpus"], results["throughput"], marker='o', label=f"{gpu_name}")
    
    if assigned_layers is None:
        assigned_layers = model.n_layers
    
    # plt.title(f"Inference Throughput Scaling (Model: {model.name}) (Num layers: {assigned_layers})")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Inference Throughput (tokens/sec)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    if not show:
        plt.savefig(f"images/output_plots/{model.name}/inference_scaling_{model.name}.png")
    else:
        plt.show()

def plot_cost(list_results: dict[dict], model: ModelSpec):
    metrics = ["cost_rent_hourly", "energy_kwh_hourly"]
    titles = ["Rental Cost (Hourly)", "Power Consumption (kWh Hourly)"]
    ylabels = ["Cost (USD/hour)", "Energy (kWh)"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle(f"Cost Metrics Scaling (Model: {model.name})")

    i = 0
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        ax = axs[i]
        for gpu_name, results in list_results.items():
            ax.plot(results["n_gpus"], results[metric], marker='o', label=f"{gpu_name}")
        i+=1
        ax.set_title(title)
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"images/output_plots/{model.name}/cost_metrics_{model.name}.png")
    # plt.show()

def plot_rental_vs_throughput(list_results: dict[dict], model: ModelSpec):
    plt.figure(figsize=(10, 6))

    for gpu_name, results in list_results.items():
        throughput_per_rent = [tp / cost if cost is not None else None for tp, cost in zip(results["throughput"], results["cost_rent_hourly"])]
        plt.plot(results["n_gpus"], throughput_per_rent, marker='o', label=f"{gpu_name}")

    plt.title(f"Inference Throughput per Dollar (Rental Cost) - Model: {model.name}")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Inference Throughput per Dollar (tokens/usd)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/output_plots/{model.name}/rental_cost_vs_throughput_{model.name}.png")
    # plt.show()

def plot_power_vs_throughput(list_results: dict[dict], model: ModelSpec):
    plt.figure(figsize=(10, 6))

    for gpu_name, results in list_results.items():
        throughput_per_energy = [tp / energy if energy is not None else None for tp, energy in zip(results["throughput"], results["energy_kwh_hourly"])]
        plt.plot(results["n_gpus"], throughput_per_energy, marker='o', label=f"{gpu_name}")

    plt.title(f"Inference Throughput per kWh - Model: {model.name}")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Inference Throughput per kWh (tokens/kWh)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/output_plots/{model.name}/power_cost_vs_throughput_{model.name}.png")
    # plt.show()

import matplotlib.pyplot as plt

def plot_scatter(list_results: dict, model: ModelSpec):
    fig, ax = plt.subplots(figsize=(14, 9))

    # Distinct style per GPU
    gpu_styles = {
        "T4":       {"color": "#EF5350", "marker": "v"},
        "L4":       {"color": "#FF7043", "marker": "o"},
        "A10":      {"color": "#AB47BC", "marker": "s"},
        "A30":      {"color": "#66BB6A", "marker": "^"},
        "A40":      {"color": "#26A69A", "marker": "D"},
        "A4000":    {"color": "#BDBDBD", "marker": "p"},
        "A6000":    {"color": "#42A5F5", "marker": "h"},
        "A100-40":  {"color": "#FFA726", "marker": "<"},
        "A100-80":  {"color": "#EC407A", "marker": ">"},
        "L40S":     {"color": "#7E57C2", "marker": "P"},
        "H100":     {"color": "#29B6F6", "marker": "*"},
    }

    KEY_COUNTS = [1, 2, 3, 4, 5, 6, 8, 16]

    for gpu_name, results in list_results.items():
        # Filter to key GPU counts with valid throughput
        filtered = []
        for i in range(len(results["n_gpus"])):
            n = results["n_gpus"][i]
            tp = results["throughput"][i]
            cost = results["cost_rent_hourly"][i]
            if n in KEY_COUNTS and tp is not None and tp > 0 and tp < 800:
                filtered.append((tp, cost, n))

        if not filtered:
            continue

        # Sort by GPU count for line connection
        filtered.sort(key=lambda x: x[2])
        tps = [p[0] for p in filtered]
        costs = [p[1] for p in filtered]
        ns = [p[2] for p in filtered]

        style = gpu_styles.get(gpu_name, {"color": "#888", "marker": "o"})

        # Draw connecting line (scaling trajectory)
        ax.plot(tps, costs, color=style["color"], linewidth=1.5, alpha=0.4, zorder=2)

        # Plot points
        ax.scatter(tps, costs, label=gpu_name, color=style["color"],
                   marker=style["marker"], s=100, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=4)

        # Label: count number on all points
        for i_pt, (tp, cost, n) in enumerate(filtered):
            ax.annotate(f"{n}", (tp, cost),
                        textcoords="offset points", xytext=(0, 7),
                        ha='center', fontsize=16, fontweight='bold', color=style["color"],
                        alpha=0.8)

        # GPU name label near the last point
        ltp, lcost, ln = filtered[-1]
        # ax.annotate(f"{gpu_name}", (ltp, lcost),
        #             textcoords="offset points", xytext=(12, 0),
        #             ha='left', fontsize=7.5, color=style["color"],
        #             fontweight='bold')

    # ax.set_title(f"Rental Cost vs Throughput — {model.name}",
    #              fontweight='bold')
    ax.set_xlabel("Throughput (Tokens / Sec)", fontsize=14)
    ax.set_ylabel("Rental Cost (USD / Hour)", fontsize=14)
    ax.set_xlim(left=100)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title="GPU Type", loc="upper left", fontsize=15, ncol=2,
              framealpha=0.9, title_fontsize=15)

    fig.tight_layout()
    fig.savefig(f"images/output_plots/{model.name}/scatter_rental_vs_perf_{model.name}.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)
# ------------ END PLOTS ------------

if __name__ == "__main__":
    # model: ModelSpec = MODEL_SPECS["LlaMa-3.1-8B"]
    model: ModelSpec = MODEL_SPECS["LLaMa70B"]
    assigned_layers = 80

    list_results: dict[dict] = {}
    for gpu in GPU_SPECS.values():
        results = simulate_inference_scaling_with_cost(model=model, gpu=gpu, max_gpus=16, batch_size=8, assigned_layers=assigned_layers)
        list_results[gpu.name] = results

    for m in MODEL_SPECS.keys():
        if not os.path.exists(f"images/output_plots/{m}"):
            os.makedirs(f"images/output_plots/{m}")
    plot_throughput(list_results=list_results, model=model, assigned_layers=assigned_layers)
    # plot_purchase_vs_throughput(list_results=list_results, model=model)
    # plot_rental_vs_throughput(list_results=list_results, model=model)
    # plot_power_vs_throughput(list_results=list_results, model=model)
    # plot_cost(list_results=list_results, model=model)
    plot_scatter(list_results=list_results, model=model)


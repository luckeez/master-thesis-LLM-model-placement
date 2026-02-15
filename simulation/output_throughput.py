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
        print(results["throughput"])
    
    if assigned_layers is None:
        assigned_layers = model.n_layers
    
    plt.title(f"Inference Throughput Scaling (Model: {model.name}) (Num layers: {assigned_layers})")
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
    plt.figure(figsize=(18, 12))
    
    # Definiamo colori o marker specifici per ogni GPU per coerenza
    # (Opzionale, ma aiuta molto la leggibilità)
    gpu_styles = {
        "T4": {"color": "tab:red", "marker": "o"},
        "L4": {"color": "tab:orange", "marker": "s"},
        "A100-80GB": {"color": "tab:green", "marker": "^"},
        "H100": {"color": "tab:blue", "marker": "D"}
    }
    
    # 1. Loop per GPU (Serie Dati)
    for gpu_name, results in list_results.items():
        
        # Filtriamo i dati PRIMA di plottare
        # Vogliamo solo i punti validi (>0) e che siano nei target [1, 2, 4...]
        valid_x = []
        valid_y = []
        valid_n = []
        
        for i in range(len(results["n_gpus"])):
            n_val = results["n_gpus"][i]
            th_val = results["throughput"][i]
            cost_val = results["cost_purchase"][i]
            
            # Check: Deve essere un numero di GPU di interesse E avere throughput valido
            # if n_val in [1, 2, 4, 8, 16] and th_val is not None and th_val > 0:
            if th_val is not None and th_val > 0 and th_val < 600:
                valid_x.append(th_val)
                valid_y.append(cost_val)
                valid_n.append(n_val)
        
        if not valid_x:
            continue # Salta se non ci sono dati validi per questa GPU

        # Recupera stile (o usa default se la GPU non è nel dizionario)
        style = gpu_styles.get(gpu_name, {"color": None, "marker": "o"})

        # 2. PLOT UNICO per l'intera serie (così la legenda funziona!)
        plt.scatter(valid_x, valid_y, 
                    label=gpu_name, # Fondamentale per la legenda
                    # color=style["color"],
                    # marker=style["marker"],
                    s=100, alpha=0.7, edgecolors="black")
        
        # 3. Annotazione intelligente (Loop sui punti filtrati)
        for x, y, n in zip(valid_x, valid_y, valid_n):
            # Scriviamo "4x" invece di "T4"
            plt.annotate(f"{n}x - {gpu_name}", (x, y), 
                         textcoords="offset points", xytext=(0, 8), 
                         ha='center', fontsize=9, fontweight='bold')

    plt.title(f"Rental Cost vs Performance Map (Model: {model.name})")
    plt.xlabel("Throughput (Tokens / Sec)")
    plt.ylabel("Rental Cost (USD / Hour)")
    
    # Imposta limiti assi partendo da 0 per chiarezza
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="GPU Type", loc="upper left")
    
    plt.tight_layout()
    plt.savefig(f"images/output_plots/{model.name}/scatter_purchase_vs_perf_{model.name}.png")
    # plt.show()
# ------------ END PLOTS ------------

if __name__ == "__main__":
    # model: ModelSpec = MODEL_SPECS["LlaMa-3.1-8B"]
    model: ModelSpec = MODEL_SPECS["LLaMa30B"]
    assigned_layers = 20

    list_results: dict[dict] = {}
    for gpu in GPU_SPECS.values():
        results = simulate_inference_scaling_with_cost(model=model, gpu=gpu, max_gpus=16, batch_size=8, assigned_layers=assigned_layers)
        list_results[gpu.name] = results

    for m in MODEL_SPECS.keys():
        if not os.path.exists(f"images/output_plots/{m}"):
            os.makedirs(f"images/output_plots/{m}")
    plot_throughput(list_results=list_results, model=model, assigned_layers=assigned_layers, show=True)
    # plot_purchase_vs_throughput(list_results=list_results, model=model)
    # plot_rental_vs_throughput(list_results=list_results, model=model)
    # plot_power_vs_throughput(list_results=list_results, model=model)
    # plot_cost(list_results=list_results, model=model)
    # plot_scatter(list_results=list_results, model=model)


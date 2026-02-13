# 2026.02.12: Luca Pedercini

from src.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec, ATOL, is_close
from src.specs import ModelSpec, MODEL_SPECS, GPUSpec, GPU_SPECS, LinkSpec, LINK_SPECS, BYTE_PER_PARAM

def calc_throughput(model: ModelSpec, gpu: GPUSpec, n_gpu: int, batch_size: int, link_type: str = None, assigned_layers: int = None, network_latency: float = 0.0) -> float:
    """ Compute throughput in token/s considering compute, memory and communication constraints """
    # Allow to compute throughput for a subset of layers
    num_layers: int = model.n_layers
    if assigned_layers is not None:
        num_layers = assigned_layers

    # Compute token latency
    token_latency: float = calc_token_latency_simple(model=model, gpu=gpu, n_gpu=n_gpu, batch_size=batch_size, link_type=link_type, num_layers=num_layers)

    # Compute throughput as batch_size / token_latency
    throughput: float = batch_size / (token_latency + network_latency)
    return throughput

def calc_token_latency_simple(model: ModelSpec, gpu: GPUSpec, n_gpu: int, batch_size: int, link_type: str, num_layers: int) -> float:
    """
    Compute token latency considering network, kernel, compute and memory latencies. 
    Simplified formula without considering attention reads, from "Inference Economics" paper.
    Formula: token_latency = network_kernel_time + max(memory_time, compute_time)
    Returns: Token latency in milliseconds.
    """
    # Compute network and kernel time
    if n_gpu == 1:
        network_kernel_time: float = 0.0
    else:
        network_kernel_time: float = calc_network_kernel_time(model=model, n_gpu=n_gpu, link_type=link_type, batch_size=batch_size, num_layers=num_layers)

    layer_fraction: float = num_layers / model.n_layers
    n_params_assigned = model.n_params_billion * 1e9 * layer_fraction

    # Compute total number of bytes that are read by the GPU
    bytes_read: int = n_params_assigned * BYTE_PER_PARAM

    # Compute total number of flops to be executed by the GPU
    total_flop: float = 2 * n_params_assigned * batch_size

    # Compute final token latency
    token_latency: float = network_kernel_time + max(bytes_read/(n_gpu * gpu.memory_bandwidth_gbps * gbps), total_flop / (n_gpu * gpu.fp16_tflops * 1e12))
    return token_latency


def calc_network_kernel_time(model: ModelSpec, n_gpu: int, link_type: str, batch_size: int, num_layers: int) -> float:
    """
    Compute network and kernel time based on the number of GPUs.
    Returns: Network and kernel time in milliseconds.
    """
    n_reduce: int = 4 # Number of all reduce per layer. Some techniques (Megatron-LM) allows to have only 2 AllReduce, but not used for inference
    g: int = model.n_heads/model.n_kv_heads # attention group size
    d_head: int = model.d_model/model.n_heads # size of vector output by a single attention head
    activation_precision: int = BYTE_PER_PARAM
    intra_node_bw: int = LINK_SPECS[link_type].bandwidth * GB

    # Compute total number of bytes reduced in a single forward pass doing AllReduce. 
    # NOTE: the d_ff term in the parenthesis is multiplied by two because modern models like Llama use SwiGLU activation,
    # that needs an extra matrix (w.r.t. paper formula)
    bytes_reduced: float = ((1 + 2/g) * model.n_heads * d_head + 2 * model.d_model + 2 * model.d_ff) * batch_size * num_layers * activation_precision

    # Compute total amount of intra node communications
    intra_node_reads: float = 2 * (n_gpu - 1) * bytes_reduced

    # Compute approximate latency for an all-reduce using tree topology, according to NVIDIA NCCL data
    t_reduce: float = (6.8 + 1.2 * (n_gpu - 1)) * 1e-6

    # Compute final network and communication time
    result: float = num_layers * n_reduce * t_reduce + intra_node_reads / (n_gpu * intra_node_bw) # TODO: check with Marco. divide only by bw?
    return result

def check_memory_constraint(model: ModelSpec, gpu: GPUSpec, n_gpu: int, batch_size: int, assigned_layers = None) -> bool:
    """ 
    Check if the model fits in memory given the number of GPUs and batch size.
    Returns True if fits, False otherwise.
    """
    if assigned_layers is None:
        assigned_layers = model.n_layers

    layer_fraction = assigned_layers / model.n_layers

    # Model weights
    total_weight_bytes = model.n_params_billion * 1e9 * BYTE_PER_PARAM * layer_fraction
    weight_per_gpu = total_weight_bytes / n_gpu

    # KV Cache
    head_dim = model.d_model / model.n_heads
    total_kv_cache_bytes = 2 * assigned_layers * model.n_kv_heads * head_dim * batch_size * BYTE_PER_PARAM 
    kv_cache_per_gpu = total_kv_cache_bytes / n_gpu

    # Total memory per GPU
    total_memory_per_gpu = weight_per_gpu + kv_cache_per_gpu

    # Check if fits in memory. Use 0.85 factor to leave some buffer
    if total_memory_per_gpu > gpu.memory_gb * 1e9 * 0.85:
        return False
    return True

def calc_purchase_cost(gpu: GPUSpec, n_gpu: int) -> float:
    """ 
    Compute the economic cost of using n_gpu of a given gpu type.
    """
    total_gpu_cost: float = gpu.buy_cost * n_gpu
    return total_gpu_cost

def calc_rental_cost(gpu: GPUSpec, n_gpu: int, hours: float) -> float:
    """ 
    Compute the rental cost of using n_gpu of a given gpu type for a certain number of hours.
    """
    total_rental_cost: float = gpu.rent_cost * n_gpu * hours
    return total_rental_cost

def calc_power_consumption(gpu: GPUSpec, n_gpu: int, hours: float) -> float:
    """ 
    Compute the power consumption in kWh of using n_gpu of a given gpu type for a certain number of hours.
    """
    total_power_watts: float = gpu.tdp_watts * n_gpu * hours
    total_power_kwh: float = total_power_watts / 1000.0
    return total_power_kwh

def calc_power_cost(gpu: GPUSpec, n_gpu: int, hours: float, cost_per_kwh: float = 0.15) -> float:
    """ 
    Compute the power cost given the power consumption in kWh and the cost per kWh.
    """
    power_kwh: float = calc_power_consumption(gpu=gpu, n_gpu=n_gpu, hours=hours)
    total_power_cost: float = power_kwh * cost_per_kwh
    return total_power_cost


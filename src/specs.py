# 2026.02.12: Luca Pedercini

from dataclasses import dataclass
from typing import List, Dict

BYTE_PER_PARAM = 2  # assuming fp16
N_CONCURRENT_REQUESTS = 10
AVERAGE_CONTEXT_WINDOW = 750

@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_params_billion: int  # in billions of parameters
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    token_size: int

    @property
    def activation_size(self) -> int:
        return self.d_model * BYTE_PER_PARAM

@dataclass
class GPUSpec:
    name: str
    fp16_tflops: float
    memory_gb: int
    memory_bandwidth_gbps: int
    buy_cost: float # cost to buy the GPU [usd]
    rent_cost: float # cost to rent the GPU [usd/hour]
    tdp_watts: int

@dataclass
class LinkSpec:
    name: str
    bandwidth: int


MODEL_SPECS: Dict[str, ModelSpec] = {
    "LLaMa8B": ModelSpec(name="LLaMa8B", n_layers=32, n_params_billion=8, d_model=4096, n_heads=32, n_kv_heads=8, d_ff=14336, token_size=2), # LlaMa-3.1-8B
    "LLaMa70B": ModelSpec(name="LLaMa70B", n_layers=80, n_params_billion=70, d_model=8192, n_heads=64, n_kv_heads=8, d_ff=28672, token_size=2), # LlaMa-3.1-70B
    "LLaMa30B": ModelSpec(name="LLaMa30B", n_layers=60, n_params_billion=30, d_model=6656, n_heads=48, n_kv_heads=8, d_ff=17920, token_size=2) # LlaMa-1-30B
}

GPU_SPECS: Dict[str, GPUSpec] = {
    "T4": GPUSpec(name="T4", fp16_tflops=65, memory_gb=16, memory_bandwidth_gbps=320, buy_cost=900, rent_cost=0.35, tdp_watts=70),
    "L4": GPUSpec(name="L4", fp16_tflops=242, memory_gb=24, memory_bandwidth_gbps=300, buy_cost=2700, rent_cost=0.47, tdp_watts=75),
    "A100": GPUSpec(name="A100-40", fp16_tflops=312, memory_gb=40, memory_bandwidth_gbps=1555, buy_cost=11000, rent_cost=1.29, tdp_watts=250),
    "A100-80": GPUSpec(name="A100-80", fp16_tflops=312, memory_gb=80, memory_bandwidth_gbps=2040, buy_cost=19000, rent_cost=1.65, tdp_watts=400),

    "H100": GPUSpec(name="H100", fp16_tflops=1979, memory_gb=80, memory_bandwidth_gbps=3350, buy_cost=25000, rent_cost=1.9, tdp_watts=700),
    "L40S": GPUSpec(name="L40S", fp16_tflops=733, memory_gb=48, memory_bandwidth_gbps=864, buy_cost=7500, rent_cost=1.36, tdp_watts=350),
    "A30": GPUSpec(name="A30", fp16_tflops=235, memory_gb=24, memory_bandwidth_gbps=933, buy_cost=5500, rent_cost=0.33, tdp_watts=165)
}

LINK_SPECS: Dict[str, LinkSpec] = {
    "PCIe": LinkSpec(name="PCIe", bandwidth=64),
    "NVLink": LinkSpec(name="NVLink", bandwidth=600)
}
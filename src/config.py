"""
配置文件 - MoE语言模型端到端效率优化
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "mistralai/Mixtral-8x7B-v0.1"
    model_path: str = ""  # 留空表示从HuggingFace直接加载
    num_experts: int = 8
    num_experts_per_tok: int = 2
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    
@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    data_path: str = ""  # 留空表示从HuggingFace直接加载
    max_samples: int = 100  # 测试时使用的样本数
    max_length: int = 512    # 最大序列长度

@dataclass
class OptimizationConfig:
    """优化配置"""
    # 量化配置
    use_quantization: bool = True
    quantization_bits: int = 8  # 4 或 8
    quantization_method: str = "bitsandbytes"  # bitsandbytes, gptq, awq
    
    # 显存优化
    use_offloading: bool = True
    offload_dir: str = "./offload"
    
    # 推理优化
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    batch_size: int = 1
    
    # 混合精度
    use_fp16: bool = True
    use_bf16: bool = False

@dataclass
class HardwareConfig:
    """硬件配置"""
    device: str = "cuda"
    num_gpus: int = 2
    gpu_memory_gb: int = 64  # K100-AI 单卡64GB
    
@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

# 默认配置
default_config = Config()

# 环境变量设置
def setup_environment():
    """设置环境变量"""
    # 使用HuggingFace镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    # 设置PyTorch内存分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    print("环境变量已设置:")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
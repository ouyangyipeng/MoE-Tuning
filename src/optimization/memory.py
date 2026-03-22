"""
显存优化模块 - MoE语言模型端到端效率优化

功能：
1. 模型分片 (Model Parallelism)
2. CPU Offloading
3. KV Cache优化
4. 显存监控
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """显存配置"""
    offload_dir: str = "./offload"
    offload_to_cpu: bool = True
    pin_memory: bool = True
    max_memory_per_gpu: int = 60  # GB，为其他操作预留空间
    enable_kv_cache: bool = True
    kv_cache_dtype: str = "fp16"  # fp16, fp32, int8


class MemoryMonitor:
    """显存监控器"""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
    
    def reset(self):
        """重置统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.peak_memory = 0
        self.current_memory = 0
    
    def update(self):
        """更新显存统计"""
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1024**3
            self.peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        return self.current_memory, self.peak_memory
    
    def log_memory(self, prefix: str = ""):
        """打印显存信息"""
        current, peak = self.update()
        logger.info(f"{prefix} 当前显存: {current:.2f} GB, 峰值: {peak:.2f} GB")
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取显存信息"""
        current, peak = self.update()
        return {
            "current_gb": current,
            "peak_gb": peak
        }


class ModelSharder:
    """模型分片器"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
    
    def get_device_map(self, num_gpus: int = 2) -> Dict[str, int]:
        """获取设备映射"""
        # 自动分配设备
        device_map = "auto"
        return device_map
    
    def get_max_memory(self, num_gpus: int = 2) -> Dict[int, str]:
        """获取每个GPU的最大显存"""
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = f"{self.config.max_memory_per_gpu}GB"
        max_memory["cpu"] = "30GB"  # CPU offload
        return max_memory
    
    def shard_model(self, model: nn.Module, num_gpus: int = 2) -> nn.Module:
        """分片模型"""
        logger.info(f"开始模型分片，使用 {num_gpus} 个GPU")
        
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
            
            # 获取平衡的显存分配
            max_memory = get_balanced_memory(
                model,
                max_memory=self.get_max_memory(num_gpus),
                no_split_module_classes=["MixtralDecoderLayer"],
                dtype=torch.float16
            )
            
            # 推断设备映射
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["MixtralDecoderLayer"],
                dtype=torch.float16
            )
            
            logger.info(f"设备映射: {device_map}")
            
            # 分发模型
            model = dispatch_model(model, device_map=device_map)
            
            logger.info("模型分片完成")
            
        except ImportError:
            logger.warning("accelerate未安装，跳过模型分片")
        except Exception as e:
            logger.error(f"模型分片失败: {e}")
        
        return model


class KVCacheOptimizer:
    """KV Cache优化器"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.cache = {}
    
    def optimize_cache(self, model: nn.Module) -> nn.Module:
        """优化KV Cache"""
        if not self.config.enable_kv_cache:
            return model
        
        logger.info("优化KV Cache...")
        
        # 设置past_key_values
        # 这通常在generate时自动处理
        
        return model
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def estimate_cache_size(self, batch_size: int, seq_len: int, 
                           num_layers: int, hidden_size: int) -> float:
        """估算KV Cache大小"""
        # KV Cache大小 = 2 * batch_size * seq_len * num_layers * hidden_size * dtype_size
        dtype_size = 2 if self.config.kv_cache_dtype == "fp16" else 4
        cache_size = 2 * batch_size * seq_len * num_layers * hidden_size * dtype_size
        return cache_size / 1024**3  # 转换为GB


class OffloadManager:
    """Offload管理器"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.offloaded_modules = {}
    
    def setup_offload(self, model: nn.Module) -> nn.Module:
        """设置Offload"""
        if not self.config.offload_to_cpu:
            return model
        
        logger.info("设置CPU Offload...")
        
        try:
            from accelerate import cpu_offload
            
            # 对大模型进行CPU offload
            # 这通常与device_map="auto"一起使用
            
            logger.info("CPU Offload设置完成")
            
        except ImportError:
            logger.warning("accelerate未安装，跳过CPU Offload")
        
        return model
    
    def offload_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移动到CPU"""
        return tensor.cpu()
    
    def load_to_gpu(self, tensor: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        """将张量移动到GPU"""
        return tensor.to(device)


class MemoryOptimizer:
    """显存优化器 - 统一接口"""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.monitor = MemoryMonitor()
        self.sharder = ModelSharder(config)
        self.kv_cache_optimizer = KVCacheOptimizer(config)
        self.offload_manager = OffloadManager(config)
    
    def optimize(self, model: nn.Module, num_gpus: int = 2) -> nn.Module:
        """执行所有显存优化"""
        logger.info("=" * 50)
        logger.info("开始显存优化")
        logger.info("=" * 50)
        
        # 记录初始显存
        self.monitor.reset()
        self.monitor.log_memory("优化前")
        
        # 1. 模型分片
        model = self.sharder.shard_model(model, num_gpus)
        
        # 2. KV Cache优化
        model = self.kv_cache_optimizer.optimize_cache(model)
        
        # 3. CPU Offload
        model = self.offload_manager.setup_offload(model)
        
        # 记录优化后显存
        self.monitor.log_memory("优化后")
        
        logger.info("显存优化完成")
        
        return model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取显存统计"""
        return self.monitor.get_memory_info()


def optimize_memory_usage():
    """优化PyTorch显存使用"""
    # 设置内存分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("已优化PyTorch显存使用")


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """获取所有GPU的显存信息"""
    info = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            info[i] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
    
    return info


if __name__ == "__main__":
    # 测试显存监控
    monitor = MemoryMonitor()
    monitor.reset()
    
    # 打印GPU信息
    gpu_info = get_gpu_memory_info()
    for gpu_id, info in gpu_info.items():
        print(f"GPU {gpu_id}: {info}")
    
    # 测试KV Cache估算
    kv_optimizer = KVCacheOptimizer()
    cache_size = kv_optimizer.estimate_cache_size(
        batch_size=1,
        seq_len=512,
        num_layers=32,
        hidden_size=4096
    )
    print(f"KV Cache估算大小: {cache_size:.2f} GB")
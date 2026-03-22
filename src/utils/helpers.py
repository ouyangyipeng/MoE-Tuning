"""
工具函数模块 - MoE语言模型端到端效率优化

功能：
1. 数据处理工具
2. 日志工具
3. 文件操作
4. 性能测量
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 配置日志
def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_json(data: Dict, filepath: str):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """获取时间戳"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def format_memory(gb: float) -> str:
    """格式化显存"""
    if gb < 1:
        return f"{gb * 1024:.2f}MB"
    else:
        return f"{gb:.2f}GB"


class Timer:
    """计时器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.name:
            print(f"{self.name}: {format_time(elapsed)}")
    
    @property
    def elapsed(self) -> float:
        """获取已用时间"""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数总量: {total_params / 1e9:.2f}B")
    print(f"可训练参数: {trainable_params / 1e9:.2f}B")
    print(f"参数占用显存 (FP16): {total_params * 2 / 1024**3:.2f}GB")


def print_gpu_info():
    """打印GPU信息"""
    import torch
    
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  设备 {i}: {props.name}")
        print(f"  显存: {props.total_memory / 1024**3:.2f}GB")
        print(f"  计算能力: {props.major}.{props.minor}")


def count_parameters(model) -> Dict[str, int]:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen
    }


def get_model_size(model) -> Dict[str, float]:
    """获取模型大小"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "params_mb": param_size / 1024**2,
        "buffers_mb": buffer_size / 1024**2,
        "total_mb": (param_size + buffer_size) / 1024**2
    }


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试计时器
    with Timer("测试操作") as t:
        time.sleep(0.5)
    
    # 测试平均值计算器
    meter = AverageMeter("延迟")
    for i in range(5):
        meter.update(i * 0.1)
    print(meter)
    
    # 测试GPU信息
    print_gpu_info()
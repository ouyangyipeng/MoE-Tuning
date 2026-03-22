"""
量化优化模块 - MoE语言模型端到端效率优化

支持的量化方法：
1. INT8量化 (bitsandbytes)
2. INT4量化 (bitsandbytes)
3. GPTQ量化
4. AWQ量化
"""

import os
import sys
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """量化配置"""
    method: str = "int8"  # int8, int4, gptq, awq
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


class Quantizer:
    """量化器"""
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self.quantization_config = None
    
    def get_quantization_config(self) -> Any:
        """获取量化配置"""
        try:
            from transformers import BitsAndBytesConfig
            
            if self.config.method == "int8":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=self.config.llm_int8_threshold,
                    llm_int8_has_fp16_weight=self.config.llm_int8_has_fp16_weight
                )
                logger.info("使用INT8量化配置")
                
            elif self.config.method == "int4":
                compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
                )
                logger.info("使用INT4量化配置")
                
            else:
                logger.warning(f"未知的量化方法: {self.config.method}，将不使用量化")
                self.quantization_config = None
            
            return self.quantization_config
            
        except ImportError:
            logger.warning("bitsandbytes未安装，无法使用量化")
            return None
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """量化模型（后训练量化）"""
        logger.info("开始后训练量化...")
        
        # 对于已经加载的模型进行量化
        if self.config.method == "int8":
            model = self._quantize_int8(model)
        elif self.config.method == "int4":
            model = self._quantize_int4(model)
        
        return model
    
    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """INT8量化"""
        try:
            import bitsandbytes as bnb
            
            # 遍历模型的所有线性层
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # 替换为INT8线性层
                    quantized_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                        threshold=self.config.llm_int8_threshold
                    )
                    
                    # 复制权重
                    quantized_module.weight = bnb.nn.Int8Params(
                        module.weight.data,
                        requires_grad=False,
                        has_fp16_weights=False
                    )
                    if module.bias is not None:
                        quantized_module.bias = module.bias
                    
                    # 替换模块
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, quantized_module)
            
            logger.info("INT8量化完成")
            
        except ImportError:
            logger.warning("bitsandbytes未安装，跳过INT8量化")
        
        return model
    
    def _quantize_int4(self, model: nn.Module) -> nn.Module:
        """INT4量化"""
        try:
            import bitsandbytes as bnb
            
            # 遍历模型的所有线性层
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # 替换为INT4线性层
                    quantized_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                        quant_type=self.config.bnb_4bit_quant_type,
                        use_double_quant=self.config.bnb_4bit_use_double_quant
                    )
                    
                    # 复制权重
                    quantized_module.weight = bnb.nn.Params4bit(
                        module.weight.data,
                        requires_grad=False,
                        quant_type=self.config.bnb_4bit_quant_type,
                        use_double_quant=self.config.bnb_4bit_use_double_quant
                    )
                    if module.bias is not None:
                        quantized_module.bias = module.bias
                    
                    # 替换模块
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, quantized_module)
            
            logger.info("INT4量化完成")
            
        except ImportError:
            logger.warning("bitsandbytes未安装，跳过INT4量化")
        
        return model


class ExpertQuantizer:
    """专家级量化器 - 针对MoE模型的特殊量化"""
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
    
    def quantize_experts(self, model: nn.Module, expert_names: list = None) -> nn.Module:
        """量化指定的专家层"""
        logger.info("开始专家级量化...")
        
        if expert_names is None:
            # 自动检测专家层
            expert_names = self._find_expert_layers(model)
        
        logger.info(f"发现 {len(expert_names)} 个专家层")
        
        # 对每个专家层应用量化
        for name in expert_names:
            self._quantize_expert(model, name)
        
        return model
    
    def _find_expert_layers(self, model: nn.Module) -> list:
        """查找专家层"""
        expert_names = []
        
        for name, module in model.named_modules():
            # Mixtral的专家层通常包含 "block_sparse_moe" 或 "experts"
            if "block_sparse_moe" in name or "experts" in name:
                if isinstance(module, nn.Linear):
                    expert_names.append(name)
        
        return expert_names
    
    def _quantize_expert(self, model: nn.Module, expert_name: str):
        """量化单个专家"""
        # 获取专家模块
        expert = model.get_submodule(expert_name)
        
        # 应用量化
        quantizer = Quantizer(self.config)
        quantized_expert = quantizer.quantize_model(expert)
        
        # 替换原专家
        parent_name = ".".join(expert_name.split(".")[:-1])
        child_name = expert_name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, quantized_expert)
        
        logger.info(f"已量化专家: {expert_name}")


def get_optimal_quantization_config(gpu_memory_gb: float = 64) -> QuantizationConfig:
    """根据GPU显存自动选择最优量化配置"""
    
    # Mixtral-8x7B 模型大小约 94GB (FP16)
    # 64GB显存需要量化才能加载
    
    if gpu_memory_gb >= 128:
        # 128GB以上可以使用FP16
        logger.info("显存充足，建议使用FP16")
        return QuantizationConfig(method="none")
    elif gpu_memory_gb >= 64:
        # 64GB可以使用INT8
        logger.info("建议使用INT8量化")
        return QuantizationConfig(method="int8")
    else:
        # 更小显存需要INT4
        logger.info("建议使用INT4量化")
        return QuantizationConfig(method="int4")


def estimate_memory_usage(model_size_gb: float, quantization: str = "fp16") -> float:
    """估算模型显存占用"""
    multipliers = {
        "fp32": 1.0,
        "fp16": 0.5,
        "int8": 0.25,
        "int4": 0.125
    }
    
    multiplier = multipliers.get(quantization, 0.5)
    return model_size_gb * multiplier


if __name__ == "__main__":
    # 测试量化配置
    config = QuantizationConfig(method="int8")
    quantizer = Quantizer(config)
    quant_config = quantizer.get_quantization_config()
    print(f"量化配置: {quant_config}")
    
    # 测试显存估算
    memory = estimate_memory_usage(94, "int8")
    print(f"INT8量化后显存估算: {memory:.2f} GB")
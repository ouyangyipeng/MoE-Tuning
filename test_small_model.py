#!/usr/bin/env python3
"""
小模型测试脚本 - 验证代码逻辑

使用小模型测试优化代码的正确性，不加载Mixtral-8x7B
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import default_config, setup_environment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment():
    """测试环境配置"""
    logger.info("=" * 50)
    logger.info("测试环境配置")
    logger.info("=" * 50)
    
    import torch
    import transformers
    import accelerate
    import datasets
    
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"Transformers版本: {transformers.__version__}")
    logger.info(f"Accelerate版本: {accelerate.__version__}")
    logger.info(f"Datasets版本: {datasets.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA可用: True")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  显存: {props.total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA不可用")
    
    return True


def test_small_model():
    """使用小模型测试代码逻辑"""
    logger.info("=" * 50)
    logger.info("使用小模型测试代码逻辑")
    logger.info("=" * 50)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # 使用小模型进行测试
    model_name = "gpt2"  # GPT-2 small, 约124M参数
    
    logger.info(f"加载小模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    logger.info("模型加载成功")
    
    # 测试推理
    test_text = "Hello, this is a test."
    inputs = tokenizer(test_text, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"输入: {test_text}")
    logger.info(f"输出: {generated_text}")
    
    return True


def test_quantization():
    """测试量化功能"""
    logger.info("=" * 50)
    logger.info("测试量化功能")
    logger.info("=" * 50)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = "gpt2"
    
    # INT8量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    logger.info("加载INT8量化模型...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("INT8量化模型加载成功")
            
            # 测试推理
            inputs = tokenizer("Test input", return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
            logger.info("量化模型推理成功")
        else:
            logger.warning("CUDA不可用，跳过量化测试")
        
        return True
    except Exception as e:
        logger.error(f"量化测试失败: {e}")
        return False


def test_dataset_loading():
    """测试数据集加载"""
    logger.info("=" * 50)
    logger.info("测试数据集加载")
    logger.info("=" * 50)
    
    from datasets import load_dataset
    
    try:
        # 加载wikitext数据集
        logger.info("加载wikitext-2数据集（较小）...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        logger.info(f"数据集大小: {len(dataset)}")
        
        # 查看数据
        sample = dataset[0]
        logger.info(f"样本: {sample['text'][:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return False


def test_perplexity_calculation():
    """测试困惑度计算"""
    logger.info("=" * 50)
    logger.info("测试困惑度计算")
    logger.info("=" * 50)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    try:
        # 加载模型和数据
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # 取少量样本计算困惑度
        texts = [item["text"] for item in dataset if len(item["text"]) > 50][:10]
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"困惑度: {perplexity:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"困惑度计算失败: {e}")
        return False


def test_memory_monitoring():
    """测试显存监控"""
    logger.info("=" * 50)
    logger.info("测试显存监控")
    logger.info("=" * 50)
    
    import torch
    
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，跳过显存监控测试")
        return True
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 分配一些显存
    x = torch.randn(1000, 1000, device="cuda")
    
    # 检查显存
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    
    logger.info(f"已分配显存: {allocated:.2f} MB")
    logger.info(f"已预留显存: {reserved:.2f} MB")
    logger.info(f"峰值显存: {peak:.2f} MB")
    
    # 释放
    del x
    torch.cuda.empty_cache()
    
    logger.info("显存监控测试成功")
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始运行所有测试")
    logger.info("=" * 60)
    
    results = {}
    
    # 测试环境
    results["environment"] = test_environment()
    
    # 测试小模型
    results["small_model"] = test_small_model()
    
    # 测试量化
    results["quantization"] = test_quantization()
    
    # 测试数据集加载
    results["dataset"] = test_dataset_loading()
    
    # 测试困惑度计算
    results["perplexity"] = test_perplexity_calculation()
    
    # 测试显存监控
    results["memory"] = test_memory_monitoring()
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        logger.info(f"  {name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    return results


if __name__ == "__main__":
    run_all_tests()
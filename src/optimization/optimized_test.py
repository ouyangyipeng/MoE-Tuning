"""
优化测试脚本 - MoE语言模型端到端效率优化

功能：
1. 加载优化后的Mixtral-8x7B模型
2. 在wikitext-103-v1数据集上测试
3. 测量推理延迟、显存占用、困惑度
4. 与基线对比
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config, setup_environment
from optimization.quantization import Quantizer, QuantizationConfig
from optimization.memory import MemoryOptimizer, MemoryConfig
from optimization.inference import InferenceOptimizer, InferenceConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedTester:
    """优化测试类"""
    
    def __init__(self, config=None):
        self.config = config or default_config
        self.model = None
        self.tokenizer = None
        self.device = self.config.hardware.device
        
        # 优化器
        self.quantizer = None
        self.memory_optimizer = None
        self.inference_optimizer = None
    
    def setup(self):
        """初始化环境"""
        logger.info("初始化测试环境...")
        setup_environment()
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA不可用，将使用CPU运行")
            self.device = "cpu"
    
    def load_model(self, quantization: str = "int8", offload: bool = False):
        """加载优化后的模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"加载优化模型: {self.config.model.model_name}")
        logger.info(f"量化方法: {quantization}, CPU Offload: {offload}")
        
        model_path = self.config.model.model_path
        if not os.path.exists(model_path):
            logger.warning(f"模型路径不存在: {model_path}，将从HuggingFace下载")
            model_path = self.config.model.model_name
        
        # 配置量化
        quant_config = None
        if quantization != "none" and self.device == "cuda":
            quant_config = QuantizationConfig(
                method=quantization
            )
            self.quantizer = Quantizer(quant_config)
            quantization_config = self.quantizer.get_quantization_config()
        else:
            quantization_config = None
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 加载模型
        logger.info("加载模型...")
        start_time = time.time()
        
        if self.device == "cuda":
            # 配置显存优化
            memory_config = MemoryConfig(
                offload_to_cpu=offload,
                max_memory_per_gpu=self.config.hardware.gpu_memory_gb - 4
            )
            self.memory_optimizer = MemoryOptimizer(memory_config)
            
            # 加载模型
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 配置推理优化
        inference_config = InferenceConfig(
            use_kv_cache=True,
            use_flash_attention=True
        )
        self.inference_optimizer = InferenceOptimizer(inference_config)
        self.model = self.inference_optimizer.optimize_model(self.model)
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型参数总量: {total_params / 1e9:.2f}B")
        logger.info(f"可训练参数: {trainable_params / 1e9:.2f}B")
        
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU显存已分配: {allocated:.2f} GB")
            logger.info(f"GPU显存已预留: {reserved:.2f} GB")
    
    def load_dataset(self):
        """加载数据集"""
        from datasets import load_dataset
        
        logger.info(f"加载数据集: {self.config.data.dataset_name}")
        
        data_path = self.config.data.data_path
        if os.path.exists(data_path):
            dataset = load_dataset(
                self.config.data.dataset_name,
                self.config.data.dataset_config,
                data_dir=data_path,
                split="test"
            )
        else:
            dataset = load_dataset(
                self.config.data.dataset_name,
                self.config.data.dataset_config,
                split="test"
            )
        
        logger.info(f"数据集大小: {len(dataset)} 条")
        return dataset
    
    def preprocess_data(self, dataset) -> List[str]:
        """预处理数据"""
        logger.info("预处理数据...")
        
        texts = []
        for item in tqdm(dataset, desc="预处理"):
            text = item["text"].strip()
            if len(text) > 50:  # 过滤太短的文本
                texts.append(text)
        
        # 限制样本数量
        max_samples = self.config.data.max_samples
        if max_samples > 0 and len(texts) > max_samples:
            texts = texts[:max_samples]
        
        logger.info(f"有效文本数量: {len(texts)}")
        return texts
    
    def measure_latency(self, texts: List[str]) -> Dict:
        """测量推理延迟"""
        logger.info("测量推理延迟...")
        
        latencies = []
        memory_peak = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for text in tqdm(texts, desc="推理"):
                # 清空缓存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.data.max_length
                )
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 计时
                start_time = time.time()
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
                
                # 记录显存峰值
                if self.device == "cuda":
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    memory_peak = max(memory_peak, peak)
        
        # 计算统计信息
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        results = {
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "total_samples": len(texts),
            "memory_peak_gb": memory_peak
        }
        
        logger.info(f"平均延迟: {avg_latency:.4f}秒")
        logger.info(f"最小延迟: {min_latency:.4f}秒")
        logger.info(f"最大延迟: {max_latency:.4f}秒")
        logger.info(f"显存峰值: {memory_peak:.2f} GB")
        
        return results
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """计算困惑度"""
        logger.info("计算困惑度...")
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="计算困惑度"):
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.data.max_length
                )
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(
                    **inputs,
                    labels=inputs["input_ids"]
                )
                
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        # 计算困惑度
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"困惑度: {perplexity:.4f}")
        
        return perplexity
    
    def run_test(self, quantization: str = "int8", offload: bool = False) -> Dict:
        """运行完整测试"""
        logger.info("=" * 50)
        logger.info("开始优化测试")
        logger.info("=" * 50)
        
        # 初始化
        self.setup()
        
        # 加载模型
        self.load_model(quantization=quantization, offload=offload)
        
        # 加载数据集
        dataset = self.load_dataset()
        texts = self.preprocess_data(dataset)
        
        # 测量延迟
        latency_results = self.measure_latency(texts)
        
        # 计算困惑度
        perplexity = self.calculate_perplexity(texts)
        
        # 汇总结果
        results = {
            "timestamp": datetime.now().isoformat(),
            "optimization": {
                "quantization": quantization,
                "offload": offload
            },
            "config": {
                "model": self.config.model.__dict__,
                "data": self.config.data.__dict__,
                "optimization": self.config.optimization.__dict__,
                "hardware": self.config.hardware.__dict__
            },
            "results": {
                **latency_results,
                "perplexity": perplexity
            }
        }
        
        # 保存结果
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict, filename: str = None):
        """保存测试结果"""
        if filename is None:
            filename = f"optimized_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {filepath}")


def compare_with_baseline(baseline_file: str, optimized_file: str):
    """与基线对比"""
    # 加载结果
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    with open(optimized_file, 'r') as f:
        optimized = json.load(f)
    
    baseline_results = baseline["results"]
    optimized_results = optimized["results"]
    
    # 计算改进
    latency_improvement = (baseline_results["avg_latency"] - optimized_results["avg_latency"]) / baseline_results["avg_latency"] * 100
    memory_improvement = (baseline_results["memory_peak_gb"] - optimized_results["memory_peak_gb"]) / baseline_results["memory_peak_gb"] * 100
    ppl_change = (optimized_results["perplexity"] - baseline_results["perplexity"]) / baseline_results["perplexity"] * 100
    
    comparison = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "improvement": {
            "latency_percent": latency_improvement,
            "memory_percent": memory_improvement,
            "perplexity_change_percent": ppl_change
        }
    }
    
    # 打印对比
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    print(f"{'指标':<20} {'基线':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 60)
    print(f"{'平均延迟(秒)':<20} {baseline_results['avg_latency']:<15.4f} {optimized_results['avg_latency']:<15.4f} {latency_improvement:<14.2f}%")
    print(f"{'显存峰值(GB)':<20} {baseline_results['memory_peak_gb']:<15.2f} {optimized_results['memory_peak_gb']:<15.2f} {memory_improvement:<14.2f}%")
    print(f"{'困惑度':<20} {baseline_results['perplexity']:<15.4f} {optimized_results['perplexity']:<15.4f} {ppl_change:<14.2f}%")
    print("=" * 60)
    
    # 检查是否满足要求
    if ppl_change > 15:
        print("⚠️ 警告: 困惑度上升超过15%，不满足比赛要求！")
    else:
        print("✅ 困惑度上升在15%以内，满足比赛要求")
    
    return comparison


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="优化测试")
    parser.add_argument("--quantization", type=str, default="int8",
                       choices=["none", "int8", "int4"],
                       help="量化方法")
    parser.add_argument("--offload", action="store_true",
                       help="启用CPU offload")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="测试样本数量")
    parser.add_argument("--baseline", type=str, default=None,
                       help="基线结果文件路径，用于对比")
    
    args = parser.parse_args()
    
    # 更新配置
    default_config.data.max_samples = args.max_samples
    
    # 创建测试器
    tester = OptimizedTester()
    
    # 运行测试
    results = tester.run_test(
        quantization=args.quantization,
        offload=args.offload
    )
    
    # 打印结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要")
    print("=" * 50)
    print(f"量化方法: {args.quantization}")
    print(f"CPU Offload: {args.offload}")
    print(f"平均单句推理延迟: {results['results']['avg_latency']:.4f} 秒")
    print(f"显存峰值: {results['results']['memory_peak_gb']:.2f} GB")
    print(f"困惑度: {results['results']['perplexity']:.4f}")
    print("=" * 50)
    
    # 与基线对比
    if args.baseline:
        compare_with_baseline(args.baseline, results.get("result_file", ""))


if __name__ == "__main__":
    main()
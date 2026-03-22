#!/usr/bin/env python3
"""
比赛运行脚本 - MoE语言模型端到端效率优化

此脚本是比赛提交的主要入口点，用于：
1. 加载Mixtral-8x7B模型
2. 应用优化策略（INT8量化、显存优化、推理优化）
3. 在wikitext-103数据集上测试
4. 输出性能指标

使用方法:
    python scripts/run_competition.py --model_path /path/to/model --output results.json
    python scripts/run_competition.py --help
"""

import os
import sys
import argparse
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompetitionRunner:
    """比赛运行器"""
    
    def __init__(
        self,
        model_path: str,
        quantization: str = "int8",
        offload: bool = False,
        max_new_tokens: int = 100,
        batch_size: int = 1,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.offload = offload
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device
        
        self.model = None
        self.tokenizer = None
        
    def setup(self):
        """初始化环境"""
        logger.info("=" * 60)
        logger.info("初始化比赛环境")
        logger.info("=" * 60)
        
        # 检查CUDA
        if torch.cuda.is_available():
            logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}")
                logger.info(f"  显存: {props.total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA不可用，将使用CPU运行")
            self.device = "cpu"
        
        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    
    def load_model(self):
        """加载模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        logger.info("=" * 60)
        logger.info("加载模型")
        logger.info("=" * 60)
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"量化方法: {self.quantization}")
        logger.info(f"CPU Offload: {self.offload}")
        
        start_time = time.time()
        
        # 配置量化
        quantization_config = None
        if self.quantization == "int8" and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            logger.info("使用INT8量化")
        elif self.quantization == "int4" and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("使用INT4量化")
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置设备映射
        if self.device == "auto":
            device_map = "auto"
        else:
            device_map = None
        
        # 加载模型
        logger.info("加载模型...")
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = device_map
        elif device_map == "auto":
            load_kwargs["device_map"] = device_map
        
        if self.offload:
            load_kwargs["offload_folder"] = "offload"
            load_kwargs["offload_state_dict"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        if self.device == "cuda" and device_map is None:
            self.model = self.model.cuda()
        
        # 设置为评估模式
        self.model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型参数量: {total_params / 1e9:.2f}B")
        logger.info(f"可训练参数量: {trainable_params / 1e9:.2f}B")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"已分配显存: {allocated:.2f} GB")
            logger.info(f"已预留显存: {reserved:.2f} GB")
    
    def load_dataset(self, dataset_name: str = "wikitext", dataset_config: str = "wikitext-103-v1"):
        """加载数据集"""
        from datasets import load_dataset
        
        logger.info("=" * 60)
        logger.info("加载数据集")
        logger.info("=" * 60)
        logger.info(f"数据集: {dataset_name}, 配置: {dataset_config}")
        
        try:
            dataset = load_dataset(dataset_name, dataset_config, split="test")
            logger.info(f"数据集大小: {len(dataset)}")
            return dataset
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def measure_latency(self, num_samples: int = 100) -> Dict:
        """测量推理延迟"""
        logger.info("=" * 60)
        logger.info("测量推理延迟")
        logger.info("=" * 60)
        
        test_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was nothing but darkness.",
            "The history of artificial intelligence dates back to",
            "One of the most important discoveries in science was",
            "The development of modern technology has led to"
        ]
        
        latencies = []
        
        # 预热
        logger.info("预热模型...")
        warmup_input = self.tokenizer("Hello, world!", return_tensors="pt")
        if torch.cuda.is_available():
            warmup_input = {k: v.cuda() for k, v in warmup_input.items()}
        with torch.no_grad():
            _ = self.model.generate(**warmup_input, max_new_tokens=10)
        
        # 测量延迟
        logger.info(f"测量 {num_samples} 个样本的推理延迟...")
        for i in tqdm(range(num_samples), desc="测量延迟"):
            prompt = test_prompts[i % len(test_prompts)]
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        # 计算统计信息
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        results = {
            "num_samples": num_samples,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "total_time": sum(latencies)
        }
        
        logger.info(f"平均延迟: {avg_latency:.4f}秒")
        logger.info(f"最小延迟: {min_latency:.4f}秒")
        logger.info(f"最大延迟: {max_latency:.4f}秒")
        
        return results
    
    def measure_memory(self) -> Dict:
        """测量显存使用"""
        logger.info("=" * 60)
        logger.info("测量显存使用")
        logger.info("=" * 60)
        
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，无法测量显存")
            return {"memory_allocated_gb": 0, "memory_reserved_gb": 0, "memory_peak_gb": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        
        results = {
            "memory_allocated_gb": round(allocated, 4),
            "memory_reserved_gb": round(reserved, 4),
            "memory_peak_gb": round(peak, 4)
        }
        
        logger.info(f"已分配显存: {allocated:.4f} GB")
        logger.info(f"已预留显存: {reserved:.4f} GB")
        logger.info(f"峰值显存: {peak:.4f} GB")
        
        return results
    
    def calculate_perplexity(self, dataset, num_samples: int = None) -> float:
        """计算困惑度"""
        logger.info("=" * 60)
        logger.info("计算困惑度")
        logger.info("=" * 60)
        
        # 获取测试文本
        if num_samples:
            texts = dataset["text"][:num_samples]
        else:
            texts = dataset["text"]
        
        # 过滤空文本
        texts = [t for t in texts if t.strip()]
        
        # 编码
        logger.info("编码文本...")
        encodings = self.tokenizer(
            "\n\n".join(texts),
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            encodings = {k: v.cuda() for k, v in encodings.items()}
        
        # 计算困惑度
        logger.info("计算困惑度...")
        max_length = 2048
        stride = 512
        seq_len = encodings["input_ids"].size(1)
        
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in tqdm(range(0, seq_len, stride), desc="计算困惑度"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_ids = encodings["input_ids"][:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        
        logger.info(f"困惑度: {ppl.item():.4f}")
        
        return ppl.item()
    
    def run(self, num_samples: int = 100) -> Dict:
        """运行完整测试"""
        logger.info("=" * 60)
        logger.info("开始比赛测试")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 初始化
        self.setup()
        
        # 加载模型
        self.load_model()
        
        # 加载数据集
        dataset = self.load_dataset()
        
        # 测量延迟
        latency_results = self.measure_latency(num_samples=num_samples)
        
        # 测量显存
        memory_results = self.measure_memory()
        
        # 计算困惑度
        perplexity = self.calculate_perplexity(dataset, num_samples=100)
        
        total_time = time.time() - start_time
        
        # 汇总结果
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "quantization": self.quantization,
            "offload": self.offload,
            "device": self.device,
            "results": {
                "avg_latency": latency_results["avg_latency"],
                "min_latency": latency_results["min_latency"],
                "max_latency": latency_results["max_latency"],
                "memory_peak_gb": memory_results["memory_peak_gb"],
                "perplexity": perplexity
            },
            "total_time_seconds": total_time
        }
        
        # 打印结果
        logger.info("=" * 60)
        logger.info("测试结果汇总")
        logger.info("=" * 60)
        logger.info(f"平均推理延迟: {latency_results['avg_latency']:.4f}秒")
        logger.info(f"峰值显存: {memory_results['memory_peak_gb']:.4f} GB")
        logger.info(f"困惑度: {perplexity:.4f}")
        logger.info(f"总耗时: {total_time:.2f}秒")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="MoE语言模型端到端效率优化比赛脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="模型路径或HuggingFace模型名称"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "int8", "int4"],
        default="int8",
        help="量化方法 (none, int8, int4)"
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="是否启用CPU offload"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="测试样本数量"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="生成的最大token数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="结果输出文件"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="运行设备"
    )
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = CompetitionRunner(
        model_path=args.model_path,
        quantization=args.quantization,
        offload=args.offload,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    # 运行测试
    results = runner.run(num_samples=args.num_samples)
    
    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存到: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
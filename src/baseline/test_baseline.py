"""
基线测试脚本 - MoE语言模型端到端效率优化

功能：
1. 加载Mixtral-8x7B模型（从HuggingFace直接加载）
2. 在wikitext-103-v1数据集上测试
3. 测量推理延迟、显存占用、困惑度
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config, setup_environment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineTester:
    """基线测试类"""
    
    def __init__(self, config=None):
        self.config = config or default_config
        self.model = None
        self.tokenizer = None
        self.device = self.config.hardware.device
        
    def setup(self):
        """初始化环境"""
        logger.info("初始化测试环境...")
        setup_environment()
        
        # 检查CUDA可用性
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            else:
                logger.warning("CUDA不可用，将使用CPU运行")
                self.device = "cpu"
        except ImportError:
            logger.error("PyTorch未安装")
            raise
    
    def load_model(self, use_quantization: bool = False):
        """加载模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        model_name = self.config.model.model_name
        logger.info(f"加载模型: {model_name}")
        logger.info("从HuggingFace直接加载（不下载到本地）...")
        
        # 量化配置
        quantization_config = None
        if use_quantization and self.device == "cuda":
            logger.info("使用INT8量化")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 加载模型
        logger.info("加载模型（这可能需要几分钟）...")
        start_time = time.time()
        
        if self.device == "cuda":
            # 多GPU设置
            import torch
            if torch.cuda.device_count() > 1:
                logger.info(f"使用 {torch.cuda.device_count()} 个GPU")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型参数总量: {total_params / 1e9:.2f}B")
        logger.info(f"可训练参数: {trainable_params / 1e9:.2f}B")
        
        if self.device == "cuda":
            import torch
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU显存已分配: {allocated:.2f} GB")
            logger.info(f"GPU显存已预留: {reserved:.2f} GB")
    
    def load_dataset(self):
        """加载数据集"""
        from datasets import load_dataset
        
        logger.info(f"加载数据集: {self.config.data.dataset_name}")
        
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
        from tqdm import tqdm
        
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
        import torch
        from tqdm import tqdm
        
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
                    pad_token_id=self.tokenizer.eos_token_id
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
        import torch
        from tqdm import tqdm
        
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
    
    def run_test(self, use_quantization: bool = False) -> Dict:
        """运行完整测试"""
        logger.info("=" * 50)
        logger.info("开始基线测试")
        logger.info("=" * 50)
        
        # 初始化
        self.setup()
        
        # 加载模型
        self.load_model(use_quantization=use_quantization)
        
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
            filename = f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {filepath}")


def main():
    """主函数"""
    # 创建测试器
    tester = BaselineTester()
    
    # 运行测试
    results = tester.run_test(use_quantization=False)
    
    # 打印结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要")
    print("=" * 50)
    print(f"平均单句推理延迟: {results['results']['avg_latency']:.4f} 秒")
    print(f"显存峰值: {results['results']['memory_peak_gb']:.2f} GB")
    print(f"困惑度: {results['results']['perplexity']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
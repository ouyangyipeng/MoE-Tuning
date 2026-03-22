#!/usr/bin/env python3
"""
主运行脚本 - MoE语言模型端到端效率优化

功能：
1. 运行基线测试
2. 运行优化测试
3. 对比性能
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import default_config, setup_environment
from src.utils.helpers import setup_logging, save_json, get_timestamp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_baseline_test(args):
    """运行基线测试"""
    from src.baseline.test_baseline import BaselineTester
    
    logger.info("=" * 60)
    logger.info("运行基线测试")
    logger.info("=" * 60)
    
    tester = BaselineTester()
    results = tester.run_test(use_quantization=False)
    
    return results


def run_optimized_test(args):
    """运行优化测试"""
    from src.baseline.test_baseline import BaselineTester
    from src.optimization.quantization import Quantizer, QuantizationConfig
    from src.optimization.memory import MemoryOptimizer, MemoryConfig
    
    logger.info("=" * 60)
    logger.info("运行优化测试")
    logger.info("=" * 60)
    
    # 配置优化
    quant_config = QuantizationConfig(
        method=args.quantization
    )
    
    memory_config = MemoryConfig(
        offload_to_cpu=args.offload,
        max_memory_per_gpu=args.max_memory
    )
    
    # 运行测试
    tester = BaselineTester()
    results = tester.run_test(use_quantization=(args.quantization != "none"))
    
    return results


def compare_results(baseline: dict, optimized: dict):
    """对比结果"""
    logger.info("=" * 60)
    logger.info("性能对比")
    logger.info("=" * 60)
    
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
    parser = argparse.ArgumentParser(description="MoE语言模型端到端效率优化")
    parser.add_argument("--mode", type=str, default="baseline",
                       choices=["baseline", "optimized", "compare"],
                       help="运行模式")
    parser.add_argument("--quantization", type=str, default="int8",
                       choices=["none", "int8", "int4"],
                       help="量化方法")
    parser.add_argument("--offload", action="store_true",
                       help="启用CPU offload")
    parser.add_argument("--max-memory", type=int, default=60,
                       help="每个GPU最大显存(GB)")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="测试样本数量")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 更新配置
    default_config.data.max_samples = args.max_samples
    
    results = None
    
    if args.mode == "baseline":
        results = run_baseline_test(args)
    elif args.mode == "optimized":
        results = run_optimized_test(args)
    elif args.mode == "compare":
        # 运行基线测试
        baseline = run_baseline_test(args)
        
        # 运行优化测试
        optimized = run_optimized_test(args)
        
        # 对比结果
        results = compare_results(baseline, optimized)
    
    # 保存结果
    if results and args.output:
        save_json(results, args.output)
        logger.info(f"结果已保存到: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
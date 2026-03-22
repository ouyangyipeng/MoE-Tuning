"""
MoE特定优化模块 - MoE语言模型端到端效率优化

针对Mixtral-8x7B的专家层优化：
1. 专家路由优化
2. 专家缓存策略
3. 专家预加载
4. 负载均衡
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """MoE配置"""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_hidden_size: int = 14336
    expert_intermediate_size: int = 14336
    enable_expert_cache: bool = True
    cache_size: int = 4  # 缓存的专家数量
    preload_strategy: str = "lru"  # lru, lf, fifo


class ExpertCache:
    """专家缓存管理器"""
    
    def __init__(self, cache_size: int = 4, strategy: str = "lru"):
        self.cache_size = cache_size
        self.strategy = strategy
        self.cache: Dict[int, torch.Tensor] = {}
        self.access_count: Dict[int, int] = defaultdict(int)
        self.last_access: Dict[int, int] = {}
        self.current_time = 0
    
    def get(self, expert_id: int) -> Optional[torch.Tensor]:
        """从缓存获取专家"""
        if expert_id in self.cache:
            self.access_count[expert_id] += 1
            self.last_access[expert_id] = self.current_time
            self.current_time += 1
            return self.cache[expert_id]
        return None
    
    def put(self, expert_id: int, expert_weights: torch.Tensor):
        """将专家放入缓存"""
        if expert_id in self.cache:
            return
        
        if len(self.cache) >= self.cache_size:
            # 需要淘汰
            self._evict()
        
        self.cache[expert_id] = expert_weights
        self.access_count[expert_id] = 1
        self.last_access[expert_id] = self.current_time
        self.current_time += 1
    
    def _evict(self):
        """淘汰策略"""
        if not self.cache:
            return
        
        if self.strategy == "lru":
            # 最近最少使用
            victim = min(self.last_access, key=self.last_access.get)
        elif self.strategy == "lf":
            # 最不频繁使用
            victim = min(self.access_count, key=self.access_count.get)
        else:  # fifo
            victim = list(self.cache.keys())[0]
        
        del self.cache[victim]
        del self.access_count[victim]
        del self.last_access[victim]
        
        logger.debug(f"淘汰专家 {victim}")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.last_access.clear()
        self.current_time = 0
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "total_accesses": sum(self.access_count.values()),
            "expert_access_counts": dict(self.access_count)
        }


class ExpertPreloader:
    """专家预加载器"""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.expert_access_history: List[List[int]] = []
        self.transition_matrix: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    def record_access(self, expert_ids: List[int]):
        """记录专家访问"""
        self.expert_access_history.append(expert_ids)
        
        # 更新转移矩阵
        if len(self.expert_access_history) > 1:
            prev_experts = self.expert_access_history[-2]
            for prev in prev_experts:
                for curr in expert_ids:
                    self.transition_matrix[prev][curr] += 1
    
    def predict_next_experts(self, current_experts: List[int], top_k: int = 2) -> List[int]:
        """预测下一个可能被访问的专家"""
        predictions = defaultdict(float)
        
        for expert in current_experts:
            if expert in self.transition_matrix:
                total = sum(self.transition_matrix[expert].values())
                for next_expert, count in self.transition_matrix[expert].items():
                    predictions[next_expert] += count / total
        
        # 排序并返回top_k
        sorted_predictions = sorted(predictions.items(), key=lambda x: -x[1])
        return [p[0] for p in sorted_predictions[:top_k]]
    
    def get_preload_suggestion(self, current_experts: List[int]) -> List[int]:
        """获取预加载建议"""
        return self.predict_next_experts(current_experts, self.config.num_experts_per_tok)


class ExpertLoadBalancer:
    """专家负载均衡器"""
    
    def __init__(self, num_experts: int = 8):
        self.num_experts = num_experts
        self.expert_load: Dict[int, float] = defaultdict(float)
        self.expert_capacity: Dict[int, float] = defaultdict(lambda: 1.0)
    
    def update_load(self, expert_id: int, load: float):
        """更新专家负载"""
        self.expert_load[expert_id] = load
    
    def get_load_balance_score(self) -> float:
        """获取负载均衡分数 (0-1, 1表示完全均衡)"""
        if not self.expert_load:
            return 1.0
        
        loads = list(self.expert_load.values())
        avg_load = sum(loads) / len(loads)
        
        if avg_load == 0:
            return 1.0
        
        # 计算变异系数
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
        cv = (variance ** 0.5) / avg_load
        
        # 转换为0-1分数
        return max(0, 1 - cv)
    
    def get_overloaded_experts(self, threshold: float = 1.5) -> List[int]:
        """获取过载的专家"""
        if not self.expert_load:
            return []
        
        avg_load = sum(self.expert_load.values()) / len(self.expert_load)
        return [e for e, l in self.expert_load.items() if l > avg_load * threshold]
    
    def get_underloaded_experts(self, threshold: float = 0.5) -> List[int]:
        """获取欠载的专家"""
        if not self.expert_load:
            return []
        
        avg_load = sum(self.expert_load.values()) / len(self.expert_load)
        return [e for e, l in self.expert_load.items() if l < avg_load * threshold]
    
    def reset(self):
        """重置负载统计"""
        self.expert_load.clear()


class MoEOptimizer:
    """MoE优化器 - 统一接口"""
    
    def __init__(self, config: MoEConfig = None):
        self.config = config or MoEConfig()
        self.expert_cache = ExpertCache(
            cache_size=self.config.cache_size,
            strategy=self.config.preload_strategy
        )
        self.preloader = ExpertPreloader(config)
        self.load_balancer = ExpertLoadBalancer(self.config.num_experts)
    
    def optimize_routing(self, router_logits: torch.Tensor) -> torch.Tensor:
        """优化路由决策"""
        # 可以添加路由优化逻辑
        # 例如：负载感知路由
        return router_logits
    
    def get_expert_from_cache(self, expert_id: int) -> Optional[torch.Tensor]:
        """从缓存获取专家"""
        return self.expert_cache.get(expert_id)
    
    def put_expert_to_cache(self, expert_id: int, weights: torch.Tensor):
        """将专家放入缓存"""
        self.expert_cache.put(expert_id, weights)
    
    def record_expert_access(self, expert_ids: List[int]):
        """记录专家访问"""
        self.preloader.record_access(expert_ids)
        
        # 更新负载
        for expert_id in expert_ids:
            self.load_balancer.update_load(expert_id, 1.0)
    
    def get_preload_suggestion(self, current_experts: List[int]) -> List[int]:
        """获取预加载建议"""
        return self.preloader.get_preload_suggestion(current_experts)
    
    def get_optimization_stats(self) -> Dict:
        """获取优化统计"""
        return {
            "cache_stats": self.expert_cache.get_stats(),
            "load_balance_score": self.load_balancer.get_load_balance_score(),
            "overloaded_experts": self.load_balancer.get_overloaded_experts(),
            "underloaded_experts": self.load_balancer.get_underloaded_experts()
        }
    
    def reset(self):
        """重置所有状态"""
        self.expert_cache.clear()
        self.load_balancer.reset()
        self.preloader.expert_access_history.clear()
        self.preloader.transition_matrix.clear()


def analyze_expert_distribution(model) -> Dict:
    """分析模型的专家分布"""
    stats = {
        "total_experts": 0,
        "expert_layers": [],
        "expert_params": {}
    }
    
    for name, module in model.named_modules():
        # 检测专家层
        if "block_sparse_moe" in name.lower() or "experts" in name.lower():
            stats["expert_layers"].append(name)
            
            # 统计参数
            if hasattr(module, "num_experts"):
                stats["total_experts"] = module.num_experts
            
            if hasattr(module, "experts"):
                for i, expert in enumerate(module.experts):
                    if hasattr(expert, "parameters"):
                        params = sum(p.numel() for p in expert.parameters())
                        stats["expert_params"][f"expert_{i}"] = params
    
    return stats


def get_expert_utilization(model, input_ids: torch.Tensor) -> Dict:
    """获取专家利用率"""
    # 这需要hook来捕获路由决策
    utilization = {
        "expert_counts": defaultdict(int),
        "total_tokens": 0
    }
    
    # 简化实现：返回空统计
    return utilization


if __name__ == "__main__":
    # 测试MoE优化器
    config = MoEConfig()
    optimizer = MoEOptimizer(config)
    
    # 模拟专家访问
    for i in range(10):
        experts = [i % 8, (i + 1) % 8]
        optimizer.record_expert_access(experts)
    
    # 获取统计
    stats = optimizer.get_optimization_stats()
    print(f"优化统计: {stats}")
    
    # 测试预加载建议
    suggestion = optimizer.get_preload_suggestion([0, 1])
    print(f"预加载建议: {suggestion}")
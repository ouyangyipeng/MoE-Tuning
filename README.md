# MoE语言模型端到端效率优化

## 项目概述

本项目为2025年全国大学生计算机系统能力大赛 - 智能计算创新设计赛（先导杯）MoE语言模型端到端效率优化赛题的解决方案。

**目标模型**: Mixtral-8x7B-v0.1

**目标平台**: K100-AI国产加速卡 (2张, 单卡64GB显存)

### 比赛要求

1. **推理延迟**: 优化平均单句推理延迟
2. **显存占用**: 优化显存使用，确保在128GB显存内运行
3. **困惑度**: 困惑度上升不超过15%

## 目录结构

```
moe/
├── README.md              # 项目说明文档
├── PROGRESS.md            # 进度记录文档
├── requirements.txt       # 依赖列表
├── run.py                 # 主运行脚本
├── download_model.py      # 模型下载脚本
├── test_small_model.py    # 小模型测试脚本
├── plans/                 # 计划文档
│   └── implementation_plan.md
├── src/                   # 源代码
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   ├── baseline/          # 基线测试代码
│   │   ├── __init__.py
│   │   └── test_baseline.py
│   ├── optimization/      # 优化代码
│   │   ├── __init__.py
│   │   ├── quantization.py   # 量化优化
│   │   ├── memory.py         # 显存优化
│   │   ├── inference.py      # 推理优化
│   │   ├── moe_optimization.py # MoE特定优化
│   │   └── optimized_test.py  # 优化测试
│   └── utils/             # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── scripts/               # 脚本目录
│   ├── check_env.py       # 环境检查脚本
│   ├── run_competition.py # 比赛运行脚本
│   └── run_all.sh         # 一键运行脚本
├── docs/                  # 文档目录
│   ├── optimization_report.md # 优化报告
│   └── project_summary.md     # 项目总结
├── models/                # 模型存储目录
├── data/                  # 数据集存储目录
└── results/               # 测试结果目录
```

## 环境配置

### 硬件要求
- GPU: K100-AI国产加速卡 (2张, 单卡64GB显存) 或 NVIDIA GPU (支持CUDA)
- 内存: 128GB以上
- 存储: 200GB以上 (用于存储模型)

### 软件要求
- Python 3.10
- PyTorch 2.1.0 (dtk24.04) 或 PyTorch 2.1.0+cu121
- Transformers 4.47.1

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 环境检查

```bash
python scripts/check_env.py
```

### 2. 小模型测试（验证代码逻辑）

```bash
python test_small_model.py
```

### 3. 比赛运行脚本

```bash
# 使用INT8量化
python scripts/run_competition.py --model_path mistralai/Mixtral-8x7B-v0.1 --quantization int8 --output results.json

# 使用INT4量化
python scripts/run_competition.py --model_path mistralai/Mixtral-8x7B-v0.1 --quantization int4 --output results.json

# 启用CPU Offload
python scripts/run_competition.py --model_path mistralai/Mixtral-8x7B-v0.1 --quantization int8 --offload --output results.json
```

### 4. 主运行脚本

```bash
# 运行基线测试
python run.py --mode baseline --max-samples 100

# 运行优化测试 (INT8量化)
python run.py --mode optimized --quantization int8 --max-samples 100

# 运行优化测试 (INT4量化)
python run.py --mode optimized --quantization int4 --max-samples 100

# 对比测试
python run.py --mode compare --quantization int8 --max-samples 100
```

### 5. 下载模型和数据集

```bash
# 下载模型
python download_model.py --download-model

# 下载数据集
python download_model.py --download-dataset

# 下载全部
python download_model.py --download-all
```

## 优化策略

### 1. 量化优化
- **INT8量化**: 使用bitsandbytes进行8位量化，减少显存占用约50%
- **INT4量化**: 使用NF4量化，进一步减少显存占用约75%
- **混合精度**: 关键层保持FP16，其他层使用低精度

### 2. 显存优化
- **模型分片**: 将模型分布到多个GPU上
- **CPU Offload**: 将部分权重卸载到CPU内存
- **KV Cache优化**: 优化KV Cache的存储和访问
- **梯度检查点**: 减少显存占用

### 3. 推理优化
- **KV Cache**: 使用KV Cache加速自回归生成
- **Flash Attention**: 使用Flash Attention加速注意力计算
- **批处理优化**: 优化批处理推理

### 4. MoE特定优化
- **专家缓存**: 缓存热门专家，减少加载时间
- **专家预加载**: 预测并预加载可能需要的专家
- **负载均衡**: 平衡专家使用，避免热点

## 预期效果

| 指标 | 基线 | INT8量化 | INT4量化 |
|------|------|----------|----------|
| 显存占用 | ~90GB | ~45GB | ~25GB |
| 推理延迟 | 基准 | 降低10-20% | 降低15-30% |
| 困惑度变化 | 0% | <1% | <5% |

## 代码说明

### 核心模块

#### 1. 配置模块 (`src/config.py`)
- 定义模型、数据、优化、硬件配置
- 提供默认配置和环境设置

#### 2. 量化模块 (`src/optimization/quantization.py`)
- 支持INT8和INT4量化
- 使用bitsandbytes库

#### 3. 显存优化模块 (`src/optimization/memory.py`)
- 模型分片
- CPU Offload
- KV Cache优化

#### 4. 推理优化模块 (`src/optimization/inference.py`)
- KV Cache管理
- Flash Attention
- 批处理优化

#### 5. MoE优化模块 (`src/optimization/moe_optimization.py`)
- 专家缓存
- 专家预加载
- 负载均衡

### 测试模块

#### 1. 基线测试 (`src/baseline/test_baseline.py`)
- 加载原始模型
- 测量基线性能

#### 2. 优化测试 (`src/optimization/optimized_test.py`)
- 加载优化模型
- 测量优化后性能

## 比赛提交说明

### 提交内容
1. 完整源代码
2. 运行脚本 (`scripts/run_competition.py`)
3. 技术文档 (`docs/optimization_report.md`)
4. README文档

### 运行环境
比赛环境将提供：
- K100-AI国产加速卡 (2张, 单卡64GB显存)
- 预装的模型文件

### 评分标准
1. **推理延迟** (40%): 平均单句推理延迟
2. **显存占用** (30%): 峰值显存使用
3. **困惑度** (30%): 困惑度上升不超过15%

## 常见问题

### Q1: 显存不足怎么办？
- 使用INT4量化
- 启用CPU Offload
- 减小批处理大小

### Q2: 如何在本地测试？
- 使用`test_small_model.py`测试代码逻辑
- 使用小模型（如GPT-2）验证功能

### Q3: 如何查看显存使用情况？
```python
import torch
print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"已预留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 开发团队

- 项目: MoE语言模型端到端效率优化
- 比赛: 2025年全国大学生计算机系统能力大赛

## 许可证

MIT License
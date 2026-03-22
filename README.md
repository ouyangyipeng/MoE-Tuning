# MoE语言模型端到端效率优化

## 项目概述

本项目为2025年全国大学生计算机系统能力大赛 - 智能计算创新设计赛（先导杯）MoE语言模型端到端效率优化赛题的解决方案。

**目标模型**: Mixtral-8x7B-v0.1

**目标平台**: K100-AI国产加速卡 (2张, 单卡64GB显存)

## 目录结构

```
moe/
├── PROGRESS.md           # 进度记录文档
├── plans/                # 计划文档
│   └── implementation_plan.md
├── src/                  # 源代码
│   ├── __init__.py
│   ├── config.py         # 配置文件
│   ├── baseline/         # 基线测试代码
│   │   ├── __init__.py
│   │   └── test_baseline.py
│   ├── optimization/     # 优化代码
│   │   ├── __init__.py
│   │   ├── quantization.py  # 量化优化
│   │   ├── memory.py        # 显存优化
│   │   └── inference.py     # 推理优化
│   └── utils/            # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── models/               # 模型存储目录
├── data/                 # 数据集存储目录
├── results/              # 测试结果目录
├── download_model.py     # 模型下载脚本
├── run.py                # 主运行脚本
└── requirements.txt      # 依赖列表
```

## 环境配置

### 软件要求
- Python 3.10
- PyTorch 2.1.0 (dtk24.04)
- Transformers 4.47.1

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 下载模型和数据集

```bash
# 下载模型
python download_model.py --download-model

# 下载数据集
python download_model.py --download-dataset

# 下载全部
python download_model.py --download-all
```

### 2. 运行基线测试

```bash
python run.py --mode baseline --max-samples 100
```

### 3. 运行优化测试

```bash
# INT8量化
python run.py --mode optimized --quantization int8 --max-samples 100

# INT4量化
python run.py --mode optimized --quantization int4 --max-samples 100

# 启用CPU Offload
python run.py --mode optimized --quantization int8 --offload --max-samples 100
```

### 4. 对比测试结果

```bash
python run.py --mode compare --quantization int8 --max-samples 100
```

## 优化策略

### 1. 量化优化
- **INT8量化**: 使用bitsandbytes进行8位量化，减少显存占用
- **INT4量化**: 使用NF4量化，进一步减少显存占用
- **混合精度**: 关键层保持FP16，其他层使用低精度

### 2. 显存优化
- **模型分片**: 将模型分布到多张GPU
- **CPU Offloading**: 将部分参数卸载到CPU
- **KV Cache优化**: 优化注意力缓存策略

### 3. 推理优化
- **Flash Attention**: 使用Flash Attention加速注意力计算
- **KV Cache**: 使用past_key_values加速生成
- **批处理优化**: 动态批处理策略

## 评分标准

### 初赛 (100分)
- 平均单句推理延迟
- 困惑度上升不超过15%

### 决赛 (100分)
| 项目 | 分值 |
|------|------|
| 推理效率 | 30分 |
| 显存占用 | 25分 |
| 困惑度保持 | 25分 |
| 创新性 | 20分 |

## 注意事项

1. 不允许使用vLLM等外部推理框架
2. 不能跳层或改变专家数量
3. 困惑度上升必须控制在15%以内

## 参考资源

- 模型下载: https://hf-mirror.com/mistralai/Mixtral-8x7B-v0.1
- 数据集下载: https://hf-mirror.com/datasets/Salesforce/wikitext
- 大赛官网: https://pra.educg.net

## 许可证

本项目仅用于比赛目的。
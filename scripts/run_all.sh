#!/bin/bash
# MoE语言模型端到端效率优化 - 一键运行脚本

set -e

echo "=========================================="
echo "MoE语言模型端到端效率优化"
echo "=========================================="

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1

# 创建必要的目录
mkdir -p models data results logs

# 检查Python版本
echo "检查Python版本..."
python --version

# 检查依赖
echo "检查依赖..."
python scripts/check_env.py

# 如果检查失败，先安装依赖
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 下载模型（如果不存在）
if [ ! -d "models/Mixtral-8x7B-v0.1" ]; then
    echo "下载Mixtral-8x7B模型..."
    python download_model.py --download-model
else
    echo "模型已存在，跳过下载"
fi

# 下载数据集（如果不存在）
if [ ! -d "data/wikitext" ]; then
    echo "下载wikitext数据集..."
    python download_model.py --download-dataset
else
    echo "数据集已存在，跳过下载"
fi

# 运行基线测试
echo "=========================================="
echo "运行基线测试..."
echo "=========================================="
python run.py --mode baseline --max-samples 100 --output results/baseline_results.json

# 运行INT8量化优化测试
echo "=========================================="
echo "运行INT8量化优化测试..."
echo "=========================================="
python run.py --mode optimized --quantization int8 --max-samples 100 --output results/int8_results.json

# 对比结果
echo "=========================================="
echo "对比测试结果..."
echo "=========================================="
python run.py --mode compare --quantization int8 --max-samples 100

echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "结果保存在 results/ 目录"
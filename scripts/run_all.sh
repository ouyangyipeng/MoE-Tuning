#!/bin/bash
# MoE语言模型端到端效率优化 - 一键运行脚本

set -e

echo "=========================================="
echo "MoE语言模型端到端效率优化"
echo "=========================================="

# 激活虚拟环境
if [ -d "venv" ]; then
    echo "激活虚拟环境..."
    . venv/bin/activate
else
    echo "创建虚拟环境..."
    python -m venv venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1

# 创建必要的目录
mkdir -p results logs

# 检查Python版本
echo "检查Python版本..."
python --version

# 检查环境
echo "检查环境..."
python scripts/check_env.py

# 运行小模型测试（验证代码逻辑）
echo "=========================================="
echo "运行小模型测试..."
echo "=========================================="
python test_small_model.py

echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo "结果保存在 results/ 目录"

echo ""
echo "注意：完整测试需要下载Mixtral-8x7B模型（约94GB）"
echo "请在有足够存储空间和GPU显存的环境中运行完整测试"
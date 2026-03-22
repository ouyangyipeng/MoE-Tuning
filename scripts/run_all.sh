#!/bin/bash
# MoE语言模型端到端效率优化 - 一键运行脚本

set -e

echo "============================================================"
echo "MoE语言模型端到端效率优化 - 自动运行脚本"
echo "============================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查虚拟环境
if [ ! -d "venv" ]; then
    print_info "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
print_info "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
print_info "检查并安装依赖..."
pip install -r requirements.txt -q

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1

# 创建必要的目录
mkdir -p models data results logs

# 解析命令行参数
MODE=${1:-"test"}
QUANTIZATION=${2:-"int8"}
MAX_SAMPLES=${3:-100}

print_info "运行模式: $MODE"
print_info "量化方法: $QUANTIZATION"
print_info "样本数量: $MAX_SAMPLES"

case $MODE in
    "check")
        print_info "运行环境检查..."
        python scripts/check_env.py
        ;;
    "test")
        print_info "运行小模型测试..."
        python test_small_model.py
        ;;
    "baseline")
        print_info "运行基线测试..."
        python run.py --mode baseline --max-samples $MAX_SAMPLES
        ;;
    "optimized")
        print_info "运行优化测试 ($QUANTIZATION)..."
        python run.py --mode optimized --quantization $QUANTIZATION --max-samples $MAX_SAMPLES
        ;;
    "compare")
        print_info "运行对比测试..."
        python run.py --mode compare --quantization $QUANTIZATION --max-samples $MAX_SAMPLES
        ;;
    "competition")
        print_info "运行比赛脚本..."
        python scripts/run_competition.py --quantization $QUANTIZATION --num-samples $MAX_SAMPLES --output results/results.json
        ;;
    "all")
        print_info "运行完整测试流程..."
        
        print_info "1. 环境检查..."
        python scripts/check_env.py
        
        print_info "2. 小模型测试..."
        python test_small_model.py
        
        print_info "3. 基线测试..."
        python run.py --mode baseline --max-samples $MAX_SAMPLES 2>&1 | tee logs/baseline.log
        
        print_info "4. INT8优化测试..."
        python run.py --mode optimized --quantization int8 --max-samples $MAX_SAMPLES 2>&1 | tee logs/optimized_int8.log
        
        print_info "5. INT4优化测试..."
        python run.py --mode optimized --quantization int4 --max-samples $MAX_SAMPLES 2>&1 | tee logs/optimized_int4.log
        
        print_info "6. 对比测试..."
        python run.py --mode compare --quantization int8 --max-samples $MAX_SAMPLES 2>&1 | tee logs/compare.log
        ;;
    *)
        print_error "未知模式: $MODE"
        echo "可用模式: check, test, baseline, optimized, compare, competition, all"
        exit 1
        ;;
esac

print_info "完成！"

# 显示结果
if [ -f "results/results.json" ]; then
    print_info "结果文件: results/results.json"
    cat results/results.json
fi
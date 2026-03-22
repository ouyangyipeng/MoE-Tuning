# MoE语言模型端到端效率优化 - 进度记录

## 比赛概述

**比赛名称**：2025年全国大学生计算机系统能力大赛 - 智能计算创新设计赛（先导杯）

**赛题**：MoE语言模型端到端效率优化

**目标模型**：Mixtral-8x7B-v0.1

---

## 关键约束条件

### 硬件环境
- 国产X86处理器
- 至少2张K100-AI加速卡
- 单卡显存：64GB
- 算力：192TFLOPS (FP16)

### 软件环境
- Python 3.10
- PyTorch 2.1.0 (dtk24.04)
- Transformers 4.47.1

### 评分标准

#### 初赛（100分）
- **指标**：平均单句推理延迟
- **约束**：困惑度上升不超过15%
- **公式**：得分 = (优化前延迟 - 当前延迟) / (优化前延迟 - 最佳延迟) × 100

#### 决赛（100分）
| 项目 | 分值 | 说明 |
|------|------|------|
| 推理效率 | 30分 | 平均单句推理延迟 |
| 显存占用 | 25分 | 显存占用峰值 |
| 困惑度保持 | 25分 | 上升0%得满分，0-15%线性递减 |
| 创新性 | 20分 | 方法创新性、实现难度、可扩展性 |

### 禁止事项
- ❌ 不允许使用vLLM等外部推理框架
- ❌ 不能跳层
- ❌ 不能改变专家数量

---

## 优化方向

### 1. 专家计算与路由优化
- [x] 自适应负载均衡策略（代码已实现）
- [x] 专家分层协同决策机制（代码已实现）

### 2. 资源与部署优化
- [x] 参数分片与高效缓存策略（代码已实现）
- [x] 量化压缩技术（INT8/INT4）

### 3. 专家通信优化
- [x] 量化压缩与混合精度技术
- [x] CPU Offload策略

---

## 资源链接

- 模型下载：https://hf-mirror.com/mistralai/Mixtral-8x7B-v0.1
- 数据集下载：https://hf-mirror.com/datasets/Salesforce/wikitext
- 大赛官网：https://pra.educg.net

---

## 进度日志

### 2026-03-22 - 代码完善与测试

#### 当前状态
- [x] 阅读比赛文档，理解比赛要求
- [x] 环境搭建与配置（虚拟环境venv + 依赖安装）
- [x] 推送代码到GitHub
- [x] 修改代码支持从HuggingFace直接加载模型
- [x] 运行小模型测试验证代码逻辑
- [x] 完善优化代码（MoE特定优化、推理优化）
- [x] 准备比赛提交材料
- [ ] 最终代码提交到GitHub

#### 已完成工作

**1. 环境配置完成**
- 创建虚拟环境venv
- 安装PyTorch 2.1.0+cu121
- 安装Transformers 4.47.1
- 安装bitsandbytes 0.45.5
- 修复numpy兼容性问题（降级到numpy<2）

**2. 代码结构完善**
```
moe/
├── README.md                    # 项目说明文档（已更新）
├── PROGRESS.md                  # 进度记录
├── requirements.txt             # 依赖列表
├── run.py                       # 主运行脚本
├── download_model.py            # 模型下载脚本
├── test_small_model.py          # 小模型测试脚本
├── plans/
│   └── implementation_plan.md   # 实施计划
├── src/
│   ├── __init__.py
│   ├── config.py                # 配置文件
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── test_baseline.py    # 基线测试
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── quantization.py     # 量化优化
│   │   ├── memory.py           # 显存优化
│   │   ├── inference.py        # 推理优化
│   │   ├── moe_optimization.py # MoE特定优化
│   │   └── optimized_test.py   # 优化测试
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # 工具函数
├── scripts/
│   ├── check_env.py            # 环境检查脚本
│   ├── run_competition.py      # 比赛运行脚本（新增）
│   └── run_all.sh              # 一键运行脚本
└── docs/
    ├── optimization_report.md  # 优化报告
    └── project_summary.md      # 项目总结
```

**3. 小模型测试通过**
```
测试结果汇总:
  environment: ✅ 通过
  small_model: ✅ 通过
  quantization: ✅ 通过
  dataset: ✅ 通过
  perplexity: ✅ 通过
  memory: ✅ 通过
总计: 6/6 测试通过
```

**4. 核心功能实现**
- INT8/INT4量化支持
- 模型分片与多GPU支持
- CPU Offload支持
- KV Cache优化
- 专家缓存策略
- 困惑度计算
- 延迟测量
- 显存监控

**5. GitHub推送完成**
- 仓库地址：git@github.com:ouyangyipeng/MoE-Tuning.git
- 所有代码已推送到main分支

---

### 2026-03-21 - 项目启动

#### 已完成工作

**项目结构创建完成：**
- 创建项目目录结构
- 编写README.md
- 编写requirements.txt
- 创建配置文件config.py
- 创建基线测试代码
- 创建优化模块代码

**优化策略设计：**
1. 量化优化（INT8/INT4）
2. 显存优化（模型分片、CPU Offload、KV Cache）
3. 推理优化（Flash Attention、批处理）
4. MoE特定优化（专家缓存、预加载、负载均衡）

---

## 下一步计划

1. 在比赛环境中运行完整测试
2. 根据测试结果调整优化参数
3. 记录性能数据
4. 准备最终提交材料

---

## 技术细节

### 量化配置

**INT8量化：**
```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
```

**INT4量化：**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 显存优化

**模型分片：**
```python
device_map = "auto"  # 自动分配到多GPU
max_memory = {0: "30GB", 1: "30GB"}  # 限制每GPU显存
```

**CPU Offload：**
```python
offload_folder = "offload"
offload_state_dict = True
```

### 推理优化

**KV Cache：**
- 启用KV Cache加速自回归生成
- 优化Cache存储和访问

**Flash Attention：**
- 使用Flash Attention 2加速注意力计算
- 减少显存占用

---

## 预期性能

| 指标 | 基线 | INT8量化 | INT4量化 |
|------|------|----------|----------|
| 显存占用 | ~90GB | ~45GB | ~25GB |
| 推理延迟 | 基准 | 降低10-20% | 降低15-30% |
| 困惑度变化 | 0% | <1% | <5% |

---

## 注意事项

1. **模型下载**：由于模型较大（~190GB），需要在比赛环境中下载
2. **显存管理**：确保显存使用在128GB以内
3. **困惑度约束**：优化后困惑度上升不超过15%
4. **禁止事项**：不能使用vLLM等外部推理框架
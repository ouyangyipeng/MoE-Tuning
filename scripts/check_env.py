#!/usr/bin/env python3
"""
环境检查脚本 - 验证所有依赖是否正确安装
"""

import sys
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("  ✅ Python版本符合要求 (3.10+)")
        return True
    else:
        print("  ❌ Python版本不符合要求，需要3.10+")
        return False

def check_package(package_name, import_name=None, required_version=None):
    """检查包是否安装"""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "未知")
        print(f"  {package_name}: {version}", end="")
        
        if required_version:
            if version == required_version:
                print(f" ✅ (符合要求: {required_version})")
                return True
            else:
                print(f" ⚠️ (建议版本: {required_version})")
                return True
        else:
            print(" ✅")
            return True
    except ImportError:
        print(f"  {package_name}: ❌ 未安装")
        return False

def check_cuda():
    """检查CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA可用: ✅")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("  CUDA不可用: ⚠️ (将使用CPU)")
            return False
    except ImportError:
        print("  CUDA检查失败: torch未安装")
        return False

def main():
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    results = []
    
    # 检查Python
    print("\n[1] Python环境")
    results.append(check_python_version())
    
    # 检查核心依赖
    print("\n[2] 核心依赖")
    results.append(check_package("torch", required_version="2.1.0"))
    results.append(check_package("transformers", required_version="4.47.1"))
    results.append(check_package("accelerate"))
    results.append(check_package("datasets"))
    
    # 检查其他依赖
    print("\n[3] 其他依赖")
    results.append(check_package("sentencepiece"))
    results.append(check_package("protobuf"))
    results.append(check_package("sklearn", "sklearn"))
    results.append(check_package("tqdm"))
    results.append(check_package("numpy"))
    
    # 检查量化依赖
    print("\n[4] 量化依赖")
    results.append(check_package("bitsandbytes"))
    
    # 检查CUDA
    print("\n[5] CUDA环境")
    check_cuda()
    
    # 总结
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"检查结果: {passed}/{total} 通过")
    print("=" * 50)
    
    if passed == total:
        print("✅ 环境配置完成，可以开始测试")
        return 0
    else:
        print("⚠️ 部分依赖未安装，请运行: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
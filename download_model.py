#!/usr/bin/env python3
"""
模型下载脚本 - MoE语言模型端到端效率优化

下载Mixtral-8x7B模型和wikitext-103-v1数据集
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_hf_mirror():
    """设置HuggingFace镜像"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    logger.info(f"已设置HuggingFace镜像: {os.environ['HF_ENDPOINT']}")


def download_model(model_name: str, save_dir: str, use_mirror: bool = True):
    """下载模型"""
    from huggingface_hub import snapshot_download
    
    if use_mirror:
        setup_hf_mirror()
    
    logger.info(f"开始下载模型: {model_name}")
    logger.info(f"保存目录: {save_dir}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载模型
    local_dir = snapshot_download(
        repo_id=model_name,
        local_dir=save_dir,
        resume_download=True,
        local_dir_use_symlinks=False
    )
    
    logger.info(f"模型下载完成: {local_dir}")
    return local_dir


def download_dataset(dataset_name: str, save_dir: str, use_mirror: bool = True):
    """下载数据集"""
    from huggingface_hub import snapshot_download
    
    if use_mirror:
        setup_hf_mirror()
    
    logger.info(f"开始下载数据集: {dataset_name}")
    logger.info(f"保存目录: {save_dir}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载数据集
    try:
        local_dir = snapshot_download(
            repo_id=dataset_name,
            local_dir=save_dir,
            repo_type="dataset",
            resume_download=True,
            local_dir_use_symlinks=False
        )
        logger.info(f"数据集下载完成: {local_dir}")
    except Exception as e:
        logger.warning(f"使用snapshot_download失败: {e}")
        logger.info("尝试使用datasets库下载...")
        
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, "wikitext-103-v1")
        dataset.save_to_disk(save_dir)
        logger.info(f"数据集下载完成: {save_dir}")
    
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="下载模型和数据集")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1",
                       help="模型名称")
    parser.add_argument("--model-dir", type=str, default="./models/Mixtral-8x7B-v0.1",
                       help="模型保存目录")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext",
                       help="数据集名称")
    parser.add_argument("--dataset-dir", type=str, default="./data/wikitext",
                       help="数据集保存目录")
    parser.add_argument("--no-mirror", action="store_true",
                       help="不使用镜像")
    parser.add_argument("--download-model", action="store_true",
                       help="下载模型")
    parser.add_argument("--download-dataset", action="store_true",
                       help="下载数据集")
    parser.add_argument("--download-all", action="store_true",
                       help="下载模型和数据集")
    
    args = parser.parse_args()
    
    use_mirror = not args.no_mirror
    
    if args.download_all:
        args.download_model = True
        args.download_dataset = True
    
    if not args.download_model and not args.download_dataset:
        parser.print_help()
        print("\n请指定要下载的内容: --download-model, --download-dataset, 或 --download-all")
        return
    
    if args.download_model:
        try:
            download_model(args.model, args.model_dir, use_mirror)
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
    
    if args.download_dataset:
        try:
            download_dataset(args.dataset, args.dataset_dir, use_mirror)
        except Exception as e:
            logger.error(f"数据集下载失败: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動下載漫畫上色系統所需的模型
"""

import os
import sys
import subprocess
from pathlib import Path

def check_git_lfs():
    """檢查並安裝 git-lfs"""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("✅ Git LFS 已安裝")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Git LFS 未安裝，正在安裝...")
        try:
            # 在 macOS 上使用 Homebrew 安裝
            subprocess.run(["brew", "install", "git-lfs"], check=True)
            subprocess.run(["git", "lfs", "install"], check=True)
            print("✅ Git LFS 安裝成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ 無法自動安裝 Git LFS，請手動安裝：")
            print("   macOS: brew install git-lfs")
            print("   Ubuntu: sudo apt install git-lfs")
            print("   Windows: 從 https://git-lfs.github.io/ 下載")
            return False

def download_model(repo_url, target_dir):
    """下載單個模型"""
    print(f"📥 正在下載: {repo_url}")
    try:
        subprocess.run([
            "git", "clone", repo_url, target_dir
        ], check=True, cwd="models")
        print(f"✅ 下載完成: {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 下載失敗: {repo_url}")
        print(f"   錯誤: {e}")
        return False

def main():
    print("🚀 漫畫上色系統 - 模型自動下載器")
    print("=" * 50)
    
    # 檢查 models 目錄
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("📁 創建 models 目錄")
    
    # 檢查 git-lfs
    if not check_git_lfs():
        sys.exit(1)
    
    # 模型清單
    models_to_download = [
        {
            "name": "Stable Diffusion v1.5",
            "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
            "dir": "stable-diffusion-v1-5",
            "size": "~4GB"
        },
        {
            "name": "ControlNet Canny",
            "url": "https://huggingface.co/lllyasviel/sd-controlnet-canny",
            "dir": "sd-controlnet-canny",
            "size": "~1.4GB"
        },
        {
            "name": "ControlNet Lineart",
            "url": "https://huggingface.co/lllyasviel/control_v11p_sd15_lineart",
            "dir": "control_v11p_sd15_lineart",
            "size": "~1.4GB"
        }
    ]
    
    print(f"\n📋 準備下載 {len(models_to_download)} 個模型")
    total_size = "~7GB"
    print(f"💾 預估總大小: {total_size}")
    print("⏰ 預估下載時間: 10-30分鐘（取決於網路速度）")
    
    # 確認下載
    response = input("\n🤔 是否開始下載？(y/n): ").lower().strip()
    if response != 'y':
        print("❌ 取消下載")
        sys.exit(0)
    
    # 開始下載
    success_count = 0
    for model in models_to_download:
        print(f"\n{'-' * 30}")
        print(f"📦 模型: {model['name']}")
        print(f"📏 大小: {model['size']}")
        
        target_path = models_dir / model['dir']
        if target_path.exists():
            print(f"⭐ 已存在，跳過: {model['dir']}")
            success_count += 1
            continue
        
        if download_model(model['url'], model['dir']):
            success_count += 1
    
    # 結果報告
    print(f"\n{'=' * 50}")
    print(f"📊 下載完成: {success_count}/{len(models_to_download)} 個模型")
    
    if success_count == len(models_to_download):
        print("🎉 所有模型下載成功！")
        print("\n📝 接下來的步驟：")
        print("1. 從 Civitai 手動下載 LoRA 模型到 models/loras/ 目錄")
        print("2. 執行: python main.py")
    else:
        print("⚠️  部分模型下載失敗，請檢查網路連接或手動下載")
    
    print(f"\n📖 詳細說明請參考: MODEL_DOWNLOAD_GUIDE.md")

if __name__ == "__main__":
    main() 
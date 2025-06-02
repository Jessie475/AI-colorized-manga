# 🎨 AI 漫畫自動上色系統

## 📖 專案簡介

一個基於深度學習的漫畫自動上色系統，提供兩種不同的上色方案：
- **輕量版**：使用 Stable Diffusion v1.5，模型小巧（~4GB）
- **專業版**：使用 Waifu Diffusion，專為動漫風格優化

### 🌟 核心特色

- **🎯 智能上色**：直接從黑白漫畫生成彩色版本
- **⚡ 簡化流程**：無需複雜預處理，一鍵上色
- **💾 資源友善**：輕量版僅需 4GB 模型
- **🌐 Web 介面**：基於 Gradio 的直觀操作界面

## 🛠 技術架構

| 版本 | 模型 | 大小 | 特色 |
|------|------|------|------|
| 輕量版 | Stable Diffusion v1.5 | ~4GB | 資源需求低，快速上色 |
| 專業版 | Waifu Diffusion | ~4GB | 專為動漫優化，效果更佳 |

## 💻 系統需求

### 🔧 硬體規格
- **顯示卡**：NVIDIA GPU 4GB+ VRAM（建議 8GB+）
- **記憶體**：8GB+ RAM（建議 16GB+）
- **儲存空間**：6GB+（模型檔案 + 系統空間）

### 📦 軟體環境
- Python 3.8 ~ 3.11
- CUDA 11.8+ （GPU 版本）
- PyTorch 2.0+

## 🚀 快速開始

### 1️⃣ 安裝依賴

```bash
# 克隆專案
git clone [repository-url]
cd manga-coloring-system

# 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
```

### 2️⃣ 下載模型

```bash
# 自動下載模型（輕量版需要）
python download_models.py

# 系統會自動下載：
# - Stable Diffusion v1.5 (~4GB)
# - ControlNet Canny (~1.4GB) 
# - ControlNet Lineart (~1.4GB)
```

### 3️⃣ 選擇版本並啟動

####  Stable Diffusion v1.5
```bash
python manga_coloring_lite.py
```

####  Waifu Diffusion
```bash
python manga_coloring_waifu.py
```

服務將在 `http://localhost:7860` 啟動 ✨

## 🎯 版本比較

### 🌿 輕量版 (manga_coloring_lite.py)

**適用場景：**
- 初次使用或硬體資源有限
- 希望快速體驗系統功能

**功能特色：**
- 📦 只需下載 SD v1.5 模型
- ⚡ 直接處理原始圖像，流程簡化
- 🎨 支援黑白線稿、灰階圖像

**操作步驟：**
1. 上傳漫畫圖片
2. 調整參數（變換強度、引導強度、生成步數）
3. 點擊「開始上色」

### 🎨 專業版 (manga_coloring_waifu.py)

**適用場景：**
- 追求專業動漫風格效果
- 處理各種類型的漫畫內容

**功能特色：**
- 🎯 使用 Waifu Diffusion 專業動漫模型
- 🌈 通用漫畫上色
- 🚀 移除邊緣檢測，直接智能上色
- ✨ 更適合專業漫畫創作需求

**操作步驟：**
1. 上傳黑白漫畫圖片
2. 點擊「自動上色」
3. 可選：調整進階參數

## 📁 專案架構

```
manga-coloring-system/
├── 📄 download_models.py         # 自動模型下載器
├── 🌿 manga_coloring_lite.py     # 輕量版上色系統
├── 🎨 manga_coloring_waifu.py    # 專業版上色系統
├── 📋 requirements.txt           # Python 依賴套件
├── 📖 README.md                  # 專案說明文件
├── 📁 models/                    # 模型檔案目錄
│   ├── 📁 stable-diffusion-v1-5/ # SD v1.5 模型
│   ├── 📁 sd-controlnet-canny/   # Canny 邊緣檢測
│   └── 📁 control_v11p_sd15_lineart/ # 線稿檢測
└── 📁 cache/                     # 系統暫存目錄
```

## ⚙️ 參數設定指南

### 🌿 輕量版參數

| 參數 | 建議範圍 | 說明 |
|------|----------|------|
| 變換強度 | 0.3-0.7 | 控制色彩變化程度 |
| 引導強度 | 10-15 | 控制風格遵從度 |
| 生成步數 | 20-30 | 平衡品質與速度 |

### 🎨 專業版參數

| 參數 | 建議範圍 | 說明 |
|------|----------|------|
| 上色強度 | 0.5-0.7 | 控制上色豐富度 |
| 色彩指導 | 10-15 | 控制色彩準確性 |
| 生成品質 | 25-30 | 控制輸出品質 |

## 📜 授權聲明

本專案僅供學術研究和個人學習使用。請遵守：
- Stable Diffusion 模型授權條款
- Waifu Diffusion 開源授權
- 相關開源專案的使用規範

## 🙏 致謝

感謝以下開源專案：
- **Stability AI** - Stable Diffusion 基礎模型
- **Hakurei** - Waifu Diffusion 動漫專用模型
- **Hugging Face** - Diffusers 函式庫與模型託管
- **Gradio 團隊** - Web UI 框架

---

**🌟 選擇適合的版本開始您的漫畫上色之旅！** 
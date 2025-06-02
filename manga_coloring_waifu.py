#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用漫畫自動上色系統 - Waifu Diffusion 版本（無邊緣檢測）
"""

import torch
import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from diffusers import StableDiffusionImg2ImgPipeline

class UniversalMangaColoringSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 使用設備: {self.device}")
        
        # 載入 Waifu Diffusion 模型
        self._load_model()
    
    def _load_model(self):
        """載入 Waifu Diffusion 模型 - 專門針對動漫風格"""
        print("🚀 載入 Waifu Diffusion 模型...")
        
        try:
            # 使用 Waifu Diffusion，更適合動漫風格
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "hakurei/waifu-diffusion",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # 啟用記憶體優化
            if self.device == "cuda":
                self.pipe.enable_memory_efficient_attention()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    print("⚠️ xformers 不可用，使用標準注意力機制")
            
            print("✅ Waifu Diffusion 載入成功！")
            
        except Exception as e:
            print(f"❌ Waifu Diffusion 載入失敗: {e}")
            print("🔄 降級到 Stable Diffusion v1.5...")
            
            # 降級選項
            try:
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                print("✅ Stable Diffusion v1.5 載入成功！")
            except Exception as e2:
                print(f"❌ 降級也失敗: {e2}")
                raise
    
    def enhance_input_image(self, image):
        """增強輸入圖像的品質"""
        # 增強對比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # 增強銳度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def create_base_image(self, original_image):
        """創建適合上色的基礎圖像（不使用邊緣檢測）"""
        # 增強原始圖像
        enhanced_image = self.enhance_input_image(original_image)
        
        # 轉為灰階
        gray_base = ImageOps.grayscale(enhanced_image).convert("RGB")
        
        # 創建更好的基礎圖像
        # 1. 調整亮度和對比度
        enhancer = ImageEnhance.Brightness(gray_base)
        gray_base = enhancer.enhance(1.15)
        
        enhancer = ImageEnhance.Contrast(gray_base)
        gray_base = enhancer.enhance(1.3)
        
        # 2. 添加輕微的暖色調，適合動漫風格
        warm_tint = Image.new("RGB", gray_base.size, (255, 248, 240))
        base_image = Image.blend(gray_base, warm_tint, 0.2)
        
        return base_image
    
    def get_universal_prompts(self):
        """通用漫畫上色提示詞 - 不限定特定風格或人物"""
        
        # 通用的漫畫上色提示詞
        positive_prompt = """masterpiece, best quality, high resolution, 
        anime style, manga coloring, cel shading, vibrant colors, 
        detailed illustration, clean lineart, professional coloring, 
        colorful anime artwork, detailed shading, bright colors, 
        anime aesthetic, manga art style, high quality coloring"""
        
        # 避免不良效果的負面提示詞
        negative_prompt = """monochrome, grayscale, black and white, 
        sketch, rough lines, unfinished, low quality, blurry, 
        bad anatomy, deformed, ugly, watermark, signature, text, 
        realistic, photorealistic, western cartoon style, 
        oversaturated, muddy colors, dull colors, dark, gloomy"""
        
        return positive_prompt, negative_prompt
    
    def process_image(self, input_image, strength, guidance_scale, num_steps):
        """通用漫畫圖像上色處理（無邊緣檢測）"""
        try:
            # 調整圖像大小，保持比例
            aspect_ratio = input_image.width / input_image.height
            if aspect_ratio > 1:
                new_width = 512
                new_height = int(512 / aspect_ratio)
            else:
                new_height = 512
                new_width = int(512 * aspect_ratio)
            
            # 確保尺寸是 8 的倍數
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            input_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 創建基礎圖像（不使用邊緣檢測）
            base_image = self.create_base_image(input_image)
            
            # 獲取通用提示詞
            prompt, negative_prompt = self.get_universal_prompts()
            
            print(f"🎨 使用通用漫畫上色提示詞（無邊緣檢測）")
            
            # 使用 Waifu Diffusion 生成
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                eta=0.0
            ).images[0]
            
            # 返回結果（不再返回邊緣圖像）
            return result, None
            
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
            return None, None

def create_interface():
    """創建簡潔的 Gradio 介面"""
    system = UniversalMangaColoringSystem()
    
    def process_wrapper(image, strength, guidance, steps):
        if image is None:
            return None, "❌ 請上傳圖片"
        
        result, _ = system.process_image(image, strength, guidance, steps)
        
        if result is None:
            return None, "❌ 處理失敗"
        
        return result, "✅ 上色完成！"
    
    # 簡潔的介面設計
    with gr.Blocks(title="🎨 通用漫畫自動上色", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 通用漫畫自動上色系統")
        gr.Markdown("### 🚀 上傳任何黑白漫畫圖片，自動智能上色")
        
        with gr.Row():
            with gr.Column():
                # 輸入區域
                input_image = gr.Image(
                    label="上傳黑白漫畫圖",
                    type="pil",
                    height=450
                )
                
                # 簡化的參數控制
                with gr.Accordion("🔧 進階設定", open=False):
                    strength = gr.Slider(
                        label="上色強度",
                        minimum=0.3,
                        maximum=0.9,
                        value=0.6,
                        step=0.1,
                        info="控制上色的強度，越高顏色越豐富"
                    )
                    
                    guidance_scale = gr.Slider(
                        label="色彩指導",
                        minimum=8,
                        maximum=20,
                        value=12.0,
                        step=1.0,
                        info="控制色彩的準確性"
                    )
                    
                    num_steps = gr.Slider(
                        label="生成品質",
                        minimum=20,
                        maximum=40,
                        value=28,
                        step=2,
                        info="越高品質越好但速度較慢"
                    )
                
                process_btn = gr.Button("🎨 自動上色", variant="primary", size="lg")
            
            with gr.Column():
                # 輸出區域（簡化）
                output_image = gr.Image(
                    label="上色結果",
                    height=450
                )
                
                status_text = gr.Textbox(
                    label="處理狀態",
                    value="等待上傳圖片...",
                    interactive=False
                )
        
        # 使用說明
        gr.Markdown("### 📖 使用說明")
        gr.Markdown("""
        **✨ 功能特色：**
        - 🎯 **智能識別**：直接從原圖識別漫畫內容進行上色
        - 🌈 **通用上色**：不限定特定風格，適用於各種漫畫
        - 🚀 **一鍵處理**：上傳圖片即可自動上色
        - 🎨 **專業效果**：使用 Waifu Diffusion 模型，專為動漫優化
        - ⚡ **簡化流程**：移除邊緣檢測，直接進行上色處理
        
        **📝 使用步驟：**
        1. 上傳黑白漫畫圖片（線稿、黑白頁面等）
        2. 點擊「自動上色」按鈕
        3. 等待處理完成
        4. 如需調整效果，可展開「進階設定」調整參數
        
        **💡 效果最佳的圖片類型：**
        - 線條清晰的黑白漫畫
        - 有明確輪廓的角色或場景
        - 對比度較高的圖片
        
        **⚙️ 參數說明：**
        - **上色強度**：0.5-0.7 適合大多數情況
        - **色彩指導**：10-15 獲得平衡的色彩
        - **生成品質**：25-30 平衡速度與效果
        
        **🔧 系統優化：**
        - 移除邊緣檢測步驟，提升處理速度
        - 直接從原圖進行智能上色
        - 減少計算量，降低記憶體使用
        """)
        
        # 示例圖片（可選）
        gr.Markdown("### 🖼️ 支援的圖片類型")
        gr.Markdown("✅ 漫畫人物、場景、物品、建築、自然景觀等各種黑白漫畫內容")
        
        # 綁定事件（簡化輸出）
        process_btn.click(
            fn=process_wrapper,
            inputs=[input_image, strength, guidance_scale, num_steps],
            outputs=[output_image, status_text]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 啟動通用漫畫自動上色系統（無邊緣檢測版）...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
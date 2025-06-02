#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
輕量版漫畫自動上色系統 - 改進版
"""

import torch
import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from diffusers import StableDiffusionImg2ImgPipeline

class LiteColoringSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 使用設備: {self.device}")
        
        # 載入輕量模型
        self._load_model()
    
    def _load_model(self):
        """載入基礎模型 - 只需要 Stable Diffusion"""
        print("🚀 載入 Stable Diffusion v1.5（輕量版）...")
        
        try:
            # 使用 img2img 管道，比 ControlNet 更輕量
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            print("✅ 模型載入成功！")
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            raise
    
    def extract_edges(self, image):
        """改進的邊緣檢測"""
        # 轉為灰階
        gray = ImageOps.grayscale(image)
        
        # 轉為 numpy array
        img_array = np.array(gray)
        
        # 使用高斯模糊減少噪音
        blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # 使用更適合漫畫的 Canny 參數
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # 擴張邊緣讓線條更粗
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 轉回 PIL Image
        edge_image = Image.fromarray(edges).convert("RGB")
        
        return edge_image
    
    def enhance_input_image(self, image):
        """增強輸入圖像的品質"""
        # 增強對比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # 增強銳度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def create_base_image(self, original_image, edge_image):
        """改進的基礎圖像創建"""
        # 增強原始圖像
        enhanced_image = self.enhance_input_image(original_image)
        
        # 轉為灰階
        gray_base = ImageOps.grayscale(enhanced_image).convert("RGB")
        
        # 創建更好的基礎圖像
        # 1. 調整亮度和對比度
        enhancer = ImageEnhance.Brightness(gray_base)
        gray_base = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(gray_base)
        gray_base = enhancer.enhance(1.2)
        
        # 2. 添加輕微的暖色調
        warm_tint = Image.new("RGB", gray_base.size, (255, 250, 240))
        base_image = Image.blend(gray_base, warm_tint, 0.15)
        
        # 3. 將邊緣資訊融合進基礎圖像
        edge_array = np.array(edge_image.convert("L"))
        base_array = np.array(base_image)
        
        # 在邊緣位置保持較深的顏色
        edge_mask = edge_array > 50
        base_array[edge_mask] = base_array[edge_mask] * 0.8
        
        base_image = Image.fromarray(base_array.astype(np.uint8))
        
        return base_image
    
    def get_improved_prompts(self, character_style):
        """改進的提示詞系統"""
        character_prompts = {
            "炭治郎 (鬼滅之刃)": {
                "main": "tanjiro kamado from demon slayer, black to red gradient hair, green and black checkered haori, scar on forehead, kind expression",
                "style": "anime coloring, cel shading, vibrant colors, official anime artwork style",
                "colors": "warm color palette, red and green accents"
            },
            "鳴人 (火影忍者)": {
                "main": "naruto uzumaki from naruto, spiky blonde hair, blue eyes, orange and blue ninja outfit, whisker marks on cheeks",
                "style": "anime coloring, bright vibrant colors, shounen anime art style",
                "colors": "orange and blue color scheme, energetic bright colors"
            },
            "路飛 (海賊王)": {
                "main": "monkey d luffy from one piece, black hair, straw hat, red vest, blue shorts, big smile",
                "style": "one piece anime art style, colorful anime artwork, vibrant shading",
                "colors": "red and blue primary colors, tropical bright palette"
            },
            "悟空 (龍珠)": {
                "main": "son goku from dragon ball, spiky black hair, orange and blue gi, martial arts uniform",
                "style": "dragon ball anime art style, classic anime coloring, dynamic shading",
                "colors": "orange and blue martial arts outfit, classic anime colors"
            },
            "通用動漫風格": {
                "main": "manga style colorful illustration, anime style, vibrant colors",
                "style": "professional anime coloring, cel shading, clean lineart, detailed coloring",
                "colors": "balanced color palette, anime-appropriate colors"
            }
        }
        
        char_info = character_prompts.get(character_style, character_prompts["通用動漫風格"])
        
        positive_prompt = f"{char_info['main']}, {char_info['style']}, {char_info['colors']}, high quality, detailed, masterpiece, best quality, clean lines"
        
        negative_prompt = "monochrome, black and white, grayscale, sketch, pencil drawing, rough lines, blurry, low quality, bad anatomy, deformed, ugly, watermark, signature, text, oversaturated, muddy colors, dull colors"
        
        return positive_prompt, negative_prompt
    
    def process_image(self, input_image, character_style, strength, guidance_scale, num_steps):
        """簡化的圖像處理 - 直接使用原始圖像"""
        try:
            # 調整圖像大小，保持比例
            aspect_ratio = input_image.width / input_image.height
            if aspect_ratio > 1:
                new_width = 512
                new_height = int(512 / aspect_ratio)
            else:
                new_height = 512
                new_width = int(512 * aspect_ratio)
            
            # 確保尺寸是 8 的倍數（Stable Diffusion 要求）
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            input_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 輕微增強原始圖像（可選）
            enhanced_image = self.enhance_input_image(input_image)
            
            # 獲取改進的提示詞
            prompt, negative_prompt = self.get_improved_prompts(character_style)
            
            print(f"🎨 提示詞: {prompt}")
            
            # 直接使用原始圖像作為基礎
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=enhanced_image,  # 直接使用原始圖像
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                eta=0.0  # 確定性更高的生成
            ).images[0]
            
            # 返回結果和原始圖像（用於顯示）
            return result, enhanced_image
            
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
            return None, None

def create_interface():
    """創建 Gradio 介面"""
    system = LiteColoringSystem()
    
    def process_wrapper(image, character, strength, guidance, steps):
        if image is None:
            return None, None, "❌ 請上傳圖片"
        
        result, processed_input = system.process_image(image, character, strength, guidance, steps)
        
        if result is None:
            return None, None, "❌ 處理失敗"
        
        return result, processed_input, "✅ 上色完成！"
    
    # 介面設計
    with gr.Blocks(title="🎨 輕量版漫畫上色系統", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 輕量版漫畫自動上色系統")
        gr.Markdown("### 📦 只需下載一個模型 (~4GB)")
        
        with gr.Row():
            with gr.Column():
                # 輸入區域
                input_image = gr.Image(
                    label="上傳黑白漫畫圖",
                    type="pil",
                    height=400
                )
                
                character_style = gr.Dropdown(
                    label="選擇角色風格",
                    choices=[
                        "炭治郎 (鬼滅之刃)",
                        "鳴人 (火影忍者)", 
                        "路飛 (海賊王)",
                        "悟空 (龍珠)",
                        "通用動漫風格"
                    ],
                    value="通用動漫風格"
                )
                
                with gr.Row():
                    strength = gr.Slider(
                        label="變換強度",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,  # 降低預設值以保持更多原始結構
                        step=0.1,
                        info="越高變化越大"
                    )
                    
                    guidance_scale = gr.Slider(
                        label="引導強度",
                        minimum=5,
                        maximum=20,
                        value=12.0,  # 提高預設值以獲得更好的色彩
                        step=0.5
                    )
                
                num_steps = gr.Slider(
                    label="生成步數",
                    minimum=15,
                    maximum=50,
                    value=25,  # 增加預設步數以獲得更好品質
                    step=5,
                    info="越高品質越好但越慢"
                )
                
                process_btn = gr.Button("🎨 開始上色！", variant="primary", size="lg")
            
            with gr.Column():
                # 輸出區域
                output_image = gr.Image(
                    label="彩色結果",
                    height=400
                )
                
                edge_image = gr.Image(
                    label="處理前圖像",
                    height=200
                )
                
                status_text = gr.Textbox(
                    label="狀態",
                    value="等待處理...",
                    interactive=False
                )
        
        # 使用說明
        gr.Markdown("### 📚 使用說明")
        gr.Markdown("""
        **改進版特色：**
        - 💾 只需下載 Stable Diffusion v1.5 (~4GB)
        - ⚡ 直接處理原始圖像，更快更簡單
        - 🎨 更精確的角色風格提示詞
        - 📱 適合資源有限的環境
        - 🖼️ 支援黑白線稿、灰階圖像和彩色漫畫
        
        **使用步驟：**
        1. 上傳漫畫圖片（黑白線稿、灰階或彩色都可以）
        2. 選擇想要的角色風格
        3. 調整參數：
           - **變換強度**：0.3-0.7 適合大多數情況（越高變化越大）
           - **引導強度**：10-15 可獲得更好的色彩
           - **生成步數**：20-30 平衡品質和速度
        4. 點擊「開始上色」按鈕
        
        **小提示：**
        - 不需要預先處理圖像，直接上傳即可
        - 彩色或灰階漫畫也能進行風格轉換
        - 可以多試幾次不同的參數組合
        - 選擇合適的角色風格很重要
        """)
        
        # 綁定事件
        process_btn.click(
            fn=process_wrapper,
            inputs=[input_image, character_style, strength, guidance_scale, num_steps],
            outputs=[output_image, edge_image, status_text]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 啟動改進版輕量漫畫上色系統...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
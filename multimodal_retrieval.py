"""
Cross-Modal Text-to-Image Retrieval using Jina CLIP v2.
Supports 89+ languages natively including Chinese and English.
"""

import os
import json
import pickle
import numpy as np
from PIL import Image
from typing import List, Dict

from config import DATA_DIR

IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMAGE_INDEX_FILE = os.path.join(DATA_DIR, "image_index.pkl")
IMAGE_METADATA_FILE = os.path.join(DATA_DIR, "image_metadata.json")


class CrossModalRetriever:

    def __init__(self, use_clip=True):
        self.model = None
        self.image_embeddings = {}
        self.image_metadata = {}
        self.model_name = "jinaai/jina-clip-v2"
        
        if use_clip:
            self._load_clip_model()
        os.makedirs(IMAGE_DIR, exist_ok=True)
        self._load_index()

    def _load_clip_model(self):
        """【真正加载 Jina CLIP v2】原生多语言，抛弃一切前置翻译！"""
        try:
            print("[Multimodal] Loading Jina CLIP v2 (Multilingual)...")
            print(f"[Multimodal] Model: {self.model_name}")
            
            # 使用 sentence-transformers 包装器（更稳定）
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print(f"[Multimodal] Jina CLIP v2 loaded via Sentence-Transformers.")
                print(f"[Multimodal] Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
                return
            except Exception as e_st:
                print(f"[Multimodal] Sentence-Transformers fallback: {e_st}")
            
            # Fallback 到直接 Hugging Face
            from transformers import AutoModel, AutoProcessor
            import torch
            self._hf_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self._hf_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self._hf_model.eval()
            self._use_hf = True
            print("[Multimodal] Jina CLIP v2 loaded via Hugging Face AutoModel.")
        except Exception as e:
            print(f"[Multimodal] Error loading Jina CLIP v2: {e}")
            self.model = None

    def _load_index(self):
        if os.path.exists(IMAGE_INDEX_FILE):
            try:
                with open(IMAGE_INDEX_FILE, "rb") as f:
                    self.image_embeddings = pickle.load(f)
                if isinstance(self.image_embeddings, dict):
                    print(f"[Multimodal] Loaded {len(self.image_embeddings)} image embeddings.")
                else:
                    self.image_embeddings = {}
            except Exception:
                os.remove(IMAGE_INDEX_FILE)
                self.image_embeddings = {}
        if os.path.exists(IMAGE_METADATA_FILE):
            try:
                with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
                    self.image_metadata = json.load(f)
            except Exception:
                self.image_metadata = {}

    def _save_index(self):
        with open(IMAGE_INDEX_FILE, "wb") as f:
            pickle.dump(self.image_embeddings, f)
        with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.image_metadata, f, ensure_ascii=False, indent=2)

    def _encode_image(self, image_path):
        """利用 Jina 视觉塔编码图片"""
        image = Image.open(image_path).convert('RGB')
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'encode'):
                vec = self.model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
            else:
                import torch
                inputs = self._hf_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._hf_model.get_image_features(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    outputs = outputs.pooler_output
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                vec = outputs.cpu().numpy().flatten()
        else:
            vec = np.zeros(1024)
        return vec

    def _encode_text(self, text):
        """利用 Jina 文本塔编码文本（直接吃中/英/中英混合语种）"""
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'encode'):
                vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            else:
                import torch
                inputs = self._hf_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self._hf_model.get_text_features(**inputs)
                if hasattr(outputs, 'pooler_output'):
                    outputs = outputs.pooler_output
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                vec = outputs.cpu().numpy().flatten()
        else:
            vec = np.zeros(1024)
        return vec

    def index_images(self):
        exts = ('.jpg', '.jpeg', '.png', '.webp')
        image_dir = IMAGE_DIR
        files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                 if f.lower().endswith(exts)]
        print(f"[Multimodal] Indexing {len(files)} images...")
        for fp in files:
            name = os.path.basename(fp)
            self.image_embeddings[name] = self._encode_image(fp)
            try:
                with Image.open(fp) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0
            self.image_metadata[name] = {"path": fp, "filename": name, "width": w, "height": h}
        self._save_index()
        print(f"[Multimodal] Indexed {len(self.image_embeddings)} images.")

    def search(self, query_text, top_k=5):
        if not self.image_embeddings:
            print("[Multimodal] No images indexed. Run index_images() first.")
            return []

        # 打印证明我们用的是 Jina CLIP v2
        print(f"\n[Multimodal] 🔍 Searching with Jina CLIP v2 (Multilingual)")
        print(f"[Multimodal]   Model: {self.model_name}")
        print(f"[Multimodal]   Query: {query_text}")

        query_vec = self._encode_text(query_text)

        results = []
        for img_id, img_vec in self.image_embeddings.items():
            results.append((img_id, float(np.dot(query_vec, img_vec))))
        results.sort(key=lambda x: x[1], reverse=True)

        output = []
        for img_id, score in results[:top_k]:
            meta = self.image_metadata.get(img_id, {})
            output.append({
                "image_id": img_id,
                "score": round(score, 6),
                "path": meta.get("path", ""),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
            })
        return output

    def get_stats(self):
        return {
            "total_images": len(self.image_embeddings),
            "embedding_dim": len(next(iter(self.image_embeddings.values()))) if self.image_embeddings else 0,
            "model": self.model_name,
        }


# ---- 兼容 512 分辨率的高清 Sample 生成 ----
def create_sample_images():
    """Create 8 visually distinct category images with labels matching Jina 512 Resolution."""
    from PIL import ImageDraw, ImageFont
    os.makedirs(IMAGE_DIR, exist_ok=True)

    specs = [
        ("ai_chip.jpg",       "#1a1a3e", "CHIP\n芯片",     ["#4444cc", "#44cccc"]),
        ("nature_forest.jpg", "#1e3a1e", "FOREST\n自然",   ["#228b22", "#90ee90"]),
        ("city_skyline.jpg",  "#3a3a3a", "CITY\n城市",     ["#888888", "#ffffff"]),
        ("food_dish.jpg",     "#5a3a1a", "FOOD\n美食",     ["#8b4513", "#ffd700"]),
        ("robot_arm.jpg",     "#2a2a4a", "ROBOT\n机器人",   ["#666688", "#aaaaff"]),
        ("solar_panel.jpg",   "#1a3a5a", "SOLAR\n太阳能",  ["#4169e1", "#ffff00"]),
        ("data_center.jpg",   "#0a0a2a", "SERVER\n服务器",  ["#191970", "#00ff00"]),
        ("electric_car.jpg",  "#2a4a1a", "EV\n电动汽车",    ["#228b22", "#ffffff"]),
    ]

    for fn, bg, label, colors in specs:
        fp = os.path.join(IMAGE_DIR, fn)
        if os.path.exists(fp):
            os.remove(fp)
        # 升级为 512x512 高清画质
        img = Image.new('RGB', (512, 512), bg)
        draw = ImageDraw.Draw(img)

        for i in range(0, 512, 64):
            c = colors[i // 64 % len(colors)]
            draw.rectangle([i, 0, i + 32, 512], fill=c)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
        except Exception:
            font = ImageFont.load_default()

        lines = label.split('\n')
        th = len(lines) * 68
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            tx = (512 - tw) // 2
            ty = (512 - th) // 2 + i * 68
            draw.text((tx + 2, ty + 2), line, fill='black', font=font)
            draw.text((tx, ty), line, fill='white', font=font)

        img.save(fp)
    print(f"[Multimodal] Created {len(specs)} high-res 512x512 sample images.")


# 向后兼容别名
CLIPImageRetriever = CrossModalRetriever
"""
Cross-Modal Multimedia Retrieval Module
Text-to-Image + Text-to-Video Retrieval using Jina CLIP v2 + sentence-transformers.

Capabilities:
1. Text-to-Image: natural language query -> relevant images
2. Text-to-Video: natural language query -> relevant videos (frame-level encoding)
3. Unified semantic space via Jina CLIP v2 (1024-dim, 90+ languages)

Architecture:
- sentence-transformers: high-level text encoding API
- Jina CLIP v2 (transformers): image/video frame encoding via get_image_features()
- OpenCV: video frame extraction
"""

# ========== Transformers 5.x clip_loss 兼容补丁 ==========
try:
    import torch
    import transformers.models.clip.modeling_clip as hf_clip_mod

    # 如果模块里没有 clip_loss，手动挂载官方原版实现
    if not hasattr(hf_clip_mod, "clip_loss"):
        def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
            import torch.nn.functional as F
            caption_loss = F.cross_entropy(similarity, torch.arange(similarity.size(0), device=similarity.device))
            image_loss = F.cross_entropy(similarity.T, torch.arange(similarity.size(0), device=similarity.device))
            return (caption_loss + image_loss) / 2.0

        # 强行挂载，让外部导入能找到
        hf_clip_mod.clip_loss = clip_loss

except Exception:
    # 静默失败，不影响主程序
    pass

import os
import json
import pickle
import pickle
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import warnings

import torch

from sentence_transformers import SentenceTransformer
from config import DATA_DIR

IMAGE_DIR = os.path.join(DATA_DIR, "images")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
IMAGE_INDEX_FILE = os.path.join(DATA_DIR, "image_index.pkl")
IMAGE_METADATA_FILE = os.path.join(DATA_DIR, "image_metadata.json")
VIDEO_INDEX_FILE = os.path.join(DATA_DIR, "video_index.pkl")
VIDEO_METADATA_FILE = os.path.join(DATA_DIR, "video_metadata.json")

MODEL_NAME = "jinaai/jina-clip-v2"
EMBEDDING_DIM = 1024
VIDEO_FPS = 1.0
VIDEO_MAX_FRAMES = 60
VIDEO_MIN_FRAMES = 5


def _extract_video_frames(video_path: str, fps: float = VIDEO_FPS) -> List[np.ndarray]:
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for video processing. Install: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Multimodal] Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    sample_interval = max(1, int(video_fps / fps))
    max_samples = min(VIDEO_MAX_FRAMES, max(VIDEO_MIN_FRAMES, int(duration * fps)))

    frames = []
    frame_idx = 0
    while len(frames) < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()

    if len(frames) < VIDEO_MIN_FRAMES and duration > 0:
        cap = cv2.VideoCapture(video_path)
        step = max(1, total_frames // VIDEO_MIN_FRAMES)
        for i in range(VIDEO_MIN_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return frames


class MultimodalRetriever:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.model_name = model_name
        self.image_embeddings: Dict[str, np.ndarray] = {}
        self.image_metadata: Dict[str, dict] = {}
        self.video_embeddings: Dict[str, np.ndarray] = {}
        self.video_metadata: Dict[str, dict] = {}

        self.model = None
        self._clip_model = None
        self._clip_processor = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self._load_index()

    # ──────────────────── model loading ────────────────────

    def _load_model(self):
        print(f"[Multimodal] Loading Jina CLIP v2 via sentence-transformers...")
        print(f"[Multimodal] Device: {self.device}")
        try:
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                device=self.device,
            )
            self._clip_model = self.model._first_module().auto_model
            self._clip_processor = self.model._first_module().processor
            print("[Multimodal] Model loaded successfully. Embedding dim = 1024, 90+ languages.")
        except Exception as e:
            print(f"[Multimodal] Failed via sentence-transformers: {e}")
            self._load_model_fallback()

    def _load_model_fallback(self):
        print("[Multimodal] Falling back to transformers AutoModel...")
        try:
            from transformers import AutoModel, AutoProcessor
            self.model = None
            self._clip_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self._clip_model.eval()
            self._clip_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            print("[Multimodal] Model loaded via transformers fallback.")
        except Exception as e:
            print(f"[Multimodal] FATAL: cannot load model: {e}")
            raise e

    # ──────────────────── text encoding ────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        try:
            if self.model is not None:
                emb = self.model.encode(
                    [text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return emb[0].astype(np.float32)

            return self._encode_text_clip(text)
        except Exception as e:
            print(f"[Multimodal] Text encoding error: {e}")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def _encode_text_clip(self, text: str) -> np.ndarray:
        inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._clip_model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().astype(np.float32)

    # ──────────────────── image encoding ────────────────────

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            return self.encode_image_pil(image)
        except Exception as e:
            print(f"[Multimodal] Image encoding error ({image_path}): {e}")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def encode_image_pil(self, image: Image.Image) -> np.ndarray:
        try:
            inputs = self._clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self._clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"[Multimodal] PIL image encoding error: {e}")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # ──────────────────── video encoding ────────────────────

    def encode_video(self, video_path: str, fps: float = VIDEO_FPS) -> np.ndarray:
        frames = _extract_video_frames(video_path, fps=fps)
        if not frames:
            print(f"[Multimodal] No frames extracted from {video_path}")
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        frame_embeddings = []
        for frame_arr in frames:
            pil_img = Image.fromarray(frame_arr)
            emb = self.encode_image_pil(pil_img)
            frame_embeddings.append(emb)

        video_emb = np.mean(frame_embeddings, axis=0)
        norm = np.linalg.norm(video_emb)
        if norm > 0:
            video_emb = video_emb / norm
        return video_emb.astype(np.float32)

    # ──────────────────── index persistence ────────────────────

    def _load_index(self):
        for file, store, label in [
            (IMAGE_INDEX_FILE, self.image_embeddings, "image embeddings"),
            (VIDEO_INDEX_FILE, self.video_embeddings, "video embeddings"),
        ]:
            if os.path.exists(file):
                try:
                    with open(file, "rb") as f:
                        store.update(pickle.load(f))
                    print(f"[Multimodal] Loaded {len(store)} {label}.")
                except Exception as e:
                    print(f"[Multimodal] Could not load {label}: {e}")

        for file, store, label in [
            (IMAGE_METADATA_FILE, self.image_metadata, "image metadata"),
            (VIDEO_METADATA_FILE, self.video_metadata, "video metadata"),
        ]:
            if os.path.exists(file):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        store.update(json.load(f))
                except Exception as e:
                    print(f"[Multimodal] Could not load {label}: {e}")

    def _save_index(self):
        with open(IMAGE_INDEX_FILE, "wb") as f:
            pickle.dump(self.image_embeddings, f)
        with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.image_metadata, f, ensure_ascii=False, indent=2)
        with open(VIDEO_INDEX_FILE, "wb") as f:
            pickle.dump(self.video_embeddings, f)
        with open(VIDEO_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.video_metadata, f, ensure_ascii=False, indent=2)

    # ──────────────────── indexing ────────────────────

    def index_images(self, image_dir: str = None,
                     extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')):
        image_dir = image_dir or IMAGE_DIR
        if not os.path.isdir(image_dir):
            print(f"[Multimodal] Image directory not found: {image_dir}")
            return

        image_files = sorted(
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        )
        if not image_files:
            print(f"[Multimodal] No images found in {image_dir}")
            return

        print(f"[Multimodal] Indexing {len(image_files)} images from {image_dir} (READ-ONLY)...")

        for i, img_path in enumerate(image_files):
            img_id = os.path.basename(img_path)
            if img_id in self.image_embeddings:
                continue

            self.image_embeddings[img_id] = self.encode_image(img_path)

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    fmt = img.format
            except Exception:
                w, h, fmt = 0, 0, "unknown"

            self.image_metadata[img_id] = {
                "path": img_path, "filename": img_id,
                "width": w, "height": h, "format": fmt,
            }

            if (i + 1) % 20 == 0 or i == len(image_files) - 1:
                print(f"  Image [{i + 1}/{len(image_files)}]")

        self._save_index()
        print(f"[Multimodal] Image indexing done. Total: {len(self.image_embeddings)}")

    def index_videos(self, video_dir: str = None,
                     extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')):
        video_dir = video_dir or VIDEO_DIR
        if not os.path.isdir(video_dir):
            print(f"[Multimodal] Video directory not found: {video_dir}")
            return

        video_files = sorted(
            os.path.join(video_dir, f) for f in os.listdir(video_dir)
            if f.lower().endswith(extensions)
        )
        if not video_files:
            print(f"[Multimodal] No videos found in {video_dir}")
            return

        print(f"[Multimodal] Indexing {len(video_files)} videos from {video_dir} (READ-ONLY)...")

        for i, vid_path in enumerate(video_files):
            vid_id = os.path.basename(vid_path)
            if vid_id in self.video_embeddings:
                continue

            print(f"  Video [{i + 1}/{len(video_files)}]: {vid_id} (extracting frames...)")
            self.video_embeddings[vid_id] = self.encode_video(vid_path)

            file_size = os.path.getsize(vid_path)
            self.video_metadata[vid_id] = {
                "path": vid_path, "filename": vid_id,
                "size_bytes": file_size,
            }

        self._save_index()
        print(f"[Multimodal] Video indexing done. Total: {len(self.video_embeddings)}")

    def index_all(self):
        self.index_images()
        self.index_videos()

    # ──────────────────── search ────────────────────

    def _cosine_search(self, query_emb: np.ndarray,
                       embeddings: Dict[str, np.ndarray],
                       metadata: Dict[str, dict],
                       top_k: int, label: str) -> List[Dict]:
        if not embeddings:
            return []

        results = []
        for item_id, item_emb in embeddings.items():
            sim = float(np.dot(query_emb, item_emb))
            results.append((item_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)


        output = []
        for item_id, score in results[:top_k]:
            meta = metadata.get(item_id, {})
            entry = {
                f"{label}_id": item_id,
                "score": round(score, 6),
                "path": meta.get("path", ""),
            }
            if label == "image":
                entry.update({
                    "width": meta.get("width", 0),
                    "height": meta.get("height", 0),
                    "format": meta.get("format", "unknown"),
                })
            elif label == "video":
                entry["size_mb"] = round(meta.get("size_bytes", 0) / (1024 * 1024), 2)
            output.append(entry)
        return output

    def search_images(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not self.image_embeddings:
            print("[Multimodal] No images indexed. Run index_images() first.")
            return []
        q_emb = self.encode_text(query_text)
        return self._cosine_search(q_emb, self.image_embeddings, self.image_metadata, top_k, "image")

    def search_videos(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not self.video_embeddings:
            print("[Multimodal] No videos indexed. Run index_videos() first.")
            return []
        q_emb = self.encode_text(query_text)
        return self._cosine_search(q_emb, self.video_embeddings, self.video_metadata, top_k, "video")

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        return self.search_images(query_text, top_k=top_k)

    def search_all(self, query_text: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        return {
            "images": self.search_images(query_text, top_k=top_k),
            "videos": self.search_videos(query_text, top_k=top_k),
        }

    # ──────────────────── stats ────────────────────

    def get_stats(self) -> Dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "embedding_dim": EMBEDDING_DIM,
            "total_images": len(self.image_embeddings),
            "total_videos": len(self.video_embeddings),
        }

    # ──────────────────── clear ────────────────────

    def clear_index(self):
        self.image_embeddings.clear()
        self.image_metadata.clear()
        self.video_embeddings.clear()
        self.video_metadata.clear()
        for f in [IMAGE_INDEX_FILE, IMAGE_METADATA_FILE, VIDEO_INDEX_FILE, VIDEO_METADATA_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("[Multimodal] Index cleared.")


# backward-compatible alias for main.py
CLIPImageRetriever = MultimodalRetriever


# ──────────────────── demo ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Multimodal Retrieval Demo (Jina CLIP v2)")
    print("  Text -> Image  |  Text -> Video")
    print("=" * 60)

    retriever = MultimodalRetriever()

    retriever.index_all()

    queries = [
        "人工智能芯片技术",
        "绿色能源环保",
        "城市建筑风景",
        "美食烹饪",
        "AI chip technology",
        "sustainable energy nature",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: \"{q}\"")

        img_results = retriever.search_images(q, top_k=3)
        if img_results:
            print("  [Images]")
            for r in img_results:
                print(f"    {r['score']:.4f}  {r['image_id']}")

        vid_results = retriever.search_videos(q, top_k=3)
        if vid_results:
            print("  [Videos]")
            for r in vid_results:
                print(f"    {r['score']:.4f}  {r['video_id']}")

    print(f"\n{'=' * 60}")
    print("Stats:", retriever.get_stats())

"""
Cross-Modal Information Retrieval Module
Text-to-Image Retrieval using CLIP (Contrastive Language-Image Pre-training)

This module enables:
1. Text-to-Image retrieval: Input natural language query, retrieve relevant images
2. Image-to-Text retrieval: Input image, retrieve relevant text documents
3. Unified semantic space: Both modalities embedded in shared vector space

CLIP Architecture:
- Text Encoder: Transformer-based, outputs text embeddings
- Image Encoder: Vision Transformer (ViT) or ResNet, outputs image embeddings
- Contrastive Learning: Aligns text and image representations in shared space

References:
[1] Radford, A., et al. (2021). Learning Transferable Visual Models From Natural 
    Language Supervision. ICML 2021.
[2] https://github.com/openai/CLIP
"""

import os
import json
import numpy as np
from PIL import Image
import pickle
from typing import List, Dict, Tuple, Optional
import warnings

# Try to import torch and transformers for CLIP
CLIP_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    warnings.warn("CLIP dependencies not available. Using fallback mode.")

from config import DATA_DIR

# Paths for multimodal data
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMAGE_INDEX_FILE = os.path.join(DATA_DIR, "image_index.pkl")
IMAGE_METADATA_FILE = os.path.join(DATA_DIR, "image_metadata.json")


class SimpleImageEncoder:
    """
    Fallback image encoder using simple features when CLIP is unavailable.
    Uses color histogram + edge detection features.
    """
    
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Extract simple visual features from image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            arr = np.array(img)
            
            # Color histogram features (3 channels * 16 bins = 48 dims)
            hist_features = []
            for i in range(3):
                hist, _ = np.histogram(arr[:, :, i].flatten(), bins=16, range=(0, 256))
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                hist_features.extend(hist)
            
            # Simple edge features using gradient
            gray = np.mean(arr, axis=2)
            grad_x = np.abs(np.diff(gray, axis=1)).mean()
            grad_y = np.abs(np.diff(gray, axis=0)).mean()
            edge_features = [grad_x, grad_y]
            
            # Combine and pad to target dimension
            features = np.array(hist_features + edge_features)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
            
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Simple text encoding based on keyword matching."""
        # Keywords for different image categories
        category_keywords = {
            "technology": ["科技", "技术", "人工智能", "AI", "computer", "芯片", "半导体", "5G", "网络", "digital"],
            "nature": ["自然", "风景", "山水", "森林", "海洋", "动物", "植物", "nature", "landscape"],
            "city": ["城市", "建筑", "高楼", "街道", "交通", "urban", "city", "building"],
            "people": ["人物", "人脸", "肖像", "人群", "person", "people", "portrait", "human"],
            "food": ["食物", "美食", "餐厅", "烹饪", "food", "restaurant", "cooking"],
        }
        
        text_lower = text.lower()
        scores = []
        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in text or kw in text_lower)
            scores.append(score)
        
        # Pad to feature dimension
        features = np.array(scores + [0.0] * (self.feature_dim - len(scores)))
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.astype(np.float32)


class CLIPImageRetriever:
    """
    Cross-modal image retriever using CLIP or fallback encoding.
    
    Supports:
    - Text-to-Image: Natural language query retrieves relevant images
    - Image indexing: Build searchable index of local images
    - Similarity ranking: Cosine similarity in shared embedding space
    """
    
    def __init__(self, use_clip: bool = True):
        self.use_clip = use_clip and CLIP_AVAILABLE
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.image_embeddings = {}
        self.image_metadata = {}
        self.simple_encoder = SimpleImageEncoder()
        
        if self.use_clip:
            self._load_clip_model()
        
        # Ensure directories exist
        os.makedirs(IMAGE_DIR, exist_ok=True)
        self._load_index()
    
    def _load_clip_model(self):
        """Load pre-trained CLIP model."""
        try:
            print("[Multimodal] Loading CLIP model...")
            # Use a smaller, faster model variant
            model_name = "openai/clip-vit-base-patch32"  # 150MB vs 600MB for large
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            print("[Multimodal] CLIP model loaded successfully.")
        except Exception as e:
            print(f"[Multimodal] Failed to load CLIP: {e}")
            print("[Multimodal] Falling back to simple encoder.")
            self.use_clip = False
    
    def _load_index(self):
        """Load existing image index."""
        if os.path.exists(IMAGE_INDEX_FILE):
            try:
                with open(IMAGE_INDEX_FILE, "rb") as f:
                    self.image_embeddings = pickle.load(f)
                print(f"[Multimodal] Loaded {len(self.image_embeddings)} image embeddings.")
            except Exception as e:
                print(f"[Multimodal] Could not load index: {e}")
        
        if os.path.exists(IMAGE_METADATA_FILE):
            try:
                with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
                    self.image_metadata = json.load(f)
            except Exception as e:
                print(f"[Multimodal] Could not load metadata: {e}")
    
    def _save_index(self):
        """Save image index to disk."""
        with open(IMAGE_INDEX_FILE, "wb") as f:
            pickle.dump(self.image_embeddings, f)
        with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.image_metadata, f, ensure_ascii=False, indent=2)
    
    def _encode_image_clip(self, image_path: str) -> np.ndarray:
        """Encode image using CLIP."""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=1)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding image with CLIP: {e}")
            return self.simple_encoder.encode_image(image_path)
    
    def _encode_text_clip(self, text: str) -> np.ndarray:
        """Encode text using CLIP."""
        try:
            inputs = self.processor(text=text, return_tensors="pt", truncation=True, max_length=77)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            return text_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding text with CLIP: {e}")
            return self.simple_encoder.encode_text(text)
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image to embedding vector."""
        if self.use_clip:
            return self._encode_image_clip(image_path)
        return self.simple_encoder.encode_image(image_path)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        if self.use_clip:
            return self._encode_text_clip(text)
        return self.simple_encoder.encode_text(text)
    
    def index_images(self, image_dir: str = None, extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.webp')):
        """
        Index all images in directory.
        
        Args:
            image_dir: Directory containing images (default: IMAGE_DIR)
            extensions: Image file extensions to index
        """
        image_dir = image_dir or IMAGE_DIR
        print(f"[Multimodal] Indexing images from {image_dir}...")
        
        image_files = []
        for ext in extensions:
            image_files.extend([
                os.path.join(image_dir, f) 
                for f in os.listdir(image_dir) 
                if f.lower().endswith(ext)
            ])
        
        print(f"[Multimodal] Found {len(image_files)} images.")
        
        for i, img_path in enumerate(image_files):
            img_id = os.path.basename(img_path)
            if img_id in self.image_embeddings:
                continue
            
            embedding = self.encode_image(img_path)
            self.image_embeddings[img_id] = embedding
            
            # Extract metadata
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    format_type = img.format
            except:
                width, height, format_type = 0, 0, "unknown"
            
            self.image_metadata[img_id] = {
                "path": img_path,
                "filename": img_id,
                "width": width,
                "height": height,
                "format": format_type,
            }
            
            if (i + 1) % 10 == 0:
                print(f"  Indexed {i + 1}/{len(image_files)} images...")
        
        self._save_index()
        print(f"[Multimodal] Indexing complete. Total: {len(self.image_embeddings)} images.")
    
    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Text-to-Image retrieval: Find images matching text query.
        
        Args:
            query_text: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of result dicts with image metadata and similarity scores
        """
        if not self.image_embeddings:
            print("[Multimodal] No images indexed. Run index_images() first.")
            return []
        
        # Encode query
        query_embedding = self.encode_text(query_text)
        
        # Compute similarities
        results = []
        for img_id, img_embedding in self.image_embeddings.items():
            similarity = np.dot(query_embedding, img_embedding)
            results.append((img_id, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Format output
        output = []
        for img_id, score in top_results:
            metadata = self.image_metadata.get(img_id, {})
            output.append({
                "image_id": img_id,
                "score": round(score, 6),
                "path": metadata.get("path", ""),
                "width": metadata.get("width", 0),
                "height": metadata.get("height", 0),
                "format": metadata.get("format", "unknown"),
            })
        
        return output
    
    def get_stats(self) -> Dict:
        """Return retriever statistics."""
        return {
            "total_images": len(self.image_embeddings),
            "using_clip": self.use_clip,
            "embedding_dim": len(next(iter(self.image_embeddings.values()))) if self.image_embeddings else 0,
        }


class CrossModalComparison:
    """
    Compare traditional VSM text retrieval with cross-modal retrieval.
    """
    
    def __init__(self, text_retriever, image_retriever):
        self.text_retriever = text_retriever  # VSM or BM25
        self.image_retriever = image_retriever  # CLIP-based
    
    def compare_retrieval(self, query: str, top_k: int = 5) -> Dict:
        """
        Run both retrieval methods and compare results.
        
        Returns:
            Comparison dict with both result sets
        """
        from preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        tokens = preprocessor.segment(query)
        
        # Text retrieval
        text_results = self.text_retriever.search(tokens, top_k=top_k)
        
        # Image retrieval
        image_results = self.image_retriever.search(query, top_k=top_k)
        
        return {
            "query": query,
            "text_results": text_results,
            "image_results": image_results,
            "text_count": len(text_results),
            "image_count": len(image_results),
        }
    
    def print_comparison(self, query: str, top_k: int = 5):
        """Print formatted comparison of retrieval methods."""
        result = self.compare_retrieval(query, top_k)
        
        print(f"\n{'='*70}")
        print(f"跨模态检索对比: \"{query}\"")
        print(f"{'='*70}")
        
        print("\n[传统文本检索 - VSM/BM25]")
        for i, r in enumerate(result["text_results"], 1):
            print(f"  {i}. [{r['score']:.4f}] {r['title'][:50]}")
        
        print("\n[跨模态图像检索 - CLIP]")
        for i, r in enumerate(result["image_results"], 1):
            print(f"  {i}. [{r['score']:.4f}] {r['image_id']} ({r['width']}x{r['height']})")
        
        print(f"\n{'='*70}")


# Sample images for demonstration
def create_sample_images():
    """Create sample images for cross-modal retrieval demo."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Create simple colored images representing different concepts
    sample_specs = [
        ("ai_chip.jpg", (100, 100, 200), "AI chip technology"),  # Blue = tech
        ("nature_forest.jpg", (50, 150, 50), "Green forest nature"),  # Green = nature
        ("city_skyline.jpg", (80, 80, 80), "Urban city skyline"),  # Gray = city
        ("food_dish.jpg", (200, 100, 50), "Delicious food"),  # Orange = food
        ("robot_arm.jpg", (150, 150, 200), "Industrial robot"),  # Light blue = tech
        ("solar_panel.jpg", (50, 100, 150), "Solar energy green"),  # Blue-green = energy
        ("data_center.jpg", (30, 30, 80), "Data center servers"),  # Dark blue = tech
        ("electric_car.jpg", (100, 200, 100), "Electric vehicle"),  # Green = eco/tech
    ]
    
    for filename, color, description in sample_specs:
        filepath = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(filepath):
            # Create simple colored image
            img = Image.new('RGB', (224, 224), color)
            img.save(filepath, quality=85)
            print(f"  Created: {filename} - {description}")
    
    print(f"[Multimodal] Sample images ready in {IMAGE_DIR}")


if __name__ == "__main__":
    # Demo
    print("Cross-Modal Retrieval Demo")
    print("=" * 50)
    
    # Create sample images
    create_sample_images()
    
    # Initialize retriever
    retriever = CLIPImageRetriever(use_clip=False)  # Use fallback for demo
    retriever.index_images()
    
    # Test queries
    queries = [
        "人工智能芯片技术",
        "绿色能源环保",
        "城市建筑风景",
        "美食烹饪",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['image_id']}")

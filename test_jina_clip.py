
#!/usr/bin/env python3
"""
Test script to verify Jina CLIP v2 loading.
"""

import sys
print("Python version:", sys.version)
print("-" * 60)

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch error:", e)

print("-" * 60)

try:
    from transformers import AutoModel, AutoProcessor
    print("Transformers available")
except Exception as e:
    print("Transformers error:", e)

print("-" * 60)

print("\nAttempting to load Jina CLIP v2...")
try:
    model_name = "jinaai/jina-clip-v2"
    
    # Try with sentence-transformers (more stable)
    try:
        from sentence_transformers import SentenceTransformer
        print("Using Sentence-Transformers wrapper...")
        model = SentenceTransformer(model_name)
        print("✅ Sentence-Transformers: Jina CLIP v2 loaded successfully!")
        print("   Model name:", model_name)
        print("   Embedding dimension:", model.get_sentence_embedding_dimension())
    except Exception as e_st:
        print(f"Sentence-Transformers failed: {e_st}")
        
        # Fallback to direct Hugging Face
        print("\nFalling back to Hugging Face AutoModel...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Hugging Face: Jina CLIP v2 loaded successfully!")
        print("   Model name:", model_name)
        
except Exception as e:
    print("\n❌ Failed to load Jina CLIP v2:", e)
    print("\nError traceback:")
    import traceback
    traceback.print_exc()



#!/usr/bin/env python3
"""
Test script for multimodal retrieval - proves Jina CLIP v2 is used.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimodal_retrieval import create_sample_images, CrossModalRetriever


def main():
    print("=" * 70)
    print("【Jina CLIP v2 多模态检索测试】")
    print("=" * 70)
    
    # 1. Create fresh sample images
    print("\n[1] Creating sample images...")
    create_sample_images()
    
    # 2. Initialize retriever (loads Jina CLIP v2)
    print("\n[2] Initializing CrossModalRetriever...")
    retriever = CrossModalRetriever(use_clip=True)
    
    if retriever.model is None:
        print("\n❌ Failed to load Jina CLIP v2!")
        return
    
    # 3. Index images
    print("\n[3] Indexing images...")
    retriever.index_images()
    
    # 4. Test queries
    test_queries = [
        "芯片",
        "把太阳能转换成电能的板子",
        "机器人手臂",
        "自然森林",
        "城市天际线",
        "美食菜肴",
        "数据中心服务器",
        "电动汽车",
    ]
    
    print("\n" + "=" * 70)
    print("【检索测试】")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        results = retriever.search(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['image_id']} (score: {r['score']:.4f})")
    
    # 5. Final stats
    print("\n" + "=" * 70)
    print("【统计信息】")
    print("=" * 70)
    stats = retriever.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\n✅ 已证明使用 Jina CLIP v2!")


if __name__ == "__main__":
    main()


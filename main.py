import json
import os
import sys
import time

from config import DOCUMENTS_FILE, INDEX_FILE
from crawler import WebCrawler
from async_crawler import run_async_crawl
from preprocessor import TextPreprocessor
from inverted_index import InvertedIndex
from vsm import VectorSpaceModel
from bm25 import BM25Model, AlgorithmComparator
from evaluator import RetrievalEvaluator
from multimodal_retrieval import CLIPImageRetriever, CrossModalComparison, create_sample_images
from visualization import generate_all_visualizations


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           信息检索系统 - Information Retrieval System            ║
║  跨模态检索 + BM25优化 + 高并发爬虫 + 可视化评价                  ║
║  Cross-Modal + BM25 + Async Crawler + Visualization             ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_menu():
    print("""
请选择操作:
  [1] 异步爬取文档 (Async Web Crawling) - 高性能并发
  [2] 构建索引 (Build Index)
  [3] 搜索文档 - TF-IDF/VSM
  [4] 搜索文档 - BM25 (优化算法)
  [5] 算法对比 (TF-IDF vs BM25)
  [6] 跨模态图像检索 (CLIP Text-to-Image)
  [7] 交互式查询 (Interactive Query)
  [8] 人工评价检索结果 (Evaluation)
  [9] 生成可视化图表 (Generate Charts)
  [10] 显示系统状态 (System Status)
  [0] 退出 (Exit)
""")


def load_documents():
    if not os.path.exists(DOCUMENTS_FILE):
        return None
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index():
    idx = InvertedIndex()
    if idx.load():
        return idx
    return None


def print_search_results(results, show_highlights=True):
    if not results:
        print("\n  未找到相关结果。")
        return
    print(f"\n  共找到 {len(results)} 个结果:\n")
    for i, r in enumerate(results):
        print(f"  {'─' * 64}")
        print(f"  [{i + 1}] 相关度: {r['score']:.4f}")
        print(f"      标题: {r['title']}")
        print(f"      摘要: {r['snippet'][:120]}")
        print(f"      日期: {r['date']}")
        print(f"      URL:  {r['url']}")
    print(f"  {'─' * 64}\n")


def do_async_crawl():
    """High-performance async crawling."""
    print("\n[ Async Crawling with asyncio + aiohttp ]")
    print("Max concurrent connections: 20")
    print("Expected speed: 50-100 pages/sec (vs 1-2 for sync)\n")
    
    start_time = time.time()
    docs = run_async_crawl()
    elapsed = time.time() - start_time
    
    print(f"\n爬取完成！")
    print(f"  总文档数: {len(docs)}")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  平均速度: {len(docs)/elapsed:.1f} 文档/秒")
    return docs


def do_build_index():
    """Build inverted index with TF-IDF and BM25 support."""
    print("\n[ Building Inverted Index ]")
    documents = load_documents()
    if not documents:
        print("错误: 没有找到文档，请先爬取文档！")
        return None
    
    preprocessor = TextPreprocessor()
    documents = preprocessor.process_documents(documents)
    
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    index = InvertedIndex()
    index.build(documents)
    index.save()
    
    stats = index.get_stats()
    print(f"\n索引构建完成！")
    print(f"  文档总数: {stats['total_documents']}")
    print(f"  词项总数: {stats['total_terms']}")
    print(f"  倒排记录总数: {stats['total_postings']}")
    print(f"  平均倒排记录数: {stats['avg_postings_per_term']:.2f}")
    print(f"\n支持算法: TF-IDF (VSM) 和 BM25")
    return index


def do_search_vsm(query_str, index=None, preprocessor=None, top_k=20):
    """Search using TF-IDF VSM."""
    if index is None:
        index = load_index()
        if index is None:
            print("错误: 索引未构建，请先构建索引！")
            return []
        documents = load_documents()
        if documents:
            index.documents = documents
    
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    tokens = preprocessor.segment(query_str)
    vsm = VectorSpaceModel(index)
    results = vsm.search(tokens, top_k=top_k)
    return results


def do_search_bm25(query_str, index=None, preprocessor=None, top_k=20):
    """Search using BM25 algorithm."""
    if index is None:
        index = load_index()
        if index is None:
            print("错误: 索引未构建，请先构建索引！")
            return []
        documents = load_documents()
        if documents:
            index.documents = documents
    
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    tokens = preprocessor.segment(query_str)
    bm25 = BM25Model(index)
    results = bm25.search(tokens, top_k=top_k)
    return results


def do_algorithm_comparison(index=None, preprocessor=None):
    """Compare TF-IDF vs BM25 on sample queries."""
    if index is None:
        index = load_index()
        if index is None:
            print("错误: 索引未构建，请先构建索引！")
            return
        documents = load_documents()
        if documents:
            index.documents = documents
    
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    vsm = VectorSpaceModel(index)
    bm25 = BM25Model(index)
    
    test_queries = [
        "人工智能技术发展",
        "5G通信网络建设",
        "新能源汽车电池",
        "芯片半导体制造",
        "大数据隐私安全",
    ]
    
    print("\n" + "=" * 70)
    print("  算法对比: TF-IDF vs BM25")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n查询: \"{query}\"")
        tokens = preprocessor.segment(query)
        
        vsm_results = vsm.search(tokens, top_k=3)
        bm25_results = bm25.search(tokens, top_k=3)
        
        print("  TF-IDF (VSM):")
        for i, r in enumerate(vsm_results[:3], 1):
            print(f"    {i}. [{r['score']:.4f}] {r['title'][:45]}")
        
        print("  BM25:")
        for i, r in enumerate(bm25_results[:3], 1):
            print(f"    {i}. [{r['score']:.4f}] {r['title'][:45]}")


def do_multimodal_search(query_str=None):
    """Cross-modal text-to-image retrieval."""
    print("\n[ Cross-Modal Text-to-Image Retrieval ]")
    print("Using CLIP or fallback encoder for semantic image search\n")
    
    # Initialize/create sample images
    create_sample_images()
    
    retriever = CLIPImageRetriever(use_clip=False)  # Fallback for compatibility
    retriever.index_images()
    
    if query_str is None:
        query_str = input("请输入图像检索查询（如：人工智能芯片、绿色能源）: ").strip()
    
    if not query_str:
        return
    
    print(f"\n查询: \"{query_str}\"")
    results = retriever.search(query_str, top_k=5)
    
    if not results:
        print("  未找到相关图像。")
        return
    
    print(f"\n  找到 {len(results)} 个相关图像:\n")
    for i, r in enumerate(results):
        print(f"  [{i + 1}] 相似度: {r['score']:.4f}")
        print(f"      图像: {r['image_id']}")
        print(f"      尺寸: {r['width']}x{r['height']}")
        print(f"      路径: {r['path']}")


def interactive_query(index=None, preprocessor=None):
    """Interactive query mode with algorithm selection."""
    if index is None:
        index = load_index()
        if index is None:
            print("错误: 索引未构建，请先构建索引！")
            return
        documents = load_documents()
        if documents:
            index.documents = documents
    
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    print("\n" + "=" * 70)
    print("  交互式查询模式")
    print("  输入 'vsm' 或 'bm25' 切换算法，输入 'quit' 退出")
    print("=" * 70)
    
    algorithm = "vsm"  # Default
    
    while True:
        print(f"\n当前算法: {algorithm.upper()}")
        query_str = input("查询> ").strip()
        
        if not query_str:
            continue
        if query_str.lower() in ("quit", "exit", "q"):
            break
        if query_str.lower() == "vsm":
            algorithm = "vsm"
            print("  已切换至 TF-IDF (VSM) 算法")
            continue
        if query_str.lower() == "bm25":
            algorithm = "bm25"
            print("  已切换至 BM25 算法")
            continue
        
        tokens = preprocessor.segment(query_str)
        print(f"分词结果: {' / '.join(tokens)}")
        
        if algorithm == "vsm":
            vsm = VectorSpaceModel(index)
            results = vsm.search(tokens, top_k=20)
        else:
            bm25 = BM25Model(index)
            results = bm25.search(tokens, top_k=20)
        
        print_search_results(results)


def do_evaluation(index=None, preprocessor=None):
    """Interactive evaluation with Precision@10 and Recall."""
    if index is None:
        index = load_index()
        if index is None:
            print("错误: 索引未构建，请先构建索引！")
            return
        documents = load_documents()
        if documents:
            index.documents = documents
    
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    vsm = VectorSpaceModel(index)
    evaluator = RetrievalEvaluator(vsm, preprocessor)
    evaluator.interactive_evaluation()


def do_generate_charts():
    """Generate all visualization charts for the report."""
    print("\n[ Generating Visualization Charts ]")
    print("Charts will be saved to data/charts/\n")
    
    # Load real comparison data if available
    index = load_index()
    if index:
        documents = load_documents()
        if documents:
            index.documents = documents
        stats = index.get_stats()
    else:
        stats = None
    
    generate_all_visualizations(index_stats=stats)
    print("\n图表生成完成！")


def show_status():
    """Display comprehensive system status."""
    print("\n[ System Status ]\n")
    
    # Documents
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"  文档存储: data/documents.json")
        print(f"  文档数量: {len(documents)}")
        if documents:
            sample = documents[0]
            print(f"  示例标题: {sample.get('title', 'N/A')[:50]}")
    else:
        print(f"  文档存储: 未找到")
    
    # Index
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        print(f"\n  索引文件: data/inverted_index.json")
        print(f"  索引文档数: {idx_data.get('doc_count', 0)}")
        print(f"  词项数: {len(idx_data.get('index', {}))}")
    else:
        print(f"\n  索引文件: 未找到")
    
    # Storage
    if os.path.exists(DOCUMENTS_FILE):
        doc_size = os.path.getsize(DOCUMENTS_FILE)
        idx_size = os.path.getsize(INDEX_FILE) if os.path.exists(INDEX_FILE) else 0
        print(f"\n  存储占用:")
        print(f"    文档文件: {doc_size / 1024:.1f} KB")
        print(f"    索引文件: {idx_size / 1024:.1f} KB")
    
    # Algorithms
    print(f"\n  支持算法:")
    print(f"    ✓ TF-IDF (VSM)")
    print(f"    ✓ BM25 (优化)")
    print(f"    ✓ 跨模态检索 (CLIP)")
    
    # Performance
    print(f"\n  性能特性:")
    print(f"    ✓ 异步并发爬虫 (asyncio + aiohttp)")
    print(f"    ✓ 倒排索引压缩")
    print(f"    ✓ 可视化图表生成")


def main():
    print_banner()
    
    index = None
    preprocessor = TextPreprocessor()
    
    # Auto-load if data exists
    if os.path.exists(DOCUMENTS_FILE) and os.path.exists(INDEX_FILE):
        print("[Info] 检测到已有数据，正在加载...")
        index = load_index()
        documents = load_documents()
        if index and documents:
            index.documents = documents
            print(f"[Info] 已加载 {len(documents)} 篇文档和索引。")
            print(f"[Info] 支持算法: TF-IDF, BM25, 跨模态检索\n")
    
    while True:
        print_menu()
        choice = input("请输入选项 [0-10]: ").strip()
        
        if choice == "1":
            do_async_crawl()
        elif choice == "2":
            index = do_build_index()
        elif choice == "3":
            query_str = input("\n请输入查询关键词: ").strip()
            if query_str:
                results = do_search_vsm(query_str, index, preprocessor)
                print_search_results(results)
        elif choice == "4":
            query_str = input("\n请输入查询关键词 (BM25): ").strip()
            if query_str:
                results = do_search_bm25(query_str, index, preprocessor)
                print_search_results(results)
        elif choice == "5":
            do_algorithm_comparison(index, preprocessor)
        elif choice == "6":
            do_multimodal_search()
        elif choice == "7":
            interactive_query(index, preprocessor)
        elif choice == "8":
            do_evaluation(index, preprocessor)
        elif choice == "9":
            do_generate_charts()
        elif choice == "10":
            show_status()
        elif choice == "0":
            print("\n感谢使用信息检索系统，再见！")
            sys.exit(0)
        else:
            print("\n无效选项，请重新选择。")


if __name__ == "__main__":
    main()

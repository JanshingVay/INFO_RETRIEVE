import json
import os
import sys
import time

from config import DOCUMENTS_FILE, INDEX_FILE, CRAWL_MIN_DOCS
from async_crawler import run_async_crawl, get_document_count, needs_crawling
from preprocessor import TextPreprocessor
from inverted_index import InvertedIndex
from vsm import VectorSpaceModel
from bm25 import BM25Model
from evaluator import RetrievalEvaluator
from multimodal_retrieval import CLIPImageRetriever, create_sample_images
from visualization import generate_all_visualizations
from relevance_feedback import FeedbackStore


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           信息检索系统 - Information Retrieval System              ║
║   跨模态检索 + BM25优化 + 高并发爬虫 + 可视化评价                      ║
║  Cross-Modal + BM25 + Async Crawler + Visualization              ║
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
        fb_mark = " ⭐(反馈优选)" if r.get("feedback_boosted") else ""
        print(f"  {'─' * 64}")
        print(f"  [{i + 1}]{fb_mark} 相关度: {r['score']:.4f}")
        print(f"      标题: {r['title']}")
        print(f"      摘要: {r['snippet'][:120]}")
        print(f"      日期: {r['date']}")
        print(f"      URL:  {r['url']}")
    print(f"  {'─' * 64}\n")


def rate_results_interactive(results, query_str, feedback_store):
    """Let user rate search results and save feedback."""
    print("\n  ── 对结果评分 ──────────────────────")
    print("  格式: 1=5,3=1,5=4   (编号=分数)")
    print("  分数: 5=非常相关 4=相关 3=一般 2=不太相关 1=完全不相关")
    print("  直接回车或输入 skip / q 跳过评分")
    user_input = input("  评分> ").strip()

    if not user_input or user_input.lower() in ("skip", "q", "quit", "exit"):
        return False

    ratings = {}
    for part in user_input.split(","):
        part = part.strip()
        if "=" in part:
            try:
                idx_str, rating_str = part.split("=", 1)
                idx = int(idx_str.strip())
                rating = int(rating_str.strip())
                if 1 <= idx <= len(results) and 1 <= rating <= 5:
                    doc_id = results[idx - 1]["id"]
                    ratings[doc_id] = rating
            except (ValueError, IndexError):
                pass

    if ratings:
        feedback_store.record_batch(ratings, query_str)
        liked = sum(1 for r in ratings.values() if r >= 4)
        disliked = sum(1 for r in ratings.values() if r <= 2)
        print(f"  已记录 {len(ratings)} 条评分 (相关:{liked}, 不相关:{disliked})")
        print(f"  反馈已保存，下次搜索时自动优化排序。")
        return True
    print("  未识别有效评分格式，已跳过。")
    return False


def ensure_data_available():
    """Auto-crawl if documents are missing or below minimum threshold."""
    doc_count = get_document_count()
    if doc_count >= CRAWL_MIN_DOCS:
        print(f"[Auto] 已有 {doc_count} 篇文档 (≥{CRAWL_MIN_DOCS})，无需爬取。")
        return True

    print(f"\n[Auto] 文档数 {doc_count} 不足 (需要≥{CRAWL_MIN_DOCS})，自动开始爬取...")
    print(f"[Auto] 将尝试从 {len(CRAWL_SOURCES) if 'CRAWL_SOURCES' in dir() else '多个'} 个数据源采集。")
    print(f"[Auto] 注意：所有数据均来自真实网络爬取，不包含伪造数据。\n")

    docs = run_async_crawl()
    if len(docs) >= CRAWL_MIN_DOCS:
        print(f"[Auto] OK: 已采集 {len(docs)} 篇文档。")
        return True
    else:
        print(f"[Auto] 警告: 仅爬取到 {len(docs)} 篇，不满足 {CRAWL_MIN_DOCS} 篇最小要求。")
        print(f"[Auto] 可能原因: 网络不通或目标网站结构变化。请检查网络后重试。")
        return len(docs) >= 100  # Allow if at least 100


def do_async_crawl():
    """High-performance async crawling."""
    print("\n[ Async Crawling with asyncio + aiohttp ]")
    print("Max concurrent connections: 20")
    print("所有数据均来自真实网络爬取，无伪造数据。\n")

    start_time = time.time()
    docs = run_async_crawl()
    elapsed = time.time() - start_time

    print(f"\n爬取完成！")
    print(f"  总文档数: {len(docs)}")
    print(f"  耗时: {elapsed:.2f} 秒")
    if elapsed > 0:
        print(f"  平均速度: {len(docs)/elapsed:.1f} 文档/秒")
    return docs


def do_build_index():
    """Build inverted index with TF-IDF and BM25 support."""
    print("\n[ Building Inverted Index ]")

    # Auto-crawl if needed
    if needs_crawling():
        print("[Auto] 文档不足，先爬取...")
        ensure_data_available()

    documents = load_documents()
    if not documents:
        print("错误: 没有找到文档，请先爬取文档 (选项1)！")
        return None

    print(f"  处理 {len(documents)} 篇文档...")

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
    """Search using TF-IDF VSM (plain search, no feedback)."""
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
    print_search_results(results)
    return results


def do_search_bm25(query_str, index=None, preprocessor=None, top_k=20):
    """Search using BM25 algorithm (plain search, no feedback)."""
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
    print_search_results(results)
    return results


def do_algorithm_comparison(index=None, preprocessor=None):
    """Interactive TF-IDF vs BM25 comparison — user types own queries."""
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

    print("\n" + "=" * 70)
    print("  算法对比: TF-IDF vs BM25")
    print("  输入你的查询词即可对比，输入 quit 返回菜单")
    print("=" * 70)

    while True:
        print()
        query_str = input("对比查询> ").strip()
        if not query_str:
            continue
        if query_str.lower() in ("quit", "exit", "q"):
            break

        print(f"\n查询: \"{query_str}\"")
        tokens = preprocessor.segment(query_str)
        print(f"分词: {' / '.join(tokens)}")

        vsm_results = vsm.search(tokens, top_k=3)
        bm25_results = bm25.search(tokens, top_k=3)

        print("\n  TF-IDF (VSM):")
        for i, r in enumerate(vsm_results[:3], 1):
            print(f"    {i}. [{r['score']:.4f}] {r['title'][:45]}")

        print("\n  BM25:")
        for i, r in enumerate(bm25_results[:3], 1):
            print(f"    {i}. [{r['score']:.4f}] {r['title'][:45]}")

        print(f"\n  {'─' * 50}")


def do_multimodal_search(query_str=None):
    """Cross-modal text-to-image retrieval."""
    print("\n[ Cross-Modal Text-to-Image Retrieval ]")
    print("Using CLIP or fallback encoder for semantic image search\n")

    create_sample_images()

    retriever = CLIPImageRetriever(use_clip=True)
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


def interactive_query(index=None, preprocessor=None, feedback_store=None):
    """Interactive query mode with algorithm selection and feedback loop."""
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

    if feedback_store is None:
        feedback_store = FeedbackStore()

    vsm = VectorSpaceModel(index)
    bm25 = BM25Model(index)

    print("\n" + "=" * 70)
    print("  交互式查询模式 (支持反馈优化)")
    print("  输入 'vsm' / 'bm25' 切换算法  |  'fb' 反馈模式")
    print("  输入 'rate 1=5,3=2' 评分结果 |  'stats' 反馈统计")
    print("  输入 'quit' 退出")
    print("=" * 70)

    algorithm = "vsm"
    use_feedback = False

    while True:
        fb_stats = feedback_store.get_feedback_stats()
        fb_info = f" [反馈: {fb_stats['total_ratings']}条]" if fb_stats['total_ratings'] > 0 else ""
        print(f"\n当前算法: {algorithm.upper()}{fb_info}"
              f"{' + 反馈优化' if use_feedback else ''}")
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
        if query_str.lower() == "fb":
            use_feedback = not use_feedback
            state = "开启" if use_feedback else "关闭"
            print(f"  反馈优化已{state}")
            continue
        if query_str.lower() == "stats":
            stats = feedback_store.get_feedback_stats()
            print(f"\n  反馈统计:")
            print(f"    总评分: {stats['total_ratings']} | 已评文档: {stats['rated_docs']}")
            print(f"    平均分: {stats['avg_rating']:.2f}/5 | 收藏: {stats['liked_docs']}篇")
            print(f"    有反馈查询: {stats['queries_with_feedback']} | 更新: {stats['last_updated'][:19]}")
            continue
        if query_str.lower().startswith("rate"):
            print("  请先执行查询后再评分。")
            continue

        tokens = preprocessor.segment(query_str)
        print(f"分词结果: {' / '.join(tokens)}")

        if algorithm == "vsm":
            if use_feedback:
                results = vsm.search_with_feedback(tokens, feedback_store, top_k=20)
            else:
                results = vsm.search(tokens, top_k=20)
        else:
            if use_feedback:
                results = bm25.search_with_feedback(tokens, feedback_store, top_k=20)
            else:
                results = bm25.search(tokens, top_k=20)

        print_search_results(results)

        if results and use_feedback:
            rated = rate_results_interactive(results, query_str, feedback_store)
            if rated:
                print(f"\n  *** 反馈已应用！输入查询再次搜索查看优化效果 ***")
        elif results:
            print("  提示: 输入 'fb' 开启反馈模式后可对结果评分并优化搜索。")


def do_evaluation(index=None, preprocessor=None, feedback_store=None):
    """Interactive evaluation with Precision@10 and Recall + feedback loop."""
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

    if feedback_store is None:
        feedback_store = FeedbackStore()

    vsm = VectorSpaceModel(index)
    evaluator = RetrievalEvaluator(vsm, preprocessor, feedback_store)
    evaluator.interactive_evaluation()


def do_generate_charts():
    """Generate all visualization charts for the report."""
    print("\n[ Generating Visualization Charts ]")
    print("Charts will be saved to data/charts/\n")

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

    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"  文档存储: data/documents.json")
        print(f"  文档数量: {len(documents)}")
        if documents:
            sample = documents[0]
            print(f"  示例标题: {sample.get('title', 'N/A')[:50]}")
    else:
        print(f"  文档存储: 未找到 (需要≥{CRAWL_MIN_DOCS}篇)")

    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        print(f"\n  索引文件: data/inverted_index.json")
        print(f"  索引文档数: {idx_data.get('doc_count', 0)}")
        print(f"  词项数: {len(idx_data.get('index', {}))}")
    else:
        print(f"\n  索引文件: 未找到")

    if os.path.exists(DOCUMENTS_FILE):
        doc_size = os.path.getsize(DOCUMENTS_FILE)
        idx_size = os.path.getsize(INDEX_FILE) if os.path.exists(INDEX_FILE) else 0
        print(f"\n  存储占用:")
        print(f"    文档文件: {doc_size / 1024:.1f} KB")
        print(f"    索引文件: {idx_size / 1024:.1f} KB")

    print(f"\n  支持算法:")
    print(f"    ✓ TF-IDF (VSM)")
    print(f"    ✓ BM25 (优化)")
    print(f"    ✓ 跨模态检索 (CLIP)")
    print(f"    ✓ 相关反馈 (Rocchio)")

    print(f"\n  性能特性:")
    print(f"    ✓ 异步并发爬虫 (asyncio + aiohttp)")
    print(f"    ✓ 10个数据源配置 (IT之家/36氪/新华网/人民网/凤凰...)")


def main():
    print_banner()

    # ---- Auto-crawl check ----
    doc_count = get_document_count()
    if doc_count < CRAWL_MIN_DOCS:
        print(f"[Auto] 当前仅 {doc_count} 篇文档，需要≥{CRAWL_MIN_DOCS}篇。")
        print(f"[Auto] 所有数据均来自真实网络爬取，不包含伪造数据。")
        print(f"[Auto] 正在自动启动爬虫...\n")
        try:
            if ensure_data_available():
                doc_count = get_document_count()
            else:
                print("[Auto] 警告: 文档不足，部分功能可能受限。")
        except KeyboardInterrupt:
            print("\n[Auto] 爬虫已被中断，使用已有数据继续。")
            doc_count = get_document_count()
    else:
        print(f"[Info] 已有 {doc_count} 篇文档 (≥{CRAWL_MIN_DOCS})。")

    index = None
    preprocessor = TextPreprocessor()
    feedback_store = FeedbackStore()

    fb_stats = feedback_store.get_feedback_stats()
    if fb_stats['total_ratings'] > 0:
        print(f"[Info] 已加载反馈数据: {fb_stats['total_ratings']} 条评分, "
              f"{fb_stats['rated_docs']} 篇文档")

    if os.path.exists(DOCUMENTS_FILE) and os.path.exists(INDEX_FILE):
        print("[Info] 检测到已有索引，正在加载...")
        index = load_index()
        documents = load_documents()
        if index and documents:
            index.documents = documents
            print(f"[Info] 已加载 {len(documents)} 篇文档和索引。")
            algos = "TF-IDF, BM25, CLIP跨模态"
            fb_info = "✓" if fb_stats['total_ratings'] > 0 else "无"
            print(f"[Info] 算法: {algos} (反馈: {fb_info})\n")

    while True:
        try:
            print_menu()
            choice = input("请输入选项 [0-10]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            choice = "0"

        if choice == "1":
            do_async_crawl()
        elif choice == "2":
            index = do_build_index()
        elif choice == "3":
            query_str = input("\n请输入查询关键词: ").strip()
            if query_str:
                do_search_vsm(query_str, index, preprocessor)
        elif choice == "4":
            query_str = input("\n请输入查询关键词 (BM25): ").strip()
            if query_str:
                do_search_bm25(query_str, index, preprocessor)
        elif choice == "5":
            do_algorithm_comparison(index, preprocessor)
        elif choice == "6":
            do_multimodal_search()
        elif choice == "7":
            interactive_query(index, preprocessor, feedback_store)
        elif choice == "8":
            do_evaluation(index, preprocessor, feedback_store)
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已退出。")
        sys.exit(0)

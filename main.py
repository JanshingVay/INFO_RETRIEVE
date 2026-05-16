import json
import os
import sys

from config import DOCUMENTS_FILE, INDEX_FILE
from crawler import WebCrawler
from preprocessor import TextPreprocessor
from inverted_index import InvertedIndex
from vsm import VectorSpaceModel
from evaluator import RetrievalEvaluator


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           信息检索系统 - Information Retrieval System            ║
║         基于向量空间模型 (VSM) + TF-IDF + 倒排索引               ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_menu():
    print("""
请选择操作:
  [1] 爬取文档 (Web Crawling)
  [2] 构建索引 (Build Index)
  [3] 搜索文档 (Search)
  [4] 交互式查询 (Interactive Query)
  [5] 人工评价检索结果 (Evaluation)
  [6] 显示系统状态 (System Status)
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


def do_crawl():
    print("\n[ Crawling documents from web sources... ]\n")
    crawler = WebCrawler()
    docs = crawler.crawl_all()
    print(f"\n爬取完成！共获取 {len(docs)} 篇文档。")
    return docs


def do_build_index():
    print("\n[ Building inverted index... ]")
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
    return index


def do_search(query_str, index=None, preprocessor=None, top_k=20):
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


def interactive_query(index=None, preprocessor=None):
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

    print("\n" + "=" * 70)
    print("  交互式查询模式")
    print("  输入自然语言查询，输入 'quit' 或 'exit' 退出")
    print("=" * 70)

    while True:
        print()
        query_str = input("查询> ").strip()
        if not query_str:
            continue
        if query_str.lower() in ("quit", "exit", "q"):
            break

        tokens = preprocessor.segment(query_str)
        print(f"分词结果: {' '.join(tokens)}")

        results = vsm.search(tokens, top_k=20)
        print_search_results(results)


def do_evaluation(index=None, preprocessor=None):
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


def show_status():
    print("\n[ System Status ]\n")

    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"  文档存储: data/documents.json")
        print(f"  文档数量: {len(documents)}")
        if documents:
            sample = documents[0]
            print(f"  示例标题: {sample.get('title', 'N/A')[:50]}")
            print(f"  示例URL:  {sample.get('url', 'N/A')[:60]}")
    else:
        print(f"  文档存储: 未找到 (请先爬取文档)")

    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        print(f"\n  索引文件: data/inverted_index.json")
        print(f"  索引文档数: {idx_data.get('doc_count', 0)}")
        print(f"  词项数: {len(idx_data.get('index', {}))}")
    else:
        print(f"\n  索引文件: 未找到 (请先构建索引)")

    if os.path.exists(DOCUMENTS_FILE):
        doc_size = os.path.getsize(DOCUMENTS_FILE)
        idx_size = os.path.getsize(INDEX_FILE) if os.path.exists(INDEX_FILE) else 0
        print(f"\n  存储占用:")
        print(f"    文档文件: {doc_size / 1024:.1f} KB")
        print(f"    索引文件: {idx_size / 1024:.1f} KB")
        print(f"    总计: {(doc_size + idx_size) / 1024:.1f} KB")


def main():
    print_banner()

    index = None
    preprocessor = TextPreprocessor()

    if os.path.exists(DOCUMENTS_FILE) and os.path.exists(INDEX_FILE):
        print("[Info] 检测到已有数据，正在加载...")
        index = load_index()
        documents = load_documents()
        if index and documents:
            index.documents = documents
            print(f"[Info] 已加载 {len(documents)} 篇文档和索引。")

    while True:
        print_menu()
        choice = input("请输入选项 [0-6]: ").strip()

        if choice == "1":
            do_crawl()
        elif choice == "2":
            index = do_build_index()
        elif choice == "3":
            query_str = input("\n请输入查询关键词: ").strip()
            if query_str:
                results = do_search(query_str, index, preprocessor)
                print_search_results(results)
        elif choice == "4":
            interactive_query(index, preprocessor)
        elif choice == "5":
            do_evaluation(index, preprocessor)
        elif choice == "6":
            show_status()
        elif choice == "0":
            print("\n感谢使用信息检索系统，再见！")
            sys.exit(0)
        else:
            print("\n无效选项，请重新选择。")


if __name__ == "__main__":
    main()

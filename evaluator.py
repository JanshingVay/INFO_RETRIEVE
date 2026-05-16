import json
import os
from config import EVAL_QUERIES_FILE


DEFAULT_EVAL_QUERIES = [
    {
        "query": "人工智能技术发展",
        "description": "关于人工智能技术发展和趋势的文章",
        "relevant_docs": []
    },
    {
        "query": "5G通信网络",
        "description": "关于5G通信技术和网络建设的文章",
        "relevant_docs": []
    },
    {
        "query": "新能源汽车",
        "description": "关于新能源汽车、电动汽车的文章",
        "relevant_docs": []
    },
    {
        "query": "芯片半导体",
        "description": "关于芯片制造、半导体产业的文章",
        "relevant_docs": []
    },
    {
        "query": "大数据云计算",
        "description": "关于大数据处理、云计算技术的文章",
        "relevant_docs": []
    },
    {
        "query": "航天探索",
        "description": "关于航天技术、太空探索的文章",
        "relevant_docs": []
    },
    {
        "query": "环境保护",
        "description": "关于环境保护、碳中和、可持续发展的文章",
        "relevant_docs": []
    },
    {
        "query": "医疗健康",
        "description": "关于医疗技术、健康科学的文章",
        "relevant_docs": []
    },
    {
        "query": "区块链技术",
        "description": "关于区块链、加密货币技术的文章",
        "relevant_docs": []
    },
    {
        "query": "机器人自动化",
        "description": "关于机器人技术、工业自动化的文章",
        "relevant_docs": []
    },
]


class RetrievalEvaluator:
    """Manual evaluation of retrieval results."""

    def __init__(self, vsm_model, preprocessor):
        self.vsm = vsm_model
        self.preprocessor = preprocessor
        self.eval_queries = []
        self._load_eval_queries()

    def _load_eval_queries(self):
        if os.path.exists(EVAL_QUERIES_FILE):
            with open(EVAL_QUERIES_FILE, "r", encoding="utf-8") as f:
                self.eval_queries = json.load(f)
        else:
            self.eval_queries = DEFAULT_EVAL_QUERIES
            self._save_eval_queries()

    def _save_eval_queries(self):
        os.makedirs(os.path.dirname(EVAL_QUERIES_FILE), exist_ok=True)
        with open(EVAL_QUERIES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.eval_queries, f, ensure_ascii=False, indent=2)

    def run_search(self, query_str, top_k=10):
        tokens = self.preprocessor.segment(query_str)
        return self.vsm.search(tokens, top_k=top_k)

    def interactive_evaluation(self):
        """Interactive interface for manually evaluating retrieval results."""
        print("\n" + "=" * 70)
        print("  检索结果人工评价系统")
        print("=" * 70)
        print(f"共 {len(self.eval_queries)} 个评测查询")
        print()

        for i, eq in enumerate(self.eval_queries):
            print(f"\n{'─' * 70}")
            print(f"查询 {i + 1}/{len(self.eval_queries)}: {eq['query']}")
            print(f"描述: {eq['description']}")
            print(f"{'─' * 70}")

            results = self.run_search(eq["query"], top_k=10)
            if not results:
                print("  无检索结果。")
                continue

            print(f"\n检索结果 (Top {len(results)}):")
            for j, r in enumerate(results):
                print(f"\n  [{j + 1}] 相关度: {r['score']:.4f}")
                print(f"      标题: {r['title'][:60]}")
                print(f"      摘要: {r['snippet'][:80]}")
                print(f"      日期: {r['date']}")
                print(f"      URL: {r['url'][:80]}")

            print("\n请评价以上结果的相关性（输入相关文档编号，用逗号分隔，如 1,3,5）：")
            print("或者输入 'skip' 跳过，'quit' 退出评价。")
            user_input = input(">>> ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "skip":
                continue
            elif user_input:
                try:
                    relevant = [int(x.strip()) for x in user_input.split(",") if x.strip().isdigit()]
                    eq["relevant_docs"] = [results[i - 1]["id"] for i in relevant if 1 <= i <= len(results)]
                    print(f"  已记录 {len(eq['relevant_docs'])} 个相关文档。")
                except (ValueError, IndexError):
                    print("  输入格式错误，跳过。")

            self._save_eval_queries()

        self._compute_metrics()

    def _compute_metrics(self):
        """Compute precision@k for each query."""
        print("\n" + "=" * 70)
        print("  评测结果汇总")
        print("=" * 70)

        total_precision = 0
        total_recall = 0
        valid = 0

        for eq in self.eval_queries:
            if not eq["relevant_docs"]:
                continue
            valid += 1
            results = self.run_search(eq["query"], top_k=10)
            retrieved_docs = [r["id"] for r in results]
            relevant_set = set(eq["relevant_docs"])
            retrieved_set = set(retrieved_docs)

            hits = relevant_set & retrieved_set
            precision = len(hits) / len(retrieved_set) if retrieved_set else 0
            recall = len(hits) / len(relevant_set) if relevant_set else 0

            total_precision += precision
            total_recall += recall

            print(f"\n查询: {eq['query']}")
            print(f"  相关文档: {len(relevant_set)}, 命中: {len(hits)}")
            print(f"  准确率 (Precision@10): {precision:.2%}")
            print(f"  召回率 (Recall): {recall:.2%}")

        if valid > 0:
            print(f"\n{'─' * 50}")
            print(f"平均准确率 (Avg Precision@10): {total_precision / valid:.2%}")
            print(f"平均召回率 (Avg Recall): {total_recall / valid:.2%}")

    def show_metrics_summary(self):
        """Display evaluation metrics summary."""
        self._compute_metrics()


if __name__ == "__main__":
    print("Evaluation module - import and use with VSM model.")

import json
import os
from config import EVAL_QUERIES_FILE
from relevance_feedback import FeedbackStore


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
    """Manual evaluation of retrieval results with feedback-driven optimization."""

    def __init__(self, vsm_model, preprocessor, feedback_store=None):
        self.vsm = vsm_model
        self.preprocessor = preprocessor
        self.feedback = feedback_store or FeedbackStore()
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

    def run_search_with_feedback(self, query_str, top_k=10):
        tokens = self.preprocessor.segment(query_str)
        return self.vsm.search_with_feedback(tokens, self.feedback, top_k=top_k)

    def _rate_results_interactive(self, results, query_str):
        """Interactive rating interface for a set of results.
        
        Returns dict: {doc_id: rating (1-5)}
        """
        print(f"\n查询: \"{query_str}\"")
        print(f"检索结果 (Top {len(results)}):")
        for j, r in enumerate(results):
            fb_mark = " ⭐" if r.get("feedback_boosted") else ""
            print(f"\n  [{j + 1}]{fb_mark} 相关度: {r['score']:.4f}")
            print(f"      标题: {r['title'][:60]}")
            print(f"      摘要: {r['snippet'][:80]}")
            print(f"      日期: {r['date']}")
            print(f"      URL: {r['url'][:80]}")

        print("\n请输入评分（格式如: 1=5, 3=1, 5=4）")
        print("  5=非常相关  4=相关  3=一般  2=不太相关  1=不相关")
        print("  或者输入 'skip' 跳过，'quit' 退出。")
        user_input = input("评分> ").strip()

        ratings = {}
        if user_input.lower() == "quit":
            return None
        elif user_input.lower() == "skip":
            return ratings

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
        return ratings

    def interactive_evaluation(self):
        """Systematic evaluation: preset queries + metrics (Precision@10, Recall).
        
        Differs from interactive mode (option 7):
        - 10 preset queries covering diverse domains (AI, 5G, EV, chips, etc.)
        - Uses PURE search (no historical feedback contamination)
        - Reports Precision@10 and Recall per query
        - Saves evaluation data for report generation
        """
        print("\n" + "=" * 70)
        print("  检索结果人工评价系统")
        print("  10个预设查询 · 纯净检索 · 自动计算Precision/Recall")
        print("=" * 70)

        fb_stats = self.feedback.get_feedback_stats()
        if fb_stats['total_ratings'] > 0:
            print(f"(已有反馈数据: {fb_stats['total_ratings']} 条，评价格式与交互模式共享)")

        # Show preset queries
        print(f"\n当前评测查询 ({len(self.eval_queries)} 个):")
        for i, eq in enumerate(self.eval_queries):
            has_fb = " [已有评价]" if eq.get("relevant_docs") else ""
            print(f"  [{i + 1}] {eq['query']}{has_fb}")

        print(f"\n{'─' * 70}")
        print("模式: [1] 逐题评价 + 反馈优化  [2] 自定义查询  [3] 反馈效果对比  [4] 查看统计")
        mode = input("选择模式> ").strip()

        if mode == "2":
            query_str = input("输入自定义查询: ").strip()
            if query_str:
                self._evaluate_custom_query(query_str)
        elif mode == "3":
            self._compare_feedback_effect()
        elif mode == "4":
            self._show_feedback_stats()
        else:
            self._evaluate_preset_queries()

    def _evaluate_preset_queries(self):
        """Evaluate all preset queries WITHOUT prior feedback bias."""
        print("\n  [评价模式] 使用纯净搜索结果（不受历史反馈影响）")
        for i, eq in enumerate(self.eval_queries):
            print(f"\n{'─' * 70}")
            print(f"查询 {i + 1}/{len(self.eval_queries)}: {eq['query']}")
            print(f"描述: {eq['description']}")
            print(f"{'─' * 70}")

            results = self.run_search(eq["query"], top_k=10)
            if not results:
                print("  无检索结果。")
                continue

            ratings = self._rate_results_interactive(results, eq["query"])
            if ratings is None:
                break
            if not ratings:
                continue

            self.feedback.record_batch(ratings, eq["query"])
            relevant = [doc_id for doc_id, r in ratings.items() if r >= 4]
            eq["relevant_docs"] = relevant
            self._save_eval_queries()

            relevant_set = set(relevant)
            retrieved = set(r["id"] for r in results)
            hits = relevant_set & retrieved
            p = len(hits) / len(retrieved) if retrieved else 0
            r = len(hits) / len(relevant_set) if relevant_set else 0
            print(f"\n  标记了 {len(ratings)} 个评分 ({len(relevant)} 个相关)")
            print(f"  Precision@10: {p:.2%}  Recall: {r:.2%}")

    def _evaluate_custom_query(self, query_str):
        """Evaluate a single custom query WITHOUT prior feedback bias."""
        print(f"\n{'─' * 70}")
        print(f"自定义查询: \"{query_str}\"")

        results = self.run_search(query_str, top_k=10)
        if not results:
            print("  无检索结果。")
            return

        ratings = self._rate_results_interactive(results, query_str)
        if ratings is None or not ratings:
            return

        self.feedback.record_batch(ratings, query_str)

        relevant = [doc_id for doc_id, r in ratings.items() if r >= 4]
        relevant_set = set(relevant)
        retrieved = set(r["id"] for r in results)
        hits = relevant_set & retrieved
        p = len(hits) / len(retrieved) if retrieved else 0
        r = len(hits) / len(relevant_set) if relevant_set else 0
        print(f"\n  标记了 {len(ratings)} 个评分 ({len(relevant)} 个相关)")
        print(f"  Precision@10: {p:.2%}  Recall: {r:.2%}")

        self.eval_queries.append({
            "query": query_str,
            "description": "用户自定义查询",
            "relevant_docs": relevant,
        })
        self._save_eval_queries()

    def _compare_feedback_effect(self):
        """Compare search results before/after feedback for all evaluated queries."""
        print("\n" + "=" * 70)
        print("  反馈效果对比 (Before vs After)")
        print("=" * 70)

        tested = 0
        for eq in self.eval_queries:
            if not eq.get("relevant_docs"):
                continue
            tested += 1
            relevant_set = set(eq["relevant_docs"])

            results_before = self.run_search(eq["query"], top_k=10)
            results_after = self.run_search_with_feedback(eq["query"], top_k=10)

            before_hits = len(relevant_set & set(r["id"] for r in results_before))
            after_hits = len(relevant_set & set(r["id"] for r in results_after))
            p_before = before_hits / len(results_before) if results_before else 0
            p_after = after_hits / len(results_after) if results_after else 0

            delta = p_after - p_before
            indicator = "✓ 提升" if delta > 0 else ("=" if delta == 0 else "✗ 下降")
            print(f"\n  {eq['query']}")
            print(f"    反馈前: {p_before:.2%} ({before_hits} hits)")
            print(f"    反馈后: {p_after:.2%} ({after_hits} hits)  {indicator}")

        if tested == 0:
            print("  尚无评价数据，请先进行评价。")

    def _show_feedback_stats(self):
        """Display comprehensive feedback statistics."""
        stats = self.feedback.get_feedback_stats()
        print("\n" + "=" * 70)
        print("  反馈数据统计")
        print("=" * 70)
        print(f"  总评分数: {stats['total_ratings']}")
        print(f"  已评价文档数: {stats['rated_docs']}")
        print(f"  平均评分: {stats['avg_rating']:.2f}/5.0")
        print(f"  收藏文档 (≥4分): {stats['liked_docs']}")
        print(f"  不相关文档 (≤2分): {stats['disliked_docs']}")
        print(f"  有反馈的查询数: {stats['queries_with_feedback']}")
        print(f"  最后更新: {stats['last_updated'][:19]}")

        liked = self.feedback.get_liked_docs()
        disliked = self.feedback.get_disliked_docs()
        if liked:
            print(f"\n  收藏文档 Top 5:")
            for doc_id in sorted(liked, key=lambda d: self.feedback.get_document_boost(d), reverse=True)[:5]:
                boost = self.feedback.get_document_boost(doc_id)
                entry = self.feedback.data["doc_ratings"].get(str(doc_id), {})
                print(f"    doc#{doc_id}: avg={entry.get('avg_rating', 0):.1f}/5, "
                      f"ratings={entry.get('rating_count', 0)}, boost={boost:+.3f}")

    def show_metrics_summary(self):
        self._compute_metrics()

    def _compute_metrics(self):
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

"""
Relevance Feedback Module
Implements user-driven search result optimization via feedback loops.

Core mechanisms:
1. FeedbackStore — persist user ratings per document per query
2. Rocchio query expansion — augment query vector with terms from liked docs
3. Document boosting — re-rank results by accumulated user ratings

Reference:
[1] Rocchio, J. J. (1971). Relevance feedback in information retrieval.
    In The SMART Retrieval System: Experiments in Automatic Document Processing.
"""

import json
import os
import math
from datetime import datetime
from collections import defaultdict, Counter

FEEDBACK_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "feedback.json"
)

DEFAULT_ALPHA = 1.0    # original query weight
DEFAULT_BETA = 0.75    # relevant docs weight
DEFAULT_GAMMA = 0.15   # irrelevant docs weight (debias)
DEFAULT_BOOST_STRENGTH = 0.5


class FeedbackStore:
    """
    Persistent store for user relevance feedback.

    Data format (feedback.json):
    {
        "doc_ratings": {
            "<doc_id>": {
                "rating_sum": 15.0,
                "rating_count": 3,
                "avg_rating": 5.0,
                "queries": ["query text 1", "query text 2"],
                "tokens": [["token1", "token2"], ...],
                "last_updated": "2026-05-18T..."
            }
        },
        "query_feedback": {
            "<query_text>": {
                "relevant_docs": [0, 5, 12],
                "irrelevant_docs": [3, 8],
                "tokens": ["token1", "token2"],
                "count": 2
            }
        },
        "stats": {
            "total_ratings": 42,
            "rated_docs": 15,
            "last_updated": "2026-05-18T..."
        }
    }
    """

    def __init__(self, filepath=None):
        self.filepath = filepath or FEEDBACK_FILE
        self.data = self._load()
        self.alpha = DEFAULT_ALPHA
        self.beta = DEFAULT_BETA
        self.gamma = DEFAULT_GAMMA

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return self._empty_data()
                data.setdefault("doc_ratings", {})
                data.setdefault("query_feedback", {})
                data.setdefault("stats", {"total_ratings": 0, "rated_docs": 0})
                return data
            except (json.JSONDecodeError, IOError):
                pass
        return self._empty_data()

    def _empty_data(self):
        return {
            "doc_ratings": {},
            "query_feedback": {},
            "stats": {"total_ratings": 0, "rated_docs": 0, "last_updated": ""},
        }

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.data["stats"]["total_ratings"] = sum(
            dr.get("rating_count", 0) for dr in self.data["doc_ratings"].values()
        )
        self.data["stats"]["rated_docs"] = len(self.data["doc_ratings"])
        self.data["stats"]["last_updated"] = datetime.now().isoformat()
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record_feedback(self, doc_id, rating, query_str=None):
        """
        Record a user rating for a document.

        Args:
            doc_id: Document ID (int)
            rating: 1-5 (1=irrelevant, 5=highly relevant)
            query_str: Optional query string that led to this result
        """
        doc_id = str(doc_id)
        if doc_id not in self.data["doc_ratings"]:
            self.data["doc_ratings"][doc_id] = {
                "rating_sum": 0.0,
                "rating_count": 0,
                "avg_rating": 0.0,
                "queries": [],
                "tokens": [],
                "last_updated": "",
            }

        entry = self.data["doc_ratings"][doc_id]
        entry["rating_sum"] += rating
        entry["rating_count"] += 1
        entry["avg_rating"] = entry["rating_sum"] / entry["rating_count"]
        entry["last_updated"] = datetime.now().isoformat()

        if query_str and query_str not in entry["queries"]:
            entry["queries"].append(query_str)

        if query_str:
            self._record_query_feedback(doc_id, rating, query_str)

        self._save()
        return entry

    def _record_query_feedback(self, doc_id, rating, query_str):
        if query_str not in self.data["query_feedback"]:
            self.data["query_feedback"][query_str] = {
                "relevant_docs": [],
                "irrelevant_docs": [],
                "tokens": [],
                "count": 0,
            }
        qf = self.data["query_feedback"][query_str]
        qf["count"] += 1
        doc_id_int = int(doc_id)
        if rating >= 4:
            if doc_id_int not in qf["relevant_docs"]:
                qf["relevant_docs"].append(doc_id_int)
        elif rating <= 2:
            if doc_id_int not in qf["irrelevant_docs"]:
                qf["irrelevant_docs"].append(doc_id_int)

    def record_batch(self, ratings_dict, query_str=None):
        """
        Record multiple ratings at once.

        Args:
            ratings_dict: {doc_id: rating, ...}
            query_str: Optional query string
        """
        for doc_id, rating in ratings_dict.items():
            self.record_feedback(doc_id, rating, query_str)

    def get_document_boost(self, doc_id, boost_strength=None):
        """
        Get boost multiplier for a document based on feedback.

        A doc with avg_rating 5 gets ~1.0x bonus normalized score added.
        A doc with no feedback gets 0.
        A doc with avg_rating 1 gets negative boost.

        Returns:
            float: Score boost additive value (not multiplier)
        """
        if boost_strength is None:
            boost_strength = DEFAULT_BOOST_STRENGTH

        entry = self.data["doc_ratings"].get(str(doc_id))
        if not entry or entry["rating_count"] == 0:
            return 0.0

        confidence = min(entry["rating_count"] / 2.0, 1.0)
        rating_z = (entry["avg_rating"] - 3.0) / 2.0
        return boost_strength * rating_z * confidence

    def get_liked_docs(self, min_rating=4, min_count=1):
        """Get set of doc IDs with positive feedback."""
        liked = set()
        for doc_id, entry in self.data["doc_ratings"].items():
            if entry["avg_rating"] >= min_rating and entry["rating_count"] >= min_count:
                liked.add(int(doc_id))
        return liked

    def get_disliked_docs(self, max_rating=2, min_count=1):
        """Get set of doc IDs with negative feedback."""
        disliked = set()
        for doc_id, entry in self.data["doc_ratings"].items():
            if entry["avg_rating"] <= max_rating and entry["rating_count"] >= min_count:
                disliked.add(int(doc_id))
        return disliked

    def expand_query_tokens(self, query_tokens, index, top_k=5):
        """
        Rocchio-inspired query expansion using feedback.
        Only expands from liked documents that overlap with the current query.
        """
        liked_docs = self.get_liked_docs()
        if not liked_docs:
            return []

        query_set = set(query_tokens)
        term_scores = defaultdict(float)

        for doc_id in liked_docs:
            doc = index.documents[doc_id] if isinstance(index.documents, list) else index.documents.get(str(doc_id))
            if not doc:
                continue
            doc_tokens = doc.get("tokens", [])
            doc_token_set = set(doc_tokens)

            if not (query_set & doc_token_set):
                continue

            boost = self.get_document_boost(doc_id)
            if boost <= 0:
                continue

            token_tf = Counter(doc_tokens)
            seen_in_doc = set()
            for token, tf in token_tf.items():
                if token in query_set or token in seen_in_doc:
                    continue
                seen_in_doc.add(token)
                idf = index.get_idf(token)
                term_scores[token] += tf * idf * boost

        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_terms:
            return []

        max_score = sorted_terms[0][1]
        normalized = [(t, s / max_score * 0.3) for t, s in sorted_terms[:top_k]]
        return normalized

    def get_suppressed_terms(self, query_tokens, index, gamma=0.3):
        """
        Rocchio gamma term: find terms from disliked documents to suppress.
        Only considers disliked docs that overlap with current query.
        """
        disliked_docs = self.get_disliked_docs()
        if not disliked_docs:
            return set()

        query_set = set(query_tokens)
        term_scores = defaultdict(float)

        for doc_id in disliked_docs:
            doc = index.documents[doc_id] if isinstance(index.documents, list) else index.documents.get(str(doc_id))
            if not doc:
                continue
            doc_tokens = doc.get("tokens", [])
            if not (query_set & set(doc_tokens)):
                continue

            penalty = abs(self.get_document_boost(doc_id, boost_strength=1.0))
            if penalty <= 0:
                continue

            token_tf = Counter(doc_tokens)
            for token, tf in token_tf.items():
                if token in query_set:
                    continue
                idf = index.get_idf(token)
                term_scores[token] += tf * idf * penalty

        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return {t for t, _ in sorted_terms[:5]}

    def get_feedback_stats(self):
        """Return summary statistics of feedback data."""
        return {
            "total_ratings": self.data["stats"].get("total_ratings", 0),
            "rated_docs": self.data["stats"].get("rated_docs", 0),
            "avg_rating": (
                sum(e["avg_rating"] for e in self.data["doc_ratings"].values())
                / max(len(self.data["doc_ratings"]), 1)
            ),
            "liked_docs": len(self.get_liked_docs()),
            "disliked_docs": len(self.get_disliked_docs()),
            "queries_with_feedback": len(self.data["query_feedback"]),
            "last_updated": self.data["stats"].get("last_updated", "never"),
        }

    def clear(self):
        """Clear all feedback data."""
        self.data = self._empty_data()
        self._save()
        print("[Feedback] All feedback data cleared.")

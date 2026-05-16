import math
import heapq
from config import VSM_TOP_K


class VectorSpaceModel:
    """Vector Space Model retrieval using TF-IDF and cosine similarity."""

    def __init__(self, inverted_index):
        self.index = inverted_index
        self.documents = inverted_index.documents
        self._doc_map = {}
        if isinstance(self.documents, list):
            for doc in self.documents:
                self._doc_map[doc["id"]] = doc
        elif isinstance(self.documents, dict):
            self._doc_map = self.documents
        self._doc_vectors = {}

    def _get_doc(self, doc_id):
        if isinstance(self.documents, list):
            return self._doc_map.get(doc_id, self.documents[doc_id] if doc_id < len(self.documents) else None)
        return self.documents.get(doc_id)

    def _compute_tfidf_vector(self, tokens):
        """Compute TF-IDF vector for a list of tokens (query or document)."""
        vec = {}
        tf_map = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0.0) + 1.0
        token_len = len(tokens) if tokens else 1
        for token, tf in tf_map.items():
            tf_normalized = tf / token_len
            idf = self.index.get_idf(token)
            vec[token] = tf_normalized * idf
        return vec

    def _cosine_similarity(self, vec_a, vec_b):
        """Compute cosine similarity between two sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0
        dot = 0.0
        for token, weight in vec_a.items():
            if token in vec_b:
                dot += weight * vec_b[token]
        norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
        norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_tokens, top_k=None):
        """
        Search documents using vector space model.
        Returns list of (doc_id, score, doc) sorted by relevance descending.
        """
        if top_k is None:
            top_k = VSM_TOP_K

        query_vec = self._compute_tfidf_vector(query_tokens)
        if not query_vec:
            return []

        candidate_docs = set()
        for token in query_tokens:
            postings = self.index.get_postings(token)
            candidate_docs.update(postings.keys())

        results = []
        for doc_id in candidate_docs:
            doc = self._get_doc(doc_id)
            if doc is None:
                continue
            doc_tokens = doc.get("tokens", doc.get("tokens", []))
            doc_vec = self._compute_tfidf_vector(doc_tokens)
            score = self._cosine_similarity(query_vec, doc_vec)
            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        output = []
        for doc_id, score in top_results:
            doc = self._get_doc(doc_id)
            if doc is None:
                continue
            snippet = self._generate_snippet(doc, query_tokens)
            output.append({
                "id": doc_id,
                "score": round(score, 6),
                "title": doc.get("title", "无标题"),
                "snippet": snippet,
                "url": doc.get("url", ""),
                "date": doc.get("date", ""),
            })
        return output

    def _generate_snippet(self, doc, query_tokens, window=60):
        """Generate a text snippet around query term matches."""
        text = doc.get("text", "")
        if not text:
            return ""
        positions = []
        for token in query_tokens:
            idx = text.find(token)
            if idx != -1:
                positions.append(idx)

        if positions:
            center = sum(positions) // len(positions)
            start = max(0, center - window // 2)
            end = min(len(text), center + window // 2)
            if start > 0:
                start = text.find("。", start) + 1 if text.find("。", start) != -1 else start
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet
        else:
            return text[:window] + "..." if len(text) > window else text


if __name__ == "__main__":
    from inverted_index import InvertedIndex
    idx = InvertedIndex()
    sample_docs = [
        {"id": 0, "tokens": ["人工智能", "技术", "发展", "深度学习", "机器学习"],
         "title": "AI发展", "text": "人工智能技术正在快速发展", "url": "http://a.com", "date": "2024-01"},
        {"id": 1, "tokens": ["人工智能", "应用", "领域", "计算机", "视觉"],
         "title": "AI应用", "text": "人工智能在各个领域的应用", "url": "http://b.com", "date": "2024-02"},
        {"id": 2, "tokens": ["技术", "创新", "发展", "区块链", "大数据"],
         "title": "技术创新", "text": "技术创新推动社会发展", "url": "http://c.com", "date": "2024-03"},
    ]
    idx.build(sample_docs)
    vsm = VectorSpaceModel(idx)
    results = vsm.search(["人工智能", "发展"])
    for r in results:
        print(f"Score: {r['score']:.4f} | {r['title']} | {r['snippet'][:50]}")

"""
BM25 Ranking Algorithm Implementation
Optimized probabilistic retrieval model based on TF-IDF.

BM25 formula:
score(D,Q) = sum_{i=1}^{n} IDF(q_i) * (f(q_i,D) * (k1 + 1)) / (f(q_i,D) + k1 * (1 - b + b * |D|/avgdl))

where:
- k1: term frequency saturation parameter (typically 1.2-2.0)
- b: length normalization parameter (typically 0.75)
- avgdl: average document length

References:
[1] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
    Foundations and Trends in Information Retrieval, 3(4), 333-389.
[2] Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval.
    Cambridge University Press. Chapter 11.
"""

import math
from collections import defaultdict
from config import VSM_TOP_K


class BM25Model:
    """
    BM25 probabilistic ranking model for information retrieval.
    
    BM25 improves upon TF-IDF by:
    1. Using probabilistic term weighting (IDF based on document frequency)
    2. Saturation function for term frequency (diminishing returns)
    3. Better length normalization (compensates for document length bias)
    
    Parameters:
        k1 (float): Controls term frequency saturation (default: 1.5)
        b (float): Controls length normalization (default: 0.75)
    """
    
    def __init__(self, inverted_index, k1=1.5, b=0.75):
        self.index = inverted_index
        self.documents = inverted_index.documents
        self.k1 = k1
        self.b = b
        
        # Pre-compute document statistics
        self._doc_map = {}
        self.doc_lengths = {}
        self.avgdl = 0.0
        self._compute_stats()
        
    def _compute_stats(self):
        """Pre-compute document length statistics."""
        total_len = 0
        if isinstance(self.documents, list):
            for doc in self.documents:
                self._doc_map[doc["id"]] = doc
                doc_len = len(doc.get("tokens", []))
                self.doc_lengths[doc["id"]] = doc_len
                total_len += doc_len
            self.avgdl = total_len / max(len(self.documents), 1)
        elif isinstance(self.documents, dict):
            self._doc_map = self.documents
            for doc_id, doc in self.documents.items():
                doc_len = len(doc.get("tokens", []))
                self.doc_lengths[doc_id] = doc_len
                total_len += doc_len
            self.avgdl = total_len / max(len(self.documents), 1)
    
    def _get_doc(self, doc_id):
        """Get document by ID."""
        return self._doc_map.get(doc_id)
    
    def _idf(self, term):
        """
        Compute BM25 IDF with smoothing.
        IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5))
        where N = total docs, n(t) = docs containing term t
        """
        N = self.index.doc_count
        n_t = len(self.index.get_postings(term))
        
        # BM25 IDF with smoothing to avoid negative values
        idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)
        return idf
    
    def _score_document(self, doc_id, query_tokens):
        """
        Compute BM25 score for a single document.
        
        score = sum_{q in Q} IDF(q) * (f(q,D) * (k1 + 1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl))
        """
        score = 0.0
        doc_len = self.doc_lengths.get(doc_id, self.avgdl)
        
        # Length normalization factor
        K = self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
        
        for token in query_tokens:
            # Get term frequency in document
            tf = self.index.get_postings(token).get(doc_id, 0)
            if tf == 0:
                continue
            
            # BM25 term weight
            idf = self._idf(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + K
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query_tokens, top_k=None):
        """
        Search documents using BM25 ranking.
        
        Args:
            query_tokens: List of query term tokens
            top_k: Number of top results to return
            
        Returns:
            List of result dictionaries sorted by BM25 score
        """
        if top_k is None:
            top_k = VSM_TOP_K
        
        if not query_tokens:
            return []
        
        # Get candidate documents (union of postings for all query terms)
        candidate_docs = set()
        for token in query_tokens:
            postings = self.index.get_postings(token)
            candidate_docs.update(postings.keys())
        
        # Score all candidates
        results = []
        for doc_id in candidate_docs:
            score = self._score_document(doc_id, query_tokens)
            if score > 0:
                results.append((doc_id, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Format output
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
    
    def search_with_feedback(self, query_tokens, feedback_store, top_k=None,
                              query_expansion=True, boost_strength=None):
        """
        Search with relevance feedback optimization.

        Two mechanisms:
        1. Query expansion — add high-IDF terms from liked documents
        2. Document boosting — boost score of well-rated documents
        """
        if top_k is None:
            top_k = VSM_TOP_K

        expanded_tokens = list(query_tokens)
        expansion_terms = []

        if query_expansion:
            expansion_terms = feedback_store.expand_query_tokens(
                query_tokens, self.index, top_k=5
            )
            if expansion_terms:
                for term, weight in expansion_terms:
                    for _ in range(max(1, int(weight * 10))):
                        expanded_tokens.append(term)

        if not expanded_tokens:
            return []

        candidate_docs = set()
        for token in set(expanded_tokens):
            postings = self.index.get_postings(token)
            candidate_docs.update(postings.keys())

        results = []
        for doc_id in candidate_docs:
            score = self._score_document(doc_id, expanded_tokens)

            fb_boost = feedback_store.get_document_boost(
                doc_id,
                boost_strength if boost_strength is not None else None
            )
            if fb_boost != 0:
                score += fb_boost

            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        liked_docs = feedback_store.get_liked_docs()
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
                "feedback_boosted": doc_id in liked_docs,
            })

        if expansion_terms:
            print(f"\n  [Rocchio扩展] 添加了 {len(expansion_terms)} 个相关词: "
                  f"{', '.join(t for t, _ in expansion_terms[:5])}")

        return output

    def _generate_snippet(self, doc, query_tokens, window=60):
        """Generate text snippet around query term matches."""
        text = doc.get("text", "")
        if not text:
            return ""
        
        # Find positions of query terms
        positions = []
        for token in query_tokens:
            idx = text.find(token)
            if idx != -1:
                positions.append(idx)
        
        if positions:
            center = sum(positions) // len(positions)
            start = max(0, center - window // 2)
            end = min(len(text), center + window // 2)
            
            # Adjust to sentence boundaries
            if start > 0:
                period_idx = text.rfind("。", 0, start)
                if period_idx != -1:
                    start = period_idx + 1
            
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet
        else:
            return text[:window] + "..." if len(text) > window else text


class AlgorithmComparator:
    """Compare TF-IDF/VSM with BM25 on the same queries."""
    
    def __init__(self, vsm_model, bm25_model, preprocessor):
        self.vsm = vsm_model
        self.bm25 = bm25_model
        self.preprocessor = preprocessor
    
    def compare_on_query(self, query_str, top_k=10):
        """Run both algorithms on the same query and return comparison."""
        tokens = self.preprocessor.segment(query_str)
        
        vsm_results = self.vsm.search(tokens, top_k=top_k)
        bm25_results = self.bm25.search(tokens, top_k=top_k)
        
        return {
            "query": query_str,
            "tokens": tokens,
            "vsm_results": vsm_results,
            "bm25_results": bm25_results,
            "vsm_count": len(vsm_results),
            "bm25_count": len(bm25_results),
        }
    
    def evaluate_precision(self, queries_with_relevance, top_k=10):
        """
        Evaluate Precision@k for both algorithms.
        
        Args:
            queries_with_relevance: List of (query_str, relevant_doc_ids) tuples
            top_k: k for Precision@k
            
        Returns:
            Comparison metrics dictionary
        """
        vsm_precisions = []
        bm25_precisions = []
        
        for query_str, relevant_ids in queries_with_relevance:
            tokens = self.preprocessor.segment(query_str)
            relevant_set = set(relevant_ids)
            
            # VSM results
            vsm_results = self.vsm.search(tokens, top_k=top_k)
            vsm_retrieved = set(r["id"] for r in vsm_results)
            vsm_hits = len(relevant_set & vsm_retrieved)
            vsm_precision = vsm_hits / len(vsm_retrieved) if vsm_retrieved else 0
            vsm_precisions.append(vsm_precision)
            
            # BM25 results
            bm25_results = self.bm25.search(tokens, top_k=top_k)
            bm25_retrieved = set(r["id"] for r in bm25_results)
            bm25_hits = len(relevant_set & bm25_retrieved)
            bm25_precision = bm25_hits / len(bm25_retrieved) if bm25_retrieved else 0
            bm25_precisions.append(bm25_precision)
        
        return {
            "vsm_avg_precision": sum(vsm_precisions) / len(vsm_precisions) if vsm_precisions else 0,
            "bm25_avg_precision": sum(bm25_precisions) / len(bm25_precisions) if bm25_precisions else 0,
            "vsm_precisions": vsm_precisions,
            "bm25_precisions": bm25_precisions,
            "improvement": (sum(bm25_precisions) - sum(vsm_precisions)) / sum(vsm_precisions) * 100 
                           if sum(vsm_precisions) > 0 else 0,
        }


if __name__ == "__main__":
    # Test BM25
    import json
    from inverted_index import InvertedIndex
    from preprocessor import TextPreprocessor
    from vsm import VectorSpaceModel
    
    with open("data/documents.json", "r") as f:
        docs = json.load(f)
    
    index = InvertedIndex()
    index.load()
    index.documents = docs
    
    preprocessor = TextPreprocessor()
    vsm = VectorSpaceModel(index)
    bm25 = BM25Model(index)
    
    query = "人工智能技术发展"
    tokens = preprocessor.segment(query)
    
    print(f"Query: {query}")
    print(f"Tokens: {tokens}")
    print("\n--- VSM (TF-IDF) Results ---")
    for r in vsm.search(tokens, top_k=5):
        print(f"  {r['score']:.4f} | {r['title'][:50]}")
    
    print("\n--- BM25 Results ---")
    for r in bm25.search(tokens, top_k=5):
        print(f"  {r['score']:.4f} | {r['title'][:50]}")

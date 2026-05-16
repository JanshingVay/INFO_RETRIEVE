import json
import os
import math
from collections import defaultdict
from config import INDEX_FILE


class InvertedIndex:
    """Build and query inverted index with TF-IDF statistics."""

    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_count = 0
        self.doc_lengths = {}
        self.idf = {}
        self.documents = []

    def build(self, documents):
        """
        Build inverted index from preprocessed documents.
        Each document must have 'id' and 'tokens' fields.
        """
        self.documents = documents
        self.doc_count = len(documents)
        self.index = defaultdict(lambda: defaultdict(int))
        self.doc_lengths = {}

        for doc in documents:
            doc_id = doc["id"]
            token_set = set()
            for token in doc["tokens"]:
                self.index[token][doc_id] += 1
                token_set.add(token)
            self.doc_lengths[doc_id] = len(doc["tokens"])
            for token in token_set:
                if "df" not in doc:
                    doc["df"] = {}
                doc["df"] = doc.get("df", {})
            for token in token_set:
                pass

        for token, postings in self.index.items():
            df = len(postings)
            self.idf[token] = math.log10(self.doc_count / (df + 1)) + 1.0

        return self

    def get_postings(self, token):
        return dict(self.index.get(token, {}))

    def get_idf(self, token):
        return self.idf.get(token, 0.0)

    def get_tf(self, token, doc_id):
        raw_tf = self.index.get(token, {}).get(doc_id, 0)
        if raw_tf == 0:
            return 0.0
        doc_len = self.doc_lengths.get(doc_id, 1)
        return raw_tf / doc_len

    def save(self, filepath=None):
        path = filepath or INDEX_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "index": {k: dict(v) for k, v in self.index.items()},
            "doc_count": self.doc_count,
            "doc_lengths": self.doc_lengths,
            "idf": self.idf,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath=None):
        path = filepath or INDEX_FILE
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.index = defaultdict(lambda: defaultdict(int))
        for k, postings in data["index"].items():
            for doc_id_str, count in postings.items():
                self.index[k][int(doc_id_str)] = count
        self.doc_count = data["doc_count"]
        self.doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        self.idf = data["idf"]
        return True

    def get_stats(self):
        """Return index statistics."""
        total_docs = self.doc_count
        total_terms = len(self.index)
        total_postings = sum(len(v) for v in self.index.values())
        return {
            "total_documents": total_docs,
            "total_terms": total_terms,
            "total_postings": total_postings,
            "avg_postings_per_term": total_postings / max(total_terms, 1),
        }


if __name__ == "__main__":
    idx = InvertedIndex()
    sample_docs = [
        {"id": 0, "tokens": ["人工智能", "技术", "发展"]},
        {"id": 1, "tokens": ["人工智能", "应用", "领域"]},
        {"id": 2, "tokens": ["技术", "创新", "发展", "应用"]},
    ]
    idx.build(sample_docs)
    print("Stats:", idx.get_stats())
    print("Postings for '人工智能':", idx.get_postings("人工智能"))

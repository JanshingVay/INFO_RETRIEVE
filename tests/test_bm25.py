import pytest
import os
import sys

# Add parent directory to sys.path to import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inverted_index import InvertedIndex
from bm25 import BM25Model

@pytest.fixture
def sample_docs():
    return [
        {"id": 0, "title": "Doc 0", "text": "人工智能技术发展", "tokens": ["人工智能", "技术", "发展"]},
        {"id": 1, "title": "Doc 1", "text": "人工智能应用领域", "tokens": ["人工智能", "应用", "领域"]},
        {"id": 2, "title": "Doc 2", "text": "技术创新发展应用", "tokens": ["技术", "创新", "发展", "应用"]},
    ]

@pytest.fixture
def index(sample_docs):
    idx = InvertedIndex()
    idx.build(sample_docs)
    return idx

@pytest.fixture
def bm25(index):
    return BM25Model(index)

def test_initialization(bm25):
    # Total tokens = 3 + 3 + 4 = 10
    # Avgdl = 10 / 3 = 3.333...
    assert pytest.approx(bm25.avgdl) == 10.0 / 3.0
    assert bm25.doc_lengths[0] == 3
    assert bm25.doc_lengths[2] == 4

def test_idf(bm25):
    # "人工智能" in 2 out of 3 docs
    idf_ai = bm25._idf("人工智能")
    # "领域" in 1 out of 3 docs
    idf_domain = bm25._idf("领域")
    
    assert idf_ai > 0
    assert idf_domain > idf_ai

def test_search_single_term(bm25):
    results = bm25.search(["人工智能"])
    
    # Docs 0 and 1 contain "人工智能"
    assert len(results) == 2
    
    # Doc 0 and Doc 1 both have 3 tokens and 1 "人工智能". Their scores should be equal.
    doc_ids = [r["id"] for r in results]
    assert 0 in doc_ids
    assert 1 in doc_ids

def test_search_multiple_terms(bm25):
    results = bm25.search(["人工智能", "应用"])
    
    # Doc 0 has "人工智能", Doc 1 has "人工智能", "应用", Doc 2 has "应用"
    # Doc 1 should rank highest
    assert len(results) > 0
    assert results[0]["id"] == 1
    
    # Doc 1 has both terms, so its score should be the sum of BM25 scores for each term
    assert results[0]["score"] > 0

def test_generate_snippet(bm25, sample_docs):
    doc = sample_docs[0]
    snippet = bm25._generate_snippet(doc, ["技术"])
    assert "技术" in snippet

def test_search_empty_query(bm25):
    results = bm25.search([])
    assert results == []

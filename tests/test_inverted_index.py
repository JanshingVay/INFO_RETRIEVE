import pytest
import os
import sys

# Add parent directory to sys.path to import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inverted_index import InvertedIndex

@pytest.fixture
def sample_docs():
    return [
        {"id": 0, "tokens": ["人工智能", "技术", "发展"]},
        {"id": 1, "tokens": ["人工智能", "应用", "领域"]},
        {"id": 2, "tokens": ["技术", "创新", "发展", "应用"]},
    ]

@pytest.fixture
def index(sample_docs):
    idx = InvertedIndex()
    idx.build(sample_docs)
    return idx

def test_build(index):
    assert index.doc_count == 3
    assert len(index.doc_lengths) == 3
    
    # "人工智能" appears in doc 0 and 1
    ai_postings = index.get_postings("人工智能")
    assert 0 in ai_postings
    assert 1 in ai_postings
    assert 2 not in ai_postings
    
    # "技术" appears in doc 0 and 2
    tech_postings = index.get_postings("技术")
    assert 0 in tech_postings
    assert 2 in tech_postings
    assert 1 not in tech_postings

def test_get_tf(index):
    # doc 0 has 3 tokens, "人工智能" appears 1 time -> tf = 1/3
    tf_ai_0 = index.get_tf("人工智能", 0)
    assert pytest.approx(tf_ai_0) == 1.0 / 3.0
    
    # tf for a non-existent token should be 0
    tf_not_exist = index.get_tf("不存在的词", 0)
    assert tf_not_exist == 0.0

def test_get_idf(index):
    # "人工智能" appears in 2 docs (out of 3)
    idf_ai = index.get_idf("人工智能")
    assert idf_ai > 0
    
    # "领域" appears in 1 doc
    idf_domain = index.get_idf("领域")
    
    # Terms appearing in fewer docs should have higher IDF
    assert idf_domain > idf_ai
    
    # IDF for non-existent token should be 0
    assert index.get_idf("不存在的词") == 0.0

def test_save_and_load(index, tmp_path):
    index_file = tmp_path / "test_index.json"
    
    # Save the index
    index.save(str(index_file))
    assert os.path.exists(index_file)
    
    # Load into a new index instance
    new_index = InvertedIndex()
    success = new_index.load(str(index_file))
    
    assert success is True
    assert new_index.doc_count == index.doc_count
    assert new_index.doc_lengths == index.doc_lengths
    assert new_index.idf == index.idf
    assert new_index.get_postings("人工智能") == index.get_postings("人工智能")

def test_get_stats(index):
    stats = index.get_stats()
    assert stats["total_documents"] == 3
    assert stats["total_terms"] > 0
    assert stats["total_postings"] > 0

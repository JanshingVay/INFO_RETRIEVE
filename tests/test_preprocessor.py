import pytest
import os
import sys

# Add parent directory to sys.path to import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_clean_text(preprocessor):
    text_with_html = "<html><body><p>Hello <b>World</b>!</p></body></html>"
    cleaned = preprocessor.clean_text(text_with_html)
    assert cleaned == "Hello World"
    
    text_with_special_chars = "Hello!@# World$%^"
    cleaned = preprocessor.clean_text(text_with_special_chars)
    assert cleaned == "Hello World"
    
    text_with_chinese = "你好，世界！"
    cleaned = preprocessor.clean_text(text_with_chinese)
    assert cleaned == "你好 世界"
    
    text_with_extra_spaces = "  Hello   World  \n "
    cleaned = preprocessor.clean_text(text_with_extra_spaces)
    assert cleaned == "Hello World"

def test_segment(preprocessor):
    # Test segmentation and stopword removal
    text = "人工智能技术正在改变世界，深度学习是其中的重要分支。"
    tokens = preprocessor.segment(text)
    
    # "是", "其中", "的" are stopwords, they should be removed
    # "人工智能技术", "正在", "改变", "世界", "深度", "学习", "重要", "分支"
    # Actually jieba might segment differently, let's just check some keywords
    assert "人工智能" in tokens or "人工智能技术" in tokens
    assert "世界" in tokens
    assert "学习" in tokens
    assert "的" not in tokens
    assert "是" not in tokens

def test_segment_numbers_and_single_chars(preprocessor):
    text = "他有 123 个苹果, A B C D"
    tokens = preprocessor.segment(text)
    
    # Numbers and single english chars should be removed according to preprocessor.py
    assert "123" not in tokens
    assert "a" not in tokens
    assert "b" not in tokens
    assert "苹果" in tokens

def test_process_documents(preprocessor):
    docs = [
        {"id": 1, "title": "标题1", "text": "内容1"},
        {"id": 2, "title": "标题2", "text": "内容2"}
    ]
    processed_docs = preprocessor.process_documents(docs)
    
    for doc in processed_docs:
        assert "tokens" in doc
        assert isinstance(doc["tokens"], list)

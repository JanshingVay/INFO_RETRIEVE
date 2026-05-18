import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json")
INDEX_FILE = os.path.join(DATA_DIR, "inverted_index.json")
STOPWORDS_FILE = os.path.join(DATA_DIR, "stopwords.txt")

CRAWL_MIN_DOCS = 100
CRAWL_TIMEOUT = 10
CRAWL_DELAY = 0.5
CRAWL_MAX_DOCS = 150

VSM_TOP_K = 20

EVAL_QUERIES_FILE = os.path.join(DATA_DIR, "eval_queries.json")

CRAWL_SOURCES = [
    {
        "name": "IT之家",
        "base_url": "https://www.ithome.com",
        "list_urls": [
            "https://www.ithome.com/",
            "https://www.ithome.com/cat/44.html",
            "https://www.ithome.com/cat/48.html",
        ],
        "article_url_patterns": ["/0/"],
        "title_selector": "h1",
        "content_selectors": [
            {"tag": "div", "class_": "post_content"},
            {"tag": "div", "class_": "content"},
        ],
        "date_format": r"(\d{4}/\d{1,2}/\d{1,2})|(\d{4}-\d{2}-\d{2})",
        "enabled": True,
    },
    {
        "name": "36氪科技",
        "base_url": "https://36kr.com",
        "list_urls": [
            "https://36kr.com/information/technology",
            "https://36kr.com/information/web",
        ],
        "article_url_patterns": ["/p/"],
        "title_selector": "h1",
        "content_selectors": [
            {"tag": "div", "class_": "common-width"},
            {"tag": "div", "class_": "content"},
            {"tag": "article"},
        ],
        "date_format": r"(\d{4}年\d{1,2}月\d{1,2}日)|(\d{4}-\d{2}-\d{2})|(\d{4}/\d{1,2}/\d{1,2})",
        "enabled": True,
    },
    {
        "name": "新华网科技",
        "base_url": "http://www.news.cn/tech",
        "list_urls": [
            "http://www.news.cn/tech/index.html",
        ],
        "article_url_patterns": ["/202"],
        "title_selector": "h1",
        "content_selectors": [
            {"tag": "div", "id": "detail-content"},
            {"tag": "div", "class_": r"(article|content)"},
        ],
        "date_format": r"(\d{4}-\d{2}-\d{2})",
        "enabled": True,
    },
]

DATA_CLEAN_CONFIG = {
    "min_content_length": 100,
    "max_title_length": 200,
    "dedup_by_url": True,
    "dedup_by_title_similarity": 0.85,
    "remove_boilerplate": True,
    "normalize_whitespace": True,
}

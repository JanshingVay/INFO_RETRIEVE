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

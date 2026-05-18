import requests
from bs4 import BeautifulSoup
import time
import random
import re
import json
import os
from urllib.parse import urlparse
from datetime import datetime
from config import (
    CRAWL_MIN_DOCS, CRAWL_TIMEOUT, CRAWL_DELAY,
    CRAWL_MAX_DOCS, DOCUMENTS_FILE, CRAWL_SOURCES
)
from data_cleaner import clean_documents, clean_text

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def get_random_headers():
    return {**HEADERS, "User-Agent": random.choice(USER_AGENTS)}


def fetch_url(url, timeout=CRAWL_TIMEOUT, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, headers=get_random_headers(),
                timeout=timeout, allow_redirects=True
            )
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return None
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                return None


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "form", "button", "input"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text


def resolve_url(href, base_url, list_url):
    """Resolve relative URLs to absolute."""
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        parsed = urlparse(list_url)
        return f"{parsed.scheme}://{parsed.netloc}{href}"
    return list_url.rstrip("/") + "/" + href


class WebCrawler:
    """Multi-source web crawler for Chinese news/articles. Configurable via config.py."""

    def __init__(self, sources=None):
        self.documents = []
        self.seen_urls = set()
        self.sources = sources or [s for s in CRAWL_SOURCES if s.get("enabled", True)]
        self.stats = {
            "fetched": 0,
            "failed": 0,
            "skipped": 0,
        }

        if os.path.exists(DOCUMENTS_FILE):
            try:
                with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                self.documents = existing
                self.seen_urls = {d["url"] for d in existing}
                print(f"[Crawler] Loaded {len(self.documents)} existing documents.")
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(DOCUMENTS_FILE), exist_ok=True)
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def _add_document(self, url, title, text, date_str=""):
        if url in self.seen_urls:
            return False
        if len(text.strip()) < 100:
            return False
        doc_id = len(self.documents)
        doc = {
            "id": doc_id,
            "url": url,
            "title": clean_text(title.strip()) if title else "无标题",
            "text": clean_text(text.strip()),
            "date": date_str,
            "crawled_at": datetime.now().isoformat(),
        }
        self.documents.append(doc)
        self.seen_urls.add(url)
        return True

    def _extract_article_urls(self, list_url, source_config):
        """Extract article URLs from a list page using source-specific patterns."""
        resp = fetch_url(list_url)
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        urls = []
        patterns = source_config.get("article_url_patterns", [])

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(pattern in href for pattern in patterns):
                href = resolve_url(href, source_config.get("base_url", ""), list_url)
                if href not in self.seen_urls:
                    urls.append(href)

        return list(set(urls))

    def _parse_article(self, url, source_config):
        """Parse a single article page using source-specific selectors."""
        resp = fetch_url(url)
        if not resp:
            self.stats["failed"] += 1
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "无标题"

        date_str = ""
        date_format = source_config.get("date_format", r"(\d{4}-\d{2}-\d{2})")
        for pat in date_format.split("|"):
            m = re.search(pat.strip("() "), resp.text)
            if m:
                date_str = m.group(0)
                break

        content_selectors = source_config.get("content_selectors", [])
        content_div = None
        for selector in content_selectors:
            tag = selector.get("tag", "div")
            find_kwargs = {}
            for k, v in selector.items():
                if k in ("tag",):
                    continue
                if k == "id":
                    find_kwargs["id"] = v
                elif k == "class_":
                    find_kwargs["class_"] = re.compile(v, re.I)
            if find_kwargs:
                content_div = soup.find(tag, **find_kwargs)
            else:
                content_div = soup.find(tag)
            if content_div:
                break

        if content_div:
            text = extract_text_from_html(str(content_div))
        else:
            text = extract_text_from_html(resp.text)

        text = clean_text(text)
        title = clean_text(title)

        if len(text.strip()) < 100:
            self.stats["skipped"] += 1
            return None

        self.stats["fetched"] += 1
        return url, title, text, date_str

    def crawl_source(self, source_config):
        """
        Crawl a single news source from config.

        Args:
            source_config: Dict with name, base_url, list_urls, article_url_patterns, etc.
        """
        source_name = source_config.get("name", "Unknown")
        print(f"[Crawler] Crawling {source_name}...")

        all_article_urls = []
        for list_url in source_config.get("list_urls", []):
            urls = self._extract_article_urls(list_url, source_config)
            all_article_urls.extend(urls)
            time.sleep(CRAWL_DELAY)

        all_article_urls = list(set(all_article_urls))
        print(f"  Found {len(all_article_urls)} candidate URLs")
        random.shuffle(all_article_urls)

        if not all_article_urls:
            print(f"  Warning: No article URLs found for {source_name}. "
                  "The site may have changed its structure. "
                  "Update article_url_patterns in config.py.")
            return

        for url in all_article_urls:
            if len(self.documents) >= CRAWL_MAX_DOCS:
                break
            result = self._parse_article(url, source_config)
            if result:
                url, title, text, date_str = result
                if self._add_document(url, title, text, date_str):
                    print(f"  [{len(self.documents)}] {title[:40]}...")
            time.sleep(random.uniform(0.3, 1.0))

    def crawl_all(self):
        print(f"[Crawler] Starting sync crawl. Target: {CRAWL_MIN_DOCS}-{CRAWL_MAX_DOCS} docs.")
        print(f"[Crawler] Configured sources: {len(self.sources)}")

        for source in self.sources:
            if len(self.documents) >= CRAWL_MAX_DOCS:
                break
            self.crawl_source(source)
            self._save()

        original_count = len(self.documents)
        self.documents = clean_documents(self.documents)
        cleaned_removed = original_count - len(self.documents)

        self._save()

        print(f"\n[Crawler] Crawling complete!")
        print(f"  Total documents: {len(self.documents)}")
        print(f"  Fetched: {self.stats['fetched']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Skipped (too short): {self.stats['skipped']}")
        print(f"  Removed by cleaning: {cleaned_removed}")

        return self.documents


if __name__ == "__main__":
    crawler = WebCrawler()
    docs = crawler.crawl_all()
    print(f"Done. {len(docs)} documents saved.")

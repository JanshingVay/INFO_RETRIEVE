"""
High-Performance Asynchronous Web Crawler
Using asyncio + aiohttp for concurrent I/O operations.

Performance comparison:
- Traditional synchronous crawler: ~1-2 pages/second
- Async crawler: ~50-100 pages/second (limited by politeness)

References:
[1] asyncio — Asynchronous I/O. Python Documentation.
    https://docs.python.org/3/library/asyncio.html
[2] aiohttp — Async HTTP client/server. https://docs.aiohttp.org/
"""

import asyncio
import aiohttp
import aiofiles
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
import re
import json
import os
import random
from datetime import datetime
from config import (
    CRAWL_MIN_DOCS, CRAWL_TIMEOUT, CRAWL_MAX_DOCS,
    DOCUMENTS_FILE, CRAWL_SOURCES
)
from data_cleaner import clean_documents, clean_text

MAX_CONCURRENT = 20
CONNECTION_TIMEOUT = ClientTimeout(total=CRAWL_TIMEOUT, connect=5)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
]


def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }


def extract_text_from_html(html):
    """Extract clean text from HTML."""
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
        from urllib.parse import urlparse
        parsed = urlparse(list_url)
        return f"{parsed.scheme}://{parsed.netloc}{href}"
    return list_url.rstrip("/") + "/" + href


class AsyncWebCrawler:
    """
    High-performance asynchronous web crawler.
    
    Features:
    - Connection pooling for efficient HTTP reuse
    - Semaphore-based concurrency control
    - Rate limiting with adaptive delays
    - Automatic retry with exponential backoff
    - Configurable data sources via config.py
    - Built-in data cleaning
    """
    
    def __init__(self, max_concurrent=MAX_CONCURRENT, sources=None):
        self.documents = []
        self.seen_urls = set()
        self.url_queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.sources = sources or [s for s in CRAWL_SOURCES if s.get("enabled", True)]
        self.stats = {
            "fetched": 0,
            "failed": 0,
            "skipped": 0,
            "cleaned_removed": 0,
        }

        if os.path.exists(DOCUMENTS_FILE):
            try:
                with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                self.documents = existing
                self.seen_urls = {d["url"] for d in existing}
                print(f"[AsyncCrawler] Loaded {len(self.documents)} existing documents.")
            except Exception as e:
                print(f"[AsyncCrawler] Warning: Could not load existing docs: {e}")
    
    async def _fetch_with_retry(self, session, url, retries=3):
        """Fetch URL with retry logic and exponential backoff."""
        for attempt in range(retries):
            try:
                async with self.semaphore:
                    async with session.get(
                        url, 
                        headers=get_random_headers(),
                        timeout=CONNECTION_TIMEOUT,
                        allow_redirects=True
                    ) as response:
                        if response.status == 200:
                            content_type = response.headers.get("Content-Type", "")
                            if "text/html" in content_type:
                                html = await response.text()
                                return html
                            else:
                                return None
                        elif response.status in (429, 503):
                            wait = 2 ** attempt + random.uniform(0, 1)
                            await asyncio.sleep(wait)
                        else:
                            return None
            except asyncio.TimeoutError:
                if attempt < retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
            except Exception:
                if attempt < retries - 1:
                    await asyncio.sleep(1)
        return None
    
    async def _extract_article_urls(self, session, list_url, source_config):
        """Extract article URLs from a list page using source-specific patterns."""
        html = await self._fetch_with_retry(session, list_url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, "lxml")
        urls = []
        patterns = source_config.get("article_url_patterns", [])
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(pattern in href for pattern in patterns):
                href = resolve_url(href, source_config.get("base_url", ""), list_url)
                if href not in self.seen_urls:
                    urls.append(href)
        
        return list(set(urls))
    
    async def _parse_article(self, session, url, source_config):
        """Parse a single article page using source-specific selectors."""
        html = await self._fetch_with_retry(session, url)
        if not html:
            self.stats["failed"] += 1
            return None
        
        soup = BeautifulSoup(html, "lxml")
        
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "无标题"
        
        date_str = ""
        date_format = source_config.get("date_format", r"(\d{4}-\d{2}-\d{2})")
        for pat in date_format.split("|"):
            m = re.search(pat.strip("() "), html)
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
                    if isinstance(v, str) and v.startswith("re:"):
                        find_kwargs["class_"] = re.compile(v[3:], re.I)
                    else:
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
            text = extract_text_from_html(html)
        
        text = clean_text(text)
        title = clean_text(title)
        
        if len(text.strip()) < 100:
            self.stats["skipped"] += 1
            return None
        
        self.stats["fetched"] += 1
        return {
            "url": url,
            "title": title,
            "text": text,
            "date": date_str,
        }
    
    async def _worker(self, session, url_queue, results_queue, source_config):
        """Worker coroutine to process URLs from queue."""
        while True:
            try:
                url = await asyncio.wait_for(url_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                break
            
            if url is None:
                break
            
            article = await self._parse_article(session, url, source_config)
            if article:
                await results_queue.put(article)
            
            await asyncio.sleep(random.uniform(0.1, 0.3))
    
    async def _results_collector(self, results_queue):
        """Collect results and add to documents."""
        while len(self.documents) < CRAWL_MAX_DOCS:
            try:
                article = await asyncio.wait_for(results_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                if results_queue.empty():
                    break
                continue
            
            if article is None:
                break
            
            if article["url"] not in self.seen_urls:
                doc_id = len(self.documents)
                self.documents.append({
                    **article,
                    "id": doc_id,
                    "crawled_at": datetime.now().isoformat(),
                })
                self.seen_urls.add(article["url"])
                print(f"  [{len(self.documents)}] {article['title'][:40]}...")
                
                if len(self.documents) % 10 == 0:
                    await self._async_save()
    
    async def _async_save(self):
        """Asynchronously save documents to disk."""
        os.makedirs(os.path.dirname(DOCUMENTS_FILE), exist_ok=True)
        async with aiofiles.open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self.documents, ensure_ascii=False, indent=2))
    
    async def crawl_source(self, source_config):
        """
        Crawl a single news source.
        
        Args:
            source_config: Dict with name, base_url, list_urls, article_url_patterns, etc.
        """
        source_name = source_config.get("name", "Unknown")
        print(f"[AsyncCrawler] Crawling {source_name}...")
        
        connector = TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=5,
            enable_cleanup_closed=True,
            force_close=True,
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=CONNECTION_TIMEOUT
        ) as session:
            all_article_urls = []
            list_tasks = [
                self._extract_article_urls(session, url, source_config)
                for url in source_config.get("list_urls", [])
            ]
            list_results = await asyncio.gather(*list_tasks, return_exceptions=True)
            
            for urls in list_results:
                if isinstance(urls, list):
                    all_article_urls.extend(urls)
            
            all_article_urls = list(set(all_article_urls))
            random.shuffle(all_article_urls)
            print(f"  Found {len(all_article_urls)} candidate URLs")
            
            if not all_article_urls:
                print(f"  Warning: No article URLs found for {source_name}. "
                      "The site may have changed its structure. "
                      "Update article_url_patterns in config.py.")
                return
            
            url_queue = asyncio.Queue()
            results_queue = asyncio.Queue()
            
            for url in all_article_urls[:CRAWL_MAX_DOCS * 2]:
                await url_queue.put(url)
            
            workers = [
                asyncio.create_task(
                    self._worker(session, url_queue, results_queue, source_config)
                )
                for _ in range(self.max_concurrent)
            ]
            
            collector = asyncio.create_task(self._results_collector(results_queue))
            
            await asyncio.gather(*workers)
            await results_queue.put(None)
            await collector
    
    async def crawl_all(self):
        """Crawl all configured sources from config.py."""
        print(f"[AsyncCrawler] Starting async crawl. Target: {CRAWL_MIN_DOCS}-{CRAWL_MAX_DOCS} docs")
        print(f"[AsyncCrawler] Max concurrent connections: {self.max_concurrent}")
        print(f"[AsyncCrawler] Configured sources: {len(self.sources)}")
        
        for source in self.sources:
            if len(self.documents) >= CRAWL_MAX_DOCS:
                break
            await self.crawl_source(source)
        
        original_count = len(self.documents)
        self.documents = clean_documents(self.documents)
        self.stats["cleaned_removed"] = original_count - len(self.documents)
        
        await self._async_save()
        
        print(f"\n[AsyncCrawler] Complete!")
        print(f"  Total documents: {len(self.documents)}")
        print(f"  Fetched: {self.stats['fetched']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Skipped (too short): {self.stats['skipped']}")
        print(f"  Removed by cleaning: {self.stats['cleaned_removed']}")
        
        return self.documents


def run_async_crawl():
    """Entry point for synchronous code to run async crawler."""
    crawler = AsyncWebCrawler(max_concurrent=20)
    return asyncio.run(crawler.crawl_all())


if __name__ == "__main__":
    docs = run_async_crawl()
    print(f"Done. {len(docs)} documents collected.")

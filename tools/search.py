"""联网搜索工具 - 为 Researcher Agent 提供搜索能力。

支持多源搜索 fallback：
1. DuckDuckGo（优先）
2. Bing（fallback）
3. duckduckgo-search 库（如果已安装，最稳定）

代理配置：通过环境变量 HTTP_PROXY / HTTPS_PROXY 设置。
"""

import json
import logging
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 搜索全局超时配置
_SEARCH_TIMEOUT = 10
_FETCH_TIMEOUT = 5
_MAX_FETCH_WORKERS = 3
_MAX_RETRIES = 2

# ========== 代理 & HTTP 基础设施 ==========

def _get_proxy_handler() -> Optional[urllib.request.ProxyHandler]:
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        logger.debug(f"Using proxy: {proxies}")
        return urllib.request.ProxyHandler(proxies)
    return None


def _build_opener() -> urllib.request.OpenerDirector:
    handlers = []
    proxy_handler = _get_proxy_handler()
    if proxy_handler:
        handlers.append(proxy_handler)

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    handlers.append(urllib.request.HTTPSHandler(context=ssl_context))

    return urllib.request.build_opener(*handlers)


_opener = _build_opener()


def _fetch(url: str, timeout: int = _FETCH_TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _opener.open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


# ========== 缓存 ==========

_search_cache: dict[str, tuple[float, list[dict]]] = {}
_SEARCH_CACHE_TTL = 300


def _get_cached_search(query: str) -> list[dict] | None:
    if query in _search_cache:
        ts, results = _search_cache[query]
        if time.time() - ts < _SEARCH_CACHE_TTL:
            logger.debug(f"Search cache hit: '{query}'")
            return results
    return None


def _set_cached_search(query: str, results: list[dict]):
    _search_cache[query] = (time.time(), results)


# ========== 结果解析通用函数 ==========

def _dedupe_results(results: list[dict], max_results: int) -> list[dict]:
    """去重并截断结果。"""
    seen = set()
    out = []
    for r in results:
        key = r.get("href", "")
        if key and key not in seen:
            seen.add(key)
            out.append(r)
        if len(out) >= max_results:
            break
    return out


# ========== 1. duckduckgo-search 库（最稳定）==========

def _try_ddg_library(query: str, max_results: int) -> list[dict] | None:
    """尝试使用 duckduckgo-search 库（如果已安装）。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS(timeout=_SEARCH_TIMEOUT, proxy=os.environ.get("HTTPS_PROXY")) as ddgs:
            raw = ddgs.text(query, max_results=max_results * 2)
            results = []
            for item in raw:
                title = item.get("title", "").strip()
                href = item.get("href", "").strip()
                if title and href and href.startswith("http"):
                    results.append({"title": title, "href": href})
                if len(results) >= max_results:
                    break
            return results if results else None
    except Exception as e:
        logger.debug(f"ddg library unavailable or failed: {e}")
        return None


# ========== 2. DuckDuckGo HTML 搜索 ==========

def _parse_duckduckgo_html(html: str, max_results: int) -> list[dict]:
    results = []
    patterns = [
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        r'<a[^>]+rel="nofollow"[^>]+class="[^"]*result[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        r'<h[23][^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</h[23]>',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            href = match.group(1)
            redirect_match = re.search(r'uddg=([^&]+)', href)
            if redirect_match:
                href = urllib.parse.unquote(redirect_match.group(1))

            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            if title and href and href.startswith("http"):
                results.append({"title": title, "href": href})
            if len(results) >= max_results * 2:
                break
        if len(results) >= max_results * 2:
            break
    return _dedupe_results(results, max_results)


def _duckduckgo_html_search(query: str, max_results: int) -> list[dict] | None:
    encoded = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    html = _fetch(url, timeout=_SEARCH_TIMEOUT)
    results = _parse_duckduckgo_html(html, max_results)
    return results if results else None


# ========== 3. Bing HTML 搜索（fallback）==========

def _parse_bing_html(html: str, max_results: int) -> list[dict]:
    """从 Bing HTML 中解析搜索结果。"""
    results = []
    # Bing 结果通常在 .b_algo 容器中
    # 标题：<a href="..." target="_blank" h="...">标题</a>
    # 或：<h2><a href="...">标题</a></h2>
    patterns = [
        r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</li>',
        r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?</h2>',
        r'<a[^>]+href="([^"]+)"[^>]*target="_blank"[^>]*>(.*?)</a>',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
            href = match.group(1)
            # Bing 有时使用相对 URL
            if href.startswith("/"):
                href = "https://www.bing.com" + href

            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            # 过滤掉导航链接和广告
            if title in ("缓存", "相似", "Cached", "Similar", ""):
                continue
            if "microsoft" in href.lower() and "bing" in href.lower():
                continue

            if title and href and href.startswith("http"):
                results.append({"title": title, "href": href})
            if len(results) >= max_results * 2:
                break
        if len(results) >= max_results * 2:
            break
    return _dedupe_results(results, max_results)


def _bing_html_search(query: str, max_results: int) -> list[dict] | None:
    encoded = urllib.parse.quote(query)
    # 使用 Bing 国际版，减少区域限制
    url = f"https://www.bing.com/search?q={encoded}&setmkt=en-US&setlang=en"
    html = _fetch(url, timeout=_SEARCH_TIMEOUT)
    results = _parse_bing_html(html, max_results)
    return results if results else None


# ========== 统一搜索入口 ==========

def duckduckgo_search(query: str, max_results: int = 2) -> list[dict]:
    """搜索入口，带多源 fallback 和重试。

    搜索优先级：
    1. duckduckgo-search 库（最稳定，需要安装）
    2. DuckDuckGo HTML 搜索
    3. Bing HTML 搜索（fallback）
    """
    cached = _get_cached_search(query)
    if cached is not None:
        return cached

    last_error = None
    sources = [
        ("ddg_library", _try_ddg_library),
        ("duckduckgo_html", _duckduckgo_html_search),
        ("bing_html", _bing_html_search),
    ]

    for source_name, source_fn in sources:
        for attempt in range(_MAX_RETRIES):
            try:
                results = source_fn(query, max_results)
                if results:
                    logger.info(f"Search '{query}' via {source_name}: {len(results)} results")
                    _set_cached_search(query, results)
                    return results
                logger.debug(f"Source {source_name} returned empty for '{query}'")
                break  # 空结果不重试，换下一个源
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Search source={source_name} attempt={attempt + 1}/{_MAX_RETRIES} "
                    f"failed for '{query}': {e}"
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))

    logger.warning(f"All search sources failed for query: {query} - {last_error}")
    return []


# ========== 网页抓取 ==========

def fetch_page_content(url: str, max_chars: int = 1000) -> str:
    """获取网页内容（简单文本提取）。超时 5 秒。"""
    try:
        html = _fetch(url, timeout=_FETCH_TIMEOUT)
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[网页内容已截断]"
        return text
    except Exception as e:
        logger.warning(f"Failed to fetch page: {url} - {e}")
        return "[无法获取网页内容]"


def _fetch_all_pages(urls: list[str], max_chars: int = 1000) -> dict[str, str]:
    contents: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=_MAX_FETCH_WORKERS) as executor:
        futures = {executor.submit(fetch_page_content, url, max_chars): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                contents[url] = future.result()
            except Exception as e:
                logger.warning(f"Parallel fetch failed for {url}: {e}")
                contents[url] = "[无法获取网页内容]"
    return contents


def search_and_summarize(query: str, max_results: int = 2, fetch_content: bool = True) -> str:
    """搜索并返回格式化的结果文本（供 LLM 使用）。"""
    results = duckduckgo_search(query, max_results=max_results)
    if not results:
        return "[搜索未找到相关结果]"

    page_contents = {}
    if fetch_content:
        urls = [r['href'] for r in results]
        page_contents = _fetch_all_pages(urls, max_chars=1000)

    output_lines = [f"## 搜索结果：'{query}'\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(f"**[{i}]** {r['title']}")
        output_lines.append(f"链接: {r['href']}")
        if fetch_content:
            content = page_contents.get(r['href'], "[无法获取网页内容]")
            output_lines.append(f"摘要: {content}\n")
        else:
            output_lines.append("")

    return "\n".join(output_lines)

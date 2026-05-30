import re
from html import unescape
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 15
USER_AGENT = "BrandPulse/1.0 (+https://github.com/brand-pulse)"
MAX_ARTICLES = 50
SUMMARY_MAX_LEN = 400


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", text)
    return unescape(cleaned).strip()


def _normalize_link(base_url: str, href: str) -> str | None:
    href = (href or "").strip()
    if not href or href.startswith("#") or href.lower().startswith("javascript:"):
        return None
    return urljoin(base_url, href)


def _same_site(base_url: str, link: str) -> bool:
    base = urlparse(base_url)
    target = urlparse(link)
    if not target.netloc:
        return True
    return target.netloc.lower() == base.netloc.lower()


def _looks_like_article_link(link: str, title: str) -> bool:
    if len(title) < 4:
        return False
    lower = link.lower()
    skip_ext = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".pdf", ".zip")
    if any(lower.endswith(ext) for ext in skip_ext):
        return False
    skip_words = ("login", "signup", "register", "about", "contact", "privacy", "javascript:")
    if any(word in lower for word in skip_words):
        return False
    return True


def _extract_summary(container) -> str:
    if container is None:
        return ""
    for selector in (".summary", ".desc", ".description", ".excerpt", ".intro", "p"):
        node = container.select_one(selector) if hasattr(container, "select_one") else None
        if node:
            text = node.get_text(" ", strip=True)
            if len(text) >= 10:
                return text[:SUMMARY_MAX_LEN]
    text = container.get_text(" ", strip=True)
    if len(text) > 80:
        return text[:SUMMARY_MAX_LEN]
    return ""


def _append_article(
    articles: list[dict],
    seen_links: set[str],
    brand_name: str,
    source: str,
    title: str,
    link: str,
    summary: str = "",
    container=None,
) -> None:
    if link in seen_links or not _looks_like_article_link(link, title):
        return
    if not summary and container is not None:
        summary = _extract_summary(container)
    articles.append(
        {
            "brand_name": brand_name,
            "title": title[:200] or "无标题",
            "link": link,
            "summary": _strip_html(summary)[:SUMMARY_MAX_LEN],
            "published": "",
            "source": source,
        }
    )
    seen_links.add(link)


def _parse_list_page(html: bytes, page_url: str, brand_name: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    base_url = page_url
    source_title = ""
    if soup.title and soup.title.string:
        source_title = soup.title.string.strip()
    if not source_title:
        source_title = page_url

    articles: list[dict] = []
    seen_links: set[str] = set()

    for article in soup.find_all("article", limit=MAX_ARTICLES):
        link_tag = article.find("a", href=True)
        if not link_tag:
            continue
        link = _normalize_link(base_url, link_tag["href"])
        if not link or not _same_site(base_url, link):
            continue
        heading = article.find(["h1", "h2", "h3", "h4"])
        title = (heading or link_tag).get_text(" ", strip=True)
        _append_article(
            articles, seen_links, brand_name, source_title, title, link, container=article
        )
        if len(articles) >= MAX_ARTICLES:
            return articles

    if len(articles) < 3:
        for heading in soup.find_all(["h2", "h3"], limit=MAX_ARTICLES):
            anchor = heading.find("a", href=True)
            if anchor is None:
                anchor = heading.find_next("a", href=True)
            if anchor is None:
                continue
            link = _normalize_link(base_url, anchor["href"])
            if not link or not _same_site(base_url, link):
                continue
            title = heading.get_text(" ", strip=True) or anchor.get_text(" ", strip=True)
            parent = heading.find_parent(["li", "article", "div"])
            _append_article(
                articles,
                seen_links,
                brand_name,
                source_title,
                title,
                link,
                container=parent,
            )
            if len(articles) >= MAX_ARTICLES:
                break

    if len(articles) < 3:
        root = soup.find("main") or soup.find("body")
        if root:
            for anchor in root.find_all("a", href=True, limit=200):
                link = _normalize_link(base_url, anchor["href"])
                if not link or not _same_site(base_url, link):
                    continue
                title = anchor.get_text(" ", strip=True)
                parent = anchor.find_parent(["li", "article", "div"])
                _append_article(
                    articles,
                    seen_links,
                    brand_name,
                    source_title,
                    title,
                    link,
                    container=parent,
                )
                if len(articles) >= MAX_ARTICLES:
                    break

    return articles


def fetch_web_list(page_url: str, brand_name: str) -> tuple[list[dict], str | None]:
    """抓取网页列表页，返回 (文章列表, 错误信息)。成功时错误信息为 None。"""
    url = page_url.strip()
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return [], "网络请求超时，请检查网页地址或稍后重试。"
    except requests.exceptions.ConnectionError:
        return [], "无法连接到目标网站，请检查网络或地址是否正确。"
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "未知"
        return [], f"网页地址返回 HTTP 错误（状态码：{status}），请确认地址有效。"
    except requests.exceptions.RequestException:
        return [], "网络请求失败，请检查网页地址或网络连接。"

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "xml" in content_type and ("rss" in content_type or "atom" in content_type):
        return [], "该地址看起来是 RSS 订阅源，请使用 RSS 地址或改用包含 feed/rss 的 URL 自动识别。"

    try:
        articles = _parse_list_page(response.content, response.url, brand_name)
    except Exception:
        return [], "网页内容解析失败，请确认地址指向可访问的文章列表页。"

    if not articles:
        return [], "网页中未识别到有效的文章标题与链接，请尝试其他列表页地址。"

    return articles, None

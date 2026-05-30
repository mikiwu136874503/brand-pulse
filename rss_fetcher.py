import calendar
import re
from html import unescape

import feedparser
import requests

REQUEST_TIMEOUT = 15
USER_AGENT = "BrandPulse/1.0 (+https://github.com/brand-pulse)"


def _strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", text)
    return unescape(cleaned).strip()


def _parse_entry_date(entry: dict) -> str:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                timestamp = calendar.timegm(parsed)
                from datetime import datetime

                return datetime.utcfromtimestamp(timestamp).isoformat(timespec="seconds")
            except (OverflowError, ValueError, TypeError):
                continue
    for key in ("published", "updated", "created"):
        raw = entry.get(key)
        if raw:
            return str(raw)
    return ""


def fetch_rss(rss_url: str, brand_name: str) -> tuple[list[dict], str | None]:
    """抓取 RSS  feed，返回 (文章列表, 错误信息)。成功时错误信息为 None。"""
    try:
        response = requests.get(
            rss_url.strip(),
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return [], "网络请求超时，请检查 RSS 地址或稍后重试。"
    except requests.exceptions.ConnectionError:
        return [], "无法连接到 RSS 服务器，请检查网络或地址是否正确。"
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "未知"
        return [], f"RSS 地址返回 HTTP 错误（状态码：{status}），请确认地址有效。"
    except requests.exceptions.RequestException:
        return [], "网络请求失败，请检查 RSS 地址或网络连接。"

    try:
        feed = feedparser.parse(response.content)
    except Exception:
        return [], "RSS 内容解析失败，请确认地址指向有效的 RSS/Atom 订阅源。"

    if feed.bozo and not feed.entries:
        bozo_msg = str(feed.bozo_exception) if feed.bozo_exception else "格式不符合标准"
        return [], f"RSS 格式错误，无法解析：{bozo_msg}"

    if not feed.entries:
        return [], "RSS 源中未找到任何文章，请确认订阅源是否有内容。"

    source_title = ""
    if feed.feed:
        source_title = feed.feed.get("title", "") or rss_url.strip()

    articles = []
    for entry in feed.entries:
        link = (entry.get("link") or entry.get("id") or "").strip()
        if not link:
            continue

        title = (entry.get("title") or "无标题").strip()
        summary = _strip_html(
            entry.get("summary") or entry.get("description") or ""
        )

        articles.append(
            {
                "brand_name": brand_name,
                "title": title,
                "link": link,
                "summary": summary,
                "published": _parse_entry_date(entry),
                "source": source_title,
            }
        )

    if not articles:
        return [], "RSS 源中的文章缺少有效链接，无法入库。"

    return articles, None

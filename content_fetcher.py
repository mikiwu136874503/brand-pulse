"""统一内容采集入口：根据 URL / source_type 选择 RSS 或网页抓取。"""

from rss_fetcher import fetch_rss
from web_scraper import fetch_web_list

RSS_URL_HINTS = ("/feed", "/rss", ".xml", "feed=", "rss=", "atom.xml", "atom/")


def detect_source_type(url: str) -> str:
    """根据 URL 判断采集方式：rss 或 web。"""
    normalized = (url or "").strip().lower()
    if not normalized:
        return "web"
    if normalized.endswith(".xml"):
        return "rss"
    if any(hint in normalized for hint in RSS_URL_HINTS):
        return "rss"
    last_segment = normalized.rstrip("/").split("/")[-1]
    if last_segment in ("feed", "rss", "atom"):
        return "rss"
    return "web"


def fetch_brand_content(
    source_url: str,
    brand_name: str,
    source_type: str | None = None,
) -> tuple[list[dict], str | None]:
    """
    抓取品牌内容源。
    返回 (文章列表, 错误信息)。成功时错误信息为 None。
    """
    url = (source_url or "").strip()
    if not url:
        return [], "请输入有效的 RSS 地址或网页地址。"

    mode = (source_type or detect_source_type(url)).lower()
    if mode == "rss":
        return fetch_rss(url, brand_name)
    return fetch_web_list(url, brand_name)

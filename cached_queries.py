"""Streamlit 查询缓存层（ttl=60s），减轻筛选切换时的数据库压力。"""

from __future__ import annotations

import streamlit as st

import db

ARTICLE_LIST_CACHE_TTL = 60


@st.cache_data(ttl=ARTICLE_LIST_CACHE_TTL, show_spinner=False)
def cached_list_articles(
    brand_name: str | None,
    favorites_only: bool,
    published_start: str | None,
    published_end: str | None,
    category: str | None,
) -> list[dict]:
    return db.list_articles(
        brand_name=brand_name,
        favorites_only=favorites_only,
        published_start=published_start,
        published_end=published_end,
        category=category,
    )


@st.cache_data(ttl=ARTICLE_LIST_CACHE_TTL, show_spinner=False)
def cached_list_brands() -> list[dict]:
    return db.list_brands()


@st.cache_data(ttl=ARTICLE_LIST_CACHE_TTL, show_spinner=False)
def cached_article_counts_by_brand() -> dict[str, int]:
    return db.count_articles_by_brand()


@st.cache_data(ttl=ARTICLE_LIST_CACHE_TTL, show_spinner=False)
def cached_count_favorite_articles(brand_name: str | None) -> int:
    return db.count_favorite_articles(brand_name)


@st.cache_data(ttl=ARTICLE_LIST_CACHE_TTL, show_spinner=False)
def cached_count_uncategorized_articles(brand_name: str | None) -> int:
    return db.count_uncategorized_articles(brand_name)


@st.cache_data(ttl=ARTICLE_LIST_CACHE_TTL, show_spinner=False)
def cached_count_articles_without_ai_summary(brand_name: str | None) -> int:
    return db.count_articles_without_ai_summary(brand_name)


def clear_query_caches() -> None:
    """写入数据库后清除缓存，避免长时间展示过期数据。"""
    cached_list_articles.clear()
    cached_list_brands.clear()
    cached_article_counts_by_brand.clear()
    cached_count_favorite_articles.clear()
    cached_count_uncategorized_articles.clear()
    cached_count_articles_without_ai_summary.clear()

import html
import json
import os
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import db
from cached_queries import (
    cached_article_counts_by_brand,
    cached_count_articles_without_ai_summary,
    cached_count_favorite_articles,
    cached_count_uncategorized_articles,
    cached_list_articles,
    cached_list_brands,
    clear_query_caches,
)
from ai_classifier import CATEGORIES, classify_article
from ai_summarizer import generate_summary
from brand_comparison import (
    build_comparison_data,
    get_category_distribution,
    render_keyword_wordclouds,
)
from brand_tone_analyzer import analyze_brand_tone
from content_strategy_generator import generate_content_strategy
from gap_analyzer import generate_gap_analysis
from competitor_activity_extractor import extract_competitor_activities
from content_fetcher import detect_source_type, fetch_brand_content

load_dotenv()

# ---------------------------------------------------------------------------
# 页面配置（必须是第一个 Streamlit 命令）
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Brand Pulse · 竞争品牌内容雷达",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 主题色
C_BG = "#0B0F19"
C_CARD = "#131A2B"
C_BORDER = "#2A3A5C"
C_TEXT = "#FFFFFF"
C_TEXT_SEC = "#B0C4DE"
C_ACCENT = "#00D4FF"
C_SUCCESS = "#00FFAA"
C_DANGER = "#FF4757"

CATEGORY_STYLES = {
    "案例研究": {"bg": "#00D4FF", "text": "#0B0F19"},
    "产品更新": {"bg": "#00FFAA", "text": "#0B0F19"},
    "行业洞察": {"bg": "#FF9F43", "text": "#0B0F19"},
    "技术博客": {"bg": "#A78BFA", "text": "#0B0F19"},
    "其他": {"bg": "#5A6A8A", "text": "#FFFFFF"},
}

PLOTLY_GRID = "#1E2D4A"
PLOTLY_COLORWAY = ["#00D4FF", "#00FFAA", "#A78BFA", "#FF9F43", "#FF4757"]


def invalidate_data_cache() -> None:
    """数据库写入后刷新共享缓存与 st.cache_data。"""
    clear_query_caches()
    for key in ("brands", "brand_names", "article_counts_by_brand"):
        st.session_state.pop(key, None)


def ensure_shared_data_cache() -> None:
    """页面级共享数据：品牌列表与按品牌文章计数。"""
    if "brands" not in st.session_state:
        st.session_state.brands = cached_list_brands()
    if "article_counts_by_brand" not in st.session_state:
        st.session_state.article_counts_by_brand = cached_article_counts_by_brand()


def get_brands() -> list[dict]:
    ensure_shared_data_cache()
    return st.session_state.brands


def get_brand_names() -> list[str]:
    if "brand_names" not in st.session_state:
        st.session_state.brand_names = ["全部品牌"] + [
            b["brand_name"] for b in get_brands()
        ]
    return st.session_state.brand_names


def get_article_counts_by_brand() -> dict[str, int]:
    ensure_shared_data_cache()
    return st.session_state.article_counts_by_brand


def render_html_action_button(
    label: str,
    *,
    html_id: str,
    disabled: bool = False,
) -> None:
    """在列内渲染 HTML 主按钮（可见）；须再调用 wire_html_button_to_streamlit 绑定隐藏 st.button。"""
    safe_label = html.escape(label)
    disabled_attr = "disabled" if disabled else ""
    cursor_style = "not-allowed; opacity: 0.45;" if disabled else "pointer;"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;width:100%;min-height:2.5rem;">
          <button type="button" id="{html_id}" {disabled_attr}
            style="width:100%;height:2.5rem;line-height:2.5rem;padding:0 12px;
            margin:0;background:{C_ACCENT};color:{C_BG};font-weight:700;border:none;
            border-radius:6px;box-sizing:border-box;font-family:Inter,Poppins,sans-serif;
            font-size:0.85rem;cursor:{cursor_style}">
            {safe_label}
          </button>
        </div>
        """,
        unsafe_allow_html=True,
    )


def wire_html_button_to_streamlit(html_id: str, streamlit_key: str) -> None:
    """将 HTML 按钮点击转发到已渲染的隐藏 st.button。"""
    html_id_js = json.dumps(html_id)
    streamlit_key_js = json.dumps(streamlit_key)
    components.html(
        f"""
        <script>
        (function () {{
            const doc = window.parent.document;
            const htmlId = {html_id_js};
            const stKey = {streamlit_key_js};
            function wire() {{
                const htmlBtn = doc.getElementById(htmlId);
                const stRoot = doc.querySelector(".st-key-" + stKey);
                if (!htmlBtn || !stRoot) return;
                const stBtn = stRoot.querySelector("button");
                if (!stBtn) return;
                if (htmlBtn.__bpHtmlBtnWired) return;
                htmlBtn.__bpHtmlBtnWired = true;
                htmlBtn.addEventListener("click", function (ev) {{
                    ev.preventDefault();
                    if (htmlBtn.disabled || stBtn.disabled) return;
                    stBtn.click();
                }});
            }}
            wire();
            [50, 150, 400, 900, 1500].forEach(function (ms) {{ setTimeout(wire, ms); }});
        }})();
        </script>
        """,
        height=0,
    )


def render_hidden_streamlit_button(
    label: str,
    *,
    streamlit_key: str,
    disabled: bool = False,
) -> bool:
    """渲染隐藏的 st.button，供 HTML 按钮通过 JS 触发。"""
    clicked = st.button(
        label,
        key=streamlit_key,
        type="primary",
        use_container_width=True,
        disabled=disabled,
    )
    streamlit_key_js = json.dumps(streamlit_key)
    components.html(
        f"""
        <script>
        (function () {{
            const doc = window.parent.document;
            const stKey = {streamlit_key_js};
            function hide() {{
                const root = doc.querySelector(".st-key-" + stKey);
                if (!root) return;
                root.style.setProperty("position", "absolute", "important");
                root.style.setProperty("left", "-10000px", "important");
                root.style.setProperty("width", "1px", "important");
                root.style.setProperty("height", "1px", "important");
                root.style.setProperty("overflow", "hidden", "important");
                root.style.setProperty("opacity", "0", "important");
                root.style.setProperty("pointer-events", "none", "important");
                root.style.setProperty("margin", "0", "important");
                root.style.setProperty("padding", "0", "important");
            }}
            hide();
            [50, 150, 400, 900].forEach(function (ms) {{ setTimeout(hide, ms); }});
        }})();
        </script>
        """,
        height=0,
    )
    return clicked


def wire_select_html_button_row(select_key: str, html_button_id: str) -> None:
    """下拉框 + HTML 按钮行：flex 居中对齐。"""
    select_key_js = json.dumps(select_key)
    html_id_js = json.dumps(html_button_id)
    components.html(
        f"""
        <script>
        (function () {{
            const doc = window.parent.document;
            const selectKey = {select_key_js};
            const htmlId = {html_id_js};
            function apply() {{
                const selRoot = doc.querySelector(".st-key-" + selectKey);
                const htmlBtn = doc.getElementById(htmlId);
                if (!selRoot || !htmlBtn) return;
                const row = selRoot.closest('[data-testid="stHorizontalBlock"]');
                if (!row) return;
                row.style.setProperty("display", "flex", "important");
                row.style.setProperty("align-items", "center", "important");
                row.style.setProperty("flex-wrap", "nowrap", "important");
                row.style.setProperty("gap", "0.75rem", "important");
                row.querySelectorAll('[data-testid="column"]').forEach(function (col) {{
                    col.style.setProperty("display", "flex", "important");
                    col.style.setProperty("align-items", "center", "important");
                }});
            }}
            apply();
            [50, 150, 400, 900].forEach(function (ms) {{ setTimeout(apply, ms); }});
        }})();
        </script>
        """,
        height=0,
    )


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', 'Poppins', sans-serif !important;
            color: {C_TEXT_SEC} !important;
        }}

        .stApp {{
            background-color: {C_BG} !important;
            background-image:
                linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px),
                radial-gradient(ellipse at 50% 0%, rgba(0, 212, 255, 0.08) 0%, transparent 60%) !important;
            background-size: 48px 48px, 48px 48px, 100% 100% !important;
        }}

        .block-container {{
            padding-top: 1.5rem;
            max-width: 1400px;
            background: transparent !important;
        }}

        .main .block-container {{
            background: transparent !important;
        }}

        h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {C_TEXT} !important;
            font-weight: 700 !important;
        }}

        p, li, .stMarkdown p, .stMarkdown li {{
            color: {C_TEXT_SEC} !important;
        }}

        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {C_TEXT_SEC} !important;
        }}

        label, .stSelectbox label, .stTextInput label, .stTextArea label {{
            color: {C_TEXT_SEC} !important;
        }}

        /* 顶部标题栏 */
        .bp-hero {{
            text-align: center;
            padding: 2rem 1rem 1.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid {C_BORDER};
            background: linear-gradient(180deg, rgba(19,26,43,0.9) 0%, rgba(11,15,25,0) 100%);
        }}
        .bp-hero-title {{
            font-size: 2.6rem;
            font-weight: 700;
            color: {C_TEXT};
            letter-spacing: 0.08em;
            text-shadow: 0 0 20px rgba(0,212,255,0.6), 0 0 40px rgba(0,212,255,0.2);
            margin: 0;
        }}
        .bp-hero-sub {{
            font-size: 1rem;
            color: {C_TEXT_SEC};
            margin-top: 0.5rem;
            letter-spacing: 0.12em;
        }}

        /* 侧边栏 */
        section[data-testid="stSidebar"] {{
            background-color: {C_BG} !important;
            border-right: 1px solid {C_BORDER} !important;
        }}
        section[data-testid="stSidebar"] > div {{
            background-color: {C_BG} !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {C_ACCENT} !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.05em;
        }}
        /* 侧边栏 - 功能导航（按钮式，与 success 提示同宽） */
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
            width: 100% !important;
            display: flex !important;
            justify-content: flex-start !important;
            text-align: left !important;
            padding: 0.65rem 0.85rem !important;
            margin: 0 0 0.35rem 0 !important;
            border-radius: 6px !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {{
            border: none !important;
            background: transparent !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            width: auto !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {{
            background: transparent !important;
            color: {C_TEXT_SEC} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] span {{
            color: {C_TEXT_SEC} !important;
            -webkit-text-fill-color: {C_TEXT_SEC} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {{
            background: rgba(0, 212, 255, 0.06) !important;
            color: {C_TEXT} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover span {{
            color: {C_TEXT} !important;
            -webkit-text-fill-color: {C_TEXT} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {{
            background: rgba(0, 212, 255, 0.12) !important;
            color: {C_TEXT} !important;
            border-left: 3px solid {C_ACCENT} !important;
            font-weight: 600 !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] span {{
            color: {C_TEXT} !important;
            -webkit-text-fill-color: {C_TEXT} !important;
            font-weight: 600 !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: {C_BORDER} !important;
        }}

        /* 表单区域 */
        [data-testid="stForm"] {{
            background: {C_CARD} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            padding: 1rem 1.25rem !important;
        }}
        [data-testid="stForm"] > div {{
            background: transparent !important;
        }}

        /* 卡片容器 */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {C_CARD} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: {C_ACCENT} !important;
            box-shadow: 0 0 12px rgba(0, 212, 255, 0.15);
        }}

        /* 折叠面板 */
        [data-testid="stExpander"] {{
            background: {C_CARD} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            overflow: hidden;
        }}
        [data-testid="stExpander"] details {{
            background: {C_CARD} !important;
        }}
        [data-testid="stExpander"] summary {{
            background: #1E2D4A !important;
            color: {C_TEXT} !important;
            border-bottom: 1px solid {C_BORDER} !important;
        }}
        .streamlit-expanderHeader {{
            background-color: #1E2D4A !important;
            color: {C_TEXT} !important;
            border-bottom: 1px solid {C_BORDER} !important;
        }}
        .streamlit-expanderHeader p, .streamlit-expanderHeader span {{
            color: {C_TEXT} !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            color: {C_ACCENT} !important;
        }}
        [data-testid="stExpander"] summary span {{
            color: {C_TEXT} !important;
        }}
        [data-testid="stExpanderDetails"] {{
            background: {C_CARD} !important;
            border-top: 1px solid {C_BORDER} !important;
        }}
        [data-testid="stExpanderDetails"] > div {{
            background: {C_CARD} !important;
            color: {C_TEXT} !important;
        }}
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] li,
        [data-testid="stExpanderDetails"] span {{
            color: {C_TEXT} !important;
        }}

        /* ===== 按钮全局穿透 ===== */
        /* 内层文字节点：禁止单独画框，避免出现「字外一圈框」 */
        .stButton > button p, .stButton > button span,
        .stFormSubmitButton > button p, .stFormSubmitButton > button span,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {{
            border: none !important;
            background: transparent !important;
            border-radius: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
            outline: none !important;
            text-shadow: none !important;
            -webkit-text-stroke: 0 !important;
            text-stroke: 0 !important;
        }}
        button, button[kind],
        button:focus, button:focus-visible,
        .stButton > button, .stButton > button:focus, .stButton > button:focus-visible,
        .stFormSubmitButton > button,
        .stFormSubmitButton > button:focus,
        .stFormSubmitButton > button:focus-visible {{
            font-weight: 600 !important;
            border-radius: 6px !important;
            outline: none !important;
            box-shadow: none !important;
            text-shadow: none !important;
            -webkit-text-stroke: 0 !important;
            text-stroke: 0 !important;
        }}
        .stButton > button p, .stButton > button span,
        .stFormSubmitButton > button p, .stFormSubmitButton > button span {{
            font-weight: inherit !important;
            -webkit-text-fill-color: currentColor !important;
        }}
        button[kind="primary"],
        .stButton > button[kind="primary"],
        .stFormSubmitButton > button[kind="primaryFormSubmit"] {{
            background-color: {C_ACCENT} !important;
            color: {C_BG} !important;
            border: 2px solid {C_ACCENT} !important;
            font-weight: bold !important;
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div,
        .stFormSubmitButton > button[kind="primaryFormSubmit"] p,
        .stFormSubmitButton > button[kind="primaryFormSubmit"] span,
        .stFormSubmitButton > button[kind="primaryFormSubmit"] div {{
            color: {C_BG} !important;
            -webkit-text-fill-color: {C_BG} !important;
            font-weight: bold !important;
        }}
        .stButton > button[kind="primary"]:not(:disabled),
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:not(:disabled) {{
            color: {C_BG} !important;
        }}
        button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {{
            background-color: {C_ACCENT} !important;
            color: {C_BG} !important;
            border-color: {C_ACCENT} !important;
            box-shadow: none !important;
        }}
        .stButton > button[kind="primary"]:hover p,
        .stButton > button[kind="primary"]:hover span,
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover p,
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover span {{
            color: {C_BG} !important;
            -webkit-text-fill-color: {C_BG} !important;
        }}
        button[kind="primary"]:disabled,
        .stButton > button[kind="primary"]:disabled,
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:disabled {{
            background-color: #1E2D4A !important;
            color: #5A6A8A !important;
            border: 1px solid {C_BORDER} !important;
        }}
        .stButton > button[kind="primary"]:disabled p,
        .stButton > button[kind="primary"]:disabled span,
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:disabled p,
        .stFormSubmitButton > button[kind="primaryFormSubmit"]:disabled span {{
            color: #5A6A8A !important;
            -webkit-text-fill-color: #5A6A8A !important;
        }}
        button[kind="secondary"],
        .stButton > button[kind="secondary"],
        .stFormSubmitButton > button[kind="secondaryFormSubmit"] {{
            background-color: transparent !important;
            color: #8899AA !important;
            border: 1px solid {C_BORDER} !important;
        }}
        .stButton > button[kind="secondary"] p,
        .stButton > button[kind="secondary"] span,
        .stFormSubmitButton > button[kind="secondaryFormSubmit"] p,
        .stFormSubmitButton > button[kind="secondaryFormSubmit"] span {{
            color: #8899AA !important;
            -webkit-text-fill-color: #8899AA !important;
        }}
        button[kind="secondary"]:hover,
        .stButton > button[kind="secondary"]:hover {{
            background-color: rgba(0, 212, 255, 0.06) !important;
            color: {C_TEXT_SEC} !important;
            border-color: {C_ACCENT} !important;
        }}
        .stButton > button[kind="secondary"]:hover p,
        .stButton > button[kind="secondary"]:hover span {{
            color: {C_TEXT_SEC} !important;
            -webkit-text-fill-color: {C_TEXT_SEC} !important;
        }}
        button[kind="secondary"]:disabled,
        .stButton > button[kind="secondary"]:disabled {{
            background-color: transparent !important;
            color: #8899AA !important;
            border: 1px solid {C_BORDER} !important;
            opacity: 0.55 !important;
        }}
        .stButton > button[kind="secondary"]:disabled p,
        .stButton > button[kind="secondary"]:disabled span {{
            color: #8899AA !important;
            -webkit-text-fill-color: #8899AA !important;
        }}
        .bp-action-btn-zone + div[data-testid="stButton"] > button {{
            background: {C_BORDER} !important;
            color: {C_ACCENT} !important;
            border: 1px solid {C_ACCENT} !important;
        }}
        .bp-action-btn-zone + div[data-testid="stButton"] > button p,
        .bp-action-btn-zone + div[data-testid="stButton"] > button span {{
            color: {C_ACCENT} !important;
            -webkit-text-fill-color: {C_ACCENT} !important;
        }}
        .bp-delete-zone + div[data-testid="stButton"] > button {{
            background: {C_DANGER} !important;
            color: {C_TEXT} !important;
            border: none !important;
        }}
        .bp-delete-zone + div[data-testid="stButton"] > button p,
        .bp-delete-zone + div[data-testid="stButton"] > button span {{
            color: {C_TEXT} !important;
            -webkit-text-fill-color: {C_TEXT} !important;
        }}
        /* 样式标记占位：不占垂直空间，避免按钮错位 */
        .bp-style-marker {{
            display: block !important;
            height: 0 !important;
            width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            line-height: 0 !important;
            border: none !important;
            font-size: 0 !important;
        }}
        /* 表单行：下拉框与按钮底对齐（避免标签把按钮顶到中间） */
        .bp-form-action-anchor + div[data-testid="stHorizontalBlock"] {{
            align-items: flex-end !important;
        }}
        .bp-form-action-anchor + div[data-testid="stHorizontalBlock"] [data-testid="column"] {{
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
        }}
        /* 收藏按钮 */
        .bp-favorite-active + div[data-testid="stButton"] > button {{
            background: rgba(255, 193, 7, 0.12) !important;
            color: #FFC107 !important;
            border: 1px solid rgba(255, 193, 7, 0.45) !important;
            font-weight: 600 !important;
            transition: border-color 0.2s ease !important;
        }}
        .bp-favorite-active + div[data-testid="stButton"] > button p,
        .bp-favorite-active + div[data-testid="stButton"] > button span {{
            color: #FFC107 !important;
            -webkit-text-fill-color: #FFC107 !important;
        }}
        .bp-favorite-active + div[data-testid="stButton"] > button:hover {{
            border-color: #FFC107 !important;
            box-shadow: none !important;
        }}
        .bp-favorite-inactive + div[data-testid="stButton"] > button {{
            background: transparent !important;
            color: #5A6A8A !important;
            border: 1px solid {C_BORDER} !important;
            transition: color 0.2s ease, border-color 0.2s ease !important;
        }}
        .bp-favorite-inactive + div[data-testid="stButton"] > button p,
        .bp-favorite-inactive + div[data-testid="stButton"] > button span {{
            color: #5A6A8A !important;
            -webkit-text-fill-color: #5A6A8A !important;
        }}
        .bp-favorite-inactive + div[data-testid="stButton"] > button:hover {{
            border-color: rgba(255, 193, 7, 0.5) !important;
            box-shadow: none !important;
        }}
        .bp-favorite-inactive + div[data-testid="stButton"] > button:hover p,
        .bp-favorite-inactive + div[data-testid="stButton"] > button:hover span {{
            color: #FFC107 !important;
            -webkit-text-fill-color: #FFC107 !important;
        }}
        /* 文章样本选择复选框 */
        .bp-pick-zone + div[data-testid="stCheckbox"] label {{
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }}
        .bp-pick-zone + div[data-testid="stCheckbox"] label > div:first-child {{
            background: rgba(19, 26, 43, 0.6) !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 4px !important;
            transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease !important;
        }}
        .bp-pick-checked + div[data-testid="stCheckbox"] label > div:first-child {{
            background: rgba(0, 212, 255, 0.14) !important;
            border: 1px solid {C_ACCENT} !important;
            box-shadow: 0 0 8px rgba(0, 212, 255, 0.35) !important;
        }}
        .bp-pick-checked + div[data-testid="stCheckbox"] label > div:first-child svg {{
            color: {C_ACCENT} !important;
            stroke: {C_ACCENT} !important;
        }}
        /* 一键更新全部品牌按钮 */
        .bp-fetch-all-zone + div[data-testid="stButton"] > button {{
            background: {C_ACCENT} !important;
            color: {C_BG} !important;
            border: 2px solid {C_ACCENT} !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }}
        .bp-fetch-all-zone + div[data-testid="stButton"] > button p,
        .bp-fetch-all-zone + div[data-testid="stButton"] > button span,
        .bp-fetch-all-zone + div[data-testid="stButton"] > button div {{
            color: {C_BG} !important;
            -webkit-text-fill-color: {C_BG} !important;
            font-weight: 700 !important;
        }}
        .bp-fetch-all-zone + div[data-testid="stButton"] > button:hover {{
            background: {C_ACCENT} !important;
            color: {C_BG} !important;
            border-color: {C_ACCENT} !important;
            box-shadow: none !important;
        }}
        .bp-fetch-all-zone + div[data-testid="stButton"] > button:hover p,
        .bp-fetch-all-zone + div[data-testid="stButton"] > button:hover span {{
            color: {C_BG} !important;
            -webkit-text-fill-color: {C_BG} !important;
        }}
        .bp-fetch-all-zone + div[data-testid="stButton"] > button:disabled {{
            background: rgba(0, 212, 255, 0.25) !important;
            color: rgba(11, 15, 25, 0.55) !important;
            box-shadow: none !important;
        }}
        .bp-fetch-all-zone + div[data-testid="stButton"] > button:disabled p,
        .bp-fetch-all-zone + div[data-testid="stButton"] > button:disabled span {{
            color: rgba(11, 15, 25, 0.55) !important;
            -webkit-text-fill-color: rgba(11, 15, 25, 0.55) !important;
        }}
        a[data-testid="stBaseLinkButton"],
        a[data-testid="stLinkButton"], [data-testid="stLinkButton"] a,
        a[data-testid="stBaseLinkButton"] p, [data-testid="stLinkButton"] a p {{
            background-color: {C_BORDER} !important;
            color: {C_TEXT_SEC} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 6px !important;
            text-decoration: none !important;
        }}

        /* ===== 输入框 / 下拉框穿透（消除双边框） ===== */
        .stTextInput > div,
        .stTextInput > div > div,
        .stTextArea > div,
        .stTextArea > div > div,
        .stNumberInput > div,
        .stNumberInput > div > div,
        [data-baseweb="input"],
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] {{
            background: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        input, textarea,
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background-color: {C_CARD} !important;
            color: {C_TEXT} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 6px !important;
            caret-color: {C_ACCENT} !important;
            -webkit-text-fill-color: {C_TEXT} !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        .stSelectbox select {{
            position: absolute !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            pointer-events: none !important;
        }}
        div[data-baseweb="select"] input {{
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            outline: none !important;
            color: {C_TEXT} !important;
            -webkit-text-fill-color: {C_TEXT} !important;
            cursor: pointer !important;
        }}
        input::placeholder, textarea::placeholder,
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color: #5A6A8A !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #5A6A8A !important;
        }}
        /* 日期输入框 */
        .stDateInput > div,
        .stDateInput > div > div,
        .stDateInput [data-baseweb="input"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        .stDateInput input {{
            background-color: {C_CARD} !important;
            color: {C_TEXT} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 6px !important;
            caret-color: {C_ACCENT} !important;
        }}
        .stDateInput input::placeholder {{
            color: #5A6A8A !important;
        }}
        .stDateInput input:focus {{
            border-color: {C_ACCENT} !important;
            box-shadow: 0 0 8px rgba(0, 212, 255, 0.25) !important;
        }}
        /* 时间筛选 radio */
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
            border-color: {C_BORDER} !important;
            background: {C_CARD} !important;
        }}
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {{
            border-color: {C_ACCENT} !important;
            background: rgba(0, 212, 255, 0.12) !important;
        }}
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child svg {{
            fill: {C_ACCENT} !important;
        }}
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"] p,
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"] span {{
            color: {C_TEXT_SEC} !important;
        }}
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] p,
        .bp-time-filter-wrap [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] span {{
            color: {C_ACCENT} !important;
            font-weight: 600 !important;
        }}
        /* Tabs */
        [data-baseweb="tab-list"] {{
            gap: 10px !important;
            border-bottom: 1px solid {C_BORDER} !important;
            margin-bottom: 0.6rem !important;
        }}
        [data-baseweb="tab"] {{
            background: transparent !important;
            color: #8899AA !important;
            border: 1px solid transparent !important;
            border-bottom: none !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 0.45rem 0.85rem !important;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            color: {C_ACCENT} !important;
            border-color: {C_ACCENT} !important;
            background: rgba(0, 212, 255, 0.08) !important;
        }}
        .bp-article-filter-wrap {{
            margin: 0.25rem 0 0.75rem 0;
        }}
        .bp-article-filter-wrap .stSelectbox label,
        .bp-article-filter-wrap .stCheckbox label {{
            color: {C_TEXT_SEC} !important;
        }}
        /* 样本工具栏三按钮：.bp-toolbar-btn（通过 st-key-* 绑定） */
        div[data-testid="stHorizontalBlock"]:has(.st-key-pick_all_visible):has(
            .st-key-set_compare_sample
        ) {{
            flex-wrap: nowrap !important;
            align-items: center !important;
            gap: 0.12rem !important;
            margin-bottom: 0.35rem !important;
        }}
        div[data-testid="stHorizontalBlock"]:has(.st-key-pick_all_visible):has(
            .st-key-set_compare_sample
        ) [data-testid="column"] {{
            min-width: 0 !important;
            align-self: center !important;
        }}
        /* 样本工具栏：三按钮统一为亮青底 + 深色字（.bp-toolbar-btn） */
        .st-key-pick_all_visible div[data-testid="stButton"] > button,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button,
        .st-key-set_compare_sample div[data-testid="stButton"] > button {{
            height: 2.8rem !important;
            min-height: 2.8rem !important;
            max-height: 2.8rem !important;
            line-height: 2.8rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 0.35rem !important;
            padding-right: 0.35rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            white-space: nowrap !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            box-sizing: border-box !important;
            border-radius: 6px !important;
            background-color: {C_ACCENT} !important;
            color: {C_BG} !important;
            border: 2px solid {C_ACCENT} !important;
            box-shadow: none !important;
        }}
        .st-key-pick_all_visible div[data-testid="stButton"] > button p,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button p,
        .st-key-set_compare_sample div[data-testid="stButton"] > button p,
        .st-key-pick_all_visible div[data-testid="stButton"] > button span,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button span,
        .st-key-set_compare_sample div[data-testid="stButton"] > button span {{
            white-space: nowrap !important;
            line-height: 1 !important;
            margin: 0 !important;
            color: {C_BG} !important;
            -webkit-text-fill-color: {C_BG} !important;
            font-weight: 600 !important;
        }}
        .st-key-pick_all_visible div[data-testid="stButton"] > button:hover,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button:hover,
        .st-key-set_compare_sample div[data-testid="stButton"] > button:hover {{
            background-color: {C_ACCENT} !important;
            color: {C_BG} !important;
            border-color: {C_ACCENT} !important;
            box-shadow: none !important;
        }}
        .st-key-pick_all_visible div[data-testid="stButton"] > button:hover p,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button:hover p,
        .st-key-set_compare_sample div[data-testid="stButton"] > button:hover p,
        .st-key-pick_all_visible div[data-testid="stButton"] > button:hover span,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button:hover span,
        .st-key-set_compare_sample div[data-testid="stButton"] > button:hover span {{
            color: {C_BG} !important;
            -webkit-text-fill-color: {C_BG} !important;
        }}
        .st-key-pick_all_visible div[data-testid="stButton"] > button:disabled,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button:disabled,
        .st-key-set_compare_sample div[data-testid="stButton"] > button:disabled {{
            background-color: #1E2D4A !important;
            color: #5A6A8A !important;
            border: 1px solid {C_BORDER} !important;
            opacity: 1 !important;
        }}
        .st-key-pick_all_visible div[data-testid="stButton"] > button:disabled p,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button:disabled p,
        .st-key-set_compare_sample div[data-testid="stButton"] > button:disabled p,
        .st-key-pick_all_visible div[data-testid="stButton"] > button:disabled span,
        .st-key-pick_clear_visible div[data-testid="stButton"] > button:disabled span,
        .st-key-set_compare_sample div[data-testid="stButton"] > button:disabled span {{
            color: #5A6A8A !important;
            -webkit-text-fill-color: #5A6A8A !important;
        }}
        input:focus, textarea:focus,
        .stTextInput input:focus, .stTextArea textarea:focus,
        .stNumberInput input:focus {{
            border: 1px solid {C_ACCENT} !important;
            outline: none !important;
            box-shadow: none !important;
            color: {C_TEXT} !important;
        }}
        .stSelectbox > div,
        .stSelectbox > div > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div > div {{
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        div[data-baseweb="select"] {{
            background-color: {C_CARD} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 6px !important;
            outline: none !important;
            box-shadow: none !important;
            cursor: pointer !important;
        }}
        div[data-baseweb="select"]:focus-within {{
            border-color: {C_ACCENT} !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[value] {{
            color: {C_TEXT} !important;
            -webkit-text-fill-color: {C_TEXT} !important;
        }}
        div[data-baseweb="select"] > div:last-child {{
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            min-width: unset !important;
        }}
        div[data-baseweb="select"] svg {{
            fill: {C_TEXT_SEC} !important;
        }}
        [data-baseweb="popover"], [data-baseweb="popover"] ul, [data-baseweb="menu"], [role="listbox"] {{
            background-color: {C_CARD} !important;
            border: 1px solid {C_BORDER} !important;
        }}
        [data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="option"] {{
            color: {C_TEXT} !important;
            background: {C_CARD} !important;
        }}
        [data-baseweb="popover"] li:hover, [role="option"]:hover {{
            background: #1E2D4A !important;
            color: {C_TEXT} !important;
        }}

        /* 指标卡片 */
        [data-testid="stMetric"] {{
            background: linear-gradient(180deg, {C_CARD} 0%, rgba(19,26,43,0.6) 100%) !important;
            border: 1px solid {C_BORDER} !important;
            border-top: 2px solid {C_ACCENT} !important;
            border-radius: 8px !important;
            padding: 1rem 1.2rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {C_TEXT_SEC} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {C_TEXT} !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricDelta"] {{
            color: {C_SUCCESS} !important;
        }}

        /* 提示框 */
        [data-testid="stAlert"] {{
            background: {C_CARD} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            color: {C_TEXT_SEC} !important;
        }}
        [data-testid="stAlert"] * {{
            color: inherit !important;
        }}

        /* 分割线 */
        hr, [data-testid="stDivider"] {{
            border-color: {C_BORDER} !important;
            background-color: {C_BORDER} !important;
            opacity: 1 !important;
        }}

        /* ===== 表格穿透 ===== */
        table, .dataframe, [data-testid="stTable"], [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div, [data-testid="stDataFrame"] div[role="grid"],
        .stDataFrame, .dvn-scroller {{
            background-color: {C_CARD} !important;
            color: {C_TEXT_SEC} !important;
            border-color: {C_BORDER} !important;
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }}
        table thead tr th, .dataframe thead tr th,
        [data-testid="stTable"] thead tr th, [data-testid="stTable"] table thead th,
        [data-testid="stDataFrame"] thead th,
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] div[role="columnheader"] {{
            background-color: #1E2D4A !important;
            color: {C_ACCENT} !important;
            border-color: {C_BORDER} !important;
            font-weight: 600 !important;
        }}
        table tbody tr td, .dataframe tbody tr td,
        [data-testid="stTable"] tbody tr td, [data-testid="stTable"] table tbody td,
        [data-testid="stDataFrame"] tbody td,
        [data-testid="stDataFrame"] [role="gridcell"],
        [data-testid="stDataFrame"] div[role="gridcell"] {{
            background-color: {C_CARD} !important;
            color: {C_TEXT_SEC} !important;
            border-color: {C_BORDER} !important;
        }}
        table, table th, table td, .dataframe, .dataframe th, .dataframe td {{
            border: 1px solid {C_BORDER} !important;
        }}
        table.bp-topics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 4px;
        }}
        table.bp-topics-table thead th {{
            background-color: #1E2D4A !important;
            color: {C_ACCENT} !important;
            padding: 10px 14px;
            text-align: left;
            font-weight: 600;
        }}
        table.bp-topics-table tbody td {{
            background-color: {C_CARD} !important;
            color: {C_TEXT_SEC} !important;
            padding: 10px 14px;
        }}
        table.bp-topics-table tbody tr:hover td {{
            background-color: #1A2438 !important;
        }}
        table.bp-activity-table tbody td {{
            color: {C_TEXT} !important;
        }}
        table.bp-activity-table tbody td a {{
            color: {C_ACCENT} !important;
            text-decoration: none !important;
        }}
        table.bp-activity-table tbody td a:hover {{
            text-decoration: underline !important;
        }}

        /* 词云图容器 */
        .bp-wordcloud-wrap {{
            background: {C_BG} !important;
            border: 1px solid {C_BORDER} !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }}
        [data-testid="stImage"] img {{
            border-radius: 6px;
        }}

        /* 进度条 */
        .stProgress > div > div {{
            background-color: {C_ACCENT} !important;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(11, 15, 25, 0.92) !important;
            border-bottom: 1px solid {C_BORDER} !important;
        }}

        /* 覆盖 Streamlit 默认白底 */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        section.main {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header() -> None:
    st.markdown(
        f"""
        <div class="bp-hero">
            <p class="bp-hero-title">Brand Pulse</p>
            <p class="bp-hero-sub">竞争品牌内容雷达 · AI 驱动的内容策略引擎</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_fig(fig):
    """统一 Plotly 深色科技风样式。"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Poppins, sans-serif", color=C_TEXT_SEC, size=12),
        colorway=PLOTLY_COLORWAY,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            font=dict(color=C_TEXT, size=12, family="Inter, Poppins, sans-serif"),
            bgcolor="rgba(19,26,43,0.85)",
            bordercolor=C_BORDER,
            borderwidth=1,
        ),
        title=None,
    )
    fig.update_xaxes(
        gridcolor=PLOTLY_GRID,
        zerolinecolor=PLOTLY_GRID,
        linecolor=C_BORDER,
        tickfont=dict(color=C_TEXT_SEC, size=11),
        title_font=dict(color=C_TEXT_SEC, size=12),
    )
    fig.update_yaxes(
        gridcolor=PLOTLY_GRID,
        zerolinecolor=PLOTLY_GRID,
        linecolor=C_BORDER,
        tickfont=dict(color=C_TEXT_SEC, size=11),
        title_font=dict(color=C_TEXT_SEC, size=12),
    )
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        textfont=dict(color=C_TEXT, size=11),
    )
    fig.update_traces(
        selector=dict(type="pie"),
        textfont=dict(color=C_TEXT, size=11),
        marker=dict(line=dict(color=C_BG, width=1)),
    )
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(color=C_TEXT_SEC),
    )
    return fig


def mark_delete_zone() -> None:
    st.markdown(
        '<div class="bp-style-marker bp-delete-zone"></div>',
        unsafe_allow_html=True,
    )


def build_topics_dataframe(topic_distribution: list[dict]) -> pd.DataFrame:
    """将主题分布数据规范化为展示用 DataFrame。"""
    rows: list[dict] = []
    for item in topic_distribution or []:
        if not isinstance(item, dict):
            continue
        topic = (
            item.get("topic")
            or item.get("主题")
            or item.get("name")
            or item.get("title")
            or ""
        )
        topic = str(topic).strip()
        if topic.lower() in ("undefined", "none", "null", ""):
            continue
        raw_pct = (
            item.get("percentage")
            if item.get("percentage") is not None
            else item.get("占比")
            if item.get("占比") is not None
            else item.get("percent")
            if item.get("percent") is not None
            else item.get("value")
        )
        try:
            percentage = round(float(raw_pct), 1) if raw_pct is not None else 0.0
        except (TypeError, ValueError):
            percentage = 0.0
        rows.append({"主题": topic, "占比 (%)": percentage})

    if not rows:
        return pd.DataFrame(columns=["主题", "占比 (%)"])
    return pd.DataFrame(rows)


def render_topics_table(topics_df: pd.DataFrame) -> None:
    """渲染主题分布表格（HTML），避免 st.dataframe 在深色主题下空白/undefined。"""
    if topics_df.empty:
        st.caption("暂无主题分布数据。")
        return
    rows_html = "".join(
        f"<tr><td>{html.escape(str(row['主题']))}</td>"
        f"<td>{row['占比 (%)']}%</td></tr>"
        for _, row in topics_df.iterrows()
    )
    st.markdown(
        f"""
        <table class="bp-topics-table">
            <thead><tr><th>主题</th><th>占比 (%)</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def get_cached_tone_analysis(brand_name: str) -> dict | None:
    """读取某品牌的调性分析缓存（含 topic_distribution）。"""
    by_brand = st.session_state.get("tone_analysis_by_brand") or {}
    entry = by_brand.get(brand_name)
    if entry:
        return entry
    legacy = st.session_state.get("tone_analysis_result")
    if legacy and legacy.get("brand_name") == brand_name:
        return legacy
    return None


def save_tone_analysis_cache(
    brand_name: str, result: dict, article_count: int
) -> None:
    """保存品牌调性分析结果，供单品牌展示与双品牌主题对比复用。"""
    if "tone_analysis_by_brand" not in st.session_state:
        st.session_state.tone_analysis_by_brand = {}
    st.session_state.tone_analysis_by_brand[brand_name] = {
        "brand_name": brand_name,
        "result": result,
        "article_count": article_count,
    }
    st.session_state.tone_analysis_result = st.session_state.tone_analysis_by_brand[
        brand_name
    ]


def build_combined_topics_bar_df(
    brand_a: str,
    topics_a: list[dict],
    brand_b: str,
    topics_b: list[dict],
) -> pd.DataFrame:
    """合并两品牌主题分布，用于分组柱状图对比。"""
    df_a = build_topics_dataframe(topics_a)
    df_b = build_topics_dataframe(topics_b)
    all_topics = sorted(set(df_a["主题"].tolist()) | set(df_b["主题"].tolist()))
    if not all_topics:
        return pd.DataFrame(columns=["主题", "品牌", "占比 (%)"])
    map_a = dict(zip(df_a["主题"], df_a["占比 (%)"]))
    map_b = dict(zip(df_b["主题"], df_b["占比 (%)"]))
    rows: list[dict] = []
    for topic in all_topics:
        rows.append({"主题": topic, "品牌": brand_a, "占比 (%)": map_a.get(topic, 0.0)})
        rows.append({"主题": topic, "品牌": brand_b, "占比 (%)": map_b.get(topic, 0.0)})
    return pd.DataFrame(rows)


def render_brand_topic_pie(brand_name: str, topic_distribution: list[dict]) -> None:
    """绘制单品牌主题分布饼图。"""
    topics_df = build_topics_dataframe(topic_distribution)
    if topics_df.empty:
        st.caption(f"「{brand_name}」暂无有效主题分布数据。")
        return
    plot_df = topics_df.rename(columns={"主题": "topic", "占比 (%)": "percentage"})
    fig = px.pie(
        plot_df,
        names="topic",
        values="percentage",
        hole=0.35,
        title=brand_name,
        labels={"topic": "主题", "percentage": "占比 (%)"},
        color_discrete_sequence=PLOTLY_COLORWAY,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(color=C_TEXT, size=10),
        marker=dict(line=dict(color=C_BG, width=1)),
    )
    style_plotly_fig(fig)
    fig.update_layout(height=380, showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def render_topic_distribution_comparison(
    brand_a: str,
    brand_b: str,
    *,
    articles_a: list[dict] | None = None,
    articles_b: list[dict] | None = None,
    use_sample: bool = False,
) -> None:
    """双品牌主题分布对比：并排饼图 + 分组柱状图。"""
    if use_sample:
        st.caption("样本模式：主题分布基于样本文章的 AI 分类统计。")
        topics_a = articles_to_topic_distribution(articles_a or [])
        topics_b = articles_to_topic_distribution(articles_b or [])
        if not articles_a:
            st.warning(f"样本中无「{brand_a}」的文章。")
        if not articles_b:
            st.warning(f"样本中无「{brand_b}」的文章。")
        if not topics_a and not topics_b:
            st.caption("样本文章暂无分类数据，请先在文章列表中使用「AI 分类」。")
            return
    else:
        cache_a = get_cached_tone_analysis(brand_a)
        cache_b = get_cached_tone_analysis(brand_b)
        topics_a = cache_a["result"]["topic_distribution"] if cache_a else None
        topics_b = cache_b["result"]["topic_distribution"] if cache_b else None

        missing_brands: list[str] = []
        if not cache_a:
            missing_brands.append(brand_a)
        if not cache_b:
            missing_brands.append(brand_b)

        for name in missing_brands:
            st.info(f"请先在品牌对比分析页对「{name}」执行「分析品牌调性」。")

        if not cache_a and not cache_b:
            return

    pie_col1, pie_col2 = st.columns(2)
    with pie_col1:
        if topics_a:
            render_brand_topic_pie(brand_a, topics_a)
        else:
            st.markdown(f"**{brand_a}**")
            st.caption("暂无主题分布数据。")
    with pie_col2:
        if topics_b:
            render_brand_topic_pie(brand_b, topics_b)
        else:
            st.markdown(f"**{brand_b}**")
            st.caption("暂无主题分布数据。")

    if topics_a and topics_b:
        combined_df = build_combined_topics_bar_df(
            brand_a, topics_a, brand_b, topics_b
        )
        if not combined_df.empty:
            st.markdown("**主题占比对比（分组柱状图）**")
            fig_bar = px.bar(
                combined_df,
                x="主题",
                y="占比 (%)",
                color="品牌",
                barmode="group",
                labels={"主题": "主题", "占比 (%)": "占比 (%)", "品牌": "品牌"},
                color_discrete_map={brand_a: C_ACCENT, brand_b: C_SUCCESS},
            )
            style_plotly_fig(fig_bar)
            fig_bar.update_layout(height=420, xaxis_tickangle=-25)
            st.plotly_chart(fig_bar, use_container_width=True)


PICK_CB_PREFIX = "pick_cb_"


def pick_checkbox_key(article_id: str) -> str:
    return f"{PICK_CB_PREFIX}{article_id}"


def init_checked_article_ids() -> None:
    """初始化跨筛选持久化的勾选文章 ID 列表（session_state 存 list，避免 set 序列化问题）。"""
    raw = st.session_state.get("checked_article_ids")
    if raw is None:
        st.session_state.checked_article_ids = []
    elif isinstance(raw, set):
        st.session_state.checked_article_ids = sorted(str(x) for x in raw if x)


def get_checked_article_ids() -> set[str]:
    """返回当前勾选 ID 集合（不修改 session_state 引用类型）。"""
    init_checked_article_ids()
    return {str(x) for x in st.session_state.checked_article_ids if x}


def set_checked_article_ids(ids: Iterable[str]) -> None:
    """写入勾选 ID（去重、统一为 str）。"""
    st.session_state.checked_article_ids = sorted(
        {str(x) for x in ids if x}
    )


def union_checked_article_ids(ids: Iterable[str]) -> None:
    """将 ID 并入勾选集合（union），不覆盖已有勾选。"""
    merged = get_checked_article_ids()
    merged.update(str(x) for x in ids if x)
    set_checked_article_ids(merged)


def subtract_checked_article_ids(ids: Iterable[str]) -> None:
    """从勾选集合移除指定 ID（仅影响传入 ID）。"""
    remove = {str(x) for x in ids if x}
    if not remove:
        return
    remaining = get_checked_article_ids() - remove
    set_checked_article_ids(remaining)


def sync_pick_checkbox_widgets(article_ids: Iterable[str], *, checked: bool) -> None:
    """批量同步可见文章的复选框 widget 状态（在 on_click 回调中调用）。"""
    for article_id in article_ids:
        st.session_state[pick_checkbox_key(str(article_id))] = checked


def _on_pick_checkbox_change(article_id: str) -> None:
    """复选框变更时同步更新 checked_article_ids。"""
    article_id = str(article_id)
    key = pick_checkbox_key(article_id)
    checked = get_checked_article_ids()
    if st.session_state.get(key):
        checked.add(article_id)
    else:
        checked.discard(article_id)
    set_checked_article_ids(checked)


def _on_pick_all_visible() -> None:
    """全选当前可见文章：union 到 checked_article_ids，不覆盖已有勾选。"""
    visible = [
        str(x) for x in (st.session_state.get("pick_toolbar_visible_ids") or []) if x
    ]
    if not visible:
        return
    union_checked_article_ids(visible)
    sync_pick_checkbox_widgets(visible, checked=True)


def _on_pick_clear_visible() -> None:
    """取消全选：仅移除当前可见文章的勾选。"""
    visible = [
        str(x) for x in (st.session_state.get("pick_toolbar_visible_ids") or []) if x
    ]
    if not visible:
        return
    subtract_checked_article_ids(visible)
    sync_pick_checkbox_widgets(visible, checked=False)


def _on_set_compare_sample() -> None:
    """将 checked_article_ids 写入 selected_article_ids（深拷贝列表）。"""
    picked_ids = list(get_checked_article_ids())
    if not picked_ids:
        st.session_state.flash_message = {
            "level": "warning",
            "text": "请先勾选文章。",
        }
        return

    valid_articles = db.get_articles_by_ids(picked_ids)
    if not valid_articles:
        st.session_state.flash_message = {
            "level": "warning",
            "text": "所选文章已不存在，请重新勾选。",
        }
        return

    valid_ids = [str(a["id"]) for a in valid_articles]
    st.session_state.selected_article_ids = deepcopy(valid_ids)

    if len(valid_ids) < len(picked_ids):
        subtract_checked_article_ids(set(picked_ids) - set(valid_ids))
        st.session_state.flash_message = {
            "level": "warning",
            "text": (
                f"部分已选文章已不存在，已设为 {len(valid_ids)} 篇有效样本。"
            ),
        }
    else:
        st.session_state.flash_message = {
            "level": "success",
            "text": f"已设置对比样本：{len(valid_ids)} 篇文章",
        }


def get_selected_sample_ids() -> list[str]:
    return list(deepcopy(st.session_state.get("selected_article_ids") or []))


def articles_to_topic_distribution(articles: list[dict]) -> list[dict]:
    """将样本文章的分类分布转换为主题占比，供样本模式下的主题对比。"""
    df = get_category_distribution(articles)
    if df.empty:
        return []
    total = int(df["count"].sum())
    if total <= 0:
        return []
    return [
        {
            "topic": row["category"],
            "percentage": round(row["count"] / total * 100, 1),
        }
        for _, row in df.iterrows()
    ]


def init_compare_data_mode() -> None:
    if "compare_data_mode" not in st.session_state:
        st.session_state.compare_data_mode = "all"


def _clear_compare_on_mode_change() -> None:
    st.session_state.pop("brand_compare_data", None)
    st.session_state.pop("gap_analysis_result", None)
    st.session_state["compare_reclick_hint"] = True


def _select_compare_data_mode(mode: str) -> None:
    previous = st.session_state.get("compare_data_mode", "all")
    st.session_state.compare_data_mode = mode
    if previous != mode:
        _clear_compare_on_mode_change()


def _select_compare_mode_all() -> None:
    _select_compare_data_mode("all")


def _select_compare_mode_sample() -> None:
    _select_compare_data_mode("sample")


def render_compare_data_mode_buttons(sample_count: int) -> None:
    """对比数据范围：primary / secondary 随 compare_data_mode 切换高亮。"""
    init_compare_data_mode()
    mode = st.session_state.compare_data_mode
    sample_btn_disabled = mode == "sample" and sample_count == 0

    st.markdown("**对比数据范围**")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        st.button(
            "📊 使用全部文章",
            key="compare_mode_all_btn",
            type="primary" if mode == "all" else "secondary",
            use_container_width=True,
            on_click=_select_compare_mode_all,
        )
    with mode_col2:
        sample_label = "📋 使用自定义样本"
        if sample_count > 0:
            sample_label = f"📋 使用自定义样本（{sample_count} 篇）"
        st.button(
            sample_label,
            key="compare_mode_sample_btn",
            type="primary" if mode == "sample" and not sample_btn_disabled else "secondary",
            use_container_width=True,
            disabled=sample_btn_disabled,
            on_click=_select_compare_mode_sample,
        )

    if st.session_state.pop("compare_reclick_hint", False):
        st.info("对比数据范围已切换，请重新点击「开始对比」。")


def resolve_compare_articles(
    brand_a: str, brand_b: str, use_sample: bool
) -> tuple[list[dict], list[dict], list[str]]:
    """解析对比用文章列表；返回 (articles_a, articles_b, warnings)。"""
    warnings: list[str] = []
    if not use_sample:
        return (
            cached_list_articles(brand_a, False, None, None, None),
            cached_list_articles(brand_b, False, None, None, None),
            warnings,
        )

    sample_ids = get_selected_sample_ids()
    if not sample_ids:
        warnings.append("尚未设置对比样本，请先在品牌内容管理页勾选文章并设为对比样本。")
        return [], [], warnings

    sample_articles = db.get_articles_by_ids(sample_ids)
    missing_count = len(sample_ids) - len(sample_articles)
    if missing_count > 0:
        warnings.append(f"样本中有 {missing_count} 篇文章已不存在，已自动忽略。")
        st.session_state["selected_article_ids"] = [a["id"] for a in sample_articles]

    articles_a = [a for a in sample_articles if a["brand_name"] == brand_a]
    articles_b = [a for a in sample_articles if a["brand_name"] == brand_b]
    if not articles_a:
        warnings.append(f"样本中无「{brand_a}」的文章。")
    if not articles_b:
        warnings.append(f"样本中无「{brand_b}」的文章。")
    return articles_a, articles_b, warnings


def mark_pick_checkbox_zone(checked: bool) -> None:
    css_class = "bp-pick-checked" if checked else "bp-pick-unchecked"
    st.markdown(
        f'<div class="bp-style-marker {css_class} bp-pick-zone"></div>',
        unsafe_allow_html=True,
    )


def mark_fetch_all_btn_zone() -> None:
    st.markdown(
        '<div class="bp-style-marker bp-fetch-all-zone"></div>',
        unsafe_allow_html=True,
    )


def fetch_all_brands_content(brands: list[dict]) -> tuple[int, int, list[str]]:
    """
    依次采集全部品牌内容并入库。
    返回 (新增总数, 跳过总数, 失败品牌描述列表)。
    """
    total = len(brands)
    total_new = 0
    total_skipped = 0
    failed_brands: list[str] = []

    progress_bar = st.progress(0, text="准备开始采集…")
    status_text = st.empty()

    for index, brand in enumerate(brands, start=1):
        brand_name = brand["brand_name"]
        status_text.caption(f"正在采集 {index}/{total}：{brand_name}…")
        progress_bar.progress(
            (index - 1) / total,
            text=f"正在采集 {index}/{total}：{brand_name}…",
        )

        try:
            articles, error = fetch_brand_content(
                brand["rss_url"],
                brand_name,
                brand.get("source_type"),
            )
        except Exception as exc:
            failed_brands.append(f"「{brand_name}」（{exc}）")
            continue

        if error:
            failed_brands.append(f"「{brand_name}」（{error}）")
            continue

        new_count, skipped_count = db.save_articles(articles)
        total_new += new_count
        total_skipped += skipped_count
        progress_bar.progress(
            index / total,
            text=f"正在采集 {index}/{total}：{brand_name}…",
        )

    progress_bar.progress(1.0, text="采集完成")
    progress_bar.empty()
    status_text.empty()
    return total_new, total_skipped, failed_brands


def _render_article_sample_toolbar_body() -> None:
    """样本工具栏 UI（由 fragment 包裹，减轻全页重跑）。"""
    init_checked_article_ids()
    picked_count = len(get_checked_article_ids())
    sample_count = len(get_selected_sample_ids())
    visible_count = len(st.session_state.get("pick_toolbar_visible_ids") or [])

    col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment="center")
    with col1:
        st.button(
            "☑ 全选",
            key="pick_all_visible",
            use_container_width=True,
            disabled=visible_count == 0,
            on_click=_on_pick_all_visible,
            help="全选当前筛选下列表中的所有文章（累加至已勾选）",
        )
    with col2:
        st.button(
            "✕ 取消",
            key="pick_clear_visible",
            use_container_width=True,
            disabled=visible_count == 0,
            on_click=_on_pick_clear_visible,
            help="取消当前筛选下列表中的勾选（保留其他筛选下已勾选的文章）",
        )
    with col3:
        st.button(
            "📌 设为样本",
            key="set_compare_sample",
            use_container_width=True,
            disabled=picked_count == 0,
            on_click=_on_set_compare_sample,
            help="将当前所有勾选的文章设为品牌对比页的自定义样本",
        )

    status_hint = f"已选择 {picked_count} 篇文章"
    if visible_count:
        status_hint += f"（当前列表 {visible_count} 篇，全选将累加勾选）"
    if sample_count:
        status_hint += f" · 当前对比样本：{sample_count} 篇（可在品牌对比页选用）"
    st.caption(status_hint)


def render_article_sample_toolbar() -> None:
    """文章样本选择工具栏（依赖 session_state.pick_toolbar_visible_ids）。"""
    _render_article_sample_toolbar_body()


def mark_action_btn_zone() -> None:
    st.markdown(
        '<div class="bp-style-marker bp-action-btn-zone"></div>',
        unsafe_allow_html=True,
    )


def mark_favorite_btn_zone(is_favorite: bool) -> None:
    """标记收藏按钮区域，供 CSS 区分已收藏 / 未收藏样式。"""
    css_class = "bp-favorite-active" if is_favorite else "bp-favorite-inactive"
    st.markdown(
        f'<div class="bp-style-marker {css_class}"></div>',
        unsafe_allow_html=True,
    )


def article_is_favorite(article: dict) -> bool:
    return int(article.get("is_favorite") or 0) == 1


ARTICLE_TIME_PRESETS = ("全部", "今日", "3日内", "7日内", "30日内")
ARTICLE_CATEGORY_FILTER_OPTIONS = ("全部主题",) + CATEGORIES + ("未分类",)


def init_article_time_filters() -> None:
    """初始化文章时间筛选 session 状态。"""
    if "article_time_preset" not in st.session_state:
        st.session_state.article_time_preset = "全部"
    if "article_time_last_preset" not in st.session_state:
        st.session_state.article_time_last_preset = "全部"
    if "article_time_use_custom" not in st.session_state:
        st.session_state.article_time_use_custom = False


def _on_time_preset_change() -> None:
    """快捷时间选项变更时退出自定义模式并记录上次快捷选项。"""
    st.session_state.article_time_use_custom = False
    preset = st.session_state.get("article_time_preset", "全部")
    if preset != "全部":
        st.session_state.article_time_last_preset = preset


def _iso_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def published_range_for_preset(preset: str) -> tuple[str | None, str | None, str]:
    """
    根据快捷时间选项计算 published 范围。
    返回 (published_start, published_end, 显示标签)。
    """
    if preset == "全部":
        return None, None, "全部"

    now = datetime.now()
    today_start = datetime.combine(now.date(), time.min)
    preset_days = {"今日": 1, "3日内": 3, "7日内": 7, "30日内": 30}
    days = preset_days.get(preset, 1)
    if preset == "今日":
        start_dt = today_start
    else:
        start_dt = today_start - timedelta(days=days - 1)
    return _iso_datetime(start_dt), _iso_datetime(now), preset


def resolve_article_published_range() -> tuple[str | None, str | None, str]:
    """
    解析当前时间筛选范围。
    返回 (published_start, published_end, 显示标签)；全部为 (None, None, "全部")。
    """
    init_article_time_filters()

    if st.session_state.get("article_time_use_custom"):
        custom_start = st.session_state.get("article_custom_start")
        custom_end = st.session_state.get("article_custom_end")
        if custom_start and custom_end:
            start_dt = datetime.combine(custom_start, time.min)
            end_dt = datetime.combine(custom_end, time(23, 59, 59))
            if start_dt > end_dt:
                start_dt, end_dt = end_dt, start_dt
            label = f"{custom_start} 至 {custom_end}"
            return _iso_datetime(start_dt), _iso_datetime(end_dt), label

    preset = st.session_state.get("article_time_preset", "全部")
    return published_range_for_preset(preset)


def init_activity_extract_time() -> None:
    if "activity_extract_time_preset" not in st.session_state:
        st.session_state.activity_extract_time_preset = "7日内"


def init_article_category_filter() -> None:
    if "article_category_filter" not in st.session_state:
        st.session_state.article_category_filter = "全部主题"


def init_article_brand_filter() -> None:
    if "article_filter" not in st.session_state:
        st.session_state.article_filter = "全部品牌"


def resolve_category_db_param(selected: str) -> str | None:
    """将主题筛选项转为 list_articles 的 category 参数。"""
    if selected == "全部主题":
        return None
    return selected


def get_article_list_query_params() -> dict:
    """读取当前文章筛选条件，供 list_articles 使用。"""
    init_article_brand_filter()
    init_article_category_filter()
    init_article_time_filters()
    selected_brand = st.session_state.get("article_filter", "全部品牌")
    brand_name = None if selected_brand == "全部品牌" else selected_brand
    category = resolve_category_db_param(
        st.session_state.get("article_category_filter", "全部主题")
    )
    published_start, published_end, time_label = resolve_article_published_range()
    return {
        "brand_name": brand_name,
        "category": category,
        "favorites_only": bool(st.session_state.get("favorites_only_filter", False)),
        "published_start": published_start,
        "published_end": published_end,
        "time_label": time_label,
    }


def fetch_filtered_articles() -> tuple[list[dict], dict]:
    """按当前筛选条件查询文章（60 秒缓存）。"""
    params = get_article_list_query_params()
    articles = cached_list_articles(
        params["brand_name"],
        params["favorites_only"],
        params["published_start"],
        params["published_end"],
        params["category"],
    )
    return articles, params


def render_article_time_filters(*, compact: bool = False) -> tuple[str | None, str | None, str]:
    """渲染时间筛选 UI，返回当前生效的 published 范围。"""
    init_article_time_filters()
    use_custom = bool(st.session_state.get("article_time_use_custom"))

    wrap_class = "bp-time-filter-wrap"
    if compact:
        wrap_class += " bp-article-filter-time"
    st.markdown(f'<div class="{wrap_class}">', unsafe_allow_html=True)
    if not compact:
        st.markdown("**发布时间筛选**")
    if use_custom:
        st.caption(
            f"当前为自定义日期范围（快捷选项已暂停）。"
            f"清空自定义后将恢复为「{st.session_state.article_time_last_preset}」。"
        )
        st.radio(
            "发布时间",
            list(ARTICLE_TIME_PRESETS),
            index=0,
            horizontal=True,
            key="article_time_preset_inactive",
            label_visibility="collapsed",
            disabled=True,
        )
    else:
        preset_index = list(ARTICLE_TIME_PRESETS).index(
            st.session_state.get("article_time_preset", "全部")
        )
        st.radio(
            "发布时间",
            list(ARTICLE_TIME_PRESETS),
            index=preset_index,
            horizontal=True,
            key="article_time_preset",
            label_visibility="collapsed",
            on_change=_on_time_preset_change,
        )

    with st.expander("自定义日期", expanded=use_custom):
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            st.date_input(
                "开始日期",
                value=None,
                key="article_custom_start",
                format="YYYY-MM-DD",
            )
        with date_col2:
            st.date_input(
                "结束日期",
                value=None,
                key="article_custom_end",
                format="YYYY-MM-DD",
            )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("应用自定义范围", key="apply_custom_date_range", use_container_width=True):
                custom_start = st.session_state.get("article_custom_start")
                custom_end = st.session_state.get("article_custom_end")
                if not custom_start or not custom_end:
                    st.warning("请选择开始日期和结束日期。")
                else:
                    if not use_custom:
                        st.session_state.article_time_last_preset = st.session_state.get(
                            "article_time_preset", "全部"
                        )
                    st.session_state.article_time_use_custom = True
                    st.rerun()
        with btn_col2:
            if st.button("清空自定义日期", key="clear_custom_date_range", use_container_width=True):
                st.session_state.article_time_use_custom = False
                st.session_state.article_time_preset = (
                    st.session_state.article_time_last_preset
                )
                for key in ("article_custom_start", "article_custom_end"):
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    return resolve_article_published_range()


def render_unified_article_filters(brand_names: list[str]) -> None:
    """文章管理 Tab 统一筛选栏：品牌、主题、时间、收藏。"""
    init_article_brand_filter()
    init_article_category_filter()
    init_article_time_filters()
    st.markdown('<div class="bp-article-filter-wrap">', unsafe_allow_html=True)
    st.caption("筛选条件")
    filter_row1_col1, filter_row1_col2, filter_row1_col3 = st.columns(
        [1.1, 1.1, 1.8], vertical_alignment="top"
    )
    with filter_row1_col1:
        st.selectbox(
            "品牌",
            brand_names,
            key="article_filter",
        )
    with filter_row1_col2:
        st.selectbox(
            "主题",
            list(ARTICLE_CATEGORY_FILTER_OPTIONS),
            key="article_category_filter",
        )
    with filter_row1_col3:
        render_article_time_filters(compact=True)
    st.checkbox("仅显示收藏", key="favorites_only_filter")
    st.markdown("</div>", unsafe_allow_html=True)


def render_article_list_empty_hint(params: dict) -> None:
    """无文章时的提示文案。"""
    brand_name = params["brand_name"]
    favorites_only = params["favorites_only"]
    published_start = params["published_start"]
    published_end = params["published_end"]
    time_label = params["time_label"]
    category = params["category"]
    fav_total = cached_count_favorite_articles(brand_name)

    brand_hint = f"「{brand_name}」" if brand_name else "当前筛选范围"
    hints: list[str] = []
    if category:
        hints.append(f"主题「{category}」")
    if time_label != "全部":
        hints.append(f"时间「{time_label}」")
    if favorites_only:
        hints.append("仅收藏")
    scope = brand_hint + (" · " + " · ".join(hints) if hints else "")

    if favorites_only:
        st.info(
            f"{scope}暂无符合条件的收藏文章。"
            f"（该品牌下共 {fav_total} 篇收藏；可调整筛选或点击「☆ 收藏」）"
        )
    elif (published_start and published_end) or category:
        st.info(f"{scope}暂无符合条件的文章，请调整筛选条件或先采集内容。")
    else:
        st.info("暂无文章数据。请先添加品牌，再点击「立即采集」抓取内容。")


def render_article_cards(articles: list[dict], *, params: dict) -> None:
    """渲染文章卡片列表（不含筛选栏与样本工具栏）。"""
    init_checked_article_ids()
    checked = get_checked_article_ids()
    brand_name = params["brand_name"]
    favorites_only = params["favorites_only"]
    time_label = params["time_label"]
    fav_total = cached_count_favorite_articles(brand_name)

    time_hint = f" · 时间范围：{time_label}" if time_label != "全部" else ""
    category = params.get("category")
    category_hint = f" · 主题：{category}" if category else ""
    if favorites_only:
        st.caption(f"共 {len(articles)} 篇收藏文章{category_hint}{time_hint}")
    else:
        st.caption(
            f"共 {len(articles)} 篇文章（已收藏 {fav_total} 篇）{category_hint}{time_hint}"
        )

    for article in articles:
        is_favorite = article_is_favorite(article)
        article_id = str(article["id"])
        is_picked = article_id in checked
        pick_key = pick_checkbox_key(article_id)
        if pick_key not in st.session_state:
            st.session_state[pick_key] = is_picked
        elif bool(st.session_state[pick_key]) != is_picked:
            st.session_state[pick_key] = is_picked
        with st.container(border=True):
            pick_col, title_col, btn_col = st.columns(
                [0.35, 4.65, 1], vertical_alignment="center"
            )
            with pick_col:
                mark_pick_checkbox_zone(is_picked)
                st.checkbox(
                    "选择",
                    key=pick_key,
                    label_visibility="collapsed",
                    on_change=_on_pick_checkbox_change,
                    args=(article_id,),
                )
            with title_col:
                fav_badge = " ★" if is_favorite else ""
                st.markdown(f"#### {article['title']}{fav_badge}")
                if article.get("category"):
                    render_category_badge(article["category"])
                else:
                    st.caption("未分类")
            with btn_col:
                mark_action_btn_zone()
                classify_btn = st.button(
                    "AI 分类",
                    key=f"classify_{article['id']}",
                    use_container_width=True,
                    disabled=not api_configured(),
                )
                mark_action_btn_zone()
                summary_btn = st.button(
                    "生成摘要",
                    key=f"summary_{article['id']}",
                    use_container_width=True,
                    disabled=not api_configured(),
                )
                if not api_configured():
                    st.caption("请先配置 API Key")

            st.markdown(
                f"**品牌：** {article['brand_name']}  "
                f"**发布时间：** {format_datetime(article['published'])}  "
                f"**来源：** {article['source'] or '未知'}"
            )
            if article.get("summary"):
                st.markdown("**原文摘要**")
                st.write(article["summary"])
            else:
                st.caption("（暂无 RSS 原文摘要）")
            if article.get("ai_summary"):
                render_ai_summary_box(article["ai_summary"])
            st.link_button("阅读原文", article["link"], use_container_width=False)

            fav_col1, _fav_col2 = st.columns([1, 4])
            with fav_col1:
                mark_favorite_btn_zone(is_favorite)
                fav_label = "★ 取消收藏" if is_favorite else "☆ 收藏"
                if st.button(
                    fav_label,
                    key=f"favorite_{article['id']}",
                    use_container_width=True,
                ):
                    new_state = db.toggle_article_favorite(article["id"])
                    if new_state is None:
                        st.warning("收藏操作失败：文章不存在。")
                    else:
                        invalidate_data_cache()
                        st.rerun()

            if classify_btn and api_configured():
                with st.spinner("正在调用 AI 分类…"):
                    classify_single_article(article)
                st.rerun()
            if summary_btn and api_configured():
                with st.spinner("正在生成 AI 摘要…"):
                    summarize_single_article(article)
                st.rerun()


def _render_article_list_body(articles: list[dict], params: dict) -> None:
    """样本工具栏 + 文章列表（与筛选联动，可在 fragment 内局部重跑）。"""
    show_flash_message()
    render_article_sample_toolbar()
    if not articles:
        render_article_list_empty_hint(params)
    else:
        render_article_cards(articles, params=params)


def render_article_list_section(brand_names: list[str]) -> None:
    """文章列表区：筛选栏 → 样本工具栏 → 列表。"""
    render_unified_article_filters(brand_names)
    articles, params = fetch_filtered_articles()
    st.session_state.pick_toolbar_visible_ids = [
        str(article["id"]) for article in articles if article.get("id")
    ]

    fragment = getattr(st, "fragment", None)
    if fragment is not None:

        @fragment
        def _article_list_fragment() -> None:
            _render_article_list_body(articles, params)

        _article_list_fragment()
    else:
        _render_article_list_body(articles, params)


def api_configured() -> bool:
    return bool(DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL)


def format_datetime(dt_str: str | None) -> str:
    if not dt_str:
        return "未知"
    try:
        return datetime.fromisoformat(dt_str).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return dt_str


def article_content_for_ai(article: dict) -> str:
    return (article.get("ai_summary") or article.get("summary") or "").strip()


def render_category_badge(category: str | None) -> None:
    if not category:
        return
    style = CATEGORY_STYLES.get(category, CATEGORY_STYLES["其他"])
    bg = style["bg"]
    text = style["text"]
    st.markdown(
        f'<span style="background:{bg};color:{text};'
        f'padding:4px 10px;border-radius:4px;'
        f'font-size:0.85em;font-weight:600;margin-right:8px;">'
        f'{html.escape(category)}</span>',
        unsafe_allow_html=True,
    )


def show_flash_message() -> None:
    msg = st.session_state.pop("flash_message", None)
    if not msg:
        return
    level = msg.get("level", "info")
    text = msg.get("text", "")
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


def set_flash_message(level: str, text: str) -> None:
    st.session_state.flash_message = {"level": level, "text": text}


def render_ai_summary_box(ai_summary: str) -> None:
    safe_text = html.escape(ai_summary)
    st.markdown(
        f'<div style="background:rgba(19,26,43,0.8);border:1px solid {C_BORDER};'
        f'border-left:3px solid {C_ACCENT};padding:12px 16px;border-radius:8px;'
        f'margin-top:8px;margin-bottom:8px;color:{C_TEXT_SEC};">'
        f'<strong style="color:{C_ACCENT};">AI 摘要：</strong>{safe_text}</div>',
        unsafe_allow_html=True,
    )


def classify_single_article(article: dict) -> bool:
    existing = str(article.get("category") or "").strip()
    if existing:
        set_flash_message("success", f"文章已有分类「{existing}」：{article['title'][:40]}")
        return True
    category, error = classify_article(
        article["title"], article_content_for_ai(article)
    )
    if error:
        set_flash_message("warning", f"「{article['title'][:30]}…」分类失败：{error}")
        return False
    db.update_article_category(article["id"], category)
    invalidate_data_cache()
    set_flash_message("success", f"已分类为「{category}」：{article['title'][:40]}")
    return True


def classify_articles_batch(articles: list[dict]) -> tuple[int, int]:
    total = len(articles)
    if total == 0:
        return 0, 0

    progress_bar = st.progress(0, text="准备开始分类…")
    status_text = st.empty()
    success_count = 0
    failed_titles: list[str] = []

    for index, article in enumerate(articles, start=1):
        status_text.caption(f"正在分类 {index}/{total}…")
        progress_bar.progress(index / total, text=f"正在分类 {index}/{total}…")
        if str(article.get("category") or "").strip():
            success_count += 1
            continue
        category, error = classify_article(
            article["title"], article_content_for_ai(article)
        )
        if error:
            failed_titles.append(article["title"])
            continue
        db.update_article_category(article["id"], category)
        success_count += 1

    progress_bar.empty()
    status_text.empty()

    if success_count:
        invalidate_data_cache()

    if failed_titles:
        preview = "、".join(failed_titles[:3])
        if len(failed_titles) > 3:
            preview += f" 等 {len(failed_titles)} 篇"
        set_flash_message(
            "warning",
            f"批量分类完成：成功 {success_count} 篇，失败 {len(failed_titles)} 篇（{preview}）。",
        )
    else:
        set_flash_message("success", f"批量分类完成：共成功分类 {success_count} 篇文章。")
    return success_count, len(failed_titles)


def summarize_single_article(article: dict) -> bool:
    existing = str(article.get("ai_summary") or "").strip()
    if existing:
        set_flash_message("success", f"文章已有 AI 摘要：{article['title'][:40]}")
        return True
    ai_summary, error = generate_summary(
        article["title"], article.get("summary") or ""
    )
    if error:
        set_flash_message("warning", f"「{article['title'][:30]}…」摘要生成失败：{error}")
        return False
    db.update_article_ai_summary(article["id"], ai_summary)
    invalidate_data_cache()
    set_flash_message("success", f"已生成 AI 摘要：{article['title'][:40]}")
    return True


def summarize_articles_batch(articles: list[dict]) -> tuple[int, int]:
    total = len(articles)
    if total == 0:
        return 0, 0

    progress_bar = st.progress(0, text="准备生成摘要…")
    status_text = st.empty()
    success_count = 0
    failed_titles: list[str] = []

    for index, article in enumerate(articles, start=1):
        status_text.caption(f"正在生成摘要 {index}/{total}…")
        progress_bar.progress(index / total, text=f"正在生成摘要 {index}/{total}…")
        if str(article.get("ai_summary") or "").strip():
            success_count += 1
            continue
        ai_summary, error = generate_summary(
            article["title"], article.get("summary") or ""
        )
        if error:
            failed_titles.append(article["title"])
            continue
        db.update_article_ai_summary(article["id"], ai_summary)
        success_count += 1

    progress_bar.empty()
    status_text.empty()

    if success_count:
        invalidate_data_cache()

    if failed_titles:
        preview = "、".join(failed_titles[:3])
        if len(failed_titles) > 3:
            preview += f" 等 {len(failed_titles)} 篇"
        set_flash_message(
            "warning",
            f"批量摘要完成：成功 {success_count} 篇，失败 {len(failed_titles)} 篇（{preview}）。",
        )
    else:
        set_flash_message("success", f"批量摘要完成：共成功生成 {success_count} 篇 AI 摘要。")
    return success_count, len(failed_titles)


def render_sidebar_nav() -> str:
    """侧边栏页面导航，宽度与系统状态提示一致。"""
    pages = ["品牌内容管理", "品牌对比分析", "内容策略生成"]
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = pages[0]
    if st.session_state.nav_page not in pages:
        st.session_state.nav_page = pages[0]

    st.sidebar.markdown("### 🌐 功能导航")
    for page_name in pages:
        is_active = st.session_state.nav_page == page_name
        if st.sidebar.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.nav_page = page_name
            st.rerun()
    return st.session_state.nav_page


def render_api_status() -> None:
    st.sidebar.divider()
    st.sidebar.subheader("系统状态")
    if DEEPSEEK_API_KEY:
        st.sidebar.success("DeepSeek API 已配置")
    else:
        st.sidebar.warning("未检测到 DEEPSEEK_API_KEY")
    if not DEEPSEEK_BASE_URL:
        st.sidebar.warning("未检测到 DEEPSEEK_BASE_URL")


def render_brand_management() -> None:
    st.header("品牌内容管理")
    st.caption("添加竞争品牌 RSS 源或网页列表地址，采集、分类并浏览已抓取的文章内容。")
    show_flash_message()

    with st.expander("添加品牌源", expanded=True):
        with st.form("add_brand_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                brand_name = st.text_input("品牌名称", placeholder="例如：竞品 Alpha")
            with col2:
                source_url = st.text_input(
                    "RSS 地址 / 网页地址",
                    placeholder="https://example.com/feed.xml 或 https://example.com/blog",
                )
            submitted = st.form_submit_button("添加品牌", type="primary", use_container_width=True)
            if submitted:
                if not brand_name.strip():
                    st.error("请输入品牌名称。")
                elif not source_url.strip():
                    st.error("请输入 RSS 地址或网页地址。")
                elif db.brand_name_exists(brand_name):
                    st.error("该品牌名称已存在，请使用其他名称。")
                else:
                    db.add_brand(brand_name, source_url)
                    invalidate_data_cache()
                    detected = detect_source_type(source_url.strip())
                    mode_label = "RSS" if detected == "rss" else "网页"
                    set_flash_message(
                        "success",
                        f"已成功添加品牌「{brand_name.strip()}」（采集方式：{mode_label}）。",
                    )
                    st.rerun()

    st.subheader("已添加品牌")
    brands = get_brands()
    article_counts = get_article_counts_by_brand()
    batch_fetch_running = bool(st.session_state.get("batch_fetch_all_running", False))
    init_activity_extract_time()

    fetch_col, _spacer_col = st.columns([1, 2], vertical_alignment="center")
    with fetch_col:
        mark_fetch_all_btn_zone()
        fetch_all_clicked = st.button(
            "🔄 一键更新全部品牌",
            key="fetch_all_brands",
            type="primary",
            use_container_width=True,
            disabled=not brands or batch_fetch_running,
        )

    if not brands:
        st.caption("暂无品牌，请先添加。")
    elif batch_fetch_running:
        st.caption("正在批量采集中，请稍候…")

    if fetch_all_clicked and brands:
        st.session_state["batch_fetch_all_running"] = True
        total_new, total_skipped, failed_brands = fetch_all_brands_content(brands)
        st.session_state["batch_fetch_all_running"] = False

        summary = f"全部完成：共新增 {total_new} 篇，跳过 {total_skipped} 篇重复"
        if failed_brands:
            preview = "、".join(failed_brands[:3])
            if len(failed_brands) > 3:
                preview += f" 等 {len(failed_brands)} 个品牌"
            set_flash_message(
                "warning",
                f"{summary}。部分品牌采集失败：{preview}",
            )
        else:
            set_flash_message("success", f"{summary}。")
        invalidate_data_cache()
        st.rerun()

    if brands:
        for brand in brands:
            with st.container(border=True):
                col_info, col_fetch, col_del = st.columns(
                    [3, 1, 1], vertical_alignment="center"
                )
                with col_info:
                    article_count = article_counts.get(brand["brand_name"], 0)
                    source_type = brand.get("source_type") or detect_source_type(
                        brand["rss_url"]
                    )
                    mode_label = "RSS" if source_type == "rss" else "网页"
                    st.markdown(
                        f"**{brand['brand_name']}**  \n"
                        f"地址：`{brand['rss_url']}`  \n"
                        f"采集方式：{mode_label}　|　已采集文章：{article_count} 篇"
                    )
                with col_fetch:
                    if st.button(
                        "立即采集",
                        key=f"fetch_{brand['id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner(f"正在采集「{brand['brand_name']}」的内容…"):
                            articles, error = fetch_brand_content(
                                brand["rss_url"],
                                brand["brand_name"],
                                brand.get("source_type"),
                            )
                        if error:
                            st.warning(error)
                        else:
                            new_count, skipped_count = db.save_articles(articles)
                            invalidate_data_cache()
                            set_flash_message(
                                "success",
                                f"「{brand['brand_name']}」采集完成：新增 {new_count} 篇，"
                                f"跳过 {skipped_count} 篇重复文章。",
                            )
                            st.rerun()
                with col_del:
                    if st.button(
                        "删除",
                        key=f"del_{brand['id']}",
                        use_container_width=True,
                    ):
                        deleted_name = db.delete_brand(brand["id"])
                        if deleted_name:
                            invalidate_data_cache()
                            set_flash_message(
                                "success",
                                f"已删除品牌「{deleted_name}」及其全部文章。",
                            )
                        st.rerun()
    else:
        st.info("暂无品牌源，请在上方表单中添加。")

    st.divider()
    tab_articles, tab_activities = st.tabs(["📰 文章管理", "📢 竞品活动"])

    with tab_articles:
        st.subheader("文章列表")
        brand_names = get_brand_names()
        init_article_brand_filter()
        init_article_category_filter()

        filter_brand = get_article_list_query_params()["brand_name"]

        schema_status = st.session_state.get("db_schema_status", {})
        if schema_status.get("has_is_favorite_column"):
            fav_in_scope = cached_count_favorite_articles(filter_brand)
            st.caption(f"收藏字段已就绪 · 当前品牌范围内 {fav_in_scope} 篇收藏")
        else:
            st.error("数据库缺少 is_favorite 字段，请重启应用以完成迁移。")

        uncategorized_count = cached_count_uncategorized_articles(filter_brand)
        without_ai_summary_count = cached_count_articles_without_ai_summary(filter_brand)

        tool_col1, tool_col2, tool_col3 = st.columns([1, 1, 2])
        with tool_col1:
            batch_classify_btn = st.button(
                "一键全部分类",
                key="batch_classify_btn",
                type="primary",
                use_container_width=True,
                disabled=not api_configured(),
            )
        with tool_col2:
            batch_summary_btn = st.button(
                "一键生成所有摘要",
                key="batch_summary_btn",
                type="primary",
                use_container_width=True,
                disabled=not api_configured(),
            )
        with tool_col3:
            if not api_configured():
                st.caption("请先配置 API Key（在 `.env` 中设置 `DEEPSEEK_API_KEY` 与 `DEEPSEEK_BASE_URL`）。")
            else:
                st.caption(
                    f"待分类：{uncategorized_count} 篇　|　待生成 AI 摘要：{without_ai_summary_count} 篇"
                )

        if batch_classify_btn and api_configured():
            if uncategorized_count <= 0:
                st.warning("当前没有未分类的文章。")
            else:
                classify_articles_batch(db.list_uncategorized_articles(filter_brand))
                st.rerun()

        if batch_summary_btn and api_configured():
            if without_ai_summary_count <= 0:
                st.warning("当前没有待生成 AI 摘要的文章。")
            else:
                summarize_articles_batch(
                    db.list_articles_without_ai_summary(filter_brand)
                )
                st.rerun()

        render_article_list_section(brand_names)

    with tab_activities:
        st.subheader("竞品活动简报")
        act_col1, act_col2 = st.columns([2, 1], vertical_alignment="center")
        with act_col1:
            st.selectbox(
                "活动提取时间范围",
                list(ARTICLE_TIME_PRESETS),
                key="activity_extract_time_preset",
            )
        with act_col2:
            render_html_action_button(
                "🔍 提取竞品活动信息",
                html_id="bp-extract-activities-html-btn",
                disabled=not api_configured(),
            )
        extract_activities_clicked = render_hidden_streamlit_button(
            "提取竞品活动信息",
            streamlit_key="extract_competitor_activities_btn",
            disabled=not api_configured(),
        )
        wire_html_button_to_streamlit(
            "bp-extract-activities-html-btn",
            "extract_competitor_activities_btn",
        )
        wire_select_html_button_row(
            "activity_extract_time_preset",
            "bp-extract-activities-html-btn",
        )

        if not api_configured():
            st.caption("请先在 `.env` 中配置 DeepSeek API Key 后再提取竞品活动信息。")

        if extract_activities_clicked and api_configured():
            preset = st.session_state.get("activity_extract_time_preset", "7日内")
            pub_start, pub_end, range_label = published_range_for_preset(preset)
            scoped_articles = cached_list_articles(
                None,
                False,
                pub_start,
                pub_end,
                None,
            )
            if not scoped_articles:
                st.warning(f"在「{range_label}」时间范围内暂无已采集文章。")
                st.session_state.competitor_activities_extracted = False
            else:
                with st.spinner(
                    f"正在从「{range_label}」内的 {len(scoped_articles)} 篇文章中提取竞品活动信息…"
                ):
                    activities, act_error = extract_competitor_activities(scoped_articles)
                if act_error:
                    st.warning(act_error)
                st.session_state.competitor_activities = activities
                st.session_state.competitor_activities_range_label = range_label
                st.session_state.competitor_activities_extracted = True

        if st.session_state.get("competitor_activities_extracted"):
            range_label = st.session_state.get(
                "competitor_activities_range_label", "全部"
            )
            st.caption(f"数据范围：{range_label} · 按活动时间倒序展示")
            render_competitor_activities_table(
                st.session_state.get("competitor_activities") or []
            )


def render_tone_analysis_cards(brand_name: str, result: dict, article_count: int) -> None:
    st.markdown(f"### 「{brand_name}」调性分析报告")
    st.caption(f"基于 {article_count} 篇文章的 AI 分析结果")

    card1, card2 = st.columns(2)
    with card1:
        with st.container(border=True):
            st.markdown("#### 🎯 语气风格")
            st.write(result["tone_style"])
    with card2:
        with st.container(border=True):
            st.markdown("#### 📊 分析概览")
            st.metric("分析文章数", article_count)
            st.metric("主题数量", len(result["topic_distribution"]))

    with st.container(border=True):
        st.markdown("#### 🔑 高频关键词 Top 10")
        keywords_df = pd.DataFrame(result["top_keywords"])
        fig_keywords = px.bar(
            keywords_df,
            x="weight",
            y="keyword",
            orientation="h",
            labels={"weight": "热度", "keyword": "关键词"},
            color="weight",
            color_continuous_scale=[[0, "#131A2B"], [0.5, "#00D4FF"], [1, "#00FFAA"]],
        )
        fig_keywords.update_layout(showlegend=False, height=420)
        style_plotly_fig(fig_keywords)
        fig_keywords.update_coloraxes(showscale=False)
        st.plotly_chart(fig_keywords, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 📂 主题分布")
        topics_df = build_topics_dataframe(result["topic_distribution"])
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            if topics_df.empty:
                st.caption("暂无主题分布数据，无法绘制图表。")
            else:
                plot_df = topics_df.rename(
                    columns={"主题": "topic", "占比 (%)": "percentage"}
                )
                fig_topics = px.pie(
                    plot_df,
                    names="topic",
                    values="percentage",
                    hole=0.35,
                    labels={"topic": "主题", "percentage": "占比 (%)"},
                    color_discrete_sequence=PLOTLY_COLORWAY,
                )
                fig_topics.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    textfont=dict(color=C_TEXT, size=11),
                    marker=dict(line=dict(color=C_BG, width=1)),
                )
                style_plotly_fig(fig_topics)
                fig_topics.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig_topics, use_container_width=True)
        with table_col:
            render_topics_table(topics_df)


def render_brand_comparison() -> None:
    st.header("品牌对比分析")
    st.caption("横向对比竞争品牌，并分析单一品牌的内容调性。")

    brands = get_brands()
    if not brands:
        st.info("请先在「品牌内容管理」中添加品牌并采集文章。")
        return

    brand_options = [b["brand_name"] for b in brands]

    st.subheader("品牌调性分析")
    tone_col1, tone_col2 = st.columns([3, 1], vertical_alignment="center")
    with tone_col1:
        tone_brand = st.selectbox("选择要分析的品牌", brand_options, key="tone_brand_select")
    with tone_col2:
        render_html_action_button(
            "分析品牌调性",
            html_id="bp-analyze-tone-html-btn",
            disabled=not api_configured(),
        )
    analyze_tone_btn = render_hidden_streamlit_button(
        "分析品牌调性",
        streamlit_key="analyze_tone_btn",
        disabled=not api_configured(),
    )
    wire_html_button_to_streamlit("bp-analyze-tone-html-btn", "analyze_tone_btn")
    wire_select_html_button_row("tone_brand_select", "bp-analyze-tone-html-btn")

    if not api_configured():
        st.warning("请先在 `.env` 中配置 `DEEPSEEK_API_KEY` 与 `DEEPSEEK_BASE_URL`。")

    tone_articles = cached_list_articles(tone_brand, False, None, None, None)
    if not tone_articles:
        st.info(f"「{tone_brand}」暂无文章，请先在品牌内容管理中采集 RSS。")
    elif analyze_tone_btn and api_configured():
        with st.spinner(f"正在分析「{tone_brand}」的品牌调性，请稍候…"):
            result, error = analyze_brand_tone(tone_brand, tone_articles)
        if error:
            st.warning(error)
        else:
            save_tone_analysis_cache(tone_brand, result, len(tone_articles))
            render_tone_analysis_cards(tone_brand, result, len(tone_articles))
    else:
        cached = get_cached_tone_analysis(tone_brand)
        if cached:
            render_tone_analysis_cards(
                tone_brand, cached["result"], cached["article_count"]
            )

    st.divider()

    if len(brands) < 2:
        st.info("添加至少两个品牌后，可使用下方的双品牌对比分析。")
        return

    st.subheader("双品牌对比")
    st.markdown(
        '<div class="bp-style-marker bp-form-action-anchor"></div>',
        unsafe_allow_html=True,
    )
    cmp_col1, cmp_col2 = st.columns(2, vertical_alignment="center")
    with cmp_col1:
        brand_a = st.selectbox("品牌 A", brand_options, index=0, key="compare_a")
    with cmp_col2:
        default_b = 1 if len(brand_options) > 1 else 0
        brand_b = st.selectbox("品牌 B", brand_options, index=default_b, key="compare_b")

    if brand_a == brand_b:
        st.warning("请选择两个不同的品牌进行对比。")
        return

    sample_ids = get_selected_sample_ids()
    sample_count = len(sample_ids)

    render_compare_data_mode_buttons(sample_count)
    use_sample = st.session_state.compare_data_mode == "sample"
    sample_empty = use_sample and not sample_ids

    if sample_empty:
        st.markdown(
            f'<p style="color:{C_DANGER}; font-size:0.95rem; margin:0.25rem 0 0.75rem 0;">'
            "请先在品牌内容管理页设置对比样本"
            "</p>",
            unsafe_allow_html=True,
        )

    start_compare_btn = st.button(
        "开始对比",
        key="start_compare_btn",
        type="primary",
        use_container_width=True,
        disabled=sample_empty,
    )

    articles_a, articles_b, resolve_warnings = resolve_compare_articles(
        brand_a, brand_b, use_sample
    )
    for warning in resolve_warnings:
        st.warning(warning)

    if start_compare_btn:
        if not use_sample and (not articles_a or not articles_b):
            st.warning("两个品牌均需有文章数据才能对比，请先采集 RSS。")
        elif use_sample and not articles_a and not articles_b:
            st.warning("当前样本范围内没有可用于对比的文章。")
        else:
            spinner_msg = (
                f"正在对比「{brand_a}」与「{brand_b}」（部分品牌无样本文章）…"
                if use_sample and (not articles_a or not articles_b)
                else f"正在对比「{brand_a}」与「{brand_b}」…"
            )
            with st.spinner(spinner_msg):
                st.session_state.brand_compare_data = build_comparison_data(
                    brand_a,
                    brand_b,
                    articles_a,
                    articles_b,
                    article_ids=sample_ids if use_sample else None,
                )

    compare_data = st.session_state.get("brand_compare_data")
    mode_matches = (
        bool(compare_data.get("use_sample"))
        if use_sample
        else not compare_data.get("use_sample")
    ) if compare_data else False
    if (
        compare_data
        and compare_data.get("brand_a") == brand_a
        and compare_data.get("brand_b") == brand_b
        and mode_matches
    ):
        data_use_sample = bool(compare_data.get("use_sample"))
        render_dual_brand_comparison_results(
            compare_data,
            articles_a=articles_a if data_use_sample else None,
            articles_b=articles_b if data_use_sample else None,
            use_sample=data_use_sample,
        )
    elif not start_compare_btn:
        st.caption("选择品牌 A、品牌 B 后，点击「开始对比」查看分析图表。")


def render_dual_brand_comparison_results(
    data: dict,
    *,
    articles_a: list[dict] | None = None,
    articles_b: list[dict] | None = None,
    use_sample: bool = False,
) -> None:
    brand_a = data["brand_a"]
    brand_b = data["brand_b"]

    if use_sample:
        st.caption(
            f"当前使用自定义样本（共 {data.get('sample_count') or 0} 篇）进行统计与图表展示。"
        )
        if data["articles_a_count"] == 0:
            st.warning(f"样本中无「{brand_a}」的文章。")
        if data["articles_b_count"] == 0:
            st.warning(f"样本中无「{brand_b}」的文章。")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(f"{brand_a} 文章数", data["articles_a_count"])
    with metric_col2:
        st.metric(f"{brand_b} 文章数", data["articles_b_count"])
    with metric_col3:
        diff = data["articles_a_count"] - data["articles_b_count"]
        st.metric(
            "数量差值",
            diff,
            delta=f"{brand_a} 领先" if diff > 0 else f"{brand_b} 领先" if diff < 0 else "持平",
        )

    with st.container(border=True):
        st.markdown("#### 📊 内容类型分布")
        df_a = data["category_a"].rename(columns={"category": "内容类型", "count": "数量"})
        df_b = data["category_b"].rename(columns={"category": "内容类型", "count": "数量"})
        df_a["品牌"] = brand_a
        df_b["品牌"] = brand_b
        category_df = pd.concat([df_a, df_b], ignore_index=True)
        if category_df.empty:
            st.caption("暂无分类数据，请先在文章列表中使用「AI 分类」。")
        else:
            fig_cat = px.bar(
                category_df,
                x="内容类型",
                y="数量",
                color="品牌",
                barmode="group",
                labels={"内容类型": "内容类型", "数量": "文章数", "品牌": "品牌"},
                color_discrete_sequence=[C_ACCENT, C_SUCCESS],
            )
            style_plotly_fig(fig_cat)
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 📈 近 3 个月发布频率")
        monthly_a = data["monthly_a"].rename(columns={"月份": "月份", "发布量": "发布量"})
        monthly_b = data["monthly_b"].rename(columns={"月份": "月份", "发布量": "发布量"})
        monthly_a["品牌"] = brand_a
        monthly_b["品牌"] = brand_b
        monthly_df = pd.concat([monthly_a, monthly_b], ignore_index=True)
        fig_line = px.line(
            monthly_df,
            x="月份",
            y="发布量",
            color="品牌",
            markers=True,
            labels={"月份": "月份", "发布量": "发布篇数", "品牌": "品牌"},
            color_discrete_sequence=[C_ACCENT, C_SUCCESS],
        )
        style_plotly_fig(fig_line)
        fig_line.update_layout(height=400)
        fig_line.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_line, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 📂 主题分布对比")
        if use_sample:
            st.caption("基于样本文章的 AI 分类统计，对比内容主题结构差异。")
        else:
            st.caption("基于各品牌的「分析品牌调性」结果，对比内容主题结构差异。")
        render_topic_distribution_comparison(
            brand_a,
            brand_b,
            articles_a=articles_a,
            articles_b=articles_b,
            use_sample=use_sample,
        )

    kw = data["keywords"]
    with st.container(border=True):
        st.markdown("#### 🔤 关键词交集与差异")
        with st.container(border=True):
            st.markdown("**共同关键词（交集）**")
            if kw["intersection"]:
                chips = " · ".join(kw["intersection"][:25])
                st.write(chips)
                if len(kw["intersection"]) > 25:
                    st.caption(f"共 {len(kw['intersection'])} 个，仅展示前 25 个。")
            else:
                st.caption("未检测到明显共同关键词。")

        img_a, img_b = render_keyword_wordclouds(data)
        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            st.markdown(f"**{brand_a} 独有高频词**")
            if img_a:
                st.image(img_a, use_container_width=True)
            else:
                st.caption("暂无足够文本生成词云。")
        with wc_col2:
            st.markdown(f"**{brand_b} 独有高频词**")
            if img_b:
                st.image(img_b, use_container_width=True)
            else:
                st.caption("暂无足够文本生成词云。")

    st.divider()
    st.markdown("#### 📋 差距分析")
    gap_col1, gap_col2 = st.columns([1, 3])
    with gap_col1:
        gap_btn = st.button(
            "生成差距分析",
            key=f"gap_analysis_{brand_a}_{brand_b}",
            type="primary",
            use_container_width=True,
            disabled=not api_configured(),
        )
    with gap_col2:
        if not api_configured():
            st.caption("请先在 `.env` 中配置 DeepSeek API Key 后使用差距分析。")
        else:
            st.caption(
                f"基于当前对比数据，分析「{brand_b}」相对「{brand_a}」的内容策略差距。"
            )

    gap_cache_key = f"{brand_a}__{brand_b}__{'sample' if use_sample else 'all'}"
    if gap_btn and api_configured():
        gap_articles_a, gap_articles_b, gap_warnings = resolve_compare_articles(
            brand_a, brand_b, use_sample
        )
        for warning in gap_warnings:
            st.warning(warning)
        if not gap_articles_a or not gap_articles_b:
            st.warning("两个品牌均需有文章数据才能生成差距分析。")
        else:
            with st.spinner(f"正在生成「{brand_a}」与「{brand_b}」的差距分析…"):
                gap_result, gap_error = generate_gap_analysis(
                    brand_a, brand_b, data, gap_articles_a, gap_articles_b
                )
            if gap_error:
                st.warning(gap_error)
            else:
                st.session_state.gap_analysis_result = {
                    "key": gap_cache_key,
                    "result": gap_result,
                    "use_sample": use_sample,
                }
                render_gap_analysis_cards(gap_result)
    else:
        cached_gap = st.session_state.get("gap_analysis_result")
        if cached_gap and cached_gap.get("key") == gap_cache_key:
            render_gap_analysis_cards(cached_gap["result"])


def render_competitor_activities_table(activities: list[dict]) -> None:
    if not activities:
        st.info("未在所选时间范围内的文章中发现竞品活动信息。")
        return

    st.success(f"共识别 {len(activities)} 条竞品活动信息")
    rows = sorted(
        list(activities),
        key=lambda x: x.get("_sort_time", ""),
        reverse=True,
    )

    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['brand_name']))}</td>"
        f"<td>{html.escape(str(row['activity_type']))}</td>"
        f"<td>{html.escape(str(row['activity_name']))}</td>"
        f"<td>{html.escape(str(row['time']))}</td>"
        f"<td>{html.escape(str(row['location']))}</td>"
        f'<td><a href="{html.escape(str(row["source_article_link"]))}" '
        f'target="_blank" rel="noopener noreferrer">阅读原文</a></td>'
        "</tr>"
        for row in rows
    )
    st.markdown(
        f"""
        <table class="bp-topics-table bp-activity-table">
            <thead>
                <tr>
                    <th>品牌名称</th>
                    <th>活动类型</th>
                    <th>活动名称</th>
                    <th>时间</th>
                    <th>地点</th>
                    <th>来源文章链接</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_gap_analysis_cards(result: dict) -> None:
    brand_a = result["brand_a"]
    brand_b = result["brand_b"]
    st.success(f"差距分析完成：{brand_b} vs {brand_a}")

    with st.container(border=True):
        st.markdown("##### 内容策略核心差异")
        st.write(result["strategy_differences"])

    adv_col, dis_col = st.columns(2)
    with adv_col:
        with st.container(border=True):
            st.markdown(f"##### ✅ {brand_b} 相对 {brand_a} 的优势")
            for item in result["advantages"]:
                st.markdown(f"- {item}")
    with dis_col:
        with st.container(border=True):
            st.markdown(f"##### ⚠️ {brand_b} 相对 {brand_a} 的劣势")
            for item in result["disadvantages"]:
                st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown(f"##### 💡 针对 {brand_a} 的改进建议")
        for index, item in enumerate(result["suggestions"], start=1):
            st.markdown(f"{index}. {item}")


def render_copy_button(text: str, button_key: str) -> None:
    payload = json.dumps(text, ensure_ascii=False)
    components.html(
        f"""
        <div style="margin-top:8px;">
          <button id="copy-{button_key}"
            style="padding:6px 16px;border-radius:6px;border:1px solid {C_ACCENT};
                   background:rgba(0,212,255,0.15);color:{C_TEXT};cursor:pointer;
                   font-size:14px;font-weight:600;font-family:Inter,Poppins,sans-serif;">
            一键复制
          </button>
          <span id="msg-{button_key}" style="margin-left:10px;color:{C_SUCCESS};font-size:13px;"></span>
        </div>
        <script>
          document.getElementById("copy-{button_key}").onclick = function() {{
            navigator.clipboard.writeText({payload}).then(function() {{
              document.getElementById("msg-{button_key}").innerText = "已复制到剪贴板";
              setTimeout(function() {{
                document.getElementById("msg-{button_key}").innerText = "";
              }}, 2000);
            }});
          }};
        </script>
        """,
        height=55,
    )


def render_strategy_content_package(result: dict) -> None:
    wechat = result["wechat_article"]
    wechat_full = f"【标题】{wechat['title']}\n\n{wechat['content']}"

    with st.expander("📝 公众号推文草稿（约 800 字）", expanded=True):
        st.markdown(f"### {wechat['title']}")
        st.markdown(wechat["content"])
        render_copy_button(wechat_full, "wechat_article")

    with st.expander("📱 社交媒体短文案（3 条）", expanded=False):
        for index, post in enumerate(result["social_posts"], start=1):
            st.markdown(f"**文案 {index}**")
            st.info(post)
            render_copy_button(post, f"social_{index}")

    with st.expander("💬 行业观点短文（3 篇 × 200 字）", expanded=False):
        for index, opinion in enumerate(result["industry_opinions"], start=1):
            st.markdown(f"**{opinion['title']}**")
            st.write(opinion["content"])
            opinion_full = f"{opinion['title']}\n\n{opinion['content']}"
            render_copy_button(opinion_full, f"opinion_{index}")

    with st.expander("📅 下周内容日历（7 天）", expanded=False):
        calendar_lines = []
        for day_item in result["content_calendar"]:
            line = f"{day_item['day']} | {day_item['topic']} | {day_item['brief']}"
            calendar_lines.append(line)
            st.markdown(f"**{day_item['day']}** — {day_item['topic']}")
            st.caption(day_item["brief"])
            st.divider()
        render_copy_button("\n".join(calendar_lines), "calendar")


def render_strategy_generation() -> None:
    st.header("内容策略生成")
    st.caption("基于竞争差距分析，一键生成公众号推文、社媒文案与内容日历。")

    if not api_configured():
        st.warning("请先在 `.env` 文件中配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL` 后重启应用。")
        return

    gap_cached = st.session_state.get("gap_analysis_result")
    gap_result = gap_cached.get("result") if gap_cached else None

    if gap_result:
        brand_a = gap_result["brand_a"]
        brand_b = gap_result["brand_b"]
        with st.container(border=True):
            st.markdown("**已加载差距分析结果**")
            st.caption(f"参照品牌（你的品牌）：{brand_a}　|　对比品牌（竞品）：{brand_b}")
            st.write(gap_result["strategy_differences"][:200] + "…")
            st.caption(
                "下方生成将 **直接使用上述差距分析**。如需更换品牌组合，"
                "请回到「品牌对比分析」重新选择品牌并点击「生成差距分析」。"
            )
    else:
        st.info(
            "建议先在「品牌对比分析」页完成双品牌对比并 **生成差距分析**，"
            "此处将自动引用分析结果生成更精准的内容策略。"
        )

    gap_use_sample = bool(
        (st.session_state.get("gap_analysis_result") or {}).get("use_sample")
    )
    if gap_result and gap_use_sample:
        sample_n = len(get_selected_sample_ids())
        st.caption(f"当前差距分析基于自定义样本（已选 {sample_n} 篇）。")

    with st.form("strategy_form"):
        if gap_result:
            st.text_input("你的品牌（参照品牌）", value=gap_result["brand_a"], disabled=True)
            st.text_input("主要竞品（对比品牌）", value=gap_result["brand_b"], disabled=True)
            display_brand = st.text_input(
                "文案署名品牌名（可选）",
                value=gap_result["brand_a"],
                placeholder="生成内容中使用的品牌名称，默认与参照品牌一致",
                help="仅影响生成文案中的品牌称呼，不改变所依据的差距分析数据。",
            )
        else:
            brands = get_brands()
            brand_options = [b["brand_name"] for b in brands] if brands else []
            display_brand = st.text_input(
                "你的品牌名称",
                placeholder="例如：我的品牌",
            )
            st.selectbox(
                "主要竞品品牌",
                options=brand_options if brand_options else ["（请先添加品牌）"],
                disabled=not brand_options,
            )
        focus_area = st.text_area(
            "关注领域 / 业务描述（可选）",
            placeholder="描述目标受众、产品定位和当前内容痛点，将融入生成内容…",
            height=100,
        )
        submitted = st.form_submit_button(
            "生成内容策略包",
            type="primary",
            disabled=not gap_result,
        )

    if submitted:
        if not gap_result:
            st.warning("请先在「品牌对比分析」页生成差距分析，再使用本功能。")
        else:
            brand_a = gap_result["brand_a"]
            brand_b = gap_result["brand_b"]
            target_name = (display_brand or brand_a).strip() or brand_a
            with st.spinner("正在调用 AI 生成内容策略包，请稍候（约 30-60 秒）…"):
                package, error = generate_content_strategy(
                    target_name,
                    brand_b,
                    gap_result,
                    focus_area,
                )
            if error:
                st.warning(error)
            else:
                st.session_state.content_strategy_package = {
                    "target": target_name,
                    "competitor": brand_b,
                    "gap_brand_a": brand_a,
                    "gap_brand_b": brand_b,
                    "result": package,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                }
                st.success(
                    f"内容策略包已生成 — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                render_strategy_content_package(package)

    cached_pkg = st.session_state.get("content_strategy_package")
    if cached_pkg and not submitted:
        st.divider()
        st.caption(
            f"上次生成：{cached_pkg.get('target')} vs {cached_pkg.get('competitor')} "
            f"（{cached_pkg.get('generated_at', '')[:16]}）"
        )
        render_strategy_content_package(cached_pkg["result"])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    inject_global_css()
    render_hero_header()

    schema_status = db.init_db()
    st.session_state["db_schema_status"] = schema_status
    if "selected_article_ids" not in st.session_state:
        st.session_state["selected_article_ids"] = []
    init_compare_data_mode()
    init_checked_article_ids()
    init_article_time_filters()
    init_article_brand_filter()
    init_article_category_filter()
    init_activity_extract_time()
    ensure_shared_data_cache()
    if not st.session_state.get("_db_schema_logged"):
        st.session_state["_db_schema_logged"] = True
        if schema_status.get("has_is_favorite_column"):
            print(
                f"[Brand Pulse] is_favorite 字段已就绪，"
                f"数据库中共 {schema_status.get('favorite_count', 0)} 篇收藏"
            )
        else:
            print("[Brand Pulse] 警告：is_favorite 字段缺失，请检查数据库迁移")

    page = render_sidebar_nav()

    render_api_status()

    if page == "品牌内容管理":
        render_brand_management()
    elif page == "品牌对比分析":
        render_brand_comparison()
    elif page == "内容策略生成":
        render_strategy_generation()


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        import sys

        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__, "--server.headless", "true"]
        sys.exit(stcli.main())
    else:
        main()

import io
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import pandas as pd

jieba.setLogLevel(20)

# 常见中文停用词（精简列表）
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "他", "她", "它", "我们", "他们", "以及", "等", "与", "及",
    "为", "被", "从", "以", "将", "对", "中", "而", "或", "其", "之", "于", "可以",
    "已", "更", "最", "能", "还", "并", "通过", "进行", "如何", "什么", "这个",
    "那个", "因为", "所以", "如果", "但是", "而且", "同时", "目前", "相关", "内容",
    "文章", "发布", "品牌", "公司", "产品", "服务", "用户", "市场", "行业",
}

CATEGORY_LABEL = "未分类"
DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
)


def _find_chinese_font() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def parse_article_date(article: dict) -> datetime | None:
    raw = (article.get("published") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00").split("+")[0])
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw[:30], fmt)
        except ValueError:
            continue
    return None


def _article_text(article: dict) -> str:
    parts = [
        article.get("title") or "",
        article.get("ai_summary") or "",
        article.get("summary") or "",
    ]
    return " ".join(parts)


def extract_keyword_freq(articles: list[dict], top_n: int = 80) -> Counter:
    counter: Counter = Counter()
    for article in articles:
        text = _article_text(article)
        if not text.strip():
            continue
        for word in jieba.cut(text):
            word = word.strip()
            if len(word) < 2:
                continue
            if word in STOPWORDS:
                continue
            if re.fullmatch(r"[\d\W]+", word):
                continue
            counter[word] += 1
    return Counter(dict(counter.most_common(top_n)))


def compare_keywords(freq_a: Counter, freq_b: Counter) -> dict:
    set_a = set(freq_a.keys())
    set_b = set(freq_b.keys())
    intersection = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    return {
        "intersection": sorted(intersection, key=lambda w: freq_a[w] + freq_b[w], reverse=True),
        "only_a": {w: freq_a[w] for w in only_a},
        "only_b": {w: freq_b[w] for w in only_b},
        "intersection_freq": {w: freq_a[w] + freq_b[w] for w in intersection},
    }


def get_category_distribution(articles: list[dict]) -> pd.DataFrame:
    counts: Counter = Counter()
    for article in articles:
        cat = (article.get("category") or "").strip() or CATEGORY_LABEL
        counts[cat] += 1
    if not counts:
        return pd.DataFrame(columns=["category", "count"])
    return pd.DataFrame(
        [{"category": k, "count": v} for k, v in counts.most_common()]
    )


def get_monthly_publish_counts(articles: list[dict], months: int = 3) -> pd.DataFrame:
    today = datetime.now().replace(day=1)
    month_starts = []
    for i in range(months - 1, -1, -1):
        year = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year -= 1
        month_starts.append(datetime(year, month, 1))

    labels = [dt.strftime("%Y-%m") for dt in month_starts]
    counts = {label: 0 for label in labels}

    window_start = month_starts[0]
    for article in articles:
        dt = parse_article_date(article)
        if not dt or dt < window_start:
            continue
        label = dt.strftime("%Y-%m")
        if label in counts:
            counts[label] += 1

    return pd.DataFrame({"月份": labels, "发布量": [counts[l] for l in labels]})


def build_comparison_data(
    brand_a: str,
    brand_b: str,
    articles_a: list[dict],
    articles_b: list[dict],
) -> dict:
    freq_a = extract_keyword_freq(articles_a)
    freq_b = extract_keyword_freq(articles_b)
    kw = compare_keywords(freq_a, freq_b)

    return {
        "brand_a": brand_a,
        "brand_b": brand_b,
        "articles_a_count": len(articles_a),
        "articles_b_count": len(articles_b),
        "category_a": get_category_distribution(articles_a),
        "category_b": get_category_distribution(articles_b),
        "monthly_a": get_monthly_publish_counts(articles_a),
        "monthly_b": get_monthly_publish_counts(articles_b),
        "freq_a": freq_a,
        "freq_b": freq_b,
        "keywords": kw,
    }


def _wordcloud_image(freq: dict, title: str) -> bytes | None:
    if not freq:
        return None
    font_path = _find_chinese_font()
    try:
        from wordcloud import WordCloud

        wc = WordCloud(
            width=800,
            height=400,
            background_color="#0B0F19",
            font_path=font_path,
            max_words=60,
            colormap="cool",
            mode="RGBA",
        ).generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0B0F19")
        ax.set_facecolor("#0B0F19")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        title_font = _matplotlib_font()
        if title_font:
            ax.set_title(title, fontsize=14, fontproperties=title_font, color="#B0C4DE")
        else:
            ax.set_title(title, fontsize=14, color="#B0C4DE")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0B0F19")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return _matplotlib_bar_image(freq, title)


def _matplotlib_font():
    from matplotlib import font_manager

    font_path = _find_chinese_font()
    if font_path:
        return font_manager.FontProperties(fname=font_path)
    return None


def _matplotlib_bar_image(freq: dict, title: str) -> bytes:
    font_prop = _matplotlib_font()
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    words = [w for w, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(words) * 0.35)))
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.barh(words[::-1], counts[::-1], color="#00D4FF")
    ax.set_xlabel("词频", fontproperties=font_prop, color="#B0C4DE")
    ax.set_title(title, fontproperties=font_prop, color="#B0C4DE")
    ax.tick_params(colors="#B0C4DE")
    ax.spines["bottom"].set_color("#2A3A5C")
    ax.spines["left"].set_color("#2A3A5C")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if font_prop:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)
            label.set_color("#B0C4DE")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0B0F19")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_keyword_wordclouds(data: dict) -> tuple[bytes | None, bytes | None]:
    """生成两品牌差异关键词词云/柱状图图片。"""
    only_a = data["keywords"]["only_a"]
    only_b = data["keywords"]["only_b"]
    img_a = _wordcloud_image(
        only_a,
        f"{data['brand_a']} 独有高频词",
    )
    img_b = _wordcloud_image(
        only_b,
        f"{data['brand_b']} 独有高频词",
    )
    return img_a, img_b

import json
import os
import re
import time

from openai import OpenAI

MODEL_NAME = "deepseek-chat"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5
MAX_ARTICLES_IN_PROMPT = 50

SYSTEM_PROMPT = """你是一个品牌营销与内容分析专家。请根据提供的品牌文章列表，分析该品牌的内容调性。

必须严格返回 JSON 格式，不要包含 markdown 代码块、注释或其他文字。JSON 结构如下：
{
  "tone_style": "语气风格，如：专业严谨、亲切活泼（可附一句简短说明）",
  "top_keywords": [
    {"keyword": "关键词1", "weight": 95},
    {"keyword": "关键词2", "weight": 88}
  ],
  "topic_distribution": [
    {"topic": "主题名称", "percentage": 35.5}
  ]
}

要求：
- top_keywords 恰好 10 个，按 weight 从高到低排列，weight 为 1-100 的整数表示相对热度
- topic_distribution 包含 3-6 个主题，percentage 之和必须等于 100
- 全部使用中文
"""


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def _build_articles_text(articles: list[dict]) -> str:
    lines = []
    for index, article in enumerate(articles[:MAX_ARTICLES_IN_PROMPT], start=1):
        content = (article.get("ai_summary") or article.get("summary") or "").strip()
        category = article.get("category") or "未分类"
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(
            f"{index}. 标题：{article.get('title', '无标题')}\n"
            f"   分类：{category}\n"
            f"   摘要：{content or '（无摘要）'}"
        )
    if len(articles) > MAX_ARTICLES_IN_PROMPT:
        lines.append(f"\n（另有 {len(articles) - MAX_ARTICLES_IN_PROMPT} 篇文章未列出）")
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)


def _normalize_result(data: dict) -> dict:
    tone_style = str(data.get("tone_style", "")).strip() or "未能识别"

    keywords_raw = data.get("top_keywords") or []
    top_keywords: list[dict] = []
    for item in keywords_raw[:10]:
        if isinstance(item, dict):
            keyword = str(item.get("keyword", "")).strip()
            weight = item.get("weight", 0)
        elif isinstance(item, str):
            keyword = item.strip()
            weight = max(10, 100 - len(top_keywords) * 8)
        else:
            continue
        if keyword:
            try:
                weight = int(float(weight))
            except (TypeError, ValueError):
                weight = 50
            top_keywords.append({"keyword": keyword, "weight": weight})

    while len(top_keywords) < 10:
        top_keywords.append({"keyword": f"关键词{len(top_keywords) + 1}", "weight": 10})

    topics_raw = data.get("topic_distribution") or []
    topic_distribution: list[dict] = []
    for item in topics_raw:
        if not isinstance(item, dict):
            continue
        topic = str(
            item.get("topic")
            or item.get("主题")
            or item.get("name")
            or item.get("title")
            or ""
        ).strip()
        if not topic or topic.lower() in ("undefined", "none", "null"):
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
            percentage = float(raw_pct) if raw_pct is not None else 0.0
        except (TypeError, ValueError):
            percentage = 0.0
        topic_distribution.append({"topic": topic, "percentage": percentage})

    if not topic_distribution:
        topic_distribution = [{"topic": "其他", "percentage": 100.0}]
    else:
        total = sum(t["percentage"] for t in topic_distribution)
        if total <= 0:
            even = 100.0 / len(topic_distribution)
            for t in topic_distribution:
                t["percentage"] = round(even, 1)
        else:
            for t in topic_distribution:
                t["percentage"] = round(t["percentage"] / total * 100, 1)
            diff = 100.0 - sum(t["percentage"] for t in topic_distribution)
            if diff != 0:
                topic_distribution[0]["percentage"] = round(
                    topic_distribution[0]["percentage"] + diff, 1
                )

    return {
        "tone_style": tone_style,
        "top_keywords": top_keywords[:10],
        "topic_distribution": topic_distribution,
    }


def analyze_brand_tone(
    brand_name: str, articles: list[dict]
) -> tuple[dict | None, str | None]:
    """
    分析品牌调性。
    返回 (分析结果字典, 错误信息)。成功时错误信息为 None。
    """
    if not articles:
        return None, "该品牌暂无文章，请先采集内容。"

    user_content = (
        f"品牌名称：{brand_name}\n"
        f"文章总数：{len(articles)}\n\n"
        f"文章列表：\n{_build_articles_text(articles)}"
    )

    last_error: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.4,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json(raw)
            return _normalize_result(parsed), None
        except json.JSONDecodeError as exc:
            last_error = f"AI 返回格式解析失败：{exc}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    return None, last_error or "品牌调性分析请求失败"

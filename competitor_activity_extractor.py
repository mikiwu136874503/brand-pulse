import json
import os
import re
import time

from openai import OpenAI

MODEL_NAME = "deepseek-chat"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5
ARTICLES_PER_BATCH = 12

SYSTEM_PROMPT = """你是一个竞争情报分析助手。从品牌内容文章中识别竞品活动信息。

活动类型包括但不限于：发布会、展会、沙龙、研讨会、峰会、论坛、路演、体验活动、开放日、签约仪式等。

规则：
1. 对每篇文章：若文中未提及任何可识别的活动，则不要为该文章输出任何记录。
2. 若一篇文章提及多个活动，可输出多条记录。
3. source_article_link 必须与用户消息中该篇文章的「链接」字段完全一致。
4. 无法从文中推断的「时间」「地点」填「未提及」。
5. brand_name 使用用户消息中的品牌名；activity_type、activity_name、time、location 均使用中文。

必须严格返回 JSON，不要包含 markdown 代码块或其他文字：
{
  "activities": [
    {
      "brand_name": "品牌名称",
      "activity_type": "活动类型",
      "activity_name": "活动名称",
      "time": "活动时间",
      "location": "活动地点",
      "source_article_link": "https://..."
    }
  ]
}

若无任何活动，返回 {"activities": []}。"""


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


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


def _build_batch_prompt(articles: list[dict]) -> str:
    blocks: list[str] = []
    for index, article in enumerate(articles, start=1):
        content = (article.get("ai_summary") or article.get("summary") or "").strip()
        if len(content) > 500:
            content = content[:500] + "…"
        blocks.append(
            f"""--- 文章 {index} ---
品牌：{article.get("brand_name", "")}
标题：{article.get("title", "无标题")}
发布时间：{article.get("published") or "未知"}
链接：{article.get("link", "")}
正文/摘要：{content or "（无）"}"""
        )
    return (
        "请从以下文章中提取竞品活动信息。无活动信息的文章请跳过，不要编造。\n\n"
        + "\n\n".join(blocks)
    )


def _normalize_activity(
    item: dict, link_to_article: dict[str, dict]
) -> dict | None:
    link = str(item.get("source_article_link") or "").strip()
    activity_name = str(item.get("activity_name") or "").strip()
    if not link or not activity_name:
        return None

    source = link_to_article.get(link)
    brand_name = str(item.get("brand_name") or "").strip()
    if source and not brand_name:
        brand_name = source.get("brand_name", "")
    if not brand_name and source:
        brand_name = source["brand_name"]

    activity_type = str(item.get("activity_type") or "").strip() or "其他活动"
    time_text = str(item.get("time") or "").strip() or "未提及"
    location = str(item.get("location") or "").strip() or "未提及"

    sort_time = ""
    if source:
        sort_time = str(source.get("published") or "").strip()

    return {
        "brand_name": brand_name or "未知品牌",
        "activity_type": activity_type,
        "activity_name": activity_name,
        "time": time_text,
        "location": location,
        "source_article_link": link,
        "_sort_time": sort_time,
    }


def _dedupe_activities(activities: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for row in activities:
        key = (
            row["brand_name"],
            row["activity_name"],
            row["source_article_link"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _extract_batch(
    articles: list[dict], link_to_article: dict[str, dict]
) -> tuple[list[dict], str | None]:
    user_content = _build_batch_prompt(articles)
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
                temperature=0.2,
                max_tokens=3000,
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json(raw)
            raw_items = parsed.get("activities") or []
            if not isinstance(raw_items, list):
                raw_items = []

            normalized: list[dict] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                row = _normalize_activity(item, link_to_article)
                if row:
                    normalized.append(row)
            return normalized, None
        except json.JSONDecodeError as exc:
            last_error = f"AI 返回格式解析失败：{exc}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    return [], last_error or "活动信息提取失败"


def extract_competitor_activities(
    articles: list[dict],
) -> tuple[list[dict], str | None]:
    """
    从全部已采集文章中提取竞品活动信息。
    返回 (活动列表, 错误信息)。成功且无活动时列表为空。
    """
    if not articles:
        return [], "暂无已采集文章，请先在品牌内容管理中采集内容。"

    link_to_article = {
        str(a.get("link") or "").strip(): a
        for a in articles
        if str(a.get("link") or "").strip()
    }

    all_activities: list[dict] = []
    batches = [
        articles[i : i + ARTICLES_PER_BATCH]
        for i in range(0, len(articles), ARTICLES_PER_BATCH)
    ]

    for batch_index, batch in enumerate(batches, start=1):
        batch_results, error = _extract_batch(batch, link_to_article)
        if error:
            if all_activities:
                return (
                    _dedupe_activities(all_activities),
                    f"第 {batch_index} 批分析失败（已保留此前结果）：{error}",
                )
            return [], error
        all_activities.extend(batch_results)

    return _dedupe_activities(all_activities), None

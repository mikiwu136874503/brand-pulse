import json
import os
import re
import time

from openai import OpenAI

MODEL_NAME = "deepseek-chat"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5

SYSTEM_PROMPT = """你是一位资深品牌内容策略师与文案撰稿人。请根据提供的竞争差距分析，为品牌 A 生成完整的内容策略包。

必须严格返回 JSON 格式，不要包含 markdown 代码块、注释或其他文字。JSON 结构如下：
{
  "wechat_article": {
    "title": "吸引人的公众号推文标题",
    "content": "约800字的公众号推文正文，分段清晰，适合直接发布"
  },
  "social_posts": [
    "微博/小红书风格短文案 1（50-120字，可带 emoji）",
    "短文案 2",
    "短文案 3"
  ],
  "industry_opinions": [
    {"title": "观点标题 1", "content": "约200字的行业观点短文，可独立发布"},
    {"title": "观点标题 2", "content": "约200字"},
    {"title": "观点标题 3", "content": "约200字"}
  ],
  "content_calendar": [
    {"day": "周一", "topic": "当日内容主题", "brief": "内容简介（50-80字）"},
    {"day": "周二", "topic": "...", "brief": "..."},
    {"day": "周三", "topic": "...", "brief": "..."},
    {"day": "周四", "topic": "...", "brief": "..."},
    {"day": "周五", "topic": "...", "brief": "..."},
    {"day": "周六", "topic": "...", "brief": "..."},
    {"day": "周日", "topic": "...", "brief": "..."}
  ]
}

要求：
- wechat_article.content 约 800 字（700-900 字均可）
- social_posts 恰好 3 条，风格活泼、适合微博或小红书
- industry_opinions 恰好 3 篇，每篇 content 约 200 字
- content_calendar 恰好 7 天（周一至周日），结合差距分析中的改进建议
- 全部使用中文，内容具体、可执行，避免空泛套话
"""


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


def _build_gap_context(gap_result: dict) -> str:
    brand_a = gap_result.get("brand_a", "品牌 A")
    brand_b = gap_result.get("brand_b", "品牌 B")
    advantages = "\n".join(f"- {x}" for x in gap_result.get("advantages", []))
    disadvantages = "\n".join(f"- {x}" for x in gap_result.get("disadvantages", []))
    suggestions = "\n".join(
        f"{i}. {x}" for i, x in enumerate(gap_result.get("suggestions", []), start=1)
    )
    return f"""=== 差距分析（{brand_b} vs {brand_a}）===

【内容策略核心差异】
{gap_result.get("strategy_differences", "")}

【{brand_b} 相对 {brand_a} 的优势】
{advantages or "（无）"}

【{brand_b} 相对 {brand_a} 的劣势】
{disadvantages or "（无）"}

【针对 {brand_a} 的改进建议】
{suggestions or "（无）"}
"""


def _normalize_result(data: dict) -> dict:
    wechat = data.get("wechat_article") or {}
    title = str(wechat.get("title", "")).strip() or "待优化标题"
    content = str(wechat.get("content", "")).strip() or "（未能生成正文）"

    social_posts = [
        str(x).strip() for x in (data.get("social_posts") or []) if str(x).strip()
    ][:3]
    while len(social_posts) < 3:
        social_posts.append("（待生成短文案）")

    opinions_raw = data.get("industry_opinions") or []
    industry_opinions = []
    for item in opinions_raw[:3]:
        if isinstance(item, dict):
            industry_opinions.append(
                {
                    "title": str(item.get("title", "")).strip() or "行业观点",
                    "content": str(item.get("content", "")).strip() or "（待生成）",
                }
            )
    while len(industry_opinions) < 3:
        industry_opinions.append({"title": "行业观点", "content": "（待生成）"})

    calendar_raw = data.get("content_calendar") or []
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    content_calendar = []
    for index, item in enumerate(calendar_raw[:7]):
        if isinstance(item, dict):
            content_calendar.append(
                {
                    "day": str(item.get("day", weekdays[index])).strip(),
                    "topic": str(item.get("topic", "")).strip() or f"主题 {index + 1}",
                    "brief": str(item.get("brief", "")).strip() or "（待补充）",
                }
            )
    while len(content_calendar) < 7:
        idx = len(content_calendar)
        content_calendar.append(
            {
                "day": weekdays[idx],
                "topic": f"{weekdays[idx]}内容主题",
                "brief": "结合竞品差距持续输出差异化内容。",
            }
        )

    return {
        "wechat_article": {"title": title, "content": content},
        "social_posts": social_posts,
        "industry_opinions": industry_opinions,
        "content_calendar": content_calendar,
    }


def generate_content_strategy(
    target_brand: str,
    competitor_brand: str,
    gap_result: dict,
    focus_area: str = "",
) -> tuple[dict | None, str | None]:
    """
    根据差距分析生成内容策略包。
    返回 (结果字典, 错误信息)。
    """
    gap_text = _build_gap_context(gap_result)
    user_content = f"""请为品牌「{target_brand}」生成内容策略包。
竞品参考品牌：{competitor_brand}
{f"品牌业务描述：{focus_area}" if focus_area.strip() else ""}

{gap_text}

请基于以上差距分析，为「{target_brand}」制定差异化内容，突出相对竞品「{competitor_brand}」的竞争优势。"""

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
                temperature=0.7,
                max_tokens=4000,
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

    return None, last_error or "内容策略生成失败"

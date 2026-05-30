import json
import os
import re
import time

from openai import OpenAI

MODEL_NAME = "deepseek-chat"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5
MAX_ARTICLES_IN_PROMPT = 30

SYSTEM_PROMPT = """你是一个竞争品牌内容策略顾问。请根据两个品牌的对比数据，输出差距分析。

必须严格返回 JSON 格式，不要包含 markdown 代码块、注释或其他文字。JSON 结构如下：
{
  "strategy_differences": "内容策略核心差异的详细分析（200-400字，中文）",
  "advantages": ["品牌 B 相对于品牌 A 的优势 1", "优势 2", "..."],
  "disadvantages": ["品牌 B 相对于品牌 A 的劣势 1", "劣势 2", "..."],
  "suggestions": ["针对品牌 A 的可操作改进建议 1", "建议 2", "..."]
}

要求：
- advantages 和 disadvantages 各 3-5 条，具体、有依据
- suggestions 恰好 3-5 条，具体可操作，面向品牌 A 的内容团队
- 全部使用中文
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


def _format_category_df(df, brand_name: str) -> str:
    if df is None or df.empty:
        return f"{brand_name}：暂无分类数据"
    lines = [f"{row['category']} {row['count']} 篇" for _, row in df.iterrows()]
    return f"{brand_name}：" + "、".join(lines)


def _format_monthly_df(df, brand_name: str) -> str:
    if df is None or df.empty:
        return f"{brand_name}：暂无发布数据"
    parts = [f"{row['月份']} {row['发布量']} 篇" for _, row in df.iterrows()]
    return f"{brand_name}：" + "、".join(parts)


def _format_top_keywords(counter, top_n: int = 15) -> str:
    if not counter:
        return "（无）"
    items = counter.most_common(top_n)
    return "、".join(f"{w}({c})" for w, c in items)


def _build_articles_summary(brand_name: str, articles: list[dict]) -> str:
    lines = [f"【{brand_name}】文章样本（共 {len(articles)} 篇）："]
    for index, article in enumerate(articles[:MAX_ARTICLES_IN_PROMPT], start=1):
        content = (article.get("ai_summary") or article.get("summary") or "").strip()
        if len(content) > 200:
            content = content[:200] + "…"
        category = article.get("category") or "未分类"
        lines.append(
            f"{index}. [{category}] {article.get('title', '无标题')}\n   摘要：{content or '（无）'}"
        )
    if len(articles) > MAX_ARTICLES_IN_PROMPT:
        lines.append(f"（另有 {len(articles) - MAX_ARTICLES_IN_PROMPT} 篇未列出）")
    return "\n".join(lines)


def _build_user_prompt(
    brand_a: str,
    brand_b: str,
    compare_data: dict,
    articles_a: list[dict],
    articles_b: list[dict],
) -> str:
    kw = compare_data.get("keywords", {})
    intersection = kw.get("intersection", [])[:20]
    only_a = list(kw.get("only_a", {}).keys())[:15]
    only_b = list(kw.get("only_b", {}).keys())[:15]

    return f"""请分析以下两个品牌的内容差距。品牌 A 为参照基准，品牌 B 为对比对象。

=== 基础数据 ===
品牌 A：{brand_a}，文章数 {compare_data.get('articles_a_count', 0)}
品牌 B：{brand_b}，文章数 {compare_data.get('articles_b_count', 0)}

=== 内容类型分布 ===
{_format_category_df(compare_data.get('category_a'), brand_a)}
{_format_category_df(compare_data.get('category_b'), brand_b)}

=== 近 3 个月发布频率 ===
{_format_monthly_df(compare_data.get('monthly_a'), brand_a)}
{_format_monthly_df(compare_data.get('monthly_b'), brand_b)}

=== 关键词对比 ===
共同关键词：{'、'.join(intersection) if intersection else '（无明显交集）'}
{brand_a} 独有关键词：{'、'.join(only_a) if only_a else '（无）'}
{brand_b} 独有关键词：{'、'.join(only_b) if only_b else '（无）'}

=== 文章样本 ===
{_build_articles_summary(brand_a, articles_a)}

{_build_articles_summary(brand_b, articles_b)}
"""


def _normalize_result(data: dict, brand_a: str, brand_b: str) -> dict:
    strategy = str(data.get("strategy_differences", "")).strip()
    if not strategy:
        strategy = "未能生成策略差异分析。"

    def _to_list(key: str, min_items: int = 1) -> list[str]:
        raw = data.get(key) or []
        items = [str(x).strip() for x in raw if str(x).strip()]
        return items if items else [f"暂无{key}数据"]

    advantages = _to_list("advantages")
    disadvantages = _to_list("disadvantages")
    suggestions = _to_list("suggestions")
    suggestions = suggestions[:5]
    while len(suggestions) < 3:
        suggestions.append("建议结合竞品动态持续优化内容组合与发布节奏。")

    return {
        "brand_a": brand_a,
        "brand_b": brand_b,
        "strategy_differences": strategy,
        "advantages": advantages,
        "disadvantages": disadvantages,
        "suggestions": suggestions,
    }


def generate_gap_analysis(
    brand_a: str,
    brand_b: str,
    compare_data: dict,
    articles_a: list[dict],
    articles_b: list[dict],
) -> tuple[dict | None, str | None]:
    """生成两品牌差距分析。返回 (结果字典, 错误信息)。"""
    if not articles_a or not articles_b:
        return None, "两个品牌均需有文章数据才能生成差距分析。"

    user_content = _build_user_prompt(
        brand_a, brand_b, compare_data, articles_a, articles_b
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
                temperature=0.5,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json(raw)
            return _normalize_result(parsed, brand_a, brand_b), None
        except json.JSONDecodeError as exc:
            last_error = f"AI 返回格式解析失败：{exc}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    return None, last_error or "差距分析请求失败"

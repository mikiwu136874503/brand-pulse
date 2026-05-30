import os
import time

from openai import OpenAI

CATEGORIES = ("案例研究", "产品更新", "行业洞察", "技术博客", "其他")
MODEL_NAME = "deepseek-chat"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5

SYSTEM_PROMPT = (
    "你是一个内容分类助手。根据文章标题和摘要，将文章归类为以下五个类别之一："
    "案例研究、产品更新、行业洞察、技术博客、其他。"
    "只返回一个分类名称，不要返回任何其他文字、标点或解释。"
)


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def _normalize_category(raw: str) -> str:
    text = (raw or "").strip().strip("「」\"'")
    for category in CATEGORIES:
        if category in text:
            return category
    return "其他"


def classify_article(title: str, summary: str) -> tuple[str | None, str | None]:
    """
    调用 DeepSeek 对文章分类。
    返回 (分类名称, 错误信息)。成功时错误信息为 None。
    """
    user_content = f"标题：{title or '无标题'}\n摘要：{summary or '（无摘要）'}"

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
                temperature=0.1,
                max_tokens=20,
            )
            raw = response.choices[0].message.content or ""
            return _normalize_category(raw), None
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    return None, last_error or "分类请求失败"

import os
import time

from openai import OpenAI

MODEL_NAME = "deepseek-chat"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5
MAX_SUMMARY_LENGTH = 100

SYSTEM_PROMPT = (
    "你是一个专业的内容摘要助手。根据文章标题和原文内容或摘要，"
    "生成简洁、准确的中文摘要。"
    "摘要必须在100字以内。"
    "只返回摘要正文，不要标题、前缀、引号或其他说明文字。"
)


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def _trim_summary(text: str) -> str:
    cleaned = (text or "").strip().strip("「」\"'")
    if len(cleaned) > MAX_SUMMARY_LENGTH:
        return cleaned[:MAX_SUMMARY_LENGTH]
    return cleaned


def generate_summary(title: str, original_summary: str) -> tuple[str | None, str | None]:
    """
    调用 DeepSeek 生成文章摘要。
    返回 (摘要文本, 错误信息)。成功时错误信息为 None。
    """
    user_content = (
        f"标题：{title or '无标题'}\n"
        f"原文内容或摘要：{original_summary or '（无原文摘要，请根据标题生成简要说明）'}"
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
                temperature=0.3,
                max_tokens=200,
            )
            raw = response.choices[0].message.content or ""
            summary = _trim_summary(raw)
            if not summary:
                return None, "AI 未返回有效摘要内容"
            return summary, None
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    return None, last_error or "摘要生成请求失败"

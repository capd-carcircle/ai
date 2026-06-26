"""
ai/agents/common.py — Gemini 호출 공통 유틸

ai_question_agent.py / summary_agent.py에서 반복되던 패턴 통합:
  - get_gemini_model(): GenerativeModel 생성
  - generate_with_retry(): 재시도 로직 포함 generate_content 호출
  - parse_json_response(): JSON 파싱 (partial recovery + regex fallback)
"""
import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from ai.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def get_gemini_model():
    """하위 호환용 — client 반환."""
    return get_genai_client()


def generate_with_retry(
    model,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_output_tokens: int = 4096,
    max_retries: int = 2,
    retry_temperature_delta: float = 0.2,
) -> str:
    client = get_genai_client()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        current_temp = min(temperature + retry_temperature_delta * attempt, 1.0)
        try:
            resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=current_temp,
                    max_output_tokens=max_output_tokens,
                ),
            )
            return resp.text
        except Exception as e:
            last_error = e
            logger.warning(f"generate_content 실패 (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if "429" in str(e) and attempt < max_retries:
                wait = 60 * (attempt + 1)
                logger.info(f"429 RPM 초과 — {wait}초 대기 후 재시도")
                time.sleep(wait)

    raise ValueError(f"Gemini 호출 {max_retries + 1}회 모두 실패: {last_error}")


def parse_json_response(text: str, array: bool = False) -> Any:
    """
    LLM 응답 텍스트에서 JSON 파싱.
    1차: 직접 json.loads
    2차: ```json ... ``` 코드블록 추출
    3차: 첫 번째 { } 또는 [ ] 블록 추출 (partial recovery)

    Args:
        text: LLM 응답 원문
        array: True면 리스트, False면 dict를 기대

    Returns:
        파싱된 Python 객체. 실패 시 [] 또는 {} 반환.
    """
    default = [] if array else {}

    # 1차: 직접 파싱
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2차: 코드블록 제거
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3차: 첫 번째 구조 추출
    pattern = r"\[.*?\]" if array else r"\{.*?\}"
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("JSON 파싱 전체 실패, 기본값 반환")
    return default

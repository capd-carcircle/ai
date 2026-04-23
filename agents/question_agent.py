"""
대화형 문진 에이전트
- 환자 기록 + 공통질문 답변을 받아 첫 질문 생성
- 환자 답변을 받아 다음 질문 생성 (멀티턴)
- 긴급 신호 감지 시 즉시 내원 안내 후 종료
"""
import json
import logging
from typing import Optional

import google.generativeai as genai

from ai.config import settings

logger = logging.getLogger(__name__)

# Gemini 초기화
genai.configure(api_key=settings.GEMINI_API_KEY)


def _build_system_prompt(record_data: dict, common_qa: list[dict], kdigo_context: str = "") -> str:
    """문진 에이전트 시스템 프롬프트"""
    kdigo_block = ""
    if kdigo_context:
        kdigo_block = f"""
[KDIGO 관련 지침]
{kdigo_context}
"""

    common_qa_text = ""
    if common_qa:
        lines = []
        for item in common_qa:
            answer = item.get("choice", "미응답")
            if item.get("text_answer"):
                answer += f" / 추가: {item['text_answer']}"
            lines.append(f"- {item['question_text']}: {answer}")
        common_qa_text = "\n[공통 질문 답변]\n" + "\n".join(lines)

    return f"""당신은 CAPD(복막투석) 환자를 돌보는 의료 AI 어시스턴트입니다.
환자의 오늘 투석 기록과 공통 질문 답변을 바탕으로, 의사가 확인해야 할 이상 징후를 추가로 파악하기 위해 대화형 문진을 진행합니다.
{kdigo_block}
[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}
{common_qa_text}

[문진 규칙]
- 한 번에 질문 하나만 하세요.
- 쉬운 한국어를 사용하세요 (의학 전문용어 최소화).
- 환자 답변에서 이상 신호가 보이면 구체적으로 파고드세요.
- 긴급 신호(복통, 고열, 극심한 증상 등) 발견 시 즉시 내원을 안내하고 문진을 종료하세요.
- 최대 {settings.MAX_TURNS}번 질문 후 마무리하세요.

[응답 형식 — 반드시 JSON으로만 응답]
정상 질문: {{"type": "question", "content": "질문 내용", "reason": "이 질문을 하는 이유"}}
긴급 상황: {{"type": "urgent", "content": "즉시 가까운 병원 응급실을 방문하시거나 119에 연락하세요. [이유]", "reason": "긴급 판단 근거"}}
문진 종료: {{"type": "done", "content": "오늘 문진이 완료되었습니다. 담당 의사가 기록을 확인할 예정입니다.", "reason": "종료 사유"}}"""


def start_conversation(
    record_data: dict,
    common_qa: list[dict],
    kdigo_context: str = "",
) -> dict:
    """
    문진 시작 — 첫 번째 AI 질문 생성

    Returns:
        {"type": "question"|"urgent"|"done", "content": str, "reason": str}
    """
    try:
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=_build_system_prompt(record_data, common_qa, kdigo_context),
        )

        # 히스토리 없이 첫 질문 요청
        response = model.generate_content(
            "환자 기록과 공통질문 답변을 분석해서 가장 중요하게 확인해야 할 첫 번째 질문을 해주세요.",
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        return _parse_response(response.text)

    except Exception as e:
        logger.error(f"문진 시작 실패: {e}")
        return {
            "type": "question",
            "content": "오늘 몸 상태가 어떠신가요? 특별히 불편한 점이 있으신가요?",
            "reason": "Gemini 호출 실패 — 기본 질문으로 대체",
        }


def next_turn(
    record_data: dict,
    common_qa: list[dict],
    history: list[dict],
    patient_answer: str,
    turn_number: int,
    kdigo_context: str = "",
) -> dict:
    """
    다음 질문 생성 (멀티턴)

    Args:
        history:        지금까지의 대화 [{"role": "ai"|"user", "content": str}, ...]
        patient_answer: 방금 환자가 한 답변
        turn_number:    현재 턴 번호 (1-based)

    Returns:
        {"type": "question"|"urgent"|"done", "content": str, "reason": str}
    """
    # 긴급 키워드 빠른 감지
    for keyword in settings.URGENT_KEYWORDS:
        if keyword in patient_answer:
            logger.warning(f"긴급 키워드 감지: '{keyword}'")
            return {
                "type": "urgent",
                "content": f"말씀하신 증상이 심각할 수 있습니다. 즉시 가까운 병원 응급실을 방문하시거나 119에 연락하세요.",
                "reason": f"환자 답변에서 긴급 키워드 '{keyword}' 감지",
            }

    # 최대 턴 초과 시 종료
    if turn_number >= settings.MAX_TURNS:
        return {
            "type": "done",
            "content": "오늘 문진이 완료되었습니다. 담당 의사가 기록을 검토할 예정입니다.",
            "reason": f"최대 문진 횟수({settings.MAX_TURNS}회) 도달",
        }

    try:
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=_build_system_prompt(record_data, common_qa, kdigo_context),
        )

        # Gemini 멀티턴 히스토리 구성
        chat_history = []
        for msg in history:
            role = "model" if msg["role"] == "ai" else "user"
            chat_history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=chat_history)

        response = chat.send_message(
            patient_answer,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        result = _parse_response(response.text)

        # Gemini가 done을 반환했더라도 마지막 턴 아니면 계속 질문
        return result

    except Exception as e:
        logger.error(f"다음 질문 생성 실패: {e}")
        return {
            "type": "done",
            "content": "오늘 문진이 완료되었습니다. 담당 의사가 기록을 검토할 예정입니다.",
            "reason": f"Gemini 호출 실패: {e}",
        }


def _parse_response(text: str) -> dict:
    """Gemini 응답 텍스트 → dict 파싱"""
    try:
        # ```json ... ``` 블록 처리
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        if data.get("type") not in ("question", "urgent", "done"):
            data["type"] = "question"

        return data

    except json.JSONDecodeError:
        logger.warning(f"JSON 파싱 실패, 원문: {text[:100]}")
        return {
            "type": "question",
            "content": text.strip()[:200],
            "reason": "JSON 파싱 실패 — 원문 반환",
        }

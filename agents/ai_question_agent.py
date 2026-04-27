"""
구조화된 설문용 AI 추천 질문 생성 에이전트
- 환자 투석 기록 이상 수치 기반으로 질문 생성
- 질문 타입: yes_no / single_select / multi_select / short_text
- 의사 공통 질문 아래 'AI 추천 질문' 섹션에 표시됨
- RAG(KDIGO 검색) 컨텍스트 주입 지원
- 고령 환자 대상: 쉬운 표현, 짧은 문장, 불안 유발 표현 금지
"""
import json
import logging
import re

import google.generativeai as genai

from ai.config import settings
from ai.tools.record_analyzer import summarize_anomalies_text

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)

# 유효한 질문 타입
VALID_TYPES = {"yes_no", "single_select", "multi_select", "short_text"}


def generate_ai_questions(
    record_data: dict,
    rejected_keys: list[str] = None,
    kdigo_context: str = "",
    historical_context: dict = None,
) -> list[dict]:
    """
    환자 투석 기록 기반 AI 추천 질문 생성 (구조화된 설문용)

    Args:
        record_data:        환자의 오늘 투석 기록 dict
        rejected_keys:      제외할 질문 패턴 키 목록
        kdigo_context:      RAG로 검색한 KDIGO 관련 문단
        historical_context: 환자 과거 기록 추세 데이터 (선택)
                            {days, bp:{avg,max,min,trend}, weight:{avg,delta_7d,trend},
                             uf:{weekly_avg,trend}, glucose:{avg,max}, risk_summary:{...}}

    Returns:
        [
            {
                "question_text": "질문 내용",
                "question_type": "yes_no" | "single_select" | "multi_select" | "short_text",
                "options": ["선택지1", "선택지2", ...] or null,
                "reason": "생성 이유"
            }
        ]
        오류 시 빈 리스트 반환
    """
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

        rejected_str = ", ".join(rejected_keys) if rejected_keys else "없음"
        anomaly_text = summarize_anomalies_text(record_data)

        # 과거 추세 블록 구성 — 기록이 1개라도 있으면 주입
        history_block = ""
        if historical_context and historical_context.get("days", 0) >= 1:
            h = historical_context
            bp = h.get("bp", {})
            wt = h.get("weight", {})
            uf = h.get("uf", {})
            gl = h.get("glucose", {})
            rs = h.get("risk_summary", {})
            uf_weekly = ", ".join(str(v) for v in uf.get("weekly_avg", [])) or "데이터 없음"
            history_block = f"""
[환자 과거 기록 추세 — {h['days']}일 기준]
- 혈압: 평균 {bp.get('avg', 'N/A')} mmHg, 최고 {bp.get('max', 'N/A')}, 추세: {bp.get('trend', 'N/A')}
- 체중: 평균 {wt.get('avg', 'N/A')} kg, 최근 변화 {wt.get('delta_7d', 'N/A')} kg, 추세: {wt.get('trend', 'N/A')}
- UF량 주간 평균(최근→과거): {uf_weekly}, 추세: {uf.get('trend', 'N/A')}
- 공복혈당: 평균 {gl.get('avg', 'N/A')}, 최고 {gl.get('max', 'N/A')} mg/dL
- 위험도 이력: 긴급 {rs.get('urgent', 0)}회 / 주의 {rs.get('caution', 0)}회 / 정상 {rs.get('normal', 0)}회
※ 오늘 수치가 이 추세와 다르게 변했다면, 그 변화를 우선적으로 질문에 반영하세요.
"""

        kdigo_block = ""
        if kdigo_context:
            kdigo_block = f"""
[KDIGO 관련 지침]
{kdigo_context}
"""

        prompt = f"""당신은 CAPD(복막투석) 환자를 담당하는 의료팀의 AI 보조 도구입니다.
아래 오늘의 투석 기록과 이상 수치 분석, 그리고 환자의 과거 추세를 종합하여 의사에게 환자 상태를 전달하기 위한 추가 질문 3~5개를 생성하세요.
{history_block}{kdigo_block}
[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[이상 수치 분석]
{anomaly_text}

[이미 제외된 패턴]
{rejected_str}

[질문 타입 선택 기준]
- yes_no: 단순 유무 확인 (예: "두통이 있었나요?")
- single_select: 여러 항목 중 하나 선택 (예: 증상 위치, 빈도)
- multi_select: 여러 항목 중 복수 선택 (예: 동반 증상 목록)
- short_text: 수치나 구체적 설명이 필요한 경우 (예: "소변량이 얼마나 되셨나요?")

[규칙]
- 이상 수치나 주의가 필요한 항목을 우선으로 하되, KDIGO 가이드라인 기반으로 임상적으로 의미 있는 질문을 생성하세요
- 과거 추세가 있다면 오늘 수치의 변화 방향(악화/개선/지속)을 반영하여 질문하세요
- KDIGO 지침이 있으면 해당 근거를 바탕으로 질문하세요
- 대부분 고령 환자임을 감안하여 쉽고 짧은 한국어 표현을 사용하세요 (의학 전문용어 금지)
- "심각한", "위험한", "응급" 등 불안감을 줄 수 있는 표현은 사용하지 마세요
- 이 질문은 진단이 아니라 의사에게 상태를 전달하기 위한 정보 수집입니다
- 제외된 패턴과 유사한 질문은 만들지 마세요
- 서로 다른 항목(혈압, 체중, 혈당, UF량, 증상 등)에 대해 다양하게 질문하세요
- question_text는 40자 이내로 간결하게 작성하세요
- single_select / multi_select 타입은 반드시 options를 3~5개 제공하세요
- yes_no / short_text 타입은 options를 null로 설정하세요

아래 JSON 배열 형식으로만 응답하세요 (3~5개):
[
  {{
    "question_text": "질문 내용",
    "question_type": "yes_no" | "single_select" | "multi_select" | "short_text",
    "options": ["선택지1", "선택지2", "선택지3"] | null,
    "reason": "이 질문을 생성한 이유 (20자 이내)"
  }},
  ...
]"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1500,
                response_mime_type="application/json",
            ),
        )

        text = response.text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 regex fallback
            match = re.search(r'"question_text"\s*:\s*"([^"]+)"', text)
            if match:
                return [{"question_text": match.group(1), "question_type": "yes_no", "options": None, "reason": ""}]
            return []

        if isinstance(data, list):
            results = []
            for q in data:
                if "question_text" in q:
                    results.append(_normalize_question(q))
            return results
        elif isinstance(data, dict) and "question_text" in data:
            return [_normalize_question(data)]

        return []

    except json.JSONDecodeError as e:
        logger.warning(f"AI 질문 JSON 파싱 실패: {e}")
        return []
    except Exception as e:
        logger.warning(f"AI 질문 생성 실패: {e}")
        return []


def _normalize_question(q: dict) -> dict:
    """질문 dict 정규화 — 필수 필드 보정"""
    q_type = q.get("question_type", "yes_no")
    if q_type not in VALID_TYPES:
        q_type = "yes_no"

    options = q.get("options")
    # yes_no / short_text는 options 불필요
    if q_type in ("yes_no", "short_text"):
        options = None
    # select 타입인데 options가 없으면 yes_no로 강제 변환
    elif q_type in ("single_select", "multi_select") and not options:
        q_type = "yes_no"
        options = None

    return {
        "question_text": q["question_text"],
        "question_type": q_type,
        "options":       json.dumps(options, ensure_ascii=False) if options else None,
        "reason":        q.get("reason", ""),
    }

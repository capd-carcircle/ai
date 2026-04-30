"""
위험도 트리아지 + AI 종합 요약 + EMR 작성 에이전트
- 기록 수치 + 공통 질문 응답 + AI 구조화 설문 응답을 종합 분석
- 의사용 AI 요약문 생성
- S/O/A/P EMR 작성
"""
import json
import logging

import google.generativeai as genai

from ai.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)


MAX_RETRIES = 2


def generate_summary_and_triage(
    record_data: dict,
    common_qa: list[dict],
    ai_survey_responses: list[dict] = None,
    historical_context: dict = None,
    rag_context: str = "",
    patient_profile: dict = None,
) -> dict:
    """
    설문 완료 후 위험도 + 요약 + EMR 생성

    Args:
        record_data:          오늘 투석 기록
        common_qa:            공통 질문 답변 목록
                              [{"question_text": str, "choice": "yes"|"no"|None, "text_answer": str}]
        ai_survey_responses:  AI 추천 질문 응답 목록
                              [{"question_text": str, "question_type": str, "answer": str}]
        historical_context:   최근 30일 집계 데이터 (없으면 생략)
                              {
                                "days": int,
                                "bp": {"avg": str, "max": str, "min": str, "trend": str},
                                "weight": {"avg": float, "delta_7d": float, "trend": str},
                                "uf": {"weekly_avg": list, "trend": str},
                                "glucose": {"avg": float, "max": float},
                                "risk_summary": {"urgent": int, "caution": int, "normal": int},
                              }

    Returns:
        {
            "risk_level":  "normal" | "caution" | "urgent",
            "ai_summary":  "의사용 요약 (2~4문장)",
            "emr_soap":    "S: ...\\nO: ...\\nA: ...\\nP: ...",
        }
    """
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

        # 공통질문 응답 정리
        common_text = "없음"
        if common_qa:
            lines = []
            for item in common_qa:
                answer = item.get("choice", "미응답") or "미응답"
                if item.get("text_answer"):
                    answer += f" / 추가: {item['text_answer']}"
                lines.append(f"- {item['question_text']}: {answer}")
            common_text = "\n".join(lines)

        # AI 설문 응답 정리
        ai_survey_text = "없음"
        if ai_survey_responses:
            lines = []
            for item in ai_survey_responses:
                q_type = item.get("question_type", "yes_no")
                answer = item.get("answer", "미응답") or "미응답"

                type_label = {
                    "yes_no":        "예/아니오",
                    "single_select": "단일 선택",
                    "multi_select":  "다중 선택",
                    "short_text":    "단답",
                }.get(q_type, q_type)

                lines.append(f"- [{type_label}] {item['question_text']}: {answer}")
            ai_survey_text = "\n".join(lines)

        # 과거 기록 집계 블록
        history_block = ""
        if historical_context and historical_context.get("days", 0) >= 3:
            h = historical_context
            bp = h.get("bp", {})
            wt = h.get("weight", {})
            uf = h.get("uf", {})
            gl = h.get("glucose", {})
            rs = h.get("risk_summary", {})
            uf_weekly = ", ".join(str(v) for v in uf.get("weekly_avg", [])) or "데이터 없음"
            history_block = f"""
[최근 {h['days']}일 추세 요약]
- 혈압: 평균 {bp.get('avg', 'N/A')}, 최고 {bp.get('max', 'N/A')}, 최저 {bp.get('min', 'N/A')}, 추세: {bp.get('trend', 'N/A')}
- 체중: 평균 {wt.get('avg', 'N/A')}kg, 최근 7일 변화 {wt.get('delta_7d', 'N/A')}kg, 추세: {wt.get('trend', 'N/A')}
- UF량 주간 평균(최근→과거): {uf_weekly}, 추세: {uf.get('trend', 'N/A')}
- 공복혈당: 평균 {gl.get('avg', 'N/A')}, 최고 {gl.get('max', 'N/A')} mg/dL
- 위험도 이력: 긴급 {rs.get('urgent', 0)}회 / 주의 {rs.get('caution', 0)}회 / 정상 {rs.get('normal', 0)}회
"""

        # 환자 프로필 블록 (self_memo + 의사 메모)
        patient_profile_block = ""
        if patient_profile:
            lines = []
            if patient_profile.get("self_memo"):
                lines.append(f"  - 환자 자기 기재 특이사항: {patient_profile['self_memo']}")
            if patient_profile.get("doctor_note"):
                lines.append(f"  - 담당 의사 임상 메모: {patient_profile['doctor_note']}")
            if lines:
                patient_profile_block = (
                    "\n[환자 개인 프로필]\n"
                    + "\n".join(lines)
                    + "\n※ 위 정보를 위험도 판단 및 요약 작성 시 맥락으로 반영하세요.\n"
                )

        # RAG 블록 구성
        rag_block = ""
        if rag_context:
            rag_block = f"""
[RAG 의학 지침 — KDIGO · ISPD · MedlinePlus 근거]
아래 지침을 위험도 판단의 기준으로 활용하세요.
{rag_context}
"""

        prompt = f"""당신은 CAPD(복막투석) 전문 의료 AI입니다.
아래 환자 데이터와 의학 지침을 바탕으로 위험도 분류, 의사용 요약, EMR(SOAP)을 작성하세요.
※ 요약과 EMR은 의사가 읽는 전문 문서입니다. 의학 용어(예: UF volume, peritonitis, hypertension, hyperglycemia, fluid overload, Kt/V 등)를 적극 사용하세요.
{history_block}{patient_profile_block}{rag_block}
[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[공통 질문 답변]
{common_text}

[AI 추천 질문 응답]
{ai_survey_text}

[위험도 분류 기준]
위 RAG 의학 지침(KDIGO · ISPD · MedlinePlus)을 기준으로 판단하세요.
- urgent(긴급): 즉각적인 의사 개입이 필요한 상태 (지침에서 immediate/urgent로 분류되는 소견)
- caution(주의): 경과 관찰 또는 조기 개입이 필요한 상태 (지침의 경고 기준에 해당하는 소견)
- normal(정상): 지침 기준 내 이상 소견 없음
※ RAG 지침이 없는 경우 임상적 판단으로 결정하세요.

[요약 작성 지침]
- 의사가 한눈에 파악할 수 있도록 2~4문장으로 핵심만 작성
- 이상 수치와 환자 호소 증상 중심으로 작성
- 최근 추세 요약이 있다면 오늘 수치를 추세 맥락에서 해석할 것 (예: "혈압이 2주 연속 상승 중")
- AI 설문 응답에서 주목할 내용이 있으면 포함
- 참고한 의학 지침이 있다면 간략히 언급 (예: "ISPD 기준 복막염 의심")
- 한국어로 작성

[EMR SOAP 작성]
- S(Subjective): 환자가 직접 호소한 증상 (설문 응답 기반)
- O(Objective): 객관적 수치 (기록 데이터 기반)
- A(Assessment): AI의 소견 및 위험도, 참고 지침
- P(Plan): 권장 조치 사항

[응답 형식 — 반드시 JSON으로만 응답]
{{
  "risk_level": "normal" | "caution" | "urgent",
  "ai_summary": "의사용 요약 텍스트",
  "emr_soap": "S: ...\\nO: ...\\nA: ...\\nP: ..."
}}"""

        for attempt in range(MAX_RETRIES + 1):
            try:
                temperature = 0.2 if attempt == 0 else 0.4
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                        # response_mime_type 제거 — constrained JSON 모드가 출력을 truncate하는 원인
                    ),
                )
                return _parse_summary_response(response.text)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"요약 생성 {attempt + 1}회차 실패, 재시도 중: {e}")
                else:
                    logger.error(f"요약/트리아지 {MAX_RETRIES + 1}회 모두 실패: {e}")

        return _fallback_triage(record_data)

    except Exception as e:
        logger.error(f"요약/트리아지 생성 실패 (프롬프트 구성 단계): {e}")
        return _fallback_triage(record_data)


def _parse_summary_response(text: str) -> dict:
    """응답 파싱 — Gemini가 문자열 값 안에 literal newline을 넣는 경우도 처리"""
    import re

    # 코드블록 제거
    clean = text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    # 1차 시도: 직접 파싱
    try:
        data = json.loads(clean)
        return _validate_summary(data)
    except json.JSONDecodeError:
        pass

    # 2차 시도: 문자열 값 안의 literal newline을 \n으로 치환 후 파싱
    try:
        fixed = re.sub(
            r'("(?:[^"\\]|\\.)*")',
            lambda m: m.group(0).replace('\n', '\\n').replace('\r', ''),
            clean,
        )
        data = json.loads(fixed)
        return _validate_summary(data)
    except Exception:
        pass

    # 3차 시도: regex로 각 필드 직접 추출
    # (?:[^"\\]|\\.)* → 이스케이프된 문자(\\") 포함한 문자열 값 올바르게 추출
    logger.warning(f"요약 JSON 파싱 실패, regex fallback 사용: {clean[:80]}")
    risk_match    = re.search(r'"risk_level"\s*:\s*"(normal|caution|urgent)"', clean)
    summary_match = re.search(r'"ai_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)
    emr_match     = re.search(r'"emr_soap"\s*:\s*"((?:[^"\\]|\\.)*)"',    clean, re.DOTALL)

    if risk_match or summary_match:
        return {
            "risk_level": risk_match.group(1) if risk_match else "caution",
            "ai_summary": summary_match.group(1).replace('\\n', '\n') if summary_match
                          else "AI 요약 생성에 실패했습니다. 의사가 직접 기록을 확인해 주세요.",
            "emr_soap":   emr_match.group(1).replace('\\n', '\n') if emr_match else "",
        }

    # 4차: 완전 실패 — 재시도 루프가 잡을 수 있도록 예외 발생
    logger.error(f"요약 JSON 완전 파싱 실패. 원본(200자): {clean[:200]}")
    raise ValueError(f"요약 JSON 완전 파싱 실패: {clean[:80]}")


def _validate_summary(data: dict) -> dict:
    if data.get("risk_level") not in ("normal", "caution", "urgent"):
        data["risk_level"] = "caution"
    return {
        "risk_level": data.get("risk_level", "caution"),
        "ai_summary": data.get("ai_summary", "요약 생성 실패"),
        "emr_soap":   data.get("emr_soap", "EMR 생성 실패"),
    }


def _fallback_triage(record_data: dict) -> dict:
    """Gemini 실패 시 규칙 기반 위험도 판단"""
    risk = "normal"

    bp = record_data.get("blood_pressure", "")
    try:
        systolic = int(bp.split("/")[0])
        if systolic >= 160 or systolic < 90:
            risk = "caution"
    except Exception:
        pass

    if record_data.get("turbid_peritoneal"):
        risk = "urgent"

    glucose = record_data.get("fasting_blood_glucose") or 0
    if glucose >= 250:
        risk = "caution"

    return {
        "risk_level": risk,
        "ai_summary": "AI 요약 생성에 실패했습니다. 의사가 직접 기록을 확인해 주세요.",
        "emr_soap":   "",
    }

"""
AI 맞춤 질문 생성 에이전트 (2-LLM 파이프라인)
문서(proj1_ANALYSE) 정의 기반:

  Step 1 — LLM 1: 기록 + 분석결과 → 5~10개 임상 쿼리/statement 생성
  Step 2 — RAG:   쿼리로 KDIGO·ISPD·MedlinePlus 관련 문단 검색
  Step 3 — LLM 2: 쿼리 + RAG 결과 → 환자용 질문 3~5개 생성

  분석결과(analytics_result)를 수치 계산 없이 LLM 추론에만 활용
  → 이상 탐지·추세 판단은 Python이 담당, Gemini는 임상 해석만 수행

SSE 스트리밍:
  generate_questions_stream() — async generator, 질문 하나씩 yield
"""
import json
import logging
import re
from typing import AsyncGenerator

from vertexai.generative_models import GenerativeModel, GenerationConfig

from ai.agents.common import get_gemini_model
from ai.config import settings  # noqa: F401 — vertexai.init() 호출 포함
from ai.rag.retriever import search_by_queries, search_kdigo_context

logger = logging.getLogger(__name__)

# 이상 없을 때 루틴 확인용 힌트 카테고리
ROUTINE_CATEGORIES = [
    "복막염 징후 (투석액 색깔·혼탁도·복통·발열)",
    "수분 균형 (부종·갈증·소변량 변화)",
    "식이 및 식욕 (염분 섭취·식욕 저하·구역감)",
    "투석 상태 (배액 속도·주입 불편감·카테터 부위)",
    "전신 컨디션 및 활동량 (피로·호흡 곤란·수면)",
]


# ════════════════════════════════════════════════════════════════
# 공통 파싱 유틸
# ════════════════════════════════════════════════════════════════

def _strip_codeblock(text: str) -> str:
    clean = text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()
    return clean


def _parse_questions(text: str) -> list[dict]:
    """
    Gemini 응답 → 질문 리스트 추출
    JSON 파싱 → partial recovery → regex fallback 순서
    """
    clean = _strip_codeblock(text)

    # 1차: 직접 파싱
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return [q for q in data if isinstance(q, dict) and "question_text" in q]
        if isinstance(data, dict) and "questions" in data:
            return [q for q in data["questions"] if isinstance(q, dict) and "question_text" in q]
    except json.JSONDecodeError:
        pass

    # 2차: partial recovery — 완성된 {...} 객체 개별 추출
    logger.warning(f"질문 JSON 파싱 실패, partial recovery 시도: {clean[:80]}")
    recovered = []
    for m in re.finditer(r'\{[^{}]*"question_text"[^{}]*\}', clean, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "question_text" in obj:
                recovered.append(obj)
        except json.JSONDecodeError:
            pass
    if recovered:
        return recovered

    # 3차: regex로 question_text만 추출
    texts = re.findall(r'"question_text"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
    if texts:
        return [{"question_text": t.replace('\\n', '\n'), "question_type": "yes_no"} for t in texts]

    return []


def _parse_queries(text: str) -> list[str]:
    """
    LLM 1 응답 → 쿼리 문자열 리스트 추출
    JSON 배열 또는 "queries" 키 dict 파싱
    """
    clean = _strip_codeblock(text)

    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return [str(q) for q in data if q]
        if isinstance(data, dict):
            for key in ("queries", "statements", "retrieval_queries", "medical_statements_for_retrieval"):
                if key in data and isinstance(data[key], list):
                    return [str(q) for q in data[key] if q]
    except json.JSONDecodeError:
        pass

    # fallback: 따옴표로 감싼 문장 추출
    return re.findall(r'"((?:[^"\\]|\\.){20,})"', clean)


# ════════════════════════════════════════════════════════════════
# 분석결과 → 프롬프트 텍스트 변환
# ════════════════════════════════════════════════════════════════

def _format_analytics_for_prompt(analytics_result: dict) -> str:
    """analytics.run_all_tasks() 결과를 LLM에 주입할 텍스트로 변환"""
    if not analytics_result:
        return ""

    lines = ["[데이터 분석 결과 (Python 계산값 — LLM은 해석만 수행)]"]

    # Trend Analysis
    trend = analytics_result.get("trend_analysis", {}).get("results", {})
    if trend:
        lines.append("\n▶ 추세 분석 (Trend Analysis)")
        for attr, res in trend.items():
            if isinstance(res, dict):
                lines.append(f"  · {attr}: {res.get('statement', '')}")

    # Anomaly Detection
    anomaly = analytics_result.get("anomaly_detection", {}).get("results", {})
    if anomaly:
        lines.append("\n▶ 이상 탐지 (Anomaly Detection)")
        for attr, res in anomaly.items():
            if isinstance(res, dict):
                lines.append(f"  · {attr}: {res.get('statement', '')}")

    # Correlation
    corr_results = analytics_result.get("attribute_correlation", {}).get("results", [])
    if corr_results:
        lines.append("\n▶ 속성 상관관계 (|r| ≥ 0.5)")
        for pair in corr_results[:5]:  # 상위 5쌍만
            lines.append(f"  · {pair.get('statement', '')}")

    # Anomaly 요약
    anomaly_attrs = analytics_result.get("anomaly_attrs", [])
    if anomaly_attrs:
        lines.append(f"\n⚠️ 이상 감지된 속성: {', '.join(anomaly_attrs)}")
    else:
        lines.append("\n✓ 이상 감지된 속성 없음")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Step 1 — LLM 1: 임상 쿼리/statement 생성
# ════════════════════════════════════════════════════════════════

def _generate_clinical_queries(
    record_data: dict,
    analytics_text: str,
    patient_profile: dict,
    model: GenerativeModel,
) -> list[str]:
    """
    분석 결과 + 오늘 기록 → RAG 검색용 임상 쿼리 5~10개 생성 (영어)
    """
    profile_block = ""
    if patient_profile:
        parts = []
        if patient_profile.get("self_memo"):
            parts.append(f"Patient note: {patient_profile['self_memo']}")
        if patient_profile.get("doctor_note"):
            parts.append(f"Doctor note: {patient_profile['doctor_note']}")
        if parts:
            profile_block = "\n[Patient Profile]\n" + "\n".join(parts) + "\n"

    prompt = f"""You are a clinician specialized in Continuous Ambulatory Peritoneal Dialysis (CAPD).

Given the patient's today's record and precomputed analytics outputs (trend analysis, anomaly detection, \
attribute correlation), generate 5 to 10 medically relevant queries or statements for downstream retrieval \
from KDIGO, ISPD, and MedlinePlus guidelines.

Rules:
- Write queries in English (the medical guidelines are in English)
- Focus on clinically significant findings from the analytics
- If no anomaly, generate queries about routine CAPD monitoring topics
- Do NOT make a final diagnosis
- Use cautious language: "may suggest", "consistent with", "should be reviewed"
{profile_block}
[Today's Record]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

{analytics_text}

[Output format — JSON array of strings only, no other text]
[
  "query or clinical statement 1",
  "query or clinical statement 2",
  ...
]"""

    try:
        resp = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=4096),
        )
        queries = _parse_queries(resp.text)
        logger.info(f"LLM 1: 임상 쿼리 {len(queries)}개 생성")
        return queries
    except Exception as e:
        logger.warning(f"LLM 1 임상 쿼리 생성 실패: {e}")
        return []


# ════════════════════════════════════════════════════════════════
# Step 3 — LLM 2: 환자용 질문 생성
# ════════════════════════════════════════════════════════════════

def _build_patient_questions_prompt(
    record_data: dict,
    analytics_text: str,
    clinical_queries: list[str],
    rag_context: str,
    rejected_keys: list[str],
    patient_profile: dict,
    has_anomaly: bool,
    common_question_responses: list[dict] = None,
) -> str:
    """LLM 2 프롬프트 빌드 (동기·스트리밍 공용)"""
    # RAG 블록
    rag_block = ""
    if rag_context:
        rag_block = f"\n[RAG 의학 지침 — KDIGO · ISPD · MedlinePlus]\n{rag_context}\n"

    # 임상 쿼리 블록
    queries_block = ""
    if clinical_queries:
        q_list = "\n".join(f"  - {q}" for q in clinical_queries)
        queries_block = f"\n[임상 분석에서 도출된 핵심 소견]\n{q_list}\n"

    # 환자 프로필 블록
    profile_block = ""
    if patient_profile:
        parts = []
        if patient_profile.get("self_memo"):
            parts.append(f"  - 환자 본인 특이사항: {patient_profile['self_memo']}")
        if patient_profile.get("doctor_note"):
            parts.append(f"  - 담당 의사 임상 메모: {patient_profile['doctor_note']}")
        if parts:
            profile_block = "\n[환자 개인 프로필]\n" + "\n".join(parts) + "\n"

    # 공통질문 답변 블록
    common_qa_block = ""
    if common_question_responses:
        lines = []
        for item in common_question_responses:
            q_text = item.get("question_text", "")
            answer = item.get("answer", "미응답")
            if q_text:
                lines.append(f"  - {q_text} → {answer}")
        if lines:
            common_qa_block = (
                "\n[환자가 이미 답변한 공통 질문 — 동일하거나 유사한 의미의 질문 절대 생성 금지]\n"
                + "\n".join(lines) + "\n"
            )

    # 기록지 수치 재확인 금지 블록
    record_fields_block = f"""
[기록지에서 이미 수집된 값 — 아래 항목을 그대로 재확인하는 질문 절대 생성 금지]
이 값들은 환자가 이미 기록지에 입력한 것이므로 다시 물어보는 것은 의미 없음.
단, 이 값을 근거로 기록지에 없는 주관적 증상·원인·맥락을 묻는 후속 질문은 허용.

- 복막액 혼탁 여부: {record_data.get('turbid_peritoneal')}
- 혈압: {record_data.get('blood_pressure')}
- 체중: {record_data.get('weight')}kg
- 소변 횟수: {record_data.get('urine_count')}회
- 공복혈당: {record_data.get('fasting_blood_glucose')}mg/dL
- 교환 회차별 배액량·주입량·농도·UF: 아래 기록지 데이터 참고

금지 예시: "오늘 복막액이 혼탁했나요?" (기록지에 이미 있음)
허용 예시: "복막액이 혼탁한 것 외에 복통이나 발열도 있었나요?" (기록지에 없는 정보)
"""

    # 이상 없을 때 루틴 힌트
    routine_block = ""
    if not has_anomaly:
        cats = "\n".join(f"  - {c}" for c in ROUTINE_CATEGORIES)
        routine_block = (
            "\n[루틴 확인 카테고리 — 이상 수치 없음, 아래에서 반드시 질문 생성]\n"
            + cats + "\n"
        )

    # 거절 패턴 블록
    if rejected_keys:
        rejected_lines = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(rejected_keys))
        rejected_block = f"""
[⛔ 의사가 이미 거절한 질문 — 아래 규칙을 반드시 준수]
다음 질문들은 의사가 부적절하다고 판단해 거절한 것입니다.
아래 질문 목록과 같은 임상 주제·의도·핵심 내용을 다루는 질문은
표현·어미·단어·문장 구조가 달라져도 절대 생성하지 마세요.
판단 기준: "환자가 이 질문에 답하면 거절된 질문에도 이미 답한 것이 되는가?" → 그렇다면 생성 금지.

{rejected_lines}
"""
    else:
        rejected_block = ""

    return f"""당신은 CAPD(복막투석) 환자를 담당하는 의료 AI 어시스턴트입니다.
아래 데이터를 종합해 의사가 환자에게 확인할 질문 3~5개를 생성하세요.
{rag_block}{queries_block}{analytics_text}{profile_block}{common_qa_block}{record_fields_block}{routine_block}{rejected_block}
[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[질문 생성 규칙]
- 반드시 3개 이상 5개 이하의 질문을 생성하세요 (이상 수치가 없어도 무조건 3개 이상)
- 임상 소견과 RAG 지침을 기반으로 질문을 생성하세요
- 과거 추세 이상이 있으면 추세 변화를 고려한 질문 포함
- 환자가 이해하기 쉬운 한국어 표현 사용
- 질문 어미: "~나요?", "~셨나요?", "~인가요?" (금지: "~는지요?", "~었는지요?")
- question_type: yes_no / single_select / multi_select / short_text
- yes_no: options에 반드시 [긍정_답변, 부정_답변] 형태 (예: ["있었다","없었다"])
- single_select·multi_select: options 배열 필수 (2~4개)
- 질문 타입을 다양하게 섞으세요 (yes_no만 쓰지 말고 single_select·short_text 등 활용)

[⛔ 절대 생성 금지 질문 유형 — 윤리·환자 안전]
- 가족 사망·이혼·별거·가정불화 등 개인 생활사 관련 질문
- 정신 건강 직접 진단형 질문 (예: "우울하신가요?", "불안하거나 절망감이 드시나요?")
- 경제적 상황·치료비 부담을 묻는 질문
- 종교·신앙·신념 관련 질문
- 의사·병원·가족에 대한 불만이나 평가를 유도하는 질문
- "더 나빠지고 있는 것 같지 않나요?" 등 예후를 부정적으로 암시하는 질문
- 환자에게 수치심·죄책감을 유발할 수 있는 질문 (예: "식이요법을 제대로 지키셨나요?")
- 위 유형에 해당하는 질문은 임상적 근거가 있더라도 생성하지 마세요

[응답 형식 — JSON 배열만 출력, 다른 텍스트 금지]
[
  {{
    "question_text": "질문 내용",
    "question_type": "yes_no",
    "options": ["있었다", "없었다"],
    "reason": "질문 생성 근거 (의사용)"
  }}
]"""


def _generate_patient_questions(
    record_data: dict,
    analytics_text: str,
    clinical_queries: list[str],
    rag_context: str,
    rejected_keys: list[str],
    patient_profile: dict,
    has_anomaly: bool,
    model: GenerativeModel,
    temperature: float = 0.5,
    common_question_responses: list[dict] = None,
) -> list[dict]:
    """임상 쿼리 + RAG 결과 → 환자용 질문 3~5개 생성 (동기, 비스트리밍 경로용)"""
    prompt = _build_patient_questions_prompt(
        record_data, analytics_text, clinical_queries, rag_context,
        rejected_keys, patient_profile, has_anomaly,
        common_question_responses or [],
    )
    try:
        resp = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=temperature, max_output_tokens=4096),
        )
        return _parse_questions(resp.text)
    except Exception as e:
        logger.warning(f"LLM 2 질문 생성 실패 (temperature={temperature}): {e}")
        return []


async def _stream_patient_questions(
    prompt: str,
    model: GenerativeModel,
    temperature: float = 0.5,
) -> AsyncGenerator[dict, None]:
    """
    Gemini 스트리밍 API로 JSON 객체를 하나씩 실시간 파싱해서 yield.

    Gemini는 JSON 배열 전체를 토큰 단위로 스트리밍 출력한다.
    중괄호 깊이(brace depth)를 추적해서 완성된 `{...}` 객체가 나올 때마다 즉시 yield.
    """
    response = await model.generate_content_async(
        prompt,
        generation_config=GenerationConfig(temperature=temperature, max_output_tokens=4096),
        stream=True,
    )

    buffer = ""
    depth = 0
    in_string = False
    escape_next = False

    async for chunk in response:
        text = getattr(chunk, "text", "") or ""
        for char in text:
            # 이스케이프 처리
            if escape_next:
                escape_next = False
                buffer += char
                continue
            if char == "\\" and in_string:
                escape_next = True
                buffer += char
                continue
            # 문자열 경계
            if char == '"':
                in_string = not in_string
                buffer += char
                continue
            if in_string:
                buffer += char
                continue
            # JSON 객체 추적
            if char == "{":
                depth += 1
                buffer += char
            elif char == "}":
                buffer += char
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        # 완성된 JSON 객체
                        try:
                            obj = json.loads(buffer)
                            if isinstance(obj, dict) and "question_text" in obj:
                                yield obj
                        except json.JSONDecodeError as e:
                            logger.warning(f"스트리밍 JSON 파싱 실패: {e} | buf={buffer[:80]}")
                        buffer = ""
            else:
                buffer += char


# ════════════════════════════════════════════════════════════════
# 퍼블릭 인터페이스
# ════════════════════════════════════════════════════════════════

def generate_ai_questions(
    record_data: dict,
    rejected_keys: list[str] = None,
    kdigo_context: str = "",                      # 하위 호환 — 미사용 시 자동 검색
    historical_context: dict = None,              # 하위 호환 — analytics_result 없을 때 폴백용
    patient_profile: dict = None,
    analytics_result: dict = None,                # analytics.run_all_tasks() 결과
    common_question_responses: list[dict] = None, # 공통질문 답변 목록 (LLM2 맥락 보강)
) -> list[dict]:
    """
    2-LLM 파이프라인으로 AI 맞춤 질문 생성 (3~5개)

    Args:
        record_data:                오늘 투석 기록 dict (exchange_records 포함 가능)
        rejected_keys:              제외할 질문 패턴 목록
        kdigo_context:              기존 단일 RAG 컨텍스트 (analytics_result 없을 때 사용)
        historical_context:         기존 단순 집계 (analytics_result 없을 때 폴백)
        patient_profile:            {"self_memo": str, "doctor_note": str}
        analytics_result:           analytics.run_all_tasks() 결과 — 있으면 2-LLM 사용
        common_question_responses:  [{"question_text": str, "answer": str}, ...]

    Returns:
        [{"question_text", "question_type", "options", "reason"}]
    """
    rejected_keys              = rejected_keys or []
    patient_profile            = patient_profile or {}
    common_question_responses  = common_question_responses or []

    try:
        model = get_gemini_model()

        # analytics_result 여부로 파이프라인 분기
        if analytics_result:
            return _pipeline_2llm(
                record_data, rejected_keys, patient_profile,
                analytics_result, model, common_question_responses,
            )
        else:
            # analytics 없으면 기존 단일 LLM 방식 (하위 호환)
            logger.info("analytics_result 없음 — 기존 단일 LLM 방식 사용")
            return _pipeline_legacy(
                record_data, rejected_keys, kdigo_context,
                historical_context, patient_profile, model
            )

    except Exception as e:
        logger.error(f"AI 질문 생성 실패: {e}")
        return []


def _pipeline_2llm(
    record_data: dict,
    rejected_keys: list[str],
    patient_profile: dict,
    analytics_result: dict,
    model: GenerativeModel,
    common_question_responses: list[dict] = None,
) -> list[dict]:
    """2-LLM 파이프라인 실행"""
    analytics_text = _format_analytics_for_prompt(analytics_result)
    has_anomaly    = analytics_result.get("has_anomaly", False)

    # Step 1: LLM 1 → 임상 쿼리 생성
    clinical_queries = _generate_clinical_queries(
        record_data, analytics_text, patient_profile, model
    )

    # Step 2: RAG — 멀티쿼리 검색
    if clinical_queries:
        rag_context = search_by_queries(clinical_queries, top_k=3)
    else:
        # LLM 1 실패 시 기존 단순 검색으로 폴백
        logger.warning("LLM 1 쿼리 생성 실패 — 단순 RAG 검색으로 폴백")
        rag_context = search_kdigo_context(record_data)

    # Step 3: LLM 2 → 환자용 질문 생성 (최대 2회 재시도)
    best: list[dict] = []
    temperature = 0.5

    for attempt in range(3):
        questions = _generate_patient_questions(
            record_data, analytics_text, clinical_queries,
            rag_context, rejected_keys, patient_profile,
            has_anomaly, model, temperature,
            common_question_responses=common_question_responses or [],
        )
        if len(questions) > len(best):
            best = questions
        if len(best) >= 3:
            break
        temperature = min(temperature + 0.15, 1.0)
        logger.warning(
            f"질문 {len(questions)}개 생성 (시도 {attempt + 1}/3), "
            f"재시도 temperature={temperature:.2f}"
        )

    if not best:
        logger.error("2-LLM 파이프라인: 질문 생성 완전 실패")
    else:
        logger.info(f"2-LLM 파이프라인: 질문 {len(best)}개 생성 완료")

    return best


# ════════════════════════════════════════════════════════════════
# SSE 스트리밍 인터페이스
# ════════════════════════════════════════════════════════════════

async def generate_questions_stream(
    record_data: dict,
    rejected_keys: list[str] = None,
    patient_profile: dict = None,
    analytics_result: dict = None,
    common_question_responses: list[dict] = None,
) -> AsyncGenerator[dict, None]:
    """
    Gemini 스트리밍으로 질문을 생성되는 즉시 하나씩 yield (SSE용).

    Step 1 (LLM1 쿼리) + Step 2 (RAG) 는 동기로 실행 후,
    Step 3 (LLM2) 는 generate_content_async(stream=True) 로 실행해
    JSON 객체가 완성될 때마다 즉시 yield한다.

    3개 미만이면 동기 재시도 후 새 질문을 추가로 yield.

    Yields:
        {"question_text": str, "question_type": str, "options": list|None, "reason": str}
        {"__error__": str}  — 오류 시
    """
    rejected_keys             = rejected_keys or []
    patient_profile           = patient_profile or {}
    common_question_responses = common_question_responses or []

    try:
        model = get_gemini_model()

        if not analytics_result:
            # analytics 없으면 legacy 동기 방식 폴백
            logger.info("generate_questions_stream: analytics_result 없음 — legacy 방식 사용")
            questions = _pipeline_legacy(
                record_data, rejected_keys, "",
                None, patient_profile, model,
            )
            for q in questions:
                yield q
            return

        # ── Step 1: LLM 1 → 임상 쿼리 (동기) ──────────────────────
        analytics_text = _format_analytics_for_prompt(analytics_result)
        has_anomaly    = analytics_result.get("has_anomaly", False)

        clinical_queries = _generate_clinical_queries(
            record_data, analytics_text, patient_profile, model
        )

        # ── Step 2: RAG (동기) ─────────────────────────────────────
        if clinical_queries:
            rag_context = search_by_queries(clinical_queries, top_k=3)
        else:
            logger.warning("LLM 1 쿼리 생성 실패 — 단순 RAG 검색으로 폴백")
            rag_context = search_kdigo_context(record_data)

        # ── Step 3: LLM 2 스트리밍 ─────────────────────────────────
        prompt = _build_patient_questions_prompt(
            record_data, analytics_text, clinical_queries, rag_context,
            rejected_keys, patient_profile, has_anomaly,
            common_question_responses,
        )

        count = 0
        yielded_texts: set[str] = set()
        temperature = 0.5

        # 1차 시도: 스트리밍
        try:
            async for q in _stream_patient_questions(prompt, model, temperature):
                yield q
                yielded_texts.add(q.get("question_text", ""))
                count += 1
                if count >= 5:
                    break
        except Exception as stream_err:
            logger.warning(f"스트리밍 실패 — 동기 방식 폴백: {stream_err}")

        # 3개 미만이면 동기 재시도 (최대 2회)
        for attempt in range(2):
            if count >= 3:
                break
            temperature = min(temperature + 0.15, 1.0)
            logger.warning(f"스트리밍 {count}개 생성 — 동기 재시도 {attempt + 1}/2 (t={temperature:.2f})")
            more = _generate_patient_questions(
                record_data, analytics_text, clinical_queries, rag_context,
                rejected_keys, patient_profile, has_anomaly, model, temperature,
                common_question_responses,
            )
            for q in more:
                if q.get("question_text") not in yielded_texts:
                    yield q
                    yielded_texts.add(q.get("question_text", ""))
                    count += 1
                    if count >= 5:
                        break

        if count == 0:
            logger.error("generate_questions_stream: 질문 생성 완전 실패")

    except Exception as e:
        logger.error(f"generate_questions_stream 실패: {e}")
        yield {"__error__": str(e)}


# ════════════════════════════════════════════════════════════════
# Legacy 파이프라인 (analytics 없을 때 폴백)
# ════════════════════════════════════════════════════════════════

def _pipeline_legacy(
    record_data: dict,
    rejected_keys: list[str],
    kdigo_context: str,
    historical_context: dict,
    patient_profile: dict,
    model: GenerativeModel,
) -> list[dict]:
    """기존 단일 LLM 방식 — analytics_result 없을 때 폴백"""
    if not kdigo_context:
        kdigo_context = search_kdigo_context(record_data)

    best: list[dict] = []
    temperature = 0.5

    for attempt in range(3):
        questions = _generate_patient_questions(
            record_data=record_data,
            analytics_text="",
            clinical_queries=[],
            rag_context=kdigo_context,
            rejected_keys=rejected_keys,
            patient_profile=patient_profile,
            has_anomaly=False,
            model=model,
            temperature=temperature,
        )
        if len(questions) > len(best):
            best = questions
        if len(best) >= 3:
            break
        temperature = min(temperature + 0.15, 1.0)
        logger.warning(
            f"legacy 질문 {len(questions)}개 생성 (시도 {attempt + 1}/3), "
            f"재시도 temperature={temperature:.2f}"
        )

    if not best:
        logger.error("legacy 파이프라인: 질문 생성 완전 실패")
    else:
        logger.info(f"legacy 파이프라인: 질문 {len(best)}개 생성 완료")

    return best

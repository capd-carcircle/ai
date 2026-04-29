"""
정적 설문용 AI 맞춤 질문 생성 에이전트
- KDIGO·ISPD·MedlinePlus RAG 컨텍스트 + 과거 추세 기반으로 3~5개 질문 생성
- 이상 수치 없을 때도 CAPD 루틴 카테고리 힌트 주입으로 반드시 생성
- response_mime_type 제거 (constrained 모드가 배열 최소화하는 원인)
- 질문 3개 미만 시 최대 2회 재시도, temperature +0.15씩 상향
"""
import json
import logging
import re

import google.generativeai as genai

from ai.config import settings
from ai.tools.record_analyzer import summarize_anomalies_text

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)

# 이상 없을 때 Gemini에 힌트로 주입할 CAPD 루틴 카테고리
ROUTINE_CATEGORIES = [
    "복막염 징후 (투석액 색깔·혼탁도·복통·발열)",
    "수분 균형 (부종·갈증·소변량 변화)",
    "식이 및 식욕 (염분 섭취·식욕 저하·구역감)",
    "투석 상태 (배액 속도·주입 불편감·카테터 부위)",
    "전신 컨디션 및 활동량 (피로·호흡 곤란·수면)",
]


def _parse_questions(text: str) -> list[dict]:
    """
    Gemini 응답 텍스트에서 질문 리스트 추출
    - 코드블록 제거 → JSON 파싱 → partial recovery → regex fallback 순서
    """
    clean = text.strip()

    # 코드블록 제거
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

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
    logger.warning(f"AI 질문 JSON 파싱 실패, partial recovery 시도: {clean[:80]}")
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


def generate_ai_questions(
    record_data: dict,
    rejected_keys: list[str] = None,
    kdigo_context: str = "",
    historical_context: dict = None,
    patient_profile: dict = None,
) -> list[dict]:
    """
    환자 투석 기록 기반 AI 맞춤 질문 생성 (3~5개)

    Args:
        record_data:        환자의 오늘 투석 기록 dict
        rejected_keys:      제외할 질문 패턴 키 목록
        kdigo_context:      RAG로 검색한 KDIGO·ISPD·MedlinePlus 관련 문단
        historical_context: 최근 30일 집계 데이터 (선택)
        patient_profile:    환자 프로필 {"self_memo": str, "doctor_note": str} (선택)

    Returns:
        [{"question_text", "question_type", "options", "reason"}] 리스트
        오류 시 빈 리스트 반환
    """
    try:
        model = genai.GenerativeModel(model_name=settings.GEMINI_MODEL)

        rejected_str = ", ".join(rejected_keys) if rejected_keys else "없음"
        anomaly_text = summarize_anomalies_text(record_data)

        # RAG 블록
        kdigo_block = ""
        if kdigo_context:
            kdigo_block = f"""
[RAG 의학 지침 — KDIGO · ISPD · MedlinePlus]
{kdigo_context}
"""

        # 과거 추세 블록
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
[최근 {h['days']}일 추세]
- 혈압: 평균 {bp.get('avg', 'N/A')}, 추세: {bp.get('trend', 'N/A')}
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
                lines.append(f"  - 환자 본인 특이사항: {patient_profile['self_memo']}")
            if patient_profile.get("doctor_note"):
                lines.append(f"  - 담당 의사 임상 메모: {patient_profile['doctor_note']}")
            if lines:
                patient_profile_block = (
                    "\n[환자 개인 프로필]\n"
                    + "\n".join(lines)
                    + "\n※ 위 정보를 참고해 기저질환·특이사항에 맞는 개인화 질문을 생성하세요.\n"
                )

        # 이상 수치 없을 때 루틴 카테고리 힌트 주입
        routine_block = ""
        if anomaly_text == "이상 수치 없음":
            cats = "\n".join(f"  - {c}" for c in ROUTINE_CATEGORIES)
            routine_block = f"""
[루틴 확인 카테고리 — 이상 수치가 없어도 아래 항목에서 반드시 질문을 만드세요]
{cats}
"""

        prompt = f"""당신은 CAPD(복막투석) 환자를 담당하는 의료 AI 어시스턴트입니다.
아래 데이터를 종합해 의사가 환자에게 확인할 질문 3~5개를 생성하세요.
{kdigo_block}{history_block}{patient_profile_block}{routine_block}
[오늘 투석 기록]
{json.dumps(record_data, ensure_ascii=False, indent=2)}

[이상 수치 분석]
{anomaly_text}

[제외할 패턴]
{rejected_str}

[질문 생성 규칙]
- 반드시 3개 이상 5개 이하의 질문을 생성하세요 (이상 수치가 없어도 무조건 3개 이상)
- 이상 수치가 있으면 해당 항목 중심으로, 없으면 루틴 카테고리에서 골고루 선택
- 과거 추세가 있으면 추세 변화를 고려한 질문 포함 (예: "혈압이 2주째 상승 중인데 두통이 있나요?")
- KDIGO·ISPD 지침이 있으면 해당 근거 기반 질문 우선
- 환자가 이해하기 쉬운 한국어 표현 사용
- 제외 패턴과 유사한 질문 금지
- 질문 어미는 "~나요?", "~셨나요?", "~인가요?" 형식으로 작성. "~는지요?", "~었는지요?" 형식 금지
- question_type: yes_no(예/아니오), single_select(단일 선택), multi_select(다중 선택), short_text(단답)
- yes_no 질문은 options에 반드시 [긍정_답변, 부정_답변] 형태 레이블 지정 (예: ["있었다","없었다"], ["아팠다","괜찮았다"], ["늘었다","줄었다"])
- single_select·multi_select는 options 배열 필수 (2~4개)
- 질문 타입을 다양하게 섞으세요 — yes_no만 쓰지 말고 single_select·short_text 등 골고루 활용
  (예: 증상 빈도 → single_select ["없었다","가끔","자주"], 통증 부위 → multi_select, 특이사항 → short_text)

[응답 형식 — JSON 배열만 출력, 다른 텍스트 금지]
[
  {{
    "question_text": "질문 내용",
    "question_type": "yes_no",
    "options": ["있었다", "없었다"],
    "reason": "질문 생성 근거 (의사용)"
  }},
  {{
    "question_text": "질문 내용",
    "question_type": "single_select",
    "options": ["선택지1", "선택지2", "선택지3"],
    "reason": "질문 생성 근거 (의사용)"
  }}
]"""

        best: list[dict] = []
        temperature = 0.5

        # 질문 3개 미만이면 최대 2회 재시도, temperature +0.15씩 상향
        for attempt in range(3):
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                    # response_mime_type 제거 — constrained JSON 모드가 배열 최소화하는 원인
                ),
            )

            questions = _parse_questions(response.text)

            # 현재까지 best 보존
            if len(questions) > len(best):
                best = questions

            if len(best) >= 3:
                break

            temperature = min(temperature + 0.15, 1.0)
            logger.warning(
                f"AI 질문 {len(questions)}개 생성 (시도 {attempt + 1}/3), "
                f"재시도 temperature={temperature:.2f}"
            )

        if not best:
            logger.error("AI 질문 생성 완전 실패 (3회 시도)")
        else:
            logger.info(f"AI 질문 {len(best)}개 생성 완료")

        return best

    except Exception as e:
        logger.error(f"AI 질문 생성 실패: {e}")
        return []

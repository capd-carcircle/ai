"""
위험도 트리아지 + AI 종합 요약 + EMR 작성 에이전트
- 기록 수치 + 공통 질문 응답 + AI 구조화 설문 응답을 종합 분석
- 의사용 AI 요약문 생성
- S/O/A/P EMR 작성
"""
import json
import logging

from ai.agents.common import generate_with_retry, get_gemini_model
from ai.config import settings

logger = logging.getLogger(__name__)


MAX_RETRIES = 2


def generate_summary_and_triage(
    record_data: dict,
    common_qa: list[dict],
    ai_survey_responses: list[dict] = None,
    historical_context: dict = None,
    rag_context: str = "",
    patient_profile: dict = None,
    analytics_result: dict = None,
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
        analytics_result:     analytics.run_all_tasks() 결과 — 있으면 분석 데이터 주입

    Returns:
        {
            "risk_level":  "normal" | "caution" | "urgent",
            "ai_summary":  "의사용 요약 (2~4문장)",
            "emr_soap":    "S: ...\\nO: ...\\nA: ...\\nP: ...",
        }
    """
    try:
        model = get_gemini_model()

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

        # 분석 결과 블록 (analytics.run_all_tasks() 결과)
        analytics_block = ""
        if analytics_result:
            lines = ["[데이터 분석 결과 (Python 계산값 — 위험도 판단 근거로 활용)]"]

            # Trend
            trend_res = analytics_result.get("trend_analysis", {}).get("results", {})
            if trend_res:
                lines.append("\n▶ 추세 분석")
                for attr, res in trend_res.items():
                    if isinstance(res, dict) and res.get("statement"):
                        lines.append(f"  · {attr}: {res['statement']}")

            # Anomaly
            anomaly_res = analytics_result.get("anomaly_detection", {}).get("results", {})
            if anomaly_res:
                lines.append("\n▶ 이상 탐지")
                for attr, res in anomaly_res.items():
                    if isinstance(res, dict) and res.get("statement"):
                        lines.append(f"  · {attr}: {res['statement']}")

            # Correlation — task3는 이제 |r| 무관하게 전체 쌍을 반환하므로, LLM에는
            # 그중 |r| >= 0.5인 것만(정렬돼 있으니 앞에서부터) 최대 5쌍 주입
            corr_pairs = analytics_result.get("attribute_correlation", {}).get("results", [])
            strong_pairs = [p for p in corr_pairs if abs(p.get("correlation", 0)) >= 0.5]
            if strong_pairs:
                lines.append("\n▶ 주요 상관관계 (|r| ≥ 0.5)")
                for pair in strong_pairs[:5]:
                    lines.append(f"  · {pair.get('statement', '')}")

            # Anomaly 요약
            anomaly_attrs = analytics_result.get("anomaly_attrs", [])
            if anomaly_attrs:
                lines.append(f"\n⚠️ 이상 감지 속성: {', '.join(anomaly_attrs)}")

            analytics_block = "\n".join(lines) + "\n"

        # RAG 블록 구성
        rag_block = ""
        if rag_context:
            rag_block = f"""
[RAG 의학 지침 — KDIGO · ISPD · MedlinePlus 근거]
아래 지침을 위험도 판단의 기준으로 활용하세요.
{rag_context}
"""

        prompt = f"""당신은 CAPD(복막투석) 전문 의료 AI입니다.
아래 환자 데이터와 의학 지침을 바탕으로 위험도 분류, 구조화된 임상 요약, EMR(SOAP)을 작성하세요.
※ 요약과 EMR은 의사가 읽는 전문 문서입니다. 의학 용어(예: UF volume, peritonitis, hypertension, hyperglycemia, fluid overload, Kt/V 등)를 적극 사용하세요.
{analytics_block}{history_block}{patient_profile_block}{rag_block}
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

[findings 작성 지침 — 핵심 소견 목록]
- 이상 소견 하나당 객체 하나. 이상 없으면 1개 이상의 info 소견으로 작성.
- severity 기준:
  * "critical": 즉각 개입 필요 (urgent 수준 소견, RAG의 immediate/urgent 기준)
  * "warning": 경과 관찰 필요 (caution 수준 소견)
  * "info": 참고 정보 (정상 범위 내 특이사항 또는 긍정적 소견)
- text 작성 규칙:
  * 수치를 반드시 명시 (예: "혈압 165/95 mmHg")
  * 임상적 의미 또는 참고 지침 포함 (예: "— 수축기 고혈압 2도, KDIGO 조절 기준 초과")
  * 한국어로 작성, 간결하게 1문장
- 최대 5개, 중요도 순으로 정렬

[trend 작성 지침]
- historical_context가 있으면: 가장 주목할 추세 변화 한 문장 (예: "혈압이 최근 14일간 지속 상승 중 (평균 142→158→165 mmHg)")
- historical_context가 없으면: null

[action 작성 지침]
- 의사가 오늘 또는 이번 주 안에 해야 할 핵심 조치 한 줄
- 복수 조치는 "·"로 구분 (예: "이뇨제 용량 조정 검토 · 내주 체중 추이 모니터링")

[keywords 작성 지침]
- 이 기록을 대표하는 임상 키워드 2~5개
- 영어 의학 용어 또는 한국어 혼용 가능 (예: "고혈압2도", "fluidOverload", "복막염의심")
- 해시태그 형식 아닌 단어만 (# 없이)

[EMR SOAP 작성]
- S(Subjective): 환자가 직접 호소한 증상 (설문 응답 기반)
- O(Objective): 객관적 수치 (기록 데이터 기반)
- A(Assessment): AI의 소견 및 위험도, 참고 지침
- P(Plan): 권장 조치 사항

⚠️ 언어 규칙 (절대 준수):
- 모든 텍스트는 반드시 한국어로 작성합니다.
- 의학 약어(UF, BP, HR, EMR, SOAP 등)는 허용하되, 설명은 한국어로 작성합니다.
- S/O/A/P 항목 내용도 반드시 한국어 문장으로 작성합니다. 영어 문장 사용 금지.
- 예시(O): "혈압 145/90 mmHg, UF volume 850 g, 체중 62.3 kg"
- 예시(P): "혈압 조절 약물 용량 조정 검토 · 다음 방문 시 체중 및 UF 추이 재평가"

[응답 형식 — 반드시 아래 JSON 구조로만 응답, 다른 텍스트 금지]
{{
  "risk_level": "normal" | "caution" | "urgent",
  "findings": [
    {{"severity": "critical" | "warning" | "info", "text": "소견 한 문장"}},
    ...
  ],
  "trend": "추세 한 문장 또는 null",
  "action": "핵심 조치 한 줄",
  "keywords": ["키워드1", "키워드2"],
  "emr_soap": "S: ...\\nO: ...\\nA: ...\\nP: ..."
}}"""

        try:
            raw_text = generate_with_retry(
                model,
                prompt,
                temperature=0.2,
                max_output_tokens=8192,
                max_retries=MAX_RETRIES,
                retry_temperature_delta=0.2,
            )
            return _parse_summary_response(raw_text)
        except ValueError as e:
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
    logger.warning(f"요약 JSON 파싱 실패, regex fallback 사용: {clean[:80]}")
    risk_match  = re.search(r'"risk_level"\s*:\s*"(normal|caution|urgent)"', clean)
    emr_match   = re.search(r'"emr_soap"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)

    if risk_match:
        return _validate_summary({
            "risk_level": risk_match.group(1),
            "findings":   [{"severity": "info", "text": "AI 파싱 일부 실패 — 의사가 직접 기록을 확인해 주세요."}],
            "trend":      None,
            "action":     "기록 직접 확인 필요",
            "keywords":   [],
            "emr_soap":   emr_match.group(1).replace('\\n', '\n') if emr_match else "",
        })

    # 4차: 완전 실패 — 재시도 루프가 잡을 수 있도록 예외 발생
    logger.error(f"요약 JSON 완전 파싱 실패. 원본(200자): {clean[:200]}")
    raise ValueError(f"요약 JSON 완전 파싱 실패: {clean[:80]}")


def _validate_summary(data: dict) -> dict:
    """새 구조화 스키마 검증 및 정규화.
    findings / trend / action / keywords 필드를 ai_summary JSON 문자열로 직렬화하여 반환.
    프론트엔드는 ai_summary가 JSON이면 구조화 렌더링, 아니면 plain text 폴백.
    """
    if data.get("risk_level") not in ("normal", "caution", "urgent"):
        data["risk_level"] = "caution"

    # findings 정규화
    raw_findings = data.get("findings") or []
    findings = []
    for f in raw_findings:
        if isinstance(f, dict) and f.get("text"):
            sev = f.get("severity", "info")
            if sev not in ("critical", "warning", "info"):
                sev = "info"
            findings.append({"severity": sev, "text": f["text"]})
    if not findings:
        findings = [{"severity": "info", "text": "특이 소견 없음"}]

    # 구조화 요약을 JSON 문자열로 직렬화 → ai_summary 필드에 저장
    structured = {
        "findings": findings,
        "trend":    data.get("trend") or None,
        "action":   data.get("action") or "특이 조치 사항 없음",
        "keywords": data.get("keywords") or [],
    }

    return {
        "risk_level": data.get("risk_level", "caution"),
        "ai_summary": json.dumps(structured, ensure_ascii=False),
        "emr_soap":   data.get("emr_soap", ""),
    }


def _fallback_triage(record_data: dict) -> dict:
    """Gemini 실패 시 규칙 기반 위험도 판단"""
    risk = "normal"
    findings = []

    bp = record_data.get("blood_pressure", "")
    try:
        systolic = int(bp.split("/")[0])
        if systolic >= 180:
            risk = "urgent"
            findings.append({"severity": "critical", "text": f"혈압 {bp} mmHg — 고혈압 위기, 즉각 평가 필요"})
        elif systolic >= 160 or systolic < 90:
            risk = "caution"
            findings.append({"severity": "warning", "text": f"혈압 {bp} mmHg — 이상 범위, 경과 관찰 필요"})
    except Exception:
        pass

    if record_data.get("turbid_peritoneal"):
        risk = "urgent"
        findings.append({"severity": "critical", "text": "복막액 혼탁 — ISPD 기준 복막염 의심, 즉각 배양 검사 필요"})

    glucose = record_data.get("fasting_blood_glucose") or 0
    if glucose >= 250:
        if risk == "normal":
            risk = "caution"
        findings.append({"severity": "warning", "text": f"공복혈당 {glucose} mg/dL — 고혈당, 혈당 조절 검토 필요"})

    if not findings:
        findings = [{"severity": "info", "text": "AI 요약 생성에 실패했습니다. 의사가 직접 기록을 확인해 주세요."}]

    structured = {
        "findings": findings,
        "trend":    None,
        "action":   "기록 직접 확인 필요",
        "keywords": [],
    }

    return {
        "risk_level": risk,
        "ai_summary": json.dumps(structured, ensure_ascii=False),
        "emr_soap":   "",
    }

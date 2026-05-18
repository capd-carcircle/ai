"""
CAPD AI 서버 — FastAPI (포트 8001)
backend 서버와 HTTP로 통신

파이프라인 흐름:
  1. backend → raw 기록 + 과거 기록 전송
  2. ai/ → data_engineering으로 Daily Model Row 생성
  3. ai/ → analytics로 4가지 분석 Task 수행
  4. ai/ → 분석 결과를 에이전트에 주입
     - ai_question_agent: 2-LLM 파이프라인
     - summary_agent: 분석 결과 기반 위험도 판단
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.agents.summary_agent import generate_summary_and_triage
from ai.agents.ai_question_agent import generate_ai_questions
from ai.rag.retriever import search_kdigo_context, _get_model
from ai.tools.data_engineering import build_daily_model_row
from ai.tools.analytics import run_all_tasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("임베딩 모델 사전 로드 중...")
    _get_model()
    logger.info("임베딩 모델 준비 완료")
    yield


app = FastAPI(
    title="CAPD AI API",
    description="AI 추천 질문 생성 / 위험도 트리아지 / 종합 요약 AI 서버",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://backend:8000",
        "http://localhost:8000",
        "https://capd-backend-cdwaxwdxfa-du.a.run.app",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── 스키마 ─────────────────────────────────────────────────────

class SummaryRequest(BaseModel):
    record_data: dict                        # 오늘 기록 (exchange_records 포함)
    common_qa: list[dict] = []
    ai_survey_responses: list[dict] = []
    historical_context: dict = {}            # 기존 단순 집계 (하위 호환)
    patient_profile: dict = {}
    historical_records: list[dict] = []      # 과거 기록 raw 데이터 (data_engineering 입력)


class SummaryResponse(BaseModel):
    risk_level: str
    ai_summary: str
    emr_soap: str


class AIQuestionsRequest(BaseModel):
    record_data: dict                        # 오늘 기록 (exchange_records 포함)
    rejected_keys: list[str] = []
    historical_context: dict = {}            # 기존 단순 집계 (하위 호환)
    patient_profile: dict = {}
    historical_records: list[dict] = []      # 과거 기록 raw 데이터


class AIQuestionsResponse(BaseModel):
    questions: list[dict]


# ── 공통 유틸 ───────────────────────────────────────────────────

def _compute_analytics(record_data: dict, historical_records: list[dict]) -> dict | None:
    """
    record_data + historical_records → Daily Model Row 생성 → analytics 실행
    실패 시 None 반환 (에이전트는 legacy 모드로 폴백)
    """
    if not historical_records:
        return None
    try:
        today_row = build_daily_model_row(
            daily_data=record_data,
            exchange_records=record_data.get("exchange_records", []),
        )
        historical_rows = [
            build_daily_model_row(
                daily_data=rec,
                exchange_records=rec.get("exchange_records", []),
            )
            for rec in historical_records
        ]
        result = run_all_tasks(today_row, historical_rows)
        logger.info(
            f"Analytics 완료 — 이상 속성: {result.get('anomaly_attrs', [])}, "
            f"상관 쌍: {len(result.get('attribute_correlation', {}).get('results', []))}"
        )
        return result
    except Exception as e:
        logger.warning(f"Analytics 실패, 에이전트 legacy 폴백: {e}")
        return None


# ── 엔드포인트 ──────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "capd-ai", "version": "3.0.0"}


@app.post("/summary", response_model=SummaryResponse)
def create_summary(body: SummaryRequest):
    """
    설문 완료 후 위험도 + 요약 + EMR 생성
    analytics_result를 summary_agent에 주입하여 정확도 향상
    """
    analytics_result = _compute_analytics(body.record_data, body.historical_records)
    rag_context = search_kdigo_context(body.record_data, top_k=5)

    result = generate_summary_and_triage(
        record_data=body.record_data,
        common_qa=body.common_qa,
        ai_survey_responses=body.ai_survey_responses,
        historical_context=body.historical_context or {},
        rag_context=rag_context,
        patient_profile=body.patient_profile or {},
        analytics_result=analytics_result,
    )
    return SummaryResponse(**result)


@app.post("/ai-questions/generate", response_model=AIQuestionsResponse)
def generate_questions(body: AIQuestionsRequest):
    """
    AI 추천 질문 생성 (2-LLM 파이프라인)
    historical_records 있으면 analytics 후 2-LLM, 없으면 legacy 단일 LLM
    """
    analytics_result = _compute_analytics(body.record_data, body.historical_records)

    questions = generate_ai_questions(
        record_data=body.record_data,
        rejected_keys=body.rejected_keys,
        historical_context=body.historical_context or {},
        patient_profile=body.patient_profile or {},
        analytics_result=analytics_result,
    )
    return AIQuestionsResponse(questions=questions)


# ── 관리자 엔드포인트 ────────────────────────────────────────────

@app.post("/admin/ingest/medlineplus")
def admin_ingest_medlineplus(force: bool = False):
    """
    MedlinePlus RAG 데이터 업데이트
    GCP Cloud Scheduler가 매주 월요일 03:00 KST에 자동 호출.
    """
    try:
        from ai.ingest.medlineplus import ingest
        ingest(force=force)
        return {"status": "ok", "message": "MedlinePlus 인제스트 완료"}
    except Exception as e:
        logger.error(f"MedlinePlus 인제스트 실패: {e}")
        return {"status": "error", "message": str(e)}

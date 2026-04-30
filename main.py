"""
CAPD AI 서버 — FastAPI (포트 8001)
backend 서버와 HTTP로 통신
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.agents.summary_agent import generate_summary_and_triage
from ai.agents.ai_question_agent import generate_ai_questions
from ai.rag.retriever import search_kdigo_context, _get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 임베딩 모델 미리 로드 (첫 요청 지연 방지)
    logger.info("임베딩 모델 사전 로드 중...")
    _get_model()
    logger.info("임베딩 모델 준비 완료")
    yield


app = FastAPI(
    title="CAPD AI API",
    description="AI 추천 질문 생성 / 위험도 트리아지 / 종합 요약 AI 서버",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 스키마 ─────────────────────────────────────────────────────

class SummaryRequest(BaseModel):
    record_data: dict
    common_qa: list[dict] = []
    ai_survey_responses: list[dict] = []        # AI 구조화 설문 응답
    historical_context: dict = {}               # 최근 30일 집계 데이터 (선택)
    patient_profile: dict = {}                  # 환자 프로필 {"self_memo": str, "doctor_note": str}


class SummaryResponse(BaseModel):
    risk_level: str   # "normal" | "caution" | "urgent"
    ai_summary: str   # 의사용 요약
    emr_soap: str     # S/O/A/P EMR


class AIQuestionsRequest(BaseModel):
    """구조화 설문용 AI 추천 질문 생성 (surveys.py에서 호출)"""
    record_data: dict
    rejected_keys: list[str] = []   # 제외할 패턴 키 목록
    historical_context: dict = {}   # 환자 과거 기록 추세 데이터 (선택)
    patient_profile: dict = {}      # 환자 프로필 {"self_memo": str, "doctor_note": str}


class AIQuestionsResponse(BaseModel):
    questions: list[dict]           # [{"question_text", "question_type", "options", "reason"}]


# ── 엔드포인트 ──────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "capd-ai"}


@app.post("/summary", response_model=SummaryResponse)
def create_summary(body: SummaryRequest):
    """
    기록 제출 후 위험도 + 요약 + EMR 생성
    backend 서버가 설문 응답 저장 후 호출
    """
    # RAG 검색 — 위험도 판단용으로 top_k=5 (질문 생성보다 더 많은 컨텍스트)
    rag_context = search_kdigo_context(body.record_data, top_k=5)

    result = generate_summary_and_triage(
        record_data=body.record_data,
        common_qa=body.common_qa,
        ai_survey_responses=body.ai_survey_responses,
        historical_context=body.historical_context or {},
        rag_context=rag_context,
        patient_profile=body.patient_profile or {},
    )
    return SummaryResponse(**result)


@app.post("/ai-questions/generate", response_model=AIQuestionsResponse)
def generate_questions(body: AIQuestionsRequest):
    """
    정적 설문 AI 추천 질문 생성
    기록 제출 후 백그라운드에서 호출 — 의사 공통 질문 아래 AI 추천 질문 섹션에 표시
    """
    kdigo_context = search_kdigo_context(body.record_data)
    questions = generate_ai_questions(
        record_data=body.record_data,
        rejected_keys=body.rejected_keys,
        kdigo_context=kdigo_context,
        historical_context=body.historical_context or {},
        patient_profile=body.patient_profile or {},
    )
    return AIQuestionsResponse(questions=questions)

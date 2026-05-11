"""
AI 서버 설정
"""
import vertexai
from pydantic_settings import BaseSettings


class AISettings(BaseSettings):
    # Vertex AI (ADC 인증 — 서비스 계정 키 불필요)
    GCP_PROJECT_ID: str
    GCP_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-2.0-flash-001"

    # Backend DB 연결 (RAG용) — 반드시 .env에 설정 (기본값 없음, 하드코딩 금지)
    DATABASE_URL: str

    class Config:
        env_file = ".env"
        extra = "allow"


settings = AISettings()

# Vertex AI 초기화 (ADC 자동 인증)
vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

"""
AI 서버 설정
"""
from pydantic_settings import BaseSettings


class AISettings(BaseSettings):
    # Gemini API — 반드시 .env 파일에 설정 (코드에 키 하드코딩 금지)
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Backend DB 연결 (RAG용) — 반드시 .env에 설정 (기본값 없음, 하드코딩 금지)
    DATABASE_URL: str

    class Config:
        env_file = ".env"
        extra = "allow"


settings = AISettings()

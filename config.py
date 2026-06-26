"""
AI 서버 설정

⚠️ 모델명 주의 ⚠️
- 반드시 gemini-2.5-flash 사용
- gemini-2.0-flash / gemini-2.0-flash-001 은 이 프로젝트(skuniv-training-2)에서 404 발생 — 사용 불가
- 모델명 변경 시 deploy.yml의 --set-env-vars 도 함께 수정할 것
"""
from pydantic_settings import BaseSettings


class AISettings(BaseSettings):
    GCP_PROJECT_ID: str = ""
    GCP_LOCATION: str = "asia-northeast3"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: str = ""

    # Backend DB 연결 (RAG용)
    DATABASE_URL: str

    # 관리자 엔드포인트 인증 키
    ADMIN_SECRET_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "allow"


settings = AISettings()

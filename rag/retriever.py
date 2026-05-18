"""
KDIGO RAG 검색 모듈
- 환자 기록 → 검색 쿼리 변환
- pgvector <=> 연산으로 관련 KDIGO 청크 top-k 검색
- (구 backend/app/services/rag_service.py 이전)
"""
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from ai.config import settings

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# DB 세션 (ai 서버에서 직접 pgvector 접근)
_engine = create_engine(settings.DATABASE_URL)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info(f"임베딩 모델 로드: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def _record_to_query(record_data: dict) -> str:
    """환자 기록 dict → 영어 검색 쿼리 (KDIGO PDF가 영어이므로)"""
    parts = []

    bp = record_data.get("blood_pressure") or ""
    try:
        systolic = int(bp.split("/")[0])
        if systolic > 140:
            parts.append(f"high blood pressure {bp} mmHg hypertension management CAPD")
    except Exception:
        pass

    uf = record_data.get("total_ultrafiltration")
    if uf is not None:
        parts.append(f"ultrafiltration volume {uf} ml peritoneal dialysis")

    if record_data.get("turbid_peritoneal"):
        parts.append("cloudy dialysate turbid peritoneal peritonitis diagnosis")

    weight = record_data.get("weight")
    if weight:
        parts.append(f"fluid overload weight gain {weight} kg edema CAPD")

    glucose = record_data.get("fasting_blood_glucose")
    if glucose and glucose > 180:
        parts.append(f"fasting blood glucose {glucose} mg/dL diabetes CKD management")

    if not parts:
        parts.append("CAPD peritoneal dialysis patient monitoring guidelines")

    return " ".join(parts)


def search_by_queries(queries: list[str], top_k: int = TOP_K) -> str:
    """
    여러 자연어 쿼리로 RAG 검색 (2-LLM 파이프라인 Step 2용)
    각 쿼리를 개별 검색 후 중복 제거하여 합친 결과 반환.

    Args:
        queries: LLM이 생성한 임상 쿼리/statement 목록 (최대 5개 사용)
        top_k:   쿼리당 검색 청크 수
    """
    db: Session = _SessionLocal()
    try:
        model = _get_model()
        seen: set = set()
        chunks: list[str] = []

        for query_text in queries[:5]:
            if not query_text or not query_text.strip():
                continue
            vec = model.encode(query_text.strip(), normalize_embeddings=True).tolist()
            rows = db.execute(
                text("""
                    SELECT source, page_num, chunk_text
                    FROM document_chunks
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT :k
                """),
                {"vec": str(vec), "k": top_k},
            ).fetchall()

            for row in rows:
                key = (row.source, row.page_num)
                if key not in seen:
                    seen.add(key)
                    source = row.source.replace(".pdf", "").replace("-", " ")
                    chunks.append(f"({source}, p.{row.page_num})\n{row.chunk_text}")

        logger.info(f"멀티쿼리 RAG {len(chunks)}개 청크 검색 완료 (쿼리 {len(queries)}개)")
        return "\n\n".join(chunks)

    except Exception as e:
        logger.warning(f"멀티쿼리 RAG 검색 실패 (무시하고 계속): {e}")
        return ""
    finally:
        db.close()


def search_kdigo_context(record_data: dict, top_k: int = TOP_K) -> str:
    """
    환자 기록 기반 KDIGO 청크 검색 → 프롬프트 주입용 텍스트 반환

    Returns:
        관련 KDIGO 문단들을 합친 문자열. 오류 시 빈 문자열.
    """
    db: Session = _SessionLocal()
    try:
        model = _get_model()
        query_text = _record_to_query(record_data)
        query_vec = model.encode(query_text, normalize_embeddings=True).tolist()

        # pgvector 코사인 거리 검색
        rows = db.execute(
            text("""
                SELECT source, page_num, chunk_text
                FROM document_chunks
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """),
            {"vec": str(query_vec), "k": top_k},
        ).fetchall()

        if not rows:
            return ""

        chunks = []
        for i, row in enumerate(rows, start=1):
            source = row.source.replace(".pdf", "").replace("-", " ")
            chunks.append(f"[{i}] ({source}, p.{row.page_num})\n{row.chunk_text}")

        logger.info(f"KDIGO {len(rows)}개 청크 검색 완료")
        return "\n\n".join(chunks)

    except Exception as e:
        logger.warning(f"RAG 검색 실패 (무시하고 계속): {e}")
        return ""
    finally:
        db.close()

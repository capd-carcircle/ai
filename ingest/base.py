"""
공통 인제스트 파이프라인: 청킹 · 임베딩 · DB 저장

KDIGO 인제스트(backend/scripts/ingest_kdigo.py)와 동일한 청킹 로직 사용.
ISPD, MedlinePlus 등 모든 소스가 이 모듈을 공유함.
"""

import gc
import logging

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"   # KDIGO와 동일 모델 (384차원)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 8


def get_db_session() -> Session:
    from ai.config import settings
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def get_embedding_model() -> SentenceTransformer:
    logger.info(f"임베딩 모델 로드: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """텍스트를 고정 크기 청크로 분할. 문장 경계 최대한 존중."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        cut = end
        for sep in (". ", "! ", "? "):
            pos = text.rfind(sep, start, end)
            if pos != -1:
                cut = pos + 1
                break
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        new_start = cut - overlap
        if new_start <= start:
            new_start = cut
        start = new_start
    return [c for c in chunks if len(c) > 50]


def save_chunks(
    db: Session,
    source: str,
    chunks_with_pages: list[tuple[int | None, str]],
    model: SentenceTransformer,
) -> int:
    """청크 리스트를 임베딩 후 document_chunks에 저장. 저장된 개수 반환."""
    total = len(chunks_with_pages)
    saved = 0
    for i in range(0, total, BATCH_SIZE):
        batch = chunks_with_pages[i:i + BATCH_SIZE]
        texts = [c[1] for c in batch]
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for (page_num, chunk_txt), emb in zip(batch, embeddings):
            db.execute(
                text("""
                    INSERT INTO document_chunks (source, page_num, chunk_text, embedding, created_at)
                    VALUES (:source, :page_num, :chunk_text, CAST(:embedding AS vector), NOW())
                """),
                {
                    "source": source,
                    "page_num": page_num,
                    "chunk_text": chunk_txt,
                    "embedding": str(emb.tolist()),
                },
            )
        db.commit()
        saved += len(batch)
        del embeddings, texts, batch
        gc.collect()
        if saved % 40 == 0 or saved == total:
            logger.info(f"  [{source}] {saved}/{total}개 저장")
    return saved


def delete_chunks_by_source(db: Session, source: str) -> int:
    """특정 source의 청크 전체 삭제. 삭제된 개수 반환."""
    result = db.execute(
        text("DELETE FROM document_chunks WHERE source = :source"),
        {"source": source},
    )
    db.commit()
    return result.rowcount


def count_chunks_by_source(db: Session, source: str) -> int:
    return db.execute(
        text("SELECT COUNT(*) FROM document_chunks WHERE source = :source"),
        {"source": source},
    ).scalar()

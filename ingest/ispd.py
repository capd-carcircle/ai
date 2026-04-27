"""
ISPD PDF 인제스트 스크립트

ai/ingest/ 폴더 안의 ISPD PDF 파일을 청킹·임베딩하여 document_chunks에 저장.
retriever.py가 KDIGO 청크와 함께 자동으로 검색함 (source 태그로 구분).

대상 파일:
  - li-et-al-2022-ispd-peritonitis-guideline-...pdf      (복막염 예방·치료)
  - morelle-et-al-2021-ispd-recommendations-...pdf       (복막막 기능·UF 기준)

사용법:
  # EC2 서버 (ai/ 디렉토리에서)
  cd ~/capd && python -m ai.ingest.ispd

  # 기존 청크 삭제 후 재처리
  cd ~/capd && python -m ai.ingest.ispd --clear
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 .env 로드 (ai/ingest/ispd.py → ai/ingest/ → ai/ → capd/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from pypdf import PdfReader

from ai.ingest.base import (
    chunk_text,
    count_chunks_by_source,
    delete_chunks_by_source,
    get_db_session,
    get_embedding_model,
    save_chunks,
)

# ISPD PDF가 위치한 폴더 (이 스크립트와 같은 위치: ai/ingest/)
INGEST_DIR = Path(__file__).resolve().parent


def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """PDF에서 페이지별 텍스트 추출. [(page_num, text), ...]"""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def ingest(clear: bool = False):
    # ai/ingest/ 안의 ISPD 관련 PDF만 선택
    pdf_files = sorted(
        f for f in INGEST_DIR.glob("*.pdf") if "ispd" in f.name.lower()
    )
    if not pdf_files:
        logger.error(f"ISPD PDF 파일을 찾을 수 없습니다: {INGEST_DIR}")
        logger.error("파일명에 'ispd'가 포함되어야 합니다.")
        sys.exit(1)

    logger.info(f"ISPD PDF {len(pdf_files)}개 발견: {[f.name for f in pdf_files]}")
    model = get_embedding_model()
    db = get_db_session()
    total_saved = 0

    try:
        for pdf_path in pdf_files:
            source = pdf_path.name
            logger.info(f"\n[{source}] 처리 시작")

            existing = count_chunks_by_source(db, source)
            if existing > 0 and not clear:
                logger.info(f"  → 이미 인제스트됨 ({existing}개 청크). 스킵. (재처리: --clear)")
                continue

            if clear and existing > 0:
                deleted = delete_chunks_by_source(db, source)
                logger.info(f"  → 기존 {deleted}개 청크 삭제")

            pages = extract_text_by_page(pdf_path)
            logger.info(f"  → {len(pages)}개 페이지 추출")

            chunks_with_pages: list[tuple[int, str]] = []
            for page_num, page_text in pages:
                for chunk in chunk_text(page_text):
                    chunks_with_pages.append((page_num, chunk))

            logger.info(f"  → {len(chunks_with_pages)}개 청크 생성, 임베딩 시작")
            saved = save_chunks(db, source, chunks_with_pages, model)
            logger.info(f"  → {saved}개 청크 저장 완료")
            total_saved += saved

    finally:
        db.close()

    logger.info(f"\n✅ ISPD 인제스트 완료: 총 {total_saved}개 청크 저장")


if __name__ == "__main__":
    ingest(clear="--clear" in sys.argv)

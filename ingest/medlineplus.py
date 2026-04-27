"""
MedlinePlus 인제스트 스크립트 (CAPD 관련 토픽)

MedlinePlus Web Service API를 호출해 CAPD 관련 건강 토픽 텍스트를 가져오고,
청킹·임베딩 후 document_chunks에 저장.
SHA256 해시 비교로 변경된 토픽만 재인제스트 (불필요한 재처리 방지).

소스 태그: 'medlineplus_{topic_key}' (예: medlineplus_peritoneal_dialysis)

사용법:
  # 변경된 토픽만 업데이트 (기본)
  cd ~/capd && python -m ai.ingest.medlineplus

  # 전체 강제 재인제스트
  cd ~/capd && python -m ai.ingest.medlineplus --force

EC2 cron 등록 (매주 월요일 새벽 3시):
  crontab -e
  0 3 * * 1  cd ~/capd && python -m ai.ingest.medlineplus >> ~/capd/logs/medlineplus_ingest.log 2>&1
"""

import hashlib
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from ai.ingest.base import (
    chunk_text,
    count_chunks_by_source,
    delete_chunks_by_source,
    get_db_session,
    get_embedding_model,
    save_chunks,
)

# CAPD 관련 검색 토픽
TOPICS: dict[str, str] = {
    "peritoneal_dialysis": "peritoneal dialysis",
    "peritonitis": "peritonitis",
    "end_stage_kidney": "end-stage kidney disease",
    "ultrafiltration": "ultrafiltration kidney",
    "dialysis_complications": "dialysis complications",
}

MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"

# 해시 저장 파일 (ai/ingest/medlineplus_hashes.json)
HASH_FILE = Path(__file__).resolve().parent / "medlineplus_hashes.json"


def fetch_topic(term: str) -> str | None:
    """MedlinePlus API 호출 → 텍스트 추출 (HTML 태그 제거)."""
    try:
        resp = requests.get(
            MEDLINEPLUS_API,
            params={"db": "healthTopics", "term": term, "rettype": "brief"},
            timeout=30,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        texts = []
        for doc in root.findall(".//document"):
            for content in doc.findall("content"):
                name = content.get("name", "")
                if name in ("title", "FullSummary", "snippet"):
                    raw = content.text or ""
                    # HTML 태그 제거
                    clean = re.sub(r"<[^>]+>", " ", raw)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    if clean:
                        texts.append(clean)

        return "\n\n".join(texts) if texts else None

    except Exception as e:
        logger.warning(f"MedlinePlus API 호출 실패 ({term}): {e}")
        return None


def load_hashes() -> dict:
    if HASH_FILE.exists():
        return json.loads(HASH_FILE.read_text(encoding="utf-8"))
    return {}


def save_hashes(hashes: dict):
    HASH_FILE.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")


def ingest(force: bool = False):
    model = get_embedding_model()
    db = get_db_session()
    hashes = load_hashes()
    total_saved = 0

    try:
        for topic_key, term in TOPICS.items():
            source = f"medlineplus_{topic_key}"
            logger.info(f"\n[{source}] API 호출: '{term}'")

            text_content = fetch_topic(term)
            if not text_content:
                logger.warning(f"  → 내용 없음, 스킵")
                continue

            content_hash = hashlib.sha256(text_content.encode()).hexdigest()

            if not force and hashes.get(topic_key) == content_hash:
                logger.info(f"  → 변경 없음 (해시 동일), 스킵")
                continue

            # 기존 청크 삭제 후 재인제스트
            existing = count_chunks_by_source(db, source)
            if existing > 0:
                deleted = delete_chunks_by_source(db, source)
                logger.info(f"  → 기존 {deleted}개 청크 삭제")

            chunks_with_pages = [(None, c) for c in chunk_text(text_content)]
            logger.info(f"  → {len(chunks_with_pages)}개 청크 생성, 임베딩 시작")

            saved = save_chunks(db, source, chunks_with_pages, model)

            # 성공 시 해시 저장
            hashes[topic_key] = content_hash
            save_hashes(hashes)

            logger.info(f"  → {saved}개 청크 저장 완료")
            total_saved += saved

    finally:
        db.close()

    logger.info(f"\n✅ MedlinePlus 인제스트 완료: 총 {total_saved}개 청크 저장/업데이트")


if __name__ == "__main__":
    ingest(force="--force" in sys.argv)

"""30개사 조사 보고서 기반 하이브리드 검색(RAG)을 제공합니다.

팀에서 미리 만든 임베딩 저장소(rag_store/)를 그대로 사용합니다.
- 임베딩: BAAI/bge-m3 (FlagEmbedding, Dense + Sparse)
- Dense 검색: FAISS IndexFlatIP
- 메타데이터·Sparse 저장: SQLite
- 결합: RRF(k=60)
"""

import sys
import threading
from functools import lru_cache

from config import RAG_STORE_DIR, RETRIEVE_K

sys.path.insert(0, str(RAG_STORE_DIR))

DB_PATH = RAG_STORE_DIR / "data" / "vector_store.sqlite"

# 병렬 노드(기술·시장·매출)가 모델을 동시에 불러오거나 인코딩하지 않도록 잠금
_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    """bge-m3 모델을 한 번만 불러옵니다 (첫 실행 시 모델 다운로드)."""
    from embed import load_model  # rag_store/embed.py (FlagEmbedding)

    print("bge-m3 임베딩 모델 로딩...")
    return load_model()


def retrieve(query: str, company_id: str | None = None, agent: str | None = None, k: int = RETRIEVE_K) -> list[dict]:
    """질문으로 하이브리드 검색합니다.

    Args:
        query: 검색 질문
        company_id: 기업 코드 (예: "DX"). 주면 해당 기업 청크만 검색
        agent: 에이전트 경로 ("technology", "market", "financial", "investment", "SWOT")
    """
    from search import search as hybrid_search  # rag_store/search.py

    with _lock:
        model = get_model()
        results = hybrid_search(DB_PATH, model, query, top_k=k, company=company_id, agent=agent)
        if not results and agent:
            # 에이전트 경로 필터로 결과가 없으면 경로 필터 없이 다시 검색
            results = hybrid_search(DB_PATH, model, query, top_k=k, company=company_id)
    return results


def format_docs(results: list[dict]) -> str:
    """검색 결과를 XML 형식의 문자열로 포매팅합니다."""
    lines = []
    for r in results:
        section = " > ".join(r["metadata"].get("section_path", []))
        lines.append(
            f"<document>"
            f"<content>{r['text']}</content>"
            f"<source>AI반도체_스타트업_30사_기업평가_자료.md / {section}</source>"
            f"</document>"
        )
    return "\n".join(lines)

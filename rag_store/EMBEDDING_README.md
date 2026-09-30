# AI 반도체 스타트업 30개사 RAG 데이터

2026-09-30 처리 완료. 기업 정보의 조사 기준일은 원문에 기재된 **2026-09-29**입니다. 기업의 외부 출처를 이번 작업에서 모두 재검증한 것은 아닙니다.

첨부한 기업 자료 Markdown을 기업명과 특징 태그가 있는 key-value JSON으로 변환하고, **실제 BAAI/bge-m3 모델로 Dense와 Sparse를 모두 계산**했습니다. Notion 프로젝트 PDF 3~4페이지의 한국어·영어 혼합 검색, 고유명사·수치 검색, 로컬 실행 요구사항을 반영했습니다. 프로젝트 PDF 자체는 기업 근거 임베딩에 포함하지 않았습니다.

## 결과

| 항목 | 결과 |
|---|---|
| 기업 | 30개사 |
| 원문 세그먼트 | A 엣지 6 / B 데이터센터 11 / C 인메모리 6 / D 인프라 7 |
| 기업 상세 정보 단위 | 678개 |
| 비교표·공통 규칙 등 문서 문맥 단위 | 245개 |
| 전체 임베딩 단위 | 923개 |
| 모델 | BAAI/bge-m3 |
| 모델 revision | 5617a9f61b028005a4858fdac845db406aefb181 |
| Dense | 923 × 1,024, float32, L2 정규화 |
| Sparse | BGE-M3의 학습된 토큰 ID별 lexical weight, 전체 비영점 51,210개 |
| Dense 검색 | FAISS IndexFlatIP |
| 메타데이터·Sparse 저장 | SQLite |
| 하이브리드 결합 | Dense 코사인 + Sparse 내적의 순위에 RRF(k=60) 적용 |
| 입력 길이 | 최대 286토큰, 실행 한도 512토큰, 잘림 0개 |
| 검증 | 검색 사례 8/8, FAISS·SQLite·NumPy 행 정합성, 재무 의미 보존 통과 |

ColBERT multi-vector와 별도 reranker는 이번 범위에 포함하지 않았습니다. FAISS는 벡터 검색 라이브러리이며, 이 결과의 메타데이터 보관과 필터 처리는 SQLite가 담당합니다. 현재 구현은 소규모 정확 검색용입니다. 검색 사례 8건은 기능 점검이며 일반적인 검색 정확도나 투자 평가 정확도를 입증하는 벤치마크는 아닙니다.

## 파일

| 파일 | 사용 용도 |
|---|---|
| `data/companies.json` | `companies[기업명]`으로 찾는 기업별 key-value 데이터 |
| `company_tags.json` | 기업 코드·별칭·세그먼트·사업 모델·제품·기술 특징 태그 |
| `data/chunks.json` | `chunks[chunk_id]`로 찾는 정보 단위 JSON |
| `data/chunks.jsonl` | 같은 청크를 한 줄 한 레코드로 저장한 임베딩 입력 |
| `data/dense.faiss` | 저장 완료된 FAISS Dense 인덱스 |
| `data/dense_vectors.npy` | 원본 Dense 벡터 행렬 |
| `data/vector_ids.json` | 벡터 행 번호 → chunk_id 매핑 |
| `data/sparse_vectors.jsonl` | 청크별 Sparse 토큰 가중치 |
| `data/embeddings.jsonl` | 실제 Dense·Sparse 값과 payload를 담은 범용 이관용 JSONL |
| `data/vector_store.sqlite` | 텍스트·기업 태그·출처·Dense·Sparse 저장소 |
| `data/sources.json` | 원문 참고 자료 177건, URL·원문 등급·원문 위치 |
| `data/revenue_records.json` | 상세 재무 표의 매출 29건을 정규화한 보조 데이터 |
| `data/embedding_manifest.json` | 모델 버전·차원·정규화·입력 해시·실행 환경 |
| `data/validation.json` | 기업 수·원문 내용 보존·출처 연결 점검 |
| `data/retrieval_checks.json` | 실제 검색 질문 8개와 검색 결과 |
| `prepare_data.py` | 원문 Markdown → 구조화 JSON |
| `embed.py` | 공식 FlagEmbedding으로 임베딩·SQLite·FAISS 생성 |
| `build_faiss.py` | 저장된 Dense 벡터로 FAISS만 재생성 |
| `search.py` | 메타데이터 필터와 하이브리드 검색 |
| `verify_retrieval.py` | 검색과 데이터 정합성 점검 |
| `source/company_evaluation.md` | 재실행을 위한 원문 사본 |

`embeddings.jsonl`은 특정 서버 API에 바로 보낼 수 있는 요청 형식이 아닙니다. 다른 벡터 DB로 이관하려면 해당 DB의 ID·Sparse·payload 형식에 맞춰 변환하십시오. 현재 실행 가능한 검색 저장소는 FAISS와 SQLite입니다.

## JSON 구조

```python
import json
companies = json.load(open('data/companies.json', encoding='utf-8'))['companies']
deepx = companies['딥엑스']
print(deepx['company_id'])        # DX
print(deepx['segment'])           # A
print(deepx['features'])          # edge_ai, NPU, INT8, LPDDR4X, LPDDR5
print(deepx['facts']['technology'])

chunks = json.load(open('data/chunks.json', encoding='utf-8'))['chunks']
chunk = chunks[deepx['chunk_ids'][0]]
print(chunk['key_value'])
print(chunk['embedding_text'])
print(chunk['metadata']['citation_ids'])
```

기업별 JSON에는 `aliases`, `country`, `segment`, `business_models`, `products`, `features`, `eligibility_flags`, `facts`, `chunk_ids`가 있습니다. `facts`는 개요·기술·검증·투자·재무·상용화·리스크·IP 소유권별 리스트입니다. 회사 한 개에 벡터 한 개를 배정하지 않았습니다.

청크는 독립된 bullet 사실 또는 표의 한 행을 기본 단위로 사용합니다. 하위 bullet에는 상위 제품·주장 문맥을 붙였습니다. 재무 표 행에는 연도·매출·손실·비고를 함께 보존합니다. 공통 비교표는 기업 상세 청크와 `scope`로 구분하며, 한 행에 여러 회사가 있으면 `company_ids`에 연결합니다. 빈 Fact Sheet 양식은 `kind=template`으로 명시됩니다.

메타데이터에는 회사명·코드·국가·세그먼트·특징·제품·항목·에이전트 경로·출처 번호·출처 등급·원문 행 범위·품질 표시가 있습니다. 특징과 사업 모델 태그는 **원문에서 도출한 검색 분류**입니다. 확정 사실 여부는 청크의 원문과 출처를 확인해야 합니다. d-Matrix는 원문대로 B에 두면서 `digital_IMC` 특징으로 C 계열과 교차 검색할 수 있습니다.

## 데이터 해석

- `source_grades`는 원문이 정한 출처 종류입니다. 회사 주장이나 특허를 독립 성능 검증으로 바꾸지 않았습니다.
- `explicit_citation_ids`는 해당 사실에 직접 붙은 인용이고, `context_citation_ids`는 제목·상위 문맥에서 물려받은 인용입니다. 일부 감사 표의 출처는 같은 절의 감사보고서 설명에서 연결했습니다. 문맥 인용이 행의 모든 값에 대한 직접 검증을 뜻하지는 않습니다.
- `literal_quality_markers`는 미공개·목표·회사 주장·미검증 같은 원문 표현을 탐지한 비배타적 표시입니다. 자동 진실 판정이나 투자 점수가 아닙니다.
- `revenue_records.json`은 상세 재무 **표**의 매출을 정규화한 보조 자료입니다. 표 밖의 매출 설명과 투자금·자산·손익은 기업별 원문 facts에 보존되어 있습니다. 모든 회사의 완성된 재무 데이터셋은 아닙니다.
- 미공개 매출과 원문 `매출 없음`은 보조 데이터에서 `null`로 보수적으로 보존합니다. 명시된 `0원`은 0입니다.
- Groq 2025년 $500M은 `forecast`이며 실적과 같은 성장률 계산에 사용하지 않습니다. 딥엑스 2024년 원래 수치와 K-IFRS 재작성 수치는 별도 레코드입니다. 비교 수치의 감사 여부가 불명확하면 감사 완료로 처리하지 않습니다.
- 재무 표에서 반올림한 금액은 그대로 표시하고, 보스반도체·리벨리온·퓨리오사AI의 원문 주석에 정확한 원 단위 금액이 있으면 연결된 주석 청크와 함께 우선 사용합니다.
- 원문 외부 링크는 이번 처리에서 일괄 재열람하지 않았으므로 `external_sources_verified=false`, `verified_at=null`입니다. 원문의 조사 기준일과 처리 날짜를 구분했습니다.

## 실행

Python 3.12의 CPU 환경에서 검증했습니다. **이미 생성된 데이터·인덱스를 검색만 하려면 다시 임베딩할 필요가 없습니다.** 모델 가중치는 용량 때문에 압축 파일에 포함하지 않았습니다. 최초 검색 실행 시 아래 고정 revision을 다운로드합니다.

```bash
cd ai_semiconductor_rag
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 기존 결과로 검색: 회사 코드와 정보 유형으로 필터
python search.py "딥엑스 DX-M1 정밀도와 연산 성능" --company DX --category technology
python search.py "Groq 2025년 매출은 실제 실적인가요" --company GRQ --category financial
python search.py "인메모리 연산과 메모리 구조" --feature digital_IMC
python search.py "CXL 실리콘 검증 근거" --segment D --agent technology

# 원문에서 전부 재생성
python prepare_data.py source/company_evaluation.md
python embed.py --batch-size 8 --threads 4
python verify_retrieval.py
```

로컬 모델 사본을 이미 보유한 경우 `embed.py`, `search.py`, `verify_retrieval.py`의 `--model-path`에 BGE-M3 경로를 지정할 수 있습니다. 질의에도 문서와 동일한 모델·revision을 사용하며 별도 질의 instruction prefix를 붙이지 않습니다. 512토큰을 넘는 입력을 재생성할 때는 모델의 8,192토큰 한도 안에서 자동으로 실행 길이를 늘리고, 한도를 넘으면 중단합니다. 조용히 잘라내지 않습니다.

RRF 점수는 확률이나 투자 적합도 점수가 아닙니다. 기술 에이전트는 `--agent technology`, 재무 에이전트는 `--category financial` 등으로 검색 범위를 줄이고, 별도 조회로 공통 해석 규칙과 적격성 플래그를 함께 확인하면 됩니다. 시장 규모·CAGR·산업 리포트는 이 기업 자료만으로 충분하지 않아 프로젝트의 추가 RAG 자료로 보강해야 합니다.

## 출처

1. 사용자 첨부 「AI 반도체 비상장 스타트업 30개사 기업 평가 자료집」, 조사 기준일 2026-09-29. 각 원문 인용은 `data/sources.json`에서 확인할 수 있습니다.
2. 사용자 첨부 「SKALA 1반 3조 Team PJT」 PDF, 3~4페이지의 모델·검색 요구사항.
3. BAAI 공식 모델 카드: https://huggingface.co/BAAI/bge-m3
4. 공식 FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding
5. FAISS 인덱스 설명: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
6. FAISS 내적·코사인 설명: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances

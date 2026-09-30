"""반도체 스타트업 투자 평가 에이전트의 설정값을 정의합니다."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# 경로
RAG_STORE_DIR = BASE_DIR / "rag_store"  # 팀 임베딩 저장소 (bge-m3 + FAISS + SQLite)
CACHE_DIR = BASE_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "outputs"

# 모델
MODEL_NAME = "gpt-4.1-mini"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 오픈소스 임베딩 (설계서 B. Embedding 모델)

# 검색
RETRIEVE_K = 6
NEWS_MAX_RESULTS = 5

# 평가 대상 30개사 (설계서 기업 리스트). id = rag_store 기업 코드
COMPANIES = [
    {"id": "HA", "name": "HyperAccel", "name_ko": "하이퍼엑셀", "country": "한국"},
    {"id": "RBL", "name": "Rebellions", "name_ko": "리벨리온", "country": "한국"},
    {"id": "FUR", "name": "FuriosaAI", "name_ko": "퓨리오사AI", "country": "한국"},
    {"id": "DM", "name": "d-Matrix", "name_ko": "d-Matrix", "country": "미국"},
    {"id": "ETC", "name": "Etched", "name_ko": "Etched", "country": "미국"},
    {"id": "POS", "name": "Positron AI", "name_ko": "Positron AI", "country": "미국"},
    {"id": "MOB", "name": "Mobilint", "name_ko": "모빌린트", "country": "한국"},
    {"id": "DX", "name": "DEEPX", "name_ko": "딥엑스", "country": "한국"},
    {"id": "BOS", "name": "BOS Semiconductors", "name_ko": "보스반도체", "country": "한국"},
    {"id": "SIMA", "name": "SiMa.ai", "name_ko": "SiMa.ai", "country": "미국"},
    {"id": "QD", "name": "Quadric", "name_ko": "Quadric", "country": "미국"},
    {"id": "MX", "name": "MemryX", "name_ko": "MemryX", "country": "미국"},
    {"id": "IHW", "name": "iHW", "name_ko": "아이에이치더블유", "country": "한국"},
    {"id": "PBS", "name": "Pebble Square", "name_ko": "페블스퀘어", "country": "한국"},
    {"id": "ART", "name": "Articron", "name_ko": "아티크론", "country": "한국"},
    {"id": "ENC", "name": "EnCharge AI", "name_ko": "EnCharge AI", "country": "미국"},
    {"id": "RAIN", "name": "Rain AI", "name_ko": "Rain AI", "country": "미국"},
    {"id": "MYT", "name": "Mythic", "name_ko": "Mythic", "country": "미국"},
    {"id": "PAN", "name": "Panmnesia", "name_ko": "파네시아", "country": "한국"},
    {"id": "XC", "name": "XCENA", "name_ko": "엑시나", "country": "한국"},
    {"id": "MB", "name": "MangoBoost", "name_ko": "망고부스트", "country": "한국"},
    {"id": "LM", "name": "Lightmatter", "name_ko": "Lightmatter", "country": "미국"},
    {"id": "AYR", "name": "Ayar Labs", "name_ko": "Ayar Labs", "country": "미국"},
    {"id": "ELY", "name": "Eliyan", "name_ko": "Eliyan", "country": "미국"},
    {"id": "GRQ", "name": "Groq", "name_ko": "Groq", "country": "미국"},
    {"id": "SN", "name": "SambaNova Systems", "name_ko": "SambaNova", "country": "미국"},
    {"id": "TT", "name": "Tenstorrent", "name_ko": "Tenstorrent", "country": "미국"},
    {"id": "MATX", "name": "MatX", "name_ko": "MatX", "country": "미국"},
    {"id": "OXQ", "name": "OXMIQ Labs", "name_ko": "OXMIQ", "country": "미국"},
    {"id": "RIV", "name": "Rivos", "name_ko": "Rivos", "country": "미국"},
]

# 투자평가 프레임 (설계서 C. 투자 판단 기준)
AREA_WEIGHTS = {"tech": 40, "market": 40, "revenue": 20}

# 기술 평가 (등급 판정 4개 + 핵심 인력 O/X)
TECH_ITEMS = {
    "core_technology": 14,     # Core Technology / Architecture (차별성)
    "manufacturability": 14,   # Manufacturability / Implementation (구현 가능성)
    "validation": 4,           # Validation / Reliability (검증 여부)
    "scalability": 4,          # System Integration / Scalability (신뢰성)
}
KEY_PERSON_POINTS = 4          # 핵심 인력 (O/X)

# 시장 평가 (실무가이드 부록 1)
MARKET_ITEMS = {"growth": 20, "competition": 10, "entry": 10}

# 투자 판단 기준 (구현_결정사항.md 2절)
INITIAL_TOTAL_THRESHOLD = 60
INITIAL_CUTOFF_RATIO = 0.40
RELAX_TOTAL_STEP = 5
RELAX_CUTOFF_STEP = 0.05
TOP_N = 3

# 수익률 계산 (구현_결정사항.md 4절)
DISCOUNT_RATE = 0.20  # 실무가이드: 비상장 자기자본비용 통상 10~25%
DILUTION_BY_ROUND = {  # Carta (2025-12-01) 라운드별 중간 희석률
    "seed": 0.195,
    "series_a": 0.18,
    "series_b": 0.14,
    "series_c": 0.10,
    "series_d": 0.075,
}
LARGE_CAP_LIMIT_USD = 100e9  # 비교기업에서 제외할 대기업 기준 (시가총액)
PEER_TICKERS = [  # 상장 반도체 설계·IP 기업 후보 (대기업은 시가총액으로 자동 제외)
    "AMBA", "CRDO", "RMBS", "LSCC", "ALAB", "SITM", "SLAB", "SMTC",
    "CEVA", "MXL", "POWI", "SYNA", "MPWR", "QRVO", "NVDA", "AVGO", "QCOM", "AMD",
]
DEFAULT_USD_KRW = 1400.0  # 환율 조회 실패 시 사용하는 기본값

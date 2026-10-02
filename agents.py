"""투자 평가 그래프의 노드(에이전트)를 정의합니다."""

import json
import re
import statistics
from datetime import date

from langchain.chat_models import init_chat_model

from config import (
    AREA_WEIGHTS,
    DILUTION_BY_ROUND,
    DISCOUNT_RATE,
    KEY_PERSON_POINTS,
    MARKET_ITEMS,
    MODEL_NAME,
    OUTPUT_DIR,
    RAG_STORE_DIR,
    TEAM_FILE_TAG,
    RELAX_CUTOFF_STEP,
    RELAX_TOTAL_STEP,
    TECH_ITEMS,
    TOP_N,
)
from charts import draw_summary_chart
from pdf import markdown_to_pdf
from rag import format_docs, retrieve
from state import (
    SWOT,
    Amount,
    GradedItem,
    GraphState,
    MarketAssessment,
    ReturnInputs,
    RevenueFacts,
    TechAssessment,
)
from tools import format_news, get_peer_multiples, get_usd_krw, search_news


def load_segments() -> dict:
    """기업 코드별 세그먼트 이름을 불러옵니다 (rag_store/company_tags.json)."""
    tags = json.loads((RAG_STORE_DIR / "company_tags.json").read_text(encoding="utf-8"))
    companies = json.loads((RAG_STORE_DIR / "data" / "companies.json").read_text(encoding="utf-8"))["companies"]
    labels = {v["company_id"]: v.get("segment_label", "AI 반도체") for v in companies.values()}
    return {cid: labels.get(cid, "AI 반도체") for cid in tags}


def load_features() -> dict:
    """기업 코드별 기술 특징 태그를 불러옵니다 (시장 뉴스 검색어에 사용)."""
    tags = json.loads((RAG_STORE_DIR / "company_tags.json").read_text(encoding="utf-8"))
    return {cid: " ".join(t.replace("_", " ") for t in v.get("features", [])[:3]) for cid, v in tags.items()}


SEGMENTS = load_segments()
FEATURES = load_features()


def get_llm(schema=None):
    """LLM을 생성합니다. schema를 주면 구조화 출력으로 반환합니다."""
    llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
    if schema is None:
        return llm
    return llm.with_structured_output(schema, method="json_schema", strict=True)


def graded_points(item: GradedItem, max_points: float) -> float:
    """5단계 등급 점수를 배점에 맞게 환산합니다.

    지침: 평가할 자료가 있으면 최소 1점, 자료가 없으면 0점.
    """
    if item.grade == "none":
        return 0.0
    points = round(item.score_5 / 5 * max_points, 1)
    return max(points, 1.0)


GRADE_SCALE = "등급 점수(5점 척도, 0.5 단위): a=4.5~5, b=3.5~4, c=2.5~3, d=1.5~2, e=0.5~1, 자료 없음=none(0점)"

SOURCE_RULE = """- 제공된 자료에 있는 사실만 사용하세요. 자료가 일부라도 있으면 그 범위에서 등급을 판정하고, 관련 자료가 전혀 없을 때만 none으로 판정하세요.
- 회사 주장 수치와 제3자 검증 수치를 구분하세요.
- 다른 회사의 제품·실적을 이 회사 것으로 쓰지 마세요. 이 회사 기술을 라이선스·인수한 회사의 제품 양산은 이 회사의 양산이 아닙니다.
- sources에는 실제로 근거로 쓴 문서 섹션 또는 뉴스 URL만 적으세요."""


# ---------- 기업 루프 ----------

def select_company(state: GraphState) -> GraphState:
    """선정 30개사 중 현재 순서의 기업을 선택합니다."""
    idx = state["current_index"]
    company = state["companies"][idx]
    print(f"\n[{idx + 1}/{len(state['companies'])}] {company['name_ko']} ({company['name']}) 평가 시작")
    return {"company": company}


def tech_agent(state: GraphState) -> GraphState:
    """기술 평가 에이전트 (RAG + 뉴스): 기술 40점."""
    c = state["company"]
    docs = retrieve(f"{c['name_ko']} {c['name']} 핵심 기술 제품 공정 성능 검증 양산 창업자 핵심 인력", c["id"], "technology")
    news = search_news(f"{c['name_ko']} 반도체 기술")

    prompt = f"""당신은 반도체 스타트업 기술 실사(Technical Due Diligence) 전문가입니다.
아래 자료로 {c['name_ko']}({c['name']})의 기술력을 평가하세요.

{GRADE_SCALE}

평가 항목과 등급 기준:
1. core_technology (핵심 기술의 구조적 차별성·성능·전력·면적 효율, 해결하는 기술 문제)
   a=독자 아키텍처로 경쟁 기술 대비 차별성이 매우 강함 / b=차별성 강함 / c=경쟁 기술과 유사 / d=차별성 미흡 / e=차별성 낮음
2. manufacturability (실제 반도체로 구현·제조 가능한 수준)
   a=양산·출하 중 / b=샘플·파일럿 공급 / c=테이프아웃·첫 실리콘 / d=FPGA·시제품 / e=설계·시뮬레이션 단계
   (양산은 자료에 양산·출하 사실이 명시된 경우만 인정합니다. 계획·예정·목표는 양산이 아닙니다.)
3. validation (실제 silicon과 검증자료로 입증)
   a=MLPerf·독립기관 실측 등 제3자 검증 / b=주요 학회(Hot Chips, ISSCC 등) 발표 / c=고객 PoC 결과 공개 / d=회사 자체 수치만 / e=검증 자료 없음
4. scalability (기존 시스템과 통합·확장 가능성)
   a=표준 인터페이스·주요 프레임워크 지원과 확장 제품군 / b=표준 지원 / c=일부 지원 / d=제한적 / e=통합 어려움
5. key_person: 창업자·CTO 등 핵심 인물이 확인되면 true

규칙:
{SOURCE_RULE}

# 조사 보고서 (RAG)
{format_docs(docs)}

# 뉴스
{format_news(news)}
"""
    result: TechAssessment = get_llm(TechAssessment).invoke(prompt)

    items = {key: graded_points(getattr(result, key), pts) for key, pts in TECH_ITEMS.items()}
    items["key_person"] = KEY_PERSON_POINTS if result.key_person else 0
    return {
        "tech_result": {
            "score": round(sum(items.values()), 1),
            "items": items,
            "detail": result.model_dump(),
            "sources": result.sources,
            "news": news,
        }
    }


def market_agent(state: GraphState) -> GraphState:
    """시장 평가 에이전트 (RAG + 뉴스): 시장 40점 (실무가이드 부록 1)."""
    c = state["company"]
    docs = retrieve(f"{c['name_ko']} {c['name']} 목표 시장 시장 규모 성장 경쟁 세그먼트", c["id"], "market")
    segment = SEGMENTS.get(c["id"], "AI 반도체")
    news = (
        search_news(f"{segment} 시장 규모 전망 성장률")
        + search_news(f"{FEATURES.get(c['id'], 'AI chip')} market size forecast CAGR", lang="en")
        + search_news(f"{c['name_ko']} 경쟁 시장")
    )

    prompt = f"""당신은 반도체 시장 분석가입니다. {c['name_ko']}({c['name']})가 속한 목표 시장을 평가하세요.

절차:
1. 목표 시장 정의: 이 기업의 제품이 적용될 시장을 정합니다.
2. 자료 수집: 목표 시장의 규모, 성장 추세, 경쟁 구조, 진입 요인을 자료에서 찾습니다.
3. 항목별 판정 (기술가치평가 실무가이드 부록 1 기준)

{GRADE_SCALE}

growth (시장 성장성)
 a=최근 고도 성장기, 향후 성장추세 매우 높음 / b=지속적 성장, 장기 지속 예측 / c=완만한 성장 / d=최근 정체, 성장 불확실 / e=지속 감소
competition (시장 경쟁성)
 a=경쟁기업·제품 거의 없음 / b=소수, 선도업체 없음 / c=다수, 선도 기업들이 시장 분할 / d=강력한 경쟁자가 과점 / e=강력한 경쟁자가 독점
entry (시장 진입성)
 a=진입장벽 낮고 법·제도 장려요인 있음 / b=진입장벽 낮고 장려요인 생길 가능성 / c=진입장벽 높지 않고 장려·제약 없음 / d=규모의 경제·비용·영업망·제도 중 한 요소로 진입장벽 매우 높음 / e=복합 제약으로 진입장벽 매우 높음

시장 규모(market_size)는 채점하지 않지만 보고서에 쓰므로 수치·연도·조사기관을 함께 적으세요.
- 자료에 연평균 성장률(CAGR)이 있으면 market_cagr_pct에 반드시 숫자로 적으세요.
- 기준 연도와 전망 연도의 시장 규모가 있으면 size_base_*, size_future_*에 달러 금액으로 적으세요 (1 billion = 1e9).
- 뉴스 제목·요약에 "$45 billion by 2030"처럼 시장 규모 수치가 있으면 반드시 추출하세요. 수치가 하나도 없으면 market_size에 "확인 안 됨"이라고만 적고, "뉴스 참조" 같은 표현은 쓰지 마세요.
이 기업의 세그먼트: {segment}
투자 회수 희망 시점: {state['investment_years']}년 후 (성장성 판단에 고려)

규칙:
{SOURCE_RULE}

# 조사 보고서 (RAG)
{format_docs(docs)}

# 뉴스
{format_news(news)}
"""
    result: MarketAssessment = get_llm(MarketAssessment).invoke(prompt)

    # 성장률이 명시되지 않았으면 두 시점의 시장 규모로 계산
    detail = result.model_dump()
    if detail["market_cagr_pct"] is None and all(
        detail[k] for k in ("size_base_usd", "size_base_year", "size_future_usd", "size_future_year")
    ) and detail["size_future_year"] > detail["size_base_year"]:
        span = detail["size_future_year"] - detail["size_base_year"]
        detail["market_cagr_pct"] = round(((detail["size_future_usd"] / detail["size_base_usd"]) ** (1 / span) - 1) * 100, 1)
        detail["market_cagr_basis"] = "두 시점 시장 규모로 계산"

    items = {key: graded_points(getattr(result, key), pts) for key, pts in MARKET_ITEMS.items()}
    return {
        "market_result": {
            "score": round(sum(items.values()), 1),
            "items": items,
            "detail": detail,
            "sources": result.sources,
            "news": news,
        }
    }


def revenue_points(facts: RevenueFacts, usd_krw: float) -> dict:
    """매출 사실을 구간표로 점수화합니다 (구현_결정사항.md 3절)."""
    revenue = facts.latest_revenue_krw
    if revenue is None and facts.latest_revenue_usd is not None:
        revenue = facts.latest_revenue_usd * usd_krw

    if revenue is None:
        current = 0
    elif revenue >= 100e8:
        current = 10
    elif revenue >= 30e8:
        current = 7
    elif revenue >= 10e8:
        current = 5
    elif revenue >= 1e8:
        current = 3
    else:
        current = 1

    cagr = facts.revenue_cagr_3y_pct
    if cagr is None:
        growth = 0
    elif cagr >= 100:
        growth = 5
    elif cagr >= 50:
        growth = 4
    elif cagr >= 0:
        growth = 3
    else:
        growth = 1

    quality = {"product": 5, "mixed": 3, "service": 1, "unknown": 0}[facts.revenue_quality]
    return {"current": current, "growth": growth, "quality": quality, "revenue_krw": revenue}


def revenue_agent(state: GraphState) -> GraphState:
    """매출 평가 에이전트 (RAG + 뉴스): 매출 20점."""
    c = state["company"]
    docs = retrieve(f"{c['name_ko']} {c['name']} 매출 영업손실 감사보고서 재무", c["id"], "financial")
    news = search_news(f"{c['name_ko']} 매출")

    prompt = f"""당신은 재무 분석가입니다. {c['name_ko']}({c['name']})의 매출 정보를 추출하세요.

- 가장 최근 연매출과 연도, 최근 3개년 매출 연평균 증가율을 찾으세요. 계산에 필요한 연도 값이 없으면 null.
- 데이터 성격(공시/보도/2차DB/회사발표/미공개)을 구분하세요. "미공개"는 0원이 아닙니다 → null.
- 투자금, 수주 잔고·약정, 기업가치, 목표·전망 매출은 매출이 아닙니다. 매출로 적지 마세요.
- 조사 보고서(RAG)를 우선합니다. 조사 보고서가 "미공개"라고 한 기업은 뉴스에 매출처럼 보이는 숫자가 있어도 null로 두세요.
- 뉴스는 조사 보고서의 매출을 보강할 때만 사용하세요.
- revenue_quality: 목표 사업 제품 매출이면 product, 제품+용역 혼재면 mixed, 용역·과제 위주면 service, 자료 없으면 unknown.

규칙:
{SOURCE_RULE}

# 조사 보고서 (RAG)
{format_docs(docs)}

# 뉴스
{format_news(news)}
"""
    facts: RevenueFacts = get_llm(RevenueFacts).invoke(prompt)
    points = revenue_points(facts, get_usd_krw())
    return {
        "revenue_result": {
            "score": points["current"] + points["growth"] + points["quality"],
            "items": {k: points[k] for k in ("current", "growth", "quality")},
            "detail": facts.model_dump() | {"revenue_krw_normalized": points["revenue_krw"]},
            "sources": facts.sources,
            "news": news,
        }
    }


def record_evaluation(state: GraphState) -> GraphState:
    """세 분석 결과를 합쳐 기업별 평가를 저장하고 다음 기업으로 넘어갑니다."""
    c = state["company"]
    scores = {
        "tech": state["tech_result"]["score"],
        "market": state["market_result"]["score"],
        "revenue": state["revenue_result"]["score"],
    }
    evaluation = {
        "company": c,
        "scores": scores,
        "total": round(sum(scores.values()), 1),
        "tech": state["tech_result"],
        "market": state["market_result"],
        "revenue": state["revenue_result"],
    }
    print(f"  → 기술 {scores['tech']} / 시장 {scores['market']} / 매출 {scores['revenue']} = 총점 {evaluation['total']}")
    return {"evaluations": [evaluation], "current_index": state["current_index"] + 1}


def route_start(state: GraphState) -> str:
    """저장된 평가를 재사용하면 분석을 건너뛰고 투자 판단부터 시작합니다."""
    if state.get("forecasts"):
        return "select_top"  # 저장된 수익률까지 재사용 → 보고서만 다시 생성
    if state.get("evaluations") and state["current_index"] >= len(state["companies"]):
        return "investment_judge"
    return "select_company"


def route_after_record(state: GraphState) -> str:
    """남은 기업이 있으면 다음 기업, 없으면 투자 판단으로 이동합니다."""
    if state["current_index"] < len(state["companies"]):
        return "select_company"
    return "investment_judge"


# ---------- 투자 판단 ----------

def investment_judge(state: GraphState) -> GraphState:
    """현재 기준(총점 + 과락)으로 통과 기업을 고릅니다. 분석은 다시 하지 않습니다."""
    total_th = state["threshold_total"]
    cutoff = state["threshold_cutoff"]
    passed = [
        e for e in state["evaluations"]
        if e["total"] >= total_th
        and all(e["scores"][area] >= AREA_WEIGHTS[area] * cutoff for area in AREA_WEIGHTS)
    ]
    print(f"\n[투자 판단] 총점 {total_th}점 이상 · 과락 {cutoff:.0%} → 통과 {len(passed)}개")
    return {"passed": passed}


def route_after_judge(state: GraphState) -> str:
    """통과 기업이 3개 미만이면 기준을 완화합니다."""
    if len(state["passed"]) >= TOP_N or len(state["evaluations"]) < TOP_N:
        return "forecast_returns"
    return "relax_criteria"


def relax_criteria(state: GraphState) -> GraphState:
    """기준 완화: 총점 -5점, 과락 -5%p (과락은 0%에서 멈춤)."""
    return {
        "threshold_total": state["threshold_total"] - RELAX_TOTAL_STEP,
        "threshold_cutoff": max(0.0, round(state["threshold_cutoff"] - RELAX_CUTOFF_STEP, 2)),
        "relax_round": state["relax_round"] + 1,
    }


UNIT_TO_USD = {"USD_M": 1e6, "USD_B": 1e9}
UNIT_TO_KRW = {"KRW_억": 1e8, "KRW_조": 1e12}


def number_in_text(value: float, text: str) -> bool:
    """숫자가 자료 본문에 실제로 나오는지 확인합니다 (쉼표·소수 표기 허용)."""
    candidates = {f"{value:g}", f"{value:,.0f}", f"{value:.1f}", f"{value:.2f}", f"{value:,.1f}"}
    if any(c in text for c in candidates):
        return True
    # "2조 6,500억"처럼 조·억이 섞인 표기는 조 단위(2.65)와 억 단위(26,500) 값으로 바꿔 비교
    for jo, eok in re.findall(r"(\d+)\s*조\s*([\d,]+)\s*억", text):
        jo, eok = int(jo), int(eok.replace(",", ""))
        if any(abs(value - v) < 1e-6 * max(v, 1) for v in (jo + eok / 10000, jo * 10000 + eok)):
            return True
    return False


def to_usd(amount: Amount, usd_krw: float, context: str) -> tuple[float | None, str]:
    """원문 표기 금액을 코드에서 달러로 환산합니다. 자료에서 확인되지 않는 숫자는 버립니다."""
    if amount.value is None or amount.unit == "none":
        return None, ""
    if not amount.quote or not number_in_text(amount.value, context):
        return None, "자료에서 숫자 확인 실패"
    if amount.unit in UNIT_TO_USD:
        usd = amount.value * UNIT_TO_USD[amount.unit]
    else:
        usd = amount.value * UNIT_TO_KRW[amount.unit] / usd_krw
    origin = "조사 보고서" if amount.from_report else "뉴스"
    return usd, f"{origin}: \"{amount.quote[:80]}\""


def estimate_valuation(resolved: dict, latest_round: str) -> tuple[float | None, str]:
    """현재 기업가치(달러)를 추정합니다.

    우선순위: 공개 기업가치(상식 검사 통과 시) → 최근 라운드 ÷ Carta 중간 희석률 → 누적 투자금 ÷ 중간 희석률.
    (그래도 없으면 forecast_returns에서 세그먼트 중간값으로 대체)
    """
    median_dilution = statistics.median(DILUTION_BY_ROUND.values())
    valuation, funding = resolved["valuation"], resolved["total_funding"]
    note = ""
    if valuation:
        # 상식 검사: 기업가치는 누적 투자금보다 커야 하고, 100배를 넘으면 추출 오류로 봄
        if funding and not (funding <= valuation <= funding * 100):
            note = " (공개 기업가치 검증 실패 → 대체값 사용)"
        else:
            return valuation, f"공개 기업가치 [{resolved['valuation_basis']}]"
    if resolved["round_amount"]:
        dilution = DILUTION_BY_ROUND.get(latest_round, median_dilution)
        return resolved["round_amount"] / dilution, f"최근 라운드 ÷ Carta 중간 희석률 {dilution:.1%}{note}"
    if funding:
        return funding / median_dilution, f"누적 투자금 ÷ Carta 중간 희석률 {median_dilution:.1%}{note}"
    return None, "기업가치 추정 불가" + note


def estimate_revenue(
    inputs: ReturnInputs,
    resolved: dict,
    target_year: int,
    market_cagr_pct: float | None,
    current_revenue_usd: float | None = None,
    current_revenue_year: int | None = None,
) -> tuple[float | None, str]:
    """n년 후 예상 매출(달러)을 추정합니다.

    우선순위: 목표 점유율 → 목표 매출(올해 이후 연도만) → 현재 매출(유사 지표). 기준 연도가 이르면 시장 성장률로 보정.
    """
    if inputs.target_share_pct and resolved["future_market"]:
        revenue = resolved["future_market"] * inputs.target_share_pct / 100
        base_year, method = inputs.future_market_year, "목표 시장 규모 × 목표 점유율"
    elif resolved["target_revenue"] and (inputs.target_revenue_year or 0) >= date.today().year:
        # 이미 지난 해의 전망치나 연도 없는 목표는 목표 매출로 쓰지 않음 (확정 실적도 아니므로 보수적으로 제외)
        revenue = resolved["target_revenue"]
        base_year, method = inputs.target_revenue_year, f"기업 공개 목표 매출 [{resolved['target_revenue_basis']}]"
    elif current_revenue_usd and market_cagr_pct:
        revenue = current_revenue_usd
        base_year, method = current_revenue_year, "현재 매출 (목표 점유율·목표 매출 미공개 → 유사 지표)"
    else:
        return None, "예상 매출 근거 없음"

    if base_year and base_year < target_year and market_cagr_pct:
        revenue *= (1 + market_cagr_pct / 100) ** (target_year - base_year)
        method += f" ({base_year}→{target_year}년, 시장 성장률 {market_cagr_pct}%로 보정)"
    return revenue, method


def median_by_segment(values: dict[str, float]) -> tuple[dict[str, float], float | None]:
    """기업 코드별 값을 세그먼트 중간값과 전체 중간값으로 요약합니다."""
    by_segment: dict[str, list[float]] = {}
    for cid, value in values.items():
        by_segment.setdefault(SEGMENTS.get(cid), []).append(value)
    overall = statistics.median(values.values()) if values else None
    return {seg: statistics.median(v) for seg, v in by_segment.items()}, overall


def forecast_returns(state: GraphState) -> GraphState:
    """통과 기업의 투자 수익률을 예측합니다 (투자 판단 에이전트).

    입력값이 비어 있으면 단계별 대체값을 써서 수익률을 항상 산출하고,
    어떤 대체값을 썼는지 method에 남깁니다.
    """
    peers = get_peer_multiples()
    usd_krw = get_usd_krw()
    years = state["investment_years"]
    target_year = date.today().year + years
    invest_usd = state["investment_amount"] / usd_krw

    # 시장 성장률: 기업별 → 같은 세그먼트 중간값 → 전체 중간값
    cagr_values = {
        ev["company"]["id"]: ev["market"]["detail"]["market_cagr_pct"]
        for ev in state["evaluations"]
        if ev["market"]["detail"].get("market_cagr_pct")
    }
    cagr_by_segment, cagr_overall = median_by_segment(cagr_values)

    # 1단계: 통과 기업의 수익률 입력값 추출 (RAG + 뉴스)
    extracted = []
    for e in state["passed"]:
        c = e["company"]
        docs = retrieve(f"{c['name_ko']} {c['name']} 목표 매출 시장 점유율 기업가치 투자 라운드 누적 투자", c["id"], "investment")
        news = search_news(f"{c['name_ko']} 목표 매출 점유율") + search_news(f"{c['name']} valuation funding round", lang="en")
        prompt = f"""{c['name_ko']}({c['name']})의 수익률 계산 입력값을 추출하세요.

- 기업이 공개한 목표 시장 점유율, 목표 시장의 전망 규모(연도), 목표 매출(연도)
- 공개된 현재 기업가치, 가장 최근 투자 라운드와 금액, 누적 투자 유치액
- 금액은 절대 환산하지 마세요. 자료에 적힌 숫자와 단위를 그대로 적고(예: "1조 원" → value 1, unit KRW_조), 그 숫자가 나온 문장을 quote에 글자 그대로 옮기세요.
- 조사 보고서(RAG)에 값이 있으면 그 값을 쓰고, 없을 때만 뉴스를 쓰세요.
- 기업가치가 범위(예: 8,000억~1조 원)로 적혀 있으면 큰 값을 쓰세요.
- 다른 회사의 금액이나, 자료에 없는 금액을 적지 마세요. 없으면 value는 null, unit은 none.
- 수주 잔고·약정은 목표 매출이 아닙니다.
- 예상 매출의 근거가 뉴스·산업 리포트로 확인되는지 신뢰성을 판정하세요.
- 목표 시장: {e['market']['detail']['target_market']} / 시장 규모 자료: {e['market']['detail']['market_size']}

규칙:
{SOURCE_RULE}

# 조사 보고서 (RAG)
{format_docs(docs)}

# 뉴스
{format_news(news)}
"""
        inputs: ReturnInputs = get_llm(ReturnInputs).invoke(prompt)

        # 환산과 검증은 코드에서 수행
        context = format_docs(docs) + format_news(news)
        resolved = {}
        for key, amount in {
            "valuation": inputs.current_valuation,
            "round_amount": inputs.latest_round_amount,
            "total_funding": inputs.total_funding,
            "target_revenue": inputs.target_revenue,
            "future_market": inputs.future_market_size,
        }.items():
            resolved[key], resolved[f"{key}_basis"] = to_usd(amount, usd_krw, context)

        valuation, valuation_method = estimate_valuation(resolved, inputs.latest_round)
        extracted.append({
            "e": e, "inputs": inputs, "resolved": resolved, "news": news,
            "valuation": valuation, "valuation_method": valuation_method,
        })

    # 기업가치 대체값: 같은 세그먼트 통과 기업의 중간값 → 전체 중간값
    valuation_by_segment, valuation_overall = median_by_segment(
        {x["e"]["company"]["id"]: x["valuation"] for x in extracted if x["valuation"]}
    )

    # 2단계: 수익률 계산
    forecasts = []
    for x in extracted:
        e, inputs = x["e"], x["inputs"]
        c = e["company"]
        segment = SEGMENTS.get(c["id"])

        market_cagr = cagr_values.get(c["id"])
        cagr_note = ""
        if not market_cagr:
            market_cagr = cagr_by_segment.get(segment) or cagr_overall
            cagr_note = " [시장 성장률: 같은 세그먼트 기업 중간값]" if cagr_by_segment.get(segment) else " [시장 성장률: 전체 기업 중간값]"
            market_cagr = round(market_cagr, 1) if market_cagr else None

        valuation, valuation_method = x["valuation"], x["valuation_method"]
        if not valuation:
            valuation = valuation_by_segment.get(segment) or valuation_overall
            valuation_method = "같은 세그먼트 통과 기업의 기업가치 중간값 (대체값)" if valuation_by_segment.get(segment) else "통과 기업 전체의 기업가치 중간값 (대체값)"

        revenue_krw = e["revenue"]["detail"].get("revenue_krw_normalized")
        revenue, revenue_method = estimate_revenue(
            inputs,
            x["resolved"],
            target_year,
            market_cagr,
            current_revenue_usd=revenue_krw / usd_krw if revenue_krw else None,
            current_revenue_year=e["revenue"]["detail"].get("revenue_year"),
        )

        roi = pv = exit_value = stake = None
        if valuation:
            if revenue:
                exit_value = revenue * peers["net_margin_median"] * peers["pe_median"]
                revenue_method += cagr_note
            elif market_cagr:
                # 매출을 추정할 수 없으면 현재 기업가치가 시장 성장률만큼 성장한다고 가정
                exit_value = valuation * (1 + market_cagr / 100) ** years
                revenue_method = f"매출 미공개 → 현재 기업가치가 시장 성장률 {market_cagr}%로 성장한다고 가정 (대체 계산){cagr_note}"
            if exit_value:
                stake = invest_usd / (valuation + invest_usd)
                pv = exit_value * stake / (1 + DISCOUNT_RATE) ** years
                roi = (pv - invest_usd) / invest_usd

        forecasts.append({
            "company": c,
            "total": e["total"],
            "roi": roi,
            "expected_revenue_usd": revenue,
            "revenue_method": revenue_method,
            "market_cagr_pct": market_cagr,
            "valuation_usd": valuation,
            "valuation_method": valuation_method,
            "stake": stake,
            "exit_value_usd": exit_value,
            "present_value_usd": pv,
            "reliability": inputs.reliability,
            "reliability_note": inputs.reliability_note,
            "sources": inputs.sources,
            "news": x["news"],
        })
        roi_text = f"{roi:.0%}" if roi is not None else "산정 불가"
        print(f"  {c['name_ko']}: 수익률 {roi_text} ({revenue_method} / {valuation_method})")

    return {"forecasts": forecasts}


def select_top(state: GraphState) -> GraphState:
    """수익률 TOP3를 선정합니다. 동점이면 총점 높은 순, 수익률 산정 불가는 뒤로."""
    ranked = sorted(
        state["forecasts"],
        key=lambda f: (f["roi"] is None, -(f["roi"] or 0), -f["total"]),
    )
    top = ranked[:TOP_N]
    print("\n[TOP3] " + ", ".join(f["company"]["name_ko"] for f in top))
    return {"top_companies": top}


# ---------- SWOT · 보고서 ----------

def swot_agent(state: GraphState) -> GraphState:
    """TOP3 기업의 SWOT을 분석합니다 (RAG)."""
    evaluations = {e["company"]["id"]: e for e in state["evaluations"]}
    results = []
    for f in state["top_companies"]:
        c = f["company"]
        e = evaluations[c["id"]]
        docs = retrieve(f"{c['name_ko']} {c['name']} 강점 약점 리스크 기회", c["id"], "SWOT")
        prompt = f"""{c['name_ko']}({c['name']})의 SWOT을 분석하세요. 자료에 있는 사실만 사용하세요.

# 기술 분석
{e['tech']['detail']['summary']}
# 시장 분석
{e['market']['detail']['summary']}
# 매출 분석
{e['revenue']['detail']['summary']}
# 조사 보고서 (RAG)
{format_docs(docs)}
"""
        swot: SWOT = get_llm(SWOT).invoke(prompt)
        results.append({"company": c, **swot.model_dump()})
    return {"swot": results}


REPORT_TOC = """# SUMMARY  (1/2페이지 이내: 추천 3곳, 투자 금액, 예측 수익률, 핵심 근거 1~2문장)
# 1. 평가 기준 및 고지  (평가 프레임, 최종 적용 기준·완화 횟수, 수익률 계산 방법과 가정, 예상 매출 신뢰성)
# 2. 기업별 분석  (추천 3곳 각각: 사업 아이디어·팀 / 기술 / 시장(목표 시장과 시장 규모) / 재무 요약 / 평가 점수·수익률 / SWOT / 사업 리스크(시장·기술·규제·경쟁))
# 3. 한계점"""

DOC_NAME = "AI반도체_스타트업_30사_기업평가_자료.md"


def load_citations() -> tuple[dict, dict]:
    """조사 문서 섹션별 원자료 인용 번호와, 인용 번호별 원자료 정보를 불러옵니다."""
    chunks = json.loads((RAG_STORE_DIR / "data" / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    by_section = {}
    for ch in chunks.values():
        meta = ch["metadata"]
        section = " > ".join(meta.get("section_path", []))
        by_section.setdefault(section, set()).update(meta.get("citation_ids", []))
    sources = json.loads((RAG_STORE_DIR / "data" / "sources.json").read_text(encoding="utf-8"))
    return by_section, sources


def news_date(published: str) -> str:
    """RSS 발행일을 YYYY-MM-DD로 바꿉니다."""
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(published).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return "날짜 확인 안 됨"


def build_references(top_companies: list[dict], evaluations: dict, peers: dict) -> str:
    """에이전트가 근거로 쓴 자료(sources)만 모아 REFERENCE 장을 만듭니다.

    조사 문서 섹션은 그 섹션이 인용한 원자료로 풀어 쓰고, 뉴스는 제목·언론사·날짜를 붙입니다.
    """
    by_section, citations = load_citations()
    cited_ids, news_refs = [], {}
    for f in top_companies:
        e = evaluations[f["company"]["id"]]
        results = [e["tech"], e["market"], e["revenue"], f]
        news_meta = {n["url"]: n for r in results for n in r.get("news", [])}
        for src in (s for r in results for s in r["sources"]):
            if src.startswith("http"):
                if src in news_meta:  # 제공한 뉴스에 없는 URL은 근거로 인정하지 않음
                    news_refs[src] = news_meta[src]
            elif DOC_NAME in src and " / " in src:
                section = src.split(" / ", 1)[1].strip()
                for key, ids in by_section.items():
                    if key.startswith(section):
                        cited_ids += [i for i in sorted(ids) if i not in cited_ids]

    lines = ["# REFERENCE", "", "**조사 문서 (RAG)**", "",
             f"1. 울산캠퍼스 1반 3조 (2026). 『AI반도체 스타트업 30사 기업평가 자료』. 팀 내부 조사 문서 (docs/{DOC_NAME})"]
    n = 1
    originals = [i for i in cited_ids if i in citations]
    if originals:
        lines += ["", "**원자료 (조사 문서가 인용한 자료)**", ""]
        for cid in originals:
            n += 1
            lines.append(f"{n}. [{cid}] {citations[cid]['description_raw'].replace(' 📁', '').strip()}")
    if news_refs:
        lines += ["", "**뉴스**", ""]
        for url, news in news_refs.items():
            n += 1
            title = news["title"].rsplit(" - ", 1)[0]
            lines.append(f"{n}. {news['source'] or '언론사 확인 안 됨'} ({news_date(news['published'])}). 「{title}」. {url}")

    lines += ["", "**평가 기준·데이터**", ""]
    n += 1
    lines.append(f"{n}. 산업통상자원부 (2014). 『기술가치평가 실무가이드』 (부록 1 시장 평가 기준, 할인율). "
                 "https://www.valuation.or.kr/data/%EA%B8%B0%EC%88%A0%EA%B0%80%EC%B9%98%ED%8F%89%EA%B0%80_%EC%8B%A4%EB%AC%B4%EA%B0%80%EC%9D%B4%EB%93%9C(2014).pdf")
    if any("Carta" in f["valuation_method"] for f in top_companies):
        n += 1
        lines.append(f"{n}. Carta (2025). Dilution by venture round medians. https://carta.com/data/linkedin-dilution-by-venture-round-medians/")
    n += 1
    tickers = ", ".join(p["ticker"] for p in peers["peers"])
    lines.append(f"{n}. Yahoo Finance ({peers['date']}). 비교기업 순이익률·PER ({tickers}). https://finance.yahoo.com")
    return "\n".join(lines) + "\n"


def format_money(usd: float | None, usd_krw: float) -> str:
    """달러 금액을 '달러 (원화)' 문자열로 표기합니다."""
    if usd is None:
        return "확인 안 됨"
    krw = usd * usd_krw
    # 큰 금액을 "3,500M"처럼 쓰면 모델이 "3,500억 달러"로 잘못 옮기므로 B·조 단위를 씀
    if usd >= 1e9:
        usd_text = f"${usd / 1e9:,.2f}B"
    elif usd >= 1e6:
        usd_text = f"${usd / 1e6:,.1f}M"
    else:
        usd_text = f"${usd / 1e3:,.0f}K"
    if krw >= 1e12:
        krw_text = f"약 {int(krw // 1e12)}조 {round(krw % 1e12 / 1e8):,}억 원"
    else:
        krw_text = f"약 {krw / 1e8:,.0f}억 원"
    return f"{usd_text} ({krw_text})"


def format_forecast(f: dict, usd_krw: float) -> dict:
    """수익률 예측값을 보고서에 그대로 옮겨 쓸 수 있는 문자열로 바꿉니다."""
    roi = f["roi"]
    return {
        "예상 수익률": f"{roi * 100:+.1f}%" if roi is not None else "산정 불가",
        "n년 후 예상 매출": format_money(f["expected_revenue_usd"], usd_krw),
        "예상 매출 산정 방식": f["revenue_method"],
        "적용 시장 성장률": f"{f['market_cagr_pct']}%" if f.get("market_cagr_pct") else "확인 안 됨",
        "현재 기업가치": format_money(f["valuation_usd"], usd_krw),
        "기업가치 산정 방식": f["valuation_method"],
        "취득 지분율": f"{f['stake'] * 100:.4f}%" if f["stake"] else "확인 안 됨",
        "Exit 시점 기업가치": format_money(f["exit_value_usd"], usd_krw),
        "회수 금액의 현재가치": format_money(f["present_value_usd"], usd_krw),
        "예상 매출 신뢰성": f["reliability"],
        "신뢰성 근거": f["reliability_note"],
    }


def report_agent(state: GraphState) -> GraphState:
    """투자 보고서를 생성합니다 (5장 이내)."""
    evaluations = {e["company"]["id"]: e for e in state["evaluations"]}
    peers = get_peer_multiples()

    top_data = []
    for f, s in zip(state["top_companies"], state["swot"]):
        e = evaluations[f["company"]["id"]]
        top_data.append({
            "company": f["company"],
            "scores": e["scores"],
            "total": e["total"],
            "tech": {k: v for k, v in e["tech"]["detail"].items() if k != "sources"},
            "market": {k: v for k, v in e["market"]["detail"].items() if k != "sources"},
            "revenue": {k: v for k, v in e["revenue"]["detail"].items() if k != "sources"},
            "수익률 예측": format_forecast(f, get_usd_krw()),
            "swot": {k: v for k, v in s.items() if k != "company"},
        })

    criteria = {
        "평가 프레임": "기술 40 / 시장 40 / 매출 20 = 100점",
        "최종 적용 기준": f"총점 {state['threshold_total']}점 이상, 영역별 과락 {state['threshold_cutoff']:.0%}",
        "기준 완화 횟수": state["relax_round"],
        "수익률 가정": {
            "투자 금액": f"기업당 {state['investment_amount'] / 1e8:,.0f}억 원 (약 ${state['investment_amount'] / get_usd_krw() / 1e6:,.2f}M)",
            "적용 환율(원/달러)": round(get_usd_krw(), 1),
            "투자 기간(년)": state["investment_years"],
            "할인율": f"{DISCOUNT_RATE:.0%} (기술가치평가 실무가이드: 비상장 자기자본비용 통상 10~25%)",
            "비교기업 순이익률 중간값": f"{peers['net_margin_median'] * 100:.1f}%",
            "비교기업 PER 중간값": f"{peers['pe_median']:.1f}배",
            "비교기업 출처": peers["source"] + f" ({peers['date']})",
        },
        "평가 기업 수": len(state["evaluations"]),
    }

    prompt = f"""당신은 벤처캐피탈 심사역입니다. 아래 데이터로 반도체 스타트업 투자 보고서를 한국어 Markdown으로 작성하세요.

목차 (반드시 이 순서):
{REPORT_TOC}

규칙:
- 전체 A4 5장 이내. SUMMARY는 개요가 아니라 핵심 요약이며 1/2페이지를 넘지 않게.
- 목차의 장만 쓰세요. REFERENCE 장은 쓰지 마세요(별도로 붙입니다). 목차 밖의 맺음말·안내 문구도 쓰지 마세요.
- 데이터에 있는 사실과 숫자만 사용하고, 없는 정보는 "확인 안 됨"으로 쓰세요.
- 통화 환산이나 수익률을 직접 계산하지 마세요. "수익률 예측"의 문자열을 글자 그대로 옮기세요.
- 데이터의 키 이름(forecast, detail, JSON 등)이나 이 작성 지시를 보고서에 언급하지 마세요.
- 인력 이동은 방향(어디에서 어디로)을 데이터 문장 그대로 쓰세요.
- 양산·제품의 주체가 이 회사가 아니면(예: 기술을 라이선스한 다른 회사) 주체를 밝혀 쓰세요.
- "예정"이라고 적힌 일정이 이미 지난 연도라면 "(YYYY년 보도 기준 예정, 이후 확인 안 됨)"처럼 시점을 밝히세요.
- "예상 수익률"은 (회수 금액의 현재가치 − 투자 금액) ÷ 투자 금액입니다. +는 이익, −는 손실이며 0%에 가까우면 원금 수준입니다.
- 1장에 수익률 계산 방법을 한 줄로 적으세요: 예상 매출 × 비교기업 순이익률 × 비교기업 PER = Exit 기업가치 → × 취득 지분율 → 할인율로 현재가치 환산.
- 시장 규모는 market_size의 수치를 쓰고, 수치가 없으면 "시장 규모 수치 확인 안 됨"이라고 쓰세요.
- 사업 리스크는 시장·기술·규제·경쟁 4가지로 나눠 쓰고, 해당 자료가 없는 항목은 "확인 안 됨"으로 쓰세요.
- 수익률 산정 불가 기업은 그 이유를 밝히세요.
- 비교기업 PER·순이익률은 현재 상장사 기준이라 Exit 가치가 크게 나올 수 있음을 고지에 적으세요.
- 예상 매출 신뢰성(근거 확인/일부 확인/근거 없음)을 기업별로 표시하세요.

# 평가 기준
{json.dumps(criteria, ensure_ascii=False, indent=2)}

# 추천 기업 데이터
{json.dumps(top_data, ensure_ascii=False, indent=2, default=str)}
"""
    report = get_llm().invoke(prompt).content
    # 모델이 REFERENCE나 맺음말을 덧붙였으면 잘라 내고, REFERENCE는 코드에서 만든 목록을 붙임
    report = re.split(r"\n#{1,3}\s*REFERENCE", report)[0]
    report = re.sub(r"(\n\s*(---|\(.*\))\s*)+$", "", report.rstrip()) + "\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 그래프: 통과 기업의 영역별 점수와 예상 수익률 → SUMMARY 바로 아래에 삽입
    chart_path = draw_summary_chart(
        state["evaluations"],
        state["forecasts"],
        [f["company"]["id"] for f in state["top_companies"]],
        OUTPUT_DIR / "summary_chart.png",
    )
    chart_md = f"\n![통과 기업 평가 점수와 예상 수익률]({chart_path.name})\n\n"
    # 제목 수준(#, ##)과 관계없이 "1." 장 제목 바로 앞에 넣고, 못 찾으면 본문 끝에 넣음
    match = re.search(r"\n#{1,3}\s*1\.", report)
    if match:
        report = report[: match.start()] + "\n" + chart_md + report[match.start() + 1 :]
    else:
        report = report + chart_md
    report += "\n---\n\n" + build_references(state["top_companies"], evaluations, peers)

    path = OUTPUT_DIR / f"investment_report_{date.today()}.md"
    path.write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "forecasts.json").write_text(
        json.dumps(state["forecasts"], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (OUTPUT_DIR / "criteria.json").write_text(
        json.dumps({k: state[k] for k in ("threshold_total", "threshold_cutoff", "relax_round")}), encoding="utf-8"
    )
    (OUTPUT_DIR / "evaluations.json").write_text(
        json.dumps(state["evaluations"], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    pdf_path = markdown_to_pdf(report, OUTPUT_DIR / f"RAG-Output_{TEAM_FILE_TAG}.pdf")
    print(f"\n보고서 저장: {path}\nPDF 저장: {pdf_path}")
    return {"report": report}

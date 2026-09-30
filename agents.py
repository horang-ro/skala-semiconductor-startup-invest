"""투자 평가 그래프의 노드(에이전트)를 정의합니다."""

import json
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
from pdf import markdown_to_pdf
from rag import format_docs, retrieve
from state import (
    SWOT,
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


SEGMENTS = load_segments()


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
            "sources": result.sources + [n["url"] for n in news],
        }
    }


def market_agent(state: GraphState) -> GraphState:
    """시장 평가 에이전트 (RAG + 뉴스): 시장 40점 (실무가이드 부록 1)."""
    c = state["company"]
    docs = retrieve(f"{c['name_ko']} {c['name']} 목표 시장 시장 규모 성장 경쟁 세그먼트", c["id"], "market")
    segment = SEGMENTS.get(c["id"], "AI 반도체")
    news = (
        search_news(f"{segment} 시장 규모 전망 성장률")
        + search_news(f"{c['name']} AI chip market size forecast", lang="en")
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

    items = {key: graded_points(getattr(result, key), pts) for key, pts in MARKET_ITEMS.items()}
    return {
        "market_result": {
            "score": round(sum(items.values()), 1),
            "items": items,
            "detail": result.model_dump(),
            "sources": result.sources + [n["url"] for n in news],
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
- 투자금, 수주 약정, 기업가치, 목표 매출은 매출이 아닙니다.
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
            "sources": facts.sources + [n["url"] for n in news],
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


def estimate_valuation(inputs: ReturnInputs) -> tuple[float | None, str]:
    """현재 기업가치(달러)를 추정합니다: 공개 기업가치 → 최근 라운드 ÷ Carta 중간 희석률."""
    if inputs.current_valuation_usd:
        return inputs.current_valuation_usd, "공개 기업가치"
    if inputs.latest_round_amount_usd and inputs.latest_round in DILUTION_BY_ROUND:
        dilution = DILUTION_BY_ROUND[inputs.latest_round]
        return inputs.latest_round_amount_usd / dilution, f"최근 라운드 ÷ Carta 중간 희석률 {dilution:.1%}"
    return None, "기업가치 추정 불가"


def estimate_revenue(
    inputs: ReturnInputs,
    target_year: int,
    market_cagr_pct: float | None,
    current_revenue_usd: float | None = None,
    current_revenue_year: int | None = None,
) -> tuple[float | None, str]:
    """n년 후 예상 매출(달러)을 추정합니다.

    우선순위: 목표 점유율 → 목표 매출 → 현재 매출(유사 지표). 기준 연도가 이르면 시장 성장률로 보정.
    """
    if inputs.target_share_pct and inputs.future_market_size_usd:
        revenue = inputs.future_market_size_usd * inputs.target_share_pct / 100
        base_year, method = inputs.future_market_year, "목표 시장 규모 × 목표 점유율"
    elif inputs.target_revenue_usd:
        revenue = inputs.target_revenue_usd
        base_year, method = inputs.target_revenue_year, "기업 공개 목표 매출"
    elif current_revenue_usd and market_cagr_pct:
        revenue = current_revenue_usd
        base_year, method = current_revenue_year, "현재 매출 (목표 점유율·목표 매출 미공개 → 유사 지표)"
    else:
        return None, "예상 매출 근거 없음"

    if base_year and base_year < target_year and market_cagr_pct:
        revenue *= (1 + market_cagr_pct / 100) ** (target_year - base_year)
        method += f" ({base_year}→{target_year}년, 시장 성장률 {market_cagr_pct}%로 보정)"
    return revenue, method


def forecast_returns(state: GraphState) -> GraphState:
    """통과 기업의 투자 수익률을 예측합니다 (투자 판단 에이전트)."""
    peers = get_peer_multiples()
    usd_krw = get_usd_krw()
    years = state["investment_years"]
    target_year = date.today().year + years
    invest_usd = state["investment_amount"] / usd_krw

    forecasts = []
    for e in state["passed"]:
        c = e["company"]
        docs = retrieve(f"{c['name_ko']} {c['name']} 목표 매출 시장 점유율 기업가치 투자 라운드", c["id"], "investment")
        news = search_news(f"{c['name_ko']} 목표 매출 점유율")
        prompt = f"""{c['name_ko']}({c['name']})의 수익률 계산 입력값을 추출하세요.

- 기업이 공개한 목표 시장 점유율, 목표 시장의 전망 규모(연도), 목표 매출(연도)
- 공개된 현재 기업가치, 가장 최근 투자 라운드와 금액 (원화는 1달러={usd_krw:.0f}원으로 환산)
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

        revenue_krw = e["revenue"]["detail"].get("revenue_krw_normalized")
        revenue, revenue_method = estimate_revenue(
            inputs,
            target_year,
            e["market"]["detail"].get("market_cagr_pct"),
            current_revenue_usd=revenue_krw / usd_krw if revenue_krw else None,
            current_revenue_year=e["revenue"]["detail"].get("revenue_year"),
        )
        valuation, valuation_method = estimate_valuation(inputs)

        roi = pv = exit_value = stake = None
        if revenue and valuation:
            net_income = revenue * peers["net_margin_median"]
            exit_value = net_income * peers["pe_median"]
            stake = invest_usd / (valuation + invest_usd)
            pv = exit_value * stake / (1 + DISCOUNT_RATE) ** years
            roi = (pv - invest_usd) / invest_usd

        forecasts.append({
            "company": c,
            "total": e["total"],
            "roi": roi,
            "expected_revenue_usd": revenue,
            "revenue_method": revenue_method,
            "valuation_usd": valuation,
            "valuation_method": valuation_method,
            "stake": stake,
            "exit_value_usd": exit_value,
            "present_value_usd": pv,
            "reliability": inputs.reliability,
            "reliability_note": inputs.reliability_note,
            "sources": inputs.sources + [n["url"] for n in news],
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


REPORT_TOC = """# SUMMARY  (1/2페이지 이내: 추천 3곳, 투자 금액, 예측 수익률, 핵심 근거)
# 1. 평가 기준 및 고지  (평가 프레임, 최종 적용 기준·완화 횟수, 수익률 가정, 예상 매출 신뢰성)
# 2. 기업별 분석  (추천 3곳 각각: 사업 아이디어·팀 / 기술·시장(시장 규모 포함) / 재무 요약 / 평가 점수·수익률 / SWOT·사업 리스크)
# 3. 한계점
# REFERENCE  (실제로 활용한 자료만. 형식: [번호] 기관·저자, 「제목」, 발행일, URL)"""


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
            "tech": e["tech"]["detail"],
            "market": e["market"]["detail"],
            "revenue": e["revenue"]["detail"],
            "forecast": {k: v for k, v in f.items() if k != "company"},
            "swot": {k: v for k, v in s.items() if k != "company"},
            "sources": sorted(set(e["tech"]["sources"] + e["market"]["sources"] + e["revenue"]["sources"] + f["sources"])),
        })

    criteria = {
        "평가 프레임": "기술 40 / 시장 40 / 매출 20 = 100점",
        "최종 적용 기준": f"총점 {state['threshold_total']}점 이상, 영역별 과락 {state['threshold_cutoff']:.0%}",
        "기준 완화 횟수": state["relax_round"],
        "수익률 가정": {
            "투자 금액(원)": state["investment_amount"],
            "투자 금액(달러)": round(state["investment_amount"] / get_usd_krw()),
            "적용 환율(원/달러)": round(get_usd_krw(), 1),
            "투자 기간(년)": state["investment_years"],
            "할인율": f"{DISCOUNT_RATE:.0%} (기술가치평가 실무가이드: 비상장 자기자본비용 통상 10~25%)",
            "비교기업 순이익률 중간값": peers["net_margin_median"],
            "비교기업 PER 중간값": peers["pe_median"],
            "비교기업 출처": peers["source"] + f" ({peers['date']})",
        },
        "평가 기업 수": len(state["evaluations"]),
    }

    prompt = f"""당신은 벤처캐피탈 심사역입니다. 아래 데이터로 반도체 스타트업 투자 보고서를 한국어 Markdown으로 작성하세요.

목차 (반드시 이 순서):
{REPORT_TOC}

규칙:
- 전체 A4 5장 이내. SUMMARY는 개요가 아니라 핵심 요약이며 1/2페이지를 넘지 않게.
- 데이터에 있는 사실과 숫자만 사용하고, 없는 정보는 "확인 안 됨"으로 쓰세요.
- 통화 환산을 직접 계산하지 마세요. 데이터에 있는 금액과 "적용 환율"만 사용하세요.
- 수익률 산정 불가 기업은 그 이유를 밝히세요.
- 비교기업 PER·순이익률은 현재 상장사 기준이라 Exit 가치가 크게 나올 수 있음을 고지에 적으세요.
- 예상 매출 신뢰성(근거 확인/일부 확인/근거 없음)을 기업별로 표시하세요.
- REFERENCE에는 sources에 있는 자료만 적으세요.

# 평가 기준
{json.dumps(criteria, ensure_ascii=False, indent=2)}

# 추천 기업 데이터
{json.dumps(top_data, ensure_ascii=False, indent=2, default=str)}
"""
    report = get_llm().invoke(prompt).content

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"investment_report_{date.today()}.md"
    path.write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "evaluations.json").write_text(
        json.dumps(state["evaluations"], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    pdf_path = markdown_to_pdf(report, OUTPUT_DIR / f"RAG-Output_{TEAM_FILE_TAG}.pdf")
    print(f"\n보고서 저장: {path}\nPDF 저장: {pdf_path}")
    return {"report": report}

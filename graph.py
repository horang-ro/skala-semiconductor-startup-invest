"""반도체 스타트업 투자 평가 LangGraph 그래프를 구성합니다."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (
    forecast_returns,
    investment_judge,
    market_agent,
    record_evaluation,
    relax_criteria,
    report_agent,
    revenue_agent,
    route_after_judge,
    route_after_record,
    select_company,
    select_top,
    swot_agent,
    tech_agent,
)
from state import GraphState


def build_graph():
    """그래프를 생성합니다.

    흐름:
        기업 선택 → [기술 · 시장 · 매출] 병렬 분석 → 점수 저장 → (다음 기업 반복)
        → 투자 판단 → (통과 3개 미만이면 기준 완화 후 재판단)
        → 수익률 예측 → TOP3 → SWOT → 보고서
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("select_company", select_company)
    workflow.add_node("tech_agent", tech_agent)
    workflow.add_node("market_agent", market_agent)
    workflow.add_node("revenue_agent", revenue_agent)
    workflow.add_node("record_evaluation", record_evaluation)
    workflow.add_node("investment_judge", investment_judge)
    workflow.add_node("relax_criteria", relax_criteria)
    workflow.add_node("forecast_returns", forecast_returns)
    workflow.add_node("select_top", select_top)
    workflow.add_node("swot_agent", swot_agent)
    workflow.add_node("report_agent", report_agent)

    workflow.add_edge(START, "select_company")

    # 병렬 분석 (fan-out) → 세 결과가 모두 끝나면 저장 (fan-in)
    workflow.add_edge("select_company", "tech_agent")
    workflow.add_edge("select_company", "market_agent")
    workflow.add_edge("select_company", "revenue_agent")
    workflow.add_edge(["tech_agent", "market_agent", "revenue_agent"], "record_evaluation")

    # 기업 루프
    workflow.add_conditional_edges(
        "record_evaluation",
        route_after_record,
        {"select_company": "select_company", "investment_judge": "investment_judge"},
    )

    # 기준 완화 루프
    workflow.add_conditional_edges(
        "investment_judge",
        route_after_judge,
        {"relax_criteria": "relax_criteria", "forecast_returns": "forecast_returns"},
    )
    workflow.add_edge("relax_criteria", "investment_judge")

    workflow.add_edge("forecast_returns", "select_top")
    workflow.add_edge("select_top", "swot_agent")
    workflow.add_edge("swot_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    return workflow.compile(checkpointer=MemorySaver())

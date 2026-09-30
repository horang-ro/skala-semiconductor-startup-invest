"""반도체 스타트업 투자 평가 에이전트 실행 진입점.

사용 예:
    python main.py --amount 1000000000 --years 5
    python main.py --limit 3          # 앞 3개사만으로 빠르게 점검
    python main.py --reuse            # 저장된 점수로 투자 판단·보고서만 다시 생성
    python main.py --report-only      # 저장된 점수·수익률로 SWOT·보고서만 다시 생성
    python main.py --draw             # 그래프 구조(mermaid)만 출력
"""

import argparse
import json

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from config import COMPANIES, INITIAL_CUTOFF_RATIO, INITIAL_TOTAL_THRESHOLD, OUTPUT_DIR
from graph import build_graph
from rag import get_model


def main():
    parser = argparse.ArgumentParser(description="반도체 스타트업 투자 평가")
    parser.add_argument("--amount", type=float, default=1_000_000_000, help="투자 금액(원)")
    parser.add_argument("--years", type=int, default=5, help="투자 유지 기간(년)")
    parser.add_argument("--limit", type=int, default=None, help="평가할 기업 수 (점검용)")
    parser.add_argument("--reuse", action="store_true", help="outputs/evaluations.json 점수를 재사용")
    parser.add_argument("--report-only", action="store_true", help="저장된 점수·수익률로 보고서만 다시 생성")
    parser.add_argument("--draw", action="store_true", help="그래프 mermaid만 출력")
    args = parser.parse_args()

    load_dotenv(override=True)
    app = build_graph()

    if args.draw:
        print(app.get_graph().draw_mermaid())
        return

    try:
        from langchain_teddynote import logging

        logging.langsmith("SKALA-Startup-Invest")
    except Exception:
        pass

    get_model()  # 병렬 실행 전에 임베딩 모델을 한 번 미리 로딩

    companies = COMPANIES[: args.limit] if args.limit else COMPANIES
    evaluations, start_index = [], 0
    extra = {}
    if args.report_only:
        args.reuse = True
        extra = json.loads((OUTPUT_DIR / "criteria.json").read_text()) if (OUTPUT_DIR / "criteria.json").exists() else {}
        extra["forecasts"] = json.loads((OUTPUT_DIR / "forecasts.json").read_text(encoding="utf-8"))
    if args.reuse:
        evaluations = json.loads((OUTPUT_DIR / "evaluations.json").read_text(encoding="utf-8"))
        companies = [e["company"] for e in evaluations]
        start_index = len(companies)
        print(f"저장된 평가 {len(evaluations)}건 재사용")

    inputs = {
        "investment_amount": args.amount,
        "investment_years": args.years,
        "companies": companies,
        "current_index": start_index,
        "evaluations": evaluations,
        "threshold_total": INITIAL_TOTAL_THRESHOLD,
        "threshold_cutoff": INITIAL_CUTOFF_RATIO,
        "relax_round": 0,
    } | extra
    config = RunnableConfig(recursion_limit=500, configurable={"thread_id": "invest-run"})

    result = app.invoke(inputs, config)
    print("\n" + "=" * 60)
    print(result["report"])


if __name__ == "__main__":
    main()

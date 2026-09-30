"""반도체 스타트업 투자 평가 에이전트 실행 진입점.

사용 예:
    python main.py --amount 1000000000 --years 5
    python main.py --limit 3          # 앞 3개사만으로 빠르게 점검
    python main.py --draw             # 그래프 구조(mermaid)만 출력
"""

import argparse

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from config import COMPANIES, INITIAL_CUTOFF_RATIO, INITIAL_TOTAL_THRESHOLD
from graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="반도체 스타트업 투자 평가")
    parser.add_argument("--amount", type=float, default=1_000_000_000, help="투자 금액(원)")
    parser.add_argument("--years", type=int, default=5, help="투자 유지 기간(년)")
    parser.add_argument("--limit", type=int, default=None, help="평가할 기업 수 (점검용)")
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

    companies = COMPANIES[: args.limit] if args.limit else COMPANIES
    inputs = {
        "investment_amount": args.amount,
        "investment_years": args.years,
        "companies": companies,
        "current_index": 0,
        "evaluations": [],
        "threshold_total": INITIAL_TOTAL_THRESHOLD,
        "threshold_cutoff": INITIAL_CUTOFF_RATIO,
        "relax_round": 0,
    }
    config = RunnableConfig(recursion_limit=500, configurable={"thread_id": "invest-run"})

    result = app.invoke(inputs, config)
    print("\n" + "=" * 60)
    print(result["report"])


if __name__ == "__main__":
    main()

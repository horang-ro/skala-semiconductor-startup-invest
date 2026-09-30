"""그래프 State와 에이전트 구조화 출력 스키마를 정의합니다."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class GraphState(TypedDict):
    # 사용자 입력
    investment_amount: Annotated[float, "투자 금액(원)"]
    investment_years: Annotated[int, "투자 유지 기간(년)"]

    # 기업 루프
    companies: Annotated[list[dict], "선정 30개사"]
    current_index: Annotated[int, "현재 평가 중인 기업 순서"]
    company: Annotated[dict, "현재 평가 중인 기업"]

    # 분석 에이전트 결과 (병렬 실행, 서로 다른 키에 기록)
    tech_result: Annotated[dict, "기술 평가 결과"]
    market_result: Annotated[dict, "시장 평가 결과"]
    revenue_result: Annotated[dict, "매출 평가 결과"]

    # 누적 기록
    evaluations: Annotated[list[dict], operator.add]  # 기업별 점수·근거

    # 투자 판단
    threshold_total: Annotated[float, "현재 총점 기준"]
    threshold_cutoff: Annotated[float, "현재 과락 비율"]
    relax_round: Annotated[int, "기준 완화 횟수"]
    passed: Annotated[list[dict], "현재 기준 통과 기업"]
    forecasts: Annotated[list[dict], "통과 기업 수익률 예측"]
    top_companies: Annotated[list[dict], "수익률 TOP3"]

    # 보고서
    swot: Annotated[list[dict], "TOP3 SWOT"]
    report: Annotated[str, "최종 투자 보고서(markdown)"]


# ---------- 구조화 출력 스키마 ----------

class GradedItem(BaseModel):
    """5단계 등급 판정 결과."""

    grade: Literal["a", "b", "c", "d", "e", "none"] = Field(
        description="판정 등급 a(최고)~e(최저). 평가할 자료가 없으면 none"
    )
    score_5: float = Field(
        description="5점 척도 점수(0.5 단위). a=4.5~5, b=3.5~4, c=2.5~3, d=1.5~2, e=0.5~1, none=0"
    )
    reason: str = Field(description="판정 근거 (자료에 있는 사실만, 1~2문장)")


class TechAssessment(BaseModel):
    """기술 평가 결과."""

    core_technology: GradedItem = Field(description="핵심 기술의 구조적 차별성, 해결하는 기술적 문제")
    manufacturability: GradedItem = Field(description="실제 반도체로 구현·제조 가능한 수준")
    validation: GradedItem = Field(description="실제 silicon과 검증자료로 입증 여부")
    scalability: GradedItem = Field(description="기존 시스템과 통합·확장 가능성")
    key_person: bool = Field(description="핵심 인물(창업자·CTO 등) 존재 여부")
    key_person_evidence: str = Field(description="핵심 인물 근거")
    summary: str = Field(description="사업 아이디어(핵심 컨셉)와 기술 요약 3~4문장")
    sources: list[str] = Field(description="근거로 쓴 출처(문서 섹션 또는 URL)")


class MarketAssessment(BaseModel):
    """시장 평가 결과."""

    target_market: str = Field(description="목표 시장 이름")
    market_size: str = Field(description="목표 시장 규모와 전망(수치·연도·조사기관). 없으면 '확인 안 됨'")
    market_cagr_pct: float | None = Field(default=None, description="목표 시장 연평균 성장률(%). 없으면 null")
    growth: GradedItem = Field(description="시장 성장성")
    competition: GradedItem = Field(description="시장 경쟁성")
    entry: GradedItem = Field(description="시장 진입성")
    summary: str = Field(description="시장 분석 요약 2~3문장")
    sources: list[str] = Field(description="근거로 쓴 출처(문서 섹션 또는 URL)")


class RevenueFacts(BaseModel):
    """매출 평가용 사실 추출 결과."""

    latest_revenue_krw: float | None = Field(default=None, description="가장 최근 연매출(원). 달러면 원화 환산 전 값을 latest_revenue_usd에")
    latest_revenue_usd: float | None = Field(default=None, description="가장 최근 연매출(달러)")
    revenue_year: int | None = Field(default=None, description="매출 연도")
    revenue_cagr_3y_pct: float | None = Field(default=None, description="최근 3개년 매출 연평균 증가율(%). 계산 불가면 null")
    revenue_quality: Literal["product", "mixed", "service", "unknown"] = Field(
        description="product=목표 사업 제품 매출, mixed=제품+용역 혼재, service=용역·과제 위주, unknown=자료 없음"
    )
    data_type: str = Field(description="데이터 성격: 공시(감사) / 정부DB 보도 / 2차DB / 언론 / 회사발표 / 미공개")
    summary: str = Field(description="매출·손익 요약 2~3문장")
    sources: list[str] = Field(description="근거로 쓴 출처")


class ReturnInputs(BaseModel):
    """수익률 계산 입력값 추출 결과."""

    target_share_pct: float | None = Field(default=None, description="기업이 공개한 목표 시장 점유율(%)")
    future_market_size_usd: float | None = Field(default=None, description="목표 시장의 전망 규모(달러)")
    future_market_year: int | None = Field(default=None, description="시장 전망 연도")
    target_revenue_usd: float | None = Field(default=None, description="기업이 공개한 목표 매출(달러 환산)")
    target_revenue_year: int | None = Field(default=None, description="목표 매출 연도")
    current_valuation_usd: float | None = Field(default=None, description="공개된 현재 기업가치(달러)")
    latest_round: Literal["seed", "series_a", "series_b", "series_c", "series_d", "unknown"] = Field(
        description="가장 최근 투자 라운드"
    )
    latest_round_amount_usd: float | None = Field(default=None, description="최근 라운드 투자금(달러)")
    reliability: Literal["근거 확인", "일부 확인", "근거 없음"] = Field(
        description="예상 매출 근거가 뉴스·산업 리포트로 확인되는 정도"
    )
    reliability_note: str = Field(description="신뢰성 판단 근거")
    sources: list[str] = Field(description="근거로 쓴 출처")


class SWOT(BaseModel):
    """SWOT 분석 결과."""

    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]
    insight: str = Field(description="핵심 인사이트와 전략적 시사점 2~3문장")

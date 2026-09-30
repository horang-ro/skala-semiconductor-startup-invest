"""외부 정보 검색 도구 (구글 뉴스 RSS, 비교기업 재무 지표)."""

import json
import re
import statistics
import time
from datetime import date
from functools import lru_cache
from urllib.parse import quote

import feedparser

from config import (
    CACHE_DIR,
    DEFAULT_USD_KRW,
    LARGE_CAP_LIMIT_USD,
    NEWS_MAX_RESULTS,
    PEER_TICKERS,
)


def search_news(query: str, max_results: int = NEWS_MAX_RESULTS, lang: str = "ko") -> list[dict]:
    """구글 뉴스 RSS로 뉴스를 검색합니다. lang="en"이면 영문 뉴스를 검색합니다."""
    locale = "hl=en-US&gl=US&ceid=US:en" if lang == "en" else "hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={quote(query)}&{locale}")
    return [
        {
            "title": entry.get("title", ""),
            "summary": re.sub(r"<[^>]+>", " ", entry.get("summary", ""))[:300],
            "url": entry.get("link", ""),
            "published": entry.get("published", ""),
            "source": entry.get("source", {}).get("title", ""),
        }
        for entry in feed.entries[:max_results]
    ]


def format_news(news: list[dict]) -> str:
    """뉴스 검색 결과를 XML 형식의 문자열로 포매팅합니다."""
    return "\n".join(
        f"<news><title>{n['title']}</title><summary>{n['summary']}</summary>"
        f"<published>{n['published']}</published><source>{n['url']}</source></news>"
        for n in news
    )


@lru_cache(maxsize=1)
def get_usd_krw() -> float:
    """원/달러 환율을 조회합니다. 실패하면 기본값을 사용합니다."""
    try:
        import yfinance as yf

        return float(yf.Ticker("KRW=X").fast_info["last_price"])
    except Exception:
        return DEFAULT_USD_KRW


def get_peer_multiples() -> dict:
    """대기업을 제외한 흑자 반도체 비교기업의 순이익률·PER 중간값을 구합니다.

    결과는 날짜별로 캐시해 같은 날에는 다시 조회하지 않습니다.
    """
    cache_file = CACHE_DIR / f"peer_multiples_{date.today()}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    import yfinance as yf

    peers = []
    for ticker in PEER_TICKERS:
        info = None
        for _ in range(2):  # 일시적 조회 실패에 대비해 한 번 더 시도
            try:
                info = yf.Ticker(ticker).info
                break
            except Exception:
                time.sleep(2)
        if not info:
            continue
        market_cap = info.get("marketCap") or 0
        net_margin = info.get("profitMargins")
        pe = info.get("trailingPE")
        if market_cap >= LARGE_CAP_LIMIT_USD:
            continue  # 대기업 제외
        if not net_margin or net_margin <= 0 or not pe:
            continue  # 흑자 기업만
        peers.append({"ticker": ticker, "market_cap": market_cap, "net_margin": net_margin, "pe": pe})

    if not peers:
        raise RuntimeError("비교기업 데이터를 가져오지 못했습니다. 네트워크를 확인하세요.")

    result = {
        "date": str(date.today()),
        "source": "Yahoo Finance (yfinance), 시가총액 $100B 이상 제외, 흑자 기업만",
        "peers": peers,
        "net_margin_median": statistics.median(p["net_margin"] for p in peers),
        "pe_median": statistics.median(p["pe"] for p in peers),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result

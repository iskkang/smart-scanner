"""
universe_builder.py — 외부 등급 기반 스캔 유니버스 구성
Claude web_search로 주요 사이트 Strong Buy 종목 실시간 수집:
  - Zacks Rank #1 (Strong Buy)
  - Seeking Alpha Quant Strong Buy
  - Morningstar 5-star (Undervalued)
  - Investing.com Strong Buy 컨센서스

수집된 종목들을 합산 → 중복 제거 → 최종 Universe 반환
이 Universe가 chart_scanner의 스캔 대상이 됨
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# 유효 티커 패턴 (1~5자 대문자)
TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

# 제외 단어 (티커처럼 보이지만 아닌 것들)
EXCLUDE_WORDS = {
    "A", "I", "AN", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "CEO", "CFO", "CTO", "ETF", "IPO", "SEC", "USA", "USD", "GDP", "CPI",
    "API", "AI", "PE", "EPS", "ROE", "ROA", "EV", "EBIT", "TTM", "YTD",
    "QOQ", "YOY", "BUY", "SELL", "HOLD", "NYSE", "NASDAQ", "SP", "SMA",
    "RSI", "VIX", "WTI", "DXY", "KST", "UTC", "EST", "PST", "Q1", "Q2",
    "Q3", "Q4", "H1", "H2", "FY", "LTM", "NTM", "DCF", "CAGR", "EPS",
    "STRONG", "TOP", "NEW", "OLD", "THE", "AND", "FOR", "FROM", "WITH",
    "RANK", "STAR", "BEST", "GOOD", "HIGH", "LOW", "ALL", "ANY",
}


# ── 개별 사이트별 검색 프롬프트 ───────────────────────────────

PROMPTS = {
    "zacks": """
web_search로 아래를 검색하세요: "Zacks Rank 1 Strong Buy stocks list today 2026"

Zacks Rank #1 (Strong Buy) 종목들의 티커 심볼 목록을 찾아서 반환하세요.
티커 심볼만 JSON 배열로 반환하세요. 예: ["AAPL", "MSFT", "XOM"]
티커 심볼 외에 다른 텍스트는 일절 없이 JSON 배열만 반환하세요.
찾지 못하면 [] 반환.
""",

    "seeking_alpha": """
web_search로 아래를 검색하세요: "Seeking Alpha Quant Strong Buy stocks list 2026"

Seeking Alpha Quant Rating이 Strong Buy인 종목들의 티커 심볼을 찾아서 반환하세요.
티커 심볼만 JSON 배열로 반환하세요. 예: ["AAPL", "MSFT", "XOM"]
티커 심볼 외에 다른 텍스트는 일절 없이 JSON 배열만 반환하세요.
찾지 못하면 [] 반환.
""",

    "morningstar": """
web_search로 아래를 검색하세요: "Morningstar 5 star undervalued stocks list 2026"

Morningstar 5-star 평가(크게 저평가) 종목들의 티커 심볼을 찾아서 반환하세요.
티커 심볼만 JSON 배열로 반환하세요. 예: ["AAPL", "MSFT", "XOM"]
티커 심볼 외에 다른 텍스트는 일절 없이 JSON 배열만 반환하세요.
찾지 못하면 [] 반환.
""",

    "investing_com": """
web_search로 아래를 검색하세요: "Investing.com technical analysis Strong Buy stocks consensus 2026"

Investing.com에서 기술적 분석 컨센서스가 Strong Buy인 주요 종목들의 티커 심볼을 찾아서 반환하세요.
티커 심볼만 JSON 배열로 반환하세요. 예: ["AAPL", "MSFT", "XOM"]
티커 심볼 외에 다른 텍스트는 일절 없이 JSON 배열만 반환하세요.
찾지 못하면 [] 반환.
""",
}


def search_rated_tickers(source_name: str, prompt: str) -> list:
    """Claude web_search로 특정 사이트의 Strong Buy 종목 수집"""
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY 미설정")
        return []

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "web-search-2025-03-05",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1000,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()

        # text 블록 추출
        content_blocks = resp.json().get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        if not text:
            return []

        # JSON 배열 파싱
        clean = text.strip().removeprefix("```json").removesuffix("```").strip()
        start = clean.find("[")
        end = clean.rfind("]") + 1
        if start >= 0 and end > start:
            tickers = json.loads(clean[start:end])
            # 유효성 필터
            valid = [
                t.upper().replace(".", "-")
                for t in tickers
                if isinstance(t, str)
                and 1 <= len(t.strip()) <= 5
                and t.strip().upper() not in EXCLUDE_WORDS
            ]
            logger.info(f"  [{source_name}] {len(valid)}종목 수집: {valid[:10]}{'...' if len(valid) > 10 else ''}")
            return valid

        # JSON 파싱 실패 시 텍스트에서 티커 추출
        found = TICKER_PATTERN.findall(text)
        valid = [t for t in found if t not in EXCLUDE_WORDS and len(t) >= 2]
        logger.info(f"  [{source_name}] 텍스트 파싱 {len(valid)}종목: {valid[:10]}")
        return list(dict.fromkeys(valid))  # 순서 유지 중복 제거

    except Exception as e:
        logger.error(f"[{source_name}] 수집 실패: {e}")
        return []


# ── 전체 Universe 구성 ─────────────────────────────────────────

def build_rated_universe(min_sources: int = 1) -> dict:
    """
    4개 사이트에서 Strong Buy 종목 수집 후 Universe 구성.

    min_sources: 최소 몇 개 사이트에서 언급되어야 포함할지 (기본 1)
                 2로 설정하면 2개 이상 사이트에서 Strong Buy인 종목만 포함
    """
    logger.info("외부 등급 유니버스 구성 시작")

    source_results = {}
    ticker_sources = {}  # ticker → [source1, source2, ...]

    for source_name, prompt in PROMPTS.items():
        logger.info(f"  {source_name} 검색 중...")
        tickers = search_rated_tickers(source_name, prompt)
        source_results[source_name] = tickers

        for t in tickers:
            ticker_sources.setdefault(t, [])
            if source_name not in ticker_sources[t]:
                ticker_sources[t].append(source_name)

        time.sleep(3)  # API rate limit

    # min_sources 이상 사이트에서 언급된 종목 필터
    universe = [
        t for t, sources in ticker_sources.items()
        if len(sources) >= min_sources
    ]

    # 여러 사이트 언급 순으로 정렬 (신뢰도 높은 것 먼저)
    universe.sort(key=lambda t: len(ticker_sources[t]), reverse=True)

    # 결과 요약
    multi_source = [t for t in universe if len(ticker_sources[t]) >= 2]
    logger.info(f"  유니버스 구성 완료: 총 {len(universe)}종목 (2개+ 사이트: {len(multi_source)}종목)")
    for source, tickers in source_results.items():
        logger.info(f"    {source}: {len(tickers)}종목")

    result = {
        "timestamp": datetime.now().isoformat(),
        "source_results": source_results,
        "ticker_sources": ticker_sources,
        "universe": universe,
        "universe_size": len(universe),
        "multi_source_tickers": multi_source,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/rated_universe.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"유니버스 저장 완료 → data/rated_universe.json")
    return result


def load_or_build_universe(max_age_hours: int = 23) -> list:
    """
    저장된 universe 재사용 (max_age_hours 이내면 재수집 생략).
    당일 이미 수집했으면 재사용.
    """
    cache_path = "data/rated_universe.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            ts = datetime.fromisoformat(cached["timestamp"])
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            if age_hours < max_age_hours:
                universe = cached.get("universe", [])
                logger.info(f"캐시된 유니버스 사용 ({age_hours:.1f}시간 전, {len(universe)}종목)")
                return universe
        except Exception:
            pass

    result = build_rated_universe()
    return result["universe"]


if __name__ == "__main__":
    result = build_rated_universe(min_sources=1)

    print(f"\n=== 외부 등급 유니버스 ===")
    for source, tickers in result["source_results"].items():
        print(f"[{source}] {len(tickers)}종목: {tickers[:5]}...")

    print(f"\n최종 유니버스: {result['universe_size']}종목")
    print(f"2개+ 사이트 언급: {len(result['multi_source_tickers'])}종목")
    print(f"  → {result['multi_source_tickers'][:20]}")

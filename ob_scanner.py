"""
universe_builder.py — Strong Buy 유니버스 구성 (4개 소스)

소스별 방법:
  1. Zacks       → Claude web_search ("Zacks Rank 1 Strong Buy stocks")
  2. Finviz      → 스크리너 직접 요청 (an_recom_strongbuy 필터, 무료)
  3. Morningstar → Claude web_search ("Morningstar 5-star undervalued stocks")
  4. yfinance    → S&P500 대형주 중 recommendationMean ≤ 2.0 (Buy 이상)

합산 → 중복 제거 → Universe 반환
"""

import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')
EXCLUDE_WORDS = {
    "A","I","AN","AT","BE","BY","DO","GO","IF","IN","IS","IT","ME","MY",
    "NO","OF","ON","OR","SO","TO","UP","US","WE","CEO","CFO","CTO","ETF",
    "IPO","SEC","USA","USD","GDP","CPI","API","AI","PE","EPS","ROE","ROA",
    "EV","EBIT","TTM","YTD","QOQ","YOY","BUY","SELL","HOLD","NYSE","NASDAQ",
    "SP","SMA","RSI","VIX","WTI","DXY","KST","UTC","EST","PST","Q1","Q2",
    "Q3","Q4","H1","H2","FY","LTM","NTM","DCF","CAGR","STRONG","TOP","NEW",
    "OLD","THE","AND","FOR","FROM","WITH","RANK","STAR","BEST","HIGH","LOW",
    "ALL","ANY","GET","SET","PUT","INC","LLC","LTD","PLC","NV","SE","SA",
    "CO","AG","AB","AS","DE","LA","LE","EL","DI","IL","UN","RE","BI","EX",
}


# ═══════════════════════════════════════════════════════════════
# 소스 1: Zacks — Claude web_search
# ═══════════════════════════════════════════════════════════════

def fetch_zacks() -> list:
    """Zacks Rank #1 Strong Buy 종목 수집 (Claude web_search)"""
    if not ANTHROPIC_API_KEY:
        return []

    prompt = """web_search로 검색하세요: "Zacks Rank 1 Strong Buy stocks list today 2026"

Zacks Rank #1 (Strong Buy) 종목들의 티커 심볼 목록을 찾아 JSON 배열로만 반환하세요.
예: ["AAPL", "MSFT", "XOM"]
JSON 배열 외 다른 텍스트 없이. 못 찾으면 [] 반환."""

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
                "max_tokens": 800,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=45,
        )
        resp.raise_for_status()
        text = "".join(
            b.get("text", "") for b in resp.json().get("content", [])
            if b.get("type") == "text"
        )
        return _parse_tickers_from_text(text)
    except Exception as e:
        logger.error(f"Zacks 수집 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# 소스 2: Finviz — 스크리너 직접 요청
# ═══════════════════════════════════════════════════════════════

def fetch_finviz() -> list:
    """
    Finviz 스크리너에서 Strong Buy 컨센서스 종목 수집.
    URL: https://finviz.com/screener.ashx?v=111&f=an_recom_strongbuy&o=-marketcap
    """
    tickers = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/",
    }

    try:
        # 페이지당 20종목, 최대 3페이지(60종목) 수집
        for page_offset in [1, 21, 41]:
            url = (
                f"https://finviz.com/screener.ashx"
                f"?v=111&f=an_recom_strongbuy&o=-marketcap&r={page_offset}"
            )
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Finviz 응답 {resp.status_code} (offset={page_offset})")
                break

            tables = pd.read_html(io.StringIO(resp.text))
            # Finviz 스크리너 테이블 찾기 (Ticker 컬럼 포함)
            for table in tables:
                cols = [str(c).strip() for c in table.columns]
                if "Ticker" in cols:
                    batch = table["Ticker"].dropna().tolist()
                    tickers.extend([str(t).upper() for t in batch if 1 <= len(str(t)) <= 5])
                    break

            time.sleep(1.5)  # 요청 간격

        tickers = list(dict.fromkeys(tickers))  # 순서 유지 중복 제거
        logger.info(f"  [finviz] {len(tickers)}종목 수집: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")
        return tickers

    except Exception as e:
        logger.error(f"Finviz 수집 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# 소스 3: Morningstar — Claude web_search
# ═══════════════════════════════════════════════════════════════

def fetch_morningstar() -> list:
    """Morningstar 5-star 저평가 종목 수집 (Claude web_search)"""
    if not ANTHROPIC_API_KEY:
        return []

    prompt = """web_search로 검색하세요: "Morningstar 5 star undervalued stocks list 2026"

Morningstar 5-star(크게 저평가) 종목들의 티커 심볼을 JSON 배열로만 반환하세요.
예: ["AAPL", "MSFT", "XOM"]
JSON 배열 외 다른 텍스트 없이. 못 찾으면 [] 반환."""

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
                "max_tokens": 800,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=45,
        )
        resp.raise_for_status()
        text = "".join(
            b.get("text", "") for b in resp.json().get("content", [])
            if b.get("type") == "text"
        )
        return _parse_tickers_from_text(text)
    except Exception as e:
        logger.error(f"Morningstar 수집 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# 소스 4: yfinance — S&P500 컨센서스 스크리닝
# ═══════════════════════════════════════════════════════════════

def fetch_yfinance_consensus() -> list:
    """
    S&P500 대형주 중 yfinance recommendationMean ≤ 2.0 (Buy 이상) 종목 수집.
    병렬 처리로 속도 최적화.
    """
    # S&P500 티커 수집
    sp500 = _fetch_sp500_tickers()
    if not sp500:
        logger.warning("S&P500 티커 수집 실패 — yfinance 소스 스킵")
        return []

    logger.info(f"  [yfinance] S&P500 {len(sp500)}종목 컨센서스 스크리닝...")

    strong_buys = []

    def check_consensus(ticker):
        try:
            info = yf.Ticker(ticker).info or {}
            rec = info.get("recommendationMean")
            n   = info.get("numberOfAnalystOpinions", 0) or 0
            # 애널리스트 3명 이상 + mean ≤ 2.0 (Strong Buy~Buy)
            if rec and n >= 3 and rec <= 2.0:
                return ticker
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(check_consensus, t): t for t in sp500}
        for future in as_completed(futures):
            result = future.result()
            if result:
                strong_buys.append(result)

    logger.info(f"  [yfinance] {len(strong_buys)}종목 수집 (recommendationMean ≤ 2.0)")
    return strong_buys


def _fetch_sp500_tickers() -> list:
    """Wikipedia에서 S&P500 티커 수집"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        logger.error(f"S&P500 수집 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# 텍스트 파싱 유틸
# ═══════════════════════════════════════════════════════════════

def _parse_tickers_from_text(text: str) -> list:
    """텍스트에서 티커 추출 (JSON 배열 우선, 실패 시 정규식)"""
    if not text:
        return []
    clean = text.strip().removeprefix("```json").removesuffix("```").strip()

    # JSON 배열 파싱 시도
    start = clean.find("[")
    end   = clean.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            items = json.loads(clean[start:end])
            valid = [
                str(t).upper().replace(".", "-")
                for t in items
                if isinstance(t, str) and 1 <= len(str(t).strip()) <= 5
                and str(t).strip().upper() not in EXCLUDE_WORDS
            ]
            return valid
        except Exception:
            pass

    # 정규식 폴백
    found = TICKER_PATTERN.findall(text)
    return list(dict.fromkeys(
        t for t in found if t not in EXCLUDE_WORDS and len(t) >= 2
    ))


# ═══════════════════════════════════════════════════════════════
# Universe 통합 구성
# ═══════════════════════════════════════════════════════════════

def build_rated_universe(min_sources: int = 1) -> dict:
    """
    4개 소스에서 Strong Buy 종목 수집 후 Universe 구성.
    min_sources: 최소 언급 소스 수 (1 = 하나라도 있으면 포함)
    """
    logger.info("외부 등급 유니버스 구성 시작")

    source_results = {}
    ticker_sources = {}  # ticker → [source, ...]

    # ── 순차 실행 (API rate limit 고려) ──
    sources = [
        ("zacks",     fetch_zacks),
        ("finviz",    fetch_finviz),
        ("morningstar", fetch_morningstar),
        ("yfinance",  fetch_yfinance_consensus),
    ]

    for name, fn in sources:
        logger.info(f"  [{name}] 수집 중...")
        try:
            tickers = fn()
        except Exception as e:
            logger.error(f"  [{name}] 실패: {e}")
            tickers = []

        source_results[name] = tickers
        logger.info(f"  [{name}] {len(tickers)}종목: {tickers[:8]}{'...' if len(tickers) > 8 else ''}")

        for t in tickers:
            ticker_sources.setdefault(t, [])
            if name not in ticker_sources[t]:
                ticker_sources[t].append(name)

        time.sleep(2)

    # min_sources 이상 언급 종목 필터
    universe = [
        t for t, srcs in ticker_sources.items()
        if len(srcs) >= min_sources
    ]
    # 다중 소스 언급 순으로 정렬 (신뢰도 높은 것 먼저)
    universe.sort(key=lambda t: len(ticker_sources[t]), reverse=True)

    multi = [t for t in universe if len(ticker_sources[t]) >= 2]
    logger.info(
        f"유니버스 완료: 총 {len(universe)}종목 | "
        f"2개+ 소스: {len(multi)}종목 → {multi[:15]}"
    )
    for name, tickers in source_results.items():
        logger.info(f"  {name}: {len(tickers)}종목")

    result = {
        "timestamp": datetime.now().isoformat(),
        "source_results": source_results,
        "ticker_sources": ticker_sources,
        "universe": universe,
        "universe_size": len(universe),
        "multi_source_tickers": multi,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/rated_universe.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def load_or_build_universe(max_age_hours: int = 23) -> list:
    """당일 캐시 재사용 (재수집 불필요 시). 없으면 새로 구성."""
    cache_path = "data/rated_universe.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            age_h = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds() / 3600
            if age_h < max_age_hours:
                universe = cached.get("universe", [])
                logger.info(f"캐시 유니버스 사용: {len(universe)}종목 ({age_h:.1f}시간 전)")
                return universe
        except Exception:
            pass

    return build_rated_universe()["universe"]


if __name__ == "__main__":
    result = build_rated_universe()
    print(f"\n=== 유니버스 결과 ===")
    for src, tickers in result["source_results"].items():
        print(f"[{src:12s}] {len(tickers):3d}종목: {tickers[:5]}")
    print(f"\n최종 유니버스: {result['universe_size']}종목")
    print(f"2개+ 소스:    {len(result['multi_source_tickers'])}종목 → {result['multi_source_tickers'][:20]}")

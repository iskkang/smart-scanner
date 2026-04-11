"""
MODULE 5: 뉴스 및 공매도 검증 (news_analyzer.py)
- NewsAPI: 종목당 최근 10개 헤드라인
- Yahoo Finance: 공매도 데이터 (shortPercentOfFloat, shortRatio)
- Claude Sonnet: 감성 분석, 성장 스토리 검증, 리스크 판단
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── 뉴스 수집 ─────────────────────────────────────────────────

def fetch_news(ticker: str, max_articles: int = 10) -> list:
    """NewsAPI에서 종목 관련 최근 헤드라인 수집"""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY 미설정 — 뉴스 수집 스킵")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "published": a.get("publishedAt", "")[:10],
                "description": (a.get("description") or "")[:200],
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        logger.error(f"{ticker} 뉴스 수집 실패: {e}")
        return []


# ── 공매도 데이터 ──────────────────────────────────────────────

def fetch_short_data(ticker: str) -> dict:
    """Yahoo Finance에서 공매도 관련 데이터 수집"""
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "short_pct_float": info.get("shortPercentOfFloat"),
            "short_ratio": info.get("shortRatio"),
            "shares_short": info.get("sharesShort"),
            "shares_short_prior": info.get("sharesShortPriorMonth"),
        }
    except Exception as e:
        logger.error(f"{ticker} 공매도 데이터 수집 실패: {e}")
        return {}


# ── Claude Sonnet 분석 ────────────────────────────────────────

ANALYSIS_PROMPT = """당신은 월가 시니어 리서치 애널리스트입니다.
아래 종목의 뉴스 헤드라인과 공매도 데이터를 분석하고, JSON만 반환하세요. 설명 없이.

ticker: {ticker}

headlines:
{headlines}

short_interest:
{short_data}

반환 형식:
{{
  "sentiment": "BULLISH|NEUTRAL|BEARISH",
  "sentiment_score": -100에서 100 사이 정수,
  "short_risk": "LOW|MEDIUM|HIGH",
  "growth_story_valid": true 또는 false,
  "key_risks": ["리스크1", "리스크2"],
  "key_positives": ["긍정1", "긍정2"],
  "summary": "2문장 한국어 요약",
  "pass": true 또는 false
}}

pass = false 조건:
- sentiment == "BEARISH"
- short_risk == "HIGH"
- growth_story_valid == false
위 중 하나라도 해당하면 pass = false
"""


def analyze_with_claude(ticker: str, headlines: list, short_data: dict) -> Optional[dict]:
    """Claude SDK로 뉴스 감성 + 공매도 리스크 분석"""
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY 미설정 — Claude 뉴스 분석 스킵")
        return None

    headlines_text = "\n".join(
        [f"- [{h['source']}] {h['title']} ({h['published']})" for h in headlines]
    ) if headlines else "(뉴스 없음)"

    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker,
        headlines=headlines_text,
        short_data=json.dumps(short_data, indent=2),
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0].text
        clean = content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except Exception as e:
        logger.error(f"{ticker} Claude 뉴스 분석 실패: {e}")
        return None


# ── 폴백: 규칙 기반 분석 (Claude 실패 시) ──────────────────────

def fallback_analysis(ticker: str, short_data: dict) -> dict:
    """Claude 호출 실패 시 공매도 데이터 기반 규칙 판단"""
    short_pct = short_data.get("short_pct_float")
    short_ratio = short_data.get("short_ratio")

    short_risk = "LOW"
    warnings = []

    if short_pct is not None:
        if short_pct > 0.20:
            short_risk = "HIGH"
            warnings.append("공매도 비율 20% 초과")
        elif short_pct > 0.10:
            short_risk = "MEDIUM"
            warnings.append("공매도 비율 10% 초과")

    if short_ratio is not None and short_ratio > 5:
        if short_risk != "HIGH":
            short_risk = "MEDIUM"
        warnings.append(f"공매도 커버 {short_ratio:.1f}일")

    # 공매도 증가 추세
    current = short_data.get("shares_short")
    prior = short_data.get("shares_short_prior")
    if current and prior and current > prior * 1.2:
        warnings.append("공매도 전월 대비 20%+ 증가")
        if short_risk == "LOW":
            short_risk = "MEDIUM"

    passed = short_risk != "HIGH"

    return {
        "sentiment": "NEUTRAL",
        "sentiment_score": 0,
        "short_risk": short_risk,
        "growth_story_valid": True,  # 뉴스 없이 판단 불가 — 보수적으로 통과
        "key_risks": warnings,
        "key_positives": [],
        "summary": "Claude 분석 불가 — 공매도 데이터 기반 규칙 판단 적용",
        "pass": passed,
        "analysis_method": "FALLBACK_RULE_BASED",
    }


# ── 개별 종목 분석 ─────────────────────────────────────────────

def analyze_ticker(ticker: str) -> dict:
    """뉴스 + 공매도 종합 분석"""
    headlines = fetch_news(ticker)
    short_data = fetch_short_data(ticker)

    # Claude 분석 시도
    ai_result = analyze_with_claude(ticker, headlines, short_data)

    if ai_result:
        ai_result["analysis_method"] = "CLAUDE_SONNET"
        ai_result["headline_count"] = len(headlines)
        ai_result["short_raw"] = short_data
        ai_result["ticker"] = ticker
        return ai_result

    # 폴백
    logger.warning(f"{ticker} Claude 분석 실패 — 폴백 규칙 적용")
    fb = fallback_analysis(ticker, short_data)
    fb["headline_count"] = len(headlines)
    fb["short_raw"] = short_data
    fb["ticker"] = ticker
    return fb


# ── 배치 실행 ──────────────────────────────────────────────────

def run_news_analysis(tickers: list) -> list:
    """기관 분석 통과 종목에 대해 뉴스/공매도 검증"""
    logger.info(f"뉴스/공매도 분석 시작 — {len(tickers)}종목")

    results = []
    for ticker in tickers:
        result = analyze_ticker(ticker)
        status = "✅ 통과" if result.get("pass") else "❌ 탈락"
        logger.info(
            f"  {status} {ticker}: {result.get('sentiment')} "
            f"(점수 {result.get('sentiment_score')}) | "
            f"공매도 {result.get('short_risk')} | "
            f"성장스토리 {result.get('growth_story_valid')}"
        )
        results.append(result)

    passed = [r for r in results if r.get("pass")]

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "passed_count": len(passed),
        "passed": passed,
        "failed": [r for r in results if not r.get("pass")],
    }

    os.makedirs("data", exist_ok=True)
    with open("data/news_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"뉴스/공매도 분석 완료 — {len(passed)}/{len(results)} 통과")
    return passed


if __name__ == "__main__":
    inst_path = "data/institutional_analysis.json"
    if os.path.exists(inst_path):
        with open(inst_path, "r") as f:
            inst_data = json.load(f)
        tickers = [r["ticker"] for r in inst_data.get("passed", [])]
        logger.info(f"기관 분석 통과 종목에서 {len(tickers)}종목 로드")
    else:
        tickers = ["AAPL", "MSFT", "NVDA"]
        logger.info("기관 분석 결과 없음 — 테스트 종목 사용")

    passed = run_news_analysis(tickers)
    print(f"\n뉴스/공매도 통과 {len(passed)}종목:")
    for r in passed:
        print(f"  {r['ticker']:6s} | {r['sentiment']} ({r['sentiment_score']}) | 공매도 {r['short_risk']}")

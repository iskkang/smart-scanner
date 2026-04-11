"""
MODULE 6: 월가 리포트 교차검증 (wallstreet_report.py)
- NewsAPI: 최근 7일 애널리스트 등급 변경 검색
- yfinance: recommendationKey, upgrades/downgrades
- 점수 산출 및 즉시 탈락 조건 적용
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# 등급 분류 매핑
BULLISH_GRADES = {"Buy", "Strong Buy", "Outperform", "Overweight", "Positive", "Market Outperform", "Top Pick"}
NEUTRAL_GRADES = {"Neutral", "Equal Weight", "Hold", "Market Perform", "Sector Perform", "In-Line", "Peer Perform"}
BEARISH_GRADES = {"Sell", "Strong Sell", "Underperform", "Underweight", "Negative", "Reduce"}


def fetch_yf_recommendations(ticker: str) -> dict:
    """yfinance에서 추천 등급 및 최근 업그레이드/다운그레이드 수집"""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # 현재 컨센서스
        rec_key = info.get("recommendationKey", "")
        rec_mean = info.get("recommendationMean")  # 1=Strong Buy ~ 5=Sell

        # 최근 등급 변경 (upgrades_downgrades)
        try:
            upgrades_df = t.upgrades_downgrades
            recent_changes = []
            if upgrades_df is not None and not upgrades_df.empty:
                cutoff = datetime.now() - timedelta(days=7)
                for idx, row in upgrades_df.iterrows():
                    try:
                        if hasattr(idx, 'date') or hasattr(idx, 'year'):
                            row_date = idx
                        else:
                            row_date = datetime.strptime(str(idx)[:10], "%Y-%m-%d")

                        if hasattr(row_date, 'tz') and row_date.tz:
                            row_date = row_date.replace(tzinfo=None)

                        if row_date >= cutoff:
                            recent_changes.append({
                                "date": str(row_date)[:10],
                                "firm": str(row.get("Firm", "")),
                                "to_grade": str(row.get("ToGrade", "")),
                                "from_grade": str(row.get("FromGrade", "")),
                                "action": str(row.get("Action", "")),
                            })
                    except Exception:
                        continue
        except Exception:
            recent_changes = []

        return {
            "recommendation_key": rec_key,
            "recommendation_mean": rec_mean,
            "recent_changes": recent_changes[-10:],  # 최근 10건
        }
    except Exception as e:
        logger.error(f"{ticker} yfinance 추천 데이터 수집 실패: {e}")
        return {"recommendation_key": "", "recommendation_mean": None, "recent_changes": []}


def fetch_news_ratings(ticker: str) -> list:
    """NewsAPI에서 애널리스트 등급 변경 뉴스 검색 (보조)"""
    if not NEWS_API_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} analyst rating upgrade downgrade",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
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
            }
            for a in articles if a.get("title")
        ]
    except Exception as e:
        logger.error(f"{ticker} 뉴스 등급 검색 실패: {e}")
        return []


def classify_grade(grade: str) -> str:
    """등급을 BULLISH/NEUTRAL/BEARISH로 분류"""
    grade_clean = grade.strip().title()
    if grade_clean in BULLISH_GRADES:
        return "BULLISH"
    if grade_clean in NEUTRAL_GRADES:
        return "NEUTRAL"
    if grade_clean in BEARISH_GRADES:
        return "BEARISH"
    # 부분 매칭
    gl = grade.lower()
    if any(w in gl for w in ["buy", "outperform", "overweight", "positive"]):
        return "BULLISH"
    if any(w in gl for w in ["sell", "underperform", "underweight", "negative", "reduce"]):
        return "BEARISH"
    return "NEUTRAL"


def score_wallstreet(ticker: str) -> dict:
    """
    월가 리포트 교차검증 점수 산출.

    Buy/Strong Buy 상향: +20
    Neutral 하향: -15
    Sell/Underperform: 즉시 탈락
    3개+ 증권사 Buy 유지: +10 추가
    """
    yf_data = fetch_yf_recommendations(ticker)
    news_ratings = fetch_news_ratings(ticker)

    score = 0
    signals = []
    warnings = []
    instant_fail = False

    changes = yf_data.get("recent_changes", [])

    bullish_count = 0
    bearish_count = 0

    for change in changes:
        to_class = classify_grade(change.get("to_grade", ""))
        from_class = classify_grade(change.get("from_grade", ""))
        firm = change.get("firm", "")

        if to_class == "BULLISH" and from_class != "BULLISH":
            # 상향
            score += 20
            signals.append(f"UPGRADE: {firm} → {change['to_grade']}")
            bullish_count += 1
        elif to_class == "BULLISH":
            # Buy 유지/재확인
            bullish_count += 1
        elif to_class == "NEUTRAL" and from_class == "BULLISH":
            # 하향
            score -= 15
            warnings.append(f"DOWNGRADE: {firm} → {change['to_grade']}")
        elif to_class == "BEARISH":
            # 즉시 탈락
            instant_fail = True
            warnings.append(f"SELL_RATING: {firm} → {change['to_grade']}")

    # 3개+ 증권사 Buy
    if bullish_count >= 3:
        score += 10
        signals.append(f"MULTI_BUY: {bullish_count}개 증권사 Buy")

    # 컨센서스 보조 판단
    rec_mean = yf_data.get("recommendation_mean")
    if rec_mean is not None:
        if rec_mean <= 2.0:
            signals.append(f"CONSENSUS_STRONG_BUY (mean={rec_mean})")
        elif rec_mean >= 4.0:
            warnings.append(f"CONSENSUS_BEARISH (mean={rec_mean})")
            if not instant_fail:
                score -= 10

    passed = (not instant_fail) and (score >= -10)

    return {
        "ticker": ticker,
        "ws_score": max(score, 0),
        "recommendation_key": yf_data.get("recommendation_key"),
        "recommendation_mean": rec_mean,
        "recent_changes": changes,
        "news_headlines": news_ratings,
        "signals": signals,
        "warnings": warnings,
        "pass": passed,
        "fail_reason": "Sell 등급 탈락" if instant_fail else (None if passed else f"점수 {score}"),
    }


# ── 배치 실행 ──────────────────────────────────────────────────

def run_wallstreet_analysis(tickers: list) -> list:
    """뉴스 분석 통과 종목에 대해 월가 리포트 교차검증"""
    logger.info(f"월가 리포트 교차검증 시작 — {len(tickers)}종목")

    results = []
    for ticker in tickers:
        result = score_wallstreet(ticker)
        status = "✅ 통과" if result["pass"] else "❌ 탈락"
        logger.info(
            f"  {status} {ticker}: 점수 {result['ws_score']} | "
            f"컨센서스 {result['recommendation_key']} | {result['signals']}"
        )
        results.append(result)

    passed = [r for r in results if r["pass"]]

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "passed_count": len(passed),
        "passed": passed,
        "failed": [r for r in results if not r["pass"]],
    }

    os.makedirs("data", exist_ok=True)
    with open("data/wallstreet_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"월가 리포트 검증 완료 — {len(passed)}/{len(results)} 통과")
    return passed


if __name__ == "__main__":
    news_path = "data/news_analysis.json"
    if os.path.exists(news_path):
        with open(news_path, "r") as f:
            news_data = json.load(f)
        tickers = [r["ticker"] for r in news_data.get("passed", [])]
        logger.info(f"뉴스 분석 통과 종목에서 {len(tickers)}종목 로드")
    else:
        tickers = ["AAPL", "MSFT", "NVDA"]
        logger.info("뉴스 분석 결과 없음 — 테스트 종목 사용")

    passed = run_wallstreet_analysis(tickers)
    print(f"\n월가 검증 통과 {len(passed)}종목:")
    for r in passed:
        print(f"  {r['ticker']:6s} | 점수 {r['ws_score']} | {r['recommendation_key']} | {r['signals']}")

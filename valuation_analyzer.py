"""
MODULE 3: 밸류에이션 분석 (valuation_analyzer.py)
- PE Z-score (5년 히스토리 기반)
- 애널리스트 목표가 괴리율
- Forward PE, PEG, P/S, P/B 수집
- 통과 기준: 밸류에이션 점수 30점 이상 + 경고 1개 이하
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import yfinance as yf
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_valuation_data(ticker: str) -> Optional[dict]:
    """yfinance에서 밸류에이션 관련 데이터 수집"""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        return {
            "ticker": ticker,
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "trailing_eps": info.get("trailingEps"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        logger.error(f"{ticker} 밸류에이션 데이터 수집 실패: {e}")
        return None


def calc_pe_zscore(ticker: str, trailing_eps: float = None) -> Optional[dict]:
    """
    5년 월별 종가 기반 PE 히스토리 → Z-score 계산.
    trailing_eps가 없으면 스킵.
    """
    if not trailing_eps or trailing_eps <= 0:
        return None

    try:
        hist = yf.Ticker(ticker).history(period="5y", interval="1mo")
        if len(hist) < 24:
            return None

        monthly_close = hist["Close"].dropna()
        pe_history = monthly_close / trailing_eps
        pe_history = pe_history.replace([np.inf, -np.inf], np.nan).dropna()

        if len(pe_history) < 12:
            return None

        current_pe = float(pe_history.iloc[-1])
        mean_pe = float(pe_history.mean())
        std_pe = float(pe_history.std())

        if std_pe == 0:
            return None

        zscore = (current_pe - mean_pe) / std_pe

        return {
            "current_pe": round(current_pe, 2),
            "mean_pe_5y": round(mean_pe, 2),
            "std_pe_5y": round(std_pe, 2),
            "pe_zscore": round(zscore, 2),
            "data_points": len(pe_history),
        }
    except Exception as e:
        logger.error(f"{ticker} PE Z-score 계산 실패: {e}")
        return None


def calc_target_gap(current_price: float, target_mean: float) -> Optional[float]:
    """애널리스트 평균 목표가 괴리율 (%)"""
    if not current_price or not target_mean or current_price <= 0:
        return None
    return round((target_mean - current_price) / current_price * 100, 2)


def score_valuation(ticker: str) -> Optional[dict]:
    """
    밸류에이션 종합 점수 산출.

    PE Z-score 점수:
      -1.5σ 이하: +35
      -1.0σ 이하: +25
      +1.5σ 이상: 탈락 (고평가)

    목표가 괴리율 점수:
      40% 이상: +35
      25% 이상: +25
      현재가 > 목표가: 감점, 탈락 후보

    통과: 점수 30 이상 + 경고 1개 이하
    """
    val_data = fetch_valuation_data(ticker)
    if not val_data:
        return None

    score = 0
    warnings = []
    details = {}

    # ── PE Z-score ──
    pe_result = calc_pe_zscore(ticker, val_data.get("trailing_eps"))
    if pe_result:
        details["pe_zscore"] = pe_result
        z = pe_result["pe_zscore"]

        if z >= 1.5:
            warnings.append("PE_Z_HIGH_OVERVALUED")
            details["pe_zscore_verdict"] = "고평가 — 탈락"
            return {
                **val_data,
                **details,
                "val_score": 0,
                "warnings": warnings,
                "pass": False,
                "fail_reason": "PE Z-score +1.5σ 이상 고평가",
            }
        elif z <= -1.5:
            score += 35
            details["pe_zscore_verdict"] = "극단 저평가 (+35)"
        elif z <= -1.0:
            score += 25
            details["pe_zscore_verdict"] = "저평가 (+25)"
        else:
            details["pe_zscore_verdict"] = "적정 범위"
    else:
        details["pe_zscore"] = None
        details["pe_zscore_verdict"] = "데이터 부족"

    # ── 목표가 괴리율 ──
    current = val_data.get("current_price")
    target = val_data.get("target_mean_price")
    gap = calc_target_gap(current, target)
    details["target_gap_pct"] = gap

    if gap is not None:
        if gap >= 40:
            score += 35
            details["target_verdict"] = "극단 저평가 (+35)"
        elif gap >= 25:
            score += 25
            details["target_verdict"] = "저평가 (+25)"
        elif gap < 0:
            score -= 10
            warnings.append("PRICE_ABOVE_TARGET")
            details["target_verdict"] = "현재가 > 목표가 (-10, 탈락 후보)"
        else:
            details["target_verdict"] = "적정 범위"
    else:
        details["target_verdict"] = "데이터 없음"

    # ── 보조 경고 ──
    peg = val_data.get("peg_ratio")
    if peg is not None and peg > 3.0:
        warnings.append("PEG_HIGH")

    fwd_pe = val_data.get("forward_pe")
    if fwd_pe is not None and fwd_pe > 60:
        warnings.append("FORWARD_PE_EXTREME")

    pb = val_data.get("price_to_book")
    if pb is not None and pb > 20:
        warnings.append("PB_EXTREME")

    analyst_n = val_data.get("analyst_count")
    if analyst_n is not None and analyst_n < 3:
        warnings.append("LOW_ANALYST_COVERAGE")

    # ── 통과 판정 ──
    passed = (score >= 30) and (len(warnings) <= 1)

    return {
        **val_data,
        **details,
        "val_score": max(score, 0),
        "warnings": warnings,
        "pass": passed,
        "fail_reason": None if passed else f"점수 {score} / 경고 {len(warnings)}개",
    }


# ── 배치 실행 ──────────────────────────────────────────────────

def run_valuation_analysis(tickers: list) -> list:
    """차트 스캔 통과 종목들에 대해 밸류에이션 분석 실행"""
    logger.info(f"밸류에이션 분석 시작 — {len(tickers)}종목")

    results = []
    for ticker in tickers:
        result = score_valuation(ticker)
        if result:
            status = "✅ 통과" if result["pass"] else "❌ 탈락"
            logger.info(f"  {status} {ticker}: 점수 {result['val_score']} | 경고 {result['warnings']}")
            results.append(result)

    passed = [r for r in results if r["pass"]]
    failed = [r for r in results if not r["pass"]]

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "passed": passed,
        "failed": failed,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/valuation_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"밸류에이션 분석 완료 — {len(passed)}/{len(results)} 통과 (data/valuation_analysis.json)")
    return passed


if __name__ == "__main__":
    # 차트 스캔 결과에서 종목 로드, 없으면 테스트용 샘플
    chart_path = "data/chart_scan.json"
    if os.path.exists(chart_path):
        with open(chart_path, "r") as f:
            chart_data = json.load(f)
        tickers = [r["ticker"] for r in chart_data.get("results", [])]
        logger.info(f"차트 스캔 결과에서 {len(tickers)}종목 로드")
    else:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
        logger.info("차트 스캔 결과 없음 — 테스트 종목 사용")

    passed = run_valuation_analysis(tickers)
    print(f"\n밸류에이션 통과 {len(passed)}종목:")
    for r in passed:
        z_info = r.get("pe_zscore", {})
        z_val = z_info.get("pe_zscore", "N/A") if z_info else "N/A"
        print(f"  {r['ticker']:6s} | 점수 {r['val_score']:3d} | Z-score {z_val} | 목표가괴리 {r.get('target_gap_pct', 'N/A')}%")

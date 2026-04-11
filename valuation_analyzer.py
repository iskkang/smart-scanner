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
    밸류에이션 필터 — "극단 고평가 차단" 용도로만 사용.

    설계 원칙:
      차트 스캔이 이미 이평선 정배열 + 눌림목을 검증함.
      밸류에이션은 명백한 버블/고평가 종목을 거르는 역할만 담당.
      데이터 부족 또는 적정 범위이면 기본값 통과.

    즉시 탈락 조건 (hard_fail):
      - 현재가가 애널리스트 평균 목표가 대비 10% 이상 고평가  (목표가 < 현재가 × 0.90)
      - PE Z-score > 2.5σ  AND  Forward PE > 40 동시 충족 (버블 신호 복합)

    점수 (참고용, 통과 기준에는 사용 안 함):
      목표가 괴리 35%+ : +35
      목표가 괴리 20%+ : +25
      목표가 괴리 10%+ : +15
      목표가 괴리  0%+ : +5
      PE Z-score ≤ -1.5σ : +20
      PE Z-score ≤ -1.0σ : +10
    """
    val_data = fetch_valuation_data(ticker)
    if not val_data:
        return None

    score = 0
    warnings = []
    details = {}
    hard_fail = False
    fail_reason = None

    # ── PE Z-score ──
    pe_result = calc_pe_zscore(ticker, val_data.get("trailing_eps"))
    fwd_pe = val_data.get("forward_pe")

    if pe_result:
        details["pe_zscore"] = pe_result
        z = pe_result["pe_zscore"]

        if z > 2.5 and fwd_pe and fwd_pe > 40:
            hard_fail = True
            fail_reason = f"PE 버블 신호: Z-score {z:.1f}σ + Forward PE {fwd_pe:.0f}"
            details["pe_zscore_verdict"] = f"버블 구간 탈락 (Z={z:.1f}, FwdPE={fwd_pe:.0f})"
        elif z > 2.5:
            warnings.append("PE_Z_EXTREME")
            details["pe_zscore_verdict"] = f"극단 고평가 경고 (Z={z:.1f})"
        elif z <= -1.5:
            score += 20
            details["pe_zscore_verdict"] = f"저평가 (+20, Z={z:.1f})"
        elif z <= -1.0:
            score += 10
            details["pe_zscore_verdict"] = f"완만 저평가 (+10, Z={z:.1f})"
        else:
            details["pe_zscore_verdict"] = f"적정 (Z={z:.1f})"
    else:
        details["pe_zscore"] = None
        details["pe_zscore_verdict"] = "데이터 없음"

    # ── 목표가 괴리율 ──
    current = val_data.get("current_price")
    target = val_data.get("target_mean_price")
    analyst_n = val_data.get("analyst_count") or 0
    gap = calc_target_gap(current, target)
    details["target_gap_pct"] = gap

    if gap is not None and analyst_n >= 2:
        if gap <= -10:
            # 현재가가 목표가보다 10%+ 높음 → 즉시 탈락
            hard_fail = True
            fail_reason = fail_reason or f"현재가 목표가 대비 {abs(gap):.1f}% 초과"
            details["target_verdict"] = f"목표가 대폭 초과 탈락 (gap={gap:.1f}%)"
        elif gap < 0:
            warnings.append("PRICE_ABOVE_TARGET")
            details["target_verdict"] = f"현재가 > 목표가 경고 (gap={gap:.1f}%)"
        elif gap >= 35:
            score += 35
            details["target_verdict"] = f"극단 저평가 (+35, gap={gap:.1f}%)"
        elif gap >= 20:
            score += 25
            details["target_verdict"] = f"저평가 (+25, gap={gap:.1f}%)"
        elif gap >= 10:
            score += 15
            details["target_verdict"] = f"완만 저평가 (+15, gap={gap:.1f}%)"
        else:
            score += 5
            details["target_verdict"] = f"소폭 상승 여지 (+5, gap={gap:.1f}%)"
    elif gap is not None:
        details["target_verdict"] = f"애널리스트 커버리지 부족 (n={analyst_n})"
    else:
        details["target_verdict"] = "목표가 데이터 없음"

    # ── 보조 경고 (탈락 없음) ──
    peg = val_data.get("peg_ratio")
    if peg is not None and peg > 4.0:
        warnings.append("PEG_EXTREME")

    # ── 통과 판정: hard_fail 없으면 기본 통과 ──
    passed = not hard_fail

    return {
        **val_data,
        **details,
        "val_score": max(score, 0),
        "warnings": warnings,
        "pass": passed,
        "fail_reason": fail_reason,
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

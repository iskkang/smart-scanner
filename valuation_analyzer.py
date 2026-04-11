"""
MODULE 3: 밸류에이션 분석 (valuation_analyzer.py)

[섹터별 최적 지표 적용 — 우리투자증권 10년 연구 기반]
  에너지·소재   → PER 저평가 (연평균 +27%p 초과수익)
  IT·기술       → Forward EPS 성장 추세 (연평균 +25%p 초과수익)
  산업재·소비재 → 영업이익 추정치 변화율
  금융          → PER (PBR 아님 — 통념과 다름)
  헬스케어      → Forward PER + EPS 성장
  유틸리티·리츠 → PBR + 배당수익률

[공통 보조지표]
  - EV/EBITDA
  - 애널리스트 목표가 괴리율
  - 매출 성장률
  - ROE

[통과 원칙]
  - 버블 복합 신호(3개+) 시만 탈락
  - 기본값 통과 (차트·기관 필터에서 이미 1차 검증됨)
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


# ── 섹터 감지 ──────────────────────────────────────────────────

def detect_sector(info: dict) -> str:
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()

    if "energy" in sector:
        return "ENERGY"
    elif "material" in sector:
        return "MATERIALS"
    elif "technology" in sector or "semiconductor" in industry or "software" in industry:
        return "TECHNOLOGY"
    elif "financial" in sector or "bank" in industry or "insurance" in industry:
        return "FINANCIALS"
    elif "industrial" in sector:
        return "INDUSTRIALS"
    elif "consumer" in sector and "discret" in sector:
        return "CONSUMER_DISC"
    elif "consumer" in sector and "staple" in sector:
        return "CONSUMER_STAPLES"
    elif "health" in sector:
        return "HEALTHCARE"
    elif "utilit" in sector:
        return "UTILITIES"
    elif "real estate" in sector or "reit" in industry:
        return "REAL_ESTATE"
    elif "communication" in sector:
        return "COMMUNICATION"
    return "GENERAL"


# ── 데이터 수집 ────────────────────────────────────────────────

def fetch_valuation_data(ticker: str) -> Optional[dict]:
    try:
        info = yf.Ticker(ticker).info or {}

        trailing_eps = info.get("trailingEps")
        forward_eps = info.get("forwardEps")
        eps_growth_fwd = None
        if trailing_eps and forward_eps and trailing_eps > 0:
            eps_growth_fwd = round((forward_eps - trailing_eps) / abs(trailing_eps) * 100, 2)

        return {
            "ticker": ticker,
            "sector_raw": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "sector_code": detect_sector(info),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "trailing_eps": trailing_eps,
            "forward_eps": forward_eps,
            "eps_growth_fwd_pct": eps_growth_fwd,
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            "roe": info.get("returnOnEquity"),
            "operating_margins": info.get("operatingMargins"),
            "peg_ratio": info.get("pegRatio"),
            "dividend_yield": info.get("dividendYield"),
            "target_mean_price": info.get("targetMeanPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions") or 0,
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        }
    except Exception as e:
        logger.error(f"{ticker} 데이터 수집 실패: {e}")
        return None


def calc_target_gap(current, target):
    if not current or not target or current <= 0:
        return None
    return round((target - current) / current * 100, 2)


# ── 섹터별 점수 함수 ───────────────────────────────────────────

def score_energy_materials(d: dict):
    """에너지·소재: 저PER + EV/EBITDA 우선"""
    score, signals, warnings = 0, [], []
    pe = d.get("trailing_pe")
    fwd_pe = d.get("forward_pe")
    ev_eb = d.get("ev_to_ebitda")

    if pe is not None:
        if pe <= 8:
            score += 35; signals.append(f"극단 저PER {pe:.1f}x")
        elif pe <= 12:
            score += 25; signals.append(f"저PER {pe:.1f}x")
        elif pe <= 18:
            score += 10; signals.append(f"적정PER {pe:.1f}x")
        elif pe > 40:
            warnings.append(f"고PER {pe:.1f}x")

    if pe and fwd_pe and fwd_pe < pe:
        score += 10; signals.append(f"이익개선기대 FwdPE {fwd_pe:.1f}x")

    if ev_eb is not None:
        if ev_eb <= 5:
            score += 20; signals.append(f"EV/EBITDA 저평가 {ev_eb:.1f}x")
        elif ev_eb <= 8:
            score += 10; signals.append(f"EV/EBITDA 적정 {ev_eb:.1f}x")
        elif ev_eb > 20:
            warnings.append(f"EV/EBITDA 고평가 {ev_eb:.1f}x")

    return score, signals, warnings


def score_technology(d: dict):
    """IT: Forward EPS 성장 추세 + PEG 우선"""
    score, signals, warnings = 0, [], []
    eps_g = d.get("eps_growth_fwd_pct")
    qoq = d.get("earnings_quarterly_growth")
    rev = d.get("revenue_growth")
    fwd_pe = d.get("forward_pe")
    peg = d.get("peg_ratio")

    if eps_g is not None:
        if eps_g >= 30:
            score += 40; signals.append(f"FwdEPS 고성장 +{eps_g:.1f}%")
        elif eps_g >= 15:
            score += 25; signals.append(f"FwdEPS 성장 +{eps_g:.1f}%")
        elif eps_g >= 5:
            score += 10; signals.append(f"FwdEPS 완만 +{eps_g:.1f}%")
        elif eps_g < -10:
            warnings.append(f"FwdEPS 역성장 {eps_g:.1f}%")

    if qoq is not None and qoq * 100 >= 25:
        score += 20; signals.append(f"분기EPS +{qoq*100:.1f}%")
    elif qoq is not None and qoq * 100 >= 10:
        score += 10

    if rev is not None and rev * 100 >= 20:
        score += 15; signals.append(f"매출고성장 +{rev*100:.1f}%")
    elif rev is not None and rev * 100 >= 10:
        score += 8

    if fwd_pe is not None:
        if fwd_pe > 80:
            warnings.append(f"FwdPE 과도 {fwd_pe:.1f}x")
        elif fwd_pe <= 20:
            score += 10; signals.append(f"IT 저FwdPE {fwd_pe:.1f}x")

    if peg is not None:
        if peg <= 1.0:
            score += 15; signals.append(f"PEG 저평가 {peg:.2f}")
        elif peg <= 2.0:
            score += 5
        elif peg > 4.0:
            warnings.append(f"PEG 과도 {peg:.2f}")

    return score, signals, warnings


def score_financials(d: dict):
    """금융: PER 우선 (연구 결과 — PBR 아님)"""
    score, signals, warnings = 0, [], []
    pe = d.get("trailing_pe")
    fwd_pe = d.get("forward_pe")
    roe = d.get("roe")
    pbr = d.get("price_to_book")

    if pe is not None:
        if pe <= 8:
            score += 35; signals.append(f"극단 저PER {pe:.1f}x")
        elif pe <= 12:
            score += 25; signals.append(f"저PER {pe:.1f}x")
        elif pe <= 15:
            score += 10

    if pe and fwd_pe and fwd_pe < pe:
        score += 10; signals.append(f"이익개선 FwdPE {fwd_pe:.1f}x")

    if roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 15:
            score += 20; signals.append(f"ROE {roe_pct:.1f}%")
        elif roe_pct >= 10:
            score += 10

    if pbr is not None and pbr < 1.0:
        score += 10; signals.append(f"PBR 순자산 미만 {pbr:.2f}x")

    return score, signals, warnings


def score_industrials_consumer(d: dict):
    """산업재·소비재: 이익 추정치 변화 + 매출 성장"""
    score, signals, warnings = 0, [], []
    eps_g = d.get("eps_growth_fwd_pct")
    rev = d.get("revenue_growth")
    op_m = d.get("operating_margins")
    pe = d.get("trailing_pe")
    fwd_pe = d.get("forward_pe")

    if eps_g is not None:
        if eps_g >= 15:
            score += 30; signals.append(f"이익추정 상향 +{eps_g:.1f}%")
        elif eps_g >= 5:
            score += 15; signals.append(f"이익개선 +{eps_g:.1f}%")
        elif eps_g < -15:
            warnings.append(f"이익 하향 {eps_g:.1f}%")

    if rev is not None and rev * 100 >= 15:
        score += 20; signals.append(f"매출성장 +{rev*100:.1f}%")
    elif rev is not None and rev * 100 >= 5:
        score += 10

    if op_m is not None and op_m * 100 >= 15:
        score += 15; signals.append(f"영업이익률 {op_m*100:.1f}%")
    elif op_m is not None and op_m * 100 >= 8:
        score += 8

    if pe is not None and pe <= 15:
        score += 10; signals.append(f"저PER {pe:.1f}x")

    if pe and fwd_pe and fwd_pe < pe * 0.9:
        score += 10; signals.append(f"FwdPE 개선 {fwd_pe:.1f}x")

    return score, signals, warnings


def score_healthcare(d: dict):
    """헬스케어: Forward PER + EPS 성장 복합"""
    score, signals, warnings = 0, [], []
    fwd_pe = d.get("forward_pe")
    eps_g = d.get("eps_growth_fwd_pct")
    rev = d.get("revenue_growth")

    if eps_g is not None:
        if eps_g >= 20:
            score += 35; signals.append(f"EPS 고성장 +{eps_g:.1f}%")
        elif eps_g >= 10:
            score += 20; signals.append(f"EPS 성장 +{eps_g:.1f}%")
        elif eps_g < -10:
            warnings.append(f"EPS 역성장 {eps_g:.1f}%")

    if fwd_pe is not None:
        if fwd_pe <= 15:
            score += 25; signals.append(f"저FwdPE {fwd_pe:.1f}x")
        elif fwd_pe <= 25:
            score += 10
        elif fwd_pe > 50:
            warnings.append(f"FwdPE 과도 {fwd_pe:.1f}x")

    if rev is not None and rev * 100 >= 10:
        score += 15; signals.append(f"매출성장 +{rev*100:.1f}%")

    return score, signals, warnings


def score_utilities_realestate(d: dict):
    """유틸리티·리츠: PBR + 배당수익률"""
    score, signals, warnings = 0, [], []
    pbr = d.get("price_to_book")
    dy = d.get("dividend_yield")
    pe = d.get("trailing_pe")

    if pbr is not None:
        if pbr <= 1.0:
            score += 35; signals.append(f"PBR 순자산 미만 {pbr:.2f}x")
        elif pbr <= 1.5:
            score += 20; signals.append(f"PBR 저평가 {pbr:.2f}x")
        elif pbr > 3.0:
            warnings.append(f"PBR 고평가 {pbr:.2f}x")

    if dy is not None:
        dy_pct = dy * 100
        if dy_pct >= 4:
            score += 25; signals.append(f"고배당 {dy_pct:.1f}%")
        elif dy_pct >= 2.5:
            score += 15; signals.append(f"배당 {dy_pct:.1f}%")

    if pe is not None and pe <= 20:
        score += 10

    return score, signals, warnings


def score_general(d: dict):
    """일반: PER + EPS 성장 복합"""
    score, signals, warnings = 0, [], []
    pe = d.get("trailing_pe")
    fwd_pe = d.get("forward_pe")
    eps_g = d.get("eps_growth_fwd_pct")

    if pe is not None:
        if pe <= 12:
            score += 25; signals.append(f"저PER {pe:.1f}x")
        elif pe <= 20:
            score += 10
        elif pe > 50:
            warnings.append(f"고PER {pe:.1f}x")

    if eps_g is not None and eps_g >= 10:
        score += 20; signals.append(f"EPS성장 +{eps_g:.1f}%")

    if pe and fwd_pe and fwd_pe < pe:
        score += 10

    return score, signals, warnings


def score_common(d: dict):
    """공통 보조: 목표가 괴리율"""
    score, signals, warnings = 0, [], []
    gap = calc_target_gap(d.get("current_price"), d.get("target_mean_price"))
    n = d.get("analyst_count") or 0

    if gap is not None and n >= 3:
        if gap >= 30:
            score += 20; signals.append(f"목표가괴리 +{gap:.1f}% (n={n})")
        elif gap >= 15:
            score += 12; signals.append(f"목표가괴리 +{gap:.1f}%")
        elif gap >= 5:
            score += 5
        elif gap <= -10:
            warnings.append(f"현재가 목표가 초과 {gap:.1f}%")

    return score, signals, warnings


# ── 버블 감지 ──────────────────────────────────────────────────

def detect_bubble(d: dict, sector: str):
    pe = d.get("trailing_pe")
    fwd_pe = d.get("forward_pe")
    pbr = d.get("price_to_book")
    eps_g = d.get("eps_growth_fwd_pct") or 0

    # 성장주 섹터는 고PER 허용
    pe_thresh = 120 if sector in ("TECHNOLOGY", "HEALTHCARE") else 60

    bubble = 0
    if pe and pe > pe_thresh: bubble += 1
    if fwd_pe and fwd_pe > 80: bubble += 1
    if pbr and pbr > 25: bubble += 1
    if eps_g < -20 and pe and pe > 30: bubble += 1

    if bubble >= 3:
        return True, f"버블 복합신호 {bubble}개 (PE={pe}, FwdPE={fwd_pe}, PBR={pbr})"
    return False, ""


# ── 통합 실행 ──────────────────────────────────────────────────

def score_valuation(ticker: str) -> Optional[dict]:
    val_data = fetch_valuation_data(ticker)
    if not val_data:
        return None

    sector = val_data["sector_code"]

    sector_fn = {
        "ENERGY": score_energy_materials,
        "MATERIALS": score_energy_materials,
        "TECHNOLOGY": score_technology,
        "FINANCIALS": score_financials,
        "INDUSTRIALS": score_industrials_consumer,
        "CONSUMER_DISC": score_industrials_consumer,
        "CONSUMER_STAPLES": score_industrials_consumer,
        "HEALTHCARE": score_healthcare,
        "UTILITIES": score_utilities_realestate,
        "REAL_ESTATE": score_utilities_realestate,
    }.get(sector, score_general)

    ms, msi, mw = sector_fn(val_data)
    cs, csi, cw = score_common(val_data)

    is_bubble, bubble_reason = detect_bubble(val_data, sector)

    return {
        **val_data,
        "val_score": max(ms + cs, 0),
        "signals": msi + csi,
        "warnings": mw + cw,
        "pass": not is_bubble,
        "fail_reason": bubble_reason if is_bubble else None,
    }


def run_valuation_analysis(tickers: list) -> list:
    logger.info(f"밸류에이션 분석 시작 — {len(tickers)}종목")
    results = []

    for ticker in tickers:
        result = score_valuation(ticker)
        if not result:
            continue
        status = "✅ 통과" if result["pass"] else "❌ 탈락"
        logger.info(
            f"  {status} {ticker} [{result['sector_code']}]: "
            f"점수 {result['val_score']} | {result['signals']}"
        )
        results.append(result)

    passed = [r for r in results if r["pass"]]
    os.makedirs("data", exist_ok=True)
    with open("data/valuation_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_analyzed": len(results),
            "passed_count": len(passed),
            "passed": passed,
            "failed": [r for r in results if not r["pass"]],
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"밸류에이션 분석 완료 — {len(passed)}/{len(results)} 통과")
    return passed


if __name__ == "__main__":
    chart_path = "data/chart_scan.json"
    if os.path.exists(chart_path):
        with open(chart_path) as f:
            tickers = [r["ticker"] for r in json.load(f).get("results", [])]
    else:
        tickers = ["XOM", "CVX", "NVDA", "MSFT", "JPM", "LLY", "NEE"]

    passed = run_valuation_analysis(tickers)
    print(f"\n통과 {len(passed)}종목:")
    for r in passed:
        print(f"  {r['ticker']:6s} [{r['sector_code']:15s}] 점수 {r['val_score']:3d} | {r['signals']}")

"""
MODULE 1: 거시환경 분석 (macro_analyzer.py)
- Yahoo Finance 전면 사용 (FRED 의존성 제거)
  - 연방기금금리 → ^IRX (13주 T-bill, 연준 금리 프록시)
  - CPI YoY → TIP ETF 12개월 수익률 (인플레 기대 프록시)
- Yahoo Finance: 10년물, 유가, 달러인덱스, VIX, S&P500, 금, 섹터ETF
- Claude Sonnet 분석: 시장 테마, 리스크, 수혜/회피 섹터 판단
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

YAHOO_TICKERS = {
    "us10y": "^TNX",       # 10년물 국채금리
    "us02y": "^IRX",       # 13주 T-bill (연방기금금리 프록시)
    "wti": "CL=F",         # WTI 유가
    "dxy": "DX-Y.NYB",     # 달러 인덱스
    "vix": "^VIX",         # 변동성 지수
    "sp500": "^GSPC",      # S&P500
    "gold": "GC=F",        # 금
}

SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLI", "XLU", "XLP", "XLB", "XLRE", "XLY", "XLC"]


def fetch_yahoo_price(ticker: str, period: str = "3mo") -> Optional[dict]:
    """Yahoo Finance에서 최근 가격 데이터 가져오기"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return None
        latest = hist.iloc[-1]
        return {
            "ticker": ticker,
            "latest_close": round(float(latest["Close"]), 4),
            "latest_date": str(hist.index[-1].date()),
            "period_high": round(float(hist["Close"].max()), 4),
            "period_low": round(float(hist["Close"].min()), 4),
        }
    except Exception as e:
        logger.error(f"Yahoo {ticker} 조회 실패: {e}")
        return None


def calc_sp500_consecutive_up_days() -> int:
    """S&P500 연속 상승일 계산"""
    try:
        hist = yf.Ticker("^GSPC").history(period="1mo")
        if hist.empty:
            return 0
        closes = hist["Close"].tolist()
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] > closes[i - 1]:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def calc_inflation_proxy() -> Optional[float]:
    """TIP ETF 12개월 수익률로 인플레 기대 프록시 계산"""
    try:
        hist = yf.Ticker("TIP").history(period="1y")
        if len(hist) < 20:
            return None
        ret = (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100
        return round(ret, 2)
    except Exception as e:
        logger.error(f"TIP 인플레 프록시 계산 실패: {e}")
        return None


def calc_sector_1m_returns() -> dict:
    """섹터 ETF 최근 1개월 수익률"""
    results = {}
    for etf in SECTOR_ETFS:
        try:
            hist = yf.Ticker(etf).history(period="1mo")
            if len(hist) >= 2:
                ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                results[etf] = round(float(ret), 2)
        except Exception:
            results[etf] = None
    return results


def calc_oil_change_pct(period: str = "3mo") -> Optional[float]:
    """유가 변동률 (3개월)"""
    try:
        hist = yf.Ticker("CL=F").history(period=period)
        if len(hist) < 2:
            return None
        return round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2)
    except Exception:
        return None


def calc_oil_trend() -> dict:
    """
    유가 단기 추세 분석.
    3개월 변동률은 후행 — 최근 1개월/2주 추세로 현재 방향 판단.

    trend 판정:
      DECLINING : 1개월 -8% 이하 OR 2주 -5% 이하 (하락 중)
      SURGING   : 1개월 +8% 이상 AND 최근 방향도 상승
      STABLE    : 그 외
    """
    try:
        hist = yf.Ticker("CL=F").history(period="3mo")
        if len(hist) < 10:
            return {"oil_change_1m_pct": None, "oil_change_2w_pct": None, "oil_trend": "STABLE"}

        current = float(hist["Close"].iloc[-1])
        price_1m  = float(hist["Close"].iloc[-22]) if len(hist) >= 22 else float(hist["Close"].iloc[0])
        price_2w  = float(hist["Close"].iloc[-10]) if len(hist) >= 10 else float(hist["Close"].iloc[0])
        price_peak = float(hist["Close"].max())

        chg_1m = round((current - price_1m) / price_1m * 100, 2)
        chg_2w = round((current - price_2w) / price_2w * 100, 2)
        # 고점 대비 낙폭
        from_peak = round((current - price_peak) / price_peak * 100, 2)

        if chg_1m <= -8 or chg_2w <= -5 or from_peak <= -15:
            trend = "DECLINING"
        elif chg_1m >= 8 and chg_2w >= 0:
            trend = "SURGING"
        else:
            trend = "STABLE"

        return {
            "oil_change_1m_pct": chg_1m,
            "oil_change_2w_pct": chg_2w,
            "oil_from_peak_pct": from_peak,
            "oil_trend": trend,
        }
    except Exception as e:
        return {"oil_change_1m_pct": None, "oil_change_2w_pct": None, "oil_trend": "STABLE"}


# ── 거시 데이터 전체 수집 ──────────────────────────────────────

def collect_macro_data() -> dict:
    """전체 거시 데이터 수집 (yfinance 전면 사용)"""
    logger.info("거시 데이터 수집 시작 (yfinance)")
    data = {}

    # 인플레 프록시 (TIP ETF 12개월 수익률)
    data["inflation_proxy_tip"] = calc_inflation_proxy()

    # Yahoo — 주요 지표 (^IRX = 13주 T-bill → 연방기금금리 프록시 포함)
    for key, ticker in YAHOO_TICKERS.items():
        result = fetch_yahoo_price(ticker)
        data[key] = result["latest_close"] if result else None

    # ^IRX를 fed_funds_rate로 매핑 (다른 모듈 호환)
    data["fed_funds_rate"] = data.pop("us02y", None)

    # S&P500 연속 상승일
    data["sp500_consecutive_up"] = calc_sp500_consecutive_up_days()

    # 유가 변동률 + 단기 추세
    data["oil_change_3m_pct"] = calc_oil_change_pct()
    oil_trend_data = calc_oil_trend()
    data.update(oil_trend_data)

    # 섹터 수익률
    data["sector_returns_1m"] = calc_sector_1m_returns()

    logger.info(f"거시 데이터 수집 완료: {json.dumps(data, indent=2, ensure_ascii=False)}")
    return data


# ── 하드코딩 섹터 매핑 (규칙 기반 프리필터) ──────────────────

def rule_based_sector_hints(data: dict) -> dict:
    """거시 조건별 수혜/회피 섹터 힌트 (Claude 분석 보조용)"""
    favored = set()
    avoid = set()

    # 유가 추세 (3개월 변동률 + 단기 추세 조합)
    oil_chg_3m = data.get("oil_change_3m_pct")
    oil_trend  = data.get("oil_trend", "STABLE")
    oil_chg_1m = data.get("oil_change_1m_pct")

    if oil_trend == "DECLINING":
        # 유가 하락 중 → 에너지 회피 (3개월 수치가 양수여도 무시)
        avoid.update(["XLE"])
        logger.info(f"  유가 하락 감지 (1m={oil_chg_1m}%) → XLE 회피섹터로 전환")
    elif oil_trend == "SURGING" and oil_chg_3m and oil_chg_3m > 20:
        # 유가 실제 상승 중 → 에너지 수혜
        favored.update(["XLE"])
        avoid.update(["XLI", "XLY"])

    # 금리
    fed = data.get("fed_funds_rate")
    us10y = data.get("us10y")
    if fed is not None and fed < 3.0:
        favored.update(["XLRE", "XLV"])
        avoid.update(["XLF"])
    elif fed is not None and fed >= 5.0:
        favored.update(["XLF"])
        avoid.update(["XLRE"])

    # VIX
    vix = data.get("vix")
    if vix and vix >= 25:
        favored.update(["XLP", "XLU"])

    # 달러 급등 (DXY > 105)
    dxy = data.get("dxy")
    if dxy and dxy > 105:
        favored.update(["XLY", "XLP"])  # 내수
        avoid.update(["XLB"])

    # 인플레 상승 (TIP 프록시 수익률 4%+ → 인플레 기대 상승)
    inflation = data.get("inflation_proxy_tip")
    if inflation and inflation > 4.0:
        favored.update(["XLB", "XLE"])

    return {
        "rule_favored": sorted(favored),
        "rule_avoid": sorted(avoid),
    }


# ── Claude Sonnet 분석 ────────────────────────────────────────

ANALYSIS_PROMPT = """당신은 월가 수석 매크로 전략가입니다. 아래 거시 데이터를 분석하고 JSON만 반환하세요.

## 수집된 거시 데이터
{macro_json}

## 규칙 기반 힌트 (참고용)
{hints_json}

## 반환 형식 (JSON만, 설명 없이)
{{
  "dominant_theme": "현재 시장 지배 테마 1줄 요약",
  "risk_level": "LOW|MEDIUM|HIGH",
  "favored_sectors": ["수혜 섹터 ETF 심볼 리스트"],
  "avoid_sectors": ["회피 섹터 ETF 심볼 리스트"],
  "rationale": "판단 근거 3줄 (한국어)",
  "special_warnings": ["특이사항 리스트"]
}}
"""


def analyze_with_claude(macro_data: dict, hints: dict) -> Optional[dict]:
    """Claude SDK로 거시환경 종합 분석"""
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY 미설정 — Claude 분석 스킵")
        return None

    prompt = ANALYSIS_PROMPT.format(
        macro_json=json.dumps(macro_data, indent=2, ensure_ascii=False),
        hints_json=json.dumps(hints, indent=2, ensure_ascii=False),
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
        logger.error(f"Claude 분석 실패: {e}")
        return None


# ── 메인 실행 ──────────────────────────────────────────────────

def run_macro_analysis() -> dict:
    """거시환경 분석 전체 파이프라인 실행"""
    macro_data = collect_macro_data()
    hints = rule_based_sector_hints(macro_data)

    analysis = analyze_with_claude(macro_data, hints)

    result = {
        "timestamp": datetime.now().isoformat(),
        "raw_data": macro_data,
        "rule_hints": hints,
        "ai_analysis": analysis,
    }

    # 결과 저장
    os.makedirs("data", exist_ok=True)
    with open("data/macro_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("거시환경 분석 완료 — data/macro_analysis.json 저장됨")
    return result


if __name__ == "__main__":
    result = run_macro_analysis()
    print(json.dumps(result, indent=2, ensure_ascii=False))

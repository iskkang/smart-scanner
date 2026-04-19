"""
MODULE 2: 종목 스캔 (chart_scanner.py)
- 이평선 정배열, 눌림목, 거래량 수축/급증, RSI 필터
- 종목 점수 산출 (0~100)
- 거시환경 분석 결과의 수혜 섹터 기반 스캔 대상 필터링
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import io
import requests
import yfinance as yf
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 섹터 → 대표 종목 매핑 ─────────────────────────────────────
# 실전에서는 S&P500 전체 또는 NASDAQ 스크리닝 대상을 사용
# 여기서는 섹터 ETF 구성종목 상위를 기본 유니버스로 제공

SECTOR_UNIVERSE = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "ORCL", "CSCO", "ACN",
            "INTC", "IBM", "INTU", "NOW", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN"],
    "XLI": ["GE", "CAT", "HON", "UNP", "BA", "RTX", "LMT", "DE", "NOC", "WM"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "VMC"],
    "XLRE": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "EA"],
}


MIN_MARKET_CAP = 10_000_000_000  # $10B 이상 대형주


def fetch_sp500_tickers() -> list:
    """Wikipedia에서 S&P 500 전종목 실시간 수집 (브라우저 헤더로 403 우회)"""
    try:
        import io
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 500 수집 완료: {len(tickers)}종목")
        return tickers
    except Exception as e:
        logger.warning(f"S&P 500 수집 실패 — 내장 유니버스 사용: {e}")
        return []


def filter_by_market_cap(tickers: list, min_cap: int = MIN_MARKET_CAP) -> list:
    """시가총액 $10B+ 필터 — 병렬로 빠르게 수집"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def get_cap(ticker):
        try:
            cap = yf.Ticker(ticker).info.get("marketCap")
            return ticker, cap
        except Exception:
            return ticker, None

    passed = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(get_cap, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, cap = future.result()
            if cap and cap >= min_cap:
                passed.append((ticker, cap))

    passed.sort(key=lambda x: x[1], reverse=True)
    result = [t for t, _ in passed]
    logger.info(f"시가총액 필터 완료: {len(tickers)}종목 → {len(result)}종목 ($10B+)")
    return result


def get_scan_universe(favored_sectors: list = None, rated_universe: list = None) -> list:
    """
    스캔 유니버스 구성 우선순위:
    1. 외부 등급 유니버스 (Zacks/SA/Morningstar Strong Buy) — 최우선
    2. S&P 500 전종목 (Wikipedia) + 시가총액 필터 — 폴백
    3. 내장 유니버스 — 최후 폴백
    수혜 섹터 종목은 앞으로 배치 (우선 스캔)
    """
    # ── 1순위: 외부 등급 유니버스 ──
    if rated_universe and len(rated_universe) > 0:
        logger.info(f"외부 등급 유니버스 사용: {len(rated_universe)}종목")
        universe = rated_universe
        if favored_sectors:
            priority = set()
            for sector in favored_sectors:
                priority.update(SECTOR_UNIVERSE.get(sector, []))
            front = [t for t in universe if t in priority]
            rest  = [t for t in universe if t not in priority]
            logger.info(f"  수혜섹터 우선배치: {len(front)}종목")
            return front + rest
        return universe

    # ── 2순위: S&P500 동적 수집 ──
    logger.info("외부 유니버스 없음 — S&P500 폴백")
    sp500 = fetch_sp500_tickers()
    if not sp500:
        all_tickers = []
        for v in SECTOR_UNIVERSE.values():
            all_tickers.extend(v)
        return sorted(set(all_tickers))

    universe = filter_by_market_cap(sp500, MIN_MARKET_CAP)
    if not favored_sectors:
        return universe

    priority = set()
    for sector in favored_sectors:
        priority.update(SECTOR_UNIVERSE.get(sector, []))
    front = [t for t in universe if t in priority]
    rest  = [t for t in universe if t not in priority]
    return front + rest


# ── 동적 스캔 임계치 ───────────────────────────────────────────

def get_dynamic_thresholds(macro_data: dict = None) -> dict:
    """
    VIX + 거시 리스크 레벨에 따라 스캔 기준을 동적 조정.

    VIX 구간:
      >= 30 (공포)   : RSI 하한 낮춤, 눌림목 더 깊어도 허용, 점수 기준 완화
      20~30 (경계)   : 기본 기준
      <= 15 (안정)   : RSI 상한 높임, 얕은 눌림목도 허용, 점수 기준 강화

    리스크 레벨 (Claude 분석 결과):
      HIGH   : 공포 구간 기준 추가 적용
      MEDIUM : 기본 기준 유지
      LOW    : 안정 구간 기준 적용
    """
    defaults = {
        "rsi_min": 38, "rsi_max": 58,
        "pullback_min": -15, "pullback_max": -5,
        "min_score": 40,
        "vol_surge_threshold": 1.5,
        "regime": "NORMAL",
    }

    if not macro_data:
        return defaults

    raw = macro_data.get("raw_data") or {}
    ai  = macro_data.get("ai_analysis") or {}

    vix        = raw.get("vix") or 20
    risk_level = ai.get("risk_level", "MEDIUM").upper()
    sp500_up   = raw.get("sp500_consecutive_up", 0)

    # ── VIX 기반 1차 조정 ──
    if vix >= 30:
        thresholds = {
            "rsi_min": 25, "rsi_max": 50,        # 더 과매도된 종목까지 포함
            "pullback_min": -25, "pullback_max": -8,  # 더 깊은 눌림목 허용
            "min_score": 35,                      # 통과 기준 완화
            "vol_surge_threshold": 1.3,           # 거래량 급증 기준 완화
            "regime": "FEAR",
        }
    elif vix >= 20:
        thresholds = {
            "rsi_min": 38, "rsi_max": 58,
            "pullback_min": -15, "pullback_max": -5,
            "min_score": 40,
            "vol_surge_threshold": 1.5,
            "regime": "CAUTION",
        }
    elif vix >= 15:
        thresholds = {
            "rsi_min": 40, "rsi_max": 60,
            "pullback_min": -12, "pullback_max": -4,
            "min_score": 45,
            "vol_surge_threshold": 1.5,
            "regime": "NORMAL",
        }
    else:  # VIX < 15 (강세장)
        thresholds = {
            "rsi_min": 45, "rsi_max": 65,        # 더 강한 종목 위주
            "pullback_min": -10, "pullback_max": -3,  # 얕은 눌림목도 허용
            "min_score": 50,                      # 통과 기준 강화
            "vol_surge_threshold": 1.8,
            "regime": "BULL",
        }

    # ── 리스크 레벨 2차 보정 ──
    if risk_level == "HIGH":
        # 추가 보수적: RSI 하한 낮추고 눌림목 더 깊게 허용
        thresholds["rsi_min"]      = max(thresholds["rsi_min"] - 5, 20)
        thresholds["pullback_min"] = max(thresholds["pullback_min"] - 5, -30)
        thresholds["min_score"]    = max(thresholds["min_score"] - 5, 30)
        thresholds["regime"]      += "_HIGH_RISK"
    elif risk_level == "LOW":
        # 추가 공격적: RSI 상한 높이고 점수 기준 강화
        thresholds["rsi_max"]   = min(thresholds["rsi_max"] + 5, 70)
        thresholds["min_score"] = min(thresholds["min_score"] + 5, 60)
        thresholds["regime"]   += "_LOW_RISK"

    # ── S&P500 연속 상승 보정 ──
    # 연속 상승 5일+ = 단기 과열 → 기준 강화
    if sp500_up >= 5:
        thresholds["min_score"] = min(thresholds["min_score"] + 5, 65)
        thresholds["regime"]   += "_OVERBOUGHT"

    logger.info(
        f"동적 임계치 적용 — 레짐: {thresholds['regime']} | "
        f"VIX {vix:.1f} | 리스크 {risk_level} | "
        f"RSI {thresholds['rsi_min']}~{thresholds['rsi_max']} | "
        f"눌림목 {thresholds['pullback_min']}%~{thresholds['pullback_max']}% | "
        f"최소점수 {thresholds['min_score']}"
    )
    return thresholds



def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def scan_ticker(ticker: str, thresholds: dict = None) -> Optional[dict]:
    """
    개별 종목 차트 스캔.
    thresholds: get_dynamic_thresholds() 결과 (None이면 기본값 사용)
    """
    t = thresholds or {
        "rsi_min": 38, "rsi_max": 58,
        "pullback_min": -15, "pullback_max": -5,
        "vol_surge_threshold": 1.5,
    }
    rsi_min  = t["rsi_min"]
    rsi_max  = t["rsi_max"]
    pb_min   = t["pullback_min"]
    pb_max   = t["pullback_max"]
    vol_thr  = t["vol_surge_threshold"]

    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 200:
            return None

        close  = hist["Close"]
        volume = hist["Volume"]

        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        latest_close  = float(close.iloc[-1])
        latest_sma20  = float(sma20.iloc[-1])
        latest_sma50  = float(sma50.iloc[-1])
        latest_sma200 = float(sma200.iloc[-1])

        # 조건 1: 정배열
        if not (latest_sma20 > latest_sma50 > latest_sma200):
            return None

        # 조건 2: 눌림목 (동적 범위)
        high_60d    = float(close.iloc[-60:].max())
        pullback_pct = (latest_close - high_60d) / high_60d * 100
        if not (pb_min <= pullback_pct <= pb_max):
            return None

        # 조건 3: 거래량 감소
        vol_5d  = float(volume.iloc[-5:].mean())
        vol_20d = float(volume.iloc[-20:].mean())
        if vol_5d >= vol_20d:
            return None

        # 조건 4: RSI (동적 범위)
        rsi        = calc_rsi(close)
        latest_rsi = float(rsi.iloc[-1])
        if not (rsi_min <= latest_rsi <= rsi_max):
            return None

        # 조건 5: 반등 신호
        vol_today  = float(volume.iloc[-1])
        vol_surge  = vol_today / vol_5d if vol_5d > 0 else 0
        above_sma5 = latest_close > float(close.rolling(5).mean().iloc[-1])
        has_bounce = (vol_surge >= vol_thr) and above_sma5

        # ── 점수 산출 ──
        score = 0

        if -12 <= pullback_pct <= -8:
            score += 30
        elif pb_min <= pullback_pct < -8 or -8 < pullback_pct <= pb_max:
            score += 15

        mid_rsi = (rsi_min + rsi_max) / 2
        if abs(latest_rsi - mid_rsi) <= 5:
            score += 25
        elif rsi_min <= latest_rsi <= rsi_max:
            score += 12

        if vol_surge >= vol_thr:
            score += 25
        elif vol_surge >= vol_thr * 0.8:
            score += 10

        if above_sma5:
            score += 10

        high_52w    = float(close.max())
        from_52w_high = (latest_close - high_52w) / high_52w * 100
        if from_52w_high >= -30:
            score += 10

        return {
            "ticker": ticker,
            "price": round(latest_close, 2),
            "pullback_pct": round(pullback_pct, 2),
            "rsi": round(latest_rsi, 2),
            "vol_surge_ratio": round(vol_surge, 2),
            "above_sma5": above_sma5,
            "has_bounce_signal": has_bounce,
            "from_52w_high_pct": round(from_52w_high, 2),
            "chart_score": min(score, 100),
            "sma20": round(latest_sma20, 2),
            "sma50": round(latest_sma50, 2),
            "sma200": round(latest_sma200, 2),
            "scan_regime": t.get("regime", "NORMAL"),
        }

    except Exception as e:
        logger.error(f"{ticker} 스캔 실패: {e}")
        return None


# ── 전체 스캔 실행 ─────────────────────────────────────────────

def run_chart_scan(favored_sectors: list = None, min_score: int = 40, macro_data: dict = None, rated_universe: list = None) -> list:
    """
    차트 스캔 전체 실행.
    macro_data: macro_analyzer 결과 (VIX + 리스크 레벨 기반 동적 임계치 적용)
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    thresholds = get_dynamic_thresholds(macro_data)
    effective_min_score = thresholds["min_score"]

    universe = get_scan_universe(favored_sectors, rated_universe=rated_universe)
    logger.info(
        f"차트 스캔 시작 — 대상 {len(universe)}종목 | "
        f"레짐 {thresholds['regime']} | "
        f"RSI {thresholds['rsi_min']}~{thresholds['rsi_max']} | "
        f"눌림목 {thresholds['pullback_min']}~{thresholds['pullback_max']}% | "
        f"최소점수 {effective_min_score}"
    )

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(scan_ticker, t, thresholds): t for t in universe}
        for future in as_completed(future_to_ticker):
            completed += 1
            try:
                result = future.result()
                if result and result["chart_score"] >= effective_min_score:
                    results.append(result)
                    logger.info(
                        f"  ✅ {result['ticker']}: 점수 {result['chart_score']}점 "
                        f"| 눌림 {result['pullback_pct']}% | RSI {result['rsi']}"
                    )
            except Exception:
                pass
            if completed % 100 == 0:
                logger.info(f"  ... {completed}/{len(universe)} 스캔 완료")
                time.sleep(1)

    results.sort(key=lambda x: x["chart_score"], reverse=True)

    os.makedirs("data", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "scan_universe_size": len(universe),
        "favored_sectors": favored_sectors,
        "thresholds": thresholds,
        "passed_count": len(results),
        "results": results,
    }
    with open("data/chart_scan.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"차트 스캔 완료 — {len(results)}종목 통과 (data/chart_scan.json)")
    return results


if __name__ == "__main__":
    # 단독 테스트: 전체 유니버스 스캔
    # 거시환경 결과가 있으면 로드
    macro_path = "data/macro_analysis.json"
    favored = None
    if os.path.exists(macro_path):
        with open(macro_path, "r") as f:
            macro = json.load(f)
        ai = macro.get("ai_analysis")
        if ai:
            favored = ai.get("favored_sectors")
            logger.info(f"거시환경 수혜섹터 로드: {favored}")

    passed = run_chart_scan(favored_sectors=favored)
    print(f"\n통과 종목 {len(passed)}개:")
    for r in passed:
        print(f"  {r['ticker']:6s} | 점수 {r['chart_score']:3d} | 눌림 {r['pullback_pct']:+.1f}% | RSI {r['rsi']:.1f}")

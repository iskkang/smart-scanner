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
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD", "OXY"],
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
    """Wikipedia에서 S&P 500 전종목 실시간 수집"""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 500 수집 완료: {len(tickers)}종목")
        return tickers
    except Exception as e:
        logger.warning(f"S&P 500 수집 실패 — 내장 유니버스 사용: {e}")
        return []


def filter_by_market_cap(tickers: list, min_cap: int = MIN_MARKET_CAP) -> list:
    """
    시가총액 필터 — $10B 이상만 통과.
    병렬로 market cap 빠르게 수집 후 필터링.
    """
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

    # 시가총액 내림차순 정렬
    passed.sort(key=lambda x: x[1], reverse=True)
    result = [t for t, _ in passed]
    logger.info(f"시가총액 필터 완료: {len(tickers)}종목 → {len(result)}종목 ($10B+)")
    return result


def get_scan_universe(favored_sectors: list = None) -> list:
    """
    스캔 유니버스 구성:
    1. S&P 500 전종목 수집 (Wikipedia)
    2. 시가총액 $10B 이상 필터
    3. 수혜 섹터 종목을 앞으로 배치 (우선 스캔)
    """
    sp500 = fetch_sp500_tickers()
    if not sp500:
        all_tickers = []
        for v in SECTOR_UNIVERSE.values():
            all_tickers.extend(v)
        return sorted(set(all_tickers))

    # 시가총액 필터
    universe = filter_by_market_cap(sp500, MIN_MARKET_CAP)

    if not favored_sectors:
        return universe

    # 수혜 섹터 종목 앞으로 배치 (제외하지 않고 우선순위만)
    priority = set()
    for sector in favored_sectors:
        priority.update(SECTOR_UNIVERSE.get(sector, []))

    front = [t for t in universe if t in priority]
    rest = [t for t in universe if t not in priority]
    return front + rest


# ── 기술적 지표 계산 ───────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def scan_ticker(ticker: str) -> Optional[dict]:
    """
    개별 종목 차트 스캔.
    조건:
      1) 20/50/200일 이평선 정배열
      2) 최근 60일 고점 대비 -5% ~ -15% 눌림목
      3) 조정 중 거래량 감소 (vol_5d < vol_20d)
      4) RSI 38~58
      5) 반등 신호: 거래량 1.5배 급증 + 5일선 위 종가
    점수 산출 (0~100)
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if len(hist) < 200:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        # 이평선
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        latest_close = float(close.iloc[-1])
        latest_sma20 = float(sma20.iloc[-1])
        latest_sma50 = float(sma50.iloc[-1])
        latest_sma200 = float(sma200.iloc[-1])

        # 조건 1: 정배열
        if not (latest_sma20 > latest_sma50 > latest_sma200):
            return None

        # 조건 2: 눌림목 (-5% ~ -15%)
        high_60d = float(close.iloc[-60:].max())
        pullback_pct = (latest_close - high_60d) / high_60d * 100
        if not (-15 <= pullback_pct <= -5):
            return None

        # 조건 3: 거래량 감소
        vol_5d = float(volume.iloc[-5:].mean())
        vol_20d = float(volume.iloc[-20:].mean())
        if vol_5d >= vol_20d:
            return None

        # 조건 4: RSI
        rsi = calc_rsi(close)
        latest_rsi = float(rsi.iloc[-1])
        if not (38 <= latest_rsi <= 58):
            return None

        # 조건 5: 반등 신호
        vol_today = float(volume.iloc[-1])
        vol_surge = vol_today / vol_5d if vol_5d > 0 else 0
        above_sma5 = latest_close > float(close.rolling(5).mean().iloc[-1])

        has_bounce = (vol_surge >= 1.5) and above_sma5

        # ── 점수 산출 ──
        score = 0

        # 눌림목 깊이
        if -12 <= pullback_pct <= -8:
            score += 30
        elif -15 <= pullback_pct < -8 or -8 < pullback_pct <= -5:
            score += 15

        # RSI
        if 42 <= latest_rsi <= 52:
            score += 25
        elif 38 <= latest_rsi < 42 or 52 < latest_rsi <= 58:
            score += 12

        # 거래량 급증
        if vol_surge >= 1.5:
            score += 25
        elif vol_surge >= 1.2:
            score += 10

        # 5일선 위 마감
        if above_sma5:
            score += 10

        # 52주 고점 대비
        high_52w = float(close.max())
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
        }

    except Exception as e:
        logger.error(f"{ticker} 스캔 실패: {e}")
        return None


# ── 전체 스캔 실행 ─────────────────────────────────────────────

def run_chart_scan(favored_sectors: list = None, min_score: int = 40) -> list:
    """
    차트 스캔 전체 실행.
    S&P 500 → 시가총액 $10B+ 필터 → 차트 조건 스캔 (병렬)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    universe = get_scan_universe(favored_sectors)
    logger.info(f"차트 스캔 시작 — 대상 {len(universe)}종목 (수혜섹터 우선 배치: {favored_sectors})")

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(scan_ticker, t): t for t in universe}
        for future in as_completed(future_to_ticker):
            completed += 1
            try:
                result = future.result()
                if result and result["chart_score"] >= min_score:
                    results.append(result)
                    logger.info(
                        f"  ✅ {result['ticker']}: 점수 {result['chart_score']}점 "
                        f"| 눌림 {result['pullback_pct']}% | RSI {result['rsi']}"
                    )
            except Exception:
                pass
            if completed % 50 == 0:
                logger.info(f"  ... {completed}/{len(universe)} 스캔 완료")

    results.sort(key=lambda x: x["chart_score"], reverse=True)

    os.makedirs("data", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "scan_universe_size": len(universe),
        "favored_sectors": favored_sectors,
        "min_score": min_score,
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

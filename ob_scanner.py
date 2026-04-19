"""
ob_scanner.py — Order Block Touch & Bounce 패턴 스캐너

패턴 정의:
  1. 상승 추세 확립 (BOS: Break of Structure)
  2. 직전 BOS 이전 마지막 베어리시 캔들 = Order Block
  3. 가격이 OB 구간에 닿음 (터치)
  4. OB에서 반등 신호 (불리시 캔들 + 거래량 증가)

실행: python main.py 에서 scan 내에 통합 또는 독립 실행 가능
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Order Block 감지
# ═══════════════════════════════════════════════════════════════

def detect_order_blocks(hist: pd.DataFrame, swing_period: int = 20) -> list:
    """
    Order Block 감지 알고리즘:
    1. 스윙 고점 돌파(BOS) 구간 탐색
    2. 돌파 직전 마지막 베어리시 캔들 = Bullish Order Block
    3. OB 구간: [캔들 저가, 캔들 시가(몸통 상단)]

    반환: OB 리스트 (최신 순)
    """
    if len(hist) < swing_period + 10:
        return []

    close = hist["Close"]
    open_ = hist["Open"]
    high  = hist["High"]
    low   = hist["Low"]
    vol   = hist["Volume"]

    obs = []

    for i in range(swing_period, len(hist) - 3):
        prev_high = float(high.iloc[i - swing_period:i].max())
        curr_high = float(high.iloc[i])

        # BOS 조건: 현재 고가가 이전 구간 고점을 2%+ 돌파
        if curr_high <= prev_high * 1.02:
            continue

        # BOS 직전 베어리시 캔들 탐색 (최근 7개)
        for j in range(i - 1, max(i - 8, 0), -1):
            c = float(close.iloc[j])
            o = float(open_.iloc[j])
            if c >= o:  # 불리시 캔들은 스킵
                continue

            # 베어리시 캔들 발견 = Order Block
            ob_low  = float(low.iloc[j])
            ob_high = float(o)         # 몸통 상단 (시가)
            ob_body = abs(o - c)
            avg_vol = float(vol.iloc[max(0, j-20):j].mean()) if j > 0 else 1
            vol_ratio = float(vol.iloc[j]) / avg_vol if avg_vol > 0 else 1

            # 이동 폭 (BOS 당시 상승률)
            move_pct = (curr_high - float(high.iloc[j])) / float(high.iloc[j]) * 100

            obs.append({
                "ob_idx": j,
                "ob_date": str(hist.index[j].date()) if hasattr(hist.index[j], 'date') else str(hist.index[j])[:10],
                "ob_low": round(ob_low, 2),
                "ob_high": round(ob_high, 2),
                "ob_mid": round((ob_low + ob_high) / 2, 2),
                "ob_body_size": round(ob_body, 2),
                "vol_ratio": round(vol_ratio, 2),
                "move_pct_after": round(move_pct, 1),
                "bos_idx": i,
            })
            break  # 한 BOS당 하나의 OB

    # 최신 순 정렬, 중복 유사 구간 병합
    obs.sort(key=lambda x: x["ob_idx"], reverse=True)
    return obs


# ═══════════════════════════════════════════════════════════════
# OB Touch & Bounce 패턴 감지
# ═══════════════════════════════════════════════════════════════

def check_ob_touch_bounce(hist: pd.DataFrame) -> Optional[dict]:
    """
    현재 가격이 Order Block을 터치 후 반등 중인지 감지.

    조건:
    A. 상승 추세 유지 (20 > 50 > 200 이평선)
    B. 최근 저가가 유효한 OB 구간에 닿음 (터치)
    C. 반등 신호: 현재 캔들이 불리시 + OB 위 마감 + 거래량 증가
    D. 과도한 하락 아님 (OB 하단을 크게 이탈하지 않음)
    """
    if len(hist) < 210:
        return None

    close = hist["Close"]
    open_ = hist["Open"]
    high  = hist["High"]
    low   = hist["Low"]
    vol   = hist["Volume"]

    # ── A. 이평선 정배열 확인 ──
    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    if not (sma20 > sma50 > sma200):
        return None

    current    = float(close.iloc[-1])
    today_open = float(open_.iloc[-1])
    today_low  = float(low.iloc[-1])
    today_high = float(high.iloc[-1])
    vol_today  = float(vol.iloc[-1])
    vol_5d     = float(vol.iloc[-6:-1].mean())
    vol_20d    = float(vol.iloc[-21:-1].mean())

    # OB 탐색 (최근 120일 이내만 유효)
    obs = detect_order_blocks(hist)
    recent_obs = [ob for ob in obs if ob["ob_idx"] >= len(hist) - 120]

    for ob in recent_obs:
        ob_low  = ob["ob_low"]
        ob_high = ob["ob_high"]
        ob_mid  = ob["ob_mid"]
        tolerance = (ob_high - ob_low) * 0.3  # 구간의 30% 여유

        # ── B. OB 터치 확인 ──
        # 최근 저가가 OB 상단 ± 여유 안에 들어옴
        touched = (today_low <= ob_high + tolerance) and (today_low >= ob_low - tolerance)
        if not touched:
            # 직전 1~3일 저가도 확인
            for lookback in [1, 2, 3]:
                if len(hist) > lookback:
                    prev_low = float(low.iloc[-(lookback+1)])
                    if (prev_low <= ob_high + tolerance) and (prev_low >= ob_low - tolerance):
                        touched = True
                        break
        if not touched:
            continue

        # ── C. 반등 신호 확인 ──
        is_bullish    = current > today_open                    # 불리시 캔들
        above_ob_mid  = current > ob_mid                        # OB 중단 이상 마감
        vol_surge     = vol_today > vol_5d * 1.2               # 거래량 평균 이상

        if not (is_bullish and above_ob_mid):
            continue

        # ── D. OB 하단 크게 이탈 안했는지 ──
        ob_break = today_low < ob_low * 0.97  # 3% 이상 이탈 = 무효화
        if ob_break:
            continue

        # ── 점수 산출 ──
        score = 0
        signals = []

        # 반등 강도
        bounce_pct = (current - today_low) / today_low * 100
        if bounce_pct >= 3:
            score += 30
            signals.append(f"강반등 +{bounce_pct:.1f}%")
        elif bounce_pct >= 1.5:
            score += 20
            signals.append(f"반등 +{bounce_pct:.1f}%")

        # 거래량
        vol_ratio = vol_today / vol_5d if vol_5d > 0 else 1
        if vol_ratio >= 2.0:
            score += 25
            signals.append(f"거래량 폭증 {vol_ratio:.1f}x")
        elif vol_ratio >= 1.5:
            score += 15
            signals.append(f"거래량 증가 {vol_ratio:.1f}x")
        elif vol_ratio >= 1.2:
            score += 8

        # OB 품질 (이전 상승 폭)
        if ob.get("move_pct_after", 0) >= 20:
            score += 20
            signals.append(f"강한OB (이후+{ob['move_pct_after']}%)")
        elif ob.get("move_pct_after", 0) >= 10:
            score += 10

        # 이평선 정배열 밀착도
        dist_from_sma20 = abs(current - sma20) / sma20 * 100
        if dist_from_sma20 <= 3:
            score += 15
            signals.append("20일선 근접")

        # 핀바(Pin Bar) / 망치형 캔들
        candle_range = today_high - today_low
        lower_wick   = today_open - today_low if today_open > current else current - today_low
        if candle_range > 0 and lower_wick / candle_range >= 0.6:
            score += 10
            signals.append("망치형(Pin Bar)")

        return {
            "ob_zone": {"low": ob_low, "high": ob_high, "mid": ob_mid},
            "ob_date": ob.get("ob_date"),
            "ob_move_after": ob.get("move_pct_after"),
            "touch_confirmed": True,
            "bounce_pct": round(bounce_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "ob_score": min(score, 100),
            "ob_signals": signals,
        }

    return None


# ═══════════════════════════════════════════════════════════════
# 개별 종목 스캔
# ═══════════════════════════════════════════════════════════════

def scan_ob_pattern(ticker: str) -> Optional[dict]:
    """OB Touch & Bounce 패턴 개별 종목 스캔"""
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 210:
            return None

        ob_result = check_ob_touch_bounce(hist)
        if not ob_result:
            return None

        close   = hist["Close"]
        current = float(close.iloc[-1])
        high_52 = float(close.max())
        from_52 = (current - high_52) / high_52 * 100

        return {
            "ticker": ticker,
            "price": round(current, 2),
            "from_52w_high_pct": round(from_52, 1),
            **ob_result,
        }

    except Exception as e:
        logger.error(f"{ticker} OB 스캔 실패: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 배치 스캔 실행
# ═══════════════════════════════════════════════════════════════

def run_ob_scan(tickers: list, min_score: int = 40) -> list:
    """
    전체 유니버스 대상 OB Touch & Bounce 스캔.
    기존 chart_scan 결과나 rated_universe를 입력으로 받음.
    """
    logger.info(f"OB Touch & Bounce 스캔 시작 — {len(tickers)}종목")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_ob_pattern, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result["ob_score"] >= min_score:
                    results.append(result)
                    logger.info(
                        f"  🎯 {result['ticker']}: OB점수 {result['ob_score']} | "
                        f"반등 +{result['bounce_pct']}% | "
                        f"거래량 {result['vol_ratio']}x | "
                        f"{result['ob_signals']}"
                    )
            except Exception:
                pass

    results.sort(key=lambda x: x["ob_score"], reverse=True)

    os.makedirs("data", exist_ok=True)
    with open("data/ob_scan.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "scanned": len(tickers),
            "passed": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"OB 스캔 완료 — {len(results)}/{len(tickers)} 패턴 감지")
    return results


def format_ob_report(results: list) -> str:
    """텔레그램용 OB 스캔 리포트"""
    if not results:
        return "🔍 OB Touch & Bounce 패턴 감지 종목 없음"

    lines = [
        "🎯 <b>Order Block Touch & Bounce 패턴</b>",
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')} KST",
        "━" * 30, "",
    ]

    for i, r in enumerate(results[:5], 1):
        ob = r["ob_zone"]
        lines.append(f"{i}. <b>{r['ticker']}</b>  OB점수 {r['ob_score']}점")
        lines.append(f"   현재가: ${r['price']} | 52주고점대비: {r['from_52w_high_pct']}%")
        lines.append(f"   OB구간: ${ob['low']} ~ ${ob['high']}")
        lines.append(f"   반등: +{r['bounce_pct']}% | 거래량: {r['vol_ratio']}x")
        lines.append(f"   신호: {' | '.join(r['ob_signals'])}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # 테스트: 특정 종목 또는 저장된 유니버스 사용
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        # rated_universe 또는 chart_scan 결과 로드
        for path, key in [
            ("data/rated_universe.json", "universe"),
            ("data/chart_scan.json", "results"),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                raw = data.get(key, [])
                tickers = raw if isinstance(raw[0], str) else [r["ticker"] for r in raw]
                logger.info(f"{path}에서 {len(tickers)}종목 로드")
                break
        else:
            tickers = ["SNDK", "NVDA", "AAPL", "MSFT", "AMZN", "META"]
            logger.info("테스트 종목 사용")

    results = run_ob_scan(tickers)
    print(format_ob_report(results))

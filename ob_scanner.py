"""
ob_scanner.py — Order Block Touch & Bounce 패턴 스캐너
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


def detect_order_blocks(hist: pd.DataFrame, swing_period: int = 20) -> list:
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
        if curr_high <= prev_high * 1.02:
            continue
        for j in range(i - 1, max(i - 8, 0), -1):
            c = float(close.iloc[j])
            o = float(open_.iloc[j])
            if c >= o:
                continue
            ob_low  = float(low.iloc[j])
            ob_high = float(o)
            ob_body = abs(o - c)
            avg_vol = float(vol.iloc[max(0, j-20):j].mean()) if j > 0 else 1
            vol_ratio = float(vol.iloc[j]) / avg_vol if avg_vol > 0 else 1
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
            break
    obs.sort(key=lambda x: x["ob_idx"], reverse=True)
    return obs


def check_ob_touch_bounce(hist: pd.DataFrame) -> Optional[dict]:
    if len(hist) < 210:
        return None
    close = hist["Close"]
    open_ = hist["Open"]
    high  = hist["High"]
    low   = hist["Low"]
    vol   = hist["Volume"]
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
    obs = detect_order_blocks(hist)
    recent_obs = [ob for ob in obs if ob["ob_idx"] >= len(hist) - 120]
    for ob in recent_obs:
        ob_low  = ob["ob_low"]
        ob_high = ob["ob_high"]
        ob_mid  = ob["ob_mid"]
        tolerance = (ob_high - ob_low) * 0.3
        touched = (today_low <= ob_high + tolerance) and (today_low >= ob_low - tolerance)
        if not touched:
            for lookback in [1, 2, 3]:
                if len(hist) > lookback:
                    prev_low = float(low.iloc[-(lookback+1)])
                    if (prev_low <= ob_high + tolerance) and (prev_low >= ob_low - tolerance):
                        touched = True
                        break
        if not touched:
            continue
        is_bullish   = current > today_open
        above_ob_mid = current > ob_mid
        if not (is_bullish and above_ob_mid):
            continue
        if today_low < ob_low * 0.97:
            continue
        score = 0
        signals = []
        bounce_pct = (current - today_low) / today_low * 100
        if bounce_pct >= 3:
            score += 30
            signals.append(f"강반등 +{bounce_pct:.1f}%")
        elif bounce_pct >= 1.5:
            score += 20
            signals.append(f"반등 +{bounce_pct:.1f}%")
        vol_ratio = vol_today / vol_5d if vol_5d > 0 else 1
        if vol_ratio >= 2.0:
            score += 25
            signals.append(f"거래량 폭증 {vol_ratio:.1f}x")
        elif vol_ratio >= 1.5:
            score += 15
            signals.append(f"거래량 증가 {vol_ratio:.1f}x")
        elif vol_ratio >= 1.2:
            score += 8
        if ob.get("move_pct_after", 0) >= 20:
            score += 20
            signals.append(f"강한OB (이후+{ob['move_pct_after']}%)")
        elif ob.get("move_pct_after", 0) >= 10:
            score += 10
        dist_from_sma20 = abs(current - sma20) / sma20 * 100
        if dist_from_sma20 <= 3:
            score += 15
            signals.append("20일선 근접")
        candle_range = today_high - today_low
        lower_wick = today_open - today_low if today_open > current else current - today_low
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


def scan_ob_pattern(ticker: str) -> Optional[dict]:
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


def run_ob_scan(tickers: list, min_score: int = 40) -> list:
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
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        tickers = ["SNDK", "NVDA", "AAPL", "MSFT", "AMZN", "META"]
    results = run_ob_scan(tickers)
    print(format_ob_report(results))

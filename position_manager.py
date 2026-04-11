"""
MODULE 8: 포지션 관리 (position_manager.py)
- 매수 등록 시 자동 손절/익절/트레일링 설정
- 포지션 상태 4단계: HOLD / WATCH / STOP / TARGET
- 최대 보유 10종목 제한
- positions.json 기반 영속 저장
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POSITIONS_FILE = os.environ.get("POSITIONS_FILE", "data/positions.json")
MAX_POSITIONS = 10


# ── 포지션 저장/로드 ──────────────────────────────────────────

def load_positions() -> list:
    """positions.json에서 포지션 로드"""
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_positions(positions: list):
    """positions.json에 포지션 저장"""
    os.makedirs(os.path.dirname(POSITIONS_FILE) or ".", exist_ok=True)
    with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)


# ── 매수 등록 ──────────────────────────────────────────────────

def add_position(ticker: str, entry_price: float, shares: int = 0, note: str = "") -> dict:
    """
    신규 포지션 등록.
    자동 설정:
      손절가: 매수가 × 0.93 (-7%)
      1차 익절: 매수가 × 1.15 (+15%)
      2차 익절: 매수가 × 1.25 (+25%)
      트레일링 스탑: 고점 대비 -8%
    """
    positions = load_positions()

    # 최대 보유 체크
    active = [p for p in positions if p.get("status") != "CLOSED"]
    if len(active) >= MAX_POSITIONS:
        logger.warning(f"최대 보유 {MAX_POSITIONS}종목 초과 — 신규 진입 차단")
        return {"error": f"최대 보유 {MAX_POSITIONS}종목 초과. 기존 종목 정리 후 진입하세요.", "blocked": True}

    # 중복 체크
    for p in active:
        if p["ticker"] == ticker:
            logger.warning(f"{ticker} 이미 보유 중 — 중복 등록 차단")
            return {"error": f"{ticker} 이미 보유 중", "blocked": True}

    position = {
        "ticker": ticker,
        "entry_price": entry_price,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "shares": shares,
        "note": note,
        "stop_loss": round(entry_price * 0.93, 2),
        "target_1": round(entry_price * 1.15, 2),
        "target_2": round(entry_price * 1.25, 2),
        "trailing_stop_pct": 8.0,
        "highest_price": entry_price,
        "trailing_stop_price": round(entry_price * 0.92, 2),
        "status": "HOLD",
        "pnl_pct": 0.0,
        "last_updated": datetime.now().isoformat(),
    }

    positions.append(position)
    save_positions(positions)
    logger.info(f"✅ {ticker} 포지션 등록: ${entry_price} | 손절 ${position['stop_loss']} | 1차 목표 ${position['target_1']}")
    return position


# ── 포지션 제거 ────────────────────────────────────────────────

def remove_position(ticker: str, reason: str = "수동 정리") -> bool:
    """포지션 종료 (CLOSED 상태로 변경)"""
    positions = load_positions()
    found = False
    for p in positions:
        if p["ticker"] == ticker and p.get("status") != "CLOSED":
            p["status"] = "CLOSED"
            p["close_date"] = datetime.now().strftime("%Y-%m-%d")
            p["close_reason"] = reason
            p["last_updated"] = datetime.now().isoformat()
            found = True
            logger.info(f"🔒 {ticker} 포지션 종료: {reason}")
            break

    if found:
        save_positions(positions)
    else:
        logger.warning(f"{ticker} 활성 포지션 없음")
    return found


# ── 포지션 트래킹 ──────────────────────────────────────────────

def update_position(position: dict) -> dict:
    """
    개별 포지션 현재가 반영 + 상태 업데이트.
    상태 4단계:
      🟢 HOLD: 정상 보유
      🟡 WATCH: -5% 이하
      🔴 STOP: -7% 터치
      🎯 TARGET: +15% 도달
    """
    ticker = position["ticker"]
    entry = position["entry_price"]

    try:
        current = float(yf.Ticker(ticker).info.get("regularMarketPrice", 0))
        if current <= 0:
            hist = yf.Ticker(ticker).history(period="1d")
            current = float(hist["Close"].iloc[-1]) if not hist.empty else 0
    except Exception:
        current = 0

    if current <= 0:
        position["last_updated"] = datetime.now().isoformat()
        return position

    pnl_pct = round((current - entry) / entry * 100, 2)

    # 고점 갱신
    highest = max(position.get("highest_price", entry), current)
    trailing_stop = round(highest * (1 - position["trailing_stop_pct"] / 100), 2)

    position["current_price"] = current
    position["pnl_pct"] = pnl_pct
    position["highest_price"] = highest
    position["trailing_stop_price"] = trailing_stop
    position["last_updated"] = datetime.now().isoformat()

    # 상태 판정
    if current <= position["stop_loss"] or current <= trailing_stop:
        position["status"] = "STOP"
        position["alert"] = f"🔴 즉시 손절! 현재가 ${current} | 손절선 ${position['stop_loss']} | 트레일링 ${trailing_stop}"
    elif current >= position["target_1"]:
        position["status"] = "TARGET"
        if current >= position["target_2"]:
            position["alert"] = f"🎯🎯 2차 목표 도달! +{pnl_pct}% | ${current}"
        else:
            position["alert"] = f"🎯 1차 목표 도달! +{pnl_pct}% | ${current} — 익절 검토"
    elif pnl_pct <= -5:
        position["status"] = "WATCH"
        position["alert"] = f"🟡 모니터링 강화: {pnl_pct}% | ${current}"
    else:
        position["status"] = "HOLD"
        position["alert"] = None

    return position


def track_all_positions() -> dict:
    """전체 포지션 트래킹"""
    positions = load_positions()
    active = [p for p in positions if p.get("status") != "CLOSED"]

    logger.info(f"포지션 트래킹 시작 — {len(active)}종목")

    alerts = []
    for p in active:
        updated = update_position(p)
        status_icon = {"HOLD": "🟢", "WATCH": "🟡", "STOP": "🔴", "TARGET": "🎯"}.get(updated["status"], "⚪")
        logger.info(
            f"  {status_icon} {updated['ticker']}: ${updated.get('current_price', 0)} "
            f"({updated['pnl_pct']:+.1f}%) | 상태: {updated['status']}"
        )
        if updated.get("alert"):
            alerts.append(updated["alert"])

    save_positions(positions)

    # 요약 통계
    pnls = [p["pnl_pct"] for p in active if "pnl_pct" in p]
    avg_pnl = round(sum(pnls) / len(pnls), 2) if pnls else 0
    stops = [p for p in active if p["status"] == "STOP"]
    targets = [p for p in active if p["status"] == "TARGET"]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "active_count": len(active),
        "max_positions": MAX_POSITIONS,
        "avg_pnl_pct": avg_pnl,
        "stop_alerts": len(stops),
        "target_alerts": len(targets),
        "alerts": alerts,
        "positions": active,
    }

    with open("data/position_tracking.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"포지션 트래킹 완료 — 평균 수익률 {avg_pnl}% | 손절 {len(stops)} | 목표 {len(targets)}")
    return summary


# ── 포트폴리오 현황 ────────────────────────────────────────────

def get_portfolio_summary() -> dict:
    """현재 포트폴리오 요약"""
    positions = load_positions()
    active = [p for p in positions if p.get("status") != "CLOSED"]
    closed = [p for p in positions if p.get("status") == "CLOSED"]

    return {
        "active": len(active),
        "max": MAX_POSITIONS,
        "available_slots": MAX_POSITIONS - len(active),
        "active_positions": [
            {
                "ticker": p["ticker"],
                "entry": p["entry_price"],
                "current": p.get("current_price"),
                "pnl": p.get("pnl_pct", 0),
                "status": p.get("status"),
                "entry_date": p.get("entry_date"),
            }
            for p in active
        ],
        "closed_count": len(closed),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "add" and len(sys.argv) >= 4:
            ticker = sys.argv[2].upper()
            price = float(sys.argv[3])
            shares = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            result = add_position(ticker, price, shares)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif cmd == "remove" and len(sys.argv) >= 3:
            ticker = sys.argv[2].upper()
            reason = sys.argv[3] if len(sys.argv) > 3 else "수동 정리"
            remove_position(ticker, reason)

        elif cmd == "track":
            summary = track_all_positions()
            print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

        elif cmd == "portfolio":
            summary = get_portfolio_summary()
            print(json.dumps(summary, indent=2, ensure_ascii=False))

        else:
            print("사용법: python position_manager.py [add TICKER PRICE [SHARES] | remove TICKER | track | portfolio]")
    else:
        # 기본: 트래킹
        summary = track_all_positions()
        for alert in summary.get("alerts", []):
            print(alert)

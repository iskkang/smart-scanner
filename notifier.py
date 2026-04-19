"""
MODULE 9: 텔레그램 리포트 (notifier.py)
- 매일 아침 종합 리포트 전송
- 포지션 알림 (손절/익절)
- 스캔 결과 추천 리포트
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_MSG_LENGTH = 4096


def send_telegram(text: str) -> bool:
    """텔레그램 메시지 전송. 4096자 초과 시 분할."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("텔레그램 설정 미완료 — 콘솔 출력만 수행")
        print(text)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    chunks = []
    while len(text) > MAX_MSG_LENGTH:
        split_at = text.rfind("\n", 0, MAX_MSG_LENGTH)
        if split_at == -1:
            split_at = MAX_MSG_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    chunks.append(text)

    success = True
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=15)
            if not resp.ok:
                logger.error(f"텔레그램 전송 실패: {resp.text}")
                success = False
        except Exception as e:
            logger.error(f"텔레그램 전송 오류: {e}")
            success = False

    return success


# ── 리포트 포맷터 ──────────────────────────────────────────────

def format_star_rating(score: int) -> str:
    """점수 기반 별점 (0~100 → ★1~5)"""
    stars = min(5, max(1, score // 20))
    return "★" * stars + "☆" * (5 - stars)


def build_daily_report() -> str:
    """매일 아침 종합 리포트 빌드"""
    lines = []
    lines.append("📊 <b>SMART SCANNER DAILY REPORT</b>")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')} KST")
    lines.append("━" * 30)

    # ── 거시환경 ──
    macro = _load_json("data/macro_analysis.json")
    if macro:
        ai = macro.get("ai_analysis") or {}
        raw = macro.get("raw_data") or {}
        lines.append("")
        lines.append("🌍 <b>거시환경</b>")
        lines.append(f"  테마: {ai.get('dominant_theme', 'N/A')}")
        lines.append(f"  리스크: {ai.get('risk_level', 'N/A')}")
        lines.append(f"  수혜섹터: {', '.join(ai.get('favored_sectors', []))}")
        lines.append(f"  회피섹터: {', '.join(ai.get('avoid_sectors', []))}")

        # 주요 지표
        vix = raw.get("vix")
        dxy = raw.get("dxy")
        us10y = raw.get("us10y")
        if vix or dxy or us10y:
            indicators = []
            if vix:
                indicators.append(f"VIX {vix}")
            if dxy:
                indicators.append(f"DXY {dxy}")
            if us10y:
                indicators.append(f"10Y {us10y}%")
            lines.append(f"  지표: {' | '.join(indicators)}")

        warnings = ai.get("special_warnings", [])
        if warnings:
            lines.append(f"  ⚠️ {', '.join(warnings)}")

    lines.append("")
    lines.append("━" * 30)

    # ── 최종 추천 ──
    backtest = _load_json("data/backtest_results.json")
    chart = _load_json("data/chart_scan.json")
    valuation = _load_json("data/valuation_analysis.json")

    passed = backtest.get("passed", []) if backtest else []
    chart_map = {r["ticker"]: r for r in (chart.get("results", []) if chart else [])}
    val_map = {}
    if valuation:
        for r in valuation.get("passed", []):
            val_map[r["ticker"]] = r

    # ── 섹터당 최대 2종목 캡 적용 ──
    def apply_sector_cap(recs: list, max_per_sector: int = 2) -> list:
        sector_count = {}
        result = []
        for rec in recs:
            t = rec["ticker"]
            sector = val_map.get(t, {}).get("sector_code", "GENERAL")
            cnt = sector_count.get(sector, 0)
            if cnt < max_per_sector:
                result.append(rec)
                sector_count[sector] = cnt + 1
            if len(result) >= 5:
                break
        return result

    top5 = apply_sector_cap(passed)
    lines.append(f"📡 <b>오늘의 최종 추천</b> ({len(top5)}종목 / 전체 {len(passed)}종목 통과)")
    lines.append("")

    if not top5:
        lines.append("  조건 충족 종목 없음")
    else:
        for i, rec in enumerate(top5, 1):
            t = rec["ticker"]
            chart_info = chart_map.get(t, {})
            val_info   = val_map.get(t, {})
            bt         = rec.get("backtest", {})
            strategies = rec.get("passed_strategies", [])

            score = chart_info.get("chart_score", 0) + val_info.get("val_score", 0)
            stars = format_star_rating(score)

            # 새 valuation 필드
            gap      = val_info.get("target_gap_pct")
            gap_str  = f"+{gap:.1f}%" if gap is not None else "N/A"
            sector   = val_info.get("sector_code", "")
            signals  = val_info.get("signals", [])
            sig_str  = " | ".join(signals[:2]) if signals else "N/A"

            # 백테스트 신호 수 포함
            bt_win    = bt.get("win_rate", 0)
            bt_avg    = bt.get("avg_return", 0)
            bt_n      = bt.get("signal_count", 0)

            lines.append(f"{i}. <b>{t}</b> {stars} [{sector}]")
            lines.append(f"   현재가: ${chart_info.get('price', 'N/A')} | 눌림: {chart_info.get('pullback_pct', 'N/A')}%")
            lines.append(f"   밸류: {sig_str}")
            lines.append(f"   목표가 괴리: {gap_str}")

            strat_str = " ".join([f"{s} ✅" for s in strategies])
            lines.append(f"   통과전략: {strat_str}")

            lines.append(f"   백테스트: 승률 {bt_win}% | 평균수익 {bt_avg:+.2f}% | 신호 {bt_n}회")
            lines.append("")

    lines.append("━" * 30)

    # ── 포지션 현황 ──
    tracking = _load_json("data/position_tracking.json")
    if tracking:
        active = tracking.get("active_count", 0)
        max_pos = tracking.get("max_positions", MAX_MSG_LENGTH)
        avg_pnl = tracking.get("avg_pnl_pct", 0)

        lines.append(f"📂 <b>포지션 현황</b> ({active}/{max_pos})")
        lines.append(f"  평균 수익률: {avg_pnl:+.1f}%")

        positions = tracking.get("positions", [])
        for p in positions:
            icon = {"HOLD": "🟢", "WATCH": "🟡", "STOP": "🔴", "TARGET": "🎯"}.get(p.get("status"), "⚪")
            lines.append(
                f"  {icon} {p['ticker']}: ${p.get('current_price', 'N/A')} "
                f"({p.get('pnl_pct', 0):+.1f}%)"
            )

        # 긴급 알림
        alerts = tracking.get("alerts", [])
        if alerts:
            lines.append("")
            lines.append("🚨 <b>긴급 알림</b>")
            for a in alerts:
                lines.append(f"  {a}")

    lines.append("")
    lines.append("━" * 30)

    return "\n".join(lines)


def build_alert_message(alerts: list) -> str:
    """포지션 알림 메시지"""
    if not alerts:
        return ""
    lines = ["🚨 <b>SMART SCANNER 포지션 알림</b>", ""]
    for a in alerts:
        lines.append(f"  {a}")
    return "\n".join(lines)


def build_scan_complete_message(passed_count: int, total_scanned: int) -> str:
    """스캔 완료 알림"""
    return (
        f"🔍 <b>스캔 완료</b>\n"
        f"  스캔 대상: {total_scanned}종목\n"
        f"  최종 통과: {passed_count}종목\n"
        f"  상세 리포트는 아침 리포트에서 확인"
    )


# ── 유틸 ───────────────────────────────────────────────────────

def _load_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ── 실행 ───────────────────────────────────────────────────────

def send_daily_report():
    """매일 아침 리포트 전송"""
    report = build_daily_report()
    logger.info("매일 아침 리포트 전송")
    send_telegram(report)
    return report


def send_position_alerts():
    """포지션 알림만 전송 (트래킹 후)"""
    tracking = _load_json("data/position_tracking.json")
    if not tracking:
        return
    alerts = tracking.get("alerts", [])
    if alerts:
        msg = build_alert_message(alerts)
        send_telegram(msg)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    if cmd == "report":
        report = send_daily_report()
    elif cmd == "alerts":
        send_position_alerts()
    elif cmd == "test":
        send_telegram("🧪 SMART SCANNER 텔레그램 연결 테스트 성공!")
    else:
        print("사용법: python notifier.py [report | alerts | test]")

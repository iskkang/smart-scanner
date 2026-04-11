"""
SMART SCANNER — 메인 오케스트레이터 (main.py)
전체 파이프라인 순차 실행:
  1. 거시환경 분석 → 2. 차트 스캔 → 3. 밸류에이션 →
  4. 기관 동향 → 5. 뉴스/공매도 → 6. 월가 리포트 →
  7. 백테스팅 → 8. 텔레그램 리포트
"""

import sys
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_scan():
    """전체 스캔 파이프라인"""
    logger.info("=" * 50)
    logger.info("SMART SCANNER 전체 스캔 시작")
    logger.info("=" * 50)

    os.makedirs("data", exist_ok=True)

    # ── 1단계: 거시환경 분석 ──
    logger.info("\n[1/7] 거시환경 분석")
    try:
        from macro_analyzer import run_macro_analysis
        macro = run_macro_analysis()
        favored = (macro.get("ai_analysis") or {}).get("favored_sectors")
        logger.info(f"  수혜섹터: {favored}")
    except Exception as e:
        logger.error(f"  거시환경 분석 실패 — 스킵: {e}")
        favored = None

    # ── 2단계: 차트 스캔 ──
    logger.info("\n[2/7] 차트 스캔")
    try:
        from chart_scanner import run_chart_scan
        chart_passed = run_chart_scan(favored_sectors=favored)
        tickers = [r["ticker"] for r in chart_passed]
        logger.info(f"  차트 통과: {len(tickers)}종목")
    except Exception as e:
        logger.error(f"  차트 스캔 실패: {e}")
        tickers = []

    if not tickers:
        logger.info("차트 통과 종목 없음 — 스캔 종료")
        _send_empty_report()
        return

    # ── 3단계: 밸류에이션 ──
    logger.info(f"\n[3/7] 밸류에이션 분석 ({len(tickers)}종목)")
    try:
        from valuation_analyzer import run_valuation_analysis
        val_passed = run_valuation_analysis(tickers)
        tickers = [r["ticker"] for r in val_passed]
        logger.info(f"  밸류에이션 통과: {len(tickers)}종목")
    except Exception as e:
        logger.error(f"  밸류에이션 분석 실패 — 차트 통과 종목 유지: {e}")

    if not tickers:
        logger.info("밸류에이션 통과 종목 없음 — 스캔 종료")
        _send_empty_report()
        return

    # ── 4단계: 기관 동향 ──
    logger.info(f"\n[4/7] 기관 동향 분석 ({len(tickers)}종목)")
    try:
        from institutional_tracker import run_institutional_analysis
        inst_passed = run_institutional_analysis(tickers)
        tickers = [r["ticker"] for r in inst_passed]
        logger.info(f"  기관 통과: {len(tickers)}종목")
    except Exception as e:
        logger.error(f"  기관 분석 실패 — 이전 통과 종목 유지: {e}")

    if not tickers:
        logger.info("기관 통과 종목 없음 — 스캔 종료")
        _send_empty_report()
        return

    # ── 5단계: 뉴스/공매도 ──
    logger.info(f"\n[5/7] 뉴스/공매도 검증 ({len(tickers)}종목)")
    try:
        from news_analyzer import run_news_analysis
        news_passed = run_news_analysis(tickers)
        tickers = [r["ticker"] for r in news_passed]
        logger.info(f"  뉴스 통과: {len(tickers)}종목")
    except Exception as e:
        logger.error(f"  뉴스 분석 실패 — 이전 통과 종목 유지: {e}")

    if not tickers:
        logger.info("뉴스 통과 종목 없음 — 스캔 종료")
        _send_empty_report()
        return

    # ── 6단계: 월가 리포트 ──
    logger.info(f"\n[6/7] 월가 리포트 교차검증 ({len(tickers)}종목)")
    try:
        from wallstreet_report import run_wallstreet_analysis
        ws_passed = run_wallstreet_analysis(tickers)
        tickers = [r["ticker"] for r in ws_passed]
        logger.info(f"  월가 검증 통과: {len(tickers)}종목")
    except Exception as e:
        logger.error(f"  월가 검증 실패 — 이전 통과 종목 유지: {e}")

    if not tickers:
        logger.info("월가 검증 통과 종목 없음 — 스캔 종료")
        _send_empty_report()
        return

    # ── 7단계: 백테스팅 ──
    logger.info(f"\n[7/7] 백테스팅 ({len(tickers)}종목)")
    try:
        from backtester import run_backtest
        final_passed = run_backtest(tickers)
        logger.info(f"  최종 통과: {len(final_passed)}종목")
    except Exception as e:
        logger.error(f"  백테스팅 실패: {e}")
        final_passed = []

    # ── 리포트 전송 ──
    logger.info("\n리포트 전송")
    try:
        from notifier import send_daily_report
        send_daily_report()
    except Exception as e:
        logger.error(f"  리포트 전송 실패: {e}")

    logger.info("=" * 50)
    logger.info("SMART SCANNER 전체 스캔 완료")
    logger.info("=" * 50)


def run_tracking():
    """포지션 트래킹 + 알림"""
    logger.info("포지션 트래킹 실행")
    try:
        from position_manager import track_all_positions
        summary = track_all_positions()

        from notifier import send_position_alerts, send_daily_report
        send_position_alerts()
        send_daily_report()
    except Exception as e:
        logger.error(f"트래킹 실패: {e}")


def run_portfolio():
    """포트폴리오 현황 조회"""
    from position_manager import get_portfolio_summary
    summary = get_portfolio_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_add(ticker: str, price: float, shares: int = 0):
    """종목 추가"""
    from position_manager import add_position
    result = add_position(ticker.upper(), price, shares)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_panic_scan_cmd(tickers: list = None):
    """패닉 셀 저가 매수 스캔"""
    logger.info("패닉 셀 저가 매수 스캔 실행")
    try:
        from panic_scanner import run_panic_scan, format_panic_report
        passed = run_panic_scan(custom_tickers=tickers if tickers else None)

        from notifier import send_telegram
        report = format_panic_report(passed)
        send_telegram(report)
    except Exception as e:
        logger.error(f"패닉 스캔 실패: {e}")


def run_remove(ticker: str, reason: str = "수동 정리"):
    """종목 제거"""
    from position_manager import remove_position
    remove_position(ticker.upper(), reason)


def _send_empty_report():
    """통과 종목 없을 때 리포트"""
    try:
        from notifier import send_daily_report
        send_daily_report()
    except Exception:
        pass


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"

    if cmd == "scan":
        run_full_scan()
    elif cmd == "track":
        run_tracking()
    elif cmd == "portfolio":
        run_portfolio()
    elif cmd == "panic":
        # 특정 종목 지정 가능: python main.py panic SNDK AMKR MU
        tickers = [t.upper() for t in sys.argv[2:]] if len(sys.argv) > 2 else None
        run_panic_scan_cmd(tickers)
    elif cmd == "add" and len(sys.argv) >= 4:
        run_add(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif cmd == "remove" and len(sys.argv) >= 3:
        run_remove(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "수동 정리")
    else:
        print("""
SMART SCANNER 사용법:
  python main.py scan                        — 전체 스캔 실행
  python main.py track                       — 포지션 트래킹 + 아침 리포트
  python main.py portfolio                   — 포트폴리오 현황
  python main.py panic                       — 패닉 셀 저가 매수 스캔 (S&P500 전체)
  python main.py panic SNDK AMKR MU          — 특정 종목만 패닉 스캔
  python main.py add TICKER PRICE [SHARES]   — 종목 추가
  python main.py remove TICKER [REASON]      — 종목 제거
        """)

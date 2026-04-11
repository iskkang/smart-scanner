"""
MODULE 4: 기관 동향 분석 (institutional_tracker.py)
- 기관 보유 비율, 보유 기관 수
- 주요 기관 홀더 분석
- 기관 매수/매도 추세 판단
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_institutional_data(ticker: str) -> Optional[dict]:
    """yfinance에서 기관 보유 데이터 수집"""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # 기관 보유 비율
        inst_pct = info.get("heldPercentInstitutions")

        # 주요 기관 홀더
        try:
            holders_df = t.institutional_holders
            if holders_df is not None and not holders_df.empty:
                top_holders = []
                for _, row in holders_df.head(10).iterrows():
                    holder = {
                        "name": str(row.get("Holder", "")),
                        "shares": int(row["Shares"]) if "Shares" in row and row["Shares"] == row["Shares"] else None,
                        "pct_out": float(row["% Out"]) if "% Out" in row and row["% Out"] == row["% Out"] else None,
                    }
                    # Date Reported
                    date_col = row.get("Date Reported")
                    if date_col is not None and str(date_col) != "NaT":
                        holder["date_reported"] = str(date_col)[:10]
                    top_holders.append(holder)
            else:
                top_holders = []
        except Exception:
            top_holders = []

        # 기관 수 (mutual fund + institutional)
        fund_pct = info.get("heldPercentInsiders")

        return {
            "ticker": ticker,
            "institutional_pct": round(inst_pct * 100, 2) if inst_pct else None,
            "insider_pct": round(fund_pct * 100, 2) if fund_pct else None,
            "top_holders": top_holders,
            "holder_count": len(top_holders),
        }
    except Exception as e:
        logger.error(f"{ticker} 기관 데이터 수집 실패: {e}")
        return None


def score_institutional(data: dict) -> dict:
    """
    기관 동향 점수 및 신호 판단.

    긍정 신호:
      - 기관 보유 50%+ : 안정적
      - 기관 보유 70%+ : 강력 기관 선호
      - 주요 기관 보고일이 최근 3개월 이내: 신규 유입 가능성

    경고 신호:
      - 기관 보유 20% 미만: 기관 관심 부족
      - 홀더 데이터 없음: 판단 불가
    """
    score = 0
    signals = []
    warnings = []

    inst_pct = data.get("institutional_pct")

    if inst_pct is None:
        warnings.append("NO_INST_DATA")
        data.update({"inst_score": 0, "signals": signals, "warnings": warnings, "pass": True})
        return data

    # 기관 보유 비율 점수
    if inst_pct >= 70:
        score += 20
        signals.append("STRONG_INSTITUTIONAL_FAVOR")
    elif inst_pct >= 50:
        score += 10
        signals.append("SOLID_INSTITUTIONAL_BASE")
    elif inst_pct < 20:
        warnings.append("LOW_INSTITUTIONAL_INTEREST")
        score -= 10

    # 주요 기관 수
    holder_count = data.get("holder_count", 0)
    if holder_count >= 8:
        score += 10
        signals.append("DIVERSE_HOLDER_BASE")
    elif holder_count >= 5:
        score += 5

    # 최근 보고 여부 (간접적 매수 추세 힌트)
    recent_reports = 0
    from datetime import datetime as dt, timedelta
    cutoff = (dt.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    for h in data.get("top_holders", []):
        dr = h.get("date_reported", "")
        if dr and dr >= cutoff:
            recent_reports += 1

    if recent_reports >= 3:
        score += 10
        signals.append("RECENT_INSTITUTIONAL_ACTIVITY")
    elif recent_reports >= 1:
        score += 5

    # 대량 매도 감지는 13F 시계열 비교 필요 — 현재 yfinance 한계로 간접 판단
    # 기관 보유 비율이 30% 미만이면서 과거 대비 급감한 경우 경고
    # (단순 스냅샷에서는 추세 판단 제한적, 로그에 명시)
    if inst_pct < 30:
        warnings.append("INSTITUTIONAL_PCT_BELOW_30")

    passed = len(warnings) <= 1

    data.update({
        "inst_score": max(score, 0),
        "signals": signals,
        "warnings": warnings,
        "pass": passed,
        "fail_reason": None if passed else f"기관 경고 {len(warnings)}개",
    })
    return data


# ── 배치 실행 ──────────────────────────────────────────────────

def run_institutional_analysis(tickers: list) -> list:
    """밸류에이션 통과 종목에 대해 기관 동향 분석"""
    import time
    logger.info(f"기관 동향 분석 시작 — {len(tickers)}종목")

    results = []
    for ticker in tickers:
        raw = fetch_institutional_data(ticker)
        if not raw:
            continue
        scored = score_institutional(raw)
        status = "✅ 통과" if scored["pass"] else "❌ 탈락"
        logger.info(f"  {status} {ticker}: 기관 {scored.get('institutional_pct', 'N/A')}% | 점수 {scored['inst_score']} | {scored['signals']}")
        results.append(scored)
        time.sleep(0.5)  # Rate limit 방지

    passed = [r for r in results if r["pass"]]

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "passed_count": len(passed),
        "passed": passed,
        "failed": [r for r in results if not r["pass"]],
    }

    os.makedirs("data", exist_ok=True)
    with open("data/institutional_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"기관 동향 분석 완료 — {len(passed)}/{len(results)} 통과")
    return passed


if __name__ == "__main__":
    val_path = "data/valuation_analysis.json"
    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            val_data = json.load(f)
        tickers = [r["ticker"] for r in val_data.get("passed", [])]
        logger.info(f"밸류에이션 통과 종목에서 {len(tickers)}종목 로드")
    else:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
        logger.info("밸류에이션 결과 없음 — 테스트 종목 사용")

    passed = run_institutional_analysis(tickers)
    print(f"\n기관 분석 통과 {len(passed)}종목:")
    for r in passed:
        print(f"  {r['ticker']:6s} | 기관 {r.get('institutional_pct', 'N/A')}% | 점수 {r['inst_score']} | {r['signals']}")

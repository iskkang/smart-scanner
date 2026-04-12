"""
패닉 셀 저가 매수 스캐너 (panic_scanner.py)

전략:
  우량 기업이 매크로/외부 이벤트(전쟁, 금리 쇼크, 지수 급락 등)로
  단기 패닉 셀을 당해 저평가 구간에 진입했을 때를 포착.

파이프라인:
  ① 품질 필터     — ROE 10%+, EPS 흑자, 부채 적정, 기관 보유 50%+
  ② 패닉 급락     — 30일 내 -15% 이상, 급락 시 거래량 2배+
                    52주 고점 대비 -20% ~ -50% 구간
  ③ 회복 신호     — BOS (저점 상승), 거래량 수축 후 반등, 5일선 회복
  ④ Claude 분석   — 급락 원인이 외부 이벤트인지 기업 자체 악재인지 판별
  ⑤ 목표가 유효성 — 애널리스트 목표가가 급락 전 수준 유지 여부
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import numpy as np
import requests
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
MIN_MARKET_CAP = 10_000_000_000  # $10B+


# ── 유니버스 수집 ──────────────────────────────────────────────

def fetch_universe() -> list:
    """S&P 500 + 시가총액 $10B+ 필터"""
    try:
        import io
        import pandas as pd
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
        sp500 = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 500 수집: {len(sp500)}종목")
    except Exception as e:
        logger.warning(f"S&P 500 수집 실패 — 주요 종목 사용: {e}")
        sp500 = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
            "JPM", "V", "MA", "UNH", "LLY", "JNJ", "XOM", "CVX", "HD", "PG",
            "AVGO", "COST", "MRK", "ABBV", "CRM", "AMD", "INTC", "MU", "AMAT",
            "QCOM", "TXN", "KLAC", "LRCX", "SNDK", "AMKR", "PSIX",
        ]

    def get_cap(ticker):
        try:
            cap = yf.Ticker(ticker).info.get("marketCap")
            return ticker, cap
        except Exception:
            return ticker, None

    passed = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(get_cap, t): t for t in sp500}
        for future in as_completed(futures):
            ticker, cap = future.result()
            if cap and cap >= MIN_MARKET_CAP:
                passed.append(ticker)

    logger.info(f"시가총액 필터 완료: {len(passed)}종목 ($10B+)")
    return passed


# ── ① 품질 필터 ────────────────────────────────────────────────

def check_quality(ticker: str) -> Optional[dict]:
    """
    우량 기업 필터.
    통과 조건 (4개 이상):
      - ROE 10%+
      - EPS 흑자 (trailingEps > 0)
      - 부채비율 200% 이하 (debtToEquity)
      - 기관 보유 50%+
      - 영업이익률 8%+
      - 매출 성장률 0%+
    """
    try:
        info = yf.Ticker(ticker).info or {}
        checks = {}
        passed = 0

        roe = info.get("returnOnEquity")
        checks["roe"] = round(roe * 100, 1) if roe else None
        if roe and roe >= 0.10:
            passed += 1

        eps = info.get("trailingEps")
        checks["eps_positive"] = eps and eps > 0
        if checks["eps_positive"]:
            passed += 1

        de = info.get("debtToEquity")
        checks["debt_to_equity"] = de
        if de is not None and de <= 200:
            passed += 1

        inst = info.get("heldPercentInstitutions")
        checks["inst_pct"] = round(inst * 100, 1) if inst else None
        if inst and inst >= 0.50:
            passed += 1

        op_margin = info.get("operatingMargins")
        checks["op_margin"] = round(op_margin * 100, 1) if op_margin else None
        if op_margin and op_margin >= 0.08:
            passed += 1

        rev_growth = info.get("revenueGrowth")
        checks["rev_growth"] = round(rev_growth * 100, 1) if rev_growth else None
        if rev_growth and rev_growth >= 0:
            passed += 1

        quality_pass = passed >= 4

        return {
            "ticker": ticker,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "target_mean_price": info.get("targetMeanPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions") or 0,
            "quality_checks": checks,
            "quality_score": passed,
            "quality_pass": quality_pass,
        }
    except Exception as e:
        logger.error(f"{ticker} 품질 필터 실패: {e}")
        return None


# ── ② 패닉 급락 감지 ──────────────────────────────────────────

def detect_panic_drop(ticker: str, quality_data: dict) -> Optional[dict]:
    """
    패닉 셀 급락 감지.
    조건:
      - 최근 30일 내 단일 세션 -5% 이상 급락일 존재
      - 해당 급락일 거래량 > 20일 평균 거래량 2배 (패닉 셀 시그니처)
      - 현재가가 30일 전 대비 -15% 이상 하락
      - 52주 고점 대비 -20% ~ -55% 구간 (낙폭과대 but 완전 붕괴 아님)
    """
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 60:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        current = float(close.iloc[-1])

        # 52주 고점
        high_52w = float(close.max())
        from_52w_high = (current - high_52w) / high_52w * 100

        # 낙폭과대 구간 체크 (-20% ~ -55%)
        if not (-55 <= from_52w_high <= -20):
            return None

        # 30일 전 대비 하락률
        price_30d_ago = float(close.iloc[-30]) if len(close) >= 30 else float(close.iloc[0])
        drop_30d = (current - price_30d_ago) / price_30d_ago * 100
        if drop_30d > -10:  # 최소 -10% 하락
            return None

        # 최근 30일 내 패닉 급락일 탐지 (단일 세션 -5% 이상 + 거래량 2배+)
        recent_30 = hist.iloc[-30:]
        vol_20d_avg = float(volume.iloc[-50:-30].mean()) if len(volume) >= 50 else float(volume.mean())

        panic_days = []
        for i in range(1, len(recent_30)):
            day_ret = (float(recent_30["Close"].iloc[i]) - float(recent_30["Close"].iloc[i-1])) / float(recent_30["Close"].iloc[i-1]) * 100
            day_vol = float(recent_30["Volume"].iloc[i])
            if day_ret <= -5 and vol_20d_avg > 0 and day_vol >= vol_20d_avg * 2:
                panic_days.append({
                    "date": str(recent_30.index[i].date()),
                    "drop_pct": round(day_ret, 2),
                    "vol_ratio": round(day_vol / vol_20d_avg, 1),
                })

        if not panic_days:
            return None

        # 최대 패닉일
        worst_panic = min(panic_days, key=lambda x: x["drop_pct"])

        return {
            **quality_data,
            "from_52w_high_pct": round(from_52w_high, 2),
            "drop_30d_pct": round(drop_30d, 2),
            "panic_days": panic_days,
            "worst_panic": worst_panic,
            "high_52w": round(high_52w, 2),
        }

    except Exception as e:
        logger.error(f"{ticker} 패닉 감지 실패: {e}")
        return None


# ── ③ 회복 신호 (BOS + 기술적 확인) ──────────────────────────

def check_recovery_signal(ticker: str, panic_data: dict) -> Optional[dict]:
    """
    회복 신호 탐지.
      BOS (Break of Structure): 최근 저점이 직전 저점보다 높아짐
      거래량 수축 후 반등 시 거래량 증가
      현재가 > 5일 이동평균 회복
      RSI 30~55 (과매도 탈출 초기)
    """
    try:
        hist = yf.Ticker(ticker).history(period="3mo")
        if len(hist) < 40:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        current = float(close.iloc[-1])

        signals = []
        signal_count = 0

        # BOS: 최근 3개 주요 저점 비교
        # 10일 단위로 저점 추출
        lows = []
        for i in range(0, len(close) - 10, 10):
            segment = close.iloc[i:i+10]
            lows.append((i, float(segment.min())))

        bos_detected = False
        if len(lows) >= 3:
            # 최근 3구간 저점이 상승하는지
            recent_lows = lows[-3:]
            if recent_lows[-1][1] > recent_lows[-2][1] > recent_lows[0][1]:
                bos_detected = True
                signal_count += 2  # BOS는 가장 중요한 신호
                signals.append("BOS 확인 (저점 상승 구조 전환)")
            elif recent_lows[-1][1] > recent_lows[-2][1]:
                signal_count += 1
                signals.append("BOS 부분 신호 (최근 저점 상승)")

        # 5일선 회복
        sma5 = float(close.rolling(5).mean().iloc[-1])
        above_sma5 = current > sma5
        if above_sma5:
            signal_count += 1
            signals.append(f"5일선 회복 (현재 ${current:.2f} > SMA5 ${sma5:.2f})")

        # 거래량 수축 후 반등 (최근 5일 거래량 < 20일 평균 → 수축 후 오늘 증가)
        vol_5d = float(volume.iloc[-5:].mean())
        vol_20d = float(volume.iloc[-20:].mean())
        vol_today = float(volume.iloc[-1])
        vol_contracting = vol_5d < vol_20d * 0.8
        vol_surge_today = vol_today > vol_20d * 1.3

        if vol_contracting and vol_surge_today:
            signal_count += 2
            signals.append(f"거래량 수축 후 급증 (오늘 {vol_today/vol_20d:.1f}배)")
        elif vol_contracting:
            signal_count += 1
            signals.append("거래량 수축 중 (눌림목 형성)")

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1])

        if 30 <= rsi <= 55:
            signal_count += 1
            signals.append(f"RSI 과매도 탈출 구간 ({rsi:.1f})")
        elif rsi < 30:
            signal_count += 1
            signals.append(f"RSI 극단 과매도 ({rsi:.1f}) — 반등 임박 가능")

        # 최소 2개 신호 이상
        recovery_pass = signal_count >= 2

        return {
            **panic_data,
            "rsi": round(rsi, 2),
            "above_sma5": above_sma5,
            "bos_detected": bos_detected,
            "vol_contracting": vol_contracting,
            "recovery_signals": signals,
            "recovery_signal_count": signal_count,
            "recovery_pass": recovery_pass,
        }

    except Exception as e:
        logger.error(f"{ticker} 회복 신호 실패: {e}")
        return None


# ── ④ Claude 뉴스 분석 — 급락 원인 판별 ──────────────────────

PANIC_ANALYSIS_PROMPT = """당신은 월가 시니어 리서치 애널리스트입니다.
아래 종목이 최근 급락했습니다. 뉴스 헤드라인을 분석해서 급락 원인을 판별하고 JSON만 반환하세요.

종목: {ticker} ({sector})
최근 30일 하락률: {drop_30d}%
52주 고점 대비: {from_52w_high}%
패닉 급락일: {panic_days}

최근 뉴스 헤드라인:
{headlines}

반환 형식 (JSON만, 설명 없이):
{{
  "drop_cause": "MACRO_EXTERNAL | COMPANY_SPECIFIC | MIXED | UNKNOWN",
  "cause_detail": "급락 원인 1줄 설명",
  "is_company_fault": true 또는 false,
  "company_fault_reason": "기업 자체 문제가 있다면 설명, 없으면 null",
  "recovery_outlook": "STRONG | MODERATE | WEAK | UNCERTAIN",
  "recovery_reason": "회복 전망 근거 2줄",
  "pass": true 또는 false
}}

pass = false 조건 (하나라도 해당하면):
  - drop_cause == "COMPANY_SPECIFIC" (기업 자체 문제: 실적 쇼크, 회계 부정, CEO 스캔들 등)
  - is_company_fault == true
  - recovery_outlook == "WEAK"
"""


def analyze_drop_cause(ticker: str, data: dict) -> dict:
    """Claude로 급락 원인이 외부 이벤트인지 기업 문제인지 분석"""

    # 뉴스 수집
    headlines = []
    if NEWS_API_KEY:
        try:
            params = {
                "q": ticker,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "apiKey": NEWS_API_KEY,
            }
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            articles = resp.json().get("articles", [])
            headlines = [
                f"- [{a['source']['name']}] {a['title']} ({a['publishedAt'][:10]})"
                for a in articles if a.get("title")
            ]
        except Exception:
            pass

    headlines_text = "\n".join(headlines) if headlines else "(뉴스 없음)"

    # Claude 분석
    if ANTHROPIC_API_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": PANIC_ANALYSIS_PROMPT.format(
                        ticker=ticker,
                        sector=data.get("sector", ""),
                        drop_30d=data.get("drop_30d_pct", 0),
                        from_52w_high=data.get("from_52w_high_pct", 0),
                        panic_days=json.dumps(data.get("panic_days", []), ensure_ascii=False),
                        headlines=headlines_text,
                    ),
                }],
            )
            content = message.content[0].text
            clean = content.strip().removeprefix("```json").removesuffix("```").strip()
            result = json.loads(clean)
            result["analysis_method"] = "CLAUDE"
            result["headline_count"] = len(headlines)
            return result
        except Exception as e:
            logger.error(f"{ticker} Claude 분석 실패: {e}")

    # 폴백: 뉴스 없으면 외부 이벤트로 가정하고 통과
    return {
        "drop_cause": "UNKNOWN",
        "cause_detail": "뉴스 분석 불가 — 외부 이벤트 가정",
        "is_company_fault": False,
        "company_fault_reason": None,
        "recovery_outlook": "UNCERTAIN",
        "recovery_reason": "데이터 부족으로 판단 불가",
        "pass": True,
        "analysis_method": "FALLBACK",
        "headline_count": len(headlines),
    }


# ── ⑤ 목표가 유효성 확인 ─────────────────────────────────────

def check_target_validity(data: dict) -> dict:
    """
    애널리스트 목표가가 여전히 유효한지 확인.
    급락 후에도 목표가가 현재가 대비 20%+ 높으면 회복 여지 있음.
    """
    current = data.get("current_price")
    target = data.get("target_mean_price")
    analyst_n = data.get("analyst_count") or 0

    if not current or not target or analyst_n < 2:
        return {
            **data,
            "target_gap_pct": None,
            "target_valid": True,  # 데이터 없으면 기본 통과
            "target_note": "목표가 데이터 부족",
        }

    gap = round((target - current) / current * 100, 2)

    # 급락 후 목표가 괴리 20%+ = 회복 여지 충분
    target_valid = gap >= 15

    note = f"목표가 ${target:.2f} | 현재가 대비 +{gap:.1f}% (n={analyst_n})"
    if not target_valid:
        note += " ← 목표가 괴리 부족, 회복 여지 제한적"

    return {
        **data,
        "target_gap_pct": gap,
        "target_valid": target_valid,
        "target_note": note,
    }


# ── 개별 종목 전체 파이프라인 ──────────────────────────────────

def scan_panic_ticker(ticker: str) -> Optional[dict]:
    """단일 종목 패닉 스캔 전체 실행"""

    # ① 품질 필터
    quality = check_quality(ticker)
    if not quality or not quality["quality_pass"]:
        return None

    # ② 패닉 급락 감지
    panic = detect_panic_drop(ticker, quality)
    if not panic:
        return None

    # ③ 회복 신호
    recovery = check_recovery_signal(ticker, panic)
    if not recovery or not recovery["recovery_pass"]:
        return None

    # ④ Claude 급락 원인 분석
    cause = analyze_drop_cause(ticker, recovery)

    # ⑤ 목표가 유효성
    final = check_target_validity({**recovery, **cause})

    # 최종 통과 판정
    final_pass = (
        cause.get("pass", True)
        and final.get("target_valid", True)
    )

    final["ticker"] = ticker
    final["final_pass"] = final_pass
    final["scan_type"] = "PANIC_RECOVERY"

    return final


# ── 전체 스캔 실행 ─────────────────────────────────────────────

def run_panic_scan(custom_tickers: list = None) -> list:
    """
    패닉 셀 저가 매수 스캔 전체 실행.
    custom_tickers: 특정 종목만 스캔할 때 (기본: S&P500 전체)
    """
    logger.info("=" * 50)
    logger.info("패닉 셀 저가 매수 스캔 시작")
    logger.info("=" * 50)

    tickers = custom_tickers or fetch_universe()
    logger.info(f"스캔 대상: {len(tickers)}종목")

    results = []
    passed = []

    # 품질+패닉+회복은 병렬로 (Claude 분석은 순차)
    quality_panic_recovery = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_scan_pre_claude, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                quality_panic_recovery.append(result)
                logger.info(
                    f"  🔍 {result['ticker']}: 급락 {result['drop_30d_pct']:.1f}% | "
                    f"52주 고점 대비 {result['from_52w_high_pct']:.1f}% | "
                    f"회복신호 {result['recovery_signal_count']}개"
                )
            if (i + 1) % 30 == 0:
                logger.info(f"  ... {i+1}/{len(tickers)} 스캔 완료")

    logger.info(f"품질+패닉+회복 필터 통과: {len(quality_panic_recovery)}종목 → Claude 분석 시작")

    # Claude 분석 + 목표가 (순차)
    for data in quality_panic_recovery:
        ticker = data["ticker"]
        cause = analyze_drop_cause(ticker, data)
        final = check_target_validity({**data, **cause})

        final_pass = cause.get("pass", True) and final.get("target_valid", True)
        final["ticker"] = ticker
        final["final_pass"] = final_pass
        final["scan_type"] = "PANIC_RECOVERY"

        results.append(final)

        status = "✅ 매수 후보" if final_pass else "❌ 제외"
        logger.info(
            f"  {status} {ticker}: {cause.get('drop_cause')} | "
            f"{cause.get('recovery_outlook')} | 목표가 괴리 {final.get('target_gap_pct', 'N/A')}%"
        )

        if final_pass:
            passed.append(final)

    # 결과 저장
    os.makedirs("data", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_scanned": len(tickers),
        "pre_filter_passed": len(quality_panic_recovery),
        "final_passed": len(passed),
        "candidates": passed,
        "all_results": results,
    }
    with open("data/panic_scan.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n패닉 스캔 완료 — 최종 매수 후보 {len(passed)}종목")
    for r in passed:
        logger.info(
            f"  ⭐ {r['ticker']}: 급락 {r['drop_30d_pct']:.1f}% | "
            f"목표가 괴리 +{r.get('target_gap_pct', 0):.1f}% | "
            f"{r.get('recovery_signals', [])}"
        )

    return passed


def _scan_pre_claude(ticker: str) -> Optional[dict]:
    """Claude 이전 단계 (품질+패닉+회복) 병렬 처리용"""
    quality = check_quality(ticker)
    if not quality or not quality["quality_pass"]:
        return None
    panic = detect_panic_drop(ticker, quality)
    if not panic:
        return None
    recovery = check_recovery_signal(ticker, panic)
    if not recovery or not recovery["recovery_pass"]:
        return None
    return recovery


# ── 텔레그램 리포트 포맷 ──────────────────────────────────────

def format_panic_report(passed: list) -> str:
    lines = []
    lines.append("🚨 <b>패닉 셀 저가 매수 후보</b>")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')} KST")
    lines.append("━" * 30)

    if not passed:
        lines.append("\n현재 조건 충족 종목 없음")
        lines.append("(패닉 급락 + 회복 신호 + 외부 이벤트 3조건 동시 충족 필요)")
        return "\n".join(lines)

    for i, r in enumerate(passed[:8], 1):
        lines.append(f"\n{i}. <b>{r['ticker']}</b> — {r.get('sector', '')}")
        lines.append(f"   급락: {r['drop_30d_pct']:.1f}% (30일) | 52주 고점 대비 {r['from_52w_high_pct']:.1f}%")
        lines.append(f"   원인: {r.get('cause_detail', 'N/A')}")
        lines.append(f"   회복전망: {r.get('recovery_outlook', 'N/A')} | 목표가 괴리 +{r.get('target_gap_pct', 0):.1f}%")
        signals = r.get("recovery_signals", [])
        if signals:
            lines.append(f"   신호: {' / '.join(signals[:2])}")
        worst = r.get("worst_panic", {})
        if worst:
            lines.append(f"   최대 패닉일: {worst.get('date')} {worst.get('drop_pct')}% (거래량 {worst.get('vol_ratio')}배)")

    lines.append("\n━" * 30)
    lines.append("⚠️ 패닉 스캔은 단기 이벤트 기반입니다. 반드시 직접 검토 후 매수하세요.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 특정 종목만 테스트: python panic_scanner.py SNDK AMKR MU
        tickers = [t.upper() for t in sys.argv[1:]]
        logger.info(f"지정 종목 스캔: {tickers}")
        passed = run_panic_scan(custom_tickers=tickers)
    else:
        passed = run_panic_scan()

    print(f"\n최종 매수 후보 {len(passed)}종목:")
    for r in passed:
        print(
            f"  {r['ticker']:6s} | 급락 {r['drop_30d_pct']:+.1f}% | "
            f"52주 고점 대비 {r['from_52w_high_pct']:+.1f}% | "
            f"목표가 +{r.get('target_gap_pct', 0):.1f}% | "
            f"{r.get('drop_cause')} | {r.get('recovery_outlook')}"
        )

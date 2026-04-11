"""
MODULE 7: 백테스팅 엔진 (backtester.py)
6개 전략가 기법:
  ① O'Neil — CAN SLIM
  ② Minervini — SEPA
  ③ Weinstein — Stage Analysis
  ④ Lynch — PEG 성장주
  ⑤ Greenblatt — Magic Formula
  ⑥ Graham — 안전마진
+ 과거 3년 시뮬레이션 (승률, 평균수익률, MDD, 샤프)
통과: 6개 중 2개+ 통과, 승률 55%+, 평균수익률 8%+
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 공통 데이터 로더
# ═══════════════════════════════════════════════════════════════

def load_ticker_data(ticker: str) -> dict:
    """종목의 가격 히스토리 + 재무 데이터 한번에 로드"""
    t = yf.Ticker(ticker)
    info = t.info or {}

    hist_3y = t.history(period="3y")
    hist_5y = t.history(period="5y")

    # 분기 재무
    try:
        quarterly_earnings = t.quarterly_earnings
    except Exception:
        quarterly_earnings = None

    try:
        quarterly_financials = t.quarterly_financials
    except Exception:
        quarterly_financials = None

    try:
        balance_sheet = t.balance_sheet
    except Exception:
        balance_sheet = None

    return {
        "info": info,
        "hist_3y": hist_3y,
        "hist_5y": hist_5y,
        "quarterly_earnings": quarterly_earnings,
        "quarterly_financials": quarterly_financials,
        "balance_sheet": balance_sheet,
    }


# ═══════════════════════════════════════════════════════════════
# ① O'Neil — CAN SLIM
# ═══════════════════════════════════════════════════════════════

def check_canslim(ticker: str, data: dict) -> dict:
    """
    6항목 중 4개 이상 통과:
    1. 최근 분기 EPS 성장률 25%+
    2. 최근 분기 매출 성장률 25%+
    3. 연간 EPS 3년 연속 성장
    4. 52주 신고가 -25% 이내
    5. 기관 매수 증가 추세
    6. 시장 전체 상승 국면
    """
    info = data["info"]
    hist = data["hist_3y"]
    checks = {}
    passed_count = 0

    # 1. 분기 EPS 성장률
    qe = data.get("quarterly_earnings")
    if qe is not None and len(qe) >= 5:
        try:
            recent_eps = float(qe.iloc[0].get("Earnings", 0))
            year_ago_eps = float(qe.iloc[4].get("Earnings", 0))
            if year_ago_eps > 0:
                eps_growth = (recent_eps - year_ago_eps) / year_ago_eps * 100
                checks["quarterly_eps_growth"] = round(eps_growth, 1)
                if eps_growth >= 25:
                    passed_count += 1
            else:
                checks["quarterly_eps_growth"] = None
        except Exception:
            checks["quarterly_eps_growth"] = None
    else:
        # yfinance 대안: earningsQuarterlyGrowth
        eqg = info.get("earningsQuarterlyGrowth")
        if eqg is not None:
            checks["quarterly_eps_growth"] = round(eqg * 100, 1)
            if eqg >= 0.25:
                passed_count += 1
        else:
            checks["quarterly_eps_growth"] = None

    # 2. 분기 매출 성장률
    rg = info.get("revenueGrowth")
    if rg is not None:
        checks["revenue_growth"] = round(rg * 100, 1)
        if rg >= 0.25:
            passed_count += 1
    else:
        checks["revenue_growth"] = None

    # 3. 연간 EPS 3년 연속 성장
    try:
        earnings = yf.Ticker(ticker).earnings
        if earnings is not None and len(earnings) >= 3:
            eps_vals = earnings.iloc[-3:]["Earnings"].tolist() if "Earnings" in earnings.columns else []
            if len(eps_vals) >= 3 and all(eps_vals[i] < eps_vals[i + 1] for i in range(len(eps_vals) - 1)):
                checks["annual_eps_3yr_growth"] = True
                passed_count += 1
            else:
                checks["annual_eps_3yr_growth"] = False
        else:
            checks["annual_eps_3yr_growth"] = None
    except Exception:
        checks["annual_eps_3yr_growth"] = None

    # 4. 52주 신고가 -25% 이내
    if not hist.empty:
        high_52w = float(hist["Close"].iloc[-252:].max()) if len(hist) >= 252 else float(hist["Close"].max())
        current = float(hist["Close"].iloc[-1])
        from_high = (current - high_52w) / high_52w * 100
        checks["from_52w_high"] = round(from_high, 1)
        if from_high >= -25:
            passed_count += 1
    else:
        checks["from_52w_high"] = None

    # 5. 기관 매수 증가
    inst_pct = info.get("heldPercentInstitutions")
    if inst_pct and inst_pct > 0.5:
        checks["institutional_strong"] = True
        passed_count += 1
    else:
        checks["institutional_strong"] = inst_pct is not None and inst_pct > 0.3

    # 6. 시장 상승 국면 (S&P500 200일선 위)
    try:
        sp = yf.Ticker("^GSPC").history(period="1y")
        sp_close = sp["Close"]
        sp_sma200 = sp_close.rolling(200).mean()
        if float(sp_close.iloc[-1]) > float(sp_sma200.iloc[-1]):
            checks["market_uptrend"] = True
            passed_count += 1
        else:
            checks["market_uptrend"] = False
    except Exception:
        checks["market_uptrend"] = None

    return {
        "strategy": "CANSLIM",
        "checks": checks,
        "passed_items": passed_count,
        "required": 4,
        "pass": passed_count >= 4,
    }


# ═══════════════════════════════════════════════════════════════
# ② Minervini — SEPA
# ═══════════════════════════════════════════════════════════════

def check_minervini(ticker: str, data: dict) -> dict:
    """
    6항목 중 5개 이상:
    1. 현재가 > 150일선 > 200일선
    2. 200일선 1개월+ 우상향
    3. 52주 저점 대비 +30%+
    4. 52주 고점 대비 -25% 이내
    5. RS 상위 70%
    6. VCP — 변동성 수축 패턴
    """
    hist = data["hist_3y"]
    if len(hist) < 200:
        return {"strategy": "MINERVINI", "pass": False, "fail_reason": "데이터 부족"}

    close = hist["Close"]
    current = float(close.iloc[-1])
    checks = {}
    passed_count = 0

    sma150 = float(close.rolling(150).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    # 1. 정배열
    checks["price_above_150_above_200"] = current > sma150 > sma200
    if checks["price_above_150_above_200"]:
        passed_count += 1

    # 2. 200일선 우상향 (1개월 전 대비)
    sma200_series = close.rolling(200).mean()
    sma200_now = float(sma200_series.iloc[-1])
    sma200_1m = float(sma200_series.iloc[-22]) if len(sma200_series) > 22 else sma200_now
    checks["sma200_uptrend"] = sma200_now > sma200_1m
    if checks["sma200_uptrend"]:
        passed_count += 1

    # 3. 52주 저점 대비 +30%+
    low_52w = float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min())
    from_low = (current - low_52w) / low_52w * 100
    checks["from_52w_low"] = round(from_low, 1)
    if from_low >= 30:
        passed_count += 1

    # 4. 52주 고점 대비 -25% 이내
    high_52w = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    from_high = (current - high_52w) / high_52w * 100
    checks["from_52w_high"] = round(from_high, 1)
    if from_high >= -25:
        passed_count += 1

    # 5. RS 상위 70% (S&P500 대비 상대강도)
    try:
        sp = yf.Ticker("^GSPC").history(period="1y")
        sp_ret = (float(sp["Close"].iloc[-1]) / float(sp["Close"].iloc[0]) - 1)
        stock_ret = (current / float(close.iloc[-252]) - 1) if len(close) >= 252 else 0
        rs = stock_ret - sp_ret
        checks["relative_strength"] = round(rs * 100, 1)
        if rs > -0.1:  # 시장 대비 -10% 이내면 상위 70% 근사
            passed_count += 1
    except Exception:
        checks["relative_strength"] = None

    # 6. VCP — 최근 3개 고저 스윙 폭 축소
    try:
        recent_60 = close.iloc[-60:]
        thirds = np.array_split(recent_60, 3)
        ranges = [(float(s.max()) - float(s.min())) / float(s.mean()) * 100 for s in thirds]
        checks["vcp_ranges"] = [round(r, 1) for r in ranges]
        if len(ranges) == 3 and ranges[0] > ranges[1] > ranges[2]:
            checks["vcp_contracting"] = True
            passed_count += 1
        else:
            checks["vcp_contracting"] = False
    except Exception:
        checks["vcp_contracting"] = None

    return {
        "strategy": "MINERVINI",
        "checks": checks,
        "passed_items": passed_count,
        "required": 5,
        "pass": passed_count >= 5,
    }


# ═══════════════════════════════════════════════════════════════
# ③ Weinstein — Stage Analysis
# ═══════════════════════════════════════════════════════════════

def check_weinstein(ticker: str, data: dict) -> dict:
    """
    30주선(150일선) 기반 스테이지 판단.
    Stage 2 진입 또는 초입만 통과.
    """
    hist = data["hist_3y"]
    if len(hist) < 200:
        return {"strategy": "WEINSTEIN", "pass": False, "fail_reason": "데이터 부족"}

    close = hist["Close"]
    sma150 = close.rolling(150).mean()

    current = float(close.iloc[-1])
    sma_now = float(sma150.iloc[-1])
    sma_1m = float(sma150.iloc[-22]) if len(sma150) > 22 else sma_now
    sma_2m = float(sma150.iloc[-44]) if len(sma150) > 44 else sma_1m

    # 30주선 기울기
    slope_recent = sma_now - sma_1m
    slope_prior = sma_1m - sma_2m

    if current > sma_now and slope_recent > 0:
        stage = 2
        detail = "Stage 2 — 상승 (현재가 30주선 위, 우상향)"
    elif current > sma_now and abs(slope_recent) < sma_now * 0.005:
        stage = 2
        detail = "Stage 2 초입 — 30주선 평평→우상향 전환"
    elif abs(slope_recent) < sma_now * 0.005 and abs(current - sma_now) < sma_now * 0.03:
        stage = 1
        detail = "Stage 1 — 횡보 (30주선 평평)"
    elif current > sma_now and slope_recent < 0:
        stage = 3
        detail = "Stage 3 — 천장 (30주선 꺾임)"
    else:
        stage = 4
        detail = "Stage 4 — 하락 (30주선 아래)"

    return {
        "strategy": "WEINSTEIN",
        "stage": stage,
        "detail": detail,
        "price_vs_sma150": round((current / sma_now - 1) * 100, 1),
        "sma150_slope_1m": round(slope_recent, 2),
        "pass": stage == 2,
    }


# ═══════════════════════════════════════════════════════════════
# ④ Lynch — PEG 성장주
# ═══════════════════════════════════════════════════════════════

def check_lynch(ticker: str, data: dict) -> dict:
    """
    전항목 충족:
    1. PEG ≤ 1.0
    2. EPS 성장률 15%+
    3. D/E < 1.0
    4. 성장 스토리 (여기서는 매출 성장 양수로 근사)
    """
    info = data["info"]
    checks = {}
    passed_count = 0

    peg = info.get("pegRatio")
    checks["peg_ratio"] = peg
    if peg is not None and 0 < peg <= 1.0:
        passed_count += 1

    eqg = info.get("earningsQuarterlyGrowth")
    if eqg is not None:
        checks["eps_growth_pct"] = round(eqg * 100, 1)
        if eqg >= 0.15:
            passed_count += 1
    else:
        checks["eps_growth_pct"] = None

    de = info.get("debtToEquity")
    if de is not None:
        checks["debt_to_equity"] = round(de / 100, 2) if de > 2 else round(de, 2)  # yfinance 때때로 %로 반환
        actual_de = de / 100 if de > 2 else de
        if actual_de < 1.0:
            passed_count += 1
    else:
        checks["debt_to_equity"] = None

    rg = info.get("revenueGrowth")
    checks["revenue_growth"] = round(rg * 100, 1) if rg else None
    if rg and rg > 0:
        passed_count += 1

    return {
        "strategy": "LYNCH",
        "checks": checks,
        "passed_items": passed_count,
        "required": 4,
        "pass": passed_count >= 4,
    }


# ═══════════════════════════════════════════════════════════════
# ⑤ Greenblatt — Magic Formula
# ═══════════════════════════════════════════════════════════════

def calc_greenblatt_metrics(ticker: str, data: dict) -> Optional[dict]:
    """ROC + Earnings Yield 계산"""
    info = data["info"]

    ebit = info.get("ebitda")  # EBITDA 근사 (EBIT 직접 제공 안 됨)
    ev = info.get("enterpriseValue")
    market_cap = info.get("marketCap")

    if not ebit or not ev or ev <= 0:
        return None

    earnings_yield = ebit / ev

    # ROC 근사: EBIT / (총자산 - 유동부채) — 간소화
    total_assets = info.get("totalAssets")
    current_liabilities = info.get("totalCurrentLiabilities")

    if total_assets and current_liabilities:
        capital = total_assets - current_liabilities
        roc = ebit / capital if capital > 0 else 0
    else:
        # 대안: returnOnAssets
        roc = info.get("returnOnAssets") or 0

    return {
        "earnings_yield": round(earnings_yield, 4),
        "roc": round(roc, 4),
    }


def check_greenblatt_batch(tickers: list, all_data: dict) -> dict:
    """
    전체 종목 대상 Magic Formula 순위 산출.
    ROC 상위 50% + EY 상위 50% → 합산 순위 상위 30%
    """
    metrics = []
    for ticker in tickers:
        data = all_data.get(ticker)
        if not data:
            continue
        m = calc_greenblatt_metrics(ticker, data)
        if m:
            m["ticker"] = ticker
            metrics.append(m)

    if not metrics:
        return {}

    df = pd.DataFrame(metrics)
    df["ey_rank"] = df["earnings_yield"].rank(ascending=False)
    df["roc_rank"] = df["roc"].rank(ascending=False)
    df["combined_rank"] = df["ey_rank"] + df["roc_rank"]
    df["combined_pct"] = df["combined_rank"].rank(pct=True)

    results = {}
    for _, row in df.iterrows():
        t = row["ticker"]
        results[t] = {
            "strategy": "GREENBLATT",
            "earnings_yield": row["earnings_yield"],
            "roc": row["roc"],
            "ey_rank": int(row["ey_rank"]),
            "roc_rank": int(row["roc_rank"]),
            "combined_rank": int(row["combined_rank"]),
            "combined_pct": round(row["combined_pct"], 2),
            "pass": row["combined_pct"] <= 0.30,
        }
    return results


# ═══════════════════════════════════════════════════════════════
# ⑥ Graham — 안전마진
# ═══════════════════════════════════════════════════════════════

def check_graham(ticker: str, data: dict) -> dict:
    """
    5항목 중 4개 이상:
    1. PBR ≤ 1.5
    2. PER ≤ 15
    3. 유동비율 ≥ 2.0
    4. 부채비율 ≤ 50%
    5. 최근 5년 EPS 흑자
    """
    info = data["info"]
    checks = {}
    passed_count = 0

    pb = info.get("priceToBook")
    checks["pbr"] = pb
    if pb is not None and pb <= 1.5:
        passed_count += 1

    pe = info.get("trailingPE")
    checks["per"] = pe
    if pe is not None and pe <= 15:
        passed_count += 1

    cr = info.get("currentRatio")
    checks["current_ratio"] = cr
    if cr is not None and cr >= 2.0:
        passed_count += 1

    de = info.get("debtToEquity")
    if de is not None:
        actual_de = de / 100 if de > 5 else de  # yfinance 보정
        checks["debt_ratio_pct"] = round(actual_de * 100, 1) if actual_de <= 1 else round(de, 1)
        if actual_de <= 0.5 or de <= 50:
            passed_count += 1
    else:
        checks["debt_ratio_pct"] = None

    # 5년 EPS 흑자
    eps = info.get("trailingEps")
    checks["trailing_eps_positive"] = eps is not None and eps > 0
    if checks["trailing_eps_positive"]:
        passed_count += 1  # 5년 연속은 데이터 한계로 최근 흑자로 근사

    return {
        "strategy": "GRAHAM",
        "checks": checks,
        "passed_items": passed_count,
        "required": 4,
        "pass": passed_count >= 4,
    }


# ═══════════════════════════════════════════════════════════════
# 백테스팅 시뮬레이션
# ═══════════════════════════════════════════════════════════════

def simulate_backtest(ticker: str, data: dict) -> dict:
    """
    과거 3년 데이터 기반 간이 백테스트.
    눌림목 진입 시점 포착 → 20/60/120일 수익률 계산.
    """
    hist = data["hist_3y"]
    if len(hist) < 300:
        return {"ticker": ticker, "backtest_available": False}

    close = hist["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    entries = []

    for i in range(200, len(close) - 120):
        c = float(close.iloc[i])
        s20 = float(sma20.iloc[i])
        s50 = float(sma50.iloc[i])
        s200 = float(sma200.iloc[i])

        # 정배열 + 눌림목 조건 간소화
        if s20 > s50 > s200:
            high_60 = float(close.iloc[i - 60:i].max())
            pullback = (c - high_60) / high_60 * 100
            if -15 <= pullback <= -5:
                # 진입
                p20 = float(close.iloc[i + 20]) if i + 20 < len(close) else None
                p60 = float(close.iloc[i + 60]) if i + 60 < len(close) else None
                p120 = float(close.iloc[i + 120]) if i + 120 < len(close) else None

                entry = {"entry_price": c, "entry_idx": i}
                if p20:
                    entry["ret_20d"] = round((p20 / c - 1) * 100, 2)
                if p60:
                    entry["ret_60d"] = round((p60 / c - 1) * 100, 2)
                if p120:
                    entry["ret_120d"] = round((p120 / c - 1) * 100, 2)

                entries.append(entry)

    if not entries:
        return {"ticker": ticker, "backtest_available": True, "signal_count": 0}

    # 60일 기준 통계
    rets = [e["ret_60d"] for e in entries if "ret_60d" in e]
    if not rets:
        return {"ticker": ticker, "backtest_available": True, "signal_count": len(entries), "no_60d_data": True}

    wins = [r for r in rets if r > 0]
    win_rate = len(wins) / len(rets) * 100
    avg_ret = np.mean(rets)
    max_dd = min(rets)

    # 샤프 근사 (무위험 0 가정)
    sharpe = (np.mean(rets) / np.std(rets)) if np.std(rets) > 0 else 0

    # 최종 점수 = 승률×0.4 + 평균수익률×0.4 + (1-MDD)×0.2
    mdd_normalized = abs(max_dd) / 100  # 0~1
    final_score = win_rate * 0.4 + avg_ret * 0.4 + (1 - mdd_normalized) * 0.2 * 100

    return {
        "ticker": ticker,
        "backtest_available": True,
        "signal_count": len(entries),
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_ret, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "final_score": round(final_score, 1),
    }


# ═══════════════════════════════════════════════════════════════
# 통합 실행
# ═══════════════════════════════════════════════════════════════

def run_backtest(tickers: list) -> list:
    """전체 백테스팅 파이프라인 실행"""
    logger.info(f"백테스팅 시작 — {len(tickers)}종목, 6개 전략")

    # 데이터 일괄 로드
    all_data = {}
    for ticker in tickers:
        try:
            all_data[ticker] = load_ticker_data(ticker)
            logger.info(f"  📥 {ticker} 데이터 로드 완료")
        except Exception as e:
            logger.error(f"  ❌ {ticker} 데이터 로드 실패: {e}")

    # Greenblatt는 배치 처리
    greenblatt_results = check_greenblatt_batch(tickers, all_data)

    results = []
    for ticker in tickers:
        if ticker not in all_data:
            continue

        data = all_data[ticker]
        logger.info(f"  🔍 {ticker} 전략 검증 중...")

        strategies = {}

        # ① CAN SLIM
        try:
            strategies["canslim"] = check_canslim(ticker, data)
        except Exception as e:
            strategies["canslim"] = {"strategy": "CANSLIM", "pass": False, "error": str(e)}

        # ② Minervini
        try:
            strategies["minervini"] = check_minervini(ticker, data)
        except Exception as e:
            strategies["minervini"] = {"strategy": "MINERVINI", "pass": False, "error": str(e)}

        # ③ Weinstein
        try:
            strategies["weinstein"] = check_weinstein(ticker, data)
        except Exception as e:
            strategies["weinstein"] = {"strategy": "WEINSTEIN", "pass": False, "error": str(e)}

        # ④ Lynch
        try:
            strategies["lynch"] = check_lynch(ticker, data)
        except Exception as e:
            strategies["lynch"] = {"strategy": "LYNCH", "pass": False, "error": str(e)}

        # ⑤ Greenblatt
        strategies["greenblatt"] = greenblatt_results.get(ticker, {"strategy": "GREENBLATT", "pass": False, "no_data": True})

        # ⑥ Graham
        try:
            strategies["graham"] = check_graham(ticker, data)
        except Exception as e:
            strategies["graham"] = {"strategy": "GRAHAM", "pass": False, "error": str(e)}

        passed_strategies = [k for k, v in strategies.items() if v.get("pass")]

        # 백테스트 시뮬레이션
        try:
            backtest = simulate_backtest(ticker, data)
        except Exception as e:
            backtest = {"ticker": ticker, "backtest_available": False, "error": str(e)}

        # 최종 통과 판정
        bt_win = backtest.get("win_rate", 0)
        bt_avg = backtest.get("avg_return", 0)
        final_pass = (
            len(passed_strategies) >= 2
            and bt_win >= 55
            and bt_avg >= 8
        )

        result = {
            "ticker": ticker,
            "strategies": strategies,
            "passed_strategies": passed_strategies,
            "passed_strategy_count": len(passed_strategies),
            "backtest": backtest,
            "final_pass": final_pass,
        }
        results.append(result)

        status = "✅ 최종 통과" if final_pass else "❌ 탈락"
        logger.info(
            f"  {status} {ticker}: 전략 {len(passed_strategies)}/6 통과 "
            f"({', '.join(passed_strategies) or '없음'}) | "
            f"승률 {bt_win}% | 평균수익 {bt_avg}%"
        )

    passed = [r for r in results if r["final_pass"]]

    output = {
        "timestamp": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "passed_count": len(passed),
        "passed": passed,
        "failed": [r for r in results if not r["final_pass"]],
    }

    os.makedirs("data", exist_ok=True)
    with open("data/backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"백테스팅 완료 — {len(passed)}/{len(results)} 최종 통과")
    return passed


if __name__ == "__main__":
    ws_path = "data/wallstreet_analysis.json"
    if os.path.exists(ws_path):
        with open(ws_path, "r") as f:
            ws_data = json.load(f)
        tickers = [r["ticker"] for r in ws_data.get("passed", [])]
        logger.info(f"월가 검증 통과 종목에서 {len(tickers)}종목 로드")
    else:
        tickers = ["AAPL", "MSFT", "NVDA"]
        logger.info("월가 검증 결과 없음 — 테스트 종목 사용")

    passed = run_backtest(tickers)
    print(f"\n백테스트 최종 통과 {len(passed)}종목:")
    for r in passed:
        bt = r["backtest"]
        print(
            f"  {r['ticker']:6s} | 전략 {r['passed_strategy_count']}/6 "
            f"({', '.join(r['passed_strategies'])}) | "
            f"승률 {bt.get('win_rate', 0)}% | 평균 +{bt.get('avg_return', 0)}%"
        )

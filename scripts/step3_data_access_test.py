"""3단계: 데이터 접근 테스트.

pykrx(KRX)로 KOSDAQ 가격·투자자별 매매가 받아지는지, 상장폐지 종목과
초기 연도까지 나오는지 확인한다. 결과는 results/step3_data_access.md 에 남긴다.
KRX 계정은 환경변수 KRX_ID, KRX_PW 로만 읽는다(코드에 적지 않는다).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step3_data_access.md"
SAMSUNG = "005930"
PAUSE = 1.0  # KRX 요청 사이 간격(초)

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def run(name: str, fn):
    log(f"\n### {name}")
    t0 = time.time()
    try:
        res = fn()
        if res is None or len(res) == 0:  # pykrx 는 오류를 삼키고 빈 표를 돌려준다
            log(f"- 결과: **실패** (빈 결과, {time.time() - t0:.1f}초)")
            return None
        log(f"- 결과: **성공** ({time.time() - t0:.1f}초)")
        return res
    except Exception as e:  # 실패해도 다음 테스트로 넘어간다
        log(f"- 결과: **실패** ({type(e).__name__}: {str(e)[:300]})")
        log("```\n" + traceback.format_exc(limit=3)[-1200:] + "\n```")
        return None
    finally:
        time.sleep(PAUSE)


def show(df: pd.DataFrame | None, n: int = 5) -> None:
    if df is None or len(df) == 0:
        log("- 데이터 없음(빈 표)")
        return
    log(f"- 크기: {df.shape[0]}행 × {df.shape[1]}열, 열: {list(df.columns)[:12]}")
    log("```\n" + df.head(n).to_string()[:1500] + "\n```")


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from krx_env import load_krx_env

    ok = load_krx_env()  # pykrx import 전에 계정을 환경변수로 준비
    log(f"# 3단계 데이터 접근 테스트 ({datetime.now():%Y-%m-%d %H:%M})")
    log(f"- KRX 계정(KRX_ID·KRX_PW): {'설정됨' if ok else '없음'}")

    from pykrx import stock
    import pykrx

    log(f"- pykrx 버전: {getattr(pykrx, '__version__', '알 수 없음')}")

    # 3-1. 삼성전자 2005-01 일별 가격
    def t_ohlcv_adj():
        df = stock.get_market_ohlcv("20050103", "20050131", SAMSUNG, adjusted=True)
        show(df)
        return df

    def t_ohlcv_raw():
        df = stock.get_market_ohlcv("20050103", "20050131", SAMSUNG, adjusted=False)
        show(df)
        return df

    run("3-1a. 삼성전자 2005-01 OHLCV (수정주가)", t_ohlcv_adj)
    run("3-1b. 삼성전자 2005-01 OHLCV (원주가, KRX 직접)", t_ohlcv_raw)

    # 3-2. 투자자별 매수·매도 거래대금
    def t_inv(on: str):
        def f():
            df = stock.get_market_trading_value_by_date(
                "20050103", "20050131", SAMSUNG, detail=True, on=on
            )
            show(df)
            return df
        return f

    run("3-2a. 삼성전자 2005-01 투자자별 매수 거래대금(세부)", t_inv("매수"))
    run("3-2b. 삼성전자 2005-01 투자자별 매도 거래대금(세부)", t_inv("매도"))

    # 3-5. 투자자 데이터 시작 연도(연도별 1월 한 달씩)
    cover = []
    for y in [1999, 2000, 2002, 2005, 2010, 2015, 2020, 2025]:
        def f(y=y):
            df = stock.get_market_trading_value_by_date(
                f"{y}0101", f"{y}0131", SAMSUNG, detail=False, on="순매수"
            )
            n = 0 if df is None else len(df)
            nz = 0 if not n else int((df.abs().sum(axis=1) > 0).sum())
            log(f"- {y}년 1월: {n}일, 값이 0이 아닌 날 {nz}일")
            cover.append((y, n, nz))
            return df
        run(f"3-5. 투자자 데이터 커버리지 {y}", f)

    # KOSDAQ 종목 목록과 상장폐지 후보
    def t_kosdaq_list():
        a = set(stock.get_market_ticker_list("20100104", market="KOSDAQ"))
        time.sleep(PAUSE)
        b = set(stock.get_market_ticker_list("20250102", market="KOSDAQ"))
        gone = sorted(a - b)
        log(f"- KOSDAQ 2010-01-04: {len(a)}종목, 2025-01-02: {len(b)}종목, 2010에 있다가 사라진 종목: {len(gone)}개")
        log(f"- 사라진 종목 예시: {gone[:10]}")
        return gone

    gone = run("KOSDAQ 종목 목록(2010 vs 2025)", t_kosdaq_list) or []

    # 3-3. 상장폐지(또는 이전상장) 종목 1개
    if gone:
        tk = gone[0]

        def t_delisted_price():
            df = stock.get_market_ohlcv("20100104", "20250102", tk, adjusted=False)
            log(f"- 종목 {tk}: 마지막 거래일 {df.index.max() if len(df) else '없음'}")
            log("```\n" + df.tail(8).to_string()[:1500] + "\n```")
            zero = int((df["거래량"] == 0).sum()) if "거래량" in df else -1
            log(f"- 거래량 0인 날: {zero}일 (3-4. 거래정지일 모양 확인용)")
            if zero > 0:
                log("```\n" + df[df["거래량"] == 0].head(5).to_string() + "\n```")
            return df

        def t_delisted_inv():
            df = stock.get_market_trading_value_by_date(
                "20100104", "20250102", tk, detail=False, on="순매수"
            )
            log(f"- 종목 {tk}: 투자자 데이터 {len(df)}일, 마지막 {df.index.max() if len(df) else '없음'}")
            log("```\n" + df.tail(5).to_string()[:1200] + "\n```")
            return df

        run(f"3-3a. 사라진 KOSDAQ 종목 {tk} 가격", t_delisted_price)
        run(f"3-3b. 사라진 KOSDAQ 종목 {tk} 투자자별 순매수", t_delisted_inv)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n저장: {OUT}")


if __name__ == "__main__":
    sys.exit(main())

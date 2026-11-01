"""파일럿 200종목의 일별 투자자별 순매수 거래대금(원, detail=True 12개 열), 2004-09-01~2012-12-31.
→ PAPERS/data/daily/investor/{ticker}.parquet. 검증을 통과한 파일은 건너뛴다(재개 가능).

주의: KRX는 같은 계정으로 새 로그인이 들어오면 기존 세션을 끊는다(pykrx의 skipDup=Y). pykrx는 세션 유효성을
시간으로만 판단해서, 끊긴 세션으로 요청하면 오류를 삼키고 빈 표를 돌려준다. 그래서
(1) 받은 표가 그 종목의 거래일(거래량>0)을 모두 덮는지 확인하고, (2) 아니면 재로그인 후 다시 받는다.
내 프로세스끼리 서로 끊지 않도록 1개 프로세스로 돌린다(다른 에이전트도 같은 계정을 쓴다).
사용: py -3 a2_collect_investor.py"""
import os
import pathlib
import sys
import time

import pandas as pd

import krx
from pykrx.website.comm.auth import get_auth_session

DATA = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily"
OUT = DATA / "investor"
OUT.mkdir(parents=True, exist_ok=True)
tickers = (DATA / "universe_pilot_20041230.txt").read_text(encoding="utf-8").split()


def valid(d, tk):
    """거래량>0인 날이 투자자 표에 모두 있어야 한다(거래정지일은 KRX가 행을 주지 않는다)."""
    p = pd.read_parquet(DATA / "ohlcv" / f"{tk}.parquet").loc["2004-09-01":"2012-12-31"]
    need = p.index[p.volume > 0]
    return len(need.difference(d.index)) == 0 and len(d.columns) == 12


def relogin():
    ok = get_auth_session().refresh(os.environ["KRX_ID"], os.environ["KRX_PW"])  # 계정은 출력하지 않는다
    print("재로그인", "성공" if ok else "실패", time.strftime("%H:%M:%S"), flush=True)


fails = 0
for i, tk in enumerate(tickers, 1):
    f = OUT / f"{tk}.parquet"
    if f.exists() and valid(pd.read_parquet(f), tk):
        continue
    for attempt in range(4):
        d = krx.stock.get_market_trading_value_by_date("20040901", "20121231", tk, detail=True)
        if valid(d, tk):
            d.rename_axis("date").astype("int64").to_parquet(f)
            fails = 0
            break
        print(tk, "검증 실패", attempt, "행", len(d), flush=True)
        time.sleep(20 * (attempt + 1))
        relogin()
    else:
        fails += 1
        if fails >= 5:
            sys.exit("연속 5종목 실패, 중단 (다시 실행하면 이어 받는다)")
    time.sleep(0.5)
    print(f"{i}/{len(tickers)} {tk}", flush=True)

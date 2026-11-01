"""파일럿 표본: 2004-12-30 KOSPI 시가총액 상위 200개 보통주의 일별 수정 OHLCV(2004-06~2012-12).
KRX는 2014-05 이전 KOSPI200 구성 종목을 주지 않아 시총 상위 200으로 대신한다(KOSPI200 근사).
시작 시점으로 종목을 고정해 '살아남은 종목만 고르는' 편향을 피한다. 종목별 parquet, 이미 받은 종목은 건너뛴다."""
import pathlib
import sys
import time

import krx

OUT = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily" / "ohlcv"
OUT.mkdir(parents=True, exist_ok=True)

cap = krx.stock.get_market_cap("20041230", market="KOSPI")
# ponytail: 보통주 = 코드 끝자리 0(우선주는 5·7·9 등). 리츠·SPAC 같은 예외는 5-1단계 종목 마스터에서 거른다.
tickers = cap[cap.index.str.endswith("0")].sort_values("시가총액", ascending=False).index[:200].tolist()
assert len(tickers) == 200
(OUT.parent / "universe_pilot_20041230.txt").write_text("\n".join(tickers), encoding="utf-8")

# KRX 응답이 느려(1년치 약 3초, pykrx가 2년 단위로 쪼개며 1초씩 쉰다) 종목당 약 30초 → 프로세스 n개로 나눠 받는다.
# 사용: py -3 01_collect.py k n  (k = 0..n-1)
k, n = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (0, 1)
for i, tk in enumerate(tickers[k::n], 1):
    f = OUT / f"{tk}.parquet"
    if f.exists():
        continue
    for attempt in range(3):
        try:
            krx.ohlcv(tk, "20040601", "20121231").to_parquet(f)
            break
        except Exception as e:
            print(tk, "재시도", attempt, e)
            time.sleep(5)
    time.sleep(0.5)
    print(f"[{k}] {i}/{len(tickers[k::n])} {tk}", flush=True)

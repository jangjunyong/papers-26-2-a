"""p03: 패널 v3 = v2 와 같은 로직(p02.build_panel)을 2000~2026 KOSPI 로 넓힌 것 (사전고정_v1 1·2절).

v2 와 다른 점 두 가지 (근거 results/q03_investor_2000_2009_memo.md):
  - 4그룹 순매수가 모두 0 인 행을 결측으로 지우지 않는다(개인만 거래한 날로 본다) → ZERO_INV_MIN_VOL = inf
  - inv_sum_closed 를 외국인 정의 변경(2026-05-26) 전 전 구간 1 로 둔다. 수급 변수는 개인·기관·외국인 3그룹만 쓰고,
    2012-05-02 이전 합계 불일치는 기타법인 누락·미확인 잔차라서다. 원래 값은 inv_sum_closed_raw 로 남긴다.
입력: club_db_ext/{price,investor}_kospi_2000_2009.parquet + club_db/{price,investor,trading}[_2010_2014].parquet
출력: PAPERS/data/daily/club_db/panel_v3.parquet, panel_v3_removed_rows.parquet
실행: py -3 p03_build_panel_v3.py   (v2 보다 약 1.8배 행, 메모리 약 7GB)
"""
import numpy as np
import pandas as pd

import p02_build_panel_v2 as p02

EXT = f"{p02.D}/daily/club_db_ext"
OUT = f"{p02.CB}/panel_v3.parquet"
OUT_REMOVED = f"{p02.CB}/panel_v3_removed_rows.parquet"


def main():
    p02.ZERO_INV_MIN_VOL = np.inf
    price = pd.concat([pd.read_parquet(f"{EXT}/price_kospi_2000_2009.parquet"), p02.load("price")], ignore_index=True)
    inv = pd.concat([pd.read_parquet(f"{EXT}/investor_kospi_2000_2009.parquet"), p02.load("investor")], ignore_index=True)
    trd = p02.load("trading")[["code", "date", "trading_value"]]
    for d in (price, inv):
        d["date"] = pd.to_datetime(d["date"])
        assert not d.duplicated(["code", "date"]).any()
    p02.t_log("입력", len(price), len(inv), len(trd))
    df, removed = p02.build_panel(price, inv, trd, verbose=True)
    df["inv_sum_closed_raw"] = df.inv_sum_closed
    df["inv_sum_closed"] = (df.foreign_def_changed == 0).astype("int8")
    df.to_parquet(OUT, index=False)
    removed.to_parquet(OUT_REMOVED, index=False)
    p02.t_log("panel_v3", df.shape, df.date.min().date(), "~", df.date.max().date(), "종목", df.ticker.nunique(), "→", OUT)
    # v2 와 겹치는 2010~ 구간은 투자자 0행 처리만 다르다 → 나머지 열이 같은지 확인
    v2 = pd.read_parquet(p02.OUT, columns=["ticker", "date", "close", "adj_close", "fwd_ret5", "f_tradable", "exclude_flag"])
    m = v2.merge(df[["ticker", "date", "close", "adj_close", "fwd_ret5", "f_tradable", "exclude_flag"]], on=["ticker", "date"], suffixes=("", "_3"))
    for c in ["close", "adj_close", "fwd_ret5", "f_tradable", "exclude_flag"]:
        a, b = m[c], m[c + "_3"]
        print(f"v2 대조 {c}: 불일치 {int(((a != b) & ~(a.isna() & b.isna())).sum())} / {len(m)}")
    print(df.groupby(df.date.dt.year).agg(rows=("ticker", "size"), tickers=("ticker", "nunique"), tradable=("f_tradable", "mean")).to_string())


if __name__ == "__main__":
    main()

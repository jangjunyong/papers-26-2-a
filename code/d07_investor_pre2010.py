"""동아리 DB 에서 KOSPI 투자자별 순매수·일별 시세 2000~2009 를 받는다 (10/29 피드백 'IMF 이후 데이터' 대응, 재개 가능).

출력: PAPERS/data/daily/club_db_ext/{investor,price}_kospi_2000_2009.parquet  (기존 club_db/ 는 건드리지 않는다)
사용: py -3 d07_investor_pre2010.py [investor] [price]   (인자 없으면 둘 다)
거래대금은 DB 에 2010 년부터만 있어 받지 않는다(패널에서 종가×거래량으로 채운다).
수집 방식(분기 조각·COUNT 대조·끊기면 절반 분할)은 d03_export_db_ext.py 를 그대로 쓴다.
"""
import sys

import d03_export_db_ext as d03

COLS = "i.code, i.date, i.indiv_buy, i.indiv_sell, i.indiv_net, i.inst_buy, i.inst_net, i.foreign_buy, i.foreign_net, i.other_corp_net"
FRM = ("FROM dataguide.daily_investor i JOIN dataguide.stock_master m ON m.code = i.code "
       "WHERE m.market = 'KOSPI' AND i.date >= :a AND i.date <= :b")
INTS = ["indiv_buy", "indiv_sell", "indiv_net", "inst_buy", "inst_net", "foreign_buy", "foreign_net", "other_corp_net"]

P_COLS = "p.code, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume, p.listed_shares, p.is_halted"
P_FRM = ("FROM dataguide.daily_price p JOIN dataguide.stock_master m ON m.code = p.code "
         "WHERE m.market = 'KOSPI' AND p.date >= :a AND p.date <= :b")
P_INTS = ["open", "high", "low", "close", "volume", "listed_shares", "is_halted"]

def price_from_dg():
    """DB 서버가 안 될 때(2026-11-01 접속 시간 초과): 볼트의 DataGuide 로컬 원본(data/daily/dg)으로 같은 스키마를 만든다.
    2010 년 대조에서 시가·종가·거래량·상장주식수·정지 여부는 DB 와 불일치 0, 수정주가만 75종목이 종목별 상수배(내려받은 시점의
    누적 조정계수 차이)라서, 2010 년 첫 겹치는 날의 DB/DG 비율로 2009 년 이전 수정주가를 맞춘다(경계 가짜 수익 방지)."""
    import pandas as pd
    dg = d03.PAPERS / "data" / "daily" / "dg"
    names = {"open": "open", "high": "high", "low": "low", "close": "close", "adj_close": "adj_close", "volume": "volume",
             "shares": "listed_shares", "halt": "is_halted"}
    w = {n: pd.read_parquet(dg / f"{n}.parquet") for n in names}
    cb = pd.read_parquet(d03.PAPERS / "data" / "daily" / "club_db" / "price_2010_2014.parquet", columns=["code", "date", "adj_close"])
    first = cb.sort_values("date").groupby("code").first()
    first.index = first.index.str[1:]
    ratio = {tk: r.adj_close / w["adj_close"].at[r.date, tk] for tk, r in first.iterrows()
             if tk in w["adj_close"] and pd.notna(w["adj_close"].at[r.date, tk]) and w["adj_close"].at[r.date, tk] > 0}
    w = {n: x.loc["2000-01-01":"2009-12-31"] for n, x in w.items()}
    w["adj_close"] = w["adj_close"] * pd.Series(ratio).reindex(w["adj_close"].columns).fillna(1.0)
    df = pd.concat({names[n]: x.stack(future_stack=True) for n, x in w.items()}, axis=1).rename_axis(["date", "ticker"]).reset_index()
    df = df[df.close.notna()]
    df.insert(0, "code", "A" + df.pop("ticker"))
    df["is_halted"] = df.is_halted.fillna(0)
    for c in ["open", "high", "low", "close", "volume", "listed_shares", "is_halted"]:
        df[c] = df[c].astype("float64")
    out = d03.OUT / "price_kospi_2000_2009.parquet"
    df.sort_values(["code", "date"]).to_parquet(out, index=False)
    print(f"price_dg: {len(df):,}행, 종목 {df.code.nunique()}, 배율 보정 종목 {sum(abs(v - 1) > 1e-6 for v in ratio.values())} → {out}")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["investor", "price"]
    if "price_dg" in todo:
        price_from_dg()
    if "investor" in todo:
        for a, b in d03.quarters(2000, 2009):
            d03.run_range("dginv_kospi_pre2010", f"SELECT {COLS} {FRM}", f"SELECT COUNT(*) {FRM}", a, b)
        d03.assemble("dginv_kospi_pre2010", "investor_kospi_2000_2009", ["code", "date"], INTS)
    if "price" in todo:
        for a, b in d03.quarters(2000, 2009):
            d03.run_range("price_kospi_pre2010", f"SELECT {P_COLS} {P_FRM}", f"SELECT COUNT(*) {P_FRM}", a, b)
        d03.assemble("price_kospi_pre2010", "price_kospi_2000_2009", ["code", "date"], P_INTS)

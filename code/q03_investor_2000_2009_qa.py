"""KOSPI 투자자별 순매수 2000~2026 연도·월별 품질 점검 (10/29 피드백 'IMF 이후 데이터' 대응).

점검: (1) 4그룹(개인·기관·외국인·기타법인) 순매수 합계가 0 인가 (2) 개인 매수−매도=순매수 항등식
      (3) 거래량이 있는데 4그룹이 모두 0 인 행(결측을 0 으로 채운 의심) (4) 가격 자료 대비 투자자 자료 커버리지
입력: data/daily/club_db_ext/investor_kospi_2000_2009.parquet, data/daily/club_db/investor_2010_2014.parquet, investor.parquet,
      data/daily/dg/{volume,close}.parquet (DataGuide 로컬, 거래량·커버리지 기준)
출력: results/q03_investor_yearly.csv, results/q03_investor_monthly.csv (표는 q03 메모에 옮겨 적는다)
사용: py -3 q03_investor_2000_2009_qa.py
"""
import pathlib
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PAPERS = pathlib.Path(__file__).resolve().parents[3]
D = PAPERS / "data" / "daily"
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
NETS = ["indiv_net", "inst_net", "foreign_net", "other_corp_net"]

inv = pd.concat([pd.read_parquet(D / "club_db_ext" / "investor_kospi_2000_2009.parquet"),
                 pd.read_parquet(D / "club_db" / "investor_2010_2014.parquet"),
                 pd.read_parquet(D / "club_db" / "investor.parquet")], ignore_index=True)
inv["code"] = inv.code.str.lstrip("A")
inv = inv[inv.date <= "2026-05-25"]                     # 이후는 외국인 정의 변경(q01 4절)이라 이 점검에서 뺀다
assert not inv.duplicated(["code", "date"]).any()

vol = pd.read_parquet(D / "dg" / "volume.parquet").stack().rename("volume").rename_axis(["date", "code"]).reset_index()
df = inv.merge(vol, on=["date", "code"], how="left")

df["s"] = df[NETS].sum(axis=1, min_count=4)            # 4열 중 하나라도 결측이면 NaN
df["all4"] = df[NETS].notna().all(axis=1)
df["nz"] = df.all4 & (df.s != 0)
df["rel"] = (df.s.abs() / df.volume).where(df.volume > 0)
df["zero_fill"] = (df[NETS] == 0).all(axis=1) & (df.volume >= 10000)
df["ident_bad"] = df[["indiv_buy", "indiv_sell", "indiv_net"]].notna().all(axis=1) & (df.indiv_buy - df.indiv_sell != df.indiv_net)


def summarize(g):
    return pd.Series({"rows": len(g), "codes": g.code.nunique(), "all4_rate": g.all4.mean(),
                      "sum_nonzero_rate": g.nz.sum() / max(g.all4.sum(), 1),
                      "rel_resid_median": g.rel.median(), "rel_resid_p99": g.rel.quantile(0.99),
                      "zero_fill_rows": int(g.zero_fill.sum()), "indiv_identity_bad": int(g.ident_bad.sum()),
                      "other_corp_nan_rate": g.other_corp_net.isna().mean(), "volume_missing_rate": g.volume.isna().mean()})


yearly = df.groupby(df.date.dt.year).apply(summarize)
monthly = df.groupby(df.date.dt.to_period("M")).apply(summarize)

# 커버리지: 그해 DataGuide 에서 거래량>0 인 날이 하루라도 있는 종목 수 대비 투자자 자료가 있는 종목 수
traded = vol[vol.volume > 0]
yearly["price_codes"] = traded.groupby(traded.date.dt.year).code.nunique()
yearly["coverage"] = yearly.codes / yearly.price_codes

RES.mkdir(exist_ok=True)
yearly.to_csv(RES / "q03_investor_yearly.csv")
monthly.to_csv(RES / "q03_investor_monthly.csv")
pd.set_option("display.width", 250)
print(yearly.round(4).to_string())
# 합계가 닫히기 시작/멈추는 경계를 일 단위로 찾는다
day = df[df.all4].groupby("date").nz.mean()
flip = day[(day > 0.05) != (day.shift() > 0.05)]
print("\n합계≠0 비율이 5%를 넘나드는 날(경계 후보):")
print(flip.round(3).to_string())

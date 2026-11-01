"""d05: 시장지수·무위험금리 공개 데이터 수집 + 검증 (약관·근거는 data/macro/macro_notes.md).

  py -3 d05_macro_public.py fetch      OECD(월별 금리·주가지수) + BIS(한국은행 기준금리 일별) → data/macro/*.csv
  py -3 d05_macro_public.py cpi        OECD 한국 월별 CPI(지수·전년비·전월비, 1995~) → data/macro/oecd_kor_cpi_monthly.csv
  py -3 d05_macro_public.py validate  패널 시총가중 수익률 vs 동아리 DB 지수·ETF·월말 KOSPI → 표 출력/저장
  py -3 d05_macro_public.py ecos       ECOS_API_KEY 환경변수가 있을 때만 일별 지수·시장금리 수집 (없으면 안내 후 종료)
  py -3 d05_macro_public.py ecos-items 802Y001   (키 필요) 통계표 항목코드 확인
  py -3 d05_macro_public.py test       rf_daily 룩어헤드 자체 점검

수집 금지: KRX(data.krx.co.kr·pykrx), Stooq·Yahoo·Naver(robots.txt 전면 Disallow), FRED 저장(약관 금지), 키 없는 ECOS/공공데이터포털.
"""
import io
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

V = pathlib.Path(__file__).resolve().parents[3]          # PAPERS/
OUT = V / "data" / "macro"
UA = {"User-Agent": "Mozilla/5.0 (academic research; contact jacob4374@gmail.com)"}

# OECD SDMX: 1회 호출로 한국 월별 4계열 (시간당 60회 제한, 이 스크립트는 1회만 쓴다)
OECD_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,/"
            "KOR.M.SHARE+IR3TIB+IRLT+IRSTCI......?startPeriod=1990-01&dimensionAtObservation=AllDimensions")
OECD_NAMES = {"IR3TIB": "ir3tib", "IRLT": "irlt", "IRSTCI": "irstci", "SHARE": "share_ix"}
BIS_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.KR?format=csv"

# ECOS (키 필요): 항목코드는 기억 기반이라 ecos-items 로 먼저 확인할 것 (미검증)
ECOS_JOBS = {  # 이름: (통계표, 주기, 항목코드)
    "kospi": ("802Y001", "D", "0001000"),
    "kosdaq": ("802Y001", "D", "0089000"),
    "cd91": ("817Y002", "D", "010502000"),
    "ktb3y": ("817Y002", "D", "010200000"),
    "call1d": ("817Y002", "D", "010101000"),
}


def get(url, accept=None):
    r = requests.get(url, headers={**UA, **({"Accept": accept} if accept else {})}, timeout=90)
    r.raise_for_status()
    return r


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(io.StringIO(get(OECD_URL, "application/vnd.sdmx.data+csv; charset=utf-8").text))
    m = d.pivot(index="TIME_PERIOD", columns="MEASURE", values="OBS_VALUE").rename(columns=OECD_NAMES)
    m.index = pd.to_datetime(m.index + "-01")                 # 월 시작일 = 그 달 '평균' 값의 라벨
    m.sort_index().rename_axis("month").to_csv(OUT / "oecd_kor_monthly.csv")
    b = pd.read_csv(io.StringIO(get(BIS_URL).text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    b.columns = ["date", "policy_rate"]
    b.to_csv(OUT / "bis_kor_policy_daily.csv", index=False)
    print(m.agg(["count"]).T.join(m.apply(lambda s: s.dropna().index.min()).rename("start")).join(m.apply(lambda s: s.dropna().index.max()).rename("end")))
    print("BIS", b.date.min(), b.date.max(), len(b))


CPI_URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,/"
           "KOR.M.N.CPI.._T.N.?startPeriod=1995-01&dimensionAtObservation=AllDimensions")


def cpi():
    """OECD 한국 월별 CPI(전 품목, 계절조정 없음): 지수(2015=100)·전년동월비·전월비 % → data/macro/oecd_kor_cpi_monthly.csv.
    라벨은 그 달 1일이지만 통계청 발표는 다음 달 초라서 국면 분할에 쓸 때는 1개월 늦춰 쓴다."""
    d = pd.read_csv(io.StringIO(get(CPI_URL, "application/vnd.sdmx.data+csv; charset=utf-8").text))
    d = d[d.TRANSFORMATION.isin(["_Z", "GY", "G1"])]
    m = d.pivot(index="TIME_PERIOD", columns="TRANSFORMATION", values="OBS_VALUE").rename(columns={"_Z": "cpi_ix", "GY": "cpi_yoy", "G1": "cpi_mom"})
    m.index = pd.to_datetime(m.index + "-01")
    m = m.sort_index().rename_axis("month")[["cpi_ix", "cpi_yoy", "cpi_mom"]]
    m.to_csv(OUT / "oecd_kor_cpi_monthly.csv")
    print(m.agg(["count", "min", "max"]).T, m.index.min().date(), m.index.max().date())


def ecos_key():
    k = os.environ.get("ECOS_API_KEY")
    if not k:
        print("ECOS_API_KEY 없음 -> 시도하지 않음. 발급: https://ecos.bok.or.kr/api/ (회원가입 후 인증키 신청), 발급 뒤 환경변수 설정.")
    return k


def ecos():
    k = ecos_key()
    if not k:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (tbl, cyc, item) in ECOS_JOBS.items():
        j = get(f"https://ecos.bok.or.kr/api/StatisticSearch/{k}/json/kr/1/100000/{tbl}/{cyc}/20050101/{pd.Timestamp.today():%Y%m%d}/{item}").json()
        rows = j.get("StatisticSearch", {}).get("row")
        if not rows:
            print(name, "실패/빈 응답:", j.get("RESULT")); continue
        df = pd.DataFrame(rows)[["TIME", "DATA_VALUE", "ITEM_NAME1", "UNIT_NAME"]]
        df.to_csv(OUT / f"ecos_{name}_daily.csv", index=False)
        print(name, df.TIME.min(), df.TIME.max(), len(df), df.ITEM_NAME1.iloc[0], df.UNIT_NAME.iloc[0])


def ecos_items(tbl):
    k = ecos_key()
    if k:
        for r in get(f"https://ecos.bok.or.kr/api/StatisticItemList/{k}/json/kr/1/200/{tbl}").json()["StatisticItemList"]["row"]:
            print(r["ITEM_CODE"], r["ITEM_NAME"], r["CYCLE"], r["START_TIME"], r["END_TIME"])


# ---------------------------------------------------------------- 무위험수익률 (룩어헤드 방지)
def _accrue(rate_pct, dates):
    """연율 % -> 거래일 사이 구간수익률(소수). ACT/365 복리, 월요일은 달력 3일치."""
    gap = pd.Series(dates, index=dates).diff().dt.days.fillna(1)
    return (1 + rate_pct / 100) ** (gap / 365) - 1


def rf_daily(monthly_pct: pd.Series, dates, lag_months: int = 1) -> pd.Series:
    """월별 금리 -> 거래일별 무위험수익률. 월 m 의 날에는 m-lag_months 월 값만 쓴다.
    monthly_pct: 인덱스 = 월 시작일, 값 = 그 달 평균 연율 %. OECD 월값은 월평균이라 같은 달 값은 월말에야 확정되므로 lag>=1 이 기본.
    (월 시작일 아침에 알 수 있는 건 전월까지의 평균뿐.)"""
    dates = pd.DatetimeIndex(dates)
    s = monthly_pct.dropna().copy()
    s.index = s.index.to_period("M").to_timestamp() + pd.DateOffset(months=lag_months)   # 값이 쓰이기 시작하는 달의 1일
    return _accrue(s.reindex(dates, method="ffill"), dates)


def rf_daily_from_daily(daily_pct: pd.Series, dates) -> pd.Series:
    """일별 금리(기준금리·ECOS CD91) -> 거래일별 무위험수익률. 전 거래일 값 사용(당일 값 금지)."""
    dates = pd.DatetimeIndex(dates)
    return _accrue(daily_pct.reindex(dates.union(daily_pct.index)).ffill().reindex(dates).shift(1), dates)


def selftest():
    m = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    d = pd.bdate_range("2020-02-03", "2020-03-31")
    r = rf_daily(m, d)
    assert abs(r["2020-02-04"] - (1.01 ** (1 / 365) - 1)) < 1e-12      # 2월 -> 1월 값
    assert abs(r["2020-03-03"] - (1.02 ** (1 / 365) - 1)) < 1e-12      # 3월 -> 2월 값
    assert abs(r["2020-02-10"] - (1.01 ** (3 / 365) - 1)) < 1e-12      # 월요일 3일치
    m2 = m.copy(); m2.iloc[2] = 99.0                                   # 3월 값을 바꿔도 3월 날짜는 불변
    assert rf_daily(m2, d)["2020-03-02":"2020-03-31"].equals(r["2020-03-02":"2020-03-31"])
    p = pd.Series([1.0, 5.0], index=pd.to_datetime(["2020-02-03", "2020-02-05"]))
    assert abs(rf_daily_from_daily(p, d)["2020-02-05"] - (1.01 ** (1 / 365) - 1)) < 1e-12   # 5일 인상, 5일 수익률엔 전일 값
    print("rf_daily OK")


# ---------------------------------------------------------------- 검증
def clubdb_index():
    """동아리 DB(읽기 전용, 쿼리 1개씩): KODEX200 / KODEX코스닥150 의 close·adj_close·기초지수. 한번 받으면 CSV 재사용."""
    f = OUT / "clubdb_etf_index_daily.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    sys.path.insert(0, r"C:\dataguide")
    os.chdir(r"C:\dataguide")
    from dataguide import DataGuide
    import time
    parts = []
    for code, tag in (("A069500", "kodex200"), ("A229200", "kodex_kosdaq150")):
        for i in range(6):                                              # 터널 직후 SSL WRONG_VERSION 은 일시적 -> 새 접속으로 재시도 (d03 과 같은 처리)
            try:
                q = DataGuide().query("SELECT date, close, adj_close, idx_close FROM kiwoom.etf_daily WHERE code = :a ORDER BY date", {"a": code})
                break
            except Exception as ex:
                print("DB 재시도", i + 1, str(ex)[:80]); time.sleep(5)
        else:
            raise RuntimeError("DB 접속 실패")
        parts.append(q.set_index("date").add_prefix(tag + "_"))
    df = pd.concat(parts, axis=1).reset_index()
    df.to_csv(f, index=False)
    return df


def cap_weighted():
    p = pd.read_parquet(V / "data" / "daily" / "club_db" / "panel.parquet", columns=["date", "ticker", "mktcap", "ret1d_clean"])
    p = p.sort_values(["ticker", "date"])
    p["w"] = p.groupby("ticker")["mktcap"].shift(1)                    # 전일 시총 가중 (당일 정보 금지)
    p = p[p.ret1d_clean.notna() & p.w.notna() & (p.w > 0)]
    g = p.assign(x=p.ret1d_clean * p.w).groupby("date")
    return (g["x"].sum() / g["w"].sum()).rename("capw"), g.size().rename("n")


def yearly(df):
    """열별 연도 복리수익률 (%) — 해당 연도의 모든 열이 있는 날만."""
    return ((1 + df).groupby(df.index.year).prod() - 1) * 100


def validate():
    e = clubdb_index().set_index("date")
    e = e.where(e > 0)                                                 # idx_close 의 0 은 결측 (KODEX200 2012-10 1일, 코스닥150 2015~16 365일)
    cw, n = cap_weighted()
    print(f"패널 시총가중: {cw.index.min():%Y-%m-%d} ~ {cw.index.max():%Y-%m-%d}, {len(cw)}일, 종목수 중앙값 {int(n.median())}")
    pc = lambda s: s.pct_change(fill_method=None)                      # 결측 다음 날 수익률도 결측 -> 아래 dropna 가 같은 날을 모든 열에서 제거
    r = pd.DataFrame({
        "capw": cw,
        "kospi200_idx": pc(e.kodex200_idx_close),
        "kodex200_close": pc(e.kodex200_close),
        "kodex200_adj": pc(e.kodex200_adj_close),
    }).dropna()
    r = r[r.index >= cw.index.min()]
    print(f"\n[일별] 겹치는 기간 {r.index.min():%Y-%m-%d} ~ {r.index.max():%Y-%m-%d}, {len(r)}일")
    for c in r.columns[1:]:
        d = r.capw - r[c]
        print(f"  capw vs {c:15s} corr={r.capw.corr(r[c]):.4f}  일별차 연환산 TE={d.std() * np.sqrt(252) * 100:.2f}%p  "
              f"|차|>1%p 일수={int((d.abs() > 0.01).sum())}")
    print(f"  kospi200_idx vs kodex200_close corr={r.kospi200_idx.corr(r.kodex200_close):.4f}  kospi200_idx vs kodex200_adj corr={r.kospi200_idx.corr(r.kodex200_adj):.4f}")
    dd = (r.capw - r.kospi200_idx)
    print("  capw-idx 차이가 큰 날 5개:", {f"{i:%Y-%m-%d}": round(v * 100, 2) for i, v in dd.reindex(dd.abs().nlargest(5).index).items()})
    ys = yearly(r)
    t = pd.DataFrame({"capw": ys.capw, "kospi200_idx": ys.kospi200_idx, "kodex200_close": ys.kodex200_close, "kodex200_adj": ys.kodex200_adj,
                      "capw-idx": ys.capw - ys.kospi200_idx, "capw-close": ys.capw - ys.kodex200_close, "capw-adj": ys.capw - ys.kodex200_adj,
                      "close-idx": ys.kodex200_close - ys.kospi200_idx, "adj-close": ys.kodex200_adj - ys.kodex200_close,
                      "days": r.groupby(r.index.year).size()}).round(2)
    print("\n[연도별 복리수익률 %, 차이 %p]  (days<240 인 연도는 부분연도; adj-close 는 분배금이면 +1~2%p 여야 함)\n", t.to_string())
    t.to_csv(OUT / "validation_yearly.csv")
    # KODEX200 adj_close 조정계수(adj/close)의 큰 하향 점프 = 분배금 재투자라면 나올 수 없는 방향
    lj = np.log(e.kodex200_adj_close / e.kodex200_close).diff().dropna()
    ng = lj[lj < -1e-3]
    print(f"\n[KODEX200 adj_close 점검] 조정계수 하향 점프(<-0.1%) {len(ng)}건: " + ", ".join(f"{d:%Y-%m-%d} {v * 100:.1f}%" for d, v in ng.items())
          + f"\n  전체 누적 조정 {lj.sum() * 100:.1f}% , 하향 점프를 0 으로 두면 {lj.where(lj >= -1e-3, 0).sum() * 100:.1f}% (연 1.5% 배당 x 24년 ~ 40% 와 일치)")

    # 월별: DataGuide 월말 KOSPI 종합(I.001) vs 패널 시총가중 월수익률 (동아리 라이선스 로컬 CSV)
    kf = V / "data" / "csv" / "코스피 업종 종가지수.csv"
    k = pd.read_csv(kf, index_col=0, parse_dates=True, usecols=[0, 1], encoding="utf-8-sig")
    k.columns = ["kospi"]
    k = k[k.index.to_period("M") < pd.Timestamp.fromtimestamp(kf.stat().st_mtime).to_period("M")]   # 파일을 받은 달 행은 부분월(월말 라벨이지만 받은 날까지) -> 제외
    km = k.kospi.pct_change().dropna()
    km.index = km.index.to_period("M")
    cm = ((1 + cw).groupby(cw.index.to_period("M")).prod() - 1)
    cm = cm[cm.index >= pd.Period(cw.index.min(), "M") + 1]            # 첫 달은 부분월이라 제외
    mm = pd.concat([cm.rename("capw"), km.rename("kospi_I001")], axis=1).dropna()
    mm = mm[mm.index < pd.Period(cw.index.max(), "M")]                 # 마지막 달 부분월 제외
    print(f"\n[월별] 패널 시총가중 vs DataGuide 월말 KOSPI(I.001): {mm.index.min()} ~ {mm.index.max()}, {len(mm)}개월, "
          f"corr={mm.capw.corr(mm.kospi_I001):.4f}, 월차 TE(연환산)={(mm.capw - mm.kospi_I001).std() * np.sqrt(12) * 100:.2f}%p")
    yy = yearly(mm.rename(columns=lambda c: c))
    print((yy.assign(diff=yy.capw - yy.kospi_I001)).round(2).to_string())

    # 공식 지수끼리: KOSPI 종합(I.001) vs KOSPI200(DB idx_close) 월말 -> 패널-KODEX 차이가 패널 탓인지 지수 구성 차이인지 가르는 기준
    mp = lambda s: s.dropna().groupby(s.dropna().index.to_period("M")).last()           # 월말값 (PeriodIndex)
    k.index = k.index.to_period("M")
    yr = lambda s: ((1 + s.pct_change().dropna()).groupby(s.pct_change().dropna().index.year).prod() - 1) * 100
    off = pd.concat([k.kospi.rename("I001"), mp(e.kodex200_idx_close).rename("kospi200")], axis=1).dropna()
    oy = pd.DataFrame({"KOSPI(I001)": yr(off.I001), "KOSPI200": yr(off.kospi200)}); oy["차"] = oy.iloc[:, 0] - oy.iloc[:, 1]
    print(f"\n[공식 월말지수끼리 연도별 %] 월수익률 corr={off.pct_change().corr().iloc[1, 0]:.4f}\n", oy.loc[2010:].round(2).T.to_string())

    # KOSDAQ: 일별 종합지수는 못 구함 -> KODEX 코스닥150(2015-10~)이 DataGuide 월말 KOSDAQ(I.201)을 얼마나 따라가는지
    qf = V / "data" / "csv" / "코스닥 업종 종가지수.csv"
    q = pd.read_csv(qf, index_col=0, parse_dates=True, usecols=[0, 1], encoding="utf-8-sig").iloc[:, 0]
    q = q[q.index.to_period("M") < pd.Timestamp.fromtimestamp(qf.stat().st_mtime).to_period("M")]
    q.index = q.index.to_period("M")
    kd = pd.concat([q.rename("I201"), mp(e.kodex_kosdaq150_adj_close).rename("kq150")], axis=1).dropna()
    qy = pd.DataFrame({"KOSDAQ(I201)": yr(kd.I201), "KODEX코스닥150(adj)": yr(kd.kq150)}); qy["차"] = qy.iloc[:, 0] - qy.iloc[:, 1]
    print(f"\n[KOSDAQ 월말 vs KODEX 코스닥150 연도별 %] {kd.index.min()}~{kd.index.max()} 월수익률 corr={kd.pct_change().corr().iloc[1, 0]:.4f}\n", qy.loc[2016:].round(2).T.to_string())

    # OECD SHARE(월평균 지수, 2015=100)가 KOSPI 종합 기반인지: 월평균 ~ (전월말+당월말)/2 근사와의 비율이 상수에 가까운가
    o = pd.read_csv(OUT / "oecd_kor_monthly.csv", index_col=0, parse_dates=True).share_ix
    o.index = o.index.to_period("M")
    ratio = (o / ((k.kospi + k.kospi.shift(1)) / 2)).dropna()
    print(f"\n[OECD SHARE / KOSPI 월평균근사] {ratio.index.min()}~{ratio.index.max()} 평균={ratio.mean():.4f} 변동계수={ratio.std() / ratio.mean() * 100:.1f}% "
          f"(2015=100 기준이므로 상수에 가까우면 KOSPI 종합 기반)  월변화율 corr={np.log(o).diff().corr(np.log((k.kospi + k.kospi.shift(1)) / 2).diff()):.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"fetch": fetch, "cpi": cpi, "validate": validate, "ecos": ecos, "test": selftest}.get(cmd, lambda: ecos_items(sys.argv[2]) if cmd == "ecos-items" else print(__doc__))()

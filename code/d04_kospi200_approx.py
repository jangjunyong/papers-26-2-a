"""d04 KOSPI200 구성종목 근사 + 시장수익률 대용 시계열.
입력: PAPERS/data/daily/club_db/panel.parquet(+trading*.parquet), 동아리 DB(읽기전용, 캐시 없을 때만 조회)
  - dataguide.daily_shares.floating_shares -> club_db/floating_shares.parquet (캐시)
  - kiwoom.etf_daily A069500(KODEX 200)    -> club_db/kodex200_daily.parquet (캐시)
출력(전부 PAPERS/data/제도일정/): market_ret_daily.parquet, kospi200_approx_mask.parquet,
  kospi200_review_dates.csv, kospi200_approx_notes.md
KRX 사이트 수집·pykrx 사용 안 함. 외부 웹 데이터 수집 없음.

### 사전 고정 규칙 (결과를 보기 전에 정했다. 결과를 본 뒤 바꾸지 않았다)
- 정기변경 효력일 = 매년 6월·12월 '둘째 목요일 다음 거래일'(가정). 2014-12 ~ 2026-06 총 25회.
- 선정 기준일 = 직전 4월말(6월 변경)·10월말(12월 변경)의 마지막 거래일(FSC 2021-02-03 보도자료의 '4월·10월말 기준' 서술).
  평균 구간 = 그 기준일까지 6개월(달력, 4월말이면 11/1~4/30).
- 후보 = 패널 종목(KOSPI 보통주) 중 구간 내 유효일 >= 60 이고 효력일에 패널 행이 있는 종목(효력일 전 상폐 확정 종목 제외).
- V1 구간 평균 유동시총(유동비율 x 상장주식수 x 종가; 유동비율 결측이면 상장주식수로 대체) 상위 200
  유동비율 = floating_shares / listed_shares 를 1·4·7·10월말 거래일 스냅샷으로만 조회해 다음 스냅샷까지 이월(서버 부하 때문, 일별 전수 조회 포기)
- V2 구간 평균 상장시총(listed_shares x 종가) 상위 200
- V3 V1 상위 200 중 평균 거래대금(tv_filled)의 5% 분위수 미만 종목을 빼고, 유동시총 다음 순위 중 같은 분위수 이상인 종목으로 200개 채움
- 구성 변경 사이에 상폐(패널 행 소멸)되면 그 날부터 빠지고 다음 변경까지 보충하지 않는다.
- 유동비율 DB 공백(2026-06-23~)은 마지막 스냅샷(2026-06-22) 이월. 선정은 이 공백 이전(2026-06 변경까지)에 끝난다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

V = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS/data"
CB = f"{V}/daily/club_db"
OUT = os.environ.get("D04_OUT", f"{V}/제도일정")                       # 환경변수는 시험 실행용
FLOAT_CACHE = os.environ.get("D04_FLOAT", f"{CB}/floating_shares.parquet")
ETF_CACHE = os.environ.get("D04_ETF", f"{CB}/kodex200_daily.parquet")

TOP_N = 200
MIN_DAYS = 60
LIQ_CUT = 0.05
MASK_START = pd.Timestamp("2015-01-01")
FLOAT_DB_LAST = pd.Timestamp("2026-06-22")
REF_MONTH = {6: 4, 12: 10}
SNAP_MONTHS = (1, 4, 7, 10)
GAP_THR = 0.01          # 지수 일수익률과 KODEX 수익률이 1%p 넘게 벌어지면 이상값 표시
PART_START, PART_END = pd.Timestamp("2021-05-03"), pd.Timestamp("2023-11-03")   # 부분허용(마지막 거래일 11-03)


# ---------------------------------------------------------------- DB (캐시 없을 때만, 한 번에 쿼리 하나)
def _dg():
    sys.path.insert(0, r"C:\dataguide")
    os.chdir(r"C:\dataguide")
    from dataguide import DataGuide
    return DataGuide()


def _q(dg, sql, prm=None, tries=240, wait=30):
    """서버가 바쁘거나 죽어 접속이 안 되면 기다렸다 재시도(공유 서버라 연타하지 않는다)."""
    for k in range(tries):
        try:
            return dg.query(sql, prm)
        except Exception as e:  # noqa: BLE001  OperationalError 등
            print("DB 재시도", k, str(e)[:90].replace(chr(10), " "), flush=True)
            time.sleep(wait)
    raise RuntimeError("DB 응답 없음")


def fetch_float(codes, dates):
    # 종목 10개 x 스냅샷 날짜 IN 조회(PK 점조회). 일별 전수(범위) 조회는 서버가 느려(종목당 수십 초) 포기했다. 조각 단위로 저장해 재개 가능.
    dg = _dg()
    pdir = f"{CB}/_floating_parts"
    os.makedirs(pdir, exist_ok=True)
    dh = ",".join(f":d{j}" for j in range(len(dates)))
    for i in range(0, len(codes), 10):
        f = f"{pdir}/p{i:04d}.parquet"
        if os.path.exists(f):
            continue
        cs = codes[i:i + 10]
        ph = ",".join(f":c{j}" for j in range(len(cs)))
        prm = {f"c{j}": c for j, c in enumerate(cs)} | {f"d{j}": d for j, d in enumerate(dates)}
        _q(dg, f"SELECT code, date, floating_shares FROM dataguide.daily_shares WHERE code IN ({ph}) AND date IN ({dh})",
           prm).to_parquet(f, index=False)
        print("float", i, flush=True)
    out = pd.concat([pd.read_parquet(f"{pdir}/{x}") for x in sorted(os.listdir(pdir))], ignore_index=True)
    out.to_parquet(FLOAT_CACHE, index=False)


def fetch_etf():
    # trace_eor_rt<5: 키움 원천 오류(2007-08-31) 제외 규칙. NULL 은 살린다.
    _q(_dg(), "SELECT date, close, adj_close, nav, idx_close, trace_eor_rt FROM kiwoom.etf_daily "
       "WHERE code='A069500' AND date >= '2009-12-01' AND (trace_eor_rt < 5 OR trace_eor_rt IS NULL) "
       "ORDER BY date").to_parquet(ETF_CACHE, index=False)


# ---------------------------------------------------------------- 유틸
def to_md(df, fmt=None, index=True):
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = list(d.columns)
    rows = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (fmt.get(c, "{:.2f}")).format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def second_thursday(y, m):
    d = pd.date_range(f"{y}-{m:02d}-01", periods=31)
    d = d[(d.month == m) & (d.dayofweek == 3)]
    return d[1]


def wret(d, wcol, rcol="ret"):
    ok = d[rcol].notna() & d[wcol].notna() & (d[wcol] > 0)
    d = d[ok]
    return (d[wcol] * d[rcol]).groupby(d.date).sum() / d[wcol].groupby(d.date).sum()


def stats(r, idx):
    """r, idx: 같은 날짜 index 의 일수익률. 상관, 추적오차(일 bp / 연 %), 평균차(bp/일), 누적수익률 차(%p)."""
    x = pd.concat([r, idx], axis=1, keys=["r", "i"]).dropna()
    d = x.r - x.i
    return dict(n=len(x), corr=x.r.corr(x.i), te_bp=d.std() * 1e4, te_ann=d.std() * np.sqrt(252) * 100,
                mdiff_bp=d.mean() * 1e4, cumdiff=((1 + x.r).prod() - (1 + x.i).prod()) * 100)


def annstats(r):
    r = r.dropna()
    cum = (1 + r).prod() - 1
    return len(r), ((1 + cum) ** (252 / len(r)) - 1) * 100, r.std() * np.sqrt(252) * 100


# ---------------------------------------------------------------- 1. 로드
panel_cols = ["ticker", "date", "close", "volume", "listed_shares", "is_halted", "ret1d_clean", "tv_filled"]
p = pd.read_parquet(f"{CB}/panel.parquet", columns=panel_cols).sort_values(["ticker", "date"]).reset_index(drop=True)
p = p.rename(columns={"ret1d_clean": "ret"})
cal = pd.DatetimeIndex(np.sort(p.date.unique()))

# 상장주식수 2026-06-23~ 결측은 trading.parquet 으로 메움(두 곳 값은 겹치는 구간에서 100% 일치), 그래도 남으면 종목별 이월
tr = pd.concat([pd.read_parquet(f"{CB}/trading.parquet"), pd.read_parquet(f"{CB}/trading_2010_2014.parquet")])
tr = tr.drop_duplicates(["code", "date"]).assign(ticker=lambda d: d.code.str[1:])[["ticker", "date", "listed_shares"]]
p = p.merge(tr.rename(columns={"listed_shares": "ls_tr"}), on=["ticker", "date"], how="left")
p["listed_shares"] = p.listed_shares.fillna(p.ls_tr)
p["listed_shares"] = p.groupby("ticker").listed_shares.ffill()
p = p.drop(columns="ls_tr")

codes = ["A" + t for t in sorted(p.ticker.unique())]
_c = pd.Series(cal, index=cal)
_c = _c[(_c >= "2014-04-01") & (_c <= FLOAT_DB_LAST)]
_m = _c[_c.index.month.isin(SNAP_MONTHS)]
SNAP = sorted(set(_m.groupby([_m.index.year, _m.index.month]).max()) | {_c.index.max()})
if not os.path.exists(FLOAT_CACHE):
    fetch_float(codes, [d.strftime("%Y-%m-%d") for d in SNAP])
if not os.path.exists(ETF_CACHE):
    fetch_etf()
fl = pd.read_parquet(FLOAT_CACHE).assign(ticker=lambda d: d.code.str[1:])[["ticker", "date", "floating_shares"]]
p = p.merge(fl, on=["ticker", "date"], how="left")
# 유동비율 = 스냅샷일 floating_shares / listed_shares. 스냅샷 NaN 은 -1 로 표시해 다음 스냅샷까지 '결측'으로 이월(예전 값으로 메우지 않음)
is_snap = p.date.isin(SNAP)
r0 = np.where(is_snap, (p.floating_shares / p.listed_shares).fillna(-1.0), np.nan)
r0 = pd.Series(r0, index=p.index).groupby(p.ticker).ffill()
ratio = r0.where(r0 > 0).clip(upper=1.0)                       # 유동비율이 1 을 넘으면 1 로 자름
p["fl_ratio"] = ratio
p["fl_sub"] = ratio.isna()                                      # 유동비율 결측 -> 상장주식수로 대체한 날
p["fl"] = p.listed_shares * ratio.fillna(1.0)
p["fcap"] = p.fl * p.close
p["lcap"] = p.listed_shares * p.close
g = p.groupby("ticker", sort=False)
p["fcap_prev"] = g.fcap.shift(1)
p["lcap_prev"] = g.lcap.shift(1)

# ---------------------------------------------------------------- 2. 시장수익률 대용 시계열
etf = pd.read_parquet(ETF_CACHE).assign(date=lambda d: pd.to_datetime(d.date)).set_index("date")
etf_p = etf[["adj_close", "idx_close"]].where(lambda d: d > 0).reindex(cal)     # 0 이하 값은 결측(예: 2012-06-20 idx_close=0, nav=0)
etf_r = etf_p.pct_change(fill_method=None)
mk = pd.DataFrame({
    "mkt_ret_cw": wret(p, "lcap_prev"),
    "mkt_ret_ew": p.dropna(subset=["ret"]).groupby("date").ret.mean(),
    "n_stocks": p.dropna(subset=["ret"]).groupby("date").ret.size(),
}).reindex(cal)
mk.index.name = "date"
mk["kodex200_ret"] = etf_r.adj_close
mk["kospi200_idx_ret"] = etf_r.idx_close
# 이상값 표시: 1 결측(그날 또는 전날 지수 없음) / 2 지수가 전일과 같은데 ETF 는 0.5% 넘게 움직임 / 3 지수-ETF 수익률 차 > 1%p
stale = (etf_p.idx_close.diff() == 0) & (mk.kodex200_ret.abs() > 0.005)
stale = stale | stale.shift(1, fill_value=False)       # 정체가 풀리는 날의 지수 수익률은 여러 날치라 같이 표시
gap = (mk.kospi200_idx_ret - mk.kodex200_ret).abs() > GAP_THR
mk["kospi200_idx_flag"] = np.select([mk.kospi200_idx_ret.isna(), stale, gap], [1, 2, 3], 0).astype("int8")
mk["kodex200_missing"] = mk.kodex200_ret.isna()
f3 = mk.kospi200_idx_flag == 3        # 불일치일에 전체종목 시총가중 수익률이 지수 쪽에 더 가까운 날 수(어느 쪽이 틀렸는지 힌트)
f3_idx_closer = int(((mk.kospi200_idx_ret - mk.mkt_ret_cw).abs() < (mk.kodex200_ret - mk.mkt_ret_cw).abs())[f3].sum())
mk.reset_index().to_parquet(f"{OUT}/market_ret_daily.parquet", index=False)
okd = mk.kospi200_idx_flag.isin([0, 3]) & mk.kodex200_ret.notna()
print("market_ret_daily", mk.shape, "flag counts", mk.kospi200_idx_flag.value_counts().to_dict())

# ---------------------------------------------------------------- 3. 정기변경일
rows = []
for y in range(2014, 2027):
    for m in (6, 12):
        d2 = second_thursday(y, m)
        later = cal[cal > d2]
        if len(later) == 0:
            continue
        eff = later[0]
        me = pd.Timestamp(y, REF_MONTH[m], 1) + pd.offsets.MonthEnd(0)
        ref = cal[cal <= me][-1]
        ws = pd.Timestamp(y, REF_MONTH[m], 1) - pd.DateOffset(months=5)
        rows.append(dict(second_thursday=d2, effective_date=eff, ref_date=ref, window_start=ws))
rv = pd.DataFrame(rows)
rv = rv[rv.effective_date >= pd.Timestamp("2014-12-01")].reset_index(drop=True)
rv.index.name = "rb"

# 자체 점검: 사전 정리해 둔 기사 수준 효력일(2021-12-10, 2022-12-09, 2023-06-09, 2025-12-12, 2026-06-12)과 규칙 비교
_chk = {"2021-12-10": 0, "2022-12-09": 0, "2023-06-09": 0, "2025-12-12": 0, "2026-06-12": 0}
for k in _chk:
    assert pd.Timestamp(k) in set(rv.effective_date), f"규칙 효력일이 기사 수준 일자와 다름: {k}"

# ---------------------------------------------------------------- 4. 선정
mem_parts, rv_extra = [], []
for rb, r in rv.iterrows():
    w = p[(p.date >= r.window_start) & (p.date <= r.ref_date)]
    gg = w.groupby("ticker")
    st = pd.DataFrame({"ndays": gg.close.count(), "avg_fcap": gg.fcap.mean(), "avg_lcap": gg.lcap.mean(),
                       "avg_tv": gg.tv_filled.mean(), "sub_frac": gg.fl_sub.mean()})
    present = set(p.loc[p.date == r.effective_date, "ticker"])
    st = st[(st.ndays >= MIN_DAYS) & st.index.isin(present) & st.avg_fcap.notna() & st.avg_lcap.notna()].copy()
    st["rank_fcap"] = st.avg_fcap.rank(ascending=False, method="first")
    st["rank_lcap"] = st.avg_lcap.rank(ascending=False, method="first")
    st["v1"] = st.rank_fcap <= TOP_N
    st["v2"] = st.rank_lcap <= TOP_N
    tau = st.loc[st.v1, "avg_tv"].quantile(LIQ_CUT)
    cand = st[st.avg_tv >= tau].sort_values("rank_fcap").head(TOP_N)
    st["v3"] = st.index.isin(cand.index)
    st["fl_sub_major"] = st.sub_frac > 0.5
    cut = st.loc[st.rank_fcap == TOP_N, "avg_fcap"].iloc[0]
    rv_extra.append(dict(n_eligible=len(st), n_v1=int(st.v1.sum()), n_v2=int(st.v2.sum()), n_v3=int(st.v3.sum()),
                         n_fl_sub_v1=int((st.v1 & st.fl_sub_major).sum()), liq_cut_tv_won=tau,
                         cutoff_fcap_won=cut, n_near_cutoff_20pct=int(((st.avg_fcap / cut - 1).abs() <= 0.2).sum()),
                         v1_v2_overlap=int((st.v1 & st.v2).sum()), v1_v3_overlap=int((st.v1 & st.v3).sum())))
    st["rb"] = rb
    mem_parts.append(st.reset_index())
mem = pd.concat(mem_parts, ignore_index=True)
for v in ("v1", "v2", "v3"):                      # 직전 변경 대비 편입·편출 수
    sets = [set(mem.loc[(mem.rb == k) & mem[v], "ticker"]) for k in rv.index]
    rv_extra_in = [np.nan] + [len(sets[k] - sets[k - 1]) for k in range(1, len(sets))]
    rv_extra_out = [np.nan] + [len(sets[k - 1] - sets[k]) for k in range(1, len(sets))]
    for k in range(len(sets)):
        rv_extra[k][f"{v}_in"], rv_extra[k][f"{v}_out"] = rv_extra_in[k], rv_extra_out[k]
rv = pd.concat([rv, pd.DataFrame(rv_extra, index=rv.index)], axis=1)
last = p.groupby("ticker").date.max()                 # 구간 중 상폐(패널 행 소멸)로 빠진 종목 수
nxt = list(rv.effective_date.iloc[1:]) + [cal[-1] + pd.Timedelta(days=1)]
for v in ("v1", "v2", "v3"):
    rv[f"{v}_delisted"] = [int((last.reindex(mem.loc[(mem.rb == k) & mem[v], "ticker"]) < cal[cal < nxt[k]][-1]).sum())
                           for k in rv.index]
rv.reset_index().to_csv(f"{OUT}/kospi200_review_dates.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 5. 일별 마스크 (패널 거래일 기준, 2015-01-01 ~)
pm = p[p.date >= MASK_START].copy()
pm["rb"] = np.searchsorted(rv.effective_date.values, pm.date.values, side="right") - 1
assert (pm.rb >= 0).all()
pm = pm.merge(mem[["rb", "ticker", "v1", "v2", "v3", "fl_sub_major", "rank_fcap"]], on=["rb", "ticker"], how="left")
for v in ("v1", "v2", "v3", "fl_sub_major"):
    pm[v] = pm[v].eq(True)
pm["rank_fcap_v1"] = pd.array(pm.rank_fcap, dtype="Int16")      # 직전 변경 시점 유동시총 순위(후보 아니면 결측). 경계 종목 판별용 추가 열
mask = pm[["ticker", "date", "v1", "v2", "v3", "fl_sub_major", "rank_fcap_v1"]].rename(columns={
    "v1": "in_kospi200_v1", "v2": "in_kospi200_v2", "v3": "in_kospi200_v3", "fl_sub_major": "float_substituted"})
mask.to_parquet(f"{OUT}/kospi200_approx_mask.parquet", index=False)
cnt = pm.groupby("date")[["v1", "v2", "v3"]].sum()
assert (cnt <= TOP_N).all().all() and (pm.date >= pm.rb.map(rv.effective_date)).all()
print("mask", mask.shape, "일별 구성 수 min/max", cnt.min().to_dict(), cnt.max().to_dict())

# ---------------------------------------------------------------- 6. 검증
# 검증 기준 지수: 결측(1)·정체(2) 날은 뺀다. 3(지수-ETF 불일치)은 어느 쪽이 틀렸는지 알 수 없어 남긴다.
idx = mk.kospi200_idx_ret.where(mk.kospi200_idx_flag.isin([0, 3]))
series = {f"V{i}": wret(pm[pm[f"v{i}"]], "fcap_prev") for i in (1, 2, 3)}
series["전체종목(시총)"] = mk.mkt_ret_cw
series["전체종목(동일)"] = mk.mkt_ret_ew
series["KODEX200"] = mk.kodex200_ret
S = pd.DataFrame(series).loc[MASK_START:]
S["IDX"] = idx.loc[MASK_START:]
S = S.dropna()
cols = ["V1", "V2", "V3", "전체종목(시총)", "전체종목(동일)", "KODEX200"]
tot = pd.DataFrame({c: stats(S[c], S.IDX) for c in cols}).T
f0 = S[mk.kospi200_idx_flag.reindex(S.index) == 0]                  # 지수-ETF 불일치(3) 날을 뺀 비교(KODEX 쪽 이상값 영향 확인용)
for c in ("V1", "KODEX200"):
    tot.loc[f"{c} (플래그 0 일만)"] = pd.Series(stats(f0[c], f0.IDX))
worst = (S.V1 - S.IDX).abs().nlargest(10).index
worst = pd.DataFrame({"V1": S.V1[worst] * 100, "지수": S.IDX[worst] * 100, "V1-지수(bp)": (S.V1 - S.IDX)[worst] * 1e4,
                      "KODEX": S.KODEX200[worst] * 100, "전체시총가중": S["전체종목(시총)"][worst] * 100}).sort_index()
worst.index = worst.index.strftime("%Y-%m-%d")
worst.index.name = "날짜"
part = S.loc[PART_START:PART_END]
tot_p = pd.DataFrame({c: stats(part[c], part.IDX) for c in cols}).T
yr = {}
for y, d in S.groupby(S.index.year):
    yr[y] = pd.DataFrame({c: stats(d[c], d.IDX) for c in cols}).T
te_y = pd.DataFrame({y: t.te_bp for y, t in yr.items()}).T
co_y = pd.DataFrame({y: t["corr"] for y, t in yr.items()}).T
cd_y = pd.DataFrame({y: t.cumdiff for y, t in yr.items()}).T
n_y = pd.Series({y: int(t.n.iloc[0]) for y, t in yr.items()})

# 반기(변경 구간)별 V1 추적오차 - 구성 노후화 확인
S["rb"] = np.searchsorted(rv.effective_date.values, S.index.values, side="right") - 1
by_rb = S.groupby("rb").apply(lambda d: pd.Series({"te_bp_V1": (d.V1 - d.IDX).std() * 1e4, "te_bp_V2": (d.V2 - d.IDX).std() * 1e4,
                                                     "te_bp_V3": (d.V3 - d.IDX).std() * 1e4}), include_groups=False)

# 시장수익률 대용 표 (전 기간 2010~, IDX 정상일 & KODEX 있는 날)
M = mk[okd & mk.mkt_ret_cw.notna()][["kospi200_idx_ret", "mkt_ret_cw", "mkt_ret_ew", "kodex200_ret"]]
M.columns = ["IDX", "CW", "EW", "KODEX"]
rowsA, rowsB, rowsC = {}, {}, {}
for y, d in list(M.groupby(M.index.year)) + [("전체", M)]:
    a = {c: annstats(d[c]) for c in M.columns}
    rowsA[y] = {"일수": a["IDX"][0], **{c: a[c][1] for c in M.columns}, "CW-IDX": a["CW"][1] - a["IDX"][1],
                "EW-IDX": a["EW"][1] - a["IDX"][1], "KODEX-IDX": a["KODEX"][1] - a["IDX"][1]}
    rowsB[y] = {**{c: a[c][2] for c in M.columns}, "CW-IDX": a["CW"][2] - a["IDX"][2], "EW-IDX": a["EW"][2] - a["IDX"][2],
                "KODEX-IDX": a["KODEX"][2] - a["IDX"][2]}
    rowsC[y] = {"CW~IDX": d.CW.corr(d.IDX), "EW~IDX": d.EW.corr(d.IDX), "KODEX~IDX": d.KODEX.corr(d.IDX),
                "CW~KODEX": d.CW.corr(d.KODEX), "CW~EW": d.CW.corr(d.EW)}
tabA, tabB, tabC = (pd.DataFrame(x).T for x in (rowsA, rowsB, rowsC))

# 부분허용 구간 공매도 가능 종목 수(근사) / 변형 간 불일치
pp = pm[(pm.date >= PART_START) & (pm.date <= PART_END)]
cp = pp.groupby("date")[["v1", "v2", "v3"]].sum()
anyv = pp[["v1", "v2", "v3"]].any(axis=1)
allv = pp[["v1", "v2", "v3"]].all(axis=1)
disagree = 1 - allv.sum() / anyv.sum()
nflag = pp[["v1", "v2", "v3"]].sum(axis=1)
mixed_share = {k: float((nflag == k).sum() / anyv.sum()) for k in (1, 2, 3)}
part_in = rv[(rv.effective_date >= PART_START) & (rv.effective_date <= PART_END)]

# 사니티 체크: 상위 N(유동시총) 가중 포트폴리오의 지수 추적 - 집중도(대형주 편중) 점검
nscan = {}
for n in (50, 100, 150, 200, 250, 300):
    r_n = wret(pm[pm.rank_fcap <= n], "fcap_prev").loc[MASK_START:]
    nscan[n] = stats(r_n.reindex(S.index), S.IDX)
nscan = pd.DataFrame(nscan).T
# 시총 점유율: 근사 구성(V1)이 패널(KOSPI 보통주) 전체 상장시총에서 차지하는 비중 vs FSC 2021-02-03 서술(88%)
sh = pm.assign(m=pm.lcap).groupby("date").apply(lambda d: d.loc[d.v1, "m"].sum() / d.m.sum(), include_groups=False)
share_by_year = sh.groupby(sh.index.year).mean() * 100
d_fsc = sh.index[sh.index >= "2021-02-03"][0]
share_fsc = sh.loc[d_fsc] * 100

# 2026-09-14 부터 근사가 지수에서 크게 이탈한다(월별 추적오차 3~5bp -> 50~60bp). 이탈 시작일은 결과를 보고 정한 사후 구분이다.
BREAK = pd.Timestamp("2026-09-14")
pre, post = S.loc[:BREAK - pd.Timedelta(days=1)], S.loc[BREAK:]
tot_pre = pd.DataFrame({c: stats(pre[c], pre.IDX) for c in cols}).T
tot_post = pd.DataFrame({c: stats(post[c], post.IDX) for c in cols}).T
mte = (S.V1 - S.IDX)
mte = (mte.groupby([mte.index.year, mte.index.month]).std() * 1e4).loc[2026]
mte.index = [f"{m}월" for m in mte.index]

# 암시 비중: 지수 일수익률을 (삼성전자, SK하이닉스, V1 나머지 가중평균)에 무절편 회귀한 계수 = 지수가 실제로 둔 비중의 추정치.
# KODEX 200/지수의 대형주 편중이 우리 V1 유동시총 비중과 맞는지 보는 점검(DB 에 실제 구성 비중이 없다)
vv = pm[pm.v1].copy()
vv["w"] = vv.fcap_prev / vv.groupby("date").fcap_prev.transform("sum")
_s = vv[vv.ticker == "005930"].set_index("date")
_h = vv[vv.ticker == "000660"].set_index("date")
_all = (vv.w * vv.ret).groupby(vv.date).sum()
_rest = (_all - (_s.w * _s.ret).reindex(_all.index).fillna(0) - (_h.w * _h.ret).reindex(_all.index).fillna(0)) / (
    1 - _s.w.reindex(_all.index).fillna(0) - _h.w.reindex(_all.index).fillna(0))
XR = pd.DataFrame({"s": _s.ret, "h": _h.ret, "r": _rest, "idx": S.IDX}).dropna()
imp = {}
for lab, a, b in [("부분허용 구간 2021-05-03~2023-11-03", "2021-05-03", "2023-11-03"), ("2025", "2025-01-01", "2025-12-31"),
                  ("2026-01~08", "2026-01-01", "2026-08-31"), ("2026-09-14~10-08 (이탈 구간)", "2026-09-14", "2026-10-08")]:
    d = XR.loc[a:b]
    A, y = d[["s", "h", "r"]].values, d.idx.values
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * ((y - A @ coef) ** 2).sum() / (len(y) - 3))
    imp[lab] = {"일수": len(d), "삼성전자 암시": coef[0] * 100, "삼성전자 ±": se[0] * 100, "삼성전자 V1평균": _s.w.loc[a:b].mean() * 100,
                "하이닉스 암시": coef[1] * 100, "하이닉스 ±": se[1] * 100, "하이닉스 V1평균": _h.w.loc[a:b].mean() * 100,
                "나머지 암시": coef[2] * 100}
imp = pd.DataFrame(imp).T

# 유동비율 결측·대체 현황(연도별)
_pp = pm.assign(year=pm.date.dt.year)
fl_diag = pd.DataFrame({
    "유동비율 결측으로 상장주식수 대체한 종목-일(%)": _pp.groupby("year").fl_sub.mean() * 100,
    "유동비율 중앙값": _pp[~_pp.fl_sub].groupby("year").fl_ratio.median(),
    "V1 구성 종목-일 중 대체(%)": _pp[_pp.v1].groupby("year").fl_sub.mean() * 100})

# 신규상장 특례 후보(기사 수준 규칙: 상장 후 15매매일 시총 50위 이내면 정기변경 전 특례편입): 규칙 V1~V3 이 못 잡는 구성
p["lcap_rank"] = p.groupby("date").lcap.rank(ascending=False)
first = p.groupby("ticker").date.min()
names = pd.read_parquet(f"{CB}/stock_master.parquet").set_index("ticker").name
rk = p.set_index(["ticker", "date"]).lcap_rank
nl = []
for t, d0 in first[first > cal[20]].items():          # 패널 시작 직후(좌측 절단)는 제외
    i = cal.get_loc(d0)
    if i + 14 < len(cal):
        nl.append(dict(ticker=t, name=names.get(t, ""), first_date=d0, d15=cal[i + 14], rank15=rk.get((t, cal[i + 14]), np.nan)))
nl = pd.DataFrame(nl)
nl_top = nl[nl.rank15 <= 50].copy()
nl_top["next_review"] = [rv.effective_date[rv.effective_date > d].min() for d in nl_top.d15]
nl_top["in_v1_at_next"] = [bool(mem[(mem.rb == rv.index[rv.effective_date == nr][0]) & (mem.ticker == t)].v1.any()) if pd.notna(nr) else False
                           for t, nr in zip(nl_top.ticker, nl_top.next_review)]


# ---------------------------------------------------------------- 7. 노트 (표·수치는 위 계산값을 그대로 넣는다)
TODAY = "2026-10-11"
SECOND = ["V1", "V2", "V3"]
HDR = {"n": "일수", "corr": "상관", "te_bp": "추적오차(bp/일)", "te_ann": "추적오차(연%)", "mdiff_bp": "평균차(bp/일)",
       "cumdiff": "누적수익률차(%p)"}
c1 = {"일수": "{:.0f}", "상관": "{:.4f}", "추적오차(bp/일)": "{:.1f}", "추적오차(연%)": "{:.2f}", "평균차(bp/일)": "{:+.2f}",
      "누적수익률차(%p)": "{:+.1f}"}


def ystr(d):
    d = d.copy()
    d.index = d.index.astype(str)
    d.index.name = "연도"
    return d


rv_show = rv.assign(effective_date=rv.effective_date.dt.strftime("%Y-%m-%d"), ref_date=rv.ref_date.dt.strftime("%Y-%m-%d"))[
    ["effective_date", "ref_date", "n_eligible", "v1_in", "v1_out", "v2_in", "v2_out", "v3_in", "v3_out", "v1_v2_overlap",
     "v1_v3_overlap", "n_fl_sub_v1", "n_near_cutoff_20pct", "v1_delisted"]]
rv_show.columns = ["효력일(가정)", "기준일", "후보", "V1 편입", "V1 편출", "V2 편입", "V2 편출", "V3 편입", "V3 편출", "V1∩V2",
                   "V1∩V3", "V1 중 유동주식 대체", "컷오프±20%", "V1 구간중 상폐"]
pw = rv[(rv.effective_date >= PART_START) & (rv.effective_date <= PART_END)]
pn = pd.DataFrame({"평균": cp.mean(), "최소": cp.min(), "최대": cp.max(), "시작일(2021-05-03)": cp.iloc[0],
                   "종료일(2023-11-03)": cp.iloc[-1]}).astype(float)
pn.index = SECOND
nlw = nl_top[nl_top.first_date <= PART_END].copy()
nlw = nlw[nlw.d15 >= PART_START]
nl_show = pd.DataFrame({"종목코드": nlw.ticker, "종목명": nlw.name, "상장일": nlw.first_date.dt.strftime("%Y-%m-%d"),
                        "15거래일 시총순위": nlw.rank15, "다음 정기변경": nlw.next_review.dt.strftime("%Y-%m-%d"),
                        "다음 변경에서 V1 편입": nlw.in_v1_at_next.map({True: "예", False: "아니오"})})
n_special = len(nlw)
n_special_miss = int((~nlw.in_v1_at_next).sum())
sy = share_by_year.round(1).astype(float).to_frame("V1 시총 점유율(%)")
sy.index = sy.index.astype(str)
sy.index.name = "연도"
b = tot.loc[SECOND, "te_bp"]
best = b.idxmin()
mean_in = pw[["v1_in", "v1_out"]].mean()
mean_ov12, mean_ov13, mean_near = pw.v1_v2_overlap.mean(), pw.v1_v3_overlap.mean(), pw.n_near_cutoff_20pct.mean()
n3, f3c = int(f3.sum()), f3_idx_closer
flags = mk.kospi200_idx_flag.value_counts().sort_index().to_dict()
nsc = nscan.rename(columns=HDR)
nsc.index = nsc.index.astype(str)
nsc.index.name = "N"

T_A = to_md(ystr(tabA), {**{c: "{:.1f}" for c in tabA.columns}, "일수": "{:.0f}"})
T_B = to_md(ystr(tabB), {c: "{:.1f}" for c in tabB.columns})
T_C = to_md(ystr(tabC), {c: "{:.3f}" for c in tabC.columns})
T_TOT = to_md(tot.rename(columns=HDR).rename_axis("대상"), c1)
T_PART = to_md(tot_p.rename(columns=HDR).rename_axis("대상"), c1)
T_TE = to_md(ystr(te_y), {c: "{:.1f}" for c in te_y.columns})
T_CO = to_md(ystr(co_y), {c: "{:.4f}" for c in co_y.columns})
T_CD = to_md(ystr(cd_y), {c: "{:+.1f}" for c in cd_y.columns})
T_NS = to_md(nsc, c1)
T_PRE = to_md(tot_pre.rename(columns=HDR).rename_axis("대상"), c1)
T_POST = to_md(tot_post.rename(columns=HDR).rename_axis("대상"), c1)
T_IMP = to_md(imp.rename_axis("기간"), {c: "{:.1f}" for c in imp.columns if c != "일수"} | {"일수": "{:.0f}"})
T_MTE = to_md(mte.round(1).to_frame("V1 월별 추적오차(bp/일)").rename_axis("2026년 월"), {"V1 월별 추적오차(bp/일)": "{:.1f}"})
T_WORST = to_md(worst, {c: "{:.2f}" for c in worst.columns})
T_SY = to_md(sy, {"V1 시총 점유율(%)": "{:.1f}"})
T_RV = to_md(rv_show.rename_axis("회차"), {c: "{:.0f}" for c in rv_show.columns})
T_PN = to_md(pn.rename_axis("변형"), {"평균": "{:.1f}", **{c: "{:.0f}" for c in pn.columns if c != "평균"}})
T_NL = to_md(nl_show, {"15거래일 시총순위": "{:.0f}"}, index=False)
fd = fl_diag.copy()
fd.index = fd.index.astype(str)
fd.index.name = "연도"
T_FL = to_md(fd, {"유동비율 결측으로 상장주식수 대체한 종목-일(%)": "{:.1f}", "유동비율 중앙값": "{:.3f}", "V1 구성 종목-일 중 대체(%)": "{:.1f}"})

NOTE = f"""---
title: KOSPI200 구성종목 근사와 시장수익률 대용 시계열 (d04)
created: 2026-10-11
updated: {TODAY}
tags: [dev/python, finance/퀀트]
source: 동아리 DB(dataguide.daily_shares 유동주식수, kiwoom.etf_daily A069500) + 패널 panel.parquet; FSC 보도자료(2021-02-03, 2021-04-29)는 [[short_selling_notes]] 에서 인용; 코드 code/d04_kospi200_approx.py
status: seed
---

# KOSPI200 구성종목 근사와 시장수익률 대용 시계열

> 공매도 부분허용 구간(2021-05-03~2023-11-03)에는 KOSPI200 구성종목 이력이 필요한데 구하지 못했다(KRX 약관상 자동 수집 금지라 KRX 사이트와 pykrx 는 쓰지 않았다). 이미 가진 자료로 **근사**했고, 근사가 얼마나 틀리는지를 숫자로 적는다. 이 마스크는 실제 구성종목이 아니다. 쓸 때마다 "근사"라고 표시한다. 관련: [[short_selling_notes]], [[market_rules_notes]].

## 0. 결론 요약

- 산출물: `market_ret_daily.parquet`, `kospi200_approx_mask.parquet`, `kospi200_review_dates.csv`, 이 노트. 코드는 `code/d04_kospi200_approx.py`.
- 근사 구성(구성 종목 안 전일 유동시총 가중)이 KOSPI200 지수 일수익률을 따라가는 정도, 상관 / 일간 추적오차 표준편차(bp):
  - 부분허용 구간 2021-05-03~2023-11-03({int(tot_p.n.iloc[0])}일): V1 {tot_p.loc['V1','corr']:.4f} / {tot_p.loc['V1','te_bp']:.1f}, V2 {tot_p.loc['V2','corr']:.4f} / {tot_p.loc['V2','te_bp']:.1f}, V3 {tot_p.loc['V3','corr']:.4f} / {tot_p.loc['V3','te_bp']:.1f} (연환산 {tot_p.loc['V1','te_ann']:.2f}% / {tot_p.loc['V2','te_ann']:.2f}% / {tot_p.loc['V3','te_ann']:.2f}%).
  - 이탈 이전 전 기간 2015-01-02~2026-09-11({int(tot_pre.n.iloc[0])}일): V1 {tot_pre.loc['V1','corr']:.4f} / {tot_pre.loc['V1','te_bp']:.1f}, V2 {tot_pre.loc['V2','corr']:.4f} / {tot_pre.loc['V2','te_bp']:.1f}, V3 {tot_pre.loc['V3','corr']:.4f} / {tot_pre.loc['V3','te_bp']:.1f}.
  - 2026-09-14 이후 이탈 구간을 포함한 전 기간({int(tot.n.iloc[0])}일): V1 {tot.loc['V1','corr']:.4f} / {tot.loc['V1','te_bp']:.1f}, V2 {tot.loc['V2','corr']:.4f} / {tot.loc['V2','te_bp']:.1f}, V3 {tot.loc['V3','corr']:.4f} / {tot.loc['V3','te_bp']:.1f}. 이탈 구간({int(tot_post.n.iloc[0])}일)만 보면 V1 {tot_post.loc['V1','corr']:.3f} / {tot_post.loc['V1','te_bp']:.0f}. 원인은 확인하지 못했다(5절).
  - 비교: KODEX 200 수정종가 수익률은 전 기간 {tot.loc['KODEX200','te_bp']:.1f}bp, 전체 KOSPI 보통주 시총가중은 {tot.loc['전체종목(시총)','te_bp']:.1f}bp.
- 부분허용 구간의 공매도 가능 종목 수(KOSPI200 몫만, 코스닥150 은 패널에 없음): V1 평균 {pn.loc['V1','평균']:.1f}개(최소 {pn.loc['V1','최소']:.0f}~최대 {pn.loc['V1','최대']:.0f}), V2 평균 {pn.loc['V2','평균']:.1f}개, V3 평균 {pn.loc['V3','평균']:.1f}개. 상한은 {TOP_N}이고 변경 사이 상폐를 보충하지 않아 그 아래로 내려간다.
- 오분류 위험: 이 구간 정기변경 {len(pw)}회 동안 하나라도 켜진 종목-일 가운데 세 변형이 모두 켠 날은 {mixed_share[3] * 100:.1f}%이고 나머지 {(1 - mixed_share[3]) * 100:.1f}%는 변형끼리 갈린다. V1 과 V2 가 겹치는 종목은 평균 {mean_ov12:.1f}/{TOP_N}(V3 는 5% 컷 때문에 구조상 V1 과 {mean_ov13:.0f}개 겹쳐 정보가 없다). 컷오프({TOP_N}위) 유동시총의 ±20% 안에 드는 후보가 변경마다 평균 {mean_near:.0f}개다.
- 사용: 경계 종목은 믿지 말고, 변형 3개 결과를 같이 돌려 결론이 유지되는지 보는 강건성 점검용으로 쓴다(7절).

## 1. 시장수익률 대용 시계열 (`market_ret_daily.parquet`)

| 열 | 정의 |
|---|---|
| date | 패널 거래일(2010-01-04~2026-10-08) |
| mkt_ret_cw | 패널(KOSPI 보통주) 전 종목 전일 상장시총 가중 일수익률. 종목 수익률은 `ret1d_clean`(수정주가 이상값·거래정지일 제외) |
| mkt_ret_ew | 같은 종목들의 동일가중 평균 |
| kodex200_ret | KODEX 200(A069500) `adj_close` 일수익률(분배금 재투자 조정) |
| kospi200_idx_ret | KODEX 200 의 `idx_close`(기초지수 KOSPI200 가격지수) 일수익률 |
| kospi200_idx_flag | 0 정상 / 1 결측(그날 또는 전날 지수 없음, 0 이하 값 포함) / 2 지수가 전일과 같은데 KODEX 는 0.5% 넘게 움직임(정체)과 그 다음 날(여러 날치 수익률) / 3 지수와 KODEX 수익률 차이가 1%p 넘음(어느 쪽이 틀렸는지 미판정) |
| n_stocks, kodex200_missing | 수익률 평균에 들어간 종목 수 / KODEX 행 없음 |

**구하지 못한 것**: KOSPI 지수 자체, 금리, VKOSPI 는 이 DB 에 없다. 그래서 위 네 개가 시장수익률 후보의 전부다. 패널 시총가중은 KOSPI 보통주만 담고 우선주가 없다. 패널 `adj_close` 는 분할·증자만 조정하고 현금배당은 조정하지 않는다(삼성전자 2023~24 배당일에 수정계수가 변하지 않음을 확인). 그래서 가격지수 `idx_close` 와 같은 기준이고, KODEX `adj_close` 는 분배금 조정이 들어간 총수익 쪽이다.

표본은 지수 플래그가 0 또는 3이고 KODEX 가 있는 날이다. 플래그 집계: 정상 {flags.get(0, 0)}일, 결측 {flags.get(1, 0)}일, 정체 {flags.get(2, 0)}일, 불일치 {flags.get(3, 0)}일(전체 {len(mk)}일). 플래그 3은 {n3}일이고, 그중 {f3c}일은 전체 시총가중 수익률이 KODEX 보다 지수 쪽에 가까워서 KODEX `adj_close` 쪽 이상으로 보인다(분배금 조정 오류 추정, 원인은 확인하지 못함). 그래서 KODEX `adj_close` 를 총수익 대용으로 쓰면 이상값일(예: 2016-12-26 -5.3%, 2019-06-12 -4.9%, 2021-11-12 -3.5%, 2024-04-18 -3.1%) 때문에 해당 연도 수익률이 지수보다 낮게 나온다(표 1-A 2016·2019·2021·2024). 표 1 의 지수 자체(`idx_close`)에는 2012-06-20 의 0 값, 2012-10 의 정체 구간이 있어 결측·정체로 표시했다.

### 표 1-A 연환산 수익률(%) (연도별, 복리 후 252일 연환산)

{T_A}

### 표 1-B 연환산 변동성(%)

{T_B}

### 표 1-C 일수익률 상관

{T_C}

읽는 법: KODEX 와 패널 시총가중은 지수와 상관이 높고 변동성도 비슷하다. 동일가중은 상관이 낮고 수익률 수준도 크게 다르다(소형주 편중). 시장수익률 대용으로는 시총가중이나 KODEX 200 을 쓰고, 동일가중은 별도 팩터로 취급한다. 시총가중과 지수의 수익률 격차가 큰 해는 지수가 대형주 쪽으로 더 쏠렸기 때문일 수 있지만 원인 분해는 하지 않았다.

## 2. 근사 규칙 (실행 전에 고정)

1. 정기변경 효력일은 6월·12월 **둘째 목요일 다음 거래일**이다(선물옵션 동시만기 다음 거래일이라는 알려진 규칙의 근사). 2014-12 ~ 2026-06 총 {len(rv)}회. 코드가 기사 수준으로 확인된 5개 일자(2021-12-10, 2022-12-09, 2023-06-09, 2025-12-12, 2026-06-12)와 규칙 결과가 같은지 실행마다 검사한다.
2. 선정 기준일은 변경 직전 **4월말(6월 변경)·10월말(12월 변경)의 마지막 거래일**이고 평균 구간은 그 날까지 달력 6개월(11/1~4/30, 5/1~10/31)이다. FSC 2021-02-03 보도자료의 '4월·10월말 기준' 서술을 따랐다. 효력일과 기준일 사이가 6~7주라 "직전 6개월"보다 한 달 반 오래된 정보다.
3. 후보는 패널 종목(KOSPI 보통주, 우선주 없음) 중 구간 내 유효일이 {MIN_DAYS}일 이상이고 효력일에 패널 행이 있는 종목이다. 효력일 전에 상폐가 확정된 종목은 빠진다(발표 시점에 알 수 있는 정보라 보고 허용).
4. 변형 3개:
   - **V1** 구간 평균 유동시총(유동비율×상장주식수×종가) 상위 {TOP_N}. 유동비율은 `floating_shares`/`listed_shares` 를 1·4·7·10월말 거래일 {len(SNAP)}개 날짜의 스냅샷으로만 DB 에서 받아 다음 스냅샷까지 이월했다(일별 전수 조회는 서버가 너무 느려 포기, 3절). 유동비율이 없으면 상장주식수로 대체(비율 1)하고, 구간의 절반 넘는 날이 대체된 종목은 `float_substituted` 열에 표시.
   - **V2** 구간 평균 상장시총(`listed_shares`×종가) 상위 {TOP_N}.
   - **V3** V1 상위 {TOP_N} 에서 평균 거래대금(`tv_filled`)이 그 {TOP_N}개의 5% 분위수 미만인 종목을 빼고, 유동시총 다음 순위 중 같은 분위수 이상인 종목으로 {TOP_N}개를 채움.
5. 변경 사이에 상폐(패널 행 소멸)되면 그 날부터 마스크에서 빠지고 다음 변경까지 보충하지 않는다.
6. 변형 3개와 위 수치({MIN_DAYS}일, 5%, {TOP_N})는 결과를 보기 전에 정했고 이후 바꾸지 않았다. 실행 중 바뀐 것은 둘이다. (1) 유동비율을 일별이 아니라 분기 스냅샷으로 조회(서버 사정, 변형별 결과를 보기 전에 결정). (2) KODEX/지수의 0 값·정체 구간 처리와 진단 표 추가(데이터 정리와 설명 보강이며 변형·규칙은 그대로). 최고 변형을 사후에 고르는 셈이라 변형 전부를 표로 공개한다.

## 3. 가정 (확인하지 못한 것)

- **효력일**: 정확한 일자는 확인하지 못했다. 위 규칙은 기사 수준 5개 일자와 같다. 한편 2021-06('다음달 10일')과 2022-06('6월 9일') 기사 표기는 규칙 결과보다 하루 이르다(기준일 종가 표기와 효력일 표기가 섞인 것으로 보인다). 어느 쪽이든 오차는 1거래일이고 가중 수익률 검증에는 영향이 작다. 공매도 가능 여부가 발표일에 바뀌는지 효력일에 바뀌는지는 공식 문서를 못 찾았다([[short_selling_notes]] 미확인 항목).
- **선정 기준일·평균 구간**: 4월·10월말 기준과 6개월 평균은 보도자료 서술과 사용자 지정을 합친 것이고 거래소 지수 방법론 원문은 보지 못했다.
- **유동주식수(유동비율)**: DB 의 `floating_shares` 를 상장주식수로 나눈 유동비율을 분기 스냅샷({len(SNAP)}개 날짜)으로만 받았고 일별 값은 다음 스냅샷까지 이월한 값이다. 사용자 지정(일별 유동주식수×종가의 6개월 평균)과 다르며, 서버가 공유 자원인 데다 한창 느려서(종목당 수십 초, 접속 거부 반복) 일별 전수 조회를 포기한 대가다. 유동비율은 천천히 변한다고 보고 허용했지만 분기 사이 변화는 못 잡는다. 액면분할·증자는 일별 상장주식수가 바로 반영하므로 영향이 없다. 비율이 1을 넘으면 1로 자르고, 비율이 없으면 상장주식수로 대체한다(대체 비중은 5절 끝 표). KRX 의 유동비율 산정과 같은지는 모른다. DB 공백(2026-06-23~)은 마지막 스냅샷(2026-06-22)을 이월했다(선정에는 영향 없음, 마지막 정기변경이 2026-06).
- **KOSDAQ150**: 패널이 KOSPI 만이라 포함하지 않는다.
- 우선주는 패널에 없다. KOSPI200 이 우선주를 제외하는지는 확인하지 못했다.

## 4. 완전히 맞출 수 없는 이유

| 이유 | 확인 수준 | 이 근사에서 |
|---|---|---|
| 산업군별 대표성 심사: 산업군(기사 수준 9~11개)별로 누적시총 약 85%, 거래대금 순위 조건 등으로 종목을 고름(FSC 2021-02-03: 11개 산업군, 누적시총 85%, 거래대금 순위 85% 이내) | 방법론 원문 미확인 | 전체 시총 순위 단일 컷. 산업군 쏠림은 반영 못 함 |
| 시총 상위 50위 특례(산업군 비중 무관 편입) | 기사 수준 | 대형주는 어차피 상위라 영향 작음 |
| 신규상장 특례: 상장 후 15거래일 시총 50위 이내면 정기변경 전에 편입 | 기사 수준 | 못 잡음(6절 표 5). 게다가 '유효일 {MIN_DAYS}일 이상' 후보 조건 때문에 상장 3개월 미만 대형주(카카오뱅크·크래프톤 등)는 첫 정기변경에서도 빠진다. 사전 고정 규칙의 약점이고 결과를 본 뒤에도 바꾸지 않았다 |
| 수시변경(상폐·합병 시 즉시 보충, 카카오뱅크 2021-09·카카오페이 2021-11 사례는 [[short_selling_notes]]) | 알려진 제도 | 상폐 후 보충 없음(요청 사항) |
| 기존 종목 우대(편출 완충) 존재 여부 | 방법론 원문 미확인 | 완충 없이 매번 순위로 다시 뽑아 편입·편출이 실제보다 많을 가능성. V1 은 부분허용 구간에서 회당 평균 편입 {mean_in['v1_in']:.1f}, 편출 {mean_in['v1_out']:.1f}개(웹 검색 결과 기사 제목 몇 건에서 본 인상으로는 실제 변경이 회당 한 자릿수 종목이며 확인된 통계는 아님) |
| 관리종목·정리매매·투자주의환기, 스팩·리츠·인프라펀드 등 제외 규정 | 규정 미확인 | 이름 기반 제외 안 함(사후 튜닝 방지). 패널의 KOSPI 보통주 전부가 후보 |
| 유동비율 산정·갱신 시점 차이 | 미확인 | DB 값(분기 스냅샷 이월) 그대로 |
| 30% 비중상한제(삼성전자): 2019 도입, 2020 폐지 보도(적용된 적 없음) | 기사 수준 | 영향 없다고 보고 가중치에 상한 없음 |

## 5. 검증

검증용 수익률은 구성 종목 안에서 **전일 유동시총 가중**으로 만들고, 같은 날 지수 수익률(`kospi200_idx_ret`, 플래그 0·3)과 비교한다. 기간 2015-01-02 ~ 2026-10-08, 겹치는 {int(tot.n.iloc[0])}일. 추적오차는 (근사 − 지수) 일수익률의 표준편차다(bp = 0.01%p). 평균차와 누적수익률 차이는 근사에서 지수를 뺀 값이다. 비교 행으로 전체 종목 시총가중·동일가중과 KODEX 200 수정종가도 같이 둔다.

### 표 2 전 기간

{T_TOT}

### 표 2-B 이탈 이전 (2015-01-02 ~ 2026-09-11)

{T_PRE}

### 표 2-C 이탈 이후 (2026-09-14 ~ 2026-10-08)

{T_POST}

**2026-09-14 부터 근사가 지수에서 크게 벗어난다.** 월별 V1 추적오차(bp/일):

{T_MTE}

삼성전자·SK하이닉스 두 종목이 V1 의 절반 넘는 비중을 차지하는 시기에 일어난 일이다. 같은 기간 KODEX 200 의 종가/지수 비율이 2026-09-29 에 100.5배에서 101.1배로 계단식으로 바뀌고 추적오차율(`trace_eor_rt`)도 0.38 에서 0.70 으로 바뀐다. 원인은 확인하지 못했다. 지수 산출 방법 변경(비중 상한 등)인지, 유동비율·주식수 변화를 분기 스냅샷이 못 따라간 것인지, 데이터 문제인지 구분할 자료가 없다(웹 검색에서도 2026-09 지수 방법 변경 공지는 찾지 못했다). 이탈 시작일(2026-09-14)은 월별 오차를 보고 정한 사후 구분이다. 부분허용 구간(2021~2023)과는 무관하지만, 2026-09-14 이후를 쓰는 분석에서는 이 마스크로 지수를 대용하지 말 것.

### 표 3 부분허용 구간만 (2021-05-03 ~ 2023-11-03)

{T_PART}

### 표 3-A 연도별 일간 추적오차 (bp/일)

{T_TE}

연도별 일수: {({int(k): int(v) for k, v in n_y.items()})}

### 표 3-B 연도별 상관

{T_CO}

### 표 3-C 연도별 누적수익률 차이(근사 − 지수, %p)

{T_CD}

주의: 최고 변형을 사후에 보는 비교다. 전 기간 추적오차가 가장 낮은 변형은 {best}이지만 세 변형의 차이(V1 {b['V1']:.1f} / V2 {b['V2']:.1f} / V3 {b['V3']:.1f}bp)는 연도별 변동에 견주면 작다. 추적오차는 시장 변동성에 비례해 커지므로(2025~26 급등락) 연도 간 비교는 수준보다 변형 간 격차로 읽는다. 누적수익률 차이는 일간 오차가 쌓인 것이라 격차가 벌어진 구간의 영향이 크다.

### 표 3-D V1 이 지수에서 가장 크게 벗어난 10일 (수익률 %, 괴리는 bp)

{T_WORST}

### 사니티 체크: 근사의 대형주 편중이 KODEX 200/지수와 맞는가

KODEX 200 의 실제 보유종목·시총 분포는 DB 에 없다. 그래서 (a) 지수 수익률을 가장 잘 따라가는 상위 종목 수를 보고, (b) FSC 가 제시한 시총 점유율과 비교했다.

**(a) 암시 비중**: 지수 일수익률을 삼성전자 수익률, SK하이닉스 수익률, V1 나머지 종목의 가중평균 수익률에 무절편 회귀한 계수는 지수가 실제로 둔 비중의 추정치다. 우리 V1 의 전일 유동시총 비중 평균과 비교(%, ± 는 표준오차):

{T_IMP}

부분허용 구간에서 삼성전자 암시 비중 {imp.iloc[0,1]:.1f}% 대 V1 평균 {imp.iloc[0,3]:.1f}%, SK하이닉스 {imp.iloc[0,4]:.1f}% 대 {imp.iloc[0,6]:.1f}% 로 거의 같다. 2025·2026-01~08 도 몇 %p 안쪽으로 맞고(2026 년 두 종목 합 V1 평균 {imp.iloc[2,3] + imp.iloc[2,6]:.0f}%), 이탈 구간(표본 {int(imp.iloc[3,0])}일이라 표준오차가 큼)에서만 어긋난다. 대형주 편중의 크기는 지수와 일관된다.

**(b)** 변경마다 유동시총 상위 N종목(후보 순위 기준)을 뽑아 전일 유동시총 가중 수익률을 지수와 비교:

{T_NS}

추적오차는 N=50 {nscan.te_bp[50]:.1f}, 100 {nscan.te_bp[100]:.1f}, 150 {nscan.te_bp[150]:.1f}, 200 {nscan.te_bp[200]:.1f}, 250 {nscan.te_bp[250]:.1f}, 300 {nscan.te_bp[300]:.1f}bp 다. 최저는 N={int(nscan.te_bp.idxmin())}이고 150~200 은 사실상 같다. 50·100 은 뚜렷이 나쁘고 250 이상은 완만하게 나빠지므로, 지수가 담은 종목 규모는 200 근처와 일관된다. 다만 150 과 200 은 구분하지 못한다(식별력이 낮은 점검이다). 누적수익률 차이가 N 이 작을수록 크게 플러스인 것은 최근 상위 몇 종목의 급등 때문이다.

**(c)** V1 구성이 패널 전체 상장시총(KOSPI 보통주만)에서 차지하는 비중은 {d_fsc.date()}에 {share_fsc:.1f}%로, FSC 2021-02-03 서술(KOSPI200 이 전체 시총의 88%, 종목 수의 22%)과 비교된다. 분모에 우선주가 없어 우리 값이 높게 나오는 방향이고, 두 분모를 맞출 수 없어 '대략 같은 수준'으로만 본다. 연도 평균:

{T_SY}

### 유동비율 결측·대체 현황 (2015~)

{T_FL}

### 표 4 정기변경별 구성 변화 (`kospi200_review_dates.csv` 요약)

{T_RV}

- 후보는 각 회차 점검 대상 종목 수. 편입·편출은 직전 회차 대비이고 첫 회차는 비교 대상이 없다.
- 컷오프±20%: {TOP_N}위 종목 평균 유동시총의 ±20% 안에 드는 후보 수. 이 안의 종목은 작은 오차로도 순위가 바뀐다.
- V1 중 유동주식 대체: 유동비율 결측으로 상장주식수를 쓴 V1 구성 종목 수.

## 6. 부분허용 구간(2021-05-03~2023-11-03)의 오분류 위험

### 공매도 가능 종목 수 (이 마스크 기준, 일별)

{T_PN}

KOSPI200 몫만 센 숫자다. 코스닥150(최대 150종목)은 패널에 없다. 일별 구성 수가 {TOP_N}보다 작은 날은 구성 종목이 상폐돼 빠졌기 때문이고, 회차마다 구간 중 상폐된 V1 종목 수는 표 4 마지막 열이다.

### 얼마나 틀릴 수 있나

- **구간 시작의 구성**: 2021-05-03 에 적용 중인 구성은 2020-12-11 효력 변경분이다(마스크 계산에 포함, 표 4 회차 12). 구간 안 정기변경은 2021-06 ~ 2023-06 5회.
- **변형 간 불일치**: 하나라도 켜진 종목-일 중 세 변형이 모두 켠 비율 {mixed_share[3] * 100:.1f}%, 둘이 켠 비율 {mixed_share[2] * 100:.1f}%, 한 변형만 켠 비율 {mixed_share[1] * 100:.1f}%. 근사 방식만 바꿔도 이만큼 갈린다. 실제 구성을 모르므로 이것은 오류율이 아니라 불확실성의 하한 감각이다.
- **경계 종목**: 변경마다 컷오프 ±20% 안에 평균 {mean_near:.0f}개 후보가 있다. 마스크의 `rank_fcap_v1`(직전 변경 시점 유동시총 순위)로 경계({TOP_N}위 부근)를 가려낼 수 있다.
- **과도한 교체**: 완충이 없는 규칙이라 V1 은 회당 평균 {mean_in['v1_in']:.1f}개를 갈아 끼운다. 실제 변경은 이보다 작은 것으로 보여(기사 제목 수준의 인상, 미확인) 경계 종목이 실제와 다르게 들락일 가능성이 크다.
- **신규상장 특례 누락**: 구간 안에서 상장 15거래일 시총이 상위 50위였던 종목 {n_special}개 중 다음 정기변경 때 V1 에 들어온 것은 {n_special - n_special_miss}개, 못 들어온 것은 {n_special_miss}개다. 이 종목들은 실제로는 특례편입으로 공매도가 가능했을 수 있는데 마스크는 다음 정기변경 때까지(또는 영영) 불가로 둔다. 분할상장·재상장도 섞여 있어 후보일 뿐 확정이 아니다.

### 표 5 구간 내 신규상장 특례 후보 (패널 첫 거래일 기준, 상장 후 15거래일 상장시총 순위 50위 이내)

{T_NL}

## 7. 사용 지침

1. **마스크는 근사다.** 공매도 가능 종목 마스크로 쓴 모든 결과에 "KOSPI200 구성종목 근사(V1~V3)"라고 적고, 변형을 정한 근거를 밝힌다. 사후에 성과가 좋은 변형을 골랐다면 그렇다고 쓴다.
2. 한 변형만 쓰지 말고 **V1·V2·V3 결과를 모두 돌려** 결론이 같은지 본다. 같은 결론이 안 나오면 부분허용 구간에 의존하는 결론이므로 약하게 쓴다.
3. 보수적으로 쓰려면 `in_kospi200_v1 & in_kospi200_v2 & in_kospi200_v3`(세 변형 일치)만 "공매도 가능", 셋 다 아닌 종목만 "금지"로 두고 나머지(불일치) 종목-일은 분석에서 빼거나 따로 처리한다. 또는 `rank_fcap_v1` 이 {TOP_N - 30}위 이내인 종목만 가능, {TOP_N + 50}위 밖만 금지로 두고 사이는 제외하는 방식을 쓴다.
4. 발표에서 효력일까지 2~3주는 가능 여부가 바뀌는 날이 불명확하니 변경일 ±10거래일은 별도 점검한다.
5. 코스닥150 은 포함돼 있지 않다. 패널이 KOSPI 만이라 지금은 문제가 없지만 코스닥으로 확장하면 따로 근사해야 한다.
6. 패널의 `short_banned` 열은 2021-05-03~2023-11-03 에 종목 단위가 아니다(패널 감사 B8). 이 구간의 종목별 가능 여부는 이 마스크(근사)로 대체하고, 구간 밖은 `short_selling_regime.csv` 의 체제(전면허용 2015-01-01~2020-03-15, 전면금지 2020-03-16~2021-05-02, 전면금지 2023-11-06~2025-03-30, 전면허용 2025-03-31~)를 그대로 쓴다.

## 8. 한계와 못 한 것

- 실제 구성종목 이력과 직접 대조하지 못했다(구하지 못함). 검증은 지수 **수익률**을 얼마나 따라가는가로 한 것이라 구성 일치도의 간접 증거다. 수익률 검증은 구성 오분류에 둔감하다: V1 과 V2 는 평균 {TOP_N - mean_ov12:.0f}개 종목이 다른데도 부분허용 구간 추적오차가 {tot_p.loc['V1','te_bp']:.1f}bp 대 {tot_p.loc['V2','te_bp']:.1f}bp 로 사실상 같다(경계 종목 가중치가 작다). 그래서 이 표들은 공매도 가능 종목 분류가 맞다는 증거가 아니다.
- 산업군 심사, 신규상장 특례, 수시변경, 편출 완충, 제외 규정(관리종목·스팩·리츠 등)은 모델링하지 않았다. 완충 규칙을 넣은 변형은 사전 고정 3개에 없어서 돌리지 않았다.
- 효력일, 선정 기준일, 유동주식수 정의는 가정이다(3절). 유동비율은 분기 스냅샷 이월이라 사용자 지정(일별)보다 거칠다.
- KOSPI 지수 자체, 금리, VKOSPI 는 DB 에 없어서 시장 대용은 시총가중, KODEX 200, KOSPI200 지수 세 가지뿐이다.
- KODEX 200 `adj_close` 에 지수와 불일치하는 날이 {n3}일 있어(분배금 조정 오류 추정, 원인 미확인) KODEX 쪽 통계는 그만큼 불리하게 나온다.
- 2026-09-14 이후 근사가 지수에서 이탈하는 원인은 확인하지 못했다(5절 표 2-C). 이 구간의 마스크·근사 수익률은 쓰지 말고, 분기 스냅샷 대신 일별 유동주식수를 받으면 달라지는지도 보지 못했다.
- 이 노트의 문장은 윤문(humanize) 단계를 거치지 않았다.

## 재현

```
python code/d04_kospi200_approx.py
```

DB 조회는 캐시(`data/daily/club_db/floating_shares.parquet`, `kodex200_daily.parquet`)가 없을 때만 하며, 종목 10개×스냅샷 날짜 IN 점조회를 한 번에 하나씩 보내고 접속이 거부되면 30초 간격으로 재시도한다. 일별 범위 조회는 종목당 수십 초(`code IN` 묶음 범위 조회는 25종목에 약 150초)로 느려서 쓰지 않았다. 조각은 `club_db/_floating_parts/` 에 남는다(재개용, 지워도 된다). KRX 사이트와 pykrx 는 쓰지 않는다.
"""
with open(f"{OUT}/kospi200_approx_notes.md", "w", encoding="utf-8") as f:
    f.write(NOTE)
print("notes written", len(NOTE), "best", best)

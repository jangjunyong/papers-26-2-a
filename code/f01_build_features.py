"""f01: 형성일 단위 도메인 변수 패널 (B1 수급 · B2 거래·가격 · B3 공시·재무) + 라벨.

기획안_v2_도메인결합_초안.md 3~5절의 정보 블록 3개를 (ticker, 형성일) 표로 만든다. 형성일 = 주(ISO 주)의 마지막 거래일
(a3_baselines.samples 와 같은 규칙). 유니버스 = f_tradable==1 & exclude_flag==0.
모든 변수는 형성일 종가까지 공표된 정보만 쓰고, 날짜별 횡단면 rank(pct=True)-0.5 열(_rk)을 같이 만든다.

실행:  py -3 f01_build_features.py        (입력 읽기 포함 약 20초, 최대 메모리 약 4.3GB)
출력:  data/features/features_v2_kospi.parquet, features_v2_kospi_columns.md (열 사전 + 입력 파일 해시)

[시험 구간 보호] LABEL_CAP 이후에 끝나는 라벨(label_end_date > 2016-12-31)은 저장 파일에서 NaN 이다(label_status=5).
변수(특징)는 전 기간이다. 시험 구간 라벨을 풀려면 이 상수를 바꾸는 diff 가 커밋에 남아야 한다 (사전 고정 후 한 번만).
설정(창 길이·최소 관측 수·발행공시 제외 등)은 아래 상수에 한 번 정해 둔 값이며 결과를 보고 바꾸지 않는다.
build() 는 입력 DataFrame 을 받는 순수 함수라 f01_verify_features.py 의 교란 실험이 같은 코드를 그대로 재실행한다.
"""
import hashlib
import pathlib
import sys
import time

import numpy as np
import pandas as pd

PAPERS = pathlib.Path(__file__).resolve().parents[3]
D = PAPERS / "data"
PANEL = D / "daily" / "club_db" / "panel_v2.parquet"
CLS = D / "dart" / "disclosures_kospi_class.parquet"
FSAV = D / "dart" / "fs_available_date.parquet"
FSCORE = D / "daily" / "club_db_ext" / "fs_line_core.parquet"
FSPRE = D / "daily" / "club_db_ext" / "fs_line_core_pre2019.parquet"   # f00_export_fs_pre2019.py: 2019 이전 'ifrs_' ID 4계정
OUT_DIR = D / "features"
OUT = OUT_DIR / "features_v2_kospi.parquet"
OUT_DOC = OUT_DIR / "features_v2_kospi_columns.md"

# ------------------------------------------------------------------ 사전 고정 상수
LABEL_CAP = pd.Timestamp("2016-12-31")      # 이 날짜 이후에 끝나는 라벨은 만들어 저장하지 않는다 (시험 구간 보호)
HOLD = 5                                     # 보유 거래일 수: 다음 거래일 시가 매수 → 5거래일째 종가
FLOW_MIN = {5: 4, 20: 15}                    # B1 창별 최소 유효 관측일 수 (창 안에서 flow 가 결측이 아닌 날)
TV_WIN, TV_MIN = 60, 30                      # 평균 거래대금: 직전 60거래일(당일 제외), 거래 가능한 날만, 최소 30일
AMI_WIN, AMI_MIN = 20, 15                    # Amihud: 직전 20거래일(당일 제외) 평균, 최소 15일
AMI_SCALE = 1e9                              # |ret| / 거래대금(원) x 1e9  (10억 원당)
DSC_WINS = (5, 20, 60)                       # 공시 건수 창 (거래일, next_td 기준 형성일 포함)
DSC_START = pd.Timestamp("2015-01-01")       # DART 수집 시작일: 이 전을 덮는 창은 NaN (0 이 아니다)
DSC_CATS = [("periodic", "정기"), ("major", "주요사항"), ("stake", "지분"), ("exch", "수시공정")]
DSC_EXCLUDE_CAT = "기타"                     # 발행공시(증권신고·투자설명서 등 33%)가 든 분류. 제외 버전을 쓴다 (정정도 이 분류 것은 제외)
FS_ACC = {"ifrs-full_Assets": "assets", "ifrs-full_Liabilities": "liab", "ifrs-full_Equity": "equity"}
FS_NI = "ifrs-full_ProfitLoss"               # 당기순이익(IS). 사업보고서(연간) 값만 쓴다
NEXT_DAY = np.timedelta64(1, "D")             # 재무: first_rcept_dt 다음 날부터 사용 (접수 시각이 없어 당일 불가)

PANEL_COLS = ["ticker", "date", "open", "close", "adj_close", "mktcap", "tv_filled", "ret1d_clean",
              "indiv_flow", "inst_flow", "foreign_flow", "str5", "str20", "mom12_1", "near52h", "vol20", "abvol",
              "f_tradable", "exclude_flag", "inv_sum_closed", "foreign_def_changed", "fwd_censored", "fwd_ret5"]

B1 = ["flow_indiv_5", "flow_indiv_20", "flow_inst_5", "flow_inst_20", "flow_for_5", "flow_for_20"]
B2_M0 = ["str5", "str20", "mom12_1", "near52h", "vol20", "abvol"]
B2_X = ["log_mktcap", "log_tv60", "amihud20"]
B3_DSC = [f"dsc_{c}_{w}" for c in ["periodic", "major", "stake", "exch", "amend"] for w in DSC_WINS]
B3_FIN = ["fin_size", "fin_lev", "fin_bm", "fin_roe"]
FEATURES = B1 + B2_M0 + B2_X + B3_DSC + B3_FIN
BLOCKS = {"B1 수급": B1, "B2 거래·가격(M0 6지표)": B2_M0, "B2 거래·가격(추가 3)": B2_X, "B3 공시": B3_DSC, "B3 재무": B3_FIN}
RK = [f + "_rk" for f in FEATURES]


# ------------------------------------------------------------------ 창 합계 도구
def win_sum(x, w, lag, tstart, tend, cc=None):
    """종목별 창 [i-lag-w+1, i-lag] (종목 경계 안으로 자름)의 유한값 합계·유효 개수·창 행 수.
    종목 안 누적합만 쓰므로 한 종목의 값이 다른 종목 결과에 섞이지 않는다 (교란 실험에서 비트 단위로 같다).
    lag 는 음수 가능(미래 창, 양성 대조용)."""
    n = len(x)
    v = np.isfinite(x)
    xs = np.where(v, x, 0.0)
    codes = cc
    cs = pd.Series(xs).groupby(codes).cumsum().to_numpy()
    cn = pd.Series(v.astype(np.int32)).groupby(codes).cumsum().to_numpy()
    idx = np.arange(n)
    b = idx - lag
    a = np.maximum(b - w + 1, tstart)
    b = np.minimum(b, tend)
    ok = (b >= tstart) & (b >= a)
    bs = np.where(ok, b, 0)
    prev = np.maximum(a - 1, 0)
    has_prev = ok & (a > tstart)
    s = cs[bs] - np.where(has_prev, cs[prev], 0.0)
    c = cn[bs] - np.where(has_prev, cn[prev], 0)
    nr = np.where(ok, b - a + 1, 0)
    return np.where(ok, s, np.nan), np.where(ok, c, 0), nr


# ------------------------------------------------------------------ B3 공시
def event_cumulative(cls, cal, tickers, last_date):
    """분류 표 → 종목 x 거래일 누적 건수 5종(정기·주요사항·지분·수시공정·정정). 발행공시('기타')는 정정 포함 제외.
    이벤트 날짜 = 접수일이 거래일이면 그날, 아니면 다음 거래일; next_td = 그 다음 거래일 (disclosure_events_daily 와 같은 규칙).
    next_td 날짜 열에 센다 → 형성일 t 의 창에는 next_td<=t 인 것만 들어간다. 상장폐지(마지막 행) 이후 접수분은 버린다."""
    c = cls[cls["cat"] != DSC_EXCLUDE_CAT]
    tk = pd.Series(np.arange(len(tickers)), index=tickers).reindex(c["ticker"]).to_numpy()
    ld = pd.Series(last_date).reindex(c["ticker"]).to_numpy()
    r = pd.to_datetime(c["rcept_dt"]).to_numpy()
    keep = ~np.isnan(tk) & (r <= ld)
    c, tk, r = c[keep], tk[keep].astype(np.int64), r[keep]
    d = np.searchsorted(cal, r, side="left")             # 접수일 이상인 첫 거래일 (이벤트 날짜)
    nxt = d + 1                                           # next_td
    keep = nxt < len(cal)
    c, tk, nxt = c[keep], tk[keep], nxt[keep]
    kind = np.full(len(c), -1, dtype=np.int64)
    for j, (_, name) in enumerate(DSC_CATS):
        kind[(c["cat"].to_numpy() == name) & ~c["is_amend"].to_numpy()] = j
    kind[c["is_amend"].to_numpy()] = len(DSC_CATS)       # 정정 (유형 무관, 발행공시 제외)
    out = {}
    for j, name in enumerate([n for n, _ in DSC_CATS] + ["amend"]):
        m = kind == j
        arr = np.zeros((len(tickers), len(cal)), dtype=np.int32)
        np.add.at(arr, (tk[m], nxt[m]), 1)
        cum = np.zeros((len(tickers), len(cal) + 1), dtype=np.int32)
        np.cumsum(arr, axis=1, out=cum[:, 1:])
        out[name] = cum
    return out


# ------------------------------------------------------------------ B3 재무
def fin_reports(fscore, fsav, tickers):
    """fs_line_core → 보고서 단위 표 (ticker, reprt_code, period_end): assets, liab, equity, ni, 연결 우선(없으면 별도), 접수일.
    조인 키는 (ticker, reprt_code, period_end). 접수일(first_rcept_dt)이 없는 보고서는 쓸 수 없어 버린다."""
    f = fscore[fscore["market"] == "KOSPI"]
    f = f[(f["sj_div"] == "BS") & f["account_id"].isin(FS_ACC) | (f["sj_div"] == "IS") & (f["account_id"] == FS_NI)]
    f = f.assign(ticker=f["code"].str[1:], item=f["account_id"].map({**FS_ACC, FS_NI: "ni"}))
    f = f[f["ticker"].isin(tickers)]
    w = f.set_index(["ticker", "reprt_code", "period_end", "fs_div", "item"])["thstrm_amount"]
    w = w[~w.index.duplicated(keep="first")].unstack("item").reset_index()
    for c in ("assets", "liab", "equity", "ni"):
        if c not in w:
            w[c] = np.nan
    w["pri"] = (w["fs_div"] == "OFS").astype(int)
    w["eq_derived"] = (w["equity"].isna() & w["assets"].notna() & w["liab"].notna()).astype(np.int8)
    w["equity"] = w["equity"].fillna(w["assets"] - w["liab"])        # 연결 재무상태표에 자본총계 태그가 없는 경우(자산-부채 항등식)
    w["bs_ok"] = w[["assets", "liab", "equity"]].notna().all(axis=1)
    w = w[w["bs_ok"]].sort_values(["ticker", "reprt_code", "period_end", "pri"])
    w = w.drop_duplicates(["ticker", "reprt_code", "period_end"], keep="first")      # 연결 BS 가 있으면 연결, 없으면 별도
    w["ofs"] = w["pri"].astype(np.int8)
    w["period_end"] = pd.to_datetime(w["period_end"])
    a = fsav[["ticker", "reprt_code", "period_end", "first_rcept_dt", "last_rcept_dt", "n_amend"]]
    w = w.merge(a, on=["ticker", "reprt_code", "period_end"], how="left")
    return w.dropna(subset=["first_rcept_dt"])[["ticker", "reprt_code", "period_end", "first_rcept_dt", "last_rcept_dt", "n_amend", "ofs",
                                                  "assets", "liab", "equity", "ni", "eq_derived"]]


def fin_state(rep, annual_only=False):
    """접수 순서대로 '지금까지 나온 가장 늦은 기간말' 보고서만 남긴다 (나중에 접수된 옛 기간 보고서는 현재 상태를 바꾸지 않음).
    usable_from = first_rcept_dt + 1일. 정정 공시(last_rcept_dt)는 값 갱신에 쓰지 않고 n_amend 표시만 한다."""
    r = rep[rep["reprt_code"] == "11011"] if annual_only else rep
    r = r.sort_values(["ticker", "first_rcept_dt", "period_end", "reprt_code"])
    cm = r.groupby("ticker")["period_end"].cummax()
    r = r[r["period_end"] >= cm]
    # 같은 날 접수된 보고서가 여럿이면 기간말이 가장 늦은 것이 마지막에 오도록 (merge_asof 가 마지막 행을 고른다). 정렬을 결정적으로 해야 입력 순서에 안 흔들린다
    return r.assign(usable_from=r["first_rcept_dt"] + NEXT_DAY).sort_values(["usable_from", "ticker", "period_end", "reprt_code"])


def fin_asof(rows, state):
    """rows(ticker, date) 에 date >= usable_from 인 가장 최근 상태 보고서를 붙인다 (원래 행 순서 유지)."""
    left = rows[["ticker", "date"]].reset_index(drop=True)
    left["_o"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("date"), state, left_on="date", right_on="usable_from", by="ticker", direction="backward")
    return m.sort_values("_o").reset_index(drop=True)


# ------------------------------------------------------------------ 본체
def _is_sorted(P):
    """(ticker, date) 오름차순이고 종목별로 붙어 있는지 (정렬 복사본을 만들지 않으려는 검사)."""
    c, u = pd.factorize(P["ticker"])
    d = P["date"].to_numpy()
    same = c[1:] == c[:-1]
    return bool(np.all(c[1:] >= c[:-1]) and np.all(u[:-1] < u[1:]) and np.all(d[1:][same] > d[:-1][same]))


def build(panel, cls, fsav, fscore, calendar=None, controls=False, label_cap=LABEL_CAP, inplace=False):
    """입력 DataFrame 들로 형성일 변수 표를 만든다. controls=True 이면 일부러 미래를 쓰는 양성 대조 열(ctl_*)을 덧붙인다.
    inplace=True 이면 panel 을 복사하지 않고 그대로 쓴다(메모리 절약; PANEL_COLS 열만 있고 RangeIndex 여야 한다)."""
    t0 = time.time()
    P = panel if inplace else panel[PANEL_COLS]
    if not _is_sorted(P):
        P = P.sort_values(["ticker", "date"]).reset_index(drop=True)
    assert isinstance(P.index, pd.RangeIndex)
    n = len(P)
    cal = np.sort(pd.unique(P["date"].to_numpy())) if calendar is None else np.sort(np.asarray(calendar, dtype="datetime64[ns]"))
    codes, tickers = pd.factorize(P["ticker"])
    tickers = np.asarray(tickers)
    cnt = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    tstart = np.repeat(starts, cnt)
    tend = np.repeat(starts + cnt - 1, cnt)
    date_ns = P["date"].to_numpy()
    didx = np.searchsorted(cal, date_ns)

    # --- 형성일: 주(ISO 주)의 마지막 거래일
    cs = pd.DatetimeIndex(cal)
    form = cs.to_series().groupby(cs.to_period("W")).max().to_numpy()
    univ = (P["f_tradable"].to_numpy() == 1) & (P["exclude_flag"].to_numpy() == 0)
    sel = np.flatnonzero(univ & np.isin(date_ns, form))
    F = pd.DataFrame({"ticker": P["ticker"].to_numpy()[sel], "date": date_ns[sel]})
    I = sel                                                          # 형성 행의 전역 위치

    # --- B1 수급: 개인·기관·외국인 flow 의 5·20거래일 합 (형성일 포함)
    pre_bad = ((P["inv_sum_closed"].to_numpy() == 0) & (P["foreign_def_changed"].to_numpy() == 0)).astype(float)   # 4그룹 합이 안 닫히는 구간
    post = (P["foreign_def_changed"].to_numpy() == 1).astype(float)                                             # 외국인 정의 변경 후
    ok_all = {}
    nobs = {}
    for g, col in (("indiv", "indiv_flow"), ("inst", "inst_flow"), ("for", "foreign_flow")):
        x = P[col].to_numpy()
        for w, mn in FLOW_MIN.items():
            s, c, nr = win_sum(x, w, 0, tstart, tend, codes)
            sb, _, _ = win_sum(pre_bad, w, 0, tstart, tend, codes)
            ok = (c >= mn) & (sb == 0)
            if g == "for":
                sp, _, _ = win_sum(post, w, 0, tstart, tend, codes)
                ok &= ~((sp > 0) & (sp < nr))                       # 창 안에 정의 변경 전·후가 섞이면 외국인만 결측
            val = np.where(ok, s * w / np.maximum(c, 1), np.nan)    # 유효 관측 평균 x 창 길이
            F[f"flow_{g}_{w}"] = val[I]
            ok_all[(g, w)] = ok[I]
            if g == "indiv":
                nobs[w] = c[I]
    for w in FLOW_MIN:
        F[f"inv_ok{w}"] = (ok_all[("indiv", w)] & ok_all[("inst", w)] & ok_all[("for", w)]).astype(np.int8)
        F[f"inv_nobs{w}"] = nobs[w].astype(np.int8)

    # --- B2 거래·가격
    for c in B2_M0:
        F[c] = P[c].to_numpy()[I]
    mc = P["mktcap"].to_numpy()
    F["mktcap"] = mc[I]
    F["log_mktcap"] = np.log(np.where(mc > 0, mc, np.nan))[I]
    tradable = P["f_tradable"].to_numpy() == 1
    tv = np.where(tradable, P["tv_filled"].to_numpy(), np.nan)
    s, c, _ = win_sum(tv, TV_WIN, 1, tstart, tend, codes)           # 직전 60거래일 (당일 제외)
    avg = np.where(c >= TV_MIN, s / np.maximum(c, 1), np.nan)
    F["avg_tv60"] = avg[I]
    F["log_tv60"] = np.log(np.where(avg > 0, avg, np.nan))[I]
    with np.errstate(divide="ignore", invalid="ignore"):
        ami = np.where(tradable & (tv > 0), np.abs(P["ret1d_clean"].to_numpy()) / tv * AMI_SCALE, np.nan)
    s, c, _ = win_sum(ami, AMI_WIN, 1, tstart, tend, codes)         # 직전 20거래일 (당일 제외)
    F["amihud20"] = np.where(c >= AMI_MIN, s / np.maximum(c, 1), np.nan)[I]

    # --- B3 공시: next_td <= 형성일 인 이벤트의 직전 5·20·60거래일 건수
    last_date = P.groupby("ticker")["date"].max()
    cum = event_cumulative(cls, cal, tickers, last_date)
    ti = didx[I]
    tk = codes[I]
    first_full = int(np.searchsorted(cal, DSC_START.to_datetime64())) + 1     # 이 인덱스부터 창 전체가 수집 구간 안
    for name in ["periodic", "major", "stake", "exch", "amend"]:
        for w in DSC_WINS:
            lo = ti + 1 - w
            ok = lo >= first_full
            v = cum[name][tk, ti + 1] - cum[name][tk, np.maximum(lo, 0)]
            F[f"dsc_{name}_{w}"] = np.where(ok, v, np.nan)

    # --- B3 재무: first_rcept_dt 다음 날부터 쓸 수 있는 가장 최근 보고서 (연결 우선)
    rep = fin_reports(fscore, fsav, set(tickers))
    st = fin_asof(F, fin_state(rep))
    st_a = fin_asof(F, fin_state(rep, annual_only=True))
    a, lb, eq = st["assets"].to_numpy(), st["liab"].to_numpy(), st["equity"].to_numpy()
    F["fin_has"] = st["assets"].notna().to_numpy().astype(np.int8)
    F["fin_size"] = np.log(np.where(a > 0, a, np.nan))
    F["fin_lev"] = np.where(a > 0, lb / np.where(a > 0, a, np.nan), np.nan)
    F["fin_bm"] = np.where(F["mktcap"].to_numpy() > 0, eq / F["mktcap"].to_numpy(), np.nan)
    F["fin_period_end"] = st["period_end"].to_numpy()
    F["fin_rcept"] = st["first_rcept_dt"].to_numpy()
    F["fin_reprt"] = st["reprt_code"].to_numpy()
    F["fin_ofs"] = st["ofs"].to_numpy()
    F["fin_eq_derived"] = st["eq_derived"].to_numpy()
    F["fin_amended"] = ((st["n_amend"] > 0) & (st["last_rcept_dt"] < F["date"])).to_numpy().astype(np.int8)   # t 이전에 정정이 끝난 이력만 (시점 정합)
    F["fin_amend_ever"] = (st["n_amend"] > 0).to_numpy().astype(np.int8)       # 사후 정보: 정정 제외 민감도 전용, 변수로 쓰지 않는다
    F["fin_age_days"] = (F["date"] - st["first_rcept_dt"]).dt.days.to_numpy()
    ea, ni = st_a["equity"].to_numpy(), st_a["ni"].to_numpy()
    F["fin_roe"] = np.where(ea > 0, ni / np.where(ea > 0, ea, np.nan), np.nan)
    F["roe_period_end"] = st_a["period_end"].to_numpy()
    F["roe_rcept"] = st_a["first_rcept_dt"].to_numpy()
    F["roe_ofs"] = st_a["ofs"].to_numpy()
    F["roe_eq_derived"] = st_a["eq_derived"].to_numpy()
    F["roe_amended"] = ((st_a["n_amend"] > 0) & (st_a["last_rcept_dt"] < F["date"])).to_numpy().astype(np.int8)
    F["roe_amend_ever"] = (st_a["n_amend"] > 0).to_numpy().astype(np.int8)     # 사후 정보 (진단 전용)

    # --- 라벨: 형성일 다음 거래일 수정시가 → 5거래일째 수정종가
    op, cl, adj = P["open"].to_numpy().astype(float), P["close"].to_numpy().astype(float), P["adj_close"].to_numpy()
    cens = P["fwd_censored"].to_numpy()
    remain = tend[I] - I
    full = remain >= HOLD
    ei = np.minimum(I + 1, n - 1)
    xi = np.where(full, I + HOLD, tend[I])
    entry_ok = (remain >= 1) & tradable[ei] & (op[ei] > 0) & (cl[ei] > 0) & np.isfinite(adj[ei]) & np.isfinite(adj[xi])
    aopen = op[ei] * adj[ei] / np.where(cl[ei] > 0, cl[ei], np.nan)
    trunc = ~full & (remain >= 1) & (cens[I] == 1)                  # 상장폐지로 잘림: 마지막(정리매매) 가격까지
    status = np.full(len(I), 4, dtype=np.int8)                      # 4: 뒤 행 없음(상폐 마지막 행)
    status[~full & (cens[I] == 2)] = 2                              # 2: 자료 끝 검열 (NaN)
    status[(full | trunc) & ~entry_ok] = 3                          # 3: 다음 거래일 시가로 못 산다
    status[(full | trunc) & entry_ok] = np.where(full, 0, 1)[(full | trunc) & entry_ok]   # 0 완전, 1 상폐 잘림(값 있음)
    ret = np.where(status <= 1, adj[xi] / aopen - 1, np.nan)
    end_date = np.where(remain >= 1, date_ns[xi], np.datetime64("NaT", "ns"))
    cens_out = cens[I].astype(np.int8)
    F["fwd_ret5_panel"] = P["fwd_ret5"].to_numpy()[I]
    if label_cap is not None:
        cap = np.datetime64(pd.Timestamp(label_cap), "ns")
        over = (date_ns[I] > cap) | (end_date > cap)                 # 청산일이 기준일을 넘는 라벨은 값도 청산일도 저장하지 않는다
        status[over] = 5                                             # 5: 시험 구간 보호로 마스크
        ret[over] = np.nan
        end_date = np.where(over, np.datetime64("NaT", "ns"), end_date)
        F.loc[over, "fwd_ret5_panel"] = np.nan
        cens_out[over] = -1                                          # 마스크된 행은 상폐·자료 끝 여부(미래 정보)도 지운다
    F["label_censored"] = cens_out
    F["label_end_date"] = end_date
    F["label_ret5"] = ret
    F["label_status"] = status

    # --- 양성 대조 (일부러 미래를 쓰는 열; 저장하지 않는다)
    if controls:
        adj_f = adj[np.minimum(I + HOLD, n - 1)]
        F["ctl_fwd_ret5"] = np.where(I + HOLD <= tend[I], adj_f / adj[I] - 1, np.nan)
        s, c, _ = win_sum(P["indiv_flow"].to_numpy(), 5, -2, tstart, tend, codes)         # 형성일 앞뒤 2일 중심 창
        F["ctl_flow5_centered"] = np.where(c >= 3, s / np.maximum(c, 1), np.nan)[I]
        for name in ("periodic", "stake", "exch"):                                          # 다음 5거래일의 공시 (미래)
            hi = np.minimum(ti + 1 + HOLD, len(cal))
            F[f"ctl_dsc_{name}_next5"] = (cum[name][tk, hi] - cum[name][tk, ti + 1]).astype(float)
        s2 = fin_asof(F, rep.assign(usable_from=rep["period_end"]).sort_values(["usable_from", "ticker", "reprt_code"]))   # 기간말 날짜로 재무 사용 (접수일 무시)
        F["ctl_fin_by_period_end"] = np.log(np.where(s2["assets"] > 0, s2["assets"], np.nan))

    # --- 날짜별 횡단면 순위 (rank(pct=True) - 0.5) 와 라벨 순위
    rk = F.groupby("date", sort=False)[FEATURES].rank(pct=True) - 0.5
    rk.columns = RK
    F = pd.concat([F, rk], axis=1)
    F["label_rk"] = F.groupby("date", sort=False)["label_ret5"].rank(pct=True) - 0.5
    F = F.sort_values(["date", "ticker"]).reset_index(drop=True)
    F.attrs["seconds"] = time.time() - t0
    return F


# ------------------------------------------------------------------ 열 사전
def column_doc():
    rows = [
        ("ticker, date", "키", "종목코드(6자리), 형성일(주의 마지막 거래일). 유니버스 f_tradable==1 & exclude_flag==0"),
        ("flow_{indiv,inst,for}_{5,20}", "B1", "개인·기관·외국인 순매수 flow(패널 분모 보정 열, [-1,1])의 직전 5/20거래일 합(형성일 포함). 유효 관측 평균 x 창 길이. 창 안 유효일이 4/5, 15/20 미만이거나 4그룹 합이 안 닫히는 날(inv_sum_closed==0, ~2012-05-02)이 포함되면 NaN. 외국인은 창에 2026-05-26 정의 변경 전후가 섞이면 NaN"),
        ("inv_ok5, inv_ok20", "B1 표시", "1 = 해당 창에서 세 그룹 모두 값이 있음 (위 규칙 통과)"),
        ("inv_nobs5, inv_nobs20", "B1 표시", "창 안 개인 flow 유효 관측일 수"),
        ("str5, str20, mom12_1, near52h, vol20, abvol", "B2 (M0 6지표)", "패널 열 그대로 형성일 값: 5일 수익, 20일 수익, t-252~t-21 수익, 수정종가/52주 고가, 20일 수익 표준편차(ret1d_clean), 5일/60일 거래대금(abvol)"),
        ("log_mktcap", "B2", "ln(종가 x 상장주식수). 2026-06-23 이후 상장주식수 공백이라 NaN"),
        ("log_tv60", "B2", "ln(직전 60거래일 평균 거래대금, 당일 제외, 거래 가능한 날만, 최소 30일). 원값은 avg_tv60"),
        ("amihud20", "B2", "직전 20거래일(당일 제외) 평균 abs(ret1d_clean) / 거래대금(원) x 1e9, 최소 15일"),
        ("dsc_{periodic,major,stake,exch,amend}_{5,20,60}", "B3 공시", "next_td <= 형성일 이고 next_td 가 직전 5/20/60거래일 안인 공시 건수: 정기, 주요사항, 지분, 수시·공정(거래소), 정정. 발행공시(분류 '기타') 제외(정정 포함). 2015-01-01 이전을 덮는 창은 NaN, 상장폐지(마지막 행) 후 접수분 제외"),
        ("fin_size", "B3 재무", "ln(자산총계). first_rcept_dt 다음 날부터 쓸 수 있는 가장 최근 보고서(분기 포함), 연결(자산·부채 모두 있을 때) 우선·없으면 별도. 자본총계 태그가 없으면 자산-부채"),
        ("fin_lev", "B3 재무", "부채총계/자산총계 (같은 보고서)"),
        ("fin_bm", "B3 재무", "자본총계 / 형성일 시가총액(mktcap). 자본잠식이면 음수"),
        ("fin_roe", "B3 재무", "가장 최근 사업보고서(연간)의 당기순이익/자본총계(기말). 자본총계<=0 이면 NaN. 분기 누적 값은 쓰지 않음"),
        ("fin_has, fin_period_end, fin_rcept, fin_reprt, fin_ofs, fin_amended, fin_age_days", "B3 표시", "쓰인 보고서의 유무, 기간말, 최초 접수일, reprt_code, 별도(OFS) 대체 여부, 형성일 전에 정정이 끝난 이력(n_amend>0 & last_rcept_dt<형성일; 값 갱신에는 쓰지 않음), 접수 후 경과일"),
        ("fin_eq_derived, roe_eq_derived", "B3 표시", "1 = 그 보고서에 자본총계(ifrs_Equity) 태그가 없어 자산총계-부채총계로 계산한 값을 쓴 경우"),
        ("fin_amend_ever, roe_amend_ever", "사후 진단", "그 보고서에 정정이 한 번이라도 있었는지(n_amend>0). 형성일 이후 정정도 포함하므로 정정 제외 민감도 분석 전용이며 변수로 쓰지 않는다"),
        ("roe_period_end, roe_rcept, roe_ofs, roe_amended", "B3 표시", "ROE 에 쓰인 사업보고서의 같은 정보"),
        ("avg_tv60, mktcap", "진단", "평균 거래대금 원값(하위 20% 필터용), 시가총액 원값"),
        ("<변수>_rk", "순위", "34개 변수 각각의 날짜별 횡단면 rank(pct=True)-0.5 (유니버스 안, 결측은 NaN 유지). 동률은 평균 순위"),
        ("label_ret5", "라벨", "형성일 다음 거래일 수정시가(open x adj_close/close)에서 5거래일째 수정종가까지 수익률. 상장폐지로 잘리면 마지막 가격까지. 다음 거래일 거래 불가면 NaN. 패널 fwd_ret5(종가→종가)와 시작점이 다르다"),
        ("label_rk", "라벨", "label_ret5 의 날짜별 rank(pct=True)-0.5"),
        ("label_status", "라벨", "0 정상, 1 상폐로 잘림(값 있음), 2 자료 끝 검열, 3 다음 거래일 매수 불가(정지·시가 없음·가격 결측), 4 뒤 행 없음, 5 시험 구간 보호로 마스크(label_end_date > 2016-12-31)"),
        ("label_end_date", "라벨", "라벨 창의 마지막 거래일(청산일). label_status=5(마스크)인 행은 NaT"),
        ("label_censored", "라벨", "패널 fwd_censored (0 완전, 1 상폐로 잘림, 2 자료 끝). 마스크된 행(label_status=5)은 -1"),
        ("fwd_ret5_panel", "라벨 진단", "패널 fwd_ret5(형성일 종가→5거래일 뒤 종가) 같은 마스크 적용"),
    ]
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_doc(F):
    L = ["---", "title: features_v2_kospi 열 사전", "created: 2026-10-12", "updated: 2026-10-12", "tags: [dev/python, finance/퀀트]",
         "source: code/f01_build_features.py 가 실행할 때마다 다시 만든다", "status: seed", "---", "",
         f"# features_v2_kospi.parquet 열 사전", "",
         f"{len(F):,}행 x {F.shape[1]}열, 형성일 {F['date'].nunique()}개 ({F['date'].min().date()} ~ {F['date'].max().date()}), 종목 {F['ticker'].nunique()}개. "
         f"변수 {len(FEATURES)}개 + 순위 열 {len(RK)}개. 시험 구간 보호: label_end_date > {LABEL_CAP.date()} 인 라벨은 NaN(label_status=5).", "",
         "| 열 | 구분 | 정의 |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in column_doc()]
    L += ["", "## 블록별 변수", ""]
    L += [f"- {k} ({len(v)}): " + ", ".join(v) for k, v in BLOCKS.items()]
    L += ["", "## 입력 파일 (재현 확인용)", "", "| 파일 | 크기(byte) | sha256 |", "|---|---|---|"]
    for p in [q for q in (PANEL, CLS, FSAV, FSCORE, FSPRE) if q.exists()]:
        L.append(f"| {p.relative_to(PAPERS)} | {p.stat().st_size:,} | {sha256(p)} |")
    OUT_DOC.write_text("\n".join(L) + "\n", encoding="utf-8")


def load_inputs():
    panel = pd.read_parquet(PANEL, columns=PANEL_COLS)
    cls = pd.read_parquet(CLS, columns=["ticker", "rcept_dt", "cat", "is_amend"])
    fsav = pd.read_parquet(FSAV)
    fscore = pd.read_parquet(FSCORE, columns=["code", "market", "reprt_code", "fs_div", "sj_div", "account_id", "period_end", "thstrm_amount"],
                             filters=[("market", "==", "KOSPI")])
    if FSPRE.exists():                  # 2019 상반기까지 보고서는 ID 접두어가 'ifrs_' 라 d03 핵심 계정 목록에서 빠졌다 → 4계정을 따로 받아 붙인다
        pre = pd.read_parquet(FSPRE, columns=list(fscore.columns))
        pre["account_id"] = pre["account_id"].str.replace(r"^ifrs_", "ifrs-full_", regex=True)
        fscore = pd.concat([fscore, pre[pre["market"] == "KOSPI"]], ignore_index=True)
        fscore = fscore.drop_duplicates(["code", "reprt_code", "period_end", "fs_div", "sj_div", "account_id"], keep="first")
    return panel, cls, fsav, fscore


def main():
    import psutil
    panel, cls, fsav, fscore = load_inputs()
    F = build(panel, cls, fsav, fscore, inplace=True)
    del panel
    for c in RK + ["label_rk"]:
        F[c] = F[c].astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    F.to_parquet(OUT, index=False)
    write_doc(F)
    peak = psutil.Process().memory_info().peak_wset / 2**30
    print(f"저장 {OUT} : {len(F):,}행 x {F.shape[1]}열, 형성일 {F['date'].nunique()}, 종목 {F['ticker'].nunique()}")
    print(f"build {F.attrs.get('seconds', 0):.0f}초, 최대 메모리 {peak:.2f}GB")
    print("라벨 상태 분포:", F["label_status"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

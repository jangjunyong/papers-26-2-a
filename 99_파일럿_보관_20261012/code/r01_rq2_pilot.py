# -*- coding: utf-8 -*-
"""r01 RQ2 예비 분석: CNN 신호 -> 개인 순매수 -> 6~20일 가격 되돌림 (탐색용 파일럿, KOSPI 2011~2012).

입력: results/kaggle/pred_*.parquet (date=형성일, ticker, ret, prob), PAPERS/data/daily/club_db/panel.parquet
출력: results/r01_tables/*.csv, r01_car_paths.png, r01_log.txt  (보고서는 results/r01_rq2_memo.md에 따로 씀)
실행: python r01_rq2_pilot.py   (노트북 CPU 수 분 이내. 딥러닝 없음)

규칙(룩어헤드 금지): 신호·분류 변수(규모/str5/abvol/형성 주 순매수)는 형성일 t 종가까지만 쓴다.
t+1 이후(CAR, 사후 순매수)는 결과 변수로만 쓴다. 설정은 코드 상단 상수로 고정하며 결과를 보고 바꾸지 않는다.
단위: CAR은 %p(일별 초과수익 단순 합), 순매수 흐름은 '직전 60일 평균 거래량(또는 거래대금)의 %' 를 기간 합산.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 경로·상수 (분석 전에 고정)
ROOT = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS"
PANEL = f"{ROOT}/data/daily/club_db/panel.parquet"
KG = f"{ROOT}/26-2/03_차트CNN_연구/results/kaggle"
OUT = f"{ROOT}/26-2/03_차트CNN_연구/results/r01_tables"
SIGNAL_FILES = {"cnn_d": "pred_d_lr1e-4", "cnn_tb": "pred_d_lr1e-4_tb", "cnn_w": "pred_w_lr1e-4"}
PANEL_LAST = "2013-06-30"       # 마지막 형성일(2012-11-30) + 60거래일 이후까지
TOP_Q = 0.90                    # 상위 10분위 = 날짜별 순위 백분위 > 0.9
BURN = 5                        # 앞 5개 형성일은 편입 이력을 알 수 없어 제외
HIST = 4                        # 최근 4개 형성일 안의 편입 이력이 있으면 신규 편입 아님
MIN_CELL = 3                    # 대조군 칸(같은 날 · 4변수)에 비이벤트 종목이 최소 3개
MIN_VALID = 0.8                 # 창 안에서 유효한 일수 비율이 80% 미만이면 결측
BAN = ("2011-08-10", "2011-11-09")
EV_WIN = (-4, 20)               # 공매도 금지와 겹치면 제외할 형성 주~6~20일 창
SIZE_Q, TERC = 5, 3
HORIZON = 60

LOG = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ---------------------------------------------------------------- 통계 도구
def cl_mean(x, g):
    """평균과 군집 표준오차(상수만 있는 회귀의 군집 SE, G/(G-1) 보정). (mean, se, t, n, G)"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(g)[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n, 0
    m = x.mean()
    s = np.bincount(pd.factorize(g)[0], weights=x - m)
    G = len(s)
    if G < 2:
        return m, np.nan, np.nan, n, G
    se = np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n
    return m, se, (m / se if se > 0 else np.nan), n, G


def fit_cluster(y, X, gd, gt=None):
    """OLS + 군집 표준오차. gt가 있으면 날짜·종목 이중 군집(statsmodels 내장 cov_cluster_2groups)."""
    Xc = sm.add_constant(X, has_constant="add")
    groups = gd if gt is None else np.column_stack([gd, gt])
    return sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})


def wsum(M, r0, r1, j):
    """행 r0..r1(포함) 합. 유효 일수가 MIN_VALID 미만이면 NaN."""
    blk = M[r0:r1 + 1, j]
    cnt = np.isfinite(blk).sum(0)
    s = np.nansum(blk, 0)
    s[cnt < int(np.ceil(MIN_VALID * (r1 - r0 + 1)))] = np.nan
    return s


def grp_bin(df, col, k, by="i"):
    r = df.groupby(by)[col].rank(method="first", pct=True)
    return np.minimum(np.ceil(r * k) - 1, k - 1)


# ---------------------------------------------------------------- 데이터
def load_wide():
    cols = ["ticker", "date", "ret1d_clean", "mktcap", "volume", "indiv_buy", "indiv_sell", "indiv_flow_w",
            "inst_flow_w", "foreign_flow_w", "indiv_flow_tv", "str5", "abvol", "mom12_1", "near52h"]
    pn = pd.read_parquet(PANEL, columns=cols)
    pn["indiv_flow_tv"] = pn["indiv_flow_tv"].replace([np.inf, -np.inf], np.nan)
    lo, hi = pn["indiv_flow_tv"].quantile([0.005, 0.995])        # 패널의 _w 열과 같은 방식(전체 표본 0.5/99.5%)
    pn["indiv_flow_tv"] = pn["indiv_flow_tv"].clip(lo, hi)
    pn = pn[pn.date <= PANEL_LAST]
    pv = {c: pn.pivot(index="date", columns="ticker", values=c) for c in cols[2:]}
    dates, tick = pv["volume"].index, pv["volume"].columns
    W = {k: v.reindex(index=dates, columns=tick).values.astype(float) for k, v in pv.items()}
    # 개인 거래 비중: 직전 60거래일(형성일 포함) 개인 (매수+매도) / (2 x 거래량)
    num = (pv["indiv_buy"] + pv["indiv_sell"]).rolling(60, min_periods=40).sum()
    den = pv["volume"].rolling(60, min_periods=40).sum()
    W["share"] = (num / (2 * den)).reindex(index=dates, columns=tick).values.astype(float)
    return W, dates, tick, (lo, hi)


def load_pred(name):
    p = pd.read_parquet(f"{KG}/{name}.parquet")
    p["date"] = pd.to_datetime(p["date"]).astype("datetime64[ns]")
    return p.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_base(W, dates, tick):
    """종목-형성일 표: 공변량(t까지)과 결과(t+1부터)를 만든다."""
    b = load_pred(SIGNAL_FILES["cnn_d"]).rename(columns={"prob": "p_cnn_d"})
    assert not b.duplicated(["date", "ticker"]).any()
    for s in ("cnn_tb", "cnn_w"):
        q = load_pred(SIGNAL_FILES[s])
        assert (q.date.values == b.date.values).all() and (q.ticker.values == b.ticker.values).all(), "pred 키 불일치"
        assert np.allclose(q.ret.values, b.ret.values), "ret 불일치"
        b[f"p_{s}"] = q.prob.values
    dpos = pd.Series(np.arange(len(dates)), index=dates)
    b["i"] = dpos.reindex(b.date).values.astype(int)
    b["j"] = pd.Index(tick).get_indexer(b.ticker)
    assert (b.j >= 0).all()
    i, j = b.i.values, b.j.values
    fd = np.sort(b.i.unique())
    b["k"] = np.searchsorted(fd, i)
    b["ym"] = dates[i].year * 12 + dates[i].month
    for name, key in dict(str5="str5", abvol="abvol", mom12_1="mom12_1", near52h="near52h",
                          mc="mktcap", share="share").items():
        b[name] = W[key][i, j]
    b["mc_prev"] = W["mktcap"][i - 1, j]
    b["logmc"] = np.log(b["mc"])
    lo, hi = b.abvol.quantile([0.01, 0.99])
    b["abvol_w"] = b.abvol.clip(lo, hi)

    n = len(b)
    new = {c: np.full(n, np.nan) for c in
           ["car15", "car620", "car2160", "fi_form", "fi_post", "fi_mid", "ff_form", "ff_post", "ff_mid",
            "fn_form", "fn_post", "fn_mid", "fit_form", "fit_post", "fit_mid", "q_size"]}
    path = np.full((n, HORIZON), np.nan, dtype=np.float32)
    FL = {"fi": W["indiv_flow_w"], "ff": W["foreign_flow_w"], "fn": W["inst_flow_w"], "fit": W["indiv_flow_tv"]}
    R = W["ret1d_clean"]
    for ii, idx in b.groupby("i").indices.items():
        jj = j[idx]
        mcp = pd.Series(b.mc_prev.values[idx])
        q = np.floor((mcp.rank(method="first") - 1) / mcp.notna().sum() * SIZE_Q).values   # NaN 유지
        new["q_size"][idx] = q
        blk = R[ii + 1:ii + 1 + HORIZON, jj]                       # 사후 60거래일 수익률 (60 x n)
        bench = np.full((HORIZON, SIZE_Q), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for g in range(SIZE_Q):
                m = q == g
                if m.any():
                    bench[:, g] = np.nanmean(blk[:, m], axis=1)    # 같은 날 같은 규모 5분위 평균
        qi = np.where(np.isfinite(q), q, 0).astype(int)
        ar = blk - bench[:, qi]
        ar[:, ~np.isfinite(q)] = np.nan
        okm = np.isfinite(ar)
        cum = np.cumsum(np.where(okm, ar, 0.0), axis=0)
        cnt = np.cumsum(okm, axis=0)

        def car(a, bb):
            c = cum[bb - 1] - (cum[a - 2] if a > 1 else 0.0)
            v = cnt[bb - 1] - (cnt[a - 2] if a > 1 else 0)
            c = c.copy()
            c[v < int(np.ceil(MIN_VALID * (bb - a + 1)))] = np.nan
            return c * 100

        new["car15"][idx], new["car620"][idx], new["car2160"][idx] = car(1, 5), car(6, 20), car(21, 60)
        p60 = cum.T * 100
        p60[~np.isfinite(new["car2160"][idx])] = np.nan             # 60일 창이 유효한 종목만 경로에 쓴다
        path[idx] = p60
        for key, M in FL.items():
            new[f"{key}_form"][idx] = wsum(M, ii - 4, ii, jj) * 100
            new[f"{key}_post"][idx] = wsum(M, ii + 1, ii + 5, jj) * 100
            new[f"{key}_mid"][idx] = wsum(M, ii + 6, ii + 20, jj) * 100
    for c, v in new.items():
        b[c] = v
    b["fi_chg"] = b.fi_post - b.fi_form
    b["fit_chg"] = b.fit_post - b.fit_form
    b["str5p"] = b.str5 * 100
    for r_, c_ in (("ri", "fi"), ("rit", "fit"), ("rf", "ff"), ("rn", "fn")):      # 꼬리 두꺼움 대비: 날짜별 백분위(0~100)
        b[r_ + "_post"] = b.groupby("i")[c_ + "_post"].rank(pct=True) * 100
    # 분류 변수: 날짜별 3분위 (형성일 t까지의 정보)
    b["size_t"] = grp_bin(b.assign(_m=b.mc_prev), "_m", TERC)
    b["str5_t"] = grp_bin(b, "str5", TERC)
    b["abvol_t"] = grp_bin(b, "abvol", TERC)
    b["fi_form_t"] = grp_bin(b, "fi_form", TERC)
    b["fit_form_t"] = grp_bin(b, "fit_form", TERC)
    b["share_t"] = grp_bin(b.assign(_s=b.share), "_s", TERC, by=["i", "size_t"])      # 규모 3분위 안 조건부 3분위
    return b, path


# ---------------------------------------------------------------- 신호와 이벤트
def resid_score(b):
    """CNN prob를 str5·abvol·mom12_1·near52h·log mktcap에 날짜별 횡단면 회귀한 잔차."""
    res = np.full(len(b), np.nan)
    r2 = []
    for ii, idx in b.groupby("i").indices.items():
        s = b.iloc[idx]
        X = np.column_stack([np.ones(len(s)), s[["str5", "abvol_w", "mom12_1", "near52h", "logmc"]].values])
        y = s.p_cnn_d.values.astype(float)
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        if ok.sum() < 100:
            continue
        beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        e = y[ok] - X[ok] @ beta
        res[idx[ok]] = e
        r2.append(1 - e.var() / y[ok].var())
    return res, np.array(r2)


def flag_events(k, tk, score, ntk, nd):
    """score가 날짜별 상위 10%이면 top. 신규 편입 = 직전 형성일 비상위 & 최근 HIST개 형성일 편입 이력 없음.
    strict = 최근 HIST개 형성일에 한 번도 상위가 아니었음. 앞 BURN개 형성일은 이벤트 아님."""
    s = pd.Series(score)
    pct = s.groupby(k).rank(method="first", pct=True).values
    top = np.isfinite(pct) & (pct > TOP_Q)
    T = np.zeros((nd, ntk), bool)
    T[k, tk] = top
    prev = np.vstack([np.zeros((1, ntk), bool), T[:-1]])
    entry = T & ~prev
    entry[0] = False
    hist = np.zeros_like(entry)
    hist_top = np.zeros_like(entry)
    for m in range(1, HIST + 1):
        hist[m:] |= entry[:-m]
        hist_top[m:] |= T[:-m]
    ev = entry & ~hist
    ev_strict = T & ~hist_top
    ev[:BURN] = False
    ev_strict[:BURN] = False
    return top, ev[k, tk], ev_strict[k, tk]


# ---------------------------------------------------------------- 매칭과 요약
OUTC = ["car15", "car620", "car2160", "fi_form", "fi_post", "fi_mid", "fi_chg", "fit_form", "fit_post", "fit_mid",
        "fit_chg", "ff_form", "ff_post", "ff_mid", "fn_form", "fn_post", "fn_mid", "ri_post", "rit_post", "rf_post",
        "rn_post", "str5p"]
LABEL = {"car15": "CAR(1,5) %p", "car620": "CAR(6,20) %p", "car2160": "CAR(21,60) %p",
         "fi_form": "개인 형성주(-4~0) [거래량]", "fi_post": "개인 사후주(1~5) [거래량]", "fi_mid": "개인 6~20 [거래량]",
         "fi_chg": "개인 사후주-형성주 [거래량]",
         "fit_form": "개인 형성주(-4~0) [거래대금]", "fit_post": "개인 사후주(1~5) [거래대금]",
         "fit_mid": "개인 6~20 [거래대금]", "fit_chg": "개인 사후주-형성주 [거래대금]",
         "ff_form": "외국인 형성주(-4~0)", "ff_post": "외국인 사후주(1~5)", "ff_mid": "외국인 6~20",
         "fn_form": "기관 형성주(-4~0)", "fn_post": "기관 사후주(1~5)", "fn_mid": "기관 6~20",
         "ri_post": "개인 사후주 순위(0~100) [거래량]", "rit_post": "개인 사후주 순위(0~100) [거래대금]",
         "rf_post": "외국인 사후주 순위(0~100)", "rn_post": "기관 사후주 순위(0~100)",
         "str5p": "형성 주 수익률 str5 %p (t까지 정보)"}
VOLCOLS = [c for c in OUTC if not c.startswith("fit") and c != "rit_post"]
TVCOLS = ["car15", "car620", "car2160", "fit_form", "fit_post", "fit_mid", "fit_chg", "rit_post"]


def match(d, ev, top, flow_t, cols, clean_ctrl=False):
    """이벤트별로 같은 날·규모5분위·str5 3분위·abvol 3분위·형성 주 개인 순매수 3분위 칸의 비이벤트 평균을 붙인다."""
    cell = ["i", "q_size", "str5_t", "abvol_t", flow_t]
    keyok = d[cell].notna().all(axis=1).values
    ctrl = (~ev) & keyok & d.car15.notna().values
    if clean_ctrl:
        ctrl &= ~top
    cm = d[ctrl].groupby(cell)[cols].mean().add_suffix("_c")
    cn = d[ctrl].groupby(cell).size().rename("n_ctrl")
    E = d[ev & keyok].join(cm, on=cell).join(cn, on=cell)
    E["n_ctrl"] = E.n_ctrl.fillna(0)
    E = E[E.n_ctrl >= MIN_CELL].copy()
    for c in cols:
        E[c + "_d"] = E[c] - E[c + "_c"]
    return E, int((ev & keyok).sum()), int(ctrl.sum())


def table_rows(d_ev, E, cols, tag):
    rows = []
    for c in cols:
        ma, sa, ta, na, _ = cl_mean(d_ev[c], d_ev.i)
        me, _, _, nm, _ = cl_mean(E[c], E.i)
        mc_, _, _, _, _ = cl_mean(E[c + "_c"], E.i)
        md, sd, td, nd_, _ = cl_mean(E[c + "_d"], E.i)
        _, _, tm, _, _ = cl_mean(E[c + "_d"], E.ym)
        rows.append(dict(설정=tag, 지표=LABEL[c], N이벤트전체=na, 이벤트평균_전체=ma, t_전체=ta, N매칭=nm, 이벤트평균_매칭=me,
                         대조평균=mc_, 차이=md, t_날짜군집=td, t_월군집=tm, 최소칸=int(E.n_ctrl.min()) if len(E) else 0,
                         중앙칸=float(E.n_ctrl.median()) if len(E) else np.nan))
    return pd.DataFrame(rows)


def share_gap(E, c):
    """개인 거래 비중 3분위(규모 3분위 안 조건부) '높음-낮음' 이벤트-대조 차이와 t(날짜 군집)."""
    Ev = E[E.share_t.notna() & E[c + "_d"].notna()]
    Xd = pd.DataFrame({"mid": (Ev.share_t == 1).astype(float), "high": (Ev.share_t == 2).astype(float)})
    r = fit_cluster(Ev[c + "_d"].values, Xd.values, Ev.i.values)
    return r.params[2], r.tvalues[2], int(r.nobs)


def ban_mask(b, dates):
    bi = np.where((dates >= BAN[0]) & (dates <= BAN[1]))[0]
    return (bi.min() <= b.i.values + EV_WIN[1]) & (bi.max() >= b.i.values + EV_WIN[0])


# ---------------------------------------------------------------- 회귀
RTERMS = ["indiv_post", "foreign_post", "inst_post", "car15", "indiv_form", "foreign_form", "inst_form", "str5",
          "logmc", "abvol"]


def design(d, scale):
    ind = "fit" if scale == "tv" else "fi"
    z = lambda c: (d[c] - d[c].mean()) / d[c].std()      # 해당 표본 전 종목-주 기준 표준화 (계수 = 1 SD당 CAR %p)
    return pd.DataFrame({"indiv_post": z(f"{ind}_post"), "foreign_post": z("ff_post"), "inst_post": z("fn_post"),
                         "car15": z("car15"), "indiv_form": z(f"{ind}_form"), "foreign_form": z("ff_form"),
                         "inst_form": z("fn_form"), "str5": z("str5"), "logmc": z("logmc"), "abvol": z("abvol_w")},
                        index=d.index)


def reg_block(d, X, mask, dummy, with_car, name, rows, extra_note=""):
    """dummy=None이면 이벤트 표본 회귀, 아니면 전 종목-주 표본 + dummy 상호작용."""
    terms = [t for t in RTERMS if with_car or t != "car15"]
    Xs = X[terms].copy()
    post = ["indiv_post", "foreign_post", "inst_post"]
    if dummy is not None:
        Xs["D"] = dummy.astype(float)
        for t in post:
            Xs["D_x_" + t] = Xs[t] * Xs["D"]
    y = d["car620"]
    ok = mask & y.notna().values & Xs.notna().all(axis=1).values
    Y, XX, dd = y[ok], Xs[ok], d[ok]
    if len(Y) < 50:
        return
    r = fit_cluster(Y.values, XX.values, dd.i.values, dd.j.values)
    names = ["const"] + list(XX.columns)
    for nm_, c, t in zip(names, r.params, r.tvalues):
        rows.append(dict(회귀=name, CAR15통제="예" if with_car else "아니오", 항=nm_, 계수=c, t=t, N=int(r.nobs),
                         날짜수=dd.i.nunique(), 종목수=dd.j.nunique(), R2=r.rsquared))
    if dummy is not None:                                   # 합계 계수(기본 + 상호작용) = 이벤트/상위 종목의 총 기울기
        for t in post:
            hyp = np.zeros(len(names))
            hyp[names.index(t)] = 1
            hyp[names.index("D_x_" + t)] = 1
            tt = r.t_test(hyp)
            rows.append(dict(회귀=name, CAR15통제="예" if with_car else "아니오", 항="합계_" + t,
                             계수=float(np.squeeze(tt.effect)), t=float(np.squeeze(tt.tvalue)), N=int(r.nobs),
                             날짜수=dd.i.nunique(), 종목수=dd.j.nunique(), R2=r.rsquared))


# ---------------------------------------------------------------- 한 번의 분석
def run(b, dates, sig, scale="vol", excl_ban=False, clean_ctrl=False, strict=False, detail=False):
    samp = (b.k.values >= BURN)
    if excl_ban:
        samp &= ~ban_mask(b, dates)
    d = b[samp].copy()
    ev = d[("evs_" if strict else "ev_") + sig].values
    top = d["top_" + sig].values
    ind = "fit" if scale == "tv" else "fi"
    flow_t = f"{ind}_form_t"
    cols = OUTC
    E, n_ev_key, n_ctrl = match(d, ev, top, flow_t, cols, clean_ctrl)
    d_ev = d[ev]
    res = dict(d=d, ev=ev, top=top, E=E, d_ev=d_ev, n_ev=int(ev.sum()), n_ev_key=n_ev_key, n_ctrl=n_ctrl,
               n_samp=len(d), n_dates=d.i.nunique())
    X = design(d, scale)
    rows = []
    reg_block(d, X, ev, None, False, "이벤트표본", rows)
    reg_block(d, X, ev, None, True, "이벤트표본", rows)
    for with_car in (False, True):
        reg_block(d, X, np.ones(len(d), bool), top, with_car, "전종목주+상위10분위더미", rows)
        reg_block(d, X, np.ones(len(d), bool), ev, with_car, "전종목주+신규편입더미", rows)
    # 꼬리 두꺼움 점검: 사후주 흐름을 날짜별 백분위로 바꾼 회귀 (형성 주 흐름 등 나머지 통제는 그대로)
    Xr = X.copy()
    zz = lambda c: (d[c] - d[c].mean()) / d[c].std()
    Xr["indiv_post"] = zz("rit_post" if scale == "tv" else "ri_post")
    Xr["foreign_post"], Xr["inst_post"] = zz("rf_post"), zz("rn_post")
    for with_car in (False, True):
        reg_block(d, Xr, ev, None, with_car, "이벤트표본(순위흐름)", rows)
        reg_block(d, Xr, np.ones(len(d), bool), top, with_car, "전종목주+상위10분위더미(순위흐름)", rows)
    res["reg"] = pd.DataFrame(rows)
    res["X"] = X
    return res


def coef(reg, model, car, term):
    r = reg[(reg.회귀 == model) & (reg.CAR15통제 == car) & (reg.항 == term)]
    return (r.계수.iloc[0], r.t.iloc[0]) if len(r) else (np.nan, np.nan)


# ---------------------------------------------------------------- 자체 점검
def selfcheck(b, W, dates, tick):
    """(1) pred ret = 패널 수정주가 5일 뒤 수익률 (달력·정렬 점검) (2) 이벤트는 t 이후 pred를 지워도 같다
    (3) cl_mean = statsmodels 군집 SE, 이중군집이 한쪽 군집이 개체별이면 단일 군집과 같다."""
    adj = pd.read_parquet(PANEL, columns=["ticker", "date", "adj_close"])
    adj = adj[adj.date <= PANEL_LAST].pivot(index="date", columns="ticker", values="adj_close").reindex(index=dates, columns=tick).values
    r5 = adj[b.i.values + 5, b.j.values] / adj[b.i.values, b.j.values] - 1
    ok = np.isfinite(r5)
    dif = np.abs(r5[ok] - b.ret.values[ok])      # 일부 종목은 수정주가 개정 시점 차이로 1e-4 안팎 어긋난다(권리락 보정 등)
    assert ok.mean() > 0.99 and (dif < 1e-4).mean() > 0.97 and dif.max() < 0.05, "pred ret가 패널과 안 맞음"
    nd, ntk = b.k.max() + 1, len(tick)
    full = flag_events(b.k.values, b.j.values, b.p_cnn_d.values, ntk, nd)[1]
    cut = b.k.values <= 50
    part = flag_events(b.k.values[cut], b.j.values[cut], b.p_cnn_d.values[cut], ntk, nd)[1]
    assert (full[cut] == part).all(), "이벤트 정의에 미래 정보가 들어감"
    rng = np.random.default_rng(0)
    x, g = rng.normal(size=500), rng.integers(0, 40, 500)
    m, se, t, n, G = cl_mean(x, g)
    r = sm.OLS(x, np.ones(500)).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert abs(se - r.bse[0]) < 1e-10, "cl_mean 표준오차 불일치"
    Xr = rng.normal(size=(500, 2))
    r1 = fit_cluster(x, Xr, g)
    r2 = fit_cluster(x, Xr, g, np.arange(500))
    assert np.allclose(r1.bse, r2.bse, rtol=1e-6), "이중군집 사용법 점검 실패"
    P("[자체 점검 통과] pred ret=패널 5일 수익률 / 이벤트 미래정보 없음 / 군집 SE 일치")


# ---------------------------------------------------------------- 메인
def main():
    os.makedirs(OUT, exist_ok=True)
    W, dates, tick, (lo_tv, hi_tv) = load_wide()
    b, path = build_base(W, dates, tick)
    P(f"종목-형성일 {len(b)}행, 형성일 {b.k.nunique()}개, 종목 {b.ticker.nunique()}, 패널 일수 {len(dates)}")
    P(f"indiv_flow_tv 윈저화 경계(전체 패널 0.5/99.5%): {lo_tv:.4f} ~ {hi_tv:.4f}")
    selfcheck(b, W, dates, tick)

    nd, ntk = b.k.max() + 1, len(tick)
    # 신호 3종(CNN 변형) + 보조 2종(단기 반전, 잔차)
    b["s_cnn_d"], b["s_cnn_tb"], b["s_cnn_w"] = b.p_cnn_d, b.p_cnn_tb, b.p_cnn_w
    b["s_str"] = -b.str5
    b["s_res"], r2 = resid_score(b)
    P(f"잔차 신호: 날짜별 횡단면 R2 평균 {r2.mean():.3f} (최소 {r2.min():.3f}, 최대 {r2.max():.3f}), 회귀 가능 날짜 {len(r2)}개, "
      f"잔차 가용 종목-주 {int(b.s_res.notna().sum())}")
    for s in ["cnn_d", "cnn_tb", "cnn_w", "str", "res"]:
        t_, e_, es_ = flag_events(b.k.values, b.j.values, b["s_" + s].values, ntk, nd)
        b["top_" + s], b["ev_" + s], b["evs_" + s] = t_, e_, es_

    # ---- 신호 진단
    samp = b[b.k >= BURN]
    ic = samp.groupby("i").apply(lambda g: g.p_cnn_d.rank().corr(g.ret.rank()), include_groups=False)
    rho_str = samp.groupby("i").apply(lambda g: g.p_cnn_d.rank().corr((-g.str5).rank()), include_groups=False)
    dec = samp.assign(dec=samp.groupby("i").p_cnn_d.rank(pct=True)).assign(top=lambda x: x.dec > 0.9, bot=lambda x: x.dec <= 0.1)
    sp = dec.groupby(["i"]).apply(lambda g: g.loc[g.top, "ret"].mean() - g.loc[g.bot, "ret"].mean(), include_groups=False)
    diag = []
    m_, _, t_, n_, _ = cl_mean(ic, np.arange(len(ic)))
    diag.append(("주간 순위상관 IC(prob, 이후 5일 수익률) 평균", m_, t_, n_))
    m_, _, t_, n_, _ = cl_mean(sp * 100, np.arange(len(sp)))
    diag.append(("상위-하위 10분위 이후 5일 수익률 차(%p/주)", m_, t_, n_))
    m_, _, t_, n_, _ = cl_mean(rho_str, np.arange(len(rho_str)))
    diag.append(("순위상관(prob, -str5) 평균", m_, t_, n_))
    ovl = []
    for ii, g in samp.groupby("i"):
        a = g.top_cnn_d.values
        ovl.append((a & g.top_str.values).sum() / max(a.sum(), 1))
    diag.append(("CNN 상위 10분위 중 -str5 상위 10분위 비율(주 평균)", float(np.mean(ovl)), np.nan, len(ovl)))
    evc = samp[samp.ev_cnn_d]
    diag.append(("CNN 신규편입 이벤트 중 -str5 상위 10분위에도 속한 비율", float(evc.top_str.mean()), np.nan, len(evc)))
    diag.append(("CNN 신규편입 이벤트 중 잔차신호 상위 10분위에도 속한 비율", float(evc.top_res.mean()), np.nan, len(evc)))
    diag = pd.DataFrame(diag, columns=["항목", "값", "t", "N"])
    diag.to_csv(f"{OUT}/r01_t0_signal_diag.csv", index=False, encoding="utf-8-sig")
    P(diag.to_string(index=False))

    # ---- 기준 분석 (CNN 대표 설정, 거래량 척도)
    main = run(b, dates, "cnn_d", "vol")
    tv = run(b, dates, "cnn_d", "tv")
    P(f"\n[기준] 표본 종목-주 {main['n_samp']}, 형성일 {main['n_dates']}, 이벤트 {main['n_ev']} (키 완비 {main['n_ev_key']}), "
      f"매칭 {len(main['E'])}, 대조 후보 {main['n_ctrl']}")
    t1 = pd.concat([table_rows(main["d_ev"], main["E"], VOLCOLS, "CNN 기준(거래량 척도 칸)"),
                    table_rows(tv["d_ev"], tv["E"], TVCOLS, "CNN 기준(거래대금 척도 칸)")])
    # 판정표 A-1 정의: CAR(1,5) > 0 인 종목에서의 CAR(6,20). 결과로 자른 하위표본이라 단기 반전 효과가 섞일 수 있어 참고용
    d_ = main["d"]
    pos = (d_.car15 > 0).values
    E_pos, _, _ = match(d_[pos], main["ev"][pos], main["top"][pos], "fi_form_t", ["car620"])
    sub = table_rows(main["d_ev"][main["d_ev"].car15 > 0], E_pos, ["car620"], "CNN 기준, CAR(1,5)>0 하위표본(사후 분류)")
    sub["지표"] = "CAR(6,20) %p | CAR(1,5)>0 하위표본"
    t1 = pd.concat([t1, sub])
    t1.to_csv(f"{OUT}/r01_t1_event_vs_control_main.csv", index=False, encoding="utf-8-sig")
    P(t1.round(3).to_string(index=False))

    # ---- 회귀
    regs = []
    for tag, rr in (("거래량", main), ("거래대금", tv)):
        x = rr["reg"].copy()
        x.insert(0, "개인척도", tag)
        regs.append(x)
    t2 = pd.concat(regs)
    t2.to_csv(f"{OUT}/r01_t2_regressions.csv", index=False, encoding="utf-8-sig")
    keep = t2[t2.항.isin(["indiv_post", "foreign_post", "inst_post", "car15", "D_x_indiv_post", "D_x_foreign_post",
                          "D_x_inst_post", "합계_indiv_post", "합계_foreign_post", "합계_inst_post"])]
    P(keep.round(3).to_string(index=False))

    # ---- 개인 거래 비중 3분위(규모 3분위 안 조건부)별 CAR(6,20)
    E = main["E"]
    rows = []
    for g in (0, 1, 2):
        s = E[E.share_t == g]
        for c in ("car15", "car620", "car2160", "fi_post"):
            m_, _, t_, n_, _ = cl_mean(s[c + "_d"], s.i)
            rows.append(dict(개인비중분위=["낮음", "중간", "높음"][g], 지표=LABEL[c], N=n_, 이벤트평균=s[c].mean(),
                             대조평균=s[c + "_c"].mean(), 차이=m_, t=t_))
    sp_rows = []
    for c in ("car15", "car620", "car2160", "fi_post"):
        g_, tg_, ng_ = share_gap(E, c)
        sp_rows.append(dict(개인비중분위="높음-낮음", 지표=LABEL[c], N=ng_, 이벤트평균=np.nan, 대조평균=np.nan, 차이=g_, t=tg_))
    t6 = pd.DataFrame(rows + sp_rows)
    t6.to_csv(f"{OUT}/r01_t6_indiv_share_split.csv", index=False, encoding="utf-8-sig")
    P(t6.round(3).to_string(index=False))

    # ---- 이벤트 안 사후주 순매수 3분위(날짜별, 사후 분류라 메커니즘 서술용) -> A2와 같은 형식
    rows7 = []
    Ev = main["d_ev"].copy()
    for who, col in (("개인", "fi_post"), ("외국인", "ff_post"), ("기관", "fn_post")):
        Ev["tt"] = grp_bin(Ev, col, 3)
        for out in ("car15", "car620"):
            sub = Ev[Ev.tt.notna() & Ev[out].notna()]
            for g in (0, 1, 2):
                q_ = sub[sub.tt == g]
                m_, _, t_, n_, _ = cl_mean(q_[out], q_.i)
                rows7.append(dict(주체=who, 결과=LABEL[out], 분위=f"T{g + 1}", N=n_, 평균=m_, t=t_))
            Xd = pd.DataFrame({"mid": (sub.tt == 1).astype(float), "high": (sub.tt == 2).astype(float)})
            r = fit_cluster(sub[out].values, Xd.values, sub.i.values)
            rows7.append(dict(주체=who, 결과=LABEL[out], 분위="T3-T1", N=int(r.nobs), 평균=r.params[2], t=r.tvalues[2]))
    t7 = pd.DataFrame(rows7)
    t7.to_csv(f"{OUT}/r01_t7_post_flow_terciles_in_events.csv", index=False, encoding="utf-8-sig")
    P(t7.round(3).to_string(index=False))

    # ---- 흐름과 같은 기간 수익의 상관 (A2 비교용): 전 종목-주와 이벤트
    cor = []
    for nm, dd in (("전 종목-주", main["d"]), ("CNN 이벤트", main["d_ev"])):
        for col in ("fi_post", "ff_post", "fn_post"):
            ok = dd[[col, "car15"]].dropna()
            cor.append(dict(표본=nm, 흐름=col, 스피어만_CAR15=ok[col].rank().corr(ok["car15"].rank()), N=len(ok)))
    cor = pd.DataFrame(cor)
    cor.to_csv(f"{OUT}/r01_t0b_flow_car15_corr.csv", index=False, encoding="utf-8-sig")
    P(cor.round(3).to_string(index=False))

    # ---- 보조 신호 + 강건성
    robust = []

    def summarize(tag, rr, sig, scale, excl, ctrl, entry):
        E_, regm = rr["E"], rr["reg"]
        ind_p = "fit_post" if scale == "tv" else "fi_post"
        row = dict(설정=tag, 신호=sig, 순매수척도=scale, 공매도금지주제외=excl, 대조군=ctrl, 편입정의=entry,
                   표본종목주=rr["n_samp"], 형성일수=rr["n_dates"], N이벤트=rr["n_ev"], N매칭=len(E_),
                   최소칸=int(E_.n_ctrl.min()) if len(E_) else 0, 중앙칸=float(E_.n_ctrl.median()) if len(E_) else np.nan)
        for c, nm in (("car15", "CAR15"), ("car620", "CAR620"), ("car2160", "CAR2160")):
            m_, _, t_, _, _ = cl_mean(rr["d_ev"][c], rr["d_ev"].i)
            row[f"이벤트{nm}"], row[f"t_이벤트{nm}"] = m_, t_
            m_, _, t_, _, _ = cl_mean(E_[c + "_d"], E_.i)
            row[f"차이{nm}"], row[f"t_차이{nm}"] = m_, t_
        for c, nm in ((ind_p, "개인사후주"), ("ff_post", "외국인사후주"), ("fn_post", "기관사후주"),
                      ("rit_post" if scale == "tv" else "ri_post", "개인사후주순위")):
            m_, _, t_, _, _ = cl_mean(E_[c + "_d"], E_.i)
            row[f"차이{nm}"], row[f"t_차이{nm}"] = m_, t_
        for t_ in ("indiv_post", "foreign_post", "inst_post"):
            row[f"회귀_{t_}"], row[f"t_회귀_{t_}"] = coef(regm, "이벤트표본", "예", t_)
        row["회귀_CAR15계수"], row["t_회귀_CAR15계수"] = coef(regm, "이벤트표본", "예", "car15")
        for c, nm in (("car620", "CAR620"), ("car2160", "CAR2160")):
            row[f"개인비중높음-낮음_차이{nm}"], row[f"t_개인비중높음-낮음_차이{nm}"], _ = share_gap(E_, c)
        for t_ in ("indiv_post", "foreign_post"):
            row[f"순위회귀_{t_}"], row[f"t_순위회귀_{t_}"] = coef(regm, "이벤트표본(순위흐름)", "예", t_)
        row["전종목_상위더미_합계_indiv"], row["t_전종목_상위더미_합계_indiv"] = coef(regm, "전종목주+상위10분위더미", "예", "합계_indiv_post")
        row["전종목_상위더미_합계_foreign"], row["t_전종목_상위더미_합계_foreign"] = coef(regm, "전종목주+상위10분위더미", "예", "합계_foreign_post")
        return row

    runs = {}
    for sig in ["cnn_d", "cnn_tb", "cnn_w", "str", "res"]:
        for scale in ["vol", "tv"]:
            for excl in [False, True]:
                rr = main if (sig == "cnn_d" and scale == "vol" and not excl) else \
                    tv if (sig == "cnn_d" and scale == "tv" and not excl) else run(b, dates, sig, scale, excl)
                runs[(sig, scale, excl)] = rr
                robust.append(summarize(f"{sig}|{scale}|금지주제외={excl}", rr, sig, scale, excl, "전체 비이벤트", "신규편입(4주 이력)"))
    rr = run(b, dates, "cnn_d", "vol", clean_ctrl=True)
    robust.append(summarize("cnn_d|vol|대조군=상위10분위 제외", rr, "cnn_d", "vol", False, "상위10분위 제외", "신규편입(4주 이력)"))
    rr = run(b, dates, "cnn_d", "vol", strict=True)
    robust.append(summarize("cnn_d|vol|엄격 편입(4주 상위 이력 없음)", rr, "cnn_d", "vol", False, "전체 비이벤트", "4주 상위 이력 없음"))
    t4 = pd.DataFrame(robust)
    t4.to_csv(f"{OUT}/r01_t4_robustness.csv", index=False, encoding="utf-8-sig")
    show = ["설정", "N이벤트", "N매칭", "최소칸", "이벤트CAR15", "t_이벤트CAR15", "이벤트CAR620", "t_이벤트CAR620", "차이CAR620",
            "t_차이CAR620", "차이개인사후주", "t_차이개인사후주", "회귀_indiv_post", "t_회귀_indiv_post", "회귀_foreign_post",
            "t_회귀_foreign_post"]
    P(t4[show].round(3).to_string(index=False))

    # 보조 신호 비교표 (t3)
    t3 = t4[t4.설정.isin(["cnn_d|vol|금지주제외=False", "str|vol|금지주제외=False", "res|vol|금지주제외=False"])].copy()
    t3.to_csv(f"{OUT}/r01_t3_aux_signals.csv", index=False, encoding="utf-8-sig")
    # 보조 신호 상세 표 (event vs control)
    parts = []
    for sig, nm in (("str", "단기반전(-str5) 상위 10분위"), ("res", "CNN 잔차 상위 10분위")):
        rr = runs[(sig, "vol", False)]
        parts.append(table_rows(rr["d_ev"], rr["E"], [c for c in OUTC if not c.startswith("fit")], nm))
    pd.concat(parts).to_csv(f"{OUT}/r01_t3b_aux_signals_detail.csv", index=False, encoding="utf-8-sig")
    # 전 설정 회귀 계수
    allreg = []
    for (sig, scale, excl), rr in runs.items():
        x = rr["reg"].copy()
        x.insert(0, "설정", f"{sig}|{scale}|금지주제외={excl}")
        allreg.append(x)
    pd.concat(allreg).to_csv(f"{OUT}/r01_t2b_regressions_all_settings.csv", index=False, encoding="utf-8-sig")

    # ---- 판정표 (A-2 가격 압력 조건)
    def row_of(tb, metric, col):
        r = tb[tb.지표 == metric]
        return r.iloc[0][col]

    ver = []

    def add(cond, measure, val, tv_, ok, note=""):
        bad = not (np.isfinite(val) and np.isfinite(tv_)) if ok is not None else False
        assert not bad, f"판정 입력이 비어 있음: {cond} / {measure}"
        ver.append(dict(조건=cond, 측정=measure, 값=val, t=tv_, 판정="충족" if ok else ("참고" if ok is None else "불충족"), 비고=note))

    d_ip = row_of(t1, LABEL["fi_post"], "차이"); t_ip = row_of(t1, LABEL["fi_post"], "t_날짜군집")
    d_ipt = row_of(t1, LABEL["fit_post"], "차이"); t_ipt = row_of(t1, LABEL["fit_post"], "t_날짜군집")
    d_chg = row_of(t1, LABEL["fi_chg"], "차이"); t_chg = row_of(t1, LABEL["fi_chg"], "t_날짜군집")
    add("편입 뒤 개인 순매수 증가", "사후주 개인 순매수, 이벤트-대조 [거래량 %]", d_ip, t_ip, d_ip > 0 and t_ip > 1.96)
    add("편입 뒤 개인 순매수 증가", "사후주 개인 순매수, 이벤트-대조 [거래대금 %]", d_ipt, t_ipt, d_ipt > 0 and t_ipt > 1.96)
    add("편입 뒤 개인 순매수 증가", "(사후주-형성주) 개인 순매수, 이벤트-대조 [거래량 %]", d_chg, t_chg, d_chg > 0 and t_chg > 1.96)
    d_r, t_r = row_of(t1, LABEL["ri_post"], "차이"), row_of(t1, LABEL["ri_post"], "t_날짜군집")
    add("편입 뒤 개인 순매수 증가", "사후주 개인 순매수 순위(0~100), 이벤트-대조 [거래량]", d_r, t_r, d_r > 0 and t_r > 1.96, "꼬리 두꺼움 점검")
    t4i = t4.set_index("설정")
    for sg, nm in (("cnn_tb", "시간 블록 변형"), ("cnn_w", "주 단위 학습")):
        r_ = t4i.loc[f"{sg}|vol|금지주제외=False"]
        add("(강건성) 편입 뒤 개인 순매수 증가", f"{nm} 신호, 사후주 개인 순매수 이벤트-대조 [거래량 %]", r_["차이개인사후주"],
            r_["t_차이개인사후주"], r_["차이개인사후주"] > 0 and r_["t_차이개인사후주"] > 1.96, "판정표 외 참고")
    v, t_ = row_of(t1, LABEL["car15"], "이벤트평균_전체"), row_of(t1, LABEL["car15"], "t_전체")
    add("CAR(1,5) > 0", "이벤트 CAR(1,5), 규모5분위 대비 %p", v, t_, v > 0 and t_ > 1.96)
    v, t_ = row_of(t1, LABEL["car15"], "차이"), row_of(t1, LABEL["car15"], "t_날짜군집")
    add("CAR(1,5) > 0", "이벤트-대조 CAR(1,5) %p", v, t_, v > 0 and t_ > 1.96)
    v, t_ = row_of(t1, LABEL["car620"], "이벤트평균_전체"), row_of(t1, LABEL["car620"], "t_전체")
    add("CAR(6,20) < 0", "이벤트 CAR(6,20), 규모5분위 대비 %p", v, t_, v < 0 and t_ < -1.96)
    v, t_ = row_of(t1, LABEL["car620"], "차이"), row_of(t1, LABEL["car620"], "t_날짜군집")
    add("CAR(6,20) < 0", "이벤트-대조 CAR(6,20) %p", v, t_, v < 0 and t_ < -1.96)
    v, t_ = row_of(t1, LABEL["car2160"], "이벤트평균_전체"), row_of(t1, LABEL["car2160"], "t_전체")
    add("(참고) CAR(21,60)", "이벤트 CAR(21,60), 규모5분위 대비 %p", v, t_, v < 0 and t_ < -1.96, "판정표 외 참고")
    hi = t6[(t6.개인비중분위 == "높음-낮음") & (t6.지표 == LABEL["car620"])].iloc[0]
    lv = t6[(t6.개인비중분위 == "높음") & (t6.지표 == LABEL["car620"])].iloc[0]
    add("개인 비중 높은 종목에서 반전이 더 큼", "CAR(6,20) 이벤트-대조의 (높음-낮음) %p", hi.차이, hi.t, hi.차이 < 0 and hi.t < -1.96,
        f"상대 격차만. 높음 구간 수준은 {lv.차이:+.2f}%p(t {lv.t:+.2f})로 유의하지 않음")
    c_, t_ = coef(main["reg"], "이벤트표본", "예", "indiv_post")
    add("개인 순매수가 많을수록 반전 (CAR(1,5) 통제)", "이벤트표본 회귀 개인 사후주 계수 (1SD당 %p)", c_, t_, c_ < 0 and t_ < -1.96)
    c_, t_ = coef(main["reg"], "이벤트표본", "아니오", "indiv_post")
    add("(참고) 같은 회귀, CAR(1,5) 미통제", "이벤트표본 회귀 개인 사후주 계수 (1SD당 %p)", c_, t_, c_ < 0 and t_ < -1.96)
    c_, t_ = coef(main["reg"], "이벤트표본", "예", "foreign_post")
    add("(대조) 외국인 순매수 뒤 반전 약함/없음", "이벤트표본 회귀 외국인 사후주 계수 (1SD당 %p)", c_, t_, None, "H2 대조군, 충족 여부 아닌 방향 참고")
    # ---- 판정표 다음 줄(유동성 공급·정보·위험 보상)에 대한 대조: 가격 압력이 안 맞을 때 어디로 기우는지 보기 위한 참고
    v, t_ = row_of(t1, LABEL["str5p"], "이벤트평균_전체"), row_of(t1, LABEL["str5p"], "t_전체")
    add("[유동성 공급] 편입 전 가격 하락", "이벤트 형성 주 수익률 str5 %p (0과 비교)", v, t_, v < 0 and t_ < -1.96, "판정표 참고")
    for who, c in (("외국인", "ff_form"), ("기관", "fn_form")):
        v, t_ = row_of(t1, LABEL[c], "이벤트평균_전체"), row_of(t1, LABEL[c], "t_전체")
        add("[유동성 공급] 편입 전 기관·외국인 순매도", f"이벤트 {who} 형성 주 순매수 (0과 비교) [거래량 %]", v, t_, v < 0 and t_ < -1.96, "판정표 참고")
    r_ = t7[(t7.주체 == "외국인") & (t7.결과 == LABEL["car620"]) & (t7.분위 == "T3-T1")].iloc[0]
    add("[정보] 외국인 순매수 주도 종목에서 수익 지속", "외국인 사후주 순매수 T3-T1 CAR(6,20) %p (양이어야 함)", r_.평균, r_.t, r_.평균 > 0 and r_.t > 1.96, "사후 분류, 참고")
    ver.append(dict(조건="팩터 통제 후 알파 유의한 음(-)", 측정="MKT·SMB·STR·MOM 회귀", 값=np.nan, t=np.nan, 판정="미검정",
                    비고="연구계획 12-7에서 수행. 이 파일럿은 규모5분위·str5·abvol 통제만 함"))
    t5 = pd.DataFrame(ver)
    t5.to_csv(f"{OUT}/r01_t5_verdict.csv", index=False, encoding="utf-8-sig")
    P(t5.round(3).to_string(index=False))

    # ---- CAR 경로 그림 + 경로 CSV
    P("\nCAR 경로 계산")
    b_path = pd.DataFrame(path, columns=[f"p{h}" for h in range(1, HORIZON + 1)], index=b.index)
    pcols = list(b_path.columns)
    bp = pd.concat([b, b_path], axis=1)
    paths = {}
    for sig, nm in (("cnn_d", "CNN 신규 편입"), ("str", "단기 반전 신규 편입"), ("res", "CNN 잔차 신규 편입")):
        d = bp[bp.k >= BURN].copy()
        ev = d["ev_" + sig].values
        E_, _, _ = match(d, ev, d["top_" + sig].values, "fi_form_t", pcols)
        ev_m, ct_m, df_m, df_lo, df_hi = [], [], [], [], []
        for h in pcols:
            m_, _, _, _, _ = cl_mean(d.loc[ev, h], d.loc[ev, "i"])
            mc_, _, _, _, _ = cl_mean(E_[h + "_c"], E_.i)
            md, sd, _, _, _ = cl_mean(E_[h + "_d"], E_.i)
            ev_m.append(m_); ct_m.append(mc_); df_m.append(md); df_lo.append(md - 1.96 * sd); df_hi.append(md + 1.96 * sd)
        paths[sig] = (nm, np.array(ev_m), np.array(ct_m), np.array(df_m), np.array(df_lo), np.array(df_hi))
        P(f"  {nm}: 이벤트 {int(ev.sum())}, 매칭 {len(E_)}")
    pd.DataFrame({f"{s}_{k}": v for s, (nm, *arr) in paths.items() for k, v in zip(["이벤트", "대조", "차이", "차이하한", "차이상한"], arr)},
                 index=range(1, HORIZON + 1)).to_csv(f"{OUT}/r01_car_paths.csv", encoding="utf-8-sig", index_label="이벤트일")
    plot_paths(paths)
    return b


def plot_paths(paths):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    col = {"cnn_d": "#2a78d6", "str": "#eb6834", "res": "#1baf7a"}
    ink, mute, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), facecolor="#fcfcfb")
    h = np.arange(0, HORIZON + 1)
    z0 = lambda v: np.r_[0.0, v]
    paths = {k: (v[0],) + tuple(z0(x) for x in v[1:]) for k, v in paths.items()}
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color("#c3c2b7")
        a.tick_params(colors=mute, length=0)
        a.axhline(0, color="#c3c2b7", lw=1)
        for v in (5, 20):
            a.axvline(v, color=grid, lw=1, ls=":")
        a.set_xlabel("이벤트 시간(거래일, 0=형성일)", color=mute)
    nm, e, c, dif, lo, hi = paths["cnn_d"]
    ax[0].plot(h, e, color=col["cnn_d"], lw=2, label="CNN 신규 편입 (규모5분위 대비)")
    ax[0].plot(h, c, color=col["cnn_d"], lw=2, ls="--", label="같은 칸 대조군")
    ax[0].set_title("누적 초과수익, CNN 이벤트와 대조군 (%p)", loc="left", color=ink, fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, labelcolor=mute)
    ends = sorted(((paths[k][3][-1], k) for k in paths))            # 끝점 라벨이 겹치지 않게 최소 간격을 둔다
    ypos, last = {}, -9
    for y, k in ends:
        last = max(y, last + 0.17)
        ypos[k] = last
    for sig in ("cnn_d", "str", "res"):
        nm, e, c, dif, lo, hi = paths[sig]
        ax[1].plot(h, dif, color=col[sig], lw=2)
        ax[1].fill_between(h, lo, hi, color=col[sig], alpha=0.08, lw=0)
        ax[1].text(HORIZON + 1.0, ypos[sig], nm, color=ink, fontsize=9, va="center")
    ax[1].set_title("누적 초과수익, 이벤트-대조군 차이와 95% 구간 (%p)", loc="left", color=ink, fontsize=11)
    ax[1].set_xlim(0, HORIZON + 22)
    fig.text(0.01, 0.01, "점선=5일·20일. 날짜 군집 95% 구간. 2011-01~2012-11 KOSPI 파일럿, 탐색용.", color=mute, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/r01_car_paths.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import time
    t0 = time.time()
    try:
        main()
    finally:
        P(f"\n소요 {time.time() - t0:.0f}초")
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/r01_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(LOG))

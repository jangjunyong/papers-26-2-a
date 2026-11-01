# -*- coding: utf-8 -*-
"""v01 r01 적대적 검증 (독립 재구현 + 군집 SE 대조 + 보고서/CSV 대조 + 감사 버그 영향 + 룩어헤드 교란 실험).

검증 대상: code/r01_rq2_pilot.py, results/r01_rq2_memo.md, results/r01_tables/*.csv  (읽기만 하고 수정하지 않는다)
입력: panel.parquet, results/kaggle/pred_d_lr1e-4.parquet, results/q01_panel_flags.parquet(B1~B5 행 플래그)
출력: results/v01_r01_review_log.txt (이 스크립트 stdout 전체). 보고서 results/v01_r01_review.md 는 이 로그를 보고 따로 쓴다.
실행: PYTHONIOENCODING=utf-8 python v01_r01_review.py   (노트북 CPU 수 분. r01 전체 재실행 포함)

독립 재구현 원칙: r01의 행렬 슬라이싱·groupby 루프 대신 (a) 종목-형성일 행을 h=1..60 오프셋 배열로 펼치고
(b) 이벤트는 종목별 파이썬 루프로 정의하고 (c) 대조군은 병합으로 만든다. 정의(상위 10분위, 신규편입 4주, 규모 5분위,
3분위 칸, MIN_CELL=3, 유효일수 80%)는 보고서 서술을 그대로 따른다.
"""
import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS"
B = f"{ROOT}/26-2/03_차트CNN_연구"
PANEL = f"{ROOT}/data/daily/club_db/panel.parquet"
KG = f"{B}/results/kaggle"
TAB = f"{B}/results/r01_tables"
FLAGS = f"{B}/results/q01_panel_flags.parquet"
R01 = f"{B}/code/r01_rq2_pilot.py"
SCR = "C:/Users/jacob/AppData/Local/Temp/claude/C--Users-jacob-Documents-Obsidian-Vault-PAPERS/2bbaf99c-976a-49cf-b4dc-ac8c0278e5f3/scratchpad"
LAST = "2013-06-30"
H = 60
LOG = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def sec(t):
    P("\n" + "=" * 100 + f"\n## {t}\n" + "=" * 100)


# ============================================================================ 데이터
class Ctx:
    pass


def load_ctx():
    """패널을 (일 x 종목) 행렬로, pred를 종목-형성일 행으로. 행렬은 np.searchsorted 인덱싱으로 직접 채운다(pivot 미사용)."""
    c = Ctx()
    cols = ["ticker", "date", "adj_close", "ret1d", "ret1d_clean", "ret_outlier", "mktcap", "volume", "is_halted",
            "indiv_net", "inst_net", "foreign_net", "other_corp_net", "indiv_flow", "inst_flow", "foreign_flow",
            "indiv_flow_w", "inst_flow_w", "foreign_flow_w", "indiv_flow_tv", "str5", "abvol", "mom12_1", "near52h",
            "indiv_buy", "indiv_sell"]
    pn = pd.read_parquet(PANEL, columns=cols)
    c.full_flow = {k: pn[k].values for k in ("indiv_flow", "inst_flow", "foreign_flow")}   # 전체 표본(2010~2026) 윈저화 경계 점검용
    c.full_dates = pn.date.values
    c.full_q = {k: pn[k].quantile([0.005, 0.995]).values for k in ("indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv")}
    pn = pn[pn.date <= LAST].copy()
    fl = pd.read_parquet(FLAGS)
    pn = pn.merge(fl[["ticker", "date", "f_resume_day", "f_liquidation_zone", "f_zero_inv_suspect", "f_prelist_otc"]],
                  on=["ticker", "date"], how="left")
    c.cal = np.sort(pn.date.unique())
    c.tick = np.sort(pn.ticker.unique())
    di = np.searchsorted(c.cal, pn.date.values)
    ti = np.searchsorted(c.tick, pn.ticker.values)
    nd, nt = len(c.cal), len(c.tick)

    def mat(col):
        M = np.full((nd, nt), np.nan)
        M[di, ti] = pn[col].values.astype(float)
        return M

    c.M = {k: mat(k) for k in cols[2:] + ["f_resume_day", "f_liquidation_zone", "f_zero_inv_suspect", "f_prelist_otc"]}
    # 4그룹 합계 불일치(항등식 위반): 4열이 모두 있을 때만
    nets = ["indiv_net", "inst_net", "foreign_net", "other_corp_net"]
    allok = pn[nets].notna().all(axis=1)
    resid = pn[nets].sum(axis=1).where(allok)
    M = np.full((nd, nt), np.nan)
    M[di, ti] = (resid != 0).where(allok).astype(float).values
    c.M["inv_mismatch"] = M
    M = np.full((nd, nt), np.nan)
    allzero = pn[nets].eq(0).all(axis=1) & (pn.volume > 0)
    M[di, ti] = allzero.astype(float).values
    c.M["allzero_broad"] = M
    c.nd, c.nt = nd, nt
    c.pn_len = len(pn)
    return c


def load_pred(c, name="pred_d_lr1e-4"):
    p = pd.read_parquet(f"{KG}/{name}.parquet")
    p["date"] = pd.to_datetime(p["date"]).astype("datetime64[ns]")
    p = p.sort_values(["date", "ticker"]).reset_index(drop=True)
    p["i"] = np.searchsorted(c.cal, p.date.values)
    assert (c.cal[p.i.values] == p.date.values).all()
    p["j"] = np.searchsorted(c.tick, p.ticker.values)
    assert (c.tick[p.j.values] == p.ticker.values).all()
    fd = np.sort(p.i.unique())
    p["k"] = np.searchsorted(fd, p.i.values)
    return p


# ============================================================================ 이벤트 정의 (종목별 루프)
def events_loop(p, burn=5, hist=4, top_q=0.90):
    """상위 10분위(날짜별 순위 백분위 > 0.9) 신규 편입. 종목별로 상위 주 집합을 만들고 진입·이력을 센다."""
    pct = p.groupby("k").prob.rank(method="first", pct=True).values
    top = pct > top_q
    ev = np.zeros(len(p), bool)
    ent_flag = np.zeros(len(p), bool)
    for j, idx in pd.Series(np.arange(len(p))).groupby(p.j.values):
        idx = idx.values
        ks = p.k.values[idx]
        kmap = dict(zip(ks.tolist(), idx.tolist()))
        tops = set(ks[top[idx]].tolist())
        entries = {k for k in tops if k > 0 and (k - 1) not in tops}
        for k in entries:
            ent_flag[kmap[k]] = True
            if k >= burn and not any((k - m) in entries for m in range(1, hist + 1)):
                ev[kmap[k]] = True
    return top, ev


# ============================================================================ CAR·흐름 (오프셋 배열)
def tile(x, by, n, rule):
    """날짜별 n분위. rule='floor'는 floor((rank-1)/N*n)(r01 규모 5분위), 'ceil'은 ceil(rank/N*n)-1(r01 3분위)."""
    s = pd.Series(x)
    r = s.groupby(by).rank(method="first")
    N = s.groupby(by).transform("count")
    if rule == "floor":
        return np.floor((r - 1) / N * n).values
    return np.minimum(np.ceil(r / N * n) - 1, n - 1).values


def window(Mx, i, j, a, b):
    """행 i, 열 j 기준 상대일 a..b 값 (n x (b-a+1)). 범위 밖은 NaN."""
    off = np.arange(a, b + 1)
    ii = i[:, None] + off[None, :]
    ok = (ii >= 0) & (ii < Mx.shape[0])
    out = np.full(ii.shape, np.nan)
    out[ok] = Mx[ii[ok], np.broadcast_to(j[:, None], ii.shape)[ok]]
    return out


def wsum_ind(blk, minv=0.8):
    cnt = np.isfinite(blk).sum(1)
    s = np.nansum(blk, 1)
    s[cnt < int(np.ceil(minv * blk.shape[1]))] = np.nan
    return s


def build_outcomes(c, p, retkey="ret1d_clean", size_rule="floor", size_lag=1, universe="pred", minv=0.8):
    """CAR(1,5), (6,20), (21,60)(%p). 벤치마크 = 같은 형성일 코호트의 같은 규모 5분위의 일별 평균 수익률."""
    i, j, k = p.i.values, p.j.values, p.k.values
    mc = c.M["mktcap"][i - size_lag, j]
    q = tile(mc, k, 5, size_rule)
    q[~np.isfinite(mc)] = np.nan
    RET = window(c.M[retkey], i, j, 1, H)
    if universe == "pred":
        df = pd.DataFrame(RET)
        df["k"], df["q"] = k, q
        bench = df.groupby(["k", "q"]).transform("mean").values
    else:      # 패널 전체 종목으로 벤치마크: 같은 날 mktcap(t-1) 5분위 구성도 패널 전체에서 만든다
        bench = np.full(RET.shape, np.nan)
        for ii in np.unique(i):
            mcall = c.M["mktcap"][ii - size_lag]
            okall = np.isfinite(mcall)
            rk = pd.Series(mcall[okall]).rank(method="first").values
            qa = np.full(len(mcall), np.nan)
            qa[okall] = np.floor((rk - 1) / okall.sum() * 5)
            Rb = c.M[retkey][ii + 1: ii + 1 + H]
            bm = np.full((H, 5), np.nan)
            for g in range(5):
                m = qa == g
                bm[:, g] = np.nanmean(Rb[:, m], axis=1)
            rows = np.where(i == ii)[0]
            qq = q[rows]
            for r_, g_ in zip(rows, qq):
                if np.isfinite(g_):
                    bench[r_] = bm[:, int(g_)]
    AR = RET - bench
    out = {}
    for nm, (a, b) in {"car15": (1, 5), "car620": (6, 20), "car2160": (21, 60)}.items():
        out[nm] = wsum_ind(AR[:, a - 1:b], minv) * 100
    out["q_size"] = q
    out["RET"] = RET
    return out


def build_flows(c, p, key_i="indiv_flow_w", key_f="foreign_flow_w", key_n="inst_flow_w"):
    i, j = p.i.values, p.j.values
    o = {}
    for tag, key in (("fi", key_i), ("ff", key_f), ("fn", key_n)):
        Mx = c.M[key] if isinstance(key, str) else key
        o[tag + "_form"] = wsum_ind(window(Mx, i, j, -4, 0)) * 100
        o[tag + "_post"] = wsum_ind(window(Mx, i, j, 1, 5)) * 100
    return o


# ============================================================================ 통계
def cl_t(x, g):
    """평균, 군집(1-way) SE, t, n, G.  V = G/(G-1) * sum_g (sum_i u_i)^2 / n^2"""
    x = np.asarray(x, float)
    g = np.asarray(g)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum()
    G = len(s)
    se = np.sqrt(G / (G - 1) * (s.values ** 2).sum()) / n
    return m, se, m / se, n, G


def ols_cluster(y, X, groups, corr=True):
    """OLS + 1·2-way 군집 공분산을 직접 구현. groups: 1열(1-way) 또는 2열(2-way, Cameron-Gelbach-Miller).
    소표본 보정은 statsmodels와 같은 G/(G-1) * (N-1)/(N-K)."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    N, K = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]

    def meat(gr):
        codes = pd.factorize(gr)[0]
        G = codes.max() + 1
        S = np.zeros((G, K))
        np.add.at(S, codes, Xu)
        c = (G / (G - 1)) * ((N - 1) / (N - K)) if corr else 1.0
        return c * (S.T @ S)

    groups = np.asarray(groups)
    if groups.ndim == 1:
        V = XtXi @ meat(groups) @ XtXi
    else:
        g1, g2 = groups[:, 0], groups[:, 1]
        pair = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
        V = XtXi @ (meat(g1) + meat(g2) - meat(pair)) @ XtXi
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se, V


def sm_cluster(y, X, groups):
    Xc = sm.add_constant(np.asarray(X, float), has_constant="add")
    g = np.asarray(groups)
    return sm.OLS(np.asarray(y, float), Xc).fit(cov_type="cluster", cov_kwds={"groups": g})


# ============================================================================ 매칭 분석 (독립 구현)
def base_table(c, p, flows_override=None, size_rule="floor", terc_rule="ceil", retkey="ret1d_clean", size_lag=1,
               universe="pred", minv=0.8):
    """종목-형성일 표: 칸 변수(형성일 t까지)와 결과(t+1부터)."""
    i, j, k = p.i.values, p.j.values, p.k.values
    d = pd.DataFrame({"i": i, "j": j, "k": k, "date": p.date.values, "ticker": p.ticker.values})
    fl = flows_override if flows_override is not None else build_flows(c, p)
    o = build_outcomes(c, p, retkey=retkey, size_rule=size_rule, size_lag=size_lag, universe=universe, minv=minv)
    for kk in ("car15", "car620", "car2160", "q_size"):
        d[kk] = o[kk]
    for kk, v in fl.items():
        d[kk] = v
    d["str5"] = c.M["str5"][i, j]
    d["abvol"] = c.M["abvol"][i, j]
    d["str5_t"] = tile(d.str5.values, k, 3, terc_rule)
    d["abvol_t"] = tile(d.abvol.values, k, 3, terc_rule)
    d["fi_form_t"] = tile(d.fi_form.values, k, 3, terc_rule)
    return d, o


def matched(d, ev, cols, keep=None, need_car15=True, min_cell=3, cell_extra=None):
    """이벤트별 같은 칸(날짜 x 규모5 x str5 3분위 x abvol 3분위 x 형성 주 개인 흐름 3분위) 비이벤트 평균을 뺀 차이."""
    cell = ["i", "q_size", "str5_t", "abvol_t", "fi_form_t"] + (cell_extra or [])
    keyok = d[cell].notna().all(axis=1).values
    keep = np.ones(len(d), bool) if keep is None else keep
    ctrl = (~ev) & keyok & keep
    if need_car15:
        ctrl &= d.car15.notna().values
    cm = d[ctrl].groupby(cell)[cols].mean()
    cm.columns = [x + "_c" for x in cm.columns]
    cn = d[ctrl].groupby(cell).size().rename("n_ctrl")
    E = d[ev & keyok & keep].join(cm, on=cell).join(cn, on=cell)
    E["n_ctrl"] = E.n_ctrl.fillna(0)
    E = E[E.n_ctrl >= min_cell].copy()
    for x in cols:
        E[x + "_d"] = E[x] - E[x + "_c"]
    return E, int((ev & keyok & keep).sum()), int(ctrl.sum())


def headline(d, ev, keep=None, **kw):
    """r01 표 1의 핵심 행. (평균, t, N)"""
    cols = ["car15", "car620", "car2160", "fi_form", "fi_post", "ff_post", "fn_post"]
    E, nk, nc = matched(d, ev, cols, keep=keep, **kw)
    out = {"n_ev": int((ev if keep is None else ev & keep).sum()), "n_match": len(E), "med_cell": float(E.n_ctrl.median())}
    for x in cols:
        m, se, t, n, G = cl_t(E[x + "_d"], E.i)
        out[x] = (m, t, n)
    de = d[ev if keep is None else ev & keep]
    for x in ("car15", "car620"):
        m, se, t, n, G = cl_t(de[x], de.i)
        out[x + "_all"] = (m, t, n)
    return out, E


# ============================================================================ 회귀·HAC 재현
def reg_inputs(c, p, d, flows):
    """r01 표 2의 설명변수를 처음부터 만든다. 표준화는 분석 표본(k>=5) 전 종목-주 기준 (r01과 같은 서술)."""
    abv = c.M["abvol"][p.i.values, p.j.values]
    lo, hi = np.nanquantile(abv, [0.01, 0.99])         # r01은 pred 전 기간(형성일 100개) 분위수
    d = d.copy()
    d["abvol_w"] = np.clip(abv, lo, hi)
    d["logmc"] = np.log(c.M["mktcap"][p.i.values, p.j.values])
    return d, (lo, hi)


def zs(x):
    x = pd.Series(x)
    return ((x - x.mean()) / x.std()).values


def reg_event(d, mask, with_car=True):
    cols = {"indiv_post": zs(d.fi_post), "foreign_post": zs(d.ff_post), "inst_post": zs(d.fn_post)}
    if with_car:
        cols["car15"] = zs(d.car15)
    cols.update({"indiv_form": zs(d.fi_form), "foreign_form": zs(d.ff_form), "inst_form": zs(d.fn_form),
                 "str5": zs(d.str5), "logmc": zs(d.logmc), "abvol": zs(d.abvol_w)})
    X = pd.DataFrame(cols, index=d.index)
    y = d.car620
    ok = mask & y.notna().values & X.notna().all(axis=1).values
    return y[ok].values, X[ok], d[ok]


def nw_mean_t(x, g, lags=(0, 2, 4, 8)):
    """풀링 평균의 주 단위 합계 s_w에 대한 Newey-West(Bartlett) t. g는 연속 주 번호(형성일 순서)."""
    x = np.asarray(x, float)
    g = np.asarray(g)
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum()
    s = s.reindex(range(int(g.min()), int(g.max()) + 1), fill_value=0.0).values
    out = {}
    for L in lags:
        V = (s ** 2).sum()
        for l in range(1, L + 1):
            V += 2 * (1 - l / (L + 1)) * (s[l:] * s[:-l]).sum()
        out[L] = m / (np.sqrt(max(V, 1e-18)) / n)
    return m, out


# ============================================================================ 섹션 함수
def load_r01():
    spec = importlib.util.spec_from_file_location("r01_mod", R01)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def s_rerun_r01(r01):
    """r01 전체를 임시 폴더로 다시 돌려 기존 CSV와 숫자가 같은지 본다(오래된 CSV 점검)."""
    sec("0. r01 재실행 일치 (CSV가 현재 코드의 출력인지)")
    out = f"{SCR}/r01_rerun"
    os.makedirs(out, exist_ok=True)
    r01.OUT = out
    import io
    import contextlib
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r01.main()
    P(f"r01.main() 재실행 {time.time() - t0:.0f}초")
    bad = []
    names = sorted(f for f in os.listdir(TAB) if f.endswith(".csv"))
    for f in names:
        a = pd.read_csv(f"{TAB}/{f}", encoding="utf-8-sig")
        b = pd.read_csv(f"{out}/{f}", encoding="utf-8-sig")
        if a.shape != b.shape:
            bad.append((f, "shape", a.shape, b.shape))
            continue
        num = a.select_dtypes("number").columns
        dmax = float(np.nanmax(np.abs(a[num].values - b[num].values))) if len(num) else 0.0
        obj = a.select_dtypes(exclude="number").columns
        same_obj = all(a[x].fillna("").equals(b[x].fillna("")) for x in obj)
        P(f"  {f:45s} shape {a.shape}  최대|차이| {dmax:.2e}  문자열열 일치 {same_obj}")
        if dmax > 1e-9 or not same_obj:
            bad.append((f, dmax, same_obj))
    P("불일치:", bad if bad else "없음 (모든 CSV가 현재 코드 재실행과 동일)")
    return bad


def s_independent(c, p):
    sec("2. 독립 재구현 대조")
    top, ev = events_loop(p)
    d, o = base_table(c, p)
    samp = d.k.values >= 5
    ds = d[samp].reset_index(drop=True)
    evs = ev[samp]
    wk = pd.Series(p.k.values[ev]).value_counts()
    P(f"(a) 이벤트 수 {int(ev.sum())} (r01 4,368), 주당 {wk.min()}~{wk.max()}, 평균 {ev.sum() / ds.k.nunique():.2f}, "
      f"종목 {p.ticker[ev].nunique()} (r01 755), 분석 형성일 {ds.k.nunique()} (r01 95), 종목-주 {len(ds)} (r01 67,629)")
    h, E = headline(ds, evs)
    rows = []
    for nm, lab, ref in (("car15_all", "(b) CAR(1,5) 이벤트 평균(규모5분위 대비)", (0.587, 5.874, 4359)),
                         ("car620_all", "(b) CAR(6,20) 이벤트 평균", (0.160, 0.987, 4345))):
        rows.append((lab, h[nm][0], h[nm][1], h[nm][2], ref))
    for nm, lab, ref in (("car15", "CAR(1,5) 이벤트-대조", (0.555, 4.528, 3669)), ("car620", "CAR(6,20) 이벤트-대조", (0.086, 0.438, 3656)),
                         ("car2160", "CAR(21,60) 이벤트-대조", (0.403, 1.231, 3643)), ("fi_form", "개인 형성주 이벤트-대조", (0.205, 0.178, 3676)),
                         ("fi_post", "개인 사후주(1~5) 이벤트-대조", (3.738, 2.681, 3676)), ("ff_post", "외국인 사후주 이벤트-대조", (-0.712, -0.680, 3676)),
                         ("fn_post", "기관 사후주 이벤트-대조", (-1.584, -1.307, 3676))):
        rows.append((lab, h[nm][0], h[nm][1], h[nm][2], ref))
    de = ds[evs]
    m_, se_, t_, n_, G_ = cl_t(de.fi_post, de.i)
    rows.append(("(c) 개인 사후주(1~5) 이벤트 평균 [flow_w합x100]", m_, t_, n_, (1.810, 1.089, 4368)))
    P(f"{'항목':50s}{'내 값':>9s}{'내 t':>8s}{'내 N':>7s} | {'r01 값':>8s}{'r01 t':>8s}{'r01 N':>7s} | 차이(값/t)")
    for lab, m, t, n, ref in rows:
        P(f"{lab:50s}{m:9.3f}{t:8.3f}{n:7d} | {ref[0]:8.3f}{ref[1]:8.3f}{ref[2]:7d} | {m - ref[0]:+.4f}/{t - ref[1]:+.4f}")
    P(f"매칭 이벤트 {h['n_match']} (r01 3,676), 칸 중앙값 {h['med_cell']}")
    return top, ev, d, ds, evs, h, E


def s_sensitivity(c, p, ev, samp):
    sec("2b. 정의·표본 민감도 (독립 구현으로 r01 숫자가 얼마나 흔들리나)")
    rows = []

    def run(tag, flows=None, keep=None, mkw=None, **kw):
        d, o = base_table(c, p, flows_override=flows, **kw)
        ds = d[samp].reset_index(drop=True)
        evs = ev[samp]
        kp = None if keep is None else keep[samp]
        h, E = headline(ds, evs, keep=kp, **(mkw or {}))
        rows.append((tag, h["n_ev"], h["n_match"], h["car15_all"], h["car620_all"], h["fi_post"], h["car15"], h["car620"], h["car2160"]))
        return h

    run("기준(r01 동일 정의)")
    run("분위 경계 규칙 교체(규모 ceil / 3분위 floor)", size_rule="ceil", terc_rule="floor")
    run("규모 5분위를 형성일 t 시총으로(t-1 아님)", size_lag=0)
    run("벤치마크 = 패널 전체 종목의 규모 5분위", universe="panel")
    run("대조군의 CAR(1,5) 유효 조건 제거(결과 조건화 해제)", mkw=dict(need_car15=False))
    run("MIN_CELL 1", mkw=dict(min_cell=1))
    run("MIN_CELL 5", mkw=dict(min_cell=5))
    run("유효일 100% 요구", minv=1.0)
    m10 = c.full_dates < np.datetime64("2011-01-01")
    fl = {}
    for tag, k in (("fi", "indiv_flow"), ("ff", "foreign_flow"), ("fn", "inst_flow")):
        lo, hi = np.nanquantile(c.full_flow[k][m10], [0.005, 0.995])
        P(f"  윈저 경계 {k}: 전체표본 {np.round(c.full_q[k], 3)} vs 2010년만 ({lo:.3f}, {hi:.3f})")
        fl[tag] = np.clip(c.M[k], lo, hi)
    run("윈저화 경계를 2010년 자료로만(룩어헤드 제거)", flows=build_flows(c, p, fl["fi"], fl["ff"], fl["fn"]))
    RES = np.nan_to_num(c.M["f_resume_day"])
    c.M["ret_noresume"] = np.where(RES == 1, np.nan, c.M["ret1d_clean"])
    run("B1: 거래재개일 수익을 결측 처리", retkey="ret_noresume")
    run("B2: ret1d 사용(정리매매·이상값 포함)", retkey="ret1d")
    P(f"{'설정':52s}{'N매칭':>6s} | {'CAR15 전체(t)':>16s} {'CAR620 전체(t)':>16s} | {'개인사후 차이(t)':>17s} | {'CAR15 차(t)':>13s} {'CAR620 차(t)':>13s} {'CAR2160 차(t)':>14s}")
    f2 = lambda x: f"{x[0]:6.2f}({x[1]:5.2f})"
    for tag, ne, nm, a15, a620, fi, d15, d620, d60 in rows:
        P(f"{tag:52s}{nm:6d} | {f2(a15):>16s} {f2(a620):>16s} | {f2(fi):>17s} | {f2(d15):>13s} {f2(d620):>13s} {f2(d60):>14s}")
    return rows


def s_cluster(c, p, d, ev, top, r01):
    sec("3. 군집 표준오차 구현 대조")
    rng = np.random.default_rng(7)
    for lab, g in (("균형 40군집", rng.integers(0, 40, 500)),
                   ("불균형 12군집(크기 편차 큼)", rng.choice(12, 500, p=np.r_[0.4, 0.2, 0.1, 0.1, np.full(8, 0.2 / 8)]))):
        x = rng.normal(size=500) + 0.3 * rng.normal(size=12)[g % 12]
        a = r01.cl_mean(x, g)
        b = cl_t(x, g)
        s = sm.OLS(x, np.ones(500)).fit(cov_type="cluster", cov_kwds={"groups": g})
        P(f"  평균 [{lab}] se: r01 {a[1]:.10f} / 내구현 {b[1]:.10f} / statsmodels {s.bse[0]:.10f}  t: {a[2]:.6f} {b[2]:.6f} {s.tvalues[0]:.6f}")
        assert abs(a[1] - s.bse[0]) < 1e-9 and abs(b[1] - s.bse[0]) < 1e-9
    n = 1500
    gd = rng.integers(0, 30, n)
    gt = rng.integers(0, 80, n)
    Xs = rng.normal(size=(n, 3))
    y = Xs @ np.array([0.5, 0, -0.3]) + rng.normal(size=30)[gd] + rng.normal(size=80)[gt] + rng.normal(size=n)
    for lab, grp in (("1-way 날짜", gd), ("2-way 날짜x종목", np.column_stack([gd, gt]))):
        r_sm = sm_cluster(y, Xs, grp)
        r_01 = r01.fit_cluster(y, Xs, gd) if grp.ndim == 1 else r01.fit_cluster(y, Xs, gd, gt)
        b_, se_, t_, V_ = ols_cluster(y, Xs, grp)
        P(f"  회귀 [{lab}] 최대|t 차이| 내구현-statsmodels {np.abs(t_ - r_sm.tvalues).max():.2e}, r01.fit_cluster-statsmodels "
          f"{np.abs(r_01.tvalues - r_sm.tvalues).max():.2e}, 공분산 최소 고유값 {np.linalg.eigvalsh(V_).min():.4f}")
    kk = d.k.values >= 5
    dfull, (lo, hi) = reg_inputs(c, p, d, None)
    ds = dfull[kk].reset_index(drop=True)
    evs = ev[kk]
    tops = top[kk]
    y, X, dd = reg_event(ds, evs, True)
    g2 = np.column_stack([dd.i.values, dd.j.values])
    b_, se_, t_, V_ = ols_cluster(y, X.values, g2)
    r_sm = sm_cluster(y, X.values, g2)
    csv = pd.read_csv(f"{TAB}/r01_t2_regressions.csv", encoding="utf-8-sig")
    sel = csv[(csv.개인척도 == "거래량") & (csv.회귀 == "이벤트표본") & (csv.CAR15통제 == "예")].set_index("항")
    P(f"  이벤트 표본 회귀 N={len(y)} (r01 CSV N={int(sel.N.iloc[0])}), 날짜 {dd.i.nunique()} 종목 {dd.j.nunique()}, abvol 윈저 경계 {lo:.3f}/{hi:.3f}, "
      f"2-way 공분산 최소 고유값 {np.linalg.eigvalsh(V_).min():.4f}")
    P(f"  {'항':14s}{'내 계수':>9s}{'내 t(2way)':>11s}{'sm t':>8s}{'1way 날짜 t':>12s} | {'r01 계수':>9s}{'r01 t':>8s}")
    t1w = ols_cluster(y, X.values, dd.i.values)[2]
    for q, nm in enumerate(["const"] + list(X.columns)):
        if nm in sel.index:
            P(f"  {nm:14s}{b_[q]:9.4f}{t_[q]:11.4f}{r_sm.tvalues[q]:8.4f}{t1w[q]:12.4f} | {sel.loc[nm, '계수']:9.4f}{sel.loc[nm, 't']:8.4f}")
    cols = {"indiv_post": zs(ds.fi_post), "foreign_post": zs(ds.ff_post), "inst_post": zs(ds.fn_post), "car15": zs(ds.car15),
            "indiv_form": zs(ds.fi_form), "foreign_form": zs(ds.ff_form), "inst_form": zs(ds.fn_form), "str5": zs(ds.str5),
            "logmc": zs(ds.logmc), "abvol": zs(ds.abvol_w)}
    X2 = pd.DataFrame(cols, index=ds.index)
    X2["D"] = tops.astype(float)
    for t in ("indiv_post", "foreign_post", "inst_post"):
        X2["D_x_" + t] = X2[t] * X2["D"]
    y2 = ds.car620
    ok = y2.notna().values & X2.notna().all(axis=1).values
    g2 = np.column_stack([ds.i.values[ok], ds.j.values[ok]])
    b2, se2, t2, V2 = ols_cluster(y2[ok].values, X2[ok].values, g2)
    names = ["const"] + list(X2.columns)
    sel2 = csv[(csv.개인척도 == "거래량") & (csv.회귀 == "전종목주+상위10분위더미") & (csv.CAR15통제 == "예")].set_index("항")
    for t in ("indiv_post", "foreign_post", "inst_post"):
        a, b = names.index(t), names.index("D_x_" + t)
        s_ = b2[a] + b2[b]
        se_s = np.sqrt(V2[a, a] + V2[b, b] + 2 * V2[a, b])
        P(f"  전종목+상위더미 합계_{t}: 내 {s_:.4f} (t {s_ / se_s:.4f}) / r01 {sel2.loc['합계_' + t, '계수']:.4f} "
          f"(t {sel2.loc['합계_' + t, 't']:.4f})  N 내 {ok.sum()} r01 {int(sel2.N.iloc[0])}")
    h, E = headline(ds, evs)
    P("  풀링 평균 t: 날짜 군집 vs 주 단위 Newey-West(Bartlett) lag 0/2/4/8")
    for nm, lab in (("fi_post", "개인 사후주 이벤트-대조"), ("car15", "CAR(1,5) 이벤트-대조"), ("car620", "CAR(6,20) 이벤트-대조"),
                    ("car2160", "CAR(21,60) 이벤트-대조")):
        m, out = nw_mean_t(E[nm + "_d"], E.k)
        P(f"    {lab:28s} 평균 {m:7.3f}  t(lag0={out[0]:.2f}, 2={out[2]:.2f}, 4={out[4]:.2f}, 8={out[8]:.2f})   날짜군집 t={cl_t(E[nm + '_d'], E.i)[2]:.2f}")
    return ds, evs


def s_signals_overlap(c, p, samp):
    sec("4b. 신호 변형 간 이벤트 겹침 (개인 순매수 부호 뒤집힘의 원인 탐색)")
    evs = {}
    probs = {}
    for nm, f in (("cnn_d", "pred_d_lr1e-4"), ("cnn_tb", "pred_d_lr1e-4_tb"), ("cnn_w", "pred_w_lr1e-4")):
        q = load_pred(c, f)
        assert (q.i.values == p.i.values).all() and (q.j.values == p.j.values).all()
        probs[nm] = q.prob.values
        evs[nm] = events_loop(q)[1]
    from scipy.stats import spearmanr
    for a, b in (("cnn_d", "cnn_tb"), ("cnn_d", "cnn_w")):
        both = evs[a] & evs[b]
        P(f"  {a} 이벤트 {int(evs[a].sum())}, {b} 이벤트 {int(evs[b].sum())}, 교집합 {int(both.sum())} "
          f"(a의 {both.sum() / evs[a].sum():.1%}, Jaccard {both.sum() / (evs[a] | evs[b]).sum():.1%}); prob 순위상관(전체 행) "
          f"{spearmanr(probs[a], probs[b])[0]:.3f}")
    d, o = base_table(c, p)
    dsx = d[samp].reset_index(drop=True)
    for tag, mask in (("cnn_d 이벤트 중 cnn_tb 이벤트이기도 함", evs["cnn_d"] & evs["cnn_tb"]),
                      ("cnn_d 이벤트 중 cnn_tb 이벤트가 아님", evs["cnn_d"] & ~evs["cnn_tb"]),
                      ("cnn_tb 이벤트 중 cnn_d 이벤트가 아님", evs["cnn_tb"] & ~evs["cnn_d"])):
        h, E = headline(dsx, mask[samp])
        P(f"  {tag:40s} N이벤트 {h['n_ev']:5d} 매칭 {h['n_match']:5d} | 개인 사후주 차이 {h['fi_post'][0]:6.2f} "
          f"(t {h['fi_post'][1]:5.2f}) | CAR(1,5) 차 {h['car15'][0]:5.2f} (t {h['car15'][1]:5.2f})")


def s_b5(c, p, ev, samp):
    sec("6. 감사 버그 B5(투자자 열) 영향 추정 + B1·B2·B6 간이 점검")
    i, j = p.i.values, p.j.values
    dates = pd.to_datetime(p.date.values)
    pre = dates <= pd.Timestamp("2012-05-02")
    Z = np.nan_to_num(c.M["f_zero_inv_suspect"])
    Zb = np.nan_to_num(c.M["allzero_broad"])
    MM = c.M["inv_mismatch"]
    wins = {"형성 주(-4~0)": (-4, 0), "사후 주(1~5)": (1, 5), "형성+사후(-4~5)": (-4, 5)}
    P("[표본 크기] r01 분석 형성일 중 2012-05-02 이전(4그룹 합계가 안 닫히는 구간): "
      f"{p.k[pre & samp].nunique()} / {p.k[samp].nunique()}주, 이벤트 {int((ev & pre).sum())} / {int(ev.sum())} ({(ev & pre).sum() / ev.sum():.1%})")
    P(f"{'창':16s}{'지표':34s}{'전 종목-주 (k>=5)':>20s}{'이벤트':>10s}")
    for wn, (a, b) in wins.items():
        for lab, M, kind in (("f_zero_inv_suspect=1 종목-일 비율", Z, "share"), ("전부0 & 거래량>0(넓은 정의) 종목-일 비율", Zb, "share"),
                             ("4그룹 합계≠0 종목-일 비율(유효 행 중)", MM, "share"),
                             ("f_zero_inv_suspect 일이 1일 이상인 종목-주 비율", Z, "any"), ("합계≠0 일이 1일 이상인 종목-주 비율", np.nan_to_num(MM), "any")):
            w = window(M, i, j, a, b)
            if kind == "share":
                v = np.nanmean(w[samp]), np.nanmean(w[ev])
            else:
                v = (np.nansum(w, 1)[samp] > 0).mean(), (np.nansum(w, 1)[ev] > 0).mean()
            P(f"{wn:16s}{lab:34s}{v[0]:20.4f}{v[1]:10.4f}")
    # 재추정
    P("\n[개인 사후 주 순매수 이벤트-대조 차이: B5 대응 변형별]")
    P(f"{'변형':58s}{'N이벤트':>8s}{'N매칭':>7s}{'차이':>8s}{'t(날짜)':>9s}{'CAR15차':>9s}{'CAR620차':>9s}")
    out = []

    def run(tag, flows=None, keep=None):
        d, o = base_table(c, p, flows_override=flows)
        ds = d[samp].reset_index(drop=True)
        kp = None if keep is None else keep[samp]
        h, E = headline(ds, ev[samp], keep=kp)
        out.append((tag, h))
        P(f"{tag:58s}{h['n_ev']:8d}{h['n_match']:7d}{h['fi_post'][0]:8.2f}{h['fi_post'][1]:9.2f}{h['car15'][0]:9.2f}{h['car620'][0]:9.2f}")

    run("기준(r01 동일)")
    anyz = window(Z, i, j, -4, 5).sum(1) > 0
    run("f_zero_inv_suspect 일이 있는 종목-주 제외(이벤트·대조 모두)", keep=~anyz)
    fl = build_flows(c, p, *(np.where(Z == 1, np.nan, c.M[k]) for k in ("indiv_flow_w", "foreign_flow_w", "inst_flow_w")))
    run("f_zero_inv_suspect 일의 흐름을 결측 처리(3분위 재계산)", flows=fl)
    fl2 = build_flows(c, p, *(np.where(Zb == 1, np.nan, c.M[k]) for k in ("indiv_flow_w", "foreign_flow_w", "inst_flow_w")))
    run("전부0&거래량>0 일의 흐름을 결측 처리(넓은 정의)", flows=fl2)
    run("형성일 <= 2012-05-02 만 (4그룹 합계 불일치 구간)", keep=pre)
    run("형성일 > 2012-05-02 만 (합계 닫히는 구간)", keep=~pre)
    mmw = window(np.nan_to_num(MM), i, j, -4, 5).sum(1) > 0
    run("4그룹 합계≠0 일이 있는 종목-주 제외", keep=~mmw)
    P(f"  (4그룹 합계≠0 일이 있는 종목-주 비율 {mmw[samp].mean():.1%})")
    for nm, key in (("거래재개일(B1)", "f_resume_day"), ("정리매매 구간(B2)", "f_liquidation_zone"), ("ret_outlier", "ret_outlier")):
        M_ = np.nan_to_num(c.M[key])
        w20, w60 = window(M_, i, j, 1, 20), window(M_, i, j, 1, 60)
        P(f"  {nm:16s} 사후 1~20일 창에 1일 이상: 전 종목-주 {(np.nansum(w20, 1)[samp] > 0).mean():.4f} / 이벤트 {(np.nansum(w20, 1)[ev] > 0).mean():.4f}; "
          f"1~60일 창: {(np.nansum(w60, 1)[samp] > 0).mean():.4f} / {(np.nansum(w60, 1)[ev] > 0).mean():.4f}")
    # 이벤트 표본 회귀(표 2 둘째 줄)를 기간별로: 2012-05-02 이전엔 foreign_net 정의가 다르다(기타외국인 제외 추정)
    d, o = base_table(c, p)
    dfull, _ = reg_inputs(c, p, d, None)
    ds = dfull[samp].reset_index(drop=True)
    evs = ev[samp]
    pre_s = pre[samp]
    P("\n[이벤트 표본 회귀 CAR(6,20) ~ 사후주 흐름 (CAR15 통제), 이중 군집 t: 기간별]")
    P(f"{'기간':22s}{'N':>6s}{'개인 계수(t)':>18s}{'외국인 계수(t)':>18s}{'기관 계수(t)':>18s}")
    for tag, m in (("전체", np.ones(len(ds), bool)), ("형성일<=2012-05-02", pre_s), ("형성일>2012-05-02", ~pre_s)):
        y, X, dd = reg_event(ds, evs & m, True)
        b_, se_, t_, V_ = ols_cluster(y, X.values, np.column_stack([dd.i.values, dd.j.values]))
        nm = ["const"] + list(X.columns)
        f3 = lambda t: f"{b_[nm.index(t)]:7.3f}({t_[nm.index(t)]:6.2f})"
        P(f"{tag:22s}{len(y):6d}{f3('indiv_post'):>18s}{f3('foreign_post'):>18s}{f3('inst_post'):>18s}")
    return out


def s_scramble(r01):
    """룩어헤드 교란 실험: 2011-12-30 이후 패널 값을 바꿔도 그 이전 형성일의 신호·칸 변수가 그대로여야 한다."""
    sec("1b. 룩어헤드 교란 실험 (r01.load_wide + build_base 를 변형 패널에 그대로 실행)")
    cols = ["ticker", "date", "ret1d_clean", "mktcap", "volume", "indiv_buy", "indiv_sell", "indiv_flow_w", "inst_flow_w",
            "foreign_flow_w", "indiv_flow_tv", "str5", "abvol", "mom12_1", "near52h"]
    pn = pd.read_parquet(PANEL, columns=cols)
    pn = pn[pn.date <= LAST].reset_index(drop=True)
    CUT = pd.Timestamp("2011-12-30")
    late = (pn.date > CUT).values
    rng = np.random.default_rng(3)
    num = cols[2:]
    A = pn.copy()
    for cc in num:                                     # A: 이후 구간 값을 열별로 뒤섞기 (분포는 보존)
        v = A.loc[late, cc].values.copy()
        rng.shuffle(v)
        A.loc[late, cc] = v
    Bm = pn.copy()
    for cc in ("indiv_flow_w", "inst_flow_w", "foreign_flow_w", "indiv_flow_tv", "abvol"):
        Bm.loc[late, cc] = Bm.loc[late, cc] * 3.0      # B: 이후 구간 척도를 3배로 (윈저화 경계가 움직인다)
    res = {}
    for tag, df in (("orig", pn), ("A_shuffle", A), ("B_scale3", Bm)):
        fp = f"{SCR}/panel_{tag}.parquet"
        df.to_parquet(fp, index=False)
        r01.PANEL = fp
        W, dates, tick, _ = r01.load_wide()
        b, path = r01.build_base(W, dates, tick)
        res[tag] = b
    b0 = res["orig"]
    ref = b0.date <= CUT
    chk = ["str5", "abvol", "mom12_1", "near52h", "mc", "mc_prev", "logmc", "share", "q_size", "size_t", "str5_t", "abvol_t",
           "fi_form_t", "fit_form_t", "share_t", "fi_form", "ff_form", "fn_form", "fit_form", "abvol_w"]
    outc = ["car15", "car620", "fi_post", "ff_post"]
    last5 = b0.k.isin(sorted(b0[ref].k.unique())[-5:]) & ref
    P(f"형성일 <= {CUT.date()} 행 {int(ref.sum())}개 (r01 형성일 {b0[ref].k.nunique()}개), 교란 대상 행(이후 구간) {int(late.sum())}")
    P(f"{'변수':14s}" + "".join(f"{t:>26s}" for t in ("A 뒤섞기: 바뀐 행", "B 3배: 바뀐 행")))
    for tag in ("A_shuffle", "B_scale3"):
        pass
    for cc in chk + ["|", "car15(결과)", "car620(결과)", "fi_post(결과)", "ff_post(결과)"]:
        if cc == "|":
            P("  --- 결과 변수는 t+1 이후를 쓰므로 교란되어야 정상(검정력 확인): 마지막 5개 형성일 행")
            continue
        base_cc = cc.replace("(결과)", "")
        sub = last5 if cc.endswith("(결과)") else ref
        s = []
        for tag in ("A_shuffle", "B_scale3"):
            a, bb = b0.loc[sub, base_cc].values.astype(float), res[tag].loc[sub, base_cc].values.astype(float)
            same = np.isclose(a, bb, rtol=1e-9, atol=1e-12, equal_nan=True)
            s.append(f"{int((~same).sum())} / {len(a)}")
        P(f"{cc:14s}{s[0]:>26s}{s[1]:>26s}")
    r01.PANEL = PANEL


def s_memo_csv(r01):
    sec("4. 보고서 숫자 ↔ CSV 전수 대조")
    L = r01.LABEL
    t1 = pd.read_csv(f"{TAB}/r01_t1_event_vs_control_main.csv", encoding="utf-8-sig")
    t2 = pd.read_csv(f"{TAB}/r01_t2_regressions.csv", encoding="utf-8-sig")
    t4 = pd.read_csv(f"{TAB}/r01_t4_robustness.csv", encoding="utf-8-sig").set_index("설정")
    t5 = pd.read_csv(f"{TAB}/r01_t5_verdict.csv", encoding="utf-8-sig")
    t6 = pd.read_csv(f"{TAB}/r01_t6_indiv_share_split.csv", encoding="utf-8-sig")
    t7 = pd.read_csv(f"{TAB}/r01_t7_post_flow_terciles_in_events.csv", encoding="utf-8-sig")
    t0 = pd.read_csv(f"{TAB}/r01_t0_signal_diag.csv", encoding="utf-8-sig")
    t0b = pd.read_csv(f"{TAB}/r01_t0b_flow_car15_corr.csv", encoding="utf-8-sig")
    VOL, TV = "CNN 기준(거래량 척도 칸)", "CNN 기준(거래대금 척도 칸)"
    checks = []

    def chk(loc, desc, memo, csv, tol=0.0051):
        ok = abs(float(csv) - float(memo)) <= tol
        checks.append((loc, desc, memo, float(csv), ok))

    def r1(setting, key):
        r = t1[(t1.설정 == setting) & (t1.지표 == L[key])]
        assert len(r) == 1, (setting, key)
        return r.iloc[0]

    # ---- 표 1 (memo 33~42)
    T1 = [(33, VOL, "car15", 3669, 0.61, 0.05, 0.56, 4.53, 5.71), (34, VOL, "car620", 3656, 0.13, 0.04, 0.09, 0.44, 0.46),
          (35, VOL, "car2160", 3643, 0.31, -0.11, 0.40, 1.23, 1.10), (36, VOL, "fi_form", 3676, 4.84, 4.64, 0.21, 0.18, 0.16),
          (37, VOL, "fi_post", 3676, 2.36, -1.38, 3.74, 2.68, 2.70), (38, TV, "fit_post", 3673, -0.87, -3.84, 2.98, 2.09, 2.04),
          (39, VOL, "ri_post", 3676, 51.39, 50.46, 0.92, 1.80, 1.69), (40, VOL, "fi_mid", 3675, -12.10, -7.73, -4.38, -1.32, -1.47),
          (41, VOL, "ff_form", 3676, -2.17, -2.11, -0.06, -0.07, None), (41, VOL, "ff_post", 3676, -1.90, -1.19, -0.71, -0.68, None),
          (41, VOL, "ff_mid", 3676, -1.69, -1.74, 0.08, 0.04, None), (42, VOL, "fn_form", 3676, -1.50, -2.41, 0.91, 0.73, None),
          (42, VOL, "fn_post", 3676, -0.07, 1.52, -1.58, -1.31, None), (42, VOL, "fn_mid", 3676, 7.58, 6.26, 1.30, 0.47, None)]
    for ln, st, key, n, ev_, ct, df, td, tm in T1:
        r = r1(st, key)
        chk(f"표1 L{ln}", f"{key} N매칭", n, r["N매칭"], tol=0.5)
        chk(f"표1 L{ln}", f"{key} 이벤트(매칭)", ev_, r["이벤트평균_매칭"])
        chk(f"표1 L{ln}", f"{key} 대조", ct, r["대조평균"])
        chk(f"표1 L{ln}", f"{key} 차이", df, r["차이"])
        chk(f"표1 L{ln}", f"{key} t(날짜)", td, r["t_날짜군집"])
        if tm is not None:
            chk(f"표1 L{ln}", f"{key} t(월)", tm, r["t_월군집"])
    # 이벤트 전체 (L44), 하위표본 (L46)
    for key, n, m, t in (("car15", 4359, 0.59, 5.87), ("car620", 4345, 0.16, 0.99), ("car2160", 4325, 0.18, 0.66)):
        r = r1(VOL, key)
        chk("L44", f"{key} 이벤트 전체 N", n, r["N이벤트전체"], 0.5)
        chk("L44", f"{key} 이벤트 전체 평균", m, r["이벤트평균_전체"])
        chk("L44", f"{key} 이벤트 전체 t", t, r["t_전체"])
    r = t1[t1.설정.str.contains("하위표본")].iloc[0]
    chk("L46", "CAR(1,5)>0 하위표본 N", 2253, r["N이벤트전체"], 0.5)
    chk("L46", "CAR(1,5)>0 하위표본 CAR(6,20) 평균(비매칭)", 0.09, r["이벤트평균_전체"])
    chk("L46", "CAR(1,5)>0 하위표본 t(비매칭)", 0.32, r["t_전체"])
    # L47 개인 비중
    for g, (m, t) in {"낮음": (0.38, 1.48), "중간": (0.39, 1.20), "높음": (-0.58, -1.44), "높음-낮음": (-0.96, -2.11)}.items():
        r = t6[(t6.개인비중분위 == g) & (t6.지표 == L["car620"])].iloc[0]
        chk("L47", f"개인비중 {g} CAR(6,20) 차이", m, r["차이"])
        chk("L47", f"개인비중 {g} CAR(6,20) t", t, r["t"])
    # ---- 표 2 (53~60)
    def rg(sc, model, car, term):
        r = t2[(t2.개인척도 == sc) & (t2.회귀 == model) & (t2.CAR15통제 == car) & (t2.항 == term)]
        assert len(r) == 1, (sc, model, car, term)
        return r.iloc[0]

    T2 = [(53, "거래량", "이벤트표본", "아니오", 4344, [("indiv_post", -0.41, -1.14), ("foreign_post", -0.57, -2.24), ("inst_post", -0.52, -1.53)]),
          (54, "거래량", "이벤트표본", "예", 4343, [("indiv_post", -0.45, -1.25), ("foreign_post", -0.54, -2.14), ("inst_post", -0.43, -1.30), ("car15", -0.47, -1.19)]),
          (55, "거래량", "이벤트표본(순위흐름)", "예", 4343, [("indiv_post", 0.16, 0.61), ("foreign_post", -0.15, -0.81), ("inst_post", 0.15, 0.70), ("car15", -0.47, -1.24)]),
          (56, "거래대금", "이벤트표본", "예", 4343, [("indiv_post", -0.52, -1.52), ("foreign_post", -0.57, -2.33), ("inst_post", -0.48, -1.48), ("car15", -0.46, -1.19)]),
          (57, "거래량", "전종목주+상위10분위더미", "예", 67224, [("indiv_post", -0.01, -0.10), ("foreign_post", -0.18, -3.02), ("inst_post", -0.03, -0.29), ("car15", -0.32, -2.53)]),
          (58, "거래량", "전종목주+상위10분위더미", "예", None, [("D_x_indiv_post", -0.22, -0.86), ("D_x_foreign_post", -0.14, -0.76), ("D_x_inst_post", -0.06, -0.25)]),
          (59, "거래량", "전종목주+상위10분위더미", "예", None, [("합계_indiv_post", -0.23, -0.89), ("합계_foreign_post", -0.32, -1.78), ("합계_inst_post", -0.09, -0.36)]),
          (60, "거래량", "전종목주+상위10분위더미", "아니오", 67267, [("indiv_post", 0.02, 0.22), ("foreign_post", -0.21, -3.47), ("inst_post", -0.10, -1.12)])]
    for ln, sc, model, car, n, terms in T2:
        for term, cf, t in terms:
            r = rg(sc, model, car, term)
            chk(f"표2 L{ln}", f"{sc}|{model}|CAR통제={car}|{term} 계수", cf, r["계수"])
            chk(f"표2 L{ln}", f"{sc}|{model}|CAR통제={car}|{term} t", t, r["t"])
            if n is not None:
                chk(f"표2 L{ln}", f"{model} N", n, r["N"], 0.5)
    r = rg("거래량", "이벤트표본", "예", "indiv_post")
    chk("L62", "이벤트 날짜수", 95, r["날짜수"], 0.5)
    chk("L62", "이벤트 종목수", 750, r["종목수"], 0.5)
    r = rg("거래량", "전종목주+상위10분위더미", "예", "indiv_post")
    chk("L62", "전종목 날짜수", 95, r["날짜수"], 0.5)
    chk("L62", "전종목 종목수", 762, r["종목수"], 0.5)
    chk("L65", "외국인 순위 흐름 이벤트", -0.15, rg("거래량", "이벤트표본(순위흐름)", "예", "foreign_post")["계수"])
    chk("L65", "외국인 순위 흐름 전종목", -0.02, rg("거래량", "전종목주+상위10분위더미(순위흐름)", "예", "foreign_post")["계수"])
    # L66 사후 순매수 3분위
    def g7(who, out, q):
        return t7[(t7.주체 == who) & (t7.결과 == L[out]) & (t7.분위 == q)].iloc[0]

    for who, (m, t) in {"개인": (1.02, 2.92), "외국인": (-0.65, -2.01), "기관": (0.09, 0.27)}.items():
        r = g7(who, "car620", "T3-T1")
        chk("L66", f"{who} T3-T1 CAR(6,20)", m, r["평균"])
        chk("L66", f"{who} T3-T1 CAR(6,20) t", t, r["t"])
    chk("L66", "개인 T3 CAR(1,5)", -0.76, g7("개인", "car15", "T3")["평균"])
    chk("L66", "개인 T1 CAR(1,5)", 2.24, g7("개인", "car15", "T1")["평균"])
    chk("L117", "개인 T3-T1 CAR(1,5)", -2.99, g7("개인", "car15", "T3-T1")["평균"])
    chk("L117", "개인 T3-T1 CAR(1,5) t", -12.7, g7("개인", "car15", "T3-T1")["t"], 0.051)
    for sample, flow, v in (("전 종목-주", "fi_post", -0.30), ("전 종목-주", "fn_post", 0.26), ("전 종목-주", "ff_post", 0.01)):
        r = t0b[(t0b.iloc[:, 0] == t0b.iloc[:, 0].unique()[0 if sample == "전 종목-주" else 1]) & (t0b.iloc[:, 1] == flow)].iloc[0]
        chk("L66", f"스피어만(CAR15,{flow}) 전 종목-주", v, r.iloc[2])
    # ---- 표 3·4 (t4)
    def r4(s):
        return t4.loc[s]

    T4 = [(72, "cnn_d|vol|금지주제외=False", dict(N이벤트=4368, N매칭=3676, 중앙칸=8, 이벤트CAR15=0.59, t_이벤트CAR15=5.87, 이벤트CAR620=0.16, t_이벤트CAR620=0.99,
                                             차이CAR620=0.09, t_차이CAR620=0.44, 차이개인사후주=3.74, t_차이개인사후주=2.68, 회귀_indiv_post=-0.45, t_회귀_indiv_post=-1.25,
                                             회귀_foreign_post=-0.54, t_회귀_foreign_post=-2.14)),
          (73, "str|vol|금지주제외=False", dict(N이벤트=3945, N매칭=3110, 중앙칸=7, 이벤트CAR15=0.16, t_이벤트CAR15=1.10, 이벤트CAR620=-0.08, t_이벤트CAR620=-0.37,
                                           차이CAR620=-0.25, t_차이CAR620=-0.97, 차이개인사후주=-1.86, t_차이개인사후주=-1.07, 회귀_indiv_post=-0.11, t_회귀_indiv_post=-0.25,
                                           회귀_foreign_post=-0.47, t_회귀_foreign_post=-1.42)),
          (74, "res|vol|금지주제외=False", dict(N이벤트=4331, N매칭=3630, 중앙칸=7, 이벤트CAR15=0.42, t_이벤트CAR15=4.07, 이벤트CAR620=0.06, t_이벤트CAR620=0.37,
                                           차이CAR620=0.03, t_차이CAR620=0.16, 차이개인사후주=0.61, t_차이개인사후주=0.46, 회귀_indiv_post=-0.05, t_회귀_indiv_post=-0.17,
                                           회귀_foreign_post=-0.21, t_회귀_foreign_post=-1.01)),
          (86, "cnn_d|vol|금지주제외=True", dict(N이벤트=3550, N매칭=2989, 이벤트CAR15=0.55, t_이벤트CAR15=4.95, 이벤트CAR620=0.18, t_이벤트CAR620=1.01, 차이CAR620=0.13,
                                            t_차이CAR620=0.66, 차이개인사후주=4.28, t_차이개인사후주=2.66, 회귀_indiv_post=-0.46, t_회귀_indiv_post=-1.09,
                                            회귀_foreign_post=-0.68, t_회귀_foreign_post=-2.38)),
          (87, "cnn_d|tv|금지주제외=False", dict(N이벤트=4368, N매칭=3673, 차이CAR620=0.06, t_차이CAR620=0.30, 차이개인사후주=2.98, t_차이개인사후주=2.09,
                                            회귀_indiv_post=-0.52, t_회귀_indiv_post=-1.52, 회귀_foreign_post=-0.57, t_회귀_foreign_post=-2.33)),
          (88, "cnn_tb|vol|금지주제외=False", dict(N이벤트=4315, N매칭=3590, 이벤트CAR15=0.56, t_이벤트CAR15=5.21, 이벤트CAR620=0.23, t_이벤트CAR620=1.26, 차이CAR620=0.15,
                                              t_차이CAR620=0.69, 차이개인사후주=-1.81, t_차이개인사후주=-1.31, 회귀_indiv_post=-0.41, t_회귀_indiv_post=-1.31,
                                              회귀_foreign_post=-0.45, t_회귀_foreign_post=-1.73)),
          (89, "cnn_w|vol|금지주제외=False", dict(N이벤트=4459, N매칭=3672, 이벤트CAR15=0.29, t_이벤트CAR15=3.48, 이벤트CAR620=0.16, t_이벤트CAR620=0.95, 차이CAR620=-0.12,
                                             t_차이CAR620=-0.63, 차이개인사후주=-1.25, t_차이개인사후주=-0.85, 회귀_indiv_post=-0.51, t_회귀_indiv_post=-1.52,
                                             회귀_foreign_post=-0.40, t_회귀_foreign_post=-1.92)),
          (90, "cnn_d|vol|대조군=상위10분위 제외", dict(N이벤트=4368, N매칭=3596, 차이CAR620=0.12, t_차이CAR620=0.59, 차이개인사후주=3.24, t_차이개인사후주=2.22)),
          (91, "cnn_d|vol|엄격 편입(4주 상위 이력 없음)", dict(N이벤트=4291, N매칭=3616, 이벤트CAR15=0.61, t_이벤트CAR15=6.00, 이벤트CAR620=0.16, t_이벤트CAR620=0.99,
                                                          차이CAR620=0.08, t_차이CAR620=0.40, 차이개인사후주=3.59, t_차이개인사후주=2.60, 회귀_indiv_post=-0.46,
                                                          t_회귀_indiv_post=-1.30, 회귀_foreign_post=-0.56, t_회귀_foreign_post=-2.25))]
    for ln, s, kv in T4:
        r = r4(s)
        for k_, v in kv.items():
            chk(f"표3·4 L{ln}", f"{s}|{k_}", v, r[k_], tol=0.5 if k_.startswith("N") or k_ == "중앙칸" else 0.0051)
    # L97 개인 비중 격차
    for s, (v, t) in {"cnn_d|vol|금지주제외=False": (-0.96, -2.11), "cnn_d|tv|금지주제외=False": (-1.00, None),
                      "cnn_d|vol|엄격 편입(4주 상위 이력 없음)": (-1.05, None), "cnn_tb|vol|금지주제외=False": (-0.70, -1.26),
                      "cnn_w|vol|금지주제외=False": (-0.31, -0.67), "str|vol|금지주제외=False": (-0.75, -1.25), "res|vol|금지주제외=False": (-0.71, -1.40)}.items():
        chk("L97", f"{s} 개인비중 높음-낮음 CAR620", v, r4(s)["개인비중높음-낮음_차이CAR620"])
        if t is not None:
            chk("L97", f"{s} 개인비중 높음-낮음 t", t, r4(s)["t_개인비중높음-낮음_차이CAR620"])
    for s, v in {"cnn_w|vol|금지주제외=True": 0.06, "cnn_w|tv|금지주제외=True": 0.10}.items():
        chk("L97", f"{s} 예외(+0.06~+0.10)", v, r4(s)["개인비중높음-낮음_차이CAR620"])
    # ---- 표 5 판정 (t5 CSV)
    def v5(meas_contains):
        r = t5[t5.측정.str.contains(meas_contains, regex=False)]
        assert len(r) >= 1, meas_contains
        return r.iloc[0]

    chk("L103", "판정표 개인 사후 거래량", 3.74, v5("사후주 개인 순매수, 이벤트-대조 [거래량 %]")["값"])
    chk("L103", "판정표 개인 사후 거래대금", 2.98, v5("[거래대금 %]")["값"])
    chk("L103", "판정표 사후-형성", 3.53, v5("(사후주-형성주)")["값"])
    chk("L103", "판정표 순위", 0.92, v5("순위(0~100)")["값"])
    chk("L105", "판정표 CAR(21,60) 이벤트", 0.18, v5("이벤트 CAR(21,60)")["값"])
    chk("L111", "형성 주 str5", -1.58, v5("str5")["값"])
    chk("L111", "형성 주 str5 t", -4.41, v5("str5")["t"])
    chk("L111", "외국인 형성 주 순매수", -1.47, v5("외국인 형성")["값"])
    chk("L111", "기관 형성 주 순매수", -1.33, v5("기관 형성")["값"])
    chk("L111", "외국인 T3-T1 CAR620", -0.65, v5("외국인 사후주 순매수 T3-T1")["값"])
    # ---- 설정·진단 수치
    def d0(s):
        return t0[t0.항목.str.contains(s, regex=False)].iloc[0]

    chk("L79", "주간 IC 평균", 0.075, d0("순위상관 IC")["값"], 0.0005)
    chk("L79", "주간 IC t", 13.7, d0("순위상관 IC")["t"], 0.051)
    chk("L79", "상위-하위 10분위 차", 1.11, d0("상위-하위")["값"])
    chk("L79", "상위-하위 t", 7.5, d0("상위-하위")["t"], 0.051)
    chk("L76", "-str5 순위상관", 0.12, d0("순위상관(prob, -str5)")["값"])
    chk("L76", "이벤트 중 -str5 상위 비율 20%", 0.20, d0("신규편입 이벤트 중 -str5")["값"], 0.0051)
    chk("L76", "이벤트 중 잔차신호 상위 비율 79%", 0.79, d0("신규편입 이벤트 중 잔차")["값"], 0.0051)
    # ---- 서술 판정 논리 점검
    P(f"\n숫자 대조 {len(checks)}건 중 정확 일치 {sum(1 for x in checks if x[4])}건, 불일치 {sum(1 for x in checks if not x[4])}건 (허용오차 0.0051 = 소수 둘째 자리 반올림)")
    dev = np.array([abs(x[3] - x[2]) for x in checks if not x[4] and not x[1].endswith("N매칭")])
    P(f"  불일치 중 N 항목 제외 최대 편차 {dev.max():.4f} (CSV 3자리 표기를 다시 반올림한 이중 반올림 수준)")
    for loc, desc, m, cv, ok in checks:
        if not ok:
            P(f"  불일치 {loc} | {desc}: 보고서 {m} / CSV {cv:.4f}")
    P("\n[서술 논리 점검]")
    t4a = t4.copy()
    P(f"  (96) CAR(6,20) 이벤트 평균: 22개 설정 중 음수이면서 |t|>1.96 인 것 {int(((t4a.이벤트CAR620 < 0) & (t4a.t_이벤트CAR620 < -1.96)).sum())}개, "
      f"양수 & t>1.96 인 것 {int(((t4a.이벤트CAR620 > 0) & (t4a.t_이벤트CAR620 > 1.96)).sum())}개")
    pos = t4a[(t4a.차이개인사후주 > 0)]
    sig = t4a[(t4a.차이개인사후주 > 0) & (t4a.t_차이개인사후주 > 1.96)]
    P(f"  (96) 개인 사후 주 이벤트-대조가 양(+)인 설정 {len(pos)}/22: {list(pos.index)}")
    P(f"       그중 t>1.96: {list(sig.index)}")
    P(f"       CNN 대표 신호 6개 변형 중 t>1.96 인 것: "
      f"{[s for s in t4a.index if s.startswith('cnn_d') and t4a.loc[s, 't_차이개인사후주'] > 1.96]}")
    cd = [s for s in t4a.index if s.startswith("cnn_d")]
    P(f"       cnn_d 계열 {len(cd)}개 설정의 개인 사후 주 t: " + ", ".join(f"{t4a.loc[s, 't_차이개인사후주']:.2f}" for s in cd))
    g = t4a["개인비중높음-낮음_차이CAR620"]
    gt = t4a["t_개인비중높음-낮음_차이CAR620"]
    P(f"  (97) 개인비중 높음-낮음 CAR620 격차: 음수 {int((g < 0).sum())}/22, 양수 {list(g[g > 0].index)}, 범위 {g.min():.3f}~{g.max():.3f}")
    P(f"       t<-1.96 인 설정: {[(s, round(gt[s], 2)) for s in gt.index if gt[s] < -1.96]}")
    P(f"       cnn_d 계열 중 t>-1.96(비유의)인 설정: {[(s, round(gt[s], 2)) for s in cd if gt[s] > -1.96]}")
    r = t6[t6.지표 == L["fi_post"]].set_index("개인비중분위")
    P(f"  (t6) 개인 사후 주 이벤트-대조: 낮음 {r.loc['낮음', '차이']:.2f}(t {r.loc['낮음', 't']:.2f}), 중간 {r.loc['중간', '차이']:.2f}(t {r.loc['중간', 't']:.2f}), "
      f"높음 {r.loc['높음', '차이']:.2f}(t {r.loc['높음', 't']:.2f}), 높음-낮음 {r.loc['높음-낮음', '차이']:.2f}(t {r.loc['높음-낮음', 't']:.2f})  -> 보고서 본문에 없음")
    r = t1[t1.설정.str.contains("하위표본")].iloc[0]
    P(f"  (46) CAR(1,5)>0 하위표본: 보고서는 비매칭 평균 {r['이벤트평균_전체']:.2f}(t {r['t_전체']:.2f}) 인용. 같은 행의 매칭 차이는 {r['차이']:.2f}(t {r['t_날짜군집']:.2f}, N {int(r['N매칭'])}, 중앙칸 {r['중앙칸']:.0f})")
    return checks


def s_facts(c, p, d, ev, samp, r01):
    sec("4c. 보고서 서술 속 보조 수치 (CSV에 없는 것) 재계산")
    ds = d[samp].reset_index(drop=True)
    evs = ev[samp]
    i, j = p.i.values, p.j.values
    P(f"  형성일 {p.k.nunique()}개 ({pd.Timestamp(p.date.min()).date()}~{pd.Timestamp(p.date.max()).date()}), 종목 {p.ticker.nunique()}, 행 {len(p)} (보고서: 100개, 767, 71,222)")
    P(f"  분석 형성일 첫날 {pd.Timestamp(p.date[p.k == 5].iloc[0]).date()} (보고서 2011-02-11), 종목-주 {int(samp.sum())} (보고서 67,629)")
    P(f"  개인 사후 주 흐름(fi_post) 표준편차(분석 표본 전체) {ds.fi_post.std():.1f} (보고서: '1 SD가 약 77')")
    # pred ret vs 패널 5일 수익률
    adj = c.M["adj_close"]
    r5 = adj[np.minimum(i + 5, c.nd - 1), j] / adj[i, j] - 1
    dif = np.abs(r5 - p.ret.values)
    ok = np.isfinite(dif)
    P(f"  pred ret vs 패널 adj_close 5일 수익률: 유효 {ok.mean():.4f}, |차이|<1e-4 비율 {(dif[ok] < 1e-4).mean():.4f}, 최대 {dif[ok].max():.4f} (보고서: 대부분 1e-4 이내, 최대 1.1%p)")
    # 개인 거래 비중
    num = pd.DataFrame(c.M["indiv_buy"] + c.M["indiv_sell"]).rolling(60, min_periods=40).sum().values
    den = pd.DataFrame(c.M["volume"]).rolling(60, min_periods=40).sum().values
    share = num / (2 * den)
    sh = share[i, j]
    P(f"  개인 거래 비중(직전 60일, 형성일 포함): 분석 표본 평균 {np.nanmean(sh[samp]):.3f}, 이벤트 평균 {np.nanmean(sh[ev]):.3f} (보고서 0.77 / 0.75)")
    # 패널 str5 타이밍
    s5 = adj[i, j] / adj[np.maximum(i - 5, 0), j] - 1
    st = c.M["str5"][i, j]
    okk = np.isfinite(s5) & np.isfinite(st)
    P(f"  패널 str5 = adj_close[t]/adj_close[t-5]-1 : 유효 {okk.sum()}행 중 |차이|<1e-9 비율 {(np.abs(s5[okk] - st[okk]) < 1e-9).mean():.6f}")
    # 공매도 금지 제외 형성일 수
    bi = np.where((c.cal >= np.datetime64("2011-08-10")) & (c.cal <= np.datetime64("2011-11-09")))[0]
    drop = (bi.min() <= i + 20) & (bi.max() >= i - 4)
    P(f"  공매도 금지 창 제외 후 형성일 {p.k[samp & ~drop].nunique()}개 (보고서 77), 제외된 형성일 {p.k[samp & drop].nunique()}개")
    # 이벤트 정렬 예시: 장형 패널에서 직접
    e0 = np.where(ev)[0][100]
    tk, dt = p.ticker[e0], p.date[e0]
    P(f"  [이벤트 시간 정렬 예시] {tk} 형성일 {pd.Timestamp(dt).date()} (t=0 종가까지의 정보로 신호). 다음 거래일 달력과 ret1d_clean:")
    P("    " + ", ".join(f"h={h}:{pd.Timestamp(c.cal[i[e0] + h]).date()} {c.M['ret1d_clean'][i[e0] + h, j[e0]]:+.4f}" for h in range(1, 6)))
    P(f"    pred.ret(t→t+5 수익률) {p.ret[e0]:+.4f} = adj[t+5]/adj[t]-1 {adj[i[e0] + 5, j[e0]] / adj[i[e0], j[e0]] - 1:+.4f}")


def holm_table(rows):
    """rows: (label, t, 사전지정?). 정규 근사 양측 p. Holm 보정 p."""
    df = pd.DataFrame(rows, columns=["검정", "t", "사전지정(A-2 핵심)"])
    df["p"] = 2 * (1 - stats.norm.cdf(np.abs(df.t)))
    df = df.sort_values("p").reset_index(drop=True)
    m = len(df)
    adj = np.maximum.accumulate((m - np.arange(m)) * df.p.values)
    df["Holm_p"] = np.minimum(adj, 1.0)
    df["Bonf_p"] = np.minimum(df.p * m, 1.0)
    return df


def s_rdf(memo_path):
    sec("5. 연구자 자유도 / 다중 비교")
    txt = open(memo_path, encoding="utf-8").read()
    import re
    t_text = re.findall(r"t\s?[−-]?\d+\.\d+", txt)
    t_par = re.findall(r"\(\s*[−+-]?\d+\.\d+\s*\)", txt)
    P(f"  보고서에 쓰인 'tX.XX' 서술형 t 표기 {len(t_text)}개, 괄호 안 숫자 (t 또는 값) {len(t_par)}개 (표 셀의 '계수 (t)' 포함, 대략치)")
    n_t = 0
    for f in sorted(os.listdir(TAB)):
        if not f.endswith(".csv"):
            continue
        a = pd.read_csv(f"{TAB}/{f}", encoding="utf-8-sig")
        tc = [x for x in a.columns if x == "t" or x.startswith("t_") or x.endswith("_t")]
        cnt = int(a[tc].notna().sum().sum()) if tc else 0
        big = int((a[tc].abs() > 1.96).sum().sum()) if tc else 0
        P(f"  {f:45s} t열 {len(tc):2d}개, t값 {cnt:5d}개, |t|>1.96 {big:5d}개 ({big / cnt:.1%})" if cnt else f"  {f:45s} t값 없음")
        n_t += cnt
    P(f"  CSV 전체 t값 {n_t}개 (보고서에 인용된 것은 위 대략 {len(t_text) + len(t_par)}개 안팎)")
    fam = [("개인 사후주 순매수 이벤트-대조 [거래량]", 2.681, "예"), ("개인 사후주 [거래대금]", 2.090, "변형"), ("개인 사후-형성 [거래량]", 2.502, "변형"),
           ("개인 사후주 순위", 1.804, "변형"), ("CAR(1,5) 이벤트 전체", 5.874, "예"), ("CAR(1,5) 이벤트-대조", 4.528, "예"),
           ("CAR(6,20) 이벤트 전체", 0.987, "예"), ("CAR(6,20) 이벤트-대조", 0.438, "예"), ("개인 비중 높음-낮음 CAR(6,20)", -2.107, "예"),
           ("회귀 개인 사후주 (CAR15 통제)", -1.250, "예"), ("회귀 개인 사후주 (CAR15 미통제)", -1.144, "참고"), ("회귀 외국인 사후주 (H2 대조)", -2.142, "예"),
           ("형성 주 str5 (사후 추가)", -4.411, "사후"), ("외국인 형성 주 순매수 (사후 추가)", -1.069, "사후"), ("기관 형성 주 순매수 (사후 추가)", -0.849, "사후"),
           ("외국인 T3-T1 CAR(6,20) (사후 추가)", -2.012, "사후")]
    df = holm_table(fam)
    P("\n  [판정표(표 5)에 쓰인 16개 검정의 Holm 보정] (정규 근사 양측)")
    for _, r in df.iterrows():
        P(f"    {r['검정']:42s} t={r['t']:7.3f} p={r['p']:.4f} Holm={r['Holm_p']:.4f} Bonf={r['Bonf_p']:.4f}  [{r['사전지정(A-2 핵심)']}]")
    return df


def s_placebo(d, ev, samp, n_draw=300):
    """무작위 가짜 이벤트(주당 실제와 같은 수)로 같은 매칭·날짜군집 t를 돌려 기각률이 5% 근처인지 본다(검정 크기 점검)."""
    sec("3b. 플라세보: 무작위 이벤트에서 매칭 + 날짜군집 t의 기각률")
    ds = d[samp].reset_index(drop=True)
    evs = ev[samp]
    rng = np.random.default_rng(11)
    cnt = ds[evs].groupby("k").size()
    idx_by_k = {k: np.where(ds.k.values == k)[0] for k in cnt.index}
    res = {"fi_post": [], "car15": [], "car620": [], "car2160": []}
    t0 = time.time()
    for b in range(n_draw):
        fake = np.zeros(len(ds), bool)
        for k, n in cnt.items():
            fake[rng.choice(idx_by_k[k], size=n, replace=False)] = True
        E, nk, nc = matched(ds, fake, list(res))
        for x in res:
            res[x].append(cl_t(E[x + "_d"], E.i)[2])
    P(f"  {n_draw}회, {time.time() - t0:.0f}초. 귀무(무작위 이벤트)에서의 t 분포:")
    for x, v in res.items():
        v = np.array(v)
        P(f"    {x:8s} 평균 {v.mean():6.3f}, 표준편차 {v.std():5.3f} (이상적 1.0), |t|>1.96 기각률 {np.mean(np.abs(v) > 1.96):.3f} (명목 0.05)")


def s_flow_alignment(c):
    """투자자 순매수 열이 같은 날 수익률과 정렬돼 있는지: 일별 횡단면 스피어만 상관을 지연(lag)별로 평균.
    개인은 같은 날 역방향(-), 기관은 순방향(+)이 알려진 모양이다. 하루 밀려 있으면 lag=0 상관이 사라진다."""
    sec("1c. 투자자 흐름 열의 날짜 정렬 점검 (flow[d] vs ret[d+lag], 2011-2012 일별 횡단면 스피어만 평균)")
    d0, d1 = np.searchsorted(c.cal, np.datetime64("2011-01-10")), np.searchsorted(c.cal, np.datetime64("2012-11-26"))
    P(f"{'흐름':14s}" + "".join(f"{'lag=' + str(l):>10s}" for l in (-2, -1, 0, 1, 2)))
    for nm in ("indiv_flow_w", "inst_flow_w", "foreign_flow_w"):
        row = []
        for lag in (-2, -1, 0, 1, 2):
            cs = []
            for dd in range(d0, d1, 3):                       # 3일 간격 표집 (속도)
                x = pd.Series(c.M[nm][dd])
                y = pd.Series(c.M["ret1d_clean"][dd + lag])
                ok = x.notna() & y.notna()
                if ok.sum() > 100:
                    cs.append(x[ok].rank().corr(y[ok].rank()))
            row.append(np.mean(cs))
        P(f"{nm:14s}" + "".join(f"{v:10.3f}" for v in row))


def s_balance(c, p, d, ev, samp):
    """매칭 후 공변량 균형과, 칸 안 잔여 불균형을 회귀로 조정했을 때 개인 사후주 효과."""
    sec("2c. 매칭 품질: 이벤트-대조 공변량 균형 + 공변량 조정 효과")
    d = d.copy()
    i, j = p.i.values, p.j.values
    d["logmc"] = np.log(c.M["mktcap"][i, j])
    d["mom12_1"] = c.M["mom12_1"][i, j]
    d["near52h"] = c.M["near52h"][i, j]
    d["str5p"] = d.str5 * 100
    d["abvol_x"] = d.abvol
    ds = d[samp].reset_index(drop=True)
    evs = ev[samp]
    cols = ["str5p", "abvol_x", "fi_form", "logmc", "mom12_1", "near52h", "fi_post", "car15", "car620"]
    E, nk, nc = matched(ds, evs, cols)
    P(f"  매칭 이벤트 {len(E)}건. 이벤트-대조 평균 차이 (t 날짜군집, 표준화 차이 = 차이/전체 표준편차)")
    for x in ["str5p", "abvol_x", "fi_form", "logmc", "mom12_1", "near52h"]:
        m, se, t, n, G = cl_t(E[x + "_d"], E.i)
        P(f"    {x:8s} 차이 {m:8.4f}  t {t:6.2f}  표준화 {m / ds[x].std():6.3f}")
    Xd = E[["str5p_d", "abvol_x_d", "fi_form_d", "logmc_d", "mom12_1_d", "near52h_d"]]
    ok = Xd.notna().all(axis=1) & E.fi_post_d.notna()
    P("  개인 사후주 이벤트-대조 차이를 칸 안 잔여 공변량 차이로 회귀(절편 = 조정 효과, 날짜 군집):")
    for lab, cc in (("조정 없음(참고)", None), ("str5만", ["str5p_d"]), ("str5+abvol+형성주 개인흐름", ["str5p_d", "abvol_x_d", "fi_form_d"]),
                    ("6개 모두", list(Xd.columns))):
        if cc is None:
            m, se, t, n, G = cl_t(E.fi_post_d[ok], E.i[ok])
            P(f"    {lab:28s} 효과 {m:6.3f} (t {t:5.2f})")
            continue
        b, se, t, Vv = ols_cluster(E.fi_post_d[ok].values, E.loc[ok, cc].values, E.i[ok].values)
        P(f"    {lab:28s} 효과 {b[0]:6.3f} (t {t[0]:5.2f})   기울기 " + ", ".join(f"{n_}={bb:.3f}({tt:.2f})" for n_, bb, tt in zip(cc, b[1:], t[1:])))


def s_misc_lookahead(c, p, d, ev, top):
    sec("1. 룩어헤드 점검 (독립 구현 쪽 실험)")
    # (1) 이벤트 정의: t 이후 pred를 지워도 같은가 (내 구현)
    cut = p.k.values <= 50
    q = p[cut].reset_index(drop=True)
    ev_cut = events_loop(q)[1]
    ok = (ev_cut == ev[cut]).all()
    P(f"  이벤트 정의 절단 불변성(k<=50로 pred를 잘라도 같은 이벤트): {ok}  (이벤트 {int(ev[cut].sum())}개)")
    # (2) 날짜 내 횡단면 분류: 분위 변수가 같은 날짜 안에서만 만들어졌는지 = 날짜별 비율 점검
    samp = d.k.values >= 5
    ds = d[samp]
    for col, n in (("q_size", 5), ("str5_t", 3), ("abvol_t", 3), ("fi_form_t", 3)):
        share = ds.groupby("i")[col].apply(lambda s: s.value_counts(normalize=True).max())
        P(f"  {col}: 날짜별 최대 칸 비중 평균 {share.mean():.3f} (균등 {1 / n:.3f}), 최대 {share.max():.3f}")
    # (3) 결과 변수 조건화: 대조군 풀이 car15 유효 여부로 걸러지는 정도
    cell = ["i", "q_size", "str5_t", "abvol_t", "fi_form_t"]
    keyok = ds[cell].notna().all(axis=1).values
    evs = ev[samp]
    ctrl_all = (~evs) & keyok
    ctrl_c15 = ctrl_all & ds.car15.notna().values
    P(f"  대조군 풀: car15 유효 조건 없이 {int(ctrl_all.sum())}, 있으면 {int(ctrl_c15.sum())} (제외 {int(ctrl_all.sum() - ctrl_c15.sum())}행 = {(1 - ctrl_c15.sum() / ctrl_all.sum()):.2%}); "
      f"이벤트 중 car15 결측 {int((evs & keyok & ds.car15.isna().values).sum())}건은 대조 제외 없이 E에 남음 (r01 L261 vs L266 비대칭)")
    # (4) 전체표본 분위수에 의존하는 변환
    P(f"  panel *_w 윈저화 경계는 2010~2026 전체 표본 0.5/99.5% (B6). 이 표본(2011-2012)에서 경계에 걸려 잘리는 셀 비율: "
      f"{np.nanmean((np.abs(c.M['indiv_flow'][d.i.values, d.j.values] - c.M['indiv_flow_w'][d.i.values, d.j.values]) > 1e-12)[samp]):.4f} (형성일 당일 셀 기준)")


def main():
    t0 = time.time()
    r01 = load_r01()
    c = load_ctx()
    p = load_pred(c)
    P(f"로드 {time.time() - t0:.0f}초: 패널 일수 {c.nd}, 종목 {c.nt}, pred 행 {len(p)}")
    top, ev, d, ds, evs, h, E = s_independent(c, p)
    samp = d.k.values >= 5
    s_misc_lookahead(c, p, d, ev, top)
    s_flow_alignment(c)
    s_sensitivity(c, p, ev, samp)
    s_balance(c, p, d, ev, samp)
    s_cluster(c, p, d, ev, top, r01)
    s_placebo(d, ev, samp)
    s_signals_overlap(c, p, samp)
    s_facts(c, p, d, ev, samp, r01)
    s_memo_csv(r01)
    s_rdf(f"{B}/results/r01_rq2_memo.md")
    s_b5(c, p, ev, samp)
    s_scramble(r01)
    bad = s_rerun_r01(r01)
    P(f"\n총 소요 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try:
        main()
    finally:
        with open(f"{B}/results/v01_r01_review_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(LOG))

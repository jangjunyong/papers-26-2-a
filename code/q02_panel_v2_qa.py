"""q02: panel_v2.parquet 검증 (p02_build_panel_v2.py 산출물). q01 감사 항목 B1~B7 해소 여부, 룩어헤드 실험, v1 대비 변화를 계산해
results/p02_panel_v2_report.md 를 새로 쓴다. 기존 파일(p01, panel.parquet, q01 산출물)은 읽기만 한다.
실행: py -3 q02_panel_v2_qa.py [compute|report]   (compute 약 1분, 최대 메모리 약 5.5GB; compute 결과는 캐시 pickle 로 남고 report 는 그걸로 문서만 다시 쓴다)
"""
import importlib.util
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PAPERS = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS"
B = f"{PAPERS}/26-2/03_차트CNN_연구"
CODE, RES = f"{B}/code", f"{B}/results"
CB = f"{PAPERS}/data/daily/club_db"
CACHE = os.environ.get("Q02_CACHE", os.path.join(tempfile.gettempdir(), "q02_cache.pkl"))
OUT_MD = f"{RES}/p02_panel_v2_report.md"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


P02 = imp("p02", f"{CODE}/p02_build_panel_v2.py")
Q01 = imp("q01", f"{CODE}/q01_panel_qa.py")
R = {}

INPUT_COLS = ["open", "high", "low", "close", "adj_close", "volume", "listed_shares", "is_halted", "trading_value"]
LABEL_COLS = ["fwd_ret5", "fwd_censored"]               # 의도적으로 미래를 쓰는 라벨
VINTAGE_COLS = ["exclude_flag", "exclude_reason"]       # 자료 끝(달력)에 따라 정해지는 표시
RANK_COLS = ["indiv_rk", "inst_rk", "foreign_rk"]       # 횡단면 순위: 모집단(종목 집합)이 바뀌면 값이 바뀐다


def tbl(df, idx=True, nd=4):
    d = df.reset_index() if idx else df.copy()
    cols = [str(c) for c in d.columns]
    rows = []
    for r in d.itertuples(index=False):
        out = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                out.append("" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else ("0" if v == 0 else f"{v:.{nd}g}")))
            elif isinstance(v, (int, np.integer)):
                out.append(f"{v:,}")
            elif isinstance(v, pd.Timestamp):
                out.append(str(v.date()))
            else:
                out.append(str(v))
        rows.append(out)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def same(a, b):
    """두 Series 가 같은지(NaN==NaN, 부동소수 허용오차) 불리언 배열."""
    if a.dtype == object or b.dtype == object or str(a.dtype) in ("string", "category"):
        return (a.astype(str).values == b.astype(str).values)
    return np.isclose(a.astype(float).values, b.astype(float).values, rtol=1e-9, atol=1e-12, equal_nan=True)


def cmp_cols(A, Bn, cols, T=None, extra_mask=None):
    m = A[["ticker", "date"] + cols].merge(Bn[["ticker", "date"] + cols], on=["ticker", "date"], suffixes=("_a", "_b"))
    if T is not None:
        m = m[m.date <= T]
    out = {c: int((~same(m[c + "_a"], m[c + "_b"])).sum()) for c in cols}
    return out, len(m)


# ------------------------------------------------------------------ 행 단위 정의 (감사 q01 setup 과 같은 정의를 독립적으로 다시 계산)
def idle_resume(P):
    idle = (P.is_halted == 1) | (P.volume == 0)
    active = (P.volume > 0) & (P.is_halted == 0)
    first = P.ticker.ne(P.ticker.shift())
    res = idle.shift(1, fill_value=False) & ~first & active
    return idle, active, res


def liq_rows(P):
    g = P.groupby("ticker", sort=False)
    dl = g.date.transform("max") < P.date.max() - pd.Timedelta(days=10)
    return dl, (dl & (g.cumcount(ascending=False) < 8))


def terminal(P, use_multi):
    g = P.groupby("ticker", sort=False)
    dl = g.date.transform("max") < P.date.max() - pd.Timedelta(days=10)
    d = P[dl & (g.cumcount(ascending=False) <= 8)].copy()
    _, _, res = idle_resume(P)
    r = d.ret1d_clean
    if use_multi:
        r = r.where(~res.loc[d.index], d.ret_multi)
    d["r"] = r
    gg = d.groupby("ticker")
    out = pd.DataFrame({"clean": gg.r.agg(lambda s: np.prod(1 + s.iloc[1:].fillna(0)) - 1), "true": gg.adj_close.agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)})
    return out[out["true"].abs() < 5]


def lim_eval(P, v2):
    """정확 판정(q01.exact_limits, 손으로 옮긴 호가표·부동소수 산식)을 기준으로 패널의 lim_up/lim_dn 정밀도·재현율."""
    g = P.groupby("ticker", sort=False)
    prev_close = g.close.shift(1)
    prev_vol, prev_h = g.volume.shift(1), g.is_halted.shift(1)
    fac = P.adj_close / P.close
    dfac = fac / fac.groupby(P.ticker, sort=False).shift(1) - 1
    _, active, _ = idle_resume(P)
    _, liq = liq_rows(P)
    elig = (prev_close.notna() & (prev_vol > 0) & (prev_h == 0) & active & (dfac.abs() < 0.01) & ~liq).values
    L = (P.limit_pct / 100).values
    up, dn = Q01.exact_limits(prev_close.fillna(1).values, P.date, L)
    ex_up, ex_dn = (P.close.values >= up) & elig, (P.close.values <= dn) & elig
    out = {}
    for nm, ex, fl in (("상한가", ex_up, P.lim_up.values == 1), ("하한가", ex_dn, P.lim_dn.values == 1)):
        tp = int((ex & fl).sum())
        out[nm] = {"기준(exact)": int(ex.sum()), "패널 표시": int(fl.sum()), "둘 다": tp, "누락": int((ex & ~fl).sum()), "과검출": int((~ex & fl).sum()),
                   "재현율": tp / ex.sum(), "정밀도": tp / fl.sum()}
    if v2:
        # 필요조건 검사(독립): 상한가 표시 행은 고가==종가, 하한가 표시 행은 저가==종가여야 하고, 종가가 한도가를 넘는 행은 없어야 한다
        eras = P02.make_tick_eras(pd.read_csv(f"{P02.IN}/tick_size.csv", parse_dates=["effective_from", "effective_to"]).query("market=='KOSPI'"))
        pu, pd_ = P02.exact_limits(prev_close.fillna(1).values, P.date, P.limit_pct.values, eras)
        out["p02 한도가 vs q01 한도가 불일치(전 행)"] = (int((pu != up).sum()), int((pd_ != dn).sum()))
        ex_ok = (P.exclude_flag == 0).values
        out["종가>상한가 / 종가<하한가 (대상 행) 전체"] = (int(((P.close.values > pu) & elig).sum()), int(((P.close.values < pd_) & elig).sum()))
        out["종가>상한가 / 종가<하한가 (대상 행) 제외 표시 뺀"] = (int(((P.close.values > pu) & elig & ex_ok).sum()), int(((P.close.values < pd_) & elig & ex_ok).sum()))
        out["고가>상한가 / 저가<하한가 (대상 행) 제외 표시 뺀"] = (int(((P.high.values > pu) & elig & ex_ok).sum()), int(((P.low.values < pd_) & elig & ex_ok).sum()))
        u, d_ = (P.lim_up == 1) & (P.exclude_flag == 0), (P.lim_dn == 1) & (P.exclude_flag == 0)
        out["상한가 행 중 고가==종가 / 하한가 행 중 저가==종가 (제외 표시 뺀)"] = (float((P.high[u] == P.close[u]).mean()), float((P.low[d_] == P.close[d_]).mean()))
        out["상한가 행 중 고가==종가 / 하한가 행 중 저가==종가 (전체)"] = (float((P.high[P.lim_up == 1] == P.close[P.lim_up == 1]).mean()), float((P.low[P.lim_dn == 1] == P.close[P.lim_dn == 1]).mean()))
        out["lim_elig 재계산 일치"] = int((elig != (P.lim_elig.values == 1)).sum())
    return out


# ------------------------------------------------------------------ 룩어헤드 실험
def perturb(IN, T, rng, only=None):
    out = {k: v.copy() for k, v in IN.items()}
    for k, v in out.items():
        m = (v.date > T) if only is None else ((v.date > T) & v.code.isin(only))
        for c in v.columns:
            if c not in ("code", "date"):
                v[c] = v[c].astype(float)
        n = int(m.sum())
        f = np.exp(rng.normal(0, 0.35, n))
        if k == "price":
            for c in ("open", "high", "low", "close", "adj_close", "listed_shares"):
                v.loc[m, c] = v.loc[m, c] * f
            v.loc[m, "volume"] = v.loc[m, "volume"] * np.exp(rng.normal(0, 1, n))
            v.loc[m, "is_halted"] = (rng.random(n) < 0.2).astype(float)
        elif k == "investor":
            for c in v.columns:
                if c not in ("code", "date"):
                    v.loc[m, c] = v.loc[m, c] * rng.normal(0, 3, n)
        else:
            v.loc[m, "trading_value"] = v.loc[m, "trading_value"] * np.exp(rng.normal(0, 1, n))
            v.loc[m, "listed_shares"] = v.loc[m, "listed_shares"] * f
    return out


def run_build(IN):
    return P02.build_panel(IN["price"].copy(), IN["investor"].copy(), IN["trading"][["code", "date", "trading_value"]].copy())[0]


def compute():
    log("V2 로드")
    V2 = pd.read_parquet(P02.OUT)
    V1 = pd.read_parquet(f"{CB}/panel.parquet")
    FL = pd.read_parquet(f"{RES}/q01_panel_flags.parquet")
    removed = pd.read_parquet(P02.OUT_REMOVED)
    R["shape"] = {"v1": V1.shape, "v2": V2.shape}
    R["cols_new"] = [c for c in V2.columns if c not in V1.columns]
    R["cols_dropped"] = [c for c in V1.columns if c not in V2.columns]
    R["tickers"] = (V1.ticker.nunique(), V2.ticker.nunique())
    R["sorted_ok"] = bool((V2.ticker.values[1:] >= V2.ticker.values[:-1]).all())
    R["dups"] = int(V2.duplicated(["ticker", "date"]).sum())
    DER = [c for c in V2.columns if c not in INPUT_COLS + ["ticker", "date"]]
    R["der_cols"] = DER

    # ---- 플래그 파일과 교차 확인 (q01_panel_flags.parquet)
    FLq = FL.rename(columns={c: "q_" + c for c in FL.columns if c not in ("ticker", "date")})
    M = V2[["ticker", "date", "f_tradable", "resume_day", "liq_zone", "f_zero_inv_suspect", "f_net_gt_volume", "tick"]].merge(FLq, on=["ticker", "date"], how="left")
    R["flag_cross"] = {"v2 행 중 플래그 파일에 없는 행": int(M.q_f_idle.isna().sum()),
                       "f_tradable 불일치": int((M.f_tradable != M.q_f_tradable).sum()),
                       "resume_day vs f_resume_day 불일치": int((M.resume_day != M.q_f_resume_day).sum()),
                       "liq_zone vs f_liquidation_zone 불일치": int((M.liq_zone != M.q_f_liquidation_zone).sum()),
                       "f_zero_inv_suspect 불일치": int((M.f_zero_inv_suspect != M.q_f_zero_inv_suspect).sum()),
                       "f_net_gt_volume 불일치": int((M.f_net_gt_volume != M.q_f_net_gt_volume).sum()),
                       "tick vs tick_fixed 불일치": int((M.tick != M.q_tick_fixed.astype(float)).sum())}
    R["flag_totals"] = {c: int(FL[c].sum()) for c in ("f_resume_day", "f_prelist_otc", "f_liquidation_zone", "f_zero_inv_suspect", "f_net_gt_volume", "f_idle", "f_tradable")}

    # ---- B1 거래재개일
    idle, active, res = idle_resume(V2)
    idle1, active1, res1 = idle_resume(V1)
    R["b1"] = {"재개일 행(독립 재계산)": int(res.sum()), "재개일 ret1d 값 있음": int(V2.ret1d[res].notna().sum()), "재개일 ret1d_clean 값 있음": int(V2.ret1d_clean[res].notna().sum()),
               "재개일 ret_multi 값 있음": int(V2.ret_multi[res].notna().sum()), "재개일 아닌데 ret_multi 값 있음": int(V2.ret_multi[~res].notna().sum()),
               "v1 재개일 행": int(res1.sum()), "v1 재개일 ret1d 값 있음": int(V1.ret1d[res1].notna().sum()), "v1 재개일 ret1d_clean 값 있음": int(V1.ret1d_clean[res1].notna().sum()),
               "v1 재개일 |ret1d_clean| 평균": float(V1.ret1d_clean[res1].abs().mean()), "v2 정상일 |ret1d_clean| 평균": float(V2.ret1d_clean[~res].abs().mean()),
               "v1 재개일 ret1d 5%p 초과 행": int((V1.ret1d[res1].abs() > 0.10).sum()), "ret_multi |값| 중앙값": float(V2.ret_multi[res].abs().median())}
    # vol20: 20행 슬라이스로 직접 계산
    tk = V2.ticker.values
    first_idx = pd.Series(np.arange(len(V2))).groupby(tk).transform("min").values
    cand = V2.index[V2.vol20.notna()].values
    pick = np.random.default_rng(7).choice(cand, 3000, replace=False)
    rc = V2.ret1d_clean.values
    bad = 0
    for i in pick:
        v = pd.Series(rc[max(first_idx[i], i - 19):i + 1]).std()
        bad += int(not np.isclose(v, V2.vol20.values[i], rtol=1e-9, atol=1e-12))
    resume_in20 = res.groupby(V2.ticker, sort=False).transform(lambda s: s.astype(int).rolling(20, min_periods=1).max())
    resume_in20_1 = res1.groupby(V1.ticker, sort=False).transform(lambda s: s.astype(int).rolling(20, min_periods=1).max())
    R["b9"] = {"표본 3000행 vol20 직접계산 불일치": bad, "v2 vol20 창에 재개일 포함(그 날 값은 NaN) 행": int((V2.vol20.notna() & (resume_in20 == 1)).sum()),
               "v1 vol20 창에 재개일 누적수익이 섞인 행": int((V1.vol20.notna() & (resume_in20_1 == 1)).sum()),
               "v2 vol20 창에 재개일 누적수익이 섞인 행": int(((V2.ret1d_clean.notna() & res).groupby(V2.ticker, sort=False).transform(lambda s: s.astype(int).rolling(20, min_periods=1).max()) == 1)[V2.vol20.notna()].sum()),
               "v1 vol20 최대": float(V1.vol20.max()), "v2 vol20 최대": float(V2.vol20.max()), "v1 vol20>0.15 행": int((V1.vol20 > 0.15).sum()), "v2 vol20>0.15 행": int((V2.vol20 > 0.15).sum())}

    # ---- B2 정리매매·상폐 손실
    sub_tk = None
    t1, t2 = terminal(V1, False), terminal(V2, True)
    t2n = terminal(V2, False)
    R["b2_all"] = {"종목수": len(t1), "실제 평균": float(t1["true"].mean()), "v1 clean 복리 평균": float(t1.clean.mean()), "v2 clean 복리 평균(재개일은 ret_multi)": float(t2.clean.mean()),
                   "v2 clean 복리 평균(ret_multi 보충 없음)": float(t2n.clean.mean()),
                   "v1 평균오차": float((t1.clean - t1["true"]).abs().mean()), "v2 평균오차": float((t2.clean - t2["true"]).abs().mean()), "v2 평균오차(보충 없음)": float((t2n.clean - t2n["true"]).abs().mean()),
                   "실제 중앙값": float(t1["true"].median()), "v1 중앙값": float(t1.clean.median()), "v2 중앙값": float(t2.clean.median())}
    sub, IN0, _ = Q01.subset(V1)
    R["sub"] = sub
    t1s, t2s = t1[t1.index.isin(sub)], t2[t2.index.isin(sub)]
    t2sn = t2n[t2n.index.isin(sub)]
    R["b2_sub"] = {"종목수": len(t1s), "실제 평균": float(t1s["true"].mean()), "v1 clean 복리 평균": float(t1s.clean.mean()), "v2 clean 복리 평균": float(t2s.clean.mean()),
                   "v1 평균오차": float((t1s.clean - t1s["true"]).abs().mean()), "v2 평균오차": float((t2s.clean - t2s["true"]).abs().mean()), "v2 평균오차(보충 없음)": float((t2sn.clean - t2sn["true"]).abs().mean())}
    _, liq1 = liq_rows(V1)
    _, liq2 = liq_rows(V2)
    R["b2_liq"] = {"정리매매 구간 행": (int(liq1.sum()), int(liq2.sum())), "그중 ret1d_clean 값 있음 v1 / v2": (int(V1.ret1d_clean[liq1].notna().sum()), int(V2.ret1d_clean[liq2].notna().sum())),
                   "그중 ret_outlier=1 v1 / v2": (int(V1.ret_outlier[liq1].sum()), int(V2.ret_outlier[liq2].sum())),
                   "그중 극저가 표시(f_penny) v2": int(V2.f_penny[liq2].sum()), "그중 정지/거래량0 일 v2": int(idle[liq2].sum()),
                   "그중 거래일(f_tradable) v2": int(active[liq2].sum()), "그중 거래일이면서 ret1d_clean 값 있음 v1 / v2": (int((V1.ret1d_clean.notna() & active1)[liq1].sum()), int((V2.ret1d_clean.notna() & active)[liq2].sum())),
                   "그중 거래일이면서 ret_outlier=1 v1 / v2": (int((V1.ret_outlier == 1)[liq1 & active1].sum()), int((V2.ret_outlier == 1)[liq2 & active].sum())),
                   "그중 거래일이면서 극저가(f_penny) 행 / 종목": (int(((V2.f_penny == 1) & active)[liq2].sum()), int(V2[liq2 & active & (V2.f_penny == 1)].ticker.nunique())),
                   "그중 극저가 종목(exclude 뺀)": int(V2[liq2 & (V2.f_penny == 1) & (V2.exclude_flag == 0)].ticker.nunique()),
                   "그중 거래일 극저가 행(exclude 뺀)": int(((V2.f_penny == 1) & active & (V2.exclude_flag == 0))[liq2].sum()),
                   "그중 ret_multi 값 있음(= 정리매매 첫날 재개일) / 평균": (int(V2.ret_multi[liq2].notna().sum()), float(V2.ret_multi[liq2 & (V2.exclude_flag == 0)].mean())),
                   "정리매매 구간 ret1d_clean 평균 v1 / v2": (float(V1.ret1d_clean[liq1].mean()), float(V2.ret1d_clean[liq2].mean())),
                   "정리매매 구간 ret1d_clean 최솟값 v1 / v2": (float(V1.ret1d_clean[liq1].min()), float(V2.ret1d_clean[liq2].min()))}
    ol2 = V2[V2.ret_outlier == 1]
    pen_mask = (V2.f_penny == 1) & (V2.ret_outlier == 1)
    R["outlier"] = {"v1 ret_outlier": int(V1.ret_outlier.sum()), "v2 ret_outlier": int(V2.ret_outlier.sum()), "v2 f_penny 표시": int(V2.f_penny.sum()), "f_penny 이면서 ret_outlier=1": int(pen_mask.sum()),
                    "정리매매 실제 호가라 비우지 않은 극저가 행": int(((V2.f_penny == 1) & (V2.ret_outlier == 0) & liq2).sum()), "극저가 아닌 이상값": int((ol2.f_penny == 0).sum()),
                    "극저가로 비운 상위 종목": ol2[ol2.f_penny == 1].groupby("ticker").size().sort_values(ascending=False).head(5).to_dict(),
                    "극저가 아닌 이상값 종목": ol2[ol2.f_penny == 0].groupby("ticker").size().sort_values(ascending=False).head(5).to_dict(),
                    "극저가로 비운 행 중 exclude_flag=1": int((pen_mask & (V2.exclude_flag == 1)).sum())}
    # 라벨
    ok = V2.fwd_censored == 0
    g2 = V2.groupby("ticker", sort=False)
    direct = g2.adj_close.shift(-5) / V2.adj_close - 1
    v1style = V2.fwd_ret5.where(V2.fwd_censored == 0)                # v1 방식 라벨: 5행 뒤가 없으면 NaN(상폐 직전 행은 빠짐)
    clean_rows = V2.exclude_flag == 0
    R["fwd"] = {"fwd_censored 분포": V2.fwd_censored.value_counts().sort_index().to_dict(), "완전창 행 직접계산 불일치": int((~same(V2.fwd_ret5[ok], direct[ok])).sum()),
                "상폐로 잘린 행(코드1) 중 값 있음": int(V2.fwd_ret5[V2.fwd_censored == 1].notna().sum()), "그 평균": float(V2.fwd_ret5[V2.fwd_censored == 1].mean()),
                "그 중앙값": float(V2.fwd_ret5[V2.fwd_censored == 1].median()), "그 중 -30% 이하 행": int((V2.fwd_ret5[V2.fwd_censored == 1] <= -0.30).sum()),
                "자료끝 행(코드2) 값 있음": int(V2.fwd_ret5[V2.fwd_censored == 2].notna().sum()), "v1식 라벨 유효 행": int(v1style.notna().sum()),
                "v2 라벨 유효 행": int(V2.fwd_ret5.notna().sum()), "추가된 행": int((V2.fwd_ret5.notna() & v1style.isna()).sum()),
                "정리매매 구간 행(제외 표시 뺀) fwd_ret5 평균 v1식 / v2": (float(v1style[liq2 & clean_rows].mean()), float(V2.fwd_ret5[liq2 & clean_rows].mean())),
                "정리매매 구간 행(제외 표시 뺀) 유효 행 v1식 / v2": (int(v1style[liq2 & clean_rows].notna().sum()), int(V2.fwd_ret5[liq2 & clean_rows].notna().sum())),
                "전체(제외 표시 뺀) fwd_ret5 평균 v1식 / v2": (float(v1style[clean_rows].mean()), float(V2.fwd_ret5[clean_rows].mean()))}

    # ---- B3 호가단위
    et = Q01.tick_vec(V2.close.values, V2.date)
    era = np.where(V2.date < "2010-10-04", "A ~2010-10-01", np.where(V2.date < "2023-01-25", "B 2010-10-04~2023-01-24", "C 2023-01-25~"))
    tt = pd.DataFrame({"era": era, "mism": ~np.isclose(V2.tick.values, et)}).groupby("era").mism.agg(["size", "sum"])
    et1 = Q01.tick_vec(V1.close.values, V1.date)
    era1 = np.where(V1.date < "2010-10-04", "A ~2010-10-01", np.where(V1.date < "2023-01-25", "B 2010-10-04~2023-01-24", "C 2023-01-25~"))
    tt1 = pd.DataFrame({"era": era1, "mism": ~np.isclose(V1.tick.values, et1)}).groupby("era").mism.agg(["size", "sum"])
    R["b3"] = pd.DataFrame({"행": tt["size"], "v1 불일치": tt1["sum"], "v2 불일치": tt["sum"]})
    a = V2.date < "2010-10-04"
    a1 = V1.date < "2010-10-04"
    R["b3_med"] = {"v1 tick_pct 중앙값(%)": float(V1.tick_pct[a1].median()), "v2": float(V2.tick_pct[a].median()), "2010-10-04~2010-12-31 v2": float(V2.tick_pct[(V2.date >= "2010-10-04") & (V2.date < "2011-01-01")].median()),
                   "tick_assumed=1 행": int(V2.tick_assumed.sum()), "tick_assumed 정의 일치": bool(((V2.date < "2010-10-04") == (V2.tick_assumed == 1)).all())}

    # ---- B4 K-OTC
    fl_otc = FL[FL.f_prelist_otc == 1][["ticker", "date"]]
    x = fl_otc.merge(V2[["ticker", "date"]], on=["ticker", "date"], how="inner")
    rm = removed[["ticker", "date"]].merge(fl_otc, on=["ticker", "date"], how="outer", indicator=True)
    otc_t = sorted(fl_otc.ticker.unique())
    firsts = V2[V2.ticker.isin(otc_t)].groupby("ticker").date.min()
    R["b4"] = {"감사 K-OTC 행(플래그)": len(fl_otc), "v2 에 남은 그 행": len(x), "삭제 목록 행": len(removed), "삭제 목록 == 플래그 행 집합": bool((rm._merge == "both").all()),
               "K-OTC 종목": otc_t, "삭제 후 첫 날짜": {t: str(firsts[t].date()) for t in otc_t},
               "삭제 후 해당 종목 투자자 자료 NaN 인 선두 행": int(sum(((V2[V2.ticker == t].date < V2[(V2.ticker == t) & V2.indiv_net.notna()].date.min())).sum() for t in otc_t)),
               "v2 행 수": len(V2), "v1 행 수": len(V1), "차이": len(V1) - len(V2),
               "v1 K-OTC 행 ret_outlier": int(V1.merge(fl_otc, on=["ticker", "date"]).ret_outlier.sum())}

    # ---- B5 투자자
    nets4 = ["indiv_net", "inst_net", "foreign_net", "other_corp_net"]
    z2 = (V2[nets4] == 0).all(axis=1) & (V2.volume >= 10000)
    z1 = (V1[nets4] == 0).all(axis=1) & (V1.volume >= 10000)
    o2 = V2[["indiv_net", "inst_net", "foreign_net"]].abs().gt(V2.volume * 1.0001, axis=0).any(axis=1)
    o1 = V1[["indiv_net", "inst_net", "foreign_net"]].abs().gt(V1.volume * 1.0001, axis=0).any(axis=1)
    tot = V2[nets4].sum(axis=1, min_count=4)
    closed = V2.inv_sum_closed == 1
    R["b5"] = {"4그룹 모두 0 & 거래량≥1만 v1 / v2": (int(z1.sum()), int(z2.sum())), "순매수>거래량 v1 / v2": (int(o1.sum()), int(o2.sum())),
               "NaN 처리된 행(0 코딩)": int(V2.f_zero_inv_suspect.sum()), "NaN 처리된 행(순매수>거래량)": int(V2.f_net_gt_volume.sum()),
               "NaN 처리 행의 투자자 8열이 모두 NaN": bool(V2.loc[(V2.f_zero_inv_suspect == 1) | (V2.f_net_gt_volume == 1), ["indiv_buy", "indiv_sell", "indiv_net", "inst_buy", "inst_net", "foreign_buy", "foreign_net", "other_corp_net"]].isna().all().all()),
               "순매수>거래량 NaN 처리 날짜": V2[V2.f_net_gt_volume == 1].groupby("date").size().to_dict(),
               "inv_sum_closed=1 구간 4그룹 합 계산 가능 행": int(tot[closed].notna().sum()), "그중 합≠0 행": int((tot[closed].notna() & (tot[closed] != 0)).sum()),
               "inv_sum_closed=0 & 2012-05-02 이전 합≠0 비율": float((tot[(V2.date <= "2012-05-02")].dropna() != 0).mean()),
               "inv_sum_closed 행 / 0인 행": (int(closed.sum()), int((~closed).sum())),
               "foreign_def_changed=1 행 / 첫 날짜": (int(V2.foreign_def_changed.sum()), str(V2.date[V2.foreign_def_changed == 1].min().date())),
               "2026-05-26~08-06 4그룹 합≠0 비율": float((tot[(V2.date >= "2026-05-26") & (V2.date <= "2026-08-06")].dropna() != 0).mean())}

    # ---- B6 flow
    rows = []
    for c in ("indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv"):
        for nm, P in (("v1", V1), ("v2", V2)):
            s = P[c]
            rows.append((c, nm, s.min(), s.max(), int((s.abs() > 1 + 1e-9).sum()), int((s.abs() > 5).sum()), s.std(), s.kurt(), int(s.notna().sum())))
    R["b6"] = pd.DataFrame(rows, columns=["열", "버전", "최솟값", "최댓값", "|x|>1", "|x|>5", "표준편차", "첨도", "비결측 행"])
    rk = {}
    for c in RANK_COLS:
        s = V2[c]
        n_by_date = s.groupby(V2.date).count()
        rk[c] = {"최솟값": s.min(), "최댓값": s.max(), "날짜별 평균 절대편차 평균": float((s.groupby(V2.date).mean()).abs().mean()), "비결측 행": int(s.notna().sum()), "정지·거래량0 행 중 값 있음": int(s[idle].notna().sum()),
                 "날짜별 유효 종목수 최솟값": int(n_by_date.min())}
    R["rk"] = pd.DataFrame(rk).T
    R["b6_w_absent"] = [c for c in V2.columns if c.endswith("_w")]
    R["b6_idle_flow"] = {"v1 정지/거래량0 일 flow==0 행": int((V1[["indiv_flow"]].eq(0).squeeze() & idle1).sum()), "v2 정지/거래량0 일 flow 값 있음": int(V2.indiv_flow[idle].notna().sum())}
    # flow 분모 재계산(독립): 무작위 2500행을 행 슬라이스로 직접 계산
    pick = np.random.default_rng(3).choice(V2.index[V2.indiv_flow.notna() & (V2.indiv_flow != 0)].values, 2500, replace=False)
    fac = (V2.adj_close / V2.close).values
    vol = V2.volume.values
    n_chk = ok_n = 0
    for i in pick:
        vals = [vol[j] / fac[j] for j in range(max(first_idx[i], i - 60), i) if vol[j] > 0]
        if len(vals) < 30:
            continue
        av = np.mean(vals) * fac[i]
        n_chk += 1
        ok_n += int(np.isclose(V2.indiv_net.values[i] / max(av, vol[i]), V2.indiv_flow.values[i], rtol=1e-6))
    R["b6_den"] = (n_chk, ok_n)

    # ---- B7 상하한가
    R["b7_v1"] = lim_eval(V1, False)
    R["b7_v2"] = lim_eval(V2, True)
    yy = V2.date.dt.year
    R["b7_year"] = pd.DataFrame({"v1 상한": V1.groupby(V1.date.dt.year).lim_up.sum(), "v2 상한": V2.groupby(yy).lim_up.sum(), "v1 하한": V1.groupby(V1.date.dt.year).lim_dn.sum(), "v2 하한": V2.groupby(yy).lim_dn.sum()})
    R["b7_counts"] = {"v1 상한/하한": (int(V1.lim_up.sum()), int(V1.lim_dn.sum())), "v2 상한/하한": (int(V2.lim_up.sum()), int(V2.lim_dn.sum())),
                      "v2 lim_elig=0 행": int((V2.lim_elig == 0).sum()), "v2 재개일 중 lim 표시": int((V2.lim_up + V2.lim_dn)[res].sum()), "v1 재개일 중 lim 표시": (int(V1.lim_up[res1].sum()), int(V1.lim_dn[res1].sum())),
                      "v2 정리매매 구간 lim 표시": int((V2.lim_up + V2.lim_dn)[liq2].sum()), "v1 정리매매 구간 lim 표시": (int(V1.lim_up[liq1].sum()), int(V1.lim_dn[liq1].sum()))}

    # ---- 제외 표시
    ex = V2.exclude_reason
    R["excl"] = {"exclude_flag=1 행": int(V2.exclude_flag.sum()), "비트1 자료 끝 3거래일": int(((ex & 1) > 0).sum()), "비트2 008080·000790": int(((ex & 2) > 0).sum()), "비트4 006740 2026-06-26~": int(((ex & 4) > 0).sum()),
                 "마지막 3거래일": [str(pd.Timestamp(d).date()) for d in np.sort(V2.date.unique())[-3:]],
                 "mktcap_missing 행": int(V2.mktcap_missing.sum()), "mktcap_missing 첫 날짜(2026)": str(V2.date[(V2.mktcap_missing == 1) & (V2.date >= "2026-01-01")].min().date()),
                 "f_tradable=1 비율": float(V2.f_tradable.mean())}

    # ---- v1 대비 변화
    i1 = V1[["ticker", "date"]].reset_index().rename(columns={"index": "i1"}).merge(V2[["ticker", "date"]].reset_index().rename(columns={"index": "i2"}), on=["ticker", "date"])
    a, b = i1.i1.values, i1.i2.values
    chg = []
    for c in [c for c in V2.columns if c in V1.columns and c not in ("ticker", "date")]:
        x1, x2 = V1[c].iloc[a].reset_index(drop=True), V2[c].iloc[b].reset_index(drop=True)
        eq = same(x1, x2)
        n1, n2 = x1.isna().values, x2.isna().values
        chg.append((c, int((~eq).sum()), float((~eq).mean()), int((n1 & ~n2).sum()), int((~n1 & n2).sum()), int((~eq & ~n1 & ~n2).sum())))
    R["chg"] = pd.DataFrame(chg, columns=["열", "바뀐 행", "비율", "NaN→값", "값→NaN", "값→다른 값"]).set_index("열")
    R["chg_n"] = len(i1)
    o1, o2 = V1.ret_outlier.values[a], V2.ret_outlier.values[b]
    R["outlier"]["v1→v2 ret_outlier 0→1 / 1→0"] = (int(((o1 == 0) & (o2 == 1)).sum()), int(((o1 == 1) & (o2 == 0)).sum()))
    R["outlier"]["0→1 중 exclude_flag=0"] = int(((o1 == 0) & (o2 == 1) & (V2.exclude_flag.values[b] == 0)).sum())
    del V1, FL, M
    log("v1 대비 변화 완료")

    # ---- 룩어헤드 (subset 118종목)
    T = pd.Timestamp("2019-06-28")
    rng = np.random.default_rng(20261011)
    base = run_build(IN0)
    R["lh_rows"] = (len(sub), len(base))
    log("subset 재현", len(base))
    stored = V2[V2.ticker.isin(sub)]
    nr = [c for c in DER if c not in RANK_COLS]
    d_rep, n_rep = cmp_cols(base, stored, nr)
    R["lh_rep"] = (n_rep, d_rep)
    # 1) 절단
    INt = {k: v[v.date <= T].copy() for k, v in IN0.items()}
    trunc = run_build(INt)
    feat = [c for c in DER if c not in LABEL_COLS + VINTAGE_COLS]
    d_tr, n_tr = cmp_cols(trunc, base, DER, T)
    R["lh_trunc"] = (n_tr, d_tr)
    log("절단 완료")
    # 2) 교란
    pert = run_build(perturb(IN0, T, rng))
    d_pt, n_pt = cmp_cols(pert, base, DER, T)
    d_af, n_af = cmp_cols(pert, base, DER)
    R["lh_pert"] = (n_pt, d_pt, {c: d_af[c] - d_pt[c] for c in DER}, n_af - n_pt)
    log("교란 완료")
    # 3) 한 종목(005930)만 교란 → 다른 종목: 순위 열은 T 이후 날짜 제외, 나머지는 전 기간
    p1 = run_build(perturb(IN0, T, rng, only={"A005930"}))
    oth = [t for t in sub if t != "005930"]
    A, Bn = p1[p1.ticker.isin(oth)], base[base.ticker.isin(oth)]
    d_o1, n_o1 = cmp_cols(A, Bn, [c for c in DER if c not in RANK_COLS])
    d_o2, n_o2 = cmp_cols(A, Bn, RANK_COLS, T)
    d_o3, _ = cmp_cols(A, Bn, RANK_COLS)
    R["lh_one"] = (n_o1, d_o1, n_o2, d_o2, {c: d_o3[c] - d_o2[c] for c in RANK_COLS})
    # 4) 양성 대조
    for X in (base, pert):
        gg = X.groupby("ticker", sort=False)
        X["ctrl_fwd5"] = gg.adj_close.shift(-5) / X.adj_close - 1
        X["ctrl_center"] = gg.adj_close.transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
    dc, _ = cmp_cols(pert, base, ["ctrl_fwd5", "ctrl_center"], T)
    R["lh_ctrl"] = dc
    # 5) 순위 열: 같은 날 횡단면 재계산 확인
    r = base.copy()
    ok = r.f_tradable == 1
    manual = r.indiv_flow.where(ok).groupby(r.date).rank(pct=True) - 0.5
    R["lh_rank_manual"] = int((~same(manual, r.indiv_rk)).sum())
    R["lh_info"] = {"기준일": str(T.date()), "표본 종목": len(sub), "교란 후 T 이후 행 수": n_af - n_pt}
    log("룩어헤드 완료")
    pd.to_pickle(R, CACHE)
    return R


# ------------------------------------------------------------------ 보고서
def f0(x):
    return f"{int(round(x)):,}"


def pc(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def b7_c(R, key):
    v = R["b7_counts"][key]
    return f"{v[0]}+{v[1]}"


def pair(t, sep=" / "):
    return sep.join(f0(v) if isinstance(v, (int, np.integer)) else str(v) for v in t)


def dstr(d):
    return ", ".join(f"{k} {f0(v)}" if isinstance(v, (int, np.integer)) else f"{k} {v}" for k, v in d.items())


DICT_NEW = [
    ("resume_day", "int8", "거래재개일: 직전 행이 정지/거래량0 이고 오늘은 거래(f_tradable=1). q01 f_resume_day 와 같은 정의(2,847행)"),
    ("ret_multi", "float", "재개일의 여러 날 누적 수익(마지막 거래 종가 대비). 재개일에만 값, 나머지 NaN. 이 날 ret1d·ret1d_clean 은 NaN"),
    ("liq_zone", "int8", "상폐 종목 마지막 8행(정리매매 7거래일 + 재개일) = 158종목 x 8. 상폐 여부를 아는 사후 표시"),
    ("f_penny", "int8", "수정주가 <= 100 이거나 전일 수정주가 <= 100. 반올림 잡음 구간 표시. 정리매매 구간의 실제 호가는 표시만 하고 ret_outlier 로는 비우지 않음"),
    ("indiv_rk / inst_rk / foreign_rk", "float", "날짜별 횡단면 순위 `rank(pct=True) - 0.5` (대상: f_tradable=1 이고 해당 *_flow 비결측, 범위 (-0.5, 0.5]). `_w` 대체"),
    ("tick_assumed", "int8", "1 이면 2010-10-04 이전이라 구호가표(2차 자료)로 채운 tick"),
    ("f_zero_inv_suspect", "int8", "4그룹 순매수가 모두 0 인데 거래량 1만 주 이상 → 결측의 0 코딩으로 보고 투자자 8열을 NaN 처리한 행 (18,275행)"),
    ("f_net_gt_volume", "int8", "순매수 절대값이 거래량 x 1.0001 초과 → 투자자 8열을 NaN 처리한 행 (172행: 2026-09-08 2행 + 10-06 36행 + 10-07 134행)"),
    ("foreign_def_changed", "int8", "date >= 2026-05-26: foreign_net 이 기타외국인을 제외한 정의로 바뀐 것으로 추정(값은 그대로, 표시만)"),
    ("inv_sum_closed", "int8", "1 = 2012-05-03 ~ 2026-05-25. 4그룹 순매수 합이 0 으로 닫힘이 확인된 구간. 그 밖(2010-01~2012-05-02, 2026-05-26~)은 0"),
    ("lim_elig", "int8", "상하한가 판정 가능한 행(직전 행 정상 거래, 조정계수 단차 < 1%, 정리매매 구간 아님, 당일 거래). 0 이면 lim_up/lim_dn=0 이 '한도 아님'이 아니라 '판정 불가'"),
    ("f_tradable", "int8", "volume > 0 이고 is_halted == 0 (q01 f_tradable 과 동일)"),
    ("exclude_flag", "int8", "분석에서 뺄 행: exclude_reason > 0"),
    ("exclude_reason", "int8", "비트마스크. 1 = 자료 끝 3거래일(2026-10-06~08), 2 = 008080·000790 전 기간(수정주가 1~4원/종가 10~115원 아티팩트), 4 = 006740 의 2026-06-26~ (거래량·거래대금 단위 불일치)"),
    ("mktcap_missing", "int8", "mktcap 이 NaN 인 행(2026-06-23 이후 listed_shares 공백). 표시만 하고 채우지 않음"),
    ("fwd_ret5", "float", "라벨: 5거래일 뒤 수정주가 수익률 `adj[t+5]/adj[t]-1`. 상폐 종목은 5행이 안 남으면 마지막 가격까지의 수익(정리매매 손실 포함). 의도적으로 미래를 쓰는 열"),
    ("fwd_censored", "int8", "fwd_ret5 의 창 상태. 0 = 완전한 5거래일, 1 = 상폐로 창이 잘림(값은 마지막 가격까지, 상폐 마지막 행은 NaN), 2 = 자료 끝이라 미관측(NaN)"),
]
DICT_CHG = [
    ("ret1d", "재개일(2,847행)이 NaN 으로 바뀜(누적 수익은 ret_multi 로 이동). K-OTC 삭제 후 5종목의 첫 행도 NaN"),
    ("ret_outlier", "정의 변경: (|수익| > 한도 + 10%p, 정리매매 구간 제외) 또는 극저가(f_penny, 정리매매 실제 호가 제외). 재개일은 ret_multi 로 판정. 298행 → 1,219행(대부분 008080·000790 극저가)"),
    ("ret1d_clean", "정리매매 구간의 실제 손실이 남음(최솟값 -38.8% → -88.0%). 재개일은 NaN"),
    ("indiv_buy ... other_corp_net (투자자 8열)", "f_zero_inv_suspect 18,275행 + f_net_gt_volume 172행이 NaN"),
    ("indiv_flow / inst_flow / foreign_flow", "분모 = max(분할 보정 60일 평균 거래량(정지일 제외, 당일 제외), 당일 거래량). 범위 [-1, 1]. 정지·거래량0 일은 0 이 아니라 NaN"),
    ("indiv_flow_tv", "같은 방식으로 [-1, 1]: 분모 = max(정지일 제외 60일 평균 거래대금, 종가 x 거래량) (감사 P6 에 없던 확장)"),
    ("lim_up / lim_dn", "호가단위 포함 정확 판정. 재개일·조정계수 단차일·정리매매 구간은 0 (lim_elig=0)"),
    ("tick / tick_pct", "2010-10-04 이전도 구호가표로 채움(tick_assumed=1). 이전엔 1.0 일괄"),
    ("vol20", "ret1d_clean 의 20행 표준편차(최소 15). 재개일 누적 수익·이상값이 섞이지 않음"),
    ("str5 / str20 / mom12_1 / near52h / abvol", "정의는 같음. ret_outlier 정의 변경과 K-OTC 행 삭제 때문에 소수 행 값이 바뀜"),
    ("(행)", "상장 전 K-OTC 1,963행 삭제(5종목). 정렬 ticker, date 유지"),
]
DICT_SAME = "date, open, high, low, close, adj_close, volume, listed_shares, is_halted, trading_value, ticker, mktcap, tv_filled, tv_is_approx, limit_pct, short_regime, short_banned, sell_tax_pct, sell_tax_assumed, hl_range"


def report(R):
    L = []
    w = L.append
    b1, b9, b2a, b2s, b2l, ol, fw = R["b1"], R["b9"], R["b2_all"], R["b2_sub"], R["b2_liq"], R["outlier"], R["fwd"]
    b3, b3m, b4, b5, b6, rk = R["b3"], R["b3_med"], R["b4"], R["b5"], R["b6"], R["rk"]
    b7a, b7b, ex = R["b7_v1"], R["b7_v2"], R["excl"]
    lh_rep, lh_tr, lh_pt, lh_one = R["lh_rep"], R["lh_trunc"], R["lh_pert"], R["lh_one"]
    LAB, VIN = ["fwd_ret5", "fwd_censored"], ["exclude_flag", "exclude_reason"]
    feat = [c for c in R["der_cols"] if c not in LAB + VIN]
    feat_nr = [c for c in feat if c not in ("indiv_rk", "inst_rk", "foreign_rk")]
    n_feat = len(feat)
    mis_tr = sum(lh_tr[1][c] for c in feat)
    mis_pt = sum(lh_pt[1][c] for c in feat)
    mis_rep = sum(lh_rep[1][c] for c in feat_nr)
    mis_one = sum(lh_one[1][c] for c in feat_nr) + sum(lh_one[3].values())
    sh1, sh2 = R["shape"]["v1"], R["shape"]["v2"]
    f1 = b6[(b6["열"] == "indiv_flow") & (b6["버전"] == "v1")].iloc[0]
    liq_pen = b2l["그중 거래일이면서 극저가(f_penny) 행 / 종목"]

    w("---")
    w("title: 차트 CNN 분석 패널 v2 (감사 패치 반영) 생성·검증 보고서")
    w("created: 2026-10-11")
    w("updated: 2026-10-11")
    w("tags: [dev/python, finance/퀀트]")
    w("source: results/q01_panel_qa.md (감사 B1~B13, 패치 P1~P7, 사용 지침) / code/p02_build_panel_v2.py / code/q02_panel_v2_qa.py 가 이 문서를 자동 생성")
    w("status: seed")
    w("---\n")
    w("# 차트 CNN 분석 패널 v2 생성·검증 보고서 (p02 / q02)\n")
    w(f"> `data/daily/club_db/panel_v2.parquet`: {f0(sh2[0])}행 x {sh2[1]}열 (v1 `panel.parquet` {f0(sh1[0])}행 x {sh1[1]}열), KOSPI 2010-01-04 ~ 2026-10-08, {R['tickers'][1]}종목. "
      "p01·panel.parquet·q01 산출물은 수정하지 않았고 새 파일만 만들었다. 이 문서는 `code/q02_panel_v2_qa.py` 를 실행할 때마다 다시 만들어진다(약 1분).")
    w("> 생성: `py -3 code/p02_build_panel_v2.py` 약 40초, 최대 메모리 약 4.0GB (노트북 CPU). 삭제된 행 목록은 `data/daily/club_db/panel_v2_removed_rows.parquet`.\n")

    w("## 0. 결론\n")
    w("**감사 q01 의 높음·중간 버그 B1~B7 과 B9 는 모두 해소됐고, 룩어헤드 실험에서 특징 열의 불일치는 0건이다.** 해소하지 못한 것은 5절에 있다(외국인 정의 변경 복원은 kiwoom `foreign_etc` 가 필요하다).\n")
    t = [
        ("B1", "재개일에 단일일 수익으로 남은 행 (ret1d / ret1d_clean)", f"{f0(b1['v1 재개일 ret1d 값 있음'])} / {f0(b1['v1 재개일 ret1d_clean 값 있음'])}", f"{b1['재개일 ret1d 값 있음']} / {b1['재개일 ret1d_clean 값 있음']} (ret_multi {f0(b1['재개일 ret_multi 값 있음'])}행 보관)", "해소"),
        ("B9", "vol20 창에 재개일 누적 수익이 섞인 행", f0(b9["v1 vol20 창에 재개일 누적수익이 섞인 행"]), str(b9["v2 vol20 창에 재개일 누적수익이 섞인 행"]), "해소"),
        ("B2", "상폐 종단 8일 수익 평균오차 (q01 과 같은 19종목 표본)", pc(b2s["v1 평균오차"]), f"{pc(b2s['v2 평균오차'])} (감사 패치 검증값 3.0%)", "해소"),
        ("B2", "같은 지표, 상폐 종목 전체 (158종목 중 |실제|<500%% 인 %d종목)" % b2a["종목수"], pc(b2a["v1 평균오차"]), pc(b2a["v2 평균오차"], 2), "해소"),
        ("B3", "2010-10-04 이전 tick 불일치 행", f0(b3.loc["A ~2010-10-01", "v1 불일치"]), str(b3.loc["A ~2010-10-01", "v2 불일치"]), "해소"),
        ("B4", "상장 전 K-OTC 행", f0(b4["감사 K-OTC 행(플래그)"]), f"{b4['v2 에 남은 그 행']} (삭제 {f0(b4['삭제 목록 행'])}행 목록 보관)", "해소"),
        ("B5", "투자자 4그룹 0 코딩(거래량 1만 이상) / 순매수>거래량 행", f"{f0(b5['4그룹 모두 0 & 거래량≥1만 v1 / v2'][0])} / {b5['순매수>거래량 v1 / v2'][0]}", f"{b5['4그룹 모두 0 & 거래량≥1만 v1 / v2'][1]} / {b5['순매수>거래량 v1 / v2'][1]} (NaN 처리)", "해소 (정의 변경 구간은 표시만)"),
        ("B6", "`indiv_flow` 절대값 최댓값 / 절대값>1 행", f"{f0(max(abs(f1['최솟값']), f1['최댓값']))} / {f0(f1['|x|>1'])}", "1.000 / 0", "해소"),
        ("B6", "`_w` 열의 미래 표본 경계", "3열", "삭제, 날짜별 순위 열 3개", "해소"),
        ("B7", "상한가 재현율 / 정밀도 (정확 판정 기준)", f"{pc(b7a['상한가']['재현율'])} / {pc(b7a['상한가']['정밀도'])}", f"{pc(b7b['상한가']['재현율'])} / {pc(b7b['상한가']['정밀도'])}", "해소"),
        ("B7", "하한가 재현율 / 정밀도", f"{pc(b7a['하한가']['재현율'])} / {pc(b7a['하한가']['정밀도'])}", f"{pc(b7b['하한가']['재현율'])} / {pc(b7b['하한가']['정밀도'])}", "해소"),
        ("B7", "재개일 / 정리매매 구간의 상하한가 오표시 (상한+하한 행)", f"{b7_c(R, 'v1 재개일 중 lim 표시')} / {b7_c(R, 'v1 정리매매 구간 lim 표시')}", "0 / 0", "해소"),
    ]
    w(tbl(pd.DataFrame(t, columns=["ID", "지표", "v1", "v2", "판정"]), idx=False))
    w("\n- B7 의 v2 100% 는 판정 산식을 q01 의 정확 판정과 같게 했기 때문이다. 순환이 되지 않도록 호가표는 p02 가 CSV 에서, q01 은 손으로 옮긴 값을 쓰고 산식도 정수 대 부동소수로 다르게 구현돼 있어서 "
      "두 구현을 전 행에서 비교해 한도가 불일치 0건을 확인했다. 산식 자체의 독립 검증은 2.7절의 필요조건 검사로 했다.")
    w("- B2 오차가 감사 패치 검증값(3.0%)보다 낮은 것(1.3%)은 정리매매 구간의 실제 호가를 극저가 규칙으로 지우지 않았기 때문이다(1절 P2).\n")

    # ------------------------------------------------------------ 1
    w("## 1. 패치별 반영 방법\n")
    pt = [
        ("P1 재개일 분리", "`resume_day` = 직전 행이 정지(`is_halted==1`)/거래량0 이고 오늘 거래(`f_tradable`). 원 수익률(정지·거래량0 일은 NaN)에서 재개일 값을 `ret_multi` 로 옮기고 `ret1d` 는 NaN. `ret1d_clean`·`vol20`(= `ret1d_clean` 20행 표준편차)이 재개일 누적 수익을 쓰지 않는다.",
         "`ret_outlier` 는 재개일에 한해 `ret_multi` 로 판정한다(그렇지 않으면 가격 재설정이 일어난 재개일이 `str5/str20/mom12_1` 창 마스크에서 빠진다). 감사 패치 코드는 재개일 수익이 NaN 이라 이 보호가 사라진다."),
        ("P2 정리매매·상폐 손실", "`liq_zone` = 상폐 종목(마지막 행이 자료 끝보다 10일 넘게 앞섬)의 마지막 8행. 이상값(|수익| > 한도 + 10%p)에서 이 구간을 제외 → `ret1d_clean` 에 실제 손실이 남는다. 수정주가 <= 100(전일 포함)은 `f_penny` 로 표시하고 `ret_outlier` 에 넣어 비운다. 라벨 `fwd_ret5` / `fwd_censored` 추가(상폐 종목은 마지막 가격까지).",
         f"정리매매 구간의 실제 호가(수정주가 == 종가, 전일도 동일)는 극저가여도 비우지 않는다. 감사 패치를 그대로 쓰면 정리매매 거래일 {b2l['그중 거래일(f_tradable) v2']}행 중 {liq_pen[0]}행(종목 {liq_pen[1]}개, 그중 exclude 종목 제외 {b2l['그중 거래일 극저가 행(exclude 뺀)']}행/{b2l['그중 극저가 종목(exclude 뺀)']}종목)의 실제 손실이 지워진다. 이 구간은 adj_close == close 라 반올림 잡음이 없다(000790 6행만 예외, exclude 종목)."),
        ("P3 구호가표", "`tick_of()` 가 `tick_size.csv`(2010-10-04~) 와 구호가표 6구간(`OLD_TICK`, market_rules_notes F12, 2차 자료)을 시대별로 묶어 계산. `tick_assumed = (date < 2010-10-04)`. 대체값 1.0 은 남아 있되 대체가 한 행이라도 생기면 `assert` 로 멈춘다.",
         "p01 의 CSV 루프를 같은 함수로 대체했다(2010-10-04 이후 값은 v1 과 같음)."),
        ("P4 K-OTC 삭제", f"투자자 자료가 시작하기 전 선두 구간이 20행 이상인 종목(5종목)의 그 구간 {f0(b4['삭제 목록 행'])}행을 삭제하고 `panel_v2_removed_rows.parquet`(ticker, date, OHLC, adj_close, volume, reason)에 남겼다. 삭제 집합이 q01 의 `f_prelist_otc` 와 같음을 확인(2.4절).",
         "플래그가 아니라 삭제가 기본이다. `f_prelist_otc` 열은 만들지 않는다(전부 0 이 될 것)."),
        ("P5 투자자 열", "(a) 4그룹 순매수가 모두 0 이고 거래량 >= 1만 주인 행, (b) 순매수 절대값이 거래량 x 1.0001 초과인 행 → 투자자 8열 전부 NaN (`f_zero_inv_suspect`, `f_net_gt_volume` 로 흔적 보존). (c) `foreign_def_changed` = 2026-05-26 이후, (d) `inv_sum_closed` = 2012-05-03~2026-05-25.",
         "감사 패치처럼 2026-05-26 이후 foreign_net 을 NaN 으로 만들지 않고 표시만 한다(요청 사항). 순매수 열만이 아니라 buy/sell 열도 함께 NaN 처리해 항등식이 어긋나지 않게 했다."),
        ("P6 flow 정규화", "분할 보정 계수 `f = adj/close`: `va = volume/f`(정지일 제외) → 직전 60행(당일 제외, 최소 30) 평균 x `f` = 분모, `flow = net / max(분모, 당일 거래량)` ∈ [-1, 1]. `_w` 3열은 만들지 않고 `indiv_rk / inst_rk / foreign_rk` = 날짜별 `rank(pct=True) - 0.5`(f_tradable 이고 flow 비결측).",
         "정지·거래량0 일의 flow 는 0 이 아니라 NaN(순위가 0 값에 끌리지 않게). `indiv_flow_tv` 도 같은 방식으로 [-1, 1] 에 묶었다(감사 P6 에는 없음)."),
        ("P7 상하한가", "`exact_limits()` = q01 산식(기준가 x 한도율을 기준가 호가단위로 절사 → 가감 → 결과 가격대 호가단위로 절사/절상)을 정수 연산으로. `lim_up = (close >= 상한가) & lim_elig`, `lim_dn = (close <= 하한가) & lim_elig`. `lim_elig` = 직전 행 정상 거래 & 당일 거래 & 조정계수 단차 < 1% & 정리매매 구간 아님.",
         "한계: 기준가를 전일 종가로 본다(배당락 등 1% 미만 조정, 호가 구간 경계를 넘는 날의 절사는 틀릴 수 있음, 5절)."),
        ("지침 1·9", "`exclude_flag`/`exclude_reason`(비트마스크): 자료 끝 3거래일(2026-10-06~08), 008080·000790 전 기간, 006740 의 2026-06-26~. `mktcap_missing`(표시만). `f_tradable` 열 포함.",
         "'008080·000790·006740 후반 구간' 은 감사 지침 1번의 표현대로 해석했다: 008080·000790 은 전 기간(상장폐지 전까지 줄곧 1~4원/10~115원 아티팩트), 006740 은 2026-06-26 이후."),
    ]
    w(tbl(pd.DataFrame(pt, columns=["패치", "구현", "감사 패치 대비 차이·메모"]), idx=False))

    # ------------------------------------------------------------ 2 (검증)
    w("\n## 2. 검증 (q02_panel_v2_qa.py)\n")
    w("v2 파일을 다시 읽어 q01 의 정의를 독립적으로 재계산했다(행 슬라이스·다른 구현). 먼저 q01 의 행 플래그 파일 `q01_panel_flags.parquet` 과의 교차 결과다. 전부 0건 불일치다(`liq_zone` 이 q01 의 `f_liquidation_zone` 과 1,264행 모두 같다는 뜻이다).\n")
    w(tbl(pd.DataFrame(list(R["flag_cross"].items()), columns=["비교", "불일치(행)"]), idx=False))
    w("\n### 2.1 B1·B9 거래재개일\n")
    w(tbl(pd.DataFrame([
        ("재개일 행 (독립 재계산 / q01 f_resume_day)", f0(b1["v1 재개일 행"]), f"{f0(b1['재개일 행(독립 재계산)'])} / {f0(R['flag_totals']['f_resume_day'])}"),
        ("재개일에 ret1d 값이 있는 행", f0(b1["v1 재개일 ret1d 값 있음"]), str(b1["재개일 ret1d 값 있음"])),
        ("재개일에 ret1d_clean 값이 있는 행", f0(b1["v1 재개일 ret1d_clean 값 있음"]), str(b1["재개일 ret1d_clean 값 있음"])),
        ("재개일 |ret1d_clean| 평균 (정상일 평균 %s)" % f"{b1['v2 정상일 |ret1d_clean| 평균']:.4f}", f"{b1['v1 재개일 |ret1d_clean| 평균']:.4f}", "NaN"),
        ("재개일 |수익| > 10% 행 (v1 ret1d)", f0(b1["v1 재개일 ret1d 5%p 초과 행"]), "ret_multi 로 이동"),
        ("ret_multi 가 값을 갖는 행 (재개일 / 재개일 아님)", "-", f"{f0(b1['재개일 ret_multi 값 있음'])} / {b1['재개일 아닌데 ret_multi 값 있음']}"),
        ("vol20 최댓값 / vol20 > 0.15 행", f"{b9['v1 vol20 최대']:.3f} / {f0(b9['v1 vol20>0.15 행'])}", f"{b9['v2 vol20 최대']:.3f} / {f0(b9['v2 vol20>0.15 행'])}"),
        ("vol20 표본 3,000행 직접 계산(20행 std) 불일치", "-", str(b9["표본 3000행 vol20 직접계산 불일치"])),
    ], columns=["지표", "v1", "v2"]), idx=False))
    w("\nvol20 창에 재개일이 들어 있는 행은 v2 에도 %s행 있지만 그 날 값이 NaN 이라 누적 수익은 쓰이지 않는다.\n" % f0(b9["v2 vol20 창에 재개일 포함(그 날 값은 NaN) 행"]))

    w("### 2.2 B2 정리매매·상폐 직전 손실\n")
    w("지표는 q01 s9 와 같다: 상폐 종목의 마지막 9행에서 `수정주가 마지막/처음 - 1`(실제)과 `(1 + ret1d_clean)` 복리(v2 는 재개일에 `ret_multi` 를 대입)의 평균 절대오차.\n")
    w(tbl(pd.DataFrame([
        ("19종목 (q01 패치검증과 같은 무작위 표본)", f0(b2s["종목수"]), pc(b2s["실제 평균"]), pc(b2s["v1 clean 복리 평균"]), pc(b2s["v2 clean 복리 평균"]), pc(b2s["v1 평균오차"]), pc(b2s["v2 평균오차"])),
        ("상폐 전 종목 (|실제|<500%)", f0(b2a["종목수"]), pc(b2a["실제 평균"]), pc(b2a["v1 clean 복리 평균"]), pc(b2a["v2 clean 복리 평균(재개일은 ret_multi)"]), pc(b2a["v1 평균오차"]), pc(b2a["v2 평균오차"], 2)),
    ], columns=["표본", "종목", "실제 평균", "v1 clean 복리", "v2 clean 복리", "v1 평균오차", "v2 평균오차"]), idx=False))
    w(f"\n- `ret1d_clean` 복리에 재개일 `ret_multi` 를 더하지 않으면 v2 오차는 {pc(b2a['v2 평균오차(보충 없음)'])} 로 올라간다. 정리매매는 거래정지가 풀리며 시작하는 경우가 많아 정리매매 첫날이 재개일이 되고, 그 날의 큰 손실이 `ret_multi` 로 옮겨갔기 때문이다"
      f"(정리매매 구간의 재개일 `ret_multi` {b2l['그중 ret_multi 값 있음(= 정리매매 첫날 재개일) / 평균'][0]}행, exclude 종목을 뺀 평균 {pc(b2l['그중 ret_multi 값 있음(= 정리매매 첫날 재개일) / 평균'][1])}). 누적 수익·라벨은 `adj_close` 로 직접 계산하거나 `fwd_ret5` 를 쓴다.")
    w(f"- 정리매매 구간 {f0(b2l['정리매매 구간 행'][1])}행 중 거래일 {b2l['그중 거래일(f_tradable) v2']}행. `ret1d_clean` 에 값이 있는 행 v1 {b2l['그중 ret1d_clean 값 있음 v1 / v2'][0]} → v2 {b2l['그중 ret1d_clean 값 있음 v1 / v2'][1]}, `ret_outlier=1` 인 행 {b2l['그중 ret_outlier=1 v1 / v2'][0]} → {b2l['그중 ret_outlier=1 v1 / v2'][1]}. "
      f"정리매매 구간 `ret1d_clean` 평균 {pc(b2l['정리매매 구간 ret1d_clean 평균 v1 / v2'][0], 2)} → {pc(b2l['정리매매 구간 ret1d_clean 평균 v1 / v2'][1], 2)}, 최솟값 {pc(b2l['정리매매 구간 ret1d_clean 최솟값 v1 / v2'][0])} → {pc(b2l['정리매매 구간 ret1d_clean 최솟값 v1 / v2'][1])}.")
    w(f"- `ret_outlier`: v1 {ol['v1 ret_outlier']}행 → v2 {f0(ol['v2 ret_outlier'])}행. 늘어난 것은 극저가 규칙({f0(ol['f_penny 이면서 ret_outlier=1'])}행; 상위 {dstr(ol['극저가로 비운 상위 종목'])})이고 그중 exclude_flag=1 이 {f0(ol['극저가로 비운 행 중 exclude_flag=1'])}행(008080·000790)이다. "
      f"극저가가 아닌 이상값은 {ol['극저가 아닌 이상값']}행({dstr(ol['극저가 아닌 이상값 종목'])}; 감사의 재개일 가격 재설정 의심 사례). 정리매매 구간이라 비우지 않은 극저가 행은 {ol['정리매매 실제 호가라 비우지 않은 극저가 행']}행.\n")
    w("**라벨.** `fwd_ret5` / `fwd_censored`:\n")
    w(tbl(pd.DataFrame([
        ("fwd_censored 0 (완전 5거래일)", f0(fw["fwd_censored 분포"][0]), f"직접 계산 `adj.shift(-5)/adj-1` 과 불일치 {fw['완전창 행 직접계산 불일치']}행"),
        ("fwd_censored 1 (상폐로 잘림)", f0(fw["fwd_censored 분포"][1]), f"값 있음 {fw['상폐로 잘린 행(코드1) 중 값 있음']}행(상폐 직전 4행 x 158), 평균 {pc(fw['그 평균'])}, -30% 이하 {fw['그 중 -30% 이하 행']}행; 마지막 행 158개는 NaN"),
        ("fwd_censored 2 (자료 끝)", f0(fw["fwd_censored 분포"][2]), f"값 있음 {fw['자료끝 행(코드2) 값 있음']} (804종목 x 5행)"),
        ("v1 식 라벨(5행 뒤가 없으면 NaN) 유효 행 → v2", f"{f0(fw['v1식 라벨 유효 행'])} → {f0(fw['v2 라벨 유효 행'])}", f"추가 {f0(fw['추가된 행'])}행"),
        ("정리매매 구간 fwd_ret5 평균 (exclude 뺀) v1 식 / v2", f"{pc(fw['정리매매 구간 행(제외 표시 뺀) fwd_ret5 평균 v1식 / v2'][0])} / {pc(fw['정리매매 구간 행(제외 표시 뺀) fwd_ret5 평균 v1식 / v2'][1])}", f"유효 행 {pair(fw['정리매매 구간 행(제외 표시 뺀) 유효 행 v1식 / v2'], ' → ')}"),
        ("전체 fwd_ret5 평균 (exclude 뺀) v1 식 / v2", f"{pc(fw['전체(제외 표시 뺀) fwd_ret5 평균 v1식 / v2'][0], 3)} / {pc(fw['전체(제외 표시 뺀) fwd_ret5 평균 v1식 / v2'][1], 3)}", "전체 평균에는 거의 영향이 없고 상폐 직전 구간에만 쓸모가 있다"),
    ], columns=["항목", "행", "비고"]), idx=False))

    w("\n### 2.3 B3 호가단위\n")
    w(tbl(b3.reset_index().rename(columns={"era": "시대"}), idx=False))
    w(f"\n- 비교 기준은 q01 의 손으로 옮긴 호가표(`tick_vec`). `tick_pct` 중앙값(2010-10-04 이전): v1 {b3m['v1 tick_pct 중앙값(%)']:.4f}% → v2 {b3m['v2']:.4f}% (2010-10-04~12-31 은 {b3m['2010-10-04~2010-12-31 v2']:.4f}%). `tick_assumed=1` {f0(b3m['tick_assumed=1 행'])}행, 정의(date < 2010-10-04) 일치 {b3m['tick_assumed 정의 일치']}.\n")

    w("### 2.4 B4 상장 전 K-OTC\n")
    w(f"- 감사 K-OTC 행 {f0(b4['감사 K-OTC 행(플래그)'])}행 중 v2 에 남은 행 {b4['v2 에 남은 그 행']}. 삭제 목록과 q01 `f_prelist_otc` 집합이 같음: {b4['삭제 목록 == 플래그 행 집합']}. v1 {f0(b4['v1 행 수'])}행 − 삭제 {f0(b4['삭제 목록 행'])} = v2 {f0(b4['v2 행 수'])}행.")
    w(f"- 종목과 삭제 후 첫 날짜: {dstr(b4['삭제 후 첫 날짜'])} (각 종목의 투자자 자료 시작일 = 상장일). 삭제 후 해당 종목에서 투자자 자료가 NaN 인 선두 행 {b4['삭제 후 해당 종목 투자자 자료 NaN 인 선두 행']}행. v1 의 K-OTC 행 중 ret_outlier=1 이던 {b4['v1 K-OTC 행 ret_outlier']}행도 함께 사라졌다.\n")

    w("### 2.5 B5 투자자 열\n")
    w(tbl(pd.DataFrame([
        ("4그룹 모두 0 & 거래량 >= 1만 행", f0(b5["4그룹 모두 0 & 거래량≥1만 v1 / v2"][0]), str(b5["4그룹 모두 0 & 거래량≥1만 v1 / v2"][1])),
        ("순매수 절대값 > 거래량 x 1.0001 행", str(b5["순매수>거래량 v1 / v2"][0]), str(b5["순매수>거래량 v1 / v2"][1])),
        ("inv_sum_closed=1 행 수 / 0 인 행 수", "-", f"{f0(b5['inv_sum_closed 행 / 0인 행'][0])} / {f0(b5['inv_sum_closed 행 / 0인 행'][1])}"),
        ("inv_sum_closed=1 구간에서 4그룹 합이 계산되는 행 중 합 ≠ 0", "-", f"{f0(b5['inv_sum_closed=1 구간 4그룹 합 계산 가능 행'])}행 중 {b5['그중 합≠0 행']}"),
        ("2012-05-02 이전 4그룹 합 ≠ 0 비율 (NaN 처리 후)", "-", pc(b5["inv_sum_closed=0 & 2012-05-02 이전 합≠0 비율"])),
        ("2026-05-26~08-06 4그룹 합 ≠ 0 비율", "-", pc(b5["2026-05-26~08-06 4그룹 합≠0 비율"])),
        ("foreign_def_changed=1 행 / 첫 날짜", "-", f"{f0(b5['foreign_def_changed=1 행 / 첫 날짜'][0])}행 / {b5['foreign_def_changed=1 행 / 첫 날짜'][1]}"),
    ], columns=["지표", "v1", "v2"]), idx=False))
    w(f"\n- NaN 처리한 행은 투자자 8열이 모두 NaN: {b5['NaN 처리 행의 투자자 8열이 모두 NaN']}. 순매수>거래량 NaN 처리 날짜: {', '.join(f'{str(k.date())} {v}행' for k, v in b5['순매수>거래량 NaN 처리 날짜'].items())}.\n")

    w("### 2.6 B6 flow 정규화\n")
    d6 = b6.copy()
    d6["최솟값"] = d6["최솟값"].map(lambda v: f"{v:,.3f}")
    d6["최댓값"] = d6["최댓값"].map(lambda v: f"{v:,.3f}")
    d6["표준편차"] = d6["표준편차"].map(lambda v: f"{v:.4f}")
    d6["첨도"] = d6["첨도"].map(lambda v: f"{v:,.2f}")
    for c in ("|x|>1", "|x|>5", "비결측 행"):
        d6[c] = d6[c].map(f0)
    w(tbl(d6, idx=False))
    w(f"\n- 분모 직접 계산: 무작위 {f0(R['b6_den'][0])}행을 행 슬라이스로 (정지일 제외 60행 평균, 분할 보정, max(·, 당일 거래량)) 다시 계산해 {f0(R['b6_den'][1])}행 일치. 정지·거래량0 일에 값이 있는 `indiv_flow` 행: v1 {f0(R['b6_idle_flow']['v1 정지/거래량0 일 flow==0 행'])}(전부 0 값) → v2 {R['b6_idle_flow']['v2 정지/거래량0 일 flow 값 있음']}. `_w` 열: {'없음' if not R['b6_w_absent'] else R['b6_w_absent']}.")
    w("- 순위 열 (`rank(pct=True) - 0.5`, 대상 f_tradable=1 & flow 비결측):\n")
    rk2 = rk.copy()
    rk2["비결측 행"] = rk["비결측 행"].map(f0)
    for c in ("최솟값", "최댓값"):
        rk2[c] = rk[c].map(lambda v: f"{v:.4f}")
    rk2["날짜별 평균 절대편차 평균"] = rk["날짜별 평균 절대편차 평균"].map(lambda v: f"{v:.5f}")
    rk2["정지·거래량0 행 중 값 있음"] = rk["정지·거래량0 행 중 값 있음"].map(f0)
    rk2["날짜별 유효 종목수 최솟값"] = rk["날짜별 유효 종목수 최솟값"].map(f0)
    w(tbl(rk2.rename_axis("열")))
    w("\n  유효 종목수 최솟값 0 은 2010-01 의 첫 30거래일(분모의 최소 30행 조건) 때문이다.\n")

    w("### 2.7 B7 상·하한가\n")
    rows = []
    for nm in ("상한가", "하한가"):
        for ver, d in (("v1", b7a[nm]), ("v2", b7b[nm])):
            rows.append((nm, ver, f0(d["기준(exact)"]), f0(d["패널 표시"]), f0(d["둘 다"]), f0(d["누락"]), f0(d["과검출"]), pc(d["재현율"]), pc(d["정밀도"])))
    w("정확 판정 기준 = `q01_panel_qa.exact_limits`(손으로 옮긴 호가표·부동소수) x 대상 행(직전 행 정상 거래, 조정계수 단차 < 1%, 당일 거래, 정리매매 구간 제외). 정밀도 = 표시된 행 전체 중 기준과 일치하는 비율. "
      "v1 수치는 q01 이 정리매매 구간을 포함해 낸 값(재현율 94.5%/83.7%, 정밀도 97.2%/84.3%)과 약간 다르다.\n")
    w(tbl(pd.DataFrame(rows, columns=["구분", "버전", "기준(exact)", "패널 표시", "둘 다", "누락", "과검출", "재현율", "정밀도"]), idx=False))
    w("\n산식을 독립 확인하는 필요조건 검사(v2):\n")
    nec = [("p02 한도가 vs q01 한도가 불일치 (상한 / 하한, 전 행)", pair(b7b["p02 한도가 vs q01 한도가 불일치(전 행)"])),
           ("대상 행 중 종가 > 상한가 / 종가 < 하한가: 전체", pair(b7b["종가>상한가 / 종가<하한가 (대상 행) 전체"])),
           ("   exclude_flag=1 행을 뺀", pair(b7b["종가>상한가 / 종가<하한가 (대상 행) 제외 표시 뺀"])),
           ("대상 행 중 고가 > 상한가 / 저가 < 하한가 (exclude 뺀)", pair(b7b["고가>상한가 / 저가<하한가 (대상 행) 제외 표시 뺀"])),
           ("상한가 표시 행 중 고가==종가 / 하한가 표시 행 중 저가==종가 (exclude 뺀)", "%s / %s" % tuple(pc(v, 2) for v in b7b["상한가 행 중 고가==종가 / 하한가 행 중 저가==종가 (제외 표시 뺀)"])),
           ("   전체 (000790 의 10~25원 호가 때문에 낮아짐)", "%s / %s" % tuple(pc(v, 2) for v in b7b["상한가 행 중 고가==종가 / 하한가 행 중 저가==종가 (전체)"])),
           ("lim_elig 재계산 불일치", str(b7b["lim_elig 재계산 일치"]))]
    w(tbl(pd.DataFrame(nec, columns=["검사", "결과"]), idx=False))
    c7 = R["b7_counts"]
    w(f"\n- 상한가 / 하한가 표시 수: v1 {c7['v1 상한/하한'][0]:,} / {c7['v1 상한/하한'][1]:,} → v2 {c7['v2 상한/하한'][0]:,} / {c7['v2 상한/하한'][1]:,}. 재개일의 표시(상한 / 하한) v1 {pair(c7['v1 재개일 중 lim 표시'])} → v2 {c7['v2 재개일 중 lim 표시']}, 정리매매 구간 v1 {pair(c7['v1 정리매매 구간 lim 표시'])} → v2 {c7['v2 정리매매 구간 lim 표시']}. 판정 불가(`lim_elig=0`) {f0(c7['v2 lim_elig=0 행'])}행.")
    w("- 연도별 표시 수(2010 은 v1 이 5원 호가 때문에 상하한가를 놓쳤던 해라 가장 많이 바뀐다):\n")
    yr = R["b7_year"].copy()
    yr.index = yr.index.astype(str)
    w(tbl(yr.rename_axis("연도")))

    w("\n### 2.8 제외 표시\n")
    w(f"- `exclude_flag=1`: {f0(ex['exclude_flag=1 행'])}행. 비트별(겹침 포함): 자료 끝 3거래일 {ex['마지막 3거래일']} {f0(ex['비트1 자료 끝 3거래일'])}행, 008080·000790 {f0(ex['비트2 008080·000790'])}행, 006740(2026-06-26~) {ex['비트4 006740 2026-06-26~']}행. "
      f"`mktcap_missing` {f0(ex['mktcap_missing 행'])}행(첫 날짜 {ex['mktcap_missing 첫 날짜(2026)']}). `f_tradable=1` 비율 {pc(ex['f_tradable=1 비율'], 2)}.\n")

    # ------------------------------------------------------------ 3 룩어헤드
    w("## 3. 룩어헤드 실험\n")
    w(f"q01 s7 과 같은 방식이다. 무작위 110종목 + 특수 10종목(q01 과 같은 표본, seed 20261010) = {R['lh_rows'][0]}종목 {f0(R['lh_rows'][1])}행의 원자료를 `build_panel()` 에 그대로 넣어 v2 로직을 재실행하고, "
      f"기준일 {R['lh_info']['기준일']} 이전 값을 비교했다. 비교 대상은 입력 열을 뺀 파생·플래그·순위 열이다(특징 열 {n_feat}개 + 라벨 2 + 자료끝 표시 2).\n")
    exp = [
        ("0. 재현: 같은 입력으로 다시 만든 값 vs 저장된 panel_v2 (순위 열 제외: 부분집합이라 모집단이 다름)", f0(lh_rep[0]), f"{len(feat_nr)}", str(mis_rep), "코드가 저장 파일을 정확히 만든다"),
        ("1. 절단: 입력을 기준일까지만 주고 다시 계산", f0(lh_tr[0]), f"{n_feat}", str(mis_tr), "미래 행이 없어도 과거 값 동일"),
        ("2. 교란: 기준일 이후 가격·거래량·상장주식수·투자자·거래대금·정지 플래그를 무작위로 바꿈", f0(lh_pt[0]), f"{n_feat}", str(mis_pt), "미래를 바꿔도 과거 값 동일(순위 열 포함)"),
        ("3. 한 종목(005930)만 교란, 다른 117종목 비교 (순위 열은 기준일 이전만)", f"{f0(lh_one[0])} / {f0(lh_one[2])}", f"{len(feat_nr)} + 3", str(mis_one), "종목 간 누수 없음"),
    ]
    w(tbl(pd.DataFrame(exp, columns=["실험", "비교 행", "비교한 특징 열", "불일치 합계", "결과"]), idx=False))
    w("\n**불일치가 0 이 아닌 열은 의도적으로 사후 정보를 쓰는 열뿐이다.**\n")
    d_tr, d_pt = lh_tr[1], lh_pt[1]
    w(tbl(pd.DataFrame([
        ("fwd_ret5 / fwd_censored", "라벨(5거래일 뒤 수익, 상폐 여부)", f"{d_tr['fwd_ret5']} / {d_tr['fwd_censored']}", f"{d_pt['fwd_ret5']} / {d_pt['fwd_censored']}", "기준일 전 5거래일 행이 미래 가격을 쓴다 → 의도된 동작. fwd_censored 는 구조(행 존재)만 보므로 값 교란에는 불변"),
        ("exclude_flag / exclude_reason", "자료 끝 3거래일 표시", f"{d_tr['exclude_flag']} / {d_tr['exclude_reason']}", f"{d_pt['exclude_flag']} / {d_pt['exclude_reason']}", "절단하면 '자료 끝'이 기준일로 옮겨가 표시가 달라진다(달력 기반 표시). 값 교란에는 불변"),
    ], columns=["열", "성격", "절단 불일치", "교란 불일치", "해석"]), idx=False))
    w("\n- `liq_zone`·`ret_outlier`·`f_penny` 는 절단·교란 모두 불일치 0 이다. 다만 `liq_zone` 은 '이 종목이 이후 상장폐지된다'는 사후 정보에서 정해지는 표시다(정리매매 일정은 보통 사전 공시되지만 이 자료로는 공시일을 확인하지 못함). "
      "실시간 신호 입력으로 쓰지 말고 라벨·평가·결측 처리용으로만 쓴다.")
    ap = lh_pt[2]
    keys = ["ret1d", "ret1d_clean", "indiv_flow", "indiv_rk", "str5", "mom12_1", "vol20", "lim_up", "tick", "resume_day", "f_tradable", "fwd_ret5"]
    w(f"- 교란이 실제로 작동했는지(양성 대조): 기준일 이후 {f0(lh_pt[3])}행에서 값이 바뀐 행 수 — " + ", ".join(f"{k} {f0(ap[k])}" for k in keys) + ". "
      f"일부러 미래를 쓰는 대조 열(5일 선행수익, 중심 이동평균)은 같은 방법으로 {R['lh_ctrl']['ctrl_fwd5']}·{R['lh_ctrl']['ctrl_center']}행 불일치로 잡힌다(검정력 확인).")
    w(f"- 순위 열: 같은 날 횡단면 계산이라 기준일 이전 값은 교란에 불변(실험 2, 불일치 0). 순위 열을 `f_tradable` 행의 `indiv_flow` 에서 날짜별로 다시 계산해 저장 값과 비교하면 불일치 {R['lh_rank_manual']}행. "
      f"한 종목만 교란하면 다른 종목의 순위는 기준일 이후 날짜에서만 바뀐다({', '.join(f'{k} {f0(v)}' for k, v in lh_one[4].items())}행; 기준일 이전 0행). 같은 날 다른 종목에 의존하는 횡단면 순위의 정상 동작이다.")
    w("- 한계: 수정주가 `adj_close` 의 *수준*은 DataGuide 스냅샷 이후의 분할·배당으로 과거가 바뀐 사후 조정 시계열이다. 비율을 쓰는 열(ret, flow 분모의 `adj/close` 비 `f_t/f_s`, str, mom, near52h)은 영향이 없지만 "
      "`f_penny` 는 수정주가 *수준*(<=100)을 쓰므로 엄밀하게는 미래 조정에 영향을 받는다. 신호로 쓰는 열이 아니라 반올림 잡음 구간을 '비우는' 용도라 두었다(거래 가능성 필터는 `close` 를 쓸 것).\n")

    # ------------------------------------------------------------ 4 v1 대비
    w("## 4. v1 대비 변화 (행·열 수준)\n")
    w(f"- 행: {f0(sh1[0])} → {f0(sh2[0])} (−{f0(sh1[0] - sh2[0])}, 상장 전 K-OTC 5종목). 종목 {R['tickers'][0]} → {R['tickers'][1]}. 정렬(ticker, date) {R['sorted_ok']}, 중복 {R['dups']}건.")
    w(f"- 열: {sh1[1]} → {sh2[1]} (+{len(R['cols_new'])} 새 열, −{len(R['cols_dropped'])} 삭제 열 {R['cols_dropped']}).")
    w(f"- 아래는 두 패널에 모두 있는 {f0(R['chg_n'])}행을 (ticker, date) 로 맞춰 열별로 값이 바뀐 행의 비율이다(NaN 끼리는 같음, 허용오차 1e-9). 바뀌지 않은 열은 생략했다: {DICT_SAME}.\n")
    ch = R["chg"][R["chg"]["바뀐 행"] > 0].copy()
    ch["비율"] = ch["비율"].map(lambda v: f"{v * 100:.4f}%")
    for c in ("바뀐 행", "NaN→값", "값→NaN", "값→다른 값"):
        ch[c] = ch[c].map(f0)
    w(tbl(ch))
    C = R["chg"]
    w("\n읽는 법: `*_flow` 가 약 32~49% 바뀐 것은 분모를 max(평균, 당일 거래량) 로 바꿨기 때문이다(당일 거래량이 평균보다 큰 날 전부). "
      f"`ret1d` 의 값→NaN {f0(C.loc['ret1d', '값→NaN'])}행 = 재개일 2,847 + K-OTC 삭제 후 5종목의 새 첫 행 5. "
      f"`ret1d_clean` 은 NaN→값 {f0(C.loc['ret1d_clean', 'NaN→값'])}행(풀린 정리매매 이상값)이고 값→NaN {f0(C.loc['ret1d_clean', '값→NaN'])}행은 재개일 2,784행과 극저가 규칙이다. "
      f"`ret_outlier` 는 0→1 {f0(ol['v1→v2 ret_outlier 0→1 / 1→0'][0])}행(그중 exclude 종목 밖 {ol['0→1 중 exclude_flag=0']}행; 대부분 008080·000790 극저가), 1→0 {ol['v1→v2 ret_outlier 0→1 / 1→0'][1]}행(주로 풀린 정리매매 구간 이상값)이다. "
      "`lim_up/lim_dn` 은 0.02% 안팎이 바뀌고(2010 년에 집중), `tick` 135,887행은 모두 2010-01~10 이다. `near52h`·`abvol`·`mom12_1`·`str5` 의 변화는 K-OTC 삭제로 5개 종목의 롤링 창 시작점이 옮겨진 행과 극저가 규칙 때문이다. "
      "투자자 열의 값→NaN 은 f_zero_inv_suspect 18,275 + f_net_gt_volume 172 이다.\n")
    w("`r01` 핵심 통계(대표 신호 pred_d_lr1e-4 의 상위 10분위 편입 수·CAR) 변화는 이 문서의 범위가 아니다(별도 에이전트).\n")

    # ------------------------------------------------------------ 5 한계
    w("## 5. 남은 한계 (해소하지 못한 것)\n")
    lim = [
        "**외국인 정의 변경(B5)**: 2026-05-26 이후 `foreign_net` 이 기타외국인을 제외한 값으로 추정된다. 표시(`foreign_def_changed`)만 했고 복원하려면 kiwoom 수급의 `foreign_etc` 가 필요하다. 2026-08-07 이후는 `other_corp_net`·buy/sell 이 전부 NaN 이라 4그룹 합을 확인할 수 없다.",
        "**2010-01~2012-05-02 순매수**: 4그룹 합이 닫히지 않는다(원천 분류 차이로 추정, 원천 문서 미확인). `inv_sum_closed=0` 으로 표시만 했고 값은 그대로다. 이 구간 순위 열은 정의가 불명확한 순매수의 순위다.",
        "**2010-10-04 이전 호가표·상하한가**: 구호가표는 2차 자료(market_rules_notes F12)이고 독립 확인을 못했다(`tick_assumed=1`). 2010 년 상하한가 표시도 이 표에 의존한다.",
        "**정확 판정의 한계(B7)**: 기준가를 전일 종가로 본다. 배당락 등 1% 미만 기준가 조정은 반영하지 못하고, 호가 구간 경계를 넘는 날(예: 035720 2014-05-28 종가 103,200 > 산식 상한가 103,000)은 상한가를 100원 낮게 계산한 것으로 보인다(두 번째 절사가 구간 경계를 넘는 날에는 적용되지 않는 것으로 추정하나 확인하지 못함). "
        f"종가가 산식 한도가를 벗어난 대상 행은 종가 > 상한가 / 종가 < 하한가 {pair(b7b['종가>상한가 / 종가<하한가 (대상 행) 전체'])}행(exclude 뺀 {pair(b7b['종가>상한가 / 종가<하한가 (대상 행) 제외 표시 뺀'])}행), 고가 > 상한가 / 저가 < 하한가는 exclude 뺀 {pair(b7b['고가>상한가 / 저가<하한가 (대상 행) 제외 표시 뺀'])}행으로 극소수다.",
        f"**극저가 규칙(P2)**: 수정주가 *수준*(<=100)을 쓰는 사후 조정 기준이다. 비운 행은 008080·000790 이 거의 전부({dstr(ol['극저가로 비운 상위 종목'])}). 008080 의 정리매매 구간에는 가격 재설정(1원 → 67,000원)이 남아 `str5` 에 극단값(최대 43,849)이 있다. exclude_flag 로 가려지는 종목이다.",
        "**사후 정보 열**: `liq_zone`, `fwd_ret5`, `fwd_censored` 는 상폐 여부·미래 가격을 쓴다. `exclude_flag` 는 자료 끝 기준이라 자료를 갱신하면 다시 만들어야 한다.",
        "**ret1d_clean 복리**: 재개일 수익이 `ret1d_clean` 에서 빠지므로 `(1+ret1d_clean)` 복리 누적 수익은 정지 중 가격 변화를 놓친다(상폐 종단 오차 %s). 누적 수익은 `adj_close` 로 직접 계산하거나 재개일에 `ret_multi` 를 더한다." % pc(b2a["v2 평균오차(보충 없음)"]),
        "**공매도(B8)·006740 단위(B11)·코스닥 이전 종목·OHLC 위반 17행(B13)**: 손대지 않았다. 부분허용 구간(2021-05-03~2023-11-03)의 `short_banned` 는 종목 단위가 아니고 2013-11-13 이전 금융주 금지는 빠져 있다. 006740 은 `exclude_flag` 로 가린다.",
        "**mktcap**: 2026-06-23 이후 NaN(`mktcap_missing`)은 채우지 않았다. financials.daily_trading 으로 보충해야 한다.",
        "**순위 열**: 날짜별 유효 종목에서만 순위를 매긴다. 첫 30거래일은 flow 자체가 NaN. 순위는 `f_tradable` 행만 대상이고 exclude_flag 행도 모집단에 들어 있다(최신 3일은 투자자 자료가 불확실).",
        "**상폐 판정**: '마지막 행이 자료 끝보다 10일 넘게 앞섬'(158종목, 전부 2026-06-22 이전 종료). 자료 끝 직전에 정리매매 중인 종목은 아직 상폐로 잡히지 않는다.",
    ]
    for i, t_ in enumerate(lim, 1):
        w(f"{i}. {t_}")

    # ------------------------------------------------------------ 6 사용 지침
    w("\n## 6. 사용 지침 (q01 9절 대비 변경점)\n")
    guide = [
        ("1 유니버스", "`f_tradable==1 & f_prelist_otc==0`, 최신 3거래일과 008080·000790·006740 후반 제외", "`f_tradable==1 & exclude_flag==0`. K-OTC 는 삭제돼 `f_prelist_otc` 필터가 필요 없다."),
        ("2 수익·라벨", "특징은 `ret1d_clean`, 라벨은 `adj_close` 로 직접 계산, `f_resume_day==1` 행 제외", "라벨은 `fwd_ret5`(상폐 종목은 마지막 가격까지, `fwd_censored` 로 상태 구분). `ret1d`·`ret1d_clean` 은 재개일이 이미 NaN 이라 별도 제외가 필요 없다."),
        ("3 체결 시점", "t 신호 → t+1 시가 이후, 가격 필터는 `close`", "변경 없음."),
        ("4 순매수", "2010~2012-05 제외, `f_zero_inv_suspect==1` 결측 처리, 2026-05-26 이후 foreign_net 주의", "0 코딩·순매수>거래량 행은 이미 NaN 이다. `inv_sum_closed==1` 이 아니면 4그룹 합을 가정하지 말고, `foreign_def_changed==1` 의 외국인 열은 그 구간 안에서만 비교한다."),
        ("5 flow", "`*_flow` 원본 말고 날짜별 순위", "`*_flow` 가 [-1, 1] 로 묶여 쓸 만하다. 횡단면 모형에는 `*_rk`. `_w` 열은 없다."),
        ("6 호가·세금", "`tick` 은 2010-10-04 이후만", "`tick` 이 전 기간 채워졌다. `tick_assumed==1`(구호가표, 2차 자료)은 가정값으로 취급. `sell_tax_assumed` 는 변경 없음."),
        ("7 상하한가", "근사, 보수적 차단", "정확 판정이다. `lim_elig==0` 이면 판정 불가이므로 거래 가능 여부는 `f_tradable` 과 함께 본다."),
        ("8 공매도", "2021-05~2023-11 부분허용 구간 제외·별도 처리", "변경 없음(미해결)."),
        ("9 vol20·mktcap", "vol20 재계산, mktcap 은 2026-06-23 이후 NaN", "vol20 은 재계산 불필요(`ret1d_clean` 기반). mktcap NaN 은 `mktcap_missing` 으로 표시된다."),
        ("10 재생성", "`q01_panel_qa.py` 를 다시 돌려 B1~B4 확인", "`py -3 code/p02_build_panel_v2.py` 후 `py -3 code/q02_panel_v2_qa.py` 로 이 문서를 다시 만든다."),
    ]
    w(tbl(pd.DataFrame(guide, columns=["지침", "q01 9절 (v1)", "v2"]), idx=False))

    # ------------------------------------------------------------ 7 열 사전
    w("\n## 7. 열 사전 (v1 대비)\n")
    w(f"### 7.1 새 열 ({len(R['cols_new'])}개)\n")
    w(tbl(pd.DataFrame(DICT_NEW, columns=["열", "형", "정의"]), idx=False))
    w("\n### 7.2 삭제 열 (3개)\n")
    w("| 열 | 사유 |\n|---|---|\n| indiv_flow_w, inst_flow_w, foreign_flow_w | 전체 표본(2010~2026) 0.5%/99.5% 분위수 경계라 미래 표본에 의존(약한 룩어헤드). 호환용으로도 두지 않았다 → `*_rk` 와 [-1, 1] 로 묶인 `*_flow` |")
    w("\n### 7.3 의미가 바뀐 열\n")
    w(tbl(pd.DataFrame(DICT_CHG, columns=["열", "변경"]), idx=False))
    w(f"\n### 7.4 값이 같은 열\n\n{DICT_SAME}\n")
    w("열 순서: 입력 열 → p01 파생 열(ret1d ... hl_range) → 새 열(7.1 표 순서). 형: 플래그는 int8(투자자·가격 원열은 v1 과 같음).\n")

    w("## 8. 재현·부록\n")
    w("- 생성: `py -3 code/p02_build_panel_v2.py` (입력 `data/daily/club_db/{price,investor,trading}[_2010_2014].parquet`, `data/제도일정/{tick_size,short_selling_regime,transaction_tax}.csv`). 약 40초, 최대 메모리 약 4.0GB, `panel_v2.parquet` 약 536MB.")
    w("- 검증·이 문서 생성: `py -3 code/q02_panel_v2_qa.py` (약 70초, 최대 메모리 약 5.5GB; `q01_panel_qa.py` 의 호가표·정확 판정·표본 선택을 import 해서 쓴다). `report` 인자를 주면 캐시(`Q02_CACHE`)로 문서만 다시 쓴다.")
    w("- 검증의 한계: 가격·거래량 원천이 같은 DataGuide 라서 가격 정합성은 q01 결론을 그대로 따른다(재검증 안 함). 상하한가 정확 판정의 독립 근거는 2.7절의 필요조건 검사까지다.")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    log("보고서 저장", OUT_MD, f"{len(''.join(L)):,}자")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compute"
    if mode == "compute":
        compute()
        report(R)
    else:
        R.update(pd.read_pickle(CACHE))
        report(R)

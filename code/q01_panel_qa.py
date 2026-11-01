"""q01: club_db/panel.parquet 독립 감사 (p01_build_panel.py 산출물). 기존 파일은 수정하지 않고 새 파일만 쓴다.
실행: py -3 q01_panel_qa.py            (노트북 CPU 약 5~8분)
출력: results/q01_panel_qa.md (보고서), results/q01_panel_flags.parquet (행 단위 QA 플래그, 분석에서 조인용)
구성: setup → s1 커버리지 → s2 가격 → s3 거래정지 → s4 투자자 → s5 거래대금 → s6 제약열 → s7 룩어헤드 → s8 함정 → 보고서 조립.
룩어헤드 실험(s7)은 p01 소스를 그대로 exec 한다(입력만 교체, 저장 줄 앞에서 자름) → p01 이 바뀌어도 시험 대상이 따라간다.
"""
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PAPERS = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS"
D = f"{PAPERS}/data"
CB = f"{D}/daily/club_db"
DG = f"{D}/daily/dg"
POL = f"{D}/제도일정"
RES = f"{PAPERS}/26-2/03_차트CNN_연구/results"
P01 = f"{PAPERS}/26-2/03_차트CNN_연구/code/p01_build_panel.py"
PRED = f"{RES}/kaggle/pred_d_lr1e-4.parquet"
OUT_MD = f"{RES}/q01_panel_qa.md"
OUT_FLAGS = f"{RES}/q01_panel_flags.parquet"
T0 = time.time()
R = {}          # 숫자 결과 저장소 (보고서 문장에 끼워 넣는다)
S = {}          # 섹션별 markdown 조각


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def f0(x):
    return f"{int(round(x)):,}"


def pc(x, nd=2):
    return f"{x * 100:.{nd}f}%"


def tbl(df, idx=True, nd=4):
    """DataFrame → markdown 표 (tabulate 없이)."""
    d = df.reset_index() if idx else df.copy()
    cols = [str(c) for c in d.columns]
    rows = []
    for r in d.itertuples(index=False):
        out = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                out.append("" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else ("0" if v == 0 else f"{v:.{nd}g}")))
            elif isinstance(v, pd.Timestamp):
                out.append(str(v.date()))
            else:
                out.append(str(v))
        rows.append(out)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


# ------------------------------------------------------------------ 독립 제도표 (CSV 가 아니라 손으로 옮긴 값: KRX 문서·법령 요약 기준)
OLD_BANDS = [(0, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 500000, 500), (500000, 1e18, 1000)]   # ~2010-10-03 (market_rules_notes F12)
MID_BANDS = [(0, 1000, 1), (1000, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 500000, 500), (500000, 1e18, 1000)]  # 2010-10-04~2023-01-24 KOSPI
NEW_BANDS = [(0, 2000, 1), (2000, 5000, 5), (5000, 20000, 10), (20000, 50000, 50), (50000, 200000, 100), (200000, 500000, 500), (500000, 1e18, 1000)]  # 2023-01-25~
TAX_EXPECT = [("2010-01-01", "2014-12-31", 0.30), ("2015-01-01", "2019-06-02", 0.30), ("2019-06-03", "2020-12-31", 0.25),
              ("2021-01-01", "2022-12-31", 0.23), ("2023-01-01", "2023-12-31", 0.20), ("2024-01-01", "2024-12-31", 0.18),
              ("2025-01-01", "2025-12-31", 0.15), ("2026-01-01", "2026-12-31", 0.20)]


def tick_vec(price, dates):
    price = np.asarray(price, float)
    dates = pd.DatetimeIndex(dates)
    out = np.full(len(price), np.nan)
    for lo, hi, bands in [("1900-01-01", "2010-10-03", OLD_BANDS), ("2010-10-04", "2023-01-24", MID_BANDS), ("2023-01-25", "2100-01-01", NEW_BANDS)]:
        m = (dates >= lo) & (dates <= hi)
        edges = np.array([b[0] for b in bands], float)
        ticks = np.array([b[2] for b in bands], float)
        idx = np.searchsorted(edges, price[m], side="right") - 1
        out[m] = ticks[np.clip(idx, 0, len(ticks) - 1)]
    return out


def tax_expected(dates):
    out = np.full(len(dates), np.nan)
    dates = pd.DatetimeIndex(dates)
    for a, b, v in TAX_EXPECT:
        out[(dates >= a) & (dates <= b)] = v
    return out


def exact_limits(prev, dates, L):
    """KRX 산식: 기준가×L → 기준가 호가단위로 절사 → 가감 → 결과 가격대 호가단위로 절사(상한)/올림(하한)."""
    prev = np.asarray(prev, float)
    t0 = tick_vec(prev, dates)
    step = np.floor(prev * L / t0) * t0
    up = prev + step
    up = np.floor(up / tick_vec(up, dates)) * tick_vec(up, dates)
    dn = prev - step
    td = tick_vec(np.maximum(dn, 1), dates)
    dn = np.ceil(dn / td) * td
    return up, dn


# ------------------------------------------------------------------ setup
def load_raw(name):
    parts = []
    for suf in ("_2010_2014", ""):
        parts.append(pd.read_parquet(f"{CB}/{name}{suf}.parquet"))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"])


def setup():
    log("setup: panel 로드")
    P = pd.read_parquet(f"{CB}/panel.parquet")
    R["n_rows"], R["n_cols"] = P.shape
    sm = pd.read_parquet(f"{CB}/stock_master.parquet")
    sm["first_date"] = pd.to_datetime(sm["first_date"])
    sm["last_date"] = pd.to_datetime(sm["last_date"])
    codes = pd.factorize(P.ticker, sort=True)[0]
    dn = P.date.values.astype("int64")
    R["sorted_ok"] = bool(((codes[1:] > codes[:-1]) | ((codes[1:] == codes[:-1]) & (dn[1:] > dn[:-1]))).all())
    R["dup_keys"] = int(P.duplicated(["ticker", "date"]).sum())
    g = P.groupby("ticker", sort=False)
    P["prev_close"] = g.close.shift(1)
    P["prev_vol"] = g.volume.shift(1)
    P["prev_h"] = g.is_halted.shift(1)
    P["prev_adj"] = g.adj_close.shift(1)
    P["raw"] = P.close / P.prev_close - 1
    P["fac"] = P.adj_close / P.close
    P["dfac"] = P.fac / g.fac.shift(1) - 1
    P["rank_end"] = g.cumcount(ascending=False)
    P["idle"] = ((P.is_halted == 1) | (P.volume == 0)).astype(int)
    key = ((P.idle != P.groupby("ticker", sort=False).idle.shift(1)) | (P.ticker != P.ticker.shift(1))).cumsum()
    run = P.idle.groupby(key).cumsum() * P.idle
    P["prior_run"] = run.groupby(P.ticker, sort=False).shift(1).fillna(0)
    P["active"] = (P.volume > 0) & (P.is_halted == 0)
    P["resume"] = (P.prior_run > 0) & P.active
    fi = P[P.indiv_net.notna()].groupby("ticker").date.min()
    P = P.merge(fi.rename("first_inv"), left_on="ticker", right_index=True, how="left")
    lead = P.groupby("ticker", sort=False).apply(lambda d: int((d.date < d.first_inv.iloc[0]).sum()) if pd.notna(d.first_inv.iloc[0]) else 0, include_groups=False)
    ktickers = lead[lead >= 20].index
    P["prelist"] = (P.ticker.isin(ktickers) & (P.date < P.first_inv)).astype(int)
    R["prelist_tickers"] = list(ktickers)
    P["dl"] = P.groupby("ticker", sort=False).date.transform("max") < "2026-10-02"        # 상장폐지 종목
    P["liq"] = (P.dl & (P.rank_end <= 7)).astype(int)                                    # 정리매매 7거래일 + 재개일
    return P, sm


# ------------------------------------------------------------------ s1 커버리지
def s1(P, sm):
    log("s1 커버리지")
    out = []
    raw_p, raw_i, raw_t = load_raw("price"), load_raw("investor"), load_raw("trading")
    R["raw_rows"] = (len(raw_p), len(raw_i), len(raw_t))
    kp = set(zip(raw_p.code, raw_p.date))
    R["inv_not_in_price"] = int((~pd.Series(list(zip(raw_i.code, raw_i.date))).isin(kp)).sum()) if False else int(len(raw_i.merge(raw_p[["code", "date"]], on=["code", "date"], how="left", indicator=True).query("_merge=='left_only'")))
    R["trd_not_in_price"] = int(len(raw_t.merge(raw_p[["code", "date"]], on=["code", "date"], how="left", indicator=True).query("_merge=='left_only'")))
    R["panel_eq_price_rows"] = len(P) == len(raw_p)
    yrs = list(range(2010, 2027))
    rows = []
    for y in yrs:
        a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        py = P[(P.date >= a) & (P.date <= b)]
        ye = py.date.max()
        pn_any, pn_ye = py.ticker.nunique(), py[py.date == ye].ticker.nunique()
        mn_any = int(((sm.first_date <= b) & (sm.last_date >= a)).sum())
        # 마스터 first_date 는 DB 시작(1979-12-24) 이 많아 '상장일'이 아니다 → 2010 이전 상장분만 신뢰. last_date 는 10-02 에서 멈춘 스냅샷.
        mn_ye = int(((sm.first_date <= ye) & (sm.last_date >= ye)).sum()) if y < 2026 else int(((sm.first_date <= ye) & (sm.last_date >= '2026-10-02')).sum())
        rows.append((y, pn_any, mn_any, pn_ye, mn_ye, pn_ye - mn_ye))
    cov = pd.DataFrame(rows, columns=["연도", "패널(해당연도 1행↑)", "마스터(해당연도 상장)", "패널(연말)", "마스터(연말)", "연말 차이"]).set_index("연도")
    R["cov"] = cov
    # 상장폐지 마지막 날짜
    gmax = P.groupby("ticker").date.agg(["min", "max", "count"])
    m = sm.set_index("ticker").join(gmax, how="left")
    R["master_n"] = len(sm)
    R["in_panel"] = int(m["count"].notna().sum())
    R["not_in_panel"] = int(m["count"].isna().sum())
    R["not_in_panel_alive2010"] = int((m["count"].isna() & (m.last_date >= "2010-01-04")).sum())
    inn = m[m["count"].notna()]
    dl = inn[inn.last_date < "2026-10-02"]
    R["dl_n"] = len(dl)
    R["dl_match"] = int((dl["max"] == dl.last_date).sum())
    R["active_n"] = int((inn.last_date >= "2026-10-02").sum())
    R["active_last_ok"] = int((inn[inn.last_date >= "2026-10-02"]["max"] == P.date.max()).sum())
    # 첫 날짜
    fm = inn.first_date.clip(lower=pd.Timestamp("2010-01-04"))
    bad = inn[(inn["min"] != fm)]
    R["first_mismatch"] = [(t, r["name"], str(r.first_date.date()), str(r["min"].date())) for t, r in bad.iterrows()]
    # 상장폐지 마지막 행이 정리매매(상폐 직전)까지 들어 있나
    last_rows = P[P.dl & (P.rank_end == 0)]
    R["dl_last_vol_pos"] = float((last_rows.volume > 0).mean())
    R["dl_last_ret_med"] = float(last_rows.ret1d.median())
    # K-OTC 사전 구간
    pre = P[P.prelist == 1]
    kt = pre.groupby("ticker").agg(rows=("date", "size"), 시작=("date", "min"), 끝=("date", "max"), 거래량중앙값=("volume", "median"))
    kt = kt.join(sm.set_index("ticker")["name"])
    kt["상장후_거래량중앙값"] = [P[(P.ticker == t) & (P.prelist == 0)].volume.median() for t in kt.index]
    R["prelist_tbl"] = kt.reset_index()
    R["prelist_rows"] = int(len(pre))
    # KOSDAQ→KOSPI 이전 16종목
    mv = sm[sm.market_note.notna()]
    mvp = P[P.ticker.isin(mv.ticker)]
    R["moved_n"], R["moved_in_panel"], R["moved_rows"] = len(mv), mvp.ticker.nunique(), len(mvp)
    # 달력 대조 (로컬 DG 인덱스)
    cal_p = pd.DatetimeIndex(sorted(P.date.unique()))
    cl = pd.read_parquet(f"{DG}/adj_close.parquet").index
    cl = cl[(cl >= "2010-01-04") & (cl <= "2026-04-03")]
    R["cal_panel_only"] = [str(d.date()) for d in cal_p[(cal_p <= "2026-04-03") & ~cal_p.isin(cl)]]
    R["cal_dg_only"] = [str(d.date()) for d in cl[~cl.isin(cal_p)]]
    R["n_days"] = len(cal_p)
    cnt = P.groupby("date").ticker.size()
    R["per_day_min_max"] = (int(cnt.min()), int(cnt.max()))
    # 신규상장 직후 5거래일 (K-OTC 사전 구간 제외)
    gi = P.groupby("ticker", sort=False).cumcount()
    first = P.groupby("ticker", sort=False).date.transform("min")
    ipo = (first > "2010-01-04") & (P.prelist == 0)
    for nm, m in (("ipo_d1", ipo & (gi == 0)), ("ipo_d2_5", ipo & (gi >= 1) & (gi <= 4)), ("base", ~ipo & (gi >= 5))):
        x = P[m]
        R[nm] = {"rows": len(x), "ret_na": float(x.ret1d.isna().mean()), "mean_abs": float(x.ret1d.abs().mean()), "outlier": int(x.ret_outlier.sum()), "lim_up_bp": float(x.lim_up.mean() * 1e4), "halt": float((x.is_halted == 1).mean())}
    # 연도별 신규상장(패널 첫 행이 해당연도, 2010 제외)
    ipo = gmax[gmax["min"] > "2010-01-04"].groupby(gmax[gmax["min"] > "2010-01-04"]["min"].dt.year).size()
    R["ipo_by_year"] = ipo
    return


# ------------------------------------------------------------------ s2 가격 정합성
def s2(P, sm):
    log("s2 가격 정합성: DG 대조")
    cols = ["close", "adj_close", "open", "high", "low", "volume"]
    d = None
    for n in cols:
        x = pd.read_parquet(f"{DG}/{n}.parquet")
        x = x.loc["2010-01-04":"2026-04-03"]
        x.index.name = "date"
        x.columns.name = "ticker"
        t = x.stack().rename("dg_" + n).reset_index()
        t["ticker"] = t.ticker.astype(str)
        d = t if d is None else d.merge(t, on=["date", "ticker"], how="outer")
    B = P[["ticker", "date", "close", "adj_close", "open", "high", "low", "volume", "is_halted", "ret1d", "ret_outlier", "fac"]].merge(d, on=["ticker", "date"], how="outer", indicator=True)
    R["dg_only_price"] = int(((B._merge == "right_only") & B.dg_close.notna()).sum())
    R["panel_only_in_dg_window"] = int(((B._merge == "left_only") & (B.date <= "2026-04-03")).sum())
    Bb = B[B._merge == "both"].sort_values(["ticker", "date"]).reset_index(drop=True)
    R["n_both"] = len(Bb)
    mr = {}
    for c in ["close", "open", "high", "low", "volume"]:
        a, b = Bb[c], Bb["dg_" + c]
        has = b.notna()                       # dg close.parquet 는 2026-04-01 까지라 04-02·03 은 비교 제외
        mr[c] = (float((a[has] == b[has]).mean()), int((a[has] != b[has]).sum()), int(has.sum()), int((~has).sum()))
    R["match"] = mr
    R["adj_eq"] = float((Bb.adj_close == Bb.dg_adj_close).mean())
    g = Bb.groupby("ticker", sort=False)
    Bb["r_p"] = g.adj_close.pct_change()
    Bb["r_d"] = g.dg_adj_close.pct_change()
    Bb["dr"] = Bb.r_p - Bb.r_d
    Bb["ratio"] = Bb.adj_close / Bb.dg_adj_close
    R["dg_adj_int_frac"] = float((Bb.dg_adj_close == Bb.dg_adj_close.round()).mean())
    R["panel_adj_int_frac"] = float((P.adj_close == P.adj_close.round()).mean())
    tol = 0.5 / Bb.adj_close + 0.5 / Bb.groupby("ticker").adj_close.shift(1) + 0.5 / Bb.dg_adj_close + 0.5 / Bb.groupby("ticker").dg_adj_close.shift(1) + 1e-9
    Bb["tol"] = tol
    ok = Bb.dr.notna()
    adr = Bb.dr.abs()
    R["dr_q"] = {f"q{q}": float(adr[ok].quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)} | {"max": float(adr[ok].max())}
    R["dr_n"] = {str(t): int((adr[ok] > t).sum()) for t in (1e-9, 1e-6, 1e-4, 1e-3, 1e-2)}
    R["dr_total"] = int(ok.sum())
    R["dr_unexpl"] = int((ok & (adr > tol)).sum())
    R["dr_unexpl_tickers"] = int(Bb[ok & (adr > tol)].ticker.nunique())
    R["ratio_ne1_tickers"] = int((Bb.groupby("ticker").ratio.agg(lambda s: abs(s.median() - 1) > 1e-9)).sum())
    jump = (Bb.ratio / Bb.groupby("ticker").ratio.shift(1) - 1).abs() > 0.02
    R["ratio_jump_rows"] = int(jump.sum())
    # 상위 20 종목
    Bb["bad"] = (adr > 1e-9) & ok
    Bb["bad_unexpl"] = ok & (adr > tol)
    top = Bb.groupby("ticker").agg(불일치일=("bad", "sum"), 반올림초과일=("bad_unexpl", "sum"), 최대차이=("dr", lambda s: s.abs().max()), 패널adj중앙값=("adj_close", "median"), dg대비배율=("ratio", "median"))
    top = top.sort_values(["반올림초과일", "불일치일"], ascending=False).head(20).join(sm.set_index("ticker")["name"])
    top["원인"] = "정수 반올림 잡음"
    top.loc[(top.dg대비배율 - 1).abs() > 1e-9, "원인"] += " + 수준 차(dg 스냅샷 이후 조정)"
    R["dg_top20"] = top.reset_index()
    R["dg_real"] = Bb[Bb.bad_unexpl][["ticker", "date", "adj_close", "dg_adj_close", "r_p", "r_d"]].reset_index(drop=True)
    # 패널 adj_close 정수 반올림이 수익률에 주는 잡음
    ar = P.adj_close
    R["adj_lt"] = {k: float((ar < k).mean()) for k in (100, 500, 1000)}
    R["adj_lt_n"] = {k: int((ar < k).sum()) for k in (100, 500, 1000)}
    R["noise_bound_q"] = {q: float((1.0 / ar).quantile(q)) for q in (0.5, 0.9, 0.99)}
    # OHLC 정합
    R["ohlc"] = {"high<low": int((P.high < P.low).sum()), "high<close": int((P.high < P.close).sum()), "low>close": int((P.low > P.close).sum()),
                 "high<open": int((P.high < P.open).sum()), "low>open": int((P.low > P.open).sum())}
    # ---- ret_outlier 298 행 분류
    O = P[P.ret_outlier == 1].copy()
    O = O.merge(sm[["ticker", "name"]], on="ticker", how="left")
    L = np.where(O.date < "2015-06-15", 0.15, 0.30)
    big_raw = O.raw.abs() > L + 0.05
    penny = (O.adj_close <= 100) | (O.prev_adj <= 100) | ((O.close <= 200) & (O.prev_close <= 200))
    cause = np.select(
        [O.prelist == 1, O.liq == 1, (O.prior_run > 0) & big_raw, penny, ~big_raw],
        ["K-OTC 사전거래(상장 전 장외)", "정리매매 구간(상폐 직전 8행)", "거래재개일 가격 재설정(감자·병합·권리락 미보정 의심)", "극저가·수정주가 반올림 아티팩트(adj_close<=100 또는 종가<=200원)",
         "원주가는 정상, 수정계수만 단차"], "기타(원주가도 제한폭 초과)")
    O["원인"] = cause
    R["out_n"], R["out_tk"] = len(O), int(O.ticker.nunique())
    cs = O.groupby("원인").agg(행=("date", "size"), 종목=("ticker", "nunique"))
    cs["대표 종목"] = O.groupby("원인").apply(lambda x: ", ".join(f"{t}({n})" for t, n in zip(x.groupby("ticker").size().sort_values(ascending=False).head(4).index, x.groupby("ticker").size().sort_values(ascending=False).head(4).values)), include_groups=False)
    R["out_cause"] = cs.sort_values("행", ascending=False).reset_index()
    R["out_neg"] = int((O.ret1d < 0).sum())
    R["out_pos"] = int((O.ret1d > 0).sum())
    R["out_top_tickers"] = O.groupby(["ticker", "name"]).size().sort_values(ascending=False).head(10).reset_index(name="행")
    R["out_year"] = O.groupby(O.date.dt.year).size().to_dict()
    act_out = O[(O.원인 == "거래재개일 가격 재설정(감자·병합·권리락 미보정 의심)") | (O.원인.str.startswith("기타")) | (O.원인.str.startswith("원주가는"))]
    R["out_review"] = act_out[["ticker", "name", "date", "prev_close", "close", "ret1d", "prior_run"]].sort_values(["ticker", "date"])
    # 상폐 종목 종단 8행 수익: 수정주가 실제 vs ret1d_clean 복리 (정리매매에 극단값이 있는 종목만)
    dl = P[P.dl & (P.rank_end <= 8)]
    gg = dl.groupby("ticker")
    tr = pd.DataFrame({"실제": gg.adj_close.agg(lambda s: s.iloc[-1] / s.iloc[0] - 1), "clean": gg.ret1d_clean.agg(lambda s: np.prod(1 + s.iloc[1:].fillna(0)) - 1), "n_out": gg.ret_outlier.sum()})
    tw = tr[tr.n_out > 0]
    tx = tw[tw.실제.abs() < 5]
    R["term"] = {"dl_n": len(tr), "n": len(tw), "true_med": float(tw.실제.median()), "clean_med": float(tw.clean.median()), "true_mean_ex": float(tx.실제.mean()), "clean_mean_ex": float(tx.clean.mean()),
                 "n_ex": len(tx), "all_true_mean_ex": float(tr[tr.실제.abs() < 5].실제.mean()), "all_clean_mean_ex": float(tr[tr.실제.abs() < 5].clean.mean())}
    # 알려진 예외 004550
    a = P[P.ticker == "004550"]
    R["t004550"] = (len(a), float(a.fac.min()), float(a.fac.max()), int((a.ret_outlier == 1).sum()))
    R["t014900_in_panel"] = bool((P.ticker == "014900").any())
    return Bb


# ------------------------------------------------------------------ s3 거래정지
def s3(P):
    log("s3 거래정지")
    ct = pd.crosstab(P.is_halted, P.volume.fillna(-1).eq(0).map({True: "volume=0", False: "volume>0 또는 NaN"}))
    R["halt_ct"] = ct
    R["price_zero"] = {c: int((P[c] == 0).sum()) for c in ("open", "high", "low", "close", "adj_close")}
    h = P[P.is_halted == 1]
    R["halt_flat"] = float(((h.open == h.close) & (h.high == h.close) & (h.low == h.close)).mean())
    v0 = P[(P.volume == 0) & (P.is_halted == 0)]
    R["v0nh_flat"] = float(((v0.open == v0.close) & (v0.high == v0.close) & (v0.low == v0.close)).mean())
    R["v0nh_n"] = len(v0)
    R["halt_vol_pos"] = int(((P.is_halted == 1) & (P.volume > 0)).sum())
    R["vol_nan"] = int(P.volume.isna().sum())
    R["halt_year"] = P.groupby(P.date.dt.year).agg(정지=("is_halted", "sum"), 거래량0=("volume", lambda s: int((s == 0).sum())), 행=("volume", "size"))
    # 재개일
    rs = P[P.resume]
    R["resume_n"] = len(rs)
    R["resume_with_ret"] = int(rs.ret1d_clean.notna().sum())
    bk = pd.cut(rs.prior_run, [0, 1, 4, 19, 1e9], labels=["1일", "2~4일", "5~19일", "20일↑"])
    base = P[P.active & ~P.resume & P.ret1d_clean.notna()].ret1d_clean.abs()
    tab = rs.groupby(bk, observed=True).agg(재개일=("ret1d", "size"), 평균절대수익=("ret1d_clean", lambda s: s.abs().mean()), 중앙절대수익=("ret1d_clean", lambda s: s.abs().median()),
                                           절대10퍼센트이상=("ret1d_clean", lambda s: (s.abs() > 0.10).mean()), clean결측=("ret1d_clean", lambda s: s.isna().mean()))
    tab.loc["정상일(비교)"] = [len(base), base.mean(), base.median(), (base > 0.10).mean(), 0.0]
    R["resume_tab"] = tab
    R["resume_gt10_n"] = int((rs.ret1d_clean.abs() > 0.10).sum())
    long = rs[rs.prior_run >= 5]
    R["resume_long_n"] = len(long)
    # 정지일의 파생 열 (flow=0 이 NaN 이 아니다)
    idle = P[P.idle == 1]
    R["idle_flow_zero"] = int((idle.indiv_flow == 0).sum())
    R["idle_flow_na"] = int(idle.indiv_flow.isna().sum())
    R["idle_n"] = len(idle)
    # is_halted=1 & volume>0 (ret NaN 처리로 수익이 사라짐), volume=0 & is_halted=0
    R["lost_ret_halt_trade"] = int(((P.is_halted == 1) & (P.volume > 0)).sum())
    # 재개일 lim 판정
    R["resume_lim"] = (int(((P.lim_up == 1) & P.resume).sum()), int(((P.lim_dn == 1) & P.resume).sum()))
    # 정지 중 tradable 필터
    R["tradable_share"] = float(P.active.mean())


# ------------------------------------------------------------------ s4 투자자
def s4(P):
    log("s4 투자자")
    x = P.dropna(subset=["indiv_buy", "indiv_sell", "indiv_net"])
    R["inv_identity"] = (len(x), int((x.indiv_buy - x.indiv_sell != x.indiv_net).sum()))
    R["inv_no_sell_cols"] = [c for c in ("inst_sell", "foreign_sell") if c not in P.columns]
    nets = P[["indiv_net", "inst_net", "foreign_net", "other_corp_net"]]
    tot = nets.sum(axis=1, min_count=4)
    P["inv_res"] = tot
    P["inv_res_ratio"] = (tot / P.volume.replace(0, np.nan)).abs()
    ok = tot.notna()
    # 구간 경계: 2012년 중 잔차 비영 비율이 5% 아래로 내려가는 첫 날
    dd = P[(P.date < "2013-01-01") & ok].groupby("date").apply(lambda d: (d.inv_res != 0).mean(), include_groups=False)
    cut = dd[dd > 0.05].index.max()
    R["inv_cut_date"] = cut
    periods = [("2010-01-04", str(cut.date())), (str((cut + pd.Timedelta(days=1)).date()), "2026-05-25"), ("2026-05-26", "2026-08-06"), ("2026-08-07", "2026-12-31")]
    rows = []
    for a, b in periods:
        m = (P.date >= a) & (P.date <= b)
        t, r = tot[m & ok], P.inv_res_ratio[m & ok]
        rows.append((f"{a}~{b}", int(m.sum()), int((m & ok).sum()), (float((t != 0).mean()) if len(t) else np.nan), (float(r.median()) if len(r) else np.nan),
                     (float(r.quantile(0.99)) if len(r) else np.nan), (float(P.other_corp_net[m].isna().mean())), float(P.indiv_buy[m].isna().mean())))
    R["inv_period"] = pd.DataFrame(rows, columns=["구간", "행", "4열 모두 있음", "합계≠0 비율", "|합계|/거래량 중앙값", "|합계|/거래량 99%", "other_corp NaN", "indiv_buy NaN"]).set_index("구간")
    ys = P.date.dt.year
    R["inv_nan_year"] = P.indiv_net.isna().groupby(ys).sum()
    z = nets.eq(0).all(axis=1) & (P.volume > 0)
    zb = z & (P.volume >= 10000)
    t = pd.DataFrame({"행": P.groupby(ys).size(), "전부0(거래량>0)": z.groupby(ys).sum(), "전부0 & 거래량≥1만": zb.groupby(ys).sum()})
    t["전부0 비율"] = t["전부0(거래량>0)"] / t["행"]
    R["inv_zero_year"] = t
    R["inv_zero_n"], R["inv_zero_big_n"] = int(z.sum()), int(zb.sum())
    P["f_zero_inv"] = zb.astype(int)
    # 결측 행 원인
    ni = P.indiv_net.isna() & (P.date < "2026-01-01")
    R["inv_na_pre26"] = int(ni.sum())
    R["inv_na_tv_same"] = int((ni & P.trading_value.isna()).sum())
    R["inv_na_prelist"] = int((ni & (P.prelist == 1)).sum())
    R["inv_na_other_top"] = P[ni & (P.prelist == 0)].groupby("ticker").size().sort_values(ascending=False).head(3).to_dict()
    # 순매수 > 거래량
    over = {c: int((P[c].abs() > P.volume * 1.0001).sum()) for c in ("indiv_net", "inst_net", "foreign_net")}
    R["net_gt_vol"] = over
    ov = (P[["indiv_net", "inst_net", "foreign_net"]].abs().gt(P.volume * 1.0001, axis=0)).any(axis=1)
    P["f_net_gt_vol"] = ov.astype(int)
    R["net_gt_vol_dates"] = P[ov].groupby("date").size().to_dict()
    R["net_gt_vol_firstlast"] = (str(P[ov].date.min().date()), str(P[ov].date.max().date()), int(ov.sum()))
    # 정규화 열
    q = {}
    for c in ("indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv"):
        s = P[c]
        q[c] = {"min": s.min(), "q0.1%": s.quantile(.001), "q99.9%": s.quantile(.999), "max": s.max(), "|x|>1": int((s.abs() > 1).sum()), "|x|>5": int((s.abs() > 5).sum()),
                "|x|>20": int((s.abs() > 20).sum()), "표준편차": s.std(), "첨도": s.kurt()}
    R["flow_q"] = pd.DataFrame(q).T
    w = {}
    for c in ("indiv_flow", "inst_flow", "foreign_flow"):
        s, sw = P[c], P[c + "_w"]
        lo, hi = sw.min(), sw.max()
        w[c] = {"하한": lo, "상한": hi, "실제 하한분위": float((s <= lo).mean()), "실제 상한분위": float((s >= hi).mean()), "표준편차(원)": s.std(), "표준편차(_w)": sw.std(), "첨도(_w)": sw.kurt()}
        # 분위수 재현
        w[c]["0.5%/99.5% 재현"] = bool(np.isclose(lo, s.quantile(0.005)) and np.isclose(hi, s.quantile(0.995)))
        # 2014 이전 표본만으로 구한 경계
        e = s[P.date <= "2014-12-31"]
        w[c]["≤2014 경계 하"], w[c]["≤2014 경계 상"] = e.quantile(0.005), e.quantile(0.995)
    R["winsor"] = pd.DataFrame(w).T
    clipped = ((P.indiv_flow != P.indiv_flow_w) & P.indiv_flow.notna()).groupby(ys).mean()
    R["clip_year"] = clipped
    # 극단값 사례
    g = P.groupby("ticker", sort=False)
    P["avgvol60"] = g.volume.transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean())
    P["zero_in60"] = g.volume.transform(lambda s: (s == 0).shift(1).rolling(60, min_periods=1).sum())
    ex = P[P.indiv_flow.abs() > 5]
    R["flow_ex5"] = {"n": len(ex), "avgvol<1000": int((ex.avgvol60 < 1000).sum()), "avgvol<1만": int((ex.avgvol60 < 10000).sum()), "창에 거래량0일 포함": int((ex.zero_in60 > 0).sum()),
                     "당일거래량≥5×avgvol60": int((ex.volume >= 5 * ex.avgvol60).sum())}
    cases = P[P.indiv_flow.abs() > 20].copy()
    cases = cases.reindex(cases.indiv_flow.abs().sort_values(ascending=False).index).head(8)
    R["flow_cases"] = cases[["ticker", "date", "volume", "avgvol60", "indiv_net", "indiv_flow", "zero_in60"]].reset_index(drop=True)
    R["flow_gt20"] = int((P.indiv_flow.abs() > 20).sum())
    # 분할 후 60거래일
    ev = P[(P.dfac.abs() > 0.20)]
    w60 = np.zeros(len(P), bool)
    pos = pd.Series(np.arange(len(P)), index=P.index)
    tk = P.ticker.values
    starts = {}
    for t in ev.ticker.unique():
        starts[t] = np.where(tk == t)[0]
    for r in ev.itertuples():
        i = pos[r.Index]
        idx = starts[r.ticker]
        j = np.searchsorted(idx, i)
        w60[idx[j:j + 60]] = True
    P["f_split60"] = w60.astype(int)
    R["split_events"] = (len(ev), int(ev.ticker.nunique()), int(w60.sum()))
    big = P.indiv_flow.abs() > 5
    R["split_big_share"] = (int((big & w60).sum()), int(big.sum()))
    # 삼성전자 분할일 사례
    sam = P[(P.ticker == "005930") & (P.date == "2018-05-04")]
    R["samsung"] = (float(sam.indiv_flow.iloc[0]), float(sam.volume.iloc[0]), float(sam.avgvol60.iloc[0])) if len(sam) else None
    # 패치 효과: 분할 보정 분모 + max(,당일거래량)
    f = P.fac
    va = P.volume / f
    av = va.groupby(P.ticker, sort=False).transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean()) * f
    den = np.maximum(av, P.volume)
    fl = (P.indiv_net / den).replace([np.inf, -np.inf], np.nan)
    R["patch_flow"] = {"|x|>1": int((fl.abs() > 1).sum()), "max": float(fl.abs().max()), "std": float(fl.std()), "kurt": float(fl.kurt())}
    # 정지일 flow=0
    # 마지막 날들
    R["tail_dates"] = P.groupby("date").agg(tv_nan=("trading_value", lambda s: s.isna().mean()), ls_nan=("listed_shares", lambda s: s.isna().mean()),
                                            net_gt_vol=("f_net_gt_vol", "sum")).tail(6)


# ------------------------------------------------------------------ s5 거래대금
def s5(P):
    log("s5 거래대금")
    ok = (P.trading_value > 0) & (P.volume > 0)
    r = (P.trading_value / (P.close * P.volume))[ok]
    R["tv_ratio_q"] = {str(q): float(r.quantile(q)) for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)}
    R["tv_within10"] = float(((r >= 0.9) & (r <= 1.1)).mean())
    R["tv_n"] = int(ok.sum())
    R["tv_tails"] = {"<0.5": int((r < 0.5).sum()), ">2": int((r > 2).sum()), "<0.7": int((r < 0.7).sum()), ">1.3": int((r > 1.3).sum())}
    ys = P.date.dt.year[ok]
    R["tv_year"] = pd.DataFrame({"중앙값": r.groupby(ys).median(), "|비율-1|>20%": r.groupby(ys).apply(lambda s: (s.sub(1).abs() > 0.2).mean()), "행": r.groupby(ys).size()})
    bad = P[ok].assign(비율=r)[(r < 0.7) | (r > 1.3)]
    bt = bad.groupby("ticker").agg(행=("date", "size"), 시작=("date", "min"), 끝=("date", "max"), 비율중앙값=("비율", "median")).sort_values("행", ascending=False)
    R["tv_bad"] = bt.head(8).reset_index()
    R["tv_bad_n"] = (len(bad), int(bad.ticker.nunique()))
    a = P[P.ticker == "006740"]
    q = (a.trading_value / (a.close * a.volume)).where(a.volume > 0)
    first_bad = a.index[(q < 0.5)]
    R["t006740"] = (str(a.loc[first_bad[0], "date"].date()) if len(first_bad) else None, int(len(first_bad)), float(q[q < 0.5].median()) if len(first_bad) else None)
    R["tv_nan"] = int(P.trading_value.isna().sum())
    R["tv_nan_last"] = int(P.trading_value.isna()[P.date == P.date.max()].sum())
    R["tv_nan_year"] = P.trading_value.isna().groupby(P.date.dt.year).sum()
    R["tv0_vol_pos"] = int(((P.trading_value == 0) & (P.volume > 0)).sum())
    R["tv_pos_vol0"] = int(((P.trading_value > 0) & (P.volume == 0)).sum())
    R["ls_nan"] = (int(P.listed_shares.isna().sum()), str(P[P.listed_shares.isna()].date.min().date()))
    R["mktcap_nan"] = int(P.mktcap.isna().sum())
    nch = P.groupby("ticker").listed_shares.nunique()
    R["ls_const_share"] = float((nch <= 1).mean())


# ------------------------------------------------------------------ s6 제약 열
def s6(P):
    log("s6 제약 열")
    ys = P.date.dt.year
    t = P.groupby(ys).agg(행=("lim_up", "size"), 상한가=("lim_up", "sum"), 하한가=("lim_dn", "sum"))
    t["상한가 bp"] = t.상한가 / t.행 * 1e4
    t["하한가 bp"] = t.하한가 / t.행 * 1e4
    R["lim_year"] = t
    L = P.limit_pct / 100
    R["limit_vals"] = P.limit_pct.value_counts().to_dict()
    R["limit_boundary"] = bool(((P.date < "2015-06-15") == (P.limit_pct == 15)).all())
    # 정확 상·하한가 재현 (호가단위 반영)
    elig = P.prev_close.notna() & (P.prev_vol > 0) & (P.prev_h == 0) & (P.volume > 0) & (P.dfac.abs() < 0.01) & P.active
    up, dn = exact_limits(P.prev_close.fillna(1).values, P.date, L.values)
    ex_up = (P.close.values >= up) & elig.values
    ex_dn = (P.close.values <= dn) & elig.values
    fu, fd = (P.lim_up == 1).values, (P.lim_dn == 1).values
    R["lim_exact"] = {"elig_rows": int(elig.sum()),
                      "up": {"exact": int(ex_up.sum()), "flag(elig)": int((fu & elig.values).sum()), "둘다": int((ex_up & fu).sum()), "exact만(누락)": int((ex_up & ~fu).sum()), "flag만(과검출)": int((~ex_up & fu & elig.values).sum())},
                      "dn": {"exact": int(ex_dn.sum()), "flag(elig)": int((fd & elig.values).sum()), "둘다": int((ex_dn & fd).sum()), "exact만(누락)": int((ex_dn & ~fd).sum()), "flag만(과검출)": int((~ex_dn & fd & elig.values).sum())}}
    # 연도별 누락/과검출 (정확 기준)
    yy = ys.values
    rows = []
    for y in range(2010, 2027):
        m = yy == y
        rows.append((y, int((ex_up & m).sum()), int((ex_up & ~fu & m).sum()), int((~ex_up & fu & elig.values & m).sum()), int((ex_dn & m).sum()), int((ex_dn & ~fd & m).sum()), int((~ex_dn & fd & elig.values & m).sum())))
    R["lim_exact_year"] = pd.DataFrame(rows, columns=["연도", "상한 exact", "상한 누락", "상한 과검출", "하한 exact", "하한 누락", "하한 과검출"]).set_index("연도")
    # 원인별 오판: 조정계수 단차, 재개일, 정리매매
    for nm, col in (("up", P.lim_up), ("dn", P.lim_dn)):
        x = P[col == 1]
        R[f"lim_fp_{nm}"] = {"flag": len(x), "계수 단차 5%↑": int((x.dfac.abs() > 0.05).sum()), "수정-원 수익 차 5%p↑": int(((x.ret1d - x.raw).abs() > 0.05).sum()), "재개일": int(x.resume.sum()),
                             "정리매매 구간": int((x.liq == 1).sum()), "직전행 정지": int(((x.prev_h == 1) | (x.prev_vol == 0)).sum())}
    fpx = P[((P.lim_up == 1) | (P.lim_dn == 1)) & (P.dfac.abs() > 0.05)]
    R["lim_fp_examples"] = fpx[["ticker", "date", "prev_close", "close", "adj_close", "raw", "ret1d", "dfac"]].head(6).reset_index(drop=True)
    R["lim_fp_ex_n"] = len(fpx)
    # 기준 −0.4%p 임계의 근거
    near = ((P.raw - L) >= -0.03) & ((P.raw - L) < -0.004) & (P.volume > 0)
    R["lim_near_below"] = int(near.sum())
    # 놓친 상한가: 수정 수익률로는 한도인데 원수익률 판정은 0
    miss = (P.lim_up == 0) & (P.ret1d >= L - 0.004) & (P.volume > 0) & (P.ret1d.abs() <= L + 0.05)
    R["lim_miss_corp"] = int(miss.sum())
    # 공매도 구간
    ss = pd.read_csv(f"{POL}/short_selling_regime.csv", parse_dates=["start_date", "end_date"])
    cal = pd.DatetimeIndex(sorted(P.date.unique()))
    exp = {}
    for r in ss.itertuples():
        exp[r.regime] = exp.get(r.regime, 0) + int(((cal >= r.start_date) & (cal <= r.end_date)).sum())
    n2011 = int(((cal >= "2011-08-10") & (cal <= "2011-11-09")).sum())
    nbefore = int((cal < "2015-01-01").sum())
    exp["전면금지"] += n2011
    exp["전면허용"] += nbefore - n2011
    got = P.groupby("short_regime").date.nunique().to_dict()
    R["short_tbl"] = pd.DataFrame({"패널 일수": got, "CSV+하드코딩 기대 일수": exp}).fillna(0).astype(int)
    R["short_tbl"]["일치"] = R["short_tbl"].iloc[:, 0] == R["short_tbl"].iloc[:, 1]
    pa = P[P.short_regime == "부분허용"]
    R["short_partial"] = (int(pa.date.nunique()), len(pa), str(pa.date.min().date()), str(pa.date.max().date()))
    R["short_banned_in_partial"] = int((pa.short_banned == 1).sum())
    fin_period = P[(P.date >= "2010-01-04") & (P.date <= "2013-11-13") & (P.short_regime == "전면허용")]
    R["short_fin_rows"] = (int(fin_period.date.nunique()), len(fin_period))
    # sell_tax: 독립 기대값 대조
    ex_t = tax_expected(P.date)
    mism = ~np.isclose(P.sell_tax_pct.values, ex_t, atol=1e-9)
    R["tax_mism"] = int(mism.sum())
    R["tax_mism_dates"] = sorted({str(d.date()) for d in P.date[mism].unique()})[:6]
    R["tax_assumed_ok"] = bool(((P.date < "2015-01-01") == (P.sell_tax_assumed == 1)).all())
    samples = ["2012-06-01", "2015-03-02", "2019-05-31", "2019-06-03", "2021-01-04", "2022-12-29", "2023-01-02", "2023-06-01", "2024-06-03", "2025-06-02", "2026-02-02"]
    rows = []
    for s in samples:
        r = P[P.date == s].iloc[0]
        e = float(tax_expected([pd.Timestamp(s)])[0])
        rows.append((s, r.ticker, r.sell_tax_pct, e, "OK" if abs(r.sell_tax_pct - e) < 1e-9 else "불일치", int(r.sell_tax_assumed)))
    R["tax_samples"] = pd.DataFrame(rows, columns=["날짜", "종목", "패널 sell_tax_pct", "수기 기대값", "판정", "assumed 플래그"])
    # tick: 독립 기대값 대조
    et = tick_vec(P.close.values, P.date)
    tm = ~np.isclose(P.tick.values, et)
    era = np.where(P.date < "2010-10-04", "A ~2010-10-01", np.where(P.date < "2023-01-25", "B 2010-10-04~2023-01-24", "C 2023-01-25~"))
    tt = pd.DataFrame({"era": era, "mism": tm}).groupby("era").mism.agg(["size", "sum"])
    tt["불일치 비율"] = tt["sum"] / tt["size"]
    R["tick_era"] = tt.rename(columns={"size": "행", "sum": "불일치 행"})
    a = P.date < "2010-10-04"
    R["tick_a"] = {"rows": int(a.sum()), "tick_pct_med_panel": float(P.tick_pct[a].median()), "tick_pct_med_true": float((pd.Series(et, index=P.index) / P.close * 100)[a].median()),
                   "ratio_med": float((pd.Series(et, index=P.index)[a] / P.tick[a]).median()), "tick_pct_med_after": float(P.tick_pct[(P.date >= "2010-10-04") & (P.date < "2011-01-01")].median())}
    rows = []
    picks = [("2010-03-02", 1000, 4999), ("2010-03-02", 20000, 49999), ("2010-10-01", 500, 900), ("2010-10-04", 500, 900), ("2016-05-02", 10000, 49999),
             ("2023-01-24", 10000, 19999), ("2023-01-25", 10000, 19999), ("2023-01-24", 1000, 1999), ("2023-01-25", 1000, 1999)]
    for d_, lo, hi in picks:
        x = P[(P.date == d_) & (P.close >= lo) & (P.close <= hi)]
        if len(x):
            r = x.iloc[0]
            e = float(tick_vec([r.close], [pd.Timestamp(d_)])[0])
            rows.append((d_, r.ticker, int(r.close), r.tick, e, "OK" if abs(r.tick - e) < 1e-9 else "불일치"))
    R["tick_samples"] = pd.DataFrame(rows, columns=["날짜", "종목", "종가", "패널 tick", "수기 기대 tick", "판정"])
    R["tick_pct_eraA_bad_n"] = int(tm[a.values].sum())


# ------------------------------------------------------------------ s7 룩어헤드
RAW_COLS = ["open", "high", "low", "close", "adj_close", "volume", "listed_shares", "is_halted", "indiv_buy", "indiv_sell", "indiv_net", "inst_buy", "inst_net", "foreign_buy", "foreign_net",
            "other_corp_net", "trading_value"]
DER = ["ret1d", "ret_outlier", "ret1d_clean", "mktcap", "tv_filled", "tv_is_approx", "indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv", "limit_pct", "lim_up", "lim_dn", "short_regime",
       "short_banned", "sell_tax_pct", "sell_tax_assumed", "tick", "tick_pct", "str5", "str20", "mom12_1", "near52h", "vol20", "abvol", "hl_range"]


def run_p01(src_body, IN):
    ns = {"_IN": IN, "__name__": "p01_exec"}
    exec(compile(src_body, "p01_exec", "exec"), ns)
    return ns["df"]


def prep_src():
    src = open(P01, encoding="utf-8").read()
    cut = src.index('out = f"{CB}/panel.parquet"')
    body = src[:cut]
    for a, b in [('price = load("price")', 'price = _IN["price"].copy()'), ('inv = load("investor")', 'inv = _IN["investor"].copy()'),
                 ('trd = load("trading")[["code", "date", "trading_value"]]', 'trd = _IN["trading"][["code", "date", "trading_value"]].copy()')]:
        assert body.count(a) == 1, a
        body = body.replace(a, b)
    return body


def cmp_cols(A, B, cols, T=None):
    """A·B 를 (ticker,date)로 맞춰 열별 불일치 행 수. T 가 있으면 date<=T 만."""
    m = A[["ticker", "date"] + cols].merge(B[["ticker", "date"] + cols], on=["ticker", "date"], suffixes=("_a", "_b"))
    if T is not None:
        m = m[m.date <= T]
    out = {}
    for c in cols:
        a, b = m[c + "_a"], m[c + "_b"]
        if a.dtype == object or b.dtype == object:
            out[c] = int((a != b).sum())
        else:
            av, bv = a.astype(float).values, b.astype(float).values
            out[c] = int((~np.isclose(av, bv, rtol=1e-9, atol=1e-12, equal_nan=True)).sum())
    return out, len(m)


SPECIAL = ["005930", "000790", "008080", "013890", "004550", "089590", "006740", "001140", "003560", "000030"]


def subset(P):
    """무작위 110종목 + 특수 종목. 룩어헤드 실험(s7)과 패치 검증(s9)이 같은 표본을 쓴다."""
    rng = np.random.default_rng(20261010)
    rawt = {"price": load_raw("price"), "investor": load_raw("investor"), "trading": load_raw("trading")}
    tickers = sorted(P.ticker.unique())
    sub = sorted(set(rng.choice(tickers, 110, replace=False)) | set(SPECIAL))
    codes = {"A" + t for t in sub}
    IN0 = {k: v[v.code.isin(codes)].copy() for k, v in rawt.items()}
    return sub, IN0, rng


def s7(P):
    log("s7 룩어헤드: p01 exec 실험")
    sub, IN0, rng = subset(P)
    body = prep_src()
    base = run_p01(body, IN0)
    log("  재현 완료", len(base))
    # (1) 재현: 저장 패널과 일치?
    comp = [c for c in DER if c in P.columns]
    stored = P[P.ticker.isin(sub)]
    rep, n = cmp_cols(base, stored, comp)
    R["rep_tickers"], R["rep_rows"], R["rep_diff"] = len(sub), n, rep
    # _w 는 분위수가 전체 표본 기준이라 부분집합 재현 대상이 아님 → 저장 경계로 clip 재현만 확인
    lo_hi = {c: (P[c + "_w"].min(), P[c + "_w"].max()) for c in ("indiv_flow", "inst_flow", "foreign_flow")}
    wd = {}
    for c, (lo, hi) in lo_hi.items():
        s = stored.set_index(["ticker", "date"])
        wd[c] = int((~np.isclose(s[c].clip(lo, hi), s[c + "_w"], equal_nan=True)).sum())
    R["w_clip_repro_diff"] = wd
    # (2) 절단(prefix) 불변성
    T = pd.Timestamp("2019-06-28")
    INt = {k: v[v.date <= T].copy() for k, v in IN0.items()}
    trunc = run_p01(body, INt)
    d_tr, n_tr = cmp_cols(trunc, base, comp, T)
    R["trunc"] = (str(T.date()), n_tr, d_tr)
    # (3) 미래 교란: T 이후 값 전부 무작위 변경
    def perturb(IN, only=None):
        out = {k: v.copy() for k, v in IN.items()}
        for k, v in out.items():
            m = (v.date > T) if only is None else ((v.date > T) & v.code.isin(only))
            for c in v.columns:
                if c in ("code", "date"):
                    continue
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
    pert = run_p01(body, perturb(IN0))
    d_pt, n_pt = cmp_cols(pert, base, comp, T)
    d_after, n_after = cmp_cols(pert, base, comp)
    R["pert"] = (str(T.date()), n_pt, d_pt)
    R["pert_after_changed"] = {c: d_after[c] - d_pt[c] for c in comp}
    # (4) 한 종목(005930)만 교란 → 다른 종목 전 기간 불변
    pert1 = run_p01(body, perturb(IN0, only={"A005930"}))
    oth = [t for t in sub if t != "005930"]
    d_one, n_one = cmp_cols(pert1[pert1.ticker.isin(oth)], base[base.ticker.isin(oth)], comp)
    R["pert_one"] = (n_one, d_one)
    # (5) _w 경계의 전체표본 의존성: 부분집합에서 미래 교란이 과거 _w 에 영향?
    bw = base.copy(); pw = pert.copy()
    wcols = ["indiv_flow_w", "inst_flow_w", "foreign_flow_w"]
    dw, nw = cmp_cols(pw, bw, wcols, T)
    R["w_future_dep"] = (nw, dw)
    # (6) 양성 대조: 일부러 미래를 쓰는 열이 실험에 걸리는가
    for X in (base, pert):
        gg = X.groupby("ticker", sort=False)
        X["ctrl_fwd5"] = gg.adj_close.shift(-5) / X.adj_close - 1
        X["ctrl_center"] = gg.adj_close.transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
    dc, _ = cmp_cols(pert, base, ["ctrl_fwd5", "ctrl_center"], T)
    R["ctrl"] = dc
    # (7) 창 정의 독립 재계산
    x = base.sort_values(["ticker", "date"]).reset_index(drop=True)
    gg = x.groupby("ticker", sort=False)
    mom = gg.adj_close.shift(21) / gg.adj_close.shift(252) - 1
    ok = x.mom12_1.notna()
    R["mom_check"] = (int(ok.sum()), int((~np.isclose(mom[ok], x.mom12_1[ok])).sum()))
    hi52 = gg.adj_close.transform(lambda s: s.rolling(252, min_periods=120).max())
    nh = x.adj_close / hi52
    ok = x.near52h.notna()
    R["near_check"] = (int(ok.sum()), int((~np.isclose(nh[ok], x.near52h[ok])).sum()), float(x.near52h.max()), int((x.near52h > 1 + 1e-12).sum()))
    # avgvol60: 행 슬라이스로 명시 계산 (오늘 제외)
    chk = x[(x.indiv_flow.notna()) & (x.indiv_flow != 0) & (x.indiv_net != 0)].sample(2500, random_state=1)
    ex_ok = ex_in = 0
    for r in chk.itertuples():
        s = x.loc[(x.ticker == r.ticker) & (x.date < r.date), "volume"].tail(60)
        den = r.indiv_net / r.indiv_flow
        if len(s) >= 30:
            ex_ok += int(np.isclose(s.mean(), den, rtol=1e-6))
            s2 = x.loc[(x.ticker == r.ticker) & (x.date <= r.date), "volume"].tail(60)
            ex_in += int(np.isclose(s2.mean(), den, rtol=1e-6))
    R["avgvol_check"] = (len(chk), ex_ok, ex_in)
    # vol20 오염: ret1d(미청소) 사용
    g = P.groupby("ticker", sort=False)
    bad20 = g.ret_outlier.transform(lambda s: s.rolling(20, min_periods=1).max())
    R["vol20"] = {"n": int(P.vol20.notna().sum()), "outlier_in_window": int((P.vol20.notna() & (bad20 == 1)).sum()), "gt0.15": int((P.vol20 > 0.15).sum()),
                  "gt0.15_with_outlier": int(((P.vol20 > 0.15) & (bad20 == 1)).sum()), "max": float(P.vol20.max())}
    resume_in20 = g.resume.transform(lambda s: s.rolling(20, min_periods=1).max()) if "resume" in P else None
    R["vol20"]["resume_in_window"] = int((P.vol20.notna() & (resume_in20 == 1)).sum())
    # str5/str20/mom 의 재개일 누적 수익 포함
    R["str_resume"] = {c: int((P[c].notna() & (g.resume.transform(lambda s, w=w: s.rolling(w, min_periods=1).max()) == 1)).sum()) for c, w in (("str5", 5), ("str20", 20))}
    # 전체표본 분위수 vs 확장창
    # (8) listed_shares 시점성 — s5 에 있음
    R["sub_n"] = len(sub)


# ------------------------------------------------------------------ s8 함정
def s8(P, sm):
    log("s8 함정")
    R["tk_len"] = P.ticker.str.len().value_counts().to_dict()
    R["tk_nondigit"] = sorted(P.ticker[~P.ticker.str.fullmatch(r"\d{6}")].unique())
    last = P.drop_duplicates("ticker").ticker.str[-1]
    pref = P.ticker.str[-1] != "0"
    R["pref"] = {"tickers": int(P[pref].ticker.nunique()), "all_tickers": int(P.ticker.nunique()), "rows": int(pref.sum()), "row_share": float(pref.mean()),
                 "digits": P[pref].drop_duplicates("ticker").ticker.str[-1].value_counts().to_dict()}
    names = sm.set_index("ticker").name
    pt = P[pref].drop_duplicates("ticker").ticker.head(6)
    R["pref_examples"] = [f"{t} {names.get(t, '')}" for t in pt]
    # 이름으로 본 우선주 의심 (끝자리 0 인데 이름이 '우' 로 끝)
    nm = P.drop_duplicates("ticker")[["ticker"]].assign(name=lambda d: d.ticker.map(names))
    sus = nm[(nm.ticker.str[-1] == "0") & nm.name.fillna("").str.contains(r"우(?:\(전환\)|[A-C]?)$|우선주", regex=True)]
    R["pref_name_sus"] = sus.values.tolist()[:8]
    # CNN 예측 조인
    out = {}
    cal = pd.DatetimeIndex(sorted(P.date.unique()))
    key = P[["ticker", "date", "volume", "is_halted", "adj_close", "prelist"]].copy()
    g = key.groupby("ticker", sort=False)
    key["fwd5"] = g.adj_close.shift(-5) / key.adj_close - 1
    rows = []
    for name in ("pred_d_lr1e-4", "pred_w_lr1e-4", "pred_d_novol", "pred_w_h20", "pred_d_lr1e-5", "pred_d_lr1e-4_tb", "pred_w_lr1e-5", "pred_w_lr1e-4_tb"):
        try:
            pr = pd.read_parquet(f"{RES}/kaggle/{name}.parquet")
        except Exception:
            continue
        pr["date"] = pd.to_datetime(pr["date"]).astype("datetime64[ns]")
        m = pr.merge(key, on=["ticker", "date"], how="left", indicator=True)
        rows.append((name, len(pr), int((m._merge == "both").sum()), float((m._merge == "both").mean()), int(pr.ticker.nunique()), int((~pr.date.isin(cal)).sum())))
        if name == "pred_d_lr1e-4":
            R["pred_df"] = pr
            R["pred_m"] = m
    R["pred_tbl"] = pd.DataFrame(rows, columns=["파일", "예측 행", "패널과 매칭", "매칭률", "종목수", "패널 달력에 없는 날짜"])
    m = R.pop("pred_m")
    pr = R.pop("pred_df")
    R["pred_unmatched"] = int((m._merge != "both").sum())
    um = m[m._merge != "both"]
    R["pred_unmatched_reasons"] = {"ticker 가 패널에 없음": int((~um.ticker.isin(P.ticker.unique())).sum()), "날짜가 패널 달력에 없음": int((~um.date.isin(cal)).sum())}
    b = m[m._merge == "both"]
    R["pred_dtype"] = (str(pr.ticker.dtype), int(pr.ticker.str.len().eq(6).sum()), int((pr.ticker.str[-1] != "0").sum()))
    R["pred_form_in_cal"] = float(pr.date.isin(cal).mean())
    # 주 마지막 거래일인지
    wk = pd.Series(cal, index=cal).groupby(cal.to_period("W")).max()
    R["pred_weekend_ok"] = float(pr.date.isin(set(wk.values)).mean())
    R["pred_vol0"] = (int((b.volume == 0).sum()), int((b.is_halted == 1).sum()), int((b.prelist == 1).sum()))
    ok = b.fwd5.notna()
    dif = (b.ret - b.fwd5)[ok].abs()
    R["pred_ret_cmp"] = {"비교 행": int(ok.sum()), "중앙값": float(dif.median()), "99%": float(dif.quantile(0.99)), "max": float(dif.max()), "|차|>1%p": int((dif > 0.01).sum()), "|차|>10%p": int((dif > 0.10).sum()),
                         "상관": float(np.corrcoef(b.ret[ok], b.fwd5[ok])[0, 1]), "패널 fwd5 NaN": int((~ok).sum())}
    big = b[ok & ((b.ret - b.fwd5).abs() > 0.10)]
    R["pred_ret_big"] = big[["ticker", "date", "ret", "fwd5"]].head(6)
    # 형성일 패널 거래일 중 정지·거래량0
    return


# ------------------------------------------------------------------ 보고서 조립
def build_flags(P):
    f = P[["ticker", "date"]].copy()
    f["f_resume_day"] = P.resume.astype("uint8")
    f["f_prelist_otc"] = P.prelist.astype("uint8")
    f["f_liquidation_zone"] = P.liq.astype("uint8")
    f["f_zero_inv_suspect"] = P.f_zero_inv.astype("uint8")
    f["f_net_gt_volume"] = P.f_net_gt_vol.astype("uint8")
    f["f_split60"] = P.f_split60.astype("uint8")
    f["f_idle"] = P.idle.astype("uint8")
    f["f_tradable"] = P.active.astype("uint8")
    f["tick_fixed"] = tick_vec(P.close.values, P.date).astype("float32")
    f.to_parquet(OUT_FLAGS, index=False)
    return f


PATCHES = {
    "P1": """# ---- P1. 거래재개일 누적수익 분리 (p01 의 ret1d 를 NaN 처리하는 줄 바로 아래) ----------------------
idle = (df.is_halted == 1) | (df.volume == 0)
prev_idle = idle.groupby(df.code, sort=False).shift(1, fill_value=False)
df["resume_day"] = (prev_idle & ~idle).astype(int)                 # 직전 행이 정지/거래량0 이고 오늘 거래
df["ret_multi"] = df["ret1d"].where(df.resume_day == 1)            # 여러 날 누적 수익은 따로 보관
df.loc[df.resume_day == 1, "ret1d"] = np.nan                        # 단일일 수익 열(ret1d, ret1d_clean, vol20 입력)에서는 제외
""",
    "P2": """# ---- P2. 정리매매·상폐 직전 손실을 '이상값'으로 지우지 않기 (ret_outlier 정의 줄을 대체) ---------------
rank_end = g.cumcount(ascending=False)
delisted = g["date"].transform("max") < df.date.max() - pd.Timedelta(days=10)   # 또는 stock_master.last_date
df["liq_zone"] = (delisted & (rank_end <= 7)).astype(int)
df["ret_outlier"] = ((df.ret1d.abs() > _L + 0.10) & (df.liq_zone == 0)).astype(int)
# 극저가 수정주가 반올림 구간(adj_close<=100)은 따로 비운다
df.loc[(df.adj_close <= 100) | (g["adj_close"].shift(1) <= 100), "ret_outlier"] = 1
# 라벨(미래수익)은 ret1d_clean 이 아니라 수정주가로 직접 만들고, 상폐 종목은 마지막 정리매매 종가까지 수익을 포함한다.
""",
    "P3": """# ---- P3. 2010-10-04 이전 호가단위 (df["tick"].fillna(1.0) 바로 앞) -------------------------------------
OLD = [(0, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 500000, 500), (500000, 1e18, 1000)]   # market_rules_notes F12 [2차]
m0 = df.date < "2010-10-04"
for lo, hi, t in OLD:
    df.loc[m0 & (df.close >= lo) & (df.close < hi), "tick"] = t
df["tick_assumed"] = m0.astype(int)
""",
    "P4": """# ---- P4. K-OTC(장외) 사전거래 구간 제외 (is_halted 정리 줄 아래, g = df.groupby(...) 앞) ---------------
first_inv = df.loc[df.indiv_net.notna()].groupby("code").date.min().rename("first_inv")
df = df.merge(first_inv, on="code", how="left")
lead = df.date < df.first_inv
df = df[~(lead & (lead.groupby(df.code).transform("sum") >= 20))].drop(columns="first_inv")   # 상장 전 20행 이상 투자자 자료가 없는 선두 구간
""",
    "P5": """# ---- P5. 투자자 열: 결측을 0 으로 채운 행과 정의가 바뀐 구간 처리 (P4 바로 아래) ----------------------
nets = ["indiv_net", "inst_net", "foreign_net", "other_corp_net"]
z = (df[nets] == 0).all(axis=1) & (df.volume >= 10000)             # 거래량이 큰데 4그룹이 모두 0 → 결측
df.loc[z, nets] = np.nan
df.loc[df.date >= "2026-05-26", "foreign_net"] = np.nan             # 기타외국인 제외 정의로 바뀐 것으로 추정 → 쓰려면 foreign_etc 를 더해 복원
over = df[["indiv_net", "inst_net", "foreign_net"]].abs().max(axis=1) > df.volume.fillna(0) * 1.0001   # 순매수 > 거래량(최근 2일)
df.loc[over, nets] = np.nan
""",
    "P6": """# ---- P6. *_flow 분모: 분할 보정 + 거래량 0 일 제외 + 상한 1 (기존 flow 반복문 아래) -----------------------
f = df.adj_close / df.close                                         # 분할 계수(과거→현재 비율만 쓰므로 룩어헤드 없음)
va = (df.volume / f).where(df.volume > 0)                           # 정지일(0)은 평균에서 제외
av = va.groupby(df.code, sort=False).transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean()) * f
for c in ("indiv", "inst", "foreign"):
    df[f"{c}_flow"] = df[f"{c}_net"] / np.maximum(av, df.volume)    # [-1, 1] 로 묶인다
# _w 대신 날짜별 횡단면 순위(미래 정보 없음): df[c + "_rk"] = df.groupby("date")[c].rank(pct=True) - 0.5
""",
    "P7": """# ---- P7. 상·하한가 판정: 호가단위 포함 정확한 한도가와 비교 (q01_panel_qa.py 의 exact_limits 함수 참고) ----
# 재개일(직전 행이 정지)·조정계수 단차일(|adj/close 변화|>1%)·정리매매 구간은 판정에서 제외한다.
""",
}
PATCH = "```python\n" + "\n".join(PATCHES.values()) + "```"


def s9(P):
    """패치 P1~P6 을 p01 소스에 끼워 넣어 부분집합(118종목)에서 실제로 돌려 효과를 확인한다."""
    log("s9 패치 검증")
    sub, IN0, _ = subset(P)
    body = prep_src()
    anchors = [
        ('df.loc[(df.is_halted == 1) | (df.volume == 0), "ret1d"] = np.nan', "after", PATCHES["P1"]),
        ('df["ret_outlier"] = (df.ret1d.abs() > _L + 0.10).astype(int)', "replace", PATCHES["P2"]),
        ('df["tick"] = df["tick"].fillna(1.0)', "before", PATCHES["P3"]),
        ('df["is_halted"] = df["is_halted"].fillna(0).astype(int)', "after", PATCHES["P4"] + PATCHES["P5"]),
        ('    df[f"{c}_flow"] = df[f"{c}_net"] / avgvol60', "after", PATCHES["P6"]),
    ]
    pb = body
    for a, how, code in anchors:
        assert pb.count(a) == 1, a
        pb = pb.replace(a, {"after": a + "\n" + code, "before": code + "\n" + a, "replace": code}[how])
    A = run_p01(body, IN0)
    B = run_p01(pb, IN0)
    out = {"rows": (len(A), len(B))}
    ref = P[P.ticker.isin(sub)]
    out["p4_removed"] = (len(A) - len(B), int((ref.prelist == 1).sum()))
    out["p1_resume"] = (int(B.resume_day.sum()), int(ref[ref.prelist == 0].resume.sum()))
    out["p1_ret_na"] = bool(B.loc[B.resume_day == 1, "ret1d"].isna().all())
    out["p1_absmean_before"] = float(A.ret1d_clean.abs().mean())
    out["p1_absmean_after"] = float(B.ret1d_clean.abs().mean())
    m = B.date < "2010-10-04"
    out["p3_rows"] = int(m.sum())
    out["p3_mism"] = int((B.tick[m].values != tick_vec(B.close[m].values, B.date[m])).sum())
    out["p3_tickpct_med"] = (float(A.tick_pct[A.date < "2010-10-04"].median()), float(B.tick_pct[m].median()))

    def term(df, patched):
        g = df.groupby("code", sort=False)
        last = g.date.transform("max") < df.date.max() - pd.Timedelta(days=10)
        rk = g.cumcount(ascending=False)
        d = df[last & (rk <= 8)].copy()
        d["r"] = d.ret1d_clean.where(d.resume_day == 0, d.ret_multi) if patched else d.ret1d_clean
        gg = d.groupby("code")
        res = pd.DataFrame({"clean": gg.r.agg(lambda s: np.prod(1 + s.iloc[1:].fillna(0)) - 1), "true": gg.adj_close.agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)})
        return res[res["true"].abs() < 5]
    ta, tb = term(A, False), term(B, True)
    out["p2_term"] = {"n": len(ta), "true_mean": float(ta["true"].mean()), "before_mean": float(ta.clean.mean()), "after_mean": float(tb.clean.mean()),
                      "before_gap": float((ta.clean - ta["true"]).abs().mean()), "after_gap": float((tb.clean - tb["true"]).abs().mean())}
    out["p2_outliers"] = (int(A.ret_outlier.sum()), int(B.ret_outlier.sum()))
    fl = B[["indiv_flow", "inst_flow", "foreign_flow"]]
    out["p6_flow"] = {"max_abs": float(fl.abs().max().max()), "gt1": int((fl.abs() > 1 + 1e-9).sum().sum()), "std": float(B.indiv_flow.std()), "std_before": float(A.indiv_flow.std()),
                      "max_before": float(A[["indiv_flow", "inst_flow", "foreign_flow"]].abs().max().max())}
    # 패치 후에도 룩어헤드가 없는지: 절단 실험
    T = pd.Timestamp("2019-06-28")
    Bt = run_p01(pb, {k: v[v.date <= T].copy() for k, v in IN0.items()})
    cols = [c for c in DER if c in B.columns] + ["resume_day", "ret_multi", "liq_zone"]
    d, n = cmp_cols(Bt, B, cols, T)
    out["trunc"] = (n, sum(d.values()), {k: v for k, v in d.items() if v})
    R["patch_check"] = out


def report(P):
    log("보고서 조립")
    r = R
    M = r["match"]
    cov, pl = r["cov"], r["prelist_tbl"]
    ex = r["lim_exact"]
    lyr = r["lim_exact_year"]
    term = r["term"]
    ct = r["out_cause"]
    rt = r["resume_tab"]
    ipw = r["inv_period"]
    fu, fd = r["lim_fp_up"], r["lim_fp_dn"]
    pcx = r["patch_check"]
    i1, i2, ib = r["ipo_d1"], r["ipo_d2_5"], r["base"]
    L = []
    A = L.append
    n_rows = f0(r["n_rows"])
    n_der = len(r["rep_diff"])
    A(f"""---
title: 차트 CNN 분석 패널 독립 감사 (q01)
created: 2026-10-10
updated: 2026-10-10
tags: [dev/python, finance/퀀트]
source: PAPERS/26-2/03_차트CNN_연구/code/q01_panel_qa.py 가 자동 생성. 대상 panel.parquet, 생성 코드 p01_build_panel.py, 대조 자료 data/daily/dg·제도일정
status: seed
---

# 차트 CNN 분석 패널 독립 감사 (q01)

> 대상 `data/daily/club_db/panel.parquet`: {n_rows}행 x {r['n_cols']}열 (설명서에는 49열이라 적혀 있으나 실제 파일은 {r['n_cols']}열, `code` 열은 저장 때 버려짐), KOSPI {r['P_info'][0]} ~ {r['P_info'][1]}, {r['P_info'][2]}종목, 거래일 {f0(r['n_days'])}일.
> 이 문서는 `code/q01_panel_qa.py` 를 실행할 때마다 다시 만들어진다(약 3분). 기존 파일(p01, panel)은 수정하지 않았다. 고칠 점은 맨 아래 '패치 제안'에 코드로 적었고, 분석에서 바로 쓸 행 단위 플래그는 `results/q01_panel_flags.parquet` 에 있다(조인 키 `ticker`, `date`).

## 0. 결론

**원자료는 믿을 만하고, 가공 단계(p01)에 고칠 점이 있다.** 가격·거래량은 로컬 DataGuide와 사실상 같은 자료(종가 {pc(M['close'][0], 3)} 일치)이고, 거래대금·투자자 항등식·정렬·중복·종목코드·CNN 예측 조인도 문제가 없다. 룩어헤드는 실험으로 확인한 결과 파생 {n_der}열에 없다(미래를 통째로 바꿔도 과거 값 불일치 0건, 단 `_w` 3열은 예외). 아래 4건은 분석 결과를 바꿀 수 있으니 먼저 처리해야 한다.

1. 거래재개일의 여러 날 누적 수익이 단일일 수익으로 남아 있다({f0(r['resume_with_ret'])}행).
2. 정리매매·상폐 직전의 큰 손실이 `ret_outlier` 로 지워져 `ret1d_clean` 에서 빠진다(상폐 종목 생존편향).
3. 2010-01-04~2010-10-01 호가단위 `tick` 이 {f0(r['tick_a']['rows'])}행({pc(r['tick_a']['rows'] / r['n_rows'], 1)}) 전부 틀렸다.
4. 상장 전 K-OTC(장외) 호가 {f0(r['prelist_rows'])}행이 KOSPI 일봉으로 섞여 있다(5종목).

### 판정 요약

| 항목 | 판정 | 한 줄 근거 |
|---|---|---|
| 1 커버리지 | 정상 + 주의 | 상폐 {r['dl_match']}/{r['dl_n']}종목이 마지막 거래일까지 있음. K-OTC 사전 구간, 코스닥 이전 종목 혼입은 주의 |
| 2 가격 정합성 | 정상 + 주의 + 버그 | OHLC·거래량 100% 일치. 수정주가는 원 단위 반올림 잡음. `ret_outlier` 정의는 버그 |
| 3 거래정지 | 버그 | 재개일 누적 수익이 단일일 수익으로 남음 |
| 4 투자자 | 주의(중) | 항등식은 2012-05 이후만 성립, 결측을 0 으로 채운 행, 극단 정규화 |
| 5 거래대금 | 정상 | close x volume 과 {pc(r['tv_within10'], 2)} 가 +-10% 이내 (006740 한 종목만 이상) |
| 6 제약 열 | 버그 1(tick) + 주의 | 세율 정상, 공매도 일수 일치, 호가단위 2010-10 이전 오류, 상하한가 일부 오판 |
| 7 룩어헤드 | 정상 (예외 `_w`) | 교란 실험 불일치 0. `_w` 경계만 전체 표본 분위수 |
| 8 함정 | 정상 | 중복 0, 정렬 정상, CNN 예측 매칭 100%, 보통주만 있음 |

### 버그·주의 목록 (심각도 순)

| ID | 심각도 | 내용 | 규모 | 영향 |
|---|---|---|---|---|
| B1 | 높음 | 거래재개일 누적 수익이 `ret1d`·`ret1d_clean` 에 단일일 수익으로 남음 | {f0(r['resume_with_ret'])}행 (정지 5일 이상 후 재개 {f0(r['resume_long_n'])}행) | 일간 수익 분포 꼬리, `vol20`, 이벤트 연구 |
| B2 | 높음 | 정리매매 극단 손실이 이상값으로 지워짐 | 이상값 {r['out_n']}행 중 {int(ct[ct['원인'].str.startswith('정리')]['행'].sum())}행이 정리매매 | 상폐 종목 손실 누락, 미래수익 라벨로 쓰면 생존편향 |
| B3 | 높음 | 2010-10-04 이전 `tick`=1.0 일괄 채움 | {f0(r['tick_a']['rows'])}행 | `tick_pct` 가 실제의 1/{r['tick_a']['ratio_med']:.0f}(중앙값 기준) |
| B4 | 높음 | K-OTC 사전거래가 KOSPI 일봉으로 포함 | {f0(r['prelist_rows'])}행, 5종목 | 상장일 전 가짜 가격·IPO 연구 오염 |
| B5 | 중간 | 투자자 열의 정의·결측 문제 | 합계 비영 {pc(ipw.iloc[0]['합계≠0 비율'], 1)}(2010~2012-05), 전부 0 인 거래량 1만 이상 {f0(r['inv_zero_big_n'])}행 | 순매수 신호 노이즈, 2026-05-26 이후 외국인 정의 변경 추정 |
| B6 | 중간 | `*_flow` 정규화가 불안정, `_w` 는 전체 표본 경계 | 절대값 5 초과 {f0(r['flow_q'].loc['indiv_flow', '|x|>5'])}행, 20 초과 {f0(r['flow_gt20'])}행 | 회귀·상관의 레버리지 점, `_w` 약한 룩어헤드 |
| B7 | 중간 | 상하한가 판정 오류 | 상한 누락 {ex['up']['exact만(누락)']}·하한 누락 {ex['dn']['exact만(누락)']}, 재개일·권리락 오판정 수백 행 | 제약 열을 거래불가 조건으로 쓰면 편향 |
| B8 | 중간 | 공매도 열 불완전 | 부분허용 {r['short_partial'][0]}일({f0(r['short_partial'][1])}행)에서 `short_banned`=0, 2010~2013-11 금융주 금지 미반영 | 공매도 가능 여부를 종목 단위로 쓰면 틀림 |
| B9 | 중간 | `vol20` 이 미청소 `ret1d` 사용 | 재개일 포함 창 {f0(r['vol20']['resume_in_window'])}행 | 변동성 신호 오염 |
| B10 | 낮음 | 수정주가 원 단위 반올림 | adj_close<1000 {f0(r['adj_lt_n'][1000])}행({pc(r['adj_lt'][1000], 2)}), <100 {f0(r['adj_lt_n'][100])}행 | 저가 수정주가 종목 수익률 잡음, 극저가 이상값 |
| B11 | 낮음 | 006740 거래대금 단위 어긋남, 2026-06-23 이후 `listed_shares` 전부 NaN | 006740 {r['t006740'][1]}행(비율 {r['t006740'][2]:.2f}), mktcap NaN {f0(r['mktcap_nan'])}행 | 최근 시가총액 사용 불가 |
| B12 | 낮음 | 마지막 2~3일 자료 미확정 | 2026-10-06·07 순매수>거래량 {r['net_gt_vol_dates'].get(pd.Timestamp('2026-10-06'), 0)}·{r['net_gt_vol_dates'].get(pd.Timestamp('2026-10-07'), 0)}종목, 10-08 거래대금 전부 NaN | 최신 구간 제외 권장 |
| B13 | 낮음 | 기타 | OHLC 위반 {sum(r['ohlc'].values())}행, volume NaN {r['vol_nan']}행, 코드에 알파벳 포함 {len(r['tk_nondigit'])}종목, 코스닥 이전 {r['moved_in_panel']}종목 {f0(r['moved_rows'])}행 | 주의만 |

## 1. 커버리지

**방법.** 연도별로 패널 종목 수를 `stock_master` 의 상장 종목 수와 비교하고, 패널 종목별 마지막·첫 날짜를 `stock_master.last_date`·`first_date` 와 대조했다. 원본 price/investor/trading 과 패널 행 수도 맞췄다.

| 연도 | 패널(해당연도 1행 이상) | 마스터(해당연도 상장) | 패널(연말) | 마스터(연말) | 연말 차이 |
|---|---|---|---|---|---|
""" + "\n".join(f"| {y} | {a} | {b} | {c} | {d} | {e} |" for y, (a, b, c, d, e) in zip(cov.index, cov.values)) + f"""

- 패널 행 수 {n_rows} = price 원본 합계({f0(r['raw_rows'][0])})로 일치. 투자자·거래대금 원본에서 가격 행에 못 붙은 행은 {r['inv_not_in_price']}·{r['trd_not_in_price']}행(무시 가능). 패널에 없는 마스터 종목 {r['not_in_panel']}개는 모두 2010-01-04 이전에 상폐된 종목이다(2010 이후 생존 종목 누락 {r['not_in_panel_alive2010']}).
- 2010~2013 연말 차이 -2는 두 종목 때문이다. {r['first_mismatch'][0][0]} {r['first_mismatch'][0][1]}(마스터 first_date {r['first_mismatch'][0][2]}, 패널 첫 행 {r['first_mismatch'][0][3]}), {r['first_mismatch'][1][0]} {r['first_mismatch'][1][1]}(마스터 {r['first_mismatch'][1][2]}, 패널 {r['first_mismatch'][1][3]}). 마스터 first_date 는 DB 시작일(1979-12-24)인 경우가 많아 상장일이 아니므로 마스터 쪽 문제다.
- 상장폐지 {r['dl_n']}종목: 패널 마지막 날짜가 `stock_master.last_date` 와 {r['dl_match']}/{r['dl_n']} 일치(정리매매 구간까지 포함). 마지막 행의 거래량>0 비율 {pc(r['dl_last_vol_pos'], 1)}, 나머지는 거래정지 상태로 끝난다. 현재 거래 중 {r['active_n']}종목은 모두 {r['P_info'][1]} 까지 있다(마스터 last_date 는 2026-10-02 스냅샷이라 6일 늦다).
- 패널 달력 {f0(r['n_days'])}일은 DataGuide 달력과 겹치는 구간에서 양쪽 차집합이 모두 0일. 하루 종목 수 {r['per_day_min_max'][0]}~{r['per_day_min_max'][1]}.
- 신규상장(패널 첫 행이 2010-01-05 이후) 연도별: {', '.join(f'{y} {n}' for y, n in r['ipo_by_year'].items())}.
- 상장 직후(K-OTC 사전 구간 제외): 첫 거래일 {i1['rows']}행은 수익이 항상 NaN({pc(i1['ret_na'], 0)})이라 상장일 급등락은 어떤 열에도 없다. 2~5번째 거래일 {f0(i2['rows'])}행의 평균 |수익| {pc(i2['mean_abs'], 2)}(그 외 {pc(ib['mean_abs'], 2)}), 상한가 {i2['lim_up_bp']:.0f}bp(그 외 {ib['lim_up_bp']:.0f}bp), 이상값 {i2['outlier']}행. 신규상장 직후 며칠은 변동이 커서 표본 전체와 분포가 다르다.
- **주의 (B4)** 상장 전 장외 구간이 들어 있다.

""" + tbl(pl.rename(columns={"ticker": "종목", "rows": "상장 전 행", "name": "이름"}), idx=False) + f"""

  이 5종목은 투자자·거래대금 자료가 상장일(제주항공 2015-11-06, 삼성SDS 2014-11-14, 지누스 2019-10-30 등)에야 시작한다. 그 전 행은 거래량 중앙값이 {int(pl['거래량중앙값'].min())}~{int(pl['거래량중앙값'].max())}주 수준이고 가격이 하루 +-30% 로 움직인다(K-OTC 장외 호가로 추정). 지누스가 1,257행으로 가장 크다. `f_prelist_otc` 플래그로 제외하면 된다.
- 코스닥에서 코스피로 이전한 16종목 중 {r['moved_in_panel']}종목({f0(r['moved_rows'])}행)이 패널에 있다. 이전 전 코스닥 시절 가격이 섞여 있을 수 있으나 이전일 정보가 없어 가려낼 수 없다(영향 {pc(r['moved_rows'] / r['n_rows'], 2)}).

**판정:** 정상(커버리지·상폐 포함·달력), 주의(K-OTC 사전 구간, 코스닥 이전 종목). **권장:** P4 패치 또는 `f_prelist_otc==0` 필터.

## 2. 가격 정합성

**방법.** 로컬 DataGuide(`data/daily/dg`, 2010-01-04~2026-04-03 겹침 {f0(r['n_both'])}행)와 종목-일 단위로 맞췄다. 수정주가는 일간 수익률을 비교했다.

| 열 | 일치율 | 불일치 행 | 비교 행 |
|---|---|---|---|
""" + "\n".join(f"| {c} | {pc(v[0], 4)} | {v[1]} | {f0(v[2])} |" for c, v in M.items()) + f"""

- 겹침 구간에서 패널에만 있는 행 {r['panel_only_in_dg_window']}, DG 에만 있는 가격 행 {r['dg_only_price']}. 패널 가격은 DG와 같은 원천이라 독립 검증은 아니다(일치는 '변환 오류 없음'만 증명한다). DG `close.parquet` 는 2026-04-01 까지라 04-02·03 두 날짜는 종가 비교에서 뺐다. 2026-04-02 이후 약 {f0(r['n_rows'] - r['n_both'])}행은 대조할 수 없다.
- **수정주가.** 두 쪽 모두 원 단위 정수로 반올림된 값이다(DG 100% 정수, 패널 {pc(r['panel_adj_int_frac'], 2)} 정수). 수준이 같은 행은 {pc(r['adj_eq'], 1)}, 나머지는 {r['ratio_ne1_tickers']}종목이 일정 배율로 어긋난다(DG 를 받은 뒤 일어난 권리락·배당 조정 차이로 추정, 배율이 시점에 따라 바뀌는 행은 {r['ratio_jump_rows']}). 일간 수익률 차이 |dr|: 중앙값 {r['dr_q']['q0.5']:.0f}(90% 지점까지 0), 99% {r['dr_q']['q0.99']:.1e}, 99.9% {r['dr_q']['q0.999']:.1e}, 최대 {r['dr_q']['max']:.4f}. 1bp 초과 {f0(r['dr_n']['0.0001'])}행, 10bp 초과 {f0(r['dr_n']['0.001'])}행, 100bp 초과 {r['dr_n']['0.01']}행. 양쪽 반올림 허용오차(각 가격 +-0.5원)를 넘는 진짜 불일치는 **{r['dr_unexpl']}행({r['dr_unexpl_tickers']}종목)** 뿐이다.

  불일치 상위 20종목은 모두 같은 원인이다(수정주가가 작을수록 정수 반올림 잡음이 커지고, 일부는 DG 스냅샷과 수준이 다르다).

""" + tbl(r["dg_top20"][["ticker", "name", "불일치일", "반올림초과일", "최대차이", "패널adj중앙값", "dg대비배율", "원인"]], idx=False) + f"""

  허용오차를 넘은 {r['dr_unexpl']}행: {', '.join(f"{t} {d.date()}" for t, d in zip(r['dg_real'].ticker, r['dg_real'].date))}.
- **반올림이 패널에 주는 영향.** 일간 수익률 잡음의 상한은 약 1/adj_close 다. adj_close<1000 인 행이 {f0(r['adj_lt_n'][1000])}개({pc(r['adj_lt'][1000], 2)}), <500 {f0(r['adj_lt_n'][500])}개, <100 {f0(r['adj_lt_n'][100])}개. 전체 중앙 잡음은 {r['noise_bound_q'][0.5] * 1e4:.1f}bp, 99% 지점 {r['noise_bound_q'][0.99] * 1e4:.0f}bp 로 작지만, 대규모 액면분할·감자로 과거 수정주가가 크게 줄어든 종목에서는 수익률이 +-33%, +-50% 같은 이산 값으로 나온다.
- OHLC 내부 정합: 고가<저가 {r['ohlc']['high<low']}, 고가<종가 {r['ohlc']['high<close']}, 저가>종가 {r['ohlc']['low>close']}, 고가<시가 {r['ohlc']['high<open']}, 저가>시가 {r['ohlc']['low>open']}(원본 오류 {sum(r['ohlc'].values())}행, 무시 가능).
- 알려진 예외 004550(대우송도개발)은 패널에 {r['t004550'][0]}행, 수정계수(adj/close) {r['t004550'][1]:.3f}~{r['t004550'][2]:.1f}, ret_outlier {r['t004550'][3]}행(2012-06-13 정리매매 첫날 -95%). 014900 은 2007 상폐라 패널에 없다(패널 포함 여부 {r['t014900_in_panel']}). README 의 '거래정지 중 감자·분할 미보정'과 같은 유형은 아래 `ret_outlier` 분석에서 다시 보인다.

### ret_outlier {r['out_n']}행 ({r['out_tk']}종목)

`ret_outlier` 는 |수정주가 수익률| > 가격제한폭 + 10%p (2015-06-15 전 25%, 이후 40%) 인 행이다. 이상값 {r['out_n']}행의 원인(위에서부터 먼저 해당하는 것으로 분류):

""" + tbl(ct, idx=False) + f"""

- 부호: 하락 {r['out_neg']}행, 상승 {r['out_pos']}행. 연도별 {', '.join(f'{y} {n}' for y, n in r['out_year'].items())}. 2010~2012 에 {r['out_year'][2010] + r['out_year'][2011] + r['out_year'][2012]}행이 몰려 있다.
- **정리매매 구간이 이상값 대부분이다.** 상폐 직전 정리매매는 가격제한폭이 없다(market_rules_notes F3·F24). 이 수익은 오류가 아니라 실제 투자자 손실인데 `ret1d_clean` 에서 NaN 이 된다. 상폐 {term['dl_n']}종목 중 {term['n']}종목이 마지막 8거래일에 이상값을 갖고, 이들의 종단 8일 수익은 중앙값 실제 {pc(term['true_med'], 1)} 대 `ret1d_clean` 복리 {pc(term['clean_med'], 1)}, 평균(극단 제외 {term['n_ex']}종목) 실제 {pc(term['true_mean_ex'], 1)} 대 {pc(term['clean_mean_ex'], 1)} 다. 미래수익 라벨이나 성과 계산에 `ret1d_clean` 을 쓰면 상폐 손실이 사라진다(상폐 수익 편향).
- 극저가·수정주가 반올림 아티팩트는 008080(에스와이코퍼레이션: 2010~2012 adj_close 1~4원)과 000790(씨앤상선: 종가 10~115원)이 거의 전부다. 이 두 종목은 분석에서 빼는 편이 안전하다.
- 거래재개일 가격 재설정 의심(README 의 감자·분할 미보정 사례와 같은 유형)과 기타 행은 아래와 같다.

""" + tbl(r["out_review"].tail(8), idx=False) + """

**판정:** OHLC·거래량 정상, 수정주가는 주의(원 단위 반올림), `ret_outlier` 정의는 버그(B2). **권장:** P2.

## 3. 거래정지 처리

**방법.** `is_halted`, `volume==0`, 가격 0값의 교차표, 정지 직후 첫 거래일(재개일)의 수익률 분포를 정지 기간 길이별로 봤다.

""" + tbl(r["halt_ct"].astype(int), idx=True) + f"""

- 가격 0값: 시가 {r['price_zero']['open']}, 종가 {r['price_zero']['close']}, 수정종가 {r['price_zero']['adj_close']} (없음). 정지 행의 {pc(r['halt_flat'], 1)} 는 시고저종이 모두 같다(직전 종가 이월). `is_halted=0 & volume=0` {f0(r['v0nh_n'])}행도 OHLC 가 모두 같다({pc(r['v0nh_flat'], 0)}). `is_halted=1 & volume>0` 은 {r['halt_vol_pos']}행(장중 정지 등)이고 이 행의 수익률은 `ret1d` 에서 NaN 이 된다.
- p01 은 `is_halted==1 | volume==0` 인 날만 수익률을 비운다. 즉 **정지일은 NaN, 재개일은 정지 전 마지막 종가 대비 누적 수익이 그대로 남는다.** 가격은 정지 중 이월되므로 재개일 수익은 정지 기간 전체의 수익이다.

""" + tbl(rt.rename(columns={"재개일": "행", "평균절대수익": "평균|수익|", "중앙절대수익": "중앙값|수익|", "절대10퍼센트이상": "|수익|>10% 비율"}), idx=True) + f"""

- 재개일 {f0(r['resume_n'])}행 중 {f0(r['resume_with_ret'])}행이 `ret1d_clean` 에 값을 갖는다. 정상일 평균 |수익| {rt.loc['정상일(비교)', '평균절대수익'] * 100:.1f}% 에 비해 5~19일 정지 후 {rt.loc['5~19일', '평균절대수익'] * 100:.1f}%, 20일 이상 정지 후 {rt.loc['20일↑', '평균절대수익'] * 100:.1f}%. |수익|>10% 인 재개일 {r['resume_gt10_n']}행. 정지 20일 이상 후 재개일 중 {pc(rt.loc['20일↑', 'clean결측'], 0)} 만 이상값으로 지워지고 나머지는 단일일 수익처럼 남는다. 정지 1일(주로 거래량 0 하루)은 정상 분포에 가깝다.
- 정지일의 `*_flow` 는 NaN 이 아니라 0 이다(정지/거래량0 {f0(r['idle_n'])}행 중 flow==0 {f0(r['idle_flow_zero'])}행). 정지일 `tv_filled`·`hl_range`·`lim_*` 도 0 으로 남는다. 분석 유니버스는 `f_tradable`(is_halted==0 & volume>0, 전체의 {pc(r['tradable_share'], 1)}) 로 거르는 것이 안전하다.
- 재개일이 상·하한가로 판정된 행: 상한 {r['resume_lim'][0]}, 하한 {r['resume_lim'][1]} (누적 수익을 한도 도달로 오판).

**판정:** 버그(B1). 정지일 NaN 처리 자체는 맞다. **권장:** P1, 임시로는 `f_resume_day==1` 행의 수익을 제외.

## 4. 투자자 자료

**방법.** `indiv_buy-indiv_sell=indiv_net` 항등식, 개인+기관+외국인+기타법인 순매수 합계의 0 근접도, 정규화 열의 극단값을 직접 계산했다.

- 항등식: indiv 만 매수·매도 열이 모두 있어 확인 가능(inst·foreign 은 매도 열 없음). {f0(r['inv_identity'][0])}행 중 위반 {r['inv_identity'][1]}건 → 정상.
- 4그룹 합계는 거래 상대가 모두 이 4그룹이면 0 이어야 한다.

""" + tbl(ipw.rename(columns={"행": "행 수"}), idx=True) + f"""

  - 2012-05-03 ~ 2026-05-25: **합계가 정확히 0**({f0(ipw.iloc[1]['4열 모두 있음'])}행 전부). 따라서 이 구간의 `foreign_net` 은 기타외국인을 포함한다(포함하지 않으면 기타외국인 거래가 있는 날 합계가 0 이 될 수 없다).
  - 2010-01 ~ {str(r['inv_cut_date'].date())}: 합계 비영 {pc(ipw.iloc[0]['합계≠0 비율'], 1)}, 99% 잔차가 거래량의 {pc(ipw.iloc[0]['|합계|/거래량 99%'], 0)}. 당시 원천의 분류가 달라 일부 그룹(기타외국인·국가·지자체 등)이 빠진 것으로 추정한다(원천 문서는 확인 못함). 이 구간 순매수는 4그룹 합이 닫히지 않는다.
  - 2026-05-26 ~ 2026-08-06: 합계 비영 {pc(ipw.iloc[2]['합계≠0 비율'], 0)}(잔차 중앙값 거래량의 {pc(ipw.iloc[2]['|합계|/거래량 중앙값'], 2)}). AI_CONTEXT 가 말하는 키움 수급 전환 시점(2026-05-26)과 일치하고, 키움 자료는 `foreign_net` 과 `foreign_etc`(기타외국인)를 따로 두므로 **이 구간 `foreign_net` 은 기타외국인을 제외한 것으로 추정**한다. 경계에서 외국인 정의가 바뀐다.
  - 2026-08-07 이후: `other_corp_net`·`*_buy`·`*_sell` 전부 NaN(순매수 3개만). 4그룹 합은 계산 불가.
- **결측을 0 으로 채운 행.** 개인·기관·외국인·기타법인이 모두 정확히 0 인데 거래량이 있는 행이 {f0(r['inv_zero_n'])}개, 그중 거래량 1만 주 이상이 {f0(r['inv_zero_big_n'])}개다. 2010~2014 에 몰려 있고(거래량 1만 이상: 2010 {int(r['inv_zero_year'].iloc[0, 2]):,}, 2011 {int(r['inv_zero_year'].iloc[1, 2]):,}, 2015 {int(r['inv_zero_year'].loc[2015].iloc[2])}, 2016 {int(r['inv_zero_year'].loc[2016].iloc[2])}, 2022 이후 0~1) 거래량이 큰데 우연히 모두 0 일 가능성은 없다 → 결측이 0 으로 코딩된 것으로 판단한다.

""" + tbl(r["inv_zero_year"], idx=True) + f"""

- 투자자 자료가 아예 NaN 인 행(2026 이전) {f0(r['inv_na_pre26'])}행 중 거래대금도 같이 NaN 인 행 {f0(r['inv_na_tv_same'])}, K-OTC 사전 구간 {f0(r['inv_na_prelist'])}, 000790 {r['inv_na_other_top'].get('000790', 0)}행. 원본 investor 파일에 그 행이 없는 것이다.
- 순매수 절대값이 거래량을 넘는 행: 개인 {r['net_gt_vol']['indiv_net']}, 기관 {r['net_gt_vol']['inst_net']}, 외국인 {r['net_gt_vol']['foreign_net']} (모두 {r['net_gt_vol_firstlast'][0]}~{r['net_gt_vol_firstlast'][1]} 사이이며 2026-10-06·07 에 집중). 최신 2일의 순매수·거래량이 어긋난다(미확정 또는 일자 불일치 추정).

### 순매수 정규화 열 (`*_flow`)

""" + tbl(r["flow_q"][["min", "q0.1%", "q99.9%", "max", "|x|>1", "|x|>5", "|x|>20", "표준편차", "첨도"]], idx=True, nd=4) + f"""

- 극단값 원인 (|indiv_flow|>5 인 {f0(r['flow_ex5']['n'])}행): 직전 60일 평균 거래량 1,000주 미만 {r['flow_ex5']['avgvol<1000']}행, 1만주 미만 {f0(r['flow_ex5']['avgvol<1만'])}행({pc(r['flow_ex5']['avgvol<1만'] / r['flow_ex5']['n'], 0)}), 60일 창에 거래량 0 일(정지)이 있는 행 {r['flow_ex5']['창에 거래량0일 포함']}, 조정계수 단차(20% 이상, {r['split_events'][0]}건/{r['split_events'][1]}종목) 후 60거래일 안 {r['split_big_share'][0]}행. 즉 대부분은 거래가 거의 없던 종목에 갑자기 거래가 몰린 날(분모가 작음)이다.
- 사례(상위):

""" + tbl(r["flow_cases"], idx=False) + f"""

  삼성전자는 2018-05-04 액면분할(50:1)일에 분모(분할 전 주식수 기준 평균 {r['samsung'][2]:,.0f}주)와 당일 거래량({r['samsung'][1]:,.0f}주)의 단위가 달라 `indiv_flow` 가 {r['samsung'][0]:.1f} 이다. 거래량은 분할 보정이 안 된 원 거래량이다.
- `indiv_flow_tv`(거래대금 기준)도 극단값 비율이 비슷하다(|x|>5: {r['flow_q'].loc['indiv_flow_tv', '|x|>5']:.0f}행). 분할에는 강하지만 분모가 작은 문제는 그대로다.
- **윈저화 `_w`.** 코드 주석은 '1%/99%'라고 쓰지만 실제는 전체 표본 **0.5%/99.5%** 분위수다(재현 확인 {all(bool(x) for x in r['winsor']['0.5%/99.5% 재현'])}).

""" + tbl(r["winsor"][["하한", "상한", "표준편차(원)", "표준편차(_w)", "첨도(_w)", "≤2014 경계 하", "≤2014 경계 상"]], idx=True) + f"""

  클리핑 뒤 표준편차가 0.24(개인)로 줄고 첨도는 8.9 로 꼬리가 크게 줄어 회귀 입력으로는 쓸 만하다. 잘리는 비율은 연도별 0.5~1.3% 로 안정적이다. 그러나 (1) 경계가 2010~2026 전체 표본에서 한 번 계산되어 **미래 데이터에 의존한다**(2014 이전 표본만 쓰면 개인 경계가 -1.21/+1.47 로 달라진다. 교란 실험에서 2019-06-28 이후 값을 바꾸자 그 이전 `_w` 가 {r['w_future_dep'][1]['indiv_flow_w']}·{r['w_future_dep'][1]['inst_flow_w']}·{r['w_future_dep'][1]['foreign_flow_w']}행 바뀌었다), (2) 분할일·정지 직후 같은 구조적 이상값은 잘라낼 뿐 원인을 고치지 않는다.
- 패치 효과(P6: 분할 보정 분모, 거래량 0 일 제외, 분모=max(평균, 당일 거래량)): |flow|>1 인 행 {r['patch_flow']['|x|>1']}개(순매수>거래량으로 어긋난 최신 2일 등), 최댓값 {r['patch_flow']['max']:.2f}, 표준편차 {r['patch_flow']['std']:.3f}, 첨도 {r['patch_flow']['kurt']:.1f}.

**판정:** 항등식 정상 / 구간별 정의 변경 주의 / 결측 0 코딩 주의 / 정규화는 주의(B5·B6). **권장:** 2010~2012-05 순매수는 신호로 쓰지 말고, 쓸 때는 `f_zero_inv_suspect==0` 로 거르고, 횡단면 순위 변환을 쓴다.

## 5. 거래대금

**방법.** `trading_value / (close x volume)` 분포(거래대금은 체결가 가중이라 정상 비율은 1 안팎).

| 분위 | 0.1% | 1% | 5% | 50% | 95% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| 비율 | {' | '.join(f'{v:.3f}' for v in r['tv_ratio_q'].values())} |

- {pc(r['tv_within10'], 2)} 가 +-10% 이내({f0(r['tv_n'])}행). 비율 <0.7 {r['tv_tails']['<0.7']}행, >1.3 {r['tv_tails']['>1.3']}행, >2 {r['tv_tails']['>2']}행. 연도별 중앙값 {r['tv_year']['중앙값'].min():.4f}~{r['tv_year']['중앙값'].max():.4f}, |비율-1|>20% 비율 최대 {pc(r['tv_year']['|비율-1|>20%'].max(), 3)}.
- 불일치 {r['tv_bad_n'][0]}행·{r['tv_bad_n'][1]}종목 중 **{r['t006740'][1]}행이 006740 블루산업개발**이다. {r['t006740'][0]} 거래재개 이후 비율이 {r['t006740'][2]:.2f}(약 1/5)로, 종가가 액면분할 전 가격 기준으로 남고 거래량·거래대금은 분할 후 값인 것으로 보인다(거래대금/거래량이 종가의 1/5). 이 종목은 2026-06-26 이후를 쓰지 말 것. 나머지는 정리매매 극단 가격일(종목당 1~2행)이다.
- NaN {f0(r['tv_nan'])}행: 투자자 결측 {f0(r['inv_na_tv_same'])}행과 같은 행(K-OTC 등)과 마지막 날 {r['tv_nan_last']}행(trading 원본이 10-07 까지). `tv_filled` 는 이때 close x volume 으로 채우며(`tv_is_approx`=1) 비율이 0.999 라 오차는 작다. `trading_value==0` 이면서 거래량>0 {r['tv0_vol_pos']}행, 거래대금>0 이면서 거래량 0 {r['tv_pos_vol0']}행으로 둘은 일관된다.
- 상장주식수: `listed_shares` 는 {r['ls_nan'][1]} 이후 전부 NaN({f0(r['ls_nan'][0])}행, AI_CONTEXT 가 말하는 DB 공백). 같은 기간 `mktcap` 도 NaN. 종목의 {pc(1 - r['ls_const_share'], 0)} 는 상장주식수가 시점별로 바뀌어 있어 최신 값을 과거에 채운 것이 아니다(룩어헤드 아님).

**판정:** 정상(006740 만 주의). **권장:** 006740 제외, 2026-06-23 이후 시가총액은 financials.daily_trading 으로 보충.

## 6. 제약 열

### 6-1. 상·하한가 `lim_up`·`lim_dn`

**방법.** (a) 연도별 빈도, (b) 호가단위까지 포함해 한도가를 직접 계산한 정확 판정(KRX 산식: 기준가 x 한도율을 기준가 호가단위로 절사, 가감 뒤 결과 가격대 호가단위로 절사/올림)과 패널 판정 비교, (c) 오판 유형 집계. 비교 대상은 직전 행이 정상 거래이고 조정계수 단차가 1% 미만인 행({f0(ex['elig_rows'])}행)이다.

""" + tbl(r["lim_year"].astype({"행": int, "상한가": int, "하한가": int}), idx=True, nd=3) + f"""

- 한도율 열: 15%/30% 경계 2015-06-15 와 정확히 일치({r['limit_boundary']}). 상한가 빈도는 +-15% 시기(2010~2015H1)에 연 43~85bp 였다가 +-30% 이후(2016~) 연 7~26bp 로 떨어진다. 하한가는 2010~2012 에 상한가의 1/3~1/4, 2016 이후는 연 0.5~2.6bp 다. 한도 확대 후 빈도가 1/4~1/8 로 줄어드는 것은 합리적이다.

| 정확 판정 기준 | exact 건수 | 패널 판정(대상 행) | 둘 다 | 누락(exact만) | 과검출(패널만) | 재현율 | 정밀도(전체 표시 대비) |
|---|---|---|---|---|---|---|---|
| 상한가 | {ex['up']['exact']} | {ex['up']['flag(elig)']} | {ex['up']['둘다']} | {ex['up']['exact만(누락)']} | {ex['up']['flag만(과검출)']} | {pc(ex['up']['둘다'] / ex['up']['exact'], 1)} | {pc(ex['up']['둘다'] / fu['flag'], 1)} |
| 하한가 | {ex['dn']['exact']} | {ex['dn']['flag(elig)']} | {ex['dn']['둘다']} | {ex['dn']['exact만(누락)']} | {ex['dn']['flag만(과검출)']} | {pc(ex['dn']['둘다'] / ex['dn']['exact'], 1)} | {pc(ex['dn']['둘다'] / fd['flag'], 1)} |

(정확 판정 코드의 검증: 패널이 한도로 표시한 대상 행의 {pc(ex['up']['둘다'] / ex['up']['flag(elig)'], 0)}(상한)·{pc(ex['dn']['둘다'] / ex['dn']['flag(elig)'], 0)}(하한)가 정확 판정과도 일치한다.)

""" + tbl(lyr, idx=True) + f"""

- 누락은 **2010년에 집중**(상한 {lyr.loc[2010, '상한 누락']}/{lyr.loc[2010, '상한 exact']}, 하한 {lyr.loc[2010, '하한 누락']}/{lyr.loc[2010, '하한 exact']}): 2010-10-04 전에는 5,000원 미만 호가단위가 5원이라 저가주의 한도가 15% 에 못 미치게 절사되어 패널의 임계값(한도-0.4%p)을 못 넘는다. 이후 연도는 상한 누락이 0.4~6.5%(2011~2015 는 2.5~4.7%), 하한은 2011~2015 에 3~12% 이고 2016 이후는 표본이 작아 0~33% 로 흔들린다. 한도에 가깝지만 임계값 아래인 행(한도-3%p ~ -0.4%p)도 {f0(r['lim_near_below'])}행 있다.
- 오판정 유형 (패널이 한도로 표시한 행 중):

| 유형 | 상한가 | 하한가 |
|---|---|---|
""" + "\n".join(f"| {'한도로 표시된 전체' if k == 'flag' else k} | {fu[k]} | {fd[k]} |" for k in fu) + f"""

  액면분할·권리락 같은 원주가 단절(`corp_action` 필터는 한도+5%p 를 넘는 하락/상승만 걸러냄)을 한도 도달로 오판한 사례는 조정계수 단차 5% 이상 기준 상한 {fu['계수 단차 5%↑']}, 하한 {fd['계수 단차 5%↑']}행이다. 예: 000640 2013-04-08 종가 -15.7%(수정주가 +14.8%), 000050 2012-12-27 종가 -15.3%(수정주가 +1.4%). 반대로 수정주가로는 한도 도달인데 원주가 판정이 놓친 행 {r['lim_miss_corp']}. 재개일 판정({fu['재개일']}·{fd['재개일']}행)은 여러 날 누적 수익이고, 정리매매 구간(상한 {fu['정리매매 구간']}·하한 {fd['정리매매 구간']}행)은 가격제한폭이 없어 판정 자체가 무의미하다.

""" + tbl(r["lim_fp_examples"], idx=False) + f"""

**판정:** 주의/버그(B7). 하한가 판정은 정밀도·재현율이 약 84%, 상한가는 약 97%·95%. 제약을 '거래불가' 조건으로 쓸 때는 정확 판정을 쓰거나 한도 근접 허용폭을 넓힌다. **권장:** P7.

### 6-2. 공매도 `short_regime`·`short_banned`

""" + tbl(r["short_tbl"], idx=True) + f"""

- 일수는 `short_selling_regime.csv`(2015 이후)와 하드코딩된 2011-08-10~11-09 금지를 합친 기대값과 **모두 일치**한다(2015 이전 전면허용 포함).
- **한계 1.** 2021-05-03 ~ 2023-11-03 `부분허용`({r['short_partial'][0]}일, {f0(r['short_partial'][1])}행)은 KOSPI200+KOSDAQ150 구성종목만 공매도 허용이다. 그런데 `short_banned` 는 이 구간에서 전부 {r['short_banned_in_partial']}(=금지 아님)이라, KOSPI200 이 아닌 약 600종목은 실제로는 금지였는데 0 으로 표시된다. 종목 단위 구분이 필요하며 패널에는 KOSPI200 구성 자료가 없다(시가총액 상위 200 으로 근사하는 방법은 있다).
- **한계 2.** 2010-01-04 ~ 2013-11-13 은 '전면허용'이지만 금융주 공매도는 2013-11-14 에야 해제됐다(csv 의 scope 설명, short_selling_notes 2013-11-13 보도). 해당 {r['short_fin_rows'][0]}일 {f0(r['short_fin_rows'][1])}행에서 금융업 종목의 공매도 가능 여부가 틀린다. 2011 금지 구간은 CSV 가 아닌 코드 하드코딩이다.

### 6-3. 거래세 `sell_tax_pct`

- 독립적으로 손으로 옮긴 연도별 세율표(0.30 / 2019-06-03 0.25 / 2021 0.23 / 2023 0.20 / 2024 0.18 / 2025 0.15 / 2026 0.20)와 전체 {n_rows}행을 비교한 불일치 {r['tax_mism']}건, `sell_tax_assumed` 는 2015 이전과 정확히 일치({r['tax_assumed_ok']}). 샘플 수기 대조:

""" + tbl(r["tax_samples"], idx=False) + """

- 2015 이전 0.30 은 문서로 확인되지 않은 가정이다(`sell_tax_assumed=1`). 거래세는 결제일 기준이라 세율 변경일 앞뒤 약 2영업일은 체결일 기준과 다를 수 있다(cost_notes F7).

### 6-4. 호가단위 `tick`·`tick_pct`

독립적으로 손으로 옮긴 호가표(2010-10-04 이전, 2010-10-04~2023-01-24, 2023-01-25~)와 전체 행을 비교했다.

""" + tbl(r["tick_era"], idx=True) + f"""

- 2010-10-04 이후는 **불일치 0건**(시점별 매핑 정상). 2010-01-04 ~ 2010-10-01 은 {f0(r['tick_a']['rows'])}행이 전부 틀렸다. p01 은 이 구간에 대응하는 표가 없어 `fillna(1.0)` 으로 채운다. 실제 호가단위는 시장 안내문(market_rules_notes F12, 2차 자료)에 따르면 5,000원 미만 5원, 5천~1만 10원, 1만~5만 50원 등이다. `tick_pct` 중앙값이 {r['tick_a']['tick_pct_med_panel']:.4f}% 로 나오지만 실제는 {r['tick_a']['tick_pct_med_true']:.3f}% 다(약 {r['tick_a']['ratio_med']:.0f}배). 같은 해 10월 이후 중앙값은 {r['tick_a']['tick_pct_med_after']:.3f}% 라서 연초~9월만 비정상적으로 작다. 이 구간의 `tick_pct` 를 스프레드 하한 근사로 쓰면 거래비용을 2010 년 초에 과소 추정한다.
- 샘플 수기 대조:

""" + tbl(r["tick_samples"], idx=False) + f"""

**판정:** 세율·공매도 일수·2010-10-04 이후 호가단위는 정상, 2010-10-04 이전 호가단위는 버그(B3), 공매도 종목 단위는 주의(B8). **권장:** P3.

## 7. 룩어헤드 감사

### 7-1. 코드 판독 (열별)

| 열 | 정의(코드) | 사용 정보 | 판정 |
|---|---|---|---|
| ret1d | 수정주가 `pct_change()`, 정지·거래량0 NaN | t-1, t 종가 | 정상 (재개일 누적은 B1) |
| ret_outlier, ret1d_clean | 수익률 크기가 한도+10%p 초과면 표시·NaN | 당일 수익률 크기 | 시간 누수 없음. 단 크기로 거르므로 라벨로는 선택편향(B2) |
| mktcap | close x listed_shares | 당일 값, 상장주식수는 시점별 값 | 정상 |
| tv_filled | 당일 거래대금, 결측은 close x volume | 당일 | 정상 |
| indiv/inst/foreign_flow | 순매수(t) / 거래량 평균(t-60..t-1, `shift(1)`) | 순매수(t)는 장 마감 후 공표 | 정상. 단 t 종가 매매에는 못 쓰고 t+1 부터 |
| *_flow_w | 위 값을 전체 표본 0.5%/99.5% 로 클립 | 경계가 미래 포함 | **약한 룩어헤드** |
| indiv_flow_tv | 순매수 x close(t) / 거래대금 평균(t-60..t-1) | 당일 + 과거 | 정상 |
| str5, str20 | 수정주가 `pct_change(5/20)`, 창 안 이상값이면 NaN | t-5/t-20 ~ t | 정상 |
| mom12_1 | adj(t-21) / adj(t-252) - 1, 창 이상값이면 NaN | t-252 ~ t-21 | 정상 (최근 1개월 건너뜀) |
| near52h | adj(t) / max(adj, 최근 252행 t 포함, 최소 120행) | t-251 ~ t | 정상. 상한 1.0 |
| vol20 | `ret1d` 20행 표준편차(최소 15) | t-19 ~ t | 정상 (미청소 수익 사용 B9) |
| abvol | 거래대금 5행 평균(t 포함) / 평균(t-60..t-1) | 당일 + 과거 | 정상 |
| hl_range | (high-low)/close | 당일 | 정상 |
| lim_up, lim_dn, limit_pct, tick, tick_pct | 당일 close, 전일 close, 달력 | 당일·전일 | 정상 |
| short_regime, short_banned, sell_tax_pct | 달력(제도 시행일) | 사전 공표된 제도 | 정상 |

- **수정주가 수준은 사후 조정 시계열이다.** 과거 수정주가의 *수준*은 이후의 액면분할·배당 이벤트에 따라 달라진다(DataGuide 스냅샷에 따라 {r['ratio_ne1_tickers']}종목에서 수준이 어긋나는 것으로 확인). 수익률·비율 열(ret, str, mom, near52h)은 같은 시계열의 비율이라 영향이 없지만, `adj_close` 의 *절대 수준*으로 걸러내는 조건(예: adj_close>2000)은 룩어헤드이므로 `close` 를 써야 한다.
- 창 정의 독립 재계산: `mom12_1` {f0(r['mom_check'][0])}행 중 불일치 {r['mom_check'][1]}, `near52h` {f0(r['near_check'][0])}행 중 불일치 {r['near_check'][1]}(최댓값 {r['near_check'][2]:.1f}, 1 초과 {r['near_check'][3]}행), `avgvol60`: 무작위 {r['avgvol_check'][0]}행을 행 슬라이스로 직접 평균해 '당일 제외' 정의와 {r['avgvol_check'][1]}/{r['avgvol_check'][0]} 일치, '당일 포함' 정의와는 {r['avgvol_check'][2]}/{r['avgvol_check'][0]} 일치 → `shift(1)` 확인.
- `vol20` 은 미청소 `ret1d` 의 20행 표준편차다. 창 안에 이상값이 있는 행 {r['vol20']['outlier_in_window']}행, 재개일 누적 수익이 들어간 행 {f0(r['vol20']['resume_in_window'])}행({pc(r['vol20']['resume_in_window'] / r['vol20']['n'], 1)}), vol20>0.15 인 {f0(r['vol20']['gt0.15'])}행 중 이상값 포함 {r['vol20']['gt0.15_with_outlier']}행. 시간 누수는 없고 오염 문제(B9)다.

### 7-2. 실험 (p01 소스를 그대로 실행)

p01 소스에서 입력 읽기 3줄만 바꾸고 저장 줄 앞에서 잘라 `exec` 한다. 표본: 무작위 110종목 + 특수 종목(삼성전자, 000790, 008080, 지누스, 004550, 089590, 006740, 001140, 003560, 000030), 합 {r['sub_n']}종목 {f0(r['rep_rows'])}행.

| 실험 | 비교 행 | 파생 열 | 불일치 합계 | 결과 |
|---|---|---|---|---|
| 1. 저장 패널 재현 (p01 exec 결과 vs panel.parquet) | {f0(r['rep_rows'])} | {n_der} | {sum(r['rep_diff'].values())} | 코드가 패널을 정확히 만든다(`_w` 는 저장 경계로 클립 재현 불일치 {sum(r['w_clip_repro_diff'].values())}) |
| 2. 절단: 입력을 {r['trunc'][0]} 까지만 주고 다시 계산, 그 이전 값 비교 | {f0(r['trunc'][1])} | {len(r['trunc'][2])} | {sum(r['trunc'][2].values())} | 미래를 없애도 과거 값 동일 |
| 3. 교란: {r['pert'][0]} 이후 가격·거래량·상장주식수·투자자·거래대금·정지 플래그를 무작위로 바꾸고 그 이전 값 비교 | {f0(r['pert'][1])} | {len(r['pert'][2])} | {sum(r['pert'][2].values())} | 미래를 바꿔도 과거 값 동일 |
| 4. 한 종목(005930) 미래만 교란, 나머지 {r['sub_n'] - 1}종목 전 기간 비교 | {f0(r['pert_one'][0])} | {len(r['pert_one'][1])} | {sum(r['pert_one'][1].values())} | 종목 간 누수 없음 |
| 5. 양성 대조: 일부러 미래를 쓰는 열(5일 선행수익, 중심 이동평균) | - | 2 | {sum(r['ctrl'].values())} | 같은 방법이 이 열들은 잡는다(검정력 확인) |

- 교란이 실제로 작동했는지: 교란 후 {r['pert'][0]} 이후 구간에서는 `ret1d` {f0(r['pert_after_changed']['ret1d'])}행, `indiv_flow` {f0(r['pert_after_changed']['indiv_flow'])}행, `mom12_1` {f0(r['pert_after_changed']['mom12_1'])}행 등이 바뀐다(`hl_range`·`tv_is_approx`·달력 열은 교란 방식상 값이 바뀌지 않아 이 실험으로는 시험되지 않지만 코드상 당일 값만 쓴다).
- **예외 `_w`**: 부분집합 재계산에서 미래 교란이 과거 `_w` 를 {r['w_future_dep'][1]['indiv_flow_w']}·{r['w_future_dep'][1]['inst_flow_w']}·{r['w_future_dep'][1]['foreign_flow_w']}행 바꾼다(전체 표본 분위수). 패널 전체에서는 경계가 2010~2026 표본의 값이다. **권장:** 날짜별 횡단면 순위 `groupby('date').rank(pct=True)` 로 대체(P6 주석).
- 시점(timing) 주의: `*_flow`·`trading_value`·종가는 모두 t 일 장 마감 후에야 확정된다. 't 신호 → t 종가 체결'은 불가능하고 't 신호 → t+1 시가 이후 체결'만 가능하다.

**판정:** 정상(파생 {n_der}열), `_w` 3열만 약한 룩어헤드.

## 8. 분석에서 흔한 함정

- 중복 (ticker,date): {r['dup_keys']}건. 정렬: ticker, date 오름차순 {r['sorted_ok']}. 종목코드는 전부 6자 문자열({f0(r['tk_len'][6])}행); 숫자만이 아닌 코드 {r['tk_nondigit']} (신규 코드 체계, `int()` 변환 금지).
- **우선주.** 패널 {r['pref']['all_tickers']}종목 중 끝자리가 0 이 아닌 종목은 {r['pref']['tickers']}개다. 이 패널은 보통주(DataGuide `listed_shares_common`)만 담고 있어 우선주 구분 열이 필요 없고, 이름이 '우' 로 끝나는 보통주 코드도 없다({len(r['pref_name_sus'])}건). CNN 쪽 `endswith("0")` 필터와 같은 유니버스다.
- **CNN 예측과 조인** (`results/kaggle/pred_*.parquet`, 키 `(date, ticker)`):

""" + tbl(r["pred_tbl"], idx=False) + f"""

  `pred_d_lr1e-4` 는 {f0(r['pred_tbl'].iloc[0]['예측 행'])}행이 전부 패널에 붙는다(매칭률 {pc(r['pred_tbl'].iloc[0]['매칭률'], 1)}). 미매칭 {r['pred_unmatched']}건이라 미매칭 사유(종목 없음 {r['pred_unmatched_reasons']['ticker 가 패널에 없음']}, 날짜 없음 {r['pred_unmatched_reasons']['날짜가 패널 달력에 없음']})도 없다. 예측 ticker 는 {r['pred_dtype'][1]}/{f0(r['pred_tbl'].iloc[0]['예측 행'])} 가 6자리이고 끝자리 0 이 아닌 값 {r['pred_dtype'][2]}. 형성일은 패널 달력에 있는 날 {pc(r['pred_form_in_cal'], 1)}, 그중 그 주의 마지막 거래일 {pc(r['pred_weekend_ok'], 1)}. 형성일에 패널상 거래량 0·정지·K-OTC 사전구간인 행은 {r['pred_vol0']}건(CNN이 창 안 정지일을 이미 뺐다).
- CNN 라벨 `ret`(DataGuide 수정주가 선행 5일)과 패널의 `adj_close.shift(-5)/adj_close-1`: 비교 {f0(r['pred_ret_cmp']['비교 행'])}행, 상관 {r['pred_ret_cmp']['상관']:.6f}, 차이 중앙값 {r['pred_ret_cmp']['중앙값']:.0e}, 99% {r['pred_ret_cmp']['99%']:.1e}, 최대 {r['pred_ret_cmp']['max']:.4f}. 같은 원천이라 사실상 동일하다(1%p 초과 {r['pred_ret_cmp']['|차|>1%p']}행). 단 CNN 라벨은 5일 안에 상폐되는 표본을 버리므로(images.py 주석 '생존편향') 상폐 직전 손실이 없다는 점은 B2 와 같은 문제다.
- 정지·거래량 0 행, 극저가(수정주가<=100) 행, 코스닥 이전 종목의 구 코스닥 구간은 `f_tradable`·가격 필터로 거를 것.

**판정:** 정상.

## 9. 사용 지침 (분석자용)

1. 유니버스는 `f_tradable==1 & f_prelist_otc==0` 로 하고 최신 3거래일(2026-10-06 이후)과 008080·000790·006740(2026-06-26~)은 뺀다.
2. 일간수익은 특징용이면 `ret1d_clean`, **라벨(미래수익)은 `adj_close` 로 직접 계산해 상폐 종목의 정리매매 손실을 포함**하며, `f_resume_day==1` 행의 일간수익은 단일일 수익이 아니니 제외한다.
3. 신호는 t 종가까지의 정보이고 순매수·거래대금은 장 마감 후 확정이니 체결은 t+1 시가 이후로 하고, 가격 수준 필터는 `adj_close` 가 아니라 `close` 로 한다.
4. 순매수: 2010-01~2012-05-02 는 4그룹 합이 닫히지 않아 신호에서 빼고, `f_zero_inv_suspect==1` 은 결측으로 보며, 2026-05-26 이후 `foreign_net` 은 정의가 달라졌다.
5. `*_flow` 원본은 쓰지 말고 `groupby('date').rank(pct=True)` 같은 날짜별 횡단면 순위로 바꾼다(`_w` 의 클립 경계는 미래 표본을 쓴다).
6. `tick`·`tick_pct` 는 2010-10-04 이후만 쓰고(`tick_fixed` 로 대체 가능), 2015 이전 `sell_tax_pct`(0.30)는 가정값(`sell_tax_assumed`)으로 취급한다.
7. `lim_up`·`lim_dn` 은 근사다(하한 정밀도·재현율 약 84%, 재개일·권리락일 오판). 거래 가능 여부는 `f_tradable` 과 한도 근접(수익률 한도-1%p 이상) 보수적 차단으로 본다.
8. `short_banned` 는 2021-05-03~2023-11-03 에 종목 단위가 아니고(KOSPI200 외 금지) 2013-11-13 이전 금융주 금지도 빠졌으니 공매도 분석은 이 구간을 따로 다루거나 제외한다.
9. `vol20` 은 재개일·이상값이 섞였으니 `f_resume_day` 를 제외하고 다시 계산하며, `mktcap` 은 2026-06-23 이후 NaN 이다.
10. 유니버스는 보통주뿐이고(알파벳 포함 코드 2개 있음), 패널을 다시 만들면 `python q01_panel_qa.py` 를 다시 돌려 B1~B4 가 사라졌는지 확인한다.

## 10. 패치 제안 (p01 에 적용할 코드, 이번 감사에서는 적용하지 않음)

""" + PATCH + f"""

### 패치 검증 (p01 소스에 P1~P6 을 끼워 넣어 {r['sub_n']}종목에서 실제로 실행)

| 패치 | 확인 | 결과 |
|---|---|---|
| P1 | 재개일 수(패치) vs 감사 기준 | {pcx['p1_resume'][0]} vs {pcx['p1_resume'][1]} (재개일 `ret1d` 전부 NaN: {pcx['p1_ret_na']}) |
| P1 | `ret1d_clean` 평균 절대값 | {pcx['p1_absmean_before']:.4f} → {pcx['p1_absmean_after']:.4f} |
| P2 | `ret_outlier` 행 수 | {pcx['p2_outliers'][0]} → {pcx['p2_outliers'][1]} |
| P2 | 상폐 종단 8일 수익 평균({pcx['p2_term']['n']}종목): 실제 / 기존 clean 복리 / 패치(재개일은 `ret_multi` 로 보충) | {pc(pcx['p2_term']['true_mean'], 1)} / {pc(pcx['p2_term']['before_mean'], 1)} / {pc(pcx['p2_term']['after_mean'], 1)} (실제와의 평균 오차 {pc(pcx['p2_term']['before_gap'], 1)} → {pc(pcx['p2_term']['after_gap'], 1)}) |
| P3 | 2010-10-04 이전 {f0(pcx['p3_rows'])}행의 `tick` 이 독립 호가표와 불일치 | {pcx['p3_mism']}건, `tick_pct` 중앙값 {pcx['p3_tickpct_med'][0]:.4f}% → {pcx['p3_tickpct_med'][1]:.3f}% |
| P4 | 제거된 행 vs 감사 기준 K-OTC 행 | {pcx['p4_removed'][0]} vs {pcx['p4_removed'][1]} |
| P5+P6 | `*_flow` 절대값 최댓값 / 1 초과 개수 / `indiv_flow` 표준편차 | {pcx['p6_flow']['max_before']:.0f} → {pcx['p6_flow']['max_abs']:.2f} / {pcx['p6_flow']['gt1']}개 / {pcx['p6_flow']['std_before']:.3f} → {pcx['p6_flow']['std']:.3f} |
| 전체 | 패치본의 절단 실험(2019-06-28 이전 값 비교) | 비교 {f0(pcx['trunc'][0])}행에서 불일치 {pcx['trunc'][1]}건 {pcx['trunc'][2] or ''} |

(P7 은 코드 조각이 아니라 판정 규칙이라 s6 의 정확 판정으로 대신 확인했다. P1 은 정지 직후 수익을 `ret_multi` 로 옮기므로 라벨 계산에는 `adj_close` 를 직접 쓰는 지침 2번이 그대로 필요하다.)

## 11. 재현·부록

- 실행: `py -3 code/q01_panel_qa.py` (노트북 CPU 약 3분, 메모리 약 6GB). 이 문서와 `q01_panel_flags.parquet` 을 새로 쓴다.
- 플래그 파일 열: `f_resume_day`(B1), `f_prelist_otc`(B4), `f_liquidation_zone`(B2), `f_zero_inv_suspect`(B5), `f_net_gt_volume`(B12), `f_split60`(분할 후 60거래일), `f_idle`, `f_tradable`, `tick_fixed`(B3 수정값).
- 재현 한 줄 예시. B3: `p = pd.read_parquet('panel.parquet'); p[p.date < '2010-10-04'].tick.unique()` → `[1.]`. B1: `idle=(p.is_halted==1)|(p.volume==0); res=idle.groupby(p.ticker).shift(1,fill_value=False)&~idle; p.loc[res,'ret1d_clean'].abs().mean()` → 약 0.06 (정상일 0.018 의 3배). B4: `p[(p.ticker=='013890')&(p.date<'2019-10-30')][['date','volume','indiv_net']]` → 거래량 수천 주, 투자자 NaN.
- 한계: 대조 원천이 같은 DataGuide 라서 가격 대조는 '변환 오류 없음'만 증명한다. 2026-04-02 이후 구간, 키움 수급 원천(foreign_etc), 2010-10-04 이전 호가표(2차 자료)는 독립 확인을 못했다.
""")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))
    log("보고서 저장", OUT_MD, f"{len(''.join(L)):,}자")


def main():
    only = set(sys.argv[1:])
    P, sm = setup()
    s1(P, sm)
    Bb = s2(P, sm)
    s3(P)
    s4(P)
    s5(P)
    s6(P)
    s8(P, sm)
    s7(P)
    s9(P)
    R["P_info"] = (str(P.date.min().date()), str(P.date.max().date()), int(P.ticker.nunique()))
    build_flags(P)
    report(P)
    import os
    if os.environ.get("Q01_CACHE"):
        pd.to_pickle(R, os.environ["Q01_CACHE"])
    return P, sm


if __name__ == "__main__":
    main()

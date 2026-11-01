"""f01 검증: (1) 룩어헤드 교란 실험 + 양성 대조, (2) 재무 시점 정합, (3) 공시 건수 재계산, (4) 결측·분포·상관, (5) 유니버스 행 수.

실행: py -3 f01_verify_features.py   (약 2분, 최대 메모리 약 6GB)
입력: data/features/features_v2_kospi.parquet (f01_build_features.py 산출물)과 같은 입력 파일들
출력: results/f01_tables/*.csv, results/f01_tables/verify_output.md (보고서에 옮겨 적는 표)
모델 성과는 계산하지 않는다. 라벨은 LABEL_CAP(2016-12-31) 이후 마스크된 채로만 쓰고(교란 실험 3번만 2016-06-24 기준),
2017-01 이후 라벨·수익은 어디서도 만들지 않는다.
"""
import bisect
import gc
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

import f01_build_features as f01
from f01_build_features import FEATURES, RK, B1, B2_M0, B2_X, B3_DSC, B3_FIN, BLOCKS, build, load_inputs

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "f01_tables"
TAB.mkdir(parents=True, exist_ok=True)
SEED = 20261012
N_PERTURB = 100
EXPERIMENTS = [("A", "2019-06-28"), ("B", "2022-07-29"), ("C", "2016-06-24")]   # A: 요청 기준일, B: 재무 시점 교란(분기 보고서 접수 직전), C: 라벨 창 점검
CHECK_FLAGS = ["inv_ok5", "inv_ok20", "inv_nobs5", "inv_nobs20", "avg_tv60", "mktcap", "fin_has", "fin_period_end", "fin_rcept", "fin_reprt",
               "fin_ofs", "fin_eq_derived", "fin_amended", "fin_age_days", "roe_period_end", "roe_rcept", "roe_ofs", "roe_eq_derived", "roe_amended"]
CHECK_COLS = FEATURES + RK + CHECK_FLAGS
LABEL_COLS = ["label_ret5", "label_rk", "label_status", "label_end_date", "label_censored", "fwd_ret5_panel"]
CTL = ["ctl_fwd_ret5", "ctl_flow5_centered", "ctl_dsc_periodic_next5", "ctl_dsc_stake_next5", "ctl_dsc_exch_next5", "ctl_fin_by_period_end"]
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def md(df, fmt="{:.4g}", index=False):
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = []
    for r in d.itertuples(index=False):
        rows.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)) for v in r) + " |")
    return head + "\n".join(rows)


def neq(a, b):
    """같지 않은 원소 수 (NaN==NaN, NaT==NaT 는 같다)."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    return int((~((a == b) | (a.isna() & b.isna()))).sum())


def mem_gb():
    import psutil
    return psutil.Process().memory_info().peak_wset / 2**30


# ------------------------------------------------------------------ (1) 교란 실험
def perturb(panel, cls, fsav, fscore, S, C, rng, cal):
    after = cal[cal > np.datetime64(C)]
    pn = panel.copy()
    m = (pn["ticker"].isin(S) & (pn["date"] > C)).to_numpy()
    k = int(m.sum())
    for c in f01.PANEL_COLS:
        if c in ("ticker", "date"):
            continue
        v = panel[c].to_numpy()
        pn.loc[m, c] = v[rng.integers(0, len(v), k)]
    cl = cls.copy()
    cl["rcept_dt"] = pd.to_datetime(cl["rcept_dt"])
    me = (cl["ticker"].isin(S) & (cl["rcept_dt"] > C)).to_numpy()
    ke = int(me.sum())
    cl.loc[me, "rcept_dt"] = after[rng.integers(0, len(after), ke)]
    cl.loc[me, "cat"] = rng.permutation(cl.loc[me, "cat"].to_numpy())
    cl.loc[me, "is_amend"] = rng.random(ke) < 0.3
    fa = fsav.copy()
    mf = (fa["ticker"].isin(S) & (fa["first_rcept_dt"] > C)).to_numpy()
    kf = int(mf.sum())
    fa.loc[mf, "first_rcept_dt"] = after[rng.integers(0, len(after), kf)]
    fa.loc[mf, "last_rcept_dt"] = fa.loc[mf, "first_rcept_dt"] + pd.to_timedelta(rng.integers(0, 30, kf), unit="D")
    fa.loc[mf, "n_amend"] = rng.integers(0, 3, kf)
    keys = fsav.loc[mf, ["ticker", "reprt_code", "period_end"]].copy()
    keys["period_end"] = keys["period_end"].dt.strftime("%Y-%m-%d")
    keys["_hit"] = 1
    fs = fscore.copy()
    fs["ticker"] = fs["code"].str[1:]
    fs = fs.merge(keys, on=["ticker", "reprt_code", "period_end"], how="left")
    hit = fs["_hit"].notna().to_numpy()
    fs.loc[hit, "thstrm_amount"] = fs.loc[hit, "thstrm_amount"].to_numpy() * np.exp(rng.normal(0, 1, int(hit.sum())))
    fs = fs.drop(columns=["ticker", "_hit"])
    info = dict(panel_rows=k, event_rows=ke, fsav_rows=kf, fs_rows=int(hit.sum()))
    return pn, cl, fa, fs, info


def experiment(name, C, base, inputs, cal, S_pool, rng):
    panel, cls, fsav, fscore = inputs
    Cts = pd.Timestamp(C)
    S = list(rng.choice(S_pool, N_PERTURB, replace=False))
    pn, cl, fa, fs, info = perturb(panel, cls, fsav, fscore, set(S), Cts, rng, cal)
    P = build(pn, cl, fa, fs, controls=True, inplace=True)
    del pn, cl, fa, fs
    gc.collect()
    A = base[base["date"] <= Cts].reset_index(drop=True)
    B = P[P["date"] <= Cts].reset_index(drop=True)
    assert neq(A["ticker"], B["ticker"]) == 0 and neq(A["date"], B["date"]) == 0, "교란 전 행 집합이 달라졌다"
    mism = {c: neq(A[c], B[c]) for c in CHECK_COLS}
    ctl = {c: neq(A[c], B[c]) for c in CTL}
    # 교란이 실제로 작동했는지: 교란 종목의 기준일 이후 행에서 값이 바뀐 비율, 다른 종목 기준일 이후 순위 변화
    post_a, post_b = base[base["date"] > Cts], P[P["date"] > Cts]
    keep = ["ticker", "date", "flow_indiv_5", "str5", "log_mktcap", "amihud20", "dsc_stake_20", "dsc_amend_60", "fin_size", "fin_roe", "flow_indiv_5_rk"]
    j = post_a[keep].merge(post_b[keep], on=["ticker", "date"], suffixes=("_a", "_b"))
    sj = j[j["ticker"].isin(S)]
    oj = j[~j["ticker"].isin(S)]
    chg = {}
    for c in ["flow_indiv_5", "str5", "log_mktcap", "amihud20", "dsc_stake_20", "dsc_amend_60", "fin_size", "fin_roe"]:
        chg[c] = (neq(sj[c + "_a"], sj[c + "_b"]), len(sj))
    chg_rk = (neq(oj["flow_indiv_5_rk_a"], oj["flow_indiv_5_rk_b"]), len(oj))
    # 종목 간 누수: 교란하지 않은 종목의 기준일 이후 원값(순위 열 제외)은 다른 종목을 바꿔도 그대로여야 한다
    ma = ((base["date"] > Cts) & ~base["ticker"].isin(S)).to_numpy()
    mb = ((P["date"] > Cts) & ~P["ticker"].isin(S)).to_numpy()
    assert ma.sum() == mb.sum() and neq(base.loc[ma, "ticker"], P.loc[mb, "ticker"]) == 0 and neq(base.loc[ma, "date"], P.loc[mb, "date"]) == 0
    cross = sum(neq(base.loc[ma, c], P.loc[mb, c]) for c in FEATURES)
    cross_rows = int(ma.sum())
    lab = None
    if name == "C":                                     # 라벨 창 점검: 청산일이 기준일 이전인 라벨은 불변, 이후인 교란 종목 라벨은 변해야 한다
        ea = pd.to_datetime(A["label_end_date"])
        done = (ea <= Cts).to_numpy()
        lab_cols = ["label_ret5", "label_status", "label_end_date", "label_censored", "fwd_ret5_panel"]
        lab_mm = {c: neq(A.loc[done, c], B.loc[done, c]) for c in lab_cols}
        fut = (~done) & A["ticker"].isin(S).to_numpy() & A["label_ret5"].notna().to_numpy()
        lab = dict(done_rows=int(done.sum()), mismatch=lab_mm, future_rows=int(fut.sum()),
                   future_changed=neq(A.loc[fut, "label_ret5"], B.loc[fut, "label_ret5"]))
    res = dict(name=name, cutoff=C, rows=len(A), tickers=A["ticker"].nunique(), dates=A["date"].nunique(), cols=len(CHECK_COLS),
               mismatch_total=sum(mism.values()), mismatch_cols={k: v for k, v in mism.items() if v}, ctl=ctl, changed=chg,
               changed_rk_others=chg_rk, cross_rows=cross_rows, cross_mismatch=cross, info=info, label=lab, S=S[:5])
    del P, A, B, j, post_a, post_b, sj, oj
    gc.collect()
    return res


def part1(inputs):
    say("## 1. 룩어헤드 교란 실험")
    panel, cls, fsav, fscore = inputs
    cal = np.sort(pd.unique(panel["date"].to_numpy()))
    t0 = time.time()
    base = build(panel, cls, fsav, fscore, controls=True)
    say(f"- 기준 실행(요청 기준일 교란 없음, controls=True): {len(base):,}행, {base.attrs['seconds']:.0f}초")
    saved = pd.read_parquet(f01.OUT)
    sc = [c for c in saved.columns]
    rep = {}
    for c in sc:
        a = base[c]
        b = saved[c]
        if str(b.dtype) == "float32":
            a = a.astype(np.float32)
        rep[c] = neq(a, b)
    say(f"- 재현: 저장 파일({f01.OUT.name}, {len(saved):,}행 x {len(sc)}열)과 같은 입력으로 다시 만든 값의 불일치 셀 수 = {sum(rep.values())} (열 {sum(1 for v in rep.values() if v)}개)")
    assert len(base) == len(saved) and neq(base["ticker"], saved["ticker"]) == 0 and neq(base["date"], saved["date"]) == 0
    del saved
    # 교란 후보: 세 기준일 모두 앞뒤로 행이 있는 종목
    d0 = panel.groupby("ticker")["date"].agg(["min", "max"])
    results = []
    for name, C in EXPERIMENTS:
        pool = d0.index[(d0["min"] < pd.Timestamp(C) - pd.DateOffset(days=400)) & (d0["max"] > pd.Timestamp(C) + pd.DateOffset(days=200))].tolist()
        rng = np.random.default_rng(SEED + ord(name))
        r = experiment(name, C, base, inputs, cal, pool, rng)
        results.append(r)
        say(f"- 실험 {name} (기준일 {C}): 교란 종목 {N_PERTURB}개, 교란 입력 행 수 {r['info']}; 비교 {r['rows']:,}행 x {r['cols']}열, "
            f"**불일치 {r['mismatch_total']}건** {r['mismatch_cols'] or ''}")
    say("")
    say("교란 실험 요약")
    rows = []
    for r in results:
        rows.append(dict(실험=r["name"], 기준일=r["cutoff"], 비교_행=r["rows"], 비교_형성일=r["dates"], 비교_열=r["cols"], 불일치=r["mismatch_total"],
                         교란_panel행=r["info"]["panel_rows"], 교란_공시행=r["info"]["event_rows"], 교란_접수일행=r["info"]["fsav_rows"], 교란_재무행=r["info"]["fs_rows"]))
    say(md(pd.DataFrame(rows)))
    say("")
    say("교란이 작동했는지(교란 종목의 기준일 이후 행에서 값이 바뀐 비율)와 다른 종목 순위 변화:")
    rows = []
    for r in results:
        d = {"실험": r["name"]}
        for c, (a, n) in r["changed"].items():
            d[c] = f"{a/n:.1%}"
        d["타종목 flow_indiv_5_rk"] = f"{r['changed_rk_others'][0]/r['changed_rk_others'][1]:.1%}"
        d["타종목 원값 불일치 (34열, 기준일 이후 행)"] = f"{r['cross_mismatch']} / {r['cross_rows']:,}행"
        rows.append(d)
    say(md(pd.DataFrame(rows)))
    say("")
    say("양성 대조(일부러 미래를 쓰는 열: 기준일 이전 행에서 불일치가 나와야 한다):")
    rows = []
    for r in results:
        d = {"실험": r["name"]}
        d.update({c: v for c, v in r["ctl"].items()})
        d["비교_행"] = r["rows"]
        rows.append(d)
    say(md(pd.DataFrame(rows)))
    lab = [r for r in results if r["label"]][0]["label"]
    say("")
    say(f"라벨 창 점검(실험 C, 기준일 2016-06-24): 청산일이 기준일 이전인 라벨 {lab['done_rows']:,}행의 불일치 {lab['mismatch']}; 청산일이 기준일 뒤인 교란 종목 라벨 {lab['future_rows']}행 중 값이 바뀐 행 {lab['future_changed']}")
    say(f"- 세 실험 + 재현 소요 {time.time()-t0:.0f}초, 최대 메모리 {mem_gb():.2f}GB")
    json.dump(results, open(TAB / "perturb_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return base


# ------------------------------------------------------------------ (2) 재무 시점 정합
def part2(F, inputs):
    panel, cls, fsav, fscore = inputs
    say("\n## 2. 재무 시점 정합")
    n_fin = int(F["fin_has"].sum())
    v1 = int((F["fin_rcept"] >= F["date"]).sum())
    v2 = int((F["roe_rcept"] >= F["date"]).sum())
    v3 = int((F["fin_amended"].astype(bool) & ~(F["fin_amend_ever"].astype(bool))).sum())
    say(f"- 재무가 붙은 {n_fin:,}행 전부에서 fin_rcept(최초 접수일) < 형성일: 위반 {v1}건. ROE 보고서도 roe_rcept < 형성일: 위반 {v2}건. fin_amended=1 인데 정정 이력 없음: {v3}건")
    say(f"- 자본총계 태그가 없어 자산-부채로 계산한 행 {int((F['fin_eq_derived']==1).sum()):,} ({(F['fin_eq_derived']==1).sum()/n_fin:.1%})")
    say(f"- fin_age_days(형성일-접수일) 최솟값 {int(F['fin_age_days'].min())}일, 중앙값 {int(F['fin_age_days'].median())}일; 정정 이력(사후 포함) 보고서를 쓴 행 {F['fin_amend_ever'].mean():.1%}, 형성일 전에 정정이 끝난 행 {F['fin_amended'].mean():.1%}; 별도(OFS) 대체 행 {(F['fin_ofs']==1).sum():,} ({(F['fin_ofs']==1).sum()/n_fin:.1%})")
    # 독립 재계산: 원자료(fs_line, fs_available_date)에서 직접 가장 최근 보고서를 찾아 값까지 다시 계산
    rng = np.random.default_rng(SEED + 2)
    has = F[F["fin_has"] == 1]
    tk = rng.choice(has["ticker"].unique(), 20, replace=False)
    f = fscore[fscore["market"] == "KOSPI"]
    f = f[f["code"].str[1:].isin(tk)]

    def stmt(ft, reprt, pe):
        """(reprt_code, 기간말) 보고서의 (구분, 자산, 부채, 자본, 순이익). 연결은 자산·부채가 모두 있을 때만, 자본 태그가 없으면 자산-부채."""
        q = ft[(ft["reprt_code"] == reprt) & (ft["period_end"] == pe)]

        def val(div, acc, sj):
            v = q[(q["fs_div"] == div) & (q["account_id"] == acc) & (q["sj_div"] == sj)]["thstrm_amount"]
            return v.iloc[0] if len(v) else np.nan
        for div in ("CFS", "OFS"):
            a, l = val(div, "ifrs-full_Assets", "BS"), val(div, "ifrs-full_Liabilities", "BS")
            if np.isfinite(a) and np.isfinite(l):
                e = val(div, "ifrs-full_Equity", "BS")
                return div, a, l, (e if np.isfinite(e) else a - l), val(div, "ifrs-full_ProfitLoss", "IS")
        return None

    rows = []
    bad = []
    roe_bad = []
    n_roe = ok_roe = 0
    for t in tk:
        ft = f[f["code"] == "A" + t]
        av = fsav[fsav["ticker"] == t].copy()
        av["pe"] = av["period_end"].dt.strftime("%Y-%m-%d")
        av["st"] = [stmt(ft, rc, pe) for rc, pe in zip(av["reprt_code"], av["pe"])]
        av = av[av["st"].notna()]
        sub = has[has["ticker"] == t]
        for y in range(2016, 2027):
            s = sub[sub["date"].dt.year == y]
            if len(s) == 0:
                continue
            for _, r in s.sample(min(2, len(s)), random_state=int(rng.integers(1e9))).iterrows():
                cand = av[av["first_rcept_dt"] < r["date"]].sort_values(["period_end", "first_rcept_dt"])
                pick = cand.iloc[-1]
                div, a, l, e, ni = pick["st"]
                a_ok = pick["period_end"] == r["fin_period_end"] and pick["first_rcept_dt"] == r["fin_rcept"] and (div == "OFS") == bool(r["fin_ofs"])
                v_ok = abs(np.log(a) - r["fin_size"]) < 1e-9 and abs(l / a - r["fin_lev"]) < 1e-9 and (not np.isfinite(r["mktcap"]) or abs(e / r["mktcap"] - r["fin_bm"]) < 1e-9)
                rows.append(dict(ticker=t, date=r["date"].date(), fin_period_end=r["fin_period_end"].date(), fin_rcept=r["fin_rcept"].date(),
                                 형성일_전_접수=bool(r["fin_rcept"] < r["date"]), 원자료_재계산_보고서_일치=bool(a_ok), 값_일치=bool(v_ok)))
                if not (a_ok and v_ok):
                    bad.append((t, str(r["date"].date())))
                an = cand[cand["reprt_code"] == "11011"]                                   # ROE: 가장 최근 사업보고서
                if len(an):
                    q = an.iloc[-1]
                    d2, a2, l2, e2, ni2 = q["st"]
                    exp = ni2 / e2 if (e2 > 0 and np.isfinite(ni2)) else np.nan
                    n_roe += 1
                    same = (q["period_end"] == r["roe_period_end"]) and ((np.isnan(exp) and np.isnan(r["fin_roe"])) or abs(exp - r["fin_roe"]) < 1e-9)
                    ok_roe += int(same)
                    if not same:
                        roe_bad.append((t, str(r["date"].date())))
    R = pd.DataFrame(rows)
    say(f"- 임의 20종목 x 연도(2016~2026, 연도당 최대 2개 형성일) {len(R)}개 표본: 접수일<형성일 {R['형성일_전_접수'].sum()}/{len(R)}, "
        f"원자료에서 직접 찾은 최근 보고서(기간말·접수일·별도 대체 여부)와 일치 {R['원자료_재계산_보고서_일치'].sum()}/{len(R)}, 자산총계·부채비율·B/M 값 일치 {R['값_일치'].sum()}/{len(R)}. 불일치 {bad[:5]}")
    say(f"- 같은 표본에서 ROE(가장 최근 사업보고서 당기순이익/자본총계)의 보고서 선택과 값 재계산 일치 {ok_roe}/{n_roe}. 불일치 {roe_bad[:5]}")
    # 삼성전자 경계
    say("\n삼성전자(005930) 2025 사업보고서(first_rcept_dt 2026-03-10) 반영 시점:")
    av = fsav[(fsav["ticker"] == "005930") & (fsav["reprt_code"] == "11011")].sort_values("period_end").tail(3)
    say(md(av[["period_end", "first_rcept_dt", "last_rcept_dt", "n_amend"]].assign(period_end=lambda d: d.period_end.dt.date, first_rcept_dt=lambda d: d.first_rcept_dt.dt.date, last_rcept_dt=lambda d: d.last_rcept_dt.dt.date)))
    rep = f01.fin_reports(fscore, fsav, {"005930"})
    st = f01.fin_state(rep)
    probe = pd.DataFrame({"ticker": "005930", "date": pd.to_datetime(["2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12"])})
    pr = f01.fin_asof(probe, st)
    say("\n임의 날짜에 직접 적용한 as-of 결과 (접수 다음 날부터 사용):")
    say(md(pr[["date", "period_end", "first_rcept_dt", "reprt_code"]].assign(date=lambda d: d.date.dt.date, period_end=lambda d: d.period_end.dt.date, first_rcept_dt=lambda d: d.first_rcept_dt.dt.date)))
    s = F[(F["ticker"] == "005930") & (F["date"] >= "2026-02-27") & (F["date"] <= "2026-03-27")][["date", "fin_period_end", "fin_rcept", "fin_reprt", "fin_size", "roe_period_end", "fin_roe"]]
    say("\n저장 파일의 삼성전자 형성일 행:")
    say(md(s.assign(date=lambda d: d.date.dt.date, fin_period_end=lambda d: d.fin_period_end.dt.date, fin_rcept=lambda d: d.fin_rcept.dt.date, roe_period_end=lambda d: d.roe_period_end.dt.date)))
    s24 = F[(F["ticker"] == "005930") & (F["date"] >= "2025-03-07") & (F["date"] <= "2025-03-21")][["date", "fin_period_end", "fin_rcept", "roe_period_end"]]
    say("\n2024 사업보고서(first_rcept_dt 2025-03-11)의 경우:")
    say(md(s24.assign(date=lambda d: d.date.dt.date, fin_period_end=lambda d: d.fin_period_end.dt.date, fin_rcept=lambda d: d.fin_rcept.dt.date, roe_period_end=lambda d: d.roe_period_end.dt.date)))
    R.to_csv(TAB / "fin_timing_samples.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ (3) 공시 정합
def part3(F, inputs):
    panel, cls, fsav, fscore = inputs
    say("\n## 3. 공시 건수 정합")
    cal = np.sort(pd.unique(panel["date"].to_numpy()))
    calp = [pd.Timestamp(x) for x in cal]
    c = cls[cls["cat"] != f01.DSC_EXCLUDE_CAT].copy()
    c["rcept_dt"] = pd.to_datetime(c["rcept_dt"])
    last = panel.groupby("ticker")["date"].max()
    rng = np.random.default_rng(SEED + 3)
    cand = F[F["dsc_stake_60"].notna()].groupby("ticker").size()
    cand = cand[cand > 200].index.to_numpy()
    tk = rng.choice(cand, 10, replace=False)
    catname = {"정기": "periodic", "주요사항": "major", "지분": "stake", "수시공정": "exch"}
    tot = bad = 0
    per = []
    example = None
    nm_map = pd.read_parquet(f01.CLS, columns=["ticker", "rcept_dt", "report_nm", "cat", "is_amend"], filters=[("ticker", "in", [str(t) for t in tk])])
    for t in tk:
        ev = c[(c["ticker"] == t) & (c["rcept_dt"] <= last[t])].merge(
            nm_map.assign(rcept_dt=pd.to_datetime(nm_map["rcept_dt"]))[["ticker", "rcept_dt", "report_nm"]].drop_duplicates(["ticker", "rcept_dt"]), on=["ticker", "rcept_dt"], how="left")
        # 손으로 세기: 접수일 -> (이벤트 날짜 = 접수일 이상 첫 거래일) -> next_td = 그 다음 거래일 (날짜 객체로 비교)
        items = []
        for r in ev.itertuples():
            d = bisect.bisect_left(calp, r.rcept_dt)
            if d + 1 >= len(calp):
                continue
            kind = "amend" if r.is_amend else catname[r.cat]
            items.append((kind, calp[d + 1], r.rcept_dt, r.report_nm))
        sub = F[(F["ticker"] == t) & F["dsc_stake_60"].notna()]
        ncell = nbad = 0
        for r in sub.itertuples():
            ti = bisect.bisect_left(calp, r.date)
            for w in f01.DSC_WINS:
                win = set(calp[ti - w + 1:ti + 1])
                for kind in ["periodic", "major", "stake", "exch", "amend"]:
                    mine = sum(1 for k, nx, *_ in items if k == kind and nx in win)
                    got = getattr(r, f"dsc_{kind}_{w}")
                    ncell += 1
                    nbad += int(mine != got)
        per.append(dict(ticker=t, 형성일=len(sub), 비교_셀=ncell, 불일치=nbad, 이벤트=len(items)))
        tot += ncell
        bad += nbad
        if example is None and (sub["dsc_exch_20"] >= 3).any() and (sub["dsc_exch_20"] <= 5).any():
            rr = sub[(sub["dsc_exch_20"] >= 3) & (sub["dsc_exch_20"] <= 5)].iloc[10]
            ti = bisect.bisect_left(calp, rr["date"])
            win = set(calp[ti - 19:ti + 1])
            ex = [(str(rd.date()), str(nx.date()), k, nm) for k, nx, rd, nm in items if k == "exch" and nx in win]
            example = (t, rr["date"], int(rr["dsc_exch_20"]), ex)
    say(f"- 임의 10종목({', '.join(tk)})의 모든 형성일 x {{5,20,60}}거래일 x 5유형을 원 목록(disclosures_kospi_class, 발행공시 '기타' 제외, 상장폐지 후 접수 제외)에서 파이썬 루프(날짜 객체 집합 멤버십)로 다시 센 값과 비교: "
        f"{tot:,}셀 중 불일치 **{bad}**")
    say(md(pd.DataFrame(per)))
    if example:
        t, d, n, ex = example
        say(f"\n손 계산 예시: 종목 {t}, 형성일 {d.date()}, 수시공정 20거래일 건수(저장값 {n}). 창 안(next_td 가 형성일 포함 직전 20거래일)에 드는 원 목록 공시:")
        say(md(pd.DataFrame(ex, columns=["접수일", "next_td", "유형", "보고서명"])))
    # 이벤트 표(disclosure_events_daily)와 대조: 같은 규칙인지
    evt = pd.read_parquet(f01.D / "dart" / "disclosure_events_daily.parquet")
    tickers = np.asarray(pd.factorize(panel.sort_values(["ticker", "date"], kind="stable")["ticker"])[1])
    cum = f01.event_cumulative(cls, cal, tickers, last)
    tk_idx = pd.Series(np.arange(len(tickers)), index=tickers)
    evt = evt[evt["ticker"].isin(tk_idx.index) & evt["next_td"].notna()]
    evt = evt[evt["date"] <= evt["ticker"].map(last)]
    ti = np.searchsorted(cal, evt["next_td"].to_numpy())
    kk = tk_idx[evt["ticker"]].to_numpy()
    diffs = {}
    for nm, col in [("periodic", "n_periodic"), ("major", "n_major"), ("stake", "n_stake"), ("exch", "n_exchange")]:
        mine = cum[nm][kk, ti + 1] - cum[nm][kk, ti]
        diffs[nm] = int((mine != evt[col].to_numpy()).sum())
    tot_mine = {nm: int((cum[nm][:, 1:] - cum[nm][:, :-1]).sum()) for nm in cum}
    tot_evt = {nm: int(evt[col].sum()) for nm, col in [("periodic", "n_periodic"), ("major", "n_major"), ("stake", "n_stake"), ("exch", "n_exchange")]}
    say(f"\n- disclosure_events_daily 와 대조(상장폐지 후 접수분 제외, 이벤트가 있는 {len(evt):,}개 (종목, next_td) 행): 4개 유형 건수 불일치 행 {diffs}; "
        f"총건수 내 계산 {tot_mine} vs 이벤트 표 {tot_evt} (정정은 이벤트 표가 발행공시 정정을 포함하므로 비교하지 않음)")
    pd.DataFrame(per).to_csv(TAB / "dsc_recount.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ (4)(5) 표
def part45(F):
    say("\n## 4. 결측률·분포·상관")
    F = F.copy()
    F["year"] = F["date"].dt.year
    blocks = {"B1 수급(6)": B1, "B2 M0 6지표": B2_M0, "B2 추가 3": B2_X, "B3 공시(15)": B3_DSC, "B3 재무(4)": B3_FIN}
    anyna = pd.DataFrame({k: F[v].isna().any(axis=1) for k, v in blocks.items()})
    anyna["year"] = F["year"]
    mt = anyna.groupby("year").mean()
    mt["전체 34 중 하나라도"] = F[FEATURES].isna().any(axis=1).groupby(F["year"]).mean()
    mt["B1~B3 완전(34 전부 있음)"] = 1 - mt["전체 34 중 하나라도"]
    say("블록 안 변수 중 하나라도 결측인 행의 비율 (연도별):")
    say(md((mt * 100).round(1).assign(), fmt="{:.1f}", index=True))
    mt.to_csv(TAB / "missing_by_year_block.csv", encoding="utf-8-sig")
    mv = F.groupby("year")[FEATURES].apply(lambda d: d.isna().mean())
    mv.T.to_csv(TAB / "missing_by_var_year.csv", encoding="utf-8-sig")
    say("\n변수별 결측률은 results/f01_tables/missing_by_var_year.csv. 전 기간 평균 결측률이 높은 변수:")
    top = F[FEATURES].isna().mean().sort_values(ascending=False).head(8)
    say(md(top.rename("결측률").to_frame().round(3), fmt="{:.3f}", index=True))
    # 분위수
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    q = F.groupby("year")[FEATURES].quantile(qs)
    q.index.names = ["year", "q"]
    q.stack().unstack("q").rename_axis(["year", "var"]).reset_index().to_csv(TAB / "raw_quantiles_by_year.csv", index=False, encoding="utf-8-sig")
    med = F.groupby("year")[FEATURES].median()
    pooled = F[FEATURES].quantile([0.05, 0.5, 0.95])
    rows = []
    for c in FEATURES:
        sc = pooled.loc[0.95, c] - pooled.loc[0.05, c]
        sc = sc if sc and np.isfinite(sc) else 1.0
        m = med[c].dropna()
        if len(m) == 0:
            continue
        z = (m - pooled.loc[0.5, c]) / sc
        yrs = F.loc[F[c].notna(), "year"].unique()
        iq = F.groupby("year")[c].agg(lambda s: s.quantile(0.75) - s.quantile(0.25)).dropna()
        rows.append(dict(변수=c, 풀링_중앙값=pooled.loc[0.5, c], 풀링_5_95폭=sc, 연도별_중앙값_최대이탈=z.abs().max(), 이탈_연도=int(z.abs().idxmax()),
                         IQR_최대_최소비=(iq.max() / iq.min()) if iq.min() > 0 else np.nan, 값있는_연도=f"{int(min(yrs))}~{int(max(yrs))}"))
    dr = pd.DataFrame(rows)
    dr.to_csv(TAB / "drift_summary.csv", index=False, encoding="utf-8-sig")
    say("\n순위 변환 전 원값의 분포 이동 요약(연도별 중앙값이 풀링 중앙값에서 벗어난 정도를 풀링 5~95% 폭으로 나눈 값의 최대, 연도별 IQR 최대/최소). 전체 분위수는 raw_quantiles_by_year.csv:")
    say(md(dr, fmt="{:.3g}"))
    # 상관: 완전한 행
    full = F[RK].dropna()
    cm = full.corr()
    cm.to_csv(TAB / "corr_rk.csv", encoding="utf-8-sig")
    say(f"\n순위 열(_rk) 상관 행렬은 34개 변수 모두 있는 {len(full):,}행(2019-11 이후)의 피어슨(= 날짜별 순위의 풀링 스피어만 근사). 전체는 corr_rk.csv.")
    bm = {k: [v + "_rk" if not v.endswith("_rk") else v for v in vs] for k, vs in blocks.items()}
    names = list(bm)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            sub = cm.loc[bm[a], bm[b]].abs().to_numpy()
            if a == b:
                n = sub.shape[0]
                sub = sub[~np.eye(n, dtype=bool)] if n > 1 else sub
            M.loc[a, b] = float(np.mean(sub))
    say("블록 간 평균 절대 상관(대각은 블록 안 쌍):")
    say(md(M.round(3), fmt="{:.3f}", index=True))
    pairs = cm.where(np.triu(np.ones(cm.shape, dtype=bool), 1)).stack().rename("corr").reset_index()
    pairs.columns = ["a", "b", "corr"]
    ba = {v: k for k, vs in bm.items() for v in vs}
    pairs["cross"] = pairs["a"].map(ba) != pairs["b"].map(ba)
    top = pairs[pairs["cross"]].reindex(pairs[pairs["cross"]]["corr"].abs().sort_values(ascending=False).index).head(10)
    say("\n블록이 다른 변수 쌍 중 |상관| 상위 10:")
    say(md(top.assign(a=top.a.str.replace("_rk", ""), b=top.b.str.replace("_rk", "")).drop(columns="cross"), fmt="{:.3f}"))
    top_in = pairs[~pairs["cross"]]
    top_in = top_in.reindex(top_in["corr"].abs().sort_values(ascending=False).index).head(5)
    say("\n블록 안 |상관| 상위 5 (중복 정보 확인용):")
    say(md(top_in.assign(a=top_in.a.str.replace("_rk", ""), b=top_in.b.str.replace("_rk", "")).drop(columns="cross"), fmt="{:.3f}"))

    say("\n## 5. 유니버스 행 수 (f_tradable==1 & exclude_flag==0, 형성일 행)")
    ry = F.groupby("year").agg(형성일=("date", "nunique"), 행=("ticker", "size"), 종목=("ticker", "nunique"))
    ry["형성일당_평균"] = (ry["행"] / ry["형성일"]).round(1)
    ry["라벨_있음(상태0·1)"] = F[F["label_status"].isin([0, 1])].groupby("year").size().reindex(ry.index).fillna(0).astype(int)
    ry["시험구간_마스크(상태5)"] = F[F["label_status"] == 5].groupby("year").size().reindex(ry.index).fillna(0).astype(int)
    ry["하위20%_거래대금_제외후"] = F.groupby("year").apply(lambda d: int((d["avg_tv60"] >= d.groupby("date")["avg_tv60"].transform(lambda s: s.quantile(0.2))).sum()))
    say(md(ry, fmt="{:.1f}", index=True))
    ry.to_csv(TAB / "rows_by_year.csv", encoding="utf-8-sig")
    st = F["label_status"].value_counts().sort_index()
    say(f"\n라벨 상태 분포: {st.to_dict()}  (0 정상, 1 상폐로 잘림, 3 다음 거래일 매수 불가, 4 뒤 행 없음, 5 시험 구간 보호 마스크)")


def main():
    t0 = time.time()
    inputs = load_inputs()
    base = part1(inputs)
    F = pd.read_parquet(f01.OUT)
    del base
    gc.collect()
    part2(F, inputs)
    part3(F, inputs)
    part45(F)
    say(f"\n전체 소요 {time.time()-t0:.0f}초, 최대 메모리 {mem_gb():.2f}GB")
    (TAB / "verify_output.md").write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

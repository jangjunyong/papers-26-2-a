"""계획 6단계 파일럿: '눈에 띄는 차트' 뒤 개인이 사고 6~20일 뒤 반전하는가(외국인 대조). GPU 불필요.

정의 (모두 형성일 t 종가까지의 정보만 쓴다)
- 표본: 2004-12-30 KOSPI 시총 상위 200 보통주(data/daily/universe_pilot_20041230.txt), 형성일 2005-01~2012-12.
- 형성일: 시장 달력(200종목 거래일 합집합)의 매주 마지막 거래일(images.py와 같은 달력).
- 자격: t일 거래(거래량>0), 과거 65거래일 가격·거래대금 완비, 기준 거래대금 B>0.
- ret5  = 종가_t / 종가_{t-5} - 1 (과거 5일 수익률)
- abvol = 평균 거래대금(t-4..t) / B,   B = 평균 거래대금(t-64..t-5) (이벤트 직전 60일 = 평상시 거래대금)
- 이벤트(기본): 자격 종목 단면에서 ret5 상위 10% 그리고 abvol 상위 10%.  대안: 각 상위 20%.
- 개인 흐름 = Σ 개인 순매수(t+1..t+5) / B.  외국인 = 외국인+기타외국인, 같은 방식. (단위: 평상시 하루 거래대금의 배수)
- AR = 일수익 - 200종목 동일가중 평균 일수익.  CAR(a,b) = Σ AR(t+a..t+b), 수익은 t+1부터 센다.
  CAR(6,20)은 흐름 창(t+1..t+5)과 겹치지 않는다. 20일 창이 다 차지 않으면(상장폐지 등) 그 관측은 뺀다.
- 분위: 흐름 3분위는 전체 기간 풀링 기준(서술 통계용, 매매 규칙 아님).
- 표준오차: 형성일 군집(요청 기준) + 월 군집(주간 형성·20일 창 중첩 보완).
사용: py -3 a2_event.py          (결과 출력 + results/a2_car_paths.png)
      py -3 a2_event.py check    (미래 정보 누출 점검: t 이후 데이터를 지워도 이벤트 판정이 같은지)
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data" / "daily"
TOP, TOP_ALT = 0.10, 0.20
H = 20


def load():
    tks = (DATA / "universe_pilot_20041230.txt").read_text(encoding="utf-8").split()
    px = {tk: pd.read_parquet(DATA / "ohlcv" / f"{tk}.parquet") for tk in tks}
    # 투자자 파일이 없는 종목은 흐름이 NaN → 분석에서 빠진다. 이벤트 순위·시장수익은 가격이 있는 200종목 전체로 계산.
    inv = {tk: pd.read_parquet(f) for tk in tks if (f := DATA / "investor" / f"{tk}.parquet").exists()}
    C = pd.DataFrame({tk: d.close for tk, d in px.items()}).sort_index()
    V = pd.DataFrame({tk: d.value for tk, d in px.items()}).reindex(C.index)
    VOL = pd.DataFrame({tk: d.volume for tk, d in px.items()}).reindex(C.index)
    has = C.notna() & C.columns.isin(list(inv))
    # 순매수가 없는 상장일(거래정지)은 0, 상장 전후·파일 없는 종목은 NaN
    P = pd.DataFrame({tk: d["개인"] for tk, d in inv.items()}).reindex(index=C.index, columns=C.columns).fillna(0).where(has)
    F = pd.DataFrame({tk: d["외국인"] + d["기타외국인"] for tk, d in inv.items()}).reindex(index=C.index, columns=C.columns).fillna(0).where(has)
    return C, V, VOL, P, F, inv


def features(C, V, VOL):
    """t일까지의 정보만으로 만든 단면 변수와 이벤트 플래그."""
    ret5 = C / C.shift(5) - 1
    B = V.shift(5).rolling(60).mean()
    abvol = V.rolling(5).mean() / B
    ok = (VOL > 0) & ret5.notna() & C.shift(64).notna() & (B > 0) & abvol.notna()
    r5, av = ret5.where(ok), abvol.where(ok)
    rk5, rkv = r5.rank(axis=1, pct=True), av.rank(axis=1, pct=True)
    ev = (rk5 > 1 - TOP) & (rkv > 1 - TOP)
    ev_alt = (rk5 > 1 - TOP_ALT) & (rkv > 1 - TOP_ALT)
    return ok, ret5, abvol, B, ev, ev_alt


def form_dates(cal):
    s = cal.to_series()
    f = s.groupby(cal.to_period("W")).max()
    return f[(f >= "2005-01-01") & (f <= "2012-12-31")].values


def panel():
    C, V, VOL, P, F, inv = load()
    ok, ret5, abvol, B, ev, ev_alt = features(C, V, VOL)
    R = C.pct_change(fill_method=None)
    AR = R.sub(R.mean(axis=1), axis=0)
    car = {k: sum(AR.shift(-j) for j in range(1, k + 1)) for k in range(1, H + 1)}  # NaN이면 창 미완 → 제외
    fwd = lambda X: sum(X.shift(-j) for j in range(1, 6)) / B                      # noqa: E731
    pre = lambda X: sum(X.shift(j) for j in range(0, 5)) / B                       # noqa: E731
    cols = {"ok": ok, "ev": ev, "ev_alt": ev_alt, "ret5": ret5, "abvol": abvol,
            "ind": fwd(P), "frg": fwd(F), "ind_pre": pre(P), "frg_pre": pre(F),
            **{f"car{k}": car[k] for k in range(1, H + 1)}}
    fd = form_dates(C.index)
    df = pd.concat({n: x.loc[fd].stack(future_stack=True) for n, x in cols.items()}, axis=1)
    df.index.names = ["date", "ticker"]
    df = df[df.ok.astype(bool)].drop(columns="ok").reset_index()
    df[["ev", "ev_alt"]] = df[["ev", "ev_alt"]].astype(bool)
    df["car1_5"] = df.car5 * 100
    df["car6_20"] = (df.car20 - df.car5) * 100
    df["month"] = df.date.dt.to_period("M").astype(str)
    return df.dropna(subset=["car20", "ind", "frg"]), inv, V


def quality(inv, V):
    """개인+외국인+기관계+기타 = 0 점검. 잔차 / 그날 거래대금."""
    res = []
    for tk, d in inv.items():
        s = d.drop(columns="전체").sum(axis=1)
        v = V[tk].reindex(d.index)
        res.append(pd.DataFrame({"abs": s.abs(), "rel": s.abs() / v.where(v > 0), "tot": d["전체"].abs()}))
    q = pd.concat(res)
    print(f"[품질] 종목 {len(inv)}개, 종목-일 {len(q):,}개, 빈 파일 {sum(len(d) == 0 for d in inv.values())}개")
    print(f"  |11개 투자자 합| 최대 {q['abs'].max():,.0f}원, 0이 아닌 비율 {(q['abs'] > 0).mean():.4%}, "
          f"거래대금 대비 최대 {q.rel.max():.2e}, '전체' 열 최대 {q.tot.max():,.0f}")


def ols(formula, d, groups):
    m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[groups])[0]})
    return m


def tertile_table(d, var, label):
    d = d.copy()
    d["q"] = pd.qcut(d[var], 3, labels=["T1", "T2", "T3"]).astype(str)
    rows = []
    for q in ["T1", "T2", "T3"]:
        x = d[d.q == q]
        r = {"분위": q, "N": len(x), f"{label} 평균": x[var].mean()}
        for y in ["car1_5", "car6_20"]:
            r[y] = x[y].mean()
            r[y + " t(일)"] = ols(f"{y} ~ 1", x, "date").tvalues.iloc[0]
            r[y + " t(월)"] = ols(f"{y} ~ 1", x, "month").tvalues.iloc[0]
        rows.append(r)
    d["hi"], d["lo"] = (d.q == "T3").astype(float), (d.q == "T1").astype(float)
    x = d[d.q != "T2"]
    diff = {g: ols("car6_20 ~ hi", x, g) for g in ("date", "month")}
    rows.append({"분위": "T3-T1", "car6_20": diff["date"].params.hi,
                 "car6_20 t(일)": diff["date"].tvalues.hi, "car6_20 t(월)": diff["month"].tvalues.hi})
    return pd.DataFrame(rows), d


def winz(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def analyze(df, evcol):
    e, n = df[df[evcol]], df[~df[evcol]]
    print(f"\n==================== 이벤트 정의 = {evcol} ====================")
    print(f"관측 {len(df):,} (형성일 {df.date.nunique()}), 이벤트 {len(e):,} ({len(e) / len(df):.1%}), "
          f"이벤트 종목 {e.ticker.nunique()}개")
    print("연도별 이벤트 수:", e.date.dt.year.value_counts().sort_index().to_dict())

    # (a) 이벤트 vs 비이벤트 평균 경로
    cols = ["ret5", "abvol", "ind_pre", "frg_pre", "ind", "frg", "car1_5", "car6_20"]
    a = pd.DataFrame({"이벤트": e[cols].mean(), "비이벤트": n[cols].mean()})
    print("\n(a) 평균 (ret5 비율, 흐름=평상시 하루 거래대금 배수, CAR %)\n" + a.to_string(float_format="%.3f"))
    df = df.assign(evd=df[evcol].astype(float))
    for y in ["car1_5", "car6_20", "ind", "frg"]:
        m = ols(f"{y} ~ evd", df, "date")
        print(f"  {y}: 이벤트-비이벤트 = {m.params.evd:.3f} (t일 {m.tvalues.evd:.2f})")
    print("  CAR 경로 k=1..20 (%): 이벤트", np.round(e[[f"car{k}" for k in range(1, H + 1)]].mean().values * 100, 2))
    print("                       비이벤트", np.round(n[[f"car{k}" for k in range(1, H + 1)]].mean().values * 100, 2))

    # (b)(c) 이벤트 안 흐름 3분위, 비교용 비이벤트
    out = {}
    for var, lab in [("ind", "개인"), ("frg", "외국인")]:
        for nm, sub in [("이벤트", e), ("비이벤트", n)]:
            t, dq = tertile_table(sub, var, lab)
            print(f"\n({'b' if var == 'ind' else 'c'}) {nm} 안 {lab} 순매수(t+1..5) 3분위\n" + t.to_string(index=False, float_format="%.3f"))
            out[(var, nm)] = dq

    # (d) 회귀: 전체 표본, 흐름은 1/99% 윈저화 후 표준화
    d = df.copy()
    for v in ["ind", "frg", "ind_pre", "frg_pre"]:
        w = winz(d[v])
        d[v + "_z"] = (w - w.mean()) / w.std()
    d["lnabv"] = np.log(d.abvol)
    print("\n(d) 회귀 CAR(6,20)% ~ 흐름(z) × 이벤트")
    for f in ["car6_20 ~ ind_z + frg_z + evd + evd:ind_z + evd:frg_z",
              "car6_20 ~ ind_z + frg_z + evd + evd:ind_z + evd:frg_z + ret5 + lnabv + ind_pre_z + frg_pre_z",
              "car6_20 ~ ind_z + frg_z + evd + evd:ind_z + evd:frg_z + car1_5 + evd:car1_5",  # 흐름 효과 vs CAR(1,5) 단기반전
              "car1_5 ~ ind_z + frg_z + evd + evd:ind_z + evd:frg_z"]:
        md, mm = ols(f, d, "date"), ols(f, d, "month")
        print(f"  {f}  (N={int(md.nobs):,}, R2={md.rsquared:.4f})")
        print(pd.DataFrame({"coef": md.params, "t(일)": md.tvalues, "t(월)": mm.tvalues}).to_string(float_format="%.3f"))
    return out


def plot(dq, n, path, n_tk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    ks = np.arange(1, H + 1)
    cols = {"T1": "#86b6ef", "T2": "#3987e5", "T3": "#104281"}          # 순차 파랑(약→강)
    names = {"T1": "개인 순매도 쪽(T1)", "T2": "중간(T2)", "T3": "개인 순매수 쪽(T3)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#898781", lw=0.8)
    ax.axvspan(0.5, 5.5, color="#e1e0d9", alpha=0.45, lw=0)
    ax.text(3, 0.97, "흐름 측정 창(t+1~5)", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#52514e")
    nb = n[[f"car{k}" for k in ks]].mean().values * 100
    ax.plot(ks, nb, color="#52514e", lw=1.5, ls="--")
    ax.annotate(f"비이벤트 (N={len(n):,})", (H, nb[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color="#52514e")
    for q in ["T1", "T2", "T3"]:
        x = dq[dq.q == q]
        y = x[[f"car{k}" for k in ks]].mean().values * 100
        ax.plot(ks, y, color=cols[q], lw=2, marker="o", ms=3.5, label=f"{names[q]} N={len(x)}")
        ax.annotate(names[q], (H, y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xlim(0.5, H + 6.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("형성일 이후 거래일 k")
    ax.set_ylabel("평균 CAR(1,k), %  (200종목 동일가중 대비)")
    ax.set_title("'눈에 띄는 차트' 이벤트 뒤 누적 초과수익: 개인 순매수(t+1~5) 3분위별", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center", bbox_to_anchor=(0.55, 0.7))
    fig.text(0.01, 0.01, f"흐름 자료가 있는 2004-12 KOSPI 시총 상위 {n_tk}종목, 2005~2012 주간 형성. 이벤트 = 200종목 단면에서 과거 5일 수익률·비정상 거래대금 각 상위 10%.",
             fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    print("그림 →", path)


def check():
    """미래 정보 누출 점검: 임의 형성일 t 이후를 NaN으로 지워도 t의 이벤트·변수가 그대로여야 한다."""
    C, V, VOL, *_ = load()
    full = features(C, V, VOL)
    for t in pd.DatetimeIndex(form_dates(C.index))[[30, 150, 300]]:
        cut = [X.where(X.index.to_series().le(t), np.nan) for X in (C, V, VOL)]
        part = features(*cut)
        for a, b in zip(full, part):
            pd.testing.assert_series_equal(a.loc[t], b.loc[t], check_names=False)
    print("누출 점검 통과: t 이후 데이터가 t의 이벤트 판정에 영향 없음")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
        sys.exit()
    pd.set_option("display.width", 200)
    df, inv, V = panel()
    quality(inv, V)
    out = analyze(df, "ev")
    analyze(df, "ev_alt")
    plot(out[("ind", "이벤트")], df[~df.ev], ROOT / "results" / "a2_car_paths.png", len(inv))

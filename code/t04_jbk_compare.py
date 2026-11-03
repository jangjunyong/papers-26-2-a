"""t04: JBK(2026) 비교용 사후 참고 계산 (판정 아님, 2026-11-03). 시험 구간 형성일 729개. 결과: results/t04/t04.json

1 비용 사다리: 주 모델, 10분위 동일가중 롱숏(bt.run, 주간 전량 교체)에 거래세·수수료 → 스프레드를 차례로 더한 샤프
2 같은 틀 비교(JBK 표 4·7): 10분위 롱숏·비용 0, 신호(CNN 단독·주 모델) x 범위(전 종목·후보) x 매수 시점 x 가중(동일·시총)
3 크기 비교: 코스피 거래대금 하위 20% vs 코스닥(동아리 DB club_db_ext), 코스피+코스닥 시총 상위 500
4 묶음별 단순 반전(직전 5일 수익의 역순) 10분위 동일가중 롱숏 샤프: 코스피 하위 20%·후보, 코스닥 시총 상위 150·나머지.
  코스닥은 CNN 점수·수급이 없어 단순 신호만 쓴다. 수익은 형성일 종가 → 5거래일 뒤 종가(수정주가), 상폐로 잘린 행은 제외,
  코스닥 원자료가 정제 전이라 두 시장 모두 날짜별 1·99% 절단. 비용 0.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bt  # noqa: E402
import m01_gbm_2x2 as m01  # noqa: E402
import t01_test_run as t01  # noqa: E402

OUT = t01.OUT.parent / "t04"
KQ = m01.DATA / "daily" / "club_db_ext"
SR = lambda x: float(x.mean() / x.std() * np.sqrt(52))  # noqa: E731  연환산(√52)


def ls(d, sig, ret, vw=False):
    """날짜별 sig 상위 10% 매수 − 하위 10% 매도, 주별 수익 → (주당 평균 %, 샤프)."""
    d = d.dropna(subset=[sig, ret, "mktcap"]).copy()
    q = d.groupby("date")[sig].rank(pct=True)
    d["leg"] = np.where(q > 0.9, 1, np.where(q <= 0.1, -1, 0))
    d = d[d.leg != 0]
    d["w"] = d.mktcap if vw else 1.0
    leg = d.groupby(["date", "leg"]).apply(lambda g: np.average(g[ret], weights=g.w), include_groups=False).unstack()
    x = leg[1] - leg[-1]
    return dict(mean_pct=round(float(x.mean() * 100), 3), sharpe=round(SR(x), 2), weeks=int(len(x)))


def daily(d):
    """일별 긴 표(ticker, date, adj_close, cap, tv) → 직전 5일 수익·5일 뒤 수익·60일 평균 거래대금."""
    d = d.sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker")
    d["past5"] = d.adj_close / g.adj_close.shift(5) - 1
    d["fwd5"] = g.adj_close.shift(-5) / d.adj_close - 1
    d["tv60"] = g.tv.transform(lambda s: s.rolling(60, min_periods=20).mean())
    return d


def main():
    dates = pd.read_parquet(t01.OUT / "scores.parquet", columns=["date"]).date.unique()
    sc = pd.read_parquet(t01.OUT / "scores.parquet", columns=["ticker", "date", t01.MAIN])
    panel = pd.read_parquet(t01.PANEL)
    out = {}

    # 1 비용 사다리
    s1 = sc.rename(columns={t01.MAIN: "score"})
    M = bt.prep(panel, s1.ticker.unique())
    out["1_cost_ladder"] = {}
    for name, C, sp in [("none", (), "cs"), ("fee", ("fee",), "cs"), ("fee+spread_tick", ("fee", "spread"), "tick"),
                        ("fee+spread_cs", ("fee", "spread"), "cs")]:
        r = bt.run(s1, M, method="a", spread=sp, constraints=C)
        out["1_cost_ladder"][name] = dict(mean_pct=round(float(r.excess.mean() * 100), 3), sharpe=round(SR(r.excess), 2))
    print(out["1_cost_ladder"], flush=True)

    # 2 같은 틀 비교
    F = pd.read_parquet(t01.FEAT_OPEN, columns=["ticker", "date", "avg_tv60", "mktcap", "label_ret5", "fwd_ret5_panel"])
    F = F[F.date.isin(dates)]
    pr = pd.concat([pd.read_parquet(f) for f in sorted(m01.PRED.glob("*/pred_seed*.parquet"))])
    pr["date"] = pd.to_datetime(pr.date)
    F = F.merge(pr.groupby(["ticker", "date"]).prob.mean().rename("cnn").reset_index(), on=["ticker", "date"], how="left")
    F = F.merge(sc, on=["ticker", "date"], how="left")
    F["cand"] = F.groupby("date").avg_tv60.rank(pct=True) > m01.LIQ_CUT
    out["2_jbk_frame"] = {}
    for sname, sig, cand_only in [("cnn_all", "cnn", False), ("cnn_cand", "cnn", True), ("main_cand", t01.MAIN, True)]:
        d = F[F.cand] if cand_only else F
        for tname, ret in [("close", "fwd_ret5_panel"), ("next_open", "label_ret5")]:
            out["2_jbk_frame"][f"{sname}|{tname}"] = dict(ew=ls(d, sig, ret), vw=ls(d, sig, ret, vw=True))
    print(json.dumps(out["2_jbk_frame"]), flush=True)

    # 3·4 코스피·코스닥 일별 표
    k = panel[["ticker", "date", "adj_close", "mktcap", "trading_value", "volume"]].rename(columns={"mktcap": "cap", "trading_value": "tv"})
    k = daily(k)
    tr = pd.read_parquet(KQ / "trading_kosdaq.parquet", columns=["code", "date", "trading_value"])
    q = pd.read_parquet(KQ / "price_kosdaq.parquet", columns=["code", "date", "close", "adj_close", "volume", "listed_shares"])
    q = q.merge(tr, on=["code", "date"], how="left").rename(columns={"code": "ticker", "trading_value": "tv"})
    q["cap"] = q.close * q.listed_shares
    q = daily(q[["ticker", "date", "adj_close", "cap", "tv", "volume"]])
    k = k[k.date.isin(dates)].dropna(subset=["tv60", "cap"])
    q = q[q.date.isin(dates)].dropna(subset=["tv60", "cap"])
    k["low20"] = k.groupby("date").tv60.rank(pct=True) <= m01.LIQ_CUT
    q["top150"] = q.groupby("date").cap.rank(ascending=False) <= 150

    # 3 크기 비교 (형성일별 값의 중앙값)
    rows = []
    for dt, g in k.groupby("date"):
        h = q[q.date == dt]
        if h.empty:
            continue
        lo, cut = g[g.low20], g[~g.low20].tv60.min()
        top500 = pd.concat([g.assign(mkt="KOSPI"), h.assign(mkt="KOSDAQ")]).nlargest(500, "cap")
        kq100 = h.nlargest(100, "cap")
        rows.append(dict(kospi_low20_cap=lo.cap.median(), kospi_low20_tv60=lo.tv60.median(),
                         kospi_cand_cap=g[~g.low20].cap.median(), kospi_cand_tv60=g[~g.low20].tv60.median(),
                         kosdaq_top100_cap=kq100.cap.median(), kosdaq_top100_tv60=kq100.tv60.median(),
                         kosdaq_share_pass_filter=(h.tv60 > cut).mean(), top500_cap_cut=top500.cap.min(),
                         top500_n_kosdaq=(top500.mkt == "KOSDAQ").sum(),
                         kospi_cand_n=(~g.low20).sum(), kospi_cand_in_top500=g[~g.low20].ticker.isin(top500.ticker).sum()))
    out["3_size_median"] = {c: float(v) for c, v in pd.DataFrame(rows).median().items()}
    print(out["3_size_median"], flush=True)

    # 4 묶음별 단순 반전
    groups = {"kospi_low20": k[k.low20], "kospi_cand": k[~k.low20], "kosdaq_top150": q[q.top150], "kosdaq_rest": q[~q.top150]}
    out["4_reversal_by_group"] = {}
    for name, g in groups.items():
        g = g[(g.volume > 0)].dropna(subset=["past5", "fwd5"]).copy()
        lo_, hi_ = g.groupby("date").fwd5.transform(lambda s: s.quantile(0.01)), g.groupby("date").fwd5.transform(lambda s: s.quantile(0.99))
        g["fwd5"] = g.fwd5.clip(lo_, hi_)
        g["sig"], g["mktcap"] = -g.past5, g.cap
        out["4_reversal_by_group"][name] = dict(n_per_date=round(float(g.groupby("date").size().mean())),
                                                tv60_median_eok=round(float(g.tv60.median() / 1e8), 1), **ls(g, "sig", "fwd5"))
    print(json.dumps(out["4_reversal_by_group"], ensure_ascii=False), flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "t04.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

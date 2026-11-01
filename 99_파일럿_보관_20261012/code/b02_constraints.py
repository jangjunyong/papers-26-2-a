"""RQ4 제약별 수익 분해 실행 스크립트 (탐색용, 인프라 검증 포함). 실행: python b02_constraints.py  (노트북 CPU 수 분)
(A) CNN 파일럿 예측 2011-01~2012-11 (공매도 전면금지 2011-08-10~11-09 주는 별도 표시)
(B) 대조용 단순 지표 신호 2015~2026: 6지표 순위평균, 5일 반전. CNN이 아니므로 인프라 검증용이다.
출력: results/b01_tables/*.csv. 가정은 b01_backtest.py 머리말과 보고서(b01_constraints_memo.md)에 적었다.
선행 조건: b01_test.py가 전부 통과해야 결과를 믿는다."""
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b01_backtest as bt

ROOT = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS/26-2/03_차트CNN_연구"
PRED = f"{ROOT}/results/kaggle/pred_d_lr1e-4.parquet"
OUT = f"{ROOT}/results/b01_tables"
PILOT = dict(hl_weekly_mean=0.010609117162449545, hl_sharpe=5.406191021421821)   # results/kaggle/results.json d_lr1e-4 (종가 체결, 비용 없음)
PPY_A, PPY_B = 52.0, 252 / 5
B_REG = [("전체 2015~", "2015-01-01", "2026-12-31"),
         ("2015~2019 전면허용", "2015-01-01", "2019-12-31"),
         ("2020-01~03-15 전면허용(코로나 급락)", "2020-01-01", "2020-03-15"),
         ("2020-03-16~2021-05-02 전면금지", "2020-03-16", "2021-05-02"),
         ("2021-05-03~2023-11-05 부분허용(전면허용 처리)", "2021-05-03", "2023-11-05"),
         ("2023-11-06~2025-03-30 전면금지", "2023-11-06", "2025-03-30"),
         ("2025-03-31~ 전면허용", "2025-03-31", "2026-12-31")]
LAB = {m: ("+".join(bt.subset(m)) or "공집합") for m in range(bt.N_SUB)}
FULL, NO_C1 = bt.N_SUB - 1, bt.N_SUB - 1 - 1        # 전부 / c1만 뺀 나머지(롱숏 유지)
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    df.to_csv(f"{OUT}/{name}.csv", index=False, encoding="utf-8-sig")
    print(f"  -> {name}.csv ({len(df)}행)")


def decompose(sample, period, R, ppy, B=500, block=4, seed=20261010):
    """R: (T x 32) 순수익. 반환: v32 행, 샤플리 행."""
    R = np.asarray(R, float)
    T = len(R)
    vm = R.mean(0)
    vs = np.array([bt.sharpe(R[:, m], ppy) for m in range(bt.N_SUB)])
    gap = max(bt.check_efficiency(vm), bt.check_efficiency(vs))
    v_rows = [dict(sample=sample, period=period, mask=m, cons=LAB[m], n_weeks=T, mean_bp=vm[m] * 1e4, sharpe=vs[m])
              for m in range(bt.N_SUB)]
    if T >= 20:
        mb, sb = bt.block_boot(R, ppy, B=B, block=block, seed=seed)
        pm, ps = mb @ bt.W5.T, sb @ bt.W5.T                              # (B x 5)
    else:
        mb = sb = pm = ps = None
    s_rows = []
    for metric, v, bv, bphi, sc in (("mean_bp", vm, mb, pm, 1e4), ("sharpe", vs, sb, ps, 1.0)):
        phi, mg = bt.shapley(v), bt.marginals(v)

        def ci(x):
            return (np.nan, np.nan) if x is None else tuple(np.percentile(x, [2.5, 97.5]) * sc)
        for i, c in enumerate(bt.CONS):
            lo, hi = ci(None if bphi is None else bphi[:, i])
            s_rows.append(dict(sample=sample, period=period, n_weeks=T, metric=metric, item=c, name=bt.NAME[c], value=phi[i] * sc,
                               ci_lo=lo, ci_hi=hi, first=mg[c]["first"] * sc, last=mg[c]["last"] * sc,
                               marg_min=mg[c]["min"] * sc, marg_max=mg[c]["max"] * sc))
        for item, idx in (("v_empty", 0), ("v_noC1", NO_C1), ("v_all", FULL)):
            lo, hi = ci(None if bv is None else bv[:, idx])
            s_rows.append(dict(sample=sample, period=period, n_weeks=T, metric=metric, item=item, name=LAB[idx], value=v[idx] * sc,
                               ci_lo=lo, ci_hi=hi))
        s_rows.append(dict(sample=sample, period=period, n_weeks=T, metric=metric, item="sum_phi_minus_(v_all-v_empty)",
                           name="효율성 검증(0이어야 함)", value=(phi.sum() - (v[FULL] - v[0])) * sc))
    return v_rows, s_rows, gap


def portfolios(sample, period, tr, sel, ppy):
    """3종 포트폴리오: 비용·체결제약 전(gross)과 후(c3+c4+c5; CS 스프레드 / 틱/2 하한). 롱온리는 시장 대비 초과와 롱 레그 총수익을 모두 보인다."""
    rows = []
    for pf, base in (("10분위 롱숏", ()), ("10분위 롱온리", ("c1",)), (f"상위20 롱온리", ("c1", "c2"))):
        for stage, extra, kw in (("gross(체결·비용 없음)", (), {}), ("net(c3+c4+c5)", ("c3", "c4", "c5"), {}),
                                 ("net(c3+c4+c5), 스프레드=틱/2", ("c3", "c4", "c5"), dict(spread="tick"))):
            r = bt.run(tr, base + extra, **kw)[sel]
            p = bt.perf(r["ret"], ppy)
            rows.append(dict(sample=sample, period=period, portfolio=pf, stage=stage, n_weeks=p["n"], mean_bp=p["mean"] * 1e4,
                             ann_ret=p["ann_ret"], sharpe=p["sharpe"], t=p["t"], mdd=p["mdd"], hit=p["hit"],
                             long_leg_raw_bp=r["long"].mean() * 1e4 if "c1" in base else np.nan,
                             market_bp=r["mkt"].mean() * 1e4 if "c1" in base else np.nan,
                             avg_cost_bp=r["cost"].mean() * 1e4, avg_nL=r["nL"].mean(), avg_nS=r["nS"].mean()))
    return rows


def variants(sample, tr_open, tr_close, ppy, sel_fn):
    """민감도: 수수료·스프레드 배율 격자와 기타(틱 하한 스프레드, 대차 4%, 매매 상쇄, 종가 체결, 롱온리 총수익 기준)."""
    base_kw = dict()
    runs = [("기본", tr_open, base_kw)]
    for fm in (0.5, 1, 2):
        for sm in (0.5, 1, 2):
            if (fm, sm) != (1, 1):
                runs.append((f"수수료x{fm} 스프레드x{sm}", tr_open, dict(fee=bt.FEE * fm, spread_mult=sm)))
    runs += [("스프레드=틱/2(하한 근사)", tr_open, dict(spread="tick")),
             ("대차 연 4%", tr_open, dict(borrow=0.04)),
             ("매매 상쇄(유지 종목 비용 0)", tr_open, dict(net_turnover=True)),
             ("롱온리를 시장 차감 없는 총수익으로", tr_open, dict(c1_mode="raw")),
             ("형성일 종가 체결", tr_close, base_kw)]
    rows = []
    for name, tr, kw in runs:
        R = bt.all_subsets(tr, **kw)
        R = R[sel_fn(tr)]
        vm = R.mean().to_numpy()
        vs = np.array([bt.sharpe(R[m], ppy) for m in range(bt.N_SUB)])
        phi = bt.shapley(vm)
        rows.append(dict(sample=sample, variant=name, n_weeks=len(R), v_empty_bp=vm[0] * 1e4, v_empty_sharpe=vs[0],
                         v_noC1_bp=vm[NO_C1] * 1e4, v_noC1_sharpe=vs[NO_C1], v_all_bp=vm[FULL] * 1e4, v_all_sharpe=vs[FULL],
                         **{f"phi_{c}_bp": phi[i] * 1e4 for i, c in enumerate(bt.CONS)}))
    return rows


def diagnostics(sample, mk, tr, ppy):
    r0 = bt.run(tr, ())
    gaps = np.diff(mk.cal.get_indexer(tr.dates))
    n_ok = tr.ok.sum()
    return dict(sample=sample, n_batches=tr.G, avg_universe=tr.n_g.mean(), rows_dropped=tr.n_dropped_rows, ok_frac=tr.ok.mean(),
                stale_exit_trades=int(tr.stale.sum()), overlap_batches=int((gaps < 5).sum()), ban_batches=int(tr.ban.sum()),
                buy_blocked_frac=float(tr.blk_buy[tr.ok].mean()), short_blocked_frac=float(tr.blk_sell_short[tr.ok].mean()),
                long_exit_carried_frac=float(tr.carried_l[tr.ok].mean()), short_exit_carried_frac=float(tr.carried_s[tr.ok].mean()),
                hs_median_pct=float(np.median(tr.hs[tr.ok]) * 100), hs_mean_pct=float(tr.hs[tr.ok].mean() * 100),
                tick_half_median_pct=float(np.median(tr.hs_tick[tr.ok]) * 100), avg_nL=r0.nL.mean(), avg_nS=r0.nS.mean(),
                mean_mkt_bp=tr.mkt.mean() * 1e4)


def retention(tr):
    """연속 형성일 간 같은 레그에 다시 편입된 종목 비율(롱 레그, 제약 없음 기준)."""
    k = np.maximum(1, np.floor(0.1 * tr.n_g + 0.5).astype(int))[tr.gid]
    m = (tr.rk_d <= k) & tr.ok
    key = tr.gid[m].astype(np.int64) * tr.NT + tr.tick_idx[m]
    prev = (tr.gid[m] - 1).astype(np.int64) * tr.NT + tr.tick_idx[m]
    return float(np.isin(prev, key)[tr.gid[m] > 0].mean())


def cs_bias_sim(seed=7, ndays=500, nstk=40, steps=390):
    """CS 추정의 하한 0 절단 편향 점검(시뮬레이션): 진짜 반스프레드를 알고 있는 가상 종목에서 추정값을 비교한다.
    일중 390구간(분 단위) 랜덤워크 + 야간 갭(표준편차 0 또는 1%) + 체결가 = 중간가 +- 반스프레드(무작위 방향)."""
    rng = np.random.default_rng(seed)
    rows = []
    for gap, sig, half in [(g, s, h) for g in (0.0, 0.01) for s in (0.015, 0.025) for h in (0.0005, 0.001, 0.0025)]:
        inc = rng.normal(0, sig / math.sqrt(steps), (nstk, ndays, steps))
        move = inc.sum(2)
        start = np.cumsum(rng.normal(0, gap, (nstk, ndays)) + move, axis=1) - move + math.log(10000.0)
        px = start[..., None] + np.cumsum(inc, axis=2) + half * rng.choice([-1, 1], inc.shape)
        H, L = np.exp(px.max(2)).T, np.exp(px.min(2)).T
        ok = np.ones_like(H, bool)
        rows.append(dict(daily_vol=sig, overnight_gap_sd=gap, true_half_spread_pct=half * 100,
                         cs_half_floor0_pct=np.nanmean(bt.cs_spread(H, L, ok)) / 2 * 100,
                         cs_half_unfloored_pct=np.nanmean(bt.cs_spread(H, L, ok, floor=False)) / 2 * 100))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    save(cs_bias_sim(), "cs_bias_sim")
    all_v, all_s, all_p, all_var, diag = [], [], [], [], []

    # ------------------------------------------------------------ (A) CNN 파일럿
    print("(A) CNN 파일럿")
    P = bt.load_panel("2010-06-01", "2013-03-01")
    mk = bt.Market(P)
    pr = pd.read_parquet(PRED).rename(columns={"prob": "score"})[["date", "ticker", "score"]]
    trA, trAc = bt.make_trades(mk, pr), bt.make_trades(mk, pr, entry="close")
    allw, ban = np.ones(trA.G, bool), trA.ban
    per = {"전체": allw, "공매도 금지 주 제외": ~ban, "공매도 금지 주(2011-08~11)": ban}
    R = bt.all_subsets(trA)
    for pname, sel in per.items():
        v, s, gap = decompose("A_CNN파일럿", pname, R[sel].to_numpy(), PPY_A)
        all_v += v
        all_s += s
        all_p += portfolios("A_CNN파일럿", pname, trA, sel, PPY_A)
    all_var += variants("A_CNN파일럿", trA, trAc, PPY_A, lambda tr: allw)
    d = diagnostics("A_CNN파일럿", mk, trA, PPY_A)
    d["retention_long_leg"] = retention(trA)
    diag.append(d)
    # 파일럿 재현 검사: 종가 체결·비용 없음·10분위 롱숏 -> 파일럿 results.json과 같아야 한다
    rc = bt.run(trAc, ())["ret"]
    rep = pd.DataFrame([dict(item="이 엔진(종가 체결, 제약 없음)", weekly_mean=rc.mean(), sharpe=bt.sharpe(rc, 52)),
                        dict(item="파일럿 results.json d_lr1e-4", weekly_mean=PILOT["hl_weekly_mean"], sharpe=PILOT["hl_sharpe"]),
                        dict(item="이 엔진(익일 시가 체결, 제약 없음)", weekly_mean=bt.run(trA, ())["ret"].mean(),
                             sharpe=bt.sharpe(bt.run(trA, ())["ret"], 52))])
    save(rep, "A_replication_pilot")
    print(rep.to_string(index=False), f"[{time.time() - t0:.0f}s]")

    # ------------------------------------------------------------ (B) 단순 지표 신호
    print("(B) 단순 지표 신호 2015~")
    P = bt.load_panel("2014-09-01", extra=list(bt.SIGN))
    mk = bt.Market(P)
    dates = bt.rebalance_dates(mk.cal, "2015-01-01")
    for sname, fn in (("B_6지표순위평균", bt.signal_rankavg), ("B_5일반전", bt.signal_rev5)):
        sig = fn(P, dates)
        tr, trc = bt.make_trades(mk, sig), bt.make_trades(mk, sig, entry="close")
        R = bt.all_subsets(tr)
        ent = tr.entry_dates
        for pname, a, b in B_REG:
            sel = np.asarray((ent >= a) & (ent <= b))
            if sel.sum() < 5:
                continue
            v, s, gap = decompose(sname, pname, R[sel].to_numpy(), PPY_B)
            all_v += v
            all_s += s
            if pname.startswith("전체") or pname.startswith("2015~2019"):
                all_p += portfolios(sname, pname, tr, sel, PPY_B)
        all_var += variants(sname, tr, trc, PPY_B, lambda t: np.ones(t.G, bool))
        d = diagnostics(sname, mk, tr, PPY_B)
        d["retention_long_leg"] = retention(tr)
        diag.append(d)
        print(f"  {sname} 완료 [{time.time() - t0:.0f}s]")

    for df, name in ((pd.DataFrame(all_v), "v32"), (pd.DataFrame(all_s), "shapley"), (pd.DataFrame(all_p), "portfolios"),
                     (pd.DataFrame(all_var), "sensitivity"), (pd.DataFrame(diag), "diagnostics")):
        save(df, name)
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()

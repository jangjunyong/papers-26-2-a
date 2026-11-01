"""t01: 시험 구간(2012-06-01~2026-05-25) 단 한 번 실행 (사전고정_v1 7절).

순서: 커밋 확인 → 라벨 보호 해제한 특성 → m01 2×2 판정(H1·H2) → bt 순수익(RQ3, 모델 3개 × 구현 5종 × 스프레드 3)
      → 실현 가능성 분류 → 제약 샤플리 분해(구현 5종) → 매크로 국면(m02) → RQ4 진단 → results/t01/
조건: 작업 트리가 깨끗하고(code/·사전고정_v1.md 변경 없음) CNN 시드 5개 예측이 있어야 한다. 결과 폴더가 이미 있으면 멈춘다.
다시 돌려야 하면(버그) 이전 결과 폴더 이름을 바꿔 남기고, 이유를 사전고정_v1 변경 이력에 적은 뒤 --rerun 으로 실행한다.
사용: py -3 t01_test_run.py [--rerun]
"""
import itertools
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bt  # noqa: E402
import f01_build_features as f01  # noqa: E402
import m01_gbm_2x2 as m01  # noqa: E402
import m02_regimes as m02  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "t01"
PANEL = f01.D / "daily" / "club_db" / "panel_v3.parquet"
FEAT_OPEN = f01.OUT_DIR / "features_v3_kospi_unlocked.parquet"
MAIN, COMPARE = "P+C+F", ["P+F", "P+C"]
METHODS = [("a", None), ("b", 40), ("b", 60), ("b", 100), ("c", None)]
SPREADS = ["cs", "tick", "zero"]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def unlocked_features():
    f01.PANEL = PANEL
    panel, cls, fsav, fscore = f01.load_inputs()
    F = f01.build(panel, cls, fsav, fscore, label_cap=pd.Timestamp("2026-12-31"), inplace=True)
    for c in f01.RK + ["label_rk"]:
        F[c] = F[c].astype(np.float32)
    F.to_parquet(FEAT_OPEN, index=False)
    return FEAT_OPEN


def ci(x):
    p, c = m01.block_boot_p(np.asarray(x.dropna(), float))
    return dict(mean=float(x.mean()), ci95=c, p_le0=p, weeks=int(x.notna().sum()))


def classify(r):
    """보고서 3.4: 두 스프레드(cs·tick) 모두 하한>0 → 실현 가능, tick 만 → 조건부, 그 밖 → 남지 않음. 벤치마크 2개 중 하나라도 미달이면 한 단계 낮춤."""
    lvl = {}
    for b in ["vw", "ew"]:
        ok = {s: r[s][f"ex_{b}"]["ci95"][0] > 0 for s in ["cs", "tick"]}
        lvl[b] = 2 if all(ok.values()) else 1 if ok["tick"] else 0
    names = ["남지 않음", "조건부", "실현 가능"]
    best = max(lvl.values())
    return names[best - 1 if min(lvl.values()) < best and best > 0 else best]


def car_6_20(panel, F):
    """RQ4: 형성일 기준 6~20거래일 수정종가 누적수익 − 같은 창 후보 동일가중 평균. 상위·하위 10분위(5일 순매수 순위)."""
    p = panel[["ticker", "date", "adj_close"]].sort_values(["ticker", "date"])
    g = p.groupby("ticker").adj_close
    p["car"] = g.shift(-20) / g.shift(-5) - 1
    d = F.merge(p[["ticker", "date", "car"]], on=["ticker", "date"], how="left")
    d["car_ex"] = d.car - d.groupby("date").car.transform("mean")
    out = {}
    for grp in ["indiv", "inst", "for"]:
        rk = d[f"flow_{grp}_5_rk"]
        w = pd.DataFrame({"top": d.car_ex.where(rk >= 0.4), "bot": d.car_ex.where(rk < -0.4), "date": d.date}).groupby("date").mean()
        out[grp] = {"top10": ci(w.top), "bottom10": ci(w.bot), "top_minus_bottom": ci(w.top - w.bot)}
    return out


def main():
    rerun = "--rerun" in sys.argv
    assert not (OUT.exists() and not rerun), f"{OUT} 가 이미 있다. 시험 구간은 한 번만 실행한다(사전고정 7절)"
    dirty = [l for l in git("status", "--porcelain").splitlines() if l[3:].startswith(("code/", "사전고정_v1.md"))]
    assert not dirty, f"커밋 안 된 변경: {dirty}"
    head = git("rev-parse", "HEAD")
    assert head, "커밋이 없다"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = dict(commit=head, started=time.strftime("%Y-%m-%d %H:%M:%S"), window=m01.WINDOWS["test"])

    # 1. 2×2 판정
    m01.FEAT = unlocked_features()
    s, ic = m01.run("test")
    res = m01.judge(ic)
    s.to_parquet(OUT / "scores.parquet", index=False)
    ic.to_csv(OUT / "ic_weekly.csv")
    (OUT / "h1h2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)

    # 2. 순수익(RQ3)
    panel = pd.read_parquet(PANEL)
    M = bt.prep(panel, s.ticker.unique())
    weekly, summ = {}, {}
    for model in [MAIN] + COMPARE:
        sc = s[["date", "ticker", model]].rename(columns={model: "score"})
        for (meth, x), sp in itertools.product(METHODS, SPREADS):
            key = f"{model}|{meth}{x or ''}|{sp}"
            r = bt.run(sc, M, method=meth, exit_rank=x, spread=sp)
            r["ex_vw"], r["ex_ew"] = r.net - r.bench_vw, r.net - r.bench_ew
            weekly[key] = r.set_index("date")
            summ[key] = {"ex_vw": ci(r.ex_vw), "ex_ew": ci(r.ex_ew), "turnover": float(r.turnover.mean()), "cost": float(r.cost.mean())}
            print(key, round(summ[key]["ex_vw"]["mean"], 5), flush=True)
    pd.concat(weekly, names=["key"]).to_parquet(OUT / "bt_weekly.parquet")
    cls_ = {f"{m}|{meth}{x or ''}": classify({sp: summ[f"{m}|{meth}{x or ''}|{sp}"] for sp in ["cs", "tick"]})
            for m in [MAIN] + COMPARE for meth, x in METHODS}
    # 주 모델 − 비교 모델 중 시험 구간 평균이 높은 쪽 (구현 방식별, cs 스프레드)
    vs = {}
    for meth, x in METHODS:
        k = f"{meth}{x or ''}"
        comp = max(COMPARE, key=lambda m: weekly[f"{m}|{k}|cs"].net.mean())
        vs[k] = dict(compare=comp, **ci(weekly[f"{MAIN}|{k}|cs"].net - weekly[f"{comp}|{k}|cs"].net))
    (OUT / "rq3.json").write_text(json.dumps(dict(summary=summ, classification=cls_, main_vs_best_compare=vs),
                                             ensure_ascii=False, indent=1), encoding="utf-8")

    # 3. 제약 샤플리 분해 (주 모델, cs, 구현 5종)
    sc = s[["date", "ticker", MAIN]].rename(columns={MAIN: "score"})
    shap = {}
    for meth, x in METHODS:
        v = {}
        for r_ in range(6):
            for S in itertools.combinations(sorted(bt.ALL), r_):
                v[frozenset(S)] = float(bt.run(sc, M, method=meth, exit_rank=x, spread="cs", constraints=S).excess.mean())
        shap[f"{meth}{x or ''}"] = dict(values=bt.shapley(v), v_none=v[frozenset()], v_all=v[bt.ALL])
        print("shapley", meth, x, shap[f"{meth}{x or ''}"], flush=True)
    (OUT / "shapley.json").write_text(json.dumps(shap, ensure_ascii=False, indent=1), encoding="utf-8")

    # 4. 매크로 국면 (판정 없음)
    wk = pd.DataFrame({"H1": ic[MAIN] - ic["P+F"], "H2": ic[MAIN] - ic["P+C"],
                       "net_ex_vw_a_cs": weekly[f"{MAIN}|a|cs"].ex_vw})
    m02.by_regime(wk).to_csv(OUT / "regimes.csv")

    # 5. RQ4 진단 (판정 없음)
    Fo = pd.read_parquet(FEAT_OPEN, columns=["ticker", "date"] + [f"flow_{g}_5_rk" for g in ["indiv", "inst", "for"]])
    t_lo, t_hi = map(pd.Timestamp, m01.WINDOWS["test"])
    (OUT / "rq4.json").write_text(json.dumps(car_6_20(panel, Fo[(Fo.date >= t_lo) & (Fo.date <= t_hi)]),
                                             ensure_ascii=False, indent=1), encoding="utf-8")

    meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=round((time.time() - t0) / 60, 1))
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("끝", meta, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

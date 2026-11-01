"""m01: 2×2 GBM(P, P+C, P+F, P+C+F) 확장 창 학습 → 형성일별 점수, 주별 순위 IC, H1·H2 판정 (사전고정_v1 2~4절).

모드
  dry  : 배관 점검. 시험 구간을 쓰지 않는다 — 가짜 시험 2011-06-01~2012-05-25, 학습 2010-01~(엠바고 25거래일).
         CNN 예측(results/kaggle_v3/*/pred_seed*.parquet)이 없으면 C 를 무작위로 채운다. 결과는 판단에 쓰지 않는다.
  test : 사전 고정 커밋 뒤 t01_test_run.py 만 부른다. 시험 2012-06-01~2026-05-25, CNN 시드 5개 필수.
하이퍼파라미터는 고정값(조정·조기 종료 없음). 라벨 = 후보 범위 안에서 label_ret5 의 같은 날 순위(−0.5~+0.5).
출력: results/m01_<모드>/{scores.parquet, ic_weekly.csv, summary.json}
사용: py -3 m01_gbm_2x2.py dry
"""
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parents[3] / "data"
FEAT = DATA / "features" / "features_v3_kospi.parquet"
PRED = ROOT / "results" / "kaggle_v3"

P = [c + "_rk" for c in ["str5", "str20", "mom12_1", "near52h", "vol20", "abvol", "log_mktcap", "log_tv60", "amihud20"]]
F = [f"flow_{g}_{w}_rk" for g in ["indiv", "inst", "for"] for w in [5, 20]]
C = ["cnn_rk"]
MODELS = {"P": P, "P+C": P + C, "P+F": P + F, "P+C+F": P + C + F}
HP = dict(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=200,
          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=2)
TRAIN_START = pd.Timestamp("2010-01-01")
EMBARGO = 25                     # 학습 라벨 청산일 ≤ 재학습 형성일의 25거래일 전
LIQ_CUT = 0.2                    # 60일 평균 거래대금 하위 20% 는 후보에서 뺀다
WINDOWS = {"dry": ("2011-06-01", "2012-05-25"), "test": ("2012-06-01", "2026-05-25")}
BLOCK, B_REPS, ALPHA = 8, 10_000, 0.05
N_SEEDS = 5


def load(mode):
    cols = ["ticker", "date", "avg_tv60", "label_ret5", "label_end_date"] + P + F
    d = pd.read_parquet(FEAT, columns=cols)
    files = sorted(PRED.glob("*/pred_seed*.parquet"))
    if mode == "test":
        assert len(files) == N_SEEDS, f"CNN 시드 {len(files)}개 (필요 {N_SEEDS})"
    if files:
        pr = pd.concat([pd.read_parquet(f).assign(seed=f.stem) for f in files])
        pr["date"] = pd.to_datetime(pr.date)
        prob = pr.groupby(["ticker", "date"]).prob.mean().rename("cnn_prob").reset_index()   # 시드 결합 = 확률 평균
        d = d.merge(prob, on=["ticker", "date"], how="left")
    else:
        print("[dry] CNN 예측 없음 → C 무작위", flush=True)
        d["cnn_prob"] = np.random.default_rng(0).random(len(d))
    d["cnn_rk"] = d.groupby("date").cnn_prob.rank(pct=True) - 0.5
    # 후보: 같은 날 60일 평균 거래대금 하위 20% 제외(값 없는 종목도 제외)
    q = d.groupby("date").avg_tv60.rank(pct=True)
    d = d[q > LIQ_CUT].copy()
    d["y"] = d.groupby("date").label_ret5.rank(pct=True) - 0.5
    return d


def run(mode):
    d = load(mode)
    cal = np.sort(d.date.unique())
    t0, t1 = map(pd.Timestamp, WINDOWS[mode])
    test_dates = pd.DatetimeIndex(cal[(cal >= t0) & (cal <= t1)])
    retrain = test_dates[np.r_[True, test_dates.year[1:] != test_dates.year[:-1]]]   # 첫 시험 형성일 + 매년 첫 형성일
    # 엠바고는 거래일 기준: 전체 거래일 달력이 아니라 형성 패널에는 주 단위만 있으므로 label_end_date 로 판정한다
    tdays = np.sort(pd.read_parquet(DATA / "daily" / "club_db" / "panel_v3.parquet", columns=["date"]).date.unique())
    out = []
    for i, r in enumerate(retrain):
        cut = tdays[np.searchsorted(tdays, r) - EMBARGO]
        nxt = retrain[i + 1] if i + 1 < len(retrain) else t1 + pd.Timedelta(days=1)
        tr = d[(d.date >= TRAIN_START) & (d.label_end_date <= cut) & d.y.notna()]
        te = d[(d.date >= r) & (d.date < nxt) & (d.date <= t1)].copy()
        assert tr.label_end_date.max() < r
        for name, cols in MODELS.items():
            m = lgb.LGBMRegressor(**HP).fit(tr[cols], tr.y)
            te[name] = m.predict(te[cols])
        out.append(te[["ticker", "date", "label_ret5"] + list(MODELS)])
        print(f"재학습 {pd.Timestamp(r).date()}: 학습 {len(tr):,}행(청산 ≤ {pd.Timestamp(cut).date()}), 예측 {len(te):,}행", flush=True)
    s = pd.concat(out, ignore_index=True)
    ic = s.dropna(subset=["label_ret5"]).groupby("date")[list(MODELS) + ["label_ret5"]].apply(
        lambda g: pd.Series({k: g[k].corr(g.label_ret5, method="spearman") for k in MODELS}))
    return s, ic


def block_boot_p(x, block=BLOCK, reps=B_REPS, seed=0):
    """이동 블록 부트스트랩 평균의 분포에서 0 이하 비율 = p (단측)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, k))
    idx = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
    m = x[idx].mean(axis=1)
    return float((m <= 0).mean()), np.percentile(m, [2.5, 97.5]).tolist()


def holm(p):
    """p: {이름: p값} → {이름: 기각 여부}. 작은 p 부터 α/m, α/(m−1) … 와 비교, 처음 실패하면 이후 전부 비기각."""
    out, ok = {}, True
    for j, (k, v) in enumerate(sorted(p.items(), key=lambda t: t[1])):
        ok = ok and v <= ALPHA / (len(p) - j)
        out[k] = ok
    return out


def judge(ic):
    diff = {"H1": ic["P+C+F"] - ic["P+F"], "H2": ic["P+C+F"] - ic["P+C"]}
    res = {}
    for h, x in diff.items():
        p, ci = block_boot_p(x.dropna())
        res[h] = dict(mean=float(x.mean()), p=p, ci95=ci, weeks=int(x.notna().sum()))
    for h, rej in holm({h: r["p"] for h, r in res.items()}).items():
        res[h]["holm_reject"] = rej
    res["interaction"] = float(((ic["P+C+F"] - ic["P+F"]) - (ic["P+C"] - ic["P"])).mean())
    res["mean_ic"] = ic.mean().round(5).to_dict()
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    assert mode in WINDOWS
    if mode == "test":
        assert "--from-t01" in sys.argv, "test 모드는 t01_test_run.py 로만 실행한다(사전고정_v1 7절)"
    # 자체 점검: Holm, 부트스트랩
    assert holm({"a": 0.01, "b": 0.04}) == {"a": True, "b": True} and holm({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}
    assert block_boot_p(np.full(50, 0.1))[0] == 0.0 and block_boot_p(np.full(50, -0.1))[0] == 1.0
    s, ic = run(mode)
    res = judge(ic)
    od = ROOT / "results" / f"m01_{mode}"
    od.mkdir(parents=True, exist_ok=True)
    s.to_parquet(od / "scores.parquet", index=False)
    ic.to_csv(od / "ic_weekly.csv")
    (od / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))

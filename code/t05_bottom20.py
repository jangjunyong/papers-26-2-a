"""t05: 거래대금 하위 20%(후보에서 뺀 종목)에 우리 모델을 적용한 사후 참고 계산 (판정 아님, 2026-11-03). 결과: results/t05/t05.json

주 모델(P+C+F)은 m01.run 과 똑같이 학습한다(후보만, 같은 재학습 일정·엠바고·하이퍼파라미터). 예측만 하위 20% 행까지 넓힌다.
자체 점검: 후보 행 예측이 t01 scores.parquet 의 P+C+F 와 같아야 한다.
백테스트는 하위 20% 종목만 넣은 bt.run(주간 전량 교체). 벤치마크·초과수익도 하위 20% 안에서 계산된다.
"""
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bt  # noqa: E402
import m01_gbm_2x2 as m01  # noqa: E402
import t01_test_run as t01  # noqa: E402

OUT = t01.OUT.parent / "t05"
SR = lambda x: float(x.mean() / x.std() * np.sqrt(52))  # noqa: E731
COLS = m01.MODELS[t01.MAIN]


def scores_all():
    """m01.run('test') 의 주 모델 학습을 그대로 따르되 하위 20% 행에도 예측한다."""
    m01.FEAT, cut0 = t01.FEAT_OPEN, m01.LIQ_CUT
    m01.LIQ_CUT = -1.0                                       # load() 의 필터를 끈다(전 종목)
    try:
        d = m01.load("test")
    finally:
        m01.LIQ_CUT = cut0
    d["cand"] = d.groupby("date").avg_tv60.rank(pct=True) > cut0
    d["y"] = d[d.cand].groupby("date").label_ret5.rank(pct=True) - 0.5      # 후보 안 순위(원래와 같다), 하위 20% 는 NaN
    d = d[d.avg_tv60.notna()]
    cal = np.sort(d.date.unique())
    t0, t1_ = map(pd.Timestamp, m01.WINDOWS["test"])
    test_dates = pd.DatetimeIndex(cal[(cal >= t0) & (cal <= t1_)])
    retrain = test_dates[np.r_[True, test_dates.year[1:] != test_dates.year[:-1]]]
    tdays = np.sort(pd.read_parquet(m01.DATA / "daily" / "club_db" / "panel_v3.parquet", columns=["date"]).date.unique())
    out = []
    for i, r in enumerate(retrain):
        cut = tdays[np.searchsorted(tdays, r.to_datetime64()) - m01.EMBARGO]
        nxt = retrain[i + 1] if i + 1 < len(retrain) else t1_ + pd.Timedelta(days=1)
        tr = d[d.cand & (d.date >= m01.TRAIN_START) & (d.label_end_date <= cut) & d.y.notna()]
        te = d[(d.date >= r) & (d.date < nxt) & (d.date <= t1_)].copy()
        te["score"] = lgb.LGBMRegressor(**m01.HP).fit(tr[COLS], tr.y).predict(te[COLS])
        out.append(te[["ticker", "date", "cand", "score", "cnn_prob"]])
    return pd.concat(out, ignore_index=True)


def main():
    s = scores_all()
    ref = pd.read_parquet(t01.OUT / "scores.parquet", columns=["ticker", "date", t01.MAIN])
    chk = s[s.cand].merge(ref, on=["ticker", "date"])
    gap = float((chk.score - chk[t01.MAIN]).abs().max())
    assert len(chk) == len(ref) and gap < 1e-9, f"후보 예측이 t01 과 다르다: {len(chk)} vs {len(ref)}, 최대 차 {gap}"
    print(f"자체 점검 통과: 후보 {len(chk):,}행 예측이 t01 과 같다(최대 차 {gap:.1e})", flush=True)

    low = s[~s.cand]
    M = bt.prep(pd.read_parquet(t01.PANEL), low.ticker.unique())
    runs = [("10분위 롱숏, 비용 0", (), "cs"), ("+ 거래세·수수료", ("fee",), "cs"),
            ("+ 스프레드(싼 가정)", ("fee", "spread"), "tick"), ("+ 스프레드(비싼 가정)", ("fee", "spread"), "cs"),
            ("주 설정(상위 20 매수만·제약 전부), 싼 가정", tuple(bt.ALL), "tick"),
            ("주 설정(상위 20 매수만·제약 전부), 비싼 가정", tuple(bt.ALL), "cs")]
    out = dict(n_per_date=round(float(low.groupby("date").size().mean())), self_check_max_gap=gap)
    for sig in ["score", "cnn_prob"]:
        sc = low[["date", "ticker", sig]].rename(columns={sig: "score"})
        out[sig] = {}
        for name, C, sp in runs:
            r = bt.run(sc, M, method="a", spread=sp, constraints=C)
            out[sig][name] = dict(mean_pct=round(float(r.excess.mean() * 100), 3), sharpe=round(SR(r.excess), 2),
                                  cost_pct=round(float(r.cost.mean() * 100), 3), turnover=round(float(r.turnover.mean()), 2))
            print(sig, name, out[sig][name], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "t05.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

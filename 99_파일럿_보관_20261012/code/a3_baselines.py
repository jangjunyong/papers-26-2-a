"""A3 베이스라인: CNN 파일럿과 같은 (date, ticker) 표본에서 단순 가격 지표가 얼마나 하는가.

표본 규칙은 images.build 그대로(주 마지막 거래일, 20일 창 안 거래정지 제외, 다음 5거래일 수익 라벨)이고
이미지만 만들지 않는다. 지표는 전부 형성일 p 종가까지의 정보만 쓴다.
점수 부호는 결과를 보기 전에 이론으로 정했다(아래 SIGN). 반대 부호 결과는 별도 열로만 보고한다.
사용: py -3 a3_baselines.py  → results/a3_baselines.csv, results/a3_corr_cnn.csv(A4 예측이 있을 때)
"""
import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "2"                      # A4가 CPU로 CNN을 학습 중이라 2스레드로 제한

import pathlib

import numpy as np
import pandas as pd

import cnn                                   # evaluate만 쓴다(torch 연산 없음)
import images

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily" / "ohlcv"
RES = ROOT / "results"
TRAIN, TEST = ("2005-01-01", "2010-11-30"), ("2011-01-01", "2012-11-30")

# 사전에 정한 이론 부호: 점수가 클수록 다음 주 수익이 높다고 본다.
SIGN = {
    "rev5": -1,      # 과거 5일 수익: 단기 반전 (Lehmann 1990)
    "rev20": -1,     # 과거 20일 수익: 1개월 반전 (Jegadeesh 1990)
    "hi52": +1,      # 종가 / 52주 최고가: George·Hwang(2004)
    "abvol": +1,     # 5일/60일 평균 거래대금: 고거래량 프리미엄 (Gervais·Kaniel·Mingelgrin 2001)
    "vol20": -1,     # 20일 일간 수익 표준편차: 저변동성 이상현상 (Ang et al. 2006)
    "mom12_1": +1,   # t-252 ~ t-21 수익: 모멘텀 (Jegadeesh·Titman 1993)
}
NAMES = list(SIGN)


def features(d):
    c, r = d.close, d.close.pct_change()
    return pd.DataFrame({
        "rev5": c / c.shift(5) - 1,
        "rev20": c / c.shift(20) - 1,
        "hi52": c / d.high.rolling(252).max(),       # 거래정지일 고가 0은 max에 영향 없음
        "abvol": d.value.rolling(5).mean() / d.value.rolling(60).mean(),
        "vol20": r.rolling(20).std(),
        "mom12_1": c.shift(21) / c.shift(252) - 1,
    }).replace([np.inf, -np.inf], np.nan)


def samples(frames, start, end, days=20, horizon=5):
    """images.build와 같은 표본 + 지표(이론 부호 적용 전 원값). 반환 meta[date, ticker, ret, 지표...]."""
    cal = pd.DatetimeIndex(sorted(set().union(*(d.index for d in frames.values()))))
    form = cal.to_series().groupby(cal.to_period("W")).max()
    form = form[(form >= start) & (form <= end)].values
    rows = []
    for tk, d in frames.items():
        d = d.sort_index()
        ok = ((d[["open", "high", "low", "volume"]] > 0).all(axis=1)).values
        fwd = (d.close.shift(-horizon) / d.close - 1).values
        f = features(d).values
        for p in d.index.get_indexer(form):
            if p < 2 * days - 1 or np.isnan(fwd[p]) or not ok[p - days + 1:p + 1].all():
                continue
            rows.append((d.index[p], tk, fwd[p], *f[p]))
    return pd.DataFrame(rows, columns=["date", "ticker", "ret", *NAMES])


def xrank(m, cols):
    """형성일별 횡단면 백분위 순위 − 0.5 (결측은 0 = 중앙값)."""
    return m.groupby("date")[cols].rank(pct=True).sub(0.5).fillna(0.0)


def eval_score(m, score):
    """점수 → 형성일별 백분위(0~1)를 확률 자리에 넣는다. 정확도 = '그 주 중앙값 위면 상승' 예측."""
    s = m.assign(s=score).dropna(subset=["s"])
    prob = s.groupby("date").s.rank(pct=True).values
    return cnn.evaluate(prob, s[["date", "ticker", "ret"]].reset_index(drop=True))


def cnn_corr(tr=None, te=None):
    """A4 예측 파일 생기면 실행: results/a4_pred_*.parquet(date, ticker, prob, ret)과
    각 지표 점수(이론 부호)의 형성일별 스피어만 상관 평균과 t값."""
    files = sorted(RES.glob("a4_pred_*.parquet"))
    if not files:
        print("A4 예측 파일 없음 → 상관 분석 건너뜀")
        return None
    if te is None:
        te = samples(load(), *TEST)
    out = []
    for f in files:
        p = pd.read_parquet(f).merge(te, on=["date", "ticker"], how="inner", suffixes=("", "_a3"))
        for k in NAMES:
            sub = p[["date", "prob"]].assign(s=SIGN[k] * p[k]).dropna()
            rk = sub.groupby("date")[["prob", "s"]].rank()
            rho = rk.groupby(sub["date"]).apply(lambda g: g.prob.corr(g.s)).dropna()
            out.append(dict(file=f.name, n=len(sub), indicator=k, spearman_mean=rho.mean(),
                            t=rho.mean() / rho.std() * np.sqrt(len(rho)), weeks=len(rho)))
    out = pd.DataFrame(out)
    out.to_csv(RES / "a3_corr_cnn.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    return out


def load():
    return {f.stem: pd.read_parquet(f) for f in sorted(DATA.glob("*.parquet"))}


def check(frames):
    """표본이 images.build와 정확히 같은지 종목 5개로 확인."""
    sub = {k: frames[k] for k in list(frames)[:5]}
    _, ref = images.build(sub, "2011-01-01", "2011-06-30")
    mine = samples(sub, "2011-01-01", "2011-06-30")
    assert len(ref) > 50
    pd.testing.assert_frame_equal(ref, mine[["date", "ticker", "ret"]])


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    frames = load()
    check(frames)
    tr, te = samples(frames, *TRAIN), samples(frames, *TEST)
    print(f"학습 표본 {len(tr)}, 평가 표본 {len(te)} (CNN: 58,153 / 17,918)")
    print(f"상승 비율: 학습 {(tr.ret > 0).mean():.3f}, 평가 {(te.ret > 0).mean():.3f}")

    rows = []
    for k in NAMES:
        r, rtr, flip = (eval_score(te, SIGN[k] * te[k]), eval_score(tr, SIGN[k] * tr[k]),
                        eval_score(te, -SIGN[k] * te[k]))
        rows.append(dict(model=k, sign=SIGN[k], n=r["n_images"], accuracy=r["accuracy"],
                         hl_weekly=r["hl_weekly_mean"], sharpe=r["hl_sharpe"], t=r["hl_t"],
                         train_hl_t=rtr["hl_t"], flipped_hl_weekly=flip["hl_weekly_mean"], flipped_t=flip["hl_t"]))

    xtr, xte, ytr = xrank(tr, NAMES), xrank(te, NAMES), (tr.ret > 0).values
    models = {"logit_all": LogisticRegression(max_iter=1000),
              # ponytail: 하이퍼파라미터는 고정 1회(조정 안 함). 내부 10% 무작위 검증으로 조기 종료.
              "tree_hgb_d3": HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                                            early_stopping=True, random_state=0)}
    for name, mdl in models.items():
        mdl.fit(xtr, ytr)
        prob = mdl.predict_proba(xte)[:, 1]
        r = cnn.evaluate(prob, te[["date", "ticker", "ret"]])
        rtr = cnn.evaluate(mdl.predict_proba(xtr)[:, 1], tr[["date", "ticker", "ret"]])
        rows.append(dict(model=name, sign=np.nan, n=r["n_images"], accuracy=r["accuracy"],
                         hl_weekly=r["hl_weekly_mean"], sharpe=r["hl_sharpe"], t=r["hl_t"], train_hl_t=rtr["hl_t"],
                         pred_up_share=float((prob > 0.5).mean())))
        if name == "logit_all":
            print("로지스틱 계수:", dict(zip(NAMES, mdl.coef_[0].round(3))), "절편", round(mdl.intercept_[0], 3))

    rows.append(dict(model="always_down", n=len(te), accuracy=float((te.ret <= 0).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "a3_baselines.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    cnn_corr(tr, te)


if __name__ == "__main__":
    main()

"""DataGuide(학교 라이선스) KOSPI 전 보통주로 파일럿을 다시 돌린다 — 논문용 데이터 경로.
학습 2005-01~2010-11, 평가 2011-01~2012-11(12월 엠바고). 시험 구간(2015~)은 불러오지도 않는다.
한 번에: CNN(lr 1e-4, 검증 손실로 고른 값) + 같은 표본의 단순 지표 베이스라인(A3) + CNN 확률과 지표의 순위상관.
사용: py -3 03_pilot_dg.py [torch 스레드 수, 기본 8] → results/dg_pilot_*.{log,json,csv,parquet}
"""
import a3_baselines as a3  # noqa: I001  (스레드 환경변수를 먼저 정한다)

import json
import sys

import numpy as np
import pandas as pd
import torch

import cnn
import dg
import images

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
RES = a3.RES
logf = open(RES / "dg_pilot_train.log", "w", encoding="utf-8")


def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


frames = {k: v for k, v in dg.frames("2004-06-01", "2012-12-31").items() if k.endswith("0")}  # 보통주
log(f"DataGuide KOSPI 보통주 {len(frames)}개 (2004-06~2012-12)")
xtr, mtr = images.build(frames, *a3.TRAIN)
xte, mte = images.build(frames, *a3.TEST)
log(f"학습 이미지 {len(xtr)}장, 평가 {len(xte)}장, 학습 상승 비율 {(mtr.ret > 0).mean():.3f}, 평가 {(mte.ret > 0).mean():.3f}")
# ponytail: DataGuide 수정주가가 거래정지 중 감자·분할을 보정하지 않은 종목이 있다(004550, 014900 확인). 규모만 기록.
log(f"5일 수익 |ret|>50% 라벨: 학습 {(mtr.ret.abs() > .5).sum()}, 평가 {(mte.ret.abs() > .5).sum()}")

# 1) 베이스라인: 같은 표본(지표는 형성일 종가까지 정보만)
tr, te = a3.samples(frames, *a3.TRAIN), a3.samples(frames, *a3.TEST)
assert len(tr) == len(mtr) and len(te) == len(mte)
rows = []
for k in a3.NAMES:
    r = a3.eval_score(te, a3.SIGN[k] * te[k])
    rows.append(dict(model=k, hl_weekly=r["hl_weekly_mean"], sharpe=r["hl_sharpe"], t=r["hl_t"]))
from sklearn.linear_model import LogisticRegression  # noqa: E402

lg = LogisticRegression(max_iter=1000).fit(a3.xrank(tr, a3.NAMES), (tr.ret > 0).values)
r = cnn.evaluate(lg.predict_proba(a3.xrank(te, a3.NAMES))[:, 1], te[["date", "ticker", "ret"]])
rows.append(dict(model="logit_all", hl_weekly=r["hl_weekly_mean"], sharpe=r["hl_sharpe"], t=r["hl_t"]))
log(pd.DataFrame(rows).round(4).to_string(index=False))

# 2) CNN
model = cnn.train(xtr, (mtr.ret > 0).values, lr=1e-4, log=log)
prob = cnn.predict(model, xte)
r = cnn.evaluate(prob, mte)
rows.append(dict(model="cnn_w_lr1e-4", hl_weekly=r["hl_weekly_mean"], sharpe=r["hl_sharpe"], t=r["hl_t"],
                 accuracy=r["accuracy"]))
mte.assign(prob=prob).to_parquet(RES / "dg_pilot_pred_cnn.parquet")
pd.DataFrame(rows).to_csv(RES / "dg_pilot_compare.csv", index=False, encoding="utf-8-sig")
(RES / "dg_pilot_cnn_result.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
log(json.dumps(r, ensure_ascii=False, indent=1))

# 3) CNN이 무엇을 배웠나: 형성일별 스피어만(CNN 확률 vs 이론 부호를 적용한 지표)
p = mte.assign(prob=prob).merge(te, on=["date", "ticker"], suffixes=("", "_a3"))
corr = []
for k in a3.NAMES:
    s = p[["date", "prob"]].assign(s=a3.SIGN[k] * p[k]).dropna()
    rho = s.groupby("date").apply(lambda g: g.prob.rank().corr(g.s.rank())).dropna()
    corr.append(dict(indicator=k, spearman_mean=rho.mean(), t=rho.mean() / rho.std() * np.sqrt(len(rho))))
log(pd.DataFrame(corr).round(3).to_string(index=False))
pd.DataFrame(corr).to_csv(RES / "dg_pilot_corr.csv", index=False, encoding="utf-8-sig")

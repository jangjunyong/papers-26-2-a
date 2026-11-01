"""파일럿: 학습 2005-01~2010-11 → 평가 2011-01~2012-11 (12월은 엠바고).
계획의 시험 구간(2015~)은 건드리지 않는다. 결과는 results/pilot_lr{학습률}_*.{json,log}.
사용: py -3 02_pilot.py [학습률, 기본 1e-5]. 학습률은 평가 성과가 아니라 검증 손실로 고른다."""
import json
import pathlib
import sys

import pandas as pd

import cnn
import images

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily" / "ohlcv"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5
tag = f"pilot_lr{lr:g}"
logf = open(RES / f"{tag}_train.log", "w", encoding="utf-8")


def log(s):
    print(s)
    logf.write(s + "\n")
    logf.flush()


frames = {f.stem: pd.read_parquet(f) for f in sorted(DATA.glob("*.parquet"))}
log(f"종목 {len(frames)}개")
xtr, mtr = images.build(frames, "2005-01-01", "2010-11-30")
xte, mte = images.build(frames, "2011-01-01", "2012-11-30")
log(f"학습 이미지 {len(xtr)}장, 평가 이미지 {len(xte)}장, 학습 라벨 상승 비율 {(mtr.ret > 0).mean():.3f}")

model = cnn.train(xtr, (mtr.ret > 0).values, lr=lr, log=log)
r = cnn.evaluate(cnn.predict(model, xte), mte)
log(json.dumps(r, ensure_ascii=False, indent=1))
(RES / f"{tag}_result.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")

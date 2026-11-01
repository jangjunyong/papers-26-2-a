"""A4 모델 실험 (파일럿 구간: 학습 2005-01~2010-11 → 평가 2011-01~2012-11, 12월 엠바고). 시드 0 하나.
  E1  학습 이미지를 매일 형성(freq="D"), 검증은 cnn.train의 무작위 70/30 (라벨이 겹쳐 검증 손실이 낙관적일 수 있음)
  E1t E1과 같은 학습 이미지, 검증만 시간 블록: 학습 형성일 ≤2009-11-30, 검증 2010-01~2010-11 (2009-12 엠바고)
  E2  주 단위, 거래량 영역(아래 13행 = 간격 1 + 거래량 12)을 0으로 지운 이미지로 학습·평가 (H4 절제)
  E3  주 단위, 라벨·평가 모두 20거래일 수익 (주 단위 형성이라 라벨이 4주씩 겹친다)
평가는 모두 주 단위(freq="W") 2011-01-01~2012-11-30, lr 1e-4.
사용: py -3 a4_run.py E1|E1t|E2|E3 [스레드 수, 기본 6]
산출: results/a4_<실험>_train.log, a4_<실험>_result.json, a4_pred_<실험>.parquet(date, ticker, prob, ret)"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import cnn
import images

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily" / "ohlcv"
RES = ROOT / "results"
LR, SEED = 1e-4, 0


def fit_split(x, y, tr, va, seed=SEED, lr=LR, bs=128, patience=2, max_epochs=50, log=print):
    """cnn.train과 같은 학습 루프, 무작위 70/30 대신 주어진 학습·검증 인덱스를 쓴다(E1t 전용)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, lossf = cnn.CNN20(), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.from_numpy(y.astype(np.int64))
    best, bad, state = np.inf, 0, None
    for ep in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // bs)):
            opt.zero_grad()
            lossf(model(cnn._t(x[b])), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([lossf(model(cnn._t(x[b])), yt[b]).item() for b in np.array_split(va, max(1, len(va) // 1024))])
        log(f"epoch {ep + 1}: val loss {vl:.5f}")
        if vl < best:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


def main(name, threads):
    assert name in ("E1", "E1t", "E2", "E3"), name
    torch.set_num_threads(threads)
    logf = open(RES / f"a4_{name}_train.log", "w", encoding="utf-8")
    vals = []

    def log(s):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
        if s.startswith("epoch"):
            vals.append(float(s.split()[-1]))

    freq = "D" if name.startswith("E1") else "W"
    h = 20 if name == "E3" else 5
    log(f"{name}: freq={freq}, horizon={h}, lr={LR:g}, seed={SEED}, threads={threads}")
    t0 = time.time()
    frames = {f.stem: pd.read_parquet(f) for f in sorted(DATA.glob("*.parquet"))}
    xtr, mtr = images.build(frames, "2005-01-01", "2010-11-30", horizon=h, freq=freq)
    xte, mte = images.build(frames, "2011-01-01", "2012-11-30", horizon=h)
    n_stocks = len(frames)
    del frames
    if name == "E2":
        xtr[:, -13:] = 0
        xte[:, -13:] = 0
    y = (mtr.ret > 0).values
    log(f"종목 {n_stocks}개, 학습 이미지 {len(xtr)}장({xtr.nbytes / 1e9:.2f}GB), 평가 이미지 {len(xte)}장, "
        f"학습 라벨 상승 비율 {y.mean():.3f}, 이미지 생성 {time.time() - t0:.0f}초")

    t1 = time.time()
    extra = {}
    if name == "E1t":
        tr = np.flatnonzero(mtr.date <= "2009-11-30")
        va = np.flatnonzero(mtr.date >= "2010-01-01")
        extra = dict(n_fit=len(tr), n_val=len(va), val_block="2010-01-01~2010-11-30", fit_block="2005-01-01~2009-11-30")
        log(f"시간 블록 분할: 학습 {len(tr)}장, 검증 {len(va)}장")
        model = fit_split(xtr, y, tr, va, log=log)
    else:
        model = cnn.train(xtr, y, seed=SEED, lr=LR, log=log)
    train_sec = time.time() - t1

    prob = cnn.predict(model, xte)
    r = dict(experiment=name, freq_train=freq, freq_eval="W", horizon=h, lr=LR, seed=SEED, n_seeds=1, threads=threads,
             n_train=len(xtr), train_up_ratio=float(y.mean()), val_losses=vals, best_val_loss=min(vals),
             best_epoch=int(np.argmin(vals)) + 1, epochs_run=len(vals), train_minutes=round(train_sec / 60, 1), **extra)
    r.update(cnn.evaluate(prob, mte))
    if name == "E3":  # 20일 보유가 겹친다 → 4번째 주마다만 쓰는 비중첩 표본도 계산, 샤프는 √13(=√52/2)로 연율화
        keep = mte.date.isin(np.sort(mte.date.unique())[::4])
        nov = cnn.evaluate(prob[keep.values], mte[keep])
        nov["hl_sharpe"] /= 2
        r["nonoverlap_every4w"] = nov
    log(json.dumps(r, ensure_ascii=False, indent=1))
    (RES / f"a4_{name}_result.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    mte.assign(prob=prob)[["date", "ticker", "prob", "ret"]].to_parquet(RES / f"a4_pred_{name}.parquet", index=False)
    log("완료")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)

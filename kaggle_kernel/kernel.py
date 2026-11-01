"""JKX(2023) 20일 CNN 학습과 평가 (계획 10단계).

모델: [5×3 합성곱 → 배치정규화 → LeakyReLU → 2×1 최대풀링] × 3 (64·128·256 채널),
첫 층 세로 stride 3·dilation 2, 드롭아웃 50% → 완전연결 2클래스. Xavier 초기화.
학습: Adam 1e-5, 배치 128, 학습 표본을 무작위 70/30으로 나눠 검증 손실이 2에폭 개선 없으면 멈춘다.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Kaggle GPU면 cuda


class CNN20(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o, **k):
            return [nn.Conv2d(i, o, (5, 3), **k), nn.BatchNorm2d(o), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]

        self.f = nn.Sequential(*blk(1, 64, stride=(3, 1), dilation=(2, 1), padding=(12, 1)),
                               *blk(64, 128, padding=(2, 1)), *blk(128, 256, padding=(2, 1)))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 3 * 60, 2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.h(self.f(x))


def _t(x):
    return torch.from_numpy(x).to(DEV).float().div_(255).unsqueeze(1)


def predict(model, x, bs=1024):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(_t(x[i:i + bs])).softmax(1)[:, 1].cpu().numpy() for i in range(0, len(x), bs)])


def train(x, y, seed=0, lr=1e-5, bs=128, patience=2, max_epochs=50, log=print, split=None):
    """split=(학습 인덱스, 검증 인덱스)를 주면 그대로 쓴다(시간 블록 검증). 없으면 무작위 70/30."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if split is None:
        idx = rng.permutation(len(x))
        cut = int(0.7 * len(x))
        tr, va = idx[:cut], idx[cut:]
    else:
        tr, va = split
    model, lossf = CNN20().to(DEV), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.from_numpy(y.astype(np.int64)).to(DEV)
    best, bad, state = np.inf, 0, None
    for ep in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // bs)):
            opt.zero_grad()
            lossf(model(_t(x[b])), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([lossf(model(_t(x[b])), yt[b]).item() for b in np.array_split(va, max(1, len(va) // 1024))])
        log(f"epoch {ep + 1}: val loss {vl:.5f}")
        if vl < best:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


def evaluate(prob, meta, n=10):
    """정확도 + 형성일별 확률 10분위 동일가중 → 상위−하위 주간 수익, 연율 샤프(√52)."""
    m = meta.assign(prob=prob)
    acc = float(((m.prob > 0.5) == (m.ret > 0)).mean())
    m["q"] = m.groupby("date").prob.transform(lambda p: pd.qcut(p.rank(method="first"), n, labels=False) if len(p) >= n else np.nan)
    dec = m.dropna(subset=["q"]).groupby(["date", "q"]).ret.mean().unstack()
    hl = dec[n - 1] - dec[0]
    return dict(n_images=len(m), accuracy=acc, weeks=len(hl), hl_weekly_mean=float(hl.mean()),
                hl_sharpe=float(hl.mean() / hl.std() * np.sqrt(52)), hl_t=float(hl.mean() / hl.std() * np.sqrt(len(hl))),
                decile_mean=dec.mean().round(5).tolist())


# ---- Kaggle 커널 본문: kaggle_push.py가 cnn.py 내용을 이 위에 붙여 하나의 스크립트로 올린다 ----
"""설정 격자를 GPU에서 한 번에 돌린다. 설정 선택은 검증 손실로 하고, 평가 성과는 전부 기록만 한다.
입력: 비공개 데이터셋의 w_train / w_eval / d_train.npz (kaggle_export.py). 출력: /kaggle/working/results.json, pred_*.parquet"""
import glob
import json
import os
import time

IN = os.environ.get("KAGGLE_IN") or os.path.dirname(glob.glob("/kaggle/input/**/w_train.npz", recursive=True)[0]) + "/"
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
print("입력", IN, "| 장치", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", flush=True)
CONFIGS = [  # 이름, 학습 묶음, 수익 기간, 학습률, 거래량 지움, 시드
    ("w_lr1e-4", "w_train", 5, 1e-4, False, 0), ("d_lr1e-4", "d_train", 5, 1e-4, False, 0),
    ("w_lr1e-4_tb", "w_train", 5, 1e-4, False, 0), ("d_lr1e-4_tb", "d_train", 5, 1e-4, False, 0),  # E1t: 시간 블록 검증
    ("w_lr1e-5", "w_train", 5, 1e-5, False, 0), ("d_lr1e-5", "d_train", 5, 1e-5, False, 0),
    ("d_novol", "d_train", 5, 1e-4, True, 0), ("w_h20", "w_train", 20, 1e-4, False, 0),
]  # 이름이 _tb로 끝나면 검증 = 2010년 형성분, 학습 = 2009-11 이전(2009-12 엠바고)


def time_block(m):
    d = m.date.values
    return (np.where(d <= np.datetime64("2009-11-30"))[0], np.where(d >= np.datetime64("2010-01-01"))[0])
_cache = {}


def load(name, horizon):
    if name not in _cache:
        z = np.load(IN + f"{name}.npz")
        _cache[name] = (np.unpackbits(z["x"], axis=-1).reshape(-1, 64, 60) * np.uint8(255),
                        pd.DataFrame(dict(date=pd.to_datetime(z["date"]), ticker=z["ticker"],
                                          ret5=z["ret5"], ret20=z["ret20"])))
    x, m = _cache[name]
    m = m.rename(columns={f"ret{horizon}": "ret"})[["date", "ticker", "ret"]]
    keep = m.ret.notna().values
    return x[keep], m[keep].reset_index(drop=True)


results = []
for name, train_set, h, lr, novol, seed in CONFIGS:
    t0 = time.time()
    xtr, mtr = load(train_set, h)
    xte, mte = load("w_eval", h)
    if novol:                                   # 간격 1행 + 거래량 12행 지움(H4 절제)
        xtr, xte = xtr.copy(), xte.copy()
        xtr[:, 51:, :] = 0
        xte[:, 51:, :] = 0
    losses = []
    model = train(xtr, (mtr.ret > 0).values, seed=seed, lr=lr, bs=128, split=time_block(mtr) if name.endswith("_tb") else None,
                  log=lambda s: (print(name, s, flush=True), losses.append(s)))
    prob = predict(model, xte)
    r = evaluate(prob, mte)
    r.update(name=name, train_set=train_set, horizon=h, lr=lr, novol=novol, seed=seed, n_train=len(xtr),
             epochs_log=losses, minutes=round((time.time() - t0) / 60, 1), device=str(DEV))
    results.append(r)
    mte.assign(prob=prob).to_parquet(f"{WORK}/pred_{name}.parquet")
    json.dump(results, open(f"{WORK}/results.json", "w"), ensure_ascii=False, indent=1)
    print(name, {k: r[k] for k in ("accuracy", "hl_weekly_mean", "hl_sharpe", "hl_t", "minutes")}, flush=True)

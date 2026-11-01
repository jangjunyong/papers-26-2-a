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

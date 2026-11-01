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

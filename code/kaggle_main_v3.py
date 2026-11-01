# ---- Kaggle v3 커널 본문 (사전고정_v1 3절): kaggle_push.py v3 가 cnn.py 와 SEEDS 줄을 이 위에 붙여 올린다 ----
"""시드별 CNN 학습 → 2010~2026-05 주간 이미지 추론 확률 저장.
학습 = d_train(2000~2007 매일 형성), 조기 종료 검증 = d_val(2008~2009, 시간 블록). Adam 1e-4, 배치 128, 인내 2에폭.
출력: /kaggle/working/pred_seed{s}.parquet(date, ticker, prob), cnn_seed{s}.pt, log_seed{s}.json"""
import glob
import json
import os
import time

IN = os.environ.get("KAGGLE_IN") or os.path.dirname(glob.glob("/kaggle/input/**/d_train.npz", recursive=True)[0]) + "/"
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
print("입력", IN, "| 장치", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", "| 시드", SEEDS, flush=True)


def load(name):
    z = np.load(IN + f"{name}.npz")
    return np.unpackbits(z["x"], axis=-1).reshape(-1, 64, 60) * np.uint8(255), z


xtr, ztr = load("d_train")
xva, zva = load("d_val")
ntr, nva = len(xtr), len(xva)
y = np.concatenate([ztr["ret5"] > 0, zva["ret5"] > 0])
x = np.concatenate([xtr, xva])
del xtr, xva
split = (np.arange(ntr), np.arange(ntr, ntr + nva))
xin, zin = load("w_infer")
print(f"학습 {ntr:,} 검증 {nva:,} 추론 {len(xin):,}장, 학습 상승 비율 {y[:ntr].mean():.3f}", flush=True)

for seed in SEEDS:
    t0, losses = time.time(), []
    model = train(x, y, seed=seed, lr=1e-4, bs=128, patience=2, max_epochs=50, split=split,
                  log=lambda s: (print(f"seed{seed}", s, flush=True), losses.append(s)))
    prob = predict(model, xin)
    pd.DataFrame(dict(date=zin["date"], ticker=zin["ticker"], prob=prob)).to_parquet(f"{WORK}/pred_seed{seed}.parquet")
    torch.save(model.state_dict(), f"{WORK}/cnn_seed{seed}.pt")
    json.dump(dict(seed=seed, epochs=losses, minutes=round((time.time() - t0) / 60, 1), n_train=ntr, n_val=nva,
                   n_infer=len(xin), device=str(DEV)), open(f"{WORK}/log_seed{seed}.json", "w"), ensure_ascii=False, indent=1)
    print(f"seed{seed} 끝 {(time.time() - t0) / 60:.1f}분", flush=True)

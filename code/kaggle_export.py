"""Kaggle GPU 학습용 데이터 묶음. 원본 가격(DataGuide 라이선스)은 올리지 않고 파생 이미지·라벨만 비공개로 올린다.
이미지는 0/255 흑백이라 비트로 묶는다(64×60 → 480바이트). 학습 ≤2010-11, 평가 2011-01~2012-11, 시험 구간 없음.
출력: PAPERS/data/kaggle_upload/<묶음>.npz (x=packbits, date, ticker, ret5, ret20)
사용: py -3 kaggle_export.py
"""
import numpy as np
import pandas as pd

import dg
import images

OUT = dg.PAPERS / "data" / "kaggle_upload"
SETS = {"w_train": ("2005-01-01", "2010-11-30", "W"), "w_eval": ("2011-01-01", "2012-11-30", "W"),
        "d_train": ("2005-01-01", "2010-11-30", "D")}


def export(frames, start, end, freq, chunk=100):
    """종목 100개씩 나눠 만들어 메모리를 아낀다. 20일 라벨은 같은 표본에 붙인다(없으면 NaN)."""
    xs, metas, keys = [], [], list(frames)
    for i in range(0, len(keys), chunk):
        sub = {k: frames[k] for k in keys[i:i + chunk]}
        x, m = images.build(sub, start, end, freq=freq)
        r20 = pd.concat({k: (d.close.shift(-20) / d.close - 1) for k, d in sub.items()}, names=["ticker", "date"])
        m["ret20"] = r20.reindex(pd.MultiIndex.from_frame(m[["ticker", "date"]])).values
        xs.append(np.packbits((x > 0).reshape(len(x), -1), axis=-1))   # 3,840비트 → 480바이트
        metas.append(m)
        print(f"  {i + len(sub)}/{len(keys)}종목 {sum(len(a) for a in xs)}장", flush=True)
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {k: v for k, v in dg.frames("2004-06-01", "2012-12-31").items() if k.endswith("0")}
    for name, (s, e, f) in SETS.items():
        x, m = export(frames, s, e, f)
        np.savez_compressed(OUT / f"{name}.npz", x=x, date=m.date.values.astype("datetime64[D]"),
                            ticker=m.ticker.values.astype("U6"), ret5=m.ret.values, ret20=m.ret20.values)
        print(name, x.shape, f"{(OUT / f'{name}.npz').stat().st_size / 1e6:.0f}MB", flush=True)

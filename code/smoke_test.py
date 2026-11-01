"""파이프라인 점검 (실제 데이터·시험 구간을 보지 않는다).
① 무작위 보행 가격 → CNN 정확도 ≈ 0.5  ② 평가 함수에 미래 수익을 넣으면 비현실적 샤프가 나와야 한다."""
import numpy as np
import pandas as pd

import cnn
import images

rng = np.random.default_rng(0)
days = pd.bdate_range("2005-01-03", periods=700)
frames = {}
for k in range(40):
    c = 10000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
    o = c * np.exp(rng.normal(0, 0.005, len(days)))
    frames[f"T{k:02d}"] = pd.DataFrame(dict(open=o, high=np.maximum(o, c) * 1.01, low=np.minimum(o, c) * 0.99,
                                            close=c, volume=rng.integers(1000, 9000, len(days)).astype(float)), index=days)
frames["T00"].iloc[100:103, :3] = 0          # 거래정지 흉내 → 그 창은 빠져야 한다

x, meta = images.build(frames, "2005-03-01", "2007-12-31")
assert x.shape[1:] == (64, 60) and len(x) == len(meta) > 3000
assert not ((meta.ticker == "T00") & meta.date.between(days[100], days[100 + 19])).any()

model = cnn.train(x, (meta.ret > 0).values, max_epochs=1, lr=1e-4)
r = cnn.evaluate(cnn.predict(model, x), meta)
print("무작위:", r)
assert 0.45 < r["accuracy"] < 0.56

cheat = cnn.evaluate(meta.ret.values, meta)
print("미래 수익 신호:", cheat["hl_sharpe"])
assert cheat["hl_sharpe"] > 10
print("smoke test 통과")

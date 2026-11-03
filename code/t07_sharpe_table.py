"""t07: 결론 요약표용 샤프 비율 (사후 계산, 판정 아님, 2026-11-03). 결과: results/t07/sharpe_table.json

네 모델(P·P+C·P+F·P+C+F) x 단계별 연환산 샤프(√52), 대표 방식 (a) 주간 전량 교체.
1 제약·비용 없음: 10분위 동일가중 롱숏(bt.run constraints=())
2 상위 20 매수만, 비용 전(gross)  3 비용 후 순수익(net, 호가 단위·Corwin–Schultz)  4 시가총액가중 시장 대비 초과(net − bench_vw)
P 는 사전고정 RQ3 대상이 아니어서 t01 에 없다. 나머지 세 모델 값은 t01 bt_weekly 와 같아야 한다(자체 점검).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bt  # noqa: E402
import t01_test_run as t01  # noqa: E402

OUT = t01.OUT.parent / "t07"
SR = lambda x: round(float(x.mean() / x.std() * np.sqrt(52)), 2)  # noqa: E731
MODELS = ["P", "P+C", "P+F", "P+C+F"]


def main():
    s = pd.read_parquet(t01.OUT / "scores.parquet")
    M = bt.prep(pd.read_parquet(t01.PANEL), s.ticker.unique())
    ref = pd.read_parquet(t01.OUT / "bt_weekly.parquet")
    ic = pd.read_csv(t01.OUT / "ic_weekly.csv", index_col=0)
    out = {}
    for m in MODELS:
        sc = s[["date", "ticker", m]].rename(columns={m: "score"})
        o = dict(ic=round(float(ic[m].mean()), 4), none=SR(bt.run(sc, M, method="a", spread="cs", constraints=()).excess))
        for sp in ["tick", "cs"]:
            r = bt.run(sc, M, method="a", spread=sp)
            o["gross"], o[f"net_{sp}"], o[f"ex_{sp}"] = SR(r.gross), SR(r.net), SR(r.net - r.bench_vw)
            if m != "P":
                gap = float((r.set_index("date").net - ref.xs(f"{m}|a|{sp}", level="key").net).abs().max())
                assert gap < 1e-12, f"{m}|a|{sp} 가 t01 과 다르다: {gap}"
        out[m] = o
        print(m, o, flush=True)
    b = ref.xs("P+C+F|a|cs", level="key")
    out["market_vw"] = SR(b.bench_vw)
    OUT.mkdir(exist_ok=True)
    (OUT / "sharpe_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame({k: v for k, v in out.items() if k in MODELS}).to_string(), "\nmarket_vw", out["market_vw"])


if __name__ == "__main__":
    main()

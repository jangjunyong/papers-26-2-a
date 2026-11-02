"""t03: RQ4 를 사전고정_v1 7절 기준(후보 동일가중 대비)으로 다시 계산한다.

t01.car_6_20() 은 거래대금 하위 20% 를 빼기 전 유니버스로 10분위·기준 평균을 계산했다(변경 이력 2026-11-02 03시 (1)).
여기서는 m01 과 같은 후보 규칙(같은 날 avg_tv60 백분위 > 0.2)으로 거른 뒤 후보 안에서 5일 순매수 순위를 다시 매기고
t01.car_6_20() 을 그대로 쓴다. 모델·점수는 쓰지 않는다. 출력: results/t01/rq4_candidates.json
사용: py -3 t03_rq4_candidates.py
"""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m01_gbm_2x2 as m01  # noqa: E402
import t01_test_run as t01  # noqa: E402

FLOW = [f"flow_{g}_5_rk" for g in ["indiv", "inst", "for"]]


def main():
    F = pd.read_parquet(t01.FEAT_OPEN, columns=["ticker", "date", "avg_tv60"] + FLOW)
    t_lo, t_hi = map(pd.Timestamp, m01.WINDOWS["test"])
    F = F[(F.date >= t_lo) & (F.date <= t_hi)]
    F = F[F.groupby("date").avg_tv60.rank(pct=True) > m01.LIQ_CUT].copy()
    F[FLOW] = F.groupby("date")[FLOW].rank(pct=True) - 0.5          # 후보 안에서 다시 순위
    out = t01.car_6_20(pd.read_parquet(t01.PANEL, columns=["ticker", "date", "adj_close"]), F)
    out["names_per_date"] = float(F.groupby("date").size().mean())
    (t01.OUT / "rq4_candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    o = main()
    print(f"형성일당 후보 {o.pop('names_per_date'):.0f}종목")
    for g, r in o.items():
        print(g, {k: (round(v["mean"] * 100, 2), [round(c * 100, 2) for c in v["ci95"]]) for k, v in r.items()})

"""t02: t01 산출물에서 '판정 없이 보고' 항목을 계산한다 (사전고정_v1 4절 마지막 줄·보고서 3.3·3.6·4.9·4.11).

모델을 다시 학습하거나 시험 구간을 다시 실행하지 않는다. 입력 = results/t01/ 와 라벨 보호를 푼 특성.
출력: results/t01/supplement.json — 나머지 칸 비교·상호작용 구간, 연도별 IC, 최소 검출 IC 차이, 샤프 비율, 단일 변수 IC
사용: py -3 t02_supplement.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m01_gbm_2x2 as m01  # noqa: E402

T01 = pathlib.Path(__file__).resolve().parents[1] / "results" / "t01"


def boot(x):
    x = x.dropna()
    p, ci = m01.block_boot_p(x.to_numpy())
    return dict(mean=float(x.mean()), ci95=ci, p_le0=p, weeks=len(x))


def main():
    ic = pd.read_csv(T01 / "ic_weekly.csv", index_col=0, parse_dates=True)
    diffs = {"H1": ic["P+C+F"] - ic["P+F"], "H2": ic["P+C+F"] - ic["P+C"],
             "C_without_F": ic["P+C"] - ic["P"], "F_without_C": ic["P+F"] - ic["P"],
             "interaction": (ic["P+C+F"] - ic["P+F"]) - (ic["P+C"] - ic["P"])}
    out = {"diffs": {k: boot(v) for k, v in diffs.items()}}
    # 최소 검출 차이(단측 5%, 검정력 80%): 블록 부트스트랩 표준오차 × (1.645 + 0.842)
    out["mde"] = {k: (out["diffs"][k]["ci95"][1] - out["diffs"][k]["ci95"][0]) / 3.92 * 2.487 for k in ["H1", "H2"]}
    yr = pd.DataFrame(diffs).assign(**{m: ic[m] for m in m01.MODELS}).groupby(ic.index.year).mean()
    out["yearly"] = yr.round(5).to_dict(orient="index")
    out["ic_positive_share"] = (ic[list(m01.MODELS)] > 0).mean().round(3).to_dict()

    # 샤프 비율(연환산 √52): 순수익·시장 대비 초과수익
    bw = pd.read_parquet(T01 / "bt_weekly.parquet")
    g = bw.groupby(level="key")
    sr = lambda s: float(s.mean() / s.std() * np.sqrt(52))  # noqa: E731
    out["sharpe"] = {k: dict(net=sr(x.net), ex_vw=sr(x.net - x.bench_vw), gross=sr(x.gross), mean_gross=float(x.gross.mean()),
                             mean_net=float(x.net.mean())) for k, x in g}
    b = bw.xs(bw.index.get_level_values("key")[0], level="key")
    out["bench"] = dict(vw_mean=float(b.bench_vw.mean()), ew_mean=float(b.bench_ew.mean()), vw_sharpe=sr(b.bench_vw), ew_sharpe=sr(b.bench_ew))

    # 단일 변수 IC(시험 구간, 후보 범위) — 누출 점검과 해석용
    m01.FEAT = m01.DATA / "features" / "features_v3_kospi_unlocked.parquet"
    d = m01.load("test")
    t0, t1 = map(pd.Timestamp, m01.WINDOWS["test"])
    d = d[(d.date >= t0) & (d.date <= t1)].dropna(subset=["label_ret5"])
    cols = m01.P + m01.F + m01.C
    one = d.groupby("date")[cols + ["label_ret5"]].apply(lambda x: x[cols].corrwith(x.label_ret5, method="spearman"))
    out["single_ic"] = one.mean().round(5).to_dict()
    out["cnn_ic_yearly"] = one.cnn_rk.groupby(one.index.year).mean().round(5).to_dict()
    (T01 / "supplement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    o = main()
    print(json.dumps({k: o[k] for k in ["diffs", "mde", "ic_positive_share", "single_ic", "cnn_ic_yearly", "bench"]}, ensure_ascii=False, indent=1))
    print(pd.DataFrame(o["yearly"]).T.round(4).to_string())
    print(pd.DataFrame(o["sharpe"]).T.round(3).to_string())

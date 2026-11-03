"""t06: 주 모델(P+C+F) GBM 설명 자료 (사후 작성, 판정 아님, 2026-11-03). 결과: results/t06/

m01.run('test') 과 같은 재학습 일정(15회)으로 주 모델만 다시 학습한다. 자체 점검: 예측이 t01 scores.parquet 과 같아야 한다.
1 tree_first.txt  첫 재학습(2012-06-01) 모델의 첫 번째 나무, 위쪽 3단계(그 아래는 칸 수·평균 잎 값으로 요약)
2 importance.json 변수별 분할 이득 비중(%) — 첫 모델, 15개 모델 평균·최소·최대
잎 값은 라벨(다음 주 수익의 날짜별 순위, −0.5~+0.5)에 대한 그 나무의 기여를 1,000배 한 값이다(학습률 0.05 반영 후).
분할 기준의 '상위 x%'는 그날 코스피 전체(후보 필터 전) 종목 중 순위다(f01 _rk 정의).
"""
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m01_gbm_2x2 as m01  # noqa: E402
import t01_test_run as t01  # noqa: E402

OUT = t01.OUT.parent / "t06"
COLS = m01.MODELS[t01.MAIN]
KO = {"str5_rk": "직전 5일 수익률", "str20_rk": "직전 20일 수익률", "mom12_1_rk": "12-1 모멘텀", "near52h_rk": "52주 고가 근접도",
      "vol20_rk": "20일 변동성", "abvol_rk": "비정상 거래량", "log_mktcap_rk": "시가총액", "log_tv60_rk": "60일 평균 거래대금",
      "amihud20_rk": "비유동성(Amihud)", "cnn_rk": "CNN 순위",
      **{f"flow_{g}_{w}_rk": f"{n} {w}일 순매수" for g, n in [("indiv", "개인"), ("inst", "기관"), ("for", "외국인")] for w in (5, 20)}}
GROUP = {**{c: "가격·거래(P)" for c in m01.P}, **{c: "수급(F)" for c in m01.F}, "cnn_rk": "차트(C)"}


def models():
    """m01.run 의 재학습 루프를 주 모델만으로 되풀이한다. (재학습일, 모델, 예측) 목록."""
    m01.FEAT = t01.FEAT_OPEN
    d = m01.load("test")
    cal = np.sort(d.date.unique())
    t0, t1_ = map(pd.Timestamp, m01.WINDOWS["test"])
    test_dates = pd.DatetimeIndex(cal[(cal >= t0) & (cal <= t1_)])
    retrain = test_dates[np.r_[True, test_dates.year[1:] != test_dates.year[:-1]]]
    tdays = np.sort(pd.read_parquet(m01.DATA / "daily" / "club_db" / "panel_v3.parquet", columns=["date"]).date.unique())
    out = []
    for i, r in enumerate(retrain):
        cut = tdays[np.searchsorted(tdays, r.to_datetime64()) - m01.EMBARGO]
        nxt = retrain[i + 1] if i + 1 < len(retrain) else t1_ + pd.Timedelta(days=1)
        tr = d[(d.date >= m01.TRAIN_START) & (d.label_end_date <= cut) & d.y.notna()]
        te = d[(d.date >= r) & (d.date < nxt) & (d.date <= t1_)].copy()
        m = lgb.LGBMRegressor(**m01.HP).fit(tr[COLS], tr.y)
        te["score"] = m.predict(te[COLS])
        out.append((r, len(tr), m, te[["ticker", "date", "score"]]))
    return out


def render(node, names, pre="", depth=0, lines=None):
    """'상위 x%인가?' 질문 트리(x > 50 이면 '하위 100-x%인가?'). 기준값보다 크면 오른쪽 자식, 이하면 왼쪽 자식."""
    def leaves(n):
        return [n] if "leaf_value" in n else leaves(n["left_child"]) + leaves(n["right_child"])
    lines = [] if lines is None else lines
    f, top = names[node["split_feature"]], 100 - (node["threshold"] + 0.5) * 100
    hi, lo = node["right_child"], node["left_child"]
    if top <= 50:
        lines.append(f"{KO[f]} 상위 {top:.0f}%인가?")
        kids = [("예", hi), ("아니오", lo)]
    else:
        lines.append(f"{KO[f]} 하위 {100 - top:.0f}%인가?")
        kids = [("예", lo), ("아니오", hi)]
    for k, (lab, ch) in enumerate(kids):
        last = k == 1
        head, sub = pre + ("└ " if last else "├ "), pre + ("   " if last else "│  ")
        if "leaf_value" in ch:
            lines.append(f"{head}{lab} → 잎 값 {ch['leaf_value'] * 1e3:+.1f}")
        elif depth == 2:
            L = leaves(ch)
            v = sum(x["leaf_value"] * x["leaf_count"] for x in L) / sum(x["leaf_count"] for x in L)
            lines.append(f"{head}{lab} → 평균 {v * 1e3:+.1f} (아래 질문 {len(L) - 1}개, 잎 {len(L)}개)")
        else:
            lines.append(f"{head}{lab} → ")
            n0 = len(lines)
            render(ch, names, sub, depth + 1, lines)
            lines[n0 - 1] += lines.pop(n0)                       # 자식 질문을 같은 줄에 잇는다
    return lines


def main():
    ms = models()
    ref = pd.read_parquet(t01.OUT / "scores.parquet", columns=["ticker", "date", t01.MAIN])
    chk = pd.concat([x[3] for x in ms]).merge(ref, on=["ticker", "date"])
    gap = float((chk.score - chk[t01.MAIN]).abs().max())
    assert len(chk) == len(ref) and gap < 1e-9, f"t01 과 다르다: {len(chk)} vs {len(ref)}, 최대 차 {gap}"
    print(f"자체 점검 통과: {len(chk):,}행 예측이 t01 과 같다", flush=True)

    r0, n0, m0, _ = ms[0]
    dm = m0.booster_.dump_model()
    head = (f"주 모델(P+C+F) 첫 재학습 {r0.date()} (학습 {n0:,}행)의 첫 번째 나무, 잎 {dm['tree_info'][0]['num_leaves']}개 중 위쪽 3단계\n"
            "잎 값 = 다음 주 수익 순위에 대한 이 나무의 기여 x 1,000\n")
    tree = "\n".join(render(dm["tree_info"][0]["tree_structure"], dm["feature_names"]))
    print(head + tree)

    imp = pd.DataFrame({x[0].year: pd.Series(x[2].booster_.feature_importance("gain"), index=COLS) for x in ms})
    imp = imp / imp.sum() * 100
    by_var = pd.DataFrame({"first": imp.iloc[:, 0], "mean": imp.mean(1), "min": imp.min(1), "max": imp.max(1)}).sort_values("mean", ascending=False)
    by_grp = imp.groupby(GROUP).sum()
    res = dict(by_variable={KO[c]: {k: round(float(v), 1) for k, v in row.items()} for c, row in by_var.iterrows()},
               by_group={g: dict(first=round(float(row.iloc[0]), 1), mean=round(float(row.mean()), 1),
                                 min=round(float(row.min()), 1), max=round(float(row.max()), 1)) for g, row in by_grp.iterrows()},
               n_models=len(ms), self_check_max_gap=gap)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / "tree_first.txt").write_text(head + tree + "\n", encoding="utf-8")
    (OUT / "importance.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

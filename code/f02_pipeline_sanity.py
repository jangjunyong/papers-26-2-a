"""f02: 파이프라인 점검 모델 (M0 로지스틱 대 M2+ GBM). 약식 확인용이며 가설 검정이 아니다.

학습 2015-06 ~ 2016-06 (검증 시작 25거래일 전까지), 검증 2016-07 ~ 2016-12. 시간 블록 분할만 쓰고 무작위 분할은 없다.
하이퍼파라미터는 아래 상수로 한 번 고정한 값이며 검증 결과를 보고 바꾸지 않는다 (조정 안 함, 조기 종료 안 함).
[시험 구간 보호] 2016-12-31 이후 행은 읽지 않고(parquet 필터), 아래 assert 가 학습·검증 데이터의 최대 날짜와 라벨 청산일이
2016-12-31 이하임을 강제한다. 2017-01 이후 IC·수익·샤프는 계산도 출력도 하지 않는다.
실행: py -3 f02_pipeline_sanity.py   (약 1분)
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from f01_build_features import B1, B2_M0, B2_X, B3_DSC, B3_FIN, RK  # noqa: E402  (변수·블록 이름만 쓴다)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parents[3] / "data"
FEAT = DATA / "features" / "features_v2_kospi.parquet"
PANEL = DATA / "daily" / "club_db" / "panel_v2.parquet"
TAB = ROOT / "results" / "f02_tables"

CAP = pd.Timestamp("2016-12-31")                    # 시험 구간 보호: 이 날짜 이후 데이터는 읽지도 쓰지도 않는다
TRAIN = (pd.Timestamp("2015-06-01"), pd.Timestamp("2016-06-30"))
VALID = (pd.Timestamp("2016-07-01"), pd.Timestamp("2016-12-31"))
EMBARGO = 25                                        # 학습 마지막 형성일과 검증 시작일 사이 최소 거래일 수
M0_RK = [c + "_rk" for c in ["str5", "str20", "mom12_1", "near52h", "vol20", "abvol"]]
HGB = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
LOGIT_C = 1.0
N_PERM = 10

# ---- 데이터 읽기 + 시험 구간 assert (이 파일에서 데이터를 읽는 유일한 두 곳, 둘 다 date <= CAP 필터)
df = pd.read_parquet(FEAT, filters=[("date", "<=", CAP)])
cal = np.sort(pd.read_parquet(PANEL, columns=["date"], filters=[("date", "<=", CAP)])["date"].unique())
assert df["date"].max() <= CAP and cal.max() <= np.datetime64(CAP), "2016-12-31 이후 데이터가 들어왔다"
assert pd.to_datetime(df["label_end_date"]).dropna().max() <= CAP, "라벨 청산일이 2016-12-31 을 넘는다"
assert df.loc[df["label_status"] == 5, ["label_ret5", "label_rk"]].isna().all().all(), "마스크된 라벨에 값이 남아 있다"



df = df[df["label_status"].isin([0, 1]) & df["label_rk"].notna()].copy()
idx = pd.Series(np.arange(len(cal)), index=pd.DatetimeIndex(cal))
iv = int(idx[idx.index >= VALID[0]].iloc[0])                            # 검증 시작일의 거래일 인덱스
df["ti"] = idx.reindex(df["date"]).to_numpy()
train = df[(df["date"] >= TRAIN[0]) & (df["date"] <= TRAIN[1]) & (df["ti"] <= iv - 1 - EMBARGO)]
valid = df[(df["date"] >= VALID[0]) & (df["date"] <= VALID[1])]
assert train["date"].max() <= CAP and valid["date"].max() <= CAP and valid["label_end_date"].max() <= CAP
assert train["label_end_date"].max() < valid["date"].min(), "학습 라벨 창이 검증 구간과 겹친다"
assert (valid["date"].min() - train["date"].max()).days >= EMBARGO

DOMAIN_RK = RK                                                          # M2+ = B1 + B2(M0 6지표 + 추가 3) + B3 (34개)
BLOCKS = {"B1 수급": B1, "B2 M0 6지표": B2_M0, "B2 추가(시총·거래대금·Amihud)": B2_X, "B3 공시": B3_DSC, "B3 재무": B3_FIN}


def rank_ic(score, y, dates):
    """날짜별 스피어만 순위 IC = 날짜 안 순위의 피어슨 상관. 종목 30개 미만인 날짜는 제외."""
    d = pd.DataFrame({"s": np.asarray(score), "y": np.asarray(y), "d": np.asarray(dates)}).dropna()
    g = d.groupby("d")
    a = g["s"].rank() - g["s"].rank().groupby(d["d"]).transform("mean")
    b = g["y"].rank() - g["y"].rank().groupby(d["d"]).transform("mean")
    num = (a * b).groupby(d["d"]).sum()
    den = np.sqrt((a * a).groupby(d["d"]).sum() * (b * b).groupby(d["d"]).sum())
    return (num / den)[g.size() >= 30].dropna()


def summ(ic):
    return dict(주수=len(ic), IC_평균=ic.mean(), IC_표준편차=ic.std(), IR=ic.mean() / ic.std(), t=ic.mean() / ic.std() * np.sqrt(len(ic)), 양수_주_비율=(ic > 0).mean())


def fit_m0(tr):
    m = LogisticRegression(C=LOGIT_C, max_iter=1000)
    m.fit(tr[M0_RK].fillna(0.0), (tr["label_rk"] > 0).astype(int))
    return m


def fit_m2(tr, cols):
    m = HistGradientBoostingClassifier(**HGB)
    m.fit(tr[cols], (tr["label_rk"] > 0).astype(int))
    return m


def md(df, fmt="{:.4g}", index=False):
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = ["| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return head + "\n".join(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    print(f"학습 {train['date'].min().date()} ~ {train['date'].max().date()}: {len(train):,}행, 형성일 {train['date'].nunique()}")
    print(f"검증 {valid['date'].min().date()} ~ {valid['date'].max().date()}: {len(valid):,}행, 형성일 {valid['date'].nunique()} (마지막 라벨 청산일 {valid['label_end_date'].max().date()})")
    print(f"엠바고: 학습 마지막 형성일 {train['date'].max().date()} -> 검증 시작 {valid['date'].min().date()} 거래일 간격 {iv - int(train['ti'].max())}")
    print(f"학습 안 B3 재무 변수 비결측 비율 {train[B3_FIN].notna().mean().round(3).to_dict()}, 공시 {train[B3_DSC].notna().mean().mean():.3f}, 수급 {train[B1].notna().mean().mean():.3f}")
    res = {}
    m0 = fit_m0(train)
    res["M0"] = m0.predict_proba(valid[M0_RK].fillna(0.0))[:, 1]
    m2 = fit_m2(train, DOMAIN_RK)
    res["M2+"] = m2.predict_proba(valid[DOMAIN_RK])[:, 1]
    ics = {k: rank_ic(v, valid["label_ret5"], valid["date"]) for k, v in res.items()}
    tab = pd.DataFrame({k: summ(v) for k, v in ics.items()}).T
    print("\n검증 구간(2016-07~2016-12) 주간 횡단면 순위 IC (예측 확률 vs 라벨 label_ret5):")
    print(md(tab.reset_index().rename(columns={"index": "모델"})))
    diff = (ics["M2+"] - ics["M0"]).dropna()
    print(f"\nM2+ − M0 주별 IC 차이: 평균 {diff.mean():.4f}, t {diff.mean()/diff.std()*np.sqrt(len(diff)):.2f} ({len(diff)}주), M2+ 가 높은 주 {(diff>0).mean():.0%}")
    # 학습 안 IC (과적합 정도 확인용)
    ic_tr = {"M0": rank_ic(m0.predict_proba(train[M0_RK].fillna(0.0))[:, 1], train["label_ret5"], train["date"]),
             "M2+": rank_ic(m2.predict_proba(train[DOMAIN_RK])[:, 1], train["label_ret5"], train["date"])}
    print("학습 구간 안 IC(표본 내): " + ", ".join(f"{k} {v.mean():.4f}" for k, v in ic_tr.items()))
    print("M0 로지스틱 계수:", dict(zip(M0_RK, m0.coef_[0].round(3))))
    # 블록 중요도: 순열(검증 구간, 날짜 안에서 블록 열을 한꺼번에 섞음) + 블록 제거 재학습
    rng = np.random.default_rng(0)
    base_ic = ics["M2+"].mean()
    rows = []
    vd = valid.reset_index(drop=True)
    groups = vd.groupby("date").indices
    for name, cols in BLOCKS.items():
        rk = [c + "_rk" for c in cols]
        deltas = []
        for _ in range(N_PERM):
            X = vd[DOMAIN_RK].copy()
            for _, ix in groups.items():
                X.iloc[ix, [X.columns.get_loc(c) for c in rk]] = X.iloc[rng.permutation(ix), [X.columns.get_loc(c) for c in rk]].to_numpy()
            deltas.append(base_ic - rank_ic(m2.predict_proba(X)[:, 1], vd["label_ret5"], vd["date"]).mean())
        keep = [c for c in DOMAIN_RK if c not in rk]
        mdrop = fit_m2(train, keep)
        ic_drop = rank_ic(mdrop.predict_proba(valid[keep])[:, 1], valid["label_ret5"], valid["date"])
        rows.append({"블록": name, "변수 수": len(cols), "순열 ΔIC(평균)": np.mean(deltas), "순열 ΔIC(표준편차)": np.std(deltas),
                     "블록 제거 재학습 IC": ic_drop.mean(), "ΔIC(전체−제거)": base_ic - ic_drop.mean()})
    imp = pd.DataFrame(rows)
    print(f"\nM2+ 블록 중요도 (전체 모델 검증 IC {base_ic:.4f}; ΔIC>0 이면 그 블록이 IC 에 기여):")
    print(md(imp))
    # 단일 변수 IC (부호 확인용, 선택에 쓰지 않음)
    sv = []
    for c in RK:
        ic = rank_ic(valid[c].fillna(0.0), valid["label_ret5"], valid["date"])
        sv.append(dict(변수=c[:-3], 검증_IC=ic.mean(), t=ic.mean() / ic.std() * np.sqrt(len(ic)), 비결측=valid[c].notna().mean()))
    sv = pd.DataFrame(sv)
    print("\n변수별 단일 순위 IC (검증 구간, _rk 값 그대로, 결측은 0; 부호 점검용이며 변수 선택에 쓰지 않음):")
    print(md(sv))
    tab.to_csv(TAB / "valid_ic.csv", encoding="utf-8-sig")
    imp.to_csv(TAB / "block_importance.csv", index=False, encoding="utf-8-sig")
    sv.to_csv(TAB / "single_var_ic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: v for k, v in ics.items()}).to_csv(TAB / "valid_ic_weekly.csv", encoding="utf-8-sig")
    # 증거: 2017-01 이후 성과 계산 없음
    print(f"\n[확인] 사용한 최대 날짜 학습 {train['date'].max().date()}, 검증 {valid['date'].max().date()}, 라벨 청산일 {valid['label_end_date'].max().date()} <= {CAP.date()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

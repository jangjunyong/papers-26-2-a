"""f03: 형성일 변수 패널 v3 = f01.build 를 panel_v3(2000~2026)에 그대로 적용 (사전고정_v1 1·2절).

라벨 보호선만 바꾼다: 시험 구간(2012-06~)을 가리려고 label_cap = 2012-05-31. 사전 고정 커밋 뒤 t01 이 한 번 푼다.
공시·재무(B3)는 2015 이전이 NaN 이다(확장 분석용, 주 분석은 P·F·C 만 쓴다).
출력: data/features/features_v3_kospi.parquet, features_v3_kospi_columns.md
실행: py -3 f03_build_features_v3.py
"""
import sys

import numpy as np
import pandas as pd

import f01_build_features as f01

LABEL_CAP_V3 = pd.Timestamp("2012-05-31")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    f01.PANEL = f01.D / "daily" / "club_db" / "panel_v3.parquet"
    f01.OUT = f01.OUT_DIR / "features_v3_kospi.parquet"
    f01.OUT_DOC = f01.OUT_DIR / "features_v3_kospi_columns.md"
    f01.LABEL_CAP = LABEL_CAP_V3                      # write_doc 의 설명 문구도 이 값을 쓴다
    panel, cls, fsav, fscore = f01.load_inputs()
    F = f01.build(panel, cls, fsav, fscore, label_cap=LABEL_CAP_V3, inplace=True)
    del panel
    for c in f01.RK + ["label_rk"]:
        F[c] = F[c].astype(np.float32)
    F.to_parquet(f01.OUT, index=False)
    f01.write_doc(F)
    assert F.loc[F.date > LABEL_CAP_V3, "label_rk"].isna().all()
    print(f"저장 {f01.OUT}: {len(F):,}행, 형성일 {F.date.nunique()}, 종목 {F.ticker.nunique()}")
    print("라벨 상태:", F.label_status.value_counts().sort_index().to_dict())
    print(F.groupby(F.date.dt.year)[["flow_indiv_5_rk", "str5_rk", "log_tv60_rk"]].apply(lambda g: g.notna().mean()).round(3).to_string())

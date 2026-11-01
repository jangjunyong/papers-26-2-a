"""m02: 매크로 국면 라벨 (사전고정_v1 6절, 판정 없음).

물가: 형성일이 속한 달의 1개월 전 CPI 전년동월비 ≥ 2%(한국은행 물가안정목표) → '물가≥2%', 아니면 '물가<2%'.
금리: 형성일 이전 마지막 기준금리 변경이 인상이면 '인상 후', 인하면 '인하 후'(변경 없으면 직전 방향 유지).
입력: data/macro/oecd_kor_cpi_monthly.csv, bis_kor_policy_daily.csv
사용: regimes(형성일들) → DataFrame(date, cpi_regime, rate_regime);  by_regime(주별 값 DataFrame) → 국면별 평균·주 수
"""
import pathlib

import numpy as np
import pandas as pd

MACRO = pathlib.Path(__file__).resolve().parents[3] / "data" / "macro"
CPI_TARGET = 2.0


def regimes(dates):
    d = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    cpi = pd.read_csv(MACRO / "oecd_kor_cpi_monthly.csv", parse_dates=["month"]).set_index("month").cpi_yoy
    prev_month = (d.date.dt.to_period("M") - 1).dt.to_timestamp()        # 발표 시차: 전월 값까지만 안다
    d["cpi_yoy_lag1"] = cpi.reindex(prev_month).to_numpy()
    d["cpi_regime"] = np.where(d.cpi_yoy_lag1 >= CPI_TARGET, "물가≥2%", "물가<2%")
    pol = pd.read_csv(MACRO / "bis_kor_policy_daily.csv", parse_dates=["date"]).set_index("date").policy_rate
    ch = pol.diff()
    ch = ch[ch != 0].dropna()
    direction = np.sign(ch).replace({1.0: "인상 후", -1.0: "인하 후"})
    pos = direction.index.searchsorted(d.date, side="left") - 1             # 형성일 '이전' 마지막 변경
    d["rate_regime"] = np.where(pos >= 0, direction.to_numpy()[np.maximum(pos, 0)], None)
    return d


def by_regime(weekly):
    """weekly: index=형성일, 열=주별 값(IC 차이·순초과수익 등). 국면별 평균과 주 수."""
    r = regimes(weekly.index).set_index("date")
    out = {}
    for col in ["cpi_regime", "rate_regime"]:
        g = weekly.join(r[col]).groupby(col)
        out[col] = g.mean().join(g.size().rename("weeks"))
    return pd.concat(out, names=["기준", "국면"])


if __name__ == "__main__":
    r = regimes(["2022-08-05", "2020-05-08", "2008-11-07", "2010-07-09"])
    assert list(r.cpi_regime) == ["물가≥2%", "물가<2%", "물가≥2%", "물가≥2%"], r
    assert r.rate_regime.iloc[0] == "인상 후" and r.rate_regime.iloc[1] == "인하 후" and r.rate_regime.iloc[2] == "인하 후", r
    print(r.to_string())
    print("m02 자체 점검 통과")

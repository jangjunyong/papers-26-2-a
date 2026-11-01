"""DataGuide 6 일별 엑셀 → 분석용 표 (학교 라이선스 데이터, 논문용 기본 경로).

원본: PAPERS/data/원본_데이터/0_data/ (KOSPI 상장+상폐 1,320종목, 2000-01-04~2026-04)
엑셀 형식: 9행 = 종목코드(A005930), 13행 = 아이템명, 15행부터 = 날짜 × 종목 값.
캐시: PAPERS/data/daily/dg/<이름>.parquet (날짜 × 6자리 코드)
"""
import pathlib
import sys

import numpy as np
import openpyxl
import pandas as pd

PAPERS = pathlib.Path(__file__).resolve().parents[3]
RAW = PAPERS / "data" / "원본_데이터" / "0_data"
CACHE = PAPERS / "data" / "daily" / "dg"
FILES = {"open": "01_price/open_price.xlsx", "high": "01_price/high_price.xlsx", "low": "01_price/low_price.xlsx",
         "close": "01_price/close_price.xlsx", "adj_close": "01_price/share_price.xlsx",
         "volume": "03_volume/trading_volume.xlsx", "shares": "02_shares/listed_shares_common.xlsx",
         "halt": "08_others/is_trading_halted.xlsx"}


def read_xlsx(path):
    rows = openpyxl.load_workbook(path, read_only=True, data_only=True).active.iter_rows(values_only=True)
    head = [next(rows) for _ in range(14)]
    data = [r for r in rows if r[0] is not None]
    codes = [str(c)[1:] if c else None for c in head[8][1:]]       # A005930 → 005930
    df = pd.DataFrame([r[1:] for r in data], index=pd.to_datetime([r[0] for r in data]), columns=codes)
    df = df.loc[:, [c is not None for c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.replace({"정지": 1, "정상": 0}).apply(pd.to_numeric, errors="coerce")   # 거래정지여부는 문자열


def load(name):
    return pd.read_parquet(CACHE / f"{name}.parquet")


def frames(start="1999-12-01", end="2026-12-31"):
    """종목별 수정 OHLCV {코드: DataFrame(open, high, low, close, volume)}.
    수정 비율 = 수정주가 / 종가. 시가·고가·저가에 곱하고 거래량은 나눈다. 거래정지일은 NaN으로 둔다."""
    t = {k: load(k).loc[start:end] for k in ("open", "high", "low", "close", "adj_close", "volume", "halt")}
    cols = t["close"].columns.intersection(t["adj_close"].columns)
    f = (t["adj_close"][cols] / t["close"][cols]).replace([np.inf, -np.inf], np.nan)
    halt = t["halt"].reindex(columns=cols).fillna(0) > 0
    out = {}
    for c in cols:
        d = pd.DataFrame({k: t[k][c] if c in t[k] else np.nan for k in ("open", "high", "low", "close", "volume")})
        d[["open", "high", "low", "close"]] = d[["open", "high", "low", "close"]].mul(f[c], axis=0)
        d["volume"] = d.volume / f[c]
        d.loc[halt[c], ["open", "high", "low", "volume"]] = np.nan
        d["value"] = d.close * d.volume          # ponytail: 거래대금 근사(종가×거래량). DataGuide 거래대금 받으면 교체
        d = d.dropna(subset=["close"])
        if len(d):
            out[c] = d
    return out


if __name__ == "__main__":  # 사용: py -3 dg.py open close ... (인자 없으면 전부) → 캐시 생성
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or FILES:
        df = read_xlsx(RAW / FILES[name])
        df.to_parquet(CACHE / f"{name}.parquet")
        print(name, df.shape, df.index.min().date(), df.index.max().date(), flush=True)

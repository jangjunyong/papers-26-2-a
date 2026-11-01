"""Kaggle v3 묶음 (사전고정_v1 1·3절). 원본 가격은 올리지 않고 파생 이미지만 비공개로 올린다.

  d_train : 매일 형성 2000-01 ~ 2007-12 (5일 라벨이 2007-12-31 안에서 끝나는 것만)
  d_val   : 매일 형성 2008-01 ~ 2009-12 (라벨이 2009-12-31 안에서 끝나는 것만) — 조기 종료용
  w_infer : 주 마지막 거래일 형성 2010-01 ~ 2026-05-25, **수익 라벨 없음**(시험 구간 보호, 추론만)
가격은 panel_v3 의 수정주가 비율(adj_close/close)로 시가·고가·저가를 곱하고 거래량은 나눈다(dg.frames 와 같은 규칙).
출력: PAPERS/data/kaggle_upload_v3/<묶음>.npz (x=packbits 480바이트, date, ticker[, ret5])
사용: py -3 kaggle_export_v3.py [묶음 ...]   (프로세스 3개, 노트북 CPU)
"""
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import charts

PAPERS = pathlib.Path(__file__).resolve().parents[3]
PANEL = PAPERS / "data" / "daily" / "club_db" / "panel_v3.parquet"
OUT = PAPERS / "data" / "kaggle_upload_v3"
DAYS, H = 20, 5
SETS = {  # 이름: (형성 시작, 형성 끝, 주기, 라벨 종료 상한 또는 None)
    "d_train": ("2000-01-01", "2007-12-31", "D", "2007-12-31"),
    "d_val": ("2008-01-01", "2009-12-31", "D", "2009-12-31"),
    "w_infer": ("2010-01-01", "2026-05-25", "W", None),
}


def frames():
    p = pd.read_parquet(PANEL, columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "is_halted"])
    f = (p.adj_close / p.close).replace([np.inf, -np.inf], np.nan)
    p[["open", "high", "low", "close"]] = p[["open", "high", "low", "close"]].mul(f, axis=0)
    p["volume"] = p.volume / f
    p.loc[p.is_halted == 1, ["open", "high", "low", "volume"]] = np.nan
    p = p.dropna(subset=["close"])
    cal = pd.DatetimeIndex(np.sort(p.date.unique()))
    return {k: d.set_index("date")[["open", "high", "low", "close", "volume"]] for k, d in p.groupby("ticker")}, cal


def _one(args):
    tk, d, form, cut = args
    ok = (d[["open", "high", "low", "volume"]] > 0).all(axis=1).to_numpy()
    ma = d.close.rolling(DAYS).mean().to_numpy()
    fwd = (d.close.shift(-H) / d.close - 1).to_numpy()
    end_dt = d.index.to_series().shift(-H).to_numpy()            # 라벨이 끝나는 날(이 종목의 5거래일 뒤)
    xs, rows = [], []
    for p in d.index.get_indexer(form):
        if p < 2 * DAYS - 1:                                      # p = -1(그날 거래 없음)도 여기서 빠진다
            continue
        if cut is not None and (np.isnan(fwd[p]) or end_dt[p] > cut):
            continue
        sl = slice(p - DAYS + 1, p + 1)
        if not ok[sl].all():
            continue
        w = d.iloc[sl]
        img = charts.chart_image(w.open, w.high, w.low, w.close, w.volume, ma[sl])
        xs.append(np.packbits((img > 0).reshape(-1)))
        rows.append((d.index[p], tk, fwd[p] if cut is not None else np.nan))
    return xs, rows


def export(fr, cal, name):
    a, b, freq, cut = SETS[name]
    form = cal.to_series()
    if freq == "W":
        form = form.groupby(cal.to_period("W")).max()
    form = pd.DatetimeIndex(form[(form >= a) & (form <= b)].to_numpy())
    cut = None if cut is None else np.datetime64(cut)
    xs, rows = [], []
    with Pool(3) as pool:
        for i, (x, r) in enumerate(pool.imap(_one, ((tk, d, form, cut) for tk, d in fr.items()), chunksize=8)):
            xs += x
            rows += r
            if i % 200 == 0:
                print(f"  {name}: {i}/{len(fr)}종목 {len(xs):,}장", flush=True)
    m = pd.DataFrame(rows, columns=["date", "ticker", "ret5"])
    out = dict(x=np.stack(xs), date=m.date.to_numpy().astype("datetime64[D]").astype(str), ticker=m.ticker.to_numpy().astype(str))
    if cut is not None:
        out["ret5"] = m.ret5.to_numpy(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", **out)
    print(f"{name}: {len(m):,}장, 형성일 {m.date.nunique()}, 종목 {m.ticker.nunique()}, {m.date.min().date()}~{m.date.max().date()}"
          + (f", 상승 비율 {(m.ret5 > 0).mean():.3f}" if cut is not None else ""), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fr, cal = frames()
    for name in sys.argv[1:] or list(SETS):
        export(fr, cal, name)

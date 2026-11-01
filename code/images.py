"""종목별 일봉 → 주 단위 20일 차트 이미지 + 다음 5거래일 수익 라벨 (계획 9단계)."""
import numpy as np
import pandas as pd

import charts


def build(frames, start, end, days=20, horizon=5, freq="W"):
    """frames: {ticker: DataFrame(index=날짜, open/high/low/close/volume, 수정주가)}.
    형성일 = freq "W"면 시장 달력 기준 주 마지막 거래일, "D"면 매 거래일(JBK 방식, 라벨이 겹친다).
    창 안에 거래정지(시가·고가·저가·거래량 0)가 있으면 뺀다.
    반환: (uint8 이미지 N×H×W, meta[date, ticker, ret])."""
    cal = pd.DatetimeIndex(sorted(set().union(*(d.index for d in frames.values()))))
    form = cal.to_series()
    if freq == "W":
        form = form.groupby(cal.to_period("W")).max()              # 전체 달력으로 주 끝을 정한 뒤 자른다
    form = form[(form >= start) & (form <= end)].values            # (먼저 자르면 구간 끝이 주 중간에서 잘린다)

    imgs, rows = [], []
    for tk, d in frames.items():
        d = d.sort_index()
        ok = ((d[["open", "high", "low", "volume"]] > 0).all(axis=1)).values
        ma = d.close.rolling(days).mean().values
        fwd = (d.close.shift(-horizon) / d.close - 1).values
        # ponytail: 5일 안에 상장폐지되면 라벨이 없어 빠진다(생존편향). 8-2단계 상폐 수익 처리 때 보완.
        for p in d.index.get_indexer(form):
            if p < 2 * days - 1 or np.isnan(fwd[p]):  # p = -1(그 주에 거래 없음)도 여기서 빠진다
                continue
            sl = slice(p - days + 1, p + 1)
            if not ok[sl].all():
                continue
            w = d.iloc[sl]
            imgs.append(charts.chart_image(w.open, w.high, w.low, w.close, w.volume, ma[sl]))
            rows.append((d.index[p], tk, fwd[p]))
    meta = pd.DataFrame(rows, columns=["date", "ticker", "ret"])
    return np.stack(imgs), meta

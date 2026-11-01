"""A1: 전 종목 일별 패널 2000-01-01~2014-12-31 (계획 5-2단계).

  py -3 a1_panel.py collect k n   날짜를 n개(최대 3)로 나눠 k번째 몫을 받는다 → data/daily/raw_by_date/YYYYMMDD.parquet
  py -3 a1_panel.py build         종목별로 이어 붙여 수정주가 → data/daily/panel_2000_2014.parquet
  py -3 a1_panel.py check         기존 200종목과 대조 + 연도·시장별 커버리지 표

원자료는 pykrx 래퍼(get_market_ohlcv) 대신 그 아래 '전종목시세'(MDCSTAT01501)를 직접 부른다.
래퍼는 거래량을 int32로 바꾸고(오버플로 위험) 시장·종목명·시총을 버리기 때문이다.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily"
RAW = DATA / "raw_by_date"
PANEL = DATA / "panel_2000_2014.parquet"
START, END = "20000101", "20141231"  # 2015-01-01 이후는 사전등록한 시험 구간 → 절대 요청·저장하지 않는다

RAW_COLS = {"ISU_SRT_CD": "ticker", "ISU_ABBRV": "name", "MKT_NM": "market",
            "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
            "CMPPREVDD_PRC": "diff", "FLUC_RT": "chg", "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value",
            "MKTCAP": "mktcap", "LIST_SHRS": "shares"}
NUM = ["open", "high", "low", "close", "diff", "chg", "volume", "value", "mktcap", "shares"]
OUT_COLS = ["date", "ticker", "market", "open", "high", "low", "close", "volume", "value"]


def dates():
    # 토요일 장은 2000년 상반기까지 있었다 → 2000년은 토요일도 묻는다(휴장이면 .holiday로 기록)
    d = pd.date_range(START, END)
    d = d[(d.dayofweek < 5) | ((d.dayofweek == 5) & (d.year == 2000))]
    return d.strftime("%Y%m%d").tolist()


def fetch_day(d, fetch):
    """하루치 전 종목 시세. 휴장일이면 None(KRX는 휴장일에도 종목 목록을 '-' 값으로 준다). 빈 응답은 오류로 본다."""
    assert START <= d <= END, d
    df = fetch(d, "ALL")
    if df is None or len(df) == 0:
        raise RuntimeError(f"{d} 빈 응답")
    df = df[list(RAW_COLS)].rename(columns=RAW_COLS)
    for c in NUM:  # '1,234' → 1234, '-' → NaN, '-70' → -70
        df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    if df.close.isna().all():
        return None
    return df.assign(date=pd.Timestamp(d))


def collect(k, n):
    assert n <= 3, "KRX 동시 프로세스는 최대 3개"
    import krx  # noqa: F401  로그인
    from pykrx.website.krx.market.core import 전종목시세
    fetch = 전종목시세().fetch
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates()[k::n] if not (RAW / f"{d}.parquet").exists() and not (RAW / f"{d}.holiday").exists()]
    fails = 0
    for i, d in enumerate(todo, 1):
        for attempt in range(3):
            try:
                df = fetch_day(d, fetch)
                if df is None:
                    (RAW / f"{d}.holiday").touch()
                else:
                    df.to_parquet(RAW / f"{d}.parquet", index=False)
                fails = 0
                break
            except Exception as e:
                print(d, "재시도", attempt, repr(e)[:120], flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            fails += 1
            if fails >= 5:  # 로그인 만료·차단 등 → 멈추고 다시 실행하면 이어 받는다
                sys.exit(f"[{k}] 연속 실패 {fails}회, 중단")
        if i % 50 == 0 or i == len(todo):
            print(f"[{k}] {i}/{len(todo)} {d}", flush=True)
        time.sleep(0.3)


def adjust(p):
    """krx.ohlcv와 같은 수정 방식을 long 표 전체에 한 번에 적용한다.
    기준가 B_t = 종가_t/(1+등락률_t/100), f_t = B_t/종가_{t-1}. |f_t-1|>1e-3이면 분할·권리락 등으로 보고
    그 이전 가격에 비율의 누적곱 k_t = Π_{s>t} f_s를 곱하고 거래량은 나눈다. 거래대금은 그대로 둔다."""
    p = p.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    f = p.close / (1 + p.chg / 100) / p.groupby("ticker").close.shift()
    f = f.where(np.isfinite(f) & (f > 0) & ((f - 1).abs() > 1e-3), 1.0)
    lf = np.log(f)
    k = np.exp(lf.groupby(p.ticker).transform("sum") - lf.groupby(p.ticker).cumsum())
    p[["open", "high", "low", "close"]] = p[["open", "high", "low", "close"]].mul(k, axis=0)
    p["volume"] = p.volume / k
    return p, f


def build():
    files = sorted(RAW.glob("*.parquet"))
    p = pd.concat([pd.read_parquet(f, columns=OUT_COLS + ["chg", "diff"]) for f in files], ignore_index=True)
    assert p.date.max() <= pd.Timestamp(END)
    nan_close = p.close.isna() | (p.close <= 0)
    print("파일", len(files), "행", len(p), "종가 없음(제외)", int(nan_close.sum()))
    p = p[~nan_close].sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    # 등락률(소수 2자리) 대신 정확한 '전일 대비'로 기준가를 잡아도 수정 사건이 같은지 확인
    prev = p.groupby("ticker").close.shift()
    ev_chg = (p.close / (1 + p.chg / 100) / prev - 1).abs() > 1e-3
    ev_diff = ((p.close - p["diff"]) / prev - 1).abs() > 1e-3
    bad = ev_chg != ev_diff
    print("수정 사건 등락률", int(ev_chg.sum()), "전일대비", int(ev_diff.sum()), "불일치", int(bad.sum()))
    if bad.any():
        print(p.loc[bad, ["date", "ticker", "close", "diff", "chg"]].assign(prev=prev[bad]).head(10).to_string())
    p, f = adjust(p)
    p[OUT_COLS].astype({"market": "category"}).to_parquet(PANEL, index=False)
    print("수정 사건", int((f != 1).sum()), "종목", p.ticker.nunique(), "→", PANEL)


def check():
    p = pd.read_parquet(PANEL)
    p["ticker"] = p.ticker.astype(str)
    # 1) 기존 200종목 대조. 수정 기준일이 다르므로(기존 2012-12-28, 패널 2014-12-30) 종목별 종가 비율이 상수인지 본다
    old_files = sorted((DATA / "ohlcv").glob("*.parquet"))
    g = dict(tuple(p[p.ticker.isin({f.stem for f in old_files})].groupby("ticker")))
    rows = []
    for f in old_files:
        old = pd.read_parquet(f)
        new = g.get(f.stem)
        if new is None:
            rows.append(dict(ticker=f.stem, old=len(old), common=0))
            continue
        j = old.join(new.set_index("date")[["close", "volume"]], rsuffix="_new", how="inner")
        r = j.close_new / j.close
        r = r / r.iloc[-1]
        v = (j.volume_new / j.volume * r)[j.volume > 0]  # 거래량은 가격과 반대로 수정된다
        rows.append(dict(ticker=f.stem, old=len(old), common=len(j), close_err=(r - 1).abs().max(),
                         vol_err=(v - 1).abs().max()))
    cmp_ = pd.DataFrame(rows)
    print("기존 200종목 대조: 종목", len(cmp_), "패널에 없음", int((cmp_.common == 0).sum()),
          "겹친 행", int(cmp_.common.sum()), "/ 기존 행", int(cmp_.old.sum()))
    print("종가 최대 상대오차", cmp_.close_err.max(), "거래량 최대 상대오차", cmp_.vol_err.max())
    print(cmp_.sort_values("close_err", ascending=False).head(5).to_string(index=False))

    # 2) 연도·시장별 커버리지
    cal = np.sort(p.date.unique())
    pos = pd.Series(np.arange(len(cal)), index=cal)
    life = p.groupby("ticker").date.agg(["min", "max"])
    p["year"] = p.date.dt.year
    ty = p.groupby(["ticker", "year"]).agg(market=("market", "last"), rows=("close", "size"),
                                            halted=("volume", lambda s: int((s == 0).sum()))).reset_index()
    a, b = pos[life["min"]].values, pos[life["max"]].values
    exp = []
    for y in range(2000, 2015):
        yd = pos[(cal >= np.datetime64(f"{y}-01-01")) & (cal <= np.datetime64(f"{y}-12-31"))]
        n = np.clip(np.minimum(b, yd.iloc[-1]) - np.maximum(a, yd.iloc[0]) + 1, 0, None)
        exp.append(pd.DataFrame({"ticker": life.index, "year": y, "expected": n}))
    ty = ty.merge(pd.concat(exp), on=["ticker", "year"])
    last = p.sort_values("date").groupby("ticker").tail(1)
    last = last.assign(year=last.date.dt.year)
    first = p.sort_values("date").groupby("ticker").head(1)
    first = first.assign(year=first.date.dt.year)
    tab = ty.groupby(["year", "market"], observed=True).agg(종목=("ticker", "size"), 행=("rows", "sum"),
                                                          기대행=("expected", "sum"), 정지=("halted", "sum"))
    tab["결측률%"] = (100 * (1 - tab.행 / tab.기대행)).round(2)
    tab["정지율%"] = (100 * tab.정지 / tab.행).round(2)
    tab["신규"] = first[first.date > cal[0]].groupby(["year", "market"], observed=True).size()
    tab["상폐"] = last[last.date < cal[-1]].groupby(["year", "market"], observed=True).size()
    tab = tab.fillna(0).astype({"신규": int, "상폐": int}).drop(columns=["기대행", "정지"])
    print(tab.to_string())
    print("전체 종목", len(life), "행", len(p), "거래일", len(cal), cal[0], "~", cal[-1],
          "| 상폐(패널 끝 전에 사라짐)", int((last.date < cal[-1]).sum()),
          "| 시장 이동", int((p.groupby("ticker").market.nunique() > 1).sum()),
          "| 중간에 20거래일 넘게 빠진 종목", int((p.groupby("ticker").date.apply(lambda s: pos[s].diff().max()) > 20).sum()))
    tab.to_csv(pathlib.Path(__file__).resolve().parents[1] / "results" / "a1_coverage.csv", encoding="utf-8-sig")


def selftest():
    """2:1 분할(기준가 51) 하루가 있는 가짜 종목 → 분할 전 종가는 반, 거래량은 두 배."""
    d = pd.DataFrame({"ticker": "X", "date": pd.date_range("2001-01-01", periods=4),
                      "open": 1.0, "high": 1.0, "low": 1.0, "close": [100.0, 102, 52, 53],
                      "chg": [np.nan, 2.0, (52 / 51 - 1) * 100, (53 / 52 - 1) * 100], "volume": 10.0})
    p, f = adjust(d)
    assert np.allclose(p.close, [50, 51, 52, 53]) and np.allclose(p.volume, [20, 20, 10, 10]), p
    assert (f != 1).sum() == 1
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"build": build, "check": check, "selftest": selftest}[cmd]()

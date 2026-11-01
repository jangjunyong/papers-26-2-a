"""bt: 주간 포트폴리오 백테스터 (사전고정_v1 5절). 함수형이고 상태는 run() 안에만 있다.

run(scores, panel, method, exit_rank, spread, constraints, n_top) -> 주별 결과 DataFrame
  scores: (date=형성일, ticker, score). 높을수록 좋다. 후보 범위 필터(2절)는 호출자가 이미 적용했다고 본다.
  panel : p02 패널 열 date, ticker, open, high, low, close, adj_close, f_tradable, liq_zone, limit_pct, mktcap.
          prep(panel) 결과(dict)를 넘기면 여러 번 돌릴 때 전처리를 한 번만 한다.
  결과 열: date 형성일, gross 총수익, cost 비용, net 순수익(= gross - cost), turnover 편도 회전율, n_hold 보유 종목 수,
          bench_vw·bench_ew 후보 전체 시가총액가중·동일가중(같은 체결 규칙, 비용 없음), excess 초과수익.
  보유 수익: 체결일(형성일 다음 거래일) 시가 -> 다음 체결일 시가. 비용·수익은 체결 직전 NAV 대비 비율.
  비용_t = sum|dw|(f + h) + sum max(-dw, 0) tau_y   (dw = 실제 체결된 비중 변화, 드리프트 반영)

제약 5개(샤플리, constraints 에 이름을 넣으면 켠다): short 공매도, conc 집중, limit 가격제한, fee 거래세·수수료, spread 스프레드.
  모두 끔 = 10분위 동일가중 롱숏, 비용 없음, excess = 롱숏 순수익 / short 켬 = 숏 레그 제거, excess = 순수익 - 시가총액가중
  conc 켬 = 상위 n_top(20) / limit 켬 = 상한가 시가 매수 금지·하한가 시가 매도 이월 / fee 켬 = 수수료 0.015% + 매도 거래세
  spread 켬 = h: "cs" Corwin-Schultz 절반 / "tick" 호가단위 절반 / "zero" 0

명세에 없어서 가장 단순하게 정한 것:
  1 체결 불가 = 체결일 f_tradable==0(정지·거래량 0) 또는 시가 없음. 그 주 후보에서 빼고 상위 N 을 다시 고른다(2절).
  2 시가 상·하한가 = p02 exact_limits(전일 종가 기준)와 시가 비교. 판정 불가 행(p02 lim_elig: 재개일·권리락/분할일·정리매매)은 제한 없음.
  3 상한가로 못 산 비중은 그 주 현금. 하한가·정지로 못 판 매도는 다음 거래일부터 매일 시가에 다시 시도하고 판 돈은 다음 체결일까지 현금.
    숏 레그도 부호로 같은 규칙(신규 공매도 = 매도 이월, 환매수 = 매수 금지).
  4 매도가 막혀 현금이 모자라면 매수를 같은 비율로 줄인다(차입 없음).
  5 비용의 dw 는 실제 체결분. 이월 매도는 체결일의 세율·스프레드를 쓴다. 회전율 = sum|체결액| / NAV / 2.
  6 거래세 = transaction_tax*.csv KOSPI 행, 매도일 기준(표 끝 이후는 마지막 값). 2000-01~2001-06 액면가 이하 매도 면세는 액면가 자료가 없어 무시.
  7 CS: 수정 고가·저가, 2일 추정치가 음수면 평균 전에 0, 두 날 중 하루라도 거래가 없으면 그 추정치 결측, 체결일 전날까지 20거래일 평균
    (있는 것만), 야간 수익 보정 없음. 결측이면 호가단위 절반으로 대신. 호가단위 절반 = 0.5 * 호가단위(체결일 시가) / 시가.
  8 상장폐지(종목 마지막 행이 패널 끝보다 앞섬) = 마지막 수정종가로 다음 체결일에 현금화. 가치가 그날부터 고정이라 결과는 그날 청산과 같다. 청산 비용 없음.
  9 (b) 보유 수는 N 고정: 이탈 기준 안의 보유 종목 유지 + 미보유 최상위로 채움. 후보에서 빠진 보유 종목은 순위 밖으로 보고 판다. 매주 동일가중으로 다시 맞춘다.
 10 (c) 코호트 = 그 주 상위 N 동일가중 x 25%. 매주 코호트·종목 비중을 다시 맞춘다(Jegadeesh-Titman 1993 방식). 첫 3주는 빈 코호트 몫이 현금.
 11 집중 끔 = 레그당 후보 수 // 10 (최소 1). (b)의 이탈 기준은 레그 크기에 비례(ceil(X * n / n_top)). 롱숏은 롱 +1, 숏 -1.
 12 벤치마크도 같은 가격제한(제약 켬일 때)·정지·상폐 처리. 시가총액가중 비중은 형성일 mktcap.
 13 마지막 형성일의 보유 끝 = 다음 ISO 주 마지막 거래일의 다음 거래일. 패널이 거기까지 없으면 그 주는 버린다. 동점은 종목코드 순.
실행: py -3 bt.py   -> 2010-01~2012-05 실데이터 스모크(시험 구간 보호: 2012-05-31 이후 행은 읽지 않는다)
"""
import itertools
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from p02_build_panel_v2 import CB, IN, exact_limits, make_tick_eras, tick_of  # noqa: E402  (호가단위·상하한가 산식 재사용)

FEE = 0.00015                       # 수수료 편도 0.015%
N_TOP = 20                          # 집중: 상위 20종목
N_COHORT = 4                        # (c) 코호트 수 = 보유 주 수
CS_WIN = 20                         # Corwin-Schultz 평균 창(거래일)
ALL = frozenset({"short", "conc", "limit", "fee", "spread"})
CAP = pd.Timestamp("2012-05-31")    # 스모크 실행의 시험 구간 보호

_tk = pd.read_csv(f"{IN}/tick_size.csv", parse_dates=["effective_from", "effective_to"])
ERAS = make_tick_eras(_tk[_tk.market == "KOSPI"])   # 2010-10-03 이전은 p02 구호가표(= tick_size_2000_2014.csv)
_tx = pd.concat([pd.read_csv(f"{IN}/{f}", parse_dates=["effective_from"]) for f in ("transaction_tax_2000_2014.csv", "transaction_tax.csv")])
_tx = _tx[_tx.market == "KOSPI"].sort_values("effective_from")


def sell_tax(dates):
    """매도일 -> 매도 측 세율(비율)."""
    i = np.searchsorted(_tx.effective_from.values, np.asarray(dates, "datetime64[ns]"), side="right") - 1
    return _tx.total_sell_side_pct.values[np.clip(i, 0, None)] / 100


def corwin_schultz(panel):
    """Corwin-Schultz(2012) 스프레드(가격 대비 비율): 2일 고저 추정치(음수 -> 0)의 그날까지 20거래일 평균. panel 과 같은 index."""
    p = panel.sort_values(["ticker", "date"])
    g = p.ticker
    fac = p.adj_close / p.close
    h, l = np.log(p.high * fac), np.log(p.low * fac)
    h1, l1 = h.groupby(g).shift(1), l.groupby(g).shift(1)
    beta = (h - l) ** 2 + (h1 - l1) ** 2
    gamma = (np.maximum(h, h1) - np.minimum(l, l1)) ** 2
    k = 3 - 2 * np.sqrt(2)
    a = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = (2 * (np.exp(a) - 1) / (1 + np.exp(a))).clip(lower=0)
    s = s.where((p.f_tradable == 1) & (p.f_tradable.groupby(g).shift(1) == 1))
    return s.groupby(g).rolling(CS_WIN, min_periods=1).mean().droplevel(0).reindex(panel.index)


def prep(panel, tickers=None):
    """긴 패널 -> 날짜 x 종목 배열 묶음(dict). run() 을 여러 번 돌릴 때 한 번 만들어 넘긴다.
    ponytail: 조밀 배열이라 2000~2026 전 종목이면 수백 MB. 모자라면 tickers 로 점수가 있는 종목만 넘긴다(run 은 그렇게 한다)."""
    cal = np.sort(panel.date.unique())
    p = panel if tickers is None else panel[panel.ticker.isin(tickers)]
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)
    fac = p.adj_close / p.close
    trad = (p.f_tradable == 1) & (p.open > 0)
    prev = g.close.shift(1)
    elig = (prev.notna() & (g.f_tradable.shift(1) == 1) & trad & (p.liq_zone == 0)
            & ((fac / fac.groupby(p.ticker).shift(1) - 1).abs() < 0.01))           # p02 lim_elig 와 같은 조건
    up, dn = exact_limits(prev.fillna(1).values, p.date, p.limit_pct.values, ERAS)
    h_tick = 0.5 * tick_of(p.open.values, p.date, ERAS) / p.open
    h_cs = corwin_schultz(p).groupby(p.ticker).shift(1) / 2                          # 체결일 전날까지의 추정치, 절반
    x = pd.DataFrame({"date": p.date, "ticker": p.ticker, "trad": trad * 1.0,
                      "ou": ((p.open >= up) & elig) * 1.0, "od": ((p.open <= dn) & elig) * 1.0,
                      "o": (p.open * fac).where(trad), "c": p.adj_close.where(trad),
                      "h_tick": h_tick, "h_cs": h_cs.fillna(h_tick), "mktcap": p.mktcap})

    def wide(c):
        return x.pivot(index="date", columns="ticker", values=c).reindex(cal)

    o, c = wide("o"), wide("c").ffill()                                              # 정지·상폐 뒤에는 마지막 거래일 종가로 고정
    trad_w = wide("trad").to_numpy() == 1
    cols = o.columns
    last = pd.Index(cal).get_indexer(g.date.max().reindex(cols))
    iso = pd.DatetimeIndex(cal).isocalendar()
    wk = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    nz = lambda a: np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0)        # noqa: E731
    return dict(dates=cal, tickers=cols, trad=trad_w, ou=wide("ou").to_numpy() == 1, od=wide("od").to_numpy() == 1,
                mark=nz(np.where(trad_w, o, c)), h_tick=nz(wide("h_tick")), h_cs=nz(wide("h_cs")), mktcap=nz(wide("mktcap")),
                last=last, ended=last < len(cal) - 1, week_end=np.flatnonzero(np.r_[wk[1:] != wk[:-1], True]), tax=sell_tax(cal))


def _sim(M, ex, tgt, limit, fee=0.0, h=None, tax=None):
    """체결 엔진(주식 수 단위). ex[k] = 주 k 체결일 인덱스, 보유 = ex[k] 시가 -> ex[k+1] 시가. tgt(k, w_pre) -> 목표 비중.
    반환 (K, 5): gross, cost, net, turnover, n_hold."""
    mark, trad, ou, od = M["mark"], M["trad"], M["ou"], M["od"]
    u, cash = np.zeros(mark.shape[1]), 1.0
    out = np.zeros((len(ex) - 1, 5))
    for k in range(len(ex) - 1):
        d = ex[k]
        px = mark[d]
        dead = (u != 0) & M["ended"] & (M["last"] < d)                  # 상장폐지: 마지막 수정종가로 현금화
        cash += u[dead] @ px[dead]
        u[dead] = 0
        nav0 = u @ px + cash
        w_pre = u * px / nav0
        w = tgt(k, w_pre)
        dv = (w - w_pre) * nav0
        buy = (dv > 0) & trad[d] & ~(limit & ou[d])
        sell = (dv < 0) & trad[d] & ~(limit & od[d])
        b, s = np.where(buy, dv, 0.0), np.where(sell, -dv, 0.0)
        if b.sum() > cash + s.sum() + 1e-12:                             # 차입 없음: 현금이 모자라면 매수를 같은 비율로 줄인다
            b *= max(cash + s.sum(), 0.0) / b.sum()
        safe = np.where(px > 0, px, 1.0)
        pend = np.where((dv < 0) & ~sell, np.where(w == 0, -u, dv / safe), 0.0)   # 이월 매도(주식 수)
        u = u + (b - s) / safe
        u[np.abs(u * px) < 1e-12 * abs(nav0)] = 0                        # 전량 매도 뒤 부동소수 잔차 제거
        cost = ((b + s) * (fee + (0.0 if h is None else h[d]))).sum() + s.sum() * (0.0 if tax is None else tax[d])
        cash += s.sum() - b.sum() - cost
        traded, n_hold = b.sum() + s.sum(), np.count_nonzero(u)
        for dd in range(d + 1, ex[k + 1]):                               # 이월 매도: 다음 거래일부터 매일 시가에 다시 시도
            if not pend.any():
                break
            ok = (pend != 0) & trad[dd] & ~(limit & od[dd])
            v = -pend[ok] * mark[dd][ok]
            c = (v * (fee + (0.0 if h is None else h[dd][ok]))).sum() + v.sum() * (0.0 if tax is None else tax[dd])
            u[ok] += pend[ok]
            pend[ok] = 0
            cash += v.sum() - c
            cost, traded = cost + c, traded + abs(v).sum()
        net = (u @ mark[ex[k + 1]] + cash) / nav0 - 1
        out[k] = net + cost / nav0, cost / nav0, net, traded / nav0 / 2, n_hold
    return out


def _top(s, n):
    """점수 상위 n 종목 인덱스(결측 = 후보 아님, 동점은 종목 순)."""
    i = np.flatnonzero(~np.isnan(s))
    return i[np.argsort(-s[i], kind="stable")[:n]]


def _eq(idx, size):
    w = np.zeros(size)
    if len(idx):
        w[idx] = 1 / len(idx)
    return w


def _band(s, n, x, held):
    """완충 한 레그: 보유 중 순위 x 안이면 유지, 나머지는 미보유 최상위로 n 개까지 채운다."""
    order = _top(s, len(s))
    keep = order[held[order] & (np.arange(1, len(order) + 1) <= x)][:n]
    return _eq(np.r_[keep, order[~held[order]][: n - len(keep)]].astype(int), len(s))


def run(scores, panel, method="a", exit_rank=None, spread="cs", constraints=ALL, n_top=N_TOP):
    """주간 백테스트 한 번. method: "a" 전량 교체, "b" 완충(exit_rank 필요), "c" 4코호트. 반환 열은 모듈 docstring."""
    C = frozenset(constraints)
    assert C <= ALL and method in ("a", "b", "c") and spread in ("cs", "tick", "zero"), (C, method, spread)
    assert method != "b" or exit_rank, "완충(b)은 exit_rank 가 필요하다"
    M = panel if isinstance(panel, dict) else prep(panel, scores.ticker.unique())
    cal, N = M["dates"], len(M["tickers"])
    f = np.sort(scores.date.unique())
    fi = np.searchsorted(cal, f)
    assert (fi < len(cal)).all() and (cal[np.minimum(fi, len(cal) - 1)] == f).all(), "형성일이 패널 거래일이 아니다"
    we = M["week_end"]
    ex = np.append(fi, we[we > fi[-1]][:1]) + 1                         # 체결일 = 형성일 다음 거래일
    ex = ex[ex < len(cal)]                                               # 패널 끝을 넘는 주는 버린다
    K = len(ex) - 1
    S = np.full((K, N), np.nan)
    r, c = np.searchsorted(f, scores.date.values), M["tickers"].get_indexer(scores.ticker)
    m = (r < K) & (c >= 0)
    S[r[m], c[m]] = scores.score.values[m]
    S[~M["trad"][ex[:-1]]] = np.nan                                      # 체결일에 살 수 없는 종목은 후보에서 뺀다
    ls, cand = "short" not in C, ~np.isnan(S)
    n = np.full(K, n_top) if "conc" in C else np.maximum(1, cand.sum(1) // 10)

    if method == "b":
        xk = np.full(K, exit_rank) if "conc" in C else np.ceil(n * exit_rank / n_top)

        def tgt(k, w_pre):
            w = _band(S[k], n[k], xk[k], w_pre > 0)
            return w - _band(-S[k], n[k], xk[k], w_pre < 0) if ls else w
    else:
        A = np.array([_eq(_top(S[k], n[k]), N) - (_eq(_top(-S[k], n[k]), N) if ls else 0) for k in range(K)]).reshape(K, N)
        if method == "c":
            A = np.array([A[max(0, k - N_COHORT + 1): k + 1].sum(0) / N_COHORT for k in range(K)]).reshape(K, N)
        tgt = lambda k, w_pre: A[k]                                      # noqa: E731

    limit = "limit" in C
    on = "fee" in C
    h = (M["h_cs"] if spread == "cs" else M["h_tick"]) if "spread" in C and spread != "zero" else None
    o = _sim(M, ex, tgt, limit, FEE if on else 0.0, h, M["tax"] if on else None)
    ew = cand / np.maximum(cand.sum(1, keepdims=True), 1)
    mc = np.where(cand, M["mktcap"][fi[:K]], 0.0)
    vw = mc / np.where(mc.sum(1, keepdims=True) > 0, mc.sum(1, keepdims=True), 1)
    res = pd.DataFrame(o, columns=["gross", "cost", "net", "turnover", "n_hold"])
    res.insert(0, "date", f[:K])
    res["n_hold"] = res.n_hold.astype(int)
    res["bench_vw"] = _sim(M, ex, lambda k, w: vw[k], limit)[:, 2]
    res["bench_ew"] = _sim(M, ex, lambda k, w: ew[k], limit)[:, 2]
    res["excess"] = res.net if ls else res.net - res.bench_vw
    return res


def shapley(v):
    """v: {frozenset(켠 제약): 값} (2^n 개 전부) -> {제약: 샤플리 값}. 값의 합 = v(전부) - v(없음)."""
    full = max(v, key=len)
    n = len(full)
    return {k: sum(math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n) * (v[S | {k}] - v[S])
                   for r in range(n) for S in map(frozenset, itertools.combinations(sorted(full - {k}), r)))
            for k in sorted(full)}


def _smoke():
    """2010-01~2012-05 실데이터 스모크. 점수는 5일 반전(-str5) 대용이며 모델이 아니다."""
    t0 = time.time()
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "f_tradable", "liq_zone", "limit_pct", "mktcap",
            "exclude_flag", "tv_filled", "str5"]
    pan = pd.read_parquet(f"{CB}/panel_v2.parquet", columns=cols, filters=[("date", "<=", CAP)])
    assert pan.date.max() <= CAP, "2012-05-31 이후 데이터가 들어왔다"
    cal = pd.DatetimeIndex(np.sort(pan.date.unique()))
    iso = cal.isocalendar()
    fdates = cal.to_series().groupby([iso.year.values, iso.week.values]).max()   # ISO 주 마지막 거래일
    pan = pan.sort_values(["ticker", "date"]).reset_index(drop=True)
    pan["tv60"] = pan.groupby("ticker").tv_filled.transform(lambda s: s.shift(1).rolling(60, min_periods=60).mean())
    u = pan[pan.date.isin(fdates) & (pan.f_tradable == 1) & (pan.exclude_flag == 0) & pan.tv60.notna() & pan.str5.notna()]
    u = u[u.groupby("date").tv60.rank(pct=True) > 0.2]                   # 2절: 직전 60일 평균 거래대금 하위 20% 제외
    scores = u.assign(score=-u.str5)[["date", "ticker", "score"]]
    t1 = time.time()
    M = prep(pan, scores.ticker.unique())
    t2 = time.time()
    rows = []
    for name, kw in [("a", {}), ("b40", dict(method="b", exit_rank=40)), ("b60", dict(method="b", exit_rank=60)),
                     ("b100", dict(method="b", exit_rank=100)), ("c", dict(method="c"))]:
        r = run(scores, M, **kw)
        rows.append(dict(방식=name, 주수=len(r), 총수익=r.gross.mean(), 비용=r.cost.mean(), 순수익=r.net.mean(), 초과수익=r.excess.mean(),
                         회전율=r.turnover.mean(), 보유수=r.n_hold.mean(), 시총가중=r.bench_vw.mean(), 동일가중=r.bench_ew.mean()))
    t3 = time.time()
    v = {frozenset(S): run(scores, M, constraints=S).excess.mean() for k in range(6) for S in itertools.combinations(sorted(ALL), k)}
    phi = shapley(v)
    t4 = time.time()
    print(f"형성일 {scores.date.nunique()}개 ({scores.date.min().date()} ~ {scores.date.max().date()}), 후보 중앙값 {scores.groupby('date').size().median():.0f}")
    print(pd.DataFrame(rows).round(5).to_string(index=False))
    print("샤플리((a), cs, 주 평균 초과수익):", {k: round(x, 5) for k, x in phi.items()},
          f"합 {sum(phi.values()):.5f} = v(전부) {v[ALL]:.5f} - v(없음) {v[frozenset()]:.5f}")
    print(f"소요: 읽기·점수 {t1 - t0:.1f}s, prep {t2 - t1:.1f}s, 5개 방식 {t3 - t2:.1f}s, 32조합 {t4 - t3:.1f}s, 합계 {t4 - t0:.1f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _smoke()

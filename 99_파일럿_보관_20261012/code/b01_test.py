"""b01_backtest 단위 테스트. 실행: python b01_test.py (pytest 불필요, 실패하면 AssertionError로 종료코드 1).
(1) 미래수익 신호 -> 비현실적 성과  (2) 무작위 신호 20시드 -> 샤프 ~ 0  (3) 3종목 손계산  (4) 상한가 진입 미체결 등 체결 제약
(5) 샤플리 효율성  (6) 룩어헤드: 미래 자료 교란·신호 지연·하루 앞당김
실제 패널을 쓰는 테스트는 2015~2019 구간을 읽는다 (수 초)."""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
import b01_backtest as bt

PPY = 252 / 5
FEE = 0.00015
OK = []


def ok(name, msg=""):
    OK.append(name)
    print(f"PASS {name} {msg}")


# ----------------------------------------------------------------------------- 3종목 미니 패널
def mini(ov=None):
    """3종목 x 9일. 형성일 D1, 진입 D2 시가, 청산 D6 종가(H=5). 수정계수 1. ov로 한 칸씩 바꾼다.
    A(점수3, 롱): 시가 100 -> 종가 110 / B(점수2): 50 -> 55 / C(점수1, 숏): 200 -> 190. 시총 A:B:C = 1:3:6."""
    dates = pd.bdate_range("2019-07-01", periods=9)
    close = {"000010": [100, 100, 102, 104, 106, 108, 110, 111, 112],
             "000020": [50, 50, 51, 52, 53, 54, 55, 56, 57],
             "000030": [200, 200, 198, 196, 194, 192, 190, 188, 186]}
    cap = {"000010": 1000.0, "000020": 3000.0, "000030": 6000.0}
    hs = {"000010": 0.001, "000020": 0.0005, "000030": 0.002}
    rows = []
    for tk, cl in close.items():
        for i, d in enumerate(dates):
            o = cl[i - 1] if i else cl[0]
            r = dict(date=d, ticker=tk, open=float(o), close=float(cl[i]), volume=1000.0, is_halted=0, ret_outlier=0,
                     mktcap=cap[tk], lim_up=0, lim_dn=0, limit_pct=30.0, sell_tax_pct=0.25, tick_pct=0.1, short_banned=0,
                     half_spread=hs[tk])
            r["adj_close"] = r["close"]
            rows.append(r)
    p = pd.DataFrame(rows)
    for (tk, i, col), v in (ov or {}).items():
        p.loc[(p.ticker == tk) & (p.date == dates[i]), col] = v
    p["high"] = np.maximum(p.open, p.close) * 1.01
    p["low"] = np.minimum(p.open, p.close) * 0.99
    sig = pd.DataFrame({"date": dates[1], "ticker": ["000010", "000020", "000030"], "score": [3.0, 2.0, 1.0]})
    return p, sig


def run_mini(ov=None, sig_fn=None, **kw):
    p, sig = mini(ov)
    if sig_fn:
        sig = sig_fn(sig)
    mk = bt.Market(p)
    tr = bt.make_trades(mk, sig)
    return tr, bt.run(tr, **kw)


def close(a, b, tol=1e-12):
    return abs(a - b) < tol


# 세율을 진입일(D2)과 청산일(D6)에 다르게 둬서 "숏은 진입일 세율, 롱은 청산일 세율"을 구분해 검증한다
TX = {(tk, i, "sell_tax_pct"): v for tk in ("000010", "000020", "000030") for i, v in ((2, 0.30), (6, 0.20))}
TAX_IN, TAX_OUT = 0.0030, 0.0020


def test_3_hand_calc():
    tr, r = run_mini(cons=())
    assert close(r.ret.iloc[0], 0.15) and r.nL.iloc[0] == 1 and r.nS.iloc[0] == 1, r
    assert close(tr.mkt[0], 0.01), tr.mkt                                  # 0.1*0.1 + 0.3*0.1 + 0.6*(-0.05)
    cl4 = FEE * (1 + 1.10) + TAX_OUT * 1.10                                # 롱: 매수 수수료 + 매도(1.10) 수수료·청산일 세금
    cs4 = FEE * (1 + 0.95) + TAX_IN * 1                                    # 숏: 진입 매도 세금(진입일 세율, 명목 1) + 환매수(0.95) 수수료
    cl5, cs5 = 0.001 * (1 + 1.10), 0.002 * (1 + 0.95)                      # 스프레드: 진입·청산 각각 반스프레드
    exp = {("c4",): 0.15 - cl4 - cs4, ("c5",): 0.15 - cl5 - cs5, ("c4", "c5"): 0.15 - cl4 - cs4 - cl5 - cs5}
    for cons, v in exp.items():
        _, r = run_mini(TX, cons=cons)
        assert close(r.ret.iloc[0], v), (cons, r.ret.iloc[0], v)
    _, r2 = run_mini(TX, cons=("c4", "c5"), spread_mult=2.0, fee=FEE * 0.5)  # 수수료 0.5배, 스프레드 2배
    assert close(r2.ret.iloc[0], 0.15 - (0.5 * FEE * (2.1 + 1.95) + TAX_OUT * 1.1 + TAX_IN) - 2 * (cl5 + cs5))
    _, r = run_mini(TX, cons=("c1", "c4"))                                 # 롱온리: 롱 순수익 - 시장
    assert close(r.ret.iloc[0], 0.10 - cl4 - 0.01), r.ret.iloc[0]
    _, r = run_mini(TX, cons=("c1", "c4"), c1_mode="raw")
    assert close(r.ret.iloc[0], 0.10 - cl4)
    _, r = run_mini(cons=(), borrow=0.04)                                  # 대차비용: 5거래일 x 4%/252
    assert close(r.ret.iloc[0], 0.15 - 0.04 * 5 / 252)
    _, r = run_mini(cons=("c2",), topn=20)                                 # 3종목이면 상위20 = 1:1
    assert close(r.ret.iloc[0], 0.15)
    # 수정주가: 이후 액면분할(계수 0.5) -> 수익 불변, 원주가 단절일에 시가 하한가 오판 없음
    ov = {("000010", i, "adj_close"): v * 0.5 for i, v in enumerate([100, 100, 102, 104, 106, 108, 110, 111, 112])}
    _, r = run_mini(ov, cons=("c3",))
    assert close(r.ret.iloc[0], 0.15) and r.nL.iloc[0] == 1
    ov = {("000010", i, "close"): v * 2 for i, v in enumerate([100, 100])}    # D0~D1 원주가 2배(분할 전), 수정종가는 그대로
    ov.update({("000010", 2, "open"): 100.0, ("000010", 1, "close"): 200.0, ("000010", 0, "close"): 200.0})
    _, r = run_mini(ov, cons=("c3",))
    assert r.nL.iloc[0] == 1, "원주가 -50% 단절을 하한가로 오판"
    # 배치 수익 = 같은 값이 3개 형성일에 나오는지(그룹 집계): 신호를 D1·D2 둘로 늘리면 두 행
    _, r = run_mini(cons=(), sig_fn=lambda s: pd.concat([s, s.assign(date=s.date + pd.offsets.BDay(1))]))
    assert len(r) == 2
    # 매매 상쇄: 같은 종목이 연속 두 배치(D1, D2 형성)에 있으면 앞 배치는 청산 비용, 뒤 배치는 진입 비용을 안 낸다
    _, r = run_mini(cons=("c4",), net_turnover=True, sig_fn=lambda s: pd.concat([s, s.assign(date=s.date + pd.offsets.BDay(1))]))
    tax = 0.0025
    g2, gs2 = 111 / 102 - 1, 188 / 198 - 1                                  # 2번째 배치: D3 시가 -> D7 종가
    b1 = (0.10 - FEE) + (0.05 - (FEE + tax))
    b2 = (g2 - (FEE + tax) * (1 + g2)) + (-gs2 - FEE * (1 + gs2))
    assert close(r.ret.iloc[0], b1) and close(r.ret.iloc[1], b2), (r.ret.tolist(), b1, b2)
    ok("(3) 3종목 손계산", f"LS 총수익 0.15, 비용 후 {exp[('c4','c5')]:.7f}")


def test_4_limits():
    # A가 진입일 시가 상한가(전일 종가 100 -> 시가 130, 한도 30%)
    ov = {("000010", 2, "open"): 130.0}
    tr, r = run_mini(ov, cons=("c3",))
    assert r.nL.iloc[0] == 0 and r.long.iloc[0] == 0.0, "상한가 시가 종목은 체결되지 않아야 한다"
    tr, r = run_mini(ov, cons=())
    assert r.nL.iloc[0] == 1 and close(r.long.iloc[0], 110 / 130 - 1), "제약을 끄면 체결"
    # 한도 근접 허용: 29.7% 상승 시가(기준 29.6% 이상)도 막힌다, 29.0%는 막히지 않는다
    for pxo, n in ((129.7, 0), (129.0, 1)):
        _, r = run_mini({("000010", 2, "open"): pxo}, cons=("c3",))
        assert r.nL.iloc[0] == n, (pxo, r.nL.iloc[0])
    # 숏 대상 C가 시가 하한가(200 -> 140): 공매도 불가
    _, r = run_mini({("000030", 2, "open"): 140.0}, cons=("c3",))
    assert r.nS.iloc[0] == 0
    # 거래정지·거래량 0이면 진입 불가
    sg = lambda s: s.assign(score=[1.0, 3.0, 2.0])                          # B가 롱 최상위, C가 최하위 -> 숏은 A
    for col, v in (("is_halted", 1), ("volume", 0.0)):
        _, r = run_mini({("000020", 2, col): v}, sig_fn=sg, cons=("c3",))
        assert r.nL.iloc[0] == 0, col
        _, r = run_mini({("000020", 2, col): v}, sig_fn=sg, cons=())
        assert r.nL.iloc[0] == 1
    # 롱 청산일(D6) 종가 하한가 -> D7 종가(111)로 이월: 수익 0.11
    _, r = run_mini({("000010", 6, "lim_dn"): 1}, cons=("c3",))
    assert close(r.long.iloc[0], 0.11), r.long.iloc[0]
    _, r = run_mini({("000010", 6, "lim_dn"): 1}, cons=())
    assert close(r.long.iloc[0], 0.10)
    # 숏 청산일 상한가 -> D7 종가(188)로 이월: 숏 수익 +0.06
    _, r = run_mini({("000030", 6, "lim_up"): 1}, cons=("c3",))
    assert close(r.short.iloc[0], 0.06), r.short.iloc[0]
    # 정지로 이틀 연속 청산 불가면 D8까지 이월(112)
    _, r = run_mini({("000010", 6, "is_halted"): 1, ("000010", 7, "volume"): 0.0}, cons=("c3",))
    assert close(r.long.iloc[0], 0.12), r.long.iloc[0]
    # 공매도 금지일 표시
    tr, _ = run_mini({("000010", 2, "short_banned"): 1, ("000020", 2, "short_banned"): 1, ("000030", 2, "short_banned"): 1})
    assert tr.ban[0]
    ok("(4) 상한가 진입 미체결·하한가 이월·정지", "")


# ----------------------------------------------------------------------------- 스프레드 공식
def test_cs():
    H = np.array([[102.0], [101.5], [103.0]])
    L = np.array([[99.0], [98.5], [100.0]])
    s = bt.cs_spread(H, L, np.ones((3, 1), bool))
    for t in (1, 2):                                                     # 스칼라로 독립 계산
        b = math.log(H[t, 0] / L[t, 0]) ** 2 + math.log(H[t - 1, 0] / L[t - 1, 0]) ** 2
        g = math.log(max(H[t, 0], H[t - 1, 0]) / min(L[t, 0], L[t - 1, 0])) ** 2
        k = 3 - 2 * math.sqrt(2)
        a = (math.sqrt(2 * b) - math.sqrt(b)) / k - math.sqrt(g / k)
        exp = max(2 * (math.exp(a) - 1) / (1 + math.exp(a)), 0.0)
        assert abs(s[t, 0] - exp) < 1e-12, (s[t, 0], exp)
    assert np.isnan(s[0, 0])
    # 하루가 무효면 그 날과 다음 날 추정이 비어야 한다
    s = bt.cs_spread(H, L, np.array([[True], [False], [True]]))
    assert np.isnan(s[1, 0]) and np.isnan(s[2, 0])
    # 매일 고저 101/100이 같은 범위면 alpha = ln(1.01) -> S = 2*0.01/2.01 (손계산)
    H2, L2 = np.full((3, 1), 101.0), np.full((3, 1), 100.0)
    assert abs(bt.cs_spread(H2, L2, np.ones((3, 1), bool))[2, 0] - 0.02 / 2.01) < 1e-12
    ok("CS 스프레드 공식", "")


# ----------------------------------------------------------------------------- 샤플리
def test_5_shapley():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(size=32)
        bt.check_efficiency(v)
    a = rng.normal(size=5)                                               # 가산 게임: 샤플리 = 각자 몫
    v = np.array([sum(a[i] for i in range(5) if m >> i & 1) for m in range(32)])
    assert np.allclose(bt.shapley(v), a)
    v = np.array([10.0 if (m & 1 and m & 2) else 0.0 for m in range(32)])    # c1·c2가 함께 있어야 10: 5씩 나눔
    assert np.allclose(bt.shapley(v), [5, 5, 0, 0, 0])
    mg = bt.marginals(v)
    assert mg["c1"]["first"] == 0 and mg["c1"]["last"] == 10 and mg["c1"]["min"] == 0
    # 엔진 출력에도 적용 (미니 패널 + 부트스트랩 표본)
    p, sig = mini()
    tr = bt.make_trades(bt.Market(p), sig)
    R = bt.all_subsets(tr)
    gap = bt.check_efficiency(R.iloc[0].to_numpy())
    # 실제 형태: 부트스트랩 재표본 평균·샤프에도 효율성
    Rn = np.random.default_rng(2).normal(0.001, 0.02, size=(120, 32))
    mb, sb = bt.block_boot(Rn, PPY, B=20)
    for b in range(20):
        bt.check_efficiency(mb[b])
        bt.check_efficiency(sb[b])
    ok("(5) 샤플리 효율성", f"합 = v(전부)-v(공집합) (오차<=1e-12, 미니패널 {gap:.1e})")


# ----------------------------------------------------------------------------- 실제 패널 기반
def real_env():
    P = bt.load_panel("2014-09-01", "2020-01-31")
    mk = bt.Market(P)
    dates = bt.rebalance_dates(mk.cal, "2015-01-01")
    dates = dates[dates <= "2019-12-31"]
    return P, mk, dates


def fwd_signal(P, dates, h=5, lag=0):
    """점수 = 형성일 t의 향후 h일 수정종가 수익(미래). lag>0이면 점수를 lag번째 이전 형성일의 값으로 쓴다(그때까지 이미 실현된 값)."""
    q = P.sort_values(["ticker", "date"])
    q["f"] = q.groupby("ticker")["adj_close"].shift(-h) / q["adj_close"] - 1
    q = q[q.date.isin(dates)][["date", "ticker", "f"]].dropna()
    if lag:
        nxt = {d: dates[i + lag] for i, d in enumerate(dates[:-lag])}
        q = q.assign(date=q.date.map(nxt)).dropna(subset=["date"])
    return q.rename(columns={"f": "score"})


def test_1_future(P, mk, dates):
    tr = bt.make_trades(mk, fwd_signal(P, dates))
    r = bt.run(tr, ())
    sh = bt.sharpe(r.ret, PPY)
    assert sh > 10, sh
    r4 = bt.run(tr, bt.CONS)                                             # 모든 제약을 켜도 여전히 매우 높아야 한다
    ok("(1) 미래수익 신호", f"비제약 샤프 {sh:.1f}, 주간 {r.ret.mean():.3%}; 전제약 샤프 {bt.sharpe(r4.ret, PPY):.1f}")
    return sh


def test_2_random(P, mk, dates):
    base = P[P.date.isin(dates)][["date", "ticker"]].reset_index(drop=True)
    shs, ts, nets = [], [], []
    for seed in range(20):
        sig = base.assign(score=np.random.default_rng(seed).normal(size=len(base)))
        tr = bt.make_trades(mk, sig)
        r = bt.run(tr, ()).ret
        shs.append(bt.sharpe(r, PPY))
        ts.append(r.mean() / r.std(ddof=1) * math.sqrt(len(r)))
        nets.append(bt.run(tr, bt.CONS).ret.mean())
    shs, ts = np.array(shs), np.array(ts)
    assert abs(shs.mean()) < 0.3, shs.mean()
    assert (np.abs(ts) < 2.5).sum() >= 18, ts
    assert np.mean(nets) < -0.002, "무작위 신호에 비용을 얹으면 뚜렷이 음수여야 한다"
    ok("(2) 무작위 신호 20시드", f"평균 샤프 {shs.mean():+.3f} (범위 {shs.min():+.2f}~{shs.max():+.2f}), |t|<2.5 {int((np.abs(ts) < 2.5).sum())}/20, 전제약 주간 {np.mean(nets):.3%}")


def test_6_lookahead(P, mk, dates, sh_future):
    # (6a) 교란: 한 날짜(D) 이후의 모든 가격·거래량을 흩뜨려도 D 이전에 청산이 끝난 배치의 결과는 한 글자도 안 바뀌어야 한다
    sig = bt.signal_rankavg(bt.load_panel("2014-09-01", "2020-01-31", extra=list(bt.SIGN)), dates)
    D = pd.Timestamp("2017-06-30")
    P2 = P.copy()
    P2[["open", "high", "low", "close", "adj_close", "volume"]] = P2[["open", "high", "low", "close", "adj_close", "volume"]].astype(float)
    rng = np.random.default_rng(0)
    m = P2.date > D
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        P2.loc[m, col] = P2.loc[m, col] * rng.uniform(0.5, 2.0, m.sum())
    for col in ("lim_up", "lim_dn", "is_halted", "ret_outlier"):
        P2.loc[m, col] = rng.integers(0, 2, m.sum())
    tr1, tr2 = bt.make_trades(mk, sig), bt.make_trades(bt.Market(P2), sig)
    iD = mk.cal.get_loc(D)
    # 엄밀: 이월(c3) 없는 제약 조합은 청산일(형성일+5)까지만 본다 -> 형성일 <= D-5 인 배치는 바이트까지 같아야 하고, D-4 형성분은 달라야 한다
    for cons in (("c4", "c5"), ("c1", "c2", "c4", "c5")):
        r1, r2 = bt.run(tr1, cons), bt.run(tr2, cons)
        cut = mk.cal[iD - 5]
        a, b = r1.loc[:cut], r2.loc[:cut]
        assert len(a) > 100 and np.array_equal(a.ret.to_numpy(), b.ret.to_numpy()), f"미래 자료가 과거 배치를 바꿨다 {cons}"
        nxt = r1.index[r1.index > cut][0]
        assert nxt > cut and mk.cal.get_loc(nxt) > iD - 5 and r1.loc[nxt, "ret"] != r2.loc[nxt, "ret"], "교란이 먹혔는지 대조"
    # 이월(c3)까지 켜면 이월 한도 10일만큼 더 본다
    r1, r2 = bt.run(tr1, bt.CONS), bt.run(tr2, bt.CONS)
    cut = mk.cal[iD - 5 - 10 - 1]
    assert np.array_equal(r1.loc[:cut].ret.to_numpy(), r2.loc[:cut].ret.to_numpy())
    assert not np.array_equal(r1.loc[D:].ret.to_numpy(), r2.loc[D:].ret.to_numpy())
    a = r1.loc[:cut]
    # (6b) 지연: 향후수익 신호를 한 형성일 늦춰 쓰면(이미 실현된 값) 성과가 사라져야 한다
    r_lag = bt.run(bt.make_trades(mk, fwd_signal(P, dates, lag=1)), ()).ret
    sh_lag = bt.sharpe(r_lag, PPY)
    assert sh_lag < 0.2 * sh_future and sh_lag < 6, (sh_lag, sh_future)
    # (6c) 하루 앞당김: 형성일 t에 t+1일 수익(다음날 시가->종가를 포함)을 점수로 주면 성과가 크게 부풀고, 제 시점(t일 수익)이면 작다
    q = P.sort_values(["ticker", "date"])
    g = q.groupby("ticker")["adj_close"]
    q["ahead"] = g.shift(-1) / q["adj_close"] - 1                        # t -> t+1
    q["now"] = q["adj_close"] / g.shift(1) - 1                           # t-1 -> t (형성일에 이미 알려진 값)
    sub = q[q.date.isin(dates)]
    sh_ahead = bt.sharpe(bt.run(bt.make_trades(mk, sub[["date", "ticker"]].assign(score=sub.ahead)), ()).ret, PPY)
    sh_now = bt.sharpe(bt.run(bt.make_trades(mk, sub[["date", "ticker"]].assign(score=sub.now)), ()).ret, PPY)
    assert sh_ahead > sh_now + 3 and sh_ahead > 3, (sh_ahead, sh_now)
    ok("(6) 룩어헤드", f"교란 후 과거 배치 {len(a)}개 동일; 향후수익 신호 샤프 {sh_future:.1f} -> 한 주 지연 {sh_lag:.2f}; 하루 앞당김 {sh_ahead:+.2f} vs 제시점 {sh_now:+.2f}")


def main():
    test_cs()
    test_3_hand_calc()
    test_4_limits()
    test_5_shapley()
    P, mk, dates = real_env()
    sh = test_1_future(P, mk, dates)
    test_2_random(P, mk, dates)
    test_6_lookahead(P, mk, dates, sh)
    print(f"\n전부 통과 ({len(OK)}개 항목): " + " | ".join(OK))


if __name__ == "__main__":
    main()

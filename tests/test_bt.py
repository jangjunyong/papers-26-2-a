"""bt.py 단위 테스트. 종목 2~6개, 거래일 16~30일의 합성 패널로 손 계산 값과 비교한다.
기본 날짜 2011-01-03(월)~: 가격제한 15%, 10000~50000원 호가 50원. d4·d9·d14 = 금요일(ISO 주 끝), d5·d10·d15 = 체결일.
실행: .venv/Scripts/python.exe -m pytest tests/test_bt.py   (pytest 없으면 py -3 tests/test_bt.py)
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import bt  # noqa: E402

F = bt.FEE
D0 = "2011-01-03"


def close(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-12), (a, b)


def mk(px, n=16, start=D0, shares=None):
    """px: {종목: 가격 목록 또는 상수}. 시가=고가=저가=종가=수정종가=가격, 전부 거래 가능."""
    dates = pd.bdate_range(start, periods=n)
    fr = []
    for t, p in px.items():
        p = np.broadcast_to(np.asarray(p, float), (n,)).copy()
        fr.append(pd.DataFrame({"date": dates, "ticker": t, "open": p, "high": p, "low": p, "close": p, "adj_close": p,
                                "f_tradable": 1, "liq_zone": 0, "limit_pct": np.where(dates < "2015-06-15", 15.0, 30.0),
                                "mktcap": p * (shares or {}).get(t, 1)}))
    return pd.concat(fr, ignore_index=True)


def step(a, b, at, n=16):
    return [a] * at + [b] * (n - at)


def day(P, i):
    return np.sort(P.date.unique())[i]


def put(P, t, days, **kv):
    m = (P.ticker == t) & P.date.isin([day(P, i) for i in days])
    for k, v in kv.items():
        P.loc[m, k] = v


def sc(P, i, d):
    return pd.DataFrame({"date": day(P, i), "ticker": list(d), "score": list(d.values())})


def scores(P, *pairs):
    return pd.concat([sc(P, i, d) for i, d in pairs], ignore_index=True)


def rnd(seed=0, n=30):
    """무작위 패널 6종목·30일 + 정지(C d12~13), 시가 상한가(B d15), 시가 하한가(D d20). 형성일 d4·9·14·19·24."""
    rng = np.random.default_rng(seed)
    tk = list("ABCDEF")
    P = mk({t: np.round(10000 * np.cumprod(1 + rng.normal(0, 0.03, n)), -1) for t in tk}, n=n)
    P["open"] = np.round(P.close * (1 + rng.normal(0, 0.01, len(P))), -1)
    put(P, "C", [12, 13], f_tradable=0)
    for t, i, m in (("B", 15, 1.2), ("D", 20, 0.8)):
        prev = P.loc[(P.ticker == t) & (P.date == day(P, i - 1)), "close"].iloc[0]
        put(P, t, [i], open=np.round(prev * m, -1))
    P["high"] = np.maximum(P.open, P.close) + 50
    P["low"] = np.minimum(P.open, P.close) - 50
    S = pd.concat([sc(P, i, dict(zip(tk, rng.normal(size=len(tk))))) for i in (4, 9, 14, 19, 24)], ignore_index=True)
    return P, S


def test_full_replacement_arithmetic():
    P = mk({"A": step(10000, 11000, 6), "B": step(10000, 9000, 11), "C": step(20000, 22000, 11), "D": 10000}, shares={"D": 2})
    S = scores(P, (4, dict(A=4, B=3, C=2, D=1)), (9, dict(A=1, B=4, C=3, D=2)))
    r = bt.run(S, P, "a", constraints={"short", "conc", "fee"}, n_top=2)
    assert len(r) == 2 and list(r.n_hold) == [2, 2]
    # 1주: A·B 50%씩 현금에서 매수(수수료만), A +10%, B 0
    close(r.gross[0], 0.05), close(r.cost[0], F), close(r.net[0], 0.05 - F), close(r.turnover[0], 0.5)
    close(r.bench_ew[0], 0.10 / 4), close(r.bench_vw[0], 0.10 * 10000 / 60000), close(r.excess[0], r.net[0] - r.bench_vw[0])
    # 2주: A 0.55 전량 매도(세 0.3%), B 0.5 -> nav0/2, C 0 -> nav0/2. B -10%, C +10% 라 총수익 0
    nav0 = 1.05 - F
    tr = 0.55 + (nav0 / 2 - 0.5) + nav0 / 2
    close(r.gross[1], 0), close(r.cost[1], (tr * F + 0.55 * 0.003) / nav0), close(r.turnover[1], tr / nav0 / 2)
    close(r.net[1], -r.cost[1]), close(r.bench_ew[1], 0)
    # 스프레드(호가 단위 절반 = 25원/시가) 포함 전부 켬. 가격제한은 이 자료에서 걸리지 않는다
    r = bt.run(S, P, "a", spread="tick", n_top=2)
    close(r.cost[0], F + 25 / 10000)
    nav0 = 1.05 - F - 25 / 10000
    c2 = 0.55 * (F + 25 / 11000) + (nav0 / 2 - 0.5) * (F + 25 / 10000) + nav0 / 2 * (F + 25 / 20000) + 0.55 * 0.003
    close(r.cost[1], c2 / nav0)
    # 제약 모두 끔 = 10분위 롱숏(4종목 -> 레그당 1), excess = 롱숏 순수익
    r = bt.run(S, P, "a", constraints=set())
    close(r.gross[0], 0.10), close(r.excess[0], 0.10)        # 롱 A +10%, 숏 D 0
    close(r.gross[1], -0.10), close(r.excess[1], -0.10)      # 롱 B -10%, 숏 A 0
    assert list(r.n_hold) == [2, 2]


def test_sell_tax_by_year():
    P = mk({"A": 10000, "B": 10000}, n=21, start="2020-12-14")     # 매도일 2020-12-28(0.25%), 2021-01-04(0.23%)
    S = scores(P, (4, dict(A=2, B=1)), (9, dict(A=1, B=2)), (14, dict(A=2, B=1)))
    r = bt.run(S, P, "a", constraints={"short", "conc", "fee"}, n_top=1)
    assert len(r) == 3
    close(r.cost[0], F)
    nav = 1 - F
    c2 = (1 + nav) * F + 1 * 0.0025
    close(r.cost[1], c2 / nav)
    nav2 = nav - c2
    c3 = (nav + nav2) * F + nav * 0.0023
    close(r.cost[2], c3 / nav2)


def test_limit_up_open_no_buy():
    P = mk({"A": [10000] * 5 + [11500] + [12000] * 10, "B": step(10000, 11000, 6), "C": 10000, "D": 10000})
    S = scores(P, (4, dict(A=4, B=3, C=2, D=1)), (9, dict(A=4, B=3, C=2, D=1)))
    on = bt.run(S, P, "a", constraints={"short", "conc", "limit"}, n_top=2)
    off = bt.run(S, P, "a", constraints={"short", "conc"}, n_top=2)
    rA = 12000 / 11500 - 1                                           # d5 시가 11500 = 상한가(10000 x 1.15)
    close(on.gross[0], 0.5 * 0.10), close(on.turnover[0], 0.25)
    assert on.n_hold[0] == 1                                         # A 몫 50% 는 현금
    close(off.gross[0], 0.5 * rA + 0.5 * 0.10)
    assert off.n_hold[0] == 2
    close(on.bench_ew[0], 0.10 / 4), close(off.bench_ew[0], (rA + 0.10) / 4)   # 벤치마크도 같은 체결 규칙


def test_limit_down_sell_deferred():
    P = mk({"A": step(10000, 8000, 10), "B": 10000, "C": 10000})
    put(P, "A", [10], open=8500, high=8500)                          # 체결일 d10 시가 = 하한가(10000 -> 8500)
    S = scores(P, (4, dict(A=3, B=2, C=1)), (9, dict(A=1, B=3, C=2)))
    on = bt.run(S, P, "a", constraints={"short", "conc", "limit"}, n_top=1)
    off = bt.run(S, P, "a", constraints={"short", "conc"}, n_top=1)
    close(on.net[0], -0.15), close(off.net[0], -0.15)
    # 켬: A 는 d11 시가 8000 에 팔리고 그 돈은 현금, B 는 살 현금이 없어 못 산다
    close(on.net[1], 0.80 / 0.85 - 1), close(on.turnover[1], 0.80 / 0.85 / 2)
    assert on.n_hold[1] == 1
    close(off.net[1], 0.0)                                           # 끔: d10 시가 8500 에 팔고 B 매수
    assert off.n_hold[1] == 1


def test_halt_price_frozen_and_resume():
    P = mk({"A": step(10000, 9000, 12), "B": 10000, "C": 10000})
    put(P, "A", range(7, 12), f_tradable=0)                          # d7~d11 정지, d12 재개 시가 9000
    S = scores(P, (4, dict(A=3, B=2, C=1)), (9, dict(A=9, B=5, C=1)))  # d10 에 A 는 정지라 후보에서 빠진다
    r = bt.run(S, P, "a", constraints={"short", "conc"}, n_top=1)
    close(r.net[0], 0.0)                                             # d10 평가가 = 정지 전 종가 10000
    close(r.net[1], -0.10), close(r.turnover[1], 0.9 / 2)            # 재개일 시가 9000 에 매도
    assert r.n_hold[1] == 1


def test_delisting_liquidation():
    P = mk({"A": [10000] * 6 + [7000, 5000, 3000] + [3000] * 7, "B": step(10000, 11000, 11)})
    P = P[~((P.ticker == "A") & (P.date > day(P, 8)))]               # A 는 d8 이 마지막 행(정리매매 종가 3000)
    S = scores(P, (4, dict(A=2, B=1)), (9, dict(B=1)))
    r = bt.run(S, P, "a", constraints={"short", "conc"}, n_top=1)
    close(r.net[0], -0.70), close(r.net[1], 0.10)                    # 남은 0.3 으로 B 를 산다
    assert r.n_hold[1] == 1


def test_buffer_rule():
    P = mk({t: step(10000, p, 11) for t, p in dict(A=10100, B=10200, C=10400, D=10800, E=11600).items()})
    S = scores(P, (4, dict(A=5, B=4, C=3, D=2, E=1)), (9, dict(C=5, D=4, A=3, B=2, E=1)))
    kw = dict(constraints={"short", "conc"}, n_top=2)
    close(bt.run(S, P, "b", exit_rank=3, **kw).gross[1], 0.5 * (0.01 + 0.04))   # A(3위) 유지, B(4위) 매도, 빈자리 C
    close(bt.run(S, P, "b", exit_rank=4, **kw).gross[1], 0.5 * (0.01 + 0.02))   # A·B 유지
    close(bt.run(S, P, "a", **kw).gross[1], 0.5 * (0.04 + 0.08))                # 전량 교체: C·D


def test_four_cohorts():
    P = mk({t: step(10000, p, 11) for t, p in dict(A=11600, B=10100, C=10200, D=10400, E=10800).items()})
    S = pd.concat([sc(P, 2 * i + 1, {t: 9 if t == "ABCDE"[i] else 1 for t in "ABCDE"}) for i in range(5)])
    r = bt.run(S, P, "c", constraints={"short", "conc"}, n_top=1)
    assert list(r.n_hold) == [1, 2, 3, 4, 4]
    close(r.gross[:4], 0)
    close(r.gross[4], 0.25 * (0.01 + 0.02 + 0.04 + 0.08))           # 1주 코호트(A)는 4주 뒤 빠진다
    close(r.turnover[0], 0.125), close(r.turnover[4], 0.25)


def test_shapley_known_game():
    a = dict(short=1.0, conc=2.0, limit=-0.5, fee=-3.0, spread=0.25)
    v = {frozenset(S): sum(a[k] for k in S) + 0.8 * ({"short", "conc"} <= set(S)) for r in range(6) for S in itertools.combinations(a, r)}
    phi = bt.shapley(v)
    for k, x in dict(a, short=1.4, conc=2.4).items():               # 상호작용 0.8 은 반씩
        close(phi[k], x)
    close(sum(phi.values()), v[frozenset(a)] - v[frozenset()])


def test_shapley_32_runs():
    P, S = rnd()
    M = bt.prep(P, S.ticker.unique())
    v = {frozenset(c): bt.run(S, M, constraints=c, n_top=2).excess.mean() for r in range(6) for c in itertools.combinations(sorted(bt.ALL), r)}
    assert len(v) == 32 and np.isfinite(list(v.values())).all() and v[bt.ALL] != v[frozenset()]
    close(sum(bt.shapley(v).values()), v[bt.ALL] - v[frozenset()])


def test_corwin_schultz():
    P = mk({"A": 100}, n=4)
    P["high"], P["low"] = 101.0, 99.0
    cs = bt.corwin_schultz(P)
    assert np.isnan(cs.iloc[0])
    close(cs.iloc[1:], 0.02)                                         # 고저 폭 일정 -> (H-L)/중간값 = 2%
    Q = mk({"A": 100}, n=2)
    Q["high"], Q["low"] = [100.0, 110.0], [99.0, 109.0]              # 큰 갭 -> 음수 추정치 -> 0
    assert bt.corwin_schultz(Q).iloc[1] == 0


def test_no_lookahead():
    P, S = rnd()
    rng = np.random.default_rng(1)
    P2, S2 = P.copy(), S.copy()
    fut = P2.date >= day(P, 17)
    f = rng.uniform(0.7, 1.3, fut.sum())
    for c in ("open", "high", "low", "close", "adj_close"):
        P2.loc[fut, c] = np.round(P2.loc[fut, c] * f, -1)
    later = S2.date >= day(P, 19)
    S2.loc[later, "score"] = rng.normal(size=later.sum())
    for kw in (dict(method="a"), dict(method="b", exit_rank=3), dict(method="c")):
        a, b = bt.run(S, P, n_top=2, **kw), bt.run(S2, P2, n_top=2, **kw)
        assert len(a) == 4                                           # 보유 끝 d10·d15·d20·d25 (d24 형성은 d30 이 없어 버림)
        pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2], check_exact=True)   # d17 이후를 바꿔도 앞 2주 불변
        assert not a.iloc[2:].equals(b.iloc[2:])                    # 바뀐 구간은 실제로 달라진다


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)

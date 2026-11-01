"""v02: b01 백테스터·제약별 샤플리 분해 적대적 검증 (읽기 전용 검증. 기존 파일은 고치지 않고 results/v02_tables/ 에 새로 쓴다).
실행: python v02_b01_review.py   (노트북 CPU, 수 분)
구성
  1. 독립 재구현(Ind): b01_backtest 를 쓰지 않고 패널 원자료(피벗)에서 주별 수익을 직접 계산 -> b01 과 대조
  2. 룩어헤드/정합 점검: CS 창 정렬, 교란 실험, 신호 시점
  3. 감사 버그(B1, B2, B7) 시나리오별 A 결과 재계산 (b01 엔진을 시나리오별 Market 으로 돌림)
  4. 해석 점검과 주별 분해
  5. 보고서 숫자 대 CSV 전수 대조
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS/26-2/03_차트CNN_연구"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b01_backtest as bt          # 대조 대상(검증 대상). 독립 재구현(Ind)은 이 모듈을 쓰지 않는다.

PRED = f"{ROOT}/results/kaggle/pred_d_lr1e-4.parquet"
FLAGS = f"{ROOT}/results/q01_panel_flags.parquet"
TAB = f"{ROOT}/results/b01_tables"
OUT = f"{ROOT}/results/v02_tables"
FEE = 0.00015


# =====================================================================================================
# 1. 독립 재구현
# =====================================================================================================
class Ind:
    """패널(long)에서 날짜x종목 피벗을 만들고 형성일 하나의 롱/숏 수익을 종목 단위로 직접 계산한다.
    정의(사양 그대로): 신호 = 형성일 종가에 확정, 10분위(상·하위 round-half-up(0.1n)개) 동일가중,
    진입 = 형성일 다음 거래일 시가 x (수정종가/원종가), 청산 = 형성일로부터 5번째 거래일 종가(수정종가)."""

    def __init__(self, P):
        P = P.sort_values(["date", "ticker"])
        pv = lambda c: P.pivot(index="date", columns="ticker", values=c)   # noqa: E731
        self.O, self.H, self.L, self.C, self.AC, self.V = (pv(c) for c in ("open", "high", "low", "close", "adj_close", "volume"))
        self.HALT, self.TAX, self.LIM, self.LU, self.LD = (pv(c) for c in ("is_halted", "sell_tax_pct", "limit_pct", "lim_up", "lim_dn"))
        self.MC, self.TKP, self.OUTL = pv("mktcap"), pv("tick_pct"), pv("ret_outlier")
        self.cal, self.col = self.O.index, self.O.columns
        self.n = len(self.cal)
        self.nan_tax = []
        self._cs = {}

    # ---- 보조
    def idle(self, x, j):
        if not np.isfinite(self.C.iat[x, j]):          # 그날 행이 없음(상폐 후): 이월할 거래일이 없다 -> 마지막 가격으로 평가
            return False
        return (self.HALT.iat[x, j] > 0) or (not self.V.iat[x, j] > 0)

    def carry(self, x, j, leg):
        """청산일 x 가 막혀 있으면(롱: 정지·하한가, 숏: 정지·상한가) 다음 거래일로, 최대 10일."""
        for _ in range(10):
            lim = (self.LD if leg == "L" else self.LU).iat[x, j] > 0
            if (self.idle(x, j) or lim) and x + 1 < self.n:
                x += 1
            else:
                break
        return x

    def entry_blocked(self, e, j, leg):
        L = self.LIM.iat[e, j] / 100
        oret = self.O.iat[e, j] / self.C.iat[e - 1, j] - 1                # 진입일 시가의 전일 종가 대비 등락 (진입일 정보)
        near = (oret >= L - 0.004) if leg == "L" else (oret <= -L + 0.004)
        return self.idle(e, j) or (near and abs(oret) <= L + 0.05 and self.V.iat[e, j] > 0)

    def cs_half(self, j, i):
        """종목 j 의 형성일 i(행 번호) 기준 Corwin-Schultz 반스프레드. 창 = 행 i-19..i, 각 일의 추정은 (d-1, d) 두 날의 고저가."""
        if (j, i) in self._cs:
            return self._cs[(j, i)]
        S = []
        for d in range(i - 19, i + 1):
            days = (d - 1, d)
            if not all(np.isfinite(self.H.iat[x, j]) and self.V.iat[x, j] > 0 and self.HALT.iat[x, j] == 0 and self.H.iat[x, j] > self.L.iat[x, j] for x in days):
                continue
            f1, f2 = (self.AC.iat[x, j] / self.C.iat[x, j] for x in days)
            if abs(f2 / f1 - 1) >= 1e-6:                                  # 수정계수가 바뀐 날(액면분할 등)은 뺀다
                continue
            (h1, l1), (h2, l2) = ((self.H.iat[x, j], self.L.iat[x, j]) for x in days)
            beta = math.log(h1 / l1) ** 2 + math.log(h2 / l2) ** 2
            gamma = math.log(max(h1, h2) / min(l1, l2)) ** 2
            k = 3 - 2 * math.sqrt(2)
            alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
            S.append(max(2 * math.tanh(alpha / 2), 0.0))                  # 2(e^a-1)/(1+e^a) = 2 tanh(a/2)
        if len(S) >= 10:
            self._cs[(j, i)] = 0.5 * float(np.mean(S))
        else:
            t = self.TKP.iat[i, j]
            self._cs[(j, i)] = t / 200 if np.isfinite(t) else 0.0
        return self._cs[(j, i)]

    def px_at(self, x, j):
        v = self.AC.iat[x, j]
        if np.isfinite(v):
            return v
        seg = self.AC.iloc[max(0, x - 20):x + 1, j].dropna()               # 상폐·결측: 마지막 가격(최대 20행)
        return seg.iloc[-1] if len(seg) else np.nan

    # ---- 형성일 하나
    def week(self, date, sig, c1=False, c3=False, c4=False, c5=False, spread="cs", drop_outliers=True, topn=None, exact_flow=False):
        i = self.cal.get_loc(pd.Timestamp(date))
        e, x0 = i + 1, i + 5
        s = sig[sig.date == pd.Timestamp(date)]
        n = len(s)
        k = int(math.floor(0.1 * n + 0.5)) if topn is None else min(topn, max(1, n // 2))
        top = s.sort_values(["prob", "ticker"], ascending=[False, True]).ticker.tolist()[:k]
        bot = s.sort_values(["prob", "ticker"], ascending=[True, True]).ticker.tolist()[:k]

        def pin_of(j):
            return self.O.iat[e, j] * self.AC.iat[e, j] / self.C.iat[e, j]

        def usable(j):
            """진입가 유효 + (b01 규칙) 진입일~(롱·숏 이월 청산일 중 늦은 날) 사이 수정주가 이상값이 있으면 제외."""
            p = pin_of(j)
            if not (np.isfinite(p) and p > 0 and np.isfinite(self.px_at(x0, j))):
                return False
            if drop_outliers:
                xm = max(self.carry(x0, j, "L"), self.carry(x0, j, "S"))
                if np.nansum(self.OUTL.iloc[e:xm + 1, j].values) > 0:
                    return False
            return True

        def leg(tickers, side):
            vals = []
            for t in tickers:
                j = self.col.get_loc(t)
                if not usable(j):
                    continue
                if c3 and self.entry_blocked(e, j, side):
                    continue
                x = self.carry(x0, j, side) if c3 else x0
                g = self.px_at(x, j) / pin_of(j) - 1
                h = 0.0
                if c5:
                    h = self.cs_half(j, i) if spread == "cs" else self.TKP.iat[i, j] / 200
                fee = FEE if c4 else 0.0
                tax = (self.TAX.iat[x, j] / 100 if side == "L" else self.TAX.iat[e, j] / 100) if c4 else 0.0
                if not np.isfinite(tax):                                    # 매도일 행이 없는 상폐 거래: 세금 0 으로 두고 기록
                    tax = 0.0
                    self.nan_tax.append((str(pd.Timestamp(date).date()), t))
                if exact_flow:                                              # 현금흐름을 곱셈으로 그대로 계산(b01 의 덧셈 근사와 비교용)
                    if side == "L":     # 중간가 1 어치 매수: (1+h)(1+fee) 지불, 중간가 (1+g) 어치 매도: (1+g)(1-h)(1-fee-tax) 수취
                        vals.append((1 + g) * (1 - h) * (1 - fee - tax) - (1 + h) * (1 + fee))
                    else:               # 1 어치 공매도: (1-h)(1-fee-tax) 수취, (1+g) 어치 환매수: (1+g)(1+h)(1+fee) 지불
                        vals.append((1 - h) * (1 - fee - tax) - (1 + g) * (1 + h) * (1 + fee))
                    continue
                cost = h * (1 + (1 + g))                                    # 진입 명목 1, 청산 명목 1+g 에 각각 반스프레드
                cost += fee * (1 + (1 + g))                                 # 수수료는 매수·매도 체결 각각
                cost += tax * (1 + g) if side == "L" else tax * 1.0         # 세금: 롱=청산일 매도 명목 1+g, 숏=진입일 매도 명목 1
                vals.append(g - cost if side == "L" else -g - cost)         # 숏: 1 매도 후 1+g 환매수 -> -g
            return float(np.mean(vals)) if vals else 0.0, len(vals)

        lm, nl = leg(top, "L")
        sm, ns = (0.0, 0) if c1 else leg(bot, "S")
        mkt = np.nan
        if c1:                                                              # 시총가중 시장수익(신호 유니버스, 비용·이월 없음)
            num = den = 0.0
            for t in s.ticker:
                j = self.col.get_loc(t)
                m = self.MC.iat[i, j]
                if not (usable(j) and np.isfinite(m) and m > 0):
                    continue
                num += m * (self.px_at(x0, j) / pin_of(j) - 1)
                den += m
            mkt = num / den
        return dict(ret=(lm - mkt) if c1 else lm + sm, long=lm, short=sm, mkt=mkt, nL=nl, nS=ns)


def blind_week(I, q, date):
    """b01 규칙(이상값 거래 제외 등)을 전혀 모르는 채 사양만으로 쓴 총수익: 신호 유니버스 전부, 가격이 있는 종목만, 필터 없음.
    (b01_backtest.py 를 읽기 전에 작성한 초안과 같은 정의. 수정주가 이상값·페니 종목도 그대로 포함한다.)"""
    i = I.cal.get_loc(pd.Timestamp(date))
    s = q[q.date == pd.Timestamp(date)].sort_values("ticker")
    j = I.col.get_indexer(s.ticker)
    pin = I.O.values[i + 1, j] * I.AC.values[i + 1, j] / I.C.values[i + 1, j]
    r = I.AC.values[i + 5, j] / pin - 1
    k = int(math.floor(0.1 * len(s) + 0.5))
    sc = s.prob.values
    top = np.lexsort((np.arange(len(s)), -sc))[:k]
    bot = np.lexsort((np.arange(len(s)), sc))[:k]
    return np.nanmean(r[top]) - np.nanmean(r[bot]), r, top, bot, s.ticker.values


def load_all():
    P = bt.load_panel("2010-06-01", "2013-03-01", extra=("str5",))
    q = pd.read_parquet(PRED)
    pr = q.rename(columns={"prob": "score"})[["date", "ticker", "score"]]
    return P, q, pr


def part1(P, q, pr):
    """독립 재구현 대조. 반환: dict of DataFrame."""
    mk = bt.Market(P)
    tr = bt.make_trades(mk, pr)
    I = Ind(P)
    dates = list(tr.dates)
    ban_dates = [d for d, b in zip(dates, tr.ban) if b]
    rng = np.random.default_rng(20261010)
    pick = sorted([dates[i] for i in rng.choice([i for i, b in enumerate(tr.ban) if not b], 4, replace=False)]
                  + [ban_dates[int(rng.integers(len(ban_dates)))]])
    out = {}
    # (a) 5개 형성일 총수익 대조: blind / Ind / b01
    R0 = bt.run(tr, ())
    rows = []
    for d in pick + [pd.Timestamp("2012-08-10")]:
        bl = blind_week(I, q, d)[0]
        ind = I.week(d, q)["ret"]
        rows.append(dict(date=d.date(), 공매도금지주=bool(R0.loc[d, "ban"]), blind_bp=bl * 1e4, Ind_bp=ind * 1e4, b01_bp=R0.loc[d, "ret"] * 1e4,
                         blind_minus_b01_bp=(bl - R0.loc[d, "ret"]) * 1e4, Ind_minus_b01_bp=(ind - R0.loc[d, "ret"]) * 1e4))
    out["five_dates"] = pd.DataFrame(rows)
    # (b) 100주 전수: blind 대 b01, Ind 대 b01 (여러 켜짐 조합)
    bl_all = np.array([blind_week(I, q, d)[0] for d in dates])
    out["blind_all"] = pd.DataFrame([dict(항목="blind 평균(bp)", 값=bl_all.mean() * 1e4), dict(항목="b01 평균(bp)", 값=R0.ret.mean() * 1e4),
                                     dict(항목="평균 차이(bp)", 값=(bl_all - R0.ret.values).mean() * 1e4),
                                     dict(항목="주별 차이 최대 절댓값(bp)", 값=np.abs(bl_all - R0.ret.values).max() * 1e4),
                                     dict(항목="차이가 1e-9 넘는 주 수", 값=int((np.abs(bl_all - R0.ret.values) > 1e-9).sum())),
                                     dict(항목="blind 샤프", 값=bt.sharpe(bl_all, 52)), dict(항목="b01 샤프", 값=bt.sharpe(R0.ret, 52))]).set_index("항목")
    # b01 이 제외한 24개 거래와 그 영향
    s = pr.sort_values(["date", "ticker"]).reset_index(drop=True)
    drop = s[~tr.ok].copy()
    det = []
    for d, g in drop.groupby("date"):
        ls, r, top, bot, tks = blind_week(I, q, d)
        tl, bl_ = dict(zip(tks[top], r[top])), dict(zip(tks[bot], r[bot]))
        for t in g.ticker:
            leg = "롱" if t in tl else ("숏" if t in bl_ else "-")
            det.append(dict(date=d.date(), ticker=t, leg=leg, r0_pct=(tl.get(t, bl_.get(t, np.nan)) * 100),
                            blind_minus_b01_bp=(ls - R0.loc[d, "ret"]) * 1e4))
    out["dropped"] = pd.DataFrame(det)
    # (c) 비용 켠 조합: Ind(덧셈 규약) 대 b01 100주 전수 + exact_flow 변형
    cases = [("제약 없음(총수익)", {}, ()), ("c1", dict(c1=True), ("c1",)), ("c2", dict(topn=20), ("c2",)), ("c1+c2", dict(c1=True, topn=20), ("c1", "c2")),
             ("c3", dict(c3=True), ("c3",)), ("c4", dict(c4=True), ("c4",)), ("c5", dict(c5=True), ("c5",)), ("c3+c4+c5", dict(c3=True, c4=True, c5=True), ("c3", "c4", "c5")),
             ("c1+c3+c4+c5", dict(c1=True, c3=True, c4=True, c5=True), ("c1", "c3", "c4", "c5")),
             ("c5(틱/2)", dict(c5=True, spread="tick"), ("c5",))]
    rows, runs = [], {}
    for nm, kw, cons in cases:
        Rb = bt.run(tr, cons, spread=("tick" if kw.get("spread") == "tick" else "cs"))["ret"]
        ind = np.array([I.week(d, q, **kw)["ret"] for d in dates])
        runs[nm] = (Rb, ind)
        ex = np.array([I.week(d, q, exact_flow=True, **kw)["ret"] for d in dates]) if nm in ("c3+c4+c5", "c1+c3+c4+c5") else None
        rows.append(dict(조합=nm, b01평균_bp=Rb.mean() * 1e4, Ind평균_bp=ind.mean() * 1e4, 주별최대차_bp=np.abs(ind - Rb.values).max() * 1e4,
                         exact현금흐름_평균_bp=(ex.mean() * 1e4 if ex is not None else np.nan),
                         exact_대_b01_주별최대차_bp=(np.abs(ex - Rb.values).max() * 1e4 if ex is not None else np.nan)))
    out["cost_all"] = pd.DataFrame(rows)
    pos_ = {d: i for i, d in enumerate(dates)}
    rr = []
    for d in pick + [pd.Timestamp("2011-01-21")]:
        for nm in runs:
            Rb, ind = runs[nm]
            rr.append(dict(date=d.date(), 조합=nm, Ind_bp=ind[pos_[d]] * 1e4, b01_bp=Rb.iloc[pos_[d]] * 1e4, 차이_bp=(ind[pos_[d]] - Rb.iloc[pos_[d]]) * 1e4))
    out["cost_dates"] = pd.DataFrame(rr)
    # (d) 손계산 표: 한 형성일의 종목별 현금흐름(c3+c4+c5, 롱숏) 상세
    hd = pd.Timestamp("2011-01-21")
    out["hand_date"] = hd
    out["hand"] = hand_table(I, q, hd)
    return out, mk, tr, I


def hand_table(I, q, date, nshow=3):
    """한 형성일의 롱·숏 각 상위 nshow 종목을 종목별로 펼쳐서 c3+c4+c5 현금흐름을 직접 계산한다(합계는 전체 종목)."""
    i = I.cal.get_loc(date)
    e, x0 = i + 1, i + 5
    s = q[q.date == date]
    k = int(math.floor(0.1 * len(s) + 0.5))
    top = s.sort_values(["prob", "ticker"], ascending=[False, True]).ticker.tolist()[:k]
    bot = s.sort_values(["prob", "ticker"], ascending=[True, True]).ticker.tolist()[:k]
    rows = []
    for side, tks in (("L", top), ("S", bot)):
        for t in tks[:nshow]:
            j = I.col.get_loc(t)
            pin = I.O.iat[e, j] * I.AC.iat[e, j] / I.C.iat[e, j]
            x = I.carry(x0, j, side)
            g = I.px_at(x, j) / pin - 1
            h = I.cs_half(j, i)
            tax = I.TAX.iat[x, j] / 100 if side == "L" else I.TAX.iat[e, j] / 100
            cost = h * (2 + g) + FEE * (2 + g) + (tax * (1 + g) if side == "L" else tax)
            rows.append(dict(leg=side, ticker=t, 시가e=I.O.iat[e, j], 종가e=I.C.iat[e, j], 수정종가e=I.AC.iat[e, j], 진입가_수정=pin,
                             청산가_수정=I.px_at(x, j), 청산일=I.cal[x].date(), g_pct=g * 100, 반스프레드_pct=h * 100, 세율_pct=tax * 100,
                             비용_bp=cost * 1e4, 순수익_bp=((g - cost) if side == "L" else (-g - cost)) * 1e4))
    return pd.DataFrame(rows)


# =====================================================================================================
# 2. 룩어헤드/정합
# =====================================================================================================
def part2(P, q, pr, mk, tr, I):
    out = {}
    dates = list(tr.dates)
    rng = np.random.default_rng(7)
    pick = [dates[i] for i in sorted(rng.choice(len(dates), 3, replace=False))]
    # (a) CS 창 정렬: b01 의 HS[형성일] 이 '형성일까지 20거래일 창'(Ind) 과 같은지, 창을 하루 앞/뒤로 밀면 달라지는지(검정력)
    rows = []
    for d in pick:
        i = I.cal.get_loc(d)
        di = mk.cal.get_loc(d)
        tks = q[q.date == d].ticker.values
        tks = tks[np.random.default_rng(1).choice(len(tks), 150, replace=False)]
        base = np.array([mk.HS[di, mk.tk.get_loc(t)] for t in tks])
        for nm, off in (("창 끝=형성일(b01 주장)", 0), ("창 끝=형성일+1(미래 1일 포함)", 1), ("창 끝=형성일-1(1일 지연)", -1)):
            alt = np.array([I.cs_half(I.col.get_loc(t), i + off) for t in tks])
            m = np.isfinite(base)
            rows.append(dict(date=d.date(), 변형=nm, 최대차_bp=np.nanmax(np.abs(alt[m] - base[m])) * 1e4, 일치종목비율=float(np.mean(np.isclose(alt[m], base[m], atol=1e-12, rtol=0)))))
    out["cs_align"] = pd.DataFrame(rows)
    # (b) 미래 교란: 형성일 이후 행의 고가·저가·거래량·정지를 뒤섞어도 그 형성일의 hs 가 그대로인가. 대조: 형성일 이전 10일을 뒤섞으면 바뀌는가
    rows = []
    s_sorted = pr.sort_values(["date", "ticker"]).reset_index(drop=True)
    for d in pick:
        sel = (s_sorted.date == d).values
        for nm, lo, hi in (("형성일 이후 전부 교란", d + pd.Timedelta(days=1), pd.Timestamp("2100-01-01")),
                           ("형성일 직전 10거래일 교란(대조)", I.cal[I.cal.get_loc(d) - 9], d)):
            Pm = P.copy()
            m = (Pm.date >= lo) & (Pm.date <= hi)
            rr = np.random.default_rng(3)
            for c in ("high", "low", "volume"):
                Pm.loc[m, c] = rr.permutation(Pm.loc[m, c].values)
            Pm.loc[m, "is_halted"] = rr.permutation(Pm.loc[m, "is_halted"].values)
            mk2 = bt.Market(Pm)
            tr2 = bt.make_trades(mk2, pr)
            a, b = tr.hs[sel], tr2.hs[sel]
            rows.append(dict(date=d.date(), 교란=nm, hs_변한_종목수=int((~np.isclose(a, b, atol=1e-15, rtol=0)).sum()), 종목수=int(sel.sum()), 최대차_bp=float(np.abs(a - b).max() * 1e4)))
    out["perturb"] = pd.DataFrame(rows)
    # (c) 신호·진입 시점: 라벨 ret = 형성일 종가 -> 5거래일 뒤 종가 인지, 진입일 = 형성일 다음 거래일 인지
    A_ = pd.read_parquet(bt.PANEL, columns=["date", "ticker", "adj_close"]).pivot(index="date", columns="ticker", values="adj_close")
    cal = A_.index
    di = cal.get_indexer(q.date)
    ti = A_.columns.get_indexer(q.ticker)
    lab = A_.values[di + 5, ti] / A_.values[di, ti] - 1
    mm = np.isfinite(lab)
    out["label"] = pd.DataFrame([dict(항목="pred.ret 대 패널 AC[t+5]/AC[t]-1 상관", 값=float(np.corrcoef(q.ret[mm], lab[mm])[0, 1])),
                                 dict(항목="절대차 99% (bp)", 값=float(np.percentile(np.abs(q.ret[mm] - lab[mm]), 99) * 1e4)),
                                 dict(항목="절대차 최대 (bp)", 값=float(np.abs(q.ret[mm] - lab[mm]).max() * 1e4)),
                                 dict(항목="패널에서 계산 안 되는 행", 값=int((~mm).sum()))]).set_index("항목")
    nxt = mk.cal[mk.cal.get_indexer(tr.dates) + 1]
    out["entry_ok"] = bool((tr.entry_dates == nxt).all() and (tr.entry_dates > tr.dates).all())
    return out


# =====================================================================================================
# 3. 감사 버그(B1, B2, B7) 시나리오: b01 엔진을 시나리오별로 바꾼 Market 으로 다시 돌린다
# =====================================================================================================
OLD_BANDS = [(0, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 500000, 500), (500000, 1e18, 1000)]
MID_BANDS = [(0, 1000, 1), (1000, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 500000, 500), (500000, 1e18, 1000)]
NEW_BANDS = [(0, 2000, 1), (2000, 5000, 5), (5000, 20000, 10), (20000, 50000, 50), (50000, 200000, 100), (200000, 500000, 500), (500000, 1e18, 1000)]


def tick_vec(price, dates):
    """q01_panel_qa.py 의 tick_vec 를 그대로 옮김(호가단위표는 q01 이 독립 입력한 값)."""
    price = np.asarray(price, float)
    dates = pd.DatetimeIndex(dates)
    out = np.full(len(price), np.nan)
    for lo, hi, bands in [("1900-01-01", "2010-10-03", OLD_BANDS), ("2010-10-04", "2023-01-24", MID_BANDS), ("2023-01-25", "2100-01-01", NEW_BANDS)]:
        m = (dates >= lo) & (dates <= hi)
        edges = np.array([b[0] for b in bands], float)
        ticks = np.array([b[2] for b in bands], float)
        idx = np.searchsorted(edges, price[m], side="right") - 1
        out[m] = ticks[np.clip(idx, 0, len(ticks) - 1)]
    return out


def exact_limits(prev, dates, L):
    """q01_panel_qa.py 의 exact_limits: KRX 산식(기준가 x 한도율을 호가단위로 절사 후 가감, 결과가 속한 가격대 호가단위로 절사/올림)."""
    prev = np.asarray(prev, float)
    t0 = tick_vec(prev, dates)
    step = np.floor(prev * L / t0) * t0
    up = prev + step
    up = np.floor(up / tick_vec(up, dates)) * tick_vec(up, dates)
    dn = prev - step
    td = tick_vec(np.maximum(dn, 1), dates)
    dn = np.ceil(dn / td) * td
    return up, dn


def flag_mat(mk, F, col):
    M = np.zeros((len(mk.cal), len(mk.tk)), bool)
    f = F[F[col] == 1]
    di, ti = mk.cal.get_indexer(f.date), mk.tk.get_indexer(f.ticker)
    ok = (di >= 0) & (ti >= 0)
    M[di[ok], ti[ok]] = True
    return M


def apply_lim(mk, LU, LD, OU, OD):
    mk.LU, mk.LD, mk.OU, mk.OD = LU, LD, OU, OD
    mk.BLK_SELL, mk.BLK_BUY = mk.NOTRADE | LD, mk.NOTRADE | LU


def scenario(P, pr, F, outlier="asis", resume_excl=False, resume_lim_off=False, lim="asis", keep_weeks=None, penny_excl=False, adj_excl=False):
    """outlier: asis(b01) | penny_only(극저가 아티팩트만 계속 제외, 정리매매·상폐 인접 이상값 거래는 복원) | none(이상값 거래 전부 복원)
    resume_excl: 보유 구간(진입일~청산일)에 재개일(f_resume_day)이 있는 거래를 제외 / resume_lim_off: 재개일의 상·하한가 판정을 끔
    lim: asis | exact(호가단위 반영 정확 한도, q01 산식) | tol1(한도-1%p 보수 차단)"""
    mk = bt.Market(P)
    nd, nt = mk.O.shape
    raw_out = np.diff(mk.OC, axis=0) > 0
    Cmat = mk.A / mk.F
    prevA = np.vstack([np.full((1, nt), np.nan), mk.A[:-1]])
    penny = (mk.A <= 100) | (prevA <= 100) | (Cmat <= 200)
    res, liq = flag_mat(mk, F, "f_resume_day"), flag_mat(mk, F, "f_liquidation_zone")
    out_mask = {"asis": raw_out, "penny_only": raw_out & penny, "none": np.zeros_like(raw_out)}[outlier]
    if resume_excl:
        out_mask = out_mask | res
    if penny_excl:                                   # 극저가(수정주가<=100 또는 종가<=200원) 종목-일이 보유 구간에 있으면 제외 (q01 권고: 008080, 000790 류)
        out_mask = out_mask | penny
    if adj_excl:                                     # 보유 구간에 수정계수가 1% 넘게 바뀐 날(액면분할·권리락 등)이 있으면 제외
        Fq = np.vstack([np.full((1, nt), np.nan), mk.F[:-1]])
        with np.errstate(all="ignore"):
            out_mask = out_mask | (np.abs(mk.F / Fq - 1) > 0.01)
    mk.OC = np.vstack([np.zeros((1, nt)), np.cumsum(out_mask, axis=0)])
    prevC = np.vstack([np.full((1, nt), np.nan), Cmat[:-1]])
    Lm = np.where(mk.cal < pd.Timestamp("2015-06-15"), 0.15, 0.30)[:, None] * np.ones((1, nt))
    if lim != "asis":
        previdle = np.vstack([np.ones((1, nt), bool), mk.NOTRADE[:-1]])
        Fp = np.vstack([np.full((1, nt), np.nan), mk.F[:-1]])
        with np.errstate(all="ignore"):
            elig = ~previdle & np.isfinite(prevC) & (np.abs(mk.F / Fp - 1) < 0.01) & (mk.O > 0) & ~liq & ~mk.NOTRADE
            if lim == "exact":
                dts = np.repeat(mk.cal.values[:, None], nt, axis=1).ravel()
                up, dn = (a.reshape(nd, nt) for a in exact_limits(prevC.ravel(), dts, Lm.ravel()))
                LU, LD, OU, OD = (Cmat >= up) & elig, (Cmat <= dn) & elig, (mk.O >= up) & elig, (mk.O <= dn) & elig
            else:
                rc, ro = Cmat / prevC - 1, mk.O / prevC - 1
                LU, LD = (rc >= Lm - 0.01) & elig & (np.abs(rc) <= Lm + 0.05), (rc <= -Lm + 0.01) & elig & (np.abs(rc) <= Lm + 0.05)
                OU, OD = (ro >= Lm - 0.01) & elig & (np.abs(ro) <= Lm + 0.05), (ro <= -Lm + 0.01) & elig & (np.abs(ro) <= Lm + 0.05)
        apply_lim(mk, LU, LD, OU, OD)
    if resume_lim_off:
        apply_lim(mk, mk.LU & ~res, mk.LD & ~res, mk.OU & ~res, mk.OD & ~res)
    tr = bt.make_trades(mk, pr)
    R = bt.all_subsets(tr)
    gaps = np.diff(mk.cal.get_indexer(tr.dates))
    keep = np.ones(tr.G, bool) if keep_weeks is None else np.r_[~(gaps < 5), True]
    return tr, R[keep], mk


FULL, NO_C1 = bt.N_SUB - 1, bt.N_SUB - 2


def summarize(name, tr, R):
    vm = R.mean().to_numpy()
    phi = bt.shapley(vm)
    d = dict(시나리오=name, 주수=len(R), 거래_ok_비율=tr.ok.mean(), v공집합_bp=vm[0] * 1e4, v공집합_샤프=bt.sharpe(R[0], 52), v_c2c5_bp=vm[NO_C1] * 1e4, v전부_bp=vm[FULL] * 1e4,
             v전부_샤프=bt.sharpe(R[FULL], 52))
    d.update({f"φ{c}": phi[i] * 1e4 for i, c in enumerate(bt.CONS)})
    d.update(롱진입불가=float(tr.blk_buy[tr.ok].mean()), 숏진입불가=float(tr.blk_sell_short[tr.ok].mean()),
             롱청산이월=float(tr.carried_l[tr.ok].mean()), 숏청산이월=float(tr.carried_s[tr.ok].mean()))
    return d


def part3(P, pr, F):
    """반환: 시나리오 표, 플래그 노출 표, c3 판정 혼동 표."""
    specs = [("S0 b01 기본(재현)", {}),
             ("S1 B2: 정리매매·상폐 인접 이상값 거래 복원(극저가 아티팩트만 제외)", dict(outlier="penny_only")),
             ("S1b B2: 이상값 거래 전부 복원(극저가 포함)", dict(outlier="none")),
             ("S2 B1: 보유 구간에 재개일 있는 거래 제외", dict(resume_excl=True)),
             ("S2b B1: 재개일의 상·하한가 판정 끔", dict(resume_lim_off=True)),
             ("S3 B7: 상·하한가를 호가단위 반영 정확 한도로 교체", dict(lim="exact")),
             ("S3b B7: 한도-1%p 보수 차단", dict(lim="tol1")),
             ("S4 감사 보정 전부(S1+S2+S3)", dict(outlier="penny_only", resume_excl=True, lim="exact")),
             ("S5 겹치는 17주 제외", dict(keep_weeks=True)),
             ("S6 S4 + 겹침 제외", dict(outlier="penny_only", resume_excl=True, lim="exact", keep_weeks=True)),
             ("S7 극저가(수정주가<=100·종가<=200원) 종목-일 포함 거래 제외", dict(penny_excl=True)),
             ("S8 보유 구간 수정계수 변경(>1%) 거래 제외", dict(adj_excl=True)),
             ("S9 가장 보수적 정제(S4+S7+S8)", dict(outlier="penny_only", resume_excl=True, lim="exact", penny_excl=True, adj_excl=True))]
    rows, trs = [], {}
    for nm, kw in specs:
        tr, R, mk = scenario(P, pr, F, **kw)
        rows.append(summarize(nm, tr, R))
        trs[nm.split()[0]] = (tr, mk)
    scen = pd.DataFrame(rows)
    # 플래그 노출: 기본 거래표 기준 (진입일~청산일 x0 사이 행에 플래그가 있는 거래 수)
    tr0, mk0 = trs["S0"]
    s = pr.sort_values(["date", "ticker"]).reset_index(drop=True)
    di, ti = mk0.cal.get_indexer(s.date), mk0.tk.get_indexer(s.ticker)
    ex = []
    for col in ("f_resume_day", "f_liquidation_zone", "f_prelist_otc", "f_split60"):
        M = flag_mat(mk0, F, col)
        C = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M, axis=0)])
        ex.append(dict(플래그=col, 보유구간_거래수=int((C[di + 6, ti] - C[di + 1, ti] > 0).sum()), 진입일_거래수=int(M[di + 1, ti].sum()), 청산일_거래수=int(M[di + 5, ti].sum())))
    ex.append(dict(플래그="ret_outlier 로 b01 이 제외한 거래", 보유구간_거래수=int((~tr0.ok).sum()), 진입일_거래수=np.nan, 청산일_거래수=np.nan))
    expo = pd.DataFrame(ex)
    # c3 판정: 기본 대 정확 한도 (ok 거래 기준 건수)
    tr3, _ = trs["S3"]
    ok = tr0.ok
    conf = []
    for nm, a, b in (("롱 진입불가", tr0.blk_buy, tr3.blk_buy), ("숏 진입불가", tr0.blk_sell_short, tr3.blk_sell_short),
                     ("롱 청산이월", tr0.carried_l, tr3.carried_l), ("숏 청산이월", tr0.carried_s, tr3.carried_s)):
        a, b = a[ok], b[ok]
        conf.append(dict(항목=nm, 기본_건수=int(a.sum()), 정확_건수=int(b.sum()), 둘다=int((a & b).sum()), 기본만=int((a & ~b).sum()), 정확만=int((~a & b).sum())))
    return scen, expo, pd.DataFrame(conf)


def survivorship(P, q, I):
    """신호 유니버스 밖 생존편향(B2 의 유니버스 수준): 형성일에 정상 거래 중이던 패널 종목 중 신호표에 없고,
    패널 행이 형성일 이후 5거래일 안에 끝나는(상폐) 종목. 이 종목들은 라벨이 없어 CNN 표본에서 빠졌다(images.py 주석)."""
    cal, col = I.cal, I.col
    end_last = pd.Series({t: cal[np.flatnonzero(np.isfinite(I.C[t].values))[-1]] for t in col})
    rows = []
    for d in sorted(q.date.unique()):
        d = pd.Timestamp(d)
        i = cal.get_loc(d)
        act = [t for t in col if np.isfinite(I.C.at[d, t]) and I.V.at[d, t] > 0 and I.HALT.at[d, t] == 0]
        have = set(q.loc[q.date == d, "ticker"])
        for t in act:
            if t in have:
                continue
            j = col.get_loc(t)
            li = cal.get_loc(end_last[t])
            if li < i + 5 and end_last[t] < pd.Timestamp("2013-02-01"):       # 5거래일 안에 행이 끝난다 = 라벨 없음
                pin = I.O.iat[i + 1, j] * I.AC.iat[i + 1, j] / I.C.iat[i + 1, j] if li >= i + 1 else np.nan
                r = I.AC.iat[li, j] / pin - 1 if np.isfinite(pin) else np.nan
                rows.append(dict(date=d.date(), ticker=t, 마지막행=end_last[t].date(), 마지막행_거래일수=li - i, r_진입시가_마지막가=r,
                                 r_형성일종가_마지막가=I.AC.iat[li, j] / I.AC.iat[i, j] - 1))
    return pd.DataFrame(rows)


# =====================================================================================================
# 4. 해석 점검, 주별 분해
# =====================================================================================================
def retention_n(tr, topn=20):
    """상위 topn 롱 레그의 연속 형성일 재편입 비율(net_turnover 절감액의 독립 추정용)."""
    k = np.minimum(topn, np.maximum(1, tr.n_g // 2))[tr.gid]
    m = (tr.rk_d <= k) & tr.ok
    key = tr.gid[m].astype(np.int64) * tr.NT + tr.tick_idx[m]
    prev = (tr.gid[m] - 1).astype(np.int64) * tr.NT + tr.tick_idx[m]
    return float(np.isin(prev, key)[tr.gid[m] > 0].mean())


def stats_line(x, ppy=52):
    x = np.asarray(x, float)
    return dict(주수=len(x), 평균_bp=x.mean() * 1e4, 샤프=bt.sharpe(x, ppy), t=x.mean() / x.std(ddof=1) * math.sqrt(len(x)), 적중률=float((x > 0).mean()))


def part4(P, pr, q, mk, tr):
    out = {}
    # (a) net_turnover / tick / raw 재현과 방향, 절감액의 독립 추정
    base = bt.all_subsets(tr)
    rows = []
    for nm, kw in (("기본", {}), ("매매 상쇄(net_turnover)", dict(net_turnover=True)), ("스프레드=틱/2", dict(spread="tick")),
                   ("시장 차감 없는 롱온리", dict(c1_mode="raw")), ("대차 연 4%", dict(borrow=0.04))):
        R = bt.all_subsets(tr, **kw) if kw else base
        vm = R.mean().to_numpy()
        phi = bt.shapley(vm)
        rows.append(dict(변형=nm, v공집합_bp=vm[0] * 1e4, v_c2c5_bp=vm[NO_C1] * 1e4, v전부_bp=vm[FULL] * 1e4, **{f"φ{c}": phi[i] * 1e4 for i, c in enumerate(bt.CONS)},
                         c3c4c5_롱숏10분위_bp=vm[bt.mask_of(("c3", "c4", "c5"))] * 1e4,
                         c4만_롱숏10분위_bp=vm[bt.mask_of(("c4",))] * 1e4))
    import copy
    tr_ew = copy.copy(tr)                                   # 시장 기준을 동일가중(같은 신호 유니버스, 진입 시가->청산 종가)으로 바꾼 변형
    w_ok = tr.ok.astype(float)
    tr_ew.mkt = np.bincount(tr.gid, w_ok * np.where(tr.ok, tr.r0, 0.0), tr.G) / np.maximum(np.bincount(tr.gid, w_ok, tr.G), 1e-300)
    R = bt.all_subsets(tr_ew)
    vm = R.mean().to_numpy()
    phi = bt.shapley(vm)
    rows.append(dict(변형="시장 기준=동일가중", v공집합_bp=vm[0] * 1e4, v_c2c5_bp=vm[NO_C1] * 1e4, v전부_bp=vm[FULL] * 1e4, **{f"φ{c}": phi[i] * 1e4 for i, c in enumerate(bt.CONS)},
                     c3c4c5_롱숏10분위_bp=vm[bt.mask_of(("c3", "c4", "c5"))] * 1e4, c4만_롱숏10분위_bp=vm[bt.mask_of(("c4",))] * 1e4))
    out["variants"] = pd.DataFrame(rows)
    r20 = retention_n(tr, 20)
    rl = bt.run(tr, ("c1", "c2", "c3", "c4", "c5"))
    hs20 = float(np.nanmean(tr.hs[(tr.rk_d <= 20) & tr.ok]))
    tax = float(np.nanmean(tr.tax_xl[(tr.rk_d <= 20) & tr.ok]))
    est = r20 * (2 * bt.FEE + tax + 2 * hs20) * 1e4
    out["netturn_check"] = pd.DataFrame([dict(항목="상위20 롱 재편입 비율", 값=r20), dict(항목="평균 반스프레드 hs (상위20)", 값=hs20), dict(항목="평균 매도세율", 값=tax),
                                         dict(항목="추정 절감 = 재편입 x (2*수수료+세금+2*hs) (bp)", 값=est),
                                         dict(항목="실제 v(전부) 증가 (bp)", 값=(out["variants"].loc[1, "v전부_bp"] - out["variants"].loc[0, "v전부_bp"]))]).set_index("항목")
    # (b) 주별 분해
    R0 = bt.run(tr, ())
    W = pd.DataFrame({"LS": R0.ret, "long": R0.long, "short": R0.short, "mkt": R0.mkt, "ban": R0.ban, "entry": R0.entry})
    W["long_ex"], W["short_ex"] = W.long - W.mkt, W.short + W.mkt
    out["weekly"] = W
    gaps = np.diff(mk.cal.get_indexer(tr.dates))
    e_ = np.r_[gaps < 5, False]                            # 다음 배치가 5거래일 안에 시작해 보유가 겹치는 배치(17개)
    a_, b_ = W.LS[e_], W.LS[~e_]
    out["overlap"] = pd.DataFrame([dict(항목="겹치는 쌍 수 (간격 4/3/2일)", 값=f"{int((gaps == 4).sum())}/{int((gaps == 3).sum())}/{int((gaps == 2).sum())}"),
                                   dict(항목="겹치는 17개 배치 평균 (bp)", 값=round(a_.mean() * 1e4, 1)), dict(항목="나머지 83개 평균 (bp)", 값=round(b_.mean() * 1e4, 1)),
                                   dict(항목="평균 차이 t (Welch)", 값=round((a_.mean() - b_.mean()) / math.sqrt(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_)), 2)),
                                   dict(항목="겹치는 인접 쌍 수익 상관", 값=round(float(np.corrcoef(W.LS.values[:-1][gaps < 5], W.LS.values[1:][gaps < 5])[0, 1]), 3)),
                                   dict(항목="전체 1차 자기상관", 값=round(float(W.LS.autocorr(1)), 3))]).set_index("항목")
    ban = W.ban.values
    pre = np.asarray(W.entry < pd.Timestamp("2011-08-10")) & ~ban          # 진입일 기준 구간 (공매도 금지 2011-08-10 ~ 11-09)
    post = np.asarray(W.entry > pd.Timestamp("2011-11-09")) & ~ban
    out["quarters"] = W.groupby(W.index.to_period("Q")).apply(lambda g: pd.Series(dict(주수=len(g), 평균_bp=g.LS.mean() * 1e4, 샤프=bt.sharpe(g.LS, 52) if len(g) > 2 else np.nan,
                                                                                    롱초과_bp=g.long_ex.mean() * 1e4, 숏초과_bp=g.short_ex.mean() * 1e4, 시장_bp=g.mkt.mean() * 1e4)), include_groups=False)
    leg = pd.DataFrame({"전체": [W.LS.mean(), W.long.mean(), W.short.mean(), W.mkt.mean(), W.long_ex.mean(), W.short_ex.mean()]},
                       index=["LS 총수익", "롱 레그 총수익", "숏 레그 손익(=-숏 종목 수익)", "시총가중 시장", "롱 초과(롱-시장)", "숏 초과(숏손익+시장)"]) * 1e4
    for nm, m in (("금지 전(2011-01~07)", pre), ("금지 주(2011-08~11)", ban), ("금지 후(2011-11~2012-11)", post)):
        leg[nm] = [W.LS[m].mean(), W.long[m].mean(), W.short[m].mean(), W.mkt[m].mean(), W.long_ex[m].mean(), W.short_ex[m].mean()]
        leg.loc[:, nm] *= 1e4
    leg.loc["주 수"] = [len(W), int(pre.sum()), int(ban.sum()), int(post.sum())]
    out["legs"] = leg
    per = pd.DataFrame({nm: stats_line(W.LS[m]) for nm, m in (("전체", np.ones(len(W), bool)), ("금지 전", pre), ("금지 주", ban), ("금지 후", post))}).T
    out["periods"] = per
    ls = W.LS.sort_values(ascending=False)
    tot = ls.sum()
    dep = []
    for k in (1, 3, 5, 10):
        rest = W.LS.drop(ls.index[:k])
        dep.append(dict(제외="상위 %d주 제외" % k, 제외분_합의_비중=float(ls.iloc[:k].sum() / tot), **stats_line(rest)))
    rest = W.LS.drop(ls.index[-5:])
    dep.append(dict(제외="하위 5주 제외", 제외분_합의_비중=float(ls.iloc[-5:].sum() / tot), **stats_line(rest)))
    crash = (W.index >= "2011-08-01") & (W.index <= "2011-10-31")
    dep.append(dict(제외="2011-08~10 형성 주 제외", 제외분_합의_비중=float(W.LS[crash].sum() / tot), **stats_line(W.LS[~crash])))
    out["dependence"] = pd.DataFrame(dep)
    out["top_weeks"] = W.loc[ls.index[:5], ["LS", "long", "short", "mkt"]].mul(1e4).round(1).assign(분포=lambda d: "상위")
    out["moments"] = pd.DataFrame([dict(항목="주간 왜도", 값=float(W.LS.skew())), dict(항목="주간 초과첨도", 값=float(W.LS.kurt())), dict(항목="1차 자기상관", 값=float(W.LS.autocorr(1))),
                                   dict(항목="최악 주(bp)", 값=float(W.LS.min() * 1e4)), dict(항목="최고 주(bp)", 값=float(W.LS.max() * 1e4)),
                                   dict(항목="주간 표준편차(bp)", 값=float(W.LS.std(ddof=1) * 1e4))]).set_index("항목")
    # (c) 신호 성격: 형성일 횡단면 순위상관 (prob 대 str5, log시총, log종가), 실현수익과의 IC
    X = P[["date", "ticker", "str5", "mktcap", "close"]].merge(pr.rename(columns={"score": "prob"}), on=["date", "ticker"])
    X["lmc"], X["lpx"] = np.log(X.mktcap), np.log(X.close)
    ic = {c: X.groupby("date").apply(lambda g, c=c: g["prob"].corr(g[c], method="spearman"), include_groups=False) for c in ("str5", "lmc", "lpx")}
    s = pr.sort_values(["date", "ticker"]).reset_index(drop=True)
    r0 = pd.Series(tr.r0, index=pd.MultiIndex.from_frame(s[["date", "ticker"]]))
    Y = s.assign(r=tr.r0, ok=tr.ok)
    Y = Y[Y.ok]
    ic["실현 5일 수익(IC)"] = Y.groupby("date").apply(lambda g: g["score"].corr(g["r"], method="spearman"), include_groups=False)
    out["char_ic"] = pd.DataFrame({k: dict(주평균=v.mean(), t=v.mean() / v.std(ddof=1) * math.sqrt(len(v)), 양수비율=float((v > 0).mean())) for k, v in ic.items()}).T
    # 레그별 특성 (롱=상위 10분위, 숏=하위 10분위; 형성일 시점 값)
    k = np.maximum(1, np.floor(0.1 * tr.n_g + 0.5).astype(int))[tr.gid]
    Xs = s.merge(P[["date", "ticker", "mktcap", "close", "tick_pct", "str5"]], on=["date", "ticker"], how="left")
    Xs["legL"], Xs["legS"] = (tr.rk_d <= k) & tr.ok, (tr.rk_a <= k) & tr.ok
    Xs["mc_pct"] = Xs.groupby("date").mktcap.rank(pct=True)
    ch = pd.DataFrame({nm: {"시총 분위(0~1) 평균": Xs.loc[m, "mc_pct"].mean(), "종가 중앙값(원)": Xs.loc[m, "close"].median(),
                            "tick/종가 중앙값(%)": Xs.loc[m, "tick_pct"].median(), "직전 5일 수익 평균(%)": Xs.loc[m, "str5"].mean() * 100}
                       for nm, m in (("전체 유니버스", np.ones(len(Xs), bool)), ("롱 레그", Xs.legL.values), ("숏 레그", Xs.legS.values))})
    out["char_legs"] = ch.T
    # 단순 5일 반전(-str5)을 같은 유니버스·같은 체결로 돌린 대조, 그리고 CNN 주간 LS 와의 상관
    rv = s.merge(P[["date", "ticker", "str5"]], on=["date", "ticker"], how="left").dropna(subset=["str5"])
    rv = rv.assign(score=-rv.str5)[["date", "ticker", "score"]]
    trr = bt.make_trades(mk, rv)
    Rr = bt.run(trr, ())
    both = pd.concat([W.LS.rename("cnn"), Rr.ret.rename("rev")], axis=1).dropna()
    rp = {nm: stats_line(Rr.ret[m]) for nm, m in (("금지 전", pre), ("금지 주", ban), ("금지 후", post))}
    out["reversal_periods"] = pd.DataFrame(rp).T
    out["reversal"] = pd.DataFrame([dict(신호="CNN d_lr1e-4", 총수익_bp=W.LS.mean() * 1e4, 샤프=bt.sharpe(W.LS, 52), 주수=len(W)),
                                    dict(신호="-str5 (직전 5일 수익 반전), 같은 유니버스", 총수익_bp=Rr.ret.mean() * 1e4, 샤프=bt.sharpe(Rr.ret, 52), 주수=len(Rr)),
                                    dict(신호="두 주간 LS 수익의 상관", 총수익_bp=float(both.cnn.corr(both.rev)), 샤프=np.nan, 주수=len(both))]).set_index("신호")
    # 유니버스 제한 재실행: 시총 상·하위 절반, 종가 5000원 이상/미만 (분위는 형성일별 신호 유니버스 안)
    Z = s.merge(P[["date", "ticker", "mktcap", "close"]], on=["date", "ticker"], how="left")
    Z["big"] = Z.groupby("date").mktcap.rank(pct=True) > 0.5
    rows = []
    for nm, m in (("전체", np.ones(len(Z), bool)), ("시총 상위 절반", Z.big.values), ("시총 하위 절반", ~Z.big.values),
                  ("종가 5000원 이상", (Z.close >= 5000).values), ("종가 5000원 미만", (Z.close < 5000).values)):
        sub = s[m & Z.mktcap.notna().values]
        trx = bt.make_trades(mk, sub)
        g = bt.run(trx, ())
        n_ = bt.run(trx, ("c3", "c4", "c5"))
        nt = bt.run(trx, ("c3", "c4", "c5"), spread="tick")
        rows.append(dict(유니버스=nm, 종목수_주평균=float(trx.n_g.mean()), 총수익_bp=g.ret.mean() * 1e4, 총수익_샤프=bt.sharpe(g.ret, 52), 순수익_CS_bp=n_.ret.mean() * 1e4,
                         순수익_틱_bp=nt.ret.mean() * 1e4, 평균_CS반스프레드_pct=float(np.nanmean(trx.hs[trx.ok])) * 100))
    out["subuniverse"] = pd.DataFrame(rows)
    return out


# =====================================================================================================
# 5. 보고서 숫자 대 CSV 전수 대조, 샤플리·부트스트랩 재현, 파일럿 재현
# =====================================================================================================
MEMO = f"{ROOT}/results/b01_constraints_memo.md"
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def md_tables(text):
    """마크다운 표를 (헤더 셀 목록, 행 목록[셀 문자열 목록]) 으로 쪼갠다."""
    out, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if cur is None:
                cur = [cells, []]
            elif all(set(c) <= set("-: ") for c in cells):
                pass
            else:
                cur[1].append(cells)
        else:
            if cur is not None:
                out.append(tuple(cur))
                cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def find_table(tables, *needles):
    for h, rows in tables:
        if all(any(n in c for c in h) for n in needles):
            return h, rows
    raise KeyError(needles)


def cell_nums(cell):
    return [(float(m), len(m.split(".")[1]) if "." in m else 0) for m in NUM.findall(cell)]


def cmp_cells(tag, cell, exp, bad, tot):
    """셀 문자열 속 숫자들을 기대값 목록과 비교(표기 자릿수의 반올림 오차 이내면 일치). exp 원소가 None 이면 셀이 비어 있어야 한다."""
    got = cell_nums(cell)
    tot[0] += 1
    if all(e is None for e in exp):
        if got:
            bad.append((tag, cell, "기대: 빈 칸"))
        return
    if len(got) != len(exp):
        bad.append((tag, cell, f"숫자 개수 {len(got)} != {len(exp)}"))
        return
    for (g, d), e in zip(got, exp):
        if e is None or not np.isfinite(e) or abs(g - e) > 0.5 * 10 ** -d + 1e-9:
            bad.append((tag, cell, f"기대 {e}"))
            return


def check_memo_vs_csv(memo=MEMO):
    """보고서의 표를 CSV 에서 다시 만든 값과 칸 단위로 대조한다. 반환: (검사한 칸 수, 불일치 목록)."""
    text = open(memo, encoding="utf-8").read()
    T = md_tables(text)
    rd = lambda n: pd.read_csv(f"{TAB}/{n}.csv", encoding="utf-8-sig")        # noqa: E731
    pf, v32, sh, sens, dg_, rep, sim = rd("portfolios"), rd("v32"), rd("shapley"), rd("sensitivity"), rd("diagnostics"), rd("A_replication_pilot"), rd("cs_bias_sim")
    bad, tot = [], [0]
    A, B6, B5 = "A_CNN파일럿", "B_6지표순위평균", "B_5일반전"

    # 3절 재현 표
    h, rows = find_table(T, "주간 평균(bp)", "연율 샤프")
    for r, (_, x) in zip(rows, rep.iterrows()):
        cmp_cells("재현표", r[1], [x.weekly_mean * 1e4], bad, tot)
        cmp_cells("재현표", r[2], [x.sharpe], bad, tot)

    # 포트폴리오 표 (4.1: 열 [포트폴리오, 단계, 순수익, 샤프, t, 롱, 시장, 비용], 5.1: [신호, 포트폴리오, 단계, 순수익, 샤프, 롱, 시장, 비용])
    def pf_check(tag, rows, sample, period, off, has_t):
        sub = pf[(pf["sample"] == sample) & (pf.period == period)].reset_index(drop=True)
        assert len(sub) == len(rows), (tag, len(sub), len(rows))
        for r, (_, x) in zip(rows, sub.iterrows()):
            c = r[off:]
            cmp_cells(tag, c[0], [x.mean_bp], bad, tot)
            cmp_cells(tag, c[1], [x.sharpe], bad, tot)
            k = 2
            if has_t:
                cmp_cells(tag, c[2], [x.t], bad, tot)
                k = 3
            cmp_cells(tag, c[k], [None if pd.isna(x.long_leg_raw_bp) else x.long_leg_raw_bp], bad, tot)
            cmp_cells(tag, c[k + 1], [None if pd.isna(x.market_bp) else x.market_bp], bad, tot)
            cmp_cells(tag, c[k + 2], [x.avg_cost_bp], bad, tot)
    for h, rows in T:
        if "포트폴리오" in h and "평균 비용(bp)" in h:
            if h[0] == "신호":
                for sig_nm, smp in (("6지표순위평균", B6), ("5일반전", B5)):
                    pf_check("5.1 " + smp, [r for r in rows if r[0] == sig_nm], smp, "전체 2015~", 3, False)
            else:
                pf_check("4.1", rows, A, "전체", 2, True)

    # 4.2 32조합
    h, rows = find_table(T, "켠 제약", "bp")
    vv = v32[(v32["sample"] == A) & (v32.period == "전체")].set_index("cons")
    for r in rows:
        for k in (0, 3):
            lab = r[k]
            cmp_cells("4.2 " + lab, r[k + 1], [vv.loc[lab, "mean_bp"]], bad, tot)
            cmp_cells("4.2 " + lab, r[k + 2], [vv.loc[lab, "sharpe"]], bad, tot)

    # 샤플리 표 (4.3, 5.2, 5.4): 어느 표인지는 v(공집합) 행 값으로 구분
    def shap_rows(smp, per):
        d = sh[(sh["sample"] == smp) & (sh.period == per)]
        return d[d.metric == "mean_bp"].set_index("item"), d[d.metric == "sharpe"].set_index("item")
    for h, rows in T:
        if "샤플리 값, bp/주 [95% CI]" in h:
            v0 = [cell_nums(r[1])[0][0] for r in rows if r[0].startswith("v(공집합)")][0]
            smp = [c for c in (A, B6, B5) if abs(shap_rows(c, "전체" if c == A else "전체 2015~")[0].loc["v_empty", "value"] - v0) < 0.06][0]
            m, s_ = shap_rows(smp, "전체" if smp == A else "전체 2015~")
            for r in rows:
                it = r[0].split()[0] if r[0].startswith("c") else ("v_empty" if r[0].startswith("v(공집합)") else ("v_noC1" if r[0].startswith("v(c2") else "v_all"))
                x, y = m.loc[it], s_.loc[it]
                cmp_cells(f"샤플리 {smp} {it}", r[1], [x.value, x.ci_lo, x.ci_hi], bad, tot)
                cmp_cells(f"샤플리 {smp} {it}", r[2], [y.value, y.ci_lo, y.ci_hi], bad, tot)
                if it.startswith("c"):
                    cmp_cells(f"샤플리 {smp} {it}", r[3], [x["first"]], bad, tot)
                    cmp_cells(f"샤플리 {smp} {it}", r[4], [x["last"]], bad, tot)
                    cmp_cells(f"샤플리 {smp} {it}", r[5], [x.marg_min, x.marg_max], bad, tot)

    # 구간별 표 (4.4: 헤더 첫 칸 '표본', 5.3 / 5.5: 'v(공집합) bp' 열이 있음)
    for h, rows in T:
        if h[0] == "표본" or "v(공집합) bp" in h:
            wide = "v(공집합) bp" in h
            if wide:
                c5 = cell_nums(rows[0][6])[0][0]
                smp = B6 if abs(c5 - sh[(sh["sample"] == B6) & (sh.period == "전체 2015~") & (sh.metric == "mean_bp") & (sh.item == "c5")].value.iloc[0]) < 0.06 else B5
            else:
                smp = A
            for r in rows:
                d = sh[(sh["sample"] == smp) & (sh.period == r[0])]
                m, s_ = d[d.metric == "mean_bp"].set_index("item"), d[d.metric == "sharpe"].set_index("item")
                cmp_cells(f"구간 {smp} {r[0]}", r[1], [float(d.n_weeks.iloc[0])], bad, tot)
                for k, it in zip(range(2, 7), ("c1", "c2", "c3", "c4", "c5")):
                    cmp_cells(f"구간 {smp} {r[0]}", r[k], [m.loc[it, "value"]], bad, tot)
                cmp_cells(f"구간 {smp} {r[0]}", r[7], [m.loc["v_empty", "value"]], bad, tot)
                if wide:
                    cmp_cells(f"구간 {smp} {r[0]}", r[8], [s_.loc["v_empty", "value"]], bad, tot)
                    cmp_cells(f"구간 {smp} {r[0]}", r[9], [m.loc["v_noC1", "value"]], bad, tot)
                    cmp_cells(f"구간 {smp} {r[0]}", r[10], [m.loc["v_all", "value"]], bad, tot)
                else:
                    cmp_cells(f"구간 {smp} {r[0]}", r[8], [m.loc["v_noC1", "value"]], bad, tot)
                    cmp_cells(f"구간 {smp} {r[0]}", r[9], [m.loc["v_all", "value"]], bad, tot)

    # 4.5 격자
    sa = sens[sens["sample"] == A].set_index("variant")
    gi = 0
    for h, rows in T:
        if h and h[0] == "" and "스프레드 x0.5" in h:
            col = "v_all_bp" if gi == 0 else "v_noC1_bp"
            gi += 1
            for r in rows:
                fm = r[0].split("x")[1]
                for k, sm in zip((1, 2, 3), ("0.5", "1", "2")):
                    nm = "기본" if (fm == "1" and sm == "1") else f"수수료x{fm} 스프레드x{sm}"
                    cmp_cells(f"4.5 {col} {nm}", r[k], [sa.loc[nm, col]], bad, tot)

    # 6절 변형 표
    for h, rows in T:
        if h[0] == "변형" and "A v(공집합)" in h:
            for r in rows:
                for k, (smp, cols) in enumerate(((A, ("v_empty_bp", "v_noC1_bp", "v_all_bp")), (B6, ("v_empty_bp", "v_noC1_bp", "v_all_bp")), (B5, ("v_empty_bp", "v_noC1_bp", "v_all_bp")))):
                    row = sens[(sens["sample"] == smp) & (sens.variant == r[0])].iloc[0]
                    for j, c in enumerate(cols):
                        cmp_cells(f"6 v {smp} {r[0]}", r[1 + 3 * k + j], [row[c]], bad, tot)
        if h[0] == "변형" and "A φ(c1)" in h:
            for r in rows:
                for k, smp in enumerate((A, B6, B5)):
                    row = sens[(sens["sample"] == smp) & (sens.variant == r[0])].iloc[0]
                    for j, c in enumerate(("phi_c1_bp", "phi_c4_bp", "phi_c5_bp")):
                        cmp_cells(f"6 φ {smp} {r[0]}", r[1 + 3 * k + j], [row[c]], bad, tot)

    # CS 시뮬레이션 표
    h, rows = find_table(T, "일 변동성", "야간 갭 표준편차")
    for r, (_, x) in zip(rows, sim.iterrows()):
        cmp_cells("CS 시뮬", r[0], [x.daily_vol * 100], bad, tot)
        cmp_cells("CS 시뮬", r[1], [x.overnight_gap_sd * 100], bad, tot)
        cmp_cells("CS 시뮬", r[2], [x.true_half_spread_pct], bad, tot)
        cmp_cells("CS 시뮬", r[3], [x.cs_half_floor0_pct], bad, tot)
        cmp_cells("CS 시뮬", r[4], [x.cs_half_unfloored_pct], bad, tot)

    # 9절 진단 표
    h, rows = find_table(T, "항목", "A CNN", "B 6지표")
    dgi = dg_.set_index("sample")
    fmap = {"배치(형성일) 수": ("n_batches", 1, 0), "형성일당 신호 종목 수": ("avg_universe", 1, 0), "보유 겹치는 배치(형성일 간격<5거래일)": ("overlap_batches", 1, 0),
            "공매도 금지 진입일 배치": ("ban_batches", 1, 0), "롱 진입 불가 비율(상한가 시가·정지)": ("buy_blocked_frac", 100, 2), "숏 진입 불가 비율(하한가 시가·정지)": ("short_blocked_frac", 100, 2),
            "롱 청산 이월 비율": ("long_exit_carried_frac", 100, 2), "숏 청산 이월 비율": ("short_exit_carried_frac", 100, 2), "CS 반스프레드 중앙값(%)": ("hs_median_pct", 1, 3),
            "CS 반스프레드 평균(%)": ("hs_mean_pct", 1, 3), "틱/2 중앙값(%)": ("tick_half_median_pct", 1, 3), "롱 레그 연속 재편입 비율": ("retention_long_leg", 100, 0),
            "가격 없어 마지막 가격으로 평가한 거래 수": ("stale_exit_trades", 1, 0)}
    for r in rows:
        if r[0] in fmap:
            col, mult, _ = fmap[r[0]]
            for k, smp in enumerate((A, B6, B5)):
                cmp_cells("9 " + r[0], r[1 + k], [dgi.loc[smp, col] * mult], bad, tot)
        elif r[0].startswith("자료 오류"):
            for k, smp in enumerate((A, B6, B5)):
                cmp_cells("9 자료 오류", r[1 + k], [(1 - dgi.loc[smp, "ok_frac"]) * 100], bad, tot)
    return tot[0], bad


def ctx_marg(v, i, j=None, state=None):
    """제약 i 의 한계기여 v(S+i)-v(S) 를 S 에 j 가 (state=1) 있을 때/(0) 없을 때, 나머지 제약 상태에 대해 평균."""
    ms = [v[m | 1 << i] - v[m] for m in range(bt.N_SUB) if not m >> i & 1 and (j is None or ((m >> j & 1) == state))]
    return float(np.mean(ms)) * 1e4


def narrative_checks(tr_open, tr_close):
    """보고서 서술 속 숫자 중 표에 없는 것을 다시 계산해 대조한다. 반환 DataFrame(주장, 보고서, 재계산, 판정)."""
    v32 = pd.read_csv(f"{TAB}/v32.csv", encoding="utf-8-sig")
    sh = pd.read_csv(f"{TAB}/shapley.csv", encoding="utf-8-sig")
    sens = pd.read_csv(f"{TAB}/sensitivity.csv", encoding="utf-8-sig")
    A, B5 = "A_CNN파일럿", "B_5일반전"
    va = v32[(v32["sample"] == A) & (v32.period == "전체")].sort_values("mask").mean_bp.to_numpy() / 1e4
    vb = v32[(v32["sample"] == B5) & (v32.period == "전체 2015~")].sort_values("mask").mean_bp.to_numpy() / 1e4
    phi = sh[(sh["sample"] == A) & (sh.period == "전체") & (sh.metric == "mean_bp")].set_index("item").value
    rows = []

    def add(claim, memo, calc, tol=0.051):
        rows.append(dict(주장=claim, 보고서=memo, 재계산=round(float(calc), 3), 판정="일치" if abs(float(memo) - float(calc)) <= tol else "불일치"))
    add("c4+c5 샤플리 합 (bp)", -142.4, phi["c4"] + phi["c5"])
    add("스프레드 몫 (%)", 65, -phi["c5"] / -(phi["c4"] + phi["c5"]) * 100, 0.5)
    add("가격제한 샤플리 / 총수익 (%)", 2, -phi["c3"] / (va[0] * 1e4) * 100, 0.5)
    add("c4 한계기여, c1 꺼짐 평균", -66.1, ctx_marg(va, 3, 0, 0))
    add("c5 한계기여, c1 꺼짐 평균", -126.7, ctx_marg(va, 4, 0, 0))
    add("c4 한계기여, c1 켜짐 평균", -33.2, ctx_marg(va, 3, 0, 1))
    add("c5 한계기여, c1 켜짐 평균", -58.9, ctx_marg(va, 4, 0, 1))
    add("c1 한계기여, c5 꺼짐 평균", -23.2, ctx_marg(va, 0, 4, 0))
    add("c1 한계기여, c5 켜짐 평균", 44.6, ctx_marg(va, 0, 4, 1))
    # 한계기여가 다른 제약 상태에 따라 10bp 넘게 달라지는 쌍 (A): |평균 한계기여(j 켜짐) - 평균 한계기여(j 꺼짐)| > 10
    pairs = []
    for i in range(5):
        for j in range(5):
            if i != j:
                d = ctx_marg(va, i, j, 1) - ctx_marg(va, i, j, 0)
                if abs(d) > 10:
                    pairs.append((bt.CONS[i], bt.CONS[j], round(d, 1)))
    rows.append(dict(주장="A: 한계기여가 상대 제약 상태에 따라 10bp 넘게 달라지는 (i,j) 쌍", 보고서="전부 c1이 낀 쌍", 재계산=str(pairs),
                     판정="일치" if all("c1" in p[:2] for p in pairs) else "불일치"))
    pairs5 = []
    for i in range(5):
        for j in range(5):
            if i != j:
                d = ctx_marg(vb, i, j, 1) - ctx_marg(vb, i, j, 0)
                if abs(d) > 10:
                    pairs5.append((bt.CONS[i], bt.CONS[j], round(d, 1)))
    rows.append(dict(주장="B 5일반전: 같은 쌍 목록 (보고서: c1-c2 +5.4 vs -45.1, c2-c5 16bp 추가)", 보고서="c1-c2, c2-c5 추가", 재계산=str(pairs5), 판정="수동 확인"))
    rows.append(dict(주장="B 5일반전 c2 한계기여, c1 꺼짐/켜짐 평균", 보고서="+5.4 vs -45.1", 재계산=f"{ctx_marg(vb, 1, 0, 0):+.1f} vs {ctx_marg(vb, 1, 0, 1):+.1f}", 판정="수동 확인"))
    rows.append(dict(주장="B 5일반전 c1 한계기여, c2 꺼짐/켜짐 평균", 보고서="+5.4 vs -45.1", 재계산=f"{ctx_marg(vb, 0, 1, 0):+.1f} vs {ctx_marg(vb, 0, 1, 1):+.1f}", 판정="수동 확인"))
    # 시장 열과 종가 체결, 종가 체결 이득 비율
    add("시장 평균, 형성일 종가 체결 (bp)", -13.3, tr_close.mkt.mean() * 1e4)
    add("시장 평균, 익일 시가 체결 (bp)", -16.9, tr_open.mkt.mean() * 1e4)
    r_c, r_o = bt.run(tr_close, ()).ret.mean(), bt.run(tr_open, ()).ret.mean()
    add("총수익 중 종가->시가 구간 비중 (%)", 10, (r_c - r_o) / r_c * 100, 0.5)
    # 수수료·세금 분해
    add("롱숏 1배치 세금 (bp)", 60, 2 * 0.30 * 1e2, 0.5)
    add("c4 단독 롱숏 순수익 (bp)", 29.7, va[bt.mask_of(("c4",))] * 1e4)
    add("c4 단독으로 줄어든 폭 (bp)", 66, (va[0] - va[bt.mask_of(("c4",))]) * 1e4, 0.5)
    return pd.DataFrame(rows)


# =====================================================================================================
# 6. 재현·교차검증 (파일럿 재현, 샤플리 무차별 대입, 부트스트랩 재현, b01_test 실행, B 최근 구간)
# =====================================================================================================
def pilot_replication(q, tr_close, tr_open):
    """results.json 의 d_lr1e-4 를 cnn.evaluate 규칙(rank first qcut 10)으로 다시 계산하고, b01 규칙(round-half-up 0.1n)을 같은 라벨에 적용한 값과 비교."""
    import json
    res = [r for r in json.load(open(f"{ROOT}/results/kaggle/results.json", encoding="utf-8")) if r["name"] == "d_lr1e-4"][0]
    m = q.copy()
    m["qb"] = m.groupby("date").prob.transform(lambda p: pd.qcut(p.rank(method="first"), 10, labels=False))
    dec = m.groupby(["date", "qb"]).ret.mean().unstack()
    hl = dec[9] - dec[0]

    def topk(g):
        k = int(math.floor(0.1 * len(g) + 0.5))
        gs = g.sort_values(["prob", "ticker"], ascending=[False, True])
        gb = g.sort_values(["prob", "ticker"], ascending=[True, True])
        return gs.ret.iloc[:k].mean() - gb.ret.iloc[:k].mean()
    hl2 = m.groupby("date")[["prob", "ticker", "ret"]].apply(topk)
    rc, ro = bt.run(tr_close, ())["ret"], bt.run(tr_open, ())["ret"]
    return pd.DataFrame([dict(항목="results.json (d_lr1e-4)", 평균_bp=res["hl_weekly_mean"] * 1e4, 샤프=res["hl_sharpe"]),
                         dict(항목="pred 라벨 + cnn.evaluate 규칙 재계산", 평균_bp=hl.mean() * 1e4, 샤프=hl.mean() / hl.std() * math.sqrt(52)),
                         dict(항목="pred 라벨 + b01 선택 규칙(round-half-up 0.1n)", 평균_bp=hl2.mean() * 1e4, 샤프=bt.sharpe(hl2, 52)),
                         dict(항목="b01 엔진, 형성일 종가 체결(패널 수정주가)", 평균_bp=rc.mean() * 1e4, 샤프=bt.sharpe(rc, 52)),
                         dict(항목="b01 엔진, 익일 시가 체결", 평균_bp=ro.mean() * 1e4, 샤프=bt.sharpe(ro, 52))]).set_index("항목")


def shapley_bruteforce(v):
    """120개 순서 전부를 돌려 한계기여를 평균한 샤플리 값(행렬 공식과 독립)."""
    from itertools import permutations
    v = np.asarray(v, float)
    phi = np.zeros(5)
    for perm in permutations(range(5)):
        m = 0
        for i in perm:
            phi[i] += v[m | 1 << i] - v[m]
            m |= 1 << i
    return phi / 120


def cross_checks(tr, R):
    import b02_constraints as b02
    out = {}
    vm = R.mean().to_numpy()
    out["샤플리 행렬공식 대 120순열 최대차(bp)"] = float(np.abs(shapley_bruteforce(vm) - bt.shapley(vm)).max() * 1e4)
    v_rows, s_rows, gap = b02.decompose("A_CNN파일럿", "전체", R.to_numpy(), 52.0)       # 같은 시드로 부트스트랩 재실행
    sh = pd.read_csv(f"{TAB}/shapley.csv", encoding="utf-8-sig")
    ref = sh[(sh["sample"] == "A_CNN파일럿") & (sh.period == "전체")].reset_index(drop=True)
    new = pd.DataFrame(s_rows)
    cols = ["value", "ci_lo", "ci_hi", "first", "last", "marg_min", "marg_max"]
    out["shapley.csv 재현 최대차(값·CI·맥락범위, 단위 혼합)"] = float(np.nanmax(np.abs(ref[cols].to_numpy(float) - new[cols].to_numpy(float))))
    v32 = pd.read_csv(f"{TAB}/v32.csv", encoding="utf-8-sig")
    r32 = v32[(v32["sample"] == "A_CNN파일럿") & (v32.period == "전체")].sort_values("mask")
    out["v32.csv 재현 최대차(bp)"] = float(np.abs(r32.mean_bp.to_numpy() - vm * 1e4).max())
    out["효율성 |합φ-(v전부-v공집합)|"] = float(abs(bt.shapley(vm).sum() - (vm[-1] - vm[0])))
    return out


def run_b01_test():
    import subprocess
    p = subprocess.run([sys.executable, "-I", "-X", "utf8", f"{ROOT}/code/b01_test.py"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.startswith(("PASS", "전부 통과"))]
    return p.returncode, lines


def b6_recent_check():
    """보고서 5.3 의 2025-03-31 이후 서술(시장 +149bp, 상위20 롱 레그 +30bp, 비용 전 초과 -119bp, 비용 109bp)을 B 6지표로 다시 계산."""
    P = bt.load_panel("2014-09-01", extra=list(bt.SIGN))
    mk = bt.Market(P)
    dates = bt.rebalance_dates(mk.cal, "2015-01-01")
    sig = bt.signal_rankavg(P, dates)
    tr = bt.make_trades(mk, sig)
    sel = np.asarray(tr.entry_dates >= pd.Timestamp("2025-03-31"))
    g = bt.run(tr, ("c1", "c2"))[sel]
    n = bt.run(tr, ("c1", "c2", "c3", "c4", "c5"))[sel]
    ls = bt.run(tr, ())[sel]
    return pd.DataFrame([dict(항목="주 수", 보고서=74, 재계산=int(sel.sum())), dict(항목="롱숏 총수익 (bp)", 보고서=89.6, 재계산=ls.ret.mean() * 1e4),
                         dict(항목="시장 (bp/주)", 보고서=149, 재계산=g.mkt.mean() * 1e4), dict(항목="상위20 롱 레그 총수익 (bp)", 보고서=30, 재계산=g.long.mean() * 1e4),
                         dict(항목="비용 전 초과 (bp)", 보고서=-119, 재계산=g.ret.mean() * 1e4), dict(항목="비용 (bp)", 보고서=109, 재계산=n.cost.mean() * 1e4),
                         dict(항목="v(전부) (bp)", 보고서=-228.7, 재계산=n.ret.mean() * 1e4)]).set_index("항목")


# =====================================================================================================
# main: 모든 부분 실행 -> results/v02_tables/*.csv + v02_tables.md (보고서에 옮길 표)
# =====================================================================================================
def fmt(x):
    if isinstance(x, (float, np.floating)):
        return "" if not np.isfinite(x) else (f"{x:.4f}" if abs(x) < 1 and x != 0 else f"{x:.2f}")
    return str(x)


def md(df, index=True):
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    return head + "\n".join("| " + " | ".join(fmt(x) for x in r) + " |" for r in d.itertuples(index=False)) + "\n"


def main():
    import time
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    md_out = []

    def emit(name, title, df, index=True):
        (df.reset_index() if index else df).to_csv(f"{OUT}/{name}.csv", index=False, encoding="utf-8-sig")
        md_out.append(f"### {title}\n\n{md(df, index)}\n")
        print(f"[{time.time() - t0:5.0f}s] {name}", flush=True)

    P, q, pr = load_all()
    F = pd.read_parquet(FLAGS)
    o1, mk, tr, I = part1(P, q, pr)
    trc = bt.make_trades(mk, pr, entry="close")
    emit("p1_five_dates", "1-a. 형성일 6개 총수익 대조 (bp)", o1["five_dates"], index=False)
    emit("p1_blind_all", "1-b. 100주 전수: spec-only 총수익 대 b01", o1["blind_all"])
    emit("p1_dropped", "1-c. b01 이 이상값으로 제외한 24개 거래", o1["dropped"], index=False)
    emit("p1_cost_all", "1-d. 비용 켠 조합 100주 전수 대조 (Ind 덧셈 규약 / exact 현금흐름)", o1["cost_all"], index=False)
    emit("p1_cost_dates", "1-d2. 형성일 6개 x 비용 조합 (bp)", o1["cost_dates"], index=False)
    emit("p1_hand", f"1-e. 손계산 표 ({o1['hand_date'].date()}, c3+c4+c5, 롱·숏 각 상위 3종목)", o1["hand"], index=False)
    md_out.append(f"Ind 가 상폐 거래에서 매도일 행 없음으로 세금 0 을 쓴 기록: {I.nan_tax}\n\n")
    o2 = part2(P, q, pr, mk, tr, I)
    emit("p2_cs_align", "2-a. CS 창 정렬", o2["cs_align"], index=False)
    emit("p2_perturb", "2-b. 미래/과거 교란 실험 (hs 변화)", o2["perturb"], index=False)
    emit("p2_label", "2-c. 라벨 정합", o2["label"])
    md_out.append(f"진입일 = 형성일 다음 거래일 (100주 전부): {o2['entry_ok']}\n\n")
    scen, expo, conf = part3(P, pr, F)
    emit("p3_scenarios", "3-a. 감사 버그 시나리오별 (A, bp/주)", scen, index=False)
    emit("p3_exposure", "3-b. 플래그 노출 (A 거래 71,222건 기준)", expo, index=False)
    emit("p3_conf", "3-c. c3 판정 혼동표 (패널 플래그 대 정확 한도)", conf, index=False)
    sv = survivorship(P, q, I)
    emit("p3_survivorship", "3-d. 신호 유니버스 밖 상폐 직전 종목 (라벨 없음)", sv, index=False)
    r_ = sv.r_진입시가_마지막가.fillna(0.0)
    kbar, nw = float(tr.n_g.mean() / 10), len(tr.dates)
    emit("p3_surv_bound", "3-e. 생존편향 경계 (이 종목들이 신호표에 있었다면)", pd.Series({
        "이벤트 수(100주)": len(sv), "진입 가능했던 건수": int(sv.r_진입시가_마지막가.notna().sum()), "수익률 합(진입 시가->마지막 가격)": float(r_.sum()),
        "전부 10분위 롱 레그였다면 LS 평균 변화(bp/주)": float(r_.sum() / kbar / nw * 1e4), "전부 숏 레그였다면 LS 평균 변화(bp/주)": float(-r_.sum() / kbar / nw * 1e4),
        "무작위 순위일 때 10분위 롱 레그 평균 변화(bp/주)": float(r_.sum() * 0.1 / kbar / nw * 1e4), "상위20 롱온리에 무작위 배치 시(bp/주)": float(r_.sum() * (20 / tr.n_g.mean()) / 20 / nw * 1e4),
        "전부 상위20 롱온리에 들었다면(bp/주)": float(r_.sum() / 20 / nw * 1e4)}, name="값").to_frame())
    o4 = part4(P, pr, q, mk, tr)
    for k, t in (("variants", "4-a. 변형 재계산 (b01 엔진)"), ("netturn_check", "4-b. 매매 상쇄 절감액 독립 추정"), ("legs", "4-c. 레그·구간별 평균 (bp/주)"),
                 ("periods", "4-d. 구간별 통계"), ("quarters", "4-e. 분기별"), ("dependence", "4-f. 특정 주 의존"), ("moments", "4-g. 분포"),
                 ("char_ic", "4-h. 신호 순위상관 (형성일 횡단면, 100주 평균)"), ("char_legs", "4-i. 레그 특성"), ("reversal", "4-j. 단순 5일 반전 대조"),
                 ("subuniverse", "4-k. 유니버스 제한 재실행"), ("reversal_periods", "4-l. 단순 5일 반전의 구간별 통계"), ("overlap", "4-m. 보유가 겹치는 17개 배치")):
        emit("p4_" + k, t, o4[k], index=k not in ("variants", "dependence", "subuniverse"))
    W = o4["weekly"]
    W.assign(ban=W.ban.astype(int)).to_csv(f"{OUT}/p4_weekly.csv", encoding="utf-8-sig")
    n, bad = check_memo_vs_csv()
    md_out.append(f"### 5-a. 보고서 표 대 CSV 칸 단위 대조\n\n검사한 칸 {n}, 불일치 {len(bad)}\n\n" + "\n".join(f"- {b}" for b in bad) + "\n\n")
    print(f"[{time.time() - t0:5.0f}s] memo table check {n} / {len(bad)}", flush=True)
    emit("p5_narrative", "5-b. 서술 속 숫자 재계산", narrative_checks(tr, trc), index=False)
    emit("p6_pilot", "6-a. 파일럿 재현", pilot_replication(q, trc, tr))
    R = bt.all_subsets(tr)
    emit("p6_cross", "6-b. 샤플리·부트스트랩 재현", pd.Series(cross_checks(tr, R), name="값").to_frame())
    rc, lines = run_b01_test()
    md_out.append(f"### 6-c. b01_test.py 실행: 종료코드 {rc}\n\n" + "\n".join(f"- {ln}" for ln in lines) + "\n\n")
    print(f"[{time.time() - t0:5.0f}s] b01_test rc={rc}", flush=True)
    emit("p6_b6recent", "6-d. B 6지표 2025-03-31 이후 서술 재계산", b6_recent_check())
    open(f"{OUT}/v02_tables.md", "w", encoding="utf-8").write("\n".join(md_out))
    print(f"완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

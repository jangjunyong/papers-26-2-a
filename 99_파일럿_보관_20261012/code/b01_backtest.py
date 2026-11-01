"""RQ4 백테스터 + 제약별 샤플리 분해 (탐색용 라이브러리). 사용: b02_constraints.py, 검증: b01_test.py

[흐름] 신호표(date, ticker, score) -> 형성일 종가에 확정 -> 10분위(또는 상위 N) 선택 -> 체결 -> 5거래일 보유 -> 비용 -> 주간 수익.
[체결 규칙] 기본 = 형성일 다음 거래일 시가 매수/매도(entry="open"), 청산 = 형성일로부터 H(=5)번째 거래일 종가.
  entry="close"는 형성일 종가 체결(민감도 옵션). 수익은 수정주가 기준: 진입가 = 시가 x (수정종가/원종가).
  배치(형성일)마다 독립으로 진입-청산한다(전량 왕복 매매 가정). 형성일 간격이 5거래일보다 짧은 주(공휴일 낀 주)는 보유가 겹친다.
  ponytail: 겹침·재편입 종목의 매매 상쇄는 기본으로 안 한다(비용 과대 쪽 가정). net_turnover=True로 상쇄 민감도.

[제약 5개] 각각 켜고 끌 수 있다 (비트 0~4, 부분집합 32개).
  c1 숏 제거        : 롱 레그만 보유. 성과 = 롱 레그 순수익 - 같은 기간 시총가중 시장수익 (c1_mode="excess", 기본).
                      c1_mode="raw"면 시장 차감 없는 롱온리 총수익.
  c2 상위 N종목 집중: 10분위 대신 상위 N(기본 20)종목 (롱숏이면 상·하위 각 N).
  c3 체결 제약      : 진입일 시가 상한가 근접 -> 매수 불가(제외), 시가 하한가 근접 -> 공매도 불가(제외),
                      거래정지·거래량 0 -> 진입 불가. 청산일 종가 하한가(롱 매도)/상한가(숏 환매수)/정지 -> 다음 거래일 종가로 이월.
                      ponytail: 제외 종목은 차순위로 대체하지 않는다(동일가중 재배분만).
  c4 거래세·수수료  : 매도 시 sell_tax_pct(해당 매도일 값), 매수·매도 각각 fee. 숏의 세금은 숏 진입 매도에 부과.
  c5 호가 스프레드  : 종목별 20일 평균 Corwin-Schultz 반스프레드를 진입·청산 각각에 부과(편도).
  대차비용은 제약이 아니라 민감도 매개변수(borrow, 연율)다. 기본 0.

[c1 기준 정의] 샤플리 게임 v(S) = 부분집합 S가 켜진 포트폴리오의 주간 순수익 평균(또는 연율 샤프).
  c1이 꺼진 상태 = 10분위 롱숏 순수익(시장 중립 비슷한 영비용 포트폴리오),
  c1이 켜진 상태 = 롱온리의 '시장 대비 초과' 순수익. 둘 다 투자금 1(롱 레그 명목) 기준의 제로코스트 포트폴리오라 비교 가능하게 맞췄다.
  따라서 c1의 한계기여 = (롱온리 초과수익 - 롱숏 수익)이며 비용이 아니라 포트폴리오가 바뀌는 효과다.
  v(공집합) = 비용·제약 없는 10분위 롱숏 총수익, v(전부) = 5개 모두 반영한 상위 N 롱온리 초과 순수익.

[스프레드] Corwin-Schultz(2012): 2일 고저가로 전체 스프레드 S 추정, 음수는 0, 종목별 20일 평균(<=형성일), 편도 = S/2.
  결측(20일 중 10일 미만 유효)이면 tick_pct/2(%)로 대체. 거래정지·거래량 0·H==L인 날과 수정계수가 바뀐 날(액면분할 등)은 추정에서 뺀다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CONS = ("c1", "c2", "c3", "c4", "c5")
NAME = {"c1": "숏 제거", "c2": "상위 20종목 집중", "c3": "상·하한가·정지 체결", "c4": "거래세·수수료", "c5": "호가 스프레드"}
FEE = 0.00015      # 편도 수수료 0.015% (가정, 매개변수)
LIM_TOL = 0.004    # 상·하한가 근접 허용오차 (p01_build_panel과 같은 값)
N_SUB = 1 << len(CONS)


def subset(m):
    return tuple(c for i, c in enumerate(CONS) if m >> i & 1)


def mask_of(cons):
    return sum(1 << CONS.index(c) for c in cons)


# ---------------------------------------------------------------- 스프레드
def cs_spread(H, L, valid, floor=True):
    """Corwin-Schultz 2일 스프레드(전체 스프레드, 가격 대비). 행=날짜(오름차순), 열=종목. valid가 아닌 날은 NaN. 음수는 0."""
    H, L = np.where(valid, H, np.nan), np.where(valid, L, np.nan)
    h1, l1 = np.roll(H, 1, 0), np.roll(L, 1, 0)
    h1[0], l1[0] = np.nan, np.nan
    with np.errstate(all="ignore"):
        beta = np.log(H / L) ** 2 + np.log(h1 / l1) ** 2
        gamma = np.log(np.maximum(H, h1) / np.minimum(L, l1)) ** 2
        k = 3 - 2 * math.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return np.maximum(s, 0.0) if floor else s    # NaN은 그대로 NaN


# ---------------------------------------------------------------- 시장 행렬
class Market:
    """일별 패널(long) -> 날짜 x 종목 행렬. 필요한 열: date,ticker,open,high,low,close,adj_close,volume,is_halted,ret_outlier,
    mktcap,lim_up,lim_dn,limit_pct,sell_tax_pct,tick_pct,short_banned (p01_build_panel 정의). 'half_spread' 열이 있으면 그대로 쓴다(테스트용)."""

    def __init__(self, panel, start=None, end=None, buffer_days=45):
        p = panel
        if start is not None:
            p = p[p["date"] >= pd.Timestamp(start) - pd.Timedelta(days=buffer_days)]
        if end is not None:
            p = p[p["date"] <= pd.Timestamp(end) + pd.Timedelta(days=buffer_days)]
        self.cal = pd.DatetimeIndex(np.sort(p["date"].unique()))
        self.tk = pd.Index(np.sort(p["ticker"].unique()))
        di, ti = self.cal.get_indexer(p["date"]), self.tk.get_indexer(p["ticker"])
        nd, nt = len(self.cal), len(self.tk)

        def mat(col, fill=np.nan):
            M = np.full((nd, nt), fill)
            M[di, ti] = p[col].to_numpy(float)
            return M

        O, Hh, Lw, C, A, V = (mat(c) for c in ("open", "high", "low", "close", "adj_close", "volume"))
        halt = mat("is_halted", 0) > 0
        with np.errstate(all="ignore"):
            has = np.isfinite(C)
            self.NOTRADE = has & (halt | ~(V > 0))                          # 정지 또는 거래량 0/결측
            F = A / C                                                        # 수정계수
            LIM = mat("limit_pct") / 100
            prevC = np.vstack([np.full((1, nt), np.nan), C[:-1]])
            oret = O / prevC - 1                                             # 시가의 전일 종가 대비 등락
            corp = np.abs(oret) > LIM + 0.05                                 # 액면분할 등 원주가 단절
            self.OU = (oret >= LIM - LIM_TOL) & ~corp & (V > 0)              # 시가 상한가 근접
            self.OD = (oret <= -LIM + LIM_TOL) & ~corp & (V > 0)             # 시가 하한가 근접
            LU, LD = mat("lim_up", 0) > 0, mat("lim_dn", 0) > 0              # 종가 기준 상·하한가 (panel 열)
            self.BLK_SELL = self.NOTRADE | LD                                # 롱 청산(매도) 불가
            self.BLK_BUY = self.NOTRADE | LU                                 # 숏 청산(환매수) 불가
            self.LU, self.LD = LU, LD
            if "half_spread" in p:
                HS = mat("half_spread")
            else:
                valid = (V > 0) & ~halt & (Hh > Lw)
                Fp = np.vstack([np.full((1, nt), np.nan), F[:-1]])
                S = np.where(np.abs(F / Fp - 1) < 1e-6, cs_spread(Hh, Lw, valid), np.nan)
                HS = 0.5 * pd.DataFrame(S).rolling(20, min_periods=10).mean().to_numpy()
                HS = np.where(np.isnan(HS), mat("tick_pct") / 200, HS)       # tick_pct는 %, 반스프레드 = 틱/2
        self.O, self.A, self.F, self.HS = O, A, F, HS
        self.HS_TICK = mat("tick_pct") / 200                                  # 틱 하한 대용: 1틱 = 전체 스프레드, 반 = 틱/2
        self.Aff = pd.DataFrame(A).ffill(limit=20).to_numpy()                # 상폐·결측일은 마지막 가격으로 평가
        self.OC = np.vstack([np.zeros((1, nt)), np.cumsum(np.nan_to_num(mat("ret_outlier", 0)), axis=0)])
        self.TAX = mat("sell_tax_pct") / 100
        self.MC = mat("mktcap")
        self.BAN = mat("short_banned", 0).max(axis=1) > 0                   # 날짜별 공매도 전면금지 여부


# ---------------------------------------------------------------- 거래 표
class Trades:
    pass


def make_trades(mk, sig, H=5, entry="open", carry_max=10):
    """신호표 -> 종목-배치 단위 거래 정보. 제약과 무관한 값(가격·플래그·비용 요소)만 미리 계산해 둔다."""
    s = sig[["date", "ticker", "score"]].dropna().copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    nd = len(mk.cal)
    di, ti = mk.cal.get_indexer(s["date"]), mk.tk.get_indexer(s["ticker"])
    keep = (di >= 0) & (ti >= 0) & (di + H < nd) & (di + 1 < nd)
    t = Trades()
    t.n_dropped_rows = int((~keep).sum())      # 달력·종목에 없거나 청산일이 자료 끝을 넘는 행
    s, di, ti = s[keep].reset_index(drop=True), di[keep], ti[keep]
    e = di + 1 if entry == "open" else di
    with np.errstate(all="ignore"):
        if entry == "open":
            pin = mk.O[e, ti] * mk.F[e, ti]
            nt_in, bu, bs = mk.NOTRADE[e, ti], mk.OU[e, ti], mk.OD[e, ti]
        else:
            pin = mk.A[di, ti]
            nt_in, bu, bs = mk.NOTRADE[di, ti], mk.LU[di, ti], mk.LD[di, ti]
        t.blk_buy, t.blk_sell_short = nt_in | bu, nt_in | bs          # 진입 불가(롱 매수, 숏 매도)
        x0 = di + H

        def carry(blk):                                               # 청산 불가면 다음 거래일 종가로 이월
            x = x0.copy()
            for _ in range(carry_max):
                b = blk[x, ti] & (x + 1 < nd)
                if not b.any():
                    break
                x = np.where(b, x + 1, x)
            return x

        xl, xs = carry(mk.BLK_SELL), carry(mk.BLK_BUY)
        xm = np.maximum(xl, xs)
        out_in = (mk.OC[xm + 1, ti] - mk.OC[e, ti]) > 0               # 보유 구간에 수정주가 이상값이 있으면 자료 오류로 제외
        Af = mk.Aff
        t.r0, t.rl, t.rs = Af[x0, ti] / pin - 1, Af[xl, ti] / pin - 1, Af[xs, ti] / pin - 1
        t.ok = np.isfinite(pin) & (pin > 0) & np.isfinite(t.r0) & np.isfinite(t.rl) & np.isfinite(t.rs) & ~out_in
        t.stale = np.isnan(mk.A[x0, ti]) & t.ok                       # 청산일 가격이 없어 마지막 가격을 쓴 거래
        t.carried_l, t.carried_s = xl > x0, xs > x0
        t.tax_e = mk.TAX[e, ti]                                       # 숏 진입 매도일 세율
        t.tax_xl = mk.TAX[xl, ti]                                     # 롱 청산 매도일 세율
        t.hs = mk.HS[di, ti]                                          # 형성일 기준 반스프레드(<=형성일 자료)
        t.nd_short = (xs - e + 1) if entry == "open" else (xs - di)   # 숏 보유 거래일 수(대차비용용)
        mc = mk.MC[di, ti]
    t.hs = np.nan_to_num(t.hs, nan=0.0)
    t.hs_tick = np.nan_to_num(mk.HS_TICK[di, ti], nan=0.0)
    t.tax_e, t.tax_xl = np.nan_to_num(t.tax_e), np.nan_to_num(t.tax_xl)
    t.tick_idx, t.score = ti, s["score"].to_numpy()
    t.gid = pd.factorize(s["date"])[0]
    G = int(t.gid.max()) + 1 if len(s) else 0
    t.G, t.n_g = G, np.bincount(t.gid, minlength=G)
    sc = pd.Series(t.score)
    t.rk_d = sc.groupby(t.gid).rank(method="first", ascending=False).to_numpy()
    t.rk_a = sc.groupby(t.gid).rank(method="first", ascending=True).to_numpy()
    first = np.r_[0, np.flatnonzero(np.diff(t.gid)) + 1]
    t.dates = pd.DatetimeIndex(s["date"].to_numpy()[first])
    t.entry_dates = mk.cal[e[first]]
    t.ban = mk.BAN[e[first]]
    w = np.where(t.ok & np.isfinite(mc), mc, 0.0)                     # 시총가중 시장수익(같은 신호 유니버스, 비용 없음, 이월 없음)
    t.mkt = np.bincount(t.gid, w * np.where(w > 0, t.r0, 0.0), G) / np.maximum(np.bincount(t.gid, w, G), 1e-300)
    t.NT = len(mk.tk)
    return t


# ---------------------------------------------------------------- 실행
def run(tr, cons=(), fee=FEE, spread_mult=1.0, borrow=0.0, tax_mult=1.0, topn=20, frac=0.10,
        net_turnover=False, c1_mode="excess", spread="cs"):
    """제약 집합 cons를 켠 포트폴리오의 배치(형성일)별 순수익. 행=형성일. spread="tick"이면 CS 대신 틱/2(하한 근사)."""
    c = set(cons)
    c1, c2, c3, c4, c5 = (x in c for x in CONS)
    ng = tr.n_g
    kg = np.minimum(topn, np.maximum(1, ng // 2)) if c2 else np.maximum(1, np.floor(frac * ng + 0.5).astype(int))
    k = np.where(ng >= 2, kg, 0)[tr.gid]
    inL = (tr.rk_d <= k) & tr.ok
    inS = (tr.rk_a <= k) & tr.ok & (not c1)
    if c3:
        inL, inS = inL & ~tr.blk_buy, inS & ~tr.blk_sell_short
        gl, gs = tr.rl, tr.rs
    else:
        gl = gs = tr.r0
    in_l = in_s = out_l = out_s = 1.0                                  # 진입·청산 비용 배수 (net_turnover면 유지 종목은 0)
    if net_turnover:
        def hold(m, dg):
            key = np.flatnonzero(m)
            kk = tr.gid[key].astype(np.int64) * tr.NT + tr.tick_idx[key]
            q = (tr.gid + dg).astype(np.int64) * tr.NT + tr.tick_idx
            return np.isin(q, kk)
        in_l, out_l = 1.0 - hold(inL, -1), 1.0 - hold(inL, 1)
        in_s, out_s = 1.0 - hold(inS, -1), 1.0 - hold(inS, 1)
    # 롱: 진입 명목 1(매수), 청산 명목 1+g(매도)
    cost_l = np.zeros(len(gl))
    if c4:
        cost_l = cost_l + fee * (in_l + out_l * (1 + gl)) + tax_mult * tr.tax_xl * out_l * (1 + gl)
    hs = tr.hs_tick if spread == "tick" else tr.hs
    if c5:
        cost_l = cost_l + spread_mult * hs * (in_l + out_l * (1 + gl))
    # 숏: 진입 명목 1(매도, 세금 부과), 청산 명목 1+g(환매수)
    cost_s = borrow * tr.nd_short / 252 * np.ones(len(gs))
    if c4:
        cost_s = cost_s + fee * (in_s + out_s * (1 + gs)) + tax_mult * tr.tax_e * in_s
    if c5:
        cost_s = cost_s + spread_mult * hs * (in_s + out_s * (1 + gs))
    netL, netS = gl - cost_l, -gs - cost_s

    def agg(x, m):
        n = np.bincount(tr.gid, m.astype(float), tr.G)
        return np.bincount(tr.gid, np.where(m, x, 0.0), tr.G) / np.maximum(n, 1), n

    lm, nl = agg(netL, inL)
    sm, ns = agg(netS, inS)
    cl, _ = agg(cost_l, inL)
    cs, _ = agg(cost_s, inS)
    ret = (lm - tr.mkt if c1_mode == "excess" else lm) if c1 else lm + sm   # 레그가 비면 현금(0)으로 둔다
    return pd.DataFrame({"ret": ret, "long": lm, "short": sm, "mkt": tr.mkt, "nL": nl, "nS": ns, "cost": cl + cs,
                         "ban": tr.ban, "entry": tr.entry_dates}, index=tr.dates)


def all_subsets(tr, **kw):
    """32개 부분집합의 배치별 순수익. 열 번호 = 비트마스크."""
    return pd.DataFrame({m: run(tr, subset(m), **kw)["ret"] for m in range(N_SUB)})


# ---------------------------------------------------------------- 성과·샤플리
def sharpe(x, ppy):
    x = np.asarray(x, float)
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return x.mean() / sd * math.sqrt(ppy) if sd and sd > 0 else np.nan


def perf(r, ppy):
    r = np.asarray(r, float)
    n, m = len(r), r.mean()
    sd = r.std(ddof=1) if n > 1 else np.nan
    eq = np.cumprod(1 + r)
    return dict(n=n, mean=m, ann_ret=m * ppy, ann_vol=sd * math.sqrt(ppy), sharpe=m / sd * math.sqrt(ppy),
                t=m / sd * math.sqrt(n), mdd=float((eq / np.maximum.accumulate(eq) - 1).min()), hit=float((r > 0).mean()))


def shapley_weights(n=len(CONS)):
    """phi = W @ v (v는 길이 2^n 벡터, 인덱스 = 비트마스크). 순서 의존을 평균으로 제거한 샤플리 값."""
    f = math.factorial
    W = np.zeros((n, 1 << n))
    for m in range(1 << n):
        s = bin(m).count("1")
        for i in range(n):
            W[i, m] = f(s - 1) * f(n - s) / f(n) if m >> i & 1 else -f(s) * f(n - s - 1) / f(n)
    return W


W5 = shapley_weights()


def shapley(v):
    return W5 @ np.asarray(v, float)


def check_efficiency(v, tol=1e-12):
    """샤플리 값의 합 == v(전부) - v(공집합)을 검증한다. 어긋나면 AssertionError."""
    v = np.asarray(v, float)
    gap = abs(shapley(v).sum() - (v[-1] - v[0]))
    assert gap < tol * max(1.0, abs(v).max()), f"샤플리 효율성 위반: {gap}"
    return gap


def marginals(v):
    """제약별 한계기여 v(S+i)-v(S)의 분포(S는 i를 뺀 16개 맥락): first=S가 공집합, last=S가 나머지 전부."""
    v = np.asarray(v, float)
    out = {}
    for i, c in enumerate(CONS):
        ms = np.array([v[m | 1 << i] - v[m] for m in range(N_SUB) if not m >> i & 1])
        out[c] = dict(first=v[1 << i] - v[0], last=v[-1] - v[(N_SUB - 1) ^ (1 << i)], min=ms.min(), max=ms.max())
    return out


def block_boot(R, ppy, B=500, block=4, seed=0):
    """R (T x 32): 주 단위 순환 블록 부트스트랩(블록 4주). 모든 부분집합에 같은 표본 인덱스를 써서 짝지어 재표본한다.
    반환: 평균 (B x 32), 샤프 (B x 32)."""
    R = np.asarray(R, float)
    T = len(R)
    rng = np.random.default_rng(seed)
    nb = -(-T // block)
    mean_b, sh_b = np.empty((B, N_SUB)), np.empty((B, N_SUB))
    for b in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)) % T).ravel()[:T]
        Rb = R[idx]
        mean_b[b] = Rb.mean(0)
        sh_b[b] = mean_b[b] / Rb.std(0, ddof=1) * math.sqrt(ppy)
    return mean_b, sh_b


PANEL = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS/data/daily/club_db/panel.parquet"
MKT_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "is_halted", "ret_outlier", "mktcap",
            "lim_up", "lim_dn", "limit_pct", "sell_tax_pct", "tick_pct", "short_banned"]


def load_panel(start, end=None, extra=()):
    """panel.parquet에서 날짜 범위와 필요한 열만 읽는다(메모리 절약)."""
    f = [("date", ">=", pd.Timestamp(start))] + ([("date", "<=", pd.Timestamp(end))] if end else [])
    return pd.read_parquet(PANEL, columns=MKT_COLS + list(extra), filters=f)


# ---------------------------------------------------------------- 신호 (대조용 단순 지표)
SIGN = {"str5": -1, "str20": -1, "near52h": 1, "abvol": 1, "vol20": -1, "mom12_1": 1}   # 이론상 부호


def rebalance_dates(cal, start, step=5):
    """start 이후 첫 거래일부터 step 거래일마다."""
    cal = pd.DatetimeIndex(cal)
    return cal[cal >= pd.Timestamp(start)][::step]


def signal_rankavg(panel, dates):
    """6개 지표의 날짜별 횡단면 순위(0~1)를 이론 부호로 평균. 6개 모두 있는 종목만. 그날 종가까지 정보만 쓴다."""
    d = panel[panel["date"].isin(dates)][["date", "ticker", *SIGN]].dropna()
    r = d.groupby("date")[list(SIGN)].rank(pct=True)
    score = sum(sg * r[k] for k, sg in SIGN.items()) / len(SIGN)
    return pd.DataFrame({"date": d["date"], "ticker": d["ticker"], "score": score})


def signal_rev5(panel, dates):
    """5일 반전: -str5."""
    d = panel[panel["date"].isin(dates)][["date", "ticker", "str5"]].dropna()
    return pd.DataFrame({"date": d["date"], "ticker": d["ticker"], "score": -d["str5"]})

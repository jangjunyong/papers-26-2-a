"""동아리 DB(KOSPI 2010~2026) + 제도 자료를 합쳐 분석용 일별 패널을 만든다.
입력: PAPERS/data/daily/club_db/{price,investor,trading}[_2010_2014].parquet, PAPERS/data/제도일정/*.csv
출력: PAPERS/data/daily/club_db/panel.parquet (종목-일 단위, 2010~2026)
규칙: 모든 지표는 해당 일 종가까지의 정보만 쓴다(룩어헤드 없음). 순매수 정규화는 '직전 60거래일 평균 거래량'(shift 1)."""
import os

import numpy as np
import pandas as pd

D = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS/data"
CB = f"{D}/daily/club_db"
IN = f"{D}/제도일정"


def load(name):
    parts = []
    for suf in ("_2010_2014", ""):
        p = f"{CB}/{name}{suf}.parquet"
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"])
    return df


price = load("price")
inv = load("investor")
trd = load("trading")[["code", "date", "trading_value"]]
df = price.merge(inv, on=["code", "date"], how="left").merge(trd, on=["code", "date"], how="left")
df["ticker"] = df.code.str[1:]
df = df.sort_values(["code", "date"]).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["is_halted"] = df["is_halted"].fillna(0).astype(int)
g = df.groupby("code", sort=False)

# --- 수익률(수정주가 기준). 거래정지일은 수익률을 비운다(가짜 가격 방지)
df["ret1d"] = g["adj_close"].pct_change()
df.loc[(df.is_halted == 1) | (df.volume == 0), "ret1d"] = np.nan
# 수정주가 이상값: 가격제한폭(±15%/±30%)보다 10%p 넘게 움직이면 수정 오류·재상장 등으로 보고 표시한다(원 수익률은 보존)
_L = np.where(df.date < "2015-06-15", 0.15, 0.30)
df["ret_outlier"] = (df.ret1d.abs() > _L + 0.10).astype(int)
df["ret1d_clean"] = df.ret1d.where(df.ret_outlier == 0)
df["mktcap"] = df.close * df.listed_shares
df["tv_filled"] = df.trading_value.fillna(df.close * df.volume)   # 2010 이전 등 결측은 종가×거래량 근사
df["tv_is_approx"] = df.trading_value.isna().astype(int)

# --- 투자자 순매수 정규화: 직전 60거래일 평균 거래량(당일 제외)으로 나눈 값
avgvol60 = g["volume"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean())
for c in ("indiv", "inst", "foreign"):
    df[f"{c}_flow"] = df[f"{c}_net"] / avgvol60
avgtv60 = df.groupby("code", sort=False)["tv_filled"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean())
df["indiv_flow_tv"] = df["indiv_net"] * df["close"] / avgtv60      # 거래대금 기준 정규화(근사)

# --- 가격제한폭: 2015-06-15 전 ±15%, 이후 ±30%. 상하한가는 원주가 수익률로 근사 판정
pl = pd.read_csv(f"{IN}/price_limit.csv")
df["limit_pct"] = np.where(df.date < "2015-06-15", 15.0, 30.0)
prev_close = g["close"].shift(1)
raw_ret = df.close / prev_close - 1
L = df.limit_pct / 100
corp_action = (raw_ret.abs() > L + 0.05)                            # 액면분할 등으로 원주가 단절
df["lim_up"] = ((raw_ret >= L - 0.004) & ~corp_action & (df.volume > 0)).astype(int)
df["lim_dn"] = ((raw_ret <= -L + 0.004) & ~corp_action & (df.volume > 0)).astype(int)

# --- 공매도 구간(전면금지 여부) / 매도 측 세율(KOSPI 합계 %)
ss = pd.read_csv(f"{IN}/short_selling_regime.csv", parse_dates=["start_date", "end_date"])
df["short_regime"] = "전면허용"
df.loc[(df.date >= "2011-08-10") & (df.date <= "2011-11-09"), "short_regime"] = "전면금지"   # 기획안·금융위(2011 금지)
for r in ss.itertuples():
    m = (df.date >= r.start_date) & (df.date <= r.end_date)
    df.loc[m, "short_regime"] = r.regime
df["short_banned"] = (df.short_regime == "전면금지").astype(int)
tx = pd.read_csv(f"{IN}/transaction_tax.csv", parse_dates=["effective_from", "effective_to"])
tx = tx[tx.market == "KOSPI"]
df["sell_tax_pct"] = 0.30                                           # 2015 이전은 0.30% 가정(미확인, 2013-08 이후 시행령 유지분 외삽)
df["sell_tax_assumed"] = (df.date < "2015-01-01").astype(int)
for r in tx.itertuples():
    m = (df.date >= r.effective_from) & (df.date <= r.effective_to)
    df.loc[m, "sell_tax_pct"] = r.total_sell_side_pct
    df.loc[m, "sell_tax_assumed"] = 0

# --- 호가단위(KOSPI) → 체결가 한 틱 비율(스프레드 하한 근사)
tk = pd.read_csv(f"{IN}/tick_size.csv", parse_dates=["effective_from", "effective_to"])
tk = tk[tk.market == "KOSPI"]
df["tick"] = np.nan
for r in tk.itertuples():
    m = (df.date >= r.effective_from) & ((df.date <= r.effective_to) if pd.notna(r.effective_to) else True)
    m &= df.close >= r.price_from
    if pd.notna(r.price_to):
        m &= df.close < r.price_to
    df.loc[m, "tick"] = r.tick_size
df["tick"] = df["tick"].fillna(1.0)
df["tick_pct"] = df.tick / df.close * 100

# --- 기준 지표(신호용, 그날 종가까지). 창 안에 수익률 이상값이 있으면 비운다
_bad = df.groupby("code", sort=False)["ret_outlier"]
df["str5"] = g["adj_close"].pct_change(5).where(_bad.transform(lambda s: s.rolling(5, min_periods=1).max()) == 0)
df["str20"] = g["adj_close"].pct_change(20).where(_bad.transform(lambda s: s.rolling(20, min_periods=1).max()) == 0)
df["mom12_1"] = (g["adj_close"].shift(21) / g["adj_close"].shift(252) - 1).where(
    _bad.transform(lambda s: s.shift(21).rolling(231, min_periods=1).max()) == 0)
hi52 = g["adj_close"].transform(lambda s: s.rolling(252, min_periods=120).max())
df["near52h"] = df.adj_close / hi52
df["vol20"] = g["ret1d"].transform(lambda s: s.rolling(20, min_periods=15).std())
tv5 = df.groupby("code", sort=False)["tv_filled"].transform(lambda s: s.rolling(5, min_periods=3).mean())
df["abvol"] = tv5 / avgtv60
df["hl_range"] = (df.high - df.low) / df.close

for c in ("indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv", "abvol", "str5", "str20", "mom12_1", "near52h"):
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)       # 거래정지로 평균 거래량이 0인 날 등

for c in ("indiv_flow", "inst_flow", "foreign_flow"):        # 거래량 극소 종목의 극단값을 1%/99%로 자른 열(분석용)
    lo, hi = df[c].quantile([0.005, 0.995])
    df[c + "_w"] = df[c].clip(lo, hi)

out = f"{CB}/panel.parquet"
df.drop(columns=["code"]).to_parquet(out, index=False)
print("panel", df.shape, df.date.min().date(), "~", df.date.max().date(), "종목", df.ticker.nunique())
print("상한가 비율 %.4f%% 하한가 비율 %.4f%%" % (df.lim_up.mean() * 100, df.lim_dn.mean() * 100))
print("공매도 구간 일수:", df.groupby("short_regime").date.nunique().to_dict())
print("거래대금 근사 비율 %.3f" % df.tv_is_approx.mean(), "| 투자자 결측 %.4f" % df.indiv_net.isna().mean())
print("수정주가 이상값 행: %d (%.4f%%), 종목 %d" % (df.ret_outlier.sum(), df.ret_outlier.mean() * 100, df.loc[df.ret_outlier == 1, "ticker"].nunique()))
print(df[["ret1d_clean", "mktcap", "indiv_flow", "foreign_flow", "str5", "abvol", "tick_pct"]].describe().round(4).T.to_string())

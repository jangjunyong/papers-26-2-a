"""p02: 동아리 DB(KOSPI 2010~2026) + 제도 자료 → 분석용 일별 패널 v2 (q01 감사 패치 P1~P7 + 지침 반영).
p01_build_panel.py 의 로직은 그대로 두고 감사에서 나온 오류만 고쳤다. p01·panel.parquet 은 건드리지 않는다.
입력: PAPERS/data/daily/club_db/{price,investor,trading}[_2010_2014].parquet, PAPERS/data/제도일정/{tick_size,short_selling_regime,transaction_tax}.csv
출력: PAPERS/data/daily/club_db/panel_v2.parquet (종목-일), panel_v2_removed_rows.parquet (P4 로 삭제한 행 목록)
규칙: 특징 열은 해당 일 종가까지의 정보만 쓴다(룩어헤드 없음). 예외(의도적으로 사후 정보를 쓰는 열):
  fwd_ret5·fwd_censored(라벨), liq_zone(상폐 여부를 아는 사후 표시), exclude_flag(자료 끝 기준), P4 삭제(상장 전 행 정리).
실행: py -3 p02_build_panel_v2.py   (노트북 CPU 약 40초, 최대 메모리 약 4GB)
build_panel() 은 입력 DataFrame 3개를 받는 함수라 q02_panel_v2_qa.py 의 룩어헤드 실험이 그대로 불러다 쓴다."""
import os
import time

import numpy as np
import pandas as pd

D = "C:/Users/jacob/Documents/Obsidian Vault/PAPERS/data"
CB = f"{D}/daily/club_db"
IN = f"{D}/제도일정"
OUT = f"{CB}/panel_v2.parquet"
OUT_REMOVED = f"{CB}/panel_v2_removed_rows.parquet"

# ---- 상수 (감사 q01_panel_qa.md 근거)
OLD_TICK = [(0, 5000, 5), (5000, 10000, 10), (10000, 50000, 50), (50000, 100000, 100), (100000, 500000, 500), (500000, 1e18, 1000)]  # ~2010-10-03 구호가표, market_rules_notes F12 [2차]
TICK_NEW_FROM = "2010-10-04"
FOREIGN_DEF_FROM = "2026-05-26"          # 키움 수급 전환일: foreign_net 이 기타외국인을 제외한 정의로 바뀐 것으로 추정
INV_SUM_OPEN_END = "2012-05-02"          # 이 날까지 4그룹 합이 0 으로 닫히지 않음
ZERO_INV_MIN_VOL = 10000                 # 4그룹 순매수가 모두 0 이어도 거래량이 이만큼 있으면 결측(0 코딩)으로 본다
DELIST_GRACE_DAYS = 10                   # 마지막 행이 자료 끝보다 이 일수 넘게 앞서면 상장폐지로 본다
LIQ_ROWS = 8                             # 정리매매 구간: 상폐 종목 마지막 8행(정리매매 7거래일 + 재개일)
PENNY_ADJ = 100                          # 수정주가 이 값 이하는 원 단위 반올림 잡음이 커서 이상값 처리
EXCL_PENNY_TICKERS = ("008080", "000790")  # 수정주가 1~4원/종가 10~115원 아티팩트 종목: 전 기간 제외
EXCL_6740_FROM = "2026-06-26"            # 006740: 분할 재개 후 종가와 거래량·거래대금 단위 불일치
N_LAST_DAYS = 3                          # 자료 끝 3거래일은 미확정


def t_log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


T0 = time.time()


def load(name):
    parts = []
    for suf in ("_2010_2014", ""):
        p = f"{CB}/{name}{suf}.parquet"
        if os.path.exists(p):
            parts.append(pd.read_parquet(p))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"])


# ------------------------------------------------------------------ 호가단위·상하한가 (P3, P7)
def make_tick_eras(tk):
    """tick_size.csv(KOSPI) + 구호가표 → [(start, end, edges, ticks)]. 2010-10-04 이전은 구호가표(2차 자료)."""
    eras = []
    first = tk.effective_from.min()
    eras.append((pd.Timestamp("1900-01-01"), first - pd.Timedelta(days=1),
                 np.array([b[0] for b in OLD_TICK], float), np.array([b[2] for b in OLD_TICK], float)))
    for a in sorted(tk.effective_from.unique()):
        d = tk[tk.effective_from == a].sort_values("price_from")
        b = d.effective_to.iloc[0]
        eras.append((pd.Timestamp(a), pd.Timestamp("2100-01-01") if pd.isna(b) else pd.Timestamp(b), d.price_from.values.astype(float), d.tick_size.values.astype(float)))
    return eras


def tick_of(price, dates, eras):
    price = np.asarray(price, float)
    dates = pd.DatetimeIndex(dates)
    out = np.full(len(price), np.nan)
    for s, e, edges, ticks in eras:
        m = (dates >= s) & (dates <= e)
        idx = np.searchsorted(edges, price[m], side="right") - 1
        out[m] = ticks[np.clip(idx, 0, len(ticks) - 1)]
    return out


def exact_limits(prev, dates, lim_pct, eras):
    """KRX 산식(q01 exact_limits 와 같은 규칙): 기준가×한도율을 기준가 호가단위로 절사 → 가감 → 결과 가격대 호가단위로 절사(상한)/절상(하한).
    가격·호가단위가 정수 원이라 정수 연산으로 계산한다(부동소수 오차 없음)."""
    prev = np.asarray(prev, np.int64)
    Lp = np.asarray(lim_pct, np.int64)
    t0 = tick_of(prev, dates, eras).astype(np.int64)
    step = (prev * Lp // (100 * t0)) * t0
    up = prev + step
    tu = tick_of(up, dates, eras).astype(np.int64)
    up = (up // tu) * tu
    dn = prev - step
    td = tick_of(np.maximum(dn, 1), dates, eras).astype(np.int64)
    dn = -((-dn) // td) * td
    return up, dn


# ------------------------------------------------------------------ 본체
def build_panel(price, inv, trd, verbose=False):
    log = t_log if verbose else (lambda *a: None)
    df = price.merge(inv, on=["code", "date"], how="left").merge(trd[["code", "date", "trading_value"]], on=["code", "date"], how="left")
    df["ticker"] = df.code.str[1:]
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["is_halted"] = df["is_halted"].fillna(0).astype(int)

    # --- P4. 상장 전 K-OTC(장외) 구간 삭제: 투자자 자료가 시작하기 전 선두 구간이 20행 이상인 종목의 그 구간
    first_inv = df.loc[df.indiv_net.notna()].groupby("code").date.min().rename("first_inv")
    df = df.merge(first_inv, on="code", how="left")
    lead = df.date < df.first_inv
    kotc = lead & (lead.groupby(df.code).transform("sum") >= 20)
    removed = df.loc[kotc, ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].assign(reason="prelist_otc").reset_index(drop=True)
    df = df.loc[~kotc].drop(columns="first_inv").reset_index(drop=True)
    log("P4 삭제", len(removed), "행", removed.ticker.nunique(), "종목")

    # --- P5. 투자자 열: 결측이 0 으로 채워진 행 / 순매수>거래량 행 → NaN, 정의·합계 구간 플래그
    inv_cols = ["indiv_buy", "indiv_sell", "indiv_net", "inst_buy", "inst_net", "foreign_buy", "foreign_net", "other_corp_net"]
    nets4 = ["indiv_net", "inst_net", "foreign_net", "other_corp_net"]
    z = (df[nets4] == 0).all(axis=1) & (df.volume >= ZERO_INV_MIN_VOL)
    over = df[["indiv_net", "inst_net", "foreign_net"]].abs().gt(df.volume * 1.0001, axis=0).any(axis=1)
    df["f_zero_inv_suspect"] = z.astype("int8")
    df["f_net_gt_volume"] = over.astype("int8")
    df.loc[z | over, inv_cols] = np.nan
    df["foreign_def_changed"] = (df.date >= FOREIGN_DEF_FROM).astype("int8")
    df["inv_sum_closed"] = ((df.date > INV_SUM_OPEN_END) & (df.date < FOREIGN_DEF_FROM)).astype("int8")

    g = df.groupby("code", sort=False)
    idle = (df.is_halted == 1) | (df.volume == 0)
    active = (df.volume > 0) & (df.is_halted == 0)
    first_row = df.code.ne(df.code.shift())
    prev_idle = idle.shift(1, fill_value=False) & ~first_row
    df["f_tradable"] = active.astype("int8")

    # --- P1. 거래재개일 누적 수익 분리 (정지·거래량0 일은 수익 NaN, 재개일 수익은 ret_multi 로 옮기고 ret1d 는 NaN)
    ret_raw = g["adj_close"].pct_change(fill_method=None).mask(idle)
    df["resume_day"] = (prev_idle & active).astype("int8")
    df["ret_multi"] = ret_raw.where(df.resume_day == 1)

    # --- P2. 정리매매 구간(상폐 종목 마지막 8행)은 이상값에서 제외, 극저가(수정주가<=100)는 따로 비운다
    rank_end = g.cumcount(ascending=False)
    delisted = g["date"].transform("max") < df.date.max() - pd.Timedelta(days=DELIST_GRACE_DAYS)
    df["liq_zone"] = (delisted & (rank_end < LIQ_ROWS)).astype("int8")
    penny = (df.adj_close <= PENNY_ADJ) | (g["adj_close"].shift(1) <= PENNY_ADJ)
    df["f_penny"] = penny.astype("int8")                               # 극저가 표시(정리매매 구간 포함, 값은 그대로)
    # 수정주가 이상값: 가격제한폭보다 10%p 넘게 움직임(재개일은 누적 수익 ret_multi 기준으로 판정해 str/mom 창을 보호)
    _L = np.where(df.date < "2015-06-15", 0.15, 0.30)
    # 극저가 반올림 잡음은 정리매매 구간의 실제 호가(adj_close == close, 전일도 동일)에는 없다 → 감사 패치와 달리 그 행은 극저가여도 비우지 않는다(실제 손실 보존)
    real_px = (df.adj_close == df.close) & (g["adj_close"].shift(1) == g["close"].shift(1))
    penny_eff = penny & ~((df.liq_zone == 1) & real_px)
    df["ret_outlier"] = (((ret_raw.abs() > _L + 0.10) & (df.liq_zone == 0)) | penny_eff).astype("int8")
    df["ret1d"] = ret_raw.mask(df.resume_day == 1)
    df["ret1d_clean"] = df.ret1d.where(df.ret_outlier == 0)
    df["mktcap"] = df.close * df.listed_shares
    df["mktcap_missing"] = df.mktcap.isna().astype("int8")      # 2026-06-23 이후 listed_shares 공백: 표시만
    df["tv_filled"] = df.trading_value.fillna(df.close * df.volume)
    df["tv_is_approx"] = df.trading_value.isna().astype(int)
    log("P1·P2 완료")

    # --- P6. 순매수 정규화: 분할 보정 + 정지일 제외 60일 평균(당일 제외), 분모 = max(평균, 당일 거래량) → [-1, 1]
    fac = df.adj_close / df.close                                   # 과거→현재 비율만 쓰므로 미래 조정분은 상쇄된다
    va = (df.volume / fac).where(df.volume > 0)
    av = va.groupby(df.code, sort=False).transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean()) * fac
    den = np.maximum(av, df.volume)
    for c in ("indiv", "inst", "foreign"):
        df[f"{c}_flow"] = (df[f"{c}_net"] / den).where(active)     # 정지일은 0 이 아니라 NaN
    avgtv60 = df.groupby("code", sort=False)["tv_filled"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean())
    avgtv60x = df.tv_filled.where(active).groupby(df.code, sort=False).transform(lambda s: s.shift(1).rolling(60, min_periods=30).mean())
    df["indiv_flow_tv"] = (df.indiv_net * df.close / np.maximum(avgtv60x, df.close * df.volume)).where(active)   # 같은 방식으로 [-1,1]
    for c in ("indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv"):
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    for c in ("indiv", "inst", "foreign"):                          # _w(전체 표본 분위수 클립) 대신 날짜별 횡단면 순위
        df[f"{c}_rk"] = df[f"{c}_flow"].groupby(df.date).rank(pct=True) - 0.5
    log("P5·P6 완료")

    # --- 가격제한폭 / 상하한가 (P7)
    tk = pd.read_csv(f"{IN}/tick_size.csv", parse_dates=["effective_from", "effective_to"])
    eras = make_tick_eras(tk[tk.market == "KOSPI"])
    df["limit_pct"] = np.where(df.date < "2015-06-15", 15.0, 30.0)
    prev_close = g["close"].shift(1)
    facl = fac.groupby(df.code, sort=False)
    dfac = fac / facl.shift(1) - 1
    elig = (prev_close.notna() & (g["volume"].shift(1) > 0) & (g["is_halted"].shift(1) == 0) & active & (dfac.abs() < 0.01) & (df.liq_zone == 0))
    up, dn = exact_limits(prev_close.fillna(1).values, df.date, df.limit_pct.values, eras)
    df["lim_elig"] = elig.astype("int8")                            # 0 이면 상하한가 판정 불가(재개일·권리락/분할일·정리매매·첫날)
    df["lim_up"] = ((df.close.values >= up) & elig.values).astype("int8")
    df["lim_dn"] = ((df.close.values <= dn) & elig.values).astype("int8")

    # --- 공매도 구간 / 매도 측 세율 (p01 과 동일)
    ss = pd.read_csv(f"{IN}/short_selling_regime.csv", parse_dates=["start_date", "end_date"])
    df["short_regime"] = "전면허용"
    df.loc[(df.date >= "2011-08-10") & (df.date <= "2011-11-09"), "short_regime"] = "전면금지"
    for r in ss.itertuples():
        m = (df.date >= r.start_date) & (df.date <= r.end_date)
        df.loc[m, "short_regime"] = r.regime
    df["short_banned"] = (df.short_regime == "전면금지").astype(int)
    tx = pd.read_csv(f"{IN}/transaction_tax.csv", parse_dates=["effective_from", "effective_to"])
    tx = tx[tx.market == "KOSPI"]
    df["sell_tax_pct"] = 0.30
    df["sell_tax_assumed"] = (df.date < "2015-01-01").astype(int)
    for r in tx.itertuples():
        m = (df.date >= r.effective_from) & (df.date <= r.effective_to)
        df.loc[m, "sell_tax_pct"] = r.total_sell_side_pct
        df.loc[m, "sell_tax_assumed"] = 0

    # --- P3. 호가단위: 2010-10-04 이전은 구호가표(tick_assumed=1)
    df["tick"] = tick_of(df.close.values, df.date, eras)
    n_tick_na = int(df.tick.isna().sum())
    df["tick"] = df["tick"].fillna(1.0)
    df["tick_assumed"] = (df.date < TICK_NEW_FROM).astype("int8")
    df["tick_pct"] = df.tick / df.close * 100
    assert n_tick_na == 0, n_tick_na

    # --- 기준 지표(신호용, 그날 종가까지). 창 안에 수익률 이상값이 있으면 비운다
    _bad = df.groupby("code", sort=False)["ret_outlier"]
    df["str5"] = g["adj_close"].pct_change(5, fill_method=None).where(_bad.transform(lambda s: s.rolling(5, min_periods=1).max()) == 0)
    df["str20"] = g["adj_close"].pct_change(20, fill_method=None).where(_bad.transform(lambda s: s.rolling(20, min_periods=1).max()) == 0)
    df["mom12_1"] = (g["adj_close"].shift(21) / g["adj_close"].shift(252) - 1).where(
        _bad.transform(lambda s: s.shift(21).rolling(231, min_periods=1).max()) == 0)
    hi52 = g["adj_close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    df["near52h"] = df.adj_close / hi52
    df["vol20"] = df.groupby("code", sort=False)["ret1d_clean"].transform(lambda s: s.rolling(20, min_periods=15).std())   # 정리된 수익으로 계산
    tv5 = df.groupby("code", sort=False)["tv_filled"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["abvol"] = tv5 / avgtv60
    df["hl_range"] = (df.high - df.low) / df.close
    for c in ("abvol", "str5", "str20", "mom12_1", "near52h"):
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    log("지표 완료")

    # --- 사용 제외 표시 (비트마스크: 1=자료 끝 3거래일, 2=008080·000790 전 기간, 4=006740 2026-06-26~)
    last_days = np.sort(df.date.unique())[-N_LAST_DAYS:]
    reason = (df.date.isin(last_days).astype(int) * 1 + df.ticker.isin(EXCL_PENNY_TICKERS).astype(int) * 2
              + ((df.ticker == "006740") & (df.date >= EXCL_6740_FROM)).astype(int) * 4)
    df["exclude_reason"] = reason.astype("int8")
    df["exclude_flag"] = (reason > 0).astype("int8")

    # --- 라벨: 5거래일 뒤 수정주가 수익률. 상폐 종목은 마지막 가격까지(정리매매 손실 포함)
    ahead = rank_end                                                # 이 행 뒤에 남은 행 수
    nxt5 = g["adj_close"].shift(-5)
    last_adj = g["adj_close"].transform("last")
    fwd = np.where(ahead >= 5, nxt5 / df.adj_close - 1, np.where(delisted & (ahead >= 1), last_adj / df.adj_close - 1, np.nan))
    df["fwd_ret5"] = fwd
    df["fwd_censored"] = np.where(ahead >= 5, 0, np.where(delisted, 1, 2)).astype("int8")   # 0 완전 / 1 상폐로 잘림 / 2 자료 끝(NaN)

    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "listed_shares", "is_halted", "indiv_buy", "indiv_sell", "indiv_net", "inst_buy", "inst_net",
            "foreign_buy", "foreign_net", "other_corp_net", "trading_value", "ticker",
            "ret1d", "ret_outlier", "ret1d_clean", "mktcap", "tv_filled", "tv_is_approx", "indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv",
            "limit_pct", "lim_up", "lim_dn", "short_regime", "short_banned", "sell_tax_pct", "sell_tax_assumed", "tick", "tick_pct",
            "str5", "str20", "mom12_1", "near52h", "vol20", "abvol", "hl_range",
            "resume_day", "ret_multi", "liq_zone", "f_penny", "indiv_rk", "inst_rk", "foreign_rk", "tick_assumed", "f_zero_inv_suspect", "f_net_gt_volume",
            "foreign_def_changed", "inv_sum_closed", "lim_elig", "f_tradable", "exclude_flag", "exclude_reason", "mktcap_missing", "fwd_ret5", "fwd_censored"]
    return df[cols], removed


def main():
    price, inv, trd = load("price"), load("investor"), load("trading")[["code", "date", "trading_value"]]
    t_log("입력", len(price), len(inv), len(trd))
    df, removed = build_panel(price, inv, trd, verbose=True)
    df.to_parquet(OUT, index=False)
    removed.to_parquet(OUT_REMOVED, index=False)
    t_log("panel_v2", df.shape, df.date.min().date(), "~", df.date.max().date(), "종목", df.ticker.nunique(), "→", OUT)
    print("삭제 행", len(removed), removed.groupby("ticker").size().to_dict())
    print("재개일 %d | 정리매매 구간 %d | 극저가 %d | ret_outlier %d | 상한가 %d 하한가 %d | exclude_flag %d" % (
        df.resume_day.sum(), df.liq_zone.sum(), df.f_penny.sum(), df.ret_outlier.sum(), df.lim_up.sum(), df.lim_dn.sum(), df.exclude_flag.sum()))
    print("투자자 0코딩 NaN화 %d | 순매수>거래량 NaN화 %d" % (df.f_zero_inv_suspect.sum(), df.f_net_gt_volume.sum()))
    fl = df[["indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv"]]
    print("flow 절대값 최댓값", fl.abs().max().round(4).to_dict())
    print(df[["ret1d_clean", "ret_multi", "indiv_flow", "indiv_rk", "str5", "vol20", "tick_pct", "fwd_ret5"]].describe().round(4).T.to_string())


if __name__ == "__main__":
    main()

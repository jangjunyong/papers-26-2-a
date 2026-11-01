# -*- coding: utf-8 -*-
"""r02 RQ2 예비 분석 재실행: 패널 독립 감사(q01) 지적을 단계적으로 반영하고 r01 결론이 바뀌는지 본다.

r01_rq2_pilot.py 의 이벤트 정의·대조군 매칭·CAR·회귀 함수를 import 해 그대로 쓴다(r01 파일은 수정하지 않음).
출력: results/r02_tables/*.csv, r02_log.txt  (보고서는 results/r02_audit_rerun_memo.md)
실행: python r02_rq2_audit_rerun.py   (노트북 CPU 수 분 이내)

================= 단계 정의 (코드 실행 전에 고정. 결과를 보고 바꾸지 않는다) =================
공통: 신호 3종(pred_d_lr1e-4=cnn_d, pred_d_lr1e-4_tb=cnn_tb, pred_w_lr1e-4=cnn_w), 이벤트·대조군·CAR·회귀 상수는 r01 그대로.
      이벤트 플래그(신규 편입)는 신호만으로 정한다 -> 모든 단계에서 동일. 판정 임계는 r01과 같은 |t|>1.96.
S0  r01 원래 설정(재현 확인용).
S1  S0 + 투자자 열(개인·외국인·기관 순매수 흐름, 거래대금 척도 흐름, 개인 거래 비중의 매수·매도·거래량)에서
    f_zero_inv_suspect==1 또는 f_net_gt_volume==1 인 종목-일을 결측(NaN) 처리.
    주간 합 규칙(사전 선택): 합산 창(형성 주 -4~0, 사후 주 1~5, 6~20) 안에 결측일이 하나라도 있으면 그 합은 NaN
    (가용일 평균으로 메우지 않는다). S1 이상의 모든 흐름 합에 적용. CAR의 80% 유효일 규칙은 r01 그대로.
S2  S1 + (a) f_resume_day==1 인 날의 수익률을 NaN(CAR·규모 5분위 벤치마크 모두에서 제외, 80% 규칙으로 창 처리),
          (b) f_prelist_otc==1 인 행은 수익·흐름·거래 비중 입력에서 NaN,
          (c) 형성일 행이 f_tradable==1 & f_prelist_otc==0 인 종목-주만 표본(이벤트·대조 후보).
S3  S2 + 순매수 흐름을 *_flow_w(전체 표본 분위수 윈저화) 대신 원 *_flow 의 날짜별 횡단면 순위
    (groupby('date').rank(pct=True) - 0.5, 마스크 적용 후)로 교체. 거래대금 척도도 같은 방식(원 indiv_flow_tv).
    주간 합은 순위의 일별 합(x100). 수준 단위가 S0~S2와 달라 t값·부호로 비교한다.
S4  S3 를 형성일 >= 2012-05-03 으로만 제한(4그룹 합이 닫히는 구간). 형성일 31개. 표본이 작다.
보조(판정에 쓰지 않음, 단계 정의 밖):
  S3a  S3 를 형성일 < 2012-05-03 으로 제한. S4 와 같은 규격의 보완 구간(S4 가 작은 표본 탓인지 가르기 위함).
  S1w  S0 + 4그룹 순매수가 모두 0이고 거래량>0 인 행 전부(거래량 1만 미만 포함)를 결측 처리(플래그 문턱 민감도).
===================================================================================================
"""
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import r01_rq2_pilot as r01

ROOT = r01.ROOT
RES = f"{ROOT}/26-2/03_차트CNN_연구/results"
FLAGS = f"{RES}/q01_panel_flags.parquet"
R01_OUT = r01.OUT
OUT = f"{RES}/r02_tables"
CUT = pd.Timestamp("2012-05-03")
SIGS = ["cnn_d", "cnn_tb", "cnn_w"]
SCALES = ["vol", "tv"]
# W 키 -> 패널 원 열 (윈저화 전). S3 에서 순위 변환 입력으로 쓴다.
RAWK = {"indiv_flow_w": "indiv_flow", "foreign_flow_w": "foreign_flow", "inst_flow_w": "inst_flow",
        "indiv_flow_tv": "indiv_flow_tv"}
STAGES = {  # inv: 투자자 결측 규칙, s2: 재개일·OTC·거래가능 유니버스, rank: 순위 변환, cut: 형성일 제한
    "S0": dict(inv=None, s2=False, rank=False, cut=None),
    "S1": dict(inv="flag", s2=False, rank=False, cut=None),
    "S2": dict(inv="flag", s2=True, rank=False, cut=None),
    "S3": dict(inv="flag", s2=True, rank=True, cut=None),
    "S4": dict(inv="flag", s2=True, rank=True, cut="ge"),
    "S3a": dict(inv="flag", s2=True, rank=True, cut="lt"),
    "S1w": dict(inv="wide", s2=False, rank=False, cut=None),
}
MAIN_STAGES = ["S0", "S1", "S2", "S3", "S4"]
LOG = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def wsum_strict(M, r0, r1, j):
    """창 안에 결측이 하나라도 있으면 NaN (S1 이상). r01.wsum 대신 끼워 넣는다."""
    blk = M[r0:r1 + 1, j]
    s = blk.sum(0)
    s[np.isfinite(blk).sum(0) < (r1 - r0 + 1)] = np.nan
    return s


# ---------------------------------------------------------------- 데이터
def to_grid(pn, col, ii, jj, shape):
    M = np.full(shape, np.nan)
    M[ii, jj] = pn[col].values.astype(float)
    return M


def load_extra(dates, tick):
    cols = ["ticker", "date", "volume", "indiv_buy", "indiv_sell", "indiv_net", "inst_net", "foreign_net",
            "other_corp_net", "indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv"]
    pn = pd.read_parquet(r01.PANEL, columns=cols)
    pn = pn[pn.date <= r01.PANEL_LAST].reset_index(drop=True)
    fl = pd.read_parquet(FLAGS)
    fl = fl[fl.date <= r01.PANEL_LAST]
    n0 = len(pn)
    pn = pn.merge(fl, on=["ticker", "date"], how="left", validate="1:1")
    assert len(pn) == n0 and pn.f_tradable.notna().all(), "플래그 조인 실패"
    ii = dates.get_indexer(pn.date)
    jj = pd.Index(tick).get_indexer(pn.ticker)
    assert (ii >= 0).all() and (jj >= 0).all()
    shape = (len(dates), len(tick))
    X = {}
    for c in ["volume", "indiv_buy", "indiv_sell", "indiv_flow", "inst_flow", "foreign_flow", "indiv_flow_tv"]:
        X[c] = to_grid(pn, c, ii, jj, shape)
        if c.endswith("flow") or c.endswith("flow_tv"):
            X[c][~np.isfinite(X[c])] = np.nan
    fb = lambda c: to_grid(pn, c, ii, jj, shape) == 1
    X["bad_flag"] = fb("f_zero_inv_suspect") | fb("f_net_gt_volume")
    X["zero_inv_suspect"], X["net_gt_vol"] = fb("f_zero_inv_suspect"), fb("f_net_gt_volume")
    X["resume"], X["prelist"], X["tradable"] = fb("f_resume_day"), fb("f_prelist_otc"), fb("f_tradable")
    X["liq"] = fb("f_liquidation_zone")
    nets = np.stack([pn[c].values for c in ["indiv_net", "inst_net", "foreign_net", "other_corp_net"]], 1)
    allzero = (nets == 0).all(1) & (pn.volume.values > 0)
    X["bad_wide"] = np.zeros(shape, bool)
    X["bad_wide"][ii[allzero], jj[allzero]] = True
    ssum = np.nansum(nets, 1)
    full4 = np.isfinite(nets).all(1)
    X["sum_nonzero_rows"] = pd.DataFrame({"date": pn.date.values, "nz": (ssum != 0) & full4, "ok": full4})
    X["allzero_rows"] = pd.DataFrame({"date": pn.date.values, "z": allzero, "zf": pn.f_zero_inv_suspect.values == 1})
    return X


def share_mat(X, mask, dates, tick):
    """r01.load_wide 의 개인 거래 비중과 같은 식. mask 인 종목-일은 매수·매도·거래량 모두 NaN."""
    ib, isl, vo = X["indiv_buy"].copy(), X["indiv_sell"].copy(), X["volume"].copy()
    for a in (ib, isl, vo):
        a[mask] = np.nan
    num = pd.DataFrame(ib + isl, index=dates, columns=tick).rolling(60, min_periods=40).sum()
    den = pd.DataFrame(vo, index=dates, columns=tick).rolling(60, min_periods=40).sum()
    return (num / (2 * den)).values


def make_W(W0, X, dates, tick, inv, s2, rank):
    """단계별 입력 행렬. 원본(W0)은 수정하지 않는다."""
    if inv is None and not s2 and not rank:
        return dict(W0)
    W = dict(W0)
    mask = np.zeros(W0["volume"].shape, bool)
    if inv == "flag":
        mask |= X["bad_flag"]
    elif inv == "wide":
        mask |= X["bad_wide"]
    if s2:
        mask |= X["prelist"]
    for k, raw in RAWK.items():
        M = (X[raw] if rank else W0[k]).copy()
        M[mask] = np.nan
        if rank:
            M = pd.DataFrame(M).rank(axis=1, pct=True).values - 0.5
        W[k] = M
    W["share"] = share_mat(X, mask, dates, tick)
    if s2:
        R = W0["ret1d_clean"].copy()
        R[X["resume"] | X["prelist"]] = np.nan
        W["ret1d_clean"] = R
    return W


def build(W, dates, tick, strict):
    r01.wsum = wsum_strict if strict else ORIG_WSUM
    b, _ = r01.build_base(W, dates, tick)
    b["s_cnn_d"], b["s_cnn_tb"], b["s_cnn_w"] = b.p_cnn_d, b.p_cnn_tb, b.p_cnn_w
    nd, ntk = b.k.max() + 1, len(tick)
    for s in SIGS:
        t_, e_, es_ = r01.flag_events(b.k.values, b.j.values, b["s_" + s].values, ntk, nd)
        b["top_" + s], b["ev_" + s], b["evs_" + s] = t_, e_, es_
    return b


ORIG_WSUM = r01.wsum


# ---------------------------------------------------------------- 요약 (r01 summarize 와 같은 열 + 보조 열)
def summarize(stage, sig, scale, rr):
    E_, regm, d_ev = rr["E"], rr["reg"], rr["d_ev"]
    ind_p = "fit_post" if scale == "tv" else "fi_post"
    row = dict(단계=stage, 신호=sig, 순매수척도=scale, 표본종목주=rr["n_samp"], 형성일수=rr["n_dates"],
               N이벤트=rr["n_ev"], N이벤트_키완비=rr["n_ev_key"], N매칭=len(E_), N대조후보=rr["n_ctrl"],
               최소칸=int(E_.n_ctrl.min()) if len(E_) else 0,
               중앙칸=float(E_.n_ctrl.median()) if len(E_) else np.nan)
    for c, nm in (("car15", "CAR15"), ("car620", "CAR620"), ("car2160", "CAR2160")):
        m_, _, t_, n_, _ = r01.cl_mean(d_ev[c], d_ev.i)
        row[f"이벤트{nm}"], row[f"t_이벤트{nm}"], row[f"N_이벤트{nm}"] = m_, t_, n_
        m_, _, t_, n_, _ = r01.cl_mean(E_[c + "_d"], E_.i)
        row[f"차이{nm}"], row[f"t_차이{nm}"], row[f"N_차이{nm}"] = m_, t_, n_
        row[f"t월_차이{nm}"] = r01.cl_mean(E_[c + "_d"], E_.ym)[2]
    for c, nm in ((ind_p, "개인사후주"), ("ff_post", "외국인사후주"), ("fn_post", "기관사후주"),
                  ("rit_post" if scale == "tv" else "ri_post", "개인사후주순위")):
        m_, _, t_, _, _ = r01.cl_mean(E_[c + "_d"], E_.i)
        row[f"차이{nm}"], row[f"t_차이{nm}"] = m_, t_
    for c, nm in (("fit_form" if scale == "tv" else "fi_form", "개인형성주"), ("ff_form", "외국인형성주"),
                  ("fn_form", "기관형성주")):
        m_, _, t_, _, _ = r01.cl_mean(E_[c + "_d"], E_.i)
        row[f"차이{nm}"], row[f"t_차이{nm}"] = m_, t_
    for t_ in ("indiv_post", "foreign_post", "inst_post"):
        row[f"회귀_{t_}"], row[f"t_회귀_{t_}"] = r01.coef(regm, "이벤트표본", "예", t_)
    row["회귀_CAR15계수"], row["t_회귀_CAR15계수"] = r01.coef(regm, "이벤트표본", "예", "car15")
    rg = regm[(regm.회귀 == "이벤트표본") & (regm.CAR15통제 == "예") & (regm.항 == "indiv_post")]
    row["회귀N"] = int(rg.N.iloc[0]) if len(rg) else 0
    row["회귀날짜수"] = int(rg.날짜수.iloc[0]) if len(rg) else 0
    for t_ in ("indiv_post", "foreign_post"):
        row[f"회귀_CAR15미통제_{t_}"], row[f"t_회귀_CAR15미통제_{t_}"] = r01.coef(regm, "이벤트표본", "아니오", t_)
    for c, nm in (("car620", "CAR620"), ("car2160", "CAR2160")):
        row[f"개인비중높음-낮음_차이{nm}"], row[f"t_개인비중높음-낮음_차이{nm}"], row[f"N_개인비중{nm}"] = r01.share_gap(E_, c)
    for t_ in ("indiv_post", "foreign_post"):
        row[f"순위회귀_{t_}"], row[f"t_순위회귀_{t_}"] = r01.coef(regm, "이벤트표본(순위흐름)", "예", t_)
    row["전종목_상위더미_합계_indiv"], row["t_전종목_상위더미_합계_indiv"] = r01.coef(regm, "전종목주+상위10분위더미", "예", "합계_indiv_post")
    row["전종목_상위더미_합계_foreign"], row["t_전종목_상위더미_합계_foreign"] = r01.coef(regm, "전종목주+상위10분위더미", "예", "합계_foreign_post")
    return row


def share_terciles(stage, sig, rr):
    E = rr["E"]
    rows = []
    for g, nm in ((0, "낮음"), (1, "중간"), (2, "높음")):
        s = E[E.share_t == g]
        for c in ("car15", "car620", "car2160"):
            m_, _, t_, n_, _ = r01.cl_mean(s[c + "_d"], s.i)
            rows.append(dict(단계=stage, 신호=sig, 개인비중분위=nm, 지표=r01.LABEL[c], N=n_, 차이=m_, t=t_))
    return rows


def verdict_row(r):
    """A-2 가격 압력 4조건(사전 고정, |t|>1.96): 개인 순매수 증가 / CAR(1,5)>0 / CAR(6,20)<0 / 개인 비중 높음-낮음 격차<0."""
    c1 = r["차이개인사후주"] > 0 and r["t_차이개인사후주"] > 1.96
    c2 = r["차이CAR15"] > 0 and r["t_차이CAR15"] > 1.96
    c3 = r["차이CAR620"] < 0 and r["t_차이CAR620"] < -1.96
    c4 = r["개인비중높음-낮음_차이CAR620"] < 0 and r["t_개인비중높음-낮음_차이CAR620"] < -1.96
    k = int(c1) + int(c2) + int(c3) + int(c4)
    ok = lambda b: "충족" if b else "불충족"
    return dict(단계=r["단계"], 신호=r["신호"], 순매수척도=r["순매수척도"], 개인순매수증가=ok(c1), CAR15양=ok(c2),
                CAR620음=ok(c3), 개인비중격차음=ok(c4), 충족수=k, 종합="충족" if k == 4 else "불충족",
                 회귀개인음_참고=ok(r["회귀_indiv_post"] < 0 and r["t_회귀_indiv_post"] < -1.96),
                값_개인순매수=r["차이개인사후주"], t_개인순매수=r["t_차이개인사후주"],
                값_CAR15=r["차이CAR15"], t_CAR15=r["t_차이CAR15"], 값_CAR620=r["차이CAR620"], t_CAR620=r["t_차이CAR620"],
                값_비중격차=r["개인비중높음-낮음_차이CAR620"], t_비중격차=r["t_개인비중높음-낮음_차이CAR620"])


# ---------------------------------------------------------------- 점검·진단
def replication_check(rows):
    t4 = pd.read_csv(f"{R01_OUT}/r01_t4_robustness.csv", encoding="utf-8-sig").set_index("설정")
    out = []
    worst = 0.0
    for sig in SIGS:
        for scale in SCALES:
            ref = t4.loc[f"{sig}|{scale}|금지주제외=False"]
            mine = rows[("S0", sig, scale)]
            cols = [c for c in t4.columns if c in mine and isinstance(mine[c], (int, float, np.integer, np.floating))
                    and isinstance(ref[c], (int, float, np.integer, np.floating))]
            dif = []
            for c in cols:
                a, b_ = float(mine[c]), float(ref[c])
                if np.isnan(a) and np.isnan(b_):
                    continue
                dif.append(abs(a - b_) if np.isfinite(a) and np.isfinite(b_) else np.inf)
            mx = max(dif) if dif else np.nan
            worst = max(worst, mx)
            out.append(dict(설정=f"{sig}|{scale}", 비교열수=len(dif), 최대절대차=mx))
    out = pd.DataFrame(out)
    out.to_csv(f"{OUT}/r02_t5_s0_replication_check.csv", index=False, encoding="utf-8-sig")
    P("[S0 재현 확인] r01_t4_robustness.csv 와 비교(숫자 열 전부):")
    P(out.to_string(index=False))
    assert worst < 1e-8, f"S0 가 r01 표와 다르다 (최대 절대차 {worst})"
    P("  -> S0 = r01 표와 일치 (최대 절대차 < 1e-8)")


def window_count(M, i, j, r0, r1):
    """각 (i,j)에서 행 i+r0 .. i+r1 구간의 True 개수 (행렬 M 은 bool, 범위 밖은 0 으로 본다)."""
    cs = np.vstack([np.zeros((1, M.shape[1]), np.int32), np.cumsum(M, axis=0, dtype=np.int32)])
    lo = np.clip(i + r0, 0, M.shape[0])
    hi = np.clip(i + r1 + 1, 0, M.shape[0])
    return cs[hi, j] - cs[lo, j]


def exposure(b0, X):
    """S0 표본에서 이벤트와 비이벤트 종목-주가 오염 의심 일을 얼마나 포함하는지."""
    d = b0[b0.k >= r01.BURN]
    i, j = d.i.values, d.j.values
    ev = d.ev_cnn_d.values
    rows = []
    wins = (("형성 주(-4~0)", -4, 0), ("사후 주(1~5)", 1, 5), ("6~20일", 6, 20))
    for nm, M in (("zero_inv_suspect (4그룹 전부 0, 거래량>=1만)", X["zero_inv_suspect"]),
                  ("보조: 4그룹 전부 0, 거래량>0 (문턱 없음)", X["bad_wide"])):
        for wn, r0, r1 in wins:
            c = window_count(M, i, j, r0, r1) > 0
            rows.append(dict(항목=nm, 창=wn, 이벤트비율=c[ev].mean(), 비이벤트비율=c[~ev].mean(), N이벤트=int(ev.sum())))
    for wn, r0, r1 in (("CAR(1,5) 창", 1, 5), ("CAR(6,20) 창", 6, 20), ("CAR(21,60) 창", 21, 60)):
        c = window_count(X["resume"], i, j, r0, r1) > 0
        rows.append(dict(항목="f_resume_day (거래재개일 수익이 창에 포함)", 창=wn, 이벤트비율=c[ev].mean(),
                         비이벤트비율=c[~ev].mean(), N이벤트=int(ev.sum())))
    for wn, r0, r1 in (("CAR(1,5) 창", 1, 5), ("CAR(6,20) 창", 6, 20), ("CAR(21,60) 창", 21, 60)):
        c = window_count(X["liq"], i, j, r0, r1) > 0
        rows.append(dict(항목="f_liquidation_zone (정리매매 구간이 창에 포함; r01은 CAR에서 빠짐, 미보정)", 창=wn,
                         이벤트비율=c[ev].mean(), 비이벤트비율=c[~ev].mean(), N이벤트=int(ev.sum())))
    # 형성일 시점 기준 위치
    pre = (d.date < CUT).values
    rows.append(dict(항목="형성일 < 2012-05-03 (4그룹 합 안 닫힌 구간)", 창="형성일", 이벤트비율=pre[ev].mean(),
                     비이벤트비율=pre[~ev].mean(), N이벤트=int(ev.sum())))
    ex = pd.DataFrame(rows)
    ex.to_csv(f"{OUT}/r02_t4_exposure.csv", index=False, encoding="utf-8-sig")
    P("\n[오염 노출도] S0 표본(r01), CNN 대표 신호 이벤트 vs 비이벤트 종목-주 중 해당 창에 의심일 포함 비율")
    P(ex.round(4).to_string(index=False))
    return ex


def data_facts(X, dates, tick):
    sz = X["sum_nonzero_rows"]
    sz = sz[(sz.date >= "2011-02-01") & (sz.date <= "2012-12-31") & sz.ok]
    pre, post = sz[sz.date < CUT], sz[sz.date >= CUT]
    z = X["allzero_rows"]
    z = z[(z.date >= "2011-02-01") & (z.date <= "2012-12-31")]
    zp, zq = z[z.date < CUT], z[z.date >= CUT]
    facts = pd.DataFrame([
        dict(항목="4그룹 합 != 0 비율 (표본 구간 2011-02~2012-12, 4열 모두 있는 행)", 컷이전=pre.nz.mean(), 컷이후=post.nz.mean(),
             행수_컷이전=len(pre), 행수_컷이후=len(post)),
        dict(항목="4그룹 전부 0 & 거래량>0 비율", 컷이전=zp.z.mean(), 컷이후=zq.z.mean(), 행수_컷이전=len(zp), 행수_컷이후=len(zq)),
        dict(항목="f_zero_inv_suspect==1 비율 (전부 0 & 거래량>=1만)", 컷이전=zp.zf.mean(), 컷이후=zq.zf.mean(),
             행수_컷이전=len(zp), 행수_컷이후=len(zq)),
    ])
    facts.to_csv(f"{OUT}/r02_t4b_data_facts.csv", index=False, encoding="utf-8-sig")
    P("\n[표본 구간 자료 사실]")
    P(facts.round(4).to_string(index=False))
    for y in (2011, 2012):
        zy = X["allzero_rows"]
        P(f"  {y}년 f_zero_inv_suspect 행 수(패널 전체 종목-일): {int(zy[zy.date.dt.year == y].zf.sum())} (q01: 2011 4,153 / 2012 2,731)")


# ---------------------------------------------------------------- 메인
def fmt(v, t, nd=2):
    if v is None or not np.isfinite(v):
        return "NA"
    return f"{v:+.{nd}f} ({t:+.2f})" if np.isfinite(t) else f"{v:+.{nd}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    P("r02 실행:", time.strftime("%Y-%m-%d %H:%M:%S"), "| 단계 정의는 파일 상단 독스트링(실행 전 고정)")
    W0, dates, tick, _ = r01.load_wide()
    X = load_extra(dates, tick)
    P(f"패널 {W0['volume'].shape} (일 x 종목), 플래그 행 수: zero_inv_suspect {int(X['zero_inv_suspect'].sum())}, "
      f"net_gt_volume {int(X['net_gt_vol'].sum())}, resume {int(X['resume'].sum())}, prelist {int(X['prelist'].sum())}, "
      f"비거래가능 {int((~X['tradable']).sum())} (2013-06-30 까지)")
    # share 재계산 식이 r01 과 같은지(마스크 없음)
    sm = share_mat(X, np.zeros(W0["volume"].shape, bool), dates, tick)
    assert np.array_equal(np.isnan(sm), np.isnan(W0["share"])) and np.allclose(sm[np.isfinite(sm)], W0["share"][np.isfinite(sm)], atol=1e-12), "share 재계산 불일치"

    # ---- 단계별 base 구성 (W 와 build 는 (inv, s2, rank, strict) 조합당 한 번)
    cache = {}
    bases = {}
    for st, sp in STAGES.items():
        key = (sp["inv"], sp["s2"], sp["rank"])
        if key not in cache:
            W = make_W(W0, X, dates, tick, *key)
            cache[key] = build(W, dates, tick, strict=(sp["inv"] is not None))
            P(f"  base 구성 {key}: 종목-형성일 {len(cache[key])}행")
        bases[st] = cache[key]
    b0 = bases["S0"]
    r01.selfcheck(b0, W0, dates, tick)
    LOG.append(r01.LOG[-1])
    # 이벤트 플래그가 단계 간 동일해야 한다(신호만으로 정의)
    for st, bb in bases.items():
        for s in SIGS:
            assert (bb["ev_" + s].values == b0["ev_" + s].values).all(), "이벤트 플래그가 단계마다 다름"

    # ---- 단계 적용 후 표본 필터
    rows, regs, shares, nfilter = {}, [], [], []
    runs = {}
    for st, sp in STAGES.items():
        bb = bases[st]
        keep = np.ones(len(bb), bool)
        if sp["s2"]:
            keep &= X["tradable"][bb.i.values, bb.j.values] & ~X["prelist"][bb.i.values, bb.j.values]
        n_univ_drop = int((~keep).sum())
        if sp["cut"] == "ge":
            keep &= (bb.date >= CUT).values
        elif sp["cut"] == "lt":
            keep &= (bb.date < CUT).values
        bf = bb[keep]
        nfilter.append(dict(단계=st, 전체행=len(bb), 유니버스제외행=n_univ_drop, 최종행=len(bf)))
        for sig in SIGS:
            for scale in SCALES:
                rr = r01.run(bf, dates, sig, scale)
                runs[(st, sig, scale)] = rr
                rows[(st, sig, scale)] = summarize(st, sig, scale, rr)
                x = rr["reg"].copy()
                x.insert(0, "순매수척도", scale)
                x.insert(0, "신호", sig)
                x.insert(0, "단계", st)
                regs.append(x)
            shares += share_terciles(st, sig, runs[(st, sig, "vol")])
        P(f"  {st} 완료 ({time.time() - t0:.0f}초)")
    pd.DataFrame(nfilter).to_csv(f"{OUT}/r02_t0_universe_filter.csv", index=False, encoding="utf-8-sig")
    P("\n[유니버스 필터] 형성일 행이 비거래가능/OTC 사전 구간이라 빠진 행 수(S2 이상)")
    P(pd.DataFrame(nfilter).to_string(index=False))

    replication_check(rows)

    t1 = pd.DataFrame(list(rows.values()))
    t1.to_csv(f"{OUT}/r02_t1_stage_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(regs).to_csv(f"{OUT}/r02_t6_regressions_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shares).to_csv(f"{OUT}/r02_t7_share_terciles.csv", index=False, encoding="utf-8-sig")

    # ---- 판정표
    v = pd.DataFrame([verdict_row(r) for r in rows.values()])
    v.to_csv(f"{OUT}/r02_t3_verdict_by_stage.csv", index=False, encoding="utf-8-sig")

    # ---- 단계 비교(가로) 표: 신호 x 지표 x 단계
    def stage_table(sig, scale="vol", stages=None):
        stages = stages or list(STAGES)
        spec = [("형성일 수", "형성일수", None), ("이벤트 수", "N이벤트", None), ("매칭 이벤트 수", "N매칭", None),
                ("대조 후보 수", "N대조후보", None), ("칸 크기 최소/중앙", None, None),
                ("CAR(1,5) 이벤트-대조", "차이CAR15", "t_차이CAR15"), ("CAR(6,20) 이벤트-대조", "차이CAR620", "t_차이CAR620"),
                ("CAR(21,60) 이벤트-대조", "차이CAR2160", "t_차이CAR2160"),
                ("CAR(1,5) 이벤트(규모5분위 대비)", "이벤트CAR15", "t_이벤트CAR15"),
                ("CAR(6,20) 이벤트(규모5분위 대비)", "이벤트CAR620", "t_이벤트CAR620"),
                ("개인 사후주 순매수 이벤트-대조", "차이개인사후주", "t_차이개인사후주"),
                ("외국인 사후주 순매수 이벤트-대조", "차이외국인사후주", "t_차이외국인사후주"),
                ("기관 사후주 순매수 이벤트-대조", "차이기관사후주", "t_차이기관사후주"),
                ("회귀 개인 (CAR15 통제)", "회귀_indiv_post", "t_회귀_indiv_post"),
                ("회귀 외국인 (CAR15 통제)", "회귀_foreign_post", "t_회귀_foreign_post"),
                ("회귀 기관 (CAR15 통제)", "회귀_inst_post", "t_회귀_inst_post"),
                ("개인 비중 높음-낮음 CAR(6,20) 격차", "개인비중높음-낮음_차이CAR620", "t_개인비중높음-낮음_차이CAR620")]
        out = []
        for lab, c, tc in spec:
            r = {"지표": lab}
            for st in stages:
                m = rows[(st, sig, scale)]
                if lab.startswith("칸"):
                    r[st] = f"{m['최소칸']}/{m['중앙칸']:.0f}"
                elif tc is None:
                    r[st] = f"{int(m[c]):,}"
                else:
                    r[st] = fmt(m[c], m[tc])
            out.append(r)
        return pd.DataFrame(out)

    allt = []
    for sig in SIGS:
        for scale in SCALES:
            tt = stage_table(sig, scale)
            tt.insert(0, "순매수척도", scale)
            tt.insert(0, "신호", sig)
            allt.append(tt)
    pd.concat(allt).to_csv(f"{OUT}/r02_t2_stage_compare.csv", index=False, encoding="utf-8-sig")

    for sig in SIGS:
        P(f"\n=== 단계 비교 (신호 {sig}, 거래량 척도; 값 (t)) ===")
        P(stage_table(sig, "vol").to_string(index=False))
    P("\n=== 판정표 (A-2 가격 압력, 거래량 척도) ===")
    P(v[v.순매수척도 == "vol"].drop(columns=["순매수척도"]).round(3).to_string(index=False))
    P("\n=== 판정 참고: 거래대금 척도 ===")
    P(v[v.순매수척도 == "tv"].drop(columns=["순매수척도"]).round(3).to_string(index=False))

    # ---- 진단
    data_facts(X, dates, tick)
    exposure(b0, X)
    P(f"\n소요 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/r02_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(LOG))

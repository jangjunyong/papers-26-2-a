"""발표 자료용 그림 5장 → 발표/figs/. 입력은 이미 나온 결과 파일뿐이다(새로 학습하지 않는다).
사용: py -3 fig_presentation.py"""
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dg  # noqa: E402
import images  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "발표" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a7a1"
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "font.size": 11})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def deciles(p, n=10):
    p = p.copy()
    p["q"] = p.groupby("date").prob.transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False))
    return p.groupby(["date", "q"]).ret.mean().unstack()


# 1) 차트 이미지 예시 (DataGuide 삼성전자, 2010년 가을 3장)
fr = {"005930": dg.frames("2009-06-01", "2010-12-31")["005930"]}
x, m = images.build(fr, "2010-09-01", "2010-11-30")
pick = [0, len(x) // 2, len(x) - 1]
fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
for ax, i in zip(axes, pick):
    ax.imshow(x[i], cmap="gray", interpolation="nearest")
    ax.set_title(f"{m.date[i]:%Y-%m-%d}\n다음 5일 수익 {m.ret[i] * 100:+.1f}%", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
fig.suptitle("CNN 입력: 삼성전자 20일 차트 이미지 (64×60)", fontsize=12)
save(fig, "fig1_chart_examples.png")

# 2) 성과 비교: 상위−하위 10분위 롱숏 주간 평균 (같은 평가 표본 71,222개)
kg = {r["name"]: r for r in json.load(open(RES / "kaggle" / "results.json", encoding="utf-8"))}
base = pd.read_csv(RES / "dg_pilot_compare.csv").set_index("model")
rows = [("CNN · 매일 이미지(98만 장)", kg["d_lr1e-4"]["hl_weekly_mean"], kg["d_lr1e-4"]["hl_t"], BLUE),
        ("CNN · 주 단위 이미지(20만 장)", kg["w_lr1e-4"]["hl_weekly_mean"], kg["w_lr1e-4"]["hl_t"], BLUE),
        ("단순 지표 6개 조합(로지스틱)", base.loc["logit_all", "hl_weekly"], base.loc["logit_all", "t"], GRAY),
        ("1개월 반전", base.loc["rev20", "hl_weekly"], base.loc["rev20", "t"], GRAY),
        ("5일 반전", base.loc["rev5", "hl_weekly"], base.loc["rev5", "t"], GRAY)][::-1]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh([r[0] for r in rows], [r[1] * 100 for r in rows], color=[r[3] for r in rows], height=0.55)
for k, (_, v, t, _) in enumerate(rows):
    ax.text(v * 100 + 0.02, k, f"{v * 100:+.2f}%  (t {t:.1f})", va="center", color=INK, fontsize=10)
ax.set_xlim(0, 1.45), ax.set_xlabel("주간 평균 수익률 (%)"), ax.grid(axis="y", visible=False)
ax.set_title("예측 상위 10% 매수 + 하위 10% 매도(롱숏), 2011~2012년, 거래비용 차감 전", fontsize=11, loc="left")
save(fig, "fig2_compare.png")

# 3) 10분위별 주간 수익 (CNN 매일 이미지)
pd_ = pd.read_parquet(RES / "kaggle" / "pred_d_lr1e-4.parquet")
dec = deciles(pd_).mean() * 100
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(range(1, 11), dec.values, color=BLUE, width=0.6)
ax.axhline(0, color=INK2, linewidth=0.8)
ax.set_xticks(range(1, 11), [f"{i}" for i in range(1, 11)])
ax.set_xlabel("CNN 예측 순위 10분위 (1 = 가장 낮음, 10 = 가장 높음)"), ax.set_ylabel("다음 5일 평균 수익 (%)")
ax.grid(axis="x", visible=False)
ax.set_title("예측이 높은 분위일수록 실제 수익도 높다 (CNN 매일 이미지)", fontsize=11, loc="left")
save(fig, "fig3_deciles.png")

# 4) 누적 수익(주간 수익 단순 합)
pw = pd.read_parquet(RES / "kaggle" / "pred_w_lr1e-4.parquet")
hl_d = (lambda d: d[9] - d[0])(deciles(pd_)).cumsum() * 100
hl_w = (lambda d: d[9] - d[0])(deciles(pw)).cumsum() * 100
mkt = pd_.groupby("date").ret.mean().cumsum() * 100
fig, ax = plt.subplots(figsize=(8, 3.8))
for s, c, lab, ls in [(hl_d, BLUE, "CNN 매일 이미지 (롱숏)", "-"), (hl_w, ORANGE, "CNN 주 단위 (롱숏)", "-")]:
    ax.plot(s.index, s.values, color=c, linewidth=2, linestyle=ls, label=lab)
    ax.text(s.index[-1], s.values[-1], f" {s.values[-1]:+.0f}%", color=INK, va="center", fontsize=10)
ax.axhline(0, color=INK2, linewidth=0.8)
ax.set_ylabel("누적 수익 (%, 주간 수익 단순 합)"), ax.legend(frameon=False, loc="upper left")
ax.set_title("롱숏 누적 수익: 주간 수익 단순 합, 동일가중, 거래비용 차감 전 (2011~2012)", fontsize=11, loc="left")
save(fig, "fig4_cumulative.png")

# 5) CNN이 무엇을 배웠나: 단순 지표와의 순위상관 (주 단위 모델)
cr = pd.read_csv(RES / "dg_pilot_corr.csv").set_index("indicator").spearman_mean
names = {"abvol": "비정상 거래량", "rev5": "5일 반전", "vol20": "낮은 변동성", "rev20": "1개월 반전",
         "mom12_1": "12개월 모멘텀", "hi52": "52주 고가 근접"}
cr = cr.rename(names).sort_values()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(cr.index, cr.values, color=[BLUE if v > 0.1 else GRAY for v in cr.values], height=0.55)
for k, v in enumerate(cr.values):
    ax.text(max(v, 0) + 0.005, k, f"{0 if abs(v) < 0.005 else v:+.2f}", va="center", fontsize=10)
ax.axvline(0, color=INK2, linewidth=0.8), ax.set_xlim(-0.05, 0.27), ax.grid(axis="y", visible=False)
ax.set_xlabel("CNN 예측과의 순위상관 (매주 계산한 평균)")
ax.set_title("CNN 예측은 단기 반전·비정상 거래량과 주로 겹친다 (주 단위 모델)", fontsize=11, loc="left")
save(fig, "fig5_what_cnn_learned.png")
print("그림 5장 →", OUT)

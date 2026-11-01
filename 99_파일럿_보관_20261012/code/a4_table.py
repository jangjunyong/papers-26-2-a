"""파일럿 2개 + A4 실험 결과를 마크다운 표로 출력한다. 사용: py -3 a4_table.py"""
import json
import pathlib
import re

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
runs = [("파일럿 lr1e-5 (주, 5일)", "pilot_lr1e-05"), ("파일럿 lr1e-4 (주, 5일)", "pilot_lr0.0001"),
        ("E1 매일 형성, 무작위 검증", "a4_E1"), ("E1t 매일 형성, 2010 블록 검증", "a4_E1t"),
        ("E2 거래량 제거", "a4_E2"), ("E3 20일 라벨", "a4_E3")]
print("| 실험 | 학습 이미지 | 에폭(최저) | 최저 검증 손실 | 정확도 | 롱숏 평균 | 샤프 | t |\n|---|---|---|---|---|---|---|---|")
for label, tag in runs:
    log = RES / f"{tag}_train.log"
    if not log.exists():
        continue
    txt = log.read_text(encoding="utf-8")
    v = [float(x) for x in re.findall(r"epoch \d+: val loss ([\d.]+)", txt)]
    n = re.search(r"학습 이미지 (\d+)장", txt)
    f = RES / f"{tag}_result.json"
    r = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    best = f"{len(v)}({v.index(min(v)) + 1})" if v else "-"
    cell = (lambda k, fmt: format(r[k], fmt) if k in r else ("-" if r else "진행 중"))
    print(f"| {label} | {n.group(1) if n else '-'} | {best} | {min(v):.4f} |" if v else f"| {label} | - | - | - |",
          cell("accuracy", ".3f"), "|", cell("hl_weekly_mean", ".4f"), "|", cell("hl_sharpe", ".2f"), "|",
          cell("hl_t", ".2f"), "|")
    if "nonoverlap_every4w" in r:
        o = r["nonoverlap_every4w"]
        print(f"| E3 비중첩(4주마다) | | | | {o['accuracy']:.3f} | {o['hl_weekly_mean']:.4f} | {o['hl_sharpe']:.2f} | {o['hl_t']:.2f} |")

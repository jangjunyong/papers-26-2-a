"""Kaggle에 비공개 데이터셋(파생 이미지)과 GPU 커널을 올리고, 끝나면 결과를 받는다.
사용: py -3 kaggle_push.py data   → 데이터셋 생성/새 버전
      py -3 kaggle_push.py run    → 커널 올리기(GPU)
      py -3 kaggle_push.py fetch  → 상태 확인, 끝났으면 results/kaggle/로 내려받기
v3(사전고정_v1): 맨 끝에 v3 를 붙인다. run/fetch 는 시드 묶음 a(0,1)·b(2,3)·c(4) 를 커널 3개로 나눈다(12시간 세션 제한).
      py -3 kaggle_push.py data v3 | run v3 [a b c] | fetch v3 [a b c]
토큰은 사용자가 ~/.kaggle/kaggle.json(또는 access_token)으로 넣는다. 여기서는 읽지도 출력하지도 않는다.
"""
import json
import pathlib
import subprocess
import sys

CODE = pathlib.Path(__file__).resolve().parent
V3 = "v3" in sys.argv[2:]
UP = CODE.parents[2] / "data" / ("kaggle_upload_v3" if V3 else "kaggle_upload")
KER = CODE.parent / ("kaggle_kernel_v3" if V3 else "kaggle_kernel")
OUT = CODE.parent / "results" / ("kaggle_v3" if V3 else "kaggle")
SEED_GROUPS = {"a": [0, 1], "b": [2, 3], "c": [4]}


def kg(*a):
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True, encoding="utf-8")
    print(r.stdout.strip(), r.stderr.strip())
    return r


user = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout.split("username:")[-1].split()[0]
DS, KN = f"{user}/kospi-chart-pilot", f"{user}/kospi-chart-cnn-pilot"
if V3:
    DS = f"{user}/kospi-chart-v3"
cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
groups = [g for g in sys.argv[2:] if g in SEED_GROUPS] or list(SEED_GROUPS)


def push(kn, title, main, ker, head=""):
    ker.mkdir(parents=True, exist_ok=True)
    (ker / "kernel.py").write_text((CODE / "cnn.py").read_text(encoding="utf-8") + "\n\n" + head
                                   + (CODE / main).read_text(encoding="utf-8"), encoding="utf-8")
    (ker / "kernel-metadata.json").write_text(json.dumps(
        {"id": kn, "title": title, "code_file": "kernel.py", "language": "python",
         "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
         "dataset_sources": [DS], "competition_sources": [], "kernel_sources": []}), encoding="utf-8")
    kg("kernels", "push", "-p", str(ker))


def fetch(kn, out):
    r = kg("kernels", "status", kn)
    if "complete" in r.stdout.lower():
        out.mkdir(parents=True, exist_ok=True)
        kg("kernels", "output", kn, "-p", str(out))


if cmd == "data":
    (UP / "dataset-metadata.json").write_text(json.dumps(
        {"title": DS.split("/")[1], "id": DS, "licenses": [{"name": "other"}]}), encoding="utf-8")
    r = kg("datasets", "version", "-p", str(UP), "-m", "update", "-r", "skip")
    if r.returncode:
        kg("datasets", "create", "-p", str(UP), "-r", "skip")          # 기본값이 비공개
elif cmd == "run" and V3:
    for g in groups:
        push(f"{user}/kospi-chart-cnn-v3-{g}", f"kospi-chart-cnn-v3-{g}", "kaggle_main_v3.py", KER / g, f"SEEDS = {SEED_GROUPS[g]}\n")
elif cmd == "run":
    push(KN, "kospi-chart-cnn-pilot", "kaggle_main.py", KER)
elif V3:
    for g in groups:
        fetch(f"{user}/kospi-chart-cnn-v3-{g}", OUT / g)
else:
    fetch(KN, OUT)

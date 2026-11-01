"""t01 전체 흐름을 시험 구간 밖(2011-06-01~2012-04-20)에서 돌려 배관만 점검. 출력은 scratchpad, 수치는 쓰지 않는다."""
import pathlib, sys
sys.path.insert(0, r"C:\Users\jacob\Documents\Obsidian Vault\PAPERS\26-2\03_차트CNN_연구\code")
sys.stdout.reconfigure(encoding="utf-8")
import t01_test_run as t
t.m01.WINDOWS["test"] = ("2011-06-01", "2012-04-20")
t.OUT = pathlib.Path(__file__).parent / "t01_dry_out"
t.unlocked_features = lambda: t.FEAT_OPEN
t.git = lambda *a: "" if a[0] == "status" else "dry"
t.main()

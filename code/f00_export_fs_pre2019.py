"""f00: 동아리 DB fs_line 의 2019년 이전 재무 4계정을 새 파일로 받는다 (읽기 전용, 쿼리 한 번에 하나).

배경: d03_export_db_ext.py 의 핵심 계정 목록은 'ifrs-full_' 접두어 ID 만 담았는데, DART XBRL 분류가 바뀌기 전(~2019 상반기 보고서)에는
같은 계정이 'ifrs_' 접두어(예: ifrs_Assets)로 적혀 있어 fs_line_core.parquet 에 2015~2019-06 재무가 거의 없다(종목 2~15개).
이 스크립트는 자산총계·부채총계·자본총계·당기순이익 4계정만 'ifrs_' ID 로 받아 fs_line_core_pre2019.parquet 에 저장한다.
f01_build_features.load_inputs 가 이 파일을 읽어 ID 를 'ifrs-full_' 로 바꿔 붙인다. 기존 fs_line_core.parquet 는 건드리지 않는다.
사용: py -3 f00_export_fs_pre2019.py   (조각 20개, 파일이 있으면 건너뜀 = 재개 가능)
"""
import os
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, r"C:\dataguide")
os.chdir(r"C:\dataguide")
from dataguide import DataGuide  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[3] / "data" / "daily" / "club_db_ext"
PARTS = OUT / "_parts_pre2019"
FINAL = OUT / "fs_line_core_pre2019.parquet"
ACCTS = ["ifrs_Assets", "ifrs_Liabilities", "ifrs_Equity", "ifrs_ProfitLoss"]
YEARS = range(2015, 2020)
REPORTS = ("11013", "11012", "11014", "11011")
COLS = ("f.code, m.market, f.bsns_year, f.reprt_code, f.fs_div, f.sj_div, f.account_id, f.account_nm, f.period_end, f.currency, "
        "f.thstrm_amount, f.frmtrm_amount, f.bfefrmtrm_amount")


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    dg = DataGuide()
    ids = {f"i{k}": a for k, a in enumerate(ACCTS)}
    inlist = ", ".join(":" + k for k in ids)
    sql = (f"SELECT {COLS} FROM financials.fs_line f JOIN dataguide.stock_master m ON m.code = f.code "
           f"WHERE f.bsns_year = :y AND f.reprt_code = :r AND m.market = 'KOSPI' AND f.account_id IN ({inlist})")
    for y in YEARS:
        for r in REPORTS:
            f = PARTS / f"fs_{y}_{r}.parquet"
            if f.exists():
                continue
            for attempt in range(4):
                try:
                    t0 = time.time()
                    df = dg.query(sql, {"y": y, "r": r, **ids})
                    break
                except Exception as e:                               # 연결 끊김 등: 잠시 쉬고 다시
                    print(f"  {y} {r} 실패({type(e).__name__}) 재시도 {attempt + 1}", flush=True)
                    time.sleep(30)
            else:
                sys.exit(f"{y} {r} 4번 실패, 중단")
            df.to_parquet(f, index=False)
            print(f"{y} {r}: {len(df):,}행, 종목 {df['code'].nunique()}, {time.time() - t0:.0f}초", flush=True)
            time.sleep(0.5)
    out = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("fs_*.parquet"))], ignore_index=True)
    key = ["code", "reprt_code", "period_end", "fs_div", "sj_div", "account_id"]
    assert not out.duplicated(key).any(), "키 중복"
    out.to_parquet(FINAL, index=False)
    print(f"저장 {FINAL}: {len(out):,}행, 종목 {out['code'].nunique()}")
    print(out.groupby(["account_id", "sj_div"]).size())


if __name__ == "__main__":
    main()

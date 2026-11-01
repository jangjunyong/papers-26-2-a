"""동아리 DB(DataGuide 팀 DB)에서 club_db 에 없는 자료를 parquet 으로 받아 검증한다 (재개 가능).

출력: PAPERS/data/daily/club_db_ext/  (기존 club_db/ 는 건드리지 않는다)
사용: py -3 d03_export_db_ext.py [단계 ...]
  단계: meta kiwoom inv_detail inv_kosdaq shares trading price fs verify   (인자 없으면 이 순서로 전부)
        fs_ifrs          — 2019 상반기 이전 `ifrs_` 접두어 계정(핵심 계정 중 자산·부채·자본·당기순이익 제외) → fs_line_core_pre2019_ext
        fs_ifrs4_kosdaq  — 위 4계정의 코스닥분 → fs_line_core_pre2019_kosdaq4 (타 에이전트의 pre2019 파일은 코스피 589~758곳뿐)
        fs_probe / fs_prefix_probe — 계정·접두어 빈도 확인(출력/_fs_prefix_probe.csv), named — 이름 있는 표본 대조(_named_checks.json)
        clean            — verify 통과 뒤 _parts/ 임시 조각을 지운다(지운 뒤 다시 돌리면 처음부터 받는다)
원칙: 서버 쿼리는 한 번에 하나. 큰 테이블은 분기(fs_line 은 연도×보고서×연결/별도) 조각으로 받고
      조각마다 같은 조건의 COUNT(*) 와 받은 행 수가 같을 때만 _parts/ 에 저장한다(파일이 있으면 건너뜀 = 재개).
      'Lost connection' 류가 나면 조각을 절반으로 쪼개 재시도하고, 서버가 DEATH_LIMIT 번 죽으면 멈춘다.
"""
import json
import os
import sys
import time
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dataguide")
os.chdir(r"C:\dataguide")
import dataguide  # noqa: E402
from dataguide import DataGuide  # noqa: E402

PAPERS = pathlib.Path(__file__).resolve().parents[3]
OUT = PAPERS / "data" / "daily" / "club_db_ext"
PARTS = OUT / "_parts"
LOG = OUT / "_piece_log.csv"
OUT.mkdir(parents=True, exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)

CUTOFF = "2026-10-08"      # DB 의 최신 주가일(2026-10-11 확인). 이후에 적재된 행은 받지 않는다 → 재실행해도 행 수 고정
DEATH_LIMIT = 4            # 쿼리 도중 서버가 끊긴 횟수가 이만큼 쌓이면 멈추고 보고한다
SLEEP = 0.4

# fs_line 핵심 계정(고정). 2024 연결(CFS) 사업보고서(11011)에서 account_id 빈도 상위(회사 2,081곳 중 해당 계정을 가진 곳)를 보고 골랐다.
# 총차입금은 단일 태그가 없어(공시 회사마다 태그가 제각각) 차입금·사채 계열 태그를 묶어서 받는다.
CORE_ACCOUNTS = [
    # 재무상태표 BS
    "ifrs-full_Assets", "ifrs-full_Liabilities", "ifrs-full_Equity", "ifrs-full_EquityAttributableToOwnersOfParent",
    "ifrs-full_CurrentAssets", "ifrs-full_CurrentLiabilities", "ifrs-full_NoncurrentAssets", "ifrs-full_NoncurrentLiabilities",
    "ifrs-full_CashAndCashEquivalents", "ifrs-full_Inventories", "ifrs-full_PropertyPlantAndEquipment",
    "ifrs-full_RetainedEarnings", "ifrs-full_IssuedCapital",
    # 차입금·사채 계열 (BS)
    "ifrs-full_ShorttermBorrowings", "ifrs-full_CurrentPortionOfLongtermBorrowings", "ifrs-full_CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings",
    "dart_LongTermBorrowingsGross", "ifrs-full_LongtermBorrowings", "ifrs-full_Borrowings",
    "ifrs-full_BondsIssued", "ifrs-full_NoncurrentPortionOfNoncurrentBondsIssued", "dart_ConvertibleBonds", "dart_CurrentPortionOfConvertibleBonds",
    # 손익계산서 IS
    "ifrs-full_Revenue", "ifrs-full_CostOfSales", "ifrs-full_GrossProfit", "dart_OperatingIncomeLoss",
    "dart_TotalSellingGeneralAdministrativeExpenses", "ifrs-full_FinanceCosts", "ifrs-full_InterestExpense",
    "ifrs-full_ProfitLossBeforeTax", "ifrs-full_IncomeTaxExpenseContinuingOperations",
    "ifrs-full_ProfitLoss", "ifrs-full_ProfitLossAttributableToOwnersOfParent", "ifrs-full_BasicEarningsLossPerShare",
    # 현금흐름표 CF
    "ifrs-full_CashFlowsFromUsedInOperatingActivities", "ifrs-full_CashFlowsFromUsedInInvestingActivities",
    "ifrs-full_CashFlowsFromUsedInFinancingActivities", "ifrs-full_InterestPaidClassifiedAsOperatingActivities",
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "ifrs-full_AdjustmentsForDepreciationExpense",
]

# ---------------------------------------------------------------- 서버 접속·조각 실행
_dg = None
_deaths = 0


TUNNEL_PORT = 3308      # 기본 터널(3307)과 겹치지 않게 직접 여는 압축 터널
_ssh = None


def _port_open(port):
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_ssh():
    """DataGuide 와 같은 SSH 터널을 `-C`(압축)로 직접 연다. 서버 업링크가 느려(13KB/s~) 전송량을 줄이려는 것이다.
    MySQL 의 TLS 는 SSH 안에서 이중 암호화라 끄고(ssl_disabled) 압축이 먹게 한다."""
    global _ssh
    import atexit
    import shutil
    import subprocess
    if _port_open(TUNNEL_PORT):
        return
    c = dataguide._read_env()
    cmd = [shutil.which("ssh"), "-C", "-N", "-L", f"{TUNNEL_PORT}:127.0.0.1:{c['ssh_remote_port']}", "-i", os.path.expanduser(c["ssh_key"]),
           "-p", str(c["ssh_port"]), "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", f"{c['ssh_user']}@{c['ssh_host']}"]
    _ssh = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    atexit.register(_ssh.terminate)
    for _ in range(60):
        if _port_open(TUNNEL_PORT):
            return
        time.sleep(0.5)
    raise OSError("SSH 터널을 열지 못함 (2003)")


def dg():
    global _dg
    if _dg is None:
        from sqlalchemy import create_engine
        ensure_ssh()
        d = DataGuide(port=TUNNEL_PORT, ssh_host="")           # 계정·DB 는 DataGuide(.env) 그대로, 터널은 위에서 직접 연 것
        d.engine = create_engine(d.engine.url, connect_args={"ssl_disabled": True}, pool_pre_ping=True)
        _dg = d
    return _dg


class Lost(Exception):
    """쿼리 도중 서버가 연결을 끊음('Lost connection') — 호출자가 조각을 더 작게 나눠 다시 시도한다."""


def _is_conn_err(e):
    s = str(e)
    return any(k in s for k in ("Lost connection", "2013", "2006", "2003", "gone away", "WRONG_VERSION", "UNEXPECTED_EOF",
                                "10054", "10053", "10061", "timed out"))


def wait_server(maxwait=1800):
    """서버가 다시 응답할 때까지 30초마다 확인한다(다른 에이전트 부하로 재시작되는 경우가 있었다). 30분 넘으면 중단."""
    global _dg
    t0 = time.time()
    while True:
        try:
            dg().query("SELECT 1")
            return
        except Exception as e:  # noqa: BLE001
            if not _is_conn_err(e):
                raise
            _dg = None
            if time.time() - t0 > maxwait:
                sys.exit("서버가 30분 넘게 응답하지 않아 중단 (다시 실행하면 이어 받는다)")
            print(f"  [서버 응답 없음 {int(time.time() - t0)}s] {str(e)[:70]}", flush=True)
            time.sleep(30)


def q(sql, params=None):
    """쿼리 하나. 접속 자체가 안 되면 서버가 돌아올 때까지 기다려 다시 시도하고,
    쿼리 도중 끊기면(Lost) 서버 복귀를 기다린 뒤 Lost 를 올린다. DEATH_LIMIT 번 끊기면 중단."""
    global _dg, _deaths
    for _ in range(3):
        try:
            df = dg().query(sql, params)
            time.sleep(SLEEP)
            return df
        except Exception as e:  # noqa: BLE001
            if not _is_conn_err(e):
                raise
            mid = "during query" in str(e) or "gone away" in str(e) or "(2013" in str(e) or "(2006" in str(e)
            _dg = None
            print(f"  [{'쿼리 도중 끊김' if mid else '접속 실패'}] {str(e)[:90]}", flush=True)
            wait_server()
            if mid:
                _deaths += 1
                if _deaths >= DEATH_LIMIT:
                    sys.exit(f"쿼리 도중 서버가 {DEATH_LIMIT}번 끊겨 중단 (다시 실행하면 이어 받는다)")
                raise Lost(str(e)[:120])
    sys.exit("접속을 3번 복구했지만 쿼리가 계속 실패해 중단")


def log_piece(name, rows, server, sec):
    new = not LOG.exists()
    with open(LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("piece,rows,server_count,seconds,at\n")
        f.write(f"{name},{rows},{server},{sec:.1f},{dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")


DATECOLS = {"date", "first_date", "last_date", "start_date", "end_date"}


def to_dt(df):
    for c in df.columns:
        if c in DATECOLS:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:  # noqa: BLE001  (9999-12-31 같은 값은 날짜 객체로 둔다)
                pass
    return df


def run_piece(name, sel, cnt, params):
    """조각 하나. 저장 = 행 수가 서버 COUNT(*) 와 같을 때만. 이미 있으면 건너뜀."""
    f = PARTS / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f).shape[0]
    t = time.time()
    n = int(q(cnt, params).iloc[0, 0])
    df = to_dt(q(sel, params))     # 0행이어도 열 이름은 오므로 그대로 저장한다
    if len(df) != n:
        raise RuntimeError(f"{name}: 받은 {len(df)}행 != 서버 COUNT {n}")
    df.to_parquet(f)
    log_piece(name, len(df), n, time.time() - t)
    print(f"  {name}: {n:,}행 ({time.time() - t:.1f}s)", flush=True)
    return n


def run_range(prefix, sel, cnt, a, b, extra=None):
    """[a,b] 날짜 구간 조각. 쿼리 도중 서버가 끊기면 구간을 절반으로 쪼개 다시 받는다."""
    p = {"a": a, "b": b, **(extra or {})}
    name = f"{prefix}_{a}_{b}"
    try:
        return run_piece(name, sel, cnt, p)
    except Lost:
        da, db = pd.Timestamp(a), pd.Timestamp(b)
        if (db - da).days < 1:
            sys.exit(f"하루 조각도 실패: {name}")
        mid = da + (db - da) / 2
        m1, m2 = mid.strftime("%Y-%m-%d"), (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  {name} 끊김 → 절반으로 쪼갬 ({a}~{m1}, {m2}~{b})", flush=True)
        time.sleep(60)
        return run_range(prefix, sel, cnt, a, m1, extra) + run_range(prefix, sel, cnt, m2, b, extra)


def run_whole(name, sel, cnt, params=None):
    """쪼갤 수 없는 작은 테이블 조각. 끊기면 한 번 더 시도하고 안 되면 멈춘다."""
    try:
        return run_piece(name, sel, cnt, params or {})
    except Lost:
        time.sleep(60)
        return run_piece(name, sel, cnt, params or {})


def quarters(y0, y1=2026):
    for y in range(y0, y1 + 1):
        for qn in range(4):
            a = dt.date(y, 3 * qn + 1, 1)
            b = (pd.Timestamp(a) + pd.offsets.QuarterEnd(0)).date()
            if str(a) > CUTOFF:
                return
            yield str(a), min(str(b), CUTOFF)


def months(y0, y1=2026):
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            a = dt.date(y, m, 1)
            b = (pd.Timestamp(a) + pd.offsets.MonthEnd(0)).date()
            if str(a) > CUTOFF:
                return
            yield str(a), min(str(b), CUTOFF)


def years(y0, y1=2026):
    for y in range(y0, y1 + 1):
        a = f"{y}-01-01"
        if a > CUTOFF:
            return
        yield a, min(f"{y}-12-31", CUTOFF)


# ---------------------------------------------------------------- 조립
def assemble(prefix, final, keys, intcols=(), sortby=None, date_check=None):
    """_parts/{prefix}_*.parquet 를 이어 붙여 OUT/{final}.parquet 로 저장. 조각 합계 == 서버 COUNT 합계, 키 중복 0 확인."""
    files = sorted(PARTS.glob(f"{prefix}_*.parquet"))
    if not files:
        raise RuntimeError(f"{prefix}: 조각 없음")
    dfs = [pd.read_parquet(f) for f in files]
    dfs = [d for d in dfs if len(d)]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    import decimal
    for c in df.columns:      # 한 조각이 전부 NULL 이면 object 열이 된다 → 실제 숫자 타입인 열만 숫자로 되돌린다(reprt_code 같은 문자열은 건드리지 않음)
        if df[c].dtype == object and df[c].dropna().map(lambda v: isinstance(v, (int, float, decimal.Decimal))).all():
            df[c] = pd.to_numeric(df[c])
    for c in intcols:
        if c in df.columns and not df[c].isna().any():
            df[c] = df[c].astype("int64")
        elif c in df.columns:
            assert not (df[c].abs().max() >= 2 ** 53), c      # float64 로 담아도 정수가 정확해야 한다
            df[c] = df[c].astype("float64")
    df = df.sort_values(sortby or keys).reset_index(drop=True)
    dup = int(df.duplicated(keys).sum())
    if dup:
        raise RuntimeError(f"{final}: 키 {keys} 중복 {dup}행")
    df.to_parquet(OUT / f"{final}.parquet", index=False)
    print(f"[조립] {final}: {len(df):,}행 {OUT / (final + '.parquet')}", flush=True)
    return df


# ---------------------------------------------------------------- 단계
def stage_meta():
    for name, sel in [
        ("stock_master_all", "SELECT code, ticker, name, market, industry, first_date, last_date, n_days, market_note FROM dataguide.stock_master"),
        ("stock_industry_history", "SELECT code, start_date, end_date, industry FROM dataguide.stock_industry_history"),
        ("item_dictionary", "SELECT market, table_name, column_name, dg_item_code, dg_item_name, description, source_file FROM dataguide.item_dictionary"),
    ]:
        tb = sel.split(" FROM ")[1]
        run_whole(f"meta_{name}_all", sel, f"SELECT COUNT(*) FROM {tb}")
    assemble("meta_stock_master_all", "stock_master_all", ["code"])
    assemble("meta_stock_industry_history", "stock_industry_history", ["code", "start_date"])
    assemble("meta_item_dictionary", "item_dictionary", ["market", "table_name", "column_name"])


KIWOOM = {   # 이름: (select 열, 테이블, 키, 정수열)
    "short_balance": ("code, date, close, volume, shorts_qty, shorts_ratio, shorts_balance_qty, shorts_amount, shorts_avg_price",
                      "kiwoom.short_balance", ["code", "date"], ["volume", "shorts_qty", "shorts_balance_qty", "shorts_amount"]),
    "program_stock": ("code, date, close, volume, prog_sell_amt, prog_buy_amt, prog_net_amt, prog_sell_qty, prog_buy_qty, prog_net_qty",
                      "kiwoom.program_stock", ["code", "date"],
                      ["volume", "prog_sell_amt", "prog_buy_amt", "prog_net_amt", "prog_sell_qty", "prog_buy_qty", "prog_net_qty"]),
    "program_market": ("date, market, arb_sell_amt, arb_buy_amt, arb_net_amt, arb_sell_qty, arb_buy_qty, arb_net_qty, nonarb_sell_amt, "
                       "nonarb_buy_amt, nonarb_net_amt, nonarb_sell_qty, nonarb_buy_qty, nonarb_net_qty",
                       "kiwoom.program_market", ["date", "market"],
                       ["arb_sell_amt", "arb_buy_amt", "arb_net_amt", "arb_sell_qty", "arb_buy_qty", "arb_net_qty", "nonarb_sell_amt",
                        "nonarb_buy_amt", "nonarb_net_amt", "nonarb_sell_qty", "nonarb_buy_qty", "nonarb_net_qty"]),
    "lending_market": ("date, contract_qty, repay_qty, net_qty, balance_qty, balance_amt", "kiwoom.lending_market", ["date"],
                       ["contract_qty", "repay_qty", "net_qty", "balance_qty", "balance_amt"]),
    "theme_member": ("date, theme_cd, code, theme_nm, theme_flu_rt, theme_stk_num, rising_cnt, falling_cnt, stock_flu_rt, stock_price, stock_volume",
                     "kiwoom.theme_member", ["date", "theme_cd", "code"], ["theme_stk_num", "rising_cnt", "falling_cnt", "stock_volume"]),
    "etf_master": ("code, ticker, name, index_name, mktcap_eok, list_shrs, first_date, last_date, n_days", "kiwoom.etf_master", ["code"],
                   ["mktcap_eok", "list_shrs", "n_days"]),
    "etf_daily": ("code, date, open, high, low, close, adj_close, volume, trading_value, nav, nav_dispty_rt, idx_dispty_rt, trace_eor_rt, idx_close",
                  "kiwoom.etf_daily", ["code", "date"], ["open", "high", "low", "close", "adj_close", "volume", "trading_value"]),
}


def stage_kiwoom():
    for name, (cols, tb, keys, ints) in KIWOOM.items():
        run_whole(f"kw_{name}_all", f"SELECT {cols} FROM {tb}", f"SELECT COUNT(*) FROM {tb}")
        assemble(f"kw_{name}", name, keys, ints)


def stage_inv_detail():
    cols = ("code, date, close, volume, indiv, foreign_net, foreign_etc, inst_total, fnnc_invt, insrnc, invtrt, etc_fnnc, bank, "
            "penfnd, samo_fund, natn, etc_corp")
    sel = f"SELECT {cols} FROM kiwoom.investor_detail WHERE unit = 'qty' AND date >= :a AND date <= :b"
    cnt = "SELECT COUNT(*) FROM kiwoom.investor_detail WHERE unit = 'qty' AND date >= :a AND date <= :b"
    for a, b in months(2026):
        if a >= "2026-05-01":
            run_range("kwinv", sel, cnt, a, b)
    ints = ["volume", "indiv", "foreign_net", "foreign_etc", "inst_total", "fnnc_invt", "insrnc", "invtrt", "etc_fnnc", "bank",
            "penfnd", "samo_fund", "natn", "etc_corp"]
    assemble("kwinv", "kiwoom_investor_detail", ["code", "date"], ints)


def stage_inv_kosdaq():
    cols = "i.code, i.date, i.indiv_buy, i.indiv_sell, i.indiv_net, i.inst_buy, i.inst_net, i.foreign_buy, i.foreign_net, i.other_corp_net"
    frm = ("FROM dataguide.daily_investor i JOIN dataguide.stock_master m ON m.code = i.code "
           "WHERE m.market = 'KOSDAQ' AND i.date >= :a AND i.date <= :b")
    ranges = list(years(2000, 2025)) + list(months(2026))
    for a, b in ranges:
        run_range("dginv_kosdaq", f"SELECT {cols} {frm}", f"SELECT COUNT(*) {frm}", a, b)
    ints = ["indiv_buy", "indiv_sell", "indiv_net", "inst_buy", "inst_net", "foreign_buy", "foreign_net", "other_corp_net"]
    assemble("dginv_kosdaq", "investor_kosdaq", ["code", "date"], ints)


def stage_shares():
    cols = "s.code, s.date, m.market, s.floating_shares, s.treasury_shares, s.short_balance_amt"
    frm = ("FROM dataguide.daily_shares s LEFT JOIN dataguide.stock_master m ON m.code = s.code "
           "WHERE s.date >= :a AND s.date <= :b")
    for a, b in quarters(2010):
        run_range("shares", f"SELECT {cols} {frm}", f"SELECT COUNT(*) {frm}", a, b)
    assemble("shares", "daily_shares", ["code", "date"], ["floating_shares", "treasury_shares"])


def stage_trading():
    cols = "t.code, t.date, t.market, t.sect_tp, t.trading_value, t.listed_shares"
    frm = ("FROM financials.daily_trading t JOIN dataguide.stock_master m ON m.code = t.code "
           "WHERE m.market = 'KOSDAQ' AND t.date >= :a AND t.date <= :b")
    for a, b in quarters(2010):
        run_range("trading_kosdaq", f"SELECT {cols} {frm}", f"SELECT COUNT(*) {frm}", a, b)
    assemble("trading_kosdaq", "trading_kosdaq", ["code", "date"], ["trading_value", "listed_shares"])


def stage_price():
    cols = "p.code, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume, p.listed_shares, p.is_halted"
    frm = ("FROM dataguide.daily_price p JOIN dataguide.stock_master m ON m.code = p.code "
           "WHERE m.market = 'KOSDAQ' AND p.date >= :a AND p.date <= :b")
    for a, b in quarters(2010):
        run_range("price_kosdaq", f"SELECT {cols} {frm}", f"SELECT COUNT(*) {frm}", a, b)
    assemble("price_kosdaq", "price_kosdaq", ["code", "date"], ["open", "high", "low", "close", "volume", "listed_shares", "is_halted"])


def stage_fs_probe():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 60)
    df = q("""SELECT f.sj_div, f.account_id, MIN(f.account_nm) nm, COUNT(*) n
              FROM financials.fs_line f JOIN dataguide.stock_master m ON m.code = f.code
              WHERE f.bsns_year = 2024 AND f.reprt_code = '11011' AND f.fs_div = 'CFS' AND m.market IN ('KOSPI', 'KOSDAQ')
              GROUP BY f.sj_div, f.account_id ORDER BY n DESC LIMIT 200""")
    print(df.to_string())
    print("연결 사업보고서 2024 회사 수:",
          q("""SELECT COUNT(DISTINCT f.code) n FROM financials.fs_line f JOIN dataguide.stock_master m ON m.code = f.code
               WHERE f.bsns_year = 2024 AND f.reprt_code = '11011' AND f.fs_div = 'CFS'""").iloc[0, 0])


def fs_piece(y, r, d, accts, suffix="", pref="fs", markets="'KOSPI', 'KOSDAQ'"):
    ids = {f"i{k}": a for k, a in enumerate(accts)}
    inlist = ", ".join(":" + k for k in ids)
    cols = ("f.code, m.market, f.bsns_year, f.reprt_code, f.fs_div, f.sj_div, f.account_id, f.account_nm, f.period_end, f.currency, "
            "f.thstrm_amount, f.frmtrm_amount, f.bfefrmtrm_amount")
    frm = ("FROM financials.fs_line f JOIN dataguide.stock_master m ON m.code = f.code "
           f"WHERE f.bsns_year = :y AND f.reprt_code = :r AND f.fs_div = :d AND m.market IN ({markets}) "
           f"AND f.account_id IN ({inlist})")
    name = f"{pref}_{y}_{r}_{d}{suffix}"
    try:
        run_piece(name, f"SELECT {cols} {frm}", f"SELECT COUNT(*) {frm}", {"y": y, "r": r, "d": d, **ids})
    except Lost:       # 한 조각이 너무 크면 account_id 목록을 절반씩 나눈다
        if len(accts) < 2:
            sys.exit(f"계정 1개 조각도 실패: {name}")
        h = len(accts) // 2
        print(f"  {name} 끊김 → 계정 목록 절반씩 분할", flush=True)
        time.sleep(60)
        fs_piece(y, r, d, accts[:h], suffix + "_p1", pref, markets)
        fs_piece(y, r, d, accts[h:], suffix + "_p2", pref, markets)


def stage_fs():
    for y in range(2015, 2027):
        for r in ("11013", "11012", "11014", "11011"):
            for d in ("CFS", "OFS"):
                fs_piece(y, r, d, CORE_ACCOUNTS)
    assemble("fs", "fs_line_core", ["code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_id"])


# 2019 상반기 이전 보고서는 IFRS 계정이 `ifrs_` 접두어다(이후는 `ifrs-full_`). 처음 목록이 `ifrs-full_` 만 담아 2019-06 이전 재무가 비었다
# (조정자 확인). 자산·부채·자본·당기순이익 4개는 다른 에이전트가 fs_line_core_pre2019.parquet 로 이미 받았으므로 나머지만 받는다.
IFRS_OLD_DONE = {"ifrs_Assets", "ifrs_Liabilities", "ifrs_Equity", "ifrs_ProfitLoss"}
IFRS_OLD = [a.replace("ifrs-full_", "ifrs_") for a in CORE_ACCOUNTS if a.startswith("ifrs-full_")]
IFRS_OLD = [a for a in IFRS_OLD if a not in IFRS_OLD_DONE]


def stage_fs_prefix_probe():
    """2015~2019 연결 보고서의 account_id 접두어 분포(회사 수 아님, 행 수). 어느 보고서까지 ifrs_ 인지, 표준 코드 없는 계정이 얼마인지 본다."""
    rows = []
    for y in range(2015, 2020):
        for r in ("11013", "11012", "11014", "11011"):
            df = q("""SELECT CASE WHEN LEFT(f.account_id, 10) = 'ifrs-full_' THEN 'ifrs-full_' WHEN LEFT(f.account_id, 5) = 'ifrs_' THEN 'ifrs_'
                             WHEN LEFT(f.account_id, 5) = 'dart_' THEN 'dart_' WHEN LEFT(f.account_id, 1) = '-' THEN 'nonstd(-)' ELSE 'other' END k,
                             COUNT(*) n, COUNT(DISTINCT f.code) nc
                      FROM financials.fs_line f JOIN dataguide.stock_master m ON m.code = f.code
                      WHERE f.bsns_year = :y AND f.reprt_code = :r AND f.fs_div = 'CFS' GROUP BY k""", {"y": y, "r": r})
            df.insert(0, "reprt", r); df.insert(0, "year", y)
            rows.append(df)
            print(df.to_string(header=False, index=False), flush=True)
    pd.concat(rows).to_csv(OUT / "_fs_prefix_probe.csv", index=False)


def stage_fs_ifrs():
    for y in range(2015, 2020):
        for r in ("11013", "11012", "11014", "11011"):
            if y == 2015 and r != "11011":
                continue                       # 2015 는 연간만 있다
            for d in ("CFS", "OFS"):
                fs_piece(y, r, d, IFRS_OLD, pref="fsold")
    assemble("fsold", "fs_line_core_pre2019_ext", ["code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_id"])


def stage_fs_ifrs4_kosdaq():
    """다른 에이전트의 fs_line_core_pre2019.parquet 는 코스피 589곳뿐이라 코스닥 자산·부채·자본·당기순이익이 비어 있다 → 코스닥만 같은 4계정을 받는다."""
    for y in range(2015, 2020):
        for r in ("11013", "11012", "11014", "11011"):
            if y == 2015 and r != "11011":
                continue
            for d in ("CFS", "OFS"):
                fs_piece(y, r, d, sorted(IFRS_OLD_DONE), pref="fsold4k", markets="'KOSDAQ'")
    assemble("fsold4k", "fs_line_core_pre2019_kosdaq4", ["code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_id"])


# ---------------------------------------------------------------- 검증
def stage_verify():
    """파일마다 (b) 키 중복 (c) 기간·종목 수·결측률 (d) 서버 직접 조회와 표본 대조 (a) 조각 COUNT 합계 대조."""
    rep = {}
    log = pd.read_csv(LOG)
    log["prefix"] = log.piece.str.replace(r"_(\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}|all)$", "", regex=True)
    log["prefix"] = log.prefix.str.replace(r"^fs_\d{4}_\d{5}_[A-Z]{3}(_p\d)*$", "fs", regex=True)
    log["prefix"] = log.prefix.str.replace(r"^fsold4k_\d{4}_\d{5}_[A-Z]{3}(_p\d)*$", "fsold4k", regex=True)
    log["prefix"] = log.prefix.str.replace(r"^fsold_\d{4}_\d{5}_[A-Z]{3}(_p\d)*$", "fsold", regex=True)
    log = log.drop_duplicates("piece", keep="last")
    spec = {   # 파일: (조각 접두어, 키, 날짜열, 서버 표본 조회 테이블, 키 조건에 쓸 열)
        "price_kosdaq": ("price_kosdaq", ["code", "date"], "date", "dataguide.daily_price"),
        "trading_kosdaq": ("trading_kosdaq", ["code", "date"], "date", "financials.daily_trading"),
        "daily_shares": ("shares", ["code", "date"], "date", "dataguide.daily_shares"),
        "investor_kosdaq": ("dginv_kosdaq", ["code", "date"], "date", "dataguide.daily_investor"),
        "kiwoom_investor_detail": ("kwinv", ["code", "date"], "date", "kiwoom.investor_detail"),
        "short_balance": ("kw_short_balance", ["code", "date"], "date", "kiwoom.short_balance"),
        "program_stock": ("kw_program_stock", ["code", "date"], "date", "kiwoom.program_stock"),
        "program_market": ("kw_program_market", ["date", "market"], "date", "kiwoom.program_market"),
        "lending_market": ("kw_lending_market", ["date"], "date", "kiwoom.lending_market"),
        "theme_member": ("kw_theme_member", ["date", "theme_cd", "code"], "date", "kiwoom.theme_member"),
        "etf_master": ("kw_etf_master", ["code"], None, "kiwoom.etf_master"),
        "etf_daily": ("kw_etf_daily", ["code", "date"], "date", "kiwoom.etf_daily"),
        "stock_master_all": ("meta_stock_master_all", ["code"], None, "dataguide.stock_master"),
        "stock_industry_history": ("meta_stock_industry_history", ["code", "start_date"], None, "dataguide.stock_industry_history"),
        "item_dictionary": ("meta_item_dictionary", ["market", "table_name", "column_name"], None, "dataguide.item_dictionary"),
        "fs_line_core": ("fs", ["code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_id"], None, "financials.fs_line"),
        "fs_line_core_pre2019_kosdaq4": ("fsold4k", ["code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_id"], None, "financials.fs_line"),
        "fs_line_core_pre2019_ext": ("fsold", ["code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_id"], None, "financials.fs_line"),
    }
    rng = np.random.default_rng(20261011)
    for fn, (pref, keys, dcol, tb) in spec.items():
        f = OUT / f"{fn}.parquet"
        if not f.exists():
            rep[fn] = {"missing": True}
            continue
        df = pd.read_parquet(f)
        sub = log[log.prefix == pref]
        r = {"rows": len(df), "size_mb": round(f.stat().st_size / 1e6, 1), "pieces": len(sub),
             "server_count_sum": int(sub.server_count.sum()), "count_match": bool(len(df) == int(sub.server_count.sum())),
             "dup_keys": int(df.duplicated(keys).sum()),
             "n_codes": int(df["code"].nunique()) if "code" in df.columns else None,
             "date_min": str(df[dcol].min().date()) if dcol and len(df) else None,
             "date_max": str(df[dcol].max().date()) if dcol and len(df) else None,
             "null_rate": {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0},
             "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
        # (d) 임의 표본 5건을 PK 로 서버에서 다시 읽어 값 대조
        bad = []
        n_s = min(5, len(df))
        for i in rng.choice(len(df), size=n_s, replace=False) if n_s else []:
            row = df.iloc[int(i)]
            cond = " AND ".join(f"{k} = :k{j}" for j, k in enumerate(keys))
            pr = {f"k{j}": (row[k].strftime("%Y-%m-%d") if isinstance(row[k], pd.Timestamp) else
                            (row[k].item() if hasattr(row[k], "item") else row[k])) for j, k in enumerate(keys)}
            extra = " AND unit = 'qty'" if tb.endswith("investor_detail") else ""
            sv = q(f"SELECT * FROM {tb} WHERE {cond}{extra}", pr)
            if len(sv) != 1:
                bad.append({"key": pr, "server_rows": len(sv)})
                continue
            sv = sv.iloc[0]
            for c in df.columns:
                if c not in sv.index or (c == "market" and tb.endswith("fs_line")):   # fs_line.market 은 한글 시장명 → 우리 파일은 stock_master 의 KOSPI/KOSDAQ
                    continue
                a, b = row[c], sv[c]
                if pd.isna(a) and pd.isna(b):
                    continue
                try:
                    same = abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
                except (TypeError, ValueError):
                    same = str(a)[:10] == str(b)[:10] if c in DATECOLS else str(a) == str(b)
                if not same:
                    bad.append({"key": pr, "col": c, "file": str(a), "server": str(b)})
        r["sample_checked"] = n_s
        r["sample_mismatch"] = bad
        rep[fn] = r
        print(fn, json.dumps({k: v for k, v in r.items() if k not in ("dtypes",)}, ensure_ascii=False, default=str), flush=True)
    (OUT / "_verify_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return rep


def stage_named():
    """이름 있는 표본 대조: 파일 값과 서버 직접 조회 값을 나란히 놓는다(삼성전자 2025 연결, 코스닥 대표 종목 2024-12-30 종가 등)."""
    out = {}
    fs = pd.read_parquet(OUT / "fs_line_core.parquet")
    for acc in ("ifrs-full_Revenue", "ifrs-full_Equity", "ifrs-full_Assets", "dart_OperatingIncomeLoss"):
        f = fs[(fs.code == "A005930") & (fs.bsns_year == 2025) & (fs.reprt_code == "11011") & (fs.fs_div == "CFS") & (fs.account_id == acc)]
        sv = q("SELECT thstrm_amount, frmtrm_amount FROM financials.fs_line WHERE code='A005930' AND bsns_year=2025 AND reprt_code='11011' "
               "AND fs_div='CFS' AND account_id=:a", {"a": acc})
        out[f"삼성전자 2025 연결 {acc}"] = {"file": float(f.thstrm_amount.iloc[0]), "server": float(sv.thstrm_amount.iloc[0]),
                                          "file_prev": float(f.frmtrm_amount.iloc[0]), "server_prev": float(sv.frmtrm_amount.iloc[0])}
    pr = pd.read_parquet(OUT / "price_kosdaq.parquet")
    tr = pd.read_parquet(OUT / "trading_kosdaq.parquet")
    for code, nm in (("A247540", "에코프로비엠"), ("A196170", "알테오젠"), ("A028300", "HLB")):
        for day in ("2024-12-30", "2026-10-08"):
            f = pr[(pr.code == code) & (pr.date == day)]
            sv = q("SELECT open, high, low, close, adj_close, volume, listed_shares FROM dataguide.daily_price WHERE code=:c AND date=:d", {"c": code, "d": day})
            t = tr[(tr.code == code) & (tr.date == day)]
            st = q("SELECT trading_value, listed_shares, sect_tp FROM financials.daily_trading WHERE code=:c AND date=:d", {"c": code, "d": day})
            out[f"{nm} {code} {day}"] = {
                "file_close": None if f.empty else int(f.close.iloc[0]), "server_close": None if sv.empty else int(sv.close.iloc[0]),
                "file_volume": None if f.empty else float(f.volume.iloc[0]), "server_volume": None if sv.empty else int(sv.volume.iloc[0]),
                "file_trading_value": None if t.empty else int(t.trading_value.iloc[0]), "server_trading_value": None if st.empty else int(st.trading_value.iloc[0]),
                "file_sect_tp": None if t.empty else t.sect_tp.iloc[0], "server_sect_tp": None if st.empty else st.sect_tp.iloc[0]}
    # 다른 에이전트가 받은 fs_line_core_pre2019.parquet (코스피만인지, 값이 서버와 같은지)
    ot = pd.read_parquet(OUT / "fs_line_core_pre2019.parquet")
    out["pre2019(타 에이전트) 시장"] = ot.market.value_counts().to_dict()
    out["pre2019(타 에이전트) 행/회사"] = [len(ot), int(ot.code.nunique())]
    bad = []
    for i in np.random.default_rng(7).choice(len(ot), 5, replace=False):
        r = ot.iloc[int(i)]
        sv = q("SELECT thstrm_amount FROM financials.fs_line WHERE code=:c AND bsns_year=:y AND reprt_code=:r AND fs_div=:d AND sj_div=:s AND account_id=:a",
               {"c": r.code, "y": int(r.bsns_year), "r": r.reprt_code, "d": r.fs_div, "s": r.sj_div, "a": r.account_id})
        if len(sv) != 1 or not (abs(float(sv.thstrm_amount.iloc[0]) - r.thstrm_amount) <= 1e-6 * max(1, abs(r.thstrm_amount)) or (pd.isna(r.thstrm_amount) and pd.isna(sv.thstrm_amount.iloc[0]))):
            bad.append(r.to_dict())
    out["pre2019(타 에이전트) 표본5 불일치"] = bad
    (OUT / "_named_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def stage_clean():
    import shutil
    shutil.rmtree(PARTS)
    print("지움:", PARTS)


STAGES = {"meta": stage_meta, "kiwoom": stage_kiwoom, "inv_detail": stage_inv_detail, "inv_kosdaq": stage_inv_kosdaq,
          "shares": stage_shares, "trading": stage_trading, "price": stage_price, "fs_probe": stage_fs_probe, "fs": stage_fs, "fs_prefix_probe": stage_fs_prefix_probe, "fs_ifrs": stage_fs_ifrs, "fs_ifrs4_kosdaq": stage_fs_ifrs4_kosdaq,
          "verify": stage_verify, "named": stage_named, "clean": stage_clean}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["meta", "kiwoom", "inv_detail", "inv_kosdaq", "shares", "trading", "price", "fs", "verify"]
    for s in todo:
        print(f"== {s} ==", flush=True)
        STAGES[s]()

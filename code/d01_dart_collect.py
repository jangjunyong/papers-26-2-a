"""d01_dart_collect.py - OpenDART list.json 공시 목록 수집 (KOSPI, 2015-01-01~현재). 재개 가능.

서브커맨드
  match      미매핑 KOSPI 종목을 corp_code.parquet 의 corp_name 과 정규화 매칭 (오프라인, API 호출 0)
  collect    종목(corp_code)별 전체 페이지를 받아 raw/{corp_code}.parquet 체크포인트로 저장 (이미 있으면 건너뜀)
  merge      raw/*.parquet -> disclosures_kospi.parquet
  typecheck  표본 종목에 pblntf_ty(A~J)를 지정한 호출 결과 저장 (규칙 분류 정확도 점검용, d02 가 읽음)
  status     호출 장부(ledger) 요약 (오늘/전체, 상태코드별)

키 취급: keyring('OpenDART','api_key') 에서만 읽는다. 요청 URL·키는 출력/로그/예외 메시지에 넣지 않는다
(requests 예외는 클래스명만 남기고 원본 예외는 버린다). 장부(raw/_ledger.csv)에는 시각·corp_code·유형·쪽·상태코드만 적는다.
한도: 호출 간격 MIN_GAP 초 이상, 하루 DAILY_CAP 건 이하, 012/020/HTTP 429 는 즉시 중단.
"""
import os, sys, re, time, argparse, datetime as dt
import requests, keyring, pandas as pd

# ------------------------------------------------------------------ 경로 상수
V = r'C:\Users\jacob\Documents\Obsidian Vault\PAPERS'
D = V + r'\data\dart'
RAW = D + r'\raw'                       # 종목별 체크포인트 {corp_code}.parquet
RAW_TY = D + r'\raw_ty'                 # pblntf_ty 지정 표본 호출 결과 {corp_code}.parquet
LEDGER = RAW + r'\_ledger.csv'          # 호출 장부(URL/키 없음)
MANIFEST = RAW + r'\_manifest.csv'      # 종목별 total_count / 수집 행수
MAP_PQ = D + r'\ticker_corp_map.parquet'
CORP_PQ = D + r'\corp_code.parquet'
COUNTS_PQ = D + r'\disclosure_counts_kospi.parquet'   # 백그라운드 견적(있으면 사용)
MAP_ALL_PQ = D + r'\corp_map_kospi_all.parquet'       # 매핑 확정본(기존 매핑 + 이름 매칭)
UNMAPPED_CSV = D + r'\unmapped_kospi.csv'
OUT_PQ = D + r'\disclosures_kospi.parquet'

# ------------------------------------------------------------------ 호출 설정
URL = 'https://opendart.fss.or.kr/api/list.json'
BGN = '20150101'
MIN_GAP = 0.3          # 초. 문서에 분당 한도가 없어 보수적으로
DAILY_CAP = 15000      # 하루 총 호출 상한 (공식 20,000 의 75%)
STOP_STATUS = {'010', '011', '012', '020', '021', '100', '101', '800', '901'}
COLS = ['corp_code', 'corp_cls', 'stock_code', 'corp_name', 'report_nm', 'rcept_no', 'flr_nm', 'rcept_dt', 'rm']


class Stop(Exception):
    """한도/차단/점검/예상 밖 상태코드. 체크포인트는 보존하고 즉시 종료."""


class Api:
    def __init__(self, preused=0, cap=DAILY_CAP):
        k = keyring.get_password('OpenDART', 'api_key')
        if not k:
            raise Stop('keyring 에 OpenDART api_key 없음')
        self._k = k
        self.s = requests.Session()
        self.cap = cap
        self.today = dt.date.today().isoformat()
        os.makedirs(RAW, exist_ok=True)
        if not os.path.exists(LEDGER):
            with open(LEDGER, 'w', encoding='utf-8') as f:
                f.write('ts,date,corp_code,pblntf_ty,page,http,status\n')
        led = pd.read_csv(LEDGER, dtype=str)
        self.used = int((led['date'] == self.today).sum()) + preused    # 오늘 이미 쓴 호출(타 작업 preused 포함)
        self.session_calls = 0
        self._last = 0.0

    def _log(self, corp, ty, page, http, status):
        now = dt.datetime.now()
        with open(LEDGER, 'a', encoding='utf-8') as f:
            f.write(f'{now:%Y-%m-%dT%H:%M:%S},{now:%Y-%m-%d},{corp},{ty or ""},{page},{http},{status}\n')

    def call(self, corp, page=1, ty=None, bgn=BGN, end=None):
        if self.used >= self.cap:
            raise Stop(f'일일 상한 {self.cap} 도달')
        p = dict(crtfc_key=self._k, corp_code=corp, bgn_de=bgn, end_de=end or dt.date.today().strftime('%Y%m%d'),
                 page_no=str(page), page_count='100')
        if ty:
            p['pblntf_ty'] = ty
        for attempt in range(3):
            wait = MIN_GAP - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.used += 1
            self.session_calls += 1
            try:
                r = self.s.get(URL, params=p, timeout=40)
            except requests.RequestException as e:
                self._log(corp, ty, page, 'EXC_' + type(e).__name__, '')
                if attempt == 2:
                    raise Stop('네트워크 오류 반복: ' + type(e).__name__) from None   # 원본 예외(URL 포함 가능)는 버린다
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code == 429:
                self._log(corp, ty, page, 429, '')
                raise Stop('HTTP 429')
            if r.status_code != 200:
                self._log(corp, ty, page, r.status_code, '')
                if attempt == 2 or r.status_code < 500:
                    raise Stop(f'HTTP {r.status_code}')
                time.sleep(3 * (attempt + 1))
                continue
            try:
                j = r.json()
            except ValueError:
                self._log(corp, ty, page, 200, 'BADJSON')
                raise Stop('JSON 파싱 실패') from None
            st = str(j.get('status'))
            self._log(corp, ty, page, 200, st)
            if st in STOP_STATUS:
                raise Stop(f'status {st} {j.get("message", "")}')
            if st == '900' and attempt < 2:
                time.sleep(5)
                continue
            if st not in ('000', '013'):
                raise Stop(f'status {st} {j.get("message", "")}')
            return j
        raise Stop('재시도 소진')


def fetch_all(api, corp, ty=None):
    """corp_code 의 2015~현재 공시 전부(모든 쪽). (DataFrame, total_count) 반환."""
    rows, page, total = [], 1, 0
    while True:
        j = api.call(corp, page, ty)
        if j['status'] == '013':
            break
        total = int(j['total_count'])
        rows += j['list']
        if page >= int(j['total_page']):
            break
        page += 1
    df = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
    return df.drop_duplicates('rcept_no'), total


def save_atomic(df, path):
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ------------------------------------------------------------------ match: 미매핑 이름 매칭
def norm_name(s):
    """괄호 안 전부((주)·(구) 등)·주식회사·공백·구두점 제거, 대문자화."""
    if s is None:
        return ''
    s = re.sub(r'\([^)]*\)', '', str(s))
    s = s.replace('㈜', '').replace('주식회사', '').replace('유한회사', '')
    return re.sub(r'[\s.,·ㆍ&\-_/]', '', s).upper()


def cmd_match(a):
    m = pd.read_parquet(MAP_PQ)
    c = pd.read_parquet(CORP_PQ)
    k = m[m.market == 'KOSPI'].copy()
    c['n'] = c.corp_name.map(norm_name)
    idx = c.groupby('n').corp_code.apply(list).to_dict()
    cstock = dict(zip(c.corp_code, c.stock_code.fillna('').str.strip()))
    cname = dict(zip(c.corp_code, c.corp_name))
    sm = pd.read_parquet(V + r'\data\daily\club_db\stock_master.parquet')
    nd = dict(zip(sm.ticker, sm.n_days))        # 가격 행수 0 인 종목은 연구에 쓸 수 없음
    out = []
    for r in k[~k.mapped].itertuples():
        cand = idx.get(norm_name(r.name), [])
        out_of_scope = (r.delisted and str(r.last_date) < '2015-01-01')
        res = dict(ticker=r.ticker, name=r.name, first_date=r.first_date, last_date=r.last_date, delisted=r.delisted,
                   n_days=nd.get(r.ticker), n_cand=len(cand), corp_code=None, corp_name=None, stock_code_in_dart=None,
                   status='', accepted=False)
        if out_of_scope:
            res['status'] = '2015 이전 폐지(수집 범위 밖)'
        elif len(cand) == 0:
            res['status'] = '이름 일치 없음'
        elif len(cand) > 1:
            res['status'] = f'동명 {len(cand)}건(불확실)'
        else:
            cc = cand[0]
            res.update(corp_code=cc, corp_name=cname[cc], stock_code_in_dart=cstock[cc])
            # 확실한 것만: 이름이 유일하고 DART 종목코드가 이 ticker 와 같을 때만 채택.
            # 종목코드가 공란인 유일 이름은 동명 비상장사일 수 있어 '후보'로만 남긴다(미채택).
            if cstock[cc] == r.ticker:
                res['accepted'] = True
                res['status'] = '이름 유일 일치 + 종목코드 일치'
            elif cstock[cc]:
                res['status'] = f'이름 일치하나 DART 종목코드 {cstock[cc]} 가 다름(기각)'
            else:
                res['status'] = '후보: 이름 유일 일치하나 DART 종목코드 공란(미채택)'
        out.append(res)
    u = pd.DataFrame(out)
    # 채택된 corp_code 가 기존 매핑 종목과 겹치면 기각
    used = set(k[k.mapped].corp_code)
    dup = u.accepted & u.corp_code.isin(used)
    u.loc[dup, ['accepted', 'status']] = [False, '기존 매핑 종목과 corp_code 중복(기각)']
    dup2 = u.accepted & u.corp_code.duplicated(keep=False)
    u.loc[dup2, ['accepted', 'status']] = [False, '미매핑끼리 corp_code 중복(기각)']
    u.to_csv(UNMAPPED_CSV, index=False, encoding='utf-8-sig')
    base = k[k.mapped][['ticker', 'name', 'corp_code', 'first_date', 'last_date', 'delisted']].copy()
    base['source'] = 'stock_code'
    add = u[u.accepted][['ticker', 'name', 'corp_code', 'first_date', 'last_date', 'delisted']].copy()
    add['source'] = 'name_match'
    allm = pd.concat([base, add], ignore_index=True)
    allm.to_parquet(MAP_ALL_PQ, index=False)
    print('KOSPI', len(k), '기존 매핑', len(base), '이름 매칭 채택', len(add), '남은 미매핑', len(u) - len(add))
    print(u.status.str.replace(r'\d+건', 'N건', regex=True).value_counts().to_string())


# ------------------------------------------------------------------ collect
def load_targets():
    if os.path.exists(MAP_ALL_PQ):
        return pd.read_parquet(MAP_ALL_PQ)
    m = pd.read_parquet(MAP_PQ)
    k = m[(m.market == 'KOSPI') & m.mapped][['ticker', 'name', 'corp_code', 'first_date', 'last_date', 'delisted']].copy()
    k['source'] = 'stock_code'
    return k


def est_pages(t):
    """견적 쪽수 (counts 파일 있으면 사용, 없으면 모름 -> 1 로 시작)."""
    if os.path.exists(COUNTS_PQ):
        c = pd.read_parquet(COUNTS_PQ)
        c = c[c.status == '000']
        return dict(zip(c.corp_code, ((c.total_2015_ + 99) // 100).clip(lower=1)))
    return {}


def cmd_collect(a):
    os.makedirs(RAW, exist_ok=True)
    t = load_targets()
    ep = est_pages(t)
    t['est'] = t.corp_code.map(ep).fillna(1).astype(int)
    # 우선순위: 쪽수 적은(=한도 안에 확실히 들어가는) 종목부터가 아니라, 입력 순서(ticker) 그대로. 한도 초과 시 --order small 사용
    if a.order == 'small':
        t = t.sort_values('est')
    # 견적 단계에서 013(2015~ 공시 없음)이 확인된 종목은 호출 없이 빈 체크포인트만 만든다
    zero = set()
    if os.path.exists(COUNTS_PQ):
        c = pd.read_parquet(COUNTS_PQ)
        zero = set(c[c.status == '013'].corp_code)
    for r in t.itertuples():
        if r.corp_code in zero and not os.path.exists(f'{RAW}\\{r.corp_code}.parquet'):
            e = pd.DataFrame(columns=COLS)
            e.insert(0, 'ticker', pd.Series(dtype=object))
            save_atomic(e, f'{RAW}\\{r.corp_code}.parquet')
    todo = [r for r in t.itertuples() if not os.path.exists(f'{RAW}\\{r.corp_code}.parquet')]
    print(f'대상 {len(t)} 종목, 완료 {len(t) - len(todo)}, 남음 {len(todo)}, 남은 견적 쪽수 {sum(r.est for r in todo)}')
    api = Api(preused=a.preused, cap=a.cap)
    print(f'오늘 사용(장부+preused) {api.used}, 상한 {api.cap}')
    done = 0
    try:
        for r in todo:
            if api.used + r.est > api.cap:      # 이 종목이 상한을 넘길 것 같으면 건너뛴다 (작은 종목은 계속 시도)
                print(f'skip {r.ticker} est={r.est} (잔여 {api.cap - api.used})')
                continue
            df, total = fetch_all(api, r.corp_code)
            df.insert(0, 'ticker', r.ticker)
            save_atomic(df, f'{RAW}\\{r.corp_code}.parquet')
            with open(MANIFEST, 'a', encoding='utf-8') as f:
                if f.tell() == 0:
                    f.write('ts,ticker,corp_code,total_count,n_rows\n')
                f.write(f'{dt.datetime.now():%Y-%m-%dT%H:%M:%S},{r.ticker},{r.corp_code},{total},{len(df)}\n')
            done += 1
            if done % 25 == 0:
                print(f'{done}/{len(todo)} 완료, 오늘 호출 {api.used}', flush=True)
    except Stop as e:
        print('STOP:', e, '| 이번 실행 호출', api.session_calls, '| 오늘 누계', api.used)
        sys.exit(2)
    print(f'종료: 이번 실행 {done} 종목, 호출 {api.session_calls}, 오늘 누계 {api.used}')


# ------------------------------------------------------------------ merge
def cmd_merge(a):
    t = load_targets()
    t2c = dict(zip(t.corp_code, t.ticker))
    parts = []
    for f in os.listdir(RAW):
        if f.endswith('.parquet') and not f.startswith('_'):
            parts.append(pd.read_parquet(f'{RAW}\\{f}'))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df['ticker'] = df.corp_code.map(t2c)
    df['rcept_dt'] = pd.to_datetime(df.rcept_dt, format='%Y%m%d').dt.date
    df = df[['ticker', 'corp_code', 'rcept_no', 'rcept_dt', 'report_nm', 'flr_nm', 'rm']].drop_duplicates('rcept_no')
    df = df.sort_values(['ticker', 'rcept_dt', 'rcept_no']).reset_index(drop=True)
    df.to_parquet(OUT_PQ, index=False)
    print(f'{OUT_PQ}: {len(df)} 행, {df.ticker.nunique()} 종목, {df.rcept_dt.min()} ~ {df.rcept_dt.max()}')


# ------------------------------------------------------------------ typecheck
def cmd_typecheck(a):
    """표본 N종목에 대해 pblntf_ty A~J 각각을 지정해 호출 -> raw_ty/{corp}.parquet (pblntf_ty 열 포함)."""
    out_dir = f'{RAW_TY}\\{a.tag}'                       # 표본 묶음(A=규칙 설계 점검, B=규칙 수정 후 보류 표본)
    os.makedirs(out_dir, exist_ok=True)
    m = pd.read_parquet(OUT_PQ)
    cnt = m.groupby(['ticker', 'corp_code']).size().reset_index(name='n')
    cnt = cnt[(cnt.n >= 30) & (cnt.n <= a.max_n)]       # 쪽수 폭주 방지
    done = {f[:-8] for t in os.listdir(RAW_TY) if os.path.isdir(f'{RAW_TY}\\{t}') and t != a.tag
            for f in os.listdir(f'{RAW_TY}\\{t}') if f.endswith('.parquet')}
    cnt = cnt[~cnt.corp_code.isin(done)]                # 다른 묶음에 쓴 종목 제외
    tk = pd.read_parquet(MAP_PQ).set_index('ticker')
    cnt['delisted'] = cnt.ticker.map(tk.delisted)
    # 상장 2/3 + 폐지 1/3 의 층화 표본 (고정 시드)
    s = pd.concat([cnt[~cnt.delisted].sample(min(a.n * 2 // 3, (~cnt.delisted).sum()), random_state=a.seed),
                   cnt[cnt.delisted].sample(min(a.n - a.n * 2 // 3, cnt.delisted.sum()), random_state=a.seed)])
    api = Api(preused=a.preused, cap=a.cap)
    print(f'표본 {a.tag}: {len(s)} 종목 (상장 {(~s.delisted).sum()}, 폐지 {s.delisted.sum()}), 오늘 사용 {api.used}')
    try:
        for r in s.itertuples():
            p = f'{out_dir}\\{r.corp_code}.parquet'
            if os.path.exists(p):
                continue
            parts = []
            for ty in 'ABCDEFGHIJ':
                df, _ = fetch_all(api, r.corp_code, ty)
                df['pblntf_ty'] = ty
                parts.append(df)
            out = pd.concat(parts, ignore_index=True)
            out.insert(0, 'ticker', r.ticker)
            save_atomic(out, p)
            print(r.ticker, len(out), '행, 오늘 호출', api.used, flush=True)
    except Stop as e:
        print('STOP:', e, '| 오늘 누계', api.used)
        sys.exit(2)
    print('typecheck 종료, 이번 실행 호출', api.session_calls)


def cmd_status(a):
    led = pd.read_csv(LEDGER, dtype=str)
    print('전체 호출', len(led))
    print(led.groupby('date').size().to_string())
    print(led.groupby(['http', 'status'], dropna=False).size().to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['match', 'collect', 'merge', 'typecheck', 'status'])
    ap.add_argument('--preused', type=int, default=0, help='장부에 없는 오늘 사용 호출(다른 작업분) 가산')
    ap.add_argument('--cap', type=int, default=DAILY_CAP)
    ap.add_argument('--order', default='ticker', choices=['ticker', 'small'])
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--tag', default='A', help='typecheck 표본 묶음 이름 (raw_ty/<tag>/)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-n', type=int, default=1500)
    a = ap.parse_args()
    {'match': cmd_match, 'collect': cmd_collect, 'merge': cmd_merge, 'typecheck': cmd_typecheck, 'status': cmd_status}[a.cmd](a)

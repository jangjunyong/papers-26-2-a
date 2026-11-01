"""d02_dart_derive.py - d01 이 모은 공시 목록(disclosures_kospi.parquet)에서 파생 표를 만든다. API 호출 없음(오프라인).

서브커맨드
  classify  보고서명 규칙 분류 -> disclosures_kospi_class.parquet (cat, rule, is_amend, amend_kind)
  fs        재무제표 시점 정합 표 -> fs_available_date.parquet   (+ 동아리 DB fs_line 과 키 대조, 읽기 전용)
  daily     (ticker, 영업일) 이벤트 표 -> disclosure_events_daily.parquet
  accuracy  d01 typecheck 표본(raw_ty/)으로 규칙 분류 정확도 계산
  stats     연도별 건수·커버리지·빈도 상위 보고서명 출력
  all       classify -> fs -> daily -> accuracy -> stats
"""
import os, re, sys, time, argparse
import numpy as np, pandas as pd

# ------------------------------------------------------------------ 경로 상수
V = r'C:\Users\jacob\Documents\Obsidian Vault\PAPERS'
D = V + r'\data\dart'
DISC = D + r'\disclosures_kospi.parquet'
CLS = D + r'\disclosures_kospi_class.parquet'
FS_OUT = D + r'\fs_available_date.parquet'
EV_OUT = D + r'\disclosure_events_daily.parquet'
FS_KEYS = D + r'\_fs_keys_cache.parquet'          # DB 키 대조용 캐시(재무 값 없음, 키만)
RAW_TY = D + r'\raw_ty'
PANEL = V + r'\data\daily\club_db\panel.parquet'
MAP_PQ = D + r'\ticker_corp_map.parquet'
STOCK_MASTER = V + r'\data\daily\club_db\stock_master.parquet'

# ------------------------------------------------------------------ 분류 규칙
AMEND_PREFIX = {'기재정정', '첨부정정', '첨부추가', '정정명령부과', '정정제출요구', '변경등록'}   # 이 접두어면 정정 공시
# (cat, rule_label, regex).  보고서명에서 [..] 접두어와 모든 공백을 지운 문자열에 re.search. 위에서부터 첫 일치.
# cat: 정기 / 주요사항 / 지분 / 수시공정 / 기타.  기타의 rule_label 앞 글자(C,E,F,J)는 DART 공시유형 추정.
RULES = [
    ('정기', 'A:사업·반기·분기보고서', r'^(사업|반기|분기)보고서(?!제출)'),
    ('정기', 'A:제출기한연장신고서·소액공모법인결산서류', r'^(사업|반기|분기)보고서제출기한연장신고서|^소액공모법인결산서류'),   # 표본 B 점검 후 추가 (A 유형으로 확인됨)
    ('수시공정', 'I:자회사·종속회사 주요경영사항', r'\((자회사|종속회사)의주요경영사항\)'),
    ('주요사항', 'B:주요사항보고서', r'^주요사항보고서'),
    ('지분', 'D:대량보유', r'^주식등의대량보유상황보고서'),
    ('지분', 'D:임원·주요주주 소유상황', r'^임원ㆍ주요주주특정증권등소유상황보고서'),
    ('지분', 'D:임원·주요주주 거래계획', r'^임원ㆍ주요주주특정증권등거래계획'),
    ('지분', 'D:공개매수', r'^공개매수'),
    ('지분', 'D:의결권대리행사권유', r'^의결권대리행사권유'),
    # --- 기타(비거래소) 유형: 발행·외부감사·공정위·기타공시. 거래소 규칙보다 먼저 걸러야 '자기주식' 등과 안 섞인다
    ('기타', 'E:자기주식 결과보고서', r'^자기주식(취득|처분)결과보고서'),
    ('기타', 'E:신탁계약·합병등종료·주식매수선택권 등', r'^(신탁계약|합병등종료보고서|주식매수선택권부여|독립사외이사에관한신고|주주총회소집보고서|시장조성|안정조작|이중상환청구권부채권)'),
    ('기타', 'F:감사보고서', r'^(연결|결합)?감사보고서(?!제출)'),
    ('기타', 'C:증권신고·투자설명서·발행', r'^(증권신고서|투자설명서|일괄신고|증권발행실적보고서|소액공모|발행사실보고서|철회신고서|증권예탁증권|정정신고서|채무증권)'),
    ('기타', 'J:공정위 공시', r'^(대규모기업집단현황공시|동일인등출자계열회사와의상품ㆍ용역거래|특수관계인|계열금융회사|약관에의한금융거래|지급수단별ㆍ지급기간별지급금액|임원의변동|최대주주등의주식보유변동)'),
    ('기타', 'E:주주총회소집공고·독립이사 신고', r'^(주주총회소집공고|독립이사의선임)'),     # 표본 A 점검 후 추가 (E 유형으로 확인됨)
    ('기타', 'C/E:기타(추정)', r'^(해외증권거래소등에신고한|효력발생안내|정정명령부과)'),      # 유형 미확인, 빈도 낮음
    # --- 거래소 수시·공정·안내공시
    ('수시공정', 'I:단일판매·공급계약', r'^단일판매'),
    ('수시공정', 'I:영업(잠정)실적·전망', r'영업\(잠정\)실적|^영업실적등에대한전망|^연결재무제표기준영업'),
    ('수시공정', 'I:매출액·손익구조 변동', r'^매출액또는손익구조'),
    ('수시공정', 'I:배당·주주명부폐쇄', r'^(현금ㆍ현물배당|주식배당)결정|^현금ㆍ현물배당을위한|^주주명부폐쇄'),
    ('수시공정', 'I:유·무상증자', r'^(유상증자|무상증자|유무상증자)(결정|신주)|^유상증자또는주식관련사채'),
    ('수시공정', 'I:자기주식(결정)', r'^자기주식(취득|처분)결정|^자기주식취득신탁|^자기(전환|교환|신주인수권부)사채|^주권관련사채권'),
    ('수시공정', 'I:최대주주·주요주주 변경', r'^(최대주주|주요주주)(변경|등소유주식변동|등의주식보유변동)'),
    ('수시공정', 'I:주주총회', r'^(정기|임시)?주주총회(소집결의|소집공고|결과|집중일)|^(정기|임시)주주총회결과'),
    ('수시공정', 'I:합병·분할·교환', r'^(회사)?(합병|분할|분할합병)결정|^주식교환ㆍ이전결정|^영업(양수|양도)결정|^중요한(자산|영업)양수도'),
    ('수시공정', 'I:자본(감자·병합·분할·소각)', r'^(감자|주식병합|주식분할|주식소각)결정'),
    ('수시공정', 'I:사채 발행·전환', r'^(전환사채|신주인수권부사채|교환사채|상각형조건부자본증권)권?발행결정|^전환가액|^전환청구권'),
    ('수시공정', 'I:타법인·시설·자산 투자', r'^(타법인주식및출자증권|타인에대한|타인을위한|신규시설투자등|시설외투자등|유형자산|비유동자산|선급금|채무인수|금전대여|단기차입금)'),
    ('수시공정', 'I:대표이사·임원·감사 변동', r'^(대표이사|감사ㆍ감사위원회)'),
    ('수시공정', 'I:조회공시·시장조치·투자유의', r'^(조회공시|풍문또는보도|불성실공시|매매거래정지|투자유의안내|기타시장안내|상장폐지|관리종목|채권상장법인불성실)'),
    ('수시공정', 'I:위험·사고(횡령·소송·회생·부도 등)', r'^(횡령|배임|소송|벌금|회생|부도|해산사유|영업정지|생산중단|생산재개|거래처와의거래중단|파생상품거래손실|매출채권|자본잠식|사채원리금|대출원리금|피보증|출자법인|중대재해|제품에대한|증권관련집단소송|채권은행|회계처리기준위반|상장채권|반기검토|감사의견|재해발생)'),
    ('수시공정', 'I:기타경영·투자판단·안내', r'^(기타경영사항|기타주요경영사항|기타안내사항|투자판단관련주요경영사항|감사보고서제출|기업지배구조보고서공시|기업설명회|결산실적공시예고|지속가능경영|특허권|자산재평가|가족친화|녹색|온실가스|정보보호|지주회사|계열회사변경|지배회사|종류주식|수시공시의무|장래사업)'),
    ('수시공정', 'I:자율·공정·안내공시 접미', r'\((자율공시|공정공시|안내공시)'),
    # 표본 A 점검 후 추가: 미분류 상위 이름. 기업인수목적회사·액면주식전환은 I 유형으로 확인, 나머지는 거래소 공시로 '추정'
    ('수시공정', 'I:기타 거래소공시(일부 추정)', r'^(기업인수목적회사|액면주식의무액면주식전환결정|경영정상화계획의이행약정체결|증권대여결정|소속부변경|사채권자집회|신주인수권행사|채무면제결정|파산신청|유상증자.*발행가액|유상증자최종발행가액|권리락|상호변경안내|증여결정|본점소재지변경|주식매수선택권행사|주권매매거래정지|전환사채\(해외전환사채포함\)|계열사주식보유변동|상장외국법인|유가증권시장상장외국법인|감사전재무제표기한내미제출|공정거래자율준수프로그램운영현황|최대주주인명목회사)'),
]
RULES_C = [(c, l, re.compile(p)) for c, l, p in RULES]


def parse_name(nm):
    """report_nm -> (접두어 리스트, 매칭용 본문(공백 제거))."""
    s = (nm or '').strip()
    m = re.match(r'^((?:\[[^\]]+\])+)', s)
    pre = re.findall(r'\[([^\]]+)\]', m.group(1)) if m else []
    body = s[m.end():] if m else s
    return pre, re.sub(r'\s+', '', body)


def classify_name(nm):
    pre, key = parse_name(nm)
    kinds = [p for p in pre if p in AMEND_PREFIX]
    cat, rule = '기타', '미분류'
    for c, l, rx in RULES_C:
        if rx.search(key):
            cat, rule = c, l
            break
    return cat, rule, bool(kinds), (kinds[0] if kinds else ('' if not pre else pre[0]))


def cmd_classify(a=None):
    d = pd.read_parquet(DISC)
    names = pd.Series(d.report_nm.unique())
    t = pd.DataFrame([classify_name(n) for n in names], columns=['cat', 'rule', 'is_amend', 'prefix'])
    t.insert(0, 'report_nm', names.values)
    d = d.merge(t, on='report_nm', how='left')
    d['ftc'] = d.rm.fillna('').str.contains('공')      # rm '공' = 공정위 소관 (표본에서 pblntf_ty=J 와 일치). 이름 규칙으로 못 잡는 J 를 걸러낼 때 사용
    d.to_parquet(CLS, index=False)
    print(f'{CLS}: {len(d)} 행, 고유 보고서명 {len(t)}')
    print(d.cat.value_counts().to_string())
    print('미분류(rule=미분류) 비율: %.2f%% (%d 건)' % (100 * (d.rule == '미분류').mean(), (d.rule == '미분류').sum()))
    print('정정 공시 비율: %.2f%%' % (100 * d.is_amend.mean()))


# ------------------------------------------------------------------ 작업 3: fs_available_date
PERIODIC = re.compile(r'^(사업|반기|분기)보고서(?!제출)')
PERIOD = re.compile(r'^(사업|반기|분기)보고서\((\d{4})\.(\d{1,2})\)')
MONTHS = {'11011': 12, '11012': 6, '11013': 3, '11014': 9}


def build_fs(d):
    """d: disclosures_kospi_class (is_amend, prefix 포함). 반환 (표, 진단 dict)."""
    p = d.copy()
    pk = p.report_nm.map(parse_name)
    p['key'] = pk.map(lambda x: x[1])
    p = p[p.key.str.match(PERIODIC)].copy()
    n_like = len(p)
    m = p.key.str.extract(PERIOD)
    p['kind'], p['py'], p['pm'] = m[0], pd.to_numeric(m[1]), pd.to_numeric(m[2])
    bad = p[p.kind.isna()]
    diag = {'periodic_like_rows': n_like, 'unparsed_rows': len(bad), 'unparsed_examples': bad.report_nm.drop_duplicates().head(8).tolist()}
    p = p[p.kind.notna()].copy()
    p['pm'] = p.pm.astype(int)
    p['py'] = p.py.astype(int)
    # 결산월: 종목별 사업보고서 기간말 월의 최빈값, 없으면 반기보고서 월 + 6
    fy = p[p.kind == '사업'].groupby('ticker').pm.agg(lambda s: s.mode().iloc[0])
    hy = p[p.kind == '반기'].groupby('ticker').pm.agg(lambda s: (s.mode().iloc[0] + 5) % 12 + 1)
    fm = fy.combine_first(hy)
    p['fm'] = p.ticker.map(fm)
    p['reprt_code'] = np.select([p.kind == '사업', p.kind == '반기'], ['11011', '11012'], default='')
    q = p.kind == '분기'
    # 분기보고서: 같은 종목의 '다음 사업보고서 기간말'까지 남은 개월 수 9 -> 1분기(11013), 3 -> 3분기(11014)
    # (결산월을 바꾼 종목도 맞도록). 다음 사업보고서가 아직 없으면 직전 사업보고서 기간말로부터 3개월 -> 1분기 / 9개월 -> 3분기, 그것도 안 되면 종목별 최빈 결산월 규칙.
    p['P'] = p.py * 12 + p.pm - 1
    ann = p.loc[p.kind == '사업', ['ticker', 'P']].rename(columns={'P': 'A'}).drop_duplicates().sort_values('A')
    qq = p.loc[q, ['ticker', 'P']].sort_values('P')
    nxt = pd.merge_asof(qq.reset_index(), ann, left_on='P', right_on='A', by='ticker', direction='forward', allow_exact_matches=False).set_index('index')
    left = (nxt.A - nxt.P).reindex(p.index)
    delta = (p.pm - p.fm) % 12            # 대체 규칙: 결산월 + 3 -> 1Q, + 9 -> 3Q
    p.loc[q & (left == 9), 'reprt_code'] = '11013'
    p.loc[q & (left == 3), 'reprt_code'] = '11014'
    prv = pd.merge_asof(qq.reset_index(), ann, left_on='P', right_on='A', by='ticker', direction='backward', allow_exact_matches=False).set_index('index')
    back = (prv.P - prv.A).reindex(p.index)      # 직전 사업보고서 기간말로부터 지난 개월 수: 3 -> 1분기, 9 -> 3분기
    fb = q & (p.reprt_code == '')
    p.loc[fb & (back == 3), 'reprt_code'] = '11013'
    p.loc[fb & (back == 9), 'reprt_code'] = '11014'
    fb = q & (p.reprt_code == '')
    p.loc[fb & (delta == 3), 'reprt_code'] = '11013'
    p.loc[fb & (delta == 9), 'reprt_code'] = '11014'
    amb = p[(p.reprt_code == '')]
    diag['quarter_ambiguous_rows'] = len(amb)
    diag['quarter_ambiguous_examples'] = amb[['ticker', 'report_nm']].head(5).values.tolist()
    p = p[p.reprt_code != ''].copy()
    mon = p.reprt_code.map(MONTHS)
    idx = p.py * 12 + p.pm - 1
    # bsns_year 규약 후보 3개 (동아리 DB 키와 대조해 cmd_fs 가 고른다). 12월 결산이면 셋 다 같다.
    p['by_fyend'] = (idx + (12 - mon)) // 12          # 회계연도가 끝나는 해
    p['by_pe'] = p.py                                 # 기간말이 속한 해
    p['by_fystart'] = (idx - (mon - 1)) // 12         # 회계연도가 시작하는 해
    p['period_end'] = (pd.to_datetime(dict(year=p.py, month=p.pm, day=1)) + pd.offsets.MonthEnd(0)).dt.date
    p['rcept_dt'] = pd.to_datetime(p.rcept_dt)
    g = p.groupby(['ticker', 'reprt_code', 'period_end', 'by_fyend', 'by_pe', 'by_fystart'])
    out = g.apply(lambda x: pd.Series({
        'first_rcept_dt': (x.loc[~x.is_amend, 'rcept_dt'].min() if (~x.is_amend).any() else x.rcept_dt.min()),
        'last_rcept_dt': x.rcept_dt.max(),
        'n_amend': int(x.is_amend.sum()),
        'orig_found': bool((~x.is_amend).any()),     # False 면 원공시가 수집 범위(2015~) 밖이라 first_rcept_dt 가 정정공시 날짜
    }), include_groups=False).reset_index()
    out['first_rcept_dt'] = pd.to_datetime(out.first_rcept_dt).dt.date
    out['last_rcept_dt'] = pd.to_datetime(out.last_rcept_dt).dt.date
    # 수집 시작(2015-01-01) 이전에 접수된 원공시의 후속 정정공시만 잡힌 오래된 기간(결산 2014-10 이전)은 first 가 무의미해 제외
    old = (~out.orig_found) & (pd.to_datetime(out.period_end) < pd.Timestamp('2014-10-01'))
    diag['dropped_pre_coverage_amend_only_rows'] = int(old.sum())
    out = out[~old].reset_index(drop=True)
    diag['rows_no_original_in_range'] = int((~out.orig_found).sum())   # 남은 것: 원공시가 목록에 없어 정정공시 날짜가 first(>=실제 최초일)
    return out, diag


def db_keys():
    """동아리 DB fs_line 의 (code,bsns_year,reprt_code,period_end) 키. 읽기 전용, 연도×보고서별로 범위를 좁혀 한 번에 한 쿼리."""
    if os.path.exists(FS_KEYS):
        return pd.read_parquet(FS_KEYS)
    sys.path.insert(0, r'C:\dataguide')
    os.chdir(r'C:\dataguide')
    from dataguide import DataGuide
    dg = DataGuide()
    from sqlalchemy import create_engine        # 터널 경유 TLS 핸드셰이크가 간헐 실패해서 TLS 끔 + 타임아웃 지정(읽기 전용 그대로)
    dg.engine = create_engine(dg.engine.url, connect_args={'ssl_disabled': True, 'connect_timeout': 15, 'read_timeout': 120}, pool_pre_ping=True)
    sql = ('SELECT code, bsns_year, reprt_code, fs_div, period_end FROM financials.fs_line '
           'WHERE bsns_year=:y AND reprt_code=:r AND account_id=:a AND sj_div=:s')
    pdir = FS_KEYS[:-8] + '_parts'
    os.makedirs(pdir, exist_ok=True)
    parts, fails = [], 0
    for y in range(2014, 2027):
        for r in ('11013', '11012', '11014', '11011'):
            f = os.path.join(pdir, f'{y}_{r}.parquet')             # 쿼리마다 저장 -> 중단 후 재실행하면 이어서
            if os.path.exists(f):
                parts.append(pd.read_parquet(f))
                continue
            while True:
                try:
                    q = dg.query(sql, {'y': y, 'r': r, 'a': 'ifrs-full_Equity', 's': 'BS'})
                    q.to_parquet(f, index=False)
                    parts.append(q)
                    print('DB', y, r, len(q), flush=True)
                    break
                except Exception as e:                     # 터널/핸드셰이크 간헐 실패 -> 재시도 (예외 문자열은 비밀번호를 담지 않지만 짧게만 남김)
                    fails += 1
                    print('DB 재시도', y, r, type(e).__name__, flush=True)
                    if fails > 40:
                        raise RuntimeError('DB 조회 반복 실패(40회)') from None
                    time.sleep(20)
    k = pd.concat(parts, ignore_index=True)
    k['ticker'] = k.code.str[1:]
    k['period_end'] = pd.to_datetime(k.period_end).dt.date
    k = k[['ticker', 'bsns_year', 'reprt_code', 'fs_div', 'period_end']]
    k.to_parquet(FS_KEYS, index=False)
    return k


def cmd_fs(a=None):
    d = pd.read_parquet(CLS)
    fs, diag = build_fs(d)
    # --- 동아리 DB fs_line 키와 대조해 bsns_year 규약을 고른다 (DB 못 쓰면 fyend)
    conv, k = 'fyend', None
    if not (a is not None and getattr(a, 'no_db', False)):
        try:
            k = db_keys()
        except Exception as e:
            print('DB 대조 생략:', type(e).__name__, str(e)[:100])
    if k is not None:
        kk = k.drop_duplicates(['ticker', 'reprt_code', 'period_end', 'bsns_year']).rename(columns={'bsns_year': 'db_by'})
        j = kk.merge(fs, on=['ticker', 'reprt_code', 'period_end'], how='inner')
        rates = {c: float((j.db_by == j['by_' + c]).mean()) for c in ('fyend', 'pe', 'fystart')}
        print('DB bsns_year 규약 일치율(같은 ticker·reprt_code·period_end 기준, %d 쌍):' % len(j), {c: round(v, 4) for c, v in rates.items()})
        diff = j[j.by_fyend != j.by_pe]
        if len(diff):
            print('  (fyend 와 pe 가 다른 비-12월 결산 쌍 %d 개 중) fyend 일치 %.1f%% / pe 일치 %.1f%% / fystart 일치 %.1f%%' % (
                len(diff), 100 * (diff.db_by == diff.by_fyend).mean(), 100 * (diff.db_by == diff.by_pe).mean(), 100 * (diff.db_by == diff.by_fystart).mean()))
        conv = max(rates, key=rates.get)
    print('bsns_year 규약:', conv)
    fs['bsns_year'] = fs['by_' + conv]
    fs = fs.drop(columns=['by_fyend', 'by_pe', 'by_fystart'])
    dup = fs.duplicated(['ticker', 'bsns_year', 'reprt_code'], keep=False)
    diag['dup_key_rows(결산월 변경 등)'] = int(dup.sum())
    fs = fs[['ticker', 'bsns_year', 'reprt_code', 'period_end', 'first_rcept_dt', 'last_rcept_dt', 'n_amend', 'orig_found']]
    fs['gap_days'] = (pd.to_datetime(fs.first_rcept_dt) - pd.to_datetime(fs.period_end)).dt.days
    o = fs.drop(columns='gap_days').copy()
    for c_ in ('period_end', 'first_rcept_dt', 'last_rcept_dt'):      # 조인 편의를 위해 저장은 datetime64
        o[c_] = pd.to_datetime(o[c_])
    o.to_parquet(FS_OUT, index=False)
    print(f'{FS_OUT}: {len(fs)} 행, {fs.ticker.nunique()} 종목')
    for kx, v in diag.items():
        print(kx, v)
    ok = fs[fs.orig_found]
    print('--- 보고서 종류별 접수 지연(일, 결산일 -> 최초 접수일, 원공시 확인된 행만)')
    print(ok.groupby('reprt_code').gap_days.describe(percentiles=[.05, .5, .95]).round(1).to_string())
    for rc in ('11011', '11012', '11013', '11014'):
        x = ok[ok.reprt_code == rc].gap_days
        print(f'{rc}: 45~120일 안 {((x >= 45) & (x <= 120)).mean():.1%} | 30~120일 안 {((x >= 30) & (x <= 120)).mean():.1%} | 30일 미만 {(x < 30).sum()} | 120일 초과 {(x > 120).sum()} | 음수 {(x < 0).sum()}')
    smp = fs.ticker.drop_duplicates().sample(20, random_state=7)
    s = ok[ok.ticker.isin(smp)]
    print('--- 임의 20종목(고정 시드) gap_days 분포(보고서별)')
    print(s.groupby('reprt_code').gap_days.describe(percentiles=[.05, .5, .95]).round(1).to_string())
    print('--- 삼성전자 005930')
    print(fs[(fs.ticker == '005930') & (fs.bsns_year >= 2023)].sort_values('period_end').to_string(index=False))
    if k is None:
        return
    kk = k.drop_duplicates(['ticker', 'reprt_code', 'period_end'])
    kk = kk[kk.ticker.isin(set(d.ticker)) & (kk.period_end >= pd.Timestamp('2014-12-31').date())]
    j = kk.merge(fs[['ticker', 'reprt_code', 'period_end', 'first_rcept_dt']], on=['ticker', 'reprt_code', 'period_end'], how='left')
    hit = j.first_rcept_dt.notna()
    print(f'--- DB 키 대조 (수집 대상 종목, period_end>=2014-12-31): DB 키 {len(j)}, 접수일 연결 {hit.mean():.1%}')
    miss = j[~hit]
    dl = pd.read_parquet(MAP_PQ).set_index('ticker').delisted
    print('미연결 DB 키의 보고서별 건수:', miss.reprt_code.value_counts().to_dict(), '| 상장폐지 종목 비중 %.1f%%' % (100 * miss.ticker.map(dl).fillna(False).astype(bool).mean()))
    print('미연결 예시:', miss.head(8)[['ticker', 'reprt_code', 'period_end']].values.tolist())
    print('미연결 DB 키 period_end 연도 분포:', miss.period_end.map(lambda x: x.year).value_counts().sort_index().to_dict())


# ------------------------------------------------------------------ 작업 4: 일별 이벤트 표
def trading_days():
    import pyarrow.parquet as pq
    s = pq.read_table(PANEL, columns=['date']).to_pandas()['date']
    return np.sort(pd.Series(pd.to_datetime(s.unique())).values.astype('datetime64[D]'))


def cmd_daily(a=None):
    d = pd.read_parquet(CLS)
    cal = trading_days()
    r = pd.to_datetime(d.rcept_dt).values.astype('datetime64[D]')
    i = np.searchsorted(cal, r, side='left')                 # rcept_dt 이상인 첫 거래일
    ok = i < len(cal)
    print(f'거래일 달력 {cal[0]} ~ {cal[-1]} ({len(cal)}일). 달력 밖(마지막 거래일 이후) 공시 {(~ok).sum()} 건은 제외')
    d = d[ok].copy()
    i = i[ok]
    d['date'] = cal[i]
    d['rolled'] = d.date.values != np.asarray(pd.to_datetime(d.rcept_dt).values.astype('datetime64[D]'))
    cal_ext = np.concatenate([cal, np.array(['NaT'], dtype='datetime64[D]')])      # 마지막 거래일의 다음은 NaT
    d['next_td'] = cal_ext[np.minimum(i + 1, len(cal))]
    o = ~d.is_amend                       # n_<유형> 은 원공시만, 정정공시는 n_amend 로 분리
    ENG = {'정기': 'periodic', '주요사항': 'major', '지분': 'stake', '수시공정': 'exchange', '기타': 'other'}
    for k, v in ENG.items():
        d['n_' + v] = (o & (d.cat == k)).astype(int)
    d['n_amend'] = d.is_amend.astype(int)
    d['n_all'] = 1
    d['n_rolled'] = d.rolled.astype(int)
    agg = {c: 'sum' for c in d.columns if c.startswith('n_')}
    agg['next_td'] = 'first'
    ev = d.groupby(['ticker', 'date'], as_index=False).agg(agg)
    ev['date'] = pd.to_datetime(ev.date).astype('datetime64[ns]')                 # datetime64 로 저장 (panel.parquet 의 date 와 바로 merge)
    ev['next_td'] = pd.to_datetime(ev.next_td).astype('datetime64[ns]')
    for c in ['periodic', 'major', 'stake', 'exchange', 'amend']:
        ev['d_' + c] = (ev['n_' + c] > 0).astype(np.int8)
    ev = ev.sort_values(['ticker', 'date']).reset_index(drop=True)
    ev.to_parquet(EV_OUT, index=False)
    print(f'{EV_OUT}: {len(ev)} 행, {ev.ticker.nunique()} 종목, 총 공시 {int(ev.n_all.sum())}, 비거래일 이월 {int(ev.n_rolled.sum())}')
    print(ev[[c for c in ev.columns if c.startswith('d_')]].mean().round(3).to_string())


# ------------------------------------------------------------------ 규칙 분류 정확도 (pblntf_ty 지정 호출 표본)
TY_TO_CAT = {'A': '정기', 'B': '주요사항', 'D': '지분', 'I': '수시공정'}      # 나머지(C,E,F,G,H,J) -> 기타


def cmd_accuracy(a=None):
    tag = getattr(a, 'tag', None)
    d0 = f'{RAW_TY}\\{tag}' if tag else RAW_TY
    fs = [os.path.join(r, f) for r, _, fl in os.walk(d0) for f in fl if f.endswith('.parquet')]
    if not fs:
        print('raw_ty 없음')
        return
    print(f'### 표본 묶음 {tag or "전체"}')
    ty = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    c = pd.read_parquet(CLS)
    ty = ty.drop_duplicates(['rcept_no', 'pblntf_ty'])
    multi = ty.rcept_no.duplicated().sum()
    ty['true_cat'] = ty.pblntf_ty.map(TY_TO_CAT).fillna('기타')
    j = ty.merge(c[['rcept_no', 'cat', 'rule', 'is_amend']], on='rcept_no', how='left')
    raw_n = c[c.ticker.isin(ty.ticker.unique())].shape[0]
    print(f'표본 종목 {ty.ticker.nunique()}, 유형 지정 호출로 받은 공시 {len(ty)} 건 (같은 접수번호가 여러 유형에 나온 경우 {multi}), '
          f'목록 수집분 {raw_n} 건 -> 유형 확인된 비율 {len(ty) / raw_n:.1%}, 분류 표에 없는 접수번호 {j.cat.isna().sum()}')
    j = j[j.cat.notna()]
    anyok = j.assign(h=(j.cat == j.true_cat)).groupby('rcept_no').h.max()
    print(f'접수번호 단위 정확도(여러 유형에 걸친 접수번호는 한 유형이라도 맞으면 적중): {anyok.mean():.2%}  ({len(anyok)} 접수번호)')
    acc = (j.cat == j.true_cat).mean()
    print(f'정확도(건수 기준): {acc:.2%}   원공시만: {(j[~j.is_amend].cat == j[~j.is_amend].true_cat).mean():.2%}   정정만: {(j[j.is_amend].cat == j[j.is_amend].true_cat).mean():.2%}')
    print('--- 혼동행렬 (행=실제 pblntf_ty 문자, 열=규칙 분류)')
    print(pd.crosstab(j.pblntf_ty, j.cat).to_string())
    print('--- 혼동행렬 (행=실제 범주, 열=규칙 분류)')
    print(pd.crosstab(j.true_cat, j.cat).to_string())
    # 기타 안의 C/E/F/J 구분까지 본 엄격한 정확도: rule 라벨 첫 글자(A,B,D,I,C,E,F,J) vs 실제 ty ('C/E:추정' 은 C 또는 E 면 일치)
    rt = j.rule.str[0]
    ok = (rt == j.pblntf_ty) | ((j.rule.str.startswith('C/E')) & j.pblntf_ty.isin(['C', 'E']))
    print(f'엄격 정확도(A~J 문자 단위, 규칙 라벨 기준): {ok.mean():.2%}  (대상 {len(j)}건)')
    print(pd.crosstab(j.pblntf_ty, rt.where(~j.rule.str.startswith('C/E'), 'C/E')).to_string())
    print('--- 범주별 재현율/정밀도')
    for k in ['정기', '주요사항', '지분', '수시공정', '기타']:
        tp = ((j.true_cat == k) & (j.cat == k)).sum()
        print(f'{k}: 재현율 {tp / max((j.true_cat == k).sum(), 1):.2%} ({(j.true_cat == k).sum()}건)  정밀도 {tp / max((j.cat == k).sum(), 1):.2%} ({(j.cat == k).sum()}건)')
    # 불일치 보고서명 상위
    j['base'] = j.report_nm.map(lambda s: re.sub(r'\d+', 'N', parse_name(s)[1]))
    bad = j[j.cat != j.true_cat].groupby(['base', 'pblntf_ty', 'cat', 'rule']).size().sort_values(ascending=False)
    print('--- 불일치 상위 (보고서명, 실제ty, 규칙cat, 규칙label)')
    print(bad.head(40).to_string())
    # 보고서명 -> ty 일관성
    cons = j.groupby('base').pblntf_ty.nunique()
    print(f'보고서명(숫자 정규화) {len(cons)} 종 중 둘 이상의 pblntf_ty 에 걸치는 이름 {(cons > 1).sum()} 종')
    j.to_parquet(D + f'\\_typecheck_joined_{tag or "all"}.parquet', index=False)


# ------------------------------------------------------------------ 통계
def cmd_stats(a=None):
    c = pd.read_parquet(CLS)
    c['year'] = pd.to_datetime(c.rcept_dt).dt.year
    print('--- 연도별 건수 / 종목 수 / 정정 비율')
    y = c.groupby('year').agg(n=('rcept_no', 'size'), tickers=('ticker', 'nunique'), amend=('is_amend', 'mean'))
    print(y.round(3).to_string())
    print('--- 범주별')
    print(c.groupby('cat').size().to_string())
    print(c[~c.is_amend].groupby('cat').size().to_string())
    print('미분류 %.2f%% (%d)' % (100 * (c.rule == '미분류').mean(), (c.rule == '미분류').sum()))
    print('--- rule label 별 건수')
    print(c.groupby(['cat', 'rule']).size().sort_values(ascending=False).to_string())
    print('--- 빈도 상위 50 보고서명 (접두어 제거, 숫자->N)')
    nm = c.report_nm.map(lambda s: re.sub(r'\d+', 'N', parse_name(s)[1]))
    print(nm.value_counts().head(50).to_string())
    print('--- 미분류 상위 40')
    print(nm[c.rule == '미분류'].value_counts().head(40).to_string())
    # 커버리지
    sm = pd.read_parquet(STOCK_MASTER)
    k = sm[sm.market == 'KOSPI']
    mp = pd.read_parquet(MAP_PQ).set_index('ticker')
    have = set(c.ticker)
    k = k.assign(mapped=k.ticker.map(mp.mapped), has=k.ticker.isin(have), delisted=k.ticker.map(mp.delisted))
    print('--- 커버리지(stock_master KOSPI 1260 기준)')
    print(f'KOSPI {len(k)}, 매핑 {int(k.mapped.sum())}, 2015~ 공시 1건 이상 {int(k.has.sum())}')
    print(k.groupby(['delisted', 'mapped', 'has']).size().to_string())
    px = k[k.n_days > 0]
    print(f'가격 데이터가 있는 종목(n_days>0) {len(px)} 중 공시 확보 {int(px.has.sum())}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['classify', 'fs', 'daily', 'accuracy', 'stats', 'all'])
    ap.add_argument('--no-db', action='store_true', help='fs: 동아리 DB 키 대조 생략')
    ap.add_argument('--tag', default=None, help='accuracy: raw_ty 표본 묶음(A/B), 생략하면 전체')
    a = ap.parse_args()
    fn = dict(classify=cmd_classify, fs=cmd_fs, daily=cmd_daily, accuracy=cmd_accuracy, stats=cmd_stats)
    for n in (['classify', 'fs', 'daily', 'accuracy', 'stats'] if a.cmd == 'all' else [a.cmd]):
        print(f'\n===== {n} =====')
        fn[n](a)

"""KRX(pykrx) 접근.

⚠️ 대량·병렬 수집 금지. 2026-10-05 이 방식으로 200종목을 병렬 수집하다가 KRX가 IP를 1일 차단했다
(이용약관 제10조 제2호: 자동화 수단을 이용한 무단 수집 금지). 논문용 데이터는 DataGuide나 KRX Open API로 받는다.

계정은 사용자가 자기 터미널에서 `setx KRX_ID 아이디`, `setx KRX_PW 비밀번호`로 등록한다(대화·볼트에 남기지 않는다).
setx는 이미 떠 있는 프로세스에 반영되지 않으므로 레지스트리(HKCU\\Environment)에서 읽어 채운다.
"""
import os
import winreg

with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as _k:
    for _n in ("KRX_ID", "KRX_PW"):
        if not os.environ.get(_n):
            try:
                os.environ[_n] = winreg.QueryValueEx(_k, _n)[0]
            except FileNotFoundError:
                pass

from pykrx import stock  # noqa: E402  환경변수를 채운 뒤에 불러와야 로그인된다

COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume", "거래대금": "value", "등락률": "chg"}


def ohlcv(ticker, start, end):
    """일별 수정 OHLCV. pykrx 수정주가는 2014-07 이전이 비어 있어 KRX 비수정 가격과 등락률로 직접 수정한다.
    등락률은 기준가 대비라서 기준가 B_t = 종가_t / (1 + 등락률_t). B_t / 종가_{t-1} 이 1이 아니면 그날 분할·권리락 등이
    있었다는 뜻이므로, 그 이전 가격에 비율을 곱하고 거래량은 나눈다. 거래대금은 그대로 둔다."""
    d = stock.get_market_ohlcv(start, end, ticker, adjusted=False).rename(columns=COLS).rename_axis("date")
    d = d[list(COLS.values())].astype(float)
    f = d.close / (1 + d.chg / 100) / d.close.shift()
    f = f.where((f - 1).abs() > 1e-3, 1.0).fillna(1.0)          # 등락률 반올림 오차(약 5e-5)는 무시
    k = f[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)       # 각 날짜 이후 모든 수정 비율의 곱
    d[["open", "high", "low", "close"]] = d[["open", "high", "low", "close"]].mul(k, axis=0)
    d["volume"] = d.volume / k
    return d.drop(columns="chg")

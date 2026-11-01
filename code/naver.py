"""네이버 일봉 — KRX 계정 없이 돌리는 스모크 테스트 전용.
상장폐지 종목이 없어 생존편향이 생기므로 본 분석에는 쓰지 않는다.
"""
import re

import pandas as pd
import requests


def daily(ticker, count=5000):
    r = requests.get("https://fchart.stock.naver.com/sise.nhn", timeout=10,
                     params=dict(symbol=ticker, timeframe="day", count=count, requestType=0))
    rows = [x.split("|") for x in re.findall(r'data="([^"]+)"', r.text)]
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").astype(float)

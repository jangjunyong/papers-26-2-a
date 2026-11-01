"""계획 3단계: 데이터 접근 테스트. 결과를 보고 pykrx로 갈지 DataGuide로 보완할지 정한다(⛳ 3-6)."""
from krx import ohlcv, stock


def show(name, f):
    try:
        d = f()
        print(f"[{'OK' if len(d) else 'EMPTY'}] {name}: {d.shape}")
        print(d.head(3).to_string(), "\n")
        return d
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}\n")


show("3-1 삼성전자 2005-01 OHLCV(수정)", lambda: ohlcv("005930", "20050103", "20050131"))
show("3-2 삼성전자 2005-01 투자자별 거래대금", lambda: stock.get_market_trading_value_by_date("20050103", "20050131", "005930", detail=True))

k05 = stock.get_index_portfolio_deposit_file("1028", "20050103")
now = set(stock.get_market_ticker_list(market="ALL"))
gone = [t for t in k05 if t not in now]
print(f"2005-01-03 KOSPI200 {len(k05)}종목 중 현재 목록에 없는 종목 {len(gone)}개: {gone[:10]}\n")
if gone:
    d = show(f"3-3 상장폐지 추정 종목 {gone[0]} 마지막 구간", lambda: ohlcv(gone[0], "20040101", "20261001").tail(5))

d = ohlcv("005930", "20180420", "20180510")
print("3-4 거래정지(2018-04-30~05-03 액면분할) 구간과 거래량 수정 여부:\n", d.to_string(), "\n")

print("3-5 투자자별 거래대금 연도별 커버리지(삼성전자, 연초 첫 주):")
for y in range(1996, 2027, 2):
    try:
        n = len(stock.get_market_trading_value_by_date(f"{y}0102", f"{y}0110", "005930"))
    except Exception:
        n = -1
    print(y, n)

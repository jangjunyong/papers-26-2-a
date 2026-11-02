---
title: DataGuide 추가 다운로드 요청 (차트 CNN 연구)
created: 2026-10-05
updated: 2026-10-05
tags: [contest/papers, finance/퀀트]
source: 볼트 기존 DataGuide 원본 헤더 대조(data/원본_데이터/0_data), 연구계획 5단계
status: seed
---

# DataGuide 추가 다운로드 요청

KOSPI 일별 가격(2000~2026, 상장폐지 포함 1,320종목)은 이미 `data/원본_데이터/0_data/`에 있다. 연구의 핵심인 **투자자별 순매수**와 **KOSDAQ**만 추가로 받으면 된다. KRX 웹 자동 수집은 약관 위반이라 쓰지 않는다([[PAPERS/26-2/연구/README|README]] 참고).

## 공통 설정 (기존 파일과 같게)

- 코드 포트폴리오: **상장폐지 종목 포함** (기존 `코스피_상장폐지포함`과 같은 방식)
- 출력주기 **일간**, 원화 / 비영업일 **제외** / 주말 **제외** / **오름차순**
- 기간: **2000-01-01 ~ 최근일자**
- 엑셀 한 파일에 아이템 하나씩. 파일이 너무 커서 안 받아지면 기간을 2000~2012, 2013~최근으로 나눈다.

## 받을 것 (위에서부터 우선)

| # | 시장 | 아이템 (DataGuide에서 검색할 이름) | 저장 경로 |
|---|---|---|---|
| 1 | KOSPI | 순매수대금 — **개인** | `0_data/09_investor/kospi_net_buy_individual.xlsx` |
| 2 | KOSPI | 순매수대금 — **외국인** (기타외국인이 따로 있으면 그것도) | `0_data/09_investor/kospi_net_buy_foreign.xlsx` |
| 3 | KOSPI | 순매수대금 — **기관계** | `0_data/09_investor/kospi_net_buy_institution.xlsx` |
| 4 | KOSPI | 거래대금(원) | `0_data/03_volume/trading_value.xlsx` |
| 5 | KOSDAQ | 1~4와 같은 아이템 | `0_data/10_kosdaq/` 아래 같은 이름 |
| 6 | KOSDAQ | 시가·고가·저가·종가·수정주가·거래량·상장주식수(보통주)·거래정지여부 | `0_data/10_kosdaq/` 아래 KOSPI와 같은 파일 이름 |

- 1~4만 있어도 KOSPI로 H1·H2 본분석을 시작할 수 있다.
- 5~6은 개인 비중이 높은 시장까지 넓히는 용도다. 반전 가설은 KOSDAQ에서 더 잘 보일 가능성이 있다.
- 순매수대금이 없고 매수대금·매도대금만 있으면 둘 다 받는다(빼서 순매수를 만든다).

## 받은 뒤

파일을 위 경로에 넣고 알려 주면 `code/dg.py`에 경로를 추가해 parquet으로 변환하고, 기존 KOSPI 가격과 합친다.

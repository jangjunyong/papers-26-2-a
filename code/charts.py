"""Jiang·Kelly·Xiu(2023) 규격 OHLC 차트 이미지.

20일 이미지 = 64x60 흑백. 하루 3픽셀(왼쪽 시가 · 가운데 고가~저가 막대 · 오른쪽 종가),
이동평균선(창 길이 = 이미지 일수), 아래 1/5은 거래량 막대(가운데 열), 가격 영역과 1픽셀 간격.
가격 영역은 창 안의 고가·저가·이동평균 최대·최소로, 거래량은 창 안 최대 거래량으로 스케일한다.
"""
import numpy as np

HEIGHT = {5: 32, 20: 64, 60: 96}


def chart_image(o, h, l, c, v, ma):
    """길이 n(5·20·60) 배열 → uint8 (H, 3n), 값은 0 또는 255. 입력에 NaN이 있으면 안 된다."""
    o, h, l, c, v, ma = (np.asarray(a, float) for a in (o, h, l, c, v, ma))
    n = len(c)
    H = HEIGHT[n]
    vol_h = H // 5                 # 20일: 12
    px_h = H - vol_h - 1           # 20일: 51
    img = np.zeros((H, 3 * n), np.uint8)

    lo, hi = min(l.min(), ma.min()), max(h.max(), ma.max())
    scale = (px_h - 1) / (hi - lo) if hi > lo else 0.0

    def row(p):  # 가격 → 행 번호(위가 0)
        return (px_h - 1 - np.round((np.asarray(p) - lo) * scale)).astype(int)

    ro, rh, rl, rc, rm = row(o), row(h), row(l), row(c), row(ma)
    for i in range(n):
        x = 3 * i
        img[ro[i], x] = 255
        img[rh[i]:rl[i] + 1, x + 1] = 255
        img[rc[i], x + 2] = 255
        if i:  # 이동평균선: 전날 가운데 열 → 오늘 가운데 열을 잇는다
            k = max(3, abs(rm[i] - rm[i - 1])) + 1
            xs = np.round(np.linspace(x - 2, x + 1, k)).astype(int)
            ys = np.round(np.linspace(rm[i - 1], rm[i], k)).astype(int)
            img[ys, xs] = 255
    img[rm[0], 1] = 255

    vh = np.round(v / v.max() * vol_h).astype(int) if v.max() > 0 else np.zeros(n, int)
    for i in range(n):
        if vh[i]:
            img[H - vh[i]:, 3 * i + 1] = 255
    return img


if __name__ == "__main__":
    # 자체 점검: 꾸며낸 20일 데이터로 규격과 위치를 확인한다.
    n = 20
    c = np.linspace(100, 119, n)
    o, h, l = c - 0.5, c + 1, c - 1
    v = np.arange(1, n + 1, dtype=float)
    ma = c - 2
    img = chart_image(o, h, l, c, v, ma)
    assert img.shape == (64, 60) and set(np.unique(img)) <= {0, 255}
    assert img[0, 58] == 255                 # 마지막 날 고가 = 창 최대 → 맨 윗줄, 가운데 열(3*19+1)
    assert img[51].sum() == 0                # 가격·거래량 사이 1픽셀 간격
    assert img[52, 58] == 255                # 최대 거래량 → 거래량 영역 맨 위
    assert img[52:, 0].sum() == 0            # 거래량은 가운데 열에만
    flat = chart_image(*(np.full(n, 5.0),) * 4, np.zeros(n), np.full(n, 5.0))
    assert flat.shape == (64, 60)            # 가격·거래량이 모두 일정해도 터지지 않는다
    print("charts.py 자체 점검 통과")

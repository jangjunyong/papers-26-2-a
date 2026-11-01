"""d06 공용: 창 나열/스크린샷 도우미(pywin32 + Pillow)."""
import win32gui, win32process, win32con, psutil
from PIL import ImageGrab

def windows(pids=None, visible_only=True):
    out = []
    def cb(h, _):
        if visible_only and not win32gui.IsWindowVisible(h): return
        _, pid = win32process.GetWindowThreadProcessId(h)
        if pids is not None and pid not in pids: return
        t = win32gui.GetWindowText(h)
        out.append((h, pid, win32gui.GetClassName(h), t, win32gui.GetWindowRect(h)))
    win32gui.EnumWindows(cb, None)
    return out

def shot(path, rect=None):
    ImageGrab.grab(bbox=rect, all_screens=True).save(path)


def shot_window(path, hwnd):
    """가려진 창도 PrintWindow(PW_RENDERFULLCONTENT)로 그 창 내용만 캡처(화면 전체를 찍지 않는다)."""
    import ctypes, win32ui
    from PIL import Image
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetWindowDC(hwnd)
    src = win32ui.CreateDCFromHandle(hdc); mem = src.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap(); bmp.CreateCompatibleBitmap(src, w, h); mem.SelectObject(bmp)
    ctypes.windll.user32.PrintWindow(hwnd, mem.GetSafeHdc(), 2)
    info = bmp.GetInfo(); bits = bmp.GetBitmapBits(True)
    img = Image.frombuffer('RGB', (info['bmWidth'], info['bmHeight']), bits, 'raw', 'BGRX', 0, 1)
    img.save(path)
    win32gui.DeleteObject(bmp.GetHandle()); mem.DeleteDC(); src.DeleteDC(); win32gui.ReleaseDC(hwnd, hdc)

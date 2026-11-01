"""d06 step2: Excel 별도 인스턴스에서 DG6 애드인 로드 여부만 점검(로그인 없음)."""
import sys, os, time, win32com.client as wc, win32process, psutil
sys.stdout.reconfigure(encoding='utf-8')
before = {p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='excel.exe'}
print('기존 EXCEL pid:', before)
xl = wc.DispatchEx('Excel.Application')          # 새 인스턴스
xl.Visible = False; xl.DisplayAlerts = False
pid = win32process.GetWindowThreadProcessId(xl.Hwnd)[1]
print('내 EXCEL pid:', pid, 'ver', xl.Version, 'build', xl.Build)
try:
    time.sleep(5)
    print('AddIns:', [(a.Name, a.Installed) for a in xl.AddIns][:20])
    print('COMAddIns:', [(a.ProgId, a.Connect) for a in xl.COMAddIns])
    rf = xl.RegisteredFunctions
    print('RegisteredFunctions:', None if rf is None else len(rf))
    if rf is not None:
        for r in list(rf)[:40]: print('  ', r)
    print('workbooks', xl.Workbooks.Count)
finally:
    xl.Quit(); time.sleep(2)
    print('남은 pid?', psutil.pid_exists(pid))

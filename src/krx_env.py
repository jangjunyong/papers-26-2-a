"""KRX 계정을 환경변수로 준비한다.

사용자가 별도 PowerShell 창에서 `setx KRX_ID ...`, `setx KRX_PW ...` 로 등록하면
이미 실행 중인 프로세스에는 반영되지 않는다. 그래서 현재 프로세스 환경변수에 없으면
Windows 사용자 환경변수(레지스트리 HKCU\\Environment)에서 읽어 채운다.
값은 출력하거나 파일에 쓰지 않는다. pykrx 를 import 하기 전에 호출해야 한다.
"""
from __future__ import annotations

import os


def load_krx_env() -> bool:
    for key in ("KRX_ID", "KRX_PW"):
        if os.environ.get(key):
            continue
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                val, _ = winreg.QueryValueEx(k, key)
            if val:
                os.environ[key] = val
        except (ImportError, OSError):
            pass
    return bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))

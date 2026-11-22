# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone gemini2api binary.

Two things are load-bearing:

* `pathex` points at the repository, because the package is installed editable in
  development and PyInstaller would otherwise not find `gemini2api` at all.
* `collect_all("curl_cffi")` gathers libcurl-impersonate and its shared objects.
  Without it the frozen binary imports but cannot open a TLS connection.

Build with: `uv run pyinstaller --noconfirm --clean packaging/gemini2api.spec`
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

PACKAGING = Path(SPECPATH).resolve()
ROOT = PACKAGING.parent

curl_datas, curl_binaries, curl_hidden = collect_all("curl_cffi")

hiddenimports = [
    *curl_hidden,
    # uvicorn resolves its loop/protocol implementations by name at runtime.
    *collect_submodules("uvicorn"),
    "gemini2api.cookies",
    "gemini2api.doctor",
    "win32_setctime",
]

analysis = Analysis(
    [str(PACKAGING / "entry.py")],
    pathex=[str(ROOT)],
    binaries=curl_binaries,
    datas=curl_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "ruff", "tkinter"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="gemini2api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# -*- mode: python -*-
# 在 TRG_SurgeryAdvisor 项目根目录执行: pyinstaller packaging/trg_app.spec
import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None
PROJ = os.getcwd()

gradio_datas, gradio_binaries, gradio_hidden = collect_all("gradio")

a = Analysis(
    [os.path.join(PROJ, "app_gradio.py")],
    pathex=[PROJ],
    binaries=gradio_binaries,
    datas=gradio_datas,
    hiddenimports=list(gradio_hidden) + ["yaml", "cv2", "PIL", "PIL.Image", "numpy", "onnxruntime"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TRG_SurgeryAdvisor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TRG_SurgeryAdvisor",
)

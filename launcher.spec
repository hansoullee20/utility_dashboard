# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# Collect all streamlit assets (static web files, runtime, etc.)
st_datas, st_binaries, st_hiddenimports = collect_all('streamlit')
altair_datas, altair_binaries, altair_hiddenimports = collect_all('altair')
sac_datas, sac_binaries, sac_hiddenimports = collect_all('streamlit_antd_components')

root = Path(__file__).resolve().parent
app_datas = [(str(path), '.') for path in root.glob('*.py')]
app_datas += [
    (str(path), '.')
    for pattern in ('*.json', '*.pdf')
    for path in root.glob(pattern)
]
if (root / 'fonts').exists():
    app_datas += [(str(path), 'fonts') for path in (root / 'fonts').glob('*') if path.is_file()]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=st_binaries + altair_binaries + sac_binaries,
    datas=st_datas + altair_datas + sac_datas + app_datas,
    hiddenimports=st_hiddenimports + altair_hiddenimports + sac_hiddenimports + [
        'pandas',
        'numpy',
        'python_calamine',
        'plotly',
        'plotly.express',
        'plotly.graph_objects',
        'plotly.io',
        'scipy',
        'scipy.stats',
        'streamlit_antd_components',
        'reportlab',
        'reportlab.lib',
        'reportlab.lib.colors',
        'reportlab.lib.pagesizes',
        'reportlab.lib.styles',
        'reportlab.lib.units',
        'reportlab.lib.enums',
        'reportlab.pdfbase',
        'reportlab.pdfbase.ttfonts',
        'reportlab.pdfbase.pdfmetrics',
        'reportlab.pdfgen.canvas',
        'reportlab.platypus',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.patches',
        'matplotlib.font_manager',
        'matplotlib.backends.backend_agg',
        'PIL',
        'PIL.Image',
        'openpyxl',
        'openpyxl.styles',
        'openpyxl.utils',
        'pyarrow',
        'pyarrow.vendored.version',
        'xlsxwriter',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='UtilityDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

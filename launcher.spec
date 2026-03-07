# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

# Collect all streamlit assets (static web files, runtime, etc.)
st_datas, st_binaries, st_hiddenimports = collect_all('streamlit')
altair_datas, altair_binaries, altair_hiddenimports = collect_all('altair')

app_datas = [
    ('app.py',                    '.'),
    ('data.py',                   '.'),
    ('features.py',               '.'),
    ('viz.py',                    '.'),
    ('billing.py',                '.'),
    ('billing_report.py',         '.'),
    ('ehp.py',                    '.'),
    ('ehp_report.py',             '.'),
    ('report.py',                 '.'),
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=st_binaries + altair_binaries,
    datas=st_datas + altair_datas + app_datas,
    hiddenimports=st_hiddenimports + altair_hiddenimports + [
        'pandas',
        'numpy',
        'plotly',
        'plotly.express',
        'plotly.graph_objects',
        'plotly.io',
        'scipy',
        'scipy.stats',
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

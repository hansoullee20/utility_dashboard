"""Generate technical design note PDF for utility_analysis dashboard."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

OUT = "utility_analysis_tech_note.pdf"

# ── Fonts ──────────────────────────────────────────────────────────────────────
# Use a system monospace font if available, fallback to Courier
_MONO = "Courier"

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=18, spaceAfter=6,
                    textColor=colors.HexColor("#1a1a2e"), leading=22)
H2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=13, spaceAfter=4,
                    spaceBefore=14, textColor=colors.HexColor("#16213e"),
                    borderPad=2, leading=17)
H3 = ParagraphStyle("H3", parent=base["Heading3"], fontSize=11, spaceAfter=3,
                    spaceBefore=8, textColor=colors.HexColor("#0f3460"), leading=14)
BODY = ParagraphStyle("Body", parent=base["Normal"], fontSize=9.5, leading=14,
                      spaceAfter=4, textColor=colors.HexColor("#222222"))
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.5, textColor=colors.HexColor("#555555"),
                         italic=True)
MONO = ParagraphStyle("Mono", fontName=_MONO, fontSize=8, leading=11,
                      spaceAfter=2, textColor=colors.HexColor("#1a1a1a"),
                      backColor=colors.HexColor("#f5f5f5"),
                      leftIndent=6, rightIndent=6)

DIVIDER_COLOR = colors.HexColor("#cccccc")


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6, spaceBefore=2)


def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def h3(text): return Paragraph(text, H3)
def p(text):  return Paragraph(text, BODY)
def cap(text): return Paragraph(text, CAPTION)
def sp(h=4):  return Spacer(1, h * mm)


def code(text):
    """Monospaced preformatted block."""
    return Preformatted(text, MONO)


def table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("LEADING",    (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    else:
        style += [("FONTNAME", (0, 0), (-1, -1), "Helvetica")]
    for i in range(1, len(data)):
        bg = colors.HexColor("#f9f9f9") if i % 2 == 0 else colors.white
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


# ── Content ────────────────────────────────────────────────────────────────────

def build():
    W, H = A4
    M = 18 * mm
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
    )

    story = []

    # ── Title block ────────────────────────────────────────────────────────────
    story += [
        sp(2),
        h1("Utility Analysis Dashboard"),
        Paragraph("Technical Design Note", ParagraphStyle("Sub", parent=BODY, fontSize=12,
                  textColor=colors.HexColor("#555555"), spaceAfter=2)),
        cap("Generated 2026-03-10  ·  Internal reference — do not distribute"),
        hr(), sp(2),
    ]

    # ── 1. Stack ───────────────────────────────────────────────────────────────
    story += [h2("1. Stack"), hr()]
    story += [
        p("<b>Runtime:</b> Python 3.11+, Streamlit, Plotly (go + px), pandas / numpy"),
        p("<b>Venv:</b> shared <font name='Courier'>../finance_vis/venv_finance</font> — do not create a new one"),
        p("<b>Launch:</b>"),
        code("source ../finance_vis/venv_finance/bin/activate && streamlit run app.py"),
        p("<b>Input:</b> Korean <font name='Courier'>.xlsm/.xlsx</font> files loaded from "
          "<font name='Courier'>./data/</font> (configurable path in sidebar)"),
        sp(),
    ]

    # ── 2. Module Map ──────────────────────────────────────────────────────────
    story += [h2("2. Module Map"), hr()]
    story += [code("""\
app.py              ← Page config, top-level routing, file loading
sidebar.py          ← File loader (dir scan → _FileEntry list), bins/tail stubs
data.py             ← Excel I/O (@st.cache_data), sheet name constants, billing period detection
features.py         ← Column engineering: create_change_columns, build_from_two_files,
                       aggregate_by_brand, split_brand_by_floor, floor parsing helpers
filters.py          ← render_meter_filters, show_filter_widgets, apply_sheet_filter,
                       brand_search_bar
viz.py              ← plot_hist_with_tails (Plotly go.Bar histogram with IQR/tail shading)
lang.py             ← t() translation helper (Korean only, key→string)

meter_view.py       ← 검침내역 full pipeline: load → filter → histograms → 5 tabs
summary.py          ← 총 유틸리티 순위 + 면적당 총비용 (water/hotwater/electricity summary)
billing.py          ← 수도광열비 부과 내역 view
ehp.py              ← EHP(OAC)검침자료 view
water.py / hotwater.py / electricity.py  ← per-sheet views
brand_profile.py    ← Brand profile tab with peer comparison charts

tab_anomaly.py      ← 이상감지 tab (composite score, heatmap, spike/cost/HVAC sub-tabs)
tab_cross.py        ← 비용분析 tab (unit costs, electricity breakdown)
tab_efficiency.py   ← 효율분析 tab (per-m² usage benchmarking)
tab_corr.py / tab_reconciliation.py  ← correlation & reconciliation tabs

anomaly_features.py ← build_anomaly_df: merges meter+billing+elec+water → anomaly scores
cross_features.py   ← build_unit_costs, build_elec_breakdown
report.py           ← generate_report_pdf (검침내역 main report)
biz_report.py       ← generate_anomaly_pdf, generate_cross_pdf, generate_efficiency_pdf
billing_report.py / ehp_report.py / hvac_report.py  ← per-sheet PDF generators"""), sp()]

    # ── 3. Navigation Architecture ─────────────────────────────────────────────
    story += [h2("3. Navigation Architecture"), hr()]
    story += [code("""\
sac.tabs (sidebar, left-position) → nav_mode
  ├── 시트 보기        → sheet selectbox → billing/ehp/water/hotwater/electricity/meter_view
  ├── 분析             → analysis selectbox
  │     ├── 요약 분析  → render_summary_view(water_df, hw_df, el_df)
  │     └── 업체 분析  → render_meter_filters → [이상감지|비용분析|효율분析] tabs
  └── 브랜드 프로필    → render_brand_profile_tab"""),
        sp(2),
        p("<b>Key quirk:</b> <font name='Courier'>sac.tabs</font> must render directly inside "
          "<font name='Courier'>with st.sidebar:</font> — <b>never</b> inside "
          "<font name='Courier'>st.empty()</font>. The empty placeholder causes the custom "
          "component to re-initialize on every rerun, losing state."),
        sp(),
    ]

    # ── 4. Data Pipeline ───────────────────────────────────────────────────────
    story += [h2("4. Data Pipeline (검침내역)"), hr()]
    story += [code("""\
read_sheet()                      # raw Excel, header=[2, 3, 4]
  └─ apply_header_rows()          # flatten MultiIndex headers → named columns
       └─ build_from_two_files()  # prev file → *_previous = prev month usage
            └─ create_change_columns()  # *_change = curr - prev, *_pct = change/prev*100
                 └─ aggregate_by_brand()    # group by (brand, building), sum usage
                      └─ split_brand_by_floor()  # if floors filtered, divide totals equally"""),
        sp(2),
        p("<b>Cumulative vs usage:</b> Meter readings in the Excel are cumulative. "
          "<font name='Courier'>build_from_two_files</font> renames them to "
          "<font name='Courier'>*_meter_curr/prev</font> and uses the pre-computed usage column "
          "(<font name='Courier'>water_usage_m3</font> etc.) as the actual current-month consumption."),
        sp(2),
        h3("Column Naming Convention"),
        table([
            ["Column", "Meaning"],
            ["water_previous / water_current", "This month's usage from prev/cur file"],
            ["water_change", "current − previous"],
            ["water_pct", "change / previous × 100"],
            ["water_meter_prev / water_meter_curr", "Original cumulative readings (kept for backward detection)"],
        ], col_widths=[90*mm, 90*mm]),
        sp(),
    ]

    # ── 5. Floor Logic ─────────────────────────────────────────────────────────
    story += [h2("5. Floor Logic"), hr()]
    story += [
        p("Floor values are compound strings: <font name='Courier'>\"1F/2F\"</font>, "
          "<font name='Courier'>\"2F~5F\"</font>, <font name='Courier'>\"B2F/B1F\"</font>."),
        sp(1),
        table([
            ["Function", "Purpose"],
            ["parse_floor_value(s)", "Returns list of individual floor strings from compound value"],
            ["get_simple_floors(df)", "Sorted unique floors for multiselect widget"],
            ["split_brand_by_floor(df, sel_floors)", "Divides brand totals equally by count of matched floors"],
        ], col_widths=[75*mm, 105*mm]),
        sp(2),
        p("When <font name='Courier'>sel_floors == [\"All\"]</font> and "
          "<font name='Courier'>sel_bldg == [\"All\"]</font> → no splitting, use aggregate totals."),
        sp(),
    ]

    # ── 6. Histogram System ────────────────────────────────────────────────────
    story += [h2("6. Histogram System (viz.py: plot_hist_with_tails)"), hr()]
    story += [
        h3("Signature (key params)"),
        code("""\
plot_hist_with_tails(s, bins, lo, hi, title,
    source_df=None, val_col=None, key="hist",
    display_cols=None, tail_pct=None, val_scale=1.0)"""),
        sp(2),
        p("• <b>lo / hi</b> are in the <b>same units as the plotted series s</b>"),
        p("• <b>val_scale</b> bridges the gap when <font name='Courier'>s</font> is scaled "
          "(e.g. <font name='Courier'>/1e4</font> for 만원) but "
          "<font name='Courier'>source_df[val_col]</font> is raw 원:"),
        code("mask = (source_df[val_col] / val_scale >= x0) & (source_df[val_col] / val_scale <= x1)"),
        p("• Style is <b>locked</b>: blue normal bars (#4C72B0), amber tail bars (#DD8A00), "
          "red median line (#C44E52), white bg, height=380px"),
        sp(2),
        h3("IQR Detection Pattern"),
        code("""\
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr    = q3 - q1
lo     = q1 - k * iqr   # k slider: 0.5–3.0, default 1.5, step 0.25
hi     = q3 + k * iqr
# Show LaTeX:
st.markdown(
    f"$$Q_1={q1:,.0f},\\;Q_3={q3:,.0f},\\;IQR={iqr:,.0f}$$\\n\\n"
    f"$$\\\\text{{Lower}}={lo:,.0f},\\;\\\\text{{Upper}}={hi:,.0f}\\;(k={k})$$"
)"""),
        sp(),
    ]

    # ── 7. Money Formatting ────────────────────────────────────────────────────
    story += [h2("7. Money Formatting"), hr()]
    story += [code("""\
def _fmt_won(v):
    if abs(v) >= 1e8: return f"{v/1e8:,.0f} 억원"
    if abs(v) >= 1e4: return f"{v/1e4:,.0f} 만원"
    return f"{v:,.0f} 원"

# For bar charts — compute _div from max(series):
_div, _unit = (1e8, "억원") if _max >= 1e8 else (1e4, "만원")
_xv = series / _div
# pass val_scale=_div to plot_hist_with_tails so bin-click filter still works"""),
        p("Rule: <b>no decimals on 원 values</b> — always <font name='Courier'>:.0f</font>."),
        sp(),
    ]

    # ── 8. Anomaly Scoring ────────────────────────────────────────────────────
    story += [h2("8. Anomaly Scoring (anomaly_features.py)"), hr()]
    story += [
        p("Composite score = weighted sum of 5 components, each normalized to [0, 1]:"),
        sp(1),
        table([
            ["Component", "Weight", "Signal", "Source sheet"],
            ["급등 Spike",       "30%", "MoM % change vs thresholds (100/50/20%)", "검침내역"],
            ["소비 Consumption", "25%", "Quadrant scores: HH=4, HL=3, LH=2, Normal=1, LL=0", "검침내역"],
            ["비용 Cost",        "25%", "Z-scores of ₩/m³, ₩/kWh, 만원/m²", "수도광열비 부과 내역"],
            ["HVAC",             "10%", "kWh/m² IQR-normalized", "전체 전기 사용내역"],
            ["일관성 Consistency","10%", "Count of zero-usage utilities", "검침내역 + sheets"],
        ], col_widths=[38*mm, 18*mm, 80*mm, 44*mm]),
        sp(2),
        p("<b>Risk levels:</b> 🔴 위험 ≥ 0.65  ·  🟠 주의 ≥ 0.40  ·  🟡 관찰 ≥ 0.20  ·  🟢 정상 < 0.20"),
        sp(),
    ]

    # ── 9. PDF Generation Pattern ─────────────────────────────────────────────
    story += [h2("9. PDF Generation Pattern (all biz tabs)"), hr()]
    story += [code("""\
_pdf_key = f"{tab}_pdf_{file_name}"
_col_gen, _col_dl = st.columns([1, 2])
with _col_gen:
    if st.button("📄 PDF 리포트 생성", key=f"gen_{tab}_pdf_{file_name}"):
        with st.spinner("PDF 생성 중…"):
            st.session_state[_pdf_key] = generate_X_pdf(df)
if _pdf_key in st.session_state:
    with _col_dl:
        st.download_button("⬇️ 다운로드", st.session_state[_pdf_key], ...)"""),
        p("Generate-then-cache pattern: PDF bytes stored in session_state to avoid "
          "re-generating on every Streamlit rerun. Generate button and download button "
          "are separate widgets in adjacent columns."),
        sp(),
    ]

    # ── 10. Common Pitfalls ───────────────────────────────────────────────────
    story += [h2("10. Common Pitfalls"), hr()]
    pitfalls = [
        ["#", "Issue", "Fix"],
        ["1", "Closure bug in loops",
         "Use default arg binding: def fn(key, _p=p, _cc=cc): — not bare p/cc"],
        ["2", "sac.tabs state loss",
         "Never wrap in st.empty(). Use with st.sidebar: directly."],
        ["3", "Bin-click scale mismatch",
         "When series is scaled before histogram, pass val_scale=scale_factor to plot_hist_with_tails"],
        ["4", "Radio option ordering",
         "Histogram first — [\"히스토그램\", ...] or [\"% Change only\", \"Change only\", \"Side by Side\"]"],
        ["5", "Decimal on 원 values",
         "Always :.0f, never :.2f"],
        ["6", "EHP column parsing",
         "Cumulative cols M–DG (0-based 12–110); stop at next ▣ header; do NOT merge two tables"],
        ["7", "sac.tabs ValueError",
         "Raises if stored index ≥ number of tabs. Clear session state key nav_{file_name} to reset"],
    ]
    story += [table(pitfalls, col_widths=[8*mm, 52*mm, 120*mm]), sp()]

    # ── 11. Session State Keys ────────────────────────────────────────────────
    story += [h2("11. Notable Session State Keys"), hr()]
    story += [
        table([
            ["Key", "Purpose"],
            ["nav_{file_name}",              "Sidebar nav tab index (sac.tabs)"],
            ["anomaly_loaded_{file_name}",   "Gate: has anomaly analysis been started?"],
            ["cross_loaded_{file_name}",     "Gate: has cross-sheet data been loaded?"],
            ["anomaly_pdf_{file_name}",      "Cached anomaly PDF bytes"],
            ["cross_pdf_{file_name}",        "Cached cross-tab PDF bytes"],
            ["efficiency_pdf_{file_name}",   "Cached efficiency PDF bytes"],
            ["{prefix}_iqr_k",              "IQR k multiplier per histogram"],
            ["{prefix}_bins / {prefix}_bins_i", "Bins slider/input sync pair"],
        ], col_widths=[90*mm, 90*mm]),
        sp(),
    ]

    # ── 12. Pending ───────────────────────────────────────────────────────────
    story += [h2("12. Pending / Known Issues"), hr()]
    story += [
        p("• <b>면적당 총비용 unit scaling</b>: Should display as 만원/㎡ (not bare 만원 or 원/m²). "
          "Implementation was reverted at user request — ready to re-implement cleanly."),
        p("• No other explicit outstanding tasks as of 2026-03-10."),
        sp(4),
    ]

    doc.build(story)
    print(f"PDF written → {OUT}")


if __name__ == "__main__":
    build()

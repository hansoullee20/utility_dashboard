"""Generate technical design note PDF for utility_analysis dashboard."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = "utility_analysis_tech_note.pdf"

# ── Fonts ──────────────────────────────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("Nanum",     "/home/hansoullee20/.fonts/NanumGothic-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Nanum-Bold","/home/hansoullee20/.fonts/NanumGothic-Bold.ttf"))

# NanumGothic is used everywhere — it covers Latin + Korean.
# Courier is avoided because it has no Korean glyphs.
_KO   = "Nanum"
_KO_B = "Nanum-Bold"


def _ko(text):
    """Wrap a Korean string so it always renders with NanumGothic inside a Paragraph."""
    return f'<font name="{_KO}">{text}</font>'


# ── Styles ─────────────────────────────────────────────────────────────────────
H1 = ParagraphStyle("H1", fontName=_KO_B, fontSize=18, spaceAfter=6,
                    textColor=colors.HexColor("#1a1a2e"), leading=26)
H2 = ParagraphStyle("H2", fontName=_KO_B, fontSize=13, spaceAfter=4,
                    spaceBefore=14, textColor=colors.HexColor("#16213e"), leading=20)
H3 = ParagraphStyle("H3", fontName=_KO_B, fontSize=11, spaceAfter=3,
                    spaceBefore=8,  textColor=colors.HexColor("#0f3460"), leading=17)
BODY = ParagraphStyle("Body", fontName=_KO, fontSize=9.5, leading=16,
                      spaceAfter=4, textColor=colors.HexColor("#222222"))
CAPTION = ParagraphStyle("Caption", fontName=_KO, fontSize=8.5,
                          textColor=colors.HexColor("#555555"), leading=13, spaceAfter=3)
# Code blocks: NanumGothic (not monospace, but Korean glyphs work — Courier has none)
CODE = ParagraphStyle("Code", fontName=_KO, fontSize=8, leading=12,
                      spaceAfter=2, textColor=colors.HexColor("#1a1a1a"),
                      backColor=colors.HexColor("#f5f5f5"),
                      leftIndent=6, rightIndent=6)

DIVIDER_COLOR = colors.HexColor("#cccccc")


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6, spaceBefore=2)

def h1(text):  return Paragraph(text, H1)
def h2(text):  return Paragraph(text, H2)
def h3(text):  return Paragraph(text, H3)
def p(text):   return Paragraph(text, BODY)
def cap(text): return Paragraph(text, CAPTION)
def sp(h=4):   return Spacer(1, h * mm)


def code(text):
    """Code block: Preformatted with NanumGothic so Korean glyphs render."""
    return Preformatted(text, CODE)


def mono(text):
    """Inline monospace-style span inside a Paragraph (uses NanumGothic)."""
    return f'<font name="{_KO}">{text}</font>'


def table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME",      (0, 0), (-1, -1), _KO),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), _KO_B),
        ]
    for i in range(1, len(data)):
        bg = colors.HexColor("#f9f9f9") if i % 2 == 0 else colors.white
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


# ── Content ────────────────────────────────────────────────────────────────────

def build():
    M = 18 * mm
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
    )

    story = []

    # ── Title ──────────────────────────────────────────────────────────────────
    story += [
        sp(2),
        h1("Utility Analysis Dashboard"),
        Paragraph("Technical Design Note",
                  ParagraphStyle("Sub", fontName=_KO, fontSize=12,
                                 textColor=colors.HexColor("#555555"), spaceAfter=2, leading=18)),
        cap("Generated 2026-03-10  ·  Internal reference"),
        hr(), sp(2),
    ]

    # ── 1. Stack ───────────────────────────────────────────────────────────────
    story += [h2("1. Stack"), hr()]
    story += [
        p("<b>Runtime:</b> Python 3.11+, Streamlit, Plotly (go + px), pandas / numpy"),
        p(f"<b>Venv:</b> shared {mono('../finance_vis/venv_finance')} — do not create a new one"),
        p("<b>Launch:</b>"),
        code("source ../finance_vis/venv_finance/bin/activate && streamlit run app.py"),
        p(f"<b>Input:</b> Korean {mono('.xlsm/.xlsx')} files loaded from "
          f"{mono('./data/')} (configurable in sidebar)"),
        sp(),
    ]

    # ── 2. Module Map ──────────────────────────────────────────────────────────
    story += [h2("2. Module Map"), hr()]
    story += [code(
"app.py              <- Page config, top-level routing, file loading\n"
"sidebar.py          <- File loader (dir scan -> _FileEntry list), bins/tail stubs\n"
"data.py             <- Excel I/O (@st.cache_data), sheet name constants, billing period\n"
"features.py         <- Column engineering: create_change_columns, build_from_two_files,\n"
"                       aggregate_by_brand, split_brand_by_floor, floor parsing helpers\n"
"filters.py          <- render_meter_filters, show_filter_widgets, apply_sheet_filter\n"
"viz.py              <- plot_hist_with_tails (Plotly go.Bar histogram, IQR shading)\n"
"lang.py             <- t() translation helper (Korean only, key->string)\n"
"\n"
"meter_view.py       <- \uac80\uce68\ub0b4\uc5ed full pipeline: load -> filter -> histograms -> 5 tabs\n"
"summary.py          <- \ucd1d \uc720\ud2f8\ub9ac\ud2f0 \uc21c\uc704 + \uba74\uc801\ub2f9 \ucd1d\ube44\uc6a9\n"
"billing.py          <- \uc218\ub3c4\uad11\uc5f4\ube44 \ubd80\uacfc \ub0b4\uc5ed view\n"
"ehp.py              <- EHP(OAC)\uac80\uce68\uc790\ub8cc view\n"
"water.py / hotwater.py / electricity.py  <- per-sheet views\n"
"brand_profile.py    <- Brand profile tab with peer comparison charts\n"
"\n"
"tab_anomaly.py      <- \uc774\uc0c1\uac10\uc9c0 tab (composite score, heatmap, spike/cost/HVAC sub-tabs)\n"
"tab_cross.py        <- \ube44\uc6a9\ubd84\uc11d tab (unit costs, electricity breakdown)\n"
"tab_efficiency.py   <- \ud6a8\uc728\ubd84\uc11d tab (per-m2 usage benchmarking)\n"
"tab_corr.py / tab_reconciliation.py  <- correlation & reconciliation tabs\n"
"\n"
"anomaly_features.py <- build_anomaly_df: meter+billing+elec+water -> anomaly scores\n"
"cross_features.py   <- build_unit_costs, build_elec_breakdown\n"
"report.py           <- generate_report_pdf (\uac80\uce68\ub0b4\uc5ed main report)\n"
"biz_report.py       <- generate_anomaly_pdf, generate_cross_pdf, generate_efficiency_pdf\n"
"billing_report.py / ehp_report.py / hvac_report.py  <- per-sheet PDF generators"
), sp()]

    # ── 3. Navigation Architecture ─────────────────────────────────────────────
    story += [h2("3. Navigation Architecture"), hr()]
    story += [code(
"sac.tabs (sidebar, left-position) -> nav_mode\n"
"  +-- \uc2dc\ud2b8 \ubcf4\uae30     -> sheet selectbox -> billing/ehp/water/hotwater/electricity/meter_view\n"
"  +-- \ubd84\uc11d          -> analysis selectbox\n"
"  |     +-- \uc694\uc57d \ubd84\uc11d -> render_summary_view(water_df, hw_df, el_df)\n"
"  |     +-- \uc5c5\uccb4 \ubd84\uc11d -> render_meter_filters -> [\uc774\uc0c1\uac10\uc9c0|\ube44\uc6a9\ubd84\uc11d|\ud6a8\uc728\ubd84\uc11d] tabs\n"
"  +-- \ube0c\ub79c\ub4dc \ud504\ub85c\ud544  -> render_brand_profile_tab"
),
        sp(2),
        p(f"<b>Key quirk:</b> {mono('sac.tabs')} must render directly inside "
          f"{mono('with st.sidebar:')} — <b>never</b> inside {mono('st.empty()')}. "
          "The empty placeholder causes the custom component to re-initialize on every "
          "rerun, losing state."),
        sp(),
    ]

    # ── 4. Data Pipeline ───────────────────────────────────────────────────────
    story += [h2(f"4. Data Pipeline ({_ko('검침내역')})"), hr()]
    story += [code(
"read_sheet()                      # raw Excel, header=[2, 3, 4]\n"
"  +-  apply_header_rows()         # flatten MultiIndex headers -> named columns\n"
"       +-  build_from_two_files() # prev file -> *_previous = prev month usage\n"
"            +-  create_change_columns()  # *_change = curr-prev, *_pct = change/prev*100\n"
"                 +-  aggregate_by_brand()    # group by (brand, building), sum usage\n"
"                      +-  split_brand_by_floor()  # divide totals equally by floor count"
),
        sp(2),
        p(f"<b>Cumulative vs usage:</b> Meter readings in Excel are cumulative. "
          f"{mono('build_from_two_files')} renames them to {mono('*_meter_curr/prev')} and "
          f"uses the pre-computed usage column ({mono('water_usage_m3')} etc.) "
          "as the actual current-month consumption."),
        sp(2),
        h3("Column Naming Convention"),
        table([
            ["Column", "Meaning"],
            ["water_previous / water_current", "This month's usage from prev/cur file"],
            ["water_change", "current - previous"],
            ["water_pct", "change / previous x 100"],
            ["water_meter_prev / water_meter_curr", "Original cumulative readings (kept for backward detection)"],
        ], col_widths=[90*mm, 90*mm]),
        sp(),
    ]

    # ── 5. Floor Logic ─────────────────────────────────────────────────────────
    story += [h2("5. Floor Logic"), hr()]
    story += [
        p(f"Floor values are compound strings: {mono('\"1F/2F\"')}, "
          f"{mono('\"2F~5F\"')}, {mono('\"B2F/B1F\"')}."),
        sp(1),
        table([
            ["Function", "Purpose"],
            ["parse_floor_value(s)", "Returns list of individual floor strings from compound value"],
            ["get_simple_floors(df)", "Sorted unique floors for multiselect widget"],
            ["split_brand_by_floor(df, sel_floors)", "Divides brand totals equally by count of matched floors"],
        ], col_widths=[75*mm, 105*mm]),
        sp(2),
        p(f"When {mono('sel_floors == [\"All\"]')} and {mono('sel_bldg == [\"All\"]')} "
          "-> no splitting, use aggregate totals."),
        sp(),
    ]

    # ── 6. Histogram System ────────────────────────────────────────────────────
    story += [h2("6. Histogram System (viz.py: plot_hist_with_tails)"), hr()]
    story += [
        h3("Signature (key params)"),
        code(
"plot_hist_with_tails(s, bins, lo, hi, title,\n"
"    source_df=None, val_col=None, key=\"hist\",\n"
"    display_cols=None, tail_pct=None, val_scale=1.0)"
),
        sp(2),
        p("• <b>lo / hi</b> must be in the <b>same units as the plotted series s</b>"),
        p(f"• <b>val_scale</b> bridges when {mono('s')} is scaled (e.g. /1e4 for "
          f"{_ko('만원')}) but {mono('source_df[val_col]')} is raw {_ko('원')}:"),
        code("mask = (source_df[val_col] / val_scale >= x0) & (source_df[val_col] / val_scale <= x1)"),
        p(f"• <b>Style locked:</b> blue normal bars (#4C72B0), amber tail bars (#DD8A00), "
          "red median line (#C44E52), white bg, height=380px"),
        p(f"• <b>Outlier table:</b> when source_df provided, auto-shows "
          f"{_ko('이상치 목록')} expander with all rows outside [lo, hi]"),
        sp(2),
        h3("IQR Detection Pattern"),
        code(
"q1, q3 = s.quantile(0.25), s.quantile(0.75)\n"
"iqr    = q3 - q1\n"
"lo     = q1 - k * iqr   # k slider: 0.5-3.0, default 1.5, step 0.25\n"
"hi     = q3 + k * iqr\n"
"# Show LaTeX:\n"
"st.markdown(\n"
"    f\"$$Q_1={q1:,.0f},\\;Q_3={q3:,.0f},\\;IQR={iqr:,.0f}$$\\n\\n\"\n"
"    f\"$$\\\\text{Lower}={lo:,.0f},\\;\\\\text{Upper}={hi:,.0f}\\;(k={k})$$\"\n"
")"
),
        sp(),
    ]

    # ── 7. Money Formatting ────────────────────────────────────────────────────
    story += [h2(f"7. Money Formatting ({_ko('원')} units)"), hr()]
    story += [code(
"def _fmt_won(v):\n"
"    if abs(v) >= 1e8: return f\"{v/1e8:,.0f} \uc5b5\uc6d0\"\n"
"    if abs(v) >= 1e4: return f\"{v/1e4:,.0f} \ub9cc\uc6d0\"\n"
"    return f\"{v:,.0f} \uc6d0\"\n"
"\n"
"# For bar charts -- compute _div from max(series):\n"
"_div, _unit = (1e8, \"\uc5b5\uc6d0\") if _max >= 1e8 else (1e4, \"\ub9cc\uc6d0\")\n"
"_xv = series / _div\n"
"# pass val_scale=_div to plot_hist_with_tails so bin-click filter still works"
),
        p(f"Rule: <b>no decimals on {_ko('원')} values</b> — always {mono(':.0f')}."),
        sp(),
    ]

    # ── 8. Anomaly Scoring ────────────────────────────────────────────────────
    story += [h2("8. Anomaly Scoring (anomaly_features.py)"), hr()]
    story += [
        p("Composite score = weighted sum of 5 components, each normalized to [0, 1]:"),
        sp(1),
        table([
            ["Component", "Weight", "Signal", "Source sheet"],
            [f"{_ko('급등')} Spike",        "30%", "MoM % change vs thresholds (100/50/20%)", _ko("검침내역")],
            [f"{_ko('소비')} Consumption",  "25%", "Quadrant scores: HH=4, HL=3, LH=2, Normal=1, LL=0", _ko("검침내역")],
            [f"{_ko('비용')} Cost",         "25%", "Z-scores of W/m3, W/kWh, 10kW/m2", _ko("수도광열비 부과 내역")],
            ["HVAC",                         "10%", "kWh/m2 IQR-normalized", _ko("전체 전기 사용내역")],
            [f"{_ko('일관성')} Consistency", "10%", "Count of zero-usage utilities", _ko("검침내역") + " + sheets"],
        ], col_widths=[38*mm, 18*mm, 80*mm, 44*mm]),
        sp(2),
        p(f"<b>Risk levels:</b> {_ko('위험')} (red) >= 0.65  ·  {_ko('주의')} (orange) >= 0.40  "
          f"·  {_ko('관찰')} (yellow) >= 0.20  ·  {_ko('정상')} (green) < 0.20"),
        sp(),
    ]

    # ── 9. PDF Generation Pattern ─────────────────────────────────────────────
    story += [h2("9. PDF Generation Pattern (all biz tabs)"), hr()]
    story += [code(
"_pdf_key = f\"{tab}_pdf_{file_name}\"\n"
"_col_gen, _col_dl = st.columns([1, 2])\n"
"with _col_gen:\n"
"    if st.button(\"PDF \ub9ac\ud3ec\ud2b8 \uc0dd\uc131\", key=f\"gen_{tab}_pdf_{file_name}\"):\n"
"        with st.spinner(\"\ud3f4 \uc0dd\uc131 \uc911...\"):\n"
"            st.session_state[_pdf_key] = generate_X_pdf(df)\n"
"if _pdf_key in st.session_state:\n"
"    with _col_dl:\n"
"        st.download_button(\"\ub2e4\uc6b4\ub85c\ub4dc\", st.session_state[_pdf_key], ...)"
),
        p("Generate-then-cache pattern: PDF bytes stored in session_state to avoid "
          "re-generating on every Streamlit rerun. Generate and download buttons "
          "are separate widgets in adjacent columns."),
        sp(),
    ]

    # ── 10. Common Pitfalls ───────────────────────────────────────────────────
    story += [h2("10. Common Pitfalls"), hr()]
    pitfalls = [
        ["#", "Issue", "Fix"],
        ["1", "Closure bug in loops",
         "Default arg binding: def fn(key, _p=p, _cc=cc): — never close over loop vars directly"],
        ["2", "sac.tabs state loss",
         f"Never wrap in {mono('st.empty()')}. Use {mono('with st.sidebar:')} directly."],
        ["3", "Bin-click scale mismatch",
         f"When series is scaled, pass {mono('val_scale=factor')} to {mono('plot_hist_with_tails')}"],
        ["4", "Radio option ordering",
         f"Histogram always first: {mono('[\"히스토그램\", ...]')} or {mono('[\"% Change only\", \"Change only\", \"Side by Side\"]')}"],
        ["5", f"Decimals on {_ko('원')} values",
         f"Always {mono(':.0f')}, never {mono(':.2f')}"],
        ["6", "EHP column parsing",
         "Cumulative cols M-DG (0-based 12-110); stop at next section header; do NOT merge two tables"],
        ["7", "sac.tabs ValueError",
         f"Stored index >= tab count. Clear {mono('nav_{{file_name}}')} session state key to reset"],
    ]
    story += [table(pitfalls, col_widths=[8*mm, 52*mm, 120*mm]), sp()]

    # ── 11. Session State Keys ────────────────────────────────────────────────
    story += [h2("11. Notable Session State Keys"), hr()]
    story += [
        table([
            ["Key", "Purpose"],
            ["nav_{file_name}",              "Sidebar nav tab index (sac.tabs)"],
            ["anomaly_loaded_{file_name}",   f"Gate: has {_ko('이상감지')} analysis been started?"],
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
        p(f"• <b>{_ko('면적당 총비용')} unit scaling</b>: Should display as "
          f"{_ko('만원')}/m2 (not bare {_ko('만원')} or {_ko('원')}/m2). "
          "Reverted at user request — ready to re-implement cleanly."),
        p("• No other outstanding tasks as of 2026-03-10."),
        sp(4),
    ]

    doc.build(story)
    print(f"PDF written -> {OUT}")


if __name__ == "__main__":
    build()

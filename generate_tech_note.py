"""Generate technical design note PDF for utility_analysis dashboard."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
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
pdfmetrics.registerFont(TTFont("Nanum",      "/home/hansoullee20/.fonts/NanumGothic-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Nanum-Bold", "/home/hansoullee20/.fonts/NanumGothic-Bold.ttf"))
_KO   = "Nanum"
_KO_B = "Nanum-Bold"


def _ko(text):
    return f'<font name="{_KO}">{text}</font>'


# ── Styles ─────────────────────────────────────────────────────────────────────
H1 = ParagraphStyle("H1", fontName=_KO_B, fontSize=18, spaceAfter=6,
                    textColor=colors.HexColor("#1a1a2e"), leading=26)
H2 = ParagraphStyle("H2", fontName=_KO_B, fontSize=13, spaceAfter=4,
                    spaceBefore=14, textColor=colors.HexColor("#16213e"), leading=20)
H3 = ParagraphStyle("H3", fontName=_KO_B, fontSize=11, spaceAfter=3,
                    spaceBefore=8,  textColor=colors.HexColor("#0f3460"), leading=17)
BODY    = ParagraphStyle("Body",    fontName=_KO, fontSize=9.5, leading=16,
                          spaceAfter=4, textColor=colors.HexColor("#222222"))
CAPTION = ParagraphStyle("Caption", fontName=_KO, fontSize=8.5,
                          textColor=colors.HexColor("#555555"), leading=13, spaceAfter=3)
CODE    = ParagraphStyle("Code",    fontName=_KO, fontSize=8, leading=12,
                          spaceAfter=2, textColor=colors.HexColor("#1a1a1a"),
                          backColor=colors.HexColor("#f5f5f5"),
                          leftIndent=6, rightIndent=6)
DIVIDER_COLOR = colors.HexColor("#cccccc")


def hr():    return HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6, spaceBefore=2)
def h1(t):   return Paragraph(t, H1)
def h2(t):   return Paragraph(t, H2)
def h3(t):   return Paragraph(t, H3)
def p(t):    return Paragraph(t, BODY)
def cap(t):  return Paragraph(t, CAPTION)
def sp(h=4): return Spacer(1, h * mm)
def code(t): return Preformatted(t, CODE)
def mono(t): return f'<font name="{_KO}">{t}</font>'


def tbl(data, col_widths=None, header=True):
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
    M   = 18 * mm
    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M)
    story = []

    # Title
    story += [
        sp(2),
        h1("Utility Analysis Dashboard"),
        Paragraph("Technical Design Note",
                  ParagraphStyle("Sub", fontName=_KO, fontSize=12,
                                 textColor=colors.HexColor("#555555"), spaceAfter=2, leading=18)),
        cap("Generated 2026-03-10  \u00b7  Internal reference"),
        hr(), sp(2),
    ]

    # 1. Stack
    story += [h2("1. Stack"), hr(),
        p("<b>Runtime:</b> Python 3.11+, Streamlit, Plotly (go + px), pandas / numpy"),
        p(f"<b>Venv:</b> shared {mono('../finance_vis/venv_finance')} — do not create a new one"),
        p("<b>Launch:</b>"),
        code("source ../finance_vis/venv_finance/bin/activate && streamlit run app.py"),
        p(f"<b>Input:</b> Korean {mono('.xlsm/.xlsx')} files — buildings A/B/C/D, "
          f"sheet {_ko('검침 내역')}"),
        sp(),
    ]

    # 2. Module Map
    story += [h2("2. Module Map"), hr(), code(
"app.py              <- Page config, top-level routing, file loading\n"
"sidebar.py          <- File loader, bins/tail stubs\n"
"data.py             <- Excel I/O (@st.cache_data), sheet constants, billing period\n"
"features.py         <- Column engineering: create_change_columns, aggregate_by_brand,\n"
"                       split_brand_by_floor, floor parsing, cols_brand_then_category\n"
"filters.py          <- render_meter_filters, show_filter_widgets, apply_sheet_filter\n"
"viz.py              <- plot_hist_with_tails (Plotly go.Bar histogram + IQR + bin slider)\n"
"utils_plot.py       <- bar_chart() shared Plotly bar helper (tab_cross + tab_efficiency)\n"
"lang.py             <- t() translation helper\n"
"\n"
"meter_view.py       <- \u00ab\uac80\uce68\ub0b4\uc5ed\u00bb full pipeline: load -> filter -> histograms -> 3 tabs\n"
"summary.py          <- \uc694\uc57d \ubd84\uc11d view\n"
"billing.py          <- \uc218\ub3c4\uad11\uc5f4\ube44 \ubd80\uacfc \ub0b4\uc5ed view\n"
"ehp.py / water.py / hotwater.py / electricity.py  <- per-sheet views\n"
"brand_profile.py    <- Brand profile tab with peer comparison charts\n"
"\n"
"tab_anomaly.py      <- \uc774\uc0c1\uac10\uc9c0 tab (composite score, heatmap, spike/cost/HVAC)\n"
"tab_cross.py        <- \ube44\uc6a9\ubd84\uc11d tab (unit costs, electricity breakdown)\n"
"tab_efficiency.py   <- \ud6a8\uc728\ubd84\uc11d tab (per-m2 usage benchmarking)\n"
"tab_corr.py / tab_reconciliation.py  <- correlation & reconciliation tabs\n"
"\n"
"anomaly_features.py <- build_anomaly_df: meter+billing+elec+water -> anomaly scores\n"
"cross_features.py   <- build_unit_costs, build_elec_breakdown\n"
"biz_report.py       <- generate_anomaly/cross/efficiency_pdf + render_pdf_buttons()\n"
"report.py           <- generate_report_pdf (\uac80\uce68\ub0b4\uc5ed main report)\n"
"billing_report.py / ehp_report.py / hvac_report.py  <- per-sheet PDF generators"
), sp()]

    # 3. Navigation
    story += [h2("3. Navigation Architecture"), hr(), code(
"sac.tabs (sidebar, left) -> nav_mode\n"
"  +-- \uc2dc\ud2b8 \ubcf4\uae30    -> sheet selectbox -> billing/ehp/water/hotwater/electricity/meter_view\n"
"  +-- \ubd84\uc11d         -> analysis selectbox\n"
"  |     +-- \uc694\uc57d \ubd84\uc11d -> render_summary_view(water_df, hw_df, el_df)\n"
"  |     +-- \uc5c5\uccb4 \ubd84\uc11d -> render_meter_filters -> [\uc774\uc0c1\uac10\uc9c0|\ube44\uc6a9\ubd84\uc11d|\ud6a8\uc728\ubd84\uc11d] tabs\n"
"  +-- \ube0c\ub79c\ub4dc \ud504\ub85c\ud544 -> render_brand_profile_tab"
),
        sp(2),
        p(f"<b>Key quirk:</b> {mono('sac.tabs')} must render directly inside "
          f"{mono('with st.sidebar:')} — never inside {mono('st.empty()')}. "
          "The empty placeholder causes the component to re-initialize on every rerun, losing state."),
        sp(),
    ]

    # 4. Data Pipeline
    story += [h2(f"4. Data Pipeline ({_ko('검침내역')})"), hr(), code(
"read_sheet()                      # raw Excel, header=[2, 3, 4]\n"
"  +-  apply_header_rows()         # flatten MultiIndex -> named columns\n"
"       +-  build_from_two_files() # prev file -> *_previous = prev month usage\n"
"            +-  create_change_columns()  # *_change = curr-prev, *_pct\n"
"                 +-  aggregate_by_brand()   # group by (brand, building)\n"
"                      +-  split_brand_by_floor()  # divide by floor count"
),
        sp(2),
        p(f"<b>Cumulative vs usage:</b> Meter readings are cumulative. "
          f"{mono('build_from_two_files')} uses the pre-computed usage column "
          f"({mono('water_usage_m3')} etc.) as actual current-month consumption."),
        sp(2),
        h3("Column Naming"),
        tbl([
            ["Column",                         "Meaning"],
            ["water_previous / water_current", "This month's usage from prev/cur file"],
            ["water_change",                   "current - previous"],
            ["water_pct",                      "change / previous x 100"],
            ["water_meter_prev/curr",          "Original cumulative readings (kept for reference)"],
        ], col_widths=[90*mm, 90*mm]),
        sp(),
    ]

    # 5. Floor Logic
    story += [h2("5. Floor Logic"), hr(),
        p(f"Floor values are compound strings: {mono('\"1F/2F\"')}, "
          f"{mono('\"2F~5F\"')}, {mono('\"B2F/B1F\"')}."),
        sp(1),
        tbl([
            ["Function",                    "Purpose"],
            ["parse_floor_value(s)",        "Returns list of individual floor strings from compound value"],
            ["get_simple_floors(df)",       "Sorted unique floors for multiselect widget"],
            ["split_brand_by_floor(df, sel_floors)", "Divides brand totals equally by floor count"],
        ], col_widths=[75*mm, 105*mm]),
        sp(2),
        p(f"When {mono('sel_floors == [\"All\"]')} and {mono('sel_bldg == [\"All\"]')} "
          "-> no splitting, use aggregate totals."),
        sp(),
    ]

    # 6. Histogram System
    story += [h2(f"6. Histogram System ({mono('viz.py: plot_hist_with_tails')})"), hr(),
        h3("Signature"),
        code(
"plot_hist_with_tails(s, bins, lo, hi, title,\n"
"    source_df=None, val_col=None, key=\"hist\",\n"
"    display_cols=None, tail_pct=None,\n"
"    val_scale=1.0, show_bins_slider=True)"
),
        sp(2),
        p(f"• {mono('show_bins_slider=True')} renders a Bins slider (5–200, default 50, step 5) "
          "above the chart. Pass {mono('False')} for callers that provide their own bins control "
          "(e.g. {mono('_hist_controls')} in meter_view.py)."),
        p(f"• {mono('val_scale')} bridges when {mono('s')} is scaled (e.g. /1e4 for "
          f"{_ko('만원')}) but {mono('source_df[val_col]')} is raw {_ko('원')}:"),
        code("mask = (source_df[val_col] / val_scale >= x0) & (...<= x1)"),
        p(f"• Outlier table: separate top / bottom tails, "
          f"columns: brand + val_col + {{prefix}}_current + {{prefix}}_previous + building + floor. "
          f"Auto-derived when {mono('val_col')} ends in {mono('_change')} or {mono('_pct')}."),
        p("• Bin-click table: same column order as outlier table, shown directly below stats row."),
        p(f"• Style locked: blue normal bars (#4C72B0), amber tails (#DD8A00), "
          "red median line (#C44E52), white bg, height=380px, y-axis: Count (건)"),
        sp(2),
        h3("IQR Controls (meter_view.py)"),
        code(
"# Per-category tab:\n"
"bins, lo, hi = _hist_controls(key_prefix)  # bins slider -> k slider -> equation\n"
"plot_hist_with_tails(..., bins=bins, show_bins_slider=False)\n"
"\n"
"# All-category section:\n"
"_abins = st.slider(\"Bins\", ...)          # bins slider first\n"
"_ak    = st.slider(\"IQR k\", ...)         # k slider second\n"
"st.markdown(\"$$Q_1=..., IQR=...$$ ...\")\n"
"plot_hist_with_tails(..., bins=_abins, show_bins_slider=False)"
),
        sp(),
    ]

    # 7. Shared bar_chart helper
    story += [h2(f"7. Shared Bar Chart ({mono('utils_plot.bar_chart')})"), hr(),
        code(
"bar_chart(df, x, y, title, y_label,\n"
"    color_col=\"building\",  # str column name or None\n"
"    key=None, height=420)"
),
        p("Used by tab_cross.py and tab_efficiency.py — replaces duplicate "
          f"{mono('_bar')} functions. Supports click-to-show-row interaction. "
          "Building colour map: A=#1f77b4, B=#d62728, C=#2ca02c, D=#9467bd."),
        sp(),
    ]

    # 8. PDF Generation
    story += [h2("8. PDF Generation Pattern"), hr(),
        h3(f"Shared helper: {mono('biz_report.render_pdf_buttons')}"),
        code(
"render_pdf_buttons(\n"
"    pdf_key,          # session_state key for cached bytes\n"
"    generator_fn,     # lambda: generate_X_pdf(df)\n"
"    dl_label,         # download button label\n"
"    dl_filename,      # .pdf filename\n"
")"
),
        p("All three biz tabs call this helper. PDF bytes are cached in session_state "
          "so regeneration is only triggered on button press, not on every rerun."),
        sp(2),
        h3("PDF button placement"),
        p("Buttons appear at the <b>top</b> of each tab — immediately after data is loaded "
          "and KPIs are shown — so they are accessible without scrolling."),
        sp(2),
        h3("Raw data dropdown"),
        p(f"Each tab shows a {mono('st.expander(\"📊 원시 데이터\")')} at the top with the "
          "underlying DataFrame — allows quick inspection of the data feeding the charts."),
        sp(),
    ]

    # 9. Money Formatting
    story += [h2(f"9. Money Formatting ({_ko('원')} units)"), hr(), code(
"def _fmt_won(v):\n"
"    if abs(v) >= 1e8: return f\"{v/1e8:,.0f} \uc5b5\uc6d0\"\n"
"    if abs(v) >= 1e4: return f\"{v/1e4:,.0f} \ub9cc\uc6d0\"\n"
"    return f\"{v:,.0f} \uc6d0\"\n"
"\n"
"_div, _unit = (1e8, \"\uc5b5\uc6d0\") if _max >= 1e8 else (1e4, \"\ub9cc\uc6d0\")\n"
"# pass val_scale=_div to plot_hist_with_tails so bin-click filter works"
),
        p(f"Rule: <b>no decimals on {_ko('원')} values</b> — always {mono(':.0f')}."),
        sp(),
    ]

    # 10. Anomaly Scoring
    story += [h2("10. Anomaly Scoring (anomaly_features.py)"), hr(),
        p("Composite score = weighted sum of 5 components, each normalized to [0, 1]:"),
        sp(1),
        tbl([
            ["Component",                     "Weight", "Signal",                                        "Source"],
            [f"{_ko('급등')} Spike",           "30%",   "MoM % change vs thresholds (100/50/20%)",       _ko("검침내역")],
            [f"{_ko('소비')} Consumption",     "25%",   "Quadrant scores HH=4 HL=3 LH=2 Normal=1 LL=0", _ko("검침내역")],
            [f"{_ko('비용')} Cost",            "25%",   "Z-scores of W/m3, W/kWh, 10kW/m2",             _ko("수도광열비 부과 내역")],
            ["HVAC",                            "10%",   "kWh/m2 IQR-normalised",                        _ko("전체 전기 사용내역")],
            [f"{_ko('일관성')} Consistency",   "10%",   "Count of zero-usage utilities",                 _ko("검침내역") + " + sheets"],
        ], col_widths=[38*mm, 18*mm, 80*mm, 44*mm]),
        sp(2),
        p(f"<b>Risk levels:</b> {_ko('위험')} >= 0.65  ·  {_ko('주의')} >= 0.40  "
          f"·  {_ko('관찰')} >= 0.20  ·  {_ko('정상')} < 0.20"),
        sp(),
    ]

    # 11. Common Pitfalls
    story += [h2("11. Common Pitfalls"), hr(),
        tbl([
            ["#", "Issue",                      "Fix"],
            ["1", "Closure bug in loops",        "Default arg binding: def fn(key, _p=p, _cc=cc)"],
            ["2", "sac.tabs state loss",          f"Never wrap in {mono('st.empty()')} — use {mono('with st.sidebar:')}"],
            ["3", "Bin-click scale mismatch",    f"Pass {mono('val_scale=factor')} to {mono('plot_hist_with_tails')}"],
            ["4", "Duplicate bins slider",        f"Pass {mono('show_bins_slider=False')} when caller controls bins"],
            ["5", f"Decimals on {_ko('원')}",    f"Always {mono(':.0f')}, never {mono(':.2f')}"],
            ["6", "EHP column parsing",           "Cumulative cols M-DG (0-based 12-110); stop at next header"],
            ["7", "sac.tabs ValueError",          f"Stored index >= tab count — clear {mono('nav_{{file_name}}')} key"],
        ], col_widths=[8*mm, 52*mm, 120*mm]), sp()]

    # 12. Session State Keys
    story += [h2("12. Notable Session State Keys"), hr(),
        tbl([
            ["Key",                              "Purpose"],
            ["nav_{file_name}",                  "Sidebar nav tab index (sac.tabs)"],
            ["anomaly_loaded_{file_name}",        f"Gate: {_ko('이상감지')} analysis started?"],
            ["cross_loaded_{file_name}",          "Gate: cross-sheet data loaded?"],
            ["anomaly_pdf_{file_name}",           "Cached anomaly PDF bytes"],
            ["cross_pdf_{file_name}",             "Cached cross-tab PDF bytes"],
            ["eff_pdf_{file_name}",               "Cached efficiency PDF bytes"],
            ["{prefix}_iqr_k",                   "IQR k multiplier per histogram section"],
            ["{prefix}_bins / {prefix}_bins_i",  "Bins slider/input sync pair"],
            ["all_bins_{p} / all_bins_i_{p}",    "Bins control for all-category histogram section"],
        ], col_widths=[90*mm, 90*mm]),
        sp(),
    ]

    # 13. Pending
    story += [h2("13. Pending / Known Issues"), hr(),
        p(f"• <b>{_ko('면적당 총비용')} unit scaling</b>: Display as "
          f"{_ko('만원')}/m2 — reverted at user request, ready to re-implement."),
        p("• No other outstanding tasks as of 2026-03-10."),
        sp(4),
    ]

    doc.build(story)
    print(f"PDF written -> {OUT}")


if __name__ == "__main__":
    build()

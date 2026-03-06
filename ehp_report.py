"""
ehp_report.py  —  Business-ready PDF for EHP 전기 사용량 분석
"""
import io
import re
import textwrap
from datetime import date as _today_date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Image, KeepTogether,
    PageBreak, Paragraph, Spacer, Table, TableStyle,
)

from report import (
    C_BLUE, C_CRITICAL, C_DIVIDER, C_LIGHT, C_NAVY, C_STABLE, C_WHITE,
    M_BAR, M_CRITICAL,
    _ensure_fonts, _make_numbered_canvas, _make_page_template,
    _make_styles, _png, _section_bar,
)

_PALETTE = ["#4C72B0", "#DD8A00", "#C44E52", "#55A868", "#8172B2", "#937860"]
_YM_PAT  = re.compile(r'^(20\d{2})_(\d{1,2})월$')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(val, decimals=0):
    if val is None:
        return "—"
    try:
        f = float(val)
        return "—" if np.isnan(f) else f"{f:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _divider_line(content_w):
    return Table(
        [[""]],
        colWidths=[content_w],
        style=TableStyle([
            ("LINEABOVE",     (0, 0), (-1, -1), 2, C_NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]),
    )


def _img_flow(buf, width_cm, styles, caption=""):
    if buf is None:
        return []
    pil = PILImage.open(buf)
    w_px, h_px = pil.size
    buf.seek(0)
    items = [Image(buf, width=width_cm * cm, height=width_cm * cm * h_px / w_px)]
    if caption:
        items.append(Paragraph(caption, styles["caption"]))
    return items


def _std_table(data, col_w, styles, *, header_bg=None):
    header_bg = header_bg or C_NAVY
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
    ])
    return Table(data, colWidths=col_w, style=ts, repeatRows=1)


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_yearly(year_totals: dict) -> io.BytesIO:
    """Bar chart of annual totals."""
    if not year_totals:
        return None
    years  = [str(y) for y in sorted(year_totals)]
    values = [float(year_totals[int(y)]) for y in years]
    avg    = float(np.mean(values)) if values else 0

    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor="white")
    bar_colors = [M_CRITICAL if v > avg * 1.3 else M_BAR for v in values]
    bars = ax.bar(years, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    if avg > 0:
        ax.axhline(avg, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8,
                   label=f"평균: {avg:,.0f} kWh")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(values) * 0.01,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_ylabel("kWh", fontsize=9)
    ax.set_title("연간 EHP 전기 사용량", fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle="--")
    if avg > 0:
        ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_monthly_trend(pivot: pd.DataFrame) -> io.BytesIO:
    """Line chart of monthly usage per year."""
    if pivot.empty:
        return None
    month_order = {f"{m}월": m for m in range(1, 13)}
    available   = [m for m in [f"{m}월" for m in range(1, 13)] if m in pivot.index]
    if not available:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 4.0), facecolor="white")
    for i, yr in enumerate(sorted(pivot.columns)):
        series = pivot.loc[[m for m in available if yr in pivot.columns], yr].dropna()
        if series.empty:
            continue
        x_ord = [month_order.get(m, 0) for m in series.index]
        sorted_pairs = sorted(zip(x_ord, series.index, series.values), key=lambda t: t[0])
        x_lbl = [t[1] for t in sorted_pairs]
        y_val = [t[2] for t in sorted_pairs]
        ax.plot(x_lbl, y_val,
                color=_PALETTE[i % len(_PALETTE)], linewidth=2,
                marker="o", markersize=4, label=str(yr))

    ax.set_ylabel("kWh", fontsize=9)
    ax.set_title("월별 EHP 전기 사용량 추세", fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_facecolor("white")
    ax.grid(color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    fig.tight_layout(pad=0.8)
    return _png(fig)


# ── Anomaly detection (mirrors ehp.py logic) ─────────────────────────────────

def _compute_anomalies(pivot: pd.DataFrame):
    if len(pivot.columns) < 2:
        return []
    records = []
    for mo_label in pivot.index:
        row = pivot.loc[mo_label].dropna()
        if len(row) < 2:
            continue
        q25, q75 = row.quantile(0.25), row.quantile(0.75)
        iqr       = q75 - q25
        hi_fence  = q75 + 1.5 * iqr
        lo_fence  = max(q25 - 1.5 * iqr, 0.0)
        med       = row.median()
        for yr, val in row.items():
            is_hi = val > hi_fence
            is_lo = (lo_fence > 0) and (val < lo_fence)
            if is_hi or is_lo:
                records.append({
                    "month":    str(mo_label),
                    "year":     str(yr),
                    "usage":    round(float(val),      0),
                    "median":   round(float(med),      0),
                    "delta":    round(float(val - med), 0),
                    "fence_hi": round(float(hi_fence),  0),
                    "fence_lo": round(float(lo_fence),  0),
                    "flag":     "High" if is_hi else "Low",
                })
    return records


# ── OAC PDF ──────────────────────────────────────────────────────────────────

def generate_ehp_oac_pdf(pivot: pd.DataFrame,
                          usage: pd.DataFrame = None,
                          context: dict = None,
                          lang: str = "ko") -> bytes:
    """
    Business-ready PDF for the OAC EHP electricity analysis.

    Parameters
    ----------
    pivot   : monthly × yearly pivot (index = "1월"…"12월", columns = years)
    usage   : per-meter usage DataFrame (optional)
    context : dict with optional 'date', 'sheet_name'
    lang    : 'ko' or 'en'
    """
    _ensure_fonts()
    styles    = _make_styles()
    ctx       = context or {}
    ko        = (lang == "ko")

    page_w, _ = A4
    margin    = 2 * cm
    content_w = page_w - 2 * margin

    footer_left     = "EHP 전기 사용량 분석 보고서  ·  대외비" if ko else "EHP Electricity Report  ·  Confidential"
    footer_page_fmt = "{n} / {total} 페이지" if ko else "Page {n} of {total}"
    T_footer        = {"footer_left": footer_left, "footer_page": footer_page_fmt}

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin,  bottomMargin=2 * cm,
    )
    doc.addPageTemplates([_make_page_template(doc, T_footer)])
    story = []

    # ── Pre-compute ──────────────────────────────────────────────────────────
    years = sorted([c for c in pivot.columns if pd.notna(c)])
    year_totals = {}
    for yr in years:
        t = pivot[yr].sum(min_count=1)
        if pd.notna(t):
            year_totals[yr] = float(t)

    grand_total = sum(year_totals.values())
    n_meters    = len(usage) if usage is not None and not usage.empty else "—"
    report_date = str(ctx.get("date", _today_date.today()))
    years_str   = ", ".join(str(y) for y in years)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═════════════════════════════════════════════════════════════════════════
    title    = "EHP 전기 사용량 분석 보고서"       if ko else "EHP Electricity Usage Report"
    subtitle = "OAC 계량기 월간 사용량 요약 분석"  if ko else "OAC Meter Monthly Usage Summary"

    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(title,    styles["cover_title"]))
    story.append(Paragraph(subtitle, styles["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_divider_line(content_w))
    story.append(Spacer(1, 0.8 * cm))

    meta_raw = [
        ("보고서 일자"   if ko else "Report Date",    report_date),
        ("데이터 기간"   if ko else "Period Covered",  years_str),
        ("계량기 수"     if ko else "Meters",           str(n_meters)),
        ("총 사용량"     if ko else "Grand Total",     f"{grand_total:,.0f} kWh"),
    ]
    if ctx.get("sheet_name"):
        meta_raw.insert(0, ("시트" if ko else "Sheet", ctx["sheet_name"]))

    meta_data = [[Paragraph(k, styles["table_cell"]), Paragraph(v, styles["table_cell"])]
                 for k, v in meta_raw]
    meta_ts   = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1),  "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
    ])
    story.append(Table(meta_data, colWidths=[4.5*cm, content_w - 4.5*cm], style=meta_ts))
    story.append(Spacer(1, 1.0 * cm))

    # Yearly totals table (cover)
    if year_totals:
        yr_list = sorted(year_totals)
        yr_hdr  = (["연도" if ko else "Year",
                    "사용량 (kWh)" if ko else "Usage (kWh)",
                    "전년 대비 (%)" if ko else "vs. Prev. Year (%)"])
        yr_data = [[Paragraph(h, styles["table_hdr"]) for h in yr_hdr]]
        yr_ts   = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
            ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
        ])
        for i, yr in enumerate(yr_list):
            val  = year_totals[yr]
            if i > 0:
                prev     = year_totals[yr_list[i - 1]]
                chg_pct  = (val - prev) / prev * 100 if prev > 0 else np.nan
                chg_str  = f"{chg_pct:+.1f}" if not np.isnan(chg_pct) else "—"
            else:
                chg_str  = "—"
            yr_data.append([
                Paragraph(str(yr), styles["table_cell_c"]),
                Paragraph(f"{val:,.0f}", styles["table_cell_c"]),
                Paragraph(chg_str, styles["table_cell_c"]),
            ])
            if i > 0:
                prev = year_totals[yr_list[i - 1]]
                if prev > 0:
                    pct = (val - prev) / prev * 100
                    if pct > 10:
                        yr_ts.add("TEXTCOLOR", (2, i + 1), (2, i + 1), C_CRITICAL)
                        yr_ts.add("FONTNAME",  (2, i + 1), (2, i + 1), "NanumGothic-Bold")
                    elif pct < -10:
                        yr_ts.add("TEXTCOLOR", (2, i + 1), (2, i + 1), C_STABLE)
                        yr_ts.add("FONTNAME",  (2, i + 1), (2, i + 1), "NanumGothic-Bold")

        yr_cw = [3.0*cm, (content_w - 3.0*cm) / 2, (content_w - 3.0*cm) / 2]
        story.append(KeepTogether([
            Paragraph("연간 사용량 요약" if ko else "Yearly Usage Summary", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(yr_data, colWidths=yr_cw, style=yr_ts),
        ]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2 — YEARLY & MONTHLY TREND
    # ═════════════════════════════════════════════════════════════════════════
    story.append(_section_bar(
        "  연간 사용량 추세" if ko else "  Yearly Usage Trend",
        styles, content_w,
    ))
    story.append(Spacer(1, 0.3 * cm))

    yr_chart_buf = _chart_yearly(year_totals)
    if yr_chart_buf:
        story += _img_flow(
            yr_chart_buf, content_w / cm, styles,
            caption="그림: 연간 총 EHP 전기 사용량 (kWh). 평균 130% 초과 연도 강조 표시."
                    if ko else
                    "Figure: Annual total EHP electricity usage (kWh). Red = >130% of average.",
        )
    story.append(Spacer(1, 0.5 * cm))

    mo_chart_buf = _chart_monthly_trend(pivot)
    if mo_chart_buf:
        story += _img_flow(
            mo_chart_buf, content_w / cm, styles,
            caption="그림: 연도별 월간 EHP 사용량 추세 (kWh)."
                    if ko else
                    "Figure: Monthly EHP usage trend per year (kWh).",
        )
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3 — MONTHLY PIVOT TABLE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(_section_bar(
        "  월별 사용량 현황" if ko else "  Monthly Usage Summary",
        styles, content_w,
    ))
    story.append(Spacer(1, 0.3 * cm))

    if not pivot.empty and years:
        month_order_lbl = [f"{m}월" for m in range(1, 13)]
        avail_months    = [m for m in month_order_lbl if m in pivot.index]

        pvt_hdr = [Paragraph("월" if ko else "Month", styles["table_hdr"])]
        for yr in years:
            pvt_hdr.append(Paragraph(f"{yr} (kWh)", styles["table_hdr"]))
            pvt_hdr.append(Paragraph("전년비 (%)" if ko else "vs Prev (%)", styles["table_hdr"]))

        pvt_data = [pvt_hdr]
        pvt_ts   = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 4),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.HexColor("#F5F7FA"), colors.white]),
            ("BACKGROUND",     (0, -1), (-1, -1), C_LIGHT),
            ("FONTNAME",       (0, -1), (-1, -1), "NanumGothic-Bold"),
            ("ALIGN",          (1, 0),  (-1, -1), "RIGHT"),
        ])

        for ri, mo in enumerate(avail_months, 1):
            row      = [Paragraph(mo, styles["table_cell"])]
            prev_val = None
            for yi, yr in enumerate(years):
                val_raw  = pivot.loc[mo, yr] if yr in pivot.columns else np.nan
                val_f    = float(val_raw) if pd.notna(val_raw) else np.nan
                row.append(Paragraph(_f(val_f) if not np.isnan(val_f) else "—", styles["table_cell_c"]))
                pct_col  = 1 + yi * 2  # "vs prev" column index
                if prev_val is not None and not np.isnan(prev_val) and not np.isnan(val_f) and prev_val > 0:
                    pct = (val_f - prev_val) / prev_val * 100
                    row.append(Paragraph(f"{pct:+.0f}", styles["table_cell_c"]))
                    if pct > 30:
                        pvt_ts.add("TEXTCOLOR", (pct_col, ri), (pct_col, ri), C_CRITICAL)
                        pvt_ts.add("FONTNAME",  (pct_col, ri), (pct_col, ri), "NanumGothic-Bold")
                    elif pct < -30:
                        pvt_ts.add("TEXTCOLOR", (pct_col, ri), (pct_col, ri), C_STABLE)
                        pvt_ts.add("FONTNAME",  (pct_col, ri), (pct_col, ri), "NanumGothic-Bold")
                else:
                    row.append(Paragraph("—", styles["table_cell_c"]))
                prev_val = val_f if not np.isnan(val_f) else None
            pvt_data.append(row)

        # Yearly total row
        tot_row  = [Paragraph("합계" if ko else "Total", styles["table_cell"])]
        prev_tot = None
        for yr in years:
            tot = year_totals.get(yr, np.nan)
            tot_row.append(Paragraph(_f(tot) if not np.isnan(tot) else "—", styles["table_cell_c"]))
            if prev_tot is not None and prev_tot > 0 and not np.isnan(tot):
                pct = (tot - prev_tot) / prev_tot * 100
                tot_row.append(Paragraph(f"{pct:+.0f}", styles["table_cell_c"]))
            else:
                tot_row.append(Paragraph("—", styles["table_cell_c"]))
            prev_tot = tot if not np.isnan(tot) else None
        pvt_data.append(tot_row)

        n_yr_cols = len(years) * 2
        mo_w      = 1.6 * cm
        yr_w      = (content_w - mo_w) / n_yr_cols if n_yr_cols else 2.0 * cm
        pvt_cw    = [mo_w] + [yr_w] * n_yr_cols
        story.append(KeepTogether([
            Paragraph(
                "각 셀: 해당 월 전체 계량기 합산 사용량 (kWh). "
                "빨간색 = 전년 대비 30% 이상 증가, 초록색 = 30% 이상 감소."
                if ko else
                "Each cell: total usage across all meters (kWh). "
                "Red = >30% above prior year; Green = >30% below.",
                styles["note"],
            ),
            Spacer(1, 0.3 * cm),
            Table(pvt_data, colWidths=pvt_cw, style=pvt_ts, repeatRows=1),
        ]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 4 — ANOMALY DETECTION
    # ═════════════════════════════════════════════════════════════════════════
    story.append(_section_bar(
        "  이상 탐지 결과" if ko else "  Anomaly Detection Results",
        styles, content_w,
    ))
    story.append(Spacer(1, 0.3 * cm))

    _anom_note = Paragraph(
        "동일 월의 연도별 사용량에 IQR × 1.5 기준을 적용하여 이상 월-연도 조합을 탐지합니다. "
        "2개 연도 이상의 데이터가 필요합니다."
        if ko else
        "IQR × 1.5 fence applied across years for each calendar month. "
        "Requires at least 2 years of data.",
        styles["note"],
    )

    anomalies = _compute_anomalies(pivot)
    if anomalies:
        anom_hdr_lbls = (["월", "연도", "사용량 (kWh)", "중앙값 (kWh)", "편차 (kWh)", "상한 (kWh)", "하한 (kWh)", "구분"]
                         if ko else
                         ["Month", "Year", "Usage (kWh)", "Median (kWh)", "Δ (kWh)", "Hi Fence (kWh)", "Lo Fence (kWh)", "Flag"])
        anom_data = [[Paragraph(h, styles["table_hdr"]) for h in anom_hdr_lbls]]
        anom_ts   = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
            ("ALIGN",          (2, 0), (6, -1),  "RIGHT"),
        ])
        for ri, rec in enumerate(anomalies, 1):
            flag_lbl = "높음" if rec["flag"] == "High" else "낮음"
            if not ko:
                flag_lbl = rec["flag"]
            anom_data.append([
                Paragraph(rec["month"],          styles["table_cell_c"]),
                Paragraph(rec["year"],           styles["table_cell_c"]),
                Paragraph(_f(rec["usage"]),      styles["table_cell_c"]),
                Paragraph(_f(rec["median"]),     styles["table_cell_c"]),
                Paragraph(f"{rec['delta']:+,.0f}", styles["table_cell_c"]),
                Paragraph(_f(rec["fence_hi"]),   styles["table_cell_c"]),
                Paragraph(_f(rec["fence_lo"]),   styles["table_cell_c"]),
                Paragraph(flag_lbl,              styles["table_cell_c"]),
            ])
            if rec["flag"] == "High":
                anom_ts.add("BACKGROUND", (7, ri), (7, ri), C_CRITICAL)
                anom_ts.add("TEXTCOLOR",  (7, ri), (7, ri), colors.white)
            else:
                anom_ts.add("BACKGROUND", (7, ri), (7, ri), C_STABLE)
                anom_ts.add("TEXTCOLOR",  (7, ri), (7, ri), colors.white)

        fixed_anom  = (1.5 + 1.2 + 2.2 + 2.2 + 1.8 + 2.2 + 2.2) * cm
        flag_w      = content_w - fixed_anom
        anom_cw     = [1.5*cm, 1.2*cm, 2.2*cm, 2.2*cm, 1.8*cm, 2.2*cm, 2.2*cm, flag_w]
        story.append(KeepTogether([
            _anom_note,
            Spacer(1, 0.3 * cm),
            Table(anom_data, colWidths=anom_cw, style=anom_ts, repeatRows=1),
        ]))
    else:
        story.append(KeepTogether([
            _anom_note,
            Spacer(1, 0.3 * cm),
            Paragraph(
                "이상 탐지된 월-연도 조합이 없습니다. 모든 값이 IQR × 1.5 기준 내에 있습니다."
                if ko else
                "No anomalies detected. All month-year values are within IQR × 1.5 fences.",
                styles["body"],
            ),
        ]))

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 5 — PER-METER YEARLY SUMMARY (optional)
    # ═════════════════════════════════════════════════════════════════════════
    if usage is not None and not usage.empty and "계량기 번호" in usage.columns:
        story.append(PageBreak())
        story.append(_section_bar(
            "  계량기별 연간 합계" if ko else "  Yearly Total by Meter",
            styles, content_w,
        ))
        story.append(Spacer(1, 0.3 * cm))

        meter_records = []
        for _, row in usage.iterrows():
            meter = str(row.get("계량기 번호", "—"))
            yr_totals_m: dict = {}
            for col in usage.columns:
                m = _YM_PAT.match(str(col))
                if not m:
                    continue
                yr  = int(m.group(1))
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val):
                    yr_totals_m[yr] = yr_totals_m.get(yr, 0) + float(val)
            record = {"meter": meter, **yr_totals_m}
            record["grand_total"] = sum(yr_totals_m.values())
            meter_records.append(record)

        if meter_records:
            m_df  = (pd.DataFrame(meter_records)
                     .sort_values("grand_total", ascending=False)
                     .reset_index(drop=True))
            yr_cols = sorted([c for c in m_df.columns if isinstance(c, int)])

            hdr_m = [Paragraph("계량기" if ko else "Meter", styles["table_hdr"])]
            for yr in yr_cols:
                hdr_m.append(Paragraph(f"{yr} (kWh)", styles["table_hdr"]))
            hdr_m.append(Paragraph("합계 (kWh)" if ko else "Total (kWh)", styles["table_hdr"]))

            m_data = [hdr_m]
            for _, row in m_df.iterrows():
                r = [Paragraph(str(row["meter"]), styles["table_cell"])]
                for yr in yr_cols:
                    v = row.get(yr, np.nan)
                    r.append(Paragraph(_f(v) if pd.notna(v) else "—", styles["table_cell_c"]))
                r.append(Paragraph(_f(row.get("grand_total", np.nan)), styles["table_cell_c"]))
                m_data.append(r)

            n_val_cols  = len(yr_cols) + 1
            meter_name_w = 3.5 * cm
            yr_val_w    = (content_w - meter_name_w) / n_val_cols
            m_cw        = [meter_name_w] + [yr_val_w] * n_val_cols
            m_ts        = TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
                ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
                ("FONTSIZE",       (0, 0), (-1, -1), 8),
                ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
                ("TOPPADDING",     (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
                ("LEFTPADDING",    (0, 0), (-1, -1), 5),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
                ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
                ("FONTNAME",       (-1, 1), (-1, -1), "NanumGothic-Bold"),
            ])
            story.append(KeepTogether([
                Paragraph(
                    "계량기별 연간 총 사용량 (kWh), 합계 기준 내림차순 정렬."
                    if ko else
                    "Total yearly usage per meter (kWh), sorted by grand total descending.",
                    styles["note"],
                ),
                Spacer(1, 0.2 * cm),
                Table(m_data, colWidths=m_cw, style=m_ts, repeatRows=1),
            ]))

    # ═════════════════════════════════════════════════════════════════════════
    # BACK MATTER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("보고서 끝" if ko else "End of Report", styles["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    end_note = (f"작성일: {report_date}. "
                "본 보고서는 내부 관리 목적으로만 사용하시기 바랍니다. "
                "단위: kWh.") if ko else \
               (f"Generated on {report_date}. "
                "For internal management use only. Unit: kWh.")
    story.append(Paragraph(end_note, styles["note"]))

    NumberedCanvas = _make_numbered_canvas(T_footer)
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


# ── Dedicated EHP PDF ─────────────────────────────────────────────────────────

def generate_ehp_dedicated_pdf(sliced_df: pd.DataFrame,
                                col0: str,
                                context: dict = None,
                                lang: str = "ko") -> bytes:
    """
    Business-ready PDF for 전용 EHP data.

    Parameters
    ----------
    sliced_df : cleaned dedicated EHP DataFrame
    col0      : name of the first (building) column
    context   : dict with optional 'date'
    lang      : 'ko' or 'en'
    """
    _ensure_fonts()
    styles    = _make_styles()
    ctx       = context or {}
    ko        = (lang == "ko")

    page_w, _ = A4
    margin    = 2 * cm
    content_w = page_w - 2 * margin

    footer_left     = "전용 EHP 분석 보고서  ·  대외비" if ko else "Dedicated EHP Report  ·  Confidential"
    footer_page_fmt = "{n} / {total} 페이지" if ko else "Page {n} of {total}"
    T_footer        = {"footer_left": footer_left, "footer_page": footer_page_fmt}

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin,  bottomMargin=2 * cm,
    )
    doc.addPageTemplates([_make_page_template(doc, T_footer)])
    story = []

    report_date = str(ctx.get("date", _today_date.today()))

    # Detect available metrics
    metric_cols = [c for c in ["전기 사용량", "매장별 가동시간", "효율 (kWh/hr)"]
                   if c in sliced_df.columns]
    metric_units = {
        "전기 사용량":      "kWh",
        "매장별 가동시간":  "hr",
        "효율 (kWh/hr)":   "kWh/hr",
    }
    all_dong = sorted(sliced_df[col0].dropna().unique(), key=str) if col0 in sliced_df.columns else []

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═════════════════════════════════════════════════════════════════════════
    title    = "전용 EHP 전기 사용량 분석 보고서" if ko else "Dedicated EHP Analysis Report"
    subtitle = "장비별 전기 사용량 및 효율 분석"  if ko else "Equipment-level Electricity & Efficiency"

    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(title,    styles["cover_title"]))
    story.append(Paragraph(subtitle, styles["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_divider_line(content_w))
    story.append(Spacer(1, 0.8 * cm))

    meta_raw = [
        ("보고서 일자" if ko else "Report Date",       report_date),
        ("건물"        if ko else "Buildings",          ", ".join(all_dong) or "—"),
        ("장비 수"     if ko else "Equipment Count",   str(len(sliced_df))),
        ("분석 항목"   if ko else "Metrics",           ", ".join(metric_cols) or "—"),
    ]
    meta_data = [[Paragraph(k, styles["table_cell"]), Paragraph(v, styles["table_cell"])]
                 for k, v in meta_raw]
    meta_ts   = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1),  "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
    ])
    story.append(Table(meta_data, colWidths=[4.5*cm, content_w - 4.5*cm], style=meta_ts))
    story.append(Spacer(1, 1.0 * cm))

    # Building-level summary table on cover
    if all_dong and metric_cols and col0 in sliced_df.columns:
        bldg_hdr = [Paragraph("건물" if ko else "Building", styles["table_hdr"])]
        for mc in metric_cols:
            bldg_hdr.append(Paragraph(f"{mc} ({metric_units.get(mc, '')})", styles["table_hdr"]))

        bldg_data_tbl = [bldg_hdr]
        for dong in all_dong:
            ddf = sliced_df[sliced_df[col0] == dong]
            row = [Paragraph(str(dong), styles["table_cell"])]
            for mc in metric_cols:
                col_vals = pd.to_numeric(ddf[mc], errors="coerce").dropna()
                row.append(Paragraph(_f(col_vals.sum()) if not col_vals.empty else "—",
                                     styles["table_cell_c"]))
            bldg_data_tbl.append(row)

        n_mc = len(metric_cols)
        bldg_cw = [3.0*cm] + [(content_w - 3.0*cm) / n_mc] * n_mc
        story.append(KeepTogether([
            Paragraph("건물별 합계" if ko else "Total by Building", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            _std_table(bldg_data_tbl, bldg_cw, styles),
        ]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PER-METRIC SECTIONS
    # ═════════════════════════════════════════════════════════════════════════
    for mc in metric_cols:
        unit = metric_units.get(mc, "")
        story.append(_section_bar(
            f"  {mc} — {'상세 분석' if ko else 'Detail Analysis'}",
            styles, content_w,
        ))
        story.append(Spacer(1, 0.3 * cm))

        tmp = sliced_df.copy()
        tmp[mc] = pd.to_numeric(tmp[mc], errors="coerce")

        # Distribution stats
        vals = tmp[mc].dropna().values.astype(float)
        if len(vals):
            stat_hdr = ["지표" if ko else "Metric", f"값 ({unit})" if ko else f"Value ({unit})"]
            stat_raw = [
                ("건수"     if ko else "Count",    str(len(vals))),
                ("합계"     if ko else "Total",    f"{vals.sum():,.0f}"),
                ("평균"     if ko else "Average",  f"{vals.mean():,.1f}"),
                ("중앙값"   if ko else "Median",   f"{np.median(vals):,.1f}"),
                ("최대"     if ko else "Max",      f"{vals.max():,.1f}"),
            ]
            stat_d = ([[Paragraph(h, styles["table_hdr"]) for h in stat_hdr]] +
                      [[Paragraph(r[0], styles["table_cell"]),
                        Paragraph(r[1], styles["table_cell_c"])] for r in stat_raw])
            stat_ts = TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0),  C_BLUE),
                ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
                ("FONTSIZE",       (0, 0), (-1, -1), 8),
                ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
                ("TOPPADDING",     (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
                ("LEFTPADDING",    (0, 0), (-1, -1), 5),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
                ("ALIGN",          (1, 1), (1, -1),  "RIGHT"),
            ])
            story.append(KeepTogether([
                Paragraph("분포 요약" if ko else "Distribution Summary", styles["sub_title"]),
                Spacer(1, 0.2 * cm),
                Table(stat_d, colWidths=[5.0*cm, content_w - 5.0*cm], style=stat_ts),
            ]))
            story.append(Spacer(1, 0.5 * cm))

        # Ranking table (by col0 / building grouping)
        if col0 in tmp.columns:
            grp = tmp.groupby(col0)[mc].sum(min_count=1).reset_index()
            grp.columns = [col0, mc]
            grp = grp.sort_values(mc, ascending=False).reset_index(drop=True)

            top_hdr = [Paragraph("순위" if ko else "#", styles["table_hdr"]),
                       Paragraph("건물" if ko else "Building", styles["table_hdr"]),
                       Paragraph(f"{mc} ({unit})", styles["table_hdr"])]
            top_data = [top_hdr]
            for rank, row in enumerate(grp.itertuples(), 1):
                top_data.append([
                    Paragraph(str(rank),             styles["table_cell_c"]),
                    Paragraph(str(getattr(row, col0, "—")), styles["table_cell"]),
                    Paragraph(_f(getattr(row, mc, np.nan), decimals=1), styles["table_cell_c"]),
                ])
            top_cw = [1.0*cm, content_w - 1.0*cm - 3.5*cm, 3.5*cm]
            story.append(KeepTogether([
                Paragraph("건물별 합계 순위" if ko else "Ranking by Building", styles["sub_title"]),
                Spacer(1, 0.2 * cm),
                _std_table(top_data, top_cw, styles),
            ]))

        # Full detail table (all rows)
        detail_cols = [c for c in [col0, "판넬명", "장비번호", "상호", mc] if c in tmp.columns]
        det_hdr_map = {col0: "건물", "판넬명": "판넬명", "장비번호": "장비번호",
                       "상호": "상호", mc: f"{mc} ({unit})"}
        det_hdr = [Paragraph(det_hdr_map.get(c, c), styles["table_hdr"]) for c in detail_cols]
        det_data = [det_hdr]
        tmp_sorted = tmp.sort_values(mc, ascending=False).reset_index(drop=True)
        for _, row in tmp_sorted.iterrows():
            r = []
            for c in detail_cols:
                val = row.get(c, "")
                if c == mc:
                    r.append(Paragraph(_f(val, decimals=1) if pd.notna(val) else "—",
                                       styles["table_cell_c"]))
                else:
                    r.append(Paragraph(
                        textwrap.shorten(str(val), 28, placeholder="…"),
                        styles["table_cell"],
                    ))
            det_data.append(r)

        fixed_det_w = 3.5 * cm  # last (metric) column
        name_cols   = [c for c in detail_cols if c != mc]
        per_name_w  = (content_w - fixed_det_w) / len(name_cols) if name_cols else content_w
        det_cw      = [per_name_w] * len(name_cols) + [fixed_det_w]

        det_ts = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
            ("ALIGN",          (-1, 0), (-1, -1), "RIGHT"),
        ])
        story.append(Spacer(1, 0.4 * cm))
        story.append(KeepTogether([
            Paragraph("전체 상세 목록" if ko else "Full Detail Listing", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(det_data, colWidths=det_cw, style=det_ts, repeatRows=1),
        ]))
        story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # BACK MATTER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("보고서 끝" if ko else "End of Report", styles["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    end_note = (f"작성일: {report_date}. "
                "본 보고서는 내부 관리 목적으로만 사용하시기 바랍니다.") if ko else \
               (f"Generated on {report_date}. For internal management use only.")
    story.append(Paragraph(end_note, styles["note"]))

    NumberedCanvas = _make_numbered_canvas(T_footer)
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()

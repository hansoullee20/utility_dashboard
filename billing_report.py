"""
billing_report.py  —  Business-ready PDF for 수도광열비 부과 내역
"""
import io
import textwrap
from datetime import date as _today_date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as _Line2D
from matplotlib.font_manager import FontProperties as _FontProperties
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
    C_BLUE, C_CRITICAL, C_DIVIDER, C_LIGHT, C_NAVY, C_WHITE,
    M_BAR, M_CRITICAL,
    _ensure_fonts, _make_numbered_canvas, _make_page_template,
    _make_styles, _png, _section_bar, _FONT_REG,
)

# ── Utility definitions (key, ko_label, en_label, excl_col, comm_col, total_col, chart_color)
_UTIL_DEFS = [
    ("water", "상하수도",  "Water",       "water_excl", "water_comm", "water_total", "#4C72B0"),
    ("elect", "전기요금",  "Electricity", "elect_excl", "elect_comm", "elect_total", "#DD8A00"),
    ("heat",  "열요금",    "Heat",        None,         None,         "heat_total",  "#C44E52"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(val, decimals=0):
    """Format a numeric value; return '—' for NaN/None."""
    if val is None:
        return "—"
    try:
        f = float(val)
        if np.isnan(f):
            return "—"
        return f"{f:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _png_from_fig(fig):
    return _png(fig)


def _divider_line(content_w):
    return Table(
        [[""]],
        colWidths=[content_w],
        style=TableStyle([
            ("LINEABOVE",      (0, 0), (-1, -1), 2, C_NAVY),
            ("TOPPADDING",     (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 0),
        ]),
    )


def _std_table(data, col_w, styles, *, header_bg=None, repeat=1):
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
    return Table(data, colWidths=col_w, style=ts, repeatRows=repeat)


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


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_hbar(df: pd.DataFrame, x_col: str, y_col: str,
                title: str, unit: str,
                color: str = "#4C72B0", top_n: int = 30) -> io.BytesIO:
    """Horizontal bar chart of top N rows sorted by x_col."""
    plot_df = df.nlargest(top_n, x_col).sort_values(x_col, ascending=True).copy()
    n = len(plot_df)
    if n == 0:
        return None
    fig_h = max(3.5, n * 0.35 + 1.0)
    fig, ax = plt.subplots(figsize=(10.0, fig_h), facecolor="white")

    vals   = plot_df[x_col].fillna(0).values.astype(float)
    labels = plot_df[y_col].astype(str).values
    avg    = float(vals.mean()) if len(vals) else 0
    bar_colors = [M_CRITICAL if v > avg * 2 else color for v in vals]

    ax.barh(range(n), vals, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(n))
    ax.set_yticklabels([textwrap.shorten(str(l), 30, placeholder="…") for l in labels], fontsize=9)
    if avg > 0:
        ax.axvline(avg, color="#555555", linewidth=1.0, linestyle="--", alpha=0.7,
                   label=f"평균: {avg:,.0f}")
    ax.set_xlabel(f"({unit})", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    x_max = float(max(vals)) * 1.2 if len(vals) else 1
    ax.set_xlim(0, x_max)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.set_facecolor("white")
    if avg > 0:
        ax.legend(prop=_FontProperties(fname=_FONT_REG, size=9), framealpha=0.9)
    xlim = ax.get_xlim()
    x_off = 0.01 * (xlim[1] - xlim[0])
    for i, v in enumerate(vals):
        ax.text(v + x_off, i, f"{v:,.0f}", va="center", ha="left", fontsize=8, color="#333333")

    fig.tight_layout(pad=0.8)
    return _png_from_fig(fig)


def _chart_hist(vals: np.ndarray, title: str, unit: str,
                color: str = "#4C72B0", tail_pct: float = 20.0) -> io.BytesIO:
    """Histogram with bottom/top tail shading (default 20%)."""
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None

    lo  = float(np.percentile(vals, tail_pct))
    hi  = float(np.percentile(vals, 100 - tail_pct))
    med = float(np.median(vals))

    n_bins = min(50, max(5, len(vals) // 3))
    counts, edges = np.histogram(vals, bins=n_bins)
    mids   = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    tail_mask  = (mids <= lo) | (mids >= hi)
    bar_colors = ["#DD8A00" if t else color for t in tail_mask]

    fig, ax = plt.subplots(figsize=(9.0, 3.2), facecolor="white")
    ax.bar(mids, counts, width=widths * 0.92, color=bar_colors,
           edgecolor="white", linewidth=0.5)

    xmin, xmax = float(vals.min()), float(vals.max())
    eps = 1e-12
    if lo > xmin + eps:
        ax.axvspan(xmin, lo, color="#DD8A00", alpha=0.10, linewidth=0)
        ax.axvline(lo, color="#555555", linewidth=1.2, linestyle="--", alpha=0.7)
    if hi < xmax - eps:
        ax.axvspan(hi, xmax, color="#DD8A00", alpha=0.10, linewidth=0)
        ax.axvline(hi, color="#555555", linewidth=1.2, linestyle="--", alpha=0.7)

    ax.axvline(med, color="#C44E52", linewidth=1.5, linestyle="--")

    # Annotation box
    ax.annotate(
        f"하위 {tail_pct:.0f}%  {lo:,.0f}\n상위 {tail_pct:.0f}%  {hi:,.0f}\n중앙값    {med:,.0f}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top",
        fontproperties=_FontProperties(fname=_FONT_REG, size=7.5),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAAAAA", lw=0.8, alpha=0.9),
    )

    ax.set_xlabel(f"({unit})", fontsize=9)
    ax.set_ylabel("건수", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlim(0, xmax * 1.05)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.legend(handles=[
        _Line2D([0], [0], color="#C44E52", linewidth=1.5, linestyle="--", label=f"중앙값: {med:,.0f}"),
    ], prop=_FontProperties(fname=_FONT_REG, size=8), framealpha=0.9)
    fig.tight_layout(pad=0.8)
    return _png_from_fig(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_billing_pdf(df: pd.DataFrame, context: dict = None, lang: str = "ko") -> bytes:
    """
    Generate a business-ready billing PDF.

    Parameters
    ----------
    df      : filtered billing DataFrame (one row per tenant)
    context : dict with optional keys 'buildings', 'date'
    lang    : 'ko' or 'en'
    """
    _ensure_fonts()
    styles    = _make_styles()
    ctx       = context or {}
    ko        = (lang == "ko")

    page_w, _ = A4
    margin    = 2 * cm
    content_w = page_w - 2 * margin

    footer_left     = "수도광열비 분석 보고서  ·  대외비" if ko else "Billing Report  ·  Confidential"
    footer_page_fmt = "{n} / {total} 페이지"            if ko else "Page {n} of {total}"
    T_footer        = {"footer_left": footer_left, "footer_page": footer_page_fmt}

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin,  bottomMargin=2 * cm,
    )
    doc.addPageTemplates([_make_page_template(doc, T_footer)])
    story = []

    # ── Derived data ─────────────────────────────────────────────────────────
    avail_utils = [
        (key, ko_lbl, en_lbl, excl, comm, total, clr)
        for (key, ko_lbl, en_lbl, excl, comm, total, clr) in _UTIL_DEFS
        if total in df.columns
    ]
    has_total    = "total" in df.columns
    sort_col     = "total" if has_total else (avail_utils[0][5] if avail_utils else None)
    buildings    = sorted(df["building"].dropna().unique()) if "building" in df.columns else []
    buildings_str= ", ".join(buildings) or "—"
    n_tenants    = len(df)
    grand_total  = float(df["total"].sum()) if has_total else None
    report_date  = str(ctx.get("date", _today_date.today()))

    def _util_name(ko_lbl, en_lbl):
        return ko_lbl if ko else en_lbl

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═════════════════════════════════════════════════════════════════════════
    title    = "수도광열비 부과 내역 보고서" if ko else "Utility Billing Report"
    subtitle = "임차인별 비용 분석 요약"    if ko else "Tenant Cost Analysis Summary"

    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(title,    styles["cover_title"]))
    story.append(Paragraph(subtitle, styles["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_divider_line(content_w))
    story.append(Spacer(1, 0.8 * cm))

    # Meta table
    meta_rows_raw = [
        ("보고서 일자" if ko else "Report Date",       report_date),
        ("건물"        if ko else "Buildings",          buildings_str),
        ("분석 임차인" if ko else "Tenants Analyzed",  str(n_tenants)),
        ("단위"        if ko else "Unit",               "만원 (VAT 별도)"),
    ]
    if grand_total is not None:
        meta_rows_raw.append(("총 합계 (만원)" if ko else "Grand Total (만원)", f"{grand_total:,.0f}"))

    meta_data = [[Paragraph(k, styles["table_cell"]), Paragraph(v, styles["table_cell"])]
                 for k, v in meta_rows_raw]
    meta_ts   = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
    ])
    story.append(Table(meta_data, colWidths=[4.5*cm, content_w - 4.5*cm], style=meta_ts))
    story.append(Spacer(1, 1.0 * cm))

    # Building totals table (cover)
    if buildings and avail_utils:
        bldg_section = "건물별 합계" if ko else "Totals by Building"
        util_hdrs    = [f"{_util_name(kl, el)} (만원)" for (_, kl, el, *_) in avail_utils]
        if has_total:
            util_hdrs.append("합계 (만원)" if ko else "Total (만원)")

        bldg_hdr = ([Paragraph("건물" if ko else "Building", styles["table_hdr"])] +
                    [Paragraph(h, styles["table_hdr"]) for h in util_hdrs])
        bldg_data = [bldg_hdr]

        for bldg in buildings:
            bdf = df[df["building"] == bldg]
            row = [Paragraph(str(bldg), styles["table_cell"])]
            for (_, kl, el, excl, comm, total_col, clr) in avail_utils:
                row.append(Paragraph(_f(bdf[total_col].sum() if total_col in bdf.columns else np.nan),
                                     styles["table_cell_c"]))
            if has_total:
                row.append(Paragraph(_f(bdf["total"].sum()), styles["table_cell_c"]))
            bldg_data.append(row)

        # Grand-total row
        tot_row = [Paragraph("합계" if ko else "Total", styles["table_cell"])]
        for (_, kl, el, excl, comm, total_col, clr) in avail_utils:
            tot_row.append(Paragraph(_f(df[total_col].sum() if total_col in df.columns else np.nan),
                                     styles["table_cell_c"]))
        if has_total:
            tot_row.append(Paragraph(_f(df["total"].sum()), styles["table_cell_c"]))
        bldg_data.append(tot_row)

        n_cols = 1 + len(avail_utils) + (1 if has_total else 0)
        bldg_cw = [2.5*cm] + [(content_w - 2.5*cm) / (n_cols - 1)] * (n_cols - 1)
        bldg_ts = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.HexColor("#F5F7FA"), colors.white]),
            ("BACKGROUND",     (0, -1), (-1, -1), C_LIGHT),
            ("FONTNAME",       (0, -1), (-1, -1), "NanumGothic-Bold"),
            ("ALIGN",          (1, 0),  (-1, -1), "RIGHT"),
        ])
        story.append(KeepTogether([
            Paragraph(bldg_section, styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(bldg_data, colWidths=bldg_cw, style=bldg_ts, repeatRows=1),
        ]))

    # Utility summary table (cover)
    if avail_utils:
        story.append(Spacer(1, 0.8 * cm))
        sum_hdr_labels = (
            ["항목" if ko else "Utility",
             "건수" if ko else "Count",
             "합계 (만원)" if ko else "Total (만원)",
             "평균 (만원)" if ko else "Average (만원)",
             "중앙값 (만원)" if ko else "Median (만원)",
             "최대 (만원)" if ko else "Max (만원)"]
        )
        sum_data = [[Paragraph(h, styles["table_hdr"]) for h in sum_hdr_labels]]
        for (_, kl, el, excl, comm, total_col, clr) in avail_utils:
            if total_col not in df.columns:
                continue
            vals = df[total_col].dropna().values.astype(float)
            vals = vals[vals > 0]
            n = len(vals)
            sum_data.append([
                Paragraph(_util_name(kl, el), styles["table_cell"]),
                Paragraph(str(n),            styles["table_cell_c"]),
                Paragraph(_f(vals.sum())     if n else "—", styles["table_cell_c"]),
                Paragraph(_f(vals.mean())    if n else "—", styles["table_cell_c"]),
                Paragraph(_f(np.median(vals)) if n else "—", styles["table_cell_c"]),
                Paragraph(_f(vals.max())     if n else "—", styles["table_cell_c"]),
            ])
        sum_cw = [3.0*cm] + [(content_w - 3.0*cm) / 5] * 5
        story.append(KeepTogether([
            Paragraph("항목별 요약" if ko else "Utility Summary", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            _std_table(sum_data, sum_cw, styles, header_bg=C_BLUE),
        ]))

    story.append(PageBreak())

    lbl_map = {"brand": "임차인" if ko else "Tenant",
               "building": "건물" if ko else "Bldg",
               "floor": "층" if ko else "Floor"}

    # ═════════════════════════════════════════════════════════════════════════
    # PER-UTILITY SECTIONS
    # ═════════════════════════════════════════════════════════════════════════
    for (key, ko_lbl, en_lbl, excl, comm, total_col, clr) in avail_utils:
        if total_col not in df.columns:
            continue

        util_name = _util_name(ko_lbl, en_lbl)
        story.append(_section_bar(
            f"  {util_name} — {'상세 분석' if ko else 'Detail Analysis'}",
            styles, content_w,
        ))
        story.append(Spacer(1, 0.3 * cm))

        vals = df[total_col].dropna().values.astype(float)
        vals = vals[vals > 0]

        # Distribution stats
        if len(vals):
            stat_hdr = ["지표" if ko else "Metric", "값 (만원)" if ko else "Value (만원)"]
            stat_rows_raw = [
                ("건수"           if ko else "Count",      str(len(vals))),
                ("합계"           if ko else "Total",      f"{vals.sum():,.0f}"),
                ("평균"           if ko else "Average",    f"{vals.mean():,.0f}"),
                ("중앙값"         if ko else "Median",     f"{np.median(vals):,.0f}"),
                ("표준편차"       if ko else "Std. Dev.",  f"{vals.std(ddof=1):,.0f}"),
                ("상위 20% 기준"  if ko else "P80",        f"{np.quantile(vals, 0.80):,.0f}"),
                ("최대"           if ko else "Max",        f"{vals.max():,.0f}"),
            ]
            stat_data = ([[Paragraph(h, styles["table_hdr"]) for h in stat_hdr]] +
                         [[Paragraph(r[0], styles["table_cell"]),
                           Paragraph(r[1], styles["table_cell_c"])]
                          for r in stat_rows_raw])
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
                Table(stat_data, colWidths=[5.0*cm, content_w - 5.0*cm], style=stat_ts),
            ]))
            story.append(Spacer(1, 0.4 * cm))

            # Histogram
            hist_buf = _chart_hist(
                vals,
                title=f"{util_name} — 임차인별 비용 분포" if ko else f"{en_lbl} — Cost Distribution",
                unit="만원", color=clr,
            )
            if hist_buf:
                story += _img_flow(
                    hist_buf, content_w / cm, styles,
                    caption=f"그림: {util_name} 임차인별 비용 분포. 빨간 점선=중앙값, 회색 점선=평균."
                            if ko else
                            f"Figure: {en_lbl} cost distribution. Red=median, grey=mean.",
                )
            story.append(Spacer(1, 0.5 * cm))

        # Top 10 table
        top10_df = df.nlargest(10, total_col).reset_index(drop=True)
        t10_base = [c for c in ["brand", "building", "floor"] if c in df.columns]
        t10_extra = []
        if excl and excl in df.columns:
            t10_extra.append((excl, "전용 (만원)" if ko else "Private (만원)"))
        if comm and comm in df.columns:
            t10_extra.append((comm, "공용 (만원)" if ko else "Common (만원)"))
        t10_extra.append((total_col, "합계 (만원)" if ko else "Total (만원)"))

        t10_hdr = (["#"] +
                   [lbl_map.get(c, c) for c in t10_base] +
                   [lbl for (_, lbl) in t10_extra])
        t10_data = [[Paragraph(h, styles["table_hdr"]) for h in t10_hdr]]
        for rank in range(len(top10_df)):
            row = top10_df.iloc[rank]
            r   = [Paragraph(str(rank + 1), styles["table_cell_c"])]
            for c in t10_base:
                val = textwrap.shorten(str(row.get(c, "")), 30, placeholder="…") if c == "brand" \
                      else str(row.get(c, ""))
                r.append(Paragraph(val,
                                   styles["table_cell"] if c == "brand" else styles["table_cell_c"]))
            for (col, _) in t10_extra:
                r.append(Paragraph(_f(row.get(col)), styles["table_cell_c"]))
            t10_data.append(r)

        t10_other = [c for c in t10_base if c != "brand"]
        t10_fixed = (0.9 + 1.5 * len(t10_other) + 2.2 * len(t10_extra)) * cm
        t10_tenant_w = content_w - t10_fixed
        t10_cw = [0.9*cm]
        for c in t10_base:
            t10_cw.append(t10_tenant_w if c == "brand" else 1.5*cm)
        t10_cw += [2.2*cm] * len(t10_extra)

        t10_ts = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
            ("ALIGN",          (len(t10_base) + 1, 0), (-1, -1), "RIGHT"),
            ("LEFTPADDING",    (0, 0), (0, -1), 2),
            ("RIGHTPADDING",   (0, 0), (0, -1), 2),
            ("ALIGN",          (0, 0), (0, -1), "CENTER"),
        ])
        story.append(KeepTogether([
            Paragraph(f"상위 10 임차인 ({util_name})" if ko else f"Top 10 Tenants ({en_lbl})",
                      styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(t10_data, colWidths=t10_cw, style=t10_ts),
        ]))

        # Building comparison bar
        if buildings and total_col in df.columns:
            bldg_grp = (df.groupby("building")[total_col].sum()
                        .reset_index()
                        .rename(columns={total_col: "value", "building": "building"}))
            bldg_bar_buf = _chart_hbar(
                bldg_grp, "value", "building",
                title=f"{util_name} — 건물별 합계" if ko else f"{en_lbl} — Total by Building",
                unit="만원", color=clr, top_n=len(bldg_grp),
            )
            if bldg_bar_buf:
                story.append(Spacer(1, 0.4 * cm))
                story += _img_flow(
                    bldg_bar_buf, content_w / cm, styles,
                    caption=f"그림: {util_name} 건물별 합계 비교."
                            if ko else f"Figure: {en_lbl} total by building.",
                )

        story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # TENANT RANKING (full list — placed last as it can be long)
    # ═════════════════════════════════════════════════════════════════════════
    if sort_col and sort_col in df.columns:
        story.append(_section_bar(
            "  임차인별 비용 순위" if ko else "  Tenant Cost Ranking",
            styles, content_w,
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Bar chart (top 30)
        bar_buf = _chart_hbar(
            df, sort_col, "brand",
            title="임차인별 총 비용" if ko else "Total Cost per Tenant",
            unit="만원", color="#4C72B0", top_n=min(30, len(df)),
        )
        if bar_buf:
            story += _img_flow(
                bar_buf, content_w / cm, styles,
                caption="그림: 임차인별 총 청구액 상위 30개 표시. 평균 2배 초과 시 붉은색 강조."
                        if ko else
                        "Figure: Total billing per tenant (top 30). Red = above 2× average.",
            )
        story.append(Spacer(1, 0.5 * cm))

        # Full ranking table
        base_cols       = [c for c in ["brand", "building", "floor"] if c in df.columns]
        util_total_cols = [(tc, f"{_util_name(kl, el)} (만원)")
                           for (_, kl, el, excl, comm, tc, clr) in avail_utils if tc in df.columns]
        if has_total:
            util_total_cols.append(("total", "합계 (만원)" if ko else "Total (만원)"))


        rank_hdr  = (["#"] +
                     [lbl_map.get(c, c) for c in base_cols] +
                     [lbl for (_, lbl) in util_total_cols])
        rank_data = [[Paragraph(h, styles["table_hdr"]) for h in rank_hdr]]

        sorted_df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        for rank in range(len(sorted_df)):
            row = sorted_df.iloc[rank]
            r   = [Paragraph(str(rank + 1), styles["table_cell_c"])]
            for c in base_cols:
                val = textwrap.shorten(str(row.get(c, "")), 30, placeholder="…") if c == "brand" \
                      else str(row.get(c, ""))
                r.append(Paragraph(val,
                                   styles["table_cell"] if c == "brand" else styles["table_cell_c"]))
            for (col, _) in util_total_cols:
                r.append(Paragraph(_f(row.get(col)), styles["table_cell_c"]))
            rank_data.append(r)

        other_base = [c for c in base_cols if c != "brand"]
        fixed_w    = (0.9 + 1.5 * len(other_base) + 2.2 * len(util_total_cols)) * cm
        tenant_w   = content_w - fixed_w
        rank_cw    = [0.9*cm]
        for c in base_cols:
            rank_cw.append(tenant_w if c == "brand" else 1.5*cm)
        rank_cw   += [2.2*cm] * len(util_total_cols)

        rank_ts = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
            ("ALIGN",          (len(base_cols) + 1, 0), (-1, -1), "RIGHT"),
            ("LEFTPADDING",    (0, 0), (0, -1), 2),
            ("RIGHTPADDING",   (0, 0), (0, -1), 2),
            ("ALIGN",          (0, 0), (0, -1), "CENTER"),
        ])
        story.append(KeepTogether([
            Paragraph(
                "전체 임차인 비용 순위 (총액 기준 내림차순)" if ko else
                "All tenants ranked by total cost (descending)",
                styles["note"],
            ),
            Spacer(1, 0.2 * cm),
            Table(rank_data, colWidths=rank_cw, style=rank_ts, repeatRows=1),
        ]))
        story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # BACK MATTER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("보고서 끝" if ko else "End of Report", styles["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    end_note = (f"작성일: {report_date}. "
                "본 보고서는 내부 관리 목적으로만 사용하시기 바랍니다. "
                "단위: 만원 (VAT 별도).") if ko else \
               (f"Generated on {report_date}. "
                "For internal management use only. Unit: 만원 (VAT excluded).")
    story.append(Paragraph(end_note, styles["note"]))

    NumberedCanvas = _make_numbered_canvas(T_footer)
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()

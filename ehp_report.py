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
    C_BLUE, C_CRITICAL, C_DIVIDER, C_LIGHT, C_NAVY, C_STABLE, C_WHITE,
    M_BAR, M_CRITICAL,
    _ensure_fonts, _make_numbered_canvas, _make_page_template,
    _make_styles, _png, _section_bar, get_report_font_paths,
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
    ax.set_ylim(0, max(values) * 1.15)
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

    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor="white")
    for i, yr in enumerate(sorted(pivot.columns)):
        series = pivot.loc[[m for m in available if yr in pivot.columns], yr].dropna()
        if series.empty:
            continue
        x_ord = [month_order.get(m, 0) for m in series.index]
        sorted_pairs = sorted(zip(x_ord, series.values), key=lambda t: t[0])
        x_num = [t[0] for t in sorted_pairs]
        y_val = [t[1] if t[1] >= 0 else np.nan for t in sorted_pairs]
        ax.plot(x_num, y_val,
                color=_PALETTE[i % len(_PALETTE)], linewidth=2,
                marker="o", markersize=4, label=str(yr))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}월" for m in range(1, 13)])

    ax.set_ylabel("kWh", fontsize=9)
    ax.set_title("월별 EHP 전기 사용량 추세", fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=9)
    all_y = [v for yr in pivot.columns for v in pivot[yr].dropna().values if not np.isnan(float(v)) and float(v) >= 0]
    ax.set_facecolor("white")
    ax.set_ylim(0, max(all_y) * 1.15 if all_y else 1)
    ax.grid(color="#DDDDDD", linewidth=0.5, linestyle="--")
    n_years = len(pivot.columns)
    ax.legend(fontsize=10, framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(n_years, 5), borderaxespad=0)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_monthly_heatmap(pivot: pd.DataFrame) -> io.BytesIO:
    """Heatmap of monthly usage: rows = months, columns = years."""
    if pivot.empty:
        return None
    month_order = [f"{m}월" for m in range(1, 13)]
    avail = [m for m in month_order if m in pivot.index]
    years = sorted(pivot.columns)
    data  = np.array([
        [float(pivot.loc[m, yr]) if yr in pivot.columns and pd.notna(pivot.loc[m, yr]) else np.nan
         for yr in years]
        for m in avail
    ])

    fig_w = max(6.0, len(years) * 1.1 + 2.0)
    fig_h = max(4.0, len(avail) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    vmax = float(np.nanmax(data)) if not np.all(np.isnan(data)) else 1.0
    im   = ax.imshow(data, cmap="YlOrRd", aspect="auto", interpolation="nearest",
                     vmin=0, vmax=vmax)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels(avail, fontsize=9)
    ax.tick_params(length=0)

    for i in range(len(avail)):
        for j in range(len(years)):
            v = data[i, j]
            if not np.isnan(v):
                text_color = "white" if v > vmax * 0.55 else "#333333"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="kWh", shrink=0.75, pad=0.02)
    ax.set_title("월별 사용량 히트맵 (kWh)", fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_anomaly_heatmap(pivot: pd.DataFrame) -> io.BytesIO:
    """Z-score heatmap for anomaly detection (red = high spike)."""
    if pivot.empty or len(pivot.columns) < 2:
        return None
    month_order = [f"{m}월" for m in range(1, 13)]
    avail = [m for m in month_order if m in pivot.index]
    years = sorted(pivot.columns)

    z_data = []
    for mo in avail:
        row = pd.to_numeric(pivot.loc[mo], errors="coerce")
        mu, sd = row.mean(), row.std()
        z_row = []
        for yr in years:
            v = row.get(yr, np.nan)
            z = (float(v) - mu) / sd if pd.notna(v) and sd > 0 else 0.0
            z_row.append(round(z, 2) if pd.notna(v) else np.nan)
        z_data.append(z_row)

    z_arr = np.array(z_data, dtype=float)
    fig_w = max(6.0, len(years) * 1.1 + 2.0)
    fig_h = max(4.0, len(avail) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    im = ax.imshow(z_arr, cmap="RdYlGn_r", aspect="auto", interpolation="nearest",
                   vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels(avail, fontsize=9)
    ax.tick_params(length=0)

    for i in range(len(avail)):
        for j in range(len(years)):
            z = z_arr[i, j]
            if not np.isnan(z):
                ax.text(j, i, f"{z:+.1f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(z) > 1.5 else "#333333")

    plt.colorbar(im, ax=ax, label="이상도", shrink=0.75, pad=0.02)
    ax.set_title("월별 사용량 이상 탐지 (빨강=급증, 초록=급감)",
                 fontsize=10, fontweight="bold", color="#1B2A3B", pad=8)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_top_meters(m_df: pd.DataFrame, top_n: int = 15) -> io.BytesIO:
    """Horizontal bar chart of top meters by grand total."""
    if m_df.empty or "grand_total" not in m_df.columns:
        return None
    top = m_df.nlargest(top_n, "grand_total").iloc[::-1]
    labels = [str(v) for v in top["meter"]]
    values = top["grand_total"].values.astype(float)

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(labels) * 0.38 + 1.0)), facecolor="white")
    bars = ax.barh(labels, values, color=M_BAR, edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f}", va="center", fontsize=8, color="#333333")
    ax.set_xlabel("kWh", fontsize=9)
    ax.set_title(f"계량기별 총 사용량 Top {min(top_n, len(labels))} (kWh)",
                 fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_metric_bars(tmp: pd.DataFrame, col0: str, mc: str, unit: str) -> io.BytesIO:
    """Horizontal bar chart of a metric grouped by building."""
    if col0 not in tmp.columns or mc not in tmp.columns:
        return None
    grp = tmp.groupby(col0)[mc].sum(min_count=1).dropna().sort_values(ascending=True)
    if grp.empty:
        return None
    labels = [str(v) for v in grp.index]
    values = grp.values.astype(float)

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, len(labels) * 0.55 + 1.2)), facecolor="white")
    bars = ax.barh(labels, values, color=M_BAR, edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,.1f}", va="center", fontsize=9, color="#333333")
    ax.set_xlabel(unit, fontsize=9)
    ax.set_title(f"건물별 {mc} 합계", fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_hvac_by_building(bill_df: pd.DataFrame) -> io.BytesIO:
    """Grouped bar chart of hvac_excl / hvac_comm by building."""
    grp = bill_df.groupby("building")[["hvac_excl", "hvac_comm"]].sum().reset_index()
    grp = grp.sort_values("hvac_excl", ascending=False)
    if grp.empty:
        return None

    x     = grp["building"].tolist()
    excl  = grp["hvac_excl"].tolist()
    comm  = grp["hvac_comm"].tolist()
    total = [e + c for e, c in zip(excl, comm)]

    fig, ax = plt.subplots(figsize=(max(5.0, len(x) * 1.2 + 1.5), 4.0), facecolor="white")
    xi = range(len(x))
    w  = 0.35
    b1 = ax.bar([i - w/2 for i in xi], excl, width=w, color=_PALETTE[0], label="전용", edgecolor="white")
    b2 = ax.bar([i + w/2 for i in xi], comm, width=w, color=_PALETTE[1], label="공용", edgecolor="white")
    for bar, v in list(zip(b1, excl)) + list(zip(b2, comm)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + max(total)*0.01,
                    f"{v:,.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xi))
    ax.set_xticklabels(x, fontsize=10)
    ax.set_ylabel("만원", fontsize=9)
    ax.set_title("건물별 냉난방 비용 (만원)", fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_ylim(0, max(total) * 1.2 if total else 1)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_hvac_top_brands(bill_df: pd.DataFrame, top_n: int = 15) -> io.BytesIO:
    """Horizontal bar chart of top brands by hvac_excl."""
    grp = (bill_df.groupby("brand")["hvac_excl"]
                  .sum().reset_index()
                  .sort_values("hvac_excl", ascending=False)
                  .head(top_n)
                  .iloc[::-1])
    if grp.empty:
        return None

    labels = grp["brand"].tolist()
    values = grp["hvac_excl"].tolist()

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(labels) * 0.38 + 1.0)), facecolor="white")
    bars = ax.barh(labels, values, color=_PALETTE[0], edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(v + max(values) * 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:,.2f}", va="center", fontsize=8, color="#333333")
    ax.set_xlabel("만원", fontsize=9)
    ax.set_title(f"상호별 냉난방 전용 Top {len(labels)} (만원)",
                 fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlim(0, max(values) * 1.2 if values else 1)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.tick_params(labelsize=8)
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
                          lang: str = "ko",
                          dedicated_df: pd.DataFrame = None,
                          dedicated_col0: str = None,
                          bill_df: pd.DataFrame = None) -> bytes:
    """
    Business-ready PDF for the OAC EHP electricity analysis.

    Parameters
    ----------
    pivot          : monthly × yearly pivot (index = "1월"…"12월", columns = years)
    usage          : per-meter usage DataFrame (optional)
    context        : dict with optional 'date', 'sheet_name'
    lang           : 'ko' or 'en'
    dedicated_df   : cleaned 전용 EHP DataFrame (optional, appended as extra section)
    dedicated_col0 : building column name for dedicated_df
    bill_df        : 관리비 고지서 DataFrame (optional, for 냉난방 section)
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
    # PAGE 3 — MONTHLY USAGE HEATMAP
    # ═════════════════════════════════════════════════════════════════════════
    story.append(_section_bar(
        "  월별 사용량 현황" if ko else "  Monthly Usage Summary",
        styles, content_w,
    ))
    story.append(Spacer(1, 0.3 * cm))

    heatmap_buf = _chart_monthly_heatmap(pivot)
    if heatmap_buf:
        story += _img_flow(
            heatmap_buf, content_w / cm, styles,
            caption="그림: 월별×연도별 사용량 히트맵. 색이 진할수록 사용량 높음."
                    if ko else
                    "Figure: Monthly usage heatmap by year. Darker = higher usage.",
        )
        story.append(Spacer(1, 0.5 * cm))

    # Yearly totals summary table (compact — one row per year, no month breakdown)
    if year_totals:
        yr_list = sorted(year_totals)
        yr_hdr  = [Paragraph(h, styles["table_hdr"]) for h in (
            ["연도", "연간 합계 (kWh)", "전년 대비 (%)"] if ko else
            ["Year", "Annual Total (kWh)", "vs. Prev. Year (%)"]
        )]
        yr_data = [yr_hdr]
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
            val = year_totals[yr]
            if i > 0:
                prev    = year_totals[yr_list[i - 1]]
                pct     = (val - prev) / prev * 100 if prev > 0 else np.nan
                pct_str = f"{pct:+.1f}" if not np.isnan(pct) else "—"
                if not np.isnan(pct):
                    col_idx = i + 1
                    if pct > 10:
                        yr_ts.add("TEXTCOLOR", (2, col_idx), (2, col_idx), C_CRITICAL)
                        yr_ts.add("FONTNAME",  (2, col_idx), (2, col_idx), "NanumGothic-Bold")
                    elif pct < -10:
                        yr_ts.add("TEXTCOLOR", (2, col_idx), (2, col_idx), C_STABLE)
                        yr_ts.add("FONTNAME",  (2, col_idx), (2, col_idx), "NanumGothic-Bold")
            else:
                pct_str = "—"
            yr_data.append([
                Paragraph(str(yr), styles["table_cell_c"]),
                Paragraph(f"{val:,.0f}", styles["table_cell_c"]),
                Paragraph(pct_str,       styles["table_cell_c"]),
            ])
        yr_cw = [3.0*cm, (content_w - 3.0*cm) / 2, (content_w - 3.0*cm) / 2]
        story.append(KeepTogether([
            Paragraph("연간 합계" if ko else "Annual Totals", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(yr_data, colWidths=yr_cw, style=yr_ts),
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

    anom_chart_buf = _chart_anomaly_heatmap(pivot)
    if anom_chart_buf:
        story += _img_flow(
            anom_chart_buf, content_w / cm, styles,
            caption="그림: Z-score 히트맵. 빨강=평균 대비 급증, 초록=급감. |Z|>1.5 강조."
                    if ko else
                    "Figure: Z-score heatmap. Red = above average, Green = below. |Z|>1.5 highlighted.",
        )
        story.append(Spacer(1, 0.4 * cm))

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

            story.append(PageBreak())
            story.append(_section_bar(
                "  계량기별 사용량" if ko else "  Usage by Meter",
                styles, content_w,
            ))
            story.append(Spacer(1, 0.3 * cm))

            # Bar chart — top meters
            meter_chart_buf = _chart_top_meters(m_df, top_n=20)
            if meter_chart_buf:
                story += _img_flow(
                    meter_chart_buf, content_w / cm, styles,
                    caption="그림: 총 사용량 상위 계량기 (kWh)."
                            if ko else
                            "Figure: Top meters by total usage (kWh).",
                )
                story.append(Spacer(1, 0.4 * cm))

            # Compact summary table — top 20 only
            top20 = m_df.head(20)
            hdr_m = [Paragraph("순위" if ko else "#", styles["table_hdr"]),
                     Paragraph("계량기" if ko else "Meter", styles["table_hdr"]),
                     Paragraph("합계 (kWh)" if ko else "Total (kWh)", styles["table_hdr"])]
            m_data = [hdr_m]
            for rank, row in enumerate(top20.itertuples(), 1):
                m_data.append([
                    Paragraph(str(rank), styles["table_cell_c"]),
                    Paragraph(str(row.meter), styles["table_cell"]),
                    Paragraph(_f(row.grand_total), styles["table_cell_c"]),
                ])
            m_ts = TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
                ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
                ("FONTSIZE",       (0, 0), (-1, -1), 8),
                ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
                ("TOPPADDING",     (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
                ("LEFTPADDING",    (0, 0), (-1, -1), 5),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
                ("ALIGN",          (0, 0), (0, -1),  "CENTER"),
                ("ALIGN",          (2, 0), (2, -1),  "RIGHT"),
                ("FONTNAME",       (2, 1), (2, -1),  "NanumGothic-Bold"),
            ])
            m_cw = [1.2*cm, content_w - 1.2*cm - 3.5*cm, 3.5*cm]
            note_txt = (f"상위 {len(top20)}개 계량기 표시 (전체 {len(m_df)}개, 합계 기준 내림차순)."
                        if ko else
                        f"Showing top {len(top20)} of {len(m_df)} meters by total usage.")
            story.append(KeepTogether([
                Paragraph(note_txt, styles["note"]),
                Spacer(1, 0.2 * cm),
                Table(m_data, colWidths=m_cw, style=m_ts),
            ]))

    # ═════════════════════════════════════════════════════════════════════════
    # 전용 EHP SECTION (optional)
    # ═════════════════════════════════════════════════════════════════════════
    if dedicated_df is not None and not dedicated_df.empty and dedicated_col0:
        story += _dedicated_ehp_story(dedicated_df, dedicated_col0, content_w, styles, ko)

    # ═════════════════════════════════════════════════════════════════════════
    # 냉난방 고지서 SECTION (optional)
    # ═════════════════════════════════════════════════════════════════════════
    if (bill_df is not None and not bill_df.empty
            and {"hvac_excl", "hvac_comm", "building", "brand"}.issubset(bill_df.columns)):

        bill_df = bill_df.copy()
        bill_df["hvac_total"] = bill_df["hvac_excl"] + bill_df["hvac_comm"]

        if not story or not isinstance(story[-1], PageBreak):
            story.append(PageBreak())
        story.append(_section_bar(
            "  냉난방 비용 (관리비 고지서)" if ko else "  HVAC Charges (Management Bill)",
            styles, content_w,
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Summary stats row
        total_excl  = bill_df["hvac_excl"].sum()
        total_comm  = bill_df["hvac_comm"].sum()
        total_hvac  = bill_df["hvac_total"].sum()
        n_units     = len(bill_df)
        stat_hdr = [Paragraph(h, styles["table_hdr"]) for h in (
            ["항목", "금액 (만원)"] if ko else ["Item", "Amount (만원)"]
        )]
        stat_data = [stat_hdr] + [
            [Paragraph(k, styles["table_cell"]), Paragraph(v, styles["table_cell_c"])]
            for k, v in [
                ("냉난방 전용 합계" if ko else "HVAC Excl. Total",   f"{total_excl:,.2f}"),
                ("냉난방 공용 합계" if ko else "HVAC Comm. Total",   f"{total_comm:,.2f}"),
                ("냉난방 합계"      if ko else "HVAC Grand Total",   f"{total_hvac:,.2f}"),
                ("분석 세대 수"     if ko else "Units Analyzed",     str(n_units)),
            ]
        ]
        stat_ts = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
            ("ALIGN",          (1, 1), (1, -1),  "RIGHT"),
            ("FONTNAME",       (0, -1), (-1, -1), "NanumGothic-Bold"),
        ])
        story.append(KeepTogether([
            Paragraph("요약" if ko else "Summary", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(stat_data, colWidths=[content_w * 0.55, content_w * 0.45], style=stat_ts),
        ]))
        story.append(Spacer(1, 0.5 * cm))

        # Building-level chart
        bldg_chart_buf = _chart_hvac_by_building(bill_df)
        if bldg_chart_buf:
            story += _img_flow(
                bldg_chart_buf, content_w / cm, styles,
                caption="그림: 건물별 냉난방 전용/공용 비용 (만원)."
                        if ko else
                        "Figure: HVAC exclusive/common charges by building (만원).",
            )
            story.append(Spacer(1, 0.4 * cm))

        # Top-brand chart
        brand_chart_buf = _chart_hvac_top_brands(bill_df, top_n=15)
        if brand_chart_buf:
            story += _img_flow(
                brand_chart_buf, content_w / cm, styles,
                caption="그림: 냉난방 전용 비용 상위 상호 (만원)."
                        if ko else
                        "Figure: Top brands by HVAC exclusive charge (만원).",
            )
            story.append(Spacer(1, 0.4 * cm))

        # Building summary table
        bldg_tbl = (bill_df.groupby("building")
                            .agg(전용=("hvac_excl",  "sum"),
                                 공용=("hvac_comm",  "sum"),
                                 합계=("hvac_total", "sum"),
                                 건수=("brand",      "count"))
                            .reset_index()
                            .sort_values("합계", ascending=False))
        tbl_hdr = [Paragraph(h, styles["table_hdr"]) for h in (
            ["건물", "전용 (만원)", "공용 (만원)", "합계 (만원)", "세대 수"] if ko else
            ["Building", "Excl. (만원)", "Comm. (만원)", "Total (만원)", "Units"]
        )]
        tbl_data = [tbl_hdr]
        for _, row in bldg_tbl.iterrows():
            tbl_data.append([
                Paragraph(str(row["building"]),    styles["table_cell_c"]),
                Paragraph(f"{row['전용']:,.2f}",   styles["table_cell_c"]),
                Paragraph(f"{row['공용']:,.2f}",   styles["table_cell_c"]),
                Paragraph(f"{row['합계']:,.2f}",   styles["table_cell_c"]),
                Paragraph(str(int(row["건수"])),   styles["table_cell_c"]),
            ])
        col_w = content_w / 5
        tbl_ts = TableStyle([
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
        story.append(KeepTogether([
            Paragraph("건물별 냉난방 비용 요약" if ko else "HVAC Cost Summary by Building",
                      styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            Table(tbl_data, colWidths=[col_w] * 5, style=tbl_ts),
        ]))

    # ═════════════════════════════════════════════════════════════════════════
    # BACK MATTER
    # ═════════════════════════════════════════════════════════════════════════
    # Ensure we start back matter on a fresh page (avoid double PageBreak)
    if not story or not isinstance(story[-1], PageBreak):
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


# ── Dedicated EHP story builder (shared between OAC and standalone reports) ────

def _dedicated_ehp_story(sliced_df: pd.DataFrame, col0: str,
                          content_w: float, styles: dict, ko: bool) -> list:
    """Return ReportLab flowables for the 전용 EHP section."""
    metric_cols = [c for c in ["전기 사용량", "매장별 가동시간", "효율 (kWh/hr)"]
                   if c in sliced_df.columns]
    metric_units = {
        "전기 사용량":      "kWh",
        "매장별 가동시간":  "hr",
        "효율 (kWh/hr)":   "kWh/hr",
    }
    all_dong = sorted(sliced_df[col0].dropna().unique(), key=str) if col0 in sliced_df.columns else []

    if not metric_cols:
        return []

    story = []

    # ── Section header ────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_section_bar(
        "  전용 EHP 분석" if ko else "  Dedicated EHP Analysis",
        styles, content_w,
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Building-level summary
    if all_dong and col0 in sliced_df.columns:
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
        story.append(Spacer(1, 0.5 * cm))

    # Per-metric detail sections
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

        # Bar chart — building-level comparison
        metric_chart_buf = _chart_metric_bars(tmp, col0, mc, unit)
        if metric_chart_buf:
            story += _img_flow(
                metric_chart_buf, content_w / cm, styles,
                caption=f"그림: 건물별 {mc} 합계 ({unit})."
                        if ko else
                        f"Figure: {mc} total by building ({unit}).",
            )
            story.append(Spacer(1, 0.4 * cm))

        # Top-20 equipment table
        top20 = tmp.sort_values(mc, ascending=False).head(20).reset_index(drop=True)
        detail_cols = [c for c in [col0, "판넬명", "장비번호", "상호", mc] if c in top20.columns]
        det_hdr_map = {col0: "건물", "판넬명": "판넬명", "장비번호": "장비번호",
                       "상호": "상호", mc: f"{mc} ({unit})"}
        det_hdr = [Paragraph(det_hdr_map.get(c, c), styles["table_hdr"]) for c in detail_cols]
        det_data = [det_hdr]
        for _, row in top20.iterrows():
            r = []
            for c in detail_cols:
                val = row.get(c, "")
                if c == mc:
                    r.append(Paragraph(_f(val, decimals=1) if pd.notna(val) else "—",
                                       styles["table_cell_c"]))
                else:
                    r.append(Paragraph(textwrap.shorten(str(val), 28, placeholder="…"),
                                       styles["table_cell"]))
            det_data.append(r)

        fixed_det_w = 3.5 * cm
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
        note_txt = (f"상위 {len(top20)}개 장비 표시 (전체 {len(tmp)}개, {mc} 기준 내림차순)."
                    if ko else
                    f"Top {len(top20)} of {len(tmp)} equipment by {mc}.")
        story.append(Paragraph(note_txt, styles["note"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Table(det_data, colWidths=det_cw, style=det_ts, repeatRows=1))
        story.append(PageBreak())

    return story


# ── Dedicated EHP PDF ─────────────────────────────────────────────────────────

def generate_ehp_dedicated_pdf(sliced_df: pd.DataFrame,
                                col0: str,
                                context: dict = None,
                                lang: str = "ko") -> bytes:
    """Business-ready PDF for 전용 EHP data."""
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

    report_date = str(ctx.get("date", _today_date.today()))
    all_dong    = sorted(sliced_df[col0].dropna().unique(), key=str) if col0 in sliced_df.columns else []
    metric_cols = [c for c in ["전기 사용량", "매장별 가동시간", "효율 (kWh/hr)"] if c in sliced_df.columns]

    story = []
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("전용 EHP 전기 사용량 분석 보고서" if ko else "Dedicated EHP Analysis Report",
                            styles["cover_title"]))
    story.append(Paragraph("장비별 전기 사용량 및 효율 분석" if ko else
                            "Equipment-level Electricity & Efficiency", styles["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_divider_line(content_w))
    story.append(Spacer(1, 0.8 * cm))

    meta_raw = [
        ("보고서 일자" if ko else "Report Date",     report_date),
        ("건물"        if ko else "Buildings",        ", ".join(all_dong) or "—"),
        ("장비 수"     if ko else "Equipment Count", str(len(sliced_df))),
        ("분석 항목"   if ko else "Metrics",         ", ".join(metric_cols) or "—"),
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
    story.append(PageBreak())

    story += _dedicated_ehp_story(sliced_df, col0, content_w, styles, ko)

    # Remove trailing PageBreak if present before back matter
    if story and isinstance(story[-1], PageBreak):
        story.pop()

    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("보고서 끝" if ko else "End of Report", styles["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        f"작성일: {report_date}. 본 보고서는 내부 관리 목적으로만 사용하시기 바랍니다." if ko else
        f"Generated on {report_date}. For internal management use only.",
        styles["note"],
    ))

    NumberedCanvas = _make_numbered_canvas(T_footer)
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()

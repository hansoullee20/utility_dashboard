"""
hvac_report.py  —  Business-ready PDF for 관리비 고지서 EHP 열(냉난방) analysis
"""
import io
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
    C_BLUE, C_DIVIDER, C_LIGHT, C_NAVY, C_WHITE,
    M_BAR, M_CRITICAL,
    _ensure_fonts, _make_numbered_canvas, _make_page_template,
    _make_styles, _png, _section_bar, _FONT_REG,
)

_C_BASE  = "#9B59B6"
_C_USAGE = "#27AE60"
_C_COMM  = "#E67E22"
_C_RED   = colors.HexColor("#C0392B")
_C_WARN  = colors.HexColor("#F39C12")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(val, decimals=0):
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


def _divider_line(content_w):
    return Table(
        [[""]],
        colWidths=[content_w],
        style=TableStyle([
            ("LINEABOVE",     (0, 0), (-1, -1), 1.5, C_NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]),
    )


def _kv_table(rows, col_w, styles, header_bg=None):
    """Two-column key-value table, no header row."""
    header_bg = header_bg or C_NAVY
    ts = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ])
    data = [[Paragraph(k, styles["table_cell"]), Paragraph(v, styles["table_cell"])]
            for k, v in rows]
    return Table(data, colWidths=col_w, style=ts)


def _stat_block(rows, styles, content_w):
    """Compact metrics row: [(label, value, note), ...]"""
    n = len(rows)
    cw = content_w / n
    hdrs = [Paragraph(lbl,  styles["table_hdr"])  for lbl, _, _ in rows]
    vals = [Paragraph(val,  styles["table_cell_c"]) for _, val, _ in rows]
    notes= [Paragraph(note, styles["caption"])      for _, _, note in rows]
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT, colors.white]),
    ])
    return Table([hdrs, vals, notes], colWidths=[cw] * n, style=ts)


def _insight_box(text, styles, content_w, bg=None):
    """Shaded insight/interpretation paragraph."""
    bg = bg or colors.HexColor("#EEF3FA")
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("ROUNDEDCORNERS",[3]),
    ])
    return Table([[Paragraph(text, styles["note"])]], colWidths=[content_w], style=ts)


def _mini_table(data, col_w, styles, header_bg=None):
    """Small header + data table (top-N outliers, etc.)."""
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
        ("ALIGN",          (1, 0), (-1, -1), "RIGHT"),
    ])
    return Table(data, colWidths=col_w, style=ts, repeatRows=1)


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_stacked_hbar(agg, base_col, usage_fee_col, comm_fee_col, fee_col,
                         top_n=25):
    cols = [(c, l, clr) for c, l, clr in [
        (base_col,     "기본요금", _C_BASE),
        (usage_fee_col,"사용요금", _C_USAGE),
        (comm_fee_col, "공용요금", _C_COMM),
    ] if c and c in agg.columns]
    if not cols:
        return None
    sort_col = fee_col if (fee_col and fee_col in agg.columns) else cols[-1][0]
    plot_df  = agg.nlargest(top_n, sort_col).sort_values(sort_col, ascending=True)
    n = len(plot_df)
    if n == 0:
        return None

    fig_h = max(4.0, n * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), facecolor="white")
    lefts = np.zeros(n)
    for c, lbl, clr in cols:
        vals = plot_df[c].fillna(0).values.astype(float)
        ax.barh(range(n), vals, left=lefts, color=clr, edgecolor="white",
                linewidth=0.4, height=0.72, label=lbl)
        lefts += vals

    totals = plot_df[sort_col].fillna(0).values.astype(float)
    x_max  = float(totals.max()) * 1.22 if totals.max() > 0 else 1
    for i, v in enumerate(totals):
        ax.text(v + x_max * 0.008, i, f"{v:,.0f}",
                va="center", ha="left", fontsize=8, color="#333333")

    ax.set_yticks(range(n))
    ax.set_yticklabels([textwrap.shorten(str(l), 28, placeholder="…")
                        for l in plot_df.index.astype(str)], fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("(원)", fontsize=9)
    ax.set_title(f"업체별 FCU 요금 상위 {n}개 (소계 내림차순)", fontsize=11,
                 fontweight="bold", color="#1B2A3B", pad=8)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.set_facecolor("white")
    ax.legend(prop=_FontProperties(fname=_FONT_REG, size=9),
              loc="lower right", framealpha=0.9)
    fig.tight_layout(pad=0.8)
    return _png_from_fig(fig)


def _chart_donut(base_pct, usage_pct, comm_pct):
    labels = ["기본요금", "사용요금"]
    values = [base_pct, usage_pct]
    clrs   = [_C_BASE, _C_USAGE]
    if comm_pct > 0:
        labels.append("공용요금"); values.append(comm_pct); clrs.append(_C_COMM)
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=clrs,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white"),
        textprops=dict(fontsize=10),
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9); t.set_color("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.5)
    return _png_from_fig(fig)


def _chart_hbar_single(series, title, unit, color=M_BAR, top_n=25, iqr_fence=None):
    s = series.dropna()
    s = s[s > 0].nlargest(top_n).sort_values(ascending=True)
    n = len(s)
    if n == 0:
        return None
    avg = float(s.mean())
    bar_colors = [M_CRITICAL if (iqr_fence and v > iqr_fence) else color for v in s.values]
    fig_h = max(3.5, n * 0.35 + 1.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), facecolor="white")
    ax.barh(range(n), s.values, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(n))
    ax.set_yticklabels([textwrap.shorten(str(l), 28, placeholder="…")
                        for l in s.index.astype(str)], fontsize=8)
    if avg > 0:
        ax.axvline(avg, color="#555555", linewidth=1.2, linestyle="--", alpha=0.8,
                   label=f"평균 {avg:,.1f}")
    if iqr_fence:
        ax.axvline(iqr_fence, color=M_CRITICAL, linewidth=1.2, linestyle=":",
                   alpha=0.85, label=f"IQR 상한 {iqr_fence:,.1f}")
    x_max = float(s.max()) * 1.2 if len(s) else 1
    ax.set_xlim(0, x_max)
    x_off = 0.01 * x_max
    for i, v in enumerate(s.values):
        ax.text(v + x_off, i, f"{v:,.1f}", va="center", ha="left", fontsize=8, color="#333333")
    ax.set_xlabel(f"({unit})", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.set_facecolor("white")
    if avg > 0 or iqr_fence:
        ax.legend(prop=_FontProperties(fname=_FONT_REG, size=8), framealpha=0.9)
    fig.tight_layout(pad=0.8)
    return _png_from_fig(fig)


def _chart_scatter(x_s, y_s, xlabel, ylabel, title):
    df = pd.DataFrame({"x": x_s, "y": y_s}).dropna()
    if len(df) < 3:
        return None
    x, y = df["x"].values.astype(float), df["y"].values.astype(float)
    m, b = np.polyfit(x, y, 1)
    r    = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(9.0, 4.5), facecolor="white")
    ax.scatter(x, y, color=M_BAR, s=60, alpha=0.85, edgecolors="white", linewidths=0.6)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color=M_CRITICAL, linewidth=1.5,
            linestyle="--", label=f"추세선  r = {r:+.2f}")
    for xi, yi, lbl in zip(x, y, df.index.astype(str)):
        ax.annotate(textwrap.shorten(lbl, 16, placeholder="…"), (xi, yi),
                    fontsize=7, color="#555555",
                    textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1B2A3B", pad=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.set_facecolor("white")
    ax.legend(prop=_FontProperties(fname=_FONT_REG, size=9), framealpha=0.9)
    fig.tight_layout(pad=0.8)
    return _png_from_fig(fig)


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _iqr_stats(s: pd.Series):
    """Return (q1, q3, iqr, upper, lower, med, mean, cv, n_upper, n_lower)."""
    s = s.dropna(); s = s[s > 0]
    if len(s) < 4:
        return None
    q1, q3   = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr      = q3 - q1
    upper    = q3 + 1.5 * iqr
    lower    = max(q1 - 1.5 * iqr, 0.0)
    med      = float(s.median())
    mean     = float(s.mean())
    cv       = float(s.std() / mean) if mean else 0
    n_upper  = int((s > upper).sum())
    n_lower  = int((lower > 0) and (s < lower).sum())
    return dict(q1=q1, q3=q3, iqr=iqr, upper=upper, lower=lower,
                med=med, mean=mean, cv=cv, n_upper=n_upper, n_lower=n_lower,
                s=s)


def _cv_label(cv):
    if cv < 0.2:  return "낮음 (균일한 분포)"
    if cv < 0.5:  return "보통"
    if cv < 1.0:  return "높음 (업체 간 편차 큼)"
    return "매우 높음 (집중 구조)"


def _skew_label(mean, med):
    ratio = mean / med if med else 1
    if ratio > 1.3:  return "우편향 (소수 고액 업체가 평균 견인)"
    if ratio < 0.77: return "좌편향"
    return "대칭 (고른 분포)"


def _conc_label(pct):
    if pct >= 70: return "매우 높음 — 소수 업체 집중"
    if pct >= 50: return "높음"
    if pct >= 30: return "보통"
    return "낮음 — 분산된 구조"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_hvac_pdf(
    brand_agg: pd.DataFrame,
    *,
    usage_col=None,
    base_col=None,
    usage_fee_col=None,
    comm_fee_col=None,
    fee_col=None,
    area_col=None,
    context=None,
) -> bytes:
    """
    Generate a business-ready HVAC billing PDF.
    brand_agg : brand-aggregated DataFrame (index = 브랜드)
    """
    _ensure_fonts()
    styles   = _make_styles()
    ctx      = context or {}
    page_w, _ = A4
    margin    = 2 * cm
    content_w = page_w - 2 * margin

    footer_left     = "EHP 열(냉난방) 요금 분석 보고서  ·  대외비"
    footer_page_fmt = "{n} / {total} 페이지"
    T_footer = {"footer_left": footer_left, "footer_page": footer_page_fmt}

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=2 * cm,
    )
    doc.addPageTemplates([_make_page_template(doc, T_footer)])
    story = []

    report_date = str(ctx.get("date", _today_date.today()))
    n_brands    = len(brand_agg)

    # ── Pre-compute key figures ────────────────────────────────────────────────
    fee_total  = float(brand_agg[fee_col].sum())      if fee_col      and fee_col      in brand_agg.columns else None
    base_total = float(brand_agg[base_col].sum())     if base_col     and base_col     in brand_agg.columns else None
    us_total   = float(brand_agg[usage_fee_col].sum()) if usage_fee_col and usage_fee_col in brand_agg.columns else None
    comm_total = float(brand_agg[comm_fee_col].sum())  if comm_fee_col and comm_fee_col in brand_agg.columns else None
    usage_total= float(brand_agg[usage_col].sum())    if usage_col    and usage_col    in brand_agg.columns else None

    base_pct  = (base_total  / fee_total * 100) if (base_total  and fee_total) else 0.0
    us_pct    = (us_total    / fee_total * 100) if (us_total    and fee_total) else 0.0
    comm_pct  = (comm_total  / fee_total * 100) if (comm_total  and fee_total) else 0.0

    # Top-5 concentration
    _top5_share = 0.0
    _top3_names = []
    if fee_col and fee_col in brand_agg.columns and fee_total:
        _fee_sorted = brand_agg[fee_col].sort_values(ascending=False)
        _top5_share = float(_fee_sorted.head(5).sum() / fee_total * 100)
        _top3_names = [str(b) for b in _fee_sorted.head(3).index]

    # IQR stats for fee
    _fee_st = _iqr_stats(brand_agg[fee_col]) if (fee_col and fee_col in brand_agg.columns) else None

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("EHP 열(냉난방) 요금 분석 보고서", styles["cover_title"]))
    story.append(Paragraph("업체별 FCU 요금 현황 및 구성 분석", styles["cover_sub"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_divider_line(content_w))
    story.append(Spacer(1, 0.8 * cm))

    meta = [
        ("보고서 일자", report_date),
        ("분석 업체 수", f"{n_brands}개"),
    ]
    if fee_total is not None:
        meta.append(("FCU 요금 총액", f"{fee_total:,.0f} 원"))
    if usage_total is not None:
        meta.append(("냉난방 사용량 총계", f"{usage_total:,.0f} Mcal"))
    if base_pct or us_pct:
        comp_str = f"기본요금 {base_pct:.1f}%  /  사용요금 {us_pct:.1f}%"
        if comm_pct:
            comp_str += f"  /  공용요금 {comm_pct:.1f}%"
        meta.append(("요금 구성", comp_str))
    if _top5_share:
        meta.append(("상위 5개 업체 집중도", f"{_top5_share:.1f}% ({_conc_label(_top5_share)})"))

    story.append(_kv_table(meta, [4.5*cm, content_w - 4.5*cm], styles))
    story.append(Spacer(1, 1.0 * cm))

    # Key metrics block
    if _fee_st:
        st = _fee_st
        story.append(KeepTogether([
            Paragraph("핵심 지표 (FCU 요금 기준)", styles["sub_title"]),
            Spacer(1, 0.2 * cm),
            _stat_block([
                ("업체 수",   str(st["s"].count()),             "요금 있는 업체"),
                ("중앙값",    f"{st['med']:,.0f} 원",           "50번째 업체 기준"),
                ("평균",      f"{st['mean']:,.0f} 원",          "전체 평균"),
                ("변동계수",  f"CV {st['cv']:.2f}",             _cv_label(st["cv"])),
                ("이상치",    f"{st['n_upper']}개",              f"IQR 상한 {st['upper']:,.0f}원 초과"),
            ], styles, content_w),
        ]))
        story.append(Spacer(1, 0.6 * cm))

        # Plain-language overview
        top3_str = " · ".join(_top3_names[:3]) if _top3_names else "—"
        overview = (
            f"분석 기간 동안 <b>{n_brands}개 업체</b>에 부과된 FCU(냉난방) 요금 총액은 "
            f"<b>{fee_total:,.0f}원</b>입니다. "
            f"업체당 중앙값은 {st['med']:,.0f}원이며, 평균은 {st['mean']:,.0f}원으로 "
        )
        if st["mean"] > st["med"] * 1.25:
            overview += (
                "평균이 중앙값보다 높습니다. 이는 일부 고액 업체가 평균을 끌어올리고 있음을 의미합니다. "
            )
        else:
            overview += "평균과 중앙값이 비슷하여 요금 분포가 비교적 고릅니다. "
        if _top5_share >= 50:
            overview += (
                f"상위 5개 업체({top3_str} 등)가 전체 요금의 <b>{_top5_share:.1f}%</b>를 "
                "차지하고 있어 소수 업체에 비용이 집중된 구조입니다. "
                "이들 업체의 요금 추이를 중점적으로 관리할 필요가 있습니다."
            )
        else:
            overview += (
                f"상위 5개 업체가 전체 요금의 {_top5_share:.1f}%를 차지하며, "
                "요금이 비교적 분산된 구조입니다."
            )
        if st["n_upper"] > 0:
            overview += (
                f" 통계적 이상치(IQR 상한 초과)로 분류된 업체가 <b>{st['n_upper']}개</b> 있으며, "
                "4절(단위 비용 비교)에서 상세 확인할 수 있습니다."
            )
        story.append(_insight_box(overview, styles, content_w))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2 — 업체별 요금 순위
    # ═════════════════════════════════════════════════════════════════════════
    if fee_col and fee_col in brand_agg.columns:
        story.append(_section_bar("  업체별 FCU 요금 순위", styles, content_w))
        story.append(Spacer(1, 0.3 * cm))

        bar_buf = _chart_stacked_hbar(brand_agg, base_col, usage_fee_col, comm_fee_col, fee_col, top_n=25)
        if bar_buf:
            story += _img_flow(
                bar_buf, content_w / cm, styles,
                caption=(
                    "막대 색상: ■ 보라=기본요금(고정)  ■ 초록=사용요금(변동)  ■ 주황=공용요금  "
                    "숫자=소계(원). 소계 내림차순으로 정렬되었습니다."
                ),
            )
        story.append(Spacer(1, 0.5 * cm))

        # Stats for fee
        if _fee_st:
            st = _fee_st
            story.append(KeepTogether([
                Paragraph("요금 분포 통계", styles["sub_title"]),
                Spacer(1, 0.2 * cm),
                _stat_block([
                    ("최솟값",   f"{st['s'].min():,.0f} 원",  ""),
                    ("Q1 (25%)", f"{st['q1']:,.0f} 원",       "하위 25%"),
                    ("중앙값",   f"{st['med']:,.0f} 원",       "정중앙"),
                    ("평균",     f"{st['mean']:,.0f} 원",      _skew_label(st["mean"], st["med"])),
                    ("Q3 (75%)", f"{st['q3']:,.0f} 원",       "상위 25%"),
                    ("최댓값",   f"{st['s'].max():,.0f} 원",   ""),
                ], styles, content_w),
            ]))
            story.append(Spacer(1, 0.4 * cm))

            # Interpretation
            cv_interp = (
                f"변동계수(CV)는 <b>{st['cv']:.2f}</b>로, {_cv_label(st['cv'])}입니다. "
            )
            if st["cv"] >= 0.5:
                cv_interp += (
                    "업체마다 냉난방 이용 규모 차이가 크므로, 일괄 기준을 적용하기보다 "
                    "업체별 개별 점검이 효과적입니다."
                )
            else:
                cv_interp += "대부분 업체의 요금 수준이 비슷하여 전체 평균을 기준으로 관리할 수 있습니다."

            spread = st["s"].max() / st["s"].min() if st["s"].min() > 0 else 0
            spread_str = f"가장 높은 업체는 가장 낮은 업체의 약 <b>{spread:.0f}배</b>에 달합니다. " if spread > 2 else ""

            iqr_str = ""
            if st["n_upper"] > 0:
                top_outliers = brand_agg[fee_col].nlargest(min(3, st["n_upper"])).index.tolist()
                iqr_str = (
                    f"IQR 기준 상위 이상치가 <b>{st['n_upper']}개</b> 있습니다 "
                    f"(상한 기준 {st['upper']:,.0f}원 초과). "
                    f"대표 업체: {', '.join(str(b) for b in top_outliers)}."
                )

            story.append(_insight_box(
                cv_interp + spread_str + iqr_str, styles, content_w
            ))

        # Top-5 callout table
        if fee_col in brand_agg.columns and fee_total:
            story.append(Spacer(1, 0.5 * cm))
            _top5 = brand_agg[fee_col].nlargest(5)
            _rank_cols = [(c, lbl) for c, lbl in [
                (base_col,     "기본요금 (원)"),
                (usage_fee_col,"사용요금 (원)"),
                (comm_fee_col, "공용요금 (원)"),
                (fee_col,      "소계 (원)"),
            ] if c and c in brand_agg.columns]
            hdr_raw = ["순위", "업체명"] + [l for _, l in _rank_cols] + ["비중 (%)"]
            hdr_row = [Paragraph(h, styles["table_hdr"]) for h in hdr_raw]
            rows    = [hdr_row]
            for rank, (brand, fval) in enumerate(_top5.items(), 1):
                r = brand_agg.loc[brand]
                row = [
                    Paragraph(str(rank), styles["table_cell_c"]),
                    Paragraph(textwrap.shorten(str(brand), 22, placeholder="…"), styles["table_cell"]),
                ]
                for c, _ in _rank_cols:
                    v = r.get(c, np.nan)
                    row.append(Paragraph(_f(v), styles["table_cell_c"]))
                row.append(Paragraph(f"{fval / fee_total * 100:.1f}%", styles["table_cell_c"]))
                rows.append(row)

            n_num = len(_rank_cols) + 1
            cw = [0.9*cm, content_w - 0.9*cm - 2.1*cm * n_num] + [2.1*cm] * n_num
            story.append(KeepTogether([
                Paragraph("요금 상위 5개 업체", styles["sub_title"]),
                Spacer(1, 0.2 * cm),
                _mini_table(rows, cw, styles),
            ]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3 — 요금 구성 분석
    # ═════════════════════════════════════════════════════════════════════════
    if base_col and usage_fee_col and fee_col and all(
        c in brand_agg.columns for c in [base_col, usage_fee_col, fee_col]
    ):
        story.append(_section_bar("  요금 구성 분석 — 기본요금 · 사용요금 · 공용요금", styles, content_w))
        story.append(Spacer(1, 0.3 * cm))

        # Overall composition metrics
        comp_rows = [("기본요금 비중", f"{base_pct:.1f}%", "고정 비용")]
        comp_rows.append(("사용요금 비중", f"{us_pct:.1f}%", "실사용 변동 비용"))
        if comm_fee_col and comm_fee_col in brand_agg.columns:
            comp_rows.append(("공용요금 비중", f"{comm_pct:.1f}%", "공용 면적 배분"))
        story.append(_stat_block(comp_rows, styles, content_w))
        story.append(Spacer(1, 0.4 * cm))

        # Donut chart
        donut_buf = _chart_donut(base_pct, us_pct, comm_pct)
        if donut_buf:
            story += _img_flow(
                donut_buf, 9.5, styles,
                caption="전체 FCU 요금 구성 비중. 보라=기본요금(고정), 초록=사용요금(변동), 주황=공용요금.",
            )
        story.append(Spacer(1, 0.4 * cm))

        # Composition interpretation
        dominant = "기본요금" if base_pct >= us_pct else "사용요금"
        dominant_pct = max(base_pct, us_pct)
        comp_interp = (
            f"전체 FCU 요금에서 <b>{dominant}이 {dominant_pct:.1f}%</b>로 가장 큰 비중을 차지합니다. "
        )
        if base_pct >= 60:
            comp_interp += (
                "기본요금 비중이 60%를 넘어, 실제 냉난방 사용량과 무관하게 발생하는 고정 비용이 "
                "전체의 과반을 차지합니다. 계약 구조나 용량 설정을 검토할 필요가 있습니다."
            )
        elif base_pct >= 40:
            comp_interp += (
                "기본요금과 사용요금이 비교적 균형 잡힌 구조입니다. "
                "에너지 절감 효과를 높이려면 사용량 관리에 집중하는 것이 효과적입니다."
            )
        else:
            comp_interp += (
                "사용요금 비중이 높아 실제 냉난방 사용량이 요금에 미치는 영향이 큽니다. "
                "절전 행동 변화나 운영 시간 조정으로 요금 절감 여지가 있습니다."
            )
        if comm_pct > 10:
            comp_interp += (
                f" 공용요금도 {comm_pct:.1f}%로 적지 않은 비중을 차지하고 있어, "
                "공용 면적 냉난방 운영 효율을 별도로 점검할 것을 권장합니다."
            )
        story.append(_insight_box(comp_interp, styles, content_w))
        story.append(Spacer(1, 0.5 * cm))

        # Per-brand composition stats
        _denom = brand_agg[fee_col].replace(0, np.nan)
        _bp = (brand_agg[base_col] / _denom * 100).dropna()
        _up = (brand_agg[usage_fee_col] / _denom * 100).dropna()
        if len(_bp) >= 2:
            n_base_heavy  = int((_bp > 60).sum())
            n_usage_heavy = int((_up > 60).sum())
            base_spread   = float(_bp.max() - _bp.min())

            brand_comp_interp = (
                f"업체별 기본요금 비중의 범위는 <b>{_bp.min():.1f}% ~ {_bp.max():.1f}%</b> "
                f"({base_spread:.1f}%p 차이)입니다. "
            )
            if base_spread > 30:
                brand_comp_interp += (
                    "업체 간 편차가 크므로 단일 기준으로 관리하기보다 개별 업체 특성에 맞는 접근이 필요합니다. "
                )
            if n_base_heavy:
                brand_comp_interp += (
                    f"기본요금 비중이 60%를 초과하는 업체가 <b>{n_base_heavy}개</b> 있습니다. "
                    "이들은 사용량에 비해 고정 비용 부담이 크므로, 계약 용량 재조정을 검토하세요. "
                )
            if n_usage_heavy:
                brand_comp_interp += (
                    f"사용요금 비중이 60%를 초과하는 업체가 <b>{n_usage_heavy}개</b>로, "
                    "에너지 절감 개선 효과가 가장 크게 나타날 수 있는 업체군입니다."
                )
            story.append(_insight_box(brand_comp_interp, styles, content_w))

            # Top-5 기본요금 비중 업체
            if n_base_heavy > 0:
                story.append(Spacer(1, 0.4 * cm))
                _bh_idx = _bp.nlargest(min(5, n_base_heavy))
                bh_hdr  = [Paragraph(h, styles["table_hdr"])
                           for h in ["순위", "업체명", "기본(%)", "사용(%)", "소계 (원)"]]
                bh_rows = [bh_hdr]
                for rank, (brand, bpct) in enumerate(_bh_idx.items(), 1):
                    bh_rows.append([
                        Paragraph(str(rank), styles["table_cell_c"]),
                        Paragraph(textwrap.shorten(str(brand), 22, placeholder="…"), styles["table_cell"]),
                        Paragraph(f"{bpct:.1f}%",           styles["table_cell_c"]),
                        Paragraph(f"{_up.get(brand, 0):.1f}%", styles["table_cell_c"]),
                        Paragraph(_f(brand_agg.loc[brand, fee_col]), styles["table_cell_c"]),
                    ])
                bh_cw = [0.9*cm, content_w - 0.9*cm - 2.0*cm*3] + [2.0*cm]*3
                story.append(KeepTogether([
                    Paragraph(f"기본요금 비중 상위 5개 업체", styles["sub_title"]),
                    Spacer(1, 0.2 * cm),
                    _mini_table(bh_rows, bh_cw, styles, header_bg=colors.HexColor("#6C3483")),
                ]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 4 — 단위 비용 비교
    # ═════════════════════════════════════════════════════════════════════════
    _has_unit = fee_col and fee_col in brand_agg.columns and (area_col or usage_col)
    if _has_unit:
        story.append(_section_bar("  단위 비용 비교 — 면적·사용량 기준 공정 비교", styles, content_w))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph(
            "단위 비용은 업체 규모(면적·사용량)의 차이를 보정하여 업체 간 요금 부담을 동일 기준으로 "
            "비교하는 지표입니다. 절대 요금이 낮더라도 단위 비용이 높으면 효율 개선 여지가 있습니다.",
            styles["note"],
        ))
        story.append(Spacer(1, 0.4 * cm))

        # Per-area
        if area_col and area_col in brand_agg.columns:
            _a    = brand_agg[area_col].where(brand_agg[area_col] > 0)
            _pm2  = (brand_agg[fee_col] / _a).dropna()
            _pm2  = _pm2[_pm2 > 0]
            _pa_st = _iqr_stats(_pm2)

            if _pa_st:
                pa = _pa_st
                story.append(KeepTogether([
                    Paragraph("단위면적당 요금 (원/㎡)", styles["sub_title"]),
                    Spacer(1, 0.2 * cm),
                    _stat_block([
                        ("중앙값",    f"{pa['med']:,.1f}",   "기준 업체 부담"),
                        ("평균",      f"{pa['mean']:,.1f}",  _skew_label(pa["mean"], pa["med"])),
                        ("IQR 상한",  f"{pa['upper']:,.1f}", "이상치 기준선"),
                        ("이상치",    f"{pa['n_upper']}개",   "초과 업체 수"),
                        ("변동계수",  f"{pa['cv']:.2f}",     _cv_label(pa["cv"])),
                    ], styles, content_w),
                ]))
                story.append(Spacer(1, 0.3 * cm))

                pa_buf = _chart_hbar_single(
                    _pm2, f"업체별 단위면적당 FCU 요금 (상위 {min(25, len(_pm2))}개)",
                    "원/㎡", color="#4C72B0", top_n=25, iqr_fence=pa["upper"],
                )
                if pa_buf:
                    story += _img_flow(
                        pa_buf, content_w / cm, styles,
                        caption=(
                            "단위면적당 FCU 요금. 빨간 막대=IQR 상한 초과 이상치 업체. "
                            "점선=평균, 점점선=IQR 상한 기준."
                        ),
                    )
                story.append(Spacer(1, 0.4 * cm))

                # Interpretation
                pa_interp = (
                    f"전체 업체의 단위면적당 요금 중앙값은 <b>{pa['med']:,.1f} 원/㎡</b>입니다. "
                )
                if pa["mean"] > pa["med"] * 1.2:
                    pa_interp += (
                        f"평균({pa['mean']:,.1f} 원/㎡)이 중앙값보다 높아, "
                        "일부 고부담 업체가 평균을 끌어올리고 있습니다. "
                    )
                if pa["n_upper"] > 0:
                    _out_brands = _pm2[_pm2 > pa["upper"]].nlargest(3).index.tolist()
                    pa_interp += (
                        f"IQR 기준 이상치(상한 {pa['upper']:,.1f} 원/㎡ 초과)로 분류된 업체가 "
                        f"<b>{pa['n_upper']}개</b>입니다. "
                        f"대표 업체: {', '.join(str(b) for b in _out_brands)}. "
                        "이들 업체는 냉난방 효율이 낮거나 계약 구조 점검이 필요합니다."
                    )
                else:
                    pa_interp += "IQR 기준 이상치 업체는 없어, 단위면적당 부담 수준이 비교적 균일합니다."
                story.append(_insight_box(pa_interp, styles, content_w))

                # Outlier table
                if pa["n_upper"] > 0:
                    story.append(Spacer(1, 0.4 * cm))
                    _out = _pm2[_pm2 > pa["upper"]].sort_values(ascending=False).head(8)
                    out_hdr = [Paragraph(h, styles["table_hdr"])
                               for h in ["순위", "업체명", "원/㎡", "원/평", "소계 (원)", "면적 (㎡)"]]
                    out_rows = [out_hdr]
                    for rank, (brand, val) in enumerate(_out.items(), 1):
                        row_v = brand_agg.loc[brand]
                        out_rows.append([
                            Paragraph(str(rank), styles["table_cell_c"]),
                            Paragraph(textwrap.shorten(str(brand), 20, placeholder="…"), styles["table_cell"]),
                            Paragraph(f"{val:,.1f}",                       styles["table_cell_c"]),
                            Paragraph(f"{val * 3.305785:,.1f}",            styles["table_cell_c"]),
                            Paragraph(_f(row_v.get(fee_col)),              styles["table_cell_c"]),
                            Paragraph(_f(row_v.get(area_col), 1),         styles["table_cell_c"]),
                        ])
                    out_cw = [0.8*cm, content_w - 0.8*cm - 2.0*cm*4] + [2.0*cm]*4
                    story.append(KeepTogether([
                        Paragraph(f"단위면적당 요금 이상치 업체 (IQR 상한 {pa['upper']:,.1f} 원/㎡ 초과)",
                                  styles["sub_title"]),
                        Spacer(1, 0.2 * cm),
                        _mini_table(out_rows, out_cw, styles, header_bg=_C_RED),
                    ]))

        # Per-usage
        if usage_col and usage_col in brand_agg.columns:
            story.append(Spacer(1, 0.8 * cm))
            _u   = brand_agg[usage_col].where(brand_agg[usage_col] > 0)
            _pmc = (brand_agg[fee_col] / _u).dropna()
            _pmc = _pmc[_pmc > 0]
            _pu_st = _iqr_stats(_pmc)

            if _pu_st:
                pu = _pu_st
                story.append(KeepTogether([
                    Paragraph("단위사용량당 요금 (원/Mcal)", styles["sub_title"]),
                    Spacer(1, 0.2 * cm),
                    _stat_block([
                        ("중앙값",   f"{pu['med']:,.1f}",   "기준 단가"),
                        ("평균",     f"{pu['mean']:,.1f}",  ""),
                        ("IQR 상한", f"{pu['upper']:,.1f}", "이상치 기준"),
                        ("이상치",   f"{pu['n_upper']}개",   "초과 업체"),
                        ("변동계수", f"{pu['cv']:.2f}",     _cv_label(pu["cv"])),
                    ], styles, content_w),
                ]))
                story.append(Spacer(1, 0.3 * cm))

                pu_buf = _chart_hbar_single(
                    _pmc, f"업체별 단위사용량당 FCU 요금 (상위 {min(25, len(_pmc))}개)",
                    "원/Mcal", color="#4C72B0", top_n=25, iqr_fence=pu["upper"],
                )
                if pu_buf:
                    story += _img_flow(
                        pu_buf, content_w / cm, styles,
                        caption="단위사용량당 FCU 요금. 같은 Mcal을 사용할 때 업체별 지출 차이를 보여줍니다.",
                    )
                story.append(Spacer(1, 0.3 * cm))

                pu_interp = (
                    f"냉난방 사용량 1 Mcal당 요금 중앙값은 <b>{pu['med']:,.1f} 원/Mcal</b>입니다. "
                )
                if pu["n_upper"] > 0:
                    _out_brands = _pmc[_pmc > pu["upper"]].nlargest(3).index.tolist()
                    pu_interp += (
                        f"단가가 IQR 상한({pu['upper']:,.1f} 원/Mcal)을 초과하는 업체가 "
                        f"<b>{pu['n_upper']}개</b>입니다 ({', '.join(str(b) for b in _out_brands)} 등). "
                        "동일한 냉난방 사용량에 대해 다른 업체보다 요금이 현저히 높으므로 "
                        "기본요금 비중 또는 계약 조건을 점검하세요."
                    )
                else:
                    pu_interp += "IQR 이상치 없이 단위 요금이 균일합니다."
                story.append(_insight_box(pu_interp, styles, content_w))

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 5 — 면적 vs 단위 요금 상관 (only if both available)
    # ═════════════════════════════════════════════════════════════════════════
    if (usage_col and area_col and fee_col and
            all(c in brand_agg.columns for c in [usage_col, area_col, fee_col])):
        story.append(PageBreak())
        story.append(_section_bar("  면적과 단위사용량 요금의 관계", styles, content_w))
        story.append(Spacer(1, 0.3 * cm))

        _a_s  = brand_agg[area_col].where(brand_agg[area_col] > 0)
        _u_s  = brand_agg[usage_col].where(brand_agg[usage_col] > 0)
        _pu_s = (brand_agg[fee_col] / _u_s).dropna()

        sc_buf = _chart_scatter(
            _a_s, _pu_s,
            "업체 면적 (㎡)", "단위사용량 요금 (원/Mcal)",
            "면적 vs 단위사용량당 요금",
        )
        if sc_buf:
            pair_df = pd.DataFrame({"x": _a_s, "y": _pu_s}).dropna()
            if len(pair_df) >= 3:
                r_val = float(np.corrcoef(pair_df["x"].values, pair_df["y"].values)[0, 1])
                r2    = r_val ** 2
                story += _img_flow(
                    sc_buf, content_w / cm, styles,
                    caption=(
                        f"산점도: 업체 면적(X축)과 단위사용량 요금(Y축). "
                        f"추세선 r = {r_val:+.2f}, R² = {r2:.2f}."
                    ),
                )
                story.append(Spacer(1, 0.4 * cm))

                if abs(r_val) >= 0.5:
                    direction = "면적이 클수록 단위 요금이 높아지는 경향" if r_val > 0 else "면적이 클수록 단위 요금이 낮아지는 경향"
                    sc_interp = (
                        f"상관계수 r = {r_val:+.2f} (R² = {r2:.2f})으로, <b>{direction}</b>이 있습니다. "
                    )
                    if r_val < -0.3:
                        sc_interp += (
                            "규모가 큰 업체일수록 단위 요금이 낮게 나타나는 규모의 경제 효과로 볼 수 있습니다. "
                            "소규모 업체는 상대적으로 단위 비용이 높을 가능성이 있어 별도 점검이 필요합니다."
                        )
                    elif r_val > 0.3:
                        sc_interp += (
                            "면적이 넓은 업체에서 단위 요금이 오히려 높게 나타납니다. "
                            "대규모 업체의 냉난방 운영 효율이나 계약 구조를 검토할 필요가 있습니다."
                        )
                else:
                    sc_interp = (
                        f"상관계수 r = {r_val:+.2f}로, 면적 크기와 단위 요금 간에 "
                        "뚜렷한 선형 관계는 없습니다. "
                        "단위 요금 차이는 업체 면적보다 계약 조건, 운영 패턴 등 다른 요인의 영향이 클 수 있습니다."
                    )
                story.append(_insight_box(sc_interp, styles, content_w))

    # ═════════════════════════════════════════════════════════════════════════
    # BACK MATTER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("보고서 끝", styles["cover_title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        f"작성일: {report_date}. "
        "본 보고서는 내부 관리 목적으로만 사용하시기 바랍니다. "
        "금액 단위: 원 · 사용량 단위: Mcal · 면적 단위: ㎡. "
        "IQR 이상치 기준: Q3 + 1.5 × IQR.",
        styles["note"],
    ))

    doc.build(story, canvasmaker=_make_numbered_canvas(T_footer))
    return buf.getvalue()

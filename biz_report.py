"""
biz_report.py — PDF report generators with charts, narrative insights, and action items.
  - generate_anomaly_pdf()        : 이상감지
  - generate_cross_pdf()          : 비용분석
  - generate_efficiency_pdf()     : 효율분석
  - generate_comprehensive_pdf()  : 종합 (all sections in one doc)
  - generate_insight_pdf()        : 인사이트 (cost + efficiency)
"""
from __future__ import annotations

import io
from datetime import date as _date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

import numpy as np
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, KeepTogether, PageBreak,
    Paragraph, Spacer, Table, TableStyle, Image,
)

from report import (
    C_BLUE, C_CRITICAL, C_DIVIDER, C_LIGHT, C_NAVY, C_WHITE,
    C_WATCH, C_ALERT, C_STABLE, C_NORMAL,
    _ensure_fonts, _make_numbered_canvas, _make_page_template,
    _make_styles, _section_bar,
    _FONT_REG,
)

# ── Risk colour map ───────────────────────────────────────────────────────────
_RISK_COLOR_RL = {
    "🔴 위험": C_CRITICAL,
    "🟠 주의": C_WATCH,
    "🟡 관찰": C_ALERT,
    "🟢 정상": C_STABLE,
}
_RISK_PLAIN = {
    "🔴 위험": "위험",
    "🟠 주의": "주의",
    "🟡 관찰": "관찰",
    "🟢 정상": "정상",
}
_RISK_MPL = {
    "🔴 위험": "#E63946",
    "🟠 주의": "#F4882A",
    "🟡 관찰": "#E8B84B",
    "🟢 정상": "#43AA6F",
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _f(val, decimals=1, suffix=""):
    if val is None:
        return "—"
    try:
        f = float(val)
        if np.isnan(f):
            return "—"
        return f"{f:,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _std_table(data, col_w, styles):
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("FONTNAME",      (0, 1), (-1, -1), "NanumGothic"),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
    ] + styles)
    return Table(data, colWidths=col_w, style=ts, repeatRows=1)


def _build_doc(buf, footer_left=None):
    _ensure_fonts()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=2.0 * cm,  bottomMargin=2.0 * cm,
    )
    T = _make_styles()
    T.setdefault("footer_left", footer_left or "비즈니스 분석 보고서  ·  대외비")
    T.setdefault("footer_page", "{n} / {total} 페이지")
    template = _make_page_template(doc, T)
    doc.addPageTemplates([template])
    return doc, T


def _cover_items(title: str, subtitle: str, context: dict, T) -> list:
    ctx = context or {}
    items = [
        Paragraph(title, T["cover_title"]),
        Spacer(1, 0.3 * cm),
        Paragraph(subtitle, T["cover_sub"]),
        Spacer(1, 0.5 * cm),
        Paragraph(f"작성일: {ctx.get('date', str(_date.today()))}", T["note"]),
    ]
    if ctx.get("period"):
        items.append(Paragraph(f"분석 기간: {ctx['period']}", T["note"]))
    if ctx.get("buildings"):
        items.append(Paragraph(f"대상 건물: {ctx['buildings']}", T["note"]))
    items.append(Spacer(1, 0.8 * cm))
    return items


def _mpl_font():
    """Get matplotlib FontProperties for NanumGothic."""
    import os
    if os.path.exists(_FONT_REG):
        return _fm.FontProperties(fname=_FONT_REG)
    return _fm.FontProperties()


def _fig_to_buf(fig):
    """Render matplotlib figure to PNG BytesIO."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _img_flowable(png_buf, width_cm):
    """Convert PNG buffer to ReportLab Image flowable."""
    pil = PILImage.open(png_buf)
    w_px, h_px = pil.size
    png_buf.seek(0)
    return Image(png_buf, width=width_cm * cm, height=width_cm * cm * h_px / w_px)


def _insight_para(text, T):
    """Paragraph with the body style for narrative insights."""
    return Paragraph(text, T["body"])


def _action_box(items: list[str], T, W) -> list:
    """Create a highlighted action-items box."""
    flowables = []
    flowables.append(_section_bar("📋 조치 권고사항", T, W))
    for i, item in enumerate(items, 1):
        flowables.append(Paragraph(f"<b>{i}.</b> {item}", T["body"]))
        flowables.append(Spacer(1, 0.15 * cm))
    flowables.append(Spacer(1, 0.3 * cm))
    return flowables


# ═══════════════════════════════════════════════════════════════════════════════
# Story builders — reusable flowable lists (no doc/cover)
# ═══════════════════════════════════════════════════════════════════════════════

def _anomaly_story(anomaly_df: pd.DataFrame, T, W) -> list:
    """Return reportlab flowables for the anomaly section with charts + insights."""
    story = []
    fp = _mpl_font()
    total = len(anomaly_df)
    risk_counts = anomaly_df["risk_level"].value_counts().to_dict() if "risk_level" in anomaly_df.columns else {}

    danger  = risk_counts.get("🔴 위험", 0)
    caution = risk_counts.get("🟠 주의", 0)
    observe = risk_counts.get("🟡 관찰", 0)
    normal  = risk_counts.get("🟢 정상", 0)
    flagged = danger + caution

    # ── Executive summary ─────────────────────────────────────────────────
    story.append(_section_bar("요약", T, W))
    pct_flagged = flagged / total * 100 if total else 0
    summary_lines = [
        f"전체 <b>{total}개</b> 브랜드 중 <b>{flagged}개({pct_flagged:.0f}%)</b>가 "
        f"위험 또는 주의 등급으로 분류되었습니다.",
    ]
    if danger:
        summary_lines.append(
            f"<font color='#E63946'><b>위험 등급 {danger}개</b></font> 브랜드는 "
            "복합 이상 점수 0.65 이상으로, 즉시 조사가 필요합니다."
        )
    if caution:
        summary_lines.append(
            f"<font color='#F4882A'><b>주의 등급 {caution}개</b></font> 브랜드는 "
            "하나 이상의 유틸리티에서 비정상적 패턴이 감지되었습니다."
        )
    for line in summary_lines:
        story.append(Paragraph(line, T["body"]))
        story.append(Spacer(1, 0.15 * cm))
    story.append(Spacer(1, 0.3 * cm))

    # ── Risk distribution chart ───────────────────────────────────────────
    labels = ["위험", "주의", "관찰", "정상"]
    values = [danger, caution, observe, normal]
    mpl_colors = ["#E63946", "#F4882A", "#E8B84B", "#43AA6F"]
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, mpl_colors) if v > 0]

    if non_zero:
        fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="white")

        # Pie chart
        pie_labels, pie_vals, pie_colors = zip(*non_zero)
        wedges, texts, autotexts = ax_pie.pie(
            pie_vals, labels=pie_labels, colors=pie_colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontproperties": fp, "fontsize": 10},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax_pie.set_title("위험 등급 분포", fontproperties=fp, fontsize=12, fontweight="bold")

        # Top 10 composite score bar chart
        top_n = anomaly_df.nlargest(min(10, total), "composite_score")
        brands = top_n["brand"].tolist()
        scores = top_n["composite_score"].tolist()
        bar_colors = [_RISK_MPL.get(r, "#888888") for r in top_n.get("risk_level", [])]
        if len(bar_colors) < len(brands):
            bar_colors = ["#E63946"] * len(brands)

        y_pos = range(len(brands))
        ax_bar.barh(y_pos, scores, color=bar_colors, edgecolor="white", height=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(brands, fontproperties=fp, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("복합 이상 점수", fontproperties=fp, fontsize=10)
        ax_bar.set_title("이상 점수 상위 브랜드", fontproperties=fp, fontsize=12, fontweight="bold")
        ax_bar.axvline(0.65, color="#E63946", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_bar.axvline(0.40, color="#F4882A", linestyle="--", linewidth=0.8, alpha=0.7)
        for s, score in zip(y_pos, scores):
            ax_bar.text(score + 0.01, s, f"{score:.2f}", va="center", fontsize=7, color="#333")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        fig.tight_layout(pad=2.0)
        chart_buf = _fig_to_buf(fig)
        story.append(_img_flowable(chart_buf, width_cm=17))
        story.append(Spacer(1, 0.3 * cm))

    # ── Score methodology note ────────────────────────────────────────────
    story.append(Paragraph(
        "<i>복합 점수 = 급등(30%) + 소비(25%) + 비용(25%) + HVAC(10%) + 일관성(10%)  |  "
        "위험 ≥ 0.65  ·  주의 ≥ 0.40  ·  관찰 ≥ 0.20  ·  정상 &lt; 0.20</i>",
        T["caption"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── KPI row ───────────────────────────────────────────────────────────
    kpi_data = [
        ["총 브랜드", "🔴 위험", "🟠 주의", "🟡 관찰", "🟢 정상"],
        [str(total), str(danger), str(caution), str(observe), str(normal)],
    ]
    kpi_col = [W / 5] * 5
    kpi_ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "NanumGothic-Bold"),
        ("FONTNAME",      (0, 1), (-1, -1), "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND",    (1, 1), (1, 1), C_CRITICAL),
        ("BACKGROUND",    (2, 1), (2, 1), C_WATCH),
        ("BACKGROUND",    (3, 1), (3, 1), C_ALERT),
        ("BACKGROUND",    (4, 1), (4, 1), C_STABLE),
        ("TEXTCOLOR",     (1, 1), (4, 1), C_WHITE),
    ])
    story.append(Table(kpi_data, colWidths=kpi_col, style=kpi_ts))
    story.append(Spacer(1, 0.5 * cm))

    # ── High-risk brands — narrative + table ──────────────────────────────
    high_risk = anomaly_df[
        anomaly_df.get("risk_level", pd.Series(dtype=str)).isin(["🔴 위험", "🟠 주의"])
    ].copy() if "risk_level" in anomaly_df.columns else pd.DataFrame()

    if not high_risk.empty:
        story.append(_section_bar(f"위험·주의 브랜드 상세 ({len(high_risk)}개)", T, W))

        # Narrative: explain WHY each top brand is flagged
        top3 = high_risk.nlargest(min(3, len(high_risk)), "composite_score")
        for _, r in top3.iterrows():
            brand = r.get("brand", "?")
            bldg = r.get("building", "")
            score = r.get("composite_score", 0)
            reason = r.get("reason", "")
            rl = _RISK_PLAIN.get(str(r.get("risk_level", "")), "")
            loc = f" ({bldg}동)" if bldg else ""
            narrative = f"<b>{brand}</b>{loc} — {rl} (점수 {score:.2f})"
            if reason and reason != "—":
                narrative += f": {reason}"
            story.append(Paragraph(narrative, T["body"]))
            story.append(Spacer(1, 0.1 * cm))
        if len(high_risk) > 3:
            story.append(Paragraph(
                f"<i>외 {len(high_risk) - 3}개 브랜드 — 아래 표 참조</i>",
                T["caption"],
            ))
        story.append(Spacer(1, 0.3 * cm))

        high_risk = high_risk.sort_values("composite_score", ascending=False)
        headers = ["브랜드", "건물", "등급", "복합점수", "급등%", "급등항목", "소비", "비용", "HVAC", "일관성"]
        col_w = [c * cm for c in [3.0, 1.0, 1.1, 1.3, 1.2, 1.5, 1.1, 1.1, 1.1, 1.1]]

        rows = [headers]
        row_styles = []
        for i, (_, r) in enumerate(high_risk.iterrows(), start=1):
            rl = str(r.get("risk_level", ""))
            rows.append([
                str(r.get("brand", "")),
                str(r.get("building", "—")),
                _RISK_PLAIN.get(rl, rl),
                _f(r.get("composite_score"), 3),
                _f(r.get("spike_max_pct"), 1, "%"),
                str(r.get("spike_worst_util", "—")),
                _f(r.get("consumption_score"), 3),
                _f(r.get("cost_score"), 3),
                _f(r.get("hvac_score"), 3),
                _f(r.get("consistency_score"), 3),
            ])
            c = _RISK_COLOR_RL.get(rl)
            if c:
                row_styles.append(("BACKGROUND", (2, i), (2, i), c))
                row_styles.append(("TEXTCOLOR",  (2, i), (2, i), C_WHITE))

        story.append(KeepTogether([_std_table(rows, col_w, row_styles)]))
        story.append(Spacer(1, 0.5 * cm))

    # ── Action items ──────────────────────────────────────────────────────
    actions = []
    if danger:
        danger_brands = high_risk[
            high_risk.get("risk_level", pd.Series(dtype=str)) == "🔴 위험"
        ]["brand"].tolist()[:5]
        actions.append(
            f"<b>즉시 조사</b>: 위험 등급 브랜드 ({', '.join(danger_brands)}) 대상 "
            "검침 데이터 교차 확인 및 현장 점검 실시"
        )
    if caution:
        actions.append(
            "<b>주의 모니터링</b>: 주의 등급 브랜드의 다음달 검침 결과를 추적하여 "
            "일시적 변동인지 지속적 이상인지 확인"
        )

    # Check for spike-driven anomalies
    if "spike_max_pct" in anomaly_df.columns:
        big_spikes = anomaly_df[anomaly_df["spike_max_pct"].fillna(0) > 50]
        if not big_spikes.empty:
            spike_brands = big_spikes.nlargest(3, "spike_max_pct")
            spike_items = [
                f"{r['brand']}(+{r['spike_max_pct']:.0f}%)"
                for _, r in spike_brands.iterrows()
            ]
            actions.append(
                f"<b>급등 확인</b>: 전월 대비 50% 이상 급등한 {', '.join(spike_items)} 등 "
                f"총 {len(big_spikes)}건 — 계량기 오작동 또는 누수 가능성 점검"
            )

    # Check for consistency issues
    if "consistency_score" in anomaly_df.columns:
        inconsistent = anomaly_df[anomaly_df["consistency_score"].fillna(0) > 0.5]
        if not inconsistent.empty:
            actions.append(
                f"<b>계량 일관성</b>: {len(inconsistent)}개 브랜드에서 시트 간 "
                "데이터 불일치 또는 미계량 항목 발견 — 검침 프로세스 검증 필요"
            )

    if not actions:
        actions.append("현재 심각한 이상이 감지되지 않았습니다. 정기 모니터링을 지속하세요.")
    story += _action_box(actions, T, W)

    # ── Full results table (top 30 by score) ──────────────────────────────
    story.append(PageBreak())
    story.append(_section_bar("전체 결과 (복합 점수 상위 30개)", T, W))
    story.append(Paragraph(
        "아래 표는 복합 이상 점수 기준 상위 30개 브랜드입니다. "
        "각 하위 점수 열은 해당 차원에서의 이상 정도를 0~1 스케일로 나타냅니다.",
        T["caption"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    id_cols = [c for c in ["brand", "building"] if c in anomaly_df.columns]
    score_cols = [c for c in ["composite_score", "risk_level", "spike_score",
                               "consumption_score", "cost_score",
                               "hvac_score", "consistency_score"] if c in anomaly_df.columns]
    full = anomaly_df[id_cols + score_cols].sort_values(
        "composite_score", ascending=False
    ).head(30)

    ko_map = {
        "brand": "브랜드", "building": "건물", "composite_score": "복합점수",
        "risk_level": "등급", "spike_score": "급등", "consumption_score": "소비",
        "cost_score": "비용", "hvac_score": "HVAC", "consistency_score": "일관성",
    }
    full_headers = [ko_map.get(c, c) for c in full.columns]
    n_cols = len(full_headers)
    col_w = [W / n_cols] * n_cols

    full_rows = [full_headers]
    full_styles = []
    for i, (_, r) in enumerate(full.iterrows(), start=1):
        row = []
        for c in full.columns:
            v = r[c]
            if c == "risk_level":
                row.append(_RISK_PLAIN.get(str(v), str(v)))
                c_color = _RISK_COLOR_RL.get(str(v))
                if c_color:
                    ci = list(full.columns).index(c)
                    full_styles.append(("BACKGROUND", (ci, i), (ci, i), c_color))
                    full_styles.append(("TEXTCOLOR",  (ci, i), (ci, i), C_WHITE))
            elif isinstance(v, float):
                row.append(_f(v, 3))
            else:
                row.append(str(v) if not pd.isna(v) else "—")
        full_rows.append(row)

    story.append(_std_table(full_rows, col_w, full_styles))
    return story


def _cross_story(unit_df, elec_df, T, W) -> list:
    """Return reportlab flowables for the cost analysis section with charts + insights."""
    story = []
    fp = _mpl_font()

    # ── Unit cost section ─────────────────────────────────────────────────
    if unit_df is not None and not unit_df.empty:
        story.append(_section_bar("단위 비용 분석", T, W))

        # Narrative overview
        cost_metrics = []
        for col, label, unit in [
            ("water_unit_cost", "수도", "₩/m³"),
            ("elect_unit_cost", "전기", "₩/kWh"),
        ]:
            if col in unit_df.columns:
                vals = pd.to_numeric(unit_df[col], errors="coerce").dropna()
                if not vals.empty:
                    cost_metrics.append((label, unit, vals.mean(), vals.std(), vals.median()))

        if cost_metrics:
            for label, unit, avg, std, med in cost_metrics:
                story.append(Paragraph(
                    f"<b>{label} 단가</b>: 평균 {avg:,.0f} {unit}, "
                    f"중앙값 {med:,.0f} {unit}, 표준편차 {std:,.0f}",
                    T["body"],
                ))
            story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph(
            "Z-점수 |Z| ≥ 2.0인 브랜드는 동종 대비 비용 이상으로 분류됩니다. "
            "양의 Z-점수는 평균보다 비싼 단가, 음의 Z-점수는 비정상적으로 낮은 단가를 의미합니다.",
            T["caption"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Chart: unit cost distribution with outlier highlighting
        chart_cols = [(c, l) for c, l in [
            ("water_unit_cost", "수도 단가(₩/m³)"),
            ("elect_unit_cost", "전기 단가(₩/kWh)"),
        ] if c in unit_df.columns]

        if chart_cols:
            n_charts = len(chart_cols)
            fig, axes = plt.subplots(1, n_charts, figsize=(5 * n_charts, 4), facecolor="white")
            if n_charts == 1:
                axes = [axes]

            for ax, (col, label) in zip(axes, chart_cols):
                z_col = col.replace("_cost", "_z") if col.endswith("_cost") else col + "_z"
                vals = pd.to_numeric(unit_df[col], errors="coerce")
                z_vals = pd.to_numeric(unit_df.get(z_col, pd.Series(dtype=float)), errors="coerce")
                brands = unit_df["brand"].tolist()

                valid = vals.notna()
                v = vals[valid].values
                b = [brands[i] for i in range(len(brands)) if valid.iloc[i]]
                z = z_vals[valid].values if len(z_vals) == len(vals) else np.zeros(len(v))

                if len(v) == 0:
                    continue

                # Sort by value
                order = np.argsort(v)[::-1]
                v, b, z = v[order], [b[i] for i in order], z[order]

                # Show top 15
                n_show = min(15, len(v))
                v, b, z = v[:n_show], b[:n_show], z[:n_show]

                bar_colors = ["#E63946" if abs(zz) >= 2.0 else "#2E6DA4" for zz in z]
                y_pos = range(len(b))
                ax.barh(y_pos, v, color=bar_colors, edgecolor="white", height=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(b, fontproperties=fp, fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel(label, fontproperties=fp, fontsize=9)
                ax.set_title(label, fontproperties=fp, fontsize=11, fontweight="bold")
                mean_v = np.mean(v)
                ax.axvline(mean_v, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
                for i, val in enumerate(v):
                    ax.text(val + max(v) * 0.01, i, f"{val:,.0f}", va="center", fontsize=6.5, color="#333")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            fig.tight_layout(pad=2.0)
            chart_buf = _fig_to_buf(fig)
            story.append(_img_flowable(chart_buf, width_cm=17))
            story.append(Paragraph(
                "<i>🔴 빨간색 = |Z| ≥ 2.0 (이상), 🔵 파란색 = 정상 범위, --- = 평균</i>",
                T["caption"],
            ))
            story.append(Spacer(1, 0.3 * cm))

        # Anomaly summary with narrative
        anom_brands = []
        cost_cols, col_labels = [], []
        for col, label in [
            ("water_unit_cost", "수도 단가(₩/m³)"),
            ("water_unit_z",    "수도 등급"),
            ("elect_unit_cost", "전기 단가(₩/kWh)"),
            ("elect_unit_z",    "전기 등급"),
            ("total_cost_per_py", "평당비용(만₩/평)"),
            ("total_cost_per_py_z", "평당비용 등급"),
            ("total_cost_per_m2", "총비용(만₩/m²)"),
            ("total_cost_per_m2_z", "총비용 등급"),
        ]:
            if col in unit_df.columns:
                cost_cols.append(col)
                col_labels.append(label)

        anom_rows = []
        for col, label in zip(cost_cols, col_labels):
            if col.endswith("_z") and col in unit_df.columns:
                flags = unit_df[unit_df[col].abs() >= 2.0]
                for _, r in flags.iterrows():
                    brand = str(r.get("brand", ""))
                    z_val = float(r.get(col, 0)) if not pd.isna(r.get(col)) else 0
                    from utils import z_to_grade as _ztg
                    direction = "고비용" if z_val > 0 else "저비용"
                    grade = _ztg(z_val)
                    anom_brands.append((brand, label.replace(" 등급", ""), z_val, direction))
                    anom_rows.append([brand, label.replace(" 등급", ""), grade])

        if anom_rows:
            story.append(_section_bar(f"비용 이상 브랜드 ({len(anom_rows)}건)", T, W))

            # Narrative for top anomalies
            for brand, metric, z_val, direction in anom_brands[:3]:
                story.append(Paragraph(
                    f"<b>{brand}</b>: {metric} Z={z_val:+.1f} ({direction}) — "
                    f"동일 건물 내 동종 대비 {'높은' if z_val > 0 else '낮은'} 단가로, "
                    f"{'과다 청구 가능성' if z_val > 0 else '계량 오류 가능성'} 검토 필요",
                    T["body"],
                ))
                story.append(Spacer(1, 0.1 * cm))
            story.append(Spacer(1, 0.2 * cm))

            anom_data = [["브랜드", "이상 항목", "Z-점수"]] + anom_rows
            anom_w = [5 * cm, 5 * cm, W - 10 * cm]
            anom_styles = [
                ("TEXTCOLOR", (2, 1), (2, -1), C_CRITICAL),
                ("FONTNAME",  (2, 1), (2, -1), "NanumGothic-Bold"),
            ]
            story.append(_std_table(anom_data, anom_w, anom_styles))
            story.append(Spacer(1, 0.3 * cm))

        # Cost action items
        cost_actions = []
        high_cost = [b for b, m, z, d in anom_brands if z > 2.0]
        low_cost = [b for b, m, z, d in anom_brands if z < -2.0]
        if high_cost:
            cost_actions.append(
                f"<b>과다 청구 확인</b>: {', '.join(high_cost[:5])} — "
                "단가가 평균 대비 2σ 이상 높음. 계약 단가 확인 및 청구서 교차 검증"
            )
        if low_cost:
            cost_actions.append(
                f"<b>미계량 의심</b>: {', '.join(low_cost[:5])} — "
                "단가가 비정상적으로 낮음. 계량기 정상 작동 여부 확인"
            )
        if not cost_actions:
            cost_actions.append("비용 단가에 특이사항이 없습니다. 다음 월에도 모니터링을 지속하세요.")
        story += _action_box(cost_actions, T, W)

    # ── Electricity breakdown section ─────────────────────────────────────
    if elec_df is not None and not elec_df.empty:
        if unit_df is not None and not unit_df.empty:
            story.append(PageBreak())
        story.append(_section_bar("전기 사용 구성 분석 (EHP / HVAC / 기저부하)", T, W))

        # Narrative
        if "hvac_pct" in elec_df.columns:
            avg_hvac = pd.to_numeric(elec_df["hvac_pct"], errors="coerce").mean()
            avg_base = pd.to_numeric(elec_df.get("base_pct", pd.Series(dtype=float)), errors="coerce").mean()
            story.append(Paragraph(
                f"전체 브랜드 평균 HVAC 비중 <b>{avg_hvac:.1f}%</b>, "
                f"기저부하 비중 <b>{avg_base:.1f}%</b>입니다. "
                "HVAC 비중이 높은 브랜드는 냉난방 시스템 효율 개선을 검토해야 합니다.",
                T["body"],
            ))
            story.append(Spacer(1, 0.3 * cm))

        # Stacked bar chart: EHP / HVAC / Base
        pct_cols = [c for c in ["ehp_pct", "hvac_pct", "base_pct"] if c in elec_df.columns]
        if pct_cols and len(elec_df) > 0:
            show_df = elec_df.nlargest(min(15, len(elec_df)),
                                        "kwh_total" if "kwh_total" in elec_df.columns else pct_cols[0])
            brands_e = show_df["brand"].tolist()
            fig, ax = plt.subplots(figsize=(10, max(3, len(brands_e) * 0.4 + 1)), facecolor="white")
            y_pos = range(len(brands_e))
            left = np.zeros(len(brands_e))
            cat_colors = {"ehp_pct": "#E63946", "hvac_pct": "#F4882A", "base_pct": "#2E6DA4"}
            cat_labels = {"ehp_pct": "EHP", "hvac_pct": "HVAC", "base_pct": "기저부하"}
            for pc in pct_cols:
                vals = pd.to_numeric(show_df[pc], errors="coerce").fillna(0).values
                ax.barh(y_pos, vals, left=left, color=cat_colors.get(pc, "#888"),
                        label=cat_labels.get(pc, pc), height=0.6, edgecolor="white")
                left += vals
            ax.set_yticks(y_pos)
            ax.set_yticklabels(brands_e, fontproperties=fp, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("비중 (%)", fontproperties=fp, fontsize=9)
            ax.set_title("전기 사용 구성 비율", fontproperties=fp, fontsize=12, fontweight="bold")
            ax.legend(prop=fp, fontsize=8, loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            chart_buf = _fig_to_buf(fig)
            story.append(_img_flowable(chart_buf, width_cm=17))
            story.append(Spacer(1, 0.3 * cm))

        # Table
        elec_cols = [c for c in ["brand", "building", "kwh_total",
                                   "ehp_pct", "hvac_pct", "base_pct",
                                   "hvac_intensity", "elect_unit_cost"] if c in elec_df.columns]
        elec_labels = {
            "brand": "브랜드", "building": "건물",
            "kwh_total": "총 전기(kWh)",
            "ehp_pct": "EHP(%)", "hvac_pct": "HVAC(%)", "base_pct": "기저(%)",
            "hvac_intensity": "HVAC 강도(kWh/m²)",
            "elect_unit_cost": "단가(₩/kWh)",
        }
        headers = [elec_labels.get(c, c) for c in elec_cols]
        n_cols = len(headers)
        col_w = [W / n_cols] * n_cols

        show = elec_df[elec_cols].sort_values(
            "kwh_total" if "kwh_total" in elec_df.columns else elec_cols[0],
            ascending=False,
        )
        rows = [headers]
        for _, r in show.iterrows():
            row = []
            for c in elec_cols:
                v = r.get(c)
                if c in ("ehp_pct", "hvac_pct", "base_pct"):
                    row.append(_f(v, 1, "%"))
                elif c == "kwh_total":
                    row.append(_f(v, 0))
                else:
                    row.append(_f(v, 2))
            rows.append(row)

        story.append(_std_table(rows, col_w, []))

        # HVAC action items
        if "hvac_intensity" in elec_df.columns:
            high_hvac = elec_df.nlargest(3, "hvac_intensity")
            hvac_actions = []
            hvac_items = [
                f"{r['brand']}({r['hvac_intensity']:.1f}kWh/m²)"
                for _, r in high_hvac.iterrows()
                if not pd.isna(r.get("hvac_intensity"))
            ]
            if hvac_items:
                hvac_actions.append(
                    f"<b>HVAC 효율 점검</b>: {', '.join(hvac_items)} — "
                    "면적 대비 HVAC 전력 사용이 높음. 냉매 충전량, 실외기 상태, "
                    "운전 스케줄 최적화 검토"
                )
            if hvac_actions:
                story.append(Spacer(1, 0.3 * cm))
                story += _action_box(hvac_actions, T, W)

    return story


_UTIL_KO = {"water": "수도", "hwater": "온수", "elect": "전기", "heat": "난방"}
_UNIT_KO = {"water": "m³/m²", "hwater": "m³/m²", "elect": "kWh/m²", "heat": "m³(MWh)/m²"}


def _efficiency_story(cur_df: pd.DataFrame, present: list[str], T, W) -> list:
    """Return reportlab flowables for the efficiency section with charts + insights."""
    story = []
    fp = _mpl_font()
    avail = {p: f"{p}_usage_per_m2" for p in present if f"{p}_usage_per_m2" in cur_df.columns}

    if not avail:
        story.append(Paragraph("전용면적 데이터가 없어 효율 분석을 생성할 수 없습니다.", T["body"]))
        return story

    # Overview narrative
    story.append(_section_bar("효율 분석 개요", T, W))
    story.append(Paragraph(
        f"총 <b>{len(avail)}개</b> 유틸리티(수도/온수/전기/난방)에 대해 "
        "단위 면적(m²)당 소비량을 산출하여 브랜드별 에너지 효율을 비교합니다. "
        "상위 20%(효율 우수)는 녹색, 하위 20%(고소비)는 빨간색으로 표시됩니다.",
        T["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    all_inefficient = []  # collect for combined action items

    for prefix, per_m2_col in avail.items():
        util_ko = _UTIL_KO.get(prefix, prefix)
        unit = _UNIT_KO.get(prefix, "unit/m²")

        df_util = cur_df[[c for c in ["brand", "building", per_m2_col]
                           if c in cur_df.columns]].dropna(subset=[per_m2_col]).copy()
        df_util[per_m2_col] = pd.to_numeric(df_util[per_m2_col], errors="coerce")
        df_util = df_util.dropna(subset=[per_m2_col]).sort_values(per_m2_col)

        if df_util.empty:
            continue

        n = len(df_util)
        top_20_thresh = max(1, n // 5)
        bottom_20_start = n - max(1, n // 5)

        # Narrative stats
        vals = df_util[per_m2_col]
        story.append(_section_bar(f"{util_ko} 효율 순위 ({unit})", T, W))
        story.append(Paragraph(
            f"<b>{util_ko}</b>: {n}개 브랜드, "
            f"평균 {vals.mean():.3f} {unit}, "
            f"중앙값 {vals.median():.3f} {unit}, "
            f"범위 {vals.min():.3f} ~ {vals.max():.3f}",
            T["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        # Chart: horizontal bar with color coding
        show_n = min(20, n)
        # Show worst + best combined
        if n > show_n:
            chart_df = pd.concat([df_util.tail(show_n // 2), df_util.head(show_n // 2)])
        else:
            chart_df = df_util

        fig, ax = plt.subplots(figsize=(8, max(3, len(chart_df) * 0.35 + 1)), facecolor="white")
        brands_c = chart_df["brand"].tolist()
        vals_c = chart_df[per_m2_col].values
        bar_colors = []
        for i, (_, r) in enumerate(chart_df.iterrows()):
            rank = df_util.index.get_loc(r.name) + 1 if r.name in df_util.index else i
            if rank <= top_20_thresh:
                bar_colors.append("#43AA6F")
            elif rank > bottom_20_start:
                bar_colors.append("#E63946")
            else:
                bar_colors.append("#2E6DA4")

        y_pos = range(len(brands_c))
        ax.barh(y_pos, vals_c, color=bar_colors, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(brands_c, fontproperties=fp, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"소비량 ({unit})", fontproperties=fp, fontsize=9)
        ax.set_title(f"{util_ko} 효율 순위", fontproperties=fp, fontsize=12, fontweight="bold")
        avg_val = vals.mean()
        ax.axvline(avg_val, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        chart_buf = _fig_to_buf(fig)
        story.append(_img_flowable(chart_buf, width_cm=16))
        story.append(Paragraph(
            "<i>🟢 상위 20% (효율 우수)  |  🔴 하위 20% (고소비, 점검 권장)  |  --- 평균</i>",
            T["caption"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        # Top/bottom narrative
        best3 = df_util.head(min(3, n))
        worst3 = df_util.tail(min(3, n))
        story.append(Paragraph(
            f"<b>효율 우수</b>: {', '.join(best3['brand'].tolist())} "
            f"(평균 {best3[per_m2_col].mean():.3f} {unit})",
            T["body"],
        ))
        story.append(Paragraph(
            f"<b>고소비</b>: {', '.join(worst3['brand'].tolist())} "
            f"(평균 {worst3[per_m2_col].mean():.3f} {unit})",
            T["body"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        # Collect inefficient brands for action items
        inefficient = df_util.tail(max(1, n // 5))
        for _, r in inefficient.iterrows():
            all_inefficient.append((r["brand"], util_ko, r[per_m2_col], unit))

        # Table
        id_cols = [c for c in ["brand", "building"] if c in df_util.columns]
        id_labels = ["브랜드", "건물"][: len(id_cols)]
        headers = id_labels + [f"소비량({unit})", "순위"]
        n_id = len(id_cols)
        id_w = [3.5 * cm, 1.2 * cm][: n_id]
        data_w = (W - sum(id_w)) / 2
        col_w = id_w + [data_w, data_w]

        rows = [headers]
        eff_styles = []
        for rank, (_, r) in enumerate(df_util.iterrows(), start=1):
            row = [str(r.get(c, "")) for c in id_cols]
            row.append(_f(r[per_m2_col], 4))
            row.append(str(rank))
            rows.append(row)
            if rank <= top_20_thresh:
                eff_styles.append(("BACKGROUND", (n_id, rank), (n_id, rank), C_STABLE))
                eff_styles.append(("TEXTCOLOR",  (n_id, rank), (n_id, rank), C_WHITE))
            elif rank > bottom_20_start:
                eff_styles.append(("BACKGROUND", (n_id, rank), (n_id, rank), C_CRITICAL))
                eff_styles.append(("TEXTCOLOR",  (n_id, rank), (n_id, rank), C_WHITE))

        story.append(_std_table(rows, col_w, eff_styles))
        story.append(Spacer(1, 0.6 * cm))

    # Combined efficiency score
    if len(avail) >= 2:
        story.append(PageBreak())
        story.append(_section_bar("종합 효율 점수 (유틸리티 순위 합산)", T, W))
        story.append(Paragraph(
            "각 유틸리티의 순위를 합산하여 종합적인 에너지 효율을 평가합니다. "
            "점수가 낮을수록 전반적으로 효율적인 브랜드입니다.",
            T["caption"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        id_cols = [c for c in ["brand", "building"] if c in cur_df.columns]
        combined = cur_df[id_cols].copy()
        combined["종합점수"] = 0
        for prefix, per_m2_col in avail.items():
            if per_m2_col in cur_df.columns:
                col_s = pd.to_numeric(cur_df[per_m2_col], errors="coerce")
                combined["종합점수"] += col_s.rank(method="min", na_option="bottom")
        combined = combined.dropna(subset=["종합점수"]).sort_values("종합점수")

        id_labels = ["브랜드", "건물"][: len(id_cols)]
        headers = id_labels + ["종합점수", "순위"]
        n_id = len(id_cols)
        id_w = [3.5 * cm, 1.2 * cm][: n_id]
        data_w = (W - sum(id_w)) / 2
        col_w = id_w + [data_w, data_w]

        rows = [headers]
        n = len(combined)
        comb_styles = []
        for rank, (_, r) in enumerate(combined.iterrows(), start=1):
            row = [str(r.get(c, "")) for c in id_cols]
            row.append(_f(r["종합점수"], 1))
            row.append(str(rank))
            rows.append(row)
            if rank <= max(1, n // 5):
                comb_styles.append(("BACKGROUND", (n_id, rank), (n_id, rank), C_STABLE))
                comb_styles.append(("TEXTCOLOR",  (n_id, rank), (n_id, rank), C_WHITE))
            elif rank > n - max(1, n // 5):
                comb_styles.append(("BACKGROUND", (n_id, rank), (n_id, rank), C_CRITICAL))
                comb_styles.append(("TEXTCOLOR",  (n_id, rank), (n_id, rank), C_WHITE))

        story.append(_std_table(rows, col_w, comb_styles))

    # Efficiency action items
    eff_actions = []
    if all_inefficient:
        # Group by brand
        brand_issues = {}
        for brand, util, val, unit in all_inefficient:
            brand_issues.setdefault(brand, []).append(f"{util}({val:.3f}{unit})")
        multi_issue = {b: issues for b, issues in brand_issues.items() if len(issues) >= 2}
        if multi_issue:
            items = [f"{b}: {', '.join(iss)}" for b, iss in list(multi_issue.items())[:3]]
            eff_actions.append(
                f"<b>복합 고소비 브랜드</b>: {'; '.join(items)} — "
                "여러 유틸리티에서 동시에 고소비. 운영 패턴 전반 점검 필요"
            )
        single_issue = {b: issues for b, issues in brand_issues.items() if len(issues) == 1}
        if single_issue:
            top_singles = list(single_issue.items())[:3]
            items = [f"{b}({iss[0]})" for b, iss in top_singles]
            eff_actions.append(
                f"<b>단일 항목 고소비</b>: {', '.join(items)} — 해당 유틸리티 설비 점검"
            )
    if not eff_actions:
        eff_actions.append("전반적으로 효율이 양호합니다. 분기별 추적을 지속하세요.")
    story += _action_box(eff_actions, T, W)

    return story


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 이상감지 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_anomaly_pdf(anomaly_df: pd.DataFrame, context: dict = None) -> bytes:
    """Generate a business-ready PDF for 이상감지 분석."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf)
    W = doc.width
    story = _cover_items("이상감지 분석 리포트", "브랜드별 복합 이상 신호 요약", context, T)
    story += _anomaly_story(anomaly_df, T, W)
    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()


# ── UI helper (shared by all biz tabs) ────────────────────────────────────────

def render_pdf_buttons(
    pdf_key: str,
    generator_fn,
    dl_label: str,
    dl_filename: str,
) -> None:
    """Generate-and-cache PDF row: [📄 생성] [⬇️ 다운로드] side by side."""
    import streamlit as _st
    c1, c2 = _st.columns([1, 2])
    with c1:
        if _st.button("📄 PDF 생성", key=f"gen_{pdf_key}"):
            with _st.spinner("PDF 생성 중…"):
                _st.session_state[pdf_key] = generator_fn()
    if pdf_key in _st.session_state:
        with c2:
            _st.download_button(
                f"⬇️ {dl_label}",
                _st.session_state[pdf_key],
                file_name=dl_filename,
                mime="application/pdf",
                key=f"dl_{pdf_key}",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 비용분석 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cross_pdf(
    unit_df: pd.DataFrame | None = None,
    elec_df: pd.DataFrame | None = None,
    context: dict = None,
) -> bytes:
    """Generate a business-ready PDF for 비용분석 (unit costs + electricity breakdown)."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf)
    W = doc.width
    story = _cover_items("비용 분석 리포트", "단위 비용 이상 감지 및 전기 사용 구성 분석", context, T)
    sections = _cross_story(unit_df, elec_df, T, W)
    if sections:
        story += sections
    else:
        story.append(Paragraph("분석 데이터를 불러올 수 없습니다.", T["body"]))
    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 효율분석 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_efficiency_pdf(
    cur_df: pd.DataFrame,
    present: list[str],
    context: dict = None,
) -> bytes:
    """Generate a business-ready PDF for 효율분석."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf)
    W = doc.width
    story = _cover_items("효율 분석 리포트", "브랜드별 단위 면적당 에너지 소비 효율 순위", context, T)
    story += _efficiency_story(cur_df, present, T, W)
    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 종합 리포트 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comprehensive_pdf(
    anomaly_df: pd.DataFrame | None = None,
    unit_df: pd.DataFrame | None = None,
    elec_br_df: pd.DataFrame | None = None,
    cur_df: pd.DataFrame | None = None,
    present: list[str] | None = None,
    context: dict = None,
) -> bytes:
    """Generate a single comprehensive PDF combining all analysis sections."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf, footer_left="종합 분석 보고서  ·  대외비")
    W = doc.width

    story = _cover_items(
        "종합 분석 리포트",
        "이상감지 · 비용분석 · 효율분석 통합 보고서",
        context, T,
    )

    # Table of contents summary
    sections_included = []
    if anomaly_df is not None and not anomaly_df.empty:
        sections_included.append("이상감지 분석")
    if (unit_df is not None and not unit_df.empty) or (elec_br_df is not None and not elec_br_df.empty):
        sections_included.append("비용 분석")
    if cur_df is not None and present:
        sections_included.append("효율 분석")

    if sections_included:
        toc_text = "포함 섹션: " + " → ".join(sections_included)
        story.append(Paragraph(toc_text, T["caption"]))
        story.append(Spacer(1, 0.5 * cm))

    has_content = False

    # Section 1: Anomaly
    if anomaly_df is not None and not anomaly_df.empty:
        has_content = True
        story += _anomaly_story(anomaly_df, T, W)

    # Section 2: Cost analysis
    cross = _cross_story(unit_df, elec_br_df, T, W)
    if cross:
        has_content = True
        story.append(PageBreak())
        story.append(_section_bar("── 비용 분석 ──", T, W))
        story.append(Spacer(1, 0.3 * cm))
        story += cross

    # Section 3: Efficiency
    if cur_df is not None and present:
        eff = _efficiency_story(cur_df, present, T, W)
        if eff:
            has_content = True
            story.append(PageBreak())
            story.append(_section_bar("── 효율 분석 ──", T, W))
            story.append(Spacer(1, 0.3 * cm))
            story += eff

    if not has_content:
        story.append(Paragraph("분석 데이터를 불러올 수 없습니다.", T["body"]))

    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()


def generate_insight_pdf(
    unit_df: pd.DataFrame | None = None,
    elec_br_df: pd.DataFrame | None = None,
    cur_df: pd.DataFrame | None = None,
    present: list[str] | None = None,
    context: dict = None,
) -> bytes:
    """Generate a combined insight PDF (cost + efficiency)."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf, footer_left="인사이트 분석 보고서  ·  대외비")
    W = doc.width

    story = _cover_items(
        "인사이트 분석 리포트",
        "비용 분석 · 효율 분석 통합",
        context, T,
    )

    has_content = False

    cross = _cross_story(unit_df, elec_br_df, T, W)
    if cross:
        has_content = True
        story += cross

    if cur_df is not None and present:
        eff = _efficiency_story(cur_df, present, T, W)
        if eff:
            has_content = True
            if cross:
                story.append(PageBreak())
            story += eff

    if not has_content:
        story.append(Paragraph("분석 데이터를 불러올 수 없습니다.", T["body"]))

    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()

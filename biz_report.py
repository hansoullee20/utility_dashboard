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


def _pct(val):
    """Format a value as percentage string, handling None/NaN."""
    if val is None:
        return "—"
    try:
        v = float(val)
        return "—" if np.isnan(v) else f"{v:+.1f}%"
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


def _highlight_table(data, col_w, styles):
    """Table with slightly larger font and more padding — for key summary tables."""
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "NanumGothic-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("FONTNAME",      (0, 1), (-1, -1), "NanumGothic"),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
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


def _prose(text, T):
    """Narrative paragraph with bottom spacing — the primary building block."""
    return [Paragraph(text, T["body"]), Spacer(1, 0.2 * cm)]


def _action_box(items: list[str], T, W) -> list:
    """Create a highlighted action-items box with flowing recommendations."""
    flowables = [_section_bar("📋 조치 권고사항", T, W)]
    for i, item in enumerate(items, 1):
        flowables.append(Paragraph(f"<b>{i}.</b> {item}", T["body"]))
        flowables.append(Spacer(1, 0.2 * cm))
    flowables.append(Spacer(1, 0.4 * cm))
    return flowables


def _divider_line(W) -> list:
    """Thin horizontal rule for visual separation between subsections."""
    t = Table([[""]],
              colWidths=[W],
              style=TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.5, C_DIVIDER)]))
    return [Spacer(1, 0.2 * cm), t, Spacer(1, 0.3 * cm)]


# ═══════════════════════════════════════════════════════════════════════════════
# Story builders — reusable flowable lists (no doc/cover)
# ═══════════════════════════════════════════════════════════════════════════════

def _anomaly_story(anomaly_df: pd.DataFrame, T, W) -> list:
    """Return reportlab flowables for the anomaly section — executive narrative style."""
    story = []
    fp = _mpl_font()
    total = len(anomaly_df)
    risk_counts = anomaly_df["risk_level"].value_counts().to_dict() if "risk_level" in anomaly_df.columns else {}

    danger  = risk_counts.get("🔴 위험", 0)
    caution = risk_counts.get("🟠 주의", 0)
    observe = risk_counts.get("🟡 관찰", 0)
    normal  = risk_counts.get("🟢 정상", 0)
    flagged = danger + caution
    pct_flagged = flagged / total * 100 if total else 0

    # ── Executive Summary ──────────────────────────────────────────────────
    story.append(_section_bar("경영진 요약", T, W))

    exec_text = (
        f"금월 검침 데이터를 기반으로 전체 <b>{total}개</b> 브랜드에 대해 "
        f"사용량 급등, 단위 비용 이상, 소비 패턴, HVAC 효율, 계량 일관성 등 "
        f"5개 차원의 복합 이상 분석을 수행한 결과, "
        f"<b>{flagged}개 브랜드({pct_flagged:.0f}%)</b>가 위험 또는 주의 등급으로 분류되었습니다."
    )
    story += _prose(exec_text, T)

    if danger:
        story += _prose(
            f"이 중 <font color='#E63946'><b>위험 등급 {danger}개</b></font> 브랜드는 "
            f"복합 이상 점수 0.65 이상으로, 검침 오류·계량기 고장·누수 등의 가능성이 높아 "
            f"<b>즉시 현장 점검이 필요</b>합니다. "
            f"이 브랜드들은 여러 유틸리티에서 동시에 비정상적 신호가 감지되고 있어, "
            f"단순 사용량 증가가 아닌 시스템적 문제일 수 있습니다.", T,
        )
    if caution:
        story += _prose(
            f"<font color='#F4882A'><b>주의 등급 {caution}개</b></font> 브랜드는 "
            f"하나 이상의 유틸리티에서 비정상적 패턴이 감지되었으나, 일시적 변동일 가능성도 있습니다. "
            f"다음 월 검침 결과와 비교하여 추세가 지속되는지 모니터링이 필요합니다.", T,
        )
    if observe > 0 and normal > 0:
        story += _prose(
            f"나머지 <b>{observe}개</b> 관찰 등급과 <b>{normal}개</b> 정상 등급 브랜드는 "
            f"현재 특이사항이 없으나, 정기 모니터링을 통해 조기 경보 체계를 유지하는 것이 바람직합니다.", T,
        )
    story.append(Spacer(1, 0.2 * cm))

    # ── KPI summary row ────────────────────────────────────────────────────
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

    # ── Risk distribution chart ────────────────────────────────────────────
    labels = ["위험", "주의", "관찰", "정상"]
    values = [danger, caution, observe, normal]
    mpl_colors = ["#E63946", "#F4882A", "#E8B84B", "#43AA6F"]
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, mpl_colors) if v > 0]

    if non_zero:
        fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="white")

        pie_labels, pie_vals, pie_colors = zip(*non_zero)
        wedges, texts, autotexts = ax_pie.pie(
            pie_vals, labels=pie_labels, colors=pie_colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontproperties": fp, "fontsize": 10},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax_pie.set_title("위험 등급 분포", fontproperties=fp, fontsize=12, fontweight="bold")

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
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(
            "<i>좌: 위험 등급 비율 | 우: 복합 이상 점수 상위 10개 브랜드 "
            "(빨간 점선 = 위험 기준 0.65, 주황 점선 = 주의 기준 0.40)</i>",
            T["caption"],
        ))
        story.append(Spacer(1, 0.4 * cm))

    # ── High-risk brand deep-dive ──────────────────────────────────────────
    high_risk = anomaly_df[
        anomaly_df.get("risk_level", pd.Series(dtype=str)).isin(["🔴 위험", "🟠 주의"])
    ].copy() if "risk_level" in anomaly_df.columns else pd.DataFrame()

    if not high_risk.empty:
        story.append(_section_bar(f"주요 점검 대상 브랜드 분석 ({len(high_risk)}개)", T, W))

        story += _prose(
            f"아래는 위험·주의 등급으로 분류된 <b>{len(high_risk)}개</b> 브랜드에 대한 상세 분석입니다. "
            f"각 브랜드의 이상 원인을 파악하여 우선순위별로 조사를 진행하시기 바랍니다.", T,
        )

        # Detailed narrative for top brands
        top5 = high_risk.nlargest(min(5, len(high_risk)), "composite_score")
        for idx, (_, r) in enumerate(top5.iterrows(), 1):
            brand = r.get("brand", "?")
            bldg = r.get("building", "")
            score = r.get("composite_score", 0)
            reason = r.get("reason", "")
            rl = _RISK_PLAIN.get(str(r.get("risk_level", "")), "")
            loc = f" ({bldg}동)" if bldg else ""

            # Build a flowing narrative per brand
            spike_pct = r.get("spike_max_pct", 0)
            spike_util = r.get("spike_worst_util", "")
            peer_ratio = r.get("spike_peer_ratio", 0)
            parts = [f"<b>{idx}. {brand}</b>{loc} — <b>{rl}</b> (복합점수 {score:.2f})"]

            details = []
            if spike_pct and not pd.isna(spike_pct) and abs(spike_pct) > 10:
                details.append(
                    f"전월 대비 {spike_util} 사용량이 <b>{spike_pct:+.0f}%</b> 변동하였습니다"
                )
            if peer_ratio and not pd.isna(peer_ratio) and peer_ratio > 1.5:
                details.append(
                    f"동일 건물 평균 대비 <b>{peer_ratio:.1f}배</b> 높은 변동폭을 보입니다"
                )
            if reason and reason != "—":
                details.append(f"주요 이상 신호: {reason}")

            if details:
                parts.append(". ".join(details) + ".")
            else:
                parts.append("복합 점수가 높아 전반적인 유틸리티 사용 패턴 점검이 필요합니다.")

            story.append(Paragraph(" ".join(parts), T["body"]))
            story.append(Spacer(1, 0.15 * cm))

        if len(high_risk) > 5:
            story += _prose(
                f"<i>이상 {len(high_risk) - 5}개 브랜드는 아래 요약 테이블을 참조하시기 바랍니다.</i>", T,
            )
        story.append(Spacer(1, 0.2 * cm))

        # Summary table — compact, key columns only
        high_risk = high_risk.sort_values("composite_score", ascending=False)
        headers = ["브랜드", "건물", "등급", "복합점수", "최대급등", "주요항목", "이유"]
        col_w = [c * cm for c in [2.5, 0.9, 0.9, 1.2, 1.2, 1.3, 9.0]]

        rows = [headers]
        row_styles = []
        for i, (_, r) in enumerate(high_risk.head(15).iterrows(), start=1):
            rl = str(r.get("risk_level", ""))
            reason_text = str(r.get("reason", "—"))
            if len(reason_text) > 60:
                reason_text = reason_text[:57] + "…"
            rows.append([
                str(r.get("brand", "")),
                str(r.get("building", "—")),
                _RISK_PLAIN.get(rl, rl),
                _f(r.get("composite_score"), 3),
                _f(r.get("spike_max_pct"), 0, "%"),
                str(r.get("spike_worst_util", "—")),
                reason_text,
            ])
            c = _RISK_COLOR_RL.get(rl)
            if c:
                row_styles.append(("BACKGROUND", (2, i), (2, i), c))
                row_styles.append(("TEXTCOLOR",  (2, i), (2, i), C_WHITE))

        story.append(KeepTogether([_std_table(rows, col_w, row_styles)]))
        story.append(Spacer(1, 0.5 * cm))

    # ── Action plan ────────────────────────────────────────────────────────
    actions = []
    if danger:
        danger_brands = high_risk[
            high_risk.get("risk_level", pd.Series(dtype=str)) == "🔴 위험"
        ]["brand"].tolist()[:5]
        actions.append(
            f"<b>즉시 현장 점검</b> — 위험 등급 브랜드({', '.join(danger_brands)})에 대해 "
            "검침 데이터와 실제 계량기 수치를 교차 확인하고, 누수·계량기 오작동·배관 이상 등을 "
            "점검하십시오. 특히 복수 유틸리티에서 동시 이상이 감지된 경우 설비 전반에 대한 "
            "종합 점검이 필요합니다."
        )
    if caution:
        actions.append(
            f"<b>주의 브랜드 모니터링 강화</b> — 주의 등급 {caution}개 브랜드의 다음 월 검침 결과를 "
            "면밀히 추적하여 일시적 변동인지 지속적 이상 추세인지 판별하십시오. "
            "2개월 연속 주의 등급 유지 시 현장 점검으로 전환을 권고합니다."
        )

    if "spike_max_pct" in anomaly_df.columns:
        big_spikes = anomaly_df[anomaly_df["spike_max_pct"].fillna(0) > 50]
        if not big_spikes.empty:
            spike_brands = big_spikes.nlargest(3, "spike_max_pct")
            spike_items = [
                f"{r['brand']}({r['spike_max_pct']:+.0f}%)"
                for _, r in spike_brands.iterrows()
            ]
            actions.append(
                f"<b>급등 원인 조사</b> — 전월 대비 50% 이상 급등한 {', '.join(spike_items)} 등 "
                f"총 {len(big_spikes)}건에 대해 계량기 오작동, 누수(수도), "
                "설비 추가 가동(전기·난방), 또는 입주사 변경 등 외부 요인을 확인하십시오."
            )

    if "consistency_score" in anomaly_df.columns:
        inconsistent = anomaly_df[anomaly_df["consistency_score"].fillna(0) > 0.5]
        if not inconsistent.empty:
            actions.append(
                f"<b>검침 프로세스 검증</b> — {len(inconsistent)}개 브랜드에서 시트 간 "
                "데이터 불일치 또는 미계량 항목이 발견되었습니다. 검침 절차를 재확인하고, "
                "집계 시트와 개별 유틸리티 시트 간 데이터 정합성을 점검하십시오."
            )

    if not actions:
        actions.append(
            "현재 심각한 이상이 감지되지 않았습니다. 월별 정기 모니터링을 지속하여 "
            "조기 경보 체계를 유지하시기 바랍니다."
        )
    story += _action_box(actions, T, W)

    # ── Methodology note ───────────────────────────────────────────────────
    story += _divider_line(W)
    story.append(Paragraph(
        "<i><b>분석 방법론</b>: 복합 이상 점수는 급등(30%), 소비 패턴(25%), 비용 이상(25%), "
        "HVAC 효율(10%), 계량 일관성(10%)의 가중 합산으로 산출됩니다. "
        "위험 ≥ 0.65, 주의 ≥ 0.40, 관찰 ≥ 0.20, 정상 &lt; 0.20 기준이 적용되었습니다.</i>",
        T["caption"],
    ))

    return story


def _cross_story(unit_df, elec_df, T, W) -> list:
    """Return reportlab flowables for the cost analysis section — narrative-driven."""
    story = []
    fp = _mpl_font()

    # ── Unit cost analysis ─────────────────────────────────────────────────
    if unit_df is not None and not unit_df.empty:
        story.append(_section_bar("단위 비용 분석", T, W))

        # Compute stats for narrative
        cost_stats = {}
        for col, label, unit in [
            ("water_unit_cost", "수도", "₩/m³"),
            ("elect_unit_cost", "전기", "₩/kWh"),
        ]:
            if col in unit_df.columns:
                vals = pd.to_numeric(unit_df[col], errors="coerce").dropna()
                if not vals.empty:
                    cost_stats[label] = {
                        "unit": unit, "mean": vals.mean(), "std": vals.std(),
                        "median": vals.median(), "min": vals.min(), "max": vals.max(),
                        "cv": vals.std() / vals.mean() * 100 if vals.mean() > 0 else 0,
                    }

        # Opening narrative
        if cost_stats:
            parts = []
            for label, s in cost_stats.items():
                parts.append(
                    f"{label} 단가는 평균 <b>{s['mean']:,.0f} {s['unit']}</b> "
                    f"(중앙값 {s['median']:,.0f}), 범위 {s['min']:,.0f}~{s['max']:,.0f}"
                )
            story += _prose(
                "브랜드별 유틸리티 단가를 비교 분석한 결과, " + ", ".join(parts) + "으로 나타났습니다. "
                "변동계수(CV)가 높은 항목은 동일 건물 내 브랜드 간 단가 차이가 크다는 의미로, "
                "계약 조건 차이 또는 계량 이상의 가능성을 시사합니다.", T,
            )

        story += _prose(
            "아래 차트에서 <font color='#E63946'><b>빨간색</b></font>으로 표시된 브랜드는 "
            "Z-점수 절대값이 2.0 이상으로, 동종 대비 통계적으로 유의미한 비용 이상이 감지된 "
            "대상입니다. 양의 Z-점수는 과다 청구 가능성, 음의 Z-점수는 미계량 또는 누락 가능성을 "
            "각각 시사합니다.", T,
        )

        # Chart: unit cost distribution
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

                order = np.argsort(v)[::-1]
                v, b, z = v[order], [b[i] for i in order], z[order]
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
                "<i>🔴 빨간색 = |Z| ≥ 2.0 (이상 감지) | 🔵 파란색 = 정상 범위 | 점선 = 전체 평균</i>",
                T["caption"],
            ))
            story.append(Spacer(1, 0.4 * cm))

        # Anomaly narrative + compact table
        anom_brands = []
        for col, label in [
            ("water_unit_z", "수도 단가"),
            ("elect_unit_z", "전기 단가"),
            ("total_cost_per_py_z", "평당 비용"),
            ("total_cost_per_m2_z", "총비용/m²"),
        ]:
            if col in unit_df.columns:
                flags = unit_df[unit_df[col].abs() >= 2.0]
                for _, r in flags.iterrows():
                    brand = str(r.get("brand", ""))
                    z_val = float(r.get(col, 0)) if not pd.isna(r.get(col)) else 0
                    direction = "고비용" if z_val > 0 else "저비용"
                    anom_brands.append((brand, label, z_val, direction))

        if anom_brands:
            story.append(_section_bar(f"비용 이상 브랜드 ({len(anom_brands)}건)", T, W))

            # Flowing narrative instead of listing
            high_cost = [(b, m, z) for b, m, z, d in anom_brands if z > 0]
            low_cost = [(b, m, z) for b, m, z, d in anom_brands if z < 0]

            if high_cost:
                hc_text_parts = []
                for b, m, z in high_cost[:3]:
                    hc_text_parts.append(f"<b>{b}</b>({m} Z={z:+.1f})")
                story += _prose(
                    f"<font color='#E63946'><b>고비용 이상</b></font>: "
                    f"{', '.join(hc_text_parts)} 등 {len(high_cost)}건이 감지되었습니다. "
                    "이들 브랜드는 동일 건물 내 동종 대비 단가가 2σ 이상 높으며, "
                    "계약 단가 차이가 아닌 경우 과다 청구 또는 계량 이상을 의심할 수 있습니다. "
                    "해당 브랜드의 청구서 원본과 계량 데이터를 교차 검증하시기 바랍니다.", T,
                )

            if low_cost:
                lc_text_parts = []
                for b, m, z in low_cost[:3]:
                    lc_text_parts.append(f"<b>{b}</b>({m} Z={z:+.1f})")
                story += _prose(
                    f"<font color='#2E6DA4'><b>저비용 이상</b></font>: "
                    f"{', '.join(lc_text_parts)} 등 {len(low_cost)}건이 감지되었습니다. "
                    "비정상적으로 낮은 단가는 계량기 미작동, 검침 누락, 또는 집계 오류의 "
                    "가능성을 시사합니다.", T,
                )

        # Cost action items
        cost_actions = []
        high_cost_brands = [b for b, m, z, d in anom_brands if z > 2.0]
        low_cost_brands = [b for b, m, z, d in anom_brands if z < -2.0]
        if high_cost_brands:
            cost_actions.append(
                f"<b>청구서 교차 검증</b> — {', '.join(high_cost_brands[:5])} 브랜드의 "
                "계약 단가와 실제 청구 단가를 비교하고, 계량기 검정 기록을 확인하십시오. "
                "계약 단가가 동일함에도 단가 차이가 발생한다면 검침 데이터 오류 가능성이 높습니다."
            )
        if low_cost_brands:
            cost_actions.append(
                f"<b>계량기 정상 작동 확인</b> — {', '.join(low_cost_brands[:5])} 브랜드의 "
                "계량기 현장 점검을 실시하고, 최근 검정 이력을 확인하십시오."
            )
        if not cost_actions:
            cost_actions.append(
                "현재 비용 단가에 통계적으로 유의미한 이상이 없습니다. "
                "분기별 추세 분석을 통해 장기적 단가 변동을 모니터링하시기 바랍니다."
            )
        story += _action_box(cost_actions, T, W)

    # ── Electricity breakdown ──────────────────────────────────────────────
    if elec_df is not None and not elec_df.empty:
        if unit_df is not None and not unit_df.empty:
            story.append(PageBreak())
        story.append(_section_bar("전기 사용 구성 분석", T, W))

        # Narrative overview
        if "hvac_pct" in elec_df.columns:
            avg_hvac = pd.to_numeric(elec_df["hvac_pct"], errors="coerce").mean()
            avg_base = pd.to_numeric(elec_df.get("base_pct", pd.Series(dtype=float)), errors="coerce").mean()
            avg_ehp = pd.to_numeric(elec_df.get("ehp_pct", pd.Series(dtype=float)), errors="coerce").mean()

            story += _prose(
                f"전체 브랜드의 전기 사용 구성을 분석한 결과, 평균적으로 "
                f"EHP(개별 냉난방) <b>{avg_ehp:.1f}%</b>, "
                f"HVAC(중앙 냉난방) <b>{avg_hvac:.1f}%</b>, "
                f"기저부하(조명·기기) <b>{avg_base:.1f}%</b>의 비율을 보였습니다.", T,
            )

            # Find outliers
            if "hvac_intensity" in elec_df.columns:
                hvac_vals = pd.to_numeric(elec_df["hvac_intensity"], errors="coerce").dropna()
                if not hvac_vals.empty:
                    hvac_mean = hvac_vals.mean()
                    hvac_std = hvac_vals.std()
                    high_hvac = elec_df[
                        pd.to_numeric(elec_df["hvac_intensity"], errors="coerce") > hvac_mean + 1.5 * hvac_std
                    ]
                    if not high_hvac.empty:
                        hh_names = high_hvac["brand"].tolist()[:3]
                        story += _prose(
                            f"특히 {', '.join(hh_names)} 등 <b>{len(high_hvac)}개</b> 브랜드는 "
                            f"HVAC 강도(kWh/m²)가 평균+1.5σ를 초과하여, 냉난방 시스템의 "
                            "효율 저하 또는 과도한 운전이 의심됩니다. 이들 브랜드는 "
                            "실외기 상태, 냉매 충전량, 운전 스케줄을 우선 점검하시기 바랍니다.", T,
                        )

        # Stacked bar chart
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
            ax.set_title("전기 사용 구성 비율 (총 전력량 상위 15개)", fontproperties=fp, fontsize=12, fontweight="bold")
            ax.legend(prop=fp, fontsize=8, loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            chart_buf = _fig_to_buf(fig)
            story.append(_img_flowable(chart_buf, width_cm=17))
            story.append(Spacer(1, 0.3 * cm))

        # HVAC action items
        if "hvac_intensity" in elec_df.columns:
            high_hvac_df = elec_df.nlargest(3, "hvac_intensity")
            hvac_items = [
                f"{r['brand']}({r['hvac_intensity']:.1f}kWh/m²)"
                for _, r in high_hvac_df.iterrows()
                if not pd.isna(r.get("hvac_intensity"))
            ]
            if hvac_items:
                story += _action_box([
                    f"<b>HVAC 효율 개선</b> — {', '.join(hvac_items)} 브랜드는 면적 대비 HVAC "
                    "전력 사용이 상위권입니다. 냉매 충전량 확인, 실외기 청소 상태, "
                    "운전 스케줄 최적화(야간·주말 감량 운전) 등을 종합적으로 검토하여 "
                    "에너지 비용 절감 기회를 모색하시기 바랍니다."
                ], T, W)

    return story


_UTIL_KO = {"water": "수도", "hwater": "온수", "elect": "전기", "heat": "난방"}
_UNIT_KO = {"water": "m³/m²", "hwater": "m³/m²", "elect": "kWh/m²", "heat": "m³(MWh)/m²"}


def _efficiency_story(cur_df: pd.DataFrame, present: list[str], T, W) -> list:
    """Return reportlab flowables for the efficiency section — narrative-driven."""
    story = []
    fp = _mpl_font()
    avail = {p: f"{p}_usage_per_m2" for p in present if f"{p}_usage_per_m2" in cur_df.columns}

    if not avail:
        story.append(Paragraph("전용면적 데이터가 없어 효율 분석을 생성할 수 없습니다.", T["body"]))
        return story

    # Opening narrative
    story.append(_section_bar("효율 분석 개요", T, W))
    story += _prose(
        f"총 <b>{len(avail)}개</b> 유틸리티({', '.join(_UTIL_KO.get(p, p) for p in avail)})에 대해 "
        "단위 면적(m²)당 소비량을 산출하여 브랜드별 에너지 효율을 비교 분석하였습니다. "
        "면적당 소비량은 브랜드의 규모 차이를 보정하여 실질적인 에너지 사용 강도를 비교할 수 있는 "
        "핵심 지표입니다. 상위 20%(효율 우수)는 녹색, 하위 20%(고소비)는 빨간색으로 구분됩니다.", T,
    )

    all_inefficient = []

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
        vals = df_util[per_m2_col]

        story.append(_section_bar(f"{util_ko} 효율 분석 ({unit})", T, W))

        # Rich narrative per utility
        best3 = df_util.head(min(3, n))
        worst3 = df_util.tail(min(3, n))
        spread_ratio = vals.max() / vals.min() if vals.min() > 0 else 0

        story += _prose(
            f"<b>{util_ko}</b> 효율 분석 대상 {n}개 브랜드의 면적당 소비량은 "
            f"평균 <b>{vals.mean():.3f} {unit}</b>(중앙값 {vals.median():.3f})이며, "
            f"최소 {vals.min():.3f}에서 최대 {vals.max():.3f}까지 "
            f"<b>{spread_ratio:.1f}배</b>의 편차를 보였습니다.", T,
        )

        story += _prose(
            f"가장 효율적인 브랜드는 <b>{', '.join(best3['brand'].tolist())}</b>"
            f"(평균 {best3[per_m2_col].mean():.3f} {unit})이며, "
            f"고소비 브랜드는 <b>{', '.join(worst3['brand'].tolist())}</b>"
            f"(평균 {worst3[per_m2_col].mean():.3f} {unit})입니다. "
            f"고소비 브랜드는 업종 특성(예: 음식점, 세탁소)을 고려하더라도 "
            f"설비 효율 점검의 우선 대상입니다.", T,
        )

        # Chart
        show_n = min(20, n)
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
        ax.set_title(f"{util_ko} 면적당 소비 효율", fontproperties=fp, fontsize=12, fontweight="bold")
        avg_val = vals.mean()
        ax.axvline(avg_val, color="#555", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        chart_buf = _fig_to_buf(fig)
        story.append(_img_flowable(chart_buf, width_cm=16))
        story.append(Paragraph(
            "<i>🟢 상위 20% (효율 우수) | 🔴 하위 20% (고소비) | 점선 = 전체 평균</i>",
            T["caption"],
        ))
        story.append(Spacer(1, 0.4 * cm))

        # Collect inefficient brands
        inefficient = df_util.tail(max(1, n // 5))
        for _, r in inefficient.iterrows():
            all_inefficient.append((r["brand"], util_ko, r[per_m2_col], unit))

    # Combined efficiency score
    if len(avail) >= 2:
        story.append(PageBreak())
        story.append(_section_bar("종합 효율 점수", T, W))
        story += _prose(
            "각 유틸리티의 효율 순위를 합산하여 종합적인 에너지 효율을 평가하였습니다. "
            "종합점수가 낮을수록 전반적으로 효율적인 브랜드이며, 여러 유틸리티에서 고르게 "
            "효율적인 운영을 하고 있음을 의미합니다.", T,
        )

        id_cols = [c for c in ["brand", "building"] if c in cur_df.columns]
        combined = cur_df[id_cols].copy()
        combined["종합점수"] = 0
        for prefix, per_m2_col in avail.items():
            if per_m2_col in cur_df.columns:
                col_s = pd.to_numeric(cur_df[per_m2_col], errors="coerce")
                combined["종합점수"] += col_s.rank(method="min", na_option="bottom")
        combined = combined.dropna(subset=["종합점수"]).sort_values("종합점수")

        # Narrative for top/bottom
        if len(combined) >= 3:
            top3 = combined.head(3)["brand"].tolist()
            bot3 = combined.tail(3)["brand"].tolist()
            story += _prose(
                f"종합 효율 상위 브랜드는 <b>{', '.join(top3)}</b>로, 모든 유틸리티에서 "
                f"균형 잡힌 효율적 운영을 보여주고 있습니다. 반면 <b>{', '.join(bot3)}</b>는 "
                f"종합 순위 하위권으로, 복수 유틸리티에서 동시에 고소비 패턴이 관찰되어 "
                f"운영 전반에 대한 점검이 필요합니다.", T,
            )

        # Top 15 table only (not all)
        id_labels = ["브랜드", "건물"][: len(id_cols)]
        headers = id_labels + ["종합점수", "순위"]
        n_id = len(id_cols)
        id_w = [3.5 * cm, 1.2 * cm][: n_id]
        data_w = (W - sum(id_w)) / 2
        col_w = id_w + [data_w, data_w]

        rows = [headers]
        n = len(combined)
        comb_styles = []
        for rank, (_, r) in enumerate(combined.head(15).iterrows(), start=1):
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

        story.append(_highlight_table(rows, col_w, comb_styles))
        if n > 15:
            story.append(Paragraph(
                f"<i>상위 15개 브랜드 표시 (전체 {n}개 중)</i>",
                T["caption"],
            ))
        story.append(Spacer(1, 0.4 * cm))

    # Efficiency action items
    eff_actions = []
    if all_inefficient:
        brand_issues = {}
        for brand, util, val, unit in all_inefficient:
            brand_issues.setdefault(brand, []).append(f"{util}({val:.3f}{unit})")
        multi_issue = {b: issues for b, issues in brand_issues.items() if len(issues) >= 2}
        if multi_issue:
            items = [f"{b}({', '.join(iss)})" for b, iss in list(multi_issue.items())[:3]]
            eff_actions.append(
                f"<b>복합 고소비 브랜드 종합 점검</b> — {'; '.join(items)} 등은 "
                "여러 유틸리티에서 동시에 하위 20%에 해당합니다. 업종 특성을 감안하더라도 "
                "설비 노후화, 운영 비효율, 또는 계량 이상의 가능성을 종합적으로 검토하십시오."
            )
        single_issue = {b: issues for b, issues in brand_issues.items() if len(issues) == 1}
        if single_issue:
            top_singles = list(single_issue.items())[:3]
            items = [f"{b}({iss[0]})" for b, iss in top_singles]
            eff_actions.append(
                f"<b>단일 항목 고소비</b> — {', '.join(items)} 브랜드는 특정 유틸리티에서 "
                "효율이 낮습니다. 해당 설비의 정비 상태 및 운전 조건을 확인하시기 바랍니다."
            )
    if not eff_actions:
        eff_actions.append(
            "전반적으로 브랜드 간 효율 편차가 크지 않으며 양호한 수준입니다. "
            "분기별 추적을 통해 효율 변화 추세를 지속 모니터링하시기 바랍니다."
        )
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

    # Executive brief on cover page
    sections_included = []
    brief_parts = []
    if anomaly_df is not None and not anomaly_df.empty:
        sections_included.append("이상감지 분석")
        total = len(anomaly_df)
        risk_counts = anomaly_df["risk_level"].value_counts().to_dict() if "risk_level" in anomaly_df.columns else {}
        danger = risk_counts.get("🔴 위험", 0)
        caution = risk_counts.get("🟠 주의", 0)
        brief_parts.append(
            f"전체 {total}개 브랜드 중 위험 {danger}개, 주의 {caution}개"
        )
    if (unit_df is not None and not unit_df.empty) or (elec_br_df is not None and not elec_br_df.empty):
        sections_included.append("비용 분석")
    if cur_df is not None and present:
        sections_included.append("효율 분석")

    if sections_included:
        story += _prose(
            f"<b>보고서 구성</b>: {' → '.join(sections_included)}", T,
        )
        if brief_parts:
            story += _prose(
                f"<b>핵심 요약</b>: {'. '.join(brief_parts)}.", T,
            )
        story.append(Spacer(1, 0.3 * cm))

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
    doc, T = _build_doc(buf, footer_left="비용·효율 분석 보고서  ·  대외비")
    W = doc.width

    story = _cover_items(
        "비용·효율 분석 리포트",
        "단위 비용 이상 감지 및 에너지 효율 분석",
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

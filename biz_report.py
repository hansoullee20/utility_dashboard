"""
biz_report.py — PDF report generators for 비즈니스 분析 horizontal tabs:
  - generate_anomaly_pdf()   : 이상감지
  - generate_cross_pdf()     : 비용분析
  - generate_efficiency_pdf(): 효율분析
"""
from __future__ import annotations

import io
from datetime import date as _date

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, KeepTogether, PageBreak,
    Paragraph, Spacer, Table, TableStyle,
)

from report import (
    C_BLUE, C_CRITICAL, C_DIVIDER, C_LIGHT, C_NAVY, C_WHITE,
    C_WATCH, C_ALERT, C_STABLE, C_NORMAL,
    _ensure_fonts, _make_numbered_canvas, _make_page_template,
    _make_styles, _section_bar,
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


def _build_doc(buf):
    _ensure_fonts()
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=2.0 * cm,  bottomMargin=2.0 * cm,
    )
    T = _make_styles()
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
        items.append(Paragraph(f"분析 기간: {ctx['period']}", T["note"]))
    if ctx.get("buildings"):
        items.append(Paragraph(f"대상 건물: {ctx['buildings']}", T["note"]))
    items.append(Spacer(1, 0.8 * cm))
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 이상감지 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_anomaly_pdf(anomaly_df: pd.DataFrame, context: dict = None) -> bytes:
    """Generate a business-ready PDF for 이상감지 분析."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf)
    W = doc.width

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story += _cover_items("이상감지 분析 리포트", "브랜드별 복합 이상 신호 요약", context, T)

    # ── Risk level summary ────────────────────────────────────────────────────
    story.append(_section_bar("위험 등급 요약", T, W))
    total = len(anomaly_df)
    risk_counts = {}
    if "risk_level" in anomaly_df.columns:
        risk_counts = anomaly_df["risk_level"].value_counts().to_dict()

    danger  = risk_counts.get("🔴 위험", 0)
    caution = risk_counts.get("🟠 주의", 0)
    observe = risk_counts.get("🟡 관찰", 0)
    normal  = risk_counts.get("🟢 정상", 0)

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

    # ── Score methodology note ────────────────────────────────────────────────
    story.append(Paragraph(
        "복합 점수 = 급등(30%) + 소비(25%) + 비용(25%) + HVAC(10%) + 일관성(10%)  |  "
        "위험 ≥ 0.65  ·  주의 ≥ 0.40  ·  관찰 ≥ 0.20  ·  정상 &lt; 0.20",
        T["caption"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── High-risk brands table ────────────────────────────────────────────────
    high_risk = anomaly_df[
        anomaly_df.get("risk_level", pd.Series(dtype=str)).isin(["🔴 위험", "🟠 주의"])
    ].copy() if "risk_level" in anomaly_df.columns else pd.DataFrame()

    if not high_risk.empty:
        story.append(_section_bar(f"위험·주의 브랜드 상세 ({len(high_risk)}개)", T, W))
        high_risk = high_risk.sort_values("composite_score", ascending=False)

        headers = ["브랜드", "건물", "등급", "복합점수", "급등%", "급등항목", "소비", "비용", "HVAC", "일관성"]
        col_w = [3.0, 1.0, 1.1, 1.3, 1.2, 1.5, 1.1, 1.1, 1.1, 1.1]
        col_w = [c * cm for c in col_w]

        rows = [headers]
        row_styles = []
        for i, (_, r) in enumerate(high_risk.iterrows(), start=1):
            rl = str(r.get("risk_level", ""))
            plain_rl = _RISK_PLAIN.get(rl, rl)
            rows.append([
                str(r.get("brand", "")),
                str(r.get("building", "—")),
                plain_rl,
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

    # ── Full results table (top 30 by score) ─────────────────────────────────
    story.append(PageBreak())
    story.append(_section_bar("전체 결과 (복합 점수 상위 30개)", T, W))

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

    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 비용분析 PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cross_pdf(
    unit_df: pd.DataFrame | None = None,
    elec_df: pd.DataFrame | None = None,
    context: dict = None,
) -> bytes:
    """Generate a business-ready PDF for 비용분析 (unit costs + electricity breakdown)."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf)
    W = doc.width

    story = []
    story += _cover_items("비용 분析 리포트", "단위 비용 이상 감지 및 전기 사용 구성 분析", context, T)

    has_content = False

    # ── Unit cost section ─────────────────────────────────────────────────────
    if unit_df is not None and not unit_df.empty:
        has_content = True
        story.append(_section_bar("단위 비용 분析 (₩/m³ · ₩/kWh)", T, W))
        story.append(Paragraph(
            "Z-점수 |Z| ≥ 2.0인 브랜드는 동종 대비 비용 이상으로 분류됩니다.",
            T["caption"],
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Determine available columns
        cost_cols = []
        col_labels = []
        for col, label in [
            ("water_unit_cost", "수도 단가(₩/m³)"),
            ("water_unit_z",    "수도 Z"),
            ("elect_unit_cost", "전기 단가(₩/kWh)"),
            ("elect_unit_z",    "전기 Z"),
            ("total_cost_per_m2", "총비용(만₩/m²)"),
            ("total_cost_per_m2_z", "총비용 Z"),
        ]:
            if col in unit_df.columns:
                cost_cols.append(col)
                col_labels.append(label)

        id_cols = [c for c in ["brand", "building"] if c in unit_df.columns]
        id_labels = ["브랜드", "건물"][: len(id_cols)]
        show = unit_df[id_cols + cost_cols].sort_values(
            cost_cols[0] if cost_cols else id_cols[0], ascending=False
        )

        headers = id_labels + col_labels
        n_id = len(id_cols)
        id_w = [3.5 * cm, 1.2 * cm][: n_id]
        data_w = [(W - sum(id_w)) / max(len(col_labels), 1)] * len(col_labels)
        col_w = id_w + data_w

        rows = [headers]
        row_styles = []
        z_cols_idx = {c: i + n_id for i, c in enumerate(cost_cols) if c.endswith("_z")}
        for ri, (_, r) in enumerate(show.iterrows(), start=1):
            row = [str(r.get(c, "—")) for c in id_cols]
            for c in cost_cols:
                v = r.get(c)
                row.append(_f(v, 2))
                if c in z_cols_idx:
                    try:
                        if abs(float(v)) >= 2.0:
                            ci = z_cols_idx[c]
                            row_styles.append(("BACKGROUND", (ci, ri), (ci, ri), C_CRITICAL))
                            row_styles.append(("TEXTCOLOR",  (ci, ri), (ci, ri), C_WHITE))
                    except (TypeError, ValueError):
                        pass
            rows.append(row)

        story.append(_std_table(rows, col_w, row_styles))
        story.append(Spacer(1, 0.5 * cm))

        # Anomaly summary
        anom_rows = []
        for col, label in zip(cost_cols, col_labels):
            if col.endswith("_z") and col in unit_df.columns:
                flags = unit_df[unit_df[col].abs() >= 2.0]
                for _, r in flags.iterrows():
                    brand = r.get("brand", "")
                    z_val = _f(r.get(col), 2)
                    anom_rows.append([str(brand), label.replace(" Z", ""), z_val])

        if anom_rows:
            story.append(_section_bar(f"비용 이상 브랜드 ({len(anom_rows)}건)", T, W))
            anom_data = [["브랜드", "이상 항목", "Z-점수"]] + anom_rows
            anom_w = [5 * cm, 5 * cm, W - 10 * cm]
            anom_styles = [
                ("TEXTCOLOR", (2, 1), (2, -1), C_CRITICAL),
                ("FONTNAME",  (2, 1), (2, -1), "NanumGothic-Bold"),
            ]
            story.append(_std_table(anom_data, anom_w, anom_styles))
            story.append(Spacer(1, 0.5 * cm))

    # ── Electricity breakdown section ─────────────────────────────────────────
    if elec_df is not None and not elec_df.empty:
        has_content = True
        story.append(PageBreak())
        story.append(_section_bar("전기 사용 구성 분析 (EHP / HVAC / 기저부하)", T, W))

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

    if not has_content:
        story.append(Paragraph("분析 데이터를 불러올 수 없습니다.", T["body"]))

    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 효율분析 PDF
# ═══════════════════════════════════════════════════════════════════════════════

_UTIL_KO = {"water": "수도", "hwater": "온수", "elect": "전기", "heat": "난방"}
_UNIT_KO = {"water": "m³/m²", "hwater": "m³/m²", "elect": "kWh/m²", "heat": "m³(MWh)/m²"}


def generate_efficiency_pdf(
    cur_df: pd.DataFrame,
    present: list[str],
    context: dict = None,
) -> bytes:
    """Generate a business-ready PDF for 효율분析."""
    buf = io.BytesIO()
    doc, T = _build_doc(buf)
    W = doc.width

    story = []
    story += _cover_items("효율 분析 리포트", "브랜드별 단위 면적당 에너지 소비 효율 순위", context, T)

    avail = {p: f"{p}_usage_per_m2" for p in present if f"{p}_usage_per_m2" in cur_df.columns}

    if not avail:
        story.append(Paragraph("전용면적 데이터가 없어 효율 분析을 생성할 수 없습니다.", T["body"]))
        doc.build(story, canvasmaker=_make_numbered_canvas(T))
        return buf.getvalue()

    for prefix, per_m2_col in avail.items():
        util_ko = _UTIL_KO.get(prefix, prefix)
        unit = _UNIT_KO.get(prefix, "unit/m²")

        df_util = cur_df[[c for c in ["brand", "building", per_m2_col]
                           if c in cur_df.columns]].dropna(subset=[per_m2_col]).copy()
        df_util[per_m2_col] = pd.to_numeric(df_util[per_m2_col], errors="coerce")
        df_util = df_util.dropna(subset=[per_m2_col]).sort_values(per_m2_col)

        if df_util.empty:
            continue

        story.append(_section_bar(f"{util_ko} 효율 순위 ({unit})", T, W))
        story.append(Paragraph(
            f"단위 면적당 {util_ko} 소비량 기준 — 낮을수록 효율적",
            T["caption"],
        ))
        story.append(Spacer(1, 0.2 * cm))

        id_cols = [c for c in ["brand", "building"] if c in df_util.columns]
        id_labels = ["브랜드", "건물"][: len(id_cols)]
        headers = id_labels + [f"소비량({unit})", "순위"]
        n_id = len(id_cols)
        id_w = [3.5 * cm, 1.2 * cm][: n_id]
        data_w = (W - sum(id_w)) / 2
        col_w = id_w + [data_w, data_w]

        rows = [headers]
        n = len(df_util)
        eff_styles = []
        for rank, (_, r) in enumerate(df_util.iterrows(), start=1):
            row = [str(r.get(c, "")) for c in id_cols]
            row.append(_f(r[per_m2_col], 4))
            row.append(str(rank))
            rows.append(row)
            # Highlight top/bottom 20%
            if rank <= max(1, n // 5):
                eff_styles.append(("BACKGROUND", (n_id, rank), (n_id, rank), C_STABLE))
                eff_styles.append(("TEXTCOLOR",  (n_id, rank), (n_id, rank), C_WHITE))
            elif rank > n - max(1, n // 5):
                eff_styles.append(("BACKGROUND", (n_id, rank), (n_id, rank), C_CRITICAL))
                eff_styles.append(("TEXTCOLOR",  (n_id, rank), (n_id, rank), C_WHITE))

        story.append(_std_table(rows, col_w, eff_styles))
        story.append(Paragraph(
            "🟢 상위 20% (효율 우수)  |  🔴 하위 20% (고소비, 점검 권장)",
            T["caption"],
        ))
        story.append(Spacer(1, 0.6 * cm))

    # ── Combined efficiency score (if multiple utilities) ─────────────────────
    if len(avail) >= 2:
        story.append(PageBreak())
        story.append(_section_bar("종합 효율 점수 (유틸리티 순위 합산)", T, W))
        story.append(Paragraph(
            "각 유틸리티의 순위를 합산 — 낮을수록 전반적으로 효율적인 브랜드",
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

    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    return buf.getvalue()

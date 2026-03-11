"""tab_anomaly.py — 이상감지 분석 (Anomaly Detection Analysis) UI.

Focused investigation view:
  1. KPI row             — risk-level brand counts
  2. Master table        — who to investigate and WHY (above fold, no expander)
  3. Visual ranking      — composite bar chart + heatmap
  4. Detail tabs:
       📈 급등 감지   — MoM spike detection with peer context (unique)
       🔍 일관성 검사 — zero-usage / sheet cross-check (unique)
  5. Reference           — PDF, scoring method, raw data

Cost / HVAC / consumption detail → Tier 2 인사이트 (no duplication).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from anomaly_features import (
    build_anomaly_df, _UTIL_PREFIXES, _UTIL_LABELS,
    _SPIKE_CRITICAL, _SPIKE_HIGH, _SPIKE_MEDIUM,
)
from data import (
    read_billing_sheet,    BILLING_SHEET_NAME,
    read_electricity_sheet, ELECTRICITY_SHEET_NAME,
    read_water_sheet,       WATER_SHEET_NAME,
    read_hotwater_sheet,    HOTWATER_SHEET_NAME,
)
from features import add_display_index, download_df_as_excel
from biz_report import render_pdf_buttons, generate_anomaly_pdf

_BLDG_COLOR = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd"}
_RISK_COLOR = {
    "🔴 위험": "#C44E52",
    "🟠 주의": "#DD8A00",
    "🟡 관찰": "#F0C040",
    "🟢 정상": "#2ca02c",
}
_UTIL_LABELS_UI = {"water": "💧 수도", "hwater": "🌡 온수",
                   "elect": "⚡ 전기",  "heat":   "🔥 난방"}

_SCORE_CSCALE = [
    [0.00, "#2ca02c"],
    [0.35, "#F0C040"],
    [0.60, "#DD8A00"],
    [1.00, "#C44E52"],
]


def _handle_chart_click(ev, df: pd.DataFrame, field: str = "x",
                         match: str = "exact", trunc: int = 0) -> None:
    """Show selected brand detail from a plotly chart click event."""
    pts = ev.selection.points if ev and hasattr(ev, "selection") else []
    if not pts:
        return
    pt = pts[0]
    brand = pt.get(field) or ""
    if isinstance(brand, (list, tuple)):
        brand = brand[0]
    if not brand:
        return
    if match == "contains" and trunc:
        fdf = df[df["brand"].str.contains(str(brand)[:trunc], regex=False)]
    else:
        fdf = df[df["brand"] == brand]
    if not fdf.empty:
        st.caption(f"선택됨: **{brand}**")
        st.dataframe(fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


# ── Sheet loader ──────────────────────────────────────────────────────────────

def _load_sheets(file_name: str, file_data: bytes, all_sheet_keys: list[str]) -> dict:
    loaders = {
        BILLING_SHEET_NAME:     read_billing_sheet,
        ELECTRICITY_SHEET_NAME: read_electricity_sheet,
        WATER_SHEET_NAME:       read_water_sheet,
        HOTWATER_SHEET_NAME:    read_hotwater_sheet,
    }
    results = {}
    for const, loader in loaders.items():
        key = next((k for k in all_sheet_keys if k.strip() == const), None)
        if key is None:
            continue
        try:
            results[const] = loader(file_name, file_data, key)
        except Exception as e:
            st.warning(f"⚠️ {const} 로드 실패: {e}")
    return results


# ── Section: KPI row ──────────────────────────────────────────────────────────

def _render_kpis(df: pd.DataFrame, has_billing: bool, has_elec: bool) -> None:
    counts = df["risk_level"].value_counts()
    cols = st.columns(5)
    cols[0].metric("분석 브랜드", f"{len(df)}개",
                   help="이상감지 분석 대상 전체 브랜드 수")
    cols[1].metric("🔴 위험", f"{counts.get('🔴 위험', 0)}개",
                   help="복합 이상 점수 ≥ 0.65 — 즉시 조사 필요")
    cols[2].metric("🟠 주의", f"{counts.get('🟠 주의', 0)}개",
                   help="복합 이상 점수 ≥ 0.40 — 모니터링 권장")
    cols[3].metric("🟡 관찰", f"{counts.get('🟡 관찰', 0)}개",
                   help="복합 이상 점수 ≥ 0.20 — 경미한 이상 신호")
    cols[4].metric("🟢 정상", f"{counts.get('🟢 정상', 0)}개",
                   help="복합 이상 점수 < 0.20 — 정상 범위")
    sources = ["검침"] + (["청구"] if has_billing else []) + (["전기"] if has_elec else [])
    st.caption(f"📂 분석 데이터: **{' · '.join(sources)}**")


# ── Section: Composite ranked bar chart ───────────────────────────────────────

def _render_composite_bar(df: pd.DataFrame, n: int, split_by_building: bool) -> None:
    top = df.head(n).copy()
    marker_color = (
        [_BLDG_COLOR.get(str(b), "#888") for b in top["building"]]
        if split_by_building and "building" in top.columns
        else [_RISK_COLOR.get(r, "#888") for r in top["risk_level"]]
    )
    fig = go.Figure(go.Bar(
        x=top["composite_score"],
        y=[str(b)[:28] for b in top["brand"]],
        orientation="h",
        marker_color=marker_color,
        text=[f'{r}  {s:.3f}' for r, s in zip(top["risk_level"], top["composite_score"])],
        textposition="outside",
        textfont=dict(size=9, color="black"),
        hovertemplate="<b>%{y}</b><br>복합 이상 점수: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"복합 이상 점수 — 상위 {n}개 브랜드",
        height=max(400, n * 22 + 80),
        xaxis=dict(title="점수 [0–1]", range=[0, 1.20],
                   gridcolor="#DDDDDD", griddash="dot"),
        plot_bgcolor="white",
        margin=dict(l=10, r=170, t=50, b=40),
        showlegend=False,
    )
    ev = st.plotly_chart(fig, use_container_width=True, key="anom_composite_bar", on_select="rerun")
    _handle_chart_click(ev, df, field="y", match="contains", trunc=26)


# ── Section: Anomaly heatmap ──────────────────────────────────────────────────

def _render_heatmap(df: pd.DataFrame, n: int) -> None:
    top = df.head(n).copy()

    heat_cols: list[str] = []
    col_labels: list[str] = []

    for pfx, label in _UTIL_LABELS_UI.items():
        qc = f"{pfx}_quad_score"
        if qc in top.columns:
            heat_cols.append(qc)
            col_labels.append(f"{label}\n사분면")

    for col, label in [
        ("water_unit_z",        "수도\n단가Z"),
        ("elect_unit_z",        "전기\n단가Z"),
        ("total_cost_per_m2_z", "총비용\n/m²Z"),
        ("hvac_intensity_z",    "HVAC\n강도Z"),
        ("n_zero_utilities",    "미계량\n항목수"),
    ]:
        if col in top.columns:
            heat_cols.append(col)
            col_labels.append(label)

    if not heat_cols:
        return

    matrix = top[heat_cols].fillna(0).copy()
    for c in heat_cols:
        if "_z" in c:
            matrix[c] = matrix[c].abs()

    norm = matrix.apply(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0,
        axis=0,
    )

    brand_labels = [str(b)[:26] for b in top["brand"]]

    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=col_labels,
        y=brand_labels,
        colorscale=_SCORE_CSCALE,
        zmin=0, zmax=1,
        customdata=matrix.values,
        hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="강도", len=0.6,
                      tickvals=[0, 0.5, 1], ticktext=["낮음", "중간", "높음"]),
    ))
    fig.update_layout(
        title=f"이상 신호 히트맵 — 상위 {n}개 브랜드",
        height=max(400, n * 20 + 120),
        xaxis=dict(side="top", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=100, t=120, b=20),
    )
    ev = st.plotly_chart(fig, use_container_width=True, key="anom_signal_heatmap", on_select="rerun")
    _handle_chart_click(ev, df, field="y", match="contains", trunc=26)


# ── Tab: 급등 감지 (MoM Spike Detection) ─────────────────────────────────────

def _render_spike_tab(df: pd.DataFrame, split_by_building: bool) -> None:
    st.subheader("📈 전월 대비 급등 감지 — 상세 분석")
    st.caption(
        f"전월 대비 사용량 증가율이 🔴 {_SPIKE_CRITICAL:.0f}% 이상 / "
        f"🟠 {_SPIKE_HIGH:.0f}% 이상 / 🟡 {_SPIKE_MEDIUM:.0f}% 이상인 브랜드를 탐지합니다. "
        "**vs건물** 열은 같은 건물 내 다른 브랜드 대비 급등 배수를 나타냅니다 (2x 이상 = 동종 대비 이상)."
    )

    spike_pct_cols = [f"{p}_spike_pct" for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns]
    if not spike_pct_cols:
        st.info("전월 데이터가 없어 급등 감지를 수행할 수 없습니다.")
        return

    # ── Threshold selector ────────────────────────────────────────────────────
    thresh = st.slider(
        "급등 기준 (전월 대비 증가율 %)", 10, 300, int(_SPIKE_HIGH), step=10,
        key="spike_thresh",
        help="선택한 % 이상 증가한 브랜드만 표시합니다.",
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    n_critical = int((df["spike_max_pct"] >= _SPIKE_CRITICAL).sum())
    n_high     = int(((df["spike_max_pct"] >= _SPIKE_HIGH) & (df["spike_max_pct"] < _SPIKE_CRITICAL)).sum())
    n_medium   = int(((df["spike_max_pct"] >= _SPIKE_MEDIUM) & (df["spike_max_pct"] < _SPIKE_HIGH)).sum())
    n_above    = int((df["spike_max_pct"] >= thresh).sum())
    kc = st.columns(4)
    kc[0].metric(f"🔴 급등 (≥{_SPIKE_CRITICAL:.0f}%)", f"{n_critical}개")
    kc[1].metric(f"🟠 주의 (≥{_SPIKE_HIGH:.0f}%)",     f"{n_high}개")
    kc[2].metric(f"🟡 관찰 (≥{_SPIKE_MEDIUM:.0f}%)",   f"{n_medium}개")
    kc[3].metric(f"기준 초과 (≥{thresh}%)",             f"{n_above}개")

    # ── Spike brands table (with peer context) ─────────────────────────────
    spike_df = df[df["spike_max_pct"] >= thresh].copy()
    if spike_df.empty:
        st.success(f"기준({thresh}%) 초과 브랜드 없음 — 급격한 급등 없음")
    else:
        peer_cols = [c for c in ["spike_bldg_avg_pct", "spike_peer_ratio"] if c in spike_df.columns]
        disp_cols = (
            [c for c in ["brand", "building", "floor"] if c in spike_df.columns]
            + ["spike_max_pct", "spike_worst_util"]
            + peer_cols
            + spike_pct_cols
        )
        col_cfg: dict = {
            "spike_max_pct":      st.column_config.NumberColumn("최대 증가율 (%)", format="%.1f"),
            "spike_worst_util":   st.column_config.TextColumn("급등 항목"),
            "spike_bldg_avg_pct": st.column_config.NumberColumn("건물평균(%)", format="%.1f"),
            "spike_peer_ratio":   st.column_config.NumberColumn("vs건물", format="%.1fx"),
        }
        util_labels = {f"{p}_spike_pct": f"{lbl} 증가율(%)" for p, lbl in _UTIL_LABELS_UI.items()}
        for c, lbl in util_labels.items():
            if c in spike_df.columns:
                col_cfg[c] = st.column_config.NumberColumn(lbl, format="%.1f")

        st.dataframe(
            spike_df[disp_cols].sort_values("spike_max_pct", ascending=False).reset_index(drop=True),
            column_config=col_cfg,
            hide_index=True,
            use_container_width=True,
        )

    # ── Spike bar chart per utility ───────────────────────────────────────────
    st.divider()
    util_sel = st.selectbox(
        "유틸리티별 전월 대비 증가율",
        [p for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns],
        format_func=lambda p: _UTIL_LABELS_UI.get(p, p),
        key="spike_util_sel",
    )
    pct_col = f"{util_sel}_spike_pct"
    flag_col = f"{util_sel}_spike_flag"

    chart_df = df[["brand"] + [c for c in ["building", pct_col, flag_col] if c in df.columns]].copy()
    chart_df = chart_df[chart_df[pct_col].notna()].sort_values(pct_col, ascending=False).head(50)

    color_col = "building" if split_by_building and "building" in chart_df.columns else None
    fig = px.bar(
        chart_df, x="brand", y=pct_col,
        color=color_col, color_discrete_map=_BLDG_COLOR,
        title=f"{_UTIL_LABELS_UI.get(util_sel, util_sel)} 전월 대비 증가율 (%) — 상위 50개",
        labels={pct_col: "증가율 (%)", "brand": "브랜드"},
    )
    for lvl, color, label in [
        (_SPIKE_CRITICAL, "#C44E52", f"급등 {_SPIKE_CRITICAL:.0f}%"),
        (_SPIKE_HIGH,     "#DD8A00", f"주의 {_SPIKE_HIGH:.0f}%"),
        (_SPIKE_MEDIUM,   "#F0C040", f"관찰 {_SPIKE_MEDIUM:.0f}%"),
    ]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top right")
    _raw_pcts = df[pct_col].clip(lower=0).fillna(0)
    _overall_avg = float(_raw_pcts.mean())
    if _overall_avg > 0:
        fig.add_hline(y=_overall_avg, line_dash="dash", line_color="#4C72B0", line_width=2,
                      annotation_text=f"전체 평균 {_overall_avg:.0f}%",
                      annotation_position="bottom right",
                      annotation_font_color="#4C72B0")
    fig.update_layout(
        height=420, xaxis_tickangle=-45,
        plot_bgcolor="white", margin=dict(t=55, b=90),
    )
    ev = st.plotly_chart(fig, use_container_width=True, key="anom_spike_bar", on_select="rerun")
    _handle_chart_click(ev, chart_df, field="x")


# ── Tab: 일관성 검사 ──────────────────────────────────────────────────────────

def _render_consistency_tab(df: pd.DataFrame) -> None:
    st.subheader("일관성 검사 — 미계량 유틸리티")
    st.caption("사용량=0으로 기록된 유틸리티 항목 수. 계량기 미설치 또는 입력 누락 가능성.")

    if "n_zero_utilities" not in df.columns:
        st.info("데이터 없음")
        return

    zero_df = (df[df["n_zero_utilities"] > 0]
               [[c for c in ["brand", "building", "floor", "n_zero_utilities",
                             "risk_level", "composite_score"] if c in df.columns]]
               .sort_values("n_zero_utilities", ascending=False))

    c1, c2 = st.columns([1, 2])
    with c1:
        total_zero = int((df["n_zero_utilities"] > 0).sum())
        st.metric("미계량 브랜드", f"{total_zero}개")
        st.metric("전 유틸리티 미계량",
                  f"{int((df['n_zero_utilities'] >= len([p for p in _UTIL_PREFIXES if f'{p}_current' in df.columns])).sum())}개")

    with c2:
        hist_df = df["n_zero_utilities"].value_counts().sort_index().reset_index()
        hist_df.columns = ["미계량 항목 수", "브랜드 수"]
        fig = px.bar(hist_df, x="미계량 항목 수", y="브랜드 수",
                     title="미계량 항목 수 분포",
                     color_discrete_sequence=["#DD8A00"])
        fig.update_layout(height=280, plot_bgcolor="white", margin=dict(t=45, b=30))
        st.plotly_chart(fig, use_container_width=True, key="anom_zero_dist_bar")

    if zero_df.empty:
        st.success("미계량 브랜드 없음 — 모든 유틸리티 정상 계량")
    else:
        st.dataframe(zero_df, hide_index=True, use_container_width=True)

    # Cross-check: water/hotwater sheet vs meter
    for sheet_col, label in [("water_sheet_m3", "💧 수도 시트 vs 검침"),
                              ("hotwater_sheet_m3", "🌡 온수 시트 vs 검침")]:
        if sheet_col not in df.columns:
            continue
        meter_col = ("water_current" if "water" in sheet_col else "hwater_current")
        if meter_col not in df.columns:
            continue
        st.caption(f"**{label}** — 시트 사용량 vs 검침 현재값 비교")
        cross = df[["brand", "building", sheet_col, meter_col]].copy()
        cross["차이"] = (cross[sheet_col] - cross[meter_col]).round(2)
        cross["불일치"] = (cross["차이"].abs() > cross[meter_col].abs() * 0.05) & cross["차이"].notna()
        mismatch = cross[cross["불일치"]].sort_values("차이", key=abs, ascending=False)
        if not mismatch.empty:
            st.warning(f"**{len(mismatch)}개** 브랜드 — 시트/검침 5% 이상 불일치")
            st.dataframe(mismatch.drop(columns=["불일치"]), hide_index=True, use_container_width=True)
        else:
            st.success("시트/검침 불일치 없음")


# ── Public render ─────────────────────────────────────────────────────────────

def render_anomaly_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    all_sheet_keys: list[str],
    split_by_building: bool = True,
) -> None:
    """Render the 이상감지 분석 view — loads immediately (no lazy-load button)."""

    with st.spinner("데이터 로드 및 이상 신호 산출 중…"):
        sheets = _load_sheets(file_name, file_data, all_sheet_keys)
        try:
            anomaly_df = build_anomaly_df(
                meter_df=cur_df,
                billing_df=sheets.get(BILLING_SHEET_NAME),
                elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
                water_df=sheets.get(WATER_SHEET_NAME),
                hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
            )
        except Exception as e:
            st.error(f"이상감지 분석 실패: {e}")
            return

    if anomaly_df.empty:
        st.error("이상감지 데이터를 생성할 수 없습니다.")
        return

    has_billing = BILLING_SHEET_NAME in sheets
    has_elec    = ELECTRICITY_SHEET_NAME in sheets

    # ── 1. KPI row — "How many problems?" ────────────────────────────────────
    _render_kpis(anomaly_df, has_billing, has_elec)

    # ── 2. Master table — "Who to investigate and WHY?" ──────────────────────
    st.subheader("🔍 조사 대상 브랜드")
    st.caption("복합 이상 점수 순으로 정렬 — **이유** 컬럼에서 각 브랜드가 왜 플래그되었는지 확인하세요.")

    id_cols    = [c for c in ["brand", "building", "floor"] if c in anomaly_df.columns]
    reason_col = ["reason"] if "reason" in anomaly_df.columns else []
    key_cols   = [c for c in ["composite_score", "risk_level",
                              "spike_max_pct", "spike_worst_util",
                              "spike_bldg_avg_pct", "spike_peer_ratio"] if c in anomaly_df.columns]
    master_show = id_cols + key_cols + reason_col
    master_view = add_display_index(anomaly_df[master_show])
    st.dataframe(
        master_view,
        column_config={
            "composite_score": st.column_config.ProgressColumn(
                "복합 점수", format="%.3f", min_value=0, max_value=1),
            "spike_max_pct":       st.column_config.NumberColumn(
                "최대 증가율(%)", format="%.1f"),
            "spike_worst_util":    st.column_config.TextColumn("급등 항목"),
            "spike_bldg_avg_pct":  st.column_config.NumberColumn(
                "건물평균(%)", format="%.1f"),
            "spike_peer_ratio":    st.column_config.NumberColumn(
                "vs건물 배수", format="%.1fx"),
            "reason":              st.column_config.TextColumn("이유", width="large"),
        },
        hide_index=True,
        use_container_width=True,
    )
    download_df_as_excel(master_view, filename="anomaly_investigation.xlsx", sheet_name="조사대상")

    st.divider()

    # ── 3. Visual ranking — "See the full picture at a glance" ───────────────
    _n = st.slider("표시 브랜드 수", 10, min(60, len(anomaly_df)),
                   min(10, len(anomaly_df)), key="anom_n")

    _chart_tab_bar, _chart_tab_heat = st.tabs(["📊 복합 점수 순위", "🗺️ 이상 히트맵"])
    with _chart_tab_bar:
        _render_composite_bar(anomaly_df, _n, split_by_building)
    with _chart_tab_heat:
        _render_heatmap(anomaly_df, _n)

    st.divider()

    # ── 4. Detail deep-dives — unique signals only ────────────────────────────
    # Spike detection (unique to anomaly) + consistency check (unique data quality)
    # Cost / HVAC / consumption detail → Tier 2 인사이트 tabs
    tab_spike, tab_chk = st.tabs(["📈 급등 감지", "🔍 일관성 검사"])

    with tab_spike:
        _render_spike_tab(anomaly_df, split_by_building)

    with tab_chk:
        _render_consistency_tab(anomaly_df)

    st.divider()

    # ── 5. Reference — scoring method, PDF, raw data ─────────────────────────
    _ref1, _ref2 = st.columns(2)
    with _ref1:
        _pdf_key = f"anomaly_pdf_{file_name}"
        render_pdf_buttons(
            _pdf_key,
            lambda: generate_anomaly_pdf(anomaly_df),
            "📥 PDF 리포트",
            "이상감지_리포트.pdf",
        )
    with _ref2:
        st.caption("💡 비용·HVAC·소비 상세 분석은 **📊 인사이트** 탭에서 확인하세요.")

    with st.expander("📖 이상 점수 계산 방법", expanded=False):
        st.markdown("""
**복합 점수** = 급등(30%) + 소비(25%) + 비용(25%) + HVAC(10%) + 일관성(10%)  — 각 구성 요소 [0, 1]

| 구성 요소 | 신호 | 시트 |
|---|---|---|
| **급등** ★ | 전월 대비 사용량 증가율 절대값 기준 — 🔴 ≥100% / 🟠 ≥50% / 🟡 ≥20% | 검침내역 |
| **소비** | 유틸리티별 사분면 점수 합산 정규화 (HH=4, HL=3, LH=2, Normal=1, LL=0) | 검침내역 |
| **비용** | 수도 ₩/m³, 전기 ₩/kWh, 총비용 만원/m² Z-점수의 최댓값 정규화 | 수도광열비 부과 내역 |
| **HVAC** | HVAC 강도 (kWh/m²) IQR-보정 정규화 | 전체 전기 사용내역 |
| **일관성** | 사용량=0 유틸리티 항목 수 정규화 | 검침내역 + 수도/온수 시트 |

**위험 등급**: 🔴 위험 ≥ 0.65 · 🟠 주의 ≥ 0.40 · 🟡 관찰 ≥ 0.20 · 🟢 정상 < 0.20

**동종 비교 (vs건물)**: 같은 건물 내 다른 브랜드 평균 급등률 대비 배수. 2x 이상 = 건물 전체 추세가 아닌 해당 브랜드만의 이상.
        """)

    with st.expander("📊 원시 데이터", expanded=False):
        st.dataframe(anomaly_df.reset_index(drop=True), hide_index=True, use_container_width=True)

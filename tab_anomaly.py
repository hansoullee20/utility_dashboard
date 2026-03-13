"""tab_anomaly.py — 이상감지 분석 (Anomaly Detection Analysis) UI.

Focused investigation view:
  1. KPI row             — risk-level brand counts
  2. Master table        — who to investigate and WHY (above fold, no expander)
  3. Visual ranking      — composite bar chart + heatmap
  4. Detail tabs:
       📈 급등 감지   — MoM spike detection with peer context (unique)
       🔍 일관성 검사 — zero-usage + 집계/부과 brand reconciliation
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


# ── Insight summary ──────────────────────────────────────────────────────────

def _render_insight_summary(anomaly_df: pd.DataFrame, sheets: dict) -> None:
    """Compact insight summary from cross-analysis at the top of 점검대상."""
    from utils import z_to_grade as _ztg

    with st.container(border=True):
        st.markdown(
            '<p style="margin:0 0 8px;font-size:0.95rem;font-weight:700;'
            'letter-spacing:0.02em;color:#4C72B0">'
            '핵심 인사이트 요약</p>',
            unsafe_allow_html=True,
        )

        # 1. Risk brand lists with reason
        danger_df = anomaly_df[anomaly_df["risk_level"] == "🔴 위험"].copy()
        caution_df = anomaly_df[anomaly_df["risk_level"] == "🟠 주의"].copy()
        has_reason = "reason" in anomaly_df.columns

        if not danger_df.empty:
            _brands_html = []
            for _, r in danger_df.iterrows():
                reason = str(r.get("reason", "")) if has_reason else ""
                _brands_html.append(
                    f'<span style="background:#C44E5218;border:1px solid #C44E5240;'
                    f'border-radius:6px;padding:3px 8px;margin:2px;display:inline-block;'
                    f'font-size:0.82rem">'
                    f'<b>{r["brand"]}</b>'
                    f'{"<br><span style=color:#888;font-size:0.75rem>" + reason + "</span>" if reason else ""}'
                    f'</span>'
                )
            st.markdown(
                f'<div style="margin-bottom:8px">'
                f'<span style="color:#C44E52;font-weight:700">🔴 즉시 조사 ({len(danger_df)})</span>'
                f'<div style="margin-top:4px">{"".join(_brands_html)}</div></div>',
                unsafe_allow_html=True,
            )

        if not caution_df.empty:
            _brands_html = []
            for _, r in caution_df.iterrows():
                reason = str(r.get("reason", "")) if has_reason else ""
                _brands_html.append(
                    f'<span style="background:#DD8A0012;border:1px solid #DD8A0035;'
                    f'border-radius:6px;padding:3px 8px;margin:2px;display:inline-block;'
                    f'font-size:0.82rem">'
                    f'<b>{r["brand"]}</b>'
                    f'{"<br><span style=color:#888;font-size:0.75rem>" + reason + "</span>" if reason else ""}'
                    f'</span>'
                )
            st.markdown(
                f'<div style="margin-bottom:8px">'
                f'<span style="color:#DD8A00;font-weight:700">🟠 모니터링 ({len(caution_df)})</span>'
                f'<div style="margin-top:4px">{"".join(_brands_html)}</div></div>',
                unsafe_allow_html=True,
            )

        # 2. Spike / cost / zero summary lines
        lines: list[str] = []

        if "spike_max_pct" in anomaly_df.columns:
            spike_df = anomaly_df[anomaly_df["spike_max_pct"] >= _SPIKE_HIGH].nlargest(5, "spike_max_pct")
            if not spike_df.empty:
                parts = [f"**{r['brand']}** +{r['spike_max_pct']:.0f}%" for _, r in spike_df.iterrows()]
                lines.append(f"📈 급등: {' · '.join(parts)}")

        _cost_checks = [
            ("water_unit_z", "water_unit_cost", "수도단가", "원/m³"),
            ("elect_unit_z", "elect_unit_cost", "전기단가", "원/kWh"),
        ]
        # Prefer per-평; fall back to per-m²
        if "total_cost_per_py_z" in anomaly_df.columns:
            _cost_checks.append(("total_cost_per_py_z", "total_cost_per_py", "평당 사용량", "만원/평"))
        elif "total_cost_per_m2_z" in anomaly_df.columns:
            _cost_checks.append(("total_cost_per_m2_z", "total_cost_per_m2", "평당 사용량", "만원/m²"))

        for z_col, val_col, label, unit in _cost_checks:
            if z_col not in anomaly_df.columns:
                continue
            extreme = anomaly_df[anomaly_df[z_col].abs() >= 2.0]
            if extreme.empty:
                continue
            top = extreme.nlargest(3, z_col, keep="first")
            parts = []
            for _, r in top.iterrows():
                val = r.get(val_col)
                val_str = f" {val:,.1f}{unit}" if val is not None and not pd.isna(val) else ""
                parts.append(f"**{r['brand']}**({_ztg(r[z_col])}{val_str})")
            lines.append(f"💰 {label} 이상: {' · '.join(parts)}")

        if "n_zero_utilities" in anomaly_df.columns:
            zero_brands = anomaly_df[anomaly_df["n_zero_utilities"] > 0]
            if not zero_brands.empty:
                n_zero = len(zero_brands)
                top_zero = zero_brands.nlargest(3, "n_zero_utilities")
                parts = [f"**{r['brand']}**({int(r['n_zero_utilities'])}항목)" for _, r in top_zero.iterrows()]
                lines.append(f"⚠️ 미계량 {n_zero}개: {' · '.join(parts)}")

        if BILLING_SHEET_NAME in sheets:
            lines.append("✅ 비용 시트 연계 완료")

        if danger_df.empty and caution_df.empty and not lines:
            st.info("특이 사항 없음")
        elif lines:
            st.markdown("  \n".join(lines))


# ── Zero-usage change detection (vs prev/yoy) ────────────────────────────────

def _build_zero_set(file_data: bytes, sheet_keys: list[str]) -> dict[str, set[str]]:
    """Build per-utility set of zero-usage brands from a period file.

    Returns {utility_prefix: {brand_name, ...}} for brands with current==0.
    """
    from data import read_sheet
    from features import apply_header_rows, build_from_two_files, create_change_columns, aggregate_by_brand
    from data import to_numeric_series

    meter_key = next((k for k in sheet_keys if k.strip() == "검침 내역"), None)
    if not meter_key:
        return {}
    try:
        raw = read_sheet("__zero__.xlsx", file_data, meter_key)
        df_cur = apply_header_rows(raw)
        df_cur["building"] = df_cur["building"].astype(str).str.strip()
        df_cur = df_cur[df_cur["building"].isin({"A", "B", "C", "D"})].copy()
        df = build_from_two_files(df_cur, None)
        raw_df = create_change_columns(df)
        agg = aggregate_by_brand(raw_df)
    except Exception:
        return {}

    result: dict[str, set[str]] = {}
    for pfx in _UTIL_PREFIXES:
        col = f"{pfx}_current"
        if col in agg.columns:
            zeros = set(agg.loc[to_numeric_series(agg[col]).fillna(0) == 0, "brand"].astype(str))
            if zeros:
                result[pfx] = zeros
    return result


def _render_zero_change(
    cur_df: pd.DataFrame,
    prev_file_data: bytes | None,
    prev_sheet_keys: list[str] | None,
    prev_label: str | None,
    yoy_file_data: bytes | None,
    yoy_sheet_keys: list[str] | None,
    yoy_label: str | None,
) -> None:
    """Compare zero-usage brands between current and previous/yoy periods."""
    from data import to_numeric_series

    # Current zero-usage sets
    cur_zeros: dict[str, set[str]] = {}
    for pfx in _UTIL_PREFIXES:
        col = f"{pfx}_current"
        if col in cur_df.columns:
            zeros = set(cur_df.loc[to_numeric_series(cur_df[col]).fillna(0) == 0, "brand"].astype(str))
            if zeros:
                cur_zeros[pfx] = zeros

    comparisons: list[tuple[str, dict[str, set[str]]]] = []
    if prev_file_data and prev_sheet_keys:
        prev_zeros = _build_zero_set(prev_file_data, prev_sheet_keys)
        if prev_zeros:
            comparisons.append((prev_label or "전월", prev_zeros))
    if yoy_file_data and yoy_sheet_keys:
        yoy_zeros = _build_zero_set(yoy_file_data, yoy_sheet_keys)
        if yoy_zeros:
            comparisons.append((yoy_label or "전년", yoy_zeros))

    if not comparisons:
        return

    st.divider()
    st.markdown("##### 미계량 변화 감지")
    st.caption("이전 기간 대비 미계량 상태가 변한 브랜드를 표시합니다.")

    for period_lbl, prev_zeros in comparisons:
        rows: list[dict] = []
        all_utils = sorted(set(cur_zeros.keys()) | set(prev_zeros.keys()))
        for pfx in all_utils:
            label = _UTIL_LABELS_UI.get(pfx, pfx)
            cur_set = cur_zeros.get(pfx, set())
            prev_set = prev_zeros.get(pfx, set())
            # Newly zero (was metered, now zero)
            for b in sorted(cur_set - prev_set):
                rows.append({"브랜드": b, "유틸리티": label, "변화": "🔴 새로 미계량",
                             "설명": f"{period_lbl}에는 계량 → 현재 미계량"})
            # Recovered (was zero, now metered)
            for b in sorted(prev_set - cur_set):
                rows.append({"브랜드": b, "유틸리티": label, "변화": "🟢 계량 복구",
                             "설명": f"{period_lbl}에는 미계량 → 현재 계량"})

        if rows:
            change_df = pd.DataFrame(rows)
            n_new = sum(1 for r in rows if "새로" in r["변화"])
            n_rec = sum(1 for r in rows if "복구" in r["변화"])
            st.markdown(f"**vs {period_lbl}**: 🔴 새로 미계량 {n_new}건 · 🟢 계량 복구 {n_rec}건")
            st.dataframe(change_df, hide_index=True, use_container_width=True)
        else:
            st.success(f"vs {period_lbl}: 미계량 변화 없음")


# ── Period spike detection (prev/yoy) ────────────────────────────────────────

def _render_period_spike(
    file_data: bytes,
    sheet_keys: list[str],
    period_label: str,
    split_by_building: bool,
    key_suffix: str = "",
) -> None:
    """Load meter data from a different period file and render spike detection."""
    from data import read_sheet
    from features import (
        apply_header_rows, build_from_two_files,
        create_change_columns, aggregate_by_brand,
    )

    meter_key = next((k for k in sheet_keys if k.strip() == "검침 내역"), None)
    if not meter_key:
        st.info(f"{period_label}: 검침 내역 시트를 찾을 수 없습니다.")
        return

    _tmp_name = f"__period_{period_label}__.xlsx"

    try:
        raw = read_sheet(_tmp_name, file_data, meter_key)
        df_cur = apply_header_rows(raw)
        df_cur["building"] = df_cur["building"].astype(str).str.strip()
        df_cur = df_cur[df_cur["building"].isin({"A", "B", "C", "D"})].copy()
        df = build_from_two_files(df_cur, None)
        raw_df = create_change_columns(df)
        agg_df = aggregate_by_brand(raw_df)
    except Exception as e:
        st.warning(f"{period_label} 데이터 로드 실패: {e}")
        return

    # Load supporting sheets and build anomaly df
    sheets = _load_sheets(_tmp_name, file_data, sheet_keys)
    try:
        period_anomaly = build_anomaly_df(
            meter_df=agg_df,
            billing_df=sheets.get(BILLING_SHEET_NAME),
            elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
            water_df=sheets.get(WATER_SHEET_NAME),
            hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
        )
    except Exception as e:
        st.warning(f"{period_label} 이상감지 분석 실패: {e}")
        return

    if period_anomaly.empty:
        st.info(f"{period_label}: 분석 가능한 데이터가 없습니다.")
        return

    _render_spike_tab(period_anomaly, split_by_building,
                      key_suffix=key_suffix)


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
    sources = ["검침"] + (["청구"] if has_billing else []) + (["전기"] if has_elec else [])
    n_total = len(df)
    n_danger = counts.get("🔴 위험", 0)
    n_caution = counts.get("🟠 주의", 0)
    n_watch = counts.get("🟡 관찰", 0)
    n_normal = counts.get("🟢 정상", 0)

    # Risk gauge summary
    _pct_risk = (n_danger + n_caution) / n_total * 100 if n_total else 0
    _gauge_color = "#C44E52" if _pct_risk >= 30 else "#DD8A00" if _pct_risk >= 15 else "#2ca02c"
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_gauge_color}10,{_gauge_color}03);'
        f'border-left:4px solid {_gauge_color};border-radius:8px;padding:12px 16px;margin-bottom:12px">'
        f'<span style="font-size:1.3rem;font-weight:800;color:{_gauge_color}">'
        f'{n_danger + n_caution}</span>'
        f'<span style="font-size:0.9rem;color:#555"> / {n_total} 브랜드 조사 필요 '
        f'({_pct_risk:.0f}%)</span>'
        f'<span style="float:right;font-size:0.8rem;color:#888">📂 {" · ".join(sources)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("분석 브랜드", f"{n_total}개",
                   help="이상감지 분석 대상 전체 브랜드 수")
    cols[1].metric("🔴 위험", f"{n_danger}개",
                   help="복합 이상 점수 ≥ 0.65 — 즉시 조사 필요")
    cols[2].metric("🟠 주의", f"{n_caution}개",
                   help="복합 이상 점수 ≥ 0.40 — 모니터링 권장")
    cols[3].metric("🟡 관찰", f"{n_watch}개",
                   help="복합 이상 점수 ≥ 0.20 — 경미한 이상 신호")
    cols[4].metric("🟢 정상", f"{n_normal}개",
                   help="복합 이상 점수 < 0.20 — 정상 범위")


# ── Section: Composite ranked bar chart ───────────────────────────────────────

def _render_composite_bar(df: pd.DataFrame, n: int, split_by_building: bool) -> None:
    top = df.head(n).copy().iloc[::-1]  # reverse so highest is at top of h-bar
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
        ("water_unit_z",        "수도\n단가등급"),
        ("elect_unit_z",        "전기\n단가등급"),
        ("total_cost_per_py_z", "평당\n비용등급"),
        ("total_cost_per_m2_z", "총비용\n/m²등급"),
        ("hvac_intensity_z",    "HVAC\n강도등급"),
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

def _render_spike_tab(df: pd.DataFrame, split_by_building: bool,
                      key_suffix: str = "") -> None:
    sfx = key_suffix
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
        key=f"spike_thresh{sfx}",
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
        key=f"spike_util_sel{sfx}",
    )
    pct_col = f"{util_sel}_spike_pct"
    flag_col = f"{util_sel}_spike_flag"

    chart_df = df[["brand"] + [c for c in ["building", pct_col, flag_col] if c in df.columns]].copy()
    chart_df = chart_df[chart_df[pct_col].notna()].sort_values(pct_col, ascending=False).head(50)

    _logy = st.checkbox("Log 스케일", key=f"spike_logy{sfx}")
    color_col = "building" if split_by_building and "building" in chart_df.columns else None
    fig = px.bar(
        chart_df, x="brand", y=pct_col,
        color=color_col, color_discrete_map=_BLDG_COLOR,
        title=f"{_UTIL_LABELS_UI.get(util_sel, util_sel)} 전월 대비 증가율 (%) — 상위 50개",
        labels={pct_col: "증가율 (%)", "brand": "브랜드"},
        log_y=_logy,
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
    ev = st.plotly_chart(fig, use_container_width=True, key=f"anom_spike_bar{sfx}", on_select="rerun")
    _handle_chart_click(ev, chart_df, field="x")


# ── Tab: 일관성 검사 ──────────────────────────────────────────────────────────

def _render_consistency_tab(
    df: pd.DataFrame,
    file_name: str | None = None,
    file_data: bytes | None = None,
    all_sheet_keys: list[str] | None = None,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
) -> None:
    st.subheader("일관성 검사 — 미계량 + 시트 간 교차검증")

    # ── Section 1: Zero-usage detection ────────────────────────────────────
    st.caption("사용량=0으로 기록된 유틸리티 항목 수. 계량기 미설치 또는 입력 누락 가능성.")

    if "n_zero_utilities" not in df.columns:
        st.info("미계량 데이터 없음")
    else:
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

    # ── Section 1b: Zero-usage CHANGE detection (vs prev/yoy) ──────────
    _render_zero_change(df, prev_file_data, prev_sheet_keys, prev_label,
                        yoy_file_data, yoy_sheet_keys, yoy_label)

    # ── Section 2: Cross-sheet brand reconciliation ────────────────────────
    _render_sheet_reconciliation(
        file_name, file_data, all_sheet_keys,
        prev_file_data=prev_file_data,
        prev_sheet_keys=prev_sheet_keys,
        prev_label=prev_label,
        yoy_file_data=yoy_file_data,
        yoy_sheet_keys=yoy_sheet_keys,
        yoy_label=yoy_label,
    )


def _render_sheet_reconciliation(
    file_name: str | None,
    file_data: bytes | None,
    all_sheet_keys: list[str] | None,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
) -> None:
    """Cross-sheet brand reconciliation: name consistency + amount verification."""
    if not file_data or not all_sheet_keys:
        return

    SH_A = "브랜드별 집계 내역"
    SH_B_match = next((s for s in all_sheet_keys
                       if s.strip() == "수도광열비 부과 내역"), None)
    if SH_A not in all_sheet_keys or SH_B_match is None:
        return
    SH_B = SH_B_match.strip()

    st.divider()

    # Show tabs for each available period
    file_pairs = [("📋 현재 파일", file_data, all_sheet_keys)]
    for extra_data, extra_sheets, extra_label, default_lbl in [
        (prev_file_data, prev_sheet_keys, prev_label, "전월"),
        (yoy_file_data, yoy_sheet_keys, yoy_label, "전년"),
    ]:
        if extra_data and extra_sheets:
            sh_b = next((s for s in extra_sheets
                         if s.strip() == "수도광열비 부과 내역"), None)
            if sh_b and SH_A in extra_sheets:
                lbl = extra_label or default_lbl
                file_pairs.append((f"📅 {lbl}", extra_data, extra_sheets))

    if len(file_pairs) > 1:
        recon_tabs = st.tabs([fp[0] for fp in file_pairs])
    else:
        recon_tabs = [st.container()]

    import io
    from brand_normalize import (
        reconcile_sheets, find_name_inconsistencies,
        normalize_brand, load_synonyms, save_synonyms,
    )
    synonyms = load_synonyms()
    n_saved = len(synonyms)

    for tab_idx, (tab_label, fdata, sheet_keys) in enumerate(file_pairs):
        with recon_tabs[tab_idx]:
            sh_b_match = next((s for s in sheet_keys
                               if s.strip() == "수도광열비 부과 내역"), None)
            if not sh_b_match:
                st.warning("수도광열비 부과 내역 시트를 찾을 수 없습니다.")
                continue

            st.subheader(f"시트 간 교차검증 — {SH_A} vs {SH_B}")

            a_raw = pd.read_excel(io.BytesIO(fdata), sheet_name=SH_A,
                                  header=None, engine="calamine")
            b_raw = pd.read_excel(io.BytesIO(fdata), sheet_name=sh_b_match,
                                  header=None, engine="calamine")

            result = reconcile_sheets(
                a_raw, b_raw,
                a_brand_col=10, b_brand_col=9,
                a_unit_col=4, b_unit_col=4,
                a_totals={"전용": 13, "공용": 14, "합계": 15},
                b_totals={"전용": 21, "공용": 22, "합계": 23},
                synonyms=synonyms,
            )
            s = result["summary"]

            _render_reconciliation_body(
                result, s, fdata, sheet_keys, synonyms, n_saved,
                SH_A, SH_B, tab_idx,
            )


def _render_reconciliation_body(
    result: dict, s: dict,
    file_data: bytes, all_sheet_keys: list[str],
    synonyms: dict, n_saved: int,
    SH_A: str, SH_B: str,
    tab_idx: int = 0,
) -> None:
    """Render reconciliation KPIs, inconsistencies, mapping UI, and mismatches."""
    from brand_normalize import (
        find_name_inconsistencies, normalize_brand, load_synonyms, save_synonyms,
    )
    sfx = f"_{tab_idx}" if tab_idx else ""

    # KPI row
    kc = st.columns(6)
    kc[0].metric(SH_A, f"{s['a_total']}개")
    kc[1].metric(SH_B, f"{s['b_total']}개")
    kc[2].metric("정확 매칭", f"{s['exact_match']}개")
    kc[3].metric("유사 매칭", f"{s['fuzzy_match']}개",
                 help="이름 정규화 후 매칭 (이전 상호 괄호 제거, 공백 통일 등)")
    kc[4].metric("추정 매칭", f"{s['fuzzy_suggested']}건",
                 help="유사도 기반 추정 매칭 (typo, 축약 등)",
                 delta=f"⚠ {s['fuzzy_suggested']}" if s['fuzzy_suggested'] else None,
                 delta_color="off")
    kc[5].metric("금액 불일치", f"{s['amount_mismatches']}건",
                 delta=f"{s['amount_mismatches']}" if s['amount_mismatches'] else None,
                 delta_color="inverse")

    # Name inconsistencies across all sheets (정규화 + 표기 + 사용 시트)
    inconsistencies = find_name_inconsistencies(file_data, all_sheet_keys, synonyms=synonyms)
    if inconsistencies:
        with st.expander(f"시트 간 명칭 불일치 ({len(inconsistencies)}건)", expanded=True):
            st.caption("동일 브랜드가 시트별로 다른 이름으로 기록된 항목")
            _SHEET_COLORS = {
                "브랜드별 집계 내역": "#4C72B0",
                "수도광열비 부과 내역": "#C44E52",
                "수도 사용 내역": "#55A868",
                "온수 사용 내역": "#DD8A00",
                "전체 전기 사용내역": "#8172B2",
            }
            def _badge(sheet: str) -> str:
                bg = _SHEET_COLORS.get(sheet, "#888")
                return (f'<span style="background:{bg};color:#fff;padding:2px 7px;'
                        f'border-radius:10px;font-size:0.82em;white-space:nowrap">'
                        f'{sheet}</span>')

            trs = []
            for item in inconsistencies:
                by_variant: dict[str, list[str]] = {}
                for sheet, raw in item["variants"].items():
                    by_variant.setdefault(raw, []).append(sheet)
                parts = [
                    f'{variant} {" ".join(_badge(sh) for sh in sheets)}'
                    for variant, sheets in by_variant.items()
                ]
                detail = "<br>".join(parts)
                trs.append(
                    f'<tr style="border-bottom:1px solid #eee">'
                    f'<td style="padding:5px;vertical-align:top">{item["normalized"]}</td>'
                    f'<td style="padding:5px">{detail}</td></tr>'
                )
            html = (
                '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
                '<thead><tr style="border-bottom:2px solid #ccc;text-align:left">'
                '<th style="padding:6px">정규화</th>'
                '<th style="padding:6px">표기 · 사용 시트</th>'
                '</tr></thead><tbody>'
                + "".join(trs)
                + "</tbody></table>"
            )
            st.markdown(html, unsafe_allow_html=True)

    # ── Interactive brand mapping UI ─────────────────────────────────────
    # Combine fuzzy_suggested + unmatched pairs for unified editing
    all_candidates = []
    for fs in result.get("fuzzy_suggested", []):
        all_candidates.append({
            "집계": fs["brand_a"], "부과": fs["brand_b"],
            "건물": fs["building"], "유사도": fs["similarity"],
            "norm_a": fs["norm_a"], "norm_b": fs["norm_b"],
            "source": "fuzzy",
        })
    # Add unmatched pairs that share same building+unit
    for a_item in result.get("only_a", []):
        if a_item["brand"] in ("계약손실",) or "사무실" in a_item["brand"]:
            continue
        for b_item in result.get("only_b", []):
            if a_item["building"] == b_item["building"]:
                all_candidates.append({
                    "집계": a_item["brand"], "부과": b_item["brand"],
                    "건물": a_item["building"], "유사도": 0,
                    "norm_a": normalize_brand(a_item["brand"]),
                    "norm_b": normalize_brand(b_item["brand"]),
                    "source": "unmatched",
                })

    if all_candidates:
        import streamlit_antd_components as _sac_m
        with st.expander(
            f"🔧 브랜드 매칭 교정 (저장: {n_saved}건, 후보: {len(all_candidates)}건)",
            expanded=bool(result.get("fuzzy_suggested")),
        ):
            if n_saved:
                st.caption(f"현재 {n_saved}건의 동의어 매핑이 저장되어 자동 적용 중")

            # ─ Step 1: Select which candidates to process ────────────
            labels = [
                f"{c['집계']} ↔ {c['부과']} ({c['건물']}동"
                + (f" · {c['유사도']}%" if c['유사도'] else "") + ")"
                for c in all_candidates
            ]
            mode = _sac_m.segmented(
                [_sac_m.SegmentedItem(label="전체"),
                 _sac_m.SegmentedItem(label="선택"),
                 _sac_m.SegmentedItem(label="제외")],
                key=f"brand_map_mode{sfx}", use_container_width=True,
            )

            if mode == "전체":
                selected_idx = set(range(len(all_candidates)))
            elif mode == "선택":
                picks = st.multiselect(
                    "매칭할 브랜드", labels, default=labels,
                    key=f"brand_pick{sfx}",
                )
                selected_idx = {i for i, l in enumerate(labels) if l in picks}
            else:  # 제외
                excludes = st.multiselect(
                    "제외할 브랜드", labels, default=[],
                    key=f"brand_excl{sfx}",
                )
                excluded = {i for i, l in enumerate(labels) if l in excludes}
                selected_idx = set(range(len(all_candidates))) - excluded

            selected = [all_candidates[i] for i in sorted(selected_idx)]

            if not selected:
                st.info("선택된 항목이 없습니다.")
            else:
                # ─ Step 2: Name format ───────────────────────────────
                fmt = st.radio(
                    "대표 이름 형식",
                    ["정규화 (표준)", "집계 이름 기준", "부과 이름 기준", "직접 입력"],
                    horizontal=True, key=f"name_fmt{sfx}",
                )

                pending: dict[str, str] = {}

                if fmt == "직접 입력":
                    st.caption("각 브랜드별 대표 이름을 입력하세요")
                    for i, c in enumerate(selected):
                        _c1, _c2 = st.columns([5, 5])
                        with _c1:
                            st.text(f"{c['집계']} ↔ {c['부과']}")
                        with _c2:
                            custom = st.text_input(
                                "이름", value=c["norm_a"],
                                key=f"custom_name_{i}{sfx}",
                                label_visibility="collapsed",
                            )
                        cn = normalize_brand(custom) if custom else ""
                        if cn:
                            if c["norm_a"] != cn:
                                pending[c["norm_a"]] = cn
                            if c["norm_b"] != cn:
                                pending[c["norm_b"]] = cn
                else:
                    # Preview table
                    preview_rows = []
                    for c in selected:
                        if fmt == "정규화 (표준)":
                            canon = c["norm_a"]  # normalized form
                        elif fmt == "집계 이름 기준":
                            canon = normalize_brand(c["집계"])
                        else:
                            canon = normalize_brand(c["부과"])

                        preview_rows.append({
                            "집계": c["집계"], "부과": c["부과"],
                            "건물": c["건물"], "→ 대표": canon,
                        })
                        if c["norm_a"] != canon:
                            pending[c["norm_a"]] = canon
                        if c["norm_b"] != canon:
                            pending[c["norm_b"]] = canon

                    st.dataframe(
                        pd.DataFrame(preview_rows),
                        hide_index=True, use_container_width=True,
                    )

                # ─ Step 3: Save ──────────────────────────────────────
                _s1, _s2 = st.columns(2)
                with _s1:
                    if st.button("💾 저장", type="primary",
                                 disabled=not pending, key=f"save_synonyms{sfx}"):
                        merged = {**synonyms, **pending}
                        save_synonyms(merged)
                        st.success(f"{len(pending)}건 저장 (총 {len(merged)}건)")
                        st.rerun()
                with _s2:
                    if n_saved and st.button("🗑️ 초기화", key=f"reset_synonyms{sfx}"):
                        save_synonyms({})
                        st.success("초기화 완료")
                        st.rerun()

    # Unmatched brands (remaining after synonym + fuzzy)
    _col_ren = {"brand": "브랜드", "building": "건물", "unit": "호수"}
    if result["only_a"] or result["only_b"]:
        _ua, _ub = st.columns(2)
        for col, label, data in [(_ua, SH_A, result["only_a"]),
                                  (_ub, SH_B, result["only_b"])]:
            with col:
                st.caption(f"[{label}]에만 존재 ({len(data)}개)")
                if data:
                    st.dataframe(
                        pd.DataFrame(data).rename(columns=_col_ren),
                        hide_index=True, use_container_width=True)
                else:
                    st.success("없음")

    # Amount mismatches — vertical layout: 브랜드+동 on first row, rest below
    if result["amount_mismatches"]:
        st.warning(f"금액 불일치: {s['amount_mismatches']}건")
        am_raw = result["amount_mismatches"]
        # Group by (brand, building), preserve field order
        from collections import OrderedDict
        groups: OrderedDict[tuple, list] = OrderedDict()
        for m in am_raw:
            key = (m["brand_a"], m["building"])
            groups.setdefault(key, []).append(m)
        rows = []
        for (brand, bldg), items in groups.items():
            for idx, m in enumerate(items):
                rows.append({
                    "브랜드": brand if idx == 0 else "",
                    "동": bldg if idx == 0 else "",
                    "항목": m["field"],
                    "집계": f"{m['a_value']:,.0f}",
                    "부과": f"{m['b_value']:,.0f}",
                    "차이": f"{m['a_value'] - m['b_value']:+,.0f}",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    elif s["exact_match"] + s["fuzzy_match"] > 0:
        st.success("매칭된 모든 브랜드의 전용/공용/합계 금액 일치")


# ── Public render ─────────────────────────────────────────────────────────────

def render_anomaly_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    all_sheet_keys: list[str],
    split_by_building: bool = True,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
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

    # ── 0. Key insight summary — quick overview from cross-analysis ──────────
    _render_insight_summary(anomaly_df, sheets)

    # ── 1. KPI row — "How many problems?" ────────────────────────────────────
    _render_kpis(anomaly_df, has_billing, has_elec)

    # ── 2. Master table + visual ranking — unified investigation view ─────────
    st.subheader("🔍 조사 대상 브랜드")

    _n = st.slider("표시 브랜드 수", 10, len(anomaly_df),
                   min(20, len(anomaly_df)), key="anom_n")

    _col_bar, _col_heat = st.columns(2)
    with _col_bar:
        _render_composite_bar(anomaly_df, _n, split_by_building)
    with _col_heat:
        _render_heatmap(anomaly_df, _n)

    st.caption("복합 이상 점수 순으로 정렬 — **이유** 컬럼에서 각 브랜드가 왜 플래그되었는지 확인하세요.")

    id_cols    = [c for c in ["brand", "building", "floor"] if c in anomaly_df.columns]
    reason_col = ["reason"] if "reason" in anomaly_df.columns else []
    key_cols   = [c for c in ["composite_score", "risk_level",
                              "spike_max_pct", "spike_worst_util",
                              "spike_bldg_avg_pct", "spike_peer_ratio"] if c in anomaly_df.columns]
    master_show = id_cols + key_cols + reason_col
    master_view = add_display_index(anomaly_df[master_show])
    _col_cfg = {
        "No":                  st.column_config.NumberColumn("No", width="small"),
        "brand":               st.column_config.TextColumn("브랜드", width="medium"),
        "building":            st.column_config.TextColumn("건물", width="small"),
        "floor":               st.column_config.TextColumn("층", width="small"),
        "composite_score":     st.column_config.ProgressColumn(
            "복합 점수", format="%.3f", min_value=0, max_value=1, width="small"),
        "risk_level":          st.column_config.TextColumn("위험도", width="small"),
        "spike_max_pct":       st.column_config.NumberColumn(
            "최대 증가율(%)", format="%.1f", width="small"),
        "spike_worst_util":    st.column_config.TextColumn("급등 항목", width="small"),
        "spike_bldg_avg_pct":  st.column_config.NumberColumn(
            "건물평균(%)", format="%.1f", width="small"),
        "spike_peer_ratio":    st.column_config.NumberColumn(
            "vs건물", format="%.1fx", width="small"),
        "reason":              st.column_config.TextColumn("이유", width="large"),
    }
    st.dataframe(
        master_view,
        column_config=_col_cfg,
        hide_index=True,
        use_container_width=True,
    )
    download_df_as_excel(master_view, filename="anomaly_investigation.xlsx", sheet_name="조사대상")

    # ── Brand → profile shortcut ──────────────────────────────────────────
    _brand_list = anomaly_df["brand"].tolist()
    _pc1, _pc2 = st.columns([3, 1])
    with _pc1:
        _sel_brand = st.selectbox(
            "🏢 브랜드 프로필 보기",
            [""] + _brand_list,
            key="anom_goto_brand",
            label_visibility="collapsed",
            placeholder="브랜드를 선택하면 프로필로 이동합니다…",
        )
    with _pc2:
        if st.button("🏢 프로필 이동", disabled=not _sel_brand,
                     key="anom_goto_btn"):
            st.session_state["_goto_profile_brand"] = _sel_brand
            st.rerun()

    st.divider()

    # ── 4. Detail deep-dives — unique signals only ────────────────────────────
    # Build period file list for comparison tabs
    _period_files = [("📋 현재", file_data, all_sheet_keys, file_name)]
    if prev_file_data and prev_sheet_keys:
        _period_files.append((f"📈 {prev_label or '전월'}", prev_file_data, prev_sheet_keys, None))
    if yoy_file_data and yoy_sheet_keys:
        _period_files.append((f"📅 {yoy_label or '전년'}", yoy_file_data, yoy_sheet_keys, None))

    tab_spike, tab_chk = st.tabs(["📈 급등 감지", "🔍 일관성 검사"])

    with tab_spike:
        if len(_period_files) > 1:
            spike_tabs = st.tabs([pf[0] for pf in _period_files])
        else:
            spike_tabs = [st.container()]
        for pi, (plabel, pdata, psheets, pfname) in enumerate(_period_files):
            with spike_tabs[pi]:
                if pi == 0:
                    _render_spike_tab(anomaly_df, split_by_building,
                                      key_suffix="_p0")
                else:
                    _render_period_spike(pdata, psheets, plabel,
                                         split_by_building,
                                         key_suffix=f"_p{pi}")

    with tab_chk:
        _render_consistency_tab(
            anomaly_df, file_name, file_data, all_sheet_keys,
            prev_file_data=prev_file_data,
            prev_sheet_keys=prev_sheet_keys,
            prev_label=prev_label,
            yoy_file_data=yoy_file_data,
            yoy_sheet_keys=yoy_sheet_keys,
            yoy_label=yoy_label,
        )

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
| **일관성** | 사용량=0 유틸리티 항목 수 정규화 | 검침내역 |

**위험 등급**: 🔴 위험 ≥ 0.65 · 🟠 주의 ≥ 0.40 · 🟡 관찰 ≥ 0.20 · 🟢 정상 < 0.20

**동종 비교 (vs건물)**: 같은 건물 내 다른 브랜드 평균 급등률 대비 배수. 2x 이상 = 건물 전체 추세가 아닌 해당 브랜드만의 이상.
        """)

    with st.expander("📊 원시 데이터", expanded=False):
        st.dataframe(anomaly_df.reset_index(drop=True), hide_index=True, use_container_width=True)

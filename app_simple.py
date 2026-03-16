"""app_simple.py — 유틸리티 청구 관리 (Simplified Billing Dashboard).

Operator-focused: find billing errors, resolve complaints, send correct invoices.
No composite scores, no tabs within tabs, no statistical jargon.

Run: streamlit run app_simple.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sidebar import setup_sidebar
from data import (
    get_sheet_names, get_billing_period, to_numeric_series,
    read_billing_sheet, BILLING_SHEET_NAME,
)
from features import (
    apply_header_rows, build_from_two_files,
    create_change_columns, aggregate_by_brand,
)
from meter_view import load_raw_meter_df
from utils import (
    fmt_won, fmt_num, display_brand, add_per_area_cols,
    UTIL_LABELS, UTIL_UNITS, UTIL_PREFIXES,
)

# ── Constants ────────────────────────────────────────────────────────────────
_C_PREV = "#A8C4E0"
_C_CURR = "#4C72B0"
_C_AVG  = "#DD8A00"

_STATUS = {
    "error": ("⛔ 검침오류", "#C44E52"),
    "spike": ("🔴 급등",     "#C44E52"),
    "watch": ("🟡 확인필요",  "#DD8A00"),
    "ok":    ("🟢 정상",     "#2ca02c"),
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_data(uploads):
    """Load current + previous meter data and optional billing."""
    file_map = {}
    for f in uploads:
        file_map[f.name] = f.getvalue()

    # Sort files by billing period
    periods = {}
    for fname, fdata in file_map.items():
        sheets = get_sheet_names(fname, fdata)
        bp = get_billing_period(fname, fdata, sheets)
        periods[fname] = (bp or fname, sheets)

    sorted_files = sorted(periods.keys(), key=lambda f: periods[f][0])
    if not sorted_files:
        return None

    cur_file = sorted_files[-1]
    prev_file = sorted_files[-2] if len(sorted_files) >= 2 else None

    cur_sheets = periods[cur_file][1]
    meter_sheet = next((s for s in cur_sheets if s.strip() == "검침 내역"), None)
    if not meter_sheet:
        st.error("검침 내역 시트를 찾을 수 없습니다.")
        return None

    prev_meter_sheet = None
    if prev_file:
        prev_sheets = periods[prev_file][1]
        prev_meter_sheet = next((s for s in prev_sheets if s.strip() == "검침 내역"), None)

    # Load raw (pre-aggregation) for backward detection
    raw_df = load_raw_meter_df(
        cur_file, file_map, meter_sheet,
        prev_file_name=prev_file, prev_sheet_name=prev_meter_sheet,
    )

    # Aggregate
    agg_df = aggregate_by_brand(raw_df)
    add_per_area_cols(agg_df)
    agg_df = display_brand(agg_df)

    # Billing
    billing_df = None
    bill_key = next((s for s in cur_sheets if s.strip() == BILLING_SHEET_NAME), None)
    if bill_key:
        try:
            billing_df = read_billing_sheet(cur_file, file_map[cur_file], bill_key)
        except Exception:
            pass

    return {
        "agg_df": agg_df,
        "raw_df": raw_df,
        "billing_df": billing_df,
        "period": periods[cur_file][0],
        "prev_period": periods[prev_file][0] if prev_file else None,
    }


# ── Backward meter detection ────────────────────────────────────────────────

def _detect_backward(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Find rows where current meter reading < previous (physically impossible)."""
    pairs = [
        ("water",  "water_previous",  "water_current",  "m³"),
        ("hwater", "hwater_previous", "hwater_current", "m³"),
        ("elect",  "elect_previous",  "elect_current",  "kWh"),
        ("heat",   "heat_previous",   "heat_current",   "m³/MWh"),
    ]
    rows = []
    for pfx, prev_col, cur_col, unit in pairs:
        if prev_col not in raw_df.columns or cur_col not in raw_df.columns:
            continue
        p = to_numeric_series(raw_df[prev_col])
        c = to_numeric_series(raw_df[cur_col])
        mask = c.notna() & p.notna() & (c < p)
        for idx in raw_df[mask].index:
            brand_col = "brand_raw" if "brand_raw" in raw_df.columns else "brand"
            rows.append({
                "업체명": raw_df.at[idx, brand_col] if brand_col in raw_df.columns else "",
                "건물": raw_df.at[idx, "building"] if "building" in raw_df.columns else "",
                "항목": UTIL_LABELS.get(pfx, pfx),
                "이전": f"{float(p.at[idx]):,.1f} {unit}",
                "현재": f"{float(c.at[idx]):,.1f} {unit}",
                "차이": f"{float(c.at[idx] - p.at[idx]):+,.1f}",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Alert table builder ──────────────────────────────────────────────────────

def _build_alerts(agg_df: pd.DataFrame, backward_brands: set) -> pd.DataFrame:
    """Build one-row-per-brand alert table with plain status and reason."""
    rows = []
    for _, r in agg_df.iterrows():
        brand = r.get("brand", "")
        bldg = r.get("building", "")

        worst_status = "ok"
        worst_util = ""
        worst_pct = 0.0
        worst_cur = 0.0
        worst_prev = 0.0
        reasons: list[str] = []

        # Count zero-usage utilities
        n_zero = 0
        for pfx in UTIL_PREFIXES:
            cur_col = f"{pfx}_current"
            if cur_col in agg_df.columns:
                cv = float(r.get(cur_col, 0) or 0)
                if cv == 0:
                    n_zero += 1

        for pfx in UTIL_PREFIXES:
            pct_col = f"{pfx}_pct"
            cur_col = f"{pfx}_current"
            prev_col = f"{pfx}_previous"

            pct = float(r.get(pct_col, 0) or 0) if pct_col in agg_df.columns else 0
            cur = float(r.get(cur_col, 0) or 0) if cur_col in agg_df.columns else 0
            prev = float(r.get(prev_col, 0) or 0) if prev_col in agg_df.columns else 0
            lbl = UTIL_LABELS.get(pfx, pfx)

            # Determine status for this utility
            if brand in backward_brands:
                status = "error"
                reasons.append(f"{lbl} 검침오류")
            elif cur == 0 and prev > 0:
                status = "spike"
                reasons.append(f"{lbl} 미계량(전월 {prev:,.0f})")
            elif abs(pct) >= 100:
                status = "spike"
                reasons.append(f"{lbl} {pct:+.0f}%")
            elif abs(pct) >= 30:
                status = "watch"
                reasons.append(f"{lbl} {pct:+.0f}%")
            else:
                status = "ok"

            # Keep the worst
            _rank = {"error": 3, "spike": 2, "watch": 1, "ok": 0}
            if _rank[status] > _rank[worst_status] or (
                _rank[status] == _rank[worst_status] and abs(pct) > abs(worst_pct)
            ):
                worst_status = status
                worst_util = lbl
                worst_pct = pct
                worst_cur = cur
                worst_prev = prev

        # All-zero = likely vacancy or total data error
        _n_avail = len([p for p in UTIL_PREFIXES if f"{p}_current" in agg_df.columns])
        if n_zero >= _n_avail and _n_avail > 0:
            worst_status = "error"
            reasons = ["전 항목 미계량 (공실/데이터 누락)"]

        lbl, _ = _STATUS[worst_status]
        rows.append({
            "상태": lbl,
            "업체명": brand,
            "건물": bldg,
            "항목": worst_util,
            "이번달": round(worst_cur, 1),
            "전월": round(worst_prev, 1),
            "변화(%)": round(worst_pct, 1),
            "사유": " · ".join(reasons[:3]) if reasons else "—",
            "_status": worst_status,
        })

    df = pd.DataFrame(rows)
    _order = {"error": 0, "spike": 1, "watch": 2, "ok": 3}
    df["_sort"] = df["_status"].map(_order)
    df = df.sort_values(["_sort", "변화(%)"], ascending=[True, False]).reset_index(drop=True)
    return df


# ── Tenant detail view ───────────────────────────────────────────────────────

def _render_detail(brand: str, building: str, agg_df: pd.DataFrame,
                   backward_brands: set, billing_df: pd.DataFrame | None):
    """Single-tenant investigation page."""

    # Back button
    if st.button("← 목록으로 돌아가기", type="primary"):
        st.session_state["_simple_view"] = "list"
        st.rerun()

    # Find this brand's data
    mask = agg_df["brand"] == brand
    if building:
        mask = mask & (agg_df["building"] == building)
    brand_row = agg_df[mask]
    if brand_row.empty:
        st.error(f"'{brand}' 데이터를 찾을 수 없습니다.")
        return
    r = brand_row.iloc[0]

    # Header
    bldg = r.get("building", "")
    floor = r.get("floor", "")
    size_py = r.get("size_py", "")
    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<div style="font-size:1.6rem;font-weight:800">{brand}</div>'
        f'<div style="font-size:0.9rem;color:inherit;opacity:0.6">'
        f'🏢 {bldg}동 · 📍 {floor} · 📐 {size_py}평</div></div>',
        unsafe_allow_html=True,
    )

    # ── Status badges per utility ─────────────────────────────────────
    _badge_cols = st.columns(len(UTIL_PREFIXES))
    for i, pfx in enumerate(UTIL_PREFIXES):
        lbl = UTIL_LABELS.get(pfx, pfx)
        pct = float(r.get(f"{pfx}_pct", 0) or 0)

        if brand in backward_brands:
            status_lbl, status_clr = _STATUS["error"]
        elif abs(pct) >= 100:
            status_lbl, status_clr = _STATUS["spike"]
        elif abs(pct) >= 30:
            status_lbl, status_clr = _STATUS["watch"]
        else:
            status_lbl, status_clr = _STATUS["ok"]

        with _badge_cols[i]:
            st.markdown(
                f'<div style="background:{status_clr}12;border:1px solid {status_clr}30;'
                f'border-radius:8px;padding:10px;text-align:center">'
                f'<div style="font-size:0.75rem;color:inherit;opacity:0.6">{lbl}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{status_clr}">'
                f'{pct:+.1f}%</div>'
                f'<div style="font-size:0.7rem;color:{status_clr}">{status_lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── This month vs last month ──────────────────────────────────────
    st.markdown("#### 이번달 vs 전월")
    _utils = []
    _cur_vals = []
    _prev_vals = []
    for pfx in UTIL_PREFIXES:
        cur_col = f"{pfx}_current"
        prev_col = f"{pfx}_previous"
        if cur_col in agg_df.columns:
            _utils.append(UTIL_LABELS.get(pfx, pfx))
            _cur_vals.append(float(r.get(cur_col, 0) or 0))
            _prev_vals.append(float(r.get(prev_col, 0) or 0))

    if _utils:
        # Separate chart per utility (different scales)
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=len(_utils), subplot_titles=_utils)
        for i, (u, cv, pv) in enumerate(zip(_utils, _cur_vals, _prev_vals)):
            fig.add_trace(go.Bar(
                x=["전월", "이번달"], y=[pv, cv],
                marker_color=[_C_PREV, _C_CURR],
                text=[f"{pv:,.1f}", f"{cv:,.1f}"],
                textposition="outside", textfont=dict(size=10),
                showlegend=False,
            ), row=1, col=i+1)
            pfx = list(UTIL_PREFIXES)[i]
            fig.update_yaxes(title_text=UTIL_UNITS.get(pfx, ""), title_font_size=9,
                             row=1, col=i+1)
        fig.update_layout(height=300, margin=dict(t=35, b=20, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True, key="detail_mom")

    # ── vs Building average ───────────────────────────────────────────
    st.markdown("#### 건물 내 비교")
    if "building" in agg_df.columns and bldg:
        bldg_df = agg_df[agg_df["building"] == bldg]
        _b_utils = []
        _brand_vals = []
        _avg_vals = []
        for pfx in UTIL_PREFIXES:
            cur_col = f"{pfx}_current"
            if cur_col in agg_df.columns:
                _b_utils.append(UTIL_LABELS.get(pfx, pfx))
                _brand_vals.append(float(r.get(cur_col, 0) or 0))
                _avg_vals.append(float(to_numeric_series(bldg_df[cur_col]).mean() or 0))

        if _b_utils:
            fig2 = make_subplots(rows=1, cols=len(_b_utils), subplot_titles=_b_utils)
            for i, (u, bv, av) in enumerate(zip(_b_utils, _brand_vals, _avg_vals)):
                fig2.add_trace(go.Bar(
                    x=["건물평균", brand[:8]], y=[av, bv],
                    marker_color=[_C_AVG, _C_CURR],
                    text=[f"{av:,.1f}", f"{bv:,.1f}"],
                    textposition="outside", textfont=dict(size=10),
                    showlegend=False,
                ), row=1, col=i+1)
                pfx = list(UTIL_PREFIXES)[i]
                fig2.update_yaxes(title_text=UTIL_UNITS.get(pfx, ""), title_font_size=9,
                                  row=1, col=i+1)
            fig2.update_layout(height=300, margin=dict(t=35, b=20, l=10, r=10))
            st.plotly_chart(fig2, use_container_width=True, key="detail_peer")

    # ── Bill breakdown ────────────────────────────────────────────────
    if billing_df is not None and not billing_df.empty:
        bill_mask = billing_df["brand"] == brand if "brand" in billing_df.columns else pd.Series(False, index=billing_df.index)
        if "building" in billing_df.columns and bldg:
            bill_mask = bill_mask & (billing_df["building"] == bldg)
        brand_bill = billing_df[bill_mask]
        if not brand_bill.empty:
            st.markdown("#### 청구 내역")
            _bill_rows = []
            _BILL_ITEMS = [
                ("수도", "water_excl", "water_comm", "water_total"),
                ("전기", "elect_excl", "elect_comm", "elect_total"),
                ("난방", "heat_total", None, None),
            ]
            for item_lbl, *cols in _BILL_ITEMS:
                row_data = {"항목": item_lbl}
                for col in cols:
                    if col and col in brand_bill.columns:
                        v = float(brand_bill.iloc[0].get(col, 0) or 0)
                        if "excl" in col:
                            row_data["전용"] = fmt_won(v * 10000)
                        elif "comm" in col:
                            row_data["공용"] = fmt_won(v * 10000)
                        elif "total" in col:
                            row_data["합계"] = fmt_won(v * 10000)
                if len(row_data) > 1:
                    _bill_rows.append(row_data)
            if _bill_rows:
                st.dataframe(pd.DataFrame(_bill_rows), hide_index=True, use_container_width=True)

    # ── Action buttons ────────────────────────────────────────────────
    st.divider()
    _actions = st.session_state.get("_action_log", {})
    _existing = _actions.get(f"{brand}_{bldg}", "")

    if _existing:
        st.info(f"이전 처리: **{_existing}**")

    _a1, _a2, _a3 = st.columns(3)
    with _a1:
        if st.button("✅ 정상 확인", use_container_width=True, key="act_ok"):
            _actions[f"{brand}_{bldg}"] = "정상 확인"
            st.session_state["_action_log"] = _actions
            st.success(f"{brand}: 정상 확인 처리됨")
    with _a2:
        if st.button("🔄 검침 재확인", use_container_width=True, key="act_recheck"):
            _actions[f"{brand}_{bldg}"] = "검침 재확인"
            st.session_state["_action_log"] = _actions
            st.warning(f"{brand}: 검침 재확인 요청됨")
    with _a3:
        if st.button("⏸ 청구 보류", use_container_width=True, key="act_hold"):
            _actions[f"{brand}_{bldg}"] = "청구 보류"
            st.session_state["_action_log"] = _actions
            st.error(f"{brand}: 청구 보류 처리됨")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="유틸리티 청구 관리", layout="wide", page_icon="🧾")
    st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; }
header[data-testid="stHeader"] { height: 0; }
h1 { font-weight: 800 !important; font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

    uploads, bins, tail, q = setup_sidebar()
    if not uploads:
        st.title("🧾 유틸리티 청구 관리")
        st.info("사이드바에서 데이터 폴더를 지정해주세요.")
        return

    data = _load_data(uploads)
    if data is None:
        return

    agg_df = data["agg_df"]
    raw_df = data["raw_df"]
    billing_df = data["billing_df"]

    # Detect backward readings
    bw_df = _detect_backward(raw_df)
    backward_brands = set(bw_df["업체명"].unique()) if not bw_df.empty else set()

    # ── Detail view ───────────────────────────────────────────────────
    if st.session_state.get("_simple_view") == "detail":
        _brand = st.session_state.get("_simple_brand", "")
        _bldg = st.session_state.get("_simple_building", "")
        _render_detail(_brand, _bldg, agg_df, backward_brands, billing_df)
        return

    # ── List view (landing) ───────────────────────────────────────────
    st.title("🧾 유틸리티 청구 관리")
    if data["period"]:
        st.caption(f"기간: {data['period']}" +
                   (f" (전월: {data['prev_period']})" if data["prev_period"] else ""))

    # Data errors — hard block
    if not bw_df.empty:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#C44E5225,#C44E5215);'
            f'border:2px solid #C44E5260;border-radius:10px;padding:14px 18px;margin-bottom:16px">'
            f'<div style="font-size:1rem;font-weight:800;color:#C44E52;margin-bottom:8px">'
            f'⛔ 검침 오류 {len(bw_df)}건 — 아래 업체의 청구서를 보류하세요</div></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(bw_df, hide_index=True, use_container_width=True)
        st.divider()

    # Search bar
    search = st.text_input(
        "🔍 업체 검색", placeholder="업체명을 입력하세요...",
        key="_simple_search", label_visibility="collapsed",
    ).strip().lower()

    # Build alert table
    alerts = _build_alerts(agg_df, backward_brands)
    if search:
        alerts = alerts[alerts["업체명"].str.lower().str.contains(search, na=False)]

    # Display
    display_df = alerts[["상태", "업체명", "건물", "항목", "이번달", "전월", "변화(%)", "사유"]].copy()

    # KPI row
    n_error = int((alerts["_status"] == "error").sum())
    n_spike = int((alerts["_status"] == "spike").sum())
    n_watch = int((alerts["_status"] == "watch").sum())
    n_ok = int((alerts["_status"] == "ok").sum())

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("⛔ 검침오류", f"{n_error}개")
    _k2.metric("🔴 급등", f"{n_spike}개")
    _k3.metric("🟡 확인필요", f"{n_watch}개")
    _k4.metric("🟢 정상", f"{n_ok}개")

    # Table with row selection
    ev = st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="alert_table",
    )

    # Handle row click
    if ev and ev.selection and ev.selection.rows:
        idx = ev.selection.rows[0]
        if idx < len(alerts):
            sel = alerts.iloc[idx]
            st.session_state["_simple_view"] = "detail"
            st.session_state["_simple_brand"] = sel["업체명"]
            st.session_state["_simple_building"] = sel["건물"]
            st.rerun()


if __name__ == "__main__":
    main()

"""app_simple.py — 청구 검증 도구 (Bill Verification Tool).

Pre-send checklist: upload Excel → see which bills look wrong → fix → send.
Sits between "Excel calculates bills" and "bills go out."

Run: streamlit run app_simple.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    fmt_won, display_brand, add_per_area_cols,
    UTIL_LABELS, UTIL_UNITS, UTIL_PREFIXES,
)

# ── Colors ───────────────────────────────────────────────────────────────────
_RED = "#C44E52"
_AMBER = "#DD8A00"
_GREEN = "#2ca02c"
_BLUE = "#4C72B0"
_LIGHT_BLUE = "#A8C4E0"


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="데이터 로드 중…")
def _load_data(_uploads_key, file_map):
    """Load current + previous meter data and billing."""
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
        return None

    prev_meter_sheet = None
    if prev_file:
        prev_sheets = periods[prev_file][1]
        prev_meter_sheet = next((s for s in prev_sheets if s.strip() == "검침 내역"), None)

    raw_df = load_raw_meter_df(
        cur_file, file_map, meter_sheet,
        prev_file_name=prev_file, prev_sheet_name=prev_meter_sheet,
    )
    agg_df = aggregate_by_brand(raw_df)
    add_per_area_cols(agg_df)
    agg_df = display_brand(agg_df)

    billing_df = None
    bill_key = next((s for s in cur_sheets if s.strip() == BILLING_SHEET_NAME), None)
    if bill_key:
        try:
            billing_df = read_billing_sheet(cur_file, file_map[cur_file], bill_key)
        except Exception:
            pass

    return {
        "agg_df": agg_df, "raw_df": raw_df, "billing_df": billing_df,
        "period": periods[cur_file][0],
        "prev_period": periods[prev_file][0] if prev_file else None,
    }


# ── Issue detection ──────────────────────────────────────────────────────────

def _detect_all_issues(agg_df, raw_df):
    """Detect all billing issues. Returns list of dicts, one per issue."""
    issues = []

    # 1. Backward meter readings
    _pairs = [
        ("water", "water_previous", "water_current"),
        ("hwater", "hwater_previous", "hwater_current"),
        ("elect", "elect_previous", "elect_current"),
        ("heat", "heat_previous", "heat_current"),
    ]
    for pfx, prev_col, cur_col in _pairs:
        if prev_col not in raw_df.columns or cur_col not in raw_df.columns:
            continue
        p = to_numeric_series(raw_df[prev_col])
        c = to_numeric_series(raw_df[cur_col])
        mask = c.notna() & p.notna() & (c < p)
        for idx in raw_df[mask].index:
            brand_col = "brand_raw" if "brand_raw" in raw_df.columns else "brand"
            issues.append({
                "업체명": raw_df.at[idx, brand_col] if brand_col in raw_df.columns else "",
                "건물": raw_df.at[idx, "building"] if "building" in raw_df.columns else "",
                "유형": "⛔ 검침 오류",
                "항목": UTIL_LABELS.get(pfx, pfx),
                "내용": f"현재({float(c.at[idx]):,.1f}) < 이전({float(p.at[idx]):,.1f})",
                "조치": "검침값 입력 오류 확인 — 숫자 바꿔 입력했을 가능성",
                "severity": 3,
            })

    # 2. Per-brand checks from aggregated data
    _n_avail = len([p for p in UTIL_PREFIXES if f"{p}_current" in agg_df.columns])
    for _, r in agg_df.iterrows():
        brand = r.get("brand", "")
        bldg = r.get("building", "")

        # All-zero (vacancy/total data loss)
        n_zero = sum(
            1 for pfx in UTIL_PREFIXES
            if f"{pfx}_current" in agg_df.columns
            and float(r.get(f"{pfx}_current", 0) or 0) == 0
        )
        if n_zero >= _n_avail and _n_avail > 0:
            issues.append({
                "업체명": brand, "건물": bldg,
                "유형": "⛔ 전항목 미계량",
                "항목": "전체",
                "내용": f"4개 항목 모두 사용량 0",
                "조치": "공실 여부 확인 — 공실 아니면 검침 누락",
                "severity": 3,
            })
            continue

        for pfx in UTIL_PREFIXES:
            pct_col = f"{pfx}_pct"
            cur_col = f"{pfx}_current"
            prev_col = f"{pfx}_previous"
            if pct_col not in agg_df.columns:
                continue

            pct = float(r.get(pct_col, 0) or 0)
            cur = float(r.get(cur_col, 0) or 0)
            prev = float(r.get(prev_col, 0) or 0)
            lbl = UTIL_LABELS.get(pfx, pfx)
            unit = UTIL_UNITS.get(pfx, "")

            # Was active, now zero
            if cur == 0 and prev > 0:
                issues.append({
                    "업체명": brand, "건물": bldg,
                    "유형": "🔴 미계량 전환",
                    "항목": lbl,
                    "내용": f"전월 {prev:,.1f}{unit} → 이번달 0",
                    "조치": "계량기 고장 또는 검침 누락 확인",
                    "severity": 2,
                })
            # Huge spike
            elif pct >= 200:
                issues.append({
                    "업체명": brand, "건물": bldg,
                    "유형": "🔴 급등",
                    "항목": lbl,
                    "내용": f"{prev:,.1f} → {cur:,.1f}{unit} (+{pct:.0f}%)",
                    "조치": "검침값 자릿수 오류 가능성 — 원본 대조 필요",
                    "severity": 2,
                })
            elif pct >= 100:
                issues.append({
                    "업체명": brand, "건물": bldg,
                    "유형": "🔴 급등",
                    "항목": lbl,
                    "내용": f"{prev:,.1f} → {cur:,.1f}{unit} (+{pct:.0f}%)",
                    "조치": "검침값 확인 — 누수 또는 입력 오류 가능성",
                    "severity": 2,
                })
            # Large drop (possible wrong entry)
            elif pct <= -50 and prev > 0:
                issues.append({
                    "업체명": brand, "건물": bldg,
                    "유형": "🟡 급감",
                    "항목": lbl,
                    "내용": f"{prev:,.1f} → {cur:,.1f}{unit} ({pct:.0f}%)",
                    "조치": "검침값 확인 — 이전 달 과다 계상 또는 이번 달 누락",
                    "severity": 1,
                })
            # Moderate spike
            elif pct >= 50:
                issues.append({
                    "업체명": brand, "건물": bldg,
                    "유형": "🟡 증가",
                    "항목": lbl,
                    "내용": f"{prev:,.1f} → {cur:,.1f}{unit} (+{pct:.0f}%)",
                    "조치": "실제 사용 증가인지 확인",
                    "severity": 1,
                })

    return sorted(issues, key=lambda x: (-x["severity"], x["업체명"]))


# ── Render: issue checklist ──────────────────────────────────────────────────

def _render_checklist(issues, agg_df, billing_df):
    """The main checklist view — one card per issue."""
    if not issues:
        st.balloons()
        st.success(
            "✅ **청구서 검증 완료 — 이상 없음**\n\n"
            "모든 검침값이 정상 범위입니다. 청구서를 발송해도 됩니다."
        )
        return

    # Summary badges
    n_critical = sum(1 for i in issues if i["severity"] >= 2)
    n_watch = sum(1 for i in issues if i["severity"] == 1)
    n_brands = len(set(i["업체명"] for i in issues))

    st.markdown(
        f'<div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap">'
        f'<span style="background:{_RED}18;color:{_RED};border:1px solid {_RED}40;'
        f'border-radius:20px;padding:6px 16px;font-weight:700;font-size:0.88rem">'
        f'⛔ 청구 보류 {n_critical}건</span>'
        f'<span style="background:{_AMBER}18;color:{_AMBER};border:1px solid {_AMBER}40;'
        f'border-radius:20px;padding:6px 16px;font-weight:700;font-size:0.88rem">'
        f'🟡 확인 필요 {n_watch}건</span>'
        f'<span style="background:rgba(128,128,128,0.08);color:inherit;opacity:0.6;'
        f'border-radius:20px;padding:6px 16px;font-size:0.85rem">'
        f'{n_brands}개 업체</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Action log
    _actions = st.session_state.get("_action_log", {})

    # Render each issue as a card
    for idx, issue in enumerate(issues):
        _key = f"{issue['업체명']}_{issue['건물']}_{issue['항목']}_{idx}"
        _done = _actions.get(_key, "")
        _sev_clr = _RED if issue["severity"] >= 2 else _AMBER
        _opacity = "0.5" if _done else "1"

        with st.container(border=True):
            _c1, _c2, _c3 = st.columns([4, 2, 2])

            with _c1:
                st.markdown(
                    f'<div style="opacity:{_opacity}">'
                    f'<span style="color:{_sev_clr};font-weight:700;font-size:0.88rem">'
                    f'{issue["유형"]}</span> '
                    f'<span style="font-weight:700;font-size:0.95rem">'
                    f'{issue["업체명"]}</span> '
                    f'<span style="opacity:0.5;font-size:0.82rem">'
                    f'{issue["건물"]}동 · {issue["항목"]}</span>'
                    f'<div style="font-size:0.85rem;margin-top:4px">{issue["내용"]}</div>'
                    f'<div style="font-size:0.78rem;color:inherit;opacity:0.55;margin-top:2px">'
                    f'💡 {issue["조치"]}</div>'
                    f'{"<div style=font-size:0.75rem;color:" + _GREEN + ";margin-top:4px>✅ " + _done + "</div>" if _done else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with _c2:
                # Mini MoM bar for this brand+utility
                _brand_data = agg_df[agg_df["brand"] == issue["업체명"]]
                if not _brand_data.empty:
                    _r = _brand_data.iloc[0]
                    _pfx = next((p for p, l in UTIL_LABELS.items() if l == issue["항목"]), None)
                    if _pfx:
                        _cur = float(_r.get(f"{_pfx}_current", 0) or 0)
                        _prev = float(_r.get(f"{_pfx}_previous", 0) or 0)
                        _unit = UTIL_UNITS.get(_pfx, "")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=["전월", "이번달"], y=[_prev, _cur],
                            marker_color=[_LIGHT_BLUE, _sev_clr],
                            text=[f"{_prev:,.0f}", f"{_cur:,.0f}"],
                            textposition="outside", textfont=dict(size=9),
                        ))
                        fig.update_layout(
                            height=120, margin=dict(t=5, b=5, l=5, r=5),
                            yaxis=dict(visible=False), xaxis=dict(tickfont=dict(size=9)),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"mini_{_key}")

            with _c3:
                if _done:
                    st.markdown(
                        f'<div style="text-align:center;padding-top:20px;color:{_GREEN};'
                        f'font-weight:700">✅ {_done}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button("✅ 확인완료", key=f"ok_{_key}", use_container_width=True):
                        _actions[_key] = "확인완료"
                        st.session_state["_action_log"] = _actions
                        st.rerun()
                    if st.button("⏸ 보류", key=f"hold_{_key}", use_container_width=True):
                        _actions[_key] = "청구 보류"
                        st.session_state["_action_log"] = _actions
                        st.rerun()

    # Summary at bottom
    st.divider()
    n_resolved = sum(1 for k, v in _actions.items() if v)
    n_total = len(issues)
    n_remaining = n_total - n_resolved

    if n_remaining == 0:
        st.success(f"✅ 모든 항목 처리 완료 ({n_total}건). 청구서 발송 가능합니다.")
    else:
        st.warning(f"⏳ {n_remaining}/{n_total}건 미처리. 모든 항목 확인 후 청구서를 발송하세요.")

    if _actions:
        with st.expander(f"📋 처리 이력 ({n_resolved}건)"):
            _log_rows = [{"항목": k, "처리": v} for k, v in _actions.items() if v]
            if _log_rows:
                st.dataframe(pd.DataFrame(_log_rows), hide_index=True, use_container_width=True)
            if st.button("🗑️ 이력 초기화"):
                st.session_state["_action_log"] = {}
                st.rerun()


# ── Render: overview stats ───────────────────────────────────────────────────

def _render_overview(agg_df, billing_df, period):
    """Quick stats bar — how the building is doing overall."""
    _cols = st.columns(len(UTIL_PREFIXES))
    for i, pfx in enumerate(UTIL_PREFIXES):
        lbl = UTIL_LABELS.get(pfx, pfx)
        pct_col = f"{pfx}_pct"
        cur_col = f"{pfx}_current"
        if pct_col not in agg_df.columns:
            continue
        avg_pct = to_numeric_series(agg_df[pct_col]).mean()
        total_cur = to_numeric_series(agg_df[cur_col]).sum() if cur_col in agg_df.columns else 0
        unit = UTIL_UNITS.get(pfx, "")

        if pd.isna(avg_pct):
            continue

        clr = _RED if avg_pct > 10 else _GREEN if avg_pct < -10 else "#888"
        with _cols[i]:
            st.metric(
                lbl,
                f"{avg_pct:+.1f}%",
                delta=f"총 {total_cur:,.0f} {unit}",
                delta_color="off",
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="청구 검증", layout="wide", page_icon="🧾")
    st.markdown("""
<style>
.block-container { padding-top: 1.2rem !important; }
header[data-testid="stHeader"] { height: 0; }
h1 { font-weight: 800 !important; font-size: 1.5rem !important; }
[data-testid="stMetricValue"] { font-size: 1.1rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

    uploads, bins, tail, q = setup_sidebar()
    if not uploads:
        st.title("🧾 청구 검증 도구")
        st.info("사이드바에서 데이터 폴더를 지정해주세요.")
        st.markdown(
            "**사용법**\n"
            "1. Excel 파일이 있는 폴더 경로 입력\n"
            "2. 자동으로 검침 오류, 급등, 미계량 감지\n"
            "3. 각 항목 확인 후 ✅ 또는 ⏸ 처리\n"
            "4. 모든 항목 처리 완료 시 청구서 발송"
        )
        return

    # Build file map
    file_map = {f.name: f.getvalue() for f in uploads}
    _cache_key = tuple(sorted(file_map.keys()))
    data = _load_data(_cache_key, file_map)

    if data is None:
        st.error("검침 내역 시트를 찾을 수 없습니다. Excel 파일을 확인해주세요.")
        return

    agg_df = data["agg_df"]
    raw_df = data["raw_df"]
    billing_df = data["billing_df"]

    # Title + period
    st.title("🧾 청구 검증 도구")
    if data["period"]:
        st.caption(
            f"📅 {data['period']}"
            + (f" vs {data['prev_period']}" if data["prev_period"] else "")
            + f" · {len(agg_df)}개 업체"
        )

    # Overview stats
    _render_overview(agg_df, billing_df, data["period"])
    st.divider()

    # Detect issues
    issues = _detect_all_issues(agg_df, raw_df)

    # Render checklist
    _render_checklist(issues, agg_df, billing_df)


if __name__ == "__main__":
    main()

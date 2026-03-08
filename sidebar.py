"""sidebar.py — Sidebar UI: file upload, presets, bins, tail %, debug toggle."""
import streamlit as st


def setup_sidebar():
    """Render sidebar and return (uploads, bins, tail, q_change, q_pct, debug)."""
    with st.sidebar:
        st.header("Upload")
        uploads = st.file_uploader(
            "Upload CSV/XLSX/Parquet",
            type=["csv", "xlsx", "xls", "xlsm", "parquet"],
            accept_multiple_files=True,
        )

        st.divider()
        st.header("⚙️ Settings")

        # ---- Presets ----
        preset_map = {"Default (20%)": 20, "Gentle (10%)": 10, "Dense (30%)": 30}

        def apply_preset():
            val = preset_map.get(st.session_state["preset_select"])
            if val is not None:
                st.session_state["tail"]       = val
                st.session_state["tail_input"] = val

        preset = st.selectbox(
            "⚡ Quick presets",
            ["Custom", "Default (20%)", "Gentle (10%)", "Dense (30%)"],
            index=0,
            key="preset_select",
            on_change=apply_preset,
            help="Pick a preset to quickly adjust tail percentage",
        )

        st.divider()

        # ---- Bins ----
        if "bins" not in st.session_state:
            st.session_state["bins"] = 50
        if "bins_input" not in st.session_state:
            st.session_state["bins_input"] = 50

        def sync_bins_slider():
            st.session_state["bins_input"] = st.session_state["bins"]

        def sync_bins_input():
            st.session_state["bins"] = st.session_state["bins_input"]

        b1, b2 = st.columns([3, 1])
        with b1:
            st.slider("Bins", 5, 200, step=1, key="bins", on_change=sync_bins_slider)
        with b2:
            st.number_input("Bins value", 5, 200, step=1, key="bins_input", label_visibility="hidden", on_change=sync_bins_input)
        bins = st.session_state["bins"]

        # ---- Tail % ----
        if "tail" not in st.session_state:
            st.session_state["tail"] = 20
        if "tail_input" not in st.session_state:
            st.session_state["tail_input"] = 20

        def sync_tail_slider():
            st.session_state["tail_input"] = st.session_state["tail"]

        def sync_tail_input():
            st.session_state["tail"] = st.session_state["tail_input"]

        t1, t2 = st.columns([3, 1])
        with t1:
            st.slider(
                "Tail %", 1, 50, step=1, key="tail",
                on_change=sync_tail_slider,
                help="Show the bottom N% and top N% of both change and pct values",
            )
        with t2:
            st.number_input("Tail value", 1, 50, step=1, key="tail_input", label_visibility="hidden", on_change=sync_tail_input)

        st.divider()
        debug = st.checkbox("Debug", value=False)

    tail = st.session_state["tail"]
    q_change = (tail / 100.0, 1.0 - tail / 100.0)
    q_pct = q_change
    bins = st.session_state["bins"]

    return uploads, bins, tail, q_change, q_pct, debug

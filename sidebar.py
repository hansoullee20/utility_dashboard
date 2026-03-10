"""sidebar.py — Sidebar UI: language, directory input, presets, bins, tail %, debug."""
import os
import glob as _glob
import streamlit as st
from lang import t


class _FileEntry:
    """Mimics Streamlit UploadedFile interface for locally loaded files."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def setup_sidebar():
    """Render sidebar and return (uploads, bins, tail, q, debug)."""
    with st.sidebar:
        # Language locked to Korean until dashboard is finalized
        st.session_state["lang"] = "ko"

        st.divider()
        st.header(t("upload"))
        dir_path = st.text_input(
            "데이터 폴더 경로",
            value="./data",
            key="dir_path_input",
        )

        uploads = []
        if dir_path:
            dir_path = os.path.abspath(dir_path)
            if os.path.isdir(dir_path):
                found = sorted(
                    f for pat in ("*.xlsx", "*.xlsm", "*.xls")
                    for f in _glob.glob(os.path.join(dir_path, pat))
                )
                if found:
                    for fpath in found:
                        try:
                            with open(fpath, "rb") as fh:
                                uploads.append(_FileEntry(os.path.basename(fpath), fh.read()))
                        except Exception as e:
                            st.warning(f"{os.path.basename(fpath)} 읽기 실패: {e}")
                    st.caption(f"{len(uploads)}개 파일 로드됨")
                else:
                    st.warning("폴더에 Excel 파일이 없습니다.")
            else:
                st.warning("폴더를 찾을 수 없습니다.")

        st.divider()
        st.header(t("settings"))

        # ---- Presets ----
        _preset_labels = [
            t("preset_custom"),
            t("preset_default"),
            t("preset_gentle"),
            t("preset_dense"),
        ]
        _preset_values = [None, 20, 10, 30]

        def apply_preset():
            idx = _preset_labels.index(st.session_state["preset_select"])
            val = _preset_values[idx]
            if val is not None:
                st.session_state["tail"]       = val
                st.session_state["tail_input"] = val

        st.selectbox(
            "⚡ " + t("quick_presets"),
            _preset_labels,
            index=0,
            key="preset_select",
            on_change=apply_preset,
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
            st.slider(t("bins"), 5, 200, step=1, key="bins", on_change=sync_bins_slider)
        with b2:
            st.number_input(t("bins"), 5, 200, step=1, key="bins_input",
                            label_visibility="hidden", on_change=sync_bins_input)
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
                t("tail_pct"), 1, 50, step=1, key="tail",
                on_change=sync_tail_slider,
                help=t("tail_help"),
            )
        with t2:
            st.number_input(t("tail_pct"), 1, 50, step=1, key="tail_input",
                            label_visibility="hidden", on_change=sync_tail_input)

        st.divider()
        debug = st.checkbox(t("debug"), value=False)

    tail = st.session_state["tail"]
    q = (tail / 100.0, 1.0 - tail / 100.0)
    bins = st.session_state["bins"]

    return uploads, bins, tail, q, debug

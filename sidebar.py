"""sidebar.py — Sidebar UI: directory input, debug."""
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
    """Render sidebar and return (uploads, bins, tail, q, debug, nav_placeholder)."""
    with st.sidebar:
        st.session_state["lang"] = "ko"

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

    tail = 20
    bins = 50
    q = (tail / 100.0, 1.0 - tail / 100.0)

    return uploads, bins, tail, q

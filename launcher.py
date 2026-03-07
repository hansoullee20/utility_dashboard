import sys
import os
import threading
import time
import webbrowser


def resource_path(rel):
    base = sys._MEIPASS if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def open_browser():
    time.sleep(5)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    threading.Thread(target=open_browser, daemon=True).start()

    from streamlit.web import bootstrap
    bootstrap.run(resource_path("app.py"), "", [], {})

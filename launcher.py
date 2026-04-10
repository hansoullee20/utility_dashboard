import os
import socket
import sys
import threading
import time
import webbrowser


def resource_path(rel):
    base = sys._MEIPASS if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def open_browser(url: str):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[-1])), timeout=1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)
    webbrowser.open(url)


if __name__ == "__main__":
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    from streamlit.web import bootstrap
    bootstrap.run(resource_path("app.py"), "", [], {})

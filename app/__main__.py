"""Launch Akane in popup or server mode with `python -m app`."""

from __future__ import annotations

import sys

from app.config import APP_MODE, SERVER_HOST, SERVER_PORT


def main() -> None:
    mode = str(sys.argv[1]).strip().lower() if len(sys.argv) > 1 else APP_MODE
    if mode == "server":
        from app.server import serve
        serve(host=SERVER_HOST, port=SERVER_PORT)
        return
    if mode == "popup":
        from app.popup import launch_popup
        launch_popup()
        return
    raise SystemExit(f"Unknown mode '{mode}'. Use 'popup' or 'server'.")


if __name__ == "__main__":
    main()

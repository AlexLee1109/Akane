"""Launch Akane in popup or server mode with `python -m app`."""

from __future__ import annotations

import sys

from app.core.config import SETTINGS


def main() -> None:
    mode = str(sys.argv[1]).strip().lower() if len(sys.argv) > 1 else SETTINGS.app_mode
    if mode == "server":
        from app.server import serve
        serve(host=SETTINGS.server_host, port=SETTINGS.server_port)
        return
    if mode == "popup":
        from app.ui.popup import launch_popup
        launch_popup()
        return
    if mode == "discord":
        from app.integrations.discord_bot import run_discord_bot
        run_discord_bot()
        return
    raise SystemExit(f"Unknown mode '{mode}'. Use 'popup', 'server', or 'discord'.")


if __name__ == "__main__":
    main()

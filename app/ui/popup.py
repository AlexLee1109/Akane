import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin
from urllib.error import HTTPError, URLError
import urllib.request
import threading
import time

import webview

from app.core.config import POPUP_BACKEND_URL, popup_backend_is_local

try:
    import AppKit
except ImportError:  # pragma: no cover - macOS-only popup behavior
    AppKit = None

AVATAR_RIGHT_MARGIN = -145
AVATAR_BOTTOM_MARGIN = -50
DEFAULT_SESSION_ID = "popup"
# Single companion window — wide enough for the bubble, tall enough for
# the bubble + avatar with a comfortable overlap between them.
COMPANION_WIDTH = 460
COMPANION_HEIGHT = 560
WINDOW_TITLE = "Akane"
_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _log_popup_timing(**values: float | int) -> None:
    if not _TIMING_ENABLED:
        return
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}s")
        else:
            parts.append(f"{key}={value}")
    print(f"[Akane:popup:timing] {' '.join(parts)}", flush=True)


@dataclass(frozen=True)
class Frame:
    x: int
    y: int
    width: int
    height: int


class PopupLayout:
    """Single companion window anchored to the bottom-right corner."""

    def __init__(self, screen):
        f = screen.frame()

        self.screen_x = int(f.origin.x)
        self.screen_y = int(f.origin.y)
        self.screen_width = int(f.size.width)
        self.screen_height = int(f.size.height)

        # Keep the window compact — just the bottom-right corner.
        window_x = self.screen_x + self.screen_width - COMPANION_WIDTH - AVATAR_RIGHT_MARGIN
        window_y = self.screen_y + self.screen_height - COMPANION_HEIGHT - AVATAR_BOTTOM_MARGIN

        self.frames = {
            "companion": Frame(window_x, window_y, COMPANION_WIDTH, COMPANION_HEIGHT),
        }


class WindowApi:
    def __init__(self, app) -> None:
        self.app = app

    def close_window(self) -> None:
        self.app.close_all_windows()

    def minimize_window(self) -> None:
        self.app.minimize_all_windows()

    def send_message_stream(self, message: str) -> None:
        self.app.send_message_stream(message)

    def toggle_composer(self) -> None:
        self.app.toggle_composer()


class PopupApp:
    def __init__(self):
        self.server = None
        self.api = WindowApi(self)
        self.backend_url = POPUP_BACKEND_URL.rstrip("/")
        self.static_index = Path(__file__).parent / "static" / "index.html"
        self.windows: dict[str, object] = {}
        self._shutting_down = False
        self._composer_visible = False
        self._ensure_server()

    def _emit_stream_event(self, payload: dict) -> None:
        window = self.windows.get("companion")
        if window is None:
            return
        try:
            event = json.dumps(payload)
            window.evaluate_js(
                "window.__akaneStreamEvent && "
                f"window.__akaneStreamEvent({event});"
            )
        except Exception:
            return

    def _run_message_stream(self, message: str) -> None:
        started_at = time.perf_counter()
        first_line_at = None
        first_delta_at = None
        line_count = 0
        message = str(message or "").strip()
        if not message:
            self._emit_stream_event({"type": "error", "error": "Message is empty."})
            return
        session_id = DEFAULT_SESSION_ID
        try:
            for line in self._remote_stream_lines(message, session_id):
                line_count += 1
                line_at = time.perf_counter()
                if first_line_at is None:
                    first_line_at = line_at
                event = self._emit_stream_line(line)
                if first_delta_at is None and event and event.get("type") == "delta":
                    first_delta_at = line_at
            done_at = time.perf_counter()
            _log_popup_timing(
                first_line=(first_line_at or done_at) - started_at,
                first_delta=(first_delta_at or done_at) - started_at,
                total=done_at - started_at,
                lines=line_count,
            )
        except Exception as exc:
            done_at = time.perf_counter()
            _log_popup_timing(
                first_line=(first_line_at or done_at) - started_at,
                first_delta=(first_delta_at or done_at) - started_at,
                total=done_at - started_at,
                lines=line_count,
            )
            self._emit_stream_event({"type": "error", "error": str(exc)})

    def _emit_stream_line(self, line: str | bytes) -> dict | None:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line or "").strip()
        if not line:
            return None
        event = json.loads(line)
        self._emit_stream_event(event)
        return event

    def _remote_stream_lines(self, message: str, session_id: str):
        payload = json.dumps(
            {
                "message": message,
                "session_id": session_id,
                "source": "popup",
                "skip_memory": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            urljoin(f"{self.backend_url}/", "api/chat/stream"),
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    yield line
        except HTTPError as exc:
            detail = exc.reason
            try:
                body = exc.read().decode("utf-8", errors="replace")
                payload = json.loads(body)
                detail = payload.get("error") or payload.get("detail") or body
            except Exception:
                pass
            raise RuntimeError(f"Remote backend returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach remote backend at {self.backend_url}: {exc.reason}") from exc

    def send_message_stream(self, message: str) -> None:
        thread = threading.Thread(
            target=self._run_message_stream,
            args=(message,),
            daemon=True,
            name="AkanePopupStream",
        )
        thread.start()

    @staticmethod
    def _window_call(window, method: str, *args) -> None:
        try:
            getattr(window, method)(*args)
        except Exception:
            pass

    def _ensure_server(self):
        if not popup_backend_is_local():
            return
        try:
            urllib.request.urlopen(urljoin(f"{self.backend_url}/", "api/state"), timeout=1)
        except Exception:
            from app.server import serve_in_thread

            self.server, _ = serve_in_thread()

    def _build_start_url(self) -> str:
        params = urlencode({"popup_role": "companion"})
        if popup_backend_is_local():
            return f"{self.backend_url}/?{params}"

        query = urlencode(
            {
                "api_base": self.backend_url,
                "popup_role": "companion",
            },
            quote_via=quote,
            safe=":/",
        )
        return f"file://{self.static_index}?{query}"

    _layout_cache = None

    def _layout(self):
        # Return cached layout if available and AppKit is available
        if self._layout_cache is not None and AppKit is not None:
            return self._layout_cache

        if AppKit is None:
            layout = PopupLayout(AppKit.NSScreen.mainScreen())
            self._layout_cache = layout
            return layout

        screens = AppKit.NSScreen.screens()

        try:
            mouse = AppKit.NSEvent.mouseLocation()
            for s in screens:
                if AppKit.NSPointInRect(mouse, s.frame()):
                    layout = PopupLayout(s)
                    self._layout_cache = layout
                    return layout
        except Exception:
            pass

        layout = PopupLayout(AppKit.NSScreen.mainScreen())
        self._layout_cache = layout
        return layout

    def _create_window(self, *, width: int, height: int):
        window = webview.create_window(
            WINDOW_TITLE,
            self._build_start_url(),
            js_api=self.api,
            width=width,
            height=height,
            min_size=(width, height),
            frameless=True,
            easy_drag=False,
            shadow=False,
            background_color="#000000",
            transparent=True,
            on_top=True,
        )
        window.events.closed += self._on_window_closed
        self.windows["companion"] = window
        return window

    def _position_windows(self) -> None:
        layout = self._layout()
        sx = layout.screen_x
        sy = layout.screen_y

        window = self.windows.get("companion")
        if window is None:
            return

        frame = layout.frames["companion"]
        # Convert global screen coords → local (relative to the screen origin).
        x = frame.x - sx
        y = frame.y - sy
        self._window_call(window, "resize", frame.width, frame.height)
        self._window_call(window, "move", x, y)

    # ── Composer visibility ───────────────────────────────────────────────

    def _set_composer_visible(self, visible: bool) -> None:
        self._composer_visible = bool(visible)
        window = self.windows.get("companion")
        if window is None:
            return
        attr_value = "true" if self._composer_visible else "false"
        try:
            window.evaluate_js(
                f"document.body.setAttribute('data-composer-open', '{attr_value}');"
            )
            hook = "__akaneComposerShown" if self._composer_visible else "__akaneComposerHidden"
            window.evaluate_js(f"window.{hook} && window.{hook}();")
        except Exception:
            pass

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_start(self) -> None:
        self._position_windows()

    def _on_window_closed(self) -> None:
        if self._shutting_down:
            return
        self.close_all_windows()

    # ── Public API ────────────────────────────────────────────────────────

    def minimize_all_windows(self) -> None:
        for window in self.windows.values():
            try:
                window.minimize()
            except Exception:
                pass

    def toggle_composer(self) -> None:
        self._set_composer_visible(not self._composer_visible)

    def close_all_windows(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        for window in list(self.windows.values()):
            try:
                window.destroy()
            except Exception:
                pass
        if self.server:
            self.server.shutdown()
        os._exit(0)

    def run(self):
        layout = self._layout()
        frame = layout.frames["companion"]
        self._create_window(width=frame.width, height=frame.height)
        webview.start(self._on_start)


def launch_popup():
    PopupApp().run()

import json
import math
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin, urlsplit
from urllib.error import HTTPError, URLError
import urllib.request
import threading
import time

import webview

from app.core.config import SETTINGS, popup_backend_is_local

try:
    import AppKit
    from PyObjCTools import AppHelper
except ImportError:  # pragma: no cover - macOS-only popup behavior
    AppKit = None
    AppHelper = None

AVATAR_RIGHT_MARGIN = -145
AVATAR_BOTTOM_MARGIN = -50
DEFAULT_SESSION_ID = "popup"
# Single companion window — wide enough for the bubble, tall enough for
# the bubble + avatar with a comfortable overlap between them.
COMPANION_WIDTH = 460
COMPANION_HEIGHT = 560
WINDOW_TITLE = "Akane"
MOUSE_UPDATE_INTERVAL = 1 / 60
# Let pywebview resolve the JS bridge call before destroying its WKWebView.
CUSTOM_CLOSE_DELAY = 0.05
_TIMING_ENABLED = SETTINGS.timing_enabled
SHUTDOWN_RUNNING = "running"
SHUTDOWN_CLOSING = "closing"
SHUTDOWN_CLOSED = "closed"


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
        if screen is None:
            self.screen_x = 0
            self.screen_y = 0
            self.screen_width = COMPANION_WIDTH
            self.screen_height = COMPANION_HEIGHT
            self.frames = {
                "companion": Frame(0, 0, COMPANION_WIDTH, COMPANION_HEIGHT),
            }
            return
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

    def request_headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {SETTINGS.server_api_token}"}
            if SETTINGS.server_api_token
            else {}
        )

    def toggle_composer(self) -> None:
        self.app.toggle_composer()

    def update_interactive_regions(
        self,
        regions: list[dict],
        passthrough_regions: list[dict],
    ) -> None:
        self.app.update_interactive_regions(regions, passthrough_regions)


class PopupApp:
    def __init__(self):
        self.server = None
        self.api = WindowApi(self)
        self.backend_url = SETTINGS.popup_backend_url.rstrip("/")
        self.static_index = Path(__file__).parent / "static" / "index.html"
        self.windows: dict[str, object] = {}
        self._shutdown_lock = threading.Lock()
        self._shutdown_state = SHUTDOWN_RUNNING
        self._native_close_requested = False
        self._cleanup_started = False
        self._composer_visible = False
        self._stream_lock = threading.Lock()
        self._stream_thread: threading.Thread | None = None
        self._stream_response = None
        self._stream_cancelled = threading.Event()
        self._region_lock = threading.Lock()
        self._interactive_regions: tuple[Frame, ...] = ()
        self._passthrough_regions: tuple[Frame, ...] = ()
        self._local_mouse_monitor = None
        self._global_mouse_monitor = None
        self._mouse_update_pending = False
        self._last_mouse_update_at = 0.0
        self._mouse_capture = False
        self._ensure_server()

    def _shutdown_started(self) -> bool:
        return getattr(self, "_shutdown_state", SHUTDOWN_RUNNING) != SHUTDOWN_RUNNING

    def _emit_stream_event(self, payload: dict) -> None:
        if self._shutdown_started():
            return
        window = self.windows.get("companion")
        if window is None or self._shutdown_started():
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
        if self._shutdown_started():
            return
        if not message:
            self._emit_stream_event({"type": "error", "error": "Message is empty."})
            return
        session_id = DEFAULT_SESSION_ID
        try:
            for line in self._remote_stream_lines(message, session_id):
                if self._shutdown_started():
                    break
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
            if not self._shutdown_started():
                self._emit_stream_event({"type": "error", "error": str(exc)})
        finally:
            with self._stream_lock:
                if self._stream_thread is threading.current_thread():
                    self._stream_thread = None

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
        stream_cancelled = getattr(self, "_stream_cancelled", threading.Event())
        stream_lock = getattr(self, "_stream_lock", threading.Lock())
        payload = json.dumps(
            {
                "message": message,
                "session_id": session_id,
                "source": "popup",
                "skip_memory": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        }
        if SETTINGS.server_api_token:
            headers["Authorization"] = f"Bearer {SETTINGS.server_api_token}"
        request = urllib.request.Request(
            urljoin(f"{self.backend_url}/", "api/chat/stream"),
            data=payload,
            headers=headers,
            method="POST",
        )
        response = None
        try:
            if stream_cancelled.is_set():
                return
            response = urllib.request.urlopen(request, timeout=600)
            with stream_lock:
                if stream_cancelled.is_set():
                    response.close()
                    return
                self._stream_response = response
            with response:
                while True:
                    if stream_cancelled.is_set():
                        break
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
            if exc.code == 401:
                raise RuntimeError(
                    "The Pi rejected the popup credentials (HTTP 401). Set "
                    "AKANE_SERVER_API_TOKEN on this computer to the same private "
                    "token used by the Pi's akane.service."
                ) from exc
            raise RuntimeError(f"Remote backend returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach remote backend at {self.backend_url}: {exc.reason}") from exc
        finally:
            with stream_lock:
                if getattr(self, "_stream_response", None) is response:
                    self._stream_response = None

    def send_message_stream(self, message: str) -> None:
        with self._shutdown_lock:
            if self._shutdown_state != SHUTDOWN_RUNNING:
                return
            with self._stream_lock:
                if self._stream_thread is not None and self._stream_thread.is_alive():
                    raise RuntimeError("A reply is already in progress.")
                thread = threading.Thread(
                    target=self._run_message_stream,
                    args=(message,),
                    daemon=True,
                    name="AkanePopupStream",
                )
                self._stream_thread = thread
                thread.start()

    @staticmethod
    def _parse_regions(regions: list[dict]) -> tuple[Frame, ...]:
        parsed = []
        for region in regions or ():
            try:
                values = tuple(
                    float(region[key]) for key in ("x", "y", "width", "height")
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in values):
                continue
            x, y, width, height = values
            if width > 0 and height > 0:
                parsed.append(Frame(x, y, width, height))
        return tuple(parsed)

    def update_interactive_regions(
        self,
        regions: list[dict],
        passthrough_regions: list[dict],
    ) -> None:
        if self._shutdown_started():
            return
        with self._region_lock:
            self._interactive_regions = self._parse_regions(regions)
            self._passthrough_regions = self._parse_regions(passthrough_regions)
        self._schedule_mouse_passthrough_update()

    def _schedule_mouse_passthrough_update(self, _event=None) -> None:
        if AppHelper is None or self._shutdown_started():
            return
        with self._region_lock:
            if self._shutdown_started() or self._mouse_update_pending:
                return
            self._mouse_update_pending = True
            delay = max(
                0.0,
                MOUSE_UPDATE_INTERVAL
                - (time.monotonic() - self._last_mouse_update_at),
            )
        if delay:
            AppHelper.callLater(delay, self._run_scheduled_mouse_update)
        else:
            AppHelper.callAfter(self._run_scheduled_mouse_update)

    def _run_scheduled_mouse_update(self) -> None:
        with self._region_lock:
            self._mouse_update_pending = False
            if self._shutdown_started():
                return
            self._last_mouse_update_at = time.monotonic()
        self._update_mouse_passthrough()

    def _handle_local_mouse_event(self, event):
        self._schedule_mouse_passthrough_update()
        return event

    def _update_mouse_passthrough(self) -> None:
        if AppKit is None or self._shutdown_started():
            return
        window = self.windows.get("companion")
        native = getattr(window, "native", None)
        if native is None:
            return
        try:
            point = AppKit.NSEvent.mouseLocation()
            frame = native.frame()
            dom_x = float(point.x - frame.origin.x)
            dom_y = float(frame.origin.y + frame.size.height - point.y)
            with self._region_lock:
                passthrough = any(
                    region.x <= dom_x <= region.x + region.width
                    and region.y <= dom_y <= region.y + region.height
                    for region in self._passthrough_regions
                )
                interactive = not passthrough and any(
                    region.x <= dom_x <= region.x + region.width
                    and region.y <= dom_y <= region.y + region.height
                    for region in self._interactive_regions
                )
            pressed = bool(AppKit.NSEvent.pressedMouseButtons())
            if pressed and self._mouse_capture:
                interactive = True
            elif pressed and interactive and not native.ignoresMouseEvents():
                self._mouse_capture = True
            elif not pressed:
                self._mouse_capture = False
            ignores_mouse = not interactive
            if bool(native.ignoresMouseEvents()) != ignores_mouse:
                native.setIgnoresMouseEvents_(ignores_mouse)
        except Exception:
            return

    def _configure_mouse_passthrough(self) -> None:
        if AppHelper is None or self._shutdown_started():
            return
        AppHelper.callAfter(self._configure_mouse_passthrough_on_main)

    def _configure_mouse_passthrough_on_main(self) -> None:
        if (
            AppKit is None
            or self._shutdown_started()
            or self._local_mouse_monitor is not None
        ):
            return
        window = self.windows.get("companion")
        native = getattr(window, "native", None)
        if native is None:
            return
        native.setIgnoresMouseEvents_(True)
        event_mask = (
            AppKit.NSEventMaskMouseMoved
            | AppKit.NSEventMaskLeftMouseDragged
            | AppKit.NSEventMaskRightMouseDragged
            | AppKit.NSEventMaskOtherMouseDragged
            | AppKit.NSEventMaskLeftMouseDown
            | AppKit.NSEventMaskRightMouseDown
            | AppKit.NSEventMaskOtherMouseDown
            | AppKit.NSEventMaskLeftMouseUp
            | AppKit.NSEventMaskRightMouseUp
            | AppKit.NSEventMaskOtherMouseUp
        )
        self._local_mouse_monitor = (
            AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                event_mask,
                self._handle_local_mouse_event,
            )
        )
        self._global_mouse_monitor = (
            AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                event_mask,
                self._schedule_mouse_passthrough_update,
            )
        )
        self._update_mouse_passthrough()

    def _remove_mouse_monitors_on_main(self) -> None:
        if AppKit is None:
            return
        for attribute in ("_local_mouse_monitor", "_global_mouse_monitor"):
            monitor = getattr(self, attribute, None)
            if monitor is not None:
                setattr(self, attribute, None)
                AppKit.NSEvent.removeMonitor_(monitor)

    @staticmethod
    def _window_call(window, method: str, *args) -> None:
        try:
            getattr(window, method)(*args)
        except Exception:
            pass

    def _ensure_server(self):
        if not popup_backend_is_local():
            return
        headers = {}
        if SETTINGS.server_api_token:
            headers["Authorization"] = f"Bearer {SETTINGS.server_api_token}"
        for delay in (0.0, 0.2, 0.4):
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(
                urljoin(f"{self.backend_url}/", "api/state"),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=1):
                    return
            except HTTPError:
                # Any HTTP response proves that a server owns this address.
                return
            except (OSError, URLError):
                continue
        from app.server import serve_in_thread

        backend = urlsplit(self.backend_url)
        host = backend.hostname or "127.0.0.1"
        port = backend.port or (443 if backend.scheme == "https" else 80)
        self.server, _ = serve_in_thread(host=host, port=port)

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
            layout = PopupLayout(None)
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
        window.events.closing += self._on_window_closing
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
        if self._shutdown_started():
            return
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
        if self._shutdown_started():
            return
        self._position_windows()
        self._configure_mouse_passthrough()

    def _on_window_closing(self) -> None:
        self._begin_shutdown()
        self._remove_mouse_monitors_on_main()

    def _finalize_window_closed(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_state == SHUTDOWN_CLOSED:
                return
            self._shutdown_state = SHUTDOWN_CLOSED
            self._stream_cancelled.set()
            self.windows.clear()
        self._start_cleanup_worker()

    # ── Public API ────────────────────────────────────────────────────────

    def minimize_all_windows(self) -> None:
        if self._shutdown_started():
            return
        for window in self.windows.values():
            try:
                window.minimize()
            except Exception:
                pass

    def toggle_composer(self) -> None:
        if self._shutdown_started():
            return
        self._set_composer_visible(not self._composer_visible)

    def close_all_windows(self) -> None:
        if not self._begin_shutdown():
            return
        with self._shutdown_lock:
            self._native_close_requested = True
        if AppHelper is not None:
            AppHelper.callLater(
                CUSTOM_CLOSE_DELAY,
                self._request_native_window_close,
            )
        else:
            self._request_native_window_close()

    def _begin_shutdown(self) -> bool:
        with self._shutdown_lock:
            if self._shutdown_state != SHUTDOWN_RUNNING:
                return False
            self._shutdown_state = SHUTDOWN_CLOSING
            self._stream_cancelled.set()
        self._start_cleanup_worker()
        return True

    def _start_cleanup_worker(self) -> None:
        with self._shutdown_lock:
            if self._cleanup_started:
                return
            self._cleanup_started = True
        threading.Thread(
            target=self._cleanup_background_resources,
            daemon=True,
            name="AkanePopupCleanup",
        ).start()

    def _cleanup_background_resources(self) -> None:
        with self._stream_lock:
            response = self._stream_response

        if response is not None:
            raw_socket = getattr(
                getattr(getattr(response, "fp", None), "raw", None),
                "_sock",
                None,
            )
            if isinstance(raw_socket, socket.socket):
                try:
                    raw_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            try:
                response.close()
            except (OSError, ValueError):
                pass

        server = self.server
        self.server = None
        if server is not None:
            server.shutdown()

    def _request_native_window_close(self) -> None:
        with self._shutdown_lock:
            if (
                self._shutdown_state != SHUTDOWN_CLOSING
                or not self._native_close_requested
            ):
                return
            self._native_close_requested = False

        windows = list(self.windows.values())
        for window in windows:
            try:
                window.destroy()
            except Exception:
                pass

    def run(self):
        layout = self._layout()
        frame = layout.frames["companion"]
        self._create_window(width=frame.width, height=frame.height)
        webview.start(self._on_start)
        self._finalize_window_closed()


def launch_popup():
    PopupApp().run()

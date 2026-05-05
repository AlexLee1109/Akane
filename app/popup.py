import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin

import webview

from app.config import POPUP_BACKEND_URL, popup_backend_is_local
from app.server import serve_in_thread
from app.ui_assets import IMAGES_DIR

try:
    import AppKit
except ImportError:  # pragma: no cover - macOS-only popup behavior
    AppKit = None

DEFAULT_SCREEN_WIDTH = 1440
DEFAULT_SCREEN_HEIGHT = 900
HIDDEN_WINDOW_SIZE = 8
AVATAR_HEIGHT = 350
AVATAR_RIGHT_MARGIN = 0
AVATAR_BOTTOM_MARGIN = 0
BUBBLE_WIDTH = 430
BUBBLE_HEIGHT = 520
BUBBLE_MIN_WIDTH = 220
BUBBLE_MAX_WIDTH = 540
BUBBLE_MIN_HEIGHT = 420
BUBBLE_MAX_HEIGHT = 2200
WINDOW_TITLES = {
    "bubble": "Akane Bubble",
    "avatar": "Akane Avatar",
}
AVATAR_IMAGE_PATH = IMAGES_DIR / "Vtuber_model.png"
AVATAR_BOUNDS_WH = (530, 1334)


@dataclass(frozen=True)
class Frame:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

def _primary_screen_frame():
    if AppKit is None:
        return None
    screens = AppKit.NSScreen.screens()
    if not screens:
        return None
    try:
        point = AppKit.NSEvent.mouseLocation()
        for screen in screens:
            frame = screen.frame()
            if AppKit.NSPointInRect(point, frame):
                return frame
    except Exception:
        pass
    main_screen = AppKit.NSScreen.mainScreen()
    if main_screen is not None:
        return main_screen.frame()
    return max(
        (screen.frame() for screen in screens),
        key=lambda rect: rect.size.width * rect.size.height,
    )

def _avatar_window_width(height: int) -> int:
    bounds_width, bounds_height = AVATAR_BOUNDS_WH
    if AVATAR_IMAGE_PATH.exists():
        try:
            from PIL import Image

            with Image.open(AVATAR_IMAGE_PATH).convert("RGBA") as image:
                bbox = image.getchannel("A").getbbox()
                if bbox:
                    bounds_width = max(bbox[2] - bbox[0], 1)
                    bounds_height = max(bbox[3] - bbox[1], 1)
        except Exception:
            pass
    return max(120, int(height * (bounds_width / bounds_height)))


class PopupLayout:
    def __init__(self, screen):
        f = screen.frame()

        self.screen_x = int(f.origin.x)
        self.screen_y = int(f.origin.y)
        self.screen_width = int(f.size.width)
        self.screen_height = int(f.size.height)

        avatar_width = _avatar_window_width(AVATAR_HEIGHT)

        avatar_x = self.screen_x + self.screen_width - avatar_width
        avatar_y = self.screen_y + self.screen_height - AVATAR_HEIGHT

        bubble_x = avatar_x + 20
        bubble_y = avatar_y - 200

        self.frames = {
            "avatar": Frame(avatar_x, avatar_y, avatar_width, AVATAR_HEIGHT),
            "bubble": Frame(bubble_x, bubble_y, BUBBLE_WIDTH, BUBBLE_HEIGHT),
        }

        self.hidden_frames = {
            "bubble": Frame(
                self.screen_x + self.screen_width - HIDDEN_WINDOW_SIZE,
                self.screen_y + HIDDEN_WINDOW_SIZE,
                HIDDEN_WINDOW_SIZE,
                HIDDEN_WINDOW_SIZE,
            )
        }

class WindowApi:
    def __init__(self, app) -> None:
        self.app = app

    def close_window(self) -> None:
        self.app.close_all_windows()

    def minimize_window(self) -> None:
        self.app.minimize_all_windows()

    def toggle_on_top(self, state: bool) -> None:
        self.app.set_on_top_all(bool(state))

    def focus_composer(self) -> None:
        return None

    def sync_bubble_height(self, height: int) -> None:
        self.app.sync_bubble_height(height)

    def sync_bubble_size(self, payload) -> None:
        try:
            width, height = payload
        except Exception:
            return
        self.app.sync_bubble_size(int(width), int(height))

    def push_bubble_text(self, text: str) -> None:
        self.app.push_bubble_text(text)

    def open_composer(self) -> None:
        self.app.open_composer()

    def close_composer(self) -> None:
        self.app.close_composer()

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
        self._bubble_base_x = 0
        self._bubble_base_y = 0
        self._bubble_width = 0
        self._bubble_height = 0
        self._bubble_visible = False
        self._bubble_text = ""
        self._composer_visible = False
        self._ensure_server()

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
            import urllib.request

            urllib.request.urlopen(urljoin(f"{self.backend_url}/", "api/state"), timeout=1)
        except Exception:
            self.server, _ = serve_in_thread()

    def _build_start_url(self, role: str) -> str:
        params = urlencode({"popup_role": role})
        if popup_backend_is_local():
            return f"{self.backend_url}/?{params}"

        query = urlencode(
            {
                "api_base": self.backend_url,
                "popup_role": role,
            },
            quote_via=quote,
            safe=":/",
        )
        return f"file://{self.static_index}?{query}"

    def _layout(self):
        if AppKit is None:
            return PopupLayout(AppKit.NSScreen.mainScreen())

        screens = AppKit.NSScreen.screens()

        try:
            mouse = AppKit.NSEvent.mouseLocation()
            for s in screens:
                if AppKit.NSPointInRect(mouse, s.frame()):
                    return PopupLayout(s)
        except Exception:
            pass

        return PopupLayout(AppKit.NSScreen.mainScreen())

    def _create_window(self, role: str, title: str, *, width: int, height: int):
        min_size = (width, height)
        if role in {"bubble", "composer"}:
            min_size = (HIDDEN_WINDOW_SIZE, HIDDEN_WINDOW_SIZE)
        window = webview.create_window(
            title,
            self._build_start_url(role),
            js_api=self.api,
            width=width,
            height=height,
            min_size=min_size,
            frameless=True,
            easy_drag=False,
            shadow=False,
            background_color="#000000",
            transparent=True,
            on_top=True,
        )
        window.events.closed += lambda: self._on_window_closed(role)
        try:
            window.events.loaded += lambda: self._on_window_loaded(role)
        except Exception:
            pass
        self.windows[role] = window
        return window

    def _position_windows(self) -> None:
        layout = self._layout()

        sx = layout.screen_x
        sy = layout.screen_y

        for role, window in self.windows.items():
            frame = layout.frames[role]

            if role == "bubble" and not self._bubble_visible:
                frame = layout.hidden_frames["bubble"]

            # ✅ convert GLOBAL → LOCAL for BOTH avatar and bubble
            x = frame.x - sx
            y = frame.y - sy

            self._window_call(window, "resize", frame.width, frame.height)
            self._window_call(window, "move", x, y)

    def _set_composer_visible(self, visible: bool) -> None:
        self._composer_visible = bool(visible)
        avatar = self.windows.get("avatar")
        if avatar is None:
            return
        try:
            hook = "__akaneComposerShown" if self._composer_visible else "__akaneComposerHidden"
            avatar.evaluate_js(f"window.{hook} && window.{hook}();")
        except Exception:
            pass

    def _set_bubble_visible(self, visible: bool) -> None:
        self._bubble_visible = bool(visible)
        bubble = self.windows.get("bubble")
        if bubble is None:
            return

        layout = self._layout()

        sx = layout.screen_x
        sy = layout.screen_y

        frame = (
            layout.frames["bubble"]
            if self._bubble_visible
            else layout.hidden_frames["bubble"]
        )

        # ✅ FIX: convert GLOBAL → LOCAL
        x = frame.x - sx
        y = frame.y - sy
        width = frame.width
        height = frame.height

        if self._bubble_visible:
            self._bubble_base_x = frame.x
            self._bubble_base_y = frame.y

        self._window_call(bubble, "resize", width, height)
        self._window_call(bubble, "move", x, y)

    def _on_start(self) -> None:
        self._position_windows()

    def _on_window_closed(self, role: str) -> None:
        if self._shutting_down:
            return
        self.close_all_windows()

    def _on_window_loaded(self, role: str) -> None:
        if role == "bubble":
            self._apply_bubble_text()

    def minimize_all_windows(self) -> None:
        for window in self.windows.values():
            try:
                window.minimize()
            except Exception:
                pass

    def set_on_top_all(self, state: bool) -> None:
        for window in self.windows.values():
            try:
                window.on_top = state
            except Exception:
                pass

    def open_composer(self) -> None:
        self._set_composer_visible(True)

    def close_composer(self) -> None:
        self._set_composer_visible(False)

    def toggle_composer(self) -> None:
        self._set_composer_visible(not self._composer_visible)

    def sync_bubble_height(self, height: int) -> None:
        self.sync_bubble_size(self._bubble_width or 430, height)

    def sync_bubble_size(self, width: int, height: int) -> None:
        window = self.windows.get("bubble")
        if window is None:
            return

        if int(width) <= 8 or int(height) <= 8:
            self._set_bubble_visible(False)
            return
        else:
            self._set_bubble_visible(True)
            target_width = max(BUBBLE_MIN_WIDTH, min(int(width) + 2, BUBBLE_MAX_WIDTH))
            target_height = max(BUBBLE_MIN_HEIGHT, min(int(height) + 240, BUBBLE_MAX_HEIGHT))
        if not self._bubble_width:
            layout = self._layout()
            x, y, width, default_height = layout.frames["bubble"].as_tuple()
            self._bubble_base_x = x
            self._bubble_base_y = y
            self._bubble_width = width
            self._bubble_height = default_height

        if abs(target_width - self._bubble_width) < 3 and abs(target_height - self._bubble_height) < 3:
            return

        self._bubble_width = target_width
        self._bubble_height = target_height

        try:
            window.resize(target_width, target_height)
        except Exception:
            return

        self._window_call(window, "move", self._bubble_base_x, self._bubble_base_y)

    def push_bubble_text(self, text: str) -> None:
        self._bubble_text = str(text or "")
        self._apply_bubble_text()

    def _apply_bubble_text(self) -> None:
        window = self.windows.get("bubble")
        if window is None:
            return

        next_text = self._bubble_text
        if next_text.strip():
            self._set_bubble_visible(True)

        safe_text = json.dumps(next_text)
        script = (
            "window.__akaneSetBubbleText && "
            f"window.__akaneSetBubbleText({safe_text});"
        )
        try:
            window.evaluate_js(script)
        except Exception:
            return

        if not next_text.strip():
            self._set_bubble_visible(False)

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
        import os

        os._exit(0)

    def run(self):
        layout = self._layout()
        for role in ("avatar", "bubble"):
            frame = layout.frames[role]
            self._create_window(role, WINDOW_TITLES[role], width=frame.width, height=frame.height)
        webview.start(self._on_start)


def launch_popup():
    PopupApp().run()
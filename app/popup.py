import json
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin

import webview

from app.config import POPUP_BACKEND_URL, popup_backend_is_local
from app.server import serve_in_thread

try:
    import AppKit
except ImportError:  # pragma: no cover - macOS-only popup behavior
    AppKit = None

DEFAULT_SCREEN_WIDTH = 1440
DEFAULT_SCREEN_HEIGHT = 900
AVATAR_HEIGHT = 800
AVATAR_RIGHT_MARGIN = -20
AVATAR_BOTTOM_MARGIN = -500
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVATAR_IMAGE_PATH = PROJECT_ROOT / "vtuber_model.png"
AVATAR_BOUNDS_WH = (530, 1334)


def _primary_screen_frame():
    if AppKit is None:
        return None
    screens = AppKit.NSScreen.screens()
    if not screens:
        return None
    return screens[0].visibleFrame()


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
    def __init__(self, screen_width: int, screen_height: int, screen_x: int = 0, screen_y: int = 0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_x = screen_x
        self.screen_y = screen_y

        edge_margin = 18
        avatar_width = _avatar_window_width(AVATAR_HEIGHT)
        avatar_height = AVATAR_HEIGHT
        avatar_x = screen_x + max(screen_width - avatar_width - AVATAR_RIGHT_MARGIN, 0)
        avatar_y = screen_y + max(screen_height - avatar_height - AVATAR_BOTTOM_MARGIN, 0)
        bubble_width = 350
        bubble_height = 200
        bubble_x = max(screen_x + 40, avatar_x - bubble_width + 150)
        bubble_y = max(screen_y + 36, avatar_y - 50)
        composer_height = 78
        composer_y = screen_y + max(screen_height - composer_height - edge_margin, 0)
        composer_width = min(max(int(screen_width * 0.20), 400), 760)
        composer_max_width = max(420, avatar_x - screen_x - (edge_margin * 2))
        composer_width = min(composer_width, composer_max_width)
        composer_hidden_frame = (
            screen_x + screen_width - 8,
            screen_y + screen_height - 8,
            8,
            8,
        )
        bubble_hidden_frame = (
            screen_x + screen_width - 8,
            screen_y + 8,
            8,
            8,
        )

        self.frames = {
            "bubble": (
                bubble_x,
                bubble_y,
                bubble_width,
                bubble_height,
            ),
            "avatar": (
                avatar_x,
                avatar_y,
                avatar_width,
                avatar_height,
            ),
            "composer": (
                screen_x + edge_margin + 1100,
                composer_y + 50,
                composer_width,
                composer_height,
            ),
        }
        self.hidden_frames = {
            "bubble": bubble_hidden_frame,
            "composer": composer_hidden_frame,
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
        self._composer_visible = False
        self._ensure_server()

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

    def _layout(self) -> PopupLayout:
        frame = _primary_screen_frame()
        if frame is None:
            return PopupLayout(DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT)
        return PopupLayout(
            int(frame.size.width),
            int(frame.size.height),
            int(frame.origin.x),
            int(frame.origin.y),
        )

    def _create_window(self, role: str, title: str, *, width: int, height: int):
        min_size = (width, height)
        if role in {"bubble", "composer"}:
            min_size = (8, 8)
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
        self.windows[role] = window
        return window

    def _position_windows(self) -> None:
        layout = self._layout()
        for role, window in self.windows.items():
            frame = layout.frames[role]
            if role == "composer" and not self._composer_visible:
                frame = layout.hidden_frames["composer"]
            x, y, width, height = frame
            if role == "bubble":
                self._bubble_base_x = x
                self._bubble_base_y = y
                self._bubble_width = width
                self._bubble_height = height
            try:
                window.resize(width, height)
            except Exception:
                pass
            window.move(x, y)
            if role == "bubble" and not self._bubble_visible:
                try:
                    window.hide()
                except Exception:
                    pass

    def _set_composer_visible(self, visible: bool) -> None:
        self._composer_visible = bool(visible)
        composer = self.windows.get("composer")
        if composer is None:
            return

        layout = self._layout()
        frame = layout.frames["composer"] if self._composer_visible else layout.hidden_frames["composer"]
        x, y, width, height = frame
        try:
            composer.resize(width, height)
        except Exception:
            pass
        try:
            composer.move(x, y)
        except Exception:
            pass
        try:
            hook = "__akaneComposerShown" if self._composer_visible else "__akaneComposerHidden"
            composer.evaluate_js(f"window.{hook} && window.{hook}();")
        except Exception:
            pass

    def _set_bubble_visible(self, visible: bool) -> None:
        self._bubble_visible = bool(visible)
        bubble = self.windows.get("bubble")
        if bubble is None:
            return
        try:
            if self._bubble_visible:
                bubble.show()
            else:
                bubble.hide()
        except Exception:
            pass

    def _on_start(self) -> None:
        self._position_windows()

    def _on_window_closed(self, role: str) -> None:
        if self._shutting_down:
            return
        self.close_all_windows()

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
            target_width = max(220, min(int(width) + 2, 430))
            target_height = max(92, min(int(height) + 12, 520))
        if not self._bubble_width:
            layout = self._layout()
            x, y, width, default_height = layout.frames["bubble"]
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

        try:
            window.move(self._bubble_base_x, self._bubble_base_y)
        except Exception:
            pass

    def push_bubble_text(self, text: str) -> None:
        window = self.windows.get("bubble")
        if window is None:
            return

        safe_text = json.dumps(text)
        script = (
            "window.__akaneSetBubbleText && "
            f"window.__akaneSetBubbleText({safe_text});"
        )
        try:
            window.evaluate_js(script)
        except Exception:
            pass

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
        self._create_window("bubble", "Akane Bubble", width=layout.frames["bubble"][2], height=layout.frames["bubble"][3])
        self._create_window("avatar", "Akane Avatar", width=layout.frames["avatar"][2], height=layout.frames["avatar"][3])
        self._create_window("composer", "Akane Composer", width=layout.frames["composer"][2], height=layout.frames["composer"][3])
        webview.start(self._on_start)


def launch_popup():
    PopupApp().run()

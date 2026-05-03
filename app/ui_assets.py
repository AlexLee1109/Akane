"""Shared UI asset paths and route helpers for the desktop popup."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"

UI_ASSET_PATHS = {
    "/images/Vtuber_model.png": IMAGES_DIR / "Vtuber_model.png",
    "/images/Text Bubble.png": IMAGES_DIR / "Text Bubble.png",
    "/images/Text Bubble decoration.png": IMAGES_DIR / "Text Bubble decoration.png",
    "/images/input_bar.png": IMAGES_DIR / "input_bar.png",
    "/images/Send_button.png": IMAGES_DIR / "Send_button.png",
    "/images/Message icon.png": IMAGES_DIR / "Message icon.png",
    "/images/Minimize icon.png": IMAGES_DIR / "Minimize icon.png",
    "/images/Close Icon.png": IMAGES_DIR / "Close Icon.png",
}


def resolve_ui_asset(route: str) -> Path | None:
    """Return the filesystem path for a served popup asset route."""
    return UI_ASSET_PATHS.get(str(route or ""))

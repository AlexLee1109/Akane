"""Historical ASGI import path for Akane Next's application server."""

from app.companion.server import create_app, serve, serve_in_thread

APP = create_app()

if __name__ == "__main__":
    serve()

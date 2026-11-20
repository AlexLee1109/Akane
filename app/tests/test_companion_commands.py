import asyncio
import json

from starlette.requests import Request

from app.companion import next_adapter, server


def test_forget_command_and_state_refresh_do_not_load_native_model(tmp_path, monkeypatch):
    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("Forget commands must not initialize the native model.")

    monkeypatch.setattr(next_adapter, "NativeModel", fail_if_loaded)
    companion = next_adapter.NextCompanion()
    companion.root = tmp_path
    app = server.create_app(companion)
    try:
        async def forget_and_reload_state():
            chat = next(route.endpoint for route in app.routes
                        if getattr(route, "path", None) == "/api/chat")
            state = next(route.endpoint for route in app.routes
                         if getattr(route, "path", None) == "/api/state")
            body = json.dumps({
                "message": "/forget_me",
                "session_id": "popup",
                "source": "popup",
            }).encode()
            sent = False

            async def receive():
                nonlocal sent
                if sent:
                    return {"type": "http.request", "body": b"", "more_body": False}
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request({
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/chat",
                "raw_path": b"/api/chat",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 80),
            }, receive)
            forgotten = await chat(request)
            assert forgotten["notice"] == "Chat reset."

            refreshed = await state(session_id="popup")
            assert refreshed["messages"] == []

        asyncio.run(forget_and_reload_state())
        assert companion._native_model is None
    finally:
        companion.close()

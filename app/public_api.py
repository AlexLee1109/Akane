"""Narrow guest-only HTTP adapter for Akane's existing conversation runtime."""

from __future__ import annotations

import hashlib
import queue
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import (
    LLAMA_CONTEXT_WINDOW,
    MAX_INPUT_CHARS,
    PUBLIC_ALLOWED_ORIGINS,
    PUBLIC_API_ENABLED,
    PUBLIC_API_HOST,
    PUBLIC_API_PORT,
    PUBLIC_GENERATION_TIMEOUT_SECONDS,
    PUBLIC_GUEST_IDLE_SECONDS,
    PUBLIC_GUEST_MAX_LIFETIME_SECONDS,
    PUBLIC_MAX_ACTIVE,
    PUBLIC_MAX_GUEST_SESSIONS,
    PUBLIC_MAX_QUEUE,
    PUBLIC_MESSAGE_LIMIT,
    PUBLIC_REQUEST_COOLDOWN_SECONDS,
    PUBLIC_RESPONSE_TOKEN_LIMIT,
)
from app.core.memory import StateStore
from app.core.model_loader import ModelManager
from app.core.session import (
    GenerationBusyError,
    GenerationCancelled,
    GenerationQueueFullError,
    clear_conversation_caches,
    clear_profile_caches,
    normalize_chat_input,
    run_companion_turn,
)

_PUBLIC_PREFIX = "public:guest:"


class PublicApiError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _error(code: str) -> PublicApiError:
    values = {
        "invalid_request": ("That request is not valid.", 400),
        "unauthorized": ("A valid guest session is required.", 401),
        "session_expired": ("This guest session has expired.", 401),
        "message_too_long": ("That message is too long.", 413),
        "rate_limited": ("Please wait a moment before sending another message.", 429),
        "queue_full": ("Akane is receiving too many requests right now.", 429),
        "busy": ("Akane is busy right now.", 429),
        "model_unavailable": ("Live Akane is temporarily unavailable.", 503),
        "generation_timeout": ("Akane took too long to respond.", 504),
        "internal_error": ("Akane could not finish that response.", 500),
    }
    message, status = values.get(code, values["internal_error"])
    return PublicApiError(code, message, status)


def _error_response(error: PublicApiError) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": error.code, "message": error.message}},
        status_code=error.status_code,
    )


def _json_line(payload: dict[str, object]) -> bytes:
    return orjson.dumps(payload) + b"\n"


def _profile_ref(profile_id: str) -> str:
    return profile_id.rsplit(":", 1)[-1][:8]


def _log(event: str, profile_id: str = "") -> None:
    suffix = f" guest={_profile_ref(profile_id)}" if profile_id else ""
    print(f"[Akane:public] {event}{suffix}", flush=True)


def _normalized_origin(value: str) -> str:
    candidate = str(value or "").strip().rstrip("/")
    parsed = urlsplit(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Public CORS origins must be exact HTTP(S) origins.")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Public non-loopback CORS origins must use HTTPS.")
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class PublicApiSettings:
    enabled: bool = PUBLIC_API_ENABLED
    host: str = PUBLIC_API_HOST
    port: int = PUBLIC_API_PORT
    allowed_origins: tuple[str, ...] = PUBLIC_ALLOWED_ORIGINS
    guest_idle_seconds: int = PUBLIC_GUEST_IDLE_SECONDS
    guest_max_lifetime_seconds: int = PUBLIC_GUEST_MAX_LIFETIME_SECONDS
    max_guest_sessions: int = PUBLIC_MAX_GUEST_SESSIONS
    max_active: int = PUBLIC_MAX_ACTIVE
    max_queue: int = PUBLIC_MAX_QUEUE
    message_limit: int = PUBLIC_MESSAGE_LIMIT
    response_token_limit: int = PUBLIC_RESPONSE_TOKEN_LIMIT
    request_cooldown_seconds: float = PUBLIC_REQUEST_COOLDOWN_SECONDS
    generation_timeout_seconds: float = PUBLIC_GENERATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        origins = tuple(dict.fromkeys(_normalized_origin(item) for item in self.allowed_origins))
        object.__setattr__(self, "allowed_origins", origins)
        if self.host != "127.0.0.1":
            raise ValueError("AKANE_PUBLIC_API_HOST must be 127.0.0.1.")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("AKANE_PUBLIC_API_PORT must be between 1 and 65535.")
        if self.enabled and not origins:
            raise ValueError("AKANE_PUBLIC_ALLOWED_ORIGINS is required in public mode.")
        if int(self.guest_idle_seconds) <= 0 or int(self.guest_max_lifetime_seconds) <= 0:
            raise ValueError("Public guest expiration limits must be positive.")
        if int(self.max_guest_sessions) <= 0:
            raise ValueError("AKANE_PUBLIC_MAX_GUEST_SESSIONS must be positive.")
        if int(self.max_active) != 1 or int(self.max_queue) < 0:
            raise ValueError("Public capacity requires one active slot and a non-negative queue.")
        if not 1 <= int(self.message_limit) <= MAX_INPUT_CHARS:
            raise ValueError("AKANE_PUBLIC_MESSAGE_LIMIT is outside the backend input limit.")
        if not 1 <= int(self.response_token_limit) < LLAMA_CONTEXT_WINDOW:
            raise ValueError("AKANE_PUBLIC_RESPONSE_TOKEN_LIMIT is outside the model context.")
        if float(self.request_cooldown_seconds) < 0.0:
            raise ValueError("AKANE_PUBLIC_REQUEST_COOLDOWN_SECONDS cannot be negative.")
        if float(self.generation_timeout_seconds) <= 0.0:
            raise ValueError("AKANE_PUBLIC_GENERATION_TIMEOUT_SECONDS must be positive.")


PUBLIC_API_SETTINGS = PublicApiSettings()


@dataclass(slots=True)
class PublicSession:
    profile_id: str
    conversation_id: str
    token_hash: str
    created_at: float
    last_seen_at: float
    last_request_at: float = 0.0
    retired: bool = False

    def expired(self, settings: PublicApiSettings, now: float) -> bool:
        return (
            now - self.last_seen_at >= settings.guest_idle_seconds
            or now - self.created_at >= settings.guest_max_lifetime_seconds
        )


def _token_hash(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


class GenerationLease:
    def __init__(self, owner: "PublicSessionManager", session: PublicSession):
        self.owner = owner
        self.session = session
        self.cancellation = threading.Event()
        self.deadline = time.monotonic() + owner.settings.generation_timeout_seconds
        self.active = False
        self.released = False
        self.timed_out = False

    def activate(self) -> None:
        with self.owner._condition:
            try:
                while self.owner._active >= self.owner.settings.max_active:
                    if self.owner._shutting_down or self.cancellation.is_set():
                        raise _error("generation_timeout" if self.timed_out else "busy")
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0.0:
                        self.timed_out = True
                        self.cancellation.set()
                        raise _error("generation_timeout")
                    self.owner._condition.wait(timeout=min(0.1, remaining))
                if self.owner._shutting_down or self.cancellation.is_set():
                    raise _error("generation_timeout" if self.timed_out else "busy")
                self.owner._active += 1
                self.active = True
            except Exception:
                self.release()
                raise

    def mark_timed_out(self) -> None:
        self.timed_out = True
        self.cancellation.set()

    def release(self) -> None:
        clear_profile = False
        with self.owner._condition:
            if self.released:
                return
            self.released = True
            if self.active:
                self.owner._active -= 1
            self.owner._admitted -= 1
            self.owner._inflight.discard(self.session.token_hash)
            clear_profile = self.session.retired
            self.owner._condition.notify_all()
        if clear_profile:
            self.owner._clear_profile(self.session.profile_id)


class PublicSessionManager:
    def __init__(self, store: StateStore, settings: PublicApiSettings = PUBLIC_API_SETTINGS):
        self.store = store
        self.settings = settings
        self._condition = threading.Condition(threading.RLock())
        self._sessions: dict[str, PublicSession] = {}
        self._inflight: set[str] = set()
        self._leases: set[GenerationLease] = set()
        self._admitted = 0
        self._active = 0
        self._shutting_down = True

    def start(self) -> None:
        with self._condition:
            self._shutting_down = False
        for profile_id in self.store.profile_ids(prefix=_PUBLIC_PREFIX):
            self._clear_profile(profile_id)
            _log("guest session expired", profile_id)

    def _clear_profile(self, profile_id: str) -> None:
        clear_profile_caches(profile_id)
        self.store.clear_profile(profile_id)

    def shutdown(self) -> None:
        with self._condition:
            self._shutting_down = True
            sessions = tuple(self._sessions.values())
            for session in sessions:
                self._remove_locked(session, "guest session deleted")
            self._condition.notify_all()
            deadline = time.monotonic() + 2.0
            while self._inflight and time.monotonic() < deadline:
                self._condition.wait(timeout=0.05)
        for session in sessions:
            self._clear_profile(session.profile_id)
        with self._condition:
            self._sessions.clear()

    def _remove_locked(self, session: PublicSession, event: str) -> None:
        self._sessions.pop(session.token_hash, None)
        session.retired = True
        for lease in tuple(self._leases):
            if lease.session is session:
                lease.cancellation.set()
        if session.token_hash not in self._inflight:
            self._clear_profile(session.profile_id)
        _log(event, session.profile_id)

    def _cleanup_locked(self, now: float) -> None:
        for session in tuple(self._sessions.values()):
            if session.token_hash not in self._inflight and session.expired(self.settings, now):
                self._remove_locked(session, "guest session expired")

    def create(self) -> tuple[PublicSession, str]:
        now = time.time()
        with self._condition:
            if self._shutting_down:
                raise _error("model_unavailable")
            self._cleanup_locked(now)
            if len(self._sessions) >= self.settings.max_guest_sessions:
                raise _error("busy")
            while True:
                token = secrets.token_urlsafe(32)
                token_hash = _token_hash(token)
                if token_hash not in self._sessions:
                    break
            profile_uuid = str(uuid.uuid4())
            session = PublicSession(
                profile_id=f"{_PUBLIC_PREFIX}{profile_uuid}",
                conversation_id=f"public:conversation:{profile_uuid}",
                token_hash=token_hash,
                created_at=now,
                last_seen_at=now,
            )
            self.store.ensure_profile(session.profile_id)
            self._sessions[token_hash] = session
        _log("guest session created", session.profile_id)
        return session, token

    def resolve(self, token: str, *, touch: bool = True) -> PublicSession:
        supplied = str(token or "").strip()
        if not supplied:
            raise _error("unauthorized")
        token_hash = _token_hash(supplied)
        now = time.time()
        with self._condition:
            self._cleanup_locked(now)
            session = self._sessions.get(token_hash)
            if session is None:
                raise _error("session_expired")
            if not self.store.profile_exists(session.profile_id):
                self._sessions.pop(token_hash, None)
                raise _error("session_expired")
            if touch:
                session.last_seen_at = now
            return session

    def reset(self, session: PublicSession) -> None:
        clear_conversation_caches(session.conversation_id, session.profile_id)
        self.store.clear_conversation(session.conversation_id, session.profile_id)
        _log("guest conversation reset", session.profile_id)

    def delete(self, token: str) -> None:
        session = self.resolve(token, touch=False)
        with self._condition:
            self._remove_locked(session, "guest session deleted")

    def acquire_generation(self, session: PublicSession) -> GenerationLease:
        now = time.monotonic()
        with self._condition:
            if self._shutting_down:
                raise _error("model_unavailable")
            if self._sessions.get(session.token_hash) is not session or session.retired:
                raise _error("session_expired")
            if session.token_hash in self._inflight:
                raise _error("busy")
            if (
                session.last_request_at
                and now - session.last_request_at < self.settings.request_cooldown_seconds
            ):
                _log("public request rate limited", session.profile_id)
                raise _error("rate_limited")
            if self._admitted >= self.settings.max_active + self.settings.max_queue:
                _log("public queue full", session.profile_id)
                raise _error("queue_full")
            session.last_request_at = now
            self._admitted += 1
            self._inflight.add(session.token_hash)
            lease = GenerationLease(self, session)
            self._leases.add(lease)
            _log("public request accepted", session.profile_id)
            return lease

    def finish_generation(self, lease: GenerationLease) -> None:
        lease.release()
        with self._condition:
            self._leases.discard(lease)

    def busy(self) -> bool:
        with self._condition:
            return self._admitted > 0


def _bearer_token(request: Request, *, required: bool = True) -> str:
    value = request.headers.get("authorization", "")
    token = value[7:].strip() if value.lower().startswith("bearer ") else ""
    if required and not token:
        raise _error("unauthorized")
    return token


async def _payload(request: Request) -> dict[str, object]:
    try:
        raw = await request.body()
        payload = orjson.loads(raw) if raw else {}
    except Exception as exc:
        raise _error("invalid_request") from exc
    if not isinstance(payload, dict):
        raise _error("invalid_request")
    return payload


def _validate_message(payload: dict[str, object], settings: PublicApiSettings) -> str:
    if set(payload) != {"message"} or not isinstance(payload.get("message"), str):
        raise _error("invalid_request")
    message = payload["message"].strip()
    if not message:
        raise _error("invalid_request")
    if len(message) > settings.message_limit:
        raise _error("message_too_long")
    return message


def _stream_error(error: Exception, lease: GenerationLease) -> PublicApiError:
    if lease.timed_out:
        return _error("generation_timeout")
    if isinstance(error, GenerationBusyError):
        return _error("busy")
    if isinstance(error, GenerationQueueFullError):
        return _error("queue_full")
    if isinstance(error, GenerationCancelled):
        return _error("busy")
    status = ModelManager.get_instance().status()
    if not status.get("loaded"):
        return _error("model_unavailable")
    return _error("internal_error")


def public_chat_events(
    manager: PublicSessionManager,
    session: PublicSession,
    message: str,
    lease: GenerationLease,
):
    events: queue.Queue[tuple[str, object]] = queue.Queue()
    worker: threading.Thread | None = None
    finished = False
    try:
        lease.activate()
        request_id = uuid.uuid4().hex
        chat = normalize_chat_input(
            text=message,
            profile_id=session.profile_id,
            conversation_id=session.conversation_id,
            source="public_guest",
            request_id=f"public:{request_id}",
        )

        def generate() -> None:
            try:
                result = run_companion_turn(
                    chat,
                    on_delta=lambda text: events.put(("delta", text)),
                    priority="guest",
                    max_tokens=manager.settings.response_token_limit,
                    cancellation=lease.cancellation,
                    queue_deadline=lease.deadline,
                    allow_tool_context=False,
                    allow_initiative=False,
                )
                events.put(("done", result))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                manager.finish_generation(lease)

        worker = threading.Thread(target=generate, daemon=True, name="AkanePublicGuest")
        worker.start()
        yield _json_line({"type": "start", "request_id": request_id})
        while True:
            remaining = lease.deadline - time.monotonic()
            if remaining <= 0.0:
                lease.mark_timed_out()
                _log("public generation timed out", session.profile_id)
                yield _json_line(
                    {"type": "error", "error": {
                        "code": "generation_timeout",
                        "message": _error("generation_timeout").message,
                    }}
                )
                return
            try:
                kind, value = events.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if kind == "delta":
                yield _json_line({"type": "delta", "text": str(value)})
                continue
            if kind == "done":
                finished = True
                _log("public generation completed", session.profile_id)
                yield _json_line({"type": "done", "request_id": request_id})
                return
            error = _stream_error(
                value if isinstance(value, Exception) else RuntimeError("public generation failed"),
                lease,
            )
            yield _json_line(
                {"type": "error", "error": {"code": error.code, "message": error.message}}
            )
            return
    except PublicApiError as error:
        manager.finish_generation(lease)
        yield _json_line(
            {"type": "error", "error": {"code": error.code, "message": error.message}}
        )
    finally:
        if worker is not None and worker.is_alive() and not finished:
            lease.cancellation.set()
            worker.join(timeout=0.25)
        manager.finish_generation(lease)


def _session_payload(session: PublicSession, token: str) -> dict[str, object]:
    return {
        "profile_type": "guest",
        "session_token": token,
        "session_id": session.profile_id,
        "expires_at": 0,
    }


def register_public_routes(
    app: FastAPI,
    session_manager: PublicSessionManager | None = None,
) -> None:
    def current_manager(request: Request) -> PublicSessionManager:
        manager = session_manager or getattr(request.app.state, "public_sessions", None)
        if manager is None:
            raise _error("model_unavailable")
        return manager

    @app.get("/api/public/health")
    async def public_health(request: Request):
        try:
            manager = current_manager(request)
        except PublicApiError:
            return JSONResponse({"status": "offline", "streaming": True, "guest_enabled": True})
        model = ModelManager.get_instance()
        status = model.status()
        availability = (
            "offline"
            if not status.get("loaded")
            else "busy" if manager.busy() or model.inference_busy() else "available"
        )
        return JSONResponse(
            {"status": availability, "streaming": True, "guest_enabled": True}
        )

    @app.post("/api/public/session")
    async def public_session(request: Request):
        try:
            manager = current_manager(request)
            if await _payload(request):
                raise _error("invalid_request")
            token = _bearer_token(request, required=False)
            session = manager.resolve(token) if token else None
            if session is None:
                session, token = manager.create()
            return JSONResponse(_session_payload(session, token))
        except PublicApiError as error:
            return _error_response(error)

    @app.post("/api/public/chat")
    async def public_chat(request: Request):
        try:
            manager = current_manager(request)
            session = manager.resolve(_bearer_token(request))
            message = _validate_message(await _payload(request), manager.settings)
            if not ModelManager.get_instance().status().get("loaded"):
                raise _error("model_unavailable")
            lease = manager.acquire_generation(session)
        except PublicApiError as error:
            return _error_response(error)
        return StreamingResponse(
            public_chat_events(manager, session, message, lease),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/public/session/reset")
    async def public_session_reset(request: Request):
        try:
            manager = current_manager(request)
            if await _payload(request):
                raise _error("invalid_request")
            manager.reset(manager.resolve(_bearer_token(request)))
            return JSONResponse({"ok": True})
        except PublicApiError as error:
            return _error_response(error)

    @app.delete("/api/public/session")
    async def public_session_delete(request: Request):
        try:
            manager = current_manager(request)
            if await _payload(request):
                raise _error("invalid_request")
            manager.delete(_bearer_token(request))
            return JSONResponse({"ok": True})
        except PublicApiError as error:
            return _error_response(error)

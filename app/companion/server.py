"""Local HTTP transport retaining the popup's NDJSON event protocol."""

import asyncio
from contextlib import asynccontextmanager
import ipaddress
import json
import mimetypes
from pathlib import Path
import secrets
import socket
import threading
import time
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

from app.core.config import SETTINGS, coerce_bool
from app.ui.assets import resolve_ui_asset
from .next_adapter import ActiveTurn, ApplicationBusy, JobConflict, JobNotFound, NextCompanion
from app.ai.continuity.store import ContinuityConflict, SessionNotFound
from app.ai.contracts import TaskReference
from app.ai.conversation.model import ContextBudgetExceeded
from app.ai.host import CommitPending

STATIC = Path(__file__).resolve().parents[1] / 'ui' / 'static'
WEBSITE = SETTINGS.project_root / 'website' / 'dist'


def local_host(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def session_key(payload):
    return payload.get('session_id') or payload.get('conversation_id') or 'popup'


def safe_error(error):
    if isinstance(error, (JobNotFound, SessionNotFound)):
        return 'Record not found.'
    if isinstance(error, (ValueError, ApplicationBusy, CommitPending, JobConflict, ContextBudgetExceeded)):
        return str(error)
    print(f'[Akane:error] {type(error).__name__}: {error}', flush=True)
    return 'Akane could not finish that request. Check the local server logs.'


async def stream_events(companion, payload):
    session = session_key(payload)
    disconnected = threading.Event()
    events = asyncio.Queue()
    loop = asyncio.get_running_loop()
    request_id = payload.get('request_id', '')
    include = coerce_bool(payload.get('include_messages', True), True)
    active = None

    def enqueue(event):
        if events is not None:
            events.put_nowait(event)

    def emit(kind, **data):
        if not disconnected.is_set():
            event = dict(type=kind, generation_id=str(active.turn_id) if active else '',
                         request_id=request_id, **data)
            loop.call_soon_threadsafe(enqueue, event)

    def work():
        nonlocal active
        try:
            active = companion.reserve(session, request_id, include_messages=include, text=payload['message'],
                                       use_task=payload.get('use_task'))
            if disconnected.is_set() and isinstance(active, ActiveTurn):
                companion.cancel(session, generation_id=str(active.turn_id))
            start = {'messages': [*active.previous, {'role': 'user', 'content': payload['message']}]} if include else {}
            emit('start', **start)
            result = companion.run_turn(active, payload['message'],
                                    skip_memory=coerce_bool(payload.get('skip_memory', False)),
                                    on_delta=lambda text: emit('delta', content=text))
            if result['outcome'] == 'completed':
                emit('flush')
                emit('done', **result)
            elif result['outcome'] == 'cancelled':
                emit('cancelled', **result)
            else:
                result.setdefault('error', 'The reply did not complete.')
                emit('error', **result)
        except Exception as error:
            recovery = {'retry_commit': True} if (isinstance(error, CommitPending)
                and isinstance(active, ActiveTurn) and active.use_task is not None) else {}
            emit('error', error=safe_error(error), outcome='failed', **recovery)

    worker = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            event = await events.get()
            yield (json.dumps(event, ensure_ascii=False) + '\n').encode()
            if event['type'] in {'done', 'error', 'cancelled'}:
                await worker
                break
    finally:
        disconnected.set()
        events = None  # Drop queued payloads before the generation thread finishes unwinding.
        if isinstance(active, ActiveTurn):
            companion.cancel(session, generation_id=str(active.turn_id))


def create_app(companion=None):
    if companion is None:
        companion = NextCompanion()

    @asynccontextmanager
    async def lifespan(app):
        try:
            await asyncio.to_thread(companion.start)
            yield
        finally:
            await asyncio.to_thread(companion.close)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.companion = companion
    app.add_middleware(CORSMiddleware, allow_origins=list(SETTINGS.cors_allowed_origins),
                       allow_methods=['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
                       allow_headers=['Authorization', 'Content-Type', 'X-Akane-Token'])

    @app.middleware('http')
    async def private_api(request, next_handler):
        if request.url.path.startswith('/api/') and request.method != 'OPTIONS':
            origin = request.headers.get('origin')
            if origin and origin not in SETTINGS.cors_allowed_origins:
                return JSONResponse(dict(error='Origin is not allowed.'), status_code=403)
            # Prevent DNS rebinding even for requests with no Origin header.
            host = request.url.hostname or ''
            if not local_host(host):
                return JSONResponse(dict(error='Local host required.'), status_code=403)
            if SETTINGS.server_api_token:
                supplied = request.headers.get('authorization', '')
                supplied = supplied[7:] if supplied.lower().startswith('bearer ') else request.headers.get('x-akane-token', '')
                if not secrets.compare_digest(supplied, SETTINGS.server_api_token):
                    return JSONResponse(dict(error='Unauthorized.'), status_code=401)
            if request.query_params.get('profile_id', '') not in ('', 'local:owner', 'local', 'popup'):
                if request.url.path == '/api/tasks' or request.url.path.startswith('/api/tasks/'):
                    return JSONResponse(dict(error='Record not found.'), status_code=404)
                return JSONResponse(dict(error='Only the local owner profile is supported.'), status_code=400)
        return await next_handler(request)

    async def payload(request):
        # Read bounded chunks before JSON parsing, including chunked requests.
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > SETTINGS.max_input_chars * 6 + 8192:
                raise ValueError('Request is too large.')
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError('Request must be a JSON object.')
        if 'message' in result and (not isinstance(result['message'], str)
                or not result['message'].strip() or len(result['message']) > SETTINGS.max_input_chars):
            raise ValueError('Message is empty or exceeds the input size limit.')
        for field in ('session_id', 'conversation_id', 'request_id'):
            if field in result and (not isinstance(result[field], str) or len(result[field]) > 200):
                raise ValueError('Invalid session or request identifier.')
        if result.get('profile_id', '') not in ('', 'local:owner', 'local', 'popup'):
            if request.url.path.startswith('/api/tasks/'):
                raise JobNotFound('No accepted task in this session.')
            raise ValueError('Only the local owner profile is supported.')
        return result

    async def chat_payload(request):
        data = await payload(request)
        if 'message' not in data:
            raise ValueError('A nonempty message is required.')
        if 'use_task' in data:
            binding = data['use_task']
            if (not isinstance(binding, dict) or set(binding) != {'task_id', 'runtime_epoch'}
                    or not all(isinstance(value, str) and len(value) == 36 for value in binding.values())
                    or not data.get('request_id')):
                raise ValueError('Task selection requires its ID, runtime epoch and a request ID.')
            try:
                data['use_task'] = TaskReference(UUID(binding['task_id']), UUID(binding['runtime_epoch']))
            except ValueError as error:
                raise ValueError('Invalid task selection.') from error
        return data

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(dict(error=str(error)), status_code=400)

    @app.exception_handler(PermissionError)
    async def denied(request, error):
        return JSONResponse(dict(error=str(error)), status_code=403)

    @app.exception_handler(SessionNotFound)
    @app.exception_handler(JobNotFound)
    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse(dict(error='Record not found.'), status_code=404)

    @app.exception_handler(ContextBudgetExceeded)
    @app.exception_handler(ContinuityConflict)
    @app.exception_handler(CommitPending)
    @app.exception_handler(JobConflict)
    @app.exception_handler(ApplicationBusy)
    async def conflict(request, error):
        return JSONResponse(dict(error=safe_error(error)), status_code=409)

    @app.exception_handler(Exception)
    async def fault(request, error):
        return JSONResponse(dict(error=safe_error(error)), status_code=500)

    @app.get('/health')
    async def health():
        owners = companion.runtime_status()
        ready = owners['l1']['ready'] and (owners['l2']['ready'] or owners['l2'].get('lazy', False))
        payload = dict(status='ok' if ready else 'stopping', runtime=companion.runtime_name,
                       model_loaded=ready, model_loading=False)
        if owners['mode'] == 'independent':
            payload['inference'] = owners
        return JSONResponse(payload, status_code=200 if ready else 503)

    @app.get('/api/state')
    async def state(session_id='popup', conversation_id='', include_messages='1', profile_id=''):
        session = conversation_id or session_id
        snapshot = await asyncio.to_thread(companion.snapshot, session)
        result = dict(version=snapshot['revision'], session_id=snapshot['session_id'],
                      model=dict(loaded=companion.ready), internal_state={})
        if coerce_bool(include_messages, True):
            result['messages'] = snapshot['turns']
        capability = companion.task_capability()
        if capability is not None:
            result['background_tasks'] = capability
        return result

    @app.post('/api/session')
    async def select_session(request: Request):
        session = await asyncio.to_thread(companion.select_session, session_key(await payload(request)))
        return jsonable_encoder(session)

    @app.get('/api/context')
    async def context_status(session_id: str = 'popup'):
        status = await asyncio.to_thread(companion.optional_context_status, session_id)
        return jsonable_encoder({'active': status is not None, 'context': status})

    @app.post('/api/chat/retry-commit')
    async def retry_commit(request: Request):
        return await asyncio.to_thread(companion.retry_commit, session_key(await payload(request)))

    @app.get('/api/memory')
    async def memories(profile_id='', session_id='popup'):
        rows = await asyncio.to_thread(companion.memories)
        snapshot = await asyncio.to_thread(companion.snapshot, session_id)
        return dict(facts=[dict(content=m['text'], **m) for m in rows],
                    source_turns=[t for t in snapshot['turns'] if t['role'] == 'user'][-12:])

    @app.post('/api/memory')
    async def add_memory(request: Request):
        return dict(memories=await asyncio.to_thread(companion.edit_memory, proposal=await payload(request)))

    @app.patch('/api/memory/{memory_id}')
    async def correct_memory(memory_id, request: Request):
        return dict(memories=await asyncio.to_thread(companion.edit_memory, memory_id=memory_id,
                                                     proposal=await payload(request)))

    @app.delete('/api/memory/{memory_id}')
    async def delete_memory(memory_id):
        return dict(memories=await asyncio.to_thread(companion.edit_memory, memory_id=memory_id, delete=True))

    @app.delete('/api/memory')
    async def clear_memory():
        return dict(memories=await asyncio.to_thread(companion.edit_memory, clear=True))

    @app.post('/api/chat/cancel')
    async def cancel(request: Request):
        data = await payload(request)
        cancelled = await asyncio.to_thread(companion.cancel, session_key(data),
                                            request_id=data.get('request_id', ''),
                                            generation_id=data.get('generation_id', ''))
        return dict(ok=True, cancelled=cancelled)

    @app.post('/api/background')
    async def submit_job(request: Request):
        data = await payload(request)
        task = data.get('task')
        if not isinstance(task, str) or not task.strip() or len(task) > SETTINGS.max_input_chars:
            raise ValueError('A bounded nonempty informational task is required.')
        source = data.get('source_turn_id')
        if source is not None and not isinstance(source, str):
            raise ValueError('Invalid source turn identifier.')
        job = await asyncio.to_thread(companion.submit_job, session_key(data), task,
                                      UUID(source) if source is not None else None)
        return JSONResponse(jsonable_encoder(job), status_code=202)

    @app.get('/api/tasks')
    async def list_tasks(session_id: str = 'popup'):
        result = await asyncio.to_thread(companion.list_tasks, session_id)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})

    @app.get('/api/tasks/{task_id}')
    async def task_status(task_id: UUID, session_id: str = 'popup'):
        result = await asyncio.to_thread(companion.task, session_id, task_id)
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})

    @app.post('/api/tasks/{task_id}/cancel')
    async def cancel_task(task_id: UUID, request: Request):
        return await asyncio.to_thread(companion.task, session_key(await payload(request)), task_id, cancel=True)

    @app.get('/api/background/{job_id}')
    async def job_status(job_id: UUID, session_id: str = 'popup'):
        return jsonable_encoder(await asyncio.to_thread(companion.job, session_id, job_id))

    @app.post('/api/background/{job_id}/{action}')
    async def job_action(job_id: UUID, action: str, request: Request):
        if action not in {'attach', 'cancel'}:
            raise KeyError(action)
        return jsonable_encoder(await asyncio.to_thread(
            companion.job, session_key(await payload(request)), job_id, action))

    @app.post('/api/background/detach')
    async def detach(request: Request):
        await asyncio.to_thread(companion.unattach, session_key(await payload(request)))
        return {'ok': True}

    async def chat_result(data):
        text = data.get('message', '')
        session = session_key(data)
        # Exact slash commands are explicit transport controls, not NLP routing.
        if text in {'/clear', '/reset_chat', '/forget_me'}:
            if data.get('use_task') is not None:
                raise ValueError('Task results can only be discussed in a chat message.')
            await asyncio.to_thread(companion.reset, session, forget=text == '/forget_me')
            return dict(reply='', messages=[], notice='Chat reset.')
        return None

    @app.post('/api/chat/stream')
    async def stream(request: Request):
        data = await chat_payload(request)
        command = await chat_result(data)
        if command is not None: return JSONResponse(command)
        return StreamingResponse(stream_events(companion, data), media_type='application/x-ndjson',
                                 headers={'Cache-Control': 'no-store, no-transform', 'X-Accel-Buffering': 'no'})

    @app.post('/api/chat')
    async def chat(request: Request):
        data = await chat_payload(request)
        command = await chat_result(data)
        if command is not None: return command
        def run():
            active = companion.reserve(session_key(data), data.get('request_id', ''),
                                       include_messages=coerce_bool(data.get('include_messages', True), True),
                                       text=data['message'], use_task=data.get('use_task'))
            return companion.run_turn(active, data['message'],
                                      skip_memory=coerce_bool(data.get('skip_memory', False)),
                                      on_delta=lambda _text: None)
        result = await asyncio.to_thread(run)
        if result['outcome'] != 'completed':
            result.setdefault('error', 'Reply cancelled.' if result['outcome'] == 'cancelled'
                              else 'The reply did not complete.')
            return JSONResponse(result, status_code=409 if result['outcome'] == 'cancelled' else 500)
        return result

    @app.get('/{asset_path:path}')
    async def assets(asset_path, request: Request):
        allowed = {'app.js', 'styles.css', 'subtitles.js', 'background-tasks.js', 'index.html'}
        if not asset_path:
            path = STATIC / 'index.html' if request.query_params.get('popup_role') or not (WEBSITE / 'index.html').exists() else WEBSITE / 'index.html'
        elif asset_path in allowed:
            path = STATIC / asset_path
        else:
            path = resolve_ui_asset('/' + asset_path)
            if path is None:
                candidate = (WEBSITE / asset_path).resolve()
                path = candidate if candidate.is_relative_to(WEBSITE.resolve()) else None
        if path is None or not path.is_file():
            return JSONResponse(dict(error='Not found.'), status_code=404)
        return FileResponse(path, media_type=mimetypes.guess_type(str(path))[0])

    return app


class _Server(uvicorn.Server):
    def handle_exit(self, sig, frame):
        self.config.app.state.companion.begin_shutdown()
        super().handle_exit(sig, frame)


class BackgroundServer:
    def __init__(self, host, port, app=None):
        if not local_host(host):
            raise ValueError('The companion server must bind a loopback address.')
        self._server = _Server(uvicorn.Config(app or create_app(), host=host, port=port,
                                             log_level='warning', access_log=False))
        family = socket.AF_INET6 if ':' in host else socket.AF_INET
        self._socket = socket.create_server((host, port), family=family)
        self._thread = None

    def run(self):
        with self._socket:
            self._server.run(sockets=[self._socket])
        if not self._server.started:
            raise RuntimeError('Akane Next did not start. See the startup error above.')

    def shutdown(self):
        self._server.config.app.state.companion.begin_shutdown()
        self._server.should_exit = True
        if self._thread:
            self._thread.join()


def serve(host=SETTINGS.server_host, port=SETTINGS.server_port):
    server = BackgroundServer(host, port)
    print(f'Akane Next starting on {host}:{port}. Readiness is available at /health.', flush=True)
    server.run()


def serve_in_thread(host=SETTINGS.server_host, port=SETTINGS.server_port):
    server = BackgroundServer(host, port)
    thread = threading.Thread(target=server.run, daemon=True, name='AkaneLocalServer')
    server._thread = thread
    thread.start()
    while thread.is_alive() and not server._server.started:
        time.sleep(0.01)
    if not server._server.started:
        raise RuntimeError('Akane local server could not start.')
    return server, thread


if __name__ == '__main__':
    serve()

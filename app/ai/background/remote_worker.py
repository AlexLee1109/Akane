"""Run a single authenticated Qwen inference service with no Akane state."""

import argparse
import asyncio
from collections import OrderedDict
from contextlib import closing
from hashlib import sha256
from hmac import compare_digest
from math import isfinite
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from uuid import uuid4

from ..contracts import ContextRecord
from ..conversation.cancellation import CancellationSignal, CancellationToken, GenerationCancelled
from ..conversation.native import NativeModel
from ..provenance import SourceAttribution
from .l2_memory import guarded_l2_load
from .native import BackgroundLimits, _finding_text, prepare_automatic_finding
from .remote import (
    HEALTH, LIMITS, MAX_WIRE_BYTES, RESOURCE_HOST, loopback_host, read_task, task_message,
    validate_worker_token, wire_json,
)


class _InferenceWorker:
    def __init__(self, model, timeout):
        if model.architecture != 'qwen35' or model.context_size < 2048:
            raise ValueError('The remote worker requires a Qwen3.5 context of at least 2048 tokens.')
        if not isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError('Inference timeout must be positive and at most 120 seconds.')
        self.model = model
        self.limits = BackgroundLimits(LIMITS.prompt_tokens, LIMITS.output_tokens, LIMITS.result_characters, timeout)
        self._lock = Lock()
        self._results = OrderedDict()

    def finding(self, task_id, text, digest, signal):
        from fastapi import HTTPException

        if not self._lock.acquire(blocking=False):
            raise HTTPException(429, 'Worker is occupied.')
        try:
            token = CancellationToken(task_id, signal, _deadline=monotonic() + self.limits.max_seconds,
                                      _deadline_origin='remote_worker')
            token.raise_if_cancelled()
            saved = self._results.get(task_id)
            if saved is not None:
                if not compare_digest(saved[0], digest):
                    raise HTTPException(409, 'Task ID already has a different source.')
                return saved[1]
            source = ContextRecord(text, (SourceAttribution('user', f'{task_id}:user'),))
            prompt = prepare_automatic_finding(self.model, text, (source,), self.limits)
            if len(prompt.tokens) > min(self.limits.max_prompt_tokens, self.model.context_size - prompt.max_tokens):
                raise HTTPException(422, 'Task exceeds the native prompt limit.')
            wire = ''
            with closing(self.model.stream(prompt, cancellation=token)) as output:
                for piece in output:
                    token.raise_if_cancelled()
                    wire += piece
                    if len(wire.encode('utf-8')) > MAX_WIRE_BYTES:
                        raise ValueError('Native finding exceeded the output bound.')
            token.raise_if_cancelled()
            if self.model.last_metrics is None or self.model.last_metrics.finish_reason != 'stop':
                raise ValueError('Native finding did not finish normally.')
            result = wire_json(task_message(task_id, _finding_text(wire, self.limits.max_result_chars), digest))
            self._results[task_id] = (digest, result)
            if len(self._results) > 64:
                self._results.popitem(last=False)
            return result
        finally:
            self._lock.release()


def create_worker_app(model, token, *, timeout=20.0):
    from fastapi import FastAPI, HTTPException, Request, Response
    from starlette.requests import ClientDisconnect

    validate_worker_token(token)
    worker = _InferenceWorker(model, timeout)
    probe = 'Verify the inference service by returning the sum of two notebooks and one notebook.'
    worker.finding(uuid4(), probe, sha256(probe.encode()).hexdigest(), CancellationSignal())
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    capabilities = {**HEALTH, 'resource_host': RESOURCE_HOST if isinstance(model, NativeModel) else None}

    def authorize(request):
        supplied = request.headers.get('authorization', '').encode('utf-8')
        expected = f'Bearer {token}'.encode('ascii')
        if not compare_digest(supplied, expected):
            raise HTTPException(401, 'Worker authentication required.')

    @app.get('/v1/health')
    async def health(request: Request):
        authorize(request)
        return capabilities

    @app.post('/v1/finding')
    async def finding(request: Request):
        authorize(request)
        if capabilities['resource_host'] and request.headers.get('x-akane-resource-host') == RESOURCE_HOST:
            raise HTTPException(409, 'Native worker and foreground share a resource host.')
        if (request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json'
                or request.headers.get('content-encoding', 'identity').lower() != 'identity'):
            raise HTTPException(415, 'Use uncompressed JSON.')
        data = bytearray()
        try:
            async with asyncio.timeout(timeout):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > MAX_WIRE_BYTES:
                        raise HTTPException(413, 'Task exceeds the transport byte limit.')
                    data.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, 'Task upload timed out.') from None
        except ClientDisconnect:
            return Response(status_code=499)
        try:
            task = read_task(data)
        except (ValueError, RecursionError):
            raise HTTPException(422, 'Invalid task ID, source or message.') from None
        signal = CancellationSignal('remote_disconnect')
        result = asyncio.create_task(asyncio.to_thread(worker.finding, *task, signal))
        disconnect = asyncio.create_task(request.receive())
        try:
            done, _ = await asyncio.wait((result, disconnect), return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                signal.set()
                return Response(status_code=499)
            return Response(result.result(), media_type='application/json')
        except GenerationCancelled:
            raise HTTPException(504, 'Worker inference deadline expired.') from None
        except (ValueError, RuntimeError):
            raise HTTPException(502, 'Worker could not produce a valid finding.') from None
        finally:
            signal.set()
            disconnect.cancel()
            await asyncio.gather(disconnect, result, return_exceptions=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--context-size', type=int, default=2048)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--tls-cert')
    parser.add_argument('--tls-key')
    args = parser.parse_args()
    token = os.environ.get('AKANE_L2_WORKER_TOKEN', '')
    validate_worker_token(token)
    tls = bool(args.tls_cert and args.tls_key)
    if bool(args.tls_cert) != bool(args.tls_key) or not (tls or loopback_host(args.host)):
        parser.error('A non-loopback worker requires both TLS certificate and key.')
    if args.context_size < 2048 or not 1 <= args.port <= 65535 or not 0 < args.timeout <= 120:
        parser.error('Invalid context size, port or timeout.')
    import uvicorn

    signal = CancellationSignal()
    cancellation = CancellationToken(uuid4(), signal)
    with closing(guarded_l2_load(
        lambda cancelled: NativeModel(args.model, context_size=args.context_size, load_cancelled=cancelled),
        cancellation, signal.set, model_path=args.model,
    )) as model:
        app = create_worker_app(model, token, timeout=args.timeout)
        uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=args.tls_cert, ssl_keyfile=args.tls_key,
                    access_log=False, limit_concurrency=16)


if __name__ == '__main__':
    main()

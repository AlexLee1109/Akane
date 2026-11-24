"""Bounded inference transport. The local scheduler owns all task state."""

from hashlib import sha256
from http.client import HTTPConnection, HTTPSConnection, HTTPException
from ipaddress import ip_address
import json
from socket import SHUT_RDWR, gethostname
from threading import Event, Thread
from time import monotonic
from urllib.parse import urlsplit
from uuid import UUID

from ..contracts import ContextRecord, DelegationResult
from ..delegation import PRODUCTION_AUTODELEGATION_LIMITS
from ..provenance import SourceAttribution


MAX_WIRE_BYTES = 32 * 1024
HEALTH = {'protocol': 1, 'architecture': 'qwen35', 'thinking': False, 'ready': True}
LIMITS = PRODUCTION_AUTODELEGATION_LIMITS
RESOURCE_HOST = sha256(gethostname().casefold().encode()).hexdigest()


def loopback_host(host):
    if host == 'localhost':
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def remote_worker_url(value):
    try:
        url = urlsplit(value)
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username is not None or url.password is not None
                or url.path not in ('', '/') or url.query or url.fragment
                or url.port == 0 or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
                or (url.scheme == 'http' and not loopback_host(url.hostname))):
            raise ValueError
        return url
    except ValueError:
        raise ValueError('Remote L2 requires a credential-free HTTPS origin or a loopback HTTP origin.') from None


def validate_worker_token(token):
    if not isinstance(token, str) or not 32 <= len(token) <= 256 or any(not 33 <= ord(char) <= 126 for char in token):
        raise ValueError('Remote L2 requires a bearer token of 32 to 256 printable ASCII characters.')


def wire_json(data):
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > MAX_WIRE_BYTES:
        raise ValueError('Remote L2 message exceeds its byte limit.')
    return encoded


def read_object(data, fields):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError('Remote L2 message contains duplicate fields.')
        return result

    if len(data) > MAX_WIRE_BYTES:
        raise ValueError('Remote L2 message exceeds its byte limit.')
    obj = json.loads(data.decode('utf-8'), object_pairs_hook=unique)
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ValueError('Remote L2 message has unexpected fields.')
    return obj


def task_message(task_id, text, source_sha256):
    return {'task_id': str(task_id), 'source_sha256': source_sha256, 'text': text}


def read_task(data):
    message = read_object(data, {'task_id', 'source_sha256', 'text'})
    value, digest, text = message['task_id'], message['source_sha256'], message['text']
    if not isinstance(value, str) or not isinstance(text, str) or not text.strip() or len(text) > LIMITS.source_characters:
        raise ValueError('Remote L2 task has invalid identity or source text.')
    task_id = UUID(value)
    if str(task_id) != value or sha256(text.encode()).hexdigest() != digest:
        raise ValueError('Remote L2 task ID or exact source digest does not match.')
    return task_id, text, digest


def read_finding(data, task_id, digest):
    message = read_object(data, {'task_id', 'source_sha256', 'text'})
    if message['task_id'] != str(task_id) or message['source_sha256'] != digest:
        raise ValueError('Remote L2 returned a different task or source.')
    from .native import _finding_text
    return _finding_text(wire_json({'text': message['text']}).decode(), LIMITS.result_characters)


class RemoteL2:
    """An authenticated origin with no filesystem or transcript access."""

    independent_resources = True

    def __init__(self, endpoint, token, *, timeout):
        self._origin = remote_worker_url(endpoint)
        validate_worker_token(token)
        self._token = token
        self.timeout = timeout
        self.ready = False
        self.colocated = False
        self.last_transport = None

    def check_ready(self):
        try:
            value = read_object(self._exchange('GET', '/v1/health', None, min(2.0, self.timeout)),
                                set(HEALTH) | {'resource_host'})
            owner = value.pop('resource_host')
            self.colocated = owner == RESOURCE_HOST
            independent = (loopback_host(self._origin.hostname) if owner is None else
                           isinstance(owner, str) and len(owner) == 64
                           and all(char in '0123456789abcdef' for char in owner) and not self.colocated)
            self.ready = (independent and value == HEALTH and type(value['protocol']) is int
                          and value['thinking'] is False and value['ready'] is True)
        except (RuntimeError, ValueError):
            self.ready = False
        return self.ready

    def execute(self, request, *, cancellation):
        if request.detached is None or not request.automatic:
            raise ValueError('Remote L2 accepts only an authorized detached task.')
        source, = request.source_context
        if (source.text != request.task or len(source.sources) != 1 or source.sources[0].owner != 'user'
                or sha256(request.task.encode()).hexdigest() != request.detached.source_sha256):
            raise ValueError('Remote L2 requires the exact current user source.')
        if not self.ready and not self.check_ready():
            raise RuntimeError('Remote L2 is unavailable or does not own independent resources.')
        digest = request.detached.source_sha256
        body = wire_json(task_message(request.request_id, request.task, digest))
        try:
            raw = self._exchange('POST', '/v1/finding', body, self.timeout, cancellation)
            text = read_finding(raw, request.request_id, digest)
        except (RuntimeError, ValueError):
            self.ready = False
            raise
        cancellation.raise_if_cancelled()
        finding = ContextRecord(text, (SourceAttribution('l2', f'model-generated:{request.request_id}'),
                                       *source.sources))
        return DelegationResult(request.request_id, request.conversation_id, request.turn_id, 'completed', (finding,))

    def _exchange(self, method, path, body, timeout, cancellation=None):
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        origin = self._origin
        host = '127.0.0.1' if origin.hostname == 'localhost' and origin.scheme == 'http' else origin.hostname
        connection_type = HTTPSConnection if origin.scheme == 'https' else HTTPConnection
        connection = connection_type(host, origin.port, timeout=min(timeout, 2.0))
        done = Event()
        started = monotonic()
        size = 0
        interrupted = Event()

        def interrupt(sock):
            while not done.wait(0.02):
                if monotonic() - started >= timeout or (cancellation is not None and cancellation.cancelled):
                    interrupted.set()
                    try:
                        sock.shutdown(SHUT_RDWR)
                    except OSError:
                        pass
                    return

        monitor = None
        try:
            connection.connect()
            sock = connection.sock
            sock.settimeout(max(0.001, timeout - (monotonic() - started)))
            monitor = Thread(target=interrupt, args=(sock,), name='akane-remote-l2-cancel', daemon=True)
            monitor.start()
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            connection.request(method, path, body=body, headers={
                'Authorization': f'Bearer {self._token}', 'Content-Type': 'application/json',
                'Accept': 'application/json', 'Connection': 'close',
                'X-Akane-Resource-Host': RESOURCE_HOST,
            })
            with connection.getresponse() as response:
                if response.status != 200:
                    raise RuntimeError(f'Remote L2 returned HTTP {response.status}.')
                if response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                    raise ValueError('Remote L2 responded with an unexpected content type.')
                if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                    raise ValueError('Remote L2 returned unsupported content encoding.')
                expected = response.length
                if expected is not None and expected > MAX_WIRE_BYTES:
                    raise ValueError('Remote L2 response exceeds its byte limit.')
                data = response.read(MAX_WIRE_BYTES + 1)
                size = len(data)
                if size > MAX_WIRE_BYTES:
                    raise ValueError('Remote L2 response exceeds its byte limit.')
                if expected is not None and size != expected:
                    raise HTTPException('Incomplete worker response.')
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if interrupted.is_set() or monotonic() - started >= timeout:
                raise RuntimeError('Remote L2 transport timed out.')
            return data
        except (OSError, HTTPException) as error:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            if isinstance(error, TimeoutError) or interrupted.is_set() or monotonic() - started >= timeout:
                raise RuntimeError('Remote L2 transport timed out.') from None
            raise RuntimeError('Remote L2 connection failed.') from None
        finally:
            done.set()
            connection.close()
            if monitor is not None:
                monitor.join()
            if method == 'POST':
                self.last_transport = {'seconds': monotonic() - started, 'request_bytes': len(body), 'response_bytes': size}

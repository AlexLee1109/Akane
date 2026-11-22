"""Application ownership and transport mapping for Akane Next."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, closing, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
from threading import Condition
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from app.core.config import SETTINGS

from app.ai.automatic_contracts import AutomaticContextPolicy
from app.ai.background.contracts import TaskState
from app.ai.background.l2_memory import guarded_l2_load
from app.ai.background.native import BackgroundLimits, NativeBackgroundExecutor
from app.ai.background.residency import LazyL2
from app.ai.background.scheduler import InferenceScheduler, JobConflict, JobNotFound
from app.ai.continuity.json_store import JSONContinuityStore
from app.ai.continuity.memory import JSONMemoryStore
from app.ai.continuity.memory_contracts import MemoryApproval
from app.ai.continuity.store import ContinuityConflict, SessionNotFound
from app.ai.contracts import ContextRecord, TaskReference
from app.ai.conversation.cancellation import CancellationToken, GenerationCancelled, TurnCancellationOwner
from app.ai.conversation.interaction import InteractionModel
from app.ai.conversation.model import DialogueModel
from app.ai.conversation.native import NativeModel
from app.ai.delegation import PRODUCTION_AUTODELEGATION_LIMITS
from app.ai.host import ConversationHost, InferenceDraining


_PROFILE_ID = uuid5(NAMESPACE_URL, "https://akane.local/profiles/local-owner")


class ApplicationBusy(RuntimeError):
    """The host is in use or closing."""


@dataclass(frozen=True)
class ActiveTurn:
    session: str
    request_id: str
    token: CancellationToken
    previous: list[dict] | None
    use_task: TaskReference | None = None

    @property
    def turn_id(self):
        return self.token.turn_id


@dataclass(frozen=True, slots=True)
class ReplayedTurn:
    turn_id: UUID
    previous: list[dict] | None
    result: dict


@dataclass(frozen=True, slots=True)
class _RequestRecord:
    turn_id: UUID
    use_task: TaskReference | None


@dataclass(frozen=True, slots=True)
class _ModelRuntime:
    model: DialogueModel | LazyL2
    scheduler: InferenceScheduler


def _open_runtime(startup, path: Path, *, role: str = 'shared', context_size: int = 4096) -> _ModelRuntime:
    if role == 'shared':
        model = NativeModel(path)
        startup.callback(model.close)
    else:
        family = {'l1': 'gemma4', 'l2': 'qwen35'}[role]
        try:
            model = NativeModel(path, context_size=context_size)
            startup.callback(model.close)
            if model.architecture != family:
                raise ValueError(f'Expected {family} metadata, found {model.architecture!r}.')
        except Exception as error:
            raise RuntimeError(f'{role.upper()} initialization failed for {path.name}: {error}') from error
    scheduler = InferenceScheduler(prioritize_tasks=True) if role == 'l2' else InferenceScheduler()
    startup.callback(scheduler.shutdown)
    return _ModelRuntime(model, scheduler)


def _validate_lazy_file(path):
    with path.open('rb') as stream:
        header = stream.read(8)
    if len(header) != 8 or header[:4] != b'GGUF' or int.from_bytes(header[4:], 'little') not in (2, 3):
        raise ValueError('Configured L2 file must have a supported GGUF header.')


def _open_lazy_runtime(startup, path, *, context_size, idle_seconds):
    model_factory, guard = NativeModel, guarded_l2_load

    def factory(cancelled):
        model = model_factory(path, context_size=context_size, load_cancelled=cancelled)
        if model.architecture != 'qwen35' or model.context_size != context_size:
            model.close()
            raise ValueError('L2 metadata or context does not match the configured Qwen owner.')
        return model

    def load(cancellation, on_abort):
        return guard(factory, cancellation, on_abort, model_path=path)

    owner = LazyL2(path, context_size, idle_seconds, load)
    scheduler = InferenceScheduler(prioritize_tasks=True, lazy_l2=owner)
    startup.callback(scheduler.shutdown)
    return _ModelRuntime(owner, scheduler)


class NextCompanion:
    """Application host with one active turn and shared or independent model ownership."""

    runtime_name = "akane_next"

    def __init__(self, *, policy: AutomaticContextPolicy | None = None) -> None:
        configured = Path(os.environ.get("AKANE_NEXT_DATA_DIR", "akane_next/data")).expanduser()
        self.root = configured if configured.is_absolute() else SETTINGS.project_root / configured
        if policy is None:
            policy = AutomaticContextPolicy(segment_exchanges=4, periodic_interval_seconds=float(
                os.environ.get("AKANE_NEXT_REFRESH_SECONDS", "300")))
        self.policy = policy
        self._lock = Condition()
        self._owner = TurnCancellationOwner()
        self._active: ActiveTurn | None = None
        self._closing = False
        self._store = self._l1 = self._l2 = self._host = None
        self._server_lock = None
        self._settings = SETTINGS
        self._auto_delegation = self._settings.next_auto_delegation
        self._async = (self._settings.next_runtime == 'independent' and self._auto_delegation
                       and self._settings.next_async_delegation)
        self._request_ids: OrderedDict[tuple[UUID, datetime], OrderedDict[str, _RequestRecord]] = OrderedDict()

    @staticmethod
    def _session_uuid(session: str) -> UUID:
        return uuid5(_PROFILE_ID, f"session:{session or 'popup'}")

    def _available(self):
        if self._closing:
            raise ApplicationBusy("Akane is shutting down.")

    def _idle(self):
        self._available()
        if self._active is not None:
            raise ApplicationBusy("A reply is already in progress.")

    def _ensure_store(self):
        if self._store is None:
            # Source data remains in its own namespace, even with overrides.
            root, legacy = self.root.resolve(), SETTINGS.data_dir.resolve()
            if root == legacy or root.is_relative_to(legacy / "companion-v1"):
                raise ValueError("Akane Next storage must be separate from V2 source data.")
            self._store = JSONContinuityStore(self.root, _PROFILE_ID)
        return self._store

    def _session(self, key):
        store = self._ensure_store()
        identifier = self._session_uuid(key)
        try:
            return store.get_session(identifier)
        except SessionNotFound:
            try:
                return store.create_session(identifier)
            except ContinuityConflict:
                return store.get_session(identifier)

    def _close_store(self):
        if self._store is not None:
            self._store.close()
            self._store = None

    def start(self):
        with self._lock, ExitStack() as startup:
            self._available()
            if self._host is not None:
                return
            settings = self._settings
            settings.validate()
            store = self._ensure_store()
            startup.callback(self._close_store)
            lease = os.open(".server.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                            0o600, dir_fd=store._profile_dir)
            startup.callback(os.close, lease)
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another Akane Next server already owns this data directory.") from error
            session = self._session("popup")
            JSONMemoryStore(store).list_active()
            if settings.next_runtime == 'independent':
                l1_path, l2_path = settings.next_l1_model_path, settings.next_l2_model_path
                for role, path in (('L1', l1_path), ('L2', l2_path)):
                    if not path.is_file():
                        raise FileNotFoundError(f'{role} model file does not exist: {path}')
                if l1_path.samefile(l2_path):
                    raise ValueError('Independent L1 and L2 require distinct GGUF files.')
                if settings.next_l2_lazy_load:
                    _validate_lazy_file(l2_path)
                l1 = _open_runtime(startup, l1_path, role='l1', context_size=settings.next_l1_context_size)
                if settings.next_l2_lazy_load:
                    l2 = _open_lazy_runtime(startup, l2_path, context_size=settings.next_l2_context_size,
                                            idle_seconds=settings.next_l2_idle_unload_seconds)
                else:
                    l2 = _open_runtime(startup, l2_path, role='l2', context_size=settings.next_l2_context_size)
            else:
                l1 = l2 = _open_runtime(startup, Path(settings.model_path))
            limits = PRODUCTION_AUTODELEGATION_LIMITS
            worker_limits = BackgroundLimits(limits.prompt_tokens, limits.output_tokens,
                                             limits.result_characters, limits.seconds)
            host = ConversationHost(
                _PROFILE_ID, session.conversation_id,
                InteractionModel(l1.model, decision_protocol='intent-only' if self._auto_delegation else 'union'),
                l1.scheduler, NativeBackgroundExecutor(_PROFILE_ID, l2.model, l2.scheduler,
                    automatic_limits=worker_limits if self._auto_delegation else None), store=store,
                automatic_context=self.policy,
                automatic_delegation=self._auto_delegation, delegation_limits=limits,
                background_scheduler=l2.scheduler, maintenance_model=l2.model,
                asynchronous_delegation=self._async,
                overlap_safe=all(getattr(runtime.model, 'supports_independent_overlap', False)
                                 for runtime in (l1, l2)),
            )
            self._l1, self._l2, self._host, self._server_lock = l1, l2, host, lease
            startup.pop_all()

    @property
    def ready(self):
        with self._lock:
            return self._host is not None and not self._closing

    @property
    def _native_model(self):
        return self._l1.model if self._l1 is not None else None

    @property
    def _scheduler(self):
        return self._l2.scheduler if self._l2 is not None else None

    def _runtimes(self):
        return tuple(runtime for runtime in (self._l1, self._l2 if self._l2 is not self._l1 else None)
                     if runtime is not None)

    def runtime_status(self):
        with self._lock:
            mode = self._settings.next_runtime
            ready = self._host is not None and not self._closing

            def describe(role, runtime):
                return {'owner': 'shared' if mode == 'shared' else role, 'ready': ready,
                        'model': runtime.model.model_path.name if runtime else None,
                        'architecture': getattr(runtime.model, 'architecture', None) if runtime else None,
                        'context_size': runtime.model.context_size if runtime else None}

            l2 = describe('l2', self._l2)
            if self._l2 is not None and isinstance(self._l2.model, LazyL2):
                l2.update(self._l2.model.snapshot(), lazy=True, **self._l2.scheduler.background_activity())
                l2['ready'] = ready and l2['lifecycle_state'] == 'ready'
            return {'mode': mode, 'l1': describe('l1', self._l1), 'l2': l2,
                    'maintenance_owner': 'l2' if mode == 'independent' else 'shared'}

    def _select(self, session):
        if self._host is None:
            raise ApplicationBusy("Akane is not ready.")
        current = self._session(session)
        if current.conversation_id != self._host.session.conversation_id:
            self._host.switch_session(current.conversation_id)
        elif current.created_at != self._host.session.created_at:
            raise JobConflict("Session was deleted and replaced.")
        return current

    def select_session(self, session):
        with self._lock:
            self._idle()
            return self._select(session)

    def snapshot(self, session="popup"):
        with self._lock:
            self._available()
            store = self._ensure_store()
            identifier = self._session_uuid(session)
            # An existing session needs one validated archive read.
            try:
                with store._locked():
                    data, entries = store._load(identifier)
            except SessionNotFound:
                self._session(session)
                with store._locked():
                    data, entries = store._load(identifier)
            return {
                "revision": data["revision"], "session_id": str(identifier),
                "turns": [
                    {"id": str(entry.turn_id), "role": entry.role, "content": entry.text,
                     "timestamp": entry.created_at.isoformat(), "outcome": entry.outcome}
                    for entry in entries
                ],
            }

    def messages(self, session="popup"):
        return self.snapshot(session)["turns"]

    @staticmethod
    def _task_data(receipt):
        return {name: str(value) if isinstance(value, UUID) else value for name, value in asdict(receipt).items()}

    def task_capability(self):
        with self._lock:
            return {'runtime_epoch': str(self._host.runtime_epoch)} if self._async and self.ready else None

    def _task_session(self, session):
        self._available()
        if not self._async or self._host is None or len(session) > 200:
            raise JobNotFound('No accepted task in this session.')
        return self._store.get_session(self._session_uuid(session))

    def _task_status(self, job):
        data = self._task_data(self._host._receipt(job))
        if data['status'] == 'completed':
            data['status'] = 'ready'
        data['label'] = ' '.join(job.request.task.split())[:96]
        return data

    def list_tasks(self, session):
        with self._lock:
            current = self._task_session(session)
            jobs = self._l2.scheduler.detached_jobs(_PROFILE_ID, current.conversation_id, current.created_at)
            return {'runtime_epoch': str(self._host.runtime_epoch), 'tasks': [self._task_status(job) for job in jobs]}

    def task(self, session, identifier, *, cancel=False):
        with self._lock:
            current = self._task_session(session)
            job = self._host.scoped_task(current, identifier)
            if cancel and job.status in (TaskState.PENDING, TaskState.LOADING, TaskState.RUNNING):
                self._l2.scheduler.cancel_background(_PROFILE_ID, identifier)
                job = self._host.scoped_task(current, identifier)
            return self._task_status(job)

    def _replay(self, session, current, record, text, include_messages):
        turn_id = record.turn_id
        try:
            if record.use_task is None:
                receipt, (user, reply) = self._host.detached_exchange(current.conversation_id, turn_id)
            else:
                task = record.use_task
                job = self._host.scoped_task(current, task.task_id)
                if job.delivered_turn_id != turn_id or job.request.detached.runtime_epoch != task.runtime_epoch:
                    raise JobNotFound('The result exchange is unavailable.')
                pair = tuple(entry for entry in self._store.transcript(current.conversation_id) if entry.turn_id == turn_id)
                if len(pair) != 2 or pair[1].outcome != 'completed':
                    raise JobNotFound('The result exchange is unavailable.')
                user, reply = pair
                receipt = self._host._receipt(job)
        except JobNotFound as error:
            raise JobConflict('This request ID has already been admitted for this session.') from error
        if text != user.text:
            raise JobConflict('This request ID was admitted with different text.')
        result = {'reply': reply.text, 'outcome': 'completed', 'task': self._task_data(receipt)}
        previous = None
        if include_messages:
            snapshot = self.snapshot(session)
            result.update(messages=snapshot['turns'], revision=snapshot['revision'])
            first = next(index for index, item in enumerate(snapshot['turns']) if item['id'] == str(turn_id))
            previous = snapshot['turns'][:first]
        return ReplayedTurn(turn_id, previous, result)

    def reserve(self, session, request_id, *, include_messages, text=None, use_task=None):
        with self._lock:
            self._idle()
            current = self._select(session)
            try:
                self._host.require_inference_ready()
            except InferenceDraining as error:
                raise ApplicationBusy(str(error)) from error
            requests = None
            key = (current.conversation_id, current.created_at)
            if self._auto_delegation and request_id:
                requests = self._request_ids.get(key, OrderedDict())
                if request_id in requests:
                    if self._async and text is not None:
                        if requests[request_id].use_task != use_task:
                            raise JobConflict('This request ID used a different task selection.')
                        return self._replay(session, current, requests[request_id], text, include_messages)
                    raise JobConflict('This request ID has already been admitted for this session.')
            if use_task is not None:
                if not self._async or use_task.runtime_epoch != self._host.runtime_epoch:
                    raise JobNotFound('No accepted task in this session.')
                job = self._host.scoped_task(current, use_task.task_id)
                if job.status != TaskState.COMPLETED or job.invalidated or job.delivered_turn_id is not None:
                    raise JobConflict('The selected task is unavailable for delivery.')
            previous = self.messages(session) if include_messages else None
            active = ActiveTurn(session, request_id, self._owner.begin(uuid4()), previous, use_task)
            self._active = active
            if requests is not None:
                requests[request_id] = _RequestRecord(active.turn_id, use_task)
                if len(requests) > 128:
                    requests.popitem(last=False)
                self._request_ids[key] = requests
                self._request_ids.move_to_end(key)
                if len(self._request_ids) > 32:
                    self._request_ids.popitem(last=False)
            return active

    def run_turn(self, active, text, *, skip_memory, on_delta):
        if isinstance(active, ReplayedTurn):
            on_delta(active.result['reply'])
            return active.result
        pieces = []
        outcome, error = "failed", None
        receipt = None
        try:
            with closing(self._host.stream(text, cancellation=active.token, skip_memory=skip_memory,
                                           use_task=active.use_task)) as events:
                for event in events:
                    if event.kind == "reply_text_delta":
                        pieces.append(event.text)
                        on_delta(event.text)
                    elif event.kind == "reply_completed":
                        outcome = "completed"
                        receipt = event.task
                    elif event.kind == "reply_cancelled":
                        outcome = "cancelled"
                    elif event.kind == "reply_failed":
                        error = event.message
            result = {"reply": "".join(pieces), "outcome": outcome}
            if receipt is not None:
                result['task'] = self._task_data(receipt)
            if error:
                result["error"] = error
            with self._lock:
                if active.previous is not None and not self._closing:
                    snapshot = self.snapshot(active.session)
                    result.update(messages=snapshot["turns"], revision=snapshot["revision"])
            return result
        except GenerationCancelled:
            return {"reply": "".join(pieces), "outcome": "cancelled"}
        finally:
            with self._lock:
                try:
                    metrics = self._host.delegation_metrics() if self._auto_delegation and SETTINGS.timing_enabled else None
                finally:
                    self._owner.finish(active.token.turn_id)
                    self._active = None
                    self._lock.notify_all()
            if metrics is not None:
                with suppress(OSError):
                    print('Akane delegation', metrics, flush=True)

    def cancel(self, session, *, request_id="", generation_id="", origin="user_cancel"):
        with self._lock:
            active = self._active
            if (active is None or active.session != session
                    or request_id and request_id != active.request_id
                    or generation_id and generation_id != str(active.token.turn_id)
                    or self._host.accepted_turn == active.turn_id):
                return False
            self._owner.cancel(active.token.turn_id, origin=origin)
            self._host.cancel_turn(active.token.turn_id)
            return True

    def begin_shutdown(self):
        with self._lock:
            if self._closing:
                return
            self._closing = True
            if self._active is not None:
                self.cancel(self._active.session, origin="shutdown")
            for runtime in self._runtimes():
                runtime.scheduler.request_shutdown()

    def reset(self, session="popup", *, forget=False):
        with self._lock:
            self._idle()
            store = self._ensure_store()
            current = self._session(session)
            active = self._host is not None and current.conversation_id == self._host.session.conversation_id
            if active:
                self._host.unattach()  # Also refuses to discard a pending commit.
                self._host.automatic.cancel()
            store.delete_session(current.conversation_id, expected_revision=current.revision)
            replacement = store.create_session(current.conversation_id)
            if active:
                self._host.switch_session(replacement.conversation_id)
            if forget:
                self.edit_memory(clear=True)

    def retry_commit(self, session):
        with self._lock:
            self._idle()
            self._select(session)
            self._host.retry_commit()
            return self.snapshot(session)

    def optional_context_status(self, session):
        with self._lock:
            if self._active is None and self._host and self._host.session.conversation_id == self._session_uuid(session):
                return self._host.automatic_status()
            return None

    def submit_job(self, session, task, source_turn_id=None):
        with self._lock:
            self._idle()
            current = self._select(session)
            context = ()
            if source_turn_id is not None:
                source = next((entry for entry in self._store.transcript(current.conversation_id)
                               if entry.turn_id == source_turn_id and entry.role == "user"), None)
                if source is None:
                    raise KeyError("Original user source was not found.")
                context = (ContextRecord(source.text, (source.source,)),)
            try:
                job = self._host.submit(task, source_context=context)
            except InferenceDraining as error:
                raise ApplicationBusy(str(error)) from error
            return self._job_data(job)

    @staticmethod
    def _job_data(job):
        return {"job_id": str(job.job_id), "status": job.status.value,
                "source_turn_id": job.request.source_turn_id, "delivered_turn_id": job.delivered_turn_id}

    def job(self, session, job_id, action=None):
        with self._lock:
            self._available()
            if self._host is None:
                raise ApplicationBusy("Akane is not ready.")
            job = self._host.status(job_id)
            current = self._ensure_store().get_session(self._session_uuid(session))
            if (job.request.snapshot is None or job.request.automatic
                    or job.conversation_id != current.conversation_id
                    or job.request.session_created_at != current.created_at):
                raise JobNotFound("No matching manual job in this session.")
            if action is not None:
                self._idle()
                self._select(session)
                if action == "attach":
                    self._host.attach(job_id)
                else:
                    self._host.cancel(job_id)
                job = self._host.status(job_id)
            result = self._l2.scheduler.get_result(_PROFILE_ID, job_id)
            return {**self._job_data(job), "invalidated": job.invalidated,
                    "error": result.error if result else None,
                    "findings": result.payload.findings if result and result.payload and not job.invalidated else ()}

    def unattach(self, session):
        with self._lock:
            self._idle()
            self._select(session)
            self._host.unattach()

    def memories(self):
        with self._lock:
            self._available()
            return [
                {"id": str(record.memory_id), "text": record.content, "status": record.status}
                for record in JSONMemoryStore(self._ensure_store()).list_active().records
            ]

    def edit_memory(self, memory_id=None, proposal=None, delete=False, clear=False):
        with self._lock:
            self._available()
            store = JSONMemoryStore(self._ensure_store())

            def approval(identifier, action):
                return MemoryApproval(_PROFILE_ID, identifier, action, datetime.now(timezone.utc), True)

            if clear or delete:
                snapshot = store.list_active()
                ids = [row.memory_id for row in snapshot.records] if clear else [UUID(str(memory_id))]
                revision = snapshot.revision
                for identifier in ids:
                    revision = store.delete(identifier, approval=approval(identifier, "delete"),
                                            expected_revision=revision)
            elif proposal is not None:
                if proposal.get("confirmed") is not True:
                    raise PermissionError("Explicit memory approval is required.")
                candidate = store.propose(
                    self._session_uuid(proposal.get("session_id") or "popup"),
                    UUID(str(proposal["source_turn_id"])), proposal.get("text", ""),
                )
                revision = store.list_active().revision
                if memory_id is None:
                    store.approve(candidate, approval=approval(candidate.memory_id, "admit"),
                                  expected_revision=revision)
                else:
                    store.supersede(UUID(memory_id), candidate,
                                    approval=approval(candidate.memory_id, "supersede"), expected_revision=revision)
            return self.memories()

    def close(self):
        self.begin_shutdown()
        with self._lock, ExitStack() as cleanup:
            self._lock.wait_for(lambda: self._active is None)
            if self._server_lock is not None:
                cleanup.callback(os.close, self._server_lock)
                self._server_lock = None
            cleanup.callback(self._close_store)
            runtimes = self._runtimes()
            for runtime in runtimes:
                if not isinstance(runtime.model, LazyL2):
                    cleanup.callback(runtime.model.close)
            for runtime in runtimes:
                cleanup.callback(runtime.scheduler.shutdown)
            if self._host is not None:
                cleanup.callback(self._host.close)
            self._host = self._l1 = self._l2 = None

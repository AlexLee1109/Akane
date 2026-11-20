"""Explicit, process-local L1/L2 handoff; L3 alone persists exchanges.

The terminal calls this coordinator serially. The scheduler alone owns concurrent
inference; no host command parses the meaning of conversational text.
"""

from collections.abc import Iterator
from contextlib import closing, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic
from uuid import UUID, uuid4

from .automatic import AutomaticContextManager
from .automatic_contracts import AutomaticContextPolicy
from .background.contracts import BackgroundJob, BackgroundResult, TaskState
from .background.scheduler import InferenceScheduler, JobConflict, JobNotFound, QueueFull, SchedulerClosed
from .background.worker import BackgroundIntelligence
from .contracts import (
    AttachedResult, ContextRecord, ContextSnapshot, ConversationContext, ConversationSession,
    DelegationRequest, HistoryEntry, InteractionRequest,
)
from .continuity.store import ContinuityConflict, ContinuityStore, SessionNotFound
from .continuity.context import ContextAssembler, ContextAssembly
from .continuity.json_store import JSONContinuityStore
from .context_contracts import ContextSelectionStatus
from .conversation.cancellation import CancellationToken, TurnCancellationOwner
from .conversation.events import InteractionEvent, ReplyCancelled, ReplyFailed
from .conversation.model import ChatMessage, DialogueModel
from .conversation.cancellation import GenerationCancelled
from .delegation import AutoDelegationLimits, AuthorizedTask
from .provenance import SourceAttribution
from .conversation.interaction import InteractionModel


class CommitPending(RuntimeError):
    """Retry the exact accepted exchange, without generating again."""


class InferenceDraining(RuntimeError):
    """Timed-out inference still owns the shared model."""


class _DelegationBudgetExceeded(ValueError):
    pass


@dataclass(slots=True)
class _Selection:
    job_id: UUID
    pin_id: UUID
    armed: bool = True


@dataclass(frozen=True, slots=True)
class _Exchange:
    user: HistoryEntry
    reply: HistoryEntry
    revision: int | None
    attachment: _Selection | None
    automatic: bool = False


class ConversationHost:
    def __init__(
        self, profile_id: UUID, conversation_id: UUID, interaction: InteractionModel,
        scheduler: InferenceScheduler, executor: BackgroundIntelligence, *,
        store: ContinuityStore | None = None, max_result_characters: int = 2048,
        max_selected_memories: int = 4, automatic_context: AutomaticContextPolicy | None = None,
        automatic_delegation: bool = False, delegation_limits: AutoDelegationLimits = AutoDelegationLimits(),
        background_scheduler: InferenceScheduler | None = None,
        maintenance_model: DialogueModel | None = None,
    ) -> None:
        if store is not None and store.profile_id != profile_id:
            raise ValueError("Store belongs to another profile.")
        if max_result_characters < 1:
            raise ValueError("Result character allowance must be positive.")
        self.profile_id = profile_id
        self.interaction = interaction
        self.scheduler = scheduler
        self.background_scheduler = scheduler if background_scheduler is None else background_scheduler
        self.executor = executor
        self.store = store
        self.session = store.get_session(conversation_id) if store else ConversationSession(
            conversation_id, profile_id, datetime.now(timezone.utc),
        )
        if type(max_selected_memories) is not int or max_selected_memories < 1:
            raise ValueError("Selected memory bound must be a positive integer.")
        self._max_memories = max_selected_memories
        self._context_assembler: ContextAssembler | None = None
        self._use_summary = False
        self._memory_ids: tuple[UUID, ...] = ()
        self.last_context_assembly: ContextAssembly | None = None
        self._automatic_policy = automatic_context
        if automatic_context is not None and not isinstance(store, JSONContinuityStore):
            raise ValueError("Automatic context requires JSON persistence.")
        if automatic_context is not None and self.background_scheduler is not scheduler and maintenance_model is None:
            raise ValueError("Independent maintenance requires a model for the background scheduler.")
        self.automatic = (AutomaticContextManager(
            store, interaction._model if maintenance_model is None else maintenance_model,
            self.background_scheduler, automatic_context)
                          if automatic_context is not None else None)
        self._summary_override = None
        self._excluded_memory_ids = ()
        self._removed_memory_ids = set()
        self._characters = max_result_characters
        self._selection: _Selection | None = None
        self._pending: _Exchange | None = None
        self._busy = False
        self.last_request: InteractionRequest | None = None
        self.last_foreground_wait_seconds: float | None = None
        if type(automatic_delegation) is not bool:
            raise ValueError('Automatic delegation enablement must be Boolean.')
        self._delegate_enabled = automatic_delegation
        self._delegate_limits = delegation_limits
        self._auto_pin = None
        self._auto_job_id = None
        self._draining_job_id = None
        self._authorized_task = None
        self._delegation_audit = {'enabled':automatic_delegation, 'state':'idle', 'passes':[]}
        if self.automatic is not None:
            self.automatic.activate(self.session)


    def _idle(self) -> None:
        if self._busy:
            raise RuntimeError("A conversation turn is active.")
        if self._pending is not None:
            raise CommitPending("Exchange commit is pending; use :retry commit before continuing.")

    def _current(self) -> ConversationSession:
        if self.store is None:
            return self.session
        current = self.store.get_session(self.session.conversation_id)
        if current.created_at != self.session.created_at:
            raise JobConflict("Session was deleted and replaced.")
        return current

    def _snapshot(self, token: CancellationToken):
        self._current()
        return self.store.snapshot(
            self.session.conversation_id, max_recent_turns=self.interaction.recent.max_turns,
            cancellation=token,
        ) if self.store else None

    def _assembler(self) -> ContextAssembler:
        if not isinstance(self.store, JSONContinuityStore):
            raise ValueError("Explicit L3 context selection requires JSON persistence.")
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler(self.store, max_memories=self._automatic_policy.max_memories
                                                        if self._automatic_policy else self._max_memories)
        return self._context_assembler

    def _selected_context(self, token: CancellationToken, *, use_summary: bool,
                          memory_ids: tuple[UUID, ...]) -> ContextAssembly:
        self._current()
        return self._assembler().assemble(
            self.session, use_summary=use_summary, memory_ids=memory_ids,
            max_recent_turns=self.interaction.recent.max_turns, cancellation=token,
        )

    def select_summary(self, enabled: bool = True) -> None:
        self._idle()
        if type(enabled) is not bool:
            raise ValueError("Summary selection must be explicitly enabled or disabled.")
        if enabled:
            self._selected_context(TurnCancellationOwner().begin(uuid4()),
                                   use_summary=True, memory_ids=self._memory_ids)
        self._summary_override = enabled
        self._use_summary = enabled
        self.last_context_assembly = None

    def select_memories(self, memory_ids: tuple[UUID, ...]) -> None:
        self._idle()
        identifiers = self._assembler().memory_ids(memory_ids)
        self._selected_context(TurnCancellationOwner().begin(uuid4()),
                               use_summary=self._use_summary, memory_ids=identifiers)
        self._memory_ids = identifiers
        self._removed_memory_ids.difference_update(identifiers)
        self.last_context_assembly = None

    def remove_memory(self, memory_id: UUID) -> None:
        self._idle()
        if not isinstance(memory_id, UUID):
            raise ValueError("Memory identifiers must be UUIDs.")
        self._memory_ids = tuple(value for value in self._memory_ids if value != memory_id)
        self._removed_memory_ids.add(memory_id)
        self.last_context_assembly = None

    def context_status(self) -> ContextSelectionStatus:
        return ContextSelectionStatus(
            self.profile_id, self.session.conversation_id, self._use_summary, self._memory_ids,
            self.last_context_assembly.summary_state if self.last_context_assembly else None,
            self.interaction.last_context_budget if self.last_context_assembly else None,
        )

    def submit(self, task: str, *, source_context: tuple[ContextRecord, ...] = ()) -> BackgroundJob:
        self._idle()
        self.require_inference_ready()
        token = TurnCancellationOwner().begin(uuid4())
        snapshot = self._snapshot(token)
        recent = self.interaction.recent.context(self.session.conversation_id, snapshot)
        source = next((entry.turn_id for entry in reversed(recent) if entry.role == "user"), None)
        request = DelegationRequest(
            uuid4(), self.session.conversation_id, source or token.turn_id, task,
            snapshot, source_context=source_context, session_created_at=self.session.created_at,
            source_turn_id=source,
        )
        return self.background_scheduler.submit_background(self.profile_id, request, self.executor)

    def status(self, job_id: UUID) -> BackgroundJob:
        return self.background_scheduler.get_job(self.profile_id, job_id)

    def show(self, job_id: UUID) -> BackgroundResult:
        result = self.background_scheduler.get_result(self.profile_id, job_id)
        if result is None:
            raise JobConflict("Job has not completed.")
        return result

    def cancel(self, job_id: UUID) -> None:
        self._idle()
        if self._selection is not None and self._selection.job_id == job_id:
            self.unattach()
        if not self.background_scheduler.cancel_background(self.profile_id, job_id):
            self.background_scheduler.invalidate_result(self.profile_id, job_id)

    def _validate(self, job: BackgroundJob, result: BackgroundResult) -> None:
        current = self._current()
        request, payload = job.request, result.payload
        if request.delivery_turn_id is not None and (
                not self._busy or self.last_request is None
                or request.delivery_turn_id != self.last_request.turn_id
                or request.snapshot is not None and request.snapshot.revision != current.revision):
            raise JobConflict('Automatic result is stale or not bound to this active user turn.')
        if (job.profile_id != self.profile_id or result.profile_id != self.profile_id
                or request.conversation_id != current.conversation_id
                or result.conversation_id != current.conversation_id
                or request.session_created_at != current.created_at):
            raise JobConflict("Result belongs to another profile, session, or session incarnation.")
        if (job.invalidated or job.delivered_turn_id is not None
                or job.status != TaskState.COMPLETED or result.status != TaskState.COMPLETED
                or payload is None or payload.status != "completed" or not payload.findings
                or not any(record.text.strip() for record in payload.findings)):
            raise JobConflict("Result is incomplete, cancelled, invalidated, empty, or consumed.")
        if (result.job_id != job.job_id or (payload.request_id, payload.conversation_id, payload.turn_id)
                != (request.request_id, request.conversation_id, request.turn_id)):
            raise JobConflict("Result does not match its originating request.")
        if request.snapshot is not None:
            if (request.snapshot.profile_id != self.profile_id
                    or request.snapshot.revision > current.revision
                    or (request.source_turn_id is not None and request.source_turn_id not in
                        {entry.turn_id for entry in request.snapshot.recent})):
                raise JobConflict("Invalid source context anchor.")
        if sum(len(record.text) for record in payload.findings) > self._characters:
            raise JobConflict("Result exceeds the character allowance.")
        if any(not any(source.owner == "l2" and source.source_id for source in record.sources)
               for record in payload.findings):
            raise JobConflict("Generated findings require L2 provenance.")

    def attach(self, job_id: UUID) -> BackgroundJob:
        self._idle()
        if self._selection is not None and self._selection.job_id != job_id:
            raise JobConflict("Remove the pending attachment before selecting another.")
        selection = self._selection or _Selection(job_id, uuid4())
        job, result = self.background_scheduler.pin_result(self.profile_id, job_id, selection.pin_id)
        try:
            self._validate(job, result)
        except BaseException:
            self.background_scheduler.unpin_result(self.profile_id, job_id, selection.pin_id)
            self._selection = None
            raise
        selection.armed = True
        self._selection = selection
        return job

    def unattach(self) -> None:
        self._idle()
        if self._selection is not None:
            self.background_scheduler.unpin_result(self.profile_id, self._selection.job_id, self._selection.pin_id)
            self._selection = None

    def switch_session(self, conversation_id: UUID | None = None) -> ConversationSession:
        self._idle()
        if self.store:
            session = self.store.get_session(conversation_id) if conversation_id else self.store.create_session()
        else:
            if conversation_id is not None:
                raise ValueError("Resuming a session requires JSON persistence.")
            session = ConversationSession(uuid4(), self.profile_id, datetime.now(timezone.utc))
        self.unattach()
        if self.automatic is not None:
            self.automatic.activate(session)
        self._summary_override = None
        self._excluded_memory_ids = ()
        self._removed_memory_ids.clear()
        self.session = session
        self._use_summary = False
        self._memory_ids = ()
        self.last_context_assembly = None
        self._auto_job_id = self._auto_pin = self._authorized_task = None
        self._delegation_audit = {'enabled':self._delegate_enabled,'state':'idle','passes':[]}
        return session

    def set_automatic_delegation(self, enabled: bool) -> None:
        self._idle()
        if type(enabled) is not bool:
            raise ValueError('Automatic delegation enablement must be Boolean.')
        self._delegate_enabled = enabled
        self._delegation_audit = {'enabled':enabled, 'state':'idle', 'passes':[]}

    def delegation_status(self):
        # Copy nested per-pass diagnostics so callers cannot alter live authority.
        from copy import deepcopy
        result = deepcopy(self._delegation_audit)
        result['enabled'] = self._delegate_enabled
        if self._auto_job_id is not None:
            job = self.background_scheduler.get_job(self.profile_id, self._auto_job_id)
            result['job'] = {'job_id':job.job_id,'status':job.status.value,
                             'acquired_monotonic':job.acquired_monotonic,
                             'cancellation_requested_monotonic':job.cancellation_requested_monotonic,
                             'delivered_turn_id':job.delivered_turn_id}
        return result

    def delegation_metrics(self):
        """Numeric timings and status only, suitable for production diagnostics."""
        audit = self._delegation_audit
        result = audit.get('result')
        metrics = result.generation if result else None
        return {
            'state': audit['state'], 'handoff': audit.get('handoff'),
            'first_visible_seconds': audit.get('first_visible_total_seconds'),
            'total_seconds': audit.get('total_seconds'),
            'background_pending': audit.get('background_pending'),
            'background_wait_seconds': audit.get('background_wait_seconds'),
            'worker_status': result.status.value if result else audit.get('worker_status'),
            'worker_prompt_tokens': metrics.prompt_tokens if metrics else None,
            'worker_sampled_tokens': metrics.sampled_tokens if metrics else None,
            'worker_seconds': metrics.finished_monotonic - metrics.started_monotonic if metrics else None,
            'worker_lease_released_at': result.released_monotonic if result else None,
            'passes': [
                {'mode': item['mode'], 'prompt_tokens': item['prompt_tokens'],
                 'first_visible_seconds': item['first_visible_seconds'],
                 'foreground_wait_seconds': item['foreground_wait_seconds'],
                 'sampled_tokens': getattr(item['metrics'], 'sampled_tokens', None),
                 'model_seconds': getattr(item['metrics'], 'total_seconds', None)}
                for item in audit['passes']
            ],
        }

    def require_inference_ready(self) -> None:
        if self._draining_job_id is None:
            return
        try:
            result = self.background_scheduler.get_result(self.profile_id, self._draining_job_id)
        except JobNotFound:
            # Only terminal entries can be evicted.
            self._draining_job_id = None
            return
        if result is None:
            raise InferenceDraining('Analysis timed out and the model is still stopping. Try again after it stops.')
        if result.status == TaskState.COMPLETED:
            self.background_scheduler.invalidate_result(self.profile_id, result.job_id)
        self._draining_job_id = None

    def cancel_turn(self, turn_id: UUID) -> bool:
        """Wake scheduler waits after the interaction owner signals its token."""
        changed = self.scheduler.cancel_foreground(self.profile_id, turn_id)
        if (self._busy and self.last_request is not None and self.last_request.turn_id == turn_id
                and self._auto_job_id is not None):
            changed = self.background_scheduler.cancel_background(self.profile_id, self._auto_job_id) or changed
        return changed

    def _refresh_request(self, request, token, *, skip_memory=False):
        attached = request.context.attached_results
        if self.automatic is not None:
            ids, self._excluded_memory_ids = ((), ()) if skip_memory else self._assembler().automatic_ids(
                self._memory_ids, excluded=tuple(self._removed_memory_ids))
            self.last_context_assembly = self._assembler().assemble(
                self.session, use_summary=self._summary_override is not False, memory_ids=ids,
                max_recent_turns=self.interaction.recent.max_turns, cancellation=token, automatic=True,
                memory_token_budget=self._automatic_policy.memory_tokens,
                protected_exchanges=self._automatic_policy.protected_exchanges,
                segment_mode=self._automatic_policy.segment_exchanges is not None,
                max_summary_segments=self._automatic_policy.max_summary_segments,
                summary_token_budget=self._automatic_policy.summary_tokens,
            )
            context = replace(self.last_context_assembly.context, attached_results=attached,
                                      delegation_failure=request.context.delegation_failure)
            snapshot = context.snapshot
            request = replace(request, context=context)
        elif self._use_summary or self._memory_ids:
            self.last_context_assembly = self._selected_context(
                token, use_summary=self._use_summary, memory_ids=() if skip_memory else self._memory_ids,
            )
            context = replace(self.last_context_assembly.context, attached_results=attached,
                                      delegation_failure=request.context.delegation_failure)
            snapshot = context.snapshot
            request = replace(request, context=context)
        else:
            # A foreground lease may have waited behind native cleanup. Ordinary
            # and manual-attachment turns need the same fresh source boundary.
            request = replace(request, context=replace(request.context, snapshot=self._snapshot(token)))
        return request

    def _check_authorized(self, authorized, cancellation):
        cancellation.raise_if_cancelled()
        current = self._current()
        if (authorized.profile_id != self.profile_id or authorized.session_id != current.conversation_id
                or authorized.session_created_at != current.created_at
                or authorized.turn_id != cancellation.turn_id
                or authorized.revision is not None and authorized.revision != current.revision):
            raise JobConflict('Automatic task source/session/revision is stale.')

    def _authorize_task(self, decision, request, cancellation):
        cancellation.raise_if_cancelled()
        current = self._current()
        snapshot = request.context.snapshot
        if (request.turn_id != cancellation.turn_id or request.conversation_id != current.conversation_id
                or snapshot is not None and (snapshot.profile_id != self.profile_id or snapshot.revision != current.revision)):
            raise JobConflict('Automatic decision does not own the active interaction context.')
        limits = self._delegate_limits
        if decision.scope != 'provided-information' or decision.context_refs != ('u0',):
            raise ValueError('Only current supplied-information analysis is authorized.')
        intent_only = getattr(self.interaction, '_intent_only', False)
        if intent_only != (decision.task is None):
            raise ValueError('Task specification does not match the selected protocol.')
        # The host alone supplies task text in Step5A5. Do not reword, trim, or
        # replace it with a generated placeholder. The native worker receives
        # this same exact source again as attributed source_context.
        task = request.text if intent_only else decision.task
        if not isinstance(task, str) or not task.strip() or not 0 < len(task) <= (limits.source_characters if intent_only else 1024) or len(request.text)>limits.source_characters:
            raise _DelegationBudgetExceeded('Automatic task/source character bound exceeded.')
        task_tokens = len(self.interaction._model.prepare((ChatMessage('user',task),), max_tokens=1).tokens)
        source_tokens = (task_tokens if intent_only else
                         len(self.interaction._model.prepare((ChatMessage('user',request.text),), max_tokens=1).tokens))
        if task_tokens>limits.task_tokens or source_tokens>limits.source_tokens:
            raise _DelegationBudgetExceeded('Automatic task/source exact token bound exceeded.')
        worker_limits = getattr(self.executor, 'automatic_limits', None) or getattr(self.executor, 'limits', None)
        if worker_limits is not None and (worker_limits.max_output_tokens>limits.output_tokens
                or worker_limits.max_prompt_tokens>limits.prompt_tokens
                or worker_limits.max_result_chars>limits.result_characters or worker_limits.max_seconds>limits.seconds):
            raise ValueError('Injected worker exceeds automatic task resource authority.')
        # L2 needs the source revision for delivery checks. L1 owns the snapshot contents.
        anchor = (ContextSnapshot(snapshot.conversation_id, snapshot.revision, profile_id=snapshot.profile_id)
                  if snapshot is not None else None)
        delegated = DelegationRequest(request.turn_id, current.conversation_id, request.turn_id, task, anchor,
                        source_context=(ContextRecord(request.text,(request.source,)),),
                        session_created_at=current.created_at, delivery_turn_id=request.turn_id)
        authorized = AuthorizedTask(self.profile_id,current.conversation_id,current.created_at,request.turn_id,
                                    snapshot.revision if snapshot else None,delegated,task_tokens,source_tokens)
        self._check_authorized(authorized,cancellation)
        return authorized

    def _foreground_events(self, request, cancellation, *, automatic=False, skip_memory=False):
        waiting = monotonic()
        # L1 still owns only its own lease. Defer L2 dispatch while L1 is using
        # Metal until concurrent native execution has a separate acceptance run.
        with (self.scheduler.foreground(self.profile_id, request.turn_id, cancellation=cancellation) as token,
              self.background_scheduler.pause_background(cancellation=token)
              if self.background_scheduler is not self.scheduler else nullcontext()):
            self.last_foreground_wait_seconds = monotonic()-waiting
            request = (self._refresh_request(request, token, skip_memory=True) if skip_memory
                       else self._refresh_request(request, token))
            if (request.control_mode == 'decision' and self.interaction._intent_only
                    and self._delegate_limits.min_task_tokens):
                source = self.interaction._model.prepare((ChatMessage('user', request.text),), max_tokens=1)
                if len(source.tokens) < self._delegate_limits.min_task_tokens:
                    request = replace(request, control_mode='plain')
            if automatic and self._authorized_task is not None:
                # Final ownership may have waited behind another caller. A fresh
                # snapshot must not legitimize findings bound to an older revision.
                self._check_authorized(self._authorized_task,token)
            elif automatic and request.context.delegation_failure is not None:
                failed_snapshot = self.last_request.context.snapshot
                if failed_snapshot is not None and self._current().revision != failed_snapshot.revision:
                    raise JobConflict('Automatic decision was superseded before the reply.')
            self.last_request = request
            for item in request.context.attached_results:
                self._validate(self.background_scheduler.get_job(self.profile_id, item.job_id), self.show(item.job_id))
            output = self.interaction.stream(request,cancellation=token)
            try:
                for event in output:
                    if event.kind == 'reply_delegated':
                        self._authorized_task = self._authorize_task(event.decision,request,token)
                    elif automatic and self._delegate_limits.wait_seconds is not None and event.kind == 'reply_failed':
                        self._delegation_audit['error'] = event.message
                        event = ReplyFailed(request.turn_id, 'The reply could not be completed.')
                    yield event
            finally:
                output.close()
                if automatic:
                    self._delegation_audit['passes'].append({
                        'mode':request.control_mode,'prompt_tokens':len(self.interaction.last_prompt.tokens) if self.interaction.last_prompt else None,
                        'wire':self.interaction.last_control_raw,'decision':self.interaction.last_decision,
                        'metrics':self.interaction.last_generation_metrics,
                        'first_visible_seconds':self.interaction.last_visible_seconds,
                        'foreground_wait_seconds':self.last_foreground_wait_seconds,
                    })

    def _wait_for_automatic_result(self, job, cancellation, started):
        limits = self._delegate_limits
        if limits.wait_seconds is None:
            return self.background_scheduler.wait_result(self.profile_id, job.job_id)
        try:
            wait = max(0.0, limits.wait_seconds - (monotonic() - started))
            return self.background_scheduler.wait_result(self.profile_id, job.job_id, timeout=wait)
        except TimeoutError:
            self._delegation_audit['worker_status'] = 'timed_out'
            self.background_scheduler.cancel_background(self.profile_id, job.job_id)
            try:
                late = self.background_scheduler.wait_result(self.profile_id, job.job_id, timeout=limits.drain_seconds)
            except TimeoutError:
                self._draining_job_id = job.job_id
                cancellation.raise_if_cancelled()
                raise InferenceDraining('Analysis timed out and the model is still stopping.') from None
            if late.status == TaskState.COMPLETED:
                self.background_scheduler.invalidate_result(self.profile_id, job.job_id)
            raise

    def _failure_data(self, turn_id, category):
        return ContextRecord(
            f'Analysis unavailable ({category}). No completed findings.',
            (SourceAttribution('l1', f'host-task-failure:{turn_id}'),),
        )

    def _final_events(self, cancellation, *, attached=(), failure=None, skip_memory=False):
        if failure is not None:
            self._delegation_audit['failure'] = failure
        self._delegation_audit['handoff'] = 'completed-findings' if attached else 'failure-data-no-findings'
        if failure is not None and self.interaction._intent_only:
            cancellation.raise_if_cancelled()
            self._delegation_audit['state'] = 'failed'
            # Report an application error through the normal terminal event.
            # The transcript retains the failed turn without assistant dialogue.
            yield ReplyFailed(cancellation.turn_id, 'Automatic analysis could not be completed. Try a narrower request.')
            return
        self._delegation_audit['state'] = 'final'
        final_context = replace(self.last_request.context, attached_results=attached, delegation_failure=failure)
        final_request = replace(self.last_request, context=final_context, control_mode='final')
        with closing(self._foreground_events(final_request, cancellation, automatic=True, skip_memory=skip_memory)) as output:
            for event in output:
                if event.kind != 'reply_started':
                    yield event
        self._delegation_audit['state'] = self.interaction.last_control_status

    def _turn_events(self, request, cancellation, *, skip_memory=False):
        if not self._delegate_enabled:
            yield from self._foreground_events(request,cancellation,skip_memory=skip_memory)
            return
        self._delegation_audit['state'] = 'deciding'
        delegated = False
        initial = replace(request,control_mode='final' if request.context.attached_results else 'decision')
        try:
            with closing(self._foreground_events(initial,cancellation,automatic=True,skip_memory=skip_memory)) as output:
                for event in output:
                    if event.kind == 'reply_delegated':
                        delegated = True
                    else:
                        yield event
        except _DelegationBudgetExceeded:
            if self._delegate_limits.wait_seconds is None:
                raise
            cancellation.raise_if_cancelled()
            yield from self._final_events(cancellation, failure=self._failure_data(request.turn_id, 'input limit'),
                                          skip_memory=skip_memory)
            return
        if not delegated:
            self._delegation_audit['state'] = self.interaction.last_control_status
            return
        # The complete L1 stream and foreground context manager have unwound.
        authorized = self._authorized_task
        self._check_authorized(authorized,cancellation)
        self._delegation_audit.update(state='queued', authorized=authorized,
            task_origin=('host-grounded u0' if getattr(self.interaction, '_intent_only', False)
                         else 'l1-model specification'))
        failure = None
        attached = ()
        try:
            started = monotonic()
            job = self.background_scheduler.submit_background(self.profile_id,authorized.request,self.executor,
                                                   budget_seconds=self._delegate_limits.seconds,cancellation=cancellation)
            self._auto_job_id = job.job_id
            if self._delegate_limits.wait_seconds is not None:
                self._delegation_audit['background_pending'] = self.background_scheduler.pending_background_count()
            result = self._wait_for_automatic_result(job, cancellation, started)
            job = self.background_scheduler.get_job(self.profile_id, job.job_id)
            if job.acquired_monotonic is not None:
                self._delegation_audit['background_wait_seconds'] = job.acquired_monotonic - started
            self._delegation_audit.update(result=result,state='handoff')
            self._check_authorized(authorized,cancellation)
            if result.status != TaskState.COMPLETED:
                failure = self._failure_data(request.turn_id, result.status.value.lower())
            else:
                pin = _Selection(job.job_id,uuid4(),False)
                job,result = self.background_scheduler.pin_result(self.profile_id,job.job_id,pin.pin_id)
                try:
                    self._validate(job,result)
                    if getattr(self.executor, 'automatic_limits', None) is not None:
                        original = authorized.request.source_context[0].sources[0]
                        expected = (SourceAttribution('l2', f'model-generated:{job.request_id}'), original)
                        if (job.request != authorized.request or len(result.payload.findings) != 1
                                or result.payload.findings[0].sources != expected):
                            raise ValueError('Automatic findings do not match the authorized source.')
                    if sum(len(record.text) for record in result.payload.findings)>self._delegate_limits.result_characters:
                        raise ValueError('Automatic findings exceed their bound.')
                except BaseException as error:
                    self.background_scheduler.unpin_result(self.profile_id,job.job_id,pin.pin_id)
                    self.background_scheduler.invalidate_result(self.profile_id,job.job_id)
                    if isinstance(error, GenerationCancelled):
                        raise
                    self._check_authorized(authorized,cancellation)
                    raise ValueError(f'Invalid background completion: {error}') from error
                self._auto_pin = self._selection = pin
                attached = (AttachedResult(job.job_id,self.profile_id,job.request,result.payload,
                                          job.created_at,result.completed_at,request.turn_id),)
        except (GenerationCancelled, JobConflict, InferenceDraining, SchedulerClosed):
            # Source/session conflict must not turn into a reply from stale data.
            raise
        except QueueFull:
            failure = self._failure_data(request.turn_id, 'queue full')
        except TimeoutError:
            failure = self._failure_data(request.turn_id, 'timed out')
        except Exception:
            failure = self._failure_data(request.turn_id, 'invalid result or worker failure')
        cancellation.raise_if_cancelled()
        self._check_authorized(authorized,cancellation)
        yield from self._final_events(cancellation, attached=attached, failure=failure, skip_memory=skip_memory)

    def stream(self, text: str, *, cancellation: CancellationToken, skip_memory: bool = False) -> Iterator[InteractionEvent]:
        self._idle()
        self.require_inference_ready()
        self._busy = True
        self.last_request = None
        self.last_context_assembly = None
        self.last_foreground_wait_seconds = None
        self._auto_pin = self._authorized_task = None
        self._auto_job_id = None
        self._delegation_audit = {'enabled':self._delegate_enabled,'state':'starting','turn_id':cancellation.turn_id,'passes':[]}
        started = False
        pieces = []
        outcome = 'failed'
        selection = None
        snapshot = None
        user_time = datetime.now(timezone.utc)
        interaction_started = monotonic()
        try:
            snapshot = self._snapshot(cancellation)
            attached = ()
            if self._selection is not None and self._selection.armed:
                selection = self._selection
                job,result = self.background_scheduler.pin_result(self.profile_id,selection.job_id,selection.pin_id)
                self._validate(job,result)
                attached = (AttachedResult(job.job_id,self.profile_id,job.request,result.payload,
                                          job.created_at,result.completed_at,cancellation.turn_id),)
                selection.armed = False
            request = InteractionRequest(self.session.conversation_id,cancellation.turn_id,text,
                                         ConversationContext(snapshot,attached_results=attached))
            output = self._turn_events(request,cancellation,skip_memory=skip_memory)
            try:
                for event in output:
                    if self.last_request is not None:
                        snapshot = self.last_request.context.snapshot
                    if event.kind == 'reply_started':started = True
                    elif event.kind == 'reply_text_delta':
                        if 'first_visible_total_seconds' not in self._delegation_audit:
                            self._delegation_audit['first_visible_total_seconds'] = monotonic()-interaction_started
                        pieces.append(event.text)
                    elif event.kind == 'reply_completed':outcome = 'completed'
                    elif event.kind == 'reply_cancelled':outcome = 'cancelled'
                    yield event
            finally:
                output.close()
        except GenerationCancelled:
            outcome = 'cancelled'
            self._delegation_audit['state'] = 'cancelled'
            yield ReplyCancelled(cancellation.turn_id)
        except GeneratorExit:
            outcome = 'cancelled'
            self._delegation_audit['state'] = 'cancelled'
            raise
        except Exception as error:
            self._delegation_audit.update(state='failed',error=f'{type(error).__name__}: {error}')
            if not self._delegate_enabled:
                raise  # Preserve previous explicit-host exception contract.
            message = str(error) if self._delegate_limits.wait_seconds is None or isinstance(error, InferenceDraining) else (
                'The conversation changed before this reply completed.' if isinstance(error, (JobConflict, SessionNotFound))
                else 'The reply could not be completed.'
            )
            yield ReplyFailed(cancellation.turn_id,message)
        finally:
            self._busy = False
            if started:
                self._pending = _Exchange(
                    HistoryEntry(cancellation.turn_id,'user',text,created_at=user_time),
                    HistoryEntry(cancellation.turn_id,'assistant',''.join(pieces),outcome,created_at=datetime.now(timezone.utc)),
                    snapshot.revision if snapshot else None,self._auto_pin or selection,self._auto_pin is not None)
                try:
                    self.retry_commit()
                finally:
                    self._delegation_audit['total_seconds'] = monotonic()-interaction_started

    def retry_commit(self) -> None:
        if self._busy:
            raise RuntimeError("A conversation turn is active.")
        exchange = self._pending
        if exchange is None:
            raise CommitPending("No exchange commit is pending.")
        try:
            self._current()
            if self.store:
                self.store.append_turn(
                    self.session.conversation_id, exchange.user, exchange.reply,
                    expected_revision=exchange.revision,
                )
        except Exception as error:
            if (isinstance(error, (ContinuityConflict, SessionNotFound, JobConflict))
                    and exchange.automatic and exchange.attachment is not None):
                selection = exchange.attachment
                self.background_scheduler.unpin_result(self.profile_id, selection.job_id, selection.pin_id)
                self.background_scheduler.invalidate_result(self.profile_id, selection.job_id)
                self._selection = None
                self._pending = replace(exchange, attachment=None)
            raise CommitPending(f"Commit failed: {error}; use :retry commit (no regeneration).") from error
        if exchange.attachment is not None and exchange.reply.outcome == "completed":
            selection = exchange.attachment
            self.background_scheduler.consume_result(
                self.profile_id, selection.job_id, pin_id=selection.pin_id, turn_id=exchange.user.turn_id,
            )
            self.background_scheduler.unpin_result(self.profile_id, selection.job_id, selection.pin_id)
            self._selection = None
        if exchange.attachment is not None and exchange.reply.outcome != 'completed' and exchange.automatic:
            selection = exchange.attachment
            self.background_scheduler.unpin_result(self.profile_id,selection.job_id,selection.pin_id)
            self.background_scheduler.invalidate_result(self.profile_id,selection.job_id)
            self._selection = None
        self._pending = None
        if self.automatic is not None and exchange.reply.outcome == 'completed':
            self.automatic.observe(self.session.conversation_id, self.interaction.last_automatic_info,
                                   context_size=self.interaction._model.context_size)

    def automatic_status(self):
        if self.automatic is None:
            return {'enabled': False}
        info = self.interaction.last_automatic_info if self.last_request is not None and self.last_request.conversation_id == self.session.conversation_id else None
        assembly = self.last_context_assembly
        included = set(info.included_memories) if info else set()
        excluded = set(info.excluded_memories) if info else set()
        records = assembly.memories if assembly else ()
        ranges = []
        if assembly:
            if assembly.summary:
                ranges.append({'reference':'legacy', 'start':0, 'end':len(assembly.summary.covered_turn_ids)})
            ranges.extend({'reference':f's{index}', 'start':segment.start, 'end':segment.end}
                          for index,segment in enumerate(assembly.segments,1))
            for item in ranges:
                item['visible'] = item['reference'] in info.included_segments if info else None
        return {
            'enabled': True, 'maintenance_enabled': self.automatic.enabled, 'maintenance': self.automatic.status(self.session.conversation_id),
            'periodic': self.automatic.periodic_status(),
            'prompt': info, 'summary_state': assembly.summary_state if assembly else None,
            'covered_turn_ids': assembly.context.l3_projection.covered_turn_ids if assembly and assembly.context.l3_projection else (),
            'stored_segments': len(assembly.segments) if assembly else 0,
            'summary_ranges': tuple(ranges),
            'included_segments': info.included_segments if info else (),
            'omitted_segments': info.omitted_segments if info else (),
            'selected_memory_ids': tuple(record.memory_id for index, record in enumerate(records, 1) if f'm{index}' in included),
            'excluded_memory_ids': (*sorted(self._removed_memory_ids, key=str), *self._excluded_memory_ids, *(record.memory_id for index, record in enumerate(records, 1) if f'm{index}' in excluded)),
        }

    def close(self) -> None:
        if self._busy:
            raise RuntimeError("Cannot close an active conversation.")
        if self.automatic is not None:
            self.automatic.close()
        if self._selection is not None:
            self.background_scheduler.unpin_result(self.profile_id, self._selection.job_id, self._selection.pin_id)
            self._selection = None

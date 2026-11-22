"""Explicit host-side L3 maintenance; L2 executes, L3 alone publishes summaries."""
from dataclasses import dataclass
from uuid import UUID, uuid4

from .background.compaction import CompactionExecutor
from .background.contracts import TaskState
from .background.scheduler import InferenceScheduler
from .background.residency import LazyL2
from .contracts import ConversationSession, DelegationRequest
from .continuity.compaction import JSONSummaryStore
from .continuity.compaction_contracts import CompactionLimits, CompactionPrefix, SummaryCheckpoint
from .continuity.segment_contracts import SegmentCapture, SummarySegment
from .continuity.store import ContinuityConflict
from .conversation.cancellation import CancellationToken, GenerationCancelled
from .conversation.model import DialogueModel


@dataclass(frozen=True, slots=True)
class CompactionOperation:
    job_id: UUID
    prefix: CompactionPrefix | SegmentCapture
    executor: CompactionExecutor
    cancellation: CancellationToken


class Compactor:
    """Use the configured maintenance model and scheduler to compute summaries."""
    def __init__(self, summaries: JSONSummaryStore, model: DialogueModel | LazyL2, scheduler: InferenceScheduler, *,
                 limits: CompactionLimits = CompactionLimits()):
        self.summaries, self.model, self.scheduler, self.limits = summaries, model, scheduler, limits

    def start(self, session_id: UUID, through_turn_id: UUID, *, cancellation: CancellationToken,
              include_diagnostics: bool = False) -> CompactionOperation:
        prefix = self.summaries.capture(session_id, through_turn_id, limits=self.limits, cancellation=cancellation,
                                        include_diagnostics=include_diagnostics)
        return self._schedule(prefix, session_id, through_turn_id, cancellation)

    def start_segment(self, session_id: UUID, *, cancellation: CancellationToken, max_exchanges=4,
                      expected_session: ConversationSession | None = None,
                      expected_start: int | None = None) -> CompactionOperation:
        captured = self.summaries.capture_segment(session_id, max_exchanges=max_exchanges,
                                                  limits=self.limits, cancellation=cancellation)
        if ((expected_session is not None and captured.session != expected_session)
                or (expected_start is not None and captured.start != expected_start)):
            raise ContinuityConflict('Segment source or frontier changed before submission.')
        return self._schedule(captured, session_id, captured.through_turn_id, cancellation)

    def _schedule(self, prefix, session_id, through_turn_id, cancellation):
        # Existing host-authorized delegation is transport for this L3-owned
        # maintenance task, not a conversational L1 decision or result attachment.
        request = DelegationRequest(uuid4(), session_id, cancellation.turn_id, 'L3 context compaction',
                                    session_created_at=prefix.session.created_at, source_turn_id=through_turn_id)
        executor = CompactionExecutor(self.summaries.profile_id, self.model, self.scheduler, request, prefix, self.limits)
        job = self.scheduler.submit_background(self.summaries.profile_id, request, executor,
                                               budget_seconds=self.limits.max_seconds, cancellation=cancellation,
                                               maintenance=True)
        return CompactionOperation(job.job_id, prefix, executor, cancellation)

    def finish(self, operation: CompactionOperation) -> SummaryCheckpoint | SummarySegment:
        if (operation.executor.scheduler is not self.scheduler or operation.executor.model is not self.model
                or operation.prefix != operation.executor.prefix
                or operation.job_id != operation.executor.request.request_id
                or operation.cancellation.turn_id != operation.executor.request.turn_id):
            raise ValueError('Compaction operation belongs to another host.')
        result = self.scheduler.wait_result(self.summaries.profile_id, operation.job_id)
        job = self.scheduler.get_job(self.summaries.profile_id, operation.job_id)
        if result.status == TaskState.CANCELLED or job.invalidated:
            raise GenerationCancelled('Compaction cancelled; no checkpoint was published.')
        if result.status != TaskState.COMPLETED or operation.executor.candidate is None:
            raise ValueError(result.error or 'Compaction produced no valid candidate.')
        # wait_result returns only after actual native unwind/lease release.
        commit = self.summaries.commit_segment if isinstance(operation.prefix, SegmentCapture) else self.summaries.commit
        return commit(operation.prefix, operation.executor.candidate, checkpoint_id=operation.job_id,
                                     limits=operation.executor.limits, cancellation=operation.cancellation)

    def compact(self, session_id: UUID, through_turn_id: UUID, *, cancellation: CancellationToken) -> SummaryCheckpoint:
        return self.finish(self.start(session_id, through_turn_id, cancellation=cancellation))

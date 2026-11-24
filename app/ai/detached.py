"""Verify detached task provenance at the worker and receipt boundaries."""

from dataclasses import dataclass
from hashlib import sha256

from .background.worker import BackgroundIntelligence
from .continuity.json_store import JSONContinuityStore
from .continuity.store import ContinuityConflict
from .continuity.task_findings import archive_finding
from .provenance import SourceAttribution


def committed_origin(store, request):
    if request.snapshot.profile_id != store.profile_id:
        raise ContinuityConflict('Task belongs to a different profile.')
    pair = store.committed_pair(request.conversation_id, request.session_created_at, request.snapshot.revision + 1)
    if (not pair or pair[0].turn_id != request.turn_id or pair[0].text != request.task
            or (pair[0].source,) != request.source_context[0].sources
            or sha256(pair[0].text.encode()).hexdigest() != request.detached.source_sha256
            or pair[1].outcome != 'completed' or not pair[1].text.strip()):
        raise ContinuityConflict('The exact acknowledged source exchange is unavailable.')
    return pair


def validated_finding(request, result, max_characters):
    expected = (SourceAttribution('l2', f'model-generated:{request.request_id}'),
                *request.source_context[0].sources)
    if (result is None or result.status != 'completed'
            or (result.request_id, result.conversation_id, result.turn_id) !=
            (request.request_id, request.conversation_id, request.turn_id)
            or len(result.findings) != 1 or result.findings[0].sources != expected
            or not result.findings[0].text.strip() or len(result.findings[0].text) > max_characters):
        raise ValueError('Detached findings do not match the authorized worker source.')
    return result.findings[0]


@dataclass(frozen=True, slots=True)
class DetachedExecutor:
    store: JSONContinuityStore
    executor: BackgroundIntelligence
    max_result_characters: int

    def commit_finding(self, request, result, cancellation):
        finding = validated_finding(request, result, self.max_result_characters)
        archive_finding(self.store, request, finding, cancellation)

    def execute(self, request, *, cancellation):
        cancellation.raise_if_cancelled()
        committed_origin(self.store, request)
        result = self.executor.execute(request, cancellation=cancellation)
        cancellation.raise_if_cancelled()
        if result.status == 'completed':
            validated_finding(request, result, self.max_result_characters)
            committed_origin(self.store, request)
        return result

"""Explicit L3 context assembly; no model, selection heuristics or compaction."""
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from ..contracts import ContextSnapshot, ConversationContext, ConversationSession, HistoryEntry
from ..provenance import SourceAttribution
from ..context_contracts import ContextReference, L3Projection, ProjectedMemory, ProjectedSegment
from ..conversation.cancellation import CancellationToken
from .compaction import JSONSummaryStore
from .compaction_contracts import SummaryCheckpoint
from .segment_contracts import SummarySegment
from .json_store import JSONContinuityStore, _identifier
from .memory import JSONMemoryStore
from .memory_contracts import MemoryNotFound, MemoryRecord
from .store import ContinuityConflict


@dataclass(frozen=True, slots=True)
class ContextAssembly:
    context: ConversationContext
    summary_state: Literal['disabled', 'missing', 'selected']
    summary: SummaryCheckpoint | None
    memories: tuple[MemoryRecord, ...]
    memory_lineages: tuple[tuple[MemoryRecord, ...], ...]
    segments: tuple[SummarySegment, ...] = ()


def _dialogue(entries: tuple[HistoryEntry, ...]) -> tuple[HistoryEntry, ...]:
    # User records remain visible even when the paired assistant was interrupted.
    return tuple(e for e in entries if e.role == 'user' or e.outcome == 'completed')


class ContextAssembler:
    """Borrow one profile store. Reader handles have no semantic/context cache.

    Package-local unlocked validators are reused under one existing profile lock;
    public get() methods would recursively acquire that lock. No schema changes.
    """
    def __init__(self, store: JSONContinuityStore, *, max_memories: int = 4):
        if type(max_memories) is not int or max_memories < 1:
            raise ValueError('Selected memory bound must be a positive integer.')
        self.store = store
        self.max_memories = max_memories
        self._summaries: JSONSummaryStore | None = None
        self._memories: JSONMemoryStore | None = None

    def memory_ids(self, values: tuple[UUID, ...]) -> tuple[UUID, ...]:
        for value in values:
            _identifier(value)
        result = tuple(dict.fromkeys(values))
        if len(result) > self.max_memories:
            raise ValueError('Too many explicitly selected memories.')
        return result

    def automatic_ids(self, pinned: tuple[UUID, ...], *, excluded: tuple[UUID, ...] = ()) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
        """Pinned first, then newest active approvals; recency is not relevance."""
        pinned = self.memory_ids(pinned)
        if self._memories is None:
            self._memories = JSONMemoryStore(self.store)
        with self.store._locked():
            _data, records = self._memories._state()
            active = {r.memory_id: r for r in records if r.status == 'active'}
            if any(identifier not in active for identifier in pinned):
                raise MemoryNotFound('Pinned memory is no longer active for this profile.')
            ordered = (*pinned, *(r.memory_id for r in sorted(active.values(),
                        key=lambda r: (r.approval.approved_at, str(r.memory_id)), reverse=True)
                        if r.memory_id not in pinned and r.memory_id not in excluded))
            return tuple(ordered[:self.max_memories]), tuple(ordered[self.max_memories:])

    def assemble(self, session: ConversationSession, *, use_summary: bool,
                 memory_ids: tuple[UUID, ...], max_recent_turns: int,
                 cancellation: CancellationToken, automatic: bool = False,
                 memory_token_budget: int = 256, protected_exchanges: int = 2,
                 segment_mode: bool = False, max_summary_segments: int = 4, summary_token_budget: int = 768) -> ContextAssembly:
        if session.profile_id != self.store.profile_id:
            raise ValueError('Context selection belongs to another profile.')
        if type(use_summary) is not bool or type(max_recent_turns) is not int or max_recent_turns < 1:
            raise ValueError('Invalid explicit summary/recent selection.')
        selected = self.memory_ids(memory_ids)
        cancellation.raise_if_cancelled()
        # Initialize only explicitly requested readers, outside the capture lock.
        if use_summary and self._summaries is None:
            self._summaries = JSONSummaryStore(self.store)
        if selected and self._memories is None:
            self._memories = JSONMemoryStore(self.store)
        with self.store._locked(cancellation):
            data, entries = self.store._load(session.conversation_id)
            current = self.store._session(data)
            if current.created_at != session.created_at:
                raise ContinuityConflict('Active session was deleted and replaced.')
            try:
                summary = self._summaries._read(current.conversation_id) if use_summary else None
            except FileNotFoundError:
                summary = None  # Removed checkpoint directory is also a missing summary.
            state = ('selected' if summary else 'missing') if use_summary else 'disabled'
            if summary is not None:
                self._summaries._validate_source(summary, data, entries)
            manifest = self._summaries._segments(data, entries, summary) if use_summary else None
            segment_mode = segment_mode or bool(manifest and manifest.segments)
            segments = manifest.segments if manifest else ()
            if segments:
                state = 'selected'
            memories, lineages, references = [], [], []
            if selected:
                _memory_data, records = self._memories._state()  # Existing evidence revalidation/redaction policy.
                by_id = {record.memory_id: record for record in records}
                for index, identifier in enumerate(selected, 1):
                    record = by_id.get(identifier)
                    if record is None or record.status != 'active':
                        raise MemoryNotFound('Selected memory is missing, deleted or superseded for this profile.')
                    memories.append(record)
                    references.append(ContextReference(f'm{index}', record.source.session_id,
                                                       record.source.turn_id, 'user', record.source.attribution))
                    chain = [record]
                    while chain[-1].supersedes is not None:
                        chain.append(by_id[chain[-1].supersedes])
                    lineages.append(tuple(reversed(chain)))
            if summary is not None or segments:
                boundary = 2 * (manifest.frontier if manifest else len(summary.covered_turn_ids))
                recent = _dialogue(entries[boundary:])  # Every original turn after coverage, without silent clipping.
                baseline = _dialogue(entries)
                references[:0] = [ContextReference(f's/{source.alias}', current.conversation_id,
                                                   source.turn_id, source.role, source.attribution, source.outcome)
                                  for source in (summary.sources if summary else ())]
            else:
                recent = _dialogue(entries if automatic else entries[-2 * max_recent_turns:])
                baseline = recent
            projected_segments = []
            if segment_mode:
                if summary:
                    projected_segments.append(ProjectedSegment('legacy', None, SourceAttribution('l3',f'summary:{summary.checkpoint_id}'),
                                                                entries=summary.candidate.entries))
                for index, segment in enumerate(segments, 1):
                    reference = f's{index}'
                    projected_segments.append(ProjectedSegment(reference, segment.candidate.text,
                                                               SourceAttribution('l3',f'segment:{segment.segment_id}'), incomplete=any(s.outcome != 'completed' for s in segment.sources)))
                    references.extend(ContextReference(f'{reference}/{s.alias}', current.conversation_id,
                                                       s.turn_id, s.role, s.attribution, s.outcome) for s in segment.sources)
            snapshot = ContextSnapshot(current.conversation_id, current.revision, recent=recent,
                                       profile_id=current.profile_id)
            projection = None
            if summary is not None or segments or memories or automatic or segment_mode:
                projection = L3Projection(
                    current.profile_id, current.conversation_id, current.created_at, current.revision,
                    summary.checkpoint_id if summary else None, summary.candidate.entries if summary and not segment_mode else (),
                    tuple(e.turn_id for e in entries[:2*manifest.frontier:2]) if manifest else (),
                    tuple(ProjectedMemory(f'm{index}', record.content, record.supersedes is not None)
                          for index, record in enumerate(memories, 1)), tuple(references), () if automatic else baseline, automatic, memory_token_budget,
                    protected_exchanges, tuple((entry.turn_id, entry.outcome) for entry in entries
                                               if entry.role == "assistant" and entry.outcome != "completed"),
                    segment_mode, tuple(projected_segments), max_summary_segments, summary_token_budget,
                )
            cancellation.raise_if_cancelled()
            return ContextAssembly(ConversationContext(snapshot, l3_projection=projection), state,
                                   summary, tuple(memories), tuple(lineages), tuple(segments))

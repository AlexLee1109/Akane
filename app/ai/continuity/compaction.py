"""L3 source capture and current summary checkpoint; never loads/runs a model."""
from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import json
import os
from uuid import UUID

from ..contracts import GenerationInfo
from ..conversation.cancellation import CancellationToken, GenerationCancelled
from ..provenance import SourceAttribution
from .compaction_contracts import (
    CompactionLimits, CompactionPrefix, SummaryCandidate, SummaryCheckpoint,
    SummaryNotFound, SummarySource,
)
from .json_store import (
    JSONContinuityStore, _date, _directory, _identifier, _keys, _read, _publish,
    _regular, _stored_uuid, _sync_directory, _timestamp,
)
from .store import ContinuityConflict
from .summary_data import LABEL, data_json, parse_output, sources_for, validate_entries


@contextmanager
def _summary_dir(store: JSONContinuityStore, *, create=False):
    descriptor = (_directory(store._profile_dir, 'summaries') if create else os.open(
        'summaries', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=store._profile_dir))
    try:
        if os.fstat(descriptor).st_mode & 0o077:
            raise ValueError('Summary directory must be private (0700).')
        yield descriptor
    finally:
        os.close(descriptor)


def _delete_session_summary(store: JSONContinuityStore, session_id: UUID) -> None:
    """Under the profile lock, remove a private checkpoint before source unlink."""
    try:
        with _summary_dir(store) as directory:
            for suffix in ('.json', '.segments.json'):
                name = _identifier(session_id) + suffix
                try:
                    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                except FileNotFoundError:
                    continue
                try:
                    _regular(descriptor)
                finally:
                    os.close(descriptor)
                os.unlink(name, dir_fd=directory)
                _sync_directory(directory)
    except FileNotFoundError:
        return


def _prefix(entries, through_turn_id: UUID, keep: int, *, include_diagnostics=False):
    _identifier(through_turn_id)
    if not entries:
        raise ValueError('Conversation has no eligible completed prefix.')
    boundary = next((index + 2 for index in range(0, len(entries), 2)
                     if entries[index].turn_id == through_turn_id), None)
    if boundary is None:
        raise ValueError('Prefix boundary is not a source turn.')
    selected = entries[:boundary]
    if not include_diagnostics and any(entry.outcome != 'completed' for entry in selected):
        raise ValueError('A contiguous compaction prefix must contain only completed exchanges.')
    if sum(entry.outcome == 'completed' for entry in entries[boundary + 1::2]) < keep:
        raise ValueError('Compaction must leave the configured recent completed exchanges outside its prefix.')
    return selected


def _digest(data, count):
    return hashlib.sha256(json.dumps(data['turns'][:count], ensure_ascii=False,
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _encode(checkpoint: SummaryCheckpoint) -> dict:
    candidate = checkpoint.candidate
    return {
        'schema_version': checkpoint.schema_version, 'checkpoint_id': str(checkpoint.checkpoint_id),
        'profile_id': str(checkpoint.profile_id), 'session_id': str(checkpoint.session_id),
        'session_created_at': _timestamp(checkpoint.session_created_at),
        'source_revision': checkpoint.source_revision, 'through_turn_id': str(checkpoint.through_turn_id),
        'covered_turn_ids': [str(value) for value in checkpoint.covered_turn_ids],
        'source_digest': checkpoint.source_digest,
        'sources': [{'alias': s.alias, 'turn_id': str(s.turn_id), 'role': s.role,
                     'created_at': _timestamp(s.created_at), 'attribution': asdict(s.attribution),
                     **({'outcome': s.outcome} if checkpoint.schema_version == 2 else {})} for s in checkpoint.sources],
        'limits': asdict(checkpoint.limits), 'generated_at': _timestamp(candidate.generated_at),
        'content': {'label': LABEL, 'entries': [asdict(entry) for entry in candidate.entries]},
        'generation': {'owner': 'l3', 'origin': 'model-generated', 'model': candidate.model,
                       'template_sha256': candidate.template_sha256, 'metrics': asdict(candidate.generation)},
        'tokens': {'source_context': candidate.source_tokens, 'summary_projection': candidate.summary_tokens},
    }


def _decode(data: dict) -> SummaryCheckpoint:
    _keys(data, {'schema_version', 'checkpoint_id', 'profile_id', 'session_id', 'session_created_at',
                 'source_revision', 'through_turn_id', 'covered_turn_ids', 'source_digest', 'sources',
                 'limits', 'generated_at', 'content', 'generation', 'tokens'})
    if (type(data['schema_version']) is not int or data['schema_version'] not in (1, 2)
            or type(data['source_revision']) is not int or data['source_revision'] < 1
            or not isinstance(data['covered_turn_ids'], list) or not isinstance(data['sources'], list)):
        raise ValueError('Invalid checkpoint schema or source revision.')
    _keys(data['limits'], {field.name for field in fields(CompactionLimits)})
    limits = CompactionLimits(**data['limits'])
    sources = []
    for item in data['sources']:
        _keys(item, {'alias', 'turn_id', 'role', 'created_at', 'attribution'}
              | ({'outcome'} if data['schema_version'] == 2 else set()))
        if item.get('outcome', 'completed') not in ('completed', 'cancelled', 'failed'):
            raise ValueError('Invalid source outcome.')
        _keys(item['attribution'], {'owner', 'source_id', 'uri'})
        sources.append(SummarySource(item['alias'], _stored_uuid(item['turn_id']), item['role'],
                                     _date(item['created_at']), SourceAttribution(**item['attribution']),
                                     item.get('outcome', 'completed')))
    _keys(data['content'], {'label', 'entries'})
    if data['content']['label'] != LABEL:
        raise ValueError('Summary trust label must remain unverified model data.')
    entries = parse_output(data_json({'entries': data['content']['entries']}), tuple(sources), limits.max_characters)
    generation = data['generation']
    _keys(generation, {'owner', 'origin', 'model', 'template_sha256', 'metrics'})
    _keys(generation['metrics'], {field.name for field in fields(GenerationInfo)})
    if (generation['owner'] != 'l3' or generation['origin'] != 'model-generated'
            or not isinstance(generation['model'], str) or not generation['model']):
        raise ValueError('Invalid summary generation provenance.')
    _keys(data['tokens'], {'source_context', 'summary_projection'})
    return SummaryCheckpoint(
        _stored_uuid(data['checkpoint_id']), _stored_uuid(data['profile_id']), _stored_uuid(data['session_id']),
        _date(data['session_created_at']), data['source_revision'], _stored_uuid(data['through_turn_id']),
        tuple(_stored_uuid(value) for value in data['covered_turn_ids']), data['source_digest'], tuple(sources), limits,
        SummaryCandidate(entries, _date(data['generated_at']), generation['model'], generation['template_sha256'],
                         data['tokens']['source_context'], data['tokens']['summary_projection'],
                         GenerationInfo(**generation['metrics'])), data['schema_version'],
    )


def _compression(candidate: SummaryCandidate, limits: CompactionLimits) -> None:
    if (any(type(value) is not int or value < 1 for value in (candidate.source_tokens, candidate.summary_tokens,
                                                           candidate.generation.prompt_tokens))
            or candidate.generation.prompt_tokens > limits.max_prompt_tokens
            or candidate.source_tokens - candidate.summary_tokens < limits.min_saved_tokens
            or candidate.summary_tokens / candidate.source_tokens > limits.max_summary_ratio):
        raise ValueError('Summary does not achieve the required exact-token compression or prompt bound.')


class JSONSummaryStore:
    """One checkpoint per session, with separate capture/commit and immutable reads."""

    def __init__(self, continuity: JSONContinuityStore):
        self._store = continuity
        with continuity._locked(), _summary_dir(continuity, create=True):
            pass

    @property
    def profile_id(self) -> UUID:
        return self._store.profile_id

    def _read(self, session_id):
        with _summary_dir(self._store) as directory:
            try:
                return _decode(_read(directory, _identifier(session_id) + '.json'))
            except FileNotFoundError:
                return None

    def _validate_source(self, checkpoint, data, entries):
        if (checkpoint.profile_id != self.profile_id or checkpoint.session_id != UUID(data['conversation_id'])
                or checkpoint.session_created_at != _date(data['created_at'])
                or checkpoint.source_revision > data['revision']):
            raise ContinuityConflict('Summary source profile, session incarnation or revision changed.')
        original = entries[:2 * checkpoint.source_revision]
        selected = _prefix(original, checkpoint.through_turn_id, checkpoint.limits.keep_recent_exchanges,
                           include_diagnostics=checkpoint.schema_version == 2)
        if (checkpoint.covered_turn_ids != tuple(entry.turn_id for entry in selected[::2])
                or checkpoint.sources != sources_for(selected, include_diagnostics=checkpoint.schema_version == 2)
                or checkpoint.source_digest != _digest(data, len(selected))):
            raise ContinuityConflict('Summary source prefix or attribution changed.')
        validate_entries(checkpoint.candidate.entries, checkpoint.sources, checkpoint.limits.max_characters)
        _compression(checkpoint.candidate, checkpoint.limits)

    def get(self, session_id: UUID) -> SummaryCheckpoint:
        _identifier(session_id)
        with self._store._locked():
            data, entries = self._store._load(session_id)
            checkpoint = self._read(session_id)
            if checkpoint is None:
                raise SummaryNotFound('No summary checkpoint exists for this session/profile.')
            self._validate_source(checkpoint, data, entries)
            return checkpoint

    def capture(self, session_id: UUID, through_turn_id: UUID, *, limits: CompactionLimits,
                cancellation: CancellationToken, include_diagnostics: bool = False) -> CompactionPrefix:
        _identifier(session_id)
        with self._store._locked(cancellation):
            data, entries = self._store._load(session_id)
            if type(include_diagnostics) is not bool:
                raise ValueError("Diagnostic coverage must be explicit.")
            selected = _prefix(entries, through_turn_id, limits.keep_recent_exchanges,
                               include_diagnostics=include_diagnostics)
            prior = self._read(session_id)
            if prior is not None:
                self._validate_source(prior, data, entries)
                if len(selected) <= 2 * len(prior.covered_turn_ids):
                    raise ContinuityConflict('Repeated compaction must advance the original-source boundary.')
            if self._segments(data, entries, prior).segments:
                raise ContinuityConflict('Legacy checkpoint is anchored by immutable segments; use incremental maintenance.')
            cancellation.raise_if_cancelled()
            return CompactionPrefix(self._store._session(data), through_turn_id, selected,
                                     _digest(data, len(selected)), prior.checkpoint_id if prior else None, include_diagnostics)

    def commit(self, prefix: CompactionPrefix, candidate: SummaryCandidate, *, checkpoint_id: UUID,
               limits: CompactionLimits, cancellation: CancellationToken) -> SummaryCheckpoint:
        _identifier(checkpoint_id)
        checkpoint = SummaryCheckpoint(
            checkpoint_id, prefix.session.profile_id, prefix.session.conversation_id, prefix.session.created_at,
            prefix.session.revision, prefix.through_turn_id, tuple(entry.turn_id for entry in prefix.entries[::2]),
            prefix.digest, sources_for(prefix.entries, include_diagnostics=prefix.include_diagnostics), limits, candidate,
            2 if prefix.include_diagnostics else 1,
        )
        encoded = _encode(checkpoint)
        with self._store._locked(cancellation):
            data, entries = self._store._load(checkpoint.session_id)
            self._validate_source(checkpoint, data, entries)
            if prefix.entries != entries[:len(prefix.entries)]:
                raise ContinuityConflict('Captured source records differ from the original prefix.')
            current = self._read(checkpoint.session_id)
            if current is not None and current.checkpoint_id == checkpoint_id:
                if _encode(current) == encoded:
                    return current  # Exact post-publication retry, without regeneration.
                raise ContinuityConflict('Checkpoint ID was reused with different content.')
            if self._segments(data, entries, current).segments:
                raise ContinuityConflict('A segment manifest now anchors this legacy checkpoint.')
            if (data['revision'] != prefix.session.revision
                    or (current.checkpoint_id if current else None) != prefix.prior_checkpoint_id):
                raise ContinuityConflict('Session or checkpoint changed during compaction; submit a new explicit request.')
            cancellation.raise_if_cancelled()
            with _summary_dir(self._store) as directory:
                try:
                    _publish(directory, str(checkpoint.session_id) + '.json', encoded, cancellation)
                except GenerationCancelled as error:
                    # Atomic replacement is the commit point. A cancellation
                    # observed only after durable publication cannot undo it.
                    try:
                        published = _read(directory, str(checkpoint.session_id) + '.json')
                    except FileNotFoundError:
                        raise error
                    if published != json.loads(json.dumps(encoded)):
                        raise error
            return checkpoint

    def _segments(self, data, entries, legacy):
        from .segments import read_validated
        return read_validated(self, data, entries, legacy)

    def get_segments(self, session_id: UUID):
        with self._store._locked():
            data, entries = self._store._load(session_id)
            legacy = self._read(session_id)
            if legacy:
                self._validate_source(legacy, data, entries)
            return self._segments(data, entries, legacy)

    def capture_segment(self, session_id: UUID, *, max_exchanges=4, limits: CompactionLimits,
                        cancellation: CancellationToken):
        from .segments import capture
        return capture(self, session_id, max_exchanges=max_exchanges, limits=limits, cancellation=cancellation)

    def commit_segment(self, captured, candidate, *, checkpoint_id: UUID, limits: CompactionLimits,
                       cancellation: CancellationToken):
        from .segments import commit
        return commit(self, captured, candidate, checkpoint_id=checkpoint_id, limits=limits, cancellation=cancellation)

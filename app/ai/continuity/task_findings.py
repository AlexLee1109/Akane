"""Private durable findings for accepted detached tasks.

The scheduler still owns availability and the runtime epoch. Findings are
archived separately from the transcript and disappear when their session is deleted.
"""

from contextlib import contextmanager
from dataclasses import asdict
import os
from uuid import UUID

from ..contracts import ContextRecord, ContextSnapshot, DelegationRequest, DelegationResult, DetachedBinding
from ..provenance import SourceAttribution
from .json_store import _date, _directory, _identifier, _keys, _publish, _read, _regular, _sync_directory, _timestamp
from .store import ContinuityConflict


@contextmanager
def _finding_directory(store, *, create=False):
    descriptor = (_directory(store._profile_dir, 'findings') if create else os.open(
        'findings', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=store._profile_dir))
    try:
        if os.fstat(descriptor).st_mode & 0o077:
            raise ValueError('Finding directory must be private (0700).')
        yield descriptor
    finally:
        os.close(descriptor)


def delivery_uri(task_id):
    return f'urn:akane:task:{_identifier(task_id)}'


def archive_finding(store, request, finding, cancellation):
    """Publish a validated result against the same durable source revision."""
    session_id = _identifier(request.conversation_id)
    task_id = _identifier(request.request_id)
    created_at = _timestamp(request.session_created_at)
    revision = request.snapshot.revision + 1
    receipt = {'runtime_epoch': _identifier(request.detached.runtime_epoch), 'origin_turn_id': _identifier(request.turn_id),
               'source_revision': revision, 'source_sha256': request.detached.source_sha256, 'finding': asdict(finding)}
    # JSON arrays must have the same representation on initial write and retry.
    receipt['finding']['sources'] = list(receipt['finding']['sources'])
    scope = {'schema_version': 1, 'conversation_id': session_id, 'session_created_at': created_at}
    with store._locked(cancellation):
        session, entries = store._load(request.conversation_id)
        pair = entries[2 * (revision - 1):2 * revision]
        if (session['created_at'] != created_at or request.snapshot.profile_id != store.profile_id
                or len(pair) != 2 or pair[0].turn_id != request.turn_id or pair[0].text != request.task
                or (pair[0].source,) != request.source_context[0].sources
                or pair[1].outcome != 'completed' or not pair[1].text.strip()):
            raise ContinuityConflict('Finding origin was replaced before archival.')
        with _finding_directory(store, create=True) as directory:
            filename = session_id + '.json'
            try:
                archive = _read(directory, filename)
            except FileNotFoundError:
                archive = {**scope, 'findings': {}}
            _keys(archive, set(scope) | {'findings'})
            if any(archive[key] != value for key, value in scope.items()) or not isinstance(archive['findings'], dict):
                raise ContinuityConflict('Finding archive belongs to another session.')
            previous = archive['findings'].get(task_id)
            if previous is not None:
                if {key: value for key, value in previous.items() if key != 'completed_at'} != receipt:
                    raise ContinuityConflict('Task already has a different archived finding.')
                return
            receipt['completed_at'] = _timestamp(store._clock())
            archive['findings'][task_id] = receipt
            # The caller holds the scheduler condition. Once this check passes,
            # complete the local write even if a lease deadline expires during fsync.
            cancellation.raise_if_cancelled()
            _publish(directory, filename, archive, None)


def recover_findings(store, session, runtime_epoch):
    """Rebind a prior runtime's unconsumed, restart-capable receipts."""
    with store._locked():
        data, entries = store._load(session.conversation_id)
        created_at = _timestamp(session.created_at)
        if data['created_at'] != created_at:
            return ()
        try:
            with _finding_directory(store) as directory:
                archive = _read(directory, f'{session.conversation_id}.json')
        except FileNotFoundError:
            return ()
        _keys(archive, {'schema_version', 'conversation_id', 'session_created_at', 'findings'})
        if (type(archive['schema_version']) is not int or archive['schema_version'] != 1
                or archive['conversation_id'] != str(session.conversation_id)
                or archive['session_created_at'] != created_at or not isinstance(archive['findings'], dict)):
            raise ContinuityConflict('Finding archive belongs to another session.')
        delivered = {entry.source.uri for entry in entries
                     if entry.role == 'assistant' and entry.outcome == 'completed' and entry.source.uri}
        # Phase 7B9 records have no durable delivery marker. They remain archived
        # but cannot safely be offered again after a restart.
        recovered = []
        for identifier, record in archive['findings'].items():
            task_id = UUID(identifier)
            if (not isinstance(record, dict) or 'completed_at' not in record
                    or delivery_uri(task_id) in delivered or record['runtime_epoch'] == str(runtime_epoch)):
                continue
            _keys(record, {'runtime_epoch', 'origin_turn_id', 'source_revision', 'source_sha256',
                           'finding', 'completed_at'})
            revision = record['source_revision']
            original_epoch = UUID(record['runtime_epoch'])
            if (type(revision) is not int or not 0 < revision <= data['revision']
                    or str(original_epoch) != record['runtime_epoch']
                    or str(task_id) != identifier or record['origin_turn_id'] != identifier):
                raise ContinuityConflict('Archived task origin is invalid.')
            source = entries[2 * (revision - 1)]
            finding = record['finding']
            _keys(finding, {'text', 'sources'})
            request = DelegationRequest(task_id, session.conversation_id, task_id, source.text,
                ContextSnapshot(session.conversation_id, revision - 1, profile_id=store.profile_id),
                source_turn_id=task_id, source_context=(ContextRecord(source.text, (source.source,)),),
                session_created_at=session.created_at, detached=DetachedBinding(runtime_epoch, record['source_sha256']))
            payload = DelegationResult(task_id, session.conversation_id, task_id, 'completed',
                (ContextRecord(finding['text'], tuple(SourceAttribution(**item) for item in finding['sources'])),))
            recovered.append((request, payload, _date(record['completed_at'])))
        return tuple(sorted(recovered, key=lambda row: row[2], reverse=True))


def delete_session_findings(store, session_id):
    """Remove a session's findings while the caller holds the profile lock."""
    try:
        with _finding_directory(store) as directory:
            filename = _identifier(session_id) + '.json'
            descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                _regular(descriptor)
            finally:
                os.close(descriptor)
            os.unlink(filename, dir_fd=directory)
            _sync_directory(directory)
    except FileNotFoundError:
        return

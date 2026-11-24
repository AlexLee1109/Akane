"""Manual, verbatim user memory, sharing the transcript store's profile lock/I/O.

Only this explicit API reads memories.json. Conversation snapshots do not use it.
Source validation and memory publication hold the same lock as session deletion.
"""

from dataclasses import replace
from uuid import UUID, uuid4

from ..conversation.cancellation import CancellationToken
from ..provenance import SourceAttribution
from .json_store import (
    JSONContinuityStore, _date, _expected_revision, _identifier, _keys, _publish,
    _read, _stored_uuid, _timestamp,
)
from .memory_contracts import (
    InvalidMemorySource, MemoryApproval, MemoryDeletion, MemoryNotFound,
    MemoryProposal, MemoryRecord, MemorySnapshot, MemorySource,
)
from .store import ContinuityConflict, SessionNotFound

MAX_MEMORY_CHARACTERS = 2048
_FILE = "memories.json"
_FIELDS = {"memory_id", "content", "source", "created_at", "approval", "status",
           "supersedes", "superseded_by", "deletion"}
_SOURCE_FIELDS = {"session_id", "session_created_at", "turn_id", "turn_created_at",
                  "start", "end", "attribution", "role"}


def _content(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_MEMORY_CHARACTERS:
        raise ValueError("A memory requires 1–2048 characters of nonblank exact source text.")


def _source_shape(source: MemorySource, content: str) -> None:
    _identifier(source.session_id)
    _identifier(source.turn_id)
    _timestamp(source.session_created_at)
    _timestamp(source.turn_created_at)
    if (source.role != "user" or source.attribution.owner != "user"
            or not isinstance(source.attribution.source_id, str) or not source.attribution.source_id
            or (source.attribution.uri is not None and not isinstance(source.attribution.uri, str))
            or type(source.start) is not int or type(source.end) is not int
            or source.start < 0 or source.end - source.start != len(content)):
        raise InvalidMemorySource("Memory evidence must identify an exact user-authored span.")


def _authorization(approval: MemoryApproval, profile_id: UUID, memory_id: UUID, action: str) -> None:
    if (not isinstance(approval, MemoryApproval) or approval.approved is not True
            or approval.actor != "user" or approval.profile_id != profile_id
            or approval.memory_id != memory_id or approval.action != action):
        raise PermissionError("Explicit user approval must match this profile, memory ID and action.")
    _timestamp(approval.approved_at)


def _approval_json(approval: MemoryApproval) -> dict:
    return {"actor": approval.actor, "action": approval.action,
            "approved": approval.approved, "approved_at": _timestamp(approval.approved_at)}


def _parse_approval(data: dict, profile_id: UUID, memory_id: UUID, action: str) -> MemoryApproval:
    _keys(data, {"actor", "action", "approved", "approved_at"})
    approval = MemoryApproval(profile_id, memory_id, data["action"], _date(data["approved_at"]),
                              data["approved"], data["actor"])
    try:
        _authorization(approval, profile_id, memory_id, action)
    except PermissionError as error:
        raise ValueError("Stored memory approval metadata is invalid.") from error
    return approval


def _encode(record: MemoryRecord) -> dict:
    source = record.source
    deletion = record.deletion
    return {
        "memory_id": str(record.memory_id), "content": record.content,
        "source": {
            "session_id": str(source.session_id), "session_created_at": _timestamp(source.session_created_at),
            "turn_id": str(source.turn_id), "turn_created_at": _timestamp(source.turn_created_at),
            "start": source.start, "end": source.end, "role": source.role,
            "attribution": {"owner": source.attribution.owner, "source_id": source.attribution.source_id,
                            "uri": source.attribution.uri},
        } if source else None,
        "created_at": _timestamp(record.created_at), "approval": _approval_json(record.approval),
        "status": record.status, "supersedes": str(record.supersedes) if record.supersedes else None,
        "superseded_by": str(record.superseded_by) if record.superseded_by else None,
        "deletion": {"reason": deletion.reason, "deleted_at": _timestamp(deletion.deleted_at),
                     "approval": _approval_json(deletion.approval) if deletion.approval else None} if deletion else None,
    }


def _parse_record(data: dict, profile_id: UUID) -> MemoryRecord:
    _keys(data, _FIELDS)
    memory_id = _stored_uuid(data["memory_id"])
    previous = _stored_uuid(data["supersedes"]) if data["supersedes"] is not None else None
    following = _stored_uuid(data["superseded_by"]) if data["superseded_by"] is not None else None
    approval = _parse_approval(data["approval"], profile_id, memory_id, "supersede" if previous else "admit")
    source, deletion = None, None
    if data["status"] == "deleted":
        if data["content"] is not None or data["source"] is not None:
            raise ValueError("Deleted memory must have its content and source erased.")
        item = data["deletion"]
        _keys(item, {"reason", "deleted_at", "approval"})
        if item["reason"] not in ("manual", "source_deletion", "source_unavailable"):
            raise ValueError("Unknown memory deletion reason.")
        deletion = MemoryDeletion(item["reason"], _date(item["deleted_at"]),
                                  _parse_approval(item["approval"], profile_id, memory_id, "delete")
                                  if item["approval"] is not None else None)
        if (deletion.reason == "manual") != (deletion.approval is not None):
            raise ValueError("Only manual memory deletion requires its own approval metadata.")
    elif data["status"] in ("active", "superseded"):
        _content(data["content"])
        if data["deletion"] is not None:
            raise ValueError("Live memory cannot have deletion metadata.")
        item = data["source"]
        _keys(item, _SOURCE_FIELDS)
        _keys(item["attribution"], {"owner", "source_id", "uri"})
        source = MemorySource(
            _stored_uuid(item["session_id"]), _date(item["session_created_at"]),
            _stored_uuid(item["turn_id"]), _date(item["turn_created_at"]), item["start"], item["end"],
            SourceAttribution(**item["attribution"]), item["role"],
        )
        _source_shape(source, data["content"])
        if (data["status"] == "superseded") != (following is not None):
            raise ValueError("Active/superseded status must match the correction lineage.")
    else:
        raise ValueError("Unknown memory status.")
    return MemoryRecord(memory_id, profile_id, data["content"], source, _date(data["created_at"]),
                        approval, data["status"], previous, following, deletion)


def _validate(data: dict, profile_id: UUID) -> tuple[MemoryRecord, ...]:
    _keys(data, {"schema_version", "profile_id", "revision", "records"})
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or _stored_uuid(data["profile_id"]) != profile_id
            or type(data["revision"]) is not int or data["revision"] < 0
            or not isinstance(data["records"], list)):
        raise ValueError("Invalid memory schema, profile or revision.")
    records = tuple(_parse_record(item, profile_id) for item in data["records"])
    by_id = {record.memory_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate memory ID.")
    if data["revision"] < len(records):
        raise ValueError("Memory revision cannot precede its admitted records.")
    for record in records:
        for neighbor, field in ((record.supersedes, "superseded_by"), (record.superseded_by, "supersedes")):
            if neighbor is not None and (neighbor not in by_id or getattr(by_id[neighbor], field) != record.memory_id):
                raise ValueError("Broken memory correction lineage.")
    visited = set()
    for record in records:
        chain = set()
        current = record
        while current.memory_id not in visited:
            if current.memory_id in chain:
                raise ValueError("Cyclic memory correction lineage.")
            chain.add(current.memory_id)
            if current.superseded_by is None:
                break
            current = by_id[current.superseded_by]
        visited.update(chain)
    return records


def _load(store: JSONContinuityStore) -> tuple[dict, tuple[MemoryRecord, ...]]:
    data = _read(store._profile_dir, _FILE)
    return data, _validate(data, store.profile_id)


def _save(store: JSONContinuityStore, data: dict, records: tuple[MemoryRecord, ...],
          cancellation: CancellationToken | None = None) -> None:
    data["records"] = [_encode(record) for record in records]
    data["revision"] += 1
    _validate(data, store.profile_id)
    _publish(store._profile_dir, _FILE, data, cancellation)


def _redact(record: MemoryRecord, deletion: MemoryDeletion) -> MemoryRecord:
    return replace(record, status="deleted", content=None, source=None, deletion=deletion)


def _delete_source_memories(store: JSONContinuityStore, session_id: UUID,
                            cancellation: CancellationToken | None) -> None:
    """Called with the existing profile lock held, before unlinking a session.

    Redaction-first is deliberately conservative: a later failed unlink may leave
    the transcript present but memories deleted, never dangling active evidence.
    """
    try:
        data, records = _load(store)
    except FileNotFoundError:
        return  # Ordinary transcript use does not create a memory file.
    deletion = MemoryDeletion("source_deletion", store._clock())
    updated = tuple(_redact(record, deletion) if record.source is not None
                    and record.source.session_id == session_id else record for record in records)
    if updated != records:
        _save(store, data, updated, cancellation)


class JSONMemoryStore:
    """Borrow an open JSONContinuityStore; it retains ownership of its descriptors.

    Verbatim admission only. Caller obtains actual user approval and supplies the
    typed assertion; this class checks bindings, not human identity/authentication.
    """

    def __init__(self, continuity: JSONContinuityStore) -> None:
        self._store = continuity
        with continuity._locked():
            try:
                _load(continuity)
            except FileNotFoundError:
                _publish(continuity._profile_dir, _FILE, {
                    "schema_version": 1, "profile_id": str(continuity.profile_id), "revision": 0, "records": [],
                }, None)

    @property
    def profile_id(self) -> UUID:
        return self._store.profile_id

    def _source(self, proposal: MemoryProposal, source_archive=None) -> None:
        _identifier(proposal.memory_id)
        if proposal.profile_id != self.profile_id:
            raise InvalidMemorySource("Memory proposal belongs to another profile.")
        _content(proposal.content)
        source = proposal.source
        _source_shape(source, proposal.content)
        if source_archive is not None and source_archive[0]['conversation_id'] == str(source.session_id):
            data, entries = source_archive
        else:
            data, entries = self._store._load(source.session_id)
        user = next((entry for entry in entries if entry.turn_id == source.turn_id and entry.role == "user"), None)
        if (user is None or self._store._session(data).created_at != source.session_created_at
                or user.created_at != source.turn_created_at or user.source != source.attribution
                or user.outcome != "completed" or user.text[source.start:source.end] != proposal.content):
            raise InvalidMemorySource("Memory span does not match its original durable user statement.")

    def _state(self, *, source_archive=None) -> tuple[dict, tuple[MemoryRecord, ...]]:
        data, records = _load(self._store)
        updated = []
        for record in records:
            if record.source is not None:
                try:
                    self._source(MemoryProposal(record.memory_id, self.profile_id, record.content, record.source),
                                 source_archive)
                except (SessionNotFound, InvalidMemorySource):
                    record = _redact(record, MemoryDeletion("source_unavailable", self._store._clock()))
            updated.append(record)
        result = tuple(updated)
        if result != records:
            _save(self._store, data, result)
        return data, result

    @staticmethod
    def _revision(expected_revision: int) -> None:
        _expected_revision(expected_revision)
        if expected_revision is None:
            raise ValueError("Memory mutations require an expected revision.")

    @staticmethod
    def _compare(data: dict, expected_revision: int) -> None:
        if data["revision"] != expected_revision:
            raise ContinuityConflict("Memory revision changed; read a fresh memory snapshot.")

    def propose(self, session_id: UUID, turn_id: UUID, quote: str, *,
                start: int | None = None, memory_id: UUID | None = None) -> MemoryProposal:
        _identifier(session_id)
        _identifier(turn_id)
        _content(quote)
        memory_id = uuid4() if memory_id is None else memory_id
        _identifier(memory_id)
        with self._store._locked():
            data, entries = self._store._load(session_id)
            user = next((entry for entry in entries if entry.turn_id == turn_id and entry.role == "user"), None)
            if user is None:
                raise InvalidMemorySource("Source user turn does not exist.")
            if start is None:
                start = user.text.find(quote)
                if start >= 0 and user.text.find(quote, start + 1) >= 0:
                    raise InvalidMemorySource("Quote is ambiguous; supply an exact character start offset.")
            if type(start) is not int or start < 0:
                raise InvalidMemorySource("Exact quote is absent or span offset is invalid.")
            proposal = MemoryProposal(memory_id, self.profile_id, quote, MemorySource(
                session_id, self._store._session(data).created_at, turn_id, user.created_at,
                start, start + len(quote), user.source,
            ))
            self._source(proposal, source_archive=(data, entries))
            return proposal

    def _admit(self, proposal: MemoryProposal, approval: MemoryApproval, expected_revision: int,
               previous: UUID | None) -> MemoryRecord:
        self._revision(expected_revision)
        _authorization(approval, self.profile_id, proposal.memory_id, "supersede" if previous else "admit")
        with self._store._locked():
            self._source(proposal)  # Recheck at approval, not merely at proposal time.
            data, records = self._state()
            by_id = {record.memory_id: record for record in records}
            existing = by_id.get(proposal.memory_id)
            if existing is not None:
                if (existing.content, existing.source, existing.approval, existing.supersedes) == (
                    proposal.content, proposal.source, approval, previous,
                ):
                    return existing  # Exact retry, even after a subsequent correction.
                raise ContinuityConflict("Memory ID already exists with different content or approval.")
            self._compare(data, expected_revision)
            if previous is not None:
                old = by_id.get(previous)
                if old is None or old.status != "active":
                    raise MemoryNotFound("Only an active memory can be superseded.")
                if (old.source.session_id, old.source.turn_id) == (proposal.source.session_id, proposal.source.turn_id):
                    raise InvalidMemorySource("Correction requires a different explicit source user turn.")
                records = tuple(replace(record, status="superseded", superseded_by=proposal.memory_id)
                                if record.memory_id == previous else record for record in records)
            record = MemoryRecord(proposal.memory_id, self.profile_id, proposal.content, proposal.source,
                                  self._store._clock(), approval, supersedes=previous)
            _save(self._store, data, (*records, record))
            return record

    def approve(self, proposal: MemoryProposal, *, approval: MemoryApproval, expected_revision: int) -> MemoryRecord:
        return self._admit(proposal, approval, expected_revision, None)

    def supersede(self, memory_id: UUID, proposal: MemoryProposal, *,
                  approval: MemoryApproval, expected_revision: int) -> MemoryRecord:
        _identifier(memory_id)
        return self._admit(proposal, approval, expected_revision, memory_id)

    def delete(self, memory_id: UUID, *, approval: MemoryApproval, expected_revision: int) -> int:
        _identifier(memory_id)
        self._revision(expected_revision)
        _authorization(approval, self.profile_id, memory_id, "delete")
        with self._store._locked():
            data, records = self._state()
            record = next((record for record in records if record.memory_id == memory_id), None)
            if record is None:
                raise MemoryNotFound("Memory is not available for this profile.")
            if record.status == "deleted":
                if record.deletion.approval == approval:
                    return data["revision"]
                raise ContinuityConflict("Memory is already deleted; original payload cannot be restored.")
            self._compare(data, expected_revision)
            deletion = MemoryDeletion("manual", self._store._clock(), approval)
            _save(self._store, data, tuple(_redact(item, deletion) if item.memory_id == memory_id else item
                                          for item in records))
            return data["revision"]

    def list_active(self) -> MemorySnapshot:
        with self._store._locked():
            data, records = self._state()
            return MemorySnapshot(self.profile_id, data["revision"], tuple(r for r in records if r.status == "active"))

    def get(self, memory_id: UUID, *, include_inactive: bool = False) -> MemoryRecord:
        _identifier(memory_id)
        with self._store._locked():
            _data, records = self._state()
            for record in records:
                if record.memory_id == memory_id and (record.status == "active" or include_inactive):
                    return record
            raise MemoryNotFound("Memory is not available in this profile/status view.")

    def history(self, memory_id: UUID) -> tuple[MemoryRecord, ...]:
        _identifier(memory_id)
        with self._store._locked():
            _data, records = self._state()
            by_id = {record.memory_id: record for record in records}
            if memory_id not in by_id:
                raise MemoryNotFound("Memory is not available for this profile.")
            record = by_id[memory_id]
            while record.supersedes is not None:
                record = by_id[record.supersedes]
            result = [record]
            while record.superseded_by is not None:
                record = by_id[record.superseded_by]
                result.append(record)
            return tuple(result)

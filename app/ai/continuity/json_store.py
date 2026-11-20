"""Private, human-readable session files with locked atomic replacement."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import stat
import time
from uuid import UUID, uuid4

from ..contracts import ContextSnapshot, ConversationSession, HistoryEntry, Profile
from ..conversation.cancellation import CancellationToken
from ..provenance import SourceAttribution
from .store import ContinuityConflict, SessionNotFound


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone.")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _identifier(value: UUID) -> str:
    if not isinstance(value, UUID):
        raise ValueError("Profile, session, and turn identifiers must be UUID values.")
    return str(value)


def _stored_uuid(value) -> UUID:
    if not isinstance(value, str):
        raise ValueError("Stored identifiers must be strings.")
    result = UUID(value)
    if str(result) != value:
        raise ValueError("Stored identifiers must be canonical UUID strings.")
    return result


def _date(value) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Stored timestamps must be strings.")
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("Stored timestamps must include a timezone.")
    return result


def _keys(value, names: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("Invalid or unsupported JSON storage fields.")


def _expected_revision(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("Expected revision must be a nonnegative integer.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def _directory(parent: int, name: str, *, private: bool = True) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
        _sync_directory(parent)
    except FileExistsError:
        pass
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    if private and (os.fstat(descriptor).st_mode & 0o077):
        os.close(descriptor)
        raise ValueError("Storage directories must be private (0700).")
    return descriptor


def _root(path: Path) -> int:
    if ".." in path.parts:
        raise ValueError("Parent traversal is not allowed in the data directory.")
    path = path.absolute()
    if path == Path(path.anchor):
        raise ValueError("The data directory cannot be the filesystem root.")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parts[1:]):
            child = _directory(descriptor, part, private=index == len(path.parts) - 2)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _regular(descriptor: int) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        raise ValueError("Storage files must be owned, regular files without hard links.")
    if info.st_mode & 0o077:
        raise ValueError("Storage files must be private (0600).")


def _read(directory: int, name: str):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        _regular(descriptor)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return json.load(stream, object_pairs_hook=_unique_object)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sync_directory(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise


def _publish(directory: int, name: str, data: dict, cancellation: CancellationToken | None) -> None:
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = f".{uuid4()}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        # Reject replaced symlinks or non-regular destination files as well.
        try:
            current = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            pass
        else:
            try:
                _regular(current)
            finally:
                os.close(current)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        _sync_directory(directory)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


class JSONContinuityStore:
    """One profile per handle. All cooperating readers/writers lock that profile.

    POSIX local filesystems only. Locks are advisory; edit files only while Akane
    is stopped. Open directory descriptors and O_NOFOLLOW reject symlink paths.
    """

    def __init__(
        self, data_dir: Path, profile_id: UUID, *, clock: Callable[[], datetime] = _now,
    ) -> None:
        _identifier(profile_id)
        self._profile_id = profile_id
        self._clock = clock
        self._descriptors: list[int] = []
        try:
            root = _root(data_dir)
            self._descriptors.append(root)
            profiles = _directory(root, "profiles")
            self._descriptors.append(profiles)
            self._profile_dir = _directory(profiles, str(profile_id))
            self._descriptors.append(self._profile_dir)
            self._sessions_dir = _directory(self._profile_dir, "sessions")
            self._descriptors.append(self._sessions_dir)
            with self._locked():
                try:
                    self._read_profile()
                except FileNotFoundError:
                    _publish(self._profile_dir, "profile.json", {
                        "schema_version": 1, "profile_id": str(profile_id),
                        "created_at": _timestamp(clock()),
                    }, None)
        except BaseException:
            self.close()
            raise

    @property
    def profile_id(self) -> UUID:
        return self._profile_id

    @contextmanager
    def _locked(self, cancellation: CancellationToken | None = None) -> Iterator[None]:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        descriptor = os.open(
            ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600, dir_fd=self._profile_dir,
        )
        try:
            _regular(descriptor)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if cancellation is not None:
                        cancellation.raise_if_cancelled()
                    time.sleep(0.01)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            yield
        finally:
            os.close(descriptor)

    def _read_profile(self) -> Profile:
        data = _read(self._profile_dir, "profile.json")
        _keys(data, {"schema_version", "profile_id", "created_at"})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported profile schema version.")
        if _stored_uuid(data["profile_id"]) != self.profile_id:
            raise ValueError("Profile metadata does not match its directory.")
        return Profile(self.profile_id, _date(data["created_at"]))

    @property
    def profile(self) -> Profile:
        with self._locked():
            return self._read_profile()

    def _validate(self, data: dict, conversation_id: UUID) -> tuple[HistoryEntry, ...]:
        _keys(data, {"schema_version", "profile_id", "conversation_id", "created_at", "updated_at", "revision", "turns"})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported conversation schema version.")
        if _stored_uuid(data["profile_id"]) != self.profile_id or _stored_uuid(data["conversation_id"]) != conversation_id:
            raise ValueError("Conversation identity does not match its path.")
        _date(data["created_at"])
        _date(data["updated_at"])
        if type(data["revision"]) is not int or data["revision"] < 0:
            raise ValueError("Invalid conversation revision.")
        if not isinstance(data["turns"], list) or len(data["turns"]) != 2 * data["revision"]:
            raise ValueError("Revision must match the number of complete exchanges.")
        entries = []
        seen = set()
        for index, item in enumerate(data["turns"]):
            _keys(item, {"turn_id", "role", "content", "timestamp", "source", "outcome"})
            _keys(item["source"], {"owner", "source_id", "uri"})
            if not isinstance(item["content"], str) or not isinstance(item["source"]["source_id"], str):
                raise ValueError("Content and source IDs must be strings.")
            if item["source"]["uri"] is not None and not isinstance(item["source"]["uri"], str):
                raise ValueError("Source URI must be a string or null.")
            entry = HistoryEntry(
                _stored_uuid(item["turn_id"]), item["role"], item["content"], item["outcome"],
                _date(item["timestamp"]), self.profile_id, conversation_id,
                SourceAttribution(**item["source"]),
            )
            if index % 2 == 0:
                if entry.role != "user" or entry.turn_id in seen:
                    raise ValueError("Invalid or duplicate user exchange.")
                seen.add(entry.turn_id)
            elif entry.role != "assistant" or entry.turn_id != entries[-1].turn_id:
                raise ValueError("Exchanges require matching user/assistant records.")
            entries.append(entry)
        return tuple(entries)

    def _load(self, conversation_id: UUID) -> tuple[dict, tuple[HistoryEntry, ...]]:
        name = _identifier(conversation_id) + ".json"
        try:
            data = _read(self._sessions_dir, name)
        except FileNotFoundError as error:
            raise SessionNotFound("Session is not available for this profile.") from error
        return data, self._validate(data, conversation_id)

    def _session(self, data: dict) -> ConversationSession:
        return ConversationSession(
            UUID(data["conversation_id"]), self.profile_id, _date(data["created_at"]), data["revision"],
        )

    def create_session(self, conversation_id: UUID | None = None) -> ConversationSession:
        conversation_id = uuid4() if conversation_id is None else conversation_id
        name = _identifier(conversation_id) + ".json"
        with self._locked():
            try:
                self._load(conversation_id)
            except SessionNotFound:
                pass
            else:
                raise ContinuityConflict("Session ID already exists.")
            timestamp = _timestamp(self._clock())
            data = {"schema_version": 1, "profile_id": str(self.profile_id),
                    "conversation_id": str(conversation_id), "created_at": timestamp,
                    "updated_at": timestamp, "revision": 0, "turns": []}
            self._validate(data, conversation_id)
            _publish(self._sessions_dir, name, data, None)
            return self._session(data)

    def get_session(self, conversation_id: UUID) -> ConversationSession:
        with self._locked():
            return self._session(self._load(conversation_id)[0])

    def list_sessions(self) -> tuple[ConversationSession, ...]:
        with self._locked():
            sessions = [self._session(self._load(_stored_uuid(name[:-5]))[0])
                        for name in os.listdir(self._sessions_dir) if name.endswith(".json")]
            return tuple(sorted(sessions, key=lambda item: (item.created_at, str(item.conversation_id))))

    def _record(self, entry: HistoryEntry, conversation_id: UUID, timestamp: datetime) -> dict:
        _identifier(entry.turn_id)
        if entry.profile_id not in (None, self.profile_id) or entry.conversation_id not in (None, conversation_id):
            raise ValueError("Entry belongs to another profile or session.")
        entry = replace(entry, profile_id=self.profile_id, conversation_id=conversation_id)
        source = entry.source
        return {"turn_id": str(entry.turn_id), "role": entry.role, "content": entry.text,
                "timestamp": _timestamp(entry.created_at if entry.created_at is not None else timestamp),
                "source": {"owner": source.owner, "source_id": source.source_id, "uri": source.uri},
                "outcome": entry.outcome}

    def append_turn(
        self, conversation_id: UUID, user: HistoryEntry, reply: HistoryEntry, *,
        expected_revision: int | None = None, cancellation: CancellationToken | None = None,
    ) -> int:
        _identifier(conversation_id)
        _expected_revision(expected_revision)
        if user.role != "user" or reply.role != "assistant" or user.turn_id != reply.turn_id:
            raise ValueError("A turn requires matching user/assistant entries.")
        if cancellation is not None and cancellation.turn_id != user.turn_id:
            raise ValueError("Storage cancellation token belongs to another turn.")
        # Validate the incoming pair before taking the lock; timestamps are
        # finalized against the accepted record on an idempotent retry.
        records = [self._record(entry, conversation_id, self._clock()) for entry in (user, reply)]
        with self._locked(cancellation):
            data, entries = self._load(conversation_id)
            for index in range(0, len(entries), 2):
                if entries[index].turn_id == user.turn_id:
                    retry = [self._record(entry, conversation_id, entries[index + offset].created_at)
                             for offset, entry in enumerate((user, reply))]
                    if retry != data["turns"][index:index + 2]:
                        raise ContinuityConflict("Turn ID was already committed with different content or metadata.")
                    if cancellation is not None:
                        cancellation.raise_if_cancelled()
                    return index // 2 + 1
            if expected_revision is not None and expected_revision != data["revision"]:
                raise ContinuityConflict("Session revision changed; read a fresh snapshot.")
            data["turns"].extend(records)
            data["revision"] += 1
            data["updated_at"] = _timestamp(self._clock())
            self._validate(data, conversation_id)
            _publish(self._sessions_dir, str(conversation_id) + ".json", data, cancellation)
            return data["revision"]

    def transcript(self, conversation_id: UUID) -> tuple[HistoryEntry, ...]:
        with self._locked():
            return self._load(conversation_id)[1]

    def snapshot(
        self, conversation_id: UUID, *, max_recent_turns: int, cancellation: CancellationToken,
    ) -> ContextSnapshot:
        if type(max_recent_turns) is not int or max_recent_turns < 1:
            raise ValueError("Recent context must allow at least one turn.")
        with self._locked(cancellation):
            data, entries = self._load(conversation_id)
            selected = []
            for index in range(max(0, len(entries) - 2 * max_recent_turns), len(entries), 2):
                selected.append(entries[index])
                if entries[index + 1].outcome == "completed":
                    selected.append(entries[index + 1])
            cancellation.raise_if_cancelled()
            return ContextSnapshot(conversation_id, data["revision"], recent=tuple(selected), profile_id=self.profile_id)

    def delete_session(
        self, conversation_id: UUID, *, expected_revision: int | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        _expected_revision(expected_revision)
        with self._locked(cancellation):
            data, _entries = self._load(conversation_id)
            if expected_revision is not None and expected_revision != data["revision"]:
                raise ContinuityConflict("Session revision changed; deletion was not applied.")
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            # Dependent memory is redacted first, under this same profile lock.
            # A failed unlink may conservatively leave history without memory,
            # but cannot leave deleted history backing an active memory record.
            from .memory import _delete_source_memories
            _delete_source_memories(self, conversation_id, cancellation)
            from .compaction import _delete_session_summary
            _delete_session_summary(self, conversation_id)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            os.unlink(str(conversation_id) + ".json", dir_fd=self._sessions_dir)
            _sync_directory(self._sessions_dir)
            if cancellation is not None:
                cancellation.raise_if_cancelled()

    def close(self) -> None:
        while self._descriptors:
            os.close(self._descriptors.pop())

    def __enter__(self) -> "JSONContinuityStore":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()

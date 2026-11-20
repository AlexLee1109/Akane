"""L3 transcript port and separately deferred memory/compaction contracts."""

from typing import Protocol
from uuid import UUID

from ..contracts import ContextRecord, ContextSnapshot, ConversationSession, HistoryEntry
from ..conversation.cancellation import CancellationToken


class ContinuityConflict(RuntimeError):
    """A write conflicts with an existing turn, session, or snapshot revision."""


class SessionNotFound(LookupError):
    """The session is unavailable to the selected profile."""


class ContinuityStore(Protocol):
    @property
    def profile_id(self) -> UUID:
        ...

    def create_session(self, conversation_id: UUID | None = None) -> ConversationSession:
        ...

    def get_session(self, conversation_id: UUID) -> ConversationSession:
        ...

    def list_sessions(self) -> tuple[ConversationSession, ...]:
        ...

    def transcript(self, conversation_id: UUID) -> tuple[HistoryEntry, ...]:
        """Return the full original transcript for this profile's session."""
        ...

    def delete_session(
        self, conversation_id: UUID, *, expected_revision: int | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        ...

    def snapshot(
        self, conversation_id: UUID, *, max_recent_turns: int,
        cancellation: CancellationToken,
    ) -> ContextSnapshot | None:
        """Return a bounded immutable view, or None when no information exists.

        Recent history contains whole turns, excluding summary coverage. Raises
        on cancellation or storage failure; callers may run L1 with empty context.
        """
        ...

    def append_turn(
        self, conversation_id: UUID, user: HistoryEntry, reply: HistoryEntry, *,
        expected_revision: int | None = None, cancellation: CancellationToken | None = None,
    ) -> int:
        """Atomically persist a user/reply pair and return the new revision.

        Require matching turn IDs and user/assistant roles. Preserve reply
        outcome and partial text. Exact retries return their original commit
        revision without a write; conflicting reuse of a turn ID is rejected.
        Check an optional expected revision against the latest locked state.
        Generation cancellation does not cancel diagnostic transcript commits.
        An optional storage token cancels before publication; cancellation after
        publication raises too, but the accepted state may already be durable.
        Retry by turn ID to resolve that uncertainty. Never infer memory from output.
        """
        ...


class MemoryAndCompactionStore(ContinuityStore, Protocol):
    """Reserved for later steps; the transcript adapter does not implement this."""

    def remember(
        self, conversation_id: UUID, memory: ContextRecord, *,
        expected_revision: int, cancellation: CancellationToken,
    ) -> int:
        """Commit an attributed memory explicitly, using revision compare-and-swap.

        Raise ContinuityConflict on a stale revision; cancellation before commit
        leaves state unchanged. Return the new revision only after durable commit.
        """
        ...

    def compact(
        self, conversation_id: UUID, summary: ContextRecord, *,
        covered_turn_ids: tuple[UUID, ...], expected_revision: int,
        cancellation: CancellationToken,
    ) -> ContextSnapshot:
        """Atomically replace the summary view, preserving the raw transcript.

        Coverage must be a complete history prefix including prior summary
        coverage. Keep the recent suffix verbatim, retain supporting provenance,
        and increment the revision. Reject stale revisions with ContinuityConflict.
        Cancellation before commit leaves the previous snapshot intact. Summary
        text is supplied explicitly; this port never invokes a model itself.
        """
        ...

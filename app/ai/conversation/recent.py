"""Bounded transient dialogue, never a persistent or extracted memory store."""

from uuid import UUID

from ..contracts import ContextSnapshot, HistoryEntry


class RecentDialogue:
    def __init__(self, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("Recent dialogue must allow at least one turn.")
        self.max_turns = max_turns
        self.conversation_id: UUID | None = None
        self.entries: tuple[HistoryEntry, ...] = ()

    def context(
        self, conversation_id: UUID, snapshot: ContextSnapshot | None,
    ) -> tuple[HistoryEntry, ...]:
        local = self.entries if self.conversation_id == conversation_id else ()
        if snapshot is None:
            return local
        # L3 alone selects persisted context. Even an empty snapshot overrides
        # local state, so warm engines and resumed engines see the same history.
        return snapshot.recent

    def record(
        self, conversation_id: UUID, history: tuple[HistoryEntry, ...],
        user: HistoryEntry, reply: HistoryEntry,
    ) -> None:
        entries = history + (user, reply)
        turn_ids = tuple(dict.fromkeys(entry.turn_id for entry in entries))[-self.max_turns:]
        retained = set(turn_ids)
        self.conversation_id = conversation_id
        self.entries = tuple(entry for entry in entries if entry.turn_id in retained)

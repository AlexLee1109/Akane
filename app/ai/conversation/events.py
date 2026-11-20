"""Immutable text events correlated by a fresh UUID for each turn."""

from dataclasses import dataclass, field
from typing import Literal, TypeAlias
from uuid import UUID

from ..provenance import SourceAttribution
from .decisions import DelegateDecision


@dataclass(frozen=True, slots=True)
class UserTextReceived:
    turn_id: UUID
    text: str
    kind: Literal["user_text_received"] = field(default="user_text_received", init=False)


@dataclass(frozen=True, slots=True)
class ReplyStarted:
    turn_id: UUID
    kind: Literal["reply_started"] = field(default="reply_started", init=False)
    owner: Literal["l1"] = field(default="l1", init=False)


@dataclass(frozen=True, slots=True)
class ReplyTextDelta:
    turn_id: UUID
    text: str
    sources: tuple[SourceAttribution, ...] = ()
    kind: Literal["reply_text_delta"] = field(default="reply_text_delta", init=False)
    owner: Literal["l1"] = field(default="l1", init=False)


@dataclass(frozen=True, slots=True)
class ReplyCompleted:
    turn_id: UUID
    kind: Literal["reply_completed"] = field(default="reply_completed", init=False)
    owner: Literal["l1"] = field(default="l1", init=False)


@dataclass(frozen=True, slots=True)
class ReplyCancelled:
    turn_id: UUID
    kind: Literal["reply_cancelled"] = field(default="reply_cancelled", init=False)
    owner: Literal["l1"] = field(default="l1", init=False)


@dataclass(frozen=True, slots=True)
class ReplyFailed:
    turn_id: UUID
    message: str
    kind: Literal["reply_failed"] = field(default="reply_failed", init=False)
    owner: Literal["l1"] = field(default="l1", init=False)


@dataclass(frozen=True, slots=True)
class ReplyDelegated:
    """Internal L1 control event; host consumes it, never conversational speech."""
    turn_id: UUID
    decision: DelegateDecision
    kind: Literal['reply_delegated'] = field(default='reply_delegated', init=False)
    owner: Literal['l1'] = field(default='l1', init=False)


InteractionEvent: TypeAlias = (
    ReplyStarted
    | ReplyTextDelta
    | ReplyCompleted
    | ReplyCancelled
    | ReplyFailed
    | ReplyDelegated
)

ConversationEvent: TypeAlias = UserTextReceived | InteractionEvent


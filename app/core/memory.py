"""Bounded conversation context plus the popup's structured user profile."""

from __future__ import annotations

import copy
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict

from app.core.config import (
    CHAT_HISTORY_CONTEXT_TOKENS,
    CONVERSATION_STALE_DAYS,
    MAX_CONVERSATIONS,
    MEMORY_PATH,
    POPUP_USER_PATH,
    SUMMARY_CONTEXT_TOKENS,
)
from app.core.persistence import atomic_write_json, read_json
from app.core.signal import TurnSignal, analyze_turn, topic_overlap
from app.core.utils import compact_text

MEMORY_SCHEMA_VERSION = 1
_SUMMARY_BATCH_TURNS = 4
_MAX_TURN_CHARS = 8_000
_MAX_PROFILE_ITEMS = 24
_REMEMBER_PATTERN = re.compile(
    r"\b(?:please\s+)?remember(?:\s+that)?\s+(?P<value>[^\n]{3,200})",
    re.IGNORECASE,
)
_PROFILE_NAME_PATTERN = re.compile(
    r"\b(?:my name is|i am called|i'm called|you can call me|call me)\s+"
    r"(?P<name>[A-Za-z][A-Za-z' -]{0,48}?)"
    r"(?=$|[.!?;,]|\s+(?:and|but)\s+I\b)",
    re.IGNORECASE,
)
_PROFILE_PREFERENCE_PATTERN = re.compile(
    r"\bI\s+(?:really\s+)?"
    r"(?P<verb>don't like|do not like|dislike|hate|like|love|prefer)\s+"
    r"(?P<value>[^\n.!?;]{1,160}?)"
    r"(?=$|[.!?;]|\s+(?:but|and)\s+I\b)",
    re.IGNORECASE,
)
_PROFILE_FAVORITE_PATTERN = re.compile(
    r"\bmy\s+favorite\s+(?P<subject>[^\n.!?]{1,48}?)\s+is\s+"
    r"(?P<value>[^\n.!?]{1,100})",
    re.IGNORECASE,
)
_PROFILE_FACT_PATTERNS = (
    ("lives in", re.compile(r"\bI\s+live\s+in\s+(?P<value>[^\n.!?;]{2,100})", re.I)),
    ("works as", re.compile(r"\bI\s+work\s+as\s+(?P<value>[^\n.!?;]{2,100})", re.I)),
    ("birthday", re.compile(r"\bmy\s+birthday\s+is\s+(?P<value>[^\n.!?;]{2,80})", re.I)),
)


def estimate_tokens(value: object) -> int:
    """Cheap conservative token estimate that never loads the model tokenizer."""

    text = str(value or "")
    if not text:
        return 0
    byte_estimate = (len(text.encode("utf-8")) + 2) // 3
    word_estimate = (len(text.split()) * 5 + 3) // 4
    return max(1, byte_estimate, word_estimate)


@dataclass(frozen=True, slots=True)
class ChatTurn:
    turn_id: str
    role: str
    content: str
    timestamp: float
    source: str

    @classmethod
    def from_dict(cls, payload: object) -> "ChatTurn | None":
        if not isinstance(payload, dict):
            return None
        role = "assistant" if payload.get("role") == "assistant" else "user"
        content = str(payload.get("content") or payload.get("text") or "").strip()
        if not content:
            return None
        try:
            timestamp = max(0.0, float(payload.get("timestamp") or 0.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        return cls(
            turn_id=compact_text(payload.get("turn_id"), 80) or uuid.uuid4().hex,
            role=role,
            content=content[:_MAX_TURN_CHARS],
            timestamp=timestamp,
            source=compact_text(payload.get("source"), 24) or "unknown",
        )

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ConversationRecord:
    conversation_id: str
    profile_id: str
    recent_turns: list[ChatTurn] = field(default_factory=list)
    summary_turns: list[ChatTurn] = field(default_factory=list)
    pending_summary_turns: list[ChatTurn] = field(default_factory=list)
    recent_topic: str = ""
    recent_intent: str = "casual"
    recent_user_tone: str = "neutral"
    current_task: str = ""
    correction: str = ""
    recent_events: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, key: str, payload: object) -> "ConversationRecord | None":
        if not isinstance(payload, dict):
            return None
        conversation_id = compact_text(payload.get("conversation_id") or key, 120)
        profile_id = compact_text(payload.get("profile_id"), 120)
        if not conversation_id or not profile_id:
            return None

        def turns(name: str) -> list[ChatTurn]:
            raw = payload.get(name)
            if not isinstance(raw, list):
                return []
            return [turn for item in raw if (turn := ChatTurn.from_dict(item)) is not None]

        try:
            updated_at = max(0.0, float(payload.get("updated_at") or 0.0))
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(
            conversation_id=conversation_id,
            profile_id=profile_id,
            recent_turns=turns("recent_turns"),
            summary_turns=turns("summary_turns"),
            pending_summary_turns=turns("pending_summary_turns")[-(_SUMMARY_BATCH_TURNS - 1) :],
            recent_topic=compact_text(payload.get("recent_topic"), 80),
            recent_intent=compact_text(payload.get("recent_intent"), 32) or "casual",
            recent_user_tone=compact_text(payload.get("recent_user_tone"), 32) or "neutral",
            current_task=compact_text(payload.get("current_task"), 100),
            correction=compact_text(payload.get("correction"), 120),
            recent_events=[
                compact_text(item, 48)
                for item in (payload.get("recent_events") or [])[-5:]
                if compact_text(item, 48)
            ],
            updated_at=updated_at,
        )

    def summary_text(self) -> str:
        turns = [*self.summary_turns, *self.pending_summary_turns]
        if not turns:
            return ""
        lines = [
            f"{'Akane' if turn.role == 'assistant' else 'User'}: {compact_text(turn.content, 180)}"
            for turn in turns
        ]
        return "Earlier conversation excerpts:\n" + "\n".join(lines)

    def public_state(self) -> dict[str, object]:
        turns = [
            {"role": turn.role, "text": turn.content, "timestamp": turn.timestamp}
            for turn in self.recent_turns
        ]
        users = [turn.content for turn in self.recent_turns if turn.role == "user"]
        assistants = [turn.content for turn in self.recent_turns if turn.role == "assistant"]
        return {
            "summary": self.summary_text(),
            "last_user_summary": compact_text(users[-1], 180) if users else "",
            "last_assistant_summary": compact_text(assistants[-1], 180) if assistants else "",
            "recent_turns": turns,
            "recent_user_messages": users[-4:],
            "recent_assistant_replies": assistants[-3:],
            "recent_events": list(self.recent_events),
            "recent_intent": self.recent_intent,
            "recent_user_tone": self.recent_user_tone,
            "recent_topic": self.recent_topic,
            "current_task": self.current_task,
            "correction": self.correction,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class MemoryContext:
    relationship: str
    rolling_summary: str
    recent_turns: tuple[ChatTurn, ...]


class _ProfileUpdates(TypedDict):
    name: str
    likes: list[str]
    dislikes: list[str]
    facts: list[str]


@dataclass(slots=True)
class PopupUserProfile:
    name: str = ""
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: object) -> "PopupUserProfile":
        if not isinstance(payload, dict):
            return cls()

        def items(key: str) -> list[str]:
            raw = payload.get(key)
            if not isinstance(raw, list):
                return []
            return [
                value
                for item in raw[-_MAX_PROFILE_ITEMS:]
                if (value := compact_text(item, 120))
            ]

        try:
            updated_at = max(0.0, float(payload.get("updated_at") or 0.0))
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(
            name=compact_text(payload.get("name"), 50),
            likes=items("likes"),
            dislikes=items("dislikes"),
            facts=items("facts"),
            updated_at=updated_at,
        )

    def prompt_text(self) -> str:
        lines: list[str] = []
        if self.name:
            lines.append(f"Name: {self.name}")
        if self.likes:
            lines.append("Likes: " + ", ".join(self.likes))
        if self.dislikes:
            lines.append("Dislikes: " + ", ".join(self.dislikes))
        if self.facts:
            lines.append("Other known facts: " + "; ".join(self.facts))
        return "Popup user profile:\n" + "\n".join(lines) if lines else ""


class MemoryStore:
    """Own recent turns and rolling excerpts for each conversation."""

    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._conversations: dict[str, ConversationRecord] = {}
        self._load()

    def preview_signal(
        self,
        conversation_id: str,
        user_text: str,
        *,
        code_context_requested: bool = False,
        code_context_attached: bool = False,
    ) -> TurnSignal:
        candidate = analyze_turn(
            user_text,
            code_context_requested=code_context_requested,
            code_context_attached=code_context_attached,
        )
        with self._lock:
            previous = self._conversations.get(_key(conversation_id, "local:conversation"))
            if previous is None:
                return candidate
            continuation = candidate.low_content or candidate.intent in {
                "gratitude",
                "praise",
                "teasing",
                "correction",
            }
            if previous.recent_topic and (
                continuation or topic_overlap(previous.recent_topic, candidate.topic) >= 0.5
            ):
                return candidate.with_context(
                    topic=previous.recent_topic,
                    task=candidate.task or previous.current_task,
                    confidence=max(candidate.topic_confidence, 0.62),
                )
        return candidate

    def build_context(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        display_name: str = "",
        include_memory: bool = True,
    ) -> MemoryContext:
        profile = _key(profile_id, "local:owner")
        conversation_key = _key(conversation_id, "popup:default")
        with self._lock:
            record = self._conversations.get(conversation_key)
            if record is not None and record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")
            recent = tuple(record.recent_turns) if record and include_memory else ()
            summary = record.summary_text() if record and include_memory else ""
            relationship = _relationship_context(
                display_name,
                bool(record and record.recent_turns and include_memory),
            )
            return MemoryContext(
                relationship=relationship,
                rolling_summary=summary,
                recent_turns=recent,
            )

    def commit_turn(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        source: str,
        user_text: str,
        assistant_text: str,
        signal: TurnSignal,
    ) -> None:
        profile = _key(profile_id, "local:owner")
        conversation_key = _key(conversation_id, "popup:default")
        now = time.time()
        user_turn = ChatTurn(uuid.uuid4().hex, "user", user_text[:_MAX_TURN_CHARS], now, source)
        assistant_turn = ChatTurn(
            uuid.uuid4().hex,
            "assistant",
            assistant_text[:_MAX_TURN_CHARS],
            now,
            source,
        )
        with self._lock:
            previous_conversations = copy.deepcopy(self._conversations)
            record = self._conversations.get(conversation_key)
            if record is None:
                record = ConversationRecord(conversation_key, profile)
                self._conversations[conversation_key] = record
            elif record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")

            record.recent_turns.extend((user_turn, assistant_turn))
            record.recent_topic = compact_text(signal.topic, 80)
            record.recent_intent = signal.intent
            record.recent_user_tone = signal.tone
            record.current_task = compact_text(signal.task, 100)
            if signal.correction:
                record.correction = compact_text(signal.correction, 120)
            if signal.trigger:
                record.recent_events.append(compact_text(signal.trigger, 48))
                record.recent_events = record.recent_events[-5:]
            record.updated_at = now
            self._trim_conversation(record)

            self._prune(now)
            try:
                self._persist()
            except Exception:
                self._conversations = previous_conversations
                raise

    def messages(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> list[dict[str, str]]:
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record and profile_id and record.profile_id != _key(profile_id, "local:owner"):
                return []
            return [turn.as_message() for turn in record.recent_turns] if record else []

    def public_conversation(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record and profile_id and record.profile_id != _key(profile_id, "local:owner"):
                return {}
            return record.public_state() if record else {}

    def clear_conversation(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> bool:
        with self._lock:
            key = _key(conversation_id, "popup:default")
            record = self._conversations.get(key)
            if record and profile_id and record.profile_id != _key(profile_id, "local:owner"):
                return False
            self._conversations.pop(key, None)
            self._persist()
            return record is not None

    def clear_profile(self, profile_id: str) -> None:
        profile = _key(profile_id, "local:owner")
        with self._lock:
            self._conversations = {
                key: value
                for key, value in self._conversations.items()
                if value.profile_id != profile
            }
            self._persist()

    def _trim_conversation(self, record: ConversationRecord) -> None:
        while (
            sum(estimate_tokens(turn.content) + 4 for turn in record.recent_turns)
            > CHAT_HISTORY_CONTEXT_TOKENS
            and len(record.recent_turns) > 2
        ):
            record.pending_summary_turns.append(record.recent_turns.pop(0))
            if len(record.pending_summary_turns) >= _SUMMARY_BATCH_TURNS:
                record.summary_turns.extend(record.pending_summary_turns)
                record.pending_summary_turns.clear()
                while (
                    sum(estimate_tokens(turn.content) + 4 for turn in record.summary_turns)
                    > SUMMARY_CONTEXT_TOKENS
                    and record.summary_turns
                ):
                    record.summary_turns.pop(0)

    def _prune(self, now: float) -> None:
        stale_before = now - CONVERSATION_STALE_DAYS * 86_400
        for key, record in list(self._conversations.items()):
            if record.updated_at and record.updated_at < stale_before:
                self._conversations.pop(key, None)
        if len(self._conversations) > MAX_CONVERSATIONS:
            ordered = sorted(self._conversations.items(), key=lambda item: item[1].updated_at)
            for key, _record in ordered[: len(self._conversations) - MAX_CONVERSATIONS]:
                self._conversations.pop(key, None)

    def _load(self) -> None:
        try:
            payload = read_json(self._path)
            if (
                not isinstance(payload, dict)
                or int(payload.get("schema_version", 0)) != MEMORY_SCHEMA_VERSION
            ):
                raise ValueError("unsupported schema")
            conversations = payload.get("conversations")
            if not isinstance(conversations, dict):
                raise ValueError("invalid memory document")
            self._conversations = {
                record.conversation_id: record
                for key, value in conversations.items()
                if (record := ConversationRecord.from_dict(str(key), value)) is not None
            }
            self._prune(time.time())
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            print(f"[Akane:memory] ignored corrupt memory file ({type(exc).__name__})", flush=True)
            self._conversations = {}

    def _persist(self) -> None:
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "conversations": {
                key: asdict(value) for key, value in self._conversations.items()
            },
        }
        atomic_write_json(self._path, payload)


def _relationship_context(display_name: str, has_history: bool) -> str:
    parts: list[str] = []
    name = compact_text(display_name, 60)
    if name:
        parts.append(f"This person is displayed as {name}; use the name only when natural")
    if has_history:
        parts.append(
            "this is an ongoing relationship, so continuity may be acknowledged "
            "without forced familiarity"
        )
    return "Relationship: " + "; ".join(parts) + "." if parts else ""


def _key(value: object, default: str) -> str:
    return compact_text(value, 120) or default


class PopupUserStore:
    """Learns durable facts from successful popup turns."""

    def __init__(self, path: Path = POPUP_USER_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._profile = PopupUserProfile()
        self._load()

    def snapshot(self) -> PopupUserProfile:
        with self._lock:
            return copy.deepcopy(self._profile)

    def prompt_text(self) -> str:
        with self._lock:
            return self._profile.prompt_text()

    def public_profile(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._profile)

    def commit(self, user_text: str) -> None:
        learned = _extract_profile_updates(user_text)
        if not any(learned.values()):
            return
        with self._lock:
            previous = copy.deepcopy(self._profile)
            if learned["name"]:
                self._profile.name = learned["name"]
            for value in learned["likes"]:
                _remove_profile_value(self._profile.dislikes, value)
                _append_profile_value(self._profile.likes, value)
            for value in learned["dislikes"]:
                _remove_profile_value(self._profile.likes, value)
                _append_profile_value(self._profile.dislikes, value)
            for value in learned["facts"]:
                _replace_profile_fact(self._profile.facts, value)
            self._profile.updated_at = time.time()
            try:
                self._persist()
            except Exception:
                self._profile = previous
                raise

    def restore(self, profile: PopupUserProfile) -> None:
        with self._lock:
            self._profile = copy.deepcopy(profile)
            self._persist()

    def clear(self) -> None:
        with self._lock:
            self._profile = PopupUserProfile()
            self._persist()

    def _load(self) -> None:
        try:
            payload = read_json(self._path)
            if (
                not isinstance(payload, dict)
                or int(payload.get("schema_version", 0)) != MEMORY_SCHEMA_VERSION
            ):
                raise ValueError("unsupported schema")
            self._profile = PopupUserProfile.from_dict(payload.get("user"))
        except FileNotFoundError:
            return
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"[Akane:popup-user] ignored corrupt profile ({type(exc).__name__})",
                flush=True,
            )

    def _persist(self) -> None:
        atomic_write_json(
            self._path,
            {"schema_version": MEMORY_SCHEMA_VERSION, "user": asdict(self._profile)},
        )


def _extract_profile_updates(text: str) -> _ProfileUpdates:
    value = compact_text(text, 600)
    result: _ProfileUpdates = {
        "name": "",
        "likes": [],
        "dislikes": [],
        "facts": [],
    }
    if not value or any(word in value.lower() for word in ("hypothetically", "just kidding")):
        return result
    if match := _PROFILE_NAME_PATTERN.search(value):
        name = compact_text(match.group("name"), 50).strip(" ,;:-")
        if 1 <= len(name.split()) <= 4:
            result["name"] = name
    if match := _REMEMBER_PATTERN.search(value):
        fact = compact_text(match.group("value"), 120).strip(" ,.;:-")
        if fact:
            result["facts"].append(fact)
    for match in _PROFILE_PREFERENCE_PATTERN.finditer(value):
        preference = compact_text(match.group("value"), 120).strip(" ,;:-")
        if not preference:
            continue
        verb = " ".join(match.group("verb").lower().split())
        key = (
            "dislikes"
            if verb in {"don't like", "do not like", "dislike", "hate"}
            else "likes"
        )
        result[key].append(preference)
    for match in _PROFILE_FAVORITE_PATTERN.finditer(value):
        subject = compact_text(match.group("subject"), 48).strip(" ,;:-")
        favorite = compact_text(match.group("value"), 100).strip(" ,;:-")
        if subject and favorite:
            result["facts"].append(f"favorite {subject}: {favorite}")
    for label, pattern in _PROFILE_FACT_PATTERNS:
        if match := pattern.search(value):
            fact = compact_text(match.group("value"), 100).strip(" ,;:-")
            if fact:
                result["facts"].append(f"{label}: {fact}")
    return result


def _normalized_profile_value(value: str) -> str:
    return " ".join(value.casefold().split())


def _remove_profile_value(items: list[str], value: str) -> None:
    target = _normalized_profile_value(value)
    items[:] = [item for item in items if _normalized_profile_value(item) != target]


def _append_profile_value(items: list[str], value: str) -> None:
    _remove_profile_value(items, value)
    items.append(value)
    del items[:-_MAX_PROFILE_ITEMS]


def _replace_profile_fact(items: list[str], value: str) -> None:
    label = value.partition(":")[0].casefold()
    items[:] = [item for item in items if item.partition(":")[0].casefold() != label]
    items.append(value)
    del items[:-_MAX_PROFILE_ITEMS]


_MEMORY: MemoryStore | None = None
_MEMORY_LOCK = threading.Lock()
_POPUP_USER: PopupUserStore | None = None
_POPUP_USER_LOCK = threading.Lock()


def get_memory_store() -> MemoryStore:
    global _MEMORY
    if _MEMORY is None:
        with _MEMORY_LOCK:
            if _MEMORY is None:
                _MEMORY = MemoryStore()
    return _MEMORY


def get_popup_user_store() -> PopupUserStore:
    global _POPUP_USER
    if _POPUP_USER is None:
        with _POPUP_USER_LOCK:
            if _POPUP_USER is None:
                _POPUP_USER = PopupUserStore()
    return _POPUP_USER


def get_session_memory() -> MemoryStore:
    """Return the shared memory owner; retained as the public runtime getter."""

    return get_memory_store()

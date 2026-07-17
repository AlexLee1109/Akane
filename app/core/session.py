"""Normalized chat input, generation ownership, and successful-turn commits."""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, replace

from app.core.config import (
    GENERATION_QUEUE_TIMEOUT_SECONDS,
    MAX_INPUT_CHARS,
    MAX_PENDING_GENERATIONS,
    MAX_TOKENS,
    PROMPT_DEBUG,
)
from app.core.life import format_life_state
from app.core.memory import (
    InternalTurnResult,
    MemoryContext,
    akane_preference_answer,
    established_akane_preference,
    format_relevant_memories,
    get_internal_state_store,
    get_memory_store,
    preference_domain,
    preference_update_requested,
)
from app.core.prompt import PromptContext, PromptPlan, TurnGuidance, build_prompt_plan
from app.core.signal import (
    SubtextAppraisal,
    TurnSignal,
    analyze_turn,
    appraise_subtext,
)
from app.core.utils import compact_text
from app.integrations.vscode_context import CodeContext, code_context_for_message

_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_COMMIT_LOCK = threading.RLock()


class GenerationBusyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChatInput:
    profile_id: str
    conversation_id: str
    text: str
    source: str
    timestamp: float
    display_name: str = ""
    reply_context: str = ""
    group_conversation: bool = False
    autonomous: bool = False


@dataclass(slots=True)
class GenerationHandle:
    generation_id: str
    conversation_id: str
    profile_id: str
    cancellation: threading.Event
    queue_deadline: float

    def raise_if_cancelled(self) -> None:
        if self.cancellation.is_set():
            raise GenerationCancelled("Generation was cancelled.")


class GenerationScheduler:
    """Bounds model waiters and permits only one active turn per conversation."""

    def __init__(self) -> None:
        self._capacity = threading.BoundedSemaphore(MAX_PENDING_GENERATIONS + 1)
        self._lock = threading.RLock()
        self._active: dict[str, GenerationHandle] = {}

    def begin(
        self,
        conversation_id: str,
        profile_id: str,
        *,
        skip_if_busy: bool = False,
    ) -> GenerationHandle:
        with self._lock:
            if skip_if_busy and self._active:
                raise GenerationBusyError("Akane is busy with another reply.")
            if conversation_id in self._active:
                raise GenerationBusyError("This conversation already has a reply in progress.")
            if any(handle.profile_id == profile_id for handle in self._active.values()):
                raise GenerationBusyError("This profile already has a reply in progress.")
        if not self._capacity.acquire(blocking=False):
            raise GenerationQueueFullError("Akane is busy; the generation queue is full.")
        handle = GenerationHandle(
            generation_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            profile_id=profile_id,
            cancellation=threading.Event(),
            queue_deadline=time.monotonic() + GENERATION_QUEUE_TIMEOUT_SECONDS,
        )
        with self._lock:
            current = self._active.get(conversation_id)
            profile_active = any(
                active.profile_id == profile_id for active in self._active.values()
            )
            if current is not None or profile_active:
                self._capacity.release()
                raise GenerationBusyError("This conversation or profile already has a reply in progress.")
            self._active[conversation_id] = handle
        return handle

    def finish(self, handle: GenerationHandle) -> None:
        with self._lock:
            if self._active.get(handle.conversation_id) is not handle:
                return
            self._active.pop(handle.conversation_id, None)
        self._capacity.release()

    def cancel(self, conversation_id: str, profile_id: str | None = None) -> bool:
        with self._lock:
            handle = self._active.get(conversation_id)
            if handle is None or (profile_id is not None and handle.profile_id != profile_id):
                return False
            handle.cancellation.set()
            return True

    def cancel_all(self) -> None:
        with self._lock:
            for handle in self._active.values():
                handle.cancellation.set()

    def cancel_profile(self, profile_id: str) -> None:
        with self._lock:
            for handle in self._active.values():
                if handle.profile_id == profile_id:
                    handle.cancellation.set()

    def active_generation_id(self, conversation_id: str) -> str:
        with self._lock:
            handle = self._active.get(conversation_id)
            return handle.generation_id if handle else ""


@dataclass(frozen=True, slots=True)
class TurnPreparation:
    chat_input: ChatInput
    prompt_plan: PromptPlan
    signal: TurnSignal
    memory_context: MemoryContext
    internal_turn: InternalTurnResult
    subtext: SubtextAppraisal | None
    handle: GenerationHandle
    max_tokens: int
    started_at: float
    prompt_seconds: float = 0.0
    preprocess_seconds: float = 0.0
    memory_seconds: float = 0.0
    subtext_seconds: float = 0.0
    code_context_attached: bool = False
    preference_anchor: str = ""

    @property
    def session_id(self) -> str:
        return self.chat_input.conversation_id

    @property
    def generation_id(self) -> str:
        return self.handle.generation_id

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self.prompt_plan.messages)


def normalize_chat_input(
    *,
    text: object,
    profile_id: object = "local:owner",
    conversation_id: object = "popup:default",
    source: object = "popup",
    timestamp: object = 0.0,
    display_name: object = "",
    reply_context: object = "",
    group_conversation: object = False,
    autonomous: object = False,
) -> ChatInput:
    message = str(text or "").strip()
    if not message:
        raise ValueError("Message is empty.")
    if len(message) > MAX_INPUT_CHARS:
        raise ValueError(f"Message exceeds the {MAX_INPUT_CHARS}-character limit.")
    try:
        created_at = float(timestamp or time.time())
    except (TypeError, ValueError):
        created_at = time.time()
    normalized_source = compact_text(source, 24).lower() or "popup"
    if normalized_source not in {"popup", "discord", "web"}:
        normalized_source = "web"
    return ChatInput(
        profile_id=compact_text(profile_id, 120) or "local:owner",
        conversation_id=compact_text(conversation_id, 120) or "popup:default",
        text=message,
        source=normalized_source,
        timestamp=created_at,
        display_name=compact_text(display_name, 60),
        reply_context=compact_text(reply_context, 600),
        group_conversation=bool(group_conversation),
        autonomous=bool(autonomous),
    )


def prepare_turn(
    chat_input: ChatInput | str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
) -> TurnPreparation:
    started_at = time.perf_counter()
    chat = (
        normalize_chat_input(text=chat_input, conversation_id=session_id or "popup:default")
        if isinstance(chat_input, str)
        else chat_input
    )
    handle = _SCHEDULER.begin(
        chat.conversation_id,
        chat.profile_id,
        skip_if_busy=skip_if_busy,
    )
    try:
        code_context = (
            code_context_for_message(chat.text)
            if chat.source in {"popup", "discord"} and not chat.autonomous
            else CodeContext(requested=False, connected=False)
        )
        memory_started_at = time.perf_counter()
        memory = get_memory_store()
        memory_context = memory.build_context(
            chat.profile_id,
            chat.conversation_id,
            display_name=chat.display_name,
            query=chat.text,
            include_memory=not skip_memory and not chat.autonomous,
        )
        long_term = get_internal_state_store()
        internal_turn = long_term.preview_turn(
            chat.profile_id,
            chat.text,
            now=chat.timestamp,
            include_memory=not skip_memory,
            code_context_requested=code_context.requested,
            code_context_attached=bool(code_context.prompt_text),
            autonomous=chat.autonomous,
        )
        memory_seconds = time.perf_counter() - memory_started_at
        signal = internal_turn.signal
        working = internal_turn.state.working
        subtext_started_at = time.perf_counter()
        subtext = None if chat.autonomous else appraise_subtext(
            chat.text,
            signal,
            memory_context.recent_turns,
            current_topic=working.current_topic,
            current_task=working.current_task,
            unresolved_problem=working.unresolved_problem,
            now=chat.timestamp,
        )
        subtext_seconds = time.perf_counter() - subtext_started_at
        preference_memory = (
            established_akane_preference(
                internal_turn.state.memories,
                chat.text,
                now=chat.timestamp,
            )
            if not skip_memory and signal.identity_attribute == "preferences"
            else None
        )
        preference_change_allowed = (
            signal.identity_attribute == "preferences"
            and preference_update_requested(chat.text)
        )
        preference_anchor = (
            "" if preference_change_allowed else akane_preference_answer(preference_memory)
        )
        if (
            not preference_anchor
            and not preference_change_allowed
            and signal.identity_attribute == "preferences"
        ):
            preference_anchor = _recent_preference_answer(
                chat.text,
                memory_context.recent_turns,
            )
        selected_memories = list(internal_turn.recalled_memories)
        if preference_memory is not None and all(
            memory.id != preference_memory.id for memory in selected_memories
        ):
            selected_memories.append(preference_memory)
        relevant_memories = tuple(
            memory
            for memory in selected_memories
            if preference_memory is None or memory.id != preference_memory.id
        )
        memory_context = replace(
            memory_context,
            memory_ids=tuple(memory.id for memory in selected_memories),
            memory_contents=tuple(memory.content for memory in selected_memories),
        )
        turn_guidance = _turn_guidance(signal, chat)
        editor_context = code_context.prompt_text
        if code_context.requested and not code_context.connected:
            editor_context = (
                "The requested editor context is unavailable. Do not claim to have inspected a file."
            )
        prompt_started_at = time.perf_counter()
        prompt_plan = build_prompt_plan(
            chat.text,
            PromptContext(
                relationship=memory_context.relationship,
                preference_continuity=(
                    (
                        f"{preference_memory.content}\nThe user explicitly asked Akane to "
                        "reconsider this preference, so a deliberate changed choice is allowed."
                        if preference_change_allowed
                        else f"{preference_memory.content}\nPreserve the named choice. Variation "
                        "may change wording, sentence structure, reason, or emphasis, but must "
                        "not replace it or introduce another favorite or interest."
                    )
                    if preference_memory is not None
                    else (
                        "Established preference from recent dialogue: "
                        f"{preference_anchor}\nPreserve the named choice; vary presentation, "
                        "not the underlying answer."
                        if preference_anchor
                        else ""
                    )
                ),
                relevant_memories=format_relevant_memories(relevant_memories),
                earlier_dialogue=memory_context.earlier_dialogue,
                recent_turns=memory_context.recent_turns,
                subtext=subtext,
                turn_guidance=turn_guidance,
                date_time=date_time_line() if _time_context_relevant(chat.text) else "",
                reply_context=chat.reply_context,
                external_context=editor_context,
                internal_context=internal_turn.prompt_context,
                life_context=format_life_state(internal_turn.state.life),
            ),
        )
        prompt_seconds = time.perf_counter() - prompt_started_at
        system_text = prompt_plan.messages[0]["content"]
        included_ids = tuple(
            memory_id
            for memory_id, content in zip(
                memory_context.memory_ids,
                memory_context.memory_contents,
            )
            if content in system_text
        )
        memory_context = replace(memory_context, memory_ids=included_ids)
        if PROMPT_DEBUG:
            metadata = prompt_plan.debug_metadata()
            print(f"[Akane:prompt] {metadata}", flush=True)
        handle.raise_if_cancelled()
        return TurnPreparation(
            chat_input=chat,
            prompt_plan=prompt_plan,
            signal=signal,
            memory_context=memory_context,
            internal_turn=internal_turn,
            subtext=subtext,
            handle=handle,
            max_tokens=MAX_TOKENS,
            started_at=started_at,
            prompt_seconds=prompt_seconds,
            preprocess_seconds=time.perf_counter() - started_at,
            memory_seconds=memory_seconds,
            subtext_seconds=subtext_seconds,
            code_context_attached=bool(code_context.prompt_text),
            preference_anchor=preference_anchor,
        )
    except Exception:
        _SCHEDULER.finish(handle)
        raise


def commit_turn(prepared: TurnPreparation, reply: str) -> None:
    prepared.handle.raise_if_cancelled()
    with _COMMIT_LOCK:
        prepared.handle.raise_if_cancelled()
        long_term = get_internal_state_store()
        previous_state = long_term.commit_turn(
            prepared.chat_input.profile_id,
            prepared.internal_turn,
            assistant_text=reply,
            used_memory_ids=prepared.memory_context.memory_ids,
            now=prepared.chat_input.timestamp,
        )
        if prepared.chat_input.autonomous:
            return
        try:
            prepared.handle.raise_if_cancelled()
            get_memory_store().commit_turn(
                profile_id=prepared.chat_input.profile_id,
                conversation_id=prepared.chat_input.conversation_id,
                source=prepared.chat_input.source,
                user_text=prepared.chat_input.text,
                assistant_text=reply,
                signal=prepared.signal,
            )
        except Exception:
            long_term.restore_internal_state(prepared.chat_input.profile_id, previous_state)
            raise


def finish_turn(prepared: TurnPreparation) -> None:
    _SCHEDULER.finish(prepared.handle)


def cancel_generation(conversation_id: str, profile_id: str | None = None) -> bool:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) if profile_id is not None else None
    return _SCHEDULER.cancel(conversation, profile)


def cancel_all_generations() -> None:
    _SCHEDULER.cancel_all()


def reset_conversation(conversation_id: str, profile_id: str) -> None:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) or "local:owner"
    with _COMMIT_LOCK:
        _SCHEDULER.cancel(conversation, profile)
        get_memory_store().clear_conversation(conversation, profile)


def forget_profile(profile_id: str) -> None:
    profile = compact_text(profile_id, 120) or "local:owner"
    with _COMMIT_LOCK:
        _SCHEDULER.cancel_profile(profile)
        get_memory_store().clear_profile(profile)
        get_internal_state_store().clear(profile)


def session_state_snapshot(
    conversation_id: str | None = None,
    profile_id: str | None = None,
) -> dict[str, object]:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) or "local:owner"
    return {
        "akane": get_internal_state_store().public_internal_state(profile),
        "memory": get_memory_store().public_conversation(conversation, profile),
        "popup_user": get_internal_state_store().public_profile(profile),
        "active_generation_id": _SCHEDULER.active_generation_id(conversation),
    }


def date_time_line() -> str:
    timestamp = time.time()
    now = time.localtime(timestamp)
    hour = time.strftime("%I", now).lstrip("0") or "0"
    zone = time.strftime("%Z", now) or "local time"
    return (
        f"Current local date and time: {time.strftime('%A, %B', now)} "
        f"{now.tm_mday}, {now.tm_year} at {hour}:{time.strftime('%M %p', now)} {zone}."
    )


def timing_enabled() -> bool:
    return _TIMING_ENABLED


def _turn_guidance(signal: TurnSignal, chat: ChatInput) -> TurnGuidance | None:
    repetition_level = 0
    if signal.repetition:
        if signal.repetition_count >= 5:
            repetition_level = 3
        elif signal.repetition_count >= 3:
            repetition_level = 2
        else:
            repetition_level = 1
    group_conversation = (
        chat.source == "discord" and chat.group_conversation and not chat.autonomous
    )
    if not (
        chat.autonomous
        or signal.identity_attribute
        or signal.current_activity
        or group_conversation
        or signal.criticism
        or signal.correction_requested
        or repetition_level
    ):
        return None
    return TurnGuidance(
        autonomous=chat.autonomous,
        identity_attribute=signal.identity_attribute,
        current_activity=signal.current_activity,
        group_conversation=group_conversation,
        criticism=signal.criticism,
        correction_requested=signal.correction_requested,
        repetition_level=repetition_level,
    )


def _recent_preference_answer(text: str, recent_turns: tuple[object, ...]) -> str:
    wanted_domain = preference_domain(text)
    for index in range(len(recent_turns) - 1, 0, -1):
        assistant = recent_turns[index]
        user = recent_turns[index - 1]
        if getattr(assistant, "role", "") != "assistant" or getattr(user, "role", "") != "user":
            continue
        user_text = str(getattr(user, "content", "") or "")
        if analyze_turn(user_text).identity_attribute != "preferences":
            continue
        if wanted_domain != "general" and preference_domain(user_text) != wanted_domain:
            continue
        return compact_text(getattr(assistant, "content", ""), 190)
    return ""


def _time_context_relevant(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        marker in lower
        for marker in (
            "today",
            "tonight",
            "tomorrow",
            "yesterday",
            "what time",
            "current time",
            "what day",
            "the date",
            "this morning",
            "this afternoon",
            "this evening",
        )
    )


_SCHEDULER = GenerationScheduler()

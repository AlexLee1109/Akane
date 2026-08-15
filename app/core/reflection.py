"""Post-turn semantic extraction followed by deterministic state validation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace

from app.core.config import SETTINGS
from app.core.context import ContextBuilder
from app.core.mind import form_self_item, revise_self_item
from app.core.inference import InferencePreempted, InferenceRuntime, InferenceTiming
from app.core.state import (
    MEMORY_KINDS,
    MEMORY_SUBJECTS,
    Memory,
    MemoryChange,
    MoodChange,
    RelationshipChange,
    SelfChange,
    StateChangeProposal,
    clamp,
)
from app.core.prompt import build_reflection_prompt
from app.core.store import Store
from app.core.utils import log_performance, log_timing, relevance, text_key

_UNSUPPORTED_EXTERNAL_EVENTS = {
    "ate", "drank", "visited", "traveled", "watched", "played", "touched", "felt",
    "smelled", "tasted", "heard", "saw", "searched", "browsed", "read",
}
_RUNTIME_CAPABILITY_TERMS = {
    "internet", "browse", "microphone", "filesystem", "command", "commands",
    "live2d", "screen", "camera",
}


def _bounded_dialogue(text: str, limit: int) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    marker = "\n...[bounded dialogue omitted]...\n"
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    return value[:head] + marker + value[-(remaining - head):]


def remember(
    profile_id: str,
    *,
    subject: str,
    kind: str,
    text: str,
    importance: float,
    confidence: float,
    source_turn_ids: tuple[str, ...],
    now: float | None = None,
) -> MemoryChange:
    if subject not in MEMORY_SUBJECTS:
        raise ValueError(f"Unknown memory subject: {subject}")
    if kind not in MEMORY_KINDS:
        raise ValueError(f"Unknown memory kind: {kind}")
    timestamp = time.time() if now is None else float(now)
    memory = Memory(
        id=f"memory_{uuid.uuid4().hex}",
        profile_id=profile_id,
        subject=subject,
        kind=kind,
        text=text.strip(),
        importance=clamp(importance, 0, 1),
        confidence=clamp(confidence, 0, 1),
        created_at=timestamp,
        updated_at=timestamp,
        source_turn_ids=source_turn_ids,
    )
    return MemoryChange("add", memory=memory)


def _evidence(value: object, source: str) -> bool:
    excerpt = " ".join(str(value or "").split()).casefold()
    return bool(excerpt) and excerpt in " ".join(source.split()).casefold()


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strings(value: object, limit: int = 8) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip()[:300] for item in value if str(item).strip())[:limit]


def parse_reflection_json(output: object) -> dict[str, object]:
    text = str(output or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except (RecursionError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload if set(payload) <= {"memories", "self", "mood", "relationship"} else {}


def validate_reflection(
    payload: dict[str, object],
    *,
    context,
    user_text: str,
    assistant_text: str,
    user_turn_id: str,
    assistant_turn_id: str,
    now: float | None = None,
) -> StateChangeProposal:
    timestamp = time.time() if now is None else float(now)
    profile_id = context.state.profile_id
    memories = []
    self_changes: list[SelfChange] = []
    rejected: list[str] = []

    raw_memories = payload.get("memories", [])
    if isinstance(raw_memories, list):
        for index, raw in enumerate(raw_memories[:4]):
            if not isinstance(raw, dict):
                rejected.append(f"memory[{index}]:schema")
                continue
            subject = str(raw.get("subject") or "")
            kind = str(raw.get("kind") or "")
            text = " ".join(str(raw.get("text") or "").split())[:1000]
            evidence = str(raw.get("evidence") or "")
            source = user_text if subject == "user" else assistant_text
            if not _evidence(evidence, source) or relevance(evidence, text) < 0.04:
                rejected.append(f"memory[{index}]:ungrounded")
                continue
            terms = set(text_key(text).split())
            if subject in {"akane", "shared"} and kind == "fact" and terms & _RUNTIME_CAPABILITY_TERMS:
                rejected.append(f"memory[{index}]:runtime-owned")
                continue
            if (
                subject == "user"
                and terms & _UNSUPPORTED_EXTERNAL_EVENTS
                and terms & {"we", "us", "akane", "together"}
            ):
                rejected.append(f"memory[{index}]:ownership")
                continue
            if subject in {"akane", "shared"} and kind in {"event", "shared_experience"} and terms & _UNSUPPORTED_EXTERNAL_EVENTS:
                rejected.append(f"memory[{index}]:unsupported-event")
                continue
            try:
                memories.append(remember(
                    profile_id,
                    subject=subject,
                    kind=kind,
                    text=text,
                    importance=_number(raw.get("importance"), 0.4),
                    confidence=_number(raw.get("confidence"), 0.6),
                    source_turn_ids=(user_turn_id, assistant_turn_id),
                    now=timestamp,
                ))
            except ValueError:
                rejected.append(f"memory[{index}]:schema")

    current_by_id = {item.id: item for item in context.state.self_items}
    raw_self = payload.get("self", [])
    if isinstance(raw_self, list):
        for index, raw in enumerate(raw_self[:4]):
            if not isinstance(raw, dict):
                rejected.append(f"self[{index}]:schema")
                continue
            evidence = str(raw.get("evidence") or "")
            # Akane's Self can only be evidenced by what Akane chose to express.
            if not _evidence(evidence, assistant_text):
                rejected.append(f"self[{index}]:not-akane-owned")
                continue
            # Quoting the user's assignment verbatim does not make it Akane's preference.
            if _evidence(evidence, user_text):
                rejected.append(f"self[{index}]:user-assignment")
                continue
            topic = " ".join(str(raw.get("topic") or "").split())[:160]
            value = " ".join(str(raw.get("value") or "").split())[:500]
            reason = " ".join(str(raw.get("reason") or "").split())[:500]
            if relevance(evidence, f"{topic} {value} {reason}") < 0.03:
                rejected.append(f"self[{index}]:ungrounded")
                continue
            action = str(raw.get("action") or "")
            target_id = str(raw.get("target_id") or "")
            try:
                if action == "form":
                    self_changes.append(form_self_item(
                        profile_id,
                        kind=str(raw.get("kind") or ""),
                        topic=topic,
                        value=value,
                        strength=max(-0.6, min(0.6, _number(raw.get("strength"), 0.25))),
                        confidence=min(0.45, max(0.1, _number(raw.get("confidence"), 0.3))),
                        reason=reason,
                        source_ids=(assistant_turn_id,),
                        now=timestamp,
                    ))
                elif action == "retire" and target_id in current_by_id:
                    self_changes.append(SelfChange("retire", target_id=target_id))
                elif action in {"reinforce", "weaken", "revise", "complete", "abandon"} and target_id in current_by_id:
                    current_item = current_by_id[target_id]
                    proposed_strength = _number(raw.get("strength"), current_item.strength)
                    proposed_confidence = _number(raw.get("confidence"), current_item.confidence)
                    strength_step = 0.15 if action == "reinforce" else 0.2 if action == "weaken" else 0.75
                    bounded_strength = max(
                        current_item.strength - strength_step,
                        min(current_item.strength + strength_step, proposed_strength),
                    )
                    bounded_confidence = min(current_item.confidence + 0.15, proposed_confidence)
                    self_changes.append(revise_self_item(
                        current_item,
                        action=action,
                        value=value,
                        strength=bounded_strength,
                        confidence=bounded_confidence,
                        reason=reason,
                        source_ids=(assistant_turn_id,),
                        now=timestamp,
                    ))
                else:
                    rejected.append(f"self[{index}]:lifecycle")
            except ValueError:
                rejected.append(f"self[{index}]:schema")

    mood = None
    raw_mood = payload.get("mood")
    if isinstance(raw_mood, dict):
        evidence = str(raw_mood.get("evidence") or "")
        if _evidence(evidence, f"{user_text}\n{assistant_text}"):
            mood = MoodChange(
                valence_delta=max(-0.25, min(0.25, _number(raw_mood.get("valence_delta")))),
                energy_delta=max(-0.25, min(0.25, _number(raw_mood.get("energy_delta")))),
                emotion=" ".join(str(raw_mood.get("emotion") or "").split())[:80],
                cause=" ".join(str(raw_mood.get("cause") or "").split())[:500],
            )
        else:
            rejected.append("mood:ungrounded")

    relationship = None
    raw_relationship = payload.get("relationship")
    if isinstance(raw_relationship, dict):
        evidence = str(raw_relationship.get("evidence") or "")
        if _evidence(evidence, f"{user_text}\n{assistant_text}"):
            relationship = RelationshipChange(
                familiarity_delta=max(-0.02, min(0.03, _number(raw_relationship.get("familiarity_delta")))),
                trust_delta=max(-0.05, min(0.05, _number(raw_relationship.get("trust_delta")))),
                closeness_delta=max(-0.04, min(0.04, _number(raw_relationship.get("closeness_delta")))),
                add_notes=_strings(raw_relationship.get("add_notes"), 2),
                resolve_notes=_strings(raw_relationship.get("resolve_notes"), 2),
                add_unresolved=_strings(raw_relationship.get("add_unresolved"), 2),
            )
        else:
            rejected.append("relationship:ungrounded")

    return StateChangeProposal(
        profile_id=profile_id,
        memories=tuple(memories),
        self_items=tuple(self_changes),
        mood=mood,
        relationship=relationship,
        origin="reflection",
        rejected=tuple(rejected),
    )


class ReflectionEngine:
    def __init__(self, store: Store, runtime: InferenceRuntime | None = None):
        self.store = store
        self.runtime = runtime or InferenceRuntime.get_instance()
        self.context_builder = ContextBuilder(store)

    def run_job(self, job: dict[str, object]) -> StateChangeProposal:
        started_at = time.perf_counter()
        claimed_at = float(job.get("claimed_monotonic_at") or started_at)
        context_started_at = started_at
        context_finished_at = started_at
        wait_started_at = started_at
        reservation_acquired_at = started_at
        inference_finished_at = started_at
        parse_started_at = started_at
        parsed_at = started_at
        committed_at = started_at
        timing = InferenceTiming(started_at)
        applied_count = 0
        rejected_count = 0
        profile_id = str(job["profile_id"])
        try:
            if not self.store.profile_exists(profile_id):
                raise RuntimeError("The reflection profile no longer exists.")
            conversation_id = str(job["conversation_id"])
            user_text = str(job["user_text"])
            assistant_text = str(job["assistant_text"])
            full_conversation_text = str(job["conversation_text"])
            conversation_text = _bounded_dialogue(
                full_conversation_text, SETTINGS.reflection_input_chars,
            )
            context_started_at = time.perf_counter()
            context = self.context_builder.build(
                profile_id=profile_id,
                conversation_id=conversation_id,
                message=f"{user_text}\n{assistant_text}",
                allow_tool_context=False,
            )
            context_finished_at = time.perf_counter()
            messages = build_reflection_prompt(context, conversation_text)
            wait_started_at = time.perf_counter()
            with self.runtime.reserve(priority="reflection") as reservation:
                reservation_acquired_at = time.perf_counter()
                while True:
                    prompt_tokens, prompt_method = self.runtime.count_prompt_tokens(messages, reservation)
                    if prompt_tokens + SETTINGS.reflection_tokens <= SETTINGS.llama_context_window:
                        timing.prompt_tokens = prompt_tokens
                        timing.prompt_token_method = prompt_method
                        break
                    if len(conversation_text) <= 512:
                        raise RuntimeError("The bounded reflection prompt does not fit the model context window.")
                    conversation_text = _bounded_dialogue(
                        full_conversation_text, max(512, int(len(conversation_text) * 0.7)),
                    )
                    messages = build_reflection_prompt(context, conversation_text)
                output = self.runtime.complete_messages(
                    messages,
                    max_tokens=SETTINGS.reflection_tokens,
                    reservation=reservation,
                    timing=timing,
                    temperature=0.15,
                    call_kind="reflection",
                )
            inference_finished_at = time.perf_counter()
            if reservation.preemption.is_set():
                raise InferencePreempted("Reflection yielded before validation.")
            parse_started_at = inference_finished_at
            proposal = validate_reflection(
                parse_reflection_json(output),
                context=context,
                user_text=user_text,
                assistant_text=assistant_text,
                user_turn_id=str(job["first_turn_id"]),
                assistant_turn_id=str(job["latest_turn_id"]),
            )
            proposal = replace(
                proposal,
                reflection_job_id=str(job["id"]),
                reflected_through_turn_id=str(job["latest_turn_id"]),
                reflected_turn_count=int(job["turn_count"]),
            )
            parsed_at = time.perf_counter()
            if reservation.preemption.is_set():
                raise InferencePreempted("Reflection yielded before persistence.")
            commit = self.store.commit(proposal)
            applied_count = len(commit.applied)
            rejected_count = len(commit.rejected)
            committed_at = time.perf_counter()
            return proposal
        finally:
            finished_at = time.perf_counter()
            log_timing(
                "reflection",
                queue_wait=started_at - claimed_at,
                context=context_finished_at - context_started_at,
                wait_for_model=reservation_acquired_at - wait_started_at,
                inference=inference_finished_at - reservation_acquired_at,
                parse=parsed_at - parse_started_at,
                commit=committed_at - parsed_at,
                total=finished_at - started_at,
            )
            log_performance(
                "reflection",
                input_tokens=timing.prompt_tokens,
                output_tokens=timing.generated_tokens,
                generation_ms=max(0.0, timing.model_finished_at - timing.model_started_at) * 1000,
                turn_count=int(job.get("turn_count") or 0),
                applied_count=applied_count,
                rejected_count=rejected_count,
                empty=int(applied_count == 0),
            )

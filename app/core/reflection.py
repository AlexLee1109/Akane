"""Post-turn semantic extraction followed by deterministic state validation."""

from __future__ import annotations

import json
import math
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
    Turn,
    clamp,
)
from app.core.prompt import build_reflection_prompt
from app.core.store import Store
from app.core.utils import log_performance, log_timing, relevance, text_key

_REFLECTION_KEYS = {"memories", "self", "mood", "relationship"}


class ReflectionOutputError(RuntimeError):
    """Reflection output was invalid and must not consume the pending range."""


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


def _evidence_turn_id(
    value: object,
    *,
    turns: tuple[Turn, ...],
    roles: frozenset[str],
    fallback_source: str,
    fallback_id: str,
) -> str:
    excerpt = " ".join(str(value or "").split()).casefold()
    if excerpt:
        for turn in reversed(turns):
            if turn.role in roles and excerpt in " ".join(turn.content.split()).casefold():
                return turn.id
    return fallback_id if _evidence(value, fallback_source) else ""


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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
        raise ReflectionOutputError("reflection:parse-invalid-json")
    if not isinstance(payload, dict):
        raise ReflectionOutputError("reflection:parse-not-object")
    if not set(payload) <= _REFLECTION_KEYS:
        raise ReflectionOutputError("reflection:parse-unknown-top-level-key")
    for key in ("memories", "self"):
        if key in payload and not isinstance(payload[key], list):
            raise ReflectionOutputError(f"reflection:parse-{key}-not-array")
    for key in ("mood", "relationship"):
        if key in payload and not isinstance(payload[key], dict):
            raise ReflectionOutputError(f"reflection:parse-{key}-not-object")
    return payload


def validate_reflection(
    payload: dict[str, object],
    *,
    context,
    user_text: str,
    assistant_text: str,
    user_turn_id: str,
    assistant_turn_id: str,
    source_turns: tuple[Turn, ...] = (),
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
            if subject == "shared":
                user_source_id = _evidence_turn_id(
                    evidence,
                    turns=source_turns,
                    roles=frozenset({"user"}),
                    fallback_source=user_text,
                    fallback_id=user_turn_id,
                )
                assistant_source_id = _evidence_turn_id(
                    evidence,
                    turns=source_turns,
                    roles=frozenset({"assistant"}),
                    fallback_source=assistant_text,
                    fallback_id=assistant_turn_id,
                )
                source_ids = tuple(dict.fromkeys((user_source_id, assistant_source_id)))
                if not user_source_id or not assistant_source_id:
                    source_ids = ()
            else:
                source_id = _evidence_turn_id(
                    evidence,
                    turns=source_turns,
                    roles=frozenset({"user"}) if subject == "user" else frozenset({"assistant"}),
                    fallback_source=user_text if subject == "user" else assistant_text,
                    fallback_id=user_turn_id if subject == "user" else assistant_turn_id,
                )
                source_ids = (source_id,) if source_id else ()
            if not source_ids or not text or relevance(evidence, text) < 0.04:
                rejected.append(f"memory[{index}]:ungrounded")
                continue
            if subject == "akane" and kind in {"fact", "event"}:
                rejected.append(f"memory[{index}]:akane-{kind}-owned-elsewhere")
                continue
            try:
                memories.append(remember(
                    profile_id,
                    subject=subject,
                    kind=kind,
                    text=text,
                    importance=_number(raw.get("importance"), 0.4),
                    confidence=_number(raw.get("confidence"), 0.6),
                    source_turn_ids=source_ids,
                    now=timestamp,
                ))
            except ValueError:
                rejected.append(f"memory[{index}]:schema")

    current_by_id = {item.id: item for item in context.state.self_items}
    selected_topics: set[tuple[str, str]] = set()
    raw_self = payload.get("self", [])
    if isinstance(raw_self, list):
        for index, raw in enumerate(raw_self[:4]):
            if not isinstance(raw, dict):
                rejected.append(f"self[{index}]:schema")
                continue
            evidence = str(raw.get("evidence") or "")
            # Akane's Self can only be evidenced by what Akane chose to express.
            evidence_turn_id = _evidence_turn_id(
                evidence,
                turns=source_turns,
                roles=frozenset({"assistant"}),
                fallback_source=assistant_text,
                fallback_id=assistant_turn_id,
            )
            if not evidence_turn_id:
                rejected.append(f"self[{index}]:not-akane-owned")
                continue
            action = str(raw.get("action") or "")
            target_id = str(raw.get("target_id") or "")
            current_item = current_by_id.get(target_id)
            topic = " ".join(str(raw.get("topic") or "").split())[:160]
            value = " ".join(str(raw.get("value") or "").split())[:500]
            reason = " ".join(str(raw.get("reason") or "").split())[:500]
            grounding = f"{topic} {value} {reason}"
            if current_item is not None:
                grounding = (
                    f"{grounding} {current_item.topic} {current_item.value} {current_item.reason}"
                )
            if relevance(evidence, grounding) < 0.03:
                rejected.append(f"self[{index}]:ungrounded")
                continue
            try:
                if action == "form":
                    kind = str(raw.get("kind") or "")
                    if not topic or not value or not reason:
                        rejected.append(f"self[{index}]:schema")
                        continue
                    topic_key = (kind, text_key(topic))
                    if topic_key in selected_topics or any(
                        item.kind == kind and text_key(item.topic) == text_key(topic)
                        for item in current_by_id.values()
                    ):
                        rejected.append(f"self[{index}]:duplicate-topic")
                        continue
                    self_changes.append(form_self_item(
                        profile_id,
                        kind=kind,
                        topic=topic,
                        value=value,
                        strength=max(-0.6, min(0.6, _number(raw.get("strength"), 0.25))),
                        confidence=min(0.45, max(0.1, _number(raw.get("confidence"), 0.3))),
                        reason=reason,
                        source_ids=(evidence_turn_id,),
                        now=timestamp,
                    ))
                    selected_topics.add(topic_key)
                elif action in {"reinforce", "weaken", "revise", "retire", "complete", "abandon"} and target_id in current_by_id:
                    assert current_item is not None
                    if action in {"complete", "abandon"} and current_item.kind != "goal":
                        rejected.append(f"self[{index}]:lifecycle")
                        continue
                    proposed_strength = _number(raw.get("strength"), current_item.strength)
                    proposed_confidence = _number(raw.get("confidence"), current_item.confidence)
                    if action == "reinforce":
                        if current_item.strength < 0:
                            bounded_strength = min(
                                current_item.strength,
                                max(current_item.strength - 0.15, proposed_strength),
                            )
                        else:
                            bounded_strength = max(
                                current_item.strength,
                                min(current_item.strength + 0.15, proposed_strength),
                            )
                        bounded_confidence = max(
                            current_item.confidence,
                            min(current_item.confidence + 0.15, proposed_confidence),
                        )
                    elif action == "weaken":
                        if current_item.strength < 0:
                            bounded_strength = min(
                                0.0,
                                max(current_item.strength, min(current_item.strength + 0.2, proposed_strength)),
                            )
                        else:
                            bounded_strength = max(
                                0.0,
                                min(current_item.strength, max(current_item.strength - 0.2, proposed_strength)),
                            )
                        bounded_confidence = min(
                            current_item.confidence,
                            max(0.0, current_item.confidence - 0.15, proposed_confidence),
                        )
                    else:
                        bounded_strength = max(
                            current_item.strength - 0.75,
                            min(current_item.strength + 0.75, proposed_strength),
                        )
                        bounded_confidence = max(
                            current_item.confidence - 0.2,
                            min(current_item.confidence + 0.15, proposed_confidence),
                        )
                    next_value = value or current_item.value
                    next_reason = reason or current_item.reason
                    if action == "revise" and (not value or not reason):
                        rejected.append(f"self[{index}]:schema")
                        continue
                    self_changes.append(revise_self_item(
                        current_item,
                        action=action,
                        value=next_value,
                        strength=bounded_strength,
                        confidence=bounded_confidence,
                        reason=next_reason,
                        source_ids=(evidence_turn_id,),
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


def _turn_batch(turns: tuple[Turn, ...]) -> tuple[str, str, str]:
    user_text = "\n".join(turn.content for turn in turns if turn.role == "user")
    assistant_text = "\n".join(turn.content for turn in turns if turn.role == "assistant")
    conversation_text = "\n".join(
        f"{'USER' if turn.role == 'user' else 'AKANE'}: {turn.content}"
        for turn in turns
    )
    return user_text, assistant_text, conversation_text


def _shorter_complete_prefix(turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
    for index in range(len(turns) - 2, -1, -1):
        if turns[index].role == "assistant":
            return turns[:index + 1]
    return ()


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
        recovery_attempted = False
        recovery_token_limit = 0
        applied_count = 0
        rejected_count = 0
        parse_status = "not_started"
        proposal_counts = {"memories": 0, "self": 0, "mood": 0, "relationship": 0}
        accepted_counts = {"memories": 0, "self": 0, "mood": 0, "relationship": 0}
        rejection_codes: tuple[str, ...] = ()
        selected_turns = tuple(
            turn for turn in job.get("selected_turns", ()) if isinstance(turn, Turn)
        )
        reflected_first_id = str(job.get("first_turn_id") or "")
        reflected_latest_id = str(job.get("latest_turn_id") or "")
        reflected_turn_count = int(job.get("turn_count") or 0)
        profile_id = str(job["profile_id"])
        try:
            if not self.store.profile_exists(profile_id):
                raise RuntimeError("The reflection profile no longer exists.")
            conversation_id = str(job["conversation_id"])
            if selected_turns:
                user_text, assistant_text, conversation_text = _turn_batch(selected_turns)
                reflected_first_id = selected_turns[0].id
                reflected_latest_id = selected_turns[-1].id
                reflected_turn_count = len(selected_turns)
            else:
                user_text = str(job["user_text"])
                assistant_text = str(job["assistant_text"])
                conversation_text = str(job["conversation_text"])
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
                    shorter = _shorter_complete_prefix(selected_turns)
                    if not shorter:
                        parse_status = "reflection:prompt-single-exchange-too-large"
                        rejection_codes = (parse_status,)
                        rejected_count = 1
                        raise ReflectionOutputError(parse_status)
                    selected_turns = shorter
                    user_text, assistant_text, conversation_text = _turn_batch(selected_turns)
                    reflected_first_id = selected_turns[0].id
                    reflected_latest_id = selected_turns[-1].id
                    reflected_turn_count = len(selected_turns)
                    context = self.context_builder.build(
                        profile_id=profile_id,
                        conversation_id=conversation_id,
                        message=f"{user_text}\n{assistant_text}",
                        allow_tool_context=False,
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
                if timing.finish_reason == "length":
                    try:
                        parse_reflection_json(output)
                    except ReflectionOutputError:
                        recovery_token_limit = min(
                            SETTINGS.reflection_tokens + 64,
                            256,
                            SETTINGS.llama_context_window - prompt_tokens,
                        )
                        if recovery_token_limit > SETTINGS.reflection_tokens:
                            recovery_attempted = True
                            retry_timing = InferenceTiming(started_at)
                            retry_timing.prompt_tokens = prompt_tokens
                            retry_timing.prompt_token_method = prompt_method
                            output = self.runtime.complete_messages(
                                messages,
                                max_tokens=recovery_token_limit,
                                reservation=reservation,
                                timing=retry_timing,
                                temperature=0.1,
                                call_kind="reflection",
                            )
                            timing = retry_timing
            inference_finished_at = time.perf_counter()
            if reservation.preemption.is_set():
                raise InferencePreempted("Reflection yielded before validation.")
            parse_started_at = inference_finished_at
            try:
                payload = parse_reflection_json(output)
            except ReflectionOutputError as exc:
                parse_status = str(exc)
                if timing.finish_reason == "length":
                    parse_status = "reflection:parse-truncated-output"
                rejection_codes = (parse_status,)
                rejected_count = 1
                raise ReflectionOutputError(parse_status) from exc
            parse_status = "ok"
            proposal_counts = {
                "memories": len(payload.get("memories", ())),
                "self": len(payload.get("self", ())),
                "mood": int("mood" in payload),
                "relationship": int("relationship" in payload),
            }
            proposal = validate_reflection(
                payload,
                context=context,
                user_text=user_text,
                assistant_text=assistant_text,
                user_turn_id=reflected_first_id,
                assistant_turn_id=reflected_latest_id,
                source_turns=selected_turns,
            )
            accepted_counts = {
                "memories": len(proposal.memories),
                "self": len(proposal.self_items),
                "mood": int(proposal.mood is not None),
                "relationship": int(proposal.relationship is not None),
            }
            rejection_codes = proposal.rejected
            proposal = replace(
                proposal,
                reflection_job_id=str(job["id"]),
                reflected_through_turn_id=reflected_latest_id,
                reflected_turn_count=reflected_turn_count,
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
            decode_tok_s = (
                timing.generated_tokens / timing.decode_seconds
                if timing.generated_tokens and timing.decode_seconds else 0.0
            )
            log_performance(
                "reflection",
                input_tokens=timing.prompt_tokens,
                output_tokens=timing.generated_tokens,
                generation_ms=max(0.0, timing.model_finished_at - timing.model_started_at) * 1000,
                prefill_ms=timing.prefill_seconds * 1000,
                decode_tok_s=decode_tok_s,
                reused_prefix_tokens=timing.reused_prefix_tokens,
                new_prompt_eval_tokens=timing.new_prompt_eval_tokens,
                model_wait_ms=max(0.0, reservation_acquired_at - wait_started_at) * 1000,
                total_ms=max(0.0, finished_at - started_at) * 1000,
                finish_reason=timing.finish_reason or "unavailable",
                recovery_attempted=int(recovery_attempted),
                recovery_token_limit=recovery_token_limit,
                parse_status=parse_status,
                turn_range=f"{reflected_first_id}:{reflected_latest_id}",
                turn_count=reflected_turn_count,
                proposed=proposal_counts,
                accepted=accepted_counts,
                applied_count=applied_count,
                rejected_count=rejected_count,
                rejection_codes=",".join(rejection_codes) or "none",
                empty=int(applied_count == 0),
            )

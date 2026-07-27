"""Shared single-generation pipeline with grounding, cancellation, and safe commit."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from app.core.character import load_character_profile
from app.core.config import (
    LONG_TERM_MEMORY_PATH,
    MEMORY_PATH,
    PROMPT_DEBUG,
)
from app.core.model_loader import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceTiming,
    ModelManager,
)
from app.core.prompt import describe_model_input
from app.core.utils import compact_text
from app.core.session import (
    ChatInput,
    CompiledStyle,
    GenerationCancelled,
    GenerationQueueFullError,
    ResponseIntention,
    TurnPreparation,
    commit_silent_turn,
    commit_turn,
    finish_turn,
    normalize_chat_input,
    parse_akane_state,
    parse_companion_decision,
    prepare_turn,
    session_state_snapshot,
    timing_enabled,
)
_MAX_METRICS = 64
_METRICS: dict[str, dict[str, object]] = {}
_METRICS_LOCK = threading.Lock()
_SERVICE_POSTURE = re.compile(
    r"\b(?:anything else|feel free to|happy to help|how can i help|let me know if)\b",
    re.IGNORECASE,
)
_GENERIC_VALIDATION = re.compile(
    r"\b(?:your feelings are valid|it is understandable that|i understand how you feel|"
    r"that(?:'s| is) (?:always )?(?:great|nice|good|wonderful)|that makes sense)\b",
    re.IGNORECASE,
)
_DIRECT_QUESTION = re.compile(
    r"(?:^|[.!]\s+)(?:are|can|could|did|do|does|has|have|how|is|may|should|"
    r"was|were|what|when|where|which|who|why|will|would|tell me|explain|"
    r"elaborate|describe|identify|name)\b|\?",
    re.IGNORECASE,
)
_INDIRECT_FOLLOW_UP = re.compile(
    r"\b(?:i(?:'m| am) curious (?:about|how|what|which|who|why)|i(?:'d| would) like "
    r"to know|i wonder(?:ed|ing)? (?:how|what|whether|which|who|why)|care to "
    r"elaborate|go on|say more|tell me more|walk me through it)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_ASSUMPTION = re.compile(
    r"\b(?:you must (?:feel|be)|you(?:'re| are) (?:clearly|obviously|probably) "
    r"(?:excited|happy|proud|relieved|sad|upset)|i know (?:how )?you feel|"
    r"that (?:must have been|was obviously) (?:hard|difficult|easy)|"
    r"after all (?:that|your) (?:hard )?work|this (?:clearly )?means a lot to you|"
    r"that(?:'s| is) (?:a )?(?:huge|major|important) milestone|"
    r"(?:the|your|that) (?:project|system|implementation|code|compiler) "
    r"(?:is|looks|sounds) (?:ambitious|complex|elegant|impressive|solid)|"
    r"you(?:'ve| have) been (?:coding|playing|reading|studying|working) "
    r"(?:all|for)\b)\b",
    re.IGNORECASE,
)
_PERSONAL_EXPERIENCE_CLAIM = re.compile(
    r"\bi\s+(?:remember|have experienced|went through|felt the same|"
    r"know what (?:that|this) is like)\b",
    re.IGNORECASE,
)
_INTERNAL_TERMS = re.compile(
    r"\b(?:affect core|dynamic guidance|persistent mood|response intention|"
    r"style compiler|system prompt|my internal state|my memory system|my response "
    r"selection|my prompt processing|processing data|analyzing inputs|monitoring|"
    r"waiting for requests|running calculations)\b",
    re.IGNORECASE,
)
_ACTIVITY_CLAIM = re.compile(
    r"\bi\s+(?:(?:am|was|have been)\s+)?(?:coding|playing|played|reading|studying|"
    r"watched|watching|working on|visited|went to)\b",
    re.IGNORECASE,
)
_TITLE_CLAIM = re.compile(
    r"\b[Ii]\s+(?:(?:am|was|have been)\s+)?(?:playing|played|reading|watched|watching)\s+"
    r"([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,4})",
)
_ACCESS_CLAIM = re.compile(
    r"\bi\s+(?:browsed|checked|looked up|searched)\s+(?:online|the internet|the web)\b",
    re.IGNORECASE,
)
_LAUGHTER = re.compile(r"(?:\b(?:haha|hehe|lol|lmao)\b|[😂🤣])", re.IGNORECASE)
@dataclass(frozen=True, slots=True)
class GenerationEvent:
    kind: str
    generation_id: str
    text: str = ""
    reply: str = ""
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ResponseValidation:
    """Read-only behavioral findings for one completed response."""

    violations: tuple[str, ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()


def validate_response_style(
    reply: str,
    style: CompiledStyle,
    intention: ResponseIntention,
    *,
    grounding_context: str = "",
    recent_outputs: tuple[str, ...] = (),
    persona_text: str = "",
    request_context: str = "",
) -> ResponseValidation:
    """Report deterministic style and grounding risks without rewriting output."""

    text = str(reply or "").strip()
    violations: list[str] = []
    evidence: list[tuple[str, str]] = []

    def finding(category: str, detail: str) -> None:
        violations.append(category)
        evidence.append((category, compact_text(detail, 120)))

    question_count = text.count("?")
    questions_prohibited = style.question_gate != "open"
    if question_count > 1:
        finding("excessive questions", f"{question_count} question marks")
    if questions_prohibited and _DIRECT_QUESTION.search(text):
        finding("prohibited question behavior", "question punctuation or interrogative syntax")
    if questions_prohibited and _INDIRECT_FOLLOW_UP.search(text):
        finding("indirect follow-up", "conversational request for more detail")
    service_posture = _SERVICE_POSTURE.search(text)
    if service_posture:
        finding("service posture", service_posture.group(0))
    generic = _GENERIC_VALIDATION.search(text)
    if generic:
        request_terms = set(_normalized_words(request_context)) - _VALIDATION_STOPWORDS
        response_terms = set(_normalized_words(text)) - _VALIDATION_STOPWORDS
        if not request_terms or not request_terms & response_terms:
            finding("generic validation", generic.group(0))
    assumption = _UNSUPPORTED_ASSUMPTION.search(text)
    supported_context = f"{request_context}\n{grounding_context}".lower()
    if assumption and assumption.group(0).lower() not in supported_context:
        finding("unsupported assumption", assumption.group(0))
    internal_term = _INTERNAL_TERMS.search(text)
    if internal_term:
        finding("internal terminology", internal_term.group(0))
    access_claim = _ACCESS_CLAIM.search(text)
    if access_claim:
        finding("unsupported access claim", access_claim.group(0))

    grounding_lower = str(grounding_context or "").lower()
    akane_grounding = "\n".join(
        line
        for line in grounding_lower.splitlines()
        if "arcane current activity" not in line
        and "current user activity" not in line
    )
    activity = _ACTIVITY_CLAIM.search(text)
    if activity:
        fragment = re.split(r"[.!?\n]", text[activity.start() :], maxsplit=1)[0]
        claim_terms = set(_normalized_words(fragment)) - {
            "a",
            "am",
            "an",
            "been",
            "coding",
            "i",
            "on",
            "played",
            "playing",
            "read",
            "reading",
            "studying",
            "the",
            "to",
            "visited",
            "was",
            "watched",
            "watching",
            "went",
            "working",
        }
        grounding_terms = set(_normalized_words(akane_grounding))
        if not akane_grounding.strip() or (
            claim_terms and claim_terms.isdisjoint(grounding_terms)
        ):
            finding("unrecorded activity", fragment)
    title = _TITLE_CLAIM.search(text)
    if title and title.group(1).lower() not in akane_grounding:
        finding("unrecorded title", title.group(1))
    personal_experience = _PERSONAL_EXPERIENCE_CLAIM.search(text)
    if personal_experience:
        fragment = re.split(
            r"[.!?\n]",
            text[personal_experience.start() :],
            maxsplit=1,
        )[0]
        claim_terms = set(_normalized_words(fragment)) - _VALIDATION_STOPWORDS - {
            "experienced", "felt", "know", "like", "remember", "same", "through", "went"
        }
        grounding_terms = set(_normalized_words(akane_grounding))
        if not akane_grounding.strip() or (
            claim_terms and claim_terms.isdisjoint(grounding_terms)
        ):
            finding("unsupported personal experience", fragment)

    paragraph_limit, sentence_limit = style.validation_limits
    paragraphs = [value for value in re.split(r"\n\s*\n", text) if value.strip()]
    sentences = re.findall(r"[^.!?\n]+[.!?]+(?:\s|$)|[^.!?\n]+$", text)
    if paragraph_limit and len(paragraphs) > paragraph_limit:
        finding("paragraph-limit violation", f"{len(paragraphs)} paragraphs")
    if sentence_limit and len(sentences) > sentence_limit:
        finding("sentence-limit violation", f"{len(sentences)} sentences")
    if style.humor == "none" and _LAUGHTER.search(text):
        finding(
            "serious-context style violation"
            if intention.primary in {"comfort", "disagree", "reassure", "set boundary"}
            else "style-intention mismatch",
            "laughter while humor is disabled",
        )

    if _has_phrase_overlap(text, style.prompt_text(), span=3):
        finding("copied compiler wording", "three-word compiler overlap")
    if persona_text and _has_phrase_overlap(text, persona_text, span=8):
        finding("phrase overlap with Identity or Soul", "eight-word persona overlap")

    opening = _edge_words(text, first=True)
    closing = _edge_words(text, first=False)
    for prior in recent_outputs:
        if opening and opening == _edge_words(prior, first=True):
            finding("repeated opening", "same four-word opening")
            break
    for prior in recent_outputs:
        if closing and closing == _edge_words(prior, first=False):
            finding("recurring closing phrase", "same four-word closing")
            break
    if text.rstrip().endswith("?") and any(
        str(prior or "").rstrip().endswith("?") for prior in recent_outputs[-2:]
    ):
        finding("repeated question ending", "recent response also ended with a question")
    unique_violations = tuple(dict.fromkeys(violations))
    unique_evidence = tuple(dict.fromkeys(evidence))
    return ResponseValidation(unique_violations, unique_evidence)


_VALIDATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "for", "from",
    "i", "in", "is", "it", "my", "of", "on", "that", "the", "this", "to",
    "was", "we", "with", "you", "your",
}


def _normalized_words(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9']+", str(value or "").lower()))


def _has_phrase_overlap(left: str, right: str, *, span: int) -> bool:
    left_words = _normalized_words(left)
    right_words = _normalized_words(right)
    if len(left_words) < span or len(right_words) < span:
        return False
    right_windows = {
        right_words[index : index + span]
        for index in range(len(right_words) - span + 1)
    }
    return any(
        left_words[index : index + span] in right_windows
        for index in range(len(left_words) - span + 1)
    )


def _edge_words(value: str, *, first: bool) -> tuple[str, ...]:
    words = _normalized_words(value)
    if len(words) < 4:
        return ()
    return words[:4] if first else words[-4:]


def prepare_reply(
    chat_input: ChatInput | str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
    exact_tokens: bool = True,
) -> TurnPreparation:
    if isinstance(chat_input, str):
        chat_input = normalize_chat_input(
            text=chat_input,
            conversation_id=session_id or "popup:default",
        )
    return prepare_turn(
        chat_input,
        skip_memory=skip_memory,
        skip_if_busy=skip_if_busy,
        token_counter=(
            ModelManager.get_instance().tokenize_prompt if exact_tokens else None
        ),
    )


def generate_reply(prepared: TurnPreparation) -> str:
    for event in _reply_events(prepared, emit_deltas=False):
        if event.kind == "done":
            return event.reply
    raise RuntimeError("Model returned no completion event.")


def stream_reply(prepared: TurnPreparation):
    """Yield one grounded reply and one completion event from one model generation."""

    yield from _reply_events(prepared, emit_deltas=True)


def commit_reply(
    prepared: TurnPreparation,
    reply: str,
    *,
    timing: InferenceTiming | None = None,
    preference_updates: tuple[dict[str, object], ...] = (),
    interest_additions: tuple[str, ...] = (),
    relationship_updates: tuple[dict[str, object], ...] = (),
    activity_update: dict[str, object] | None = None,
    next_activity: dict[str, object] | None = None,
    emotion_update: dict[str, object] | None = None,
) -> None:
    """Commit a completed deterministic or generated reply and record diagnostics."""

    profile = load_character_profile()
    validation = validate_response_style(
        reply,
        prepared.turn_context.compiled_style,
        prepared.turn_context.response_intention,
        grounding_context="\n".join(
            value
            for value in (
                prepared.turn_context.life_context,
                prepared.turn_context.relevant_memories,
            )
            if value
        ),
        recent_outputs=tuple(
            turn.content
            for turn in prepared.memory_context.recent_turns
            if turn.role == "assistant"
        ),
        persona_text=f"{profile.identity}\n{profile.soul}",
        request_context=prepared.chat_input.text,
    )
    commit_turn(
        prepared,
        reply,
        preference_updates=preference_updates,
        interest_additions=interest_additions,
        relationship_updates=relationship_updates,
        activity_update=activity_update,
        next_activity=next_activity,
        emotion_update=emotion_update,
    )
    _remember_metrics(
        prepared,
        committed=True,
        timing=timing,
        validation=validation,
    )


def _reply_events(prepared: TurnPreparation, *, emit_deltas: bool):
    first_delivery_at = 0.0
    output_chunks = 0
    timing = InferenceTiming(requested_at=time.perf_counter())

    try:
        _remember_metrics(prepared, committed=False)
        prepared.handle.raise_if_cancelled()
        if (
            prepared.chat_input.autonomous
            and not prepared.turn_context.initiative_worthwhile
        ):
            yield GenerationEvent(
                "done",
                prepared.generation_id,
                metadata={"skipped": "no worthwhile initiative"},
            )
            return
        manager = ModelManager.get_instance()
        messages = prepared.prompt_plan.messages
        _log_model_input(
            prepared,
            generation_mode="streaming" if emit_deltas else "buffered_stream",
        )
        parts: list[str] = []
        for text in manager.stream(
            messages,
            prompt_tokens=prepared.prompt_plan.token_ids,
            template_stop_sequences=prepared.prompt_plan.stop_sequences,
            max_tokens=prepared.max_tokens,
            cancellation=prepared.handle.cancellation,
            queue_deadline=prepared.handle.queue_deadline,
            timing=timing,
        ):
            output_chunks += 1
            if not timing.first_token_at:
                timing.first_token_at = time.perf_counter()
            parts.append(text)
        prepared.handle.raise_if_cancelled()
        if not timing.model_started_at:
            timing.model_started_at = timing.requested_at
        if not timing.model_finished_at:
            timing.model_finished_at = time.perf_counter()

        postprocess_started_at = time.perf_counter()
        raw_reply = "".join(parts).strip()
        if not raw_reply:
            raise RuntimeError("Model returned no visible reply.")
        parsed = parse_companion_decision(raw_reply)
        state = parse_akane_state(parsed.message)
        reply = state.message
        postprocess_seconds = time.perf_counter() - postprocess_started_at

        if emit_deltas and parsed.decision.should_respond and reply:
            prepared.handle.raise_if_cancelled()
            first_delivery_at = time.perf_counter()
            yield GenerationEvent("delta", prepared.generation_id, text=reply)

        prepared.handle.raise_if_cancelled()
        persistence_started_at = time.perf_counter()
        if parsed.decision.should_respond:
            commit_reply(
                prepared,
                reply,
                timing=timing,
                preference_updates=state.preference_updates,
                interest_additions=state.interest_additions,
                relationship_updates=state.relationship_updates,
                activity_update=state.activity_update,
                next_activity=state.next_activity,
                emotion_update=state.emotion_update,
            )
        else:
            commit_silent_turn(
                prepared,
                preference_updates=state.preference_updates,
                interest_additions=state.interest_additions,
                relationship_updates=state.relationship_updates,
                activity_update=state.activity_update,
                next_activity=state.next_activity,
                emotion_update=state.emotion_update,
            )
        persistence_seconds = time.perf_counter() - persistence_started_at
        _timing_log(
            prepared,
            reply,
            timing=timing,
            first_delivery_at=first_delivery_at,
            output_chunks=output_chunks,
            postprocess_seconds=postprocess_seconds,
            persistence_seconds=persistence_seconds,
        )
        yield GenerationEvent(
            "done",
            prepared.generation_id,
            reply=reply,
            metadata={
                "exact_prompt_tokens": prepared.prompt_plan.rendered_prompt_tokens,
                "context_window": prepared.prompt_plan.context_window,
            },
        )
    except InferenceCancelled as exc:
        raise GenerationCancelled(str(exc)) from exc
    except InferenceQueueTimeout as exc:
        raise GenerationQueueFullError(str(exc)) from exc
    finally:
        finish_turn(prepared)


def _log_model_input(
    prepared: TurnPreparation,
    *,
    generation_mode: str,
) -> None:
    if not PROMPT_DEBUG:
        return
    metadata = describe_model_input(
        prepared.prompt_plan.messages,
        transport=prepared.chat_input.source,
        conversation_id=prepared.chat_input.conversation_id,
        loaded_recent_turns=len(prepared.memory_context.recent_turns),
        summary_turns=len(prepared.memory_context.earlier_turns),
        current_user_text=prepared.chat_input.text,
        generation_mode=generation_mode,
    )
    print(f"[Akane:model-input] {metadata}", flush=True)
    trace = prepared.internal_turn.affect_trace
    if trace is not None:
        print(
            "[Akane:affect] "
            f"interface={prepared.chat_input.source} prior={trace.previous} "
            f"immediate={trace.immediate} candidate={trace.candidate} "
            f"boundary={trace.boundary_level} applied={trace.applied}",
            flush=True,
        )


def _debug_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def _debug_absent(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "none",
        "neutral",
        "null",
        "unknown",
        "unavailable",
    }


def _debug_name(value: object) -> str:
    text = _debug_text(value, 80)
    if _debug_absent(text):
        return "None"
    aliases = {
        "unresolved_conversation": "Unresolved conversation",
        "conversation_topic": "Active conversation topic",
        "grounded_activity": "Grounded activity",
        "relevant_memory": "Relevant memory",
        "active_emotion": "Active emotion",
        "clarification": "Ask for clarification",
        "invited": "Invited continuation",
        "unresolved": "Continue unresolved topic",
    }
    return aliases.get(text.lower(), text.replace("_", " ").capitalize())


def _debug_bool(value: object) -> str:
    return "Yes" if bool(value) else "No"


def debug_state_report(
    conversation_id: str | None,
    profile_id: str | None = None,
    *,
    verbose: bool = False,
) -> str:
    conversation = str(conversation_id or "popup:default")
    profile = str(profile_id or "local:owner")
    snapshot = session_state_snapshot(conversation, profile)
    memory = snapshot.get("memory") or {}
    akane = snapshot.get("akane") or {}
    companion = snapshot.get("companion_decision") or {}
    life_worker = snapshot.get("life_worker") or {}
    presence = akane.get("presence") or {}
    loaded_emotion = akane.get("emotion") or {}
    with _METRICS_LOCK:
        metrics = dict(_METRICS.get(conversation, {}))
    presence_debug = metrics.get("presence_debug") or {}
    current_activity = presence.get("current_activity") or {}
    previous_activity = presence.get("previous_activity") or {}
    next_activity = presence.get("next_activity") or {}
    runtime = ModelManager.get_instance().runtime_report(include_model_hash=verbose)
    prompt = metrics.get("prompt_debug") or {}
    trimmed = prompt.get("trimmed") or ()
    lines = [
        "Akane Debug",
        "",
        "Request",
        f"  Intent: {metrics.get('intent') or memory.get('recent_intent') or 'None'}",
        f"  Topic: {metrics.get('topic') or memory.get('recent_topic') or 'None'}",
        f"  Repetition Count: {int(metrics.get('repetition_count') or 1)}",
        f"  Embodied Action: {metrics.get('embodied_action') or 'None'}",
        f"  Continued After Objection: {_debug_bool(metrics.get('continued_after_objection'))}",
        "",
        "Emotion",
        f"  Previous: {metrics.get('emotion_previous') or 'neutral 0.00'}",
        f"  Immediate: {metrics.get('emotion_immediate') or 'neutral 0.00'}",
        f"  Candidate Persistent: {metrics.get('emotion_candidate') or 'neutral 0.00'}",
        f"  Loaded: {loaded_emotion.get('primary', 'neutral')} {float(loaded_emotion.get('intensity') or 0.0):.2f}",
        f"  Applied: {_debug_bool(metrics.get('emotion_applied'))}",
        f"  Committed: {_debug_bool(metrics.get('committed'))}",
        "",
        "Companion Decision",
        "  Decision Parsed: "
        f"{_debug_bool(companion.get('decision_parsed'))}",
        "  Should Respond: "
        f"{_debug_bool(companion.get('should_respond'))}",
        f"  Pause Seconds: {companion.get('pause_seconds') or 'None'}",
        "  Currently Paused: "
        f"{_debug_bool(companion.get('currently_paused'))}",
        "  Message Suppressed: "
        f"{_debug_bool(companion.get('message_suppressed'))}",
        "",
        "Autonomous Life",
        f"  Current Activity: {current_activity.get('activity') or 'None'}",
        f"  Activity Continuity: {metrics.get('activity_continuity') or 'none'}",
        "  Current Activity Started At: "
        + str(
            metrics.get("current_activity_started_at")
            if metrics.get("current_activity_started_at") is not None
            else "None"
        ),
        "  Last Assistant Turn At: "
        + str(
            metrics.get("last_assistant_turn_at")
            if metrics.get("last_assistant_turn_at") is not None
            else "None"
        ),
        "  Activity Detail Grounded: "
        f"{metrics.get('activity_detail_grounded') or 'not applicable'}",
        f"  Configured Timezone: {metrics.get('configured_timezone') or 'None'}",
        f"  Current Local Time: {metrics.get('current_local_time') or 'None'}",
        f"  Current Daypart: {metrics.get('current_daypart') or 'None'}",
        f"  Current Activity Ends At: {current_activity.get('ends_at') or 'None'}",
        f"  Previous Activity: {previous_activity.get('activity') or 'None'}",
        f"  Next Activity: {next_activity.get('activity') or 'None'}",
        f"  Life Pending: {_debug_bool(presence.get('life_pending'))}",
        f"  Life Reason: {presence.get('life_reason') or 'None'}",
        f"  Life Next Run At: {presence.get('life_next_run_at') or 'None'}",
        "  Activity Expired This Turn: "
        f"{_debug_bool(presence_debug.get('activity_expired_this_turn'))}",
        "  Activity Activated This Turn: "
        f"{_debug_bool(presence_debug.get('activity_activated_this_turn'))}",
        f"  Autonomous Proposal: {life_worker.get('Autonomous Proposal') or 'None'}",
        f"  Proposal Rejected: {_debug_bool(life_worker.get('Proposal Rejected'))}",
        f"  Rejection Reason: {life_worker.get('Rejection Reason') or 'None'}",
        f"  Activity Pattern: {presence.get('activity_pattern') or {}}",
        f"  Life Worker Started: {_debug_bool(life_worker.get('Life Worker Started'))}",
        f"  Pending Profiles: {', '.join(life_worker.get('Pending Profiles') or ()) or 'None'}",
        f"  Life Job Claimed: {life_worker.get('Life Job Claimed') or 'None'}",
        f"  Claim Age: {float(life_worker.get('Claim Age') or 0.0):.1f}",
        f"  Life Inference Started: {_debug_bool(life_worker.get('Life Inference Started'))}",
        f"  Life Block Parsed: {_debug_bool(life_worker.get('Life Block Parsed'))}",
        f"  Life Activity Persisted: {_debug_bool(life_worker.get('Life Activity Persisted'))}",
        f"  Life Job Completed: {life_worker.get('Life Job Completed') or 'None'}",
        f"  Life Job Failed: {life_worker.get('Life Job Failed') or 'None'}",
        f"  Next Retry At: {life_worker.get('Next Retry At') or 'None'}",
        "",
        "Conversation",
        f"  Complete Pairs: {int(metrics.get('complete_pairs') or 0)}",
        f"  Role Sequence: {metrics.get('role_sequence') or 'None'}",
        f"  Current Message Count: {int(metrics.get('current_message_count') or 0)}",
        "",
        "Prompt",
        f"  Exact Tokens: {prompt.get('exact_tokens') if prompt.get('exact_tokens') is not None else 'None'}",
        f"  Context Window: {prompt.get('context_window') or runtime['context_window']}",
        f"  Trimmed Content: {', '.join(str(item) for item in trimmed) if trimmed else 'None'}",
        "",
    ]
    return "\n".join(lines)


def _remember_metrics(
    prepared: TurnPreparation,
    *,
    committed: bool,
    timing: InferenceTiming | None = None,
    validation: ResponseValidation | None = None,
    presence_debug: dict[str, object] | None = None,
) -> None:
    del validation
    signal = prepared.internal_turn.signal
    trace = prepared.internal_turn.affect_trace
    messages = prepared.prompt_plan.messages
    history = messages[1:-1] if len(messages) >= 2 else []
    prompt_debug = prepared.prompt_plan.debug_metadata()
    with _METRICS_LOCK:
        _METRICS[prepared.session_id] = {
            "prompt_debug": prompt_debug,
            "intent": signal.intent,
            "topic": signal.topic,
            "repetition_count": signal.repetition_count,
            "embodied_action": (
                f"{signal.embodied_action} -> {signal.embodied_target}"
                if signal.embodied_action else ""
            ),
            "continued_after_objection": signal.continued_after_objection,
            "emotion_previous": trace.previous if trace else "neutral 0.00",
            "emotion_immediate": trace.immediate if trace else "neutral 0.00",
            "emotion_candidate": trace.candidate if trace else "neutral 0.00",
            "emotion_applied": signal.emotion_applied,
            "complete_pairs": sum(item.get("role") == "assistant" for item in history),
            "role_sequence": ",".join(item.get("role", "unknown") for item in messages),
            "current_message_count": int(
                bool(messages)
                and messages[-1].get("role") == "user"
                and messages[-1].get("content") == prepared.chat_input.text
            ),
            "committed": committed,
            "activity_continuity": prepared.turn_context.activity_continuity,
            "current_activity_started_at": (
                prepared.turn_context.current_activity_started_at
            ),
            "last_assistant_turn_at": prepared.turn_context.last_assistant_turn_at,
            "activity_detail_grounded": (
                prepared.turn_context.activity_detail_grounded
            ),
            "configured_timezone": prepared.turn_context.configured_timezone,
            "current_local_time": prepared.turn_context.current_local_time,
            "current_daypart": prepared.turn_context.current_daypart,
            "exact_prompt_tokens": (
                timing.prompt_tokens if timing and timing.prompt_tokens
                else prepared.prompt_plan.rendered_prompt_tokens
            ),
            "updated_at": time.time(),
            "presence_debug": dict(presence_debug or {}),
        }
        if len(_METRICS) > _MAX_METRICS:
            oldest = min(_METRICS, key=lambda key: float(_METRICS[key].get("updated_at") or 0.0))
            _METRICS.pop(oldest, None)


def _timing_log(
    prepared: TurnPreparation,
    reply: str,
    *,
    timing: InferenceTiming,
    first_delivery_at: float = 0.0,
    output_chunks: int = 0,
    postprocess_seconds: float = 0.0,
    persistence_seconds: float = 0.0,
) -> None:
    if not timing_enabled():
        return
    done = time.perf_counter()
    fields = [
        "[Akane:timing]",
        f"total={done - prepared.started_at:.3f}s",
        f"prompt_tokens_exact={timing.prompt_tokens}",
        f"output_chars={len(reply)}",
        f"stream_chunks={output_chunks}",
        f"postprocess={postprocess_seconds:.3f}s",
        f"persistence={persistence_seconds:.3f}s",
    ]
    if timing.first_token_at:
        fields.insert(1, f"first_token={timing.first_token_at - prepared.started_at:.3f}s")
    if first_delivery_at:
        fields.insert(2, f"first_delivery={first_delivery_at - prepared.started_at:.3f}s")
    print(" ".join(fields), flush=True)

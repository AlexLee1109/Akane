"""Shared single-generation pipeline with grounding, cancellation, and safe commit."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass

from app.core.character import load_character_profile
from app.core.config import (
    LONG_TERM_MEMORY_PATH,
    MEMORY_PATH,
    PROMPT_DEBUG,
    STREAM_CHUNK_CHARS,
    STREAM_FLUSH_SECONDS,
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
    commit_turn,
    finish_turn,
    normalize_chat_input,
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
_DEBUG_SOURCE_LABELS = {
    "identity": "Identity",
    "soul": "Soul",
    "hard_constraints": "Hard rules",
    "dynamic_guidance": "Dynamic guidance",
    "earlier_user": "Recent conversation",
    "earlier_assistant": "Recent conversation",
    "recent_user": "Recent conversation",
    "recent_assistant": "Recent conversation",
    "relationship": "Relationship context",
    "retrieved_memory": "Memory",
    "memory": "Memory",
    "preference_continuity": "Memory",
    "current_user": "Current message",
    "editor_context": "Code context",
    "code_context": "Code context",
    "reply_quote": "Reply context",
    "life_context": "Activity context",
    "date_time": "Date and time",
}

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
    commit_turn(prepared, reply)
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
        pending: list[str] = []
        pending_chars = 0
        last_flush_at = time.monotonic()
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
            if not emit_deltas:
                continue
            pending.append(text)
            pending_chars += len(text)
            now = time.monotonic()
            if (
                pending_chars >= STREAM_CHUNK_CHARS
                or now - last_flush_at >= STREAM_FLUSH_SECONDS
            ):
                prepared.handle.raise_if_cancelled()
                if not first_delivery_at:
                    first_delivery_at = time.perf_counter()
                yield GenerationEvent(
                    "delta",
                    prepared.generation_id,
                    text="".join(pending),
                )
                pending.clear()
                pending_chars = 0
                last_flush_at = now

        if emit_deltas and pending:
            prepared.handle.raise_if_cancelled()
            if not first_delivery_at:
                first_delivery_at = time.perf_counter()
            yield GenerationEvent(
                "delta",
                prepared.generation_id,
                text="".join(pending),
            )
        prepared.handle.raise_if_cancelled()
        if not timing.model_started_at:
            timing.model_started_at = timing.requested_at
        if not timing.model_finished_at:
            timing.model_finished_at = time.perf_counter()

        postprocess_started_at = time.perf_counter()
        reply = "".join(parts).strip()
        if not reply:
            raise RuntimeError("Model returned no visible reply.")
        postprocess_seconds = time.perf_counter() - postprocess_started_at

        prepared.handle.raise_if_cancelled()
        persistence_started_at = time.perf_counter()
        commit_reply(prepared, reply, timing=timing)
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


def _legacy_debug_state_report(
    conversation_id: str | None,
    profile_id: str | None = None,
    *,
    verbose: bool = False,
) -> str:
    conversation = str(conversation_id or "popup:default")
    snapshot = session_state_snapshot(conversation, profile_id)
    memory = snapshot.get("memory") or {}
    akane = snapshot.get("akane") or {}
    emotion = akane.get("emotion") or {}
    with _METRICS_LOCK:
        metrics = dict(_METRICS.get(conversation, {}))
    prompt_debug = metrics.get("prompt_debug")
    prompt_debug = prompt_debug if isinstance(prompt_debug, dict) else {}
    versions = prompt_debug.get("persona_versions")
    versions = versions if isinstance(versions, dict) else {}
    category_tokens = prompt_debug.get("sections")
    category_tokens = category_tokens if isinstance(category_tokens, dict) else {}
    source_roles = prompt_debug.get("source_roles")
    source_roles = source_roles if isinstance(source_roles, dict) else {}
    activity = akane.get("activity")
    activity = activity if isinstance(activity, dict) else {}
    activity_source = _debug_text(metrics.get("grounded_activity_source"), 32)
    if _debug_absent(activity_source):
        activity_source = _debug_text(activity.get("source"), 32)
    sources = prompt_debug.get("sources")
    sources = sources if isinstance(sources, (list, tuple)) else ()
    trimmed = prompt_debug.get("trimmed")
    trimmed = trimmed if isinstance(trimmed, (list, tuple)) else ()
    disposition = _debug_disposition(
        metrics.get("final_disposition") or emotion.get("dominant")
    )
    current_mood = _debug_name(emotion.get("mood") or metrics.get("prior_mood"))
    reaction = _debug_reaction(
        metrics.get("contextual_reaction"),
        metrics.get("detected_signal"),
    )
    short_emotion = _debug_scored_value(metrics.get("active_emotion"))
    mood_changes = _debug_mood_changes(
        metrics.get("state_changes"),
        metrics.get("decay_applied"),
    )
    focus = _debug_text(memory.get("recent_topic"), 80)
    included = _debug_source_names(sources)
    history_messages = sum(
        str(source).startswith(("earlier_", "recent_")) for source in sources
    )
    prompt_tokens, count_label = _debug_prompt_tokens(metrics)
    reserved_value = prompt_debug.get("reserved_output_tokens")
    reserved_tokens = int(reserved_value) if reserved_value is not None else None
    window_value = prompt_debug.get("context_window")
    context_window = int(window_value) if window_value is not None else None
    total_reserved = (
        prompt_tokens + reserved_tokens
        if prompt_tokens is not None and reserved_tokens is not None
        else None
    )
    available = (
        max(0, context_window - total_reserved)
        if context_window is not None and total_reserved is not None
        else None
    )
    interface = _debug_name(metrics.get("interface"))
    activity_present = not _debug_absent(activity_source)
    activity_description = _debug_text(activity.get("description"), 80)
    activity_is_recent = (
        activity_source == "offscreen_schedule"
        or _debug_text(activity.get("status"), 24).lower() == "completed"
    )
    elapsed = float(metrics.get("elapsed_seconds") or 0.0)
    circadian_lines = _debug_delta_lines(metrics.get("circadian_delta"))
    ambient_lines = _debug_delta_lines(metrics.get("ambient_delta"))
    event_count = len(metrics.get("event_ids") or ())
    signal_label, _signal_score = _debug_scored_parts(metrics.get("detected_signal"))
    signal_confidence = float(metrics.get("signal_confidence") or 0.0)
    activation_threshold = float(metrics.get("activation_threshold") or 0.0)
    emotion_applied = bool(
        signal_label
        and signal_label != "neutral"
        and signal_confidence >= activation_threshold
    )
    life_metrics = metrics.get("life_trace")
    life_metrics = life_metrics if isinstance(life_metrics, dict) else {}
    previous_life = life_metrics.get("previous_activity")
    previous_life = previous_life if isinstance(previous_life, dict) else {}
    interaction_life = life_metrics.get("current_interaction")
    interaction_life = interaction_life if isinstance(interaction_life, dict) else {}
    background_life = life_metrics.get("background_activity")
    background_life = background_life if isinstance(background_life, dict) else {}
    recent_life = life_metrics.get("recent_completed_activity")
    recent_life = recent_life if isinstance(recent_life, dict) else {}
    creative_life = life_metrics.get("active_creative_event")
    creative_life = creative_life if isinstance(creative_life, dict) else {}
    recorded_life = background_life or recent_life or creative_life
    interaction_present = bool(
        interaction_life.get("description") or activity.get("current_interaction")
    )
    life_need_changes = _debug_delta_lines(life_metrics.get("need_changes"))
    life_mood_effects = _debug_delta_lines(life_metrics.get("mood_effects"))
    memory_trace = metrics.get("memory_trace")
    memory_trace = memory_trace if isinstance(memory_trace, dict) else {}
    stored_working = akane.get("working")
    stored_working = stored_working if isinstance(stored_working, dict) else {}
    arcane_activity = stored_working.get("arcane_current_activity")
    arcane_activity = arcane_activity if isinstance(arcane_activity, dict) else {}
    intention = metrics.get("response_intention")
    intention = intention if isinstance(intention, dict) else {}
    style = metrics.get("style_compiler")
    style = style if isinstance(style, dict) else {}
    validation = metrics.get("validation_results")
    validation = validation if isinstance(validation, (list, tuple)) else ()
    semantic_event = metrics.get("semantic_event")
    semantic_event = semantic_event if isinstance(semantic_event, dict) else {}
    lines = [
        "Akane Debug",
        "",
        "Request",
        f"  Interface: {interface}",
        f"  Intent: {_debug_name(memory.get('recent_intent'))}",
        f"  Focus: {_debug_quoted(focus)}",
        f"  Subject: {_debug_quoted(semantic_event.get('subject'))}",
        f"  Event: {_debug_name(semantic_event.get('event_type'))}",
        f"  Status: {_debug_name(semantic_event.get('status'))}",
        f"  Actor: {_debug_name(semantic_event.get('actor'))}",
        f"  Negated: {_debug_bool(semantic_event.get('negated'))}",
        f"  State Committed: {_debug_bool(metrics.get('committed'))}",
        "",
        "Response State",
        f"  Mood: {current_mood}",
        f"  Reaction: {reaction}",
        f"  Disposition: {disposition}",
        f"  Short Emotion: {short_emotion}",
    ]
    if mood_changes:
        lines.append("  Mood Changes:")
        lines.extend(f"    - {change}" for change in mood_changes)
    else:
        lines.append("  Mood Changes: None")
    lines.extend(
        (
            "",
            "Companion Planning",
            f"  Primary: {_debug_name(intention.get('primary'))}",
            f"  Optional Behavior: {_debug_name(intention.get('optional_behavior'))}",
            f"  Continuity: {_debug_name(intention.get('continuity'))}",
            f"  Grounding: {_debug_name(intention.get('grounding'))}",
            f"  Question Permitted: {_debug_bool(intention.get('question_permitted'))}",
            f"  Callback Permitted: {_debug_bool(intention.get('callback_permitted'))}",
            "  Grounded Detail Permitted: "
            f"{_debug_bool(intention.get('grounded_detail_permitted'))}",
            "  Suppression Reasons: "
            f"{_debug_items(intention.get('suppression_reasons')) or 'None'}",
        )
    )
    lines.extend(
        (
            "",
            "Style Compiler",
            "  Directives: "
            f"{_debug_style_directives(style.get('directives')) or 'None'}",
            f"  Humor Policy: {_debug_name(style.get('humor'))}",
            f"  Question Gate: {_debug_name(style.get('question_gate'))}",
            "  Validation Limits: "
            f"{_debug_items(style.get('validation_limits')) or 'None'}",
            "  Validation Results: "
            f"{_debug_items(validation) or 'None'}",
        )
    )
    lines.extend(("", "Time Evolution", f"  Elapsed: {_debug_duration(elapsed)}"))
    meaningful_time_change = bool(circadian_lines or ambient_lines or event_count)
    if meaningful_time_change:
        lines.append(
            "  Circadian Effect: "
            + (", ".join(circadian_lines) if circadian_lines else "None")
        )
        lines.append(
            "  Ambient Drift: "
            + (", ".join(ambient_lines) if ambient_lines else "None")
        )
        lines.append(f"  Offscreen Events: {event_count}")
    else:
        lines.append("  Mood Update: No meaningful elapsed-time change")
    lines.extend(
        (
            "",
            "Emotion Analysis",
            f"  User Signal: {_debug_name(signal_label)}",
            f"  Confidence: {signal_confidence:.2f}",
            f"  Activation Threshold: {activation_threshold:.2f}",
            f"  Applied: {_debug_bool(emotion_applied)}",
            "  Completion Appraisal: "
            f"{_debug_bool(metrics.get('completion_appraisal'))}",
        )
    )
    lines.extend(
        (
            "",
            "Memory",
            f"  Records Considered: {int(memory_trace.get('records_considered') or 0)}",
            f"  Records Used: {int(memory_trace.get('records_used') or 0)}",
            "  Kinds Used: "
            f"{_debug_mapping(memory_trace.get('retrieved_by_kind')) or 'None'}",
            "  Use Decisions: "
            f"{_debug_pairs_text(memory_trace.get('memory_uses'))}",
            "  Active Correction: "
            f"{_debug_text(memory_trace.get('active_correction'), 48) or 'None'}",
            "  Active Thread: "
            f"{_debug_text(memory_trace.get('active_thread'), 48) or 'None'}",
            "  Grounded Self Event: "
            f"{_debug_text(memory_trace.get('grounded_self_event'), 48) or 'None'}",
            "  Arcane Current Activity: "
            f"{_debug_text(arcane_activity.get('content'), 140) or 'None'}",
            "  Candidate Writes: "
            f"{int(memory_trace.get('candidate_writes') or 0)}",
            "  Candidate Updates: "
            f"{int(memory_trace.get('candidate_updates') or 0)}",
            f"  Commit Status: {_debug_name(memory_trace.get('commit_result'))}",
        )
    )
    lines.extend(
        (
            "",
            "Offscreen Life",
            "  Elapsed Since Update: "
            f"{_debug_duration(life_metrics.get('elapsed_seconds'))}",
            "  Current Interaction: "
            f"{_debug_activity_description(life_metrics.get('current_interaction'))}",
            "  Background Activity: "
            f"{_debug_activity_description(life_metrics.get('background_activity'))}",
            "  Recent Completed Activity: "
            f"{_debug_activity_description(life_metrics.get('recent_completed_activity'))}",
            "  Active Creative Event: "
            f"{_debug_activity_description(life_metrics.get('active_creative_event'))}",
            "  Event Source: "
            f"{_debug_name(recorded_life.get('source'))}",
            "  Grounded Details: "
            + (
                "Available"
                if any(
                    recorded_life.get(name)
                    for name in ("description", "subject", "reaction")
                )
                else "None"
            ),
            "  Previous Background Activity: "
            f"{_debug_activity_description(previous_life)}",
            "  Completed Activities: "
            f"{int(life_metrics.get('completed_count') or 0)}",
            "  Recent Reaction: "
            f"{_debug_text(life_metrics.get('recent_reaction'), 80) or 'None'}",
            "  Mood Effect: "
            f"{', '.join(life_mood_effects) if life_mood_effects else 'None'}",
        )
    )
    if life_need_changes:
        lines.append("  Need Changes:")
        lines.extend(f"    - {change}" for change in life_need_changes)
    else:
        lines.append("  Life Changes: None")
    lines.extend(
        (
            "",
            "Grounding",
            "  Permitted Self Claims: "
            + (
                "Recorded activity details"
                if activity_present
                else "Current interaction only"
                if interaction_present
                else "None"
            ),
            "  Prohibited Self Claims: Invented activity, history, title, or external access",
            "  External Access: Not permitted unless verified context is supplied",
            "  Current Activity: "
            + (
                activity_description or "Grounded activity available"
                if activity_present and not activity_is_recent
                else "None"
            ),
            "  Recent Activity: "
            + (
                activity_description or "Recorded activity available"
                if activity_present and activity_is_recent
                else "None"
            ),
        )
    )
    if activity_present:
        lines.extend(
            (
                f"  Activity Source: {_debug_name(activity_source)}",
                "  Activity Age: "
                f"{_debug_duration(metrics.get('grounded_activity_age_seconds'))}",
            )
        )
    lines.extend(
        (
            "",
            "Prompt",
            "  Included:",
        )
    )
    lines.extend(f"    - {source}" for source in included or ("None",))
    lines.extend(
        (
            "",
            f"  History Messages: {history_messages}",
            "  Final Message Count: "
            f"{_debug_number(prompt_debug.get('message_count'))}",
        )
    )
    trimmed_names = _debug_trimmed_sources(trimmed)
    if trimmed_names:
        lines.append("  Trimmed:")
        lines.extend(f"    - {source}" for source in trimmed_names)
    else:
        lines.append("  Trimmed: Nothing")
    lines.extend(
        (
            "",
            "Context Usage",
            f"  Prompt Tokens: {_debug_number(prompt_tokens)}",
            f"  Reserved Response Tokens: {_debug_number(reserved_tokens)}",
            f"  Context Window: {_debug_number(context_window)}",
            "  Total Reserved: "
            + (
                f"{total_reserved} / {context_window}"
                if total_reserved is not None and context_window is not None
                else "None"
            ),
            f"  Available Context: {_debug_number(available)}",
            f"  Token Count: {count_label}",
        )
    )
    if verbose:
        lines.extend(
            _debug_internal_details(
                metrics,
                prompt_debug,
                versions,
                source_roles,
                category_tokens,
                sources,
                trimmed,
                akane,
            )
        )
    return "\n".join(lines)


def _legacy_remember_metrics(
    prepared: TurnPreparation,
    *,
    committed: bool,
    timing: InferenceTiming | None = None,
    validation: ResponseValidation | None = None,
) -> None:
    turn = prepared.internal_turn
    trace = turn.affect_trace
    life_trace: dict[str, object] = {}
    response_intention = getattr(
        prepared.turn_context,
        "response_intention",
        ResponseIntention("acknowledge"),
    )
    compiled_style = getattr(
        prepared.turn_context,
        "compiled_style",
        CompiledStyle(
            "dry",
            (("Goal", "acknowledge"), ("Length", "concise")),
        ),
    )
    memory_trace = dict(getattr(turn, "memory_trace", {}) or {})
    memory_trace["commit_result"] = "committed" if committed else "proposed"
    signal = getattr(turn, "signal", None)
    semantic_event = getattr(signal, "semantic_event", None)
    with _METRICS_LOCK:
        _METRICS[prepared.session_id] = {
            "rendered_prompt_tokens": prepared.prompt_plan.rendered_prompt_tokens,
            "backend_prompt_tokens": timing.prompt_tokens if timing else 0,
            "prompt_counting_method": prepared.prompt_plan.counting_method,
            "prompt_count_is_exact": prepared.prompt_plan.token_count_is_exact,
            "prompt_debug": prepared.prompt_plan.debug_metadata(),
            "code_context_attached": prepared.code_context_attached,
            "prior_mood": trace.previous if trace else "",
            "baseline_persistent_mood": (
                trace.baseline_persistent_mood if trace else ""
            ),
            "prior_persistent_mood": trace.prior_persistent_mood if trace else "",
            "resulting_persistent_mood": (
                trace.resulting_persistent_mood if trace else ""
            ),
            "detected_signal": (
                trace.immediate if trace else ""
            ),
            "contextual_reaction": signal.contextual_reaction.kind if signal else "",
            "signal_confidence": trace.signal_confidence if trace else 0.0,
            "activation_threshold": trace.activation_threshold if trace else 0.0,
            "reaction_mapping": trace.reaction_mapping if trace else "none",
            "state_changes": ", ".join(trace.state_changes) if trace else "",
            "candidate_mood_delta": trace.candidate_mood_delta if trace else "none",
            "active_emotion": (
                trace.candidate if trace else ""
            ),
            "decay_applied": trace.decay_applied if trace else 0.0,
            "elapsed_seconds": trace.elapsed_seconds if trace else 0.0,
            "elapsed_decay_delta": trace.elapsed_decay_delta if trace else (),
            "circadian_target": trace.circadian_target if trace else 0.0,
            "circadian_delta": trace.circadian_delta if trace else (),
            "ambient_delta": trace.ambient_delta if trace else (),
            "event_delta": trace.event_delta if trace else (),
            "conversation_delta": trace.conversation_delta if trace else (),
            "short_emotion_change": (
                trace.short_emotion_change if trace else "none"
            ),
            "event_ids": trace.event_ids if trace else (),
            "reason_codes": (
                tuple(
                    dict.fromkeys(
                        (
                            *trace.reason_codes,
                            *(("state_not_committed",) if not committed else ()),
                        )
                    )
                )
                if trace
                else (("state_not_committed",) if not committed else ())
            ),
            "life_trace": life_trace,
            "semantic_event": asdict(semantic_event) if semantic_event is not None else {},
            "completion_appraisal": bool(getattr(signal, "task_success", False)),
            "memory_trace": memory_trace,
            "response_intention": asdict(response_intention),
            "style_compiler": asdict(compiled_style),
            "validation_results": validation.violations if validation else (),
            "validation_evidence": validation.evidence if validation else (),
            "final_disposition": trace.final_disposition if trace else "",
            "grounded_activity_source": turn.grounded_activity_source,
            "grounded_activity_age_seconds": turn.grounded_activity_age_seconds,
            "dynamic_state_chars": len(prepared.turn_context.behavioral_summary),
            "state_schema_version": turn.state.version,
            "committed": committed,
            "interface": prepared.chat_input.source,
            "updated_at": time.time(),
        }
        if len(_METRICS) > _MAX_METRICS:
            oldest = min(
                _METRICS,
                key=lambda key: float(_METRICS[key].get("updated_at") or 0.0),
            )
            _METRICS.pop(oldest, None)


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


def _legacy_timing_log(
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
    model_seconds = max(0.0, timing.model_finished_at - timing.model_started_at)
    decode_seconds = max(0.0, timing.model_finished_at - timing.first_token_at)
    prompt_eval_seconds = max(
        0.0,
        timing.first_token_at
        - timing.model_started_at
        - timing.chat_template_seconds
        - timing.prompt_tokenization_seconds,
    )
    fields = [
        "[Akane:timing]",
        f"total={done - prepared.started_at:.3f}s",
        f"preprocess={prepared.preprocess_seconds:.3f}s",
        f"prompt={prepared.prompt_seconds:.3f}s",
        f"memory={prepared.memory_seconds:.3f}s",
        f"queue={max(0.0, timing.model_started_at - timing.requested_at):.3f}s",
        f"chat_template={timing.chat_template_seconds:.3f}s",
        f"tokenization={timing.prompt_tokenization_seconds:.3f}s",
        f"prompt_eval={prompt_eval_seconds:.3f}s",
        f"model_total={model_seconds:.3f}s",
        f"postprocess={postprocess_seconds:.3f}s",
        f"persistence={persistence_seconds:.3f}s",
        f"prompt_tokens_rendered={prepared.prompt_plan.rendered_prompt_tokens or 0}",
        f"prompt_tokens_backend={timing.prompt_tokens}",
        f"output_chars={len(reply)}",
    ]
    if decode_seconds > 0.0:
        fields.append(f"output_chars_per_second={len(reply) / decode_seconds:.2f}")
    if output_chunks:
        fields.append(f"stream_chunks={output_chunks}")
    if timing.first_token_at:
        fields.insert(1, f"first_token={timing.first_token_at - prepared.started_at:.3f}s")
    if first_delivery_at:
        fields.insert(2, f"first_delivery={first_delivery_at - prepared.started_at:.3f}s")
    print(" ".join(fields), flush=True)


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


def _debug_number(value: object) -> str:
    return "None" if value is None else str(int(float(value)))


def _debug_quoted(value: object) -> str:
    text = _debug_text(value, 80)
    return f'"{text.replace(chr(34), chr(39))}"' if text else "None"


def _debug_disposition(value: object) -> str:
    text = _debug_text(value, 80)
    while text.lower().startswith("disposition:"):
        text = text.partition(":")[2].strip()
    text = text.rstrip(". ")
    return text.capitalize() if text else "None"


def _debug_scored_parts(value: object) -> tuple[str, float]:
    text = _debug_text(value, 80)
    if _debug_absent(text):
        return "", 0.0
    label, separator, raw_score = text.rpartition(" ")
    if separator:
        try:
            return label, float(raw_score)
        except ValueError:
            pass
    return text, 0.0


def _debug_scored_value(value: object) -> str:
    label, score = _debug_scored_parts(value)
    if not label or score <= 0.0:
        return "None"
    name = "Concerned" if label == "concern" else _debug_name(label)
    return f"{name} ({score:.2f})"


def _debug_reaction(contextual: object, detected: object) -> str:
    contextual_text = _debug_text(contextual, 48)
    detected_label, score = _debug_scored_parts(detected)
    label = contextual_text if not _debug_absent(contextual_text) else detected_label
    if _debug_absent(label) or score <= 0.0:
        return "None" if _debug_absent(label) else _debug_name(label)
    return f"{_debug_name(label)} ({score:.2f})"


def _debug_mood_changes(changes: object, decay: object) -> tuple[str, ...]:
    if isinstance(changes, (list, tuple)):
        items = [str(item) for item in changes]
    else:
        items = str(changes or "").split(",")
    rendered: list[str] = []
    for item in items:
        name, separator, raw_delta = item.strip().rpartition(" ")
        if not separator:
            continue
        try:
            delta = float(raw_delta)
        except ValueError:
            continue
        if abs(delta) >= 0.005:
            rendered.append(f"{_debug_name(name)}: {delta:+.2f}")
    decay_value = float(decay or 0.0)
    if decay_value >= 0.01:
        rendered.append(f"Decay Applied: {decay_value:.2f}")
    return tuple(rendered)


def _debug_duration(value: object) -> str:
    total = max(0, int(round(float(value or 0.0))))
    if total < 60:
        return f"{total} second{'s' if total != 1 else ''}"
    minutes = total // 60
    if minutes < 60:
        seconds = total % 60
        parts = [f"{minutes} minute{'s' if minutes != 1 else ''}"]
        if seconds:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        return ", ".join(parts)
    hours, minutes = divmod(minutes, 60)
    parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return ", ".join(parts)


def _debug_delta_lines(value: object, *, minimum: float = 0.005) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rendered: list[str] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            delta = float(item[1])
        except (TypeError, ValueError):
            continue
        if abs(delta) >= minimum:
            rendered.append(f"{_debug_name(item[0])} {delta:+.2f}")
    return tuple(rendered)


def _debug_delta_text(value: object) -> str:
    return ", ".join(_debug_delta_lines(value, minimum=0.0)) or "None"


def _debug_activity_description(value: object) -> str:
    if not isinstance(value, dict):
        return "None"
    description = _debug_text(value.get("description"), 100)
    return description[:1].upper() + description[1:] if description else "None"


def _debug_pairs_text(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "None"
    pairs = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                rendered = f"{float(item[1]):.3f}"
            except (TypeError, ValueError):
                rendered = _debug_name(item[1])
            pairs.append(f"{_debug_name(item[0])}={rendered}")
    return ", ".join(pairs) or "None"


def _debug_generated_activities(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "None"
    activities = []
    for item in value:
        if not isinstance(item, dict):
            continue
        description = _debug_text(item.get("description"), 60)
        event_id = _debug_text(item.get("event_id"), 20)
        status = _debug_name(item.get("status"))
        if description:
            started = float(item.get("started_at") or 0.0)
            completed = float(item.get("completed_at") or 0.0)
            activities.append(
                f"{description} [{status}; {event_id or 'no ID'}; "
                f"{started:.0f}->{completed:.0f}]"
            )
    return "; ".join(activities) or "None"


def _debug_source_names(sources: tuple | list) -> tuple[str, ...]:
    names: list[str] = []
    for source in sources:
        raw = str(source or "").strip()
        name = _DEBUG_SOURCE_LABELS.get(raw, _debug_name(raw))
        if name not in names:
            names.append(name)
    return tuple(names)


def _debug_trimmed_sources(trimmed: tuple | list) -> tuple[str, ...]:
    names: list[str] = []
    for item in trimmed:
        raw = str(item or "").strip()
        source, separator, reason = raw.partition(":")
        name = _DEBUG_SOURCE_LABELS.get(source, _debug_name(source))
        if separator and reason:
            name += f" ({_debug_name(reason)})"
        if name not in names:
            names.append(name)
    return tuple(names)


def _debug_prompt_tokens(metrics: dict[str, object]) -> tuple[int | None, str]:
    backend = int(metrics.get("backend_prompt_tokens") or 0)
    rendered = metrics.get("rendered_prompt_tokens")
    if backend > 0:
        prompt_tokens: int | None = backend
    elif rendered is not None:
        prompt_tokens = int(rendered)
    else:
        prompt_tokens = None
    exact = bool(metrics.get("prompt_count_is_exact")) and (
        backend > 0 or rendered is not None
    )
    return prompt_tokens, "Exact" if exact else "None"


def _debug_internal_details(
    metrics: dict[str, object],
    prompt_debug: dict,
    versions: dict,
    source_roles: dict,
    category_tokens: dict,
    sources: tuple | list,
    trimmed: tuple | list,
    akane: dict,
) -> tuple[str, ...]:
    rendered_tokens = metrics.get("rendered_prompt_tokens")
    backend_tokens = int(metrics.get("backend_prompt_tokens") or 0)
    life = metrics.get("life_trace")
    life = life if isinstance(life, dict) else {}
    previous_life = life.get("previous_activity")
    current_life = life.get("background_activity") or life.get("current_activity")
    memory_trace = metrics.get("memory_trace")
    memory_trace = memory_trace if isinstance(memory_trace, dict) else {}
    intention = metrics.get("response_intention")
    intention = intention if isinstance(intention, dict) else {}
    style = metrics.get("style_compiler")
    style = style if isinstance(style, dict) else {}
    validation = metrics.get("validation_results")
    validation = validation if isinstance(validation, (list, tuple)) else ()
    validation_evidence = metrics.get("validation_evidence")
    validation_evidence = (
        validation_evidence if isinstance(validation_evidence, (list, tuple)) else ()
    )
    semantic_event = metrics.get("semantic_event")
    semantic_event = semantic_event if isinstance(semantic_event, dict) else {}
    return (
        "",
        "Internal Details",
        "  Personality: Stable Akane",
        "  Candidate Status: "
        + (
            "Proposed, not committed"
            if not metrics.get("committed")
            else "Committed"
        ),
        "  Mood Baseline Vector: "
        f"{_debug_text(metrics.get('baseline_persistent_mood'), 160) or 'None'}",
        f"  Prior Mood: {_debug_name(metrics.get('prior_mood'))}",
        "  Persistent Mood Vector: "
        f"{_debug_text(metrics.get('prior_persistent_mood'), 160) or 'None'}",
        "  Resulting Persistent Vector: "
        f"{_debug_text(metrics.get('resulting_persistent_mood'), 160) or 'None'}",
        f"  Elapsed Time: {_debug_duration(metrics.get('elapsed_seconds'))}",
        "  Elapsed Decay Delta: "
        f"{_debug_delta_text(metrics.get('elapsed_decay_delta'))}",
        "  Circadian Target: "
        f"{float(metrics.get('circadian_target') or 0.0):.3f}",
        f"  Circadian Delta: {_debug_delta_text(metrics.get('circadian_delta'))}",
        f"  Ambient Delta: {_debug_delta_text(metrics.get('ambient_delta'))}",
        f"  Event Delta: {_debug_delta_text(metrics.get('event_delta'))}",
        "  Conversation Delta: "
        f"{_debug_delta_text(metrics.get('conversation_delta'))}",
        "  Detected Signal: "
        f"{_debug_text(metrics.get('detected_signal'), 80) or 'None'}",
        "  Semantic Event: "
        f"subject={_debug_text(semantic_event.get('subject'), 100) or 'None'}; "
        f"type={_debug_name(semantic_event.get('event_type'))}; "
        f"status={_debug_name(semantic_event.get('status'))}; "
        f"actor={_debug_name(semantic_event.get('actor'))}; "
        f"target={_debug_text(semantic_event.get('target'), 100) or 'None'}; "
        f"temporal={_debug_name(semantic_event.get('temporal_state'))}; "
        f"negated={_debug_bool(semantic_event.get('negated'))}; "
        f"confidence={float(semantic_event.get('confidence') or 0.0):.3f}",
        f"  Signal Confidence: {float(metrics.get('signal_confidence') or 0.0):.3f}",
        "  Activation Threshold: "
        f"{float(metrics.get('activation_threshold') or 0.0):.3f}",
        "  Contextual Reaction: "
        f"{_debug_name(metrics.get('contextual_reaction'))}",
        f"  Reaction Mapping: {_debug_text(metrics.get('reaction_mapping'), 80) or 'None'}",
        "  Exact Mood Deltas: "
        f"{_debug_text(metrics.get('state_changes'), 160) or 'None'}",
        "  Candidate Mood Delta: "
        f"{_debug_name(metrics.get('candidate_mood_delta'))}",
        f"  Active Short Emotion: {_debug_scored_value(metrics.get('active_emotion'))}",
        "  Short Emotion Change: "
        f"{_debug_name(metrics.get('short_emotion_change'))}",
        f"  Decay Amount: {float(metrics.get('decay_applied') or 0.0):.3f}",
        f"  Event IDs: {_debug_items(metrics.get('event_ids')) or 'None'}",
        f"  Reason Codes: {_debug_items(metrics.get('reason_codes')) or 'None'}",
        "  Dynamic State Tokens: "
        f"{int(metrics.get('dynamic_state_tokens') or 0)}",
        "  Offscreen Previous Activity: "
        f"{_debug_activity_description(previous_life)}",
        "  Offscreen Generated Activities: "
        f"{_debug_generated_activities(life.get('generated_activities'))}",
        "  Offscreen Current Activity: "
        f"{_debug_activity_description(current_life)}",
        "  Offscreen Selection Bucket: "
        f"{_debug_number(life.get('selection_bucket'))}",
        "  Offscreen Selection Candidates: "
        f"{_debug_pairs_text(life.get('selection_candidates'))}",
        "  Offscreen Repetition Penalties: "
        f"{_debug_pairs_text(life.get('repetition_penalties'))}",
        f"  Offscreen Cooldowns: {_debug_items(life.get('cooldowns')) or 'None'}",
        "  Offscreen Needs Before: "
        f"{_debug_mapping(life.get('needs_before')) or 'None'}",
        "  Offscreen Needs After: "
        f"{_debug_mapping(life.get('needs_after')) or 'None'}",
        "  Offscreen Mood Effects: "
        f"{_debug_delta_text(life.get('mood_effects'))}",
        "  Offscreen Preference Effects: "
        f"{_debug_delta_text(life.get('preference_changes'))}",
        "  Offscreen No-Activity Reason: "
        f"{_debug_name(life.get('reason_no_activity'))}",
        f"  Offscreen Authority: {_debug_name(life.get('authority_source'))}",
        "  Offscreen Last Processed: "
        f"{float(life.get('last_processed_at') or 0.0):.3f}",
        "  Offscreen Next Opportunity: "
        f"{float(life.get('next_opportunity_at') or 0.0):.3f}",
        "  Memory Migration Version: "
        f"{int(memory_trace.get('migration_version') or 0)}",
        "  Intention Direct Request: "
        f"{_debug_bool(intention.get('direct_request'))}",
        "  Intention Active Thread: "
        f"{_debug_bool(intention.get('active_thread'))}",
        "  Intention Relationship Safe: "
        f"{_debug_bool(intention.get('relationship_safe'))}",
        "  Intention Prompt Representation: "
        f"Goal={_debug_name(intention.get('primary'))}; "
        f"Optional={_debug_name(intention.get('optional_behavior'))}; "
        f"Continuity={_debug_name(intention.get('continuity'))}; "
        f"Grounding={_debug_name(intention.get('grounding'))}",
        "  Style Directive Values: "
        f"{_debug_style_directives(style.get('directives')) or 'None'}",
        "  Style Resolved Gates: "
        f"question={_debug_name(style.get('question_gate'))}; "
        f"humor={_debug_name(style.get('humor'))}",
        "  Style Validation Limits: "
        f"{_debug_items(style.get('validation_limits')) or 'None'}",
        "  Raw Validation Results: "
        f"{_debug_items(validation) or 'None'}",
        "  Validation Evidence: "
        f"{_debug_style_directives(validation_evidence) or 'None'}",
        "  State Schema Version: "
        f"{int(metrics.get('state_schema_version') or akane.get('state_schema_version') or 0)}",
        "  Prompt Builder Version: "
        f"{_debug_text(prompt_debug.get('prompt_builder_version'), 16) or 'None'}",
        f"  Identity Hash: {_debug_text(versions.get('identity'), 16) or 'None'}",
        f"  Soul Hash: {_debug_text(versions.get('soul'), 16) or 'None'}",
        "  Hard Rules Hash: "
        f"{_debug_text(versions.get('hard_constraints'), 16) or 'None'}",
        f"  Raw Prompt Sources: {_debug_items(sources) or 'None'}",
        f"  Raw Source Role Mappings: {_debug_mapping(source_roles) or 'None'}",
        f"  Category Token Counts: {_debug_mapping(category_tokens) or 'None'}",
        "  Rendered Token Count: "
        f"{rendered_tokens if rendered_tokens is not None else 'None'}",
        "  Backend Observed Token Count: "
        f"{backend_tokens if backend_tokens else 'None'}",
        "  Formatter: "
        f"{_debug_text(metrics.get('prompt_counting_method'), 80) or 'None'}",
        f"  Raw Trimmed Sources: {_debug_items(trimmed) or 'None'}",
        "  Code Context Attached: "
        f"{_debug_bool(metrics.get('code_context_attached'))}",
        f"  Commit Status: {_debug_bool(metrics.get('committed'))}",
    )


def _debug_items(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return ", ".join(_debug_text(item, 64) for item in value if str(item or "").strip())


def _debug_mapping(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(
        f"{_debug_text(key, 32)}={_debug_text(item, 32)}"
        for key, item in value.items()
    )


def _debug_style_directives(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "; ".join(
        f"{_debug_text(item[0], 24)}={_debug_text(item[1], 100)}"
        for item in value
        if isinstance(item, (list, tuple)) and len(item) == 2
    )


# The active debug path is deliberately compact; legacy formatting helpers above
# remain private for compatibility with older callers.
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
    loaded_emotion = akane.get("emotion") or {}
    with _METRICS_LOCK:
        metrics = dict(_METRICS.get(conversation, {}))
    runtime = ModelManager.get_instance().runtime_report(include_model_hash=verbose)
    prompt = metrics.get("prompt_debug") or {}
    versions = prompt.get("persona_versions") or {}
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
        f"  Boundary Level: {int(metrics.get('boundary_level') or 0)}",
        f"  Applied: {_debug_bool(metrics.get('emotion_applied'))}",
        f"  Committed: {_debug_bool(metrics.get('committed'))}",
        "",
        "Conversation",
        f"  Complete Pairs: {int(metrics.get('complete_pairs') or 0)}",
        f"  Role Sequence: {metrics.get('role_sequence') or 'None'}",
        f"  Current Message Count: {int(metrics.get('current_message_count') or 0)}",
        "",
        "Persistence",
        f"  Profile ID: {profile}",
        f"  Conversation ID: {conversation}",
        f"  JSON Paths: {MEMORY_PATH.expanduser().resolve()} | {LONG_TERM_MEMORY_PATH.expanduser().resolve()}",
        f"  Save Result: {'success' if metrics.get('committed') else 'not committed'}",
        "",
        "Prompt",
        f"  Exact Tokens: {prompt.get('exact_tokens') if prompt.get('exact_tokens') is not None else 'None'}",
        f"  Context Window: {prompt.get('context_window') or runtime['context_window']}",
        f"  Trimmed Content: {', '.join(str(item) for item in trimmed) if trimmed else 'None'}",
        "",
        "Runtime",
        f"  Model: {runtime['model_path']} ({runtime['model_size']} bytes)",
        f"  Model SHA-256: {runtime['model_sha256']}",
        f"  llama-cpp-python: {runtime['llama_cpp_python']}",
        f"  Chat Template: {runtime['chat_template']}",
        f"  enable_thinking: {runtime['enable_thinking']}",
        "  Sampling: "
        f"temperature={runtime['temperature']} top_p={runtime['top_p']} "
        f"top_k={runtime['top_k']} min_p={runtime['min_p']} "
        f"repeat_penalty={runtime['repeat_penalty']} seed={runtime['seed']}",
        f"  Identity Hash: {versions.get('identity') or 'None'}",
        f"  Soul Hash: {versions.get('soul') or 'None'}",
        f"  Hard-rules Hash: {versions.get('hard_constraints') or 'None'}",
    ]
    return "\n".join(lines)


def _remember_metrics(
    prepared: TurnPreparation,
    *,
    committed: bool,
    timing: InferenceTiming | None = None,
    validation: ResponseValidation | None = None,
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
            "boundary_level": signal.emotion_state.boundary_level,
            "emotion_applied": signal.emotion_applied,
            "complete_pairs": sum(item.get("role") == "assistant" for item in history),
            "role_sequence": ",".join(item.get("role", "unknown") for item in messages),
            "current_message_count": int(
                bool(messages)
                and messages[-1].get("role") == "user"
                and messages[-1].get("content") == prepared.chat_input.text
            ),
            "committed": committed,
            "exact_prompt_tokens": (
                timing.prompt_tokens if timing and timing.prompt_tokens
                else prepared.prompt_plan.rendered_prompt_tokens
            ),
            "updated_at": time.time(),
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

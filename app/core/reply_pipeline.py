"""Shared single-generation pipeline with grounding, cancellation, and safe commit."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass

from app.core.character import load_character_profile
from app.core.config import PROMPT_DEBUG, STREAM_CHUNK_CHARS, STREAM_FLUSH_SECONDS
from app.core.life import life_evolution_debug
from app.core.model_loader import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceTiming,
    ModelManager,
)
from app.core.memory import (
    apply_opinion_candidate,
    estimate_tokens,
    extract_opinion_candidate,
)
from app.core.prompt import describe_model_input
from app.core.signal import topic_overlap
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
_UNSUPPORTED_OPINION_HISTORY = re.compile(
    r"\b(?:I(?:'ve| have) always (?:thought|believed|preferred)|"
    r"I (?:used to|long) (?:think|believe|prefer)|for years I(?:'ve| have))\b",
    re.IGNORECASE,
)
_INTERNAL_TERMS = re.compile(
    r"\b(?:affect core|dynamic guidance|persistent mood|response intention|"
    r"style compiler|system prompt|my internal state|my memory system|my response "
    r"selection|my prompt processing|processing data|analyzing inputs|monitoring|"
    r"waiting for requests|running calculations|ran calculations|"
    r"(?:processing|processed) (?:conversation )?patterns|"
    r"(?:analyzing|analyzed) prompts|(?:generating|generated) (?:a |my )?"
    r"repl(?:y|ies)|tracking token "
    r"probabilities|updating embeddings|inspecting hidden state)\b",
    re.IGNORECASE,
)
_OFFSCREEN_ACTIVITY_REQUEST = re.compile(
    r"\b(?:what (?:have|had) you been (?:doing|thinking about)|what (?:did|were) you do(?:ing)?|"
    r"been up to|anything happen|time alone|did you work on anything|"
    r"what happened since|how was your (?:time|day)|how(?:'s| is) life)\b",
    re.IGNORECASE,
)
_TECHNICAL_DISCUSSION = re.compile(
    r"\b(?:code|implementation|model|prompt|inference|system|architecture|"
    r"validator|debug|technical)\b",
    re.IGNORECASE,
)
_OFFSCREEN_ACTIVITY_CLAIM = re.compile(
    r"\bi\s+(?:(?:have|'ve)\s+been|was|had been)\s+(?:"
    r"reading|researching|writing|drawing|coding|watching|listening|talking|"
    r"visiting|practicing|working on|thinking through|reviewing|processing|"
    r"analyzing|generating|checking|using|accessing|browsing|searching)\b|"
    r"\bi\s+(?:processed|analyzed|generated|reviewed|checked|used|accessed|"
    r"browsed|searched|coded|wrote|drew|read|researched|watched|listened)\b",
    re.IGNORECASE,
)
_NO_ACTIVITY_GROUNDING = re.compile(
    r"no meaningful activity was established during the gap",
    re.IGNORECASE,
)
_LEADING_ROLE_MARKER = re.compile(
    r"\A\s*(?:\[\s*Akane\s*\]|Akane\s*:|\[\s*assistant\s*\]|Assistant\s*:)\s*",
    re.IGNORECASE,
)
_PROVENANCE_MARKER = re.compile(
    r"(?im)^\s*\[(?:RECENT ASSISTANT MESSAGE|EARLIER ASSISTANT EXCERPT)\s+"
    r"[—-]\s+UNVERIFIED\]\s*(?:\n|$)"
)
_PERSONAL_JUDGMENT = re.compile(
    r"\b(?:I\s+(?:personally\s+)?(?:choose|prefer|want|would rather|lean)|"
    r"I\s+(?:think|believe|feel|find|consider)\b[^.!?]{0,100}\b"
    r"(?:better|worse|worthwhile|preferable|would choose|would prefer)|"
    r"my (?:choice|preference|view|position)(?:\s+is)?|"
    r"I(?:'m| am) (?:uncertain|torn|not sure))\b",
    re.IGNORECASE,
)
_GENERIC_NEUTRALITY = re.compile(
    r"\b(?:both (?:options|sides|forms) (?:have|offer)|each (?:has|offers)|"
    r"no (?:objectively |universally )?(?:better|worse|right) (?:choice|option|answer)|"
    r"simply different|it depends on (?:your|one's) priorities)\b",
    re.IGNORECASE,
)
_CLINICAL_REFLECTIVE_TERMS = re.compile(
    r"\b(?:interesting variable|collected patterns|new iteration|operate on|"
    r"flow of continuous thought|processing data|calculated response|"
    r"objective comparison|system objective|experimental condition|data flow)\b",
    re.IGNORECASE,
)
_UNCERTAIN_JUDGMENT = re.compile(
    r"\b(?:I(?:'m| am) (?:uncertain|torn|not sure|undecided)|"
    r"I (?:do not|don't) have a (?:settled|clear) (?:view|choice|preference))\b",
    re.IGNORECASE,
)
_SPECIFIC_UNCERTAINTY = re.compile(
    r"\b(?:because|unless|until|whether|depends on|unresolved|"
    r"not sure (?:about|if)|torn between|do not know (?:if|whether)|"
    r"don't know (?:if|whether))\b[^.!?]{2,160}",
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


def clean_generated_reply(reply: str) -> str:
    """Remove only known generated structural markers, leaving ordinary prose intact."""

    text = _PROVENANCE_MARKER.sub("", str(reply or ""))
    previous = None
    while text != previous:
        previous = text
        text = _LEADING_ROLE_MARKER.sub("", text, count=1)
    return text.strip()


class _StreamingOutputCleaner:
    """Hold only the ambiguous leading bytes so role labels cannot reach deltas."""

    _prefixes = (
        "[akane]",
        "akane:",
        "[assistant]",
        "assistant:",
        "[recent assistant message — unverified]",
        "[recent assistant message - unverified]",
        "[earlier assistant excerpt — unverified]",
        "[earlier assistant excerpt - unverified]",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._resolved = False

    def feed(self, chunk: str) -> str:
        if self._resolved:
            return chunk
        self._buffer += chunk
        return self._resolve(final=False)

    def finish(self) -> str:
        return "" if self._resolved else self._resolve(final=True)

    def _resolve(self, *, final: bool) -> str:
        value = self._buffer.lstrip()
        while value:
            role_match = _LEADING_ROLE_MARKER.match(value)
            if role_match is not None:
                value = value[role_match.end() :].lstrip()
                continue
            provenance_match = _PROVENANCE_MARKER.match(value)
            if provenance_match is not None:
                value = value[provenance_match.end() :].lstrip()
                continue
            break
        lowered = value.casefold()
        if not final and (not value or any(prefix.startswith(lowered) for prefix in self._prefixes)):
            self._buffer = value
            return ""
        self._buffer = ""
        self._resolved = True
        return value


def validate_response_style(
    reply: str,
    style: CompiledStyle,
    intention: ResponseIntention,
    *,
    grounding_context: str = "",
    recent_outputs: tuple[str, ...] = (),
    persona_text: str = "",
    request_context: str = "",
    opinion_history_available: bool = False,
    question_mode: str = "none",
    active_opinion: object | None = None,
    active_conflict: bool = False,
    changed_condition: str = "",
    affected_reason: str = "",
    reconsideration_warranted: bool = False,
    profile_reset_cleared_scopes: tuple[str, ...] = (),
) -> ResponseValidation:
    """Report deterministic style and grounding risks without rewriting output."""

    text = str(reply or "").strip()
    violations: list[str] = []
    evidence: list[tuple[str, str]] = []
    personal_activity_question = bool(_OFFSCREEN_ACTIVITY_REQUEST.search(request_context))
    technical_discussion = bool(_TECHNICAL_DISCUSSION.search(request_context))

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
    if internal_term and (personal_activity_question or not technical_discussion):
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
    offscreen_claim = _OFFSCREEN_ACTIVITY_CLAIM.search(text)
    if personal_activity_question and offscreen_claim:
        fragment = re.split(r"[.!?\n]", text[offscreen_claim.start() :], maxsplit=1)[0]
        claim_terms = set(_normalized_words(fragment)) - _VALIDATION_STOPWORDS - {
            "been", "had", "have", "i", "was",
        }
        grounding_terms = set(_normalized_words(akane_grounding))
        if _NO_ACTIVITY_GROUNDING.search(akane_grounding):
            finding("unrecorded offscreen activity", fragment)
        elif not akane_grounding.strip() or (
            claim_terms and claim_terms.isdisjoint(grounding_terms)
        ):
            finding("unrecorded offscreen activity", fragment)
    activity = None if personal_activity_question else _ACTIVITY_CLAIM.search(text)
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
    unsupported_history = _UNSUPPORTED_OPINION_HISTORY.search(text)
    if unsupported_history and not opinion_history_available:
        finding("unsupported opinion history", unsupported_history.group(0))

    leading_role = _LEADING_ROLE_MARKER.match(text)
    if leading_role:
        finding("leading speaker-label leak", leading_role.group(0))
    provenance = _PROVENANCE_MARKER.search(text)
    if provenance:
        finding("provenance-marker leak", provenance.group(0))
    if question_mode in {"personal_choice", "hypothetical_choice"}:
        personal_judgment = _PERSONAL_JUDGMENT.search(text)
        if not personal_judgment:
            finding("personal choice answered as objective comparison", "no personal stance")
        if active_opinion is not None and not personal_judgment and not reconsideration_warranted:
            finding("active opinion omitted without reconsideration", "stored view not consumed")
        neutrality = _GENERIC_NEUTRALITY.search(text)
        uncertainty = _UNCERTAIN_JUDGMENT.search(text)
        specific_uncertainty = bool(_SPECIFIC_UNCERTAINTY.search(text))
        if active_conflict and neutrality and (
            not personal_judgment or uncertainty and not specific_uncertainty
        ):
            finding("conflict forced artificial neutrality", neutrality.group(0))
        if uncertainty and not specific_uncertainty:
            finding("genuine uncertainty lacks unresolved condition", uncertainty.group(0))
        clinical = _CLINICAL_REFLECTIVE_TERMS.search(text)
        if clinical:
            finding("excessive clinical framing", clinical.group(0))
    if reconsideration_warranted:
        condition_terms = set(_normalized_words(changed_condition))
        response_terms = set(_normalized_words(text))
        acknowledges_condition = bool(
            condition_terms and condition_terms & response_terms
            or affected_reason and affected_reason.casefold() in text.casefold()
        )
        if not acknowledges_condition:
            finding("material hypothetical condition ignored", changed_condition or affected_reason)
    protected_reset_scopes = {
        "opinions",
        "values",
        "conflicts",
        "self_events",
        "offscreen_life",
        "identity_state",
    }
    cleared_self_state = tuple(
        scope
        for item in profile_reset_cleared_scopes
        if (scope := str(item).strip().lower()) in protected_reset_scopes
    )
    if cleared_self_state:
        finding(
            "user-profile reset cleared Akane self-state",
            ", ".join(cleared_self_state),
        )

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
            ModelManager.get_instance().count_prompt_tokens if exact_tokens else None
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
        opinion_history_available=any(
            opinion.status == "superseded"
            and opinion.topic_key == prepared.turn_context.opinion_topic_key
            for opinion in prepared.internal_turn.state.opinions
        ),
        question_mode=prepared.internal_turn.signal.semantic_event.question_mode,
        active_opinion=prepared.turn_context.active_opinion,
        active_conflict=bool(
            prepared.turn_context.active_internal_conflict is not None
            and prepared.turn_context.active_internal_conflict.status == "active"
        ),
        changed_condition=prepared.turn_context.changed_condition,
        affected_reason=prepared.turn_context.affected_reason,
        reconsideration_warranted=prepared.turn_context.reconsideration_warranted,
    )
    opinion_candidate = None
    if not validation.violations:
        life = prepared.internal_turn.state.life
        opinion_activity = next(
            (
                activity
                for activity in (
                    life.activity,
                    *reversed(life.recent_events),
                )
                if activity is not None
                and activity.source != "conversation"
                and topic_overlap(
                    f"{activity.subject} {activity.description}",
                    prepared.chat_input.text,
                )
                >= 0.25
            ),
            None,
        )
        opinion_candidate = extract_opinion_candidate(
            user_text=prepared.chat_input.text,
            response=reply,
            intention=prepared.turn_context.response_intention.primary,
            semantic_subject=(
                prepared.turn_context.active_opinion.subject
                if prepared.turn_context.reconsideration_warranted
                and prepared.turn_context.active_opinion is not None
                else prepared.internal_turn.signal.semantic_event.subject
            ),
            signal_topic=prepared.internal_turn.signal.topic,
            now=prepared.chat_input.timestamp,
            grounded_experience=opinion_activity is not None,
            autonomous=prepared.chat_input.autonomous,
            reconsideration_warranted=prepared.turn_context.reconsideration_warranted,
            changed_condition=prepared.turn_context.changed_condition,
            affected_reason=prepared.turn_context.affected_reason,
            reason_effect=prepared.turn_context.reason_effect,
        )
        if opinion_candidate is not None:
            _candidate_state, transition = apply_opinion_candidate(
                prepared.internal_turn.state,
                opinion_candidate,
            )
            if transition.get("commit_status") == "change rejected":
                violation = (
                    "primary opinion reason removed but stance preserved without another reason"
                    if transition.get("affected_reason")
                    else "active opinion contradicted without reconsideration"
                )
                validation = ResponseValidation(
                    (*validation.violations, violation),
                    (*validation.evidence, (violation, "stored stance remains active")),
                )
                opinion_candidate = None
        elif (
            prepared.turn_context.response_intention.primary
            in {"state opinion", "disagree"}
            and re.search(
                r"\b(?:I (?:think|believe|prefer|favor)|my (?:view|position))\b",
                reply,
                re.I,
            )
        ):
            violation = "generated stance could not be parsed"
            validation = ResponseValidation(
                (*validation.violations, violation),
                (*validation.evidence, (violation, "no opinion candidate")),
            )
    if validation.violations:
        _remember_metrics(
            prepared,
            committed=False,
            timing=timing,
            validation=validation,
        )
        return
    opinion_trace = commit_turn(prepared, reply, opinion_candidate)
    active_opinion = prepared.turn_context.active_opinion
    opinion_trace.update(
        {
            "topic_considered": (
                active_opinion.subject
                if active_opinion is not None
                else prepared.turn_context.opinion_topic_key.replace("-", " ")
            ),
            "active_opinion": active_opinion.target if active_opinion else "",
            "retrieval_used": active_opinion is not None,
        }
    )
    if not opinion_trace.get("relevant_values"):
        opinion_trace["relevant_values"] = tuple(
            value.value_key for value in prepared.turn_context.relevant_values
        )
    conflict = prepared.turn_context.active_internal_conflict
    if conflict is not None and not opinion_trace.get("conflict_topic"):
        opinion_trace.update(
            {
                "conflict_topic": conflict.topic_key,
                "conflict_pulls": (conflict.side_a_value, conflict.side_b_value),
                "conflict_status": conflict.status,
                "conflict_related_opinion": (
                    prepared.turn_context.active_opinion.subject
                    if prepared.turn_context.active_opinion is not None
                    else conflict.topic_key.replace("-", " ")
                ),
                "conflict_resolution": conflict.selected_value,
            }
        )
    if active_opinion is not None:
        opinion_trace.setdefault("stance", active_opinion.stance)
        opinion_trace.setdefault("strength", active_opinion.strength)
        opinion_trace.setdefault("confidence", active_opinion.confidence)
        opinion_trace.setdefault("reason_tags", active_opinion.reason_tags)
    if validation.violations and prepared.turn_context.response_intention.primary in {
        "state opinion",
        "disagree",
    }:
        opinion_trace["commit_status"] = "validation rejected"
    _remember_metrics(
        prepared,
        committed=True,
        timing=timing,
        validation=validation,
        opinion_trace=opinion_trace,
    )


def _reply_events(prepared: TurnPreparation, *, emit_deltas: bool):
    first_delivery_at = 0.0
    output_chunks = 0
    timing = InferenceTiming(requested_at=time.perf_counter())
    output_cleaner = _StreamingOutputCleaner()

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
            max_tokens=prepared.max_tokens,
            cancellation=prepared.handle.cancellation,
            queue_deadline=prepared.handle.queue_deadline,
            timing=timing,
        ):
            output_chunks += 1
            if not timing.first_token_at:
                timing.first_token_at = time.perf_counter()
            parts.append(text)
            visible_text = output_cleaner.feed(text)
            if not emit_deltas:
                continue
            if visible_text:
                pending.append(visible_text)
                pending_chars += len(visible_text)
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

        trailing_text = output_cleaner.finish()
        if emit_deltas and trailing_text:
            pending.append(trailing_text)
            pending_chars += len(trailing_text)
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
        reply = clean_generated_reply("".join(parts))
        if not reply:
            raise RuntimeError("Model returned no visible reply.")
        postprocess_seconds = time.perf_counter() - postprocess_started_at

        prepared.handle.raise_if_cancelled()
        persistence_started_at = time.perf_counter()
        commit_reply(
            prepared,
            reply,
            timing=timing,
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
                "estimated_prompt_tokens": prepared.prompt_plan.estimated_tokens,
                "rendered_prompt_tokens": prepared.prompt_plan.rendered_prompt_tokens,
                "backend_observed_prompt_tokens": timing.prompt_tokens or None,
                "prompt_token_count_is_exact": prepared.prompt_plan.token_count_is_exact,
            },
        )
    except InferenceCancelled as exc:
        raise GenerationCancelled(str(exc)) from exc
    except InferenceQueueTimeout as exc:
        raise GenerationQueueFullError(str(exc)) from exc
    finally:
        finish_turn(prepared)


def debug_state_report(
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
    opinion_trace = metrics.get("opinion_trace")
    opinion_trace = opinion_trace if isinstance(opinion_trace, dict) else {}
    values = akane.get("values")
    values = values if isinstance(values, (list, tuple)) else ()
    conflicts = akane.get("conflicts")
    conflicts = conflicts if isinstance(conflicts, (list, tuple)) else ()
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
    lines.extend(
        (
            "",
            "Opinions",
            f"  Question Mode: {_debug_name(opinion_trace.get('question_mode'))}",
            "  Canonical Topic: "
            f"{_debug_name(opinion_trace.get('canonical_topic') or opinion_trace.get('topic_considered'))}",
            "  Active Opinion Retrieved: "
            f"{_debug_bool(opinion_trace.get('active_opinion_retrieved'))}",
            "  Current Stance: "
            f"{_debug_name(opinion_trace.get('current_position') or opinion_trace.get('current_stance') or opinion_trace.get('stance'))}",
            f"  Strength: {_debug_name(opinion_trace.get('strength'))}",
            f"  Confidence: {_debug_name(opinion_trace.get('confidence'))}",
            "  Relevant Reasons: "
            f"{_debug_items(opinion_trace.get('relevant_reasons') or opinion_trace.get('reason_tags')) or 'None'}",
            "  Relevant Values: "
            f"{_debug_items(opinion_trace.get('planning_relevant_values') or opinion_trace.get('relevant_values')) or 'None'}",
            "  Active Conflict: "
            f"{_debug_items(opinion_trace.get('active_conflict')) or 'None'}",
            "  Choice Requirement: "
            f"{_debug_bool(opinion_trace.get('choice_requirement'))}",
            f"  Retrieval Used: {_debug_bool(opinion_trace.get('retrieval_used'))}",
            "  Candidate Created: "
            f"{_debug_bool(opinion_trace.get('candidate_created'))}",
            "  Candidate Change: "
            f"{_debug_bool(opinion_trace.get('candidate_change'))}",
            f"  Commit Status: {_debug_name(opinion_trace.get('commit_status'))}",
        )
    )
    if opinion_trace.get("candidate_change"):
        lines.extend(
            (
                "  Previous Stance: "
                f"{_debug_name(opinion_trace.get('previous_stance'))}",
                f"  New Stance: {_debug_name(opinion_trace.get('new_stance'))}",
                "  Change Trigger: "
                f"{_debug_name(opinion_trace.get('change_trigger'))}",
                f"  Superseded: {_debug_bool(opinion_trace.get('superseded'))}",
            )
        )
    if opinion_trace.get("question_mode") == "hypothetical_choice":
        lines.extend(
            (
                "  Changed Condition: "
                f"{_debug_name(opinion_trace.get('changed_condition'))}",
                "  Affected Reason: "
                f"{_debug_name(opinion_trace.get('affected_reason'))}",
                f"  Reason Effect: {_debug_name(opinion_trace.get('reason_effect'))}",
                "  Reconsideration Warranted: "
                f"{_debug_bool(opinion_trace.get('reconsideration_warranted'))}",
                "  Previous Stance: "
                f"{_debug_name(opinion_trace.get('previous_position') or opinion_trace.get('previous_stance'))}",
                "  Candidate Stance: "
                f"{_debug_name(opinion_trace.get('candidate_position') or opinion_trace.get('new_stance') or opinion_trace.get('stance'))}",
                "  Candidate Change: "
                f"{_debug_bool(opinion_trace.get('candidate_change'))}",
            )
        )
    lines.extend(
        (
            "",
            "Decision Rendering",
            "  Opinion Consumed by Intention: "
            f"{_debug_bool(opinion_trace.get('opinion_consumed_by_intention'))}",
            "  Opinion Consumed by Style: "
            f"{_debug_bool(opinion_trace.get('opinion_consumed_by_style'))}",
            "  Values Rendered: "
            f"{_debug_items(opinion_trace.get('values_rendered')) or 'None'}",
            "  Conflict Rendered: "
            f"{_debug_bool(opinion_trace.get('conflict_rendered'))}",
            "  Reconsideration Rendered: "
            f"{_debug_bool(opinion_trace.get('reconsideration_rendered'))}",
            "  Clinical-Language Suppression: "
            f"{_debug_bool(opinion_trace.get('clinical_language_suppression'))}",
        )
    )
    relevant_values = opinion_trace.get("relevant_values") or ()
    lines.extend(
        (
            "",
            "Values",
            "  Relevant Values: "
            f"{_debug_items(relevant_values) or 'None'}",
            "  Strengths: "
            + (
                ", ".join(
                    f"{_debug_name(item.get('key'))}={_debug_name(item.get('strength'))}"
                    for item in values
                    if isinstance(item, dict)
                    and item.get("key") in set(relevant_values)
                )
                or "None"
            ),
            "  Update Candidates: "
            f"{_debug_items(opinion_trace.get('value_update_candidates')) or 'None'}",
            "  Commit Status: "
            f"{_debug_name(opinion_trace.get('values_commit_status'))}",
            "",
            "Internal Conflicts",
            f"  Topic: {_debug_name(opinion_trace.get('conflict_topic'))}",
            "  Pulls: "
            f"{_debug_items(opinion_trace.get('conflict_pulls')) or 'None'}",
            f"  Status: {_debug_name(opinion_trace.get('conflict_status'))}",
            "  Related Opinion: "
            f"{_debug_name(opinion_trace.get('conflict_related_opinion'))}",
            f"  Resolution: {_debug_name(opinion_trace.get('conflict_resolution'))}",
            "  Candidate Change: "
            f"{_debug_bool(opinion_trace.get('conflict_candidate_change'))}",
        )
    )
    if verbose and conflicts:
        lines.append(
            "  Active Collection: "
            + "; ".join(
                f"{_debug_name(item.get('topic'))}: {_debug_items(item.get('pulls'))} "
                f"({_debug_name(item.get('status'))})"
                for item in conflicts
                if isinstance(item, dict)
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
            "  Current Interaction: "
            f"{_debug_activity_description(life_metrics.get('current_interaction'))}",
            "  Background Activity: "
            f"{_debug_activity_description(life_metrics.get('background_activity'))}",
            "  Activity Status: "
            f"{_debug_name(background_life.get('status'))}",
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
            "  Recent Reaction: "
            f"{_debug_text(life_metrics.get('recent_reaction'), 80) or 'None'}",
            "  Mood Effect: "
            f"{', '.join(life_mood_effects) if life_mood_effects else 'None'}",
            "  Included In Prompt: "
            f"{_debug_bool(metrics.get('life_context_included'))}",
            "  Absence Constraint Included: "
            f"{_debug_bool(metrics.get('life_absence_constraint_included'))}",
            "  Validation Result: "
            f"{_debug_items(validation) or 'Passed'}",
            "  Commit Status: "
            f"{_debug_bool(metrics.get('committed'))}",
        )
    )
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
    estimate = int(metrics.get("estimated_prompt_tokens") or 0)
    if _debug_material_estimate_difference(estimate, prompt_tokens):
        lines.extend(
            (
                f"  Preflight Estimate: {estimate}",
                f"  Estimate Difference: {abs(estimate - int(prompt_tokens or 0))}",
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


def _remember_metrics(
    prepared: TurnPreparation,
    *,
    committed: bool,
    timing: InferenceTiming | None = None,
    validation: ResponseValidation | None = None,
    opinion_trace: dict[str, object] | None = None,
) -> None:
    turn = prepared.internal_turn
    trace = turn.affect_trace
    life_trace = life_evolution_debug(getattr(turn, "life_evolution", None))
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
    active_opinion = prepared.turn_context.active_opinion
    active_conflict = prepared.turn_context.active_internal_conflict
    style_directives = dict(compiled_style.directives)
    decision_trace: dict[str, object] = {
        "question_mode": getattr(semantic_event, "question_mode", "none"),
        "canonical_topic": prepared.turn_context.opinion_topic_key,
        "active_opinion_retrieved": active_opinion is not None,
        "current_stance": active_opinion.stance if active_opinion is not None else "",
        "current_position": (
            f"{active_opinion.stance} {active_opinion.target}"
            if active_opinion is not None
            else ""
        ),
        "strength": active_opinion.strength if active_opinion is not None else "",
        "confidence": active_opinion.confidence if active_opinion is not None else "",
        "relevant_reasons": (
            tuple(
                reason
                for reason in active_opinion.reason_tags
                if not (
                    prepared.turn_context.reason_effect == "removes"
                    and reason == prepared.turn_context.affected_reason
                )
            )
            if active_opinion is not None
            else ()
        ),
        "relevant_values": tuple(
            value.value_key for value in prepared.turn_context.relevant_values
        ),
        "planning_relevant_values": tuple(
            value.value_key for value in prepared.turn_context.relevant_values
        ),
        "active_conflict": (
            (active_conflict.side_a_value, active_conflict.side_b_value)
            if active_conflict is not None and active_conflict.status == "active"
            else ()
        ),
        "choice_requirement": response_intention.choice_required,
        "changed_condition": prepared.turn_context.changed_condition,
        "affected_reason": prepared.turn_context.affected_reason,
        "reason_effect": prepared.turn_context.reason_effect,
        "reconsideration_warranted": prepared.turn_context.reconsideration_warranted,
        "previous_stance": active_opinion.stance if active_opinion is not None else "",
        "previous_position": (
            f"{active_opinion.stance} {active_opinion.target}"
            if active_opinion is not None
            else ""
        ),
        "opinion_consumed_by_intention": bool(
            active_opinion is not None
            and response_intention.primary in {"state opinion", "disagree"}
        ),
        "opinion_consumed_by_style": "View" in style_directives,
        "values_rendered": (
            tuple(value.value_key for value in prepared.turn_context.relevant_values)
            if "Priorities" in style_directives
            else ()
        ),
        "conflict_rendered": bool(
            active_conflict is not None and "Priorities" in style_directives
        ),
        "reconsideration_rendered": "Reconsideration" in style_directives,
        "clinical_language_suppression": "clinical AI"
        in style_directives.get("Avoid", ""),
    }
    decision_trace.update(opinion_trace or {})
    if decision_trace.get("candidate_created"):
        candidate_stance = str(
            decision_trace.get("new_stance") or decision_trace.get("stance") or ""
        )
        candidate_target = str(decision_trace.get("candidate_target") or "")
        decision_trace["candidate_position"] = " ".join(
            value for value in (candidate_stance, candidate_target) if value
        )
    with _METRICS_LOCK:
        _METRICS[prepared.session_id] = {
            "estimated_prompt_tokens": prepared.prompt_plan.estimated_tokens,
            "rendered_prompt_tokens": prepared.prompt_plan.rendered_prompt_tokens,
            "backend_prompt_tokens": timing.prompt_tokens if timing else 0,
            "prompt_counting_method": prepared.prompt_plan.counting_method,
            "prompt_count_is_exact": prepared.prompt_plan.token_count_is_exact,
            "prompt_debug": prepared.prompt_plan.debug_metadata(),
            "code_context_attached": prepared.code_context_attached,
            "prior_mood": trace.prior_mood if trace else "",
            "baseline_persistent_mood": (
                trace.baseline_persistent_mood if trace else ""
            ),
            "prior_persistent_mood": trace.prior_persistent_mood if trace else "",
            "resulting_persistent_mood": (
                trace.resulting_persistent_mood if trace else ""
            ),
            "detected_signal": (
                f"{trace.detected_signal} {trace.signal_intensity:.2f}" if trace else ""
            ),
            "contextual_reaction": trace.contextual_reaction if trace else "",
            "signal_confidence": trace.signal_confidence if trace else 0.0,
            "activation_threshold": trace.activation_threshold if trace else 0.0,
            "reaction_mapping": trace.reaction_mapping if trace else "none",
            "state_changes": ", ".join(trace.state_changes) if trace else "",
            "candidate_mood_delta": trace.candidate_mood_delta if trace else "none",
            "active_emotion": (
                f"{trace.active_emotion} {trace.active_intensity:.2f}" if trace else ""
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
            "opinion_trace": decision_trace,
            "response_intention": asdict(response_intention),
            "style_compiler": asdict(compiled_style),
            "validation_results": validation.violations if validation else (),
            "validation_evidence": validation.evidence if validation else (),
            "final_disposition": trace.final_disposition if trace else "",
            "grounded_activity_source": turn.grounded_activity_source,
            "grounded_activity_age_seconds": turn.grounded_activity_age_seconds,
            "life_context_included": any(
                source.kind == "life_context" for source in prepared.prompt_plan.sources
            ),
            "life_absence_constraint_included": bool(
                _NO_ACTIVITY_GROUNDING.search(prepared.turn_context.life_context)
            ),
            "dynamic_state_tokens": estimate_tokens(
                prepared.turn_context.behavioral_summary
            ),
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
            f"interface={prepared.chat_input.source} prior={trace.prior_mood} "
            f"signal={trace.detected_signal}:{trace.signal_intensity:.2f} "
            f"reaction={trace.contextual_reaction} active={trace.active_emotion}:"
            f"{trace.active_intensity:.2f} decay={trace.decay_applied:.3f} "
            f"final={trace.final_disposition}",
            flush=True,
        )


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
    model_seconds = max(0.0, timing.model_finished_at - timing.model_started_at)
    decode_seconds = max(0.0, timing.model_finished_at - timing.first_token_at)
    prompt_eval_seconds = max(
        0.0,
        timing.first_token_at
        - timing.model_started_at
        - timing.chat_template_seconds
        - timing.prompt_tokenization_seconds,
    )
    output_tokens = estimate_tokens(reply)
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
        f"prompt_tokens_est={prepared.prompt_plan.estimated_tokens}",
        f"prompt_tokens_rendered={prepared.prompt_plan.rendered_prompt_tokens or 0}",
        f"prompt_tokens_backend={timing.prompt_tokens}",
        f"output_tokens_est={output_tokens}",
        f"output_chars={len(reply)}",
    ]
    if decode_seconds > 0.0:
        fields.append(f"tokens_per_second_est={output_tokens / decode_seconds:.2f}")
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
    estimated = int(metrics.get("estimated_prompt_tokens") or 0)
    if backend > 0:
        prompt_tokens: int | None = backend
    elif rendered is not None:
        prompt_tokens = int(rendered)
    else:
        prompt_tokens = estimated or None
    exact = bool(metrics.get("prompt_count_is_exact")) and (
        backend > 0 or rendered is not None
    )
    return prompt_tokens, "Exact" if exact else "Estimated" if prompt_tokens else "None"


def _debug_material_estimate_difference(estimate: int, actual: int | None) -> bool:
    if not estimate or actual is None or estimate == actual:
        return False
    difference = abs(estimate - actual)
    return difference > max(64, actual * 0.10)


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
        "  Offscreen Current Activity: "
        f"{_debug_activity_description(current_life)}",
        "  Offscreen Mood Effects: "
        f"{_debug_delta_text(life.get('mood_effects'))}",
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
        "  Preflight Token Estimate: "
        f"{int(metrics.get('estimated_prompt_tokens') or 0)}",
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

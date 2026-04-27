"""General request and follow-up analysis for Akane's routing layer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.codebase_search import CodebaseSearch

FILE_REF_PATTERN = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_CODING_REQUEST_PATTERN = re.compile(
    r"\b(code|coding|program|function|class|method|script|module|file|vscode|debug|fix|bug|refactor|edit|change|rewrite|implement|add|create|write|save|format|patch|python|javascript|typescript|js|ts|html|css|json|yaml|toml|regex|sql)\b",
    re.IGNORECASE,
)
_OPEN_VSCODE_REQUEST_PATTERN = re.compile(
    r"\b(open|launch|start)\b.*\b(vs\s*code|vscode|project|workspace)\b",
    re.IGNORECASE,
)
_CODEBASE_REQUEST_PATTERN = re.compile(
    r"\b(look at|check|inspect|review|read|open|source|codebase|implementation|in my code|in the code|in the project|file|files)\b",
    re.IGNORECASE,
)
_FOLLOWUP_REFERENCE_PATTERN = re.compile(
    r"\b(it|this|that|those|them|same|specific|that one|this one|do that|fix that|change that|improve that)\b",
    re.IGNORECASE,
)
_REFERENTIAL_FOLLOWUP_PATTERN = re.compile(
    r"\b(?:it|that|this|those|them|the first one|the second one|that one|this one|same one)\b",
    re.IGNORECASE,
)
_EXPLANATION_FOLLOWUP_PATTERN = re.compile(
    r"\b(?:talk more about|more about|explain|go deeper on|expand on|tell me more about|what about)\b",
    re.IGNORECASE,
)
_AFFIRMATION_PATTERN = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|ok|okay|please|do it|go ahead|show me|sounds good|let's do it)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_AFFIRMATION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|ok|okay|please|sounds good|go ahead|do it|do that|show me)\b",
    re.IGNORECASE,
)
_ASSISTANT_OFFER_PATTERN = re.compile(
    r"\b(?:would you like me|want me to|do you want me to|should i|can i|i can show|i can sketch|i can rewrite|i can walk you through|i can do that)\b",
    re.IGNORECASE,
)
_ACTION_OFFER_PATTERN = re.compile(
    r"\b(?:i(?:'|’)ll|i will|i can|want me to|do you want me to|should i|would you like me to)\b",
    re.IGNORECASE,
)
_ACTION_VERB_PATTERN = re.compile(
    r"\b(?:apply|change|edit|rewrite|update|fix|merge|implement|patch|make|refactor|replace|write)\b",
    re.IGNORECASE,
)
_EXECUTION_REQUEST_PATTERN = re.compile(
    r"\b(?:do it|do that|make the change|make those changes|apply it|apply that|change it|edit it|fix it|rewrite it|merge them|update the file|actually do it|actually change it)\b",
    re.IGNORECASE,
)
_DETAIL_REQUEST_PATTERN = re.compile(
    r"\b(detail|details|deeper|deep dive|full|full breakdown|walk through|walkthrough|show me|explain more|longer|full analysis)\b",
    re.IGNORECASE,
)
_SUGGESTION_REQUEST_PATTERN = re.compile(
    r"\b(suggest|suggestion|suggestions|improve|improvement|improvements|recommend|recommendation|recommendations|what should i change|how should i change|how can i improve)\b",
    re.IGNORECASE,
)
_QUESTION_START_PATTERN = re.compile(
    r"^\s*(?:how|why|what|when|where|who|which|can|could|would|should|do|does|did|is|are)\b",
    re.IGNORECASE,
)
_TOPIC_SHIFT_PATTERN = re.compile(
    r"\b(?:also|instead|switching|different|another|new topic|separately|unrelated|by the way|on a different note)\b",
    re.IGNORECASE,
)
_BRAINSTORM_TOKENS = {
    "idea",
    "ideas",
    "feature",
    "features",
    "brainstorm",
    "integration",
    "integrate",
    "chrome",
    "game",
    "games",
    "goal",
    "companion",
    "vtuber",
    "access",
    "ability",
    "abilities",
    "workflow",
}
_PLANNING_TOKENS = {
    "add",
    "build",
    "make",
    "create",
    "use",
    "do",
    "thinking",
    "think",
    "improve",
    "expand",
}
_SMALLTALK_TOKENS = {
    "hello",
    "hi",
    "hey",
    "yo",
    "sup",
    "morning",
    "afternoon",
    "evening",
    "night",
    "thanks",
    "thank",
    "cool",
    "nice",
    "awesome",
    "lol",
    "lmao",
    "bye",
    "goodbye",
}
_INTERROGATIVE_STARTERS = {
    "how",
    "why",
    "what",
    "when",
    "where",
    "who",
    "which",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
}
_FIRST_PERSON_SUBJECTS = {"i", "we", "my", "our"}
_ASSISTANT_TARGET_TOKENS = {"you", "akane"}


@dataclass(frozen=True)
class RequestSnapshot:
    last_user: str = ""
    last_assistant: str = ""
    recent_code_targets: tuple[str, ...] = ()
    active_file: str = ""
    open_tabs: tuple[str, ...] = ()
    recent_text: str = ""
    editor_connected: bool = False


@dataclass(frozen=True)
class RequestAnalysis:
    user_input: str
    query_tokens: tuple[str, ...]
    open_vscode: bool
    coding: bool
    codebase_direct: bool
    codebase_followup: bool
    explicit_file_reference: bool
    referential_followup: bool
    followup_reference: bool
    explanation_followup: bool
    affirmative_followup: bool
    short_followup: bool
    assistant_invited_continuation: bool
    topic_overlap: bool
    wants_execution: bool
    should_carry_last_reply: bool
    wants_detail: bool
    wants_single_suggestion: bool
    wants_brainstorm: bool
    should_force_vscode: bool

    @property
    def codebase_context(self) -> bool:
        return self.codebase_direct or self.codebase_followup

    @property
    def coding_like(self) -> bool:
        return self.coding or self.codebase_context


class RequestAnalyzer:
    """Analyze user requests in a general way using recent conversation context."""

    def __init__(self, search: CodebaseSearch):
        self.search = search

    def extract_query_tokens(self, text: str) -> list[str]:
        return self.search.extract_query_tokens(text)

    def has_recent_topic_overlap(self, user_input: str, recent_text: str) -> bool:
        return self.search.has_recent_topic_overlap(user_input, recent_text)

    def is_affirmative_followup(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        return bool(_AFFIRMATION_PATTERN.match(lowered) or _AFFIRMATION_PREFIX_PATTERN.match(lowered))

    def assistant_invited_continuation(self, text: str) -> bool:
        message = str(text or "").strip()
        if not message or "?" not in message:
            return False
        lowered = message.lower()
        if _ASSISTANT_OFFER_PATTERN.search(lowered):
            return True
        return bool(_ACTION_OFFER_PATTERN.search(lowered) and _ACTION_VERB_PATTERN.search(lowered))

    def has_explicit_file_reference(self, text: str) -> bool:
        if FILE_REF_PATTERN.search(str(text or "")):
            return True
        lowered = str(text or "").lower()
        return any(token in lowered for token in (".py", ".js", ".ts", ".json", ".md", ".tsx", ".jsx", ".toml", ".yaml", ".yml"))

    def _token_count(self, text: str) -> int:
        return len(_WORD_PATTERN.findall(str(text or "")))

    def _token_set(self, query_tokens: tuple[str, ...]) -> set[str]:
        return {token.lower() for token in query_tokens if token}

    def _shape_tokens(self, text: str) -> list[str]:
        return [token.lower() for token in _WORD_PATTERN.findall(str(text or "")) if token]

    def _starts_with_first_person_update(self, tokens: list[str]) -> bool:
        if not tokens:
            return False
        first = tokens[0]
        if first in _FIRST_PERSON_SUBJECTS:
            return True
        if first.startswith("i") and len(first) <= 3:
            return True
        if first.startswith("we") and len(first) <= 4:
            return True
        return False

    def _looks_like_direct_assistant_request(self, tokens: list[str]) -> bool:
        if not tokens:
            return False
        if "please" in tokens:
            return True
        if tokens[0] in _INTERROGATIVE_STARTERS:
            return True
        return any(token in _ASSISTANT_TARGET_TOKENS for token in tokens[:5])

    def _has_code_context(self, snapshot: RequestSnapshot, last_assistant: str) -> bool:
        if snapshot.recent_code_targets or snapshot.active_file:
            return True
        context_text = " ".join(part for part in (snapshot.last_user, last_assistant, snapshot.recent_text) if part)
        return bool(
            context_text
            and (
                _CODING_REQUEST_PATTERN.search(context_text)
                or _CODEBASE_REQUEST_PATTERN.search(context_text)
            )
        )

    def _looks_like_topic_shift(self, user_input: str, snapshot: RequestSnapshot) -> bool:
        lowered = str(user_input or "").lower().strip()
        if not lowered:
            return False
        if _TOPIC_SHIFT_PATTERN.search(lowered):
            return True
        recent_context = " ".join(
            part for part in (snapshot.last_user, snapshot.last_assistant, snapshot.recent_text) if part
        )
        if not recent_context:
            return False
        recent_coding = bool(
            _CODING_REQUEST_PATTERN.search(recent_context)
            or _CODEBASE_REQUEST_PATTERN.search(recent_context)
        )
        current_coding = bool(
            _CODING_REQUEST_PATTERN.search(lowered)
            or _CODEBASE_REQUEST_PATTERN.search(lowered)
        )
        if recent_coding != current_coding and self._token_count(user_input) >= 5:
            return True
        return False

    def _looks_like_brainstorm(
        self,
        user_input: str,
        query_tokens: tuple[str, ...],
        *,
        codebase_direct: bool,
        explicit_file_reference: bool,
        wants_execution: bool,
    ) -> bool:
        if codebase_direct or explicit_file_reference or wants_execution:
            return False
        lowered = str(user_input or "").lower()
        tokens = self._token_set(query_tokens)
        brainstorming_hits = len(tokens & _BRAINSTORM_TOKENS)
        planning_hits = len(tokens & _PLANNING_TOKENS)
        phrase_hits = sum(
            1
            for phrase in (
                "what else can i",
                "what other things can i",
                "is this a good idea",
                "what do you think",
                "thinking about",
                "maybe a way to",
                "could i add",
                "should i add",
            )
            if phrase in lowered
        )
        return bool(
            phrase_hits >= 1
            or (brainstorming_hits >= 1 and planning_hits >= 1)
            or ("vscode" in tokens and brainstorming_hits >= 1)
        )

    def _looks_like_smalltalk(
        self,
        user_input: str,
        query_tokens: tuple[str, ...],
        *,
        coding: bool,
        explicit_file_reference: bool,
        followup_reference: bool,
        explanation_followup: bool,
        affirmative_followup: bool,
    ) -> bool:
        if coding or explicit_file_reference or followup_reference or explanation_followup or affirmative_followup:
            return False
        tokens = self._token_set(query_tokens)
        if not tokens:
            return False
        if len(tokens) <= 3 and tokens <= _SMALLTALK_TOKENS:
            return True
        lowered = str(user_input or "").lower().strip()
        return lowered in {"hello", "hi", "hey", "thanks", "thank you", "good morning", "good afternoon", "good evening"}

    def _looks_like_status_update(
        self,
        user_input: str,
        query_tokens: tuple[str, ...],
        *,
        explicit_file_reference: bool,
        codebase_direct: bool,
        wants_execution: bool,
    ) -> bool:
        if explicit_file_reference or codebase_direct or wants_execution:
            return False
        lowered = str(user_input or "").lower().strip()
        if not lowered or "?" in lowered:
            return False
        shape_tokens = self._shape_tokens(user_input)
        if not self._starts_with_first_person_update(shape_tokens):
            return False
        if self._looks_like_direct_assistant_request(shape_tokens):
            return False
        if shape_tokens and shape_tokens[0] in _INTERROGATIVE_STARTERS:
            return False
        return True

    def analyze(self, user_input: str, snapshot: RequestSnapshot) -> RequestAnalysis:
        lowered = str(user_input or "").lower().strip()
        last_assistant = str(snapshot.last_assistant or "").strip()
        last_user = str(snapshot.last_user or "").strip()
        query_tokens = tuple(self.extract_query_tokens(user_input))

        open_vscode = bool(_OPEN_VSCODE_REQUEST_PATTERN.search(lowered))
        explicit_file_reference = self.has_explicit_file_reference(user_input)
        codebase_direct = bool(explicit_file_reference or _CODEBASE_REQUEST_PATTERN.search(lowered))
        coding = bool(_CODING_REQUEST_PATTERN.search(lowered) or codebase_direct)

        referential_followup = bool(_REFERENTIAL_FOLLOWUP_PATTERN.search(lowered))
        followup_reference = bool(_FOLLOWUP_REFERENCE_PATTERN.search(lowered) or referential_followup)
        explanation_followup = bool(_EXPLANATION_FOLLOWUP_PATTERN.search(lowered))
        affirmative_followup = self.is_affirmative_followup(user_input)
        short_followup = self._token_count(user_input) <= 10
        assistant_invited_continuation = self.assistant_invited_continuation(last_assistant)
        recent_context = " ".join(part for part in (last_user, last_assistant, snapshot.recent_text) if part)
        topic_overlap = self.has_recent_topic_overlap(user_input, recent_context)
        has_code_context = self._has_code_context(snapshot, last_assistant)
        topic_shift = self._looks_like_topic_shift(user_input, snapshot)
        ambiguous_followup = bool(
            followup_reference
            or affirmative_followup
            or explanation_followup
            or short_followup
            or len(query_tokens) <= 4
        )
        explicit_new_question = bool(
            self._token_count(user_input) >= 4
            and _QUESTION_START_PATTERN.search(lowered)
            and not followup_reference
            and not affirmative_followup
            and not explanation_followup
            and not topic_overlap
            and not assistant_invited_continuation
        )
        explicit_new_subject = bool(
            has_code_context
            and coding
            and self._token_count(user_input) >= 4
            and not explicit_file_reference
            and not followup_reference
            and not affirmative_followup
            and not explanation_followup
            and not topic_overlap
        )
        if explicit_new_subject or explicit_new_question:
            topic_shift = True
        if followup_reference or affirmative_followup or explanation_followup:
            topic_shift = False
        code_context_carry = bool(
            has_code_context
            and not topic_shift
            and not explicit_file_reference
            and (coding or ambiguous_followup or topic_overlap)
        )

        carry_score = 0
        if followup_reference:
            carry_score += 2
        if affirmative_followup:
            carry_score += 2
        if explanation_followup:
            carry_score += 2
        if short_followup:
            carry_score += 1
        if topic_overlap:
            carry_score += 1
        if len(query_tokens) <= 2:
            carry_score += 1
        should_carry_last_reply = bool(last_assistant or last_user) and (
            (carry_score >= 2 and not topic_shift)
            or (ambiguous_followup and not topic_shift)
            or code_context_carry
        )

        followup_score = 0
        if has_code_context:
            if followup_reference:
                followup_score += 2
            if affirmative_followup:
                followup_score += 2
            if explanation_followup:
                followup_score += 2
            if topic_overlap:
                followup_score += 1
            if short_followup:
                followup_score += 1
            if assistant_invited_continuation:
                followup_score += 1
            if len(query_tokens) <= 4 and (_ACTION_VERB_PATTERN.search(lowered) or coding):
                followup_score += 1
        codebase_followup = bool(
            has_code_context
            and not open_vscode
            and not topic_shift
            and (
                followup_score >= 2
                or code_context_carry
                or (assistant_invited_continuation and ambiguous_followup)
            )
        )

        wants_execution = False
        if has_code_context and _EXECUTION_REQUEST_PATTERN.search(lowered):
            wants_execution = True
        elif has_code_context and _ACTION_VERB_PATTERN.search(lowered):
            execution_score = 0
            if followup_reference:
                execution_score += 2
            if affirmative_followup:
                execution_score += 1
            if topic_overlap:
                execution_score += 1
            if short_followup:
                execution_score += 1
            if assistant_invited_continuation:
                execution_score += 1
            wants_execution = execution_score >= 2

        wants_detail = bool(_DETAIL_REQUEST_PATTERN.search(lowered))
        wants_single_suggestion = bool(_SUGGESTION_REQUEST_PATTERN.search(lowered)) and not wants_detail
        wants_brainstorm = self._looks_like_brainstorm(
            user_input,
            query_tokens,
            codebase_direct=codebase_direct,
            explicit_file_reference=explicit_file_reference,
            wants_execution=wants_execution,
        )
        looks_like_status_update = self._looks_like_status_update(
            user_input,
            query_tokens,
            explicit_file_reference=explicit_file_reference,
            codebase_direct=codebase_direct,
            wants_execution=wants_execution,
        )
        looks_like_smalltalk = self._looks_like_smalltalk(
            user_input,
            query_tokens,
            coding=coding,
            explicit_file_reference=explicit_file_reference,
            followup_reference=followup_reference,
            explanation_followup=explanation_followup,
            affirmative_followup=affirmative_followup,
        )
        if looks_like_smalltalk:
            code_context_carry = False
            codebase_followup = False
            if not topic_shift and (last_assistant or last_user):
                should_carry_last_reply = True
        should_force_vscode = bool(
            open_vscode
            or (
                snapshot.editor_connected
                and (
                    wants_execution
                    or (coding and not wants_brainstorm and not (codebase_direct or codebase_followup))
                )
            )
        )
        if looks_like_status_update:
            coding = False
            codebase_followup = False
            should_force_vscode = False

        return RequestAnalysis(
            user_input=str(user_input or ""),
            query_tokens=query_tokens,
            open_vscode=open_vscode,
            coding=coding,
            codebase_direct=codebase_direct,
            codebase_followup=codebase_followup,
            explicit_file_reference=explicit_file_reference,
            referential_followup=referential_followup,
            followup_reference=followup_reference,
            explanation_followup=explanation_followup,
            affirmative_followup=affirmative_followup,
            short_followup=short_followup,
            assistant_invited_continuation=assistant_invited_continuation,
            topic_overlap=topic_overlap,
            wants_execution=wants_execution,
            should_carry_last_reply=should_carry_last_reply,
            wants_detail=wants_detail,
            wants_single_suggestion=wants_single_suggestion,
            wants_brainstorm=wants_brainstorm,
            should_force_vscode=should_force_vscode,
        )

    def candidate_paths(
        self,
        user_input: str,
        snapshot: RequestSnapshot,
        *,
        limit: int = 3,
    ) -> list[str]:
        analysis = self.analyze(user_input, snapshot)
        recent_texts = [text for text in (snapshot.recent_text, snapshot.last_assistant) if text]
        if snapshot.last_user:
            recent_texts.append(snapshot.last_user)
        return self.search.candidate_paths(
            user_input,
            file_ref_pattern=FILE_REF_PATTERN,
            recent_targets=list(snapshot.recent_code_targets),
            active_file=snapshot.active_file,
            open_tabs=list(snapshot.open_tabs[:5]),
            recent_texts=recent_texts,
            coding_request=analysis.coding,
            explicit_file_reference=analysis.explicit_file_reference,
            followup_reference=analysis.followup_reference or analysis.codebase_followup,
            reuse_recent_targets=analysis.codebase_followup or analysis.followup_reference or analysis.topic_overlap,
            limit=limit,
        )

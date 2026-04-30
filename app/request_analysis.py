"""General request and follow-up analysis for Akane's routing layer."""

from __future__ import annotations

from dataclasses import dataclass

from app.codebase_search import CodebaseSearch

_FILE_SUFFIXES = {
    ".py", ".js", ".ts", ".json", ".md", ".tsx", ".jsx", ".toml", ".yaml", ".yml",
    ".css", ".html", ".sql", ".txt", ".sh",
}
_FOLLOWUP_REFERENCE_TOKENS = {"it", "this", "that", "those", "them", "same", "specific"}
_QUESTION_STARTERS = {
    "how", "why", "what", "when", "where", "who", "which", "can", "could", "would",
    "should", "do", "does", "did", "is", "are",
}
_FIRST_PERSON_SUBJECTS = {"i", "we", "my", "our"}
_ASSISTANT_TARGET_TOKENS = {"you", "akane"}


def _normalize(text: str) -> str:
    return " ".join(str(text or "").replace("’", "'").lower().split())


def _word_tokens(text: str) -> list[str]:
    chars: list[str] = []
    tokens: list[str] = []
    for ch in str(text or "").lower():
        if ch.isalnum() or ch in {"_", ".", "/", "-"}:
            chars.append(ch)
        elif chars:
            tokens.append("".join(chars))
            chars = []
    if chars:
        tokens.append("".join(chars))
    return tokens


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _contains_any_token(tokens: set[str], expected: set[str]) -> bool:
    return bool(tokens & expected)


def _starts_with_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _strip_trailing_punct(text: str) -> str:
    return text.rstrip(" \t\r\n.!?,;:")


def _starts_with_token_stem(token: str, stems: tuple[str, ...]) -> bool:
    return any(token.startswith(stem) for stem in stems)


def _is_pathish_token(token: str) -> bool:
    return "/" in token or any(token.endswith(suffix) for suffix in _FILE_SUFFIXES)


def _looks_like_code_token(token: str) -> bool:
    return (
        _is_pathish_token(token)
        or "_" in token
        or token in {"py", "js", "ts", "json", "yaml", "yml", "toml", "html", "css", "sql", "vscode"}
        or _starts_with_token_stem(
            token,
            ("code", "file", "func", "clas", "meth", "scri", "modu", "debug", "bug", "fix",
             "refa", "edit", "rewr", "impl", "patc", "writ", "format", "regex", "python", "java", "type"),
        )
    )


def _looks_like_codebase_request(lowered: str, tokens: list[str], explicit_file_reference: bool) -> bool:
    if explicit_file_reference or any(_looks_like_code_token(token) for token in tokens):
        return True
    if not tokens:
        return False
    first = tokens[0]
    return first in {"look", "check", "inspect", "review", "read", "open"} and any(
        token in {"code", "project", "source", "file", "files", "implementation"} for token in tokens[1:6]
    )


def _looks_like_open_editor(lowered: str, tokens: list[str]) -> bool:
    if not tokens or tokens[0] not in {"open", "launch", "start"}:
        return False
    return "vscode" in tokens or ("vs" in tokens and "code" in tokens) or "project" in tokens or "workspace" in tokens


def _looks_like_affirmation(lowered: str, tokens: list[str]) -> bool:
    if not lowered or len(tokens) > 5:
        return False
    if lowered in {"yes", "yeah", "yep", "sure", "ok", "okay", "please"}:
        return True
    if not tokens:
        return False
    first = tokens[0]
    return first in {"yes", "yeah", "yep", "sure", "ok", "okay", "please"} or (
        len(tokens) <= 3 and first in {"do", "go", "show"}
    )


def _looks_like_assistant_offer(lowered: str, tokens: list[str]) -> bool:
    if "?" not in lowered:
        return False
    if lowered.startswith(("can i ", "should i ", "want me ", "would you like me", "do you want me")):
        return True
    return lowered.startswith(("i can ", "i'll ", "i will "))


def _looks_like_explanation_followup(lowered: str, tokens: list[str]) -> bool:
    if lowered.startswith(("what about", "how about")):
        return True
    return any(_starts_with_token_stem(token, ("explain", "deeper", "expand", "detail", "more")) for token in tokens)


def _looks_like_execution_request(lowered: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    first = tokens[0]
    if first in {"do", "make", "apply", "change", "edit", "fix", "rewrite", "merge", "update"}:
        return True
    return len(tokens) <= 4 and any(
        _starts_with_token_stem(token, ("apply", "chang", "edit", "fix", "rewr", "merge", "updat", "make"))
        for token in tokens
    )


def _looks_like_action_verb(token: str) -> bool:
    return _starts_with_token_stem(token, ("apply", "chang", "edit", "rewr", "updat", "fix", "merge", "impl", "patch", "make", "refa", "repl", "writ"))


def _looks_like_detail_request(lowered: str, tokens: list[str]) -> bool:
    if "full " in lowered:
        return True
    return any(_starts_with_token_stem(token, ("detail", "deep", "walk", "long", "explain")) for token in tokens)


def _looks_like_suggestion_request(lowered: str, tokens: list[str]) -> bool:
    if lowered.startswith(("what should", "how should", "how can")):
        return True
    return any(_starts_with_token_stem(token, ("suggest", "improv", "recommend")) for token in tokens)


def _looks_like_topic_shift(lowered: str, tokens: list[str]) -> bool:
    if lowered.startswith(("by the way", "on a different", "new topic")):
        return True
    return bool(tokens and tokens[0] in {"also", "instead", "switching", "different", "another", "separately", "unrelated"})


def _looks_like_self_identity_question(tokens: list[str]) -> bool:
    if not tokens or tokens[0] not in {"who", "what"}:
        return False
    return (
        "you" in tokens
        or "your" in tokens
        or any(_starts_with_token_stem(token, ("name", "creat", "made")) for token in tokens[1:5])
    )


def _looks_like_contextual_continuation_prefix(lowered: str, tokens: list[str]) -> bool:
    if lowered.startswith(("what about", "how about")):
        return True
    return bool(tokens and tokens[0] in {"and", "also", "so", "then", "anyway", "okay", "ok", "right", "wait", "plus"})


def _looks_like_brainstorm_tokens(tokens: set[str], lowered: str) -> bool:
    planning = sum(1 for token in tokens if _starts_with_token_stem(token, ("add", "build", "make", "creat", "use", "think", "improv", "expand")))
    ideation = sum(1 for token in tokens if _starts_with_token_stem(token, ("idea", "feature", "brain", "integrat", "goal", "workflow", "companion", "vtuber", "access", "abilit", "game", "chrome")))
    return "?" in lowered or ideation >= 1 and planning >= 1


def _looks_like_smalltalk_tokens(tokens: set[str], lowered: str) -> bool:
    if lowered in {"hello", "hi", "hey", "thanks", "thank you", "good morning", "good afternoon", "good evening"}:
        return True
    return bool(tokens) and len(tokens) <= 3 and all(
        token in {"hello", "hi", "hey", "yo", "sup", "morning", "afternoon", "evening", "night",
                  "thanks", "thank", "cool", "nice", "awesome", "lol", "lmao", "bye", "goodbye"}
        for token in tokens
    )


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

    def extract_file_refs(self, text: str) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for token in _word_tokens(text):
            cleaned = token.strip("`'\".,:;()[]{}")
            if not cleaned:
                continue
            if "/" in cleaned or any(cleaned.endswith(suffix) for suffix in _FILE_SUFFIXES):
                if cleaned not in seen:
                    seen.add(cleaned)
                    refs.append(cleaned)
        return refs

    def has_recent_topic_overlap(self, user_input: str, recent_text: str) -> bool:
        current_tokens = set(self.extract_query_tokens(user_input))
        recent_tokens = set(self.extract_query_tokens(recent_text))
        if not current_tokens or not recent_tokens:
            return False
        return bool(current_tokens & recent_tokens)

    def is_affirmative_followup(self, text: str) -> bool:
        lowered = _strip_trailing_punct(_normalize(text))
        if not lowered:
            return False
        return _looks_like_affirmation(lowered, _word_tokens(lowered))

    def assistant_invited_continuation(self, text: str) -> bool:
        message = str(text or "").strip()
        if not message or "?" not in message:
            return False
        lowered = _normalize(message)
        return _looks_like_assistant_offer(lowered, _word_tokens(lowered))

    def has_explicit_file_reference(self, text: str) -> bool:
        return bool(self.extract_file_refs(text))

    @staticmethod
    def _starts_with_first_person_update(tokens: list[str]) -> bool:
        if not tokens:
            return False
        first = tokens[0]
        if first in _FIRST_PERSON_SUBJECTS:
            return True
        return (first.startswith("i") and len(first) <= 3) or (first.startswith("we") and len(first) <= 4)

    @staticmethod
    def _looks_like_direct_assistant_request(tokens: list[str]) -> bool:
        if not tokens:
            return False
        if "please" in tokens:
            return True
        if tokens[0] in _QUESTION_STARTERS:
            return True
        return any(token in _ASSISTANT_TARGET_TOKENS for token in tokens[:5])

    def _has_code_context(self, snapshot: RequestSnapshot, last_assistant: str) -> bool:
        context_text = _normalize(" ".join(part for part in (snapshot.last_user, last_assistant) if part))
        if not context_text:
            return False
        context_tokens = _word_tokens(context_text)
        return _looks_like_codebase_request(context_text, context_tokens, False)

    def _looks_like_topic_shift(self, user_input: str, snapshot: RequestSnapshot) -> bool:
        lowered = _normalize(user_input)
        if not lowered:
            return False
        current_tokens = set(_word_tokens(lowered))
        if _looks_like_topic_shift(lowered, list(current_tokens)):
            return True
        recent_context = _normalize(" ".join(part for part in (snapshot.last_user, snapshot.last_assistant, snapshot.recent_text) if part))
        if not recent_context:
            return False
        recent_tokens = set(_word_tokens(recent_context))
        recent_coding = _looks_like_codebase_request(recent_context, list(recent_tokens), False)
        current_coding = _looks_like_codebase_request(lowered, list(current_tokens), False)
        return recent_coding != current_coding and len(current_tokens) >= 5

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
        lowered = _normalize(user_input)
        tokens = set(query_tokens)
        return _looks_like_brainstorm_tokens(tokens, lowered)

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
        tokens = set(query_tokens)
        if not tokens:
            return False
        lowered = _normalize(user_input)
        return _looks_like_smalltalk_tokens(tokens, lowered)

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
        lowered = _normalize(user_input)
        if not lowered or "?" in lowered:
            return False
        shape_tokens = _word_tokens(user_input)
        if not self._starts_with_first_person_update(shape_tokens):
            return False
        if self._looks_like_direct_assistant_request(shape_tokens):
            return False
        return not (shape_tokens and shape_tokens[0] in _QUESTION_STARTERS)

    def _looks_like_contextual_continuation(
        self,
        user_input: str,
        *,
        topic_shift: bool,
        coding: bool,
        explicit_file_reference: bool,
    ) -> bool:
        if topic_shift or coding or explicit_file_reference:
            return False
        lowered = _normalize(user_input)
        if not lowered:
            return False
        tokens = _word_tokens(user_input)
        if _looks_like_contextual_continuation_prefix(lowered, tokens):
            return True
        return bool(tokens and len(tokens) <= 14 and any(token in {"too", "though", "still", "instead", "that", "this", "it"} for token in tokens[:5]))

    def _looks_like_short_question_continuation(
        self,
        user_input: str,
        *,
        topic_shift: bool,
        coding: bool,
        explicit_file_reference: bool,
        topic_overlap: bool,
    ) -> bool:
        if topic_shift or coding or explicit_file_reference:
            return False
        lowered = _normalize(user_input)
        if not lowered:
            return False
        tokens = _word_tokens(lowered)
        if not tokens or len(tokens) > 8 or tokens[0] not in _QUESTION_STARTERS:
            return False
        if topic_overlap:
            return True
        if len(tokens) <= 3:
            return True
        return any(token in _FOLLOWUP_REFERENCE_TOKENS for token in tokens[1:6])

    def analyze(self, user_input: str, snapshot: RequestSnapshot) -> RequestAnalysis:
        lowered = _normalize(user_input)
        last_assistant = str(snapshot.last_assistant or "").strip()
        last_user = str(snapshot.last_user or "").strip()
        query_tokens = tuple(self.extract_query_tokens(user_input))
        tokens = _word_tokens(lowered)
        token_set = set(tokens)
        token_count = len(tokens)
        wants_execution = False
        self_identity_request = _looks_like_self_identity_question(tokens)

        open_vscode = _looks_like_open_editor(lowered, tokens)
        explicit_file_reference = self.has_explicit_file_reference(user_input)
        codebase_direct = _looks_like_codebase_request(lowered, tokens, explicit_file_reference)
        coding = bool(codebase_direct or any(_looks_like_code_token(token) for token in tokens))

        referential_followup = bool(
            _contains_any_token(token_set, {"it", "that", "this", "those", "them", "same"})
            or lowered.startswith(("the first", "the second"))
        )
        followup_reference = bool(
            referential_followup
            or _contains_any_token(token_set, _FOLLOWUP_REFERENCE_TOKENS)
            or (len(tokens) >= 2 and tokens[0] in {"do", "fix", "change", "improve"} and tokens[1] in _FOLLOWUP_REFERENCE_TOKENS)
        )
        explanation_followup = _looks_like_explanation_followup(lowered, tokens)
        affirmative_followup = self.is_affirmative_followup(user_input)
        short_followup = token_count <= 10
        assistant_invited_continuation = self.assistant_invited_continuation(last_assistant)
        recent_context = " ".join(part for part in (last_user, last_assistant, snapshot.recent_text) if part)
        topic_overlap = self.has_recent_topic_overlap(user_input, recent_context)
        has_code_context = self._has_code_context(snapshot, last_assistant)
        topic_shift = self._looks_like_topic_shift(user_input, snapshot)
        ambiguous_followup = bool(followup_reference or affirmative_followup or explanation_followup or short_followup or len(query_tokens) <= 4)
        shape_tokens = tokens
        explicit_new_question = bool(
            token_count >= 4
            and bool(shape_tokens)
            and shape_tokens[0] in _QUESTION_STARTERS
            and not followup_reference
            and not affirmative_followup
            and not explanation_followup
            and not topic_overlap
            and not assistant_invited_continuation
        )
        explicit_new_subject = bool(
            has_code_context
            and coding
            and token_count >= 4
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
        contextual_continuation = self._looks_like_contextual_continuation(
            user_input,
            topic_shift=topic_shift,
            coding=coding,
            explicit_file_reference=explicit_file_reference,
        )
        short_question_continuation = self._looks_like_short_question_continuation(
            user_input,
            topic_shift=topic_shift,
            coding=coding,
            explicit_file_reference=explicit_file_reference,
            topic_overlap=topic_overlap,
        )
        if short_question_continuation:
            topic_shift = False
        if has_code_context and _looks_like_execution_request(lowered, tokens):
            wants_execution = True
        elif has_code_context and any(_looks_like_action_verb(token) for token in token_set):
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

        wants_detail = _looks_like_detail_request(lowered, tokens)
        wants_single_suggestion = _looks_like_suggestion_request(lowered, tokens) and not wants_detail

        code_followup_signal = bool(
            coding
            or explicit_file_reference
            or followup_reference
            or explanation_followup
            or affirmative_followup
            or wants_execution
            or wants_detail
            or wants_single_suggestion
            or topic_overlap
        )
        code_context_carry = bool(
            has_code_context and not topic_shift and not explicit_file_reference and code_followup_signal
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
        if contextual_continuation:
            carry_score += 2
        if short_question_continuation:
            carry_score += 2
        should_carry_last_reply = bool(last_assistant or last_user) and (
            (carry_score >= 2 and not topic_shift) or (ambiguous_followup and not topic_shift) or code_context_carry
        )

        conversational_carry = bool(
            (last_assistant or last_user)
            and not topic_shift
            and not coding
            and not codebase_direct
            and not explicit_file_reference
            and not wants_execution
            and (short_followup or topic_overlap or assistant_invited_continuation or contextual_continuation or short_question_continuation or len(query_tokens) <= 16)
        )
        if conversational_carry:
            should_carry_last_reply = True

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
            if contextual_continuation:
                followup_score += 1
            if len(query_tokens) <= 4 and (any(_looks_like_action_verb(token) for token in token_set) or coding):
                followup_score += 1
        codebase_followup = bool(
            has_code_context and not open_vscode and not topic_shift and code_followup_signal and (
                followup_score >= 2 or code_context_carry or (assistant_invited_continuation and ambiguous_followup)
            )
        )
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
            or (snapshot.editor_connected and (wants_execution or (coding and not wants_brainstorm and not (codebase_direct or codebase_followup))))
        )
        if looks_like_status_update:
            coding = False
            codebase_followup = False
            should_force_vscode = False
        if self_identity_request:
            coding = False
            codebase_direct = False
            codebase_followup = False
            wants_execution = False
            should_carry_last_reply = False
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
            file_refs=self.extract_file_refs(user_input),
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
